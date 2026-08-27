import io
import re
import math
import time
import numpy as np
import s3fs
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import ray
import ray.train
import ray.train.torch as ray_torch
from ray.train import ScalingConfig, RunConfig

from vqvae_bert_pretraining import BERT, ddp_is_initialized, is_main_process
from vqvae_model import PaddedLoader
from vqvae_bert_sentence_dataset import MAX_SENTENCE_LEN
from vqvae_ecg_waveforms_dataset import bucket_out, NLEADS
from vqvae_bert_finetuning_sentences_dataset import (
    DATASETS,
    get_or_build_finetuning_sentences,
    create_finetuning_worker_dataloaders,
)
from vqvae_bert_pretraining import CONFIG as PRETRAIN_CONFIG

import os
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"     # Deterministic


#-----------------------parameters definitions-----------------------#
CONFIG = {
    "dataset": "ptbxl_superclasses",
    "use_frac": 1.0,
    "in_channels": 12,
    "batch_size": 32,      # 32
    "num_epochs": 100,
    "lr": 1e-4,
    "lr_head_factor": 0.1,
    "min_lr_factor": 0.1,
    "warmup_epochs": 5,
    "dropout": 0.1,      # 0.2
    "weight_decay": 1e-4,    # 1e-4, 3e-4, 1e-3
    "wtdecay_head_factor": 3.0,
    "patience": 15,
    'min_delta': 1e-4,
    "grad_clip_norm": 1.0,
    "seed": 42,
    # Ray
    "num_ray_workers": 4,
    "use_gpu": True,
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules":  ["W_Q", "W_V", "W_K", "fc", "fc1", "fc2"],
    # ["W_Q", "W_V", "W_K", "fc", "fc1", "fc2"], ["W_Q", "W_V", "W_K"], ["W_Q", "W_V", "W_K", "fc"], ["W_Q", "W_V"]
    "modules_to_save": ["classifier", "fc_classifier"],   # ["classifier", "fc_classifier"], ["classifier"]
}


# ---------------------------------------------------------------------------

prefix_root = "aruna-files/vqvae"
prefix_finetuning = "ecgvectbert/vqvae/bert_finetuning"
prefix_root_bert_model = "aruna-files/vqvae_final_12lead_vqenc/vqvae/bert_pretraining"
prefix_sentences = f"{prefix_finetuning}/sentences"

NUM_CLASSES = {"ptbxl_superclasses": 5,
               "ptbxl_subclasses": 23,
               "ptbxl_form": 19,
               "ptbxl_rhythm": 12,
               "cpsc2018": 9,
               "cs": 11,
               "csn": 38,
               "dandelion": 2,  # binary: EF <= 40% (one-hot [B, 2])
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


#----------model definition (classifier head) -----------#

class BERTForClassification(nn.Module):
    """Wrap pre-trained BERT with a softmax classifier head."""

    def __init__(self, base_model: BERT, d_model: int, num_classes: int, dropout: float, dataset_name: str = "ptbxl_superclasses"):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(d_model, num_classes)
        self.fc_classifier = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.dataset_name = dataset_name

    def forward(self, input_ids, use_cnn_features, cnn_features, labels=None):
        pooled_output = self.bert(input_ids, use_cnn_features, cnn_features)  # [B, d_model]
        pooled_output = self.activation(self.fc_classifier(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.dataset_name == "dandelion":
                loss = torch.nn.functional.cross_entropy(logits, labels.float())
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return logits, loss


def evaluate_metrics(dataset_name: str, model, loader, use_cnn_features: bool, num_classes: int, device: torch.device):
    model.eval()
    eval_model = unwrap_model(model)
    total_loss = 0.0
    num_batches = 0
    all_labels, all_probs, all_preds, all_ecg_idxs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            cnn_feats = (batch["cnn_embeddings"].to(device, non_blocking=True)
                          if use_cnn_features and "cnn_embeddings" in batch else None)
            labels = batch["labels"].to(device, non_blocking=True)  # multilabel (B, num_classes) float
            ecg_idxs = batch["ecg_idxs"]

            logits, loss = eval_model(input_ids, use_cnn_features, cnn_feats, labels=labels)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)          # multilabel
            preds = (probs > 0.5).long()           # multilabel

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_ecg_idxs.append(np.array(ecg_idxs, dtype=np.str_))

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, num_classes))
    all_probs  = np.concatenate(all_probs,  axis=0) if all_probs  else np.zeros((0, num_classes))
    all_preds  = np.concatenate(all_preds,  axis=0) if all_preds  else np.zeros((0, num_classes), dtype=np.int64)
    all_ecg_idxs = np.concatenate(all_ecg_idxs, axis=0) if all_ecg_idxs else np.zeros((0,), dtype=np.str_)

    if ddp_is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (all_labels, all_probs, all_preds, all_ecg_idxs, total_loss, num_batches))
        all_labels  = np.concatenate([g[0] for g in gathered], axis=0)
        all_probs   = np.concatenate([g[1] for g in gathered], axis=0)
        all_preds   = np.concatenate([g[2] for g in gathered], axis=0)
        all_ecg_idxs = np.concatenate([g[3] for g in gathered], axis=0)
        total_loss  = sum(g[4] for g in gathered)
        num_batches = sum(g[5] for g in gathered)

    # ---------overall sentence-level metrics------
    # sklearn's accuracy_score and f1_score natively accept 2D multilabel-indicator
    # arrays. accuracy_score on a 2D array computes
    # *exact-match* (subset) accuracy across all classes per row.
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / max(1, num_batches)
    macro_auroc = _macro_auroc(all_labels, all_probs, num_classes, dataset_name)

    # ---- record (ecg_idx)-level aggregation ----
    unique_ecgs = np.unique(all_ecg_idxs)
    record_true  = np.empty((len(unique_ecgs), num_classes), dtype=np.int64)
    record_pred  = np.empty((len(unique_ecgs), num_classes), dtype=np.int64)
    record_probs = np.empty((len(unique_ecgs), num_classes), dtype=np.float64)

    for i, ecg_idx in enumerate(unique_ecgs):
        mask = all_ecg_idxs == ecg_idx
        record_true[i] = all_labels[mask][0]  # one-hot/multi-hot vector, identical across a record's sentences
        # majority vote per class across this record's sentence-level predictions
        vote_counts = all_preds[mask].sum(axis=0)
        record_pred[i] = (vote_counts > (mask.sum() / 2)).astype(np.int64)
        record_probs[i] = all_probs[mask].mean(axis=0)

    record_acc = accuracy_score(record_true, record_pred)
    record_f1 = f1_score(record_true, record_pred, average="macro")
    record_auroc = _macro_auroc(record_true, record_probs, num_classes, f"{dataset_name} (record-level)")

    return {
        "loss": avg_loss,
        "acc": acc, "f1": macro_f1, "auroc": macro_auroc,
        "record_acc": record_acc, "record_f1": record_f1, "record_auroc": record_auroc,
    }


# Macro-Averaged AUROC (robust if some classes are absent, esp. for 1% and 10% splits)
def _macro_auroc(labels: np.ndarray, probs: np.ndarray, num_classes: int, tag: str) -> float:
    valid_aucs, valid_classes = [], []
    for c in range(num_classes):
        y_true_c = labels[:, c].astype(int)   
        pos, neg = y_true_c.sum(), len(y_true_c) - y_true_c.sum()
        if pos == 0 or neg == 0:
            log.info(f"Warning: label {c} in {tag} has all pos or all neg, cannot compute AUROC for this class")
            continue
        try:
            valid_aucs.append(roc_auc_score(y_true_c, probs[:, c]))
            valid_classes.append(c)
        except ValueError:
            pass
    macro_auroc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    log.info(f"Macro-averaged AUROC = {macro_auroc:.4f} averaged over {len(valid_classes)}/{num_classes} classes for {tag}")
    if not valid_aucs:
        log.info(f"Warning: no valid classes for AUC in this split (all missing or single-class) for {tag}")
    return macro_auroc


#----------- pretrained weight loading -----------#

def load_pretrained_bert_weights(bert: BERT, use_cnn_features: bool, s3_fs: s3fs.S3FileSystem, key: str) -> None:
    """Load pretraining checkpoint into a freshly built BERT module.

    Pretraining saves ``model.module.state_dict()`` where ``model.module`` is a
    ``torch.compile``-wrapped BERT, so keys may carry ``_orig_mod.`` / ``module.``
    prefixes that must be stripped before loading into a plain BERT instance.
    """
    log.info(f"Loading pretrained BERT weights from s3://{bucket_out}/{key}")
    with s3_fs.open(f"s3://{bucket_out}/{key}", "rb") as f:
        buf = io.BytesIO(f.read())
    state_dict = torch.load(buf, map_location="cpu")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        while nk.startswith("module.") or nk.startswith("_orig_mod."):
            nk = nk.split(".", 1)[1]
        cleaned_state_dict[nk] = v

    missing, unexpected = bert.load_state_dict(cleaned_state_dict, strict=False)
    cnn_mismatch = [k for k in (missing + unexpected) if "cnn_linear" in k]
    if cnn_mismatch:
        expected = "expected (use_cnn_features=True)" if use_cnn_features else "not expected (use_cnn_features=False)"
        raise ValueError(
            f"cnn_linear keys mismatched: {cnn_mismatch}. This finetuning run has "
            f"use_cnn_features={use_cnn_features} ({expected}), but the checkpoint disagrees."
        )
    total_params = sum(p.numel() for p in bert.parameters())
    if is_main_process():
        log.info(f"Pretrained BERT weights loaded. Missing keys: {missing}. Unexpected keys: {unexpected}")
        log.info(f"BERT total parameters: {total_params:,} ")


def build_finetune_model(vocab_size: int, use_cnn_features: bool, cnn_embed_dim: int, num_classes: int, lora_r: int, lora_alpha: int, lora_dropout: float, 
                         dropout: float, prefix_bert_model: str) -> nn.Module:
    """Build a BERTForClassification model, load pretrained BERT weights, and wrap it with LoRA."""
    d_model  = PRETRAIN_CONFIG["d_model"]
    n_layers = PRETRAIN_CONFIG["n_layers"]
    n_heads  = PRETRAIN_CONFIG["n_heads"]
    d_k = d_v = d_model // n_heads
    d_ff = d_model * 4
    cnn_scale_init = PRETRAIN_CONFIG["cnn_scale_init"]

    # BERT model using default BERT dropout
    bert = BERT(
        vocab_size, MAX_SENTENCE_LEN, d_model, n_layers, n_heads,
        d_k, d_v, d_ff, use_cnn_features, cnn_embed_dim, cnn_scale_init
    )

    _s3 = s3fs.S3FileSystem()
    load_pretrained_bert_weights(bert, use_cnn_features, _s3, prefix_bert_model)

    model = BERTForClassification(bert, d_model, num_classes, dropout)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=CONFIG["target_modules"],  # W_Q/W_V/W_K (BERT attention) — LoRA fine-tuned
        modules_to_save=CONFIG["modules_to_save"],  # classifier/fc_classifier (head) — trained fully, not via LoRA
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # ---- Assert LoRA actually attached to real modules ----
    lora_layer_names = [
        name for name, module in peft_model.named_modules()
        if isinstance(module, LoraLayer)
    ]
    assert len(lora_layer_names) > 0, (
        f"LoRA target_modules={CONFIG['target_modules']} matched zero modules in the model. "
        "get_peft_model() silently no-ops when target_modules doesn't match any submodule "
        "name — check that these strings match the actual attribute names in your attention "
        "implementation (e.g. MultiHeadAttention.W_Q / W_K / W_V)."
    )
    if is_main_process():
        log.info(
            f"LoRA attached to {len(lora_layer_names)} module(s): "
            f"{lora_layer_names[:6]}{' ...' if len(lora_layer_names) > 6 else ''}"
        )

    # ---- Assert modules_to_save params are actually trainable ----
    # PEFT renames these modules under a ".original_module" / ".modules_to_save.<adapter_name>"
    # wrapper (ModulesToSaveWrapper), and marks the active adapter copy's params
    # requires_grad=True while freezing everything else. If a name in
    # CONFIG["modules_to_save"] doesn't match any submodule, PEFT silently
    # skips it — the module stays frozen along with the rest of the base model.
    for save_name in CONFIG["modules_to_save"]:
        pattern = rf"(^|\.){re.escape(save_name)}\.modules_to_save\."
        matched_params = [
            (n, p) for n, p in peft_model.named_parameters()
            if re.search(pattern, n)
        ]
        assert len(matched_params) > 0, (
            f"modules_to_save entry '{save_name}' matched zero parameters after "
            "get_peft_model() — check it matches an actual submodule attribute name "
            "(e.g. BERTForClassification.classifier / fc_classifier)."
        )
        not_trainable = [n for n, p in matched_params if not p.requires_grad]
        assert not not_trainable, (
            f"modules_to_save entry '{save_name}' matched parameters but they are "
            f"frozen (requires_grad=False): {not_trainable}. This module will not "
            "be fine-tuned."
        )
    if is_main_process():
        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in peft_model.parameters())
        log.info(
            f"modules_to_save {CONFIG['modules_to_save']} verified trainable. "
            f"Trainable params: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / n_total:.2f}%)"
        )

    return peft_model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


#----------- checkpoint / metrics persistence -----------#

def save_lora_checkpoint(model: nn.Module, s3_fs: s3fs.S3FileSystem, use_frac: float, dataset_name: str, seed: int) -> None:
    
    prefix_out = f"{prefix_finetuning}/{dataset_name}"
    peft_model = unwrap_model(model)
    state_dict = get_peft_model_state_dict(peft_model)
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    key = f"{prefix_out}/lora_bert_finetuned_{dataset_name}_{use_frac}_seed{seed}.pt"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"LoRA checkpoint saved to s3://{bucket_out}/{key}")


def load_lora_checkpoint(model: nn.Module, s3_fs: s3fs.S3FileSystem, use_frac: float, dataset_name: str, seed: int) -> None:
    prefix_out = f"{prefix_finetuning}/{dataset_name}"
    key = f"{prefix_out}/lora_bert_finetuned_{dataset_name}_{use_frac}_seed{seed}.pt"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "rb") as f:
        buf = io.BytesIO(f.read())
    state_dict = torch.load(buf, map_location="cpu")
    result = set_peft_model_state_dict(unwrap_model(model), state_dict)
    if is_main_process():
        if result.missing_keys or result.unexpected_keys:
            log.warning(
                f"[{dataset_name}] LoRA checkpoint load for use_frac={use_frac} had "
                f"mismatched keys — missing: {result.missing_keys}, "
                f"unexpected: {result.unexpected_keys}"
            )
        else:
            log.info(
                f"[{dataset_name}] LoRA checkpoint for use_frac={use_frac} loaded "
                f"cleanly — no missing/unexpected keys."
            )
    return result


def save_epoch_metrics_npz(
    dataset_name: str,
    use_frac: float,
    seed: int,
    train_losses, train_accs, train_f1s, train_aurocs,
    train_record_accs, train_record_f1s, train_record_aurocs,
    val_losses, val_accs, val_f1s, val_aurocs,
    val_record_accs, val_record_f1s, val_record_aurocs,
    s3_fs: s3fs.S3FileSystem,
) -> None:
    buf = io.BytesIO()
    np.savez(
        buf,
        epoch=np.arange(1, len(train_losses) + 1),
        train_loss=np.array(train_losses), train_accuracy=np.array(train_accs),
        train_f1=np.array(train_f1s), train_auroc=np.array(train_aurocs),
        train_record_accuracy=np.array(train_record_accs),
        train_record_f1=np.array(train_record_f1s), train_record_auroc=np.array(train_record_aurocs),
        val_loss=np.array(val_losses), val_accuracy=np.array(val_accs),
        val_f1=np.array(val_f1s), val_auroc=np.array(val_aurocs),
        val_record_accuracy=np.array(val_record_accs),
        val_record_f1=np.array(val_record_f1s), val_record_auroc=np.array(val_record_aurocs),
    )
    buf.seek(0)
    
    prefix_out = f"{prefix_finetuning}/{dataset_name}"
    key = f"{prefix_out}/finetune_train_val_metrics_{dataset_name}_{use_frac}_seed{seed}.npz"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"Train/val metrics saved to s3://{bucket_out}/{key}")


def _load_existing_npz(s3_fs: s3fs.S3FileSystem, key: str) -> dict:
    """Load an existing npz from S3 into a plain dict, or {} if it doesn't exist yet."""
    if not s3_fs.exists(f"s3://{bucket_out}/{key}"):
        return {}
    with s3_fs.open(f"s3://{bucket_out}/{key}", "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
        return {k: data[k] for k in data.files}


def save_test_metrics_npz(
    dataset_name: str, use_frac: float, seed: int,
    test_loss, test_acc, test_f1, test_auroc,
    test_record_acc, test_record_f1, test_record_auroc, s3_fs: s3fs.S3FileSystem,
) -> None:
    prefix_out = f"{prefix_finetuning}/{dataset_name}"
    key = f"{prefix_out}/finetune_test_metrics_{dataset_name}_{use_frac}.npz"

    existing = _load_existing_npz(s3_fs, key)
    seeds = existing.get("seeds", np.array([], dtype=np.int64)).tolist()

    fields = {
        "test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1, "test_auroc": test_auroc,
        "test_record_accuracy": test_record_acc, "test_record_f1": test_record_f1, "test_record_auroc": test_record_auroc,
    }
    values = {name: existing.get(name, np.array([])).tolist() for name in fields}

    if seed in seeds:
        idx = seeds.index(seed)          # rerun of an existing seed — overwrite in place
        for name, val in fields.items():
            values[name][idx] = val
    else:
        seeds.append(seed)
        for name, val in fields.items():
            values[name].append(val)

    save_kwargs = {"seeds": np.array(seeds, dtype=np.int64)}
    save_kwargs.update({name: np.array(vals, dtype=np.float64) for name, vals in values.items()})

    buf = io.BytesIO()
    np.savez(buf, **save_kwargs)
    buf.seek(0)
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"Test metrics for seed={seed} appended to s3://{bucket_out}/{key} (seeds so far: {seeds})")


#----------- Ray + DDP training loop -----------#

def train_loop_per_worker(loop_config: dict) -> None:
    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    local_rank = ctx.get_local_rank()
    device = ray_torch.get_device()

    seed = loop_config["seed"] + world_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)   # warn_only avoids hard-crashing on any op without a deterministic kernel

    dataset_name  = loop_config["dataset_name"]
    use_frac      = loop_config["use_frac"]
    batch_size    = loop_config["batch_size"]
    in_channels   = loop_config["in_channels"]
    num_classes   = loop_config["num_classes"]
    num_epochs    = loop_config["num_epochs"]
    patience      = loop_config["patience"]
    min_delta     = loop_config["min_delta"]
    min_lr_factor = loop_config["min_lr_factor"]
    warmup_epochs = loop_config["warmup_epochs"]
    lora_r = loop_config["lora_r"]
    lora_alpha = loop_config["lora_alpha"]
    lora_dropout = loop_config["lora_dropout"]
    dropout = loop_config["dropout"]

    prefix_bert_model = f"{prefix_root_bert_model}/bert_model_nleads_{in_channels}.pt"

    use_cnn_features = PRETRAIN_CONFIG["use_cnn_features"]

    """
    # Not using this because this method uses pkl model file that only works in 
    # finetuning sentence construction file
    if world_rank == 0:
        get_or_build_finetuning_sentences(
            dataset_name=dataset_name,
            local_rank=local_rank,
            in_channels=in_channels,
            use_frac=use_frac,
            batch_size=batch_size,
        )
    dist.barrier()
    """

    # Data loaders
    loaders = create_finetuning_worker_dataloaders(dataset_name, use_frac, batch_size, seed=loop_config["seed"])
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    # Check number of classes
    actual_num_classes = train_loader.dataset._labels.shape[1]
    assert actual_num_classes == num_classes, (
        f"Config num_classes={num_classes} doesn't match loaded label width "
        f"{actual_num_classes} for dataset={dataset_name} — check NUM_CLASSES dict."
    )

    # After creating loaders, before the epoch loop:
    # synchronize the number of training steps across all ranks and pad loaders
    batches = torch.tensor([len(train_loader)], dtype=torch.long, device=device)
    dist.all_reduce(batches, op=dist.ReduceOp.MAX)
    max_batches = batches.item()
    # Wrap loader — ranks with fewer batches cycle; ranks at MAX just iterate normally
    padded_train_loader = PaddedLoader(train_loader, max_batches)

    vocab_size    = train_loader.dataset.vocab_size
    cnn_embed_dim = train_loader.dataset.cnn_embedding_dim if use_cnn_features else 0

    if is_main_process():
        log.info(
            f"[{dataset_name}] Train: {len(train_loader.dataset):,} | "
            f"Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,} | "
            f"vocab_size={vocab_size} | num_classes={num_classes} "
            f"use_frac={use_frac} "
        )

    model = build_finetune_model(vocab_size, use_cnn_features, cnn_embed_dim, num_classes, lora_r, lora_alpha, lora_dropout, dropout, prefix_bert_model)
    model = ray_torch.prepare_model(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    def _is_head_param(name: str) -> bool:
        return "classifier" in name or "fc_classifier" in name
    
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and _is_head_param(n)]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and not _is_head_param(n)]
    
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": loop_config["lr"], "weight_decay": loop_config["weight_decay"]},
            {"params": head_params, "lr": loop_config["lr"] * loop_config["lr_head_factor"], 
             "weight_decay": loop_config["weight_decay"] * loop_config["wtdecay_head_factor"]},
        ]
    )

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs         # ramps 1/W, 2/W, ... W/W
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay
    
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    steps_per_epoch = len(padded_train_loader)  # varies a lot by use_frac
    warmup_steps = min(500, max(steps_per_epoch, 10))

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=loop_config["lr"] * loop_config["min_lr_factor"]
    )
    global_step = 0
    warmup_done = False

    # Metrics, sequence-level
    train_losses, train_accs, train_f1s, train_aurocs = [], [], [], []
    val_losses, val_accs, val_f1s, val_aurocs = [], [], [], []
    # Metrics, record level
    train_record_acc, train_record_f1, train_record_auroc = [], [], []
    val_record_acc, val_record_f1, val_record_auroc = [], [], []

    epochs_no_improve = 0
    best_val_auroc = -float("inf")
    best_val_acc = -float("inf")
    val_auroc_history: list[float] = []
    val_acc_history: list[float] = []
    SMOOTH_WINDOW = 5
    _s3 = s3fs.S3FileSystem()

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()

        # --- linear warmup: ramp LR up over the first warmup_epochs, bypassing the scheduler ---
        #if warmup_epochs > 0 and epoch < warmup_epochs:
        #    warmup_factor = (epoch + 1) / warmup_epochs   # 1/W, 2/W, ..., W/W
        #    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        #        group["lr"] = base_lr * warmup_factor

        for i, batch in enumerate(padded_train_loader):
            if not warmup_done:
                global_step += 1
                warmup_factor = min(1.0, global_step / warmup_steps)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * warmup_factor
                if global_step >= warmup_steps:
                    warmup_done = True
            if i % 20 == 0 and is_main_process():
                log.info(f"  Epoch {epoch + 1}, batch {i}/{max_batches}, "
                         f"elapsed={(time.time() - start_time) / 60:.1f}min")

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            cnn_feats = (batch["cnn_embeddings"].to(device, non_blocking=True)
                         if use_cnn_features and "cnn_embeddings" in batch else None)
            labels = batch["labels"].to(device, non_blocking=True)  
            # labels are one-hot-encoding (actually multi-hot since more than one class can be 1, if multilabled)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids, use_cnn_features, cnn_feats, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=loop_config["grad_clip_norm"])
            optimizer.step()

        dist.barrier()
        #scheduler.step()

        train_metrics = evaluate_metrics(dataset_name, model, train_loader, use_cnn_features, num_classes, device)
        val_metrics = evaluate_metrics(dataset_name, model, val_loader, use_cnn_features, num_classes, device)

        # --- only hand control to ReduceLROnPlateau after warmup is complete ---
        #if epoch >= warmup_steps:
        if warmup_done:
            scheduler.step(val_metrics["auroc"])
        
        train_losses.append(train_metrics["loss"])
        train_accs.append(train_metrics["acc"])
        train_f1s.append(train_metrics["f1"])
        train_aurocs.append(train_metrics["auroc"])
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["acc"])
        val_f1s.append(val_metrics["f1"])
        val_aurocs.append(val_metrics["auroc"])
        train_record_acc.append(train_metrics["record_acc"])
        train_record_f1.append(train_metrics["record_f1"])
        train_record_auroc.append(train_metrics["record_auroc"])
        val_record_acc.append(val_metrics["record_acc"])
        val_record_f1.append(val_metrics["record_f1"])
        val_record_auroc.append(val_metrics["record_auroc"])

        end_time = time.time()
        if is_main_process():
            log.info(f"Epoch {epoch + 1}/{num_epochs} ({(end_time - start_time) / 60.0:.2f} min) "
                     #f"lr={scheduler.get_last_lr()[0]:.2e}")
                     f"lr={optimizer.param_groups[0]['lr']}")
            log.info(f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} "
                     f"f1={train_metrics['f1']:.4f} AUROC={train_metrics['auroc']:.4f} "
                     f"| record_acc={train_metrics['record_acc']:.4f} record_f1={train_metrics['record_f1']:.4f} "
                     f" record_AUROC={train_metrics['record_auroc']:.4f}")
            log.info(f"  Val:   loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
                     f"f1={val_metrics['f1']:.4f} AUROC={val_metrics['auroc']:.4f} "
                     f"| record_acc={val_metrics['record_acc']:.4f} record_f1={val_metrics['record_f1']:.4f} "
                     f" record_AUROC={val_metrics['record_auroc']:.4f}")
        
        ray.train.report({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"], "train_auroc": train_metrics["auroc"],
            "train_record_auroc": train_metrics["record_auroc"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"], "val_auroc": val_metrics["auroc"],
            "val_record_auroc": val_metrics["record_auroc"],
        })

        # Early stopping + best-checkpoint saving, decided on rank 0 and broadcast.

        val_auroc_history.append(val_metrics["auroc"])
        val_acc_history.append(val_metrics["acc"])
        smoothed_val_auroc = float(np.mean(val_auroc_history[-SMOOTH_WINDOW:]))
        smoothed_val_acc = float(np.mean(val_acc_history[-SMOOTH_WINDOW:]))
        
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if is_main_process():
            # Wait until at least SMOOTH_WINDOW runs
            window_full = len(val_auroc_history) >= SMOOTH_WINDOW

            # If AUROC is NaN (due to class imbalance), use accuracy; otherwise use AUROC
            if np.isnan(smoothed_val_auroc):
                improved = window_full and (smoothed_val_acc - best_val_acc) > min_delta
                metric_name = "accuracy"
            else:
                improved = window_full and (smoothed_val_auroc - best_val_auroc) > min_delta
                metric_name = "AUROC"

            if improved:
                if not np.isnan(smoothed_val_auroc):
                    best_val_auroc = smoothed_val_auroc
                else:
                    best_val_acc = smoothed_val_acc
                epochs_no_improve = 0
                save_lora_checkpoint(model, _s3, use_frac, dataset_name, loop_config["seed"])
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"Early stopping after {epoch + 1} epochs. "
                             f"No improvement in val {metric_name} for {patience} consecutive epochs.")
                    stop_flag[0] = 1

        if ddp_is_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            break

    if is_main_process():
        save_epoch_metrics_npz(
            dataset_name,
            use_frac,
            loop_config["seed"],
            train_losses, train_accs, train_f1s, train_aurocs,
            train_record_acc, train_record_f1, train_record_auroc,
            val_losses, val_accs, val_f1s, val_aurocs,
            val_record_acc, val_record_f1, val_record_auroc,
            _s3,
        )

    dist.barrier()  # make sure rank 0's last save has landed on S3 before everyone reads it
    load_lora_checkpoint(model, _s3, use_frac, dataset_name, loop_config["seed"])
    test_metrics = evaluate_metrics(dataset_name, model, test_loader, use_cnn_features, num_classes, device)
    test_loss = test_metrics["loss"]
    test_acc = test_metrics["acc"]
    test_f1 = test_metrics["f1"]
    test_auroc = test_metrics["auroc"]
    test_record_acc = test_metrics["record_acc"]
    test_record_f1 = test_metrics["record_f1"]
    test_record_auroc = test_metrics["record_auroc"] 

    ray.train.report({
            "test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1, "test_auroc": test_auroc,
            "test_record_acc": test_record_acc, "test_record_f1": test_record_f1, "test_record_auroc": test_record_auroc,
        })

    if is_main_process():
        log.info(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} "
                 f"f1={test_f1:.4f} AUROC={test_auroc:.4f} "
                 f"| record_acc={test_record_acc:.4f} record_f1={test_record_f1:.4f} record_AUROC={test_record_auroc:.4f}")
        save_test_metrics_npz(dataset_name, use_frac, loop_config["seed"], test_loss, test_acc, test_f1, test_auroc, 
                              test_record_acc, test_record_f1, test_record_auroc, _s3)


def run_finetuning(seeds: list[int] = None):
    seeds = seeds or [CONFIG["seed"]]
    results = []
    completed_seeds = []
    for seed in seeds:
        ray.init(ignore_reinit_error=True)
        run_config_copy = dict(CONFIG, seed=seed)
        trainer = ray_torch.TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                "dataset_name": run_config_copy["dataset"],
                "use_frac": run_config_copy["use_frac"],
                "batch_size": run_config_copy["batch_size"],
                "in_channels": run_config_copy["in_channels"],
                "num_classes": NUM_CLASSES[run_config_copy["dataset"]],
                "lr": run_config_copy["lr"],
                "lr_head_factor": run_config_copy["lr_head_factor"],
                "min_lr_factor": run_config_copy["min_lr_factor"],
                "weight_decay": run_config_copy["weight_decay"],
                "wtdecay_head_factor": run_config_copy["wtdecay_head_factor"],
                "dropout": run_config_copy["dropout"],
                "lora_r": run_config_copy["lora_r"],
                "lora_alpha": run_config_copy["lora_alpha"],
                "lora_dropout": run_config_copy["lora_dropout"],
                "num_epochs": run_config_copy["num_epochs"],
                "patience": run_config_copy["patience"],
                "min_delta": run_config_copy["min_delta"],
                "warmup_epochs": run_config_copy["warmup_epochs"],
                "grad_clip_norm": run_config_copy["grad_clip_norm"],
                "seed": run_config_copy["seed"],   
            },
            scaling_config=ScalingConfig(num_workers=CONFIG["num_ray_workers"], use_gpu=CONFIG["use_gpu"]),
            run_config=RunConfig(name=f"vqvae-bert-finetune-{run_config_copy['dataset']}-{run_config_copy['use_frac']}-seed{seed}".replace('.', 'p')),
        )
        result = trainer.fit()
        results.append(result.metrics)
        completed_seeds.append(seed)
        log.info(f"Fine-tuning finished for seed: {seed}, metrics: {result.metrics}")
        ray.shutdown()

    # --- aggregate summary across seeds ---
    test_aurocs = [r["test_auroc"] for r in results if "test_auroc" in r]
    paired_seeds = [s for s, r in zip(completed_seeds, results) if "test_auroc" in r]
    if test_aurocs:
        mean_auroc = float(np.mean(test_aurocs))
        std_auroc  = float(np.std(test_aurocs))
        log.info(
            f"[{CONFIG['dataset']}] use_frac={CONFIG['use_frac']}: "
            f"Test AUROC across {len(test_aurocs)} seed(s) = {mean_auroc:.4f} ± {std_auroc:.4f} "
            f"(per-seed: {dict(zip(paired_seeds, test_aurocs))})"
        )

        summary_key = (
            f"{prefix_finetuning}/{CONFIG['dataset']}/"
            f"finetune_test_summary_{CONFIG['dataset']}_{CONFIG['use_frac']}.npz"
        )
        buf = io.BytesIO()
        np.savez(
            buf,
            seeds=np.array(paired_seeds, dtype=np.int64),
            test_auroc=np.array(test_aurocs, dtype=np.float64),
            mean_test_auroc=np.array([mean_auroc]),
            std_test_auroc=np.array([std_auroc]),
        )
        buf.seek(0)
        _s3 = s3fs.S3FileSystem()
        with _s3.open(f"s3://{bucket_out}/{summary_key}", "wb") as f:
            f.write(buf.read())
        log.info(f"Saved multi-seed summary to s3://{bucket_out}/{summary_key}")
    else:
        log.warning("No test_auroc found in any seed's reported metrics — check ray.train.report call.")

    return results


if __name__ == "__main__":

    # ----------------------- CLI hyper-parameter overrides ---------------------
    parser = argparse.ArgumentParser(description="Fine-tune parameters")
    parser.add_argument("--dataset", type=str, help="dataset", choices=DATASETS)
    parser.add_argument("--use_frac", type=float, choices=[0.01, 0.1, 1.0], help="fraction of the dataset to use")
    parser.add_argument("--in_channels", type=int, choices=[1, NLEADS], help="num leads")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lora_r", type=int, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--weight_decay", type=float, help="weight decay")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds, e.g. '42,0,1'")
    parser.add_argument("--num_ray_workers", type=int, help="number of Ray/DDP workers (1 per GPU)")
    cli_args, _ = parser.parse_known_args()
    for key in ["dataset", "use_frac", "in_channels", "lr", "batch_size", "lora_r",
                "lora_alpha", "lora_dropout", "weight_decay", "num_ray_workers"]:
        if getattr(cli_args, key) is not None:
            CONFIG[key] = getattr(cli_args, key)

    seeds = [int(s) for s in cli_args.seeds.split(",")] if cli_args.seeds else None
    # None if not provided; run_finetuning() defaults to [CONFIG["seed"]]
    run_finetuning(seeds)
