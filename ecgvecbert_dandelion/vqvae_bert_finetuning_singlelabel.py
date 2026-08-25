import io
import re
import math
import time
import numpy as np
import s3fs
import argparse
import logging

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

from vqvae_bert_finetuning_sentences_dataset_singlelabel import (
    DATASETS,
    get_or_build_finetuning_sentences,
    create_finetuning_worker_dataloaders,
)
from vqvae_bert_pretraining import CONFIG as PRETRAIN_CONFIG


#-----------------------parameters definitions-----------------------#
CONFIG = {
    "dataset": "ptbxl",
    "use_frac": 1.0,
    "in_channels": 1,
    "batch_size": 32,
    "num_epochs": 100,
    "lr": 1e-4,
    "min_lr_factor": 0.1,
    "warmup_epochs": 5,
    "dropout": 0.1,
    "weight_decay": 1e-4,
    "patience": 10,
    'min_delta': 1e-6,
    "grad_clip_norm": 1.0,
    "seed": 42,
    # Ray
    "num_ray_workers": 4,
    "use_gpu": True,
    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["W_Q", "W_V", "W_K", "fc", "fc1", "fc2"],
    "modules_to_save": ["classifier", "fc_classifier"],
}


# ---------------------------------------------------------------------------
prefix_sentences = "aruna-files/vqvae_final_12lead_singlelabel/vqvae/bert_finetuning/sentences"
prefix_root = "aruna-files/vqvae_final_12lead_singlelabel/vqvae"

NUM_CLASSES = {"ptbxl": 5,
               "cpsc2018": 9,
               "cs": 11,
}
# Reference table of each dataset's label values (0-indexed: 0 to num_classes-1)
LABELS = {"ptbxl": np.arange(NUM_CLASSES["ptbxl"]),
          "cpsc2018": np.arange(NUM_CLASSES["cpsc2018"]),
          "cs": np.arange(NUM_CLASSES["cs"]),
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


#----------model definition (classifier head) -----------#

class BERTForClassification(nn.Module):
    """Wrap pre-trained BERT with a softmax classifier head."""

    def __init__(self, base_model: BERT, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(d_model, num_classes)
        self.fc_classifier = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, cnn_features, labels=None):
        pooled_output = self.bert(input_ids, cnn_features)  # [B, d_model]
        pooled_output = self.activation(self.fc_classifier(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return logits, loss


def evaluate_metrics(dataset_name:str, model, loader, num_classes: int, device: torch.device):
    model.eval()
    # Bypass the DDP wrapper for eval forward passes. Eval loaders are unpadded
    # (unlike padded_train_loader) so ranks can have unequal batch counts; calling
    # the DDP-wrapped model directly triggers a broadcast_buffers collective on
    # every forward call, which deadlocks once a rank with fewer batches exits
    # its loop while others are still iterating.
    eval_model = unwrap_model(model)
    total_loss = 0.0
    num_batches = 0
    all_labels, all_probs, all_preds, all_ecg_idxs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            cnn_feats = (batch["xresnet_embeddings"].to(device, non_blocking=True)
                          if "xresnet_embeddings" in batch else None)
            labels = batch["labels"].to(device, non_blocking=True)  # labels are 0-indexed (0..num_classes-1)
            ecg_idxs = batch["ecg_idxs"]    # list[str], already on cpu; book-keeping only

            logits, loss = eval_model(input_ids, cnn_feats, labels=labels)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_ecg_idxs.append(np.array(ecg_idxs, dtype=np.str_))

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
    all_probs  = np.concatenate(all_probs,  axis=0) if all_probs  else np.zeros((0, num_classes))
    all_preds  = np.concatenate(all_preds,  axis=0) if all_preds  else np.zeros((0,), dtype=np.int64)
    all_ecg_idxs = np.concatenate(all_ecg_idxs, axis=0) if all_ecg_idxs else np.zeros((0,), dtype=np.str_)

    # ---- DDP: gather per-rank predictions so metrics are computed over the full split ----
    if ddp_is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (all_labels, all_probs, all_preds, all_ecg_idxs, total_loss, num_batches))
        all_labels  = np.concatenate([g[0] for g in gathered], axis=0)
        all_probs   = np.concatenate([g[1] for g in gathered], axis=0)
        all_preds   = np.concatenate([g[2] for g in gathered], axis=0)
        all_ecg_idxs = np.concatenate([g[3] for g in gathered], axis=0)
        total_loss  = sum(g[4] for g in gathered)
        num_batches = sum(g[5] for g in gathered)

    # overall metrics
    # ---- sentence-level metrics--------------
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / max(1, num_batches)
    macro_auroc = _macro_auroc(all_labels, all_probs, num_classes, dataset_name)
    
    # ---- record (ecg_idx)-level aggregation ----
    unique_ecgs = np.unique(all_ecg_idxs)
    record_true  = np.empty(len(unique_ecgs), dtype=np.int64)
    record_pred  = np.empty(len(unique_ecgs), dtype=np.int64)
    record_probs = np.empty((len(unique_ecgs), num_classes), dtype=np.float64)

    for i, ecg_idx in enumerate(unique_ecgs):
        mask = all_ecg_idxs == ecg_idx
        # all sentences from the same ECG share one ECG-level label by construction
        record_true[i] = all_labels[mask][0]
        # majority vote across this record's sentence-level predictions
        votes = np.bincount(all_preds[mask], minlength=num_classes)
        record_pred[i] = np.argmax(votes)
        # mean softmax score across sentences — used only for AUROC
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
        y_true_c = (labels == c).astype(int)
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
    # Log valid classes
    log.info(f"Macro-averaged AUROC = {macro_auroc:.4f} averaged over {len(valid_classes)}/{num_classes} classes for {tag}")
    if not valid_aucs:
        # log splits with no valid classes
        log.info(f"Warning: no valid classes for AUC in this split (all missing or single-class) for {tag}")
    return macro_auroc


#----------- pretrained weight loading -----------#

def load_pretrained_bert_weights(bert: BERT, s3_fs: s3fs.S3FileSystem, key: str) -> None:
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
    log.info(f"Pretrained BERT weights loaded. Missing keys: {missing}. Unexpected keys: {unexpected}")


def build_finetune_model(vocab_size: int, cnn_embed_dim: int, num_classes: int, lora_r: int, lora_alpha: int, lora_dropout: float, dropout: float, prefix_bert_model: str) -> nn.Module:
    """Build a BERTForClassification model, load pretrained BERT weights, and wrap it with LoRA."""
    d_model  = PRETRAIN_CONFIG["d_model"]
    n_layers = PRETRAIN_CONFIG["n_layers"]
    n_heads  = PRETRAIN_CONFIG["n_heads"]
    d_k = d_v = d_model // n_heads
    d_ff = d_model * 4

    # BERT model using default BERT dropout
    bert = BERT(
        vocab_size, MAX_SENTENCE_LEN, d_model, n_layers, n_heads,
        d_k, d_v, d_ff, cnn_embed_dim
    )

    _s3 = s3fs.S3FileSystem()
    load_pretrained_bert_weights(bert, _s3, prefix_bert_model)

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

def save_lora_checkpoint(model: nn.Module, s3_fs: s3fs.S3FileSystem, use_frac: float, dataset_name: str) -> None:
    
    prefix_out = f"{prefix_root}/bert_finetuning/{dataset_name}"
    peft_model = unwrap_model(model)
    state_dict = get_peft_model_state_dict(peft_model)
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    key = f"{prefix_out}/lora_bert_finetuned_{dataset_name}_{use_frac}.pt"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"LoRA checkpoint saved to s3://{bucket_out}/{key}")


def load_lora_checkpoint(model: nn.Module, s3_fs: s3fs.S3FileSystem, use_frac: float, dataset_name: str) -> None:
    prefix_out = f"{prefix_root}/bert_finetuning/{dataset_name}"
    key = f"{prefix_out}/lora_bert_finetuned_{dataset_name}_{use_frac}.pt"
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
    
    prefix_out = f"{prefix_root}/bert_finetuning/{dataset_name}"
    key = f"{prefix_out}/finetune_train_val_metrics_{dataset_name}_{use_frac}.npz"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"Train/val metrics saved to s3://{bucket_out}/{key}")


def save_test_metrics_npz(
    dataset_name: str, use_frac: float, test_loss, test_acc, test_f1, test_auroc, 
    test_record_acc, test_record_f1, test_record_auroc, s3_fs: s3fs.S3FileSystem,
) -> None:
    buf = io.BytesIO()
    np.savez(
        buf,
        test_loss=np.array([test_loss]), test_accuracy=np.array([test_acc]),
        test_f1=np.array([test_f1]), test_auroc=np.array([test_auroc]),
        test_record_accuracy=np.array([test_record_acc]),
        test_record_f1=np.array([test_record_f1]), test_record_auroc=np.array([test_record_auroc]),
    )
    buf.seek(0)
    
    prefix_out = f"{prefix_root}/bert_finetuning/{dataset_name}"
    key = f"{prefix_out}/finetune_test_metrics_{dataset_name}_{use_frac}.npz"
    with s3_fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.read())
    log.info(f"Test metrics saved to s3://{bucket_out}/{key}")


#----------- Ray + DDP training loop -----------#

def train_loop_per_worker(loop_config: dict) -> None:
    ctx         = ray.train.get_context()
    world_rank  = ctx.get_world_rank()
    local_rank  = ctx.get_local_rank()
    device      = ray_torch.get_device()

    torch.manual_seed(CONFIG["seed"] + world_rank)

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

    prefix_bert_model = f"{prefix_root}/bert_pretraining/bert_model_nleads_{in_channels}.pt"

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
    loaders = create_finetuning_worker_dataloaders(dataset_name, use_frac, batch_size)
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    # After creating loaders, before the epoch loop:
    # synchronize the number of training steps across all ranks and pad loaders
    batches = torch.tensor([len(train_loader)], dtype=torch.long, device=device)
    dist.all_reduce(batches, op=dist.ReduceOp.MAX)
    max_batches = batches.item()
    # Wrap loader — ranks with fewer batches cycle; ranks at MAX just iterate normally
    padded_train_loader = PaddedLoader(train_loader, max_batches)

    vocab_size    = train_loader.dataset.vocab_size
    cnn_embed_dim = train_loader.dataset.xresnet_embedding_dim

    if is_main_process():
        log.info(
            f"[{dataset_name}] Train: {len(train_loader.dataset):,} | "
            f"Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,} | "
            f"vocab_size={vocab_size} | num_classes={num_classes} "
            f"use_frac={use_frac} "
        )

    model = build_finetune_model(vocab_size, cnn_embed_dim, num_classes, lora_r, lora_alpha, lora_dropout, dropout, prefix_bert_model)
    model = ray_torch.prepare_model(model)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=loop_config["lr"],
        weight_decay=loop_config["weight_decay"],
    )

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs         # ramps 1/W, 2/W, ... W/W
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Metrics, sequence-level
    train_losses, train_accs, train_f1s, train_aurocs = [], [], [], []
    val_losses, val_accs, val_f1s, val_aurocs = [], [], [], []
    # Metrics, record level
    train_record_acc, train_record_f1, train_record_auroc = [], [], []
    val_record_acc, val_record_f1, val_record_auroc = [], [], []

    best_val_auroc = -float("inf")
    epochs_no_improve = 0
    _s3 = s3fs.S3FileSystem()

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for i, batch in enumerate(padded_train_loader):
            if i % 20 == 0 and is_main_process():
                log.info(f"  Epoch {epoch + 1}, batch {i}/{max_batches}, "
                         f"elapsed={(time.time() - start_time) / 60:.1f}min")

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            cnn_feats = (batch["xresnet_embeddings"].to(device, non_blocking=True)
                         if "xresnet_embeddings" in batch else None)
            labels    = batch["labels"].to(device, non_blocking=True)  # labels are 0-indexed (0..num_classes-1)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids, cnn_feats, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=loop_config["grad_clip_norm"])
            optimizer.step()

        dist.barrier()
        scheduler.step()

        train_metrics = evaluate_metrics(dataset_name, model, train_loader, num_classes, device)
        val_metrics   = evaluate_metrics(dataset_name, model, val_loader,   num_classes, device)
        
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
                     f"lr={scheduler.get_last_lr()[0]:.2e}")
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
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if is_main_process():
            improved = (val_metrics["auroc"] - best_val_auroc) > min_delta
            if improved:
                best_val_auroc = val_metrics["auroc"]
                epochs_no_improve = 0
                save_lora_checkpoint(model, _s3, use_frac, dataset_name)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"Early stopping after {epoch + 1} epochs. "
                             f"No improvement in val auroc for {patience} consecutive epochs.")
                    stop_flag[0] = 1

        if ddp_is_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            break

    if is_main_process():
        save_epoch_metrics_npz(
            dataset_name,
            use_frac,
            train_losses, train_accs, train_f1s, train_aurocs,
            train_record_acc, train_record_f1, train_record_auroc,
            val_losses, val_accs, val_f1s, val_aurocs,
            val_record_acc, val_record_f1, val_record_auroc,
            _s3,
        )

    dist.barrier()  # make sure rank 0's last save has landed on S3 before everyone reads it
    load_lora_checkpoint(model, _s3, use_frac, dataset_name)
    test_metrics = evaluate_metrics(dataset_name, model, test_loader, num_classes, device)
    test_loss = test_metrics["loss"]
    test_acc = test_metrics["acc"]
    test_f1 = test_metrics["f1"]
    test_auroc = test_metrics["auroc"]
    test_record_acc = test_metrics["record_acc"]
    test_record_f1 = test_metrics["record_f1"]
    test_record_auroc = test_metrics["record_auroc"] 
     
    if is_main_process():
        log.info(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} "
                 f"f1={test_f1:.4f} AUROC={test_auroc:.4f} "
                 f"| record_acc={test_record_acc:.4f} record_f1={test_record_f1:.4f} record_AUROC={test_record_auroc:.4f}")
        save_test_metrics_npz(dataset_name, use_frac, test_loss, test_acc, test_f1, test_auroc, 
                              test_record_acc, test_record_f1, test_record_auroc, _s3)

 
def run_finetuning():
    ray.init(ignore_reinit_error=True)

    trainer = ray_torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "dataset_name": CONFIG["dataset"],
            "use_frac": CONFIG["use_frac"],
            "batch_size": CONFIG["batch_size"],
            "in_channels": CONFIG["in_channels"],
            "num_classes": NUM_CLASSES[CONFIG["dataset"]],
            "lr": CONFIG["lr"],
            "weight_decay": CONFIG["weight_decay"],
            "dropout": CONFIG["dropout"],
            "lora_r": CONFIG["lora_r"],
            "lora_alpha": CONFIG["lora_alpha"],
            "lora_dropout": CONFIG["lora_dropout"],
            "num_epochs": CONFIG["num_epochs"],
            "patience": CONFIG["patience"],
            "min_delta": CONFIG["min_delta"],
            "min_lr_factor": CONFIG["min_lr_factor"],
            "warmup_epochs": CONFIG["warmup_epochs"],
            "grad_clip_norm": CONFIG["grad_clip_norm"],
        },
        scaling_config=ScalingConfig(
            num_workers=CONFIG["num_ray_workers"],
            use_gpu=CONFIG["use_gpu"],
        ),
        run_config=RunConfig(name=f"vqvae-bert-finetune-{CONFIG['dataset']}-{CONFIG['use_frac']}".replace('.', 'p')),
    )

    result = trainer.fit()
    log.info(f"Fine-tuning finished. Final metrics: {result.metrics}")
    return result


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
    parser.add_argument("--num_ray_workers", type=int, help="number of Ray/DDP workers (1 per GPU)")
    cli_args, _ = parser.parse_known_args()
    for key in ["dataset", "use_frac", "in_channels", "lr", "batch_size", "lora_r",
                "lora_alpha", "lora_dropout", "weight_decay", "num_ray_workers"]:
        if getattr(cli_args, key) is not None:
            CONFIG[key] = getattr(cli_args, key)
    
    run_finetuning()
