"""
One-shot evaluation of a finished Dandelion finetuning run on a held-out sentence set.

Split-2 protocol (2026-08-31): the 7,000 FDA validation patients are a LOCKED final test set,
evaluated exactly ONCE on the single chosen model. This script is that one evaluation. It never
trains anything: it rebuilds the model from the run tag, loads the saved LoRA checkpoint(s), predicts
every sentence, aggregates to one prediction per patient with the SAME rules as training
(majority vote of sentence predictions for sens/spec, mean sentence probability for AUROC) and
saves per-patient probabilities + metrics to S3.

Two sentence sets:
  --split internal_test   split_2_related/sentences/dandelion_test_full_*      (2,090 pts, the usual "test")
                          -> DRY RUN. Numbers must reproduce the run's row in results_dandelion_split2.csv.
  --split 7k              split_2_related/sentences_7k/dandelion_7k_full_*     (6,977 of the 7,000 FDA pts;
                          23 dropped at sentence building for too few R-peaks, reported as-is)
                          -> THE FINAL RUN. Needs --i_confirm_final_7k yes and refuses to overwrite.

Seeds: every seed's checkpoint is evaluated on its own, then the seeds' sentence probabilities are
averaged into an ensemble that is aggregated the same way. Sentence order is identical across seeds
(same shards, no shuffling), so the average is aligned by construction.

Usage (SageMaker, via evaluate_final_7k_launcher.py) or locally with GPU/CPU:
    python evaluate_final_7k.py --run_tag cw3.0_lr1e-05_r8_do0.1_ld0.1_wd1e-04_tmall_ckvloss \
        --seeds 42,0,1,812,995 --split internal_test
    python evaluate_final_7k.py --run_tag <winner> --seeds 42,0,1,812,995 --split 7k --i_confirm_final_7k yes

Outputs (S3, under split_2_related/results/):
    eval_internal_test/<run_tag>/   or   final_7k/<run_tag>/
        predictions_seed<S>.npz     ecg_idxs, label, mean_prob_low_ef, vote_pred_low_ef  (one row per patient)
        predictions_ensemble.npz    same, from the seed-averaged sentence probabilities
        metrics.json                per-seed + ensemble metrics, counts, checkpoint keys, timestamp
"""

import argparse
import datetime as _dt
import io
import json
import logging
import os
import re

import numpy as np
import s3fs
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from vqvae_bert_finetuning import (
    CONFIG, NUM_CLASSES, PRETRAIN_CONFIG, prefix_dandelion_split2, prefix_root_bert_model,
    build_finetune_model, load_lora_checkpoint, unwrap_model,
)
from vqvae_bert_finetuning_sentences_dataset import (
    NUM_GENERATION_SHARDS, SEGMENTS, FinetuningBeatSentenceDataset, _concat_sentence_dicts,
)
from vqvae_ecg_waveforms_dataset import bucket_out
from collect_results import TAG_RE

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATASET = "dandelion"
USE_FRAC = 1.0
SPLITS = {
    # split name -> (S3 prefix of the shards, file stem, results sub-prefix)
    "internal_test": (f"{prefix_dandelion_split2}/sentences", "dandelion_test_full", "eval_internal_test"),
    "7k":            (f"{prefix_dandelion_split2}/sentences_7k", "dandelion_7k_full", "final_7k"),
}
ATTN_ONLY = ["W_Q", "W_K", "W_V"]


# ----------------------------------------------------------------------------- config from the run tag
def config_from_run_tag(run_tag: str) -> dict:
    """Recover the architecture knobs the checkpoint was trained with. LoRA r / alpha / target modules
    MUST match or the checkpoint will not load (or will load into a differently scaled adapter)."""
    m = TAG_RE.search(run_tag)
    if not m:
        raise ValueError(f"run_tag {run_tag!r} does not parse; expected e.g. cw2.2_lr1e-05_r8_do0.1_ld0.1_wd1e-04_tmall_ckvloss")
    g = m.groupdict()
    return {
        "class_weight": float(g["cw"]),
        "lora_r": int(g["r"]),
        "lora_alpha": int(g["a"]) if g.get("a") else int(CONFIG["lora_alpha"]),
        "lora_dropout": float(g["ld"]),
        "dropout": float(g["do"]),
        "target_modules": list(CONFIG["target_modules"]) if g["tm"] == "all" else ATTN_ONLY,
    }


# ----------------------------------------------------------------------------- data
def load_split_sentences(fs: s3fs.S3FileSystem, split: str, num_shards: int = NUM_GENERATION_SHARDS) -> dict:
    prefix, stem, _ = SPLITS[split]
    parts = []
    for si in range(num_shards):
        key = f"s3://{bucket_out}/{prefix}/{stem}_shard_{si:04d}_of_{num_shards:04d}.npz"
        with fs.open(key, "rb") as f:
            d = np.load(io.BytesIO(f.read()), allow_pickle=True)
            part = {
                "sentence_tokens": d["sentence_tokens"], "labels": d["labels"], "ecg_idxs": d["ecg_idxs"],
                "num_beats": d["num_beats"], "beat_idx_start": d["beat_idx_start"], "beat_idx_end": d["beat_idx_end"],
                "segment_order": list(d["segment_order"].astype(str)),
                "vocab_size": int(np.array(d["vocab_size"]).reshape(-1)[0]),
                "vocab_offsets": {seg: int(np.array(d[f"vocab_offset_{seg}"]).reshape(-1)[0]) for seg in SEGMENTS},
            }
            if "cnn_embeddings" in d:
                part["cnn_embeddings"] = d["cnn_embeddings"]
        parts.append(part)
        if si % 16 == 15:
            log.info(f"  loaded {si + 1}/{num_shards} shards of {split}")
    merged = _concat_sentence_dicts(parts)
    n_ecg = len(np.unique(merged["ecg_idxs"]))
    log.info(f"[{split}] {len(merged['sentence_tokens']):,} sentences / {n_ecg:,} ECGs from {prefix}/{stem}_*")
    return merged


# ----------------------------------------------------------------------------- predict + aggregate
@torch.no_grad()
def predict_sentences(model, loader: DataLoader, use_cnn: bool, device: torch.device):
    """Sentence-level softmax probabilities in loader order. Returns (probs[N,2], labels[N,2], ecg_idxs[N])."""
    model.eval()
    m = unwrap_model(model)
    probs, labels, ecg_idxs = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        cnn = batch["cnn_embeddings"].to(device, non_blocking=True) if use_cnn and "cnn_embeddings" in batch else None
        logits, _ = m(input_ids, use_cnn, cnn, labels=None)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch["labels"].numpy())
        ecg_idxs.append(np.array(batch["ecg_idxs"], dtype=np.str_))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(ecg_idxs)


def aggregate_records(probs: np.ndarray, labels: np.ndarray, ecg_idxs: np.ndarray) -> dict:
    """Per-patient aggregation, identical to evaluate_metrics() in vqvae_bert_finetuning.py:
    vote_pred  = low EF iff a STRICT majority of the record's sentences argmax to low EF (ties -> normal)
    mean_prob  = mean of the record's sentence P(low EF), used for AUROC.
    Also reports the alternative rule mean_prob >= 0.5 (not used for selection; for information only)."""
    uniq, inv = np.unique(ecg_idxs, return_inverse=True)
    n_rec = len(uniq)
    sent_pred = (np.argmax(probs, axis=1) == 1).astype(np.int64)
    votes = np.bincount(inv, weights=sent_pred, minlength=n_rec)
    counts = np.bincount(inv, minlength=n_rec)
    mean_prob = np.bincount(inv, weights=probs[:, 1], minlength=n_rec) / counts
    vote_pred = (votes > counts / 2).astype(np.int64)
    label = np.zeros(n_rec, dtype=np.int64)
    label[inv] = labels[:, 1].astype(np.int64)          # identical across a record's sentences
    meanprob_pred = (mean_prob >= 0.5).astype(np.int64)

    def _metrics(pred):
        sens = recall_score(label, pred, pos_label=1, zero_division=0)
        spec = recall_score(label, pred, pos_label=0, zero_division=0)
        prec = precision_score(label, pred, pos_label=1, zero_division=0)
        return {"sens": float(sens), "spec": float(spec), "prec": float(prec), "gmean": float(np.sqrt(sens * spec)),
                "tp": int(((pred == 1) & (label == 1)).sum()), "fn": int(((pred == 0) & (label == 1)).sum()),
                "tn": int(((pred == 0) & (label == 0)).sum()), "fp": int(((pred == 1) & (label == 0)).sum())}

    auroc = float(roc_auc_score(label, mean_prob)) if 0 < label.sum() < n_rec else float("nan")
    out = {"n_records": int(n_rec), "n_sentences": int(len(probs)), "n_low_ef": int(label.sum()),
           "auroc": auroc, "vote": _metrics(vote_pred), "meanprob_at_0.5": _metrics(meanprob_pred)}
    arrays = {"ecg_idxs": uniq, "label": label, "mean_prob_low_ef": mean_prob, "vote_pred_low_ef": vote_pred,
              "n_sentences": counts}
    return out, arrays


def _fmt(m: dict) -> str:
    v = m["vote"]
    return (f"sens={v['sens']:.4f} spec={v['spec']:.4f} G={v['gmean']:.4f} AUROC={m['auroc']:.4f} "
            f"(TP {v['tp']} FN {v['fn']} TN {v['tn']} FP {v['fp']}; n={m['n_records']})")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="One-shot evaluation of a Dandelion finetuning run")
    p.add_argument("--run_tag", type=str, required=True, help="run tag of the finished run (as in results_dandelion_split2.csv)")
    p.add_argument("--seeds", type=str, default="42", help="comma-separated seeds whose checkpoints exist for this tag")
    p.add_argument("--split", type=str, default="internal_test", choices=sorted(SPLITS))
    p.add_argument("--i_confirm_final_7k", type=str, default="no", help="must be 'yes' for --split 7k")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force", type=str, default="no", help="'yes' overwrites existing results (never for 7k)")
    p.add_argument("--num_shards", type=int, default=NUM_GENERATION_SHARDS, help="debug: read only the first N shards")
    p.add_argument("--local_out", type=str, default=None, help="optional local dir for a copy of the outputs")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    fs = s3fs.S3FileSystem()
    _, _, results_sub = SPLITS[a.split]
    out_prefix = f"{prefix_dandelion_split2}/results/{results_sub}/{a.run_tag}"
    metrics_key = f"s3://{bucket_out}/{out_prefix}/metrics.json"

    if a.split == "7k":
        if a.i_confirm_final_7k != "yes":
            raise SystemExit("--split 7k is the ONE locked final evaluation; pass --i_confirm_final_7k yes")
        if fs.exists(metrics_key):
            raise SystemExit(f"REFUSING: {metrics_key} already exists. The 7K is evaluated once; there is no --force for it.")
    elif fs.exists(metrics_key) and a.force != "yes":
        raise SystemExit(f"{metrics_key} exists; pass --force yes to overwrite")

    cfg = config_from_run_tag(a.run_tag)
    log.info(f"run_tag={a.run_tag} -> {cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cnn = PRETRAIN_CONFIG["use_cnn_features"]

    sentences = load_split_sentences(fs, a.split, num_shards=a.num_shards)
    dataset = FinetuningBeatSentenceDataset(sentences)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=0, drop_last=False)
    cnn_dim = dataset.cnn_embedding_dim if use_cnn else 0

    prefix_bert_model = f"{prefix_root_bert_model}/bert_model_nleads_{CONFIG['in_channels']}.pt"
    results = {"run_tag": a.run_tag, "split": a.split, "seeds": seeds, "config": cfg,
               "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "sentences_prefix": f"{SPLITS[a.split][0]}/{SPLITS[a.split][1]}_*", "num_shards_read": a.num_shards,
               "aggregation": "vote = strict majority of sentence argmax (as in training); auroc = mean sentence P(low EF)",
               "per_seed": {}, "checkpoints": {}}
    prob_sum = None
    ref_idxs = None
    for seed in seeds:
        model = build_finetune_model(dataset.vocab_size, use_cnn, cnn_dim, NUM_CLASSES[DATASET], cfg["lora_r"], cfg["lora_alpha"],
                                     cfg["lora_dropout"], cfg["dropout"], prefix_bert_model, dataset_name=DATASET,
                                     class_weight=[1.0, cfg["class_weight"]], target_modules=cfg["target_modules"])
        res = load_lora_checkpoint(model, fs, USE_FRAC, DATASET, seed, run_tag=a.run_tag)
        # A LoRA state dict holds only adapter + head weights; frozen base-model keys are always "missing".
        # What must NOT happen: unexpected keys (architecture mismatch) or a missing adapter/head weight.
        bad_missing = [k for k in res.missing_keys if "lora_" in k or "modules_to_save" in k]
        assert not res.unexpected_keys and not bad_missing, (
            f"seed {seed}: checkpoint/architecture mismatch; unexpected={res.unexpected_keys[:5]} "
            f"missing adapter/head={bad_missing[:5]}")
        results["checkpoints"][str(seed)] = (f"{prefix_dandelion_split2}/results/lora_bert_finetuned_{DATASET}_{USE_FRAC}"
                                             f"_{a.run_tag}_seed{seed}.pt")
        model.to(device)
        probs, labels, ecg_idxs = predict_sentences(model, loader, use_cnn, device)
        if ref_idxs is None:
            ref_idxs, ref_labels = ecg_idxs, labels
        else:
            assert np.array_equal(ref_idxs, ecg_idxs), "sentence order differs between seeds"
        prob_sum = probs if prob_sum is None else prob_sum + probs
        m, arrays = aggregate_records(probs, labels, ecg_idxs)
        results["per_seed"][str(seed)] = m
        log.info(f"[{a.split}] seed {seed}: {_fmt(m)}")
        _save_npz(fs, f"{out_prefix}/predictions_seed{seed}.npz", arrays, a.local_out)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(seeds) > 1:
        m, arrays = aggregate_records(prob_sum / len(seeds), ref_labels, ref_idxs)
        results["ensemble"] = m
        log.info(f"[{a.split}] ENSEMBLE of {len(seeds)} seeds: {_fmt(m)}")
        _save_npz(fs, f"{out_prefix}/predictions_ensemble.npz", arrays, a.local_out)

    body = json.dumps(results, indent=2)
    with fs.open(metrics_key, "w") as f:
        f.write(body)
    if a.local_out:
        os.makedirs(a.local_out, exist_ok=True)
        with open(os.path.join(a.local_out, "metrics.json"), "w") as f:
            f.write(body)
    log.info(f"metrics -> {metrics_key}")
    if a.split == "7k":
        log.info("FINAL 7K EVALUATION DONE. Do not run again on this run_tag.")


def _save_npz(fs, key: str, arrays: dict, local_out: str | None):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    with fs.open(f"s3://{bucket_out}/{key}", "wb") as f:
        f.write(buf.getvalue())
    if local_out:
        os.makedirs(local_out, exist_ok=True)
        with open(os.path.join(local_out, os.path.basename(key)), "wb") as f:
            f.write(buf.getvalue())


if __name__ == "__main__":
    main()
