"""
Plot per-epoch training curves of Dandelion finetuning runs from the S3 metrics files.

For each run (one `finetune_train_val_metrics_dandelion_1.0[_<run_tag>]_seed<S>.npz` on S3):
    <out_dir>/curves_<run_tag>_seed<S>.png   left: train vs val loss | right: val sens / spec / G-mean, best epoch marked
Plus one comparison figure:
    <out_dir>/compare_val_gmean.png          val G-mean per epoch, all runs overlaid

Reads only from S3; writes only PNGs into --out_dir (default ~/Desktop/ECGVecBert/results_split2).

Usage:
    python plot_curves.py                      # all runs
    python plot_curves.py --match cw3.5 cw4.5  # only runs whose tag contains one of these
"""

import argparse
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import s3fs

BUCKET = "walkky-ml"
PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/results"
FILE_RE = re.compile(r"finetune_train_val_metrics_dandelion_1\.0(?:_(?P<tag>.+?))?_seed(?P<seed>\d+)\.npz$")


def load_runs(fs, match):
    runs = []
    for key in fs.ls(f"s3://{BUCKET}/{PREFIX}"):
        m = FILE_RE.search(key)
        if not m:
            continue
        tag = m.group("tag") or "untagged"
        if match and not any(s in tag for s in match):
            continue
        with fs.open(key, "rb") as f:
            d = np.load(io.BytesIO(f.read()), allow_pickle=True)
            runs.append((tag, int(m.group("seed")), {k: d[k] for k in d.files}))
    return sorted(runs)


def short(tag):
    """cw3.5_lr1e-04_r8_do0.1_ld0.1_wd1e-04_tmall -> cw3.5 (only the parts that differ from the default)"""
    if tag == "untagged":
        return "cw1.9 (untagged run)"
    default = {"lr": "1e-04", "r": "8", "do": "0.1", "ld": "0.1", "wd": "1e-04", "tm": "all", "hf": "0.1", "a": "16"}
    parts = []
    for p in tag.split("_"):
        m = re.match(r"(cw|lr|r|do|ld|wd|tm|ck|hf|a)(.*)", p)
        if not m:
            continue
        k, v = m.groups()
        if k == "cw" or default.get(k) != v:
            parts.append(p)
    return " ".join(parts) if parts else tag


def plot_run(tag, seed, d, out_dir):
    ep = d["epoch"]
    best = int(d["best_epoch"]) if "best_epoch" in d else None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(ep, d["train_loss"], "o-", label="train loss")
    ax1.plot(ep, d["val_loss"], "s-", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("weighted cross-entropy"); ax1.set_title("Loss"); ax1.grid(alpha=0.3); ax1.legend()
    if "val_record_gmean" in d:
        ax2.plot(ep, d["val_record_sensitivity"], "o-", label="val sens (per patient)")
        ax2.plot(ep, d["val_record_specificity"], "s-", label="val spec (per patient)")
        ax2.plot(ep, d["val_record_gmean"], "k^-", label="val G-mean")
    ax2.plot(ep, d["val_record_auroc"], "--", color="gray", label="val AUROC (per patient)")
    ax2.set_xlabel("epoch"); ax2.set_ylim(0.5, 1.0); ax2.set_title("Validation, per patient, 0.5 line"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    if best:
        for ax in (ax1, ax2):
            ax.axvline(best, color="red", ls=":", lw=1.5)
        ax2.text(best + 0.2, 0.52, f"best epoch {best}", color="red", fontsize=9)
    fig.suptitle(f"{short(tag)}   (seed {seed})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"curves_{tag}_seed{seed}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_compare(runs, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for tag, seed, d in runs:
        if "val_record_gmean" not in d:
            continue
        lbl = short(tag)
        ax1.plot(d["epoch"], d["val_record_gmean"], "o-", ms=3, label=lbl)
        ax2.plot(d["epoch"], d["val_loss"], "o-", ms=3, label=lbl)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val G-mean (per patient)"); ax1.set_title("Val G-mean per epoch"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val loss (weighted, not comparable across weights)"); ax2.set_title("Val loss per epoch"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "compare_val_gmean.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.expanduser("~/Desktop/ECGVecBert/results_split2"))
    ap.add_argument("--match", nargs="*", default=None, help="only tags containing one of these substrings")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fs = s3fs.S3FileSystem()
    runs = load_runs(fs, args.match)
    if not runs:
        raise SystemExit("no runs found")
    for tag, seed, d in runs:
        print("wrote", plot_run(tag, seed, d, args.out_dir))
    print("wrote", plot_compare(runs, args.out_dir))
