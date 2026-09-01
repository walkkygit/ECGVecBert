"""
Collect every Dandelion finetuning run from S3 into one table (one row per run tag x seed).

Reads s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/results/
    finetune_test_metrics_dandelion_1.0[_<run_tag>].npz            (test metrics, one row per seed)
    finetune_train_val_metrics_dandelion_1.0[_<run_tag>]_seed<S>.npz (per-epoch val curves + best_epoch)

Writes results_dandelion_split2.csv locally and to the same S3 folder, and prints the table sorted by
val record G-mean (the sweep judge). Test columns are for reporting only; never pick a config on them.

Usage:  python collect_results.py            (works locally or on SageMaker; needs S3 read access)
"""

import io
import re
import sys

import numpy as np
import pandas as pd
import s3fs

BUCKET = "walkky-ml"
PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/results"
OUT_NAME = "results_dandelion_split2.csv"

TAG_RE = re.compile(r"cw(?P<cw>[\d.]+)_lr(?P<lr>[\d.e+-]+)_r(?P<r>\d+)_do(?P<do>[\d.]+)_ld(?P<ld>[\d.]+)"
                    r"_wd(?P<wd>[\d.e+-]+)_tm(?P<tm>[A-Za-z]+)(?:_ck(?P<ck>[A-Za-z]+))?")


def load_npz(fs, key):
    with fs.open(f"s3://{BUCKET}/{key}", "rb") as f:
        d = np.load(io.BytesIO(f.read()), allow_pickle=True)
        return {k: d[k] for k in d.files}


def main():
    fs = s3fs.S3FileSystem()
    keys = [k.split(f"{BUCKET}/", 1)[1] for k in fs.ls(f"s3://{BUCKET}/{PREFIX}")]
    test_keys = [k for k in keys if re.search(r"/finetune_test_metrics_dandelion_1\.0(_.+)?\.npz$", k)]
    rows = []
    for tk in sorted(test_keys):
        m = re.search(r"finetune_test_metrics_dandelion_1\.0(?:_(?P<tag>.+))?\.npz$", tk)
        tag = m.group("tag") or ""                       # "" = legacy untagged runs
        t = load_npz(fs, tk)
        cfg = TAG_RE.search(tag).groupdict() if tag and TAG_RE.search(tag) else {}
        for i, seed in enumerate(t["seeds"].tolist()):
            row = {"run_tag": tag or "(untagged)", "seed": int(seed)}
            row.update({"class_weight": cfg.get("cw"), "lr": cfg.get("lr"), "lora_r": cfg.get("r"),
                        "dropout": cfg.get("do"), "lora_dropout": cfg.get("ld"), "weight_decay": cfg.get("wd"),
                        "target_modules": cfg.get("tm"),
                        "ckpt_metric": "val_loss" if cfg.get("ck") == "vloss" else ("gmean" if cfg else None)})
            # per-epoch val curve for this seed -> value at the best epoch
            vk = f"{PREFIX}/finetune_train_val_metrics_dandelion_1.0{'_' + tag if tag else ''}_seed{seed}.npz"
            if fs.exists(f"s3://{BUCKET}/{vk}"):
                v = load_npz(fs, vk)
                be = int(v["best_epoch"]) if "best_epoch" in v else None
                row["best_epoch"] = be
                row["epochs_run"] = int(len(v["epoch"]))
                if be and "val_record_gmean" in v:
                    j = be - 1
                    row["val_gmean"] = float(v["val_record_gmean"][j])
                    row["val_sens"] = float(v["val_record_sensitivity"][j])
                    row["val_spec"] = float(v["val_record_specificity"][j])
                    row["val_auroc"] = float(v["val_record_auroc"][j])
            def g(name):
                return float(t[name][i]) if name in t and i < len(t[name]) else np.nan
            row.update({"test_gmean": g("test_record_gmean"), "test_sens": g("test_record_sensitivity"),
                        "test_spec": g("test_record_specificity"), "test_auroc": g("test_record_auroc"),
                        "test_acc": g("test_record_accuracy"), "test_f1": g("test_record_f1")})
            rows.append(row)

    if not rows:
        print("no result files found"); sys.exit(1)
    cols = ["run_tag", "seed", "class_weight", "lr", "lora_r", "dropout", "lora_dropout", "weight_decay",
            "target_modules", "best_epoch", "epochs_run", "val_gmean", "val_sens", "val_spec", "val_auroc",
            "test_gmean", "test_sens", "test_spec", "test_auroc", "test_acc", "test_f1"]
    df = pd.DataFrame(rows).reindex(columns=cols).sort_values(["val_gmean", "val_auroc"], ascending=False, na_position="last")
    df.to_csv(OUT_NAME, index=False)
    with fs.open(f"s3://{BUCKET}/{PREFIX}/{OUT_NAME}", "w") as f:
        df.to_csv(f, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df.round(4).to_string(index=False))
    print(f"\n{len(df)} rows -> {OUT_NAME} and s3://{BUCKET}/{PREFIX}/{OUT_NAME}")


if __name__ == "__main__":
    main()
