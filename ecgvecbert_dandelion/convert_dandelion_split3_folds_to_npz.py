"""
Build the split-3-folds NPZ for Dandelion LVEF fold training (phase 3, 2026-09-03).

Same 10,519 patients / signals / labels / strat_fold as the split-2 NPZ (val 1,050 and internal test 2,100
unchanged, the 7K stays locked). The only addition is `fold_count` (N, 3) int8: how many times each patient
appears in training fold 1 / 2 / 3 of split_patients/training_patients.pkl (0 = not in that fold).
Folds 1 and 2 are 3,600 unique patients each (2,160 normal + 1,440 low EF); fold 3 lists 3,600 entries but
only 3,106 unique patients (some appear twice), so the count matters: the training loader can either
de-duplicate (count > 0) or repeat (count times).

Inputs (read only, nothing under split 1 / split 2 is modified):
  - npz_split2/dandelion_split2_combined_signals.npz   (local; S3 split_2_related/datasets/ on SageMaker)
  - split_patients/training_patients.pkl                (fold, label_0, label_1)
Outputs:
  - npz_split3_folds/dandelion_split3_folds_combined_signals.npz   (2.5 GB, full signals + fold_count)
  - npz_split3_folds/dandelion_split3_folds_fold_ids.npz           (small sidecar: ids, labels, strat_fold,
                                                                     fold_count; the training loader reads this)
  - both uploaded to s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/dandelion/split_3_folds/datasets/

Usage:
  python convert_dandelion_split3_folds_to_npz.py            # build + upload
  python convert_dandelion_split3_folds_to_npz.py --no_upload
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs

S3_BUCKET = "walkky-ml"
S3_SPLIT2_NPZ = f"s3://{S3_BUCKET}/ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/datasets/dandelion_split2_combined_signals.npz"
S3_OUT_PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_3_folds/datasets"

ROOT = Path(__file__).resolve().parent.parent            # ~/Desktop/ECGVecBert
SPLIT2_NPZ = ROOT / "npz_split2" / "dandelion_split2_combined_signals.npz"
FOLD_PKL = ROOT / "split_patients" / "training_patients.pkl"
OUT_DIR = ROOT / "npz_split3_folds"
OUT_NPZ = OUT_DIR / "dandelion_split3_folds_combined_signals.npz"
OUT_IDS = OUT_DIR / "dandelion_split3_folds_fold_ids.npz"

NUM_FOLDS = 3
EXPECTED_ENTRIES = {1: (2160, 1440), 2: (2160, 1440), 3: (2160, 1440)}      # (label_0, label_1) entries per fold
EXPECTED_UNIQUE = {1: 3600, 2: 3600, 3: 3106}                               # unique patients per fold

parser = argparse.ArgumentParser()
parser.add_argument("--no_upload", action="store_true")
args = parser.parse_args()

print("[1] Loading split-2 NPZ (read only) ...")
if SPLIT2_NPZ.exists():
    data = np.load(SPLIT2_NPZ, allow_pickle=True)
    print(f"    local {SPLIT2_NPZ}")
else:
    import io
    with s3fs.S3FileSystem().open(S3_SPLIT2_NPZ, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
    print(f"    {S3_SPLIT2_NPZ}")
ecg_signals = data["ecg_signals"]
lengths = data["lengths"]
labels = data["labels"]
ids = data["ids"].astype(str)
strat_fold = data["strat_fold"]
N = len(ids)
print(f"    {N:,} records, signals {ecg_signals.shape} {ecg_signals.dtype}, "
      f"train {(strat_fold == 0).sum():,} / val {(strat_fold == 1).sum():,} / test {(strat_fold == 2).sum():,}")
assert len(np.unique(ids)) == N, "duplicate ids in the split-2 NPZ"

print("[2] Reading folds ...")
df = pd.read_pickle(FOLD_PKL)
assert sorted(df["fold"].tolist()) == list(range(1, NUM_FOLDS + 1)), df["fold"].tolist()
id_to_row = {pid: i for i, pid in enumerate(ids)}
fold_count = np.zeros((N, NUM_FOLDS), dtype=np.int8)
for r in df.itertuples():
    l0 = [str(x) for x in r.label_0]
    l1 = [str(x) for x in r.label_1]
    assert (len(l0), len(l1)) == EXPECTED_ENTRIES[int(r.fold)], (r.fold, len(l0), len(l1))
    assert not set(l0) & set(l1), f"fold {r.fold}: a patient listed under both labels"
    for pid in l0 + l1:
        fold_count[id_to_row[pid], int(r.fold) - 1] += 1
    rows0 = [id_to_row[p] for p in set(l0)]
    rows1 = [id_to_row[p] for p in set(l1)]
    assert np.all(labels[rows0, 1] == 0) and np.all(labels[rows1, 1] == 1), f"fold {r.fold}: label mismatch vs NPZ"
    assert np.all(strat_fold[rows0 + rows1] == 0), f"fold {r.fold}: patient outside the split-2 train set"
    n_unique = int((fold_count[:, int(r.fold) - 1] > 0).sum())
    n_entries = int(fold_count[:, int(r.fold) - 1].sum())
    assert n_unique == EXPECTED_UNIQUE[int(r.fold)], (r.fold, n_unique)
    print(f"    fold {r.fold}: {n_entries:,} entries, {n_unique:,} unique patients "
          f"({len(set(l0)):,} normal / {len(set(l1)):,} low EF), max repeat {int(fold_count[:, int(r.fold) - 1].max())}")

in_any = fold_count.sum(axis=1) > 0
assert np.array_equal(in_any, strat_fold == 0), "union of folds != split-2 train set"
assert fold_count[strat_fold != 0].sum() == 0
print(f"    union of folds = {int(in_any.sum()):,} = split-2 train set (val / internal test untouched)")

print("[3] Writing ...")
OUT_DIR.mkdir(exist_ok=True)
np.savez(OUT_NPZ, ecg_signals=ecg_signals, lengths=lengths, labels=labels, ids=data["ids"],
         strat_fold=strat_fold, fold_count=fold_count)
np.savez(OUT_IDS, ids=data["ids"], labels=labels, strat_fold=strat_fold, fold_count=fold_count)
print(f"    {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1e9:.2f} GB)")
print(f"    {OUT_IDS} ({OUT_IDS.stat().st_size / 1e3:.0f} KB)")

print("[4] Round-trip check ...")
chk = np.load(OUT_NPZ, allow_pickle=True)
assert np.array_equal(chk["ids"].astype(str), ids) and np.array_equal(chk["fold_count"], fold_count)
assert np.array_equal(chk["strat_fold"], strat_fold) and np.array_equal(chk["labels"], labels)
assert np.array_equal(chk["lengths"], lengths) and chk["ecg_signals"].shape == ecg_signals.shape
assert np.array_equal(chk["ecg_signals"][:50], ecg_signals[:50]) and np.array_equal(chk["ecg_signals"][-50:], ecg_signals[-50:])
side = np.load(OUT_IDS, allow_pickle=True)
assert np.array_equal(side["ids"].astype(str), ids) and np.array_equal(side["fold_count"], fold_count)
print("    OK")

if args.no_upload:
    print("[5] upload skipped (--no_upload)")
else:
    fs = s3fs.S3FileSystem()
    for local in (OUT_IDS, OUT_NPZ):
        dest = f"s3://{S3_BUCKET}/{S3_OUT_PREFIX}/{local.name}"
        print(f"[5] Uploading {local.name} -> {dest} ...")
        fs.put(str(local), dest)
        assert fs.size(dest) == local.stat().st_size, f"size mismatch after upload: {dest}"
        print(f"    OK ({fs.size(dest):,} bytes)")
print("DONE")
