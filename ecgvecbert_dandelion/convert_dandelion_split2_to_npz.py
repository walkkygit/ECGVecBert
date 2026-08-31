"""
Convert Dandelion feathers to NPZ using the NEW patient split (phase 2 protocol).

Differences from convert_dandelion_to_npz_full.py (which is kept unchanged):
  - Splits come from the pickles in ECGVecBert/split_patients/, NOT from
    train_test_split: train = union of folds 1-3 (7,369), val = 1,050,
    internal test = 2,100.
  - The official 7K testing feather is NEVER loaded. The locked final test
    set is not present in this NPZ at all.
  - Output: npz_split2/dandelion_split2_combined_signals.npz, uploaded to
    s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/datasets_split2/

strat_fold: 0 = train, 1 = val, 2 = internal test (test_patients_not7K).

Usage:
  python convert_dandelion_split2_to_npz.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import s3fs
from pathlib import Path

# Settings
S3_BUCKET = "walkky-ml"
S3_DATA_PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/feather_files"  # feathers (read)
S3_OUT_PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/datasets"  # NPZ (write)
SPLIT_DIR = Path(__file__).resolve().parent.parent / "split_patients"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "npz_split2"
MAX_LEN = 5000
LEAD_COLS = ['lead_I', 'lead_II', 'lead_III',
             'lead_aVR', 'lead_aVL', 'lead_aVF',
             'lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6']

print("=" * 70)
print("DANDELION → NPZ CONVERSION (SPLIT 2: fixed patient lists, no 7K)")
print("=" * 70)

# ────────────────────────────────────────────────────────────────────────────
# Load the patient split lists
# ────────────────────────────────────────────────────────────────────────────

print("\n[1] Loading patient split pickles...")
with open(SPLIT_DIR / "training_patients_list.pkl", "rb") as f:
    train_ids_set = set(pickle.load(f))
with open(SPLIT_DIR / "validation_patients.pkl", "rb") as f:
    val_ids_set = set(pickle.load(f))
with open(SPLIT_DIR / "test_patients_not7K.pkl", "rb") as f:
    test_ids_set = set(pickle.load(f))

assert len(train_ids_set) == 7369, f"train list has {len(train_ids_set)}"
assert len(val_ids_set) == 1050, f"val list has {len(val_ids_set)}"
assert len(test_ids_set) == 2100, f"internal test list has {len(test_ids_set)}"
assert not (train_ids_set & val_ids_set)
assert not (train_ids_set & test_ids_set)
assert not (val_ids_set & test_ids_set)
print(f"  train {len(train_ids_set):,} / val {len(val_ids_set):,} / "
      f"internal test {len(test_ids_set):,} — disjoint OK")

# ────────────────────────────────────────────────────────────────────────────
# Load the merged feather (the 7K testing feather is deliberately NOT loaded)
# ────────────────────────────────────────────────────────────────────────────

print("\n[2] Loading merged Dandelion feather from S3...")
s3_fs = s3fs.S3FileSystem()
merged_path = f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/df_fullz_merged_dandelion.feather"
print(f"  Loading {merged_path}")
with s3_fs.open(merged_path, "rb") as f:
    df_merged = pd.read_feather(f)
print(f"  Merged: {len(df_merged):,} records")

def fold_for(ecg_filename):
    if ecg_filename in train_ids_set:
        return 0
    if ecg_filename in val_ids_set:
        return 1
    if ecg_filename in test_ids_set:
        return 2
    return -1

df_merged["strat_fold"] = df_merged["ecg_filename"].map(fold_for)
unassigned = int((df_merged["strat_fold"] == -1).sum())
assert unassigned == 0, f"{unassigned} feather records not in any split list"
counts = df_merged["strat_fold"].value_counts().to_dict()
assert counts == {0: 7369, 1: 1050, 2: 2100}, f"unexpected split counts: {counts}"
for fold, name in [(0, "train"), (1, "val"), (2, "internal test")]:
    sub = df_merged[df_merged["strat_fold"] == fold]
    pos = int(sub["label"].sum())
    print(f"  {name}: {len(sub):,} records, label 1 = {pos:,} ({pos/len(sub):.1%})")

# ────────────────────────────────────────────────────────────────────────────
# Convert to NPZ arrays
# ────────────────────────────────────────────────────────────────────────────

print(f"\n[3] Converting {len(df_merged):,} records to NPZ format...")

ecg_signals_list = []
lengths_list = []
labels_list = []
ids_list = []
strat_fold_list = []
skipped = 0

for n, (idx, row) in enumerate(df_merged.iterrows(), start=1):
    try:
        ecg_leads = []
        for lead_col in LEAD_COLS:
            lead_data = row[lead_col]
            if isinstance(lead_data, np.ndarray):
                lead_array = lead_data.astype(np.float32)
            else:
                lead_array = np.array(lead_data, dtype=np.float32)
            ecg_leads.append(lead_array)

        ecg_signal = np.stack(ecg_leads, axis=1).astype(np.float32)
        actual_len = ecg_signal.shape[0]
        assert ecg_signal.ndim == 2 and ecg_signal.shape[1] == 12

        if actual_len < MAX_LEN:
            ecg_signal = np.pad(ecg_signal, ((0, MAX_LEN - actual_len), (0, 0)), mode='constant')
        elif actual_len > MAX_LEN:
            ecg_signal = ecg_signal[:MAX_LEN]

        binary_label = int(row["label"])
        one_hot = np.array([1, 0] if binary_label == 0 else [0, 1], dtype=np.int32)

        ecg_signals_list.append(ecg_signal)
        lengths_list.append(actual_len)
        labels_list.append(one_hot)
        ids_list.append(str(row["ecg_filename"]))
        strat_fold_list.append(int(row["strat_fold"]))

        if n % 2000 == 0:
            print(f"  Processed {n:,}/{len(df_merged):,}")
    except Exception as e:
        skipped += 1
        print(f"  ⚠️  Skipped record {idx}: {e}")

assert skipped == 0, f"{skipped} records failed conversion"

ecg_signals = np.stack(ecg_signals_list, axis=0)
lengths = np.array(lengths_list, dtype=np.int32)
labels = np.stack(labels_list, axis=0)
ids = np.array(ids_list, dtype=str)
strat_fold = np.array(strat_fold_list, dtype=np.int32)

print(f"  ✅ Converted {len(ecg_signals):,} records")
print(f"    ecg_signals: {ecg_signals.shape}")
print(f"    labels: {labels.shape}  ids: {ids.shape} (dtype {ids.dtype})")

# ────────────────────────────────────────────────────────────────────────────
# Save and upload
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
combined_npz = OUTPUT_DIR / "dandelion_split2_combined_signals.npz"
print(f"\n[4] Saving {combined_npz} ...")
np.savez(
    combined_npz,
    ecg_signals=ecg_signals,
    lengths=lengths,
    labels=labels,
    ids=ids,
    strat_fold=strat_fold,
)
print(f"  ✅ {combined_npz} ({combined_npz.stat().st_size / 1e9:.2f} GB)")

s3_out = f"s3://{S3_BUCKET}/{S3_OUT_PREFIX}/dandelion_split2_combined_signals.npz"
print(f"\n[5] Uploading to {s3_out} ...")
s3_fs.put(str(combined_npz), s3_out)
print("  ✅ Uploaded")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE (split 2)")
print("=" * 70)
