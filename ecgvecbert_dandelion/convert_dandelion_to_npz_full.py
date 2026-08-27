"""
Convert Dandelion feather files to NPZ format (100% of data).

Same as convert_dandelion_to_npz.py but uses FULL dataset (1.0 instead of 0.1).

Output structure:
  ecg_signals: (N, MAX_LEN, 12) float32 — padded signals
  lengths:    (N,) int32 — actual length per ECG
  labels:     (N, 1) int32 — binary label [0 or 1] for EF <= 40%
  ids:        (N,) str — ECG IDs

Usage:
  python convert_dandelion_to_npz_full.py
"""

import os
import numpy as np
import pandas as pd
import s3fs
from pathlib import Path
from sklearn.model_selection import train_test_split

# Settings
S3_BUCKET = "walkky-ml"
S3_DATA_PREFIX = "ecgvectbert/vqvae/bert_finetuning/datasets"
MAX_LEN = 5000  # Fixed length (like CS)
DANDELION_USE_FRAC = 1.0  # 100% of merged data (NOT 10%)
LEAD_COLS = ['lead_I', 'lead_II', 'lead_III',
             'lead_aVR', 'lead_aVL', 'lead_aVF',
             'lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6']

print("=" * 70)
print("DANDELION → NPZ CONVERSION (FULL DATASET 100%)")
print("=" * 70)

# Initialize S3
s3_fs = s3fs.S3FileSystem()

# Detect local vs S3 path
if os.path.exists("/opt/ml/"):
    print("🔧 SageMaker detected, using S3 paths")
    data_path = f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}"
else:
    print("💻 Local environment, using S3 paths")
    data_path = f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}"

# ────────────────────────────────────────────────────────────────────────────
# Load Dandelion data
# ────────────────────────────────────────────────────────────────────────────

print("\n[1] Loading Dandelion feathers from S3...")

# Merged data (ALL)
merged_path = os.path.join(data_path, "df_fullz_merged_dandelion.feather")
print(f"  Loading {merged_path}")
with s3_fs.open(merged_path, "rb") as f:
    df_merged = pd.read_feather(f)
print(f"  Merged (100%): {len(df_merged):,} records")

# Split into 80% train / 20% val WITH STRATIFICATION (preserve 78/22 ratio)
df_train, df_val = train_test_split(
    df_merged,
    test_size=0.2,
    stratify=df_merged['label'],
    random_state=42
)
print(f"  Train (80%, stratified): {len(df_train):,} records")
print(f"  Val (20%, stratified): {len(df_val):,} records")

# Test data (official, full)
test_path = os.path.join(data_path, "df_fullz_testing.feather")
print(f"  Loading {test_path}")
with s3_fs.open(test_path, "rb") as f:
    df_test = pd.read_feather(f)
print(f"  Test (official): {len(df_test):,} records")

# ────────────────────────────────────────────────────────────────────────────
# Convert to NPZ format
# ────────────────────────────────────────────────────────────────────────────

def dandelion_to_npz(df, split_name="train", strat_fold_value=0):
    """Convert Dandelion dataframe to NPZ arrays (CS format).

    Parameters
    ----------
    strat_fold_value : int
        0 for train split, 1 for test split
    """
    print(f"\n[2] Converting {split_name} to NPZ format ({len(df):,} records)...")

    ecg_signals_list = []
    lengths_list = []
    labels_list = []
    ids_list = []
    strat_fold_list = []

    for idx, row in df.iterrows():
        try:
            # Extract leads
            ecg_leads = []
            for lead_col in LEAD_COLS:
                if lead_col not in row.index:
                    raise ValueError(f"Missing {lead_col}")

                lead_data = row[lead_col]
                if isinstance(lead_data, list):
                    lead_array = np.array(lead_data, dtype=np.float32)
                elif isinstance(lead_data, np.ndarray):
                    lead_array = lead_data.astype(np.float32)
                else:
                    lead_array = np.array([lead_data], dtype=np.float32)

                ecg_leads.append(lead_array)

            # Stack to (L, 12)
            ecg_signal = np.stack(ecg_leads, axis=1).astype(np.float32)
            actual_len = ecg_signal.shape[0]

            # Verify shape
            assert ecg_signal.ndim == 2 and ecg_signal.shape[1] == 12, \
                f"Signal shape must be (L, 12), got {ecg_signal.shape}"

            # Pad or truncate to MAX_LEN
            if actual_len < MAX_LEN:
                ecg_signal = np.pad(ecg_signal, ((0, MAX_LEN - actual_len), (0, 0)), mode='constant')
            elif actual_len > MAX_LEN:
                ecg_signal = ecg_signal[:MAX_LEN]

            # Binary label: use label directly (already 0 or 1)
            binary_label = int(row["label"]) if "label" in row.index else 0

            # One-hot encode: [1, 0] for class 0, [0, 1] for class 1
            one_hot = np.array([1, 0] if binary_label == 0 else [0, 1], dtype=np.int32)

            # ID
            ecg_id = str(row["ecg_filename"]) if "ecg_filename" in row.index else f"dandelion_{idx}"

            ecg_signals_list.append(ecg_signal)
            lengths_list.append(actual_len)
            labels_list.append(one_hot)
            ids_list.append(ecg_id)
            strat_fold_list.append(strat_fold_value)

            if (idx + 1) % 2000 == 0:
                print(f"  Processed {idx + 1:,}/{len(df):,}")

        except Exception as e:
            print(f"  ⚠️  Skipped record {idx}: {e}")
            continue

    # Stack arrays
    ecg_signals = np.stack(ecg_signals_list, axis=0)  # (N, MAX_LEN, 12)
    lengths = np.array(lengths_list, dtype=np.int32)  # (N,)
    labels = np.stack(labels_list, axis=0)  # (N, 2) — one-hot encoded
    ids = np.array(ids_list, dtype='U26')  # (N,)
    strat_fold = np.array(strat_fold_list, dtype=np.int32)  # (N,)

    print(f"  ✅ Converted {len(ecg_signals):,} records")
    print(f"    ecg_signals: {ecg_signals.shape}")
    print(f"    lengths: {lengths.shape}")
    print(f"    labels: {labels.shape}")
    print(f"    ids: {ids.shape}")
    print(f"    strat_fold: {strat_fold.shape}")

    return ecg_signals, lengths, labels, ids, strat_fold

# Convert train (80% of merged)
train_signals, train_lengths, train_labels, train_ids, train_strat_fold = dandelion_to_npz(
    df_train, "train (80% of merged)", strat_fold_value=0
)

# Convert val (20% of merged)
val_signals, val_lengths, val_labels, val_ids, val_strat_fold = dandelion_to_npz(
    df_val, "val (20% of merged)", strat_fold_value=1
)

# Convert test (official 7K, never touched)
test_signals, test_lengths, test_labels, test_ids, test_strat_fold = dandelion_to_npz(
    df_test, "test (official)", strat_fold_value=2
)

# ────────────────────────────────────────────────────────────────────────────
# Save as NPZ
# ────────────────────────────────────────────────────────────────────────────

output_dir = Path("/tmp/dandelion_npz_full")
output_dir.mkdir(exist_ok=True)

print("\n[3] Saving NPZ files...")

# Stack all splits
all_signals = np.vstack([train_signals, val_signals, test_signals])
all_lengths = np.concatenate([train_lengths, val_lengths, test_lengths])
all_labels = np.vstack([train_labels, val_labels, test_labels])
all_ids = np.concatenate([train_ids, val_ids, test_ids])
all_strat_fold = np.concatenate([train_strat_fold, val_strat_fold, test_strat_fold])

combined_npz = output_dir / "dandelion_100p_combined_signals.npz"
np.savez(
    combined_npz,
    ecg_signals=all_signals,
    lengths=all_lengths,
    labels=all_labels,
    ids=all_ids,
    strat_fold=all_strat_fold,
)
print(f"  ✅ {combined_npz} ({combined_npz.stat().st_size / 1e9:.2f} GB)")
print(f"    Train: {len(train_signals):,}, Val: {len(val_signals):,}, Test: {len(test_signals):,}")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)
print(f"\nFiles saved to: {output_dir}/")
print(f"Ready for upload to S3!")
