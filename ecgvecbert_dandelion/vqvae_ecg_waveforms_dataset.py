"""
PyTorch Dataset / DataLoader for ECG waveforms on S3.
 
Reads h5 files, assigns each patient_id to exactly one of train / validation / test,
and exposes samples for a chosen segment type (e.g. QRS for VQ-VAE).
Excludes segments with zero original_len.
 
Design: FileShardedECGDataset
  - Each DDP rank loads a disjoint subset of h5 files fully into RAM at construction time.
    With 32 h5 files and 8 GPUs (ml.g5.48xlarge), each rank holds 4 files 
    Total ~ 100M segments × 1 lead × 176 floats × 4 bytes = ~70 GB total per lead. 
    With 8 GPUs each loading 4 files (~9 GB per GPU per lead). At SIGNAL_LEN=176, float32, 1 lead: N * 176 * 4 bytes per rank.
    Changed to float16 to reduce RAM requirement by half (54 GB total for 12 leads instead of 108 GB).
  - __getitem__ is a pure numpy index — zero S3 I/O during training.
  - Patient splits done by every rank on its own file shard (h5file_list[rank::world_size]);
    results are exchanged via all_gather_object in _get_or_create_patient_splits.
  - DistributedSampler is NOT used — data is already partitioned at the file level.
    Each rank independently shuffles its own shard each epoch.

Requirements: torch, h5py, hdf5plugin, boto3, numpy, tqdm

Exports used by vqvae_model.py:
  - FileShardedECGDataset
  - collate_batch
  - list_h5_files
  - collect_unique_patient_ids
  - split_patients
  - save_patient_splits
  - load_patient_splits
  - SplitName
  - SEGMENTS
  
Example
-------
    # On each rank:
    ds = FileShardedECGDataset(
        segment="QRS",
        h5file_list=h5file_list,
        patient_ids=splits["train"],
        lead_indices=[1],
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True,
                        num_workers=4, pin_memory=True, collate_fn=collate_batch)
    batch = next(iter(loader))
    # batch["waveform"]: (B, C, SIGNAL_LEN) float32
"""


import logging
from dataclasses import dataclass
from typing import Any, Literal
from collections import defaultdict
import json
import io

import boto3
import s3fs
import h5py
import hdf5plugin  # registers LZ4 filter for reading h5 files
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

SEGMENTS = ("P", "PQ", "QRS", "ST", "T")          # ("P", "QRS", "T")     # ("P", "PQ", "QRS", "ST", "T", "TP")  
NLEADS = 12    # Total number of leads in h5 files

bucket_in = "walkky-datasets"
prefix_in = "extracted-ecg-waveforms/mimic-iv/"
bucket_out = "walkky-ml"
prefix_out_base = "aruna-files/vqvae"

SplitName = Literal["train", "val", "test"]
s3 = boto3.client("s3")
s3_fs = s3fs.S3FileSystem()


# ---------------------------------------------------------------------------
# S3 file listing
# ---------------------------------------------------------------------------

def list_h5_files(bucket: str = bucket_in, prefix: str = prefix_in) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5"):
                files.append(key)
    return files


# ---------------------------------------------------------------------------
# Patient split helpers  (rank-0 computes, all ranks load)
# ---------------------------------------------------------------------------

def collect_unique_patient_ids(
    h5file_list: list[str],
    segment: str,
) -> set[str]:
    """
    Scan h5 files and return the set of all patient_ids for the given segment.
    Called by every rank on its own file shard (h5file_list[rank::world_size]);
    results are exchanged via all_gather_object in _get_or_create_patient_splits.
    """
    patients: set[str] = set()
    for filename in tqdm(h5file_list, desc="Collecting patient ids"):
        with s3_fs.open(f"s3://{bucket_in}/{filename}", "rb") as f:
            with h5py.File(f, "r") as h5file:
                if segment not in h5file:
                    continue
                n = int(h5file[segment]["patient_id"].shape[0])
                if n == 0:
                    continue
                pids = h5file[segment]["patient_id"][:]
                unique_raw = set(pids.tolist())
                patients.update(
                    p.decode("utf-8") if isinstance(p, bytes) else str(p)
                    for p in unique_raw
                )
    return patients


def split_patients(
    patient_ids: set[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[SplitName, set[str]]:
    """
    Randomly assign each patient_id to exactly one split. Ratios must sum to 1.0.
    Deterministic given the same seed.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}"
        )
    patients = np.array(sorted(set(patient_ids)), dtype=object)
    if len(patients) == 0:
        raise ValueError("No patient_ids found — cannot build splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = set(patients[:n_train])
    val_set = set(patients[n_train: n_train + n_val])
    test_set = set(patients[n_train + n_val:])

    assert len(train_set) + len(val_set) + len(test_set) == n
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    return {"train": train_set, "val": val_set, "test": test_set}


def save_patient_splits(
    splits: dict[SplitName, set[str]],
    vqvae_patient_splits_key: str,
) -> None:
    """Persist split assignments as JSON to S3 (called once by rank 0)."""
    payload = {k: sorted(v) for k, v in splits.items()}
    s3.put_object(
        Bucket=bucket_out,
        Key=vqvae_patient_splits_key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )
    log.info(f"Saved patient splits to s3://{bucket_out}/{vqvae_patient_splits_key}")


def load_patient_splits(
    vqvae_patient_splits_key: str,
) -> dict[SplitName, set[str]] | None:
    """Load split assignments from S3. Returns None if the key does not exist."""
    try:
        response = s3.get_object(Bucket=bucket_out, Key=vqvae_patient_splits_key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        return {k: set(v) for k, v in payload.items()}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise
    except Exception as e:
        log.error(f"Failed to load patient splits: {e}")
        return None


# ---------------------------------------------------------------------------
# Core dataset — one h5-file-set loaded fully into RAM per rank
# ---------------------------------------------------------------------------


class RawShardedECGData:
    """
    Memory-resident ECG segment dataset.
    Loads this rank's file shard ONCE, for ALL patients (no train/val/test
    filtering at load time). Train/val/test datasets are then built as cheap
    in-memory views over this object via FileShardedECGDataset below.

    Each DDP rank loads a disjoint subset of h5 files at construction time
    (files[rank::world_size]).
    
    Parameters
    ----------
    segment     : one of SEGMENTS ("P", "QRS", "T", …)
    h5file_list : ordered list of S3 keys (same order on every rank)
    lead_indices: which of the 12 leads to include (e.g. [1] for lead II)
    rank        : this process's DDP rank (0-based)
    world_size  : total number of DDP ranks

    Notes
    -----
    - Records are already shuffled in the h5 files, so file-level sharding
      produces balanced patient distributions across ranks.
    - Shard imbalance (max/min samples across ranks) is logged after loading.
      If it exceeds ~1.5x, consider offline re-sharding by patient_id.
    """

    def __init__(
        self,
        segment: str,
        signal_len: int,
        h5file_list: list[str],
        lead_indices: list[int],
        rank: int = 0,
        world_size: int = 1,
    ):
        if segment not in SEGMENTS:
            raise ValueError(f"segment must be one of {SEGMENTS}, got '{segment}'")

        self.segment = segment
        self.lead_indices = lead_indices

        my_files = h5file_list[rank::world_size]
        log.info(
            f"[rank {rank}/{world_size}] Loading {len(my_files)} / {len(h5file_list)} "
            f"h5 files for segment={segment} (all patients)"
        )

        arrays: list[np.ndarray] = []
        meta_accum: list[tuple[str, str, int, int]] = []
        for filename in tqdm(my_files, desc=f"[rank {rank}] {segment}"):
            result = self._load_file(filename, segment, signal_len, lead_indices)
            if result is not None:
                waveforms, meta = result
                arrays.append(waveforms)
                meta_accum.extend(meta)

        if arrays:
            self.data        = np.concatenate(arrays, axis=0)  # (N, num_leads, signal_len)
            self.patient_ids = np.array([m[0] for m in meta_accum], dtype=object)
            self.study_ids   = np.array([m[1] for m in meta_accum], dtype=object)
            self.beat_idxs   = np.array([m[2] for m in meta_accum], dtype=np.int32)
            self.orig_lens = np.array([m[3] for m in meta_accum], dtype=np.int32)
        else:
            self.data        = np.empty((0, len(lead_indices), signal_len), dtype=np.float16)
            self.patient_ids = np.empty(0, dtype=object)
            self.study_ids   = np.empty(0, dtype=object)
            self.beat_idxs   = np.empty(0, dtype=np.int32)
            self.orig_lens   = np.empty(0, dtype=np.int32)

        log.info(
            f"[rank {rank}] Loaded {len(self.data):,} segments (all patients)| "
            f"{self.data.nbytes / 1e9:.2f} GB RAM"
        )

    @staticmethod
    def _load_file(
        filename: str,
        segment: str,
        signal_len: int,
        lead_indices: list[int],
    ) -> tuple[np.ndarray, list[tuple[str, str, int, int]]] | None:
        """
        Read one h5 file into RAM.

        Reads ALL waveform data in a single S3 request (grp["data"][:]),
        then filters and assembles beat arrays in memory.

        Returns
        -------
        (waveforms, meta) where
            waveforms : np.ndarray (N_beats, num_leads, signal_len)
            meta      : list of (patient_id, study_id, beat_idx, orig_lens) tuples, len N_beats
        or None if the file yields no valid beats.
        """
        with s3_fs.open(f"s3://{bucket_in}/{filename}", "rb") as f:
            with h5py.File(f, "r") as h5:
                if segment not in h5:
                    return None
                grp = h5[segment]
                n = grp["data"].shape[0]
                if n == 0:
                    return None

                # --- bulk metadata reads
                pids = np.array(
                    [p.decode() if isinstance(p, bytes) else p
                     for p in grp["patient_id"][:]]
                )
                orig_len = grp["original_len"][:]
                lead_idx_arr = grp["lead_idx"][:]
                beat_idx_arr = grp["beat_idx"][:]
                study_ids = np.array(
                    [s.decode() if isinstance(s, bytes) else s
                     for s in grp["study_id"][:]]
                )

                # --- bulk waveform read, truncated to signal_len
                # Using float16 due to RAM limitation for 12 leads data
                all_data = grp["data"][:].astype(np.float16)[:, :signal_len]

        # All h5 access done; pure numpy from here.

        # Filter: drop zero-length segments 
        valid_mask = (orig_len > 0) 
        valid_rows = np.where(valid_mask)[0]   # unpack tuple

        if len(valid_rows) == 0:
            return None

        # Group valid rows by (patient_id, study_id, beat_idx)
        beat_to_rows: dict[tuple, dict[int, int]] = defaultdict(dict)
        for row in valid_rows:
            key = (pids[row], study_ids[row], int(beat_idx_arr[row]))
            beat_to_rows[key][int(lead_idx_arr[row])] = int(row)

        # Assemble per-beat waveform arrays; skip incomplete beats
        beat_arrays: list[np.ndarray] = []
        meta_list: list[tuple[str, str, int, int]] = []
        skipped = 0
        for key, lead_map in beat_to_rows.items():
            if len(lead_map) != NLEADS:
                skipped += 1
                continue
            pid, sid, bidx = key
            waveform = np.stack(
                [all_data[lead_map[li]] for li in lead_indices], axis=0
            )  # (num_leads, signal_len)
            orig_len_waveform = orig_len[lead_map[0]]   # same for all leads
            beat_arrays.append(waveform)
            meta_list.append((str(pid), str(sid), int(bidx), int(orig_len_waveform)))

        if skipped:
            log.warning(
                f"{filename}: skipped {skipped} beats with incomplete leads"
            )

        if not beat_arrays:
            return None

        return np.stack(beat_arrays, axis=0), meta_list  # (N_beats, num_leads, signal_len)


class FileShardedECGDataset(Dataset):
    """
    Thin, zero-I/O view over a RawShardedECGData object, filtered to one
    split's patient_ids. Construction is just a boolean mask + np.where —
    no S3 access, no h5py, no re-reading files.
    """
    # ------------------------------------------------------------------
    # Dataset contract
    # ------------------------------------------------------------------

    def __init__(self, raw: RawShardedECGData, patient_ids: set[str]):
        mask = np.isin(raw.patient_ids, np.fromiter(patient_ids, dtype=object, count=len(patient_ids)))
        self._indices = np.where(mask)[0]
        self._raw = raw

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = self._indices[idx]
        return {
            "waveform":   torch.from_numpy(self._raw.data[i]).float(),        # becomes float32
            "patient_id": str(self._raw.patient_ids[i]),
            "study_id":   str(self._raw.study_ids[i]),
            "beat_idx":   int(self._raw.beat_idxs[i]),
            "orig_len":   int(self._raw.orig_lens[i]),
        }


class InMemoryECGDataset(Dataset):
    """
    Plain in-memory dataset over already-loaded arrays — no S3, no h5py.
    Used to assemble the gathered test set on rank 0 from other ranks'
    already-loaded raw shards, instead of re-reading files from S3.
    """

    def __init__(self, data, patient_ids, study_ids, beat_idxs, orig_lens):
        self.data = data
        self.patient_ids = patient_ids
        self.study_ids = study_ids
        self.beat_idxs = beat_idxs
        self.orig_lens = orig_lens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "waveform":   torch.from_numpy(self.data[idx]).float(),      # becomes float32
            "patient_id": str(self.patient_ids[idx]),
            "study_id":   str(self.study_ids[idx]),
            "beat_idx":   int(self.beat_idxs[idx]),
            "orig_len":   int(self.orig_lens[idx]),
        }

        
# ---------------------------------------------------------------------------
# Collate function  (imported by vqvae_model.py)
# ---------------------------------------------------------------------------

def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack waveform tensors and collect metadata into a batch."""
    return {
        "waveform":   torch.stack([s["waveform"] for s in samples], dim=0),  # (B, C, L)
        "patient_id": [s["patient_id"] for s in samples],                    # list[str] len B
        "study_id":   [s["study_id"]   for s in samples],                    # list[str] len B
        "beat_idx":   torch.tensor([s["beat_idx"] for s in samples],
                                   dtype=torch.long),                         # (B,)
        "orig_len":   torch.tensor([s["orig_len"] for s in samples],
                                   dtype=torch.long),                         # (B,)
    }


# ---------------------------------------------------------------------------
# Convenience factory  (kept for smoke-testing; training uses FileShardedECGDataset directly)
# ---------------------------------------------------------------------------

@dataclass
class ECGDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    splits: dict[SplitName, set[str]]
    datasets: dict[SplitName, FileShardedECGDataset]
    h5file_list: list

    def __getitem__(self, key: str) -> DataLoader:
        return getattr(self, key)


def create_ecg_dataloaders(
    h5file_list: list[str],
    lead_indices: list[int],
    segment: str,
    signal_len: int,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    rank: int = 0,
    world_size: int = 1,
) -> ECGDataLoaders:
    """
    Convenience wrapper for single-process smoke tests and notebooks.
    DDP training should call FileShardedECGDataset directly (see vqvae_model.py).
    """
    # Load this rank's file shard ONCE, for all patients
    # This gives us every patient_id already in RAM, so there's no need
    # for a separate collect_unique_patient_ids() S3 scan beforehand.
    raw_shard = RawShardedECGData(
        segment=segment,
        signal_len=signal_len,
        h5file_list=h5file_list,
        lead_indices=lead_indices,
        rank=rank,
        world_size=world_size,
    )

    patient_ids = set(raw_shard.patient_ids.tolist())
    splits = split_patients(
        patient_ids=patient_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    log.info(
        f"Patient splits: train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    datasets: dict[SplitName, FileShardedECGDataset] = {}
    loaders: dict[SplitName, DataLoader] = {}

    for split_name, split_patient_set in splits.items():
        ds = FileShardedECGDataset(
            raw_shard,
            patient_ids=split_patient_set,
        )
        datasets[split_name] = ds
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=collate_batch,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context='forkserver',
        )

    return ECGDataLoaders(
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        splits=splits,
        datasets=datasets,
        h5file_list=h5file_list,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Smoke-test ECG DataLoaders")
    p.add_argument("--segment", default="P", choices=SEGMENTS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_files", type=int, default=1,
                   help="Limit to first N h5 files for fast testing")
    args = p.parse_args()

    h5filelist = list_h5_files()[:args.num_files]
    log.info(f"Testing with {len(h5filelist)} file(s)")

    # For testing only, otherwise passed as params
    SIGNAL_LEN = 176
    LEAD_INDICES = [1]  # Lead II by default

    bundle = create_ecg_dataloaders(
        h5file_list=h5filelist,
        lead_indices=LEAD_INDICES,
        segment=args.segment,
        signal_len=SIGNAL_LEN,
        batch_size=args.batch_size,
    )

    batch = next(iter(bundle.train))
    print(f"waveform shape : {batch['waveform'].shape}")
    print(f"waveform example: {batch['waveform'][0]}")
    print(f"train size     : {len(bundle.datasets['train']):,}")
    print(f"val size       : {len(bundle.datasets['val']):,}")
    print(f"test size      : {len(bundle.datasets['test']):,}")