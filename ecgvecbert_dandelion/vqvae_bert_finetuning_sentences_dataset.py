import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import boto3
import s3fs
import io
import pickle
import os
import random
from sklearn.utils import shuffle
import neurokit2 as nk
import logging
from typing import Any, Literal
from collections import defaultdict
import hashlib
from sklearn.model_selection import train_test_split

from ecg_extraction import (SAMPLING_RATE, LEAD_II_IDX, NUM_LEADS, 
                            ECG_SEGMENT_KEYS, MIN_R_PEAKS, 
                            _extract_all_waveforms)

from encoder_features_ecg_waveforms import vqvae_encoder_embedding
from rocket_features_ecg_waveforms import ROCKETFeatureExtractor
from vqvae_model import Model, ECGResidualEncoder, ECGResidualDecoder, VectorQuantizerEMA, VectorQuantizer, ResidualStack, Residual

from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from vqvae_bert_sentence_dataset import (
    SIGNAL_LENS,
    _embed_batch,
    get_or_create_rocket_extractor,
    SENTENCE_LENGTHS,
    TOKENS_PER_BEAT,
    MAX_BEATS,
    MAX_SENTENCE_LEN,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    FIRST_REAL_TOKEN_ID,
    _TOKEN_TYPE_TEMPLATE,
)

from vqvae_ecg_waveforms_dataset import (
    SEGMENTS,
    NLEADS,
    bucket_out,
)

NUM_GENERATION_SHARDS = 4

DATASETS = ["dandelion"]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Output files from fine-tuning data prep
prefix_out = "ecgvectbert/vqvae/bert_finetuning"
SplitName = Literal["train", "val", "test"]
Splits = dict[SplitName, tuple[list, list, list]]

# VQ-VAE model location
model_prefix = "aruna-files/vqvae"

# Initialize the S3 client
s3_client = boto3.client('s3')
s3_fs = s3fs.S3FileSystem()


# Get dataset filekey in s3
def _signals_npz_key(dataset_name: str) -> str:
    return f"{prefix_out}/{dataset_name}_signals.npz"


# Read datasets from s3 
def read_ecg_signals_from_s3(dataset_name: str) -> dict:
    """
    Returns
    -------
    dict with:
        signals    : list[np.ndarray]  — one (length_i, 12) array per record,
                                          padding trimmed off
        labels     : list[np.ndarray] - one one-hot-encoding per record
        ids        : (N,) array
        strat_fold : (N,) int array
    """
    s3_path = f"s3://{bucket_out}/{_signals_npz_key(dataset_name)}"
    with s3_fs.open(s3_path, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)

        padded  = data["ecg_signals"]   # (N, max_len, n_leads)
        lengths = data["lengths"]       # (N,)
        signals = [padded[i, :lengths[i], :] for i in range(len(padded))]

        result = {
            "signals": signals,
            "labels":  data["labels"],
            "ids":     data["ids"].astype(str),
        }
        if "strat_fold" in data:
            result["strat_fold"] = data["strat_fold"]
        if "ptbxl" in dataset_name:
            # overwrite patient ids with record index, as we need only record id here
            result["ids"] = np.arange(len(padded)).astype(str)

    log.info(f"[{dataset_name}] loaded {len(signals):,} signals ← {s3_path}")
    return result


def read_dandelion_data(use_frac=1.0, data_path=None):
    """
    Read Dandelion ECG + LVEF labels from feather files.

    Returns
    -------
    dict with:
        signals    : list[np.ndarray]  — one (length_i, 12) array per record
        labels     : list[np.ndarray] — binary LVSD labels (0/1)
        ids        : (N,) array — record identifiers
        strat_fold : (N,) int array — split indicators (0=train, 1=test)
    """
    if data_path is None:
        is_sagemaker = os.path.exists("/opt/ml/")
        if is_sagemaker:
            data_path = "s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/datasets"
        else:
            data_path = "/Users/burcuozek/Desktop/dandelion data"

    LEAD_COLS = ['lead_I', 'lead_II', 'lead_III',
                 'lead_aVR', 'lead_aVL', 'lead_aVF',
                 'lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6']

    signals = []
    labels = []
    ids = []
    strat_fold = []

    try:
        train_path = os.path.join(data_path, "df_fullz_merged_dandelion.feather")
        log.info(f"Loading training feathers from {train_path}")
        if train_path.startswith("s3://"):
            with s3_fs.open(train_path, "rb") as f:
                df_train = pd.read_feather(f)
        else:
            df_train = pd.read_feather(train_path)

        test_path = os.path.join(data_path, "df_fullz_testing.feather")
        log.info(f"Loading test feathers from {test_path}")
        if test_path.startswith("s3://"):
            with s3_fs.open(test_path, "rb") as f:
                df_test = pd.read_feather(f)
        else:
            df_test = pd.read_feather(test_path)

        df_train['__split__'] = 0
        df_test['__split__'] = 1
        df_all = pd.concat([df_train, df_test], ignore_index=False).reset_index(drop=True)

        for idx, row in df_all.iterrows():
            try:
                ecg_leads = []
                for lead_col in LEAD_COLS:
                    if lead_col in row.index:
                        lead_data = row[lead_col]
                        if isinstance(lead_data, list):
                            ecg_leads.append(np.array(lead_data, dtype=np.float32))
                        elif isinstance(lead_data, np.ndarray):
                            ecg_leads.append(lead_data.astype(np.float32))
                        else:
                            ecg_leads.append(np.array([lead_data], dtype=np.float32))
                    else:
                        raise ValueError(f"Missing lead column: {lead_col}")

                ecg_array = np.array(ecg_leads, dtype=np.float32)
                signal = ecg_array.T

                label = int(row["label"]) if "label" in row.index else 0
                ecg_id = str(row["ecg_filename"]) if "ecg_filename" in row.index else f"dandelion_{idx}"
                split = int(row["__split__"])

                signals.append(signal)
                labels.append(np.array([label], dtype=np.float32))
                ids.append(ecg_id)
                strat_fold.append(split)

            except Exception as e:
                log.warning(f"Failed to process Dandelion record {idx}: {e}")
                continue

        log.info(f"[dandelion] loaded {len(signals):,} signals from {data_path}")

        return {
            "signals": signals,
            "labels": labels,
            "ids": np.array(ids, dtype=str),
            "strat_fold": np.array(strat_fold, dtype=int),
        }

    except Exception as e:
        log.error(f"Error reading Dandelion data from {data_path}: {e}")
        raise


# Return train, val and test splits of ecg and labels for dataset_name specified
def read_ecg_data(dataset_name, use_frac, train_frac=0.7, val_frac=0.1, test_frac=0.2, min_class_count=10, seed=0) -> Splits:

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Fractions must sum to 1")

    if dataset_name == "dandelion":
        store = read_dandelion_data(use_frac=use_frac)
        signals = store["signals"]
        labels = store["labels"]
        ids = np.asarray(store["ids"])
        strat_fold = store["strat_fold"]

        N = len(signals)
        all_idx = np.arange(N)

        train_idx = all_idx[strat_fold == 0]
        test_idx = all_idx[strat_fold == 1]
        val_idx = np.array([], dtype=int)

        log.info(f"[dandelion] using pre-defined splits:")
        log.info(f"  train: {len(train_idx):,} records")
        log.info(f"  val:   {len(val_idx):,} records (not used)")
        log.info(f"  test:  {len(test_idx):,} records")

        return {
            "train": (signals, labels, ids[train_idx]),
            "val": ([], [], []),
            "test": (signals, labels, ids[test_idx]),
        }

    store = read_ecg_signals_from_s3(dataset_name)
    signals = store["signals"]          # list[(length_i, 12)]
    labels  = store["labels"]           # list[(num_classes,)]
    ids     = np.asarray(store["ids"])

    N = len(signals)
    all_idx = np.arange(N)

    if "ptbxl" in dataset_name:
        # Use official split 80/10/10
        strat_fold = store["strat_fold"]
        train_idx = all_idx[strat_fold <= 8]
        val_idx = all_idx[strat_fold == 9]
        test_idx = all_idx[strat_fold == 10]
    elif dataset_name=="cpsc2018" or dataset_name=="csn":
        # Use MERL split used in data prep
        strat_fold = store["strat_fold"]
        train_idx = all_idx[strat_fold == 0]
        val_idx = all_idx[strat_fold == 1]
        test_idx = all_idx[strat_fold == 2]
    else:
        # CS: single-labeled records; do stratified splits by numeric label
        singlelabels = np.argmax(labels, axis=1)
        # if number of class labels is less than min_class_count, routed to train
        # Fixing the seed produces the same split each time
        # Stratified split by numeric label: each record id has exactly one
        # numeric label, train_test_split keeps label
        # proportions consistent across train/val/test.
        df = pd.DataFrame({'id': ids, 'label_idx': singlelabels})
        id_labels = df.drop_duplicates(subset='id').set_index('id')['label_idx']
        unique_ids = id_labels.index.to_numpy()
        unique_labels = id_labels.to_numpy()

        class_counts = pd.Series(unique_labels).value_counts()
        log.info(f"[{dataset_name}] unique-id class counts:\n{class_counts}")

        rare_classes = class_counts[class_counts < min_class_count].index.tolist()
        common_mask = ~np.isin(unique_labels, rare_classes)

        common_ids, common_labels = unique_ids[common_mask], unique_labels[common_mask]
        rare_ids = unique_ids[~common_mask]

        id_to_split: dict = {}

        # --- Rare classes: route entirely to train, never appear in val/test ---
        if len(rare_classes):
            log.info(
                f"[{dataset_name}] Routing rare classes entirely to Train "
                f"(< {min_class_count} records, excluded from val/test): "
                f"{[(c, int(class_counts[c])) for c in rare_classes]}"
            )
            for uid in rare_ids:
                id_to_split[uid] = 'train'

        # --- Common classes: stratified split  ---
        if len(common_ids):
            # First split off train vs (val+test), stratified by label
            train_ids, temp_ids, train_labels, temp_labels = train_test_split(
                common_ids, common_labels,
                train_size=train_frac,
                stratify=common_labels,
                random_state=seed,
            )

            # Then split (val+test) into val vs test, stratified by label
            relative_val_frac = val_frac / (val_frac + test_frac)
            val_ids, test_ids = train_test_split(
                temp_ids,
                train_size=relative_val_frac,
                stratify=temp_labels,
                random_state=seed,
            )

            temp_labels_arr = np.asarray(temp_labels)
            val_mask  = np.isin(temp_ids, val_ids)
            test_mask = np.isin(temp_ids, test_ids)
            temp_classes = set(pd.unique(temp_labels_arr))
            missing_val  = temp_classes - set(pd.unique(temp_labels_arr[val_mask]))
            missing_test = temp_classes - set(pd.unique(temp_labels_arr[test_mask]))
            if missing_val or missing_test:
               log.info(
                   f"[{dataset_name}] Stratified split left classes with zero "
                   f"records — missing from val: {missing_val or 'none'}, "
                   f"missing from test: {missing_test or 'none'}."
                )

            for uid in train_ids:
                id_to_split[uid] = 'train'
            for uid in val_ids:
                id_to_split[uid] = 'val'
            for uid in test_ids:
                id_to_split[uid] = 'test'
                
        split_arr = np.array([id_to_split[i] for i in ids])
        train_idx = all_idx[split_arr == 'train']
        val_idx   = all_idx[split_arr == 'val']
        test_idx  = all_idx[split_arr == 'test']

    def _select(idx):
        X = [signals[i] for i in idx]
        y = [labels[i] for i in idx]
        id_sel = ids[idx]        # record id, one record per patient except for ptbxl
        return X, y, id_sel

    X_train, y_train, id_train = _select(train_idx)
    X_val, y_val, id_val     = _select(val_idx)
    X_test, y_test , id_test  = _select(test_idx)

    X_train, y_train, id_train = shuffle(X_train, y_train, id_train, random_state=seed)
    X_val, y_val, id_val     = shuffle(X_val, y_val, id_val, random_state=seed)
    X_test, y_test, id_test   = shuffle(X_test, y_test, id_test, random_state=seed)

    nfrac_train = int(len(X_train) * use_frac)
    # The entire val and test set should be used, and use_frac should be applied only to train

    return {
        'train': (id_train[:nfrac_train], X_train[:nfrac_train], y_train[:nfrac_train]),
        'val':   (id_val, X_val, y_val),
        'test':  (id_test, X_test, y_test),
    }


# Extract ecg segments
def ecg_extract(ecg_data, sampling_rate=SAMPLING_RATE):

    try:
        ecg_data = np.array(ecg_data)
        clean_ecg_data = np.stack(
            [nk.ecg_clean(ecg_data[:, i], sampling_rate=sampling_rate)
             for i in range(NUM_LEADS)],
            axis=1,
        )
        clean_ref_lead = clean_ecg_data[:, LEAD_II_IDX]
    except Exception as e:
        log.warning(e)
        return None
 
    # ── R-peaks ───────────────────────────────────────────────────────────────
    try:
        _, rpeaks_info = nk.ecg_peaks(clean_ref_lead, sampling_rate=sampling_rate)
    except Exception as e:
        log.warning(e)
        return None
 
    r_peaks = rpeaks_info["ECG_R_Peaks"]
    if len(r_peaks) < MIN_R_PEAKS:
        log.warning("too few R peaks")
        return None
 
    # ── Delineation ───────────────────────────────────────────────────────────
    try:
        _, waves_info = nk.ecg_delineate(
            clean_ref_lead, rpeaks_info,
            sampling_rate=sampling_rate, method="dwt", show=False,
        )
    except Exception as e:
        log.warning(e)
        return None
 
    # Store as float64 so np.isnan() works correctly on NaN landmarks.
    indices: dict[str, np.ndarray] = {
        key: np.array(waves_info[nk2_key], dtype=float)
        for key, nk2_key in ECG_SEGMENT_KEYS.items()
    }
 
    # ── Extract waveforms ─────────────────────────────────────────────────────
    beat_cycles, _ = _extract_all_waveforms(
        clean_ecg_data, indices
    )

    return beat_cycles


def iter_finetuning_ecg_beats(ids: list, X: list, y: list):
    """
    Iterate over ECG signals and yield extracted beat cycles with their labels.

    This replaces the DataLoader used in assign_codebook_indices: instead of
    reading pre-extracted waveforms from H5 files, it runs ecg_extract on each
    raw ECG signal and yields the resulting beat cycles together with the
    ECG-level label.

    Yields
    ------
    beat_cycles : list[dict]  — output of ecg_extract; each dict has keys
                               "beat_idx", "segments", "original_lengths"
    ecg_idx     : str        — positional index into X/y identifying the source ECG
    label                     — label from y for this ECG
    """
    for ecg_idx, ecg_raw, label in zip(ids, X, y):
        try:
            beat_cycles = ecg_extract(np.array(ecg_raw))
        except Exception as e:
            log.warning(f"ecg_extract failed for ecg_idx={ecg_idx}: {e}")
            continue
        if not beat_cycles:
            log.warning(f"No beat cycles extracted for ecg_idx={ecg_idx}")
            continue
        yield beat_cycles, ecg_idx, label


def _load_model_from_s3(model_key: str) -> torch.nn.Module:
    """Load a pickled VQ-VAE model from S3."""
    log.info(f"Loading model from s3://{bucket_out}/{model_key}")
    response = s3_client.get_object(Bucket=bucket_out, Key=model_key)
    model = pickle.loads(response["Body"].read())
    model.eval()
    return model


@torch.no_grad()
def assign_finetuning_codebook_indices(
    models: dict[str, torch.nn.Module],
    device: torch.device,
    splits: Splits,
    in_channels: int,
    cnn_embed_type: str | None = None, 
    cnn_embed_scale: float = 1.0,
    rocket_extractors: dict[str, ROCKETFeatureExtractor] | None = None,
    batch_size: int = 256,
) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Assign VQ-VAE codebook indices for ALL segment types in a single pass.

    Runs ecg_extract exactly once per ECG (instead of once per segment) and
    routes each beat's P/QRS/T waveforms into per-segment buffers, flushing
    each buffer through its own VQ-VAE model once it reaches batch_size.

    Parameters
    ----------
    models    : {segment: trained VQ-VAE model}, each with encode_to_indices(waveform)
    device    : torch device to run all models on
    splits    : pre-computed patient splits from read_ecg_data
    cnn_embed_type
    cnn_embed_scale
    rocket_extractors
    batch_size: beat waveforms per GPU forward pass, per segment

    Returns
    -------
    (assignments, codebook_sizes)
        assignments[seg][split_name] = {
            "beat_idxs", "ecg_idxs", "token_indices", "labels",
            "embeddings" (optional)
        }
        codebook_sizes[seg] = int
    """
    for model in models.values():
        model.to(device).eval()

    codebook_sizes = {seg: get_codebook_size(models[seg]) for seg in SEGMENTS}

    assignments: dict[str, dict] = {seg: {} for seg in SEGMENTS}

    if in_channels == 1:
        lead_indices = [1]         # Lead II
    elif in_channels == NLEADS:
        lead_indices = list(range(NLEADS))      # All leads
    else:
        raise ValueError("in_channels does not match allowed leads")

    for split_name, (ids, X, y) in splits.items():
        log.info(f"Processing {split_name} split ({len(X):,} ECGs) for segments {SEGMENTS}...")

        bufs: dict[str, list] = {seg: [] for seg in SEGMENTS}
        results: dict[str, dict] = {
            seg: {
                "beat_idxs": [], "ecg_idxs": [], "token_indices": [], "labels": [],
                "embeddings": [] if cnn_embed_type is not None else None,
            }
            for seg in SEGMENTS
        }

        def _flush(seg: str) -> None:
            buf = bufs[seg]
            if not buf:
                return
            waveforms = torch.stack([item[0] for item in buf]).to(device)
            indices = models[seg].encode_to_indices(waveforms).cpu().numpy().astype(np.int16)
            result = results[seg]
            if cnn_embed_type is not None:
                result["embeddings"].append(_embed_batch(cnn_embed_type, cnn_embed_scale, models[seg], device, waveforms, rocket_extractors[seg]))
            for k, (_, beat_idx, ecg_idx, label) in enumerate(buf):
                result["beat_idxs"].append(beat_idx)
                result["ecg_idxs"].append(ecg_idx)
                result["token_indices"].append(int(indices[k]))
                result["labels"].append(label)
            buf.clear()

        # Single delineation pass per ECG — fan out into per-segment buffers.
        for beat_cycles, ecg_idx, label in iter_finetuning_ecg_beats(ids, X, y):
            for cycle in beat_cycles:
                for seg in SEGMENTS:
                    if seg not in cycle["segments"]:
                        continue
                    seg_data = cycle["segments"][seg]
                    signal_len = SIGNAL_LENS[seg]
                    waveform_np = np.stack(
                        [seg_data[li][:signal_len] for li in lead_indices],
                        axis=0,
                    ).astype(np.float32)   #### mismatch with VQ-VAE training
                    # .astype(np.float16).astype(np.float32)
                    # quantize to float16 precision (matches VQ-VAE/BERT pretraining input), then upcast — model expects float32
                    bufs[seg].append((torch.from_numpy(waveform_np), cycle["beat_idx"], ecg_idx, label))
                    if len(bufs[seg]) >= batch_size:
                        _flush(seg)

        for seg in SEGMENTS:
            _flush(seg)

        for seg in SEGMENTS:
            result = results[seg]
            split_out = {
                "beat_idxs":     np.array(result["beat_idxs"],     dtype=np.int32),
                "ecg_idxs":      np.array(result["ecg_idxs"],      dtype=np.str_),
                "token_indices": np.array(result["token_indices"], dtype=np.int16),
                "labels":        np.array(result["labels"], dtype=np.float32),
            }
            if result["embeddings"]:
                split_out["embeddings"] = np.concatenate(result["embeddings"])
            n = len(split_out["beat_idxs"])
            log.info(f"  [{seg}] {split_name}: {n:,} beats encoded across {len(X):,} ECGs")
            assignments[seg][split_name] = split_out

    return assignments, codebook_sizes


def build_finetuning_beat_sentences(
    assignments: dict[str, dict],
    codebook_sizes: dict[str, int],
    seed: int = 42,
) -> dict:
    """
    Build variable-length beat sentences from fine-tuning codebook assignments.

    Mirrors build_beat_sentences from vqvae_bert_sentence_dataset but uses
    (ecg_idx, beat_idx) as the join key instead of (patient_id, study_id, beat_idx),
    and carries the ECG-level label per sentence.

    Token IDs are shifted to the same BERT vocab positions as in pre-training:
        P   : FIRST_REAL_TOKEN_ID  ..  FIRST_REAL_TOKEN_ID + num_P - 1
        QRS : P_end + 1            ..  P_end + num_QRS
        T   : QRS_end + 1          ..  QRS_end + num_T
        SEP : SEP_TOKEN_ID  (1)
        PAD : PAD_TOKEN_ID  (0)
        or similar for P, PQ, QRS, ST, T

    Parameters
    ----------
    assignments    : dict[segment → dict]  — output of assign_finetuning_codebook_indices
                     for a single split.  Each segment dict must contain arrays:
                     "beat_idxs", "ecg_idxs", "token_indices", "labels".
                     Optionally: "embeddings" (N, D) float16.
    codebook_sizes : {segment: int}  — VQ-VAE codebook size per segment
    seed           : RNG seed for sentence-length sampling

    Returns
    -------
    dict with one row per sentence:
        sentence_tokens    : (N, MAX_SENTENCE_LEN) int32
        labels             : (N,num_classes)     — ECG-level one hot encoding
        ecg_idxs           : (N,) str            — source ECG index
        num_beats          : (N,) int8
        beat_idx_start     : (N,) int32
        beat_idx_end       : (N,) int32
        segment_order      : list[str]
        vocab_size         : int
        vocab_offsets      : dict[str, int]
        cnn_embeddings : (N, MAX_SENTENCE_LEN, D) float16  — only when present
    """
    vocab_offsets: dict[str, int] = {SEGMENTS[0]: FIRST_REAL_TOKEN_ID}
    for i in range(1, len(SEGMENTS)):
        vocab_offsets[SEGMENTS[i]] = vocab_offsets[SEGMENTS[i - 1]] + codebook_sizes[SEGMENTS[i - 1]]
    vocab_size = vocab_offsets[SEGMENTS[-1]] + codebook_sizes[SEGMENTS[-1]]
    log.info(f"Fine-tuning vocab_size={vocab_size}, offsets={vocab_offsets}")

    rng = np.random.default_rng(seed)
    has_embeddings = all("embeddings" in data for data in assignments.values())
    embedding_dim = assignments[SEGMENTS[0]]["embeddings"].shape[1] if has_embeddings else 0

    # Index every beat by (ecg_idx, beat_idx); record label per ecg_idx
    indexed: dict[tuple, dict] = {}
    ecg_label: dict[str, Any] = {}

    for seg, data in assignments.items():
        embs = data.get("embeddings")
        for i, (ecg_idx, bidx, tok, label) in enumerate(zip(
            data["ecg_idxs"], data["beat_idxs"],
            data["token_indices"], data["labels"],
        )):
            key = (str(ecg_idx), int(bidx))
            indexed.setdefault(key, {})[seg] = int(tok)
            if embs is not None:
                indexed[key][f"{seg}_emb"] = embs[i]
            ecg_label[str(ecg_idx)] = label

    full_keys = [k for k, v in indexed.items() if all(s in v for s in SEGMENTS)]
    log.info(
        f"Beats with all segments: {len(full_keys):,} "
        f"(dropped {len(indexed) - len(full_keys):,} incomplete beats)"
    )

    # Group sorted beat indices per ECG
    ecg_beats: dict[str, list[int]] = defaultdict(list)
    for (ecg_idx, bidx) in full_keys:
        ecg_beats[ecg_idx].append(bidx)
    for ecg_idx in ecg_beats:
        ecg_beats[ecg_idx].sort()

    sentence_tokens_list: list[list[int]] = []
    cnn_list:         list[np.ndarray] = []
    labels_list:          list = []
    ecg_idx_list:         list[str] = []
    num_beats_list:       list[int] = []
    beat_idx_start_list:  list[int] = []
    beat_idx_end_list:    list[int] = []

    for ecg_idx, beat_seq in ecg_beats.items():
        label = ecg_label[ecg_idx]
        pos = 0
        while pos < len(beat_seq):
            remaining = len(beat_seq) - pos
            valid = [l for l in SENTENCE_LENGTHS if l <= remaining]
            if not valid:
                break
            n = int(rng.choice(valid))
            window = beat_seq[pos: pos + n]
            tokens: list[int] = []
            if has_embeddings:
                emb_row = np.zeros((MAX_SENTENCE_LEN, embedding_dim), dtype=np.float16)
            for beat_pos, bidx in enumerate(window):
                v = indexed[(ecg_idx, bidx)]
                for seg_id, seg in enumerate(SEGMENTS):
                    tokens.append(v[seg] + vocab_offsets[seg])
                    if has_embeddings:
                        emb_row[beat_pos * TOKENS_PER_BEAT + seg_id] = v[f"{seg}_emb"]
            tokens += [SEP_TOKEN_ID]
            tokens += [PAD_TOKEN_ID] * (MAX_SENTENCE_LEN - len(tokens))
            sentence_tokens_list.append(tokens)
            if has_embeddings:
                cnn_list.append(emb_row)
            labels_list.append(label)
            ecg_idx_list.append(ecg_idx)
            num_beats_list.append(n)
            beat_idx_start_list.append(window[0])
            beat_idx_end_list.append(window[-1])
            pos += n

    N = len(sentence_tokens_list)
    log.info(f"Built {N:,} fine-tuning sentences from {len(ecg_beats):,} ECGs")

    result = {
        "sentence_tokens": np.array(sentence_tokens_list, dtype=np.int32),
        "labels":          np.array(labels_list, dtype=np.float32),
        "ecg_idxs":        np.array(ecg_idx_list,        dtype=np.str_),
        "num_beats":       np.array(num_beats_list,       dtype=np.int8),
        "beat_idx_start":  np.array(beat_idx_start_list,  dtype=np.int32),
        "beat_idx_end":    np.array(beat_idx_end_list,    dtype=np.int32),
        "segment_order":   list(SEGMENTS),
        "vocab_size":      vocab_size,
        "vocab_offsets":   vocab_offsets,
    }
    if has_embeddings:
        result["cnn_embeddings"] = np.stack(cnn_list)
    return result


def _sentences_s3_key(
    dataset_name: str, split_name: str, use_frac: float, shard_idx: int = 0, num_shards: int = NUM_GENERATION_SHARDS
) -> str:
    use_frac_str = f"{use_frac:.2f}" if split_name=="train" else "full"
    if num_shards == 1:
        return f"{prefix_out}/sentences/{dataset_name}_{split_name}_{use_frac_str}.npz"
    return (
        f"{prefix_out}/sentences/"
        f"{dataset_name}_{split_name}_{use_frac_str}_shard_{shard_idx:04d}_of_{num_shards:04d}.npz"
    )


def _build_save_kwargs(sentences: dict) -> dict:
    """Pack a sentences dict into NPZ-serializable arrays."""
    kwargs = {
        "sentence_tokens": sentences["sentence_tokens"],
        "labels":          sentences["labels"],
        "ecg_idxs":        sentences["ecg_idxs"],
        "num_beats":       sentences["num_beats"],
        "beat_idx_start":  sentences["beat_idx_start"],
        "beat_idx_end":    sentences["beat_idx_end"],
        "segment_order":   np.array(sentences["segment_order"]),
        "vocab_size":      np.array([sentences["vocab_size"]], dtype=np.int32),
    }
    for seg, offset in sentences["vocab_offsets"].items():
        kwargs[f"vocab_offset_{seg}"] = np.array([offset], dtype=np.int32)
    if "cnn_embeddings" in sentences:
        kwargs["cnn_embeddings"] = sentences["cnn_embeddings"]
    return kwargs


def _write_npz_to_s3(kwargs: dict, s3_key: str) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **kwargs)
    buf.seek(0)
    with s3_fs.open(f"s3://{bucket_out}/{s3_key}", "wb") as f:
        f.write(buf.read())


def save_finetuning_sentences(
    sentences: dict,
    dataset_name: str,
    split_name: str,
    use_frac: float,
    num_shards: int = NUM_GENERATION_SHARDS,
) -> list[str]:
    """Save sentences to S3 as ``num_shards`` compressed NPZ files.

    When ``num_shards == 1`` a single file is written (backward-compatible key).
    When ``num_shards > 1`` sentences are partitioned by ``ecg_idx % num_shards``
    so every sentence for a given ECG lands in exactly one shard — the shard a
    Ray worker will load based on its world rank.

    Returns the list of S3 keys written.
    """
    N         = len(sentences["sentence_tokens"])
    all_keys  = _build_save_kwargs(sentences)
    row_keys  = {"sentence_tokens", "labels", "ecg_idxs", "num_beats",
                 "beat_idx_start", "beat_idx_end", "cnn_embeddings"}

    if num_shards == 1:
        s3_key = _sentences_s3_key(dataset_name, split_name, use_frac, 0, 1)
        _write_npz_to_s3(all_keys, s3_key)
        log.info(f"Saved {N:,} sentences → s3://{bucket_out}/{s3_key}")
        return [s3_key]

    # Deterministic hash of the full ecg_idx string — avoids collisions from
    # different IDs that share the same digits (e.g. "A6163" vs "B6163"),
    # and works for numeric-looking IDs like "15709.0" without truncating them.
    shard_ids = np.array(
        [int(hashlib.md5(str(eid).encode()).hexdigest(), 16) % num_shards
         for eid in sentences["ecg_idxs"]],
        dtype=np.int64,
    )
    written: list[str] = []
    for si in range(num_shards):
        mask       = shard_ids == si
        shard_kw   = {
            k: v[mask] if k in row_keys and isinstance(v, np.ndarray) and v.shape[0] == N else v
            for k, v in all_keys.items()
        }
        s3_key = _sentences_s3_key(dataset_name, split_name, use_frac, si, num_shards)
        _write_npz_to_s3(shard_kw, s3_key)
        log.info(f"  shard {si}/{num_shards}: {mask.sum():,} sentences → s3://{bucket_out}/{s3_key}")
        written.append(s3_key)
    return written


def get_codebook_size(model: torch.nn.Module) -> int:
    return int(model._vq_vae._num_embeddings)


def get_or_build_finetuning_sentences(
    dataset_name: str,
    local_rank: int,
    in_channels: int,
    use_frac: float,
    cnn_embed_type: str,
    cnn_embed_scale: float,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    """Generate and save any missing per-split shard files to S3.

    Designed to be called by **world rank 0** inside the Ray
    ``train_loop_per_worker`` before a ``torch.distributed.barrier()``.
    All shards for all splits are checked independently; existing files are
    reused without touching the model.

    Parameters
    ----------

    """
    
    num_shards = NUM_GENERATION_SHARDS
    split_names = ("train", "val", "test")
    missing = [
        (split, si)
        for split in split_names
        for si in range(num_shards)
        if not s3_fs.exists(
            f"s3://{bucket_out}/{_sentences_s3_key(dataset_name, split, use_frac, si, num_shards)}"
        )
    ]
    if not missing:
        log.info("All fine-tuning sentence shards already exist — skipping generation.")
        return
    splits_needed = sorted({split for split, _ in missing})
    log.info(f"Generating sentence shards for splits: {splits_needed}")

    device = torch.device(f"cuda:{local_rank}")

    # Generate the patient split once so every segment sees identical train/val/test membership.
    splits = read_ecg_data(dataset_name, use_frac, train_frac, val_frac, test_frac)
    splits = {k: v for k, v in splits.items() if k in splits_needed}
    
    # Build per-segment models and embedders up front.
    models: dict[str, torch.nn.Module] = {}
    rocket_extractors: dict[str, ROCKETFeatureExtractor] = {}
    
    for seg in SEGMENTS:
        signal_len = SIGNAL_LENS[seg]
        # Load and rocket_extractor
        if cnn_embed_type=='rocket':
            rocket_extractors[seg] = get_or_create_rocket_extractor(segment=seg,
                                                                    in_channels=in_channels,
                                                                    signal_len=signal_len,
                                                                    device=device)
        else:
            rocket_extractors[seg] = None
            
        model_key = f"{model_prefix}/segment_{seg}_nleads_{in_channels}/vqvae_model_{seg}.pkl"
        models[seg] = _load_model_from_s3(model_key)

    seg_split_results, codebook_sizes = assign_finetuning_codebook_indices(
        models=models, device=device, splits=splits, in_channels=in_channels,
        cnn_embed_type=cnn_embed_type, cnn_embed_scale=cnn_embed_scale,
        rocket_extractors=rocket_extractors, batch_size=batch_size
    )

    # Restructure {seg: {split_name: data}} → {split_name: {seg: data}}
    all_assignments: dict[str, dict] = defaultdict(dict)
    for seg, split_data in seg_split_results.items():
        for split_name, data in split_data.items():
            all_assignments[split_name][seg] = data

    for split_name in splits_needed:
        sentences = build_finetuning_beat_sentences(
            assignments=all_assignments[split_name],
            codebook_sizes=codebook_sizes,
            seed=seed,
        )
        save_finetuning_sentences(sentences, dataset_name, split_name, use_frac, num_shards=num_shards)


class FinetuningBeatSentenceDataset(Dataset):
    """PyTorch Dataset of VQ-VAE beat sentences with ECG-level labels.

    Each Ray worker loads only its own pre-sharded data file from S3, so no
    single process ever holds the full split in RAM.  No ``DistributedSampler``
    is needed — the data is already partitioned across workers by ``ecg_idx``.

    Typical Ray + PyTorch DDP usage
    --------------------------------
    ::

        def train_loop_per_worker(config):
            ctx        = ray.train.get_context()
            world_rank = ctx.get_world_rank()
            world_size = ctx.get_world_size()

            # Rank 0 creates missing shard files, then all workers wait
            if world_rank == 0:
                get_or_build_finetuning_sentences(
                    dataset_name=dataset_name,
                    local_rank=local_rank,
                    in_channels=in_channels,
                    use_frac=use_frac,
                    cnn_embed_type=cnn_embed_type,
                    cnn_embed_scale=cnn_embed_scale,
                    batch_size=batch_size,
                )
            torch.distributed.barrier()

            loaders = create_finetuning_worker_dataloaders(
                dataset_name=config["dataset_name"],
                use_frac=config["use_frac"],
                batch_size=config["batch_size"],
            )
            model = ray.train.torch.prepare_model(config["model"])
            for epoch in range(config["n_epochs"]):
                for batch in loaders["train"]:
                    input_ids = batch["input_ids"]   # already on device via prepare_data_loader
                    labels    = batch["labels"]
                    ...

        trainer = ray.train.torch.TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            scaling_config=ray.train.ScalingConfig(num_workers=4, use_gpu=True),
        )

    __getitem__ keys
    ----------------
    input_ids           : (MAX_SENTENCE_LEN,) int64
    attention_mask      : (MAX_SENTENCE_LEN,) int64  — 1 for beat tokens + SEP, 0 for PAD
    token_type_ids      : (MAX_SENTENCE_LEN,) int64  — 0-based beat index per token position
    num_beats           : () int64
    labels              : (num_classes,) float32  — ECG-level class one hot encoding
    ecg_idxs            : () str                 - patient record id
    cnn_embeddings  : (MAX_SENTENCE_LEN, D) float32  — only when present in the file
    """

    def __init__(self, sentences: dict) -> None:
        self._tokens  = sentences["sentence_tokens"].astype(np.int32)
        self._n_beats = sentences["num_beats"].astype(np.int32)
        self._labels  = sentences["labels"].astype(np.float32)
        self._ecg_idxs = sentences["ecg_idxs"].astype(np.str_)
        self._cnn = sentences.get("cnn_embeddings")
        self.vocab_size    = int(sentences["vocab_size"])
        self.vocab_offsets = sentences["vocab_offsets"]
        self.cnn_embedding_dim = (
            int(self._cnn.shape[-1]) if self._cnn is not None else 0
        )
        log.info(
            f"FinetuningBeatSentenceDataset: {len(self._tokens):,} sentences"
            + (f", cnn_dim={self.cnn_embedding_dim}" if self._cnn is not None else "")
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n_beats = int(self._n_beats[idx])
        n_real  = n_beats * TOKENS_PER_BEAT   # number of real beat tokens before SEP

        attention_mask = np.zeros(MAX_SENTENCE_LEN, dtype=np.int64)
        attention_mask[:n_real + 1] = 1       # beat tokens + SEP attended; PAD masked

        token_type_ids = _TOKEN_TYPE_TEMPLATE.copy()
        token_type_ids[n_real]     = MAX_BEATS  # SEP gets its dedicated type
        token_type_ids[n_real + 1:] = 0         # PAD positions

        item = {
            "input_ids":      torch.tensor(self._tokens[idx].copy(), dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,            dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,            dtype=torch.long),
            "num_beats":      torch.tensor(n_beats,                   dtype=torch.long),
            "labels":         torch.tensor(self._labels[idx],         dtype=torch.float32),
            "ecg_idxs":       str(self._ecg_idxs[idx]),
        }
        if self._cnn is not None:
            item["cnn_embeddings"] = torch.tensor(
                self._cnn[idx].astype(np.float32), dtype=torch.float32
            )
        return item


def _make_split_dataloader(
    dataset_name: str,
    split_name: str,
    use_frac: float,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, FinetuningBeatSentenceDataset]:
    """Load one split from S3 (all shards, single process) and return
    (DataLoader, dataset) — no Ray required."""
    sentences = load_finetuning_sentences_for_worker(
        dataset_name, split_name, use_frac,
        world_rank=0, world_size=1,
        num_shards=NUM_GENERATION_SHARDS,
    )
    dataset = FinetuningBeatSentenceDataset(sentences)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_name == "train"),
        num_workers=num_workers,
        drop_last=False,
    )
    return loader, dataset


def _init_distributed() -> tuple[int, int, int]:
    """Initialize the default process group started by torchrun and return
    (global_rank, local_rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def main(in_channels: int, use_frac: float, batch_size: int, cnn_embed_type: str, cnn_embed_scale: float) -> None:
    """
    Generate fine-tuning sentences for all datasets and pull one batch from
    each split (train / val / test) for verification.
    """
    global_rank, local_rank, world_size = _init_distributed()

    for dataset_name in DATASETS:
        log.info(f"\n{'=' * 64}")
        log.info(f"  Dataset: {dataset_name}  |  use_frac={use_frac} |  in_channels={in_channels}")
        log.info(f"{'=' * 64}")

        # Only rank 0 generates missing shards; everyone else waits.
        if global_rank == 0:
            get_or_build_finetuning_sentences(
                dataset_name=dataset_name,
                local_rank=local_rank,
                in_channels=in_channels,
                use_frac=use_frac,
                cnn_embed_type=cnn_embed_type,
                cnn_embed_scale=cnn_embed_scale,
                batch_size=batch_size,
            )
        if dist.is_initialized():
            dist.barrier()

        for split_name in ("train", "val", "test"):
            loader, dataset = _make_split_dataloader(dataset_name, split_name, use_frac, batch_size)
            batch = next(iter(loader))
            actual_bs = batch["input_ids"].shape[0]

            log.info(f"\n  [{dataset_name}] {split_name}")
            log.info(f"    dataset length : {len(dataset):,} sentences")
            log.info(f"    num batches    : {len(loader):,}")
            log.info(f"    batch size     : {actual_bs}")
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    log.info(f"    batch['{key}']  shape={tuple(val.shape)}  dtype={val.dtype}")

            # --- example: first sample in the batch ---
            log.info(f"\n    -- example (sample 0) --")
            log.info(f"    input_ids      : {batch['input_ids'][0].tolist()}")
            log.info(f"    attention_mask : {batch['attention_mask'][0].tolist()}")
            log.info(f"    token_type_ids : {batch['token_type_ids'][0].tolist()}")
            log.info(f"    num_beats      : {int(batch['num_beats'][0])}")
            log.info(f"    label          : {batch['labels'][0].tolist()}")
            if "cnn_embeddings" in batch:
                emb = batch["cnn_embeddings"][0].float()
                log.info(
                    f"    cnn_emb    : shape={tuple(emb.shape)}"
                    f"  mean={emb.mean():.4f}  std={emb.std():.4f}"
                )


def _worker_shard_indices(world_rank: int, world_size: int, num_shards: int = NUM_GENERATION_SHARDS) -> list[int]:
    """Which of the NUM_GENERATION_SHARDS fixed shard files this worker
    should load, given the *current* world_size."""
    if world_size > num_shards:
        raise ValueError(
            f"world_size ({world_size}) exceeds num_shards ({num_shards}); "
            f"increase NUM_GENERATION_SHARDS and regenerate, or reduce world_size."
        )
    return [si for si in range(num_shards) if si % world_size == world_rank]


def _load_single_shard(
    dataset_name: str, split_name: str, use_frac: float, shard_idx: int, num_shards: int = NUM_GENERATION_SHARDS,
) -> dict:
    """Load exactly one physical shard file — no rank/world_size mapping."""
    s3_path = f"s3://{bucket_out}/{_sentences_s3_key(dataset_name, split_name, use_frac, shard_idx, num_shards)}"
    with s3_fs.open(s3_path, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
        result = {
            "sentence_tokens": data["sentence_tokens"],
            "labels":          data["labels"],
            "ecg_idxs":        data["ecg_idxs"],
            "num_beats":       data["num_beats"],
            "beat_idx_start":  data["beat_idx_start"],
            "beat_idx_end":    data["beat_idx_end"],
            "segment_order":   list(data["segment_order"].astype(str)),
            "vocab_size":      int(np.array(data["vocab_size"]).reshape(-1)[0]),
            "vocab_offsets":   {
                seg: int(np.array(data[f"vocab_offset_{seg}"]).reshape(-1)[0])
                for seg in SEGMENTS
            },
        }
        if "cnn_embeddings" in data:
            result["cnn_embeddings"] = data["cnn_embeddings"]
    return result


def _concat_sentence_dicts(parts: list[dict]) -> dict:
    """Merge multiple per-shard sentence dicts into one, concatenating the
    row-wise arrays. vocab_size/vocab_offsets/segment_order are identical
    across shards of the same dataset — taken from the first part."""
    if not parts:
        raise ValueError("No shard parts to concatenate")
    row_keys = ("sentence_tokens", "labels", "ecg_idxs", "num_beats",
                "beat_idx_start", "beat_idx_end", "cnn_embeddings")
    merged: dict = {
        "segment_order": parts[0]["segment_order"],
        "vocab_size":    parts[0]["vocab_size"],
        "vocab_offsets": parts[0]["vocab_offsets"],
    }
    for key in row_keys:
        if all(key in p for p in parts):
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def load_finetuning_sentences_for_worker(
    dataset_name: str,
    split_name: str,
    use_frac: float,
    world_rank: int,
    world_size: int,
    num_shards: int = NUM_GENERATION_SHARDS,
) -> dict:
    """Load and concatenate all fixed shards assigned to this worker for the
    current world_size. For Ray usage —
    the shard→worker mapping is computed at load time, not baked into
    generation, so world_size can change between runs without regenerating.
    """
    shard_indices = _worker_shard_indices(world_rank, world_size, num_shards)
    parts = [
        _load_single_shard(dataset_name, split_name, use_frac, si, num_shards)
        for si in shard_indices
    ]
    merged = _concat_sentence_dicts(parts)
    log.info(
        f"Loaded {len(merged['sentence_tokens']):,} sentences "
        f"(rank {world_rank}/{world_size}, shards {shard_indices}) "
        f"for [{dataset_name}] {split_name}"
    )
    return merged


def create_finetuning_worker_dataloaders(
    dataset_name: str,
    use_frac: float,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    import ray.train
    import ray.train.torch as ray_torch

    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    g = torch.Generator()
    g.manual_seed(seed)
    
    def _worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        sentences = load_finetuning_sentences_for_worker(
            dataset_name, split_name, use_frac,
            world_rank=world_rank,
            world_size=world_size,
            num_shards=NUM_GENERATION_SHARDS,
        )
        dataset = FinetuningBeatSentenceDataset(sentences)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
            generator=g if split_name == "train" else None,
            worker_init_fn=_worker_init_fn,
        )
        loaders[split_name] = ray_torch.prepare_data_loader(loader)
    return loaders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VQ-VAE fine-tuning sentences and verify one batch per split."
    )
    parser.add_argument(
        "--in_channels", type=int, default=NLEADS, choices=[1, NLEADS],
        help=f"1 = Lead II only, {NLEADS} = all leads",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Sentences per VQ-VAE forward pass / DataLoader batch (default: 32)",
    )
    parser.add_argument(
        "--use_frac", type=float, default=1.0,
        help="Use fraction of dataset (default 1.0)",
    )

    parser.add_argument("--cnn_embed_type", type=str, default=None, choices=['vqvae_encoder', 'rocket'])
    parser.add_argument("--cnn_embed_scale", type=float, default=1.0)

    args = parser.parse_args()
    main(args.in_channels, args.use_frac, args.batch_size, args.cnn_embed_type, args.cnn_embed_scale)