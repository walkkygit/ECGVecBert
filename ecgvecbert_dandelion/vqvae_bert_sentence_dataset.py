"""
Codebook assignment for VQ-VAE ECG waveforms and beat sentence construction.

Loads a trained VQ-VAE model and assigns each waveform to its nearest
codebook vector.  Outputs an NPZ file per segment containing:
  - patient_id  : (N,)  str
  - study_id    : (N,)  str
  - beat_idx    : (N,)  int32
  - segment     : str   ("P", "QRS", "T", ..)
  - token_index : (N,)  int16  — codebook index in [0, num_embeddings)

These per-segment NPZ files are then joined on (patient_id, study_id, beat_idx)
to produce beat sentences for BERT pre-training.

The sentence output can be split into num_shards NPZ files (sharded by patient_id
hash).  The BERT DataLoader uses file-sharding for DDP: each rank reads
  my_shards = all_shard_keys[rank::world_size]
Patient train/val disjointness is enforced by a globally-computed split that is
saved once to S3 and reloaded on every subsequent run.

Usage for testing:
    python vqvae_bert_sentence_dataset.py \
        --in_channels 1 \

"""

import argparse
import os
import io
import logging
import pickle
from collections import defaultdict

import boto3
from botocore.exceptions import ClientError
import numpy as np
import s3fs
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from datetime import timedelta

from vqvae_ecg_waveforms_dataset import (
    create_ecg_dataloaders,
    collate_batch,
    list_h5_files,
    SEGMENTS,
    NLEADS,
    bucket_out,
)
from encoder_features_ecg_waveforms import vqvae_encoder_embedding
from rocket_features_ecg_waveforms import ROCKETFeatureExtractor
from vqvae_model import Model, ECGResidualEncoder, ECGResidualDecoder, VectorQuantizerEMA, VectorQuantizer, ResidualStack, Residual


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Per-segment waveform lengths — must match signal_lens used during VQ-VAE training
SIGNAL_LENS: dict[str, int] = {
    "P":   88,
    "PQ": 88,
    "QRS": 176,
    "ST": 128,
    "T":   120,
}

# Beat-sentence construction
SENTENCE_LENGTHS = [1, 2, 6, 8]           # beats per sentence (sampled uniformly)
TOKENS_PER_BEAT  = len(SEGMENTS)                      # 3 if [P, QRS, T], 5 if [P, PQ, QRS, ST, T],
MAX_BEATS        = max(SENTENCE_LENGTHS)  # 8
MAX_SENTENCE_LEN = MAX_BEATS * TOKENS_PER_BEAT + 1  # 25 = 24 beat tokens + 1 SEP if [P, QRS, T], 41 if [P, PQ, QRS, ST, T],

# BERT-standard special token IDs (CLS unused here)
PAD_TOKEN_ID        = 0    # [PAD] fills positions after SEP
SEP_TOKEN_ID        = 1  # [SEP] appended after each sentence's beat tokens
MASK_TOKEN_ID       = 2  # [MASK] replaces tokens during MLM training
FIRST_REAL_TOKEN_ID = 3  # P codebook index 0 maps here; QRS and T follow contiguously

base_prefix = "aruna-files/vqvae"
# S3 key for the persisted patient split (written once, read on every subsequent run)
PATIENT_SPLIT_KEY = f"{base_prefix}/bert/patient_split.npz"

s3_client = boto3.client("s3")
s3_fs     = s3fs.S3FileSystem()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_key: str, bucket: str = bucket_out) -> torch.nn.Module:
    """Load a pickled VQ-VAE model from S3."""
    log.info(f"Loading model from s3://{bucket}/{model_key}")
    response = s3_client.get_object(Bucket=bucket, Key=model_key)
    model = pickle.loads(response["Body"].read())
    model.eval()
    return model


def get_codebook_size(model: torch.nn.Module) -> int:
    return int(model._vq_vae._num_embeddings)


@torch.no_grad()
def _embed_batch(cnn_embed_type: str, cnn_embed_scale: float, model: torch.nn.Module, device: torch.device, 
                 waveform: torch.Tensor, rocket_extractor: ROCKETFeatureExtractor) -> np.ndarray:
    """Extract cnn embeddings for a batch of waveforms.

    Parameters
    ----------
    waveform : (B, n_leads, L) float32
    Returns  : (B, embedding_dim) float16 on CPU
               embedding_dim = 2*hidden_dim*canonical_len=2560 if cnn_embed_type=='vqvae_encoder'
               embedding_dim = 2000 if cnn_embed_type=='rocket' with 1000 kernels, fixed seed
               Load saved ROCKET model for fine-tuning, if used.
    """
    if cnn_embed_type == 'vqvae_encoder':
        embeddings = vqvae_encoder_embedding(model, device, waveform)
        embeddings = embeddings*cnn_embed_scale
        return embeddings.cpu().float().numpy().astype(np.float16)
    elif cnn_embed_type == 'rocket':
        embeddings = rocket_extractor(waveform)*cnn_embed_scale
        return embeddings.cpu().float().numpy().astype(np.float16)
    else:
        raise ValueError(f"Unknown cnn_embed_type {cnn_embed_type}")


def get_or_create_rocket_extractor(
    segment: str,
    in_channels: int,
    signal_len: int,
    device: torch.device,
    bucket: str = bucket_out,
    num_kernels: int = 1000,
    seed: int = 42,
) -> ROCKETFeatureExtractor:
    """Load a previously-saved ROCKETFeatureExtractor from S3, or build and save
    one if it doesn't exist yet. Kernels are generated exactly once per
    (segment, in_channels) — never regenerated from seed on subsequent runs."""

    rocket_key = f"{base_prefix}/rocket/rocket_extractor_seg_{segment}_nleads_{in_channels}.pkl"
    s3_path = f"s3://{bucket}/{rocket_key}"

    if s3_fs.exists(s3_path):
        try:
            log.info(f"  {rocket_key} already exists — loading instead of recomputing")
            return load_rocket_extractor(rocket_key, device, bucket)
        except Exception as e:
            log.warning(f"  {rocket_key} exists but failed to load ({e!r}) — recomputing")

    log.info(f"Building new ROCKETFeatureExtractor for segment={segment}, "
             f"in_channels={in_channels}, signal_len={signal_len}, num_kernels={num_kernels}, seed={seed}")
    extractor = ROCKETFeatureExtractor(
        in_channels=in_channels,
        signal_len=signal_len,
        num_kernels=num_kernels,
        seed=seed,
    )  # built on CPU deliberately -- keeps kernel generation hardware-independent

    buffer = io.BytesIO()
    pickle.dump(extractor, buffer)
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=rocket_key, Body=buffer.getvalue())
    log.info(f"  Saved new ROCKETFeatureExtractor to {s3_path}")

    return extractor.to(device)


# Load saved ROCKET feature extractor model 
def load_rocket_extractor(rocket_key: str,  device: torch.device, bucket: str = bucket_out) -> ROCKETFeatureExtractor:
    response = s3_client.get_object(Bucket=bucket, Key=rocket_key)
    extractor = pickle.loads(response["Body"].read())
    extractor.eval()
    return extractor.to(device)


# ── Assignment pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def assign_codebook_indices(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cnn_embed_type: str,
    cnn_embed_scale: float = 1.0,
    rocket_extractor: ROCKETFeatureExtractor | None = None,
) -> dict:
    """
    Run a full forward pass over `loader` and collect codebook indices.

    Returns
    -------
    dict with keys:
        patient_ids, study_ids, beat_idxs : lists / arrays of metadata
        token_indices                      : (N,) int16 ndarray
        embeddings                         : (N, embedding_dim) float16 — only if cnn_embed_type given
    """
    patient_ids, study_ids, beat_idxs = [], [], []
    token_indices = []
    embeddings = [] if cnn_embed_type is not None else None

    for i, batch in enumerate(loader):
        waveform = batch["waveform"].to(device)
        indices = model.encode_to_indices(waveform)        # [B] LongTensor
        token_indices.append(indices.cpu().numpy().astype(np.int16))

        if cnn_embed_type is not None:
            emb_batch = _embed_batch(cnn_embed_type, cnn_embed_scale, model, device, waveform, rocket_extractor)
            if emb_batch is not None:
                embeddings.append(emb_batch)

        patient_ids.extend(batch["patient_id"])
        study_ids.extend(batch["study_id"])
        beat_idxs.extend(batch["beat_idx"].tolist())

        if (i + 1) % 100 == 0:
            log.info(f"  processed {(i+1) * loader.batch_size:,} samples...")

    result = {
        "patient_ids":   np.array(patient_ids, dtype=object),
        "study_ids":     np.array(study_ids,   dtype=object),
        "beat_idxs":     np.array(beat_idxs,   dtype=np.int32),
        "token_indices": np.concatenate(token_indices),
    }
    if cnn_embed_type is not None and embeddings is not None:
        result["embeddings"] = np.concatenate(embeddings)
    return result


# ── Save / load assignments ────────────────────────────────────────────────────

def save_assignments(results: dict, segment: str, output_key: str, codebook_size: int):
    """Save token assignments (and optional cnn embeddings) as a compressed NPZ to S3."""
    save_kwargs = dict(
        patient_id=results["patient_ids"].astype(str),
        study_id=results["study_ids"].astype(str),
        beat_idx=results["beat_idxs"],
        token_index=results["token_indices"],
        segment=np.array([segment]),
        codebook_size=np.array([codebook_size], dtype=np.int32),
    )
    if "embeddings" in results:
        save_kwargs["embeddings"] = results["embeddings"]
        log.info(f"  embedding shape per sample: {results['embeddings'].shape[1]}-dim float16")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **save_kwargs)
    buffer.seek(0)
    s3_path = f"s3://{bucket_out}/{output_key}"
    log.info(f"Saving assignments to {s3_path}")
    with s3_fs.open(s3_path, "wb") as f:
        f.write(buffer.read())
    log.info(f"Saved {len(results['token_indices']):,} assignments for {segment} (codebook_size={codebook_size}).")


# ── Beat sentence builder ──────────────────────────────────────────────────────

def build_beat_sentences(
    assignments: dict[str, dict],
    output_key: str,
    codebook_sizes: dict[str, int],
    seed: int = 42,
    num_shards: int = 1,
) -> list:
    """
    Join P, QRS, T token assignments on (patient_id, study_id, beat_idx),
    group beats by study, and build variable-length beat sentences for BERT.
    More generally, P, PQ, QRS, ST, T segments.

    Token IDs are shifted to their final BERT vocab positions:
        P   : FIRST_REAL_TOKEN_ID  ..  FIRST_REAL_TOKEN_ID + num_P - 1
        QRS : P_end + 1            ..  P_end + num_QRS
        T   : QRS_end + 1          ..  QRS_end + num_T
        SEP : SEP_TOKEN_ID  (1)
        PAD : PAD_TOKEN_ID  (0)

    Parameters
    ----------
    output_key : S3 key stem.  With num_shards=1 the file is written exactly
                 here; with num_shards>1 the stem is suffixed with
                 _shard_NNNN_of_MMMM.npz
    num_shards : split output across this many NPZ files, distributed by
                 deterministic patient_id hash.  All sentences for a given
                 patient always land in a single shard.  Set this to 1 so
                 that create_bert_dataloaders can distribute one shard per rank
                 without any inter-rank communication.

    Each shard NPZ contains (one row per sentence):
        sentence_tokens    : (N, MAX_SENTENCE_LEN) int32
        num_beats          : (N,) int8
        patient_id         : (N,) str
        study_id           : (N,) str
        beat_idx_start     : (N,) int32
        beat_idx_end       : (N,) int32
        segment_order      : ['P', 'QRS', 'T'] or ['P', 'PQ', 'QRS', 'ST', 'T']
        vocab_offsets : (1,) int32 for each segment
        vocab_size         : (1,) int32
        cnn_embeddings : (N, MAX_SENTENCE_LEN, embedding_dim) float16  — when present
    """
    vocab_offsets: dict[str, int] = {}
    vocab_offsets["P"] = FIRST_REAL_TOKEN_ID
    for i in range(1, len(SEGMENTS)):
        vocab_offsets[SEGMENTS[i]] = vocab_offsets[SEGMENTS[i-1]] + codebook_sizes[SEGMENTS[i-1]]
        log.info(f"Offset for segment {SEGMENTS[i]}: {vocab_offsets[SEGMENTS[i]]}")
    vocab_size = vocab_offsets[SEGMENTS[-1]] + codebook_sizes[SEGMENTS[-1]]
    log.info(f"(total vocab size: {vocab_size})")

    rng = np.random.default_rng(seed)

    has_embeddings = all("embeddings" in data for data in assignments.values())
    embedding_dim  = assignments["P"]["embeddings"].shape[1] if has_embeddings else 0
    if has_embeddings:
        log.info(f"cnn embeddings present — embedding_dim={embedding_dim}")

    # Index each beat by (patient_id, study_id, beat_idx)
    indexed: dict[tuple, dict] = {}
    for seg, data in assignments.items():
        embs = data.get("embeddings")
        for i, (pid, sid, bidx, tok) in enumerate(zip(
            data["patient_ids"], data["study_ids"],
            data["beat_idxs"],   data["token_indices"]
        )):
            key = (str(pid), str(sid), int(bidx))
            indexed.setdefault(key, {})[seg] = int(tok)
            if embs is not None:
                indexed[key][f"{seg}_emb"] = embs[i]

    full_keys = [k for k, v in indexed.items() if all(s in v for s in SEGMENTS)]
    log.info(f"Beats with all segments: {len(full_keys):,} "
             f"(dropped {len(indexed) - len(full_keys):,} incomplete beats)")

    study_beats: dict[tuple, list[int]] = defaultdict(list)
    for (pid, sid, bidx) in full_keys:
        study_beats[(pid, sid)].append(bidx)
    for key in study_beats:
        study_beats[key].sort()

    sentence_tokens_list:    list[list[int]]  = []
    cnn_embeddings_list: list[np.ndarray] = []
    num_beats_list:          list[int] = []
    patient_id_list:         list[str] = []
    study_id_list:           list[str] = []
    beat_idx_start_list:     list[int] = []
    beat_idx_end_list:       list[int] = []

    for (pid, sid), beat_seq in study_beats.items():
        pos = 0
        while pos < len(beat_seq):
            remaining = len(beat_seq) - pos
            valid = [l for l in SENTENCE_LENGTHS if l <= remaining]
            if not valid:
                break
            n = int(rng.choice(valid))
            if pos + n > len(beat_seq):
                break
            window = beat_seq[pos: pos + n]
            tokens: list[int] = []
            if has_embeddings:
                emb_row = np.zeros((MAX_SENTENCE_LEN, embedding_dim), dtype=np.float16)
            for beat_pos, bidx in enumerate(window):
                v = indexed[(pid, sid, bidx)]
                current_beat: list[int] = []
                for seg in SEGMENTS:
                    current_beat.append(v[seg] + vocab_offsets[seg])
                tokens += current_beat
                if has_embeddings:
                    base = beat_pos * TOKENS_PER_BEAT
                    for seg_id, seg in enumerate(SEGMENTS):
                        emb_row[base + seg_id] = v[f"{seg}_emb"]
            tokens += [SEP_TOKEN_ID]
            tokens += [PAD_TOKEN_ID] * (MAX_SENTENCE_LEN - len(tokens))
            sentence_tokens_list.append(tokens)
            if has_embeddings:
                cnn_embeddings_list.append(emb_row)
            num_beats_list.append(n)
            patient_id_list.append(pid)
            study_id_list.append(sid)
            beat_idx_start_list.append(window[0])
            beat_idx_end_list.append(window[-1])
            pos += n

    N = len(sentence_tokens_list)
    log.info(f"Built {N:,} beat sentences from {len(study_beats):,} studies")

    # Pack into arrays once; shards index into these
    all_tokens     = np.array(sentence_tokens_list,  dtype=np.int32)  # (N, 25)
    all_n_beats    = np.array(num_beats_list,         dtype=np.int8)
    all_patient_id = np.array(patient_id_list,        dtype=str)
    all_study_id   = np.array(study_id_list,          dtype=str)
    all_idx_start  = np.array(beat_idx_start_list,    dtype=np.int32)
    all_idx_end    = np.array(beat_idx_end_list,       dtype=np.int32)
    all_cnn    = np.stack(cnn_embeddings_list) if has_embeddings else None

    common_meta = dict(
        segment_order=np.array(SEGMENTS),
        vocab_size=np.array([vocab_size], dtype=np.int32),
    )
    for seg in SEGMENTS:
        common_meta[f"vocab_offset_{seg}"] = np.array([vocab_offsets[seg]], dtype=np.int32)

    def _save_shard(row_indices: np.ndarray, shard_s3_key: str) -> None:
        kwargs = dict(
            sentence_tokens=all_tokens[row_indices],
            num_beats=all_n_beats[row_indices],
            patient_id=all_patient_id[row_indices],
            study_id=all_study_id[row_indices],
            beat_idx_start=all_idx_start[row_indices],
            beat_idx_end=all_idx_end[row_indices],
            **common_meta,
        )
        if all_cnn is not None:
            kwargs["cnn_embeddings"] = all_cnn[row_indices]
        buf = io.BytesIO()
        np.savez_compressed(buf, **kwargs)
        buf.seek(0)
        s3_path = f"s3://{bucket_out}/{shard_s3_key}"
        with s3_fs.open(s3_path, "wb") as f:
            f.write(buf.read())
        log.info(f"  shard saved: {len(row_indices):,} sentences → {s3_path}")

    assert num_shards == 1, ("num_shards must be set to 1")
    _save_shard(np.arange(N), output_key)
    log.info(f"Done. {N:,} sentences saved to s3://{bucket_out}/{output_key}")
    return [output_key]


# ── Token-type template ────────────────────────────────────────────────────────

# positions 0,1,2 → beat 0; positions 3,4,5 → beat 1; …; position 24 → SEP, reserve type MAX_BEATS (= 8) for SEP
_TOKEN_TYPE_TEMPLATE = np.concatenate([
    np.repeat(np.arange(MAX_BEATS), TOKENS_PER_BEAT),
    [MAX_BEATS],   # SEP position
]).astype(np.int64)  # shape (25,) or (41,)


# ── Patient split: compute once, persist to S3 ────────────────────────────────

def compute_or_load_patient_split(
    all_shard_keys: list[str],
    split_s3_key: str = PATIENT_SPLIT_KEY,
    train_ratio: float = 0.85,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_patient_ids, val_patient_ids) as sorted str ndarrays.

    First call (split file absent from S3):
      - Rank 0 scans every shard NPZ to collect all unique patient_ids.
      - Rank 0 shuffles deterministically, splits, and saves to S3.
      - Rank 0 broadcasts the result to all other ranks in-memory.

    Subsequent calls:
      - Rank 0 loads the split from S3 and broadcasts to all other ranks.

    Only rank 0 ever reads from or writes to S3, eliminating concurrent writes
    and redundant scans.  Other ranks receive the arrays via broadcast.

    Parameters
    ----------
    all_shard_keys : complete list of sentence-shard S3 keys (relative to
                     bucket_out).  Used only when the split does not yet exist.
    split_s3_key   : S3 key (relative to bucket_out) where the split is stored.
    train_ratio    : fraction of unique patients assigned to the training set.
    seed           : RNG seed — must be the same across all runs.
    rank           : this process's DDP rank.
    world_size     : total number of DDP processes.
    """

    s3_path = f"s3://{bucket_out}/{split_s3_key}"
    payload: list[np.ndarray | None] = [None, None]  # [train_pids, val_pids]

    if rank == 0:
        # ── Try to load existing split ──────────────────────────────────────
        if s3_fs.exists(s3_path):
            with s3_fs.open(s3_path, "rb") as f:
                data = np.load(io.BytesIO(f.read()), allow_pickle=True)
            train_pids = data["train_patient_ids"].astype(str)
            val_pids   = data["val_patient_ids"].astype(str)
            log.info(
                f"Loaded patient split from {s3_path}: "
                f"{len(train_pids):,} train / {len(val_pids):,} val patients"
            )
        else:
            # ── Compute from shard files ────────────────────────────────────
            log.info("Patient split not found on S3 — computing from all shard files...")
            all_pids: set[str] = set()
            for key in all_shard_keys:
                with s3_fs.open(f"s3://{bucket_out}/{key}", "rb") as f:
                    data = np.load(io.BytesIO(f.read()), allow_pickle=True)
                all_pids.update(str(p) for p in data["patient_id"])
                log.info(f"  scanned {key}: running total {len(all_pids):,} unique patients")

            unique_patients = np.array(sorted(all_pids), dtype=str)
            rng = np.random.default_rng(seed)
            rng.shuffle(unique_patients)

            n_train    = int(len(unique_patients) * train_ratio)
            train_pids = unique_patients[:n_train]
            val_pids   = unique_patients[n_train:]

            log.info(
                f"Patient split computed: {len(train_pids):,} train / {len(val_pids):,} val "
                f"({len(unique_patients):,} total unique patients)"
            )

            buf = io.BytesIO()
            np.savez_compressed(buf, train_patient_ids=train_pids, val_patient_ids=val_pids)
            buf.seek(0)
            with s3_fs.open(s3_path, "wb") as f:
                f.write(buf.read())
            log.info(f"Patient split saved to {s3_path}")

        payload = [train_pids, val_pids]

    # Broadcast from rank 0 to all other ranks — no S3 read needed on non-0 ranks.
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)

    train_pids, val_pids = payload[0], payload[1] 
    return train_pids, val_pids


# ── File-sharded BERT Dataset ──────────────────────────────────────────────────

class FileShardedBeatSentenceDataset(Dataset):
    """
    Loads a list of sentence-shard NPZ files from S3 and presents them as a
    single PyTorch Dataset, optionally filtered to a specific patient set.

    Design for DDP
    --------------
    Each rank instantiates with its own disjoint subset of shards:

        my_shards = all_shard_keys[rank::world_size]
        train_pids, val_pids = compute_or_load_patient_split(all_shard_keys)
        train_ds = FileShardedBeatSentenceDataset(my_shards, set(train_pids))
        val_ds   = FileShardedBeatSentenceDataset(my_shards, set(val_pids))

    File-sharding replaces DistributedSampler: no inter-rank coordination is
    needed during training.

    Parameters
    ----------
    shard_s3_keys  : S3 keys (relative to bucket_out) of the shard NPZs to load.
    patient_split : if given, only sentences whose patient_id is in this set
                     are included, enforcing train/val disjointness.

    Attributes (set after __init__)
    --------------------------------
    vocab_size, vocab_offsets  — from the first loaded shard
    cnn_embedding_dim      — 0 if no embeddings in the shards

    __getitem__ returns
    -------------------
    input_ids           : (MAX_SENTENCE_LEN,) int64
    attention_mask      : (MAX_SENTENCE_LEN,) int64  — 1 for beat tokens + SEP, 0 for PAD
    token_type_ids      : (MAX_SENTENCE_LEN,) int64  — beat index (0-based) per token position
    num_beats           : () int64
    cnn_embeddings  : (MAX_SENTENCE_LEN, embedding_dim) float32  — only when present
    """

    def __init__(
        self,
        shard_s3_keys: list[str],
        patient_split: set[str] | None = None,
    ):
        if not shard_s3_keys:
            raise ValueError("shard_s3_keys is empty")

        tokens_chunks:  list[np.ndarray] = []
        n_beats_chunks: list[np.ndarray] = []
        cnn_chunks: list[np.ndarray] = []
        has_cnn     = True
        vocab_meta: dict | None = None

        for key in shard_s3_keys:
            log.info(f"  loading shard s3://{bucket_out}/{key}")
            with s3_fs.open(f"s3://{bucket_out}/{key}", "rb") as f:
                data = np.load(io.BytesIO(f.read()), allow_pickle=True)

            if patient_split is not None:
                pids = data["patient_id"].astype(str)
                mask = np.fromiter(
                    (p in patient_split for p in pids), dtype=bool, count=len(pids)
                )
            else:
                mask = np.ones(len(data["sentence_tokens"]), dtype=bool)

            tokens_chunks.append(data["sentence_tokens"][mask].astype(np.int32))
            n_beats_chunks.append(data["num_beats"][mask].astype(np.int32))

            if has_cnn:
                if "cnn_embeddings" in data:
                    cnn_chunks.append(data["cnn_embeddings"][mask])
                else:
                    has_cnn = False
                    cnn_chunks.clear()

            if vocab_meta is None:
                vocab_meta: dict[str, int] = {}
                vocab_meta["vocab_size"] = int(np.array(data["vocab_size"]).reshape(-1)[0])
                for segname in SEGMENTS:
                    vocab_meta[f"vocab_offset_{segname}"] = int(np.array(data[f"vocab_offset_{segname}"]).reshape(-1)[0])

        self._tokens  = np.concatenate(tokens_chunks,  axis=0)  # (N, MAX_SENTENCE_LEN) int32
        self._n_beats = np.concatenate(n_beats_chunks, axis=0)  # (N,) int32
        self._cnn = (
            np.concatenate(cnn_chunks, axis=0)
            if (has_cnn and cnn_chunks)
            else None
        )
        self.cnn_embedding_dim = int(self._cnn.shape[-1]) if self._cnn is not None else 0

        self.vocab_offsets: dict[str, int] = {}
        for segname in SEGMENTS:
            self.vocab_offsets[segname] = vocab_meta[f"vocab_offset_{segname}"]    # type: ignore[index]
        self.vocab_size = vocab_meta["vocab_size"]

        log.info(
            f"FileShardedBeatSentenceDataset: {len(self._tokens):,} sentences "
            f"from {len(shard_s3_keys)} shard(s)"
            + (f", cnn_dim={self.cnn_embedding_dim}" if self._cnn is not None else "")
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n_beats = int(self._n_beats[idx])
        n_real  = n_beats * TOKENS_PER_BEAT

        attention_mask = np.zeros(MAX_SENTENCE_LEN, dtype=np.int64)
        attention_mask[:n_real + 1] = 1     # beat tokens + SEP attended; PAD masked

        token_type_ids = _TOKEN_TYPE_TEMPLATE.copy()
        token_type_ids[n_real] = MAX_BEATS      # SEP always gets the dedicated SEP type
        token_type_ids[n_real + 1:] = 0         # PAD positions

        item = {
            "input_ids":      torch.tensor(self._tokens[idx].copy(), dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,            dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids,            dtype=torch.long),
            "num_beats":      torch.tensor(n_beats,                   dtype=torch.long),
        }
        if self._cnn is not None:
            item["cnn_embeddings"] = torch.tensor(
                self._cnn[idx].astype(np.float32), dtype=torch.float32
            )
        return item


# ── DDP-ready DataLoader factory ───────────────────────────────────────────────

def create_bert_dataloaders(
    all_shard_keys: list[str],
    rank: int,
    world_size: int,
    split_s3_key: str = PATIENT_SPLIT_KEY,
    batch_size: int = 512,
    num_workers: int = 4,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Primary entry point called once per DDP rank from the BERT pretraining script.

    Steps
    -----
    1. Loads (or computes-and-saves) the global patient split from S3.
    2. Assigns a disjoint file subset to this rank:
           my_shards = all_shard_keys[rank::world_size]
    3. Instantiates separate train / val FileShardedBeatSentenceDatasets by
       filtering sentences against the train or val patient set.
    4. Returns (train_loader, val_loader) ready for immediate use.

    No DistributedSampler is needed: file-sharding handles data distribution
    with padded loader.

    Parameters
    ----------
    all_shard_keys : complete ordered list of sentence-shard S3 keys (relative
                     to bucket_out).  The list length should be a multiple of
                     world_size; pad with duplicates if necessary.
                     Typically produced by:
                         stem = "aruna-files/vqvae/bert/beat_sentences..."
                         all_shard_keys = [
                             f"{stem}_shard_{i:04d}_of_{num_shards:04d}.npz"
                             for i in range(num_shards)
                         ]
    rank           : this process's DDP rank  (dist.get_rank())
    world_size     : total DDP processes       (dist.get_world_size())
    split_s3_key   : S3 key for the persisted patient split
    batch_size     : sentences per batch *per rank*
    num_workers    : DataLoader worker processes per rank
    train_ratio    : fraction of patients in training (default 0.85)
    seed           : RNG seed — must be identical across all ranks and runs

    Returns
    -------
    train_loader, val_loader

    Typical usage in a torchrun / DDP training script
    --------------------------------------------------
        import torch.distributed as dist
        dist.init_process_group("nccl")

        train_loader, val_loader = create_bert_dataloaders(
            all_shard_keys = all_shard_keys,
            rank           = dist.get_rank(),
            world_size     = dist.get_world_size(),
        )
        for epoch in range(n_epochs):
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ...
    """
    train_pids, val_pids = compute_or_load_patient_split(
        all_shard_keys, split_s3_key, train_ratio, seed, rank, world_size
    )
    train_patient_set = set(train_pids.tolist())
    val_patient_set   = set(val_pids.tolist())

    my_shards = all_shard_keys[rank::world_size]
    log.info(
        f"Rank {rank}/{world_size}: assigned {len(my_shards)} / {len(all_shard_keys)} shards"
    )

    train_dataset = FileShardedBeatSentenceDataset(
        my_shards, patient_split=train_patient_set,
    )
    val_dataset = FileShardedBeatSentenceDataset(
        my_shards, patient_split=val_patient_set, 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,  # pretraining uses padded loader
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader


# ── Main (codebook assignment + sentence building) ────────────────────────────

def run_assignment(
    h5_files: str,
    segment: str,
    model_key: str,
    output_key: str,
    signal_len: int,
    in_channels: int,
    lead_indices: list[int],
    batch_size: int,
    device: torch.device,
    cnn_embed_type: str | None = None,
    cnn_embed_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], int]:

    s3_path = f"s3://{bucket_out}/{output_key}"
    if s3_fs.exists(s3_path):
        try:
            log.info(f"  {output_key} already exists — loading instead of recomputing")
            return _load_saved_assignment(output_key)
        except Exception as e:
            log.warning(f"  {output_key} exists but failed to load ({e!r}) — recomputing")

    log.info(f"Running codebook assignment for segment={segment} on {device} "
             f"with number of leads={in_channels}, signal_len={signal_len}")

    model = load_model(model_key)
    codebook_size = get_codebook_size(model)
    log.info(f"Codebook size for segment={segment}: {codebook_size}")
    model = model.to(device)

    if cnn_embed_type is not None:
        log.info(f"Generating cnn embeddings from {cnn_embed_type} "
                 f"n_leads={in_channels}")

    # Encode all splits; patient splits were fixed during VQ-VAE training so
    # there is no leakage risk — we are only encoding, not fitting.
    loaders = create_ecg_dataloaders(
        h5file_list=h5_files,
        lead_indices=lead_indices,
        segment=segment,
        signal_len=signal_len,
        batch_size=batch_size,
    )

    accumulate_keys = ["patient_ids", "study_ids", "beat_idxs", "token_indices"]
    if cnn_embed_type is not None:
        accumulate_keys.append("embeddings")
    
    # Define and save or load and rocket_extractor
    if cnn_embed_type=='rocket':
        rocket_extractor = get_or_create_rocket_extractor(segment, in_channels, 
                                                          signal_len, device)
    else:
        rocket_extractor = None

    all_results: dict[str, list] = {k: [] for k in accumulate_keys}

    for split_name in ["train", "val", "test"]:
        log.info(f"Encoding {split_name} split...")
        split_results = assign_codebook_indices(model, loaders[split_name], device, cnn_embed_type, cnn_embed_scale, rocket_extractor)
        for k in accumulate_keys:
            if k in split_results:
                v = split_results[k]
                all_results[k].append(v if isinstance(v, np.ndarray) else np.array(v))

    merged = {k: np.concatenate(all_results[k]) for k in accumulate_keys}
    save_assignments(merged, segment, output_key, codebook_size)
    return merged, codebook_size


def create_assignments_sentences(
        h5filelist: list[str],
        in_channels: int,
        lead_indices: list[int],
        cnn_embed_type: str,
        cnn_embed_scale: float = 1.0,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        batch_size: int = 512,
    ) -> list[str]:

    # Return if it already exists
    sentence_output_key = f"{base_prefix}/bert/beat_sentences_nleads_{in_channels}_ws{world_size:04d}_rank{rank:04d}.npz"
    if s3_fs.exists(f"s3://{bucket_out}/{sentence_output_key}"):
        try:
            with s3_fs.open(f"s3://{bucket_out}/{sentence_output_key}", "rb") as f:
                np.load(io.BytesIO(f.read()), allow_pickle=True)
            log.info(f"Rank {rank}: sentence shard already exists — skipping")
            return [sentence_output_key]
        except Exception as e:
            log.info(f"Rank {rank}: sentence shard exists but failed to load ({e!r}) — recomputing")

    # Each rank processes a disjoint subset of H5 files.
    # All segments are handled per rank, so no cross-rank join is needed.
    my_files = h5filelist[rank::world_size]
    log.info(f"Rank {rank}/{world_size}: processing {len(my_files)}/{len(h5filelist)} files")

    assignments: dict[str, dict] = {}
    codebook_sizes: dict[str, int] = {}
    device = torch.device(f"cuda:{local_rank}")
    for seg in SEGMENTS:
        signal_len = SIGNAL_LENS[seg]
        model_key = f"{base_prefix}/segment_{seg}_nleads_{in_channels}/vqvae_model_{seg}.pkl"
        seg_output_key = (
            f"{base_prefix}/assignments/"
            f"segment_{seg}_nleads_{in_channels}_ws{world_size:04d}_rank{rank:04d}.npz"
        )
        assignments[seg], codebook_sizes[seg] = run_assignment(
            my_files, seg, model_key, seg_output_key, signal_len,
            in_channels, lead_indices, batch_size, device, cnn_embed_type, cnn_embed_scale
        )

    # Each rank writes its own shard; all ranks' shards are later gathered.
    sentence_output_key = (
        f"{base_prefix}/bert/beat_sentences_nleads_{in_channels}_ws{world_size:04d}_rank{rank:04d}.npz"
    )
    shard_keys = build_beat_sentences(
        assignments=assignments,
        output_key=sentence_output_key,
        codebook_sizes=codebook_sizes,
        num_shards=1,
    )
    return shard_keys


def all_sentence_shards_exist(
    keys: list[str],
    rank: int,
    world_size: int,
) -> bool:
    """Check whether a beat_sentences npz file exists for every rank."""
    files_exist = True
    if rank == 0:
        for key in keys:
            try:
                s3_client.head_object(Bucket=bucket_out, Key=key)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    files_exist = False
                    break
                else:
                    raise  # some other S3 error (permissions, etc.) 

    # Broadcast from rank 0 to all other ranks — no S3 read needed on non-0 ranks.
    payload = [files_exist]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    files_exist = payload[0]
    return files_exist


def _load_saved_assignment(output_key: str, bucket: str = bucket_out) -> tuple[dict, int]:
    """Load a previously saved per-segment assignment NPZ from S3.

    Returns (assignment_dict, codebook_size) using the same keys that
    assign_codebook_indices() produces, so callers can't tell whether the
    result came from disk or from a fresh forward pass.
    """
    s3_path = f"s3://{bucket}/{output_key}"
    with s3_fs.open(s3_path, "rb") as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)

    assignment = {
        "patient_ids":   data["patient_id"].astype(str),
        "study_ids":     data["study_id"].astype(str),
        "beat_idxs":     data["beat_idx"].astype(np.int32),
        "token_indices": data["token_index"].astype(np.int16),
    }
    if "embeddings" in data.files:
        assignment["embeddings"] = data["embeddings"]

    codebook_size = int(np.array(data["codebook_size"]).reshape(-1)[0])
    return assignment, codebook_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VQ-VAE codebook and beat sentences construction")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--cnn_embed_type", type=str, default=None, choices=['vqvae_encoder', 'rocket'])
    parser.add_argument("--cnn_embed_scale", type=float, default=1.0)
    parser.add_argument("--num_files", type=int,
                        help="Limit to first N h5 files for fast testing")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])   # set by torchrun
    torch.cuda.set_device(local_rank)         # <-- must happen before init_process_group's first collective

    dist.init_process_group("nccl", timeout=timedelta(hours=4))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_files = args.num_files
    allh5files = list_h5_files()
    h5filelist = allh5files[:num_files] if num_files and num_files>0 else allh5files
    log.info(f"Testing with {len(h5filelist)} file(s)")

    batch_size  = args.batch_size
    in_channels = args.in_channels
    cnn_embed_type = args.cnn_embed_type
    cnn_embed_scale = args.cnn_embed_scale

    if in_channels == 1:
        lead_indices = [1]  # lead II
    elif in_channels == NLEADS:
        lead_indices = list(range(NLEADS))
    else:
        raise ValueError("in_channels does not match lead II or all leads")

    # Beat sentence key list in s3
    all_shard_keys = [
        f"{base_prefix}/bert/beat_sentences_nleads_{in_channels}_ws{world_size:04d}_rank{r:04d}.npz" 
        for r in range(world_size)
    ]

    if all_sentence_shards_exist(all_shard_keys, rank, world_size):
         log.info(f"All {world_size} beat-sentence shards already exist in S3 — skipping creation.")
         # Use the key list directly rather than calling the function
    else:
        my_shard_keys = create_assignments_sentences(
            h5filelist=h5filelist,
            in_channels=in_channels,
            lead_indices=lead_indices,
            cnn_embed_type=cnn_embed_type,
            cnn_embed_scale=cnn_embed_scale,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            batch_size=batch_size,
        )
        # all_shard_keys is already fully known deterministically — no need to
        # gather my_shard_keys from other ranks.
        assert my_shard_keys == [all_shard_keys[rank]], (
            f"rank {rank}: shard key mismatch — check output_key naming"
        )
    dist.barrier()  # ensure all shards are written before any rank starts reading

    train_loader, val_loader = create_bert_dataloaders(
        all_shard_keys=all_shard_keys,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        train_ratio=0.85,
    )

    train_batch = next(iter(train_loader))
    val_batch   = next(iter(val_loader))
    log.info(f"Shape : {train_batch['input_ids'].shape}")
    log.info(f"Example input ids: {train_batch['input_ids'][0].tolist()}")
    log.info(f"Example attention mask: {train_batch['attention_mask'][0].tolist()}")
    log.info(f"Example token type ids: {train_batch['token_type_ids'][0].tolist()}")
    log.info(f"Example num_beats: {train_batch['num_beats'][0]}")
    log.info(f"train size     : {len(train_batch['input_ids']):,}")
    if "cnn_embeddings" in train_batch:
        emb = train_batch["cnn_embeddings"][0].float()
        log.info(
            f"    cnn_emb    : shape={tuple(emb.shape)}"
            f"  mean={emb.mean():.4f}  std={emb.std():.4f}"
        )
    log.info(f"val size       : {len(val_batch['input_ids']):,}")
