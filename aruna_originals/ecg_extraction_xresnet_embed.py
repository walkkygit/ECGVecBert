"""
ECG xresnet101 Embedding Script — SageMaker Processing Job
============================================================
Separate, standalone job. Reuses read/clean/delineate helpers from
ecg_extraction.py but does not use its waveform HDF5 files or manifest.

Re-derives the cleaned record and segment onset/offset bounds directly
(cheap — no ProcessPoolExecutor coordination shared with the waveform job),
then computes one xresnet101 embedding per (beat, segment) via embed-once +
ROI-pooling over the full 10s record.

Output: its own HDF5 shards + its own manifest, keyed by
(patient_id, study_id, beat_idx, segment) for joining against the
waveform manifest downstream.

Usage (local)
-------------
    python ecg_extraction_xresnet_embed.py \
        --s3_bucket "walkky-datasets" \
        --s3_prefix_in "mimic-iv/files_unzip/" \
        --s3_prefix_out "extracted-ecg-embeddings/mimic-iv/xresnet101_embeddings" \
        --xresnet_model_path "s3://walkky-ml/aruna-files/vqvae/models/fastai_xresnet1d101.pth" \
        --workers 32
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "neurokit2>=0.2.7", "wfdb>=4.1.0", "hdf5plugin>=4.4.0",
])

import argparse
import io
import json
import logging
import math
import os
import re
import shutil
import time
import itertools
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlparse

import boto3
import h5py
import hdf5plugin
import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from tqdm import tqdm

from models.xresnet1d import xresnet1d101

# Reuse everything possible from the existing extraction script — no
# duplication of read/clean/delineate logic, no changes to that file.
from ecg_extraction import (
    _read_data,
    _list_all_files,
    ECG_SEGMENT_KEYS,
    NUM_LEADS,
    LEAD_II_IDX,
    SAMPLING_RATE,
    MIN_R_PEAKS,
    SEGMENT_TYPES,          # ["P", "PQ", "QRS", "ST", "T"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
DS_FACTOR      = 5    # 500 -> 100 Hz
MIN_STEPS      = 3
MARGIN_STEPS   = 2
HDF5_CHUNK     = 1024
MANIFEST_VERSION = 1

# Segment -> (onset_key, offset_key) in ECG_SEGMENT_KEYS' short names.
# Must mirror ecg_extraction.py's segment_bounds_keys exactly (P/PQ/QRS/ST/T,
# no TP) since we need the *same* boundary definitions, just not the
# waveform-extraction side effects.
SEGMENT_BOUND_KEYS = {
    "P":   ("p_onsets",  "p_offsets"),
    "PQ":  ("p_offsets", "qrs_onsets"),
    "QRS": ("qrs_onsets", "qrs_offsets"),
    "ST":  ("qrs_offsets", "t_onsets"),
    "T":   ("t_onsets", "t_offsets"),
}
assert set(SEGMENT_BOUND_KEYS) == set(SEGMENT_TYPES), \
    "SEGMENT_BOUND_KEYS must match ecg_extraction.SEGMENT_TYPES exactly"

_s3_client = None


# ─────────────────────────────────────────────────────────────────────────────
# xresnet101 loading (12-lead joint input, matches pretraining)
# ─────────────────────────────────────────────────────────────────────────────

def _is_state_dict(obj) -> bool:
    return (isinstance(obj, Mapping) and obj and
            all(isinstance(k, str) for k in obj.keys()) and
            all(torch.is_tensor(v) or isinstance(v, torch.nn.Parameter) for v in obj.values()))


def _extract_state_dict(obj, _depth=0, _max_depth=5):
    if _is_state_dict(obj):
        return dict(obj)
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            sd = obj.state_dict()
            if _is_state_dict(sd): return dict(sd)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        for key in ("state_dict","model_state","weights","params","model","net","module","ema","model_ema","teacher","student"):
            if key in obj:
                try:
                    return _extract_state_dict(obj[key], _depth+1, _max_depth)
                except Exception:
                    pass
        if _depth < _max_depth:
            for v in obj.values():
                try:
                    return _extract_state_dict(v, _depth+1, _max_depth)
                except Exception:
                    continue
    if isinstance(obj, (list, tuple)):
        for it in obj:
            try:
                return _extract_state_dict(it, _depth+1, _max_depth)
            except Exception:
                continue
    raise ValueError("Could not find a state_dict in the provided checkpoint.")


def _clean_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        while True:
            if   k.startswith("module."): k = k[7:]
            elif k.startswith("model.") : k = k[6:]
            elif k.startswith("net.")   : k = k[4:]
            else: break
        out[k] = v
    return out


def _load_xresnet_state_dict(model_path: str, s3_client) -> dict:
    if model_path.startswith("s3://"):
        parsed = urlparse(model_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        raw = torch.load(buf, map_location="cpu", weights_only=False)
    else:
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
    return _clean_state_dict_keys(_extract_state_dict(raw))


def _zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where((sd < eps) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(~np.isfinite(mu), 0.0, mu)
    return (x - mu) / (sd + eps)


class ECGEmbeddingExtractor:
    """Embed-once + ROI-pool per (beat, segment). Joint 12-lead input."""

    def __init__(self, model_path: str, s3_client, device: str = "cpu"):
        self.device = torch.device(device)
        model = xresnet1d101(input_channels=NUM_LEADS, num_classes=1)
        sd = _load_xresnet_state_dict(model_path, s3_client)
        ret = model.load_state_dict(sd, strict=False)

        num_children = len(list(model.children()))
        head_prefix = f"{num_children - 1}."
        backbone_missing = [k for k in ret.missing_keys if not k.startswith(head_prefix)]
        backbone_unexpected = [k for k in ret.unexpected_keys if not k.startswith(head_prefix)]
        if backbone_missing or backbone_unexpected:
            raise ValueError(
                f"xresnet BACKBONE mismatch — missing={backbone_missing}, unexpected={backbone_unexpected}"
            )

        self.body = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.body.eval()

        with torch.no_grad():
            a, b = 500, 1000
            f1 = self.body(torch.zeros(1, NUM_LEADS, a, device=self.device))
            f2 = self.body(torch.zeros(1, NUM_LEADS, b, device=self.device))
            self.out_channels = int(f1.shape[1])
            d_in, d_out = (b - a), int(f2.shape[-1] - f1.shape[-1])
            self.eff_stride = int(round(d_in / d_out)) if d_out > 0 else 1

        self.embedding_dim = self.out_channels * 2
        log.info(f"xresnet loaded: C={self.out_channels} eff_stride@100Hz={self.eff_stride} "
                  f"embedding_dim={self.embedding_dim}")

    def _prep_record(self, clean_record_500hz: np.ndarray) -> np.ndarray:
        """clean_record_500hz: [L500, 12] -> [12, L100], per-channel z-scored."""
        rec = clean_record_500hz.T.astype(np.float32)
        rec100 = resample_poly(rec, up=1, down=DS_FACTOR, axis=-1)
        return _zscore_per_channel(rec100).astype(np.float32)

    def get_feature_map(self, clean_record_500hz: np.ndarray) -> torch.Tensor:
        rec100 = self._prep_record(clean_record_500hz)
        x = torch.from_numpy(rec100).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.body(x)  # [1, C, Lf]

    def roi_embed(self, fmap: torch.Tensor, start_500: float, end_500: float) -> np.ndarray:
        Lf = int(fmap.shape[-1])
        s100 = int(round(start_500 / DS_FACTOR))
        e100 = int(round(end_500 / DS_FACTOR))
        if e100 <= s100:
            e100 = s100 + 1

        sF = s100 // self.eff_stride
        eF = math.ceil(e100 / self.eff_stride)
        sF = max(0, sF - MARGIN_STEPS)
        eF = min(Lf, eF + MARGIN_STEPS)
        if (eF - sF) < MIN_STEPS:
            need = MIN_STEPS - (eF - sF)
            sF = max(0, sF - need // 2)
            eF = min(Lf, eF + (need - need // 2))
            if eF <= sF:
                eF = min(Lf, sF + 1)

        region = fmap[:, :, sF:eF]
        region = torch.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0)
        avg = region.mean(dim=-1, keepdim=True)
        mx = region.amax(dim=-1, keepdim=True)
        emb = torch.cat([avg, mx], dim=1).squeeze(0).squeeze(-1)
        return emb.detach().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker HDF5 writer for embeddings only
# ─────────────────────────────────────────────────────────────────────────────

class HDF5EmbeddingWriter:
    """
    One group per segment type: /P/, /PQ/, /QRS/, /ST/, /T/
    Each group: embedding (N, D) float32, patient_id, study_id, beat_idx (N,)
    Beat-grain (NOT lead-grain) — one embedding per (beat, segment), computed
    jointly across all 12 leads.
    """
    def __init__(self, local_path: str, embedding_dim: int):
        self._path = local_path
        self._file = h5py.File(local_path, "a", libver="earliest")
        self._ds: dict[str, dict] = {}
        self._counts: dict[str, int] = {}
        for seg in SEGMENT_TYPES:
            if seg in self._file:
                self._ds[seg] = {k: self._file[seg][k] for k in self._file[seg]}
                self._counts[seg] = self._file[seg]["embedding"].shape[0]
            else:
                self._init_group(seg, embedding_dim)

    def _init_group(self, segment: str, embedding_dim: int):
        grp = self._file.create_group(segment)
        self._ds[segment] = {
            "embedding": grp.create_dataset(
                "embedding", shape=(0, embedding_dim), maxshape=(None, embedding_dim),
                dtype="float32", chunks=(HDF5_CHUNK, embedding_dim),
                **hdf5plugin.LZ4(nbytes=0),
            ),
            "patient_id": grp.create_dataset(
                "patient_id", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(), chunks=(HDF5_CHUNK,)),
            "study_id": grp.create_dataset(
                "study_id", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(), chunks=(HDF5_CHUNK,)),
            "beat_idx": grp.create_dataset(
                "beat_idx", shape=(0,), maxshape=(None,),
                dtype="int32", chunks=(HDF5_CHUNK,)),
        }
        self._counts[segment] = 0

    def write(self, patient_id: str, study_id: str, rows: dict[str, list]):
        """rows[segment] = list of (beat_idx, embedding_np_array)."""
        for seg in SEGMENT_TYPES:
            seg_rows = rows.get(seg, [])
            if not seg_rows:
                continue
            n, cur = len(seg_rows), self._counts[seg]
            ds = self._ds[seg]
            for name in ("embedding", "patient_id", "study_id", "beat_idx"):
                ds[name].resize(cur + n, axis=0)
            ds["embedding"][cur:cur+n] = np.stack([r[1] for r in seg_rows], axis=0)
            ds["patient_id"][cur:cur+n] = [patient_id] * n
            ds["study_id"][cur:cur+n]   = [study_id] * n
            ds["beat_idx"][cur:cur+n]   = [r[0] for r in seg_rows]
            self._counts[seg] += n

    def flush(self):
        self._file.flush()

    def close(self):
        if self._file.id.valid:
            self._file.flush()
            self._file.close()

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker state
# ─────────────────────────────────────────────────────────────────────────────

_worker_writer: "HDF5EmbeddingWriter | None" = None
_worker_embedder: "ECGEmbeddingExtractor | None" = None
_worker_hdf5_path: "str | None" = None
_worker_id: "int | None" = None


def _init_worker(worker_hdf5_dir: str, worker_id: int, xresnet_model_path: str):
    global _worker_writer, _worker_embedder, _worker_hdf5_path, _worker_id, _s3_client
    torch.set_num_threads(1)  # CRITICAL — avoid oversubscription across many processes
    _s3_client = boto3.client("s3")
    _worker_id = worker_id
    _worker_hdf5_path = os.path.join(worker_hdf5_dir, f"embed_worker_{worker_id}.h5")
    _worker_embedder = ECGEmbeddingExtractor(xresnet_model_path, _s3_client, device="cpu")
    _worker_writer = HDF5EmbeddingWriter(_worker_hdf5_path, _worker_embedder.embedding_dim)
    log.info(f"[worker {worker_id}] embedder + writer ready: {_worker_hdf5_path}")


def _init_worker_with_id(worker_hdf5_dir, id_counter_path, xresnet_model_path):
    import fcntl
    with open(id_counter_path, "r+b") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        worker_id = int(fh.read().strip() or 0)
        fh.seek(0)
        fh.write(str(worker_id + 1).encode())
        fh.truncate()
        fcntl.flock(fh, fcntl.LOCK_UN)
    _init_worker(worker_hdf5_dir, worker_id, xresnet_model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file worker: reconstruct bounds directly, compute embeddings
# ─────────────────────────────────────────────────────────────────────────────

def process_one_file(bucket: str, filename: str) -> dict:
    try:
        ecg_data = _read_data(bucket, filename)
    except Exception as e:
        return {"drop_reason": "failed_read", "filename": filename, "error": str(e)}

    if ecg_data is None:
        return {"drop_reason": "empty", "filename": filename}
    if ecg_data.ndim < 2 or ecg_data.shape[1] != NUM_LEADS:
        return {"drop_reason": f"missing_{NUM_LEADS}_leads", "filename": filename}

    m = re.search(r"/p(\d+)/s(\d+)/", filename)
    patient_id = m.group(1) if m else Path(filename).stem
    study_id   = m.group(2) if m else Path(filename).stem

    try:
        clean_ecg_data = np.stack(
            [nk.ecg_clean(ecg_data[:, i], sampling_rate=SAMPLING_RATE) for i in range(NUM_LEADS)],
            axis=1,
        )
        clean_ref_lead = clean_ecg_data[:, LEAD_II_IDX]
    except Exception as e:
        return {"drop_reason": "failed_cleaning", "filename": filename, "error": str(e)}

    try:
        _, rpeaks_info = nk.ecg_peaks(clean_ref_lead, sampling_rate=SAMPLING_RATE)
    except Exception:
        return {"drop_reason": "ecg_peaks_error", "filename": filename}

    if len(rpeaks_info["ECG_R_Peaks"]) < MIN_R_PEAKS:
        return {"drop_reason": "too_few_r_peaks", "filename": filename}

    try:
        _, waves_info = nk.ecg_delineate(
            clean_ref_lead, rpeaks_info, sampling_rate=SAMPLING_RATE, method="dwt", show=False,
        )
    except Exception:
        return {"drop_reason": "ecg_delineate_error", "filename": filename}

    indices = {key: np.array(waves_info[nk2_key], dtype=float)
               for key, nk2_key in ECG_SEGMENT_KEYS.items()}

    num_waves = min(len(indices[name]) for name in set(
        name for pair in SEGMENT_BOUND_KEYS.values() for name in pair
    ))

    if num_waves == 0:
        return {"drop_reason": "no_beat_cycles", "filename": filename}

    # ── one feature map for the whole record ──
    fmap = _worker_embedder.get_feature_map(clean_ecg_data)

    rows: dict[str, list] = {seg: [] for seg in SEGMENT_TYPES}
    n_embedded = 0
    for i in range(num_waves):
        for segment, (start_key, end_key) in SEGMENT_BOUND_KEYS.items():
            start = indices[start_key][i]
            end = indices[end_key][i]
            missing_wave = (
                (np.isnan(float(start)) or np.isnan(float(end)) or (float(start) > float(end)))
                and i > 0 and i < num_waves - 1
            )
            if missing_wave or np.isnan(float(start)) or np.isnan(float(end)) or end <= start:
                continue  # no valid bounds -> no embedding row for this (beat, segment)
            emb = _worker_embedder.roi_embed(fmap, float(start), float(end))
            rows[segment].append((i, emb))
            n_embedded += 1

    if n_embedded == 0:
        return {"drop_reason": "no_beat_cycles", "filename": filename}

    if _worker_writer is None:
        return {"drop_reason": "writer_not_initialized", "filename": filename}

    _worker_writer.write(patient_id, study_id, rows)

    return {
        "patient_id": patient_id,
        "study_id": study_id,
        "filename": filename,
        "n_beats": num_waves,
        "n_embedded_rows": n_embedded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest (separate key/prefix — does not touch the waveform manifest)
# ─────────────────────────────────────────────────────────────────────────────

def _manifest_s3_key(s3_prefix_out: str) -> str:
    return f"{s3_prefix_out}__shard_manifest.json"

def list_worker_files(s3, bucket: str, prefix_in: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_in):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".h5") and "__embed_worker_" in obj["Key"]:
                keys.append(obj["Key"])
    return sorted(keys)

def _inspect_h5_segment_rows(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return {seg: int(f[seg]["embedding"].shape[0]) for seg in SEGMENT_TYPES if seg in f}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding_job(
    s3_bucket: str,
    s3_prefix_in: str,
    s3_prefix_out: str,
    xresnet_model_path: str,
    local_hdf5: str,
    workers: int = 32,
) -> dict:
    file_counts = {
        "failed_read": 0, "empty": 0, f"missing_{NUM_LEADS}_leads": 0,
        "failed_cleaning": 0, "ecg_peaks_error": 0, "too_few_r_peaks": 0,
        "ecg_delineate_error": 0, "no_beat_cycles": 0,
        "writer_not_initialized": 0, "accepted": 0,
    }
    n_embedded_total = 0

    worker_hdf5_dir = os.path.join(local_hdf5, "embed_workers")
    os.makedirs(worker_hdf5_dir, exist_ok=True)

    global _s3_client
    _s3_client = boto3.client("s3")
    t0 = time.time()

    id_counter_path = os.path.join(worker_hdf5_dir, "_worker_id_counter.txt")
    with open(id_counter_path, "w") as fh:
        fh.write("0")

    dat_file_names = _list_all_files(s3_bucket, s3_prefix_in)
    log.info(f"Found {len(dat_file_names):,} .dat files")

    MAX_INFLIGHT = workers * 2

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_with_id,
        initargs=(worker_hdf5_dir, id_counter_path, xresnet_model_path),
    ) as pool:
        file_iter = iter(dat_file_names)
        total = len(dat_file_names)
        pending = {}
        for fn in itertools.islice(file_iter, MAX_INFLIGHT):
            fut = pool.submit(process_one_file, s3_bucket, fn)
            pending[fut] = fn

        pbar = tqdm(total=total, desc="ECG embedding")
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    del pending[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.warning(f"Unexpected worker error: {exc}")
                        file_counts["failed_read"] += 1
                    else:
                        if "drop_reason" in result:
                            dr = result["drop_reason"]
                            file_counts[dr] = file_counts.get(dr, 0) + 1
                        else:
                            file_counts["accepted"] += 1
                            n_embedded_total += result["n_embedded_rows"]
                    pbar.update(1)
                    try:
                        next_fn = next(file_iter)
                        new_fut = pool.submit(process_one_file, s3_bucket, next_fn)
                        pending[new_fut] = next_fn
                    except StopIteration:
                        pass
        finally:
            pbar.close()

    # ── close + upload worker files ──
    for p in Path(worker_hdf5_dir).glob("embed_worker_*.h5"):
        pass  # workers already closed their own writer via process exit; ensure flush below

    worker_paths = sorted(Path(worker_hdf5_dir).glob("embed_worker_*.h5"))
    shards = []
    for wp in worker_paths:
        try:
            segment_rows = _inspect_h5_segment_rows(str(wp))
        except Exception as e:
            log.error(f"Cannot inspect {wp.name}: {e}")
            continue
        if not any(v > 0 for v in segment_rows.values()):
            continue
        m = re.match(r"embed_worker_(\d+)", wp.stem)
        worker_id = int(m.group(1))
        s3_key = f"{s3_prefix_out}__embed_worker_{worker_id}.h5"
        log.info(f"Uploading {wp.name} -> s3://{s3_bucket}/{s3_key}")
        _s3_client.upload_file(str(wp), s3_bucket, s3_key)
        shards.append({"s3_key": s3_key, "worker_id": worker_id, "segment_rows": segment_rows})

    manifest = {
        "version": MANIFEST_VERSION,
        "bucket": s3_bucket,
        "s3_prefix_out": s3_prefix_out,
        "xresnet_model_path": xresnet_model_path,
        "in_channels": NUM_LEADS,
        "segment_types": SEGMENT_TYPES,
        "shards": sorted(shards, key=lambda s: s["worker_id"]),
    }
    _s3_client.put_object(
        Bucket=s3_bucket,
        Key=_manifest_s3_key(s3_prefix_out),
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"Embedding manifest uploaded ({len(manifest['shards'])} shards)")

    elapsed = (time.time() - t0) / 60
    log.info("=" * 55)
    log.info("  xresnet101 Embedding Summary")
    log.info("=" * 55)
    log.info(f"  Elapsed              : {elapsed:.1f} min")
    for k, v in file_counts.items():
        log.info(f"  {k:22s}: {v:,}")
    log.info(f"  Total embedded rows  : {n_embedded_total:,}")
    log.info("=" * 55)

    shutil.rmtree(worker_hdf5_dir, ignore_errors=True)
    return file_counts


def _parse_args():
    p = argparse.ArgumentParser(description="ECG xresnet101 embedding job")
    p.add_argument("--s3_bucket", default="walkky-datasets")
    p.add_argument("--s3_prefix_in", default="mimic-iv/files_unzip/")
    p.add_argument("--s3_prefix_out", default="extracted-ecg-embeddings/mimic-iv/xresnet101_embeddings")
    p.add_argument("--xresnet_model_path", required=True)
    p.add_argument("--local_hdf5", default="/opt/ml/processing/output/embeddings")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_embedding_job(
        s3_bucket=args.s3_bucket,
        s3_prefix_in=args.s3_prefix_in,
        s3_prefix_out=args.s3_prefix_out,
        xresnet_model_path=args.xresnet_model_path,
        local_hdf5=args.local_hdf5,
        workers=args.workers,
    )