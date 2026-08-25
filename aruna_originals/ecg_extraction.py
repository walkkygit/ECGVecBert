"""
ECG Extraction Script — SageMaker Processing Job
=================================================
Entry point for sagemaker.processing.ScriptProcessor or SKLearnProcessor.

------------------------------
1. Truncate + zero-pad :
     waveforms longer than TARGET_LEN are truncated from the right;
     waveforms shorter are zero-padded on the right.
2. HDF5 written by each worker (rotated to S3 when a shard exceeds the local size limit)
3. Shard manifest JSON on S3 listing every worker shard (including rotations)
4. argparse entry point so SageMaker can pass hyperparameters / paths.

Training jobs should read the shard manifest directly.

Usage (local)
-------------
    python ecg_extraction.py \
    --s3_bucket "walkky-datasets" \
    --s3_prefix_in "mimic-iv/files_unzip/" \
    --s3_prefix_out "extracted-ecg-waveforms/mimic-iv/extracted_waveforms.h5" \
    --workers 32 \

SageMaker launcher: see ecg_extraction_launcher.py
"""


# Run pip install
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "neurokit2>=0.2.7",
    "wfdb>=4.1.0",
    "hdf5plugin>=4.4.0",
])

import argparse
import json
import logging
import os
import tempfile
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import wfdb
import boto3
import h5py
import hdf5plugin
import shutil
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import itertools
from concurrent.futures import wait, FIRST_COMPLETED
import random
import atexit
import fcntl
import glob
from botocore.exceptions import ClientError

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── tuneable constants (overridable via argparse) ─────────────────────────────
SAMPLING_RATE     = 500
MIN_R_PEAKS       = 4
LEAD_II_IDX       = 1
NUM_LEADS         = 12
HDF5_CHUNK        = 1024

TARGET_LEN = 800     # Target waveform length

ECG_SEGMENT_KEYS = {
    "p_onsets":   "ECG_P_Onsets",
    "p_offsets":  "ECG_P_Offsets",
    "qrs_onsets": "ECG_R_Onsets",
    "qrs_offsets":"ECG_R_Offsets",
    "t_onsets":   "ECG_T_Onsets",
    "t_offsets":  "ECG_T_Offsets",
}


SEGMENT_TYPES = ["P", "PQ", "QRS", "ST", "T", "TP"]
ALL_COLS = ("data", "patient_id", "study_id", "beat_idx", "lead_idx", "original_len")
MANIFEST_VERSION = 1
_s3_client = None


# ─────────────────────────────────────────────────────────────────────────────
# Read file list
# ─────────────────────────────────────────────────────────────────────────────


# Function to list all files in nested folders automatically
def _list_all_files(bucket: str, prefix: str) -> list:

    files = []
    paginator = _s3_client.get_paginator('list_objects_v2') #gives an iterator that automatically requests every page
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        files.extend([content['Key'] for content in page.get('Contents', [])]) #add full path of each object to files
    dat_file_names = [file for file in files if file.endswith('.dat')]
    # Sort alphabetically first to guarantee a deterministic baseline
    dat_file_names.sort()
    random.seed(42)
    random.shuffle(dat_file_names)
    return dat_file_names


# Function to read ECG signal data as a NumPy array
def _read_data(bucket: str, dat_key: str) -> np.ndarray | None:
    hea_key = dat_key.replace(".dat", ".hea")
    base = os.path.basename(dat_key).split(".")[0]
    with tempfile.TemporaryDirectory() as tmp:
        for key, name in [(dat_key, f"{base}.dat"), (hea_key, f"{base}.hea")]:
            _s3_client.download_file(bucket, key, os.path.join(tmp, name))
        try:
            rec = wfdb.rdrecord(os.path.join(tmp, base))
            return rec.p_signal
        except Exception as e:
            log.warning(f"wfdb failed on {dat_key}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def _z_normalized(seg: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance. Handles flat lines without crashing."""
    seg = seg.astype(np.float32)
    std = seg.std()
    return (seg - seg.mean()) / std if std > 0 else seg - seg.mean()


def _extract_waveform(
    signal: np.ndarray,
    start: int,
    end: int,
    target_len: int,
) -> tuple[np.ndarray, int] | None:
    """
    Extract a waveform by truncating or zero-padding to target_len.

    - Waveform longer than target_len  → keep first target_len samples.
    - Waveform shorter than target_len → right-pad with zeros.

    Returns None for degenerate slices (empty, out-of-bounds, or flat).
    """
    if (np.isnan(float(start)) or np.isnan(float(end))
            or end <= start
            or start < 0
            or end > len(signal)):
        return None
    start = int(start)
    end = int(end)
    raw = signal[start:end].astype(np.float32)
    if len(raw) < 2:
        return None

    actual_len = len(raw)
    if actual_len >= target_len:
        # Truncate and normalize
        normalized = _z_normalized(raw[:target_len])
    else:
        # Normalize and zero-pad on the right
        normalized = np.zeros(target_len, dtype=np.float32)
        normalized[:actual_len] = _z_normalized(raw)

    return normalized, actual_len


def _extract_all_waveforms(
    clean_ecg_data: np.ndarray,
    indices: dict,
) -> tuple[list[dict], int]:
    """
    Extract every beat's waveforms from all leads of a single ECG record,
    using delineation indices derived from the delineation lead (lead II by default).

    A heart beat consists of P wave - PQ segment - QRS complex - ST segment - T wave
    - TP segment

    The delineation boundaries from lead II are applied to every lead.
    A cycle is only kept if ALL segments are valid on ALL leads.
    Segments with NaN or messy onsets/offsets are zero-filled with length set to 0,
    only for internal beats, excluding the first and last beats which may have NaN endpoints

    Beat cycles are keyed as:
        cycle[segment][lead_idx] = np.ndarray of shape (TARGET_LEN,)

    The returned beat_idx values reflect the original position of each beat
    in the full sequence (i.e. gaps are preserved when cycles are skipped).

    Parameters
    ----------
    clean_ecg_data : np.ndarray, shape (num_samples, num_leads)
        Cleaned multi-lead ECG signal (every lead filtered with nk.ecg_clean).
    indices : dict
        Delineation boundary arrays (from lead II).

    Returns
    -------
    beat_cycles : list of dicts
        Each dict has keys:
            "beat_idx"  : int   - original 0-based position in the beat sequence
            "segments"  : dict  - segment → {lead_idx → waveforms}
            "original_lengths": dict - segment → original segment lengths
    skipped_cycles_count : int
    """

    beat_cycles = []
    skipped_cycles_count = 0
    num_leads = NUM_LEADS

    segment_bounds = {
        "P":   ("p_onsets",   "p_offsets"),
        "PQ":  ("p_offsets",  "qrs_onsets"),
        "QRS": ("qrs_onsets",    "qrs_offsets"),
        "ST":  ("qrs_offsets",    "t_onsets"),
        "T":   ("t_onsets",   "t_offsets"),
        "TP": ("t_offsets",   "p_onsets"),
    }

    num_waves = min(len(indices[name]) for name in set(
        name for namepair in segment_bounds.values() for name in namepair
        ))

    for i in range(num_waves-1):
        cycle_segments: dict[str, dict] = {}
        original_lengths: dict[str, int] = {}
        skip_cycle = False
        for segment, (start_key, end_key) in segment_bounds.items():
            if segment != "TP":
                start = indices[start_key][i]
                end = indices[end_key][i]
            else:
                start = indices[start_key][i]
                end = indices[end_key][i+1]

            missing_wave = (
                (np.isnan(float(start)) or np.isnan(float(end)) or (float(start)>float(end)))
                and i > 0
                and i < num_waves-1
            )
            segment_original_length = 0
            lead_waveforms: dict[int, np.ndarray] = {}
            for lead_idx in range(num_leads):
                if missing_wave:
                    lead_waveforms[lead_idx] = np.zeros(TARGET_LEN, dtype=np.float32)
                else:
                    extracted = _extract_waveform(clean_ecg_data[:, lead_idx],
                                                      start, end, TARGET_LEN)
                    if extracted is not None:
                        w, actual_len = extracted
                        lead_waveforms[lead_idx] = w
                        if lead_idx == LEAD_II_IDX:
                            segment_original_length = actual_len
                            # or (end-start) for any lead
                    else:
                        skip_cycle = True
                        break
            if not skip_cycle:
                cycle_segments[segment] = lead_waveforms
                original_lengths[segment] = segment_original_length
            else:
                break
        if not skip_cycle:
            beat_cycles.append({"beat_idx": i,
                                "segments": cycle_segments,
                                "original_lengths": original_lengths})
        else:
            skipped_cycles_count += 1

    return beat_cycles, skipped_cycles_count


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────
 
class HDF5WaveformWriter:
    """
    Appends ECG beat segments to a local HDF5 file.
 
    Layout
    ------
    One top-level group per segment type: /P/, /PQ/, /QRS/, /ST/, /T/, /TP/
 
    Each group contains row-aligned datasets:
        data         (N, TARGET_LEN)  float32
        patient_id   (N,)             str
        study_id     (N,)             str
        beat_idx     (N,)             int32  — 0-based beat position in record
        lead_idx     (N,)             int32  — 0..11
        original_len (N,)             int32  — pre-pad/truncate length
    """
 
    def __init__(self, local_path: str):
        self._path     = local_path
        self._file     = h5py.File(local_path, "a", libver="earliest")  # append so workers reuse
        self._ds:      dict[str, dict] = {}
        self._counts:  dict[str, int]  = {}
        # Pre-create all segment groups (rebuilding counts if file exists)
        for seg in SEGMENT_TYPES:
            if seg in self._file:
                self._ds[seg] = {k: self._file[seg][k] for k in self._file[seg]}
                self._counts[seg] = self._file[seg]["data"].shape[0]
            else:
                self._init_group(seg)
 
    def _init_group(self, segment: str):
        grp = self._file.create_group(segment)
        self._ds[segment] = {
            "data": grp.create_dataset(
                "data",
                shape=(0, TARGET_LEN), maxshape=(None, TARGET_LEN),
                dtype="float32",
                chunks=(HDF5_CHUNK, TARGET_LEN),
                **hdf5plugin.LZ4(nbytes=0) 
            ),
            "patient_id": grp.create_dataset(
                "patient_id", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(), chunks=(HDF5_CHUNK,),
            ),
            "study_id": grp.create_dataset(
                "study_id", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(), chunks=(HDF5_CHUNK,),
            ),
            "beat_idx": grp.create_dataset(
                "beat_idx", shape=(0,), maxshape=(None,),
                dtype="int32", chunks=(HDF5_CHUNK,),
            ),
            "original_len": grp.create_dataset(
                "original_len", shape=(0,), maxshape=(None,),
                dtype="int32", chunks=(HDF5_CHUNK,),
            ),
            "lead_idx": grp.create_dataset(
                "lead_idx", shape=(0,), maxshape=(None,),
                dtype="int32", chunks=(HDF5_CHUNK,),
            ),
        }
        self._counts[segment] = 0
 
    def write_record(
        self,
        patient_id:  str,
        study_id:    str,
        beat_cycles: list[dict],
    ):
        if not beat_cycles:
            return
 
        first_segs  = beat_cycles[0]["segments"]
        lead_indices = sorted(next(iter(first_segs.values())).keys())
        num_leads   = len(lead_indices)
        n_beats     = len(beat_cycles)
        n_rows      = n_beats * num_leads
 
        for seg in SEGMENT_TYPES:

            cur = self._counts[seg]
 
            batch_data       = np.zeros((n_rows, TARGET_LEN), dtype=np.float32)
            batch_beat_idx   = np.empty(n_rows, dtype=np.int32)
            batch_orig_len   = np.empty(n_rows, dtype=np.int32)
            batch_lead_idx   = np.empty(n_rows, dtype=np.int32)
            batch_patient_id = [patient_id] * n_rows
            batch_study_id   = [study_id]   * n_rows
 
            row = 0
            for cycle in beat_cycles:
                orig_beat_idx = cycle["beat_idx"]
                orig_seg_len  = cycle["original_lengths"][seg]
                for lead in lead_indices:
                    waveform = cycle["segments"][seg].get(
                        lead, np.zeros(TARGET_LEN, dtype=np.float32)
                    )
                    batch_data[row]     = waveform
                    batch_beat_idx[row] = orig_beat_idx
                    batch_orig_len[row] = orig_seg_len
                    batch_lead_idx[row] = lead
                    row += 1
 
            ds = self._ds[seg]
            for name in ("data", "patient_id", "study_id",
                         "beat_idx", "original_len", "lead_idx"):
                ds[name].resize(cur + n_rows, axis=0)
 
            ds["data"][cur:cur+n_rows]        = batch_data
            ds["patient_id"][cur:cur+n_rows]  = batch_patient_id
            ds["study_id"][cur:cur+n_rows]    = batch_study_id
            ds["beat_idx"][cur:cur+n_rows]    = batch_beat_idx
            ds["lead_idx"][cur:cur+n_rows]    = batch_lead_idx
            ds["original_len"][cur:cur+n_rows] = batch_orig_len
 
            self._counts[seg] += n_rows
 
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
# Per-worker persistent writer  (process-global state)
# ─────────────────────────────────────────────────────────────────────────────

# Worker
_worker_writer: "HDF5WaveformWriter | None" = None
_worker_hdf5_path: "str | None" = None
_worker_hdf5_dir: "str | None" = None
_worker_id: "int | None" = None
_worker_shard_idx: int = 0
_worker_s3_bucket: "str | None" = None
_worker_s3_prefix: "str | None" = None


def _init_worker(worker_hdf5_dir: str, worker_id: int, s3_bucket: str, s3_prefix_out: str) -> None:
    global _worker_writer, _worker_hdf5_path, _worker_hdf5_dir, _s3_client, _worker_id
    global _worker_shard_idx, _worker_s3_bucket, _worker_s3_prefix
    _s3_client = boto3.client("s3")
    _worker_id = worker_id
    _worker_hdf5_dir = worker_hdf5_dir
    _worker_shard_idx = 0
    _worker_s3_bucket = s3_bucket
    _worker_s3_prefix = s3_prefix_out
    _worker_hdf5_path = os.path.join(worker_hdf5_dir, f"worker_{worker_id}.h5")
    _worker_writer = HDF5WaveformWriter(_worker_hdf5_path)
    atexit.register(_close_active_worker_writer)
    log.info(f"[worker {worker_id}] HDF5 writer opened: {_worker_hdf5_path}")


def _init_worker_with_id(worker_hdf5_dir: str, id_counter_path: str, s3_bucket: str, s3_prefix_out: str) -> None:
    try:
        with open(id_counter_path, "r+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            worker_id = int(fh.read().strip() or 0)
            fh.seek(0)
            fh.write(str(worker_id + 1).encode())
            fh.truncate()
            fcntl.flock(fh, fcntl.LOCK_UN)
        _init_worker(worker_hdf5_dir, worker_id, s3_bucket, s3_prefix_out)
    except Exception as exc:
        # Initializer exceptions propagate to the main process as BrokenProcessPool.
        log.error(f"Worker initializer failed: {exc}", exc_info=True)
        raise


def _rotate_if_needed():
    global _worker_writer, _worker_hdf5_path, _worker_shard_idx

    MAX_LOCAL_HDF5_BYTES = 10 * 1024**3

    if _worker_writer is None or not _worker_hdf5_path or not os.path.exists(_worker_hdf5_path):
        return
    # Flush HDF5 internal buffers to disk before measuring — otherwise
    # os.path.getsize can undercount and rotation never triggers.
    _worker_writer.flush()
    size = os.path.getsize(_worker_hdf5_path)
    if size < MAX_LOCAL_HDF5_BYTES:
        return

    if not _worker_s3_bucket or not _worker_s3_prefix:
        raise RuntimeError(
            f"[worker {_worker_id}] S3 bucket/prefix not set — cannot rotate shard"
        )

    segment_rows = dict(_worker_writer.counts)
    _worker_writer.close()
    _worker_writer = None

    s3_key = f"{_worker_s3_prefix}__worker_{_worker_id}_shard_{_worker_shard_idx}.h5"
    log.info(
        f"[worker {_worker_id}] Rotating: uploading {size/1e9:.1f}GB shard to "
        f"s3://{_worker_s3_bucket}/{s3_key}"
    )
    try:
        _s3_client.upload_file(_worker_hdf5_path, _worker_s3_bucket, s3_key)
    except Exception as exc:
        log.error(
            f"[worker {_worker_id}] Shard upload failed: {exc}. "
            f"Shard preserved at {_worker_hdf5_path} — main process will recover it."
        )
        # Don't remove the sealed shard — main process upload loop will find
        # and upload it via the worker_*.h5 glob.
        # Reopen at a new path so this worker keeps processing.
        _worker_shard_idx += 1
        _worker_hdf5_path = os.path.join(
            _worker_hdf5_dir, f"worker_{_worker_id}_recovery_{_worker_shard_idx}.h5"
        )
        _worker_writer = HDF5WaveformWriter(_worker_hdf5_path)
        return  # Manifest reconciliation happens via S3 scan in collect_shard_manifest
    # Upload succeeded
    _append_rotated_shard_record(s3_key, segment_rows)

    os.remove(_worker_hdf5_path)
    _worker_shard_idx += 1
    # Generate a new path for the next shard
    _worker_hdf5_path = os.path.join(
        _worker_hdf5_dir, f"worker_{_worker_id}_shard_{_worker_shard_idx}.h5"
    )
    _worker_writer = HDF5WaveformWriter(_worker_hdf5_path)


def _log_disk_usage(path: str):
    total, used, free = shutil.disk_usage(path)
    log.info(
        f"Disk usage at {path}: "
        f"{used/1e9:.1f}GB used / {total/1e9:.1f}GB total "
        f"({free/1e9:.1f}GB free)"
    )


def _close_active_worker_writer() -> None:
    """Close whichever HDF5 writer is currently active (handles rotation)."""
    global _worker_writer
    if _worker_writer is not None:
        try:
            _worker_writer.close()
        except Exception as e:
            log.warning(f"[worker {_worker_id}] Error closing HDF5 writer: {e}")
        _worker_writer = None


def _append_rotated_shard_record(s3_key: str, segment_rows: dict[str, int]) -> None:
    """Append one uploaded shard entry to this worker's JSONL sidecar."""
    path = _worker_shard_manifest_path(_worker_hdf5_dir, _worker_id)
    record = {
        "s3_key": s3_key,
        "worker_id": _worker_id,
        "shard_idx": _worker_shard_idx,
        "segment_rows": segment_rows,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(record) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


# ─

def _clear_h5_consistency_flags(path: str) -> None:
    """Run h5clear on a file to reset SWMR/consistency flags left by a crashed writer."""
    try:
        result = subprocess.run(
            ["h5clear", "-s", path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            log.warning(f"h5clear failed on {path}: {result.stderr.strip()}")
        else:
            log.info(f"h5clear succeeded on {path}")
    except FileNotFoundError:
        # h5clear not on PATH — fall back to python-only approach
        log.warning("h5clear not found; skipping flag clear (may still fail to open)")
    except Exception as e:
        log.warning(f"h5clear error on {path}: {e}")


def _inspect_h5_segment_rows(path: str, max_retries: int = 3, retry_delay: float = 10.0) -> dict[str, int]:
    """
    Return row counts per segment group for a closed worker shard file.
    Retries with h5clear if the file has stale write-consistency flags.
    """
    for attempt in range(max_retries):
        try:
            with h5py.File(path, "r", swmr=False) as f:
                return {
                    seg: int(f[seg]["data"].shape[0])
                    for seg in SEGMENT_TYPES
                    if seg in f and "data" in f[seg]
                }
        except OSError as e:
            msg = str(e)
            if "file is already open for write" in msg or "file consistency flags" in msg:
                log.warning(
                    f"[attempt {attempt+1}/{max_retries}] HDF5 consistency flag set on {path} — "
                    f"{'clearing with h5clear' if attempt == 0 else 'retrying after delay'}…"
                )
                if attempt == 0:
                    _clear_h5_consistency_flags(path)
                time.sleep(retry_delay)
            else:
                raise
    # Last attempt — raise so the caller can skip this shard
    raise OSError(f"Cannot open {path} after {max_retries} attempts")


# 

def _wait_for_workers_to_close_hdf5(
    worker_hdf5_dir: str,
    timeout: int = 3600,
    poll_interval: float = 30.0,
):
    """
    Block until all worker HDF5 files can be opened for reading.
    Uses /proc/pid/fd as a fast early-exit, then confirms with h5py open-test.
    """
    time.sleep(5)  # Let atexit handlers start

    deadline = time.time() + timeout

    while time.time() < deadline:
        current_worker_paths = {
            str(p.resolve())
            for p in Path(worker_hdf5_dir).glob("worker_*.h5")
        }
        if not current_worker_paths:
            log.info("No local worker HDF5 files remain — all rotated to S3.")
            return

        # Phase 1: check /proc/fd
        open_files: set[str] = set()
        for fd_path in glob.glob("/proc/*/fd/*"):
            try:
                resolved = os.path.realpath(fd_path)
                if resolved in current_worker_paths:
                    open_files.add(resolved)
            except (OSError, PermissionError):
                pass

        still_open_fd = current_worker_paths & open_files
        if still_open_fd:
            log.info(
                f"Waiting for {len(still_open_fd)} worker HDF5 file(s) "
                f"still open per /proc/fd: "
                + ", ".join(Path(p).name for p in still_open_fd)
            )
            time.sleep(poll_interval)
            continue

        # Phase 2: confirm h5py can actually open each file for reading
        # (catches the case where the fd is gone but SWMR flags are still set)
        unreadable: list[str] = []
        for path in current_worker_paths:
            try:
                with h5py.File(path, "r", swmr=False):
                    pass
            except OSError:
                unreadable.append(path)

        if not unreadable:
            log.info("All worker HDF5 files confirmed readable by h5py.")
            return

        log.info(
            f"Waiting: {len(unreadable)} HDF5 file(s) not yet readable by h5py: "
            + ", ".join(Path(p).name for p in unreadable)
        )
        time.sleep(poll_interval)

    log.warning(f"Timed out after {timeout}s. Proceeding anyway (h5clear will be used per-file).")



# ─────────────────────────────────────────────────────────────────────────────
# Per-file worker  (runs in worker processes)
# ─────────────────────────────────────────────────────────────────────────────
 
def process_one_file(
    bucket:               str,
    filename:             str,
    sampling_rate:        int = SAMPLING_RATE,
    min_r_peaks:          int = MIN_R_PEAKS,
    delineation_lead_idx: int = LEAD_II_IDX,
) -> dict:
    """
    Process one ECG file end-to-end and write waveforms directly to a
    per-worker HDF5 file on local disk.
 
    Returns a dict with either:
        {"drop_reason": str, "filename": str}          — on failure
        {"patient_id", "study_id", "filename",
         "skipped_cycles_count", "n_beats",
         "worker_hdf5": str}                           — on success
    """
    # ── Read ──────────────────────────────────────────────────────────────────

    #t_read  = time.time()
    try:
        ecg_data = _read_data(bucket, filename)
    except Exception as e:
        return {"drop_reason": "failed_read", "filename": filename, "error": str(e)}
 
    if ecg_data is None:
        return {"drop_reason": "empty", "filename": filename,}
 
    if ecg_data.ndim < 2 or ecg_data.shape[1] != NUM_LEADS:
        return {"drop_reason": f"missing_{NUM_LEADS}_leads", "filename": filename,}
 
    # ── IDs ───────────────────────────────────────────────────────────────────
    m = re.search(r"/p(\d+)/s(\d+)/", filename)
    patient_id = m.group(1) if m else Path(filename).stem
    study_id   = m.group(2) if m else Path(filename).stem
 
    # ── Clean ─────────────────────────────────────────────────────────────────
    try:
        clean_ecg_data = np.stack(
            [nk.ecg_clean(ecg_data[:, i], sampling_rate=sampling_rate)
             for i in range(NUM_LEADS)],
            axis=1,
        )
        clean_ref_lead = clean_ecg_data[:, delineation_lead_idx]
    except Exception as e:
        return {"drop_reason": "failed_cleaning", "filename": filename, "error": str(e)}
 
    # ── R-peaks ───────────────────────────────────────────────────────────────
    try:
        _, rpeaks_info = nk.ecg_peaks(clean_ref_lead, sampling_rate=sampling_rate)
    except Exception:
        return {"drop_reason": "ecg_peaks_error", "filename": filename}
 
    r_peaks = rpeaks_info["ECG_R_Peaks"]
    if len(r_peaks) < min_r_peaks:
        return {"drop_reason": "too_few_r_peaks", "filename": filename}
 
    # ── Delineation ───────────────────────────────────────────────────────────
    try:
        _, waves_info = nk.ecg_delineate(
            clean_ref_lead, rpeaks_info,
            sampling_rate=sampling_rate, method="dwt", show=False,
        )
    except Exception:
        return {"drop_reason": "ecg_delineate_error", "filename": filename}
 
    # Store as float64 so np.isnan() works correctly on NaN landmarks.
    indices: dict[str, np.ndarray] = {
        key: np.array(waves_info[nk2_key], dtype=float)
        for key, nk2_key in ECG_SEGMENT_KEYS.items()
    }
 
    # ── Extract waveforms ─────────────────────────────────────────────────────
    #t_proc  = time.time()
    beat_cycles, skipped_cycles_count = _extract_all_waveforms(
        clean_ecg_data, indices
    )
 
    if not beat_cycles:
        return {"drop_reason": "no_beat_cycles", "filename": filename}
 
    # ── Write to the persistent per-worker writer — no open/close/flush here
    if _worker_writer is None:
        return {"drop_reason": "writer_not_initialized", "filename": filename,}
    #t_write = time.time()

    # ── Write — flush is guaranteed even if rotate or something else raises ──
    try:
        _worker_writer.write_record(patient_id, study_id, beat_cycles)
        _rotate_if_needed()
        #t_done = time.time()
    except Exception as exc:
        # Flush whatever was written so HDF5 stays consistent, then re-raise.
        # The future's exception propagates to the main process result handler.
        try:
            _worker_writer.flush()
        except Exception:
            pass
        raise exc
    
    #log.info(
    #    f"[{filename}] "
    #    f"read={t_proc-t_read:.2f}s  "
    #    f"process={t_write-t_proc:.2f}s  "
    #    f"write={t_done-t_write:.2f}s"
    #)
 
    return {
        "patient_id":           patient_id,
        "study_id":             study_id,
        "filename":             filename,
        "skipped_cycles_count": skipped_cycles_count,
        "n_beats":              len(beat_cycles),
        "worker_hdf5":          _worker_hdf5_path,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Shard Manifest
# ─────────────────────────────────────────────────────────────────────────────

def _manifest_s3_key(s3_prefix_out: str) -> str:
    return f"{s3_prefix_out}__shard_manifest.json"


def _worker_shard_manifest_path(worker_hdf5_dir: str, worker_id: int) -> str:
    return os.path.join(worker_hdf5_dir, f"worker_{worker_id}_uploaded_shards.jsonl")


def list_worker_files(s3: object, bucket: str, prefix_in: str) -> list[str]:
    """
    Return all S3 keys matching  <prefix_in>__worker_*.h5
    sorted by worker index so the merge order is deterministic.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_in):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5") and "__worker_" in key:
                keys.append(key)
 
    # Sort by the integer worker index embedded in the filename.
    def _worker_idx(key: str) -> int:
        stem = Path(key).stem  # e.g. "extracted_waveforms__worker_0_shard_2"
        try:
            part = stem.split("__worker_")[-1]      # "0_shard_2"
            worker_num = int(part.split("_shard_")[0])  # 0
            shard_num  = int(part.split("_shard_")[1]) if "_shard_" in part else 0  # 2
            return (worker_num, shard_num)
        except (ValueError, IndexError):
            return (-1, -1)

    return sorted(keys, key=_worker_idx)


def _load_worker_uploaded_shard_records(worker_hdf5_dir: str) -> list[dict]:
    """Load JSONL records written by workers when they rotate shards to S3."""
    records: list[dict] = []
    if not Path(worker_hdf5_dir).exists():
        log.warning(f"worker_hdf5_dir {worker_hdf5_dir} does not exist — skipping JSONL load")
        return records
    for path in sorted(Path(worker_hdf5_dir).glob("worker_*_uploaded_shards.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _shard_sort_key(entry: dict) -> tuple[int, int]:
    worker_id = int(entry["worker_id"])
    shard_idx = entry.get("shard_idx")
    if shard_idx is None:
        # Final per-worker file uploaded by the main process (after rotated shards).
        return (worker_id, 10**9)
    return (worker_id, int(shard_idx))


def _is_nonempty_shard(segment_rows: dict[str, int]) -> bool:
    return any(int(segment_rows.get(seg, 0)) > 0 for seg in SEGMENT_TYPES)


def collect_shard_manifest(
    s3: object,
    bucket: str,
    s3_prefix_out: str,
    worker_hdf5_dir: str,
    final_local_uploads: list[dict],
    target_len: int = TARGET_LEN,
) -> dict:
    """
    Build a canonical shard manifest covering rotated shards and final worker files.

    Parameters
    ----------
    final_local_uploads : list[dict]
        Entries produced when the main process uploads remaining worker_*.h5 files.
        Each dict: {s3_key, worker_id, shard_idx, segment_rows}.
    """
    by_key: dict[str, dict] = {}

    for record in _load_worker_uploaded_shard_records(worker_hdf5_dir):
        by_key[record["s3_key"]] = record

    for record in final_local_uploads:
        by_key[record["s3_key"]] = record

    # Reconcile against S3 (e.g. job restarted, or a worker manifest file was lost).
    for s3_key in list_worker_files(s3, bucket, s3_prefix_out):
        if s3_key in by_key:
            continue
        log.warning(f"Shard {s3_key} found on S3 but not in local manifests — inspecting…")
        local_path = download_to_tempfile(s3, bucket, s3_key)
        try:
            by_key[s3_key] = {
                "s3_key": s3_key,
                "worker_id": _parse_worker_id_from_key(s3_key),
                "shard_idx": _parse_shard_idx_from_key(s3_key),
                "segment_rows": _inspect_h5_segment_rows(local_path),
            }
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    shards = sorted(by_key.values(), key=_shard_sort_key)
    return {
        "version": MANIFEST_VERSION,
        "bucket": bucket,
        "s3_prefix_out": s3_prefix_out,
        "target_len": target_len,
        "segment_types": SEGMENT_TYPES,
        "shards": shards,
    }


def _parse_worker_id_from_key(s3_key: str) -> int:
    stem = Path(s3_key).stem
    part = stem.split("__worker_")[-1]   # "0", "0_shard_1", or "0_recovery_1"
    m = re.match(r"(\d+)", part)
    return int(m.group(1))               # always extracts just the leading digits
    

def _parse_shard_idx_from_key(s3_key: str) -> int | None:
    stem = Path(s3_key).stem
    part = stem.split("__worker_")[-1]
    if "_shard_" in part:
        return int(part.split("_shard_")[1])
    return None


def save_shard_manifest(s3: object, bucket: str, s3_prefix_out: str, manifest: dict) -> str:
    key = _manifest_s3_key(s3_prefix_out)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"Shard manifest uploaded → s3://{bucket}/{key} ({len(manifest['shards'])} shards)")
    return key


def download_to_tempfile(s3: object, bucket: str, key: str) -> str:
    """
    Download an S3 object to a named temporary file and return its path.
    The caller is responsible for deleting the file when done.
    """
    suffix = Path(key).suffix
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    log.info(f"  Downloading s3://{bucket}/{key} → {path}")
    s3.download_file(bucket, key, path)
    return path
 

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_extraction(
    delineation_lead_idx: int = LEAD_II_IDX,
    sampling_rate:        int = SAMPLING_RATE,
    min_r_peaks:          int = MIN_R_PEAKS,
    s3_bucket:            str = "walkky-datasets",
    s3_prefix_in:         str = "mimic-iv/files_unzip/",
    s3_prefix_out:        str = "extracted-ecg-waveforms/mimic-iv/extracted_waveforms",
    local_hdf5:           str = "/opt/ml/processing/output/waveforms",
    workers:              int = 32,
) -> dict:
    file_counts = {
        "failed_read":         0,
        "empty": 0,
        f"missing_{NUM_LEADS}_leads": 0,
        "failed_cleaning": 0,
        "ecg_peaks_error": 0,
        "too_few_r_peaks": 0,
        "ecg_delineate_error": 0,
        "no_beat_cycles": 0,
        "writer_not_initialized": 0,
        "accepted":            0,
    }
    skipped_cycles_count = 0
    n_beats = 0

    worker_hdf5_dir = os.path.join(local_hdf5, "ecg_workers")
    
    os.makedirs(worker_hdf5_dir, exist_ok=True)
    os.makedirs(local_hdf5 or ".", exist_ok=True)

    global _s3_client
    _s3_client = boto3.client("s3")
    t0 = time.time()

    log.info(
        f"Starting extraction: bucket={s3_bucket} prefix={s3_prefix_in} "
        f"workers={workers}"
    )


    id_counter_path = os.path.join(worker_hdf5_dir, "_worker_id_counter.txt")
    with open(id_counter_path, "w") as fh:
        fh.write("0")


    dat_file_names = _list_all_files(s3_bucket, s3_prefix_in)
    log.info(f"Found {len(dat_file_names):,} .dat files")

    # At most this many futures live in memory at once.
    # Small enough to be cheap; large enough to keep all workers busy.
    MAX_INFLIGHT = workers * 2

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_with_id,
        initargs=(worker_hdf5_dir, id_counter_path, s3_bucket, s3_prefix_out),
    ) as pool:
        file_iter = iter(dat_file_names)
        total     = len(dat_file_names)

        # Seed the pool with the initial batch of futures.
        pending: dict = {}
        for fn in itertools.islice(file_iter, MAX_INFLIGHT):
            fut = pool.submit(
                process_one_file,
                s3_bucket, fn,
                sampling_rate, min_r_peaks, delineation_lead_idx,
            )
            pending[fut] = fn

        pbar = tqdm(total=total, desc="ECG files")

        try:
            while pending:
                # Block until at least one future finishes.
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
    
                for fut in done:
                    del pending[fut]
    
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.warning(f"Unexpected worker error: {exc}")
                        file_counts["failed_read"] += 1
                    else:
                        # Handle results
                        if "drop_reason" in result:
                            drop_reason = result["drop_reason"]
                            if drop_reason in file_counts:
                                file_counts[drop_reason] += 1
                            else:
                                log.warning(f"Unknown drop_reason '{drop_reason}' for {result.get('filename')}")
                                file_counts["failed_read"] += 1   # catch-all key
                        else:
                            file_counts["accepted"] += 1
                            skipped_cycles_count += result["skipped_cycles_count"]
                            n_beats += result["n_beats"]
    
                    pbar.update(1)
                    pbar.set_postfix(
                        accepted=file_counts["accepted"],
                        dropped=sum(v for k, v in file_counts.items()
                                    if k != "accepted"),
                    )
                    
                    # Log disk usage periodically
                    if file_counts["accepted"] > 0 and file_counts["accepted"] % 50000 == 0:
                        _log_disk_usage(worker_hdf5_dir)
                    
                    # Refill: submit one new future for each completed one.
                    try:
                        next_fn = next(file_iter)
                        new_fut = pool.submit(
                            process_one_file,
                            s3_bucket, next_fn,
                            sampling_rate, min_r_peaks, delineation_lead_idx,
                        )
                        pending[new_fut] = next_fn
                    except StopIteration:
                        pass  # No more files; let pending drain naturally.
                    
    
        finally:
            pbar.close()
    # ── ProcessPoolExecutor.__exit__ has returned here.
    log.info("All worker processes terminated. Waiting for HDF5 files to be released…")
    _wait_for_workers_to_close_hdf5(worker_hdf5_dir, timeout=3600, poll_interval=30.0)

    # ── Upload remaining worker files (rotated shards are already on S3 ) ──
    worker_paths = sorted(
        p for p in Path(worker_hdf5_dir).glob("worker_*.h5")
        if re.match(r"worker_\d+", p.stem)   # match includes normal and recovery files
    )
    log.info(f"Uploading {len(worker_paths)} remaining worker HDF5 file(s) to S3…")

    upload_ok = True
    final_local_uploads: list[dict] = []
    for wp in worker_paths:
        try:
            try:
                segment_rows = _inspect_h5_segment_rows(str(wp))
            except OSError as open_err:
                log.error(
                    f"  Cannot inspect {wp.name} after retries: {open_err}. "
                    f"Attempting h5clear and one final read…"
                )
                _clear_h5_consistency_flags(str(wp))
                time.sleep(5)
                try:
                    segment_rows = _inspect_h5_segment_rows(str(wp), max_retries=1, retry_delay=0)
                except OSError:
                    log.error(f"  Giving up on {wp.name} — skipping (will not block manifest).")
                    continue 

            if not _is_nonempty_shard(segment_rows):
                log.info(f"  Skipping {wp.name} (no rows in any segment)")
                continue

            m = re.match(r"worker_(\d+)", wp.stem)
            worker_id = int(m.group(1))
            s3_key = f"{s3_prefix_out}__{wp.name}"
            size_mb = wp.stat().st_size / 1e6
            log.info(f"  Uploading {wp.name} ({size_mb:.0f} MB) → s3://{s3_bucket}/{s3_key}")
            _s3_client.upload_file(str(wp), s3_bucket, s3_key)
            final_local_uploads.append({
                "s3_key": s3_key,
                "worker_id": worker_id,
                "shard_idx": None,
                "segment_rows": segment_rows,
            })
        except Exception as e:
            log.error(f"  Failed to upload {wp.name}: {e}")
            upload_ok = False

    elapsed = (time.time() - t0) / 60
    _print_summary(file_counts, skipped_cycles_count, n_beats, elapsed)

    if upload_ok:
        _s3_client.put_object(
            Bucket=s3_bucket,
            Key=f"{s3_prefix_out}__file_counts.json",
            Body=json.dumps(file_counts),
        )

        log.info("Building shard manifest (rotated + final worker files)…")
        manifest = collect_shard_manifest(
            _s3_client,
            s3_bucket,
            s3_prefix_out,
            worker_hdf5_dir,
            final_local_uploads,
            target_len=TARGET_LEN,
        )
        if not manifest["shards"]:
            log.error("Shard manifest is empty — no data was uploaded to S3.")
            return file_counts

        save_shard_manifest(_s3_client, s3_bucket, s3_prefix_out, manifest)
        log.info(
            f"Manifest covers {len(manifest['shards'])} shard(s); "
            f"row totals: "
            + ", ".join(
                f"{seg}={sum(s.get('segment_rows', {}).get(seg, 0) for s in manifest['shards']):,}"
                for seg in SEGMENT_TYPES
            )
        )

        shutil.rmtree(worker_hdf5_dir, ignore_errors=True)
        log.info("Worker HDF5 scratch directory removed after upload.")
    else:
        log.warning(
            f"Some uploads failed — manifest skipped. "
            f"Worker files preserved at: {worker_hdf5_dir}"
        )

    return file_counts


def _print_summary(
    file_counts:    dict,
    skipped_cycles: int,
    n_beats: int,   
    elapsed_min:    float,
):

    total = sum(file_counts.values())
    log.info("=" * 55)
    log.info("  ECG Extraction Summary")
    log.info("=" * 55)
    log.info(f"  Elapsed                 : {elapsed_min:.1f} min")
    log.info(f"  Total files processed   : {total:,}")

    for k, v in file_counts.items():
        log.info(f"{k} : {v:,}")

    log.info(f"  Skipped beat cycles     : {skipped_cycles:,}")
    log.info(f"  #Beats  across records  : {n_beats:,}")
    log.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / SageMaker entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="ECG waveform extraction job")
    p.add_argument("--s3_bucket",       default="walkky-datasets")
    p.add_argument("--s3_prefix_in",    default="mimic-iv/files_unzip/")
    p.add_argument("--s3_prefix_out",   default="extracted-ecg-waveforms/mimic-iv/extracted_waveforms")
    p.add_argument("--local_hdf5", default="/opt/ml/processing/output/waveforms")
    p.add_argument("--delineation_lead_idx", type=int, default=LEAD_II_IDX)
    p.add_argument("--sampling_rate",   type=int, default=SAMPLING_RATE)
    p.add_argument("--min_r_peaks",     type=int, default=MIN_R_PEAKS)
    p.add_argument("--workers",         type=int, default=32)
    
    return p.parse_args()
 
 
if __name__ == "__main__":

    args = _parse_args()

    run_extraction(
        delineation_lead_idx=args.delineation_lead_idx,
        sampling_rate=args.sampling_rate,
        min_r_peaks=args.min_r_peaks,
        s3_bucket=args.s3_bucket,
        s3_prefix_in=args.s3_prefix_in,
        s3_prefix_out=args.s3_prefix_out,
        local_hdf5=args.local_hdf5,
        workers=args.workers,
    )