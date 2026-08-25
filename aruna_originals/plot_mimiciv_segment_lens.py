"""
Collect pre-pad segment lengths (original_len), 
and plot a histogram per segment type (P, PQ, QRS, ST, T, TP).

Requirements: matplotlib, h5py, hdf5plugin, boto3, numpy, tqdm

Example 
    python plot_segment_lengths.py 
"""


import boto3, s3fs
import h5py
import hdf5plugin  
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

# Segments to plot (order for subplot layout)
SEGMENTS = ("P", "PQ", "QRS", "ST", "T", "TP")
TARGET_LEN = 800
s3 = boto3.client("s3")
bucket = "walkky-datasets"
prefix = "extracted-ecg-waveforms/mimic-iv/"


def list_h5_files(bucket: str, prefix: str) -> list[str]:    
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5"):
                files.append(key)
    
    return files


def collect_segment_lengths(chunk_size: int = 10_000) -> dict[str, np.ndarray]:
    """
    Read ``original_len`` for each segment type.
    Returns
    -------
    dict
        Keys are segment names; values are 1-D int32 arrays of all lengths.
    """

    chunks: dict[str, list[np.ndarray]] = {seg: [] for seg in SEGMENTS}
    h5files = list_h5_files(bucket, prefix)

    fs = s3fs.S3FileSystem()
    for filename in tqdm(h5files, desc="Reading original_len from files"):
        with fs.open(f"s3://{bucket}/{filename}", 'rb') as f:
            with h5py.File(f, 'r') as h5file:
                for seg in SEGMENTS:
                    if seg not in h5file:
                        continue
                    n = int(h5file[seg]["original_len"].shape[0])
                    if n == 0:
                        continue
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        chunks[seg].append(np.array(h5file[seg]["original_len"][start:end], dtype=np.int32))

    out: dict[str, np.ndarray] = {}
    for seg in SEGMENTS:
        if chunks[seg]:
            out[seg] = np.concatenate(chunks[seg])
    return out


def plot_segment_length_histograms(lengths_by_segment: dict[str, np.ndarray]) -> None:
    """One histogram per segment type in a single figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, seg in enumerate(SEGMENTS):
        ax = axes_flat[i]
        data = lengths_by_segment.get(seg, np.empty(0, dtype=np.int32))
        if data.size == 0:
            ax.set_title(f"{seg} (no data)")
            ax.text(0.5, 0.5, "No rows", ha="center", va="center", transform=ax.transAxes)
            continue
        data = data[data > 0] # Drop 0 values
        ax.hist(data, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(
            f"{seg}  (n={data.size:,} p99= {np.percentile(data, 99):.1f})"
        )
        ax.set_xlabel("original length")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig("figs/segment_original_len_histograms.png")


def print_summary(lengths_by_segment: dict[str, np.ndarray]) -> None:
    print("\nSegment  original length summary")
    print("-" * 90)
    print(f"{'segment':<8} {'zeros':>8} {'nonzero count':>8} {'mean':>8} {'std':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'Target length pctl':>8}")
    print("-" * 90)
    for seg in SEGMENTS:
        d = lengths_by_segment.get(seg, np.empty(0))
        if d.size == 0:
            print(f"{seg:<8} {'0':>12}")
            continue
        numzero_len = np.count_nonzero(d==0)
        d = d[d > 0] # Drop zeros
        print(
            f"{seg:<8} {numzero_len:>8} {d.size:8,} {d.mean():8.1f} {d.std():8.1f} "
            f"{np.percentile(d, 50):8.0f} {np.percentile(d, 95):8.0f} {np.percentile(d, 99):8.0f} "
            f"{d.max():8.0f} {stats.percentileofscore(d, TARGET_LEN):8.0f} "
        )
    print("-" * 90)


if __name__ == "__main__":
    lengths = collect_segment_lengths()
    print_summary(lengths)
    plot_segment_length_histograms(lengths)