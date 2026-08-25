"""
SageMaker launcher for VQ-VAE ECG training.

Launches one job per segment (P, QRS, T) or (P, PQ, QRS, ST, T) using PyTorch DDP via
SageMaker's PyTorch estimator with torchrun as the distribution backend.

Usage:
    python vqvae_training_launcher.py --segment P    # single segment
    python vqvae_training_launcher.py --all          # all segments

Requirements:
    pip install sagemaker boto3
"""

import argparse
import logging
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# SageMaker / AWS settings
BUCKET_OUT = "walkky-ml"
S3_OUTPUT_PREFIX = "aruna-files/vqvae/sagemaker-outputs"

# Training container
PYTORCH_VERSION = "2.2.0"
PYTHON_VERSION = "py310"

# Instance settings
# ml.p3.8xlarge  = 4x V100 (16 GB each)  — good baseline
# ml.p3.16xlarge = 8x V100               — for larger batch / faster runs
# ml.p4d.24xlarge = 8x A100 (40 GB each) — for full-dataset runs
# ml.g5.12xlarge, 4 GPUs for small runs
INSTANCE_TYPE = "ml.g5.48xlarge"
INSTANCE_COUNT = 1          # nodes; each node uses all GPUs on it
GPU_PER_NODE = 8            # must match the instance type above

# Job naming
JOB_PREFIX = "vqvae-12lead"

# Source code directory (uploaded to S3 by SageMaker automatically)
SOURCE_DIR = "."   # directory containing the training scripts

SEGMENTS = ["P", "PQ", "QRS", "ST", "T"]           # ["P", "QRS", "T"]     ["P", "PQ", "QRS", "ST", "T"]   II, all
EMBEDDING_DIMS = [128, 128, 128, 128, 128]         # [64, 64, 64]          [64, 64, 64, 64, 64]        [128, 128, 128, 128, 128]    256, 64 was worse
NUM_EMBEDDINGS = [48, 48, 48, 48, 48]              # [32, 32, 32]          [32, 32, 32, 32, 32]        [48, 48, 48, 48, 48]  
SIGNAL_LENS = [88, 88, 176, 128, 120]              # [88, 176, 120]        [88, 88, 176, 128, 120]
IN_CHANNELS = 12

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session_and_role():
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    ROLE_ARN = sagemaker.get_execution_role()
    return sm_session, ROLE_ARN

sm_session, ROLE_ARN = get_session_and_role()


def make_estimator(segment: str, embedding_dim: int, num_embeddings: int, signal_len: int, in_channels: int) -> PyTorch:
    """
    Build a SageMaker PyTorch estimator configured for torchrun DDP.
    """
    job_name = f"{JOB_PREFIX}-{segment.lower()}"

    # Metric definitions — these are scraped from CloudWatch logs
    metric_definitions = [
        {"Name": "train:recon_error", "Regex": r"train recon_error\s*=\s*(\S+)"},
        {"Name": "train:loss",        "Regex": r"train loss\s*=\s*(\S+)"},
        {"Name": "train:perplexity",  "Regex": r"train perplexity\s*=\s*(\S+)"},
        {"Name": "val:recon_error",   "Regex": r"val recon_error\s*=\s*(\S+)"},
        {"Name": "val:loss",          "Regex": r"val loss\s*=\s*(\S+)"},
        {"Name": "val:perplexity",    "Regex": r"val perplexity\s*=\s*(\S+)"},
    ]
    
    estimator = PyTorch(
        entry_point="vqvae_model.py",
        source_dir=SOURCE_DIR,
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        base_job_name=job_name,
        output_path=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/{segment}/",
        # torchrun handles DDP process launch across GPUs on each node
        distribution={
            "torch_distributed": {
                "enabled": True,
            }
        },
        hyperparameters={
            "segment": segment,
            "embedding_dim": embedding_dim,
            "num_embeddings": num_embeddings,
            "signal_len": signal_len,
            "in_channels": in_channels,
        },
        metric_definitions=metric_definitions,
        # Install extra dependencies not in the base container
        # If using a requirements.txt in SOURCE_DIR, SageMaker installs it automatically
        environment={
            # Needed for nccl on p3/p4 instances
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
        },
        # Keep training logs (useful for debugging)
        #keep_alive_period_in_seconds=5,
        # Checkpoint to S3 so training can be resumed if interrupted
        #checkpoint_s3_uri=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/{segment}/checkpoints/",
        #checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=100,     # GB — increase if h5 files are cached locally
        max_run=86400,       # 24 hours max by default (86400), 5 days max allowed (432000), 60 hrs: 216000
    )

    return estimator


def launch(segment: str, embedding_dim: int, num_embeddings: int, signal_len: int, in_channels: int, wait: bool = False):
    estimator = make_estimator(segment, embedding_dim, num_embeddings, signal_len, in_channels)

    log.info(f"Launching VQ-VAE training job for segment={segment} "
             f"on {INSTANCE_COUNT}x {INSTANCE_TYPE} ({GPU_PER_NODE} GPUs/node)...")

    estimator.fit(wait=wait, logs="All")

    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    return estimator


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch VQ-VAE training on SageMaker")
    parser.add_argument("--segment", choices=SEGMENTS, default=None,
                        help="Single segment to train (P, PQ, QRS, ST, T, ...)")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dim")
    parser.add_argument("--num_embeddings", type=int, default=None,
                        help="Num embeddings")
    parser.add_argument("--signal_len", type=int, default=None,
                        help="Signal len")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Number of input channels (leads)")
    parser.add_argument("--all", action="store_true",
                        help="Launch all segments as separate jobs")
    parser.add_argument("--wait", action="store_true",
                        help="Block until each job completes (default: fire and return)")
    args = parser.parse_args()

    if args.all:
        estimators = {}
        for seg, embedding_dim, num_embeddings, signal_len in zip(SEGMENTS, EMBEDDING_DIMS, NUM_EMBEDDINGS, SIGNAL_LENS):
            est = launch(seg, embedding_dim, num_embeddings, signal_len, IN_CHANNELS, wait=False)
            estimators[seg] = est
            log.info(f"  {seg}: {est.latest_training_job.name}")

        if args.wait:
            log.info("Waiting for all jobs to complete...")
            for seg, est in estimators.items():
                est.latest_training_job.wait()
                log.info(f"  {seg} finished: {est.latest_training_job.describe()['TrainingJobStatus']}")
    elif args.segment:
        idx = SEGMENTS.index(args.segment)
        emb_dim = args.embedding_dim or EMBEDDING_DIMS[idx]
        num_emb = args.num_embeddings or NUM_EMBEDDINGS[idx]
        signal_len = args.signal_len or SIGNAL_LENS[idx]
        in_channels = args.in_channels or IN_CHANNELS
        launch(args.segment, emb_dim, num_emb, signal_len, in_channels, wait=args.wait)
    else:
        parser.print_help()