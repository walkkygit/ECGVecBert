"""
SageMaker launcher for Dandelion VQ-VAE BERT Finetuning ECG beat sentence construction.

Uses PyTorch DDP via SageMaker's PyTorch estimator with torchrun as the distribution backend.
Adapted from Aruna's launcher for Dandelion dataset.

Usage:
    python dandelion_sentences_launcher.py              # 100% dataset
    python dandelion_sentences_launcher.py --use_frac 0.1  # 10% for testing

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
S3_OUTPUT_PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/job_logs"

# Training container
PYTORCH_VERSION = "2.2.0"
PYTHON_VERSION = "py310"

# Instance settings
# ml.g5.8xlarge  = 1x A100 (40 GB)  — Aruna's proven config
INSTANCE_TYPE = "ml.g5.8xlarge"
INSTANCE_COUNT = 1          # nodes; each node uses all GPUs on it
GPU_PER_NODE = 1            # must match the instance type above; limit to NUM_GENERATION_SHARDS

# Job naming
JOB_PREFIX = "dandelion-vqvae-bert-finetuning-sentences"

# Source code directory (uploaded to S3 by SageMaker automatically)
SOURCE_DIR = "."   # directory containing the training scripts

IN_CHANNELS = 12
USE_FRAC = 1.0  # Dandelion: 100% by default
CNN_EMBED_TYPE = "vqvae_encoder"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session_and_role():
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    ROLE_ARN = sagemaker.get_execution_role()
    return sm_session, ROLE_ARN


def make_estimator(sm_session, ROLE_ARN, in_channels: int, use_frac: float, hp) -> PyTorch:
    """
    Build a SageMaker PyTorch estimator configured for torchrun DDP.
    Uses Aruna's proven DDP + NCCL configuration.
    """
    job_name = f"{JOB_PREFIX}-{in_channels}-{use_frac:.2f}".replace('.', 'p')

    estimator = PyTorch(
        entry_point="vqvae_bert_finetuning_sentences_dataset.py",
        source_dir=SOURCE_DIR,
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        base_job_name=job_name,
        output_path=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/",
        code_location=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/code",   # source tarballs here, not at the bucket root
        # torchrun handles DDP process launch across GPUs on each node
        distribution={
            "torch_distributed": {
                "enabled": True,
            }
        },
        hyperparameters=hp,
        # Install extra dependencies not in the base container
        # If using a requirements.txt in SOURCE_DIR, SageMaker installs it automatically
        environment={
            # Needed for nccl on p3/p4 instances
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
        },
        volume_size=100,     # GB — increase if h5 files are cached locally
        max_run=86400,       # 24 hours max by default (86400), 5 days max allowed (432000), 60 hrs: 216000
    )

    return estimator


def launch(dataset_name: str, in_channels: int, use_frac: float, cnn_embed_type: str):
    sm_session, ROLE_ARN = get_session_and_role()
    hp = {
        "in_channels": in_channels,
        "batch_size": 32,
        "use_frac": use_frac,
        "dataset_name": dataset_name,
    }
    if cnn_embed_type is not None:
        hp["cnn_embed_type"] = cnn_embed_type
    estimator = make_estimator(sm_session, ROLE_ARN, in_channels, use_frac, hp)

    log.info(f"Launching Dandelion VQ-VAE BERT finetuning sentence construction job"
             f" on {INSTANCE_COUNT}x {INSTANCE_TYPE} ({GPU_PER_NODE} GPUs/node)...")
    log.info(f"Dataset: {dataset_name}, use_frac={use_frac}, cnn_embed_type={cnn_embed_type}")

    estimator.fit(wait=False)

    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    return estimator


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Dandelion VQ-VAE BERT finetuning sentence construction on SageMaker")

    parser.add_argument("--dataset", type=str, default="dandelion",
                        help="Dataset name (default: dandelion)")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Number of input channels (leads)")
    parser.add_argument("--use_frac", type=float, default=None,
                        help="Use fraction of dataset (default: 1.0)")
    parser.add_argument("--cnn_embed_type", type=str, default=None,
                        choices=['vqvae_encoder', 'rocket'],
                        help="CNN embedding type (default: vqvae_encoder)")
    args = parser.parse_args()

    dataset_name = args.dataset or "dandelion"
    in_channels = args.in_channels or IN_CHANNELS
    use_frac = args.use_frac if args.use_frac is not None else USE_FRAC
    cnn_embed_type = args.cnn_embed_type or CNN_EMBED_TYPE

    launch(dataset_name, in_channels, use_frac, cnn_embed_type)
