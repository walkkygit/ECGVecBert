"""
SageMaker launcher for VQ-VAE BERT Finetuning ECG beat sentence construction.

Uses PyTorch DDP via SageMaker's PyTorch estimator with torchrun as the distribution backend.

Usage:
    python vqvae_bert_finetuning_build_sentences_launcher_singlelabel.py --all
    
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
S3_OUTPUT_PREFIX = "aruna-files/vqvae_final_12lead_singlelabel/vqvae/bert_finetuning/sagemaker-outputs"

# Training container
PYTORCH_VERSION = "2.2.0"
PYTHON_VERSION = "py310"

# Instance settings
# ml.p3.8xlarge  = 4x V100 (16 GB each)  — good baseline
# ml.p3.16xlarge = 8x V100               — for larger batch / faster runs
# ml.p4d.24xlarge = 8x A100 (40 GB each) — for full-dataset runs
INSTANCE_TYPE = "ml.g5.2xlarge"
INSTANCE_COUNT = 1          # nodes; each node uses all GPUs on it
GPU_PER_NODE = 1            # must match the instance type above; limit to NUM_GENERATION_SHARDS

# Job naming
JOB_PREFIX = "vqvae-bert-finetuning-sentences"

# Source code directory (uploaded to S3 by SageMaker automatically)
SOURCE_DIR = "."   # directory containing the training scripts

IN_CHANNELS = 12
USE_FRAC = [0.01, 0.1, 1.0]

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session_and_role():
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    ROLE_ARN = sagemaker.get_execution_role()
    return sm_session, ROLE_ARN


def make_estimator(sm_session, ROLE_ARN, in_channels: int, use_frac: float) -> PyTorch:
    """
    Build a SageMaker PyTorch estimator configured for torchrun DDP.
    """
    job_name = f"{JOB_PREFIX}-{in_channels}-{use_frac:.2f}".replace('.', 'p')
  
    estimator = PyTorch(
        entry_point="vqvae_bert_finetuning_sentences_dataset_singlelabel.py",
        source_dir=SOURCE_DIR,
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        base_job_name=job_name,
        output_path=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/",
        # torchrun handles DDP process launch across GPUs on each node
        distribution={
            "torch_distributed": {
                "enabled": True,
            }
        },
        hyperparameters={
            "in_channels": in_channels,
            "batch_size": 32,
            "use_frac": use_frac,
        },
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


def launch(in_channels: int, use_frac: float):
    sm_session, ROLE_ARN = get_session_and_role()
    estimator = make_estimator(sm_session, ROLE_ARN, in_channels, use_frac)
    
    log.info(f"Launching VQ-VAE BERT finetuning sentence construction job"
             f" on {INSTANCE_COUNT}x {INSTANCE_TYPE} ({GPU_PER_NODE} GPUs/node)...")

    estimator.fit(wait=False)

    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    return estimator


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch VQ-VAE BERT finetuning sentence construction on SageMaker")

    parser.add_argument("--in_channels", type=int, default=None,
                        help="Number of input channels (leads)")
    parser.add_argument("--use_frac", type=float, default=None,
                        help="Use fraction of dataset")
    parser.add_argument("--all", action="store_true",
                        help="Launch one job per fraction in USE_FRAC")
    args = parser.parse_args()
    in_channels = args.in_channels or IN_CHANNELS
    if args.all:
        for frac in USE_FRAC:
            launch(in_channels, frac)
    else:
        use_frac = args.use_frac if args.use_frac is not None else USE_FRAC[-1]
        launch(in_channels, use_frac)
        # If no other arguments specified, all datasets at use_frac 1.0 by default