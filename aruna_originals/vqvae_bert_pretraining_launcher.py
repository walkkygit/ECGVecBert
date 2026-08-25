"""
SageMaker launcher for VQ-VAE BERT MLM pretraining.

Uses PyTorch DDP via SageMaker's PyTorch estimator with torchrun as the distribution backend.

Usage:
    python vqvae_bert_pretraining_launcher.py

Requirements:
    pip install sagemaker boto3
"""


import logging
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# SageMaker / AWS settings
BUCKET_OUT = "walkky-ml"
S3_OUTPUT_PREFIX = "aruna-files/vqvae/bert/pretraining/sagemaker-outputs"

# Training container
PYTORCH_VERSION = "2.2.0"
PYTHON_VERSION = "py310"

# Instance settings
# ml.p3.8xlarge  = 4x V100 (16 GB each)  — good baseline
# ml.p3.16xlarge = 8x V100               — for larger batch / faster runs
# ml.p4d.24xlarge = 8x A100 (40 GB each) — for full-dataset runs
# ml.g5.12xlarge = 4 GPUs - for smaller runs
INSTANCE_TYPE = "ml.g5.48xlarge"
INSTANCE_COUNT = 2          # nodes; each node uses all GPUs on it
GPU_PER_NODE = 8            # must match the instance type above

# Job naming
JOB_PREFIX = "vqvae-bert-pretrain"

# Source code directory (uploaded to S3 by SageMaker automatically)
SOURCE_DIR = "."   # directory containing the training scripts

IN_CHANNELS = 12

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session_and_role():
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    ROLE_ARN = sagemaker.get_execution_role()
    return sm_session, ROLE_ARN


def make_estimator(sm_session, ROLE_ARN) -> PyTorch:
    """
    Build a SageMaker PyTorch estimator configured for torchrun DDP.
    """
    job_name = f"{JOB_PREFIX}-nleads-{IN_CHANNELS}"

    estimator = PyTorch(
        entry_point="vqvae_bert_pretraining.py",
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
        # Install extra dependencies not in the base container
        # If using a requirements.txt in SOURCE_DIR, SageMaker installs it automatically
        environment={
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "^lo,docker",     # use anything except loopback and docker
        },
        volume_size=100,     # GB — increase if h5 files are cached locally
        max_run=86400,       # default is 24 hours (86400), 5 days max allowed (432000), 60 hrs: 216000
    )

    return estimator


def launch():
    sm_session, ROLE_ARN = get_session_and_role()
    estimator = make_estimator(sm_session, ROLE_ARN)
    
    log.info(f"Launching VQ-VAE BERT MLM pretraining job"
             f" on {INSTANCE_COUNT}x {INSTANCE_TYPE} ({GPU_PER_NODE} GPUs/node)...")

    estimator.fit(wait=False)

    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    log.info(
        f"Console: https://console.aws.amazon.com/sagemaker/home"
        f"#/jobs/{estimator.latest_training_job.name}"
    )
    return estimator


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    launch()
