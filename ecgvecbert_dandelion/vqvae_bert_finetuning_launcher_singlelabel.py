"""
SageMaker launcher for VQ-VAE BERT Finetuning.

IMPORTANT: The training entry point (vqvae_bert_finetuning.py) uses Ray Train's
TorchTrainer with ScalingConfig(num_workers=..., use_gpu=True). Ray owns the
DDP process launching itself (it spawns one Ray actor per GPU inside the
single process that calls trainer.fit()). Do NOT also enable SageMaker's
torch_distributed (torchrun) launcher here — that would spawn N processes per
node via torchrun, each of which would independently call ray.init() and spin
up its own set of Ray workers, oversubscribing the GPUs (e.g. 4 torchrun
procs x 4 Ray workers = 16 workers fighting over 4 GPUs). SageMaker must launch
a single plain process per node; Ray handles the rest internally.

Usage:
    python vqvae_bert_finetuning_launcher_singlelabel.py --all

Requirements:
    pip install sagemaker boto3
"""

import itertools
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
INSTANCE_COUNT = 1          # nodes; Ray Train manages GPUs on this single node.
                            # NOTE: if you ever raise this above 1, ray.init()
                            # in the training script needs to join a real
                            # multi-node Ray cluster (head + worker nodes) —
                            # each node independently calling ray.init() would
                            # just create separate, disconnected 1-node clusters.
GPU_PER_NODE = 1            # must match the instance type above

# Job naming
JOB_PREFIX = "vqvae-bert-finetune"

# Source code directory (uploaded to S3 by SageMaker automatically)
SOURCE_DIR = "."   # directory containing the training scripts

IN_CHANNELS = 12
USE_FRAC = [0.01, 0.1, 1.0]
DATASETS = ["ptbxl", "cpsc2018", "cs"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_session_and_role():
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    ROLE_ARN = sagemaker.get_execution_role()
    return sm_session, ROLE_ARN


def make_estimator(sm_session, ROLE_ARN, in_channels: int, use_frac: float, dataset: str, num_ray_workers: int) -> PyTorch:
    """
    Build a SageMaker PyTorch estimator that runs a single process per node.
    Ray Train (inside vqvae_bert_finetuning.py) is responsible for spawning
    the per-GPU workers via ScalingConfig — no torchrun/torch_distributed here.
    """
    job_name = f"{JOB_PREFIX}-{in_channels}-{dataset}-{use_frac:.2f}".replace('.', 'p')

    estimator = PyTorch(
        entry_point="vqvae_bert_finetuning_singlelabel.py",
        source_dir=SOURCE_DIR,
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        base_job_name=job_name,
        output_path=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/",
        # No `distribution=` here on purpose: Ray Train's TorchTrainer already
        # launches one worker process per GPU internally. Adding SageMaker's
        # torch_distributed launcher on top would double-spawn workers.
        hyperparameters={
            "in_channels": in_channels,
            "batch_size": 32,
            "use_frac": use_frac,
            "dataset": dataset,
            "num_ray_workers": num_ray_workers,
        },
        # Install extra dependencies not in the base container
        # If using a requirements.txt in SOURCE_DIR, SageMaker installs it automatically
        environment={
            # Needed for nccl on p3/p4 instances (Ray Train's DDP backend still uses NCCL)
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
        },
        volume_size=100,     # GB — increase if h5 files are cached locally
        max_run=86400,       # 24 hours max by default (86400), 5 days max allowed (432000), 60 hrs: 216000
    )

    return estimator


def launch(in_channels: int, use_frac: float, dataset: str, num_ray_workers: int):
    sm_session, ROLE_ARN = get_session_and_role()
    estimator = make_estimator(sm_session, ROLE_ARN, in_channels, use_frac, dataset, num_ray_workers)

    log.info(f"Launching VQ-VAE BERT finetuning job (Ray-managed, single process/node)"
             f" on {INSTANCE_COUNT}x {INSTANCE_TYPE} ({GPU_PER_NODE} GPUs/node, "
             f"num_ray_workers={num_ray_workers})...")

    estimator.fit(wait=False)

    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    return estimator


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch VQ-VAE BERT finetuning on SageMaker")

    parser.add_argument("--in_channels", type=int, default=None, choices=[1, 12],
                        help="Number of input channels (leads)")
    parser.add_argument("--use_frac", type=float, default=1.0, choices=USE_FRAC,
                        help="Use fraction of dataset")
    parser.add_argument("--dataset", type=str, default="ptbxl", choices=DATASETS,
                        help="Finetuning dataset")
    parser.add_argument("--num_ray_workers", type=int, default=GPU_PER_NODE,
                        help="num ray workers (1 per GPU)")

    parser.add_argument("--all", action="store_true",
                        help="Launch one job per fraction in USE_FRAC per dataset in DATASETS")
    args = parser.parse_args()

    in_channels = args.in_channels or IN_CHANNELS
    num_ray_workers = args.num_ray_workers or GPU_PER_NODE

    if args.all:
        for usefrac, dataset in itertools.product(USE_FRAC, DATASETS):
            launch(in_channels, usefrac, dataset, num_ray_workers)
    else:
        usefrac = args.use_frac if args.use_frac is not None else USE_FRAC[-1]
        dataset = args.dataset if args.dataset is not None else DATASETS[0]
        launch(in_channels, usefrac, dataset, num_ray_workers)
        # ptbxl with usefrac 1.0 if no arguments specified