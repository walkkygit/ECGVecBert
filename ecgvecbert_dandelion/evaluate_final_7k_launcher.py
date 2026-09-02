"""
SageMaker launcher for evaluate_final_7k.py (single process, one small GPU, no Ray, no DDP).

    python evaluate_final_7k_launcher.py --run_tag <tag> --seeds 42,0,1,812,995                 # dry run on internal test
    python evaluate_final_7k_launcher.py --run_tag <tag> --seeds 42,0,1,812,995 --split 7k --i_confirm_final_7k yes

Job logs go to split_2_related/job_logs/, results to split_2_related/results/{eval_internal_test|final_7k}/<run_tag>/.
"""

import argparse
import logging
import re

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BUCKET_OUT = "walkky-ml"
S3_OUTPUT_PREFIX = "ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/job_logs"
PYTORCH_VERSION = "2.2.0"
PYTHON_VERSION = "py310"
INSTANCE_TYPE = "ml.g4dn.2xlarge"   # 1x T4; inference only, ~9K-30K sentences x <= 5 seeds -> a few minutes
SOURCE_DIR = "."


def launch(run_tag: str, seeds: str, split: str, confirm: str, force: str, instance_type: str):
    boto_session = boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    role = sagemaker.get_execution_role()
    hp = {"run_tag": run_tag, "seeds": seeds, "split": split, "i_confirm_final_7k": confirm, "force": force}
    short = re.sub(r"[^a-zA-Z0-9-]", "-", f"eval-{'7k' if split == '7k' else 'int'}-{run_tag}")[:38].strip("-")
    estimator = PyTorch(
        entry_point="evaluate_final_7k.py",
        source_dir=SOURCE_DIR,
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=sm_session,
        base_job_name=short,
        output_path=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/",
        code_location=f"s3://{BUCKET_OUT}/{S3_OUTPUT_PREFIX}/code",   # source tarballs here, not at the bucket root
        hyperparameters=hp,
        volume_size=50,
        max_run=3 * 3600,
    )
    log.info(f"Launching {split} evaluation of {run_tag} (seeds {seeds}) on {instance_type}: {hp}")
    estimator.fit(wait=False)
    log.info(f"Job submitted: {estimator.latest_training_job.name}")
    return estimator


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Launch evaluate_final_7k.py on SageMaker")
    p.add_argument("--run_tag", type=str, required=True)
    p.add_argument("--seeds", type=str, default="42,0,1,812,995")
    p.add_argument("--split", type=str, default="internal_test", choices=["internal_test", "7k"])
    p.add_argument("--i_confirm_final_7k", type=str, default="no")
    p.add_argument("--force", type=str, default="no")
    p.add_argument("--instance_type", type=str, default=INSTANCE_TYPE)
    a = p.parse_args()
    if a.split == "7k" and a.i_confirm_final_7k != "yes":
        p.error("--split 7k is the one locked final evaluation; add --i_confirm_final_7k yes")
    launch(a.run_tag, a.seeds, a.split, a.i_confirm_final_7k, a.force, a.instance_type)
