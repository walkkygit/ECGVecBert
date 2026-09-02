"""
Submit 100% sentence building as a SageMaker Training Job.
Usage: python submit_sentence_training_job.py
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
import os
from datetime import datetime

# Initialize SageMaker session
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

print(f"Region: {region}")
print(f"Role: {role}")

# Job configuration
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"dandelion-sentences-split2-{timestamp}"
instance_type = "ml.g4dn.8xlarge"
instance_count = 1

# Use the PyTorch Estimator
estimator = PyTorch(
    entry_point="vqvae_bert_finetuning_sentences_dataset.py",
    role=role,
    instance_type=instance_type,
    instance_count=instance_count,
    framework_version="2.0",
    py_version="py310",
    source_dir=".",
    output_path=f"s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/job_logs/",
    code_location="s3://walkky-ml/ecgvectbert/vqvae/bert_finetuning/dandelion/split_2_related/job_logs/code",   # source tarballs here, not at the bucket root
    hyperparameters={
        "in_channels": 12,
        "batch_size": 32,
        "use_frac": 1.0,
        "cnn_embed_type": "vqvae_encoder",
    },
    volume_size=100,  # 100 GB EBS volume
    max_run=86400,  # 24 hours max
    environment={
        "AWS_DEFAULT_REGION": region,
    },
)

print(f"\nSubmitting SageMaker Training Job: {job_name}")
estimator.fit(job_name=job_name, wait=False)  # wait=False so it runs in background

print(f"✅ Job submitted! Job name: {job_name}")
print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
