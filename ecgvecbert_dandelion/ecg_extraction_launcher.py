"""
SageMaker Launcher — ECG Extraction Job
========================================
Run this from Jupyter notebook or local machine to submit
ecg_extraction.py as a SageMaker Processing job.

Requirements
------------
    pip install sagemaker boto3

The processing instance downloads the script, installs dependencies,
runs ecg_extraction.py
"""

import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor

# ── configuration ─────────────────────────────────────────────────────────────
#ROLE_ARN        = "arn:aws:iam::130295451369:role/sm-processing-role"


# Instance: ml.m5.4xlarge = 16 vCPUs, 64 GB RAM 
# For 800K records - ml.m5.12xlarge (48 vCPUs).
INSTANCE_TYPE   = "ml.m5.12xlarge"
INSTANCE_COUNT  = 1                        

# Match --workers to the vCPU count of the instance
# ml.m5.4xlarge  → 16 vCPUs  → workers=14  (leave 2 for OS/SageMaker overhead)
# ml.m5.12xlarge → 48 vCPUs  → workers=44 max
WORKERS         = 32


# ── arguments  ───────────────────────────────────────────────────────────────────
S3_BUCKET =  "walkky-datasets"
S3_PREFIX_IN = "mimic-iv/files_unzip/"
S3_PREFIX_OUT = "extracted-ecg-waveforms/mimic-iv/extracted_waveforms"
LOCAL_HDF5 = "/opt/ml/processing/output/waveforms"
DELINEATION_LEAD_IDX = 1
SAMPLING_RATE = 500
MIN_R_PEAKS = 4


# ── session ───────────────────────────────────────────────────────────────────
session = sagemaker.Session()
ROLE_ARN = sagemaker.get_execution_role()
AWS_REGION = boto3.Session().region_name


# ── processor ─────────────────────────────────────────────────────────────────
processor = ScriptProcessor(
    image_uri   = f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
    command     = ["python3"],
    role        = ROLE_ARN,
    instance_type  = INSTANCE_TYPE,
    instance_count = INSTANCE_COUNT,
    base_job_name  = "ecg-extraction",
    sagemaker_session = session,
    volume_size_in_gb = 1024,
    # SageMaker Processing containers have /tmp mounted on the instance SSD.
    # 800K × 12 leads × ~10 beats, 5 waveform types × 100 float32 ≈ 200 GB uncompressed;
    # compressed HDF5 is typically 1/4th.  /tmp is 30 GB by default.
    #max_runtime_in_seconds = 86400,
    # 24 hours by default
)


# ── inputs / outputs ──────────────────────────────────────────────────────────


# ── job arguments ─────────────────────────────────────────────────────────────
job_args = [
    #"--s3_bucket", S3_BUCKET,
    #"--s3_prefix_in", S3_PREFIX_IN,
    #"--s3_prefix_out", S3_PREFIX_OUT,
    #"--local_hdf5",  LOCAL_HDF5,
    #"--delineation_lead_idx", str(DELINEATION_LEAD_IDX),
    #"--sampling_rate", str(SAMPLING_RATE),
    #"--min_r_peaks",   str(MIN_R_PEAKS),
    "--workers",   str(WORKERS),
]


# ── submit ────────────────────────────────────────────────────────────────────
processor.run(
    code        = "ecg_extraction.py",          # local path — SageMaker uploads it
    arguments   = job_args,
    wait        = False,    # set True to block until job finishes
    logs        = True,
)


print(f"Monitor at:   https://console.aws.amazon.com/sagemaker/home"
      f"?region={AWS_REGION}#/processing-jobs")
