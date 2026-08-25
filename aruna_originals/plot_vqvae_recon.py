import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import os
import boto3
import matplotlib.pyplot as plt

from vqvae_ecg_waveforms_dataset import (
    RawShardedECGData,
    FileShardedECGDataset,
    collate_batch,
    list_h5_files,
    load_patient_splits,
)

from vqvae_model import Model, ECGResidualEncoder, ECGResidualDecoder, VectorQuantizerEMA, VectorQuantizer, ResidualStack, Residual

from vqvae_ecg_waveforms_dataset import SEGMENTS, bucket_out
from vqvae_bert_sentence_dataset import SIGNAL_LENS as SIGNAL_LENS_DICT


SIGNAL_LENS = SIGNAL_LENS_DICT.values()
in_channels = 1
lead_indices = [1]

s3_client = boto3.client("s3")


def plot_vqvae_recon(segment, signal_len):
    # S3 location for model outputs
    prefix_out = f"aruna-files/vqvae/segment_{segment}_nleads_{in_channels}"
    model_output_filename = f"vqvae_model_{segment}.pkl"
    model_output_key = f"{prefix_out}/{model_output_filename}"
    # Load model from s3
    response = s3_client.get_object(
        Bucket=bucket_out,
        Key=model_output_key
    )
    # Read the bytes body and deserialize
    model = pickle.loads(response['Body'].read())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test split
    prefix_out_patients = f"aruna-files/vqvae/segment_{segment}"
    splits_key = f"{prefix_out_patients}/patient_splits_{segment}.json"
    splits = load_patient_splits(splits_key)

    # h5 files
    full_h5filelist = list_h5_files()
    h5filelist = full_h5filelist[:1]

    # Test dataset
    raw_shard = RawShardedECGData(
        segment=segment,
        signal_len=signal_len,
        h5file_list=h5filelist,
        lead_indices=lead_indices,
        rank=0,
        world_size=1,
    )
    test_ds = FileShardedECGDataset(
        raw_shard,
        patient_ids=splits["test"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context='forkserver',
    )

    # Location of output figures
    os.makedirs("figs", exist_ok=True)

    # Test reconstructions
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        test_data = batch["waveform"].to(device)
        vq_loss, data_recon, perplexity = model(test_data)
        for i in range(len(test_data)):
            plt.figure()
            plt.plot(test_data.cpu().numpy()[i].squeeze(), label="Original")
            plt.plot(data_recon.cpu().numpy()[i].squeeze(), label="Reconstruction")
            plt.legend()
            plt.savefig(f"figs/test_recon_{segment}_{i}.png")
            plt.close()


if __name__ == "__main__":
    for seg, signal_len in zip(SEGMENTS, SIGNAL_LENS):
        plot_vqvae_recon(seg, signal_len)
