"""
Per-lead reconstruction MSE evaluation for a trained 12-lead VQ-VAE segment model

Loads the pickled model (saved as `vqvae_model_{segment}.pkl`) from S3, rebuilds
the test split using the same patient_splits JSON used during training, runs a
single forward pass (no DDP, no gradients), and reports MSE per lead (0-11)

Usage:
    python eval_vqvae_model_test_perlead_losses.py --segment QRS --batch_size 256
"""

import argparse
import os
import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

import boto3

from vqvae_ecg_waveforms_dataset import (
    RawShardedECGData,
    FileShardedECGDataset,
    collate_batch,
    list_h5_files,
    load_patient_splits,
)
from vqvae_ecg_waveforms_dataset import NLEADS, bucket_out
from vqvae_model import Model, ECGResidualEncoder, ECGResidualDecoder, VectorQuantizerEMA, VectorQuantizer, ResidualStack, Residual
from vqvae_bert_sentence_dataset import SIGNAL_LENS

# Logging
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = f"{script_name}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_filename),   # Save logs to file
        logging.StreamHandler()           # Also print logs to console
    ]
)
log = logging.getLogger(__name__)

s3_client = boto3.client("s3")


def load_pickled_model(segment: str, device: torch.device):
    """Load the pickled full model object (already unwrapped from DDP at save time)."""
    prefix_out = f"aruna-files/vqvae/segment_{segment}_nleads_{NLEADS}"
    model_key = f"{prefix_out}/vqvae_model_{segment}.pkl"

    obj = s3_client.get_object(Bucket=bucket_out, Key=model_key)
    model_bytes = obj["Body"].read()
    model = pickle.loads(model_bytes)
    model = model.to(device)
    model.eval()
    return model


def build_test_loader(segment: str, batch_size: int):
    """Build the test split"""
    prefix_out_patients = f"aruna-files/vqvae/segment_{segment}"
    splits_key = f"{prefix_out_patients}/patient_splits_{segment}.json"
    splits = load_patient_splits(splits_key)
    if splits is None:
        raise RuntimeError(
            f"Could not load cached patient splits from {splits_key}. "
            "Training must have completed at least once for this segment."
        )

    h5filelist = list_h5_files()[:4]    # Load 4 files
    signal_len = SIGNAL_LENS[segment]

    # Single-process: rank=0, world_size=1 -> RawShardedECGData loads ALL files,
    # no partitioning, so we see every patient (test patients included).
    raw_shard = RawShardedECGData(
        segment=segment,
        signal_len=signal_len,
        h5file_list=h5filelist,
        lead_indices=list(range(NLEADS)),
        rank=0,
        world_size=1,
    )

    test_ds = FileShardedECGDataset(raw_shard, patient_ids=splits["test"])
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context="forkserver",
    )
    log.info(f"Test set: {len(test_ds)} samples, {len(test_loader)} batches")
    return test_loader


@torch.no_grad()
def evaluate_per_lead(model, test_loader, device, mask_padding: bool = False):
    """
    Returns:
        per_lead_mse: np.ndarray [NLEADS]  (mean over all valid timesteps/samples)
        overall_mse:  float
    """
    sq_err_sum = torch.zeros(NLEADS, dtype=torch.float64, device=device)
    valid_count = torch.zeros(NLEADS, dtype=torch.float64, device=device)

    for batch in test_loader:
        x = batch["waveform"].to(device, non_blocking=True)      # [B, 12, L]
        orig_lens = batch["orig_len"].to(device, non_blocking=True)

        _, x_recon, _ = model(x)                                  # [B, 12, L]

        if mask_padding:
            L = x.shape[-1]
            pos = torch.arange(L, device=device).unsqueeze(0)     # [1, L]
            valid_mask = (pos < orig_lens.unsqueeze(1)).float()   # [B, L]
            valid_mask = valid_mask.unsqueeze(1)                  # [B, 1, L], broadcasts over leads
        else:
            valid_mask = torch.ones_like(x[:, :1, :])

        sq_err = (x_recon - x) ** 2 * valid_mask                  # [B, 12, L]
        sq_err_sum += sq_err.sum(dim=(0, 2)).double()
        valid_count += valid_mask.expand(-1, NLEADS, -1).sum(dim=(0, 2)).double()

    per_lead_mse = (sq_err_sum / valid_count.clamp(min=1)).cpu().numpy()
    overall_mse = float(sq_err_sum.sum() / valid_count.sum().clamp(min=1))
    return per_lead_mse, overall_mse


@torch.no_grad()
def evaluate_codebook_usage(model, test_loader, device):
    """
    Accumulates a histogram of codebook index selections over the full test set,
    using model.encode_to_indices 
 
    Returns:
        usage_counts:   np.ndarray [num_embeddings]  (raw selection count per code)
        nonzero_count:  int   number of codes selected at least once
        perplexity_implied: float  exp(entropy) computed from the full-test-set
                                    usage distribution 
        num_embeddings: int
    """
    num_embeddings = model._vq_vae._num_embeddings
    usage_counts = torch.zeros(num_embeddings, dtype=torch.float64, device=device)
 
    for batch in test_loader:
        x = batch["waveform"].to(device, non_blocking=True)
        indices = model.encode_to_indices(x)                      # [B]
        batch_counts = torch.bincount(indices, minlength=num_embeddings).double()
        usage_counts += batch_counts
 
    total = usage_counts.sum()
    probs = usage_counts / total.clamp(min=1)
    # Perplexity over the entire test set
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity_implied = float(torch.exp(entropy))
    nonzero_count = int((usage_counts > 0).sum().item())
 
    return usage_counts.cpu().numpy(), nonzero_count, perplexity_implied, num_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segment", required=True, choices=["P", "PQ", "QRS", "ST", "T"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mask", action="store_true", default=False,
                   help="Zero-padding mask when computing MSE (matches masked training loss)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = load_pickled_model(args.segment, device)
    test_loader = build_test_loader(args.segment, args.batch_size)

    per_lead_mse, overall_mse = evaluate_per_lead(
        model, test_loader, device, mask_padding=args.mask
    )

    usage_counts, nonzero_count, perplexity_implied, num_embeddings = evaluate_codebook_usage(
        model, test_loader, device
    )

    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    log.info(f"Segment: {args.segment}")
    log.info(f"Overall MSE (all leads): {overall_mse:.6f}")
    log.info("Per-lead MSE:")
    for name, mse in zip(lead_names, per_lead_mse):
        bar = "#" * int(mse / max(per_lead_mse) * 40)
        log.info(f"  {name:>4}: {mse:.6f}  {bar}")

    worst_idx = int(np.argmax(per_lead_mse))
    best_idx = int(np.argmin(per_lead_mse))
    log.info(
        f"Highest error lead: {lead_names[worst_idx]} ({per_lead_mse[worst_idx]:.6f}), "
        f"lowest error lead: {lead_names[best_idx]} ({per_lead_mse[best_idx]:.6f}), "
        f"ratio={per_lead_mse[worst_idx]/max(per_lead_mse[best_idx], 1e-12):.2f}x"
    )

    log.info("")
    log.info("Codebook usage over full test set:")
    log.info(f"  num_embedding:        {num_embeddings}")
    log.info(f"  nonzero-usage codes (>=1 selection): {nonzero_count}  "
              f"({100*nonzero_count/num_embeddings:.1f}% of table)")
    log.info(f"  perplexity-implied effective codes:  {perplexity_implied:.2f}  "
              f"({100*perplexity_implied/num_embeddings:.1f}% of table)")
    if nonzero_count > 0:
        gap_ratio = nonzero_count / perplexity_implied
        log.info(f"  nonzero / perplexity ratio: {gap_ratio:.2f}x")
        if gap_ratio > 1.5:
            log.info(
                "  -> Usage is skewed: many codes get occasional use but a few "
                "dominate most samples. Larger codebook IS being explored, "
                "just not evenly -- worth revisiting commitment_cost/decay "
                "rather than assuming the codebook itself is oversized."
            )
        else:
            log.info(
                "  -> Nonzero-usage count tracks perplexity closely: usage really "
                "is concentrated on ~perplexity-many codes, with little long-tail "
                "activity beyond that."
            )
    top_k = min(10, num_embeddings)
    top_idx = np.argsort(usage_counts)[::-1][:top_k]
    log.info(f"  Top {top_k} most-used codes (index: count):")
    for idx in top_idx:
        log.info(f"    code {int(idx):3d}: {int(usage_counts[idx])}")


if __name__ == "__main__":
    main()