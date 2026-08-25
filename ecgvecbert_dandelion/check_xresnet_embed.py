"""
Diagnostic: Check ECGSegmentEmbedder's forward pass

Run this on the SageMaker GPU instance where the real pipeline runs.
It isolates the xresnet embedder completely from the rest of the
pipeline (no S3 signal reads, no delineation, no VQ-VAE) so any
divergence found here can only come from the embedder itself.

Usage:
    python check_xresnet_embed.py \
        --model_path s3://walkky-ml/aruna-files/vqvae/models/fastai_xresnet1d101.pth \
        --in_channels 12 \
        --signal_len 176

Run it twice as SEPARATE PROCESSES (not just twice in a loop) to also
catch anything that differs across process boundaries but not within
one — e.g. CUDA context/algorithm-selection state.
"""

import argparse
import hashlib
import numpy as np
import torch

from xresnet1d101_features_ecg_waveforms import ECGSegmentEmbedder, _load_state_dict_from_path


def hash_state_dict(sd) -> str:
    """Hash the actual tensor bytes of a state_dict, not just the file."""
    h = hashlib.md5()
    for k in sorted(sd.keys()):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
        else:
            h.update(str(v).encode())
    return h.hexdigest()


def compare(a: np.ndarray, b: np.ndarray, label: str):
    equal = np.array_equal(a, b)
    close = np.allclose(a, b, atol=1e-6)
    max_abs_diff = np.max(np.abs(a - b))
    max_rel_diff = np.max(np.abs(a - b) / (np.abs(a) + 1e-8))
    print(f"[{label}] exact_equal={equal}  allclose(1e-6)={close}  "
          f"max_abs_diff={max_abs_diff:.3e}  max_rel_diff={max_rel_diff:.3e}")
    return equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--in_channels", type=int, default=12)
    parser.add_argument("--signal_len", type=int, default=176)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_deterministic", action="store_true",
                         help="Set cudnn.deterministic=True / benchmark=False before testing")
    args = parser.parse_args()

    if args.force_deterministic:
        print(">>> Forcing deterministic cuDNN settings")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        print(">>> Using DEFAULT cuDNN settings (no determinism flags set)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Checkpoint identity check (rules out "different file loaded") ---
    sd = _load_state_dict_from_path(args.model_path)
    sd_hash = hash_state_dict(sd)
    print(f"Checkpoint tensor-content MD5: {sd_hash}")

    # --- Build embedder (loads checkpoint fresh, sets eval mode) ---
    embedder = ECGSegmentEmbedder(
        signal_len=args.signal_len,
        in_channels=args.in_channels,
        model_path=args.model_path,
        device=device,
    )
    print(f"embedding_dim = {embedder.embedding_dim}")

    # --- Fixed random input, seeded so it's identical across separate process runs ---
    rng = np.random.default_rng(args.seed)
    batch_np = rng.standard_normal(
        (args.batch_size, args.in_channels, args.signal_len)
    ).astype(np.float32)
    x = torch.from_numpy(batch_np)

    print(f"\nInput batch shape: {tuple(x.shape)}, "
          f"input checksum: {hashlib.md5(batch_np.tobytes()).hexdigest()}")

    # ============================================================
    # TEST 1: same tensor, same process, called back-to-back N times
    # ============================================================
    print("\n=== TEST 1: repeated calls, same process, same input tensor ===")
    outs = []
    with torch.no_grad():
        for i in range(5):
            emb = embedder.generate_embedding_batch(x) if hasattr(embedder, "generate_embedding_batch") else None
            if emb is None:
                # embedder only exposes single-sample generate_embedding; call body directly instead
                xb = x.to(embedder.device)
                feat = embedder.body(xb)
                avg = torch.nn.functional.adaptive_avg_pool1d(feat, 1)
                mx = torch.nn.functional.adaptive_max_pool1d(feat, 1)
                emb = torch.cat([avg, mx], dim=1).squeeze(-1).cpu().numpy()
            outs.append(emb)

    all_equal_within_process = all(
        compare(outs[0], outs[i], f"run0 vs run{i}") for i in range(1, len(outs))
    )
    print(f"--> All 5 repeated in-process calls identical: {all_equal_within_process}")

    # ============================================================
    # TEST 2: save this process's output so a SEPARATE process invocation
    # can compare against it (catches cross-process/algo-selection variance)
    # ============================================================
    out_path = "/tmp/xresnet_determinism_test_output.npy"
    try:
        prev = np.load(out_path)
        print(f"\n=== TEST 2: comparing against PREVIOUS process run ({out_path}) ===")
        compare(prev, outs[0], "previous_process vs this_process")
    except FileNotFoundError:
        print(f"\n=== TEST 2: no previous run found at {out_path} — saving this run's output ===")
        print("    Run this script again (fresh `python` invocation) to compare across processes.")

    np.save(out_path, outs[0])

    print("\nDone. Interpretation:")
    print("  - If TEST 1 is NOT all-equal: nondeterminism happens even within one")
    print("    process/call sequence -> almost certainly cuDNN algorithm nondeterminism.")
    print("    Re-run with --force_deterministic and see if it becomes reproducible.")
    print("  - If TEST 1 IS all-equal but TEST 2 differs across two `python` invocations:")
    print("    nondeterminism is tied to process/CUDA-context state, not the forward pass")
    print("    logic itself -> same fix (force_deterministic) should still resolve it.")


if __name__ == "__main__":
    main()