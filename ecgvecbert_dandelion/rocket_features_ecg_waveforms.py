"""
ROCKET-style fixed random convolutional feature extractor for 12-lead ECG waveforms.

Adapted from the ROCKET paper (Dempster, Petitjean & Webb, 2020) for short,
multivariate (12-lead) ECG segments (P, PQ, QRS, ST, T; length ~88-176 samples).

Design goals, matching the requirements discussed:
  - FIXED, not trained: all kernel weights/biases/dilations/paddings/channel
    masks are sampled once at construction from a seeded generator, then
    frozen (stored as buffers, requires_grad=False, permanently in eval mode).
  - REPRODUCIBLE: same seed + same (in_channels, signal_len, num_kernels)
    always produces bit-identical kernels and therefore identical features,
    across processes/ranks/runs (subject to the caveat in the module
    docstring about PyTorch RNG algorithm stability across versions).
  - UNBIASED w.r.t. domain: kernels are random, not pretrained on any
    ECG or image corpus, so no external dataset's inductive bias is
    smuggled into the feature space.
  - Efficient over large datasets (e.g. MIMIC-IV scale): kernels are
    bucketed by (kernel_length, dilation, padding) so the whole kernel
    bank is evaluated using a handful of batched conv1d calls, not a
    Python loop per kernel. Runs equally well on CPU or GPU.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROCKETFeatureExtractor(nn.Module):
    """
    Fixed random convolutional kernel feature extractor (ROCKET-style),
    adapted for multivariate 12-lead ECG waveform segments (short length,
    e.g. 88-176 samples).

    For each of `num_kernels` random kernels:
      - kernel_length: sampled uniformly from `kernel_lengths` (default
        {7, 9, 11}, as in the original ROCKET paper).
      - dilation: sampled from an exponential-in-log2 distribution, capped
        so the kernel's receptive field can never exceed signal_len.
      - channel subset: a random subset of the leads is used per kernel
        (native multivariate ROCKET behavior -- kernels mix across a
        random number of leads). Unselected-lead weights are zeroed
        rather than gathered, so the conv can still be run as a single
        batched conv1d over all input channels per (length, dilation,
        padding) group.
      - weights: iid standard normal, then mean-centered per kernel (over
        the channels/taps actually used) so each kernel responds to
        *shape*, not to a constant offset in the signal.
      - bias: iid Uniform(-1, 1), used as a floating threshold for the
        proportion-of-positive-values (PPV) feature.
      - padding: each kernel independently uses either 'same'-style zero
        padding or no padding ('valid'), each with probability 0.5 (as
        in the original ROCKET).

    Output: 2 features per kernel (PPV, max) => 2 * num_kernels by default.
    Since this is meant to sit alongside a discrete VQ-VAE token as a
    continuous feature channel, raw dimensionality is often not what you
    want to feed downstream as-is; set `proj_dim` to reduce it via a
    *fixed* random Gaussian projection (also seeded and frozen).

    Reproducibility caveat: this relies on torch.Generator's CPU RNG
    stream. Kernels are generated once and cached as buffers, so as long
    as you construct the module with the same seed/args, results are
    stable within a given PyTorch version. If you need long-term
    bit-for-bit reproducibility across PyTorch upgrades, save the
    generated buffers to disk once (state_dict) and reload them, rather
    than regenerating from the seed each time.
    """

    def __init__(
        self,
        in_channels: int,
        signal_len: int,
        num_kernels: int = 1000,
        kernel_lengths: Sequence[int] = (7, 9, 11),
        seed: int = 42,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        if signal_len <= max(kernel_lengths):
            raise ValueError(
                f"signal_len={signal_len} must be > largest kernel_length={max(kernel_lengths)}"
            )

        self.in_channels = in_channels
        self.signal_len = signal_len
        self.num_kernels = num_kernels
        self.kernel_lengths = tuple(kernel_lengths)
        self.seed = seed
        self.proj_dim = proj_dim

        g = torch.Generator().manual_seed(seed)

        # ---- sample per-kernel hyperparameters ----
        length_choices = torch.tensor(self.kernel_lengths)
        lengths = length_choices[torch.randint(len(self.kernel_lengths), (num_kernels,), generator=g)]

        dilations = torch.empty(num_kernels, dtype=torch.long)
        for i in range(num_kernels):
            L = int(lengths[i])
            max_exponent = math.log2((signal_len - 1) / (L - 1)) if L > 1 else 0.0
            max_exponent = max(max_exponent, 0.0)
            exponent = torch.rand(1, generator=g).item() * max_exponent
            dilations[i] = max(1, int(2 ** exponent))

        use_padding = torch.rand(num_kernels, generator=g) < 0.5
        # at least 1 lead, up to all in_channels leads, per kernel
        num_channels_used = 1 + torch.randint(in_channels, (num_kernels,), generator=g)

        # ---- bucket kernels by (length, dilation, padding) for batched conv1d ----
        groups: dict = {}
        for i in range(num_kernels):
            key = (int(lengths[i]), int(dilations[i]), bool(use_padding[i]))
            groups.setdefault(key, []).append(i)

        self._groups = []
        offset = 0

        for (L, dilation, pad_flag), idxs in groups.items():
            n = len(idxs)
            w = torch.randn(n, in_channels, L, generator=g)

            for local_i, global_i in enumerate(idxs):
                k = int(num_channels_used[global_i])
                perm = torch.randperm(in_channels, generator=g)[:k]
                mask = torch.zeros(in_channels)
                mask[perm] = 1.0
                w[local_i] = w[local_i] * mask.unsqueeze(-1)
                # mean-center over used channels/taps only -> ~zero DC response
                used = mask.bool()
                w[local_i, used] = w[local_i, used] - w[local_i, used].mean()

            bias = torch.rand(n, generator=g) * 2 - 1  # Uniform(-1, 1)
            padding = ((L - 1) * dilation) // 2 if pad_flag else 0

            self.register_buffer(f"weight_{offset}", w, persistent=True)
            self.register_buffer(f"bias_{offset}", bias, persistent=True)
            self._groups.append(
                dict(
                    offset=offset,
                    n=n,
                    length=L,
                    dilation=dilation,
                    padding=padding,
                    indices=torch.tensor(idxs, dtype=torch.long),
                )
            )
            offset += n

        assert offset == num_kernels
        self.register_buffer("_output_permutation", self._build_output_permutation(), persistent=True)

        # ---- optional fixed random projection down to proj_dim ----
        if proj_dim is not None:
            proj = torch.randn(2 * num_kernels, proj_dim, generator=g) / math.sqrt(2 * num_kernels)
            self.register_buffer("_projection", proj, persistent=True)

        # freeze: no learnable parameters, permanently in eval mode
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _build_output_permutation(self) -> torch.Tensor:
        # maps [grouped-kernel order] -> [original kernel order], so output
        # feature i always corresponds to the i-th sampled kernel regardless
        # of internal bucketing
        perm = torch.empty(self.num_kernels, dtype=torch.long)
        for grp in self._groups:
            perm[grp["indices"]] = grp["offset"] + torch.arange(grp["n"])
        return perm

    def train(self, mode: bool = True):
        # no learnable state ever; keep permanently in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, signal_len]
        returns: [B, 2 * num_kernels]  (or [B, proj_dim] if proj_dim was set)
        """
        assert x.dim() == 3 and x.shape[1] == self.in_channels and x.shape[2] == self.signal_len, (
            f"expected [B, {self.in_channels}, {self.signal_len}], got {tuple(x.shape)}"
        )
        x = x.to(self.weight_0.dtype)

        ppv_chunks, max_chunks = [], []
        for grp in self._groups:
            w = getattr(self, f"weight_{grp['offset']}")
            b = getattr(self, f"bias_{grp['offset']}")
            out = F.conv1d(x, w, bias=None, stride=1, padding=grp["padding"], dilation=grp["dilation"])
            out = out + b.view(1, -1, 1)
            ppv_chunks.append((out > 0).float().mean(dim=-1))   # [B, n]
            max_chunks.append(out.max(dim=-1).values)           # [B, n]

        ppv_all = torch.cat(ppv_chunks, dim=1)[:, self._output_permutation]
        max_all = torch.cat(max_chunks, dim=1)[:, self._output_permutation]

        # interleave: [ppv_0, max_0, ppv_1, max_1, ...]
        features = torch.stack([ppv_all, max_all], dim=-1).reshape(x.size(0), -1)

        if self.proj_dim is not None:
            features = features @ self._projection

        return features


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # One extractor per segment type, since each has a different signal_len
    # in your pipeline (e.g. P=88, QRS=176, T=120). Build once, reuse for
    # every batch/record of that segment across the whole MIMIC-IV pass.
    extractor = ROCKETFeatureExtractor(
        in_channels=12,
        signal_len=128,
        num_kernels=1000,   # -> 2000 gives 4000-dim raw feature; 1000 gives 2000-dim feature
        seed=42,
        proj_dim=None,      # e.g. set to 384 to match your BERT d_model
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = extractor.to(device)

    dummy_batch = torch.randn(256, 12, 128, device=device)  # [B, leads, L]
    feats = extractor(dummy_batch)
    print(feats.shape)  # torch.Size([256, 4000])

    # Reproducibility check: rebuilding with the same seed gives identical output
    extractor2 = ROCKETFeatureExtractor(in_channels=12, signal_len=128, num_kernels=1000, seed=42).to(device)
    feats2 = extractor2(dummy_batch)
    print(torch.allclose(feats, feats2))  # True