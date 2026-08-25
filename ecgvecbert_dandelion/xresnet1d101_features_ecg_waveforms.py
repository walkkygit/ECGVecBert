import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from urllib.parse import urlparse
import boto3
import io
from models.xresnet1d import xresnet1d101
from collections.abc import Mapping
from scipy.signal import resample_poly


# Source:
# https://github.com/helme/ecg_ptbxl_benchmarking/tree/master/output/exp0/models/fastai_xresnet1d101/models
# https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/models/xresnet1d.py


def _is_state_dict(obj) -> bool:
    return (
        isinstance(obj, Mapping) and obj and
        all(isinstance(k, str) for k in obj.keys()) and
        all(torch.is_tensor(v) or isinstance(v, torch.nn.Parameter) for v in obj.values())
    )


def _load_state_dict_from_path(model_path: str):
    if model_path.startswith("s3://"):
        parsed = urlparse(model_path)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        buffer = io.BytesIO(obj["Body"].read())
        return torch.load(buffer, map_location="cpu", weights_only=False)
    else:
        return torch.load(model_path, map_location="cpu", weights_only=False)


def _extract_state_dict(obj, _depth=0, _max_depth=5):
    """Find a state_dict in dict/module/list/tuple checkpoint objects."""
    if _is_state_dict(obj):
        return dict(obj)
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            sd = obj.state_dict()
            if _is_state_dict(sd): return dict(sd)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        for key in ("state_dict","model_state","weights","params","model","net","module","ema","model_ema","teacher","student"):
            if key in obj:
                try:
                    return _extract_state_dict(obj[key], _depth+1, _max_depth)
                except Exception:
                    pass
        if _depth < _max_depth:
            for v in obj.values():
                try:
                    return _extract_state_dict(v, _depth+1, _max_depth)
                except Exception:
                    continue
    if isinstance(obj, (list, tuple)):
        for it in obj:
            try:
                return _extract_state_dict(it, _depth+1, _max_depth)
            except Exception:
                continue
    raise ValueError("Could not find a state_dict in the provided checkpoint.")


def _clean_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        while True:
            if   k.startswith("module."): k = k[7:]
            elif k.startswith("model.") : k = k[6:]
            elif k.startswith("net.")   : k = k[4:]
            else: break
        out[k] = v
    return out


def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # x: [C, L]
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where((sd < eps) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(~np.isfinite(mu), 0.0, mu)
    return (x - mu) / (sd + eps)


def downsample_500_to_100(x: np.ndarray) -> np.ndarray:
    # x: [C, L] or [L]
    return resample_poly(x.astype(np.float32), up=1, down=5, axis=-1)


def pad_or_crop_center(x: np.ndarray, target_len: int) -> np.ndarray:
    # x: [C, L]
    L = x.shape[-1]
    if L == target_len:
        return x
    if L < target_len:
        pad_total = target_len - L
        pl, pr = pad_total // 2, pad_total - pad_total // 2
        return np.pad(x, ((0, 0), (pl, pr)), mode="constant")
    start = (L - target_len) // 2
    return x[:, start:start+target_len]


class ECGSegmentEmbedder:
    def __init__(self, signal_len: int, in_channels: int, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if (device == "cpu") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.signal_length = signal_len
        
        # Build xresnet backbone
        model = xresnet1d101(input_channels=in_channels, num_classes=1)
        
        # Load weights (non‑strict to ignore classifier head mismatch)
        sd = _load_state_dict_from_path(model_path)
        sd = _extract_state_dict(sd)                   # unwraps "model"/"opt" etc.
        sd = _clean_state_dict_keys(sd)                # strips module./model./net. prefixes
        ret = model.load_state_dict(sd, strict=False)

        # The final top-level child (classifier head) is dropped in self.body via
        # nn.Sequential(*list(model.children())[:-1]) — mismatches confined to that
        # index are harmless and expected (checkpoint's head architecture differs
        # from this xresnet1d101's default head, but the head is never used).
        num_children = len(list(model.children()))
        head_prefix = f"{num_children - 1}."
        
        backbone_missing    = [k for k in ret.missing_keys    if not k.startswith(head_prefix)]
        backbone_unexpected = [k for k in ret.unexpected_keys  if not k.startswith(head_prefix)]
        
        assert not backbone_missing and not backbone_unexpected, (
            f"xresnet BACKBONE checkpoint mismatch (not just head) — "
            f"missing={backbone_missing}, unexpected={backbone_unexpected}"
        )
        if ret.missing_keys or ret.unexpected_keys:
            print(
                f"[load_state_dict] head-only mismatch ignored (head is dropped in self.body): "
                f"missing={ret.missing_keys}, unexpected={ret.unexpected_keys}"
            )
        
        # Drop classifier head; keep body for features
        self.body = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.body.eval()
        
        # Probe embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_len, device=self.device)
            feat = self.body(dummy)  # [1,C,L']
            self.out_channels = int(feat.shape[1])
        self.embedding_dim = self.out_channels*2   # avg and max

    def generate_embedding(self, signal: np.ndarray) -> np.ndarray:
        """Generate signal embedding """
        x = downsample_500_to_100(signal)
        x = zscore_per_channel(x)
        x = pad_or_crop_center(x, self.signal_length)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)  # [1,in_channels,signal_len]
        
        with torch.no_grad():
            feat = self.body(x)                    # [B=1,C,L']
            avg = F.adaptive_avg_pool1d(feat, 1)                  # [1,C,1]
            mx  = F.adaptive_max_pool1d(feat, 1)                  # [1,C,1]
            emb = torch.cat([avg, mx], dim=1).squeeze(0).squeeze(-1)  # [2C]
        return emb.cpu().numpy()


if __name__ == "__main__":
    # ---------------- Example usage ----------------
    #--------xresnet101_1d extractor-----------------
    # One ECG segment with 12 leads, length 176
    signal_len = 176
    in_channels = 12
    segment = np.random.randn(in_channels, signal_len).astype(np.float32)
    extractor = ECGSegmentEmbedder(signal_len=signal_len,
                                   in_channels=in_channels,
                                   model_path="./models/fastai_xresnet1d101.pth",
                                   device="cuda")
    embedding = extractor.generate_embedding(segment)

    print("Embedding shape:", embedding.shape)  # (embedding_dim,)
    print("Embedding dim:", extractor.embedding_dim)  # (embedding_dim)
