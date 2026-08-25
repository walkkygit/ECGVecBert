import torch
import torch.nn.functional as F
import numpy as np
import boto3
import pickle
from vqvae_model import Model, ECGResidualEncoder, ECGResidualDecoder, VectorQuantizerEMA, VectorQuantizer, ResidualStack, Residual


@torch.no_grad()
def vqvae_encode_conv_features(encoder, x):
    """
    Runs ECGResidualEncoder up through the residual stack only —
    skips self.linear and the final L2-normalize/scale step.

    Args:
        encoder: an ECGResidualEncoder instance (e.g. model._encoder,
                 or model.module._encoder if still DDP-wrapped)
        x: input waveform tensor, [B, in_channels, signal_len]

    Returns:
        features: [B, 2*num_hiddens * canonical_len]  (flattened conv/residual output)
    
    Usage:
        features = vqvae_encode_conv_features(encoder, batch["waveform"].to(device))
    """

    # Choose a canonical length; output embedding will be canonical length*hidden_dim
    canonical_len = 10   # output = 20*128
    # Encoder
    encoder.eval()
    z = encoder._conv_1(x)
    z = F.gelu(z)
    z = encoder._conv_2(z)
    z = F.gelu(z)
    z = encoder._conv_3(z)
    z = encoder._residual_stack(z)     # [B, num_hiddens, out_len], out_len = signal_len//4
    # Adaptive pooling to canonical_len to ensure all waveform types give equal output dimensions
    avg_pool = F.adaptive_avg_pool1d(z, canonical_len)    # [B, num_hiddens, canonical_len]
    max_pool = F.adaptive_max_pool1d(z, canonical_len)    # [B, num_hiddens, canonical_len]
    z = torch.cat([avg_pool, max_pool], dim=1)            # [B, 2*num_hiddens, canonical_len]
    features = torch.flatten(z, 1)     # [B, 2*num_hiddens*canonical_len]
    return features


def vqvae_encoder_embedding(model, device, x):
    model = model.to(device)
    model.eval()
    
    encoder = model._encoder  # unwrap .module first if this is still a DDP model
    with torch.no_grad():
        features = vqvae_encode_conv_features(encoder, x.to(device))

    return features


if __name__ == "__main__":

    #------------Example usage---------------
    seg = "QRS"
    signal_len = 176
    in_channels = 12
    bucket_name = "walkky-ml"
    base_prefix = "aruna-files/vqvae"
    model_key = f"{base_prefix}/segment_{seg}_nleads_{in_channels}/vqvae_model_{seg}.pkl"

    segment = np.random.randn(in_channels, signal_len).astype(np.float32)      # [in_channels,signal_len]
    x = torch.from_numpy(segment).unsqueeze(0) # [1,in_channels,signal_len]

    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model = pickle.loads(response["Body"].read())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding = vqvae_encoder_embedding(model, device, x)

    print(embedding, embedding.shape)