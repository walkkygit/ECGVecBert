"""
VQ-VAE Model
Uses convolutional encoder-decoder with residual architecture
A simple convolutional encoder-decoder option is also provided
VQ-VAE model with EMA is used, and VQ-VAE with gradient updates is also provided
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import datetime
import math

import pickle
import json
import io
import boto3
import s3fs

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from vqvae_ecg_waveforms_dataset import (
    RawShardedECGData,
    FileShardedECGDataset,
    InMemoryECGDataset,
    collate_batch,
    list_h5_files,
    collect_unique_patient_ids,
    split_patients,
    save_patient_splits,
    load_patient_splits,
    SplitName,
)
from vqvae_ecg_waveforms_dataset import SEGMENTS, NLEADS, bucket_out

log = logging.getLogger(__name__)

# Set parameters
NUM_h5_FILES = None    # Full dataset

s3_client = boto3.client("s3")
s3_fs = s3fs.S3FileSystem()

# Input dimensions: Batch size B, number of channels C (1 or 12), input signal length L (176)
# Input signal length as 99th percentile signal length is 176 for QRS , 88 for P and 120 for T 

#---------------CONSTANTS------------------------------------------------------

config = {
    "use_batchnorm": True,
    "use_dropout": False,
    "weight_decay": 1e-4,
    "lr": 5e-4,  # Tried 3e-3, 1e-3, 5e-4, 1e-4
    "warmup_epochs": 10,
    "min_lr_factor": 0.1,    # increased from 0.01
    "grad_clip_max_norm": 1.0,
    "batch_size": 256,   # per GPU, increased from 128
    "pooled_dim": 16,   # adaptive pooling output length, tried 8 and 16
    "epochs": 400,
    "dropout": 0.0,
    "patience": 30,
    "min_delta": 1e-6,
}
log_step = 1
MASK = False      #True        # whether or not to mask zero-padding in recon loss calc

#SIGNAL_LEN = 176
#LEAD_INDICES = [1]
#IN_CHANNELS = 1
#EMBEDDING_DIM = 32
#NUM_EMBEDDINGS = 32      # Tried 16, 32

COMMITMENT_COST = 0.25    # Tried 0.25, 0.1, 0.05 (too low)
DECAY = 0.99    # dropped from 0.99 to 0.95 or 0.9, also tried 0.8

# Residual architecture (reduce signal length by 1/4th from 176 to 44 or 88 to 22 or 120 to 30)
res_config = {
    "num_residual_layers": 4,
    "num_hiddens": 128,
    "num_residual_hiddens": 32,
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
}

# Convolutional Architecture (reduce signal length by 1/8th from 176 to 22 or 88 to 11 or 120 to 15)
LAYERS   = 3
KERNELS  = [3, 3, 3]
CHANNELS = [32, 64, 128]
STRIDES  = [2, 2, 2]
PADDING = 1


# ---------------Convolutional Autoencoder --------------------------------------------------

# Encoder
class ECGConvEncoder(nn.Module):
    def __init__(self, embedding_dim, signal_len, in_channels, num_layers=LAYERS,
                 kernel_list=KERNELS, channel_list=CHANNELS, stride_list=STRIDES, padding=PADDING,
                 use_batchnorm=config["use_batchnorm"], use_dropout=config["use_dropout"], dropout=config["dropout"], 
                 pooled_dim=config["pooled_dim"]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.signal_len = signal_len
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.kernel_list = kernel_list
        self.channel_list = channel_list
        self.stride_list = stride_list
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = dropout
        self.conv_layers = self._get_conv_layers()
        self.pooled_dim = pooled_dim

        # Skipping pooling
        #linear_in = 2*self.channel_list[-1]*self.pooled_dim   # 2 due to concatenation of adaptive avg and max
        #self.linear = nn.Linear(linear_in, self.embedding_dim)

        # Save output length, Output length from encoder's 3 conv layers=signal_len//8
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_len)
            dummy = self.conv_layers(dummy)
            self.out_len = dummy.shape[-1]

        # Dropping pooling and using linear layer instead:
        linear_in = self.channel_list[-1] * self.out_len
        self.linear = nn.Linear(linear_in, self.embedding_dim)
        self.pre_vq_norm = nn.LayerNorm(self.embedding_dim)

    def _get_conv_layers(self):
        conv_layers = nn.Sequential()
        for i in range(self.num_layers):
            cur_in_channels = self.in_channels if i==0 else self.channel_list[i-1]
            cur_conv_layer = nn.Conv1d(in_channels=cur_in_channels, 
                                       out_channels=self.channel_list[i],
                                       kernel_size=self.kernel_list[i],
                                       stride=self.stride_list[i],
                                       padding=self.padding)
            conv_layers.append(cur_conv_layer)
            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm1d(self.channel_list[i]))
            conv_layers.append(nn.GELU())
            if self.use_dropout:
                conv_layers.append(nn.Dropout1d(self.dropout))
        return conv_layers

    def forward(self, x):
        x = self.conv_layers(x)

        # Skipping pooling:
        # Global average pooling, Concatenate adaptive average and max pooling
        #a = F.adaptive_avg_pool1d(x, self.pooled_dim)  # [B, C, pooled_dim]
        #m = F.adaptive_max_pool1d(x, self.pooled_dim)
        #pooled = torch.cat([a, m], dim=1)               # [B, 2C, pooled_dim]
        #z = self.linear(torch.flatten(pooled, 1))           # [B, embedding_dim]

        z = self.linear(x.reshape(x.shape[0], -1))
        z = self.pre_vq_norm(z)                  # stabilizes VQ-VAE training
        return z


# Decoder
class ECGConvDecoder(nn.Module):
    def __init__(self, embedding_dim, out_len,  signal_len, out_channels,
                 num_layers=LAYERS, kernel_list=KERNELS, channel_list=CHANNELS, stride_list=STRIDES,
                 padding=PADDING, use_batchnorm=config["use_batchnorm"],
                 use_dropout=config["use_dropout"], dropout=config["dropout"],
                 pooled_dim=config["pooled_dim"]):
        super().__init__()
        self.pooled_dim = pooled_dim
        self.out_len = out_len
        self.signal_len = signal_len
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.kernel_list = kernel_list
        self.channel_list = list(channel_list)
        self.stride_list = stride_list
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = dropout
        
        # If pooling: channel_list[-1] mirrors encoder; *2 because encoder concatenated avg+max
        # the decoder works on the VQ embedding
        #self.linear = nn.Linear(embedding_dim, 2*channel_list[-1] * pooled_dim)
        #self.channel_proj = nn.Conv1d(2*channel_list[-1], channel_list[-1], kernel_size=1)
        # Going directly to channel_list[-1]:
        #self.linear = nn.Linear(embedding_dim, channel_list[-1] * pooled_dim)
        
        # Skipping pooling:
        self.linear = nn.Linear(embedding_dim, channel_list[-1] * self.out_len)
        self.deconv_layers = self._get_deconv_layers()

    def _get_deconv_layers(self):
        layers = nn.Sequential()
        dec_channel_list = self.channel_list[::-1] + [self.out_channels]
        # Reverse the encoder layer order
        rev_kernels = list(reversed(self.kernel_list))
        rev_strides = list(reversed(self.stride_list))
        for i in range(self.num_layers):
            cur_in = dec_channel_list[i]
            cur_out = dec_channel_list[i+1] 
            kernel_size = rev_kernels[i]
            stride = rev_strides[i]
            layers.append(nn.ConvTranspose1d(
                in_channels=cur_in,
                out_channels=cur_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                output_padding=stride-1   # keeps length symmetric
            ))
            # Do not apply batch norm, GELU and dropout after the final output layer
            if i < self.num_layers-1: 
                if self.use_batchnorm:
                    layers.append(nn.BatchNorm1d(cur_out))
                layers.append(nn.GELU())
                if self.use_dropout:
                    layers.append(nn.Dropout1d(self.dropout))
        return layers

    def forward(self, z):
        # z: [B, embedding_dim]
        x = self.linear(z)       # [B, C[-1]*out_len] if not pooling. If pooling: [B, C[-1] * pooled_dim],  not 2*C[-1] here

        # Skipped pooling:
        #x = x.view(x.size(0), self.channel_list[-1], self.pooled_dim)   # [B, C[-1], pooled_dim],  not 2*C[-1] here
        #x = self.channel_proj(x)      # [B, C[-1], pooled_dim]
        #x = F.interpolate(x, size=self.out_len, mode='linear', align_corners=False) # [B, C[-1], out_len)

        # Linear without pooling
        x = x.view(x.size(0), self.channel_list[-1], self.out_len)   # [B, C[-1], out_len]
        x = self.deconv_layers(x)      # [B, out_channels, signal_len] 
        # Trim or pad to exactly signal_len (transposed convs can be off by 1)
        x = x[..., :self.signal_len]
        x = F.pad(x, (0, max(0, self.signal_len - x.size(-1))))
        return x     # [B, out_channels, signal_len]


# --------------------Convolutional Autoencoder with Residual Stack------------------

# Residual Stack
class Residual(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels=num_hiddens,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, num_residual_layers, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.gelu(x)


# Convolutional Encoder with Residual Stack
class ECGResidualEncoder(nn.Module):
    def __init__(self, signal_len, embedding_dim, num_residual_layers, num_hiddens, num_residual_hiddens,
                 in_channels, kernel_size, stride, padding, pooled_dim=config["pooled_dim"]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_residual_layers = num_residual_layers
        self.num_hiddens = num_hiddens
        self.num_residual_hiddens = num_residual_hiddens 
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._conv_1 = nn.Conv1d(in_channels=self.in_channels,
                                 out_channels=self.num_hiddens//2,
                                 kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding)
        
        self._conv_2 = nn.Conv1d(in_channels=self.num_hiddens//2,
                                 out_channels=self.num_hiddens,
                                 kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding)

        self._conv_3 = nn.Conv1d(in_channels=self.num_hiddens,
                                 out_channels=self.num_hiddens, 
                                 kernel_size=3, 
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(num_residual_layers=self.num_residual_layers,
                                             num_hiddens=self.num_hiddens,
                                             num_residual_hiddens=self.num_residual_hiddens)

        #self.pooled_dim = pooled_dim
        #linear_in = 2*self.num_hiddens*self.pooled_dim   # 2 due to concatenation of adaptive avg and max
        #self.linear = nn.Linear(linear_in, self.embedding_dim)

        # Get output shape: input signal len/4 after convs using res_config params
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_len)
            x = F.gelu(self._conv_1(dummy))
            x = F.gelu(self._conv_2(x))
            x = self._conv_3(x)
            self.out_len = x.shape[-1]   # ground truth, not assumed

        # Note: pooling loses temporal structure and did not perform well. Replace with linear layer:
        # to go from num_hiddens*out_len (which is input signal size/4 after convs) to embedding_dim
        # with layer normalization

        self.linear = nn.Linear(num_hiddens * self.out_len, embedding_dim)
        # Removed LayerNorm as training was stalling early; replaced with individual L2 norm
        #self.pre_vq_norm = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.gelu(x)

        x = self._conv_2(x)
        x = F.gelu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)   # output: B, num_hidden, L/4

        # If not pooling, use linear layer with layer norm:
        z = self.linear(torch.flatten(x, 1))     # [B, embedding_dim]
        # Not using LayerNorm, using individual L2 norm
        #z = self.pre_vq_norm(z)                  # stabilizes VQ-VAE training
        z = F.normalize(z, dim=-1) * math.sqrt(self.embedding_dim)

        # Pooling did not perform well; skipping this
        # Global average pooling, Concatenate adaptive average and max pooling
        #a = F.adaptive_avg_pool1d(x, self.pooled_dim)  # [B, num_hidden, pooled_dim]
        #m = F.adaptive_max_pool1d(x, self.pooled_dim)
        #pooled = torch.cat([a, m], dim=1)               # [B, 2*num_hidden, pooled_dim]
        #z = self.linear(torch.flatten(pooled, 1))           # [B, embedding_dim]

        return z


# Convolutional Decoder with Residual Stack
class ECGResidualDecoder(nn.Module):
    def __init__(self, signal_len, out_len, embedding_dim, num_residual_layers, num_hiddens,
                 num_residual_hiddens, out_channels, kernel_size,
                 stride, padding, pooled_dim=config["pooled_dim"]):
        super().__init__()
        #self.pooled_dim = pooled_dim
        self.num_hiddens = num_hiddens
        self.signal_len = signal_len
        self.out_len = out_len   # Output length from encoder conv layers = signal len/4 with res_config params.

        # Skipped pooling
        # Invert the linear projection: embedding -> encoder pooling output
        # encoder pooling output -> feature map from encoder conv layers
        #self.linear = nn.Linear(embedding_dim, 2*num_hiddens * pooled_dim)
        # Project channels: 2*num_hiddens -> num_hiddens (no spatial change)
        #self.channel_proj = nn.Conv1d(2*num_hiddens, num_hiddens, kernel_size=1)

        # Embedding -> encoder conv output
        self.linear = nn.Linear(embedding_dim, num_hiddens * self.out_len)

        # Residual stack operates as the bottleneck (same as encoder)
        self._residual_stack = ResidualStack(
            num_residual_layers=num_residual_layers,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens
        )

        # Mirror _conv_3 (stride=1, shape-preserving)
        self._deconv_3 = nn.Conv1d(num_hiddens, num_hiddens,
                                   kernel_size=3, stride=1, padding=1)

        # Mirror _conv_2 (stride=2 -> upsample 2x)
        self._deconv_2 = nn.ConvTranspose1d(
            num_hiddens, num_hiddens // 2,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=stride - 1
        )

        # Mirror _conv_1 (stride=2 -> upsample 2x, back to out channels=original input channels)
        self._deconv_1 = nn.ConvTranspose1d(
            num_hiddens // 2, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=stride - 1
        )

    def forward(self, z):
        # z: [B, embedding_dim]
        x = self.linear(z)      # [B, num_hiddens*out_len]
        x = x.view(x.size(0), self.num_hiddens, self.out_len)  #  [B, num_hiddens, out_len]
        
        # Skipped pooling
        #x = x.view(x.size(0), 2*self.num_hiddens, self.pooled_dim) # [B, 2*num_hiddens, pooled_dim]
        #x = self.channel_proj(x)  # [B, num_hiddens, pooled_dim]
        #x = F.interpolate(x, size=self.out_len, mode='linear', align_corners=False) #  [B, num_hiddens, out_len]

        x = self._residual_stack(x)
        x = F.gelu(self._deconv_3(x))
        x = F.gelu(self._deconv_2(x))
        x = self._deconv_1(x)                 # [B, out_channels, signal_len]

        # Trim or pad to signal length
        x = x[..., :self.signal_len]
        x = F.pad(x, (0, max(0, self.signal_len - x.size(-1))))
        return x                              # [B, out_channels, signal_len]


# ---------------VQ-VAE--------------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, commitment_cost, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Inputs: [B, embedding_dim]
        #input_shape = inputs.shape
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return loss, quantized, perplexity
        return loss, quantized, perplexity


# ---------------VQ-VAE, EMA --------------------------------------------------------------------------------

class VectorQuantizerEMA(nn.Module):
    def __init__(self, commitment_cost, decay, num_embeddings, embedding_dim, world_size=1, epsilon=1e-6):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()
        self._embedding.weight.requires_grad = False
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())

        self._decay = decay
        self._epsilon = epsilon
        self.world_size = world_size

    def forward(self, inputs):
        # Inputs: [B, embedding_dim]
        input_shape = inputs.shape
        batch_size = input_shape[0]
        
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding, [B, num_embeddings] one-hot vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantized, [B, embedding_dim] 
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Use EMA to update the embedding vectors
        if self.training:
            new_cluster_size = torch.sum(encodings, 0)
            dw = torch.matmul(encodings.t(), inputs)
            # synchronize across ranks
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(new_cluster_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(dw, op=dist.ReduceOp.SUM)

            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * new_cluster_size 

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) * n
                / (n + self._num_embeddings * self._epsilon) 
            )

            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            # After EMA update, reset dead codes: codes with very low usage
            # Threshold scales with batch size
            dead_threshold = 0.5 * (batch_size * self.world_size) / self._num_embeddings if  self._num_embeddings > 0 else 1.0
            dead_codes = self._ema_cluster_size < dead_threshold
            num_dead = dead_codes.sum().item()
            if num_dead > 0:
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()

                    # Gather this rank's local batch size, then pad every rank's
                    # `inputs` to the max local size so all_gather has matching shapes.
                    local_size = torch.tensor([inputs.size(0)], device=inputs.device)
                    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
                    dist.all_gather(sizes, local_size)
                    sizes = [s.item() for s in sizes]
                    max_size = max(sizes)

                    padded = torch.zeros(max_size, inputs.size(1), device=inputs.device, dtype=inputs.dtype)
                    padded[: inputs.size(0)] = inputs
                    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                    dist.all_gather(gathered, padded)

                    # Trim each rank's chunk back to its true (unpadded) size before concatenating,
                    # so every rank ends up with the identical global candidate pool.
                    all_inputs = torch.cat(
                        [gathered[r][: sizes[r]] for r in range(world_size)], dim=0
                    )

                    # Rank 0 picks the reinit indices and broadcasts them, so every
                    # rank applies the *same* reset — keeping the codebook in sync.
                    if rank == 0:
                        random_indices = torch.randint(
                            0, all_inputs.size(0), (num_dead,), device=inputs.device
                        )
                    else:
                        random_indices = torch.zeros(num_dead, dtype=torch.long, device=inputs.device)
                    dist.broadcast(random_indices, src=0)
                else:
                    all_inputs = inputs
                    random_indices = torch.randint(0, all_inputs.size(0), (num_dead,), device=inputs.device)

                reset_vectors = all_inputs[random_indices].detach()
                self._embedding.weight.data[dead_codes] = reset_vectors
                self._ema_w[dead_codes] = reset_vectors
                # Reset to a small but proportionate value
                reset_val = dead_threshold * 0.5
                self._ema_cluster_size[dead_codes] = reset_val

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        # Return loss, quantized, perplexity
        return loss, quantized, perplexity


class Model(nn.Module):
    def __init__(self, commitment_cost, decay, signal_len, in_channels, 
                 out_channels, num_residual_layers, num_hiddens, num_residual_hiddens,
                 kernel_size, stride, padding, pooled_dim,
                 num_embeddings, embedding_dim, world_size):

        super().__init__()

        # autoencoder with residual stack
        self._encoder = ECGResidualEncoder(signal_len, embedding_dim, num_residual_layers, 
                                           num_hiddens, num_residual_hiddens,
                                           in_channels, kernel_size, stride, 
                                           padding, pooled_dim)

        # simple conv autoencoder
        #self._encoder =  ECGConvEncoder(embedding_dim=embedding_dim, signal_len=signal_len, in_channels=in_channels)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(commitment_cost=commitment_cost, decay=decay,
                                              num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                              world_size=world_size)
        else:
            self._vq_vae = VectorQuantizer(commitment_cost=commitment_cost,
                                           num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        out_len = self._encoder.out_len
        # autoencoder with residual stack
        self._decoder = ECGResidualDecoder(signal_len, out_len, embedding_dim, num_residual_layers, num_hiddens,
                                           num_residual_hiddens, out_channels, kernel_size,
                                           stride, padding, pooled_dim)

        # simple conv autoencoder
        #self._decoder = ECGConvDecoder(embedding_dim=embedding_dim, out_len=out_len, 
        #                               signal_len=signal_len, out_channels=in_channels)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def encode_to_indices(self, x):
        """
        Encode input waveforms to discrete codebook indices.
        Used during inference / codebook assignment pass.
        Returns:
            indices: LongTensor [B] — codebook index for each sample
        """
        z = self._encoder(x)
        distances = (torch.sum(z**2, dim=1, keepdim=True)
                     + torch.sum(self._vq_vae._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z, self._vq_vae._embedding.weight.t()))
        indices = torch.argmin(distances, dim=1)   # [B]
        return indices


# Early stopping
class EarlyStopping:
    def __init__(self, ckpt_key, device, patience=config["patience"], min_delta=config["min_delta"]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.ckpt_key = ckpt_key
        self.device = device

    def __call__(self, val_loss, model):
        improved = self.best_loss is None or val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _save_checkpoint(self, model):
        ckpt_buffer = io.BytesIO()
        torch.save(model.module.state_dict(), ckpt_buffer)
        ckpt_buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket_out,
            Key=self.ckpt_key,
            Body=ckpt_buffer.read(),
        )
        log.info(f"New best val_loss={self.best_loss:.6f}, checkpoint saved to {self.ckpt_key}")

    def load_best_checkpoint(self, model, is_main, global_rank):
        """Load best checkpoint on rank 0 and broadcast state dict to all ranks."""
        if self.best_loss is None:
            log.warning("No checkpoint was saved (no improvement seen); skipping restore.")
            return
        if is_main:
            ckpt_obj = s3_client.get_object(Bucket=bucket_out, Key=self.ckpt_key)
            ckpt_bytes = ckpt_obj["Body"].read()
            ckpt_tensor = torch.frombuffer(bytearray(ckpt_bytes), dtype=torch.uint8).to(self.device)
            length_tensor = torch.tensor([len(ckpt_bytes)], device=self.device)
        else:
            length_tensor = torch.tensor([0], device=self.device)

        dist.broadcast(length_tensor, src=0)

        if not is_main:
            ckpt_tensor = torch.zeros(length_tensor.item(), dtype=torch.uint8, device=self.device)

        dist.broadcast(ckpt_tensor, src=0)

        state_dict = torch.load(
            io.BytesIO(ckpt_tensor.cpu().numpy().tobytes()),
            map_location=self.device,
            weights_only=True,
        )
        model.module.load_state_dict(state_dict)
        best_loss_str = f"{self.best_loss:.6f}" if self.best_loss is not None else "unknown"
        log.info(f"Rank {global_rank}: restored best checkpoint (val_loss={best_loss_str})")


# Padded data loader
class PaddedLoader:
    """Wraps a DataLoader to yield exactly `total_steps` batches.
    Yields exactly `total_steps` batches. If the underlying loader
    is exhausted before total_steps, it is restarted from the beginning
    (with a fresh shuffle if the loader uses shuffle=True)."""
    def __init__(self, loader, total_steps):
        self.loader = loader
        self.total_steps = total_steps

    @property
    def dataset(self):
        return self.loader.dataset   # delegates to the underlying DataLoader
        
    def __iter__(self):
        if len(self.loader) == 0:
            raise RuntimeError(
                f"PaddedLoader wraps an empty loader (0 batches) but total_steps="
                f"{self.total_steps} > 0. This would hang forever (infinite restart "
                "loop). Check dataset sharding / use_frac for this rank."
            )
        seen = 0
        while seen < self.total_steps:
            for batch in self.loader:          # fresh shuffle on every restart
                if seen >= self.total_steps:
                    break
                yield batch
                seen += 1

    def __len__(self):
        return self.total_steps


def setup_ddp():
    dist.init_process_group(
        backend="nccl",  # nccl is optimal for GPU-GPU comms
        timeout=datetime.timedelta(hours=3),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def teardown_ddp():
    dist.destroy_process_group()


def reduce_mean(value: float, count: int, device) -> float:
    """Average a metric across all DDP ranks, weighted by batch count."""
    t = torch.tensor([value, count], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def _get_or_create_patient_splits(
    h5file_list: list[str],
    segment: str,
    global_rank: int,
    world_size: int,
    device,
) -> dict:
    """
    Patient ID collection is sharded across all ranks (same file
    partitioning as RawShardedECGData), so the scan wall-time scales with
    files/world_size instead of files. Splits are cached to S3 for later
    runs, but computed independently (and identically) on every rank once
    they've all exchanged their local patient_id sets — no broadcast needed
    since split_patients() is deterministic given the same input set + seed.
    """
    prefix_out_patients = f"aruna-files/vqvae/segment_{segment}"
    splits_key = f"{prefix_out_patients}/patient_splits_{segment}.json"

    # Rank 0 checks the cache first; broadcast the result so every rank
    # knows whether to skip the scan entirely.
    if global_rank == 0:
        cached = load_patient_splits(splits_key)
        cache_hit = cached is not None
    else:
        cached = None
        cache_hit = False

    cache_hit_tensor = torch.tensor([int(cache_hit)], device=device)
    dist.broadcast(cache_hit_tensor, src=0)
    cache_hit = bool(cache_hit_tensor.item())

    if cache_hit:
        if global_rank == 0:
            log.info(
                f"Loaded cached patient splits: train={len(cached['train'])}, "
                f"val={len(cached['val'])}, test={len(cached['test'])}"
            )
            splits = cached
        else:
            splits = load_patient_splits(splits_key)
            if splits is None:
                raise RuntimeError(
                    f"Rank {global_rank} could not load cached patient splits at {splits_key}"
                )
        dist.barrier()
        return splits

    # No cache — every rank scans its own file shard in parallel.
    my_files = h5file_list[global_rank::world_size]
    log.info(f"[rank {global_rank}] Scanning {len(my_files)}/{len(h5file_list)} files for patient_ids")
    local_patient_ids = collect_unique_patient_ids(my_files, segment)

    # Exchange local sets — this is a collective, but each rank's scan is
    # now bounded by files/world_size, well inside the NCCL watchdog timeout.
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, local_patient_ids)

    all_patient_ids: set = set()
    for s in gathered:
        all_patient_ids |= s

    # Deterministic given the same set + seed — every rank computes the
    # identical splits independently, no broadcast required.
    splits = split_patients(all_patient_ids)

    if global_rank == 0:
        save_patient_splits(splits, splits_key)
        verify = load_patient_splits(splits_key)
        if verify is None:
            raise RuntimeError(f"Rank 0 failed to verify patient splits write to {splits_key}")
        log.info(
            f"Splits computed via parallel scan and saved: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

    dist.barrier()  # just a final sync point, not gating on unbounded work
    return splits
    

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def _write_test_shard(payload, prefix_out, rank):
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    buf.seek(0)
    s3_client.put_object(Bucket=bucket_out, Key=f"{prefix_out}/test_shards/rank_{rank}.npz", Body=buf.read())


def _read_all_test_shards(prefix_out, world_size):
    shards = []
    for r in range(world_size):
        obj = s3_client.get_object(Bucket=bucket_out, Key=f"{prefix_out}/test_shards/rank_{r}.npz")
        arr = np.load(io.BytesIO(obj["Body"].read()), allow_pickle=True)
        shards.append({k: arr[k] for k in arr.files})
    return shards


# Training code
def training(segment, embedding_dim, num_embeddings, signal_len, in_channels):

    # DDP
    setup_ddp()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main = global_rank == 0   # only rank 0 logs and saves

    device = torch.device(f"cuda:{local_rank}")

    # S3 location for outputs
    prefix_out = f"aruna-files/vqvae/segment_{segment}_nleads_{in_channels}"
    # Checkpoint location
    best_ckpt_key = f"{prefix_out}/best_model_checkpoint_{segment}.pt"

    # Hyperparameters
    # Training
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    # AdamW with weight decay
    weight_decay = config["weight_decay"]
    # learning rate: warmup with cosine weight decay
    learning_rate = config["lr"]
    warmup_epochs = config["warmup_epochs"]
    min_lr_factor = config["min_lr_factor"]
    # Gradient clipping
    grad_clip_max_norm = config["grad_clip_max_norm"]
    # Conv autoencoder with residual stack
    num_hiddens = res_config["num_hiddens"]
    num_residual_hiddens = res_config["num_residual_hiddens"]
    num_residual_layers = res_config["num_residual_layers"]
    kernel_size = res_config["kernel_size"]
    stride = res_config["stride"]
    padding = res_config["padding"]
    pooled_dim = config["pooled_dim"]
    # VQ-VAE 
    #embedding_dim = embedding_dim  # redundant self-assignment
    #num_embeddings = num_embeddings  # redundant self-assignment
    commitment_cost = COMMITMENT_COST
    decay = DECAY
    # Input data parameters
    #segment = segment    # redundant
    #signal_len = signal_len   # redundant
    #in_channels = in_channels   # redundant
    out_channels = in_channels

    # Define lead_indices: [1] for lead II if 1 channel, or all 12 channels
    if in_channels==1:
        lead_indices = [1]   # lead II
    elif in_channels==NLEADS:
        lead_indices = list(range(NLEADS))
    else:
        raise ValueError("Number of channels does not match lead II vs. all leads options")

    # Input data, read file list on rank 0 and broadcast
    if global_rank == 0:
        full_h5filelist = list_h5_files()
        # Serialize to JSON string for broadcast
        filelist_str = json.dumps(full_h5filelist)
        filelist_bytes = filelist_str.encode()
        length_tensor = torch.tensor([len(filelist_bytes)], device=device)
    else:
        length_tensor = torch.tensor([0], device=device)
    
    dist.broadcast(length_tensor, src=0)
    
    if global_rank == 0:
        data_tensor = torch.frombuffer(bytearray(filelist_bytes), dtype=torch.uint8).to(device)
    else:
        data_tensor = torch.zeros(length_tensor.item(), dtype=torch.uint8, device=device)
    
    dist.broadcast(data_tensor, src=0)
    
    if global_rank != 0:
        full_h5filelist = json.loads(data_tensor.cpu().numpy().tobytes().decode())
    
    h5filelist = full_h5filelist[:NUM_h5_FILES] if NUM_h5_FILES and NUM_h5_FILES > 0 else full_h5filelist
    
    if is_main:
        log.info(f"Using {len(h5filelist)} h5 files")

    # ---- patient splits (rank-0 computes once, all ranks load) ----
    splits = _get_or_create_patient_splits(h5filelist, segment, global_rank, world_size, device)

    # ---- datasets: each rank loads its file shard into RAM ----
    # Load this rank's file shard ONCE, for all patients
    raw_shard = RawShardedECGData(
        segment=segment,
        signal_len=signal_len,
        h5file_list=h5filelist,
        lead_indices=lead_indices,
        rank=global_rank,
        world_size=world_size,
    )

    # No DistributedSampler; data is already partitioned at the file level.
    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        ds = FileShardedECGDataset(
            raw_shard,
            patient_ids=splits[split_name],
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=4,       # pure numpy indexing — no I/O in workers
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context='forkserver',
            drop_last=False,   # uses padded train loader
        )
 
    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)

    # After creating loaders, before the epoch loop:
    # synchronize the number of training steps across all ranks and pad loaders
    def get_loader_len(loader, device):
        length = torch.tensor([len(loader)], dtype=torch.long, device=device)
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
        return length.item()

    train_steps = get_loader_len(train_loader, device)
    # Wrap train loader — ranks with fewer batches cycle; ranks at MAX just iterate normally
    train_loader = PaddedLoader(train_loader, train_steps)
 
    # Log shard sizes and imbalance once all ranks have loaded
    my_train_size = torch.tensor([len(train_loader.dataset)], device=device)
    all_train_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_train_sizes, my_train_size)
    if is_main:
        sizes = [int(s.item()) for s in all_train_sizes]
        ratio = max(sizes) / max(min(sizes), 1)
        log.info(f"Train samples per rank: {sizes}  (max/min={ratio:.2f})")
        if ratio > 1.5:
            log.warning(
                "Shard imbalance >1.5×. Consider offline re-sharding by patient_id "
                "if this causes straggler slowdowns."
            )

    # Model training
    model = Model(commitment_cost, decay, signal_len, in_channels, out_channels,
                  num_residual_layers, num_hiddens, num_residual_hiddens,
                  kernel_size, stride, padding, pooled_dim,
                  num_embeddings, embedding_dim, world_size)
    model = model.to(device)
    # Sync initial weights across all ranks before wrapping in DDP
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf, src=0)
    #model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer AdamW: # Only optimize parameters that actually require gradients
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=False,
    )

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs         # ramps 1/W, 2/W, ... W/W
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Early stopping
    early_stopping = EarlyStopping(
        ckpt_key=best_ckpt_key,
        device=device,
    )

    # Masked recostruction loss
    def masked_recon_loss(x_recon, x, orig_lens):
        # Mask
        pos = torch.arange(signal_len, device=device).unsqueeze(0)   # [1,L]
        valid_mask = pos < orig_lens.unsqueeze(1)   # [1,L], [B,1] -> [B,L]
        valid_mask = valid_mask.unsqueeze(1).float()   # [B, C, L]
        # Masked recon error
        criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")  # Huber
        #criterion = nn.MSELoss(reduction='none')  # MSE
        recon_error = criterion(x_recon, x)
        recon_loss = (recon_error * valid_mask).sum() / (valid_mask.sum() * x.size(1) + 1e-8)
        return recon_loss
    
    train_recon_error, train_loss, train_perplexity = [], [], []
    val_recon_error, val_loss, val_perplexity = [], [], []

    for i in range(epochs):
        # Training
        model.train() 
        train_epoch_loss, train_epoch_recon_error, train_epoch_perplexity = 0.0, 0.0, 0.0

        for batch in train_loader:
            train_data = batch["waveform"].to(device, non_blocking=True)
            orig_lens = batch["orig_len"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
        
            vq_loss, data_recon, perplexity = model(train_data)
            if not MASK:
                criterion = nn.SmoothL1Loss(beta=1.0)   # Huber
                #criterion = nn.MSELoss()    # MSE
                recon_error = criterion(data_recon, train_data)
            else:
                recon_error = masked_recon_loss(data_recon, train_data, orig_lens)
                
            loss = recon_error + vq_loss
            loss.backward() # DDP all-reduces gradients automatically here
            # Gradient clipping
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=grad_clip_max_norm, norm_type=2.0)
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_recon_error += recon_error.item()
            train_epoch_perplexity += perplexity.item()

        dist.barrier()   # all ranks done with training batches
        train_epoch_loss        = reduce_mean(train_epoch_loss, train_steps, device)
        train_epoch_recon_error = reduce_mean(train_epoch_recon_error, train_steps, device)
        train_epoch_perplexity  = reduce_mean(train_epoch_perplexity, train_steps, device)
        # Per epoch
        if is_main:
            train_loss.append(train_epoch_loss)
            train_recon_error.append(train_epoch_recon_error)
            train_perplexity.append(train_epoch_perplexity)

        # Validation
        model.eval()
        eval_model = unwrap_model(model)
        with torch.no_grad():  
            val_epoch_loss, val_epoch_recon_error, val_epoch_perplexity = 0.0, 0.0, 0.0
            num_val_batches = 0
            for batch in val_loader:
                val_data = batch["waveform"].to(device, non_blocking=True)
                orig_lens = batch["orig_len"].to(device, non_blocking=True)
                vq_loss, data_recon, perplexity = eval_model(val_data)
                if not MASK:
                    criterion = nn.SmoothL1Loss(beta=1.0)      # Huber
                    #criterion = nn.MSELoss()       # MSE
                    recon_error = criterion(data_recon, val_data)
                else:
                    recon_error = masked_recon_loss(data_recon, val_data, orig_lens)
                loss = recon_error + vq_loss

                val_epoch_loss += loss.item()
                val_epoch_recon_error += recon_error.item()
                val_epoch_perplexity += perplexity.item()
                num_val_batches += 1

        dist.barrier()   # all ranks done with val batches
        val_epoch_loss = reduce_mean(val_epoch_loss, num_val_batches, device)
        val_epoch_recon_error = reduce_mean(val_epoch_recon_error, num_val_batches, device)
        val_epoch_perplexity = reduce_mean(val_epoch_perplexity, num_val_batches, device)
        # Per epoch
        if is_main:
            val_loss.append(val_epoch_loss)
            val_recon_error.append(val_epoch_recon_error)
            val_perplexity.append(val_epoch_perplexity)

            # Logging per log steps for train and val metrics
            if (i+1) % log_step == 0:
                current_lr = scheduler.get_last_lr()[0]
                log.info(f"""Epoch {i}: lr = {current_lr:.6f}""")
                log.info(f"""Epoch {i}: train recon_error = {train_epoch_recon_error:.4f},
                train loss = {train_epoch_loss:.4f}, 
                train perplexity = {train_epoch_perplexity:.4f}""")
                log.info(f"""Epoch {i}: val recon_error = {val_epoch_recon_error:.4f}, 
                val loss = {val_epoch_loss:.4f}, 
                val perplexity = {val_epoch_perplexity:.4f}""")
                enc_grad = sum(p.grad.norm().item() for p in model.module._encoder.parameters() if p.requires_grad and p.grad is not None)
                dec_grad = sum(p.grad.norm().item() for p in model.module._decoder.parameters() if p.requires_grad and p.grad is not None)
                log.info(f"Epoch {i} grad norms - encoder: {enc_grad:.4f}, decoder: {dec_grad:.4f}")

            # Early stopping
            early_stopping(val_epoch_loss, model)

        # Broadcast early-stop decision from rank 0 to all ranks
        stop_tensor = torch.tensor(int(early_stopping.early_stop), device=device)
        dist.broadcast(stop_tensor, src=0)
        # Learning rate scheduler
        scheduler.step()
        if stop_tensor.item():
            if is_main:
                log.info(f"Early stopping at epoch {i}")
            break

    # Test dataset on Trained model
    # Restore best checkpoint on all ranks before test eval
    dist.barrier()
    early_stopping.load_best_checkpoint(model, is_main, global_rank)
    dist.barrier()

    # Every rank filters its ALREADY-LOADED raw_shard down to test patient_ids
    # — pure numpy, zero additional S3/h5py I/O.
    test_patient_ids = np.fromiter(splits["test"], dtype=object, count=len(splits["test"]))
    test_mask = np.isin(raw_shard.patient_ids, test_patient_ids)
    test_idx = np.where(test_mask)[0]

    local_test_payload = {
        "data":         raw_shard.data[test_idx],
        "patient_ids":  raw_shard.patient_ids[test_idx],
        "study_ids":    raw_shard.study_ids[test_idx],
        "beat_idxs":    raw_shard.beat_idxs[test_idx],
        "orig_lens":    raw_shard.orig_lens[test_idx],
    }
    
    # Collective call — ALL ranks must participate, even though only rank 0
    # receives the gathered result (object_gather_list must be None elsewhere)
    #gathered = [None] * world_size if is_main else None
    #dist.gather_object(local_test_payload, gathered, dst=0)

    _write_test_shard(local_test_payload, prefix_out, global_rank)
    dist.barrier()

    if is_main:
        gathered = _read_all_test_shards(prefix_out, world_size)
        test_data        = np.concatenate([g["data"]        for g in gathered], axis=0)
        test_patient_arr = np.concatenate([g["patient_ids"]  for g in gathered], axis=0)
        test_study_arr   = np.concatenate([g["study_ids"]    for g in gathered], axis=0)
        test_beat_arr    = np.concatenate([g["beat_idxs"]    for g in gathered], axis=0)
        test_orig_arr    = np.concatenate([g["orig_lens"]    for g in gathered], axis=0)
    
        test_ds = InMemoryECGDataset(
            test_data, test_patient_arr, test_study_arr, test_beat_arr, test_orig_arr
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=4,
            pin_memory=True,
            multiprocessing_context='forkserver',
        )
        
        model.eval()
        eval_model = unwrap_model(model)
        test_loss, test_recon_error, test_perplexity = 0.0, 0.0, 0.0
        with torch.no_grad():  
            for batch in test_loader:
                test_data_batch = batch["waveform"].to(device, non_blocking=True)
                orig_lens = batch["orig_len"].to(device, non_blocking=True)
                vq_loss, data_recon, perplexity = eval_model(test_data_batch)
                if not MASK:
                    criterion = nn.SmoothL1Loss(beta=1.0)     # Huber
                    #criterion = nn.MSELoss()       # MSE
                    recon_error = criterion(data_recon, test_data_batch)
                else:
                    recon_error = masked_recon_loss(data_recon, test_data_batch, orig_lens)
                loss = recon_error + vq_loss
                
                test_loss += loss.item()
                test_recon_error += recon_error.item()
                test_perplexity += perplexity.item()
        test_loss = test_loss / len(test_loader)
        test_recon_error = test_recon_error / len(test_loader)
        test_perplexity = test_perplexity / len(test_loader)
    
        log.info(f"""test recon_error = {test_recon_error:.4f}, 
        test loss = {test_loss:.4f}, 
        test perplexity = {test_perplexity:.4f}""")

        # Save outputs - unwrap DDP with .module before pickling
        # Output files
        training_output_filename = f"training_outputs_{segment}.npz"
        training_output_filepath = f"s3://{bucket_out}/{prefix_out}/{training_output_filename}"
        model_output_filename = f"vqvae_model_{segment}.pkl"
        model_output_key = f"{prefix_out}/{model_output_filename}"
        model_output_statedict = f"vqvae_model_{segment}.pt"
        statedict_output_path = f"s3://{bucket_out}/{prefix_out}/{model_output_statedict}"

        buffer = io.BytesIO()
        np.savez_compressed(buffer,
                            train_loss=train_loss,
                            train_recon_error=train_recon_error,
                            train_perplexity=train_perplexity,
                            val_loss=val_loss,
                            val_recon_error=val_recon_error,
                            val_perplexity=val_perplexity,
                            test_loss=[test_loss],
                            test_recon_error=[test_recon_error],
                            test_perplexity=[test_perplexity],
                            vqvae_codebook=model.module._vq_vae._embedding.weight.detach().cpu().numpy(),
                           )
        
        buffer.seek(0)
        with s3_fs.open(training_output_filepath, 'wb') as f:
            f.write(buffer.read())
    
        # Save model
        # Serialize the entire model object with pickle
        model_bytes = pickle.dumps(model.module)
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_out,
            Key=model_output_key,   
            Body=model_bytes,
        )

        # Save state dict
        model_buffer = io.BytesIO()
        torch.save({
            "state_dict": model.module.state_dict(),
            "config": dict(commitment_cost=commitment_cost, decay=decay, signal_len=signal_len,
                           in_channels=in_channels, out_channels=out_channels,
                           num_residual_layers=num_residual_layers, num_hiddens=num_hiddens,
                           num_residual_hiddens=num_residual_hiddens, kernel_size=kernel_size,
                           stride=stride, padding=padding, pooled_dim=pooled_dim,
                           num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                           world_size=world_size),
        }, model_buffer)
        model_buffer.seek(0)
        with s3_fs.open(statedict_output_path, 'wb') as f:
            f.write(model_buffer.read())
        
        log.info("Saved model and training outputs")

    teardown_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__=="__main__":
    import argparse

    rank_str = os.environ.get("RANK", "?")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [rank{rank_str}] %(message)s",
    )

    p = argparse.ArgumentParser(description="VQ-VAE model training")
    p.add_argument("--segment", default="P", choices=SEGMENTS)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--num_embeddings", type=int, default=32)
    p.add_argument("--signal_len", type=int, default=176)
    p.add_argument("--in_channels", type=int, default=1, choices=[1,NLEADS])
    args = p.parse_args()
    
    training(args.segment, args.embedding_dim, args.num_embeddings, 
             args.signal_len, args.in_channels)