import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math
import os
import time
import io
import re
import s3fs
import argparse
import logging
from datetime import timedelta

# ================== DDP imports & helpers ==================

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ================== Dataset imports ==================

from vqvae_bert_sentence_dataset import (
    create_bert_dataloaders,
    MAX_SENTENCE_LEN,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    FIRST_REAL_TOKEN_ID,
    bucket_out,
    base_prefix,
)
from vqvae_model import PaddedLoader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================

CONFIG = {
    'batch_size': 256,
    'n_layers': 8,
    'n_heads': 12,
    'd_model': 384,
    'num_epochs': 200,
    'patience': 10,
    'min_delta': 1e-6,
    'max_lr': 1e-4,
    'mask_ratio': 0.15,          # 0.2
    'weight_decay': 3e-4,
    'train_ratio': 0.85,
    'seed': 42,
    'warmup_epochs': 10,
    'min_lr_factor': 0.1,
    'use_cnn_features': True,
    'bert_dropout': 0.1,
    'cnn_scale_init': 1.0,
}


# ------------------- CLI overrides -------------------
# Wrapped in a function and only called under main
# Parsing argv at module import time breaks any caller that imports this
# module without wanting its CLI behavior — in particular a Ray worker
# process, whose argv contains Ray's own launch args, not this script's.
def apply_cli_overrides():
    parser = argparse.ArgumentParser(description="BERT pretraining arguments")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--max_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--bert_dropout", type=float)
    parser.add_argument("--cnn_scale_init", type=float)
    parser.add_argument("--use_cnn_features", action=argparse.BooleanOptionalAction)

    cli_args, _ = parser.parse_known_args()
    for k in ["batch_size", "n_layers", "n_heads", "d_model", "bert_dropout", "cnn_scale_init",
              "max_lr", "weight_decay", "mask_ratio", "use_cnn_features"]:
        if getattr(cli_args, k) is not None:
            CONFIG[k] = getattr(cli_args, k)


# ================== DDP helpers ==================

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    return not ddp_is_initialized() or dist.get_rank() == 0

def ddp_allreduce_scalar(scalar: float, device: torch.device) -> float:
    t = torch.tensor(scalar, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ================== Dynamic MLM masking ==================

def apply_dynamic_masking(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_ratio: float,
    device: torch.device,
) -> tuple:
    """
    Apply dynamic MLM masking (80% [MASK] / 10% random / 10% unchanged).
    Only real beat tokens — not PAD (0) or SEP (1) — are eligible.
    vocab_size: includes PAD, SEP, MASK, and codebook sizes of all waveforms

    Returns
    -------
    masked_input_ids : (B, L)         input_ids with some tokens replaced
    masked_pos       : (B, max_masks) positions of masked tokens, zero-padded
    masked_tokens    : (B, max_masks) original token ids at those positions, zero-padded
    """

    B, L = input_ids.shape
    maskable = (input_ids != SEP_TOKEN_ID) & (input_ids != PAD_TOKEN_ID)  # (B, L)

    # Sample a uniform random score per position, set non-maskable to inf so they're never chosen
    scores = torch.rand(B, L, device=device)
    scores[~maskable] = 2.0  # push non-maskable positions to the back

    max_masks = max(1, math.ceil(L * mask_ratio))
    # Take the top-k lowest scores as masked positions
    _, all_masked_pos = scores.topk(max_masks, dim=1, largest=False, sorted=False)  # (B, max_masks)

    all_masked_tokens = input_ids.gather(1, all_masked_pos)  # (B, max_masks) original tokens

    # Build masked input
    masked_input_ids = input_ids.clone()
    rand_vals = torch.rand(B, max_masks, device=device)

    mask_mask = rand_vals < 0.8                             # 80%: replace with MASK
    rand_mask = (rand_vals >= 0.8) & (rand_vals < 0.9)     # 10%: replace with random token
    # 10%: unchanged — no action needed

    replace_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    replace_rand = torch.zeros(B, L, dtype=torch.bool, device=device)
    replace_mask.scatter_(1, all_masked_pos, mask_mask)
    replace_rand.scatter_(1, all_masked_pos, rand_mask)

    masked_input_ids[replace_mask] = MASK_TOKEN_ID
    if replace_rand.any():
        n_rand = replace_rand.sum().item()
        masked_input_ids[replace_rand] = torch.randint(
            FIRST_REAL_TOKEN_ID, vocab_size, (n_rand,), device=device
        )

    return masked_input_ids, all_masked_pos, all_masked_tokens


# ================== BERT ==================

def get_attn_pad_mask(seq_q, seq_k):
    '''
    Marks PAD positions so they are excluded from attention
    for key sequence, broadcast to (batch_size, len_q, len_k)
    '''
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD_TOKEN_ID)  # (batch_size, len_k)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(-1, len_q, -1)
    # broadcast to (batch_size, len_q, len_k)
    return pad_attn_mask.to(seq_q.device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # q_s, k_s shape: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # shape: (batch_size, num_heads, seq_len, seq_len)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # (batch_size, num_heads, seq_len, d_v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attn = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V: batch_size, seq_len, d_model
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.fc1(inputs))
        output = self.dropout(output)
        output = self.fc2(output)
        return self.norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, use_cnn_features, cnn_embed_dim, cnn_scale_init):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.cnn_norm = (
            nn.LayerNorm(cnn_embed_dim, elementwise_affine=False)  #  # per-vector standardize -> unit scale, zero mean
            if (use_cnn_features and cnn_embed_dim > 0) else None
        )
        self.cnn_linear = (
            nn.Linear(cnn_embed_dim, d_model)
            if (use_cnn_features and cnn_embed_dim > 0) else None
        )
        self.cnn_scale = (
            nn.Parameter(torch.tensor(cnn_scale_init))  # small init
            if (use_cnn_features and cnn_embed_dim > 0) else None
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, use_cnn_features, cnn_features):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        tok_embedding = self.tok_embed(x)
        pos_embedding = self.pos_embed(pos)
        combined = tok_embedding + pos_embedding 
        if use_cnn_features and cnn_features is not None:
            cnn_embedding = self.cnn_linear(self.cnn_norm(cnn_features))*self.cnn_scale
            combined += cnn_embedding
        return self.norm(combined)


class BERT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads, d_k, d_v, d_ff, use_cnn_features, cnn_embed_dim, cnn_scale_init, dropout=0.1):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_seq_len, d_model, use_cnn_features, cnn_embed_dim, cnn_scale_init)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.activ = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        # Decoder weight tying is done in forward
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, use_cnn_features, cnn_features, masked_pos=None):
        output = self.embedding(input_ids, use_cnn_features, cnn_features)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        # output shape: [batch_size, seq_len, d_model]
        # During pre-training, uses logits
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(self.activ(self.linear(h_masked)))
            # Decoder weight tying
            logits_lm = F.linear(h_masked, self.embedding.tok_embed.weight, self.decoder_bias)
            return logits_lm
        else:
            # mask shape: [batch_size, seq_len, 1]
            mask = (input_ids != PAD_TOKEN_ID).unsqueeze(-1).float()
            masked_output = output * mask      # [batch_size, seq_len, d_model]
            h_pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)     # [batch_size, d_model]
            return h_pooled


def calculate_mlm_accuracy_counts(logits_lm, masked_tokens):
    """Return (#correct, #valid) so we can aggregate across DDP ranks properly."""
    pred_tokens = logits_lm.argmax(dim=2)
    valid_mask = masked_tokens.ne(0) & masked_tokens.ne(SEP_TOKEN_ID)  # Ignore 0 slots and SEP
    correct = (pred_tokens == masked_tokens) & valid_mask
    return correct.sum().item(), valid_mask.sum().item()


def evaluate_bert_loader(model, loader, use_cnn_features, cnn_embed_dim, device, criterion, vocab_size, mask_ratio):
    if ddp_is_initialized():
        dist.barrier()  # ensure all ranks enter eval together
    rank_id = dist.get_rank() if ddp_is_initialized() else 0

    local_batches = len(loader)
    log.info(f"Rank {rank_id}: entering eval, {local_batches} local batches")

    model.eval()
    loss_sum    = 0.0
    acc_correct = 0
    acc_total   = 0
    num_batches = 0

    val_iter = iter(loader)

    with torch.no_grad():
        for batch in val_iter:

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            
            # Dynamic masking of tokens and CNN embeddings
            masked_input_ids, masked_pos, masked_tokens = apply_dynamic_masking(
                input_ids, vocab_size, mask_ratio, device
            )

            if use_cnn_features and cnn_embed_dim > 0:
                cnn_features = batch["cnn_embeddings"].to(device, non_blocking=True).clone()
                cnn_mask_bool = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
                cnn_mask_bool.scatter_(1, masked_pos, True)
                cnn_features[cnn_mask_bool] = 0.0
            else:
                cnn_features = None

            logits_lm = model(masked_input_ids, use_cnn_features, cnn_features, masked_pos)
            # logits_lm: [batch_size, max_masks, vocab_size]
            # CrossEntropyLoss requires logits shape: [batch_size, vocab_size, max_masks]
            masked_tokens_for_loss = masked_tokens.masked_fill(masked_tokens == SEP_TOKEN_ID, 0)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens_for_loss)
            correct, total = calculate_mlm_accuracy_counts(logits_lm, masked_tokens)

            if not torch.isfinite(loss_lm):
                if is_main_process():
                    log.warning(f"Non-finite val loss on rank {rank_id} — skipping this batch from aggregation")
                continue
                
            loss_sum    += loss_lm.item()
            acc_correct += correct
            acc_total   += total
            num_batches += 1

    # ---- DDP: aggregate sums across ranks ----
    if ddp_is_initialized():
        stats = torch.tensor(
            [float(loss_sum), float(num_batches), float(acc_correct), float(acc_total)],
            dtype=torch.float64, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, num_batches, acc_correct, acc_total = stats.tolist()

    loss_avg = loss_sum / max(1.0, num_batches)
    acc_avg  = acc_correct / max(1.0, acc_total)
    return loss_avg, acc_avg


def mlm_pretraining():

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(CONFIG["seed"] + rank)

    num_epochs = CONFIG["num_epochs"]
    batch_size = CONFIG["batch_size"]
    mask_ratio = CONFIG["mask_ratio"]
    train_ratio = CONFIG["train_ratio"]

    train_losses, train_mlm_accuracies = [], []
    val_losses, val_mlm_accuracies = [], []

    # Discover all sentence shards written by vqvae_bert_sentence_dataset.py
    _s3 = s3fs.S3FileSystem()
    raw_paths = _s3.glob(f"{bucket_out}/{base_prefix}/bert/beat_sentences_*.npz")
    all_shard_keys = sorted(p.replace(f"{bucket_out}/", "") for p in raw_paths)

    if not all_shard_keys:
        # Destroy process group before raising so other ranks aren't left hanging
        if ddp_is_initialized():
            dist.destroy_process_group()
        raise RuntimeError("No sentence shards found on S3 in bucket_out and base_prefix")
    
    if is_main_process():
        log.info(f"Found {len(all_shard_keys)} sentence shard(s) on S3")
    
    m = re.search(r"nleads_(\d+)", all_shard_keys[0])
    in_channels = int(m.group(1)) if m else 1
    
    train_loader, val_loader = create_bert_dataloaders(
        all_shard_keys=all_shard_keys,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=CONFIG["seed"],
    )

    if is_main_process():
        log.info(f"Dataloaders ready. Train: {len(train_loader.dataset):,} samples, "
                 f"Val: {len(val_loader.dataset):,} samples")
        
    # After creating loaders, before the epoch loop:
    # synchronize the number of training steps across all ranks and pad loaders
    batches = torch.tensor([len(train_loader)], dtype=torch.long, device=device)
    dist.all_reduce(batches, op=dist.ReduceOp.MAX)
    max_batches = batches.item()
    # Wrap train loader — ranks with fewer batches cycle; ranks at MAX just iterate normally
    padded_train_loader = PaddedLoader(train_loader, max_batches)

    # Derive BERT hyper-parameters from the dataset and CONFIG
    vocab_size    = train_loader.dataset.vocab_size
    max_seq_len   = MAX_SENTENCE_LEN
    d_model       = CONFIG["d_model"]
    n_layers      = CONFIG["n_layers"]
    n_heads       = CONFIG["n_heads"]
    d_k           = d_model // n_heads
    d_v           = d_model // n_heads
    d_ff          = d_model * 4
    bert_dropout = CONFIG["bert_dropout"]
    use_cnn_features = CONFIG["use_cnn_features"]
    cnn_embed_dim = train_loader.dataset.cnn_embedding_dim if use_cnn_features else 0
    cnn_scale_init = CONFIG["cnn_scale_init"]

    bert = BERT(
        vocab_size, max_seq_len, d_model, n_layers, n_heads,
        d_k, d_v, d_ff, use_cnn_features, cnn_embed_dim, cnn_scale_init, bert_dropout
    ).to(device)
    model = DDP(bert, device_ids=[local_rank], output_device=local_rank)
    # find_unused_parameters=True needed only if h_pooled used unused params not needed in MLM
    model = torch.compile(model)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0, reduction='mean') # ignore 0 slots in MLM masked tokens
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=CONFIG["max_lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.99),
    )

    # Learning rate cosine annealing
    warmup_epochs = CONFIG["warmup_epochs"]
    min_lr_factor = CONFIG["min_lr_factor"]
    
    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs         # ramps 1/W, 2/W, ... W/W
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience          = CONFIG["patience"]
    min_delta         = CONFIG["min_delta"]
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    global_step       = 0

    for epoch in range(num_epochs):
        model.train()
      
        epoch_loss_local  = 0.0
        mlm_correct_local = 0
        mlm_total_local   = 0
        num_batches_local = 0
        start_time        = time.time()

        for i, batch in enumerate(padded_train_loader):
        
            if i % 20 == 0 and is_main_process():
                log.info(f"  Epoch {epoch+1}, batch {i}/{max_batches}, "
                         f"elapsed={( time.time()-start_time)/60:.1f}min")
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            # Dynamic masking of tokens and CNN embeddings
            masked_input_ids, masked_pos, masked_tokens = apply_dynamic_masking(
                input_ids, vocab_size, mask_ratio, device
            )
            
            if use_cnn_features and cnn_embed_dim > 0:
                cnn_features = batch["cnn_embeddings"].to(device, non_blocking=True).clone()
                cnn_mask_bool = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
                cnn_mask_bool.scatter_(1, masked_pos, True)
                cnn_features[cnn_mask_bool] = 0.0   # mask the same positions as masked_input_ids
            else:
                cnn_features = None

            optimizer.zero_grad(set_to_none=True)

            logits_lm = model(masked_input_ids, use_cnn_features, cnn_features, masked_pos)
            masked_tokens_for_loss = masked_tokens.masked_fill(masked_tokens == SEP_TOKEN_ID, 0)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens_for_loss)
            correct, total = calculate_mlm_accuracy_counts(logits_lm, masked_tokens)

            # Check to make sure loss_lm does not have NaNs, for example, no valid masks in the batch if all sequences were very short
            # Check finiteness locally, then sync across ranks so every rank makes the same
            # skip/proceed decision — backward() is a collective op under DDP and cannot be
            # called on a subset of ranks without deadlocking.
            local_bad = torch.tensor(
                [0.0 if torch.isfinite(loss_lm) else 1.0], device=device
            )
            if ddp_is_initialized():
                dist.all_reduce(local_bad, op=dist.ReduceOp.SUM)
            any_bad = local_bad.item() > 0
            
            if any_bad:
                if is_main_process():
                    log.warning(f"Non-finite loss on at least one rank at epoch {epoch+1}, batch {i} — all ranks skipping this batch's update")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_lm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_local  += loss_lm.item()
            mlm_correct_local += correct
            mlm_total_local   += total
            num_batches_local += 1
            global_step       += 1

        # ---- DDP: aggregate training stats for logging ----
        train_loss_epoch    = float(epoch_loss_local)
        train_batches_epoch = float(num_batches_local)
        train_correct_epoch = float(mlm_correct_local)
        train_total_epoch   = float(mlm_total_local)

        # Ensure all ranks finished taining loop
        if ddp_is_initialized():
            dist.barrier() 

        if ddp_is_initialized():
            stats = torch.tensor(
                [train_loss_epoch, train_batches_epoch, train_correct_epoch, train_total_epoch],
                dtype=torch.float64, device=device
                )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss_epoch, train_batches_epoch, train_correct_epoch, train_total_epoch = stats.tolist()

        epoch_loss    = train_loss_epoch    / max(1.0, train_batches_epoch)
        mlm_acc_total = train_correct_epoch / max(1.0, train_total_epoch)
        train_losses.append(epoch_loss)
        train_mlm_accuracies.append(mlm_acc_total)

        val_loss, val_accuracy = evaluate_bert_loader(
            model, val_loader, use_cnn_features, cnn_embed_dim, device, criterion, vocab_size, mask_ratio
        )
        
        val_losses.append(val_loss)
        val_mlm_accuracies.append(val_accuracy)

        scheduler.step()

        end_time = time.time()
        # Logging results
        if is_main_process():
            log.info(f"Epoch: {epoch + 1}")
            log.info(f"Train loss: {epoch_loss:.4f}, Train MLM Accuracy: {mlm_acc_total:.4f}")
            log.info(f"Val loss: {val_loss:.4f}, Val MLM Accuracy: {val_accuracy:.4f}")
            log.info(f"{(end_time - start_time) / 60.0:.2f} minutes")

        # Early stopping
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if is_main_process():
            improved = (best_val_loss - val_loss) > min_delta
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model_checkpt = model.module.state_dict()
                buf = io.BytesIO()
                torch.save(model_checkpt, buf)
                buf.seek(0)
                s3_model_key = f"{bucket_out}/{base_prefix}/bert_pretraining/bert_model_nleads_{in_channels}.pt"
                with _s3.open(s3_model_key, "wb") as f:
                    f.write(buf.read())
                log.info(f"Model saved to s3://{s3_model_key}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"Early stopping after {epoch + 1} epochs. "
                          f"No improvement in validation loss for {patience} consecutive epochs.")
                    stop_flag[0] = 1

        if ddp_is_initialized():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            break

    train_results_key = (
        f"{bucket_out}/{base_prefix}/bert_pretraining/"
        f"bert_pretrain_results_nleads_{in_channels}_d{d_model}_l{n_layers}_h{n_heads}.npz"
    )

    if is_main_process():
        log.info("Model training is done")
        results_buf = io.BytesIO()
        np.savez(
            results_buf,
            epoch=np.arange(1, len(train_losses) + 1),
            train_loss=np.array(train_losses),
            train_accuracy=np.array(train_mlm_accuracies),
            val_loss=np.array(val_losses),
            val_accuracy=np.array(val_mlm_accuracies),
        )
        results_buf.seek(0)
        with _s3.open(train_results_key, "wb") as f:
            f.write(results_buf.read())
        log.info(f"Train results have been saved to s3://{train_results_key}")

    if ddp_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    apply_cli_overrides()
    mlm_pretraining()
