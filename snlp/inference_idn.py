"""
Comprehensive IDN layer-parallel inference evaluation.

Methods: Sequential, IDN batched, Fused, Fused+split_o_mlp, Chunkwise
Init strategies: h0, warm, batch_fwd, preheat
Evaluations: PPL (1M tokens), PPL timing, AR generation, AR timing

Usage:
    python -m scripts.eval_jacobi_idn_new --model-tag d32s_idn05_npar24_4800 --n-par 8
"""

import argparse
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.engine import KVCache
from nanochat.jacobi_forward import _embed, _run_block, _logits, PreheatCache, calibrate_preheat

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

device = torch.device(autodetect_device_type())
dtype = COMPUTE_DTYPE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_tag, step=None, device=None, cache_hc=False):
    from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
    from nanochat.gpt import GPT, GPTConfig
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    path = os.path.join(base_dir, 'base_checkpoints', model_tag)
    if step is None:
        step = find_last_step(path)
    print(f"  Loading {path} step {step}")
    md, _, meta = load_checkpoint(path, step, device)
    md = {k.removeprefix("_orig_mod."): v for k, v in md.items()}
    cfg = meta["model_config"]
    _patch_missing_config_keys(cfg)
    config = GPTConfig(**cfg)
    _patch_missing_keys(md, config)
    md = {k: v.to(device) for k, v in md.items()}
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(md, strict=True, assign=True)
    model.eval()
    if cache_hc and getattr(config, 'use_mhc', False):
        for block in model.transformer.h:
            block.hc_attn.cache_hc_matrices(config.mhc_sinkhorn_iters, config.mhc_sinkhorn_tau)
            block.hc_mlp.cache_hc_matrices(config.mhc_sinkhorn_iters, config.mhc_sinkhorn_tau)
    return model


# ---------------------------------------------------------------------------
# Weight precomputation (offline, not timed)
# ---------------------------------------------------------------------------

class PrecomputedWeights:
    """Cache stacked and fused weights for all methods."""

    def __init__(self, model, par):
        blocks = [model.transformer.h[i] for i in par]
        dt = COMPUTE_DTYPE
        n_par = len(par)

        # Stacked weights for IDN batched / fused+split_o_mlp
        self.W_q_s = torch.stack([b.attn.c_q.weight for b in blocks]).to(dt)
        self.W_k_s = torch.stack([b.attn.c_k.weight for b in blocks]).to(dt)
        self.W_v_s = torch.stack([b.attn.c_v.weight for b in blocks]).to(dt)
        self.W_o_s = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dt)
        self.W_fc_s = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dt)
        self.W_proj_s = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dt)

        # Stacked lambdas (for no_x0_resid: resid=1, x0=0)
        no_x0 = getattr(model.config, 'no_x0_resid', False)
        dev = blocks[0].attn.c_q.weight.device
        if no_x0:
            self.resid_lambdas = torch.ones(n_par, device=dev, dtype=dt)
            self.x0_lambdas = torch.zeros(n_par, device=dev, dtype=dt)
        else:
            self.resid_lambdas = torch.stack([model.resid_lambdas[i] for i in par]).to(dt)
            self.x0_lambdas = torch.stack([model.x0_lambdas[i] for i in par]).to(dt)

        # Fused weights for full fusion (concatenated)
        self.W_q_f = torch.cat([b.attn.c_q.weight for b in blocks], dim=0).to(dt)
        self.W_k_f = torch.cat([b.attn.c_k.weight for b in blocks], dim=0).to(dt)
        self.W_v_f = torch.cat([b.attn.c_v.weight for b in blocks], dim=0).to(dt)
        self.W_o_f = torch.cat([b.attn.c_proj.weight for b in blocks], dim=1).to(dt)
        self.W_fc_f = torch.cat([b.mlp.c_fc.weight for b in blocks], dim=0).to(dt)
        self.W_proj_f = torch.cat([b.mlp.c_proj.weight for b in blocks], dim=1).to(dt)
        if no_x0:
            self.avg_resid = 1.0
            self.avg_x0 = 0.0
        else:
            self.avg_resid = sum(model.resid_lambdas[i].item() for i in par) / n_par
            self.avg_x0 = sum(model.x0_lambdas[i].item() for i in par) / n_par

        # Per-layer VE gate weights (for batched methods)
        self.ve_gate_weights = []
        self.ve_gate_channels = []
        for b in blocks:
            if b.attn.ve_gate is not None:
                self.ve_gate_weights.append(b.attn.ve_gate.weight.to(dt))
                self.ve_gate_channels.append(b.attn.ve_gate_channels)
            else:
                self.ve_gate_weights.append(None)
                self.ve_gate_channels.append(0)

        self.par = par
        self.n_par = n_par
        self.blocks = blocks


class ChunkWeights:
    """Precomputed fused weights for one chunk of layers."""

    def __init__(self, model, chunk_layers):
        dt = COMPUTE_DTYPE
        blocks = [model.transformer.h[i] for i in chunk_layers]
        n = len(chunk_layers)

        self.W_q = torch.cat([b.attn.c_q.weight for b in blocks], dim=0).to(dt)
        self.W_k = torch.cat([b.attn.c_k.weight for b in blocks], dim=0).to(dt)
        self.W_v = torch.cat([b.attn.c_v.weight for b in blocks], dim=0).to(dt)
        self.W_o = torch.cat([b.attn.c_proj.weight for b in blocks], dim=1).to(dt)
        self.W_fc = torch.cat([b.mlp.c_fc.weight for b in blocks], dim=0).to(dt)
        self.W_proj = torch.cat([b.mlp.c_proj.weight for b in blocks], dim=1).to(dt)
        if getattr(model.config, 'no_x0_resid', False):
            self.avg_resid = 1.0
            self.avg_x0 = 0.0
        else:
            self.avg_resid = sum(model.resid_lambdas[i].item() for i in chunk_layers) / n
            self.avg_x0 = sum(model.x0_lambdas[i].item() for i in chunk_layers) / n
        self.layers = chunk_layers
        self.n = n


# ---------------------------------------------------------------------------
# Forward methods
# ---------------------------------------------------------------------------

def _batched_forward(all_h_prev, x0, cos_sin, ve, pw, model):
    """Batched forward via stacked einsum. Returns (n_par, B, T, D)."""
    n_par = pw.n_par
    par = pw.par
    B, T, D = x0.shape
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head

    all_x_in = (pw.resid_lambdas.view(n_par,1,1,1) * all_h_prev
                + pw.x0_lambdas.view(n_par,1,1,1) * x0.unsqueeze(0))
    all_x_normed = F.rms_norm(all_x_in, (D,))

    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_q_s)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_k_s)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_v_s)

    all_q = all_q.view(n_par,B,T,n_head,head_dim).permute(0,1,3,2,4)
    all_k = all_k.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4)
    all_v = all_v.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4)

    # Value embeddings
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_j.view(B,T,n_kv_head,head_dim)
            gate = 3*torch.sigmoid(F.linear(all_x_normed[j,:,:,:12], pw.ve_gate_weights[j]))
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1,2)*ve_r.permute(0,2,1,3)

    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0,1,3,2,4)
    sin_r = sin.unsqueeze(0).permute(0,1,3,2,4)
    def _rope(x):
        d=x.shape[-1]//2; x1,x2=x[...,:d],x[...,d:]
        return torch.cat([x1*cos_r+x2*sin_r, x1*(-sin_r)+x2*cos_r], dim=-1)
    all_q = F.rms_norm(_rope(all_q),(head_dim,))*1.2
    all_k = F.rms_norm(_rope(all_k),(head_dim,))*1.2

    if n_kv_groups > 1:
        all_k = all_k[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,T,head_dim).reshape(n_par,B,n_head,T,head_dim)
        all_v = all_v[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,T,head_dim).reshape(n_par,B,n_head,T,head_dim)

    q_s = all_q.reshape(n_par*B,n_head,T,head_dim)
    k_s = all_k.reshape(n_par*B,n_head,T,head_dim)
    v_s = all_v.reshape(n_par*B,n_head,T,head_dim)
    y_s = F.scaled_dot_product_attention(q_s,k_s,v_s,is_causal=True)

    attn_out = y_s.reshape(n_par,B,n_head,T,head_dim).permute(0,1,3,2,4).reshape(n_par,B,T,D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o_s)
    all_hidden = all_x_in + attn_out

    h_n = F.rms_norm(all_hidden, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    return all_hidden + mlp


def _fused_forward_chunk(h_in, x0, cos_sin, cw, model, avg=False):
    """Run one fused chunk. Returns single (B, T, D) output for the chunk.

    If avg=True, divide O projection and MLP output by chunk size (cw.n)
    to average rather than sum the cross-layer contributions.
    """
    B, T, D = h_in.shape
    n_head = model.config.n_head
    total_heads = cw.n * n_head
    head_dim = D // n_head

    x_in = cw.avg_resid * h_in + cw.avg_x0 * x0
    x_normed = norm(x_in)

    q = F.linear(x_normed, cw.W_q).view(B,T,total_heads,head_dim)
    k = F.linear(x_normed, cw.W_k).view(B,T,total_heads,head_dim)
    v = F.linear(x_normed, cw.W_v).view(B,T,total_heads,head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin); k = apply_rotary_emb(k, cos, sin)
    q, k = norm(q)*1.2, norm(k)*1.2

    q_t = q.transpose(1,2); k_t = k.transpose(1,2); v_t = v.transpose(1,2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    y = y.transpose(1,2).contiguous().view(B,T,-1)

    attn_out = F.linear(y, cw.W_o)
    if avg and cw.n > 1:
        attn_out = attn_out / cw.n
    x_mid = x_in + attn_out
    x_mid_n = norm(x_mid)
    hf = F.linear(x_mid_n, cw.W_fc)
    mlp = F.linear(F.relu(hf).square(), cw.W_proj)
    if avg and cw.n > 1:
        mlp = mlp / cw.n
    return x_mid + mlp


def _batched_chunks_forward(all_h_in, x0, cos_sin, chunk_weights_list, model, avg=False):
    """Batched forward for C equal-size chunks simultaneously.

    Instead of looping over chunks, stacks all chunk weights and inputs
    into (C, ...) tensors and processes in one batched call.
    All chunks must have the same size (number of layers).

    Args:
        all_h_in: (C, B, T, D) - per-chunk inputs
        chunk_weights_list: list of C ChunkWeights (all same size)
        avg: if True, divide O proj and MLP by chunk size
    Returns:
        all_out: (C, B, T, D) - per-chunk outputs
    """
    C = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n
    assert all(cw.n == chunk_size for cw in chunk_weights_list), "All chunks must have same size"

    B, T, D = x0.shape
    n_head = model.config.n_head
    heads_per_chunk = chunk_size * n_head
    head_dim = D // n_head
    dt = all_h_in.dtype

    # Stack chunk weights: (C, out_dim, D) for QKV, etc.
    W_q = torch.stack([cw.W_q for cw in chunk_weights_list])  # (C, chunk_size*n_head*hd, D)
    W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
    W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
    W_o = torch.stack([cw.W_o for cw in chunk_weights_list])  # (C, D, chunk_size*n_head*hd)
    W_fc = torch.stack([cw.W_fc for cw in chunk_weights_list])
    W_proj = torch.stack([cw.W_proj for cw in chunk_weights_list])

    avg_resids = torch.tensor([cw.avg_resid for cw in chunk_weights_list], device=x0.device, dtype=dt).view(C,1,1,1)
    avg_x0s = torch.tensor([cw.avg_x0 for cw in chunk_weights_list], device=x0.device, dtype=dt).view(C,1,1,1)

    # Per-chunk input blending
    all_x_in = avg_resids * all_h_in + avg_x0s * x0.unsqueeze(0)  # (C, B, T, D)
    all_x_normed = F.rms_norm(all_x_in, (D,))

    # QKV via einsum: (C, B, T, D) × (C, out, D) → (C, B, T, out)
    all_q = torch.einsum('cbtd,chd->cbth', all_x_normed, W_q)
    all_k = torch.einsum('cbtd,chd->cbth', all_x_normed, W_k)
    all_v = torch.einsum('cbtd,chd->cbth', all_x_normed, W_v)

    # Reshape to (C, B, heads_per_chunk, T, hd) for attention
    all_q = all_q.view(C, B, T, heads_per_chunk, head_dim).permute(0,1,3,2,4)
    all_k = all_k.view(C, B, T, heads_per_chunk, head_dim).permute(0,1,3,2,4)
    all_v = all_v.view(C, B, T, heads_per_chunk, head_dim).permute(0,1,3,2,4)

    # RoPE
    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0,1,3,2,4)  # (1, 1, 1, T, hd//2)
    sin_r = sin.unsqueeze(0).permute(0,1,3,2,4)
    def _rope(x):
        d = x.shape[-1]//2; x1, x2 = x[...,:d], x[...,d:]
        return torch.cat([x1*cos_r + x2*sin_r, x1*(-sin_r) + x2*cos_r], dim=-1)
    all_q = F.rms_norm(_rope(all_q), (head_dim,)) * 1.2
    all_k = F.rms_norm(_rope(all_k), (head_dim,)) * 1.2

    # SDPA: (C*B, heads_per_chunk, T, hd)
    q_s = all_q.reshape(C*B, heads_per_chunk, T, head_dim)
    k_s = all_k.reshape(C*B, heads_per_chunk, T, head_dim)
    v_s = all_v.reshape(C*B, heads_per_chunk, T, head_dim)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

    # Reshape back: (C*B, H, T, hd) → (C, B, T, D_chunk)
    y = y_s.reshape(C, B, heads_per_chunk, T, head_dim).permute(0,1,3,2,4)
    y = y.reshape(C, B, T, heads_per_chunk * head_dim)

    # O projection: (C, B, T, chunk_size*n_head*hd) × (C, D, chunk_size*n_head*hd) → (C, B, T, D)
    attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
    if avg and chunk_size > 1:
        attn_out = attn_out / chunk_size

    # First residual
    all_hidden = all_x_in + attn_out

    # MLP
    h_n = F.rms_norm(all_hidden, (D,))
    hf = torch.einsum('cbtd,cid->cbti', h_n, W_fc)
    mlp = torch.einsum('cbti,cdi->cbtd', F.relu(hf).square(), W_proj)
    if avg and chunk_size > 1:
        mlp = mlp / chunk_size

    return all_hidden + mlp


def _fused_split_forward(h_in, x0, cos_sin, ve, pw, model, avg=False):
    """Fused QKV+SDPA, split per-layer O+MLP. Returns (n_par, B, T, D).
    Note: avg not applicable here since O+MLP are already split per-layer."""
    B, T, D = h_in.shape
    n_par = pw.n_par
    n_head = model.config.n_head
    head_dim = D // n_head
    total_heads = n_par * n_head

    x_in = pw.avg_resid * h_in + pw.avg_x0 * x0
    x_normed = norm(x_in)

    q = F.linear(x_normed, pw.W_q_f).view(B,T,total_heads,head_dim)
    k = F.linear(x_normed, pw.W_k_f).view(B,T,total_heads,head_dim)
    v = F.linear(x_normed, pw.W_v_f).view(B,T,total_heads,head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin); k = apply_rotary_emb(k, cos, sin)
    q, k = norm(q)*1.2, norm(k)*1.2

    q_t = q.transpose(1,2); k_t = k.transpose(1,2); v_t = v.transpose(1,2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    # Split per-layer: (B, n_par*H, T, hd) → (n_par, B, H, T, hd)
    y_split = y.view(B, n_par, n_head, T, head_dim).permute(1,0,3,2,4)
    y_split = y_split.reshape(n_par, B, T, n_head*head_dim)

    # Per-layer O + MLP
    attn_out = torch.einsum('lbtd,lod->lbto', y_split, pw.W_o_s)
    x_in_exp = x_in.unsqueeze(0).expand(n_par,-1,-1,-1)
    all_hidden = x_in_exp + attn_out
    h_n = F.rms_norm(all_hidden, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    return all_hidden + mlp


def _idn_correction(all_out, hs, h_init, n_par, elk_k=0.0):
    """IDN prefix-sum correction. Updates hs in-place.

    all_out[j] = f_j(h_prev_j) — output of layer j given its input
    hs[0] = h_init (prefix output, fixed)
    hs[j+1] = estimate of output of layer j (= input to layer j+1)

    The correction: h_corr[j] = f_j + a * (h_corr[j-1] - h_old[j-1])
    where h_old[j] is the pre-correction estimate.
    """
    a = 1.0 - elk_k
    # Save pre-correction values
    h_old = [hs[j+1].clone() for j in range(n_par)]

    # Prefix-sum correction
    h_corr = all_out[0]
    hs[1] = h_corr
    for j in range(1, n_par):
        h_corr = all_out[j] + a * (h_corr - h_old[j-1])
        hs[j+1] = h_corr


# ---------------------------------------------------------------------------
# IDN diagnostics: per-layer error and spectral radius
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_per_layer_idn_error(model, idx, seq_layers, pw):
    """Compute per-layer hidden state error between sequential and IDN K=1.

    Returns dict with:
      'h_seq': list of sequential hidden states per parallel layer
      'h_idn': list of IDN-corrected hidden states per parallel layer
      'rel_err': list of relative errors ||h_idn - h_seq|| / ||h_seq|| per layer
      'abs_err': list of absolute errors ||h_idn - h_seq|| per layer
    """
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    x0, cos_sin, ve = _embed(model, idx)

    # Run sequential to get ground truth per-layer outputs
    h_seq = [None] * L; x = x0
    for i in range(L):
        x = _run_block(model, i, x, x0, cos_sin, ve); h_seq[i] = x

    # Run IDN K=1
    h_init = h_seq[seq_layers - 1] if seq_layers > 0 else x0
    all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
    all_out = _batched_forward(all_h0, x0, cos_sin, ve, pw, model)

    # IDN prefix-sum correction
    hs = [h_init] + [h_init.clone() for _ in range(n_par)]
    _idn_correction(list(all_out), hs, h_init, n_par, elk_k=0.0)

    # Per-layer errors
    rel_errs = []
    abs_errs = []
    for j, li in enumerate(par):
        diff = (hs[j+1].float() - h_seq[li].float()).norm(dim=-1)  # (B, T)
        ref = h_seq[li].float().norm(dim=-1).clamp(min=1e-6)
        rel_errs.append((diff / ref).mean().item())
        abs_errs.append(diff.mean().item())

    return {'rel_err': rel_errs, 'abs_err': abs_errs, 'layers': par}


@torch.inference_mode()
def estimate_spectral_radius(model, idx, seq_layers, n_iters=10):
    """Estimate spectral radius σ_max(∂f_i/∂h) per layer via power iteration.

    Uses finite-difference approximation:
      σ_max ≈ ||f(h + εv) - f(h)|| / (ε||v||)
    where v is updated each iteration to approximate the top singular vector.

    Returns list of σ_max estimates for each layer in range(seq_layers, L).
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)
    B, T, D = x0.shape

    # Run sequential to get per-layer inputs
    h = [None] * L; x = x0
    for i in range(L):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x

    eps = 1e-1  # must be large enough to survive bf16 quantization
    spectral_radii = []

    for li in range(seq_layers, L):
        h_in = h[li - 1] if li > 0 else x0
        f_h = h[li]

        # Power iteration with proper per-token normalization
        v = torch.randn_like(h_in)
        v = v / v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)

        for it in range(n_iters):
            # Perturb in float32, cast to model dtype for forward
            h_pert = h_in.float() + eps * v
            f_pert = _run_block(model, li, h_pert.to(h_in.dtype), x0, cos_sin, ve)
            Jv = (f_pert.float() - f_h.float()) / eps
            # Per-token sigma: ||Jv|| (v is unit norm)
            sigma_per_token = Jv.norm(dim=-1)  # (B, T)
            # Normalize v for next iteration
            v = Jv / sigma_per_token.unsqueeze(-1).clamp(min=1e-12)

        spectral_radii.append(sigma_per_token.mean().item())

    return spectral_radii


# ---------------------------------------------------------------------------
# High-level forward functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, idx):
    return model.forward(idx)


@torch.inference_mode()
def forward_idn_batched(model, idx, seq_layers, pw, K=1, elk_k=0.0, init_h=None,
                         init='h0', preheat_cache=None):
    """IDN batched with K iterations and init support.

    init_h: pre-computed init (legacy, used when init='h0' with external init).
    init: 'h0', 'batch_fwd', 'preheat'. Computed internally to avoid
          redundant _embed/prefix computation. 'warm' uses init_h from caller.
    """
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    x0, cos_sin, ve = _embed(model, idx)

    h = [None]*L; x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
    h_init = x

    # Initialize — computed here to reuse x0/cos_sin/ve/h_init
    if init_h is not None:
        hs = [h_init] + [init_h[par[j]] for j in range(n_par)]
    elif init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward(all_h0, x0, cos_sin, ve, pw, model)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    elif init == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        hs = [h_init] + [h_pred[par[j]] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _k in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward(all_hp, x0, cos_sin, ve, pw, model)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    for j, li in enumerate(par): h[li] = hs[j+1]
    return _logits(model, h)


@torch.inference_mode()
def forward_idn_fused(model, idx, seq_layers, pw, chunk_weights, K=1, elk_k=0.0, init_h=None, avg=False):
    """Fully fused (one mega-block for all parallel layers, no per-layer correction).

    This is a pure approximation: concatenated weights, averaged lambdas,
    mixed O/MLP projections. No IDN correction. K>1 re-runs on own output.
    Init strategy is ignored (fused always uses h_init with averaged lambdas).
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)

    h = [None]*L; x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
    h_init = x

    fused_out = _fused_forward_chunk(h_init, x0, cos_sin, chunk_weights[0], model, avg=avg)
    for _k in range(1, K):
        fused_out = _fused_forward_chunk(fused_out, x0, cos_sin, chunk_weights[0], model, avg=avg)

    # Fill h for _logits (needs h[-1] and h[backout])
    for i in range(seq_layers, L):
        h[i] = h_init
    h[L-1] = fused_out
    return _logits(model, h)


@torch.inference_mode()
def forward_idn_fused_split(model, idx, seq_layers, pw, K=1, elk_k=0.0, init_h=None, avg=False,
                         init='h0', preheat_cache=None):
    """Fused QKV+SDPA, split per-layer O+MLP + IDN correction."""
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    x0, cos_sin, ve = _embed(model, idx)

    h = [None]*L; x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
    h_init = x

    if init_h is not None:
        hs = [h_init] + [init_h[par[j]] for j in range(n_par)]
    elif init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward(all_h0, x0, cos_sin, ve, pw, model)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    elif init == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        hs = [h_init] + [h_pred[par[j]] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _k in range(K):
        all_out = _fused_split_forward(h_init, x0, cos_sin, ve, pw, model)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    for j, li in enumerate(par): h[li] = hs[j+1]
    return _logits(model, h)


@torch.inference_mode()
def forward_chunkwise(model, idx, seq_layers, pw, chunk_weights_list, K=1, elk_k=0.0, init_h=None, avg=False,
                       init='h0', preheat_cache=None):
    """Chunkwise fused: fuse within chunks, IDN correct between chunks."""
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    n_chunks = len(chunk_weights_list)
    x0, cos_sin, ve = _embed(model, idx)

    h = [None]*L; x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
    h_init = x

    if init_h is not None:
        chunk_estimates = [init_h[cw.layers[-1]] for cw in chunk_weights_list]
    elif init == 'batch_fwd':
        # Run one fused forward per chunk on h_init to get init estimates
        chunk_estimates = [_fused_forward_chunk(h_init, x0, cos_sin, cw, model, avg=avg) for cw in chunk_weights_list]
    elif init == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        chunk_estimates = [h_pred[cw.layers[-1]] for cw in chunk_weights_list]
    else:
        chunk_estimates = [h_init.clone() for _ in range(n_chunks)]

    hs = [h_init] + list(chunk_estimates)

    for _k in range(K):
        chunk_outs = []
        for c, cw in enumerate(chunk_weights_list):
            out = _fused_forward_chunk(hs[c], x0, cos_sin, cw, model, avg=avg)
            chunk_outs.append(out)
        _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

    corrected = [hs[c+1] for c in range(n_chunks)]
    _fill_h_from_chunks(h, corrected, chunk_weights_list, seq_layers, h_init, L)
    return _logits(model, h)


def _fill_h_from_chunks(h, corrected, chunk_weights_list, seq_layers, h_init, L):
    """Fill h list for _logits from chunk corrected outputs."""
    for i in range(seq_layers, L):
        h[i] = h_init
    h[L-1] = corrected[-1]
    backout = L // 2
    if backout >= seq_layers:
        for c, cw in enumerate(chunk_weights_list):
            if backout <= cw.layers[-1]:
                h[backout] = corrected[c]
                break


@torch.inference_mode()
def forward_idn_chunkwise(model, idx, seq_layers, pw, chunk_weights_list, K=1, elk_k=0.0, init_h=None, avg=False,
                               init='h0', preheat_cache=None):
    """Chunkwise with batched chunk evaluation. All chunks must have same size."""
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    n_chunks = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n
    assert all(cw.n == chunk_size for cw in chunk_weights_list), \
        f"All chunks must have same size for batched version, got {[cw.n for cw in chunk_weights_list]}"

    x0, cos_sin, ve = _embed(model, idx)
    h = [None]*L; x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
    h_init = x

    if init_h is not None:
        chunk_estimates = [init_h[cw.layers[-1]] for cw in chunk_weights_list]
    elif init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_chunks, -1, -1, -1)
        all_init = _batched_chunks_forward(all_h0, x0, cos_sin, chunk_weights_list, model, avg=avg)
        chunk_estimates = [all_init[c] for c in range(n_chunks)]
    elif init == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        chunk_estimates = [h_pred[cw.layers[-1]] for cw in chunk_weights_list]
    else:
        chunk_estimates = [h_init.clone() for _ in range(n_chunks)]

    hs = [h_init] + list(chunk_estimates)

    for _k in range(K):
        # Stack chunk inputs: (C, B, T, D)
        all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
        # Batched forward for all chunks simultaneously
        all_out = _batched_chunks_forward(all_h_in, x0, cos_sin, chunk_weights_list, model, avg=avg)
        # Unstack to list
        chunk_outs = [all_out[c] for c in range(n_chunks)]
        # Inter-chunk IDN correction
        _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

    corrected = [hs[c+1] for c in range(n_chunks)]
    _fill_h_from_chunks(h, corrected, chunk_weights_list, seq_layers, h_init, L)
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Init strategies
# ---------------------------------------------------------------------------

@torch.inference_mode()
def get_init_h(model, idx, par, strategy='h0', preheat_cache=None, pw=None):
    """Compute init hidden states for parallel layers."""
    L = model.config.n_layer
    if strategy == 'h0':
        return None  # methods handle h0 internally

    x0, cos_sin, ve = _embed(model, idx)
    B, T, D = x0.shape

    if strategy == 'warm':
        # Run sequential on full sequence, use position 0's h for all positions
        h_seq = [None]*L; x = x0
        for i in range(L):
            x = _run_block(model, i, x, x0, cos_sin, ve); h_seq[i] = x
        init_h = {}
        for li in par:
            init_h[li] = h_seq[li][:, 0:1, :].expand(B, T, D).clone()
        return init_h

    elif strategy == 'batch_fwd':
        # Evaluate all parallel layers on h_init via batched forward (not sequential loop)
        assert pw is not None, "batch_fwd init requires PrecomputedWeights"
        h = [None]*L; x = x0
        seq_layers = par[0]
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve); h[i] = x
        h_init = x
        n_par = len(par)
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward(all_h0, x0, cos_sin, ve, pw, model)
        init_h = {par[j]: all_out[j] for j in range(n_par)}
        return init_h

    elif strategy == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        init_h = {li: h_pred[li] for li in par}
        return init_h

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_data(device, max_tokens=1_000_000, seq_len=128):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR')
    import pyarrow.parquet as pq
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    table = pq.read_table(os.path.join(base_dir, 'base_data_climbmix', 'shard_06542.parquet'), columns=['text'])
    all_tokens = []
    need_tokens = max_tokens + seq_len + 1
    for text in table.column('text').to_pylist():
        all_tokens.extend(tokenizer.encode(text, prepend=bos))
        if len(all_tokens) >= need_tokens: break
    batches = []
    offset = 0
    while offset + seq_len + 1 <= len(all_tokens) and len(batches) * seq_len < max_tokens:
        batches.append(torch.tensor([all_tokens[offset:offset + seq_len + 1]], dtype=torch.long, device=device))
        offset += seq_len
    return batches


def bench(fn, device, n_warm=50, n_run=150):
    for _ in range(n_warm): fn()
    if device.type == 'cuda': torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if device.type == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times)/len(times)*1000


# ---------------------------------------------------------------------------
# AR generation
# ---------------------------------------------------------------------------

AR_PROMPTS = [
    "The quick brown fox jumps over",
    "Machine learning is a subset of",
    "Once upon a time there lived a",
    "The capital of France is Paris",
    "In a galaxy far far away",
    "To be or not to be that",
    "The meaning of life is to",
    "Artificial intelligence will change",
]


def _fused_forward_chunk_kv(h_in, x0, cos_sin, cw, model, cached_k=None, cached_v=None):
    """Fused chunk forward with KV cache for AR generation.

    Like _fused_forward_chunk but uses explicit matmul attention with
    concatenated cached K,V. The chunk has chunk_size * n_head attention heads,
    so its KV cache has chunk_size * n_kv_head KV heads.

    Returns (output, new_k, new_v) where new_k/v are post-RoPE, post-norm
    for cache insertion.
    """
    B, T, D = h_in.shape
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    total_q_heads = cw.n * n_head
    total_kv_heads = cw.n * n_kv_head
    head_dim = D // n_head
    n_kv_groups = n_head // n_kv_head

    x_in = cw.avg_resid * h_in + cw.avg_x0 * x0
    x_normed = norm(x_in)

    q = F.linear(x_normed, cw.W_q).view(B, T, total_q_heads, head_dim)
    k = F.linear(x_normed, cw.W_k).view(B, T, total_kv_heads, head_dim)
    v = F.linear(x_normed, cw.W_v).view(B, T, total_kv_heads, head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q = norm(q) * 1.2
    k = norm(k) * 1.2

    # Save new K,V for cache (post-RoPE, post-norm)
    # Shape: (B, T, total_kv_heads, head_dim)
    new_k = k.clone()
    new_v = v.clone()

    # Transpose to (B, H, T, D) for matmul attention
    q = q.transpose(1, 2)  # (B, total_q_heads, T, head_dim)
    k = k.transpose(1, 2)  # (B, total_kv_heads, T, head_dim)
    v = v.transpose(1, 2)

    # Concat cached KV
    if cached_k is not None and cached_k.shape[2] > 0:
        # cached_k: (B, total_kv_heads, S, head_dim)
        k = torch.cat([cached_k, k], dim=2)
        v = torch.cat([cached_v, v], dim=2)

    S_plus_T = k.shape[2]

    # GQA expand
    if n_kv_groups > 1:
        # Expand each KV head group: (B, total_kv_heads, S+T, D) -> (B, total_q_heads, S+T, D)
        k = k.view(B, total_kv_heads, 1, S_plus_T, head_dim).expand(
            B, total_kv_heads, n_kv_groups, S_plus_T, head_dim).reshape(
            B, total_q_heads, S_plus_T, head_dim)
        v = v.view(B, total_kv_heads, 1, S_plus_T, head_dim).expand(
            B, total_kv_heads, n_kv_groups, S_plus_T, head_dim).reshape(
            B, total_q_heads, S_plus_T, head_dim)

    # Explicit matmul attention
    scaling = head_dim ** -0.5
    attn_w = torch.matmul(q, k.transpose(-2, -1)) * scaling  # (B, H, T, S+T)
    if T > 1:
        ri = torch.arange(T, device=h_in.device).unsqueeze(1)
        ci = torch.arange(S_plus_T, device=h_in.device).unsqueeze(0)
        S = S_plus_T - T
        causal = ci <= (ri + S)
        attn_w = attn_w.masked_fill(~causal[None, None], float('-inf'))
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(h_in.dtype)
    attn_out = torch.matmul(attn_w, v)  # (B, total_q_heads, T, head_dim)

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
    attn_out = F.linear(attn_out, cw.W_o)
    x_mid = x_in + attn_out
    x_mid_n = norm(x_mid)
    hf = F.linear(x_mid_n, cw.W_fc)
    mlp = F.linear(F.relu(hf).square(), cw.W_proj)
    return x_mid + mlp, new_k, new_v


def _update_chunk_kv_cache(chunk_kv_caches, chunk_weights_list, corrected_inputs,
                            x0, cos_sin, model, pos):
    """Recompute K,V from corrected chunk inputs and insert into chunk KV caches.

    For chunk c, recompute K,V from corrected_inputs[c] using chunk's fused weights.
    Only K,V projections + RoPE + norm, no full forward.
    """
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head

    cos, sin = cos_sin
    for c, cw in enumerate(chunk_weights_list):
        total_kv_heads = cw.n * n_kv_head
        h_in = corrected_inputs[c]
        x_in = cw.avg_resid * h_in + cw.avg_x0 * x0
        x_normed = norm(x_in)

        k = F.linear(x_normed, cw.W_k).view(1, 1, total_kv_heads, head_dim)
        v = F.linear(x_normed, cw.W_v).view(1, 1, total_kv_heads, head_dim)
        k = apply_rotary_emb(k, cos, sin)
        k = norm(k) * 1.2

        # Insert into chunk KV cache: (B, S_max, total_kv_heads, head_dim)
        chunk_kv_caches[c][0][:, pos:pos+1, :, :] = k  # k_cache
        chunk_kv_caches[c][1][:, pos:pos+1, :, :] = v  # v_cache


@torch.inference_mode()
def ar_generate_chunkwise(model, prompt_ids, pw, seq_layers, chunk_weights_list,
                           max_new=16, K=1, elk_k=0.0, init='warm'):
    """AR generation with chunkwise fused layers + KV cache.

    Each chunk is treated as a "wide layer" with its own KV cache.
    Prefill: run chunks sequentially to fill per-chunk KV caches.
    Decode: run chunks with cached KV, inter-chunk IDN correction,
            then update KV caches from corrected inputs.
    """
    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    n_chunks = len(chunk_weights_list)
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // model.config.n_head
    max_seq = len(prompt_ids) + max_new + 1

    # Standard KV cache for sequential prefix layers
    kv = KVCache(1, n_kv_head, max_seq, head_dim, L, device, dtype)
    ids = torch.tensor([prompt_ids], device=device)

    # === Prefill ===
    # 1. Sequential prefix (layers 0..seq_layers-1) with KV cache
    logits = model.forward(ids, kv_cache=kv)

    # 2. Run chunks sequentially on the prompt to fill chunk KV caches.
    #    We re-run the model to get the hidden states at each chunk boundary.
    x0_pf, cs_pf, ve_pf = _embed(model, ids)
    prev_chunk_h = [None] * n_chunks  # warm-start for decode

    x = x0_pf
    for i in range(seq_layers):
        x_in = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0_pf
        x = model.transformer.h[i](x_in, ve_pf.get(str(i)), cs_pf, model.window_sizes[i], None)

    # x is now h_init for the parallel section
    # Run each chunk sequentially (for prefill KV cache filling)
    chunk_kv_caches = []
    h_chunk_in = x
    for c, cw in enumerate(chunk_weights_list):
        total_kv_heads = cw.n * n_kv_head
        # Allocate KV cache for this chunk: (B, S_max, total_kv_heads, head_dim)
        ck = torch.zeros(1, max_seq, total_kv_heads, head_dim, device=device, dtype=dtype)
        cv = torch.zeros(1, max_seq, total_kv_heads, head_dim, device=device, dtype=dtype)

        # Run prefill through this chunk (with explicit matmul attention, no cache yet)
        # For prefill we run the chunk on the full prompt; the KV from this run IS the cache
        chunk_out, new_k, new_v = _fused_forward_chunk_kv(
            h_chunk_in, x0_pf, cs_pf, cw, model)

        # new_k, new_v: (B, T_prompt, total_kv_heads, head_dim)
        T_prompt = new_k.shape[1]
        ck[:, :T_prompt, :, :] = new_k.transpose(1, 2).transpose(1, 2)  # already (B,T,H,D)
        cv[:, :T_prompt, :, :] = new_v

        chunk_kv_caches.append((ck, cv))
        prev_chunk_h[c] = chunk_out[:, -1:, :].clone()
        h_chunk_in = chunk_out  # next chunk's input = this chunk's output

    # Track position for chunk caches (same as prefix KV cache)
    chunk_cache_pos = len(prompt_ids)  # prefill filled positions 0..len-1

    # === Decode ===
    gen = []
    for step in range(max_new):
        tok = logits[0, -1].argmax().item()
        gen.append(tok)
        ids_step = torch.tensor([[tok]], device=device)

        # Embed
        x0 = model.transformer.wte(ids_step).to(dtype)
        x0 = norm(x0)
        if kv.prev_embedding is not None:
            gate = model.smear_lambda.to(dtype) * torch.sigmoid(model.smear_gate(x0[:, :, :24]))
            x0 = x0 + gate * kv.prev_embedding

        pos = kv.get_pos()
        cos_sin = model.cos[:, pos:pos+1], model.sin[:, pos:pos+1]
        if getattr(model.config, 'no_ve', False):
            ve = {}
        else:
            ve = {k_: model.value_embeds[k_](ids_step).to(dtype) for k_ in model.value_embeds}

        # Sequential prefix with KV cache
        x = x0
        h_bo_prefix = None
        bo = L // 2
        for i in range(seq_layers):
            x_in = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            x = model.transformer.h[i](x_in, ve.get(str(i)), cos_sin, model.window_sizes[i], kv)
            if i == bo:
                h_bo_prefix = x
        h_init = x

        # Initialize chunk estimates
        if init == 'warm' and prev_chunk_h[0] is not None:
            hs = [h_init] + [prev_chunk_h[c].clone() for c in range(n_chunks)]
        else:
            # h0: all chunks init to h_init
            hs = [h_init] + [h_init.clone() for _ in range(n_chunks)]

        # Extract cached K,V for each chunk
        cached_kvs = []
        for c in range(n_chunks):
            ck, cv = chunk_kv_caches[c]
            # (B, S, total_kv_heads, head_dim) -> (B, total_kv_heads, S, head_dim)
            cached_kvs.append((
                ck[:, :pos, :, :].transpose(1, 2),
                cv[:, :pos, :, :].transpose(1, 2),
            ))

        # Newton iterations
        for _k in range(K):
            chunk_outs = []
            for c, cw in enumerate(chunk_weights_list):
                out, _, _ = _fused_forward_chunk_kv(
                    hs[c], x0, cos_sin, cw, model,
                    cached_k=cached_kvs[c][0], cached_v=cached_kvs[c][1])
                chunk_outs.append(out)
            _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

        # Update chunk KV caches from corrected inputs (hs[c] = input to chunk c)
        _update_chunk_kv_cache(chunk_kv_caches, chunk_weights_list,
                                [hs[c] for c in range(n_chunks)],
                                x0, cos_sin, model, pos)

        # Advance prefix KV cache + smear
        kv.cache_seqlens += 1
        kv.prev_embedding = norm(model.transformer.wte(ids_step).to(dtype))

        # Save warm-start
        if init == 'warm':
            for c in range(n_chunks):
                prev_chunk_h[c] = hs[c + 1].clone()

        # Logits: need h[L-1] and h[backout]
        x_final = hs[n_chunks]  # last chunk's corrected output
        h_bo = h_bo_prefix
        if bo >= seq_layers:
            for c, cw in enumerate(chunk_weights_list):
                if bo <= cw.layers[-1]:
                    h_bo = hs[c + 1]
                    break
        if h_bo is not None:
            x_final = x_final - model.backout_lambda.to(dtype) * h_bo
        x_final = norm(x_final)
        logits = model.lm_head(x_final)[..., :model.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)

    return gen


@torch.inference_mode()
def ar_generate_sequential(model, prompt_ids, max_new=16):
    """Sequential AR generation (ground truth)."""
    L = model.config.n_layer
    kv = KVCache(1, model.config.n_kv_head, len(prompt_ids)+max_new+1,
                 model.config.n_embd//model.config.n_head, L, device, dtype)
    ids = torch.tensor([prompt_ids], device=device)
    logits = model.forward(ids, kv_cache=kv)
    gen = []
    for _ in range(max_new):
        tok = logits[0,-1].argmax().item(); gen.append(tok)
        logits = model.forward(torch.tensor([[tok]], device=device), kv_cache=kv)
    return gen


def _update_kv_cache(model, kv, par, h_step, h_init, x0, cos_sin, ve, pw):
    """Batched KV cache update: compute K,V for all parallel layers at once.

    Like Qwen demo's _batched_cache_update: batched RMSNorm + K/V projection
    via einsum, then per-layer cache insert.
    """
    n_par = len(par)
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // model.config.n_head
    D = model.config.n_embd
    pos = kv.get_pos()

    # Build batched inputs: (n_par, B, T, D)
    inputs = [h_init if j == 0 else h_step[par[j] - 1] for j in range(n_par)]
    all_corr_in = torch.stack(inputs)  # (n_par, B, T, D)
    all_x_in = (pw.resid_lambdas.view(n_par,1,1,1) * all_corr_in
                + pw.x0_lambdas.view(n_par,1,1,1) * x0.unsqueeze(0))
    all_x_n = F.rms_norm(all_x_in, (D,))

    # Batched K,V projections via einsum
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_n, pw.W_k_s)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_n, pw.W_v_s)
    all_k = all_k.view(n_par, 1, 1, n_kv_head, head_dim)
    all_v = all_v.view(n_par, 1, 1, n_kv_head, head_dim)

    # VE gate per layer
    for j, li in enumerate(par):
        ve_i = ve.get(str(li))
        if ve_i is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_i.view(1, 1, n_kv_head, head_dim)
            gate = 3 * torch.sigmoid(F.linear(all_x_n[j, :, :, :12], pw.ve_gate_weights[j]))
            all_v[j] = all_v[j] + gate.unsqueeze(-1) * ve_r

    # Batched RoPE on K
    cos_k, sin_k = cos_sin  # each (B, T, n_kv_head, head_dim) or similar
    all_k_r = all_k.view(n_par, 1, 1, n_kv_head, head_dim)
    all_k_r = apply_rotary_emb(all_k_r.view(n_par, 1, n_kv_head, head_dim),
                                cos_k.expand(n_par, -1, -1, -1),
                                sin_k.expand(n_par, -1, -1, -1))
    all_k = F.rms_norm(all_k_r.view(n_par, 1, 1, n_kv_head, head_dim),
                        (head_dim,)) * 1.2

    # Insert into cache
    for j, li in enumerate(par):
        kv.k_cache[li, :, pos:pos+1, :, :] = all_k[j]
        kv.v_cache[li, :, pos:pos+1, :, :] = all_v[j]


def _batched_block_forward_kv(all_h_prev, x0, cos_sin, ve, par, model,
                               cached_k=None, cached_v=None, pw=None):
    """Batched forward with KV cache (for AR generation).
    Extracts cached K,V, concatenates with current token's K,V, runs attention.
    Returns (output, new_k, new_v) for cache update.

    If pw (PrecomputedWeights) is provided, uses precomputed stacked weights
    instead of re-stacking from model each call.

    WARNING: Known bug — produces K,V values that differ significantly from
    sequential Block.forward with KV cache (max error ~2000 in bf16). This
    causes AR generation to diverge from sequential even at high K. Use
    ar_generate_no_kv (in eval_jacobi_idn_full.py) for correct AR eval.
    """
    n_par = len(par)
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head
    B, T, D = x0.shape
    cos, sin = cos_sin
    dt = all_h_prev.dtype

    if pw is not None:
        W_q, W_k, W_v = pw.W_q_s, pw.W_k_s, pw.W_v_s
        W_o, W_fc, W_proj = pw.W_o_s, pw.W_fc_s, pw.W_proj_s
        rl = pw.resid_lambdas.view(n_par,1,1,1)
        xl = pw.x0_lambdas.view(n_par,1,1,1)
    else:
        blocks = [model.transformer.h[i] for i in par]
        W_q = torch.stack([b.attn.c_q.weight for b in blocks]).to(dt)
        W_k = torch.stack([b.attn.c_k.weight for b in blocks]).to(dt)
        W_v = torch.stack([b.attn.c_v.weight for b in blocks]).to(dt)
        W_o = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dt)
        W_fc = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dt)
        W_proj = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dt)
        rl = torch.stack([model.resid_lambdas[i] for i in par]).to(dt).view(n_par,1,1,1)
        xl = torch.stack([model.x0_lambdas[i] for i in par]).to(dt).view(n_par,1,1,1)

    all_x_in = rl * all_h_prev + xl * x0.unsqueeze(0)
    all_x_normed = F.rms_norm(all_x_in, (D,))

    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_q)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_k)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_v)

    all_q = all_q.view(n_par,B,T,n_head,head_dim).permute(0,1,3,2,4)
    all_k = all_k.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4)
    all_v = all_v.view(n_par,B,T,n_kv_head,head_dim).permute(0,1,3,2,4)

    # Value embeddings
    # Value embeddings
    v_list = list(all_v.unbind(0))
    if pw is not None:
        for j, li in enumerate(par):
            ve_j = ve.get(str(li))
            if ve_j is not None and pw.ve_gate_weights[j] is not None:
                ve_r = ve_j.view(B,T,n_kv_head,head_dim)
                gate = 3*torch.sigmoid(F.linear(all_x_normed[j,:,:,:12], pw.ve_gate_weights[j]))
                v_list[j] = v_list[j] + gate.unsqueeze(-1).transpose(1,2)*ve_r.permute(0,2,1,3)
    else:
        for j, li in enumerate(par):
            ve_j = ve.get(str(li))
            if ve_j is not None and blocks[j].attn.ve_gate is not None:
                ve_r = ve_j.view(B,T,n_kv_head,head_dim)
                gate = 3*torch.sigmoid(F.linear(all_x_normed[j,:,:,:12], blocks[j].attn.ve_gate.weight.to(dt)))
                v_list[j] = v_list[j] + gate.unsqueeze(-1).transpose(1,2)*ve_r.permute(0,2,1,3)
    all_v = torch.stack(v_list)

    cos_r = cos.unsqueeze(0).permute(0,1,3,2,4)
    sin_r = sin.unsqueeze(0).permute(0,1,3,2,4)
    def _rope(x):
        d=x.shape[-1]//2; x1,x2=x[...,:d],x[...,d:]
        return torch.cat([x1*cos_r+x2*sin_r, x1*(-sin_r)+x2*cos_r], dim=-1)
    all_q = _rope(all_q)
    all_k = _rope(all_k)
    all_q = F.rms_norm(all_q,(head_dim,))*1.2
    all_k = F.rms_norm(all_k,(head_dim,))*1.2

    new_k = all_k.clone()
    new_v = all_v.clone()

    if cached_k is not None and cached_k.shape[2] > 0:
        ck = cached_k.transpose(2, 3)
        cv = cached_v.transpose(2, 3)
        all_k = torch.cat([ck, all_k], dim=-2)
        all_v = torch.cat([cv, all_v], dim=-2)

    S_plus_T = all_k.shape[-2]
    if n_kv_groups > 1:
        all_k = all_k[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,S_plus_T,head_dim).reshape(n_par,B,n_head,S_plus_T,head_dim)
        all_v = all_v[:,:,:,None,:,:].expand(n_par,B,n_kv_head,n_kv_groups,S_plus_T,head_dim).reshape(n_par,B,n_head,S_plus_T,head_dim)

    scaling = head_dim ** -0.5
    attn_w = torch.matmul(all_q, all_k.transpose(-2,-1)) * scaling
    if T > 1:
        ri = torch.arange(T,device=all_h_prev.device).unsqueeze(1)
        ci = torch.arange(S_plus_T,device=all_h_prev.device).unsqueeze(0)
        S = S_plus_T - T
        causal = ci <= (ri + S)
        attn_w = attn_w.masked_fill(~causal[None,None,None], float('-inf'))
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(dt)
    attn_out = torch.matmul(attn_w, all_v)

    attn_out = attn_out.permute(0,1,3,2,4).reshape(n_par,B,T,D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, W_o)
    all_hidden = all_x_in + attn_out

    h_n = F.rms_norm(all_hidden, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, W_fc)
    mlp = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), W_proj)

    return all_hidden + mlp, new_k, new_v


@torch.inference_mode()
def ar_generate_idn_batched(model, prompt_ids, pw, seq_layers, max_new=16,
                             K=1, elk_k=0.0, init='warm', preheat_cache=None):
    """AR generation with IDN batched + KV cache.

    Init strategies for AR (different from PPL!):
      - h0: all parallel layers init to h_init each step (no memory)
      - warm: use previous step's hidden states (default)
      - batch_fwd: run one extra batched forward on h_init each step
      - preheat: predict from current token's x0

    Only IDN batched supports AR gen because it processes layers separately
    with per-layer KV caches. Fused/Chunkwise merge layers so they can't
    maintain per-layer KV caches.
    """

    L = model.config.n_layer
    par = pw.par; n_par = pw.n_par
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // model.config.n_head

    kv = KVCache(1, n_kv_head, len(prompt_ids)+max_new+1, head_dim, L, device, dtype)
    ids = torch.tensor([prompt_ids], device=device)

    # Prefill sequentially
    logits = model.forward(ids, kv_cache=kv)

    # Get warm-start hidden states from prefill (only needed for 'warm' init)
    prev_h = None
    if init == 'warm':
        x0_pf, cs_pf, ve_pf = _embed(model, ids)
        prev_h = [None]*L; x = x0_pf
        for i in range(L):
            x_in = model.resid_lambdas[i]*x + model.x0_lambdas[i]*x0_pf
            x = model.transformer.h[i](x_in, ve_pf.get(str(i)), cs_pf, model.window_sizes[i], None)
            prev_h[i] = x[:,-1:,:].clone()

    gen = []
    for step in range(max_new):
        tok = logits[0,-1].argmax().item(); gen.append(tok)
        ids_step = torch.tensor([[tok]], device=device)

        # Embed
        x0 = model.transformer.wte(ids_step).to(dtype); x0 = norm(x0)
        if kv.prev_embedding is not None:
            gate = model.smear_lambda.to(dtype)*torch.sigmoid(model.smear_gate(x0[:,:,:24]))
            x0 = x0 + gate * kv.prev_embedding

        pos = kv.get_pos()
        cos_sin = model.cos[:,pos:pos+1], model.sin[:,pos:pos+1]
        if getattr(model.config, 'no_ve', False):
            ve = {}
        else:
            ve = {k: model.value_embeds[k](ids_step).to(dtype) for k in model.value_embeds}

        # Sequential prefix with KV cache
        x = x0; h_step = [None]*L
        for i in range(seq_layers):
            x_in = model.resid_lambdas[i]*x + model.x0_lambdas[i]*x0
            x = model.transformer.h[i](x_in, ve.get(str(i)), cos_sin, model.window_sizes[i], kv)
            h_step[i] = x
        h_init = x

        # Extract cached K,V for parallel layers
        cached_k = kv.k_cache[par, :, :pos, :, :]
        cached_v = kv.v_cache[par, :, :pos, :, :]

        # Initialize parallel layers based on strategy
        if init == 'warm' and prev_h is not None:
            hs = [h_init] + [prev_h[par[j]].clone() for j in range(n_par)]
        elif init == 'batch_fwd':
            # One extra batched forward on h_init to get init estimates
            all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
            all_init, _, _ = _batched_block_forward_kv(
                all_h0, x0, cos_sin, ve, par, model,
                cached_k=cached_k, cached_v=cached_v, pw=pw)
            hs = [h_init] + [all_init[j] for j in range(n_par)]
        elif init == 'preheat' and preheat_cache is not None:
            h_pred = preheat_cache.predict(x0)
            hs = [h_init] + [h_pred[par[j]] for j in range(n_par)]
        else:
            # h0: all layers init to h_init
            hs = [h_init] + [h_init.clone() for _ in range(n_par)]

        for _k in range(K):
            all_hp = torch.stack([hs[j] for j in range(n_par)])
            all_out, _, _ = _batched_block_forward_kv(
                all_hp, x0, cos_sin, ve, par, model,
                cached_k=cached_k, cached_v=cached_v, pw=pw)
            _idn_correction(list(all_out), hs, h_init, n_par, elk_k)

        for j, li in enumerate(par):
            h_step[li] = hs[j+1]

        # Batched KV cache update (like Qwen demo's _batched_cache_update)
        _update_kv_cache(model, kv, par, h_step, h_init, x0, cos_sin, ve, pw)

        # Advance cache + smear
        kv.cache_seqlens += 1
        kv.prev_embedding = norm(model.transformer.wte(ids_step).to(dtype))

        # Save hidden states for warm-start (all inits save for potential future use)
        if init == 'warm':
            if prev_h is None:
                prev_h = [None]*L
            for i in range(L):
                if h_step[i] is not None:
                    prev_h[i] = h_step[i].clone()

        # Logits
        x_final = h_step[L-1]
        bo = L//2
        if bo < L and h_step[bo] is not None:
            x_final = x_final - model.backout_lambda.to(dtype)*h_step[bo]
        x_final = norm(x_final)
        logits = model.lm_head(x_final)[...,:model.config.vocab_size].float()
        logits = 15*torch.tanh(logits/15)

    return gen


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--n-par", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--avg", action="store_true", help="Use avg reduction in fused O/MLP (divide by chunk size)")
    args = parser.parse_args()
    use_avg = args.avg

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()

    model = load_model(args.model_tag, device=device)
    L = model.config.n_layer
    n_par = args.n_par
    seq_layers = L - n_par
    par = list(range(seq_layers, L))

    print(f"Model: {args.model_tag}, L={L}, n_par={n_par}, seq_layers={seq_layers}")

    # Load data
    batches = load_val_data(device, args.max_tokens)
    total_tokens = len(batches)*128
    print(f"Val: {total_tokens:,} tokens, {len(batches)} batches, BS=1")

    # Precompute weights
    print("Precomputing weights...")
    pw = PrecomputedWeights(model, par)

    # Chunk configs
    chunk_configs = {}
    # 1 × fused_n_par (= pure fused)
    chunk_configs['1xF'+str(n_par)] = [ChunkWeights(model, par)]
    # n_par × fused1 (= chunksize 1)
    chunk_configs[str(n_par)+'xF1'] = [ChunkWeights(model, [li]) for li in par]
    # 2 × fused(n_par/2) if divisible
    if n_par % 2 == 0:
        half = n_par // 2
        chunk_configs['2xF'+str(half)] = [
            ChunkWeights(model, par[:half]),
            ChunkWeights(model, par[half:]),
        ]
    # 4 × fused(n_par/4) if divisible
    if n_par % 4 == 0:
        q = n_par // 4
        chunk_configs['4xF'+str(q)] = [
            ChunkWeights(model, par[i*q:(i+1)*q]) for i in range(4)
        ]

    # Preheat calibration
    print("Calibrating preheat (1M tokens)...")
    preheat = calibrate_preheat(model, n_samples=min(128, len(batches)), seq_len=128, rank=64)
    for i in range(preheat.n_layer):
        preheat.V[i] = preheat.V[i].to(dtype)
        preheat.U[i] = preheat.U[i].to(dtype)
        preheat.bias[i] = preheat.bias[i].to(dtype)

    # Sequential baseline
    idx_bench = batches[0][:, :-1]
    seq_ms = bench(lambda: forward_sequential(model, idx_bench), device)
    seq_losses = []
    if device.type == 'cuda': torch.cuda.synchronize()
    t_seq_ppl_start = time.perf_counter()
    for batch in batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(model, idx)
        seq_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
    if device.type == 'cuda': torch.cuda.synchronize()
    seq_ppl_s = time.perf_counter() - t_seq_ppl_start
    seq_ppl = math.exp(sum(seq_losses)/len(seq_losses))
    print(f"Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}, ppl_eval={seq_ppl_s:.1f}s")

    # AR generation ground truth
    ar_refs = {}
    for prompt in AR_PROMPTS:
        tokens = tokenizer.encode(prompt, prepend=bos)[:8]  # prefill 8
        ar_refs[prompt] = ar_generate_sequential(model, tokens, max_new=16)

    # Evaluate all configs
    results = {'seq_ppl': seq_ppl, 'seq_ms': seq_ms, 'seq_ppl_s': seq_ppl_s,
                'model': args.model_tag, 'n_par': n_par, 'avg': use_avg}

    K_values = [1, 2, 4, 8]
    elk_k_configs = [(1, 0.1)]  # extra: K=1 with elk_k=0.1
    init_strategies = ['h0', 'warm', 'batch_fwd', 'preheat']

    # Methods now take init/preheat_cache internally (no external get_init_h for timing)
    # For 'warm' init, we still use get_init_h externally (pre-computed, free in AR)
    methods = {
        'IDN_batched': lambda idx, K, ek, init, ih: forward_idn_batched(model, idx, seq_layers, pw, K=K, elk_k=ek, init_h=ih, init=init, preheat_cache=preheat),
        'Fused': lambda idx, K, ek, init, ih: forward_fused(model, idx, seq_layers, pw, chunk_configs['1xF'+str(n_par)], K=K, elk_k=ek, init_h=ih, avg=use_avg),
        'Fused+split': lambda idx, K, ek, init, ih: forward_fused_split(model, idx, seq_layers, pw, K=K, elk_k=ek, init_h=ih, avg=use_avg, init=init, preheat_cache=preheat),
    }
    for cname, cws in chunk_configs.items():
        methods[f'Chunk_{cname}'] = lambda idx, K, ek, init, ih, _cws=cws: forward_chunkwise(model, idx, seq_layers, pw, _cws, K=K, elk_k=ek, init_h=ih, avg=use_avg, init=init, preheat_cache=preheat)
    for cname, cws in chunk_configs.items():
        if len(set(cw.n for cw in cws)) == 1 and len(cws) > 1:
            methods[f'ChunkB_{cname}'] = lambda idx, K, ek, init, ih, _cws=cws: forward_chunkwise_batched(model, idx, seq_layers, pw, _cws, K=K, elk_k=ek, init_h=ih, avg=use_avg, init=init, preheat_cache=preheat)

    all_configs = []
    for K in K_values:
        all_configs.append((K, 0.0))
    all_configs.append((1, 0.1))

    for K, elk_k in all_configs:
        k_key = f'K={K}_elk={elk_k}'
        results[k_key] = {}

        for method_name, method_fn in methods.items():
            results[k_key][method_name] = {}

            for init_name in init_strategies:
                print(f"  {k_key} {method_name} {init_name}...", end='', flush=True)

                # For warm: pre-compute init_h externally (free in AR simulation)
                # For batch_fwd/preheat: computed internally (cost included in timing)
                # For h0: no init needed
                if init_name == 'warm':
                    warm_ih = get_init_h(model, idx_bench, par, 'warm', preheat, pw)
                    ms = bench(lambda _mfn=method_fn, _K=K, _ek=elk_k, _ih=warm_ih:
                               _mfn(idx_bench, _K, _ek, 'h0', _ih), device)
                else:
                    ms = bench(lambda _mfn=method_fn, _K=K, _ek=elk_k, _init=init_name:
                               _mfn(idx_bench, _K, _ek, _init, None), device)

                # PPL eval
                losses = []
                for batch in batches:
                    idx, tgt = batch[:, :-1], batch[:, 1:]
                    if init_name == 'warm':
                        ih = get_init_h(model, idx, par, 'warm', preheat, pw)
                        logits = method_fn(idx, K, elk_k, 'h0', ih)
                    else:
                        logits = method_fn(idx, K, elk_k, init_name, None)
                    losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
                ppl = math.exp(min(sum(losses)/len(losses), 20))

                speed = seq_ms / ms
                dppl = 100*(ppl - seq_ppl)/seq_ppl

                results[k_key][method_name][init_name] = {
                    'ms': ms, 'speed': speed, 'ppl': ppl, 'dppl': dppl,
                }
                print(f" {ms:.1f}ms {speed:.2f}x PPL={ppl:.2f} ({dppl:+.1f}%)")

    # ---------------------------------------------------------------------------
    # AR generation evaluation
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"AR Generation (prefill=8, gen=16)")
    print(f"{'='*60}")

    # AR ground truth
    for prompt in AR_PROMPTS:
        tokens = tokenizer.encode(prompt, prepend=bos)[:8]
        ar_refs[prompt] = ar_generate_sequential(model, tokens, max_new=16)

    # AR timing (sequential baseline)
    tokens_bench = tokenizer.encode(AR_PROMPTS[0], prepend=bos)[:8]
    ar_seq_ms = bench(lambda: ar_generate_sequential(model, tokens_bench, max_new=16), device)
    print(f"  Sequential AR: {ar_seq_ms:.2f}ms")

    ar_inits = ['h0', 'warm', 'batch_fwd', 'preheat']

    # --- IDN batched AR ---
    print(f"\n  IDN_batched:")
    for K in [1, 2, 4, 8]:
        for elk_k in [0.0]:
            for ar_init in ar_inits:
                k_key = f'AR_K={K}_elk={elk_k}_{ar_init}'
                print(f"    K={K} elk={elk_k} init={ar_init}...", end='', flush=True)

                total_match = 0
                total_gen = 0
                for prompt in AR_PROMPTS:
                    tokens = tokenizer.encode(prompt, prepend=bos)[:8]
                    gen = ar_generate_idn_batched(
                        model, tokens, pw, seq_layers,
                        max_new=16, K=K, elk_k=elk_k,
                        init=ar_init, preheat_cache=preheat)
                    ref = ar_refs[prompt]
                    match = sum(1 for a, b in zip(gen, ref) if a == b)
                    total_match += match
                    total_gen += len(ref)

                match_rate = total_match / total_gen

                ar_ms = bench(lambda _K=K, _ek=elk_k, _init=ar_init: ar_generate_idn_batched(
                    model, tokens_bench, pw, seq_layers, max_new=16,
                    K=_K, elk_k=_ek, init=_init, preheat_cache=preheat), device)
                ar_speed = ar_seq_ms / ar_ms

                results[k_key] = {
                    'match_rate': match_rate,
                    'ms': ar_ms,
                    'speed': ar_speed,
                }
                print(f" match={match_rate:.3f} ({total_match}/{total_gen}), {ar_ms:.2f}ms ({ar_speed:.2f}x)")

    # --- Chunkwise AR (each chunk = "wide layer" with its own KV cache) ---
    ar_chunk_inits = ['h0', 'warm']
    for cname, cws in chunk_configs.items():
        print(f"\n  Chunk_{cname}:")
        for K in [1, 4, 8]:
            for ar_init in ar_chunk_inits:
                k_key = f'AR_Chunk_{cname}_K={K}_{ar_init}'
                print(f"    K={K} init={ar_init}...", end='', flush=True)

                total_match = 0
                total_gen = 0
                for prompt in AR_PROMPTS:
                    tokens = tokenizer.encode(prompt, prepend=bos)[:8]
                    gen = ar_generate_chunkwise(
                        model, tokens, pw, seq_layers, cws,
                        max_new=16, K=K, init=ar_init)
                    ref = ar_refs[prompt]
                    match = sum(1 for a, b in zip(gen, ref) if a == b)
                    total_match += match
                    total_gen += len(ref)

                match_rate = total_match / total_gen

                ar_ms = bench(lambda _K=K, _init=ar_init, _cws=cws: ar_generate_chunkwise(
                    model, tokens_bench, pw, seq_layers, _cws,
                    max_new=16, K=_K, init=_init), device)
                ar_speed = ar_seq_ms / ar_ms

                results[k_key] = {
                    'match_rate': match_rate,
                    'ms': ar_ms,
                    'speed': ar_speed,
                }
                print(f" match={match_rate:.3f} ({total_match}/{total_gen}), {ar_ms:.2f}ms ({ar_speed:.2f}x)")

    # Save results
    outpath = args.output or f'cache/eval_logs/idn_new_{args.model_tag}_npar{n_par}.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
