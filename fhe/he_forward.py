"""
HE-approximate forward passes for sequential and SNLP inference.

Replaces nonlinear operations (RMSNorm, softmax, sigmoid, tanh) with
polynomial approximations to simulate FHE-compatible inference.

The key test: does SNLP (K iterations) accumulate less approximation error
than sequential (L layers)?

Usage:
    from fhe.he_forward import forward_sequential_he, forward_idn_batched_he
    from fhe.he_approx import HEApproxConfig

    cfg = HEApproxConfig.uniform(degree=4)
    logits_seq = forward_sequential_he(model, idx, cfg)
    logits_snlp = forward_idn_batched_he(model, idx, seq_layers, pw, cfg, K=2)
"""

import torch
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.jacobi_forward import _embed
from snlp.inference_idn import PrecomputedWeights, ChunkWeights, _idn_correction
from snlp.inference_hcn import (
    MHCPrecomputedWeights, _hc_width, _hc_depth, _mhc_newton_correction,
    _embed_mhc, _logits_mhc, _run_block_mhc,
)

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from fhe.he_approx import (
    HEApproxConfig, poly_rms_norm, poly_softmax, poly_sigmoid, poly_tanh,
    poly_scaled_dot_product_attention, add_ckks_noise,
)


# ---------------------------------------------------------------------------
# Per-block forward with HE approximations (sequential path)
# ---------------------------------------------------------------------------

def _run_block_he(model, i, h_prev, x0, cos_sin, value_embeds, cfg: HEApproxConfig):
    """Single block forward with polynomial approximations.

    Mirrors _run_block + Block.forward but with HE-approximate nonlinearities.
    When cfg.is_exact, produces identical results to the standard forward.
    """
    D = model.config.n_embd
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = D // n_head
    window_size = model.window_sizes[i]

    # Residual blending
    if getattr(model.config, 'no_x0_resid', False):
        x_in = h_prev
    else:
        x_in = model.resid_lambdas[i] * h_prev + model.x0_lambdas[i] * x0

    # Pre-attention norm (HE-approx)
    x_normed = poly_rms_norm(x_in, (D,), cfg)
    x_normed = add_ckks_noise(x_normed, cfg.ckks_noise_bits)

    block = model.transformer.h[i]
    B, T, _ = x_normed.shape

    # QKV projections (linear — HE-friendly)
    q = block.attn.c_q(x_normed).view(B, T, n_head, head_dim)
    k = block.attn.c_k(x_normed).view(B, T, n_kv_head, head_dim)
    v = block.attn.c_v(x_normed).view(B, T, n_kv_head, head_dim)

    # Value embeddings with sigmoid gate (HE-approx)
    ve = value_embeds.get(str(i))
    if ve is not None and block.attn.ve_gate is not None:
        ve_r = ve.view(B, T, n_kv_head, head_dim)
        gate_input = block.attn.ve_gate(x_normed[..., :block.attn.ve_gate_channels])
        gate = 3 * poly_sigmoid(gate_input, cfg)
        gate = add_ckks_noise(gate, cfg.ckks_noise_bits)
        v = v + gate.unsqueeze(-1) * ve_r

    # RoPE (linear — HE-friendly)
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # QK-norm (HE-approx)
    q = poly_rms_norm(q, (head_dim,), cfg) * 1.2
    k = poly_rms_norm(k, (head_dim,), cfg) * 1.2
    q = add_ckks_noise(q, cfg.ckks_noise_bits)
    k = add_ckks_noise(k, cfg.ckks_noise_bits)

    # Attention
    if cfg.softmax_degree is None:
        # Exact: use the same flash_attn path as the model
        y = fa.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    else:
        # HE-approx: manual SDPA with polynomial softmax (no window support)
        n_kv_groups = n_head // n_kv_head
        if n_kv_groups > 1:
            k = k[:, :, :, None, :].expand(B, T, n_kv_head, n_kv_groups, head_dim).reshape(B, T, n_head, head_dim)
            v = v[:, :, :, None, :].expand(B, T, n_kv_head, n_kv_groups, head_dim).reshape(B, T, n_head, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = poly_scaled_dot_product_attention(q_t, k_t, v_t, cfg, is_causal=True)
        y = y.transpose(1, 2)  # back to (B, T, H, D)
    y = add_ckks_noise(y, cfg.ckks_noise_bits)

    # O projection (linear)
    y = y.contiguous().view(B, T, -1)
    attn_out = block.attn.c_proj(y)

    # Residual connection
    h = x_in + attn_out

    # Pre-MLP norm (HE-approx)
    h_normed = poly_rms_norm(h, (D,), cfg)
    h_normed = add_ckks_noise(h_normed, cfg.ckks_noise_bits)

    # MLP: FC up → ReLU² → FC down (ReLU² is already polynomial)
    mlp_out = block.mlp.c_fc(h_normed)
    mlp_out = F.relu(mlp_out).square()
    mlp_out = add_ckks_noise(mlp_out, cfg.ckks_noise_bits)
    mlp_out = block.mlp.c_proj(mlp_out)

    return h + mlp_out


def _run_block_mhc_he(model, i, h_prev, x0, cos_sin, value_embeds, cfg: HEApproxConfig):
    """Single mHC block forward with HE approximations.

    HC width/depth connections are purely linear (no HE approximation needed).
    Only the attention and MLP sublayers get polynomial approximations.
    Input/output: (B*S, T, D) where S = mhc_num_streams.
    """
    D = model.config.n_embd
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = D // n_head
    S = model.config.mhc_num_streams
    window_size = model.window_sizes[i]
    block = model.transformer.h[i]

    x_in = h_prev  # mHC always has no_x0_resid=True

    # --- Attention ---
    # HC width: (B*S, T, D) → branch_in (B, T, D)
    branch_in, res_out, H_post = block.hc_attn.width_connection(
        x_in, model.config.mhc_sinkhorn_iters, model.config.mhc_sinkhorn_tau)
    B, T, _ = branch_in.shape

    x_normed = poly_rms_norm(branch_in, (D,), cfg)
    x_normed = add_ckks_noise(x_normed, cfg.ckks_noise_bits)

    q = block.attn.c_q(x_normed).view(B, T, n_head, head_dim)
    k = block.attn.c_k(x_normed).view(B, T, n_kv_head, head_dim)
    v = block.attn.c_v(x_normed).view(B, T, n_kv_head, head_dim)

    ve = value_embeds.get(str(i))
    if ve is not None and block.attn.ve_gate is not None:
        ve_r = ve.view(B, T, n_kv_head, head_dim)
        gate_input = block.attn.ve_gate(x_normed[..., :block.attn.ve_gate_channels])
        gate = 3 * poly_sigmoid(gate_input, cfg)
        gate = add_ckks_noise(gate, cfg.ckks_noise_bits)
        v = v + gate.unsqueeze(-1) * ve_r

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    q = poly_rms_norm(q, (head_dim,), cfg) * 1.2
    k = poly_rms_norm(k, (head_dim,), cfg) * 1.2
    q = add_ckks_noise(q, cfg.ckks_noise_bits)
    k = add_ckks_noise(k, cfg.ckks_noise_bits)

    if cfg.softmax_degree is None:
        y = fa.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    else:
        n_kv_groups = n_head // n_kv_head
        if n_kv_groups > 1:
            k = k[:, :, :, None, :].expand(B, T, n_kv_head, n_kv_groups, head_dim).reshape(B, T, n_head, head_dim)
            v = v[:, :, :, None, :].expand(B, T, n_kv_head, n_kv_groups, head_dim).reshape(B, T, n_head, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = poly_scaled_dot_product_attention(q_t, k_t, v_t, cfg, is_causal=True)
        y = y.transpose(1, 2)
    y = add_ckks_noise(y, cfg.ckks_noise_bits)

    y = y.contiguous().view(B, T, -1)
    attn_out = block.attn.c_proj(y)

    # HC depth: (B, T, D) → (B*S, T, D)
    x = block.hc_attn.depth_connection(attn_out, res_out, H_post)

    # --- MLP ---
    branch_in_mlp, res_out_mlp, H_post_mlp = block.hc_mlp.width_connection(
        x, model.config.mhc_sinkhorn_iters, model.config.mhc_sinkhorn_tau)

    h_normed = poly_rms_norm(branch_in_mlp, (D,), cfg)
    h_normed = add_ckks_noise(h_normed, cfg.ckks_noise_bits)

    mlp_out = block.mlp.c_fc(h_normed)
    mlp_out = F.relu(mlp_out).square()
    mlp_out = add_ckks_noise(mlp_out, cfg.ckks_noise_bits)
    mlp_out = block.mlp.c_proj(mlp_out)

    return block.hc_mlp.depth_connection(mlp_out, res_out_mlp, H_post_mlp)


def _logits_he(model, h, cfg: HEApproxConfig):
    """Compute logits with HE-approximate operations."""
    if cfg.is_exact:
        from nanochat.jacobi_forward import _logits
        return _logits(model, h)

    n_layer = model.config.n_layer
    x_final = h[-1]
    backout_layer = n_layer // 2
    if backout_layer < n_layer:
        x_final = x_final - model.backout_lambda.to(x_final.dtype) * h[backout_layer]

    x_final = poly_rms_norm(x_final, (model.config.n_embd,), cfg)
    x_final = add_ckks_noise(x_final, cfg.ckks_noise_bits)

    softcap = 15
    logits = model.lm_head(x_final)
    logits = logits[..., :model.config.vocab_size]
    logits = logits.float()
    logits = softcap * poly_tanh(logits / softcap, cfg)

    return logits


# ---------------------------------------------------------------------------
# Sequential forward with HE approximations
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential_he(model, idx, cfg: HEApproxConfig):
    """Full sequential forward pass with polynomial approximations in every block."""
    L = model.config.n_layer
    use_mhc = getattr(model.config, 'use_mhc', False)

    if use_mhc:
        x0, cos_sin, ve = _embed_he(model, idx, cfg)
    else:
        x0, cos_sin, ve = _embed(model, idx)

    h = [None] * L
    x = x0
    for i in range(L):
        if use_mhc:
            x = _run_block_mhc_he(model, i, x, x0, cos_sin, ve, cfg)
        else:
            x = _run_block_he(model, i, x, x0, cos_sin, ve, cfg)
        h[i] = x

    if use_mhc:
        return _logits_mhc_he(model, h, cfg)
    return _logits_he(model, h, cfg)


# ---------------------------------------------------------------------------
# Batched forward with HE approximations (SNLP path)
# ---------------------------------------------------------------------------

def _batched_forward_he(all_h_prev, x0, cos_sin, ve, pw, model, cfg: HEApproxConfig):
    """Batched forward via stacked einsum with HE-approximate nonlinearities.

    Mirrors _batched_forward from inference_idn.py but replaces nonlinear ops.
    Returns (n_par, B, T, D).
    """
    n_par = pw.n_par
    par = pw.par
    B, T, D = x0.shape
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head

    # Residual blending (linear)
    all_x_in = (pw.resid_lambdas.view(n_par, 1, 1, 1) * all_h_prev
                + pw.x0_lambdas.view(n_par, 1, 1, 1) * x0.unsqueeze(0))

    # Pre-attention norm (HE-approx)
    all_x_normed = poly_rms_norm(all_x_in, (D,), cfg)
    all_x_normed = add_ckks_noise(all_x_normed, cfg.ckks_noise_bits)

    # QKV projections (linear — HE-friendly)
    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_q_s)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_k_s)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, pw.W_v_s)

    all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

    # Value embeddings with sigmoid gate (HE-approx)
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_j.view(B, T, n_kv_head, head_dim)
            gate_input = F.linear(all_x_normed[j, :, :, :12], pw.ve_gate_weights[j])
            gate = 3 * poly_sigmoid(gate_input, cfg)
            gate = add_ckks_noise(gate, cfg.ckks_noise_bits)
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_r.permute(0, 2, 1, 3)

    # RoPE (linear — HE-friendly)
    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r, x1 * (-sin_r) + x2 * cos_r], dim=-1)

    # QK-norm (HE-approx)
    all_q = poly_rms_norm(_rope(all_q), (head_dim,), cfg) * 1.2
    all_k = poly_rms_norm(_rope(all_k), (head_dim,), cfg) * 1.2
    all_q = add_ckks_noise(all_q, cfg.ckks_noise_bits)
    all_k = add_ckks_noise(all_k, cfg.ckks_noise_bits)

    # GQA expansion
    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)

    # Attention with polynomial softmax (HE-approx)
    q_s = all_q.reshape(n_par * B, n_head, T, head_dim)
    k_s = all_k.reshape(n_par * B, n_head, T, head_dim)
    v_s = all_v.reshape(n_par * B, n_head, T, head_dim)
    y_s = poly_scaled_dot_product_attention(q_s, k_s, v_s, cfg, is_causal=True)
    y_s = add_ckks_noise(y_s, cfg.ckks_noise_bits)

    # O projection (linear)
    attn_out = y_s.reshape(n_par, B, n_head, T, head_dim).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o_s)
    all_hidden = all_x_in + attn_out

    # Pre-MLP norm (HE-approx)
    h_n = poly_rms_norm(all_hidden, (D,), cfg)
    h_n = add_ckks_noise(h_n, cfg.ckks_noise_bits)

    # MLP: ReLU² is already polynomial
    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    mlp = add_ckks_noise(mlp, cfg.ckks_noise_bits)

    return all_hidden + mlp


# ---------------------------------------------------------------------------
# SNLP IDN batched forward with HE approximations
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_idn_batched_he(model, idx, seq_layers, pw, cfg: HEApproxConfig,
                            K=1, elk_k=0.0, init='h0'):
    """SNLP IDN batched inference with polynomial approximations.

    The prefix layers (0..seq_layers-1) also use HE approximations since
    in full-FHE mode everything runs under encryption. The suffix layers
    use the batched parallel path with HE-approximate block evaluations.
    IDN correction itself is purely additive (HE-friendly, no approximation needed).
    """
    L = model.config.n_layer
    par = pw.par
    n_par = pw.n_par
    x0, cos_sin, ve = _embed(model, idx)

    # Sequential prefix (with HE approximations)
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_he(model, i, x, x0, cos_sin, ve, cfg)
        h[i] = x
    h_init = x

    # Initialize suffix
    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward_he(all_h0, x0, cos_sin, ve, pw, model, cfg)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:  # h0
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    # K iterations of batched forward + IDN correction
    for _k in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward_he(all_hp, x0, cos_sin, ve, pw, model, cfg)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    for j, li in enumerate(par):
        h[li] = hs[j + 1]

    return _logits_he(model, h, cfg)


# ---------------------------------------------------------------------------
# mHC-Newton batched forward with HE approximations
# ---------------------------------------------------------------------------

def _embed_he(model, idx, cfg: HEApproxConfig):
    """Embed with HE-approximate smear gate sigmoid."""
    if cfg.is_exact:
        return _embed_mhc(model, idx) if getattr(model.config, 'use_mhc', False) else _embed(model, idx)

    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    if T > 1:
        gate = model.smear_lambda.to(x.dtype) * poly_sigmoid(model.smear_gate(x[:, 1:, :24]), cfg)
        gate = add_ckks_noise(gate, cfg.ckks_noise_bits)
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    value_embeds = {}
    if not getattr(model.config, 'no_ve', False):
        for key in model.value_embeds:
            value_embeds[key] = model.value_embeds[key](idx).to(x.dtype)

    if getattr(model.config, 'use_mhc', False):
        S = model.config.mhc_num_streams
        D = x.shape[-1]
        x = x.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)

    return x, cos_sin, value_embeds


def _logits_mhc_he(model, h, cfg: HEApproxConfig):
    """Compute logits for mHC models with HE-approximate operations."""
    if cfg.is_exact:
        return _logits_mhc(model, h)

    n_layer = model.config.n_layer
    x_final = h[-1]
    backout_layer = n_layer // 2

    if getattr(model.config, 'use_mhc', False):
        S = model.config.mhc_num_streams
        BS, T, D = x_final.shape
        B = BS // S
        x_final = x_final.view(B, S, T, D).sum(dim=1)
        if backout_layer < n_layer and h[backout_layer] is not None:
            x_back = h[backout_layer].view(B, S, T, D).sum(dim=1)
            x_final = x_final - model.backout_lambda.to(x_final.dtype) * x_back
    else:
        if backout_layer < n_layer and h[backout_layer] is not None:
            x_final = x_final - model.backout_lambda.to(x_final.dtype) * h[backout_layer]

    x_final = poly_rms_norm(x_final, (model.config.n_embd,), cfg)
    x_final = add_ckks_noise(x_final, cfg.ckks_noise_bits)

    softcap = 15
    logits = model.lm_head(x_final)
    logits = logits[..., :model.config.vocab_size]
    logits = logits.float()
    logits = softcap * poly_tanh(logits / softcap, cfg)
    return logits


def _batched_forward_mhc_he(all_h_prev, x0, cos_sin, ve, pw, model, cfg: HEApproxConfig):
    """Batched forward for mHC models with HE-approximate nonlinearities.

    Mirrors _batched_forward_mhc from inference_hcn.py.
    HC width/depth connections are purely linear — no HE approximation needed.
    """
    n_par = pw.n_par
    par = pw.par
    S = pw.S
    BS, T, D = x0.shape
    B = BS // S
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = D // n_head
    n_kv_groups = n_head // n_kv_head

    all_x = all_h_prev

    # --- Attention HC (linear — no HE approx) ---
    branch_in, res_out = _hc_width(all_x, pw.H_res_attn, pw.H_pre_attn, S)

    bn = poly_rms_norm(branch_in, (D,), cfg)
    bn = add_ckks_noise(bn, cfg.ckks_noise_bits)

    all_q = torch.einsum('lbtd,lhd->lbth', bn, pw.W_q_s)
    all_k = torch.einsum('lbtd,lhd->lbth', bn, pw.W_k_s)
    all_v = torch.einsum('lbtd,lhd->lbth', bn, pw.W_v_s)

    all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_j.view(B, T, n_kv_head, head_dim)
            gate = 3 * poly_sigmoid(F.linear(bn[j, :, :, :12], pw.ve_gate_weights[j]), cfg)
            gate = add_ckks_noise(gate, cfg.ckks_noise_bits)
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_r.permute(0, 2, 1, 3)

    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r, x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = poly_rms_norm(_rope(all_q), (head_dim,), cfg) * 1.2
    all_k = poly_rms_norm(_rope(all_k), (head_dim,), cfg) * 1.2
    all_q = add_ckks_noise(all_q, cfg.ckks_noise_bits)
    all_k = add_ckks_noise(all_k, cfg.ckks_noise_bits)

    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)

    q_s = all_q.reshape(n_par * B, n_head, T, head_dim)
    k_s = all_k.reshape(n_par * B, n_head, T, head_dim)
    v_s = all_v.reshape(n_par * B, n_head, T, head_dim)

    if cfg.softmax_degree is None:
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)
    else:
        y_s = poly_scaled_dot_product_attention(q_s, k_s, v_s, cfg, is_causal=True)
    y_s = add_ckks_noise(y_s, cfg.ckks_noise_bits)

    attn_out = y_s.reshape(n_par, B, n_head, T, head_dim).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o_s)

    all_x = _hc_depth(attn_out, res_out, pw.H_post_attn, S)

    # --- MLP HC ---
    branch_in_mlp, res_out_mlp = _hc_width(all_x, pw.H_res_mlp, pw.H_pre_mlp, S)

    h_n = poly_rms_norm(branch_in_mlp, (D,), cfg)
    h_n = add_ckks_noise(h_n, cfg.ckks_noise_bits)

    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp_out = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    mlp_out = add_ckks_noise(mlp_out, cfg.ckks_noise_bits)

    return _hc_depth(mlp_out, res_out_mlp, pw.H_post_mlp, S)


@torch.inference_mode()
def forward_mhc_batched_he(model, idx, seq_layers, pw, cfg: HEApproxConfig,
                            K=1, init='h0'):
    """mHC-Newton batched inference with HE approximations."""
    L = model.config.n_layer
    n_par = pw.n_par
    use_mhc = pw.use_mhc

    x0, cos_sin, ve = _embed_he(model, idx, cfg)

    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_mhc_he(model, i, x, x0, cos_sin, ve, cfg)
        h[i] = x
    h_init = x

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward_mhc_he(all_h0, x0, cos_sin, ve, pw, model, cfg)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _k in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward_mhc_he(all_hp, x0, cos_sin, ve, pw, model, cfg)
        if use_mhc:
            _mhc_newton_correction(all_out, hs, h_init, n_par, pw)
        else:
            from snlp.inference_hcn import _idn_correction as _idn_corr_hcn
            _idn_corr_hcn(all_out, hs, h_init, n_par)

    for j, li in enumerate(pw.par):
        h[li] = hs[j + 1]
    return _logits_mhc_he(model, h, cfg)


# ---------------------------------------------------------------------------
# ChunkB (chunkwise fused) forward with HE approximations
# ---------------------------------------------------------------------------

def _fused_forward_chunk_he(h_in, x0, cos_sin, cw, model, cfg: HEApproxConfig, avg=False):
    """One fused chunk forward with HE-approximate nonlinearities.

    Mirrors _fused_forward_chunk from inference_idn.py.
    """
    B, T, D = h_in.shape
    n_head = model.config.n_head
    total_heads = cw.n * n_head
    head_dim = D // n_head

    x_in = cw.avg_resid * h_in + cw.avg_x0 * x0
    x_normed = poly_rms_norm(x_in, (D,), cfg)
    x_normed = add_ckks_noise(x_normed, cfg.ckks_noise_bits)

    q = F.linear(x_normed, cw.W_q).view(B, T, total_heads, head_dim)
    k = F.linear(x_normed, cw.W_k).view(B, T, total_heads, head_dim)
    v = F.linear(x_normed, cw.W_v).view(B, T, total_heads, head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    q = poly_rms_norm(q, (head_dim,), cfg) * 1.2
    k = poly_rms_norm(k, (head_dim,), cfg) * 1.2
    q = add_ckks_noise(q, cfg.ckks_noise_bits)
    k = add_ckks_noise(k, cfg.ckks_noise_bits)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if cfg.softmax_degree is None:
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    else:
        y = poly_scaled_dot_product_attention(q_t, k_t, v_t, cfg, is_causal=True)
    y = add_ckks_noise(y, cfg.ckks_noise_bits)

    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    attn_out = F.linear(y, cw.W_o)
    if avg and cw.n > 1:
        attn_out = attn_out / cw.n
    x_mid = x_in + attn_out

    x_mid_n = poly_rms_norm(x_mid, (D,), cfg)
    x_mid_n = add_ckks_noise(x_mid_n, cfg.ckks_noise_bits)

    hf = F.linear(x_mid_n, cw.W_fc)
    mlp = F.linear(F.relu(hf).square(), cw.W_proj)
    mlp = add_ckks_noise(mlp, cfg.ckks_noise_bits)
    if avg and cw.n > 1:
        mlp = mlp / cw.n
    return x_mid + mlp


@torch.inference_mode()
def forward_idn_chunkwise_he(model, idx, seq_layers, pw, chunk_weights_list,
                              cfg: HEApproxConfig, K=1, elk_k=0.0, init='h0'):
    """ChunkB inference with HE approximations.

    Evaluates C equal-size fused chunks in parallel, with inter-chunk IDN correction.
    """
    L = model.config.n_layer
    n_par = pw.n_par

    x0, cos_sin, ve = _embed_he(model, idx, cfg)

    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_he(model, i, x, x0, cos_sin, ve, cfg)
        h[i] = x
    h_init = x

    C = len(chunk_weights_list)
    cs = n_par // C

    if init == 'batch_fwd':
        hs_list = [h_init]
        for c_idx in range(C):
            out = _fused_forward_chunk_he(h_init, x0, cos_sin, chunk_weights_list[c_idx], model, cfg)
            for _ in range(cs):
                hs_list.append(out.clone())
    else:
        hs_list = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _k in range(K):
        all_chunk_out = []
        for c_idx in range(C):
            chunk_in = hs_list[c_idx * cs]  # input to this chunk
            if c_idx > 0:
                chunk_in = hs_list[c_idx * cs]
            out = _fused_forward_chunk_he(chunk_in, x0, cos_sin, chunk_weights_list[c_idx], model, cfg)
            all_chunk_out.append(out)

        # IDN correction across chunks
        all_out = []
        for c_idx in range(C):
            for _ in range(cs):
                all_out.append(all_chunk_out[c_idx])
        _idn_correction(all_out, hs_list, h_init, n_par, elk_k)

    for j in range(n_par):
        li = seq_layers + j
        h[li] = hs_list[j + 1]

    return _logits_he(model, h, cfg)


# ---------------------------------------------------------------------------
# Sanity check: exact config should match standard forward
# ---------------------------------------------------------------------------

def _sanity_check():
    """Verify that HE-forward with exact config matches standard forward."""
    import math
    from snlp.inference_idn import load_model, load_val_data, forward_sequential, forward_idn_batched

    device = torch.device('cuda')
    model = load_model('d32s_idn00625_npar24_s3', device=device)
    batches = load_val_data(device, 100_000, seq_len=2048)

    cfg_exact = HEApproxConfig.exact()
    L = model.config.n_layer
    n_par = 24
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    pw = PrecomputedWeights(model, par)

    # Test sequential
    idx = batches[0][:, :-1]
    logits_std = forward_sequential(model, idx)
    logits_he = forward_sequential_he(model, idx, cfg_exact)
    diff_seq = (logits_std - logits_he).abs().max().item()
    print(f"Sequential exact match: max_diff = {diff_seq:.6e}")

    # Test SNLP
    logits_snlp_std = forward_idn_batched(model, idx, seq_layers, pw, K=2, init='h0')
    logits_snlp_he = forward_idn_batched_he(model, idx, seq_layers, pw, cfg_exact, K=2, init='h0')
    diff_snlp = (logits_snlp_std - logits_snlp_he).abs().max().item()
    print(f"SNLP K=2 exact match: max_diff = {diff_snlp:.6e}")

    # PPL comparison
    tgt = batches[0][:, 1:]
    ppl_std = math.exp(F.cross_entropy(logits_std.reshape(-1, logits_std.size(-1)), tgt.reshape(-1)).item())
    ppl_he = math.exp(F.cross_entropy(logits_he.reshape(-1, logits_he.size(-1)), tgt.reshape(-1)).item())
    print(f"Sequential PPL: std={ppl_std:.4f}, he_exact={ppl_he:.4f}, diff={abs(ppl_std-ppl_he):.4f}")

    assert diff_seq < 0.01, f"Sequential mismatch too large: {diff_seq}"
    assert diff_snlp < 0.01, f"SNLP mismatch too large: {diff_snlp}"
    print("Sanity check PASSED!")


if __name__ == '__main__':
    _sanity_check()
