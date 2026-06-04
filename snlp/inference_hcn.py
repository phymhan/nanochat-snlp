"""
Evaluate mHC models: sequential PPL and mHC-Newton batched PPL.

The batched forward runs all parallel layers simultaneously by stacking weights
and HC matrices, exactly like _batched_forward in eval_jacobi_idn_new.py but
with HC width/depth connections around the attention and MLP.

Usage:
  CUDA_VISIBLE_DEVICES=4 NANOCHAT_BASE_DIR=cache PYTHONUNBUFFERED=1 \
    uv run python -m scripts.eval_mhc --model-tag d32s_mhc4_baseline --n-par 8 12 16
"""
import os, sys, time, json, math, argparse
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, sinkhorn_log, norm, apply_rotary_emb
from nanochat.common import COMPUTE_DTYPE


# ---------------------------------------------------------------------------
# Precomputed weights
# ---------------------------------------------------------------------------

class MHCPrecomputedWeights:
    """Cache stacked weights and HC matrices for batched mHC forward."""

    def __init__(self, model, par):
        blocks = [model.transformer.h[i] for i in par]
        dt = COMPUTE_DTYPE
        n_par = len(par)
        use_mhc = getattr(model.config, 'use_mhc', False)

        # Stacked attention weights (same as standard IDN batched)
        self.W_q_s = torch.stack([b.attn.c_q.weight for b in blocks]).to(dt)
        self.W_k_s = torch.stack([b.attn.c_k.weight for b in blocks]).to(dt)
        self.W_v_s = torch.stack([b.attn.c_v.weight for b in blocks]).to(dt)
        self.W_o_s = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dt)
        self.W_fc_s = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dt)
        self.W_proj_s = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dt)

        # VE gate weights
        self.ve_gate_weights = []
        for b in blocks:
            if b.attn.ve_gate is not None:
                self.ve_gate_weights.append(b.attn.ve_gate.weight.to(dt))
            else:
                self.ve_gate_weights.append(None)

        # Precomputed HC matrices (Sinkhorn + softmax, stacked)
        self.use_mhc = use_mhc
        if use_mhc:
            sk_iters = model.config.mhc_sinkhorn_iters
            sk_tau = model.config.mhc_sinkhorn_tau
            S = model.config.mhc_num_streams
            self.S = S

            H_res_attn_list, H_pre_attn_list, H_post_attn_list = [], [], []
            H_res_mlp_list, H_pre_mlp_list, H_post_mlp_list = [], [], []
            for b in blocks:
                H_res_attn_list.append(sinkhorn_log(b.hc_attn.H_res_logits, sk_iters, sk_tau))
                H_pre_attn_list.append(F.softmax(b.hc_attn.H_pre_logits, dim=-1))
                H_post_attn_list.append(F.softmax(b.hc_attn.H_post_logits, dim=-1))
                H_res_mlp_list.append(sinkhorn_log(b.hc_mlp.H_res_logits, sk_iters, sk_tau))
                H_pre_mlp_list.append(F.softmax(b.hc_mlp.H_pre_logits, dim=-1))
                H_post_mlp_list.append(F.softmax(b.hc_mlp.H_post_logits, dim=-1))

            self.H_res_attn = torch.stack(H_res_attn_list).to(dt)  # (n_par, S, S)
            self.H_pre_attn = torch.stack(H_pre_attn_list).to(dt)  # (n_par, S)
            self.H_post_attn = torch.stack(H_post_attn_list).to(dt)
            self.H_res_mlp = torch.stack(H_res_mlp_list).to(dt)
            self.H_pre_mlp = torch.stack(H_pre_mlp_list).to(dt)
            self.H_post_mlp = torch.stack(H_post_mlp_list).to(dt)

            # Block Jacobians for Newton correction: H^res_mlp @ H^res_attn
            self.J_blocks = torch.stack([
                (H_res_mlp_list[j] @ H_res_attn_list[j]) for j in range(n_par)
            ]).to(dt)  # (n_par, S, S)

        self.par = par
        self.n_par = n_par


# ---------------------------------------------------------------------------
# HC width/depth connection (batched over n_par layers)
# ---------------------------------------------------------------------------

def _hc_width(x, H_res, H_pre, S):
    """Batched HC width connection.

    x: (n_par, B*S, T, D) — stream-expanded input
    H_res: (n_par, S, S)
    H_pre: (n_par, S)
    Returns: branch_input (n_par, B, T, D), residuals_out (n_par, B, S, T, D)
    """
    n_par, BS, T, D = x.shape
    B = BS // S
    residuals = x.view(n_par, B, S, T, D)
    residuals_out = torch.einsum('lij,lbjkm->lbikm', H_res, residuals)
    branch_input = torch.einsum('lj,lbjkm->lbkm', H_pre, residuals)
    return branch_input, residuals_out


def _hc_depth(branch_output, residuals_out, H_post, S):
    """Batched HC depth connection.

    branch_output: (n_par, B, T, D)
    residuals_out: (n_par, B, S, T, D)
    H_post: (n_par, S)
    Returns: (n_par, B*S, T, D)
    """
    output = branch_output.unsqueeze(2) * H_post[:, None, :, None, None]  # (n_par, B, S, T, D)
    result = output + residuals_out
    n_par, B, S, T, D = result.shape
    return result.view(n_par, B * S, T, D)


# ---------------------------------------------------------------------------
# Batched forward for mHC models
# ---------------------------------------------------------------------------

def _batched_forward_mhc(all_h_prev, x0, cos_sin, ve, pw, model):
    """Batched forward for mHC models. All parallel layers run simultaneously.

    all_h_prev: (n_par, B*S, T, D) — stream-expanded hidden states
    x0: (B*S, T, D) — stream-expanded embedding
    Returns: (n_par, B*S, T, D)
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

    # no_x0_resid is always True for mHC
    all_x = all_h_prev  # (n_par, B*S, T, D)

    # --- Attention HC ---
    # Width connection: separate streams, mix, pool to branch input
    branch_in, res_out = _hc_width(all_x, pw.H_res_attn, pw.H_pre_attn, S)
    # branch_in: (n_par, B, T, D) — pooled single-stream input for attention

    # Attention on pooled input (same as standard _batched_forward but batch=B not B*S)
    bn = F.rms_norm(branch_in, (D,))

    all_q = torch.einsum('lbtd,lhd->lbth', bn, pw.W_q_s)
    all_k = torch.einsum('lbtd,lhd->lbth', bn, pw.W_k_s)
    all_v = torch.einsum('lbtd,lhd->lbth', bn, pw.W_v_s)

    all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

    # VE (uses original B, not expanded)
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_j.view(B, T, n_kv_head, head_dim)
            gate = 3 * torch.sigmoid(F.linear(bn[j, :, :, :12], pw.ve_gate_weights[j]))
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_r.permute(0, 2, 1, 3)

    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r, x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = F.rms_norm(_rope(all_q), (head_dim,)) * 1.2
    all_k = F.rms_norm(_rope(all_k), (head_dim,)) * 1.2

    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, T, head_dim).reshape(n_par, B, n_head, T, head_dim)

    q_s = all_q.reshape(n_par * B, n_head, T, head_dim)
    k_s = all_k.reshape(n_par * B, n_head, T, head_dim)
    v_s = all_v.reshape(n_par * B, n_head, T, head_dim)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

    attn_out = y_s.reshape(n_par, B, n_head, T, head_dim).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o_s)
    # attn_out: (n_par, B, T, D)

    # Depth connection: distribute attn output to streams, add to mixed residuals
    all_x = _hc_depth(attn_out, res_out, pw.H_post_attn, S)
    # all_x: (n_par, B*S, T, D)

    # --- MLP HC ---
    branch_in_mlp, res_out_mlp = _hc_width(all_x, pw.H_res_mlp, pw.H_pre_mlp, S)
    # branch_in_mlp: (n_par, B, T, D)

    h_n = F.rms_norm(branch_in_mlp, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp_out = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    # mlp_out: (n_par, B, T, D)

    # Depth connection for MLP
    all_x = _hc_depth(mlp_out, res_out_mlp, pw.H_post_mlp, S)
    # all_x: (n_par, B*S, T, D)

    return all_x


# ---------------------------------------------------------------------------
# Embed / logits with stream expansion/reduction
# ---------------------------------------------------------------------------

def _embed_mhc(model, idx):
    """Embed tokens. For mHC: also returns stream-expanded x0."""
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)
    if T > 1:
        gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    value_embeds = {}
    if not getattr(model.config, 'no_ve', False):
        for key in model.value_embeds:
            value_embeds[key] = model.value_embeds[key](idx).to(x.dtype)

    if model.config.use_mhc:
        S = model.config.mhc_num_streams
        D = x.shape[-1]
        x = x.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)

    return x, cos_sin, value_embeds


def _run_block_mhc(model, i, h_prev, x0, cos_sin, value_embeds):
    """Run a single block (for sequential prefix)."""
    if getattr(model.config, 'no_x0_resid', False):
        x_in = h_prev
    else:
        x_in = model.resid_lambdas[i] * h_prev + model.x0_lambdas[i] * x0
    ve = value_embeds.get(str(i))
    return model.transformer.h[i](x_in, ve, cos_sin, model.window_sizes[i], None)


def _logits_mhc(model, h):
    """Compute logits. For mHC: stream-reduce before lm_head."""
    n_layer = model.config.n_layer
    x_final = h[-1]
    backout_layer = n_layer // 2

    if model.config.use_mhc:
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

    x_final = norm(x_final)
    softcap = 15
    logits = model.lm_head(x_final)
    logits = logits[..., :model.config.vocab_size]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    return logits


# ---------------------------------------------------------------------------
# IDN / mHC-Newton correction
# ---------------------------------------------------------------------------

def _idn_correction(all_out, hs, h_init, n_par):
    """Standard identity Newton correction (J=I)."""
    h_old = [hs[j + 1].clone() for j in range(n_par)]
    h_corr = all_out[0]
    hs[1] = h_corr
    for j in range(1, n_par):
        h_corr = all_out[j] + (h_corr - h_old[j - 1])
        hs[j + 1] = h_corr


def _mhc_newton_correction(all_out, hs, h_init, n_par, pw):
    """mHC-Newton correction using J_block = H^res_mlp @ H^res_attn."""
    S = pw.S
    h_old = [hs[j + 1].clone() for j in range(n_par)]
    h_corr = all_out[0]
    hs[1] = h_corr
    for j in range(1, n_par):
        J = pw.J_blocks[j]  # (S, S)
        delta = h_corr - h_old[j - 1]
        BS, T_d, D_d = delta.shape
        B_eff = BS // S
        delta = delta.view(B_eff, S, T_d, D_d)
        delta = torch.einsum('ij,bjkl->bikl', J, delta)
        delta = delta.reshape(BS, T_d, D_d)
        h_corr = all_out[j] + delta
        hs[j + 1] = h_corr


# ---------------------------------------------------------------------------
# Chunkwise fused: fuse M layers into one mega-block with averaged HC matrices
# ---------------------------------------------------------------------------

class MHCChunkWeights:
    """Precomputed fused weights + averaged HC matrices for one chunk of layers."""

    def __init__(self, model, chunk_layers):
        dt = COMPUTE_DTYPE
        blocks = [model.transformer.h[i] for i in chunk_layers]
        n = len(chunk_layers)
        S = model.config.mhc_num_streams
        sk_iters = model.config.mhc_sinkhorn_iters
        sk_tau = model.config.mhc_sinkhorn_tau

        # Fused attention weights: concat QKV, sum O (same as standard fused)
        self.W_q = torch.cat([b.attn.c_q.weight for b in blocks], dim=0).to(dt)
        self.W_k = torch.cat([b.attn.c_k.weight for b in blocks], dim=0).to(dt)
        self.W_v = torch.cat([b.attn.c_v.weight for b in blocks], dim=0).to(dt)
        self.W_o = torch.cat([b.attn.c_proj.weight for b in blocks], dim=1).to(dt)
        self.W_fc = torch.cat([b.mlp.c_fc.weight for b in blocks], dim=0).to(dt)
        self.W_proj = torch.cat([b.mlp.c_proj.weight for b in blocks], dim=1).to(dt)

        # Averaged HC matrices (element-wise mean of Sinkhorn/softmax outputs)
        H_res_attn = [sinkhorn_log(b.hc_attn.H_res_logits, sk_iters, sk_tau) for b in blocks]
        H_res_mlp = [sinkhorn_log(b.hc_mlp.H_res_logits, sk_iters, sk_tau) for b in blocks]
        self.avg_H_res_attn = torch.stack(H_res_attn).mean(0).to(dt)  # (S, S)
        self.avg_H_pre_attn = torch.stack([F.softmax(b.hc_attn.H_pre_logits, -1) for b in blocks]).mean(0).to(dt)
        self.avg_H_post_attn = torch.stack([F.softmax(b.hc_attn.H_post_logits, -1) for b in blocks]).mean(0).to(dt)
        self.avg_H_res_mlp = torch.stack(H_res_mlp).mean(0).to(dt)
        self.avg_H_pre_mlp = torch.stack([F.softmax(b.hc_mlp.H_pre_logits, -1) for b in blocks]).mean(0).to(dt)
        self.avg_H_post_mlp = torch.stack([F.softmax(b.hc_mlp.H_post_logits, -1) for b in blocks]).mean(0).to(dt)

        # Chunk Jacobian for inter-chunk Newton correction
        self.J_chunk = (self.avg_H_res_mlp @ self.avg_H_res_attn).to(dt)  # (S, S)

        self.layers = chunk_layers
        self.n = n
        self.S = S


def _batched_chunks_forward_mhc(all_h_in, cos_sin, chunk_weights_list, model):
    """Batched forward for C equal-size fused mHC chunks simultaneously.

    Stacks all chunk weights and averaged HC matrices into (C, ...) tensors
    and processes in one batched call. All chunks must have the same size.

    Args:
        all_h_in: (C, B*S, T, D) — per-chunk stream-expanded inputs
        chunk_weights_list: list of C MHCChunkWeights (all same size)
    Returns:
        all_out: (C, B*S, T, D) — per-chunk outputs
    """
    C = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n
    S = chunk_weights_list[0].S
    BS, T, D = all_h_in.shape[1:]
    B = BS // S
    n_head = model.config.n_head
    heads_per_chunk = chunk_size * n_head
    head_dim = D // n_head

    # Stack chunk weights
    W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
    W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
    W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
    W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
    W_fc = torch.stack([cw.W_fc for cw in chunk_weights_list])
    W_proj = torch.stack([cw.W_proj for cw in chunk_weights_list])

    # Stack averaged HC matrices: (C, S, S) and (C, S)
    H_res_attn = torch.stack([cw.avg_H_res_attn for cw in chunk_weights_list])
    H_pre_attn = torch.stack([cw.avg_H_pre_attn for cw in chunk_weights_list])
    H_post_attn = torch.stack([cw.avg_H_post_attn for cw in chunk_weights_list])
    H_res_mlp = torch.stack([cw.avg_H_res_mlp for cw in chunk_weights_list])
    H_pre_mlp = torch.stack([cw.avg_H_pre_mlp for cw in chunk_weights_list])
    H_post_mlp = torch.stack([cw.avg_H_post_mlp for cw in chunk_weights_list])

    # --- Attention HC ---
    branch_in, res_out = _hc_width(all_h_in, H_res_attn, H_pre_attn, S)
    # branch_in: (C, B, T, D), res_out: (C, B, S, T, D)

    bn = F.rms_norm(branch_in, (D,))
    all_q = torch.einsum('cbtd,chd->cbth', bn, W_q)
    all_k = torch.einsum('cbtd,chd->cbth', bn, W_k)
    all_v = torch.einsum('cbtd,chd->cbth', bn, W_v)

    all_q = all_q.view(C, B, T, heads_per_chunk, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(C, B, T, heads_per_chunk, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(C, B, T, heads_per_chunk, head_dim).permute(0, 1, 3, 2, 4)

    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r, x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = F.rms_norm(_rope(all_q), (head_dim,)) * 1.2
    all_k = F.rms_norm(_rope(all_k), (head_dim,)) * 1.2

    q_s = all_q.reshape(C * B, heads_per_chunk, T, head_dim)
    k_s = all_k.reshape(C * B, heads_per_chunk, T, head_dim)
    v_s = all_v.reshape(C * B, heads_per_chunk, T, head_dim)
    y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

    y = y_s.reshape(C, B, heads_per_chunk, T, head_dim).permute(0, 1, 3, 2, 4)
    y = y.reshape(C, B, T, heads_per_chunk * head_dim)
    attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)  # (C, B, T, D)

    all_x = _hc_depth(attn_out, res_out, H_post_attn, S)  # (C, B*S, T, D)

    # --- MLP HC ---
    branch_in_mlp, res_out_mlp = _hc_width(all_x, H_res_mlp, H_pre_mlp, S)

    h_n = F.rms_norm(branch_in_mlp, (D,))
    hf = torch.einsum('cbtd,cid->cbti', h_n, W_fc)
    mlp_out = torch.einsum('cbti,cdi->cbtd', F.relu(hf).square(), W_proj)  # (C, B, T, D)

    return _hc_depth(mlp_out, res_out_mlp, H_post_mlp, S)  # (C, B*S, T, D)


@torch.inference_mode()
def forward_hcn_chunkwise(model, idx, seq_layers, n_par, chunk_weights_list, pw, K=1, init='h0'):
    """Chunkwise fused with mHC-Newton correction between chunks.

    All C chunks run simultaneously via _batched_chunks_forward_mhc.
    Between chunks, Newton correction uses the chunk's J = avg_H^res_mlp @ avg_H^res_attn.
    """
    L = model.config.n_layer
    S = model.config.mhc_num_streams
    C = len(chunk_weights_list)

    x0, cos_sin, ve = _embed_mhc(model, idx)

    # Sequential prefix
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_mhc(model, i, x, x0, cos_sin, ve)
        h[i] = x
    h_init = x

    # Initialize: one state per chunk
    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(C, -1, -1, -1)
        chunk_outs = _batched_chunks_forward_mhc(all_h0, cos_sin, chunk_weights_list, model)
        chunk_hs = [h_init] + [chunk_outs[c] for c in range(C)]
    else:
        chunk_hs = [h_init] + [h_init.clone() for _ in range(C)]

    # K iterations
    for _k in range(K):
        all_hp = torch.stack([chunk_hs[c] for c in range(C)])
        chunk_outs = _batched_chunks_forward_mhc(all_hp, cos_sin, chunk_weights_list, model)

        # Newton correction between chunks using chunk Jacobians
        h_old = [chunk_hs[c + 1].clone() for c in range(C)]
        h_corr = chunk_outs[0]
        chunk_hs[1] = h_corr
        for c in range(1, C):
            J = chunk_weights_list[c].J_chunk
            delta = h_corr - h_old[c - 1]
            BS_d, T_d, D_d = delta.shape
            B_eff = BS_d // S
            delta = delta.view(B_eff, S, T_d, D_d)
            delta = torch.einsum('ij,bjkl->bikl', J, delta)
            delta = delta.reshape(BS_d, T_d, D_d)
            h_corr = chunk_outs[c] + delta
            chunk_hs[c + 1] = h_corr

    # Map chunk outputs to per-layer h
    for c, cw in enumerate(chunk_weights_list):
        for li in cw.layers:
            h[li] = chunk_hs[c + 1]

    return _logits_mhc(model, h)


def _batched_forward_mhc_kv(all_h_prev, x0, cos_sin, ve, pw, model,
                             cached_k=None, cached_v=None):
    """Batched mHC forward with KV cache for AR generation.

    all_h_prev: (n_par, B*S, T, D) — stream-expanded hidden states
    x0: (B*S, T, D) — stream-expanded embedding (unused, mHC has no_x0_resid)
    cached_k/v: (n_par, B, S_cache, n_kv_head, head_dim)
    Returns: (output, new_k, new_v)
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
    dt = all_h_prev.dtype

    all_x = all_h_prev

    # --- Attention HC width ---
    branch_in, res_out = _hc_width(all_x, pw.H_res_attn, pw.H_pre_attn, S)

    bn = F.rms_norm(branch_in, (D,))
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
            gate = 3 * torch.sigmoid(F.linear(bn[j, :, :, :12], pw.ve_gate_weights[j]))
            all_v[j] = all_v[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_r.permute(0, 2, 1, 3)

    cos, sin = cos_sin
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r, x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = _rope(all_q)
    all_k = _rope(all_k)
    all_q = F.rms_norm(all_q, (head_dim,)) * 1.2
    all_k = F.rms_norm(all_k, (head_dim,)) * 1.2

    new_k = all_k.clone()
    new_v = all_v.clone()

    if cached_k is not None and cached_k.shape[2] > 0:
        ck = cached_k.transpose(2, 3)
        cv = cached_v.transpose(2, 3)
        all_k = torch.cat([ck, all_k], dim=-2)
        all_v = torch.cat([cv, all_v], dim=-2)

    S_plus_T = all_k.shape[-2]
    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, S_plus_T, head_dim).reshape(n_par, B, n_head, S_plus_T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(n_par, B, n_kv_head, n_kv_groups, S_plus_T, head_dim).reshape(n_par, B, n_head, S_plus_T, head_dim)

    scaling = head_dim ** -0.5
    attn_w = torch.matmul(all_q, all_k.transpose(-2, -1)) * scaling
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(dt)
    attn_out = torch.matmul(attn_w, all_v)

    attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o_s)

    # Attention HC depth
    all_x = _hc_depth(attn_out, res_out, pw.H_post_attn, S)

    # --- MLP HC ---
    branch_in_mlp, res_out_mlp = _hc_width(all_x, pw.H_res_mlp, pw.H_pre_mlp, S)
    h_n = F.rms_norm(branch_in_mlp, (D,))
    hf = torch.einsum('lbtd,lid->lbti', h_n, pw.W_fc_s)
    mlp_out = torch.einsum('lbti,ldi->lbtd', F.relu(hf).square(), pw.W_proj_s)
    all_x = _hc_depth(mlp_out, res_out_mlp, pw.H_post_mlp, S)

    return all_x, new_k, new_v


def _update_kv_cache_mhc(kv, par, h_step, h_init, x0, cos_sin, ve, pw, model):
    """Update KV cache for mHC parallel layers from corrected stream-expanded states.

    For each layer: HC width pool → K,V projection → RoPE → cache insert.
    """
    S = pw.S
    D = model.config.n_embd
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = D // n_head
    pos = kv.get_pos()
    cos, sin = cos_sin

    for j, li in enumerate(par):
        h_in = h_init if j == 0 else h_step.get(li - 1, h_init)
        BS = h_in.shape[0]
        B = BS // S
        residuals = h_in.view(1, B, S, 1, D)
        H_pre = pw.H_pre_attn[j:j+1]
        branch = torch.einsum('lj,lbjkm->lbkm', H_pre, residuals)[0]

        bn = F.rms_norm(branch, (D,))
        k = F.linear(bn, pw.W_k_s[j]).view(B, 1, n_kv_head, head_dim)
        v = F.linear(bn, pw.W_v_s[j]).view(B, 1, n_kv_head, head_dim)

        ve_j = ve.get(str(li))
        if ve_j is not None and pw.ve_gate_weights[j] is not None:
            ve_r = ve_j.view(B, 1, n_kv_head, head_dim)
            gate = 3 * torch.sigmoid(F.linear(bn[:, :, :12], pw.ve_gate_weights[j]))
            v = v + gate.unsqueeze(-1) * ve_r

        from nanochat.gpt import apply_rotary_emb
        k = apply_rotary_emb(k, cos, sin)
        k = F.rms_norm(k.view(B, 1, n_kv_head, head_dim), (head_dim,)) * 1.2

        kv.k_cache[li, :, pos:pos+1, :, :] = k
        kv.v_cache[li, :, pos:pos+1, :, :] = v


@torch.inference_mode()
def ar_generate_mhc(model, prompt_ids, pw, seq_layers, max_new=16, K=1, init='h0'):
    """AR generation with mHC-Newton + per-layer KV cache.

    Uses HC-aware batched forward for parallel layers.
    Sequential prefix handled by Block.forward() which includes HC.
    """
    L = model.config.n_layer
    par = pw.par
    n_par = pw.n_par
    S = pw.S
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // model.config.n_head
    dtype = COMPUTE_DTYPE

    from nanochat.engine import KVCache
    dev = model.transformer.wte.weight.device
    kv = KVCache(1, n_kv_head, len(prompt_ids) + max_new + 1, head_dim, L, dev, dtype)

    ids = torch.tensor([prompt_ids], device=model.transformer.wte.weight.device)

    # Prefill: model.forward handles HC + KV cache
    logits = model.forward(ids, kv_cache=kv)

    # Warm-start: run model without KV cache to get per-layer hidden states
    prev_h = None
    if init == 'warm':
        x0_pf, cs_pf, ve_pf = _embed_mhc(model, ids)
        prev_h = [None] * L
        x = x0_pf
        for i in range(L):
            x = _run_block_mhc(model, i, x, x0_pf, cs_pf, ve_pf)
            prev_h[i] = x[:, -1:, :].clone()

    gen = []
    dev = model.transformer.wte.weight.device
    for step in range(max_new):
        tok = logits[0, -1].argmax().item()
        gen.append(tok)
        ids_step = torch.tensor([[tok]], device=dev)

        # Embed + stream expand
        x0_single = model.transformer.wte(ids_step).to(dtype)
        x0_single = norm(x0_single)
        if kv.prev_embedding is not None:
            gate = model.smear_lambda.to(dtype) * torch.sigmoid(model.smear_gate(x0_single[:, :, :24]))
            x0_single = x0_single + gate * kv.prev_embedding

        pos = kv.get_pos()
        cos_sin = model.cos[:, pos:pos+1], model.sin[:, pos:pos+1]
        if getattr(model.config, 'no_ve', False):
            ve = {}
        else:
            ve = {k_: model.value_embeds[k_](ids_step).to(dtype) for k_ in model.value_embeds}

        # Stream expand for mHC blocks
        D = x0_single.shape[-1]
        x0_exp = x0_single.unsqueeze(1).expand(1, S, 1, D).reshape(S, 1, D)

        # Sequential prefix with KV cache (Block.forward handles HC)
        x = x0_exp
        h_step = {}
        for i in range(seq_layers):
            x_in = x  # no_x0_resid
            x = model.transformer.h[i](x_in, ve.get(str(i)), cos_sin, model.window_sizes[i], kv)
            h_step[i] = x
        h_init = x

        # Extract cached K,V for parallel layers
        cached_k = kv.k_cache[par, :, :pos, :, :]
        cached_v = kv.v_cache[par, :, :pos, :, :]

        # Initialize parallel layers
        if init == 'warm' and prev_h is not None:
            hs = [h_init] + [prev_h[par[j]].clone() for j in range(n_par)]
        elif init == 'batch_fwd':
            all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
            all_init, _, _ = _batched_forward_mhc_kv(
                all_h0, x0_exp, cos_sin, ve, pw, model,
                cached_k=cached_k, cached_v=cached_v)
            hs = [h_init] + [all_init[j] for j in range(n_par)]
        else:
            hs = [h_init] + [h_init.clone() for _ in range(n_par)]

        for _k in range(K):
            all_hp = torch.stack([hs[j] for j in range(n_par)])
            all_out, _, _ = _batched_forward_mhc_kv(
                all_hp, x0_exp, cos_sin, ve, pw, model,
                cached_k=cached_k, cached_v=cached_v)
            _mhc_newton_correction(list(all_out), hs, h_init, n_par, pw)

        for j, li in enumerate(par):
            h_step[li] = hs[j + 1]

        # Update KV cache from corrected states
        _update_kv_cache_mhc(kv, par, h_step, h_init, x0_exp, cos_sin, ve, pw, model)

        kv.cache_seqlens += 1
        kv.prev_embedding = norm(model.transformer.wte(ids_step).to(dtype))

        if init == 'warm':
            if prev_h is None:
                prev_h = [None] * L
            for i in range(L):
                if i in h_step:
                    prev_h[i] = h_step[i].clone()

        # Logits: stream-reduce
        x_final = h_step.get(L - 1, hs[n_par])
        BS_f = x_final.shape[0]
        B_f = BS_f // S
        x_final = x_final.view(B_f, S, 1, D).sum(dim=1)
        bo = L // 2
        if bo in h_step:
            x_back = h_step[bo].view(B_f, S, 1, D).sum(dim=1)
            x_final = x_final - model.backout_lambda.to(dtype) * x_back
        x_final = norm(x_final)
        logits = model.lm_head(x_final)[..., :model.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)

    return gen


def get_chunk_configs(n_par):
    """Generate chunk count set: {1, 2, 4, n_par/4, n_par/2}, deduplicated, divisible only."""
    counts = set()
    for n in [1, 2, 4]:
        if n_par % n == 0:
            counts.add(n)
    for div in [4, 2]:
        c = n_par // div
        if c >= 1 and n_par % c == 0:
            counts.add(c)
    return sorted(counts)


# ---------------------------------------------------------------------------
# High-level forward functions
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, idx):
    return model.forward(idx)


@torch.inference_mode()
def forward_hcn_batched(model, idx, seq_layers, pw, K=1, init='h0'):
    """Batched Newton forward: runs parallel layers simultaneously.

    For mHC models: uses _batched_forward_mhc and mHC-Newton correction.
    For standard models: uses _batched_forward_mhc still works (HC matrices are identity-like).
    """
    L = model.config.n_layer
    n_par = pw.n_par
    use_mhc = pw.use_mhc

    x0, cos_sin, ve = _embed_mhc(model, idx)

    # Sequential prefix
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_mhc(model, i, x, x0, cos_sin, ve)
        h[i] = x
    h_init = x

    # Initialize
    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = _batched_forward_mhc(all_h0, x0, cos_sin, ve, pw, model)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:  # h0
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    # K iterations
    for _k in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = _batched_forward_mhc(all_hp, x0, cos_sin, ve, pw, model)
        if use_mhc:
            _mhc_newton_correction(all_out, hs, h_init, n_par, pw)
        else:
            _idn_correction(all_out, hs, h_init, n_par)

    for j, li in enumerate(pw.par):
        h[li] = hs[j + 1]
    return _logits_mhc(model, h)


# ---------------------------------------------------------------------------
# Data loading + PPL eval
# ---------------------------------------------------------------------------

def load_val_data(device, max_tokens=1_000_000, seq_len=128):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR')
    import pyarrow.parquet as pq
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    table = pq.read_table(
        os.path.join(base_dir, 'base_data_climbmix', 'shard_06542.parquet'),
        columns=['text']
    )
    all_tokens = []
    need_tokens = max_tokens + seq_len + 1
    for text in table.column('text').to_pylist():
        all_tokens.extend(tokenizer.encode(text, prepend=bos))
        if len(all_tokens) >= need_tokens:
            break
    batches = []
    offset = 0
    while offset + seq_len + 1 <= len(all_tokens) and len(batches) * seq_len < max_tokens:
        batches.append(torch.tensor([all_tokens[offset:offset + seq_len + 1]], dtype=torch.long, device=device))
        offset += seq_len
    return batches


@torch.inference_mode()
def eval_ppl(model, batches, forward_fn, **kwargs):
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        idx = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = forward_fn(model, idx, **kwargs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += tgt.numel()
    avg_bpb = total_loss / total_tokens / math.log(2)
    ppl = math.exp(total_loss / total_tokens)
    return ppl, avg_bpb


@torch.inference_mode()
def eval_ppl_with_metrics(model, batches, forward_fn, **kwargs):
    """PPL + top-1 + cos_sim in one pass."""
    losses, top1s, cossims = [], [], []
    for batch in batches:
        idx = batch[:, :-1]
        tgt = batch[:, 1:]
        logits_seq = forward_sequential(model, idx)
        logits_method = forward_fn(model, idx, **kwargs)
        losses.append(F.cross_entropy(logits_method.view(-1, logits_method.size(-1)), tgt.reshape(-1)).item())
        ref = logits_seq.reshape(-1, logits_seq.size(-1)).float()
        test = logits_method.reshape(-1, logits_method.size(-1)).float()
        top1s.append((ref.argmax(-1) == test.argmax(-1)).float().mean().item())
        cos = F.cosine_similarity(ref, test, dim=-1)
        cossims.append(cos[~cos.isnan()].mean().item() if not cos.isnan().all() else 0.0)
    ppl = math.exp(min(sum(losses) / len(losses), 20))
    avg_bpb = sum(losses) / len(losses) / math.log(2)
    return ppl, avg_bpb, sum(top1s) / len(top1s), sum(cossims) / len(cossims)


def bench(fn, device, n_warm=50, n_run=200):
    """Benchmark single-batch latency (ms). 50 warmup + 200 measured runs."""
    for _ in range(n_warm):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', type=str, required=True)
    parser.add_argument('--n-par', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--max-tokens', type=int, default=1_000_000)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--cache-hc', action='store_true', help='Cache HC matrices for faster inference')
    args = parser.parse_args()

    device = torch.device('cuda')
    from snlp.inference_idn import load_model
    model = load_model(args.model_tag, device=device, cache_hc=args.cache_hc)
    L = model.config.n_layer
    use_mhc = getattr(model.config, 'use_mhc', False)
    S = model.config.mhc_num_streams if use_mhc else 1
    print(f"Model: {args.model_tag}, L={L}, use_mhc={use_mhc}, streams={S}")

    batches = load_val_data(device, max_tokens=args.max_tokens)
    print(f"Loaded {len(batches)} validation batches ({len(batches) * 128} tokens)")

    results = {'model': args.model_tag, 'use_mhc': use_mhc, 'n_layer': L}

    # Sequential PPL
    print("\n--- Sequential ---")
    seq_ppl, seq_bpb = eval_ppl(model, batches, forward_sequential)
    print(f"  PPL={seq_ppl:.2f}, BPB={seq_bpb:.6f}")
    results['sequential'] = {'ppl': seq_ppl, 'bpb': seq_bpb}

    # Sequential timing (single batch, BS=1)
    bench_idx = batches[0][:, :-1]
    seq_ms = bench(lambda: forward_sequential(model, bench_idx), device, n_warm=50, n_run=200)
    print(f"  Timing: {seq_ms:.2f} ms")
    results['sequential']['ms'] = seq_ms

    results['methods'] = {}
    for n_par in args.n_par:
        if n_par >= L:
            print(f"\nSkipping n_par={n_par} (>= L={L})")
            continue
        seq_layers = L - n_par
        par = list(range(seq_layers, L))
        pw = MHCPrecomputedWeights(model, par)
        print(f"\n{'='*60}")
        print(f"n_par={n_par} (seq_layers={seq_layers})")
        print(f"{'='*60}")
        npar_key = str(n_par)
        results['methods'][npar_key] = {}

        # --- Newton batched (per-layer, no fusion) ---
        results['methods'][npar_key]['newton_batched'] = {}
        for init in ['h0', 'batch_fwd']:
            results['methods'][npar_key]['newton_batched'][init] = {}
            for K in args.k_values:
                ppl, bpb, top1, cos_sim = eval_ppl_with_metrics(
                    model, batches, forward_hcn_batched,
                    seq_layers=seq_layers, pw=pw, K=K, init=init
                )
                label = "mHC-Newton" if use_mhc else "IDN"
                _init, _K = init, K
                ms = bench(
                    lambda: forward_hcn_batched(model, bench_idx, seq_layers=seq_layers, pw=pw, K=_K, init=_init),
                    device, n_warm=50, n_run=200
                )
                print(f"  {label}_batched K={K} init={init}: PPL={ppl:.2f} ({ms:.2f}ms, {seq_ms/ms:.2f}x) top1={top1:.3f} cos={cos_sim:.4f}")
                entry = {'ppl': ppl, 'bpb': bpb, 'ms': ms, 'speedup': seq_ms / ms, 'top1': top1, 'cos_sim': cos_sim}
                results['methods'][npar_key]['newton_batched'][init][str(K)] = entry

        # --- Chunkwise fused (ChunkB_NxFM) ---
        chunk_counts = get_chunk_configs(n_par)
        for n_chunks in chunk_counts:
            chunk_size = n_par // n_chunks
            method_name = f'ChunkB_{n_chunks}xF{chunk_size}'
            print(f"\n  --- {method_name} ---")

            # Build chunk weights
            chunk_weights_list = []
            for c in range(n_chunks):
                start = seq_layers + c * chunk_size
                chunk_layers = list(range(start, start + chunk_size))
                chunk_weights_list.append(MHCChunkWeights(model, chunk_layers))

            results['methods'][npar_key][method_name] = {}
            for init in ['h0', 'batch_fwd']:
                results['methods'][npar_key][method_name][init] = {}
                for K in args.k_values:
                    ppl, bpb, top1, cos_sim = eval_ppl_with_metrics(
                        model, batches, forward_hcn_chunkwise,
                        seq_layers=seq_layers, n_par=n_par,
                        chunk_weights_list=chunk_weights_list, pw=pw, K=K, init=init
                    )
                    _init, _K, _cwl = init, K, chunk_weights_list
                    ms = bench(
                        lambda: forward_hcn_chunkwise(model, bench_idx, seq_layers=seq_layers, n_par=n_par,
                                                      chunk_weights_list=_cwl, pw=pw, K=_K, init=_init),
                        device, n_warm=50, n_run=200
                    )
                    print(f"  {method_name} K={K} init={init}: PPL={ppl:.2f} ({ms:.2f}ms, {seq_ms/ms:.2f}x) top1={top1:.3f} cos={cos_sim:.4f}")
                    entry = {'ppl': ppl, 'bpb': bpb, 'ms': ms, 'speedup': seq_ms / ms, 'top1': top1, 'cos_sim': cos_sim}
                    results['methods'][npar_key][method_name][init][str(K)] = entry

    # Save
    outpath = args.output or f'cache/eval_logs/mhc_{args.model_tag}.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
