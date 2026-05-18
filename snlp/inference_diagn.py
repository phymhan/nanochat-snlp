"""
Evaluate diagonal Newton (quasi-DEER) layer-parallel inference with associative scan.

Implements the Newton correction from arXiv:2508.18413 Eq. (2):
    s_t^{new} = f_t(s_{t-1}^{old}) + J_t * (s_{t-1}^{new} - s_{t-1}^{old})

with J approximated by diag(J) estimated via Hutchinson:
    diag(J) ~ z * (J*z)  where z ~ Rademacher

The diagonal linear recurrence is solved via parallel associative scan.

Strategies compared:
  1. Sequential:      standard forward (ground truth)
  2. IDN (loop):      identity Newton via prefix-sum (J=I baseline)
  3. Diag scan (FD):  diagonal Newton via finite-difference Hutchinson + scan
  4. Diag scan (VJP): diagonal Newton via backward-mode AD Hutchinson + scan
  5. Diag loop (VJP): diagonal Newton via VJP + sequential forward substitution

Usage:
    python -m scripts.eval_jacobi_diag --model-tag d32_idn05_npar8 --n-par 8
    python -m scripts.eval_jacobi_diag --model-tag d32_idn05_npar8 --elk-k 0.1
    python -m scripts.eval_jacobi_diag --model-tag d32_diag05_npar8_v4 --jvp vjp --n-probes 4
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb
from nanochat.jacobi_forward import _embed, _run_block, _logits


# ---------------------------------------------------------------------------
# Model loading and data
# ---------------------------------------------------------------------------

from snlp.inference_idn import load_model  # noqa: E402


def load_batches(model, device, seq_len=128, batch_size=1, n_batches=8):
    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')
    try:
        import pyarrow.parquet as pq
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        bos = tokenizer.get_bos_token_id()
        parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
        table = pq.read_table(os.path.join(data_dir, parquet_files[-1]), columns=['text'])
        all_tokens = []
        for text in table.column('text').to_pylist():
            all_tokens.extend(tokenizer.encode(text, prepend=bos))
            if len(all_tokens) >= (seq_len + 1) * batch_size * n_batches * 2:
                break
        batches = []
        offset = 0
        for _ in range(n_batches):
            seqs = []
            for _ in range(batch_size):
                seq = all_tokens[offset:offset + seq_len + 1]
                if len(seq) < seq_len + 1:
                    break
                seqs.append(seq)
                offset += seq_len
            if len(seqs) < batch_size:
                break
            batches.append(torch.tensor(seqs, dtype=torch.long, device=device))
        return batches
    except Exception as e:
        print(f"  Dataset load failed ({e}), using random tokens")
        vocab = model.config.vocab_size
        return [torch.randint(0, vocab, (batch_size, seq_len + 1), device=device)
                for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Associative scan for diagonal linear recurrence
# ---------------------------------------------------------------------------

def associative_scan_diagonal(a, b):
    """
    Parallel prefix scan for the diagonal linear recurrence:
        h_l = a_l * h_{l-1} + b_l,   l = 0, ..., L-1

    Returns a_scan, b_scan such that:
        h_l = a_scan[l] * h_0 + b_scan[l]
    """
    num_layers = a.shape[0]
    if num_layers <= 1:
        return a, b
    a_scan = a.clone()
    b_scan = b.clone()
    num_steps = int(math.ceil(math.log2(num_layers)))
    for depth in range(num_steps):
        stride = 1 << depth
        idx = torch.arange(stride, num_layers, device=a.device)
        src = idx - stride
        new_a = a_scan[idx] * a_scan[src]
        new_b = a_scan[idx] * b_scan[src] + b_scan[idx]
        a_scan = a_scan.clone()
        b_scan = b_scan.clone()
        a_scan[idx] = new_a
        b_scan[idx] = new_b
    return a_scan, b_scan


# ---------------------------------------------------------------------------
# Batched block forward (differentiable wrt all_h_prev for VJP)
# ---------------------------------------------------------------------------

def _batched_block_forward(all_h_prev, x0, cos_sin, ve, par, model, use_math_sdpa=False):
    """
    Batched forward for n_par parallel nanochat blocks.
    Differentiable wrt all_h_prev (needed for VJP/JVP diagonal estimation).

    Args:
        all_h_prev: (n_par, B, T, D) - per-layer input hidden states
        x0: (B, T, D) - original embeddings
        cos_sin: tuple (cos, sin) for RoPE
        ve: dict of value embeddings from _embed()
        par: list of layer indices for parallel layers
        model: GPT model
        use_math_sdpa: if True, force math SDPA backend (needed for JVP/forward-mode AD,
            since flash SDPA backward has no second-order derivative)

    Returns:
        all_out: (n_par, B, T, D) - per-layer block outputs
    """
    n_par = len(par)
    n_head = model.config.n_head
    n_kv_head = model.config.n_kv_head
    head_dim = model.config.n_embd // n_head
    n_kv_groups = n_head // n_kv_head
    B, T, D = x0.shape
    cos, sin = cos_sin
    blocks = [model.transformer.h[i] for i in par]

    # Stack and cast weights to compute dtype (model stores float32, einsum needs matching dtypes)
    dtype = all_h_prev.dtype
    W_q = torch.stack([b.attn.c_q.weight for b in blocks]).to(dtype)
    W_k = torch.stack([b.attn.c_k.weight for b in blocks]).to(dtype)
    W_v = torch.stack([b.attn.c_v.weight for b in blocks]).to(dtype)
    W_o = torch.stack([b.attn.c_proj.weight for b in blocks]).to(dtype)
    W_fc = torch.stack([b.mlp.c_fc.weight for b in blocks]).to(dtype)
    W_proj = torch.stack([b.mlp.c_proj.weight for b in blocks]).to(dtype)

    # Per-layer input blending: x_in[j] = resid_lambda[j] * h_prev[j] + x0_lambda[j] * x0
    # Cast lambdas to compute dtype to match the per-layer loop path (avoids float32
    # promotion from stacked float32 lambdas × bf16 hidden states).
    resid_lambdas = torch.stack([model.resid_lambdas[i] for i in par]).to(dtype).view(n_par, 1, 1, 1)
    x0_lambdas = torch.stack([model.x0_lambdas[i] for i in par]).to(dtype).view(n_par, 1, 1, 1)
    all_x_in = resid_lambdas * all_h_prev + x0_lambdas * x0.unsqueeze(0)

    # RMS norm (no learnable weight in nanochat)
    all_x_normed = F.rms_norm(all_x_in, (D,))

    # QKV projections via einsum
    all_q = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_q)
    all_k = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_k)
    all_v = torch.einsum('lbtd,lhd->lbth', all_x_normed, W_v)

    # Reshape: (n_par, B, T, heads*hd) -> (n_par, B, H, T, hd)
    all_q = all_q.view(n_par, B, T, n_head, head_dim).permute(0, 1, 3, 2, 4)
    all_k = all_k.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)
    all_v = all_v.view(n_par, B, T, n_kv_head, head_dim).permute(0, 1, 3, 2, 4)

    # Value embeddings (ResFormer) - use list+stack to avoid in-place autograd issues
    # Must use normed input for gate (matching model: attn receives norm(x_in))
    v_list = list(all_v.unbind(0))
    for j, li in enumerate(par):
        ve_j = ve.get(str(li))
        if ve_j is not None and blocks[j].attn.ve_gate is not None:
            ve_j_r = ve_j.view(B, T, n_kv_head, head_dim)
            gate = 3 * torch.sigmoid(
                F.linear(all_x_normed[j, :, :, :12], blocks[j].attn.ve_gate.weight.to(dtype))
            )
            v_list[j] = v_list[j] + gate.unsqueeze(-1).transpose(1, 2) * ve_j_r.permute(0, 2, 1, 3)
    all_v = torch.stack(v_list)

    # RoPE
    cos_r = cos.unsqueeze(0).permute(0, 1, 3, 2, 4)  # (1, 1, 1, T, hd//2)
    sin_r = sin.unsqueeze(0).permute(0, 1, 3, 2, 4)

    def _rope(x):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos_r + x2 * sin_r,
                          x1 * (-sin_r) + x2 * cos_r], dim=-1)

    all_q = _rope(all_q)
    all_k = _rope(all_k)

    # QK norm
    all_q = F.rms_norm(all_q, (head_dim,)) * 1.2
    all_k = F.rms_norm(all_k, (head_dim,)) * 1.2

    # GQA expand
    if n_kv_groups > 1:
        all_k = all_k[:, :, :, None, :, :].expand(
            n_par, B, n_kv_head, n_kv_groups, T, head_dim
        ).reshape(n_par, B, n_head, T, head_dim)
        all_v = all_v[:, :, :, None, :, :].expand(
            n_par, B, n_kv_head, n_kv_groups, T, head_dim
        ).reshape(n_par, B, n_head, T, head_dim)

    # SDPA: (n_par*B, H, T, hd) -- each layer is a separate batch element
    q_sdpa = all_q.reshape(n_par * B, n_head, T, head_dim)
    k_sdpa = all_k.reshape(n_par * B, n_head, T, head_dim)
    v_sdpa = all_v.reshape(n_par * B, n_head, T, head_dim)

    windows = [model.window_sizes[i] for i in par]
    all_full = all(w[0] < 0 or w[0] >= T for w in windows)

    # Math SDPA backend is needed for JVP (forward-mode AD) because flash SDPA's
    # backward doesn't have a second-order derivative. Math SDPA uses explicit
    # matmul+softmax which supports all AD modes.
    sdpa_ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                if use_math_sdpa else torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                     torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                     torch.nn.attention.SDPBackend.MATH]))

    with sdpa_ctx:
        if all_full:
            y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
        else:
            row_idx = torch.arange(T, device=all_h_prev.device).unsqueeze(1)
            col_idx = torch.arange(T, device=all_h_prev.device).unsqueeze(0)
            causal = col_idx <= row_idx
            masks = []
            for j in range(n_par):
                w = windows[j][0]
                m = causal if (w < 0 or w >= T) else (causal & ((row_idx - col_idx) <= w))
                masks.append(m.unsqueeze(0).expand(B, -1, -1))
            attn_mask = torch.cat(masks, dim=0).unsqueeze(1)
            y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

    # Reshape back: (n_par*B, H, T, hd) -> (n_par, B, T, D)
    attn_out = y_sdpa.reshape(n_par, B, n_head, T, head_dim)
    attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)

    # O projection
    attn_out = torch.einsum('lbtd,lod->lbto', attn_out, W_o)

    # First residual
    all_hidden = all_x_in + attn_out

    # MLP: norm -> fc -> relu^2 -> proj
    all_h_normed = F.rms_norm(all_hidden, (D,))
    h_fc = torch.einsum('lbtd,lid->lbti', all_h_normed, W_fc)
    h_sq = F.relu(h_fc).square()
    mlp_out = torch.einsum('lbti,ldi->lbtd', h_sq, W_proj)

    # Second residual
    return all_hidden + mlp_out


# ---------------------------------------------------------------------------
# Batched diagonal Newton (uses PrecomputedWeights from inference_idn)
# ---------------------------------------------------------------------------

def _make_doubled_pw(pw):
    """Create doubled PrecomputedWeights for fused FD (2*n_par layers)."""
    import copy
    pw2 = copy.copy(pw)
    pw2.n_par = pw.n_par * 2
    pw2.par = pw.par + pw.par
    for attr in ['W_q_s', 'W_k_s', 'W_v_s', 'W_o_s', 'W_fc_s', 'W_proj_s', 'resid_lambdas', 'x0_lambdas']:
        if hasattr(pw, attr):
            setattr(pw2, attr, torch.cat([getattr(pw, attr)] * 2, dim=0))
    if hasattr(pw, 've_gate_weights'):
        pw2.ve_gate_weights = pw.ve_gate_weights + pw.ve_gate_weights
    if hasattr(pw, 've_gate_channels'):
        pw2.ve_gate_channels = pw.ve_gate_channels + pw.ve_gate_channels
    if hasattr(pw, 'blocks'):
        pw2.blocks = pw.blocks + pw.blocks
    for attr in ['H_res_attn', 'H_pre_attn', 'H_post_attn',
                 'H_res_mlp', 'H_pre_mlp', 'H_post_mlp', 'J_blocks']:
        if hasattr(pw, attr):
            setattr(pw2, attr, torch.cat([getattr(pw, attr)] * 2, dim=0))
    return pw2


def forward_diagn_batched(model, idx, seq_layers, pw, K=1,
                          jvp_method='fd', fd_eps=1e-2, n_probes=1,
                          init='h0', preheat_cache=None):
    """Diagonal Newton with batched forward + associative scan.

    Uses PrecomputedWeights from inference_idn for batched layer evaluation.
    """
    from snlp.inference_idn import _batched_forward

    L = model.config.n_layer
    par = pw.par
    n_par = pw.n_par
    use_mhc = getattr(model.config, 'use_mhc', False)

    if use_mhc:
        from snlp.inference_hcn import (
            _embed_mhc, _run_block_mhc, _logits_mhc, _batched_forward_mhc,
        )
        embed_fn, block_fn, logits_fn = _embed_mhc, _run_block_mhc, _logits_mhc
        batched_fwd = lambda hp, x0, cs, ve: _batched_forward_mhc(hp, x0, cs, ve, pw, model)
        pw_doubled = _make_doubled_pw(pw)
        batched_fwd_doubled = lambda hp, x0, cs, ve: _batched_forward_mhc(hp, x0, cs, ve, pw_doubled, model)
    else:
        embed_fn, block_fn, logits_fn = _embed, _run_block, _logits
        batched_fwd = lambda hp, x0, cs, ve: _batched_forward(hp, x0, cs, ve, pw, model)
        pw_doubled = _make_doubled_pw(pw)
        batched_fwd_doubled = lambda hp, x0, cs, ve: _batched_forward(hp, x0, cs, ve, pw_doubled, model)

    x0, cos_sin, ve = embed_fn(model, idx)

    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = block_fn(model, i, x, x0, cos_sin, ve)
        h[i] = x
    h_init = x

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        with torch.no_grad():
            all_init = batched_fwd(all_h0, x0, cos_sin, ve)
        hs = [h_init] + [all_init[j] for j in range(n_par)]
    elif init == 'preheat' and preheat_cache is not None:
        h_pred = preheat_cache.predict(x0)
        hs = [h_init] + [h_pred[par[j]] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _k in range(K):
        all_h_prev = torch.stack([hs[j] for j in range(n_par)])

        if jvp_method == 'fd' and n_probes == 1:
            with torch.no_grad():
                z = torch.randint(0, 2, all_h_prev.shape, device=all_h_prev.device, dtype=all_h_prev.dtype) * 2 - 1
                stacked_h = torch.cat([all_h_prev, all_h_prev + fd_eps * z], dim=0)
                stacked_out = batched_fwd_doubled(stacked_h, x0, cos_sin, ve)
                all_fx = stacked_out[:n_par]
                all_fx_pert = stacked_out[n_par:]
                jz = ((all_fx_pert.float() - all_fx.float()) / fd_eps).to(all_h_prev.dtype)
                diag_j_accum = z * jz
        elif jvp_method == 'fd':
            with torch.no_grad():
                all_fx = batched_fwd(all_h_prev, x0, cos_sin, ve)
                diag_j_accum = torch.zeros_like(all_h_prev)
                for _ in range(n_probes):
                    z = torch.randint(0, 2, all_h_prev.shape, device=all_h_prev.device, dtype=all_h_prev.dtype) * 2 - 1
                    all_fx_pert = batched_fwd(all_h_prev + fd_eps * z, x0, cos_sin, ve)
                    jz = ((all_fx_pert.float() - all_fx.float()) / fd_eps).to(all_h_prev.dtype)
                    diag_j_accum = diag_j_accum + z * jz
        elif jvp_method == 'vjp':
            with torch.enable_grad():
                all_h_g = all_h_prev.detach().requires_grad_(True)
                all_fx = batched_fwd(all_h_g, x0, cos_sin, ve)
                diag_j_accum = torch.zeros_like(all_h_prev)
                for p in range(n_probes):
                    z = torch.randint(0, 2, all_h_prev.shape, device=all_h_prev.device, dtype=all_h_prev.dtype) * 2 - 1
                    jtz = torch.autograd.grad(
                        all_fx, all_h_g, grad_outputs=z,
                        retain_graph=(p < n_probes - 1),
                    )[0]
                    diag_j_accum = diag_j_accum + z * jtz.detach()
                all_fx = all_fx.detach()

        diag_j = diag_j_accum / n_probes
        bias = all_fx - diag_j * all_h_prev

        a_full = torch.cat([torch.ones_like(diag_j[:1]), diag_j], dim=0)
        b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
        a_scan, b_scan = associative_scan_diagonal(a_full, b_full)

        with torch.no_grad():
            for l in range(n_par):
                hs[l + 1] = a_scan[l + 1] * h_init + b_scan[l + 1]

    with torch.no_grad():
        for j, li in enumerate(par):
            h[li] = hs[j + 1]
        return logits_fn(model, h)


# ---------------------------------------------------------------------------
# Inference strategies (per-layer, non-batched)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_sequential(model, idx):
    """Standard sequential forward (ground truth)."""
    return model.forward(idx)


@torch.inference_mode()
def forward_idn_loop(model, idx, seq_layers, K=1, elk_k=0.0):
    """
    Identity Newton via sequential loop with Scale-ELK damping.
    J=I baseline: prefix-sum correction.
    """
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x
    par = list(range(seq_layers, L))
    for i in par:
        h[i] = x.clone()
    a = 1.0 - elk_k
    for _ in range(K):
        F_h = {}
        for i in par:
            F_h[i] = _run_block(model, i, h[i - 1], x0, cos_sin, ve)
        h_guess = {i: h[i] for i in par}
        h_corr = F_h[par[0]]
        h[par[0]] = h_corr
        for i in par[1:]:
            h_corr = F_h[i] + a * (h_corr - h_guess[i - 1])
            h[i] = h_corr
    return _logits(model, h)


def forward_diag_scan(model, idx, seq_layers, K=1, elk_k=0.0,
                      jvp_method='vjp', fd_eps=1e-3, n_probes=1):
    """
    Diagonal Newton with associative scan.

    Estimates diag(J) via Hutchinson, then solves the diagonal
    linear recurrence h_l = diag_j_l * h_{l-1} + bias_l via parallel scan.

    Args:
        jvp_method: 'jvp' (forward-mode AD, correct J*z, needs math SDPA),
                    'vjp' (backward-mode AD, uses J^T*z -- WRONG for non-symmetric J),
                    'fd' (finite-difference, noisy in bf16)
        fd_eps: perturbation for finite differences
        n_probes: number of Rademacher vectors for Hutchinson averaging
    """
    L = model.config.n_layer
    par = list(range(seq_layers, L))
    n_par = len(par)

    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)
        h = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h[i] = x
        h_init = x
        # Initialize: all parallel layers start with prefix output
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    dtype = h_init.dtype
    device = h_init.device

    for _ in range(K):
        # Stack current estimates: hs[j] is input to j-th parallel layer
        all_h_prev = torch.stack([hs[j] for j in range(n_par)])  # (n_par, B, T, D)

        # --- Estimate f(x) and diag(J) ---
        diag_j_accum = torch.zeros_like(all_h_prev)

        if jvp_method == 'fd':
            with torch.no_grad():
                all_fx = _batched_block_forward(all_h_prev, x0, cos_sin, ve, par, model)
                for _ in range(n_probes):
                    z = torch.randint(0, 2, all_h_prev.shape, device=device, dtype=dtype) * 2 - 1
                    all_fx_pert = _batched_block_forward(
                        all_h_prev + fd_eps * z, x0, cos_sin, ve, par, model)
                    # float32 subtraction to avoid bf16 cancellation
                    jz = ((all_fx_pert.float() - all_fx.float()) / fd_eps).to(dtype)
                    diag_j_accum = diag_j_accum + z * jz

        elif jvp_method == 'vjp':
            with torch.enable_grad():
                all_h_g = all_h_prev.detach().requires_grad_(True)
                all_fx = _batched_block_forward(all_h_g, x0, cos_sin, ve, par, model)
                for p in range(n_probes):
                    z = torch.randint(0, 2, all_h_prev.shape, device=device, dtype=dtype) * 2 - 1
                    jtz = torch.autograd.grad(
                        all_fx, all_h_g, grad_outputs=z,
                        retain_graph=(p < n_probes - 1),
                    )[0]
                    diag_j_accum = diag_j_accum + z * jtz.detach()
                all_fx = all_fx.detach()

        elif jvp_method == 'jvp':
            # Forward-mode AD: computes exact J*z (not J^T*z like VJP).
            # Requires math SDPA backend (flash SDPA has no second-order derivative).
            from torch.autograd.functional import jvp as autograd_jvp

            def batched_fwd(h):
                return _batched_block_forward(h, x0, cos_sin, ve, par, model,
                                              use_math_sdpa=True)

            # Compute f(x) first (with default SDPA for speed)
            with torch.no_grad():
                all_fx = _batched_block_forward(all_h_prev, x0, cos_sin, ve, par, model)

            for _ in range(n_probes):
                z = torch.randint(0, 2, all_h_prev.shape, device=device, dtype=dtype) * 2 - 1
                with torch.enable_grad():
                    _, jz = autograd_jvp(batched_fwd, (all_h_prev.detach(),), (z,))
                diag_j_accum = diag_j_accum + z * jz.detach()

        diag_j = diag_j_accum / n_probes

        # Scale-ELK damping: attenuate eigenvalues by (1-k)
        if elk_k > 0:
            diag_j = (1.0 - elk_k) * diag_j

        # Affine form: f(h) ~ diag_j * h + bias
        bias = all_fx - diag_j * all_h_prev

        # Associative scan: prepend identity element (a=1, b=0) for h_init
        a_full = torch.cat([torch.ones_like(diag_j[:1]), diag_j], dim=0)
        b_full = torch.cat([torch.zeros_like(bias[:1]), bias], dim=0)
        a_scan, b_scan = associative_scan_diagonal(a_full, b_full)

        # Update: h_l = a_scan[l] * h_init + b_scan[l]
        with torch.no_grad():
            for l in range(n_par):
                hs[l + 1] = a_scan[l + 1] * h_init + b_scan[l + 1]

    # Build final h list for logits
    with torch.no_grad():
        for j, li in enumerate(par):
            h[li] = hs[j + 1]
        return _logits(model, h)


def forward_diag_loop(model, idx, seq_layers, K=1, elk_k=0.0,
                      jvp_method='vjp', fd_eps=1e-3, n_probes=1):
    """
    Diagonal Newton via sequential forward substitution (for correctness checking).
    Same linear recurrence as diag_scan, solved sequentially instead of with scan.
    """
    L = model.config.n_layer
    par = list(range(seq_layers, L))
    n_par = len(par)

    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)
        h = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h[i] = x
        h_init = x
        for i in par:
            h[i] = h_init.clone()

    dtype = h_init.dtype
    device = h_init.device

    for _ in range(K):
        h_old = {i: h[i].detach().clone() for i in par}
        h_new = [h_init]  # h_new[0] = prefix output (fixed)

        for j, li in enumerate(par):
            # Old INPUT to this layer = output of previous layer from old estimates
            # (not the output of this layer itself -- that distinction matters for K>1)
            x_prev = h_init if j == 0 else h_old[par[j - 1]]

            # Compute f(x_prev) and diag(J) at x_prev
            diag_j_accum = torch.zeros_like(x_prev)

            if jvp_method == 'fd':
                with torch.no_grad():
                    fx = _run_block(model, li, x_prev, x0, cos_sin, ve)
                    for _ in range(n_probes):
                        z = torch.randint(0, 2, x_prev.shape, device=device, dtype=dtype) * 2 - 1
                        fx_pert = _run_block(model, li, x_prev + fd_eps * z, x0, cos_sin, ve)
                        jz = ((fx_pert.float() - fx.float()) / fd_eps).to(dtype)
                        diag_j_accum = diag_j_accum + z * jz

            elif jvp_method == 'vjp':
                with torch.enable_grad():
                    x_g = x_prev.detach().requires_grad_(True)
                    x_in = model.resid_lambdas[li] * x_g + model.x0_lambdas[li] * x0
                    ve_li = ve.get(str(li))
                    fx = model.transformer.h[li](x_in, ve_li, cos_sin, model.window_sizes[li], None)
                    for p in range(n_probes):
                        z = torch.randint(0, 2, x_prev.shape, device=device, dtype=dtype) * 2 - 1
                        jtz = torch.autograd.grad(
                            fx, x_g, grad_outputs=z,
                            retain_graph=(p < n_probes - 1),
                        )[0]
                        diag_j_accum = diag_j_accum + z * jtz.detach()
                    fx = fx.detach()

            diag_j = diag_j_accum / n_probes

            # Scale-ELK
            if elk_k > 0:
                diag_j = (1.0 - elk_k) * diag_j

            # Forward substitution: h_new[j+1] = f(x_prev) + diag_j * (h_new[j] - x_prev)
            if j == 0:
                h_new.append(fx)  # correction = 0 since h_new[0] = h_init = x_prev
            else:
                h_new.append(fx + diag_j * (h_new[j] - x_prev))

            h[li] = h_new[-1]

    with torch.no_grad():
        return _logits(model, h)


# ---------------------------------------------------------------------------
# Metrics and timing
# ---------------------------------------------------------------------------

def compute_metrics(logits_ref, logits_test, targets):
    loss = F.cross_entropy(logits_test.reshape(-1, logits_test.size(-1)),
                           targets.reshape(-1), reduction='mean').item()
    ppl = math.exp(min(loss, 20.0))
    r = logits_ref.reshape(-1, logits_ref.size(-1))
    t = logits_test.reshape(-1, logits_test.size(-1))
    top1 = (r.argmax(-1) == t.argmax(-1)).float().mean().item()
    cs = F.cosine_similarity(r, t, dim=-1)
    cos_sim = cs[~cs.isnan()].mean().item() if not cs.isnan().all() else 0.0
    return ppl, top1, cos_sim


def bench(fn, device, n_warm=20, n_run=100):
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
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, batches, n_par_list, device, K=1, elk_k=0.0,
                   jvp_methods=('vjp',), fd_eps=1e-3, n_probes=1,
                   n_warm=20, n_run=100):
    L = model.config.n_layer

    idx_bench = batches[0][:, :-1]

    # Sequential baseline
    seq_ms = bench(lambda: forward_sequential(model, idx_bench), device, n_warm, n_run)
    seq_ppls = []
    for batch in batches:
        idx, targets = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(model, idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction='mean').item()
        seq_ppls.append(math.exp(min(loss, 20.0)))
    seq_ppl = sum(seq_ppls) / len(seq_ppls)
    print(f"  Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}")
    print()

    for n_par in n_par_list:
        if n_par >= L:
            continue
        seq_layers = L - n_par

        # Build method list
        methods = []
        # IDN loop baseline
        methods.append(('IDN (loop)',
                        lambda: forward_idn_loop(model, idx_bench, seq_layers, K=K, elk_k=elk_k),
                        lambda idx: forward_idn_loop(model, idx, seq_layers, K=K, elk_k=elk_k)))

        # Diagonal scan for each JVP method
        for jm in jvp_methods:
            name = f'Diag scan ({jm})'
            methods.append((name,
                            lambda _jm=jm: forward_diag_scan(
                                model, idx_bench, seq_layers, K=K, elk_k=elk_k,
                                jvp_method=_jm, fd_eps=fd_eps, n_probes=n_probes),
                            lambda idx, _jm=jm: forward_diag_scan(
                                model, idx, seq_layers, K=K, elk_k=elk_k,
                                jvp_method=_jm, fd_eps=fd_eps, n_probes=n_probes)))

        # Diagonal loop (VJP only, for correctness check)
        if 'vjp' in jvp_methods:
            methods.append(('Diag loop (vjp)',
                            lambda: forward_diag_loop(
                                model, idx_bench, seq_layers, K=K, elk_k=elk_k,
                                jvp_method='vjp', fd_eps=fd_eps, n_probes=n_probes),
                            lambda idx: forward_diag_loop(
                                model, idx, seq_layers, K=K, elk_k=elk_k,
                                jvp_method='vjp', fd_eps=fd_eps, n_probes=n_probes)))

        # --- Timing ---
        timings = {}
        for name, bench_fn, _ in methods:
            timings[name] = bench(bench_fn, device, n_warm, n_run)

        # --- Quality ---
        results = {}
        for name, _, eval_fn in methods:
            ppls, top1s, css = [], [], []
            for batch in batches:
                idx, targets = batch[:, :-1], batch[:, 1:]
                ref = forward_sequential(model, idx)
                test = eval_fn(idx)
                ppl, top1, cs = compute_metrics(ref, test, targets)
                ppls.append(ppl); top1s.append(top1); css.append(cs)
            results[name] = (sum(ppls)/len(ppls), sum(top1s)/len(top1s), sum(css)/len(css))

        # --- Print table ---
        print(f"  n_par={n_par} (seq={seq_layers} + {n_par} parallel, K={K}, elk_k={elk_k}, probes={n_probes})")
        print(f"  {'Method':<22s} | {'Time':>7s} | {'Speed':>6s} | {'PPL':>8s} | {'dPPL%':>7s} | {'Top-1':>6s} | {'CosSim':>7s}")
        print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
        print(f"  {'Sequential':<22s} | {seq_ms:6.2f}ms | {'1.00x':>6s} | {seq_ppl:8.2f} | {'---':>7s} | {'---':>6s} | {'---':>7s}")
        for name in [n for n, _, _ in methods]:
            ms = timings[name]
            ppl, top1, cs = results[name]
            dppl = 100 * (ppl - seq_ppl) / seq_ppl
            speedup = seq_ms / ms
            ppl_s = f"{ppl:8.2f}" if ppl < 1e5 else f"{ppl:8.0f}"
            print(f"  {name:<22s} | {ms:6.2f}ms | {speedup:5.2f}x | {ppl_s} | {dppl:+6.1f}% | {top1:6.3f} | {cs:7.4f}")

        # --- Debug: max diff vs IDN loop ---
        idx = batches[0][:, :-1]
        out_idn = forward_idn_loop(model, idx, seq_layers, K=K, elk_k=elk_k)
        print(f"\n  Debug max diff vs IDN (loop):")
        for name, _, eval_fn in methods:
            if name == 'IDN (loop)':
                continue
            out = eval_fn(idx)
            diff = (out_idn.float() - out.float()).abs().max().item()
            print(f"    {name:<22s}: {diff:.2e}")

        # --- Debug: scan vs loop agreement ---
        if 'vjp' in jvp_methods:
            out_scan = forward_diag_scan(model, idx, seq_layers, K=K, elk_k=elk_k,
                                         jvp_method='vjp', fd_eps=fd_eps, n_probes=n_probes)
            out_loop = forward_diag_loop(model, idx, seq_layers, K=K, elk_k=elk_k,
                                         jvp_method='vjp', fd_eps=fd_eps, n_probes=n_probes)
            diff_sl = (out_scan.float() - out_loop.float()).abs().max().item()
            print(f"    scan vs loop (vjp):   {diff_sl:.2e}  (should be small if same random seed)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate diagonal Newton (quasi-DEER) inference")
    parser.add_argument("--model-tag", type=str, action='append', required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--n-par", type=str, default="8",
                        help="comma-separated n_par values")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--K", type=int, default=1, help="Newton iterations")
    parser.add_argument("--elk-k", type=float, default=0.0,
                        help="Scale-ELK damping: 0=full DEER, 1=no correction, (0,1)=damped")
    parser.add_argument("--jvp", type=str, default="fd,vjp",
                        help="comma-separated JVP methods: fd, vjp")
    parser.add_argument("--fd-eps", type=float, default=1e-3,
                        help="Finite-difference epsilon")
    parser.add_argument("--n-probes", type=int, default=1,
                        help="Number of Rademacher vectors for Hutchinson averaging")
    parser.add_argument("--n-warm", type=int, default=20, help="Warmup runs for timing")
    parser.add_argument("--n-run", type=int, default=100, help="Timing runs")
    args = parser.parse_args()

    # Force SDPA (FA3 segfaults in some eval paths)
    import nanochat.flash_attention as fa
    fa._override_impl = 'sdpa'
    fa.USE_FA3 = False

    device = torch.device(autodetect_device_type())
    n_par_list = [int(x) for x in args.n_par.split(',')]
    jvp_methods = [s.strip() for s in args.jvp.split(',')]

    for tag in args.model_tag:
        print(f"\n{'='*80}")
        print(f"  Model: {tag}")
        print(f"{'='*80}")
        model = load_model(tag, step=args.step, device=device)
        batches = load_batches(model, device, args.seq_len, args.batch_size, args.n_batches)
        evaluate_model(model, batches, n_par_list, device,
                       K=args.K, elk_k=args.elk_k,
                       jvp_methods=jvp_methods, fd_eps=args.fd_eps,
                       n_probes=args.n_probes,
                       n_warm=args.n_warm, n_run=args.n_run)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
