"""
Calibrate activation ranges for Chebyshev polynomial fitting.

Runs forward passes through a model and records the input distributions
(min, max, mean, std, percentiles) for each nonlinear operation at each layer.
This determines the optimal intervals for polynomial approximations.

Usage:
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=4 uv run python -m fhe.exp_calibrate \
        --model-tag d32s_idn00625_npar24_s3 --n-par 24 --n-batches 20
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import norm

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from snlp.inference_idn import load_model, load_val_data, PrecomputedWeights
from nanochat.jacobi_forward import _embed, _run_block

device = torch.device(autodetect_device_type())


def tensor_stats(x: torch.Tensor) -> dict:
    x_flat = x.float().flatten()
    # Sample if too large for quantile (>1M elements)
    if x_flat.numel() > 1_000_000:
        idx = torch.randperm(x_flat.numel(), device=x_flat.device)[:1_000_000]
        x_sample = x_flat[idx]
    else:
        x_sample = x_flat
    return {
        'min': x_flat.min().item(),
        'max': x_flat.max().item(),
        'mean': x_flat.mean().item(),
        'std': x_flat.std().item(),
        'p01': x_sample.quantile(0.01).item(),
        'p05': x_sample.quantile(0.05).item(),
        'p95': x_sample.quantile(0.95).item(),
        'p99': x_sample.quantile(0.99).item(),
        'abs_max': x_flat.abs().max().item(),
    }


@torch.inference_mode()
def calibrate_model(model, batches, n_par):
    """Run forward passes and collect statistics for all nonlinear operations."""
    L = model.config.n_layer
    D = model.config.n_embd
    n_head = model.config.n_head
    head_dim = D // n_head
    seq_layers = L - n_par
    par = list(range(seq_layers, L))

    stats = {
        'rms_norm_input_mean_sq': {f'layer_{i}': [] for i in range(L)},
        'qk_norm_input_mean_sq': {f'layer_{i}': [] for i in range(L)},
        'sigmoid_gate_input': {f'layer_{i}': [] for i in range(L)},
        'attn_scores': {f'layer_{i}': [] for i in range(L)},
        'tanh_logit_input': {'global': []},
        'smear_gate_input': {'global': []},
        'relu_input': {f'layer_{i}': [] for i in range(L)},
    }

    pw = PrecomputedWeights(model, par)

    for batch_idx, batch in enumerate(batches):
        idx = batch[:, :-1]
        x0, cos_sin, ve = _embed(model, idx)
        B, T, _ = x0.shape

        if T > 1:
            smear_input = model.smear_gate(x0[:, 1:, :24])
            stats['smear_gate_input']['global'].append(tensor_stats(smear_input))

        h_prev = x0
        for i in range(L):
            if getattr(model.config, 'no_x0_resid', False):
                x_in = h_prev
            else:
                x_in = model.resid_lambdas[i] * h_prev + model.x0_lambdas[i] * x0

            mean_sq_pre_attn = (x_in.float() ** 2).mean(dim=-1) + 1e-6
            stats['rms_norm_input_mean_sq'][f'layer_{i}'].append(tensor_stats(mean_sq_pre_attn))

            x_normed = F.rms_norm(x_in, (D,))

            block = model.transformer.h[i]
            q = block.attn.c_q(x_normed)
            k = block.attn.c_k(x_normed)

            q = q.view(B, T, n_head, head_dim)
            k = k.view(B, T, model.config.n_kv_head, head_dim)

            cos, sin = cos_sin
            from nanochat.gpt import apply_rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

            qk_mean_sq = (q.float() ** 2).mean(dim=-1) + 1e-6
            stats['qk_norm_input_mean_sq'][f'layer_{i}'].append(tensor_stats(qk_mean_sq))

            q_n = F.rms_norm(q, (head_dim,)) * 1.2
            k_n = F.rms_norm(k, (head_dim,)) * 1.2

            q_t = q_n.transpose(1, 2)
            k_t = k_n.transpose(1, 2)
            attn_scores = (q_t @ k_t.transpose(-2, -1)) * (head_dim ** -0.5)
            stats['attn_scores'][f'layer_{i}'].append(tensor_stats(attn_scores))

            ve_j = ve.get(str(i))
            if ve_j is not None and block.attn.ve_gate is not None:
                gate_input = block.attn.ve_gate(x_normed[:, :, :12])
                stats['sigmoid_gate_input'][f'layer_{i}'].append(tensor_stats(gate_input))

            h_prev = _run_block(model, i, h_prev, x0, cos_sin, ve)

            fc_in = F.rms_norm(h_prev, (D,))  # simplified: actual is on post-attn hidden
            fc_out = block.mlp.c_fc(fc_in)
            stats['relu_input'][f'layer_{i}'].append(tensor_stats(fc_out))

        h = [None] * L
        h[-1] = h_prev
        n_layer = model.config.n_layer
        x_final = h[-1]
        logit_input = model.lm_head(norm(x_final)).float() / 15.0
        stats['tanh_logit_input']['global'].append(tensor_stats(logit_input))

        if (batch_idx + 1) % 5 == 0:
            print(f"  Calibrated {batch_idx + 1}/{len(batches)} batches")

    aggregated = {}
    for op_name, layer_stats in stats.items():
        aggregated[op_name] = {}
        for layer_key, batch_list in layer_stats.items():
            if not batch_list:
                continue
            agg = {}
            for key in ['min', 'max', 'mean', 'std', 'p01', 'p05', 'p95', 'p99', 'abs_max']:
                vals = [s[key] for s in batch_list]
                if key in ('min', 'p01', 'p05'):
                    agg[key] = min(vals)
                elif key in ('max', 'p99', 'p95', 'abs_max'):
                    agg[key] = max(vals)
                else:
                    agg[key] = sum(vals) / len(vals)
            aggregated[op_name][layer_key] = agg

    return aggregated


def compute_intervals(calibration: dict) -> dict:
    """Derive Chebyshev fitting intervals from calibration data."""
    intervals = {}

    rms_stats = calibration.get('rms_norm_input_mean_sq', {})
    all_mins = [s['p01'] for s in rms_stats.values() if s]
    all_maxs = [s['p99'] for s in rms_stats.values() if s]
    if all_mins and all_maxs:
        intervals['rsqrt'] = (max(min(all_mins) * 0.5, 1e-4), max(all_maxs) * 2.0)

    attn_stats = calibration.get('attn_scores', {})
    all_mins = [s['p01'] for s in attn_stats.values() if s]
    all_maxs = [s['p99'] for s in attn_stats.values() if s]
    if all_mins and all_maxs:
        intervals['exp'] = (min(all_mins) * 1.5, min(max(all_maxs) * 1.5, 10.0))

    sig_stats = calibration.get('sigmoid_gate_input', {})
    all_abs = [s['abs_max'] for s in sig_stats.values() if s]
    if all_abs:
        r = max(all_abs) * 1.2
        intervals['sigmoid'] = (-r, r)

    tanh_stats = calibration.get('tanh_logit_input', {})
    if 'global' in tanh_stats:
        r = tanh_stats['global']['abs_max'] * 1.2
        intervals['tanh'] = (-r, r)

    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--n-par', type=int, default=24)
    parser.add_argument('--n-batches', type=int, default=20)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=1_000_000)
    parser.add_argument('--output-dir', default='cache/logs_fhe_claude')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_tag}")
    model = load_model(args.model_tag, device=device, cache_hc=False)

    print(f"Loading validation data (seq_len={args.seq_len})")
    batches = load_val_data(device, args.max_tokens, seq_len=args.seq_len)
    batches = batches[:args.n_batches]
    print(f"Using {len(batches)} batches")

    print("Calibrating...")
    calibration = calibrate_model(model, batches, args.n_par)
    intervals = compute_intervals(calibration)

    result = {
        'model_tag': args.model_tag,
        'n_par': args.n_par,
        'n_batches': len(batches),
        'seq_len': args.seq_len,
        'calibration': calibration,
        'recommended_intervals': intervals,
    }

    out_path = os.path.join(args.output_dir, f'calibration_{args.model_tag}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibration saved to {out_path}")

    print("\nRecommended intervals:")
    for op, interval in intervals.items():
        print(f"  {op}: [{interval[0]:.4f}, {interval[1]:.4f}]")

    print("\nKey statistics:")
    for op_name in ['rms_norm_input_mean_sq', 'attn_scores', 'sigmoid_gate_input', 'tanh_logit_input']:
        data = calibration.get(op_name, {})
        if not data:
            continue
        all_mins = [s.get('min', 0) for s in data.values() if s]
        all_maxs = [s.get('max', 0) for s in data.values() if s]
        print(f"  {op_name}: global_min={min(all_mins):.4f}, global_max={max(all_maxs):.4f}")


if __name__ == '__main__':
    main()
