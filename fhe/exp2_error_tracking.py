"""
Experiment 2b: Per-layer error tracking.

Compares hidden state error accumulation between sequential and SNLP
under polynomial approximations. Tests the core hypothesis:
- Sequential: error grows with layer depth (accumulates over L layers)
- SNLP: error stays bounded (accumulates over K iterations, not L layers)

Usage:
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=4 uv run python -m fhe.exp2_error_tracking \
        --model-tag d32s_idn00625_npar24_s3 --n-par 24 --degree 4
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from snlp.inference_idn import (
    load_model, load_val_data, PrecomputedWeights,
    _batched_forward, _idn_correction,
)
from nanochat.jacobi_forward import _embed, _run_block, _logits
from fhe.he_forward import _run_block_he, _batched_forward_he, _logits_he
from fhe.he_approx import HEApproxConfig

device = torch.device(autodetect_device_type())


@torch.inference_mode()
def track_sequential_errors(model, batches, cfg, n_batches=10):
    """Compute per-layer hidden state error for sequential forward."""
    L = model.config.n_layer
    errors_per_layer = [[] for _ in range(L)]
    logit_errors = []

    for batch in batches[:n_batches]:
        idx = batch[:, :-1]
        x0, cos_sin, ve = _embed(model, idx)

        # Exact forward
        h_exact = [None] * L
        x = x0
        for i in range(L):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h_exact[i] = x

        # HE-approx forward
        h_he = [None] * L
        x = x0
        for i in range(L):
            x = _run_block_he(model, i, x, x0, cos_sin, ve, cfg)
            h_he[i] = x

        # Per-layer relative error
        for i in range(L):
            err = (h_he[i] - h_exact[i]).float().norm() / (h_exact[i].float().norm() + 1e-8)
            errors_per_layer[i].append(err.item())

        # Logit error
        logits_exact = _logits(model, h_exact)
        logits_he = _logits_he(model, h_he, cfg)
        logit_err = (logits_he - logits_exact).float().norm() / (logits_exact.float().norm() + 1e-8)
        logit_errors.append(logit_err.item())

    return {
        'per_layer': [sum(e) / len(e) for e in errors_per_layer],
        'logit_error': sum(logit_errors) / len(logit_errors),
    }


@torch.inference_mode()
def track_snlp_errors(model, batches, cfg, n_par, K, init='h0', n_batches=10):
    """Compute per-layer hidden state error for SNLP IDN batched forward."""
    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    pw = PrecomputedWeights(model, par)

    errors_per_layer = [[] for _ in range(L)]
    logit_errors = []

    for batch in batches[:n_batches]:
        idx = batch[:, :-1]
        x0, cos_sin, ve = _embed(model, idx)

        # --- Exact SNLP forward ---
        h_exact = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h_exact[i] = x
        h_init_exact = x

        if init == 'batch_fwd':
            all_h0 = h_init_exact.unsqueeze(0).expand(n_par, -1, -1, -1)
            all_out = _batched_forward(all_h0, x0, cos_sin, ve, pw, model)
            hs_exact = [h_init_exact] + [all_out[j] for j in range(n_par)]
        else:
            hs_exact = [h_init_exact] + [h_init_exact.clone() for _ in range(n_par)]

        for _k in range(K):
            all_hp = torch.stack([hs_exact[j] for j in range(n_par)])
            all_out = _batched_forward(all_hp, x0, cos_sin, ve, pw, model)
            _idn_correction(all_out, hs_exact, h_init_exact, n_par)

        for j, li in enumerate(par):
            h_exact[li] = hs_exact[j + 1]

        # --- HE-approx SNLP forward ---
        h_he = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block_he(model, i, x, x0, cos_sin, ve, cfg)
            h_he[i] = x
        h_init_he = x

        if init == 'batch_fwd':
            all_h0 = h_init_he.unsqueeze(0).expand(n_par, -1, -1, -1)
            all_out = _batched_forward_he(all_h0, x0, cos_sin, ve, pw, model, cfg)
            hs_he = [h_init_he] + [all_out[j] for j in range(n_par)]
        else:
            hs_he = [h_init_he] + [h_init_he.clone() for _ in range(n_par)]

        for _k in range(K):
            all_hp = torch.stack([hs_he[j] for j in range(n_par)])
            all_out = _batched_forward_he(all_hp, x0, cos_sin, ve, pw, model, cfg)
            _idn_correction(all_out, hs_he, h_init_he, n_par)

        for j, li in enumerate(par):
            h_he[li] = hs_he[j + 1]

        # Per-layer errors
        for i in range(L):
            if h_exact[i] is not None and h_he[i] is not None:
                err = (h_he[i] - h_exact[i]).float().norm() / (h_exact[i].float().norm() + 1e-8)
                errors_per_layer[i].append(err.item())

        # Logit error
        logits_exact = _logits(model, h_exact)
        logits_he = _logits_he(model, h_he, cfg)
        logit_err = (logits_he - logits_exact).float().norm() / (logits_exact.float().norm() + 1e-8)
        logit_errors.append(logit_err.item())

    return {
        'per_layer': [sum(e) / len(e) if e else 0 for e in errors_per_layer],
        'logit_error': sum(logit_errors) / len(logit_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--n-par', type=int, default=24)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--degrees', type=int, nargs='+', default=None,
                        help='Multiple degrees to sweep')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--init', default='h0', choices=['h0', 'batch_fwd'])
    parser.add_argument('--n-batches', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=1_000_000)
    parser.add_argument('--output-dir', default='cache/logs_fhe_claude')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_tag}")
    model = load_model(args.model_tag, device=device)
    L = model.config.n_layer

    print(f"Loading validation data (seq_len={args.seq_len})")
    batches = load_val_data(device, args.max_tokens, seq_len=args.seq_len)

    cfg = HEApproxConfig.uniform(degree=args.degree)
    print(f"HE config: rms_iters={cfg.rms_norm_iters}, softmax_deg={cfg.softmax_degree}, "
          f"sigmoid_deg={cfg.sigmoid_degree}, tanh_deg={cfg.tanh_degree}")

    # Sequential error tracking
    print("\n--- Sequential error tracking ---")
    seq_errors = track_sequential_errors(model, batches, cfg, args.n_batches)
    print(f"Logit error: {seq_errors['logit_error']:.6f}")
    print("Per-layer errors:")
    for i, e in enumerate(seq_errors['per_layer']):
        bar = '#' * min(int(e * 200), 60)
        print(f"  Layer {i:2d}: {e:.6f} {bar}")

    # SNLP error tracking
    print(f"\n--- SNLP error tracking (n_par={args.n_par}, K={args.K}, init={args.init}) ---")
    snlp_errors = track_snlp_errors(model, batches, cfg, args.n_par, args.K, args.init, args.n_batches)
    print(f"Logit error: {snlp_errors['logit_error']:.6f}")
    print("Per-layer errors:")
    for i, e in enumerate(snlp_errors['per_layer']):
        bar = '#' * min(int(e * 200), 60)
        print(f"  Layer {i:2d}: {e:.6f} {bar}")

    # Comparison
    seq_layers = L - args.n_par
    print(f"\n{'='*60}")
    print("Error comparison (sequential vs SNLP)")
    print(f"{'='*60}")
    print(f"{'Layer':>6} {'Sequential':>12} {'SNLP':>12} {'Ratio':>8}")
    print('-' * 42)
    for i in range(L):
        s = seq_errors['per_layer'][i]
        n = snlp_errors['per_layer'][i]
        ratio = s / n if n > 1e-10 else float('inf')
        marker = " *" if i >= seq_layers else ""
        print(f"  {i:4d} {s:12.6f} {n:12.6f} {ratio:8.2f}x{marker}")
    print(f"\n  Logit: {seq_errors['logit_error']:12.6f} {snlp_errors['logit_error']:12.6f} "
          f"{seq_errors['logit_error']/max(snlp_errors['logit_error'],1e-10):8.2f}x")
    print(f"\n  * = parallel layer (SNLP evaluates these in parallel)")

    # Save results
    result = {
        'model_tag': args.model_tag,
        'n_par': args.n_par,
        'K': args.K,
        'init': args.init,
        'degree': args.degree,
        'n_batches': args.n_batches,
        'seq_len': args.seq_len,
        'sequential_errors': seq_errors,
        'snlp_errors': snlp_errors,
    }

    out_path = os.path.join(args.output_dir,
                            f'exp2_error_tracking_{args.model_tag}_n{args.n_par}_K{args.K}_deg{args.degree}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
