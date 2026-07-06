"""
Experiment 2: PPL evaluation under HE polynomial approximations.

Compares PPL degradation for sequential vs SNLP under polynomial
approximations of varying degree. Sweeps over (n_par, K, init, degree).

Core hypothesis: SNLP accumulates less approximation error than sequential
because errors only compose over K iterations, not L layers.

Supports IDN_batched, ChunkB, and mHC-Newton methods.

Usage:
    # Smoke test: one model, one degree, selected configs
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=6 uv run python -m fhe.exp2_ppl_eval \
        --model-tag d32s_idn00625_npar24_s3 --degree 12

    # Full sweep: all n_par/K combos
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=6 uv run python -m fhe.exp2_ppl_eval \
        --model-tag d32s_idn00625_npar24_s3 --degree 12 --sweep

    # Degree comparison
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=6 uv run python -m fhe.exp2_ppl_eval \
        --model-tag d32s_idn00625_npar24_s3 --degrees 8 10 12 14

    # Explicit output path
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=6 uv run python -m fhe.exp2_ppl_eval \
        --model-tag d32s_idn00625_npar24_s3 --degree 12 --output cache/logs_fhe_claude/phase2/my_result.json

    # mHC-Newton method
    NANOCHAT_BASE_DIR=cache_old CUDA_VISIBLE_DEVICES=6 uv run python -m fhe.exp2_ppl_eval \
        --model-tag d32s_mhc4_x0ve_newton05 --degree 12 --method mHC-Newton
"""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from snlp.inference_idn import (
    load_model, load_val_data, PrecomputedWeights, ChunkWeights,
    forward_sequential, forward_idn_batched, forward_idn_chunkwise,
)
from snlp.inference_hcn import (
    MHCPrecomputedWeights, MHCChunkWeights,
    forward_hcn_batched as forward_mhc_newton,
    forward_sequential as forward_sequential_mhc,
)
from fhe.he_forward import (
    forward_sequential_he, forward_idn_batched_he,
    forward_mhc_batched_he, forward_idn_chunkwise_he,
)
from fhe.he_approx import HEApproxConfig

device = torch.device(autodetect_device_type())


def eval_ppl(forward_fn, batches):
    """Compute PPL over batches."""
    losses = []
    for batch in batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = forward_fn(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
        if math.isnan(loss) or math.isinf(loss):
            loss = 20.0
        losses.append(loss)
    return math.exp(min(sum(losses) / len(losses), 20))


def run_config(model, batches, n_par, K, init, cfg, seq_layers, pw, method='IDN_batched',
               chunks=None, chunk_weights=None):
    """Run a single (n_par, K, init, degree) config and return results."""
    L = model.config.n_layer
    nfe = seq_layers + K

    # Build forward functions based on method
    if method == 'IDN_batched':
        forward_he = lambda idx: forward_idn_batched_he(model, idx, seq_layers, pw, cfg, K=K, init=init)
        forward_exact = lambda idx: forward_idn_batched(model, idx, seq_layers, pw, K=K, init=init)
    elif method == 'mHC-Newton':
        forward_he = lambda idx: forward_mhc_batched_he(model, idx, seq_layers, pw, cfg, K=K, init=init)
        forward_exact = lambda idx: forward_mhc_newton(model, idx, seq_layers=seq_layers, pw=pw, K=K, init=init)
    elif method == 'ChunkB':
        forward_he = lambda idx: forward_idn_chunkwise_he(
            model, idx, seq_layers, pw, chunk_weights, cfg, K=K, elk_k=0.0, init=init)
        forward_exact = lambda idx: forward_idn_chunkwise(
            model, idx, seq_layers, pw, chunk_weights, K=K, elk_k=0.0, init=init)
    else:
        raise ValueError(f"Unknown method: {method}")

    ppl_he = eval_ppl(forward_he, batches)
    ppl_exact = eval_ppl(forward_exact, batches)

    degradation_pct = 100 * (ppl_he - ppl_exact) / max(ppl_exact, 1e-8)

    return {
        'n_par': n_par, 'K': K, 'init': init, 'method': method,
        'NFE': nfe,
        'ppl_exact': round(ppl_exact, 2),
        'ppl_he': round(ppl_he, 2),
        'degradation_pct': round(degradation_pct, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="FHE PPL evaluation sweep")
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--degree', type=int, default=12)
    parser.add_argument('--degrees', type=int, nargs='+', default=None,
                        help='Multiple degrees to sweep (overrides --degree)')
    parser.add_argument('--method', type=str, default=None,
                        choices=['IDN_batched', 'mHC-Newton', 'ChunkB'],
                        help='Inference method (auto-detected if omitted)')
    parser.add_argument('--chunks', type=int, default=None,
                        help='Number of chunks for ChunkB')
    parser.add_argument('--sweep', action='store_true',
                        help='Full sweep over n_par/K combos')
    parser.add_argument('--noise-bits', type=int, default=None,
                        help='CKKS noise bits (None = no noise)')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=1_000_000)
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (overrides --output-dir auto-naming)')
    parser.add_argument('--output-dir', default='cache/logs_fhe_claude')
    args = parser.parse_args()

    print(f"Loading model: {args.model_tag}")
    model = load_model(args.model_tag, device=device, cache_hc=False)
    L = model.config.n_layer

    # Auto-detect model type and cache HC matrices if needed
    use_mhc = getattr(model.config, 'use_mhc', False)
    no_x0 = getattr(model.config, 'no_x0_resid', False)
    model_type = "mhc" if use_mhc else ("nox0ve" if no_x0 else "basic")

    if use_mhc:
        for block in model.transformer.h:
            block.hc_attn.cache_hc_matrices(model.config.mhc_sinkhorn_iters, model.config.mhc_sinkhorn_tau)
            block.hc_mlp.cache_hc_matrices(model.config.mhc_sinkhorn_iters, model.config.mhc_sinkhorn_tau)

    # Auto-detect method if not specified
    method = args.method
    if method is None:
        method = 'mHC-Newton' if use_mhc else 'IDN_batched'
    print(f"Model: {args.model_tag}, L={L}, type={model_type}, method={method}")

    print(f"Loading validation data (seq_len={args.seq_len})")
    batches = load_val_data(device, args.max_tokens, seq_len=args.seq_len)
    print(f"Loaded {len(batches)} batches, {len(batches) * args.seq_len:,} tokens")

    degrees = args.degrees if args.degrees else [args.degree]

    # Define configs to sweep
    if args.sweep:
        configs = []
        for n_par in [8, 12, 16, 20, 24]:
            if n_par > L:
                continue
            for K in [1, 2, 4, 8]:
                for init in ['h0', 'batch_fwd']:
                    configs.append((n_par, K, init))
    else:
        configs = [
            (24, 1, 'h0'), (24, 2, 'h0'), (24, 4, 'h0'),
            (12, 1, 'h0'), (12, 2, 'h0'), (12, 2, 'batch_fwd'),
            (8, 1, 'h0'), (8, 1, 'batch_fwd'), (8, 4, 'h0'), (8, 4, 'batch_fwd'),
            (20, 4, 'batch_fwd'), (20, 4, 'h0'),
        ]

    all_results = {
        'model_tag': args.model_tag,
        'model_type': model_type,
        'method': method,
        'seq_len': args.seq_len,
        'n_batches': len(batches),
        'noise_bits': args.noise_bits,
    }

    # Sequential baselines
    print("\n--- Sequential baselines ---")
    seq_fn = (lambda idx: forward_sequential_mhc(model, idx)) if use_mhc else (lambda idx: forward_sequential(model, idx))
    seq_ppl_exact = eval_ppl(seq_fn, batches)
    print(f"Sequential exact PPL: {seq_ppl_exact:.2f}")
    all_results['seq_ppl_exact'] = round(seq_ppl_exact, 2)

    for deg in degrees:
        cfg = HEApproxConfig.uniform(degree=deg, noise_bits=args.noise_bits)
        seq_ppl_he = eval_ppl(lambda idx, c=cfg: forward_sequential_he(model, idx, c), batches)
        seq_deg_pct = 100 * (seq_ppl_he - seq_ppl_exact) / seq_ppl_exact
        print(f"Sequential HE deg={deg}: PPL={seq_ppl_he:.2f} ({seq_deg_pct:+.2f}%)")
        all_results[f'seq_ppl_he_deg{deg}'] = round(seq_ppl_he, 2)
        all_results[f'seq_degradation_pct_deg{deg}'] = round(seq_deg_pct, 2)

    # SNLP configs
    results_by_degree = {}
    for deg in degrees:
        cfg = HEApproxConfig.uniform(degree=deg, noise_bits=args.noise_bits)
        print(f"\n--- SNLP configs (method={method}, degree={deg}, noise_bits={args.noise_bits}) ---")
        print(f"{'n_par':>5} {'K':>3} {'init':>10} {'NFE':>4} {'PPL_exact':>10} "
              f"{'PPL_HE':>10} {'degrad%':>8} {'vs_seq%':>8}")
        print('-' * 70)

        deg_results = []
        for n_par, K, init in configs:
            seq_layers = L - n_par
            par = list(range(seq_layers, L))

            # Build appropriate weight cache
            if method == 'mHC-Newton':
                pw = MHCPrecomputedWeights(model, par)
            else:
                pw = PrecomputedWeights(model, par)

            chunk_weights = None
            if method == 'ChunkB' and args.chunks:
                cs = n_par // args.chunks
                chunk_weights = [ChunkWeights(model, par[i*cs:(i+1)*cs]) for i in range(args.chunks)]

            result = run_config(model, batches, n_par, K, init, cfg, seq_layers, pw,
                                method=method, chunks=args.chunks, chunk_weights=chunk_weights)

            seq_he_ppl = all_results.get(f'seq_ppl_he_deg{deg}', seq_ppl_exact)
            vs_seq = 100 * (result['ppl_he'] - seq_he_ppl) / max(seq_he_ppl, 1e-8)
            result['vs_seq_he_pct'] = round(vs_seq, 2)
            deg_results.append(result)

            print(f"{n_par:5d} {K:3d} {init:>10s} {result['NFE']:4d} "
                  f"{result['ppl_exact']:10.2f} {result['ppl_he']:10.2f} "
                  f"{result['degradation_pct']:+8.2f}% {vs_seq:+8.2f}%")

        results_by_degree[f'deg{deg}'] = deg_results

    all_results['configs'] = results_by_degree

    # Summary: compare error amplification
    print(f"\n{'='*60}")
    print("Summary: Error amplification (PPL_HE / PPL_exact)")
    print(f"{'='*60}")
    for deg in degrees:
        seq_he = all_results.get(f'seq_ppl_he_deg{deg}', seq_ppl_exact)
        seq_ratio = seq_he / max(seq_ppl_exact, 1e-8)
        print(f"\nDegree {deg}:")
        print(f"  Sequential (NFE={L}): {seq_ratio:.4f}x")
        for r in results_by_degree[f'deg{deg}']:
            ratio = r['ppl_he'] / max(r['ppl_exact'], 1e-8)
            print(f"  n{r['n_par']} K{r['K']} {r['init']:>10s} (NFE={r['NFE']:2d}): {ratio:.4f}x")

    # Save
    if args.output:
        out_path = args.output
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = args.model_tag
        noise_str = f"_noise{args.noise_bits}" if args.noise_bits else ""
        deg_str = "_".join(f"deg{d}" for d in degrees)
        sweep_str = "_sweep" if args.sweep else ""
        method_str = f"_{method.replace('-', '').lower()}" if method != 'IDN_batched' else ""
        out_path = os.path.join(args.output_dir, f'exp2_ppl_{tag}_{deg_str}{noise_str}{sweep_str}{method_str}.json')

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
