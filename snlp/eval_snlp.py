"""
SNLP single-config evaluator: PPL, timing, top-1, cos_sim.
Supports IDN, ChunkB, HCN-Newton, DiagN methods on basic, nox0ve, and mHC models.

Usage:
    # IDN batched
    uv run python -m snlp.eval_snlp --model-tag d32s_idn05_npar24_s3 \
        --n-par 24 --method IDN_batched --K 2 --init h0

    # ChunkB (12 chunks of 2 fused layers)
    uv run python -m snlp.eval_snlp --model-tag d32s_idn05_npar24_s3 \
        --n-par 24 --method ChunkB --chunks 12 --K 2 --init h0

    # mHC-Newton
    uv run python -m snlp.eval_snlp --model-tag d32s_mhc4_x0ve_newton05 \
        --n-par 20 --method mHC-Newton --K 4 --init h0 --cache-hc

    # DiagN (VJP estimator)
    uv run python -m snlp.eval_snlp --model-tag d32s_idn05_npar24_s3 \
        --n-par 24 --method DiagN --K 4 --init batch_fwd --jvp-method vjp

    # Sequential only (just report seq PPL + timing)
    uv run python -m snlp.eval_snlp --model-tag d32s_baseline_4800 --n-par 12
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type, COMPUTE_DTYPE

import nanochat.flash_attention as fa
fa._override_impl = 'sdpa'
fa.USE_FA3 = False

from snlp.inference_idn import (
    load_model, PrecomputedWeights, ChunkWeights, load_val_data, bench,
    forward_sequential, forward_idn_batched, forward_idn_fused,
    forward_idn_fused_split, forward_idn_chunkwise,
)
from snlp.inference_hcn import (
    MHCPrecomputedWeights, MHCChunkWeights,
    forward_hcn_batched as forward_mhc_newton,
    forward_hcn_chunkwise,
    forward_sequential as forward_sequential_mhc,
)
from snlp.inference_diagn import forward_diagn_batched

device = torch.device(autodetect_device_type())


@torch.inference_mode()
def compute_top1_cossim(seq_fn, method_fn, batches, n_batches=100):
    top1s, cossims = [], []
    for batch in batches[:n_batches]:
        idx = batch[:, :-1]
        logits_seq = seq_fn(idx)
        logits_method = method_fn(idx)
        ref = logits_seq.reshape(-1, logits_seq.size(-1)).float()
        test = logits_method.reshape(-1, logits_method.size(-1)).float()
        top1s.append((ref.argmax(-1) == test.argmax(-1)).float().mean().item())
        cos = F.cosine_similarity(ref, test, dim=-1)
        cossims.append(cos[~cos.isnan()].mean().item() if not cos.isnan().all() else 0.0)
    return sum(top1s) / len(top1s), sum(cossims) / len(cossims)


def eval_ppl(method_fn, batches):
    losses = []
    for batch in batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = method_fn(idx)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
    return math.exp(min(sum(losses) / len(losses), 20))


def main():
    parser = argparse.ArgumentParser(description="SNLP single-config evaluator")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--n-par", type=int, required=True)
    parser.add_argument("--method", type=str, default=None,
                        choices=['IDN_batched', 'ChunkB', 'mHC-Newton', 'ChunkB_mHC',
                                 'Fused', 'Fused+split', 'DiagN'],
                        help="Inference method (omit for sequential-only)")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--init", type=str, default='h0', choices=['h0', 'batch_fwd'])
    parser.add_argument("--elk", type=float, default=0.0)
    parser.add_argument("--chunks", type=int, default=None,
                        help="Number of chunks for ChunkB (chunk_size = n_par / chunks)")
    parser.add_argument("--jvp-method", type=str, default='vjp', choices=['fd', 'vjp'],
                        help="Jacobian estimator for DiagN")
    parser.add_argument("--cache-hc", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--top1-batches", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model = load_model(args.model_tag, device=device, cache_hc=args.cache_hc)
    L = model.config.n_layer
    n_par = args.n_par
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    use_mhc = getattr(model.config, 'use_mhc', False)
    no_x0 = getattr(model.config, 'no_x0_resid', False)
    model_type = "mhc" if use_mhc else ("nox0ve" if no_x0 else "basic")

    print(f"Model: {args.model_tag}, L={L}, n_par={n_par}, type={model_type}")

    batches = load_val_data(device, args.max_tokens)
    print(f"Val: {len(batches) * 128:,} tokens, {len(batches)} batches")

    # Precompute weights
    if use_mhc:
        pw = MHCPrecomputedWeights(model, par)
    else:
        pw = PrecomputedWeights(model, par)

    # Sequential baseline
    idx_bench = batches[0][:, :-1]
    seq_fn = (lambda idx: forward_sequential_mhc(model, idx)) if use_mhc else (lambda idx: forward_sequential(model, idx))
    seq_ms = bench(lambda: seq_fn(idx_bench), device)
    seq_ppl = eval_ppl(seq_fn, batches)
    print(f"Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}")

    results = {
        'model': args.model_tag, 'n_par': n_par, 'model_type': model_type,
        'seq_ppl': seq_ppl, 'seq_ms': seq_ms,
    }

    if args.method is None:
        print("No method specified — sequential-only evaluation done.")
    else:
        # Build the forward function for the requested config
        method = args.method
        K, elk_k, init = args.K, args.elk, args.init

        if method == 'IDN_batched':
            ppl_fn = lambda idx: forward_idn_batched(
                model, idx, seq_layers, pw, K=K, elk_k=elk_k, init=init)

        elif method == 'Fused':
            fused_cw = [ChunkWeights(model, par)]
            ppl_fn = lambda idx: forward_idn_fused(
                model, idx, seq_layers, pw, fused_cw, K=K, elk_k=elk_k)

        elif method == 'Fused+split':
            ppl_fn = lambda idx: forward_idn_fused_split(
                model, idx, seq_layers, pw, K=K, elk_k=elk_k, init=init)

        elif method == 'ChunkB':
            nc = args.chunks
            if nc is None:
                parser.error("ChunkB requires --chunks N")
            cs = n_par // nc
            if use_mhc:
                cwl = [MHCChunkWeights(model, list(range(seq_layers + i*cs, seq_layers + (i+1)*cs)))
                       for i in range(nc)]
                ppl_fn = lambda idx: forward_hcn_chunkwise(
                    model, idx, seq_layers, n_par, cwl, pw, K=K, init=init)
            else:
                cwl = [ChunkWeights(model, par[i*cs:(i+1)*cs]) for i in range(nc)]
                ppl_fn = lambda idx: forward_idn_chunkwise(
                    model, idx, seq_layers, pw, cwl, K=K, elk_k=elk_k, init=init)

        elif method == 'mHC-Newton':
            ppl_fn = lambda idx: forward_mhc_newton(
                model, idx, seq_layers=seq_layers, pw=pw, K=K, init=init)

        elif method == 'ChunkB_mHC':
            nc = args.chunks
            if nc is None:
                parser.error("ChunkB_mHC requires --chunks N")
            cs = n_par // nc
            cwl = [MHCChunkWeights(model, list(range(seq_layers + i*cs, seq_layers + (i+1)*cs)))
                   for i in range(nc)]
            ppl_fn = lambda idx: forward_hcn_chunkwise(
                model, idx, seq_layers, n_par, cwl, pw, K=K, init=init)

        elif method == 'DiagN':
            ppl_fn = lambda idx: forward_diagn_batched(
                model, idx, seq_layers, pw, K=K,
                jvp_method=args.jvp_method, init=init)

        config_name = method
        if method in ('ChunkB', 'ChunkB_mHC'):
            config_name = f"{method}_{args.chunks}xF{n_par // args.chunks}"
        print(f"\nEvaluating: {config_name} K={K} {init} elk={elk_k}")

        # Timing
        ms = bench(lambda: ppl_fn(idx_bench), device)
        speed = seq_ms / ms

        # PPL
        ppl = eval_ppl(ppl_fn, batches)
        dppl = 100 * (ppl - seq_ppl) / seq_ppl

        # Top-1 and cos_sim
        top1, cos_sim = compute_top1_cossim(seq_fn, ppl_fn, batches, args.top1_batches)

        print(f"  PPL={ppl:.2f} ({dppl:+.1f}%) {ms:.2f}ms ({speed:.2f}x) top1={top1:.3f} cos={cos_sim:.4f}")

        results['config'] = {
            'method': method, 'K': K, 'init': init, 'elk': elk_k,
            'chunks': args.chunks, 'jvp_method': args.jvp_method if method == 'DiagN' else None,
            'ppl': ppl, 'dppl': dppl, 'ms': ms, 'speedup': speed,
            'top1': top1, 'cos_sim': cos_sim,
        }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
