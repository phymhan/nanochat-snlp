"""
SNLP evaluation for off-the-shelf HuggingFace models (Qwen, TinyLlama, Gemma).

Uses the consolidated inference_idn_ots module with backend dispatch.
Evaluates: sequential PPL, IDN batched, ChunkB with timing.

Usage:
    HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      uv run python -m snlp.eval_snlp_ots \
      --model Qwen/Qwen2.5-0.5B-Instruct --n-par 8

    # Auto-detect model type
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-par 4 8
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from snlp.inference_idn_ots import (
    get_backend, forward_sequential, forward_idn_batched,
    forward_chunkwise_batched, load_val_data, bench,
)

MODEL_DEFAULTS = {
    'qwen': 'Qwen/Qwen2.5-0.5B-Instruct',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'gemma': 'google/gemma-3-1b-it',
}


def build_chunk_configs(backend, par, n_par):
    configs = {}
    configs[f'{n_par}xF1'] = [backend.chunk_weights([li]) for li in par]

    candidate_N = {n_par}
    if n_par >= 2: candidate_N.add(2)
    if n_par >= 4: candidate_N.add(4)
    if n_par % 2 == 0: candidate_N.add(n_par // 2)
    if n_par % 4 == 0: candidate_N.add(n_par // 4)

    for N in sorted(candidate_N):
        if N <= 0 or N > n_par or n_par % N != 0:
            continue
        M = n_par // N
        key = f'{N}xF{M}'
        if key not in configs:
            configs[key] = [backend.chunk_weights(par[i*M:(i+1)*M]) for i in range(N)]
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name or path")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=list(MODEL_DEFAULTS.keys()),
                        help="Model type (auto-detected if not set)")
    parser.add_argument("--n-par", type=int, nargs='+', required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    if args.model is None and args.model_type is not None:
        args.model = MODEL_DEFAULTS[args.model_type]
    elif args.model is None:
        parser.error("Provide --model or --model-type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16,
    ).eval().to(device)

    backend = get_backend(model)
    L = model.config.num_hidden_layers
    print(f"Model: {args.model}, type={model.config.model_type}, L={L}")

    print("Loading wikitext-2 validation data...")
    batches = load_val_data(tokenizer, device, args.seq_len, args.max_tokens)
    print(f"Val: {len(batches) * args.seq_len:,} tokens, {len(batches)} batches")

    idx_bench = batches[0][:, :-1]
    seq_ms = bench(lambda: forward_sequential(backend, idx_bench), device)

    seq_losses = []
    for batch in batches:
        idx, tgt = batch[:, :-1], batch[:, 1:]
        logits = forward_sequential(backend, idx)
        seq_losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
    seq_ppl = math.exp(sum(seq_losses) / len(seq_losses))
    print(f"Sequential: {seq_ms:.2f}ms, PPL={seq_ppl:.2f}")

    results = {'model': args.model, 'seq_ppl': seq_ppl, 'seq_ms': seq_ms, 'L': L}

    K_values = [1, 2, 4, 8]
    init_strategies = ['h0', 'batch_fwd']

    for n_par in args.n_par:
        if n_par >= L:
            print(f"\nSkipping n_par={n_par} (>= L={L})")
            continue
        seq_layers = L - n_par
        par = list(range(seq_layers, L))

        print(f"\n{'='*60}")
        print(f"n_par={n_par} (seq_layers={seq_layers})")
        print(f"{'='*60}")

        pw = backend.precompute_weights(par)
        chunk_configs = build_chunk_configs(backend, par, n_par)
        print(f"Chunk configs: {list(chunk_configs.keys())}")

        npar_key = f'n_par={n_par}'
        results[npar_key] = {}

        for K in K_values:
            for init in init_strategies:
                key = f'IDN_batched_K={K}_{init}'
                losses = []
                for batch in batches:
                    idx, tgt = batch[:, :-1], batch[:, 1:]
                    logits = forward_idn_batched(
                        backend, idx, seq_layers, pw, K=K, init=init)
                    losses.append(F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
                ppl = math.exp(sum(losses) / len(losses))
                _K, _init = K, init
                ms = bench(lambda: forward_idn_batched(
                    backend, idx_bench, seq_layers, pw, K=_K, init=_init), device)
                speed = seq_ms / ms
                dppl = 100 * (ppl - seq_ppl) / seq_ppl
                print(f"  {key}: PPL={ppl:.2f} ({dppl:+.1f}%) {ms:.2f}ms ({speed:.2f}x)")
                results[npar_key][key] = {'ppl': ppl, 'ms': ms, 'speedup': speed}

            for cname, cws in chunk_configs.items():
                if len(cws) <= 1:
                    continue
                for init in init_strategies:
                    key = f'ChunkB_{cname}_K={K}_{init}'
                    losses = []
                    for batch in batches:
                        idx, tgt = batch[:, :-1], batch[:, 1:]
                        logits = forward_chunkwise_batched(
                            backend, idx, seq_layers, cws, K=K, init=init)
                        losses.append(F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item())
                    ppl = math.exp(sum(losses) / len(losses))
                    _K, _init, _cws = K, init, cws
                    ms = bench(lambda: forward_chunkwise_batched(
                        backend, idx_bench, seq_layers, _cws, K=_K, init=_init), device)
                    speed = seq_ms / ms
                    dppl = 100 * (ppl - seq_ppl) / seq_ppl
                    print(f"  {key}: PPL={ppl:.2f} ({dppl:+.1f}%) {ms:.2f}ms ({speed:.2f}x)")
                    results[npar_key][key] = {'ppl': ppl, 'ms': ms, 'speedup': speed}

    outpath = args.output or f'ots_eval_{model.config.model_type}.json'
    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else '.', exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
