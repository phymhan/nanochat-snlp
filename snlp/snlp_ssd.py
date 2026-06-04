"""
Self-Speculative Decoding (SSD) with SNLP drafter and sequential verifier.

Drafter: SNLP forward (layer-parallel, fast approximate)
Verifier: Sequential forward (exact, same model weights)
No KV cache — both forward functions recompute from scratch.

Supports greedy (T=0) and sampling (T>0) modes with temperature sweep.

Usage:
  CUDA_VISIBLE_DEVICES=6 NANOCHAT_BASE_DIR=cache_old PYTHONUNBUFFERED=1 \\
    uv run python -m snlp.snlp_ssd \\
    --model-tag d32s_idn00625_npar24_s3 \\
    --n-par 24 --method IDN_batched --K 2 --init h0 \\
    --block-sizes 2,4,8,16,32 \\
    --temperatures 0.0,0.3,0.5,0.7,1.0 \\
    --n-samples 512 --prefill-len 16 --gen-len 1024 \\
    --output cache/eval_logs_ssd/idn00625_K2.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nanochat.common import autodetect_device_type
from nanochat.tokenizer import get_tokenizer

import nanochat.flash_attention as fa
fa._override_impl = "sdpa"
fa.USE_FA3 = False

from snlp.inference_idn import (
    ChunkWeights,
    PrecomputedWeights,
    forward_idn_batched,
    forward_idn_chunkwise,
    forward_idn_fused,
    forward_sequential,
    load_model,
    load_val_data,
    bench,
)
from snlp.inference_hcn import (
    MHCChunkWeights,
    MHCPrecomputedWeights,
    forward_hcn_batched,
    forward_hcn_chunkwise,
    forward_sequential as forward_sequential_mhc,
)

device = torch.device(autodetect_device_type())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ar_samples(n_samples, prefill_len, gen_len, shard="shard_06542.parquet",
                    sample_offset=0):
    """Load deterministic contiguous AR prompts from validation stream."""
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir is None:
        raise RuntimeError("NANOCHAT_BASE_DIR must be set")
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    table = pq.read_table(
        os.path.join(base_dir, "base_data_climbmix", shard),
        columns=["text"],
    )
    seq_len = prefill_len + gen_len
    total_needed = sample_offset + n_samples
    needed = seq_len * total_needed
    all_tokens = []
    for text in table.column("text").to_pylist():
        all_tokens.extend(tokenizer.encode(text, prepend=bos))
        if len(all_tokens) >= needed:
            break
    all_samples = []
    offset = 0
    while len(all_samples) < total_needed and offset + seq_len <= len(all_tokens):
        all_samples.append(all_tokens[offset : offset + seq_len])
        offset += seq_len
    if len(all_samples) < total_needed:
        raise RuntimeError(f"only {len(all_samples)} samples, need {total_needed}")
    samples = all_samples[sample_offset : sample_offset + n_samples]
    return samples


# ---------------------------------------------------------------------------
# Forward function builders
# ---------------------------------------------------------------------------

def build_fns(model, cfg):
    """Build draft_fn (SNLP) and verify_fn (sequential)."""
    n_par = int(cfg["n_par"])
    method = cfg["method"]
    K = int(cfg.get("K", 1))
    init = cfg.get("init", "h0")
    chunks = cfg.get("chunks")

    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    use_mhc = getattr(model.config, "use_mhc", False)

    if use_mhc:
        pw = MHCPrecomputedWeights(model, par)
        verify_fn = lambda idx: forward_sequential_mhc(model, idx)
    else:
        pw = PrecomputedWeights(model, par)
        verify_fn = lambda idx: forward_sequential(model, idx)

    if method == "IDN_batched":
        draft_fn = lambda idx: forward_idn_batched(
            model, idx, seq_layers, pw, K=K, init=init
        )
    elif method == "ChunkB":
        if chunks is None:
            raise ValueError("ChunkB requires chunks")
        chunks = int(chunks)
        cs = n_par // chunks
        if use_mhc:
            cwl = [
                MHCChunkWeights(
                    model,
                    list(range(seq_layers + i * cs, seq_layers + (i + 1) * cs)),
                )
                for i in range(chunks)
            ]
            draft_fn = lambda idx: forward_hcn_chunkwise(
                model, idx, seq_layers, n_par, cwl, pw, K=K, init=init
            )
        else:
            cwl = [
                ChunkWeights(model, par[i * cs : (i + 1) * cs])
                for i in range(chunks)
            ]
            draft_fn = lambda idx: forward_idn_chunkwise(
                model, idx, seq_layers, pw, cwl, K=K, init=init
            )
    elif method == "Fused":
        cwl = [ChunkWeights(model, par)]
        draft_fn = lambda idx: forward_idn_fused(
            model, idx, seq_layers, pw, cwl, K=K
        )
    elif method == "mHC-Newton":
        draft_fn = lambda idx: forward_hcn_batched(
            model, idx, seq_layers=seq_layers, pw=pw, K=K, init=init
        )
    else:
        raise ValueError(f"unknown method: {method}")

    return draft_fn, verify_fn


# ---------------------------------------------------------------------------
# PPL sanity check
# ---------------------------------------------------------------------------

def sanity_check_ppl(draft_fn, verify_fn, seq_len=2048):
    """Compute PPL for both sequential and SNLP forward at given seq_len."""
    batches = load_val_data(device, max_tokens=1_000_000, seq_len=seq_len)
    print(f"  PPL sanity check: {len(batches)} batches, seq_len={seq_len}")

    def _ppl(fn, batches):
        losses = []
        for batch in batches:
            idx, tgt = batch[:, :-1], batch[:, 1:]
            with torch.inference_mode():
                logits = fn(idx)
            losses.append(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
                ).item()
            )
        return math.exp(min(sum(losses) / len(losses), 20))

    seq_ppl = _ppl(verify_fn, batches)
    snlp_ppl = _ppl(draft_fn, batches)
    dppl = 100 * (snlp_ppl - seq_ppl) / seq_ppl
    print(
        f"  seq_ppl={seq_ppl:.2f}  snlp_ppl={snlp_ppl:.2f}  delta={dppl:+.1f}%"
    )
    return {"seq_ppl": seq_ppl, "snlp_ppl": snlp_ppl, "delta_ppl_pct": dppl}


# ---------------------------------------------------------------------------
# Core speculative decoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def speculative_decode(draft_fn, verify_fn, prompt_ids, max_new, block_size,
                       temperature=0.0):
    """
    Self-speculative decoding: SNLP drafts, sequential verifies.

    temperature=0: greedy (argmax, deterministic accept/reject)
    temperature>0: sampling (softmax/T, probabilistic accept/reject)

    Returns (gen_tokens, stats).
    """
    all_ids = list(prompt_ids)
    gen = []
    total_accepted = 0
    total_drafted = 0
    n_rounds = 0
    bonus_tokens = 0
    accept_counts = []  # per-round accepted count

    greedy = temperature == 0.0

    while len(gen) < max_new:
        remaining = max_new - len(gen)
        B = min(block_size, remaining)

        # --- Draft phase: generate B tokens ---
        draft_ids = list(all_ids)
        drafts = []
        draft_probs = []  # p(d_j) under draft distribution (sampling only)

        for _ in range(B):
            idx = torch.tensor([draft_ids], device=device, dtype=torch.long)
            logits_d = draft_fn(idx)
            logits_last = logits_d[0, -1].float()

            if greedy:
                tok = int(logits_last.argmax().item())
            else:
                probs = F.softmax(logits_last / temperature, dim=-1)
                tok = int(torch.multinomial(probs, 1).item())
                draft_probs.append(probs)

            drafts.append(tok)
            draft_ids.append(tok)

        total_drafted += B

        # --- Verify phase: single forward on prefix + all drafts ---
        verify_ids = all_ids + drafts
        idx_v = torch.tensor([verify_ids], device=device, dtype=torch.long)
        logits_v = verify_fn(idx_v)

        prefix_len = len(all_ids)
        n_accepted = 0

        if greedy:
            # Deterministic: accept while argmax matches
            for j in range(B):
                v_tok = int(logits_v[0, prefix_len - 1 + j].argmax().item())
                if drafts[j] == v_tok:
                    n_accepted += 1
                else:
                    break
        else:
            # Probabilistic: accept with min(1, q/p)
            for j in range(B):
                q_logits = logits_v[0, prefix_len - 1 + j].float()
                q_probs = F.softmax(q_logits / temperature, dim=-1)
                p_d = draft_probs[j][drafts[j]].item()
                q_d = q_probs[drafts[j]].item()

                if p_d == 0:
                    # Draft assigned 0 probability — reject
                    # Sample from verifier distribution as replacement
                    replacement = int(torch.multinomial(q_probs, 1).item())
                    drafts[j] = replacement  # store for the rejection case
                    break

                ratio = q_d / p_d
                if ratio >= 1.0 or torch.rand(1).item() < ratio:
                    n_accepted += 1
                else:
                    # Reject: sample from adjusted distribution max(0, q - p)
                    p_probs = draft_probs[j]
                    adjusted = (q_probs - p_probs).clamp(min=0)
                    adj_sum = adjusted.sum()
                    if adj_sum > 0:
                        adjusted = adjusted / adj_sum
                        replacement = int(torch.multinomial(adjusted, 1).item())
                    else:
                        replacement = int(torch.multinomial(q_probs, 1).item())
                    drafts[j] = replacement
                    break

        accept_counts.append(n_accepted)

        # Append accepted drafts
        for j in range(n_accepted):
            gen.append(drafts[j])
            all_ids.append(drafts[j])
        total_accepted += n_accepted

        if len(gen) >= max_new:
            break

        if n_accepted == B:
            # All accepted — bonus token from verifier's last position
            bonus_logits = logits_v[0, prefix_len + B - 1].float()
            if greedy:
                bonus_tok = int(bonus_logits.argmax().item())
            else:
                bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                bonus_tok = int(torch.multinomial(bonus_probs, 1).item())
            gen.append(bonus_tok)
            all_ids.append(bonus_tok)
            bonus_tokens += 1
        else:
            # Use verifier's token at rejection point
            if greedy:
                v_tok = int(
                    logits_v[0, prefix_len - 1 + n_accepted].argmax().item()
                )
            else:
                # drafts[n_accepted] was already set to the replacement above
                v_tok = drafts[n_accepted]
            gen.append(v_tok)
            all_ids.append(v_tok)

        n_rounds += 1

    gen = gen[:max_new]
    alpha = total_accepted / total_drafted if total_drafted > 0 else 0.0

    return gen, {
        "acceptance_rate": alpha,
        "total_accepted": total_accepted,
        "total_drafted": total_drafted,
        "n_rounds": n_rounds,
        "bonus_tokens": bonus_tokens,
        "avg_tokens_per_round": len(gen) / n_rounds if n_rounds else 0,
        "accept_counts": accept_counts,
    }


# ---------------------------------------------------------------------------
# Theoretical speedup
# ---------------------------------------------------------------------------

def theoretical_speedup(alpha, B, s):
    """
    Compute theoretical SD speedup over sequential.

    E[tokens/round] = (1 - alpha^{B+1}) / (1 - alpha)
    cost/round = B/s + 1
    speedup = E / cost

    alpha: per-token acceptance rate
    B: block size (number of draft tokens)
    s: forward speedup of SNLP over sequential
    """
    if alpha >= 1.0:
        e_tokens = B + 1
    else:
        e_tokens = (1 - alpha ** (B + 1)) / (1 - alpha)
    cost = B / s + 1
    return e_tokens / cost


# ---------------------------------------------------------------------------
# Correctness check (greedy only)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def verify_correctness(verify_fn, prompt_ids, gen_tokens, check_len=64):
    """
    Verify SSD output matches sequential greedy by running sequential
    generation step-by-step and comparing token-by-token.

    Only checks the first check_len tokens to keep it fast.
    Only valid for greedy mode (T=0).
    """
    n_check = min(check_len, len(gen_tokens))
    all_ids = list(prompt_ids)
    for i in range(n_check):
        idx = torch.tensor([all_ids], device=device, dtype=torch.long)
        logits = verify_fn(idx)
        pred = int(logits[0, -1].argmax().item())
        if pred != gen_tokens[i]:
            return False, i
        all_ids.append(gen_tokens[i])
    return True, n_check


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(args):
    """Load configs from CLI args or JSON file."""
    if args.configs_json:
        with open(args.configs_json) as f:
            data = json.load(f)
        configs = data["configs"] if isinstance(data, dict) else data
        return configs
    return [
        {
            "n_par": args.n_par,
            "method": args.method,
            "K": args.K,
            "init": args.init,
            "chunks": args.chunks,
        }
    ]


def config_label(cfg):
    """Short human-readable label for a config."""
    method = cfg["method"]
    n_par = cfg["n_par"]
    K = cfg.get("K", 1)
    init = cfg.get("init", "h0")
    chunks = cfg.get("chunks")
    label = f"{method} n{n_par} K{K} {init}"
    if chunks is not None:
        cs = n_par // int(chunks)
        label = f"{method}_{chunks}xF{cs} n{n_par} K{K} {init}"
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-Speculative Decoding with SNLP drafter"
    )

    # Model
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--cache-hc", action="store_true")

    # Config (single or batch)
    parser.add_argument("--n-par", type=int)
    parser.add_argument(
        "--method",
        choices=["IDN_batched", "ChunkB", "Fused", "mHC-Newton"],
    )
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--init", default="h0")
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--configs-json")

    # SSD parameters
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated block sizes B",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.0,0.3,0.5,0.7,1.0",
        help="Comma-separated temperatures to sweep",
    )

    # Data
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--gen-len", type=int, default=1024)
    parser.add_argument("--shard", default="shard_06542.parquet")
    parser.add_argument("--sample-offset", type=int, default=0,
                        help="Skip first N samples (for multi-GPU parallelism)")

    # Options
    parser.add_argument(
        "--correctness-check",
        type=int,
        default=3,
        help="Number of samples for greedy correctness check (0 to skip)",
    )
    parser.add_argument(
        "--forward-speedup",
        type=float,
        default=None,
        help="Override auto-measured forward speedup s",
    )
    parser.add_argument("--ppl-seq-len", type=int, default=2048)

    # Output
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    configs = load_configs(args)
    for cfg in configs:
        if cfg.get("n_par") is None or cfg.get("method") is None:
            raise ValueError("each config requires n_par and method")

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    temperatures = [float(x) for x in args.temperatures.split(",")]

    # Load model
    model = load_model(args.model_tag, device=device, cache_hc=args.cache_hc)
    use_mhc = getattr(model.config, "use_mhc", False)
    no_x0 = getattr(model.config, "no_x0_resid", False)
    model_type = "mhc" if use_mhc else ("nox0ve" if no_x0 else "standard")
    print(
        f"Model: {args.model_tag}  type={model_type}  L={model.config.n_layer}",
        flush=True,
    )

    # Load AR samples
    samples = load_ar_samples(
        args.n_samples, args.prefill_len, args.gen_len, args.shard,
        sample_offset=args.sample_offset,
    )
    prompts = [s[: args.prefill_len] for s in samples]
    print(f"Loaded {len(prompts)} prompts (prefill={args.prefill_len}, offset={args.sample_offset})", flush=True)

    results = {
        "model": args.model_tag,
        "model_type": model_type,
        "n_samples": args.n_samples,
        "prefill_len": args.prefill_len,
        "gen_len": args.gen_len,
        "block_sizes": block_sizes,
        "temperatures": temperatures,
        "configs": [],
    }

    for ci, cfg in enumerate(configs, start=1):
        label = config_label(cfg)
        print(f"\n{'='*60}", flush=True)
        print(f"[Config {ci}/{len(configs)}] {label}", flush=True)
        print(f"{'='*60}", flush=True)

        draft_fn, verify_fn = build_fns(model, cfg)

        # PPL sanity check
        ppl_info = sanity_check_ppl(
            draft_fn, verify_fn, seq_len=args.ppl_seq_len
        )

        # Measure forward speedup
        if args.forward_speedup is not None:
            s = args.forward_speedup
            bench_info = {"speedup": s, "source": "user_override"}
            print(f"  Forward speedup (override): {s:.2f}x", flush=True)
        else:
            bench_data = load_val_data(device, max_tokens=128 * 200, seq_len=128)
            bench_idx = bench_data[0][:, :-1]
            draft_ms = bench(lambda: draft_fn(bench_idx), device, n_warm=20, n_run=50)
            verify_ms = bench(
                lambda: verify_fn(bench_idx), device, n_warm=20, n_run=50
            )
            s = verify_ms / draft_ms
            bench_info = {
                "draft_ms": round(draft_ms, 2),
                "verify_ms": round(verify_ms, 2),
                "speedup": round(s, 3),
                "bench_seq_len": 128,
                "source": "auto",
            }
            print(
                f"  Forward speedup: draft={draft_ms:.2f}ms  "
                f"verify={verify_ms:.2f}ms  s={s:.2f}x",
                flush=True,
            )

        config_results = {
            **cfg,
            "label": label,
            "ppl_sanity": ppl_info,
            "forward_speedup": bench_info,
            "temperature_results": {},
        }

        for T in temperatures:
            greedy = T == 0.0
            mode = "greedy" if greedy else f"T={T}"
            print(f"\n  --- Temperature {T} ({mode}) ---", flush=True)

            temp_results = {}

            for B in block_sizes:
                print(f"    B={B}:", end="", flush=True)
                all_alphas = []
                all_tokens_per_round = []
                t_start = time.time()

                for si, prompt in enumerate(prompts):
                    gen, stats = speculative_decode(
                        draft_fn, verify_fn, prompt, args.gen_len, B,
                        temperature=T,
                    )
                    all_alphas.append(stats["acceptance_rate"])
                    all_tokens_per_round.append(stats["avg_tokens_per_round"])

                    # Greedy correctness check on first N samples
                    if greedy and si < args.correctness_check:
                        ok, pos = verify_correctness(verify_fn, prompt, gen)
                        if not ok:
                            print(
                                f" CORRECTNESS FAIL at sample {si} pos {pos}!",
                                flush=True,
                            )

                    if (si + 1) % 100 == 0:
                        elapsed = time.time() - t_start
                        running_alpha = sum(all_alphas) / len(all_alphas)
                        print(
                            f" [{si+1}/{len(prompts)} "
                            f"alpha={running_alpha:.3f} "
                            f"{elapsed:.0f}s]",
                            end="",
                            flush=True,
                        )

                elapsed = time.time() - t_start
                mean_alpha = sum(all_alphas) / len(all_alphas)
                std_alpha = (
                    sum((a - mean_alpha) ** 2 for a in all_alphas) / len(all_alphas)
                ) ** 0.5
                mean_tpr = sum(all_tokens_per_round) / len(all_tokens_per_round)
                th_speed = theoretical_speedup(mean_alpha, B, s)

                print(
                    f" alpha={mean_alpha:.4f}+-{std_alpha:.4f}"
                    f" toks/rnd={mean_tpr:.2f}"
                    f" th_speedup={th_speed:.3f}x"
                    f" {elapsed:.0f}s",
                    flush=True,
                )

                temp_results[str(B)] = {
                    "mean_acceptance_rate": round(mean_alpha, 5),
                    "std_acceptance_rate": round(std_alpha, 5),
                    "mean_tokens_per_round": round(mean_tpr, 3),
                    "theoretical_speedup": round(th_speed, 4),
                    "wall_clock_s": round(elapsed, 1),
                    "per_sample_acceptance_rates": [
                        round(a, 4) for a in all_alphas
                    ],
                }

            config_results["temperature_results"][str(T)] = temp_results

        results["configs"].append(config_results)

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
