"""
Evaluate autoregressive SNLP agreement metrics for selected inference configs.

Metrics:
  - ar_match: token match rate between the SNLP AR sequence and the matching
    sequential reference for the approximated model. For ChunkB this reference
    runs fused chunks sequentially. For non-fused IDN/mHC methods it is the
    original model's sequential forward.
  - emb_sim: sentence-embedding cosine similarity between the SNLP AR
    continuation and the original model's sequential continuation.

Example:
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache_old PYTHONUNBUFFERED=1 \
  uv run python -m snlp.eval_snlp_ar \
  --model-tag d32s_idn05_npar24_s3 \
  --n-par 24 --method ChunkB --chunks 12 --K 2 --init h0 \
  --n-samples 128 --prefill-len 16 --gen-len 64 \
  --output cache/eval_logs_ar/d32s_idn05_npar24_s3_a2.json

Multiple configs can be evaluated with --configs-json. The file should contain
either a list of config objects, or {"configs": [...]} where each config has
the same fields as the CLI flags: n_par, method, K, init, chunks, jvp_method.
"""

import argparse
import json
import os
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
    _embed,
    _fused_forward_chunk,
    _logits,
    _run_block,
    forward_idn_batched,
    forward_idn_chunkwise,
    forward_idn_fused,
    forward_idn_fused_split,
    forward_sequential,
    load_model,
)
from snlp.inference_hcn import (
    MHCChunkWeights,
    MHCPrecomputedWeights,
    _batched_chunks_forward_mhc,
    _embed_mhc,
    _logits_mhc,
    _run_block_mhc,
    forward_hcn_batched,
    forward_hcn_chunkwise,
    forward_sequential as forward_sequential_mhc,
)
from snlp.inference_diagn import forward_diagn_batched


device = torch.device(autodetect_device_type())


def _parse_cutoffs(text, default_max):
    if text is None:
        return [default_max]
    vals = sorted({int(x) for x in text.split(",") if x.strip()})
    if not vals or vals[-1] > default_max:
        raise ValueError(f"cutoffs must be nonempty and <= gen_len={default_max}: {text}")
    return vals


def load_ar_samples(device, n_samples, prefill_len, gen_len, shard):
    """Load deterministic contiguous AR prompts from the validation stream."""
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
    needed = seq_len * n_samples
    all_tokens = []
    for text in table.column("text").to_pylist():
        all_tokens.extend(tokenizer.encode(text, prepend=bos))
        if len(all_tokens) >= needed:
            break

    samples = []
    offset = 0
    while len(samples) < n_samples and offset + seq_len <= len(all_tokens):
        samples.append(all_tokens[offset:offset + seq_len])
        offset += seq_len
    if len(samples) < n_samples:
        raise RuntimeError(f"only loaded {len(samples)} samples; requested {n_samples}")
    return samples


def _call_forward(fn, idx, requires_grad=False):
    if requires_grad:
        with torch.enable_grad():
            return fn(idx)
    with torch.no_grad():
        return fn(idx)


def ar_generate_no_kv(forward_fn, prompt_ids, max_new, requires_grad=False):
    """Greedy AR generation that recomputes the full prefix every step."""
    gen = []
    all_ids = list(prompt_ids)
    for _ in range(max_new):
        idx = torch.tensor([all_ids], device=device, dtype=torch.long)
        logits = _call_forward(forward_fn, idx, requires_grad=requires_grad)
        tok = int(logits[0, -1].argmax().item())
        gen.append(tok)
        all_ids.append(tok)
    return gen


@torch.inference_mode()
def forward_chunkwise_sequential(model, idx, seq_layers, chunk_weights_list):
    """Sequential reference for a fused ChunkB model."""
    L = model.config.n_layer
    x0, cos_sin, ve = _embed(model, idx)
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, ve)
        h[i] = x
    for cw in chunk_weights_list:
        x = _fused_forward_chunk(x, x0, cos_sin, cw, model)
        for li in cw.layers:
            h[li] = x
    return _logits(model, h)


@torch.inference_mode()
def forward_chunkwise_sequential_mhc(model, idx, seq_layers, chunk_weights_list):
    """Sequential reference for a fused mHC ChunkB model."""
    L = model.config.n_layer
    x0, cos_sin, ve = _embed_mhc(model, idx)
    h = [None] * L
    x = x0
    for i in range(seq_layers):
        x = _run_block_mhc(model, i, x, x0, cos_sin, ve)
        h[i] = x
    for cw in chunk_weights_list:
        out = _batched_chunks_forward_mhc(x.unsqueeze(0), cos_sin, [cw], model)
        x = out[0]
        for li in cw.layers:
            h[li] = x
    return _logits_mhc(model, h)


def build_forward_fns(model, cfg):
    """Return (method_forward, match_reference_forward, metadata)."""
    n_par = int(cfg["n_par"])
    method = cfg["method"]
    K = int(cfg.get("K", 1))
    init = cfg.get("init", "h0")
    chunks = cfg.get("chunks")
    jvp_method = cfg.get("jvp_method", "vjp")

    L = model.config.n_layer
    seq_layers = L - n_par
    par = list(range(seq_layers, L))
    use_mhc = getattr(model.config, "use_mhc", False)

    if use_mhc:
        pw = MHCPrecomputedWeights(model, par)
        seq_fn = lambda idx: forward_sequential_mhc(model, idx)
    else:
        pw = PrecomputedWeights(model, par)
        seq_fn = lambda idx: forward_sequential(model, idx)

    chunk_label = None
    if method == "IDN_batched":
        method_fn = lambda idx: forward_idn_batched(model, idx, seq_layers, pw, K=K, init=init)
        match_ref_fn = seq_fn
    elif method == "Fused":
        cwl = [ChunkWeights(model, par)]
        method_fn = lambda idx: forward_idn_fused(model, idx, seq_layers, pw, cwl, K=K)
        match_ref_fn = method_fn
        chunk_label = f"1xF{n_par}"
    elif method == "Fused+split":
        method_fn = lambda idx: forward_idn_fused_split(model, idx, seq_layers, pw, K=K, init=init)
        match_ref_fn = seq_fn
    elif method == "ChunkB":
        if chunks is None:
            raise ValueError("ChunkB requires chunks")
        chunks = int(chunks)
        cs = n_par // chunks
        if use_mhc:
            cwl = [
                MHCChunkWeights(model, list(range(seq_layers + i * cs, seq_layers + (i + 1) * cs)))
                for i in range(chunks)
            ]
            method_fn = lambda idx: forward_hcn_chunkwise(model, idx, seq_layers, n_par, cwl, pw, K=K, init=init)
            match_ref_fn = lambda idx: forward_chunkwise_sequential_mhc(model, idx, seq_layers, cwl)
        else:
            cwl = [ChunkWeights(model, par[i * cs:(i + 1) * cs]) for i in range(chunks)]
            method_fn = lambda idx: forward_idn_chunkwise(model, idx, seq_layers, pw, cwl, K=K, init=init)
            match_ref_fn = lambda idx: forward_chunkwise_sequential(model, idx, seq_layers, cwl)
        chunk_label = f"{chunks}xF{cs}"
    elif method == "mHC-Newton":
        method_fn = lambda idx: forward_hcn_batched(model, idx, seq_layers=seq_layers, pw=pw, K=K, init=init)
        match_ref_fn = seq_fn
    elif method == "ChunkB_mHC":
        if chunks is None:
            raise ValueError("ChunkB_mHC requires chunks")
        chunks = int(chunks)
        cs = n_par // chunks
        cwl = [
            MHCChunkWeights(model, list(range(seq_layers + i * cs, seq_layers + (i + 1) * cs)))
            for i in range(chunks)
        ]
        method_fn = lambda idx: forward_hcn_chunkwise(model, idx, seq_layers, n_par, cwl, pw, K=K, init=init)
        match_ref_fn = lambda idx: forward_chunkwise_sequential_mhc(model, idx, seq_layers, cwl)
        chunk_label = f"{chunks}xF{cs}"
    elif method == "DiagN":
        method_fn = lambda idx: forward_diagn_batched(
            model, idx, seq_layers, pw, K=K, jvp_method=jvp_method, init=init)
        match_ref_fn = seq_fn
    else:
        raise ValueError(f"unknown method: {method}")

    meta = {
        "n_par": n_par,
        "method": method,
        "K": K,
        "init": init,
        "chunks": chunks,
        "chunk_label": chunk_label,
        "jvp_method": jvp_method if method == "DiagN" else None,
        "match_reference": (
            "sequential_fused_chunk" if method in {"ChunkB", "ChunkB_mHC"}
            else "method_self" if method == "Fused"
            else "sequential_original"
        ),
        "requires_grad": method == "DiagN",
    }
    return method_fn, match_ref_fn, seq_fn, meta


class TransformerEmbedder:
    """Small dependency-light sentence embedder using transformers mean pooling."""

    def __init__(self, model_name, device_name="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device_name)
        self.model.eval()
        self.device = torch.device(device_name)

    @torch.inference_mode()
    def encode(self, texts, batch_size=32):
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outs.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(outs, dim=0)


def compute_embedding_similarity(texts_a, texts_b, embedder, batch_size):
    emb_a = embedder.encode(texts_a, batch_size=batch_size)
    emb_b = embedder.encode(texts_b, batch_size=batch_size)
    sims = F.cosine_similarity(emb_a, emb_b, dim=-1)
    return float(sims.mean().item()), [float(x) for x in sims.tolist()]


def compute_top1_cossim(ref_fn, method_fn, samples, max_batches, method_requires_grad=False):
    """Logit top-1 agreement and cosine similarity on fixed sample contexts."""
    top1s = []
    cossims = []
    for sample in samples[:max_batches]:
        idx = torch.tensor([sample[:-1]], device=device, dtype=torch.long)
        logits_ref = _call_forward(ref_fn, idx, requires_grad=False)
        logits_method = _call_forward(method_fn, idx, requires_grad=method_requires_grad)
        ref = logits_ref.reshape(-1, logits_ref.size(-1)).float()
        test = logits_method.reshape(-1, logits_method.size(-1)).float()
        top1s.append((ref.argmax(-1) == test.argmax(-1)).float().mean().item())
        cos = F.cosine_similarity(ref, test, dim=-1)
        cossims.append(cos[~cos.isnan()].mean().item() if not cos.isnan().all() else 0.0)
    return sum(top1s) / len(top1s), sum(cossims) / len(cossims)


def _decode_continuations(tokenizer, prompts, gens, cutoff):
    texts = []
    for prompt, gen in zip(prompts, gens):
        p = tokenizer.decode(prompt)
        t = tokenizer.decode(gen[:cutoff])
        texts.append(f'Continuation of "{p}": {t}')
    return texts


def evaluate_config(
    model,
    tokenizer,
    samples,
    prompts,
    seq_gens,
    cfg,
    ar_cutoffs,
    emb_cutoffs,
    embedder,
    emb_batch_size,
    top1_batches,
):
    method_fn, match_ref_fn, seq_fn, meta = build_forward_fns(model, cfg)

    gen_len = max(ar_cutoffs + emb_cutoffs)

    print("    top1/cos_sim...", flush=True)
    top1, cos_sim = compute_top1_cossim(
        seq_fn, method_fn, samples, top1_batches, method_requires_grad=meta["requires_grad"])

    print("    method generation...", flush=True)
    method_gens = [
        ar_generate_no_kv(method_fn, p, gen_len, requires_grad=meta["requires_grad"])
        for p in prompts
    ]

    if meta["match_reference"] == "sequential_original":
        match_ref_top1, match_ref_cos_sim = top1, cos_sim
        match_refs = seq_gens
    elif meta["match_reference"] == "method_self":
        match_ref_top1, match_ref_cos_sim = 1.0, 1.0
        match_refs = method_gens
    else:
        print("    match-reference refs...", flush=True)
        match_ref_top1, match_ref_cos_sim = compute_top1_cossim(
            match_ref_fn, method_fn, samples, top1_batches,
            method_requires_grad=meta["requires_grad"])
        match_refs = [ar_generate_no_kv(match_ref_fn, p, gen_len) for p in prompts]

    ar_match = {}
    for cutoff in ar_cutoffs:
        matches = sum(
            sum(1 for a, b in zip(gen[:cutoff], ref[:cutoff]) if a == b)
            for gen, ref in zip(method_gens, match_refs)
        )
        total = len(method_gens) * cutoff
        ar_match[str(cutoff)] = matches / total

    emb_sim = {}
    emb_sim_per_sample = {}
    for cutoff in emb_cutoffs:
        method_texts = _decode_continuations(tokenizer, prompts, method_gens, cutoff)
        seq_texts = _decode_continuations(tokenizer, prompts, seq_gens, cutoff)
        mean_sim, per_sample = compute_embedding_similarity(
            method_texts, seq_texts, embedder, batch_size=emb_batch_size)
        emb_sim[str(cutoff)] = mean_sim
        emb_sim_per_sample[str(cutoff)] = per_sample

    return {
        **meta,
        "top1": top1,
        "cos_sim": cos_sim,
        "top1_reference": "sequential_original",
        "match_ref_top1": match_ref_top1,
        "match_ref_cos_sim": match_ref_cos_sim,
        "ar_match": ar_match,
        "emb_sim": emb_sim,
        "emb_sim_per_sample": emb_sim_per_sample,
    }


def load_configs(args):
    if args.configs_json:
        with open(args.configs_json) as f:
            data = json.load(f)
        configs = data["configs"] if isinstance(data, dict) else data
        if not isinstance(configs, list):
            raise ValueError("--configs-json must contain a list or {'configs': list}")
        return configs
    return [{
        "n_par": args.n_par,
        "method": args.method,
        "K": args.K,
        "init": args.init,
        "chunks": args.chunks,
        "jvp_method": args.jvp_method,
    }]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--n-par", type=int)
    parser.add_argument("--method", choices=[
        "IDN_batched", "ChunkB", "mHC-Newton", "ChunkB_mHC",
        "Fused", "Fused+split", "DiagN",
    ])
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--init", default="h0", choices=["h0", "batch_fwd"])
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--jvp-method", default="vjp", choices=["fd", "vjp"])
    parser.add_argument("--configs-json")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--ar-cutoffs", default="8,16,32,64")
    parser.add_argument("--emb-cutoffs", default="32,64")
    parser.add_argument("--shard", default="shard_06542.parquet")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--top1-batches", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    configs = load_configs(args)
    for cfg in configs:
        if cfg.get("n_par") is None or cfg.get("method") is None:
            raise ValueError("each config requires n_par and method")

    ar_cutoffs = _parse_cutoffs(args.ar_cutoffs, args.gen_len)
    emb_cutoffs = _parse_cutoffs(args.emb_cutoffs, args.gen_len)

    tokenizer = get_tokenizer()
    samples = load_ar_samples(device, args.n_samples, args.prefill_len, args.gen_len, args.shard)
    print(f"loaded {len(samples)} AR samples from {args.shard}", flush=True)

    model = load_model(args.model_tag, device=device, cache_hc=True)
    use_mhc = getattr(model.config, "use_mhc", False)
    no_x0 = getattr(model.config, "no_x0_resid", False)
    model_type = "mhc" if use_mhc else ("nox0ve" if no_x0 else "standard")
    print(f"model={args.model_tag} type={model_type}", flush=True)

    print(f"loading embedding model {args.embedding_model} on {args.embedding_device}", flush=True)
    embedder = TransformerEmbedder(args.embedding_model, device_name=args.embedding_device)

    gen_len = max(ar_cutoffs + emb_cutoffs)
    prompts = [s[:args.prefill_len] for s in samples]
    seq_fn = (lambda idx: forward_sequential_mhc(model, idx)) if use_mhc else (lambda idx: forward_sequential(model, idx))
    print("computing original sequential AR refs once...", flush=True)
    seq_gens = [ar_generate_no_kv(seq_fn, p, gen_len) for p in prompts]

    results = {
        "model": args.model_tag,
        "model_type": model_type,
        "n_samples": args.n_samples,
        "prefill_len": args.prefill_len,
        "gen_len": args.gen_len,
        "ar_cutoffs": ar_cutoffs,
        "emb_cutoffs": emb_cutoffs,
        "embedding_model": args.embedding_model,
        "configs": [],
    }

    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg}", flush=True)
        results["configs"].append(evaluate_config(
            model, tokenizer, samples, prompts, seq_gens, cfg, ar_cutoffs, emb_cutoffs,
            embedder, args.embedding_batch_size, args.top1_batches,
        ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
