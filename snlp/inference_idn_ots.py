"""
Unified IDN layer-parallel inference for off-the-shelf HuggingFace models.

Supports:
  - TinyLlama 1.1B (LlamaBackend)
  - Qwen 2.5 0.5B  (QwenBackend)
  - Gemma 3 1B      (GemmaBackend)

Each backend implements model-specific weight stacking, batched/fused forward
passes, and prefix computation.  The public API dispatches through a backend
obtained via ``get_backend(model)``.

Public API
----------
- get_backend(model) -> ModelBackend
- forward_sequential(backend, input_ids)
- forward_idn_batched(backend, input_ids, seq_layers, pw, ...)
- forward_fused(backend, input_ids, seq_layers, chunk_weights, ...)
- forward_chunkwise_batched(backend, input_ids, seq_layers, chunk_weights_list, ...)
- load_val_data(tokenizer, device, ...)
- bench(fn, device, ...)
"""

import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _rms_norm(x, weight, eps):
    """Standard RMSNorm (LLaMA / Qwen style)."""
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * weight).to(x.dtype)


def _rms_norm_gemma(x, weight, eps):
    """Gemma3 RMSNorm: uses (1 + weight) instead of weight."""
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * (1.0 + weight.float())).to(x.dtype)


def _rms_norm_headwise(x, weight, eps):
    """Gemma3 RMSNorm per-head. weight shape: (head_dim,) or (L, head_dim)."""
    x_f = x.float()
    x_norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_norm * (1.0 + weight.float())).to(x.dtype)


def _apply_rope_batched(x, cos, sin):
    """RoPE for batched tensors. x: (L, B, H, T, hd), cos/sin: (1, B, 1, T, hd)."""
    hd_half = x.shape[-1] // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _apply_rope(x, cos, sin):
    """RoPE for single tensors. x: (B, T, H, hd), cos/sin: (B, T, hd)."""
    hd_half = x.shape[-1] // 2
    x1, x2 = x[..., :hd_half], x[..., hd_half:]
    c = cos.unsqueeze(2)
    s = sin.unsqueeze(2)
    return x * c + torch.cat((-x2, x1), dim=-1) * s


def _idn_correction(all_out, hs, h_init, n_units, elk_k=0.0):
    """IDN prefix-sum correction. Updates *hs* in-place."""
    a = 1.0 - elk_k
    h_old = [hs[j + 1].clone() for j in range(n_units)]
    h_corr = all_out[0]
    hs[1] = h_corr
    for j in range(1, n_units):
        h_corr = all_out[j] + a * (h_corr - h_old[j - 1])
        hs[j + 1] = h_corr


def load_val_data(tokenizer, device, seq_len=128, max_tokens=1_000_000):
    """Load wikitext-2 validation data as list of (1, seq_len+1) tensors."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = enc["input_ids"][0]
    batches = []
    offset = 0
    while offset + seq_len + 1 <= len(all_ids) and len(batches) * seq_len < max_tokens:
        batches.append(all_ids[offset:offset + seq_len + 1].unsqueeze(0).to(device))
        offset += seq_len
    return batches


def bench(fn, device, n_warm=50, n_run=150):
    """Benchmark *fn* and return mean wall-clock time in milliseconds."""
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
# Weight containers
# ---------------------------------------------------------------------------

class StackedWeights:
    """Generic container filled by a backend's ``precompute_weights``."""
    def __init__(self):
        self.par = None
        self.n_par = 0


class FusedWeights:
    """Generic container filled by a backend's ``chunk_weights``."""
    def __init__(self):
        self.layers = None
        self.n = 0


# ---------------------------------------------------------------------------
# ModelBackend base class
# ---------------------------------------------------------------------------

class ModelBackend:
    """Abstract base for model-specific SNLP operations."""

    def __init__(self, model):
        self.model = model
        self.config = model.config

    # -- weight building -----------------------------------------------------
    def precompute_weights(self, par):
        raise NotImplementedError

    def chunk_weights(self, chunk_layers):
        raise NotImplementedError

    # -- core forward --------------------------------------------------------
    def run_prefix(self, input_ids, seq_layers):
        raise NotImplementedError

    def batched_forward(self, all_h_prev, pos_emb, pw):
        raise NotImplementedError

    def fused_forward_chunk(self, h_in, pos_emb, cw, avg=False):
        raise NotImplementedError

    def batched_chunks_forward(self, all_h_in, pos_emb, chunk_weights_list, avg=False):
        raise NotImplementedError

    # -- final projection / sequential ---------------------------------------
    def final_proj(self, hidden):
        return self.model.lm_head(self.model.model.norm(hidden)).float()

    def forward_sequential(self, input_ids):
        return self.model(input_ids).logits.float()


# ===========================================================================
# LlamaBackend  (TinyLlama 1.1B)
# ===========================================================================

class LlamaBackend(ModelBackend):

    # -- weights -------------------------------------------------------------
    def precompute_weights(self, par):
        model = self.model
        layers = [model.model.layers[i] for i in par]
        dt = next(model.parameters()).dtype

        pw = StackedWeights()
        pw.W_q = torch.stack([l.self_attn.q_proj.weight for l in layers]).to(dt)
        pw.W_k = torch.stack([l.self_attn.k_proj.weight for l in layers]).to(dt)
        pw.W_v = torch.stack([l.self_attn.v_proj.weight for l in layers]).to(dt)
        pw.W_o = torch.stack([l.self_attn.o_proj.weight for l in layers]).to(dt)

        pw.W_gate = torch.stack([l.mlp.gate_proj.weight for l in layers]).to(dt)
        pw.W_up = torch.stack([l.mlp.up_proj.weight for l in layers]).to(dt)
        pw.W_down = torch.stack([l.mlp.down_proj.weight for l in layers]).to(dt)

        pw.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).to(dt)
        pw.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).to(dt)

        pw.par = par
        pw.n_par = len(par)
        return pw

    def chunk_weights(self, chunk_layers):
        model = self.model
        layers = [model.model.layers[i] for i in chunk_layers]
        dt = next(model.parameters()).dtype
        n = len(chunk_layers)

        cw = FusedWeights()
        cw.W_q = torch.cat([l.self_attn.q_proj.weight for l in layers], dim=0).to(dt)
        cw.W_k = torch.cat([l.self_attn.k_proj.weight for l in layers], dim=0).to(dt)
        cw.W_v = torch.cat([l.self_attn.v_proj.weight for l in layers], dim=0).to(dt)
        cw.W_o = torch.cat([l.self_attn.o_proj.weight for l in layers], dim=1).to(dt)

        cw.W_gate = torch.cat([l.mlp.gate_proj.weight for l in layers], dim=0).to(dt)
        cw.W_up = torch.cat([l.mlp.up_proj.weight for l in layers], dim=0).to(dt)
        cw.W_down = torch.cat([l.mlp.down_proj.weight for l in layers], dim=1).to(dt)

        cw.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).mean(0).to(dt)
        cw.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).mean(0).to(dt)

        cw.layers = chunk_layers
        cw.n = n
        return cw

    # -- prefix --------------------------------------------------------------
    def run_prefix(self, input_ids, seq_layers):
        model = self.model
        hidden = model.model.embed_tokens(input_ids)
        B, T, D = hidden.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        if seq_layers > 0:
            from transformers.masking_utils import create_causal_mask
            cache_position = torch.arange(T, device=input_ids.device)
            causal_mask = create_causal_mask(
                config=model.config, input_embeds=hidden,
                attention_mask=None, cache_position=cache_position,
                past_key_values=None, position_ids=position_ids,
            )
            for i in range(seq_layers):
                layer = model.model.layers[i]
                mask = causal_mask
                if isinstance(causal_mask, dict):
                    mask = causal_mask.get(getattr(layer, 'attention_type', 'full_attention'), None)
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        return hidden, position_embeddings

    # -- batched forward -----------------------------------------------------
    def batched_forward(self, all_h_prev, pos_emb, pw):
        config = self.config
        n_par, B, T, D = all_h_prev.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        hd = D // n_head
        n_groups = n_head // n_kv

        x_norm = _rms_norm(all_h_prev, pw.ln1_w[:, None, None, :], eps)

        all_q = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_q)
        all_k = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_k)
        all_v = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_v)

        all_q = all_q.view(n_par, B, T, n_head, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)

        cos, sin = pos_emb
        cos_r = cos.unsqueeze(0).unsqueeze(2)
        sin_r = sin.unsqueeze(0).unsqueeze(2)
        all_q = _apply_rope_batched(all_q, cos_r, sin_r)
        all_k = _apply_rope_batched(all_k, cos_r, sin_r)

        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)

        q_s = all_q.reshape(n_par * B, n_head, T, hd)
        k_s = all_k.reshape(n_par * B, n_head, T, hd)
        v_s = all_v.reshape(n_par * B, n_head, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

        attn_out = y_s.reshape(n_par, B, n_head, T, hd).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
        attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o)

        hidden = all_h_prev + attn_out

        h_norm = _rms_norm(hidden, pw.ln2_w[:, None, None, :], eps)
        gate = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_gate)
        up = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_up)
        mlp = torch.einsum('lbti,ldi->lbtd', F.silu(gate) * up, pw.W_down)

        return hidden + mlp

    # -- fused forward chunk -------------------------------------------------
    def fused_forward_chunk(self, h_in, pos_emb, cw, avg=False):
        config = self.config
        B, T, D = h_in.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        total_q = cw.n * n_head
        total_kv = cw.n * n_kv
        hd = D // n_head
        n_groups = n_head // n_kv

        x_norm = _rms_norm(h_in, cw.ln1_w, eps)

        q = F.linear(x_norm, cw.W_q).view(B, T, total_q, hd)
        k = F.linear(x_norm, cw.W_k).view(B, T, total_kv, hd)
        v = F.linear(x_norm, cw.W_v).view(B, T, total_kv, hd)

        cos, sin = pos_emb
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if n_groups > 1:
            k = k[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)
            v = v[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        attn_out = F.linear(y, cw.W_o)
        if avg and cw.n > 1:
            attn_out = attn_out / cw.n
        x_mid = h_in + attn_out

        h_norm = _rms_norm(x_mid, cw.ln2_w, eps)
        gate = F.linear(h_norm, cw.W_gate)
        up = F.linear(h_norm, cw.W_up)
        mlp = F.linear(F.silu(gate) * up, cw.W_down)
        if avg and cw.n > 1:
            mlp = mlp / cw.n
        return x_mid + mlp

    # -- batched chunks forward ----------------------------------------------
    def batched_chunks_forward(self, all_h_in, pos_emb, chunk_weights_list, avg=False):
        config = self.config
        C = len(chunk_weights_list)
        chunk_size = chunk_weights_list[0].n
        B, T, D = all_h_in.shape[1:]
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        q_per_chunk = chunk_size * n_head
        kv_per_chunk = chunk_size * n_kv
        hd = D // n_head
        n_groups = n_head // n_kv

        W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
        W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
        W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
        W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
        W_gate = torch.stack([cw.W_gate for cw in chunk_weights_list])
        W_up = torch.stack([cw.W_up for cw in chunk_weights_list])
        W_down = torch.stack([cw.W_down for cw in chunk_weights_list])

        ln1_w = torch.stack([cw.ln1_w for cw in chunk_weights_list])
        ln2_w = torch.stack([cw.ln2_w for cw in chunk_weights_list])

        x_norm = _rms_norm(all_h_in, ln1_w[:, None, None, :], eps)

        all_q = torch.einsum('cbtd,chd->cbth', x_norm, W_q)
        all_k = torch.einsum('cbtd,chd->cbth', x_norm, W_k)
        all_v = torch.einsum('cbtd,chd->cbth', x_norm, W_v)

        all_q = all_q.view(C, B, T, q_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)

        cos, sin = pos_emb
        cos_r = cos.unsqueeze(0).unsqueeze(2)
        sin_r = sin.unsqueeze(0).unsqueeze(2)
        all_q = _apply_rope_batched(all_q, cos_r, sin_r)
        all_k = _apply_rope_batched(all_k, cos_r, sin_r)

        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)

        q_s = all_q.reshape(C * B, q_per_chunk, T, hd)
        k_s = all_k.reshape(C * B, q_per_chunk, T, hd)
        v_s = all_v.reshape(C * B, q_per_chunk, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

        y = y_s.reshape(C, B, q_per_chunk, T, hd).permute(0, 1, 3, 2, 4)
        y = y.reshape(C, B, T, q_per_chunk * hd)

        attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
        if avg and chunk_size > 1:
            attn_out = attn_out / chunk_size

        hidden = all_h_in + attn_out

        h_norm = _rms_norm(hidden, ln2_w[:, None, None, :], eps)
        gate = torch.einsum('cbtd,cid->cbti', h_norm, W_gate)
        up = torch.einsum('cbtd,cid->cbti', h_norm, W_up)
        mlp = torch.einsum('cbti,cdi->cbtd', F.silu(gate) * up, W_down)
        if avg and chunk_size > 1:
            mlp = mlp / chunk_size

        return hidden + mlp


# ===========================================================================
# QwenBackend  (Qwen 2.5 0.5B)
# ===========================================================================

class QwenBackend(ModelBackend):

    # -- weights -------------------------------------------------------------
    def precompute_weights(self, par):
        model = self.model
        layers = [model.model.layers[i] for i in par]
        dt = next(model.parameters()).dtype

        pw = StackedWeights()
        pw.W_q = torch.stack([l.self_attn.q_proj.weight for l in layers]).to(dt)
        pw.W_k = torch.stack([l.self_attn.k_proj.weight for l in layers]).to(dt)
        pw.W_v = torch.stack([l.self_attn.v_proj.weight for l in layers]).to(dt)
        pw.W_o = torch.stack([l.self_attn.o_proj.weight for l in layers]).to(dt)

        pw.b_q = torch.stack([l.self_attn.q_proj.bias for l in layers]).to(dt)
        pw.b_k = torch.stack([l.self_attn.k_proj.bias for l in layers]).to(dt)
        pw.b_v = torch.stack([l.self_attn.v_proj.bias for l in layers]).to(dt)

        pw.W_gate = torch.stack([l.mlp.gate_proj.weight for l in layers]).to(dt)
        pw.W_up = torch.stack([l.mlp.up_proj.weight for l in layers]).to(dt)
        pw.W_down = torch.stack([l.mlp.down_proj.weight for l in layers]).to(dt)

        pw.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).to(dt)
        pw.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).to(dt)

        pw.par = par
        pw.n_par = len(par)
        return pw

    def chunk_weights(self, chunk_layers):
        model = self.model
        layers = [model.model.layers[i] for i in chunk_layers]
        dt = next(model.parameters()).dtype
        n = len(chunk_layers)

        cw = FusedWeights()
        cw.W_q = torch.cat([l.self_attn.q_proj.weight for l in layers], dim=0).to(dt)
        cw.W_k = torch.cat([l.self_attn.k_proj.weight for l in layers], dim=0).to(dt)
        cw.W_v = torch.cat([l.self_attn.v_proj.weight for l in layers], dim=0).to(dt)
        cw.W_o = torch.cat([l.self_attn.o_proj.weight for l in layers], dim=1).to(dt)

        cw.b_q = torch.cat([l.self_attn.q_proj.bias for l in layers], dim=0).to(dt)
        cw.b_k = torch.cat([l.self_attn.k_proj.bias for l in layers], dim=0).to(dt)
        cw.b_v = torch.cat([l.self_attn.v_proj.bias for l in layers], dim=0).to(dt)

        cw.W_gate = torch.cat([l.mlp.gate_proj.weight for l in layers], dim=0).to(dt)
        cw.W_up = torch.cat([l.mlp.up_proj.weight for l in layers], dim=0).to(dt)
        cw.W_down = torch.cat([l.mlp.down_proj.weight for l in layers], dim=1).to(dt)

        cw.ln1_w = torch.stack([l.input_layernorm.weight for l in layers]).mean(0).to(dt)
        cw.ln2_w = torch.stack([l.post_attention_layernorm.weight for l in layers]).mean(0).to(dt)

        cw.layers = chunk_layers
        cw.n = n
        return cw

    # -- prefix --------------------------------------------------------------
    def run_prefix(self, input_ids, seq_layers):
        model = self.model
        hidden = model.model.embed_tokens(input_ids)
        B, T, D = hidden.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        if seq_layers > 0:
            from transformers.masking_utils import create_causal_mask
            cache_position = torch.arange(T, device=input_ids.device)
            causal_mask = create_causal_mask(
                config=model.config, input_embeds=hidden,
                attention_mask=None, cache_position=cache_position,
                past_key_values=None, position_ids=position_ids,
            )
            for i in range(seq_layers):
                layer = model.model.layers[i]
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask.get(layer.attention_type, None) if isinstance(causal_mask, dict) else causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        return hidden, position_embeddings

    # -- batched forward -----------------------------------------------------
    def batched_forward(self, all_h_prev, pos_emb, pw):
        config = self.config
        n_par, B, T, D = all_h_prev.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        hd = D // n_head
        n_groups = n_head // n_kv

        # Input LayerNorm
        x_norm = _rms_norm(all_h_prev, pw.ln1_w[:, None, None, :], eps)

        # QKV
        all_q = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_q) + pw.b_q[:, None, None, :]
        all_k = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_k) + pw.b_k[:, None, None, :]
        all_v = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_v) + pw.b_v[:, None, None, :]

        all_q = all_q.view(n_par, B, T, n_head, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)

        # RoPE
        cos, sin = pos_emb
        cos_r = cos.unsqueeze(0).unsqueeze(2)  # (1, B, 1, T, hd)
        sin_r = sin.unsqueeze(0).unsqueeze(2)
        all_q = _apply_rope_batched(all_q, cos_r, sin_r)
        all_k = _apply_rope_batched(all_k, cos_r, sin_r)

        # GQA expand
        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)

        # SDPA
        q_s = all_q.reshape(n_par * B, n_head, T, hd)
        k_s = all_k.reshape(n_par * B, n_head, T, hd)
        v_s = all_v.reshape(n_par * B, n_head, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

        attn_out = y_s.reshape(n_par, B, n_head, T, hd).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, D)
        attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o)

        hidden = all_h_prev + attn_out

        # Post-attn LN + SwiGLU MLP
        h_norm = _rms_norm(hidden, pw.ln2_w[:, None, None, :], eps)
        gate = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_gate)
        up = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_up)
        mlp = torch.einsum('lbti,ldi->lbtd', F.silu(gate) * up, pw.W_down)

        return hidden + mlp

    # -- fused forward chunk -------------------------------------------------
    def fused_forward_chunk(self, h_in, pos_emb, cw, avg=False):
        config = self.config
        B, T, D = h_in.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        total_q = cw.n * n_head
        total_kv = cw.n * n_kv
        hd = D // n_head
        n_groups = n_head // n_kv

        x_norm = _rms_norm(h_in, cw.ln1_w, eps)

        q = F.linear(x_norm, cw.W_q, cw.b_q).view(B, T, total_q, hd)
        k = F.linear(x_norm, cw.W_k, cw.b_k).view(B, T, total_kv, hd)
        v = F.linear(x_norm, cw.W_v, cw.b_v).view(B, T, total_kv, hd)

        cos, sin = pos_emb
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # (B, total_q, T, hd)
        k = k.transpose(1, 2)  # (B, total_kv, T, hd)
        v = v.transpose(1, 2)

        # GQA expand
        if n_groups > 1:
            k = k[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)
            v = v[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        attn_out = F.linear(y, cw.W_o)
        if avg and cw.n > 1:
            attn_out = attn_out / cw.n
        x_mid = h_in + attn_out

        h_norm = _rms_norm(x_mid, cw.ln2_w, eps)
        gate = F.linear(h_norm, cw.W_gate)
        up = F.linear(h_norm, cw.W_up)
        mlp = F.linear(F.silu(gate) * up, cw.W_down)
        if avg and cw.n > 1:
            mlp = mlp / cw.n
        return x_mid + mlp

    # -- batched chunks forward ----------------------------------------------
    def batched_chunks_forward(self, all_h_in, pos_emb, chunk_weights_list, avg=False):
        config = self.config
        C = len(chunk_weights_list)
        chunk_size = chunk_weights_list[0].n
        B, T, D = all_h_in.shape[1:]
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        q_per_chunk = chunk_size * n_head
        kv_per_chunk = chunk_size * n_kv
        hd = D // n_head
        n_groups = n_head // n_kv

        W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
        W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
        W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
        W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
        W_gate = torch.stack([cw.W_gate for cw in chunk_weights_list])
        W_up = torch.stack([cw.W_up for cw in chunk_weights_list])
        W_down = torch.stack([cw.W_down for cw in chunk_weights_list])

        b_q = torch.stack([cw.b_q for cw in chunk_weights_list])
        b_k = torch.stack([cw.b_k for cw in chunk_weights_list])
        b_v = torch.stack([cw.b_v for cw in chunk_weights_list])

        ln1_w = torch.stack([cw.ln1_w for cw in chunk_weights_list])
        ln2_w = torch.stack([cw.ln2_w for cw in chunk_weights_list])

        # Input LN
        x_norm = _rms_norm(all_h_in, ln1_w[:, None, None, :], eps)

        # QKV
        all_q = torch.einsum('cbtd,chd->cbth', x_norm, W_q) + b_q[:, None, None, :]
        all_k = torch.einsum('cbtd,chd->cbth', x_norm, W_k) + b_k[:, None, None, :]
        all_v = torch.einsum('cbtd,chd->cbth', x_norm, W_v) + b_v[:, None, None, :]

        all_q = all_q.view(C, B, T, q_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)

        # RoPE
        cos, sin = pos_emb
        cos_r = cos.unsqueeze(0).unsqueeze(2)
        sin_r = sin.unsqueeze(0).unsqueeze(2)
        all_q = _apply_rope_batched(all_q, cos_r, sin_r)
        all_k = _apply_rope_batched(all_k, cos_r, sin_r)

        # GQA expand
        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)

        # SDPA
        q_s = all_q.reshape(C * B, q_per_chunk, T, hd)
        k_s = all_k.reshape(C * B, q_per_chunk, T, hd)
        v_s = all_v.reshape(C * B, q_per_chunk, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True)

        y = y_s.reshape(C, B, q_per_chunk, T, hd).permute(0, 1, 3, 2, 4)
        y = y.reshape(C, B, T, q_per_chunk * hd)

        attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
        if avg and chunk_size > 1:
            attn_out = attn_out / chunk_size

        hidden = all_h_in + attn_out

        # Post-attn LN + MLP
        h_norm = _rms_norm(hidden, ln2_w[:, None, None, :], eps)
        gate = torch.einsum('cbtd,cid->cbti', h_norm, W_gate)
        up = torch.einsum('cbtd,cid->cbti', h_norm, W_up)
        mlp = torch.einsum('cbti,cdi->cbtd', F.silu(gate) * up, W_down)
        if avg and chunk_size > 1:
            mlp = mlp / chunk_size

        return hidden + mlp


# ===========================================================================
# GemmaBackend  (Gemma 3 1B)
# ===========================================================================

class GemmaBackend(ModelBackend):

    # -- weights -------------------------------------------------------------
    def precompute_weights(self, par):
        model = self.model
        layers = [model.model.layers[i] for i in par]
        dt = next(model.parameters()).dtype

        pw = StackedWeights()
        pw.W_q = torch.stack([l.self_attn.q_proj.weight for l in layers]).to(dt)
        pw.W_k = torch.stack([l.self_attn.k_proj.weight for l in layers]).to(dt)
        pw.W_v = torch.stack([l.self_attn.v_proj.weight for l in layers]).to(dt)
        pw.W_o = torch.stack([l.self_attn.o_proj.weight for l in layers]).to(dt)

        pw.W_gate = torch.stack([l.mlp.gate_proj.weight for l in layers]).to(dt)
        pw.W_up = torch.stack([l.mlp.up_proj.weight for l in layers]).to(dt)
        pw.W_down = torch.stack([l.mlp.down_proj.weight for l in layers]).to(dt)

        pw.ln_input = torch.stack([l.input_layernorm.weight for l in layers]).to(dt)
        pw.ln_post_attn = torch.stack([l.post_attention_layernorm.weight for l in layers]).to(dt)
        pw.ln_pre_ffn = torch.stack([l.pre_feedforward_layernorm.weight for l in layers]).to(dt)
        pw.ln_post_ffn = torch.stack([l.post_feedforward_layernorm.weight for l in layers]).to(dt)

        pw.q_norm = torch.stack([l.self_attn.q_norm.weight for l in layers]).to(dt)
        pw.k_norm = torch.stack([l.self_attn.k_norm.weight for l in layers]).to(dt)

        pw.is_sliding = [l.self_attn.is_sliding for l in layers]

        pw.par = par
        pw.n_par = len(par)
        return pw

    def chunk_weights(self, chunk_layers):
        model = self.model
        layers = [model.model.layers[i] for i in chunk_layers]
        dt = next(model.parameters()).dtype
        n = len(chunk_layers)

        cw = FusedWeights()
        cw.W_q = torch.cat([l.self_attn.q_proj.weight for l in layers], dim=0).to(dt)
        cw.W_k = torch.cat([l.self_attn.k_proj.weight for l in layers], dim=0).to(dt)
        cw.W_v = torch.cat([l.self_attn.v_proj.weight for l in layers], dim=0).to(dt)
        cw.W_o = torch.cat([l.self_attn.o_proj.weight for l in layers], dim=1).to(dt)

        cw.W_gate = torch.cat([l.mlp.gate_proj.weight for l in layers], dim=0).to(dt)
        cw.W_up = torch.cat([l.mlp.up_proj.weight for l in layers], dim=0).to(dt)
        cw.W_down = torch.cat([l.mlp.down_proj.weight for l in layers], dim=1).to(dt)

        cw.ln_input = torch.stack([l.input_layernorm.weight for l in layers]).mean(0).to(dt)
        cw.ln_post_attn = torch.stack([l.post_attention_layernorm.weight for l in layers]).mean(0).to(dt)
        cw.ln_pre_ffn = torch.stack([l.pre_feedforward_layernorm.weight for l in layers]).mean(0).to(dt)
        cw.ln_post_ffn = torch.stack([l.post_feedforward_layernorm.weight for l in layers]).mean(0).to(dt)

        cw.q_norm = torch.stack([l.self_attn.q_norm.weight for l in layers]).mean(0).to(dt)
        cw.k_norm = torch.stack([l.self_attn.k_norm.weight for l in layers]).mean(0).to(dt)

        cw.layers = chunk_layers
        cw.n = n
        return cw

    # -- prefix --------------------------------------------------------------
    def run_prefix(self, input_ids, seq_layers):
        model = self.model
        hidden = model.model.embed_tokens(input_ids)

        B, T, D = hidden.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb_global = model.model.rotary_emb(hidden, position_ids)
        pos_emb_local = model.model.rotary_emb_local(hidden, position_ids)

        if seq_layers > 0:
            from transformers.masking_utils import create_causal_mask
            cache_position = torch.arange(T, device=input_ids.device)
            causal_mask = create_causal_mask(
                config=model.config, input_embeds=hidden,
                attention_mask=None, cache_position=cache_position,
                past_key_values=None, position_ids=position_ids,
            )
            for i in range(seq_layers):
                layer = model.model.layers[i]
                mask = causal_mask
                if isinstance(causal_mask, dict):
                    mask = causal_mask.get(layer.attention_type, None)
                out = layer(
                    hidden,
                    position_embeddings_global=pos_emb_global,
                    position_embeddings_local=pos_emb_local,
                    attention_mask=mask,
                    position_ids=position_ids,
                )
                hidden = out[0] if isinstance(out, tuple) else out

        return hidden, pos_emb_global, pos_emb_local

    # -- batched forward -----------------------------------------------------
    def batched_forward(self, all_h_prev, pos_emb, pw):
        config = self.config
        pos_emb_global, pos_emb_local = pos_emb
        n_par, B, T, D = all_h_prev.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        hd = config.head_dim
        n_groups = n_head // n_kv
        attn_dim = n_head * hd

        # Input LayerNorm
        x_norm = _rms_norm_gemma(all_h_prev, pw.ln_input[:, None, None, :], eps)

        # QKV
        all_q = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_q)
        all_k = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_k)
        all_v = torch.einsum('lbtd,lhd->lbth', x_norm, pw.W_v)

        all_q = all_q.view(n_par, B, T, n_head, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(n_par, B, T, n_kv, hd).permute(0, 1, 3, 2, 4)

        # QK RMSNorm (per-layer, per-head)
        all_q = _rms_norm_headwise(all_q, pw.q_norm[:, None, None, None, :], eps)
        all_k = _rms_norm_headwise(all_k, pw.k_norm[:, None, None, None, :], eps)

        # RoPE: select local vs global per layer, apply per-layer
        cos_g, sin_g = pos_emb_global
        cos_l, sin_l = pos_emb_local
        cos_g_r = cos_g.unsqueeze(0).unsqueeze(2)
        sin_g_r = sin_g.unsqueeze(0).unsqueeze(2)
        cos_l_r = cos_l.unsqueeze(0).unsqueeze(2)
        sin_l_r = sin_l.unsqueeze(0).unsqueeze(2)

        # Build per-layer cos/sin: (n_par, B, 1, T, hd)
        cos_stack = torch.where(
            torch.tensor(pw.is_sliding, device=all_q.device).view(n_par, 1, 1, 1, 1),
            cos_l_r.expand(n_par, -1, -1, -1, -1),
            cos_g_r.expand(n_par, -1, -1, -1, -1),
        )
        sin_stack = torch.where(
            torch.tensor(pw.is_sliding, device=all_q.device).view(n_par, 1, 1, 1, 1),
            sin_l_r.expand(n_par, -1, -1, -1, -1),
            sin_g_r.expand(n_par, -1, -1, -1, -1),
        )

        all_q = _apply_rope_batched(all_q, cos_stack, sin_stack)
        all_k = _apply_rope_batched(all_k, cos_stack, sin_stack)

        # GQA expand
        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                n_par, B, n_kv, n_groups, T, hd).reshape(n_par, B, n_head, T, hd)

        # SDPA
        q_s = all_q.reshape(n_par * B, n_head, T, hd)
        k_s = all_k.reshape(n_par * B, n_head, T, hd)
        v_s = all_v.reshape(n_par * B, n_head, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True,
                                              scale=config.head_dim ** -0.5)

        attn_out = y_s.reshape(n_par, B, n_head, T, hd).permute(0, 1, 3, 2, 4).reshape(n_par, B, T, attn_dim)
        attn_out = torch.einsum('lbtd,lod->lbto', attn_out, pw.W_o)

        # Post-attention LN (applied to attn output, before residual)
        attn_out = _rms_norm_gemma(attn_out, pw.ln_post_attn[:, None, None, :], eps)
        hidden = all_h_prev + attn_out

        # Pre-FFN LN + GELU MLP + Post-FFN LN
        h_norm = _rms_norm_gemma(hidden, pw.ln_pre_ffn[:, None, None, :], eps)
        gate = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_gate)
        up = torch.einsum('lbtd,lid->lbti', h_norm, pw.W_up)
        mlp = torch.einsum('lbti,ldi->lbtd', F.gelu(gate, approximate='tanh') * up, pw.W_down)
        mlp = _rms_norm_gemma(mlp, pw.ln_post_ffn[:, None, None, :], eps)

        return hidden + mlp

    # -- fused forward chunk -------------------------------------------------
    def fused_forward_chunk(self, h_in, pos_emb, cw, avg=False,
                            is_sliding_majority=True):
        config = self.config
        pos_emb_global, pos_emb_local = pos_emb
        B, T, D = h_in.shape
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        hd = config.head_dim
        total_q = cw.n * n_head
        total_kv = cw.n * n_kv
        n_groups = n_head // n_kv
        attn_dim = total_q * hd

        x_norm = _rms_norm_gemma(h_in, cw.ln_input, eps)

        q = F.linear(x_norm, cw.W_q).view(B, T, total_q, hd)
        k = F.linear(x_norm, cw.W_k).view(B, T, total_kv, hd)
        v = F.linear(x_norm, cw.W_v).view(B, T, total_kv, hd)

        # QK norm with averaged weights
        q = _rms_norm_headwise(q, cw.q_norm, eps)
        k = _rms_norm_headwise(k, cw.k_norm, eps)

        # Use local RoPE for sliding-majority chunks, global for full-attention
        rope_emb = pos_emb_local if is_sliding_majority else pos_emb_global
        cos, sin = rope_emb
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if n_groups > 1:
            k = k[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)
            v = v[:, :, None, :, :].expand(B, total_kv, n_groups, T, hd).reshape(B, total_q, T, hd)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=hd ** -0.5)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        attn_out = F.linear(y, cw.W_o)
        if avg and cw.n > 1:
            attn_out = attn_out / cw.n
        attn_out = _rms_norm_gemma(attn_out, cw.ln_post_attn, eps)
        x_mid = h_in + attn_out

        h_norm = _rms_norm_gemma(x_mid, cw.ln_pre_ffn, eps)
        gate = F.linear(h_norm, cw.W_gate)
        up = F.linear(h_norm, cw.W_up)
        mlp = F.linear(F.gelu(gate, approximate='tanh') * up, cw.W_down)
        if avg and cw.n > 1:
            mlp = mlp / cw.n
        mlp = _rms_norm_gemma(mlp, cw.ln_post_ffn, eps)
        return x_mid + mlp

    # -- batched chunks forward ----------------------------------------------
    def batched_chunks_forward(self, all_h_in, pos_emb, chunk_weights_list,
                               avg=False, chunk_sliding_flags=None):
        config = self.config
        pos_emb_global, pos_emb_local = pos_emb
        C = len(chunk_weights_list)
        chunk_size = chunk_weights_list[0].n
        B, T, D = all_h_in.shape[1:]
        eps = config.rms_norm_eps
        n_head = config.num_attention_heads
        n_kv = config.num_key_value_heads
        hd = config.head_dim
        q_per_chunk = chunk_size * n_head
        kv_per_chunk = chunk_size * n_kv
        n_groups = n_head // n_kv

        W_q = torch.stack([cw.W_q for cw in chunk_weights_list])
        W_k = torch.stack([cw.W_k for cw in chunk_weights_list])
        W_v = torch.stack([cw.W_v for cw in chunk_weights_list])
        W_o = torch.stack([cw.W_o for cw in chunk_weights_list])
        W_gate = torch.stack([cw.W_gate for cw in chunk_weights_list])
        W_up = torch.stack([cw.W_up for cw in chunk_weights_list])
        W_down = torch.stack([cw.W_down for cw in chunk_weights_list])

        ln_input = torch.stack([cw.ln_input for cw in chunk_weights_list])
        ln_post_attn = torch.stack([cw.ln_post_attn for cw in chunk_weights_list])
        ln_pre_ffn = torch.stack([cw.ln_pre_ffn for cw in chunk_weights_list])
        ln_post_ffn = torch.stack([cw.ln_post_ffn for cw in chunk_weights_list])
        q_norm_w = torch.stack([cw.q_norm for cw in chunk_weights_list])
        k_norm_w = torch.stack([cw.k_norm for cw in chunk_weights_list])

        x_norm = _rms_norm_gemma(all_h_in, ln_input[:, None, None, :], eps)

        all_q = torch.einsum('cbtd,chd->cbth', x_norm, W_q)
        all_k = torch.einsum('cbtd,chd->cbth', x_norm, W_k)
        all_v = torch.einsum('cbtd,chd->cbth', x_norm, W_v)

        all_q = all_q.view(C, B, T, q_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_k = all_k.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)
        all_v = all_v.view(C, B, T, kv_per_chunk, hd).permute(0, 1, 3, 2, 4)

        # QK norm
        all_q = _rms_norm_headwise(all_q, q_norm_w[:, None, None, None, :], eps)
        all_k = _rms_norm_headwise(all_k, k_norm_w[:, None, None, None, :], eps)

        # Per-chunk RoPE selection
        cos_g, sin_g = pos_emb_global
        cos_l, sin_l = pos_emb_local
        cos_g_r = cos_g.unsqueeze(0).unsqueeze(2)
        sin_g_r = sin_g.unsqueeze(0).unsqueeze(2)
        cos_l_r = cos_l.unsqueeze(0).unsqueeze(2)
        sin_l_r = sin_l.unsqueeze(0).unsqueeze(2)

        sliding_t = torch.tensor(chunk_sliding_flags, device=all_q.device).view(C, 1, 1, 1, 1)
        cos_stack = torch.where(sliding_t, cos_l_r.expand(C, -1, -1, -1, -1), cos_g_r.expand(C, -1, -1, -1, -1))
        sin_stack = torch.where(sliding_t, sin_l_r.expand(C, -1, -1, -1, -1), sin_g_r.expand(C, -1, -1, -1, -1))

        all_q = _apply_rope_batched(all_q, cos_stack, sin_stack)
        all_k = _apply_rope_batched(all_k, cos_stack, sin_stack)

        if n_groups > 1:
            all_k = all_k[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)
            all_v = all_v[:, :, :, None, :, :].expand(
                C, B, kv_per_chunk, n_groups, T, hd).reshape(C, B, q_per_chunk, T, hd)

        q_s = all_q.reshape(C * B, q_per_chunk, T, hd)
        k_s = all_k.reshape(C * B, q_per_chunk, T, hd)
        v_s = all_v.reshape(C * B, q_per_chunk, T, hd)
        y_s = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, scale=hd ** -0.5)

        y = y_s.reshape(C, B, q_per_chunk, T, hd).permute(0, 1, 3, 2, 4)
        y = y.reshape(C, B, T, q_per_chunk * hd)

        attn_out = torch.einsum('cbth,cdh->cbtd', y, W_o)
        if avg and chunk_size > 1:
            attn_out = attn_out / chunk_size
        attn_out = _rms_norm_gemma(attn_out, ln_post_attn[:, None, None, :], eps)
        hidden = all_h_in + attn_out

        h_norm = _rms_norm_gemma(hidden, ln_pre_ffn[:, None, None, :], eps)
        gate = torch.einsum('cbtd,cid->cbti', h_norm, W_gate)
        up = torch.einsum('cbtd,cid->cbti', h_norm, W_up)
        mlp = torch.einsum('cbti,cdi->cbtd', F.gelu(gate, approximate='tanh') * up, W_down)
        if avg and chunk_size > 1:
            mlp = mlp / chunk_size
        mlp = _rms_norm_gemma(mlp, ln_post_ffn[:, None, None, :], eps)

        return hidden + mlp


# ===========================================================================
# Public forward functions
# ===========================================================================

@torch.inference_mode()
def forward_sequential(backend, input_ids):
    """Standard sequential forward through the full model."""
    return backend.forward_sequential(input_ids)


@torch.inference_mode()
def forward_idn_batched(backend, input_ids, seq_layers, pw, K=1, elk_k=0.0,
                         init='h0'):
    """IDN batched: sequential prefix + batched parallel layers + IDN correction."""
    prefix_result = backend.run_prefix(input_ids, seq_layers)
    h_init = prefix_result[0]
    pos_emb = prefix_result[1:]  # 1 item for Llama/Qwen, 2 items for Gemma
    n_par = pw.n_par

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_par, -1, -1, -1)
        all_out = backend.batched_forward(all_h0, pos_emb[0] if len(pos_emb) == 1 else pos_emb, pw)
        hs = [h_init] + [all_out[j] for j in range(n_par)]
    else:
        hs = [h_init] + [h_init.clone() for _ in range(n_par)]

    for _ in range(K):
        all_hp = torch.stack([hs[j] for j in range(n_par)])
        all_out = backend.batched_forward(all_hp, pos_emb[0] if len(pos_emb) == 1 else pos_emb, pw)
        _idn_correction(all_out, hs, h_init, n_par, elk_k)

    return backend.final_proj(hs[n_par])


@torch.inference_mode()
def forward_fused(backend, input_ids, seq_layers, chunk_weights, K=1,
                   elk_k=0.0, avg=False, **kwargs):
    """Fully fused: sequential prefix + single mega-block (no per-layer correction).

    Extra kwargs (e.g. is_sliding_majority) are forwarded to the backend.
    """
    prefix_result = backend.run_prefix(input_ids, seq_layers)
    h_init = prefix_result[0]
    pos_emb = prefix_result[1:]

    pe = pos_emb[0] if len(pos_emb) == 1 else pos_emb
    out = backend.fused_forward_chunk(h_init, pe, chunk_weights[0], avg=avg, **kwargs)
    for _ in range(1, K):
        out = backend.fused_forward_chunk(out, pe, chunk_weights[0], avg=avg, **kwargs)

    return backend.final_proj(out)


@torch.inference_mode()
def forward_chunkwise_batched(backend, input_ids, seq_layers,
                                chunk_weights_list, K=1, elk_k=0.0,
                                avg=False, init='h0', **kwargs):
    """Chunkwise batched: fuse within chunks, IDN correct between chunks.

    Extra kwargs (e.g. chunk_sliding_flags for Gemma) are forwarded.
    """
    prefix_result = backend.run_prefix(input_ids, seq_layers)
    h_init = prefix_result[0]
    pos_emb = prefix_result[1:]
    pe = pos_emb[0] if len(pos_emb) == 1 else pos_emb

    n_chunks = len(chunk_weights_list)
    chunk_size = chunk_weights_list[0].n

    if init == 'batch_fwd':
        all_h0 = h_init.unsqueeze(0).expand(n_chunks, -1, -1, -1)
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_init = backend.batched_chunks_forward(
                all_h0, pe, chunk_weights_list, avg=avg, **kwargs)
            chunk_estimates = [all_init[c] for c in range(n_chunks)]
        else:
            chunk_estimates = [
                backend.fused_forward_chunk(h_init, pe, cw, avg=avg, **kwargs)
                for cw in chunk_weights_list
            ]
    else:
        chunk_estimates = [h_init.clone() for _ in range(n_chunks)]

    hs = [h_init] + list(chunk_estimates)

    for _ in range(K):
        if all(cw.n == chunk_size for cw in chunk_weights_list) and n_chunks > 1:
            all_h_in = torch.stack([hs[c] for c in range(n_chunks)])
            all_out = backend.batched_chunks_forward(
                all_h_in, pe, chunk_weights_list, avg=avg, **kwargs)
            chunk_outs = [all_out[c] for c in range(n_chunks)]
        else:
            chunk_outs = [
                backend.fused_forward_chunk(hs[c], pe, cw, avg=avg, **kwargs)
                for c, cw in enumerate(chunk_weights_list)
            ]
        _idn_correction(chunk_outs, hs, h_init, n_chunks, elk_k)

    return backend.final_proj(hs[n_chunks])


# ===========================================================================
# Factory
# ===========================================================================

def get_backend(model):
    """Return the appropriate ModelBackend for a HuggingFace model."""
    name = model.config.model_type.lower()
    if 'qwen' in name:
        return QwenBackend(model)
    elif 'llama' in name:
        return LlamaBackend(model)
    elif 'gemma' in name:
        return GemmaBackend(model)
    raise ValueError(f"Unknown model type: {name}")
