"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    no_x0_resid: bool = False  # disable x0_lambdas (no embedding blending) and resid scaling
    no_ve: bool = False        # disable value embeddings
    # mHC (manifold-constrained HyperConnections)
    use_mhc: bool = False           # enable mHC residual connections
    mhc_num_streams: int = 4        # number of residual streams
    mhc_sinkhorn_iters: int = 10    # Sinkhorn iterations for doubly stochastic projection
    mhc_sinkhorn_tau: float = 0.05  # Sinkhorn temperature


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

def sinkhorn_log(logits, num_iters=10, tau=0.05):
    """Project logits to doubly stochastic matrix via Sinkhorn-Knopp in log-space."""
    Z = logits / tau
    n = logits.shape[-1]
    log_marginal = torch.zeros(n, device=logits.device, dtype=logits.dtype)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if (has_ve(layer_idx, config.n_layer) and not config.no_ve) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class HyperConnectionMHC(nn.Module):
    """mHC wrapper for a single sublayer (attn or MLP).
    Implements width connection (stream mixing + branch input pooling)
    and depth connection (branch output distribution + residual add)."""
    def __init__(self, num_streams, layer_index):
        super().__init__()
        self.num_streams = num_streams
        init_stream = layer_index % num_streams
        # H_res: (S, S) logits -> Sinkhorn -> doubly stochastic residual mixing
        init_h_res = torch.full((num_streams, num_streams), -0.5)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)
        # H_pre: (S,) logits -> softmax -> pools streams to branch input
        init_h_pre = torch.full((num_streams,), -1.0)
        init_h_pre[init_stream] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)
        # H_post: (S,) logits -> softmax -> distributes branch output to streams
        self.H_post_logits = nn.Parameter(torch.zeros(num_streams))

    def cache_hc_matrices(self, sinkhorn_iters, sinkhorn_tau):
        """Precompute and cache HC matrices for inference. Call once after model.eval()."""
        with torch.no_grad():
            self._cached_H_res = sinkhorn_log(self.H_res_logits, num_iters=sinkhorn_iters, tau=sinkhorn_tau).detach()
            self._cached_H_pre = F.softmax(self.H_pre_logits, dim=-1).detach()
            self._cached_H_post = F.softmax(self.H_post_logits, dim=-1).detach()

    def width_connection(self, x, sinkhorn_iters, sinkhorn_tau):
        S = self.num_streams
        BS, T, D = x.shape
        B = BS // S
        residuals = x.view(B, S, T, D)
        if hasattr(self, '_cached_H_res'):
            H_res = self._cached_H_res.to(x.dtype)
            H_pre = self._cached_H_pre.to(x.dtype)
            H_post = self._cached_H_post.to(x.dtype)
        else:
            H_res = sinkhorn_log(self.H_res_logits, num_iters=sinkhorn_iters, tau=sinkhorn_tau).to(x.dtype)
            H_pre = F.softmax(self.H_pre_logits, dim=-1).to(x.dtype)
            H_post = F.softmax(self.H_post_logits, dim=-1).to(x.dtype)
        residuals_out = torch.einsum('ij,bjkl->bikl', H_res, residuals)
        branch_input = torch.einsum('j,bjkl->bkl', H_pre, residuals)
        return branch_input, residuals_out, H_post

    def depth_connection(self, branch_output, residuals_out, H_post):
        output = branch_output.unsqueeze(1) * H_post[None, :, None, None]
        result = output + residuals_out
        B, S, T, D = result.shape
        return result.reshape(B * S, T, D)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.use_mhc = config.use_mhc
        if self.use_mhc:
            self.mhc_sinkhorn_iters = config.mhc_sinkhorn_iters
            self.mhc_sinkhorn_tau = config.mhc_sinkhorn_tau
            self.hc_attn = HyperConnectionMHC(config.mhc_num_streams, layer_idx * 2)
            self.hc_mlp = HyperConnectionMHC(config.mhc_num_streams, layer_idx * 2 + 1)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if self.use_mhc:
            branch_in, res_out, h_post = self.hc_attn.width_connection(x, self.mhc_sinkhorn_iters, self.mhc_sinkhorn_tau)
            attn_out = self.attn(norm(branch_in), ve, cos_sin, window_size, kv_cache)
            x = self.hc_attn.depth_connection(attn_out, res_out, h_post)
            branch_in, res_out, h_post = self.hc_mlp.width_connection(x, self.mhc_sinkhorn_iters, self.mhc_sinkhorn_tau)
            mlp_out = self.mlp(norm(branch_in))
            x = self.hc_mlp.depth_connection(mlp_out, res_out, h_post)
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        # Reg warmup factor as a buffer (0-d tensor) so torch.compile doesn't
        # specialize on it.  Updated externally by the training loop each step.
        self.register_buffer("reg_warmup_factor", torch.ones(1), persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # mHC HyperConnection parameters
        # Init values chosen so H_res ≈ identity and H_pre ≈ one-hot after Sinkhorn/softmax
        # at tau=0.05, but with non-vanishing gradients (unlike the reference's -8 which
        # gives exp(-160)=0 and kills gradient flow in float32).
        if self.config.use_mhc:
            for i, block in enumerate(self.transformer.h):
                for hc_idx, hc in enumerate([block.hc_attn, block.hc_mlp]):
                    hc.H_res_logits.data.fill_(-0.5)
                    hc.H_res_logits.data.fill_diagonal_(0.0)
                    init_stream = (i * 2 + hc_idx) % self.config.mhc_num_streams
                    hc.H_pre_logits.data.fill_(-1.0)
                    hc.H_pre_logits.data[init_stream] = 0.0
                    hc.H_post_logits.data.fill_(0.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, per-layer scalars, and mHC routing params
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        mhc_numel = sum(p.numel() for block in self.transformer.h
                        for hc in [getattr(block, 'hc_attn', None), getattr(block, 'hc_mlp', None)]
                        if hc is not None for p in hc.parameters()) if self.config.use_mhc else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel + mhc_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # With mHC, each layer processes S× more samples (S streams per batch element)
        stream_multiplier = self.config.mhc_num_streams if self.config.use_mhc else 1
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = (6 * (nparams - nparams_exclude) + attn_flops) * stream_multiplier
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # When mHC is enabled, separate HC params from matrix params (HC uses AdamW, not Muon)
        mhc_params = []
        if self.config.use_mhc:
            mhc_param_ids = set()
            for block in self.transformer.h:
                for p in block.hc_attn.parameters():
                    mhc_params.append(p); mhc_param_ids.add(id(p))
                for p in block.hc_mlp.parameters():
                    mhc_params.append(p); mhc_param_ids.add(id(p))
            matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in mhc_param_ids]
        else:
            matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(mhc_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # When no_ve/no_x0_resid: these params exist but aren't used in forward,
        # so they get no gradient. Exclude from optimizer to avoid None grad errors.
        if self.config.no_ve:
            value_embeds_params = []
        if self.config.no_x0_resid:
            resid_params = []
            x0_params = []

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mhc_params, lr=0.01, betas=(0.9, 0.999), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', jacobi_reg=0.0, identity_newton_reg=0.0, diag_newton_reg=0.0, mhc_newton_reg=0.0):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # mHC stream expansion: (B, T, D) -> (B*S, T, D)
        idx_ve = idx  # keep original idx for VE lookups (all streams share same tokens)
        if self.config.use_mhc:
            S = self.config.mhc_num_streams
            D = x.shape[-1]
            x = x.unsqueeze(1).expand(B, S, T, D).reshape(B * S, T, D)

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        jacobi_reg_loss = 0.0  # accumulate contractive regularization
        all_h = []  # save per-layer outputs for identity Newton reg
        for i, block in enumerate(self.transformer.h):
            if self.config.no_x0_resid:
                pass  # pure residual: x unchanged (standard transformer)
            else:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None if self.config.no_ve else (self.value_embeds[str(i)](idx_ve).to(x.dtype) if str(i) in self.value_embeds else None)
            x_pre = x  # save pre-block input for regularization
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if jacobi_reg > 0:
                # Contractive regularization: penalize the Jacobian spectral radius.
                # Estimate σ_max(∂block/∂input) via finite-difference power iteration:
                #   σ_max ≈ ||block(x + εv) - block(x)|| / (ε||v||)
                # Then penalize ||block(x) - x||² scaled by max(0, σ_max - target)
                # so that the gradient flows through the MAIN forward path.
                eps_fd = 1e-2
                v = torch.randn_like(x_pre)
                v = v / (v.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                with torch.no_grad():
                    x_pert = block(x_pre + eps_fd * v, ve, cos_sin, self.window_sizes[i], kv_cache)
                    x_unpert = block(x_pre, ve, cos_sin, self.window_sizes[i], kv_cache)
                    Jv_approx = (x_pert - x_unpert) / eps_fd
                    sigma_est = Jv_approx.float().norm(dim=-1).mean().item()
                # The FD estimate is no-grad (just for measuring), but we penalize
                # the actual residual norm of the block in the forward path, weighted
                # by how much the spectral radius exceeds 1.
                if sigma_est > 0.9:  # only penalize if spectral radius is near/above 1
                    residual = x - x_pre
                    reg_i = (residual.float().norm(dim=-1) / (x_pre.float().norm(dim=-1) + 1e-6)).mean()
                    weight = min(sigma_est, 3.0)  # stronger penalty for larger spectral radius
                    jacobi_reg_loss = jacobi_reg_loss + weight * reg_i
            all_h.append(x)
            if i == backout_layer:
                x_backout = x
        # Identity Newton regularization: penalize the gap between sequential output
        # and what identity Newton K=1 would produce, for the last few layers.
        # This trains the model so that running the last N layers in parallel (with
        # one Newton correction using J≈I) gives output close to sequential.
        # n_par configs for identity/diagonal Newton regularization.
        # Set self.n_par_configs before torch.compile (fixed for the entire run).
        # Default: [3, 5, 7, 10, 12, 16] clamped to valid range.
        n_par_configs = getattr(self, 'n_par_configs', None)
        if n_par_configs is None:
            n_par_configs = [3, 5, min(7, n_layer-1), min(10, n_layer-1), min(12, n_layer-1), min(16, n_layer-2)]

        # Identity Newton regularization: penalize the gap between sequential output
        # and what identity Newton K=1 would produce for the last N layers.
        # Trains layers to be input-invariant (J ≈ I).
        #
        # idn_stride controls where the loss is applied:
        #   stride=0 (default): loss only on the last layer (original behavior)
        #   stride=S (S>0): run full IDN forward substitution across all n_par layers,
        #     add loss at every S-th layer comparing h_corr to h_seq at that position.
        identity_newton_loss = 0.0
        idn_stride = getattr(self, 'idn_stride', 0)
        idn_detach = getattr(self, 'idn_detach_target', False)
        if identity_newton_reg > 0 and targets is not None:
            for n_par in n_par_configs:
                seq_layers = n_layer - n_par
                h_init = all_h[seq_layers - 1] if seq_layers > 0 else x0
                if idn_stride <= 0:
                    # Original: loss only on the last layer
                    li = n_layer - 1
                    if self.config.no_x0_resid:
                        x_in_last = h_init
                    else:
                        x_in_last = self.resid_lambdas[li] * h_init + self.x0_lambdas[li] * x0
                    ve_last = None if self.config.no_ve else (self.value_embeds[str(li)](idx_ve).to(x_in_last.dtype) if str(li) in self.value_embeds else None)
                    h_newton_final = self.transformer.h[li](x_in_last, ve_last, cos_sin, self.window_sizes[li], kv_cache)
                    h_seq_final = all_h[n_layer - 1].detach() if idn_detach else all_h[n_layer - 1]
                    diff = (h_newton_final.float() - h_seq_final.float()).norm(dim=-1)
                    ref_norm = h_seq_final.float().norm(dim=-1).clamp(min=1e-6)
                    identity_newton_loss = identity_newton_loss + (diff / ref_norm).mean()
                else:
                    # Stride: run IDN forward substitution, add loss every stride layers
                    h_newton = []
                    for j in range(n_par):
                        li = seq_layers + j
                        if self.config.no_x0_resid:
                            x_in = h_init
                        else:
                            x_in = self.resid_lambdas[li] * h_init + self.x0_lambdas[li] * x0
                        ve_j = None if self.config.no_ve else (self.value_embeds[str(li)](idx_ve).to(x_in.dtype) if str(li) in self.value_embeds else None)
                        h_j = self.transformer.h[li](x_in, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                        h_newton.append(h_j)
                    # IDN K=1 forward substitution (J=I)
                    h_corr = h_newton[0]
                    for j in range(1, n_par):
                        h_corr = h_newton[j] + (h_corr - h_init)
                        # Add loss at stride positions and at the last layer
                        if (j + 1) % idn_stride == 0 or j == n_par - 1:
                            li = seq_layers + j
                            h_seq_j = all_h[li].detach() if idn_detach else all_h[li]
                            diff = (h_corr.float() - h_seq_j.float()).norm(dim=-1)
                            ref_norm = h_seq_j.float().norm(dim=-1).clamp(min=1e-6)
                            identity_newton_loss = identity_newton_loss + (diff / ref_norm).mean()

        # Diagonal Newton regularization: a weaker alternative to identity Newton.
        # Uses the actual diagonal of the Jacobian in a forward-substitution
        # correction.  Weaker constraint → less base PPL degradation.
        #
        # Config attributes (set on model before torch.compile):
        #   self.diag_newton_jvp: 'fd' | 'vjp'  — how to estimate diag(J)
        #     fd:  finite-difference Hutchinson (1 extra no-grad forward / layer)
        #     vjp: backward-mode AD Hutchinson (1 forward + 1 backward / layer)
        #     NOTE: forward-mode AD (jvp) unavailable — FA3/SDPA lack JVP rules.
        #   self.diag_stride: int — estimate diag(J) every N layers, use J≈I for
        #     the rest. Reduces FD/VJP cost from n_par to n_par/stride evaluations.
        #     Static condition (j % stride == 0) → torch.compile friendly.
        diag_newton_loss = 0.0
        if diag_newton_reg > 0 and targets is not None:
            jvp_method = getattr(self, 'diag_newton_jvp', 'fd')
            diag_stride = getattr(self, 'diag_stride', 1)
            fd_eps = 1e-2
            for n_par in n_par_configs:
                seq_layers = n_layer - n_par
                h_init = all_h[seq_layers - 1] if seq_layers > 0 else x0
                # Rademacher vector for Hutchinson diagonal estimation (shared across layers)
                z = torch.randint(0, 2, h_init.shape, device=h_init.device, dtype=h_init.dtype) * 2 - 1
                h_newton = []
                diags = []
                for j in range(n_par):
                    li = seq_layers + j
                    x_in = h_init if self.config.no_x0_resid else (self.resid_lambdas[li] * h_init + self.x0_lambdas[li] * x0)
                    ve_j = None if self.config.no_ve else (self.value_embeds[str(li)](idx_ve).to(x_in.dtype) if str(li) in self.value_embeds else None)
                    # Estimate diag(J) only every diag_stride layers (static condition
                    # for torch.compile). Skipped layers use J≈I (identity Newton).
                    estimate_diag = (diag_stride <= 1 or j % diag_stride == 0)
                    if estimate_diag and jvp_method == 'vjp':
                        with torch.enable_grad():
                            h_init_g = h_init.detach().requires_grad_(True)
                            x_in_g = h_init_g if self.config.no_x0_resid else (self.resid_lambdas[li] * h_init_g + self.x0_lambdas[li] * x0.detach())
                            h_j_g = self.transformer.h[li](x_in_g, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                            vjp_z = torch.autograd.grad(h_j_g, h_init_g, grad_outputs=z, retain_graph=False)[0]
                        diags.append((z * vjp_z).detach())
                        h_j = self.transformer.h[li](x_in, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                    elif estimate_diag:  # 'fd'
                        h_j = self.transformer.h[li](x_in, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                        x_in_pert = (h_init + fd_eps * z) if self.config.no_x0_resid else (self.resid_lambdas[li] * (h_init + fd_eps * z) + self.x0_lambdas[li] * x0)
                        with torch.no_grad():
                            h_j_pert = self.transformer.h[li](x_in_pert, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                            diag_j = z * (h_j_pert.float() - h_j.float().detach()) / fd_eps
                            diag_j = diag_j.clamp(-2, 2)
                        diags.append(diag_j.to(h_j.dtype).detach())
                    else:
                        # Skip diag estimation: use identity (J≈I)
                        h_j = self.transformer.h[li](x_in, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                        diags.append(None)  # sentinel for identity
                    h_newton.append(h_j)
                # Diagonal Newton K=1 correction via forward substitution:
                #   h_corr[0] = f_0(h_init)  (first layer input is correct)
                #   h_corr[j] = f_j(h_init) + diag_j * (h_corr[j-1] - h_init)
                # For layers where diag was skipped (None), use identity: diag_j=1
                h_corr = h_newton[0]
                for j in range(1, n_par):
                    if diags[j] is not None:
                        h_corr = h_newton[j] + diags[j] * (h_corr - h_init)
                    else:
                        h_corr = h_newton[j] + (h_corr - h_init)  # identity Newton
                h_seq_final = all_h[n_layer - 1].detach()
                diff = (h_corr.float() - h_seq_final.float()).norm(dim=-1)
                ref_norm = h_seq_final.float().norm(dim=-1).clamp(min=1e-6)
                diag_newton_loss = diag_newton_loss + (diff / ref_norm).mean()

        # mHC-Newton regularization: uses H^res as the known Jacobian for Newton K=1
        # correction. Trains the non-residual branch to be small so J_block ≈ H^res_mlp @ H^res_attn.
        # idn_stride support: if stride>0, add loss at every stride-th layer (same as IDN stride).
        mhc_newton_loss = 0.0
        if mhc_newton_reg > 0 and targets is not None and self.config.use_mhc:
            S = self.config.mhc_num_streams
            mhc_stride = getattr(self, 'idn_stride', 0)
            for n_par in n_par_configs:
                seq_layers = n_layer - n_par
                h_init = all_h[seq_layers - 1] if seq_layers > 0 else x0
                # Run all N parallel layers with h_init as input
                h_par = []
                for j in range(n_par):
                    li = seq_layers + j
                    ve_j = None if self.config.no_ve else (self.value_embeds[str(li)](idx_ve).to(h_init.dtype) if str(li) in self.value_embeds else None)
                    h_j = self.transformer.h[li](h_init, ve_j, cos_sin, self.window_sizes[li], kv_cache)
                    h_par.append(h_j)
                # Newton K=1 correction with H^res Jacobians
                h_corr = h_par[0]
                for j in range(1, n_par):
                    li = seq_layers + j
                    block = self.transformer.h[li]
                    H_res_attn = sinkhorn_log(block.hc_attn.H_res_logits, self.config.mhc_sinkhorn_iters, self.config.mhc_sinkhorn_tau)
                    H_res_mlp = sinkhorn_log(block.hc_mlp.H_res_logits, self.config.mhc_sinkhorn_iters, self.config.mhc_sinkhorn_tau)
                    J_block = (H_res_mlp @ H_res_attn).to(h_corr.dtype)  # (S, S)
                    delta = h_corr - h_init
                    BS_d, T_d, D_d = delta.shape
                    B_eff = BS_d // S
                    delta = delta.view(B_eff, S, T_d, D_d)
                    delta = torch.einsum('ij,bjkl->bikl', J_block, delta)
                    delta = delta.reshape(BS_d, T_d, D_d)
                    h_corr = h_par[j] + delta
                    # Stride: add loss at every stride-th layer and at the last layer
                    if mhc_stride > 0 and ((j + 1) % mhc_stride == 0 or j == n_par - 1):
                        h_seq_j = all_h[li].detach() if idn_detach else all_h[li]
                        diff = (h_corr.float() - h_seq_j.float()).norm(dim=-1)
                        ref_norm = h_seq_j.float().norm(dim=-1).clamp(min=1e-6)
                        mhc_newton_loss = mhc_newton_loss + (diff / ref_norm).mean()
                if mhc_stride <= 0:
                    # Original: loss only on the last layer
                    h_seq_final = all_h[n_layer - 1].detach() if idn_detach else all_h[n_layer - 1]
                    diff = (h_corr.float() - h_seq_final.float()).norm(dim=-1)
                    ref_norm = h_seq_final.float().norm(dim=-1).clamp(min=1e-6)
                    mhc_newton_loss = mhc_newton_loss + (diff / ref_norm).mean()

        # mHC stream reduction: (B*S, T, D) -> (B, T, D)
        if self.config.use_mhc:
            S = self.config.mhc_num_streams
            D = x.shape[-1]
            x = x.view(B, S, T, D).sum(dim=1)
            if x_backout is not None:
                x_backout = x_backout.view(B, S, T, D).sum(dim=1)

        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # reg_warmup_factor is a registered buffer (0-d tensor) updated externally
            # each step.  Using a tensor (not a Python float) avoids torch.compile
            # recompilation when the warmup value changes.
            warmup = self.reg_warmup_factor
            if jacobi_reg > 0:
                loss = loss + warmup * jacobi_reg * jacobi_reg_loss / n_layer
            if identity_newton_reg > 0:
                loss = loss + warmup * identity_newton_reg * identity_newton_loss
            if diag_newton_reg > 0:
                loss = loss + warmup * diag_newton_reg * diag_newton_loss
            if mhc_newton_reg > 0:
                loss = loss + warmup * mhc_newton_reg * mhc_newton_loss
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
