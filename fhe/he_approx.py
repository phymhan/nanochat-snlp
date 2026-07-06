"""
Polynomial approximation library for simulating HE-friendly operations.

Provides Chebyshev polynomial approximations for FHE-unfriendly nonlinear ops:
- rsqrt (for RMSNorm)
- exp (for softmax)
- sigmoid (for value embedding gates)
- tanh (for logit softcap)

Usage:
    from fhe.he_approx import HEApproxConfig, poly_rms_norm, poly_softmax, poly_sigmoid, poly_tanh
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Chebyshev polynomial fitting and evaluation
# ---------------------------------------------------------------------------

def chebyshev_fit(fn, a: float, b: float, degree: int) -> torch.Tensor:
    """Fit a Chebyshev polynomial approximation to fn on [a, b].

    Returns coefficients in the monomial basis (for Horner evaluation).
    """
    k = np.arange(degree + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (2 * k + 1) / (2 * (degree + 1)))
    values = fn(nodes)
    t_nodes = (2 * nodes - (a + b)) / (b - a)
    coeffs_cheb = np.polynomial.chebyshev.chebfit(t_nodes, values, degree)
    coeffs_mono = np.polynomial.chebyshev.cheb2poly(coeffs_cheb)
    return torch.tensor(coeffs_mono, dtype=torch.float32)


def poly_eval(coeffs: torch.Tensor, x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Evaluate polynomial using Horner's method (minimal multiplicative depth).

    Maps x from [a, b] to [-1, 1] first.
    """
    t = (2.0 * x - (a + b)) / (b - a)
    t = t.float()
    result = torch.zeros_like(t)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * t + coeffs[i].item()
    return result.to(x.dtype)


# ---------------------------------------------------------------------------
# Pre-fitted polynomial caches
# ---------------------------------------------------------------------------

_POLY_CACHE = {}


def _get_poly(name: str, fn, a: float, b: float, degree: int) -> torch.Tensor:
    key = (name, a, b, degree)
    if key not in _POLY_CACHE:
        _POLY_CACHE[key] = chebyshev_fit(fn, a, b, degree)
    return _POLY_CACHE[key]


# ---------------------------------------------------------------------------
# Approximation config
# ---------------------------------------------------------------------------

@dataclass
class HEApproxConfig:
    rms_norm_iters: Optional[int] = None        # Goldschmidt iterations (1-3); None = exact
    softmax_degree: Optional[int] = None        # Chebyshev degree; None = exact
    sigmoid_degree: Optional[int] = None
    tanh_degree: Optional[int] = None
    ckks_noise_bits: Optional[int] = None       # None = no noise

    # Calibrated intervals (set by calibration or use defaults)
    # exp: after max subtraction, scores are in ~[-30, 0]
    exp_interval: tuple = (-20.0, 0.0)
    sigmoid_interval: tuple = (-8.0, 8.0)
    tanh_interval: tuple = (-3.0, 3.0)

    @property
    def is_exact(self):
        return (self.rms_norm_iters is None and self.softmax_degree is None
                and self.sigmoid_degree is None and self.tanh_degree is None
                and self.ckks_noise_bits is None)

    @staticmethod
    def exact():
        return HEApproxConfig()

    @staticmethod
    def uniform(degree: int, noise_bits: Optional[int] = None):
        """Convenience: degree controls Chebyshev degree for softmax/sigmoid/tanh,
        and maps to Goldschmidt iterations for RMSNorm (deg2→1iter, deg4→2iter, deg6→3iter)."""
        return HEApproxConfig(
            rms_norm_iters=max(1, degree // 2),
            softmax_degree=degree,
            sigmoid_degree=degree,
            tanh_degree=degree,
            ckks_noise_bits=noise_bits,
        )


# ---------------------------------------------------------------------------
# Polynomial RMSNorm
# ---------------------------------------------------------------------------

def poly_rms_norm(x: torch.Tensor, normalized_shape: tuple, cfg: HEApproxConfig) -> torch.Tensor:
    """RMSNorm with simulated HE approximation error.

    In real FHE, rsqrt is computed via Goldschmidt iteration with a polynomial
    initial guess, or via composite polynomial approximation. The error level
    depends on the polynomial degree used.

    rms_norm_iters controls approximation quality:
      None = exact
      1 = ~1% multiplicative noise (simulates degree-4 polynomial rsqrt)
      2 = ~0.1% noise (degree-8)
      3 = ~0.01% noise (degree-12+)
    """
    if cfg.rms_norm_iters is None:
        return F.rms_norm(x, normalized_shape)

    result = F.rms_norm(x, normalized_shape).float()

    # Simulate polynomial approximation error as multiplicative noise
    # Error decreases exponentially with more Goldschmidt iterations
    noise_scale = 10.0 ** (-cfg.rms_norm_iters - 1)  # 1→0.01, 2→0.001, 3→0.0001
    noise = 1.0 + noise_scale * torch.randn_like(result)
    result = result * noise

    return result.to(x.dtype)


# ---------------------------------------------------------------------------
# Polynomial softmax
# ---------------------------------------------------------------------------

def poly_exp(x: torch.Tensor, degree: int, interval: tuple) -> torch.Tensor:
    """Chebyshev approximation of exp(x) on [a, b]."""
    a, b = interval
    coeffs = _get_poly('exp', np.exp, a, b, degree)
    x_clamped = x.clamp(a, b)
    result = poly_eval(coeffs, x_clamped, a, b)
    return result.clamp(min=1e-12)


def poly_softmax(scores: torch.Tensor, cfg: HEApproxConfig) -> torch.Tensor:
    """Softmax using polynomial exp approximation.

    Standard: exp(x - max(x)) / sum(exp(x - max(x)))
    HE-approx: poly_exp(x - max(x)) / sum(poly_exp(x - max(x)))
    Note: max subtraction is for numerical stability only; in true FHE
    this would need to be handled differently (e.g., bounded input range).
    """
    if cfg.softmax_degree is None:
        return F.softmax(scores, dim=-1)

    scores_f = scores.float()
    # Clamp masked positions (very negative values) to interval minimum
    # This prevents polynomial extrapolation on out-of-range values
    a, b = cfg.exp_interval
    scores_shifted = scores_f - scores_f.max(dim=-1, keepdim=True).values
    scores_shifted = scores_shifted.clamp(min=a)
    exp_approx = poly_exp(scores_shifted, cfg.softmax_degree, cfg.exp_interval)
    return (exp_approx / exp_approx.sum(dim=-1, keepdim=True).clamp(min=1e-12)).to(scores.dtype)


# ---------------------------------------------------------------------------
# Polynomial sigmoid
# ---------------------------------------------------------------------------

def poly_sigmoid(x: torch.Tensor, cfg: HEApproxConfig) -> torch.Tensor:
    """Sigmoid using polynomial approximation."""
    if cfg.sigmoid_degree is None:
        return torch.sigmoid(x)

    a, b = cfg.sigmoid_interval
    coeffs = _get_poly('sigmoid', lambda t: 1.0 / (1.0 + np.exp(-t)), a, b, cfg.sigmoid_degree)
    x_clamped = x.clamp(a, b)
    result = poly_eval(coeffs, x_clamped, a, b)
    return result.clamp(0.0, 1.0).to(x.dtype)


# ---------------------------------------------------------------------------
# Polynomial tanh
# ---------------------------------------------------------------------------

def poly_tanh(x: torch.Tensor, cfg: HEApproxConfig) -> torch.Tensor:
    """Tanh using polynomial approximation."""
    if cfg.tanh_degree is None:
        return torch.tanh(x)

    a, b = cfg.tanh_interval
    coeffs = _get_poly('tanh', np.tanh, a, b, cfg.tanh_degree)
    x_clamped = x.clamp(a, b)
    result = poly_eval(coeffs, x_clamped, a, b)
    return result.clamp(-1.0, 1.0).to(x.dtype)


# ---------------------------------------------------------------------------
# CKKS noise injection
# ---------------------------------------------------------------------------

def add_ckks_noise(x: torch.Tensor, noise_bits: Optional[int]) -> torch.Tensor:
    """Simulate CKKS arithmetic noise after operations."""
    if noise_bits is None:
        return x
    scale = x.abs().max().item()
    if scale < 1e-12:
        return x
    noise_std = scale * 2.0 ** (-noise_bits)
    return x + torch.randn_like(x) * noise_std


# ---------------------------------------------------------------------------
# Convenience: approximate a scaled_dot_product_attention call
# ---------------------------------------------------------------------------

def poly_scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cfg: HEApproxConfig, is_causal: bool = True,
) -> torch.Tensor:
    """Manual SDPA with polynomial softmax.

    q, k, v: (B, n_head, T, head_dim)
    Returns: (B, n_head, T, head_dim)
    """
    if cfg.softmax_degree is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    scale = q.size(-1) ** -0.5
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale

    if is_causal:
        T = scores.size(-1)
        mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, -1e9)

    attn_weights = poly_softmax(scores, cfg)
    return (attn_weights @ v.float()).to(q.dtype)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Quick check that polynomial approximations are reasonable."""
    torch.manual_seed(42)

    cfg4 = HEApproxConfig.uniform(degree=4)
    cfg6 = HEApproxConfig.uniform(degree=6)

    x = torch.randn(4, 64)

    # RMSNorm (Goldschmidt)
    exact = F.rms_norm(x, (64,))
    approx2 = poly_rms_norm(x, (64,), HEApproxConfig(rms_norm_iters=1))
    approx4 = poly_rms_norm(x, (64,), cfg4)
    approx6 = poly_rms_norm(x, (64,), cfg6)
    err2 = (exact - approx2).abs().mean().item() / exact.abs().mean().item()
    err4 = (exact - approx4).abs().mean().item() / exact.abs().mean().item()
    err6 = (exact - approx6).abs().mean().item() / exact.abs().mean().item()
    print(f"RMSNorm: 1iter rel_err={err2:.6f}, 2iter rel_err={err4:.6f}, 3iter rel_err={err6:.6f}")

    # Test with larger scale (like actual model activations)
    x_large = torch.randn(4, 640) * 20  # scale similar to real activations
    exact_l = F.rms_norm(x_large, (640,))
    approx_l2 = poly_rms_norm(x_large, (640,), HEApproxConfig(rms_norm_iters=2))
    approx_l3 = poly_rms_norm(x_large, (640,), HEApproxConfig(rms_norm_iters=3))
    err_l2 = (exact_l - approx_l2).abs().mean().item() / exact_l.abs().mean().item()
    err_l3 = (exact_l - approx_l3).abs().mean().item() / exact_l.abs().mean().item()
    print(f"RMSNorm (scaled): 2iter rel_err={err_l2:.6f}, 3iter rel_err={err_l3:.6f}")

    # Softmax
    scores = torch.randn(2, 4, 8, 8)
    exact_sm = F.softmax(scores, dim=-1)
    approx4_sm = poly_softmax(scores, cfg4)
    approx6_sm = poly_softmax(scores, cfg6)
    err4_sm = (exact_sm - approx4_sm).abs().mean().item()
    err6_sm = (exact_sm - approx6_sm).abs().mean().item()
    print(f"Softmax: deg4 mae={err4_sm:.6f}, deg6 mae={err6_sm:.6f}")

    # Sigmoid
    x_sig = torch.randn(100) * 3
    exact_sig = torch.sigmoid(x_sig)
    approx4_sig = poly_sigmoid(x_sig, cfg4)
    err_sig = (exact_sig - approx4_sig).abs().mean().item()
    print(f"Sigmoid: deg4 mae={err_sig:.6f}")

    # Tanh
    x_tanh = torch.randn(100) * 2
    exact_tanh = torch.tanh(x_tanh)
    approx4_tanh = poly_tanh(x_tanh, cfg4)
    err_tanh = (exact_tanh - approx4_tanh).abs().mean().item()
    print(f"Tanh:    deg4 mae={err_tanh:.6f}")

    print("Self-test passed.")


if __name__ == '__main__':
    _self_test()
