# Layer-Parallel Inference Reduces Encrypted Nonlinear Depth in Transformers

Code for simulating FHE-friendly inference using SNLP (Structured Newton Layer Parallelism).

[![arXiv](https://img.shields.io/badge/arXiv-2605.17842-b31b1b.svg)](https://arxiv.org/abs/2605.17842)

## Overview

Standard Transformer inference under fully homomorphic encryption (FHE) requires sequentially composing $L$ nonlinear encrypted blocks, each approximated by polynomials. SNLP restructures this computation: instead of $L$ sequential nonlinear stages, it evaluates all parallel layers simultaneously in $K$ iterations with linear Newton-style corrections ($K \ll L$). The IDN correction is purely additive (zero FHE multiplicative depth), so the effective encrypted nonlinear depth reduces from $O(L)$ to $O(K)$.

This branch contains a simulation framework (`fhe/`) that replaces nonlinear operations with Chebyshev polynomial approximations and measures how approximation errors accumulate under sequential vs. SNLP inference. Key findings across 8 models and 4 architecture families:

- SNLP reduces bootstraps from 53 to 20 (2.65x) with +1.2% PPL degradation
- Error amplification is lower for SNLP (1.35x vs. 1.42x for sequential)
- Softmax is the dominant error source; degree-12 Chebyshev is the practical minimum
- CKKS arithmetic noise is negligible compared to polynomial approximation error
- mHC architectures are inherently more FHE-friendly (1.24x vs. 1.42x amplification)

## Usage

Set `NANOCHAT_BASE_DIR` to the directory containing `base_checkpoints/` before running.

### Sanity check (exact config matches standard forward)

```bash
uv run python -m fhe.he_forward
# Output: max_diff = 0.000000e+00 for both sequential and SNLP
```

### PPL evaluation under HE polynomial approximation

```bash
# Degree-12 approximation, selected configs
uv run python -m fhe.exp2_ppl_eval \
    --model-tag d32s_idn00625_npar24_s3 --degree 12 --max-tokens 200000

# Full (n_par, K) sweep
uv run python -m fhe.exp2_ppl_eval \
    --model-tag d32s_idn00625_npar24_s3 --degree 12 --sweep

# With CKKS noise overlay
uv run python -m fhe.exp2_ppl_eval \
    --model-tag d32s_idn00625_npar24_s3 --degree 12 --noise-bits 30

# mHC-Newton method
uv run python -m fhe.exp2_ppl_eval \
    --model-tag d32s_mhc4_x0ve_newton05 --degree 12 --method mHC-Newton
```

### Per-layer error tracking

```bash
uv run python -m fhe.exp2_error_tracking \
    --model-tag d32s_idn00625_npar24_s3 --n-par 24 --K 1 --degree 12 --n-batches 10
```

### Symbolic FHE cost table

```bash
uv run python -m fhe.exp1_circuit_depth  # no GPU needed
```

### Activation range calibration

```bash
uv run python -m fhe.exp_calibrate \
    --model-tag d32s_idn00625_npar24_s3 --n-par 24 --n-batches 20
```

## Citation

```bibtex
@article{han2026snlp,
  title={SNLP: Layer-Parallel Inference via Structured Newton Corrections},
  author={Han, Ligong and Xu, Kai and Wang, Hao and Srivastava, Akash},
  journal={arXiv preprint arXiv:2605.17842},
  year={2026}
}
```

## Acknowledgement

This codebase builds on several excellent open-source projects:

- [Nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [ELK](https://github.com/lindermanlab/elk) by Gonzalez et al.
- [mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections) by Xie et al.
- [SJD](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/) by Teng et al.

We thank the authors for generously open-sourcing their work.
