"""
Experiment 1: Symbolic FHE circuit-depth and cost accounting.

Compares standard sequential inference vs SNLP for various (n_par, K) configs.
Computes: NFE (nonlinear forward evaluations), total multiplicative depth,
number of bootstraps needed, and rotation counts.

No GPU needed — purely analytical.

Usage:
    uv run python -m fhe.exp1_circuit_depth --output cache/logs_fhe_claude/exp1_cost_table.json
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class HEOpCost:
    """FHE cost of a single operation."""
    name: str
    mult_depth: int
    ct_ct_mults: int
    ct_pt_mults: int
    additions: int
    rotations: int

    @property
    def total_mults(self):
        return self.ct_ct_mults + self.ct_pt_mults


def block_cost(D: int, n_head: int, head_dim: int, T: int, vocab_size: int,
               poly_degree: int = 4) -> dict:
    """Compute per-block FHE cost for each operation.

    poly_degree controls the degree of polynomial approximations for
    nonlinear operations (higher = more accurate but deeper circuit).
    """
    n_slots = D  # elements per ciphertext (CKKS packing)

    rms_norm_cost = HEOpCost(
        name='RMSNorm',
        mult_depth=poly_degree,  # degree-d polynomial for rsqrt
        ct_ct_mults=D + poly_degree,  # x^2 computation + polynomial eval
        ct_pt_mults=poly_degree,
        additions=D + poly_degree,
        rotations=int(D).bit_length(),  # log2(D) rotations for mean reduction
    )

    qk_norm_cost = HEOpCost(
        name='QK-Norm',
        mult_depth=poly_degree,
        ct_ct_mults=2 * (head_dim + poly_degree),  # Q and K separately
        ct_pt_mults=2 * poly_degree + 2,  # +2 for *1.2 scaling
        additions=2 * (head_dim + poly_degree),
        rotations=2 * int(head_dim).bit_length(),
    )

    softmax_cost = HEOpCost(
        name='Softmax',
        mult_depth=poly_degree + 2,  # poly_exp + normalization
        ct_ct_mults=T * n_head * poly_degree,  # per attention position
        ct_pt_mults=T * n_head * poly_degree,
        additions=T * n_head * (poly_degree + 1),
        rotations=int(T).bit_length() * n_head,  # for sum reduction
    )

    qkv_proj_cost = HEOpCost(
        name='QKV_proj',
        mult_depth=1,
        ct_ct_mults=0,
        ct_pt_mults=3 * D * D,  # Q, K, V projections
        additions=3 * D * D,
        rotations=3 * int(D).bit_length(),
    )

    o_proj_cost = HEOpCost(
        name='O_proj',
        mult_depth=1,
        ct_ct_mults=0,
        ct_pt_mults=D * D,
        additions=D * D,
        rotations=int(D).bit_length(),
    )

    relu_sq_cost = HEOpCost(
        name='ReLU_sq',
        mult_depth=2,  # sign approx (deg ~5-7) then square, but ReLU^2 = max(0,x)^2
        ct_ct_mults=D * 4,  # 4D intermediate dim
        ct_pt_mults=0,
        additions=D * 4,
        rotations=0,
    )

    fc_proj_cost = HEOpCost(
        name='FC_proj',
        mult_depth=1,
        ct_ct_mults=0,
        ct_pt_mults=D * 4 * D + 4 * D * D,  # up + down projections
        additions=D * 4 * D + 4 * D * D,
        rotations=2 * int(D).bit_length(),
    )

    sigmoid_cost = HEOpCost(
        name='Sigmoid_gate',
        mult_depth=poly_degree,
        ct_ct_mults=poly_degree * 12,  # small input dim (12)
        ct_pt_mults=poly_degree,
        additions=poly_degree + 12,
        rotations=0,
    )

    rope_cost = HEOpCost(
        name='RoPE',
        mult_depth=1,
        ct_ct_mults=0,
        ct_pt_mults=4 * head_dim * n_head,  # cos/sin are plaintext
        additions=2 * head_dim * n_head,
        rotations=0,
    )

    residual_cost = HEOpCost(
        name='Residual',
        mult_depth=0,
        ct_ct_mults=0,
        ct_pt_mults=2,  # resid_lambda * h + x0_lambda * x0
        additions=2,
        rotations=0,
    )

    all_ops = [rms_norm_cost, qk_norm_cost, softmax_cost, qkv_proj_cost,
               o_proj_cost, relu_sq_cost, fc_proj_cost, sigmoid_cost,
               rope_cost, residual_cost]

    total_mult_depth = (
        residual_cost.mult_depth           # residual blending
        + rms_norm_cost.mult_depth         # pre-attn norm
        + qkv_proj_cost.mult_depth         # QKV projection
        + rope_cost.mult_depth             # RoPE
        + qk_norm_cost.mult_depth          # QK norm
        + softmax_cost.mult_depth          # attention softmax
        + o_proj_cost.mult_depth           # O projection
        + rms_norm_cost.mult_depth         # pre-MLP norm
        + fc_proj_cost.mult_depth          # FC up projection
        + relu_sq_cost.mult_depth          # ReLU^2
        + fc_proj_cost.mult_depth          # FC down projection (already counted)
    )

    nonlinear_depth = (
        rms_norm_cost.mult_depth * 2       # 2x RMSNorm
        + qk_norm_cost.mult_depth          # QK norm
        + softmax_cost.mult_depth          # softmax
        + relu_sq_cost.mult_depth          # ReLU^2
        + sigmoid_cost.mult_depth          # sigmoid gate
    )

    return {
        'ops': {op.name: asdict(op) for op in all_ops},
        'total_block_mult_depth': total_mult_depth,
        'nonlinear_block_depth': nonlinear_depth,
        'total_ct_ct_mults': sum(op.ct_ct_mults for op in all_ops),
        'total_ct_pt_mults': sum(op.ct_pt_mults for op in all_ops),
        'total_additions': sum(op.additions for op in all_ops),
        'total_rotations': sum(op.rotations for op in all_ops),
    }


def idn_correction_cost(n_par: int, D: int) -> dict:
    """FHE cost of IDN prefix-sum correction (n_par layers)."""
    return {
        'mult_depth': 0,
        'ct_ct_mults': 0,
        'ct_pt_mults': n_par * D if False else 0,  # a=1.0 → no mults needed
        'additions': 2 * n_par * D,  # subtraction + addition per layer
        'rotations': 0,
    }


def compute_config_cost(L: int, n_par: int, K: int, D: int, n_head: int,
                        head_dim: int, T: int, vocab_size: int,
                        poly_degree: int = 4) -> dict:
    """Compute total FHE cost for a given (n_par, K) configuration."""
    seq_layers = L - n_par
    nfe = seq_layers + K  # nonlinear forward evaluations

    blk = block_cost(D, n_head, head_dim, T, vocab_size, poly_degree)
    corr = idn_correction_cost(n_par, D)

    # Standard CKKS modulus chain: each multiplication consumes one level
    # Typical budget: 10-20 levels before bootstrapping
    ckks_levels = 15  # typical budget

    # Sequential: depth = L * block_depth
    seq_total_depth = L * blk['total_block_mult_depth']
    seq_nonlinear_depth = L * blk['nonlinear_block_depth']
    seq_bootstraps = max(0, seq_total_depth // ckks_levels)

    # SNLP: depth = prefix_depth + K * (block_depth + correction_depth)
    prefix_depth = seq_layers * blk['total_block_mult_depth']
    snlp_iter_depth = blk['total_block_mult_depth'] + corr['mult_depth']
    snlp_total_depth = prefix_depth + K * snlp_iter_depth
    snlp_nonlinear_depth = seq_layers * blk['nonlinear_block_depth'] + K * blk['nonlinear_block_depth']
    snlp_bootstraps = max(0, snlp_total_depth // ckks_levels)

    # Correction cost across all K iterations
    total_corr_mults = K * corr['ct_ct_mults']
    total_corr_adds = K * corr['additions']

    # Total arithmetic (sequential uses L block evals, SNLP uses seq_layers + K * n_par)
    # But for depth, only K matter since n_par blocks are evaluated in parallel
    seq_total_block_evals = L
    snlp_total_block_evals = seq_layers + K * n_par  # total work (more than sequential if K*n_par > n_par)

    depth_reduction = seq_nonlinear_depth / max(snlp_nonlinear_depth, 1)
    nfe_reduction = L / max(nfe, 1)
    bootstrap_reduction = (seq_bootstraps + 1) / max(snlp_bootstraps + 1, 1)

    return {
        'config': {
            'L': L, 'n_par': n_par, 'K': K,
            'seq_layers': seq_layers, 'NFE': nfe,
            'poly_degree': poly_degree,
        },
        'sequential': {
            'total_depth': seq_total_depth,
            'nonlinear_depth': seq_nonlinear_depth,
            'bootstraps': seq_bootstraps,
            'block_evals': seq_total_block_evals,
        },
        'snlp': {
            'total_depth': snlp_total_depth,
            'nonlinear_depth': snlp_nonlinear_depth,
            'bootstraps': snlp_bootstraps,
            'block_evals': snlp_total_block_evals,
            'correction_mults': total_corr_mults,
            'correction_adds': total_corr_adds,
        },
        'reduction': {
            'depth': depth_reduction,
            'NFE': nfe_reduction,
            'bootstraps': bootstrap_reduction,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='cache/logs_fhe_claude/exp1_cost_table.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Model specs
    models = {
        'd32s (0.5B)': {'L': 32, 'D': 640, 'n_head': 5, 'head_dim': 128, 'T': 2048, 'vocab_size': 32768},
        'd32 (3B)':    {'L': 32, 'D': 2048, 'n_head': 16, 'head_dim': 128, 'T': 2048, 'vocab_size': 32768},
    }

    n_pars = [4, 8, 12, 16, 20, 24, 28]
    Ks = [1, 2, 4, 8]
    poly_degrees = [2, 4, 6]

    results = {}

    for model_name, spec in models.items():
        L = spec['L']
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (L={L}, D={spec['D']})")
        print(f"{'='*60}")

        # Block cost at deg-4
        blk = block_cost(spec['D'], spec['n_head'], spec['head_dim'], spec['T'], spec['vocab_size'], 4)
        print(f"\nPer-block cost (poly_degree=4):")
        print(f"  Total mult depth: {blk['total_block_mult_depth']}")
        print(f"  Nonlinear depth:  {blk['nonlinear_block_depth']}")
        print(f"  CT-CT mults:      {blk['total_ct_ct_mults']:,}")
        print(f"  CT-PT mults:      {blk['total_ct_pt_mults']:,}")
        print(f"  Rotations:        {blk['total_rotations']:,}")

        model_results = {}

        print(f"\n{'n_par':>5} {'K':>3} {'NFE':>4} {'seq_depth':>10} {'snlp_depth':>11} "
              f"{'depth_red':>10} {'NFE_red':>8} {'seq_boots':>10} {'snlp_boots':>11} {'boot_red':>9}")
        print('-' * 95)

        for n_par in n_pars:
            if n_par > L:
                continue
            for K in Ks:
                cost = compute_config_cost(poly_degree=4, **spec, n_par=n_par, K=K)
                key = f"n{n_par}_K{K}"
                model_results[key] = cost

                c = cost
                print(f"{n_par:5d} {K:3d} {c['config']['NFE']:4d} "
                      f"{c['sequential']['nonlinear_depth']:10d} "
                      f"{c['snlp']['nonlinear_depth']:11d} "
                      f"{c['reduction']['depth']:10.2f}x "
                      f"{c['reduction']['NFE']:8.2f}x "
                      f"{c['sequential']['bootstraps']:10d} "
                      f"{c['snlp']['bootstraps']:11d} "
                      f"{c['reduction']['bootstraps']:9.2f}x")

        results[model_name] = model_results

    # Print interesting configs (best NFE reduction)
    print(f"\n{'='*60}")
    print("FHE-optimal configs (sorted by NFE reduction):")
    print(f"{'='*60}")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        sorted_configs = sorted(model_results.items(),
                                key=lambda x: x[1]['reduction']['NFE'], reverse=True)
        for key, cost in sorted_configs[:10]:
            c = cost['config']
            r = cost['reduction']
            print(f"  {key:>10s}: NFE={c['NFE']:2d}, NFE_red={r['NFE']:.2f}x, "
                  f"depth_red={r['depth']:.2f}x, boot_red={r['bootstraps']:.2f}x")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
