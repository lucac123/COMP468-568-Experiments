#!/usr/bin/env python3
"""Verify the Step 5 output: Y = (A_hat * X) * W0 computed in the CUDA binary.

Usage:
    python3 scripts/check_gemm.py --graph data/cora --hidden 128 [--tol 1e-4]

Reads:
    data/<graph>.csr, .feat  (C++ binary format)
    weights.bin              (concatenated layer weights dumped by ./bin/dgcn)
    gemm_out.bin             (Step 5 test-block output from ./bin/dgcn)

Produces a PASS/FAIL report comparing against a numpy reference.

Tolerance default is 1e-4 (not 1e-5 like check_spmm.py): a GEMM over K=1433
inner-dim accumulates meaningfully more rounding noise than a pure SpMM.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from check_spmm import load_csr, load_features, compute_normalized_values  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--hidden", type=int, required=True,
                   help="Hidden dim used by the C++ run")
    p.add_argument("--weights", default="weights.bin")
    p.add_argument("--gemm-out", default="gemm_out.bin")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    num_nodes, nnz, row_offsets, col_indices = load_csr(args.graph)
    X = load_features(args.graph, num_nodes)
    values = compute_normalized_values(row_offsets, col_indices)
    feat_dim = X.shape[1]
    H = args.hidden

    AX = np.zeros_like(X)
    for i in range(num_nodes):
        rs, re = row_offsets[i], row_offsets[i + 1]
        if rs == re:
            continue
        AX[i] = (values[rs:re, None] * X[col_indices[rs:re]]).sum(axis=0)

    w_all = np.fromfile(args.weights, dtype=np.float32)
    need = feat_dim * H
    if w_all.size < need:
        print(f"ERROR: weights.bin has {w_all.size} floats, need at least {need}",
              file=sys.stderr)
        sys.exit(1)
    W0 = w_all[:need].reshape(feat_dim, H)

    Y_ref = (AX.astype(np.float64) @ W0.astype(np.float64)).astype(np.float32)
    Y_cuda = np.fromfile(args.gemm_out, dtype=np.float32).reshape(num_nodes, H)

    diff = np.abs(Y_cuda - Y_ref)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ref_scale = float(np.abs(Y_ref).max())

    print(f"n={num_nodes}  feat_dim={feat_dim}  H={H}")
    print(f"Y_ref[0][:5]  = {Y_ref[0][:5]}")
    print(f"Y_cuda[0][:5] = {Y_cuda[0][:5]}")
    print(f"|Y_ref|_max = {ref_scale:.3e}  (scale of output)")
    print(f"max_diff    = {max_diff:.3e}")
    print(f"mean_diff   = {mean_diff:.3e}")

    rel = max_diff / max(ref_scale, 1e-12)
    print(f"rel_err     = {rel:.3e}")

    if max_diff > args.tol:
        print(f"FAIL: max_diff {max_diff:.3e} > tol {args.tol:.3e}", file=sys.stderr)
        sys.exit(2)
    print("PASS")


if __name__ == "__main__":
    main()
