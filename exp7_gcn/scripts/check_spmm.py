#!/usr/bin/env python3
"""Compute Y = A_hat * X in numpy and compare against spmm_out.bin from the CUDA run.

Usage:
    python3 scripts/check_spmm.py --graph data/cora [--tol 1e-5]

What it does:
    1. Loads the CSR graph and features from the C++-format binaries.
    2. Computes the symmetric normalization coefficients the same way
       build_graph_from_files does (1 / sqrt(deg[i] * deg[j])).
    3. Materializes A_hat as a scipy sparse matrix and computes A_hat @ X.
    4. Loads spmm_out.bin (produced by the Step 4 test block in main.cu).
    5. Reports max/mean absolute difference.

Run this after `./bin/dgcn --graph data/cora ...` creates spmm_out.bin.
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def load_csr(prefix: str):
    """Load the C++-format CSR binary."""
    raw = open(prefix + ".csr", "rb").read()
    num_nodes, nnz = np.frombuffer(raw[:8], dtype=np.int32)
    off_bytes = (num_nodes + 1) * 4
    row_offsets = np.frombuffer(raw[8 : 8 + off_bytes], dtype=np.int32)
    col_indices = np.frombuffer(raw[8 + off_bytes : 8 + off_bytes + nnz * 4], dtype=np.int32)
    return int(num_nodes), int(nnz), row_offsets, col_indices


def load_features(prefix: str, num_nodes: int) -> np.ndarray:
    raw = np.fromfile(prefix + ".feat", dtype=np.float32)
    feature_dim = raw.size // num_nodes
    assert raw.size == num_nodes * feature_dim, "Feature file size inconsistent"
    return raw.reshape(num_nodes, feature_dim)


def compute_normalized_values(row_offsets: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
    """Match compute_symmetric_normalization from gcn_layers.cuh."""
    num_nodes = row_offsets.size - 1
    deg = (row_offsets[1:] - row_offsets[:-1]).astype(np.float32)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    row_ids = np.repeat(np.arange(num_nodes, dtype=np.int32),
                        row_offsets[1:] - row_offsets[:-1])
    values = inv_sqrt[row_ids] * inv_sqrt[col_indices]
    return values.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, help="Graph prefix (e.g. data/cora)")
    p.add_argument("--spmm-out", default="spmm_out.bin",
                   help="Path to the SpMM dump produced by the C++ run")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="Max allowed absolute difference")
    args = p.parse_args()

    num_nodes, nnz, row_offsets, col_indices = load_csr(args.graph)
    X = load_features(args.graph, num_nodes)
    values = compute_normalized_values(row_offsets, col_indices)

    Y_ref = np.zeros_like(X)
    for i in range(num_nodes):
        rs, re = row_offsets[i], row_offsets[i + 1]
        if rs == re:
            continue
        cols = col_indices[rs:re]
        vals = values[rs:re]
        Y_ref[i] = (vals[:, None] * X[cols]).sum(axis=0)

    out_path = Path(args.spmm_out)
    if not out_path.exists():
        print(f"ERROR: {out_path} not found. Did you run ./bin/dgcn first?", file=sys.stderr)
        sys.exit(1)
    Y_cuda = np.fromfile(out_path, dtype=np.float32).reshape(num_nodes, -1)
    if Y_cuda.shape != Y_ref.shape:
        print(f"ERROR: shape mismatch: cuda {Y_cuda.shape} vs ref {Y_ref.shape}", file=sys.stderr)
        sys.exit(1)

    diff = np.abs(Y_cuda - Y_ref)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"n={num_nodes}  nnz={nnz}  feat_dim={X.shape[1]}")
    print(f"Y_ref[0][:5]  = {Y_ref[0][:5]}")
    print(f"Y_cuda[0][:5] = {Y_cuda[0][:5]}")
    print(f"max_diff  = {max_diff:.3e}")
    print(f"mean_diff = {mean_diff:.3e}")

    if max_diff > args.tol:
        print(f"FAIL: max_diff {max_diff:.3e} > tol {args.tol:.3e}", file=sys.stderr)
        sys.exit(2)
    print("PASS")


if __name__ == "__main__":
    main()
