#!/usr/bin/env python3
"""Verify the Step 6 full-forward-pass output against a numpy reference.

Usage:
    python3 scripts/check_forward.py --graph data/cora --hidden 128 --layers 2 [--tol 1e-3]

Reads:
    data/<graph>.csr, .feat, .label   (C++ binary format)
    weights.bin                        (dumped by ./bin/dgcn)
    outputs.bin                        (dumped by ./bin/dgcn --dump outputs.bin)

Computes the same forward pass in numpy and reports max/mean abs diff against
the CUDA dump. Also reports accuracy (argmax == label) for both for context.

Tolerance default 1e-3 — the full pass accumulates fp32 noise across two
SpMMs and two GEMMs plus ReLU, so it's looser than check_gemm.py.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from check_spmm import load_csr, load_features, compute_normalized_values  # noqa: E402


def spmm(AX_out, row_offsets, col_indices, values, X):
    AX_out[:] = 0
    n = row_offsets.size - 1
    for i in range(n):
        rs, re = row_offsets[i], row_offsets[i + 1]
        if rs == re:
            continue
        AX_out[i] = (values[rs:re, None] * X[col_indices[rs:re]]).sum(axis=0)


def layer_weight_dims(feat_dim, hidden, num_classes, layers):
    """Mirror the C++ layer_weight_dims helper."""
    if layers == 1:
        return [(feat_dim, num_classes)]
    dims = [(feat_dim, hidden)]
    for _ in range(layers - 2):
        dims.append((hidden, hidden))
    dims.append((hidden, num_classes))
    return dims


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--weights", default="weights.bin")
    p.add_argument("--outputs", default="outputs.bin")
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    num_nodes, nnz, row_offsets, col_indices = load_csr(args.graph)
    X = load_features(args.graph, num_nodes)
    values = compute_normalized_values(row_offsets, col_indices)
    labels = np.fromfile(args.graph + ".label", dtype=np.int32)
    num_classes = int(labels.max() + 1)

    feat_dim = X.shape[1]
    dims = layer_weight_dims(feat_dim, args.hidden, num_classes, args.layers)

    w_all = np.fromfile(args.weights, dtype=np.float32)
    total_expected = sum(i * o for i, o in dims)
    if w_all.size != total_expected:
        print(f"ERROR: weights.bin has {w_all.size} floats, expected {total_expected}",
              file=sys.stderr)
        sys.exit(1)
    weights = []
    off = 0
    for in_d, out_d in dims:
        weights.append(w_all[off : off + in_d * out_d].reshape(in_d, out_d))
        off += in_d * out_d

    h = X.astype(np.float64)
    for l, ((in_d, out_d), W) in enumerate(zip(dims, weights)):
        AH = np.empty_like(h)
        spmm(AH, row_offsets, col_indices, values.astype(np.float64), h)
        h = AH @ W.astype(np.float64)
        if l + 1 < len(dims):
            h = np.maximum(h, 0.0)
    Y_ref = h.astype(np.float32)

    Y_cuda = np.fromfile(args.outputs, dtype=np.float32).reshape(num_nodes, num_classes)

    diff = np.abs(Y_cuda - Y_ref)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    ref_scale = float(np.abs(Y_ref).max())

    pred_ref = Y_ref.argmax(axis=1)
    pred_cuda = Y_cuda.argmax(axis=1)
    acc_ref = float((pred_ref == labels).mean())
    acc_cuda = float((pred_cuda == labels).mean())

    print(f"n={num_nodes}  feat_dim={feat_dim}  H={args.hidden}  classes={num_classes}")
    print(f"Y_ref[0]  = {Y_ref[0]}")
    print(f"Y_cuda[0] = {Y_cuda[0]}")
    print(f"|Y_ref|_max = {ref_scale:.3e}")
    print(f"max_diff    = {max_diff:.3e}")
    print(f"mean_diff   = {mean_diff:.3e}")
    print(f"acc (numpy ref)   = {acc_ref:.4f}")
    print(f"acc (cuda output) = {acc_cuda:.4f}")

    if np.allclose(Y_cuda, 0):
        print("FAIL: Y_cuda is all zeros — forward pass not running?", file=sys.stderr)
        sys.exit(3)
    if not np.isfinite(Y_cuda).all():
        print("FAIL: Y_cuda contains non-finite values", file=sys.stderr)
        sys.exit(3)

    if max_diff > args.tol:
        print(f"FAIL: max_diff {max_diff:.3e} > tol {args.tol:.3e}", file=sys.stderr)
        sys.exit(2)
    print("PASS")


if __name__ == "__main__":
    main()
