#!/usr/bin/env python3
"""Run a reference GCN in DGL/PyTorch and compare logits with the CUDA implementation."""

import argparse
import pathlib
import sys
import time
from typing import Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import GraphConv
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This script requires DGL. Install via pip install dgl-cu12.") from exc


class DGLGCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if layers == 1:
            self.layers.append(GraphConv(in_dim, num_classes, norm='both', weight=True, bias=True))
        else:
            self.layers.append(GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True))
            for _ in range(layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True))
            self.layers.append(GraphConv(hidden_dim, num_classes, norm='both', weight=True, bias=True))

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = feat
        for idx, layer in enumerate(self.layers):
            x = layer(graph, x)
            if idx + 1 < len(self.layers):
                x = F.relu(x)
        return x

def sync_weights(model):
    """Load weights from weights.bin and assign to DGL model to ensure consistency with C++."""
    if not os.path.exists("weights.bin"):
        print("Warning: weights.bin not found! Verification will likely FAIL.")
        return

    print("Loading weights from weights.bin to sync with C++...")

    w_flat = np.fromfile("weights.bin", dtype=np.float32)
    
    ptr = 0
    with torch.no_grad():
        for layer in model.layers:
            # Get the weight shape of the current layer (in_feats, out_feats)
            w_shape = layer.weight.shape
            w_size = w_shape[0] * w_shape[1]
            
            # Slice the corresponding segment from the flat array
            if ptr + w_size > w_flat.size:
                print("Error: weights.bin is smaller than model requirements!")
                break
                
            w_data = w_flat[ptr : ptr + w_size]
            ptr += w_size
            
            # Assign to PyTorch model
            # Note: Both C++ and PyTorch typically use row-major memory layout, so reshape directly
            layer.weight.data = torch.from_numpy(w_data).view(w_shape)
            
            # Important: C++ experiments usually do not compute Bias, so forcibly set Python Bias to 0
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

def load_graph(prefix: str) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    csr_path = pathlib.Path(prefix).with_suffix('.csr')
    feat_path = pathlib.Path(prefix).with_suffix('.feat')
    label_path = pathlib.Path(prefix).with_suffix('.label')

    csr_data = np.load(csr_path, allow_pickle=True)
    indptr = csr_data['indptr'].astype(np.int64)
    indices = csr_data['indices'].astype(np.int64)

    num_nodes = indptr.size - 1
    edges = dgl.graph(('csr', (indptr, indices, [])), num_nodes=num_nodes)
    features = torch.from_numpy(np.fromfile(feat_path, dtype=np.float32).reshape(num_nodes, -1))
    labels = torch.from_numpy(np.fromfile(label_path, dtype=np.int32))
    return edges, features, labels


def load_cuda_logits(path: pathlib.Path, num_nodes: int, num_classes: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.float32)
    expected = num_nodes * num_classes
    if raw.size != expected:
        raise ValueError(f"Expected {expected} floats in {path}, found {raw.size}")
    return torch.from_numpy(raw.copy()).view(num_nodes, num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cuSPARSE GCN outputs with DGL")
    parser.add_argument("--graph", required=True, help="Graph prefix (matches --graph used by C++ binary)")
    parser.add_argument("--hidden", type=int, required=True, help="Hidden dimension used in C++ run")
    parser.add_argument("--layers", type=int, required=True, help="Number of GCN layers")
    parser.add_argument("--outputs", required=True, help="Binary logits file dumped by C++")
    parser.add_argument("--tol", type=float, default=1e-3, help="Maximum allowed absolute difference")
    args = parser.parse_args()

    graph, features, labels = load_graph(args.graph)
    num_nodes = features.shape[0]
    num_classes = len(torch.unique(labels))

    model = DGLGCN(features.shape[1], args.hidden, num_classes, args.layers)
    sync_weights(model)
    model.eval()

    start = time.time()
    with torch.no_grad():
        torch_logits = model(graph, features)
    torch_elapsed = (time.time() - start) * 1000.0

    cpp_logits = load_cuda_logits(pathlib.Path(args.outputs), num_nodes, num_classes)
    diff = (torch_logits - cpp_logits).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    preds_torch = torch.argmax(torch_logits, dim=1)
    preds_cpp = torch.argmax(cpp_logits, dim=1)
    acc_torch = float((preds_torch == labels).float().mean().item())
    acc_cpp = float((preds_cpp == labels).float().mean().item())

    edges_per_ms = graph.num_edges() / torch_elapsed

    print(f"DGL   time(ms)={torch_elapsed:.2f} edges/ms={edges_per_ms:.2f} acc={acc_torch:.4f}")
    print(f"C++/cuSPARSE acc={acc_cpp:.4f} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    if max_diff > args.tol:
        print("Warning: difference exceeds tolerance", file=sys.stderr)
        sys.exit(1)
    print("Outputs match within tolerance")


if __name__ == "__main__":
    main()
