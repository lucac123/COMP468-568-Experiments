## cuSPARSE Graph Convolution Mini-Lab

This experiment asks senior undergraduates to rebuild a two-layer GCN (Kipf & Welling) using CUDA primitives. Students ingest a sparse graph in CSR, implement normalized message passing with cuSPARSE SpMM, and stitch on linear transforms (cuBLAS) plus activations. A provided CLI harness orchestrates I/O, timing, and logging, but key GPU plumbing is left as TODOs. To ground the results, you will also run the same architecture in DGL/PyTorch and compare logits + throughput.

### Learning Goals
- Represent large graphs efficiently on GPUs (CSR/COO transforms, normalization, batching).
- Use cuSPARSE `SpMM` / `SpMM_segmented` to implement GCN aggregation with shared memory staging when needed.
- Integrate custom SpMM with dense cuBLAS layers, activations, and loss evaluation.
- Benchmark vs. DGL: analyze accuracy gaps, kernel-level performance, and memory footprint.

### Directory Layout
- `src/main.cu` – CLI parsing, data loading stubs, cuSPARSE/cuBLAS handle management, training loop skeleton.
- `src/gcn_layers.cuh` – helper structs for CSR graphs, normalization utilities, cuSPARSE launch wrappers (TODO-heavy).
- `scripts/measure.sh` – batch automation for sweeping datasets / hidden dims / precision.
- `scripts/prepare_data.py` – downloads and converts a citation network (Cora or Citeseer) into the binary formats consumed by C++ and the DGL comparison script. Run once before building.
- `scripts/compare_with_dgl.py` – DGL-based reference implementation that reads dumped logits and reports diff + perf.
- `Makefile` – builds `bin/dgcn` and links CUDA, cuSPARSE, cuBLAS.
- `data/` – create this directory for graph binaries and measurement CSVs.

### Getting Started
1. Install CUDA 12+, cuSPARSE, cuBLAS, and Python with DGL + PyTorch.
2. Prepare graph data (Cora shown; replace `--dataset citeseer` for Citeseer):
   ```bash
   mkdir -p data
   python scripts/prepare_data.py --out data
   ```
   This produces two file sets under `data/`:
   - **C++ format** (`cora.csr`, `cora.feat`, `cora.label`): `cora.csr` is raw binary containing `[num_nodes, nnz]` (int32) followed by `row_offsets` (int32) and `col_indices` (int32). `cora.feat` is row-major float32 features. `cora.label` is int32 node labels.
   - **DGL format** (`cora_dgl.csr`, `cora_dgl.feat`, `cora_dgl.label`): `cora_dgl.csr` is a pickle dict with keys `indptr`, `indices`, `data` (loadable via `np.load(path, allow_pickle=True)`). Feature and label files use the same raw format.
3. Build the harness:
   ```bash
   make
   ./bin/dgcn --graph data/cora --hidden 128 --layers 2 --impl baseline --dump outputs.bin --no-verify
   ```
4. Open `src/main.cu` and `src/gcn_layers.cuh` to review all TODO tags before editing.

### Experiment Tasks
1. **Graph Prep & Pre-Lab**
   - Implement or script the normalization constants \(\hat{A} = D^{-1/2}(A + I)D^{-1/2}\).
   - Confirm CSR dimensions, nnz, and feature shapes; answer the worksheet on memory footprint & FLOPs.
2. **cuSPARSE Aggregation (`src/gcn_layers.cuh`)**
   - Implement `build_graph_from_files` to read CSR/feature binaries into host buffers, then copy to device.
   - Fill in `run_sparse_dense_mm` using `cusparseSpMM` with descriptors for graph + dense feature matrix.
   - Extend `apply_activation`, `apply_dropout`, and `softmax_cross_entropy` helpers as needed.
3. **Host Orchestration (`src/main.cu`)**
   - Allocate device buffers for features, intermediate activations, weights, and logits.
   - Implement the forward pass: normalized sparse aggregation → dense weight multiply (cuBLAS) → activation → repeat.
   - Add timing with CUDA events and compute throughput (edges/sec, GFLOP/s).
   - Implement optional backward/SGD loop or keep it inference-only (document your choice).
4. **DGL Baseline (`scripts/compare_with_dgl.py`)**
   - The provided script loads the same graph/features, builds a DGL `GraphConv` stack, and compares logits.
   - Complete any TODOs inside the Python script (dataset loading, accuracy computation) and ensure it consumes the binary dump from C++.
5. **Performance Study**
   - Use `scripts/measure.sh` to sweep `{Cora, Citeseer}` and hidden sizes {64, 128, 256} for both `baseline` and `fused` implementations.
   - Report wall-clock, effective SpMM bandwidth, and accuracy vs. DGL.
   - Explain any deltas between cuSPARSE and DGL kernels (e.g., layout, fusion, caching).
6. **Report**
   - ≤6 pages covering methodology, profiling, accuracy tables, and lessons learned. Include both C++ and DGL metrics.

### Deliverables
- Completed CUDA source with TODOs resolved.
- Measurement CSVs / plots comparing cuSPARSE and DGL across configs.
- Final report discussing correctness, performance, and design decisions.

### Rubric (20 pts)
- Correctness (6) – cuSPARSE logits match DGL reference within 1e-3 absolute error, and classification accuracy is within 2 percentage points of DGL on both Cora and Citeseer.
- Performance (6) – achieves ≥80% of DGL throughput (edges/sec) on Cora, or provides a profiling-backed explanation for the gap.
- Analysis (4) – insightful discussion of sparse formats, algorithm choice, and memory behavior.
- Presentation (4) – clean code, reproducible plots, and concise writing.

### Suggested Timeline
| Day | Milestone |
|-----|-----------|
| 1   | Prep datasets, inspect TODOs, answer pre-lab worksheet |
| 2   | Implement CSR loading + cuSPARSE descriptors |
| 3   | Finish forward pass + verification hooks |
| 4   | Run sweeps, profile kernels, analyze vs. DGL |
| 5   | Write report, finalize plots/code |

### DGL Comparison Workflow
1. Run your CUDA binary with dumping enabled: `./bin/dgcn --graph data/cora --hidden 128 --layers 2 --dump outputs.bin --impl baseline`.
2. Execute `python scripts/compare_with_dgl.py --graph data/cora_dgl --hidden 128 --layers 2 --outputs outputs.bin --tol 1e-3` to compare logits and gather DGL timing. The script prints accuracy, max error, and throughput for both paths. Note: use the `_dgl` prefix for the Python script (pickle format) and the plain prefix for the C++ binary (raw binary format).

### Make Targets
```bash
make        # build bin/dgcn
make clean  # remove bin/
```

### Academic Integrity
Discuss high-level strategies with classmates, but all code/report content must be your own. Cite any external references (papers, repos, blog posts) that informed your design.
