## LeNet GPU Mini-Lab (cuDNN + cuBLAS)

This lab guides senior undergraduates through rebuilding the classic LeNet-5 inference stack using NVIDIA libraries. Students wire together cuDNN convolution/pooling descriptors and cuBLAS GEMMs to implement the forward pass for grayscale images (e.g., MNIST). The harness parses CLI arguments, seeds reproducible tensors, and outlines timing/verification, but leaves key CUDA+library plumbing for students to fill in.

### Learning Goals
- Configure cuDNN tensor, convolution, activation, and pooling descriptors.
- Launch cuBLAS GEMMs for fully-connected layers and manage data layouts between conv and dense stages.
- Benchmark latency/throughput, reason about workspace memory, and validate numerics against a CPU (or PyTorch-exported) baseline.
- Experiment with alternative convolution algorithms and fusions.

### Directory Layout
- `src/main.cu` – entry point, data orchestration, cuDNN/cuBLAS handle management (TODOs inside).
- `src/lenet_layers.cuh` – helper structs, descriptor builders, and kernel launch wrappers for conv, pool, activation, and GEMM layers (TODOs inside).
- `scripts/measure.sh` – template for sweeping batch sizes, convolution algorithms, and precision modes.
- `Makefile` – builds `bin/dlenet` and links CUDA, cuDNN, cuBLAS.
- `data/` – create to stash CSV logs or verification dumps.

### Getting Started
1. Install CUDA 12+ plus matching cuDNN (ensure headers/libs visible to `nvcc`).
2. Update `ARCH` in the `Makefile` to match your GPU (e.g., `sm_90`).
3. Build & smoke-test the untouched harness:
   ```bash
   make
   ./bin/dlenet --batch 32 --algo implicit_gemm --impl baseline
   ```
4. Read every TODO in `src/main.cu` and `src/lenet_layers.cuh` before modifying anything.

### Experiment Tasks
1. **Architecture Refresher**
   - Re-derive the output shapes for the canonical LeNet stack (Conv(6x5x5) → Pool → Conv(16x5x5) → Pool → FC 120 → FC 84 → FC 10).
   - Compute total FLOPs and memory footprint per batch; identify hot layers.
2. **cuDNN Layer Plumbing (`src/lenet_layers.cuh`)**
   - Implement helpers that create/destroy tensor, filter, convolution, activation, and pooling descriptors for each layer.
   - Fill in `run_lenet_conv` to select a cuDNN forward algorithm (based on CLI `--algo`) and launch it with the proper workspace.
   - Implement pooling and activation wrappers (e.g., tanh/ReLU) that match the LeNet spec.
   - Write the GEMM helpers using cuBLAS for the dense layers and integrate bias additions.
3. **Host Orchestration (`src/main.cu`)**
   - Allocate/copy device buffers for activations, weights, biases, and cuDNN workspace.
   - Implement the LeNet forward loop: conv1 → pool1 → activation → conv2 → pool2 → reshape → fc1 → fc2 → fc3.
   - Record CUDA events around inference to report latency + GFLOP/s.
   - Complete the CPU reference (or optional PyTorch checker described below) to verify GPU outputs on small batches.
4. **Performance Study**
   - Extend `scripts/measure.sh` to sweep batch sizes {32, 64, 128} and algorithms {implicit_gemm, implicit_precomp, fft} where supported.
   - Record runtime, effective GFLOP/s, and workspace bytes for each configuration.
   - Compare the provided `baseline` (library-only) path vs. an optional `fused` path where you combine activation+pooling or fuse bias into GEMMs.
5. **Stretch Goals (optional)**
   - Add FP16/TF32 mixed-precision support using cuDNN math types and cublasGemmEx.
   - Fuse the last conv block with the first FC layer via an im2col trick.
   - Export intermediate activations and visualize them in Python.

### Deliverables
- Completed CUDA source with TODOs resolved (both main and helper files).
- Measurement logs (CSV) showing sweeps across batch sizes/algorithms.
- A ≤6-page PDF report summarizing architecture choices, performance findings, and lessons learned.

### Rubric (20 pts)
- Correctness (6) – GPU logits match CPU/PyTorch baseline to ≤1e-3 absolute error.
- Performance (6) – Target ≥70% of cuDNN's own implicit-GEMM throughput (i.e., the peak GFLOP/s you observe when calling `cudnnConvolutionForward` with `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`) for batch 128 on your GPU. This is a soft target, not a hard pass/fail gate; partial credit is awarded based on effort, profiling, and analysis of any gap. Report your measured GFLOP/s for each algorithm and batch size.
- Analysis (4) – Insightful discussion of workspace limits, algorithm choices, and bottlenecks.
- Presentation (4) – Clear code, plots, and written exposition.

### Suggested Timeline
| Day | Milestone |
|-----|-----------|
| 1   | Install deps, review LeNet math, inspect TODOs |
| 2   | Implement descriptor builders + conv/pool wrappers |
| 3   | Implement dense layers + CPU reference |
| 4   | Run benchmarking sweeps, tune algorithms |
| 5   | Write report, polish code/comments |

### PyTorch Output Checker
After your C++ code can dump logits (see `--dump outputs.bin` flag in `src/main.cu`), you can cross-check against a deterministic PyTorch reference that mirrors the harness seeding:
```bash
./bin/dlenet --batch 32 --impl baseline --dump outputs.bin --no-verify
python scripts/verify_with_pytorch.py --batch 32 --output outputs.bin --tol 1e-3
```
The script reconstructs the same synthetic inputs/weights used by the harness and reports the max absolute error. Passing this check is a strong signal that your CUDA + cuDNN plumbing is numerically correct.

### Make Targets
```bash
make        # build bin/dlenet
make clean  # remove bin/
```

### Academic Integrity
Collaborate on high-level ideas only. All code/report content must be your own, and you must cite any outside references (papers, repos, blog posts) that influenced your implementation.
