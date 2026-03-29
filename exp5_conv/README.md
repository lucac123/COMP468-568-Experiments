## 2D Convolution CUDA Experiment

This mini-lab targets senior undergraduates who already know CUDA basics and linear algebra. Students implement high-performance 2D convolutions on NVIDIA GPUs, starting from a harness that parses CLI arguments, seeds reproducible tensors, and times kernels. The provided source intentionally omits core CUDA logic so students can design and optimize their own kernels.

### Learning Goals
- Refresh 2D convolution math, tensor indexing, and receptive fields.
- Implement both naive and tiled shared-memory CUDA kernels.
- Reason about memory bandwidth vs. compute ceilings for stencil operators.
- Compare custom kernels against a CPU baseline and analyze accuracy/performance gaps.

### Directory Layout
- `src/main.cu` – entry point, tensor allocation, reference checker (TODOs tagged).
- `src/conv_kernel.cuh` – kernel declarations, launch helpers, shared-memory tiling skeletons.
- `scripts/measure.sh` – automation template for sweeping spatial sizes / filter counts.
- `data/` – create this to store CSV timing logs and validation summaries.

### Getting Started
1. Install CUDA 12+ and verify `nvcc --version` and `nvidia-smi`.
2. Ensure you have a compute capability ≥ 7.0 GPU (adjust `ARCH` in `Makefile` if needed).
3. Build the starter harness:
   ```bash
   make          # produces bin/dconv2d
   ./bin/dconv2d --height 256 --width 256 --channels 32 --filters 64 --ksize 3 --impl baseline
   ```
4. Read through every TODO in `src/main.cu` and `src/conv_kernel.cuh` before coding.

### Experiment Tasks
1. **Math & Pre-Lab**
   - Derive the output tensor shape given `stride`, `padding`, `H`, `W`, `Cin`, `Cout`, `K`.
   - Estimate FLOPs and bytes moved for your chosen workload; sketch the roofline point.
2. **Host Orchestration (`src/main.cu`)**
   - Implement the TODOs that initialize host tensors, move data to/from the GPU, select kernels, and compute GFLOP/s.
   - Complete the CPU reference convolution (already outlined) to verify device outputs.
3. **CUDA Kernels (`src/conv_kernel.cuh`)**
   - Finish the naive kernel where each thread computes one output pixel/filter pair.
   - Implement the tiled kernel that stages input patches/filters in shared memory (BLOCK_SIZE default 16, but feel free to tune).
   - Extend the launch helpers to support arbitrary tensor shapes and validate grid/block math.
4. **Performance Study**
   - Modify `scripts/measure.sh` to sweep {256, 512, 1024} spatial resolutions and multiple channel/filter configs.
   - Plot throughput (GFLOP/s) vs. problem size, and discuss occupancy / memory reuse limits.
5. **Stretch Ideas (optional)**
   - Add half-precision or tensor-core paths.
   - Explore Winograd or FFT-based convolution for large kernels.
   - Integrate cuDNN as a gold-standard comparison if available on your system.

### Deliverables
- Completed CUDA source with TODOs resolved and clearly documented kernels.
- A ≤5 page PDF report describing design decisions, performance analysis, and lessons learned.
- CSV/JSON logs under `data/` capturing your measurement sweeps.

### Rubric (20 pts)
- Correctness (6) – numerical parity vs. CPU baseline (max abs error ≤ 1e-3).
- Performance (6) – achieves ≥60% of GPU roofline prediction or shows thoughtful tuning.
- Analysis (4) – insightful discussion of memory hierarchy, tiling, or algorithmic trade-offs.
- Presentation (4) – clean code, readable plots, and concise writing.

### Suggested Timeline
| Day | Milestone |
|-----|-----------|
| 1   | Build harness, answer pre-lab worksheet |
| 2   | Implement & validate naive kernel |
| 3   | Implement shared-memory tiling + tuning |
| 4   | Run measurement sweeps, gather data |
| 5   | Write report, polish code/comments |

### Make Targets
```bash
make        # build bin/dconv2d
make clean  # remove build artifacts
```

### Academic Integrity
Discuss high-level strategies with classmates, but write your own code/report. Cite any external sources you leaned on.
