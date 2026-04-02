
// spmm_baseline.cu — Two-Step GNN: SDDMM + SpMM (STUDENT SKELETON)
#include <cassert>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename KernelLauncher>
float time_kernel_ms(KernelLauncher launch, int warmup = 5, int iters = 50) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return total_ms / iters;
}

extern void load_csr_from_edgelist(const std::string &filename, int &M, int &K,
                                   std::vector<int> &row_ptr,
                                   std::vector<int> &col_idx,
                                   std::vector<float> &vals);

extern void sddmm_cpu(int M, int D, const std::vector<int> &row_ptr,
                      const std::vector<int> &col_idx,
                      const std::vector<float> &E,
                      std::vector<float> &vals_out);

extern void spmm_cpu(int M, int K, int N, const std::vector<int> &row_ptr,
                     const std::vector<int> &col_idx,
                     const std::vector<float> &vals,
                     const std::vector<float> &B, std::vector<float> &C);

extern float max_abs_err(const std::vector<float> &A,
                         const std::vector<float> &B);

using float_t = float;

/*
===============================================================
 SDDMM BASELINE KERNEL — STUDENT DONE
 One thread per nonzero edge.
 row_indices[p] gives the source row for edge p.
===============================================================
*/
__global__ void sddmm_csr_baseline_kernel(
    int nnz, int D,
    const int *__restrict__ d_row_indices, // nnz: row index for each edge
    const int *__restrict__ d_col_idx,     // nnz: column index for each edge
    const float_t *__restrict__ d_E,       // M x D embedding matrix
    float_t *__restrict__ d_vals)          // nnz: output edge weights
{
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= nnz)
    return;

  // Compute dot(E[i,:], E[j,:]) = d_[p]

  // DONE student: read source row i from d_row_indices[p]
  int i = d_row_indices[p];
  // DONE student: read destination column j from d_col_idx[p]
  int j = d_col_idx[p];
  // DONE student: compute dot product of E[i,:] and E[j,:] over D dimensions
  float_t prod = 0.0f;

  int e_i_start = i * D;
  int e_j_start = j * D;

  for (int k = 0; k < D; k++) {
    prod += d_E[e_i_start + k] * d_E[e_j_start + k];
  }
  // DONE student: write result to d_vals[p]
  d_vals[p] = prod;
}

/*
===============================================================
 SpMM BASELINE KERNEL — STUDENT DONE
 One thread per row of the output matrix.
 Computes C[row,:] = sum over nonzeros in row of val * E[col,:]
===============================================================
*/
__global__ void spmm_csr_row_kernel(int M, int N,
                                    const int *__restrict__ d_row_ptr,
                                    const int *__restrict__ d_col_idx,
                                    const float_t *__restrict__ d_vals,
                                    const float_t *__restrict__ d_B,
                                    float_t *__restrict__ d_C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  // DONE: student: init output row
  int out_start = row * N;
  for (int i = 0; i < N; i++) {
    d_C[out_start + i] = 0;
  }

  // DONE: student: load start, end
  int start, end;
  start = d_row_ptr[row];
  end = d_row_ptr[row + 1];

  // DONE: student: loop over p in row nnz
  for (int i = start; i < end; i++) {
    int k = d_col_idx[i];

    float_t v = d_vals[i];

    // DONE: student: accumulate into d_C[row*N + j]
    for (int j = 0; j < N; j++) {
      float_t b_val = d_B[k * N + j];

      d_C[out_start + j] += v * b_val;
    }
  }
}

int main(int argc, char **argv) {
  int M, K;
  int D = 64;
  if (argc > 1)
    D = std::atoi(argv[1]);

  std::vector<int> row_ptr, col_idx;
  std::vector<float> vals;

  load_csr_from_edgelist("graph_edges.txt", M, K, row_ptr, col_idx, vals);
  int nnz = row_ptr.back();
  assert(M == K && "Adjacency matrix must be square");

  std::cout << "Loaded graph: M=" << M << " nnz=" << nnz << " D=" << D << "\n";

  // --- Generate random embedding E (M x D) ---
  std::vector<float> E((size_t)M * D);
  srand(42);
  for (size_t i = 0; i < E.size(); i++)
    E[i] = float(rand()) / RAND_MAX;

  // --- Build row_indices array (for baseline SDDMM kernel) ---
  std::vector<int> row_indices(nnz);
  for (int i = 0; i < M; i++)
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
      row_indices[p] = i;

  // === CPU Reference ===
  // Step 1: SDDMM — compute edge weights
  std::vector<float> vals_ref;
  sddmm_cpu(M, D, row_ptr, col_idx, E, vals_ref);

  // Step 2: SpMM — C = A_weighted * E
  std::vector<float> C_ref;
  spmm_cpu(M, M, D, row_ptr, col_idx, vals_ref, E, C_ref);

  // === GPU Setup ===
  int *d_row_ptr, *d_col_idx, *d_row_indices;
  float *d_vals, *d_E, *d_C;
  cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
  cudaMalloc(&d_col_idx, nnz * sizeof(int));
  cudaMalloc(&d_row_indices, nnz * sizeof(int));
  cudaMalloc(&d_vals, nnz * sizeof(float));
  cudaMalloc(&d_E, (size_t)M * D * sizeof(float));
  cudaMalloc(&d_C, (size_t)M * D * sizeof(float));

  cudaMemcpy(d_row_ptr, row_ptr.data(), (M + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_indices, row_indices.data(), nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_E, E.data(), (size_t)M * D * sizeof(float),
             cudaMemcpyHostToDevice);

  // === Step 1: SDDMM on GPU ===
  int sddmm_block = 256;
  int sddmm_grid = (nnz + sddmm_block - 1) / sddmm_block;

  float sddmm_ms = time_kernel_ms(
      [&]() {
        sddmm_csr_baseline_kernel<<<sddmm_grid, sddmm_block>>>(
            nnz, D, d_row_indices, d_col_idx, d_E, d_vals);
      },
      5, 50);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Validate SDDMM
  std::vector<float> vals_gpu(nnz);
  cudaMemcpy(vals_gpu.data(), d_vals, nnz * sizeof(float),
             cudaMemcpyDeviceToHost);
  float sddmm_err = max_abs_err(vals_ref, vals_gpu);
  std::cout << "SDDMM max error = " << sddmm_err << "\n";
  if (sddmm_err < 1e-5)
    std::cout << "SDDMM PASSED\n";
  else
    std::cout << "SDDMM FAILED\n";

  // === Step 2: SpMM on GPU (uses SDDMM output d_vals) ===
  int spmm_block = 256;
  int spmm_grid = (M + spmm_block - 1) / spmm_block;

  float spmm_ms = time_kernel_ms(
      [&]() {
        spmm_csr_row_kernel<<<spmm_grid, spmm_block>>>(
            M, D, d_row_ptr, d_col_idx, d_vals, d_E, d_C);
      },
      5, 50);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Validate SpMM
  std::vector<float> C_gpu((size_t)M * D);
  cudaMemcpy(C_gpu.data(), d_C, (size_t)M * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  float spmm_err = max_abs_err(C_ref, C_gpu);
  std::cout << "SpMM  max error = " << spmm_err << "\n";
  if (spmm_err < 1e-4)
    std::cout << "SpMM  PASSED\n";
  else
    std::cout << "SpMM  FAILED\n";

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Baseline SDDMM avg time (ms): " << sddmm_ms << "\n";
  std::cout << "Baseline SpMM  avg time (ms): " << spmm_ms << "\n";

  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_row_indices);
  cudaFree(d_vals);
  cudaFree(d_E);
  cudaFree(d_C);
  return 0;
}
