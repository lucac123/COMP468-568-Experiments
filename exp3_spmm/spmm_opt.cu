
// spmm_opt.cu — STUDENT OPTIMIZATION SKELETON

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

extern void generate_random_csr(int M, int K, double density,
                                std::vector<int> &row_ptr,
                                std::vector<int> &col_idx,
                                std::vector<float> &vals, unsigned seed);

extern void spmm_cpu(int M, int K, int N, const std::vector<int> &row_ptr,
                     const std::vector<int> &col_idx,
                     const std::vector<float> &vals,
                     const std::vector<float> &B, std::vector<float> &C);

extern float max_abs_err(const std::vector<float> &A,
                         const std::vector<float> &B);

using float_t = float;

/*
=================================================================
 OPTIMIZED KERNEL (SKELETON)
 Warp processes ONE ROW, each thread handles j = lane, lane+32, ...
 STUDENT DONE:
    - Fetch row range
    - Loop over nonzeros
    - Load B[k,j] and accumulate
=================================================================
*/
__global__ void spmm_csr_warp_kernel(int M, int N,
                                     const int *__restrict__ d_row_ptr,
                                     const int *__restrict__ d_col_idx,
                                     const float_t *__restrict__ d_vals,
                                     const float_t *__restrict__ d_B,
                                     float_t *__restrict__ d_C) {
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp = global_tid / 32;
  int lane = threadIdx.x % 32;

  if (warp >= M)
    return;
  int row = warp;

  // DONE (student): get start = d_row_ptr[row], end = d_row_ptr[row+1]
  int start = d_row_ptr[row];
  int end = d_row_ptr[row + 1];

  // Loop over columns j assigned to this lane
  for (int j = lane; j < N; j += 32) {

    float_t sum = 0.0f;

    // DONE (student): loop over nonzeros in this row
    for (int i = start; i < end; i++) {
      int k = d_col_idx[i];
      float_t v = d_vals[i];
      float_t b_val = d_B[k * N + j];

      sum += v * b_val;
    }

    // DONE (student): write result to d_C
    d_C[row * N + j] = sum;
  }
}

/*
===========================================================
 MAIN DRIVER  (placeholder)
===========================================================
*/
int main() {
  std::cout << "This file contains student TODOs. Compile with spmm_ref.cpp to "
               "link reference functions if needed."
            << std::endl;
  // After students complete the kernel, they can use the code below to test it.
  int M = 512, K = 512, N = 64;
  double density = 0.01;
  unsigned seed = 1234;

  std::vector<int> row_ptr, col_idx;
  std::vector<float> vals;
  generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
  int nnz = row_ptr.back();
  std::cout << "Optimized SpMM: M=" << M << " K=" << K << " N=" << N
            << " nnz=" << nnz << "\n";

  // Create B
  std::vector<float> B((size_t)K * N);
  for (size_t i = 0; i < B.size(); i++)
    B[i] = float(rand()) / RAND_MAX;

  // CPU reference
  std::vector<float> C_ref;
  spmm_cpu(M, K, N, row_ptr, col_idx, vals, B, C_ref);

  // Copy to device
  int *d_row_ptr, *d_col_idx;
  float *d_vals, *d_B, *d_C;
  cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
  cudaMalloc(&d_col_idx, nnz * sizeof(int));
  cudaMalloc(&d_vals, nnz * sizeof(float));
  cudaMalloc(&d_B, (size_t)K * N * sizeof(float));
  cudaMalloc(&d_C, (size_t)M * N * sizeof(float));

  cudaMemcpy(d_row_ptr, row_ptr.data(), (M + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), (size_t)K * N * sizeof(float),
             cudaMemcpyHostToDevice);

  int block = 256; // 每个 Block 有 256 个线程 (即 8 个 Warps)
  long long total_threads_needed = (long long)M * 32;
  int grid = (total_threads_needed + block - 1) / block;
  std::cout << "Launching Kernel with Grid=" << grid << ", Block=" << block
            << "\n";

  spmm_csr_warp_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B,
                                        d_C);

  // Copy result back
  std::vector<float> C((size_t)M * N);
  cudaMemcpy(C.data(), d_C, (size_t)M * N * sizeof(float),
             cudaMemcpyDeviceToHost);
  // Compare (will be wrong until students complete TODOs)
  float err = max_abs_err(C_ref, C);
  std::cout << "Max error = " << err << "\n";

  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_vals);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
