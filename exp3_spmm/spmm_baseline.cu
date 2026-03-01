
// spmm_baseline.cu — STUDENT SKELETON
#include <cassert>
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
===============================================================
 BASELINE KERNEL — one thread processes ONE ROW of A
 STUDENT TODO:
   - Fill missing loops
   - Compute C[row, j] += value * B[k, j]
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

  // DONE (student): Initialize output row C[row, :]
  int out_start = row * N;
  for (int i = 0; i < N; i++) {
    d_C[out_start + i] = 0;
  }

  // Find nonzero range
  int start, end;
  // DONE (student): load start, end
  start = d_row_ptr[row];
  end = d_row_ptr[row + 1];

  // Loop over nonzeros in this row
  // DONE (student):
  for (int i = start; i < end; i++) {
    // DONE (student): retrieve column index k
    int k = d_col_idx[i];
    // DONE (student): retrieve value v
    float_t v = d_vals[i];

    // DONE (student): loop over all columns j of output (0..N-1)
    //                 and accumulate:
    for (int j = 0; j < N; j++) {
      float_t b_val = d_B[k * N + j];

      d_C[out_start + j] += v * b_val;
    }
  }
}

/*
===============================================================
 MAIN PROGRAM
===============================================================
*/
int main(int argc, char **argv) {
  int M = 512, K = 512, N = 64;
  double density = 0.01;
  unsigned seed = 1234;

  std::vector<int> row_ptr, col_idx;
  std::vector<float> vals;
  generate_random_csr(M, K, density, row_ptr, col_idx, vals, seed);
  int nnz = row_ptr.back();
  std::cout << "nnz = " << nnz << "\n";

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

  // Launch incomplete student kernel
  int block = 256;
  int grid = (M + block - 1) / block;

  spmm_csr_row_kernel<<<grid, block>>>(M, N, d_row_ptr, d_col_idx, d_vals, d_B,
                                       d_C);
  cudaDeviceSynchronize();

  // Copy back
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
