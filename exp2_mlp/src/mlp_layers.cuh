#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

void check_cublas(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : cuBLAS error");
  }
}

struct LayerShape {
  int batch;
  int in_dim;
  int out_dim;
};

inline double layer_flops(const LayerShape &shape) {
  return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int> &layers, int batch,
                         double millis) {
  double total_flops = 0.0;
  for (size_t i = 0; i + 1 < layers.size(); ++i) {
    LayerShape shape{batch, layers[i], layers[i + 1]};
    total_flops += layer_flops(shape);
  }
  return total_flops / (millis * 1e6);
}

/**
 * shape gives B = batch, M = in_dim, N = out_dim
 *
 * bias: N
 * activations: B x N
 */
__global__ void bias_add_kernel(const float *__restrict__ bias,
                                float *__restrict__ activations,
                                LayerShape shape) {
  /* DONE(student): each thread should add the bias for its neuron across the
   * batch. */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < shape.batch * shape.out_dim) {
    activations[idx] += bias[idx % shape.out_dim];
  }
}

__global__ void relu_kernel(float *__restrict__ activations, size_t elements) {
  /* DONE(student): ReLU activation (set negative values to zero). */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < elements) {
    activations[idx] = fmaxf(0.0f, activations[idx]);
  }
}

__global__ void gelu_kernel(float *__restrict__ activations, size_t elements) {
  /* DONE(student): Approximate GELU, e.g., 0.5*x*(1+tanh(...)). */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < elements) {
    float x = activations[idx];

    float param = sqrtf(2 / M_PI) * (x + 0.044715 * x * x * x);
    float val = 0.5 * x * (1 + tanhf(param));

    activations[idx] = val;
  }
}

inline void launch_bias_add(const float *bias, float *activations,
                            const LayerShape &shape, cudaStream_t stream) {
  const int threads = 256;
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
  (void)elements; // silence unused warnings until kernel implemented
}

inline void launch_activation(const std::string &activation, float *activations,
                              const LayerShape &shape, cudaStream_t stream) {
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  if (activation == "relu") {
    relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
  } else if (activation == "gelu") {
    gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
  } else {
    // TODO(student): add more activations as desired
  }
}

__global__ void fused_bias_activation_kernel(const float *__restrict__ bias,
                                             float *__restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
  /* DONE(student): fuse bias add + activation.
     activation_type: 0=ReLU, 1=GELU, extend as needed. */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < shape.batch * shape.out_dim) {
    activations[idx] += bias[idx % shape.out_dim];

    if (activation_type == 0) {
      activations[idx] = fmaxf(0.0f, activations[idx]);
    } else {
      float x = activations[idx];

      float param = sqrtf(2 / M_PI) * (x + 0.044715 * x * x * x);
      float val = 0.5 * x * (1 + tanhf(param));

      activations[idx] = val;
    }
  }
}

inline void launch_fused_bias_activation(const float *bias,
                                         const std::string &activation,
                                         float *activations,
                                         const LayerShape &shape,
                                         cudaStream_t stream) {
  int activation_type = (activation == "gelu") ? 1 : 0;
  const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(
      bias, activations, shape, activation_type);
  (void)elements;
}

/**
 * shape gives B (batch_size), M (input dim), and N (output dim)
 *
 * All matrices are row-major
 *
 * input: B x M
 * weight: N x M
 * output: B x N
 */
inline void run_gemm_layer(const float *input, const float *weight,
                           float *output, const LayerShape &shape,
                           cublasHandle_t handle) {
  /* DONE(student): call cublasSgemm (or StridedBatched) with the correct
     transpose options. Remember cuBLAS assumes column-major by default;
     consider using CUBLAS_OP_T to match row-major data. */
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int B = shape.batch;
  const int M = shape.in_dim;
  const int N = shape.out_dim;
  check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, B, M, &alpha,
                           weight, M, input, M, &beta, output, N),
               "cublasSgemm");
}
