#pragma once

#include "cudnn_graph.h"
#include "cudnn_ops.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdexcept>

#include <array>
#include <string>
#include <vector>

struct LenetShape {
  int batch;
  // Assume MNIST-style 1x32x32 input.
  static constexpr int in_channels = 1;
  static constexpr int in_height = 32;
  static constexpr int in_width = 32;

  static constexpr int conv1_out_channels = 6;
  static constexpr int conv1_kernel = 5;
  static constexpr int conv2_out_channels = 16;
  static constexpr int conv2_kernel = 5;

  static constexpr int pool_stride = 2;

  static constexpr int fc1_out = 120;
  static constexpr int fc2_out = 84;
  static constexpr int fc3_out = 10;

  size_t input_elements;
  size_t conv1_out_elems;
  size_t pool1_out_elems;
  size_t conv2_out_elems;
  size_t pool2_out_elems;
  size_t fc1_out_elems;
  size_t fc2_out_elems;
  size_t output_elements;

  size_t total_weight_elements;
  size_t total_bias_elements;
  std::vector<size_t> weight_offsets;
  std::vector<size_t> bias_offsets;
};

inline LenetShape make_lenet_shape(int batch) {
  LenetShape s{};
  s.batch = batch;
  const int conv1_out_h = LenetShape::in_height - LenetShape::conv1_kernel +
                          1; // stride=1, padding=0
  const int conv1_out_w = LenetShape::in_width - LenetShape::conv1_kernel + 1;
  const int pool1_out_h = conv1_out_h / LenetShape::pool_stride;
  const int pool1_out_w = conv1_out_w / LenetShape::pool_stride;

  const int conv2_in_h = pool1_out_h;
  const int conv2_in_w = pool1_out_w;
  const int conv2_out_h = conv2_in_h - LenetShape::conv2_kernel + 1;
  const int conv2_out_w = conv2_in_w - LenetShape::conv2_kernel + 1;
  const int pool2_out_h = conv2_out_h / LenetShape::pool_stride;
  const int pool2_out_w = conv2_out_w / LenetShape::pool_stride;

  const int flattened =
      LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;

  s.input_elements = static_cast<size_t>(batch) * LenetShape::in_channels *
                     LenetShape::in_height * LenetShape::in_width;
  s.conv1_out_elems = static_cast<size_t>(batch) *
                      LenetShape::conv1_out_channels * conv1_out_h *
                      conv1_out_w;
  s.pool1_out_elems = static_cast<size_t>(batch) *
                      LenetShape::conv1_out_channels * pool1_out_h *
                      pool1_out_w;
  s.conv2_out_elems = static_cast<size_t>(batch) *
                      LenetShape::conv2_out_channels * conv2_out_h *
                      conv2_out_w;
  s.pool2_out_elems = static_cast<size_t>(batch) *
                      LenetShape::conv2_out_channels * pool2_out_h *
                      pool2_out_w;
  s.fc1_out_elems = static_cast<size_t>(batch) * LenetShape::fc1_out;
  s.fc2_out_elems = static_cast<size_t>(batch) * LenetShape::fc2_out;
  s.output_elements = static_cast<size_t>(batch) * LenetShape::fc3_out;

  s.weight_offsets = std::vector<size_t>(5, 0);
  s.bias_offsets = std::vector<size_t>(5, 0);
  size_t cursor_w = 0;
  size_t cursor_b = 0;
  const auto push = [&](size_t elements, std::vector<size_t> &offsets,
                        size_t &cursor) {
    offsets.push_back(cursor);
    cursor += elements;
  };
  s.weight_offsets[0] = cursor_w;
  cursor_w += static_cast<size_t>(LenetShape::conv1_out_channels) *
              LenetShape::in_channels * LenetShape::conv1_kernel *
              LenetShape::conv1_kernel;
  s.weight_offsets[1] = cursor_w;
  cursor_w += static_cast<size_t>(LenetShape::conv2_out_channels) *
              LenetShape::conv1_out_channels * LenetShape::conv2_kernel *
              LenetShape::conv2_kernel;
  s.weight_offsets[2] = cursor_w;
  cursor_w += static_cast<size_t>(LenetShape::fc1_out) * flattened;
  s.weight_offsets[3] = cursor_w;
  cursor_w += static_cast<size_t>(LenetShape::fc2_out) * LenetShape::fc1_out;
  s.weight_offsets[4] = cursor_w;
  cursor_w += static_cast<size_t>(LenetShape::fc3_out) * LenetShape::fc2_out;

  s.bias_offsets[0] = cursor_b;
  cursor_b += LenetShape::conv1_out_channels;
  s.bias_offsets[1] = cursor_b;
  cursor_b += LenetShape::conv2_out_channels;
  s.bias_offsets[2] = cursor_b;
  cursor_b += LenetShape::fc1_out;
  s.bias_offsets[3] = cursor_b;
  cursor_b += LenetShape::fc2_out;
  s.bias_offsets[4] = cursor_b;
  cursor_b += LenetShape::fc3_out;

  s.total_weight_elements = cursor_w;
  s.total_bias_elements = cursor_b;
  return s;
}

inline double lenet_gflops(const LenetShape &shape, double millis) {
  const double conv1_flops =
      static_cast<double>(shape.batch) * LenetShape::conv1_out_channels *
      LenetShape::in_channels * LenetShape::conv1_kernel *
      LenetShape::conv1_kernel * 2.0 * 28 * 28;
  const double conv2_flops =
      static_cast<double>(shape.batch) * LenetShape::conv2_out_channels *
      LenetShape::conv1_out_channels * LenetShape::conv2_kernel *
      LenetShape::conv2_kernel * 2.0 * 10 * 10;
  const double fc1_in = LenetShape::conv2_out_channels * 5 * 5;
  const double fc_flops = static_cast<double>(shape.batch) *
                          (2.0 * fc1_in * LenetShape::fc1_out +
                           2.0 * LenetShape::fc1_out * LenetShape::fc2_out +
                           2.0 * LenetShape::fc2_out * LenetShape::fc3_out);
  const double total = conv1_flops + conv2_flops + fc_flops;
  return total / (millis * 1e6);
}

struct LenetDescriptors {
  cudnnTensorDescriptor_t input_desc = nullptr;
  cudnnTensorDescriptor_t conv1_out_desc = nullptr;
  cudnnTensorDescriptor_t pool1_out_desc = nullptr;
  cudnnTensorDescriptor_t conv2_out_desc = nullptr;
  cudnnTensorDescriptor_t pool2_out_desc = nullptr;
  cudnnTensorDescriptor_t fc1_desc = nullptr;
  cudnnTensorDescriptor_t fc2_desc = nullptr;
  cudnnTensorDescriptor_t fc3_desc = nullptr;

  // Bias descriptors (1, C, 1, 1)
  cudnnTensorDescriptor_t conv1_bias_desc = nullptr;
  cudnnTensorDescriptor_t conv2_bias_desc = nullptr;

  cudnnFilterDescriptor_t conv1_filter = nullptr;
  cudnnFilterDescriptor_t conv2_filter = nullptr;

  cudnnConvolutionDescriptor_t conv1_desc = nullptr;
  cudnnConvolutionDescriptor_t conv2_desc = nullptr;

  cudnnActivationDescriptor_t activation = nullptr;
  cudnnPoolingDescriptor_t pool = nullptr;
};

inline void create_lenet_descriptors(const LenetShape &shape,
                                     LenetDescriptors &d) {
  // --- Tensor descriptors (NCHW, float) ---
  // Input: (N, 1, 32, 32)
  cudnnCreateTensorDescriptor(&d.input_desc);
  cudnnSetTensor4dDescriptor(d.input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             shape.batch, shape.in_channels, shape.in_height,
                             shape.in_width);

  // Conv1 output / activation input: (N, 6, 28, 28)
  cudnnCreateTensorDescriptor(&d.conv1_out_desc);
  cudnnSetTensor4dDescriptor(d.conv1_out_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, shape.batch,
                             shape.conv1_out_channels, 28, 28);

  // Pool1 output: (N, 6, 14, 14)
  cudnnCreateTensorDescriptor(&d.pool1_out_desc);
  cudnnSetTensor4dDescriptor(d.pool1_out_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, shape.batch,
                             shape.conv1_out_channels, 14, 14);

  // Conv2 output / activation input: (N, 16, 10, 10)
  cudnnCreateTensorDescriptor(&d.conv2_out_desc);
  cudnnSetTensor4dDescriptor(d.conv2_out_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, shape.batch,
                             shape.conv2_out_channels, 10, 10);

  // Pool2 output: (N, 16, 5, 5)
  cudnnCreateTensorDescriptor(&d.pool2_out_desc);
  cudnnSetTensor4dDescriptor(d.pool2_out_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, shape.batch,
                             shape.conv2_out_channels, 5, 5);

  // FC descriptors: treat as (N, C, 1, 1) so cuDNN bias-add works uniformly
  cudnnCreateTensorDescriptor(&d.fc1_desc);
  cudnnSetTensor4dDescriptor(d.fc1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             shape.batch, shape.fc1_out, 1, 1);

  cudnnCreateTensorDescriptor(&d.fc2_desc);
  cudnnSetTensor4dDescriptor(d.fc2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             shape.batch, shape.fc2_out, 1, 1);

  cudnnCreateTensorDescriptor(&d.fc3_desc);
  cudnnSetTensor4dDescriptor(d.fc3_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             shape.batch, shape.fc3_out, 1, 1);

  // --- Filter descriptors (out_channels, in_channels, kH, kW) ---
  cudnnCreateFilterDescriptor(&d.conv1_filter);
  cudnnSetFilter4dDescriptor(d.conv1_filter, CUDNN_DATA_FLOAT,
                             CUDNN_TENSOR_NCHW, shape.conv1_out_channels,
                             shape.in_channels, shape.conv1_kernel,
                             shape.conv1_kernel);

  cudnnCreateFilterDescriptor(&d.conv2_filter);
  cudnnSetFilter4dDescriptor(d.conv2_filter, CUDNN_DATA_FLOAT,
                             CUDNN_TENSOR_NCHW, shape.conv2_out_channels,
                             shape.conv1_out_channels, shape.conv2_kernel,
                             shape.conv2_kernel);

  // --- Convolution descriptors (padding=0, stride=1, dilation=1) ---
  cudnnCreateConvolutionDescriptor(&d.conv1_desc);
  cudnnSetConvolution2dDescriptor(d.conv1_desc, 0, 0, // pad_h, pad_w
                                  1, 1,               // stride_h, stride_w
                                  1, 1,               // dilation_h, dilation_w
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnCreateConvolutionDescriptor(&d.conv2_desc);
  cudnnSetConvolution2dDescriptor(d.conv2_desc, 0, 0, 1, 1, 1, 1,
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  // --- Activation descriptor (tanh, shared across all activation calls) ---
  cudnnCreateActivationDescriptor(&d.activation);
  cudnnSetActivationDescriptor(d.activation, CUDNN_ACTIVATION_TANH,
                               CUDNN_PROPAGATE_NAN, 0.0);

  // --- Pooling descriptor (2x2 avg, stride 2, shared for pool1 and pool2) ---
  cudnnCreatePoolingDescriptor(&d.pool);
  cudnnSetPooling2dDescriptor(d.pool,
                              CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                              CUDNN_PROPAGATE_NAN, 2, 2, // window_h, window_w
                              0, 0,                      // pad_h, pad_w
                              2, 2);                     // stride_h, stride_w

  // --- Bias descriptors for cudnnAddTensor: (1, C, 1, 1) ---
  cudnnCreateTensorDescriptor(&d.conv1_bias_desc);
  cudnnSetTensor4dDescriptor(d.conv1_bias_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, 1, shape.conv1_out_channels, 1,
                             1);

  cudnnCreateTensorDescriptor(&d.conv2_bias_desc);
  cudnnSetTensor4dDescriptor(d.conv2_bias_desc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, 1, shape.conv2_out_channels, 1,
                             1);
}

inline void destroy_lenet_descriptors(LenetDescriptors &d) {
  cudnnDestroyTensorDescriptor(d.input_desc);
  cudnnDestroyTensorDescriptor(d.conv1_out_desc);
  cudnnDestroyTensorDescriptor(d.pool1_out_desc);
  cudnnDestroyTensorDescriptor(d.conv2_out_desc);
  cudnnDestroyTensorDescriptor(d.pool2_out_desc);
  cudnnDestroyTensorDescriptor(d.fc1_desc);
  cudnnDestroyTensorDescriptor(d.fc2_desc);
  cudnnDestroyTensorDescriptor(d.fc3_desc);
  cudnnDestroyFilterDescriptor(d.conv1_filter);
  cudnnDestroyFilterDescriptor(d.conv2_filter);
  cudnnDestroyConvolutionDescriptor(d.conv1_desc);
  cudnnDestroyConvolutionDescriptor(d.conv2_desc);
  cudnnDestroyActivationDescriptor(d.activation);
  cudnnDestroyPoolingDescriptor(d.pool);
  cudnnDestroyTensorDescriptor(d.conv1_bias_desc);
  cudnnDestroyTensorDescriptor(d.conv2_bias_desc);
}

inline cudnnConvolutionFwdAlgo_t parse_algo(const std::string &name) {
  if (name == "implicit_gemm")
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  if (name == "implicit_precomp")
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  if (name == "fft")
    return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  // DONE(student): extend with more options.
  if (name == "gemm")
    return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  if (name == "direct")
    return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  if (name == "fft_tiling")
    return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  if (name == "winograd")
    return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  if (name == "winograd_nonfused")
    return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  throw std::invalid_argument("Unknown algo: " + name);
}

inline size_t query_conv_workspace(cudnnHandle_t handle,
                                   const LenetShape &shape,
                                   const LenetDescriptors &descs,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   bool second_conv) {
  /* DONE(student): call cudnnGetConvolutionForwardWorkspaceSize for
   * conv1/conv2. */

  (void)shape;
  size_t workspace_bytes = 0;
  if (!second_conv) {
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, descs.input_desc, descs.conv1_filter, descs.conv1_desc,
        descs.conv1_out_desc, algo, &workspace_bytes);
  } else {
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, descs.pool1_out_desc, descs.conv2_filter, descs.conv2_desc,
        descs.conv2_out_desc, algo, &workspace_bytes);
  }
  return workspace_bytes;
}

inline void run_lenet_conv(cudnnHandle_t handle, const LenetShape &shape,
                           const LenetDescriptors &descs, const float *d_input,
                           const float *d_filter, const float *d_bias,
                           float *d_output, void *d_workspace,
                           size_t workspace_bytes, const std::string &algo_name,
                           bool second_conv) {
  /* DONE(student): select descriptors (conv1 vs conv2), pick algo, and call
     cudnnConvolutionForward. After conv, optionally launch cudnnBiasAdd +
     cudnnActivationForward (tanh/ReLU). */

  (void)shape;
  const float alpha = 1.0f, beta = 0.0f, bias_alpha = 1.0f;
  cudnnConvolutionFwdAlgo_t algo = parse_algo(algo_name);

  cudnnTensorDescriptor_t x_desc =
      second_conv ? descs.pool1_out_desc : descs.input_desc;
  cudnnFilterDescriptor_t w_desc =
      second_conv ? descs.conv2_filter : descs.conv1_filter;
  cudnnConvolutionDescriptor_t c_desc =
      second_conv ? descs.conv2_desc : descs.conv1_desc;
  cudnnTensorDescriptor_t y_desc =
      second_conv ? descs.conv2_out_desc : descs.conv1_out_desc;
  cudnnTensorDescriptor_t b_desc =
      second_conv ? descs.conv2_bias_desc : descs.conv1_bias_desc;

  // Convolution
  cudnnConvolutionForward(handle, &alpha, x_desc, d_input, w_desc, d_filter,
                          c_desc, algo, d_workspace, workspace_bytes, &beta,
                          y_desc, d_output);

  // Bias add: broadcast (1, C, 1, 1) bias over (N, C, H, W) output
  cudnnAddTensor(handle, &bias_alpha, b_desc, d_bias, &alpha, y_desc, d_output);

  // Tanh activation in-place
  cudnnActivationForward(handle, descs.activation, &alpha, y_desc, d_output,
                         &beta, y_desc, d_output);
}

inline void run_lenet_conv_fused(cudnnHandle_t handle, const LenetShape &shape,
                                 const LenetDescriptors &descs,
                                 const float *d_input, const float *d_filter,
                                 const float *d_bias, float *d_output,
                                 void *d_workspace, size_t workspace_bytes,
                                 const std::string &algo_name,
                                 bool second_conv) {
  (void)shape;
  const float alpha1 = 1.0f, alpha2 = 0.0f;
  cudnnConvolutionFwdAlgo_t algo = parse_algo(algo_name);

  cudnnTensorDescriptor_t x_desc =
      second_conv ? descs.pool1_out_desc : descs.input_desc;
  cudnnFilterDescriptor_t w_desc =
      second_conv ? descs.conv2_filter : descs.conv1_filter;
  cudnnConvolutionDescriptor_t c_desc =
      second_conv ? descs.conv2_desc : descs.conv1_desc;
  cudnnTensorDescriptor_t y_desc =
      second_conv ? descs.conv2_out_desc : descs.conv1_out_desc;
  cudnnTensorDescriptor_t b_desc =
      second_conv ? descs.conv2_bias_desc : descs.conv1_bias_desc;

  // Single-kernel fused: y = tanh(alpha1 * conv(x, w) + alpha2 * z + bias)
  // alpha2=0 means z is ignored; pass z = d_output as a valid but unused
  // pointer.
  cudnnStatus_t status = cudnnConvolutionBiasActivationForward(
      handle, &alpha1, x_desc, d_input, w_desc, d_filter, c_desc, algo,
      d_workspace, workspace_bytes, &alpha2, y_desc,
      d_output, // zDesc, z (ignored since alpha2=0)
      b_desc, d_bias,
      descs.activation, // existing tanh descriptor — reused
      y_desc, d_output);

  if (status == CUDNN_STATUS_NOT_SUPPORTED) {
    // Fall back: fused conv+bias with identity, then separate tanh
    throw std::invalid_argument("Does not support activation used");
  }
}

inline void run_lenet_pool(cudnnHandle_t handle, const LenetDescriptors &descs,
                           const float *d_input, float *d_output,
                           bool second_pool) {
  /* DONE(student): use cudnnPoolingForward for pool1 or pool2. */
  const float alpha = 1.0f, beta = 0.0f;
  cudnnTensorDescriptor_t x_desc =
      second_pool ? descs.conv2_out_desc : descs.conv1_out_desc;
  cudnnTensorDescriptor_t y_desc =
      second_pool ? descs.pool2_out_desc : descs.pool1_out_desc;

  cudnnPoolingForward(handle, descs.pool, &alpha, x_desc, d_input, &beta,
                      y_desc, d_output);
}

__device__ inline float apply_activation(float x, bool use_tanh) {
  return use_tanh ? tanhf(x) : x;
}

__global__ void fc_bias_act_kernel(float *output, const float *bias, int N,
                                   int out_f, bool use_tanh) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; // feature index
  int n = blockIdx.y;                            // batch index
  if (j >= out_f || n >= N)
    return;
  float val = output[n * out_f + j] + bias[j];
  output[n * out_f + j] = apply_activation(val, use_tanh);
}

inline void run_fc_layer(cublasHandle_t handle, const LenetShape &shape,
                         int layer_idx, const float *d_input,
                         const float *d_weight, const float *d_bias,
                         float *d_output, cudaStream_t stream) {
  /* DONE(student): implement row-major GEMM via cublasSgemm / cublasGemmEx +
     bias add + activation. layer_idx ∈ {0:fc1,1:fc2,2:fc3}; use shape metadata
     to determine dims. */

  const int in_dims[3] = {shape.conv2_out_channels * 5 * 5, shape.fc1_out,
                          shape.fc2_out};
  const int out_dims[3] = {shape.fc1_out, shape.fc2_out, shape.fc3_out};
  const int in_f = in_dims[layer_idx];
  const int out_f = out_dims[layer_idx];
  const int N = shape.batch;

  cublasSetStream(handle, stream);
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_f, N, in_f, &alpha,
              d_weight, in_f, d_input, in_f, &beta, d_output, out_f);

  // Bias add + activation (tanh for fc1/fc2, identity for fc3)
  dim3 block(256);
  dim3 grid((out_f + 255) / 256, N);
  fc_bias_act_kernel<<<grid, block, 0, stream>>>(d_output, d_bias, N, out_f,
                                                 layer_idx < 2);
}

inline void reshape_conv_to_fc(const LenetShape &shape, const float *d_input,
                               float *d_output, cudaStream_t stream) {
  /* DONE(student): implement or call cudaMemcpy to treat tensor as flattened
     (B, flattened). A simple kernel can copy/reshape pool2 output into
     row-major batches for GEMM. */

  // pool2 output (N, 16, 5, 5) is contiguous and identical in memory to (N,
  // 400). If input and output are the same buffer this is a no-op; otherwise
  // copy.
  if (d_input != d_output) {
    cudaMemcpyAsync(d_output, d_input, shape.pool2_out_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
  }
}
