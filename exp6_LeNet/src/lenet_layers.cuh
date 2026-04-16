#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

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
    const int conv1_out_h = LenetShape::in_height - LenetShape::conv1_kernel + 1;  // stride=1, padding=0
    const int conv1_out_w = LenetShape::in_width - LenetShape::conv1_kernel + 1;
    const int pool1_out_h = conv1_out_h / LenetShape::pool_stride;
    const int pool1_out_w = conv1_out_w / LenetShape::pool_stride;

    const int conv2_in_h = pool1_out_h;
    const int conv2_in_w = pool1_out_w;
    const int conv2_out_h = conv2_in_h - LenetShape::conv2_kernel + 1;
    const int conv2_out_w = conv2_in_w - LenetShape::conv2_kernel + 1;
    const int pool2_out_h = conv2_out_h / LenetShape::pool_stride;
    const int pool2_out_w = conv2_out_w / LenetShape::pool_stride;

    const int flattened = LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;

    s.input_elements = static_cast<size_t>(batch) * LenetShape::in_channels * LenetShape::in_height * LenetShape::in_width;
    s.conv1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * conv1_out_h * conv1_out_w;
    s.pool1_out_elems = static_cast<size_t>(batch) * LenetShape::conv1_out_channels * pool1_out_h * pool1_out_w;
    s.conv2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * conv2_out_h * conv2_out_w;
    s.pool2_out_elems = static_cast<size_t>(batch) * LenetShape::conv2_out_channels * pool2_out_h * pool2_out_w;
    s.fc1_out_elems = static_cast<size_t>(batch) * LenetShape::fc1_out;
    s.fc2_out_elems = static_cast<size_t>(batch) * LenetShape::fc2_out;
    s.output_elements = static_cast<size_t>(batch) * LenetShape::fc3_out;

    s.weight_offsets = std::vector<size_t>(5, 0);
    s.bias_offsets = std::vector<size_t>(5, 0);
    size_t cursor_w = 0;
    size_t cursor_b = 0;
    const auto push = [&](size_t elements, std::vector<size_t>& offsets, size_t& cursor) {
        offsets.push_back(cursor);
        cursor += elements;
    };
    s.weight_offsets[0] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv1_out_channels) * LenetShape::in_channels * LenetShape::conv1_kernel * LenetShape::conv1_kernel;
    s.weight_offsets[1] = cursor_w;
    cursor_w += static_cast<size_t>(LenetShape::conv2_out_channels) * LenetShape::conv1_out_channels * LenetShape::conv2_kernel * LenetShape::conv2_kernel;
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

inline double lenet_gflops(const LenetShape& shape, double millis) {
    const double conv1_flops = static_cast<double>(shape.batch) * LenetShape::conv1_out_channels * LenetShape::in_channels *
                               LenetShape::conv1_kernel * LenetShape::conv1_kernel * 2.0 * 28 * 28;
    const double conv2_flops = static_cast<double>(shape.batch) * LenetShape::conv2_out_channels * LenetShape::conv1_out_channels *
                               LenetShape::conv2_kernel * LenetShape::conv2_kernel * 2.0 * 10 * 10;
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

    cudnnFilterDescriptor_t conv1_filter = nullptr;
    cudnnFilterDescriptor_t conv2_filter = nullptr;

    cudnnConvolutionDescriptor_t conv1_desc = nullptr;
    cudnnConvolutionDescriptor_t conv2_desc = nullptr;

    cudnnActivationDescriptor_t activation = nullptr;
    cudnnPoolingDescriptor_t pool = nullptr;
};

inline void create_lenet_descriptors(const LenetShape& shape, LenetDescriptors& d) {
    /* TODO(student): cudnnCreate* all descriptors and configure tensor dimensions/strides. */
    (void)shape;
    (void)d;
}

inline void destroy_lenet_descriptors(LenetDescriptors& d) {
    /* TODO(student): destroy all descriptors created above. */
    (void)d;
}

inline cudnnConvolutionFwdAlgo_t parse_algo(const std::string& name) {
    if (name == "implicit_gemm") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    if (name == "implicit_precomp") return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (name == "fft") return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    // TODO(student): extend with more options.
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
}

inline size_t query_conv_workspace(cudnnHandle_t handle,
                                   const LenetShape& shape,
                                   const LenetDescriptors& descs,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   bool second_conv) {
    /* TODO(student): call cudnnGetConvolutionForwardWorkspaceSize for conv1/conv2. */
    (void)handle;
    (void)shape;
    (void)descs;
    (void)algo;
    (void)second_conv;
    return 0;
}

inline void run_lenet_conv(cudnnHandle_t handle,
                           const LenetShape& shape,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           const float* d_filter,
                           float* d_output,
                           void* d_workspace,
                           size_t workspace_bytes,
                           const std::string& algo_name,
                           bool second_conv) {
    /* TODO(student): select descriptors (conv1 vs conv2), pick algo, and call cudnnConvolutionForward.
       After conv, optionally launch cudnnBiasAdd + cudnnActivationForward (tanh/ReLU). */
    (void)handle;
    (void)shape;
    (void)descs;
    (void)d_input;
    (void)d_filter;
    (void)d_output;
    (void)d_workspace;
    (void)workspace_bytes;
    (void)algo_name;
    (void)second_conv;
}

inline void run_lenet_pool(cudnnHandle_t handle,
                           const LenetDescriptors& descs,
                           const float* d_input,
                           float* d_output,
                           bool second_pool) {
    /* TODO(student): use cudnnPoolingForward for pool1 or pool2. */
    (void)handle;
    (void)descs;
    (void)d_input;
    (void)d_output;
    (void)second_pool;
}

inline void run_fc_layer(cublasHandle_t handle,
                         const LenetShape& shape,
                         int layer_idx,
                         const float* d_input,
                         const float* d_weight,
                         const float* d_bias,
                         float* d_output,
                         cudaStream_t stream) {
    /* TODO(student): implement row-major GEMM via cublasSgemm / cublasGemmEx + bias add + activation.
       layer_idx ∈ {0:fc1,1:fc2,2:fc3}; use shape metadata to determine dims. */
    (void)handle;
    (void)shape;
    (void)layer_idx;
    (void)d_input;
    (void)d_weight;
    (void)d_bias;
    (void)d_output;
    (void)stream;
}

inline void reshape_conv_to_fc(const LenetShape& shape, const float* d_input, float* d_output, cudaStream_t stream) {
    /* TODO(student): implement or call cudaMemcpy to treat tensor as flattened (B, flattened).
       A simple kernel can copy/reshape pool2 output into row-major batches for GEMM. */
    (void)shape;
    (void)d_input;
    (void)d_output;
    (void)stream;
}
