#pragma once

#include <cuda_runtime.h>

struct Conv2dShape {
    int height;
    int width;
    int channels;
    int filters;
    int kernel;
    int stride;
    int padding;
    int out_height;
    int out_width;
};

inline Conv2dShape make_shape(int height,
                              int width,
                              int channels,
                              int filters,
                              int kernel,
                              int stride,
                              int padding) {
    Conv2dShape shape{height,
                      width,
                      channels,
                      filters,
                      kernel,
                      stride,
                      padding,
                      0,
                      0};
    shape.out_height = (height + 2 * padding - kernel) / stride + 1;
    shape.out_width = (width + 2 * padding - kernel) / stride + 1;
    return shape;
}

inline __host__ __device__ int input_index(const Conv2dShape& shape, int c, int h, int w) {
    return (c * shape.height + h) * shape.width + w;
}

inline __host__ __device__ int weight_index(const Conv2dShape& shape, int oc, int ic, int kh, int kw) {
    return ((oc * shape.channels + ic) * shape.kernel + kh) * shape.kernel + kw;
}

inline __host__ __device__ int output_index(const Conv2dShape& shape, int oc, int oh, int ow) {
    return (oc * shape.out_height + oh) * shape.out_width + ow;
}

constexpr int BLOCK_SIZE = 16;

inline dim3 make_conv_grid(const Conv2dShape& shape) {
    return dim3((shape.out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (shape.out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                shape.filters);
}

__global__ void conv2d_naive_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;
    if (ow >= shape.out_width || oh >= shape.out_height || oc >= shape.filters) {
        return;
    }

    float acc = 0.0f;
    /* TODO(student): loop over channels/ksize and accumulate into acc. Remember padding offsets:
       ih = oh * stride - padding + kh;
       iw = ow * stride - padding + kw;
       Skip taps that fall outside the padded image. */
    output[output_index(shape, oc, oh, ow)] = acc;
    (void)input;
    (void)weight;
}

__global__ void conv2d_tiled_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output,
                                    Conv2dShape shape) {
    extern __shared__ float tile[];
    float* tile_input = tile;                              // BLOCK_SIZE^2 * Cin chunk (student-defined)
    float* tile_weight = tile + BLOCK_SIZE * BLOCK_SIZE;   // placeholder layout suggestion
    /* TODO(student): stage input patches / filter tiles into shared memory and reuse them
       across threads within the block. Carefully handle halos caused by padding. */
    (void)tile_input;
    (void)tile_weight;
    (void)output;
    (void)shape;
    (void)input;
    (void)weight;
}

inline void launch_naive_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    conv2d_naive_kernel<<<grid, block, 0, stream>>>(d_input, d_weight, d_output, shape);
    /* TODO(student): check cudaGetLastError() and optionally cudaDeviceSynchronize() when debugging. */
}

inline void launch_tiled_conv2d(const float* d_input,
                                const float* d_weight,
                                float* d_output,
                                const Conv2dShape& shape,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_conv_grid(shape);
    size_t shared_bytes = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    conv2d_tiled_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_weight, d_output, shape);
    /* TODO(student): choose a better shared-memory layout/size expression once kernels are implemented. */
    (void)d_input;
    (void)d_weight;
    (void)d_output;
}

inline double conv_gflops(const Conv2dShape& shape, double millis) {
    const double flops = static_cast<double>(shape.filters) * shape.out_height * shape.out_width *
                         shape.channels * shape.kernel * shape.kernel * 2.0;
    return flops / (millis * 1e6);
}
