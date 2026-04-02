#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

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

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " : " +
                             cudaGetErrorString(err));
  }
}

inline Conv2dShape make_shape(int height, int width, int channels, int filters,
                              int kernel, int stride, int padding) {
  Conv2dShape shape{height, width,   channels, filters, kernel,
                    stride, padding, 0,        0};
  shape.out_height = (height + 2 * padding - kernel) / stride + 1;
  shape.out_width = (width + 2 * padding - kernel) / stride + 1;
  return shape;
}

inline __host__ __device__ int input_index(const Conv2dShape &shape, int c,
                                           int h, int w) {
  return (c * shape.height + h) * shape.width + w;
}

inline __host__ __device__ int weight_index(const Conv2dShape &shape, int oc,
                                            int ic, int kh, int kw) {
  return ((oc * shape.channels + ic) * shape.kernel + kh) * shape.kernel + kw;
}

inline __host__ __device__ int output_index(const Conv2dShape &shape, int oc,
                                            int oh, int ow) {
  return (oc * shape.out_height + oh) * shape.out_width + ow;
}

constexpr int BLOCK_SIZE = 16;

inline dim3 make_conv_grid(const Conv2dShape &shape) {
  return dim3((shape.out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (shape.out_height + BLOCK_SIZE - 1) / BLOCK_SIZE, shape.filters);
}

__global__ void conv2d_naive_kernel(const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    float *__restrict__ output,
                                    Conv2dShape shape) {
  const int ow = blockIdx.x * blockDim.x + threadIdx.x;
  const int oh = blockIdx.y * blockDim.y + threadIdx.y;
  const int oc = blockIdx.z;
  if (ow >= shape.out_width || oh >= shape.out_height || oc >= shape.filters) {
    return;
  }

  float acc = 0.0f;
  /* DONE(student): loop over channels/ksize and accumulate into acc.
  Remember
     padding offsets: ih = oh * stride - padding + kh; iw = ow * stride -
     padding + kw; Skip taps that fall outside the padded image. */

  for (int ic = 0; ic < shape.channels; ic++) {
    for (int kh = 0; kh < shape.kernel; kh++) {
      for (int kw = 0; kw < shape.kernel; kw++) {
        const int ih = oh * shape.stride - shape.padding + kh;
        const int iw = ow * shape.stride - shape.padding + kw;

        float input_val = 0.0f;
        if (ih >= 0 && ih < shape.height && iw >= 0 && iw < shape.width) {
          input_val = input[input_index(shape, ic, ih, iw)];
        }

        acc += input_val * weight[weight_index(shape, oc, ic, kh, kw)];
      }
    }
  }

  output[output_index(shape, oc, oh, ow)] = acc;
}

__global__ void conv2d_tiled_kernel(const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    float *__restrict__ output,
                                    Conv2dShape shape) {
  // Each block computes a BLOCK_SIZE x BLOCK_SIZE output tile for one
  // output channel oc = blockIdx.z.
  //
  // Shared memory layout:
  //   [ input tile | weight tile ]
  //
  // For an output tile of size BLOCK_SIZE x BLOCK_SIZE, the required input
  // region must include the halo induced by stride and kernel size:
  //
  //   tile_input_dim = BLOCK_SIZE * stride + kernel - stride
  //
  // This is equivalent to:
  //
  //   (BLOCK_SIZE - 1) * stride + kernel
  //
  // which is exactly the span of input touched by the output tile.
  const int tile_input_width =
      BLOCK_SIZE * shape.stride + shape.kernel - shape.stride;
  const int tile_input_height =
      BLOCK_SIZE * shape.stride + shape.kernel - shape.stride;

  extern __shared__ float tile[];
  float *tile_input = tile;
  float *tile_weight = tile + tile_input_height * tile_input_width;

  // Thread coordinates within the block.
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Output coordinates handled by this thread.
  const int ow = blockIdx.x * BLOCK_SIZE + tx;
  const int oh = blockIdx.y * BLOCK_SIZE + ty;
  const int oc = blockIdx.z;

  // Whether this thread corresponds to a valid output location.
  const bool valid_output =
      (ow < shape.out_width) && (oh < shape.out_height) && (oc < shape.filters);

  float acc = 0.0f;

  // Process one input channel at a time. For each channel:
  //   1. stage the needed input tile into shared memory
  //   2. stage the corresponding kernel slice into shared memory
  //   3. accumulate this channel's contribution
  for (int ic = 0; ic < shape.channels; ++ic) {
    // ------------------------------------------------------------
    // Load the input tile for channel ic into shared memory.
    //
    // Threads cooperatively fill the entire tile using 2D strided loops.
    // Out-of-bounds input accesses are treated as zero due to padding.
    // ------------------------------------------------------------
    for (int th = ty; th < tile_input_height; th += blockDim.y) {
      for (int tw = tx; tw < tile_input_width; tw += blockDim.x) {
        const int ih =
            blockIdx.y * BLOCK_SIZE * shape.stride - shape.padding + th;
        const int iw =
            blockIdx.x * BLOCK_SIZE * shape.stride - shape.padding + tw;

        if (ih >= 0 && ih < shape.height && iw >= 0 && iw < shape.width) {
          tile_input[th * tile_input_width + tw] =
              input[input_index(shape, ic, ih, iw)];
        } else {
          tile_input[th * tile_input_width + tw] = 0.0f;
        }
      }
    }

    // ------------------------------------------------------------
    // Load the kernel slice for (oc, ic) into shared memory.
    //
    // Unlike the slower implementation, this is loaded cooperatively:
    // threads whose (ty, tx) coordinates fall inside the kernel footprint
    // each load one weight element.
    // ------------------------------------------------------------
    if (ty < shape.kernel && tx < shape.kernel) {
      tile_weight[ty * shape.kernel + tx] =
          weight[weight_index(shape, oc, ic, ty, tx)];
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Compute this thread's output value for channel ic.
    //
    // Relative to the shared input tile, the top-left corner of the receptive
    // field for output (oh, ow) is:
    //
    //   base_y = ty * stride
    //   base_x = tx * stride
    // ------------------------------------------------------------
    if (valid_output) {
      const int base_y = ty * shape.stride;
      const int base_x = tx * shape.stride;

      for (int kh = 0; kh < shape.kernel; ++kh) {
        for (int kw = 0; kw < shape.kernel; ++kw) {
          const float input_val =
              tile_input[(base_y + kh) * tile_input_width + (base_x + kw)];
          const float weight_val = tile_weight[kh * shape.kernel + kw];
          acc += input_val * weight_val;
        }
      }
    }

    __syncthreads();
  }

  if (valid_output) {
    output[output_index(shape, oc, oh, ow)] = acc;
  }
}

inline void launch_naive_conv2d(const float *d_input, const float *d_weight,
                                float *d_output, const Conv2dShape &shape,
                                cudaStream_t stream) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid = make_conv_grid(shape);

  int tile_in_width = BLOCK_SIZE * shape.stride + shape.kernel - shape.stride;
  int tile_in_height = BLOCK_SIZE + shape.stride + shape.kernel - shape.stride;
  size_t shared_bytes =
      (tile_in_width * tile_in_height + shape.kernel * shape.kernel) *
      sizeof(float);

  conv2d_naive_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_weight,
                                                             d_output, shape);
  /* DONE(student): check cudaGetLastError() and optionally
   * cudaDeviceSynchronize() when debugging. */

  cudaError_t err = cudaGetLastError();

  check_cuda(err, "Launch naive kernel");
}

inline void launch_tiled_conv2d(const float *d_input, const float *d_weight,
                                float *d_output, const Conv2dShape &shape,
                                cudaStream_t stream) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid = make_conv_grid(shape);
  size_t shared_bytes = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
  conv2d_tiled_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_weight,
                                                             d_output, shape);
  /* DONE(student): choose a better shared-memory layout/size expression once
   * kernels are implemented. */
  cudaError_t err = cudaGetLastError();

  check_cuda(err, "Launch tiled kernel");
}

inline double conv_gflops(const Conv2dShape &shape, double millis) {
  const double flops = static_cast<double>(shape.filters) * shape.out_height *
                       shape.out_width * shape.channels * shape.kernel *
                       shape.kernel * 2.0;
  return flops / (millis * 1e6);
}
