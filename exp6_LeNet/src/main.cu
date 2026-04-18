#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lenet_layers.cuh"

struct Options {
  int batch = 32;
  std::string algo = "implicit_gemm"; // cuDNN conv algo hint
  std::string impl = "baseline";      // baseline | fused
  bool verify = true;
  std::string dump_path = ""; // optional binary file for logits
};

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if ((strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0) &&
        i + 1 < argc) {
      opt.batch = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
      opt.algo = argv[++i];
    } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
      opt.impl = argv[++i];
    } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      opt.dump_path = argv[++i];
    } else if (strcmp(argv[i], "--no-verify") == 0) {
      opt.verify = false;
    } else if (strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: ./dlenet --batch N --algo implicit_gemm --impl "
                   "baseline|fused \\\n  [--dump outputs.bin] [--no-verify]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
    }
  }
  if (opt.batch <= 0) {
    throw std::invalid_argument("Batch must be > 0");
  }
  return opt;
}

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " : " +
                             cudaGetErrorString(err));
  }
}

void check_cudnn(cudnnStatus_t status, const char *msg) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : " +
                             cudnnGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : cuBLAS error");
  }
}

void seed_tensor(std::vector<float> &vec, float scale) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = scale * std::sin(0.017f * static_cast<float>(i));
  }
}

void lenet_cpu_reference(const Options &opt, const LenetShape &shape,
                         const std::vector<float> &weights,
                         const std::vector<size_t> &weight_offsets,
                         const std::vector<float> &biases,
                         const std::vector<size_t> &bias_offsets,
                         const std::vector<float> &input,
                         std::vector<float> &output) {
  /* DONE(student): implement a simple CPU LeNet forward
     (conv/pool/activations/GEMM). Keep it single-threaded for simplicity or
     call into a reference framework. */
  (void)opt;
  const int N = shape.batch;
  const int IH = shape.in_height, IW = shape.in_width;
  const int K1 = shape.conv1_kernel, C1 = shape.conv1_out_channels;
  const int K2 = shape.conv2_kernel, C2 = shape.conv2_out_channels;
  const int PS = shape.pool_stride;

  const int c1h = IH - K1 + 1, c1w = IW - K1 + 1;   // 28x28
  const int p1h = c1h / PS, p1w = c1w / PS;         // 14x14
  const int c2h = p1h - K2 + 1, c2w = p1w - K2 + 1; // 10x10
  const int p2h = c2h / PS, p2w = c2w / PS;         // 5x5
  const int flat = C2 * p2h * p2w;                  // 400

  std::vector<float> conv1(N * C1 * c1h * c1w);
  std::vector<float> pool1(N * C1 * p1h * p1w);
  std::vector<float> conv2(N * C2 * c2h * c2w);
  std::vector<float> pool2(N * C2 * p2h * p2w);
  std::vector<float> fc1(N * shape.fc1_out);
  std::vector<float> fc2(N * shape.fc2_out);

  // --- Conv1: (N,1,32,32) -> (N,6,28,28), bias, tanh ---
  const float *w1 = weights.data() + weight_offsets[0];
  const float *b1 = biases.data() + bias_offsets[0];
  for (int n = 0; n < N; ++n)
    for (int oc = 0; oc < C1; ++oc)
      for (int oh = 0; oh < c1h; ++oh)
        for (int ow = 0; ow < c1w; ++ow) {
          float acc = b1[oc];
          for (int ic = 0; ic < 1; ++ic)
            for (int kh = 0; kh < K1; ++kh)
              for (int kw = 0; kw < K1; ++kw) {
                float x = input[n * IH * IW + (oh + kh) * IW + (ow + kw)];
                float w = w1[oc * K1 * K1 + kh * K1 + kw];
                acc += x * w;
              }
          conv1[n * C1 * c1h * c1w + oc * c1h * c1w + oh * c1w + ow] =
              std::tanh(acc);
        }

  // --- Pool1: 2x2 avg, stride 2: (N,6,28,28) -> (N,6,14,14) ---
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C1; ++c)
      for (int oh = 0; oh < p1h; ++oh)
        for (int ow = 0; ow < p1w; ++ow) {
          float acc = 0.f;
          for (int kh = 0; kh < PS; ++kh)
            for (int kw = 0; kw < PS; ++kw)
              acc += conv1[n * C1 * c1h * c1w + c * c1h * c1w +
                           (oh * PS + kh) * c1w + (ow * PS + kw)];
          pool1[n * C1 * p1h * p1w + c * p1h * p1w + oh * p1w + ow] =
              acc / (PS * PS);
        }

  // --- Conv2: (N,6,14,14) -> (N,16,10,10), bias, tanh ---
  const float *w2 = weights.data() + weight_offsets[1];
  const float *b2 = biases.data() + bias_offsets[1];
  for (int n = 0; n < N; ++n)
    for (int oc = 0; oc < C2; ++oc)
      for (int oh = 0; oh < c2h; ++oh)
        for (int ow = 0; ow < c2w; ++ow) {
          float acc = b2[oc];
          for (int ic = 0; ic < C1; ++ic)
            for (int kh = 0; kh < K2; ++kh)
              for (int kw = 0; kw < K2; ++kw) {
                float x = pool1[n * C1 * p1h * p1w + ic * p1h * p1w +
                                (oh + kh) * p1w + (ow + kw)];
                float w = w2[oc * C1 * K2 * K2 + ic * K2 * K2 + kh * K2 + kw];
                acc += x * w;
              }
          conv2[n * C2 * c2h * c2w + oc * c2h * c2w + oh * c2w + ow] =
              std::tanh(acc);
        }

  // --- Pool2: 2x2 avg, stride 2: (N,16,10,10) -> (N,16,5,5) ---
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C2; ++c)
      for (int oh = 0; oh < p2h; ++oh)
        for (int ow = 0; ow < p2w; ++ow) {
          float acc = 0.f;
          for (int kh = 0; kh < PS; ++kh)
            for (int kw = 0; kw < PS; ++kw)
              acc += conv2[n * C2 * c2h * c2w + c * c2h * c2w +
                           (oh * PS + kh) * c2w + (ow * PS + kw)];
          pool2[n * C2 * p2h * p2w + c * p2h * p2w + oh * p2w + ow] =
              acc / (PS * PS);
        }

  // --- FC1: (N,400) -> (N,120), bias, tanh ---
  // weight layout: (out_f, in_f) row-major, same as GPU cublasSgemm expects
  const float *wfc1 = weights.data() + weight_offsets[2];
  const float *bfc1 = biases.data() + bias_offsets[2];
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < shape.fc1_out; ++j) {
      float acc = bfc1[j];
      for (int k = 0; k < flat; ++k)
        acc += pool2[n * flat + k] * wfc1[j * flat + k];
      fc1[n * shape.fc1_out + j] = std::tanh(acc);
    }

  // --- FC2: (N,120) -> (N,84), bias, tanh ---
  const float *wfc2 = weights.data() + weight_offsets[3];
  const float *bfc2 = biases.data() + bias_offsets[3];
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < shape.fc2_out; ++j) {
      float acc = bfc2[j];
      for (int k = 0; k < shape.fc1_out; ++k)
        acc += fc1[n * shape.fc1_out + k] * wfc2[j * shape.fc1_out + k];
      fc2[n * shape.fc2_out + j] = std::tanh(acc);
    }

  // --- FC3: (N,84) -> (N,10), bias, no activation ---
  const float *wfc3 = weights.data() + weight_offsets[4];
  const float *bfc3 = biases.data() + bias_offsets[4];
  output.resize(N * shape.fc3_out);
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < shape.fc3_out; ++j) {
      float acc = bfc3[j];
      for (int k = 0; k < shape.fc2_out; ++k)
        acc += fc2[n * shape.fc2_out + k] * wfc3[j * shape.fc2_out + k];
      output[n * shape.fc3_out + j] = acc;
    }
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  LenetShape shape = make_lenet_shape(opt.batch);

  std::vector<float> h_input(shape.input_elements);
  std::vector<float> h_weights(shape.total_weight_elements);
  std::vector<float> h_biases(shape.total_bias_elements);
  std::vector<float> h_output(shape.output_elements, 0.0f);
  std::vector<float> h_ref(shape.output_elements, 0.0f);

  seed_tensor(h_input, 1.0f);
  seed_tensor(h_weights, 0.05f);
  seed_tensor(h_biases, 0.01f);

  float *d_input = nullptr;
  float *d_workspace = nullptr;
  float *d_conv1_out = nullptr;
  float *d_conv2_out = nullptr;
  float *d_pool1_out = nullptr;
  float *d_pool2_out = nullptr;
  float *d_fc1_out = nullptr;
  float *d_fc2_out = nullptr;
  float *d_fc3_out = nullptr;
  float *d_weights = nullptr;
  float *d_biases = nullptr;

  /* DONE(student): cudaMalloc all required activation and weight buffers + copy
   * host data. */
  // Activation buffers
  check_cuda(cudaMalloc(&d_input, shape.input_elements * sizeof(float)),
             "malloc d_input");
  check_cuda(cudaMalloc(&d_conv1_out, shape.conv1_out_elems * sizeof(float)),
             "malloc d_conv1_out");
  check_cuda(cudaMalloc(&d_pool1_out, shape.pool1_out_elems * sizeof(float)),
             "malloc d_pool1_out");
  check_cuda(cudaMalloc(&d_conv2_out, shape.conv2_out_elems * sizeof(float)),
             "malloc d_conv2_out");
  check_cuda(cudaMalloc(&d_pool2_out, shape.pool2_out_elems * sizeof(float)),
             "malloc d_pool2_out");
  check_cuda(cudaMalloc(&d_fc1_out, shape.fc1_out_elems * sizeof(float)),
             "malloc d_fc1_out");
  check_cuda(cudaMalloc(&d_fc2_out, shape.fc2_out_elems * sizeof(float)),
             "malloc d_fc2_out");
  check_cuda(cudaMalloc(&d_fc3_out, shape.output_elements * sizeof(float)),
             "malloc d_fc3_out");
  // Weight and bias buffers (flat, indexed via shape.*_offsets)
  check_cuda(
      cudaMalloc(&d_weights, shape.total_weight_elements * sizeof(float)),
      "malloc d_weights");
  check_cuda(cudaMalloc(&d_biases, shape.total_bias_elements * sizeof(float)),
             "malloc d_biases");

  // Copy host data to device
  check_cuda(cudaMemcpy(d_input, h_input.data(),
                        shape.input_elements * sizeof(float),
                        cudaMemcpyHostToDevice),
             "memcpy d_input");
  check_cuda(cudaMemcpy(d_weights, h_weights.data(),
                        shape.total_weight_elements * sizeof(float),
                        cudaMemcpyHostToDevice),
             "memcpy d_weights");
  check_cuda(cudaMemcpy(d_biases, h_biases.data(),
                        shape.total_bias_elements * sizeof(float),
                        cudaMemcpyHostToDevice),
             "memcpy d_biases");

  cudnnHandle_t cudnn;
  check_cudnn(cudnnCreate(&cudnn), "cudnnCreate");
  cublasHandle_t cublas;
  check_cublas(cublasCreate(&cublas), "cublasCreate");

  LenetDescriptors descs;
  /* DONE(student): initialize tensor/filter/conv/pool descriptors using helpers
   * in lenet_layers.cuh. */
  create_lenet_descriptors(shape, descs);

  // Workspace
  // Query workspace size for both conv layers, take the max so one buffer
  // covers both
  cudnnConvolutionFwdAlgo_t algo_enum = parse_algo(opt.algo);
  size_t ws1 = query_conv_workspace(cudnn, shape, descs, algo_enum, false);
  size_t ws2 = query_conv_workspace(cudnn, shape, descs, algo_enum, true);
  size_t ws_bytes = std::max(ws1, ws2);
  if (ws_bytes > 0) {
    check_cuda(cudaMalloc(&d_workspace, ws_bytes), "malloc d_workspace");
  }

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start");
  check_cuda(cudaEventCreate(&stop), "create stop");

  float elapsed_ms = 0.0f;
  if (opt.impl == "baseline") {
    check_cuda(cudaEventRecord(start), "record start baseline");
    /* DONE(student):
       1. run_lenet_conv for conv1/conv2 using opt.algo
       2. launch_lenet_pool for each pooling stage
       3. reshape tensor for FC input (either via dedicated kernel or by
       treating memory as-is)
       4. run_fc_layer (cuBLAS GEMM + bias + activation) for the dense blocks
    */

    // Conv1 → pool1
    run_lenet_conv(cudnn, shape, descs, d_input,
                   d_weights + shape.weight_offsets[0],
                   d_biases + shape.bias_offsets[0], d_conv1_out, d_workspace,
                   ws_bytes, opt.algo, false);
    run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);
    // Conv2 → pool2
    run_lenet_conv(cudnn, shape, descs, d_pool1_out,
                   d_weights + shape.weight_offsets[1],
                   d_biases + shape.bias_offsets[1], d_conv2_out, d_workspace,
                   ws_bytes, opt.algo, true);
    run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);
    // Reshape pool2 output (N,16,5,5) -> (N,400) — no-op since same buffer
    // layout
    reshape_conv_to_fc(shape, d_pool2_out, d_pool2_out, 0);
    // FC1: (N,400) -> (N,120) + tanh
    run_fc_layer(cublas, shape, 0, d_pool2_out,
                 d_weights + shape.weight_offsets[2],
                 d_biases + shape.bias_offsets[2], d_fc1_out, 0);
    // FC2: (N,120) -> (N,84) + tanh
    run_fc_layer(cublas, shape, 1, d_fc1_out,
                 d_weights + shape.weight_offsets[3],
                 d_biases + shape.bias_offsets[3], d_fc2_out, 0);
    // FC3: (N,84) -> (N,10), no activation
    run_fc_layer(cublas, shape, 2, d_fc2_out,
                 d_weights + shape.weight_offsets[4],
                 d_biases + shape.bias_offsets[4], d_fc3_out, 0);

    check_cuda(cudaEventRecord(stop), "record stop baseline");
    check_cuda(cudaEventSynchronize(stop), "sync stop baseline");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "elapsed baseline");
  } else if (opt.impl == "fused") {
    check_cuda(cudaEventRecord(start), "record start fused");

    /* DONE(student): same as baseline but fuse activation/bias where possible.
     */

    // Conv1 → pool1
    run_lenet_conv_fused(cudnn, shape, descs, d_input,
                         d_weights + shape.weight_offsets[0],
                         d_biases + shape.bias_offsets[0], d_conv1_out,
                         d_workspace, ws_bytes, opt.algo, false);
    run_lenet_pool(cudnn, descs, d_conv1_out, d_pool1_out, false);
    // Conv2 → pool2
    run_lenet_conv_fused(cudnn, shape, descs, d_pool1_out,
                         d_weights + shape.weight_offsets[1],
                         d_biases + shape.bias_offsets[1], d_conv2_out,
                         d_workspace, ws_bytes, opt.algo, true);
    run_lenet_pool(cudnn, descs, d_conv2_out, d_pool2_out, true);
    // Reshape pool2 output (N,16,5,5) -> (N,400) — no-op since same buffer
    // layout
    reshape_conv_to_fc(shape, d_pool2_out, d_pool2_out, 0);
    // FC1: (N,400) -> (N,120) + tanh
    run_fc_layer(cublas, shape, 0, d_pool2_out,
                 d_weights + shape.weight_offsets[2],
                 d_biases + shape.bias_offsets[2], d_fc1_out, 0);
    // FC2: (N,120) -> (N,84) + tanh
    run_fc_layer(cublas, shape, 1, d_fc1_out,
                 d_weights + shape.weight_offsets[3],
                 d_biases + shape.bias_offsets[3], d_fc2_out, 0);
    // FC3: (N,84) -> (N,10), no activation
    run_fc_layer(cublas, shape, 2, d_fc2_out,
                 d_weights + shape.weight_offsets[4],
                 d_biases + shape.bias_offsets[4], d_fc3_out, 0);

    check_cuda(cudaEventRecord(stop), "record stop fused");
    check_cuda(cudaEventSynchronize(stop), "sync stop fused");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
  } else {
    throw std::invalid_argument("Unknown --impl=" + opt.impl);
  }

  /* DONE(student): copy logits from device to h_output. */
  check_cuda(cudaMemcpy(h_output.data(), d_fc3_out,
                        shape.output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "memcpy d_fc3_out -> h_output");

  if (!opt.dump_path.empty()) {
    std::ofstream ofs(opt.dump_path, std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
    }
    ofs.write(reinterpret_cast<const char *>(h_output.data()),
              static_cast<std::streamsize>(h_output.size() * sizeof(float)));
    ofs.close();
  }

  if (opt.verify) {
    lenet_cpu_reference(opt, shape, h_weights, shape.weight_offsets, h_biases,
                        shape.bias_offsets, h_input, h_ref);
    /* DONE(student): compute and print max abs diff between h_output and h_ref.
     */

    float max_diff = 0.f, mean_diff = 0.f;
    for (size_t i = 0; i < h_output.size(); ++i) {
      float d = std::abs(h_output[i] - h_ref[i]);
      max_diff = std::max(max_diff, d);
      mean_diff += d;
    }
    mean_diff /= static_cast<float>(h_output.size());
    std::cout << std::fixed << std::setprecision(9)
              << "Verify: max_abs_diff=" << max_diff
              << " mean_abs_diff=" << mean_diff
              << (max_diff < 1e-3f ? "  PASS" : "  FAIL") << std::endl;
  }

  if (elapsed_ms > 0.0f) {
    std::cout << std::fixed << std::setprecision(2) << "Impl=" << opt.impl
              << " Batch=" << opt.batch << " Algo=" << opt.algo
              << " Time(ms)=" << elapsed_ms
              << " GFLOP/s=" << lenet_gflops(shape, elapsed_ms) << std::endl;
  } else {
    std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
  }

  /* DONE(student): destroy descriptors, handles, free device buffers, destroy
   * events. */

  // Handles
  cudnnDestroy(cudnn);
  cublasDestroy(cublas);
  // Device buffers
  cudaFree(d_input);
  cudaFree(d_conv1_out);
  cudaFree(d_pool1_out);
  cudaFree(d_conv2_out);
  cudaFree(d_pool2_out);
  cudaFree(d_fc1_out);
  cudaFree(d_fc2_out);
  cudaFree(d_fc3_out);
  cudaFree(d_weights);
  cudaFree(d_biases);
  cudaFree(d_workspace);
  // Events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
