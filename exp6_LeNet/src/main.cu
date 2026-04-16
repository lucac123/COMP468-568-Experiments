#include <cuda_runtime.h>
#include <cublas_v2.h>
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
    std::string algo = "implicit_gemm";       // cuDNN conv algo hint
    std::string impl = "baseline";            // baseline | fused
    bool verify = true;
    std::string dump_path = "";               // optional binary file for logits
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc) {
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
            std::cout << "Usage: ./dlenet --batch N --algo implicit_gemm --impl baseline|fused \\\n  [--dump outputs.bin] [--no-verify]\n";
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

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : " + cudnnGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

void seed_tensor(std::vector<float>& vec, float scale) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = scale * std::sin(0.017f * static_cast<float>(i));
    }
}

void lenet_cpu_reference(const Options& opt,
                         const LenetShape& shape,
                         const std::vector<float>& weights,
                         const std::vector<size_t>& weight_offsets,
                         const std::vector<float>& biases,
                         const std::vector<size_t>& bias_offsets,
                         const std::vector<float>& input,
                         std::vector<float>& output) {
    /* TODO(student): implement a simple CPU LeNet forward (conv/pool/activations/GEMM).
       Keep it single-threaded for simplicity or call into a reference framework. */
    (void)opt;
    (void)shape;
    (void)weights;
    (void)weight_offsets;
    (void)biases;
    (void)bias_offsets;
    (void)input;
    (void)output;
}

int main(int argc, char** argv) {
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

    float* d_input = nullptr;
    float* d_workspace = nullptr;
    float* d_conv1_out = nullptr;
    float* d_conv2_out = nullptr;
    float* d_pool1_out = nullptr;
    float* d_pool2_out = nullptr;
    float* d_fc1_out = nullptr;
    float* d_fc2_out = nullptr;
    float* d_fc3_out = nullptr;
    float* d_weights = nullptr;
    float* d_biases = nullptr;
    /* TODO(student): cudaMalloc all required activation and weight buffers + copy host data. */
    (void)d_workspace;
    (void)d_conv1_out;
    (void)d_conv2_out;
    (void)d_pool1_out;
    (void)d_pool2_out;
    (void)d_fc1_out;
    (void)d_fc2_out;
    (void)d_fc3_out;
    (void)d_weights;
    (void)d_biases;

    cudnnHandle_t cudnn;
    check_cudnn(cudnnCreate(&cudnn), "cudnnCreate");
    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    LenetDescriptors descs;
    /* TODO(student): initialize tensor/filter/conv/pool descriptors using helpers in lenet_layers.cuh. */
    (void)descs;

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start), "record start baseline");
        /* TODO(student):
           1. run_lenet_conv for conv1/conv2 using opt.algo
           2. launch_lenet_pool for each pooling stage
           3. reshape tensor for FC input (either via dedicated kernel or by treating memory as-is)
           4. run_fc_layer (cuBLAS GEMM + bias + activation) for the dense blocks
        */
        check_cuda(cudaEventRecord(stop), "record stop baseline");
        check_cuda(cudaEventSynchronize(stop), "sync stop baseline");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline");
    } else if (opt.impl == "fused") {
        check_cuda(cudaEventRecord(start), "record start fused");
        /* TODO(student): same as baseline but fuse activation/bias where possible. */
        check_cuda(cudaEventRecord(stop), "record stop fused");
        check_cuda(cudaEventSynchronize(stop), "sync stop fused");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
    } else {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    /* TODO(student): copy logits from device to h_output. */

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_output.data()),
                  static_cast<std::streamsize>(h_output.size() * sizeof(float)));
        ofs.close();
    }

    if (opt.verify) {
        lenet_cpu_reference(opt,
                            shape,
                            h_weights,
                            shape.weight_offsets,
                            h_biases,
                            shape.bias_offsets,
                            h_input,
                            h_ref);
        /* TODO(student): compute and print max abs diff between h_output and h_ref. */
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Batch=" << opt.batch
                  << " Algo=" << opt.algo
                  << " Time(ms)=" << elapsed_ms
                  << " GFLOP/s=" << lenet_gflops(shape, elapsed_ms) << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    /* TODO(student): destroy descriptors, handles, free device buffers, destroy events. */
    return 0;
}
