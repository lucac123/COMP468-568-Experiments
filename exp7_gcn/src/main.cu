#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gcn_layers.cuh"

struct Options {
    std::string graph_prefix = "data/cora";  // expects graph_prefix.csr, graph_prefix.feat, graph_prefix.label
    int hidden_dim = 128;
    int layers = 2;
    std::string impl = "baseline";  // baseline | fused
    bool verify = true;
    std::string dump_path = "";
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--graph") == 0 || strcmp(argv[i], "-g") == 0) && i + 1 < argc) {
            opt.graph_prefix = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            opt.hidden_dim = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            opt.layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            opt.dump_path = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgcn --graph data/cora --hidden 128 --layers 2 --impl baseline \\\n  [--dump outputs.bin] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    if (opt.hidden_dim <= 0 || opt.layers < 1) {
        throw std::invalid_argument("hidden and layers must be positive");
    }
    return opt;
}





int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    GraphData graph;
    /* TODO(student): load CSR graph + features + labels from opt.graph_prefix using helpers. */
    (void)graph;

    cusparseHandle_t cusparse;
    check_cusparse(cusparseCreate(&cusparse), "cusparseCreate");
    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    DeviceGCNWorkspace workspace;
    /* TODO(student): allocate device buffers for features, normalized adjacency, intermediate activations, weights. */
    (void)workspace;


    // keep weights on host for dumping later
    //  --dump is required; --verify may also need it
    // if (!opt.dump_path.empty()) {
    //     std::ofstream ofs("weights.bin", std::ios::binary);
    //     if (ofs) {
    //         ofs.write(reinterpret_cast<const char*>(h_weights.data()), h_weights.size() * sizeof(float));
    //         ofs.close();
    //         std::cout << "Weights dumped to weights.bin" << std::endl;
    //     } else {
    //         std::cerr << "Error: Could not write to weights.bin" << std::endl;
    //     }
    // }
    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline") {
        check_cuda(cudaEventRecord(start), "record baseline start");
        /* TODO(student): run forward pass using cusparseSpMM + cublasSgemm per layer. */
        check_cuda(cudaEventRecord(stop), "record baseline stop");
        check_cuda(cudaEventSynchronize(stop), "sync baseline stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed baseline");
    } else if (opt.impl == "fused") {
        check_cuda(cudaEventRecord(start), "record fused start");
        /* TODO(student): implement fused kernels (e.g., combine aggregation + activation) and time here. */
        check_cuda(cudaEventRecord(stop), "record fused stop");
        check_cuda(cudaEventSynchronize(stop), "sync fused stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
    } else {
        throw std::invalid_argument("Unknown --impl=" + opt.impl);
    }

    std::vector<float> h_logits(graph.num_nodes * graph.num_classes, 0.0f);
    /* TODO(student): copy device logits back into h_logits. */

    if (!opt.dump_path.empty()) {
        std::ofstream ofs(opt.dump_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
        }
        ofs.write(reinterpret_cast<const char*>(h_logits.data()),
                  static_cast<std::streamsize>(h_logits.size() * sizeof(float)));
        ofs.close();
    }

    if (opt.verify) {
        /* TODO(student): run DGL/PyTorch reference (e.g., via subprocess) or CPU path to compare logits. */
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2)
                  << "Impl=" << opt.impl
                  << " Graph=" << opt.graph_prefix
                  << " Hidden=" << opt.hidden_dim
                  << " Layers=" << opt.layers
                  << " Time(ms)=" << elapsed_ms
                  << " Edges/s=" << graph.nnz / (elapsed_ms * 1e-3)
                  << std::endl;
    } else {
        std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
    }

    /* TODO(student): free device buffers, destroy cuBLAS/cuSPARSE handles, destroy events. */
    return 0;
}
