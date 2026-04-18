#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
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
  std::string graph_prefix =
      "data/cora"; // expects graph_prefix.csr, graph_prefix.feat,
                   // graph_prefix.label
  int hidden_dim = 128;
  int layers = 2;
  std::string impl = "baseline"; // baseline | fused
  bool verify = true;
  std::string dump_path = "";
};

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if ((strcmp(argv[i], "--graph") == 0 || strcmp(argv[i], "-g") == 0) &&
        i + 1 < argc) {
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
      std::cout << "Usage: ./dgcn --graph data/cora --hidden 128 --layers 2 "
                   "--impl baseline \\\n  [--dump outputs.bin] [--no-verify]\n";
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

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);

  GraphData graph;
  /* DONE(student): load CSR graph + features + labels from opt.graph_prefix
   * using helpers. */
  build_graph_from_files(opt.graph_prefix, graph);

  std::cout << "Loaded: nodes=" << graph.num_nodes << " nnz=" << graph.nnz
            << " edges=" << graph.num_edges << " feat_dim=" << graph.feature_dim
            << " classes=" << graph.num_classes << std::endl;

  cusparseHandle_t cusparse;
  check_cusparse(cusparseCreate(&cusparse), "cusparseCreate");
  cublasHandle_t cublas;
  check_cublas(cublasCreate(&cublas), "cublasCreate");

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "create start event");
  check_cuda(cudaEventCreate(&stop), "create stop event");

  DeviceGCNWorkspace workspace;
  /* DONE(student): allocate device buffers for features, normalized adjacency,
   * intermediate activations, weights. */
  allocate_device_graph(graph, opt.hidden_dim, opt.layers, workspace);

  const auto w_dims = layer_weight_dims(graph.feature_dim, opt.hidden_dim,
                                        graph.num_classes, opt.layers);
  std::vector<float> h_weights;
  init_weights_xavier(h_weights, w_dims, /*seed=*/42);
  check_cuda(cudaMemcpy(workspace.d_weights, h_weights.data(),
                        h_weights.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "memcpy weights H2D");

  std::cout << "Weights initialized:";
  for (size_t i = 0; i < w_dims.size(); ++i) {
    std::cout << " W" << i << "=" << w_dims[i].first << "x" << w_dims[i].second;
  }
  std::cout << " total=" << h_weights.size() << " floats" << std::endl;

  if (!opt.dump_path.empty()) {
    std::ofstream ofs("weights.bin", std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Could not open weights.bin for writing");
    }
    ofs.write(reinterpret_cast<const char *>(h_weights.data()),
              static_cast<std::streamsize>(h_weights.size() * sizeof(float)));
    ofs.close();
    std::cout << "Weights dumped to weights.bin ("
              << h_weights.size() * sizeof(float) << " bytes)" << std::endl;
  }

  std::vector<size_t> w_offsets(w_dims.size() + 1, 0);
  for (size_t i = 0; i < w_dims.size(); ++i) {
    w_offsets[i + 1] =
        w_offsets[i] + static_cast<size_t>(w_dims[i].first) * w_dims[i].second;
  }

  float elapsed_ms = 0.0f;
  if (opt.impl == "baseline") {
    check_cuda(cudaEventRecord(start), "record baseline start");
    /* DONE(student): run forward pass using cusparseSpMM + cublasSgemm per
     * layer. */

    const float *cur_input = workspace.d_features_in;
    const int L = static_cast<int>(w_dims.size());
    for (int l = 0; l < L; ++l) {
      const int in_dim = w_dims[l].first;
      const int out_dim = w_dims[l].second;
      const bool is_last = (l == L - 1);
      float *gemm_out = is_last ? workspace.d_logits : workspace.d_features_out;

      run_sparse_dense_mm(cusparse, workspace,
                          /*rows=*/graph.num_nodes,
                          /*cols=*/in_dim,
                          /*K=*/0, cur_input, workspace.d_temp);
      run_dense_layer(cublas,
                      /*M=*/graph.num_nodes,
                      /*K=*/in_dim,
                      /*N=*/out_dim, workspace.d_temp,
                      workspace.d_weights + w_offsets[l], gemm_out);
      if (!is_last) {
        apply_activation(gemm_out, graph.num_nodes * out_dim,
                         /*stream=*/0);
      }
      cur_input = gemm_out;
    }

    check_cuda(cudaEventRecord(stop), "record baseline stop");
    check_cuda(cudaEventSynchronize(stop), "sync baseline stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "elapsed baseline");
  } else if (opt.impl == "fused") {
    check_cuda(cudaEventRecord(start), "record fused start");
    /* TODO(student): implement fused kernels (e.g., combine aggregation +
     * activation) and time here. */
    check_cuda(cudaEventRecord(stop), "record fused stop");
    check_cuda(cudaEventSynchronize(stop), "sync fused stop");
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed fused");
  } else {
    throw std::invalid_argument("Unknown --impl=" + opt.impl);
  }

  std::vector<float> h_logits(graph.num_nodes * graph.num_classes, 0.0f);
  /* DONE(student): copy device logits back into h_logits. */
  check_cuda(cudaMemcpy(h_logits.data(), workspace.d_logits,
                        h_logits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "D2H d_logits -> h_logits");

  // // TEMP: Remove
  // // Eyeball sanity: first row of logits. Values should be finite and of
  // // modest magnitude (typically ~1e-3 to ~1e-1 given Xavier init + ReLU).
  // std::cout << "Logits row 0:";
  // for (int c = 0; c < graph.num_classes; ++c)
  //   std::cout << " " << h_logits[c];
  // std::cout << std::endl;

  if (!opt.dump_path.empty()) {
    std::ofstream ofs(opt.dump_path, std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Failed to open dump path: " + opt.dump_path);
    }
    ofs.write(reinterpret_cast<const char *>(h_logits.data()),
              static_cast<std::streamsize>(h_logits.size() * sizeof(float)));
    ofs.close();
  }

  if (opt.verify) {
    /* TODO(student): run DGL/PyTorch reference (e.g., via subprocess) or CPU
     * path to compare logits. */
  }

  if (elapsed_ms > 0.0f) {
    std::cout << std::fixed << std::setprecision(2) << "Impl=" << opt.impl
              << " Graph=" << opt.graph_prefix << " Hidden=" << opt.hidden_dim
              << " Layers=" << opt.layers << " Time(ms)=" << elapsed_ms
              << " Edges/s=" << graph.nnz / (elapsed_ms * 1e-3) << std::endl;
  } else {
    std::cout << "Forward pass executed (timing TODO incomplete)." << std::endl;
  }

  /* DONE(student): free device buffers, destroy cuBLAS/cuSPARSE handles,
   * destroy events. */

  destroy_device_graph(workspace);
  check_cuda(cudaEventDestroy(start), "destroy start event");
  check_cuda(cudaEventDestroy(stop), "destroy stop event");
  check_cublas(cublasDestroy(cublas), "cublasDestroy");
  check_cusparse(cusparseDestroy(cusparse), "cusparseDestroy");

  return 0;
}
