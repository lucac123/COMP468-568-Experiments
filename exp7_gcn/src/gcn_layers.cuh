#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct GraphData {
  int num_nodes = 0;
  int num_edges = 0; // undirected edges counted twice
  int nnz = 0;       // CSR nnz (including self loops)
  int feature_dim = 0;
  int num_classes = 0;

  std::vector<int> h_csr_row_offsets;
  std::vector<int> h_csr_col_indices;
  std::vector<float> h_csr_values;
  std::vector<float> h_features; // row-major: num_nodes x feature_dim
  std::vector<int> h_labels;     // node labels
};

struct DeviceGCNWorkspace {
  int *d_csr_row_offsets = nullptr;
  int *d_csr_col_indices = nullptr;
  float *d_csr_values = nullptr;

  float *d_features_in = nullptr;
  float *d_features_out = nullptr;
  float *d_weights = nullptr;
  float *d_logits = nullptr;
  float *d_temp = nullptr;

  cusparseSpMatDescr_t spmat = nullptr;
  cusparseDnMatDescr_t dn_left = nullptr;
  cusparseDnMatDescr_t dn_right = nullptr;
  cusparseDnMatDescr_t dn_out = nullptr;
};

inline void check_cusparse(cusparseStatus_t status, const char *msg) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : cuSPARSE error");
  }
}

inline void check_cublas(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " : cuBLAS error");
  }
}

inline void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " : " +
                             cudaGetErrorString(err));
  }
}

inline std::streamsize file_size_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return f.tellg();
}

inline void build_graph_from_files(const std::string &prefix,
                                   GraphData &graph) {
  /* DONE(student):
     1. Read CSR metadata from prefix + ".csr" (e.g., row ptr count, nnz,
     columns, values).
     2. Read dense features from prefix + ".feat" (float32 rows).
     3. Read labels from prefix + ".label" (int32).
     4. Populate graph struct and add self-loops + normalization coefficients.
   */

  const std::string csr_path = prefix + ".csr";
  const std::string feat_path = prefix + ".feat";
  const std::string label_path = prefix + ".label";

  // ---- Read CSR ----
  // Layout: [num_nodes (int32), nnz (int32)] || row_offsets[num_nodes+1]
  // (int32) || col_indices[nnz] (int32)
  std::ifstream csr_file(csr_path, std::ios::binary);
  if (!csr_file) {
    throw std::runtime_error("Failed to open CSR file: " + csr_path);
  }

  int header[2] = {0, 0};
  csr_file.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!csr_file) {
    throw std::runtime_error("Failed to read CSR header from: " + csr_path);
  }
  graph.num_nodes = header[0];
  graph.nnz = header[1];
  if (graph.num_nodes <= 0 || graph.nnz <= 0) {
    throw std::runtime_error(
        "Invalid CSR header (num_nodes or nnz non-positive) in: " + csr_path);
  }

  graph.h_csr_row_offsets.resize(static_cast<size_t>(graph.num_nodes) + 1);
  csr_file.read(
      reinterpret_cast<char *>(graph.h_csr_row_offsets.data()),
      static_cast<std::streamsize>((graph.num_nodes + 1) * sizeof(int)));
  if (!csr_file) {
    throw std::runtime_error("Failed to read row_offsets from: " + csr_path);
  }

  graph.h_csr_col_indices.resize(static_cast<size_t>(graph.nnz));
  csr_file.read(reinterpret_cast<char *>(graph.h_csr_col_indices.data()),
                static_cast<std::streamsize>(graph.nnz * sizeof(int)));
  if (!csr_file) {
    throw std::runtime_error("Failed to read col_indices from: " + csr_path);
  }

  // Validate CSR: last row offset should equal nnz; col indices should be
  // in-range.
  if (graph.h_csr_row_offsets.back() != graph.nnz) {
    throw std::runtime_error("CSR inconsistency: row_offsets.back() != nnz");
  }
  // num_edges counts non-self-loop nonzeros, roughly. Self-loops contribute
  // num_nodes to nnz.
  graph.num_edges = graph.nnz - graph.num_nodes;

  // Placeholder normalization values (Step 3 will replace with
  // 1/sqrt(deg[i]*deg[j])).
  graph.h_csr_values.assign(static_cast<size_t>(graph.nnz), 1.0f);

  // ---- Read features ----
  // File is raw row-major float32; we infer feature_dim from file size.
  const std::streamsize feat_bytes = file_size_bytes(feat_path);
  if (feat_bytes %
          (static_cast<std::streamsize>(graph.num_nodes) * sizeof(float)) !=
      0) {
    throw std::runtime_error(
        "Feature file size not divisible by num_nodes*sizeof(float): " +
        feat_path);
  }
  graph.feature_dim =
      static_cast<int>(feat_bytes / (graph.num_nodes * sizeof(float)));
  if (graph.feature_dim <= 0) {
    throw std::runtime_error("Computed feature_dim non-positive for: " +
                             feat_path);
  }

  graph.h_features.resize(static_cast<size_t>(graph.num_nodes) *
                          graph.feature_dim);
  {
    std::ifstream feat_file(feat_path, std::ios::binary);
    feat_file.read(reinterpret_cast<char *>(graph.h_features.data()),
                   feat_bytes);
    if (!feat_file) {
      throw std::runtime_error("Failed to read features from: " + feat_path);
    }
  }

  // ---- Read labels ----
  const std::streamsize label_bytes = file_size_bytes(label_path);
  const std::streamsize expected_label_bytes =
      static_cast<std::streamsize>(graph.num_nodes) * sizeof(int);
  if (label_bytes != expected_label_bytes) {
    throw std::runtime_error("Label file size mismatch (expected " +
                             std::to_string(expected_label_bytes) +
                             " bytes) in: " + label_path);
  }
  graph.h_labels.resize(static_cast<size_t>(graph.num_nodes));
  {
    std::ifstream label_file(label_path, std::ios::binary);
    label_file.read(reinterpret_cast<char *>(graph.h_labels.data()),
                    label_bytes);
    if (!label_file) {
      throw std::runtime_error("Failed to read labels from: " + label_path);
    }
  }

  // num_classes = max(labels) + 1 (Cora/Citeseer use dense 0-indexed labels).
  int max_label = 0;
  for (int l : graph.h_labels) {
    if (l < 0) {
      throw std::runtime_error("Encountered negative label in: " + label_path);
    }
    if (l > max_label)
      max_label = l;
  }
  graph.num_classes = max_label + 1;
}

inline void allocate_device_graph(const GraphData &graph,
                                  DeviceGCNWorkspace &workspace) {
  /* TODO(student): cudaMalloc / cudaMemcpy CSR + feature buffers, create
   * cusparse descriptors. */
  (void)graph;
  (void)workspace;
}

inline void destroy_device_graph(DeviceGCNWorkspace &workspace) {
  /* TODO(student): destroy descriptors and cudaFree buffers. */
  (void)workspace;
}

inline void run_sparse_dense_mm(cusparseHandle_t handle,
                                DeviceGCNWorkspace &workspace, int rows,
                                int cols, int K, const float *d_input,
                                float *d_output) {
  /* TODO(student): configure cusparseDnMatDescr_t for input/output and call
     cusparseSpMM. rows = num_nodes, cols = hidden_dim, K = feature_dim. */
  (void)handle;
  (void)workspace;
  (void)rows;
  (void)cols;
  (void)K;
  (void)d_input;
  (void)d_output;
}

inline void run_dense_layer(cublasHandle_t handle, int M, int K, int N,
                            const float *d_input, const float *d_weight,
                            float *d_output) {
  /* TODO(student): call cublasSgemm to compute (M x K) * (K x N). */
  (void)handle;
  (void)M;
  (void)K;
  (void)N;
  (void)d_input;
  (void)d_weight;
  (void)d_output;
}

inline void apply_activation(float *d_tensor, int elements,
                             cudaStream_t stream) {
  /* TODO(student): implement ReLU or ELU kernel. */
  (void)d_tensor;
  (void)elements;
  (void)stream;
}

inline void apply_dropout(float *d_tensor, int elements, float drop_prob,
                          cudaStream_t stream) {
  /* TODO(student): optional – implement dropout. */
  (void)d_tensor;
  (void)elements;
  (void)drop_prob;
  (void)stream;
}

inline void softmax_cross_entropy(const float *d_logits, const int *d_labels,
                                  int num_nodes, int num_classes,
                                  float *d_loss) {
  /* TODO(student): compute loss/accuracy or copy logits for host-side
   * evaluation. */
  (void)d_logits;
  (void)d_labels;
  (void)num_nodes;
  (void)num_classes;
  (void)d_loss;
}
