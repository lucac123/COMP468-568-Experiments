#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
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

  void *d_spmm_workspace = nullptr;
  size_t spmm_workspace_bytes = 0;

  size_t features_in_elems = 0;
  size_t activation_elems = 0;
  size_t logits_elems = 0;
  size_t weights_elems = 0;
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

// Fill csr_values with the symmetric GCN normalization coefficient
//   value(i, j) = 1 / sqrt(deg(i) * deg(j))
// where deg(k) is the out-degree after self-loops have been added.
// This matches DGL's GraphConv(norm='both') exactly, which is what
// compare_with_dgl.py uses as the reference.
//
// Assumes the CSR already contains self-loops (prepare_data.py adds them).
// If a node has degree 0 we leave the corresponding values as 0; the row is
// empty anyway, so any value there is dead weight.
inline void compute_symmetric_normalization(const std::vector<int> &row_offsets,
                                            const std::vector<int> &col_indices,
                                            std::vector<float> &csr_values) {
  const int num_nodes = static_cast<int>(row_offsets.size()) - 1;
  const int nnz = static_cast<int>(col_indices.size());

  if (static_cast<int>(csr_values.size()) != nnz) {
    throw std::runtime_error(
        "compute_symmetric_normalization: csr_values size mismatch");
  }

  // Degrees are just row lengths.
  std::vector<float> inv_sqrt_deg(static_cast<size_t>(num_nodes), 0.0f);
  for (int i = 0; i < num_nodes; ++i) {
    const int deg = row_offsets[i + 1] - row_offsets[i];
    if (deg > 0) {
      inv_sqrt_deg[i] = 1.0f / std::sqrt(static_cast<float>(deg));
    }
    // deg == 0 leaves inv_sqrt_deg[i] = 0; products with 0 stay 0.
  }

  // Fill values.
  for (int i = 0; i < num_nodes; ++i) {
    const int row_start = row_offsets[i];
    const int row_end = row_offsets[i + 1];
    const float di = inv_sqrt_deg[i];
    for (int p = row_start; p < row_end; ++p) {
      const int j = col_indices[p];
      csr_values[p] = di * inv_sqrt_deg[j];
    }
  }
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

  // Compute symmetric normalization coefficients for the CSR values.
  graph.h_csr_values.assign(static_cast<size_t>(graph.nnz), 0.0f);
  compute_symmetric_normalization(graph.h_csr_row_offsets,
                                  graph.h_csr_col_indices, graph.h_csr_values);

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

// Describe the per-layer weight slabs for a GCN with `layers` layers,
// feature_dim -> hidden_dim (repeated) -> num_classes.
// Returns a vector of (in_dim, out_dim) pairs, one per layer, in the same
// order DGL's DGLGCN iterates self.layers.
inline std::vector<std::pair<int, int>> layer_weight_dims(int feature_dim,
                                                          int hidden_dim,
                                                          int num_classes,
                                                          int layers) {
  std::vector<std::pair<int, int>> dims;
  dims.reserve(static_cast<size_t>(layers));
  if (layers == 1) {
    dims.emplace_back(feature_dim, num_classes);
  } else {
    dims.emplace_back(feature_dim, hidden_dim);
    for (int i = 0; i < layers - 2; ++i) {
      dims.emplace_back(hidden_dim, hidden_dim);
    }
    dims.emplace_back(hidden_dim, num_classes);
  }
  return dims;
}

// Xavier/Glorot uniform init: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).
// Deterministic via fixed seed — essential for reproducing logits in
// compare_with_dgl.py (which reloads this exact buffer from weights.bin).
inline void init_weights_xavier(std::vector<float> &h_weights,
                                const std::vector<std::pair<int, int>> &dims,
                                uint32_t seed = 42) {
  size_t total = 0;
  for (auto [in_d, out_d] : dims)
    total += static_cast<size_t>(in_d) * out_d;
  h_weights.resize(total);

  std::mt19937 rng(seed);
  size_t offset = 0;
  for (auto [in_d, out_d] : dims) {
    const float a = std::sqrt(6.0f / static_cast<float>(in_d + out_d));
    std::uniform_real_distribution<float> dist(-a, a);
    const size_t n = static_cast<size_t>(in_d) * out_d;
    for (size_t i = 0; i < n; ++i) {
      h_weights[offset + i] = dist(rng);
    }
    offset += n;
  }
}

inline void allocate_device_graph(const GraphData &graph, int hidden_dim,
                                  int layers, DeviceGCNWorkspace &workspace) {
  /* DONE(student): cudaMalloc / cudaMemcpy CSR + feature buffers, create
   * cusparse descriptors. */

  // --- Sizes ---
  const size_t nodes = static_cast<size_t>(graph.num_nodes);
  const size_t nnz = static_cast<size_t>(graph.nnz);
  const size_t feat_dim = static_cast<size_t>(graph.feature_dim);
  const size_t num_classes = static_cast<size_t>(graph.num_classes);
  const size_t hid = static_cast<size_t>(hidden_dim);

  // Activation buffers must fit the widest intermediate. During the forward
  // pass the per-node width at any point is one of: feature_dim (before
  // layer 0), hidden_dim (between layers), or num_classes (after last GEMM).
  // Pick the max so d_temp/d_features_out can hold any of them.
  const size_t max_width = std::max({feat_dim, hid, num_classes});
  const size_t activation_elems = nodes * max_width;
  const size_t logits_elems = nodes * num_classes;
  const size_t features_in_elems = nodes * feat_dim;

  // --- CSR on device ---
  check_cuda(
      cudaMalloc(&workspace.d_csr_row_offsets, (nodes + 1) * sizeof(int)),
      "cudaMalloc d_csr_row_offsets");
  check_cuda(cudaMalloc(&workspace.d_csr_col_indices, nnz * sizeof(int)),
             "cudaMalloc d_csr_col_indices");
  check_cuda(cudaMalloc(&workspace.d_csr_values, nnz * sizeof(float)),
             "cudaMalloc d_csr_values");

  check_cuda(cudaMemcpy(workspace.d_csr_row_offsets,
                        graph.h_csr_row_offsets.data(),
                        (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice),
             "memcpy row_offsets H2D");
  check_cuda(cudaMemcpy(workspace.d_csr_col_indices,
                        graph.h_csr_col_indices.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice),
             "memcpy col_indices H2D");
  check_cuda(cudaMemcpy(workspace.d_csr_values, graph.h_csr_values.data(),
                        nnz * sizeof(float), cudaMemcpyHostToDevice),
             "memcpy csr_values H2D");

  // --- Input features ---
  check_cuda(
      cudaMalloc(&workspace.d_features_in, features_in_elems * sizeof(float)),
      "cudaMalloc d_features_in");
  check_cuda(cudaMemcpy(workspace.d_features_in, graph.h_features.data(),
                        features_in_elems * sizeof(float),
                        cudaMemcpyHostToDevice),
             "memcpy features H2D");
  workspace.features_in_elems = features_in_elems;

  // --- Ping-pong activation buffers + logits ---
  check_cuda(cudaMalloc(&workspace.d_temp, activation_elems * sizeof(float)),
             "cudaMalloc d_temp");
  check_cuda(
      cudaMalloc(&workspace.d_features_out, activation_elems * sizeof(float)),
      "cudaMalloc d_features_out");
  check_cuda(cudaMalloc(&workspace.d_logits, logits_elems * sizeof(float)),
             "cudaMalloc d_logits");
  workspace.activation_elems = activation_elems;
  workspace.logits_elems = logits_elems;

  // Zero the activation buffers so that stale NaNs never leak into output.
  // Not strictly required for correctness (forward pass overwrites), but
  // it's cheap and makes debugging easier.
  check_cuda(cudaMemset(workspace.d_temp, 0, activation_elems * sizeof(float)),
             "memset d_temp");
  check_cuda(
      cudaMemset(workspace.d_features_out, 0, activation_elems * sizeof(float)),
      "memset d_features_out");
  check_cuda(cudaMemset(workspace.d_logits, 0, logits_elems * sizeof(float)),
             "memset d_logits");

  // --- Weights (allocate; host fills + copies in main) ---
  const auto dims = layer_weight_dims(graph.feature_dim, hidden_dim,
                                      graph.num_classes, layers);
  size_t weights_elems = 0;
  for (auto [in_d, out_d] : dims)
    weights_elems += static_cast<size_t>(in_d) * out_d;
  check_cuda(cudaMalloc(&workspace.d_weights, weights_elems * sizeof(float)),
             "cudaMalloc d_weights");
  workspace.weights_elems = weights_elems;

  // --- cuSPARSE sparse descriptor ---
  // The sparse matrix is N x N, nnz nonzeros, 0-based indexing, row-major CSR.
  // Index type is 32-bit int to match our CSR arrays; value type is fp32.
  check_cusparse(
      cusparseCreateCsr(&workspace.spmat, graph.num_nodes, graph.num_nodes,
                        graph.nnz, workspace.d_csr_row_offsets,
                        workspace.d_csr_col_indices, workspace.d_csr_values,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
      "cusparseCreateCsr");
}

inline void destroy_device_graph(DeviceGCNWorkspace &workspace) {
  /* DONE(student): destroy descriptors and cudaFree buffers. */

  // Destroy cuSPARSE descriptors first (they reference the CSR buffers).
  if (workspace.dn_left) {
    cusparseDestroyDnMat(workspace.dn_left);
    workspace.dn_left = nullptr;
  }
  if (workspace.dn_right) {
    cusparseDestroyDnMat(workspace.dn_right);
    workspace.dn_right = nullptr;
  }
  if (workspace.dn_out) {
    cusparseDestroyDnMat(workspace.dn_out);
    workspace.dn_out = nullptr;
  }
  if (workspace.spmat) {
    cusparseDestroySpMat(workspace.spmat);
    workspace.spmat = nullptr;
  }

  // Free device buffers. cudaFree(nullptr) is a documented no-op, but the
  // explicit checks make intent obvious.
  auto free_if = [](void *&p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_if(reinterpret_cast<void *&>(workspace.d_spmm_workspace));
  free_if(reinterpret_cast<void *&>(workspace.d_weights));
  free_if(reinterpret_cast<void *&>(workspace.d_logits));
  free_if(reinterpret_cast<void *&>(workspace.d_features_out));
  free_if(reinterpret_cast<void *&>(workspace.d_temp));
  free_if(reinterpret_cast<void *&>(workspace.d_features_in));
  free_if(reinterpret_cast<void *&>(workspace.d_csr_values));
  free_if(reinterpret_cast<void *&>(workspace.d_csr_col_indices));
  free_if(reinterpret_cast<void *&>(workspace.d_csr_row_offsets));

  workspace.features_in_elems = 0;
  workspace.activation_elems = 0;
  workspace.logits_elems = 0;
  workspace.weights_elems = 0;
  workspace.spmm_workspace_bytes = 0;
}

// Grow workspace.d_spmm_workspace to at least `needed` bytes, reallocating if
// the current buffer is too small. No-op when the buffer is already big enough.
inline void ensure_spmm_workspace(DeviceGCNWorkspace &workspace,
                                  size_t needed) {
  if (needed <= workspace.spmm_workspace_bytes)
    return;
  if (workspace.d_spmm_workspace) {
    cudaFree(workspace.d_spmm_workspace);
    workspace.d_spmm_workspace = nullptr;
  }
  check_cuda(cudaMalloc(&workspace.d_spmm_workspace, needed),
             "cudaMalloc d_spmm_workspace");
  workspace.spmm_workspace_bytes = needed;
}

inline void run_sparse_dense_mm(cusparseHandle_t handle,
                                DeviceGCNWorkspace &workspace, int rows,
                                int cols, int K, const float *d_input,
                                float *d_output) {
  /* DONE(student): configure cusparseDnMatDescr_t for input/output and call
     cusparseSpMM. rows = num_nodes, cols = hidden_dim, K = feature_dim. */
  (void)K; // SpMM does not have a K dim; the starter comment was a template
           // leftover.

  // cuSPARSE descriptors for X and Y. Create per-call because shapes vary
  // across layers; destroy at the end to avoid leaking on early exit.
  cusparseDnMatDescr_t dn_X = nullptr;
  cusparseDnMatDescr_t dn_Y = nullptr;
  check_cusparse(cusparseCreateDnMat(&dn_X, rows, cols, /*ld=*/cols,
                                     const_cast<float *>(d_input), CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW),
                 "cusparseCreateDnMat X");
  check_cusparse(cusparseCreateDnMat(&dn_Y, rows, cols, /*ld=*/cols, d_output,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW),
                 "cusparseCreateDnMat Y");

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Ask the library how much scratch space it needs, then ensure we have it.
  size_t buf_bytes = 0;
  check_cusparse(
      cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              workspace.spmat, dn_X, &beta, dn_Y, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, &buf_bytes),
      "cusparseSpMM_bufferSize");
  ensure_spmm_workspace(workspace, buf_bytes);

  check_cusparse(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              workspace.spmat, dn_X, &beta, dn_Y, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT,
                              workspace.d_spmm_workspace),
                 "cusparseSpMM");

  cusparseDestroyDnMat(dn_X);
  cusparseDestroyDnMat(dn_Y);
}

inline void run_dense_layer(cublasHandle_t handle, int M, int K, int N,
                            const float *d_input, const float *d_weight,
                            float *d_output) {
  /* DONE(student): call cublasSgemm to compute (M x K) * (K x N). */
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_weight, N, d_input, K, &beta, d_output, N),
               "cublasSgemm");
}

__global__ inline void relu_kernel(float *x, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && x[idx] < 0.0f) {
    x[idx] = 0.0f;
  }
}

inline void apply_activation(float *d_tensor, int elements,
                             cudaStream_t stream) {
  /* DONE(student): implement ReLU or ELU kernel. */
  if (elements <= 0)
    return;
  constexpr int kBlock = 256;
  const int grid = (elements + kBlock - 1) / kBlock;
  relu_kernel<<<grid, kBlock, 0, stream>>>(d_tensor, elements);
}

constexpr int GEMM_RELU_TILE = 16;

__global__ inline void gemm_relu_kernel(int M, int K, int N,
                                        const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C) {
  __shared__ float As[GEMM_RELU_TILE][GEMM_RELU_TILE];
  __shared__ float Bs[GEMM_RELU_TILE][GEMM_RELU_TILE];

  const int row = blockIdx.y * GEMM_RELU_TILE + threadIdx.y; // row in C
  const int col = blockIdx.x * GEMM_RELU_TILE + threadIdx.x; // col in C

  float acc = 0.0f;

  // Loop over K in tiles. Each iteration cooperatively loads a TILE x TILE
  // block of A and a TILE x TILE block of B into shared memory, then does
  // TILE dot-product accumulations.
  const int num_k_tiles = (K + GEMM_RELU_TILE - 1) / GEMM_RELU_TILE;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    // A tile row to load: (row, kt*TILE + threadIdx.x).
    const int a_col = kt * GEMM_RELU_TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

    // B tile: (kt*TILE + threadIdx.y, col).
    const int b_row = kt * GEMM_RELU_TILE + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

// Accumulate TILE partial products from shared memory.
#pragma unroll
    for (int k = 0; k < GEMM_RELU_TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  // Store with ReLU epilogue. This is the fusion: the pre-ReLU value of
  // acc never touches DRAM.
  if (row < M && col < N) {
    C[row * N + col] = acc > 0.0f ? acc : 0.0f;
  }
}

// Row-major GEMM with a fused ReLU epilogue.
// Computes C(M x N) = max(A(M x K) * B(K x N), 0) in one kernel launch.
inline void run_dense_layer_relu(int M, int K, int N, const float *d_input,
                                 const float *d_weight, float *d_output,
                                 cudaStream_t stream = 0) {
  if (M <= 0 || N <= 0 || K <= 0)
    return;
  const dim3 block(GEMM_RELU_TILE, GEMM_RELU_TILE);
  const dim3 grid((N + GEMM_RELU_TILE - 1) / GEMM_RELU_TILE,
                  (M + GEMM_RELU_TILE - 1) / GEMM_RELU_TILE);
  gemm_relu_kernel<<<grid, block, 0, stream>>>(M, K, N, d_input, d_weight,
                                               d_output);
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
