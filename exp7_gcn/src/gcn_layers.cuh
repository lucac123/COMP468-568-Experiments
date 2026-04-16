#pragma once

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

struct GraphData {
    int num_nodes = 0;
    int num_edges = 0;  // undirected edges counted twice
    int nnz = 0;        // CSR nnz (including self loops)
    int feature_dim = 0;
    int num_classes = 0;

    std::vector<int> h_csr_row_offsets;
    std::vector<int> h_csr_col_indices;
    std::vector<float> h_csr_values;
    std::vector<float> h_features;   // row-major: num_nodes x feature_dim
    std::vector<int> h_labels;       // node labels
};

struct DeviceGCNWorkspace {
    int* d_csr_row_offsets = nullptr;
    int* d_csr_col_indices = nullptr;
    float* d_csr_values = nullptr;

    float* d_features_in = nullptr;
    float* d_features_out = nullptr;
    float* d_weights = nullptr;
    float* d_logits = nullptr;
    float* d_temp = nullptr;

    cusparseSpMatDescr_t spmat = nullptr;
    cusparseDnMatDescr_t dn_left = nullptr;
    cusparseDnMatDescr_t dn_right = nullptr;
    cusparseDnMatDescr_t dn_out = nullptr;
};

inline void check_cusparse(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuSPARSE error");
    }
}

inline void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

inline void build_graph_from_files(const std::string& prefix, GraphData& graph) {
    /* TODO(student):
       1. Read CSR metadata from prefix + ".csr" (e.g., row ptr count, nnz, columns, values).
       2. Read dense features from prefix + ".feat" (float32 rows).
       3. Read labels from prefix + ".label" (int32).
       4. Populate graph struct and add self-loops + normalization coefficients. */
    (void)prefix;
    (void)graph;
}

inline void allocate_device_graph(const GraphData& graph, DeviceGCNWorkspace& workspace) {
    /* TODO(student): cudaMalloc / cudaMemcpy CSR + feature buffers, create cusparse descriptors. */
    (void)graph;
    (void)workspace;
}

inline void destroy_device_graph(DeviceGCNWorkspace& workspace) {
    /* TODO(student): destroy descriptors and cudaFree buffers. */
    (void)workspace;
}

inline void run_sparse_dense_mm(cusparseHandle_t handle,
                                DeviceGCNWorkspace& workspace,
                                int rows,
                                int cols,
                                int K,
                                const float* d_input,
                                float* d_output) {
    /* TODO(student): configure cusparseDnMatDescr_t for input/output and call cusparseSpMM.
       rows = num_nodes, cols = hidden_dim, K = feature_dim. */
    (void)handle;
    (void)workspace;
    (void)rows;
    (void)cols;
    (void)K;
    (void)d_input;
    (void)d_output;
}

inline void run_dense_layer(cublasHandle_t handle,
                            int M,
                            int K,
                            int N,
                            const float* d_input,
                            const float* d_weight,
                            float* d_output) {
    /* TODO(student): call cublasSgemm to compute (M x K) * (K x N). */
    (void)handle;
    (void)M;
    (void)K;
    (void)N;
    (void)d_input;
    (void)d_weight;
    (void)d_output;
}

inline void apply_activation(float* d_tensor, int elements, cudaStream_t stream) {
    /* TODO(student): implement ReLU or ELU kernel. */
    (void)d_tensor;
    (void)elements;
    (void)stream;
}

inline void apply_dropout(float* d_tensor, int elements, float drop_prob, cudaStream_t stream) {
    /* TODO(student): optional – implement dropout. */
    (void)d_tensor;
    (void)elements;
    (void)drop_prob;
    (void)stream;
}

inline void softmax_cross_entropy(const float* d_logits,
                                  const int* d_labels,
                                  int num_nodes,
                                  int num_classes,
                                  float* d_loss) {
    /* TODO(student): compute loss/accuracy or copy logits for host-side evaluation. */
    (void)d_logits;
    (void)d_labels;
    (void)num_nodes;
    (void)num_classes;
    (void)d_loss;
}
