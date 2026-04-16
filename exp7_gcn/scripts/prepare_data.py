import os
import argparse
import numpy as np
import urllib.request
import tarfile
import ssl
import pickle

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_and_process_cora(data_dir):
    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    tgz_path = os.path.join(data_dir, "cora.tgz")

    print(f"Downloading raw Cora from {url}...")
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(url, tgz_path)

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    # paths to raw files
    content_path = os.path.join(data_dir, "cora", "cora.content")
    cites_path = os.path.join(data_dir, "cora", "cora.cites")

    print("Parsing raw text files...")

    # 1. Read Content (Features + Labels)
    paper_ids = []
    features = []
    labels_raw = []

    with open(content_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            paper_ids.append(parts[0])
            features.append([float(x) for x in parts[1:-1]])
            labels_raw.append(parts[-1])

    num_nodes = len(paper_ids)
    id_map = {pid: i for i, pid in enumerate(paper_ids)}
    label_map = {l: i for i, l in enumerate(sorted(list(set(labels_raw))))}
    labels = np.array([label_map[l] for l in labels_raw], dtype=np.int32)
    features = np.array(features, dtype=np.float32)

    # 2. Read Cites (Graph)
    edge_src = []
    edge_dst = []

    with open(cites_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            target, source = parts[0], parts[1]
            if target in id_map and source in id_map:
                u, v = id_map[source], id_map[target]
                edge_src.extend([u, v])
                edge_dst.extend([v, u])

    # 3. Build CSR
    edge_src = np.array(edge_src, dtype=np.int32)
    edge_dst = np.array(edge_dst, dtype=np.int32)

    # Add self-loops
    self_loop = np.arange(num_nodes, dtype=np.int32)
    edge_src = np.concatenate([edge_src, self_loop])
    edge_dst = np.concatenate([edge_dst, self_loop])

    # Sort edges by source node
    order = np.lexsort((edge_dst, edge_src))
    edge_src = edge_src[order]
    edge_dst = edge_dst[order]

    row_counts = np.bincount(edge_src, minlength=num_nodes)
    row_offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    col_indices = edge_dst.astype(np.int32)
    nnz = len(col_indices)

    print(f"Graph Info: {num_nodes} nodes, {nnz} edges, {features.shape[1]} features.")

    name = "cora"

    # ==========================================
    # Save two sets of data (C++ version and Python verification version)
    # ==========================================

    # 1. Raw Binary (C++ 用)
    print(f"Saving Raw Binary to {name}.csr/.feat/.label ...")
    with open(os.path.join(data_dir, f"{name}.csr"), 'wb') as f:
        np.array([num_nodes, nnz], dtype=np.int32).tofile(f)
        row_offsets.tofile(f)
        col_indices.tofile(f)
    features.tofile(os.path.join(data_dir, f"{name}.feat"))
    labels.tofile(os.path.join(data_dir, f"{name}.label"))

    # 2. Pickle Format (Python verification script)
    # We use pickle instead of np.savez because np.savez forces a .npz suffix, causing the script to not find the file
    dgl_name = f"{name}_dgl"
    print(f"Saving Pickle Format to {dgl_name}.csr/.feat/.label ...")

    edge_data = np.ones(nnz, dtype=np.float32) # Add missing 'data'

    save_dict = {
        'indptr': row_offsets,
        'indices': col_indices,
        'data': edge_data
    }

    # Save as pickle dictionary, np.load(allow_pickle=True) can read directly
    with open(os.path.join(data_dir, f"{dgl_name}.csr"), 'wb') as f:
        pickle.dump(save_dict, f)

    features.tofile(os.path.join(data_dir, f"{dgl_name}.feat"))
    labels.tofile(os.path.join(data_dir, f"{dgl_name}.label"))

    print("Done! Data prepared correctly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    download_and_process_cora(args.out)
