import os
import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx

def generate_synthetic_dataset(graph_type, num_nodes, num_features, num_classes, out_dir):
    print(f"Generating synthetic graph of type '{graph_type}' with {num_nodes} nodes...")
    
    if graph_type == 'scale-free':
        # Barabási–Albert
        m = 5
        graph = nx.barabasi_albert_graph(num_nodes, m)
    elif graph_type == 'random':
        # Erdős–Rényi
        p = 10.0 / num_nodes
        graph = nx.erdos_renyi_graph(num_nodes, p)
    elif graph_type == 'small-world':
        # Watts–Strogatz
        k = 10
        p = 0.1
        graph = nx.watts_strogatz_graph(num_nodes, k, p)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    print("Converting to CSR format...")
    adj = nx.to_scipy_sparse_array(graph, format='csr')
    
    indptr = adj.indptr.astype(np.int32)
    indices = adj.indices.astype(np.int32)
    
    print("Generating random features and labels...")
    features = np.random.rand(num_nodes, num_features).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_nodes,)).astype(np.int32)
    
    print("Extracting random split indices...")
    all_indices = np.random.permutation(num_nodes)
    train_end = int(0.8 * num_nodes)
    valid_end = int(0.9 * num_nodes)
    
    train_idx = all_indices[:train_end].astype(np.int32)
    valid_idx = all_indices[train_end:valid_end].astype(np.int32)
    test_idx = all_indices[valid_end:].astype(np.int32)
    
    dataset_name = f"synthetic_{graph_type}"
    save_dir = os.path.join(out_dir, f"{dataset_name}_{num_nodes}_csr")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving binary files to '{save_dir}'...")
    
    indptr.tofile(os.path.join(save_dir, 'indptr.bin'))
    indices.tofile(os.path.join(save_dir, 'indices.bin'))
    features.tofile(os.path.join(save_dir, 'features.bin'))
    labels.tofile(os.path.join(save_dir, 'labels.bin'))
    train_idx.tofile(os.path.join(save_dir, 'train_idx.bin'))
    valid_idx.tofile(os.path.join(save_dir, 'valid_idx.bin'))
    test_idx.tofile(os.path.join(save_dir, 'test_idx.bin'))
    
    with open(os.path.join(save_dir, 'meta.txt'), 'w') as f:
        f.write(f"num_nodes {num_nodes}\n")
        f.write(f"num_edges {adj.nnz}\n")
        f.write(f"num_features {num_features}\n")
        f.write(f"num_classes {num_classes}\n")

    print("\n--- Export Summary ---")
    print(f"Dataset:     {dataset_name}")
    print(f"Nodes:       {num_nodes}")
    print(f"Edges:       {adj.nnz}")
    print(f"Features:    {num_features}")
    print("----------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['scale-free', 'random', 'small-world'], default='scale-free')
    parser.add_argument('--nodes', type=int, default=250000)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--classes', type=int, default=40)
    parser.add_argument('--out_dir', type=str, default='data')

    args = parser.parse_args()
    generate_synthetic_dataset(args.type, args.nodes, args.features, args.classes, args.out_dir)

