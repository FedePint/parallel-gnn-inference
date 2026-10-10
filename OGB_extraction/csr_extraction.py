import os
import argparse
import numpy as np
import scipy.sparse as sp
from ogb.nodeproppred import NodePropPredDataset
import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)

def export_ogb_to_csr(dataset_name, out_dir):
    """
    Downloads an OGB dataset and exports it into binary files (CSR format + features)
    that can be efficiently loaded in C++/CUDA.
    """
    print(f"Loading dataset '{dataset_name}'...")
    
    dataset = NodePropPredDataset(name=dataset_name, root=out_dir)
    graph, labels = dataset[0] 
    
    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index'] 
    features = graph['node_feat']    
    
    print("Converting edge index to CSR format...")

    vals = np.ones(edge_index.shape[1], dtype=np.float32)
    

    adj = sp.csr_matrix((vals, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    
  
    indptr = adj.indptr.astype(np.int32)
    indices = adj.indices.astype(np.int32)
    
   
    if features is not None:
        features = features.astype(np.float32)
    
    if labels is not None:

        labels = np.nan_to_num(labels, nan=-1).astype(np.int32).flatten()
    
    print("Extracting split indices...")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].astype(np.int32)
    valid_idx = split_idx['valid'].astype(np.int32)
    test_idx = split_idx['test'].astype(np.int32)
    
    
    save_dir = os.path.join(out_dir, f"{dataset_name}_csr")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving binary files to '{save_dir}'...")
    
    

    indptr.tofile(os.path.join(save_dir, 'indptr.bin'))
    indices.tofile(os.path.join(save_dir, 'indices.bin'))
    
    if features is not None:
        features.tofile(os.path.join(save_dir, 'features.bin'))
    
    if labels is not None:
        labels.tofile(os.path.join(save_dir, 'labels.bin'))
        
    train_idx.tofile(os.path.join(save_dir, 'train_idx.bin'))
    valid_idx.tofile(os.path.join(save_dir, 'valid_idx.bin'))
    test_idx.tofile(os.path.join(save_dir, 'test_idx.bin'))
    
    
    with open(os.path.join(save_dir, 'meta.txt'), 'w') as f:
        f.write(f"num_nodes {num_nodes}\n")
        f.write(f"num_edges {adj.nnz}\n")
        
        num_features = features.shape[1] if features is not None else 0
        f.write(f"num_features {num_features}\n")
        
        try:
            num_classes = dataset.num_classes
            f.write(f"num_classes {num_classes}\n")
        except AttributeError:
            pass 

    print("\n--- Export Summary ---")
    print(f"Nodes:       {num_nodes}")
    print(f"Edges:       {adj.nnz}")
    print(f"Features:    {num_features if features is not None else 0}")
    print("----------------------\n")
    print("C++ Loading Example:")
    print("  FILE* f = fopen(\"indptr.bin\", \"rb\");")
    print("  fread(indptr_array, sizeof(int), num_nodes + 1, f);")
    print("  fclose(f);")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OGB datasets to C++ friendly binary CSR format.")
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Name of the OGB dataset (e.g., ogbn-arxiv, ogbn-products)')
    parser.add_argument('--out_dir', type=str, default='data', help='Directory to save the downloaded data and output binaries')

    args = parser.parse_args()
    export_ogb_to_csr(args.dataset, args.out_dir)
