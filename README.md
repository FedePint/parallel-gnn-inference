# Project Q1: Graph Neural Network Inference on GPUs

This project implements a Graph Neural Network (GNN) inference engine (forward pass only) targeting multi-core CPUs (via C++ and `std::thread`) and GPUs (via CUDA). 

The goal of this project is to explore and evaluate different parallelization strategies and memory access patterns (such as Thread-per-Node, Thread-per-Feature, Thread-per-Edge) and their impact on memory throughput and execution time.

## 1. Compilation Instructions

The project uses a `Makefile` to handle compilation. It requires both a C++ compiler supporting C++20 (for `std::atomic_ref`) and the NVIDIA CUDA Toolkit (`nvcc`).

To compile the project, simply open a terminal in the project's root directory and run:
```bash
make
```
This will automatically compile all `.cpp` files with `g++` and `.cu` files with `nvcc`, linking them into a single executable named `gnn_inference`.

To clean the compiled object files and the executable, run:
```bash
make clean
```

## 2. Execution and Parameters

The program is designed to be executed interactively. You don't need to pass command-line arguments. Simply launch the executable:

```bash
./gnn_inference
```

Upon launch, the program will prompt you to select:

1. **Dataset Selection**: You will be asked to select the graph dataset (e.g., `ogbn-arxiv`, `ogbn-products`, or synthetic datasets like `scale-free`, `random`, `small-world`).
2. **Algorithm Selection**: You will be asked to choose the execution strategy:
   - `1`: Sequential CPU
   - `2`: CPU Node Parallelism
   - `3`: CPU Node Parallelism (Chunked)
   - `4`: CPU Edge Parallelism
   - `5`: CUDA (Thread-per-Node)
   - `6`: CUDA (Thread-per-Feature)
   - `7`: CUDA (Thread-per-Feature with FP16 Compression)
   - `8`: CUDA (Thread-per-Edge)

After making your selections, the program will load the binary dataset, generate random weights, run the inference, and finally evaluate the accuracy on the test split. NVTX markers are integrated for profiling with Nsight Systems/Compute.

## 3. Dataset Formats

All datasets must be located in the `data/` directory (e.g., `data/ogbn-arxiv_csr/`).


Each dataset folder must contain the following files:

*   `meta.txt`: A plain text file containing key-value pairs for metadata (e.g., `num_nodes`, `num_features`, `num_classes`, `num_edges`).
*   `features.bin`: A raw binary file containing a flattened 1D array of `float` (32-bit) representing the dense feature matrix of the nodes (size: `num_nodes * num_features`).
*   `indptr.bin`: A raw binary file containing a 1D array of `int` (32-bit). These are the CSR row pointers (size: `num_nodes + 1`).
*   `indices.bin`: A raw binary file containing a 1D array of `int` (32-bit). These are the CSR column indices representing the destination nodes of the edges (size: `num_edges`).
*   `labels.bin`: A raw binary file containing a 1D array of `int` (32-bit) representing the ground-truth class labels for the nodes.
*   `test_idx.bin`: A raw binary file containing a 1D array of `int` (32-bit) representing the indices of the nodes that belong to the test split.

You can generate or convert these files using the provided Python scripts in the `OGB_extraction/` directory.

### Downloading OGB Datasets
To download and convert real-world OGB datasets (e.g., `ogbn-arxiv` or `ogbn-products`) to the required binary CSR format, use `csr_extraction.py`. It requires `ogb` and `torch` to be installed.
```bash
cd OGB_extraction
python csr_extraction.py --dataset ogbn-arxiv --out_dir ../data
```

### Generating Synthetic Datasets
To generate synthetic graphs (e.g., `scale-free`, `random`, `small-world`) for scalability testing, use `generate_synthetic.py`. It requires `networkx`.
```bash
cd OGB_extraction
python generate_synthetic.py --type scale-free --nodes 250000 --features 128 --classes 40 --out_dir ../data
```
