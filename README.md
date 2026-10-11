# High-Performance Parallel GNN Inference (CPU & CUDA)

[![Language](https://img.shields.io/badge/Language-C%2B%2B20%20%7C%20CUDA%20C%2B%2B-blue.svg)](#prerequisites)
[![Platform](https://img.shields.io/badge/Platform-Linux-orange.svg)](#prerequisites)
[![CUDA Toolkit](https://img.shields.io/badge/CUDA-11.0%2B%20%7C%2012.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Documentation](https://img.shields.io/badge/Documentation-PDF%20Available-red.svg)](docs/DOCUMENTATION.pdf)

An optimized forward-pass inference engine for **Graph Neural Networks (GNNs)** targeting multi-core CPUs and NVIDIA GPUs. This repository investigates different parallelization paradigms, memory access patterns, and precision optimizations (FP32 vs FP16) to evaluate their direct impact on memory bandwidth, cache utilization, synchronization overhead, and execution latency.

---

## Table of Contents

- [Overview](#overview)
- [Project Documentation](#project-documentation)
- [Parallelization Strategies](#parallelization-strategies)
- [Repository Structure](#repository-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Quick Start](#quick-start)
  - [1. Python Dependencies](#1-python-dependencies)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Compilation](#3-compilation)
  - [4. Running Inference](#4-running-inference)
- [Dataset Formats](#dataset-formats)
- [Profiling with NVIDIA Nsight](#profiling-with-nvidia-nsight)
- [License & Authors](#license--authors)

---

## Overview

Graph Neural Network (GNN) inference typically alternates between two phases:
1. **Neighborhood Aggregation (Message Passing)**: Sparse and irregular memory access over graph topologies represented in Compressed Sparse Row (CSR) format.
2. **Feature Transformation (Dense Linear Layer)**: Computationally intensive matrix multiplications followed by activation functions (e.g., ReLU) and classification logits.

This project implements and benchmarks multiple parallel execution models for both CPU and GPU to understand performance bottlenecks:
- Multi-threading scalability and chunk-based scheduling on multi-core CPUs.
- Thread-level and block-level parallelism on GPUs.
- Coalesced memory accesses and shared memory utilization.
- Memory bandwidth reduction through **FP16 half-precision compression**.
- Atomic contention during edge-centric updates.

---

## Project Documentation

A comprehensive technical report covering theoretical background, algorithmic design, memory access analysis, profiling insights, and benchmark evaluations is available in the repository:

**[Read the Full Documentation (PDF)](docs/DOCUMENTATION.pdf)**

The documentation details:
- Theoretical complexity of vertex-centric vs. edge-centric message passing.
- Architectural bottlenecks (memory latency, bandwidth saturation, atomic hazards).
- Roofline model considerations and NVTX profiling results.
- Performance scaling on both real-world graphs (e.g., OGB) and synthetic graphs (Scale-Free, Small-World, Random).

---

## Parallelization Strategies

The engine supports 8 different inference algorithms selectable at runtime:

| ID | Mode | Algorithm Strategy | Key Characteristics |
|:---:|:---:|:---|:---|
| **1** | **CPU** | Sequential Baseline | Single-threaded reference implementation. |
| **2** | **CPU** | Node Parallelism | Multi-threaded node distribution using a custom `ThreadPool`. |
| **3** | **CPU** | Chunked Node Parallelism | Nodes divided into contiguous chunks to reduce task-dispatching overhead. |
| **4** | **CPU** | Edge Parallelism | Edges processed in parallel with atomic destination feature updates via `std::atomic_ref` (C++20). |
| **5** | **GPU** | CUDA: Thread-per-Node | 1 CUDA thread per graph vertex; sequential neighbor traversal within thread. |
| **6** | **GPU** | CUDA: Thread-per-Feature | 2D thread grid with shared memory cooperative caching; maximizes memory coalescing. |
| **7** | **GPU** | CUDA: Thread-per-Feature (FP16) | Uses `__half` precision for node features, halving global memory bandwidth pressure. |
| **8** | **GPU** | CUDA: Thread-per-Edge | 1 CUDA thread per directed edge; destination feature aggregation via `atomicAdd`. |

---

## Repository Structure

```text
parallel-gnn-inference/
├── docs/
│   └── DOCUMENTATION.pdf          # Full technical report & experimental analysis
├── OGB_extraction/
│   ├── csr_extraction.py          # Script to download & convert OGB datasets to binary CSR
│   ├── generate_synthetic.py      # Script to generate synthetic graph topologies (NetworkX)
│   ├── bin_conversion.h/.cpp      # Binary file I/O utilities for CSR, features, and labels
├── data/                          # Target directory for CSR binary datasets (git-ignored)
├── thread_per_node.cu             # CUDA Kernel: 1 thread per node
├── thread_per_feature.cu          # CUDA Kernel: 1 thread per feature with shared memory
├── thread_per_feature_compressed.cu # CUDA Kernel: FP16 half-precision feature compression
├── thread_per_edge.cu             # CUDA Kernel: 1 thread per edge with atomicAdd
├── node_parallelism.cpp           # CPU ThreadPool node-parallel implementation
├── node_parallelism_chunk.cpp     # CPU ThreadPool chunked node-parallel implementation
├── edge_parallelism_chunk.cpp     # CPU edge-parallel implementation with atomic aggregation
├── sequential.cpp                 # CPU single-threaded reference implementation
├── random_weights_matrix.h/.cpp   # Random weight initialization utilities
├── globals.h                      # Shared data structures and kernel interfaces
├── main.cpp                       # Interactive CLI entry point and timing/accuracy harness
├── Makefile                       # Build configuration for GCC and NVCC
├── requirements.txt               # Python dependencies for dataset extraction
└── README.md                      # Project documentation
```

---

## Prerequisites & Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04 / 22.04 LTS).
- **C++ Compiler**: `g++` supporting **C++20** (for `std::atomic_ref` and `<thread>`).
- **NVIDIA CUDA Toolkit**: `nvcc` with CUDA 11.0 or higher.
- **NVIDIA NVTX Library**: `libnvToolsExt` (included in CUDA Toolkit for timeline profiling).
- **Python**: Version 3.8+ (for dataset downloading and synthetic generation).

---

## Quick Start

### 1. Python Dependencies

Install the necessary Python packages for dataset preparation:

```bash
pip install -r requirements.txt
```

*(Includes `torch`, `ogb`, and `networkx`)*.

### 2. Dataset Preparation

Ensure the `data/` directory exists:

```bash
mkdir -p data
```

#### Option A: Real-World Datasets (OGB)
Download and convert Open Graph Benchmark datasets into the binary CSR format:

```bash
cd OGB_extraction
python csr_extraction.py --dataset ogbn-arxiv --out_dir ../data
cd ..
```

#### Option B: Synthetic Datasets
Generate synthetic graphs for scalability and topological benchmarks:

```bash
cd OGB_extraction
# Generate Scale-Free graph (Barabási-Albert)
python generate_synthetic.py --type scale-free --nodes 250000 --features 128 --classes 40 --out_dir ../data

# Generate Small-World graph (Watts-Strogatz)
python generate_synthetic.py --type small-world --nodes 100000 --features 128 --classes 40 --out_dir ../data

# Generate Random graph (Erdős-Rényi)
python generate_synthetic.py --type random --nodes 100000 --features 128 --classes 40 --out_dir ../data
cd ..
```

### 3. Compilation

Compile both CPU and CUDA implementations with optimal optimization flags (`-O3`):

```bash
make
```

To clean compiled object files and binary:

```bash
make clean
```

### 4. Running Inference

Launch the interactive CLI:

```bash
./gnn_inference
```

Follow the interactive menu prompts:
1. **Choose Dataset**: Select from available real or synthetic graph datasets.
2. **Choose Algorithm**: Select one of the 8 execution strategies (CPU Sequential, CPU Node/Edge Parallel, or CUDA kernels).

The program will:
- Load the binary graph into memory (CSR indices, row pointers, features, labels).
- Initialize weight matrices for GNN layers.
- Execute forward-pass inference.
- Print detailed timings (`[Timer]` checkpoints).
- Evaluate test accuracy (%) on the dataset split.

---

## Dataset Formats

All datasets must be located in subdirectories within `data/` (e.g., `data/ogbn-arxiv_csr/`). Each dataset folder contains:

| File | Type | Description | Size / Dimensions |
|:---|:---:|:---|:---|
| `meta.txt` | ASCII | Metadata key-value pairs (`num_nodes`, `num_features`, `num_classes`, `num_edges`) | Single text file |
| `features.bin` | Float32 | Raw dense node feature matrix | `num_nodes * num_features * sizeof(float)` |
| `indptr.bin` | Int32 | CSR row pointers array | `(num_nodes + 1) * sizeof(int)` |
| `indices.bin` | Int32 | CSR column indices (destination nodes) | `num_edges * sizeof(int)` |
| `labels.bin` | Int32 | Ground truth node class labels | `num_nodes * sizeof(int)` |
| `test_idx.bin` | Int32 | Node indices belonging to the test evaluation split | Variable (`num_test_nodes * sizeof(int)`) |

---

## Profiling with NVIDIA Nsight

All inference engines are instrumented with **NVTX (NVIDIA Tools Extension)** markers (`nvtxRangePushA` / `nvtxRangePop`).

### Profiling with Nsight Systems (`nsys`)

Capture a complete timeline trace including CPU threads, CUDA kernels, memory transfers, and NVTX ranges:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=gnn_profile --force-overwrite=true ./gnn_inference
```

Open the resulting `gnn_profile.nsys-rep` file in **NVIDIA Nsight Systems GUI** to inspect:
- Memory copy vs. kernel execution overlaps.
- Warp occupancy and SM active cycles.
- Differences in memory throughput between FP32 and FP16 kernels.

### Profiling with Nsight Compute (`ncu`)

Analyze specific kernel metrics (e.g., roofline analysis, cache hit rates, warp divergence):

```bash
ncu --set full -o gnn_kernel_analysis ./gnn_inference
```

---

## License & Authors

Developed as part of the parallel computing & GPU architecture exploration project for Graph Neural Networks.

For detailed methodological explanations, benchmarks, and experimental graphs, please refer to [`docs/DOCUMENTATION.pdf`](docs/DOCUMENTATION.pdf).
