# cuda-kernels

Personal CUDA kernel implementations — written from scratch.

Each kernel includes Nsight Systems profiling data.

## Kernels

### 01 — Vector Add
Element-wise addition of two float arrays.
Covers: 1D thread indexing, basic host-device memory pattern.

### 02 — Naive Matrix Multiply
Each thread computes one output element C[row][col] via a full dot product.
Covers: 2D thread indexing, row-major indexing, global memory bottleneck.

### 03 — Tiled Matrix Multiply
Block-level cooperative loading into shared memory to reduce global memory reads.
Covers: `__shared__memory`, `__syncthreads()`, tiling, data reuse.

## Profiling (RTX 3050, CUDA 13.0)

| Kernel | Time | Memory Throughput | Notes |
|--------|------|-------------------|-------|
| Vector Add | - | - | - |
| Naive Matmul | - | - | - |
| Tiled Matmul | - | - | memory bound → compute bound |

*Numbers to be filled after Nsight profiling*

## Build

\```bash
nvcc <kernel>.cu -o <kernel> -arch=sm_86
\```

## Stack
- GPU: NVIDIA RTX 3050 8GB
- CUDA: 13.0
- OS: Pop!_OS 22.04
- Profiler: Nsight Systems + Nsight Compute
