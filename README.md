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

## Profiling Results (RTX 3050, CUDA 13.0, N=4096)

| Metric | Vector Add | Naive Matmul | Tiled Matmul |
|--------|-----------|--------------|--------------|
| Duration | 1.4 μs | 281.13 ms | 212.14 ms |
| Speedup vs Naive | — | 1x | 1.32x |
| Instructions | — | 9.4B | 6.7B (-28%) |
| Memory Throughput | — | 65.27 GB/s | 86.29 GB/s |
| DRAM Throughput | — | 30.02% | 39.69% |
| L1 Hit Rate | — | 87.14% | 0.02% |
| L2 Hit Rate | — | 48.65% | 47.01% |
| Shared Memory/Block | 0 KB | 0 KB | 2.05 KB |
| Shared Mem Loads | — | 0 | 3.22B |
| Achieved Occupancy | — | 99.90% | 99.90% |
| Warp Stall Rate | — | 73% | 74.45% |

**Key Insight: ** In the naive matmul operation, hardware was already taking care of caching the necessary duplicate reads from GPU RAM (87% L1 hit rate). Yet, tiled is faster because there was explicit loading of what was required. L1 cache is fast but sometimes it can unexpectedly evict data when another threads needs that cache. Hence we see the 1.3x speed up. On a bigger GPU with larger matrices where L1 can't hold everything, tiling will give you 5-10x improvement. 

Interesting blog comparing various matmul kernels - https://siboehm.com/articles/22/CUDA-MMM


## Build

`nvcc <kernel>.cu -o <kernel> -arch=sm_86`

## Stack
- GPU: NVIDIA RTX 3050 8GB
- CUDA: 13.0
- OS: Pop!_OS 22.04
- Profiler: Nsight Systems + Nsight Compute
