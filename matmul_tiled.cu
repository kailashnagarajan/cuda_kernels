#include <stdio.h>
#include <cuda_runtime.h>

#define tile_width 16

/*
  1. Shared memory (__shared__), on-chip memory, ~100x faster than global memory.
  2. Cooperative loading — 256 threads in a block work together to load a 16×16 tile of A
  and a 16×16 tile of B into shared memory. Each thread loads exactly one element from each matrix.
  3. __syncthreads(), barrier synchronization. First sync : ensure all threads finished loading before
  any thread starts computing. Second sync : ensure all threads finished computing before the next tile
  overwrites shared memory.
  4, Tiling over K - Instead of one big dot product over N = 1024 elements from global memory you do
  N/16 = 64 partial dot products of 16 elements each from shared memory.
  5. Each element loaded into shared memory is used 16 elements (once per hthread in the perpendicular dimension).
  Global memory reads become (2/tile width)xN^3. Lower memory bandwidth utilization.
 */

__global__ void matmul_tiled(float * A, float * B, float * C, int N)
{
    __shared__ float As[tile_width][tile_width];
    __shared__ float Bs[tile_width][tile_width];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float c_output = 0;

    for (int m = 0; m < N/tile_width; ++m)
    {
        //each thread loads one element of A & one of B from global memory to smem.
        As[threadIdx.y][threadIdx.x] = A[row * N + (m * tile_width + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m * tile_width + threadIdx.y) * N + col];

        // we wait for all threads in the 16x16 block to finish loading into smem so that it contains two 16x16 tiles
        __syncthreads();

        for (int k = 0; k<tile_width; ++k)
        {
            c_output += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // wait for all threads to finish computing.
    }

    C[row * N + col] = c_output;
}


int main()
{
    int N = 4096;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = 3.0f;       // 0, 1, 2, 3...
        B[i] = 3.0f;  // 1024, 1023, 1022...
    }

    float *C = (float*)malloc(N * N * sizeof(float));

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, N * N * sizeof(float));
    cudaMalloc(&B_d, N * N * sizeof(float));
    cudaMalloc(&C_d, N * N * sizeof(float));

    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // mm_naive<<<N/256, 256>>>(A_d,B_d,C_d,N);
    dim3 blockDim(16,16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    matmul_tiled<<<gridDim, blockDim>>>(A_d, B_d, C_d, N);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(C, C_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // for (int i = 0; i < N; i++) {
    //     if (fabs(C[i] - N) > 1e-5) {
    //         printf("WRONG at %d: got %f\n", i, C[i]);
    //         return 1;
    //     }
    // }
    // printf("CORRECT\n");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}
