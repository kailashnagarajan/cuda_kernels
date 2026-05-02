#include <stdio.h>
#include <cuda_runtime.h>

/*
 1. 2D Thread indexing - one thread per output element C[row][col]
 2. 2D grid & block dimensions - maps to matrix structure.
 3. Row-major indexing, A[row][k] = A[row * N + k]
 4. Inner loop over k, each thread computes a full dot product independently.
 5. Problem : Each thread independently loads its entire row of A & column of B from
    global memory -> massive reduntant reads -> memory bound.

Every element of A is read N times (once per output column). Every element of B is read N times (once per output row).
Total reads = 2 x N^3 from slow global memory.
 */

__global__ void mm_naive (float* A, float* B, float* C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for(int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


int main()
{
    int N = 4096;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = 2.0f;       // 0, 1, 2, 3...
        B[i] = 2.0f;  // 1024, 1023, 1022...
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

    mm_naive<<<gridDim, blockDim>>>(A_d, B_d, C_d, N);

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
