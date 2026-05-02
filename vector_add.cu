#include <stdio.h>
#include <cuda_runtime.h>

/* 
 * Notes for reference : 
 1. 1D Thread Indexing, one thread per element of the vector. 
 2. Bounds check - handle case where N is not divisible by block size. 
 3. Each thread does one independent operation, no communication needed.
 4. Host pattern - cpu allocation (malloc) -> cudaMalloc (GPU allocation) 
      -> cuda memcpy (H2D) -> cuda memcpy (D2H) -> cudaFree.

 The kernel executes the code within the function per thread automatically.     
    CUDA takes that kernel function and runs it on 4×256 = 1024 threads simultaneously. 
    Each thread executes the exact same code but with different values of threadIdx and blockIdx,
    which is why computing idx gives each thread a unique element to work on.
    
    This is the SPMD model — Single Program Multiple Data. Same program, different data per thread.
 */


__global__ void vec_add_kernel(float * A, float * B, float * C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    int N = 1024;
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = (float)i;        // 0, 1, 2, 3...
        B[i] = (float)(N - i);  // 1024, 1023, 1022...
    }

    float *C = (float*)malloc(N * sizeof(float));

    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, N * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));

    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * sizeof(float), cudaMemcpyHostToDevice);

    vec_add_kernel<<<N/256, 256>>>(A_d,B_d,C_d,N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(C, C_d, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    for (int i = 0; i < N; i++) {
        if (fabs(C[i] - N) > 1e-5) {
            printf("WRONG at %d: got %f\n", i, C[i]);
            return 1;
        }
    }
    printf("CORRECT\n");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}
