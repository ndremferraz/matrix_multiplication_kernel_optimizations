#include <iostream>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define BLOCKSIZE 1024

__global__ void matrix_multiplication_kernel_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {

    const uint cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x /BLOCKSIZE);
    const uint cColumn = blockIdx.y * BLOCKSIZE + (threadIdx.y % BLOCKSIZE);

    // if statement is necessary to make things work under tile quantization
    if (cRow < M && cColumn < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += A[cRow * K + i] * B[i * N + cColumn];
        }
        C[cRow * K + cColumn] = tmp;
    }

}


__global__ void matrix_multiplication_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * K + y] = tmp;
    }

}

int main() {    
    const int N = 1024;
    const int M = 1024;
    const int K = 1024;
    float *A, *B, *C; 

    cudaMallocManaged(&A, M * N * sizeof(float));
    cudaMallocManaged(&B, N * K * sizeof(float));
    cudaMallocManaged(&C, M * K * sizeof(float));

    //I know this is not going to work as initially inteded for floats, but I just want some random values
    memset(A, 1, sizeof(A));
    memset(B, 2, sizeof(B));
    memset(C, 0, sizeof(C));   
    

    dim3 gridDim(CEIL_DIV(M,32), CEIL_DIV(N,32), 1);
    dim3 blockDim(32,32,1);

    // Launch non-coalesced kernel
    matrix_multiplication_kernel_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();

    memset(C, 0, sizeof(C));

    matrix_multiplication_kernel_coalesced<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();


    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}