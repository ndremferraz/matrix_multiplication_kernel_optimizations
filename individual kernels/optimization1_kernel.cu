#include <iostream>
#include <cuda_runtime.h>

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