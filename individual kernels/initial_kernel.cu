#include "cuda_runtime.h"
#include "iostream"

__global__ void matrix_multiplication_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * K + y] = tmp;
    }

}