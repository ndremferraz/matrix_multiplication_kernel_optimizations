# Matrix Multiplication Kernel Optimizations with CUDA(Work in Progress🚦)
### The Matrix Multiplication is one of the most fundamental computations needed for Machine Learning. So, inspired by [Siboehm](https://siboehm.com/articles/22/CUDA-MMM), I decided put some of my hardware and parallel processors knowledge to work and attempt some optimizations using CUDA

I started off with the following matrix multiplication kernel:
``` c
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
```
The kernel takes in three matrices A, B, and C, of dimensions MxN, NxK, and MxK, and performs AxB = C.
And the idea was to optimize this kernel and benchmark the performance improvements! 

1. The first Optimized kernel reanrranged the Thread executions to make use of memory coalescing
