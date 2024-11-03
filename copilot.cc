// create a matrix multiplication C=AB
// for arbitrary size matrices using CUDA 
#include <iostream>
#include <cuda.h>
__global__ void matrix_mult_cuda(float *A, float *B, float *C, 
                                int m, int n, int p, int q) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    C[i*q+j] = 0;
    for (int k=0; k<n; k++) {
        C[i*q+j] += A[i*n+k]*B[k*q+j];
    }
}

