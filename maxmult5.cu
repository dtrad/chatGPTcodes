// nvcc -arch=sm_86 maxmult5.cu -o maxmult5
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(1); \
        } \
    }

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < P) {
        float value = 0.0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

void matrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void cpuMatrixMultiply(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float value = 0.0;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = value;
        }
    }
}

void verifyResult(float *A, float *B, float *C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float value = 0;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * P + j];
            }
            assert(fabs(C[i * P + j] - value) < 1e-2);
        }
    }
}

void benchmark(const char* desc, void (*func)(float*, float*, float*, int, int, int), float* A, float* B, float* C, int M, int N, int P) {
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C, M, N, P);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << desc << ": " << duration.count() << " seconds." << std::endl;
}

int main() {
    // Test 1: Square matrices
    int M1 = 1024, N1 = 1024, P1 = 1024;
    size_t size1 = M1 * N1 * sizeof(float);
    float *A1 = (float *)malloc(size1);
    float *B1 = (float *)malloc(size1);
    float *C1 = (float *)malloc(size1);
    for (int i = 0; i < M1 * N1; ++i) {
        A1[i] = static_cast<float>(rand()) / RAND_MAX;
        B1[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    benchmark("GPU Matrix Multiplication (Square)", matrixMultiply, A1, B1, C1, M1, N1, P1);
    verifyResult(A1, B1, C1, M1, N1, P1);
    benchmark("CPU Matrix Multiplication (Square)", cpuMatrixMultiply, A1, B1, C1, M1, N1, P1);
    free(A1);
    free(B1);
    free(C1);

    // Test 2: Non-square matrices
    int M2 = 512, N2 = 1024, P2 = 768;
    size_t sizeA2 = M2 * N2 * sizeof(float);
    size_t sizeB2 = N2 * P2 * sizeof(float);
    size_t sizeC2 = M2 * P2 * sizeof(float);
    float *A2 = (float *)malloc(sizeA2);
    float *B2 = (float *)malloc(sizeB2);
    float *C2 = (float *)malloc(sizeC2);
    for (int i = 0; i < M2 * N2; ++i) {
        A2[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N2 * P2; ++i) {
        B2[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    benchmark("GPU Matrix Multiplication (Non-Square)", matrixMultiply, A2, B2, C2, M2, N2, P2);
    verifyResult(A2, B2, C2, M2, N2, P2);
    benchmark("CPU Matrix Multiplication (Non-Square)", cpuMatrixMultiply, A2, B2, C2, M2, N2, P2);
    free(A2);
    free(B2);
    free(C2);

    std::cout << "All tests passed successfully." << std::endl;

    return 0;
}
