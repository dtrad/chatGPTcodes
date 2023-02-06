#include <iostream>
#include <cuda_runtime.h>

__global__ void convolutionKernel(const float *signalA, const float *signalB, float *result, int N, int M)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            int index = tid + i;
            if (index >= N) {
                break;
            }
            sum += signalA[index] * signalB[i];
        }
        result[tid] = sum;
    }
}

void convolution(const float *signalA, const float *signalB, float *result, int N, int M)
{
    float *d_signalA, *d_signalB, *d_result;

    cudaMalloc((void**)&d_signalA, N * sizeof(float));
    cudaMalloc((void**)&d_signalB, M * sizeof(float));
    cudaMalloc((void**)&d_result, N * sizeof(float));

    cudaMemcpy(d_signalA, signalA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signalB, signalB, M * sizeof(float), cudaMemcpyHostToDevice);

    int numThreads = 512;
    int numBlocks = (N + numThreads - 1) / numThreads;

    convolutionKernel<<<numBlocks, numThreads>>>(d_signalA, d_signalB, d_result, N, M);

    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signalA);
    cudaFree(d_signalB);
    cudaFree(d_result);
}

int main()
{
    int N = 1024;
    int M = 256;

    float signalA[N];
    float signalB[M];
    float result[N];

    for (int i = 0; i < N; i++) {
        signalA[i] = (float)i;
    }

    for (int i = 0; i < M; i++) {
        signalB[i] = (float)i;
    }

    convolution(signalA, signalB, result, N, M);

    for (int i = 0; i < N; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
