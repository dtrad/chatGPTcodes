#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                         \
        cudaError_t err__ = (call);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at " << __FILE__      \
                      << ":" << __LINE__ << std::endl;                                          \
            std::exit(EXIT_FAILURE);                                                              \
        }                                                                                        \
    } while (0)

__global__ void scaleVector(const float *input, float *output, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * factor;
    }
}

int main() {
    const int elementCount = 16;
    const size_t bytes = elementCount * sizeof(float);
    const float scale = 1.5f;

    std::vector<float> hostInput(elementCount);
    std::vector<float> hostOutput(elementCount, 0.0f);

    for (int i = 0; i < elementCount; ++i) {
        hostInput[i] = static_cast<float>(i);
    }

    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));

    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytes, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 128;
    const int blocksPerGrid = (elementCount + threadsPerBlock - 1) / threadsPerBlock;

    scaleVector<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, scale, elementCount);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Scaled vector:" << std::endl;
    for (int i = 0; i < elementCount; ++i) {
        std::cout << "  " << hostInput[i] << " -> " << hostOutput[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));

    return 0;
}
