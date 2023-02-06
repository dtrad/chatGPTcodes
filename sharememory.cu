#include <iostream>
#include <cuda_runtime.h>

__global__ void sharedMemoryKernel(int *d_input, int *d_output)
{
    __shared__ int sharedMemory[64];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    sharedMemory[tid] = d_input[tid];
    __syncthreads();

    for (int i = 1; i < blockSize; i *= 2) {
        int index = 2 * i * tid;
        if (index < blockSize) {
            sharedMemory[index] += sharedMemory[index + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_output[blockIdx.x] = sharedMemory[0];
    }
}

int main()
{
    int h_input[64];
    int h_output[1];

    int *d_input;
    int *d_output;

    cudaMalloc((void**)&d_input, 64 * sizeof(int));
    cudaMalloc((void**)&d_output, 1 * sizeof(int));

    for (int i = 0; i < 64; i++) {
        h_input[i] = i;
    }

    cudaMemcpy(d_input, h_input, 64 * sizeof(int), cudaMemcpyHostToDevice);
    sharedMemoryKernel<<<1, 64>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << h_output[0] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
