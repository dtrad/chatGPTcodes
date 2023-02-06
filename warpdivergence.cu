#include <iostream>
#include <cuda_runtime.h>

__global__ void warpDivergenceKernel(int *d_input, int *d_output)
{
    int tid = threadIdx.x;

    if (tid < 32) {
        if (tid % 2 == 0) {
            d_output[tid] = d_input[tid] * 2;
        }
        else {
            d_output[tid] = d_input[tid] + 2;
        }
    }
}

int main()
{
    int h_input[32];
    int h_output[32];

    int *d_input;
    int *d_output;

    cudaMalloc((void**)&d_input, 32 * sizeof(int));
    cudaMalloc((void**)&d_output, 32 * sizeof(int));

    for (int i = 0; i < 32; i++) {
        h_input[i] = i;
    }

    cudaMemcpy(d_input, h_input, 32 * sizeof(int), cudaMemcpyHostToDevice);
    warpDivergenceKernel<<<1, 32>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
