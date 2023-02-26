#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// compile with nvcc -arch=sm_86 reduceIntegev0.cu -o reduceIntegev0

#define BLOCK_SIZE 256

// Device function to perform reduction on a single block
__global__ void reduce_block(const int *g_idata, int *g_odata, const int n)
{
    extern __shared__ int sdata[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    sdata[tid] = (idx < n) ? g_idata[idx] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[bid] = sdata[0];
    }
}

// Host function to perform reduction on the device
int reduce(const int *d_input, const int n)
{
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    int *d_output, *d_intermediate;
    cudaMalloc((void **)&d_output, num_blocks * sizeof(int));
    cudaMalloc((void **)&d_intermediate, num_blocks * sizeof(int));
    reduce_block<<<num_blocks, threads_per_block, threads_per_block * sizeof(int)>>>(d_input, d_intermediate, n);
    reduce_block<<<1, threads_per_block, threads_per_block * sizeof(int)>>>(d_intermediate, d_output, num_blocks);
    int result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_intermediate);
    return result;
}

// Host main function to test the reduce algorithm
int main()
{
    const int n = 1 << 20;
    int *h_input = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        h_input[i] = i;
    }
    int *d_input;
    cudaMalloc((void **)&d_input, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    int result = reduce(d_input, n);
    printf("The sum of the first %d integers is %d\n", n, result);
    cudaFree(d_input);
    free(h_input);
    return 0;
}
