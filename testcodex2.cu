// write below the command line necessary to compile this code
// nvcc -o testcodex2 testcodex2.cu -arch=sm_86
// CUDA vector addition example using shared memory

#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernelShared(int *c, const int *a, const int *b, int n) {
    extern __shared__ int s[]; // layout: [blockDim.x ints for A][blockDim.x ints for B]
    int *sA = s;
    int *sB = s + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Stage from global to shared
    if (i < n) {
        sA[tid] = a[i];
        sB[tid] = b[i];
    } else {
        // Avoid reading uninitialized shared memory later
        sA[tid] = 0;
        sB[tid] = 0;
    }

    __syncthreads();

    // Compute and write back to global
    if (i < n) {
        c[i] = sA[tid] + sB[tid];
    }
}

int main() {
    const int arraySize = 5;
    int a[arraySize] = {1, 2, 3, 4, 5};
    int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch with multiple blocks and shared memory
    int blockSize = 2;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    size_t shmemBytes = 2 * blockSize * sizeof(int); // A and B tiles
    // Time the kernel execution using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addKernelShared<<<numBlocks, blockSize, shmemBytes>>>(dev_c, dev_a, dev_b, arraySize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time (shared mem): " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Verification
    for (int i = 0; i < arraySize; i++) {
        if (c[i] == a[i] + b[i]) {
            std::cout << "Element " << i << " is correct." << std::endl;
        } else {
            std::cout << "Element " << i << " is incorrect." << std::endl;
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
