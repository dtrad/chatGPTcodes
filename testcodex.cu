// write below the command line necessary to compile this code
// nvcc -o testcodex testcodex.cu -arch=sm_86
// create a cuda example with codex
#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int arraySize = 5;
    int a[arraySize] = {1, 2, 3, 4, 5};
    int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // some elements are incorrect, fix
    
    // use several blocks
    int blockSize = 2;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(dev_c, dev_a, dev_b, arraySize);

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // add comparison
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
