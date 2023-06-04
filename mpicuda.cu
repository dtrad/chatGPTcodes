#include <iostream>
#include <mpi.h>

// CUDA kernel for vector addition
// compile with nvcc -o mpicuda mpicuda.cu -lcudart -lmpi
// run with mpirun -np 2 mpicuda 

__global__ void vectorAdd(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000;
    const int numElements = N / size;

    int* hostA = new int[N];
    int* hostB = new int[N];
    int* hostC = new int[N];

    // Initialize input vectors on the host
    for (int i = 0; i < N; ++i) {
        hostA[i] = i;
        hostB[i] = i;
    }

    // Allocate device memory
    int* devA, * devB, * devC;
    cudaMalloc((void**)&devA, N * sizeof(int));
    cudaMalloc((void**)&devB, N * sizeof(int));
    cudaMalloc((void**)&devC, N * sizeof(int));

    // Copy input vectors from host to device
    cudaMemcpy(devA, hostA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform vector addition on GPU
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(devA + (rank * numElements), devB + (rank * numElements),
                                       devC + (rank * numElements), numElements);

    // Copy result back to host
    cudaMemcpy(hostC, devC, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result on each process
    std::cout << "Process " << rank << " Result: ";
    for (int i = 0; i < numElements; ++i) {
        std::cout << hostC[rank * numElements + i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;

    MPI_Finalize();
    return 0;
}
