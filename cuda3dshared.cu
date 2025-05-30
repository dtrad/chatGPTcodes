#include <cuda_runtime.h>
#include <iostream>

// Define cube dimensions
#define NX 64 // X-dimension
#define NY 64 // Y-dimension
#define NZ 64 // Z-dimension

// CUDA kernel to access a 3D cube using shared memory
__global__ void access3DCubeShared(float *cube, int nx, int ny, int nz) {
    // Shared memory allocation for a tile of the 3D cube
    extern __shared__ float sharedCube[];

    // Calculate 3D indices from thread and block IDs
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Local thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Linear index for shared memory
    int localIdx = tz * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;

    // Flatten global index into 1D index for linear memory access
    if (x < nx && y < ny && z < nz) {
        int globalIdx = z * (nx * ny) + y * nx + x;

        // Load data from global memory to shared memory
        sharedCube[localIdx] = cube[globalIdx];
        __syncthreads();

        // Example: Perform computation on shared memory
        sharedCube[localIdx] += 1.0f; // Modify as needed

        __syncthreads();

        // Write back the result to global memory
        cube[globalIdx] = sharedCube[localIdx];
    }
}

int main() {
    // Allocate and initialize 3D cube on host
    size_t size = NX * NY * NZ * sizeof(float);
    float *h_cube = (float *)malloc(size);
    
    // Initialize host data
    for (int i = 0; i < NX * NY * NZ; i++) {
        h_cube[i] = static_cast<float>(i);
    }

    // Allocate memory on device
    float *d_cube;
    cudaMalloc((void **)&d_cube, size);

    // Copy data from host to device
    cudaMemcpy(d_cube, h_cube, size, cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(8, 8, 8); // Threads per block (8x8x8 = 512 threads per block)
    dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, 
                 (NY + blockDim.y - 1) / blockDim.y,
                 (NZ + blockDim.z - 1) / blockDim.z); // Blocks in grid

    // Launch kernel with shared memory size
    size_t sharedMemSize = blockDim.x * blockDim.y * blockDim.z * sizeof(float);
    access3DCubeShared<<<gridDim, blockDim, sharedMemSize>>>(d_cube, NX, NY, NZ);

    // Copy data back to host
    cudaMemcpy(h_cube, d_cube, size, cudaMemcpyDeviceToHost);

    // Print some values for verification
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                int idx = z * (NX * NY) + y * NX + x;
                std::cout << "cube[" << x << ", " << y << ", " << z << "] = " << h_cube[idx] << std::endl;
            }
        }
    }

    // Free memory
    free(h_cube);
    cudaFree(d_cube);

    return 0;
}
