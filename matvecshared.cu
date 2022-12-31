#include <cuda_runtime.h>
#include <iostream>
// compile with nvcc -O2 -arch=sm_75 matvecshared.cu -o matvecshared

// Matrix-vector multiplication kernel (very simple example because there is no windowing)
// as an exercise, add the code needed to multiply matrix-vector of arbitrary size

__global__ void matvec_kernel(const float* A, const float* x, float* y, int n) {
  // Load the matrix and vector elements into shared memory
  __shared__ float As[16][16];
  __shared__ float xs[16];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  As[ty][tx] = A[ty * n + tx];
  xs[tx] = x[tx];

  // Multiply the matrix and vector elements
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += As[ty][i] * xs[i];
  }

  // Store the result in global memory
  y[ty] = sum;
}

int main() {
  // Allocate host and device arrays
  const int n = 4;
  float h_A[n][n] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  float h_x[n] = {1, 2, 3, 4};
  float h_y[n];
  float* d_A;
  float* d_x;
  float* d_y;
  cudaMalloc((void**)&d_A, n * n * sizeof(float));
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_y, n * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  dim3 blockSize(4, 4);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
  matvec_kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y, n);

  // Copy the result back to the host
  cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the result
  std::cout << "Result: ";
  for (int i = 0; i < n; i++) {
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;

  // Clean up
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
