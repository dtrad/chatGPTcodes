#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16
using namespace std;
void multiply(float* A, const float* x, float* y, int M, int N){
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            y[i] += A[i*N+j] * x[j];
        }
    }
    return;
}

__global__ void matvec_kernelv2(const float* A, const float* x, float* y, int M, int N) {
    // Declare shared memory
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  
    // Load a block of A into shared memory
    int i = threadIdx.x;
    int j = threadIdx.y;
    sA[i][j] = A[i * N + j];
  
    // Wait for all threads to finish loading A
    __syncthreads();
  
    // Compute the matrix-vector product
    float sum = 0;
    for (int k = 0; k < N; k++) {
      sum += sA[i][k] * x[k];
    }
  
    // Store the result in the output array
    y[i] = sum;
  }
  

__global__ void matvec_kernel(float *A, float *x, float *y, int M, int N) {
    // Determine the thread's row and column within the block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory for the block
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float xs[BLOCK_SIZE];

    // Load the element of A and x into shared memory
    As[threadIdx.y][threadIdx.x] = (row < M && col < N) ? A[row * N + col] : 0.0f;
    xs[threadIdx.x] = (col < N) ? x[col] : 0.0f;
    __syncthreads();

    // Perform the dot product of the row of A and x
    float result = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        result += As[threadIdx.y][i] * xs[i];
    }

    // Store the result in the output vector y
    if (row < M) {
        y[row] = result;
    }
}

int main() {
    // Allocate host and device arrays
    const int n = 16;
    float* h_A =(float*) malloc(n*n*sizeof(float));
    float h_x[n];
    float h_y[n];
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) h_A[i*n+j]=i+j;
    for (int i=0;i<n;i++) h_x[i]=i;
    for (int i=0;i<n;i++) h_y[i]=0;
    multiply(h_A, h_x, h_y, n, n);

    // Print the result
    std::cout << "CPU Result: ";
    for (int i = 0; i < n; i++) std::cout << h_y[i] << " ";
    std::cout << std::endl;

    float* d_A=0;
    float* d_x=0;
    float* d_y=0;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
  
    
    // Launch the kernel
    dim3 blockSize(n, n);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    
    matvec_kernel<<< gridSize, blockSize >>>(d_A, d_x, d_y, n, n);
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
     
    // Print the result
    std::cout << "GPU Result: ";
    for (int i = 0; i < n; i++) std::cout << h_y[i] << " ";
    std::cout << std::endl;
  
  
    // Clean up
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
  
    return 0;
  }
  