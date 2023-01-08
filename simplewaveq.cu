#include <stdio.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <valarray>

using namespace std;

#define BLOCK_SIZE 16

__global__ void update(float *u, float *v, int nx, int ny, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        v[i*ny+j] = 2*u[i*ny+j] - v[i*ny+j] + factor *(u[(i+1)*ny+j] - 2*u[i*ny+j] + u[(i-1)*ny+j]) + factor*(u[i*ny+j+1] - 2*u[i*ny+j] + u[i*ny+j-1]);
    }
}

__global__ void swap(float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}


int main(int argc, char **argv) {
    // Initialize grid dimensions and time step
    int nx = 100;
    int ny = 100;
    float dx = 10;
    //float dy = 10;
    float dt = 0.001;
    float vel = 1000;

    // Allocate device arrays
    float *u, *v, *temp;
    cudaMalloc((void **)&u, nx*ny*sizeof(float));
    cudaMalloc((void **)&v, nx*ny*sizeof(float));
    cudaMalloc((void **)&temp, nx*ny*sizeof(float));
    valarray<float> h_v(nx*ny);
    h_v=0;

    h_v[nx/2*ny+ny/2]=1;
    h_v[nx/2*ny+ny/2+1]=-1;
    cudaMemcpy(u, &h_v[0], nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    
    ofstream outputFile("wave.bin", ios::binary);
    if (0){ // test 
        swap<<<(nx*ny)/BLOCK_SIZE, BLOCK_SIZE>>>(u, v, nx*ny);
        cudaMemcpy(&h_v[0], v, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
        outputFile.write((char*) &h_v[0], nx*ny*sizeof(float));
    }

    float factor = vel*vel*dt*dt/dx/dx;
    cout << "factor = " << factor << endl;



    // Iterate over time steps
    for (int t = 0; t < 1000; t++) {
        // Launch kernel
        

        update<<<(nx*ny)/BLOCK_SIZE, BLOCK_SIZE>>>(u, v, nx, ny, factor);
        swap<<<(nx*ny)/BLOCK_SIZE, BLOCK_SIZE>>>(u, v, nx*ny);
        
        if (((t%10)==0)&&(1)){
            cudaMemcpy(&h_v[0], u, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
            outputFile.write((char*) &h_v[0], nx*ny*sizeof(float));
        }
    }
    outputFile.close();
    
    // Free device memory
    cudaFree(u);
    cudaFree(v);
    cudaFree(temp);
    

    return 0;
}
