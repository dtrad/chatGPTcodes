#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                          \
        cudaError_t err__ = (call);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at " << __FILE__       \
                      << ":" << __LINE__ << std::endl;                                           \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                         \
    } while (0)

struct SimulationGrid {
    int nx{};
    int ny{};
    int nt{};
    float dx{};
    float dt{};
};

struct AcquisitionGeometry {
    std::vector<int> sourceIndices;   // flattened 2D grid positions
    std::vector<int> receiverIndices; // flattened 2D grid positions
};

struct FwiWorkspace {
    SimulationGrid grid{};
    AcquisitionGeometry acquisition{};

    int numSources{0};
    int numReceivers{0};
    size_t gridSize{0};

    float *d_model{nullptr};

    float *d_forwardPrev{nullptr};
    float *d_forwardCur{nullptr};
    float *d_forwardNext{nullptr};
    float *d_forwardHistory{nullptr};

    float *d_sourceWavelet{nullptr}; // [nt * numSources]
    int *d_sourceIndices{nullptr};

    float *d_syntheticData{nullptr}; // [nt * numReceivers]
    float *d_observedData{nullptr};  // [nt * numReceivers]
    float *d_residualData{nullptr};  // [nt * numReceivers]
    int *d_receiverIndices{nullptr};

    float *d_adjointPrev{nullptr};
    float *d_adjointCur{nullptr};
    float *d_adjointNext{nullptr};
    float *d_gradient{nullptr};
};

__host__ __device__ inline int flattenIndex(int ix, int iy, int nx) {
    return iy * nx + ix;
}

__global__ void clearArray(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

__global__ void forwardTimeStep(const float *velocity, const float *prev, const float *cur, float *next,
                                SimulationGrid grid) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid.nx || iy >= grid.ny) {
        return;
    }
    int idx = flattenIndex(ix, iy, grid.nx);
    if (ix == 0 || iy == 0 || ix == grid.nx - 1 || iy == grid.ny - 1) {
        next[idx] = 0.0f;
        return;
    }

    float laplacian = cur[flattenIndex(ix + 1, iy, grid.nx)] + cur[flattenIndex(ix - 1, iy, grid.nx)] +
                      cur[flattenIndex(ix, iy + 1, grid.nx)] + cur[flattenIndex(ix, iy - 1, grid.nx)] -
                      4.0f * cur[idx];
    float vel = velocity[idx];
    float coeff = (vel * grid.dt / grid.dx);
    coeff *= coeff;
    next[idx] = 2.0f * cur[idx] - prev[idx] + coeff * laplacian;
}

__global__ void addSources(float *wavefield, const float *amplitudes, const int *indices, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int target = indices[tid];
        wavefield[target] += amplitudes[tid];
    }
}

__global__ void recordReceivers(const float *wavefield, float *data, const int *indices, int count, int it,
                                int nt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[it * count + tid] = wavefield[indices[tid]];
    }
}

__global__ void computeResiduals(const float *syntheticData, const float *observedData, float *residualData,
                                 int nSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nSamples) {
        residualData[idx] = syntheticData[idx] - observedData[idx];
    }
}

__global__ void adjointTimeStep(const float *velocity, const float *prev, const float *cur, float *next,
                                SimulationGrid grid) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= grid.nx || iy >= grid.ny) {
        return;
    }
    int idx = flattenIndex(ix, iy, grid.nx);
    if (ix == 0 || iy == 0 || ix == grid.nx - 1 || iy == grid.ny - 1) {
        next[idx] = 0.0f;
        return;
    }

    float laplacian = cur[flattenIndex(ix + 1, iy, grid.nx)] + cur[flattenIndex(ix - 1, iy, grid.nx)] +
                      cur[flattenIndex(ix, iy + 1, grid.nx)] + cur[flattenIndex(ix, iy - 1, grid.nx)] -
                      4.0f * cur[idx];
    float vel = velocity[idx];
    float coeff = (vel * grid.dt / grid.dx);
    coeff *= coeff;
    next[idx] = 2.0f * cur[idx] - prev[idx] + coeff * laplacian;
}

__global__ void injectAdjointSources(float *wavefield, const float *residualSlice, const int *indices, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int target = indices[tid];
        wavefield[target] += residualSlice[tid];
    }
}

__global__ void accumulateGradient(const float *forwardState, const float *adjointState, float *gradient, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gradient[idx] += forwardState[idx] * adjointState[idx];
    }
}

std::vector<float> buildRickerWavelet(int nt, float dt, float f0, float t0) {
    std::vector<float> wavelet(nt, 0.0f);
    for (int it = 0; it < nt; ++it) {
        float t = it * dt - t0;
        float arg = static_cast<float>(M_PI) * f0 * t;
        float arg2 = arg * arg;
        wavelet[it] = (1.0f - 2.0f * arg2) * std::exp(-arg2);
    }
    return wavelet;
}

void allocateWorkspace(FwiWorkspace &ws) {
    ws.numSources = static_cast<int>(ws.acquisition.sourceIndices.size());
    ws.numReceivers = static_cast<int>(ws.acquisition.receiverIndices.size());
    ws.gridSize = static_cast<size_t>(ws.grid.nx) * static_cast<size_t>(ws.grid.ny);

    size_t fieldBytes = ws.gridSize * sizeof(float);
    size_t historyBytes = static_cast<size_t>(ws.grid.nt) * fieldBytes;
    size_t shotSamples = static_cast<size_t>(ws.grid.nt) * ws.numSources;
    size_t receiverSamples = static_cast<size_t>(ws.grid.nt) * ws.numReceivers;

    CUDA_CHECK(cudaMalloc(&ws.d_model, fieldBytes));

    CUDA_CHECK(cudaMalloc(&ws.d_forwardPrev, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_forwardCur, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_forwardNext, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_forwardHistory, historyBytes));

    CUDA_CHECK(cudaMalloc(&ws.d_sourceWavelet, shotSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_sourceIndices, ws.numSources * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&ws.d_syntheticData, receiverSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_observedData, receiverSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_residualData, receiverSamples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_receiverIndices, ws.numReceivers * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&ws.d_adjointPrev, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_adjointCur, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_adjointNext, fieldBytes));
    CUDA_CHECK(cudaMalloc(&ws.d_gradient, fieldBytes));
}

void freeWorkspace(FwiWorkspace &ws) {
    cudaFree(ws.d_model);
    cudaFree(ws.d_forwardPrev);
    cudaFree(ws.d_forwardCur);
    cudaFree(ws.d_forwardNext);
    cudaFree(ws.d_forwardHistory);
    cudaFree(ws.d_sourceWavelet);
    cudaFree(ws.d_sourceIndices);
    cudaFree(ws.d_syntheticData);
    cudaFree(ws.d_observedData);
    cudaFree(ws.d_residualData);
    cudaFree(ws.d_receiverIndices);
    cudaFree(ws.d_adjointPrev);
    cudaFree(ws.d_adjointCur);
    cudaFree(ws.d_adjointNext);
    cudaFree(ws.d_gradient);
}

void forwardModeling(FwiWorkspace &ws) {
    dim3 block(16, 16);
    dim3 gridDim((ws.grid.nx + block.x - 1) / block.x, (ws.grid.ny + block.y - 1) / block.y);
    int historyStride = ws.numSources;
    int receiverStride = ws.numReceivers;

    CUDA_CHECK(cudaMemset(ws.d_forwardPrev, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_forwardCur, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_forwardNext, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_forwardHistory, 0, static_cast<size_t>(ws.grid.nt) * ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_syntheticData, 0, static_cast<size_t>(ws.grid.nt) * receiverStride * sizeof(float)));

    for (int it = 0; it < ws.grid.nt; ++it) {
        const float *sourceSlice = ws.d_sourceWavelet + static_cast<size_t>(it) * historyStride;
        forwardTimeStep<<<gridDim, block>>>(ws.d_model, ws.d_forwardPrev, ws.d_forwardCur, ws.d_forwardNext, ws.grid);
        CUDA_CHECK(cudaGetLastError());

        int threads = 256;
        int blocks = (ws.numSources + threads - 1) / threads;
        if (ws.numSources > 0) {
            addSources<<<blocks, threads>>>(ws.d_forwardNext, sourceSlice, ws.d_sourceIndices, ws.numSources);
            CUDA_CHECK(cudaGetLastError());
        }

        blocks = (ws.numReceivers + threads - 1) / threads;
        if (ws.numReceivers > 0) {
            recordReceivers<<<blocks, threads>>>(ws.d_forwardNext, ws.d_syntheticData, ws.d_receiverIndices,
                                                ws.numReceivers, it, ws.grid.nt);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemcpy(ws.d_forwardHistory + static_cast<size_t>(it) * ws.gridSize, ws.d_forwardNext,
                              ws.gridSize * sizeof(float), cudaMemcpyDeviceToDevice));

        std::swap(ws.d_forwardPrev, ws.d_forwardCur);
        std::swap(ws.d_forwardCur, ws.d_forwardNext);
    }
}

void adjointModeling(FwiWorkspace &ws) {
    dim3 block(16, 16);
    dim3 gridDim((ws.grid.nx + block.x - 1) / block.x, (ws.grid.ny + block.y - 1) / block.y);

    CUDA_CHECK(cudaMemset(ws.d_adjointPrev, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_adjointCur, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_adjointNext, 0, ws.gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(ws.d_gradient, 0, ws.gridSize * sizeof(float)));

    int threads = 256;

    for (int it = ws.grid.nt - 1; it >= 0; --it) {
        const float *residualSlice = ws.d_residualData + static_cast<size_t>(it) * ws.numReceivers;
        int blocks = (ws.numReceivers + threads - 1) / threads;
        if (ws.numReceivers > 0) {
            injectAdjointSources<<<blocks, threads>>>(ws.d_adjointCur, residualSlice, ws.d_receiverIndices,
                                                      ws.numReceivers);
            CUDA_CHECK(cudaGetLastError());
        }

        adjointTimeStep<<<gridDim, block>>>(ws.d_model, ws.d_adjointPrev, ws.d_adjointCur, ws.d_adjointNext, ws.grid);
        CUDA_CHECK(cudaGetLastError());

        const float *forwardSlice = ws.d_forwardHistory + static_cast<size_t>(it) * ws.gridSize;
        int gradBlocks = static_cast<int>((ws.gridSize + threads - 1) / threads);
        accumulateGradient<<<gradBlocks, threads>>>(forwardSlice, ws.d_adjointNext, ws.d_gradient, ws.gridSize);
        CUDA_CHECK(cudaGetLastError());

        std::swap(ws.d_adjointPrev, ws.d_adjointCur);
        std::swap(ws.d_adjointCur, ws.d_adjointNext);
    }
}

float evaluateObjective(const FwiWorkspace &ws) {
    std::vector<float> residual(ws.grid.nt * ws.numReceivers);
    CUDA_CHECK(cudaMemcpy(residual.data(), ws.d_residualData,
                          residual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double misfit = 0.0;
    for (float r : residual) {
        misfit += static_cast<double>(r) * static_cast<double>(r);
    }
    return static_cast<float>(0.5 * misfit);
}

void initializeConstantModel(std::vector<float> &model, const SimulationGrid &grid, float velocity) {
    model.assign(static_cast<size_t>(grid.nx) * grid.ny, velocity);
}

void injectVelocityAnomaly(std::vector<float> &model, const SimulationGrid &grid, float velocity, int ix0, int ix1,
                           int iy0, int iy1) {
    for (int iy = iy0; iy < iy1; ++iy) {
        for (int ix = ix0; ix < ix1; ++ix) {
            model[flattenIndex(ix, iy, grid.nx)] = velocity;
        }
    }
}

int main() {
    SimulationGrid grid{};
    grid.nx = 200;
    grid.ny = 200;
    grid.nt = 2000;
    grid.dx = 10.0f;
    grid.dt = 0.001f;

    AcquisitionGeometry acquisition{};
    acquisition.sourceIndices.push_back(flattenIndex(grid.nx / 2, grid.ny / 10, grid.nx));

    for (int ix = grid.nx / 4; ix < 3 * grid.nx / 4; ix += 5) {
        acquisition.receiverIndices.push_back(flattenIndex(ix, grid.ny / 10, grid.nx));
    }

    FwiWorkspace workspace{};
    workspace.grid = grid;
    workspace.acquisition = acquisition;

    allocateWorkspace(workspace);

    std::vector<float> velocityModel;
    initializeConstantModel(velocityModel, grid, 2000.0f);
    injectVelocityAnomaly(velocityModel, grid, 1500.0f, grid.nx / 2 - 10, grid.nx / 2 + 10, grid.ny / 2,
                          grid.ny / 2 + 20);

    CUDA_CHECK(cudaMemcpy(workspace.d_model, velocityModel.data(), workspace.gridSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<float> wavelet = buildRickerWavelet(grid.nt, grid.dt, 10.0f, 1.0f / 10.0f);
    std::vector<float> sourceBuffer(static_cast<size_t>(grid.nt) * workspace.numSources, 0.0f);

    for (int it = 0; it < grid.nt; ++it) {
        for (int is = 0; is < workspace.numSources; ++is) {
            sourceBuffer[static_cast<size_t>(it) * workspace.numSources + is] = wavelet[it];
        }
    }

    CUDA_CHECK(cudaMemcpy(workspace.d_sourceWavelet, sourceBuffer.data(),
                          sourceBuffer.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(workspace.d_sourceIndices, workspace.acquisition.sourceIndices.data(),
                          workspace.numSources * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> observed(static_cast<size_t>(grid.nt) * workspace.numReceivers, 0.0f);
    CUDA_CHECK(cudaMemcpy(workspace.d_observedData, observed.data(), observed.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(workspace.d_receiverIndices, workspace.acquisition.receiverIndices.data(),
                          workspace.numReceivers * sizeof(int), cudaMemcpyHostToDevice));

    forwardModeling(workspace);

    int totalSamples = grid.nt * workspace.numReceivers;
    int threads = 256;
    int blocks = (totalSamples + threads - 1) / threads;
    computeResiduals<<<blocks, threads>>>(workspace.d_syntheticData, workspace.d_observedData,
                                          workspace.d_residualData, totalSamples);
    CUDA_CHECK(cudaGetLastError());

    adjointModeling(workspace);

    float objective = evaluateObjective(workspace);
    std::cout << "Current objective value: " << objective << std::endl;

    std::vector<float> gradient(workspace.gridSize);
    CUDA_CHECK(cudaMemcpy(gradient.data(), workspace.d_gradient, gradient.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::cout << "Gradient sample at grid center: "
              << gradient[flattenIndex(grid.nx / 2, grid.ny / 2, grid.nx)] << std::endl;

    freeWorkspace(workspace);
    return 0;
}
