#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_fresh.h"

__global__ void printThreadIndex(const float *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    printf("thread_id:(%d, %d), block_id: (%d, %d), coordinate: (%d, %d), global index %2d ival %2f \n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main() {
    initDevice(0);
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // Malloc
    auto *A_host = (float *)malloc(nBytes);
    initialData(A_host, nxy);
    printMatrix(A_host, nx, ny);

    // Cuda Malloc
    float *A_dev = nullptr;
    CHECK(cudaMalloc((void **)&A_dev, nBytes));

    cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    printThreadIndex<<<grid, block>>>(A_dev, nx, ny);

    CHECK(cudaDeviceSynchronize());
    cudaFree(A_dev);
    free(A_host);

    cudaDeviceReset();
    return 0;
}