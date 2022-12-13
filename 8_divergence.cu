#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "cuda_fresh.h"

__global__ void warmup(float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0, b = 0.0;

    if ((idx / warpSize) % 2 == 0) {
        a = 100.0f;
    }else {
        b = 200.0f;
    }

    c[idx] = a + b;
}

__global__ void mathKernel1(float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0, b = 0.0;

    if (idx % 2 == 0) {
        a = 100.0f;
    }else {
        b = 200.0f;
    }

    c[idx] = a + b;
}

__global__ void mathKernel2(float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0, b = 0.0;

    if ((idx / warpSize) % 2 == 0) {
        a = 100.0f;
    }else {
        b = 200.0f;
    }
    c[idx] = a + b;
}

__global__ void mathKernel3(float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0, b = 0.0;
    bool ipred= (idx % 2 == 0);
    if (ipred) {
        a = 100.0f;
    }else {
        b = 200.0f;
    }
    c[idx] = a + b;
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
   std::cout << "using Device " << dev << deviceProp.name << std::endl;

    // set up data size
    int size = 64;
    int blocksize = 64;
    std::cout << "Data size " << size << std::endl;

    // set up ececution configuarion
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    std::cout << "Execution configure: block=" << block.x << " grid=" << grid.x << std::endl;

    // allocate gpu memory
    float *C_dev;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float **)&C_dev, nBytes);
    float *C_host = (float *)malloc(nBytes);

    // run a warmmup kernel to remove overhead
    double iStart = 0, iElaps = 0;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmup<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    std::cout << "warmup: " << iElaps << std::endl;

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    std::cout << "mathKernel1: " << iElaps << std::endl;
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost);

    // run kernel2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    std::cout << "mathKernel2: " << iElaps << std::endl;
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost);

    // run kernel3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    std::cout << "mathKernel3: " << iElaps << std::endl;
    cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost);

    free(C_host);
    cudaFree(C_dev);
    cudaDeviceReset();
    return 0;
}
