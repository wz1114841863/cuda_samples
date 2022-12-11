#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_fresh.h"

void sumArray(const float *a, const float *b, float *res, const int size) {
    for (int i = 0; i < size; i += 4) {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void sumArrayGPU(const float *a, const float *b, float *res) {
    auto blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    auto threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    auto M = blockDim.x * blockDim.y * blockDim.z;
    int idx = int(threadId + M * blockId);
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("idx: %d, i: %d", idx, i);
    res[idx] = a[idx] + b[idx];
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1 << 14;
    printf("Vector size: %d \n", nElem);
    int nByte = sizeof(float) * nElem;
    float *a_h = (float *)malloc(nByte);
    float *b_h = (float *)malloc(nByte);
    float *res_h= (float *) malloc(nByte);
    float *res_from_gpu_h = (float *)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float **)&a_d, nByte));
    CHECK(cudaMalloc((float **)&b_d, nByte));
    CHECK(cudaMalloc((float **)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    sumArray(a_h,b_h,res_h,nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem / block.x);
    sumArrayGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration<<<%d, %d>>> \n", grid.x, block.x);
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));

    checkResult(res_h, res_from_gpu_h,nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}

