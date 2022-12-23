#include <cuda_runtime.h>
#include <cstdio>
#include "./cuda_fresh.h"


// cpu实现交错配对归约计算, 递归调用
int recursiveReduce(int *data, const int size) {
    if (size == 1) return data[0];

    const int stride = size / 2;
    if (size % 2 == 1) {
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    }else {
        for (int i = 0; i < stride; ++i) {
            data[i] += data[i + stride];
        }
    }

    return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundry check
    if (tid >= n) return;
    int *idata= g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundry check
    if (tid >= n) return;
    int *idata= g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx> n) return;
    for (int stride = 1; stride < blockDim.x;stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx> n) return;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main() {
    initDevice(0);
    bool bResult = false;

    // initialization
    int size = 1 << 24;
    printf("with array size %d ", size);

    // execution configuration
    int blocksize = 1024;
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid %d block %d \n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *idata_host = (int *)malloc(bytes);
    int *odata_host = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialize the array
    initialData_int(idata_host, size);

    memcpy(tmp, idata_host, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // device memory
    int *idata_dev = nullptr;
    int *odata_dev = nullptr;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, grid.x * sizeof(int)));

    //cpu reduction
    int cpu_sum = 0;
    iStart = cpuSecond();
    // cpu_sum = recursiveReduce(tmp, size);
    for (int i = 0; i < size; i++)
        cpu_sum += tmp[i];
    printf("cpu sum:%d \n", cpu_sum);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    //kernel 1:warmup
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warmup <<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu warmup elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    //kernel 1:reduceNeighbored
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighbored <<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighbored elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    //kernel 2:reduceNeighboredLess
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighboredLess <<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighboredLess elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    //kernel 3:reduceInterleaved
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceInterleaved <<<grid, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceInterleaved elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(idata_host);
    free(odata_host);
    CHECK(cudaFree(idata_dev));
    CHECK(cudaFree(odata_dev));

    //reset device
    cudaDeviceReset();

    //check the results
    if (gpu_sum == cpu_sum) {
        printf("Test success!\n");
    }
    return 0;
}