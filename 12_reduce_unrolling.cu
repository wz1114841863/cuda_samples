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

__global__ void reduceUnroll2(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return ;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 2;
    if (idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll4(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return ;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 4;
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x * 1];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll8(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return ;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x * 1];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return ;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (idx + blockDim.x * 7 < n) {
        g_idata[idx] += g_idata[idx + blockDim.x * 1];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid < 32) {
        // volatile int类型变量是控制变量结果写回到内存，而不是存在共享内存，或者缓存中
        // 因为下一步的计算马上要用到它，如果写入缓存，可能造成下一步的读取会读到错误的数据
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];

    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (idx + blockDim.x * 7< n) {
        g_idata[idx] += g_idata[idx + blockDim.x * 1];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid]+=idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid]+=idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid]+=idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid]+=idata[tid + 64];
    __syncthreads();

    //write result for this block to global mem
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];

}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary chexk
    if (tid >= n)
        return;
    // convert global data pointer
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;
    if (idx + blockDim.x * 7< n) {
        g_idata[idx] += g_idata[idx + blockDim.x * 1];
        g_idata[idx] += g_idata[idx + blockDim.x * 2];
        g_idata[idx] += g_idata[idx + blockDim.x * 3];
        g_idata[idx] += g_idata[idx + blockDim.x * 4];
        g_idata[idx] += g_idata[idx + blockDim.x * 5];
        g_idata[idx] += g_idata[idx + blockDim.x * 6];
        g_idata[idx] += g_idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    if (iBlockSize >= 1024 && tid < 512)
        idata[tid]+=idata[tid + 512];
    __syncthreads();
    if (iBlockSize >= 512 && tid < 256)
        idata[tid]+=idata[tid + 256];
    __syncthreads();
    if (iBlockSize >= 256 && tid < 128)
        idata[tid]+=idata[tid + 128];
    __syncthreads();
    if (iBlockSize >= 128 && tid < 64)
        idata[tid]+=idata[tid + 64];
    __syncthreads();

    //write result for this block to global mem
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
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
    int *idata_host = (int *) malloc(bytes);
    int *odata_host = (int *) malloc(grid.x * sizeof(int));
    int *tmp = (int *) malloc(bytes);

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

    // kernel 0: warmup
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warmup <<<grid.x / 2, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("gpu warmup elapsed %lf ms \n", iElaps);

    // kernel 1: reduceUnrolling2
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll2<<<grid.x / 2, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling2 elapsed %lf ms gpu_sum: %d<<<%d, %d>>> \n",
           iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 1.1: reduceUnrolling4
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll4<<<grid.x / 4, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling4 elapsed %lf ms gpu_sum: %d<<<%d, %d>>> \n",
           iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 1.2: reduceUnrolling8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll8<<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling2 elapsed %lf ms gpu_sum: %d<<<%d, %d>>> \n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 2: reduceUnrollingWarp8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrollWarp8<<<grid.x / 8, block >>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrollingWarp8 elapsed %lf ms gpu_sum: %d<<<%d, %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 3: reduceCompleteUnrollWarp8
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceCompleteUnrollWarp8 <<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceCompleteUnrollWarp8 elapsed %lf ms gpu_sum: %d<<<%d, %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 4: reduceCompleteUnroll
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    switch(blocksize) {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x / 8, block>>>(idata_dev, odata_dev, size);
            break;
    }
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += odata_host[i];
    printf("reduceCompleteUnroll elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 8, block.x);

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
    return EXIT_SUCCESS;
}

