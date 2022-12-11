#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int nElem = 1024;
    dim3 block(1024);
    dim3 grid((nElem - 1) / block.x + 1);
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x=512;
    grid.x=(nElem-1)/block.x+1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x=256;
    grid.x=(nElem-1)/block.x+1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    block.x=128;
    grid.x=(nElem-1)/block.x+1;
    printf("grid.x %d block.x %d\n",grid.x,block.x);

    cudaDeviceReset();
    return 0;
}