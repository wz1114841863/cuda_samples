#include <cstdio>

__global__ void hello_world() {
    printf("GPU: Hello world! \n");
}

int main() {
    printf("CPU: Hello world! \n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}