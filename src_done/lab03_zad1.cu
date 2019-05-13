#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <cmath>
#include <iostream>     // cout

#define checkCuda(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline cudaError_t cudaAssert(cudaError_t result, const char *file, int line, bool abort = true) {
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Error: \"%s\" in %s:%d\n", cudaGetErrorString(result), file, line);
        if (abort) {
            exit(result);
        }
    }
    return result;
}

int vectorSize = 128;
const int blocksize = vectorSize;

__global__ void rewrite(int *inputVector, int *resultVector, int n) {
    extern __shared__ float shared[];

    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = rowNo * n + columnNo;

    // write to shared
    if (tid < n) {
        shared[tid] = inputVector[tid];
        __syncthreads();
    }

    // write result
    if (tid < n && tid > 0) {
        resultVector[tid] = shared[tid - 1];
    } else if (tid == 0) {
        resultVector[tid] = shared[tid];
    }
    __syncthreads();
}

int main() {
    thrust::host_vector<int> h_tab;
    thrust::device_vector<int> d_tab;

    thrust::host_vector<int> h_tab_result;
    thrust::device_vector<int> d_tab_result;

    // host initialization
    for (int i=0; i < vectorSize; i++) {
        h_tab.push_back(i);
    }

    //Kopiowanie host->device
    d_tab = h_tab;
    d_tab_result.resize(vectorSize);    // zajmuje i uznaje za wypelnione

    // jeden blok z vectorSize watkami
    rewrite << <1, blocksize, vectorSize * sizeof(int)>> > (d_tab.data().get(), d_tab_result.data().get(), d_tab.size());

    //Kopiowanie device->host
    h_tab_result = d_tab_result;

    for (int i = 0; i < vectorSize; i++) {
        std::cout << h_tab_result[i] << " ";
    }

    return 0;
}

