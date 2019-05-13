#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>     // cout
#include <algorithm>

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

const int blocksize = 32;
const int blockCount = 2;
const int n = blocksize * blockCount;    // vector size
const int sharedArraySize = n;


__global__ void sum_front(int *inputVector, float *resultVector, int inputFullSize) {
    extern __shared__ float shared[];

    // 1 block per row
    // (blocksize x blockCount matrix)
    // inputFullSize = blocksize * blockCount
    int rowLength = blockDim.x;
    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = rowNo * rowLength + columnNo; // global thread id (gtid)
    int bid = threadIdx.x;      // thread id in block (btid)

    // write to shared
    // bid == tid
    if (tid < inputFullSize) {
        shared[bid] = inputVector[tid];
    }
    __syncthreads();

    // write result
    if (tid < inputFullSize) {
        int sum = 0;
        for (int i = 0; i <= bid; i++) {
            sum += shared[i];
        }
        resultVector[tid] = sum;
    }
    __syncthreads();
}

int main() {
    thrust::host_vector<int> h_tab;
    thrust::device_vector<int> d_tab;

    thrust::host_vector<float> h_tab_result;
    thrust::device_vector<float> d_tab_result;

    // host initialization
    for (int i=0; i < n; i++) {
        h_tab.push_back(2*i);
        std::cout << h_tab[i] << " ";
    }
    std::cout << "\n";

    //Kopiowanie host->device
    d_tab = h_tab;
    d_tab_result.resize(n);    // zajmuje i uznaje za wypelnione

    sum_front << <blockCount, blocksize, sharedArraySize * blockCount * sizeof(int)>> > (d_tab.data().get(), d_tab_result.data().get(), d_tab.size());

    //Kopiowanie device->host
    h_tab_result = d_tab_result;

    for (int i = 0; i < n; i++) {
        std::cout << h_tab_result[i] << " ";
    }

    return 0;
}

