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

int d = 5;      // window size
const int blocksize = 64;
const int blockCount = 5;
const int n = blocksize * blockCount;    // vector size
const int sharedArraySize = blocksize + d - 1;

__device__ int maxValue(int value, int minValue) {
    if (minValue > value) return minValue;
    else return value;
}

__device__ int minValue(int value, int shared_size) {
    if (shared_size > value) return value;
    else return shared_size;
}


__global__ void compute_avg(int *inputVector, float *resultVector, int inputFullSize, int d) {
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
    if (tid < inputFullSize) {
        shared[bid] = inputVector[maxValue(tid - ((d - 1) / 2), 0)];
    }
    if (bid < d - 1) {
    // pierwsze watki uzupelniaja ostatnie brakujace
        shared[blockDim.x + bid] = inputVector[minValue(bid + tid - (d - 1) / 2, bid + d - 1)];
    }
    __syncthreads();


    // write result
    if (tid < inputFullSize) {
        float sum = 0.0f;
        for (int i = bid; i < (bid + d); i++) {
            sum += float (shared[i]);
        }
        resultVector[tid] = sum / float (d);
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
        if (i % d == 0) h_tab.push_back(2);
        else h_tab.push_back(2);
        std::cout << h_tab[i] << " ";
    }
    std::cout << "\n";

    //Kopiowanie host->device
    d_tab = h_tab;
    d_tab_result.resize(n);    // zajmuje i uznaje za wypelnione

    compute_avg << <blockCount, blocksize, sharedArraySize * blockCount * sizeof(int)>> > (d_tab.data().get(), d_tab_result.data().get(), d_tab.size(), d);

    //Kopiowanie device->host
    h_tab_result = d_tab_result;

    for (int i = 0; i < n; i++) {
        std::cout << h_tab_result[i] << " ";
    }

    return 0;
}

