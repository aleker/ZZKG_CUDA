#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>

__global__ void parity(unsigned int *inputMatrix, float *resultMatrix)
{
    int rowLength = blockDim.x;
    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = rowNo * rowLength + columnNo; // global thread id (gtid)
    int bid = threadIdx.x;      // thread id in block (btid)

    int parity = inputMatrix[tid] % 2;
    resultMatrix[tid] = parity;
}

int dimGrid = 32;
int dimBlock = 4;
int n = dimBlock * dimGrid;

int main() {
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    thrust::host_vector<unsigned int> h_a;
    thrust::host_vector<float> h_b;
    thrust::device_vector<unsigned int> d_a;
    thrust::device_vector<float> d_b;

    // host initialization
    for (int j=0; j < n; j++) {
        h_a.push_back(rand() % 20 + 1);
        std::cout << h_a[j] << "\t";
    }
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";

    //Kopiowanie host->device
    d_a = h_a;
    d_b.resize(d_a.size());    // zajmuje i uznaje za wypelnione

    // kernel
    parity << <dimGrid, dimBlock >> > (d_a.data().get(), d_b.data().get());

    //Kopiowanie device->host
    h_b = d_b;

    for (int i = 0; i < n; i++) {
        std::cout << h_b[i] << "\t ";
    }
    std::cout << "\n";

    int d_sum = thrust::reduce(d_b.begin(),d_b.end());
    std::cout << "result = " << d_sum << "\n";

    return 0;
}

