#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
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
    thrust::host_vector<int> h_a;
    thrust::host_vector<int> h_b;
    thrust::device_vector<int> d_a;

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

    thrust::sort(d_a.begin(),d_a.end());

    //Kopiowanie device->host
    h_b = d_a;

    for (int i = 0; i < n; i++) {
        std::cout << h_b[i] << "\t ";
    }
    std::cout << "\n";
    std::cout << h_b[n/2] << "\t ";

    return 0;
}

