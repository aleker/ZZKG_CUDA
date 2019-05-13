#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>

struct ifEquals1ThanSave {
    __host__ __device__ bool operator()(const int x) {
        return x == 1;
    }
};

__global__ void searchForSubstring(unsigned int *inputMatrix, unsigned int *substring, int n, int substringSize, int *positionsMatrix)
{
    int rowLength = blockDim.x;
    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = rowNo * rowLength + columnNo; // global thread id (gtid)
    int bid = threadIdx.x;      // thread id in block (btid)

    if (tid < n) {
        if (n - tid < substringSize) {
            positionsMatrix[tid] = 0;
            return;
        }
        for (int i = 0; i < substringSize; i++) {
            if (inputMatrix[tid + i] != substring[i]) {
                positionsMatrix[tid] = 0;
                return;
            }
        }
    }
    positionsMatrix[tid] = 1;
}

int dimGrid = 32;
int dimBlock = 4;
int n = dimBlock * dimGrid;

int main() {
    std::srand(std::time(nullptr)); // use current time as seed for random generator

    thrust::host_vector<unsigned int> h_a;
    thrust::host_vector<unsigned int> h_substring;
    thrust::host_vector<int> h_positions;

    thrust::device_vector<unsigned int> d_a;
    thrust::device_vector<unsigned int> d_substring;
    thrust::device_vector<int> d_positions;
    thrust::device_vector<int> d_result;


    // host initialization
    for (int j=0; j < n; j++) {
        h_a.push_back(j);
    }
    for (int j=0; j < 5; j++) {
        h_substring.push_back(j);
        h_a[j+4] = j;
        h_a[j+16] = j;
        h_a[j+18] = j;
    }
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";

    //Kopiowanie host->device
    d_a = h_a;
    d_substring = h_substring;
    d_positions.resize(d_a.size());    // zajmuje i uznaje za wypelnione
    d_result.resize(d_a.size());    // zajmuje i uznaje za wypelnione

    // kernel
    searchForSubstring << <dimGrid, dimBlock >> > (d_a.data().get(), d_substring.data().get(), n, h_substring.size(), d_positions.data().get());

    //Kopiowanie device->host
    h_positions = d_positions;

    for (int i = 0; i < n; i++) {
        std::cout << h_positions[i] << "\t ";
    }
    std::cout << "\n";

    // do
    thrust::device_vector<int>::iterator d_b_newend = thrust::copy_if(thrust::make_counting_iterator(0),
                                                                      thrust::make_counting_iterator(n),
                                                                      d_positions.begin(),
                                                                      d_result.begin(),ifEquals1ThanSave());
    //Kopiowanie device->host
    h_positions = d_result;

    for (int i = 0; i < n; i++) {
        std::cout << h_positions[i] << "\t ";
    }
    std::cout << "\n";
    return 0;
}

