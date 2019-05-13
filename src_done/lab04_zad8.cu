#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>

struct ifGreaterThanTenThanSaveId{
    __host__ __device__ bool operator()(const int x) {
        return x > 10;
    }
};

int dimGrid = 32;
int dimBlock = 4;
int n = dimBlock * dimGrid;

int main() {
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    thrust::host_vector<int> h_a;
    thrust::device_vector<int> d_a;

    thrust::host_vector<int> h_result;
    thrust::device_vector<int> d_result;

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
    d_result.resize(d_a.size());    // zajmuje i uznaje za wypelnione

    // do
    thrust::device_vector<int>::iterator d_b_newend = thrust::copy_if(thrust::make_counting_iterator(0),
                                                                      thrust::make_counting_iterator(n),d_a.begin(),
            d_result.begin(),ifGreaterThanTenThanSaveId());

    //Kopiowanie device->host
    h_result = d_result;
    for (int i = 0; i < n; i++) {
        std::cout << h_result[i] << "\t ";
    }
    std::cout << "\n";

    // projekt: zbiory czeste
    return 0;
}

