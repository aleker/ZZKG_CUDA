#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>

__global__ void parity(char *inputMatrix, char *inputMatrix2, int *resultMatrix, int string_len)
{
    int rowLength = blockDim.x;
    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = rowNo * rowLength + columnNo; // global thread id (gtid)
    int bid = threadIdx.x;      // thread id in block (btid)

    if (bid < string_len) {
        int parity = inputMatrix[tid] == inputMatrix2[tid];
        resultMatrix[tid] = parity;
    }

}

struct myMin: public thrust::binary_function<float,float,float>
{
    __host__ __device__ float operator()(float x, float y) {
        return x<y?x:y;
    }
};

int dimGrid = 32;
int dimBlock = 1;
int n = dimBlock * dimGrid;

int main() {
    thrust::host_vector<char> h_a;
    thrust::host_vector<char> h_a2;

    thrust::device_vector<char> d_input;
    thrust::device_vector<char> d_input2;

    thrust::device_vector<int> d_result;
    thrust::device_vector<int> d_result2;

    // host initialization
    for (int j=0; j < n; j++) {
        h_a.push_back('a');
        h_a2.push_back('a');
        std::cout << h_a[j] << "\t";
    }
    h_a2[6] = 'b';

    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";

    //Kopiowanie host->device
    d_input = h_a;
    d_input2 = h_a2;
    d_result.resize(d_input.size());    // zajmuje i uznaje za wypelnione
    d_result2.resize(d_input.size());

    // kernel
    parity << <dimGrid, dimBlock >> > (d_input.data().get(), d_input2.data().get(), d_result.data().get(), n);
    thrust::inclusive_scan(d_result.begin(),d_result.end(),d_result2.begin(), myMin());

    int d_sum = thrust::reduce(d_result2.begin(),d_result2.end());
    std::cout << "result = " << d_sum << "\n";

    return 0;
}

