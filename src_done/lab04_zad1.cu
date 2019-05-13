#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>



struct myMax: public thrust::binary_function<float,float,float>
{
    __host__ __device__ float operator()(float x, float y) {
        return x<y?y:x;
    }
};

int blocksize = 32;

int main() {
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    thrust::host_vector<unsigned int> h_a;
    thrust::host_vector<float> h_b;
    thrust::device_vector<float> d_a;
    thrust::device_vector<float> d_b;

    // host initialization
    for (int j=0; j < blocksize; j++) {
        h_a.push_back(rand() % 20 + 1);
        std::cout << h_a[j] << "\t";
    }
    std::cout << "\n";

    //Kopiowanie host->device
    d_a = h_a;
    d_b.resize(d_a.size());    // zajmuje i uznaje za wypelnione

    thrust::inclusive_scan(d_a.begin(),d_a.end(),d_b.begin(),myMax());

    //Kopiowanie device->host
    h_b = d_b;

    for (int i = 0; i < blocksize; i++) {
        std::cout << h_b[i] << "\t ";
    }
    std::cout << "\n";

    return 0;
}

