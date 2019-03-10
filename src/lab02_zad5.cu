#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <string>       // stream
#include <iostream>     // cout
#include <sstream>      // stringstream

#define FLT_MAX 3.402823466e+38F /* max value */

int vectorSize = 10;

struct wsp {
    float a;
    float b;
    float c;
    wsp(float a,float b, float c) : a(a), b(b), c(c) { }
};

struct res {
    int how_many_results;
    float x0_or_x1;
    float x2;
    float max;
};

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

__global__ void compute(wsp* data_vector, res* result_vector, unsigned int count)
{
    // thread id in warp (column no)
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    res result;

    if (tid < count) {
        // 0 = ax2 + bx + c, delta = b2 + 4ac, x0 = -(+) b -sqrt(delta)/2a
        float delta = data_vector[tid].b * data_vector[tid].b + 4.0f * data_vector[tid].a * data_vector[tid].c;
        if (delta < 0) {
            result.how_many_results = 0;
        } else if (delta == 0) {
            // -b/2a
            result.how_many_results = 1;
            result.x0_or_x1 = -data_vector[tid].b / 2.0f * data_vector[tid].a;
        } else if (delta > 0) {
            result.how_many_results = 2;
            result.x0_or_x1 = (data_vector[tid].b - sqrt(delta))/ 2.0f * data_vector[tid].a;
            result.x2 = (-data_vector[tid].b - sqrt(delta))/ 2.0f * data_vector[tid].a;
        }
        // MAX
        if (data_vector[tid].a >= 0) result.max = FLT_MAX;
        else if (data_vector[tid].a < 0) {
            // q = -delta/4a
            result.max = -delta/ (4.0f * data_vector[tid].a);
        }
        result_vector[tid] = result;
    }
}

int main() {
    thrust::host_vector<wsp> h_tab;
    thrust::device_vector<wsp> d_tab;

    thrust::host_vector<res> h_tab_result;
    thrust::device_vector<res> d_tab_result;

    // host initialization
    for (int i=0; i < vectorSize; i++) {
        h_tab.push_back(wsp(rand(), rand(),rand()));
    }

    //Kopiowanie host->device
    d_tab = h_tab;
    d_tab_result.resize(vectorSize);    // zajmuje i uznaje za wypelnione

    // popracuj na GPU

    compute << <1, 10 >> > (d_tab.data().get(), d_tab_result.data().get(), d_tab.size());

    //Kopiowanie device->host
    h_tab_result = d_tab_result;

    for (int i = 0; i < vectorSize; i++) {
        std::stringstream str;
        str << "(a,b,c)= (" << h_tab[i].a << ", " << h_tab[i].b << ", " << h_tab[i].c << "), ResCount: " <<
            h_tab_result[i].how_many_results << ", x1: " << h_tab_result[i].x0_or_x1 << ", x2: " <<
            h_tab_result[i]. x2 << ", max: " << h_tab_result[i].max << "\n";
        std::cout << str.str();
    }

    return 0;
}

