#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cassert>
#include <iostream>

#include "helper.hpp"
#include "kernels.hpp"

int main() {
    // input
    std::string X; // = "1232412";
    std::string Y; // = "243121";
    std::cout << "Enter first substring:\n";
    std::cin >> X;
    std::cout << "\nEnter second substring:\n";
    std::cin >> Y;
    std::cout << "\nComputing...\n";


    // variables
    thrust::host_vector<unsigned char> h_sub_x(X.begin(), X.end());
    thrust::device_vector<unsigned char> d_sub_x;

    thrust::host_vector<unsigned char> h_sub_y(Y.begin(), Y.end());
    thrust::device_vector<unsigned char> d_sub_y;

    thrust::host_vector<unsigned int> h_lcs_mtx;
    thrust::device_vector<unsigned int> d_lcs_mtx;
    thrust::device_vector<unsigned int> d_reduced_lcs_mtx;

    thrust::host_vector<unsigned int> h_result_mtx;
    thrust::device_vector<unsigned int> d_result_mtx;


    // compute dimensions
    int i_count = h_sub_x.size() + 1;
    int j_count = h_sub_y.size() + 1;
    int elementsCount = i_count * j_count;

    // Copy host -> device
    d_sub_x = h_sub_x;
    d_sub_y = h_sub_y;
    // resize
    d_lcs_mtx.resize(elementsCount);
    d_reduced_lcs_mtx.resize(elementsCount);

    // PHASE I: compute LCS
    int iterationsCount = i_count + j_count - 1;
    std::cout << "columns = " << i_count << "\nrows = " << j_count << "\n";
    std::cout << "iterationsCount = " << iterationsCount;
    for (int i = 0; i < iterationsCount; i++) {
        int blockInRowCount = computeBlockInRowCount(blockSize, std::min(i_count, i + 1));
        int blockInColumnCount = std::min(i + 1, j_count);
        int threadsCount = blockInColumnCount * blockInRowCount * blockSize;
        std::cout << "I" << i << ": " << blockInRowCount << "[" << blockSize <<
            "]" << " x " << blockInColumnCount << "(" << threadsCount << ")\n";
        dim3 dimGrid(blockInRowCount, blockInColumnCount);
        // todo thread count optimalization
        createLCS << < dimGrid, blockSize >> >
            (d_sub_x.data().get(), i_count, d_sub_y.data().get(), j_count, i, d_lcs_mtx.data().get());
    }

    // Copy device -> host
    h_lcs_mtx = d_lcs_mtx;
    printArray(i_count, j_count,  h_lcs_mtx.begin(), "PHASE I - compute LCS");

    const int maxChainLength = h_lcs_mtx[elementsCount - 1];
    std::cout << "MAX CHAIN LENGTH: " << maxChainLength << "\n\n";

    // PHASE II reduce
    int blockInRowCount = computeBlockInRowCount(blockSize, i_count);
    int blockInColumnCount = j_count;
    dim3 dimGrid(blockInRowCount, blockInColumnCount);
    reduceLCS << < dimGrid, blockSize >> >
                            (i_count, j_count, maxChainLength, d_lcs_mtx.data().get(), d_reduced_lcs_mtx.data().get());

    // Copy device -> host
    h_lcs_mtx = d_reduced_lcs_mtx;
    printArray(i_count, j_count, h_lcs_mtx.begin(), "PHASE II - REDUCE LCS");


    // PHASE III collect ids of found elements
    d_result_mtx.resize(i_count * maxChainLength);
    thrust::host_vector<unsigned int> h_elementCountArray(maxChainLength * 2);
    int permutationsCount = 1;
    for (int i = 0; i < maxChainLength; i++) {
        thrust::device_vector<unsigned int>::iterator d_b_newend = thrust::copy_if(thrust::make_counting_iterator(0),
                                                                                   thrust::make_counting_iterator(elementsCount),
                                                                                   d_reduced_lcs_mtx.begin(),
                                                                                   d_result_mtx.begin() + i * i_count,
                                                                                   isEqual(i + 1));
        int foundCount = d_b_newend - (d_result_mtx.begin() + i * i_count);
        h_elementCountArray[i] = foundCount;
        permutationsCount *= foundCount;
        assert(permutationsCount != 0);
        h_elementCountArray[i + maxChainLength] = permutationsCount;
        std::cout << (i + 1) << ": Found " << foundCount << " elements.\n";
    }

    // Copy device -> host
    h_result_mtx = d_result_mtx;
    printArray(maxChainLength, 2, h_elementCountArray.begin(), "founded -> permutation");
    printArray(i_count, maxChainLength, h_result_mtx.begin(), "PHASE III - copy_if - DISPLAY POSITIONS");

    // PHASE IV - find permutations (permutationsCount x maxChainLength)
    thrust::device_vector<unsigned int> d_elementCountArray;
    d_elementCountArray = h_elementCountArray;

    thrust::host_vector<unsigned int> h_permutations_mtx(permutationsCount * maxChainLength);
    thrust::device_vector<unsigned int> d_permutations_mtx;
    d_permutations_mtx = h_permutations_mtx;

    blockInRowCount = computeBlockInRowCount(blockSize, permutationsCount);
    blockInColumnCount = maxChainLength;
    dim3 dimGrid2(blockInRowCount, blockInColumnCount);
    int widthOfResult_mtx = i_count;
    generatePermutations << < dimGrid2, blockSize >> >
                            (widthOfResult_mtx, permutationsCount, maxChainLength, d_result_mtx.data().get(), d_elementCountArray.data().get(), d_permutations_mtx.data().get());
    h_permutations_mtx = d_permutations_mtx;
    printArray(permutationsCount, maxChainLength, h_permutations_mtx.begin(), "PHASE IV - PERMUTATIONS OF POSITIONS");
    std::cout << "PERMUTATIONS COUNT:" << permutationsCount << "\n";

    // PHASE V - filter proper chains and change to char
    thrust::host_vector<unsigned char> h_result_chains;
    thrust::device_vector<unsigned char> d_result_chains;
    thrust::device_vector<unsigned int> d_result_count;
    thrust::host_vector<unsigned int> h_result_count;

    d_result_chains.resize(permutationsCount*maxChainLength);
    d_result_count.resize(permutationsCount);

    blockInRowCount = computeBlockInRowCount(blockSize, permutationsCount);
    blockInColumnCount = 1;
    dim3 dimGrid3(blockInRowCount, blockInColumnCount);
    filterPermutations << < dimGrid3, blockSize >> >
        (permutationsCount, maxChainLength, i_count, d_sub_x.data().get(), d_permutations_mtx.data().get(), d_result_chains.data().get(), d_result_count.data().get());
    h_result_count = d_result_count;
    int d_sum = thrust::reduce(d_result_count.begin(),d_result_count.end());

//    if (permutationsCount  < 20) {
        h_result_chains = d_result_chains;
        printCharArray(permutationsCount, maxChainLength, h_result_chains.begin(), h_result_count.begin(), "PHASE V - FILTER + RESULT");
//    }
    std::cout << "RESULTS COUNT:" << d_sum << "\n";

    return 0;
}

