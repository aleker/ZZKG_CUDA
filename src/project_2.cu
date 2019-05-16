#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cassert>

#include "helper.hpp"

__device__ int maxValue(int value, int minValue) {
    if (minValue > value) return minValue;
    else return value;
}

__device__ int computeTid(int columnNo, int rowNo, int columnCount) {
    return rowNo * columnCount + columnNo;
}

__device__ int getRowForTid(int tid, int columnCount) {
    return tid / columnCount;
}

__device__ int getColumnForTid(int tid, int columnCount) {
    return tid % columnCount;
}

__device__ bool possibleToGo(int currentCharTid, int nextCharTid, int columnCountOfCharArray) {
    int currentColumn = getColumnForTid(currentCharTid, columnCountOfCharArray);
    int nextCharColumn = getColumnForTid(nextCharTid, columnCountOfCharArray);
    int currentRow = getRowForTid(currentCharTid, columnCountOfCharArray);
    int nextCharRow = getRowForTid(nextCharTid, columnCountOfCharArray);
    return nextCharColumn > currentColumn && nextCharRow > currentRow;
}

__device__ int getGroup(int perGroupCount, int columnNo) {
    return columnNo / perGroupCount;
}

struct isEqual {
    int threshold;
    isEqual(int thr) : threshold(thr) {};
    __host__ __device__ bool operator()(const int x) {
        return x == threshold;
    }
};

__global__ void createLCS(
        unsigned char *sub_x,
        int i_count,            // column < i_count
        unsigned char *sub_y,
        int j_count,            // row < j_count
        int iterationNo,
        unsigned int *lcs_mtx
) {
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int gridColumnNo = blockIdx.x;
    int gridRowNo = blockIdx.y;
    int threadXPositionInBlock = threadIdx.x;
    int threadYPositionInBlock = threadIdx.y;
    //
    int columnNo = gridColumnNo * blockWidth + threadXPositionInBlock;
    int rowNo = gridRowNo * blockHeight + threadYPositionInBlock;
    int tid = computeTid(columnNo, rowNo, i_count);

    if (columnNo < i_count && rowNo < j_count) {
        // round No 0
        if (iterationNo == 0) {
            // set first column and first row
            if (rowNo == 0 || columnNo == 0) {
                lcs_mtx[tid] = 0;
            }
        } // round No 1
        else if (iterationNo == 1) {
            if (rowNo == 1 && columnNo == 1) {
                lcs_mtx[tid] = 0;
            }
        } // remaining rounds
        else if (columnNo > 0 && rowNo > 0 && (rowNo + columnNo == iterationNo)) {
            if ((rowNo < iterationNo && columnNo <= iterationNo) || (rowNo <= iterationNo && columnNo < iterationNo)) {
                int x_idx = columnNo - 1;
                int y_idx = rowNo - 1;
                if (sub_x[x_idx] == sub_y[y_idx]) {
                    int prev_tid = computeTid(columnNo - 1, rowNo - 1, i_count);
                    lcs_mtx[tid] = lcs_mtx[prev_tid] + 1;
                } else if (sub_x[x_idx] != sub_y[y_idx]) {
                    int prev_tid_up = computeTid(columnNo, rowNo - 1, i_count);
                    int prev_tid_left = computeTid(columnNo - 1, rowNo, i_count);
                    lcs_mtx[tid] = maxValue(lcs_mtx[prev_tid_up], lcs_mtx[prev_tid_left]);
                }
//                lcs_mtx[tid] = iterationNo;
            }
        }
    }
}

__global__ void reduceLCS(
        int i_count,            // column < i_count
        int j_count,            // row < j_count
        int chainLength,
        unsigned int *lcs_mtx,
        unsigned int *reduced_lcs_mtx
) {
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int gridColumnNo = blockIdx.x;
    int gridRowNo = blockIdx.y;
    int threadXPositionInBlock = threadIdx.x;
    int threadYPositionInBlock = threadIdx.y;
    //
    int columnNo = gridColumnNo * blockWidth + threadXPositionInBlock;
    int rowNo = gridRowNo * blockHeight + threadYPositionInBlock;
    int tid = computeTid(columnNo, rowNo, i_count);

    if (columnNo < i_count && rowNo < j_count) {
        int ignoreLastRowsCount = maxValue(0, chainLength - columnNo);  // <0; chainLength>
        if (rowNo < (j_count - ignoreLastRowsCount)) { // not ignored cell
            int prev_tid_up = computeTid(columnNo, rowNo - 1, i_count);
            int prev_tid_left = computeTid(columnNo - 1, rowNo, i_count);
            if ((lcs_mtx[tid] == lcs_mtx[prev_tid_up] + 1) && (lcs_mtx[tid] == lcs_mtx[prev_tid_left] + 1)) {
                reduced_lcs_mtx[tid] = lcs_mtx[tid];
            }
        }
    }
}

__global__ void generatePermutations(
        int i_array_with_possitions,
        int i_permutationsCount,            // column < i_count
        int j_maxChainLength,            // row < j_count
        unsigned int *array_with_possitions,
        unsigned int *infoFoundAndPermCount,
        unsigned int *result_permutations
) {
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int gridColumnNo = blockIdx.x;
    int gridRowNo = blockIdx.y;
    int threadXPositionInBlock = threadIdx.x;
    int threadYPositionInBlock = threadIdx.y;
    //
    int columnNo = gridColumnNo * blockWidth + threadXPositionInBlock;
    int rowNo = gridRowNo * blockHeight + threadYPositionInBlock;
    int tid = computeTid(columnNo, rowNo, i_permutationsCount);

    if (columnNo < i_permutationsCount && rowNo < j_maxChainLength) {
        int possibleValuesCountForThisRow = infoFoundAndPermCount[rowNo];
        int groupCountForThisRow = infoFoundAndPermCount[j_maxChainLength + rowNo]; // how many groups per row
        if (groupCountForThisRow == 1)
            result_permutations[tid] = array_with_possitions[computeTid(0, rowNo, i_array_with_possitions)];
        else {
            int howManyInGroup = i_permutationsCount / groupCountForThisRow;
            int groupIdForThisThread = getGroup(howManyInGroup, columnNo);
            int indexOfCharId = groupIdForThisThread % possibleValuesCountForThisRow; // index of charId in array_with_possitions
            result_permutations[tid] = array_with_possitions[computeTid(indexOfCharId, rowNo, i_array_with_possitions)];
        }
    }
}

__global__ void filterPermutations(
        int i_permutationsCount,            // column < i_count
        int j_maxChainLength,            // row < j_count
        int i_count,
        unsigned char * word_of_row,
        unsigned int *arrayWithPermutations,
        unsigned char *result_permutations
) {
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int gridColumnNo = blockIdx.x;
    int gridRowNo = blockIdx.y;
    int threadXPositionInBlock = threadIdx.x;
    int threadYPositionInBlock = threadIdx.y;
    //
    int columnNo = gridColumnNo * blockWidth + threadXPositionInBlock;
    int rowNo = gridRowNo * blockHeight + threadYPositionInBlock;
    int tid = computeTid(columnNo, rowNo, i_permutationsCount);

    if (columnNo < i_permutationsCount && rowNo < j_maxChainLength) {
        if (rowNo == 0) { // one per column
            int currentRowNo = rowNo;
            int nextRowNo = rowNo + 1;
            while (nextRowNo < j_maxChainLength) {
                int currentTidInPermutationMtx = computeTid(columnNo, currentRowNo, i_permutationsCount);
                int currentCharId = arrayWithPermutations[currentTidInPermutationMtx];
                int nextCharId = arrayWithPermutations[computeTid(columnNo, nextRowNo, i_permutationsCount)];
                if (possibleToGo(currentCharId, nextCharId, i_count)) {
                    result_permutations[currentTidInPermutationMtx] = word_of_row[getColumnForTid(currentCharId, i_count) - 1];
                } else return;
                currentRowNo++;
                nextRowNo++;
            }
            int currentTidInPermutationMtx = computeTid(columnNo, currentRowNo, i_permutationsCount);
            int currentCharId = arrayWithPermutations[currentTidInPermutationMtx];
            result_permutations[currentTidInPermutationMtx] = word_of_row[getColumnForTid(currentCharId, i_count) - 1];
        }
    }
}

int main() {
    // input
    std::string X = "1232412";
    std::string Y = "243121";

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
    std::cout << "maxChainLength: " << maxChainLength << "\n\n";

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
    printArray(i_count, maxChainLength, h_result_mtx.begin(), "PHASE III - copy_if - count positions");

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
    printArray(permutationsCount, maxChainLength, h_permutations_mtx.begin(), "PHASE IV - PERMUTATIONS");

    // PHASE V - filter proper chains and change to char
    thrust::host_vector<unsigned char> h_result_chains;
    thrust::device_vector<unsigned char> d_result_chains;

    d_result_chains.resize(permutationsCount*maxChainLength);

    blockInRowCount = computeBlockInRowCount(blockSize, permutationsCount);
    blockInColumnCount = 1;
    dim3 dimGrid3(blockInRowCount, blockInColumnCount);
    filterPermutations << < dimGrid3, blockSize >> >
        (permutationsCount, maxChainLength, i_count, d_sub_x.data().get(), d_permutations_mtx.data().get(), d_result_chains.data().get());

    h_result_chains = d_result_chains;
    printCharArray(permutationsCount, maxChainLength, h_result_chains.begin(), "PHASE V - RESULT");

    return 0;
}

