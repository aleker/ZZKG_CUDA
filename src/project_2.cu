#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "helper.hpp"

__device__ int maxValue(int value, int minValue) {
    if (minValue > value) return minValue;
    else return value;
}

__device__ int computeTid(int columnNo, int rowNo, int columnCount) {
    return rowNo * columnCount + columnNo;
}

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
        else if (rowNo > 0 && columnNo > 0 && (rowNo + columnNo == (iterationNo + 1)))  {
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

__global__ void computeResult(
        unsigned char *sub_x,
        int i_count,            // column < i_count
        unsigned char *sub_y,
        int j_count,            // row < j_count
        int iterationNo,
        unsigned int *lcs_mtx,
        unsigned char *result_mtx
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

    if (columnNo < i_count && rowNo < j_count && columnNo > 0 && rowNo > 0) {
        if (rowNo + columnNo == iterationNo)  {
            if ((rowNo < iterationNo && columnNo <= iterationNo) || (rowNo <= iterationNo && columnNo < iterationNo)) {
                // todo
                // result_mtx[tid] = iterationNo + 47;
            }
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

    thrust::host_vector<unsigned char> h_result_mtx;
    thrust::device_vector<unsigned char> d_result_mtx;

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
    d_result_mtx.resize(elementsCount);

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
    for (int j = 0; j < j_count; j++) {
        for (int i = 0; i < i_count; i++) {
            std::cout << h_lcs_mtx[j * i_count + i] << "\t ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    int maxChainLength = h_lcs_mtx[elementsCount - 1];
    std::cout << "maxChainLength" << maxChainLength << "\n\n";

    // PHASE II reduce
    int blockInRowCount = computeBlockInRowCount(blockSize, i_count);
    int blockInColumnCount = j_count;
    dim3 dimGrid(blockInRowCount, blockInColumnCount);
    reduceLCS << < dimGrid, blockSize >> >
                            (i_count, j_count, maxChainLength, d_lcs_mtx.data().get(), d_reduced_lcs_mtx.data().get());

    // Copy device -> host
    h_lcs_mtx = d_reduced_lcs_mtx;
    for (int j = 0; j < j_count; j++) {
        for (int i = 0; i < i_count; i++) {
            std::cout << h_lcs_mtx[j * i_count + i] << "\t ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // PHASE III results
    for (int i = iterationsCount; i > 0; i--) {
        int blockInRowCount = computeBlockInRowCount(blockSize, std::min(i_count, i + 1));
        int blockInColumnCount = std::min(i + 1, j_count);
        int threadsCount = blockInColumnCount * blockInRowCount * blockSize;
        std::cout << "I" << i << ": " << blockInRowCount << "[" << blockSize <<
                  "]" << " x " << blockInColumnCount << "(" << threadsCount << ")\n";
        dim3 dimGrid(blockInRowCount, blockInColumnCount);
        // todo thread count optimalization
        computeResult << < dimGrid, blockSize >> >
                                (d_sub_x.data().get(), i_count, d_sub_y.data().get(), j_count, i,
                                        d_reduced_lcs_mtx.data().get(), d_result_mtx.data().get());
    }

    // Copy device -> host
    h_result_mtx = d_result_mtx;
    for (int j = 0; j < j_count; j++) {
        for (int i = 0; i < i_count; i++) {
            std::cout << h_result_mtx[j * i_count + i] << "\t ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    return 0;
}

