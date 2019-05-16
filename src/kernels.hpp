
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
        unsigned char *result_permutations,
        unsigned int *resultCount
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
            resultCount[columnNo] = 1;
        }
    }
}

