#include <iostream>     // cout

/*
 * Im więcej blokow tym lepiej - jeden blok na jednym multiprocesorze - chcemy wykorzystac ich jak najwiecej
 * w ramach bloku wielokrotnosc 32 watkow - najlepiej >= 64 watki
 *
 * the total number of threads you can run on a single SM in order to fully utilize it is 2048,
 * so you can have 2 blocks and 1024 thread each,
 * or you can have 4 blocks and 512 each and so on
 */

const bool printArrays = false;

const int MAX_THREADS_PER_ONE_DIM_BLOCK = 512;
const int MAX_THREADS_PER_TWO_DIM_BLOCK = 1024;
const int BLOCKS_COUNT_PER_ONE_SM = 8;    // Each SM can run up to 8 block in parallel
const int TOTAL_NUMBER_OF_THREADS_PER_SM = 2048;
const int THREAD_PER_BLOCK_MULTIPLE = 32;
const int MIN_THREAD_COUNT_PER_BLOCK = 2 * THREAD_PER_BLOCK_MULTIPLE;
const int SM_COUNT = 384;                 // GeForce MX150

const int blockSize = MIN_THREAD_COUNT_PER_BLOCK;          // 64
//const int blockSize = MAX_THREADS_PER_ONE_DIM_BLOCK;    // 512

int computeBlockInRowCount(int blocksize, int matrixWidth) {
    if (blocksize > matrixWidth)
        return 1;
    return (matrixWidth + blocksize - 1) / blocksize;
}

void printArray(int columnCount, int rowCount, thrust::host_vector<unsigned int>::iterator printIterator, std::string title = "") {
    if (!printArrays) return;
    std::cout << title << "\n";
    for (int i = 0; i < columnCount; i++) {
        if (i == 0)
            std::cout << "X_\t";
        std::cout << i << "_\t ";
    }
    std::cout << "\n";
    for (int j = 0; j < rowCount; j++) {
        for (int i = 0; i < columnCount; i++) {
            if (i == 0)
                std::cout << j << "|\t";
            std::cout << printIterator[j * columnCount + i] << "\t ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void printCharArray(int columnCount, int rowCount, thrust::host_vector<unsigned char>::iterator printIterator, thrust::host_vector<unsigned int>::iterator proper_col, std::string title = "") {
    std::cout << title << "\n";
    for (int i = 0; i < columnCount; i++) {
        if (proper_col[i] != 1) continue;
        std::cout << " * ";
        for (int j = 0; j < rowCount; j++) {
            std::cout << printIterator[j * columnCount + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
