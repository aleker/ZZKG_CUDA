#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>     // cout
#include <thrust/functional.h>
#include <stdlib.h>     /* srand, rand */
#include <ctime>
#include <algorithm>

/*
 * Im więcej blokow tym lepiej - jeden blok na jednym multiprocesorze - chcemy wykorzystac ich jak najwiecej
 * w ramach bloku wielokrotnosc 32 watkow - najlepiej >= 64 watki
 *
 * the total number of threads you can run on a single SM in order to fully utilize it is 2048, 
 * so you can have 2 blocks and 1024 thread each, 
 * or you can have 4 blocks and 512 each and so on
 */

const int MAX_THREADS_PER_ONE_DIM_BLOCK = 512;
const int MAX_THREADS_PER_TWO_DIM_BLOCK = 1024;
const int BLOCKS_COUNT_PER_ONE_SM = 8;    // Each SM can run up to 8 block in parallel
const int TOTAL_NUMBER_OF_THREADS_PER_SM = 2048;
const int THREAD_PER_BLOCK_MULTIPLE = 32;
const int MIN_THREAD_COUNT_PER_BLOCK = 2 * THREAD_PER_BLOCK_MULTIPLE;
const int SM_COUNT = 384;                 // GeForce MX150
// const int blockSize = MIN_THREAD_COUNT_PER_BLOCK;
const int blockSize = MAX_THREADS_PER_ONE_DIM_BLOCK;

__device__ int maxValue(int value, int minValue) {
   if (minValue > value) return minValue;
   else return value;
}

__global__ void searchForSubstring(
   unsigned char *sub_x, 
   int i_count, 
   unsigned char *sub_y, 
   int j_count,
   int iterationNo,
   unsigned int *lcs_mtx
   ) {
   
   int rowLength = blockDim.x; //512
   int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
   int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
   int tid = rowNo * rowLength + columnNo; // global thread id (gtid)
   int bid = threadIdx.x;      // thread id in block (btid)

   columnNo = threadIdx.x; // 15
   rowNo = blockIdx.x;      // 2
   tid = rowNo * rowLength + columnNo;

    if (columnNo < i_count && rowNo < j_count) {
       // runda zerowa
       if (iterationNo == 0) {
         // set first column and first row
         if ((rowNo == 0 && columnNo < i_count) || (columnNo == 0 && rowNo < j_count)) {
            lcs_mtx[tid] = 0;
         }
      } // pierwsza runda
      else if (iterationNo == 1) {
         if (rowNo == 1 && columnNo == 1) {
            lcs_mtx[tid] = 0;
         }
      } // pozostale rundy
      else {
         if ((rowNo < iterationNo && columnNo <= iterationNo) || (rowNo <= iterationNo && columnNo < iterationNo)) {
            // przesuniete indeksy
            int x_idx = columnNo - 1;
            int y_idx = rowNo - 1;
            if (sub_x[x_idx] == sub_y[y_idx]) {
                int prev_tid = (rowNo - 1) * rowLength + (columnNo - 1);
                lcs_mtx[tid] = lcs_mtx[prev_tid] + 1;
            } else if (sub_x[x_idx] != sub_y[y_idx]) {
               // todo wlasna funkcja max na device
               int prev_tid_1 = (rowNo - 1) * rowLength + (columnNo);
               int prev_tid_2 = (rowNo) * rowLength + (columnNo - 1);
               lcs_mtx[tid] = maxValue(lcs_mtx[prev_tid_1], lcs_mtx[prev_tid_2]);
            }
         }
      }
   }

}

int main() {
    // input
    std::string X = "1232412";
    std::string Y = "243121";

   thrust::host_vector<unsigned char> h_sub_x(X.begin(), X.end());
   thrust::device_vector<unsigned char> d_sub_x;

    thrust::host_vector<unsigned char> h_sub_y(Y.begin(), Y.end());
    thrust::device_vector<unsigned char> d_sub_y;

   thrust::host_vector<unsigned int> h_lcs_mtx;
   thrust::device_vector<unsigned int> d_lcs_mtx;


   // compute dimensions
   int i_count = X.length() + 1;
   int j_count = Y.length() + 1;
   int elementsCount = i_count * j_count;


   //Kopiowanie host->device
   d_sub_x = h_sub_x;
   d_sub_y = h_sub_y;
   d_lcs_mtx.resize(elementsCount);    // zajmuje i uznaje za wypelnione

   // I PHASE: compute LCS
   int iterationsCount = std::max(i_count, j_count);
   for (int i = 0; i < iterationsCount; i++) {
      int elementsCountInIteration = i + 1;
      // one dimentional grid of 64-thread one dimentional blocks
      int blockCount = i + 1;  // tyle ile rows
      // todo optymalizacja liczby watkow
      searchForSubstring << <blockCount, blockSize >> > 
         (d_sub_x.data().get(), i_count, d_sub_y.data().get(), j_count, i, d_lcs_mtx.data().get());

   }

   h_lcs_mtx = d_lcs_mtx;

   for (int j = 0; j < j_count; j++) {
      for (int i = 0; i < i_count; i++) {
         std::cout << h_lcs_mtx[j*i_count + i] << "\t ";
      }
       std::cout << "\n";
   }
   std::cout << "\n";
   

   return 0;
}

