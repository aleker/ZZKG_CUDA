
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

/*
 * Im więcej blokow tym lepiej - jeden blok na jednym multiprocesorze - chcemy wykorzystac ich jak najwiecej
 * w ramach blku wielokrotnosc 32 watkow - najlepiej >= 64 watki
 *
 */

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

const int matrixLength = 100;
const int blocksize = 16;		// blocksize = 16*16 = 256 threads in one block

__global__ void add_matrix(float *firstMatrix, float *secondMatrix, float *resultMatrix, int maxWidth)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * maxWidth + column;
    if (column < maxWidth && row < maxWidth)
        resultMatrix[index] = firstMatrix[index] + secondMatrix[index];
}


int main() {
    // host matrices declarations
	float *firstMatrix = new float[matrixLength*matrixLength];
	float *secondMatrix = new float[matrixLength*matrixLength];
	float *resultMatrix = new float[matrixLength*matrixLength];
	for (int i = 0; i < matrixLength*matrixLength; ++i) {
		firstMatrix[i] = 1.0f;
		secondMatrix[i] = 3.5f;
	}

	// cuda matrices declarations
    const int matrixByteSize = matrixLength * matrixLength * sizeof(float);
    float *c_firstMatrix, *c_secondMatrix, *c_resultMatrix;
    checkCuda(cudaMalloc((void**)&c_firstMatrix, matrixByteSize));
    checkCuda(cudaMalloc((void**)&c_secondMatrix, matrixByteSize));
    checkCuda(cudaMalloc((void**)&c_resultMatrix, matrixByteSize));

	// copy host -> cuda
    checkCuda(cudaMemcpy(c_firstMatrix, firstMatrix, matrixByteSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(c_secondMatrix, secondMatrix, matrixByteSize, cudaMemcpyHostToDevice));

	// declare block and grid
	dim3 dimBlock(blocksize, blocksize);
	int block_columns_count = ceil((float) matrixLength / (float) dimBlock.x);
	int block_rows_count = ceil((float) matrixLength / (float) dimBlock.y);
	printf("grid dim: %d blocks x %d blocks\n", block_columns_count, block_rows_count);
	dim3 dimGrid(block_columns_count, block_rows_count);

	// call cuda function
	add_matrix << <dimGrid, dimBlock >> > (c_firstMatrix, c_secondMatrix, c_resultMatrix, matrixLength);

	// copy cuda -> host
    checkCuda(cudaMemcpy(resultMatrix, c_resultMatrix, matrixByteSize, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	// display
	for (int row = 0; row < matrixLength; row++) {
		for (int column=0; column < matrixLength; column++) {
			printf("%.1f ", resultMatrix[row * matrixLength + column]);
		}
		printf("\n");
	}

	// free memory on cuda and host
    checkCuda(cudaFree(c_firstMatrix)); checkCuda(cudaFree(c_secondMatrix)); checkCuda(cudaFree(c_resultMatrix));
	delete[] firstMatrix; delete[] secondMatrix; delete[] resultMatrix;

	return 0;
}


