#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

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

const int matrixWidth = 10;
const int matrixLength = 32;
const int blocksize = 16;

__global__ void revert(float *inputMatrix, float *resultMatrix, int maxYPosition, int maxXPosition)
{
    int columnNo = blockIdx.x * blockDim.x + threadIdx.x;
    int rowNo = blockIdx.y * blockDim.y + threadIdx.y;
    int index = rowNo * matrixWidth + columnNo;
    int revertIndex = maxXPosition * maxYPosition - 1 - index;
    if (columnNo < maxXPosition && rowNo < maxYPosition)
        resultMatrix[index] = inputMatrix[revertIndex];
}

int main() {
    // host declarations
    float *resultMatrix = new float[matrixWidth*matrixLength];
    float *inputMatrix = new float[matrixWidth*matrixLength];
    for (int i = 0; i < matrixLength * matrixWidth; ++i) {
        inputMatrix[i] = 2* i;
    }

    // cuda declarations
    const int matrixByteSize = matrixWidth * matrixLength * sizeof(float);
    float *c_resultMatrix, *c_inputMatrix;
    checkCuda(cudaMalloc((void**)&c_resultMatrix, matrixByteSize));
    checkCuda(cudaMalloc((void**)&c_inputMatrix, matrixByteSize));

    // copy host -> cuda
    checkCuda(cudaMemcpy(c_inputMatrix, inputMatrix, matrixByteSize, cudaMemcpyHostToDevice));

    // declare block and grid
    dim3 dimBlock(blocksize, blocksize);
    int block_columns_count = ceil((float) matrixWidth / (float) dimBlock.y);
    int block_row_count = ceil((float) matrixLength / (float) dimBlock.x);
    printf("grid dim: %d blocks height x %d blocks width\n", block_row_count, block_columns_count);
    dim3 dimGrid(block_columns_count, block_row_count);

    // call cuda function
    revert << <dimGrid, dimBlock >> > (c_inputMatrix, c_resultMatrix, matrixLength, matrixWidth);

    // copy cuda -> host
    checkCuda(cudaMemcpy(resultMatrix, c_resultMatrix, matrixByteSize, cudaMemcpyDeviceToHost));

    checkCuda(cudaDeviceSynchronize());
    // display
    for (int row = 0; row < matrixLength; row++) {
        for (int column=0; column < matrixWidth; column++) {
            printf("%d ", (int) resultMatrix[row * matrixWidth + column]);
        }
        printf("\n");
    }
	for (int row = 0; row < matrixLength * matrixWidth; row++) {
		printf("%d ", (int) resultMatrix[row]);
	}

    // free memory on cuda and host
    cudaFree(c_resultMatrix);
    cudaFree(c_inputMatrix);
    delete[] resultMatrix;
    delete[] inputMatrix;
    return 0;
}

