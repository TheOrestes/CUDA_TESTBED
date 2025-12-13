
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <device_launch_parameters.h>

#include "cuda_runtime.h"

//---------------------------------------------------------------------------------------------------------------------
__global__ void Hello()
{
	printf("\nHello from the GPU!");
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void VectorAdd(const float* a, const float* b, float* ans, int n)
{
	// calculate unique thread index
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < n)
	{
		ans[i] = a[i] + b[i];
	}
}

//---------------------------------------------------------------------------------------------------------------------
void VectorAddition()
{
	int N = 512;
	size_t vecSize = N * sizeof(float);

	float* vec1 = static_cast<float*>(malloc(vecSize));
	float* vec2 = static_cast<float*>(malloc(vecSize));
	float* outVec = static_cast<float*>(malloc(vecSize));

	// Initialize host array
	for (int i = 0; i < N; ++i)
	{
		vec1[i] = i * 1.0f;
		vec2[i] = i * 2.0f;
	}

	// Allocate Memory on the Device/GPU
	float* gpuVec1; float* gpuVec2; float* gpuResult;
	cudaMalloc((void**)&gpuVec1, vecSize);
	cudaMalloc((void**)&gpuVec2, vecSize);
	cudaMalloc((void**)&gpuResult, vecSize);

	// Copy data from CPU to GPU
	cudaMemcpy(gpuVec1, vec1, vecSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuVec2, vec2, vecSize, cudaMemcpyHostToDevice);

	// Launch Kernel
	//int threadsPerBlock = 256;
	//int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VectorAdd << <8, 64 >> > (gpuVec1, gpuVec2, gpuResult, vecSize);

	// Copy result back from the GPU to CPU!
	cudaMemcpy(outVec, gpuResult, vecSize, cudaMemcpyDeviceToHost);

	// Print Result
	for (int i = 0; i < N; ++i)
	{
		printf("%f ", outVec[i]);
	}

	cudaFree(gpuVec1);
	cudaFree(gpuVec2);
	cudaFree(gpuResult);
	free(vec1);
	free(vec2);
	free(outVec);
}

//---------------------------------------------------------------------------------------------------------------------
__global__ void MatrixMult(const float* a, const float* b, float* ans, int N)
{
	// calculate unique thread index
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	// boundary check
	if(row < N && col < N)
	{
		float value = 0.0f;

		for(int i = 0 ; i < N ; i++)
		{
			value += a[row * N + i] * b[i * N + col];
		}

		ans[row * N + col] = value;
	}
}

//---------------------------------------------------------------------------------------------------------------------
void MatrixMultiplication()
{
	int N = 512;
	size_t SIZE = N * N * sizeof(float);

	float* cpuMatrixInput1 = static_cast<float*>(malloc(SIZE));
	float* cpuMatrixInput2 = static_cast<float*>(malloc(SIZE));
	float* cpuMatrixResult = static_cast<float*>(malloc(SIZE));

	// Initialize host array
	for (int i = 0; i < N * N; ++i)
	{
		cpuMatrixInput1[i] = sin(i);
		cpuMatrixInput2[i] = cos(i);
	}

	// Allocate Memory on the Device/GPU
	float* gpuMatrix1; float* gpuMatrix2; float* gpuResult;
	cudaMalloc((void**)&gpuMatrix1, SIZE);
	cudaMalloc((void**)&gpuMatrix2, SIZE);
	cudaMalloc((void**)&gpuResult, SIZE);

	// Copy data from CPU to GPU
	cudaMemcpy(gpuMatrix1, cpuMatrixInput1, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuMatrix2, cpuMatrixInput2, SIZE, cudaMemcpyHostToDevice);

	// configure launch kernel parameter
	dim3 threadPerBLock(16, 16);
	dim3 blocksPerGrid((N + threadPerBLock.x - 1) / threadPerBLock.x, 
						(N + threadPerBLock.y - 1) / threadPerBLock.y);
	// Launch Kernel
	MatrixMult << <blocksPerGrid, threadPerBLock >> > (gpuMatrix1, gpuMatrix2, gpuResult, N);

	// Copy result back from the GPU to CPU!
	cudaMemcpy(cpuMatrixResult, gpuResult, SIZE, cudaMemcpyDeviceToHost);

	// Print Result
	for (int i = 0; i < N; ++i)
	{
		printf("%f ", cpuMatrixResult[i]);
	}

	cudaFree(gpuMatrix1);
	cudaFree(gpuMatrix2);
	cudaFree(gpuResult);
	free(cpuMatrixInput1);
	free(cpuMatrixInput2);
	free(cpuMatrixResult);
}

//---------------------------------------------------------------------------------------------------------------------
int main()
{
	printf("\n--------------- Start from the CPU!\n");

	//VectorAddition();
	MatrixMultiplication();

	printf("\n--------------- End from the CPU!\n");

	return 0;
}