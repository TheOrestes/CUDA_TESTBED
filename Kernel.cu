
#include <cstdio>
#include <cstdlib>

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
int main()
{
	//Hello << <1, 4 >> > ();

	int N = 512;
	size_t vecSize = N * sizeof(float);

	float* vec1 = static_cast<float*>(malloc(vecSize));
	float* vec2 = static_cast<float*>(malloc(vecSize));
	float* outVec = static_cast<float*>(malloc(vecSize));

	// Initialize host array
	for (int i = 0 ; i < N ; ++i)
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
	VectorAdd << <8, 64>> > (gpuVec1, gpuVec2, gpuResult, vecSize);

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

	// wait for the GPU to finish
	// cudaDeviceSynchronize();

	printf("\nHello from the CPU!");

	return 0;
}