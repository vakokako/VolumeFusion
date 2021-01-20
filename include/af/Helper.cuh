#ifndef HELPER_CUH
#define HELPER_CUH

#include "af/Constants.h"
#include "af/eigen_extension.cuh"
#include <cuda_runtime_api.h>
// fixing intellisense not recognising cuda identifiers such as threadIdx, blockDim, atomicAdd...
#include <device_launch_parameters.h>
// #include "device_atomic_functions.hpp"

#include <string>
#include <iostream>

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

// compute grid size from block size
inline dim3 computeGrid1D(const dim3 &block, const int w)
{
    return dim3((w - 1) / block.x + 1, 1, 1);
}

inline dim3 computeGrid2D(const dim3 &block, const int w, const int h)
{
    return dim3((w - 1) / block.x + 1, (h - 1) / block.y + 1, 1);
}

inline dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s)
{
    return dim3((w - 1) / block.x + 1, (h - 1) / block.y + 1, (s - 1) / block.z + 1);
}

inline void setupBlock1D(dim3& block) {
	block = dim3(1024, 1, 1);
}
inline void setupBlock2D(dim3& block) {
	block = dim3(32, 32, 1);
}
inline void setupBlock3D(dim3& block) {
	block = dim3(16, 8, 8);
}

inline void setupBlockGrid(dim3& block, dim3& grid, const std::size_t w) {
	setupBlock1D(block);
	grid = computeGrid1D(block, w);
}
inline void setupBlockGrid(dim3& block, dim3& grid, const std::size_t w, const std::size_t h) {
	setupBlock2D(block);
	grid = computeGrid2D(block, w, h);
}
inline void setupBlockGrid(dim3& block, dim3& grid, const std::size_t w, const std::size_t h, const std::size_t s) {
	setupBlock3D(block);
	grid = computeGrid3D(block, w, h, s);
}


template<class DeviceType>
DeviceType* allocateDeviceArray(const std::size_t cSize) {
    DeviceType* data_d = NULL;
    cudaMalloc(&data_d, sizeof(DeviceType) * cSize);CUDA_CHECK;
    return data_d;
}

template<class Type, class DeviceType = typename std::remove_cv<Type>::type>
DeviceType* allocateDeviceArrayCopy(Type* data, const std::size_t cSize) {
    DeviceType* data_d = allocateDeviceArray<DeviceType>(cSize);
    cudaMemcpy(data_d, data, sizeof(Type) * cSize, cudaMemcpyHostToDevice);CUDA_CHECK;
    return data_d;
}

template<class Type>
void copyToDevice(Type* data_d, const Type* data, const std::size_t cSize) {
    cudaMemcpy(data_d, data, sizeof(Type) * cSize, cudaMemcpyHostToDevice);
    CUDA_CHECK;
}

template<class Type>
void copyToHost(Type* data, const Type* data_d, const std::size_t cSize) {
    cudaMemcpy(data, data_d, sizeof(Type) * cSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK;
}

void runSquareKernel(float* output, const float* cInput, unsigned int inputSize);
void runInitIdentityKernel(float* matrix, unsigned int matrixDiagSize);

template<class Type>
__global__ void substrKernel(Type* output, const Type* first, const Type* second, unsigned int inputSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= inputSize)
        return;
    output[cIdx] = first[cIdx] - second[cIdx];
}
template<class Type>
void runSubstrKernel(Type* output, const Type* first, const Type* second, unsigned int inputSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, inputSize);
    substrKernel<<<grid, block>>>(output, first, second, inputSize);
}



// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

// __VA_ARGS__ is used because in the call expression there could be commas that are cannot be guarded, e.g. template type
#define Time(aName, timer, ...)  \
    timer.start(); \
    __VA_ARGS__ \
    timer.end(); \
    std::cout << "Time: " << aName << ": " << timer.get() << "s.\n";

#define Bench(aName, timer, ...)  \
    { int runsCount = 20; \
    timer.start(); \
    for (int i=0;i<runsCount;++i) \
        __VA_ARGS__ \
    timer.end(); \
    std::cout << "Time: " << aName << ": " << timer.get()/runsCount << "s.\n";
#define TimedLoop(aName, aIterations, timer, ...) { \
    timer.start(); \
    for (size_t vLoopIter = 0; vLoopIter < aIterations; ++vLoopIter) \
        __VA_ARGS__ \
    timer.end(); \
    std::cout << "Time: " << aName << ": " << timer.get()/aIterations << "s.\n";

#endif