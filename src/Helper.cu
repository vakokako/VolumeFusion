#include <af/Helper.cuh>


// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cout << "CUDA_CHECK error:\n"
                                 + file + ", line " + std::to_string(line) + ": " + cudaGetErrorString(e) + " (" + std::to_string(e) + ")\n"
                                 + "Previous CUDA call:\n"
                                 + prev_file + ", line " + std::to_string(prev_line) + "\n";
        throw std::runtime_error("");
    }
    prev_file = file;
    prev_line = line;
}

__global__ void squareKernel(float* output, const float* cInput, unsigned int inputSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= inputSize)
        return;
    float value = cInput[cIdx];
    output[cIdx] = value * value;
}

void runSquareKernel(float* output, const float* cInput, unsigned int inputSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, inputSize);
    squareKernel<<<grid, block>>>(output, cInput, inputSize);
    cudaDeviceSynchronize();
}

__global__ void initIdentityKernel(float* matrix, unsigned int matrixDiagSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= matrixDiagSize)
        return;
    matrix[cIdx * matrixDiagSize + cIdx] = 1.f;
}

void runInitIdentityKernel(float* matrix, unsigned int matrixDiagSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, matrixDiagSize);
    initIdentityKernel<<<grid, block>>>(matrix, matrixDiagSize);

    cudaDeviceSynchronize();
}