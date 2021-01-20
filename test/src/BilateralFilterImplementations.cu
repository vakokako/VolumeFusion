#include "af/Helper.cuh"

// Reference -
// http://ecee.colorado.edu/~siewerts/extra/code/example_code_archive/a490dmis_code/CUDA/cuda_work/samples/3_Imaging/bilateralFilter/bilateral_kernel.cu
#include <af/stdio.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

#define M_PI 3.14159265358979323846
#define TILE_X 32           // block width
#define TILE_Y 32           // block height
#define OFFSET_X 5          // this parameter is used by share memory pre-allocation
#define GRAY_GASS_STEPS 30  // this parameter is used to divide 3*sigma into GRAY_GASS_STEPS sections
#define GRAY_GASS_BUF_LEN (GRAY_GASS_STEPS + 1)

#define __FUNCTION_START__ printf("\n start %s... \n", __FUNCTION__)
#define __FUNCTION_END__ printf("\n finish %s. \n", __FUNCTION__)

using namespace std;
using namespace cv;
// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
__constant__ float cGaussian[64];
__constant__ float cSumGassian[2];
__constant__ float cGaussian2D[400];
__constant__ int cClrGassian[GRAY_GASS_BUF_LEN];

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Initialize texture memory to store the input
texture<unsigned char, 2, cudaReadModeElementType> inTexture;

/*
   GAUSSIAN IN 1D FOR SPATIAL DIFFERENCE

   Here, exp(-[(x_centre - x_curr)^2 + (y_centre - y_curr)^2]/(2*sigma*sigma)) can be broken down into ...
   exp[-(x_centre - x_curr)^2 / (2*sigma*sigma)] * exp[-(y_centre - y_curr)^2 / (2*sigma*sigma)]
   i.e, 2D gaussian -> product of two 1D Gaussian

   A constant Gaussian 1D array can be initialzed to store the gaussian values
   Eg: For a kernel size 5, the pixel difference array will be ...
   [-2, -1, 0, 1 , 2] for which the gaussian kernel is applied

   To minimize the product operation, the following function do the two optimization:
   1.  calculate the 2D Gaussian parameter directly
   2.  use a fixed-length integer lookup table instead of calculating the intensity-gaussian weight each time.

*/
void updateGaussian16(int r, double sd) {
    float fGaussian[64];
    for (int i = 0; i < 2 * r + 1; i++) {
        float x      = i - r;
        fGaussian[i] = expf(-(x * x) / (2 * sd * sd));
    }
    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * r + 1));

    float fSumGassian = 0;
    int index         = 0;
    float w           = 0;
    float fGaussian2D[400];
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            w                  = (fGaussian[dy + r] * fGaussian[dx + r]);
            fGaussian2D[index] = w;
            fSumGassian += w;
            index += 1;
        }
    }
    fSumGassian = 1.0 / fSumGassian;
    cudaMemcpyToSymbol(cGaussian2D, fGaussian2D, sizeof(float) * (2 * r + 1) * (2 * r + 1));
    cudaMemcpyToSymbol(cSumGassian, &fSumGassian, sizeof(float));

    // calculate intensity gaussian
    float fClrGassian[GRAY_GASS_BUF_LEN];
    int nClrGassian[GRAY_GASS_BUF_LEN];
    for (int i = 0; i <= GRAY_GASS_STEPS; i++) {
        fClrGassian[i] = exp(-(pow(i / 10.0, 2)) / (2));
    }
    for (int i = 0; i <= GRAY_GASS_STEPS; i++) {
        nClrGassian[i] = (int)(fClrGassian[i] / fClrGassian[GRAY_GASS_STEPS]);
        // printf("nClrGassian[%d] = %d, fClrGassian[%d] = %.5f\n", i, nClrGassian[i], i, fClrGassian[i]);
    }
    cudaMemcpyToSymbol(cClrGassian, nClrGassian, sizeof(int) * GRAY_GASS_BUF_LEN);
}

// Gaussian function for range difference
__device__ inline float gaussian16(float x, float sigma) { return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2))); }

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * while several optimization skills is used:
 * 1. use  fixed-length lookup tables to avoid calculate filtering weight each time;
 * 2. use Texture instead of access global memory directly, a tiny speed up is achieved by this skill.
 *
 * Input parameters are described:
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuFilterTextureOpm(float* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // If within image size
    if ((x < width) && (y < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        float centrePx = tex2D(texRef, x, y);

        // Iterate through filter size from centre pixel
        int index        = 0;
        float normFactor = GRAY_GASS_STEPS / (3 * sI);
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = tex2D(texRef, x + dx, y + dy);
                int x0       = abs((centrePx - currPx) * normFactor);
                if (x0 > GRAY_GASS_STEPS || x0 < 0) {
                    x0 = GRAY_GASS_STEPS;
                }

                float w = cGaussian2D[index] * cClrGassian[x0];
                iFiltered += w * currPx;
                wP += w;
                index += 1;
            }
        }
        output[y * width + x] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * while several optimization skills is used:
 * 1. use  fixed-length lookup tables to avoid calculate filtering weight each time;
 * 2. use Texture instead of access global memory directly, a tiny speed up is achieved by this skill.
 *
 * Input parameters are described:
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuFilterOpm(float* output, const float* input, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    // unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // If within image size
    if ((x < width) && (y < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        float centrePx = input[y * width + x];

        // Iterate through filter size from centre pixel
        int index        = 0;
        float normFactor = GRAY_GASS_STEPS / (3 * sI);
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = input[(y + dy) * width + x + dx];
                int x0       = abs((centrePx - currPx) * normFactor);
                if (x0 > GRAY_GASS_STEPS || x0 < 0) {
                    x0 = GRAY_GASS_STEPS;
                }

                float w = cGaussian2D[index] * cClrGassian[x0];
                iFiltered += w * currPx;
                wP += w;
                index += 1;
            }
        }
        output[y * width + x] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * in this function, the pixel value is fetched through texture
 *
 * Input parameters are described:
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuFilterTexture(float* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // If within image size
    if ((x < width) && (y < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        float centrePx = tex2D(texRef, x, y);

        // Iterate through filter size from centre pixel
        int index = 0;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = tex2D(texRef, x + dx, y + dy);
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian16(centrePx - currPx, sI);
                iFiltered += w * currPx;
                wP += w;
            }
        }
        output[y * width + x] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * in this function, the pixel value is wrapped as 8 bit and fetched through texture
 *
 * Input parameters are described:
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuBiFilter8(unsigned char* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // If within image size
    if ((x < width) && (y < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        unsigned char centrePx = tex2D(texRef, x, y);

        // Iterate through filter size from centre pixel
        int index = 0;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                unsigned char currPx = tex2D(texRef, x + dx, y + dy);
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian16(centrePx - currPx, sI);
                iFiltered += w * currPx;
                wP += w;
            }
        }
        output[y * width + x] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * in this function, the pixel value is wrapped as float
 * and accessed using global memory pointer
 *
 * Input parameters are described:
 * @param input	the memory pointer of source image data
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuBiFilter32(float* input, float* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
    int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

    // If within image size
    if ((txIndex < width) && (tyIndex < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        float centrePx = input[tyIndex * width + txIndex];
        // Iterate through filter size from centre pixel
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = input[(tyIndex + dx) * width + (dy + txIndex)];
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian16(centrePx - currPx, sI);
                iFiltered += w * currPx;
                wP += w;
            }
        }
        output[tyIndex * width + txIndex] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * in this function, the pixel value is wrapped as float
 * and accessed using global memory pointer
 *
 * Input parameters are described:
 * @param input	the memory pointer of source image data
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuBiFilterGlobalOpm(float* input, float* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
    int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

    // If within image size
    if ((txIndex < width) && (tyIndex < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        float centrePx = input[tyIndex * width + txIndex];
        // Iterate through filter size from centre pixel
        int index        = 0;
        float normFactor = GRAY_GASS_STEPS / (3 * sI);
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = input[(tyIndex + dy) * width + (dx + txIndex)];
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)

                int x0 = abs((centrePx - currPx) * normFactor);
                if (x0 > GRAY_GASS_STEPS || x0 < 0) {
                    x0 = GRAY_GASS_STEPS;
                }

                float w = cGaussian2D[index] * cClrGassian[x0];
                iFiltered += w * currPx;
                wP += w;
                index += 1;
            }
        }
        output[tyIndex * width + txIndex] = iFiltered / wP;
    }
}

// Bilateral filter kernel
/** This function is based on the bilaternal filter demo provided by navidia,
 * in this function, the pixel value is wrapped as 16bit unsigned short
 * and accessed using global memory pointer
 *
 * Input parameters are described:
 * @param input	the memory pointer of source image data
 * @param output	the memory pointer of image data after processed
 * @param width			the width of input and output images
 * @param height			the height of input and output images
 * @param r				the radius of spatial gaussian kernel, so the actual kernel size is (2*r+1)^2
 * @param sI				the sigma of intensity gaussian filter
 * @param sS				the sigma of spatial gaussian filter
 *
 */
__global__ void gpuBiFilter16(unsigned short* input, unsigned short* output, int width, int height, int r, double sI, double sS) {
    // Initialize global Tile indices along x,y and xy
    int txIndex = __mul24(blockIdx.x, TILE_X) + threadIdx.x;
    int tyIndex = __mul24(blockIdx.y, TILE_Y) + threadIdx.y;

    // If within image size
    if ((txIndex < width) && (tyIndex < height)) {
        float iFiltered = 0;
        float wP        = 0;
        // Get the centre pixel value
        // float centrePx = tex2D(inTexture, txIndex, tyIndex);
        float centrePx = input[tyIndex * width + txIndex];

        // Iterate through filter size from centre pixel
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                // Get the current pixe; value
                float currPx = input[(tyIndex + dx) * width + (dy + txIndex)];
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                float w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian16(centrePx - currPx, sI);
                iFiltered += w * currPx;
                wP += w;
            }
        }
        output[tyIndex * width + txIndex] = iFiltered / wP;
    }
}

void bilateralFilterTexture(const Mat& srcMat, Mat& dstMat, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float costTime = 0.0f;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Create gaussain 2d array
    updateGaussian16(r, sS);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    int width = srcMat.cols, height = srcMat.rows;
    float angle = 0;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((srcMat.cols + block.x - 1) / block.x, (srcMat.rows + block.y - 1) / block.y);
    cudaEventRecord(start, 0);  // start timer
    // copy to device memory
    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        cudaMemcpyToArray(cuArray, 0, 0, (float*)(srcMat.data), width * height * sizeof(float), cudaMemcpyHostToDevice);
        texRef.addressMode[0] = cudaAddressModeWrap;
        texRef.addressMode[1] = cudaAddressModeWrap;
        // texRef.filterMode = cudaFilterModeLinear;
        texRef.filterMode = cudaFilterModePoint;
        texRef.normalized = false;

        // Bind the array to the texture reference
        cudaBindTextureToArray(texRef, cuArray, channelDesc);
        // Invoke kernel
        gpuFilterTexture<<<grid, block>>>(output, width, height, r, sI, sS);

        cudaMemcpy(dstMat.ptr(), output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&costTime, start, stop);

    printf("Time for the GPU Bilateral Filter using Texture: %f ms\n", costTime / nIter);
    clock_t endTime  = clock();
    float pixelValue = *(dstMat.ptr<float>(100) + 100);
    printf("output[100][100] = %.1f\n", pixelValue);
    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    __FUNCTION_END__;
}

void bilateralFilterTextureOpm(const Mat& srcMat, Mat& dstMat, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float costTime = 0.0f;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Create gaussain 2d array
    updateGaussian16(r, sS);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    int width = srcMat.cols, height = srcMat.rows;
    float angle = 0;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((srcMat.cols + block.x - 1) / block.x, (srcMat.rows + block.y - 1) / block.y);
    cudaEventRecord(start, 0);  // start timer
    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        // copy to device memory
        cudaMemcpyToArray(cuArray, 0, 0, (float*)(srcMat.data), width * height * sizeof(float), cudaMemcpyHostToDevice);
        texRef.addressMode[0] = cudaAddressModeWrap;
        texRef.addressMode[1] = cudaAddressModeWrap;
        // texRef.filterMode = cudaFilterModeLinear;
        texRef.filterMode = cudaFilterModePoint;
        texRef.normalized = false;

        // Bind the array to the texture reference
        cudaBindTextureToArray(texRef, cuArray, channelDesc);
        // Invoke kernel
        gpuFilterTextureOpm<<<grid, block>>>(output, width, height, r, sI, sS);
        cudaMemcpy(dstMat.ptr(), output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&costTime, start, stop);

    printf("Time for the GPU Bilateral Filter using Texture with lookup table: %f ms\n", costTime / nIter);
    clock_t endTime  = clock();
    float pixelValue = *(dstMat.ptr<float>(100) + 100);
    printf("output[100][100] = %.1f\n", pixelValue);
    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    __FUNCTION_END__;
}

void bilateralFilterOpm(const Mat& srcMat, Mat& dstMat, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float costTime = 0.0f;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Create gaussain 2d array
    updateGaussian16(r, sS);

    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // cudaArray* cuArray;
    int width = srcMat.cols, height = srcMat.rows;
    // cudaMallocArray(&cuArray, &channelDesc, width, height);
    float* input;
    cudaMalloc(&input, width * height * sizeof(float));

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((srcMat.cols + block.x - 1) / block.x, (srcMat.rows + block.y - 1) / block.y);
    cudaEventRecord(start, 0);  // start timer
    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        // copy to device memory
        cudaMemcpy(input, srcMat.ptr(), width * height * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpyToArray(cuArray, 0, 0, (float*)(srcMat.data), width * height * sizeof(float), cudaMemcpyHostToDevice);
        // texRef.addressMode[0] = cudaAddressModeWrap;
        // texRef.addressMode[1] = cudaAddressModeWrap;
        // texRef.filterMode = cudaFilterModeLinear;
        // texRef.filterMode = cudaFilterModePoint;
        // texRef.normalized = false;

        // Bind the array to the texture reference
        // cudaBindTextureToArray(texRef, cuArray, channelDesc);
        // Invoke kernel
        gpuFilterOpm<<<grid, block>>>(output, input, width, height, r, sI, sS);
        cudaMemcpy(dstMat.ptr(), output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&costTime, start, stop);

    printf("Time for the GPU Bilateral Filter using Texture with lookup table: %f ms\n", costTime / nIter);
    clock_t endTime  = clock();
    float pixelValue = *(dstMat.ptr<float>(100) + 100);
    printf("output[100][100] = %.1f\n", pixelValue);
    // Free device memory
    // cudaFreeArray(cuArray);
    cudaFree(input);
    cudaFree(output);
    __FUNCTION_END__;
}

void bilateralFilterOpenCV(const Mat& input, Mat& output, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time

    float time = 0;

    clock_t startTime = clock();
    double cvStart    = static_cast<double>(getTickCount());

    int nIter = 1;
    for (int i = 0; i < nIter; i++) {
        cv::bilateralFilter(input, output, r, sI, sS);
    }
    clock_t endTime = clock();
    double cvEnd    = static_cast<double>(getTickCount());
    time            = endTime - startTime;
    printf("Time for the OPENCV (by clock): %f ms, (by TickCount): %f ms\n", time * 1000 / CLOCKS_PER_SEC / nIter,
           (cvEnd - cvStart) / getTickFrequency() * 1000 / nIter);

    float pixelValue = *(output.ptr<float>(100) + 100);
    printf("output[100][100] = %.1f\n", pixelValue);

    __FUNCTION_END__;
}

void bilateralFilter8(const Mat& input, Mat& output, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    // Size of image
    int gray_size = input.step * input.rows;

    // Variables to allocate space for input and output GPU variables
    size_t pitch;  // Avoids bank conflicts (Read documentation for further info)
    unsigned char* input_d = NULL;
    unsigned char* output_d;

    // Create gaussain 2d array
    updateGaussian16(r, sS);

    // Allocate device memory
    cudaMallocPitch(&input_d, &pitch, sizeof(unsigned char) * input.step, input.rows);  // Find pitch

    cudaMalloc<unsigned char>(&output_d, gray_size);  // output variable

    // Creating the block size
    dim3 block(TILE_X, TILE_Y);

    // Calculate grid size to cover the whole image
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    cudaEventRecord(start, 0);  // start timer
    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        cudaMemcpy2D(input_d, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step,
                     input.rows, cudaMemcpyHostToDevice);                         // create input padded with pitch
        cudaBindTexture2D(0, inTexture, input_d, input.step, input.rows, pitch);  // bind the new padded input to texture memory
        // Kernel call

        gpuBiFilter8<<<grid, block>>>(output_d, input.cols, input.rows, r, sI, sS);

        // Copy output from device to host
        cudaMemcpy(output.ptr(), output_d, gray_size, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the GPU: %f ms\n", time / nIter);

    char pixelValue = *(output.ptr<unsigned short>(100) + 100);
    printf("output[100][100] = %d\n", pixelValue);
    // Free GPU variables
    cudaFree(input_d);
    cudaFree(output_d);
    __FUNCTION_END__;
}

void bilateralFilter16(const Mat& input, Mat& output, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float time;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Size of image
    int gray_size = input.step * input.rows;

    // Variables to allocate space for input and output GPU variables
    unsigned short *input_d = NULL, *output_d = NULL;

    // Create gaussain 1d array
    updateGaussian16(r, sS);

    cudaError_t error;

    error = cudaMalloc((void**)&input_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&output_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Creating the block size
    dim3 block(TILE_X, TILE_Y);

    // Calculate grid size to cover the whole image
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
    clock_t startTime = clock();

    cudaEventRecord(start, 0);  // start timer

    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        cudaMemcpy(input_d, input.ptr(), gray_size, cudaMemcpyHostToDevice);
        // Kernel call

        gpuBiFilter16<<<grid, block>>>(input_d, output_d, input.cols, input.rows, r, sI, sS);

        // Copy output from device to host
        cudaMemcpy(output.ptr(), output_d, gray_size, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the GPU: %f ms\n", time / nIter);
    clock_t endTime = clock();

    printf("Time for the GPU (calc by CPU1): %f ms\n", (endTime - startTime) * 1000 / CLOCKS_PER_SEC / nIter);
    unsigned short pixelValue = *(output.ptr<unsigned short>(100) + 100);
    printf("output[100][100] = %d\n", pixelValue);
    // Free GPU variables
    cudaFree(input_d);
    cudaFree(output_d);
    __FUNCTION_END__;
}

void bilateralFilter32(const Mat& input, Mat& output, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float time;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Size of image
    int gray_size = input.step * input.rows;

    // Variables to allocate space for input and output GPU variables
    float *input_d = NULL, *output_d = NULL;

    // Create gaussain 1d array
    updateGaussian16(r, sS);

    cudaError_t error;

    error = cudaMalloc((void**)&input_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&output_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy data from OpenCV input image to device memory
    // cudaMemcpy(input_d, input.ptr(), gray_size, cudaMemcpyHostToDevice);

    // Creating the block size
    dim3 block(TILE_X, TILE_Y);

    // Calculate grid size to cover the whole image
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
    clock_t startTime = clock();

    cudaEventRecord(start, 0);  // start timer

    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        cudaMemcpy(input_d, input.ptr(), gray_size, cudaMemcpyHostToDevice);
        // Kernel call

        gpuBiFilter32<<<grid, block>>>(input_d, output_d, input.cols, input.rows, r, sI, sS);

        // Copy output from device to host
        cudaMemcpy(output.ptr(), output_d, gray_size, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the GPU: %f ms\n", time / nIter);
    clock_t endTime = clock();
    printf("Time for the GPU (calc by CPU1): %f ms\n", (float)(endTime - startTime) * 1000 / CLOCKS_PER_SEC / nIter);
    unsigned short pixelValue = *(output.ptr<float>(100) + 100);
    printf("output[100][100] = %d\n", pixelValue);
    // Free GPU variables
    cudaFree(input_d);
    cudaFree(output_d);

    __FUNCTION_END__;
}

void bilateralFilterGlobalOpm(const Mat& input, Mat& output, int r, double sI, double sS) {
    __FUNCTION_START__;
    // Events to calculate gpu run time
    cudaEvent_t start, stop, hToDTransEnd, dToHTransEnd;
    float time = 0.f;
    cudaEventCreate(&start);

    cudaEventCreate(&stop);

    cudaEventCreate(&hToDTransEnd);

    cudaEventCreate(&dToHTransEnd);

    // Size of image
    int gray_size = input.step * input.rows;

    // Variables to allocate space for input and output GPU variables
    float *input_d = NULL, *output_d = NULL;

    // Create gaussain 1d array
    updateGaussian16(r, sS);

    cudaError_t error;

    error = cudaMalloc((void**)&input_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&output_d, gray_size);

    if (error != cudaSuccess) {
        printf("cudaMalloc A_d returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy data from OpenCV input image to device memory
    // cudaMemcpy(input_d, input.ptr(), gray_size, cudaMemcpyHostToDevice);

    // Creating the block size
    dim3 block(TILE_X, TILE_Y);

    // Calculate grid size to cover the whole image
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
    clock_t startTime = clock();

    cudaEventRecord(start, 0);  // start timer

    int nIter = 20;
    for (int i = 0; i < nIter; i++) {
        cudaMemcpy(input_d, input.ptr(), gray_size, cudaMemcpyHostToDevice);
        // Kernel call

        gpuBiFilterGlobalOpm<<<grid, block>>>(input_d, output_d, input.cols, input.rows, r, sI, sS);

        // Copy output from device to host
        cudaMemcpy(output.ptr(), output_d, gray_size, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);  // stop timer
    cudaEventSynchronize(stop);

    // Calculate and print kernel run time
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the GPU: %f ms\n", time / nIter);
    clock_t endTime = clock();
    printf("Time for the GPU (calc by CPU1): %f ms\n", (float)(endTime - startTime) * 1000 / CLOCKS_PER_SEC / nIter);
    unsigned short pixelValue = *(output.ptr<float>(100) + 100);
    printf("output[100][100] = %d\n", pixelValue);
    // Free GPU variables
    cudaFree(input_d);
    cudaFree(output_d);

    __FUNCTION_END__;
}
