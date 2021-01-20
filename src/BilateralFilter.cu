#include "af/BilateralFilter.cuh"
#include "af/Helper.cuh"

#define GRAY_GASS_STEPS 30  // this parameter is used to divide 3*sigma into GRAY_GASS_STEPS sections
#define GRAY_GASS_BUF_LEN (GRAY_GASS_STEPS + 1)

namespace af {

__constant__ float cGaussian2D[400];
__constant__ int cClrGassian[GRAY_GASS_BUF_LEN];

void updateGaussian16(int r, double sd) {
    static int rLast     = 0;
    static double sdLast = 0;

    if (r == rLast && sd == sdLast) {
        return;
    }
    rLast  = r;
    sdLast = sd;

    float fGaussian[64];
    for (int i = 0; i < 2 * r + 1; i++) {
        float x      = i - r;
        fGaussian[i] = expf(-(x * x) / (2 * sd * sd));
    }

    int index = 0;
    float w   = 0;
    float fGaussian2D[400];
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            w                  = (fGaussian[dy + r] * fGaussian[dx + r]);
            fGaussian2D[index] = w;
            index += 1;
        }
    }
    cudaMemcpyToSymbol(cGaussian2D, fGaussian2D, sizeof(float) * (2 * r + 1) * (2 * r + 1));

    // calculate intensity gaussian
    float fClrGassian[GRAY_GASS_BUF_LEN];
    int nClrGassian[GRAY_GASS_BUF_LEN];
    for (int i = 0; i <= GRAY_GASS_STEPS; i++) {
        fClrGassian[i] = exp(-(pow(i / 10.0, 2)) / (2));
    }
    for (int i = 0; i <= GRAY_GASS_STEPS; i++) {
        nClrGassian[i] = (int)(fClrGassian[i] / fClrGassian[GRAY_GASS_STEPS]);
    }
    cudaMemcpyToSymbol(cClrGassian, nClrGassian, sizeof(int) * GRAY_GASS_BUF_LEN);
}

__global__ void gpuFilterTextureOpmShared(float* output, const float* input, int height, int width, int r, double sI, double sS, float thr) {
    int tx         = threadIdx.x;
    int ty         = threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + tx;
    unsigned int y = blockIdx.y * blockDim.y + ty;

    if (x >= width || y >= height) {
        return;
    }

    extern __shared__ float shared[];

    int sharedWidth  = blockDim.x + 2 * r;
    int sharedHeight = blockDim.y + 2 * r;
    int sharedSize   = sharedWidth * sharedHeight;

    int elemsPerThread = ceilf(float(sharedSize) / (blockDim.x * blockDim.y)) + 1;
    int itThread       = threadIdx.x + threadIdx.y * blockDim.x;
    for (int itShared = itThread * elemsPerThread; itShared < (itThread + 1) * elemsPerThread && itShared < sharedSize;
         ++itShared) {
        int xShared = itShared % sharedWidth;
        int yShared = itShared / sharedWidth;
        int xGlobal = xShared + (blockDim.x * blockIdx.x) - r;
        int yGlobal = yShared + (blockDim.y * blockIdx.y) - r;

        // clamping values out of matrix boundaries
        if (xGlobal < 0)
            xGlobal = 0;
        if (yGlobal < 0)
            yGlobal = 0;
        if (xGlobal >= width)
            xGlobal = width - 1;
        if (yGlobal >= height)
            yGlobal = height - 1;

        shared[itShared] = input[yGlobal * width + xGlobal];
    }

    __syncthreads();

    int kdiameter = r * 2 + 1;

    float centrePx = shared[(ty + r) * sharedWidth + tx + r];
    if (std::isinf(centrePx)) {
        output[y * width + x] = std::numeric_limits<float>::infinity();
        return;
    }

    float iFiltered = 0;
    float wP        = 0;

    int iGauss       = 0;
    float normFactor = GRAY_GASS_STEPS / (3 * sI);

    for (int a = 0; a < kdiameter; ++a) {
        for (int b = 0; b < kdiameter; ++b, ++iGauss) {
            float currPx = shared[(ty + b) * sharedWidth + tx + a];
            if (std::isinf(currPx)) {
                continue;
            }
            float diff = abs(centrePx - currPx);
            if (diff > thr) {
                continue;
            }
            int x0 = abs((diff) * normFactor);
            if (x0 > GRAY_GASS_STEPS || x0 < 0) {
                x0 = GRAY_GASS_STEPS;
            }

            float w = cGaussian2D[iGauss] * cClrGassian[x0];
            iFiltered += (currPx * w);
            wP += w;
        }
    }
    output[y * width + x] = iFiltered / wP;
}

__global__ void gpuFilterOpmSafe(float* output, const float* input, int height, int width, int r, double sI, double sS) {
    // int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    // int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float iFiltered = 0;
    float wP        = 0;
    float centrePx  = input[y * width + x];

    int index        = 0;
    float normFactor = GRAY_GASS_STEPS / (3 * sI);
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++, index++) {
            int yT = y + dy;
            int xT = x + dx;
            if (yT < 0 || yT >= height || xT < 0 || xT >= width)
                continue;

            float currPx = input[(yT)*width + xT];
            if (std::isinf(currPx)) {
                continue;
            }

            int x0       = abs((centrePx - currPx) * normFactor);
            if (x0 > GRAY_GASS_STEPS || x0 < 0) {
                x0 = GRAY_GASS_STEPS;
            }

            float w = cGaussian2D[index] * cClrGassian[x0];
            iFiltered += w * currPx;
            wP += w;
        }
    }
    output[y * width + x] = iFiltered / wP;
}

void bilateralFilterOpmSafe(float* output, const float* input, int height, int width, int r, double sI, double sS) {
    updateGaussian16(r, sS);

    dim3 block, grid;
    setupBlockGrid(block, grid, width, height);
    gpuFilterOpmSafe<<<grid, block>>>(output, input, height, width, r, sI, sS);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

void bilateralFilterTextureOpmShared(float* output, const float* input, int height, int width, int r, double sI, double sS, float thr) {
    updateGaussian16(r, sS);

    dim3 block, grid;
    setupBlockGrid(block, grid, width, height);
    std::size_t sharedSize = (block.x + 2 * r) * (block.y + 2 * r) * sizeof(float);
    gpuFilterTextureOpmShared<<<grid, block, sharedSize>>>(output, input, height, width, r, sI, sS, thr);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

}  // namespace af