#include <af/TSDFIntegration.cuh>
#include <iostream>

namespace af {

__device__ void voxelToWorld(Vec3f& worldPoint, const Vec3i& cVoxel, const Vec3f& cVoxelSize, const Vec3f& cVolumeSize) {
    worldPoint = cVoxel.cast<float>().cwiseProduct(cVoxelSize) - cVolumeSize * 0.5f;
}

__device__ void worldToVoxel(Vec3i& voxel, const Vec3f& cWorldPoint, const Vec3f& cVoxelSize, const Vec3f& cVolumeSize) {
    Vec3f voxelSizeInv(1.0 / cVoxelSize[0], 1.0 / cVoxelSize[1], 1.0 / cVoxelSize[2]);
    Vec3f voxelF = (cWorldPoint + 0.5f * cVolumeSize).cwiseProduct(voxelSizeInv);
    voxel        = voxelF.cast<int>();
}

__device__ void project3dToFrame(Vec2i& pixelCoord, const Vec3f& c3dPoint, const Mat3f& cIntrinsics) {
    float zInv    = 1.0f / c3dPoint[2];
    float uf      = (cIntrinsics(0, 0) * c3dPoint[0] * zInv) + cIntrinsics(0, 2);
    float vf      = (cIntrinsics(1, 1) * c3dPoint[1] * zInv) + cIntrinsics(1, 2);
    pixelCoord[0] = static_cast<int>(uf + 0.5f);
    pixelCoord[1] = static_cast<int>(vf + 0.5f);
}

__device__ bool updateTsdf(float* tsdf,
                           float* tsdfWeights,
                           const Vec3f& cVertex,
                           const int cIdx,
                           const float* cDepthFrame,
                           const Vec2i& cFrameDims,
                           const float cDelta,
                           const Mat3f& cDepthIntrinsics) {
    if (cVertex[2] < 1.e-4f)
        return false;

    Vec2i pixelXY;
    project3dToFrame(pixelXY, cVertex, cDepthIntrinsics);

    if (pixelXY[0] < 0 || pixelXY[0] >= cFrameDims[0] || pixelXY[1] < 0 || pixelXY[1] >= cFrameDims[1])
        return false;

    const size_t cPixelIdx = pixelXY[1] * cFrameDims[0] + pixelXY[0];
    float pixelDepth       = cDepthFrame[cPixelIdx];
    if (std::isinf(pixelDepth) || pixelDepth < 1.e-4f)
        return false;

    float sdf = cVertex[2] - pixelDepth;
    if (sdf > cDelta)
        return false;

    float tsdfNew;
    float wTsdfNew;
    if (sdf < -cDelta) {
        tsdfNew  = -1.f;
        wTsdfNew = 1.0f;
    } else {
        tsdfNew  = sdf / cDelta;  // normalize tsdf value to interval [-1.0,...,1.0]
        wTsdfNew = 1.0f;
    }

    float wTsdfOld = tsdfWeights[cIdx];
    float tsdfOld  = tsdf[cIdx];

    tsdf[cIdx]        = (tsdfOld * wTsdfOld + tsdfNew * wTsdfNew) / (wTsdfOld + wTsdfNew);
    tsdfWeights[cIdx] = std::min(wTsdfOld + wTsdfNew, 15.0f);

    return true;
}

__global__ void integrateTSDFFromFrameKernel(float* tsdf,
                                             float* tsdfWeights,
                                             const float* cDepthFrame,
                                             const Vec2i cFrameDims,
                                             const float cDelta,
                                             const Vec3i cDims,
                                             const Vec3f cVoxelSize,
                                             const Vec3f cVolumeSize,
                                             const Mat3f cVolumeRotate,
                                             const Vec3f cVolumeTranslate,
                                             const Mat3f cDepthIntrinsics) {
    std::size_t cX      = threadIdx.x + blockDim.x * blockIdx.x;
    std::size_t cY      = threadIdx.y + blockDim.y * blockIdx.y;
    std::size_t cZ      = threadIdx.z + blockDim.z * blockIdx.z;
    const int cDims01   = cDims[0] * cDims[1];
    const int cGridSize = cDims01 * cDims[2];
    if (cX >= cDims[0] || cY >= cDims[1] || cZ >= cDims[2])
        return;

    std::size_t cIdx = cZ * cDims01 + cY * cDims[0] + cX;

    Vec3i canonVoxel(cX, cY, cZ);
    Vec3f canonPointTransf;
    voxelToWorld(canonPointTransf, canonVoxel, cVoxelSize, cVolumeSize);
    canonPointTransf = cVolumeRotate * canonPointTransf + cVolumeTranslate;

    updateTsdf(tsdf, tsdfWeights, canonPointTransf, cIdx, cDepthFrame, cFrameDims, cDelta, cDepthIntrinsics);
}

void runIntegrateTSDFFromFrameKernel(float* tsdf,
                                     float* tsdfWeights,
                                     const float* cDepthFrame,
                                     const Vec2i cFrameDims,
                                     const float cDelta,
                                     const Vec3i cDims,
                                     const Vec3f cVoxelSize,
                                     const Vec3f cVolumeSize,
                                     const Mat3f cVolumeRotate,
                                     const Vec3f cVolumeTranslate,
                                     const Mat3f cDepthIntrinsics) {
    dim3 block, grid;
    // setupBlockGrid(block, grid, cDims[0], cDims[1], cDims[2]);
    setupBlockGrid(block, grid, cDims[0], cDims[1], cDims[2]);
    integrateTSDFFromFrameKernel<<<grid, block>>>(tsdf, tsdfWeights, cDepthFrame, cFrameDims, cDelta, cDims, cVoxelSize,
                                                  cVolumeSize, cVolumeRotate, cVolumeTranslate, cDepthIntrinsics);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void integrateTSDFFromCameraFrameKernel(float* tsdf,
                                                   float* tsdfWeights,
                                                   const float* cDepthFrame,
                                                   const Vec2i cFrameDims,
                                                   const float cDelta,
                                                   const Vec3i cDims,
                                                   const Vec3f cVoxelSize,
                                                   const Vec3f cVolumeSize,
                                                   const Mat3f cWorld2CameraRotate,
                                                   const Vec3f cWorld2CameraTranslate,
                                                   const Mat3f cVolumeRotate,
                                                   const Vec3f cVolumeTranslate,
                                                   const Mat3f cDepthIntrinsics) {
    std::size_t cX      = threadIdx.x + blockDim.x * blockIdx.x;
    std::size_t cY      = threadIdx.y + blockDim.y * blockIdx.y;
    std::size_t cZ      = threadIdx.z + blockDim.z * blockIdx.z;
    const int cDims01   = cDims[0] * cDims[1];
    const int cGridSize = cDims01 * cDims[2];
    if (cX >= cDims[0] || cY >= cDims[1] || cZ >= cDims[2])
        return;

    std::size_t cIdx = cZ * cDims01 + cY * cDims[0] + cX;

    Vec3i canonVoxel(cX, cY, cZ);
    Vec3f canonPointTransf;
    voxelToWorld(canonPointTransf, canonVoxel, cVoxelSize, cVolumeSize);
    canonPointTransf = cVolumeRotate * canonPointTransf + cVolumeTranslate;
    canonPointTransf = cWorld2CameraRotate * canonPointTransf + cWorld2CameraTranslate;

    updateTsdf(tsdf, tsdfWeights, canonPointTransf, cIdx, cDepthFrame, cFrameDims, cDelta, cDepthIntrinsics);
}

void runIntegrateTSDFFromCameraFrameKernel(float* tsdf,
                                           float* tsdfWeights,
                                           const float* cDepthFrame,
                                           const Vec2i cFrameDims,
                                           const float cDelta,
                                           const Vec3i cDims,
                                           const Vec3f cVoxelSize,
                                           const Vec3f cVolumeSize,
                                           const Mat3f cWorld2CameraRotate,
                                           const Vec3f cWorld2CameraTranslate,
                                           const Mat3f cVolumeRotate,
                                           const Vec3f cVolumeTranslate,
                                           const Mat3f cDepthIntrinsics) {
    dim3 block, grid;
    // setupBlockGrid(block, grid, cDims[0], cDims[1], cDims[2]);
    setupBlockGrid(block, grid, cDims[0], cDims[1], cDims[2]);
    integrateTSDFFromCameraFrameKernel<<<grid, block>>>(tsdf, tsdfWeights, cDepthFrame, cFrameDims, cDelta, cDims, cVoxelSize,
                                                        cVolumeSize, cWorld2CameraRotate, cWorld2CameraTranslate, cVolumeRotate,
                                                        cVolumeTranslate, cDepthIntrinsics);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void integrateWarpedTSDFFromFrameKernel(float* tsdf,
                                                   float* tsdfWeights,
                                                   const Vec3f* cTsdfVertices,
                                                   const int* collisionCounts,
                                                   const int* cKnnCounts,
                                                   const float* cDepthFrame,
                                                   const Vec2i cFrameDims,
                                                   const float cDelta,
                                                   const Vec3f cVoxelSize,
                                                   const Vec3f cVolumeSize,
                                                   const Vec3i cVolumeDim,
                                                   const Mat3f cVolumeRotateInverse,
                                                   const Vec3f cVolumeTranslate,
                                                   const Mat3f cDepthIntrinsics) {
    std::size_t cX      = threadIdx.x + blockDim.x * blockIdx.x;
    std::size_t cY      = threadIdx.y + blockDim.y * blockIdx.y;
    std::size_t cZ      = threadIdx.z + blockDim.z * blockIdx.z;
    const int cDims01   = cVolumeDim[0] * cVolumeDim[1];
    const int cGridSize = cDims01 * cVolumeDim[2];
    if (cX >= cVolumeDim[0] || cY >= cVolumeDim[1] || cZ >= cVolumeDim[2])
        return;

    std::size_t cIdx = cZ * cDims01 + cY * cVolumeDim[0] + cX;

    const bool isVoxelWarped = cKnnCounts[cIdx];
    if (!isVoxelWarped) {
        return;
    }
    if (!isVoxelWarped && collisionCounts[cIdx]) {
        return;
    }
    Vec3f warpedVoxelCenter = cTsdfVertices[cIdx];
    if (!isVoxelWarped) {
        updateTsdf(tsdf, tsdfWeights, warpedVoxelCenter, cIdx, cDepthFrame, cFrameDims, cDelta, cDepthIntrinsics);
        return;
    }

    Vec3f verticeInOriginCoords = cVolumeRotateInverse * (warpedVoxelCenter - cVolumeTranslate);
    Vec3i warpedVoxel;
    worldToVoxel(warpedVoxel, verticeInOriginCoords, cVoxelSize, cVolumeSize);

    if ((warpedVoxel.array() < Arr3i::Zero()).any() || (warpedVoxel.array() >= cVolumeDim.array()).any()) {
        return;
    }

    int warpedVoxelId = warpedVoxel[2] * cVolumeDim[0] * cVolumeDim[1] + warpedVoxel[1] * cVolumeDim[0] + warpedVoxel[0];
    int collisionsAtWarpedVoxel = collisionCounts[warpedVoxelId];

    if (collisionsAtWarpedVoxel > 1) {
        return;
    }

    updateTsdf(tsdf, tsdfWeights, warpedVoxelCenter, cIdx, cDepthFrame, cFrameDims, cDelta, cDepthIntrinsics);
}

void runIntegrateWarpedTSDFFromFrameKernel(float* tsdf,
                                           float* tsdfWeights,
                                           const Vec3f* cTsdfVertices,
                                           const int* collisionCounts,
                                           const int* cKnnCounts,
                                           const float* cDepthFrame,
                                           const Vec2i cFrameDims,
                                           const float cDelta,
                                           const Vec3f cVoxelSize,
                                           const Vec3f cVolumeSize,
                                           const Vec3i cVolumeDim,
                                           const Mat3f cVolumeRotate,
                                           const Vec3f cVolumeTranslate,
                                           const Mat3f cDepthIntrinsics) {
    dim3 block, grid;
    // setupBlockGrid(block, grid, cDims[0], cDims[1], cDims[2]);
    setupBlockGrid(block, grid, cVolumeDim[0], cVolumeDim[1], cVolumeDim[2]);
    integrateWarpedTSDFFromFrameKernel<<<grid, block>>>(tsdf, tsdfWeights, cTsdfVertices, collisionCounts, cKnnCounts,
                                                        cDepthFrame, cFrameDims, cDelta, cVoxelSize, cVolumeSize, cVolumeDim,
                                                        cVolumeRotate.inverse(), cVolumeTranslate, cDepthIntrinsics);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void countCollisionsPerVoxelKernel(int* collisionCounts,
                                              const Vec3f* tsdfVertices,
                                              const int* cKnnCounts,
                                              const Vec3f cVoxelSize,
                                              const Vec3f cVolumeSize,
                                              const Vec3i cVolumeDim,
                                              const Mat3f cVolumeRotateInverse,
                                              const Vec3f cVolumeTranslate) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    const int cY = threadIdx.y + blockDim.y * blockIdx.y;
    const int cZ = threadIdx.z + blockDim.z * blockIdx.z;
    if (cX >= cVolumeDim[0] || cY >= cVolumeDim[1] || cZ >= cVolumeDim[2])
        return;

    const int cGlobalId    = cZ * cVolumeDim[0] * cVolumeDim[1] + cY * cVolumeDim[0] + cX;
    const int currKnnCount = cKnnCounts[cGlobalId];
    if (!currKnnCount) {
        return;
    }

    Vec3f warpedVoxelCenter     = tsdfVertices[cGlobalId];
    Vec3f verticeInOriginCoords = cVolumeRotateInverse * (warpedVoxelCenter - cVolumeTranslate);
    Vec3i warpedVoxel;
    worldToVoxel(warpedVoxel, verticeInOriginCoords, cVoxelSize, cVolumeSize);

    if ((warpedVoxel.array() < Arr3i::Zero()).any() || (warpedVoxel.array() >= cVolumeDim.array()).any()) {
        return;
    }

    int voxelGlobalId = warpedVoxel[2] * cVolumeDim[0] * cVolumeDim[1] + warpedVoxel[1] * cVolumeDim[0] + warpedVoxel[0];
    atomicAdd(collisionCounts + voxelGlobalId, 1);
}

void runCountCollisionsPerVoxelKernel(int* collisionCounts,
                                      const Vec3f* tsdfVertices,
                                      const int* cKnnCounts,
                                      const Vec3f cVoxelSize,
                                      const Vec3f cVolumeSize,
                                      const Vec3i cVolumeDim,
                                      const Mat3f cVolumeRotate,
                                      const Vec3f cVolumeTranslate) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cVolumeDim[0], cVolumeDim[1], cVolumeDim[2]);
    countCollisionsPerVoxelKernel<<<grid, block>>>(collisionCounts, tsdfVertices, cKnnCounts, cVoxelSize, cVolumeSize, cVolumeDim,
                                                   cVolumeRotate.inverse(), cVolumeTranslate);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

}  // namespace af