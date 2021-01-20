#include <af/TSDFWarp.h>

#include <af/Helper.cuh>


namespace detail {

__device__ void voxelToWorld(Vec3f& worldPoint, const Vec3i& cVoxel, const Vec3f& cVoxelSize, const Vec3f& cVolumeSize) {
    worldPoint = cVoxel.cast<float>().cwiseProduct(cVoxelSize) - cVolumeSize * 0.5f;
}
__device__ void worldToVoxel(Vec3i& voxel, const Vec3f& cWorldPoint, const Vec3f& cVoxelSize, const Vec3f& cVolumeSize) {
    Vec3f voxelSizeInv(1.0 / cVoxelSize[0], 1.0 / cVoxelSize[1], 1.0 / cVoxelSize[2]);
    Vec3f voxelF = (cWorldPoint + 0.5f * cVolumeSize).cwiseProduct(voxelSizeInv);
    voxel        = voxelF.cast<int>();
}

__device__ float weight(float radius, float distance) { return exp(-(distance * distance) / (2 * pow(radius, 2))); }


}

namespace af {
__global__ void tsdfKNNGraph(Vecui<Constants::motionGraphKNN>* knnIdxs,
                             Vecf<Constants::motionGraphKNN>* knnDists,
                             int* knnCounts,
                             const Vec3f* cGraph,
                             const float* cGraphRadiuses,
                             const std::size_t cGraphSize,
                             const Vec3f cVoxelSize,
                             const Vec3f cVolumeSize,
                             const Vec3i cVolumeDim,
                             const Vec3i cVoxelsPerNode) {
    std::size_t cVoxelId         = threadIdx.x + blockDim.x * blockIdx.x;
    std::size_t cGraphId         = threadIdx.y + blockDim.y * blockIdx.y;
    const int cVoxelsPerNode01   = cVoxelsPerNode[0] * cVoxelsPerNode[1];
    const int cVoxelsPerNodeSize = cVoxelsPerNode01 * cVoxelsPerNode[2];
    if (cVoxelId >= cVoxelsPerNodeSize || cGraphId >= cGraphSize)
        return;

    int voxlZ = cVoxelId / cVoxelsPerNode01;
    int voxlY = (cVoxelId % cVoxelsPerNode01) / cVoxelsPerNode[0];
    int voxlX = (cVoxelId % cVoxelsPerNode01) % cVoxelsPerNode[0];

    Vec3f node       = cGraph[cGraphId];
    float nodeRadius = cGraphRadiuses[cGraphId];
    Vec3i nodeVoxel;
    detail::worldToVoxel(nodeVoxel, node, cVoxelSize, cVolumeSize);
    Vec3i voxel = nodeVoxel - (cVoxelsPerNode - Vec3i::Ones()) / 2 + Vec3i(voxlX, voxlY, voxlZ);
    if (voxel[0] < 0 || voxel[1] < 0 || voxel[2] < 0 || voxel[0] >= cVolumeDim[0] || voxel[1] >= cVolumeDim[1] || voxel[2] >= cVolumeDim[2])
        return;

    Vec3f tsdfPoint;
    detail::voxelToWorld(tsdfPoint, voxel, cVoxelSize, cVolumeSize);
    float distVoxlToNode = (tsdfPoint - node).norm();
    if (distVoxlToNode > nodeRadius * 2.0f)
        return;

    int voxelGlobalId = voxel[2] * cVolumeDim[0] * cVolumeDim[1] + voxel[1] * cVolumeDim[0] + voxel[0];
    int voxelWarpId   = cGraphId * cVoxelsPerNodeSize + cVoxelId;

    int lastKnnCount = atomicAdd(knnCounts + voxelGlobalId, 1);
    if (lastKnnCount >= Constants::motionGraphKNN) {
        atomicAdd(knnCounts + voxelGlobalId, -1);
        return;
    }
    knnIdxs[voxelGlobalId][lastKnnCount]  = cGraphId;
    knnDists[voxelGlobalId][lastKnnCount] = distVoxlToNode;
}

void runTsdfKNNGraph(Vecui<Constants::motionGraphKNN>* knnIdxs,
                     Vecf<Constants::motionGraphKNN>* knnDists,
                     int* knnCounts,
                     const Vec3f* cGraph,
                     const float* cGraphRadiuses,
                     const std::size_t cGraphSize,
                     const Vec3f cVoxelSize,
                     const Vec3f cVolumeSize,
                     const Vec3i cVolumeDim,
                     const Vec3i cVoxelsPerNode) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cVoxelsPerNode.prod(), cGraphSize);
    tsdfKNNGraph<<<grid, block>>>(knnIdxs, knnDists, knnCounts, cGraph, cGraphRadiuses, cGraphSize, cVoxelSize, cVolumeSize,
                                  cVolumeDim, cVoxelsPerNode);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void computeTsdfWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* cKnnWeights,
                                             const float* cMotionGraphRadiuses,
                                             const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                             const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                             const int* cKnnCounts,
                                             const std::size_t cGridSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cGridSize)
        return;
    const int currKnnCount = cKnnCounts[cIdx];
    if (!currKnnCount)
        return;

    Vecf<Constants::motionGraphKNN> weights;
    weights.fill(0.f);

    auto cCurrKnnIdx   = cKnnIdxs[cIdx];
    auto cCurrKnnDists = cKnnDists[cIdx];
    float weightsSum   = 0;
    for (size_t j = 0; j < currKnnCount; j++) {
        weights[j] = detail::weight(cMotionGraphRadiuses[cCurrKnnIdx[j]], cCurrKnnDists[j]);
        weightsSum += weights[j];
    }

    if (weightsSum != 0.f)
        weights /= weightsSum;

    cKnnWeights[cIdx] = weights;
}

void runComputeTsdfWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* cKnnWeights,
                                     const float* cMotionGraphRadiuses,
                                     const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                     const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                     const int* cKnnCounts,
                                     const std::size_t cGridSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cGridSize);
    computeTsdfWarpWeightsKernel<<<grid, block>>>(cKnnWeights, cMotionGraphRadiuses, cKnnIdxs, cKnnDists, cKnnCounts, cGridSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void generateTsdfVerticesKernel(Vec3f* tsdfVertices,
                                              const Vec3f cVoxelSize,
                                              const Vec3f cVolumeSize,
                                              const Vec3i cVolumeDim) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    const int cY = threadIdx.y + blockDim.y * blockIdx.y;
    const int cZ = threadIdx.z + blockDim.z * blockIdx.z;
    if (cX >= cVolumeDim[0] || cY >= cVolumeDim[1] || cZ >= cVolumeDim[2])
        return;

    const int cGlobalId = cZ * cVolumeDim[0] * cVolumeDim[1] + cY * cVolumeDim[0] + cX;
    Vec3f tsdfPoint;
    detail::voxelToWorld(tsdfPoint, Vec3i(cX, cY, cZ), cVoxelSize, cVolumeSize);
    tsdfVertices[cGlobalId] = tsdfPoint;
}

void runGenerateTsdfVerticesKernel(Vec3f* tsdfVertices,
                                      const Vec3f cVoxelSize,
                                      const Vec3f cVolumeSize,
                                      const Vec3i cVolumeDim) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cVolumeDim[0], cVolumeDim[1], cVolumeDim[2]);
    generateTsdfVerticesKernel<<<grid, block>>>(tsdfVertices, cVoxelSize, cVolumeSize, cVolumeDim);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void warpTsdfKernel(Vec3f* tsdfVertices,
                               const Mat4f* cMotionGraph,
                               const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                               const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                               const int* cKnnCounts,
                               const Vec3f cVoxelSize,
                               const Vec3f cVolumeSize,
                               const Vec3i cVolumeDim) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    const int cY = threadIdx.y + blockDim.y * blockIdx.y;
    const int cZ = threadIdx.z + blockDim.z * blockIdx.z;
    if (cX >= cVolumeDim[0] || cY >= cVolumeDim[1] || cZ >= cVolumeDim[2])
        return;

    const int cGlobalId    = cZ * cVolumeDim[0] * cVolumeDim[1] + cY * cVolumeDim[0] + cX;
    const int currKnnCount = cKnnCounts[cGlobalId];
    if (!currKnnCount)
        return;

    auto cCurrKnnIdx     = cKnnIdxs[cGlobalId];
    auto cCurrKnnWeights = cKnnWeights[cGlobalId];

    if (cCurrKnnWeights.isZero())
        return;

    Vec3f warpedPoint = Vec3f::Zero();
    Vec3f tsdfPoint   = tsdfVertices[cGlobalId];
    for (size_t j = 0; j < currKnnCount; ++j) {
        warpedPoint += cCurrKnnWeights[j] * (cMotionGraph[cCurrKnnIdx[j]] * tsdfPoint.homogeneous()).head(3);
    }

    tsdfVertices[cGlobalId] = warpedPoint;
}

void runWarpTsdfKernel(Vec3f* tsdfVertices,
                       const Mat4f* cMotionGraph,
                       const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                       const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                       const int* cKnnCounts,
                       const Vec3f cVoxelSize,
                       const Vec3f cVolumeSize,
                       const Vec3i cVolumeDim) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cVolumeDim[0], cVolumeDim[1], cVolumeDim[2]);
    warpTsdfKernel<<<grid, block>>>(tsdfVertices, cMotionGraph, cKnnIdxs, cKnnWeights, cKnnCounts, cVoxelSize, cVolumeSize,
                                    cVolumeDim);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

}  // namespace af