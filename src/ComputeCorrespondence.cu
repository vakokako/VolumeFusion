#include <af/ComputeCorrespondence.cuh>

namespace af {

__global__ void warpKernel(Vec3f* warpedMesh,
                           const Vec3f* cMesh,
                           const Mat4f* cMotionGraph,
                           const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                           const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                           const std::size_t cMeshSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cMeshSize)
        return;

    auto cCurrKnnIdx     = cKnnIdxs[cIdx];
    auto cCurrKnnWeights = cKnnWeights[cIdx];

    Vec3f currVertex   = cMesh[cIdx];
    Vec3f warpedVertex = Vec3f::Zero();
    if (cCurrKnnWeights.isZero()) {
        warpedVertex = currVertex;
    } else {
        const Vec4f cInputPointHom = currVertex.homogeneous();
        for (size_t j = 0; j < Constants::motionGraphKNN; ++j) {
            warpedVertex += cCurrKnnWeights[j] * (cMotionGraph[cCurrKnnIdx[j]] * cInputPointHom).head(3);
        }
    }

    warpedMesh[cIdx] = warpedVertex;
}

void runWarpKernel(Vec3f* warpedMesh,
                   const Vec3f* cMesh,
                   const Mat4f* cMotionGraph,
                   const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                   const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                   const std::size_t cMeshSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cMeshSize);
    warpKernel<<<grid, block>>>(warpedMesh, cMesh, cMotionGraph, cKnnIdxs, cKnnWeights, cMeshSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void correspondencePairsKernel(Vec2i* corrIdxs,
                                          unsigned int* corrIdxsSize,
                                          const Vec3f* cWarpedMesh,
                                          const std::size_t cWarpedMeshSize,
                                          const Mat3f K,
                                          const float* const cPtrDepth,
                                          const Vec2i cDepthDim,
                                          const Vec3f* const cFrameMesh,
                                          const float thresholdDist) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cWarpedMeshSize)
        return;

    Vec3f vertex    = cWarpedMesh[cIdx];
    const float cFx = K(0, 0);
    const float cFy = K(1, 1);
    const float cCx = K(0, 2);
    const float cCy = K(1, 2);

    float y0 = vertex[1] / vertex[2];
    float x0 = vertex[0] / vertex[2];

    int y = static_cast<int>(y0 * cFy + cCy);
    int x = static_cast<int>(x0 * cFx + cCx);
    if (y < 0 || x < 0 || y >= cDepthDim[0] || x >= cDepthDim[1])
        return;

    int cDepthIdx = y * cDepthDim[1] + x;

    // TODO remove for a better design
    if ((cFrameMesh[cDepthIdx] - vertex).norm() > thresholdDist)
        return;

    if (cPtrDepth[cDepthIdx] > 1.e-4f) {
        unsigned int lastIndex = atomicAdd(corrIdxsSize, 1);
        corrIdxs[lastIndex]    = Vec2i(cIdx, cDepthIdx);
    }
}

void runCorrespondencePairsKernel(Vec2i* corrIdxs,
                                  unsigned int* corrIdxsSize,
                                  const Vec3f* cWarpedMesh,
                                  const std::size_t cWarpedMeshSize,
                                  const Mat3f& K,
                                  const float* const cPtrDepth,
                                  const Vec2i cDepthDim,
                                  const Vec3f* const cFrameMesh,
                                  const float thresholdDist) {
    dim3 grid, block;
    setupBlockGrid(block, grid, cWarpedMeshSize);
    correspondencePairsKernel<<<grid, block>>>(corrIdxs, corrIdxsSize, cWarpedMesh, cWarpedMeshSize, K, cPtrDepth, cDepthDim,
                                               cFrameMesh, thresholdDist);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

}  // namespace af