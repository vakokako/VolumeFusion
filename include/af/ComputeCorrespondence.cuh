#ifndef COMPUTECORRESPONDENCE_CUH
#define COMPUTECORRESPONDENCE_CUH

#include <af/Helper.cuh>
#include <af/VertexManipulation.cuh>

#include "af/DataTypes.cuh"

namespace af {

void runWarpKernel(Vec3f* warpedMesh,
                   const Vec3f* cMesh,
                   const Mat4f* cMotionGraph,
                   const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                   const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                   const std::size_t cMeshSize);

void runCorrespondencePairsKernel(Vec2i* corrIdxs,
                                  unsigned int* corrIdxsSize,
                                  const Vec3f* cWarpedMesh,
                                  const std::size_t cWarpedMeshSize,
                                  const Mat3f& K,
                                  const float* const cPtrDepth,
                                  const Vec2i cDepthDim,
                                  const Vec3f* const cFrameMesh,
                                  const float thresholdDist);

}  // namespace af

#endif