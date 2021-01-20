#ifndef VERTEXMANIPULATION_CUH
#define VERTEXMANIPULATION_CUH

#include <af/Constants.h>
#include <af/eigen_extension.h>

#include "af/DataTypes.cuh"

namespace af {

void runFilterDepthKernel(float* depth, unsigned int depthSize, float threshold, float fillValue = std::numeric_limits<float>::infinity());
void runMaskDepthKernel(float* depth, const bool* cDepthMask, unsigned int depthSize);

void runDepthToMeshKernel(Vec3f* mesh, const float* cDepth, unsigned int depthWidth, unsigned int depthHeight, const Mat3f& K);

void runTranslateKernel(Vec3f* mesh, unsigned int meshSize, Vec3f translation);
void runTransformKernel(Vec3f* mesh, unsigned int meshSize, Mat4f transform);

void runComputeWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* knnWeights,
                                 const float* cMotionGraphRadiuses,
                                 const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                 const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                 const std::size_t cMeshSize);

void runDepthVerticesNormalsKernel(Vec3f* normals,
                                   Vec3f* vertices,
                                   const float* depth,
                                   int h,
                                   int w,
                                   std::pair<float, float> normExclAngleRange);

void runRemoveDepthWithoutNormalsKernel(Vec3f* normals, Vec3f* vertices, float* depth, int h, int w);

void runRejectInvalidTransforms(Mat4f* transforms, const Mat4f* validTransforms, const Vec3f* graphNodes, const std::size_t graphSize, float threshold);

}  // namespace af

#endif