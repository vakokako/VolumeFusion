#ifndef TSDFINTEGRATION_CUH
#define TSDFINTEGRATION_CUH

#include <af/Helper.cuh>

namespace af {

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
                                     const Mat3f cDepthIntrinsics);

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
                                           const Mat3f cDepthIntrinsics);

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
                                           const Mat3f cDepthIntrinsics);

void runCountCollisionsPerVoxelKernel(int* collisionCounts,
                                      const Vec3f* tsdfVertices,
                                      const int* cKnnCounts,
                                      const Vec3f cVoxelSize,
                                      const Vec3f cVolumeSize,
                                      const Vec3i cVolumeDim,
                                      const Mat3f cVolumeRotate,
                                      const Vec3f cVolumeTranslate);

}  // namespace af

#endif