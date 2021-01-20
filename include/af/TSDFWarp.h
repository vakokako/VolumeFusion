#ifndef TSDFWARP_H
#define TSDFWARP_H

#include <af/Constants.h>
#include <af/eigen_extension.h>

namespace af {

void runTsdfKNNGraph(Vecui<Constants::motionGraphKNN>* knnIdxs,
                     Vecf<Constants::motionGraphKNN>* knnDists,
                     int* knnCounts,
                     const Vec3f* cGraph,
                     const float* cGraphRadiuses,
                     const std::size_t cGraphSize,
                     const Vec3f cVoxelSize,
                     const Vec3f cVolumeSize,
                     const Vec3i cVolumeDim,
                     const Vec3i cVoxelsPerNode);

void runComputeTsdfWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* cKnnWeights,
                                     const float* cMotionGraphRadiuses,
                                     const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                     const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                     const int* cKnnCounts,
                                     const std::size_t cGridSize);

void runGenerateTsdfVerticesKernel(Vec3f* tsdfVertices, const Vec3f cVoxelSize, const Vec3f cVolumeSize, const Vec3i cVolumeDim);

void runWarpTsdfKernel(Vec3f* tsdfVertices,
                       const Mat4f* cMotionGraph,
                       const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                       const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                       const int* cKnnCounts,
                       const Vec3f cVoxelSize,
                       const Vec3f cVolumeSize,
                       const Vec3i cVolumeDim);

}  // namespace af

#endif