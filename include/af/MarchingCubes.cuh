#ifndef MARCHINGCUBES_CUH
#define MARCHINGCUBES_CUH

#include <af/Helper.cuh>

namespace af {

void initEdgeTableDevice();

void runMarchingCubesMeshKernel(Vec3f* mesh,
                                unsigned int* meshSize,
                                const float* cTsdf,
                                const float* cTsdfWeights,
                                const Vec3i cDims,
                                const Vec3f& cVoxelSize,
                                const Vec3f& cVolSize,
                                const float cIsoValue);

void runMarchingCubesFullKernel(Vec3f* mesh,
                                unsigned int* meshSize,
                                Vec3i* faces,
                                unsigned int* facesSize,
                                const float* cTsdf,
                                const float* cTsdfWeights,
                                const Vec3i cDims,
                                const Vec3f& cVoxelSize,
                                const Vec3f& cVolSize,
                                const float cIsoValue);

}  // namespace af

#endif