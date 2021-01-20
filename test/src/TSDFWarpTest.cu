#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <af/Helper.cuh>

TEST(TSDFWarpTest, TSDFKNNMotionGraph) {
    const Vec3f volSize(0.8, 0.8, 0.8);
    const Vec3i volDim(256, 256, 256);
    const Vec3f voxelSize      = volSize.cwiseQuotient(volDim.cast<float>());
    const std::size_t gridSize = volDim.prod();

    thrust::device_vector<Vec3f> graph_d;
    const int cGraphSize = 10;

    float cMotionGraphMinRadius = Constants::motionGraphRadius;
    Vec3f grapMinRadius(cMotionGraphMinRadius, cMotionGraphMinRadius, cMotionGraphMinRadius);
    Vec3i radiusInVoxels  = (grapMinRadius.cwiseQuotient(voxelSize).cast<int>() + Vec3i::Ones());
    Vec3i voxelPerNodeDim = 2 * radiusInVoxels + Vec3i::Ones();
    int voxelPerNodeCount = voxelPerNodeDim.prod();

    const int cVoxelWarpSize = cGraphSize * voxelPerNodeCount;
    thrust::device_vector<Vecui<Constants::motionGraphKNN>> knnIdxs(gridSize);
    thrust::device_vector<Vecf<Constants::motionGraphKNN>> knnDists(gridSize);
    thrust::device_vector<int> knnCounts(gridSize, 0);

    dim3 block, grid;
    setupBlockGrid(block, grid, voxelPerNodeCount, cGraphSize);
    // tsdfKNNGraph<<<grid, block>>>();
}
