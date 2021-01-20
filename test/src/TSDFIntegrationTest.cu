#include <af/dataset.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <af/Helper.cuh>
#include <af/TSDFIntegration.cuh>
#include <af/TestHelper.cuh>
#include <af/TSDFVolume.h>
#include <af/Constants.h>
#include <fstream>

TEST(TSDFIntegrationTest, CompareGPUvsCPU) {
    Timer timer;

    Mat3f K;
    cv::Mat color, depth;
    loadFilteredFrameAndIntrinsics(K, depth, color);
    std::size_t frameSize = depth.rows * depth.cols;

    cv::Mat vertexMap;
    af::depthToVertexMap(K, depth, vertexMap);
    Mat4f poseVolume                  = Mat4f::Identity();
    Vec3f centroidPoint               = af::centroid(vertexMap);
    poseVolume.topRightCorner<3, 1>() = centroidPoint;

    Vec3i volDim(256, 256, 256);
    Vec3f volSize(1.5f, 1.5f, 1.5f);
    Vec3f voxlSize       = volSize.cwiseQuotient(volDim.cast<float>());
    std::size_t gridSize = volDim[0] * volDim[1] * volDim[2];
    float delta          = 0.02f;

    TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, K);
    tsdfResult->setDelta(delta);
    if (std::ifstream(Constants::dataFolder + "/testDump/tsdfResult.txt").good()) {
        tsdfResult->load(Constants::dataFolder + "/testDump/tsdfResult.txt");
    } else {
        tsdfResult->integrate(poseVolume, color, depth);
        tsdfResult->save(Constants::dataFolder + "/testDump/tsdfResult.txt");
    }

    thrust::device_vector<float> tsdf_d(gridSize, -1.f);
    thrust::device_vector<float> tsdfWeights_d(gridSize, 0.f);
    thrust::device_vector<float> depthFrame_d(frameSize);
    thrust::copy_n((float*)depth.data, frameSize, depthFrame_d.begin());

    Time("runIntegrateTSDFFromFrameKernel", timer,
         af::runIntegrateTSDFFromFrameKernel(tsdf_d.data().get(), tsdfWeights_d.data().get(), depthFrame_d.data().get(),
                                             Vec2i(depth.cols, depth.rows), delta, volDim, voxlSize, volSize, Mat3f::Identity(),
                                             centroidPoint, K););

    std::vector<float> tsdfKernel(gridSize);
    std::vector<float> tsdfWeightsKernel(gridSize);
    thrust::copy(tsdf_d.begin(), tsdf_d.end(), tsdfKernel.begin());
    thrust::copy(tsdfWeights_d.begin(), tsdfWeights_d.end(), tsdfWeightsKernel.begin());

    for (std::size_t i = 0; i < gridSize; ++i) {
        ASSERT_FLOAT_EQ(tsdfKernel[i], tsdfResult->tsdf()[i]);
        ASSERT_FLOAT_EQ(tsdfWeightsKernel[i], tsdfResult->tsdfWeights()[i]);
    }
}