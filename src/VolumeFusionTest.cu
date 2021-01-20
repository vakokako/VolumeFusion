#include <gtest/gtest.h>

#include "af/DataTypes.cuh"
#include "af/Settings.h"
#include "af/DepthFrameComponent.cuh"
#include "af/dataset.h"
#include "af/eigen_extension.cuh"
#include "af/DeviceTsdf.cuh"
#include "af/LinearSolver.cuh"
#include "af/VertexManipulation.cuh"
#include "af/WarpField.cuh"

#include <thrust/sort.h>
#include <thrust/unique.h>

#ifndef BUFF_SIZE_SMALL
#define BUFF_SIZE_SMALL 10000
#endif
#ifndef BUFF_SIZE_MEDIUM
#define BUFF_SIZE_MEDIUM 1000000
#endif
#ifndef BUFF_SIZE_BIG
#define BUFF_SIZE_BIG 10000000
#endif
#ifndef BUFF_SIZE_HUGE
#define BUFF_SIZE_HUGE 40000000
#endif

void testMain() {
    const Vec3f volSize(0.7, 0.7, 0.7);
    const Vec3i volDim(256, 256, 256);
    float delta = 0.01f;
    af::Settings settings;
    settings.framesRange.second = 1;

    af::CameraModel camera;

    af::DepthFrameProcessor depthPreprocessor(camera, settings);

    af::DepthFrameComponent depthFrame_d(camera);
    Vec3f initFrameCentr;
    // data
    af::DeviceBufferCounted<Vec3f> pointCloudCanon_d(BUFF_SIZE_MEDIUM);
    af::DeviceBufferCounted<Vec3f> pointCloudWarped_d(BUFF_SIZE_MEDIUM);
    af::WarpField warpField(BUFF_SIZE_SMALL, settings);
    af::DeviceKnnWeighted<Constants::motionGraphKNN> pointCloudKnn_d(BUFF_SIZE_MEDIUM);
    thrust::device_vector<Vecui<Constants::energyMRegKNN>> motionGraphKnn_d(BUFF_SIZE_SMALL);  // 240000

    af::loadIntrinsics(settings.dataFolder + "/depthIntrinsics.txt", camera.depthIntrinsics);

    depthPreprocessor.loadFrame(0, depthFrame_d);
    af::computeCentroid(depthFrame_d.pointCloud(), initFrameCentr);

    af::DeviceTsdf<> tsdf_d(volSize, volDim, delta);
    tsdf_d.setCenter(initFrameCentr);
    tsdf_d.integrate(depthFrame_d.depth());


    auto handleDn = af::createHandleDn();
    for (int frameId = 1; frameId < settings.framesRange.second; ++frameId) {
        // Important
        // meshSize_d[0]     = 0;
        // corrIdxsSize_d[0] = 0;
        // thrust::fill_n(tsdfKnnCounts_d.begin(), gridSize, 0);

        depthPreprocessor.loadFrame(frameId, depthFrame_d);

        tsdf_d.extractPointCloud(pointCloudCanon_d);

        af::runTranslateKernel(pointCloudCanon_d.bufferPtr(), pointCloudCanon_d.size(), initFrameCentr);

        warpField.spreadMotionGraph(pointCloudCanon_d);
        warpField.computeGraphKnns();
        warpField.computePointCloudWeightedKnns(pointCloudKnn_d);

        warpField.warp(pointCloudWarped_d, pointCloudCanon_d, pointCloudKnn_d);


    }
}