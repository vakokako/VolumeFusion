
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <thread>

#include "af/Algorithms.cuh"
#include "af/AzureReader.h"
#include "af/BilateralFilter.cuh"
#include "af/CameraCalibration.h"
#include "af/CameraModel.h"
#include "af/DebugHelper.h"
#include "af/Settings.h"
#include "af/DepthFrameComponent.cuh"
#include "af/DepthReader.cuh"
#include "af/DeviceBuffer.cuh"
#include "af/DeviceTsdf.cuh"
#include "af/PointCloudStream.cuh"
#include "af/TimeMap.h"
#include "af/dataset.h"

namespace af {

void startPointCloudStream(PointCloudBuffers& buffers, const af::Settings& settings) {
    // af::initEdgeTableDevice();
    [[maybe_unused]] const std::size_t cBufferSizeSmall = 10000;
    const std::size_t cBufferSizeMedium                 = 1000000;
    const std::size_t cBufferSizeBig                    = 10000000;
    const std::size_t cBufferSizeHuge                   = 40000000;

    TimeMap timeMap;

    // Setup

    af::CameraDistortion depthDistortion;
    af::CameraModel depthCamera;
    af::DeviceDepthImage depthFrame_d(depthCamera);
    af::cuda::DepthReader<af::AzureDepthReader> depthReader(settings.dataFolder + "/" + settings.depthFilesPattern);
    thrust::device_vector<Vec3f> framePointCloud_d;

    // Algorithm

    af::loadIntrinsics(settings.dataFolder + "/depthIntrinsics.txt", depthCamera.depthIntrinsics);
    af::loadCameraDistortion(settings.dataFolder + "/depthDistortion.txt", depthDistortion);

    std::cout << "depth intrinsics : " << depthCamera.depthIntrinsics << "\n";

    depthCamera.intrinsics.cx = depthCamera.depthIntrinsics(0, 0);
    depthCamera.intrinsics.cy = depthCamera.depthIntrinsics(1, 1);
    depthCamera.intrinsics.fx = depthCamera.depthIntrinsics(0, 2);
    depthCamera.intrinsics.fy = depthCamera.depthIntrinsics(1, 2);
    depthCamera.distortion    = depthDistortion;

    cv::Mat depthIntrinsicsCV(3, 3, CV_32FC1, cv::Scalar(0.f));
    cv::Mat depthDistortionCV(1, 8, CV_32FC1);
    cv::Mat E = cv::Mat::eye(3, 3, cv::DataType<float>::type);
    cv::Mat map1;
    cv::Mat map2;

    depthIntrinsicsCV.at<float>(0, 0) = depthCamera.intrinsics.cx;
    depthIntrinsicsCV.at<float>(1, 1) = depthCamera.intrinsics.cy;
    depthIntrinsicsCV.at<float>(0, 2) = depthCamera.intrinsics.fx;
    depthIntrinsicsCV.at<float>(1, 2) = depthCamera.intrinsics.fy;
    depthIntrinsicsCV.at<float>(2, 2) = 1.f;
    depthDistortionCV.at<float>(0)    = depthCamera.distortion.k1;
    depthDistortionCV.at<float>(1)    = depthCamera.distortion.k2;
    depthDistortionCV.at<float>(2)    = depthCamera.distortion.p1;
    depthDistortionCV.at<float>(3)    = depthCamera.distortion.p2;
    depthDistortionCV.at<float>(4)    = depthCamera.distortion.k3;
    depthDistortionCV.at<float>(5)    = depthCamera.distortion.k4;
    depthDistortionCV.at<float>(6)    = depthCamera.distortion.k5;
    depthDistortionCV.at<float>(7)    = depthCamera.distortion.k6;

    depthReader.hostReader().setCurrentIndex(settings.framesRange.first - 2);
    depthReader.readNext(depthFrame_d);

    depthCamera.width  = depthFrame_d.width();
    depthCamera.height = depthFrame_d.height();

    cv::Mat undistortedDepthCV(depthCamera.height, depthCamera.width, CV_32FC1);
    cv::Mat depth_xy_table(depthCamera.height, depthCamera.width, CV_32FC2);
    af::create_xy_lookup_table(depthCamera, depth_xy_table);
    thrust::device_vector<float2> xyLookupTable_d(depthCamera.height * depthCamera.width);
    thrust::copy_n((float2*)depth_xy_table.data, depth_xy_table.total(), xyLookupTable_d.begin());

    // debug
    cv::FileStorage file("xy_lookup_table.xml", cv::FileStorage::WRITE);
    file << "xy_lookup_table" << depth_xy_table;

    for (int i = settings.framesRange.first; i < settings.framesRange.second; ++i) {
        if (!buffers._keepRunning.test_and_set()) {
            std::cout << "Direct mesh reconstruction aborted!\n";
            return;
        }
        std::cout << "processing frame " << depthReader.hostReader().currentIndex() + 1 << "\n";

        buffers._stepper.wait_for_step("frame start");
        timeMap.clearTimer();

        depthReader.readNext(depthFrame_d);

        if (depthCamera.distortion.is_distorted) {
            undistortedDepthCV = cv::Mat(depthReader.hostBuffer().size(), depthReader.hostBuffer().type());

            cv::initUndistortRectifyMap(depthIntrinsicsCV, depthDistortionCV, E, depthIntrinsicsCV,
                                        {undistortedDepthCV.cols, undistortedDepthCV.rows}, CV_32FC1, map1, map2);
            cv::remap(depthReader.hostBuffer(), undistortedDepthCV, map1, map2, cv::INTER_NEAREST, CV_HAL_BORDER_CONSTANT);

            thrust::copy_n((float*)undistortedDepthCV.data, undistortedDepthCV.total(), depthFrame_d.begin());
        }

        // depthFrame_d.nullifyUpperBound(settings.depthThreshold);
        // if (settings.bilateralFiltr) {
        //     thrust::device_vector<float> depthFiltered_d(depthFrame_d.size());
        //     af::bilateralFilterTextureOpmShared(depthFiltered_d.data().get(), depthFrame_d.dataPtr(), depthFrame_d.height(),
        //                                         depthFrame_d.width(), settings.bilateralD, settings.bilateralSigmaI,
        //                                         settings.bilateralSigmaS, settings.bilateralThreshold);
        //     thrust::copy(depthFiltered_d.begin(), depthFiltered_d.end(), depthFrame_d.begin());
        // }
        timeMap.addMarker("Depth reading");

        depthFrame_d.backProject(framePointCloud_d);
        timeMap.addMarker("Backprojection");

        {
            std::lock_guard<std::mutex> lock(buffers._dataLoadMutex);

            buffers._pointsFrame.loadFromDevice(framePointCloud_d.data().get(), framePointCloud_d.size());
        }

        timeMap.addMarker("Copying to buffers");
    }

    std::cout << "Time Map : \n" << timeMap << "\n";
}

}  // namespace af