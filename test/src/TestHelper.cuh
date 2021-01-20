#ifndef TESTHELPER_H
#define TESTHELPER_H

#include <gtest/gtest.h>
#include <af/dataset.h>
#include <af/Constants.h>
#include <af/VertexManipulation.h>
#include <af/Helper.cuh>

void loadFilteredFrameAndIntrinsics(Mat3f& K, cv::Mat& depth, cv::Mat& color) {

    af::loadIntrinsics(Constants::dataFolder + "/depthIntrinsics.txt", K);

    af::loadFrame(Constants::dataFolder, 0, color, depth);
    af::filterDepth(depth, Constants::depthThreshold);
}

#endif