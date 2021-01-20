#include <gtest/gtest.h>

#include <af/CameraModel.h>
#include <af/MarchingCubes.h>
#include <af/Constants.h>
#include <af/TSDFVolume.h>
#include <af/VertexManipulation.h>
#include <af/dataset.h>

// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <iostream>

#include <vector>


// TEST(TSDFVolumeTest, Serialize) {

//     Vec3i volDim(3, 3, 3);
//     Vec3f volSize(0.2f, 0.2f, 0.2f);
//     Eigen::Matrix3f K;

//     TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, K);
//     for (int i = 0; i < 9; i++) {
//         tsdfResult->ptrTsdf()[i] = (float)i;
//     }

//     std::ofstream outFile("/home/mvankovych/Tmp/tsdfVolume.txt");
//     boost::archive::text_oarchive outArchive(outFile);
//     outArchive << tsdfResult;
// }

TEST(TSDFVolumeTest, Integrate) {

    Vec3i volDim(256, 256, 256);
    Vec3f volSize(1.5f, 1.5f, 1.5f);

    // load camera intrinsics
    af::loadIntrinsics(Constants::dataFolder + "/depthIntrinsics.txt", CameraModel::KDepth);
    std::cout << "K depth: " << std::endl
              << CameraModel::KDepth << std::endl;

    float delta = 0.02f;

    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth;


    TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, CameraModel::KDepth);
    tsdfResult->setDelta(delta);

    // load input frame
    af::loadFrame(Constants::dataFolder, 0, color, depth);
    af::filterDepth(depth, Constants::depthThreshold);

    cv::Mat vertexMap;
    af::depthToVertexMap(CameraModel::KDepth, depth, vertexMap);
    Vec3f frameCentroid = af::centroid(vertexMap);
    poseVolume.topRightCorner<3, 1>() = frameCentroid;


    if (std::ifstream(Constants::dataFolder + "/testDump/tsdfResult.txt").good()) {
        tsdfResult->load(Constants::dataFolder + "/testDump/tsdfResult.txt");
    } else {
        tsdfResult->integrate(poseVolume, color, depth);
        tsdfResult->save(Constants::dataFolder + "/testDump/tsdfResult.txt");
    }

    bool tsdfvluesExist = false;
    size_t count = 0;
    for (size_t i = 0; i < tsdfResult->tsdf().size(); ++i) {
        if (tsdfResult->tsdf()[i] > -1.f && tsdfResult->tsdf()[i] < 1.f) {
            tsdfvluesExist = true;
            ++count;
        }
    }
    EXPECT_TRUE(tsdfvluesExist);
    EXPECT_TRUE(count > 50);


    MarchingCubes mc(volDim, volSize);
    Mesh outputMesh;
    mc.computeIsoSurface(outputMesh, tsdfResult->tsdf(), tsdfResult->tsdfWeights(), tsdfResult->colorR(), tsdfResult->colorG(), tsdfResult->colorB());


}

TEST(TSDFVolumeTest, OpencvLoad) {

    cv::Mat depthIn = cv::imread("/home/mvankovych/Uni/thesis/VolumeFusion/data/minion/data/frame-000000.depth.png", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    double max, min;
    cv::minMaxLoc(depthIn, &min, &max);
    std::cout << "min: " << min << " max: " << max << "\n";
    cv::Mat depth;
    depthIn.convertTo(depth, CV_32FC1, (1.0 / 1000.0));
    cv::minMaxLoc(depth, &min, &max);
    std::cout << "depth min: " << min << " max: " << max << "\n";

    std::cout << "input type : " << depthIn.type() << "\n";
    std::cout << "depth type : " << depth.type() << "\n";
}

TEST(TSDFVolumeTest, FilterDepth) {
    cv::Mat color, depth, depthTmp;

    // load input frame
    for (int i = 0; i < 500; ++i) {
        af::loadFrame(Constants::dataFolder, i, color, depth);
        for (float threshold = 0.8f; threshold < 1.61f; threshold += 0.1f) {
            depthTmp = depth.clone();
            af::filterDepth(depthTmp, threshold);
            af::saveDepth("/home/mvankovych/Tmp/minion_threshold/" + std::to_string(i) + "_" + std::to_string(threshold), depthTmp);
        }
    }
}