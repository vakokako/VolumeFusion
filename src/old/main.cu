#include <iostream>
#include <vector>

#include "af/eigen_extension.h"

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "af/helper.cuh"
#include "af/dataset.h"
#include "af/TSDFVolume.h"
#include "af/MarchingCubes.h"
#include <af/CameraModel.h>
#include <fstream>
#include <vector>

#define STR1(x)  #x
#define STR(x)  STR1(x)

#define _VNAME(x) #x
#define Print(x) std::cout<<_VNAME(x)<<":\n"<<x<<std::endl;

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


bool depthToVertexMap(const Mat3f &K, const cv::Mat &depth, cv::Mat &vertexMap)
{
    if (depth.type() != CV_32FC1 || depth.empty())
        return false;

    int w = depth.cols;
    int h = depth.rows;
    vertexMap = cv::Mat::zeros(h, w, CV_32FC3);
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / fx;
    float fyInv = 1.0f / fy;
    float* ptrVert = (float*)vertexMap.data;

    int tmp = 0;

    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float depthMeter = ptrDepth[y*w + x];
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;

            size_t off = (y*w + x) * 3;
            ptrVert[off] = x0 * depthMeter;
            ptrVert[off+1] = y0 * depthMeter;
            ptrVert[off+2] = depthMeter;
        }
    }

    return true;
}


Vec3f centroid(const cv::Mat &vertexMap)
{
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y)
    {
        for (int x = 0; x < vertexMap.cols; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0)
            {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centroid /= float(cnt);

    return centroid;
}



int main(int argc, char *argv[])
{
    // default input sequence in folder
    std::string dataFolder() = std::string(STR(SDF2SDF_SOURCE_DIR)) + "/data/umbrella/data";

    // parse command line parameters
    const char *params = {
        "{i|input| |input rgb-d sequence}"
        "{f|frames|551|number of frames to process (0=all)}"
        "{n|iterations|100|max number of GD iterations}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input sequence
    // download from http://campar.in.tum.de/personal/slavcheva/3d-printed-dataset/index.html
    std::string inputSequence = cmd.get<std::string>("input");
    if (inputSequence.empty())
    {
        inputSequence = dataFolder();
    }
    std::cout << "input sequence: " << inputSequence << std::endl;
    // number of frames to process
    size_t frames = (size_t)cmd.get<int>("frames");
    std::cout << "# frames: " << frames << std::endl;
    // max number of GD iterations
    size_t iterations = (size_t)cmd.get<int>("iterations");
    std::cout << "iterations: " << iterations << std::endl;

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    // load camera intrinsics
    Eigen::Matrix3f K;
    Eigen::Matrix3f Kcolor;
    if (!loadIntrinsics(inputSequence + "/depthIntrinsics.txt", K)) {
        std::cerr << "No depth intrinsics file found!" << std::endl;
        return 1;
    }
    if (!loadIntrinsics(inputSequence + "/colorIntrinsics.txt", K)) {
        std::cerr << "No color intrinsics file found!" << std::endl;
        return 1;
    }
    std::cout << "K depth: " << std::endl << K << std::endl;
    std::cout << "K color: " << std::endl << Kcolor << std::endl;

    CameraModel::KDepth = K;
    CameraModel::KColor = Kcolor;

    // create tsdf volume
    Vec3i volDim(256, 256, 256);
    Vec3f volSize(0.2f, 0.2f, 0.2f);
    //Vec3f volSize(1.0f, 1.0f, 1.0f);
    Vec3f voxelSize = volSize.cwiseQuotient(volDim.cast<float>());
    std::cout << "voxelSize: " << voxelSize.transpose() << std::endl;
    float delta = 0.02f;

    TSDFVolume* tsdfReference = new TSDFVolume(volDim, volSize, K);
    tsdfReference->setDelta(delta);
    TSDFVolume* tsdfCurrent;
    TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, K);
    tsdfResult->setDelta(delta);

    // process frames
    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth, mask;

    std::ifstream myfile(inputSequence + "/synthetic_circle_poses.txt");

    int iteration;
    std::vector<double> vec(16,0);
    Mat4f pose_gt;
    Mat4f pose_gt_global = Mat4f::Identity();

    for (size_t i = 0; i < frames; ++i)
    {
        std::cout << "Frame " << i << "..." << std::endl;

        // load input frame
        if (!loadFrame(inputSequence, i, color, depth, mask))
        {
            // std::cerr << "Frame " << i << " could not be loaded!" << std::endl;
            //return 1;
            break;
        }

        // filter depth values outside of mask
        filterDepth(mask, depth);

        // get initial volume pose from centroid of first depth map
        if (i == 0)
        {
            // initial pose for volume by computing centroid of first depth/vertex map
            cv::Mat vertMap;
            depthToVertexMap(K, depth, vertMap);
            Vec3f transCentroid = centroid(vertMap);
            poseVolume.topRightCorner<3,1>() = transCentroid;
            std::cout << "pose centroid" << std::endl << poseVolume << std::endl;
            tsdfReference->integrate(poseVolume, color, depth);
            tsdfResult->integrate(poseVolume, color, depth);

            myfile >> iteration;
            for(int i=0;i<16;++i){
                myfile>>vec[i];
                pose_gt(i/4, i%4) = vec[i];
            }
            // Print(pose_gt);
        } else {
            tsdfCurrent = new TSDFVolume(volDim, volSize, K);
            tsdfCurrent->setDelta(delta);
            tsdfCurrent->integrate(poseVolume, color, depth);

            Mat4f found_pose = findPose(poseVolume, *tsdfReference, *tsdfCurrent, 0.00005, 0.002);
            pose_gt_global *= found_pose.inverse();

            TSDFVolume* tsdf = new TSDFVolume(volDim, volSize, K);
            tsdf->setDelta(delta);
            tsdf->integrate(poseVolume, color, depth);

            construct_sdf(*tsdf, *tsdfResult, pose_gt_global);

            delete tsdf;

            TSDFVolume *temp = tsdfReference;
            tsdfReference = tsdfCurrent;
            tsdfCurrent = temp;
            delete tsdfCurrent;
        }
    }

    // extract mesh using marching cubes
    std::cout << "Extracting mesh..." << std::endl;
    MarchingCubes mc(volDim, volSize);
    Mesh outputMesh;
    mc.computeIsoSurface(outputMesh, tsdfResult->ptrTsdf(), tsdfResult->ptrTsdfWeights(), tsdfResult->ptrColorR(), tsdfResult->ptrColorG(), tsdfResult->ptrColorB());

    // save mesh
    std::cout << "Saving mesh..." << std::endl;
    const std::string meshFilename = inputSequence + "/mesh3.ply";
    if (!outputMesh.savePly(meshFilename))
    {
        std::cerr << "Could not save mesh!" << std::endl;
    }

    // clean up
    //delete tsdfCurrent;
    delete tsdfReference;
    delete tsdfResult;
    cv::destroyAllWindows();

    return 0;
}
