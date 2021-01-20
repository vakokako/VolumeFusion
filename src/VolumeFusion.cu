#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "af/VolumeFusion.h"
#include "af/ComputeCorrespondence.cuh"
#include "af/DebugHelper.h"
#include "af/GraphBuilder.h"
#include "af/Helper.cuh"
#include "af/LinearSolver.cuh"
#include "af/MarchingCubes.cuh"
#include "af/MotionGraph.h"
#include "af/MotionOptimization.cuh"
#include "af/TSDFIntegration.cuh"
#include "af/TSDFWarp.h"
#include "af/VertexManipulation.h"
#include "af/thrust_extension.cuh"
// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/text_oarchive.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "af/Algorithms.cuh"
#include "af/BilateralFilter.cuh"
#include "af/CameraModel.h"
#include "af/Constants.h"
#include "af/DataTypes.cuh"
#include "af/DepthFrameComponent.cuh"
#include "af/DeviceBuffer.cuh"
#include "af/MarchingCubes.h"
#include "af/TSDFVolume.h"
#include "af/TimeMap.h"
#include "af/dataset.h"
#include "af/eigen_extension.h"

#define USE_BUFFERS 0

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

#define CHECK_BUFF_SIZE(buffSize, maxBuffSize) ::details::checkBuffSize(buffSize, maxBuffSize, __FILE__, __LINE__)

namespace details {

void checkBuffSize(std::size_t buffSize, std::size_t maxBuffSize, std::string file = "", int line = 0) {
    if (buffSize > maxBuffSize) {
        std::cout << file + ":" + std::to_string(line) + ": error: Buffer size is too small.\n";
        throw std::runtime_error("");
    }
}

void warpMeshGPU(Vec3f* pointsPtr, std::size_t pointsCount, const MotionGraph& graph) {
    std::cout << "starting mesh warping\n";

    std::vector<Vec3f> pointsHost(pointsCount);
    // thrust::device_vector<Vec3f> warpedMesh_d(pointsCount);
    af::DeviceKnnWeighted<Constants::motionGraphKNN> meshKnn_d(pointsCount);  // 24000000 * 3

    thrust::device_vector<Mat4f> motionGraphTransforms_d(graph.graph().size());  // 640000
    thrust::device_vector<float> motionGraphRadiuses_d(graph.graph().size());    // 40000

    thrust::copy_n(thrust::device_ptr<Vec3f>(pointsPtr), pointsCount, pointsHost.begin());
    // thrust::copy_n(pointsHost.begin(), pointsCount, warpedMesh_d.begin());
    CUDA_CHECK;

    auto countNotIdentity = std::count_if(graph.transforms().begin(), graph.transforms().end(), [](const auto& mat){return !(mat.isIdentity(0.01f));});
    copyToDevice(motionGraphTransforms_d.data().get(), graph.transforms().data(), graph.graph().size());
    copyToDevice(motionGraphRadiuses_d.data().get(), graph.radiuses().data(), graph.graph().size());

    std::vector<Vecui<Constants::motionGraphKNN>> meshKnnIdxs;
    std::vector<Vecf<Constants::motionGraphKNN>> meshKnnDists;

    af::getKnnData(meshKnnIdxs, meshKnnDists, graph, pointsHost);
    copyToDevice(meshKnn_d.idxsPtr(), meshKnnIdxs.data(), pointsCount);
    copyToDevice(meshKnn_d.distsPtr(), meshKnnDists.data(), pointsCount);
    meshKnn_d.setSize(pointsCount);

    af::runComputeWarpWeightsKernel(meshKnn_d.weightsPtr(), motionGraphRadiuses_d.data().get(), meshKnn_d.idxsPtr(),
                                    meshKnn_d.distsPtr(), pointsCount);
    af::runWarpKernel(pointsPtr, pointsPtr, motionGraphTransforms_d.data().get(), meshKnn_d.idxsPtr(),
                      meshKnn_d.weightsPtr(), pointsCount);
    CUDA_CHECK;

    std::cout << "pointsCount : " << pointsCount << "\n";
    std::cout << "countNotIdentity : " << countNotIdentity << "\n";
    std::cout << "mesh warped\n";
}

}  // namespace details

namespace debug {
float meanCpu(const float* data, std::size_t size) {
    float mean(0.f);
    int count = 0;
    for (std::size_t i = 0; i < size; ++i) {
        mean += data[i];
        count++;
    }
    return mean / count;
}
float meanCpu(thrust::device_vector<float>& data_d) {
    std::vector<float> data(data_d.size());
    thrust::copy_n(data_d.begin(), data_d.size(), data.begin());
    float mean(0.f);
    int count = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        mean += data[i];
        count++;
    }
    return mean / count;
}
Vec3f centroidCpu(thrust::device_vector<Vec3f>& data_d) {
    std::vector<Vec3f> data(data_d.size());
    thrust::copy_n(data_d.begin(), data_d.size(), data.begin());
    Vec3f centroid(0, 0, 0);
    int count = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        centroid += data[i];
        if (hasPositiveDepth()(data[i]))
            count++;
    }
    return centroid / count;
}
}  // namespace debug

namespace af {

void loadProcessFrame(cv::Mat& depth,
                      thrust::device_vector<float>& depthFrame_d,
                      thrust::device_vector<Vec3f>& depthMesh_d,
                      thrust::device_vector<Vec3f>& depthMeshNormals_d,
                      Vec3f& centroid,
                      size_t index,
                      const Mat3f& cDepthIntrinsics,
                      const af::CameraDistortion& cDepthDistortion,
                      af::Settings settings) {
    af::loadDepthFrame(settings.dataFolder + "/" + settings.depthFilesPattern, index, depth);

    if (cDepthDistortion.is_distorted) {
        cv::Mat depthDistorted = depth.clone();
        cv::Mat depthIntrinsicsCV(3, 3, CV_32FC1, cv::Scalar(0.f));
        depthIntrinsicsCV.at<float>(0, 0) = cDepthIntrinsics(0, 0);
        depthIntrinsicsCV.at<float>(1, 1) = cDepthIntrinsics(1, 1);
        depthIntrinsicsCV.at<float>(0, 2) = cDepthIntrinsics(0, 2);
        depthIntrinsicsCV.at<float>(1, 2) = cDepthIntrinsics(1, 2);
        depthIntrinsicsCV.at<float>(2, 2) = cDepthIntrinsics(2, 2);
        cv::Mat depthDistortionCV(1, 8, CV_32FC1);
        depthDistortionCV.at<float>(0) = cDepthDistortion.k1;
        depthDistortionCV.at<float>(1) = cDepthDistortion.k2;
        depthDistortionCV.at<float>(2) = cDepthDistortion.p1;
        depthDistortionCV.at<float>(3) = cDepthDistortion.p2;
        depthDistortionCV.at<float>(4) = cDepthDistortion.k3;
        depthDistortionCV.at<float>(5) = cDepthDistortion.k4;
        depthDistortionCV.at<float>(6) = cDepthDistortion.k5;
        depthDistortionCV.at<float>(7) = cDepthDistortion.k6;
        cv::undistort(depthDistorted, depth, depthIntrinsicsCV, depthDistortionCV);
    }

    /////////  FRAME RPOCESSING  ////////////////

    depthFrame_d.resize(depth.total());
    depthMesh_d.resize(depth.total());
    depthMeshNormals_d.resize(depth.total());

    thrust::copy_n((float*)depth.data, depth.total(), depthFrame_d.begin());

    af::runFilterDepthKernel(depthFrame_d.data().get(), depth.total(), settings.depthThreshold);
    cudaDeviceSynchronize();

    if (settings.bilateralFiltr) {
        // Timer timer;
        // Time("cv::bilateralFilter", timer,
        //      cv::Mat filteredDepth(depth.size(), depth.type());
        //      thrust::copy_n(depthFrame_d.begin(), depth.total(), (float*)filteredDepth.data);
        //      cv::bilateralFilter(filteredDepth, depth, settings.bilateralD, settings.bilateralSigma,
        //                          settings.bilateralSigma, cv::BORDER_REPLICATE);
        //      thrust::copy_n((float*)depth.data, depth.total(), depthFrame_d.begin());
        //      af::runMaskDepthKernel(depthFrame_d.data().get(), depthFrameMask_d.data().get(), depth.total()););
        float* depthFiltered = NULL;
        cudaMalloc(&depthFiltered, depth.total() * sizeof(float));
        af::bilateralFilterTextureOpmShared(depthFiltered, depthFrame_d.data().get(), depth.rows, depth.cols, settings.bilateralD,
                                            settings.bilateralSigmaI, settings.bilateralSigmaS, settings.bilateralThreshold);
        cudaMemcpy(depthFrame_d.data().get(), depthFiltered, depth.total() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(depthFiltered);
    }

    af::runDepthToMeshKernel(depthMesh_d.data().get(), depthFrame_d.data().get(), depth.cols, depth.rows, cDepthIntrinsics);
    cudaDeviceSynchronize();
    af::runDepthVerticesNormalsKernel(depthMeshNormals_d.data().get(), depthMesh_d.data().get(), depthFrame_d.data().get(),
                                      depth.rows, depth.cols, settings.normExclAngleRange);
    af::runRemoveDepthWithoutNormalsKernel(depthMeshNormals_d.data().get(), depthMesh_d.data().get(), depthFrame_d.data().get(),
                                           depth.rows, depth.cols);
    cudaDeviceSynchronize();
    centroid = thrust::reduce(depthMesh_d.begin(), depthMesh_d.end(), Vec3f(0, 0, 0));
    cudaDeviceSynchronize();
    centroid /= thrust::count_if(thrust::device, depthMesh_d.begin(), depthMesh_d.end(), hasPositiveDepth());
    cudaDeviceSynchronize();
}

void runVolumeFusion(VolumeFusionOutput& output, const af::Settings& settings) {
    af::initEdgeTableDevice();
    Timer timer;
    TimeMap timeMap;

    const Vec3i volDim(256, 256, 256);
    const Vec3f volSize        = settings.tsdfSize;
    const Vec3f voxelSize      = volSize.cwiseQuotient(volDim.cast<float>());
    const std::size_t gridSize = volDim[0] * volDim[1] * volDim[2];

    // const float settings.energyWeightDepth = 1.f;
    // const float settings.energyWeightMReg  = sqrt(5);

    // load camera intrinsics
    Mat3f depthIntrinsics;
    af::CameraDistortion depthDistortion;
    af::loadIntrinsics(settings.dataFolder + "/depthIntrinsics.txt", depthIntrinsics);
    af::loadCameraDistortion(settings.dataFolder + "/depthDistortion.txt", depthDistortion);

    /////////  FRAME RPOCESSING  ////////////////

    cv::Mat depth;                                    // 307200
    thrust::device_vector<float> depthFrame_d;        // 1228800
    thrust::device_vector<Vec3f> depthMesh_d;         // 3686400
    thrust::device_vector<Vec3f> depthMeshNormals_d;  // 3686400
    Vec3f initFrameCentr;
    Vec3f currFrameCentr;

    // 16777216 gridSize
    thrust::device_vector<float> tsdf_d(gridSize, -1.f);                   // 67108864
    thrust::device_vector<float> tsdfWeights_d(gridSize, 0.f);             // 67108864
    thrust::device_vector<int> tsdfWarpedCollisionsCounts_d(gridSize, 0);  // 67108864
    // 1476395008 bytes
    thrust::device_vector<Vec3f> tsdfVertices_d(gridSize);                              // 201326592
    thrust::device_vector<Vecui<Constants::motionGraphKNN>> tsdfKnnIdxs_d(gridSize);    // 402653184
    thrust::device_vector<Vecf<Constants::motionGraphKNN>> tsdfKnnDists_d(gridSize);    // 402653184
    thrust::device_vector<Vecf<Constants::motionGraphKNN>> tsdfKnnWeights_d(gridSize);  // 402653184
    thrust::device_vector<int> tsdfKnnCounts_d(gridSize, 0);                            // 67108864
    CUDA_CHECK;

    std::vector<Vecui<Constants::motionGraphKNN>> meshKnnIdxs;
    std::vector<Vecf<Constants::motionGraphKNN>> meshKnnDists;
    std::vector<Vecui<Constants::energyMRegKNN>> graphKnnIdxs;
    std::vector<Vecf<Constants::energyMRegKNN>> graphKnnDists;

    output.motionGraph.clear();
    MotionGraph& graph = output.motionGraph;
    std::vector<Vec3f> meshHostCopy;

    thrust::device_vector<Vec3f> mesh_d(BUFF_SIZE_MEDIUM);  // 12000000
    thrust::device_vector<unsigned int> meshSize_d(1, 0);

    af::DeviceBufferCounted<Vec3f> meshPoints_d(BUFF_SIZE_HUGE);
    af::DeviceBufferCounted<Vec3i> meshFaces_d(BUFF_SIZE_BIG);
    CUDA_CHECK;

    thrust::device_vector<Vec3f> warpedMesh_d(BUFF_SIZE_MEDIUM);                   // 12000000
    af::DeviceKnnWeighted<Constants::motionGraphKNN> meshKnn_d(BUFF_SIZE_MEDIUM);  // 24000000 * 3
    // thrust::device_vector<Vecui<Settings::motionGraphKNN>> meshKnnIdxs_d(BUFF_SIZE_MEDIUM);    // 24000000
    // thrust::device_vector<Vecf<Settings::motionGraphKNN>> meshKnnDists_d(BUFF_SIZE_MEDIUM);    // 24000000
    // thrust::device_vector<Vecf<Settings::motionGraphKNN>> meshKnnWeights_d(BUFF_SIZE_MEDIUM);  // 24000000
    CUDA_CHECK;

    thrust::device_vector<Vec3f> motionGraph_d(BUFF_SIZE_SMALL);                             // 120000
    thrust::device_vector<Mat4f> motionGraphTransforms_d(BUFF_SIZE_SMALL);                   // 640000
    thrust::device_vector<Mat4f> motionGraphTransformsPending_d(BUFF_SIZE_SMALL);            // 640000
    thrust::device_vector<float> motionGraphRadiuses_d(BUFF_SIZE_SMALL);                     // 40000
    thrust::device_vector<Vecui<Constants::energyMRegKNN>> graphKnnIdxs_d(BUFF_SIZE_SMALL);  // 240000
    CUDA_CHECK;

    thrust::device_vector<Vec2i> corrIdxs_d(BUFF_SIZE_MEDIUM);  // 8000000
    thrust::device_vector<unsigned int> corrIdxsSize_d(1, 0);
    CUDA_CHECK;

#if USE_BUFFERS
    thrust::device_vector<Vec6f> dv_J(BUFF_SIZE_HUGE, Vec6f::Zero());  // 2400000000
    thrust::device_vector<float> dv_r(BUFF_SIZE_BIG, 0);               // 4000000
    CUDA_CHECK;

    thrust::device_vector<Vec2i> JTJContribPairs_d(BUFF_SIZE_MEDIUM);        // 8000000
    thrust::device_vector<unsigned int> JTJContribRows_d(BUFF_SIZE_MEDIUM);  // 4000000
    thrust::device_vector<unsigned int> JTJContribSize_d(1, 0);
    thrust::device_vector<Vec2i> JTrContribElems_d(BUFF_SIZE_MEDIUM);  // 8000000
    thrust::device_vector<unsigned int> JTrContribElemsSize_d(1, 0);
    CUDA_CHECK;

    thrust::device_vector<Mat6f> multipliedJTJContrPairs_d(BUFF_SIZE_MEDIUM);  // 144000000
    thrust::device_vector<Vec2i> JTJElemCoords_d(BUFF_SIZE_MEDIUM);            // 8000000
    thrust::device_vector<Vec6f> multipliedJTrElems_d(BUFF_SIZE_MEDIUM);       // 24000000
    thrust::device_vector<Vec2i> JTrElemCoords_d(BUFF_SIZE_MEDIUM);            // 8000000
    CUDA_CHECK;

    thrust::device_vector<float> JTJ_d(144000000, 0.f);                               // 576000000
    thrust::device_vector<Vec6f> JTr_d(BUFF_SIZE_SMALL, Vec6f::Zero());               // 240000
    thrust::device_vector<Vec6f> transformsUpdate_d(BUFF_SIZE_SMALL, Vec6f::Zero());  // 240000
    thrust::device_vector<int> iPiv_d(BUFF_SIZE_SMALL * 6, 0);                        // 240000

    thrust::device_vector<float> errorFuncVals_d(BUFF_SIZE_MEDIUM, 0);  // 4000000
    CUDA_CHECK;
#endif
    timeMap.addMarker("Allocation initial");

    int startFrameId = settings.framesRange.first;

    af::loadProcessFrame(depth, depthFrame_d, depthMesh_d, depthMeshNormals_d, initFrameCentr, startFrameId, depthIntrinsics,
                         depthDistortion, settings);
    cudaDeviceSynchronize();
    timeMap.addMarker("loadProcessFrame 0");
    output.centroid.load(&initFrameCentr, 1);

    timeMap.clearTimer();
    af::runIntegrateTSDFFromFrameKernel(tsdf_d.data().get(), tsdfWeights_d.data().get(), depthFrame_d.data().get(),
                                        Vec2i(depth.cols, depth.rows), settings.tsdfDelta, volDim, voxelSize, volSize,
                                        Mat3f::Identity(), initFrameCentr, depthIntrinsics);
    cudaDeviceSynchronize();
    timeMap.addMarker("Integrate depth to tsdf");

    // Start of each frame iteration

    auto handleDn = af::createHandleDn();
    for (int frameId = startFrameId + 1; frameId < settings.framesRange.second; ++frameId) {
        // stop if flag was set to false from outside
        if (!output.keepRunning.test_and_set()) {
            return;
        }

        std::cout << "----------------\n";
        std::cout << "frameId : " << frameId << "\n";
        if (frameId == settings.frameBreak) {
            output.stepper.enable(true);
        }
        output.stepper.wait_for_step("frame start");
        timeMap.clearTimer();
        meshSize_d[0]     = 0;
        corrIdxsSize_d[0] = 0;
        thrust::fill_n(tsdfKnnCounts_d.begin(), gridSize, 0);
        thrust::fill_n(tsdfWarpedCollisionsCounts_d.begin(), gridSize, 0);

#if USE_BUFFERS
        JTJContribSize_d[0]      = 0;
        JTrContribElemsSize_d[0] = 0;
#endif
        timeMap.addMarker("frame/Clearing sizes in the beginning of the loop");

        af::loadProcessFrame(depth, depthFrame_d, depthMesh_d, depthMeshNormals_d, currFrameCentr, frameId, depthIntrinsics,
                             depthDistortion, settings);
        cudaDeviceSynchronize();
        timeMap.addMarker("frame/loadProcessFrame loop");

        af::runMarchingCubesMeshKernel(mesh_d.data().get(), meshSize_d.data().get(), tsdf_d.data().get(),
                                       tsdfWeights_d.data().get(), volDim, voxelSize, volSize, 0);
        cudaDeviceSynchronize();
        timeMap.addMarker("frame/Marching cubes pointcloud");

        std::size_t tmpMeshSize = meshSize_d[0];
        CHECK_BUFF_SIZE(tmpMeshSize, BUFF_SIZE_MEDIUM);
        std::cout << "tmpMeshSize after marching cubes : " << tmpMeshSize << "\n";

        // Remove duplicates
        thrust::sort(mesh_d.begin(), mesh_d.begin() + tmpMeshSize);
        auto itEndUnique            = thrust::unique(mesh_d.begin(), mesh_d.begin() + tmpMeshSize);
        const std::size_t cMeshSize = itEndUnique - mesh_d.begin();
        CHECK_BUFF_SIZE(cMeshSize, BUFF_SIZE_MEDIUM);
        std::cout << "cMeshSize after sorting and unique : " << cMeshSize << "\n";
        timeMap.addMarker("frame/Marching cubes pointcloud remove duplicates");

        af::runTranslateKernel(mesh_d.data().get(), cMeshSize, initFrameCentr);
        timeMap.addMarker("frame/Translate pointcloud");

        meshHostCopy.resize(cMeshSize);
        copyToHost(meshHostCopy.data(), mesh_d.data().get(), cMeshSize);
        const std::size_t cGraphSizeOld = graph.graph().size();
        if (frameId == 1 || settings.updateGraph) {
            af::buildGraph(graph, meshHostCopy, Constants::motionGraphKNN, settings.motionGraphRadius);
        }
        const std::size_t cGraphSize     = graph.graph().size();
        const std::size_t cCountNewNodes = cGraphSize - cGraphSizeOld;
        CHECK_BUFF_SIZE(cGraphSize, BUFF_SIZE_SMALL);
        copyToDevice(motionGraph_d.data().get() + cGraphSizeOld, graph.graph().vec_.data() + cGraphSizeOld, cCountNewNodes);
        copyToDevice(motionGraphTransforms_d.data().get() + cGraphSizeOld, graph.transforms().data() + cGraphSizeOld,
                     cCountNewNodes);
        copyToDevice(motionGraphRadiuses_d.data().get() + cGraphSizeOld, graph.radiuses().data() + cGraphSizeOld, cCountNewNodes);
        timeMap.addMarker("frame/Build graph cpu and copy to gpu");

        // debug
        std::cout << "cCountNewNodes : " << cCountNewNodes << "\n";

        af::getKnnData(meshKnnIdxs, meshKnnDists, graph, meshHostCopy);
        copyToDevice(meshKnn_d.idxsPtr(), meshKnnIdxs.data(), cMeshSize);
        copyToDevice(meshKnn_d.distsPtr(), meshKnnDists.data(), cMeshSize);
        meshKnn_d.setSize(cMeshSize);
        timeMap.addMarker("frame/Mesh knn cpu and copy to gpu");

        af::getKnnData(graphKnnIdxs, graphKnnDists, graph, graph.graph().vec_, true);
        af::filterDisconnected(graphKnnIdxs, graphKnnDists, graph);
        copyToDevice(graphKnnIdxs_d.data().get(), graphKnnIdxs.data(), cGraphSize);
        timeMap.addMarker("frame/Graph knn cpu and copy to gpu");

        af::runComputeWarpWeightsKernel(meshKnn_d.weightsPtr(), motionGraphRadiuses_d.data().get(), meshKnn_d.idxsPtr(),
                                        meshKnn_d.distsPtr(), cMeshSize);
        af::runWarpKernel(warpedMesh_d.data().get(), mesh_d.data().get(), motionGraphTransforms_d.data().get(),
                          meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr(), cMeshSize);
        output.pointsWarped.loadFromDevice(warpedMesh_d.data().get(), cMeshSize);
        timeMap.addMarker("frame/Warp mesh");

        output.pointsFrame.loadFromDevice(depthMesh_d.data().get(), depthMesh_d.size());
        output.pointsCanon.loadFromDevice(mesh_d.data().get(), cMeshSize);
        output.pointsGraph.loadFromDevice(motionGraph_d.data().get(), cGraphSize);
        output.lines.load(graphKnnIdxs.data(), graphKnnIdxs.size());
        std::cout << "loaded initial points\n";

        for (int icpIter = 0; icpIter < settings.icpIterations; ++icpIter) {
            output.stepper.wait_for_step("icp iteration");
            output.stepper.enableStep("icp iteration", false);
            timeMap.clearTimer();
            std::cout << "icpIter : " << icpIter << "\n";
            af::runWarpKernel(warpedMesh_d.data().get(), mesh_d.data().get(), motionGraphTransforms_d.data().get(),
                              meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr(), cMeshSize);
            output.pointsWarped.loadFromDevice(warpedMesh_d.data().get(), cMeshSize);
            timeMap.clearTimer();
            corrIdxsSize_d[0] = 0;
            af::runCorrespondencePairsKernel(corrIdxs_d.data().get(), corrIdxsSize_d.data().get(), warpedMesh_d.data().get(),
                                             cMeshSize, depthIntrinsics, depthFrame_d.data().get(), Vec2i(depth.rows, depth.cols),
                                             depthMesh_d.data().get(), settings.correspThreshDist);
            const unsigned int cCorrIdxsSize = corrIdxsSize_d[0];
            CHECK_BUFF_SIZE(cCorrIdxsSize, BUFF_SIZE_MEDIUM);

            output.correspondeceCanonToFrame.loadFromDevice(corrIdxs_d.data().get(), cCorrIdxsSize);
            timeMap.addMarker("icpiter/Icp projective correspondance pairs");

            ////    MOTION OPTIMIZATION
            std::size_t jDepthHeight = cCorrIdxsSize;
            std::size_t jMRegHeight  = cGraphSize * (Constants::energyMRegKNN)*3;
            std::size_t jHeight      = jDepthHeight + jMRegHeight;
            std::size_t jWidth       = cGraphSize;

#if USE_BUFFERS
            CHECK_BUFF_SIZE(jHeight * jWidth, BUFF_SIZE_HUGE);
            CHECK_BUFF_SIZE(jHeight, BUFF_SIZE_MEDIUM);
            thrust::memset(dv_J, 0, jHeight * jWidth);
            thrust::memset(dv_r, 0, jHeight);
#else
            thrust::device_vector<Vec6f> dv_J(jHeight * jWidth, Vec6f::Zero());
            thrust::device_vector<float> dv_r(jHeight, 0);
#endif

            Vec6f* JDepth_d = dv_J.data().get();
            Vec6f* JMreg_d  = JDepth_d + jDepthHeight * jWidth;
            af::runFill_J_DepthKernel(JDepth_d, settings.energyWeightDepth, jDepthHeight, jWidth, corrIdxs_d.data().get(),
                                      cCorrIdxsSize, mesh_d.data().get(), meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr(),
                                      depthMesh_d.data().get(), depthMeshNormals_d.data().get());

            float* rDepth_d = dv_r.data().get();
            float* rMReg_d  = rDepth_d + jDepthHeight;

            timeMap.addMarker("icpiter/Fill J depth matrix");

            ///  Contributing pairs

            std::size_t maxJTJContribSizeDepth = jDepthHeight
                                                 * (af::combinations(Constants::motionGraphKNN, 2) + Constants::motionGraphKNN);
            std::size_t maxJTJContribSizeMReg     = jMRegHeight * 3;
            std::size_t maxJTJContribSize         = maxJTJContribSizeDepth + maxJTJContribSizeMReg;
            std::size_t maxJTrContribElsSizeDepth = jDepthHeight * Constants::motionGraphKNN;
            std::size_t maxJTrContribElsSizeMReg  = jMRegHeight * 2;
            std::size_t maxJTrContribElsSize      = maxJTrContribElsSizeDepth + maxJTrContribElsSizeMReg;

#if USE_BUFFERS
            CHECK_BUFF_SIZE(maxJTJContribSize, BUFF_SIZE_MEDIUM);
            CHECK_BUFF_SIZE(maxJTrContribElsSize, BUFF_SIZE_MEDIUM);
            JTJContribSize_d[0]      = 0;
            JTrContribElemsSize_d[0] = 0;
#else
            thrust::device_vector<Vec2i> JTJContribPairs_d(maxJTJContribSize);
            thrust::device_vector<unsigned int> JTJContribRows_d(maxJTJContribSize);
            thrust::device_vector<unsigned int> JTJContribSize_d(1, 0);
            thrust::device_vector<Vec2i> JTrContribElems_d(maxJTrContribElsSize);
            thrust::device_vector<unsigned int> JTrContribElemsSize_d(1, 0);
#endif
            timeMap.addMarker("icpiter/Allocation JTJ, JTr pairs");

            af::runComputeJTJAndJTrContribJDepthKernel(JTJContribPairs_d.data().get(), JTJContribRows_d.data().get(),
                                                       JTJContribSize_d.data().get(), JTrContribElems_d.data().get(),
                                                       JTrContribElemsSize_d.data().get(), corrIdxs_d.data().get(), cCorrIdxsSize,
                                                       meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr());
            timeMap.addMarker("icpiter/Compute JTJ, JTr pairs from JDepth");
            af::runComputeJTJAndJTrContribJMRegKernel(
                JTJContribPairs_d.data().get(), JTJContribRows_d.data().get(), jDepthHeight, JTJContribSize_d.data().get(),
                JTrContribElems_d.data().get(), JTrContribElemsSize_d.data().get(), cGraphSize, graphKnnIdxs_d.data().get());
            unsigned int jTJContribPairsSize = JTJContribSize_d[0];
            unsigned int jTrContribElemsSize = JTrContribElemsSize_d[0];
            timeMap.addMarker("icpiter/Compute JTJ, JTr pairs from JMReg");

            thrust::sort_by_key(JTJContribPairs_d.begin(), JTJContribPairs_d.begin() + jTJContribPairsSize,
                                JTJContribRows_d.begin());
            timeMap.addMarker("icpiter/Sort thrust JTJ pairs");

            thrust::sort(JTrContribElems_d.begin(), JTrContribElems_d.begin() + jTrContribElemsSize, compareAtDim<1>());
            timeMap.addMarker("icpiter/Sort thrust JTr pairs");

            /// PUT IN SEPARATE FUNCTION motionOptimization()

#if USE_BUFFERS
            CHECK_BUFF_SIZE(jTJContribPairsSize, BUFF_SIZE_MEDIUM);
            CHECK_BUFF_SIZE(jTrContribElemsSize, BUFF_SIZE_MEDIUM);
#else
            thrust::device_vector<Mat6f> multipliedJTJContrPairs_d(jTJContribPairsSize);
            thrust::device_vector<Vec2i> JTJElemCoords_d(jTJContribPairsSize);
            thrust::device_vector<Vec6f> multipliedJTrElems_d(jTrContribElemsSize);
            thrust::device_vector<Vec2i> JTrElemCoords_d(jTrContribElemsSize);
#endif
            timeMap.addMarker("icpiter/Allocation JTJ, Jtr multiplied");
            const unsigned int cJTJWidth  = cGraphSize * 6;
            const unsigned int cJTJHeight = cGraphSize * 6;
            const unsigned int cJTJSize   = cJTJWidth * cJTJHeight;
            const unsigned int cJTrHeight = cGraphSize;
#if USE_BUFFERS
            CHECK_BUFF_SIZE(cJTJSize, std::size_t{144000000});
            CHECK_BUFF_SIZE(cJTrHeight, BUFF_SIZE_SMALL);
            thrust::memset(JTJ_d, 0, cJTJSize);
            thrust::memset(JTr_d, 0, cJTrHeight);
            thrust::memset(errorFuncVals_d, 0, jHeight);
#else
            thrust::device_vector<float> JTJ_d(cJTJSize, 0.f);                           // 576000000
            thrust::device_vector<Vec6f> JTr_d(cJTrHeight, Vec6f::Zero());               // 240000
            thrust::device_vector<Vec6f> transformsUpdate_d(cJTrHeight, Vec6f::Zero());  // 240000
            thrust::device_vector<float> transfUpdateSquares_d(cJTrHeight * 6, 0.f);     // 240000
            thrust::device_vector<int> iPiv_d(cJTrHeight * 6, 0);                        // 240000
            thrust::device_vector<float> errorFuncVals_d(jHeight, 0);
#endif
            timeMap.addMarker("icpiter/Allocation JTJ, Jtr, transformsUpdate");

            std::cout << "loaded warped points\n";
            ///////// Info
            std::cout << "cMeshSize : " << cMeshSize << "\n";
            std::cout << "cGraphSize : " << cGraphSize << "\n";
            std::cout << "initFrameCentr : " << initFrameCentr.transpose() << "\n";
            std::cout << "jHeight : " << jHeight << "\n";
            std::cout << "jDepthHeight : " << jDepthHeight << "\n";
            std::cout << "jMRegHeight : " << jMRegHeight << "\n";
            std::cout << "jWidth : " << jWidth << "\n";
            std::cout << "maxJTJContribSize : " << maxJTJContribSize << "\n";
            std::cout << "maxJTrContribElsSize : " << maxJTrContribElsSize << "\n";
            std::cout << "jTJContribPairsSize : " << jTJContribPairsSize << "\n";
            std::cout << "jTrContribElemsSize : " << jTrContribElemsSize << "\n";

            thrust::copy_n(motionGraphTransforms_d.begin(), cGraphSize, motionGraphTransformsPending_d.begin());
            float energy = std::numeric_limits<float>::max();
            for (unsigned int iterGN = 0; iterGN < 20; ++iterGN) {
                std::cout << "iterGN : " << iterGN << "\n";
                timeMap.clearTimer();

                af::runFill_r_DepthKernel(rDepth_d, settings.energyWeightDepth, corrIdxs_d.data().get(), cCorrIdxsSize,
                                          warpedMesh_d.data().get(), depthMesh_d.data().get(), depthMeshNormals_d.data().get());

                timeMap.addMarker("gaussIter/Fill r depth");

                af::runFill_r_MRegKernel(rMReg_d, settings.energyWeightMReg, motionGraph_d.data().get(),
                                         motionGraphTransformsPending_d.data().get(), cGraphSize, graphKnnIdxs_d.data().get());

                timeMap.addMarker("gaussIter/Fill r mreg");

                runSquareKernel(errorFuncVals_d.data().get(), dv_r.data().get(), jHeight);
                float energyDepth = thrust::reduce(errorFuncVals_d.begin(), errorFuncVals_d.begin() + jDepthHeight, 0.f);
                float energyMReg  = thrust::reduce(errorFuncVals_d.begin() + jDepthHeight,
                                                  errorFuncVals_d.begin() + jDepthHeight + jMRegHeight, 0.f);
                if (energy - (energyDepth + energyMReg) < settings.energyMinStep) {
                    std::cout << "energy break : " << energyDepth + energyMReg << "\n";
                    std::cout << "energyDepth break : " << energyDepth << "\n";
                    std::cout << "energyMReg break : " << energyMReg << "\n";
                    break;
                }
                energy = energyDepth + energyMReg;

                if (iterGN != 0)
                    thrust::copy_n(motionGraphTransformsPending_d.begin(), cGraphSize, motionGraphTransforms_d.begin());

                output.pointsWarped.loadFromDevice(warpedMesh_d.data().get(), cMeshSize);

                timeMap.addMarker("gaussIter/Compute energy, copy transforms from pending");

                std::cout << "energy : " << energy << "\n";
                std::cout << "energyDepth : " << energyDepth << "\n";
                std::cout << "energyMReg : " << energyMReg << "\n";

                af::runFill_J_MRegKernel(JMreg_d, settings.energyWeightMReg, jMRegHeight, jWidth, motionGraph_d.data().get(),
                                         motionGraphTransformsPending_d.data().get(), cGraphSize, graphKnnIdxs_d.data().get());
                timeMap.addMarker("gaussIter/Fill J mreg matrix");

                // Multiplying and summing contributing pairs of JTJ
                af::runMultiplyJTJContribPairsKernel(multipliedJTJContrPairs_d.data().get(), JTJContribPairs_d.data().get(),
                                                     JTJContribRows_d.data().get(), jTJContribPairsSize, dv_J.data().get(),
                                                     jWidth);
                timeMap.addMarker("gaussIter/Multiply JTJ pairs");

                auto jTJReducedItEndPair = thrust::reduce_by_key(
                    JTJContribPairs_d.begin(), JTJContribPairs_d.begin() + jTJContribPairsSize, multipliedJTJContrPairs_d.begin(),
                    JTJElemCoords_d.begin(), multipliedJTJContrPairs_d.begin());
                const unsigned int cJTJElemsSize = jTJReducedItEndPair.first - JTJElemCoords_d.begin();
                timeMap.addMarker("gaussIter/Sum JTJ same pairs");

                // Multiplying and summing contributing elements of JTr
                af::runMultiplyJTrContribElemsKernel(multipliedJTrElems_d.data().get(), JTrContribElems_d.data().get(),
                                                     jTrContribElemsSize, dv_J.data().get(), dv_r.data().get(), jWidth);
                timeMap.addMarker("gaussIter/Multiply JTr pairs");

                auto jTrReducedItEndPair = thrust::reduce_by_key(
                    JTrContribElems_d.begin(), JTrContribElems_d.begin() + jTrContribElemsSize, multipliedJTrElems_d.begin(),
                    JTrElemCoords_d.begin(), multipliedJTrElems_d.begin(), equalAtDim<1>());
                const unsigned int cJTrElemsSize = jTrReducedItEndPair.first - JTrElemCoords_d.begin();
                timeMap.addMarker("gaussIter/Sum JTr same pairs");

                // Fill JTJ
                cudaDeviceSynchronize();
                thrust::memset(JTJ_d, 0, cJTJSize);
                runInitIdentityKernel(JTJ_d.data().get(), cJTJWidth);
                af::runFillJTJKernel(JTJ_d.data().get(), JTJElemCoords_d.data().get(), multipliedJTJContrPairs_d.data().get(),
                                     cJTJElemsSize, cJTJWidth);
                timeMap.addMarker("gaussIter/Fill JTJ matrix");

                thrust::memset(JTr_d, 0, cJTrHeight);
                af::runFillJTrKernel(JTr_d.data().get(), JTrElemCoords_d.data().get(), multipliedJTrElems_d.data().get(),
                                     cJTrElemsSize);
                timeMap.addMarker("gaussIter/Fill JTr matrix");
                cudaDeviceSynchronize();

                // debug
                // debug::print(JTJ_d, cJTJHeight, cJTJWidth, "JTJ_d" + std::to_string(frameId) + "_" + std::to_string(iterGN));

                float *JTrPtr_d, *JTJPtr_d, *transformsUpdatePtr_d;
                JTrPtr_d              = reinterpret_cast<float*>(JTr_d.data().get());
                transformsUpdatePtr_d = reinterpret_cast<float*>(transformsUpdate_d.data().get());
                JTJPtr_d              = JTJ_d.data().get();

                if (!af::linearSolverLUStable(handleDn, cJTJHeight, JTJPtr_d, cJTJHeight, JTrPtr_d, iPiv_d.data().get(),
                                              transformsUpdatePtr_d)) {
                    std::cout << "af::runVolumeFusion(): Unsolvable linear system.\n";
                    throw std::runtime_error("");
                }
                timeMap.addMarker("gaussIter/Solving Gauss Newton with LU");

                // if (settings.useCholesky) {
                //     if (!af::linSolvCholDn(handleDn, JTJ_d, JTr_d, cJTJHeight)) {
                //         std::cout << "af::runVolumeFusion(): Unsolvable linear system.\n";
                //         throw std::runtime_error("");
                //     }
                // } else {
                //     if (!af::linSolvLUDn(handleDn, JTJ_d, JTr_d, cJTJHeight)) {
                //         std::cout << "af::runVolumeFusion(): Unsolvable linear system.\n";
                //         throw std::runtime_error("");
                //     }
                // }
                cudaDeviceSynchronize();

                // runSquareKernel(transfUpdateSquares_d.data().get(),transformsUpdate_d, cJTJHeight);
                // float updateNorm = std::sqrt(thrust::reduce(transfUpdateSquares_d.begin(), transfUpdateSquares_d.end()));
                // std::cout << "updateNorm : " << updateNorm << "\n";
                // if (updateNorm < 1.e-4f) {
                //     break;
                // }

                af::runUpdateTransformsKernel(motionGraphTransformsPending_d.data().get(), transformsUpdate_d.data().get(),
                                              cJTrHeight);

                af::runWarpKernel(warpedMesh_d.data().get(), mesh_d.data().get(), motionGraphTransformsPending_d.data().get(),
                                  meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr(), cMeshSize);

                timeMap.addMarker("gaussIter/Update tranforms and warp");
            }
        }

        // af::runWarpKernel(warpedMesh_d.data().get(), mesh_d.data().get(), motionGraphTransforms_d.data().get(),
        //                   meshKnn_d._idxs.data().get(), meshKnn_d._weights.data().get(), cMeshSize);

        /// motionOptimization() END

        // TSDF Integration
        timeMap.clearTimer();

        Vec3f grapMinRadius3d;
        grapMinRadius3d.fill(settings.motionGraphRadius);
        Vec3i radiusInVoxels  = (grapMinRadius3d.cwiseQuotient(voxelSize).cast<int>() + Vec3i::Ones());
        Vec3i voxelPerNodeDim = 2 * radiusInVoxels + Vec3i::Ones();
        int voxelPerNodeCount = voxelPerNodeDim.prod();

        af::runTranslateKernel(motionGraph_d.data().get(), cGraphSize, -initFrameCentr);
        af::runTsdfKNNGraph(tsdfKnnIdxs_d.data().get(), tsdfKnnDists_d.data().get(), tsdfKnnCounts_d.data().get(),
                            motionGraph_d.data().get(), motionGraphRadiuses_d.data().get(), cGraphSize, voxelSize, volSize,
                            volDim, voxelPerNodeDim);
        timeMap.addMarker("frame/Knn for tsdf");

        af::runComputeTsdfWarpWeightsKernel(tsdfKnnWeights_d.data().get(), motionGraphRadiuses_d.data().get(),
                                            tsdfKnnIdxs_d.data().get(), tsdfKnnDists_d.data().get(), tsdfKnnCounts_d.data().get(),
                                            gridSize);
        timeMap.addMarker("frame/Knn weights for tsdf");
        af::runGenerateTsdfVerticesKernel(tsdfVertices_d.data().get(), voxelSize, volSize, volDim);
        timeMap.addMarker("frame/Tsdf generate pointcloud");

        af::runTranslateKernel(motionGraph_d.data().get(), cGraphSize, initFrameCentr);
        timeMap.addMarker("frame/Translate graph");
        af::runTranslateKernel(tsdfVertices_d.data().get(), gridSize, initFrameCentr);
        timeMap.addMarker("frame/Translate tsdf");

        af::runWarpTsdfKernel(tsdfVertices_d.data().get(), motionGraphTransforms_d.data().get(), tsdfKnnIdxs_d.data().get(),
                              tsdfKnnWeights_d.data().get(), tsdfKnnCounts_d.data().get(), voxelSize, volSize, volDim);
        timeMap.addMarker("frame/Tsdf warp");

        af::runCountCollisionsPerVoxelKernel(tsdfWarpedCollisionsCounts_d.data().get(), tsdfVertices_d.data().get(),
                                             tsdfKnnCounts_d.data().get(), voxelSize, volSize, volDim, Mat3f::Identity(),
                                             initFrameCentr);
        timeMap.addMarker("frame/Tsdf count collisions");

        if (settings.integrateTsdf) {
            af::runIntegrateWarpedTSDFFromFrameKernel(tsdf_d.data().get(), tsdfWeights_d.data().get(),
                                                      tsdfVertices_d.data().get(), tsdfWarpedCollisionsCounts_d.data().get(),
                                                      tsdfKnnCounts_d.data().get(), depthFrame_d.data().get(),
                                                      Vec2i(depth.cols, depth.rows), settings.tsdfDelta, voxelSize, volSize,
                                                      volDim, Mat3f::Identity(), initFrameCentr, depthIntrinsics);
        }
        timeMap.addMarker("frame/Tsdf integrate");

        {
            meshPoints_d.resetSize();
            meshFaces_d.resetSize();
            af::runMarchingCubesFullKernel(meshPoints_d.bufferPtr(), meshPoints_d.size_dPtr(), meshFaces_d.bufferPtr(),
                                           meshFaces_d.size_dPtr(), tsdf_d.data().get(), tsdfWeights_d.data().get(), volDim,
                                           voxelSize, volSize, 0);
            meshPoints_d.syncHostSize();
            meshFaces_d.syncHostSize();
            af::runTranslateKernel(meshPoints_d.data().get(), meshPoints_d.size(), initFrameCentr);
            timeMap.addMarker("frame/Marching cubes mesh");

            thrust::copy_n(motionGraphTransforms_d.begin(), cGraphSize, graph.transforms().begin());

            {
                std::lock_guard<std::mutex> lock(output.dataLoadMutex);
                if (settings.isMeshWarped) {
                    details::warpMeshGPU(meshPoints_d.bufferPtr(), meshPoints_d.size(), graph);
                }
                output.meshPoints.loadFromDevice(meshPoints_d.bufferPtr(), meshPoints_d.size());
                output.meshFaces.loadFromDevice(meshFaces_d.bufferPtr(), meshFaces_d.size());
            }

            std::cout << "meshPoints_d.size() : " << meshPoints_d.size() << "\n";
            std::cout << "meshFaces_d.size() : " << meshFaces_d.size() << "\n";
        }
    }
    timeMap.clearTimer();
    cusolverDnDestroy(handleDn);

    // unsigned int finalMeshSize  = meshSize_d[0];
    // unsigned int finalFacesSize = facesSize_d[0];
    // output.meshResult.vertexCloud().vec_.resize(finalMeshSize);
    // output.meshResult.colors().resize(finalMeshSize);
    // output.meshResult.faces().resize(finalFacesSize);
    // thrust::copy_n(mesh_d.begin(), finalMeshSize, output.meshResult.vertexCloud().vec_.begin());
    // thrust::copy_n(faces_d.begin(), finalFacesSize, output.meshResult.faces().begin());
    // timeMap.addMarker("Copy mesh to cpu");

    std::cout << "Time Map VolumeFusion : \n" << timeMap << "\n";
}

}  // namespace af
