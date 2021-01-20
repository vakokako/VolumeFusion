#include <af/CameraModel.h>
#include <af/DebugHelper.h>
#include <af/GraphBuilder.h>
#include <af/VertexManipulation.h>
#include <af/dataset.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <af/ComputeCorrespondence.cuh>
#include <af/DataTypes.cuh>
#include <af/TSDFIntegration.cuh>
#include <af/TestHelper.cuh>
#include <af/VertexManipulation.cuh>

#include "MarchingCubes.cu"

TEST(VertexManipulationTest, depthVerticesNormals) {
    Timer timer;

    Mat3f K;
    cv::Mat color, depth;
    loadFilteredFrameAndIntrinsics(K, depth, color);

    cv::Mat vertexMap;
    af::depthToVertexMap(K, depth, vertexMap);

    Time("normals : ", timer, cv::Mat normals = cv::Mat::zeros(vertexMap.rows, vertexMap.cols, CV_32FC3);
         af::normals(vertexMap, normals););

    float* depth_d     = allocateDeviceArrayCopy((float*)depth.data, depth.total());
    Vec3f* vertexMap_d = allocateDeviceArrayCopy((Vec3f*)vertexMap.data, vertexMap.total());
    Vec3f* normals_d   = allocateDeviceArray<Vec3f>(vertexMap.total());
    Time("normals kernel : ", timer,
         af::runDepthVerticesNormalsKernel(normals_d, vertexMap_d, depth_d, vertexMap.rows, vertexMap.cols, {1.3, 2.1}););

    cv::Mat normalsCuda = cv::Mat::zeros(vertexMap.rows, vertexMap.cols, CV_32FC3);
    copyToHost((Vec3f*)normalsCuda.data, normals_d, normalsCuda.total());

    int nonZeroCount = 0;
    for (std::size_t i = 0; i < normals.rows; ++i) {
        for (std::size_t j = 0; j < normals.cols; ++j) {
            if (normalsCuda.at<Vec3f>(i, j).norm() != 0)
                ++nonZeroCount;

            EXPECT_NEAR(normals.at<Vec3f>(i, j)[0], normalsCuda.at<Vec3f>(i, j)[0], 1.e-05f);
            EXPECT_NEAR(normals.at<Vec3f>(i, j)[1], normalsCuda.at<Vec3f>(i, j)[1], 1.e-05f);
            EXPECT_NEAR(normals.at<Vec3f>(i, j)[2], normalsCuda.at<Vec3f>(i, j)[2], 1.e-05f);
        }
    }
    EXPECT_TRUE(nonZeroCount != 0);
}

TEST(VertexManipulationTest, FilterDepthGPUCPU) {
    Timer timer;

    cv::Mat color, depth;
    af::loadFrame(Constants::dataFolder, 0, color, depth);

    thrust::device_vector<float> depth_d(depth.total());
    thrust::copy_n((float*)depth.data, depth.total(), depth_d.begin());

    Time("filterDepth", timer, af::filterDepth(depth, Constants::depthThreshold););
    Time("runFilterDepthKernel", timer,
         af::runFilterDepthKernel(depth_d.data().get(), depth_d.size(), Constants::depthThreshold););

    std::vector<float> depthKernel(depth_d.size());
    thrust::copy_n(depth_d.begin(), depth_d.size(), depthKernel.begin());

    for (std::size_t i = 0; i < depth.total(); ++i) {
        ASSERT_FLOAT_EQ(((float*)depth.data)[i], depthKernel[i]);
    }
}

TEST(VertexManipulationTest, DepthToMeshCPUGPU) {
    Timer timer;

    Mat3f K;
    cv::Mat color, depth;
    loadFilteredFrameAndIntrinsics(K, depth, color);

    thrust::device_vector<float> depth_d(depth.total());
    thrust::device_vector<Vec3f> mesh_d(depth.total());
    thrust::copy_n((float*)depth.data, depth.total(), depth_d.begin());

    cv::Mat vertexMap;
    Time("depthToVertexMap", timer, af::depthToVertexMap(K, depth, vertexMap););
    Time("runDepthToMeshKernel", timer,
         af::runDepthToMeshKernel(mesh_d.data().get(), depth_d.data().get(), depth.cols, depth.rows, K););

    std::vector<Vec3f> meshKernel(mesh_d.size());
    thrust::copy_n(mesh_d.begin(), mesh_d.size(), meshKernel.begin());

    for (std::size_t i = 0; i < depth.total(); ++i) {
        ASSERT_NEAR((((Vec3f*)vertexMap.data)[i] - meshKernel[i]).norm(), 0, 1.e-07);
    }
}

TEST(VertexManipulationTest, CentroidCPUGPU) {
    Timer timer;

    Mat3f K;
    cv::Mat color, depth;
    loadFilteredFrameAndIntrinsics(K, depth, color);

    thrust::device_vector<float> depth_d(depth.total());
    thrust::device_vector<Vec3f> mesh_d(depth.total());
    thrust::copy_n((float*)depth.data, depth.total(), depth_d.begin());

    cv::Mat vertexMap;
    Time("depthToVertexMap", timer, af::depthToVertexMap(K, depth, vertexMap););
    Time("runDepthToMeshKernel", timer,
         af::runDepthToMeshKernel(mesh_d.data().get(), depth_d.data().get(), depth.cols, depth.rows, K););

    std::vector<Vec3f> meshKernel(mesh_d.size());
    thrust::copy_n(mesh_d.begin(), mesh_d.size(), meshKernel.begin());

    for (std::size_t i = 0; i < depth.total(); ++i) {
        ASSERT_NEAR((((Vec3f*)vertexMap.data)[i] - meshKernel[i]).norm(), 0, 1.e-07);
    }

    Vec3f centrDbg;
    size_t cntDbg;
    Vec3f centroid = af::centroidDBG(vertexMap, cntDbg, centrDbg);
    std::cout << "centrDbg : " << centrDbg.transpose() << "\n";
    std::cout << "cntDbg : " << cntDbg << "\n";

    Vec3f centroidCuda = thrust::reduce(mesh_d.begin(), mesh_d.end(), Vec3f(0.f, 0.f, 0.f));
    float countCuda    = thrust::count_if(thrust::device, mesh_d.begin(), mesh_d.end(), hasPositiveDepth());
    std::cout << "centroidCuda : " << centroidCuda.transpose() << "\n";
    std::cout << "countCuda : " << countCuda << "\n";

    centroidCuda /= countCuda;

    std::cout << "centroid : " << centroid.transpose() << "\n";
    std::cout << "centroidCuda : " << centroidCuda.transpose() << "\n";
    std::cout << "centroidCudaNoThrust : " << af::centroid(meshKernel).transpose() << "\n";
}

TEST(VertexManipulationTest, Warp) {
    af::initEdgeTableDevice();
    Timer timer;

    const Vec3i volDim(256, 256, 256);
    const Vec3f volSize(1.5f, 1.5f, 1.5f);
    const Vec3f voxelSize      = volSize.cwiseQuotient(volDim.cast<float>());
    const std::size_t gridSize = volDim[0] * volDim[1] * volDim[2];
    const float delta          = 0.02f;

    const float energyWeightDepth = 1;
    const float energyWeightMReg  = sqrt(5);

    // load camera intrinsics
    af::loadIntrinsics(Constants::dataFolder + "/depthIntrinsics.txt", CameraModel::KDepth);

    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth;
    af::loadFrame(Constants::dataFolder, 0, color, depth);

    /////////  FRAME RPOCESSING  ////////////////

    thrust::device_vector<float> depthFrame_d(depth.total());
    thrust::copy_n((float*)depth.data, depth.total(), depthFrame_d.begin());

    af::runFilterDepthKernel(depthFrame_d.data().get(), depthFrame_d.size(), Constants::depthThreshold);

    thrust::device_vector<Vec3f> depthMesh_d(depthFrame_d.size());
    thrust::device_vector<Vec3f> depthMeshNormals_d(depth.total());

    af::runDepthToMeshKernel(depthMesh_d.data().get(), depthFrame_d.data().get(), depth.cols, depth.rows, CameraModel::KDepth);
    af::runDepthVerticesNormalsKernel(depthMeshNormals_d.data().get(), depthMesh_d.data().get(), depthFrame_d.data().get(),
                                      depth.rows, depth.cols, {1.3, 2.1});
    Vec3f frameCentroid = thrust::reduce(depthMesh_d.begin(), depthMesh_d.end(), Vec3f(0, 0, 0));
    frameCentroid /= thrust::count_if(thrust::device, depthMesh_d.begin(), depthMesh_d.end(), hasPositiveDepth());

    thrust::device_vector<float> tsdf_d(gridSize, -1.f);
    thrust::device_vector<float> tsdfWeights_d(gridSize, 0.f);

    af::runIntegrateTSDFFromFrameKernel(tsdf_d.data().get(), tsdfWeights_d.data().get(), depthFrame_d.data().get(),
                                        Vec2i(depth.cols, depth.rows), delta, volDim, voxelSize, volSize, Mat3f::Identity(),
                                        frameCentroid, CameraModel::KDepth);

    thrust::device_vector<Vec3f> meshBuff_d(100000);
    thrust::device_vector<unsigned int> meshSize_d(1, 0);
    af::runMarchingCubesMeshKernel(meshBuff_d.data().get(), meshSize_d.data().get(), tsdf_d.data().get(),
                                   tsdfWeights_d.data().get(), volDim, voxelSize, volSize, 0);

    // Remove duplicates
    thrust::sort(meshBuff_d.begin(), meshBuff_d.end());
    auto itEndUnique = thrust::unique(meshBuff_d.begin(), meshBuff_d.end());
    meshSize_d[0]    = itEndUnique - meshBuff_d.begin();

    const std::size_t cMeshSize = meshSize_d[0];

    af::runTranslateKernel(meshBuff_d.data().get(), meshSize_d[0], frameCentroid);

    MotionGraph graph;
    std::vector<Vec3f> meshHostCopy(cMeshSize);
    thrust::copy_n(meshBuff_d.begin(), cMeshSize, meshHostCopy.begin());
    af::buildGraph(graph, meshHostCopy, Constants::motionGraphKNN, Constants::motionGraphRadius);

    std::vector<Vecui<Constants::motionGraphKNN>> meshKnnIdxs;
    std::vector<Vecf<Constants::motionGraphKNN>> meshKnnDists;
    af::getKnnData(meshKnnIdxs, meshKnnDists, graph, meshHostCopy);

    std::vector<Vecui<Constants::energyMRegKNN>> graphKnnIdxs;
    std::vector<Vecf<Constants::energyMRegKNN>> graphKnnDists;
    af::getKnnData(graphKnnIdxs, graphKnnDists, graph, graph.graph().vec_);

    const std::size_t cGraphSize = graph.graph().size();
    thrust::device_vector<Vec3f> warpedMesh_d(cMeshSize);
    af::DeviceKnnWeighted<Constants::motionGraphKNN> meshKnn_d(cMeshSize);

    copyToDevice(meshKnn_d.idxsPtr(), meshKnnIdxs.data(), cMeshSize);
    copyToDevice(meshKnn_d.distsPtr(), meshKnnDists.data(), cMeshSize);

    thrust::device_vector<Vec3f> motionGraph_d                           = graph.graph().vec_;
    thrust::device_vector<Mat4f> motionGraphTransforms_d                 = graph.transforms();
    thrust::device_vector<float> motionGraphRadiuses_d                   = graph.radiuses();
    thrust::device_vector<Vecui<Constants::energyMRegKNN>> graphKnnIdxs_d = graphKnnIdxs;

    af::runComputeWarpWeightsKernel(meshKnn_d.weightsPtr(), motionGraphRadiuses_d.data().get(), meshKnn_d.idxsPtr(),
                                    meshKnn_d.distsPtr(), meshKnn_d.size());
    af::runWarpKernel(warpedMesh_d.data().get(), meshBuff_d.data().get(), motionGraphTransforms_d.data().get(),
                      meshKnn_d.idxsPtr(), meshKnn_d.weightsPtr(), cMeshSize);
    // debug::print(meshKnn_d._weights_d, meshKnn_d._weights_d.size(), 1, "meshKnn_d._weights");
    debug::saveMesh(warpedMesh_d, "warpedMeshTest_d");
}


template<int knn>
void GetKnnDataDynamicTest(bool exludeClosest) {
    MotionGraph graph;
    graph.push_back(Vec3f(0.f, 0.f, 0.f), 1.f, Mat4f::Identity());
    graph.push_back(Vec3f(1.f, 1.f, 1.f), 1.f, Mat4f::Identity());
    graph.push_back(Vec3f(2.f, 2.f, 2.f), 1.f, Mat4f::Identity());

    std::vector<Vec3f> points = {Vec3f(0.f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f), Vec3f(0.4, 0.4, 0.4), Vec3f(0.6, 0.6, 0.6)};
    std::vector<unsigned int> knnIdxsDyn;
    std::vector<float> knnDistsDyn;
    std::vector<Vecui<knn>> knnIdxsStat;
    std::vector<Vecf<knn>> knnDistsStat;

    af::getKnnData(knnIdxsStat, knnDistsStat, graph, points, exludeClosest);
    af::getKnnData(knnIdxsDyn, knnDistsDyn, graph, points, knn, exludeClosest);

    std::cout << "static : \n";
    for (int i = 0; i < points.size(); ++i) {
        std::cout << knnIdxsStat[i].transpose() << " - " << knnDistsStat[i].transpose() << "\n";
    }

    std::cout << "\ndynamic : \n";
    for (int i = 0; i < points.size(); ++i) {
        for (int k = 0; k < knn; ++k) {
            EXPECT_EQ(knnIdxsStat[i][k], knnIdxsDyn[i + k * knn]);
            EXPECT_FLOAT_EQ(knnDistsStat[i][k], knnDistsDyn[i + k * knn]);
            std::cout << knnIdxsDyn[i + k * points.size()] << ",";
        }
        std::cout << " - ";
        for (int k = 0; k < knn; ++k) {
            std::cout << knnDistsDyn[i + k * points.size()] << ",";
        }
        std::cout << "\n";
    }
}
TEST(VertexManipulationTest, GetKnnDataDynamic) {
    GetKnnDataDynamicTest<1>(false);
    GetKnnDataDynamicTest<2>(false);
    GetKnnDataDynamicTest<3>(false);
    GetKnnDataDynamicTest<4>(false);
    GetKnnDataDynamicTest<1>(true);
    GetKnnDataDynamicTest<2>(true);
    GetKnnDataDynamicTest<3>(true);
    GetKnnDataDynamicTest<4>(true);
}