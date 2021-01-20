#include <af/GraphBuilder.h>
#include <af/MarchingCubes.h>
#include <af/Constants.h>
#include <af/TSDFVolume.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <af/Helper.cuh>
#include <af/MarchingCubes.cuh>
#include <af/TestHelper.cuh>
#include <fstream>

bool compareVecs(const Vec3f& a, const Vec3f& b) {
    if (a[0] < b[0])
        return true;
    if (b[0] < a[0])
        return false;
    if (a[1] < b[1])
        return true;
    if (b[1] < a[1])
        return false;
    return a[2] < b[2];
}

TEST(MarchingCubesTest, IsoVsPointCloud) {
    Timer timer;

    Mat3f K;
    cv::Mat color, depth;
    loadFilteredFrameAndIntrinsics(K, depth, color);

    Vec3i volDim(256, 256, 256);
    Vec3f volSize(1.5f, 1.5f, 1.5f);
    Vec3f voxlSize = volSize.cwiseQuotient(volDim.cast<float>());
    float delta    = 0.02f;

    TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, K);
    tsdfResult->setDelta(delta);
    if (std::ifstream(Constants::dataFolder + "/testDump/tsdfResult.txt").good()) {
        tsdfResult->load(Constants::dataFolder + "/testDump/tsdfResult.txt");
    } else {
        throw std::runtime_error("don't want to integrate, load tsdf!");
    }

    MarchingCubes mc(volDim, volSize);

    Mesh meshSurface;
    Time("computeIsoSurface", timer, mc.computeIsoSurface(meshSurface, *tsdfResult););

    Mesh meshCloud;
    Time("computePointCloud", timer, mc.computePointCloud(meshCloud, *tsdfResult););

    std::vector<Vec3f> meshSurfaceNoDupl;
    std::vector<Vec3f> meshCloudNoDupl;
    Time("removeDuplicates surface", timer,
        af::removeDuplicates(meshSurfaceNoDupl, meshSurface.vertexCloud().vec_);
    );
    Time("removeDuplicates cloud", timer,
        af::removeDuplicates(meshCloudNoDupl, meshCloud.vertexCloud().vec_);
    );

    std::cout << "meshSurface.vertexCloud().vec_.size() : " << meshSurface.vertexCloud().vec_.size() << "\n";
    std::cout << "meshSurfaceNoDupl.size() : " << meshSurfaceNoDupl.size() << "\n";

    std::cout << "meshCloud.vertexCloud().vec_.size() : " << meshCloud.vertexCloud().vec_.size() << "\n";
    std::cout << "meshCloudNoDupl.size() : " << meshCloudNoDupl.size() << "\n";

    std::sort(meshCloudNoDupl.begin(), meshCloudNoDupl.end(), compareVecs);
    std::sort(meshSurfaceNoDupl.begin(), meshSurfaceNoDupl.end(), compareVecs);

    // GPU computaiton
    af::initEdgeTableDevice();

    thrust::device_vector<Vec3f> mesh_d(100000);
    thrust::device_vector<unsigned int> meshSize_d(1, 0);
    thrust::device_vector<float> tsdf_d        = tsdfResult->tsdf();
    thrust::device_vector<float> tsdfWeights_d = tsdfResult->tsdfWeights();
    CUDA_CHECK;
    Time("runMarchingCubesMeshKernel", timer,
         af::runMarchingCubesMeshKernel(mesh_d.data().get(), meshSize_d.data().get(), tsdf_d.data().get(),
                                  tsdfWeights_d.data().get(), volDim, voxlSize, volSize, 0);
    );

    unsigned int meshSizeKernel;
    thrust::copy_n(meshSize_d.begin(), 1, &meshSizeKernel); CUDA_CHECK;
    std::cout << "meshSizeKernel raw: " << meshSizeKernel << "\n";

    Time("sort(mesh_d)", timer,
        thrust::sort(mesh_d.begin(), mesh_d.end());
    );
    Time("unique(mesh_d)", timer,
        auto itEndUnique = thrust::unique(mesh_d.begin(), mesh_d.end());
    );
    meshSizeKernel = itEndUnique - mesh_d.begin();
    meshSize_d[0] = meshSizeKernel;
    std::cout << "meshSizeKernel : " << meshSizeKernel << "\n";

    std::vector<Vec3f> meshCloudKernel(meshSizeKernel);
    thrust::copy_n(mesh_d.begin(), meshSizeKernel, meshCloudKernel.data());

    // Compare all results
    EXPECT_EQ(meshCloudNoDupl.size(), meshSurfaceNoDupl.size());
    EXPECT_EQ(meshCloudNoDupl.size(), meshCloudKernel.size());
    for (std::size_t i = 4299; i < 4320; ++i) {
        std::cout << "i: " << i << "\t" << meshCloudNoDupl[i].transpose() << "\t" << meshCloudKernel[i].transpose() << "\n";
    }
    for (std::size_t i = 0; i < meshCloudNoDupl.size() && i < meshSurfaceNoDupl.size() && i < meshCloudKernel.size(); ++i) {
        ASSERT_EQ(meshCloudNoDupl[i], meshSurfaceNoDupl[i]) << "iteration(i): " << i;
        ASSERT_EQ(meshCloudNoDupl[i], meshCloudKernel[i]) << "iteration(i): " << i;
        // if (meshCloudNoDupl[i] != meshCloudKernel[i])
        //     std::cout << i << ", ";
    }
    std::cout << "\n";
}