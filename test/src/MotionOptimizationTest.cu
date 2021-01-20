#include <af/DebugHelper.h>
#include <gtest/gtest.h>

#include <af/Helper.cuh>

#include <af/MotionOptimization.cuh>

TEST(MotionOptimizationTest, FillJDepth) {

    thrust::device_vector<Vecui<Constants::motionGraphKNN>> knnIdxs_d;
    thrust::device_vector<Vecf<Constants::motionGraphKNN>> knnWeights_d;
    Vecui<Constants::motionGraphKNN> knnIdxs1, knnIdxs2, knnIdxs3;
    Vecf<Constants::motionGraphKNN> knnWeights1, knnWeights2, knnWeights3;
    knnIdxs1 << 1, 4, 0, 0, 0, 0;
    knnWeights1 << 0.5, 0.5, 0, 0, 0, 0;
    knnIdxs2 << 1, 0, 2, 3, 5, 0;
    knnWeights2 << 0.2, 0.2, 0.2, 0.2, 0.2, 0;
    knnIdxs3 << 0, 1, 3, 0, 0, 0;
    knnWeights3 << 0.3, 0.3, 0.4, 0, 0, 0;
    knnIdxs_d.push_back(knnIdxs1);
    knnIdxs_d.push_back(knnIdxs2);
    knnIdxs_d.push_back(knnIdxs3);
    knnWeights_d.push_back(knnWeights1);
    knnWeights_d.push_back(knnWeights2);
    knnWeights_d.push_back(knnWeights3);

    thrust::device_vector<Vec3f> canonMesh_d;
    canonMesh_d.push_back(Vec3f(1.f, 1.f, 1.f));
    canonMesh_d.push_back(Vec3f(2.f, 1.f, 1.f));
    canonMesh_d.push_back(Vec3f(1.f, 2.f, 1.f));

    thrust::device_vector<Vec3f> depthMesh_d;
    depthMesh_d.push_back(Vec3f(1.f, 1.f, 1.f));
    depthMesh_d.push_back(Vec3f(2.f, 1.f, 1.f));
    depthMesh_d.push_back(Vec3f(1.f, 2.f, 1.f));

    thrust::device_vector<Vec3f> depthMeshNormals_d;
    depthMeshNormals_d.push_back(Vec3f(0, 0, 1.f));
    depthMeshNormals_d.push_back(Vec3f(0, 1.f, 0));
    depthMeshNormals_d.push_back(Vec3f(1.f, 1.f, 0));

    thrust::device_vector<Vec2i> corrIdxs_d;
    corrIdxs_d.push_back(Vec2i(0, 0));
    corrIdxs_d.push_back(Vec2i(1, 1));
    corrIdxs_d.push_back(Vec2i(2, 2));
    unsigned int corrIdxsSize = corrIdxs_d.size();

    unsigned int jDepthHeight = corrIdxs_d.size();
    unsigned int jDepthWidth  = 6;
    thrust::device_vector<Vec6f> jDepth_d(jDepthHeight * jDepthWidth);

    af::runFill_J_DepthKernel(jDepth_d.data().get(), 1., jDepthHeight, jDepthWidth, corrIdxs_d.data().get(), corrIdxsSize,
                              canonMesh_d.data().get(), knnIdxs_d.data().get(), knnWeights_d.data().get(), depthMesh_d.data().get(),
                              depthMeshNormals_d.data().get());

    debug::print(jDepth_d, jDepthHeight, jDepthWidth);
}

TEST(MotionOptimizationTest, FillJTJSparse) {

}