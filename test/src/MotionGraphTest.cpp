#include <af/MotionGraph.h>
#include <af/VertexCloud.h>
#include <gtest/gtest.h>

#include <af/GraphBuilder.h>
#include <chrono>

TEST(MeshTest, Normals) {
    Mat3f K = Mat3f::Identity();
    K(0, 0) = 570.342;
    K(1, 1) = 570.342;
    K(0, 2) = 320;
    K(1, 2) = 240;

    int height = 4;
    int width = 4;
    for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            Vec3f pixel((float)i, (float)j, 1);
            float depth = i*j;
            Vec3f point3d = (K.inverse() * pixel) * depth;
            std::cout << "pixel : \n" << pixel << "\n";
            std::cout << "point3d : \n" << point3d << "\n";
        }
    }
}

#include <af/dataset.h>
#include <af/VertexManipulation.h>
#include <iostream>

TEST(MeshTest, NormalsFile) {
    Mat3f K;
    af::loadIntrinsics(Constants::dataFolder + "/depthIntrinsics.txt", K);
    cv::Mat color, depth;
    // load input frame
    af::loadFrame(Constants::dataFolder, 0, color, depth);
    af::filterDepth(depth, Constants::depthThreshold);

    cv::Mat vertexMap;
    af::depthToVertexMap(K, depth, vertexMap);

    // std::cout << vertexMap << "\n";

    int count = 0;
    for (std::size_t i = 1; i < vertexMap.rows - 1; ++i) {
        if (count > 100)
            break;
        for (std::size_t j = 1; j < vertexMap.cols - 1; ++j) {
            if (vertexMap.at<Vec3f>(i, j) == Vec3f(0, 0, 0))
                continue;
            std::cout << "(" << (vertexMap.at<Vec3f>(i - 1, j)).transpose() << "\n";
            std::cout << "(" << (vertexMap.at<Vec3f>(i, j)).transpose() << "\n";
            std::cout << "(" << (vertexMap.at<Vec3f>(i + 1, j)).transpose() << "\n";
            // std::cout << "(" << vertexMap.at<float>(i - 1, j, 0) << ", " << vertexMap.at<float>(i - 1, j, 1) << ", " << vertexMap.at<float>(i - 1, j, 2) << ")" << "\n";
            // std::cout << "(" << vertexMap.at<float>(i, j, 0) << ", " << vertexMap.at<float>(i, j, 1) << ", " << vertexMap.at<float>(i, j, 2) << ")" << "\n";
            // std::cout << "(" << vertexMap.at<float>(i + 1, j, 0) << ", " << vertexMap.at<float>(i + 1, j, 1) << ", " << vertexMap.at<float>(i + 1, j, 2) << ")" << "\n";
            std::cout << "--------\n";
            ++count;
        }
        std::cout << "||||||||||||||||||||||||||||||\n";
    }
}

TEST(Nanoflann, RebuildIndex) {
    int resultsCount = 3;
    std::vector<unsigned int> outputIndicies(resultsCount);
    std::vector<float> outputDists(resultsCount);

    Vec3f point(0, 0, 0);
    VertexCloud<Vec3f> graph;
    graph.push_back(Vec3f(1, 0, 0));
    graph.push_back(Vec3f(0, 1, 0));
    graph.push_back(Vec3f(0, 0, 1));
    graph.push_back(Vec3f(2, 0, 0));

    VertexCloundL2Tree index(3, graph, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    resultsCount = index.knnSearch(&point[0], resultsCount, &outputIndicies[0], &outputDists[0]);

    for (auto&& i : outputIndicies) {
        std::cout << graph.vec_[i] << "\n-\n";
    }
    std::cout << "-------\n";

    graph.push_back(Vec3f(0.5, 0, 0));
    index.buildIndex();

    resultsCount = index.knnSearch(&point[0], resultsCount, &outputIndicies[0], &outputDists[0]);

    for (auto&& i : outputIndicies) {
        std::cout << graph.vec_[i] << "\n-\n";
    }
    std::cout << "-------\n";
}

TEST(MotionGraphTest, Warp) {
    Mat4f transform1 = Mat4f::Identity();
    transform1(0, 3) = 1.f;
    transform1(1, 3) = 1.f;
    transform1(2, 3) = 1.f;
    Mat4f transform2 = Mat4f::Identity();
    transform2(0, 3) = -1.f;
    transform2(1, 3) = -1.f;
    transform2(2, 3) = -1.f;
    Mat4f transform3;
    transform3 << 2.0, 1.5, 1.5, 1., 1., 1.5, 2.5, 1., 2., 2.5, 4.0, 1., 0., 0., 0., 1.;

    MotionGraph vGraph;
    vGraph.push_back(Vec3f(1, 0, 0), 0.9f, Mat4f::Identity());
    vGraph.push_back(Vec3f(0, 1, 0), 0.9f, transform1);
    vGraph.push_back(Vec3f(0, 0, 1), 0.9f, transform2);
    vGraph.push_back(Vec3f(0, 0, 2), 1.1f, transform3);

    std::vector<Vec3f> mesh{Vec3f(0, 0, 0), Vec3f(0, 0.5f, 0.5f), Vec3f(0, 0, 1.5f), Vec3f(0, 0, 1.2f), Vec3f(0, 0.5f, 1.5f)};
    std::vector<Vec3f> meshWarpedExpected{Vec3f(0, 0, 0), Vec3f(0, 0.5f, 0.5f), Vec3f(1.13855075, 1.89333343, 3.77072477),
                                          Vec3f(0.74052113, 1.29015922, 2.76497864), Vec3f(1.56375527, 2.57650637, 4.47382068)};
    std::vector<Vec3f> meshWarped;
    vGraph.warp(meshWarped, mesh, 3);

    std::cout << "mesh:\n";
    for (auto&& vertex : mesh) {
        std::cout << vertex.transpose() << "\n";
    }

    std::cout << "\n\n warped mesh:\n";
    for (auto&& vertex : meshWarped) {
        std::cout << std::fixed << std::setprecision(9) << vertex.transpose() << "\n";
    }

    EXPECT_EQ(meshWarped, meshWarpedExpected);
}

TEST(MotionGraphTest, BuildGraph) {
    MotionGraph graph;
    std::vector<Vec3f> mesh{Vec3f(0.5, 0.5, 0.5), Vec3f(0, 0, 0), Vec3f(0, 1, 0),  Vec3f(1, 0, 0),
                            Vec3f(0, 0, 1),       Vec3f(1, 1, 1), Vec3f(1, 0.5, 1)};
    af::buildGraph(graph, mesh, 3, 0.6);
    std::cout << "mesh.size() : \n" << mesh.size() << "\n";
    std::cout << "graph.graph().size() : \n" << graph.graph().size() << "\n";
    for (std::size_t i = 0; i < graph.graph().vec_.size(); ++i) {
        std::cout << "node " << i << ": " << graph.graph().vec_[i].transpose() << "\n";
    }
}



TEST(MotionGraphTest, SpeedTest) {
    MotionGraph graph;
    MotionGraph graphNoDuplicates;
    Mesh mesh;
    mesh.loadPly(Constants::dataFolder + "/testDump/mesh3_without_mask.ply");
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    // auto start = std::chrono::steady_clock::now();
    // std::vector<Vec3f> meshNoDuplicates;
    // af::removeDuplicatesSet(meshNoDuplicates, mesh.vertexCloud().vec_);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "meshNoDuplicates.size() : \n" << meshNoDuplicates.size() << "\n";
    // std::cout << "duplicates remove time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //           << " ms\n";

    start = std::chrono::steady_clock::now();
    std::vector<Vec3f> meshNoDuplicatesHash;
    af::removeDuplicates(meshNoDuplicatesHash, mesh.vertexCloud().vec_);
    end = std::chrono::steady_clock::now();
    std::cout << "meshNoDuplicatesHash.size() : \n" << meshNoDuplicatesHash.size() << "\n";
    std::cout << "hash duplicates remove time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    // Vec3f max = mesh.vertexCloud().vec_[0];
    // Vec3f min = mesh.vertexCloud().vec_[0];
    // for (auto &&vertex : mesh.vertexCloud().vec_) {
    //     if (vertex[0] > max[0]) max[0] = vertex[0];
    //     if (vertex[1] > max[1]) max[1] = vertex[1];
    //     if (vertex[2] > max[2]) max[2] = vertex[2];
    //     if (vertex[0] < min[0]) min[0] = vertex[0];
    //     if (vertex[1] < min[1]) min[1] = vertex[1];
    //     if (vertex[2] < min[2]) min[2] = vertex[2];
    // }
    // std::cout << "max.transpose() : \n" << max.transpose() << "\n";
    // std::cout << "min.transpose() : \n" << min.transpose() << "\n";

    float radius = Constants::motionGraphRadius;
    for (std::size_t i = 0; i < 1; ++i) {
        graph.clear();
        graphNoDuplicates.clear();

        std::cout << "--- " << i << " ---\n";
        std::cout << "radius : \n" << radius << "\n";
        std::cout << "Settings::motionGraphKNN : " << Constants::motionGraphKNN << "\n";
        std::cout << "mesh.vertexCloud().size() : " << mesh.vertexCloud().size() << "\n";

        start = std::chrono::steady_clock::now();
        af::buildGraph(graph, mesh.vertexCloud().vec_, Constants::motionGraphKNN, radius);
        end = std::chrono::steady_clock::now();
        std::cout << "graph.graph().size() : " << graph.graph().size() << "\n";
        std::cout << "af::buildGraph time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms\n";

        start = std::chrono::steady_clock::now();
        af::buildGraph(graphNoDuplicates, meshNoDuplicatesHash, Constants::motionGraphKNN, radius);
        end = std::chrono::steady_clock::now();
        std::cout << "graphNoDuplicates.graph().size() : " << graphNoDuplicates.graph().size() << "\n";
        std::cout << "af::buildGraph time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms\n";

        radius *= 0.1;
    }
}