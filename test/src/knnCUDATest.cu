#include <gtest/gtest.h>
#include <af/knncuda.h>

#include <GraphBuilder.cpp>
#include <GraphBuilder.cu>
#include <af/Helper.cuh>
// #include <knncuda.cu>

TEST(knnCUDATest, BuildGraphSpeed) {
    Timer timer;

    MotionGraph graph;
    Mesh mesh;
    mesh.loadPly(Constants::dataFolder + "/testDump/mesh3_without_mask.ply");

    std::vector<Vec3f> meshNoDuplicatesHash;
    af::removeDuplicates(meshNoDuplicatesHash, mesh.vertexCloud().vec_);

    float radius    = Constants::motionGraphRadius;
    std::size_t knn = Constants::motionGraphKNN;

    std::cout << "radius : " << radius << "\n";
    std::cout << "knn : " << knn << "\n";
    std::cout << "meshNoDuplicatesHash.size() : \n" << meshNoDuplicatesHash.size() << "\n";

    Time("buildGraph", timer, af::buildGraph(graph, meshNoDuplicatesHash, Constants::motionGraphKNN, radius););
    std::cout << "graph.graph().size() : " << graph.graph().size() << "\n";


    FAIL() << "building graph in cuda is currently not working";


    graph.clear();
    Time("buildGraphCuda", timer, af::buildGraphCuda(graph, meshNoDuplicatesHash, Constants::motionGraphKNN, radius););
    std::cout << "graph.graph().size() : " << graph.graph().size() << "\n";

    thrust::device_vector<Vec3f> graph_d(meshNoDuplicatesHash.size());
    unsigned int graphSize = 0;
    graph.clear();
    Time("buildGraphCuda", timer, af::buildGraphCudaMine(graph_d, graphSize, meshNoDuplicatesHash, Constants::motionGraphKNN, radius););
    std::cout << "graphSize : " << graphSize << "\n";

}

TEST(knnCUDATest, Speed) {
    Timer timer;

    MotionGraph graph;
    Mesh mesh;
    mesh.loadPly(Constants::dataFolder + "/testDump/mesh3_without_mask.ply");

    std::vector<Vec3f> meshNoDuplicatesHash;
    af::removeDuplicates(meshNoDuplicatesHash, mesh.vertexCloud().vec_);

    float radius    = Constants::motionGraphRadius;
    std::size_t knn = Constants::motionGraphKNN;

    std::cout << "radius : " << radius << "\n";
    std::cout << "knn : " << knn << "\n";

    Time("buildGraph", timer, af::buildGraph(graph, meshNoDuplicatesHash, Constants::motionGraphKNN, radius););
    std::cout << "meshNoDuplicatesHash.size() : \n" << meshNoDuplicatesHash.size() << "\n";
    std::cout << "graph.graph().size() : " << graph.graph().size() << "\n";
    graph.push_back(Vec3f(0, 0, 0), radius, Mat4f::Identity());

    float* knn_dist = new float[meshNoDuplicatesHash.size() * knn];
    int* knn_index  = new int[meshNoDuplicatesHash.size() * knn];
    Time("knn_cuda_global : ", timer,
         bool cuda_success_global = knn_cuda_global(&(graph.graph().vec_[0][0]), graph.graph().size(),
                                                    &(meshNoDuplicatesHash[0][0]), meshNoDuplicatesHash.size(), 3, knn, knn_dist,
                                                    knn_index););
    Time("knn_cuda_texture : ", timer,
         bool cuda_success_texture = knn_cuda_texture(&(graph.graph().vec_[0][0]), graph.graph().size(),
                                                      &(meshNoDuplicatesHash[0][0]), meshNoDuplicatesHash.size(), 3, knn,
                                                      knn_dist, knn_index););
    // Time("knn_cublas : ", timer,
    //      bool cuda_success_cublas = knn_cublas(&(graph.graph().vec_[0][0]), graph.graph().size(),
    //      &(meshNoDuplicatesHash[0][0]),
    //                                          meshNoDuplicatesHash.size(), 3, knn, knn_dist, knn_index););

    Vecui<Constants::motionGraphKNN> outputIndicies;
    Vecf<Constants::motionGraphKNN> outputDists;
    graph.l2Tree().freeIndex(graph.l2Tree());
    Time(
        "knn_graph : ", timer, for (size_t i = 0; i < meshNoDuplicatesHash.size(); i++) {
            graph.knnSearch(outputIndicies, outputDists, meshNoDuplicatesHash[i]);
        });

    std::cout << "cuda_success_global : " << cuda_success_global << "\n";
    std::cout << "cuda_success_texture : " << cuda_success_texture << "\n";
    // std::cout << "cuda_success_cublas : " << cuda_success_cublas << "\n";
}

TEST(knnCUDATest, LessThanKNNNeighbours) {
    Timer timer;

    int knn = 2;

    std::vector<Vec3f> graph;
    std::vector<Vec3f> mesh;

    graph.push_back(Vec3f(0, 0, 0));
    mesh.push_back(Vec3f(1, 1, 1));

    float* knn_dist = new float[mesh.size() * knn];
    int* knn_index  = new int[mesh.size() * knn];

    Time("knn_cuda_global : ", timer,
         bool cuda_success_global = knn_cuda_global(&(graph[0][0]), graph.size(), &(mesh[0][0]), mesh.size(), 3, knn, knn_dist,
                                                    knn_index););

    for (std::size_t i = 0; i < mesh.size() * knn; ++i) {
        std::cout << i << ": i(" << knn_index[i] << "), d(" << knn_dist[i] << ")\n";
    }
}

TEST(knnCUDATest, SpeedHuge) {
    Timer timer;

    int refSize = 3000;
    int querySize = 256*128;
    int querySizeCount = 512;
    int querySizeFull = querySize * querySizeCount;

    std::vector<Vec3f> query(querySizeFull);

    VertexCloud<Vec3f> mGraph;
    mGraph.vec_.resize(refSize);
    // std::vector<Vec3f> ref(refSize);

    for (int i = 0; i < querySizeFull; ++i) {
        query[i] = (Vec3f::Random());
    }
    for (int i = 0; i < refSize; ++i) {
        mGraph.vec_[i] = (Vec3f::Random());
    }


    float radius    = Constants::motionGraphRadius;
    std::size_t knn = Constants::motionGraphKNN;

    std::cout << "radius : " << radius << "\n";
    std::cout << "knn : " << knn << "\n";

    float* knn_dist = new float[querySizeFull * knn];
    int* knn_index  = new int[querySizeFull * knn];
    // #if 0
    Time("knn_cuda_global : ", timer,
        for (int i = 0; i < querySizeCount; ++i) {
            bool cuda_success_global = knn_cuda_global((float*)(mGraph.vec_.data()), refSize,
                                                    (float*)(query.data() + i * querySize), querySize, 3, knn, knn_dist + i * querySize,
                                                    knn_index + i * querySize);
        }
    );
    Time("knn_cuda_texture : ", timer,
        for (int i = 0; i < querySizeCount; ++i) {
         bool cuda_success_texture = knn_cuda_texture((float*)(mGraph.vec_.data()), refSize,
                                                    (float*)(query.data() + i * querySize), querySize, 3, knn, knn_dist + i * querySize,
                                                    knn_index + i * querySize);
        }
    );
    // #endif

    Time("cpu build tree", timer,
        VertexCloundL2Tree mL2Tree { 3, mGraph, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) };
        mL2Tree.buildIndex();
    );

    Time("knn search", timer,
        std::vector<unsigned int> outputIndicies(knn);
        std::vector<float> outputDists(knn);

        for (int i = 0; i < querySizeFull; ++i) {
            unsigned int resultsCount = mL2Tree.knnSearch(&query[i][0], knn, &outputIndicies[0], &outputDists[0]);
        }
    );
    // Time("knn_cublas : ", timer,
    //      bool cuda_success_cublas = knn_cublas(&(graph.graph().vec_[0][0]), graph.graph().size(),
    //      &(meshNoDuplicatesHash[0][0]),
    //                                          meshNoDuplicatesHash.size(), 3, knn, knn_dist, knn_index););

    // std::cout << "cuda_success_global : " << cuda_success_global << "\n";
    // std::cout << "cuda_success_texture : " << cuda_success_texture << "\n";
    // std::cout << "cuda_success_cublas : " << cuda_success_cublas << "\n";
}