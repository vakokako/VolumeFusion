#include <af/Mesh.h>
#include <af/MotionGraph.h>
#include <af/Constants.h>

#include <KNN.cu>
#include <knncuda.cu>
#include <set>
#include <unordered_set>

namespace af {

void buildGraphCuda(MotionGraph& graph,
                    const std::vector<Vec3f>& mesh,
                    const std::size_t cMotionGraphKnn,
                    const float cMotionGraphMinRadius) {
    if (mesh.size() == 0)
        return;

    std::vector<int> outputIndicies(cMotionGraphKnn);
    std::vector<float> outputDists(cMotionGraphKnn);

    if (graph.graph().size() == 0) {
        graph.push_back(mesh[0], cMotionGraphMinRadius, Mat4f::Identity());
    }
    int knn = 0;
    for (size_t i = 1; i < mesh.size(); i++) {
        knn = std::min(graph.graph().size(), cMotionGraphKnn);

        knn_cuda_global(&(graph.graph().vec_[0][0]), graph.graph().size(), &(mesh[i][0]), 1, 3, knn, outputDists.data(),
                        outputIndicies.data());
        // graph.knnSearch(outputIndicies, outputDists, mesh[i], cMotionGraphKnn);

        bool isSupported = false;
        for (std::size_t j = 0; j < knn; ++j) {
            if (outputDists[j] < graph.radiuses()[outputIndicies[j]]) {
                isSupported = true;
                break;
            }
        }
        if (isSupported)
            continue;

        graph.push_back(mesh[i], cMotionGraphMinRadius, Mat4f::Identity());
    }
}

void buildGraphCudaMine(thrust::device_vector<Vec3f>& graph_d,
                        unsigned int& graphSize,
                        const std::vector<Vec3f>& mesh,
                        const unsigned int cMotionGraphKnn,
                        const float cMotionGraphMinRadius) {
    std::vector<int> outputIndicies(cMotionGraphKnn);
    std::vector<float> outputDists(cMotionGraphKnn);

    thrust::device_vector<float> graphRadiuses_d(mesh.size());
    thrust::device_vector<Mat4f> graphTransforms_d(mesh.size());

    graph_d[graphSize]           = mesh[0];
    graphRadiuses_d[graphSize]   = cMotionGraphMinRadius;
    graphTransforms_d[graphSize] = Mat4f::Identity();
    ++graphSize;
    int knn = 0;
    for (size_t i = 1; i < mesh.size(); i++) {
        knn = std::min(graphSize, cMotionGraphKnn);

        af::knn(graph_d.data().get(), graphSize, mesh[i], knn, outputIndicies, outputDists);
        // knn_cuda_global(&(graph.graph().vec_[0][0]), graph.graph().size(), &(mesh[i][0]), 1, 3,
        //                 knn, outputDists.data(), outputIndicies.data());
        // graph.knnSearch(outputIndicies, outputDists, mesh[i], cMotionGraphKnn);

        bool isSupported = false;
        for (std::size_t j = 0; j < knn; ++j) {
            if (outputDists[j] < graphRadiuses_d[outputIndicies[j]]) {
                isSupported = true;
                break;
            }
        }
        if (isSupported)
            continue;

        graph_d[graphSize]           = mesh[i];
        graphRadiuses_d[graphSize]   = cMotionGraphMinRadius;
        graphTransforms_d[graphSize] = Mat4f::Identity();
        ++graphSize;
    }
}

}  // namespace af
