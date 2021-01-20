#ifndef GRAPHBUILDER_H
#define GRAPHBUILDER_H

#include <af/Mesh.h>
#include <af/MotionGraph.h>
#include <af/Constants.h>

#include <iostream>
#include <set>
#include <unordered_set>

namespace af {

void buildGraph(MotionGraph& graph,
                const std::vector<Vec3f>& mesh,
                const std::size_t cMotionGraphKnn,
                const float cMotionGraphMinRadius);

// TODO Optimize: this additionally runs the knn on all mesh. This data should be already accessible from buildGraph
template<int KNN>
void getKnnData(std::vector<Vecui<KNN>>& knnIdxs,
                std::vector<Vecf<KNN>>& knnDists,
                const MotionGraph& cGraph,
                const std::vector<Vec3f>& cMesh,
                bool excludeClosest = 0) {
    knnIdxs.resize(cMesh.size());
    knnDists.resize(cMesh.size());

    for (size_t i = 0; i < cMesh.size(); i++) {
        cGraph.knnSearch(knnIdxs[i], knnDists[i], cMesh[i], excludeClosest);
    }
}

template<int KNN>
void filterDisconnected(std::vector<Vecui<KNN>>& knnIdxs, std::vector<Vecf<KNN>>& knnDists, const MotionGraph& cGraph) {
    for (std::size_t i = 0; i < knnIdxs.size(); ++i) {
        for (int j = 0; j < KNN; ++j) {
            unsigned int neighbourId = knnIdxs[i][j];
            if (neighbourId >= cGraph.graph().size()) {
                continue;
            } else if (knnDists[i][j] <= (cGraph.radiuses()[i] * 1.3f)) {
                continue;
            }
            knnIdxs[i][j]  = std::numeric_limits<unsigned int>::max();
            knnDists[i][j] = std::numeric_limits<float>::max();
        }
    }
}

void getKnnData(std::vector<unsigned int>& knnIdxs,
                std::vector<float>& knnDists,
                MotionGraph& cGraph,
                const std::vector<Vec3f>& cMesh,
                const int cKnn,
                bool excludeClosest = 0);

void removeDuplicates(std::vector<Vec3f>& output, const std::vector<Vec3f>& cInput);

void translate(std::vector<Vec3f>& data, const Vec3f& cTransVec);

}  // namespace af

#endif