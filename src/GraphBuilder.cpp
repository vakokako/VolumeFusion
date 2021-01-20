#include <af/Mesh.h>
#include <af/MotionGraph.h>
#include <af/Constants.h>
#include <af/GraphBuilder.h>

#include <numeric>
#include <set>
#include <unordered_set>

namespace af {

void buildGraph(MotionGraph& graph,
                const std::vector<Vec3f>& mesh,
                const std::size_t cMotionGraphKnn,
                const float cMotionGraphMinRadius) {
    if (mesh.size() == 0)
        return;

    if (graph.graph().size() == 0)
        graph.push_back(mesh[0], cMotionGraphMinRadius, Mat4f::Identity());

    std::vector<unsigned int> outputIndicies;
    std::vector<float> outputDists;

    for (size_t i = 0; i < mesh.size(); i++) {
        graph.knnSearch(outputIndicies, outputDists, mesh[i], cMotionGraphKnn);

        bool isSupported = false;
        for (std::size_t j = 0; j < outputIndicies.size(); ++j) {
            if (outputDists[j] < graph.radiuses()[outputIndicies[j]]) {
                isSupported = true;
                break;
            }
        }
        if (isSupported)
            continue;

        // int countNeighb = std::min(outputIndicies.size(), Constants::energyMRegKNN);
        int countNeighb = 1;
        float distsSum  = std::accumulate(outputDists.begin(), outputDists.begin() + countNeighb, 0.f);

        Mat4f transformNew = Mat4f::Zero();

        for (int j = 0; j < countNeighb; ++j) {
            transformNew += graph.transforms()[outputIndicies[j]] * (outputDists[j] / distsSum);
        }

        graph.push_back(mesh[i], cMotionGraphMinRadius, transformNew);
    }
}

void getKnnData(std::vector<unsigned int>& knnIdxs,
                std::vector<float>& knnDists,
                MotionGraph& cGraph,
                const std::vector<Vec3f>& cMesh,
                const int cKnn,
                bool excludeClosest) {
    const std::size_t cMeshSize = cMesh.size();
    knnIdxs.resize(cMeshSize * cKnn);
    knnDists.resize(cMeshSize * cKnn);

    std::fill(knnIdxs.begin(), knnIdxs.end(), std::numeric_limits<unsigned int>::max());
    std::fill(knnDists.begin(), knnDists.end(), std::numeric_limits<float>::max());

    std::vector<unsigned int> idxs;
    std::vector<float> dists;
    idxs.reserve(cKnn + excludeClosest);
    dists.reserve(cKnn + excludeClosest);

    for (size_t i = 0; i < cMeshSize; i++) {
        cGraph.knnSearch(idxs, dists, cMesh[i], cKnn + excludeClosest);

        for (std::size_t k = 0; k < idxs.size() - excludeClosest; ++k) {
            knnIdxs[i + k * cMeshSize] = idxs[k + excludeClosest];
            knnDists[i + k * cMeshSize] = dists[k + excludeClosest];
        }
    }
}
// void removeDuplicatesSet(std::vector<Vec3f>& output, const std::vector<Vec3f>& cInput) {
//     std::set<Vec3f> set(cInput.begin(), cInput.end());
//     output.resize(set.size());
//     std::copy(set.begin(), set.end(), output.begin());
// }

void removeDuplicates(std::vector<Vec3f>& output, const std::vector<Vec3f>& cInput) {
    // const auto hash = boost::hash_range(begin(cInput), end(cInput));
    std::unordered_set<Vec3f> set(cInput.begin(), cInput.end());
    output.resize(set.size());
    std::copy(set.begin(), set.end(), output.begin());
}

void translate(std::vector<Vec3f>& data, const Vec3f& cTransVec) {
    for (auto&& point : data) {
        point += cTransVec;
    }
}

}  // namespace af
