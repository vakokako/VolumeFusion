#ifndef MOTIONGRAPH_H
#define MOTIONGRAPH_H

#include <af/Constants.h>
#include <af/VertexCloud.h>

class MotionGraph {
public:
    MotionGraph()          = default;
    virtual ~MotionGraph() = default;

    VertexCloud<Vec3f>& graph() { return mGraph; }
    std::vector<float>& radiuses() { return mRadiuses; }
    std::vector<Mat4f>& transforms() { return mTransforms; }
    const VertexCloud<Vec3f>& graph() const { return mGraph; }
    const std::vector<float>& radiuses() const { return mRadiuses; }
    const std::vector<Mat4f>& transforms() const { return mTransforms; }
    void push_back(const Vec3f& cPos, const float cRadius, const Mat4f& cTransform);
    void updateAt(const std::size_t cIndex, const Vec3f& cPos, const float cRadius, const Mat4f& cTransform);
    void eraseAt(const std::size_t cIndex);
    void clear();

    float weight(size_t nodeId, float distance);

    // excludeClosest removes closest from results (useful when knn for graph node to not contain itself as first neighbour)
    template<int KNN>
    void knnSearch(Vecui<KNN>& outputIndicies, Vecf<KNN>& outputDists, const Vec3f& cPoint, bool excludeClosest = 0) const;
    void knnSearch(std::vector<unsigned int>& outputIndicies,
                   std::vector<float>& outputDists,
                   const Vec3f& cPoint,
                   size_t resultsCount) const;

    void warp(std::vector<Vec3f>& warpedPoints,
              const std::vector<Vec3f>& cInputPoints,
              const size_t cKnnCount = Constants::motionGraphKNN);

    VertexCloundL2Tree& l2Tree() { return mL2Tree; }

private:
    void rebuildTree() const;

    VertexCloud<Vec3f> mGraph;
    std::vector<float> mRadiuses;
    std::vector<Mat4f> mTransforms;

    mutable VertexCloundL2Tree mL2Tree{3, mGraph, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)};
    mutable bool mTreeNeedsUpdate = true;
};

template<int KNN>
void MotionGraph::knnSearch(Vecui<KNN>& outputIndicies, Vecf<KNN>& outputDists, const Vec3f& cPoint, bool excludeClosest) const {
    // reconstruct a kd-tree index if the vertexes have been changed:
    rebuildTree();
    std::array<unsigned int, KNN + 1> indicies;
    std::array<float, KNN + 1> dists;
    std::size_t resultsCount = mL2Tree.knnSearch(&cPoint[0], KNN + excludeClosest, indicies.data(), dists.data())
                               - excludeClosest;
    auto indiciesOffsetBegin = indicies.begin() + excludeClosest;
    auto distsOffsetBegin = dists.begin() + excludeClosest;

    std::for_each(distsOffsetBegin, distsOffsetBegin + resultsCount, [](float& value) { value = sqrt(value); });

    std::copy(indiciesOffsetBegin, indiciesOffsetBegin + resultsCount, outputIndicies.begin());
    std::copy(distsOffsetBegin, distsOffsetBegin + resultsCount, outputDists.begin());

    for (; resultsCount < KNN; ++resultsCount) {
        outputIndicies[resultsCount] = std::numeric_limits<unsigned int>::max();
        outputDists[resultsCount]    = std::numeric_limits<float>::max();
    }
}

#endif