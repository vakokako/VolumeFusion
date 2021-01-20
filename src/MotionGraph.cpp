#include <af/MotionGraph.h>
#include <af/Constants.h>

#include <limits>

void MotionGraph::push_back(const Vec3f& cPos, const float cRadius, const Mat4f& cTransform) {
    if (cRadius < 0)
        throw std::runtime_error("MotionGraph::push_back(): warp radius cannot be <0.");
    mGraph.vec_.push_back(cPos);
    mRadiuses.push_back(cRadius);
    mTransforms.push_back(cTransform);
    mTreeNeedsUpdate = true;
}

void MotionGraph::updateAt(const std::size_t cIndex, const Vec3f& cPos, const float cRadius, const Mat4f& cTransform) {
    if (cRadius < 0)
        throw std::runtime_error("MotionGraph::updateAt(): warp radius cannot be <0.");
    mGraph.vec_[cIndex] = cPos;
    mRadiuses[cIndex]   = cRadius;
    mTransforms[cIndex] = cTransform;
    mTreeNeedsUpdate    = true;
}

void MotionGraph::eraseAt(const std::size_t cIndex) {
    mGraph.vec_.erase(mGraph.vec_.begin() + cIndex);
    mRadiuses.erase(mRadiuses.begin() + cIndex);
    mTransforms.erase(mTransforms.begin() + cIndex);
    mTreeNeedsUpdate = true;
}

void MotionGraph::clear() {
    mGraph.clear();
    mRadiuses.clear();
    mTransforms.clear();
    mL2Tree.freeIndex(mL2Tree);
    mTreeNeedsUpdate = true;
}

float MotionGraph::weight(size_t nodeId, float distance) {
    return (distance >= mRadiuses[nodeId]) ? 0.f : exp(-(distance * distance) / (2 * pow(mRadiuses[nodeId], 2)));
}

void MotionGraph::knnSearch(std::vector<unsigned int>& outputIndicies,
                            std::vector<float>& outputDists,
                            const Vec3f& cPoint,
                            size_t resultsCount) const {
    // reconstruct a kd-tree index if the vertexes have been changed:
    rebuildTree();

    outputIndicies.resize(resultsCount);
    outputDists.resize(resultsCount);

    resultsCount = mL2Tree.knnSearch(&cPoint[0], resultsCount, &outputIndicies[0], &outputDists[0]);

    // In case of less points in the tree than requested:
    outputIndicies.resize(resultsCount);
    outputDists.resize(resultsCount);

    std::for_each(outputDists.begin(), outputDists.end(), [](float& value) { value = sqrt(value); });
}

void MotionGraph::warp(std::vector<Vec3f>& warpedPoints, const std::vector<Vec3f>& cInputPoints, const size_t cKnnCount) {
    warpedPoints.resize(cInputPoints.size());

    std::vector<unsigned int> knnIdicies(cKnnCount);
    std::vector<float> knnDists(cKnnCount);
    std::vector<float> weights(cKnnCount);

    for (size_t i = 0; i < cInputPoints.size(); i++) {
        knnSearch(knnIdicies, knnDists, cInputPoints[i], cKnnCount);

        // normalizing weights later so that sum == 1
        float weightsSum = 0;
        for (size_t j = 0; j < knnIdicies.size(); j++) {
            weights[j] = weight(knnIdicies[j], knnDists[j]);
            weightsSum += weights[j];
        }

        if (weightsSum == 0.f) {
            warpedPoints[i] = cInputPoints[i];
            continue;
        }

        warpedPoints[i].fill(0);
        const Vec4f cInputPointHom = cInputPoints[i].homogeneous();
        for (size_t j = 0; j < knnIdicies.size(); ++j) {
            warpedPoints[i] += weights[j] / weightsSum * (transforms()[knnIdicies[j]] * cInputPointHom).head(3);
        }
    }
}

void MotionGraph::rebuildTree() const {
    if (!mTreeNeedsUpdate)
        return;

    mL2Tree.buildIndex();
    mTreeNeedsUpdate = false;
}