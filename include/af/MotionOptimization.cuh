#ifndef MOTIONOPTIMIZATION_CUH
#define MOTIONOPTIMIZATION_CUH

#include <af/Helper.cuh>
#include <af/VertexManipulation.cuh>

namespace af {

void runFill_J_DepthKernel(Vec6f* jDepth,
                           float energyWeight,
                           unsigned int jDepthHeight,
                           unsigned int jDepthWidth,
                           const Vec2i* cCorrIdxs,
                           unsigned int corrIdxsSize,
                           const Vec3f* cCanonMesh,
                           const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                           const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                           const Vec3f* cDepthMesh,
                           const Vec3f* cDepthMeshNormals);

void runFill_J_MRegKernel(Vec6f* jMReg,
                          float energyWeight,
                          unsigned int jMRegHeight,
                          unsigned int jMRegWidth,
                          const Vec3f* cGraph,
                          const Mat4f* cGraphTransforms,
                          const unsigned int cGraphSize,
                          const Vecui<Constants::energyMRegKNN>* cKNNIdxs);

void runFill_r_DepthKernel(float* rDepth,
                           float energyWeight,
                           const Vec2i* cCorrIdxs,
                           unsigned int corrIdxsSize,
                           const Vec3f* cCanonMeshWarped,
                           const Vec3f* cDepthMesh,
                           const Vec3f* cDepthMeshNormals);

void runFill_r_MRegKernel(float* rMReg,
                          float energyWeight,
                          const Vec3f* cGraph,
                          const Mat4f* cGraphTransforms,
                          const unsigned int cGraphSize,
                          const Vecui<Constants::energyMRegKNN>* cKNNIdxs);

constexpr unsigned int factorial(const unsigned int k) { return k == 0 ? 1 : k * factorial(k - 1); }
constexpr int combinations(const unsigned int n, const unsigned int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

/** @brief Computing all pairs of transforms that contribute to the same error function (i.e. in the same row) */
void runComputeJTJAndJTrContribJDepthKernel(Vec2i* jTJContribPairs,
                                            unsigned int* rowNumbers,
                                            unsigned int* jTJContribPairsSize,
                                            Vec2i* jTrContribElems,
                                            unsigned int* jTrContribElemsSize,
                                            const Vec2i* cCorrIdxs,
                                            unsigned int corrIdxsSize,
                                            const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                            const Vecf<Constants::motionGraphKNN>* cKnnWeights);

void runComputeJTJAndJTrContribJMRegKernel(Vec2i* jTJContribPairs,
                                           unsigned int* rowNumbers,
                                           unsigned int rowNumberOffset,
                                           unsigned int* jTJContribPairsSize,
                                           Vec2i* jTrContribElems,
                                           unsigned int* jTrContribElemsSize,
                                           const unsigned int cGraphSize,
                                           const Vecui<Constants::energyMRegKNN>* cKNNIdxs);

void runMultiplyJTJContribPairsKernel(Mat6f* multipliedContribPairs,
                                      const Vec2i* cJTJContribPairs,
                                      const unsigned int* cRowNumbers,
                                      unsigned int jTJContribPairsSize,
                                      const Vec6f* cJ,
                                      const unsigned int cJWidth);

void runMultiplyJTrContribElemsKernel(Vec6f* multipliedJTrContribElems,
                                      const Vec2i* cJTrContribElems,
                                      const unsigned int cJTrContribElemsSize,
                                      const Vec6f* cJ,
                                      const float* cr,
                                      const unsigned int cJWidth);

void runFillJTJKernel(float* JTJ,
                      const Vec2i* cJTJElemsCoords,
                      const Mat6f* cJTJElems,
                      unsigned int jTJElemsSize,
                      const unsigned int cJTJWidth);

void runFillJTrKernel(Vec6f* JTr, const Vec2i* cJTrElemsCoordinates, const Vec6f* cJTrElems, const unsigned int cJTrElemsSize);

void runUpdateTransformsKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize);
void runUpdateTransformsReverseKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize);

}  // namespace af

#endif