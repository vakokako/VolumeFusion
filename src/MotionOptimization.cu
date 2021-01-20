#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <af/MotionOptimization.cuh>

namespace af {

// clang-format off
__device__ void gradOfXTransfVertex(Vec6f& grad, const Vec3f& cVertex) {
    grad << 0, cVertex[2], -cVertex[1], 1, 0, 0;
}

__device__ void gradOfYTransfVertex(Vec6f& grad, const Vec3f& cVertex) {
    grad << -cVertex[2], 0, cVertex[0], 0, 1, 0;
}

__device__ void gradOfZTransfVertex(Vec6f& grad, const Vec3f& cVertex) {
    grad << cVertex[1], -cVertex[0], 0, 0, 0, 1;
}

__device__ void gradOfTransfVertex(Mat3_6f& grad, const Vec3f& cVertex) {
    grad <<  0,           cVertex[2], -cVertex[1],    1, 0, 0,
            -cVertex[2],  0,           cVertex[0],    0, 1, 0,
             cVertex[1], -cVertex[0],  0,             0, 0, 1;
}
// clang-format on

__global__ void fill_J_DepthKernel(Vec6f* jDepth,
                                   float energyWeight,
                                   unsigned int jDepthHeight,
                                   unsigned int jDepthWidth,
                                   const Vec2i* cCorrIdxs,
                                   unsigned int corrIdxsSize,
                                   const Vec3f* cCanonMesh,
                                   const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                   const Vecf<Constants::motionGraphKNN>* cKnnWeights,
                                   const Vec3f* cDepthMesh,
                                   const Vec3f* cDepthMeshNormals) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= corrIdxsSize)
        return;

    const Vec2i cCorrPair        = cCorrIdxs[cIdx];
    const int cCanonVertexIdx    = cCorrPair[0];
    const int cDepthVertexIdx    = cCorrPair[1];
    const Vec3f cCurrDepthNormal = cDepthMeshNormals[cDepthVertexIdx];
    auto cCurrKnnIdxs            = cKnnIdxs[cCanonVertexIdx];
    auto cCurrKnnWeights         = cKnnWeights[cCanonVertexIdx];

    Mat3_6f gradTv;
    af::gradOfTransfVertex(gradTv, cCanonMesh[cCanonVertexIdx]);

    for (std::size_t i = 0; i < Constants::motionGraphKNN; ++i) {
        float weight = cCurrKnnWeights[i];
        if (weight < 1.e-4f)
            continue;
        int jDepthIdx     = cIdx * jDepthWidth + cCurrKnnIdxs[i];
        jDepth[jDepthIdx] = (cCurrDepthNormal * weight).transpose() * gradTv * energyWeight;
    }
}

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
                           const Vec3f* cDepthMeshNormals) {
    dim3 block, grid;
    setupBlockGrid(block, grid, corrIdxsSize);
    fill_J_DepthKernel<<<grid, block>>>(jDepth, energyWeight, jDepthHeight, jDepthWidth, cCorrIdxs, corrIdxsSize, cCanonMesh,
                                        cKnnIdxs, cKnnWeights, cDepthMesh, cDepthMeshNormals);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void fill_J_MRegKernel(Vec6f* jMReg,
                                  float energyWeight,
                                  unsigned int jMRegHeight,
                                  unsigned int jMRegWidth,
                                  const Vec3f* cGraph,
                                  const Mat4f* cGraphTransforms,
                                  const unsigned int cGraphSize,
                                  const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cGraphSize)
        return;

    auto cCurrKNNIdxs     = cKNNIdxs[cIdx];
    Mat4f currTransform   = cGraphTransforms[cIdx];
    Vec3f currVertex      = cGraph[cIdx];
    Vec4f currVertexHom   = currVertex.homogeneous();
    Vec3f transformedCurr = (currTransform * currVertexHom).head(3);
    // Mat3_6f gradCurr;
    // af::gradOfTransfVertex(gradCurr, currVertex);
    Vec6f gradCurrX;
    Vec6f gradCurrY;
    Vec6f gradCurrZ;
    af::gradOfXTransfVertex(gradCurrX, currVertex);
    af::gradOfYTransfVertex(gradCurrY, currVertex);
    af::gradOfZTransfVertex(gradCurrZ, currVertex);
    gradCurrX *= energyWeight;
    gradCurrY *= energyWeight;
    gradCurrZ *= energyWeight;

    // Start from index 1, as 0 will always be the graph node itself
    for (std::size_t i = 0; i < Constants::energyMRegKNN; ++i) {
        if (cCurrKNNIdxs[i] >= cGraphSize) {
            continue;
        }

        const int cRowId          = (cIdx * (Constants::energyMRegKNN) + i) * 3;
        const int cJMRegIdxCurr   = cRowId * jMRegWidth + cIdx;
        const int cJMRegIdxNeighb = cRowId * jMRegWidth + cCurrKNNIdxs[i];

        // X component of gradient
        jMReg[cJMRegIdxCurr]   = gradCurrX;
        jMReg[cJMRegIdxNeighb] = -gradCurrX;
        // Y component of gradient
        jMReg[cJMRegIdxCurr + jMRegWidth]   = gradCurrY;
        jMReg[cJMRegIdxNeighb + jMRegWidth] = -gradCurrY;
        // Z component of gradient
        jMReg[cJMRegIdxCurr + 2 * jMRegWidth]   = gradCurrZ;
        jMReg[cJMRegIdxNeighb + 2 * jMRegWidth] = -gradCurrZ;

        // Mat4f neighbourTransform = cGraphTransforms[cCurrKNNIdxs[i]];
        // Vec3f diff               = transformedCurr - (neighbourTransform * currVertexHom).head(3);
        // float diffNorm           = diff.norm();
        // Vec6f jMRegValue;
        // if (diffNorm == 0.f) {
        //     jMRegValue = Vec6f::Zero();
        // } else {
        //     diff /= diffNorm;
        //     jMRegValue = diff.transpose() * gradCurr;
        // }

        // jMReg[cJMRegIdxCurr]   = jMRegValue * energyWeight;
        // jMReg[cJMRegIdxNeighb] = -jMRegValue * energyWeight;
    }
}

void runFill_J_MRegKernel(Vec6f* jMReg,
                          float energyWeight,
                          unsigned int jMRegHeight,
                          unsigned int jMRegWidth,
                          const Vec3f* cGraph,
                          const Mat4f* cGraphTransforms,
                          const unsigned int cGraphSize,
                          const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cGraphSize);
    fill_J_MRegKernel<<<grid, block>>>(jMReg, energyWeight, jMRegHeight, jMRegWidth, cGraph, cGraphTransforms, cGraphSize,
                                       cKNNIdxs);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void fill_r_DepthKernel(float* rDepth,
                                   float energyWeight,
                                   const Vec2i* cCorrIdxs,
                                   unsigned int corrIdxsSize,
                                   const Vec3f* cCanonMeshWarped,
                                   const Vec3f* cDepthMesh,
                                   const Vec3f* cDepthMeshNormals) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= corrIdxsSize)
        return;

    const Vec2i cCorrPair     = cCorrIdxs[cIdx];
    const int cCanonVertexIdx = cCorrPair[0];
    const int cDepthVertexIdx = cCorrPair[1];

    Vec3f diff = cCanonMeshWarped[cCanonVertexIdx] - cDepthMesh[cDepthVertexIdx];

    rDepth[cIdx] = energyWeight * cDepthMeshNormals[cDepthVertexIdx].dot(diff);
}

void runFill_r_DepthKernel(float* rDepth,
                           float energyWeight,
                           const Vec2i* cCorrIdxs,
                           unsigned int corrIdxsSize,
                           const Vec3f* cCanonMeshWarped,
                           const Vec3f* cDepthMesh,
                           const Vec3f* cDepthMeshNormals) {
    dim3 block, grid;
    setupBlockGrid(block, grid, corrIdxsSize);
    fill_r_DepthKernel<<<grid, block>>>(rDepth, energyWeight, cCorrIdxs, corrIdxsSize, cCanonMeshWarped, cDepthMesh,
                                        cDepthMeshNormals);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void fill_r_MRegKernel(float* rMReg,
                                  float energyWeight,
                                  const Vec3f* cGraph,
                                  const Mat4f* cGraphTransforms,
                                  const unsigned int cGraphSize,
                                  const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cGraphSize)
        return;

    auto cCurrKNNIdxs     = cKNNIdxs[cIdx];
    Mat4f currTransform   = cGraphTransforms[cIdx];
    Vec3f currVertex      = cGraph[cIdx];
    Vec4f currVertexHom   = currVertex.homogeneous();
    Vec3f transformedCurr = (currTransform * currVertexHom).head(3);

    for (std::size_t i = 0; i < Constants::energyMRegKNN; ++i) {
        if (cCurrKNNIdxs[i] >= cGraphSize) {
            continue;
        }

        const int cRowId = (cIdx * (Constants::energyMRegKNN) + i) * 3;

        Mat4f neighbourTransform = cGraphTransforms[cCurrKNNIdxs[i]];
        Vec3f diff               = transformedCurr - (neighbourTransform * currVertexHom).head(3);
        rMReg[cRowId]            = diff[0] * energyWeight;
        rMReg[cRowId + 1]        = diff[1] * energyWeight;
        rMReg[cRowId + 2]        = diff[2] * energyWeight;
    }
}

void runFill_r_MRegKernel(float* rMReg,
                          float energyWeight,
                          const Vec3f* cGraph,
                          const Mat4f* cGraphTransforms,
                          const unsigned int cGraphSize,
                          const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cGraphSize);

    fill_r_MRegKernel<<<grid, block>>>(rMReg, energyWeight, cGraph, cGraphTransforms, cGraphSize, cKNNIdxs);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void computeJTJAndJTrContribJDepthKernel(Vec2i* jTJContribPairs,
                                                    unsigned int* rowNumbers,
                                                    unsigned int* jTJContribPairsSize,
                                                    Vec2i* jTrContribElems,
                                                    unsigned int* jTrContribElemsSize,
                                                    const Vec2i* cCorrIdxs,
                                                    unsigned int corrIdxsSize,
                                                    const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                                    const Vecf<Constants::motionGraphKNN>* cKnnWeights) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= corrIdxsSize)
        return;

    Vec2i corrIdx                                  = cCorrIdxs[cIdx];
    Vecui<Constants::motionGraphKNN> cCurrKnnIdxs   = cKnnIdxs[corrIdx[0]];
    Vecf<Constants::motionGraphKNN> cCurrKnnWeights = cKnnWeights[corrIdx[0]];

    unsigned int pairsIdx = cIdx * combinations(Constants::motionGraphKNN, 2);
    for (std::size_t i = 0; i < Constants::motionGraphKNN; ++i) {
        if (cCurrKnnWeights[i] == 0.f)
            continue;
        for (std::size_t j = i; j < Constants::motionGraphKNN; ++j) {
            if (cCurrKnnWeights[j] == 0.f)
                continue;

            unsigned int lastSize = atomicAdd(jTJContribPairsSize, 1);
            rowNumbers[lastSize]  = cIdx;
            if (cCurrKnnIdxs[i] < cCurrKnnIdxs[j])
                jTJContribPairs[lastSize] = Vec2i(cCurrKnnIdxs[i], cCurrKnnIdxs[j]);
            else
                jTJContribPairs[lastSize] = Vec2i(cCurrKnnIdxs[j], cCurrKnnIdxs[i]);
        }
        // All non zero elements will contribute to JTr
        unsigned int insertIdx     = atomicAdd(jTrContribElemsSize, 1);
        jTrContribElems[insertIdx] = Vec2i(cIdx, cCurrKnnIdxs[i]);
    }
}

void runComputeJTJAndJTrContribJDepthKernel(Vec2i* jTJContribPairs,
                                            unsigned int* rowNumbers,
                                            unsigned int* jTJContribPairsSize,
                                            Vec2i* jTrContribElems,
                                            unsigned int* jTrContribElemsSize,
                                            const Vec2i* cCorrIdxs,
                                            unsigned int corrIdxsSize,
                                            const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                            const Vecf<Constants::motionGraphKNN>* cKnnWeights) {
    dim3 block, grid;
    setupBlockGrid(block, grid, corrIdxsSize);
    computeJTJAndJTrContribJDepthKernel<<<grid, block>>>(jTJContribPairs, rowNumbers, jTJContribPairsSize, jTrContribElems,
                                                         jTrContribElemsSize, cCorrIdxs, corrIdxsSize, cKnnIdxs, cKnnWeights);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void computeJTJAndJTrContribJMRegKernel(Vec2i* jTJContribPairs,
                                                   unsigned int* rowNumbers,
                                                   unsigned int rowNumberOffset,
                                                   unsigned int* jTJContribPairsSize,
                                                   Vec2i* jTrContribElems,
                                                   unsigned int* jTrContribElemsSize,
                                                   const unsigned int cGraphSize,
                                                   const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cGraphSize)
        return;

    auto cCurrKNNIdxs = cKNNIdxs[cIdx];
    for (std::size_t i = 0; i < Constants::energyMRegKNN; ++i) {
        const unsigned int connectedNodeId = cCurrKNNIdxs[i];
        if (connectedNodeId >= cGraphSize) {
            continue;
        }

        const int cRowId = (cIdx * (Constants::energyMRegKNN) + i) * 3 + rowNumberOffset;
        // if (cIdx == cCurrKNNIdxs[i])
        //     continue;

        for (std::size_t j = 0; j < 3; ++j) {
            unsigned int lastSize = atomicAdd(jTJContribPairsSize, 3);
            rowNumbers[lastSize]  = cRowId + j;
            if (cIdx < connectedNodeId)
                jTJContribPairs[lastSize] = Vec2i(cIdx, connectedNodeId);
            else
                jTJContribPairs[lastSize] = Vec2i(connectedNodeId, cIdx);

            rowNumbers[lastSize + 1]      = cRowId + j;
            jTJContribPairs[lastSize + 1] = Vec2i(cIdx, cIdx);
            rowNumbers[lastSize + 2]      = cRowId + j;
            jTJContribPairs[lastSize + 2] = Vec2i(connectedNodeId, connectedNodeId);

            // All elements will contibute to JTr
            unsigned int insertIdx         = atomicAdd(jTrContribElemsSize, 2);
            jTrContribElems[insertIdx]     = Vec2i(cRowId + j, cIdx);
            jTrContribElems[insertIdx + 1] = Vec2i(cRowId + j, connectedNodeId);
        }
    }
}

void runComputeJTJAndJTrContribJMRegKernel(Vec2i* jTJContribPairs,
                                           unsigned int* rowNumbers,
                                           unsigned int rowNumberOffset,
                                           unsigned int* jTJContribPairsSize,
                                           Vec2i* jTrContribElems,
                                           unsigned int* jTrContribElemsSize,
                                           const unsigned int cGraphSize,
                                           const Vecui<Constants::energyMRegKNN>* cKNNIdxs) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cGraphSize);
    computeJTJAndJTrContribJMRegKernel<<<grid, block>>>(jTJContribPairs, rowNumbers, rowNumberOffset, jTJContribPairsSize,
                                                        jTrContribElems, jTrContribElemsSize, cGraphSize, cKNNIdxs);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void multiplyJTJContribPairsKernel(Mat6f* multipliedContribPairs,
                                              const Vec2i* cJTJContribPairs,
                                              const unsigned int* cRowNumbers,
                                              unsigned int jTJContribPairsSize,
                                              const Vec6f* cJ,
                                              const unsigned int cJWidth) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= jTJContribPairsSize)
        return;

    Vec2i currContribPairIdxs    = cJTJContribPairs[cIdx];
    unsigned int currRowNumber   = cRowNumbers[cIdx];
    unsigned int j0Idx           = currRowNumber * cJWidth + currContribPairIdxs[0];
    unsigned int j1Idx           = currRowNumber * cJWidth + currContribPairIdxs[1];
    multipliedContribPairs[cIdx] = cJ[j0Idx] * cJ[j1Idx].transpose();
}

void runMultiplyJTJContribPairsKernel(Mat6f* multipliedContribPairs,
                                      const Vec2i* cJTJContribPairs,
                                      const unsigned int* cRowNumbers,
                                      unsigned int jTJContribPairsSize,
                                      const Vec6f* cJ,
                                      const unsigned int cJWidth) {
    dim3 block, grid;
    setupBlockGrid(block, grid, jTJContribPairsSize);
    multiplyJTJContribPairsKernel<<<grid, block>>>(multipliedContribPairs, cJTJContribPairs, cRowNumbers, jTJContribPairsSize, cJ,
                                                   cJWidth);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void multiplyJTrContribElemsKernel(Vec6f* multipliedJTrContribElems,
                                              const Vec2i* cJTrContribElems,
                                              const unsigned int cJTrContribElemsSize,
                                              const Vec6f* cJ,
                                              const float* cr,
                                              const unsigned int cJWidth) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cJTrContribElemsSize)
        return;

    Vec2i currJTrContribElem        = cJTrContribElems[cIdx];
    unsigned int idxJ               = currJTrContribElem[0] * cJWidth + currJTrContribElem[1];
    multipliedJTrContribElems[cIdx] = cJ[idxJ] * cr[currJTrContribElem[0]];
}

void runMultiplyJTrContribElemsKernel(Vec6f* multipliedJTrContribElems,
                                      const Vec2i* cJTrContribElems,
                                      const unsigned int cJTrContribElemsSize,
                                      const Vec6f* cJ,
                                      const float* cr,
                                      const unsigned int cJWidth) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cJTrContribElemsSize);
    multiplyJTrContribElemsKernel<<<grid, block>>>(multipliedJTrContribElems, cJTrContribElems, cJTrContribElemsSize, cJ, cr,
                                                   cJWidth);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void fillJTJKernel(float* JTJ,
                              const Vec2i* cJTJElemsCoords,
                              const Mat6f* cJTJElems,
                              unsigned int jTJElemsSize,
                              const unsigned int cJTJWidth) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= jTJElemsSize)
        return;

    Vec2i currCoords = cJTJElemsCoords[cIdx];
    Mat6f currElem   = cJTJElems[cIdx];

    unsigned int globalOffset1 = cJTJWidth * 6 * currCoords[0] + 6 * currCoords[1];
    for (std::size_t i = 0; i < currElem.rows(); ++i) {
        for (std::size_t j = 0; j < currElem.cols(); ++j) {
            unsigned int idx1 = globalOffset1 + cJTJWidth * i + j;
            JTJ[idx1]         = currElem(i, j);
        }
    }

    if (currCoords[0] == currCoords[1])
        return;

    // fillling symmetric
    unsigned int globalOffset2 = cJTJWidth * 6 * currCoords[1] + 6 * currCoords[0];
    for (std::size_t i = 0; i < currElem.rows(); ++i) {
        for (std::size_t j = 0; j < currElem.cols(); ++j) {
            unsigned int idx2 = globalOffset2 + cJTJWidth * i + j;
            JTJ[idx2]         = currElem(j, i);
        }
    }
}

void runFillJTJKernel(float* JTJ,
                      const Vec2i* cJTJElemsCoords,
                      const Mat6f* cJTJElems,
                      unsigned int jTJElemsSize,
                      const unsigned int cJTJWidth) {
    dim3 block, grid;
    setupBlockGrid(block, grid, jTJElemsSize);
    fillJTJKernel<<<grid, block>>>(JTJ, cJTJElemsCoords, cJTJElems, jTJElemsSize, cJTJWidth);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void fillJTrKernel(Vec6f* JTr,
                              const Vec2i* cJTrElemsCoordinates,
                              const Vec6f* cJTrElems,
                              const unsigned int cJTrElemsSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cJTrElemsSize)
        return;

    Vec2i currJTrElemsCoordinates   = cJTrElemsCoordinates[cIdx];
    JTr[currJTrElemsCoordinates[1]] = cJTrElems[cIdx];
}

void runFillJTrKernel(Vec6f* JTr, const Vec2i* cJTrElemsCoordinates, const Vec6f* cJTrElems, const unsigned int cJTrElemsSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cJTrElemsSize);
    fillJTrKernel<<<grid, block>>>(JTr, cJTrElemsCoordinates, cJTrElems, cJTrElemsSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void updateTransformsKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= transformsSize)
        return;

    Mat4f currTransform = transforms[cIdx];
    Vec6f currUpdate    = cUpdates[cIdx];
    Mat4f updateMat;
    // clang-format off
    updateMat <<             0, -currUpdate[2],  currUpdate[1], currUpdate[3],
                 currUpdate[2],              0, -currUpdate[0], currUpdate[4],
                -currUpdate[1],  currUpdate[0],              0, currUpdate[5],
                             0,              0,              0,             0;
    // clang-format on
    transforms[cIdx] = currTransform - updateMat;
}

void runUpdateTransformsKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, transformsSize);
    updateTransformsKernel<<<grid, block>>>(transforms, cUpdates, transformsSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void updateTransformsReverseKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= transformsSize)
        return;

    Mat4f currTransform = transforms[cIdx];
    Vec6f currUpdate    = cUpdates[cIdx];
    Mat4f updateMat;
    // clang-format off
    updateMat <<             0, -currUpdate[2],  currUpdate[1], currUpdate[3],
                 currUpdate[2],              0, -currUpdate[0], currUpdate[4],
                -currUpdate[1],  currUpdate[0],              0, currUpdate[5],
                             0,              0,              0,             0;
    // clang-format on
    transforms[cIdx] = currTransform + updateMat;
}

void runUpdateTransformsReverseKernel(Mat4f* transforms, const Vec6f* cUpdates, unsigned int transformsSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, transformsSize);
    updateTransformsKernel<<<grid, block>>>(transforms, cUpdates, transformsSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

// __global__ void fillJTJSparse() {
//     const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
//     if (cIdx >= jTJContribPairsSize)
//         return;
// }

}  // namespace af
