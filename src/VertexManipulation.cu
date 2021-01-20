#include <af/Helper.cuh>
#include <af/VertexManipulation.cuh>

namespace af {

__global__ void filterDepthKernel(float* depth, unsigned int depthSize, float threshold, float fillValue) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= depthSize)
        return;

    float val = depth[cIdx];
    if (val > threshold || val < 1.e-4f || isnan(val)) {
        depth[cIdx]     = fillValue;
    }
}

void runFilterDepthKernel(float* depth, unsigned int depthSize, float threshold, float fillValue) {
    dim3 block, grid;
    setupBlockGrid(block, grid, depthSize);
    filterDepthKernel<<<grid, block>>>(depth, depthSize, threshold, fillValue);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void maskDepthKernel(float* depth, const bool* cDepthMask, unsigned int depthSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= depthSize)
        return;

    if (!cDepthMask[cIdx]) {
        depth[cIdx] = std::numeric_limits<float>::infinity();
    }
}

void runMaskDepthKernel(float* depth, const bool* cDepthMask, unsigned int depthSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, depthSize);
    maskDepthKernel<<<grid, block>>>(depth, cDepthMask, depthSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void depthToMeshKernel(Vec3f* mesh,
                                  const float* cDepth,
                                  unsigned int depthWidth,
                                  unsigned int depthHeight,
                                  const Mat3f K) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    const int cY = threadIdx.y + blockDim.y * blockIdx.y;
    if (cX >= depthWidth || cY >= depthHeight)
        return;

    const int cId    = cY * depthWidth + cX;
    float pixelDepth = cDepth[cId];
    if (std::isinf(pixelDepth) || pixelDepth < 1.e-4f) {
        mesh[cId] = Vec3f(0.f, 0.f, 0.f);
        return;
    }
    float x0 = (float(cX) - K(0, 2)) * (1 / K(0, 0));
    float y0 = (float(cY) - K(1, 2)) * (1 / K(1, 1));

    mesh[cId] = Vec3f(x0 * pixelDepth, y0 * pixelDepth, pixelDepth);
}

void runDepthToMeshKernel(Vec3f* mesh, const float* cDepth, unsigned int depthWidth, unsigned int depthHeight, const Mat3f& K) {
    dim3 block, grid;
    setupBlockGrid(block, grid, depthWidth, depthHeight);
    depthToMeshKernel<<<grid, block>>>(mesh, cDepth, depthWidth, depthHeight, K);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void translateKernel(Vec3f* mesh, unsigned int meshSize, Vec3f translation) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    if (cX >= meshSize)
        return;

    mesh[cX] += translation;
}

void runTranslateKernel(Vec3f* mesh, unsigned int meshSize, Vec3f translation) {
    dim3 block, grid;
    setupBlockGrid(block, grid, meshSize);
    translateKernel<<<grid, block>>>(mesh, meshSize, translation);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void transformKernel(Vec3f* mesh, unsigned int meshSize, Mat4f transform) {
    const int cX = threadIdx.x + blockDim.x * blockIdx.x;
    if (cX >= meshSize)
        return;

    mesh[cX] = (transform * mesh[cX].homogeneous()).head(3);
}

void runTransformKernel(Vec3f* mesh, unsigned int meshSize, Mat4f transform) {
    dim3 block, grid;
    setupBlockGrid(block, grid, meshSize);
    transformKernel<<<grid, block>>>(mesh, meshSize, transform);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__device__ float weight(float radius, float distance) {
    return (distance >= radius) ? 0.f : exp(-(distance * distance) / (2 * pow(radius, 2)));
}

__global__ void computeWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* cKnnWeights,
                                         const float* cMotionGraphRadiuses,
                                         const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                         const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                         const std::size_t cMeshSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cMeshSize)
        return;

    Vecf<Constants::motionGraphKNN> weights;

    auto cCurrKnnIdx   = cKnnIdxs[cIdx];
    auto cCurrKnnDists = cKnnDists[cIdx];
    float weightsSum   = 0;
    for (size_t j = 0; j < Constants::motionGraphKNN; j++) {
        weights[j] = af::weight(cMotionGraphRadiuses[cCurrKnnIdx[j]], cCurrKnnDists[j]);
        weightsSum += weights[j];
    }

    if (weightsSum != 0.f)
        weights /= weightsSum;

    cKnnWeights[cIdx] = weights;
}

void runComputeWarpWeightsKernel(Vecf<Constants::motionGraphKNN>* knnWeights,
                                 const float* cMotionGraphRadiuses,
                                 const Vecui<Constants::motionGraphKNN>* cKnnIdxs,
                                 const Vecf<Constants::motionGraphKNN>* cKnnDists,
                                 const std::size_t cMeshSize) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cMeshSize);

    computeWarpWeightsKernel<<<grid, block>>>(knnWeights, cMotionGraphRadiuses, cKnnIdxs, cKnnDists, cMeshSize);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void depthVerticesNormalsKernel(Vec3f* normals,
                                           Vec3f* vertices,
                                           const float* depth,
                                           int h,
                                           int w,
                                           float normExclAngleStart,
                                           float normExclAngleEnd) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x + 1;
    const int y = threadIdx.y + blockDim.y * blockIdx.y + 1;
    if (x >= w - 1 || y >= h - 1)
        return;

    const int cId    = y * w + x;
    Vec3f currVertex = vertices[cId];
    if (currVertex[2] < 1.e-4f) {
        normals[cId] = Vec3f(0.f, 0.f, 0.f);
        return;
    }

    // clang-format off
    int iXl =  y      * w + (x - 1);
    int iXr =  y      * w + (x + 1);
    int iYt = (y - 1) * w +  x;
    int iYb = (y + 1) * w +  x;
    // clang-format on

    Vec3f cVerticesXr            = vertices[iXr];
    Vec3f cVerticesXl            = vertices[iXl];
    Vec3f cVerticesYb            = vertices[iYb];
    Vec3f cVerticesYt            = vertices[iYt];
    bool isNotAllNeighboursValid = cVerticesXr[2] < 1.e-4f || cVerticesXl[2] < 1.e-4f || cVerticesYb[2] < 1.e-4f
                                   || cVerticesYt[2] < 1.e-4f;
    if (isNotAllNeighboursValid) {
        normals[cId] = Vec3f(0.f, 0.f, 0.f);
        return;
    }
    Vec3f diffX = cVerticesXr - cVerticesXl;
    Vec3f diffY = cVerticesYb - cVerticesYt;
    Vec3f gX    = (diffX) / ((diffX).head(2).norm());
    Vec3f gY    = (diffY) / ((diffY).head(2).norm());
    Vec3f norm  = gY.cross(gX);
    norm.normalize();

// Vec3f projectionVec = (currVertex - Vec3f(x, y, 0)).normalized();
#if 1
    Vec3f projectionVec(0.f, 0.f, 1.f);
    float dot            = projectionVec.dot(norm);
    float angle          = acos(dot);
    bool isAngleNotValid = angle > normExclAngleStart && angle < normExclAngleEnd;
    if (isAngleNotValid) {
        normals[cId] = Vec3f(0.f, 0.f, 0.f);
        return;
    }
#endif
    // dot = x1*x2 + y1*y2 + z1*z2
    // lenSq1 = x1*x1 + y1*y1 + z1*z1
    // lenSq2 = x2*x2 + y2*y2 + z2*z2
    // angle = acos(dot/sqrt(lenSq1 * lenSq2))

    normals[cId] = norm;
}

void runDepthVerticesNormalsKernel(Vec3f* normals,
                                   Vec3f* vertices,
                                   const float* depth,
                                   int h,
                                   int w,
                                   std::pair<float, float> normExclAngleRange) {
    dim3 block, grid;
    setupBlockGrid(block, grid, w - 2, h - 2);

    depthVerticesNormalsKernel<<<grid, block>>>(normals, vertices, depth, h, w, normExclAngleRange.first, normExclAngleRange.second);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void removeDepthWithoutNormalsKernel(Vec3f* normals, Vec3f* vertices, float* depth, int h, int w) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x + 1;
    const int y = threadIdx.y + blockDim.y * blockIdx.y + 1;
    if (x >= w - 1 || y >= h - 1)
        return;

    const int cId    = y * w + x;
    Vec3f currNormal = normals[cId];
    if (currNormal.isZero(1.e-4f)) {
        vertices[cId] = Vec3f(0.f, 0.f, 0.f);
        depth[cId]    = std::numeric_limits<float>::infinity();
        return;
    }
}

void runRemoveDepthWithoutNormalsKernel(Vec3f* normals, Vec3f* vertices, float* depth, int h, int w) {
    dim3 block, grid;
    setupBlockGrid(block, grid, w - 2, h - 2);

    removeDepthWithoutNormalsKernel<<<grid, block>>>(normals, vertices, depth, h, w);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void rejectInvalidTransformsKernel(Mat4f* transforms, const Mat4f* validTransforms, const Vec3f* graphNodes, const std::size_t graphSize, float threshold) {
    const int cGraphId = threadIdx.x + blockDim.x * blockIdx.x;
    if (cGraphId >= graphSize)
        return;
    
    Mat4f pendingTransform = transforms[cGraphId];
    Mat4f validTransform = validTransforms[cGraphId];
    Vec3f graphNode = graphNodes[cGraphId];
    Vec3f validWarpedNode = (validTransform * graphNode.homogeneous()).head(3);
    Vec3f warpedNode = (pendingTransform * graphNode.homogeneous()).head(3);

    if ((warpedNode - validWarpedNode).norm() < threshold)
        return;

    transforms[cGraphId] = validTransforms[cGraphId];
}

void runRejectInvalidTransforms(Mat4f* transforms, const Mat4f* validTransforms, const Vec3f* graphNodes, const std::size_t graphSize, float threshold) {
    dim3 block, grid;
    setupBlockGrid(block, grid, graphSize);
    rejectInvalidTransformsKernel<<<grid, block>>>(transforms, validTransforms, graphNodes, graphSize, threshold);
    CUDA_CHECK;
    cudaDeviceSynchronize();
}

}  // namespace af