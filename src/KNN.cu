#include <af/Helper.cuh>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

namespace af {

__global__ void computeDistsKernel(float* dists, const Vec3f* cRefPoints, const unsigned int cRefPointsSize, const Vec3f cInputPoint) {

    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= cRefPointsSize)
        return;

    dists[cIdx] = (cRefPoints[cIdx] - cInputPoint).norm();
}

void runComputeDistsKernel(float* dists, const Vec3f* cRefPoints, const unsigned int cRefPointsSize, const Vec3f cInputPoint) {
    dim3 block, grid;
    setupBlockGrid(block, grid, cRefPointsSize);
    computeDistsKernel<<<grid, block>>>(dists, cRefPoints,  cRefPointsSize, cInputPoint);
    cudaDeviceSynchronize();
}

template<class Idxs, class Dists>
void knn(const Vec3f* cRefPoints, unsigned int cRefPointsSize, const Vec3f cInputPoint, unsigned int knn, Idxs& idxs, Dists& dists) {
    thrust::device_vector<unsigned int> idxs_d(cRefPointsSize);
    thrust::sequence(idxs_d.begin(), idxs_d.end(), 0);
    thrust::device_vector<float> dists_d(cRefPointsSize);

    runComputeDistsKernel(dists_d.data().get(), cRefPoints, cRefPointsSize, cInputPoint);
    thrust::sort_by_key(dists_d.begin(), dists_d.end(), idxs_d.begin());

    thrust::copy_n(dists_d.begin(), knn, idxs.begin());
    thrust::copy_n(idxs_d.begin(), knn, dists.begin());
}

// template<class Idxs, class Dists>
// void knn(const Vec3f* cRefPoints, unsigned int cRefPointsSize, const Vec3f* cRefPoints, unsigned int knn, Idxs& idxs, Dists& dists) {
//     thrust::device_vector<unsigned int> idxs_d(cRefPointsSize);
//     thrust::sequence(idxs_d.begin(), idxs_d.end(), 0);
//     thrust::device_vector<float> dists_d(cRefPointsSize);

//     runComputeDistsKernel(dists_d.data().get(), cRefPoints, cRefPointsSize, cInputPoint);
//     thrust::sort_by_key(dists_d.begin(), dists_d.end(), idxs_d.begin());

//     thrust::copy_n(dists_d.begin(), knn, idxs.begin());
//     thrust::copy_n(idxs_d.begin(), knn, dists.begin());
// }

}