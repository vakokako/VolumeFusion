#include <thrust/device_vector.h>

#include "af/Algorithms.cuh"
#include "af/BilateralFilter.cuh"
#include "af/DeviceImage.cuh"
#include "af/eigen_extension.h"

namespace af {

void bilateralFilter(DeviceImage<float>& out, const DeviceImage<float>& in, int r, double sI, double sS, float thr) {
    out.resize(in.height(), in.width());
    af::bilateralFilterTextureOpmShared(out.dataPtr(), in.dataPtr(), in.height(), in.width(), r, sI, sS, thr);
}

void computeNormals(thrust::device_vector<Vec3f>& normals,
                    thrust::device_vector<Vec3f>& vertices,
                    DeviceImage<float>& depth,
                    std::pair<float, float> normAngleExclRange) {
    af::runDepthVerticesNormalsKernel(normals.data().get(), vertices.data().get(), depth.dataPtr(), depth.height(), depth.width(),
                                      normAngleExclRange);
    af::runRemoveDepthWithoutNormalsKernel(normals.data().get(), vertices.data().get(), depth.dataPtr(), depth.height(),
                                           depth.width());
}

}  // namespace af
