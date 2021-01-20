#ifndef ALGORITHMS_CUH
#define ALGORITHMS_CUH

#include "af/eigen_extension.h"

#include <thrust/device_vector.h>
#include <utility>


namespace af {
template<typename Type>
class DeviceImage;

void bilateralFilter(DeviceImage<float>& out, const DeviceImage<float>& in, int r, double sI, double sS, float thr);

void computeNormals(thrust::device_vector<Vec3f>& normals,
                    thrust::device_vector<Vec3f>& vertices,
                    DeviceImage<float>& depth,
                    std::pair<float, float> normAngleExclRange);

}  // namespace af

#endif