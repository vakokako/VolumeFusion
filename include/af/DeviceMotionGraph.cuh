#ifndef DEVICEMOTIONGRAPH_CUH
#define DEVICEMOTIONGRAPH_CUH

#include <thrust/device_vector.h>

#include "af/eigen_extension.cuh"

namespace af {

class DeviceMotionGraph {
public:
    DeviceMotionGraph(std::size_t size) : _points_d(size), _transforms_d(size), _radiuses_d(size) {}

    Vec3f* pointsPtr() { return _points_d.data().get(); }
    Mat4f* transformsPtr() { return _transforms_d.data().get(); }
    float* radiusesPtr() { return _radiuses_d.data().get(); }
    const Vec3f* pointsPtr() const { return _points_d.data().get(); }
    const Mat4f* transformsPtr() const { return _transforms_d.data().get(); }
    const float* radiusesPtr() const { return _radiuses_d.data().get(); }

    std::size_t bufferSize() const { return _size; }
    std::size_t count() const { return _count; }
    void setCount(std::size_t count) { _count = count; }

private:
    thrust::device_vector<Vec3f> _points_d;      // 120000
    thrust::device_vector<Mat4f> _transforms_d;  // 640000
    thrust::device_vector<float> _radiuses_d;    // 40000
    std::size_t _size;
    std::size_t _count;
};

}  // namespace af

#endif