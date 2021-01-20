#ifndef DEVICEIMAGE_CUH
#define DEVICEIMAGE_CUH

#include <thrust/device_vector.h>

#include "af/VertexManipulation.cuh"
#include "af/eigen_extension.h"
#include "af/CameraModel.h"

namespace af {

template<typename Type>
class DeviceImage {
public:
    using storage        = thrust::device_vector<Type>;
    using iterator       = typename storage::iterator;
    using const_iterator = typename storage::const_iterator;

    DeviceImage() {}
    DeviceImage(std::size_t height, std::size_t width) : _height(height), _width(width) {}

    void nullifyUpperBound(Type upperBound) { af::runFilterDepthKernel(dataPtr(), size(), upperBound); }

    Type* dataPtr() { return _image_d.data().get(); }
    const Type* dataPtr() const { return _image_d.data().get(); }

    void resize(std::size_t height, std::size_t width) {
        _height = height;
        _width  = width;
        _image_d.resize(_height * _width);
    }

    iterator begin() { return _image_d.begin(); }
    const_iterator begin() const { return _image_d.begin(); }

    const storage& image_d() const { return _image_d; }
    std::size_t height() const { return _height; }
    std::size_t width() const { return _width; }

    std::size_t size() const { return _height * _width; }

private:
    storage _image_d;
    std::size_t _height;
    std::size_t _width;
};

class DeviceDepthImage : public DeviceImage<float> {
public:
    DeviceDepthImage(const CameraModel& camera) : _camera(camera) {}
    DeviceDepthImage(std::size_t height, std::size_t width, const CameraModel& camera)
        : DeviceImage(height, width), _camera(camera) {}

    void backProject(thrust::device_vector<Vec3f>& pointCloud) {
        pointCloud.resize(size());
        af::runDepthToMeshKernel(pointCloud.data().get(), dataPtr(), width(), height(), camera().depthIntrinsics);
        af::runTransformKernel(pointCloud.data().get(), pointCloud.size(), camera().transfW2C.inverse());
    }

    const CameraModel& camera() const { return _camera; }
    const Mat3f& depthIntrinsics() const { return _camera.depthIntrinsics; }

private:
    const CameraModel& _camera;
};

}  // namespace af

#endif