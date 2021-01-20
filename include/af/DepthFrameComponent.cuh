#ifndef DEPTHFRAMECOMPONENT_CUH
#define DEPTHFRAMECOMPONENT_CUH

#include <thrust/device_vector.h>

#include <opencv2/core.hpp>

#include "af/DeviceImage.cuh"
#include "af/eigen_extension.h"
#include "af/CameraModel.h"

namespace af {

class Settings;

class DepthFrameComponent;

class DepthFrameProcessor {
public:
    explicit DepthFrameProcessor(const CameraModel& camera, const af::Settings& settings);

    // template<typename Type>
    void loadFrame(size_t index, DepthFrameComponent& depthFrame_d);

private:
    cv::Mat _depthFrame;
    const CameraModel& _camera;
    const af::Settings& _settings;
};

void computeCentroid(const thrust::device_vector<Vec3f>& pointCloud_d, Vec3f& centroid);

class DepthFrameComponent {
public:
    DepthFrameComponent(const CameraModel& camera) : _depth_d(camera) {}

    void resize(std::size_t height, std::size_t width) {
        _depth_d.resize(height, width);
        _pointCloud_d.resize(height * width);
        _normals_d.resize(height * width);
    }

    af::DeviceDepthImage& depth() { return _depth_d; }
    thrust::device_vector<Vec3f>& pointCloud() { return _pointCloud_d; }
    thrust::device_vector<Vec3f>& normals() { return _normals_d; }
    const af::DeviceDepthImage& depth() const { return _depth_d; }
    const thrust::device_vector<Vec3f>& pointCloud() const { return _pointCloud_d; }
    const thrust::device_vector<Vec3f>& normals() const { return _normals_d; }

private:
    af::DeviceDepthImage _depth_d;
    thrust::device_vector<Vec3f> _pointCloud_d;
    thrust::device_vector<Vec3f> _normals_d;
};

}  // namespace af

#endif