#ifndef DEVICETSDF_CUH
#define DEVICETSDF_CUH

#include <thrust/device_vector.h>

#include "af/DeviceBuffer.cuh"
#include "af/MarchingCubes.cuh"
#include "af/TSDFIntegration.cuh"
#include "af/eigen_extension.cuh"

namespace af {

template<typename Precision = float>
class DeviceTsdf {
public:
    DeviceTsdf() {}
    DeviceTsdf(const Vec3f& volSize, const Vec3i& volDim, const Vec3f& center, const Mat3f& rotation, float delta = 0.01f)
        : _tsdf_d(volDim[0] * volDim[1] * volDim[2], -1.f), _weights_d(volDim[0] * volDim[1] * volDim[2], 0.f), _delta(delta),
          _volDim(volDim), _volSize(volSize), _center(center), _rotation(rotation) {}
    DeviceTsdf(const Vec3f& volSize, const Vec3i& volDim, float delta = 0.01f)
        : DeviceTsdf(volSize, volDim, Vec3f::Zero(), Mat3f::Identity(), delta) {}

    void initDefault() {
        thrust::fill(_tsdf_d.begin(), _tsdf_d.end(), -1.f);
        thrust::fill(_weights_d.begin(), _weights_d.end(), 0.f);
    }

    void integrate(const af::DeviceDepthImage& depthFrame_d) {
        af::runIntegrateTSDFFromFrameKernel(tsdfPtr(), weightsPtr(), depthFrame_d.dataPtr(),
                                            Vec2i(depthFrame_d.width(), depthFrame_d.height()), _delta, _volDim, voxelSize(),
                                            _volSize, rotation(), center(), depthFrame_d.depthIntrinsics());
    }

    void extractPointCloud(af::DeviceBufferCounted<Vec3f>& pointCloud) {
        af::runMarchingCubesMeshKernel(pointCloud.bufferPtr(), pointCloud.size_dPtr(), this->tsdfPtr(), this->weightsPtr(),
                                       _volDim, voxelSize(), _volSize, 0);
        pointCloud.syncHostSize();

        // Remove duplicates
        thrust::sort(pointCloud.begin(), pointCloud.end());
        auto itEndUnique = thrust::unique(pointCloud.begin(), pointCloud.end());
        pointCloud.setSize(itEndUnique - pointCloud.begin());
    }

    void setDelta(float delta) { _delta = delta; }
    float delta() const { return _delta; }

    void setVolDim(const Vec3i& volDim) {
        _volDim = volDim;
        _tsdf_d.resize(_volDim[0] * _volDim[1] * _volDim[2], -1.f);
        _weights_d.resize(_volDim[0] * _volDim[1] * _volDim[2], 0.f);
    }
    Vec3i volDim() const { return _volDim; }

    void setVolSize(const Vec3f& volSize) { _volSize = volSize; }
    Vec3f volSize() const { return _volSize; }

    void setCenter(const Vec3f& center) { _center = center; }
    Vec3f center() const { return _center; }

    void setRotation(const Mat3f& rotation) { _rotation = rotation; }
    Mat3f rotation() const { return _rotation; }

    Vec3f voxelSize() const { return _volSize.cwiseQuotient(_volDim.cast<float>()); }
    std::size_t voxelCount() const { return _volDim[0] * _volDim[1] * _volDim[2]; }

    Precision* tsdfPtr() { return _tsdf_d.data().get(); }
    const Precision* tsdfPtr() const { return _tsdf_d.data().get(); }

    Precision* weightsPtr() { return _weights_d.data().get(); }
    const Precision* weightsPtr() const { return _weights_d.data().get(); }

    const thrust::device_vector<Precision>& tsdf_d() const { return _tsdf_d; }
    const thrust::device_vector<Precision>& weights_d() const { return _weights_d; }

private:
    thrust::device_vector<Precision> _tsdf_d;
    thrust::device_vector<Precision> _weights_d;
    float _delta;
    Vec3i _volDim;
    Vec3f _volSize;
    Vec3f _center;
    Mat3f _rotation;
};

template class DeviceTsdf<float>;

}  // namespace af

#endif