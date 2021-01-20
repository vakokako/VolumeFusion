#include "af/DeviceImage.cuh"
#include "af/KinectReader.h"

#include <iomanip>

namespace af {

namespace cuda {

template<typename HostReader = af::KinectDepthReader>
class DepthReader {
public:
    explicit DepthReader(std::string folder)
        : _hostReader(std::move(folder)) {}

    void readNext(DeviceDepthImage& depth_d);
    HostReader& hostReader() { return _hostReader; }
    const HostReader& hostReader() const { return _hostReader; }

    const cv::Mat& hostBuffer() const { return _depthBuffer; }

private:
    cv::Mat _depthBuffer;
    HostReader _hostReader;
};

template<typename HostReader>
void DepthReader<HostReader>::readNext(DeviceDepthImage& depth_d) {
    _hostReader.readNext(_depthBuffer);

    depth_d.resize(_depthBuffer.rows, _depthBuffer.cols);

    thrust::copy_n((float*)_depthBuffer.data, _depthBuffer.total(), depth_d.begin());
}

template class DepthReader<af::KinectDepthReader>;

}  // namespace cuda
}  // namespace af