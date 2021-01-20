#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "af/DataTypes.cuh"
#include "af/Settings.h"
#include "af/DepthFrameComponent.cuh"
#include "af/VertexManipulation.cuh"
#include "af/dataset.h"
#include "af/eigen_extension.cuh"
#include "af/eigen_extension.h"

namespace af {

DepthFrameProcessor::DepthFrameProcessor(const CameraModel& camera, const af::Settings& settings)
    : _camera(camera), _settings(settings) {}

void DepthFrameProcessor::loadFrame(size_t index, DepthFrameComponent& depthFrame_d) {
    af::loadDepthFrame(_settings.dataFolder, index, _depthFrame);

    /////////  FRAME RPOCESSING  ////////////////

    depthFrame_d.resize(_depthFrame.rows, _depthFrame.cols);
    // depthFrame_d.depth().resize();
    // depthFrame_d.pointCloud().resize(_depthFrame.total());
    // depthFrame_d.normals().resize(_depthFrame.total());

    thrust::copy_n((float*)_depthFrame.data, _depthFrame.total(), depthFrame_d.depth().begin());

    af::runFilterDepthKernel(depthFrame_d.depth().dataPtr(), _depthFrame.total(),
                             _settings.depthThreshold);
    cudaDeviceSynchronize();

    if (_settings.bilateralFiltr) {
        cv::Mat filteredDepth(_depthFrame.size(), _depthFrame.type());
        thrust::copy_n(depthFrame_d.depth().begin(), _depthFrame.total(), (float*)filteredDepth.data);
        cv::bilateralFilter(filteredDepth, _depthFrame, _settings.bilateralD, _settings.bilateralSigmaI,
                            _settings.bilateralSigmaI, cv::BORDER_REPLICATE);
        thrust::copy_n((float*)_depthFrame.data, _depthFrame.total(), depthFrame_d.depth().begin());
        // af::runMaskDepthKernel(depthFrame_d.depth().dataPtr(), _depthFrameMask_d.data().get(), _depthFrame.total());
    }

    af::runDepthToMeshKernel(depthFrame_d.pointCloud().data().get(), depthFrame_d.depth().dataPtr(), _depthFrame.cols, _depthFrame.rows,
                             _camera.depthIntrinsics);
    cudaDeviceSynchronize();
    af::runDepthVerticesNormalsKernel(depthFrame_d.normals().data().get(), depthFrame_d.pointCloud().data().get(), depthFrame_d.depth().dataPtr(),
                                      _depthFrame.rows, _depthFrame.cols, _settings.normExclAngleRange);
    af::runRemoveDepthWithoutNormalsKernel(depthFrame_d.normals().data().get(), depthFrame_d.pointCloud().data().get(), depthFrame_d.depth().dataPtr(),
                                           _depthFrame.rows, _depthFrame.cols);
    cudaDeviceSynchronize();
}

void computeCentroid(const thrust::device_vector<Vec3f>& pointCloud_d, Vec3f& centroid) {
    centroid = thrust::reduce(pointCloud_d.begin(), pointCloud_d.end(), Vec3f(0, 0, 0));
    cudaDeviceSynchronize();
    centroid /= thrust::count_if(thrust::device, pointCloud_d.begin(), pointCloud_d.end(), hasPositiveDepth());
}

}  // namespace af