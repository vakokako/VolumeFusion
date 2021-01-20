#include "af/DataTypes.cuh"
#include "af/Settings.h"
#include "af/DeviceMotionGraph.cuh"
#include "af/GraphBuilder.h"
#include "af/Helper.cuh"
#include "af/WarpField.cuh"

namespace af {

// void GraphBuilder::spreadMotionGraph(const af::DeviceBuffer<Vec3f>& pointCloud_d) {
//     _pointCloudCpu.resize(pointCloud_d.size());
//     copyToHost(_pointCloudCpu.data(), pointCloud_d.bufferPtr(), pointCloud_d.size());

//     if (_settings.updateGraph) {
//         af::buildGraph(_motionGraphCpu, _pointCloudCpu, Settings::motionGraphKNN, _settings.motionGraphRadius);
//     }

//     const std::size_t cUpdatedGraphSize = _motionGraphCpu.graph().size();
//     const std::size_t cCountNewNodes    = cUpdatedGraphSize - _motionGraph_d.count();

//     copyToDevice(_motionGraph_d.pointsPtr() + _motionGraph_d.count(),
//                  _motionGraphCpu.graph().vec_.data() + _motionGraph_d.count(), cCountNewNodes);
//     copyToDevice(_motionGraph_d.transformsPtr() + _motionGraph_d.count(),
//                  _motionGraphCpu.transforms().data() + _motionGraph_d.count(), cCountNewNodes);
//     copyToDevice(_motionGraph_d.radiusesPtr() + _motionGraph_d.count(),
//                  _motionGraphCpu.radiuses().data() + _motionGraph_d.count(), cCountNewNodes);
//     _motionGraph_d.setCount(cUpdatedGraphSize);
// }

void WarpField::computeGraphKnns() {
    af::getKnnData(_graphKnnIdxs, _graphKnnDists, _motionGraphCpu, _motionGraphCpu.graph().vec_, true);
    copyToDevice(_motionGraphknnIdxs_d.bufferPtr(), _graphKnnIdxs.data(), _graphKnnIdxs.size());
    _motionGraphknnIdxs_d.setSize(_graphKnnIdxs.size());
}

void WarpField::computePointCloudKnns(af::DeviceKnn<Constants::motionGraphKNN>& knnResult_d) {
    af::getKnnData(_meshKnnIdxs, _meshKnnDists, _motionGraphCpu, _pointCloudCpu);
    copyToDevice(knnResult_d.idxsPtr(), _meshKnnIdxs.data(), _meshKnnIdxs.size());
    copyToDevice(knnResult_d.distsPtr(), _meshKnnDists.data(), _meshKnnDists.size());
    knnResult_d.setSize(_meshKnnIdxs.size());
}

void WarpField::computePointCloudWeightedKnns(af::DeviceKnnWeighted<Constants::motionGraphKNN>& knnResult_d) {
    this->computePointCloudKnns(knnResult_d);
    af::runComputeWarpWeightsKernel(knnResult_d.weightsPtr(), _motionGraph_d.radiusesPtr(), knnResult_d.idxsPtr(),
                                    knnResult_d.distsPtr(), knnResult_d.size());
}

// void GraphBuilder::getKNNResults(const af::DeviceBuffer<Vec3f>& pointCloud_d) {
//     // get knn for point cloud

//     CHECK_BUFF_SIZE(cGraphSize, BUFF_SIZE_SMALL);
// }

}  // namespace af