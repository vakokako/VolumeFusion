#ifndef WARPFIELD_CUH
#define WARPFIELD_CUH

#include "af/DeviceMotionGraph.cuh"
#include "af/GraphBuilder.h"
#include "af/Helper.cuh"
#include "af/MotionGraph.h"
#include "af/eigen_extension.cuh"
#include "af/DeviceBuffer.cuh"
#include "af/ComputeCorrespondence.cuh"


namespace af {

struct Settings;

template<int K>
class DeviceKnn;

class WarpField {
public:
    WarpField(std::size_t bufferSize, const af::Settings& settings)
        : _motionGraph_d(bufferSize), _motionGraphknnIdxs_d(bufferSize), _settings(settings) {}

    template<template<class> class Container>
    void spreadMotionGraph(const Container<Vec3f>& pointCloud_d);

    void computeGraphKnns();
    void computePointCloudKnns(af::DeviceKnn<Constants::motionGraphKNN>& knnResult_d);
    void computePointCloudWeightedKnns(af::DeviceKnnWeighted<Constants::motionGraphKNN>& knnResult_d);

    template<int K, template<class> class Container>
    void computeKnns(af::DeviceKnn<K>& knnResult_d, const Container<Vec3f>& pointCloud_d);

    template<int K>
    void warp(af::DeviceBuffer<Vec3f>& warpedPoints, const af::DeviceBuffer<Vec3f>& points, const af::DeviceKnnWeighted<K>& pointsKnn) {
        warpedPoints.setSize(points.size());
        af::runWarpKernel(warpedPoints.data().get(), points.data().get(), _motionGraph_d.transformsPtr(),
                          pointsKnn.idxsPtr(), pointsKnn.weightsPtr(), points.size());
    }

    // template<int K>
    // void getKNNResults(af::DeviceKnnResult<K>& knnResult_d, const af::DeviceBuffer<Vec3f>& pointCloud_d) {
    //     std::vector<Vecui<K>> _knnIdxs;
    //     std::vector<Vecf<K>> _knnDists;
    //     af::getKnnData(_knnIdxs, _knnDists, _motionGraphCpu, _pointCloudCpu);
    //     copyToDevice(knnResult_d.idxsPtr(), _knnIdxs.data(), _knnIdxs.size());
    //     copyToDevice(knnResult_d.distsPtr(), _knnDists.data(), _knnDists.size());
    // }
    // template<int K>
    // void getKNNResults(thrust::device_vector<Vecui<K>>& knnIdxs_d, const af::DeviceBuffer<Vec3f>& pointCloud_d) {
    //     std::vector<Vecui<K>> _knnIdxs;
    //     std::vector<Vecf<K>> _knnDists;
    //     af::getKnnData(_knnIdxs, _knnDists, _motionGraphCpu, _pointCloudCpu);
    //     copyToDevice(knnIdxs_d.data().get(), _knnIdxs.data(), _knnIdxs.size());
    // }

    af::DeviceMotionGraph& motionGraph_d() { return _motionGraph_d; }

private:
    af::DeviceMotionGraph _motionGraph_d;
    af::DeviceBuffer<Vecui<Constants::energyMRegKNN>> _motionGraphknnIdxs_d;
    // thrust::device_vector<Vecui<Constants::energyMRegKNN>> _motionGraphknnIdxs_d;

    // Cpu copies
    MotionGraph _motionGraphCpu;
    std::vector<Vec3f> _pointCloudCpu;

    std::vector<Vecui<Constants::motionGraphKNN>> _meshKnnIdxs;
    std::vector<Vecf<Constants::motionGraphKNN>> _meshKnnDists;
    std::vector<Vecui<Constants::energyMRegKNN>> _graphKnnIdxs;
    std::vector<Vecf<Constants::energyMRegKNN>> _graphKnnDists;

    const af::Settings& _settings;
};

template<template<class> class Container>
void WarpField::spreadMotionGraph(const Container<Vec3f>& pointCloud_d) {
    _pointCloudCpu.resize(pointCloud_d.size());
    thrust::copy_n(pointCloud_d.begin(), pointCloud_d.size(), _pointCloudCpu.begin());

    if (_settings.updateGraph) {
        af::buildGraph(_motionGraphCpu, _pointCloudCpu, Constants::motionGraphKNN, _settings.motionGraphRadius);
    }

    const std::size_t cUpdatedGraphSize = _motionGraphCpu.graph().size();
    const std::size_t cCountNewNodes    = cUpdatedGraphSize - _motionGraph_d.count();

    copyToDevice(_motionGraph_d.pointsPtr() + _motionGraph_d.count(),
                 _motionGraphCpu.graph().vec_.data() + _motionGraph_d.count(), cCountNewNodes);
    copyToDevice(_motionGraph_d.transformsPtr() + _motionGraph_d.count(),
                 _motionGraphCpu.transforms().data() + _motionGraph_d.count(), cCountNewNodes);
    copyToDevice(_motionGraph_d.radiusesPtr() + _motionGraph_d.count(),
                 _motionGraphCpu.radiuses().data() + _motionGraph_d.count(), cCountNewNodes);
    _motionGraph_d.setCount(cUpdatedGraphSize);
}


template<int K, template<class> class Container>
void WarpField::computeKnns(af::DeviceKnn<K>& knnResult_d, const Container<Vec3f>& pointCloud_d) {
    std::vector<Vecui<K>> knnIdxs;
    std::vector<Vecf<K>> knnDists;

    std::vector<Vec3f> pointCloudCpu(pointCloud_d.size());
    thrust::copy_n(pointCloud_d.begin(), pointCloud_d.size(), pointCloudCpu.begin());

    af::getKnnData(knnIdxs, knnDists, _motionGraphCpu, pointCloudCpu);
    copyToDevice(knnResult_d.idxsPtr(), knnIdxs.data(), knnIdxs.size());
    copyToDevice(knnResult_d.distsPtr(), knnDists.data(), knnDists.size());
    knnResult_d.setSize(knnIdxs.size());
}

}  // namespace af

#endif