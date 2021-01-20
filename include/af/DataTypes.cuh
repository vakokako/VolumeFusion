#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <af/eigen_extension.h>
#include <thrust/device_vector.h>

namespace af {

template<int K>
class DeviceKnn {
public:
    template<typename Type>
    using storage = thrust::device_vector<Type>;

    explicit DeviceKnn(std::size_t bufferSize)
        : _bufferSize(bufferSize), _idxs_d(bufferSize), _dists_d(bufferSize) {}

    Vecui<K>* idxsPtr() { return _idxs_d.data().get(); }
    Vecf<K>* distsPtr() { return _dists_d.data().get(); }
    const Vecui<K>* idxsPtr() const { return _idxs_d.data().get(); }
    const Vecf<K>* distsPtr() const { return _dists_d.data().get(); }

    void setSize(std::size_t size) {
        if (size > _bufferSize)
            throw std::runtime_error("DeviceKnnResult::setCount(): count > bufferSize");
        _size = size;
    }
    std::size_t size() const { return _size; }

    virtual void resizeBuffer(std::size_t bufferSize) {
        _bufferSize = bufferSize;
        _idxs_d.resize(_bufferSize);
        _dists_d.resize(_bufferSize);
    }
    std::size_t bufferSize() const { return _bufferSize; }

private:
    std::size_t _bufferSize;
    std::size_t _size;

    storage<Vecui<K>> _idxs_d;    // 402653184
    storage<Vecf<K>> _dists_d;    // 402653184
};

template<int K>
class DeviceKnnWeighted : public DeviceKnn<K> {
public:
    using base_type = DeviceKnn<K>;

    explicit DeviceKnnWeighted(std::size_t bufferSize) : base_type(bufferSize), _weights_d(bufferSize) {}

    void resizeBuffer(std::size_t bufferSize) override {
        base_type::resizeBuffer(bufferSize);
        _weights_d.resize(bufferSize);
    }

    Vecf<K>* weightsPtr() { return _weights_d.data().get(); }
    const Vecf<K>* weightsPtr() const { return _weights_d.data().get(); }

private:
    typename base_type::storage<Vecf<K>> _weights_d;  // 402653184
};

#if 0

// struct PointCloud {
//     thrust::device_vector<Vec3f> _points;   // 3686400
//     thrust::device_vector<Vec3f> _normals;  // 3686400
// };

// struct TSDF {
//     thrust::device_vector<float> _tsdf_d;     // 67108864
//     thrust::device_vector<float> _weights_d;  // 67108864
//     TSDF(std::size_t size) : _tsdf_d(size, -1.f), _weights_d(size, 0.f) {}
// };

#endif

}  // namespace af

#endif