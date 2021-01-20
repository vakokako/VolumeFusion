#ifndef DEVICEBUFFER_CUH
#define DEVICEBUFFER_CUH

#include <thrust/device_vector.h>

namespace af {

// template<typename Type>
// class DeviceBuffer {
// public:
//     thrust::device_vector<Type> _buffer;
//     std::size_t _count;
//     thrust::device_vector<std::size_t> _countDevice;
//     DeviceBuffer(std::size_t size, std::size_t count = 0) : _buffer(size), _count(count), _countDevice(1) {
//         _countDevice[0] = _count;
//     }
//     DeviceBuffer(std::size_t size, const Type& value, std::size_t count = 0) : _buffer(size, value), _count(count),
//     _countDevice(1) {
//         _countDevice[0] = _count;
//     }
// };

template<typename Type>
struct Device_Buff_View {
    Type* _buffer;
    unsigned int* _count;
    unsigned int _size;

    __device__ void push_back(const Type& value) {
        unsigned int lastCount = atomicAdd(_count, 1);
        _buffer[lastCount]     = value;
    }
};

template<typename Type>
class DeviceBuffer {
public:
    using iterator       = typename thrust::device_vector<Type>::iterator;
    using const_iterator = typename thrust::device_vector<Type>::const_iterator;
    using pointer        = typename thrust::device_vector<Type>::pointer;
    using const_pointer  = typename thrust::device_vector<Type>::const_pointer;
    using size_type      = typename thrust::device_vector<Type>::size_type;
    using value_type     = typename thrust::device_vector<Type>::value_type;

    DeviceBuffer(unsigned int bufferSize, unsigned int size = 0) : _buffer(bufferSize), _size(size) {}

    iterator begin(void) { return _buffer.begin(); }
    const_iterator begin(void) const { return _buffer.begin(); }

    iterator end(void) { return _buffer.begin() + _size; }
    const_iterator end(void) const { return _buffer.begin() + _size; }

    size_type bufferSize(void) const { return _buffer.size(); }
    void resizeBuffer(size_type new_size, const value_type& x = value_type()) { _buffer.resize(new_size, x); }

    Type* bufferPtr() { return _buffer.data().get(); }
    const Type* bufferPtr() const { return _buffer.data().get(); }

    pointer data() { return _buffer.data(); }
    const_pointer data() const { return _buffer.data(); }

    virtual void setSize(unsigned int size) {
        if (size > _buffer.size()) {
            throw std::runtime_error("af::DeviceBuffer::setSize(): size > bufferSize (" + std::to_string(size) + " > " + std::to_string(_buffer.size()) + ").");
        }
        _size = size;
    }
    void resetSize() { setSize(0); }

    unsigned int size(void) const { return _size; }

private:
    thrust::device_vector<Type> _buffer;
    unsigned int _size;
};

template<typename Type>
class DeviceBufferCounted : public DeviceBuffer<Type> {
public:
    DeviceBufferCounted(unsigned int bufferSize, unsigned int size = 0)
        : DeviceBuffer<Type>(bufferSize, size), _size_d(1, size) {}

    void syncHostSize() { DeviceBuffer<Type>::setSize(_size_d[0]); }
    void setSize(unsigned int size) override {
        DeviceBuffer<Type>::setSize(size);
        _size_d[0] = size;
    }

    unsigned int* size_dPtr() { return _size_d.data().get(); }
    const unsigned int* size_dPtr() const { return _size_d.data().get(); }

private:
    thrust::device_vector<unsigned int> _size_d;
};

template<typename Type>
Device_Buff_View<Type> createView(DeviceBufferCounted<Type>& deviceBuffer) {
    return {deviceBuffer.bufferPtr(), deviceBuffer.size_dPtr(), deviceBuffer.bufferSize()};
}

template<typename Type>
Device_Buff_View<Type> createView(const DeviceBufferCounted<Type>& deviceBuffer) {
    return {deviceBuffer.bufferPtr(), deviceBuffer.size_dPtr(), deviceBuffer.bufferSize()};
}

}  // namespace af

#endif