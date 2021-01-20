#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "af/ui/BufferVector.h"
#include "af/eigen_extension.h"

namespace af {

template<typename T>
void BufferVector<T>::load(bool isHost, const T* data, std::size_t size) {
    {
        std::lock_guard<std::mutex> lock(this->mutex());
        _vector.resize(size);

        if (isHost) {
            thrust::copy_n(data, size, _vector.data());
        } else {
            thrust::device_ptr<const T> ptr_d(data);
            thrust::copy_n(ptr_d, size, _vector.data());
        }
    }

    this->modified();
}

template class BufferVector<int>;
template class BufferVector<float>;
template class BufferVector<Vec3f>;
template class BufferVector<Vec3i>;
template class BufferVector<Vec2i>;
template class BufferVector<Vecui<4>>;

}  // namespace af