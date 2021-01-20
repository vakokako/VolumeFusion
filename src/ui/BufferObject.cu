#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "af/eigen_extension.h"
#include "af/ui/BufferObject.h"

namespace af {

template<typename T>
void BufferObject<T>::load(bool isHost, const T* data, std::size_t size) {
    {
        std::lock_guard<std::mutex> lock(this->mutex());
        if (size != 1) {
            throw std::runtime_error("BufferObject::load(): trying to write more than one object to the buffer.");
        }
        if (isHost) {
            thrust::copy_n(thrust::host, data, 1, &_object);
        } else {
            thrust::copy_n(thrust::device, data, 1, &_object);
        }
    }

    this->modified();
}

template class BufferObject<Vec3f>;
template class BufferObject<Vec3i>;

}  // namespace af