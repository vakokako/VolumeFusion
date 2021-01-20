#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "af/ui/BufferVtkPoints.h"

namespace af {

void BufferVtkPoints::load(bool isHost, const Vec3f* data, std::size_t size) {
    {
        std::lock_guard<std::mutex> lock(this->mutex());

        _points->Resize(size);
        _points->SetNumberOfPoints(size);

        if (isHost) {
            thrust::copy_n(data, size, static_cast<Vec3f*>(_points->GetVoidPointer(0)));
        } else {
            thrust::device_ptr<const Vec3f> ptr_d(data);
            thrust::copy_n(ptr_d, size, static_cast<Vec3f*>(_points->GetVoidPointer(0)));
        }
    }

    if (!_silent) {
        this->modified();
    }
}
}  // namespace af