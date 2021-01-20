#ifndef BUFFERVECTOR_H
#define BUFFERVECTOR_H

#include <vector>

#include "af/ui/BufferSignaling.h"

namespace af {

template<typename T>
class BufferVector : public SignalingStreamBuffer<T> {
public:
    void load(const T* data, std::size_t size) final { load(true, data, size); }
    void loadFromDevice(const T* data, std::size_t size) final { load(false, data, size); }
    std::vector<T>& vector() { return _vector; }
    const std::vector<T>& vector() const { return _vector; }
    std::size_t size() const { return _vector.size(); }

private:
    void load(bool isHost, const T* data, std::size_t size);

    std::vector<T> _vector;
};

}  // namespace af

#endif