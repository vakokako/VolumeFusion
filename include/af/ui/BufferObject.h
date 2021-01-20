#ifndef BUFFEROBJECT_H
#define BUFFEROBJECT_H

#include "af/ui/BufferSignaling.h"

namespace af {

template<typename T>
class BufferObject : public SignalingStreamBuffer<T> {
public:
    void load(const T* data, std::size_t size) final { load(true, data, size); }
    void loadFromDevice(const T* data, std::size_t size) final { load(false, data, size); }

    T& object() { return _object; }
    const T& object() const { return _object; }

private:
    void load(bool isHost, const T* data, std::size_t size);

    T _object;
};

}  // namespace af

#endif