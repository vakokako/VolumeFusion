#ifndef STREAMBUFFER_H
#define STREAMBUFFER_H

#include <mutex>

namespace af {

template<typename T>
class StreamBufferAbstract {
public:
    virtual void load(const T* data, std::size_t size)           = 0;
    virtual void loadFromDevice(const T* data, std::size_t size) = 0;

    std::mutex& mutex() { return _mutex; }

private:
    std::mutex _mutex;
};

}  // namespace af

#endif