#ifndef BUFFERSIGNALING_H
#define BUFFERSIGNALING_H

#include "af/ui/ModifySignaler.h"
#include "af/StreamBuffer.h"

namespace af {

template<typename T>
class SignalingStreamBuffer : public StreamBufferAbstract<T> {
public:
    virtual void modified() { _signaler.modified(); }
    ModifySignaler& signaler() { return _signaler; }

private:
    ModifySignaler _signaler;
};

}

#endif