#ifndef VECTORQWRAP_H
#define VECTORQWRAP_H

#include <af/ui/ModifySignaler.h>

#include <vector>

class ModifySignaler;

template<typename ValueType>
class VectorQWrap {
public:
    explicit VectorQWrap() {}
    VectorQWrap(const VectorQWrap& vec) : _vector(vec._vector), _signaler(vec._signaler) {}

    void resize(unsigned int newSize) { _vector.resize(newSize); }
    unsigned int size() { return _vector.size(); }
    void* data() { return _vector.data(); }
    std::vector<ValueType>& vector() { return _vector; }
    void modified() {
        _signaler.modified();
    }
    ModifySignaler& signaler() { return _signaler; }

private:
    std::vector<ValueType> _vector;
    ModifySignaler _signaler;
};

#endif