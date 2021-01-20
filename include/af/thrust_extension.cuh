#ifndef THRUST_EXTENSION_CUH
#define THRUST_EXTENSION_CUH

#include <thrust/device_vector.h>

namespace thrust {

template<class Type>
void memset(thrust::device_vector<Type>& data_d, int value, std::size_t count) {
    cudaMemset(data_d.data().get(), value, sizeof(Type) * count);
}

}

#endif