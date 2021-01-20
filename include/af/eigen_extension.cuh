#ifndef EIGEN_EXTENSION_CUH
#define EIGEN_EXTENSION_CUH

#include <af/eigen_extension.h>

namespace thrust {
__forceinline__ __device__ __host__ bool operator<(const Vec2i& a, const Vec2i& b) {
    if (a[0] < b[0])
        return true;
    if (b[0] < a[0])
        return false;
    return a[1] < b[1];
}

__forceinline__ __device__ __host__ bool operator<(const Vec3f& a, const Vec3f& b) {
    if (a[0] < b[0])
        return true;
    if (b[0] < a[0])
        return false;
    if (a[1] < b[1])
        return true;
    if (b[1] < a[1])
        return false;
    return a[2] < b[2];
}

}  // namespace thrust

template<unsigned int Dim>
struct compareAtDim {
    __host__ __device__ bool operator()(const Vec2i& v1, const Vec2i& v2) { return v1[Dim] < v2[Dim]; }
};
template<unsigned int Dim>
struct equalAtDim {
    __host__ __device__ bool operator()(const Vec2i& v1, const Vec2i& v2) { return v1[Dim] == v2[Dim]; }
};

struct isZero {
    __host__ __device__ bool operator()(const Vec3f& v) { return v.isZero(); }
};
struct isNotZero {
    __host__ __device__ bool operator()(const Vec3f& v) { return !v.isZero(); }
};
struct hasPositiveDepth {
    __host__ __device__ bool operator()(const Vec3f& v) { return v[2] > 1.e-4f; }
};

#endif