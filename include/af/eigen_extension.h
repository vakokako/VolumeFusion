#ifndef EIGEN_EXTENSION_H
#define EIGEN_EXTENSION_H

#ifndef WIN64
#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>
#include <boost/functional/hash.hpp>

typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f; 
typedef Eigen::Matrix3f Mat3f;
typedef Eigen::Matrix4f Mat4f;

typedef Eigen::Vector2d Vec2d;
typedef Eigen::Vector3d Vec3d;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::Matrix4d Mat4d;

typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;

typedef Eigen::Matrix<float, 6, 1> Vec6f;
typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 3, 6> Mat3_6f;

template<int Length>
using Vecf = Eigen::Matrix<float, Length, 1>;
template<int Length>
using Veci = Eigen::Matrix<int, Length, 1>;
template<int Length>
using Vecsizet = Eigen::Matrix<std::size_t, Length, 1>;
template<int Length>
using Vecui = Eigen::Matrix<unsigned int, Length, 1>;
template<typename Type>
using Vec3 = Eigen::Matrix<Type, 3, 1>;
template<typename Type, int Length>
using Vec = Eigen::Matrix<Type, Length, 1>;


using Arr3i = Eigen::Array3i;

template <typename T>
struct is_eigen_vec_impl : std::false_type {};

template <typename T, int Rows>
struct is_eigen_vec_impl<Eigen::Matrix<T, Rows, 1>> : std::true_type {};

template <typename T>
constexpr bool is_eigen_vec = is_eigen_vec_impl<std::decay_t<T>>::value;

template<class T, int... Is>
auto to_vector(const Eigen::Matrix<T, Is...>& mat) {
    return std::vector<T>(mat.data(), mat.data() + mat.size());
}


// template< class T >
// struct is_eigen_vector
//      : std::integral_constant<
//          bool,
//          std::is_same<float, typename std::remove_cv<T>::type>::value  ||
//          std::is_same<double, typename std::remove_cv<T>::type>::value  ||
//          std::is_same<long double, typename std::remove_cv<T>::type>::value
//      > {};

// namespace boost {
// namespace serialization {

// template<class Archive, class Type, int Rows, int Cols>
// void serialize(Archive& ar, Eigen::Matrix<Type, Rows, Cols>& mat, const unsigned int version) {
//     for (int i = 0; i < Cols; i++) {
//         for (int j = 0; j < Rows; j++) {
//             ar& mat(j, i);
//         }
//     }
// }

// }  // namespace serialization
// }  // namespace boost

namespace std {

// bool operator<(const Vec3f& a, const Vec3f& b) {
//     if (a[0] < b[0])
//         return true;
//     if (b[0] < a[0])
//         return false;
//     if (a[1] < b[1])
//         return true;
//     if (b[1] < a[1])
//         return false;
//     return a[2] < b[2];
// }

template<>
struct hash<Vec3f> {
    size_t operator()(const Vec3f& point) const { return boost::hash_range(point.data(), point.data() + 3); }
};

}  // namespace std

#endif
