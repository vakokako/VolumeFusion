// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_GRADIENT_CUH
#define TUM_GRADIENT_CUH

#include "af/eigen_extension.h"

void computeGradientByXCuda(Vec3f *gradientByX, const float *sdfs, Vec3f voxelSize, Vec3i dim);
void computeGradientCuda(Vec6f *gradient, Vec3f *gradientByX, Mat4f projection, Vec3f centroid, Vec3f voxelSize, Vec3f size, Vec3i dim);
void computeStepParametersCuda(
        const float *tsdf_cur,
        const float *wsdf_cur,
        const float *tsdf_ref,
        const float *wsdf_ref,
        Mat4f projection_current,
        Vec6f *gradient,
        Vec3f centroid,
        Vec3f voxelSize,
        Vec3f size,
        Vec3i dim,
        Vec3i completeDim,
        int offsetZ,
        Vec6f ksi_cur,
        // results:
        float *A,
        float *b,
        float *error_function
);
void initCuda(float *input_d, int n);
void constructSDFCuda(float* sdfs_ref, float* wsdfs_ref, float* sdfs_cur, float* wsdfs_cur, Mat4f& projection, Vec3f& centroid, Vec3f& voxelSize, Vec3f& size, Vec3i& dim);

__device__
float interpolate3voxel(float x, float y, float z, const float* m_tsdf, const Vec3i& m_dim);

__device__
Vec3f voxelToCamera(const Vec3i &pt, const Vec3f &centroid, const Vec3f &voxelSize, const Vec3f &size);

__device__
Vec3i cameraToVoxel(const Vec3f &pt, const Vec3f &centroid, const Vec3f &voxelSize, const Vec3f &size);

__device__
Vec3f transform(Vec3f point, Mat4f projection);



__device__
void normalize(Vec3f& point);

__device__
void skewSymmetric(const Vec3f &vec, Mat3f &matX);

#endif
