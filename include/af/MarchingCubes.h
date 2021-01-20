#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <vector>

#include "af/eigen_extension.h"
#include <af/Mesh.h>
#include <af/TSDFVolume.h>

class MarchingCubes
{
public:
    MarchingCubes(const Vec3i &dimensions, const Vec3f &size);

    ~MarchingCubes();

    bool computeIsoSurface(Mesh& outputMesh, const TSDFVolume& tsdfVolume);
    bool computeIsoSurface(Mesh& outputMesh, const std::vector<float>& tsdf, const std::vector<float>& weights, const std::vector<unsigned char>& red, const std::vector<unsigned char>& green, const std::vector<unsigned char>& blue, float isoValue = 0.0f);
    bool computePointCloud(Mesh& outputMesh, const TSDFVolume& tsdfVolume);
    bool computePointCloud(Mesh& outputMesh, const std::vector<float>& tsdf, const std::vector<float>& weights, const std::vector<unsigned char>& red, const std::vector<unsigned char>& green, const std::vector<unsigned char>& blue, float isoValue = 0.0f);

protected:

    inline int computeLutIndex(int i, int j, int k, float isoValue);

    Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float isoValue);

    Vec3f getVertex(int i1, int j1, int k1, int i2, int j2, int k2, float isoValue);

    Vec3b getColor(int x1, int y1, int z1, int x2, int y2, int z2, float isoValue);

    void computeTriangles(Mesh& outputMesh, int cubeIndex, const Vec3f edgePoints[12], const Vec3b edgeColors[12]);

    Vec3f voxelToWorld(int i, int j, int k) const;

    Vec3i dim_;
    Vec3f size_;
    Vec3f voxelSize_;

    const std::vector<float>* tsdf_;
    const std::vector<float>* weights_;
    const std::vector<unsigned char>* red_;
    const std::vector<unsigned char>* green_;
    const std::vector<unsigned char>* blue_;
};

#endif
