#ifndef TSDFVOLUME_H
#define TSDFVOLUME_H

// #include "boost/serialization/access.hpp"
#include "af/eigen_extension.h"
#include <opencv2/core/core.hpp>
#include <vector>


class TSDFVolume {
public:
    TSDFVolume(const Vec3i& dimensions, const Vec3f& size, const Mat3f& K);
    ~TSDFVolume();

    bool init();

    void release();

    void bbox(Vec3d& min, Vec3d& max) const;

    void setSize(const Vec3f& size);
    Vec3f size() const;
    Vec3f voxelSize() const;

    size_t gridSize() const { return m_gridSize; }

    Vec3i dimensions() const;

    Vec3f voxelToWorld(const Vec3i& voxel) const;
    Vec3f voxelToWorld(int i, int j, int k) const;
    Vec3f voxelToWorld(const Vec3f& voxel) const;

    Vec3i worldToVoxel(const Vec3f& pt) const;
    Vec3f worldToVoxelF(const Vec3f& pt) const;
    Vec3f cameraToObject(const Vec3f& pt) const;
    Vec3i cameraToVoxel2(const Vec3f& pt, Mat3f& R) const;
    Vec3f cameraToObject2(const Vec3f& pt, Mat3f& R) const;



    // MODIFIED
    int correct_voxel_index(int index, unsigned dimension);
    Vec3f voxelToCamera(const Vec3i& voxel) const;
    Vec3i cameraToVoxel(const Vec3f& pt) const;
    // MODIFIED



    void integrate(const Mat4f& pose, const cv::Mat& color, const cv::Mat& vertexMap, const cv::Mat& normals = cv::Mat());

    bool load(const std::string& filename);
    bool save(const std::string& filename);

    float truncate(float sdf) const;

    float interpolate3(float x, float y, float z) const;

    void setDelta(float delta) {
        m_delta = delta;
        m_deltaInv = 1.0f / m_delta;
    }
    float delta() const { return m_delta; }

    std::vector<float>& tsdf() { return m_tsdf; }
    std::vector<float>& tsdfWeights() { return m_weights; }
    std::vector<unsigned char>& colorR() { return m_colorR; }
    std::vector<unsigned char>& colorG() { return m_colorG; }
    std::vector<unsigned char>& colorB() { return m_colorB; }
    const std::vector<float>& tsdf() const { return m_tsdf; }
    const std::vector<float>& tsdfWeights() const { return m_weights; }
    const std::vector<unsigned char>& colorR() const { return m_colorR; }
    const std::vector<unsigned char>& colorG() const { return m_colorG; }
    const std::vector<unsigned char>& colorB() const { return m_colorB; }
    float* ptrTsdf() { return m_tsdf.data(); }
    float* ptrTsdfWeights() { return m_weights.data(); }
    unsigned char* ptrColorR() { return m_colorR.data(); }
    unsigned char* ptrColorG() { return m_colorG.data(); }
    unsigned char* ptrColorB() { return m_colorB.data(); }
    const float* ptrTsdf() const { return m_tsdf.data(); }
    const float* ptrTsdfWeights() const { return m_weights.data(); }
    const unsigned char* ptrColorR() const { return m_colorR.data(); }
    const unsigned char* ptrColorG() const { return m_colorG.data(); }
    const unsigned char* ptrColorB() const { return m_colorB.data(); }

    Vec3f centroid() const { return m_centroid; }

    Vec3f surfaceNormal(int i, int j, int k);
    float interpolate3voxel(float x, float y, float z) const;

protected:
    Vec3i m_dim;
    size_t m_gridSize;
    Vec3f m_size;
    Vec3f m_voxelSize;

    std::vector<float> m_tsdf;
    std::vector<float> m_weights;
    std::vector<unsigned char> m_colorR;
    std::vector<unsigned char> m_colorG;
    std::vector<unsigned char> m_colorB;
    std::vector<float> m_weightsColor;
    Mat3f m_K;

    float m_delta;
    float m_deltaInv;

    Vec3f m_centroid;

private:
    // friend class boost::serialization::access;
    // template<class Archive>
    // void serialize(Archive& ar, const unsigned int version);
};

// template<class Archive>
// void TSDFVolume::serialize(Archive& ar, const unsigned int version) {
//     ar& m_dim;
//     ar& m_gridSize;
//     ar& m_size;
//     ar& m_voxelSize;

//     int totalSize = m_dim[0] * m_dim[1] * m_dim[2];
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_tsdf[i];
//     }
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_weights[i];
//     }
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_colorR[i];
//     }
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_colorG[i];
//     }
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_colorB[i];
//     }
//     for (int i = 0; i < totalSize; i++) {
//         ar& m_weightsColor[i];
//     }

//     ar& m_K;

//     ar& m_delta;
//     ar& m_deltaInv;

//     ar& m_centroid;
// }

#endif
