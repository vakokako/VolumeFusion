#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <af/eigen_extension.h>

class CameraModel {
public:
    static Mat3f KDepth;
    static Mat3f KColor;
};

namespace af {

struct CameraIntrinsics {
    float cx{0.f};
    float cy{0.f};
    float fx{0.f};
    float fy{0.f};
};

struct CameraDistortion {
    float k1{0.f};
    float k2{0.f};
    float k3{0.f};
    float k4{0.f};
    float k5{0.f};
    float k6{0.f};
    float codx{0.f};
    float cody{0.f};
    float p1{0.f};
    float p2{0.f};
    bool is_distorted{false};
};

struct CameraModel {
    unsigned int width{0};
    unsigned int height{0};
    CameraIntrinsics intrinsics;
    CameraDistortion distortion;
    Mat3f depthIntrinsics = Mat3f::Identity();
    Mat4f transfW2C = Mat4f::Identity();
};

}  // namespace af

#endif