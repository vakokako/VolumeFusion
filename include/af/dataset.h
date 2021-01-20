#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#ifndef WIN64
#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "af/CameraModel.h"

namespace af {

std::string insertIntoPattern(const std::string& folderName, size_t index);

void loadIntrinsics(const std::string& intrinsicsFile, Eigen::Matrix3f& K);
bool loadCameraDistortion(const std::string& filename, af::CameraDistortion& distCoeffs);
bool loadCameraTransform(const std::string& filename, Mat4f& transfMatrix);

void loadDepthFrame(const std::string& folder, size_t index, cv::Mat& depth);
void loadColorFrame(const std::string& folder, size_t index, cv::Mat& color);
void loadFrame(const std::string& folder, size_t index, cv::Mat& color, cv::Mat& depth);

void saveDepth(const std::string& fileName, const cv::Mat& depth);

}  // namespace af

#endif
