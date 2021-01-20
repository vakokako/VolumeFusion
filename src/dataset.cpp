#include <af/dataset.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <stdexcept>
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

void loadIntrinsics(const std::string& intrinsicsFile, Eigen::Matrix3f& K) {
    if (intrinsicsFile.empty())
        throw(std::runtime_error("loadIntrinsics(): Intrinsics file name is empty."));

    std::ifstream intrIn(intrinsicsFile.c_str());
    if (!intrIn.is_open())
        throw(std::runtime_error("loadIntrinsics(): Intrinsics file " + intrinsicsFile + " couldn't be open."));

    // camera intrinsics
    K          = Eigen::Matrix3f::Identity();
    float fVal = 0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            intrIn >> fVal;
            K(i, j) = fVal;
        }
        // Drop fourth 0 element;
        // intrIn >> fVal;
    }
    intrIn.close();
}

bool loadCameraDistortion(const std::string& filename, af::CameraDistortion& distCoeffs) {
    distCoeffs.is_distorted = false;

    if (filename.empty()) {
        std::cout << "loadCameraDistortion(): Distortion file name is empty.\n";
        return false;
    }

    std::ifstream distortionFile(filename.c_str());
    if (!distortionFile.is_open()) {
        std::cout << "loadCameraDistortion(): Intrinsics file \"" + filename + "\" couldn't be open.\n";
        return false;
    }

    // camera intrinsics
    constexpr int distCoeffsCount = 8;
    std::array<float, distCoeffsCount> distCoeffsArray;
    for (auto& coeff : distCoeffsArray)
        distortionFile >> coeff;
    distortionFile.close();


	distCoeffs.k1 = distCoeffsArray[0];
	distCoeffs.k2 = distCoeffsArray[1];
	distCoeffs.k3 = distCoeffsArray[4];
	distCoeffs.k4 = distCoeffsArray[5];
	distCoeffs.k5 = distCoeffsArray[6];
	distCoeffs.k6 = distCoeffsArray[7];
    distCoeffs.codx = 0.f;
    distCoeffs.cody = 0.f;
	distCoeffs.p1 = distCoeffsArray[2];
	distCoeffs.p2 = distCoeffsArray[3];
    distCoeffs.is_distorted = true;

    return true;
}

bool loadCameraTransform(const std::string& filename, Mat4f& transfMatrix) {
    if (filename.empty()) {
        std::cout << "loadCameraTransform(): Camera transforms file name is empty.\n";
        return false;
    }

    std::ifstream distortionFile(filename.c_str());
    if (!distortionFile.is_open()) {
        std::cout << "loadCameraTransform(): Camera transforms file \"" + filename + "\" couldn't be open.\n";
        return false;
    }

    transfMatrix.setIdentity();

    Eigen::Quaternionf rotationQuat;
    distortionFile >> rotationQuat.w();
    distortionFile >> rotationQuat.x();
    distortionFile >> rotationQuat.y();
    distortionFile >> rotationQuat.z();
    transfMatrix.block<3,3>(0,0) = rotationQuat.normalized().toRotationMatrix();

    distortionFile >> transfMatrix(0, 3);
    distortionFile >> transfMatrix(1, 3);
    distortionFile >> transfMatrix(2, 3);

    distortionFile.close();
    return true;
}

std::string fileName(const std::string& folderName,
                     size_t index,
                     const std::string& suffix,
                     const std::string& extension = ".png") {
    // build postfix
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << index;
    std::string frameName = "frame-" + ss.str();
    return folderName + "/" + frameName + suffix + extension;
}

std::string fileNameAlt(const std::string& folderName,
                        size_t index,
                        const std::string& suffix,
                        const std::string& extension = ".tiff") {
    // build postfix
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << index;
    return folderName + "/" + ss.str() + suffix + extension;
}

std::string insertIntoPattern(const std::string& pattern,
                                size_t index) {
    // build postfix
    std::string result = pattern;

    std::size_t placeholderCount = std::count(result.begin(), result.end(), '*');
    std::string placeholderString(placeholderCount, '*');
    auto placeholderStart = result.find(placeholderString);
    if (placeholderStart == std::string::npos){
        // throw std::runtime_error("file name pattern is not correct");
        std::cout << "file name pattern is not correct\n";
        return std::string();
    }

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(placeholderCount) << index;
    std::string placeholderReplacement = ss.str();
    std::copy(placeholderReplacement.begin(), placeholderReplacement.end(), result.begin() + placeholderStart);
    return result;
}

std::string fileName2(const std::string& folderName,
                      size_t index,
                      const std::string& prefix,
                      const std::string& extension = ".png") {
    // build postfix
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << index;
    std::string frameName = prefix + ss.str();
    return folderName + "/" + frameName + extension;
}

void loadFrame(const std::string& folder, size_t index, cv::Mat& color, cv::Mat& depth) {
    af::loadDepthFrame(folder, index, depth);
    af::loadColorFrame(folder, index, color);
}

void loadDepthFrame(const std::string& folder, size_t index, cv::Mat& depth) {
    bool Kinect = true;
    cv::Mat depthIn;
    if (Kinect) {
        // std::string depthFile = fileName(folder, index, ".depth");
        std::string depthFile = insertIntoPattern(folder, index);
        std::cout << "depth file name : " << depthFile << "\n";

        depthIn = cv::imread(depthFile, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        if (depthIn.empty())
            throw(std::runtime_error("loadDepthFrame(): Depth couldn't be loadded"));

    } else {
        std::string depthFile = fileName2(folder, index, "depth_", ".exr");

        depthIn = cv::imread(depthFile, cv::IMREAD_UNCHANGED);
        if (depthIn.empty())
            throw(std::runtime_error("loadDepthFrame(): Depth couldn't be loadded"));
        cv::cvtColor(depthIn, depthIn, cv::COLOR_RGB2GRAY);
    }
    depthIn.convertTo(depth, CV_32FC1, (1.0 / 1000.0));
}
void loadColorFrame(const std::string& folder, size_t index, cv::Mat& color) {
    std::string colorFile = fileName(folder, index, ".color");

    color = cv::imread(colorFile);
    if (color.empty())
        throw(std::runtime_error("loadColorFrame(): Color couldn't be loadded"));
}

void saveDepth(const std::string& fileName, const cv::Mat& depth) {
    cv::Mat depthOut;
    depth.convertTo(depthOut, CV_16U, (1000.0));
    cv::imwrite(fileName + ".depth.png", depthOut);
}

}  // namespace af