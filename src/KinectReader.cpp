#include "af/KinectReader.h"

#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include "af/dataset.h"

namespace af {

void KinectDepthReader::readNext(cv::Mat& depth) {
    if (_folder.empty()) {
        return;
    }
    _currIndex++;

    std::string depthFile = af::insertIntoPattern(_folder, _currIndex);

    _depthBuffer = cv::imread(depthFile, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (_depthBuffer.empty())
        throw(std::runtime_error("loadDepthFrame(): Depth couldn't be loadded"));
    _depthBuffer.convertTo(depth, CV_32FC1, (1.0 / 1000.0));
}

std::string KinectDepthReader::filename(const std::string& folderName, int index) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << index;
    std::string frameName = "frame-" + ss.str();
    return folderName + "/" + frameName + ".depth.png";
}

}  // namespace af