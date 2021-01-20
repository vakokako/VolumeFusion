#include "af/AzureReader.h"

#include <iomanip>
#include <opencv2/imgcodecs.hpp>

#include "af/dataset.h"

namespace af {

void AzureDepthReader::readNext(cv::Mat& depth) {
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

std::string AzureDepthReader::filename(const std::string& folderName, int index) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << index;
    return folderName + "/" + ss.str() + "_depth.tiff";
}

}  // namespace af