#ifndef KINECTREADER_H
#define KINECTREADER_H

#include <opencv2/core.hpp>
#include <string>

namespace af {

class KinectDepthReader {
public:
    KinectDepthReader(std::string folder = "") : _folder(std::move(folder)) {}

    void readNext(cv::Mat& depth);

    std::string folder() const { return _folder; }
    int currentIndex() const { return _currIndex; }
    void setFolder(std::string folder) {
        _folder    = std::move(folder);
        _currIndex = -1;
    }
    void setCurrentIndex(int index) { _currIndex = index; }

private:
    std::string filename(const std::string& folderName, int index);

private:
    std::string _folder;
    int _currIndex = -1;
    cv::Mat _depthBuffer;
};

}  // namespace af

#endif