#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <opencv2/imgproc.hpp>

#include "af/CameraModel.h"

namespace af {

bool create_xy_lookup_table(const af::CameraModel& camera, cv::Mat& xy_table);

}

#endif