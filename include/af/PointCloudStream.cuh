#ifndef POINTCLOUDSTREAM_CUH
#define POINTCLOUDSTREAM_CUH

#include "af/eigen_extension.h"

#include "af/StreamBuffer.h"
#include "af/Mesh.h"
#include "af/Stepper.h"

#include <map>
#include <mutex>


namespace af {

class Settings;

struct PointCloudBuffers {
    StreamBufferAbstract<Vec3f>& _pointsFrame;
    af::Stepper& _stepper;
    std::atomic_flag& _keepRunning;
    std::mutex& _dataLoadMutex;
};

void startPointCloudStream(PointCloudBuffers& buffers, const af::Settings& settings);

}


#endif