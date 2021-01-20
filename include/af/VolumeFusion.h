#ifndef VOLUMEFUSION_H
#define VOLUMEFUSION_H

#include "af/Settings.h"
#include "af/Mesh.h"
#include "af/Constants.h"
#include "af/eigen_extension.h"
#include "af/StreamBuffer.h"
#include "af/Stepper.h"
#include "af/MotionGraph.h"

#include <map>
#include <atomic>
#include <mutex>

namespace af {

struct VolumeFusionOutput {
    MotionGraph& motionGraph;
    af::StreamBufferAbstract<Vec3f>& pointsCanon;
    af::StreamBufferAbstract<Vec3f>& pointsWarped;
    af::StreamBufferAbstract<Vec3f>& pointsGraph;
    af::StreamBufferAbstract<Vec3f>& pointsFrame;
    af::StreamBufferAbstract<Vec3f>& meshPoints;
    af::StreamBufferAbstract<Vec3i>& meshFaces;
    af::StreamBufferAbstract<Vec2i>& correspondeceCanonToFrame;
    af::StreamBufferAbstract<Vecui<Constants::energyMRegKNN>>& lines;
    af::StreamBufferAbstract<Vec3f>& centroid;
    af::Stepper& stepper;
    std::atomic_flag& keepRunning;
    std::mutex& dataLoadMutex;
    Mesh& meshResult;
};

void runVolumeFusion(VolumeFusionOutput& output, const af::Settings& settings);

}  // namespace af

#endif