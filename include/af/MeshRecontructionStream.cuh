#ifndef MESHRECONTRUCTIONSTREAM_CUH
#define MESHRECONTRUCTIONSTREAM_CUH

#include "af/eigen_extension.h"

#include "af/StreamBuffer.h"
#include "af/Mesh.h"
#include "af/Stepper.h"

#include <map>
#include <mutex>


namespace af {

class Settings;

struct MeshReconstructionBuffers {
    StreamBufferAbstract<Vec3f>& _pointsFrame;
    StreamBufferAbstract<Vec3f>& _pointsMesh;
    StreamBufferAbstract<Vec3i>& _faces;
    af::Stepper& _stepper;
    std::atomic_flag& _keepRunning;
    std::mutex& _dataLoadMutex;
    Mesh& _meshResult;
};

void startMeshReconstructionStream(MeshReconstructionBuffers& buffers, const af::Settings& settings);

}

#endif