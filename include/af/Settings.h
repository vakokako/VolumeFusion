#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>
#include <vector>

#include "af/eigen_extension.h"

namespace af {
struct Settings {
    Vec3f tsdfSize;
    Vec3i tsdfDim;
    Vec3f tsdfCenter;
    Vec3i tsdfRotation;
    std::pair<float, float> normExclAngleRange;
    std::pair<int, int> framesRange;
    int frameBreak;
    std::string dataFolder;
    uint32_t cameraCount;
    std::string depthFilesPattern;
    float depthThreshold;
    bool integrateTsdf;
    bool bilateralFiltr;
    bool updateGraph;
    bool useCopyHack;
    bool useCholesky;
    int bilateralD;
    float bilateralSigmaI;
    float bilateralSigmaS;
    float bilateralThreshold;
    float energyWeightDepth;
    float energyWeightMReg;
    float energyMinStep;
    int icpIterations;

    float motionGraphRadius;
    float motionGraphSampleDistance;
    float tsdfDelta;

    float correspThreshDist;

    bool isMeshWarped;

    Settings();
};

std::vector<Settings> loadSettingsPresets(const std::string& filename);
void saveSettingsPresets(const std::string& filename, const std::vector<Settings>& settings);


}  // namespace af

#endif