#include <af/VolumeFusion.h>
#include <af/ui/PointsQWrap.h>

#include <gtest/gtest.h>

TEST(VolumeFusionSpeedTest, Run) {
    af::Settings settings;

    std::map<std::string, PointsQWrap> emptyBuffers;
    Mesh resultMesh;
    VectorQWrap<Vecui<Constants::energyMRegKNN>> lines;
    VectorQWrap<Vec3f> centr;

    af::VolumeFusionOutput afResult{emptyBuffers, lines, centr, resultMesh};

    af::runVolumeFusion(afResult, settings);
}