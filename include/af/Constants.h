#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

class Constants {
public:
    static const std::string appFolder;
    static const std::string dataFolder;

    static const float depthThreshold;

    static constexpr std::size_t motionGraphKNN = 6;
    static const float motionGraphRadius;

    static const float tsdfDelta;

    static const std::size_t energyMRegKNN = 4;

    static constexpr int distortionCoeffsCount = 8;
};

#endif