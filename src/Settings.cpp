#include <af/Settings.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace af {
Settings::Settings() {
    framesRange        = {1, 1000};
    frameBreak         = -1;
    tsdfSize           = Vec3f(1.0f, 1.0f, 1.0f);
    tsdfDim            = Vec3i(400, 400, 400);
    tsdfCenter         = Vec3f(0.0f, 0.f, 0.f);
    tsdfRotation       = Vec3i(45, -45, 0);
    normExclAngleRange = {1.3, 2.1};
    cameraCount        = 1;
    depthThreshold     = 1.0f;
    integrateTsdf      = true;
    bilateralFiltr     = true;
    updateGraph        = true;
    useCopyHack        = true;
    useCholesky        = true;
    bilateralD         = 5;
    bilateralSigmaI    = 75;
    bilateralSigmaS    = 75;
    bilateralThreshold = 0.025f;
    energyWeightDepth  = 1.f;
    energyWeightMReg   = 2.2;
    energyMinStep      = 0.0001f;
    icpIterations      = 3;

    motionGraphRadius         = 0.05f;
    motionGraphSampleDistance = 0.03f;
    tsdfDelta                 = 0.01f;

    correspThreshDist = 0.01f;

    isMeshWarped = false;
}

std::vector<Settings> loadSettingsPresets(const std::string& filename) {
    if (filename.empty()) {
        std::cout << "loadSettingsPresets(): settings filename is empty.\n";
        return {};
    }

    std::ifstream settingsFile(filename.c_str());
    if (!settingsFile.is_open()) {
        std::cout << "loadSettingsPresets(): settings file \"" + filename + "\" couldn't be open.\n";
        return {};
    }

    nlohmann::json settingsFileJson;
    settingsFile >> settingsFileJson;

    std::vector<Settings> settingsPresets(settingsFileJson.size());
    for (int i = 0; i < settingsFileJson.size(); ++i) {
        auto setIfExists = [&](nlohmann::json& json, auto& variable, const std::string& name) {
            using var_type = std::decay_t<decltype(variable)>;
            if (auto it = json.find(name); it != json.end()) {
                if constexpr (is_eigen_vec<var_type>) {
                    using vec_type = std::vector<typename var_type::Scalar>;
                    variable       = var_type((*it).get<vec_type>().data());
                } else {
                    variable = *it;
                }
            }
        };

        auto& settingsJson = settingsFileJson[i];
        setIfExists(settingsJson, settingsPresets[i].dataFolder, "dataFolder");
        setIfExists(settingsJson, settingsPresets[i].cameraCount, "cameraCount");
        setIfExists(settingsJson, settingsPresets[i].depthFilesPattern, "depthFilesPattern");
        setIfExists(settingsJson, settingsPresets[i].framesRange.first, "framesRangeStart");
        setIfExists(settingsJson, settingsPresets[i].framesRange.second, "framesRangeEnd");
        setIfExists(settingsJson, settingsPresets[i].frameBreak, "frameBreak");
        setIfExists(settingsJson, settingsPresets[i].tsdfSize, "tsdfSize");
        setIfExists(settingsJson, settingsPresets[i].tsdfDim, "tsdfDim");
        setIfExists(settingsJson, settingsPresets[i].tsdfCenter, "tsdfCenter");
        setIfExists(settingsJson, settingsPresets[i].tsdfRotation, "tsdfRotation");
        setIfExists(settingsJson, settingsPresets[i].normExclAngleRange.first, "normExclAngleFrom");
        setIfExists(settingsJson, settingsPresets[i].normExclAngleRange.second, "normExclAngleTo");
        setIfExists(settingsJson, settingsPresets[i].depthThreshold, "depthThreshold");

        setIfExists(settingsJson, settingsPresets[i].bilateralD, "bilateralD");
        setIfExists(settingsJson, settingsPresets[i].bilateralSigmaI, "bilateralSigmaI");
        setIfExists(settingsJson, settingsPresets[i].bilateralSigmaS, "bilateralSigmaS");
        setIfExists(settingsJson, settingsPresets[i].bilateralThreshold, "bilateralThreshold");
        setIfExists(settingsJson, settingsPresets[i].energyWeightDepth, "energyWeightDepth");
        setIfExists(settingsJson, settingsPresets[i].energyWeightMReg, "energyWeightMReg");
        setIfExists(settingsJson, settingsPresets[i].energyMinStep, "energyMinStep");
        setIfExists(settingsJson, settingsPresets[i].icpIterations, "icpIterations");

        setIfExists(settingsJson, settingsPresets[i].motionGraphRadius, "motionGraphRadius");
        setIfExists(settingsJson, settingsPresets[i].motionGraphSampleDistance, "motionGraphSampleDistance");
        setIfExists(settingsJson, settingsPresets[i].tsdfDelta, "tsdfDelta");
        setIfExists(settingsJson, settingsPresets[i].correspThreshDist, "correspThreshDist");
    }

    return settingsPresets;
}

void saveSettingsPresets(const std::string& filename, const std::vector<Settings>& settings) {
    if (filename.empty()) {
        std::cout << "saveSettingsPresets(): settings filename is empty.\n";
        return;
    }

    std::ofstream settingsFile(filename.c_str());
    if (!settingsFile.is_open()) {
        std::cout << "saveSettingsPresets(): settings file \"" + filename + "\" couldn't be open.\n";
        return;
    }

    nlohmann::json settingsFileJson;
    for (auto& preset : settings) {
        nlohmann::json settingsJson;
        settingsJson["dataFolder"]        = preset.dataFolder;
        settingsJson["cameraCount"]       = preset.cameraCount;
        settingsJson["depthFilesPattern"] = preset.depthFilesPattern;
        settingsJson["framesRangeStart"]  = preset.framesRange.first;
        settingsJson["framesRangeEnd"]    = preset.framesRange.second;
        settingsJson["frameBreak"]        = preset.frameBreak;
        settingsJson["tsdfSize"]          = to_vector(preset.tsdfSize);
        settingsJson["tsdfDim"]           = to_vector(preset.tsdfDim);
        settingsJson["tsdfCenter"]        = to_vector(preset.tsdfCenter);
        settingsJson["tsdfRotation"]      = to_vector(preset.tsdfRotation);
        settingsJson["normExclAngleFrom"] = preset.normExclAngleRange.first;
        settingsJson["normExclAngleTo"]   = preset.normExclAngleRange.second;
        settingsJson["depthThreshold"]    = preset.depthThreshold;

        settingsJson["bilateralD"]         = preset.bilateralD;
        settingsJson["bilateralSigmaI"]    = preset.bilateralSigmaI;
        settingsJson["bilateralSigmaS"]    = preset.bilateralSigmaS;
        settingsJson["bilateralThreshold"] = preset.bilateralThreshold;
        settingsJson["energyWeightDepth"]  = preset.energyWeightDepth;
        settingsJson["energyWeightMReg"]   = preset.energyWeightMReg;
        settingsJson["energyMinStep"]      = preset.energyMinStep;
        settingsJson["icpIterations"]      = preset.icpIterations;

        settingsJson["motionGraphRadius"]         = preset.motionGraphRadius;
        settingsJson["motionGraphSampleDistance"] = preset.motionGraphSampleDistance;
        settingsJson["tsdfDelta"]                 = preset.tsdfDelta;
        settingsJson["correspThreshDist"]         = preset.correspThreshDist;

        settingsFileJson.push_back(settingsJson);
    }

    settingsFile << settingsFileJson.dump(4);
}

}  // namespace af