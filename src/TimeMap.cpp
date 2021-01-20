#include "af/TimeMap.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>

class TimeMap {
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    explicit TimeMap() { clearTimer(); }

    void clearTimer() {
        _lastTimePoint = std::chrono::high_resolution_clock::now();
    }

    void addMarker(const std::string& label) {
        auto currTimePoint = std::chrono::high_resolution_clock::now();
        auto duration      = std::chrono::duration_cast<std::chrono::microseconds>(currTimePoint - _lastTimePoint);
        _markers[label].push_back(duration);
        _lastTimePoint = currTimePoint;
    }

private:
    time_point _lastTimePoint;
    std::map<std::string, std::vector<std::chrono::microseconds>> _markers;
};