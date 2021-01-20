#ifndef TIMEMAP_H
#define TIMEMAP_H

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <numeric>

class TimeMap {
public:
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    explicit TimeMap() { clearTimer(); }

    void clearTimer() {
        _lastTimePoint = std::chrono::high_resolution_clock::now();
    }

    void addMarker(const std::string& label) {
        auto currTimePoint = std::chrono::high_resolution_clock::now();
        int64_t duration      = std::chrono::duration_cast<std::chrono::microseconds>(currTimePoint - _lastTimePoint).count();
        _markers[label].push_back(duration);
        clearTimer();
    }

    const std::map<std::string, std::vector<int64_t>>& markers() const { return _markers; }

    friend std::ostream& operator<<(std::ostream& out, TimeMap& timeMap) {
        std::map<double, std::string> means;
        for (auto&& marker : timeMap.markers()) {
            double mean = static_cast<double>(std::accumulate(marker.second.begin(), marker.second.end(), 0)) / marker.second.size();
            means[mean] = marker.first;
        }
        for (auto&& mean : means) {
            out << mean.second << " : " << mean.first * 0.001 << "ms\n";
        }
        return out;
    }

private:
    time_point _lastTimePoint;
    std::map<std::string, std::vector<int64_t>> _markers;
};

#endif