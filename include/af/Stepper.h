#ifndef STEPPER_H
#define STEPPER_H

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

namespace af {
class Stepper {
public:
    void wait_for_step(const std::string& stepName) {
        if (!_enabled) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(_stepsSwitchesMutex);
            if (!_stepsSwitches.count(stepName)) {
                _stepsSwitches[stepName] = true;
            }

            if (!_stepsSwitches[stepName]) {
                return;
            }
        }

        std::unique_lock<std::mutex> lock(_mutex);
        _cond_v.wait(lock, [this] { return _check; });
        _check = false;
    }

    void step() {
        std::lock_guard<std::mutex> guard(_mutex);
        _check = true;
        _cond_v.notify_one();
    }

    bool isEnabled() { return _enabled; }

    void enable(bool enabled = true) {
        std::cout << "stepper enabled : " << enabled << "\n";
        _enabled = enabled;
    }

    void enableStep(const std::string& stepName, bool enable = true) {
        std::lock_guard<std::mutex> guard(_stepsSwitchesMutex);
        if (!_stepsSwitches.count(stepName)) {
            return;
        }

        _stepsSwitches[stepName] = enable;
    }

    std::map<std::string, bool>& stepsSwitches() { return _stepsSwitches; }

private:
    std::mutex _mutex;
    std::condition_variable _cond_v;
    bool _check = false;

    std::atomic<bool> _enabled{true};

    std::mutex _stepsSwitchesMutex;
    std::map<std::string, bool> _stepsSwitches;
};
}  // namespace af

#endif