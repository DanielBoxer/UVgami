#pragma once

#include <chrono>
#include <string>

namespace uvgami {

class ChronoTimer {
  public:
    ChronoTimer() : ChronoTimer("") {}
    ChronoTimer(std::string msg) : message(msg) { start(); }
    void start(void) { startTime = std::chrono::high_resolution_clock::now(); }
    void finish(void) {
        finishTime = std::chrono::high_resolution_clock::now();
        printf("%s : %f ms\n", message.c_str(), duration());
    }

    float duration(void) {
        std::chrono::duration<double> elapsed = finishTime - startTime;

        return elapsed.count() * 1000;
    }

  private:
    std::string message;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point finishTime;
};

} // namespace uvgami
