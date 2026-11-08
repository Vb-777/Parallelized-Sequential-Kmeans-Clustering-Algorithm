// Two clocks, because the difference between them is the whole story of
// this project's original measurements.
//
// steady_clock is elapsed time: what a user waits. clock() - what the
// original used - is processor time summed over every thread that ran. On a
// program that parallelises perfectly, elapsed time falls by a factor of p
// while CPU time stays flat, so a speedup measured with clock() is
// approximately 1.00 no matter how many threads are used. That is not a
// small error in the recorded numbers; it means the committed program could
// not have produced the committed speedup table.
//
// Reporting both costs nothing and makes the ratio - the average number of
// cores kept busy - a number you can read straight off any run.

#ifndef KMEANS_TIMER_HPP
#define KMEANS_TIMER_HPP

#include <chrono>
#include <ctime>

namespace kmeans {

class Timer {
public:
    Timer() { reset(); }

    void reset() {
        wall_start_ = std::chrono::steady_clock::now();
        cpu_start_ = std::clock();
    }

    [[nodiscard]] double wall_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - wall_start_).count();
    }

    [[nodiscard]] double cpu_seconds() const {
        return static_cast<double>(std::clock() - cpu_start_) / CLOCKS_PER_SEC;
    }

private:
    std::chrono::steady_clock::time_point wall_start_;
    std::clock_t cpu_start_ = 0;
};

}  // namespace kmeans

#endif  // KMEANS_TIMER_HPP
