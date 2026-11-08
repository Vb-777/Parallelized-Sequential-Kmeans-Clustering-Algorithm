// Core types.
//
// Two decisions in this file shape everything else.
//
// Structure of arrays. The original stored `struct Point { double x, y; int
// cluster; }` in a vector, which is 24 bytes per point with the label
// interleaved between the coordinates the distance kernel reads. Separate
// arrays let the kernel load eight contiguous doubles into an AVX-512
// register, and keep the label array - written once per point per iteration
// - out of the hot cache lines entirely.
//
// A fixed block decomposition that does not depend on the thread count. The
// points are split into `blocks` ranges once; a thread claims whole blocks.
// Partial sums are accumulated per block and merged in block order, so the
// order in which floating-point values are added is a property of the
// problem rather than of the schedule. One thread or thirty-two, pthreads or
// OpenMP or MPI, the centroids come out bit for bit identical - which is
// what makes "the parallel version computes the same thing" a test rather
// than a hope.
//
// It also removes a bug by construction. The original computed
// `chunkSize = n / numThreads` and gave thread t the range
// [t*chunkSize, (t+1)*chunkSize), so the last n % numThreads points were
// never assigned to anyone. Here a block's bounds are computed from the
// index, and the last block ends at n.

#ifndef KMEANS_CORE_HPP
#define KMEANS_CORE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace kmeans {

using Scalar = double;
using Label = std::int32_t;

constexpr Scalar kInfinity = std::numeric_limits<Scalar>::infinity();

/// Points, as separate coordinate arrays.
struct PointCloud {
    std::vector<Scalar> x;
    std::vector<Scalar> y;
    std::vector<Label> label;

    [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
    [[nodiscard]] bool empty() const noexcept { return x.empty(); }

    void resize(std::size_t n) {
        x.resize(n);
        y.resize(n);
        label.assign(n, -1);
    }
};

/// Cluster centres, in the same layout for the same reason.
struct Centroids {
    std::vector<Scalar> x;
    std::vector<Scalar> y;

    [[nodiscard]] std::size_t size() const noexcept { return x.size(); }

    void resize(std::size_t k) {
        x.resize(k);
        y.resize(k);
    }
};

struct Params {
    int k = 20;
    int max_iterations = 100;

    /// Stop when no centroid moves further than this in one iteration.
    /// The original always ran its full 100 iterations, so most of its
    /// measured time was spent recomputing a fixed point.
    Scalar tolerance = 1e-6;

    std::uint64_t seed = 42;

    /// Reduction granularity. Fixed, and deliberately independent of the
    /// thread count - see the note at the top of this file. Must be at
    /// least as large as the number of threads or ranks for the work to
    /// spread, and costs blocks * k * 24 bytes of accumulator.
    int blocks = 64;

    int threads = 1;
};

struct Result {
    Centroids centroids;
    int iterations = 0;
    bool converged = false;

    /// Sum of squared distances from each point to its centroid. The
    /// objective Lloyd's algorithm actually minimises, and the cheapest
    /// honest way to say whether two runs found the same solution.
    Scalar inertia = 0;

    /// Both clocks are reported, always. The original measured with
    /// clock(), which accumulates CPU time over all threads: on p busy
    /// threads it reports about p times the elapsed time, so a perfectly
    /// scaling program appears to take constant time. Printing the two side
    /// by side makes that visible rather than arguable - cpu/wall is the
    /// average number of cores kept busy.
    double wall_seconds = 0;
    double cpu_seconds = 0;

    /// Wall time spent in k-means++ seeding, included in wall_seconds.
    /// Broken out because the original's seeding was O(n*k^2) and serial,
    /// so at large k it quietly dominated the run it was preparing.
    double seed_seconds = 0;

    [[nodiscard]] double cores_used() const noexcept {
        return wall_seconds > 0 ? cpu_seconds / wall_seconds : 0.0;
    }
};

/// Half-open range of point indices.
struct Range {
    std::size_t begin = 0;
    std::size_t end = 0;

    [[nodiscard]] std::size_t size() const noexcept { return end - begin; }
};

/// Bounds of block `b` of `blocks`, over `n` points.
///
/// Computed from the index rather than from a chunk size, so the ranges
/// always tile [0, n) exactly whatever n and blocks are. This is the fix for
/// the dropped-remainder bug, and tests/partition_test.cpp holds it to it.
inline Range block_range(std::size_t n, int blocks, int b) noexcept {
    const auto total = static_cast<std::uint64_t>(n);
    const auto count = static_cast<std::uint64_t>(blocks);
    const auto index = static_cast<std::uint64_t>(b);
    return Range{static_cast<std::size_t>((total * index) / count),
                 static_cast<std::size_t>((total * (index + 1)) / count)};
}

/// Per-block accumulators, merged in block order.
///
/// Laid out as blocks x k so each block writes a contiguous span and two
/// blocks never touch the same cache line except at the boundary - the
/// false sharing that would otherwise dominate a k of 20.
struct Partials {
    int blocks = 0;
    int k = 0;

    std::vector<Scalar> sum_x;      ///< blocks * k
    std::vector<Scalar> sum_y;      ///< blocks * k
    std::vector<std::int64_t> count;  ///< blocks * k
    std::vector<Scalar> inertia;    ///< blocks

    /// The point furthest from its centroid in each block, kept so that an
    /// empty cluster can be given a real point to sit on. One candidate per
    /// block, so up to `blocks` empty clusters can be repaired per
    /// iteration; beyond that the remainder wait for the next one.
    std::vector<Scalar> farthest_distance;  ///< blocks
    /// Its coordinates, rather than its index. Under MPI the block that
    /// found the point and the rank that needs it to reposition a centroid
    /// are different processes, and an index into an array the receiver does
    /// not hold is useless to it.
    std::vector<Scalar> farthest_x;  ///< blocks
    std::vector<Scalar> farthest_y;  ///< blocks

    void reset(int block_count, int cluster_count) {
        blocks = block_count;
        k = cluster_count;
        const std::size_t cells =
            static_cast<std::size_t>(blocks) * static_cast<std::size_t>(k);
        sum_x.assign(cells, 0);
        sum_y.assign(cells, 0);
        count.assign(cells, 0);
        inertia.assign(static_cast<std::size_t>(blocks), 0);
        farthest_distance.assign(static_cast<std::size_t>(blocks), 0);
        farthest_x.assign(static_cast<std::size_t>(blocks), 0);
        farthest_y.assign(static_cast<std::size_t>(blocks), 0);
    }

    void clear() {
        std::fill(sum_x.begin(), sum_x.end(), Scalar{0});
        std::fill(sum_y.begin(), sum_y.end(), Scalar{0});
        std::fill(count.begin(), count.end(), std::int64_t{0});
        std::fill(inertia.begin(), inertia.end(), Scalar{0});
        // Zero, not -1, so that an MPI all-reduce by summation leaves each
        // block's entry exactly as its owning rank computed it: every other
        // rank contributes an exact zero.
        std::fill(farthest_distance.begin(), farthest_distance.end(), Scalar{0});
        std::fill(farthest_x.begin(), farthest_x.end(), Scalar{0});
        std::fill(farthest_y.begin(), farthest_y.end(), Scalar{0});
    }

    [[nodiscard]] std::size_t cell(int block, int cluster) const noexcept {
        return static_cast<std::size_t>(block) * static_cast<std::size_t>(k) +
               static_cast<std::size_t>(cluster);
    }
};

}  // namespace kmeans

#endif  // KMEANS_CORE_HPP
