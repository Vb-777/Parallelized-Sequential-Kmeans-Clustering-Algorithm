#include "kmeans/kmeans.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "kmeans/kernels.hpp"
#include "kmeans/timer.hpp"

namespace kmeans {
namespace {

void validate(const PointCloud& points, const Params& params) {
    if (points.empty()) {
        throw std::invalid_argument("no points");
    }
    if (params.k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (static_cast<std::size_t>(params.k) > points.size()) {
        throw std::invalid_argument("k exceeds the number of points");
    }
    if (params.blocks <= 0) {
        throw std::invalid_argument("blocks must be positive");
    }
}

}  // namespace

Centroids kmeanspp(const PointCloud& points, const Params& params,
                   Executor& executor) {
    validate(points, params);

    const std::size_t n = points.size();
    const int k = params.k;
    const int blocks = params.blocks;

    Centroids centroids;
    centroids.resize(static_cast<std::size_t>(k));

    std::mt19937_64 rng(params.seed);
    std::uniform_int_distribution<std::size_t> any_point(0, n - 1);

    const std::size_t first = any_point(rng);
    centroids.x[0] = points.x[first];
    centroids.y[0] = points.y[first];

    // Running squared distance from each point to its nearest chosen
    // centroid. Carrying this forward is what turns O(n*k^2) into O(n*k).
    std::vector<Scalar> nearest(n, kInfinity);
    std::vector<Scalar> block_sum(static_cast<std::size_t>(blocks), 0);

    std::uniform_real_distribution<Scalar> unit(0, 1);

    for (int c = 1; c < k; ++c) {
        const Scalar cx = centroids.x[static_cast<std::size_t>(c - 1)];
        const Scalar cy = centroids.y[static_cast<std::size_t>(c - 1)];

        executor.for_each_block(blocks, [&](int b) {
            const Range range = block_range(n, blocks, b);
            Scalar sum = 0;
            for (std::size_t i = range.begin; i < range.end; ++i) {
                const Scalar dx = points.x[i] - cx;
                const Scalar dy = points.y[i] - cy;
                const Scalar distance = dx * dx + dy * dy;
                if (distance < nearest[i]) {
                    nearest[i] = distance;
                }
                sum += nearest[i];
            }
            block_sum[static_cast<std::size_t>(b)] = sum;
        });

        // Block order again, so the total - and therefore the point the
        // sample lands on - does not depend on the schedule.
        Scalar total = 0;
        for (int b = 0; b < blocks; ++b) {
            total += block_sum[static_cast<std::size_t>(b)];
        }

        std::size_t chosen = 0;
        if (!(total > 0)) {
            // Every point coincides with a centroid already: there is no
            // weight to sample by, so fall back to a uniform draw rather
            // than dividing by zero.
            chosen = any_point(rng);
        } else {
            Scalar target = unit(rng) * total;

            // Find the block the sample falls in from the block sums, then
            // scan only that block: O(blocks + n/blocks) instead of O(n).
            int block = blocks - 1;
            for (int b = 0; b < blocks; ++b) {
                const Scalar weight = block_sum[static_cast<std::size_t>(b)];
                if (target < weight) {
                    block = b;
                    break;
                }
                target -= weight;
            }

            const Range range = block_range(n, blocks, block);
            chosen = range.end > range.begin ? range.end - 1 : range.begin;
            for (std::size_t i = range.begin; i < range.end; ++i) {
                target -= nearest[i];
                if (target <= 0) {
                    chosen = i;
                    break;
                }
            }
        }

        centroids.x[static_cast<std::size_t>(c)] = points.x[chosen];
        centroids.y[static_cast<std::size_t>(c)] = points.y[chosen];
    }

    return centroids;
}

Result run(PointCloud& points, const Params& params, Executor& executor) {
    validate(points, params);

    const std::size_t n = points.size();
    const int blocks = params.blocks;

    Timer timer;

    Result result;
    Centroids centroids = kmeanspp(points, params, executor);
    result.seed_seconds = timer.wall_seconds();

    Partials partials;
    partials.reset(blocks, params.k);
    Centroids next;

    const Scalar tolerance_squared = params.tolerance * params.tolerance;

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        partials.clear();

        executor.for_each_block(blocks, [&](int b) {
            assign_block(points, centroids, block_range(n, blocks, b), b, partials);
        });

        int empty_clusters = 0;
        const Scalar shift_squared =
            merge_and_update(partials, centroids, next, empty_clusters);
        if (empty_clusters > 0) {
            repair_empty_clusters(partials, next);
        }

        centroids = next;
        result.iterations = iteration + 1;

        // An iteration that moved a centroid onto a distant point has not
        // converged however small the other shifts were, so the empty-
        // cluster repair suppresses the convergence test for that round.
        if (empty_clusters == 0 && shift_squared <= tolerance_squared) {
            result.converged = true;
            break;
        }
    }

    // The labels in `points` belong to the centroids as they were before
    // the last update. One more assignment pass makes them, and the
    // inertia, describe the centroids actually returned.
    partials.clear();
    executor.for_each_block(blocks, [&](int b) {
        assign_block(points, centroids, block_range(n, blocks, b), b, partials);
    });

    result.centroids = centroids;
    result.inertia = total_inertia(partials);
    result.wall_seconds = timer.wall_seconds();
    result.cpu_seconds = timer.cpu_seconds();
    return result;
}

Result run(PointCloud& points, const Params& params, Backend backend) {
    auto executor = make_executor(backend, params.threads);
    return run(points, params, *executor);
}

}  // namespace kmeans
