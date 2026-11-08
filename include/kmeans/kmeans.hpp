// k-means++ seeding and Lloyd's algorithm, written once for every backend.

#ifndef KMEANS_KMEANS_HPP
#define KMEANS_KMEANS_HPP

#include "kmeans/core.hpp"
#include "kmeans/executor.hpp"

namespace kmeans {

/// k-means++ seeding: D-squared weighted sampling, in O(n*k).
///
/// The original recomputed the distance from every point to every chosen
/// centroid each time it picked a new one, which is O(n*k^2) - at k = 100
/// that is a hundred times more distance computations than the algorithm
/// needs, and it ran single-threaded while the rest of the program was
/// being parallelised. Here a running nearest-centroid distance is carried
/// forward and updated once per new centroid, which is the standard
/// formulation, and the update runs on the executor.
///
/// The sampling itself stays on one thread and draws from a seeded
/// mt19937_64, so a given seed gives the same seeding on every backend and
/// every thread count.
Centroids kmeanspp(const PointCloud& points, const Params& params,
                   Executor& executor);

/// Lloyd's algorithm to convergence or `max_iterations`, whichever is
/// first.
///
/// On return `points.label` holds the assignment under the returned
/// centroids - a final assignment pass runs after the loop, because the
/// labels left over from the last update belong to the centroids as they
/// were before it.
Result run(PointCloud& points, const Params& params, Executor& executor);

/// Convenience: build the executor, run, and return. Used by the CLI and
/// the tests.
Result run(PointCloud& points, const Params& params, Backend backend);

}  // namespace kmeans

#endif  // KMEANS_KMEANS_HPP
