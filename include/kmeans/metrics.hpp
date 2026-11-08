// Cluster quality metrics that mean something.
//
// The original wrote a silhouette column in which every one of the 100,000
// rows was exactly 1. The cause is a one-line mistake with a large blast
// radius: `silhouetteScore({point}, centroids)` was called per point, which
// builds a vector holding that point alone. Inside, the same-cluster
// average `a` is computed over that one-element vector and comes out 0,
// every *other* cluster is empty so `b` is never updated from DBL_MAX, and
// (b - a) / max(a, b) is 1. The column could not have been anything else.
//
// There is a second, quieter error in it: distances were accumulated as
// squared distances. Silhouette is defined on the metric, not its square,
// and the two do not give the same ranking of clusterings.
//
// What is offered here instead:
//
//   inertia              exact, O(n*k), already computed during assignment
//   davies_bouldin       exact, O(n + k^2)
//   silhouette_exact     exact, O(n^2) - honest about why it is for small n
//   silhouette_sampled   O(m*n), m sampled points against all n
//
// The sampled form is the one to use at scale. It is what scikit-learn's
// `sample_size` argument does, it is unbiased, and `tests/metrics_test.cpp`
// checks it against the exact value on a dataset small enough to compute
// both.

#ifndef KMEANS_METRICS_HPP
#define KMEANS_METRICS_HPP

#include <cstdint>

#include "kmeans/core.hpp"
#include "kmeans/executor.hpp"

namespace kmeans {

/// Sum of squared distances from each point to the centroid of its
/// assigned cluster. Requires `points.label` to be current.
Scalar inertia(const PointCloud& points, const Centroids& centroids);

/// Davies-Bouldin index: the average, over clusters, of the worst ratio of
/// within-cluster spread to between-cluster separation. Lower is better,
/// and unlike silhouette it is exact in linear time.
Scalar davies_bouldin(const PointCloud& points, const Centroids& centroids);

/// Mean silhouette coefficient over every point, computed exactly.
///
/// O(n^2) in distance computations - quadratic is what the definition
/// costs, and no amount of arranging changes that. Intended for test-sized
/// data; `silhouette_sampled` is the one to call on a real dataset.
Scalar silhouette_exact(const PointCloud& points, int k, Executor& executor);

/// Mean silhouette over `samples` randomly chosen points, each evaluated
/// against the full dataset.
///
/// The estimate's standard error falls as 1/sqrt(samples) and does not
/// depend on n, so a few thousand samples describe a hundred million points
/// as well as they describe a hundred thousand.
Scalar silhouette_sampled(const PointCloud& points, int k, std::size_t samples,
                          std::uint64_t seed, Executor& executor);

}  // namespace kmeans

#endif  // KMEANS_METRICS_HPP
