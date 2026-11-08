// Synthetic datasets.
//
// The original clustered `rand() % 1000` in both coordinates - uniform
// noise with no cluster structure at all. k-means will happily partition
// uniform data into k Voronoi cells, and the result looks like clustering,
// but nothing is being recovered, so no quality metric computed on it means
// anything and no run can be checked for correctness.
//
// `blobs` generates k isotropic Gaussian clusters with known centres and
// known membership, which is what makes "the algorithm found the clusters"
// a test rather than a claim. `uniform` is kept so the original's
// configuration can still be reproduced exactly.

#ifndef KMEANS_DATASET_HPP
#define KMEANS_DATASET_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "kmeans/core.hpp"

namespace kmeans {

struct Dataset {
    PointCloud points;

    /// Which blob each point was drawn from, for `blobs` only. Empty for
    /// `uniform`, where there is no ground truth to have.
    std::vector<Label> truth;

    /// The centres the blobs were drawn around. Empty for `uniform`.
    Centroids truth_centroids;
};

/// `n` points, uniform on [0, extent) in both coordinates.
Dataset make_uniform(std::size_t n, Scalar extent, std::uint64_t seed);

/// `n` points drawn from `clusters` isotropic Gaussians of standard
/// deviation `spread`, with centres themselves spread over [0, extent).
Dataset make_blobs(std::size_t n, int clusters, Scalar extent, Scalar spread,
                   std::uint64_t seed);

/// Read points from a two-column CSV with a header line, so the benchmark
/// can be pointed at real data.
Dataset load_csv(const std::string& path);

/// Write points and their labels, for plotting.
void save_csv(const PointCloud& points, const std::string& path);

}  // namespace kmeans

#endif  // KMEANS_DATASET_HPP
