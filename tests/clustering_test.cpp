// Does it actually cluster?
//
// The original was only ever run on `rand() % 1000` in both coordinates -
// uniform noise. k-means partitions uniform data into k cells quite
// happily, and the output looks like a clustering, but there is no right
// answer to compare against, so no run could have been wrong in a way
// anyone would have noticed.
//
// These tests use blobs with known centres, so "it found the clusters" is
// something that can fail.

#include <gtest/gtest.h>

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"
#include "kmeans/metrics.hpp"

namespace {

kmeans::Scalar distance(kmeans::Scalar ax, kmeans::Scalar ay, kmeans::Scalar bx,
                        kmeans::Scalar by) {
    const kmeans::Scalar dx = ax - bx;
    const kmeans::Scalar dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

TEST(Clustering, RecoversWellSeparatedBlobs) {
    // Five blobs with a standard deviation of 5 spread over a 1000-unit
    // square: far apart relative to their width, so there is exactly one
    // sensible answer and the algorithm has to find it.
    constexpr int k = 5;
    kmeans::Dataset data = kmeans::make_blobs(20'000, k, 1000, 5, 11);

    kmeans::Params params;
    params.k = k;
    params.seed = 3;
    const kmeans::Result result =
        kmeans::run(data.points, params, kmeans::Backend::Serial);

    // Every true centre must have a recovered centroid close to it. The
    // tolerance is a few standard deviations, not a few units: the sample
    // mean of a blob is not its generating centre.
    for (std::size_t t = 0; t < data.truth_centroids.size(); ++t) {
        kmeans::Scalar closest = kmeans::kInfinity;
        for (std::size_t c = 0; c < result.centroids.size(); ++c) {
            closest = std::min(closest, distance(data.truth_centroids.x[t],
                                                 data.truth_centroids.y[t],
                                                 result.centroids.x[c],
                                                 result.centroids.y[c]));
        }
        EXPECT_LT(closest, 2.0) << "no centroid found blob " << t;
    }
}

TEST(Clustering, AssignsEveryPointToTheBlobItCameFrom) {
    constexpr int k = 4;
    kmeans::Dataset data = kmeans::make_blobs(10'000, k, 1000, 4, 17);

    kmeans::Params params;
    params.k = k;
    params.seed = 5;
    kmeans::run(data.points, params, kmeans::Backend::Serial);

    // Cluster labels are arbitrary, so check the partition rather than the
    // names: every point from a given blob must land in the same cluster,
    // and distinct blobs in distinct clusters.
    std::map<kmeans::Label, std::set<kmeans::Label>> blob_to_clusters;
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        blob_to_clusters[data.truth[i]].insert(data.points.label[i]);
    }

    std::set<kmeans::Label> used;
    for (const auto& [blob, clusters] : blob_to_clusters) {
        EXPECT_EQ(clusters.size(), 1u)
            << "blob " << blob << " was split across clusters";
        used.insert(*clusters.begin());
    }
    EXPECT_EQ(used.size(), blob_to_clusters.size())
        << "two blobs were merged into one cluster";
}

TEST(Clustering, EveryPointIsLabelled) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // The remainder bug again, from the other end: after a run, no point
    // may still carry the -1 it was initialised with.
    kmeans::Dataset data = kmeans::make_blobs(50'003, 7, 1000, 30, 2);

    kmeans::Params params;
    params.k = 7;
    params.threads = 4;
    kmeans::run(data.points, params, kmeans::Backend::Pthreads);

    for (std::size_t i = 0; i < data.points.size(); ++i) {
        ASSERT_GE(data.points.label[i], 0) << "point " << i << " was never assigned";
        ASSERT_LT(data.points.label[i], 7) << "point " << i << " has a bogus label";
    }
}

TEST(Clustering, InertiaNeverIncreases) {
    // Lloyd's algorithm is a descent method: both halves of each iteration
    // can only reduce the objective. An implementation whose inertia goes
    // up has a bug, and this is the cheapest way to catch it.
    kmeans::Dataset data = kmeans::make_blobs(20'000, 6, 1000, 60, 23);

    kmeans::Scalar previous = kmeans::kInfinity;
    for (int iterations = 1; iterations <= 12; ++iterations) {
        kmeans::Dataset fresh = kmeans::make_blobs(20'000, 6, 1000, 60, 23);
        kmeans::Params params;
        params.k = 6;
        params.seed = 99;
        params.max_iterations = iterations;
        params.tolerance = 0;

        const kmeans::Result result =
            kmeans::run(fresh.points, params, kmeans::Backend::Serial);

        EXPECT_LE(result.inertia, previous * (1 + 1e-12))
            << "inertia rose at iteration " << iterations;
        previous = result.inertia;
    }
}

TEST(Clustering, ReportedInertiaMatchesTheReturnedCentroids) {
    // The labels left over from the final update belong to the centroids as
    // they were *before* it. Without the extra assignment pass at the end
    // of run(), the reported inertia would describe a clustering the caller
    // was never given.
    kmeans::Dataset data = kmeans::make_blobs(10'000, 5, 1000, 40, 31);

    kmeans::Params params;
    params.k = 5;
    params.max_iterations = 7;
    params.tolerance = 0;
    const kmeans::Result result =
        kmeans::run(data.points, params, kmeans::Backend::Serial);

    // Not bitwise: run() sums the per-block inertias in block order while
    // kmeans::inertia walks the points in index order, and floating-point
    // addition is not associative. Agreement to a relative 1e-12 is what
    // "the same quantity, added up differently" looks like; a real
    // mismatch - stale labels - would be visible at the first digit.
    EXPECT_NEAR(result.inertia, kmeans::inertia(data.points, result.centroids),
                result.inertia * 1e-12);
}

TEST(Clustering, StopsEarlyWhenItHasConverged) {
    // The original always ran its full 100 iterations, so most of what it
    // measured was recomputing a fixed point.
    kmeans::Dataset data = kmeans::make_blobs(20'000, 4, 1000, 5, 41);

    kmeans::Params params;
    params.k = 4;
    params.max_iterations = 500;
    params.tolerance = 1e-6;
    const kmeans::Result result =
        kmeans::run(data.points, params, kmeans::Backend::Serial);

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 500);
}

TEST(Clustering, EmptyClustersAreGivenRealPoints) {
    // k far larger than the structure in the data, which is how clusters
    // end up empty. The original left such a centroid parked where it was,
    // so k-means with k clusters quietly became k-means with fewer; here
    // every cluster must end up with at least one point.
    constexpr int k = 40;
    kmeans::Dataset data = kmeans::make_blobs(4'000, 3, 1000, 2, 53);

    kmeans::Params params;
    params.k = k;
    params.seed = 61;
    params.max_iterations = 60;
    const kmeans::Result result =
        kmeans::run(data.points, params, kmeans::Backend::Serial);

    std::vector<int> population(static_cast<std::size_t>(k), 0);
    for (std::size_t i = 0; i < data.points.size(); ++i) {
        ++population[static_cast<std::size_t>(data.points.label[i])];
    }
    const auto empty = std::count(population.begin(), population.end(), 0);
    EXPECT_EQ(empty, 0) << empty << " clusters ended up with no points";
    (void)result;
}

TEST(Clustering, RejectsImpossibleParameters) {
    kmeans::Dataset data = kmeans::make_blobs(100, 3, 1000, 10, 1);

    kmeans::Params params;
    params.k = 200;  // more clusters than points
    EXPECT_THROW(kmeans::run(data.points, params, kmeans::Backend::Serial),
                 std::invalid_argument);

    params.k = 0;
    EXPECT_THROW(kmeans::run(data.points, params, kmeans::Backend::Serial),
                 std::invalid_argument);

    params.k = 3;
    params.blocks = 0;
    EXPECT_THROW(kmeans::run(data.points, params, kmeans::Backend::Serial),
                 std::invalid_argument);
}

}  // namespace
