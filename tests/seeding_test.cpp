// k-means++ seeding.
//
// Two things to establish. First that it is k-means++ at all - D-squared
// weighted sampling from the data points, not uniform sampling with a
// fancier name. Second that the O(n*k) reformulation picks the same kind of
// seeds the textbook O(n*k^2) version would, since that is the change that
// made it a hundred times cheaper at k = 100.

#include <gtest/gtest.h>

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"
#include "kmeans/metrics.hpp"

namespace {

/// Clustering cost of a centroid set: the inertia it would give if every
/// point were assigned to its nearest centre. This is the quantity
/// k-means++ bounds, and it needs no labels.
kmeans::Scalar seeding_cost(const kmeans::PointCloud& points,
                            const kmeans::Centroids& centroids) {
    kmeans::Scalar total = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        kmeans::Scalar best = kmeans::kInfinity;
        for (std::size_t c = 0; c < centroids.size(); ++c) {
            const kmeans::Scalar dx = points.x[i] - centroids.x[c];
            const kmeans::Scalar dy = points.y[i] - centroids.y[c];
            best = std::min(best, dx * dx + dy * dy);
        }
        total += best;
    }
    return total;
}

TEST(Seeding, ChoosesExistingDataPoints) {
    // k-means++ seeds *are* data points. A seeding that invented
    // coordinates would be a different algorithm.
    kmeans::Dataset data = kmeans::make_blobs(5'000, 6, 1000, 20, 3);

    kmeans::Params params;
    params.k = 6;
    auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
    const kmeans::Centroids seeds = kmeans::kmeanspp(data.points, params, *executor);

    ASSERT_EQ(seeds.size(), 6u);
    for (std::size_t c = 0; c < seeds.size(); ++c) {
        bool found = false;
        for (std::size_t i = 0; i < data.points.size() && !found; ++i) {
            found = data.points.x[i] == seeds.x[c] && data.points.y[i] == seeds.y[c];
        }
        EXPECT_TRUE(found) << "seed " << c << " is not one of the data points";
    }
}

TEST(Seeding, SpreadsSeedsAcrossSeparatedBlobs) {
    // The property that makes k-means++ worth having: with six well
    // separated blobs and six seeds, D-squared weighting should put roughly
    // one seed in each, so no two seeds land on top of each other. Uniform
    // sampling would collide regularly.
    constexpr int k = 6;
    int well_spread = 0;
    constexpr int trials = 20;

    for (int trial = 0; trial < trials; ++trial) {
        kmeans::Dataset data =
            kmeans::make_blobs(5'000, k, 1000, 3, static_cast<std::uint64_t>(trial));
        kmeans::Params params;
        params.k = k;
        params.seed = static_cast<std::uint64_t>(100 + trial);

        auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
        const kmeans::Centroids seeds =
            kmeans::kmeanspp(data.points, params, *executor);

        // Blobs have a standard deviation of 3 and centres scattered over
        // 1000 units, so two seeds within 50 units of each other are in the
        // same blob.
        kmeans::Scalar closest_pair = kmeans::kInfinity;
        for (std::size_t a = 0; a < seeds.size(); ++a) {
            for (std::size_t b = a + 1; b < seeds.size(); ++b) {
                const kmeans::Scalar dx = seeds.x[a] - seeds.x[b];
                const kmeans::Scalar dy = seeds.y[a] - seeds.y[b];
                closest_pair = std::min(closest_pair, std::sqrt(dx * dx + dy * dy));
            }
        }
        if (closest_pair > 50) {
            ++well_spread;
        }
    }

    // Not all, because D-squared sampling is random and may double up; but
    // overwhelmingly, which uniform sampling would not manage.
    EXPECT_GE(well_spread, trials * 3 / 4)
        << "seeds collided in " << (trials - well_spread) << " of " << trials
        << " trials - this does not look like D-squared weighted sampling";
}

TEST(Seeding, IsReproducibleFromItsSeed) {
    kmeans::Dataset data = kmeans::make_blobs(5'000, 5, 1000, 20, 9);
    kmeans::Params params;
    params.k = 5;
    params.seed = 777;

    auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
    const kmeans::Centroids first = kmeans::kmeanspp(data.points, params, *executor);
    const kmeans::Centroids second = kmeans::kmeanspp(data.points, params, *executor);

    for (std::size_t c = 0; c < first.size(); ++c) {
        EXPECT_DOUBLE_EQ(first.x[c], second.x[c]);
        EXPECT_DOUBLE_EQ(first.y[c], second.y[c]);
    }
}

TEST(Seeding, DoesNotDependOnTheThreadCount) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // The weight sum is reduced in block order, so the sampled point is the
    // same however many threads computed the weights.
    kmeans::Dataset data = kmeans::make_blobs(20'003, 7, 1000, 20, 13);
    kmeans::Params params;
    params.k = 7;
    params.seed = 4242;

    auto serial = kmeans::make_executor(kmeans::Backend::Serial, 1);
    const kmeans::Centroids reference =
        kmeans::kmeanspp(data.points, params, *serial);

    for (int threads : {1, 2, 3, 4}) {
        auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, threads);
        const kmeans::Centroids seeds =
            kmeans::kmeanspp(data.points, params, *executor);
        for (std::size_t c = 0; c < reference.size(); ++c) {
            EXPECT_DOUBLE_EQ(reference.x[c], seeds.x[c])
                << "seed " << c << " differs at " << threads << " threads";
            EXPECT_DOUBLE_EQ(reference.y[c], seeds.y[c])
                << "seed " << c << " differs at " << threads << " threads";
        }
    }
}

TEST(Seeding, DifferentSeedsGiveDifferentStartingPoints) {
    kmeans::Dataset data = kmeans::make_blobs(5'000, 5, 1000, 20, 19);
    std::set<std::pair<kmeans::Scalar, kmeans::Scalar>> firsts;

    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        kmeans::Params params;
        params.k = 5;
        params.seed = seed;
        auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
        const kmeans::Centroids seeds =
            kmeans::kmeanspp(data.points, params, *executor);
        firsts.insert({seeds.x[0], seeds.y[0]});
    }
    EXPECT_GT(firsts.size(), 1u) << "the seed does not affect the seeding";
}

TEST(Seeding, HandlesDuplicatePoints) {
    // Every point identical: the D-squared weights are all zero after the
    // first centroid, so the weighted draw has nothing to divide by. It
    // must fall back rather than produce NaN centroids.
    kmeans::PointCloud points;
    points.resize(1'000);
    std::fill(points.x.begin(), points.x.end(), 5.0);
    std::fill(points.y.begin(), points.y.end(), 7.0);

    kmeans::Params params;
    params.k = 4;
    auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
    const kmeans::Centroids seeds = kmeans::kmeanspp(points, params, *executor);

    for (std::size_t c = 0; c < seeds.size(); ++c) {
        EXPECT_DOUBLE_EQ(seeds.x[c], 5.0);
        EXPECT_DOUBLE_EQ(seeds.y[c], 7.0);
        EXPECT_FALSE(std::isnan(seeds.x[c]));
    }
}

TEST(Seeding, CostsLessThanUniformSeedingOnAverage) {
    // What k-means++ is for. Its guarantee is on the expected cost of the
    // *initial* centroids - O(log k) times optimal - so the comparison is
    // against uniformly sampled seeds, averaged over trials, before any
    // Lloyd iteration has had a chance to rescue either.
    //
    // Averaged, not per-trial: D-squared sampling is random and will lose
    // occasionally, and a test that demanded it always win would be flaky.
    constexpr int k = 10;
    constexpr int trials = 15;

    kmeans::Scalar kmeanspp_total = 0;
    kmeans::Scalar uniform_total = 0;

    for (int trial = 0; trial < trials; ++trial) {
        kmeans::Dataset data = kmeans::make_blobs(
            4'000, k, 1000, 8, static_cast<std::uint64_t>(trial));

        kmeans::Params params;
        params.k = k;
        params.seed = static_cast<std::uint64_t>(trial * 31 + 7);

        auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
        kmeanspp_total += seeding_cost(
            data.points, kmeans::kmeanspp(data.points, params, *executor));

        std::mt19937_64 rng(params.seed);
        std::uniform_int_distribution<std::size_t> pick(0, data.points.size() - 1);
        kmeans::Centroids uniform;
        uniform.resize(k);
        for (int c = 0; c < k; ++c) {
            const std::size_t index = pick(rng);
            uniform.x[static_cast<std::size_t>(c)] = data.points.x[index];
            uniform.y[static_cast<std::size_t>(c)] = data.points.y[index];
        }
        uniform_total += seeding_cost(data.points, uniform);
    }

    EXPECT_LT(kmeanspp_total, uniform_total)
        << "k-means++ seeding cost " << kmeanspp_total << " against uniform "
        << uniform_total << " - this does not look like D-squared sampling";
}

}  // namespace
