// The metrics, and in particular the bug that made the original's
// silhouette column 500,000 copies of the number 1.

#include <gtest/gtest.h>

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <set>
#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"
#include "kmeans/metrics.hpp"

namespace {

std::unique_ptr<kmeans::Executor> serial() {
    return kmeans::make_executor(kmeans::Backend::Serial, 1);
}

/// Two tight, far-apart groups: the clustering any method should find, and
/// a silhouette close to 1 is the right answer for it.
kmeans::PointCloud two_groups() {
    kmeans::PointCloud points;
    points.resize(200);
    for (std::size_t i = 0; i < 200; ++i) {
        const bool right = i >= 100;
        points.x[i] = (right ? 1000.0 : 0.0) + static_cast<double>(i % 10) * 0.1;
        points.y[i] = static_cast<double>(i % 7) * 0.1;
        points.label[i] = right ? 1 : 0;
    }
    return points;
}

TEST(Silhouette, IsNotAlwaysOne) {
    // The regression test for the original defect. It called
    // silhouetteScore({point}, centroids) per point - a one-element vector -
    // so a came out 0, every other cluster was empty, b stayed at DBL_MAX,
    // and (b - a) / max(a, b) was 1 for every row in the file.
    //
    // Here the same clustering must produce a range of per-point values,
    // and a mean strictly below 1.
    kmeans::Dataset data = kmeans::make_blobs(600, 4, 100, 12, 5);
    kmeans::Params params;
    params.k = 4;
    kmeans::run(data.points, params, kmeans::Backend::Serial);

    auto executor = serial();
    const kmeans::Scalar score = kmeans::silhouette_exact(data.points, 4, *executor);

    EXPECT_LT(score, 1.0) << "silhouette came out at the maximum again";
    EXPECT_GT(score, -1.0);
}

TEST(Silhouette, IsHighForWellSeparatedGroups) {
    kmeans::PointCloud points = two_groups();
    auto executor = serial();
    EXPECT_GT(kmeans::silhouette_exact(points, 2, *executor), 0.9);
}

TEST(Silhouette, IsLowWhenTheClusteringIsWrong) {
    // Same points, but split across the groups rather than along them. A
    // metric that cannot tell these two labellings apart is not measuring
    // anything - which is precisely what the original's could not do.
    kmeans::PointCloud points = two_groups();
    for (std::size_t i = 0; i < points.size(); ++i) {
        points.label[i] = static_cast<kmeans::Label>(i % 2);
    }

    auto executor = serial();
    EXPECT_LT(kmeans::silhouette_exact(points, 2, *executor), 0.1);
}

TEST(Silhouette, ASingletonClusterScoresZeroNotOne) {
    // The exact case the original got backwards. A cluster of one point has
    // no within-cluster distance, so its silhouette is undefined and is
    // conventionally reported as 0 - not as 1, the best possible score.
    kmeans::PointCloud points;
    points.resize(101);
    for (std::size_t i = 0; i < 100; ++i) {
        points.x[i] = static_cast<double>(i % 10);
        points.y[i] = static_cast<double>(i / 10);
        points.label[i] = 0;
    }
    points.x[100] = 500;
    points.y[100] = 500;
    points.label[100] = 1;  // alone in its cluster

    auto executor = serial();
    const kmeans::Scalar score = kmeans::silhouette_exact(points, 2, *executor);

    // 100 points scoring near 1 and one scoring 0 averages just under 1;
    // if the singleton had scored 1 the mean would be 1 exactly.
    EXPECT_LT(score, 1.0);
    EXPECT_GT(score, 0.9);
}

TEST(Silhouette, SampledAgreesWithExact) {
    // The sampled estimator is what makes the metric usable at scale. On a
    // dataset small enough to compute both, they must agree.
    kmeans::Dataset data = kmeans::make_blobs(2'000, 5, 1000, 30, 7);
    kmeans::Params params;
    params.k = 5;
    kmeans::run(data.points, params, kmeans::Backend::Serial);

    auto executor = serial();
    const kmeans::Scalar exact = kmeans::silhouette_exact(data.points, 5, *executor);
    const kmeans::Scalar sampled =
        kmeans::silhouette_sampled(data.points, 5, 500, 1, *executor);

    // The standard error of a mean over 500 samples of a quantity bounded
    // in [-1, 1] is well under 0.05.
    EXPECT_NEAR(sampled, exact, 0.05);
}

TEST(Silhouette, SamplingMoreThanTheDatasetIsExact) {
    kmeans::Dataset data = kmeans::make_blobs(300, 3, 1000, 20, 11);
    kmeans::Params params;
    params.k = 3;
    kmeans::run(data.points, params, kmeans::Backend::Serial);

    auto executor = serial();
    EXPECT_DOUBLE_EQ(kmeans::silhouette_sampled(data.points, 3, 10'000, 1, *executor),
                     kmeans::silhouette_exact(data.points, 3, *executor));
}

TEST(Silhouette, DoesNotDependOnTheThreadCount) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    kmeans::Dataset data = kmeans::make_blobs(1'500, 4, 1000, 25, 13);
    kmeans::Params params;
    params.k = 4;
    kmeans::run(data.points, params, kmeans::Backend::Serial);

    auto reference_executor = serial();
    const kmeans::Scalar reference =
        kmeans::silhouette_sampled(data.points, 4, 400, 9, *reference_executor);

    for (int threads : {2, 3, 4}) {
        auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, threads);
        EXPECT_DOUBLE_EQ(kmeans::silhouette_sampled(data.points, 4, 400, 9, *executor),
                         reference);
    }
}

/// The centroids a labelling implies, so a metric is scored against the
/// clustering it is actually describing.
kmeans::Centroids centroids_of(const kmeans::PointCloud& points, int k) {
    kmeans::Centroids centroids;
    centroids.resize(static_cast<std::size_t>(k));
    std::vector<double> count(static_cast<std::size_t>(k), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto c = static_cast<std::size_t>(points.label[i]);
        centroids.x[c] += points.x[i];
        centroids.y[c] += points.y[i];
        count[c] += 1;
    }
    for (std::size_t c = 0; c < centroids.size(); ++c) {
        if (count[c] > 0) {
            centroids.x[c] /= count[c];
            centroids.y[c] /= count[c];
        }
    }
    return centroids;
}

TEST(DaviesBouldin, PrefersTheBetterClustering) {
    // Lower is better. The good labelling splits along the gap between the
    // two groups; the bad one interleaves them, so each cluster spans the
    // whole 1000-unit spread while their centroids nearly coincide - large
    // spread over small separation, which is exactly what the index
    // punishes.
    kmeans::PointCloud good = two_groups();

    kmeans::PointCloud bad = two_groups();
    for (std::size_t i = 0; i < bad.size(); ++i) {
        bad.label[i] = static_cast<kmeans::Label>(i % 2);
    }

    const kmeans::Scalar good_score =
        kmeans::davies_bouldin(good, centroids_of(good, 2));
    const kmeans::Scalar bad_score =
        kmeans::davies_bouldin(bad, centroids_of(bad, 2));

    EXPECT_LT(good_score, bad_score)
        << "good " << good_score << " vs bad " << bad_score;
    EXPECT_LT(good_score, 0.1);
}

TEST(Inertia, IsZeroWhenEveryPointSitsOnItsCentroid) {
    kmeans::PointCloud points;
    points.resize(10);
    std::fill(points.x.begin(), points.x.end(), 3.0);
    std::fill(points.y.begin(), points.y.end(), 4.0);
    std::fill(points.label.begin(), points.label.end(), 0);

    kmeans::Centroids centroids;
    centroids.resize(1);
    centroids.x[0] = 3.0;
    centroids.y[0] = 4.0;

    EXPECT_DOUBLE_EQ(kmeans::inertia(points, centroids), 0.0);
}

TEST(Inertia, CountsSquaredDistances) {
    kmeans::PointCloud points;
    points.resize(2);
    points.x[0] = 0;
    points.y[0] = 0;
    points.label[0] = 0;
    points.x[1] = 3;
    points.y[1] = 4;
    points.label[1] = 0;

    kmeans::Centroids centroids;
    centroids.resize(1);
    centroids.x[0] = 0;
    centroids.y[0] = 0;

    EXPECT_DOUBLE_EQ(kmeans::inertia(points, centroids), 25.0);
}

}  // namespace
