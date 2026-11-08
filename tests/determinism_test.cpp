// The claim that makes the parallel version trustworthy: every backend, at
// every thread count, computes the same thing - bitwise, not approximately.
//
// This is not a free property. Floating-point addition is not associative,
// so a parallel reduction that merges partial sums in whatever order the
// threads happen to finish gives a different answer at 2 threads than at 4,
// and a different one again on the next run. The usual response is to
// compare with a tolerance and hope the tolerance is generous enough, which
// means a real divergence hides inside it.
//
// Here the points are divided into a fixed number of blocks that does not
// depend on the thread count, and the per-block sums are merged in block
// order. The schedule decides who computes a block, never what is added to
// what. So equality is exact, and these tests can demand it.

#include <gtest/gtest.h>

#include "test_support.hpp"

#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"

namespace {

kmeans::Params test_params() {
    kmeans::Params params;
    params.k = 8;
    params.max_iterations = 30;
    params.tolerance = 0;  // run every iteration, so every run does equal work
    params.seed = 2024;
    params.blocks = 64;
    return params;
}

kmeans::Dataset test_data() {
    // A size that divides neither the block count nor any thread count
    // used here, so the remainder path is exercised throughout.
    return kmeans::make_blobs(50'003, 8, 1000, 25, 7);
}

kmeans::Result run_with(kmeans::Backend backend, int threads,
                        kmeans::PointCloud& points) {
    kmeans::Params params = test_params();
    params.threads = threads;
    auto executor = kmeans::make_executor(backend, threads);
    return kmeans::run(points, params, *executor);
}

void expect_identical(const kmeans::Result& reference, const kmeans::Result& actual,
                      const std::vector<kmeans::Label>& reference_labels,
                      const std::vector<kmeans::Label>& actual_labels,
                      const std::string& what) {
    ASSERT_EQ(reference.centroids.size(), actual.centroids.size()) << what;
    for (std::size_t c = 0; c < reference.centroids.size(); ++c) {
        EXPECT_DOUBLE_EQ(reference.centroids.x[c], actual.centroids.x[c])
            << what << ": centroid " << c << " x";
        EXPECT_DOUBLE_EQ(reference.centroids.y[c], actual.centroids.y[c])
            << what << ": centroid " << c << " y";
    }
    EXPECT_DOUBLE_EQ(reference.inertia, actual.inertia) << what << ": inertia";
    EXPECT_EQ(reference.iterations, actual.iterations) << what << ": iterations";
    EXPECT_EQ(reference_labels, actual_labels) << what << ": labels";
}

class Determinism : public ::testing::Test {
protected:
    void SetUp() override {
        reference_data_ = test_data();
        reference_ = run_with(kmeans::Backend::Serial, 1, reference_data_.points);
        reference_labels_ = reference_data_.points.label;
    }

    void check(kmeans::Backend backend, int threads) {
        kmeans::Dataset data = test_data();
        const kmeans::Result result = run_with(backend, threads, data.points);
        expect_identical(reference_, result, reference_labels_, data.points.label,
                         std::string(kmeans::to_string(backend)) + " x " +
                             std::to_string(threads));
    }

    kmeans::Dataset reference_data_;
    kmeans::Result reference_;
    std::vector<kmeans::Label> reference_labels_;
};

TEST_F(Determinism, SerialRepeatsItself) {
    check(kmeans::Backend::Serial, 1);
}

TEST_F(Determinism, PthreadsMatchesSerialAtEveryThreadCount) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // Including thread counts that do not divide the block count, where a
    // naive implementation's reduction order would change.
    for (int threads : {1, 2, 3, 4, 5, 8}) {
        check(kmeans::Backend::Pthreads, threads);
    }
}

TEST_F(Determinism, OpenMpMatchesSerialAtEveryThreadCount) {
    KMEANS_SKIP_WITHOUT_OPENMP();
    for (int threads : {1, 2, 3, 4, 8}) {
        check(kmeans::Backend::OpenMP, threads);
    }
}

TEST_F(Determinism, PthreadsAndOpenMpAgreeWithEachOther) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    KMEANS_SKIP_WITHOUT_OPENMP();
    kmeans::Dataset left = test_data();
    kmeans::Dataset right = test_data();
    const kmeans::Result a = run_with(kmeans::Backend::Pthreads, 4, left.points);
    const kmeans::Result b = run_with(kmeans::Backend::OpenMP, 4, right.points);
    expect_identical(a, b, left.points.label, right.points.label,
                     "pthreads vs openmp");
}

TEST_F(Determinism, RepeatedRunsOnOneExecutorAgree) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // The pool is reused across runs; stale state in it would show up here
    // and nowhere else.
    kmeans::Params params = test_params();
    params.threads = 4;
    auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, 4);

    kmeans::Dataset first = test_data();
    kmeans::Dataset second = test_data();
    const kmeans::Result a = kmeans::run(first.points, params, *executor);
    const kmeans::Result b = kmeans::run(second.points, params, *executor);
    expect_identical(a, b, first.points.label, second.points.label,
                     "two runs on one pool");
}

TEST_F(Determinism, BlockCountChangesTheAnswerAndIsAllowedTo) {
    // Honesty about the scope of the guarantee. Results are independent of
    // the *thread* count, not of the block count - a different block count
    // is a different summation order, and may differ in the last few bits.
    // Changing --blocks between runs that will be compared is therefore a
    // mistake, and the benchmark harness never does.
    kmeans::Dataset data = test_data();
    kmeans::Params params = test_params();
    params.blocks = 32;
    auto executor = kmeans::make_executor(kmeans::Backend::Serial, 1);
    const kmeans::Result other = kmeans::run(data.points, params, *executor);

    // Same clustering, to any sane tolerance - just not the same bits.
    EXPECT_NEAR(other.inertia, reference_.inertia,
                reference_.inertia * 1e-9);
}

}  // namespace
