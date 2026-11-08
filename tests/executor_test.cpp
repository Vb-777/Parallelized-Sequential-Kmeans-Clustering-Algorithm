// The dispatch layer itself: every block run exactly once, pools reusable,
// and no thread left behind.

#include <gtest/gtest.h>

#include "test_support.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

#include "kmeans/executor.hpp"

namespace {

std::vector<int> run_counts(kmeans::Backend backend, int threads, int blocks) {
    auto executor = kmeans::make_executor(backend, threads);
    std::vector<std::atomic<int>> counts(static_cast<std::size_t>(blocks));
    for (auto& count : counts) {
        count.store(0);
    }

    executor->for_each_block(blocks, [&](int b) {
        counts[static_cast<std::size_t>(b)].fetch_add(1);
    });

    std::vector<int> plain(static_cast<std::size_t>(blocks));
    for (std::size_t i = 0; i < plain.size(); ++i) {
        plain[i] = counts[i].load();
    }
    return plain;
}

TEST(Executor, RunsEveryBlockExactlyOnce) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    for (kmeans::Backend backend :
         {kmeans::Backend::Serial, kmeans::Backend::Pthreads}) {
        for (int threads : {1, 2, 4, 7}) {
            const std::vector<int> counts = run_counts(backend, threads, 64);
            for (std::size_t b = 0; b < counts.size(); ++b) {
                EXPECT_EQ(counts[b], 1)
                    << kmeans::to_string(backend) << " x " << threads << ": block "
                    << b << " ran " << counts[b] << " times";
            }
        }
    }
}

TEST(Executor, OpenMpRunsEveryBlockExactlyOnce) {
    KMEANS_SKIP_WITHOUT_OPENMP();
    for (int threads : {1, 2, 4}) {
        const std::vector<int> counts =
            run_counts(kmeans::Backend::OpenMP, threads, 64);
        for (std::size_t b = 0; b < counts.size(); ++b) {
            EXPECT_EQ(counts[b], 1) << "openmp x " << threads << ": block " << b;
        }
    }
}

TEST(Executor, HandlesMoreThreadsThanBlocks) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // Eight threads over two blocks: six of them have nothing to do and
    // must still reach the barrier rather than hanging.
    const std::vector<int> counts = run_counts(kmeans::Backend::Pthreads, 8, 2);
    EXPECT_EQ(counts, (std::vector<int>{1, 1}));
}

TEST(Executor, HandlesNoBlocksAtAll) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, 4);
    int calls = 0;
    executor->for_each_block(0, [&](int) { ++calls; });
    EXPECT_EQ(calls, 0);
}

TEST(Executor, ThePoolIsReusableAcrossRounds) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // The point of the pool. The original created and joined threads inside
    // every iteration; this must survive being driven repeatedly.
    auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, 4);

    std::atomic<int> total{0};
    for (int round = 0; round < 200; ++round) {
        executor->for_each_block(32, [&](int) { total.fetch_add(1); });
    }
    EXPECT_EQ(total.load(), 200 * 32);
}

TEST(Executor, BlocksReallyRunConcurrently) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    // A pool that quietly ran every block on the calling thread would pass
    // every test above, and would also pass a test that merely counted
    // distinct thread ids - under load the caller can drain the queue
    // before the workers wake, which made an earlier version of this test
    // fail intermittently in CI for no good reason.
    //
    // So this asserts the property directly rather than a symptom of it:
    // with exactly as many blocks as threads, every block waits until all
    // of them have arrived. If the blocks run concurrently they all pass
    // through together; if they are serialised, the first one waits alone
    // and the timeout expires.
    constexpr int threads = 4;
    auto executor = kmeans::make_executor(kmeans::Backend::Pthreads, threads);

    std::mutex guard;
    std::condition_variable arrived;
    int present = 0;
    std::atomic<int> timed_out{0};

    executor->for_each_block(threads, [&](int) {
        std::unique_lock<std::mutex> lock(guard);
        ++present;
        arrived.notify_all();
        // Generous, because this only has to distinguish "concurrent" from
        // "serialised", and a loaded CI machine can be slow to schedule.
        if (!arrived.wait_for(lock, std::chrono::seconds(5),
                              [&] { return present == threads; })) {
            timed_out.fetch_add(1);
        }
    });

    EXPECT_EQ(timed_out.load(), 0)
        << "blocks did not run concurrently: " << timed_out.load() << " of "
        << threads << " waited for colleagues that never arrived";
    EXPECT_EQ(present, threads);
}

TEST(Executor, ReportsItsThreadCount) {
    KMEANS_SKIP_WITHOUT_PTHREADS();
    EXPECT_EQ(kmeans::make_executor(kmeans::Backend::Serial, 8)->threads(), 1);
    EXPECT_EQ(kmeans::make_executor(kmeans::Backend::Pthreads, 1)->threads(), 1);
    EXPECT_EQ(kmeans::make_executor(kmeans::Backend::Pthreads, 4)->threads(), 4);
}

TEST(Executor, NamesParse) {
    kmeans::Backend backend{};
    EXPECT_TRUE(kmeans::parse_backend("serial", backend));
    EXPECT_EQ(backend, kmeans::Backend::Serial);
    EXPECT_TRUE(kmeans::parse_backend("pthreads", backend));
    EXPECT_EQ(backend, kmeans::Backend::Pthreads);
    EXPECT_TRUE(kmeans::parse_backend("openmp", backend));
    EXPECT_EQ(backend, kmeans::Backend::OpenMP);
    EXPECT_FALSE(kmeans::parse_backend("mpi", backend));
    EXPECT_FALSE(kmeans::parse_backend("", backend));
}

TEST(Executor, OpenMpIsRefusedClearlyWhenAbsent) {
    if (kmeans::openmp_available()) {
        GTEST_SKIP() << "built with OpenMP, so there is nothing to refuse";
    }
    EXPECT_THROW(kmeans::make_executor(kmeans::Backend::OpenMP, 2),
                 std::runtime_error);
}

}  // namespace
