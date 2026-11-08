// The dropped-remainder bug, held down.
//
// The original computed `chunkSize = n / numThreads` and gave thread t the
// range [t*chunkSize, (t+1)*chunkSize). When numThreads did not divide n,
// the last n % numThreads points were never assigned by anyone: they kept
// whatever label they had from the previous iteration, and contributed to
// no centroid. At n = 500,000 and 32 threads that is 16 points - small
// enough never to be noticed, large enough to make the parallel result
// differ from the serial one for no reason anybody could see.
//
// These tests are about one property: the block ranges tile [0, n) exactly,
// for every n and every block count, with no gaps and no overlaps.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "kmeans/core.hpp"

namespace {

using kmeans::block_range;
using kmeans::Range;

void expect_exact_tiling(std::size_t n, int blocks) {
    std::vector<int> covered(n, 0);
    std::size_t previous_end = 0;

    for (int b = 0; b < blocks; ++b) {
        const Range range = block_range(n, blocks, b);

        EXPECT_GE(range.end, range.begin) << "block " << b << " runs backwards";
        EXPECT_EQ(range.begin, previous_end)
            << "block " << b << " does not start where block " << b - 1 << " ended";
        previous_end = range.end;

        for (std::size_t i = range.begin; i < range.end; ++i) {
            ++covered[i];
        }
    }

    EXPECT_EQ(previous_end, n) << "the last block does not reach n";
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(covered[i], 1) << "point " << i << " is covered " << covered[i]
                                 << " times, not once";
    }
}

TEST(Partition, TilesExactlyWhenBlocksDivideN) {
    expect_exact_tiling(1024, 64);
}

TEST(Partition, TilesExactlyWhenBlocksDoNotDivideN) {
    // The case the original got wrong. 500,000 points over 32 ways leaves a
    // remainder of 16.
    expect_exact_tiling(500'000 % 1'000 + 977, 32);
    expect_exact_tiling(1003, 32);
    expect_exact_tiling(97, 10);
}

TEST(Partition, HandlesMoreBlocksThanPoints) {
    // Some blocks must come out empty rather than overlapping or running
    // past the end.
    expect_exact_tiling(5, 64);
}

TEST(Partition, HandlesASingleBlock) {
    const Range range = block_range(12345, 1, 0);
    EXPECT_EQ(range.begin, 0u);
    EXPECT_EQ(range.end, 12345u);
}

TEST(Partition, HandlesZeroPoints) {
    expect_exact_tiling(0, 8);
}

TEST(Partition, BlockBoundsDoNotDependOnTheThreadCount) {
    // The property the whole determinism story rests on: the decomposition
    // is a function of (n, blocks) and nothing else.
    constexpr std::size_t n = 123'457;
    constexpr int blocks = 64;

    std::vector<std::size_t> boundaries;
    for (int b = 0; b <= blocks; ++b) {
        boundaries.push_back(b < blocks ? block_range(n, blocks, b).begin : n);
    }

    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int b = 0; b < blocks; ++b) {
            EXPECT_EQ(block_range(n, blocks, b).begin,
                      boundaries[static_cast<std::size_t>(b)]);
        }
    }
}

TEST(Partition, SizesDifferByAtMostOne) {
    // Load balance: no block may be more than one point larger than any
    // other, or the slowest block would set the pace for every round.
    constexpr std::size_t n = 100'003;
    constexpr int blocks = 64;

    std::size_t smallest = n;
    std::size_t largest = 0;
    for (int b = 0; b < blocks; ++b) {
        const std::size_t size = block_range(n, blocks, b).size();
        smallest = std::min(smallest, size);
        largest = std::max(largest, size);
    }
    EXPECT_LE(largest - smallest, 1u);
}

}  // namespace
