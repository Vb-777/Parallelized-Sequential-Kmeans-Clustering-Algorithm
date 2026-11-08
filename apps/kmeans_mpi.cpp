// Distributed k-means over MPI.
//
// Each rank owns a contiguous span of the fixed block decomposition that
// the shared-memory backends already use, so the parallel structure is the
// same one level up: a rank computes partial sums for its blocks, the ranks
// all-reduce the per-block arrays, and every rank merges them in block
// order. Because each block is owned by exactly one rank, the all-reduce
// adds one real value and p-1 exact zeros per cell, so the sum is the
// owner's value bit for bit. The distributed result is therefore identical
// to the serial one, not merely close - `--self-check` asserts exactly that,
// and CI runs it on three ranks.
//
// Two honest notes about scope.
//
// Data distribution is not implemented. Every rank generates the same
// dataset from the same seed and then keeps only its own span, freeing the
// rest. That makes generation O(n) per rank rather than O(n/p), and a real
// deployment would read a partitioned file instead. It is done this way so
// that the distributed run and the shared-memory run operate on provably
// identical data, which is what makes the bitwise comparison meaningful.
//
// Communication is one all-reduce of blocks*k cells per iteration, which is
// the right shape but is not latency-optimised: at small k the message is
// tiny and the cost is dominated by the collective's latency, so the
// distributed version only pays off when n per rank is large. The scaling
// numbers in README.md say where that crossover sits on the machine they
// were taken on.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kernels.hpp"
#include "kmeans/kmeans.hpp"
#include "kmeans/timer.hpp"

namespace {

struct Options {
    std::size_t n = 1'000'000;
    int k = 20;
    int iterations = 50;
    int blocks = 64;
    std::uint64_t seed = 42;
    kmeans::Scalar extent = 1000;
    kmeans::Scalar spread = 20;
    bool self_check = false;
};

const char* value_for(int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        std::cerr << argv[i] << " needs a value\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return argv[++i];
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--n") {
            options.n = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--k") {
            options.k = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--iterations") {
            options.iterations = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--blocks") {
            options.blocks = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--seed") {
            options.seed = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--self-check") {
            // A smaller problem, so the check is quick enough for CI.
            options.self_check = true;
            options.n = 200'000;
            options.k = 12;
            options.iterations = 25;
        } else if (flag == "--help" || flag == "-h") {
            std::cout << "usage: mpiexec -n P kmeans_mpi [--n N] [--k K] "
                         "[--iterations N] [--blocks N] [--seed S] [--self-check]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown option '" << flag << "'\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    return options;
}

/// Blocks [first, last) owned by `rank` of `size`.
struct BlockSpan {
    int first = 0;
    int last = 0;
};

BlockSpan blocks_for(int blocks, int rank, int size) {
    return BlockSpan{blocks * rank / size, blocks * (rank + 1) / size};
}

/// Sum every per-block accumulator across the ranks.
///
/// Summation is exact here rather than merely accurate: for each cell,
/// exactly one rank holds a value and every other contributes a literal
/// zero, so no rounding can occur however the collective orders its
/// additions. That is what lets the distributed result be compared bitwise
/// against the serial one.
void all_reduce(kmeans::Partials& partials) {
    MPI_Allreduce(MPI_IN_PLACE, partials.sum_x.data(),
                  static_cast<int>(partials.sum_x.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.sum_y.data(),
                  static_cast<int>(partials.sum_y.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.count.data(),
                  static_cast<int>(partials.count.size()), MPI_INT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.inertia.data(),
                  static_cast<int>(partials.inertia.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.farthest_distance.data(),
                  static_cast<int>(partials.farthest_distance.size()), MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.farthest_x.data(),
                  static_cast<int>(partials.farthest_x.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partials.farthest_y.data(),
                  static_cast<int>(partials.farthest_y.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
}

/// Distributed k-means++ seeding.
///
/// Identical in structure to the shared-memory version: carry the running
/// nearest-centroid distance, all-reduce the per-block weights, and have
/// every rank draw the same number from the same seeded generator so they
/// all agree on which block the sample fell in. Only the rank owning that
/// block can scan it, so it finds the point and broadcasts its coordinates.
kmeans::Centroids seed_distributed(const kmeans::PointCloud& local,
                                   std::size_t offset, const Options& options,
                                   BlockSpan span) {
    const std::size_t n = options.n;
    const int blocks = options.blocks;

    kmeans::Centroids centroids;
    centroids.resize(static_cast<std::size_t>(options.k));

    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<std::size_t> any_point(0, n - 1);

    // Every rank runs the same generator, so they agree without talking.
    const std::size_t first = any_point(rng);
    {
        kmeans::Scalar coordinates[2] = {0, 0};
        const int owner_block = [&] {
            for (int b = 0; b < blocks; ++b) {
                const kmeans::Range range = kmeans::block_range(n, blocks, b);
                if (first >= range.begin && first < range.end) {
                    return b;
                }
            }
            return blocks - 1;
        }();
        if (owner_block >= span.first && owner_block < span.last) {
            coordinates[0] = local.x[first - offset];
            coordinates[1] = local.y[first - offset];
        }
        MPI_Allreduce(MPI_IN_PLACE, coordinates, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        centroids.x[0] = coordinates[0];
        centroids.y[0] = coordinates[1];
    }

    std::vector<kmeans::Scalar> nearest(local.size(), kmeans::kInfinity);
    std::vector<kmeans::Scalar> block_sum(static_cast<std::size_t>(blocks), 0);
    std::uniform_real_distribution<kmeans::Scalar> unit(0, 1);

    for (int c = 1; c < options.k; ++c) {
        const kmeans::Scalar cx = centroids.x[static_cast<std::size_t>(c - 1)];
        const kmeans::Scalar cy = centroids.y[static_cast<std::size_t>(c - 1)];

        std::fill(block_sum.begin(), block_sum.end(), kmeans::Scalar{0});
        for (int b = span.first; b < span.last; ++b) {
            const kmeans::Range range = kmeans::block_range(n, blocks, b);
            kmeans::Scalar sum = 0;
            for (std::size_t i = range.begin; i < range.end; ++i) {
                const std::size_t local_index = i - offset;
                const kmeans::Scalar dx = local.x[local_index] - cx;
                const kmeans::Scalar dy = local.y[local_index] - cy;
                const kmeans::Scalar distance = dx * dx + dy * dy;
                if (distance < nearest[local_index]) {
                    nearest[local_index] = distance;
                }
                sum += nearest[local_index];
            }
            block_sum[static_cast<std::size_t>(b)] = sum;
        }
        MPI_Allreduce(MPI_IN_PLACE, block_sum.data(), blocks, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        kmeans::Scalar total = 0;
        for (int b = 0; b < blocks; ++b) {
            total += block_sum[static_cast<std::size_t>(b)];
        }

        kmeans::Scalar coordinates[2] = {0, 0};
        if (!(total > 0)) {
            const std::size_t chosen = any_point(rng);
            const int owner_block = [&] {
                for (int b = 0; b < blocks; ++b) {
                    const kmeans::Range range = kmeans::block_range(n, blocks, b);
                    if (chosen >= range.begin && chosen < range.end) {
                        return b;
                    }
                }
                return blocks - 1;
            }();
            if (owner_block >= span.first && owner_block < span.last) {
                coordinates[0] = local.x[chosen - offset];
                coordinates[1] = local.y[chosen - offset];
            }
        } else {
            kmeans::Scalar target = unit(rng) * total;
            int block = blocks - 1;
            for (int b = 0; b < blocks; ++b) {
                const kmeans::Scalar weight = block_sum[static_cast<std::size_t>(b)];
                if (target < weight) {
                    block = b;
                    break;
                }
                target -= weight;
            }

            if (block >= span.first && block < span.last) {
                const kmeans::Range range = kmeans::block_range(n, blocks, block);
                std::size_t chosen = range.end > range.begin ? range.end - 1 : range.begin;
                for (std::size_t i = range.begin; i < range.end; ++i) {
                    target -= nearest[i - offset];
                    if (target <= 0) {
                        chosen = i;
                        break;
                    }
                }
                coordinates[0] = local.x[chosen - offset];
                coordinates[1] = local.y[chosen - offset];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, coordinates, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        centroids.x[static_cast<std::size_t>(c)] = coordinates[0];
        centroids.y[static_cast<std::size_t>(c)] = coordinates[1];
    }

    return centroids;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Options options = parse(argc, argv);

    if (options.blocks < size) {
        if (rank == 0) {
            std::cerr << "--blocks (" << options.blocks << ") must be at least the "
                      << "number of ranks (" << size << ")\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const BlockSpan span = blocks_for(options.blocks, rank, size);
    const kmeans::Range owned_begin =
        kmeans::block_range(options.n, options.blocks, span.first);
    const kmeans::Range owned_end =
        kmeans::block_range(options.n, options.blocks, span.last - 1);
    const std::size_t offset = owned_begin.begin;

    // Generate the whole dataset, keep this rank's span, release the rest.
    // See the note at the top of the file.
    kmeans::PointCloud local;
    {
        kmeans::Dataset whole = kmeans::make_blobs(options.n, options.k,
                                                   options.extent, options.spread,
                                                   options.seed);
        const std::size_t count = owned_end.end - offset;
        local.resize(count);
        std::copy(whole.points.x.begin() + static_cast<std::ptrdiff_t>(offset),
                  whole.points.x.begin() +
                      static_cast<std::ptrdiff_t>(offset + count),
                  local.x.begin());
        std::copy(whole.points.y.begin() + static_cast<std::ptrdiff_t>(offset),
                  whole.points.y.begin() +
                      static_cast<std::ptrdiff_t>(offset + count),
                  local.y.begin());
    }

    kmeans::Timer timer;

    kmeans::Centroids centroids =
        seed_distributed(local, offset, options, span);
    const double seed_seconds = timer.wall_seconds();

    kmeans::Partials partials;
    partials.reset(options.blocks, options.k);
    kmeans::Centroids next;
    int iterations = 0;

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        partials.clear();

        for (int b = span.first; b < span.last; ++b) {
            const kmeans::Range global = kmeans::block_range(options.n, options.blocks, b);
            const kmeans::Range local_range{global.begin - offset, global.end - offset};
            kmeans::assign_block(local, centroids, local_range, b, partials);
        }
        all_reduce(partials);

        int empty_clusters = 0;
        kmeans::merge_and_update(partials, centroids, next, empty_clusters);
        if (empty_clusters > 0) {
            kmeans::repair_empty_clusters(partials, next);
        }
        centroids = next;
        iterations = iteration + 1;
    }

    // Final assignment, so the reported inertia belongs to these centroids.
    partials.clear();
    for (int b = span.first; b < span.last; ++b) {
        const kmeans::Range global = kmeans::block_range(options.n, options.blocks, b);
        const kmeans::Range local_range{global.begin - offset, global.end - offset};
        kmeans::assign_block(local, centroids, local_range, b, partials);
    }
    all_reduce(partials);
    const kmeans::Scalar inertia = kmeans::total_inertia(partials);

    const double wall = timer.wall_seconds();
    double slowest = wall;
    MPI_Reduce(&wall, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = 0;

    if (options.self_check) {
        // Rank 0 reruns the whole problem serially, in this process, and
        // compares. Bitwise: anything less would not distinguish "the same
        // algorithm" from "a similar answer".
        if (rank == 0) {
            kmeans::Dataset whole = kmeans::make_blobs(options.n, options.k,
                                                       options.extent,
                                                       options.spread, options.seed);
            kmeans::Params params;
            params.k = options.k;
            params.max_iterations = options.iterations;
            params.tolerance = 0;  // run every iteration, as the MPI loop does
            params.seed = options.seed;
            params.blocks = options.blocks;

            const kmeans::Result reference =
                kmeans::run(whole.points, params, kmeans::Backend::Serial);

            bool identical = reference.centroids.size() == centroids.size();
            for (std::size_t c = 0; identical && c < centroids.size(); ++c) {
                identical = reference.centroids.x[c] == centroids.x[c] &&
                            reference.centroids.y[c] == centroids.y[c];
            }
            identical = identical && reference.inertia == inertia;

            std::cout << std::setprecision(17)
                      << "ranks:            " << size << '\n'
                      << "serial inertia:   " << reference.inertia << '\n'
                      << "mpi inertia:      " << inertia << '\n'
                      << (identical
                              ? "self-check: PASS - centroids and inertia are "
                                "bitwise identical to the serial run\n"
                              : "self-check: FAIL - the distributed result "
                                "differs from the serial one\n");
            status = identical ? 0 : 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        std::cout << std::fixed << std::setprecision(4)
                  << "ranks:       " << size << '\n'
                  << "points:      " << options.n << '\n'
                  << "clusters:    " << options.k << '\n'
                  << "blocks:      " << options.blocks << '\n'
                  << "iterations:  " << iterations << '\n'
                  << "seeding:     " << seed_seconds << " s\n"
                  << "wall:        " << slowest << " s (slowest rank)\n"
                  << "inertia:     " << inertia << '\n';
    }

    MPI_Finalize();
    return status;
}
