#include "kmeans/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace kmeans {
namespace {

/// One tile: find the nearest centroid for each of its points.
///
/// Written so that the inner loops carry no dependence across i. The
/// branchless select is deliberate - a conditional store would leave the
/// compiler emitting a scalar loop with an unpredictable branch, which on
/// uniform data mispredicts roughly half the time.
void assign_tile(const Scalar* __restrict px, const Scalar* __restrict py,
                 std::size_t count, const Centroids& centroids,
                 Scalar* __restrict best, Label* __restrict best_label) {
    const int k = static_cast<int>(centroids.size());

    for (std::size_t i = 0; i < count; ++i) {
        best[i] = kInfinity;
        best_label[i] = 0;
    }

    for (int c = 0; c < k; ++c) {
        const Scalar cx = centroids.x[static_cast<std::size_t>(c)];
        const Scalar cy = centroids.y[static_cast<std::size_t>(c)];

        for (std::size_t i = 0; i < count; ++i) {
            // dx*dx, not pow(dx, 2). At -O0 - which is how the original
            // baseline was built - pow is a real libm call, and that single
            // difference accounts for most of the 17x the baseline left on
            // the table.
            const Scalar dx = px[i] - cx;
            const Scalar dy = py[i] - cy;
            const Scalar distance = dx * dx + dy * dy;

            const bool closer = distance < best[i];
            best[i] = closer ? distance : best[i];
            best_label[i] = closer ? static_cast<Label>(c) : best_label[i];
        }
    }
}

}  // namespace

void assign_block(PointCloud& points, const Centroids& centroids,
                  Range range, int block, Partials& partials) {
    Scalar* const sum_x = partials.sum_x.data() + partials.cell(block, 0);
    Scalar* const sum_y = partials.sum_y.data() + partials.cell(block, 0);
    std::int64_t* const count = partials.count.data() + partials.cell(block, 0);

    Scalar block_inertia = 0;
    Scalar farthest = -1;
    Scalar farthest_x = range.begin < points.size() ? points.x[range.begin] : 0;
    Scalar farthest_y = range.begin < points.size() ? points.y[range.begin] : 0;

    std::array<Scalar, kTileSize> best{};
    std::array<Label, kTileSize> best_label{};

    for (std::size_t start = range.begin; start < range.end; start += kTileSize) {
        const std::size_t count_in_tile = std::min(kTileSize, range.end - start);

        assign_tile(points.x.data() + start, points.y.data() + start,
                    count_in_tile, centroids, best.data(), best_label.data());

        // Scatter phase. Indexed writes into k accumulators, so this does
        // not vectorise - but it is one pass over the tile with k-sized
        // working set, against the k passes the assignment just did.
        for (std::size_t i = 0; i < count_in_tile; ++i) {
            const std::size_t index = start + i;
            const Label label = best_label[i];

            points.label[index] = label;
            sum_x[label] += points.x[index];
            sum_y[label] += points.y[index];
            count[label] += 1;
            block_inertia += best[i];

            if (best[i] > farthest) {
                farthest = best[i];
                farthest_x = points.x[index];
                farthest_y = points.y[index];
            }
        }
    }

    partials.inertia[static_cast<std::size_t>(block)] = block_inertia;
    partials.farthest_distance[static_cast<std::size_t>(block)] =
        farthest < 0 ? 0 : farthest;
    partials.farthest_x[static_cast<std::size_t>(block)] = farthest_x;
    partials.farthest_y[static_cast<std::size_t>(block)] = farthest_y;
}

Scalar merge_and_update(const Partials& partials, const Centroids& previous,
                        Centroids& next, int& empty_clusters) {
    const int k = partials.k;
    next.resize(static_cast<std::size_t>(k));
    empty_clusters = 0;

    Scalar largest_shift = 0;

    for (int c = 0; c < k; ++c) {
        Scalar sum_x = 0;
        Scalar sum_y = 0;
        std::int64_t count = 0;

        // Ascending block order, always. This is the line that makes the
        // result independent of the thread count: the additions happen in
        // this sequence whoever computed the terms.
        for (int b = 0; b < partials.blocks; ++b) {
            const std::size_t cell = partials.cell(b, c);
            sum_x += partials.sum_x[cell];
            sum_y += partials.sum_y[cell];
            count += partials.count[cell];
        }

        const auto cluster = static_cast<std::size_t>(c);
        if (count == 0) {
            ++empty_clusters;
            next.x[cluster] = previous.x[cluster];
            next.y[cluster] = previous.y[cluster];
            continue;
        }

        const auto divisor = static_cast<Scalar>(count);
        next.x[cluster] = sum_x / divisor;
        next.y[cluster] = sum_y / divisor;

        const Scalar dx = next.x[cluster] - previous.x[cluster];
        const Scalar dy = next.y[cluster] - previous.y[cluster];
        largest_shift = std::max(largest_shift, dx * dx + dy * dy);
    }

    return largest_shift;
}

void repair_empty_clusters(const Partials& partials, Centroids& centroids) {
    const int k = partials.k;

    // Candidates: each block's farthest point, strongest first. Ties break
    // on the block index, so the choice is a function of the data and the
    // block decomposition - never of which thread finished first.
    std::vector<int> order(static_cast<std::size_t>(partials.blocks));
    for (int b = 0; b < partials.blocks; ++b) {
        order[static_cast<std::size_t>(b)] = b;
    }
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return partials.farthest_distance[static_cast<std::size_t>(lhs)] >
               partials.farthest_distance[static_cast<std::size_t>(rhs)];
    });

    std::size_t next_candidate = 0;
    for (int c = 0; c < k; ++c) {
        std::int64_t count = 0;
        for (int b = 0; b < partials.blocks; ++b) {
            count += partials.count[partials.cell(b, c)];
        }
        if (count != 0) {
            continue;
        }
        if (next_candidate >= order.size()) {
            // More empty clusters than blocks. The rest keep their
            // positions and get another chance next iteration; with the
            // default 64 blocks this needs a badly chosen k to happen.
            break;
        }

        const int block = order[next_candidate++];
        const auto source = static_cast<std::size_t>(block);
        centroids.x[static_cast<std::size_t>(c)] = partials.farthest_x[source];
        centroids.y[static_cast<std::size_t>(c)] = partials.farthest_y[source];
    }
}

Scalar total_inertia(const Partials& partials) {
    Scalar total = 0;
    for (int b = 0; b < partials.blocks; ++b) {
        total += partials.inertia[static_cast<std::size_t>(b)];
    }
    return total;
}

}  // namespace kmeans
