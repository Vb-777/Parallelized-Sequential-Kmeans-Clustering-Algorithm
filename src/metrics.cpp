#include "kmeans/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace kmeans {
namespace {

/// a(i) and b(i) for one point, against the whole dataset.
///
/// Both are means of the Euclidean distance - not its square. Accumulating
/// squared distances, as the original did, measures something that is not
/// the silhouette coefficient and does not rank clusterings the same way.
struct Cohesion {
    Scalar within = 0;   ///< a(i)
    Scalar nearest = 0;  ///< b(i)
    bool defined = false;
};

Cohesion cohesion_for(const PointCloud& points, int k, std::size_t index) {
    const int own = points.label[index];
    std::vector<Scalar> sums(static_cast<std::size_t>(k), 0);
    std::vector<std::int64_t> counts(static_cast<std::size_t>(k), 0);

    const Scalar px = points.x[index];
    const Scalar py = points.y[index];

    for (std::size_t j = 0; j < points.size(); ++j) {
        const int label = points.label[j];
        if (label < 0 || label >= k) {
            continue;
        }
        const Scalar dx = points.x[j] - px;
        const Scalar dy = points.y[j] - py;
        sums[static_cast<std::size_t>(label)] += std::sqrt(dx * dx + dy * dy);
        counts[static_cast<std::size_t>(label)] += 1;
    }

    Cohesion result;
    const auto own_index = static_cast<std::size_t>(own);

    // The point is counted in its own cluster, at distance zero, so the
    // divisor excludes it. A singleton cluster has no within-cluster
    // distance at all; its silhouette is conventionally 0, and reporting 0
    // is the behaviour this replaces - the original reported 1, the best
    // possible score, for exactly that case.
    const std::int64_t same = counts[own_index] - 1;
    if (same <= 0) {
        return result;  // defined == false: contributes 0
    }
    result.within = sums[own_index] / static_cast<Scalar>(same);

    Scalar nearest = kInfinity;
    for (int c = 0; c < k; ++c) {
        const auto other = static_cast<std::size_t>(c);
        if (c == own || counts[other] == 0) {
            continue;
        }
        nearest = std::min(nearest, sums[other] / static_cast<Scalar>(counts[other]));
    }
    if (nearest == kInfinity) {
        return result;  // only one non-empty cluster: silhouette undefined
    }

    result.nearest = nearest;
    result.defined = true;
    return result;
}

Scalar coefficient(const Cohesion& cohesion) {
    if (!cohesion.defined) {
        return 0;
    }
    const Scalar denominator = std::max(cohesion.within, cohesion.nearest);
    if (denominator <= 0) {
        return 0;
    }
    return (cohesion.nearest - cohesion.within) / denominator;
}

/// Mean of per-point coefficients over `indices`, evaluated on the
/// executor and summed in block order so the answer does not depend on the
/// thread count.
Scalar mean_coefficient(const PointCloud& points, int k,
                        const std::vector<std::size_t>& indices,
                        Executor& executor) {
    if (indices.empty()) {
        return 0;
    }

    constexpr int kBlocks = 64;
    std::vector<Scalar> block_sum(kBlocks, 0);

    executor.for_each_block(kBlocks, [&](int b) {
        const Range range = block_range(indices.size(), kBlocks, b);
        Scalar sum = 0;
        for (std::size_t i = range.begin; i < range.end; ++i) {
            sum += coefficient(cohesion_for(points, k, indices[i]));
        }
        block_sum[static_cast<std::size_t>(b)] = sum;
    });

    Scalar total = 0;
    for (int b = 0; b < kBlocks; ++b) {
        total += block_sum[static_cast<std::size_t>(b)];
    }
    return total / static_cast<Scalar>(indices.size());
}

}  // namespace

Scalar inertia(const PointCloud& points, const Centroids& centroids) {
    Scalar total = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const int label = points.label[i];
        if (label < 0) {
            continue;
        }
        const auto cluster = static_cast<std::size_t>(label);
        const Scalar dx = points.x[i] - centroids.x[cluster];
        const Scalar dy = points.y[i] - centroids.y[cluster];
        total += dx * dx + dy * dy;
    }
    return total;
}

Scalar davies_bouldin(const PointCloud& points, const Centroids& centroids) {
    const auto k = static_cast<int>(centroids.size());
    if (k < 2) {
        return 0;
    }

    std::vector<Scalar> spread(static_cast<std::size_t>(k), 0);
    std::vector<std::int64_t> counts(static_cast<std::size_t>(k), 0);

    for (std::size_t i = 0; i < points.size(); ++i) {
        const int label = points.label[i];
        if (label < 0 || label >= k) {
            continue;
        }
        const auto cluster = static_cast<std::size_t>(label);
        const Scalar dx = points.x[i] - centroids.x[cluster];
        const Scalar dy = points.y[i] - centroids.y[cluster];
        spread[cluster] += std::sqrt(dx * dx + dy * dy);
        counts[cluster] += 1;
    }
    for (int c = 0; c < k; ++c) {
        const auto cluster = static_cast<std::size_t>(c);
        if (counts[cluster] > 0) {
            spread[cluster] /= static_cast<Scalar>(counts[cluster]);
        }
    }

    Scalar total = 0;
    int considered = 0;
    for (int i = 0; i < k; ++i) {
        const auto left = static_cast<std::size_t>(i);
        if (counts[left] == 0) {
            continue;
        }
        Scalar worst = 0;
        for (int j = 0; j < k; ++j) {
            const auto right = static_cast<std::size_t>(j);
            if (i == j || counts[right] == 0) {
                continue;
            }
            const Scalar dx = centroids.x[left] - centroids.x[right];
            const Scalar dy = centroids.y[left] - centroids.y[right];
            const Scalar separation = std::sqrt(dx * dx + dy * dy);
            if (separation <= 0) {
                continue;  // coincident centroids carry no information
            }
            worst = std::max(worst, (spread[left] + spread[right]) / separation);
        }
        total += worst;
        ++considered;
    }
    return considered > 0 ? total / static_cast<Scalar>(considered) : 0;
}

Scalar silhouette_exact(const PointCloud& points, int k, Executor& executor) {
    std::vector<std::size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    return mean_coefficient(points, k, indices, executor);
}

Scalar silhouette_sampled(const PointCloud& points, int k, std::size_t samples,
                          std::uint64_t seed, Executor& executor) {
    const std::size_t n = points.size();
    if (n == 0) {
        return 0;
    }
    if (samples >= n) {
        return silhouette_exact(points, k, executor);
    }

    // Sampling without replacement: a point drawn twice would be weighted
    // twice, which biases the estimate on small samples.
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), std::size_t{0});
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < samples; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, n - 1);
        std::swap(all[i], all[pick(rng)]);
    }
    all.resize(samples);
    // Sorted, so the summation order - and the answer - depends only on
    // the seed and the sample size.
    std::sort(all.begin(), all.end());

    return mean_coefficient(points, k, all, executor);
}

}  // namespace kmeans
