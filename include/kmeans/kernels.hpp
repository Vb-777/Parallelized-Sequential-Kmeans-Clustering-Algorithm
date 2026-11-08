// The compute kernels: one block's worth of work, with no threading in
// sight. Everything here is called from every backend, including MPI.

#ifndef KMEANS_KERNELS_HPP
#define KMEANS_KERNELS_HPP

#include "kmeans/core.hpp"

namespace kmeans {

/// Points processed per tile in the assignment kernel.
///
/// A tile's working set is three arrays of this length - the running best
/// distance, the running best label, and the coordinates being read - which
/// at 2048 points is about 40 KB and stays in L1/L2 while the kernel sweeps
/// the centroids over it.
constexpr std::size_t kTileSize = 2048;

/// Assign every point in [range.begin, range.end) to its nearest centroid,
/// writing labels into `points.label` and the block's partial sums,
/// inertia and farthest-point candidate into `partials`.
///
/// The loop nest is inverted relative to the obvious version. The original
/// held a point and looped over centroids, which makes the inner loop a
/// dependent min-and-argmin over k - a serial chain the compiler cannot
/// vectorise. This holds a centroid and loops over points, so the inner
/// loop is eight independent distance computations and a masked update per
/// AVX-512 register. `scripts/check_vectorisation.sh` reads the compiler's
/// own vectorisation report to confirm it actually happened, rather than
/// assuming it from the source.
void assign_block(PointCloud& points, const Centroids& centroids,
                  Range range, int block, Partials& partials);

/// Sum the per-block partials in block order and divide, writing the new
/// centroids into `next`. Returns the largest squared distance any centroid
/// moved, which is what the convergence test compares against.
///
/// Clusters that ended up empty keep their previous position unless
/// `repair_empty` is called; the count is returned through `empty_clusters`.
Scalar merge_and_update(const Partials& partials, const Centroids& previous,
                        Centroids& next, int& empty_clusters);

/// Move empty clusters onto real points - the points furthest from their
/// own centroids, which is where an extra cluster does most good.
///
/// Lloyd's algorithm has no rule for an empty cluster, and leaving one
/// parked where it was (as the original did) means k-means with k clusters
/// quietly becomes k-means with fewer. Candidates come from the per-block
/// farthest-point records, so the choice depends only on the block
/// decomposition and is identical on every backend.
/// Takes only the partials: the candidate points travel as coordinates, so
/// this works unchanged on a rank that holds none of them.
void repair_empty_clusters(const Partials& partials, Centroids& centroids);

/// Total inertia from the per-block values, summed in block order.
Scalar total_inertia(const Partials& partials);

}  // namespace kmeans

#endif  // KMEANS_KERNELS_HPP
