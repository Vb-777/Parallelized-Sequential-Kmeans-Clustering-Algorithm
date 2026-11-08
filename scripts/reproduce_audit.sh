#!/usr/bin/env bash
#
# Reproduces the three measurements that reframed this project, using the
# original sources in legacy/ exactly as they were committed in v1.0.
#
#   scripts/reproduce_audit.sh [output-directory]
#
# 1. -O0 versus -O2 on the sequential program. The recorded 31.9 s baseline
#    was an unoptimised build; optimising it is worth more than the
#    parallelisation was.
# 2. clock() versus wall-clock on the parallel program. clock() accumulates
#    CPU time across threads, so it reports roughly constant time no matter
#    how many threads run - it cannot produce the recorded speedup curve.
# 3. The silhouette output: every score is exactly 1.
#
# The legacy programs hardcode their problem size, so the script copies them
# to a scratch directory and rewrites exactly three integer literals to put
# both programs on the same configuration. Nothing else is touched; the diff
# is printed so you can check that.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
cd "${REPO_ROOT}"

OUT_DIR="${1:-audit}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
readonly OUT_DIR

WORK="$(mktemp -d)"
readonly WORK
trap 'rm -rf "${WORK}"' EXIT

# The configuration the CV's headline number was recorded at.
readonly N=500000
readonly K=20
readonly ITERATIONS=100

echo "reproducing the audit at n=${N}, k=${K}, ${ITERATIONS} iterations"
echo "cores available: $(nproc)"
echo

# --------------------------------------------------------------- patching

# A fixed seed, so the two builds cluster identical data and the comparison
# is of the build and nothing else.
patch_common() {
  sed -e 's/srand(time(0));/srand(12345);/' "$1"
}

patch_common legacy/original_sequential.cpp |
  sed -e "s/int dataSetSize = 100000;/int dataSetSize = ${N};/" \
      -e "s/int numClusters = 20;/int numClusters = ${K};/" \
      > "${WORK}/sequential.cpp"

patch_common legacy/original_parallel.cpp |
  sed -e "s/int dataSetSize = 500000;/int dataSetSize = ${N};/" \
      -e "s/int numClusters = 100;/int numClusters = ${K};/" \
      > "${WORK}/parallel.cpp"

echo "--- the only changes made to the committed sources ---"
diff -u legacy/original_sequential.cpp "${WORK}/sequential.cpp" || true
diff -u legacy/original_parallel.cpp "${WORK}/parallel.cpp" || true
echo

# ------------------------------------------------- 1. the -O0 measurement

echo "=== 1. the baseline was unoptimised ==="
{
  echo "build,wall_seconds"
} > "${OUT_DIR}/legacy_optimisation.csv"

for opt in -O0 -O2 -O3; do
  g++ "${opt}" -std=c++17 "${WORK}/sequential.cpp" -o "${WORK}/seq${opt}"
  # /usr/bin/time is not assumed; bash's SECONDS has 1 s resolution, so the
  # timing is taken from the shell's own high-resolution clock.
  start=$(date +%s.%N)
  (cd "${WORK}" && "./seq${opt}" > /dev/null)
  end=$(date +%s.%N)
  wall=$(python3 -c "print(f'{${end} - ${start}:.3f}')")
  printf '  %-4s  %8s s\n' "${opt}" "${wall}"
  echo "${opt},${wall}" >> "${OUT_DIR}/legacy_optimisation.csv"
done
echo

# ------------------------------------- 2. clock() cannot measure speedup

echo "=== 2. clock() measures CPU time, not elapsed time ==="
g++ -O2 -std=c++17 -pthread "${WORK}/parallel.cpp" -o "${WORK}/par"

# The committed program loops threads 2,4,...,32 internally and prints
# clock()-derived times. Running it once gives the whole curve, and the
# wall-clock total is measured here for contrast.
start=$(date +%s.%N)
(cd "${WORK}" && ./par | tee "${OUT_DIR}/legacy_parallel_clock.txt")
end=$(date +%s.%N)
total=$(python3 -c "print(f'{${end} - ${start}:.3f}')")

echo
echo "  the program's own clock()-derived times are above"
echo "  measured wall-clock for the whole sweep: ${total} s"
echo "  on a ${NPROC:-$(nproc)}-core machine, clock() reports roughly"
echo "  threads x wall, so the reported time cannot fall as threads rise"
echo

# -------------------------------------------- 3. every silhouette is 1

echo "=== 3. the silhouette output carries no information ==="
# The sequential program writes silhouette_scoreskmN1l.csv next to itself.
SILHOUETTE="${WORK}/silhouette_scoreskmN1l.csv"
if [[ -f "${SILHOUETTE}" ]]; then
  cp "${SILHOUETTE}" "${OUT_DIR}/legacy_silhouette.csv"
  rows=$(( $(wc -l < "${SILHOUETTE}") - 1 ))
  distinct=$(tail -n +2 "${SILHOUETTE}" | cut -d, -f4 | sort -u | tr '\n' ' ')
  echo "  rows:            ${rows}"
  echo "  distinct values: ${distinct}"
  echo
  echo "  silhouetteScore({point}, centroids) is called with a one-point"
  echo "  vector, so the same-cluster average a is 0/1 = 0, every other"
  echo "  cluster is empty, b stays at DBL_MAX, and (b-a)/max(a,b) is 1."
else
  echo "  silhouette file not produced; skipping" >&2
fi

echo
echo "results written to ${OUT_DIR}/"

# ------------------------------------------- 4. the seeding was O(n * k^2)

echo "=== 4. k-means++ seeding was quadratic in k ==="
# The committed initializeCentroids recomputes the distance from every point
# to every already-chosen centroid each time it picks a new one. Carrying a
# running nearest-centroid distance forward - the standard formulation -
# makes it O(n*k). Both are timed here: the legacy function is called
# directly out of the committed source, with its main() renamed away so the
# file can be linked against a harness.
cat > "${WORK}/seed_bench.cpp" <<'CPP'
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Point { double x, y; int cluster; };
struct Centroid { double x, y; };
void initializeCentroids(std::vector<Point>&, std::vector<Centroid>&);
int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 500000;
    const int k = argc > 2 ? std::atoi(argv[2]) : 100;
    std::srand(12345);
    std::vector<Point> points(static_cast<std::size_t>(n));
    for (auto& p : points) { p.x = std::rand() % 1000; p.y = std::rand() % 1000; }
    std::vector<Centroid> centroids(static_cast<std::size_t>(k));
    const auto start = std::chrono::steady_clock::now();
    initializeCentroids(points, centroids);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    std::printf("%.4f\n", elapsed.count());
    return 0;
}
CPP

readonly SEED_N=500000
echo "implementation,k,seconds" > "${OUT_DIR}/seeding.csv"
for seed_k in 20 100; do
  g++ -O2 -std=c++17 -Dmain=legacy_main_unused -c "${WORK}/sequential.cpp" \
      -o "${WORK}/legacy_seed.o"
  g++ -O2 -std=c++17 "${WORK}/seed_bench.cpp" "${WORK}/legacy_seed.o" \
      -o "${WORK}/seed_bench"
  legacy_time="$("${WORK}/seed_bench" "${SEED_N}" "${seed_k}")"
  printf '  k=%-4s legacy O(n*k^2) seeding: %8s s\n' "${seed_k}" "${legacy_time}"
  echo "legacy,${seed_k},${legacy_time}" >> "${OUT_DIR}/seeding.csv"
done

if [[ -x build/kmeans ]]; then
  for seed_k in 20 100; do
    rewritten="$(./build/kmeans --n "${SEED_N}" --k "${seed_k}" --dataset uniform \
                 --iterations 1 2>/dev/null | awk '/^seeding:/ {print $2}')"
    printf '  k=%-4s rewritten O(n*k) seeding: %8s s\n' "${seed_k}" "${rewritten}"
    echo "rewritten,${seed_k},${rewritten}" >> "${OUT_DIR}/seeding.csv"
  done
else
  echo "  (build/kmeans not present; build it to compare the rewritten seeding)"
fi
