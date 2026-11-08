#!/usr/bin/env bash
#
# Configure, build, test, measure, plot - one command.
#
#   scripts/run_experiments.sh [--quick]
#
# Everything in README.md's results section comes from this script, so a
# reader can regenerate the numbers rather than take them on trust. That is
# the point: v1.0's tables were produced by editing constants in a source
# file, running it by hand and pasting the output into a spreadsheet, which
# is why nobody noticed that the timer was wrong.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
cd "${REPO_ROOT}"

QUICK=0
for argument in "$@"; do
    case "${argument}" in
        --quick) QUICK=1 ;;
        -h|--help)
            echo "usage: $0 [--quick]"
            echo "  --quick   smaller problem and fewer repeats, for a smoke test"
            exit 0
            ;;
        *) echo "unknown argument '${argument}'" >&2; exit 2 ;;
    esac
done

readonly BUILD_DIR=build
readonly RESULTS_DIR=results
readonly GRAPHS_DIR=graphs

if (( QUICK )); then
    readonly POINTS=250000
    readonly REPEATS=3
    readonly ITERATIONS=10
else
    readonly POINTS=2000000
    readonly REPEATS=9
    readonly ITERATIONS=30
fi

# Thread counts up to twice the core count, so the chart shows what happens
# when threads start sharing cores rather than stopping at the flattering
# point.
CORES="$(nproc)"
readonly CORES
thread_list="1"
threads=2
while (( threads <= CORES * 2 )); do
    thread_list="${thread_list},${threads}"
    threads=$(( threads * 2 ))
done
readonly thread_list

echo "cores:   ${CORES}"
echo "threads: ${thread_list}"
echo "points:  ${POINTS}   repeats: ${REPEATS}   iterations: ${ITERATIONS}"
echo

# ------------------------------------------------------------- 1. build

echo "=== configure and build ==="
cmake -S . -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DKMEANS_NATIVE_ARCH=ON \
      -DKMEANS_WERROR=ON
cmake --build "${BUILD_DIR}" --parallel "${CORES}"
echo

# -------------------------------------------------------------- 2. test

echo "=== tests ==="
ctest --test-dir "${BUILD_DIR}" --output-on-failure --parallel "${CORES}"
echo

echo "=== vectorisation ==="
./scripts/check_vectorisation.sh
echo

# ----------------------------------------------------------- 3. measure

mkdir -p "${RESULTS_DIR}" "${GRAPHS_DIR}"

backends="pthreads"
if "${BUILD_DIR}/kmeans" --version | grep -q "openmp:     yes"; then
    backends="pthreads,openmp"
fi
readonly backends

echo "=== strong scaling ==="
"${BUILD_DIR}/kmeans_bench" \
    --mode strong --n "${POINTS}" --k 20 --iterations "${ITERATIONS}" \
    --threads "${thread_list}" --repeats "${REPEATS}" --backends "${backends}" \
    --csv "${RESULTS_DIR}/strong_scaling.csv" |
    tee "${RESULTS_DIR}/strong_scaling.txt"
echo

echo "=== weak scaling ==="
"${BUILD_DIR}/kmeans_bench" \
    --mode weak --n $(( POINTS / 2 )) --k 20 --iterations "${ITERATIONS}" \
    --threads "1,${CORES}" --repeats "${REPEATS}" --backends pthreads \
    --csv "${RESULTS_DIR}/weak_scaling.csv" |
    tee "${RESULTS_DIR}/weak_scaling.txt"
echo

if command -v mpiexec > /dev/null && [[ -x "${BUILD_DIR}/kmeans_mpi" ]]; then
    echo "=== mpi ==="
    mpiexec -n "${CORES}" "${BUILD_DIR}/kmeans_mpi" \
        --n "${POINTS}" --k 20 --iterations "${ITERATIONS}" |
        tee "${RESULTS_DIR}/mpi.txt"
    echo
fi

# -------------------------------------------------------------- 4. plot

if python3 -c "import matplotlib" 2> /dev/null; then
    echo "=== plots ==="
    python3 scripts/plot_results.py "${RESULTS_DIR}/strong_scaling.csv" \
            --weak "${RESULTS_DIR}/weak_scaling.csv" --out "${GRAPHS_DIR}"
else
    echo "matplotlib is not installed; skipping the plots" >&2
    echo "  pip install -r requirements.txt" >&2
fi

echo
echo "results in ${RESULTS_DIR}/, charts in ${GRAPHS_DIR}/"
