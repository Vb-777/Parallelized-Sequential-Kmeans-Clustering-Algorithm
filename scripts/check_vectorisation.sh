#!/usr/bin/env bash
#
# Does the distance kernel actually vectorise?
#
#   scripts/check_vectorisation.sh
#
# "I wrote it so it would vectorise" is a hope. This reads the compiler's
# own optimisation report and fails if the inner loop of `assign_tile` -
# the loop that is something like 95% of the runtime - was not vectorised.
# It is wired into CI, so a change that quietly defeats the vectoriser
# (introducing a branch, an aliasing pointer, a call) breaks the build
# rather than halving the throughput unnoticed.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT
cd "${REPO_ROOT}"

CXX="${CXX:-g++}"
# -march=native, because the question is whether the loop vectorises on the
# machine the binary will run on. At the baseline x86-64 ISA (SSE2, two
# doubles per register) GCC's cost model declines the loop, which is a
# correct decision there and tells us nothing about the target.
# An array, so an empty MARCH passes no argument at all rather than an
# empty one, which the compiler would reject.
read -r -a MARCH_FLAGS <<< "${MARCH:--march=native}"
readonly KERNEL="src/kernels.cpp"
readonly MARKER="assign_tile"

WORK="$(mktemp -d)"
readonly WORK
trap 'rm -rf "${WORK}"' EXIT

echo "compiler: $(${CXX} --version | head -1)"
echo "arch:     ${MARCH_FLAGS[*]:-default}"
echo

# -fopt-info-vec is GCC and Clang spelling; the report goes to stderr.
"${CXX}" -std=c++17 -O3 "${MARCH_FLAGS[@]}" -Iinclude -c "${KERNEL}" -o "${WORK}/kernels.o" \
    -fopt-info-vec 2> "${WORK}/report.txt" || {
        echo "compilation failed" >&2
        cat "${WORK}/report.txt" >&2
        exit 1
    }

# The inner loop of assign_tile. Found by line number rather than by name,
# because the report names files and lines, not functions.
kernel_line="$(grep -n 'const Scalar distance = dx \* dx + dy \* dy;' "${KERNEL}" |
               cut -d: -f1)"
if [[ -z "${kernel_line}" ]]; then
    echo "could not locate the distance expression in ${KERNEL}" >&2
    echo "if the kernel was rewritten, update the pattern in this script" >&2
    exit 1
fi

echo "--- vectorisation report ---"
grep 'optimized' "${WORK}/report.txt" || true
echo

# The vectorised loop is reported at the line the loop starts on, a few
# lines above the distance expression.
window_start=$((kernel_line - 8))
vectorised=0
while read -r line_number; do
    if (( line_number >= window_start && line_number <= kernel_line + 4 )); then
        vectorised=1
    fi
done < <(grep 'loop vectorized' "${WORK}/report.txt" |
         sed -n "s|^${KERNEL}:\([0-9]*\):.*|\1|p")

if (( vectorised == 1 )); then
    width="$(grep 'loop vectorized' "${WORK}/report.txt" |
             grep -o '[0-9]* byte vectors' | sort -u | tr '\n' ' ')"
    echo "PASS: the ${MARKER} distance loop vectorised (${width})"
    echo
    echo "note: GCC defaults to 256-bit vectors even on AVX-512 hardware."
    echo "      Configure with -DKMEANS_VECTOR_WIDTH=512 to override, and"
    echo "      measure - it is a win on some parts and a loss on others."
    exit 0
fi

echo "FAIL: the ${MARKER} distance loop (around line ${kernel_line}) was not" >&2
echo "      vectorised. Something in it now defeats the vectoriser - a call," >&2
echo "      a branch the compiler cannot flatten, or a pointer it must" >&2
echo "      assume may alias." >&2
exit 1
