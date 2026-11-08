# What was wrong with v1.0, and how it was found

This is the audit that prompted the rewrite. Everything here was measured on
a 2-core Intel Xeon (AVX-512) running Linux, using the v1.0 sources exactly
as they were committed — they are preserved in [`legacy/`](legacy/), and
[`scripts/reproduce_audit.sh`](scripts/reproduce_audit.sh) regenerates every
number below.

Two things worth saying first, because the headline finding is unflattering
and the context matters.

**The parallelisation itself was sound.** Partitioned assignment with
thread-local accumulation is the textbook design, and the k-means++ seeding
was real D²-weighted sampling, not random initialisation under a better
name. The rewrite changed how the work is scheduled and measured; it did not
change the strategy, because the strategy was right.

**The recorded numbers were transcribed faithfully.** 31.915 / 6.105 = 5.23×
and 153.48 / 20.589 = 7.45× are exactly what the committed tables say, and
the two tables agree to within 1% where they overlap. Nothing was invented.
The problem is not honesty; it is that the measurements could not support
what was concluded from them.

---

## 1. The baseline was compiled without optimisation

The finding that reframes the project.

The v1.0 sequential program, n = 500,000, k = 20, 100 iterations:

| Build | Wall time |
|---|---:|
| `-O0` | **69.69 s** |
| `-O2` | 5.07 s |
| `-O3` | 4.06 s |

**A compiler flag is worth 17.2×.** The recorded baseline of 31.9 s sits in
`-O0` territory; `-O2` on the same code and the same machine would have been
around 5 s.

The single largest contributor is `pow(dx, 2)`. At `-O0` that is a genuine
call into libm for every coordinate of every point against every centroid; at
`-O2` the compiler recognises the constant exponent and emits a multiply.
Writing `dx * dx` makes the two builds agree.

So the 5.2× speedup was real — *at `-O0`*. It was parallelism measured
against a serial baseline that was roughly seventeen times slower than it
needed to be. Optimising the serial code first is the classic first rule of
performance work, and it would have returned more than the parallelisation
did.

## 2. `clock()` cannot measure a speedup

`clock()` returns processor time summed over every thread, not elapsed time.
On *p* busy threads it reports approximately *p* × wall, so a program that
parallelises perfectly appears to take constant time.

Running the committed parallel program unchanged, on 2 cores, printing its
own `clock()`-derived numbers:

| Threads | v1.0 reports |
|---:|---:|
| 2 | 3.14 s |
| 4 | 3.23 s |
| 8 | 3.40 s |
| 16 | 3.34 s |
| 32 | 3.59 s |

The reported time **rises** with the thread count. The committed program
cannot produce the committed speedup table — not approximately, not on a
larger machine, not at all. Whatever produced those tables, it was not this
code path.

The rewrite reports both clocks on every run, and their ratio: `cpu/wall` is
the average number of cores kept busy, which makes the distinction visible
rather than arguable.

## 3. Every silhouette score was exactly 1

All 500,000 rows of the generated CSV contain the value `1`. One distinct
value, no information.

```cpp
file << ... << silhouetteScore({point}, centroids) << endl;
```

The call passes a vector containing *one* point. Inside, the same-cluster
average `a` is computed over that one-element vector and comes out 0; every
other cluster is therefore empty, so `b` is never updated from `DBL_MAX`; and
`(b - a) / max(a, b)` is 1. The column could never have been anything else.

There is a second, quieter error: distances were accumulated as *squared*
distances. The silhouette coefficient is defined on the metric, and the two
do not rank clusterings the same way.

The rewrite computes the coefficient properly, reports a singleton cluster as
0 rather than as the best possible score, and offers a sampled estimator so
the metric is usable above a few thousand points — the exact form is O(n²),
which is what the definition costs.

## 4. The seeding was quadratic in k

`initializeCentroids` recomputed the distance from every point to every
already-chosen centroid each time it picked a new one: O(n·k²). Carrying a
running nearest-centroid distance forward gives the same seeds in O(n·k).

n = 500,000:

| | k = 20 | k = 100 |
|---|---:|---:|
| v1.0, O(n·k²) | 0.115 s | 3.355 s |
| rewritten, O(n·k) | 0.032 s | 0.084 s |
| | 3.6× | **39.8×** |

The growth confirms the complexity: a five-fold increase in k cost v1.0 29×
and costs the rewrite 2.6×. It was also the one part of the program that was
never parallelised, so at k = 100 the serial seeding was a large fraction of
a run that existed to demonstrate parallel speedup.

## 5. Points were silently dropped

```cpp
int chunkSize = n / numThreads;
threadData[t] = {..., t * chunkSize, (t + 1) * chunkSize, ...};
```

The last `n % numThreads` points fall outside every thread's range. They are
never assigned, keep whatever label they held from the previous iteration,
and contribute to no centroid. At n = 500,000 with 32 threads that is 16
points — too few to notice, enough to make the parallel result differ from
the serial one for no visible reason.

The rewrite computes block bounds from the index rather than from a chunk
size, so the ranges tile `[0, n)` exactly for any n and any block count.
[`tests/partition_test.cpp`](tests/partition_test.cpp) holds it there.

## 6. 6,400 thread creations per run

Threads were created and joined inside every iteration, twice: 100 iterations
× 2 phases × 32 threads. Thread creation costs tens of microseconds, so this
was a visible fraction of the runtime — and it *grew with the thread count*,
which is precisely the wrong direction for a scaling study.

The rewrite creates its threads once and synchronises with barriers.

## 7. Runs were not independent

The thread-count sweep reused the `centroids` vector across iterations of its
own loop without reseeding. Each configuration therefore started from the
converged state the previous one had reached, so later thread counts began
from a better initial state than earlier ones — which biases the comparison
in favour of exactly the configurations the study wanted to show winning.

Combined with single samples and no repeats, this explains the outliers in
the recorded tables (the k = 10, 16-thread entry sits above the k = 20 one).

## 8. The mutex did nothing

```cpp
pthread_mutex_lock(&lock);
for (int c = 0; c < numClusters; ++c) { /* merge partials */ }
pthread_mutex_unlock(&lock);
```

This runs on the main thread after every worker has been joined. There is no
other thread alive to exclude. The CV's "mutex-merged partial sums" describes
a lock that protects nothing — the merge was already serial.

## 9. Non-standard C++ and warnings

`pthread_t threads[numThreads]` and `ThreadData threadData[numThreads]` are
variable-length arrays: a GCC extension, not C++. MSVC rejects them, which is
why this code could never have built on Windows. There is also a
`-Wsign-compare` in the silhouette loop.

The rewrite builds clean under `-Wall -Wextra -Wpedantic -Wconversion
-Wsign-conversion -Wvla -Wshadow -Werror`.

## 10. No convergence test

The algorithm always ran its full 100 iterations. The clustering converges
long before that on the datasets used, so most of what was being timed was
recomputing a fixed point.

## 11. The data had no clusters in it

Both programs clustered `rand() % 1000` in both coordinates — uniform noise.
k-means will partition uniform data into k cells quite happily and the result
looks like a clustering, but there is no right answer to compare against, so
no run could have been wrong in a way anyone would have noticed. `rand() %
1000` on a `double` also discards every fractional coordinate.

The rewrite generates Gaussian blobs with known centres and known membership,
which is what turns "it found the clusters" into something that can fail.

---

## What the rewrite does not claim

- **It has not been run above 2 cores.** Every number here and in the README
  comes from a 2-core container. The strong-scaling curve is measured to 2
  threads and then oversubscribed to 4 to show what that looks like; whether
  the design holds at 32 threads is untested, and the reduction's `blocks × k`
  accumulator array is the first thing that would need checking.
- **The MPI program does not distribute its input.** Each rank generates the
  whole dataset and keeps its own span. The computation and the reduction are
  genuinely distributed; the I/O is not.
- **No GPU work exists.** There is no CUDA implementation, and none is
  claimed.
