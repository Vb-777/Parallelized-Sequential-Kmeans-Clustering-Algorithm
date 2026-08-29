# Parallelized K-Means++ Clustering for Efficient Data Analysis

**Status: Research / Experimental Prototype**

This project studies the performance of K-Means clustering when the computationally intensive point-to-centroid assignment and centroid-update phases are parallelized with POSIX threads (`pthreads`). It includes a sequential baseline, a threaded implementation, performance plots/tables, and an accompanying report.

The project also discusses K-Means++ initialization and the possible combination of clustering with Isolation Forest for anomaly detection. The currently supplied C++ programs implement K-Means/K-Means++ clustering and silhouette-related output; an Isolation Forest implementation is **not** included in the current source tree.

## Objectives

- Establish a sequential K-Means baseline.
- Initialize centroids using K-Means++.
- Parallelize point-to-centroid assignment with pthreads.
- Parallelize local centroid accumulation with thread-local sums/counts.
- Compare execution behavior across thread counts.
- Study the effect of dataset size, cluster count, iterations and dimensionality.
- Examine the trade-off between parallel speedup and thread/synchronization overhead.
- Preserve a reproducible research structure for future optimization.

## Repository Contents

```text
Parallelized-KMeans/
├── code/
│   ├── kmN01_Sequential (overall).cpp
│   └── kmN1_Parallel (overall).cpp
│
├── graphs/
│   ├── ClusterCount_based_performance_comparison.png
│   └── DatasetSize_based_performance_comparison.png
│
├── tables/
│   ├── Cluster Count based Comparision.xlsx
│   └── Dataset Size  based Comparision.xlsx
│
├── Ppt.pdf
├── Report.pdf
├── README.md
├── DEVELOPMENT_STATUS.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Algorithm

### K-Means

The clustering loop consists of:

1. Assign each point to its nearest centroid.
2. Accumulate per-cluster coordinate sums and counts.
3. Update each centroid using the mean of its assigned points.
4. Repeat for a configured maximum number of iterations.

For a point `x_i` and centroid `c_j`, squared Euclidean distance is used:

```text
d(x_i,c_j) = (x_i.x-c_j.x)^2 + (x_i.y-c_j.y)^2
```

### K-Means++ initialization

The supplied programs initialize the first centroid randomly and select later centroids using distance-weighted sampling based on the nearest already-selected centroid.

Importantly, the current K-Means++ initialization itself is **sequential** in both supplied programs. The project therefore primarily demonstrates parallelization of the iterative clustering phase rather than a fully parallel K-Means++ pipeline.

## Parallel Design

The threaded implementation uses pthreads. The dataset is divided into contiguous chunks. During assignment, each thread independently updates the cluster labels of its own points. During centroid accumulation, each thread maintains private arrays:

```text
localSumX[k]
localSumY[k]
localCount[k]
```

After all worker threads finish, the main thread combines the local results and updates the global centroids.

Conceptually:

```text
                 Dataset
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Thread 0  Thread 1   Thread P
       assignment assignment assignment
          │         │         │
          └──── local sums/counts ────┘
                    │
                    ▼
             Global centroid update
                    │
                    ▼
                 next iteration
```

This avoids simultaneous writes to shared centroid accumulators during the local accumulation phase.

## Experimental Dimensions

The accompanying report discusses experiments varying: 

- number of threads;
- dataset size;
- number of clusters;
- maximum allowed iterations;
- initialization method (random vs. K-Means++);
- dimensionality/features;
- sequential versus parallel execution.

The supplied graphs specifically cover dataset-size and cluster-count comparisons. The report describes additional observations.

## Performance Interpretation

The reported behavior is consistent with the expected parallel-computing trade-off:

```text
small workload  → thread overhead can dominate
large workload  → parallel computation becomes more useful
more threads    → speedup eventually shows diminishing returns
```

The report observes that parallel execution generally becomes more advantageous for larger datasets/cluster counts, while very small workloads can favor sequential execution because thread creation and synchronization overhead are not amortized.

## Complexity

For `N` points, `K` clusters and `I` K-Means iterations, the dominant assignment/update work is approximately:

```text
O(I × N × K)
```

For the 2-D implementation, the dimensionality factor is constant. With `P` threads, the idealized computation component approaches roughly:

```text
O(I × N × K / P)
```

but actual performance is limited by thread creation, memory bandwidth, synchronization, load imbalance, and the sequential centroid-combination and initialization portions.

## Building

A Linux/Unix environment with a C++ compiler and pthread support is recommended.

Sequential:

```bash
g++ -std=c++17 -O2 -pthread "code/kmN01_Sequential (overall).cpp" -o kmeans_sequential
```

Parallel:

```bash
g++ -std=c++17 -O2 -pthread "code/kmN1_Parallel (overall).cpp" -o kmeans_parallel
```

Run:

```bash
./kmeans_sequential
./kmeans_parallel
```

The programs currently generate their CSV outputs in the working directory. These generated files are ignored by `.gitignore`.

## Current Source-Level Review Notes

The existing source is suitable as an experimental prototype, but several issues should be addressed before treating benchmark numbers as rigorous evidence.

### 1. Timing method

The programs use `clock()` and `CLOCKS_PER_SEC`. For multithreaded wall-clock benchmarking, `std::chrono::steady_clock` or `clock_gettime(CLOCK_MONOTONIC, ...)` is preferable. `clock()` measures process CPU time rather than elapsed wall-clock time, so it can distort comparisons between different thread counts.

### 2. No actual convergence test

The K-Means loop is bounded by `maxIterations`, but the current implementation does not stop early when centroids stop moving or assignments stabilize. Therefore the code should be described as fixed-iteration K-Means rather than convergence-controlled K-Means.

### 3. Parallel chunk remainder

The parallel implementation uses:

```text
chunkSize = n / numThreads
start = t * chunkSize
end   = (t + 1) * chunkSize
```

If `n` is not exactly divisible by the thread count, the final remainder points are not processed. This must be fixed before generalizing the benchmark.

### 4. Reusing centroids across thread-count runs

The parallel `main()` runs several thread counts sequentially using the same centroid vector. Consequently, a later thread-count experiment starts from the centroids left by the previous run. For a fair thread-scaling benchmark, each run should use the same initial centroids (or the same random seed and initialization procedure).

### 5. Mutex usage

The centroid aggregation is performed only after all worker threads have joined, so the displayed mutex around that aggregation does not provide useful synchronization. It can be removed from the current design.

### 6. Silhouette implementation

The helper used for writing silhouette output passes a single-point vector into `silhouetteScore()`. A conventional silhouette calculation requires distances from each point to other points in its own cluster and to points in other clusters. The current output should therefore be treated as experimental/debug output rather than a validated standard silhouette implementation.

### 7. Non-standard variable-length arrays

The parallel program declares `pthread_t threads[numThreads]` and `ThreadData threadData[numThreads]`. Variable-length arrays are not standard C++. `std::vector` should be used for portable C++17 code.

### 8. K-Means++ is not parallelized

The project title refers to K-Means++, but initialization remains sequential. A future version can separately benchmark K-Means++ initialization and parallelize its distance-update/sampling stages if required.

These are intentionally documented here rather than silently changing the supplied source code.

## K-Means + Isolation Forest

The presentation discusses combining K-Means clustering with Isolation Forest for anomaly/outlier detection. The motivation is to first organize heterogeneous data into clusters and then identify anomalous points within or across those groups.

The current repository, however, does **not** contain an Isolation Forest implementation. Future work can add it as a separate module and evaluate whether clustering improves anomaly-detection quality.

## Reproducibility Recommendations

For future benchmark runs:

1. Use identical input data for all compared modes.
2. Use identical initial centroids for sequential and parallel runs.
3. Fix the random seed.
4. Measure wall-clock elapsed time.
5. Warm up the system or repeat each configuration several times.
6. Report mean and variation rather than one timing.
7. Record compiler, compiler flags, CPU model and thread count.
8. Ensure every input point is processed exactly once.
9. Separate initialization time from K-Means iteration time when comparing algorithms.
10. Validate clustering quality independently from runtime.

## Future Work

- Fix remainder handling in thread partitioning.
- Replace `clock()` with a wall-clock timer.
- Add convergence detection.
- Make all experiment parameters configurable from the command line.
- Make thread-count experiments independent and reproducible.
- Implement a correct scalable silhouette calculation.
- Replace VLA declarations with standard C++ containers.
- Benchmark speedup and parallel efficiency.
- Add OpenMP as a second parallelization strategy for comparison.
- Add higher-dimensional input support.
- Add Isolation Forest implementation and cluster-aware anomaly evaluation.
- Investigate parallel K-Means++ initialization.

## Academic Context

This project is an experimental study of shared-memory parallelization of K-Means clustering, with emphasis on pthread-based decomposition of assignment and centroid-update work and empirical performance analysis.

The supplied `Ppt.pdf` and `Report.pdf` contain the original project presentation and analysis.

## Author

**Vaibhav Barsaiyan**  
M.Tech. — Computer Science & Engineering  
Indian Institute of Technology Hyderabad
