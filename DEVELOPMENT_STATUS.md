# Development Status

**Current phase: Experimental prototype / ongoing optimization**

## Present
- [x] Sequential K-Means baseline
- [x] K-Means++ centroid initialization
- [x] pthread-based parallel assignment
- [x] thread-local centroid accumulation
- [x] performance graphs and comparison tables
- [x] project report and presentation

## Needs improvement before rigorous benchmarking
- [ ] Correct wall-clock timing
- [ ] Handle dataset/thread remainders
- [ ] Add true convergence detection
- [ ] Ensure independent/reproducible runs
- [ ] Correct and validate silhouette calculation
- [ ] Remove unnecessary mutex / improve synchronization design
- [ ] Replace non-standard VLAs with `std::vector`

## Future extensions
- [ ] Parallel K-Means++ initialization
- [ ] OpenMP implementation
- [ ] Higher-dimensional datasets
- [ ] Speedup/efficiency benchmark suite
- [ ] Isolation Forest integration

This file intentionally distinguishes implemented functionality from planned work.
