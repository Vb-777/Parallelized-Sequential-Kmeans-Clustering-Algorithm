# Development status

**Version 2.** The clustering, the four backends, the benchmark harness and
the test suite are complete and measured; see [README.md](README.md) for the
numbers and [BUGS.md](BUGS.md) for the audit of v1.0 that prompted the
rewrite.

v1.0's status file listed the defects in this code honestly — the wrong
timer, the dropped remainder points, the reused centroids, the useless mutex,
the broken silhouette, the variable-length arrays. Every item on that list is
now either fixed and covered by a test, or recorded in `BUGS.md` with a
measurement. That file is the replacement for this one.

## Verified on a 2-core Linux x86-64 container

- 47 GoogleTest cases and an MPI self-check at 3 ranks: all passing
- clean under AddressSanitizer, UndefinedBehaviorSanitizer and
  ThreadSanitizer (the last with OpenMP disabled; see README.md for why)
- clean at `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wvla
  -Wshadow -Werror` with GCC 13
- `shellcheck` clean on `scripts/*.sh`
- strong and weak scaling measured over 9 repeats per point
- every audit claim in `BUGS.md` reproduced from the sources in `legacy/`

## Not yet verified

- **Anything above 2 cores.** This is the single largest gap. The scaling
  study needs a real multi-core node to say anything about 8, 16 or 32
  threads.
- **The MSVC and macOS CI jobs.** Configured, never run.
- **MPI above 4 ranks, and across more than one host.** The self-check has
  only run on a single machine.

## Next

1. Re-run the scaling study on a multi-core node.
2. Generalise the kernels from 2-D to d dimensions, which makes the
   arithmetic-intensity question interesting rather than trivially
   bandwidth-bound.
3. A roofline measurement, once there is a machine whose bandwidth is worth
   measuring against.
