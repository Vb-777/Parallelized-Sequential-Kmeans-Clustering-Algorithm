// How blocks are dispatched - the only thing the three shared-memory
// backends disagree about.
//
// Lloyd's algorithm is written once, in lloyd.cpp, in terms of "run this
// function over every block". Serial runs them in a loop, OpenMP runs them
// in a parallel for, pthreads runs them on a persistent pool. Because the
// blocks and their merge order are fixed by the problem rather than by the
// schedule, all three produce identical results, and the whole parallel
// implementation is three small files instead of three copies of the
// algorithm that must be kept in step.
//
// The pool matters. The original created and joined two sets of threads
// inside every iteration: 100 iterations x 2 phases x 32 threads is 6,400
// pthread_create calls per run, which cost more than some of the work they
// were dividing. Here the threads are created once per executor and
// synchronised with a barrier.

#ifndef KMEANS_EXECUTOR_HPP
#define KMEANS_EXECUTOR_HPP

// pthreads are POSIX; MSVC has no <pthread.h>. The pool is therefore
// compiled only where it exists, and OpenMP is the parallel backend on
// Windows. Substituting std::thread under the name "pthreads" would make
// the backend name untrue, which is worse than the gap.
#ifndef KMEANS_HAVE_PTHREADS
#if defined(_WIN32)
#define KMEANS_HAVE_PTHREADS 0
#else
#define KMEANS_HAVE_PTHREADS 1
#endif
#endif

#include <functional>
#include <memory>
#include <string>

namespace kmeans {

/// Work for one block. Called once per block, on some thread.
using BlockTask = std::function<void(int block)>;

class Executor {
public:
    virtual ~Executor() = default;

    /// Run `task` for every block in [0, blocks) and return once all have
    /// finished. Blocks may run in any order, on any thread; the algorithm
    /// relies only on all of them having completed on return.
    virtual void for_each_block(int blocks, const BlockTask& task) = 0;

    [[nodiscard]] virtual int threads() const noexcept = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

enum class Backend { Serial, Pthreads, OpenMP };

/// Parse a backend name, as the CLI spells it. Returns false on a name that
/// is not one of serial, pthreads, openmp.
bool parse_backend(const std::string& name, Backend& out);

const char* to_string(Backend backend);

/// Build an executor. `threads` is ignored by the serial backend. Throws
/// std::runtime_error if the binary was built without support for the
/// backend asked for - OpenMP is optional at configure time.
std::unique_ptr<Executor> make_executor(Backend backend, int threads);

/// Whether this binary was compiled with OpenMP support.
bool openmp_available() noexcept;

/// Whether the pthreads backend exists in this build (false on Windows).
bool pthreads_available() noexcept;

}  // namespace kmeans

#endif  // KMEANS_EXECUTOR_HPP
