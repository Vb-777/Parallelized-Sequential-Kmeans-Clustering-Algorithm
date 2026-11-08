#include "kmeans/executor.hpp"

#if KMEANS_HAVE_PTHREADS
#include <pthread.h>
#endif

#include <atomic>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kmeans {
namespace {

class SerialExecutor final : public Executor {
public:
    void for_each_block(int blocks, const BlockTask& task) override {
        for (int b = 0; b < blocks; ++b) {
            task(b);
        }
    }

    [[nodiscard]] int threads() const noexcept override { return 1; }
    [[nodiscard]] std::string name() const override { return "serial"; }
};

#if KMEANS_HAVE_PTHREADS
/// A persistent pool of pthreads, synchronised with two barriers.
///
/// The original called pthread_create and pthread_join twice per iteration
/// per thread: 6,400 of each for a 100-iteration run on 32 threads. Thread
/// creation is on the order of tens of microseconds, so at the problem
/// sizes measured that overhead was a visible fraction of the runtime it
/// was supposed to be reducing - and it grew with the thread count, which
/// is exactly the wrong direction for a scaling study.
///
/// Here the threads are created once. Each round, the main thread publishes
/// the task and releases the workers through the first barrier; workers
/// claim blocks from a shared atomic counter until none are left, then meet
/// the main thread at the second barrier. Claiming rather than
/// pre-assigning gives dynamic load balance for free, and costs nothing in
/// reproducibility because which thread runs a block does not affect the
/// result.
class PthreadExecutor final : public Executor {
public:
    /// `worker_count` is the number of *extra* threads; the calling
    /// thread works too, so a worker count of zero is one thread total.
    explicit PthreadExecutor(int worker_count) : threads_(worker_count) {
        if (worker_count < 0) {
            throw std::invalid_argument("worker count cannot be negative");
        }
        if (pthread_barrier_init(&start_, nullptr,
                                 static_cast<unsigned>(worker_count + 1)) != 0 ||
            pthread_barrier_init(&done_, nullptr,
                                 static_cast<unsigned>(worker_count + 1)) != 0) {
            throw std::runtime_error("pthread_barrier_init failed");
        }

        workers_.resize(static_cast<std::size_t>(worker_count));
        for (int t = 0; t < worker_count; ++t) {
            if (pthread_create(&workers_[static_cast<std::size_t>(t)], nullptr,
                               &PthreadExecutor::worker_entry, this) != 0) {
                throw std::runtime_error("pthread_create failed");
            }
        }
    }

    ~PthreadExecutor() override {
        stopping_ = true;
        // Release the workers one last time; they see `stopping_` and
        // return instead of waiting on the done barrier.
        pthread_barrier_wait(&start_);
        for (pthread_t worker : workers_) {
            pthread_join(worker, nullptr);
        }
        pthread_barrier_destroy(&start_);
        pthread_barrier_destroy(&done_);
    }

    PthreadExecutor(const PthreadExecutor&) = delete;
    PthreadExecutor& operator=(const PthreadExecutor&) = delete;
    PthreadExecutor(PthreadExecutor&&) = delete;
    PthreadExecutor& operator=(PthreadExecutor&&) = delete;

    void for_each_block(int blocks, const BlockTask& task) override {
        task_ = &task;
        blocks_ = blocks;
        next_block_.store(0, std::memory_order_relaxed);

        pthread_barrier_wait(&start_);
        // The main thread takes blocks too rather than idling: on two cores
        // a pool of two plus an idle master would leave a third of the
        // machine doing nothing.
        run_blocks();
        pthread_barrier_wait(&done_);

        task_ = nullptr;
    }

    [[nodiscard]] int threads() const noexcept override { return threads_ + 1; }
    [[nodiscard]] std::string name() const override { return "pthreads"; }

private:
    static void* worker_entry(void* self) {
        static_cast<PthreadExecutor*>(self)->worker_loop();
        return nullptr;
    }

    void worker_loop() {
        while (true) {
            pthread_barrier_wait(&start_);
            if (stopping_) {
                return;
            }
            run_blocks();
            pthread_barrier_wait(&done_);
        }
    }

    void run_blocks() {
        const BlockTask& task = *task_;
        while (true) {
            const int block = next_block_.fetch_add(1, std::memory_order_relaxed);
            if (block >= blocks_) {
                return;
            }
            task(block);
        }
    }

    int threads_;
    std::vector<pthread_t> workers_;
    pthread_barrier_t start_{};
    pthread_barrier_t done_{};

    const BlockTask* task_ = nullptr;
    int blocks_ = 0;
    std::atomic<int> next_block_{0};
    bool stopping_ = false;
};
#endif  // KMEANS_HAVE_PTHREADS

#ifdef _OPENMP
class OpenMpExecutor final : public Executor {
public:
    explicit OpenMpExecutor(int thread_count) : threads_(thread_count) {
        if (thread_count < 1) {
            throw std::invalid_argument("thread count must be at least 1");
        }
    }

    void for_each_block(int blocks, const BlockTask& task) override {
        // Dynamic with a chunk of one, to match the pthreads pool's
        // claim-a-block policy. Blocks are equal in point count but not in
        // cost - a block whose points all fall near a centroid boundary
        // does the same work, but cache behaviour varies - and a static
        // schedule would make every round wait for the slowest block.
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_)
        for (int b = 0; b < blocks; ++b) {
            task(b);
        }
    }

    [[nodiscard]] int threads() const noexcept override { return threads_; }
    [[nodiscard]] std::string name() const override { return "openmp"; }

private:
    int threads_;
};
#endif

}  // namespace

bool parse_backend(const std::string& name, Backend& out) {
    if (name == "serial") {
        out = Backend::Serial;
    } else if (name == "pthreads") {
        out = Backend::Pthreads;
    } else if (name == "openmp") {
        out = Backend::OpenMP;
    } else {
        return false;
    }
    return true;
}

const char* to_string(Backend backend) {
    switch (backend) {
        case Backend::Serial:
            return "serial";
        case Backend::Pthreads:
            return "pthreads";
        case Backend::OpenMP:
            return "openmp";
    }
    return "unknown";
}

bool pthreads_available() noexcept {
    // POSIX threads are POSIX. The barrier-based pool has no Windows
    // equivalent that would still be honestly called "pthreads", so rather
    // than quietly substituting std::thread and letting the name lie, the
    // backend is absent there and OpenMP covers Windows.
    return KMEANS_HAVE_PTHREADS != 0;
}

bool openmp_available() noexcept {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

std::unique_ptr<Executor> make_executor(Backend backend, int threads) {
    switch (backend) {
        case Backend::Serial:
            return std::make_unique<SerialExecutor>();
        case Backend::Pthreads:
#if KMEANS_HAVE_PTHREADS
            // The pool holds threads-1 workers; the calling thread is the
            // other one, so `--threads 4` means four threads doing work.
            return std::make_unique<PthreadExecutor>(threads > 1 ? threads - 1 : 0);
#else
            throw std::runtime_error(
                "the pthreads backend is POSIX-only and this is a Windows "
                "build; use --backend openmp");
#endif
        case Backend::OpenMP:
#ifdef _OPENMP
            return std::make_unique<OpenMpExecutor>(threads);
#else
            throw std::runtime_error(
                "this binary was built without OpenMP; configure with "
                "-DKMEANS_ENABLE_OPENMP=ON");
#endif
    }
    throw std::invalid_argument("unknown backend");
}

}  // namespace kmeans
