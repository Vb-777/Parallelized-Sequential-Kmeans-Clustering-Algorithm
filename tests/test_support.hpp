// Shared guards, so a test that needs a backend skips rather than fails
// where that backend does not exist.

#ifndef KMEANS_TESTS_TEST_SUPPORT_HPP
#define KMEANS_TESTS_TEST_SUPPORT_HPP

#include <gtest/gtest.h>

#include "kmeans/executor.hpp"

// pthreads are POSIX. On a Windows build the backend is absent by design
// (see executor.hpp), and OpenMP covers the same ground there.
#define KMEANS_SKIP_WITHOUT_PTHREADS()                       \
    do {                                                     \
        if (!kmeans::pthreads_available()) {                 \
            GTEST_SKIP() << "no pthreads backend (Windows)"; \
        }                                                    \
    } while (false)

#define KMEANS_SKIP_WITHOUT_OPENMP()                     \
    do {                                                 \
        if (!kmeans::openmp_available()) {               \
            GTEST_SKIP() << "built without OpenMP";      \
        }                                                \
    } while (false)

#endif  // KMEANS_TESTS_TEST_SUPPORT_HPP
