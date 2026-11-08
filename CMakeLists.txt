cmake_minimum_required(VERSION 3.16)
project(kmeans_hpc LANGUAGES CXX)

# The original was built by hand with g++ and no flags recorded anywhere,
# which is how a -O0 baseline ends up in a speedup table. The build type is
# therefore explicit, defaulted, and printed by every binary that reports a
# measurement: `kmeans --version` tells you the flags it was compiled with,
# so a number can never again be reported without its build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(THREADS_PREFER_PTHREAD_FLAG ON)

option(KMEANS_BUILD_TESTS "Build the test suite" ON)
option(KMEANS_ENABLE_OPENMP "Build the OpenMP backend if OpenMP is found" ON)
option(KMEANS_ENABLE_MPI "Build the MPI program if MPI is found" ON)
option(KMEANS_WERROR "Treat warnings as errors" OFF)
option(KMEANS_NATIVE_ARCH
       "Compile for the building machine's ISA (-march=native). Faster, and \
not portable - leave off for a binary that will run elsewhere." OFF)

find_package(Threads REQUIRED)

# ------------------------------------------------------------------ flags

add_library(kmeans_flags INTERFACE)
target_compile_options(kmeans_flags INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        # The original used variable-length arrays - `pthread_t
        # threads[numThreads]` - which are a GCC extension, not C++. MSVC
        # rejects them outright, which is why that code never built there.
        -Wvla
        -Wshadow
        $<$<BOOL:${KMEANS_WERROR}>:-Werror>
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 $<$<BOOL:${KMEANS_WERROR}>:/WX>
    >
)
if(KMEANS_NATIVE_ARCH)
    target_compile_options(kmeans_flags INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>)
endif()

# GCC defaults to -mprefer-vector-width=256 even on AVX-512 hardware,
# because 512-bit instructions can pull the clock down on some Skylake-SP
# parts and lose more than they gain. For this kernel, on the machine the
# README's numbers were taken on, they gain: 0.535 s to 0.392 s, a 37%
# improvement. That is a property of the kernel and the chip together, so it
# is an option rather than a default - measure before setting it.
set(KMEANS_VECTOR_WIDTH "" CACHE STRING
    "Pass -mprefer-vector-width=N (e.g. 512). Empty leaves the compiler's default.")
if(KMEANS_VECTOR_WIDTH)
    target_compile_options(kmeans_flags INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-mprefer-vector-width=${KMEANS_VECTOR_WIDTH}>)
endif()

# ---------------------------------------------------------------- library

add_library(kmeans_core
    src/dataset.cpp
    src/executor.cpp
    src/kernels.cpp
    src/kmeans.cpp
    src/metrics.cpp
)
target_include_directories(kmeans_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kmeans_core PUBLIC Threads::Threads PRIVATE kmeans_flags)

if(KMEANS_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(kmeans_core PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP backend: enabled")
    else()
        message(STATUS "OpenMP backend: not available (OpenMP not found)")
    endif()
else()
    message(STATUS "OpenMP backend: disabled by KMEANS_ENABLE_OPENMP=OFF")
endif()

# Record the flags the binaries were built with, so a result can always be
# traced back to its build.
string(TOUPPER "${CMAKE_BUILD_TYPE}" KMEANS_BUILD_TYPE_UPPER)
set(KMEANS_BUILD_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${KMEANS_BUILD_TYPE_UPPER}}")
target_compile_definitions(kmeans_core PUBLIC
    KMEANS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    KMEANS_BUILD_FLAGS="${KMEANS_BUILD_FLAGS}"
    KMEANS_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

# ------------------------------------------------------------------- apps

add_executable(kmeans apps/kmeans_cli.cpp)
target_link_libraries(kmeans PRIVATE kmeans_core kmeans_flags)

add_executable(kmeans_bench bench/bench_main.cpp)
target_link_libraries(kmeans_bench PRIVATE kmeans_core kmeans_flags)

if(KMEANS_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        add_executable(kmeans_mpi apps/kmeans_mpi.cpp)
        target_link_libraries(kmeans_mpi PRIVATE kmeans_core kmeans_flags MPI::MPI_CXX)
        message(STATUS "MPI program: enabled (${MPI_CXX_COMPILER})")
    else()
        message(STATUS "MPI program: not available (MPI not found)")
    endif()
else()
    message(STATUS "MPI program: disabled by KMEANS_ENABLE_MPI=OFF")
endif()

# ------------------------------------------------------------------ tests

if(KMEANS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(kmeans_tests
        tests/partition_test.cpp
        tests/test_support.hpp
        tests/determinism_test.cpp
        tests/clustering_test.cpp
        tests/seeding_test.cpp
        tests/metrics_test.cpp
        tests/executor_test.cpp
    )
    target_link_libraries(kmeans_tests PRIVATE kmeans_core kmeans_flags GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(kmeans_tests)

    if(MPI_CXX_FOUND)
        # One MPI test, run through mpiexec: that the distributed result is
        # identical to the shared-memory one. Added as a plain test rather
        # than a gtest, because it has to be launched by the MPI runtime.
        add_test(NAME mpi_matches_shared_memory
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:kmeans_mpi>
                         --self-check ${MPIEXEC_POSTFLAGS})
    endif()
endif()
