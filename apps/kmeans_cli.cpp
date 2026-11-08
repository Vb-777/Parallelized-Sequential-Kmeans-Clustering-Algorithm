// Command-line front end: cluster a dataset and report what happened.
//
// Every run prints the build it was produced by. A timing without its
// compiler flags is not a measurement, and this repository exists partly
// because a 31.9 s number was recorded without anyone noting that it came
// from an unoptimised build.

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"
#include "kmeans/metrics.hpp"
#include "kmeans/timer.hpp"

namespace {

struct Options {
    std::size_t n = 1'000'000;
    int k = 20;
    int iterations = 100;
    kmeans::Scalar tolerance = 1e-6;
    std::uint64_t seed = 42;
    int threads = 1;
    int blocks = 64;
    kmeans::Backend backend = kmeans::Backend::Serial;

    std::string dataset = "blobs";
    kmeans::Scalar extent = 1000;
    kmeans::Scalar spread = 20;

    std::string input;
    std::string output;
    std::size_t silhouette_samples = 0;
};

void print_usage(const char* program) {
    std::cout
        << "usage: " << program << " [options]\n\n"
        << "  --n N                  points to generate (default 1000000)\n"
        << "  --k K                  clusters (default 20)\n"
        << "  --iterations N         maximum Lloyd iterations (default 100)\n"
        << "  --tolerance T          stop when no centroid moves further (default 1e-6)\n"
        << "  --seed S               RNG seed (default 42)\n"
        << "  --backend B            serial | pthreads | openmp (default serial)\n"
        << "  --threads N            threads for the parallel backends (default 1)\n"
        << "  --blocks N             reduction blocks; must be >= threads (default 64)\n"
        << "  --dataset D            blobs | uniform (default blobs)\n"
        << "  --extent E             coordinate range (default 1000)\n"
        << "  --spread S             blob standard deviation (default 20)\n"
        << "  --input FILE.csv       cluster a CSV of x,y instead of generating\n"
        << "  --output FILE.csv      write points with their labels\n"
        << "  --silhouette N         estimate silhouette from N sampled points\n"
        << "  --version              print the build and exit\n"
        << "  --help\n";
}

void print_build() {
    std::cout << "compiler:   " << KMEANS_COMPILER << '\n'
              << "build type: " << KMEANS_BUILD_TYPE << '\n'
              << "flags:      " << KMEANS_BUILD_FLAGS << '\n'
              << "openmp:     " << (kmeans::openmp_available() ? "yes" : "no")
              << '\n';
}

/// Fetch the value after `--flag`, or exit with a message naming the flag.
const char* value_for(int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        std::cerr << argv[i] << " needs a value\n";
        std::exit(2);
    }
    return argv[++i];
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (flag == "--version") {
            print_build();
            std::exit(0);
        } else if (flag == "--n") {
            options.n = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--k") {
            options.k = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--iterations") {
            options.iterations = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--tolerance") {
            options.tolerance = std::strtod(value_for(argc, argv, i), nullptr);
        } else if (flag == "--seed") {
            options.seed = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--threads") {
            options.threads = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--blocks") {
            options.blocks = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--backend") {
            const std::string name = value_for(argc, argv, i);
            if (!kmeans::parse_backend(name, options.backend)) {
                std::cerr << "unknown backend '" << name
                          << "'; expected serial, pthreads or openmp\n";
                std::exit(2);
            }
        } else if (flag == "--dataset") {
            options.dataset = value_for(argc, argv, i);
        } else if (flag == "--extent") {
            options.extent = std::strtod(value_for(argc, argv, i), nullptr);
        } else if (flag == "--spread") {
            options.spread = std::strtod(value_for(argc, argv, i), nullptr);
        } else if (flag == "--input") {
            options.input = value_for(argc, argv, i);
        } else if (flag == "--output") {
            options.output = value_for(argc, argv, i);
        } else if (flag == "--silhouette") {
            options.silhouette_samples =
                std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else {
            std::cerr << "unknown option '" << flag << "'\n\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);

    try {
        kmeans::Timer load_timer;
        kmeans::Dataset dataset;
        if (!options.input.empty()) {
            dataset = kmeans::load_csv(options.input);
        } else if (options.dataset == "uniform") {
            dataset = kmeans::make_uniform(options.n, options.extent, options.seed);
        } else if (options.dataset == "blobs") {
            dataset = kmeans::make_blobs(options.n, options.k, options.extent,
                                         options.spread, options.seed);
        } else {
            std::cerr << "unknown dataset '" << options.dataset
                      << "'; expected blobs or uniform\n";
            return 2;
        }
        const double load_seconds = load_timer.wall_seconds();

        kmeans::Params params;
        params.k = options.k;
        params.max_iterations = options.iterations;
        params.tolerance = options.tolerance;
        params.seed = options.seed;
        params.blocks = options.blocks;
        params.threads = options.threads;

        auto executor = kmeans::make_executor(options.backend, options.threads);
        const kmeans::Result result =
            kmeans::run(dataset.points, params, *executor);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "points:      " << dataset.points.size() << '\n'
                  << "clusters:    " << options.k << '\n'
                  << "backend:     " << executor->name() << " x "
                  << executor->threads() << '\n'
                  << "blocks:      " << options.blocks << '\n'
                  << "dataset:     " << (options.input.empty() ? options.dataset
                                                               : options.input)
                  << " (" << load_seconds << " s to prepare)\n"
                  << "iterations:  " << result.iterations
                  << (result.converged ? " (converged)" : " (iteration limit)")
                  << '\n'
                  << "seeding:     " << result.seed_seconds << " s\n"
                  << "wall:        " << result.wall_seconds << " s\n"
                  << "cpu:         " << result.cpu_seconds << " s\n"
                  << "cores used:  " << std::setprecision(2) << result.cores_used()
                  << "   <- cpu/wall; this is why clock() cannot measure speedup\n"
                  << std::setprecision(4)
                  << "inertia:     " << result.inertia << '\n'
                  << "davies-bouldin: "
                  << kmeans::davies_bouldin(dataset.points, result.centroids)
                  << "  (lower is better)\n";

        if (options.silhouette_samples > 0) {
            const kmeans::Scalar score = kmeans::silhouette_sampled(
                dataset.points, options.k, options.silhouette_samples,
                options.seed, *executor);
            std::cout << "silhouette:  " << score << "  (sampled over "
                      << options.silhouette_samples << " points, higher is better)\n";
        }

        if (!options.output.empty()) {
            kmeans::save_csv(dataset.points, options.output);
            std::cout << "wrote " << options.output << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
