// The scaling harness.
//
// What the original did: one run per configuration, timed with clock(), on
// centroids left over from the previous configuration's run. Three separate
// reasons the resulting table cannot be read as a scaling curve - no
// repeats so no idea of the noise, the wrong clock, and each run starting
// from a better initial state than the one before it, which biases later
// thread counts in their favour.
//
// What this does: every configuration gets a fresh dataset and a fresh
// seeding, a discarded warm-up, and `--repeats` timed runs reported as mean,
// standard deviation and minimum. Speedup is computed against the one-thread
// run of the same backend measured the same way, in the same process.
//
//   kmeans_bench --mode strong --n 2000000 --k 20 --threads 1,2,4 --repeats 5
//   kmeans_bench --mode weak   --n 1000000 --k 20 --threads 1,2,4
//
// Strong scaling fixes the problem and adds threads: speedup should
// approach p, and the gap from p is what parallel overhead costs. Weak
// scaling grows the problem with the thread count, so the time should stay
// flat; it is the honest measure when a machine's memory bandwidth, rather
// than its core count, is the limit.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "kmeans/dataset.hpp"
#include "kmeans/kmeans.hpp"

namespace {

struct Options {
    std::string mode = "strong";
    std::size_t n = 1'000'000;
    int k = 20;
    int iterations = 50;
    int blocks = 64;
    std::uint64_t seed = 42;
    int repeats = 5;
    std::vector<int> threads{1, 2, 4};
    std::vector<kmeans::Backend> backends{kmeans::Backend::Pthreads};
    std::string csv;
    kmeans::Scalar spread = 20;
    kmeans::Scalar extent = 1000;
};

struct Sample {
    double mean = 0;
    double stddev = 0;
    double best = 0;
};

Sample summarise(const std::vector<double>& values) {
    Sample sample;
    if (values.empty()) {
        return sample;
    }
    sample.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    sample.best = *std::min_element(values.begin(), values.end());

    if (values.size() > 1) {
        double sum_squares = 0;
        for (double value : values) {
            const double delta = value - sample.mean;
            sum_squares += delta * delta;
        }
        // Sample standard deviation: n-1, because these are repeats drawn
        // from a noisy machine, not the whole population of possible runs.
        sample.stddev = std::sqrt(sum_squares / static_cast<double>(values.size() - 1));
    }
    return sample;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoi(item.c_str()));
        }
    }
    return values;
}

const char* value_for(int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        std::cerr << argv[i] << " needs a value\n";
        std::exit(2);
    }
    return argv[++i];
}

void print_usage(const char* program) {
    std::cout << "usage: " << program << " [options]\n\n"
              << "  --mode strong|weak     scaling study (default strong)\n"
              << "  --n N                  points; per thread in weak mode\n"
              << "  --k K                  clusters (default 20)\n"
              << "  --iterations N         fixed iteration count (default 50)\n"
              << "  --threads 1,2,4        thread counts to measure\n"
              << "  --backends pthreads,openmp,serial\n"
              << "  --repeats N            timed runs per configuration (default 5)\n"
              << "  --blocks N             reduction blocks (default 64)\n"
              << "  --seed S               RNG seed (default 42)\n"
              << "  --csv FILE             write the per-run measurements\n";
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (flag == "--mode") {
            options.mode = value_for(argc, argv, i);
        } else if (flag == "--n") {
            options.n = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--k") {
            options.k = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--iterations") {
            options.iterations = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--blocks") {
            options.blocks = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--seed") {
            options.seed = std::strtoull(value_for(argc, argv, i), nullptr, 10);
        } else if (flag == "--repeats") {
            options.repeats = std::atoi(value_for(argc, argv, i));
        } else if (flag == "--threads") {
            options.threads = parse_int_list(value_for(argc, argv, i));
        } else if (flag == "--csv") {
            options.csv = value_for(argc, argv, i);
        } else if (flag == "--backends") {
            options.backends.clear();
            std::istringstream stream(value_for(argc, argv, i));
            std::string name;
            while (std::getline(stream, name, ',')) {
                kmeans::Backend backend{};
                if (!kmeans::parse_backend(name, backend)) {
                    std::cerr << "unknown backend '" << name << "'\n";
                    std::exit(2);
                }
                options.backends.push_back(backend);
            }
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
    const bool weak = options.mode == "weak";

    std::cout << "compiler:   " << KMEANS_COMPILER << '\n'
              << "build type: " << KMEANS_BUILD_TYPE << '\n'
              << "flags:      " << KMEANS_BUILD_FLAGS << '\n'
              << "mode:       " << options.mode << "   repeats: " << options.repeats
              << "   k: " << options.k << "   iterations: " << options.iterations
              << "\n\n";

    std::ofstream csv;
    if (!options.csv.empty()) {
        csv.open(options.csv);
        csv << "backend,threads,points,k,blocks,repeat,wall_seconds,cpu_seconds,"
               "iterations,inertia\n";
    }

    for (kmeans::Backend backend : options.backends) {
        std::cout << kmeans::to_string(backend) << '\n'
                  << "  threads      points   wall mean      sd       best   "
                  << (weak ? "   t1/tp  efficiency" : " speedup  efficiency")
                  << "  cores  inertia\n";

        double baseline = 0;

        for (int threads : options.threads) {
            const std::size_t n =
                weak ? options.n * static_cast<std::size_t>(threads) : options.n;

            kmeans::Params params;
            params.k = options.k;
            params.max_iterations = options.iterations;
            // A fixed iteration count, not convergence, for the timing
            // study: a run that converges early does less work, and
            // comparing different amounts of work across thread counts
            // would not be a scaling measurement.
            params.tolerance = 0;
            params.seed = options.seed;
            params.blocks = options.blocks;
            params.threads = threads;

            auto executor = kmeans::make_executor(backend, threads);

            std::vector<double> walls;
            std::vector<double> cpus;
            kmeans::Result last;

            // One discarded warm-up: the first run pays for page faults on
            // freshly allocated memory and for the CPU still being at its
            // idle clock.
            for (int repeat = -1; repeat < options.repeats; ++repeat) {
                kmeans::Dataset dataset = kmeans::make_blobs(
                    n, options.k, options.extent, options.spread, options.seed);

                last = kmeans::run(dataset.points, params, *executor);
                if (repeat < 0) {
                    continue;
                }

                walls.push_back(last.wall_seconds);
                cpus.push_back(last.cpu_seconds);

                if (csv) {
                    csv << kmeans::to_string(backend) << ',' << threads << ',' << n
                        << ',' << options.k << ',' << options.blocks << ',' << repeat
                        << ',' << last.wall_seconds << ',' << last.cpu_seconds << ','
                        << last.iterations << ',' << last.inertia << '\n';
                }
            }

            const Sample wall = summarise(walls);
            const Sample cpu = summarise(cpus);

            if (baseline == 0) {
                baseline = wall.mean;
            }
            // The two modes measure different things and must not share a
            // formula. Strong scaling: speedup is t1/tp and efficiency is
            // that over p. Weak scaling: the work grew with p, so there is
            // no speedup to report at all, and efficiency is t1/tp
            // directly - 100% means the time stayed flat as the problem and
            // the machine grew together.
            const double speedup = baseline / wall.mean;
            const double efficiency = weak ? speedup : speedup / threads;

            std::cout << std::fixed << std::setw(9) << threads << std::setw(12) << n
                      << std::setprecision(4) << std::setw(11) << wall.mean
                      << std::setw(9) << wall.stddev << std::setw(11) << wall.best
                      << std::setprecision(2) << std::setw(9) << speedup << "x"
                      << std::setw(11) << (efficiency * 100) << "%" << std::setw(7)
                      << (wall.mean > 0 ? cpu.mean / wall.mean : 0)
                      << std::setprecision(1) << std::setw(14) << last.inertia
                      << '\n';
        }
        std::cout << '\n';
    }

    if (csv) {
        std::cout << "per-run measurements written to " << options.csv << '\n';
    }
    return 0;
}
