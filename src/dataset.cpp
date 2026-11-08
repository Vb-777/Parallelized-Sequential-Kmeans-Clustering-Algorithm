#include "kmeans/dataset.hpp"

#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace kmeans {
namespace {

// mt19937_64 rather than rand(). rand() has a 32,768-value period on some
// implementations and is not reproducible across platforms, so the original
// could not have regenerated its own datasets - and `rand() % 1000` on
// doubles throws away every fractional coordinate besides.
using Rng = std::mt19937_64;

}  // namespace

Dataset make_uniform(std::size_t n, Scalar extent, std::uint64_t seed) {
    Dataset dataset;
    dataset.points.resize(n);

    Rng rng(seed);
    std::uniform_real_distribution<Scalar> coordinate(0, extent);
    for (std::size_t i = 0; i < n; ++i) {
        dataset.points.x[i] = coordinate(rng);
        dataset.points.y[i] = coordinate(rng);
    }
    return dataset;
}

Dataset make_blobs(std::size_t n, int clusters, Scalar extent, Scalar spread,
                   std::uint64_t seed) {
    if (clusters <= 0) {
        throw std::invalid_argument("make_blobs needs at least one cluster");
    }

    Dataset dataset;
    dataset.points.resize(n);
    dataset.truth.resize(n);
    dataset.truth_centroids.resize(static_cast<std::size_t>(clusters));

    Rng rng(seed);
    std::uniform_real_distribution<Scalar> centre(0, extent);
    for (int c = 0; c < clusters; ++c) {
        dataset.truth_centroids.x[static_cast<std::size_t>(c)] = centre(rng);
        dataset.truth_centroids.y[static_cast<std::size_t>(c)] = centre(rng);
    }

    std::normal_distribution<Scalar> offset(0, spread);
    for (std::size_t i = 0; i < n; ++i) {
        // Round-robin rather than random assignment, so every blob has a
        // known population and an empty-cluster test can be written against
        // a dataset whose answer is known in advance.
        const auto blob = static_cast<std::size_t>(i % static_cast<std::size_t>(clusters));
        dataset.truth[i] = static_cast<Label>(blob);
        dataset.points.x[i] = dataset.truth_centroids.x[blob] + offset(rng);
        dataset.points.y[i] = dataset.truth_centroids.y[blob] + offset(rng);
    }
    return dataset;
}

Dataset load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }

    Dataset dataset;
    std::string line;
    std::getline(file, line);  // header

    std::size_t line_number = 1;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        std::string x_text;
        std::string y_text;
        if (!std::getline(fields, x_text, ',') || !std::getline(fields, y_text, ',')) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                     ": expected two comma-separated columns");
        }
        dataset.points.x.push_back(std::stod(x_text));
        dataset.points.y.push_back(std::stod(y_text));
    }
    dataset.points.label.assign(dataset.points.size(), -1);
    return dataset;
}

void save_csv(const PointCloud& points, const std::string& path) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("cannot write " + path);
    }
    file << "x,y,cluster\n";
    for (std::size_t i = 0; i < points.size(); ++i) {
        file << points.x[i] << ',' << points.y[i] << ','
             << points.label[i] << '\n';
    }
}

}  // namespace kmeans
