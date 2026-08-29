#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <numeric> // Include this for accumulate function
#include <fstream> // For writing to CSV file

using namespace std;

struct Point {
    double x, y;
    int cluster;
};

struct Centroid {
    double x, y;
};

struct ThreadData {
    vector<Point>* points;
    vector<Centroid>* centroids;
    vector<double> localSumX, localSumY;
    vector<int> localCount;
    int startIdx, endIdx;
    int numClusters;
};

pthread_mutex_t lock;

// Function to assign points to clusters
void* assignClusters(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    vector<Point>& points = *(data->points);
    vector<Centroid>& centroids = *(data->centroids);

    for (int i = data->startIdx; i < data->endIdx; i++) {
        double minDist = numeric_limits<double>::max();
        for (int c = 0; c < data->numClusters; c++) {
            double dist = pow(points[i].x - centroids[c].x, 2) + pow(points[i].y - centroids[c].y, 2);
            if (dist < minDist) {
                minDist = dist;
                points[i].cluster = c;
            }
        }
    }
    return nullptr;
}

// Function to compute local centroid updates
void* updateCentroids(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    vector<Point>& points = *(data->points);

    for (int i = data->startIdx; i < data->endIdx; i++) {
        int clusterIdx = points[i].cluster;
        data->localSumX[clusterIdx] += points[i].x;
        data->localSumY[clusterIdx] += points[i].y;
        data->localCount[clusterIdx]++;
    }

    return nullptr;
}

// Function to perform K-Means
void kMeans(vector<Point>& points, vector<Centroid>& centroids, int numThreads, int maxIterations = 100) {
    int numClusters = centroids.size();
    int n = points.size();
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    for (int iter = 0; iter < maxIterations; iter++) {
        int chunkSize = n / numThreads;

        // Cluster assignment
        for (int t = 0; t < numThreads; t++) {
            threadData[t] = {&points, &centroids, {}, {}, {}, t * chunkSize, (t + 1) * chunkSize, numClusters};
            pthread_create(&threads[t], nullptr, assignClusters, &threadData[t]);
        }
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], nullptr);
        }

        // Initialize local data for threads
        for (int t = 0; t < numThreads; t++) {
            threadData[t].localSumX = vector<double>(numClusters, 0);
            threadData[t].localSumY = vector<double>(numClusters, 0);
            threadData[t].localCount = vector<int>(numClusters, 0);
            pthread_create(&threads[t], nullptr, updateCentroids, &threadData[t]);
        }
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], nullptr);
        }

        // Global centroid update with mutex
        pthread_mutex_lock(&lock);
        for (int c = 0; c < numClusters; c++) {
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int t = 0; t < numThreads; t++) {
                sumX += threadData[t].localSumX[c];
                sumY += threadData[t].localSumY[c];
                count += threadData[t].localCount[c];
            }
            if (count > 0) {
                centroids[c].x = sumX / count;
                centroids[c].y = sumY / count;
            }
        }
        pthread_mutex_unlock(&lock);
    }
}

// Function to initialize centroids using K-Means++ method
void initializeCentroids(vector<Point>& points, vector<Centroid>& centroids) {
    int numClusters = centroids.size();
    int n = points.size();
    centroids[0] = {points[rand() % n].x, points[rand() % n].y};

    for (int c = 1; c < numClusters; c++) {
        vector<double> distances(n, numeric_limits<double>::max());
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < c; k++) {
                double dist = pow(points[i].x - centroids[k].x, 2) + pow(points[i].y - centroids[k].y, 2);
                distances[i] = min(distances[i], dist);
            }
        }
        double totalDistance = accumulate(distances.begin(), distances.end(), 0.0);
        double randomValue = (rand() / (double)RAND_MAX) * totalDistance;
        for (int i = 0; i < n; i++) {
            if ((randomValue -= distances[i]) <= 0) {
                centroids[c] = {points[i].x, points[i].y};
                break;
            }
        }
    }
}

// Function to compute silhouette score for individual point
double silhouetteScore(const vector<Point>& points, const vector<Centroid>& centroids) {
    double totalScore = 0.0;
    for (const auto& point : points) {
        // Calculate average distance to points in the same cluster (a)
        double a = 0.0;
        int sameClusterCount = 0;
        for (const auto& p : points) {
            if (p.cluster == point.cluster) {
                a += pow(p.x - point.x, 2) + pow(p.y - point.y, 2);
                sameClusterCount++;
            }
        }
        a /= sameClusterCount;

        // Calculate the minimum average distance to points in other clusters (b)
        double b = numeric_limits<double>::max();
        for (int c = 0; c < centroids.size(); c++) {
            if (c != point.cluster) {
                double clusterDist = 0.0;
                int otherClusterCount = 0;
                for (const auto& p : points) {
                    if (p.cluster == c) {
                        clusterDist += pow(p.x - point.x, 2) + pow(p.y - point.y, 2);
                        otherClusterCount++;
                    }
                }
                clusterDist /= otherClusterCount;
                b = min(b, clusterDist);
            }
        }

        // Calculate silhouette score for the point
        totalScore += (b - a) / max(a, b);
    }
    return totalScore / points.size();
}

// Function to save silhouette scores to CSV file
void saveSilhouetteScoresToCSV(const vector<Point>& points, const string& filename, const vector<Centroid>& centroids) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error opening silhouette scores file." << endl;
        return;
    }
    file << "Point X,Point Y,Cluster,Silhouette Score" << endl;

    for (const auto& point : points) {
        file << point.x << "," << point.y << "," << point.cluster << "," << silhouetteScore({point}, centroids) << endl;
    }
    file.close();
}

int main() {
    pthread_mutex_init(&lock, nullptr);
    srand(time(0));

    int dataSetSize = 500000; // 10^5
    int numClusters = 100;
    vector<Point> points(dataSetSize);
    vector<Centroid> centroids(numClusters);

    // Initialize random points
    for (auto& p : points) {
        p.x = rand() % 1000;
        p.y = rand() % 1000;
    }

    // Initialize centroids using K-Means++
    initializeCentroids(points, centroids);

    // Open output file for time results
    ofstream timeFile("kmeans_time_results.csv");
    if (!timeFile) {
        cerr << "Error opening time results file." << endl;
        return 1;
    }
    timeFile << "Number of Threads,Time Taken (seconds)" << endl;

    // Loop over different thread counts
    for (int threads = 2; threads <= 32; threads *= 2) {
        double startTime = clock();
        kMeans(points, centroids, threads);
        double endTime = clock();
        double timeTaken = (endTime - startTime) / CLOCKS_PER_SEC;
        cout << "Threads: " << threads << ", Time: " << timeTaken << " seconds\n";

        // Save the time taken to the CSV file
        timeFile << threads << "," << timeTaken << endl;
    }

    // Save silhouette scores to a CSV file
    saveSilhouetteScoresToCSV(points, "silhouette_scores.csv", centroids);

    timeFile.close();
    pthread_mutex_destroy(&lock);
    return 0;
}
