#include <iostream>
#include <vector>
#include <cmath>
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

// Function to assign points to clusters
void assignClusters(vector<Point>& points, vector<Centroid>& centroids, int numClusters) {
    for (auto& point : points) {
        double minDist = numeric_limits<double>::max();
        for (int c = 0; c < numClusters; c++) {
            double dist = pow(point.x - centroids[c].x, 2) + pow(point.y - centroids[c].y, 2);
            if (dist < minDist) {
                minDist = dist;
                point.cluster = c;
            }
        }
    }
}

// Function to compute centroid updates
void updateCentroids(vector<Point>& points, vector<Centroid>& centroids, int numClusters) {
    vector<double> localSumX(numClusters, 0);
    vector<double> localSumY(numClusters, 0);
    vector<int> localCount(numClusters, 0);

    for (const auto& point : points) {
        int clusterIdx = point.cluster;
        localSumX[clusterIdx] += point.x;
        localSumY[clusterIdx] += point.y;
        localCount[clusterIdx]++;
    }

    for (int c = 0; c < numClusters; c++) {
        if (localCount[c] > 0) {
            centroids[c].x = localSumX[c] / localCount[c];
            centroids[c].y = localSumY[c] / localCount[c];
        }
    }
}

// Function to perform K-Means
void kMeans(vector<Point>& points, vector<Centroid>& centroids, int maxIterations = 100) {
    int numClusters = centroids.size();
    
    for (int iter = 0; iter < maxIterations; iter++) {
        // Cluster assignment
        assignClusters(points, centroids, numClusters);
        
        // Centroid update
        updateCentroids(points, centroids, numClusters);
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
    srand(time(0));

    int dataSetSize = 100000; // 10^5
    int numClusters = 20;
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
    ofstream timeFile("kmeans_time_resultsKMN1l.csv");
    if (!timeFile) {
        cerr << "Error opening time results file." << endl;
        return 1;
    }
    timeFile << "Time Taken (seconds)" << endl;

    // Record time for sequential K-Means execution
    double startTime = clock();
    kMeans(points, centroids);
    double endTime = clock();
    double timeTaken = (endTime - startTime) / CLOCKS_PER_SEC;
    cout << "Time: " << timeTaken << " seconds\n";

    // Save the time taken to the CSV file
    timeFile << timeTaken << endl;

    // Save silhouette scores to a CSV file
    saveSilhouetteScoresToCSV(points, "silhouette_scoreskmN1l.csv", centroids);

    timeFile.close();
    return 0;
}
