#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================================================
// UTILITY STRUCTURES AND FUNCTIONS
// ============================================================

struct Point {
    vector<double> values;
    int cluster = -1;
};

struct ClusteringResult {
    string algorithm;
    string dataset;
    int n;
    int k;
    double sse;
    double runtime_seconds;
    double memory_mb;
};

// Euclidean distance
double distance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double distance(const Point& a, const Point& b) {
    return distance(a.values, b.values);
}

// ============================================================
// CSV LOADER
// ============================================================

vector<vector<double>> loadCSVNumeric(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << "\n";
        return data;
    }
    
    string line;
    bool first_line = true;

    while (getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        
        stringstream ss(line);
        string cell;
        vector<double> row;
        int col_count = 0;

        while (getline(ss, cell, ',') && col_count < 10) {
            try {
                row.push_back(stod(cell));
                col_count++;
            } catch (...) {
                break;
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

vector<Point> loadCSVPoints(const string& filename) {
    vector<Point> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "ERROR: Could not open " << filename << endl;
        return data;
    }

    string line;
    getline(file, line); // skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> nums;

        while (getline(ss, value, ',')) {
            try {
                nums.push_back(stod(value));
            } catch (...) {}
        }

        if (nums.size() >= 2) {
            Point p;
            p.values = {nums[0], nums[1]};
            data.push_back(p);
        }
    }

    file.close();
    return data;
}

// ============================================================
// K-MEANS CLUSTERING
// ============================================================

double computeSSE(const vector<Point>& points, const vector<Point>& centers) {
    double sse = 0.0;
    for (const auto& p : points) {
        if (p.cluster >= 0 && p.cluster < (int)centers.size()) {
            sse += pow(distance(p, centers[p.cluster]), 2);
        }
    }
    return sse;
}

void kMeans(vector<Point>& points, int k, int max_iters, vector<Point>& centroids) {
    int n = points.size();
    int dim = points[0].values.size();

    centroids.resize(k);

    for (int i = 0; i < k; i++)
        centroids[i] = points[rand() % n];

    for (int iter = 0; iter < max_iters; iter++) {
        bool changed = false;

        for (int i = 0; i < n; i++) {
            double bestDist = numeric_limits<double>::max();
            int best = -1;

            for (int j = 0; j < k; j++) {
                double d = distance(points[i], centroids[j]);
                if (d < bestDist) {
                    bestDist = d;
                    best = j;
                }
            }

            if (points[i].cluster != best) {
                points[i].cluster = best;
                changed = true;
            }
        }

        if (!changed) break;

        vector<vector<double>> sums(k, vector<double>(dim, 0));
        vector<int> counts(k, 0);

        for (auto& p : points) {
            for (int d = 0; d < dim; d++)
                sums[p.cluster][d] += p.values[d];
            counts[p.cluster]++;
        }

        for (int j = 0; j < k; j++) {
            if (counts[j] == 0) continue;
            for (int d = 0; d < dim; d++)
                centroids[j].values[d] = sums[j][d] / counts[j];
        }
    }
}

ClusteringResult runKMeans(const string& filename, int k, int max_iters = 100) {
    cerr << "[K-Means] Loading data from " << filename << "\n";
    vector<Point> points = loadCSVPoints(filename);
    
    if (points.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return ClusteringResult{"K-Means", filename, 0, 0, 0, 0, 0};
    }

    cerr << "[K-Means] Loaded " << points.size() << " points\n";
    cerr << "[K-Means] Running K-Means with k=" << k << "\n";

    auto start = chrono::high_resolution_clock::now();

    vector<Point> centroids;
    kMeans(points, k, max_iters, centroids);

    auto end = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end - start).count();

    double sse = computeSSE(points, centroids);
    double memory_mb = (points.size() * sizeof(Point)) / (1024.0 * 1024.0);

    return ClusteringResult{"K-Means", filename, (int)points.size(), k, sse, runtime, memory_mb};
}

// ============================================================
// K-MEDOIDS CLUSTERING
// ============================================================

void kMedoids(vector<Point>& points, int k, int max_iters, vector<Point>& medoids) {
    int n = points.size();

    medoids.clear();
    for (int i = 0; i < k; i++)
        medoids.push_back(points[rand() % n]);

    for (int iter = 0; iter < max_iters; iter++) {
        bool changed = false;

        // Assignment step
        for (int i = 0; i < n; i++) {
            double bestDist = numeric_limits<double>::max();
            int best = -1;

            for (int j = 0; j < k; j++) {
                double d = distance(points[i], medoids[j]);
                if (d < bestDist) {
                    bestDist = d;
                    best = j;
                }
            }

            if (points[i].cluster != best) {
                points[i].cluster = best;
                changed = true;
            }
        }

        if (!changed) break;

        // Update step
        for (int j = 0; j < k; j++) {
            vector<Point> clusterPoints;

            for (auto& p : points)
                if (p.cluster == j)
                    clusterPoints.push_back(p);

            if (clusterPoints.empty()) continue;

            double bestCost = numeric_limits<double>::max();
            Point bestMedoid = clusterPoints[0];

            for (auto& candidate : clusterPoints) {
                double cost = 0;

                for (auto& p : clusterPoints)
                    cost += distance(candidate, p);

                if (cost < bestCost) {
                    bestCost = cost;
                    bestMedoid = candidate;
                }
            }

            medoids[j] = bestMedoid;
        }
    }
}

ClusteringResult runKMedoids(const string& filename, int k, int max_iters = 100) {
    cerr << "[K-Medoids] Loading data from " << filename << "\n";
    vector<Point> points = loadCSVPoints(filename);
    
    if (points.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return ClusteringResult{"K-Medoids", filename, 0, 0, 0, 0, 0};
    }

    cerr << "[K-Medoids] Loaded " << points.size() << " points\n";
    cerr << "[K-Medoids] Running K-Medoids with k=" << k << "\n";

    auto start = chrono::high_resolution_clock::now();

    vector<Point> medoids;
    kMedoids(points, k, max_iters, medoids);

    auto end = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end - start).count();

    double sse = computeSSE(points, medoids);
    double memory_mb = (points.size() * sizeof(Point)) / (1024.0 * 1024.0);

    return ClusteringResult{"K-Medoids", filename, (int)points.size(), k, sse, runtime, memory_mb};
}

// ============================================================
// HIERARCHICAL AGGLOMERATIVE CLUSTERING (HAC)
// ============================================================

enum LinkageType { SINGLE, COMPLETE, AVERAGE };

double euclidean(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double cluster_distance(
    const vector<int>& A,
    const vector<int>& B,
    const vector<vector<double>>& X,
    LinkageType linkage
) {
    if (linkage == SINGLE) {
        double min_dist = numeric_limits<double>::infinity();
        for (int i : A) {
            for (int j : B) {
                double dist = euclidean(X[i], X[j]);
                if (dist < min_dist) min_dist = dist;
            }
        }
        return min_dist;
    } else if (linkage == COMPLETE) {
        double max_dist = 0.0;
        for (int i : A) {
            for (int j : B) {
                double dist = euclidean(X[i], X[j]);
                if (dist > max_dist) max_dist = dist;
            }
        }
        return max_dist;
    } else { // AVERAGE
        double total = 0.0;
        int count = 0;
        for (int i : A) {
            for (int j : B) {
                total += euclidean(X[i], X[j]);
                count++;
            }
        }
        return total / count;
    }
}

double calculate_sse_hac(const vector<vector<double>>& X, const vector<int>& labels, int k) {
    vector<vector<double>> centroids(k, vector<double>(X[0].size(), 0.0));
    vector<int> cluster_sizes(k, 0);
    
    for (size_t i = 0; i < X.size(); i++) {
        int cluster_id = labels[i];
        for (size_t j = 0; j < X[i].size(); j++) {
            centroids[cluster_id][j] += X[i][j];
        }
        cluster_sizes[cluster_id]++;
    }
    
    for (int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            for (size_t j = 0; j < centroids[c].size(); j++) {
                centroids[c][j] /= cluster_sizes[c];
            }
        }
    }
    
    double total_sse = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
        int cluster_id = labels[i];
        double dist = euclidean(X[i], centroids[cluster_id]);
        total_sse += dist * dist;
    }
    
    return total_sse;
}

ClusteringResult runHAC(const string& filename, int k, const string& linkage_str) {
    LinkageType linkage;
    if (linkage_str == "single") {
        linkage = SINGLE;
    } else if (linkage_str == "complete") {
        linkage = COMPLETE;
    } else if (linkage_str == "average") {
        linkage = AVERAGE;
    } else {
        cerr << "Error: Invalid linkage type\n";
        return ClusteringResult{"HAC", filename, 0, 0, 0, 0, 0};
    }

    cerr << "[HAC] Loading data from " << filename << "\n";
    vector<vector<double>> X = loadCSVNumeric(filename);
    
    if (X.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return ClusteringResult{"HAC", filename, 0, 0, 0, 0, 0};
    }

    cerr << "[HAC] Loaded " << X.size() << " points with " << X[0].size() << " features\n";
    cerr << "[HAC] Running HAC with k=" << k << ", linkage=" << linkage_str << "\n";

    auto start_time = chrono::high_resolution_clock::now();

    int n = X.size();
    vector<vector<int>> clusters;
    clusters.reserve(n);
    for (int i = 0; i < n; i++) {
        clusters.push_back({i});
    }

    while ((int)clusters.size() > k) {
        double bestDist = numeric_limits<double>::infinity();
        int bestA = -1, bestB = -1;

        for (size_t i = 0; i < clusters.size(); i++) {
            for (size_t j = i + 1; j < clusters.size(); j++) {
                double dist = cluster_distance(clusters[i], clusters[j], X, linkage);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestA = i;
                    bestB = j;
                }
            }
        }

        clusters[bestA].insert(
            clusters[bestA].end(),
            clusters[bestB].begin(),
            clusters[bestB].end()
        );

        clusters.erase(clusters.begin() + bestB);
    }

    vector<int> labels(n, -1);
    for (int c = 0; c < (int)clusters.size(); c++) {
        for (int idx : clusters[c]) {
            labels[idx] = c;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end_time - start_time).count();
    
    double sse = calculate_sse_hac(X, labels, k);
    double memory_mb = (X.size() * X[0].size() * sizeof(double)) / (1024.0 * 1024.0);

    return ClusteringResult{"HAC", filename, (int)X.size(), k, sse, runtime, memory_mb};
}

// ============================================================
// PARTICLE SWARM OPTIMIZATION
// ============================================================

double randDouble(double low, double high) {
    return low + (high - low) * ((double) rand() / RAND_MAX);
}

double computeWCSS(const vector<vector<double>>& data, const vector<vector<double>>& centroids) {
    double wcss = 0.0;
    for (const auto& point : data) {
        double bestDist = 1e18;
        for (const auto& c : centroids) {
            double d = distance(point, c);
            bestDist = min(bestDist, d);
        }
        wcss += bestDist * bestDist;
    }
    return wcss;
}

struct Particle {
    vector<vector<double>> position;
    vector<vector<double>> velocity;
    vector<vector<double>> pBestPos;
    double pBestVal;
};

Particle initParticle(int k, int d, const vector<vector<double>>& data, double minVal, double maxVal) {
    Particle p;
    p.position.resize(k, vector<double>(d));
    p.velocity.resize(k, vector<double>(d, 0.0));
    p.pBestPos.resize(k, vector<double>(d));
    p.pBestVal = 1e18;

    for (int i = 0; i < k; i++)
        for (int j = 0; j < d; j++)
            p.position[i][j] = randDouble(minVal, maxVal);
    
    p.pBestPos = p.position;
    p.pBestVal = computeWCSS(data, p.position);

    return p;
}

vector<vector<double>> PSO_Clustering(const vector<vector<double>>& data, int k, int swarmSize, int iterations, double w = 0.7, double c1 = 1.5, double c2 = 1.5) {
    int d = data[0].size();

    double minVal = 1e18, maxVal = -1e18;
    for (auto& row : data)
        for (double v : row) {
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
        }

    vector<Particle> swarm;
    swarm.reserve(swarmSize);

    for (int i = 0; i < swarmSize; i++)
        swarm.push_back(initParticle(k, d, data, minVal, maxVal));

    vector<vector<double>> gBestPos = swarm[0].pBestPos;
    double gBestVal = swarm[0].pBestVal;

    for (auto& p : swarm)
        if (p.pBestVal < gBestVal) {
            gBestVal = p.pBestVal;
            gBestPos = p.pBestPos;
        }
    
    for (int iter = 0; iter < iterations; iter++) {
        for (auto& p : swarm) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < d; j++) {
                    double r1 = randDouble(0, 1);
                    double r2 = randDouble(0, 1);
                    
                    p.velocity[i][j] = w * p.velocity[i][j]
                                     + c1 * r1 * (p.pBestPos[i][j] - p.position[i][j])
                                     + c2 * r2 * (gBestPos[i][j] - p.position[i][j]);
                    
                    p.position[i][j] += p.velocity[i][j];
                }
            }

            double val = computeWCSS(data, p.position);

            if (val < p.pBestVal) {
                p.pBestVal = val;
                p.pBestPos = p.position;
            }

            if (val < gBestVal) {
                gBestVal = val;
                gBestPos = p.position;
            }
        }
    }

    return gBestPos;
}

ClusteringResult runPSO(const string& filename, int k, int swarmSize = 50, int iterations = 100) {
    cerr << "[PSO] Loading data from " << filename << "\n";
    vector<vector<double>> data = loadCSVNumeric(filename);
    
    if (data.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return ClusteringResult{"PSO", filename, 0, 0, 0, 0, 0};
    }

    cerr << "[PSO] Loaded " << data.size() << " points\n";
    cerr << "[PSO] Running PSO with k=" << k << ", swarmSize=" << swarmSize << ", iterations=" << iterations << "\n";

    auto start = chrono::high_resolution_clock::now();
    auto centroids = PSO_Clustering(data, k, swarmSize, iterations);
    auto end = chrono::high_resolution_clock::now();

    double runtime = chrono::duration<double>(end - start).count();
    double wcss = computeWCSS(data, centroids);
    double memory_mb = (data.size() * data[0].size() * sizeof(double)) / (1024.0 * 1024.0);

    return ClusteringResult{"PSO", filename, (int)data.size(), k, wcss, runtime, memory_mb};
}

// ============================================================
// MENU AND MAIN DRIVER
// ============================================================

void printMenu() {
    cout << "\n===== CLUSTERING ALGORITHM DRIVER =====\n";
    cout << "1. Run K-Means\n";
    cout << "2. Run K-Medoids\n";
    cout << "3. Run HAC (Hierarchical Agglomerative Clustering)\n";
    cout << "4. Run PSO (Particle Swarm Optimization)\n";
    cout << "5. Run All Algorithms\n";
    cout << "6. Exit\n";
    cout << "======================================\n";
    cout << "Select an option (1-6): ";
}

void printResults(const vector<ClusteringResult>& results) {
    cout << "\n===== RESULTS SUMMARY =====\n";
    cout << setw(12) << "Algorithm" 
         << setw(20) << "Dataset" 
         << setw(8) << "N" 
         << setw(5) << "K"
         << setw(12) << "SSE"
         << setw(12) << "Runtime (s)"
         << setw(12) << "Memory (MB)\n";
    cout << string(91, '-') << "\n";
    
    for (const auto& r : results) {
        cout << setw(12) << r.algorithm
             << setw(20) << (r.dataset.length() > 18 ? r.dataset.substr(r.dataset.length()-18) : r.dataset)
             << setw(8) << r.n
             << setw(5) << r.k
             << setw(12) << fixed << setprecision(2) << r.sse
             << setw(12) << fixed << setprecision(4) << r.runtime_seconds
             << setw(12) << fixed << setprecision(4) << r.memory_mb << "\n";
    }
    cout << string(91, '-') << "\n";
}

int main() {
    srand(time(0));

    vector<ClusteringResult> allResults;

    while (true) {
        printMenu();
        int choice;
        cin >> choice;
        cin.ignore();

        if (choice == 6) {
            cout << "Exiting...\n";
            break;
        }

        if (choice < 1 || choice > 6) {
            cout << "Invalid option. Please try again.\n";
            continue;
        }

        cout << "\nEnter dataset path (e.g., ../data/synthetic/blobs_100.csv): ";
        string dataset;
        getline(cin, dataset);

        cout << "Enter number of clusters (k): ";
        int k;
        cin >> k;
        cin.ignore();

        if (choice == 1) {
            cout << "Enter max iterations for K-Means (default 100): ";
            string iterStr;
            getline(cin, iterStr);
            int max_iters = iterStr.empty() ? 100 : stoi(iterStr);
            auto result = runKMeans(dataset, k, max_iters);
            allResults.push_back(result);
            cout << "\nK-Means completed!\n";
        }
        else if (choice == 2) {
            cout << "Enter max iterations for K-Medoids (default 100): ";
            string iterStr;
            getline(cin, iterStr);
            int max_iters = iterStr.empty() ? 100 : stoi(iterStr);
            auto result = runKMedoids(dataset, k, max_iters);
            allResults.push_back(result);
            cout << "\nK-Medoids completed!\n";
        }
        else if (choice == 3) {
            cout << "Enter linkage type (single/complete/average): ";
            string linkage;
            getline(cin, linkage);
            auto result = runHAC(dataset, k, linkage);
            allResults.push_back(result);
            cout << "\nHAC completed!\n";
        }
        else if (choice == 4) {
            cout << "Enter swarm size (default 50): ";
            string swarmStr;
            getline(cin, swarmStr);
            int swarmSize = swarmStr.empty() ? 50 : stoi(swarmStr);
            
            cout << "Enter iterations (default 100): ";
            string iterStr;
            getline(cin, iterStr);
            int iterations = iterStr.empty() ? 100 : stoi(iterStr);
            
            auto result = runPSO(dataset, k, swarmSize, iterations);
            allResults.push_back(result);
            cout << "\nPSO completed!\n";
        }
        else if (choice == 5) {
            cout << "Enter max iterations (default 100): ";
            string iterStr;
            getline(cin, iterStr);
            int max_iters = iterStr.empty() ? 100 : stoi(iterStr);
            
            cout << "\nRunning all algorithms...\n";
            allResults.push_back(runKMeans(dataset, k, max_iters));
            allResults.push_back(runKMedoids(dataset, k, max_iters));
            allResults.push_back(runHAC(dataset, k, "average"));
            allResults.push_back(runPSO(dataset, k, 50, 100));
            cout << "\nAll algorithms completed!\n";
        }

        if (!allResults.empty()) {
            printResults(allResults);
        }

        cout << "\nPress Enter to continue...";
        cin.ignore();
    }

    return 0;
}
