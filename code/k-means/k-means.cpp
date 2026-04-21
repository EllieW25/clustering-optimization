#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

using namespace std;

// -------------------- Point --------------------
struct Point {
    vector<double> values;
    int cluster = -1;
};

// -------------------- Distance --------------------
double distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (int i = 0; i < a.values.size(); i++) {
        sum += pow(a.values[i] - b.values[i], 2);
    }
    return sqrt(sum);
}

// -------------------- SSE --------------------
double computeSSE(const vector<Point>& points, const vector<Point>& centers) {
    double sse = 0.0;
    for (const auto& p : points) {
        sse += pow(distance(p, centers[p.cluster]), 2);
    }
    return sse;
}

// -------------------- DATA LOADING --------------------
vector<Point> loadCSV(string filename) {
    vector<Point> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "ERROR: Could not open " << filename << endl;
        return data;
    }

    string line;
    getline(file, line); // header

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> nums;

        while (getline(ss, value, ',')) {
            try {
                nums.push_back(stod(value));
            } catch (...) {}
        }

        if (nums.size() != 3) continue;

        Point p;
        p.values = {nums[1], nums[2]}; // Income + Score
        data.push_back(p);
    }

    return data;
}

// -------------------- SYNTHETIC DATA --------------------
vector<Point> generateData(int n, int dim, int k) {
    vector<Point> data;
    vector<vector<double>> centers(k, vector<double>(dim));

    for (int i = 0; i < k; i++)
        for (int d = 0; d < dim; d++)
            centers[i][d] = rand() % 100;

    for (int i = 0; i < n; i++) {
        Point p;
        int c = rand() % k;

        for (int d = 0; d < dim; d++) {
            double noise = (rand() % 10) - 5;
            p.values.push_back(centers[c][d] + noise);
        }

        data.push_back(p);
    }

    return data;
}

// =======================================================
// K-MEANS 
// =======================================================
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


// -------------------- EXPERIMENT RUNNER --------------------
void runExperiment(ofstream& file, vector<Point> baseData, string label,
                   int k, int max_iters, int trials,
                   void (*clusteringFunc)(vector<Point>&, int, int, vector<Point>&)) {

    double total_sse = 0;
    double total_time = 0;

    for (int t = 0; t < trials; t++) {

        vector<Point> data = baseData;
        vector<Point> centers;

        auto start = chrono::high_resolution_clock::now();

        clusteringFunc(data, k, max_iters, centers);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        total_time += elapsed.count();
        total_sse += computeSSE(data, centers);
    }

    cout << label << " | SSE=" << total_sse / trials
         << " | Time=" << total_time / trials << endl;

    file << label << ","
         << baseData.size() << ","
         << k << ","
         << total_sse / trials << ","
         << total_time / trials << "\n";
}

// -------------------- MAIN --------------------
int main() {
    srand(time(0));

    int k = 3;
    int max_iters = 100;
    int trials = 10;

    vector<int> sizes = {100, 200, 400, 800, 1600};

    // ================= K-MEANS =================
    ofstream kmFile("kmeans_results.csv");
    kmFile << "dataset,n,k,avg_sse,avg_runtime\n";

    for (int n : sizes) {
        vector<Point> data = generateData(n, 2, k);
        runExperiment(kmFile, data, "synthetic", k, max_iters, trials, kMeans);
    }

    vector<Point> realData = loadCSV("mall.csv");
    if (!realData.empty())
        runExperiment(kmFile, realData, "mall_dataset", k, max_iters, trials, kMeans);

    kmFile.close();

}
