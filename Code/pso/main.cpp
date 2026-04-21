#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "pso.h"
using namespace std;

// CSV loader is implemented in pso.cpp.
vector<vector<double>> loadCSV(const string& filename);

// Synthetic generator reused for testing/graphs
vector<vector<double>> generateSynthetic(int n) {
    vector<vector<double>> data(n, vector<double>(2));

    for (int i = 0; i < n; i++) {
        if (i < n/3) {
            data[i][0] = randDouble(0, 5);
            data[i][1] = randDouble(0, 5);
        } else if (i < 2*n/3) {
            data[i][0] = randDouble(10, 15);
            data[i][1] = randDouble(10, 15);
        } else {
            data[i][0] = randDouble(20, 25);
            data[i][1] = randDouble(0, 5);
        }
    }

    return data;
}

/* Simple test program:
   - Generates synthetic 2D data in 3 clusters
   - Runs PSO to find centroids */
int main() {
    srand((unsigned)time(NULL));

    // SELECT DATASET HERE
    bool useMallDataset = true;   // true = Mall CSV, false = synthetic blobs

    vector<vector<double>> data;

    if (useMallDataset) {
        cout << "Loading Mall Customers dataset...\n";
        data = loadCSV("../data/real/mall.csv");

        if (data.empty()) {
            cout << "Failed to load mall.csv\n";
            return 1;
        }

        cout << "Loaded " << data.size() << " rows with "
             << data[0].size() << " features.\n";
    } else {
        cout << "Generating synthetic blob dataset...\n";
        int n = 1000;
        data = generateSynthetic(n);
        cout << "Synthetic dataset generated: " << n << " points.\n";
    }

    int k = useMallDataset ? 5 : 3;  // typical: 5 for Mall, 3 for synthetic
    int swarmSize = 50;
    int iterations = 100;

    // Run PSO clustering
    auto centroids = PSO_Clustering(data, k, swarmSize, iterations);

    // Print final centroids
    cout << "\nFinal Centroids:\n";
    for (auto& c : centroids) {
        for (double v : c) cout << v << " ";
        cout << "\n";
    }

    return 0;
}