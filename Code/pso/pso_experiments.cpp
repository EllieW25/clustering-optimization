#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "pso.h"
using namespace std;

// Synthetic dataset generator
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

int main() {
    srand((unsigned)time(NULL));

    vector<int> sizes = {100, 200, 400, 800, 1600};

    ofstream out("../results/pso_runtime.csv");
    out << "n,runtime\n";

    int k = 3;
    int swarmSize = 50;
    int iterations = 100;

    for (int n : sizes) {
        cout << "\nRunning PSO on dataset size n = " << n << "...\n";

        vector<vector<double>> data = generateSynthetic(n);

        auto start = chrono::high_resolution_clock::now();
        auto centroids = PSO_Clustering(data, k, swarmSize, iterations);
        auto end = chrono::high_resolution_clock::now();

        double runtime = chrono::duration<double>(end - start).count();

        out << n << "," << runtime << "\n";
        cout << "Runtime: " << runtime << " seconds\n";
    }

    out.close();
    cout << "\nExperiment complete. Results saved to ../../results/pso_runtime.csv\n";

    return 0;
}