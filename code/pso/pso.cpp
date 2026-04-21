#include "pso.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

/* Compute Euclidean distance between two points.*/
double distanceEuclid(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); i++)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

/* Compute WCSS for a given set of centroids..
   For each point, find the nearest centroid and add squared distance. */
double computeWCSS(const vector<vector<double>>& data, const vector<vector<double>>& centroids) {
    double wcss = 0.0;

    for (const auto& point : data) {
        double bestDist = 1e18;
        // Find nearest centroid
        for (const auto& c : centroids){
            double d = distanceEuclid(point, c);
            bestDist = min(bestDist, d);
        }
        wcss += bestDist * bestDist;
    }
    return wcss;
}  

// Generate a random double in [low, high].
double randDouble(double low, double high) {
    return low + (high - low) * ((double) rand() / RAND_MAX);
}

/* Initialize a particle:
   - Random centroid positions
   - Zero velocity
   - pBest = initial position */
Particle initParticle(int k, int d, const vector<vector<double>>& data, double minVal, double maxVal){
    Particle p;

    p.position.resize(k, vector<double>(d));
    p.velocity.resize(k, vector<double>(d, 0.0));
    p.pBestPos.resize(k, vector<double>(d));
    p.pBestVal = 1e18;

    // Randomly initialize centroids
    for (int i = 0; i < k; i++)
        for (int j = 0; j < d; j++)
            p.position[i][j] = randDouble(minVal, maxVal);
    
    // Initial personal best = intital position
    p.pBestPos = p.position;
    p.pBestVal = computeWCSS(data, p.position);

    return p;
}

/* Main PSO algorithm for clustering.
   Each particle represents a set of k centroids.
   The swarm searches for the centroid set that minimizes WCSS. */

vector<vector<double>> PSO_Clustering(const vector<vector<double>>& data, int k, int swarmSize, int iterations, double w, double c1, double c2) {
    int d = data[0].size();

    // Determine min/max values for centroid initialization
    double minVal = 1e18, maxVal = -1e18;
    for (auto& row : data)
        for (double v : row) {
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
        }
    // Create swarm
    vector<Particle> swarm;
    swarm.reserve(swarmSize);

    for (int i = 0; i < swarmSize; i++)
        swarm.push_back(initParticle(k, d, data, minVal, maxVal));

    // Initialize global best
    vector<vector<double>> gBestPos = swarm[0].pBestPos;
    double gBestVal = swarm[0].pBestVal;

    for (auto& p : swarm)
        if (p.pBestVal < gBestVal) {
            gBestVal = p.pBestVal;
            gBestPos = p.pBestPos;
        }
    
    // Main PSO loop
    for (int iter = 0; iter < iterations; iter++) {
        for (auto& p : swarm) {
            // Update velocity and position for each centroid
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < d; j++) {

                    double r1 = randDouble(0, 1);
                    double r2 = randDouble(0, 1);

                    // PSO velocity update equation
                    p.velocity[i][j] = w * p.velocity[i][j] + c1 * r1 * (p.pBestPos[i][j] - p.position[i][j]) + c2 * r2 * (gBestPos[i][j] - p.position[i][j]);

                    // Update centroid position
                    p.position[i][j] += p.velocity[i][j];
                }
            }

            // Evaluate new WCSS
            double currVal = computeWCSS(data, p.position);

            // Update personal best
            if (currVal < p.pBestVal) {
                p.pBestVal = currVal;
                p.pBestPos = p.position;
            }

            // Update global best
            if (currVal < gBestVal) {
                gBestVal = currVal;
                gBestPos = p.position;
            }
        }
        // Print convergence progress
        cout << "Iteration " << iter << "  Best WCSS = " << gBestVal << "\n";
    }
    return gBestPos;
}

vector<vector<double>> loadCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            if (!value.empty())
                row.push_back(stod(value));
        }

        if (!row.empty())
            data.push_back(row);
    }

    return data;
}
