#ifndef PSO_H
#define PSO_H

#include <vector>
#include <string>
using namespace std;

/* Particle structre for PSO. */
struct Particle {
    vector<vector<double>> position;
    vector<vector<double>> velocity;
    vector<vector<double>> pBestPos;
    double pBestVal;
};

/* Compute Euclidean distance between two d-dimensional points. */
double distanceEuclid(const vector<double>& a, const vector<double>& b);

/* Compute WCSS (Within-Cluster Sum of Squares) for a set of centroids. */
double computeWCSS(const vector<vector<double>>& data, const vector<vector<double>>& centroids);

/* Generate a random double in [low, high].*/
double randDouble(double low, double high);

/* Initialize a particle with random centroid positions.
   Velocities start at zero. */
Particle initParticle(int k, int d, const vector<vector<double>>& data, double minVal, double maxVal);

/* Main PSO clustering function.
   Returns the best set of centroids found by the swarm. */
vector<vector<double>> PSO_Clustering(const vector<vector<double>>& data, int k, int swarmSize, int iterations, double w = 0.7, double c1 = 1.5, double c2 = 1.5);

// CSV loader
vector<vector<double>> loadCSV(const string& filename);

#endif