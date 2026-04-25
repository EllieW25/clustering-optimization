#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;
//HOW TO RUN
//compile with g++ and run. Example executions below
//./hac ../../data/synthetic/blobs_100.csv 5 average
//./hac path/to/csv [#clusters] [linkage: single|complete|average]
// ------------------------------------------------------------
// Linkage types
// ------------------------------------------------------------
enum LinkageType {
    SINGLE,
    COMPLETE,
    AVERAGE
};

// ------------------------------------------------------------
// CSV LOADER
// ------------------------------------------------------------
vector<vector<double>> load_csv(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << "\n";
        return data;
    }
    
    string line;
    bool first_line = true;

    while (getline(file, line)) {
        // Skip header row
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

    return data;
}

// ------------------------------------------------------------
// Euclidean distance
// ------------------------------------------------------------
double euclidean(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// ------------------------------------------------------------
// Linkage distance between two clusters
// ------------------------------------------------------------
double cluster_distance(
    const vector<int>& A,
    const vector<int>& B,
    const vector<vector<double>>& X,
    LinkageType linkage
) {
    if (linkage == SINGLE) {
        // Single linkage: minimum distance
        double min_dist = numeric_limits<double>::infinity();
        for (int i : A) {
            for (int j : B) {
                double dist = euclidean(X[i], X[j]);
                if (dist < min_dist) min_dist = dist;
            }
        }
        return min_dist;
    } else if (linkage == COMPLETE) {
        // Complete linkage: maximum distance
        double max_dist = 0.0;
        for (int i : A) {
            for (int j : B) {
                double dist = euclidean(X[i], X[j]);
                if (dist > max_dist) max_dist = dist;
            }
        }
        return max_dist;
    } else if (linkage == AVERAGE) {
        // Average linkage: average distance
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
    
    return 0.0; // Should not reach here
}

// ------------------------------------------------------------
// Calculate SSE (Sum of Squared Errors)
// ------------------------------------------------------------
double calculate_sse(const vector<vector<double>>& X, const vector<int>& labels, int k) {
    vector<vector<double>> centroids(k, vector<double>(X[0].size(), 0.0));
    vector<int> cluster_sizes(k, 0);
    
    // Calculate centroids
    for (size_t i = 0; i < X.size(); i++) {
        int cluster_id = labels[i];
        for (size_t j = 0; j < X[i].size(); j++) {
            centroids[cluster_id][j] += X[i][j];
        }
        cluster_sizes[cluster_id]++;
    }
    
    // Normalize centroids
    for (int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            for (size_t j = 0; j < centroids[c].size(); j++) {
                centroids[c][j] /= cluster_sizes[c];
            }
        }
    }
    
    // Calculate total SSE
    double total_sse = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
        int cluster_id = labels[i];
        double dist = euclidean(X[i], centroids[cluster_id]);
        total_sse += dist * dist;
    }
    
    return total_sse;
}

// ------------------------------------------------------------
// HAC Result struct
// ------------------------------------------------------------
struct HACResult {
    vector<int> labels;
    double runtime_seconds;
    double memory_mb;
    double average_sse;
};

// ------------------------------------------------------------
// HAC (Hierarchical Agglomerative Clustering)
// ------------------------------------------------------------
HACResult hac(const vector<vector<double>>& X, int k, LinkageType linkage) {
    auto start_time = chrono::high_resolution_clock::now();
    
    int n = X.size();

    // Start with each point as its own cluster
    vector<vector<int>> clusters;
    clusters.reserve(n);
    for (int i = 0; i < n; i++) {
        clusters.push_back({i});
    }

    // Merge until k clusters remain
    int merges = 0;
    int total_merges = n - k;
    while ((int)clusters.size() > k) {
        if (merges % 100 == 0) {
            cerr << "  Merging... " << clusters.size() << " clusters remaining\n";
            cerr.flush();
        }
        
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

        // Merge B into A
        clusters[bestA].insert(
            clusters[bestA].end(),
            clusters[bestB].begin(),
            clusters[bestB].end()
        );

        // Remove B
        clusters.erase(clusters.begin() + bestB);
        merges++;
    }

    // Produce final labels
    vector<int> labels(n, -1);
    for (int c = 0; c < (int)clusters.size(); c++) {
        for (int idx : clusters[c]) {
            labels[idx] = c;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end_time - start_time).count();
    
    // Calculate SSE
    double sse = calculate_sse(X, labels, k);
    
    // Estimate memory usage (approximate)
    // Data: n * features * 8 bytes (double)
    // Clusters: sum of all indices stored
    double data_memory = (n * X[0].size() * sizeof(double)) / (1024.0 * 1024.0);
    double clusters_memory = (n * sizeof(int) * 2) / (1024.0 * 1024.0); // rough estimate
    double total_memory = data_memory + clusters_memory;

    return HACResult{labels, runtime, total_memory, sse};
}

// ------------------------------------------------------------
// MAIN — standalone HAC runner
// Usage: ./hac <csv_path> <k> <linkage>
// Linkage options: single, complete, average
// ------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: ./hac <csv_path> <k> <linkage>\n";
        cout << "Linkage options: single, complete, average\n";
        return 1;
    }

    string filename = argv[1];
    int k = stoi(argv[2]);
    string linkage_str = argv[3];
    
    LinkageType linkage;
    if (linkage_str == "single") {
        linkage = SINGLE;
    } else if (linkage_str == "complete") {
        linkage = COMPLETE;
    } else if (linkage_str == "average") {
        linkage = AVERAGE;
    } else {
        cerr << "Error: Invalid linkage type '" << linkage_str << "'\n";
        cerr << "Valid options: single, complete, average\n";
        return 1;
    }

    cerr << "Loading dataset: " << filename << "\n";
    vector<vector<double>> X = load_csv(filename);
    
    if (X.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return 1;
    }
    
    cerr << "Loaded " << X.size() << " points with " << X[0].size() << " features\n";
    cerr << "Running HAC with k = " << k << " and " << linkage_str << " linkage\n";
    
    HACResult result = hac(X, k, linkage);

    // Ask for output filename
    string output_name;
    cerr << "Enter output filename (without extension): ";
    cerr.flush();
    getline(cin, output_name);
    
    string output_file = output_name + ".json";

    // Write JSON to file
    ofstream outfile(output_file);
    outfile << "{\n";
    outfile << "  \"algorithm\": \"HAC\",\n";
    outfile << "  \"k\": " << k << ",\n";
    outfile << "  \"n\": " << X.size() << ",\n";
    outfile << "  \"linkage\": \"" << linkage_str << "\",\n";
    outfile << "  \"runtime_seconds\": " << result.runtime_seconds << ",\n";
    outfile << "  \"memory_mb\": " << result.memory_mb << ",\n";
    outfile << "  \"average_sse\": " << result.average_sse << ",\n";
    outfile << "  \"labels\": [";
    
    for (int i = 0; i < (int)result.labels.size(); i++) {
        if (i > 0) outfile << ",";
        outfile << result.labels[i];
    }
    
    outfile << "]\n";
    outfile << "}\n";
    outfile.close();
    
    cerr << "Results saved to " << output_file << "\n";
    cerr << "Runtime: " << result.runtime_seconds << " seconds\n";
    cerr << "Memory: " << result.memory_mb << " MB\n";
    cerr << "Average SSE: " << result.average_sse << "\n";

    return 0;
}
