#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

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
// Average linkage distance between two clusters
// ------------------------------------------------------------
double cluster_distance(
    const vector<int>& A,
    const vector<int>& B,
    const vector<vector<double>>& X
) {
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

// ------------------------------------------------------------
// HAC (Average Linkage)
// ------------------------------------------------------------
vector<int> hac(const vector<vector<double>>& X, int k) {
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
                double dist = cluster_distance(clusters[i], clusters[j], X);
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

    return labels;
}

// ------------------------------------------------------------
// MAIN — standalone HAC runner
// Usage: ./hac <csv_path> <k>
// ------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./hac <csv_path> <k>\n";
        return 1;
    }

    string filename = argv[1];
    int k = stoi(argv[2]);

    cerr << "Loading dataset: " << filename << "\n";
    vector<vector<double>> X = load_csv(filename);
    
    if (X.empty()) {
        cerr << "Error: Failed to load dataset\n";
        return 1;
    }
    
    cerr << "Loaded " << X.size() << " points with " << X[0].size() << " features\n";
    cerr << "Running HAC with k = " << k << "\n";
    
    vector<int> labels = hac(X, k);

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
    outfile << "  \"linkage\": \"average\",\n";
    outfile << "  \"labels\": [";
    
    for (int i = 0; i < (int)labels.size(); i++) {
        if (i > 0) outfile << ",";
        outfile << labels[i];
    }
    
    outfile << "]\n";
    outfile << "}\n";
    outfile.close();
    
    cerr << "Results saved to " << output_file << "\n";

    return 0;
}
