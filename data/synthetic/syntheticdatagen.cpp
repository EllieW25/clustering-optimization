/*
HOW TO RUN
compile with g++ and run the executable. You will be prompted
for n (number of points) k (number of clusters) and d (dimensions)
*/

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

// -------------------- MAIN --------------------
int main() {
    srand(time(0));

    int n, k, dim;

    cout << "Enter number of samples (n): ";
    cin >> n;

    cout << "Enter number of clusters (k): ";
    cin >> k;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');  // consume newline

    cout << "Enter number of dimensions (dim) [default 10]: ";
    string dim_input;
    getline(cin, dim_input);
    if (dim_input.empty()) {
        dim = 10;
    } else {
        dim = stoi(dim_input);
    }

    vector<Point> data = generateData(n, dim, k);

    string filename = "blobs_" + to_string(n) + ".csv";
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        return 1;
    }

    // Write header
    file << "feature0";
    for (int d = 1; d < dim; d++) {
        file << ",feature" << d;
    }
    file << ",label" << endl;

    // Write data
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < dim; d++) {
            file << fixed << setprecision(2) << data[i].values[d];
            if (d < dim - 1) file << ",";
        }
        file << "," << (rand() % k) << endl;  // random label
    }

    file.close();
    cout << "Generated " << filename << " with " << n << " points, " << dim << " dimensions, " << k << " clusters." << endl;

    return 0;
}
