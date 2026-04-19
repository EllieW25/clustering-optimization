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
