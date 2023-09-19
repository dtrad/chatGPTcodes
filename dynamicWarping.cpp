#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Define a function to calculate the Euclidean distance between two points
double euclideanDistance(double x1, double x2) {
    return std::abs(x1 - x2);
}

// Function to compute the DTW distance between two time series
double computeDTW(const std::vector<double>& series1, const std::vector<double>& series2) {
    int m = series1.size();
    int n = series2.size();

    // Create a 2D matrix for dynamic programming
    std::vector<std::vector<double>> dp(m, std::vector<double>(n, 0.0));

    // Initialize the first row and first column of the matrix
    dp[0][0] = euclideanDistance(series1[0], series2[0]);
    for (int i = 1; i < m; ++i) {
        dp[i][0] = dp[i - 1][0] + euclideanDistance(series1[i], series2[0]);
    }
    for (int j = 1; j < n; ++j) {
        dp[0][j] = dp[0][j - 1] + euclideanDistance(series1[0], series2[j]);
    }

    // Fill in the rest of the matrix using dynamic programming
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            double cost = euclideanDistance(series1[i], series2[j]);
            dp[i][j] = cost + std::min(std::min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);

        }
    }

    // The final DTW distance is in the bottom-right cell of the matrix
    return dp[m - 1][n - 1];
}

int main() {
    std::vector<double> timeSeries1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> timeSeries2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    double dtwDistance = computeDTW(timeSeries1, timeSeries2);

    std::cout << "DTW Distance: " << dtwDistance << std::endl;

    return 0;
}
