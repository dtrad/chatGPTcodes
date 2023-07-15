#include <iostream>
#include <vector>
#include <algorithm>

std::vector<double> interpolate_vectors(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& xnew) {
    std::vector<double> ynew(xnew.size());
    for (size_t i = 0; i < xnew.size(); ++i) {
        auto it = std::lower_bound(x.begin(), x.end(), xnew[i]);
        if (it == x.begin()) {
            ynew[i] = y[0];
        } else if (it == x.end()) {
            ynew[i] = y[x.size() - 1];
        } else {
            size_t index = std::distance(x.begin(), it);
            double t = (xnew[i] - x[index - 1]) / (x[index] - x[index - 1]);
            ynew[i] = y[index - 1] + t * (y[index] - y[index - 1]);
        }
    }
    return ynew;
}

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {10, 15, 7, 12, 8};
    std::vector<double> xnew = {1.5, 2.5, 3.5};

    std::vector<double> ynew = interpolate_vectors(x, y, xnew);
    for (const auto& val : ynew) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
