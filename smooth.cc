#include <valarray>
#include <iostream>

// Smooth valarray using a simple moving average
std::valarray<double> smooth_valarray(const std::valarray<double>& input, int window_size) {
    if (window_size < 1 || window_size % 2 == 0) {
        throw std::invalid_argument("Window size must be a positive odd integer.");
    }

    std::valarray<double> output(input.size());
    int half_window = window_size / 2;

    for (std::size_t i = 0; i < input.size(); ++i) {
        int start = std::max<int>(0, i - half_window);
        int end = std::min<int>(input.size() - 1, i + half_window);

        double sum = 0.0;
        int count = 0;

        for (int j = start; j <= end; ++j) {
            sum += input[j];
            ++count;
        }

        output[i] = sum / count;
    }

    return output;
}


int main() {
    std::valarray<double> data = {1, 2, 3, 4, 5, 6, 7};
    int window = 3;

    std::valarray<double> smoothed = smooth_valarray(data, window);

    for (double v : smoothed) {
        std::cout << v << " ";
    }

    return 0;
}
