#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
// compile with g++ -o matchseries matchseries.cpp --std=c++17
std::vector<double> matchFilter(std::vector<double> signal, std::vector<double> templ) {
    int signalSize = signal.size();
    int templSize = templ.size();
    int resultSize = signalSize - templSize + 1;
    std::vector<double> result(resultSize);

    for(int i = 0; i < resultSize; i++) {
        double sum = 0;
        for(int j = 0; j < templSize; j++) {
            sum += signal[i+j] * templ[j];
        }
        result[i] = sum;
    }

    return result;
}

int main(){
    std::vector<double> signal = {1, 2, 3, 4, 5};
    std::vector<double> templ = {2, 3};
    std::vector<double> result = matchFilter(signal, templ);
    auto max_it = std::max_element(result.begin(), result.end());
    int max_index = std::distance(result.begin(), max_it);
    std::cout << max_index << std::endl; 
    return 0;
}
