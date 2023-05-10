#include <iostream>
#include <valarray>
#include <unordered_set>
#include <chrono>
int main() {
  std::valarray<float> values = {1.1, 2.2, 3.3, 2.2, 4.4, 1.1, 5.5, 6.6};

  std::unordered_set<float> uniqueValues;
  auto startTime = std::chrono::high_resolution_clock::now(); // Start the timer
  for (float value : values) {
    uniqueValues.insert(value);
  }
  auto endTime = std::chrono::high_resolution_clock::now(); // Stop the timer
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
  int numUniqueValues = uniqueValues.size();

  std::cout << "Number of unique values: " << numUniqueValues << std::endl;
  std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
  return 0;
}
