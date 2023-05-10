#include <iostream>
#include <valarray>
#include <set>
#include <vector>
#include <chrono>

int main() {
  std::valarray<float> values = {1.1, 2.2, 3.3, 2.2, 4.4, 1.1, 5.5, 6.6};

  std::set<float> uniqueValuesSet;
  std::vector<float> uniqueValuesVector;

  auto startTime = std::chrono::high_resolution_clock::now(); // Start the timer

  for (float value : values) {
    if (uniqueValuesSet.insert(value).second) {
      uniqueValuesVector.push_back(value);
    }
  }

  auto endTime = std::chrono::high_resolution_clock::now(); // Stop the timer
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

  int numUniqueValues = uniqueValuesVector.size();

  std::cout << "Number of unique values: " << numUniqueValues << std::endl;
  std::cout << "Unique values in order: ";
  for (float value : uniqueValuesVector) {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

  return 0;
}
