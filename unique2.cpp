#include <iostream>
#include <valarray>
#include <set>
#include <vector>

int main() {
  std::valarray<float> values = {1.1, 2.2, 3.3, 2.2, 4.4, 1.1, 5.5, 6.6};

  std::set<float> uniqueValuesSet;
  std::vector<float> uniqueValuesVector;

  for (float value : values) {
    if (uniqueValuesSet.insert(value).second) {
      uniqueValuesVector.push_back(value);
    }
  }

  int numUniqueValues = uniqueValuesVector.size();

  std::cout << "Number of unique values: " << numUniqueValues << std::endl;

  std::cout << "Unique values in order: ";
  for (float value : uniqueValuesVector) {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  return 0;
}
