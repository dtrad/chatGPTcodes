#include <iostream>
#include <random>
#include <vector>

int main() {
  float N = 100.0f; // Upper limit of the range
  int arraySize = 10; // Number of elements in the array

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> intervalDist(1.0f, 10.0f);

  std::vector<float> floatArray;
  float currentVal = 0.0f;
  
  for (int i = 0; i < arraySize; ++i) {
    floatArray.push_back(currentVal);
    float interval = intervalDist(gen);    
    currentVal += std::floor(interval * 10.0f) / 10.0f;
    if (currentVal >= N)
      break;
  }

  std::cout << "Generated array: ";
  for (float value : floatArray) {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  return 0;
}
