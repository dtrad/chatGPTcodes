#include <unordered_set>
#include <valarray>
#include <iostream>

struct FloatPairHash {
    size_t operator()(const std::pair<float, float>& pair) const {
        // Hash based on the combination of the two floats
        return std::hash<float>()(pair.first) ^ std::hash<float>()(pair.second);
    }
};

struct FloatPairEqual {
    bool operator()(const std::pair<float, float>& lhs, const std::pair<float, float>& rhs) const {
        // Compare both floats in the pair
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

class FdmGeom {
public:
    int unique(std::valarray<std::pair<float, float>>& values) {
        std::unordered_set<std::pair<float, float>, FloatPairHash, FloatPairEqual> uniqueValues;
        for (const auto& value : values) {
            uniqueValues.insert(value);
        }
        int numUniqueValues = uniqueValues.size();
        std::cout << "Number of unique values: " << numUniqueValues << std::endl;
        return numUniqueValues;
    }
};

int main() {
    FdmGeom geom;
    std::valarray<std::pair<float, float>> values{{1.1f, 2.2f}, {3.3f, 4.4f}, {1.1f, 2.2f}, {5.5f, 6.6f}};
    geom.unique(values);
    return 0;
}
