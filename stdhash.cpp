#include <unordered_set>
#include <iostream>

struct MyStruct {
    int x;
    std::string y;
};

// Specializing std::hash for MyStruct
namespace std {
    template<>
    struct hash<MyStruct> {
        size_t operator()(const MyStruct& obj) const {
            size_t hashValue = std::hash<int>()(obj.x);
            hashValue ^= std::hash<std::string>()(obj.y);
            return hashValue;
        }
    };
}

int main() {
    std::unordered_set<MyStruct> mySet;

    MyStruct obj1{42, "Hello"};
    MyStruct obj2{123, "World"};

    mySet.insert(obj1);
    mySet.insert(obj2);

    for (const auto& obj : mySet) {
        std::cout << "x: " << obj.x << ", y: " << obj.y << std::endl;
    }

    return 0;
}
