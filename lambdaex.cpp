#include <iostream>

int main() {
    int a = 5;
    int b = 10;

    auto sum = [](int x, int y) -> int {
        return x + y;
    };

    int result = sum(a, b);
    std::cout << "Sum: " << result << std::endl;

    return 0;
}
