#include <iostream>
#include <future>
#include <chrono>

// notice to compile you need -pthread and c++17
// g++ -std=c++17 -pthread newcppconcurrency.cpp -o newcppconcurrency

// A function that sleeps for a given number of seconds and then returns the square of that number
int square(int x)
{
    std::this_thread::sleep_for(std::chrono::seconds(x));
    return x * x;
}

int main()
{
    // Launch a task concurrently using std::async
    std::future<int> f = std::async(square, 3);

    // Do some other work here...

    // Wait for the task to complete and get the result
    int result = f.get();

    std::cout << "Result: " << result << '\n';

    return 0;
}
