#include <iostream>
#include <future>
#include <chrono>

// notice to compile you need -pthread and c++17
// g++ -std=c++17 -pthread newcppconcurrency.cpp -o newcppconcurrency

// This code defines a function square that sleeps for a given number of seconds 
// and then returns the square of that number. It then launches this function 
// concurrently using the std::async function and stores the result in a 
// std::future object. The std::future object can be used to wait for the task 
// to complete and retrieve the result.

// In this example, the main thread will continue to execute while the task launched 
// with std::async is running concurrently. When the task is finished, the main 
// thread will wait for the result using the get method of the std::future object.

// This is just a simple example of using concurrent execution in C++. 
// The C++17 <future> header provides a number of other functions and classes 
// for working with asynchronous tasks, such as std::promise, std::packaged_task, 
// and std::shared_future.


// A function that sleeps for a given number of seconds and then returns the square 
// of that number
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
