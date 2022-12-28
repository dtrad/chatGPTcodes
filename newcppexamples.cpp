#include <iostream>
#include <algorithm>
#include <vector>
#include <tuple>

// examples using c++ 17 from chatGPT 

// fold expressions allow you to apply a binary operator to a parameter pack
template <typename... Args>
void print(Args... args){
    (std::cout << ... << args) << '\n';
}


#include <type_traits>
template <typename T>
void foo(T t){
    if constexpr (std::is_same_v<T, int>)
    {
        std::cout << "input was integer" << std::endl;// Code executed only if T is int
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        std::cout << "input was double" << std::endl; // Code executed only if T is double
    }
    // ...
}


#include <string_view>



int main()
{
    /*
    Lambda functions are a useful feature in C++ that allow you to define 
    anonymous functions inline with your code. They are often used as arguments 
    to algorithms or as simple one-line functions.
    */

    // Create a vector of integers
    std::vector<int> v = {1, 2, 3, 4, 5};

    // Use std::for_each to apply a lambda function to each element in the vector
    std::for_each(v.begin(), v.end(), [](int x) {
        std::cout << x << " ";
    });

    /*
    Structured binding declarations
    Structured binding declarations allow you to bind multiple variables to the 
    elements of a tuple or struct, like this:
    */

    
    std::tuple<int, std::string, double> t(1, "hello", 3.14);
    auto [a, b, c] = t;  // a is 1, b is "hello", c is 3.14
    print('\n',a,b,c);

    // Fold expressions allow you to apply a binary operator to a parameter pack, like this:
    print(1, 2, 3, 4, 5);  // Outputs "1 2 3 4 5\n" (see function print above)

    // const expressions
    foo(1);  // Executes the first branch of the if constexpr statement
    foo(3.14);  // Executes the second branch of the if constexpr statement



    // std::string_view is a lightweight alternative to std::string that provides a view
    // into a string without owning the underlying memory. It can be used like this:
    std::string s = "hello world";
    std::string_view sv = s;  // sv points to the same data as s
    std::cout << sv << '\n';  // Outputs "hello world"

    // Range-based for loops are a convenient way to iterate over the elements of a 
    // container in C++. They are especially useful when you don't need to access 
    // the iterator or index of the current element, and you just want to operate 
    // on the element itself.
    
    // Create a vector of integers
    std::vector<int> v2 = {1, 2, 3, 4, 5};
    // Use a range-based for loop to iterate over the vector
    for (int x : v2){
        std::cout << x << " ";
    }

    return 0;
}
