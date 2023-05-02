#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

int main()
{
    const int N = 10;
    thrust::host_vector<float> h_x(N);
    thrust::host_vector<float> h_y(N);
    thrust::device_vector<float> d_x(N);
    thrust::device_vector<float> d_y(N);

    // Initialize input vectors on host
    for (int i = 0; i < N; i++)
    {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i + 1);
    }

    // Transfer input vectors to device
    d_x = h_x;
    d_y = h_y;

    // Perform vector addition on device
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), thrust::plus<float>());

    // Transfer result back to host
    h_y = d_y;

    // Print result
    for (int i = 0; i < N; i++)
    {
        std::cout << h_y[i] << std::endl;
    }

    return 0;
}
