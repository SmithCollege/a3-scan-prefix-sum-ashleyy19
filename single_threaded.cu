#include <iostream>
#include <vector>
#include <chrono> // For measuring runtime

//compute prefix sum on the CPU
void prefixSumCPU(const std::vector<int> &arr, std::vector<int> &result) {
    int n = arr.size();
    result[0] = arr[0]; // Initialize the first element
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + arr[i]; // Compute prefix sum iteratively
    }
}

int main() {
    // Size of the array
    const int n = 100000;

    // Generate random input data
    std::vector<int> arr(n);
    std::vector<int> result(n, 0); // Output array initialized to zero
    srand(time(0));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }

    // Measure CPU prefix sum runtime
    auto start = std::chrono::high_resolution_clock::now();
    prefixSumCPU(arr, result);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "CPU Runtime: " << elapsed.count() << " seconds" << std::endl;

    // Verify the result (optional, print first 10 elements)
    std::cout << "First 10 elements of the prefix sum:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
