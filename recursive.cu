#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// CUDA kernel for recursive doubling
__global__ void recursiveDoublingKernel(int *arr, int *result, int step, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= step && i < n) {
        result[i] += result[i - step];
    }
}

int main() {
    // Initialize array size
    const int n = 1 << 20; // Example size: 2^20 (1 million elements)
    const int blockSize = 256; // Threads per block
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate and initialize host memory
    int *h_arr = new int[n];
    int *h_result = new int[n];
    srand(time(0));
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100; // Random values between 0 and 99
        h_result[i] = h_arr[i];
    }

    // Allocate device memory
    int *d_result;
    cudaMalloc((void **)&d_result, n * sizeof(int));
    cudaMemcpy(d_result, h_result, n * sizeof(int), cudaMemcpyHostToDevice);

    // Timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);

    // Recursive doubling
    int step = 1;
    while (step < n) {
        recursiveDoublingKernel<<<numBlocks, blockSize>>>(d_result, d_result, step, n);
        cudaDeviceSynchronize(); // Ensure all threads finish before next iteration
        step *= 2;
    }

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print runtime
    std::cout << "Recursive Doubling GPU Runtime: " << milliseconds << " ms" << std::endl;

    // Verify correctness for a small part of the array
    for (int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_arr;
    delete[] h_result;
    cudaFree(d_result);

    return 0;
}
