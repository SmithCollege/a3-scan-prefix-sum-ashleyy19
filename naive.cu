#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdio.h> 

// CUDA kernel for naive prefix sum
__global__ void naivePrefixSumGPU(int *arr, int *result, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    if (i < n) {
        int temp = 0;
        for (int j = 0; j <= i; j++) {
            temp += arr[j]; // Accumulate sum up to index i
        }
        result[i] = temp;
    }
}

int main() {
    // Initialize array size
    const int n = 100000;
    const int blockSize = 256; // Threads per block
    const int numBlocks = (n + blockSize - 1) / blockSize; // Number of blocks

    // Allocate and initialize host memory
    int *h_arr = new int[n];
    int *h_result = new int[n];
    srand(time(0));
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100; // Random values between 0 and 99
        h_result[i] = 0;
    }

    // Allocate device memory
    int *d_arr, *d_result;
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMalloc((void **)&d_result, n * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Timing the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);

    // Launch the naive prefix sum kernel
    naivePrefixSumGPU<<<numBlocks, blockSize>>>(d_arr, d_result, n);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to host
    cudaMemcpy(h_result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print runtime
    std::cout << "Naive GPU Runtime: " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Verify correctness for a small part of the array
    std::cout << "First 10 elements of the prefix sum:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_arr;
    delete[] h_result;
    cudaFree(d_arr);
    cudaFree(d_result);

    return 0;
}
