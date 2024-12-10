from numba import cuda
import numpy as np
import time

@cuda.jit
def naive_prefix_sum_gpu(arr, result):
    i = cuda.grid(1)
    if i < arr.size:
        temp = 0
        for j in range(i + 1):
            temp += arr[j]
        result[i] = temp

# Example usage
n = 100000
arr = np.random.randint(0, 100, size=n).astype(np.int32)
result = np.zeros(n, dtype=np.int32)
d_arr = cuda.to_device(arr)
d_result = cuda.to_device(result)

start = time.time()
naive_prefix_sum_gpu[n // 256, 256](d_arr, d_result)
cuda.synchronize()
end = time.time()

print("Naive GPU Runtime:", end - start)
