import time
import numpy as np

def prefix_sum_cpu(arr):
    n = len(arr)
    result = [0] * n
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = result[i - 1] + arr[i]
    return result

# Example usage
arr = np.random.randint(0, 100, size=100000)
start = time.time()
cpu_result = prefix_sum_cpu(arr)
end = time.time()
print("CPU Runtime:", end - start)
