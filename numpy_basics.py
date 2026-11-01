# NumPy Basics - vectors, matrices and operations
# Practicing fundamental numpy operations

import numpy as np
import time

# creating arrays in different ways
a = np.zeros(4)
print(f"np.zeros(4): {a}, shape: {a.shape}, dtype: {a.dtype}")

a = np.random.random_sample(4)
print(f"random array: {a}, shape: {a.shape}")

a = np.arange(4.)
print(f"np.arange(4.): {a}")

a = np.array([5, 4, 3, 2])
print(f"np.array: {a}, dtype: {a.dtype}")

a = np.array([5.0, 4, 3, 2])
print(f"mixed types: {a}, dtype: {a.dtype}")  # notice it becomes float64


# indexing
print("\n--- Indexing ---")
a = np.arange(10)
print(f"a = {a}")
print(f"a[2] = {a[2]}")
print(f"a[-1] = {a[-1]}")  # last element

# what happens if we go out of bounds?
try:
    print(a[10])
except IndexError as e:
    print(f"a[10] gives IndexError: {e}")


# slicing
print("\n--- Slicing ---")
a = np.arange(10)
print(f"a = {a}")
print(f"a[2:7:1] = {a[2:7:1]}")   # start:stop:step
print(f"a[2:7:2] = {a[2:7:2]}")
print(f"a[3:]    = {a[3:]}")
print(f"a[:3]    = {a[:3]}")


# element-wise operations
print("\n--- Operations ---")
a = np.array([1, 2, 3, 4])
print(f"a = {a}")
print(f"-a = {-a}")
print(f"sum = {np.sum(a)}")
print(f"mean = {np.mean(a)}")
print(f"a**2 = {a**2}")

b = np.array([-1, -2, 3, 4])
print(f"a + b = {a + b}")

# this should fail - different shapes
c = np.array([1, 2])
try:
    print(a + c)
except ValueError as e:
    print(f"shape mismatch error: {e}")

print(f"5 * a = {5 * a}")


# dot product - my own implementation vs numpy
print("\n--- Dot Product ---")

def my_dot(a, b):
    """manual dot product with a loop"""
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])

print(f"my_dot: {my_dot(a, b)}")
print(f"np.dot: {np.dot(a, b)}")  # should be the same


# speed comparison - vectorization vs loops
print("\n--- Speed Test ---")
np.random.seed(1)
big_a = np.random.rand(10_000_000)
big_b = np.random.rand(10_000_000)

tic = time.time()
c = np.dot(big_a, big_b)
toc = time.time()
print(f"np.dot: {c:.4f}, took {1000*(toc-tic):.2f} ms")

tic = time.time()
c = my_dot(big_a, big_b)
toc = time.time()
print(f"my_dot: {c:.4f}, took {1000*(toc-tic):.2f} ms")
# numpy is way faster! vectorization matters

del big_a, big_b


# 2D arrays / matrices
print("\n--- Matrices ---")
a = np.arange(6).reshape(-1, 2)  # 3x2 matrix
print(f"a:\n{a}")
print(f"a[2,0] = {a[2,0]}")
print(f"a[2] = {a[2]}")  # whole row

# slicing a 2D array
a = np.arange(20).reshape(-1, 10)
print(f"\na (2x10):\n{a}")
print(f"a[0, 2:7:1] = {a[0, 2:7:1]}")
print(f"a[:, 2:7:1]:\n{a[:, 2:7:1]}")

print("\nDone!")
