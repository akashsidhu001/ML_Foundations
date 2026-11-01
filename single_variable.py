# Linear Regression with one variable
# predicting profits based on city population

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)

# dataset - population (10k) vs profit ($10k)
np.random.seed(42)
x_train = np.array([
    6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764,
    8.5781, 6.4862, 5.0546, 5.7107, 14.164, 5.734,  8.4084,
    5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708, 6.1891,
    20.27,  5.4901, 6.3261, 5.8282, 9.8172, 4.0959, 5.8598,
    5.7694, 6.4631, 5.3012, 7.4387, 5.0431, 8.1207, 7.3117,
    6.6803, 6.173,  8.7163, 7.7213, 5.0144, 5.1694, 6.8526,
    9.5228, 17.6316, 5.1906, 5.4893, 8.5396, 6.5851, 6.7772,
])
y_train = np.array([
    17.592,  9.1302, 13.662, 11.854,  6.8233, 11.886, 4.3483,
    12.0,    6.5987,  3.8166, 3.2522, 15.505,  3.1551, 7.2258,
    0.7162,  3.5129,  5.3048, 0.56,   3.6518, 5.3893, 3.1386,
    21.767,  4.263,   5.1875, 3.0825, 22.638,  4.2586, 6.8233,
    2.0403,  6.6886,  3.5129, 5.8648, 3.4989, 14.338, 8.1672,
    5.2377,  6.1422, 11.928, 11.587,  3.9166, 4.4127, 6.0639,
    11.104, 18.489,   3.1386, 7.0657, 13.417, 6.5159, 5.3893,
])

print(f"x shape: {x_train.shape}, y shape: {y_train.shape}")
print(f"m = {len(x_train)} training examples")

# plot the data first
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population of City in 10,000s")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "linear_regression_data.png"), dpi=120)
plt.show()


# cost function - J(w,b) = 1/2m * sum((f(x) - y)^2)
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i])**2
    total_cost = total_cost / (2 * m)
    return total_cost

# testing with some initial values
initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f"\nCost at w={initial_w}, b={initial_b}: {cost:.3f}")


# gradient computation
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# test gradient at zero
dj_dw, dj_db = compute_gradient(x_train, y_train, 0, 0)
print(f"Gradient at w=0, b=0 -> dj_dw={dj_dw:.4f}, dj_db={dj_db:.4f}")


# gradient descent
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

# run it
print("\nRunning gradient descent...")
initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01

w, b, J_history, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                                       compute_cost, compute_gradient, alpha, iterations)
print(f"w,b found by gradient descent: {w:.4f}, {b:.4f}")

# convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(J_history[:200])
ax1.set_title("Cost (first 200 iters)")
ax1.set_ylabel("Cost J"); ax1.set_xlabel("Iteration")
ax1.grid(alpha=0.3)

ax2.plot(J_history)
ax2.set_title("Cost (all iterations)")
ax2.set_ylabel("Cost J"); ax2.set_xlabel("Iteration")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "linear_regression_convergence.png"), dpi=120)
plt.show()

# plot the fit
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.figure(figsize=(8, 5))
plt.plot(x_train, predicted, c='b', label=f"fit: y = {w:.2f}x + {b:.2f}")
plt.scatter(x_train, y_train, marker='x', c='r', label="training data")
plt.title("Profits vs. Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population of City in 10,000s")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "linear_regression_fit.png"), dpi=120)
plt.show()

# predictions on new data
predict1 = 3.5 * w + b
print(f'\nFor population = 35,000, we predict a profit of ${predict1*10000:.2f}')

predict2 = 7.0 * w + b
print(f'For population = 70,000, we predict a profit of ${predict2*10000:.2f}')
