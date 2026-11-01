# Multiple Linear Regression with Feature Normalization
# house price prediction using multiple features

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)

# house data - features: size(sqft), bedrooms, floors, age
# target: price in 1000s
X_train = np.array([
    [2104, 5, 1, 45], [1416, 3, 2, 40], [852,  2, 1, 35],
    [1534, 3, 2, 30], [1940, 4, 2, 20], [2489, 4, 1, 10],
    [1416, 2, 1, 32], [1203, 3, 1, 42], [1590, 3, 1, 25],
    [1320, 2, 2, 28], [1808, 3, 1, 15], [2150, 4, 2,  5],
    [1100, 2, 1, 40], [1695, 3, 2, 22], [2300, 4, 1, 12],
    [1860, 3, 2, 18], [1002, 2, 1, 50], [1450, 3, 1, 35],
    [1700, 3, 2, 20], [2200, 4, 1,  8],
])
y_train = np.array([
    460, 232, 178, 270, 380, 485, 220, 205, 290, 250,
    340, 465, 175, 310, 445, 355, 155, 260, 325, 450,
])
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

# visualize each feature vs price
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "multi_feature_scatter.png"), dpi=120)
plt.show()


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))
        if i % math.ceil(num_iters/10) == 0:
            print(f"  Iteration {i:5d}: Cost {J_history[-1]:12.2f}")

    return w, b, J_history


# trying different learning rates without normalization
print("\n--- Trying different learning rates (no normalization) ---")
for alpha_val in [9.9e-7, 9e-7, 1e-7]:
    print(f"\nalpha = {alpha_val}")
    w_init = np.zeros(X_train.shape[1])
    gradient_descent(X_train, y_train, w_init, 0., alpha_val, 10)
# learning rate matters a lot when features aren't normalized!


# z-score normalization
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)       # mean of each column
    sigma = np.std(X, axis=0)     # std of each column
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)

X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"\nMeans: {X_mu}")
print(f"Stds: {X_sigma}")
print(f"Peak-to-peak raw: {np.ptp(X_train,axis=0)}")
print(f"Peak-to-peak normalized: {np.ptp(X_norm,axis=0).round(2)}")

# distribution after normalization
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    ax[i].hist(X_norm[:,i], bins=8, edgecolor='black', alpha=0.7)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count")
fig.suptitle("distribution of features after normalization")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "normalization_hist.png"), dpi=120)
plt.show()

# now gradient descent works much better with normalized features
print("\n--- Gradient descent with normalized features ---")
w_norm, b_norm, hist = gradient_descent(X_norm, y_train,
                                         np.zeros(X_norm.shape[1]), 0.,
                                         alpha=1.0e-1, num_iters=1000)

print(f"\nw: {w_norm.round(2)}, b: {b_norm:.2f}")

# convergence plot
plt.figure(figsize=(7, 4))
plt.plot(hist)
plt.title("Cost convergence (normalized features)")
plt.xlabel("iteration"); plt.ylabel("cost")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "multi_convergence.png"), dpi=120)
plt.show()

# predictions vs actual
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i], yp, color='orange', label='predict')
ax[0].set_ylabel("Price"); ax[0].legend()
fig.suptitle("target vs prediction using z-score normalized model")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "multi_predictions.png"), dpi=120)
plt.show()

# predict for a new house
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"\npredicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
