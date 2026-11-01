# Feature Engineering - polynomial features and feature combining
# also comparing squared error vs cross-entropy loss

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def zscore_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1  # avoid dividing by zero
    return (X - mu) / sigma, mu, sigma

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.
    for i in range(m):
        cost += (np.dot(X[i], w) + b - y[i])**2
    return cost / (2*m)

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]
        dj_db += err
    return dj_dw/m, dj_db/m

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))
        if i % math.ceil(num_iters/5) == 0:
            print(f"  Iteration {i:5d}: Cost {J_history[-1]:.6f}")

    return w, b, J_history


# ====== 1. fitting cos(x/2) with polynomial features ======

print("1. Polynomial fit of cos(x/2)")
print("trying degree 2, 6, and 14 to see under/good/overfit\n")

np.random.seed(0)
x = np.arange(0, 20, 1, dtype=float)
y = np.cos(x/2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [2, 6, 14]
titles = ["Degree 2 (underfit)", "Degree 6 (good fit)", "Degree 14 (overfit)"]

for ax, deg, title in zip(axes, degrees, titles):
    # make polynomial features
    X_poly = np.column_stack([x**d for d in range(1, deg+1)])
    X_norm, mu, sigma = zscore_normalize(X_poly)

    print(f"--- Degree {deg} ---")
    w_init = np.zeros(deg)
    w, b, _ = gradient_descent(X_norm, y, w_init, 0., alpha=1e-1, num_iters=10000)

    # smooth curve for plotting
    x_plot = np.linspace(0, 20, 200)
    X_plot = np.column_stack([x_plot**d for d in range(1, deg+1)])
    X_plot_norm = (X_plot - mu) / sigma
    y_plot = np.dot(X_plot_norm, w) + b

    ax.scatter(x, y, c='red', marker='x', s=50, label='Training data')
    ax.plot(x_plot, y_plot, c='blue', lw=2, label=f'Degree {deg}')
    ax.plot(x_plot, np.cos(x_plot/2), c='gray', ls='--', alpha=0.5, label='True cos(x/2)')
    ax.set_title(title)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Polynomial Regression - underfit vs good fit vs overfit")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "polynomial_fit.png"), dpi=120)
plt.show()
# degree 2 can't capture the curve, 6 is just right, 14 goes crazy


# ====== 2. feature engineering - combining features ======

print("\n2. Feature engineering - combining raw features")

# house data with size and bedrooms
# adding size^2 and size*bedrooms as extra features
np.random.seed(5)
sizes = np.random.uniform(800, 2500, 30)
bedrooms = np.random.choice([2, 3, 4], 30)
prices = 50 + 0.12*sizes + 5*bedrooms + 0.00003*sizes**2 + 3*np.random.randn(30)

X_raw = np.column_stack([sizes, bedrooms])
X_eng = np.column_stack([sizes, bedrooms, sizes**2, sizes*bedrooms])

# normalize both
X_raw_n, mu_r, sig_r = zscore_normalize(X_raw)
X_eng_n, mu_e, sig_e = zscore_normalize(X_eng)

print("\n--- Raw features (size, beds) ---")
w_raw, b_raw, hist_raw = gradient_descent(X_raw_n, prices, np.zeros(2), 0., alpha=0.1, num_iters=5000)

print("\n--- Engineered features (size, beds, size^2, size*beds) ---")
w_eng, b_eng, hist_eng = gradient_descent(X_eng_n, prices, np.zeros(4), 0., alpha=0.1, num_iters=5000)

print(f"\nFinal cost (raw features):       {hist_raw[-1]:.4f}")
print(f"Final cost (engineered features): {hist_eng[-1]:.4f}")
# engineered features give a lower cost - the extra features help!

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hist_raw, label='Raw features (2)', lw=2)
ax.plot(hist_eng, label='Engineered features (4)', lw=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Cost J')
ax.set_title('Feature Engineering - Convergence Comparison')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "feature_engineering_convergence.png"), dpi=120)
plt.show()


# ====== 3. loss function comparison ======

print("\n3. Squared error vs cross-entropy loss")
print("showing why cross-entropy is better for classification")

f_vals = np.linspace(0.01, 0.99, 200)

# when target y = 1
loss_sq_y1 = (f_vals - 1)**2
loss_ce_y1 = -np.log(f_vals)

# when target y = 0
loss_sq_y0 = f_vals**2
loss_ce_y0 = -np.log(1 - f_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(f_vals, loss_sq_y1, label='Squared error', lw=2)
ax1.plot(f_vals, loss_ce_y1, label='Cross-entropy', lw=2)
ax1.set_title('Loss when y = 1')
ax1.set_xlabel('f(x) (prediction)'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.set_ylim(0, 5); ax1.grid(alpha=0.3)

ax2.plot(f_vals, loss_sq_y0, label='Squared error', lw=2)
ax2.plot(f_vals, loss_ce_y0, label='Cross-entropy', lw=2)
ax2.set_title('Loss when y = 0')
ax2.set_xlabel('f(x) (prediction)'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.set_ylim(0, 5); ax2.grid(alpha=0.3)

fig.suptitle("Why cross-entropy works better for classification")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "loss_comparison.png"), dpi=120)
plt.show()

# cross-entropy penalizes confident wrong predictions way more
# that gives stronger gradients so it learns faster
print("cross-entropy penalizes confident wrong predictions much harder")
print("which gives stronger gradients and faster convergence")

print("\nDone!")
