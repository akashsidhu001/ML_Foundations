# Regularization - L2 regularization for linear and logistic regression
# seeing how lambda affects overfitting

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- regularized cost for linear regression ---

print("1. Regularized cost - linear regression")

def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    n = len(w)

    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost /= (2*m)

    # add regularization term
    reg_cost = 0.
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = (lambda_ / (2*m)) * reg_cost

    return cost + reg_cost

# quick test
np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]) - 0.5
b_tmp = 0.5

cost = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, 0.7)
print(f"Regularized cost (linear): {cost:.4f}")


# --- regularized cost for logistic regression ---

print("\n2. Regularized cost - logistic regression")

def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost /= m

    reg_cost = 0.
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = (lambda_ / (2*m)) * reg_cost

    return cost + reg_cost

# test with same data
np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]) - 0.5
b_tmp = 0.5

cost = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, 0.7)
print(f"Regularized cost (logistic): {cost:.4f}")


# --- regularized gradient for linear regression ---

print("\n3. Regularized gradient - linear regression")

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    # regularization on weights only, not bias
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_db, dj_dw

# test
np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5

dj_db, dj_dw = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, 0.7)
print(f"dj_db: {dj_db:.4f}")
print(f"Regularized dj_dw: {dj_dw.round(4).tolist()}")


# --- regularized gradient for logistic regression ---

print("\n4. Regularized gradient - logistic regression")

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i

    dj_dw /= m
    dj_db /= m

    # regularization (weights only)
    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]

    return dj_db, dj_dw

# test
np.random.seed(1)
X_tmp = np.random.rand(5, 3)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5

dj_db, dj_dw = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, 0.7)
print(f"dj_db: {dj_db:.4f}")
print(f"Regularized dj_dw: {dj_dw.round(4).tolist()}")


# --- overfitting demo ---

print("\n5. Overfitting demo - effect of lambda on polynomial fit")

def map_feature(x, degree):
    return np.column_stack([x**d for d in range(1, degree+1)])

def fit_regularized(X, y, lambda_, alpha=0.01, num_iters=5000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.
    for _ in range(num_iters):
        dj_db, dj_dw = compute_gradient_linear_reg(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

# noisy sin wave data
np.random.seed(42)
x_data = np.linspace(0, 6, 25)
y_data = np.sin(x_data) + 0.15 * np.random.randn(25)

degree = 10
X_poly = map_feature(x_data, degree)

# normalize
mu = np.mean(X_poly, axis=0)
sigma = np.std(X_poly, axis=0)
sigma[sigma == 0] = 1
X_poly_norm = (X_poly - mu) / sigma

# try different lambda values
lambdas = [0, 0.01, 1.0]
titles = ["lambda=0 (overfit)", "lambda=0.01 (good fit)", "lambda=1 (underfit)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x_plot = np.linspace(0, 6, 200)
X_plot_poly = map_feature(x_plot, degree)
X_plot_norm = (X_plot_poly - mu) / sigma

for ax, lam, title in zip(axes, lambdas, titles):
    w, b = fit_regularized(X_poly_norm, y_data, lam, alpha=0.1, num_iters=10000)
    y_pred = np.dot(X_plot_norm, w) + b

    ax.scatter(x_data, y_data, c='red', marker='x', s=40, label='Data')
    ax.plot(x_plot, y_pred, c='blue', lw=2, label='Model')
    ax.plot(x_plot, np.sin(x_plot), c='gray', ls='--', alpha=0.5, label='True sin(x)')
    ax.set_title(title)
    ax.set_ylim(-1.5, 1.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Effect of regularization on polynomial regression (degree=10)")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "regularization_comparison.png"), dpi=120)
plt.show()

# you can clearly see lambda=0 overfits, lambda=1 underfits
# lambda=0.01 is the sweet spot

print("\nDone!")
