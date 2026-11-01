# Scikit-Learn SGDRegressor - comparing with my own gradient descent
# same house price dataset as the multi-feature regression

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)

# same house data as before
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

print(f"Training data: {X_train.shape[0]} examples, {X_train.shape[1]} features")


# normalize with sklearn's StandardScaler
# this is basically the same as z-score normalization
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

print(f"\nOriginal X[0]:   {X_train[0]}")
print(f"Normalized X[0]: {X_norm[0].round(3)}")
print(f"Feature means:   {scaler.mean_.round(2)}")
print(f"Feature stds:    {scaler.scale_.round(2)}")


# train with SGDRegressor
print("\n--- SGDRegressor training ---")
sgdr = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgdr.fit(X_norm, y_train)

print(f"Iterations: {sgdr.n_iter_}")
print(f"Weights: {sgdr.coef_.round(2)}")
print(f"Bias: {sgdr.intercept_[0]:.2f}")

# predictions
y_pred_sgd = sgdr.predict(X_norm)

# verify manually - should match
y_pred_manual = np.dot(X_norm, sgdr.coef_) + sgdr.intercept_

print(f"\n{'':4s} {'Actual':>8s}  {'SGD':>8s}  {'Manual':>8s}")
print("-" * 35)
for i in range(min(10, len(y_train))):
    print(f"{i:4d} {y_train[i]:8.0f}  {y_pred_sgd[i]:8.1f}  {y_pred_manual[i]:8.1f}")
# yep they match


# now compare with my own gradient descent
print("\n--- Comparing with from-scratch gradient descent ---")

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        dj_dw += err * X[i]
        dj_db += err
    return dj_dw/m, dj_db/m

# from scratch
w_scratch = np.zeros(X_norm.shape[1])
b_scratch = 0.
alpha = 0.01

for iteration in range(1000):
    dj_dw, dj_db = compute_gradient(X_norm, y_train, w_scratch, b_scratch)
    w_scratch -= alpha * dj_dw
    b_scratch -= alpha * dj_db

y_pred_scratch = np.dot(X_norm, w_scratch) + b_scratch

print(f"SKLearn  - w: {sgdr.coef_.round(2)}, b: {sgdr.intercept_[0]:.2f}")
print(f"Scratch  - w: {w_scratch.round(2)}, b: {b_scratch:.2f}")
# pretty similar! sklearn just does it with less code


# plots
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for i in range(4):
    axes[i].scatter(X_train[:,i], y_train, label='Actual', s=30)
    axes[i].scatter(X_train[:,i], y_pred_sgd, c='orange', label='Predicted', s=30)
    axes[i].set_xlabel(X_features[i])
axes[0].set_ylabel("Price (1000's)")
axes[0].legend(fontsize=8)
fig.suptitle("SGDRegressor - actual vs predicted")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "sklearn_predictions.png"), dpi=120)
plt.show()

# residual plot
residuals = y_train - y_pred_sgd
plt.figure(figsize=(7, 4))
plt.bar(range(len(residuals)), residuals, color='steelblue', alpha=0.8)
plt.axhline(y=0, color='red', ls='--', lw=1)
plt.xlabel('Sample index'); plt.ylabel('Residual (actual - predicted)')
plt.title('Residuals - SGDRegressor')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "sklearn_residuals.png"), dpi=120)
plt.show()

# predict a new house
x_new = np.array([[1200, 3, 1, 40]])
x_new_norm = scaler.transform(x_new)
price = sgdr.predict(x_new_norm)[0]
print(f"\nPrediction for 1200sqft, 3bed, 1floor, 40yr old: ${price*1000:.0f}")

print("\nDone!")
