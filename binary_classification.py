# Logistic Regression - binary classification
# two examples: exam admission + microchip QA testing

import os
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'assets')
os.makedirs(_ASSETS_DIR, exist_ok=True)


def sigmoid(z):
    """sigmoid / logistic function"""
    return 1 / (1 + np.exp(-z))

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0"):
    """plot binary classification data"""
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos,0], X[pos,1], marker='+', c='blue', label=pos_label, s=60)
    ax.scatter(X[neg,0], X[neg,1], marker='o', c='red', label=neg_label, s=30, alpha=0.7)

def map_feature(x1, x2, degree=6):
    """map two features to polynomial features up to given degree"""
    out = []
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((x1**(i-j)) * (x2**j))
    return np.stack(out, axis=1)


# testing sigmoid
print("sigmoid(0) =", sigmoid(0))
print("sigmoid([-1, 0, 1, 2]) =", sigmoid(np.array([-1, 0, 1, 2])))


# ======= DATASET 1: exam scores -> admission =======
print("\n=== Dataset 1: Exam Score Admission ===")

# generating some data that looks like exam scores
np.random.seed(1)
pos_data = np.random.multivariate_normal([65, 65], [[80, 30], [30, 80]], 30)
neg_data = np.random.multivariate_normal([40, 45], [[90, 20], [20, 90]], 30)
X_train = np.vstack([pos_data, neg_data])
y_train = np.array([1]*30 + [0]*30, dtype=float)

print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")

fig, ax = plt.subplots(figsize=(6,5))
plot_data(X_train, y_train, ax, "Admitted", "Not admitted")
ax.set_xlabel("Exam 1 score"); ax.set_ylabel("Exam 2 score")
ax.legend(loc="upper right")
ax.set_title("Exam Score Dataset")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "logistic_data1.png"), dpi=120)
plt.show()


# logistic regression cost function
def compute_cost(X, y, w, b, lambda_=0):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost/m

    # regularization
    if lambda_ > 0:
        reg = 0.
        for j in range(n):
            reg += w[j]**2
        cost += (lambda_/(2*m)) * reg

    return cost

# test
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print(f"Cost at zeros: {cost:.3f}")


# gradient
def compute_gradient(X, y, w, b, lambda_=0):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i,j]
        z_wb += b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n):
            dj_dw[j] += dj_db_i * X[i,j]

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    # regularization on weights only (not bias)
    if lambda_ > 0:
        for j in range(n):
            dj_dw[j] += (lambda_/m) * w[j]

    return dj_db, dj_dw


# gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b, lambda_))
        if i % math.ceil(num_iters/10) == 0:
            print(f"  Iteration {i:4d}: Cost {J_history[-1]:.6f}")

    return w, b, J_history


# predict function
def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i,j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


# train on dataset 1
print("\nTraining on exam data...")
np.random.seed(1)
w_init = 0.01 * (np.random.rand(n) - 0.5)
b_init = -8

w, b, J_hist = gradient_descent(X_train, y_train, w_init, b_init,
                                 alpha=0.001, num_iters=10000)

p = predict(X_train, w, b)
print(f'\nTrain Accuracy: {np.mean(p == y_train) * 100:.1f}%')

# decision boundary
fig, ax = plt.subplots(figsize=(6,5))
plot_data(X_train, y_train, ax, "Admitted", "Not admitted")

# the boundary is where w0*x0 + w1*x1 + b = 0
x0_vals = np.array([X_train[:,0].min()-2, X_train[:,0].max()+2])
x1_vals = -(w[0]*x0_vals + b)/w[1]
ax.plot(x0_vals, x1_vals, c='green', lw=2, label="Decision boundary")

ax.set_xlabel("Exam 1 score"); ax.set_ylabel("Exam 2 score")
ax.legend(loc="upper right")
ax.set_title("Decision Boundary")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "logistic_boundary1.png"), dpi=120)
plt.show()


# ======= DATASET 2: microchip QA (needs polynomial features) =======
print("\n=== Dataset 2: Microchip QA ===")

# circular-ish boundary - can't separate with a straight line
np.random.seed(2)
n2 = 80
theta = np.random.uniform(0, 2*np.pi, n2)
r_pos = 0.5 + 0.15*np.random.randn(n2//2)
r_neg = 1.0 + 0.15*np.random.randn(n2//2)
X2_pos = np.column_stack([r_pos * np.cos(theta[:n2//2]), r_pos * np.sin(theta[:n2//2])])
X2_neg = np.column_stack([r_neg * np.cos(theta[n2//2:]), r_neg * np.sin(theta[n2//2:])])
X_train2 = np.vstack([X2_pos, X2_neg])
y_train2 = np.array([1]*(n2//2) + [0]*(n2//2), dtype=float)

fig, ax = plt.subplots(figsize=(5,5))
plot_data(X_train2, y_train2, ax, "Accepted", "Rejected")
ax.set_xlabel("Test 1"); ax.set_ylabel("Test 2")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "logistic_data2.png"), dpi=120)
plt.show()

# map to polynomial features
X_mapped = map_feature(X_train2[:,0], X_train2[:,1])
print(f"Original shape: {X_train2.shape}")
print(f"After feature mapping: {X_mapped.shape}")  # 28 features from degree-6

# train with regularization
print("\nTraining with regularization (lambda=0.01)...")
np.random.seed(1)
w2_init = np.random.rand(X_mapped.shape[1]) - 0.5
b2_init = 1.

w2, b2, J_hist2 = gradient_descent(X_mapped, y_train2, w2_init, b2_init,
                                    alpha=0.01, num_iters=10000, lambda_=0.01)

p2 = predict(X_mapped, w2, b2)
print(f'Train Accuracy: {np.mean(p2 == y_train2) * 100:.1f}%')

# non-linear decision boundary
fig, ax = plt.subplots(figsize=(5,5))
plot_data(X_train2, y_train2, ax, "Accepted", "Rejected")

# plot boundary by evaluating on a grid
u = np.linspace(-1.5, 1.5, 150)
v = np.linspace(-1.5, 1.5, 150)
Z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        feat = map_feature(np.array([u[i]]), np.array([v[j]]))
        Z[i,j] = sigmoid(np.dot(feat[0], w2) + b2)

ax.contour(u, v, Z.T, levels=[0.5], colors='green', linewidths=2)
ax.set_xlabel("Test 1"); ax.set_ylabel("Test 2")
ax.legend(loc="upper right")
ax.set_title("Non-linear Decision Boundary (regularized)")
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "logistic_boundary2.png"), dpi=120)
plt.show()

# convergence for both datasets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(J_hist); ax1.set_title("Dataset 1 convergence")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("Cost"); ax1.grid(alpha=0.3)
ax2.plot(J_hist2); ax2.set_title("Dataset 2 convergence")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("Cost"); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_ASSETS_DIR, "logistic_convergence.png"), dpi=120)
plt.show()

print("\nDone!")
