import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

def sinc_function(x):
    return np.sinc(x)

l = 200
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, l)
y_true = sinc_function(x)


noise = np.random.normal(0, 0.1, l)
y_noisy = y_true + noise

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def compute_kernel_matrix(X, gamma):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K

def sor_algorithm(H, E, c, C, tolerance=1e-3, max_iter=1000, omega=1.0):
    A = H + E
    n = len(c)
    a = np.zeros(n)
    for iteration in range(max_iter):
        a_prev = a.copy()
        for j in range(n):
            sum1 = sum(A[j, k] * a[k] for k in range(j))
            sum2 = sum(A[j, k] * a_prev[k] for k in range(j, n))
            a[j] = a_prev[j] - omega / A[j, j] * (sum1 + sum2 - c[j])
            a[j] = max(0, min(C, a[j]))
        if np.linalg.norm(a - a_prev) < tolerance:
            break
    return a

def predict(x_new, x_train, a, b, gamma):
    n = len(x_train)
    y_pred = np.zeros(len(x_new))
    for i in range(len(x_new)):
        y_pred[i] = b
        for j in range(n):
            y_pred[i] += (a[j] - a[j + n]) * rbf_kernel(x_train[j], x_new[i], gamma)
    return y_pred

C = 100
epsilon = 0.1
gamma = 0.1

K = compute_kernel_matrix(x, gamma)

H = np.block([[K, -K], [-K, K]])
E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

a = sor_algorithm(H, E, c, C)

b = np.sum(a[l:] - a[:l])

x_test = np.linspace(x_min, x_max, 500)
y_pred = predict(x_test, x, a, b, gamma)

plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label="True function", color="black", linestyle="--")
plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
plt.plot(x_test, y_pred, label="Predicted function", color="red")
plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
plt.legend()
plt.title("Support Vector Regression with SOR")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

mae = mean_absolute_error(y_true, predict(x, x, a, b, gamma))
print(f"Mean Absolute Error (MAE): {mae:.4f}")