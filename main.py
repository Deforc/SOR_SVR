import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

np.random.seed(42)


class Info:
    def __init__(self, size):
        self.history_of_norm = np.zeros(size)

    def show_history_of_norm(self):
        plt.plot(np.log10(self.history_of_norm))
        plt.grid()
        plt.title("History of norm")
        plt.show()


def sinc_function(x):
    return np.sinc(x)


l = 200
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, l)
y_true = sinc_function(x)

noise = np.random.normal(0, 0.1, l)
y_noisy = y_true + noise


def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def compute_kernel_matrix(X, gamma):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K


def sor_algorithm(H, E, c, C, tolerance=1e-3, max_iter=10000, omega=1.5):
    info = Info(max_iter)
    A = H + E
    a = np.zeros_like(c)

    # Разложение A = L + D + L^T
    D = np.diag(np.diag(A))  # Диагональная часть
    L = np.tril(A, k=-1)  # Строго нижняя треугольная часть

    # Инвертированная диагональ
    diag = np.diag(D)
    if np.any(diag == 0):
        raise ValueError("Diagonal elements of D contain zeros!")
    diag_inv = 1 / diag

    for iteration in range(max_iter):
        a_prev = a.copy()

        # Обновление a с использованием цикла по j
        for j in range(len(a)):
            sum_L = np.dot(L[j, :j], a[:j])  # Вклад от строго нижней треугольной части
            sum_D_U = np.dot(A[j, j:], a_prev[j:])  # Вклад от диагональной и верхней частей
            a[j] = a_prev[j] - omega * diag_inv[j] * (sum_L + sum_D_U - c[j])
            a[j] = max(0, min(C, a[j]))  # Проекция на [0, C]

        # Проверка на сходимость
        norm = np.linalg.norm(a - a_prev)
        info.history_of_norm[iteration] = norm
        if norm < tolerance:
            break

    return a, info


def predict(x_new, x_train, a, b, gamma):
    n = len(x_train)
    y_pred = np.zeros(len(x_new))
    for i in range(len(x_new)):
        y_pred[i] = b
        for j in range(n):
            y_pred[i] += (a[j] - a[j + n]) * rbf_kernel(x_train[j], x_new[i], gamma)
    return y_pred


param_grid = {
    'C': [0.1, 1, 5],
    'epsilon': [0.05, 0.075, 0.1],
    'gamma': [0.1, 1, 5],
    'omega': [0.5, 1, 1.5]
}

results = []
best_mae = float('inf')
best_params = None
best_info = None

total_iterations = len(list(ParameterGrid(param_grid)))
with tqdm(total=total_iterations, desc="Grid Search Progress", unit="iteration") as pbar:
    for params in ParameterGrid(param_grid):
        C, epsilon, gamma, omega = params['C'], params['epsilon'], params['gamma'], params['omega']

        K = compute_kernel_matrix(x, gamma)
        H = np.block([[K, -K], [-K, K]])
        E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
        c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

        a, info = sor_algorithm(H, E, c, C, omega=omega)
        b = -np.sum(a[l:] - a[:l])

        y_pred = predict(x, x, a, b, gamma)

        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma,
            'omega': omega,
            'MAE': mae
        })

        if mae < best_mae:
            best_mae = mae
            best_params = params
            best_info = info

        pbar.update(1)

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
results_df = pd.DataFrame(results)

# Вывод лучших параметров
print(f"Best MAE: {best_mae:.4f}")
print(f"Best parameters: {best_params}")
best_info.show_history_of_norm()

# Тепловая карта для анализа результатов
for omega in results_df['omega'].unique():
    subset = results_df[results_df['omega'] == omega]
    pivot_table = subset.pivot_table(
        values='MAE',
        index=['C', 'epsilon'],
        columns='gamma'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title(f"Heatmap of MAE by Hyperparameters (Omega = {omega})")
    plt.xlabel("Gamma")
    plt.ylabel("C, Epsilon")
    plt.show()

# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".4f")
# plt.title("Heatmap of MAE by Hyperparameters")
# plt.xlabel("Gamma")
# plt.ylabel("C, Epsilon")
# plt.show()

# График для наилучшей комбинации гиперпараметров
C_best, epsilon_best, gamma_best = best_params['C'], best_params['epsilon'], best_params['gamma']

K_best = compute_kernel_matrix(x, gamma_best)
H_best = np.block([[K_best, -K_best], [-K_best, K_best]])
E_best = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
c_best = np.concatenate([y_noisy - epsilon_best, -y_noisy - epsilon_best])

a_best, info_best = sor_algorithm(H_best, E_best, c_best, C_best)
b_best = -np.sum(a_best[l:] - a_best[:l])

# Используем x_test для плавного графика
x_test = np.linspace(x_min, x_max, 500)
y_pred_best = predict(x_test, x, a_best, b_best, gamma_best)

plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label="True function", color="black", linestyle="--")
plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
plt.plot(x_test, y_pred_best, label="Predicted function", color="red")
plt.fill_between(x_test, y_pred_best - epsilon_best, y_pred_best + epsilon_best, color="gray", alpha=0.2,
                 label="ε-tube")
plt.legend()
plt.title("Support Vector Regression with Best Parameters")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

