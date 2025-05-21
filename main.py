import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

np.random.seed(42)
def count_support_vectors(a, C):
    l = len(a) // 2
    count = 0

    for i in range(l):
        if 0 < a[i] < C or 0 < a[i + l] < C:
            count += 1

    return count

def extract_bounded_support_vectors(a, x_train, C):
    """
    Выделяет связные векторы (граничные опорные векторы).
    :param a: Массив двойственных переменных (размер 2*l).
    :param x_train: Обучающие данные (размер l).
    :param C: Параметр регуляризации.
    :return: Индексы связных векторов и их координаты.
    """
    l = len(x_train)
    bounded_support_vector_indices = []

    for i in range(l):
        if np.isclose(a[i], C) or np.isclose(a[i + l], C):  # Проверяем условия для a_i и a_i^*
            bounded_support_vector_indices.append(i)

    bounded_support_vectors = x_train[bounded_support_vector_indices]
    return bounded_support_vector_indices, bounded_support_vectors
def extract_support_vectors(a, x_train, C):
    l = len(x_train)
    support_vector_indices = []

    for i in range(l):
        if 0 < a[i] < C or 0 < a[i + l] < C:
            support_vector_indices.append(i)

    support_vectors = x_train[support_vector_indices]
    return support_vector_indices, support_vectors
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
    # return np.sin(np.pi * x / 4.0) + 0.5 * np.sin(np.pi * x)

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


def sor_algorithm(H, E, c, C, tolerance=1e-3, max_iter=10000, omega=1.0):
    info = Info(max_iter)
    A = H + E

    a = np.zeros_like(c)



    # Разложение A = L + D + L^T
    D = np.diag(np.diag(A))  # Диагональная часть
    L = np.tril(A, k=-1)  # Строго нижняя треугольная часть

    diag = np.diag(D)
    if np.any(diag == 0):
        raise ValueError("Diagonal elements of D contain zeros!")
    diag_inv = 1 / diag

    for iteration in range(max_iter):
        a_prev = a.copy()

        for j in range(len(a)):
            sum_L = np.dot(L[j, :j], a[:j])  # Вклад от строго нижней треугольной части
            sum_D_U = np.dot(A[j, j:], a_prev[j:])  # Вклад от диагональной и верхней частей
            a[j] = a_prev[j] - omega * diag_inv[j] * (sum_L + sum_D_U - c[j])
            a[j] = max(0, min(C, a[j]))  # Проекция на [0, C]

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
    'C': np.logspace(-1, 2, 9),  # 9 значений для C
    'epsilon': np.linspace(0.01, 0.2, 9),  # 9 значений для epsilon
    'gamma': np.logspace(-1, 1, 9)  # 9 значений для gamma
}

# results = []
# best_mae = float('inf')
# best_params = None
# best_info = None
#
# total_iterations = len(list(ParameterGrid(param_grid)))
# with tqdm(total=total_iterations, desc="Grid Search Progress", unit="iteration") as pbar:
#     for params in ParameterGrid(param_grid):
#         C, epsilon = params['C'], params['epsilon']
#         gamma = 0.5
#         K = compute_kernel_matrix(x, gamma)
#         H = np.block([[K, -K], [-K, K]])
#         E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
#         c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])
#
#         a, info = sor_algorithm(H, E, c, C)
#         b = -np.sum(a[l:] - a[:l])
#
#         y_pred = predict(x, x, a, b, gamma)
#
#         mae = mean_absolute_error(y_true, y_pred)
#         sv_count = count_support_vectors(a, C)
#         results.append({
#             'C': C,
#             'epsilon': epsilon,
#             'gamma': gamma,
#             'sv_count': sv_count,
#             'MAE': mae
#         })
#
#         if mae < best_mae:
#             best_mae = mae
#             best_params = params
#             best_info = info
#
#         pbar.update(1)
#
# x_test = np.linspace(x_min, x_max, 500)
# y_pred = predict(x_test, x, a, b, gamma)
#
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_true, label="True function", color="black", linestyle="--")
# plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
# plt.plot(x_test, y_pred, label="Predicted function", color="red")
# plt.fill_between(x_test, y_pred - epsilon, y_pred + epsilon, color="gray", alpha=0.2, label="ε-tube")
# plt.legend()
# plt.title("Support Vector Regression with SOR")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()
# results_df = pd.DataFrame(results)
#
# # Группируем данные по sv_count и вычисляем среднее значение MAE для каждого sv_count
# grouped_results = results_df.groupby('sv_count')['MAE'].mean().reset_index()
#
# # Сортируем данные по sv_count для корректного отображения на графике
# grouped_results = grouped_results.sort_values(by='sv_count')
#
# # Построение графика
# plt.figure(figsize=(10, 6))
# plt.plot(grouped_results['sv_count'], grouped_results['MAE'], marker='o', linestyle='-', color='b')
# plt.xlabel("Number of Support Vectors (sv_count)")
# plt.ylabel("Mean Absolute Error (MAE)")
# plt.title("MAE vs Number of Support Vectors")
# plt.grid()
# plt.show()
#
# # Выводим таблицу с колонками: C, epsilon, gamma, sv_count, MAE
# table = results_df[['C', 'epsilon', 'gamma', 'sv_count', 'MAE']]
#
# # Сортируем таблицу по C, epsilon, gamma для удобства анализа
# table = table.sort_values(by=['C', 'epsilon', 'gamma'])
#
# # Выводим таблицу
# print(table)
#
# # Если хотите сохранить таблицу в файл (например, CSV):
# table.to_csv('results_table.csv', index=False)
#
# # Вывод лучших параметров
# print(f"Best MAE: {best_mae:.4f}")
# print(f"Best parameters: {best_params}")
# best_info.show_history_of_norm()
#
# # Тепловая карта для анализа результатов
# # for omega in results_df['omega'].unique():
# subset = results_df
# pivot_table = subset.pivot_table(
#     values='MAE',
#     index=['C', 'epsilon'],
#     columns='gamma'
# )
#
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".4f")
# plt.title(f"Heatmap of MAE by Hyperparameters")
# plt.xlabel("Gamma")
# plt.ylabel("C, Epsilon")
# plt.show()
# #_______________________________________4.2 Лучшие параметры по MAE для noisy_data_________________________________________
# # График для наилучшей комбинации гиперпараметров
# C_best, epsilon_best, gamma_best = best_params['C'], best_params['epsilon'], 0.5
#
# K_best = compute_kernel_matrix(x, gamma_best)
# H_best = np.block([[K_best, -K_best], [-K_best, K_best]])
# E_best = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
# c_best = np.concatenate([y_noisy - epsilon_best, -y_noisy - epsilon_best])
#
# a_best, info_best = sor_algorithm(H_best, E_best, c_best, C_best)
# b_best = -np.sum(a_best[l:] - a_best[:l])
#
# # Используем x_test для плавного графика
# x_test = np.linspace(x_min, x_max, 500)
# y_pred_best = predict(x_test, x, a_best, b_best, gamma_best)
#
# sup_vectors_indices, _ = extract_support_vectors(a_best, x, C_best)
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_true, label="True function", color="black", linestyle="--")
# plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
# plt.scatter(x[sup_vectors_indices], y_noisy[sup_vectors_indices], color="red", alpha=0.9,
#                 label="Support vectors")
# plt.plot(x_test, y_pred_best, label="Predicted function", color="red")
# plt.fill_between(x_test, y_pred_best - epsilon_best, y_pred_best + epsilon_best, color="gray", alpha=0.2,
#                  label="ε-tube")
# plt.legend()
# plt.title("Support Vector Regression with Best Parameters")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()
# #______________________________________________________________________________________________________________________
#
#
# # ___________________________________________4.1 Лучшие параметры по MAE для true_data___________________________________________
# y_exact = y_true
# K_exact = compute_kernel_matrix(x, gamma_best)
# H_exact = np.block([[K_exact, -K_exact], [-K_exact, K_exact]])
# E_exact = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
# c_exact = np.concatenate([y_exact - epsilon_best, -y_exact - epsilon_best])
#
# a_exact, info_exact = sor_algorithm(H_exact, E_exact, c_exact, C_best, omega=0.5)
# b_exact = -np.sum(a_exact[l:] - a_exact[:l])
#
# x_test = np.linspace(x_min, x_max, 500)
# y_pred_exact = predict(x_test, x, a_exact, b_exact, gamma_best)
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_true, label="True function", color="black", linestyle="--")
# plt.scatter(x, y_exact, label="Exact data", color="green", alpha=0.8)
# plt.plot(x_test, y_pred_exact, label="Predicted function", color="red")
# plt.fill_between(x_test, y_pred_exact - epsilon_best, y_pred_exact + epsilon_best, color="gray", alpha=0.2,
#                  label="ε-tube")
# plt.legend()
# plt.title("Support Vector Regression with Exact Data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()
#
# #_____________________________________________________________________________________________________________________
# # Добавление пункта 4.3 (Зависимость MAE от числа опорных векторов)
#
#
#
# C_values = [0.1, 1, 3, 5, 7, 10]
# epsilon_values = [0.01, 0.05, 0.075, 0.1, 0.125, 0.15]
# gamma_fixed = 0.5
#
# results_4_3 = []
#
# for C in C_values:
#     for epsilon in epsilon_values:
#         K = compute_kernel_matrix(x, gamma_fixed)
#         H = np.block([[K, -K], [-K, K]])
#         E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
#         c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])
#
#         a, _ = sor_algorithm(H, E, c, C, omega=0.5)
#         b = -np.sum(a[l:] - a[:l])
#
#         y_pred = predict(x, x, a, b, gamma_fixed)
#         mae = mean_absolute_error(y_true, y_pred)
#         num_support_vectors = count_support_vectors(a, C)
#
#         results_4_3.append({
#             'C': C,
#             'epsilon': epsilon,
#             'MAE': mae,
#             'NumSupportVectors': num_support_vectors
#         })
#
# results_4_3_df = pd.DataFrame(results_4_3)
#
# plt.figure(figsize=(12, 6))
#
# fixed_epsilon = 0.1
# subset_epsilon = results_4_3_df[results_4_3_df['epsilon'] == fixed_epsilon]
# plt.subplot(1, 2, 1)
# plt.plot(subset_epsilon['NumSupportVectors'], subset_epsilon['MAE'], marker='o', linestyle='-', color='b')
# plt.xlabel("Number of Support Vectors")
# plt.ylabel("MAE")
# plt.title(f"MAE vs Number of SVs (Fixed ε={fixed_epsilon})")
# plt.grid()
#
# fixed_C = 100
# subset_C = results_4_3_df[results_4_3_df['C'] == fixed_C]
# plt.subplot(1, 2, 2)
# plt.plot(subset_C['NumSupportVectors'], subset_C['MAE'], marker='o', linestyle='-', color='r')
# plt.xlabel("Number of Support Vectors")
# plt.ylabel("MAE")
# plt.title(f"MAE vs Number of SVs (Fixed C={fixed_C})")
# plt.grid()
#
# plt.tight_layout()
# plt.show()

# Выбираем omega=1.0 и лучшие гиперпараметры
# omega = 1.0
# best_params = {'C': 1.0, 'epsilon': 0.01, 'gamma': 1.0}
# C_best, epsilon_best, gamma_best = best_params['C'], best_params['epsilon'], best_params['gamma']
#
# # Вычисляем матрицы и решаем задачу для лучших параметров
# K_best = compute_kernel_matrix(x, gamma_best)
# H_best = np.block([[K_best, -K_best], [-K_best, K_best]])
# E_best = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
# c_best = np.concatenate([y_noisy - epsilon_best, -y_noisy - epsilon_best])
# a_best, info_best = sor_algorithm(H_best, E_best, c_best, C_best, omega=omega)
# b_best = -np.sum(a_best[l:] - a_best[:l])
#
# # Используем x_test для плавного графика
# x_test = np.linspace(x_min, x_max, 200)
# y_pred_best = predict(x_test, x, a_best, b_best, gamma_best)
# mae = mean_absolute_error(y_true, y_pred_best)
# print(mae)
# print(f"Кол-во итераций {len(info_best.history_of_norm[info_best.history_of_norm > 0.0])}")
# # Извлекаем опорные и связные векторы
# sup_vectors_indices, _ = extract_support_vectors(a_best, x, C_best)
# bounded_sup_vectors_indices, _ = extract_bounded_support_vectors(a_best, x, C_best)
#
# # Количество опорных и связных векторов
# num_support_vectors = len(sup_vectors_indices)
# num_bounded_support_vectors = len(bounded_sup_vectors_indices)
#
# # Строим график
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_true, label="True function", color="black", linestyle="--")
# plt.scatter(x, y_noisy, label="Noisy data", color="blue", alpha=0.5)
#
# # Отображаем опорные векторы
# plt.scatter(x[sup_vectors_indices], y_noisy[sup_vectors_indices], color="red", alpha=0.9,
#             label=f"Support vectors ({num_support_vectors})")
#
# # Отображаем связные векторы
# plt.scatter(x[bounded_sup_vectors_indices], y_noisy[bounded_sup_vectors_indices], color="green", alpha=0.9,
#             label=f"Bounded support vectors ({num_bounded_support_vectors})")
#
# plt.plot(x_test, y_pred_best, label="Predicted function", color="red")
# plt.fill_between(x_test, y_pred_best - epsilon_best, y_pred_best + epsilon_best, color="gray", alpha=0.2,
#                  label="ε-tube")
# #
# # Указываем количество опорных и связных векторов в легенде
# plt.legend()
# plt.title("Support Vector Regression with Best Parameters and Bounded Vectors")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# # Выводим количество опорных и связных векторов под графиком
# plt.figtext(0.5, -0.1,
#             f"Number of Support Vectors: {num_support_vectors}\n"
#             f"Number of Bounded Support Vectors: {num_bounded_support_vectors}",
#             wrap=True, ha="center", fontsize=12)
#
# plt.savefig('output.png')
# plt.show()

# # Загружаем данные из CSV-файлов
# omega_values = [0.5, 1.0, 1.5, 1.9]
# results_dfs = {omega: pd.read_csv(f'results_omega_{omega}.csv') for omega in omega_values}
#
# # Создаем тепловые карты для каждого значения omega
# for omega, results_df in results_dfs.items():
#     # Находим лучшее значение MAE и соответствующие параметры
#     best_row = results_df.loc[results_df['MAE'].idxmin()]
#     best_mae = best_row['MAE']
#     best_C = best_row['C']
#     best_epsilon = best_row['epsilon']
#     best_gamma = best_row['gamma']
#
#     # Выводим лучшие параметры и MAE для текущего omega
#     print(f"Results for omega={omega}:")
#     print(f"Best MAE: {best_mae:.4f}")
#     print(f"Best parameters: C={best_C}, epsilon={best_epsilon}, gamma={best_gamma}")
#     print("-" * 50)
#
#     # Формируем сводную таблицу (pivot table)
#     pivot_table = results_df.pivot_table(
#         values='MAE',                # Значения для отображения
#         index=['C', 'epsilon'],      # Строки: C и epsilon
#         columns='gamma'              # Столбцы: gamma
#     )
#
#     # Построение тепловой карты
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".4f")
#     plt.title(f"Heatmap of MAE for Omega={omega}\nBest MAE: {best_mae:.4f}\nBest params: C={best_C}, epsilon={best_epsilon}, gamma={best_gamma}")
#     plt.xlabel("Gamma")
#     plt.ylabel("C, Epsilon")
#     plt.tight_layout()
#
#     # Сохраняем график в файл
#     plt.savefig(f'heatmap_omega_{omega}.png', bbox_inches='tight')
#     plt.show()
#     plt.close()  # Закрываем график для освобождения памяти
#
# print("All heatmaps have been created and saved.")

# Загружаем данные для omega=1.0

# C_best = 2  # Лучшее значение C
# gamma_best = 1.0           # Фиксированное значение gamma
# omega = 1.0                # Фиксированное значение omega
#
# # Генерируем 20 значений epsilon
# epsilon_values = np.linspace(0.01, 0.25, 20)
#
# # Список для хранения результатов
# results_epsilon_variation = []
#
# # Цикл по значениям epsilon
# for epsilon in tqdm(epsilon_values, desc="Epsilon Variation Progress", unit="value"):
#     # Вычисляем матрицы
#     K = compute_kernel_matrix(x, gamma_best)
#     H = np.block([[K, -K], [-K, K]])
#     E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])
#     c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])
#
#     # Решаем задачу с помощью SOR
#     a, _ = sor_algorithm(H, E, c, C_best, omega=omega)
#     b = -np.sum(a[l:] - a[:l])
#
#     # Предсказываем значения
#     y_pred = predict(x, x, a, b, gamma_best)
#
#     # Вычисляем MAE
#     mae = mean_absolute_error(y_true, y_pred)
#
#     # Считаем количество опорных векторов
#     sv_count = count_support_vectors(a)
#
#     # Сохраняем результаты
#     results_epsilon_variation.append({
#         'epsilon': epsilon,
#         'MAE': mae,
#         'NumSupportVectors': sv_count
#     })
#
# # Преобразуем результаты в DataFrame
# results_epsilon_variation_df = pd.DataFrame(results_epsilon_variation)
#
# # Построение графика зависимости MAE от числа опорных векторов
# plt.figure(figsize=(10, 6))
# plt.plot(results_epsilon_variation_df['NumSupportVectors'], results_epsilon_variation_df['MAE'],
#          marker='o', linestyle='-', color='b')
# plt.xlabel("Number of Support Vectors")
# plt.ylabel("Mean Absolute Error (MAE)")
# plt.title("MAE vs Number of Support Vectors (Varying Epsilon)")
# plt.grid()
# plt.tight_layout()
#
# # Сохраняем график в файл
# plt.savefig('mae_vs_sv_count_varying_epsilon.png', bbox_inches='tight')
# plt.close()
#
# print("Graph 'MAE vs Number of Support Vectors (Varying Epsilon)' has been created and saved.")
#
# # Выводим таблицу с результатами
# print(results_epsilon_variation_df)

def study_mae_vs_support_vectors():
    """
    Исследование зависимости ошибки MAE от количества опорных векторов.
    """
    l = 200
    x_min, x_max = -5, 5
    noise_sigma = 0.1
    x = np.linspace(x_min, x_max, l)
    gamma = 1
    y_true = sinc_function(x)
    C = 2
    # Добавление шума
    noise = np.random.normal(0, noise_sigma, l)
    y_noisy = y_true + noise

    # Параметры исследования
    epsilon_values = np.linspace(0.1, 0.45, 15)
    results = []

    # Списки для хранения данных
    iteration_counts = []
    support_vector_counts = []

    for epsilon in tqdm(epsilon_values, desc="Epsilon Variation Progress", unit="value"):
        K = compute_kernel_matrix(x, gamma)

        H = np.block([[K, -K], [-K, K]])
        E = np.block([[np.ones((l, l)), -np.ones((l, l))], [-np.ones((l, l)), np.ones((l, l))]])

        c = np.concatenate([y_noisy - epsilon, -y_noisy - epsilon])

        a, info = sor_algorithm(H, E, c, C, omega=1)

        b = -np.sum(a[l:] - a[:l])

        # Вычисление MAE
        y_pred = predict(x, x, a, b, gamma)
        mae = mean_absolute_error(y_true, y_pred)

        # Подсчёт количества опорных векторов
        support_vector_indices, _ = extract_support_vectors(a, x, C)
        num_support_vectors = len(support_vector_indices)
        support_vector_counts.append(num_support_vectors)

        # Подсчёт количества итераций
        num_iterations = len(info.history_of_norm[info.history_of_norm > 0.0])
        iteration_counts.append(num_iterations)

        results.append((epsilon, num_support_vectors, mae))

    # Преобразование результатов в массивы
    epsilon_values_, num_support_vectors, mae_values = zip(*results)

    # Визуализация
    plt.figure(figsize=(12, 12))  # Увеличиваем размер фигуры для 2x2 сетки

    # График 1: Зависимость MAE от числа опорных векторов
    plt.subplot(2, 2, 1)
    plt.plot(num_support_vectors, mae_values, marker='o', color='blue')
    plt.xlabel("Number of Support Vectors")
    plt.ylabel("MAE")
    plt.title("MAE vs Number of Support Vectors")
    plt.grid()

    # График 2: Зависимость MAE от параметра epsilon
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values_, mae_values, marker='o', color='green')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("MAE")
    plt.title("MAE vs eps")
    plt.grid()

    # График 3: Зависимость числа опорных векторов от параметра epsilon
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_values_, support_vector_counts, marker='o', color='purple')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("Number of Support Vectors")
    plt.title("Number of Support Vectors vs eps")
    plt.grid()

    # График 4: Зависимость количества итераций от параметра epsilon
    plt.subplot(2, 2, 4)
    plt.plot(epsilon_values_, iteration_counts, marker='o', color='red')
    plt.xlabel("Eps (Tube parameter)")
    plt.ylabel("Number of Iterations")
    plt.title("Number of Iterations vs eps")
    plt.grid()

    plt.tight_layout()
    plt.savefig("Amount_sv_vs_MAE_and_Iterations.jpg")
    plt.show()

study_mae_vs_support_vectors()