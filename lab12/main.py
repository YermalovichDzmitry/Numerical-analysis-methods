import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures


## Задача 1
# a = np.sin(8)
# b = np.cos(8)
# x = sp.symbols('x')
# y = sp.symbols('y')
#
# h = 2 / 10
#
# x_vals = np.arange(-1, 1 + 0.00000001, h)
#
# y_vals = []
# y_vals.append(0)
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(0)
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     system_of_eq.append(
#         y_vals[i] - (2 + (h ** 2) * (1 + b * (x_vals[i + 1] ** 2))) * y_vals[i + 1] + y_vals[i + 2] + h ** 2)
#
# # for eq in system_of_eq:
# #     print(eq)
# v = [y_vals[1], y_vals[2], y_vals[3], y_vals[4]]
# res = sp.solve(system_of_eq, y_vals)
# # print(res)
# y_res = []
# y_res.append(0)
# for k, v in res.items():
#     y_res.append(res[k])
# y_res.append(0)
#
# # print()
# # for x_val, y_val in zip(x_vals, y_res):
# #     print(f"{x_val} {y_val}")
#
# x_vals_1 = x_vals
# y_res_1 = y_res
#
# x_vals = np.array(x_vals)
# y_res = np.array(y_res)
# x_vals = np.expand_dims(x_vals, axis=1)
# y_res = np.expand_dims(y_res, axis=1)
#
# pp = PolynomialFeatures(degree=4)
# x_vals_poly = pp.fit_transform(x_vals)
# lr = LinearRegression()
# lr = lr.fit(x_vals_poly, y_res)
#
# h_new = 2 / 20
# x_vals_new = np.arange(-1, 1 + 0.00000001, h_new)
# x_vals_new = np.expand_dims(x_vals_new, axis=1)
# x_vals_new_tranc = pp.transform(x_vals_new)
# y_pred = lr.predict(x_vals_new_tranc)
#
# ## уменьшаем шаг в два раза
# h = 2 / 20
#
# x_vals = np.arange(-1, 1 + 0.00000001, h)
#
# y_vals = []
# y_vals.append(0)
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(0)
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     system_of_eq.append(
#         y_vals[i] - (2 + (h ** 2) * (1 + b * (x_vals[i + 1] ** 2))) * y_vals[i + 1] + y_vals[i + 2] + h ** 2)
#
# res = sp.solve(system_of_eq, y_vals)
#
# y_res = []
# y_res.append(0)
# for k, v in res.items():
#     y_res.append(res[k])
# y_res.append(0)
#
# x_vals_2 = np.array(x_vals)
# y_res_2 = np.array(y_res)
# y_res_2 = np.expand_dims(y_res_2, axis=1)
# dist = y_res_2 - y_pred
# dist = dist.astype(np.float32)
# print(np.linalg.norm(dist))

# print()
# for x_val, y_val in zip(x_vals, y_res):
#     print(f"{x_val} {y_val}")

# plt.plot(x_vals_new, y_pred)
# plt.show()

## Задача 2
# x = sp.symbols('x')
# y = sp.symbols('y')
#
# h = 2 / 10 ## Неправильный шаг
#
# x_vals = np.arange(1, 3 + 0.00000001, h)
#
# y_vals = []
# y_vals.append(-1)
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(4)
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     f1 = (y_vals[i + 2] - 2 * y_vals[i + 1] + y_vals[i]) / (h ** 2)
#     f2 = (1 + (np.cos(x_vals[i + 1])) ** 2) * ((y_vals[i + 2] - y_vals[i + 1]) / (2 * h))
#     f3 = (x_vals[i + 1] ** 2 + 1) * y_vals[i + 1]
#     f4 = (x_vals[i + 1] ** 2 + 1) * np.cos(x_vals[i + 1])
#     system_of_eq.append(f1 + f2 + f3 - f4)
#
# # for eq in system_of_eq:
# #     print(eq)
# res = sp.solve(system_of_eq, y_vals)
# y_res = []
# y_res.append(-1)
# for k, v in res.items():
#     y_res.append(res[k])
# y_res.append(4)
# plt.plot(x_vals, y_res)
# plt.show()


## Задача 3
# x = sp.symbols('x')
# y = sp.symbols('y')
#
# h = 2 / 10
#
# x_vals = np.arange(2.2, 4.2 + 0.0000000001, h)
#
# y_vals = []
# y_vals.append(sp.symbols("y0"))
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(sp.symbols("yN"))
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     f1 = (y_vals[i + 2] - 2 * y_vals[i + 1] + y_vals[i]) / (h ** 2)
#     f2 = -6 * x_vals[i + 1] * ((y_vals[i + 2] - y_vals[i + 1]) / (2 * h))
#     f3 = 0.5 * y_vals[i + 1]
#     f4 = -3
#     system_of_eq.append(f1 + f2 + f3 - f4)
#
# system_of_eq.append(0.2 - 0.1 * ((-y_vals[2] + 4 * y_vals[1] - 3 * y_vals[0]) / (2 * h)) - y_vals[0])
# system_of_eq.append((3 * y_vals[len(y_vals) - 1] - 4 * y_vals[len(y_vals) - 2] + y_vals[len(y_vals) - 3]) / (2 * h) - 4)
#
# # for eq in system_of_eq:
# #     print(eq)
#
# res = sp.solve(system_of_eq, y_vals)
# print(res)
# y_res = []
# for k, v in res.items():
#     y_res.append(res[k])
# plt.plot(x_vals, y_res)
# plt.show()

## Задача 4
# x = sp.symbols('x')
# y = sp.symbols('y')
#
# h = 1.875 / 20
#
# x_vals = np.arange(0, 1.875 + 0.0000000001, h)
#
# y_vals = []
# y_vals.append(sp.symbols("y0"))
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(sp.symbols("yN"))
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     f1 = -(1.5) * (y_vals[i + 2] - 2 * y_vals[i + 1] + y_vals[i]) / (h ** 2)
#     f3 = 8.3 * y_vals[i + 1]
#     f4 = 7 * np.exp(-0.5 * x_vals[i + 1])
#     system_of_eq.append(f1 + f3 - f4)
# system_of_eq.append(-1.5 * ((-y_vals[2] + 4 * y_vals[1] - 3 * y_vals[0]) / (2 * h)) + 0.5 * y_vals[0])
# system_of_eq.append(
#     1.5 * (3 * y_vals[len(y_vals) - 1] - 4 * y_vals[len(y_vals) - 2] + y_vals[len(y_vals) - 3]) / (2 * h) + 0.5 *
#     y_vals[len(y_vals) - 1])

# for eq in system_of_eq:
#     print(eq)
#
# res = sp.solve(system_of_eq, y_vals)
#
# y_res = []
# for k, v in res.items():
#     y_res.append(res[k])
#
# x_vals_1 = x_vals
# y_res_1 = y_res

# plt.plot(x_vals, y_res)
# plt.show()


# h = (3 - 1.875) / 20

# x_vals = np.arange(1.875, 3 + 0.0000000001, h)
#
# y_vals = []
# y_vals.append(sp.symbols("y0"))
# for i in range(len(x_vals) - 2):
#     y_vals.append(sp.symbols(f"y{i + 1}"))
# y_vals.append(sp.symbols("yN"))
#
# system_of_eq = []
# for i in range(len(x_vals) - 2):
#     f1 = -(0.6) * (y_vals[i + 2] - 2 * y_vals[i + 1] + y_vals[i]) / (h ** 2)
#     f3 = 12 * y_vals[i + 1]
#     f4 = 7 * np.exp(-0.5 * x_vals[i + 1])
#     system_of_eq.append(f1 + f3 - f4)
# system_of_eq.append(-0.6 * ((-y_vals[2] + 4 * y_vals[1] - 3 * y_vals[0]) / (2 * h)) + 0.5 * y_vals[0])
# system_of_eq.append(
#     0.6 * (3 * y_vals[len(y_vals) - 1] - 4 * y_vals[len(y_vals) - 2] + y_vals[len(y_vals) - 3]) / (2 * h) + 0.5 *
#     y_vals[len(y_vals) - 1])
#
# res = sp.solve(system_of_eq, y_vals)
#
# y_res = []
# for k, v in res.items():
#     y_res.append(res[k])
#
# x_vals_2 = x_vals
# y_res_2 = y_res
#
# x_vals_1 = list(x_vals_1)
# for x, y in zip(x_vals_2, y_res_2):
#     x_vals_1.append(x)
#     y_res_1.append(y)
#
# print(x_vals_1)
#
# plt.plot(x_vals_1, y_res_1, marker='o')
# plt.show()

# x = sp.symbols('x')
# h = 0.1
# teta = 0.1
# a = 0.1
# b = 0.8
# T_big = 1
#
# g = []
# unknown = []
# for i in range(np.int(T_big / teta)):
#     g.append([])
#     for j in range(np.int((b - a) / h)):
#         s = np.int(T_big / teta) - i - 1
#         s = f"y{s}{j}"
#         g[i].append(sp.symbols(s))
#         unknown.append(s)
#
# g = np.array(g)
# for i in range(np.int(T_big / teta)):
#     for j in range(np.int((b - a) / h)):
#         if i == np.int(T_big / teta) - 1 and j != 0 and j != np.int((b - a) / h) - 1:
#             g[i, j] = j
#             continue
#
#         if j == 0:
#             g[i, j] = 6
#             continue
#
#         if j == np.int((b - a) / h) - 1:
#             g[i, j] = 0.6
#             continue
# print(g)
# print(teta)
# print(h)
# # print((np.int(T_big / teta) - 1))
# system_of_eq = []
# for i in range(np.int(T_big / teta)):
#     for j in range(np.int((b - a) / h)):
#         if j != 0 and i != (np.int(T_big / teta) - 1) and j != (((b - a) / h) - 1):
#             # print(f"{i} {j}")
#             r_p = (g[i, j] - g[i - 1, j]) / teta
#             l_p = (g[i, j] - g[i, j - 1]) / h - j * (g[i, j - 1] - 2 * g[i, j] + g[i, j + 1]) / h ** 2 + (
#                     j + (j ** (1 / 3))) * (1 - np.exp(-j))
#             system_of_eq.append(r_p - l_p)
#
# res = sp.solve(system_of_eq, unknown)
# print(res)
#
# for i in range(np.int(T_big / teta)):
#     for j in range(np.int((b - a) / h)):
#         if j != 0 and i != (np.int(T_big / teta) - 1) and j != (((b - a) / h) - 1):
#             index = g[i, j]
#             g[i, j] = np.round(float(res[index]), 4)
# print()
# print()
# print(g)
# y_res = []
# for k, v in res.items():
#     y_res.append(res[k])
## Пузырьковая сортировка
# def list_sort(lst):
#     buf = None
#     for i in range(len(lst)):
#         for j in range(len(lst) - 1 - i):
#             if abs(lst[j]) < abs(lst[j + 1]):
#                 buf = lst[j]
#                 lst[j] = lst[j + 1]
#                 lst[j + 1] = buf
#     return lst
#
#
# print(list_sort([-5, -6, -7, -8, -9, -1, -2]))

def count_it(sequence):
    d = {}
    # Формируем словарь
    for i in sequence:
        key = int(i)
        if key in d:
            d[key] += 1
        else:
            d[key] = 1

    # Сортируем словарь
    sorted_dict = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

    # Фурмируем новый словарь из 3-х самых часто встречаемых чисел
    top3_dict = {}
    k = 0
    for key, value in sorted_dict.items():
        top3_dict[key] = value
        k += 1
        if k == 3:
            break
    return top3_dict


print(count_it("123456543234543234565434543"))
