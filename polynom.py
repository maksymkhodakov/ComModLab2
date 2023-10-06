import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Дані
mas_x = np.array([-1.6000, -1.2000, -0.8000, -0.4000, 0, 0.4000, 0.8000, 1.2000, 1.6000, 2.0000]) + 10
mas_y = np.array([4.3200, 3.2800, 2.8800, 3.1200, 4.0000, 5.5200, 7.6800, 10.4800, 13.9200, 18.0000]) + - 0.1

print("Mass x:")
print(mas_x)
print("Mas y:")
print(mas_y)


# Функція квадратного трьохчлена
def quadratic_func(x, A, B, C):
    return A * x ** 2 + B * x + C


def least_squares_fit(mas_x, mas_y):
    # Побудова матриці дизайну
    X = np.vstack([mas_x ** 2, mas_x, np.ones_like(mas_x)]).T

    # Виконання методу найменших квадратів
    params = np.linalg.lstsq(X, mas_y, rcond=None)[0]

    return params


params = least_squares_fit(mas_x, mas_y)

A, B, C = params

print(f"A: {A}")
print(f"B: {B}")
print(f"C: {C}")

# Побудова графіку
plt.scatter(mas_x, mas_y, color="blue", label="Data")
plt.plot(mas_x, quadratic_func(mas_x, *params), color="red", label=f"y = {A:.4f}x^2 + {B:.4f}x + {C:.4f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Curve Fitting using Least Squares Method")
plt.grid(True)
plt.show()
