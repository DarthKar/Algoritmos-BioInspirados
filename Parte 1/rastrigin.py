import numpy as np
import matplotlib.pyplot as plt

LB, UB = -5.12, 5.12
A = 10




# ---------------------------------------------------------------
# Función objetivo (acepta un punto (n,) o una población (m, n))
# ---------------------------------------------------------------


def rastrigin(X):
    X = np.atleast_2d(X)
    f = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    return f if f.size > 1 else f[0]


def rastrigin_grad(x):
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


# ---------------------------------------------------------------
# 1. Descenso por gradiente
# ---------------------------------------------------------------
def gradient_descent(dim, lr=0.002, max_iter=250, rng=None):
    rng = rng or np.random.default_rng()
    x = rng.uniform(LB, UB, dim)                 # condición inicial aleatoria
    best_hist = [rastrigin(x)]
    for _ in range(max_iter):
        x = np.clip(x - lr * rastrigin_grad(x), LB, UB)
        best_hist.append(rastrigin(x))
    return x, rastrigin(x), np.array(best_hist)




