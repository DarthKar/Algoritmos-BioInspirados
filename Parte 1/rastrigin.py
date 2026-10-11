import numpy as np
import matplotlib.pyplot as plt

# --- Configuración del problema (función de Rastrigin) ---
LB, UB = -5.12, 5.12  # Límites inferior y superior del dominio de búsqueda por variable
A = 10                # Constante de Rastrigin: controla la profundidad de los mínimos locales


# ---------------------------------------------------------------
# Función objetivo y su gradiente
# ---------------------------------------------------------------

def rastrigin(X):
    """
    Evalúa la función de Rastrigin, usada como benchmark de optimización.

    Fórmula:
        f(x) = A*n + sum(x_i**2 - A*cos(2*pi*x_i))

    Es una función multimodal con muchos mínimos locales y un único
    mínimo global en x = (0, ..., 0), donde f(x) = 0. El dominio de
    búsqueda habitual es [-5.12, 5.12] en cada variable.

    Parámetros
    ----------
    X : array_like, forma (n,) o (m, n)
        Un individuo con n variables, o una población de m individuos.

    Retorna
    -------
    float o numpy.ndarray de forma (m,)
        Valor de la función para cada individuo (menor es mejor).
        Devuelve un escalar si se evalúa un solo individuo.

    Notas
    -----
    Usa la constante global A (normalmente A = 10), que debe estar
    definida antes de llamar a la función.
    """
    X = np.atleast_2d(X)  # Transforma X en un arreglo de al menos 2 dimensiones

    f = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

    return f if f.size > 1 else f[0]


def rastrigin_grad(x):
    """
    Gradiente analítico de la función de Rastrigin.

    Fórmula (componente i):
        df/dx_i = 2*x_i + 2*pi*A*sin(2*pi*x_i)

    Parámetros
    ----------
    x : array_like, forma (n,)
        Punto en el que se evalúa el gradiente.

    Retorna
    -------
    numpy.ndarray, forma (n,)
        Gradiente de Rastrigin en x.

    Notas
    -----
    Usa la constante global A. Requiere `import numpy as np`.
    """
    x = np.asarray(x, dtype=float)
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


# ---------------------------------------------------------------
# 1. Descenso por gradiente con gradiente analítico
# ---------------------------------------------------------------

def grad_desc(x0, grad_func, max_iter=250, learning_rate=0.002, tolerancia=1e-5,
              limites=None):
    """
    Aplica descenso por gradiente desde un punto inicial.

    Regla de actualización (en cada iteración):
        x <- clip(x - learning_rate * grad f(x), LB, UB)   (si hay límites)

    Actualiza iterativamente el punto en la dirección opuesta al gradiente,
    con una tasa de aprendizaje fija. Se detiene al alcanzar el número máximo
    de iteraciones o cuando la norma del cambio entre puntos es menor que la
    tolerancia.

    Parámetros
    ----------
    x0 : array_like, forma (n,)
        Punto inicial de la búsqueda.
    grad_func : callable
        Función que calcula el gradiente en un punto.
    max_iter : int, opcional
        Número máximo de actualizaciones. Por defecto 250.
    learning_rate : float, opcional
        Tasa de aprendizaje aplicada al gradiente. Por defecto 0.002.
    tolerancia : float, opcional
        Umbral de la norma del cambio para detener las iteraciones.
    limites : tuple (lb, ub), opcional
        Si se indica, cada punto se recorta con np.clip para no salir del
        dominio. Si es None, no se recorta.

    Retorna
    -------
    list de numpy.ndarray, cada uno de forma (n,)
        Puntos visitados, comenzando por x0 y seguido por cada punto
        calculado. La lista contiene hasta max_iter + 1 elementos.

    Notas
    -----
    No valida las formas de x0 y del gradiente; deben ser compatibles para
    la resta. Si max_iter es menor o igual que cero, retorna únicamente [x0].
    Requiere `import numpy as np`.
    """
    iteraciones = [np.array(x0, dtype=float)]
    for t in range(max_iter):
        x_new = iteraciones[t] - learning_rate * grad_func(iteraciones[t])
        if limites is not None:
            x_new = np.clip(x_new, limites[0], limites[1])
        iteraciones.append(x_new)
        cambio = np.linalg.norm(iteraciones[t + 1] - iteraciones[t])
        if cambio < tolerancia:
            break

    return iteraciones


# ---------------------------------------------------------------
# 2. Descenso por gradiente con gradiente numérico
# ---------------------------------------------------------------

def dif_marginal(x, f, i=0, h=0.001):
    """
    Aproxima la derivada parcial de f respecto a la coordenada i.

    Usa una diferencia finita hacia adelante:
        df/dx_i ~ (f(x + h*e_i) - f(x)) / h

    Parámetros
    ----------
    x : array_like, forma (n,)
        Punto en el que se aproxima la derivada.
    f : callable
        Función evaluada en el punto original y en el punto perturbado.
    i : int, opcional
        Índice de la coordenada que se perturba.
    h : float, opcional
        Tamaño de la perturbación hacia adelante.

    Retorna
    -------
    escalar
        Aproximación de la derivada parcial de f respecto a x[i].

    Notas
    -----
    i debe ser un índice válido y h debe ser distinto de cero.
    Requiere `import numpy as np`.
    """
    x_mas = np.array(x, dtype=float)
    x_mas[i] += h
    return (f(x_mas) - f(x)) / h


def gradiente_numerico(x, f, h=0.001):
    """
    Aproxima el gradiente de f en el punto x.

    Calcula cada componente mediante una diferencia finita hacia adelante
    (ver dif_marginal).

    Parámetros
    ----------
    x : array_like, forma (n,)
        Punto en el que se calcula el gradiente.
    f : callable
        Función cuyo gradiente se aproxima.
    h : float, opcional
        Tamaño de paso usado para perturbar cada coordenada.

    Retorna
    -------
    numpy.ndarray, forma (n,)
        Aproximación numérica del gradiente de f en x.

    Notas
    -----
    Para un x sin elementos retorna un array vacío de forma (0,). h debe ser
    distinto de cero y f debe poder evaluarse tanto en x como en los puntos
    perturbados.
    """
    grad = []
    for i in range(len(x)):
        grad.append(dif_marginal(x, f, i, h))
    return np.array(grad)


def grad_desc_num(x0, f_obj, max_iter=250, learning_rate=0.002, tolerancia=1e-5,
                  h=0.001, limites=None):
    """
    Minimiza una función mediante descenso por gradiente numérico.

    Regla de actualización (en cada iteración):
        x <- clip(x - learning_rate * grad_num f(x), LB, UB)   (si hay límites)

    En cada iteración aproxima el gradiente con diferencias finitas hacia
    adelante y actualiza el punto en dirección opuesta. Se detiene al
    alcanzar max_iter o cuando la norma del cambio es menor que tolerancia.

    Parámetros
    ----------
    x0 : array_like, forma (n,)
        Punto inicial, convertido a un array de tipo float.
    f_obj : callable
        Función escalar objetivo que se desea minimizar.
    max_iter : int, opcional
        Número máximo de actualizaciones. Por defecto 250.
    learning_rate : float, opcional
        Tasa de aprendizaje aplicada al gradiente aproximado.
    tolerancia : float, opcional
        Umbral de la norma del cambio para detener las iteraciones.
    h : float, opcional
        Tamaño de paso de la diferencia finita. Por defecto 0.001.
    limites : tuple (lb, ub), opcional
        Si se indica, cada punto se recorta con np.clip para no salir del
        dominio. Si es None, no se recorta.

    Retorna
    -------
    list de numpy.ndarray, cada uno de forma (n,)
        Puntos visitados, empezando por el inicial y seguido por cada punto
        calculado. La lista contiene hasta max_iter + 1 puntos.

    Notas
    -----
    Si max_iter es menor o igual que cero, retorna únicamente el punto
    inicial. x0 debe ser un vector para que gradiente_numerico pueda
    recorrer sus coordenadas. Requiere `import numpy as np`.
    """
    x = [np.array(x0, dtype=float)]
    for t in range(max_iter):
        gradiente = gradiente_numerico(x[t], f_obj, h)
        x_new = x[t] - learning_rate * gradiente
        if limites is not None:
            x_new = np.clip(x_new, limites[0], limites[1])
        x.append(x_new)
        cambio = np.linalg.norm(x[t + 1] - x[t])
        if cambio < tolerancia:
            break
    return x


# ---------------------------------------------------------------
# Utilidad: historial de la función objetivo a lo largo del recorrido
# ---------------------------------------------------------------

def historial_fitness(puntos, f=rastrigin):
    """
    Evalúa f en cada punto visitado por un descenso.

    Parámetros
    ----------
    puntos : list de numpy.ndarray, cada uno de forma (n,)
        Puntos devueltos por grad_desc o grad_desc_num.
    f : callable, opcional
        Función objetivo. Por defecto rastrigin.

    Retorna
    -------
    numpy.ndarray, forma (len(puntos),)
        Valor de f en cada punto (menor es mejor). Sirve para graficar la
        convergencia.
    """
    return np.array([f(p) for p in puntos])


