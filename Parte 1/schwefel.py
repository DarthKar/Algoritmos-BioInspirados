import numpy as np
import pandas as pd


SEED = np.random.seed()


def grad_desc(x0, grad_func, max_iter = 50, learning_rate=0.01, tolerancia=0.00001):
    """Aplica descenso por gradiente desde un punto inicial.

    Actualiza iterativamente el punto en la dirección opuesta al gradiente,
    con una tasa de aprendizaje fija. Se detiene al alcanzar el número máximo
    de iteraciones o cuando la norma del cambio entre puntos es menor que la
    tolerancia.

    Parámetros
    ----------
    x0 : escalar o numpy.ndarray, forma compatible con grad_func
        Punto inicial de la búsqueda.
    grad_func : callable
        Función que calcula el gradiente en un punto.
    max_iter : int
        Número máximo de actualizaciones.
    learning_rate : float
        Tasa de aprendizaje aplicada al gradiente.
    tolerancia : float
        Umbral de la norma del cambio para detener las iteraciones.

    Retorna
    -------
    list
        Puntos visitados, comenzando por x0 y seguido por cada punto
        calculado. La lista contiene hasta max_iter + 1 elementos.

    Notas
    -----
    Requiere numpy, importado como np, para calcular la norma euclídea del
    cambio. No evalúa la función objetivo ni valida las formas de x0 y del
    gradiente; estos deben ser compatibles para la resta. Si max_iter es
    menor o igual que cero, retorna únicamente [x0].
    """

    Iteraciones = [x0]
    for t in range(max_iter):
        x_new = Iteraciones[t]-learning_rate*grad_func(Iteraciones[t])
        Iteraciones.append(x_new)
        cambio = np.linalg.norm(Iteraciones[t+1]-Iteraciones[t])
        if cambio < tolerancia:
            break

    return Iteraciones


def dif_marginal(x, f, i=0, h=0.001):
    """Aproxima la derivada parcial de f respecto a la coordenada i.

    Usa una diferencia finita hacia adelante para estimar la variación de f
    al perturbar una coordenada. Puede emplearse para construir gradientes
    numéricos en métodos de optimización.

    Parámetros
    ----------
    x : array_like, forma (n,)
        Punto en el que se aproxima la derivada.
    f : callable
        Función evaluada en el punto original y en el punto perturbado.
    i : int
        Índice de la coordenada que se perturba.
    h : float
        Tamaño de la perturbación hacia adelante.

    Retorna
    -------
    escalar
        Aproximación de la derivada parcial de f respecto a x[i].

    Notas
    -----
    Requiere numpy, importado como np. x_mas es una copia de x convertida a
    tipo float; f(x) recibe x sin conversión. i debe ser un índice válido y h
    debe ser distinto de cero.
    """
    x_mas = np.array(x, dtype=float)
    x_mas[i] += h
    return (f(x_mas) - f(x)) / h

def gradiente_numerico(x, f, h=0.001):
    """Aproxima el gradiente de f en el punto x.

    Calcula cada componente mediante una diferencia finita hacia adelante.
    El gradiente resultante puede utilizarse en algoritmos de optimización
    que actualizan soluciones usando derivadas.

    Parámetros
    ----------
    x : array_like, forma (n,)
        Punto en el que se calcula el gradiente.
    f : callable
        Función cuyo gradiente se aproxima.
    h : float
        Tamaño de paso usado para perturbar cada coordenada.

    Retorna
    -------
    numpy.ndarray, forma (n,)
        Aproximación numérica del gradiente de f en x.

    Notas
    -----
    Requiere numpy, importado como np. Para un x sin elementos retorna un
    array vacío de forma (0,). h debe ser distinto de cero y f debe poder
    evaluarse tanto en x como en los puntos perturbados.
    """
    grad = []
    for i in range(len(x)):
        grad.append(dif_marginal(x, f, i, h))
    return np.array(grad)

def grad_desc_num(x0, f_obj, max_iter=50, learning_rate=0.01, tolerancia=0.00001):
    """Minimiza una función mediante descenso por gradiente numérico.

    En cada iteración aproxima el gradiente con diferencias finitas hacia
    adelante y actualiza el punto en dirección opuesta al gradiente. Se
    detiene al alcanzar max_iter o cuando la norma del cambio es menor que
    tolerancia.

    Parámetros
    ----------
    x0 : array_like, forma (n,)
        Punto inicial, convertido a un array de tipo float.
    f_obj : callable
        Función escalar objetivo que se desea minimizar.
    max_iter : int
        Número máximo de actualizaciones.
    learning_rate : float
        Tasa de aprendizaje aplicada al gradiente aproximado.
    tolerancia : float
        Umbral de la norma del cambio para detener las iteraciones.

    Retorna
    -------
    list de numpy.ndarray, cada uno de forma (n,)
        Puntos visitados, empezando por el inicial y seguido por cada punto
        calculado. La lista contiene hasta max_iter + 1 puntos; entre las
        soluciones visitadas, un menor valor de f_obj es mejor.

    Notas
    -----
    Requiere numpy, importado como np. El gradiente usa h=0.001, valor
    predeterminado de gradiente_numerico. Si max_iter es menor o igual que
    cero, retorna únicamente el punto inicial. x0 debe ser un vector para que
    gradiente_numerico pueda recorrer sus coordenadas.
    """
    x = [np.array(x0, dtype=float)]
    for t in range(max_iter):
        gradiente = gradiente_numerico(x[t], f_obj)
        x_new = x[t] - learning_rate * gradiente
        x.append(x_new)
        cambio = np.linalg.norm(x[t+1] - x[t])
        if cambio < tolerancia:
            break
    return x
