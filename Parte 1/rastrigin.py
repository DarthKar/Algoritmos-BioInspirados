import numpy as np
import matplotlib.pyplot as plt

# --- Configuración del problema (función de Rastrigin) ---
LB, UB = -5.12, 5.12  # Límites inferior y superior del dominio de búsqueda por variable
A = 10                # Constante de Rastrigin: controla la profundidad de los mínimos locales

# ---------------------------------------------------------------
# Función objetivo (acepta un punto (n,) o una población (m, n))
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

    X = np.atleast_2d(X) # Transforma el X en un arreglo de al menos 2 dimensiones

    f = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

    return f if f.size > 1 else f[0] 


def rastrigin_grad(x):
    """
    Derivada de la funcion rastrigin
    """
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


# ---------------------------------------------------------------
# 1. Descenso por gradiente
# ---------------------------------------------------------------
def gradient_descent(dim, lr=0.002, max_iter=250, rng=None):
    """
    Minimiza la función de Rastrigin con descenso de gradiente desde un punto inicial aleatorio.

    Regla de actualización (en cada iteración):
        x <- clip(x - lr * grad f(x), LB, UB)

    El descenso de gradiente es un método de optimización local: desde un
    punto inicial, se mueve repetidamente en la dirección opuesta al gradiente
    de la función. Como Rastrigin es multimodal, el método converge al mínimo
    local más cercano al punto inicial y, en general, no garantiza alcanzar el
    mínimo global en x = (0, ..., 0). Se usa como línea base para comparar
    contra los algoritmos bioinspirados.

    Parámetros
    ----------
    dim : int
        Número de variables (dimensión) del problema.
    lr : float, opcional
        Tasa de aprendizaje: tamaño del paso en cada iteración. Por defecto 0.002.
    max_iter : int, opcional
        Número de iteraciones del descenso. Por defecto 250.
    rng : numpy.random.Generator, opcional
        Generador de números aleatorios para la condición inicial. Si es None,
        se crea uno nuevo sin semilla fija. Pasar un generador con semilla
        permite reproducir los resultados.

    Retorna
    -------
    x : numpy.ndarray, forma (dim,)
        Posición final alcanzada tras max_iter iteraciones.
    fitness : float
        Valor de Rastrigin en la posición final (menor es mejor).
    best_hist : numpy.ndarray, forma (max_iter + 1,)
        Historial del valor de la función: el primer elemento corresponde al
        punto inicial y los siguientes a cada iteración. Sirve para graficar
        la convergencia.

    Notas
    -----
    Depende de las constantes globales LB y UB (límites del dominio) y de las
    funciones rastrigin y rastrigin_grad, que deben estar definidas antes de
    llamar a esta función. Requiere `import numpy as np`.
    Tras cada paso, la posición se recorta con np.clip para no salir de
    [LB, UB]. Una sola ejecución usa un único punto inicial, por lo que el
    resultado depende de la semilla.
    """
     
    rng = rng or np.random.default_rng()
    x = rng.uniform(LB, UB, dim)    # condición inicial aleatoria
    mejor_hist = [rastrigin(x)]

    for _ in range(max_iter):
        x = np.clip(x - lr * rastrigin_grad(x), LB, UB)
        mejor_hist.append(rastrigin(x))
    return x, rastrigin(x), np.array(mejor_hist)




