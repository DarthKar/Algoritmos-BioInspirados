"""
Optimización de la función de Rastrigin con cuatro métodos:

    1. Descenso por gradiente (gradiente analítico)
    2. Descenso por gradiente (gradiente numérico)
    3. Algoritmo evolutivo
    4. Optimización por Enjambre de Partículas (PSO)
    5. Evolución diferencial (DE)

"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 0. CONFIGURACIÓN DEL PROBLEMA
# =====================================================================

LB, UB = -5.12, 5.12  # Límites inferior y superior del dominio por variable
A = 10                # Constante de Rastrigin: profundidad de los mínimos locales


# =====================================================================
# 1. FUNCIÓN OBJETIVO Y SU GRADIENTE
# =====================================================================

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


# =====================================================================
# 2. DESCENSO POR GRADIENTE
# =====================================================================

# ---------------------------------------------------------------------
# 2.1 Con gradiente analítico
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 2.2 Con gradiente numérico
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 2.3 Utilidad: historial de la función objetivo en un recorrido
# ---------------------------------------------------------------------

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


# =====================================================================
# 3. ALGORITMO EVOLUTIVO
# =====================================================================

def pob_inicial(N=30, d=5, LB=None, UB=None):
    """Genera una población inicial aleatoria dentro de los límites dados.

    X_ij = r_ij * (UB_j - LB_j) + LB_j, con r_ij en [0, 1).

    Crea N individuos con d variables escalando números uniformes a los
    límites. No evalúa la calidad de los individuos. También se usa en PSO
    para generar las posiciones iniciales de las partículas.

    Parámetros
    ----------
    N : int
        Número de individuos (o partículas).
    d : int
        Número de variables por individuo.
    LB : array_like o None, forma (d,) o escalar
        Límite inferior. Si es None, se usa -1.
    UB : array_like o None, forma (d,) o escalar
        Límite superior. Si es None, se usa 1.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Población inicial; cada fila es un individuo.

    Notas
    -----
    Requiere numpy como np. Usa el generador global np.random. No se valida
    que LB sea menor que UB.
    """
    if LB is None:
        LB = -np.ones(d)
    if UB is None:
        UB = np.ones(d)

    return np.random.rand(N, d) * (UB - LB) + LB


def pob_eval(Pob, f_obj):
    """Evalúa y ordena una población de menor a mayor valor de f_obj.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (m, n)
        Población de m individuos con n variables.
    f_obj : callable
        Recibe un individuo de forma (n,) y devuelve un escalar.

    Retorna
    -------
    tuple de (numpy.ndarray, numpy.ndarray)
        Población ordenada, forma (m, n), y fitness ordenado, forma (m,).
        El menor fitness es mejor.

    Notas
    -----
    Requiere numpy como np. No modifica la población recibida. f_obj se
    llama fila por fila.
    """
    Pob = Pob.copy()
    fitness_val = np.zeros(Pob.shape[0])

    for i in range(Pob.shape[0]):
        fitness_val[i] = f_obj(Pob[i, :])

    ranking = np.argsort(fitness_val)
    return Pob[ranking], fitness_val[ranking]


def calcular_conteos(N, fr_elitismo, fr_mutacion):
    """Calcula cuántos individuos produce cada operador por generación.

    n_mutacion = floor(N*fr_mutacion)
    n_cruce    = 2*floor(N*(1 - fr_elitismo - fr_mutacion)/2)
    n_elite    = N - n_mutacion - n_cruce

    El conteo de cruce es siempre par, porque los individuos se cruzan por
    pares. Los élites absorben el resto, de modo que la suma siempre es N.

    Parámetros
    ----------
    N : int
        Tamaño de la población.
    fr_elitismo : float
        Fracción nominal de élites.
    fr_mutacion : float
        Fracción de mutantes.

    Retorna
    -------
    tuple de (int, int, int)
        (n_elite, n_cruce, n_mutacion).

    Notas
    -----
    Requiere numpy como np. Si fr_elitismo + fr_mutacion > 1, n_cruce es
    negativo y las funciones posteriores fallan. No se validan las fracciones.
    """
    n_mutacion = int(np.floor(N * fr_mutacion))
    n_cruce = int(2 * np.floor(N * (1 - fr_elitismo - fr_mutacion) / 2))
    n_elite = N - n_mutacion - n_cruce
    return n_elite, n_cruce, n_mutacion


def elitismo(Pob, n_elite):
    """Conserva como élite las primeras filas de la población.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (N, d)
        Población ordenada de mejor a peor.
    n_elite : int
        Número de individuos a conservar.

    Retorna
    -------
    numpy.ndarray, forma (n_elite, d)
        Copia de las primeras n_elite filas.

    Notas
    -----
    No ordena la población: se espera que ya venga ordenada por fitness.
    Con n_elite = 0 retorna un arreglo de forma (0, d).
    """
    return Pob[:n_elite, :].copy()


def cruzamiento(Pob_cruce):
    """Genera descendientes por cruce en un punto entre pares de individuos.

    La fila i se cruza con la fila i + n//2. Con un punto de corte pt:
    hijo1 = padre_i[:pt] + padre_j[pt:], hijo2 = padre_j[:pt] + padre_i[pt:].

    Parámetros
    ----------
    Pob_cruce : numpy.ndarray, forma (n, d)
        Individuos seleccionados para cruce (n debería ser par).

    Retorna
    -------
    numpy.ndarray, forma (n, d)
        Descendientes. Si n es impar, la última fila queda en ceros.

    Notas
    -----
    Requiere numpy como np. El punto de corte se sortea entre 1 y d-1, por lo
    que ningún hijo es copia exacta de un padre. Con d = 1 el corte es 0 y
    solo se intercambian los padres. Con n = 0 retorna forma (0, d).
    """
    Pob_cruce = Pob_cruce.copy()
    n_row, n_col = Pob_cruce.shape
    Pob_hijos = np.zeros_like(Pob_cruce)
    mitad = n_row // 2

    for i in range(mitad):
        pt = np.random.randint(1, n_col) if n_col > 1 else 0

        Pob_hijos[i, :] = np.concatenate(
            (Pob_cruce[i, :pt], Pob_cruce[i + mitad, pt:])
        )
        Pob_hijos[i + mitad, :] = np.concatenate(
            (Pob_cruce[i + mitad, :pt], Pob_cruce[i, pt:])
        )

    return Pob_hijos


def mutacion(Pob, id_mutantes=None, coord_mut=None, LB=None, UB=None):
    """Reemplaza una coordenada de cada individuo seleccionado por un valor uniforme.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (n, d)
        Población de n individuos con d variables.
    id_mutantes : array_like de int, forma (k,), opcional
        Índices de las filas a mutar. Por defecto, todas.
    coord_mut : array_like de int, forma (n,), opcional
        Coordenada a mutar por cada fila de Pob. Por defecto, aleatoria.
    LB : array_like, forma (d,)
        Límites inferiores.
    UB : array_like, forma (d,)
        Límites superiores.

    Retorna
    -------
    numpy.ndarray, forma (k, d)
        Copia de las filas indicadas con una coordenada mutada.

    Notas
    -----
    Requiere numpy como np. LB y UB son obligatorios aunque tengan None por
    defecto. No modifica Pob. Si hay índices repetidos en id_mutantes,
    cada repetición comparte la misma coordenada mutada.
    """
    Pob = Pob.copy()
    n_row, n_col = Pob.shape

    if id_mutantes is None:
        id_mutantes = np.arange(n_row)
    if coord_mut is None:
        coord_mut = np.random.randint(0, n_col, size=n_row)

    for i in id_mutantes:
        c = coord_mut[i]
        Pob[i, c] = np.random.rand() * (UB[c] - LB[c]) + LB[c]

    return Pob[id_mutantes, :]


def next_gen(Pob, fr_elitismo=0.2, fr_mutacion=0.1, LB=None, UB=None):
    """Construye la siguiente generación con élites, cruces y mutantes.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (N, d)
        Población ordenada de mejor a peor.
    fr_elitismo : float
        Fracción nominal de élites.
    fr_mutacion : float
        Fracción de mutantes.
    LB : array_like, forma (d,)
        Límites inferiores, requeridos por mutacion.
    UB : array_like, forma (d,)
        Límites superiores, requeridos por mutacion.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Nueva población (sin evaluar): élites, descendientes y mutantes.

    Notas
    -----
    Requiere numpy como np y las funciones calcular_conteos, elitismo,
    cruzamiento y mutacion. Los padres se eligen sin reemplazo de toda la
    población. Los mutantes se eligen con reemplazo entre las N filas. Si
    n_cruce > N o es negativo (fracciones inválidas), falla.
    """
    N = Pob.shape[0]
    n_elite, n_cruce, n_mutacion = calcular_conteos(N, fr_elitismo, fr_mutacion)

    # Élites
    Pob_elite = elitismo(Pob, n_elite)

    # Descendientes
    idx_cruce = np.random.choice(N, size=n_cruce, replace=False)
    Pob_cruce = cruzamiento(Pob[idx_cruce, :])

    # Mutantes (índices de FILA)
    id_mutantes = np.random.randint(0, N, size=n_mutacion)
    Pob_mutante = mutacion(Pob, id_mutantes=id_mutantes, LB=LB, UB=UB)

    return np.concatenate((Pob_elite, Pob_cruce, Pob_mutante), axis=0)


def actualizar_mejor(Pob, fitness, best_sol, best_fit):
    """Actualiza el mejor individuo registrado si el actual lo supera.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (N, d)
        Población ordenada de mejor a peor.
    fitness : numpy.ndarray, forma (N,)
        Fitness ordenado de menor a mayor.
    best_sol : numpy.ndarray, forma (d,)
        Mejor individuo registrado hasta ahora.
    best_fit : float
        Fitness de best_sol (menor es mejor).

    Retorna
    -------
    tuple de (numpy.ndarray, float)
        Mejor individuo (forma (d,)) y su fitness tras la comparación.

    Notas
    -----
    Solo compara fitness[0], por lo que exige que la población ya esté
    ordenada (como la devuelve pob_eval). Guarda una copia de Pob[0].
    """
    if fitness[0] < best_fit:
        return Pob[0].copy(), fitness[0]
    return best_sol, best_fit


def mi_algoritmo_evolutivo(f_obj, d, LB=-5.12, UB=5.12, N=50, max_gen=200,
                           fr_elitismo=0.1, fr_mutacion=0.2):
    """Minimiza una función objetivo con un algoritmo evolutivo de codificación real.

    Por generación: next_gen -> pob_eval -> actualizar_mejor.

    Genera una población inicial uniforme en [LB, UB], la evalúa y en cada
    generación crea la siguiente con élites, cruce en un punto y mutación.
    Todas las poblaciones creadas se evalúan, incluida la última. Sirve como
    línea base de un algoritmo genético en benchmarks como Rastrigin.

    Parámetros
    ----------
    f_obj : callable
        Recibe un individuo de forma (d,) y devuelve un escalar (menor es
        mejor). Se llama fila por fila.
    d : int
        Número de variables por individuo.
    LB : float o array_like, forma (d,)
        Límite inferior (por defecto -5.12).
    UB : float o array_like, forma (d,)
        Límite superior (por defecto 5.12).
    N : int
        Tamaño de la población (por defecto 50).
    max_gen : int
        Número de generaciones (por defecto 200).
    fr_elitismo : float
        Fracción nominal de élites (por defecto 0.1).
    fr_mutacion : float
        Fracción de mutantes (por defecto 0.2). El resto se genera por cruce.

    Retorna
    -------
    best_sol : numpy.ndarray, forma (d,)
        Mejor individuo encontrado.
    best_fit : float
        Valor de f_obj en best_sol (menor es mejor).
    historial : numpy.ndarray, forma (max_gen + 1,)
        Mejor fitness tras la población inicial y tras cada generación.
        Es monótono no creciente.

    Notas
    -----
    Requiere numpy como np y las funciones pob_inicial, pob_eval, next_gen y
    actualizar_mejor. Usa np.random global (reproducible con np.random.seed).
    LB y UB se convierten a arreglos de forma (d,). f_obj se evalúa
    N*(max_gen + 1) veces. No se validan las fracciones. Con max_gen = 0
    devuelve el mejor de la población inicial.
    """
    LB = np.broadcast_to(np.asarray(LB, dtype=float), (d,)).copy()
    UB = np.broadcast_to(np.asarray(UB, dtype=float), (d,)).copy()

    Pob = pob_inicial(N, d, LB, UB)
    Pob, fitness = pob_eval(Pob, f_obj)

    best_sol, best_fit = Pob[0].copy(), fitness[0]
    historial = np.zeros(max_gen + 1)
    historial[0] = best_fit

    for gen in range(max_gen):
        Pob = next_gen(Pob, fr_elitismo, fr_mutacion, LB, UB)
        Pob, fitness = pob_eval(Pob, f_obj)
        best_sol, best_fit = actualizar_mejor(Pob, fitness, best_sol, best_fit)
        historial[gen + 1] = best_fit

    return best_sol, best_fit, historial


# =====================================================================
# 4. OPTIMIZACIÓN POR ENJAMBRE DE PARTÍCULAS (PSO)
# =====================================================================

def vel_inicial(N, d, v_max):
    """Genera velocidades iniciales aleatorias dentro de [-v_max, v_max].

    Parámetros
    ----------
    N : int
        Número de partículas.
    d : int
        Número de variables por partícula.
    v_max : array_like, forma (d,)
        Velocidad máxima permitida en cada coordenada.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Velocidades iniciales; cada fila corresponde a una partícula.

    Notas
    -----
    Requiere numpy como np. Usa el generador global np.random.
    """
    return (2 * np.random.rand(N, d) - 1) * v_max


def evaluar_enjambre(Pob, f_obj):
    """Evalúa un enjambre sin reordenarlo.

    A diferencia de pob_eval (algoritmo evolutivo), aquí el orden de las
    filas importa: la fila i es siempre la partícula i y su pbest vive en
    la misma fila, por eso no se ordena.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (N, d)
        Posiciones de N partículas con d variables.
    f_obj : callable
        Recibe una partícula de forma (d,) y devuelve un escalar.

    Retorna
    -------
    numpy.ndarray, forma (N,)
        Fitness de cada partícula. El menor fitness es mejor.

    Notas
    -----
    Requiere numpy como np. No modifica la población recibida. f_obj se
    llama fila por fila.
    """
    fitness_val = np.zeros(Pob.shape[0])

    for i in range(Pob.shape[0]):
        fitness_val[i] = f_obj(Pob[i, :])

    return fitness_val


def actualizar_velocidad(vel, pos, pbest_pos, gbest_pos, w, c1, c2, v_max):
    """Calcula la nueva velocidad de cada partícula.

    V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X), con r1, r2 en [0, 1).

    Los tres términos son inercia, componente cognitivo (hacia el mejor
    punto propio) y componente social (hacia el mejor punto del enjambre).
    Al final se acota la velocidad a [-v_max, v_max].

    Parámetros
    ----------
    vel : numpy.ndarray, forma (N, d)
        Velocidades actuales.
    pos : numpy.ndarray, forma (N, d)
        Posiciones actuales.
    pbest_pos : numpy.ndarray, forma (N, d)
        Mejor posición histórica de cada partícula.
    gbest_pos : numpy.ndarray, forma (d,)
        Mejor posición histórica del enjambre.
    w : float
        Coeficiente de inercia.
    c1 : float
        Coeficiente cognitivo.
    c2 : float
        Coeficiente social.
    v_max : array_like, forma (d,)
        Velocidad máxima permitida en cada coordenada.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Nuevas velocidades.

    Notas
    -----
    Requiere numpy como np. Usa el generador global np.random. No modifica
    los arreglos recibidos. r1 y r2 se sortean por partícula y coordenada.
    """
    N, d = pos.shape
    r1 = np.random.rand(N, d)
    r2 = np.random.rand(N, d)

    vel_nueva = (w * vel
                 + c1 * r1 * (pbest_pos - pos)
                 + c2 * r2 * (gbest_pos - pos))

    return np.clip(vel_nueva, -v_max, v_max)


def actualizar_posicion(pos, vel, LB, UB):
    """Mueve las partículas y las mantiene dentro de los límites.

    X = clip(X + V, LB, UB)

    Parámetros
    ----------
    pos : numpy.ndarray, forma (N, d)
        Posiciones actuales.
    vel : numpy.ndarray, forma (N, d)
        Velocidades ya calculadas.
    LB : array_like, forma (d,)
        Límites inferiores.
    UB : array_like, forma (d,)
        Límites superiores.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Nuevas posiciones, dentro de [LB, UB].

    Notas
    -----
    Requiere numpy como np. No modifica los arreglos recibidos. Las
    partículas que chocan con la frontera se quedan en ella; su velocidad
    no se reinicia.
    """
    return np.clip(pos + vel, LB, UB)


def actualizar_pbest(pos, fitness, pbest_pos, pbest_fit):
    """Actualiza la mejor posición histórica de cada partícula.

    Una partícula reemplaza su pbest solo si su fitness actual es
    estrictamente menor que el registrado.

    Parámetros
    ----------
    pos : numpy.ndarray, forma (N, d)
        Posiciones actuales.
    fitness : numpy.ndarray, forma (N,)
        Fitness de las posiciones actuales.
    pbest_pos : numpy.ndarray, forma (N, d)
        Mejores posiciones registradas hasta ahora.
    pbest_fit : numpy.ndarray, forma (N,)
        Fitness de pbest_pos.

    Retorna
    -------
    tuple de (numpy.ndarray, numpy.ndarray)
        pbest_pos de forma (N, d) y pbest_fit de forma (N,) actualizados.

    Notas
    -----
    No modifica los arreglos recibidos: trabaja sobre copias. Las filas de
    pos y de pbest_pos deben corresponder a la misma partícula.
    """
    pbest_pos = pbest_pos.copy()
    pbest_fit = pbest_fit.copy()

    mejora = fitness < pbest_fit
    pbest_pos[mejora] = pos[mejora]
    pbest_fit[mejora] = fitness[mejora]

    return pbest_pos, pbest_fit


def actualizar_gbest(pbest_pos, pbest_fit, gbest_pos, gbest_fit):
    """Actualiza el mejor individuo global del enjambre.

    Parámetros
    ----------
    pbest_pos : numpy.ndarray, forma (N, d)
        Mejores posiciones individuales.
    pbest_fit : numpy.ndarray, forma (N,)
        Fitness de pbest_pos.
    gbest_pos : numpy.ndarray, forma (d,)
        Mejor posición global registrada hasta ahora.
    gbest_fit : float
        Fitness de gbest_pos (menor es mejor).

    Retorna
    -------
    tuple de (numpy.ndarray, float)
        Mejor posición global (forma (d,)) y su fitness tras la comparación.

    Notas
    -----
    Requiere numpy como np. Guarda una copia de la fila ganadora. Solo
    reemplaza si la mejora es estricta.
    """
    idx = np.argmin(pbest_fit)

    if pbest_fit[idx] < gbest_fit:
        return pbest_pos[idx].copy(), pbest_fit[idx]
    return gbest_pos, gbest_fit


def mi_pso(f_obj, d, LB=-5.12, UB=5.12, N=40, max_iter=200,
           w=0.7, c1=1.5, c2=1.5, frac_vmax=0.2):
    """Minimiza una función objetivo con Optimización por Enjambre de Partículas.

    Por iteración: actualizar_velocidad -> actualizar_posicion ->
    evaluar_enjambre -> actualizar_pbest -> actualizar_gbest.

    Genera un enjambre uniforme en [LB, UB], lo evalúa y en cada iteración
    mueve las partículas según inercia, memoria propia e información del
    grupo. Sirve como contraparte de mi_algoritmo_evolutivo en benchmarks
    como Rastrigin.

    Parámetros
    ----------
    f_obj : callable
        Recibe una partícula de forma (d,) y devuelve un escalar (menor es
        mejor). Se llama fila por fila.
    d : int
        Número de variables por partícula.
    LB : float o array_like, forma (d,)
        Límite inferior (por defecto -5.12).
    UB : float o array_like, forma (d,)
        Límite superior (por defecto 5.12).
    N : int
        Tamaño del enjambre (por defecto 40).
    max_iter : int
        Número de iteraciones (por defecto 200).
    w : float
        Coeficiente de inercia (por defecto 0.7).
    c1 : float
        Coeficiente cognitivo (por defecto 1.5).
    c2 : float
        Coeficiente social (por defecto 1.5).
    frac_vmax : float
        Fracción del rango (UB - LB) usada como velocidad máxima
        (por defecto 0.2).

    Retorna
    -------
    gbest_pos : numpy.ndarray, forma (d,)
        Mejor posición encontrada.
    gbest_fit : float
        Valor de f_obj en gbest_pos (menor es mejor).
    historial : numpy.ndarray, forma (max_iter + 1,)
        Mejor fitness tras el enjambre inicial y tras cada iteración.
        Es monótono no creciente.

    Notas
    -----
    Requiere numpy como np y las funciones pob_inicial, vel_inicial,
    evaluar_enjambre, actualizar_velocidad, actualizar_posicion,
    actualizar_pbest y actualizar_gbest. Usa np.random global (reproducible
    con np.random.seed). LB y UB se convierten a arreglos de forma (d,).
    f_obj se evalúa N*(max_iter + 1) veces. Con max_iter = 0 devuelve la
    mejor partícula del enjambre inicial.
    """
    LB = np.broadcast_to(np.asarray(LB, dtype=float), (d,)).copy()
    UB = np.broadcast_to(np.asarray(UB, dtype=float), (d,)).copy()
    v_max = frac_vmax * (UB - LB)

    # Enjambre inicial
    pos = pob_inicial(N, d, LB, UB)
    vel = vel_inicial(N, d, v_max)
    fitness = evaluar_enjambre(pos, f_obj)

    # Memoria individual y global
    pbest_pos, pbest_fit = pos.copy(), fitness.copy()
    idx = np.argmin(pbest_fit)
    gbest_pos, gbest_fit = pbest_pos[idx].copy(), pbest_fit[idx]

    historial = np.zeros(max_iter + 1)
    historial[0] = gbest_fit

    for t in range(max_iter):
        vel = actualizar_velocidad(vel, pos, pbest_pos, gbest_pos,
                                   w, c1, c2, v_max)
        pos = actualizar_posicion(pos, vel, LB, UB)
        fitness = evaluar_enjambre(pos, f_obj)

        pbest_pos, pbest_fit = actualizar_pbest(pos, fitness,
                                                pbest_pos, pbest_fit)
        gbest_pos, gbest_fit = actualizar_gbest(pbest_pos, pbest_fit,
                                                gbest_pos, gbest_fit)
        historial[t + 1] = gbest_fit

    return gbest_pos, gbest_fit, historial


# =====================================================================
# 5. EVOLUCIÓN DIFERENCIAL (DE)
# =====================================================================

def evolucion_diferencial(
    f_obj,
    d,
    LB=-5.12,
    UB=5.12,
    N=50,
    max_gen=200,
    F=0.8,
    CR=0.9,
    semilla=None
):
    """
    Minimiza una función mediante Evolución Diferencial (DE/rand/1/bin).

    En cada generación:
        1. Mutación: v = x_r1 + F * (x_r2 - x_r3)
        2. Cruce binomial entre el mutante y el individuo original.
        3. Selección: conserva el candidato con menor fitness.

    Parámetros
    ----------
    f_obj : callable
        Función objetivo que recibe un vector de forma (d,).
    d : int
        Número de variables del problema.
    LB, UB : float o array_like
        Límites inferiores y superiores por variable.
    N : int
        Tamaño de la población. Debe ser al menos 4.
    max_gen : int
        Número máximo de generaciones.
    F : float
        Factor de mutación, normalmente entre 0 y 2.
    CR : float
        Probabilidad de cruce, entre 0 y 1.
    semilla : int o None
        Semilla para reproducir la ejecución.

    Retorna
    -------
    best_sol : numpy.ndarray, forma (d,)
        Mejor solución encontrada.
    best_fit : float
        Valor de la función en la mejor solución.
    historial : numpy.ndarray, forma (max_gen + 1,)
        Mejor fitness de la población inicial y de cada generación.
    """

    if d < 1:
        raise ValueError("d debe ser mayor o igual que 1.")
    if N < 4:
        raise ValueError("N debe ser al menos 4.")
    if max_gen < 0:
        raise ValueError("max_gen no puede ser negativo.")
    if not 0 <= CR <= 1:
        raise ValueError("CR debe estar entre 0 y 1.")
    if not 0 <= F <= 2:
        raise ValueError("F debe estar entre 0 y 2.")

    rng = np.random.default_rng(semilla)

    LB = np.broadcast_to(np.asarray(LB, dtype=float), (d,)).copy()
    UB = np.broadcast_to(np.asarray(UB, dtype=float), (d,)).copy()

    if np.any(LB >= UB):
        raise ValueError("Cada límite inferior debe ser menor que el superior.")

    # 1. Población inicial
    Pob = rng.uniform(LB, UB, size=(N, d))
    fitness = np.array([f_obj(ind) for ind in Pob])

    idx = np.argmin(fitness)
    best_sol = Pob[idx].copy()
    best_fit = float(fitness[idx])

    historial = np.zeros(max_gen + 1)
    historial[0] = best_fit

    # 2. Evolución de la población
    for gen in range(max_gen):
        nueva_pob = Pob.copy()
        nuevo_fitness = fitness.copy()

        for i in range(N):
            # Elegir tres individuos diferentes de i y entre sí
            candidatos = np.delete(np.arange(N), i)
            r1, r2, r3 = rng.choice(
                candidatos, size=3, replace=False
            )

            # Mutación diferencial
            mutante = Pob[r1] + F * (Pob[r2] - Pob[r3])

            # Reparar coordenadas fuera del dominio
            mutante = np.clip(mutante, LB, UB)

            # Cruce binomial: al menos una coordenada viene del mutante
            mascara = rng.random(d) < CR
            mascara[rng.integers(d)] = True

            prueba = np.where(mascara, mutante, Pob[i])

            # Evaluación y selección
            valor_prueba = f_obj(prueba)

            if valor_prueba <= fitness[i]:
                nueva_pob[i] = prueba
                nuevo_fitness[i] = valor_prueba

        Pob = nueva_pob
        fitness = nuevo_fitness

        # Mejor solución global encontrada
        idx = np.argmin(fitness)

        if fitness[idx] < best_fit:
            best_sol = Pob[idx].copy()
            best_fit = float(fitness[idx])

        historial[gen + 1] = best_fit

    return best_sol, best_fit, historial