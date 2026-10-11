import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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







def pob_inicial(N=30, d=5, LB=None, UB=None):
    """Genera una población inicial aleatoria dentro de los límites dados.

    X_ij = r_ij * (UB_j - LB_j) + LB_j, con r_ij en [0, 1).

    Crea N individuos con d variables para inicializar un algoritmo de
    optimización. Si se proporcionan límites inferior y superior, las
    coordenadas se escalan a partir de ellos; la función no evalúa la calidad
    de los individuos.

    Parámetros
    ----------
    N : int
        Número de individuos de la población.
    d : int
        Número de variables por individuo.
    LB : array_like o None, forma (d,) o escalar
        Límite inferior. Si es None, se usa -1 en cada variable.
    UB : array_like o None, forma (d,) o escalar
        Límite superior. Si es None, se usa 1 en cada variable.

    Retorna
    -------
    numpy.ndarray, forma (N, d)
        Población inicial; cada fila representa un individuo y cada columna
        una variable.

    Notas
    -----
    Requiere numpy, importado como np. Usa el generador global
    np.random.rand sin establecer una semilla dentro de la función. LB y UB
    deben ser compatibles con las operaciones sobre la forma (N, d); no se
    valida que LB sea menor que UB. Para N igual a cero, retorna un array de
    forma (0, d).
    """
    if LB is None:
        LB = -np.ones(d)

    if UB is None:
        UB = np.ones(d)

    Pob_ini = np.random.rand(N, d)
    Pob_ini = Pob_ini * (UB - LB) + LB

    return Pob_ini


def pob_eval(Pob, f_obj):
    """Evalúa y ordena una población según el valor de su función objetivo.

    Evalúa cada individuo de la población y la ordena de menor a mayor
    fitness, como paso de selección o clasificación en algoritmos de
    optimización. La función no modifica la población recibida; un menor
    valor de la función objetivo se considera mejor.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (m, n)
        Población de m individuos, cada uno representado por n variables.
    f_obj : callable
        Función objetivo que recibe un individuo de forma (n,) y devuelve un
        valor numérico escalar.

    Retorna
    -------
    tuple de (numpy.ndarray, numpy.ndarray)
        Población ordenada, de forma (m, n), y valores de fitness ordenados,
        de forma (m,). El menor fitness es mejor.

    Notas
    -----
    Requiere numpy importado como np. La población se copia antes de
    evaluarla y ordenarla. Para una población vacía de forma (0, n), retorna
    arreglos con formas (0, n) y (0,), respectivamente. Cada llamada a
    f_obj debe poder asignarse a un elemento escalar del arreglo de fitness.
    """
    Pob = Pob.copy()

    n_row = Pob.shape[0]
    fitness_val = np.zeros(n_row)

    for i in range(n_row):
        fitness_val[i] = f_obj(Pob[i, :])

    # Índices ordenados de menor a mayor
    pop_ranking = np.argsort(fitness_val)

    # Valores de fitness ordenados
    f_best = fitness_val[pop_ranking]

    return Pob[pop_ranking], f_best


def elitismo(Poblacion, frac_e):
    """
    Conserva como élite las primeras filas de la población.

    Se utiliza en algoritmos evolutivos para preservar individuos entre
    generaciones. La función no ordena la población, por lo que se espera
    que sus mejores individuos estén al principio.

    Parámetros
    ----------
    Poblacion : numpy.ndarray, forma (n, d)
        Población de n individuos con d variables cada uno.
    frac_e : float
        Fracción de individuos que se conservará como élite.

    Retorna
    -------
    numpy.ndarray, forma (n_elite, d)
        Primeras filas de la población, con n_elite = round(n * frac_e)
        según el redondeo de NumPy.

    Notas
    -----
    Requiere numpy importado como np. No valida frac_e ni el orden de la
    población; la cantidad de filas seleccionadas depende también del
    slicing de NumPy si frac_e produce un valor fuera de rango.
    """
    n = Poblacion.shape[0]

    n_elite = int(np.round(n * frac_e))

    return Poblacion[0:n_elite, :]


def mutacion(Pob, id_mutantes=None, coord_mut=None, LB=None, UB=None):
    """Aplica una mutación aleatoria a las coordenadas seleccionadas.

    Para cada fila indicada, reemplaza una coordenada por un valor uniforme
    dentro de los límites correspondientes. Es un operador de mutación para
    algoritmos de optimización; devuelve únicamente las filas mutadas y no
    modifica el arreglo recibido.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (n, d)
        Población de n individuos con d variables cada uno.
    id_mutantes : array_like de int, forma (k,), opcional
        Índices de las filas que se mutarán. Por defecto, incluye todas las
        filas, de 0 a n - 1.
    coord_mut : array_like de int, forma (n,), opcional
        Índice de la coordenada que se mutará en cada fila. Por defecto,
        genera una coordenada aleatoria entre 0 y d - 1 para cada fila.
    LB : array_like, forma (d,), opcional
        Límites inferiores de las coordenadas.
    UB : array_like, forma (d,), opcional
        Límites superiores de las coordenadas.

    Retorna
    -------
    numpy.ndarray, forma (k, d)
        Copia de las filas indicadas, con la coordenada seleccionada de cada
        fila reemplazada por un valor aleatorio dentro de sus límites.

    Notas
    -----
    Requiere numpy importado como np. Aunque LB y UB tienen valores por
    defecto None, deben proporcionarse para ejecutar la mutación. Los índices
    de id_mutantes y coord_mut deben ser válidos para Pob y para los límites.
    """
    Pob = Pob.copy()
    n_row, n_col = Pob.shape

    if id_mutantes is None:
        id_mutantes = range(n_row)

    if coord_mut is None:
        coord_mut = np.random.randint(0, n_col, size=n_row)

    for i in id_mutantes:
        mutacion_val = (
            np.random.rand()
            * (UB[coord_mut[i]] - LB[coord_mut[i]])
            + LB[coord_mut[i]]
        )

        Pob[i, coord_mut[i]] = mutacion_val

    return Pob[id_mutantes, :]


def mutacion_gaussiana(Poblacion, frac_m, lim_inf, lim_sup):
    """
    Aplica mutación gaussiana a una coordenada de cada individuo seleccionado.

    x'_ij = clip(x_ij + z*sigma_j, lim_inf[j], lim_sup[j]), z ~ N(0, 1)

    Operador estocástico de variación para poblaciones en optimización. Cada
    individuo seleccionado recibe una perturbación en una sola coordenada; su
    desviación estándar es la desviación de esa coordenada en la población.

    Parámetros
    ----------
    Poblacion : numpy.ndarray, forma (n, m)
        Población de n individuos con m coordenadas.
    frac_m : float
        Fracción de individuos que se seleccionará para mutación.
    lim_inf : array_like, forma (m,)
        Límites inferiores por coordenada.
    lim_sup : array_like, forma (m,)
        Límites superiores por coordenada.

    Retorna
    -------
    numpy.ndarray, forma (n_mutantes, m)
        Copia de los individuos seleccionados con una coordenada mutada y
        recortada a sus límites; no incluye los individuos no seleccionados.

    Notas
    -----
    Requiere NumPy importado como np. n_mutantes = int(n * frac_m); debe estar
    entre 0 y n para que la selección sin reemplazo sea válida. Si n_mutantes
    es 0, retorna un arreglo vacío de forma (0, m). Las desviaciones estándar
    se calculan sobre Poblacion con ddof=0. Los límites deben poder indexarse
    por coordenada.
    """
    Poblacion_mutada = Poblacion.copy()

    n, m = Poblacion.shape

    n_mutantes = int(n * frac_m)

    indices_mutantes = np.random.choice(
        n,
        n_mutantes,
        replace=False
    )

    posicion_coordenada = np.random.choice(
        m,
        n_mutantes,
        replace=True
    )

    desv_est = Poblacion.std(axis=0)

    desviaciones = desv_est[posicion_coordenada]

    desviacion_a_sumar = (
        np.random.standard_normal(n_mutantes)
        * desviaciones
    )

    nuevos_valores = (
        Poblacion[indices_mutantes, posicion_coordenada]
        + desviacion_a_sumar
    )

    Poblacion_mutada[
        indices_mutantes,
        posicion_coordenada
    ] = np.clip(
        nuevos_valores,
        lim_inf[posicion_coordenada],
        lim_sup[posicion_coordenada]
    )

    return Poblacion_mutada[indices_mutantes, :]


def cruzamiento(Pob_cruce):
    """Genera descendientes intercambiando segmentos entre pares de individuos.

    En cada par, selecciona un punto de corte aleatorio y combina el segmento
    inicial de un individuo con el segmento final del otro. Es un operador de
    cruzamiento para algoritmos genéticos; no evalúa la aptitud ni garantiza
    mejoras en la población.

    Parámetros
    ----------
    Pob_cruce : numpy.ndarray, forma (n, d)
        Población de n individuos con d variables cada uno.

    Retorna
    -------
    numpy.ndarray, forma (n, d)
        Población de descendientes. Las filas se cruzan por pares entre la
        primera y la segunda mitad; si n es impar, la última fila queda en
        ceros.

    Notas
    -----
    Requiere NumPy importado como np. Usa el generador aleatorio global de
    NumPy. Si n es mayor que uno y d es cero, la selección del punto de corte
    produce un error. No valida la dimensionalidad de la entrada.
    """
    Pob_cruce = Pob_cruce.copy()

    n_row, n_col = Pob_cruce.shape

    Pob_hijos = np.zeros_like(Pob_cruce)

    n_cruzamiento = int(np.floor(n_row / 2))

    for i in range(n_cruzamiento):

        coord_cruzamiento = np.random.randint(
            low=0,
            high=n_col,
            size=1
        )

        # Hijo 1:
        # Parte izquierda de Padre i +
        # parte derecha de Padre (i + n_cruzamiento)
        Pob_hijos[i, :] = np.concatenate(
            (
                Pob_cruce[i, :coord_cruzamiento[0]],
                Pob_cruce[i + n_cruzamiento, coord_cruzamiento[0]:]
            )
        )

        # Hijo 2:
        # Parte izquierda de Padre (i + n_cruzamiento) +
        # parte derecha de Padre i
        Pob_hijos[i + n_cruzamiento, :] = np.concatenate(
            (
                Pob_cruce[i + n_cruzamiento, :coord_cruzamiento[0]],
                Pob_cruce[i, coord_cruzamiento[0]:]
            )
        )

    return Pob_hijos


def next_gen(Pob, fr_elitismo=0.2, fr_mutacion=0.1, LB=None, UB=None):
    """Construye una nueva generación mediante elitismo, cruzamiento y mutación.

    Conserva las primeras filas como élite, cruza una selección aleatoria de
    individuos y agrega las filas mutadas. Es un operador de reemplazo para
    algoritmos genéticos; no evalúa ni ordena los individuos por su aptitud.

    Parámetros
    ----------
    Pob : numpy.ndarray, forma (n, d)
        Población de n individuos con d variables cada uno.
    fr_elitismo : float
        Fracción nominal destinada al elitismo; el número efectivo de élites
        se obtiene como diferencia respecto a los conteos de mutación y cruce.
    fr_mutacion : float
        Fracción destinada a mutación; el conteo se calcula como floor(n *
        fr_mutacion).
    LB : array_like, forma compatible con (d,), opcional
        Límites inferiores requeridos por la función mutacion.
    UB : array_like, forma compatible con (d,), opcional
        Límites superiores requeridos por la función mutacion.

    Retorna
    -------
    numpy.ndarray, forma normalmente (n, d)
        Nueva población compuesta por élites, descendientes cruzados y filas
        mutadas. No garantiza una mejora de aptitud; esta debe evaluarse por
        separado (menor es mejor en problemas de minimización).

    Notas
    -----
    Requiere numpy importado como np y las funciones cruzamiento y mutacion.
    Usa el generador aleatorio global de NumPy. Las élites son las primeras
    filas y la población no se ordena por aptitud. El cruzamiento selecciona
    individuos sin reemplazo; mutacion requiere que LB y UB estén definidos.
    No se validan las fracciones ni las dimensiones. El número de individuos
    cruzados se calcula como 2 * floor(n * (1 - fr_elitismo - fr_mutacion) / 2).
    """
    Pob = Pob.copy()

    n_row, n_col = Pob.shape

    n_mutacion = int(np.floor(n_row * fr_mutacion))

    n_cruzamiento = int(
        2 * np.floor(
            n_row * (1 - fr_elitismo - fr_mutacion) / 2
        )
    )

    n_elite = n_row - n_mutacion - n_cruzamiento

    # Individuos élite
    Pob_elite = Pob[:n_elite, :]

    # Individuos para cruzamiento
    indices_cruzamiento = np.random.choice(
        n_row,
        size=n_cruzamiento,
        replace=False
    )

    Pob_cruce = cruzamiento(
        Pob[indices_cruzamiento, :]
    )

    # Individuos mutantes
    id_mutantes = np.random.randint(
        low=0,
        high=n_col,
        size=n_mutacion
    )

    Pob_mutante = mutacion(
        Pob,
        id_mutantes=id_mutantes,
        LB=LB,
        UB=UB
    )

    # Nueva generación
    Pob_next_gen = np.concatenate(
        (
            Pob_elite,
            Pob_cruce,
            Pob_mutante
        ),
        axis=0
    )

    return Pob_next_gen


def mi_algoritmo_evolutivo(
    f_obj,
    N=30,
    d=3,
    LB=None,
    UB=None,
    fr_elitismo=0.2,
    fr_mutacion=0.10,
    max_iter=50,
    semilla=None
):
    """Ejecuta un algoritmo evolutivo durante un número fijo de iteraciones.

    Inicializa una población y, en cada iteración, la evalúa, registra el mejor
    valor y genera la siguiente población. La calidad y el criterio de ordenación
    dependen de `pob_eval`; el algoritmo no incluye una condición de parada por
    convergencia.

    Parámetros
    ----------
    f_obj : callable
        Función objetivo que recibe individuos según lo requerido por `pob_eval`.
    N : int, opcional
        Número de individuos de la población inicial.
    d : int, opcional
        Número de variables de cada individuo.
    LB : array_like o escalar, opcional
        Límite inferior de las variables, usado por la inicialización y la
        generación de descendientes.
    UB : array_like o escalar, opcional
        Límite superior de las variables, usado por la inicialización y la
        generación de descendientes.
    fr_elitismo : float, opcional
        Fracción de la población reservada para elitismo en `next_gen`.
    fr_mutacion : float, opcional
        Fracción de la población asignada a mutación en `next_gen`.
    max_iter : int, opcional
        Número de iteraciones; se generan `max_iter` poblaciones nuevas.
    semilla : int o None, opcional
        Semilla para el generador aleatorio de NumPy. Si es `None`, no se
        reinicia la semilla.

    Retorna
    -------
    tuple
        `(Pob, np.array(f_best))`, donde `Pob` es una lista de `max_iter + 1`
        poblaciones, cada una normalmente con forma (N, d), y `f_best` es un
        array de forma (max_iter,) con el mejor valor registrado en cada
        iteración (menor es mejor si así lo ordena `pob_eval`).

    Notas
    -----
    Requiere NumPy y las funciones `pob_inicial`, `pob_eval` y `next_gen`.
    Si `semilla` no es `None`, modifica el estado global del generador aleatorio
    de NumPy. Con `max_iter=0`, retorna únicamente la población inicial y un
    array vacío de valores registrados.
    """
    if semilla is not None:
        np.random.seed(semilla)

    # Población inicial
    Pob0 = pob_inicial(
        N=N,
        d=d,
        LB=LB,
        UB=UB
    )

    Pob = [Pob0]

    f_best = []

    for i in range(max_iter):

        # Evaluar y ordenar la población
        Pob_ranked, f_best_gen_i = pob_eval(
            Pob[i],
            f_obj
        )

        f_best.append(f_best_gen_i)

        # Generar la siguiente generación
        Pob.append(
            next_gen(
                Pob_ranked,
                fr_elitismo=fr_elitismo,
                fr_mutacion=fr_mutacion,
                LB=LB,
                UB=UB
            )
        )

    return Pob, np.array(f_best)
