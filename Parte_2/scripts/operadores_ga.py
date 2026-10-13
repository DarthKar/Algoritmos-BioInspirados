"""Operadores genéticos para permutaciones (TSP).

Los cruces de PyGAD (single_point, two_points, uniform, scattered) repiten ciudades,
por eso el cruce se escribe aquí.

La mutación también se escribe aquí. Las de PyGAD 3.8.1 ("inversion", "swap",
"scramble") conservan la permutación, pero con mutation_probability sortean cada
gen por separado y operan sobre esas posiciones dispersas: "inversion" no invierte
un tramo contiguo del recorrido y la probabilidad no es la de mutar un individuo.

Regla de reproducibilidad: todo azar sale de ga.numpy_random_generator,
nunca de np.random. Con np.random, la misma semilla da corridas distintas.
"""
import numpy as np


def ox_par(a, b, rng):
    """Order Crossover (OX) de un par de padres.

    Copia un segmento de `a` en la misma posición. Las ciudades restantes se
    toman de `b` empezando después del segundo corte (de forma circular) y se
    colocan en los huecos, también empezando después del segundo corte.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    hijo = np.full(n, -1, dtype=int)
    hijo[i:j + 1] = a[i:j + 1]
    en_segmento = np.zeros(n, dtype=bool)
    en_segmento[a[i:j + 1]] = True
    b_desde_corte = np.roll(b, -(j + 1))                             # b leído desde la posición j+1
    huecos = np.roll(np.arange(n), -(j + 1))[:n - (j - i + 1)]       # j+1, ..., n-1, 0, ..., i-1
    hijo[huecos] = b_desde_corte[~en_segmento[b_desde_corte]]
    return hijo


def pmx_par(a, b, rng):
    """Partially Mapped Crossover (PMX) de un par de padres."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    hijo = np.full(n, -1, dtype=int)
    hijo[i:j + 1] = a[i:j + 1]
    pos_en_b = np.empty(n, dtype=int)
    pos_en_b[b] = np.arange(n)
    segmento = set(a[i:j + 1].tolist())
    for k in range(i, j + 1):
        g = b[k]
        if g in segmento:
            continue
        p = k
        while i <= p <= j:            # seguir el mapeo a[p] -> posición de a[p] en b
            p = pos_en_b[a[p]]
        hijo[p] = g
    vacios = hijo == -1
    hijo[vacios] = b[vacios]
    return hijo


def _hacer_cruce(funcion_par):
    def cruce(parents, offspring_size, ga_instance):
        rng = ga_instance.numpy_random_generator
        m = len(parents)
        hijos = np.empty(offspring_size, dtype=int)
        for k in range(offspring_size[0]):
            hijos[k] = funcion_par(parents[k % m], parents[(k + 1) % m], rng)
        return hijos
    cruce.__name__ = funcion_par.__name__.replace("_par", "")
    return cruce


ox = _hacer_cruce(ox_par)
pmx = _hacer_cruce(pmx_par)
CRUCES = {"ox": ox, "pmx": pmx}


def inversion_ind(tour, rng):
    """Invierte un tramo contiguo del recorrido (dos cortes al azar).

    Cambia 2 arcos en los extremos del tramo y recorre el tramo en sentido
    contrario. Como la matriz es asimétrica, eso también cambia el costo interno.
    """
    i, j = sorted(rng.choice(len(tour), size=2, replace=False))
    tour[i:j + 1] = tour[i:j + 1][::-1].copy()


def insercion_ind(tour, rng):
    """Saca la ciudad de la posición i y la inserta en la posición j.

    Cambia 3 arcos y ninguna ciudad intermedia cambia de sentido.
    """
    i, j = rng.choice(len(tour), size=2, replace=False)
    if i < j:
        tour[i:j + 1] = np.roll(tour[i:j + 1], -1)
    else:
        tour[j:i + 1] = np.roll(tour[j:i + 1], 1)


def swap_ind(tour, rng):
    """Intercambia las ciudades de dos posiciones al azar (cambia hasta 4 arcos)."""
    i, j = rng.choice(len(tour), size=2, replace=False)
    tour[i], tour[j] = tour[j], tour[i]


def mixta_ind(tour, rng):
    """Aplica inversión o inserción, cada una con probabilidad 1/2."""
    if rng.random() < 0.5:
        inversion_ind(tour, rng)
    else:
        insercion_ind(tour, rng)


MUTACIONES = {"inversion": inversion_ind, "insercion": insercion_ind, "swap": swap_ind, "mixta": mixta_ind}


def hacer_mutacion(nombre, p_mutacion):
    """Mutación para PyGAD: cada hijo muta una sola vez con probabilidad p_mutacion."""
    funcion_ind = MUTACIONES[nombre]

    def mutacion(offspring, ga_instance):
        rng = ga_instance.numpy_random_generator
        for k in range(offspring.shape[0]):
            if rng.random() < p_mutacion:
                funcion_ind(offspring[k], rng)
        return offspring
    mutacion.__name__ = nombre
    return mutacion
