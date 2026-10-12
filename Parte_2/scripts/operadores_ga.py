"""Operadores genéticos para permutaciones (TSP).

Los cruces de PyGAD (single_point, two_points, uniform, scattered) repiten ciudades,
por eso el cruce se escribe aquí. Para la mutación se usa la de PyGAD
(mutation_type="inversion" o "swap"), que sí conserva la permutación.

Regla de reproducibilidad: todo azar sale de ga.numpy_random_generator,
nunca de np.random. Con np.random, la misma semilla da corridas distintas.
"""
import numpy as np


def ox_par(a, b, rng):
    """Order Crossover (OX) de un par de padres.

    Copia un segmento de `a` en la misma posición y llena los huecos con las
    ciudades restantes en el orden en que aparecen en `b` (de izquierda a derecha).
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    hijo = np.full(n, -1, dtype=int)
    hijo[i:j + 1] = a[i:j + 1]
    en_segmento = np.zeros(n, dtype=bool)
    en_segmento[a[i:j + 1]] = True
    hijo[hijo == -1] = b[~en_segmento[b]]
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
