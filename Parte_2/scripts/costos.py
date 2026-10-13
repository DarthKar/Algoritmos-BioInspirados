"""Construye la matriz de costos del TSP a partir de los CSV de data/.

    c[i, j] = valor_hora * horas[i, j] + peaje_eur[i, j] + (km[i, j] / 100) * consumo_l_100km * precio_eur_l

Las tres componentes de un arco salen de la MISMA ruta: la ruta más rápida en
carro de HERE Routing (data/here_peaje.csv: dist_m, tiempo_s, peaje_eur). Las
matrices de OSRM (data/osrm_*.csv) no entran al costo: OSRM elige a veces otra
ruta, y mezclar su tiempo y distancia con el peaje de HERE cobra peajes de vías
que esa ruta no recorre. Quedan como verificación independiente.

La matriz es ASIMÉTRICA (c[i, j] != c[j, i] en general): la ruta más rápida de
ida no siempre es la de vuelta. Los algoritmos deben tratarla como dirigida.

Ejemplo:
    python scripts/costos.py --valor-hora 15 --consumo 5.5 --precio 1.849
"""
import argparse
import csv
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"


def leer_capitales():
    with open(DATA / "capitales.csv", encoding="utf-8") as f:
        filas = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    return [r["capital"] for r in filas], np.array([[float(r["lat"]), float(r["lon"])] for r in filas])


def leer_matriz(nombre):
    with open(DATA / nombre, encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector)
        return np.array([[float(x) for x in fila[1:]] for fila in lector])


def leer_here(n, columna, archivo="here_peaje.csv"):
    """Matriz n x n con una columna de HERE Routing (valor [i, j] del arco i -> j)."""
    M = np.full((n, n), np.nan)
    np.fill_diagonal(M, 0.0)
    with open(DATA / archivo, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r[columna] not in ("", "None"):
                M[int(r["i"]), int(r["j"])] = float(r[columna])
    if np.isnan(M).any():
        raise ValueError(f"{archivo}: faltan {int(np.isnan(M).sum())} arcos sin {columna}")
    return M


def leer_peajes(n, archivo="here_peaje.csv"):
    """Matriz de peajes en EUR (HERE Routing, peaje_eur[i, j] del arco i -> j)."""
    return leer_here(n, "peaje_eur", archivo)


def matriz_costos(valor_hora, consumo_l_100km, precio_eur_l, peajes):
    n = peajes.shape[0]
    km = leer_here(n, "dist_m") / 1000.0          # misma ruta de HERE a la que corresponde el peaje
    horas = leer_here(n, "tiempo_s") / 3600.0
    combustible = km / 100.0 * consumo_l_100km * precio_eur_l
    C = valor_hora * horas + peajes + combustible
    np.fill_diagonal(C, 0.0)
    return C, {"km": km, "horas": horas, "peajes": peajes, "combustible": combustible}


def costo_tour(C, tour):
    """Costo de un recorrido cerrado. tour = permutación de 0..n-1 (vuelve al inicio)."""
    tour = np.asarray(tour)
    return float(C[tour, np.roll(tour, -1)].sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--valor-hora", type=float, required=True, help="EUR por hora del vendedor")
    ap.add_argument("--consumo", type=float, required=True, help="L/100 km (WLTP del vehículo elegido)")
    ap.add_argument("--precio", type=float, required=True, help="EUR por litro del combustible")
    a = ap.parse_args()
    nombres, _ = leer_capitales()
    P = leer_peajes(len(nombres))
    C, partes = matriz_costos(a.valor_hora, a.consumo, a.precio, P)
    print("Peajes HERE (EUR): total", round(float(P.sum()), 2), "| arcos con peaje:", int((P > 0).sum()), "de", P.size - len(nombres))
    print("Matriz de costos", C.shape, "| asimetría máx (EUR):", round(float(np.abs(C - C.T).max()), 2))
    print("Tour 0,1,...,46 en orden de id:", round(costo_tour(C, range(len(nombres))), 2), "EUR")
    # Verificación independiente: cuánto difiere HERE de OSRM (0 en la diagonal, se excluye)
    fuera = ~np.eye(len(nombres), dtype=bool)
    dif_km = partes["km"][fuera] / (leer_matriz("osrm_distancia_m.csv")[fuera] / 1000.0) - 1
    dif_h = partes["horas"][fuera] / (leer_matriz("osrm_tiempo_s.csv")[fuera] / 3600.0) - 1
    print("HERE frente a OSRM | distancia: media", round(100 * float(dif_km.mean()), 1), "%, arcos con más de 10 %:", int((np.abs(dif_km) > 0.10).sum()),
          "| tiempo: media", round(100 * float(dif_h.mean()), 1), "%")

    