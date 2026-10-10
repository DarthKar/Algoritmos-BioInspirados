"""Construye la matriz de costos del TSP a partir de los CSV de data/.

    c[i, j] = valor_hora * horas[i, j] + peaje_eur[i, j] + (km[i, j] / 100) * consumo_l_100km * precio_eur_l

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


def leer_peajes(n, archivo="here_peaje.csv"):
    """Matriz de peajes en EUR. Si todavía no existe el archivo de HERE, devuelve None."""
    ruta = DATA / archivo
    if not ruta.exists():
        return None
    P = np.full((n, n), np.nan)
    np.fill_diagonal(P, 0.0)
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["peaje_eur"] not in ("", "None"):
                P[int(r["i"]), int(r["j"])] = float(r["peaje_eur"])
    if np.isnan(P).any():
        raise ValueError(f"{archivo}: faltan {int(np.isnan(P).sum())} arcos sin peaje consultado")
    return P


def matriz_costos(valor_hora, consumo_l_100km, precio_eur_l, peajes=None):
    km = leer_matriz("osrm_distancia_m.csv") / 1000.0
    horas = leer_matriz("osrm_tiempo_s.csv") / 3600.0
    n = km.shape[0]
    if peajes is None:
        peajes = np.zeros((n, n))   # SOLO para desarrollo: sin peajes reales el costo está subestimado
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
    print("Peajes:", "HERE" if P is not None else "NO CARGADOS (0 EUR, solo desarrollo)")
    print("Matriz de costos", C.shape, "| asimetría máx (EUR):", round(float(np.abs(C - C.T).max()), 2))
    print("Tour 0,1,...,46 en orden de id:", round(costo_tour(C, range(len(nombres))), 2), "EUR")

    