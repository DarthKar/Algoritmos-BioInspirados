"""Algoritmo genético para el TSP asimétrico de las 47 capitales, con PyGAD.

Representación: permutación de los ids 0..46 (recorrido cerrado).
Aptitud: -costo_tour (PyGAD maximiza).

Guarda en outputs/ (raíz del repositorio), con el sufijo dado en --version:
    resultados_GA_<version>.json   parámetros, semilla, mejor recorrido, costo y su desglose
    Recorrido_GA_<version>.png     mejor recorrido sobre el mapa
    Evolucion_GA_<version>.gif     evolución del mejor recorrido

Ejemplo:
    python Parte_2/scripts/algoritmo_genetico.py --valor-hora 15 --version 0001
    python Parte_2/scripts/algoritmo_genetico.py --valor-hora 25 --poblacion 200 --generaciones 800 --cruce pmx --mutacion insercion --p-mutacion 0.5 --semilla 7 --version 0002
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pygad
from matplotlib.animation import FuncAnimation, PillowWriter

from costos import DATA, RAIZ, costo_tour, leer_capitales, leer_peajes, matriz_costos
from operadores_ga import CRUCES, MUTACIONES, hacer_mutacion

SALIDAS = RAIZ.parent / "outputs"


def construir_y_correr(C, cfg, semilla):
    """Corre el GA. Devuelve un dict con el mejor tour, su costo, evaluaciones e historial.

    cfg: dict con claves poblacion, generaciones, padres, elite, k_torneo,
         cruce ("ox" | "pmx"), mutacion ("inversion" | "insercion" | "swap"),
         p_mutacion (probabilidad de que un hijo mute, entre 0 y 1).
    """
    n = C.shape[0]
    contador = {"evals": 0}
    historial = {"mejor_costo": [], "mejor_tour": [], "evals": []}

    def fitness(ga_instance, solution, solution_idx):
        # Cuenta llamadas reales. PyGAD no llama a esta función para los élites ni para
        # los hijos idénticos a un élite o a un padre: reutiliza la aptitud ya calculada.
        contador["evals"] += 1
        return -costo_tour(C, solution.astype(int))

    def on_generation(ga_instance):
        sol, fit, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        historial["mejor_costo"].append(float(-fit))
        historial["mejor_tour"].append(sol.astype(int).tolist())
        historial["evals"].append(contador["evals"])

    rng = np.random.default_rng(semilla)
    poblacion_inicial = np.array([rng.permutation(n) for _ in range(cfg["poblacion"])])

    ga = pygad.GA(
        num_generations=cfg["generaciones"],
        sol_per_pop=cfg["poblacion"],
        num_parents_mating=cfg["padres"],
        num_genes=n,
        initial_population=poblacion_inicial,
        gene_type=int,
        fitness_func=fitness,
        parent_selection_type="tournament",
        K_tournament=cfg["k_torneo"],
        crossover_type=CRUCES[cfg["cruce"]],
        mutation_type=hacer_mutacion(cfg["mutacion"], cfg["p_mutacion"]),
        keep_elitism=cfg["elite"],
        on_generation=on_generation,
        random_seed=semilla,
        suppress_warnings=True,
    )
    t0 = time.perf_counter()
    ga.run()
    seg = time.perf_counter() - t0

    sol, fit, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    tour = sol.astype(int)
    assert sorted(tour.tolist()) == list(range(n)), "el mejor individuo no es una permutación"
    return {
        "semilla": semilla,
        "mejor_tour": tour.tolist(),
        "mejor_costo": costo_tour(C, tour),
        "evaluaciones_fitness": contador["evals"],
        # Tope si se evaluara a todos: población inicial + hijos de cada generación.
        "evaluaciones_nominales": cfg["poblacion"] + cfg["generaciones"] * (cfg["poblacion"] - cfg["elite"]),
        "segundos": round(seg, 2),
        "historial": historial,
    }


def leer_contorno_espana():
    """Polígonos (lon, lat) del contorno de España, del GeoJSON guardado en data/."""
    with open(DATA / "espana.geojson", encoding="utf-8") as f:
        geometria = json.load(f)["features"][0]["geometry"]
    return [np.array(poligono[0]) for poligono in geometria["coordinates"]]


def crear_animacion(historial, ubicaciones, version, max_cuadros=60, fps=5, exportar_gif=True):
    """Dibuja la evolución del mejor recorrido sobre el mapa (GIF) y el recorrido final (PNG).

    Adaptada de crear_animacion de colonia_de_hormigas.py, con el mismo estilo.
    Un cuadro por cada generación en que mejoró el mejor costo; si son más de
    max_cuadros, se toma una muestra repartida de forma uniforme.
    """
    tours = historial["mejor_tour"]
    costos = historial["mejor_costo"]

    indices = [0] + [g for g in range(1, len(costos)) if costos[g] < costos[g - 1]]
    if len(indices) > max_cuadros:
        muestra = np.linspace(0, len(indices) - 1, max_cuadros).astype(int)
        indices = [indices[k] for k in muestra]
    indices = indices + [indices[-1]] * (2 * fps)        # pausa de 2 s en el recorrido final

    fig, ax = plt.subplots(figsize=(11, 9))

    # Fondo cartográfico de España
    for poligono in leer_contorno_espana():
        ax.fill(poligono[:, 0], poligono[:, 1], facecolor="#e8efe3", edgecolor="#52665b", linewidth=1.3)

    # Delimitar la España peninsular
    ax.set_xlim(-10, 4.5)
    ax.set_ylim(35.5, 44.5)
    ax.set_aspect("equal")

    ax.scatter(ubicaciones[:, 1], ubicaciones[:, 0], color="#173d68", s=22, zorder=5, label="Capitales")
    linea, = ax.plot([], [], color="red", linewidth=1.6, alpha=0.85, zorder=4, label="Mejor recorrido")
    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=11,
        bbox=dict(facecolor="white", alpha=0.90, edgecolor="gray", boxstyle="round,pad=0.6"),
    )

    ax.set_title("Optimización de rutas - Algoritmo Genético", fontsize=15)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)

    def actualizar(frame):
        g = indices[frame]
        tour = tours[g]
        coordenadas = ubicaciones[tour + [tour[0]]]       # se repite la primera ciudad para cerrar el recorrido
        linea.set_data(coordenadas[:, 1], coordenadas[:, 0])
        info.set_text(
            f"Generación: {g + 1} / {len(tours)}\n"
            f"Evaluaciones: {historial['evals'][g]:,}\n"
            f"Mejor costo: {costos[g]:,.2f} euros"
        )
        return linea, info

    if exportar_gif:
        animacion = FuncAnimation(fig, actualizar, frames=len(indices), interval=1000 / fps, blit=False, repeat=False)
        animacion.save(SALIDAS / f"Evolucion_GA_{version}.gif", writer=PillowWriter(fps=fps), dpi=90)
        print("GIF guardado correctamente.")

    actualizar(len(indices) - 1)
    fig.savefig(SALIDAS / f"Recorrido_GA_{version}.png", dpi=120)
    print("PNG guardado correctamente.")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Algoritmo genético para el TSP de las 47 capitales peninsulares")

    ap.add_argument("--valor-hora", type=float, required=True, help="EUR por hora del vendedor")
    ap.add_argument("--consumo", type=float, default=5.5, help="L/100 km (WLTP del vehículo elegido)")
    ap.add_argument("--precio", type=float, default=1.849, help="EUR por litro del combustible")
    ap.add_argument("--poblacion", type=int, default=100, help="Tamaño de la población")
    ap.add_argument("--generaciones", type=int, default=1500, help="Número de generaciones")
    ap.add_argument("--padres", type=int, default=50, help="Padres seleccionados por generación")
    ap.add_argument("--elite", type=int, default=2, help="Mejores individuos que pasan sin cambios a la siguiente generación")
    ap.add_argument("--k-torneo", type=int, default=3, help="Tamaño del torneo de selección")
    ap.add_argument("--cruce", choices=sorted(CRUCES), default="ox", help="Operador de cruce")
    ap.add_argument("--mutacion", choices=sorted(MUTACIONES), default="inversion", help="Operador de mutación")
    ap.add_argument("--p-mutacion", type=float, default=0.3, help="Probabilidad de que un hijo mute (0 a 1)")
    ap.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria")
    ap.add_argument("--sin-gif", action="store_true", help="No generar el GIF (solo JSON y PNG)")
    ap.add_argument("--version", type=str, required=True, help="Sufijo de los archivos de salida")

    a = ap.parse_args()

    nombres, ubicaciones = leer_capitales()
    P = leer_peajes(len(nombres))
    C, partes = matriz_costos(a.valor_hora, a.consumo, a.precio, P)

    cfg = {
        "poblacion": a.poblacion, "generaciones": a.generaciones, "padres": a.padres, "elite": a.elite,
        "k_torneo": a.k_torneo, "cruce": a.cruce, "mutacion": a.mutacion, "p_mutacion": a.p_mutacion,
    }
    r = construir_y_correr(C, cfg, a.semilla)

    # Desglose del costo: suma de cada componente sobre los arcos del recorrido cerrado
    tour = np.array(r["mejor_tour"])
    siguiente = np.roll(tour, -1)
    horas = float(partes["horas"][tour, siguiente].sum())
    resultados = {
        "algoritmo": "GA",
        "version": a.version,
        "semilla": a.semilla,
        "parametros": {"valor_hora": a.valor_hora, "consumo": a.consumo, "precio": a.precio, **cfg},
        "mejor_costo_eur": round(r["mejor_costo"], 2),
        "desglose": {
            "km": round(float(partes["km"][tour, siguiente].sum()), 1),
            "horas": round(horas, 2),
            "costo_tiempo_eur": round(a.valor_hora * horas, 2),
            "peajes_eur": round(float(partes["peajes"][tour, siguiente].sum()), 2),
            "combustible_eur": round(float(partes["combustible"][tour, siguiente].sum()), 2),
        },
        "evaluaciones_fitness": r["evaluaciones_fitness"],
        "evaluaciones_nominales": r["evaluaciones_nominales"],
        "segundos": r["segundos"],
        "mejor_tour_ids": r["mejor_tour"],
        "mejor_ruta": [nombres[i] for i in r["mejor_tour"]] + [nombres[r["mejor_tour"][0]]],
    }

    print(f"Mejor costo encontrado: {r['mejor_costo']:.2f} euros (semilla {a.semilla}, {r['segundos']} s)")
    d = resultados["desglose"]
    print(f"  tiempo {d['costo_tiempo_eur']} + peajes {d['peajes_eur']} + combustible {d['combustible_eur']} euros"
          f" | {d['km']} km, {d['horas']} h")
    print(f"  evaluaciones de la aptitud: {r['evaluaciones_fitness']} reales de {r['evaluaciones_nominales']} nominales")
    print("  ruta: " + " -> ".join(resultados["mejor_ruta"]))

    SALIDAS.mkdir(exist_ok=True)
    with open(SALIDAS / f"resultados_GA_{a.version}.json", "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=4, ensure_ascii=False)

    crear_animacion(r["historial"], ubicaciones, a.version, exportar_gif=not a.sin_gif)


if __name__ == "__main__":
    main()

