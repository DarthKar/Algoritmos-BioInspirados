"""Algoritmo genético para el TSP asimétrico de las 47 capitales, con PyGAD.

Representación: permutación de los ids 0..46 (recorrido cerrado).
Aptitud: -costo_tour (PyGAD maximiza).

Guarda en outputs/ (raíz del repositorio), con el sufijo dado en --version:
    resultados_GA_<version>.json   parámetros, semilla, mejor recorrido, costo y su desglose
    Recorrido_GA_<version>.png     mejor recorrido sobre el mapa
    Evolucion_GA_<version>.gif     evolución del mejor recorrido

Ejemplo:
    python Parte_2/scripts/algoritmo_genetico.py --valor-hora 15 --version 0001
    python Parte_2/scripts/algoritmo_genetico.py --valor-hora 25 --poblacion 100 --generaciones 1500 --cruce ox --mutacion inversion --p-mutacion 0.3 --semilla 7 --version 0002

El menú interactivo y los experimentos (30 corridas, barridos, valor de la hora)
se lanzan desde Parte_2/orquestador.py.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")            # las figuras solo se guardan en archivos; no se abren ventanas
import matplotlib.pyplot as plt
import numpy as np
import pygad
from matplotlib.animation import FuncAnimation, PillowWriter

from costos import DATA, RAIZ, costo_tour, leer_capitales, leer_peajes, matriz_costos
from operadores_ga import CRUCES, MUTACIONES, hacer_mutacion

SALIDAS = RAIZ.parent / "outputs"

# Claves del diccionario cfg que recibe construir_y_correr
CLAVES_CFG = ("poblacion", "generaciones", "padres", "elite", "k_torneo", "cruce", "mutacion", "p_mutacion")


def validar(cfg):
    """Lanza ValueError con un mensaje claro si los parámetros del GA no son coherentes."""
    if cfg["poblacion"] < 2:
        raise ValueError("poblacion debe ser al menos 2.")
    if cfg["generaciones"] < 1:
        raise ValueError("generaciones debe ser al menos 1.")
    if not 2 <= cfg["padres"] <= cfg["poblacion"]:
        raise ValueError(f"padres debe estar entre 2 y poblacion ({cfg['poblacion']}).")
    if not 0 <= cfg["elite"] < cfg["poblacion"]:
        raise ValueError(f"elite debe estar entre 0 y poblacion - 1 ({cfg['poblacion'] - 1}).")
    if cfg["k_torneo"] < 1:
        raise ValueError("k_torneo debe ser al menos 1.")
    if not 0 <= cfg["p_mutacion"] <= 1:
        raise ValueError("p_mutacion es una probabilidad: debe estar entre 0 y 1.")
    if cfg["cruce"] not in CRUCES:
        raise ValueError(f"cruce debe ser uno de: {', '.join(sorted(CRUCES))}.")
    if cfg["mutacion"] not in MUTACIONES:
        raise ValueError(f"mutacion debe ser una de: {', '.join(sorted(MUTACIONES))}.")


def construir_y_correr(C, cfg, semilla):
    """Corre el GA. Devuelve un dict con el mejor tour, su costo, evaluaciones e historial.

    cfg: dict con claves poblacion, generaciones, padres, elite, k_torneo,
         cruce ("ox" | "pmx"), mutacion ("inversion" | "insercion" | "swap" | "mixta"),
         p_mutacion (probabilidad de que un hijo mute, entre 0 y 1).

    El historial guarda, por generación, el mejor recorrido encontrado HASTA esa
    generación. Con elite >= 1 coincide con el mejor de la población; con elite = 0
    el mejor puede perderse de una generación a otra, pero aquí se conserva.
    """
    validar(cfg)
    n = C.shape[0]
    contador = {"evals": 0}
    mejor = {"costo": np.inf, "tour": None}
    historial = {"mejor_costo": [], "mejor_tour": [], "evals": []}

    def fitness(ga_instance, solution, solution_idx):
        # Cuenta llamadas reales: una por cada recorrido que se evalúa. PyGAD solo se
        # la ahorra con los élites (y los hijos idénticos a un élite), cuya aptitud ya conoce.
        contador["evals"] += 1
        return -costo_tour(C, solution.astype(int))

    def on_generation(ga_instance):
        sol, fit, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        if -fit < mejor["costo"]:
            mejor["costo"], mejor["tour"] = float(-fit), sol.astype(int).tolist()
        historial["mejor_costo"].append(mejor["costo"])
        historial["mejor_tour"].append(mejor["tour"])
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
        # Los padres no pasan directamente a la siguiente generación: solo los élites.
        # Sin esto, con elite = 0 PyGAD conserva a todos los padres (keep_parents = -1).
        keep_parents=0,
        on_generation=on_generation,
        random_seed=semilla,
        suppress_warnings=True,
    )
    t0 = time.perf_counter()
    ga.run()
    seg = time.perf_counter() - t0

    tour = np.array(mejor["tour"])
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


def desglose(partes, tour, valor_hora):
    """Suma de cada componente del costo sobre los arcos del recorrido cerrado."""
    tour = np.asarray(tour)
    siguiente = np.roll(tour, -1)
    horas = float(partes["horas"][tour, siguiente].sum())
    return {
        "km": round(float(partes["km"][tour, siguiente].sum()), 1),
        "horas": round(horas, 2),
        "costo_tiempo_eur": round(valor_hora * horas, 2),
        "peajes_eur": round(float(partes["peajes"][tour, siguiente].sum()), 2),
        "combustible_eur": round(float(partes["combustible"][tour, siguiente].sum()), 2),
    }


def leer_contorno_espana():
    """Polígonos (lon, lat) del contorno de España, del GeoJSON guardado en data/."""
    with open(DATA / "espana.geojson", encoding="utf-8") as f:
        geometria = json.load(f)["features"][0]["geometry"]
    return [np.array(poligono[0]) for poligono in geometria["coordinates"]]


def dibujar_mapa(ax, ubicaciones):
    """Fondo cartográfico de la España peninsular con las capitales."""
    for poligono in leer_contorno_espana():
        ax.fill(poligono[:, 0], poligono[:, 1], facecolor="#e8efe3", edgecolor="#52665b", linewidth=1.3)
    ax.set_xlim(-10, 4.5)
    ax.set_ylim(35.5, 44.5)
    ax.set_aspect("equal")
    ax.scatter(ubicaciones[:, 1], ubicaciones[:, 0], color="#173d68", s=22, zorder=5, label="Capitales")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(alpha=0.15)


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
    dibujar_mapa(ax, ubicaciones)
    linea, = ax.plot([], [], color="red", linewidth=1.6, alpha=0.85, zorder=4, label="Mejor recorrido")
    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=11,
        bbox=dict(facecolor="white", alpha=0.90, edgecolor="gray", boxstyle="round,pad=0.6"),
    )
    ax.set_title("Optimización de rutas - Algoritmo Genético", fontsize=15)
    ax.legend(loc="lower right")

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
        ruta_gif = SALIDAS / f"Evolucion_GA_{version}.gif"
        animacion.save(ruta_gif, writer=PillowWriter(fps=fps), dpi=90)
        print(f"  GIF guardado en: {ruta_gif}")

    actualizar(len(indices) - 1)
    ruta_png = SALIDAS / f"Recorrido_GA_{version}.png"
    fig.savefig(ruta_png, dpi=120)
    print(f"  PNG guardado en: {ruta_png}")
    plt.close(fig)


def corrida(p):
    """Una corrida completa: matriz de costos, GA, resumen en consola y archivos en outputs/.

    p: dict con valor_hora, consumo, precio, semilla, version, sin_gif y las claves de CLAVES_CFG.
    Devuelve el diccionario que se guarda en resultados_GA_<version>.json.
    """
    cfg = {k: p[k] for k in CLAVES_CFG}
    nombres, ubicaciones = leer_capitales()
    P = leer_peajes(len(nombres))
    C, partes = matriz_costos(p["valor_hora"], p["consumo"], p["precio"], P)

    r = construir_y_correr(C, cfg, p["semilla"])

    resultados = {
        "algoritmo": "GA",
        "version": p["version"],
        "semilla": p["semilla"],
        "parametros": {"valor_hora": p["valor_hora"], "consumo": p["consumo"], "precio": p["precio"], **cfg},
        "mejor_costo_eur": round(r["mejor_costo"], 2),
        "desglose": desglose(partes, r["mejor_tour"], p["valor_hora"]),
        "evaluaciones_fitness": r["evaluaciones_fitness"],
        "evaluaciones_nominales": r["evaluaciones_nominales"],
        "segundos": r["segundos"],
        "mejor_tour_ids": r["mejor_tour"],
        "mejor_ruta": [nombres[i] for i in r["mejor_tour"]] + [nombres[r["mejor_tour"][0]]],
    }

    print(f"Mejor costo encontrado: {r['mejor_costo']:.2f} euros (semilla {p['semilla']}, {r['segundos']} s)")
    d = resultados["desglose"]
    print(f"  tiempo {d['costo_tiempo_eur']} + peajes {d['peajes_eur']} + combustible {d['combustible_eur']} euros"
          f" | {d['km']} km, {d['horas']} h")
    print(f"  evaluaciones de la aptitud: {r['evaluaciones_fitness']} reales de {r['evaluaciones_nominales']} nominales")
    print("  ruta: " + " -> ".join(resultados["mejor_ruta"]))

    SALIDAS.mkdir(exist_ok=True)
    ruta_json = SALIDAS / f"resultados_GA_{p['version']}.json"
    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=4, ensure_ascii=False)
    print(f"  JSON guardado en: {ruta_json}")

    crear_animacion(r["historial"], ubicaciones, p["version"], exportar_gif=not p["sin_gif"])
    return resultados


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
    ap.add_argument("--mutacion", choices=sorted(MUTACIONES), default="mixta", help="Operador de mutación")
    ap.add_argument("--p-mutacion", type=float, default=0.3, help="Probabilidad de que un hijo mute (0 a 1)")
    ap.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria")
    ap.add_argument("--sin-gif", action="store_true", help="No generar el GIF (solo JSON y PNG)")
    ap.add_argument("--version", type=str, required=True, help="Sufijo de los archivos de salida")

    try:
        corrida(vars(ap.parse_args()))
    except ValueError as e:
        ap.exit(2, f"Error de parámetros: {e}\n")


if __name__ == "__main__":
    main()
