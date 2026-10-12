"""Algoritmo genético para el TSP asimétrico de las 47 capitales, con PyGAD.

Representación: permutación de los ids 0..46 (recorrido cerrado).
Aptitud: -costo_tour (PyGAD maximiza).
"""
import time

import numpy as np
import pygad

from costos import costo_tour
from operadores_ga import CRUCES, hacer_mutacion


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

