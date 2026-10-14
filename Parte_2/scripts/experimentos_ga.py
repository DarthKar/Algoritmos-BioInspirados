"""Experimentos del algoritmo genético sobre el TSP de las 47 capitales.

    comparar_operadores   varias corridas independientes por cada pareja cruce + mutación
    estudio_valor_hora    ¿cambia la mejor ruta al variar el valor de la hora? ¿a partir de qué valor?

Se lanzan desde Parte_2/orquestador.py. Las dos reciben un diccionario `p` con los
parámetros y guardan una figura en outputs/ con el sufijo p["version"].
La corrida k de cada configuración usa la semilla p["semilla"] + k.
"""
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from algoritmo_genetico import CLAVES_CFG, SALIDAS, construir_y_correr, desglose, validar
from costos import leer_capitales, leer_peajes, matriz_costos
from operadores_ga import CRUCES, MUTACIONES


# =====================================================================
# 1. CORRIDAS EN LOTE
# =====================================================================

def _una_corrida(tarea):
    """Corre el GA una vez. Es una función de módulo para poder repartirla entre procesos."""
    C, cfg, semilla = tarea
    r = construir_y_correr(C, cfg, semilla)
    del r["historial"]            # aquí solo interesa el resultado final de cada corrida
    return r


def correr_lote(tareas, procesos=1):
    """Corre una lista de tareas (C, cfg, semilla) y devuelve los resultados en el mismo orden.

    Con procesos > 1 las corridas se reparten entre procesos. El resultado de cada
    corrida no cambia (depende solo de su semilla); cambia el tiempo de reloj.
    """
    ejecutor = ProcessPoolExecutor(max_workers=procesos) if procesos > 1 else None
    resultados = []
    try:
        corridas = ejecutor.map(_una_corrida, tareas) if ejecutor else map(_una_corrida, tareas)
        for k, r in enumerate(corridas, 1):
            resultados.append(r)
            if k % 10 == 0 or k == len(tareas):
                print(f"    {k}/{len(tareas)} corridas terminadas", flush=True)
    finally:
        if ejecutor:
            ejecutor.shutdown(cancel_futures=True)
    return resultados


# =====================================================================
# 2. COMPARACIÓN DE OPERADORES
# =====================================================================

def comparar_operadores(p):
    """Corre p["repeticiones"] veces cada pareja cruce + mutación, con el resto de parámetros iguales.

    Imprime la tabla de estadísticas y guarda un diagrama de cajas. Éxito de una
    corrida: costo <= mejor costo de todo el experimento * (1 + tolerancia / 100).
    """
    R = p["repeticiones"]
    if R < 1:
        raise ValueError("repeticiones debe ser al menos 1.")
    if p["tolerancia"] < 0:
        raise ValueError("tolerancia debe ser mayor o igual que 0.")
    cfg = {k: p[k] for k in CLAVES_CFG}
    variantes = [(f"{c} + {m}", dict(cfg, cruce=c, mutacion=m)) for c in sorted(CRUCES) for m in sorted(MUTACIONES)]
    for _, cfg_variante in variantes:
        validar(cfg_variante)
    nombres, _ = leer_capitales()
    C, _ = matriz_costos(p["valor_hora"], p["consumo"], p["precio"], leer_peajes(len(nombres)))
    semillas = [p["semilla"] + k for k in range(R)]

    print(f"  {len(variantes)} configuraciones x {R} corridas (semillas {semillas[0]}..{semillas[-1]}), "
          f"{p['procesos']} proceso(s)")
    todas = correr_lote([(C, cfg_variante, s) for _, cfg_variante in variantes for s in semillas], p["procesos"])
    res = {etiqueta: todas[i * R:(i + 1) * R] for i, (etiqueta, _) in enumerate(variantes)}

    mejor = min(r["mejor_costo"] for r in todas)
    umbral = mejor * (1 + p["tolerancia"] / 100)

    print(f"\nComparación de operadores | valor de la hora = {p['valor_hora']:g} EUR/h | {R} corridas por configuración")
    print(f"Éxito: costo <= {umbral:.2f} EUR (mejor costo del experimento, {mejor:.2f} EUR, + {p['tolerancia']:g} %)")
    print(f"{'Configuración':<20}{'Media':>10}{'Desv.':>9}{'Mejor':>10}{'Peor':>10}{'Éxito':>8}{'Evals':>10}{'t (s)':>8}")
    print("-" * 85)
    for etiqueta, rs in res.items():
        v = np.array([r["mejor_costo"] for r in rs])
        desv = v.std(ddof=1) if R > 1 else 0.0           # desviación muestral (n - 1)
        print(f"{etiqueta:<20}{v.mean():>10.2f}{desv:>9.2f}{v.min():>10.2f}{v.max():>10.2f}"
              f"{100 * np.mean(v <= umbral):>7.0f}%{np.mean([r['evaluaciones_fitness'] for r in rs]):>10.0f}"
              f"{np.mean([r['segundos'] for r in rs]):>8.2f}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([[r["mejor_costo"] for r in rs] for rs in res.values()])
    ax.set_xticks(range(1, len(res) + 1), list(res), rotation=20, ha="right")
    ax.axhline(umbral, color="green", linestyle="--", linewidth=1, label=f"Umbral de éxito ({umbral:.0f} EUR)")
    ax.set_ylabel("Costo del mejor recorrido (EUR)")
    ax.set_title(f"Comparación de operadores - valor de la hora {p['valor_hora']:g} EUR/h, {R} corridas")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / f"comparar_GA_{p['version']}.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  Figura guardada en: {ruta}")


# =====================================================================
# 3. ESTUDIO DEL VALOR DE LA HORA
# =====================================================================

def envolvente(H, F, vh_min, vh_max):
    """Tramos (desde, hasta, k) en los que el recorrido k es el más barato.

    El costo del recorrido k es una recta en el valor de la hora: vh * H[k] + F[k]
    (H = horas totales, F = peajes + combustible). Se parte del más barato en
    vh_min. Al subir vh solo puede superarlo un recorrido con menos horas, y lo
    iguala en vh = (F[j] - F[k]) / (H[k] - H[j]). El primer cruce da el siguiente tramo.
    """
    H, F = np.asarray(H, dtype=float), np.asarray(F, dtype=float)
    k = int(np.lexsort((H, vh_min * H + F))[0])          # el más barato en vh_min; si empatan, el de menos horas
    tramos, desde = [], float(vh_min)
    while True:
        candidatos = np.where(H < H[k])[0]
        if len(candidatos) == 0:
            break
        cruce = (F[candidatos] - F[k]) / (H[k] - H[candidatos])
        j = int(np.lexsort((H[candidatos], cruce))[0])   # el primer cruce; si empatan, el de menos horas
        if cruce[j] >= vh_max:
            break
        if cruce[j] - desde > 1e-9:
            tramos.append((desde, float(cruce[j]), k))
            desde = float(cruce[j])
        k = int(candidatos[j])
    tramos.append((desde, float(vh_max), k))
    return tramos


def estudio_valor_hora(p):
    """¿Cambia la mejor ruta al variar el valor de la hora? ¿A partir de qué valor?

    1. Corre el GA p["rep_hora"] veces para cada valor de la hora de la rejilla.
    2. Junta todos los recorridos encontrados. El costo de cada uno es una recta en
       el valor de la hora, así que cada recorrido se evalúa de forma exacta para
       cualquier valor, no solo para aquel con el que se encontró.
    3. La envolvente inferior de esas rectas dice qué recorrido es el mejor en cada
       tramo y el valor exacto en que cambia.
    Las rutas son las mejores ENCONTRADAS por el GA: no hay garantía de que sean óptimas.
    """
    if p["vh_paso"] <= 0 or p["vh_max"] < p["vh_min"] or p["vh_min"] < 0:
        raise ValueError("se necesita 0 <= vh_min <= vh_max y vh_paso > 0.")
    if p["rep_hora"] < 1:
        raise ValueError("rep_hora debe ser al menos 1.")
    cfg = {k: p[k] for k in CLAVES_CFG}
    validar(cfg)
    valores = [round(float(v), 6) for v in np.arange(p["vh_min"], p["vh_max"] + p["vh_paso"] / 2, p["vh_paso"])]
    semillas = [p["semilla"] + k for k in range(p["rep_hora"])]

    nombres, _ = leer_capitales()
    _, partes = matriz_costos(0.0, p["consumo"], p["precio"], leer_peajes(len(nombres)))
    horas = partes["horas"]
    fijo = partes["peajes"] + partes["combustible"]       # lo que no depende del valor de la hora

    print(f"  {len(valores)} valores de la hora ({valores[0]:g} a {valores[-1]:g} EUR/h) x {len(semillas)} corridas "
          f"(semillas {semillas[0]}..{semillas[-1]}), {p['procesos']} proceso(s)")
    corridas = correr_lote([(vh * horas + fijo, cfg, s) for vh in valores for s in semillas], p["procesos"])

    # Recorridos distintos. Se rotan para empezar en la ciudad 0; no se invierten,
    # porque con costos asimétricos el sentido contrario es otro recorrido.
    tours = []
    for r in corridas:
        t = r["mejor_tour"]
        i = t.index(0)
        t = tuple(t[i:] + t[:i])
        if t not in tours:
            tours.append(t)
    H = np.array([horas[t, np.roll(t, -1)].sum() for t in tours])
    F = np.array([fijo[t, np.roll(t, -1)].sum() for t in tours])
    tramos = envolvente(H, F, valores[0], valores[-1])

    print(f"\nEstudio del valor de la hora | {len(corridas)} corridas | {len(tours)} recorridos distintos encontrados")
    print("Mejor ruta conocida en cada tramo (costo = valor_hora x horas + peajes + combustible):")
    print(f"{'Ruta':<6}{'Desde':>8}{'Hasta':>8}{'Horas':>9}{'Peajes':>9}{'Combust.':>10}{'km':>9}")
    print("-" * 59)
    for n_ruta, (desde, hasta, k) in enumerate(tramos, 1):
        d = desglose(partes, tours[k], 0.0)
        print(f"R{n_ruta:<5}{desde:>8.2f}{hasta:>8.2f}{d['horas']:>9.2f}{d['peajes_eur']:>9.2f}{d['combustible_eur']:>10.2f}"
              f"{d['km']:>9.1f}")

    if len(tramos) == 1:
        print(f"\nLa mejor ruta conocida NO cambia entre {valores[0]:g} y {valores[-1]:g} EUR/h.")
    for n_ruta, ((_, corte, ka), (_, _, kb)) in enumerate(zip(tramos, tramos[1:]), 1):
        ahorro, sobrecosto = H[ka] - H[kb], F[kb] - F[ka]
        print(f"A partir de {corte:.2f} EUR/h conviene R{n_ruta + 1} en lugar de R{n_ruta}: ahorra {ahorro:.2f} h y cuesta "
              f"{sobrecosto:.2f} EUR más en peajes y combustible ({sobrecosto:.2f} / {ahorro:.2f} = {corte:.2f} EUR/h).")
    for n_ruta, (_, _, k) in enumerate(tramos, 1):
        t = tours[k]
        print(f"R{n_ruta}: " + " -> ".join(nombres[i] for i in t + t[:1]))

    # Sobrecosto de cada ruta de la envolvente frente a la mejor, en todo el rango
    vh = np.linspace(valores[0], valores[-1], 500)
    rutas = [k for _, _, k in tramos]
    minimo = np.min([v * H[rutas] + F[rutas] for v in vh], axis=1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n_ruta, k in enumerate(rutas, 1):
        ax.plot(vh, vh * H[k] + F[k] - minimo, label=f"R{n_ruta}")
    for _, corte, _ in tramos[:-1]:
        ax.axvline(corte, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Valor de la hora del vendedor (EUR/h)")
    ax.set_ylabel("Sobrecosto frente a la mejor ruta conocida (EUR)")
    ax.set_title("Mejor ruta según el valor de la hora (0 = es la mejor)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / f"hora_GA_{p['version']}.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  Figura guardada en: {ruta}")
