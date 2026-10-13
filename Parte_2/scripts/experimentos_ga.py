"""Experimentos del algoritmo genético sobre el TSP de las 47 capitales.

    comparar_operadores   varias corridas independientes por cada pareja cruce + mutación
    barrido               varias corridas por cada valor de un parámetro del GA
    estudio_valor_hora    ¿cambia la mejor ruta al variar el valor de la hora? ¿a partir de qué valor?

Se lanzan desde Parte_2/orquestador.py. Todas reciben un diccionario `p` con los
parámetros y guardan sus archivos en outputs/ con el sufijo p["version"].
La corrida k de cada configuración usa la semilla p["semilla"] + k.
"""
import csv
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from algoritmo_genetico import CLAVES_CFG, SALIDAS, construir_y_correr, desglose, dibujar_mapa, validar
from costos import leer_capitales, leer_peajes, matriz_costos
from operadores_ga import CRUCES, MUTACIONES

# Parámetros numéricos del GA que admite el barrido, con su tipo y los valores que se prueban por defecto
BARRIDOS = {
    "p_mutacion": (float, [0.05, 0.1, 0.3, 0.5, 0.8]),
    "elite": (int, [0, 1, 2, 5, 10, 20]),
    "k_torneo": (int, [2, 3, 5, 8]),
    "poblacion": (int, [50, 100, 200, 400]),
    "padres": (int, [10, 25, 50, 100]),
    "generaciones": (int, [250, 500, 1000, 2000]),
}


# =====================================================================
# 1. CORRIDAS EN LOTE
# =====================================================================

def _una_corrida(tarea):
    """Corre el GA una vez. Es una función de módulo para poder repartirla entre procesos."""
    C, cfg, semilla = tarea
    r = construir_y_correr(C, cfg, semilla)
    historial = r.pop("historial")
    r["hist_costo"], r["hist_evals"] = historial["mejor_costo"], historial["evals"]
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


def datos_problema(p, valor_hora=None):
    """Nombres, ubicaciones, matriz de costos y sus componentes para un valor de la hora."""
    nombres, ubicaciones = leer_capitales()
    vh = p["valor_hora"] if valor_hora is None else valor_hora
    C, partes = matriz_costos(vh, p["consumo"], p["precio"], leer_peajes(len(nombres)))
    return nombres, ubicaciones, C, partes


# =====================================================================
# 2. VARIAS CORRIDAS POR CONFIGURACIÓN: TABLA, CSV Y FIGURAS
# =====================================================================

def experimento(titulo, variantes, p, nombre):
    """Corre p["repeticiones"] veces cada variante y resume los resultados.

    variantes: lista de (etiqueta, cfg). Éxito de una corrida: costo <= umbral, con
    umbral = referencia * (1 + tolerancia / 100). Si no se da p["referencia"], la
    referencia es el mejor costo de todas las corridas del experimento.
    Devuelve el diccionario que se guarda en <nombre>_GA_<version>_resumen.json.
    """
    R = p["repeticiones"]
    if R < 1:
        raise ValueError("repeticiones debe ser al menos 1.")
    if p["tolerancia"] < 0:
        raise ValueError("tolerancia debe ser mayor o igual que 0.")
    for _, cfg in variantes:
        validar(cfg)
    nombres, _, C, _ = datos_problema(p)
    semillas = [p["semilla"] + k for k in range(R)]

    print(f"  {len(variantes)} configuraciones x {R} corridas (semillas {semillas[0]}..{semillas[-1]}), "
          f"{p['procesos']} proceso(s)")
    todas = correr_lote([(C, cfg, s) for _, cfg in variantes for s in semillas], p["procesos"])
    res = {etiqueta: todas[i * R:(i + 1) * R] for i, (etiqueta, _) in enumerate(variantes)}

    mejor = min(todas, key=lambda r: r["mejor_costo"])
    referencia = p["referencia"] if p["referencia"] is not None else mejor["mejor_costo"]
    umbral = referencia * (1 + p["tolerancia"] / 100)

    print(f"\n{titulo} | valor de la hora = {p['valor_hora']:g} EUR/h | {R} corridas por configuración")
    print(f"Éxito: costo <= {umbral:.2f} EUR (referencia {referencia:.2f} EUR + {p['tolerancia']:g} %; "
          f"referencia = {'valor dado' if p['referencia'] is not None else 'mejor costo del experimento'})")
    print(f"{'Configuración':<24}{'Media':>10}{'Desv.':>9}{'Mejor':>10}{'Peor':>10}{'Éxito':>8}"
          f"{'Evals':>10}{'Nominales':>11}{'t (s)':>8}")
    print("-" * 100)
    filas = []
    for etiqueta, rs in res.items():
        v = np.array([r["mejor_costo"] for r in rs])
        fila = {
            "configuracion": etiqueta,
            "corridas": R,
            "media": round(float(v.mean()), 2),
            "desviacion": round(float(v.std(ddof=1)), 2) if R > 1 else 0.0,   # desviación muestral (n - 1)
            "mejor": round(float(v.min()), 2),
            "peor": round(float(v.max()), 2),
            "exito_pct": round(100 * float(np.mean(v <= umbral)), 1),
            "evaluaciones_media": round(float(np.mean([r["evaluaciones_fitness"] for r in rs])), 1),
            "evaluaciones_nominales": rs[0]["evaluaciones_nominales"],
            "segundos_media": round(float(np.mean([r["segundos"] for r in rs])), 2),
        }
        filas.append(fila)
        print(f"{etiqueta:<24}{fila['media']:>10.2f}{fila['desviacion']:>9.2f}{fila['mejor']:>10.2f}{fila['peor']:>10.2f}"
              f"{fila['exito_pct']:>7.0f}%{fila['evaluaciones_media']:>10.0f}{fila['evaluaciones_nominales']:>11d}"
              f"{fila['segundos_media']:>8.2f}")
    print(f"Mejor recorrido del experimento: {mejor['mejor_costo']:.2f} EUR")

    SALIDAS.mkdir(exist_ok=True)
    base = f"{nombre}_GA_{p['version']}"

    with open(SALIDAS / f"{base}_corridas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuracion", "semilla", "costo_eur", "exito", "evaluaciones_fitness",
                    "evaluaciones_nominales", "segundos", "tour"])
        for etiqueta, rs in res.items():
            for r in rs:
                w.writerow([etiqueta, r["semilla"], round(r["mejor_costo"], 2), int(r["mejor_costo"] <= umbral),
                            r["evaluaciones_fitness"], r["evaluaciones_nominales"], r["segundos"],
                            " ".join(map(str, r["mejor_tour"]))])

    resumen = {
        "experimento": titulo,
        "parametros": {k: p[k] for k in ("valor_hora", "consumo", "precio", *CLAVES_CFG)},
        "semillas": semillas,
        "referencia_eur": round(referencia, 2),
        "tolerancia_pct": p["tolerancia"],
        "umbral_exito_eur": round(umbral, 2),
        "tabla": filas,
        "mejor_costo_eur": round(mejor["mejor_costo"], 2),
        "mejor_tour_ids": mejor["mejor_tour"],
        "mejor_ruta": [nombres[i] for i in mejor["mejor_tour"]] + [nombres[mejor["mejor_tour"][0]]],
    }
    with open(SALIDAS / f"{base}_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    figura_cajas(res, umbral, f"{titulo} - valor de la hora {p['valor_hora']:g} EUR/h, {R} corridas", base)
    figura_convergencia(res, f"{titulo} - mediana de {R} corridas", base)
    print(f"  Archivos guardados en {SALIDAS}: {base}_corridas.csv, {base}_resumen.json, "
          f"{base}_cajas.png, {base}_convergencia.png")
    return resumen


def figura_cajas(res, umbral, titulo, base):
    """Diagrama de cajas del costo final de cada configuración."""
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(res) + 3), 5.5))
    ax.boxplot([[r["mejor_costo"] for r in rs] for rs in res.values()])
    ax.set_xticks(range(1, len(res) + 1), list(res), rotation=20, ha="right")
    ax.axhline(umbral, color="green", linestyle="--", linewidth=1, label=f"Umbral de éxito ({umbral:.0f} EUR)")
    ax.set_ylabel("Costo del mejor recorrido (EUR)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDAS / f"{base}_cajas.png", dpi=130)
    plt.close(fig)


def figura_convergencia(res, titulo, base):
    """Mediana del mejor costo contra el número de evaluaciones de la aptitud.

    Cada corrida es una función escalonada: su mejor costo con un presupuesto de e
    evaluaciones es el de la última generación que terminó con e evaluaciones o menos.
    """
    inicio = max(r["hist_evals"][0] for rs in res.values() for r in rs)
    fin = max(r["hist_evals"][-1] for rs in res.values() for r in rs)
    presupuesto = np.linspace(inicio, fin, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for etiqueta, rs in res.items():
        curvas = []
        for r in rs:
            g = np.searchsorted(r["hist_evals"], presupuesto, side="right") - 1
            curvas.append(np.asarray(r["hist_costo"])[g])
        ax.plot(presupuesto, np.median(curvas, axis=0), label=etiqueta)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel("Evaluaciones de la aptitud")
    ax.set_ylabel("Mejor costo (EUR, escala log)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDAS / f"{base}_convergencia.png", dpi=130)
    plt.close(fig)


def comparar_operadores(p):
    """Todas las parejas cruce + mutación, con el resto de parámetros iguales."""
    cfg = {k: p[k] for k in CLAVES_CFG}
    variantes = [(f"{c} + {m}", dict(cfg, cruce=c, mutacion=m)) for c in sorted(CRUCES) for m in sorted(MUTACIONES)]
    return experimento("Comparación de operadores", variantes, p, "comparar")


def barrido(p):
    """Un parámetro del GA toma varios valores; el resto queda fijo."""
    nombre = p["parametro"]
    if nombre not in BARRIDOS:
        raise ValueError(f"parametro debe ser uno de: {', '.join(BARRIDOS)}.")
    tipo, por_defecto = BARRIDOS[nombre]
    valores = por_defecto if not p["valores"] else [tipo(v) for v in p["valores"]]
    cfg = {k: p[k] for k in CLAVES_CFG}
    variantes = [(f"{nombre} = {v:g}", dict(cfg, **{nombre: v})) for v in valores]
    return experimento(f"Barrido de {nombre}", variantes, p, f"barrido_{nombre}")


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

    nombres, ubicaciones, _, partes = datos_problema(p, valor_hora=0.0)
    horas = partes["horas"]
    fijo = partes["peajes"] + partes["combustible"]       # lo que no depende del valor de la hora

    print(f"  {len(valores)} valores de la hora ({valores[0]:g} a {valores[-1]:g} EUR/h) x {len(semillas)} corridas "
          f"(semillas {semillas[0]}..{semillas[-1]}), {p['procesos']} proceso(s)")
    tareas = [(vh * horas + fijo, cfg, s) for vh in valores for s in semillas]
    corridas = correr_lote(tareas, p["procesos"])

    # Conjunto de recorridos distintos. Se rotan para empezar en la ciudad 0; no se
    # invierten, porque con costos asimétricos el sentido contrario es otro recorrido.
    conjunto = {}
    filas_csv = []
    for (vh, s), r in zip([(vh, s) for vh in valores for s in semillas], corridas):
        t = r["mejor_tour"]
        i = t.index(0)
        t = tuple(t[i:] + t[:i])
        sig = np.roll(t, -1)
        H_t, F_t = float(horas[t, sig].sum()), float(fijo[t, sig].sum())
        conjunto.setdefault(t, {"H": H_t, "F": F_t, "hallado_con": []})["hallado_con"].append(vh)
        filas_csv.append([vh, s, round(r["mejor_costo"], 2), round(H_t, 4), round(F_t, 2),
                          r["evaluaciones_fitness"], r["segundos"], " ".join(map(str, t))])

    tours = list(conjunto)
    H = np.array([conjunto[t]["H"] for t in tours])
    F = np.array([conjunto[t]["F"] for t in tours])
    tramos = envolvente(H, F, valores[0], valores[-1])

    def arcos(t):
        return {(a, b) for a, b in zip(t, t[1:] + t[:1])}

    def aristas(t):
        return {frozenset(a) for a in arcos(t)}

    print(f"\nEstudio del valor de la hora | {len(corridas)} corridas | {len(tours)} recorridos distintos encontrados")
    print("Mejor ruta conocida en cada tramo (costo = valor_hora x horas + peajes + combustible):")
    print(f"{'Ruta':<6}{'Desde':>8}{'Hasta':>8}{'Horas':>9}{'Peajes':>9}{'Combust.':>10}{'km':>9}{'Pérdida máx.':>14}"
          "   Hallada con valor de la hora")
    print("-" * 110)
    info_tramos = []
    for n_ruta, (desde, hasta, k) in enumerate(tramos, 1):
        t = tours[k]
        d = desglose(partes, t, 0.0)
        hallado = sorted(set(conjunto[t]["hallado_con"]))
        # Usar siempre esta ruta: lo que se paga de más frente a la mejor es una función
        # convexa del valor de la hora, así que su máximo está en un extremo del rango.
        perdida = max(float(v * H[k] + F[k] - np.min(v * H + F)) for v in (valores[0], valores[-1]))
        print(f"R{n_ruta:<5}{desde:>8.2f}{hasta:>8.2f}{d['horas']:>9.2f}{d['peajes_eur']:>9.2f}{d['combustible_eur']:>10.2f}"
              f"{d['km']:>9.1f}{perdida:>14.2f}   {', '.join(f'{v:g}' for v in hallado)}")
        info_tramos.append({
            "ruta": f"R{n_ruta}", "desde_eur_h": round(desde, 4), "hasta_eur_h": round(hasta, 4),
            "horas": d["horas"], "km": d["km"], "peajes_eur": d["peajes_eur"], "combustible_eur": d["combustible_eur"],
            "costo_fijo_eur": round(float(F[k]), 2), "perdida_maxima_si_se_usa_siempre_eur": round(perdida, 2),
            "hallado_con_valor_hora": hallado,
            "tour_ids": list(t), "ruta_nombres": [nombres[i] for i in t] + [nombres[t[0]]],
        })

    print("Pérdida máx. = lo más que se paga de más (EUR), en todo el rango, si se usa siempre esa ruta.")

    cambios = []
    if len(tramos) == 1:
        print(f"\nLa mejor ruta conocida NO cambia entre {valores[0]:g} y {valores[-1]:g} EUR/h.")
    else:
        print("\nCambios de ruta:")
    for n_ruta, ((_, corte, ka), (_, _, kb)) in enumerate(zip(tramos, tramos[1:]), 1):
        ta, tb = tours[ka], tours[kb]
        salen = sorted(arcos(ta) - arcos(tb))
        entran = sorted(arcos(tb) - arcos(ta))
        distintas = len(aristas(ta) - aristas(tb))
        cambio = {
            "valor_hora_eur_h": round(corte, 4), "de": f"R{n_ruta}", "a": f"R{n_ruta + 1}",
            "horas_ahorradas": round(float(H[ka] - H[kb]), 4), "sobrecosto_fijo_eur": round(float(F[kb] - F[ka]), 2),
            "tramos_de_carretera_distintos": distintas,
            "arcos_que_salen": [f"{nombres[a]} -> {nombres[b]}" for a, b in salen],
            "arcos_que_entran": [f"{nombres[a]} -> {nombres[b]}" for a, b in entran],
        }
        cambios.append(cambio)
        print(f"  A partir de {corte:.2f} EUR/h conviene R{n_ruta + 1} en lugar de R{n_ruta}: ahorra "
              f"{cambio['horas_ahorradas']:.2f} h y cuesta {cambio['sobrecosto_fijo_eur']:.2f} EUR más en peajes y combustible "
              f"({cambio['sobrecosto_fijo_eur']:.2f} / {cambio['horas_ahorradas']:.2f} = {corte:.2f} EUR/h).")
        print(f"    Tramos de carretera distintos (sin contar el sentido): {distintas} de {len(ta)}")
        if distintas <= 6:
            print(f"    Salen : {'; '.join(cambio['arcos_que_salen'])}")
            print(f"    Entran: {'; '.join(cambio['arcos_que_entran'])}")

    print("\nPor valor de la hora: mejor costo del GA con ese valor frente a la mejor ruta del conjunto")
    print(f"{'EUR/h':>7}{'GA con ese valor':>18}{'Mejor del conjunto':>20}{'Diferencia':>12}{'Ruta':>6}")
    por_valor = []
    for i, vh in enumerate(valores):
        propio = min(r["mejor_costo"] for r in corridas[i * len(semillas):(i + 1) * len(semillas)])
        conj = min(float(np.min(vh * H + F)), propio)     # iguala a `propio` salvo redondeo si la ruta es la misma
        ruta = next(f"R{n}" for n, (desde, hasta, _) in enumerate(tramos, 1) if desde <= vh <= hasta)
        print(f"{vh:>7g}{propio:>18.2f}{conj:>20.2f}{100 * (propio / conj - 1):>11.2f}%{ruta:>6}")
        por_valor.append({"valor_hora": vh, "mejor_ga_con_ese_valor_eur": round(propio, 2),
                          "mejor_del_conjunto_eur": round(conj, 2), "ruta": ruta})

    SALIDAS.mkdir(exist_ok=True)
    base = f"hora_GA_{p['version']}"
    with open(SALIDAS / f"{base}_corridas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["valor_hora", "semilla", "costo_eur", "horas", "costo_fijo_eur", "evaluaciones_fitness", "segundos", "tour"])
        w.writerows(filas_csv)
    resumen = {
        "experimento": "Estudio del valor de la hora",
        "parametros": {"consumo": p["consumo"], "precio": p["precio"], **cfg},
        "valores_hora": valores, "semillas": semillas, "recorridos_distintos": len(tours),
        "tramos": info_tramos, "cambios": cambios, "por_valor": por_valor,
    }
    with open(SALIDAS / f"{base}_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    figura_valor_hora(tramos, tours, H, F, valores, ubicaciones, base)
    print(f"  Archivos guardados en {SALIDAS}: {base}_corridas.csv, {base}_resumen.json, {base}.png")
    return resumen


def figura_valor_hora(tramos, tours, H, F, valores, ubicaciones, base):
    """Izquierda: sobrecosto de cada ruta de la envolvente frente a la mejor. Derecha: esas rutas en el mapa."""
    vh = np.linspace(valores[0], valores[-1], 500)
    rutas = [k for _, _, k in tramos]
    minimo = np.min([v * H[rutas] + F[rutas] for v in vh], axis=1)
    colores = plt.cm.tab10(np.arange(len(rutas)) % 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    for n_ruta, (k, color) in enumerate(zip(rutas, colores), 1):
        ax1.plot(vh, vh * H[k] + F[k] - minimo, color=color, label=f"R{n_ruta}")
    for _, corte, _ in tramos[:-1]:
        ax1.axvline(corte, color="gray", linestyle=":", linewidth=1)
    ax1.set_xlabel("Valor de la hora del vendedor (EUR/h)")
    ax1.set_ylabel("Sobrecosto frente a la mejor ruta conocida (EUR)")
    ax1.set_title("Mejor ruta según el valor de la hora (0 = es la mejor)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    dibujar_mapa(ax2, ubicaciones)
    for n_ruta, (k, color) in enumerate(zip(rutas, colores), 1):
        t = list(tours[k])
        coordenadas = ubicaciones[t + [t[0]]]
        ax2.plot(coordenadas[:, 1], coordenadas[:, 0], color=color, linewidth=3.2 - 2.2 * (n_ruta - 1) / max(1, len(rutas) - 1),
                 alpha=0.8, zorder=4, label=f"R{n_ruta}")
    ax2.set_title("Rutas de cada tramo (donde coinciden, las líneas se superponen)")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SALIDAS / f"{base}.png", dpi=130)
    plt.close(fig)
