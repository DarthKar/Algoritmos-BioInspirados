import argparse
import copy
import importlib
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Módulos con las implementaciones (se importan por separado porque
# definen funciones con el mismo nombre pero distinta firma/comportamiento):
#
#   rastrigin.py  -> módulo principal: rastrigin, GD, evolutivo, PSO, DE
#   schwefel.py   -> módulo secundario: versión alterna del evolutivo y
#                    mutación gaussiana (y, si la defines allí, schwefel)
# ---------------------------------------------------------------------
MODULO_POR_DEFECTO = "rastrigin"
MODULO2_POR_DEFECTO = "schwefel"

# Módulo secundario cargado (None si no está disponible)
EXTRA = {"mod2": None}


# =====================================================================
# FUNCIÓN DE SCHWEFEL (por defecto; si schwefel.py define `schwefel` y
# `schwefel_grad`, se usan las de ese archivo)
# =====================================================================

def schwefel(X):
    """Función de Schwefel.

    f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))

    Dominio habitual [-500, 500]. Mínimo global en x_i = 420.9687 con
    f = 0. Es engañosa: el mejor mínimo local queda lejos del segundo mejor.
    Misma convención que rastrigin: escalar si se evalúa un solo individuo.
    """
    X = np.atleast_2d(X)
    f = 418.9828872724339 * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
    return f if f.size > 1 else f[0]


def schwefel_grad(x):
    """Gradiente analítico de Schwefel.

    df/dx_i = -sin(s_i) - (s_i / 2) * cos(s_i),  con s_i = sqrt(|x_i|)
    """
    s = np.sqrt(np.abs(np.asarray(x, dtype=float)))
    return -np.sin(s) - 0.5 * s * np.cos(s)


# Valores por defecto de cada función
CONFIG_FUNCIONES = {
    "rastrigin": {"lb": -5.12, "ub": 5.12, "lr": 0.002, "optimo": 0.0},
    "schwefel": {"lb": -500.0, "ub": 500.0, "lr": 1.0, "optimo": 420.9687},
}


# =====================================================================
# UTILIDADES
# =====================================================================

def cargar_modulo(nombre, obligatorio=True):
    """Importa un módulo. Si falla: sale (obligatorio) o devuelve None."""
    try:
        return importlib.import_module(nombre)
    except ModuleNotFoundError as e:
        if e.name == nombre:
            msg = (f"No se encontró el módulo '{nombre}'. Guarda el código como "
                   f"'{nombre}.py' junto a main.py.")
        else:
            msg = (f"El módulo '{nombre}' necesita la librería '{e.name}' "
                   f"(instálala con: pip install {e.name}).")
    except Exception as e:
        msg = f"Error al importar '{nombre}': {type(e).__name__}: {e}"
    if obligatorio:
        print(f"[Error] {msg}")
        sys.exit(1)
    print(f"[Aviso] {msg}")
    return None


def exigir_mod2():
    """Devuelve el módulo secundario o lanza un error claro si no está."""
    if EXTRA["mod2"] is None:
        raise ValueError("Este comando necesita el módulo secundario (schwefel.py), "
                         "que no se pudo importar. Usa --modulo2 o revisa el aviso inicial.")
    return EXTRA["mod2"]


def obtener(mod, a):
    """Devuelve (función objetivo, gradiente analítico) según a.funcion."""
    if a.funcion == "schwefel":
        m2 = EXTRA["mod2"]
        f = getattr(m2, "schwefel", None) if m2 is not None else None
        g = getattr(m2, "schwefel_grad", None) if m2 is not None else None
        return (f or schwefel), (g or schwefel_grad)
    return mod.rastrigin, mod.rastrigin_grad


def punto_optimo(a):
    return np.full(a.d, CONFIG_FUNCIONES[a.funcion]["optimo"])


def resolver(a):
    """Rellena con los valores por defecto de la función lo que sea None."""
    cfg = CONFIG_FUNCIONES[a.funcion]
    if a.lb is None:
        a.lb = cfg["lb"]
    if a.ub is None:
        a.ub = cfg["ub"]
    if hasattr(a, "lr") and a.lr is None:
        a.lr = cfg["lr"]


def aplicar_config(mod, a):
    """Resuelve valores por defecto, propaga al módulo y fija la semilla."""
    resolver(a)
    mod.A = a.A
    mod.LB, mod.UB = a.lb, a.ub
    if a.seed is not None:
        np.random.seed(a.seed)
    print(f"\n>>> Función: {a.funcion}  |  d = {a.d}  |  dominio = [{a.lb}, {a.ub}]")


def semilla_rep(a, rep):
    return None if a.seed is None else a.seed + rep


def punto_inicial(a, nombre="x0"):
    """Devuelve el punto dado por el usuario o uno aleatorio en el dominio."""
    valores = getattr(a, nombre, None)
    if valores:
        x = np.array(valores, dtype=float)
        if len(x) != a.d:
            print(f"[Aviso] {nombre} tiene {len(x)} valores; se usa d = {len(x)}.")
            a.d = len(x)
        return x
    return np.random.uniform(a.lb, a.ub, a.d)


def fmt(x):
    return np.array2string(np.asarray(x), precision=4, suppress_small=True,
                           max_line_width=100)


def resumen(r, a):
    print(f"\n=== {r['nombre']} ({a.funcion}) ===")
    print(f"  Mejor fitness : {r['best_fit']:.6g}")
    print(f"  Mejor solución: {fmt(r['best_sol'])}")
    dist = np.linalg.norm(np.asarray(r['best_sol']) - punto_optimo(a))
    print(f"  Distancia al óptimo global (norma): {dist:.4g}")
    for k, v in r.get("info", {}).items():
        print(f"  {k}: {v}")
    print(f"  Tiempo        : {r['tiempo']:.3f} s")


def graficar(resultados, titulo, a, mod):
    """Curva de convergencia (+ mapa de contorno si d == 2)."""
    f_obj, _ = obtener(mod, a)
    d2 = a.d == 2
    fig, axes = plt.subplots(1, 2 if d2 else 1, figsize=(13 if d2 else 7, 5.5))
    ax = axes[0] if d2 else axes

    for r in resultados:
        h = np.maximum(np.asarray(r["historial"], dtype=float), 1e-16)
        ax.plot(h, label=r["nombre"], lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Iteración / generación")
    ax.set_ylabel("Mejor fitness (escala log)")
    ax.set_title("Convergencia")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    if d2:
        ax2 = axes[1]
        g = np.linspace(a.lb, a.ub, 300)
        XX, YY = np.meshgrid(g, g)
        ZZ = np.asarray(f_obj(np.column_stack([XX.ravel(), YY.ravel()])))
        cs = ax2.contourf(XX, YY, ZZ.reshape(XX.shape), levels=40, cmap="viridis")
        fig.colorbar(cs, ax=ax2, label="f(x)")
        for r in resultados:
            tr = r.get("trayectoria")
            if tr is not None:
                tr = np.array(tr)
                ax2.plot(tr[:, 0], tr[:, 1], ".-", ms=3, lw=0.8, alpha=0.8)
            ax2.plot(*r["best_sol"], "o", mec="k", label=r["nombre"])
        opt = punto_optimo(a)
        ax2.plot(*opt, "r*", ms=14, label="Óptimo global")
        ax2.set_title("Trayectorias y soluciones")
        ax2.set_xlim(a.lb, a.ub)
        ax2.set_ylim(a.lb, a.ub)
        ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{titulo} - {a.funcion}")
    fig.tight_layout()
    if a.guardar:
        ruta = a.guardar
        if getattr(a, "_sufijo", None):
            base, ext = os.path.splitext(ruta)
            ruta = f"{base}_{a._sufijo}{ext}"
        fig.savefig(ruta, dpi=150)
        print(f"  Gráfico guardado en: {ruta}")
    if not a.no_plot:
        plt.show()
    plt.close(fig)


# =====================================================================
# EJECUTORES DE CADA MÉTODO (devuelven un diccionario de resultados)
# =====================================================================

def ejecutar_gd(mod, a, numerico=False):
    f_obj, grad = obtener(mod, a)
    limites = (a.lb, a.ub) if a.clip else None
    mejor = None
    t0 = time.perf_counter()
    for k in range(a.inicios):
        x0 = punto_inicial(a) if k == 0 else np.random.uniform(a.lb, a.ub, a.d)
        if numerico:
            pts = mod.grad_desc_num(x0, f_obj, a.iter, a.lr, a.tol, a.h, limites)
        else:
            pts = mod.grad_desc(x0, grad, a.iter, a.lr, a.tol, limites)
        hist = mod.historial_fitness(pts, f_obj)
        if mejor is None or hist[-1] < mejor["hist"][-1]:
            mejor = {"pts": pts, "hist": hist, "x0": x0}
    tiempo = time.perf_counter() - t0

    pts, hist = mejor["pts"], mejor["hist"]
    n_iter = len(pts) - 1
    info = {
        "Punto inicial (mejor corrida)": fmt(mejor["x0"]),
        "Learning rate": a.lr,
        "Iteraciones realizadas": f"{n_iter} de {a.iter}"
                                   + (" (paró por tolerancia)" if n_iter < a.iter else ""),
        "Inicios aleatorios": a.inicios,
    }
    return {"nombre": "GD numérico" if numerico else "GD analítico",
            "best_sol": pts[-1], "best_fit": float(hist[-1]),
            "historial": hist, "trayectoria": pts, "tiempo": tiempo, "info": info}


def ejecutar_evo(mod, a, semilla=None):
    f_obj, _ = obtener(mod, a)
    if semilla is not None:
        np.random.seed(semilla)
    t0 = time.perf_counter()
    sol, fit, hist = mod.mi_algoritmo_evolutivo(
        f_obj, a.d, a.lb, a.ub, a.N, a.iter, a.elitismo, a.mutacion)
    n_el, n_cr, n_mu = mod.calcular_conteos(a.N, a.elitismo, a.mutacion)
    return {"nombre": "Evolutivo", "best_sol": sol, "best_fit": float(fit),
            "historial": hist, "tiempo": time.perf_counter() - t0,
            "info": {"Por generación (élite/cruce/mutación)": f"{n_el}/{n_cr}/{n_mu}",
                     "Evaluaciones": a.N * (a.iter + 1)}}


def ejecutar_evo2(mod, a, semilla=None):
    """Evolutivo de schwefel.py.

    Esa versión devuelve (lista de poblaciones, fitness ordenado por iteración)
    y no entrega el mejor individuo ni evalúa la última población, así que aquí
    se adapta su salida al mismo formato que el resto de métodos.
    """
    m2 = exigir_mod2()
    f_obj, _ = obtener(mod, a)
    LB = np.full(a.d, a.lb)
    UB = np.full(a.d, a.ub)
    t0 = time.perf_counter()
    pobs, f_best = m2.mi_algoritmo_evolutivo(
        f_obj, N=a.N, d=a.d, LB=LB, UB=UB, fr_elitismo=a.elitismo,
        fr_mutacion=a.mutacion, max_iter=a.iter, semilla=semilla)

    f_best = np.asarray(f_best, dtype=float)
    mejores = f_best[:, 0] if f_best.ndim == 2 else f_best.ravel()
    # la última población se crea pero el algoritmo no la evalúa
    _, fit_ult = m2.pob_eval(pobs[-1], f_obj)
    curva = np.append(mejores, fit_ult[0])
    idx = int(np.argmin(curva))
    pob_idx, fit_idx = m2.pob_eval(pobs[idx], f_obj)
    historial = np.minimum.accumulate(curva)

    n_el, n_cr, n_mu = mod.calcular_conteos(a.N, a.elitismo, a.mutacion)
    return {"nombre": "Evolutivo v2", "best_sol": pob_idx[0], "best_fit": float(fit_idx[0]),
            "historial": historial, "tiempo": time.perf_counter() - t0,
            "info": {"Por generación (élite/cruce/mutación)": f"{n_el}/{n_cr}/{n_mu}",
                     "Mejor encontrado en la generación": idx,
                     "Evaluaciones": a.N * (a.iter + 1)}}


def ejecutar_pso(mod, a, semilla=None):
    f_obj, _ = obtener(mod, a)
    if semilla is not None:
        np.random.seed(semilla)
    t0 = time.perf_counter()
    sol, fit, hist = mod.mi_pso(f_obj, a.d, a.lb, a.ub, a.N, a.iter,
                                a.w, a.c1, a.c2, a.frac_vmax)
    return {"nombre": "PSO", "best_sol": sol, "best_fit": float(fit),
            "historial": hist, "tiempo": time.perf_counter() - t0,
            "info": {"Evaluaciones": a.N * (a.iter + 1)}}


def ejecutar_de(mod, a, semilla=None):
    f_obj, _ = obtener(mod, a)
    t0 = time.perf_counter()
    sol, fit, hist = mod.evolucion_diferencial(
        f_obj, a.d, a.lb, a.ub, a.N, a.iter, a.F, a.CR, semilla)
    return {"nombre": "DE", "best_sol": sol, "best_fit": float(fit),
            "historial": hist, "tiempo": time.perf_counter() - t0,
            "info": {"Evaluaciones": a.N * (a.iter + 1)}}


# =====================================================================
# COMANDOS
# =====================================================================

def cmd_eval(mod, a):
    aplicar_config(mod, a)
    f_obj, _ = obtener(mod, a)
    x = punto_inicial(a, "x")
    print(f"x = {fmt(x)}")
    print(f"f(x) = {f_obj(x):.6g}")
    print(f"f(óptimo global) = {f_obj(punto_optimo(a)):.4g}  en x = {fmt(punto_optimo(a))}")


def cmd_grad(mod, a):
    aplicar_config(mod, a)
    f_obj, grad = obtener(mod, a)
    x = punto_inicial(a, "x")
    ga = grad(x)
    gn = mod.gradiente_numerico(x, f_obj, a.h)
    print(f"x                 = {fmt(x)}")
    print(f"Gradiente analít. = {fmt(ga)}")
    print(f"Gradiente numér.  = {fmt(gn)}   (h = {a.h})")
    print(f"Error (norma)     = {np.linalg.norm(ga - gn):.3e}")
    print("\nEfecto del tamaño de paso h en el error:")
    print(f"  {'h':>10}  {'error':>12}")
    for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        err = np.linalg.norm(ga - mod.gradiente_numerico(x, f_obj, h))
        print(f"  {h:>10.0e}  {err:>12.3e}")


def _simple(mod, a, resultado, titulo):
    resumen(resultado, a)
    graficar([resultado], titulo, a, mod)


def cmd_gd(mod, a):
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_gd(mod, a), "Descenso por gradiente (analítico)")


def cmd_gdnum(mod, a):
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_gd(mod, a, numerico=True),
            "Descenso por gradiente (numérico)")


def cmd_evo(mod, a):
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_evo(mod, a), "Algoritmo evolutivo")


def cmd_evo2(mod, a):
    exigir_mod2()
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_evo2(mod, a), "Algoritmo evolutivo (versión schwefel.py)")


def cmd_pso(mod, a):
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_pso(mod, a), "PSO")


def cmd_de(mod, a):
    aplicar_config(mod, a)
    _simple(mod, a, ejecutar_de(mod, a, a.seed), "Evolución diferencial")


def cmd_mutgauss(mod, a):
    """Muestra la mutación gaussiana de schwefel.py sobre una población."""
    m2 = exigir_mod2()
    aplicar_config(mod, a)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    LB = np.full(a.d, a.lb)
    UB = np.full(a.d, a.ub)

    Pob = mod.pob_inicial(a.N, a.d, LB, UB)
    print("\n[1] Población aleatoria")
    print(Pob)
    print("\n[2] Desviación estándar por coordenada (sigma_j de la mutación)")
    print(Pob.std(axis=0))
    mutantes = m2.mutacion_gaussiana(Pob, a.frac_m, LB, UB)
    print(f"\n[3] Mutantes: int(N * frac_m) = {len(mutantes)} individuos "
          f"(una coordenada perturbada y recortada al dominio)")
    print(mutantes)


def cmd_comparar(mod, a):
    """Corre todos los métodos (con sus parámetros por defecto) y compara."""
    aplicar_config(mod, a)
    defaults = {m: _parse([m]) for m in ("gd", "gdnum", "evo", "pso", "de")}
    for m, ns in defaults.items():
        for k in ("d", "lb", "ub", "A", "seed", "no_plot", "guardar", "funcion"):
            setattr(ns, k, getattr(a, k))
        resolver(ns)
        if m in ("evo", "pso", "de") and a.N is not None:
            ns.N = a.N
        if a.iter is not None:
            ns.iter = a.iter
        ns.x0 = None

    nombres = ["GD analítico", "GD numérico", "Evolutivo", "PSO", "DE"]
    finales = {n: [] for n in nombres}
    tiempos = {n: [] for n in nombres}
    primeros = []

    for rep in range(a.repeticiones):
        s = semilla_rep(a, rep)
        if s is not None:
            np.random.seed(s)
        corridas = [
            ejecutar_gd(mod, defaults["gd"]),
            ejecutar_gd(mod, defaults["gdnum"], numerico=True),
            ejecutar_evo(mod, defaults["evo"], s),
            ejecutar_pso(mod, defaults["pso"], s),
            ejecutar_de(mod, defaults["de"], s),
        ]
        for r in corridas:
            finales[r["nombre"]].append(r["best_fit"])
            tiempos[r["nombre"]].append(r["tiempo"])
        if rep == 0:
            primeros = corridas

    print(f"\nComparación en {a.funcion} (d = {a.d}, {a.repeticiones} repetición(es))")
    print(f"{'Método':<14}{'Mejor':>12}{'Media':>12}{'Std':>12}{'Tiempo medio':>15}")
    print("-" * 65)
    for n in nombres:
        f = np.array(finales[n])
        print(f"{n:<14}{f.min():>12.4g}{f.mean():>12.4g}{f.std():>12.4g}"
              f"{np.mean(tiempos[n]):>13.3f} s")
    graficar(primeros, f"Comparación de métodos (d = {a.d}, primera repetición)", a, mod)


def cmd_operadores(mod, a):
    """Recorre uno a uno los operadores del algoritmo evolutivo."""
    aplicar_config(mod, a)
    f_obj, _ = obtener(mod, a)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    LB = np.full(a.d, a.lb)
    UB = np.full(a.d, a.ub)

    print("\n[1] pob_inicial: población aleatoria")
    Pob = mod.pob_inicial(a.N, a.d, LB, UB)
    print(Pob)

    print("\n[2] pob_eval: población ordenada de mejor a peor + fitness")
    Pob, fit = mod.pob_eval(Pob, f_obj)
    print(np.column_stack([Pob, fit]), "\n(última columna = fitness)")

    print("\n[3] calcular_conteos: (n_elite, n_cruce, n_mutacion)")
    n_el, n_cr, n_mu = mod.calcular_conteos(a.N, a.elitismo, a.mutacion)
    print(f"  {(n_el, n_cr, n_mu)} -> suma = {n_el + n_cr + n_mu}")

    print("\n[4] elitismo: las primeras filas")
    print(mod.elitismo(Pob, n_el))

    print("\n[5] cruzamiento: padres -> hijos (fila i con i + n/2)")
    idx = np.random.choice(a.N, size=n_cr, replace=False)
    padres = Pob[idx]
    print("Padres:\n", padres)
    print("Hijos:\n", mod.cruzamiento(padres))

    print("\n[6] mutacion: una coordenada por individuo cambia")
    ids = np.random.randint(0, a.N, size=n_mu)
    print("Filas elegidas:", ids)
    print("Originales:\n", Pob[ids])
    print("Mutantes:\n", mod.mutacion(Pob, id_mutantes=ids, LB=LB, UB=UB))

    print("\n[7] next_gen: nueva población completa (sin evaluar)")
    nueva = mod.next_gen(Pob, a.elitismo, a.mutacion, LB, UB)
    print(nueva, f"\nForma: {nueva.shape}")

    print("\n[8] actualizar_mejor")
    nueva, fit_n = mod.pob_eval(nueva, f_obj)
    sol, f = mod.actualizar_mejor(nueva, fit_n, Pob[0], fit[0])
    print(f"  Mejor anterior: {fit[0]:.4f} | mejor nueva gen: {fit_n[0]:.4f} -> se conserva {f:.4f}")


def cmd_pso_paso(mod, a):
    """Muestra el estado del enjambre en las primeras iteraciones."""
    aplicar_config(mod, a)
    f_obj, _ = obtener(mod, a)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    LB = np.full(a.d, a.lb)
    UB = np.full(a.d, a.ub)
    v_max = a.frac_vmax * (UB - LB)

    pos = mod.pob_inicial(a.N, a.d, LB, UB)
    vel = mod.vel_inicial(a.N, a.d, v_max)
    fit = mod.evaluar_enjambre(pos, f_obj)
    pbest_pos, pbest_fit = pos.copy(), fit.copy()
    g = np.argmin(pbest_fit)
    gbest_pos, gbest_fit = pbest_pos[g].copy(), pbest_fit[g]

    print(f"\nv_max = {v_max}")
    print("Posiciones iniciales:\n", pos)
    print("Velocidades iniciales:\n", vel)
    print(f"Fitness inicial: {fit}\ngbest_fit inicial = {gbest_fit:.4f}")

    for t in range(1, a.pasos + 1):
        vel = mod.actualizar_velocidad(vel, pos, pbest_pos, gbest_pos,
                                       a.w, a.c1, a.c2, v_max)
        pos = mod.actualizar_posicion(pos, vel, LB, UB)
        fit = mod.evaluar_enjambre(pos, f_obj)
        pbest_pos, pbest_fit = mod.actualizar_pbest(pos, fit, pbest_pos, pbest_fit)
        gbest_pos, gbest_fit = mod.actualizar_gbest(pbest_pos, pbest_fit,
                                                    gbest_pos, gbest_fit)
        print(f"\n--- Iteración {t} ---")
        print("Velocidades:\n", vel)
        print("Posiciones:\n", pos)
        print("Fitness:", fit)
        print("pbest_fit:", pbest_fit)
        print(f"gbest_fit = {gbest_fit:.6g}   gbest_pos = {gbest_pos}")


# =====================================================================
# PARSER (sigue funcionando por línea de comandos si lo necesitas)
# =====================================================================

def construir_parser():
    comun = argparse.ArgumentParser(add_help=False)
    g = comun.add_argument_group("problema y salida")
    g.add_argument("--modulo", default=MODULO_POR_DEFECTO,
                   help="módulo principal (sin .py): rastrigin, GD, evo, PSO, DE")
    g.add_argument("--modulo2", default=MODULO2_POR_DEFECTO,
                   help="módulo secundario (sin .py): evolutivo v2, mutación gaussiana")
    g.add_argument("--funcion", choices=["rastrigin", "schwefel", "ambas"],
                   default="rastrigin", help="función objetivo (o 'ambas')")
    g.add_argument("--d", type=int, default=2, help="número de variables")
    g.add_argument("--lb", type=float, default=None,
                   help="límite inferior (por defecto: el de la función)")
    g.add_argument("--ub", type=float, default=None,
                   help="límite superior (por defecto: el de la función)")
    g.add_argument("--A", type=float, default=10.0,
                   help="constante de Rastrigin (no afecta a Schwefel)")
    g.add_argument("--seed", type=int, default=None, help="semilla aleatoria")
    g.add_argument("--no-plot", action="store_true", dest="no_plot",
                   help="no mostrar gráficos")
    g.add_argument("--guardar", metavar="RUTA", default=None,
                   help="guardar el gráfico en un archivo (png, pdf...)")

    p = argparse.ArgumentParser(
        description="Orquestador de optimización: Rastrigin y Schwefel. "
                    "Sin argumentos abre el modo interactivo.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True, metavar="comando")

    def nuevo(nombre, ayuda):
        return sub.add_parser(nombre, parents=[comun], help=ayuda,
                              description=ayuda)

    s = nuevo("eval", "Evalúa la función en un punto")
    s.add_argument("--x", type=float, nargs="+", help="punto (vacío = aleatorio)")

    s = nuevo("grad", "Compara gradiente analítico vs numérico")
    s.add_argument("--x", type=float, nargs="+", help="punto (vacío = aleatorio)")
    s.add_argument("--h", type=float, default=1e-3, help="paso de la dif. finita")

    for nom, ayuda, num in (("gd", "Descenso por gradiente analítico", False),
                            ("gdnum", "Descenso por gradiente numérico", True)):
        s = nuevo(nom, ayuda)
        s.add_argument("--x0", type=float, nargs="+", help="punto inicial")
        s.add_argument("--iter", type=int, default=250, help="iteraciones máx.")
        s.add_argument("--lr", type=float, default=None,
                       help="tasa de aprendizaje (rastrigin 0.002, schwefel 1.0)")
        s.add_argument("--tol", type=float, default=1e-5, help="tolerancia de parada")
        s.add_argument("--clip", action="store_true",
                       help="recortar los puntos al dominio [lb, ub]")
        s.add_argument("--inicios", type=int, default=1,
                       help="nº de inicios aleatorios (multi-start); se reporta el mejor")
        if num:
            s.add_argument("--h", type=float, default=1e-3, help="paso de la dif. finita")

    s = nuevo("evo", "Algoritmo evolutivo")
    s.add_argument("--N", type=int, default=50, help="tamaño de la población")
    s.add_argument("--iter", type=int, default=200, help="generaciones")
    s.add_argument("--elitismo", type=float, default=0.1, help="fracción de élites")
    s.add_argument("--mutacion", type=float, default=0.2, help="fracción de mutantes")

    s = nuevo("evo2", "Algoritmo evolutivo (versión de schwefel.py)")
    s.add_argument("--N", type=int, default=30, help="tamaño de la población")
    s.add_argument("--iter", type=int, default=50, help="generaciones")
    s.add_argument("--elitismo", type=float, default=0.2, help="fracción de élites")
    s.add_argument("--mutacion", type=float, default=0.1, help="fracción de mutantes")

    s = nuevo("pso", "Optimización por enjambre de partículas")
    s.add_argument("--N", type=int, default=40, help="partículas")
    s.add_argument("--iter", type=int, default=200, help="iteraciones")
    s.add_argument("--w", type=float, default=0.7, help="inercia")
    s.add_argument("--c1", type=float, default=1.5, help="coef. cognitivo")
    s.add_argument("--c2", type=float, default=1.5, help="coef. social")
    s.add_argument("--frac-vmax", type=float, default=0.2, dest="frac_vmax",
                   help="fracción del rango usada como velocidad máx.")

    s = nuevo("de", "Evolución diferencial")
    s.add_argument("--N", type=int, default=50, help="tamaño de la población (>= 4)")
    s.add_argument("--iter", type=int, default=200, help="generaciones")
    s.add_argument("--F", type=float, default=0.8, help="factor de mutación [0, 2]")
    s.add_argument("--CR", type=float, default=0.9, help="prob. de cruce [0, 1]")

    s = nuevo("comparar", "Compara todos los métodos con parámetros por defecto")
    s.add_argument("--N", type=int, default=None, help="tamaño población/enjambre (todos)")
    s.add_argument("--iter", type=int, default=None, help="iteraciones (todos)")
    s.add_argument("--repeticiones", type=int, default=1,
                   help="nº de ejecuciones independientes para estadísticas")

    s = nuevo("operadores", "Paso a paso de los operadores evolutivos")
    s.add_argument("--N", type=int, default=8, help="tamaño de la población de ejemplo")
    s.add_argument("--elitismo", type=float, default=0.2)
    s.add_argument("--mutacion", type=float, default=0.2)

    s = nuevo("mutgauss", "Mutación gaussiana (operador de schwefel.py)")
    s.add_argument("--N", type=int, default=8, help="tamaño de la población de ejemplo")
    s.add_argument("--frac-m", type=float, default=0.5, dest="frac_m",
                   help="fracción de individuos a mutar")

    s = nuevo("pso-paso", "Paso a paso de las iteraciones del PSO")
    s.add_argument("--N", type=int, default=5, help="partículas")
    s.add_argument("--pasos", type=int, default=3, help="iteraciones a mostrar")
    s.add_argument("--w", type=float, default=0.7)
    s.add_argument("--c1", type=float, default=1.5)
    s.add_argument("--c2", type=float, default=1.5)
    s.add_argument("--frac-vmax", type=float, default=0.2, dest="frac_vmax")

    nuevo("interactivo", "Menú interactivo")
    return p


PARSER = construir_parser()


def _parse(argv):
    return PARSER.parse_args(argv)


COMANDOS = {
    "eval": cmd_eval, "grad": cmd_grad, "gd": cmd_gd, "gdnum": cmd_gdnum,
    "evo": cmd_evo, "evo2": cmd_evo2, "pso": cmd_pso, "de": cmd_de,
    "comparar": cmd_comparar, "operadores": cmd_operadores,
    "mutgauss": cmd_mutgauss, "pso-paso": cmd_pso_paso,
}


def correr(mod, cmd, a):
    """Ejecuta un comando; con --funcion ambas lo repite para cada función."""
    if a.funcion == "ambas":
        for nombre in CONFIG_FUNCIONES:
            b = copy.copy(a)
            b.funcion = nombre
            b._sufijo = nombre
            # límites/lr de la otra función no deben arrastrarse
            if a.lb is None:
                b.lb = None
            if a.ub is None:
                b.ub = None
            COMANDOS[cmd](mod, b)
    else:
        COMANDOS[cmd](mod, a)


# =====================================================================
# MODO INTERACTIVO
# =====================================================================
# Flujo: menú principal -> panel de parámetros del método -> ejecutar.
# Los valores se recuerdan entre ejecuciones (dentro de la sesión), así
# puedes cambiar un solo parámetro y volver a correr sin reescribir nada.

PARAMS_INTERACTIVO = {
    "eval": [("x", "punto x")],
    "grad": [("x", "punto x"), ("h", "paso h de la dif. finita")],
    "gd": [("x0", "punto inicial"), ("iter", "iteraciones máx."),
           ("lr", "learning rate"), ("tol", "tolerancia de parada"),
           ("clip", "recortar al dominio (s/n)"),
           ("inicios", "nº de inicios aleatorios")],
    "gdnum": [("x0", "punto inicial"), ("iter", "iteraciones máx."),
              ("lr", "learning rate"), ("tol", "tolerancia de parada"),
              ("h", "paso h de la dif. finita"),
              ("clip", "recortar al dominio (s/n)"),
              ("inicios", "nº de inicios aleatorios")],
    "evo": [("N", "tamaño de la población"), ("iter", "generaciones"),
            ("elitismo", "fracción de élites"), ("mutacion", "fracción de mutantes")],
    "evo2": [("N", "tamaño de la población"), ("iter", "generaciones"),
             ("elitismo", "fracción de élites"), ("mutacion", "fracción de mutantes")],
    "pso": [("N", "partículas"), ("iter", "iteraciones"), ("w", "inercia"),
            ("c1", "coef. cognitivo"), ("c2", "coef. social"),
            ("frac_vmax", "fracción del rango como v_max")],
    "de": [("N", "tamaño de la población"), ("iter", "generaciones"),
           ("F", "factor de mutación F"), ("CR", "prob. de cruce CR")],
    "comparar": [("N", "tamaño población/enjambre (todos)"),
                 ("iter", "iteraciones (todos)"),
                 ("repeticiones", "nº de repeticiones")],
    "operadores": [("N", "tamaño de la población"), ("elitismo", "fracción de élites"),
                   ("mutacion", "fracción de mutantes")],
    "mutgauss": [("N", "tamaño de la población"),
                 ("frac_m", "fracción de individuos a mutar")],
    "pso-paso": [("N", "partículas"), ("pasos", "iteraciones a mostrar"),
                 ("w", "inercia"), ("c1", "coef. cognitivo"), ("c2", "coef. social"),
                 ("frac_vmax", "fracción del rango como v_max")],
}

PARAMS_GLOBALES = [
    ("funcion", "función objetivo (rastrigin / schwefel / ambas)"),
    ("d", "nº de variables (d)"),
    ("lb", "límite inferior"),
    ("ub", "límite superior"),
    ("A", "constante A de Rastrigin"),
    ("seed", "semilla aleatoria"),
    ("no_plot", "ocultar gráficos (s/n)"),
    ("guardar", "guardar gráfico en archivo (ruta, ej. grafico.png)"),
    ("modulo", "módulo principal (archivo sin .py)"),
    ("modulo2", "módulo secundario (archivo sin .py)"),
]

GLOBAL_DEFECTOS = {"funcion": "rastrigin", "d": 2, "lb": None, "ub": None,
                   "A": 10.0, "seed": None, "no_plot": False, "guardar": None,
                   "modulo": MODULO_POR_DEFECTO, "modulo2": MODULO2_POR_DEFECTO}

MENU = [
    ("eval", "Evaluar la función en un punto"),
    ("grad", "Gradiente analítico vs numérico"),
    ("gd", "Descenso por gradiente (analítico)"),
    ("gdnum", "Descenso por gradiente (numérico)"),
    ("evo", "Algoritmo evolutivo"),
    ("pso", "PSO"),
    ("de", "Evolución diferencial"),
    ("comparar", "Comparar todos los métodos"),
    ("operadores", "Operadores evolutivos paso a paso"),
    ("pso-paso", "PSO paso a paso"),
]

# Opciones que dependen del módulo secundario (schwefel.py)
MENU_EXTRA = [
    ("evo2", "Algoritmo evolutivo (versión del módulo secundario)"),
    ("mutgauss", "Mutación gaussiana (módulo secundario)"),
]

TIPOS = {"x": "lista", "x0": "lista", "clip": "bool", "no_plot": "bool",
         "N": "int", "iter": "int", "inicios": "int", "repeticiones": "int",
         "pasos": "int", "d": "int", "seed": "int",
         "guardar": "str", "modulo": "str", "modulo2": "str",
         "funcion": "funcion"}

SI = ("s", "si", "sí", "y", "yes", "1", "true", "t")
NO = ("n", "no", "0", "false", "f")


def convertir(nombre, texto):
    """Convierte el texto escrito al tipo del parámetro (ValueError si es inválido)."""
    tipo = TIPOS.get(nombre, "float")
    if tipo == "lista":
        return [float(v) for v in texto.replace(",", " ").split()]
    if tipo == "bool":
        t = texto.lower()
        if t in SI:
            return True
        if t in NO:
            return False
        raise ValueError("responde s o n")
    if tipo == "int":
        return int(texto)
    if tipo == "str":
        return texto
    if tipo == "funcion":
        t = texto.lower()
        if t not in ("rastrigin", "schwefel", "ambas"):
            raise ValueError("elige rastrigin, schwefel o ambas")
        return t
    return float(texto)


def mostrar_valor(v):
    if v is None or v == []:
        return "(por defecto / aleatorio)"
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, list):
        return fmt(v)
    return str(v)


def panel(titulo, params, valores, defectos, ejecutar=None):
    """Muestra un panel de parámetros editable.

    - Escribe el número de un parámetro para editarlo (Enter vacío = valor por defecto).
    - Enter sin nada: ejecuta (si hay `ejecutar`) o vuelve.
    - 'v': volver al menú anterior.
    """
    while True:
        print(f"\n--- {titulo} ---")
        for i, (nombre, desc) in enumerate(params, 1):
            print(f"  {i:>2}. {desc:<52} = {mostrar_valor(valores[nombre])}")
        if ejecutar:
            print("  [Enter] ejecutar   |   [número] editar parámetro   |   [v] volver")
        else:
            print("  [número] editar parámetro   |   [Enter / v] volver")
        op = input("> ").strip().lower()

        if op == "v" or (op == "" and ejecutar is None):
            return
        if op == "":
            ejecutar()
            continue
        if not op.isdigit() or not (1 <= int(op) <= len(params)):
            print("  Opción no válida.")
            continue

        nombre, desc = params[int(op) - 1]
        texto = input(f"  {desc} [{mostrar_valor(valores[nombre])}] "
                      f"(vacío = valor por defecto): ").strip()
        if texto == "":
            nuevo = defectos[nombre]
        else:
            try:
                nuevo = convertir(nombre, texto)
            except ValueError as e:
                print(f"  Valor inválido ({e}); se conserva el actual.")
                continue
        if nombre == "funcion" and nuevo != valores["funcion"]:
            # cada función tiene su dominio: volver a los límites por defecto
            valores["lb"] = valores["ub"] = None
        valores[nombre] = nuevo


def sincronizar_modulos(cfg, cargados):
    """Si en la configuración global cambió el nombre de un módulo, lo reimporta."""
    for clave in ("modulo", "modulo2"):
        if cfg[clave] == cargados["nombres"][clave]:
            continue
        nuevo = cargar_modulo(cfg[clave], obligatorio=False)
        if nuevo is None:
            print(f"  Se conserva el módulo anterior '{cargados['nombres'][clave]}'.")
            cfg[clave] = cargados["nombres"][clave]
            continue
        cargados["nombres"][clave] = cfg[clave]
        if clave == "modulo":
            cargados["mod"] = nuevo
        else:
            EXTRA["mod2"] = nuevo
        print(f"  Módulo '{cfg[clave]}' cargado.")


def modo_interactivo(mod, nombres):
    cfg = dict(GLOBAL_DEFECTOS)
    cfg["modulo"], cfg["modulo2"] = nombres
    cargados = {"mod": mod, "nombres": {"modulo": nombres[0], "modulo2": nombres[1]}}

    todos = [c for c, _ in MENU + MENU_EXTRA]
    defectos = {cmd: {n: getattr(_parse([cmd]), n) for n, _ in PARAMS_INTERACTIVO[cmd]}
                for cmd in todos}
    estado = {cmd: dict(vals) for cmd, vals in defectos.items()}

    def ejecutar(cmd):
        sincronizar_modulos(cfg, cargados)
        a = _parse([cmd])
        for k, v in cfg.items():
            setattr(a, k, v)
        for k, v in estado[cmd].items():
            setattr(a, k, v)
        try:
            correr(cargados["mod"], cmd, a)
        except KeyboardInterrupt:
            print("\n[Interrumpido]")
        except Exception as e:  # mostrar el error sin cerrar el menú
            print(f"\n[Error] {type(e).__name__}: {e}")

    print("\n=== Modo interactivo - Optimización (Rastrigin / Schwefel) ===")
    while True:
        menu = MENU + (MENU_EXTRA if EXTRA["mod2"] is not None else [])
        lb = cfg["lb"] if cfg["lb"] is not None else "defecto"
        ub = cfg["ub"] if cfg["ub"] is not None else "defecto"
        print(f"\nConfig global: función={cfg['funcion']}  d={cfg['d']}  "
              f"dominio=[{lb}, {ub}]  A={cfg['A']}  semilla={cfg['seed']}  "
              f"gráficos={'no' if cfg['no_plot'] else 'sí'}")
        print(f"Módulos: principal={cfg['modulo']}.py  "
              f"secundario={cfg['modulo2']}.py"
              + ("" if EXTRA["mod2"] is not None else "  (NO cargado)"))
        for i, (_, texto) in enumerate(menu, 1):
            print(f"  {i:>2}. {texto}")
        n_cfg = len(menu) + 1
        print(f"  {n_cfg:>2}. Configuración global (función, d, límites, A, semilla, "
              f"gráficos, módulos)")
        print("   0. Salir")
        op = input("Opción: ").strip()

        if op in ("0", "q", "salir"):
            print("¡Hasta luego!")
            return
        if not op.isdigit() or not (1 <= int(op) <= n_cfg):
            print("Opción no válida.")
            continue
        if int(op) == n_cfg:
            panel("Configuración global", PARAMS_GLOBALES, cfg, GLOBAL_DEFECTOS)
            sincronizar_modulos(cfg, cargados)
            continue

        cmd, texto = menu[int(op) - 1]
        panel(f"{texto}  ('{cmd}')", PARAMS_INTERACTIVO[cmd], estado[cmd],
              defectos[cmd], ejecutar=lambda c=cmd: ejecutar(c))


# =====================================================================
# MAIN
# =====================================================================

def main():
    """
Orquestador para optimizar las funciones de Rastrigin y Schwefel.

USO PRINCIPAL (interactivo):
    python main.py

Se abre un menú en la terminal donde eliges el método, cambias los
parámetros y ejecutas, sin escribir comandos largos. Los valores se
recuerdan mientras la sesión siga abierta.

Módulos que importa (deben estar junto a main.py):
    rastrigin.py   principal: rastrigin, GD, evolutivo, PSO, DE
    schwefel.py    secundario: evolutivo v2 y mutación gaussiana

También se puede usar por línea de comandos:
    python main.py gd --funcion rastrigin --d 2 --lr 0.002 --iter 500 --inicios 5
    python main.py evo --funcion schwefel --d 5 --N 80 --iter 300 --seed 1
    python main.py evo2 --funcion rastrigin --d 3
    python main.py comparar --funcion ambas --d 5 --repeticiones 10 --seed 0
"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--modulo", default=MODULO_POR_DEFECTO)
    pre.add_argument("--modulo2", default=MODULO2_POR_DEFECTO)
    conocidos, resto = pre.parse_known_args()

    def preparar(nombre1, nombre2):
        mod = cargar_modulo(nombre1)
        EXTRA["mod2"] = cargar_modulo(nombre2, obligatorio=False)
        return mod

    # Sin argumentos (o solo --modulo/--modulo2) -> modo interactivo directo
    if not resto:
        mod = preparar(conocidos.modulo, conocidos.modulo2)
        try:
            modo_interactivo(mod, (conocidos.modulo, conocidos.modulo2))
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
        return

    args = PARSER.parse_args()
    mod = preparar(args.modulo, args.modulo2)

    if args.comando == "interactivo":
        try:
            modo_interactivo(mod, (args.modulo, args.modulo2))
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
        return
    try:
        correr(mod, args.comando, args)
    except ValueError as e:
        print(f"[Error de parámetros] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()