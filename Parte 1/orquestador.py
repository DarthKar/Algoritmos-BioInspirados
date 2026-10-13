"""
Orquestador de la Parte 1: Rastrigin y Schwefel con descenso por gradiente
(analítico y numérico), algoritmo evolutivo, PSO y evolución diferencial.

Uso:
    python orquestador.py                menú interactivo
    python orquestador.py gd --funcion schwefel --d 3 --gif
    python orquestador.py evo --tipo-mutacion gaussiana --N 80 --iter 300
    python orquestador.py comparar --funcion ambas --d 2 --repeticiones 30 --seed 0
    python orquestador.py limpiar        borra las figuras y GIF generados

Comandos: gd, gdnum, evo, pso, de, comparar, limpiar.
Los algoritmos y las animaciones están en funciones.py (misma carpeta).
Las figuras y GIF se guardan en la carpeta salidas/, junto a este archivo.
"""

import argparse                  # lee los parámetros de la línea de comandos
import os                        # crea la carpeta de salidas
import sys                       # sale con mensaje de error si algo falla
import time                      # mide el tiempo de cada corrida

import numpy as np
import matplotlib.pyplot as plt  # figuras de convergencia y de comparación

import funciones as fn

# Carpeta de salida junto a este archivo (dentro de "Parte 1"), sin importar
# desde qué carpeta se ejecute el programa.
SALIDAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")


# =====================================================================
# 1. FUNCIONES OBJETIVO Y SU CONFIGURACIÓN
# =====================================================================

A = 10  # constante de Rastrigin


def rastrigin(X):
    """f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)). Mínimo global 0 en x = 0."""
    X = np.atleast_2d(X)
    f = A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    return f if f.size > 1 else f[0]


def rastrigin_grad(x):
    """df/dx_i = 2*x_i + 2*pi*A*sin(2*pi*x_i)."""
    x = np.asarray(x, dtype=float)
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


def schwefel(X):
    """f(x) = 418.9829*n - sum(x_i*sin(sqrt(|x_i|))). Mínimo global ~0 en x_i = 420.9687."""
    X = np.atleast_2d(X)
    f = 418.9828872724339 * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
    return f if f.size > 1 else f[0]


def schwefel_grad(x):
    """df/dx_i = -sin(s_i) - (s_i/2)*cos(s_i), con s_i = sqrt(|x_i|)."""
    s = np.sqrt(np.abs(np.asarray(x, dtype=float)))
    return -np.sin(s) - 0.5 * s * np.cos(s)


# Valores propios de cada función (se usan cuando el parámetro es None):
#   lb, ub : dominio     lr : tasa de aprendizaje del descenso
#   optimo : coordenada del mínimo global    umbral : éxito si f_final < umbral
# El umbral es menor que la diferencia entre el mínimo global y el mejor mínimo
# local (~1 en Rastrigin, ~118 en Schwefel), así que "éxito" = cuenca correcta.
FUNCIONES = {
    "rastrigin": {"f": rastrigin, "grad": rastrigin_grad, "lb": -5.12, "ub": 5.12,
                  "lr": 0.002, "optimo": 0.0, "umbral": 0.1},
    "schwefel": {"f": schwefel, "grad": schwefel_grad, "lb": -500.0, "ub": 500.0,
                 "lr": 1.0, "optimo": 420.9687, "umbral": 1.0},
}


# =====================================================================
# 2. PARÁMETROS (una sola tabla: alimenta la línea de comandos y el menú)
# =====================================================================
# nombre: (valor por defecto, tipo, descripción). None = usar el valor propio
# de la función (ver FUNCIONES) o, en x0, un punto aleatorio.

PARAMS = {
    # Problema y salida
    "funcion": ("rastrigin", str, "función objetivo (rastrigin / schwefel / ambas)"),
    "d": (2, int, "nº de variables (2 o 3 para animar)"),
    "lb": (None, float, "límite inferior (por defecto el de la función)"),
    "ub": (None, float, "límite superior (por defecto el de la función)"),
    "seed": (0, int, "semilla aleatoria (la corrida k usa seed + k)"),
    "no_plot": (False, bool, "no mostrar las figuras en pantalla (se guardan igual)"),
    "gif": (False, bool, "generar el GIF animado de la búsqueda"),
    # Descenso por gradiente
    "x0": (None, list, "punto inicial (vacío = aleatorio)"),
    "iter_gd": (250, int, "iteraciones máximas del descenso"),
    "lr": (None, float, "tasa de aprendizaje (por defecto la de la función)"),
    "tol": (1e-5, float, "tolerancia de parada del descenso"),
    "h": (1e-3, float, "paso h de las diferencias finitas (GD numérico)"),
    # Métodos de población
    "N": (50, int, "tamaño de la población / enjambre"),
    "iter": (200, int, "generaciones / iteraciones"),
    "elitismo": (0.1, float, "evolutivo: fracción de élites"),
    "mutacion": (0.2, float, "evolutivo: fracción de mutantes"),
    "tipo_mutacion": ("uniforme", str, "evolutivo: mutación (uniforme / gaussiana)"),
    "w": (0.7, float, "PSO: inercia"),
    "c1": (1.5, float, "PSO: coeficiente cognitivo"),
    "c2": (1.5, float, "PSO: coeficiente social"),
    "frac_vmax": (0.2, float, "PSO: fracción del rango usada como v_max"),
    "F": (0.8, float, "DE: factor de mutación [0, 2]"),
    "CR": (0.9, float, "DE: probabilidad de cruce [0, 1]"),
    # Comparación
    "repeticiones": (30, int, "comparar: nº de corridas independientes"),
    "umbral": (None, float, "comparar: éxito si f_final < umbral (por defecto el de la función)"),
    "costo_grad": (None, float, "evaluaciones de f que equivalen a un ∇f (por defecto 2*d)"),
}

# Parámetros que muestra el menú de cada comando
PARAMS_COMANDO = {
    "gd": ["x0", "iter_gd", "lr", "tol", "gif"],
    "gdnum": ["x0", "iter_gd", "lr", "tol", "h", "gif"],
    "evo": ["N", "iter", "elitismo", "mutacion", "tipo_mutacion", "gif"],
    "pso": ["N", "iter", "w", "c1", "c2", "frac_vmax", "gif"],
    "de": ["N", "iter", "F", "CR", "gif"],
    "comparar": ["repeticiones", "umbral", "costo_grad", "N", "iter", "iter_gd", "lr"],
}
PARAMS_GLOBALES = ["funcion", "d", "lb", "ub", "seed", "no_plot"]

TITULOS = {
    "gd": "Descenso por gradiente (gradiente analítico)",
    "gdnum": "Descenso por gradiente (gradiente numérico)",
    "evo": "Algoritmo evolutivo",
    "pso": "Optimización por enjambre de partículas (PSO)",
    "de": "Evolución diferencial",
    "comparar": "Comparar todos los métodos (varias corridas)",
}
NOMBRES = {"gd": "GD analítico", "gdnum": "GD numérico", "evo": "Evolutivo",
           "pso": "PSO", "de": "DE"}


def resolver(P, nombre):
    """Copia de P para la función `nombre`, con los None sustituidos por sus valores propios."""
    base = FUNCIONES[nombre]
    p = dict(P, funcion=nombre)
    for k in ("lb", "ub", "lr", "umbral"):
        if p[k] is None:
            p[k] = base[k]
    if p["costo_grad"] is None:
        p["costo_grad"] = 2 * p["d"]  # costo de un gradiente por diferencias centrales
    return p


def optimo(p):
    return np.full(p["d"], FUNCIONES[p["funcion"]]["optimo"])


# =====================================================================
# 3. EJECUCIÓN DE UN MÉTODO (con conteo de evaluaciones)
# =====================================================================

class Contador:
    """Envuelve una función y cuenta cuántas veces se la llama."""

    def __init__(self, func):
        self.func, self.n = func, 0

    def __call__(self, x):
        self.n += 1
        return self.func(x)


def ejecutar(metodo, p, semilla, guardar=False):
    """Corre un método y devuelve un diccionario con el resultado.

    Conteo de evaluaciones (la regla de comparación justa):
      n_f    : llamadas a f hechas por el algoritmo. En el GD numérico incluye
               las 2*d de cada gradiente por diferencias finitas.
      n_grad : llamadas al gradiente analítico (solo GD analítico).
      equiv  : n_f + costo_grad * n_grad.
    Los dos descensos suman 1 evaluación de f por el valor final. Evaluar f
    en cada punto del recorrido (para graficar) no se cuenta.
    `guardar` conserva el recorrido o las poblaciones para graficar/animar.
    """
    base = FUNCIONES[p["funcion"]]
    f, cf, cg = base["f"], Contador(base["f"]), Contador(base["grad"])
    lb, ub, d = p["lb"], p["ub"], p["d"]
    np.random.seed(semilla)
    t0 = time.perf_counter()

    es_gd = metodo in ("gd", "gdnum")
    if es_gd:
        if p["x0"]:
            x0 = np.array(p["x0"], dtype=float)
            if len(x0) != d:
                raise ValueError(f"x0 tiene {len(x0)} valores y d = {d}.")
        else:
            x0 = np.random.uniform(lb, ub, d)
        if metodo == "gd":
            pts = fn.grad_desc(x0, cg, p["iter_gd"], p["lr"], p["tol"], (lb, ub))
        else:
            pts = fn.grad_desc_num(x0, cf, p["iter_gd"], p["lr"], p["tol"], p["h"], (lb, ub))
        hist = fn.historial_fitness(pts, f)
        sol, fit, pob = pts[-1], float(hist[-1]), pts
        cf.n += 1
    else:
        if metodo == "evo":
            res = fn.mi_algoritmo_evolutivo(
                cf, d, lb, ub, p["N"], p["iter"], p["elitismo"], p["mutacion"],
                p["tipo_mutacion"], semilla, guardar_poblaciones=guardar)
        elif metodo == "pso":
            res = fn.mi_pso(cf, d, lb, ub, p["N"], p["iter"], p["w"], p["c1"], p["c2"],
                            p["frac_vmax"], guardar_poblaciones=guardar)
        else:
            res = fn.evolucion_diferencial(cf, d, lb, ub, p["N"], p["iter"], p["F"], p["CR"],
                                           semilla, guardar_poblaciones=guardar)
        sol, fit, hist, *resto = res
        fit, pob = float(fit), (resto[0] if resto else None)

    return {"nombre": NOMBRES[metodo], "gd": es_gd, "sol": sol, "fit": fit, "hist": hist,
            "pob": pob, "n_f": cf.n, "n_grad": cg.n,
            "equiv": cf.n + p["costo_grad"] * cg.n, "tiempo": time.perf_counter() - t0}


# =====================================================================
# 4. SALIDAS: resumen, figuras y GIF
# =====================================================================

def guardar_y_mostrar(fig, nombre, p):
    os.makedirs(SALIDAS, exist_ok=True)
    ruta = os.path.join(SALIDAS, f"{nombre}_{p['funcion']}_d{p['d']}.png")
    fig.savefig(ruta, dpi=150)
    print(f"  Figura guardada en: {ruta}")
    if not p["no_plot"]:
        plt.show()
    plt.close(fig)


def resumen(r, p):
    print(f"\n=== {r['nombre']} ({p['funcion']}, d = {p['d']}) ===")
    print(f"  Mejor fitness : {r['fit']:.6g}")
    print(f"  Mejor solución: {np.array2string(np.asarray(r['sol']), precision=4)}")
    print(f"  Distancia al óptimo global: {np.linalg.norm(r['sol'] - optimo(p)):.4g}")
    print(f"  Evaluaciones  : f = {r['n_f']} | ∇f analítico = {r['n_grad']} | "
          f"equivalentes (1 ∇f = {p['costo_grad']:g} f) = {r['equiv']:g}")
    print(f"  Tiempo        : {r['tiempo']:.3f} s")


def graficar(r, metodo, p):
    """Convergencia (log) y, si d = 2, mapa de contorno con el recorrido o la población."""
    d2 = p["d"] == 2
    fig, ax = plt.subplots(1, 2 if d2 else 1, figsize=(12 if d2 else 6, 5), squeeze=False)
    ax = ax[0]
    ax[0].plot(np.maximum(r["hist"], 1e-16))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Iteración / generación")
    ax[0].set_ylabel("f (escala log)" if r["gd"] else "Mejor f (escala log)")
    ax[0].set_title("Convergencia")
    ax[0].grid(alpha=0.3, which="both")
    if d2:
        fn.dibujar_contorno(ax[1], FUNCIONES[p["funcion"]]["f"], p["lb"], p["ub"])
        if r["gd"]:
            tr = np.array(r["pob"])
            ax[1].plot(tr[:, 0], tr[:, 1], ".-", color="white", ms=3, lw=1, label="recorrido")
        else:
            ax[1].scatter(*r["pob"][0].T, marker="x", color="gray", label="población inicial")
            ax[1].scatter(*r["pob"][-1].T, color="white", edgecolors="k", s=20,
                          label="población final")
        ax[1].plot(*optimo(p), "r*", ms=14, label="óptimo global", zorder=3)
        ax[1].plot(*r["sol"], "o", color="orange", mec="k", ms=7, label="mejor solución",
                   zorder=5)  # zorder mayor: queda por encima de la estrella
        ax[1].legend(fontsize=7, loc="upper right")
        ax[1].set_title("Búsqueda sobre f(x)")
    fig.suptitle(f"{r['nombre']} - {p['funcion']}")
    fig.tight_layout()
    guardar_y_mostrar(fig, metodo, p)


def hacer_gif(r, metodo, p):
    os.makedirs(SALIDAS, exist_ok=True)
    ruta = os.path.join(SALIDAS, f"{metodo}_{p['funcion']}_d{p['d']}.gif")
    print("  Generando GIF (puede tardar unos segundos; luego se abre en una ventana)...")
    fn.animar_busqueda(r["pob"], FUNCIONES[p["funcion"]]["f"], p["lb"], p["ub"], ruta,
                       titulo=f"{r['nombre']} - {p['funcion']}", estela=r["gd"],
                       optimo=optimo(p), mostrar=not p["no_plot"])
    print(f"  GIF guardado en: {ruta}")


def borrar_salidas(confirmar=True):
    """Borra las figuras (.png) y los GIF generados en la carpeta de salidas."""
    archivos = []
    if os.path.isdir(SALIDAS):
        archivos = [a for a in os.listdir(SALIDAS) if a.lower().endswith((".png", ".gif"))]
    if not archivos:
        print("  No hay figuras ni GIF para borrar.")
        return
    print(f"  Se borrarán {len(archivos)} archivo(s) (.png y .gif) de: {SALIDAS}")
    if confirmar and input("  ¿Continuar? (s/n): ").strip().lower() not in ("s", "si", "sí", "y"):
        print("  Cancelado.")
        return
    for a in archivos:
        os.remove(os.path.join(SALIDAS, a))
    print("  Listo.")


# =====================================================================
# 5. COMANDOS
# =====================================================================

def comando_simple(metodo, p):
    r = ejecutar(metodo, p, p["seed"], guardar=True)
    resumen(r, p)
    graficar(r, metodo, p)
    if p["gif"]:
        hacer_gif(r, metodo, p)


# Variantes que se comparan: (etiqueta, método, parámetros que cambian)
VARIANTES = [
    ("GD analítico", "gd", {}),
    ("GD numérico", "gdnum", {}),
    ("Evolutivo (mut. uniforme)", "evo", {"tipo_mutacion": "uniforme"}),
    ("Evolutivo (mut. gaussiana)", "evo", {"tipo_mutacion": "gaussiana"}),
    ("PSO", "pso", {}),
    ("DE", "de", {}),
]


def comando_comparar(p):
    """Varias corridas independientes por método y tabla de estadísticas."""
    if p["repeticiones"] < 30:
        raise ValueError("Para la comparación deben ejecutarse al menos 30 corridas por método.")
    if p["costo_grad"] is None or p["costo_grad"] <= 0:
        raise ValueError("costo_grad debe ser un valor positivo.")
    if p["umbral"] is None:
        raise ValueError("Debe definirse un umbral de éxito para la comparación.")

    p = dict(p, x0=None)  # el punto inicial del descenso es aleatorio en cada corrida
    R = p["repeticiones"]
    res = {}

    print(f"\nRegla de comparación justa: equivalente = n_f + {p['costo_grad']} * n_∇f")
    print(f"Umbral de éxito: f_final < {p['umbral']}")
    print(f"Se ejecutarán {R} corridas independientes por método")

    for etiqueta, metodo, extra in VARIANTES:
        print(f"  Corriendo {etiqueta} ({R} corridas)...")
        q = dict(p, **extra)
        res[etiqueta] = [ejecutar(metodo, q, p["seed"] + k) for k in range(R)]

    print(f"\nComparación en {p['funcion']} (d = {p['d']}, {R} corridas, semillas "
          f"{p['seed']}..{p['seed'] + R - 1})")
    print(f"Éxito: f_final < {p['umbral']:g}.  Equivalente = evals f + {p['costo_grad']:g} * evals ∇f")
    print(f"{'Método':<28}{'Media':>10}{'Desv.':>10}{'Mejor':>10}{'Peor':>10}{'Éxito':>8}"
          f"{'Evals f':>10}{'Evals ∇f':>10}{'Equiv.':>10}{'t (s)':>8}")
    print("-" * 114)
    for etiqueta, rs in res.items():
        v = np.array([r["fit"] for r in rs])
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"{etiqueta:<28}{v.mean():>10.4g}{v.std():>10.4g}{v.min():>10.4g}{v.max():>10.4g}"
              f"{100 * np.mean(v < p['umbral']):>7.0f}%{m('n_f'):>10.0f}{m('n_grad'):>10.0f}"
              f"{m('equiv'):>10.0f}{m('tiempo'):>8.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([np.maximum([r["fit"] for r in rs], 1e-16) for rs in res.values()])
    ax.set_xticks(range(1, len(res) + 1), list(res), rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("f final (escala log)")
    ax.set_title(f"Valor final en {R} corridas - {p['funcion']} (d = {p['d']})")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    guardar_y_mostrar(fig, "comparar", p)


def correr(comando, P):
    """Ejecuta un comando; con funcion = ambas lo repite para cada función."""
    nombres = list(FUNCIONES) if P["funcion"] == "ambas" else [P["funcion"]]
    for nombre in nombres:
        p = resolver(P, nombre)
        print(f"\n>>> {TITULOS[comando]} | {nombre} | d = {p['d']} | "
              f"dominio = [{p['lb']}, {p['ub']}] | semilla = {p['seed']}")
        if comando == "comparar":
            comando_comparar(p)
        else:
            comando_simple(comando, p)


# =====================================================================
# 6. MENÚ INTERACTIVO
# =====================================================================

def convertir(nombre, texto):
    """Convierte el texto escrito al tipo del parámetro (ValueError si es inválido)."""
    tipo = PARAMS[nombre][1]
    if tipo is bool:
        t = texto.lower()
        if t in ("s", "si", "sí", "y", "1"):
            return True
        if t in ("n", "no", "0"):
            return False
        raise ValueError("responde s o n")
    if tipo is list:
        return [float(v) for v in texto.replace(",", " ").split()]
    if nombre == "funcion" and texto not in (*FUNCIONES, "ambas"):
        raise ValueError("elige rastrigin, schwefel o ambas")
    if nombre == "tipo_mutacion" and texto not in ("uniforme", "gaussiana"):
        raise ValueError("elige uniforme o gaussiana")
    return tipo(texto)


def mostrar(v):
    if v is None or v == []:
        return "(por defecto)"
    if isinstance(v, bool):
        return "sí" if v else "no"
    return str(v)


def panel(titulo, nombres, P, ejecutar_fn=None):
    """Panel editable de parámetros. Enter ejecuta (si hay ejecutar_fn) o vuelve; v vuelve."""
    while True:
        print(f"\n--- {titulo} ---")
        for i, n in enumerate(nombres, 1):
            print(f"  {i:>2}. {PARAMS[n][2]:<66} = {mostrar(P[n])}")
        print(f"  [Enter] {'ejecutar' if ejecutar_fn else 'volver'}  |  "
              f"[número] editar parámetro  |  [v] volver")
        op = input("> ").strip().lower()
        if op == "v" or (op == "" and ejecutar_fn is None):
            return
        if op == "":
            ejecutar_fn()
            continue
        if not (op.isdigit() and 1 <= int(op) <= len(nombres)):
            print("  Opción no válida.")
            continue
        n = nombres[int(op) - 1]
        texto = input("  Nuevo valor (vacío = valor por defecto): ").strip()
        try:
            P[n] = PARAMS[n][0] if texto == "" else convertir(n, texto)
        except ValueError as e:
            print(f"  Valor inválido ({e}); se conserva el actual.")
            continue
        if n == "funcion":  # cada función tiene su dominio, lr y umbral
            for k in ("lb", "ub", "lr", "umbral"):
                P[k] = None


def menu():
    P = {n: v[0] for n, v in PARAMS.items()}  # parámetros compartidos por todos los comandos
    comandos = list(TITULOS)

    def lanzar(c):
        try:
            correr(c, P)
        except KeyboardInterrupt:
            print("\n[Interrumpido]")
        except Exception as e:  # mostrar el error sin cerrar el menú
            print(f"\n[Error] {type(e).__name__}: {e}")

    print("\n=== Optimización: Rastrigin y Schwefel ===")
    while True:
        print(f"\nConfig: función={P['funcion']}  d={P['d']}  semilla={P['seed']}  "
              f"figuras en pantalla={'no' if P['no_plot'] else 'sí'}")
        for i, c in enumerate(comandos, 1):
            print(f"  {i}. {TITULOS[c]}")
        n_lim, n_cfg = len(comandos) + 1, len(comandos) + 2
        print(f"  {n_lim}. Borrar figuras y GIF generados")
        print(f"  {n_cfg}. Configuración global (función, d, límites, semilla, figuras)")
        print("  0. Salir")
        op = input("Opción: ").strip()
        if op in ("0", "q", "salir"):
            print("¡Hasta luego!")
            return
        if not (op.isdigit() and 1 <= int(op) <= n_cfg):
            print("Opción no válida.")
        elif int(op) == n_lim:
            borrar_salidas()
        elif int(op) == n_cfg:
            panel("Configuración global", PARAMS_GLOBALES, P)
        else:
            c = comandos[int(op) - 1]
            panel(TITULOS[c], PARAMS_COMANDO[c], P, lambda c=c: lanzar(c))


# =====================================================================
# 7. MAIN
# =====================================================================

def main():
    if len(sys.argv) == 1:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
        return

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", choices=[*TITULOS, "limpiar"],
                        help="método a ejecutar, o 'limpiar' para borrar figuras y GIF")
    for n, (defecto, tipo, ayuda) in PARAMS.items():
        flag = "--" + n.replace("_", "-")
        if tipo is bool:
            parser.add_argument(flag, dest=n, action="store_true", help=ayuda)
        elif tipo is list:
            parser.add_argument(flag, dest=n, type=float, nargs="+", help=ayuda)
        elif n == "funcion":
            parser.add_argument(flag, dest=n, default=defecto,
                                choices=[*FUNCIONES, "ambas"], help=ayuda)
        elif n == "tipo_mutacion":
            parser.add_argument(flag, dest=n, default=defecto,
                                choices=["uniforme", "gaussiana"], help=ayuda)
        else:
            parser.add_argument(flag, dest=n, type=tipo, default=defecto, help=ayuda)
    args = parser.parse_args()

    if args.comando == "limpiar":
        borrar_salidas(confirmar=False)
        return
    try:
        correr(args.comando, {n: getattr(args, n) for n in PARAMS})
    except ValueError as e:
        sys.exit(f"[Error de parámetros] {e}")

if __name__ == "__main__":
    main()