"""
Orquestador de la Parte 2: recorrido de las 47 capitales de provincia de la
España peninsular (TSP) con algoritmo genético.

Uso:
    python Parte_2/orquestador.py                menú interactivo
    python Parte_2/orquestador.py ga --valor-hora 25 --mutacion mixta
    python Parte_2/orquestador.py comparar --repeticiones 30 --procesos 8
    python Parte_2/orquestador.py barrido --parametro elite --valores 0 2 10 20
    python Parte_2/orquestador.py hora --vh-min 0 --vh-max 60 --vh-paso 5
    python Parte_2/orquestador.py limpiar        borra los archivos generados por el GA

Comandos: ga, comparar, barrido, hora, limpiar.
El algoritmo está en scripts/algoritmo_genetico.py y scripts/operadores_ga.py,
los experimentos en scripts/experimentos_ga.py y los costos en scripts/costos.py.
Los archivos se guardan en la carpeta outputs/ de la raíz del repositorio.
"""

import argparse                  # lee los parámetros de la línea de comandos
import os                        # cuenta los núcleos para repartir las corridas
import re                        # valida el sufijo de los archivos de salida
import sys                       # sale con mensaje de error si algo falla
from pathlib import Path

# Los módulos de la Parte 2 están en scripts/, junto a este archivo.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import algoritmo_genetico as ag
import experimentos_ga as ex
from operadores_ga import CRUCES, MUTACIONES


# =====================================================================
# 1. PARÁMETROS (una sola tabla: alimenta la línea de comandos y el menú)
# =====================================================================
# nombre: (valor por defecto, tipo, descripción). None = valor propio del comando.

PARAMS = {
    # Problema y salida
    "valor_hora": (15.0, float, "valor de la hora del vendedor (EUR/h)"),
    "consumo": (5.5, float, "consumo del vehículo (L/100 km)"),
    "precio": (1.849, float, "precio del combustible (EUR/L)"),
    "semilla": (0, int, "semilla aleatoria (la corrida k usa semilla + k)"),
    "version": ("0001", str, "sufijo de los archivos de salida"),
    # Algoritmo genético
    "poblacion": (100, int, "tamaño de la población"),
    "generaciones": (1500, int, "número de generaciones"),
    "padres": (50, int, "padres seleccionados por generación (torneo)"),
    "elite": (2, int, "mejores individuos que pasan sin cambios (0 = sin elitismo)"),
    "k_torneo": (3, int, "tamaño del torneo de selección"),
    "cruce": ("ox", str, f"operador de cruce ({' / '.join(sorted(CRUCES))})"),
    "mutacion": ("mixta", str, f"operador de mutación ({' / '.join(sorted(MUTACIONES))})"),
    "p_mutacion": (0.3, float, "probabilidad de que un hijo mute (0 a 1)"),
    "sin_gif": (False, bool, "no generar el GIF (solo JSON y PNG)"),
    # Experimentos con varias corridas
    "repeticiones": (30, int, "nº de corridas independientes por configuración"),
    "tolerancia": (2.0, float, "éxito si el costo <= referencia x (1 + tolerancia/100)"),
    "referencia": (None, float, "costo de referencia en EUR (por defecto, el mejor del experimento)"),
    "procesos": (max(1, (os.cpu_count() or 2) // 2), int, "procesos en paralelo (no cambia los resultados, solo el tiempo)"),
    "parametro": ("p_mutacion", str, f"barrido: parámetro que varía ({' / '.join(ex.BARRIDOS)})"),
    "valores": (None, list, "barrido: valores a probar (por defecto, los propios del parámetro)"),
    # Estudio del valor de la hora
    "vh_min": (0.0, float, "hora: valor de la hora mínimo (EUR/h)"),
    "vh_max": (60.0, float, "hora: valor de la hora máximo (EUR/h)"),
    "vh_paso": (5.0, float, "hora: paso entre valores (EUR/h)"),
    "rep_hora": (20, int, "hora: corridas por cada valor de la hora"),
}

# Parámetros que muestra el menú de cada comando
PARAMS_GA = ["poblacion", "generaciones", "padres", "elite", "k_torneo", "cruce", "mutacion", "p_mutacion"]
PARAMS_COMANDO = {
    "ga": [*PARAMS_GA, "sin_gif"],
    "comparar": ["repeticiones", "tolerancia", "referencia", "procesos",
                 "poblacion", "generaciones", "padres", "elite", "k_torneo", "p_mutacion"],
    "barrido": ["parametro", "valores", "repeticiones", "tolerancia", "referencia", "procesos", *PARAMS_GA],
    "hora": ["vh_min", "vh_max", "vh_paso", "rep_hora", "procesos", *PARAMS_GA],
}
PARAMS_GLOBALES = ["valor_hora", "consumo", "precio", "semilla", "version"]

TITULOS = {
    "ga": "Algoritmo genético: una corrida (JSON, mapa y GIF)",
    "comparar": "Comparar operadores de cruce y mutación (varias corridas)",
    "barrido": "Barrido de un parámetro del GA (varias corridas)",
    "hora": "Estudio del valor de la hora (¿cambia la ruta? ¿desde qué valor?)",
}
FUNCIONES_COMANDO = {
    "ga": ag.corrida,
    "comparar": ex.comparar_operadores,
    "barrido": ex.barrido,
    "hora": ex.estudio_valor_hora,
}
OPCIONES = {"cruce": sorted(CRUCES), "mutacion": sorted(MUTACIONES), "parametro": list(ex.BARRIDOS)}


# =====================================================================
# 2. COMANDOS
# =====================================================================

def correr(comando, P):
    """Valida los parámetros comunes y ejecuta un comando."""
    p = dict(P)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", p["version"]):
        raise ValueError("version solo admite letras, números, guion y guion bajo.")
    if p["valor_hora"] < 0 or p["consumo"] < 0 or p["precio"] < 0:
        raise ValueError("valor_hora, consumo y precio no pueden ser negativos.")
    if p["procesos"] < 1:
        raise ValueError("procesos debe ser al menos 1.")
    hora = "" if comando == "hora" else f"valor de la hora = {p['valor_hora']:g} EUR/h | "
    print(f"\n>>> {TITULOS[comando]}\n    {hora}consumo = {p['consumo']:g} L/100 km | "
          f"precio = {p['precio']:g} EUR/L | semilla = {p['semilla']} | versión = {p['version']}")
    FUNCIONES_COMANDO[comando](p)


def archivos_ga():
    """Archivos generados por el GA en outputs/ (no toca los de otros algoritmos)."""
    if not ag.SALIDAS.is_dir():
        return []
    return sorted(a for a in ag.SALIDAS.iterdir()
                  if a.is_file() and "_GA_" in a.name and a.suffix.lower() in (".png", ".gif", ".json", ".csv"))


def borrar_salidas(confirmar=True):
    """Borra los archivos generados por el GA (.png, .gif, .json y .csv con _GA_ en el nombre)."""
    archivos = archivos_ga()
    if not archivos:
        print("  No hay archivos del GA para borrar.")
        return
    print(f"  Se borrarán {len(archivos)} archivo(s) del GA de: {ag.SALIDAS}")
    if confirmar and input("  ¿Continuar? (s/n): ").strip().lower() not in ("s", "si", "sí", "y"):
        print("  Cancelado.")
        return
    for a in archivos:
        a.unlink()
    print("  Listo.")


# =====================================================================
# 3. MENÚ INTERACTIVO
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
    if nombre in OPCIONES and texto not in OPCIONES[nombre]:
        raise ValueError("elige " + ", ".join(OPCIONES[nombre]))
    try:
        if tipo is list:
            return [float(v) for v in texto.replace(",", " ").split()]
        return tipo(texto)
    except ValueError:
        raise ValueError("se esperaba un número entero" if tipo is int else "se esperaba un número") from None


def mostrar(v):
    if v is None or v == []:
        return "(por defecto)"
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, list):
        return " ".join(f"{x:g}" for x in v)
    return str(v)


def panel(titulo, nombres, P, ejecutar_fn=None):
    """Panel editable de parámetros. Enter ejecuta (si hay ejecutar_fn) o vuelve; v vuelve."""
    while True:
        print(f"\n--- {titulo} ---")
        ancho = max(len(PARAMS[n][2]) for n in nombres)
        for i, n in enumerate(nombres, 1):
            print(f"  {i:>2}. {PARAMS[n][2]:<{ancho}} = {mostrar(P[n])}")
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
        if n == "parametro":  # cada parámetro del barrido tiene sus propios valores por defecto
            P["valores"] = None


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

    print("\n=== Parte 2: recorrido de las 47 capitales (algoritmo genético) ===")
    while True:
        print(f"\nConfig: valor de la hora={P['valor_hora']:g} EUR/h  consumo={P['consumo']:g} L/100 km  "
              f"precio={P['precio']:g} EUR/L  semilla={P['semilla']}  versión={P['version']}")
        for i, c in enumerate(comandos, 1):
            print(f"  {i}. {TITULOS[c]}")
        n_lim, n_cfg = len(comandos) + 1, len(comandos) + 2
        print(f"  {n_lim}. Borrar los archivos generados por el GA")
        print(f"  {n_cfg}. Configuración global (valor de la hora, consumo, precio, semilla, versión)")
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
# 4. MAIN
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
                        help="comando a ejecutar, o 'limpiar' para borrar los archivos del GA")
    for n, (defecto, tipo, ayuda) in PARAMS.items():
        flag = "--" + n.replace("_", "-")
        if tipo is bool:
            parser.add_argument(flag, dest=n, action="store_true", help=ayuda)
        elif tipo is list:
            parser.add_argument(flag, dest=n, type=float, nargs="+", help=ayuda)
        elif n in OPCIONES:
            parser.add_argument(flag, dest=n, default=defecto, choices=OPCIONES[n], help=ayuda)
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
