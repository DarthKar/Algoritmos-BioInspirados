"""
Orquestador de la Parte 2: recorrido de las 47 capitales de provincia de la
España peninsular (TSP) con algoritmo genético y colonia de hormigas.

Uso:
    python Parte_2/orquestador.py                menú interactivo
    python Parte_2/orquestador.py ga --valor-hora 25 --mutacion mixta
    python Parte_2/orquestador.py aco --valor-hora 25 --hormigas 40 --semilla 7 --version 0002
    python Parte_2/orquestador.py comparar --repeticiones 30 --procesos 8
    python Parte_2/orquestador.py hora --vh-min 0 --vh-max 60 --vh-paso 5
    python Parte_2/orquestador.py limpiar        borra los archivos generados por el GA
    python Parte_2/orquestador.py --config otra.json ga      usa otro archivo de configuración

Comandos: ga, comparar, hora, aco, limpiar.
Al arrancar se lee Parte_2/config_ejemplo.json: sus valores sustituyen a los de
la tabla PARAMS. Prioridad: línea de comandos > archivo de configuración > tabla.
El algoritmo está en scripts/algoritmo_genetico.py y scripts/operadores_ga.py,
los experimentos en scripts/experimentos_ga.py y los costos en scripts/costos.py.
Los archivos se guardan en la carpeta outputs/ de la raíz del repositorio.
ACO se ejecuta en un proceso independiente mediante scripts/colonia_de_hormigas.py.
Su semilla se toma de "semilla" y se transmite como ACO_RANDOM_SEED al proceso hijo.
Mantiene el máximo de 500 iteraciones y la parada anticipada del script original.
"""

import argparse                  # lee los parámetros de la línea de comandos
import json                      # lee el archivo de configuración
import os                        # cuenta los núcleos para repartir las corridas
import re                        # valida el sufijo de los archivos de salida
import sys                       # sale con mensaje de error si algo falla
import math                      # valida parámetros ACO finitos
import subprocess                # ejecuta ACO sin cambiar el estado del GA
from pathlib import Path

# Los módulos de la Parte 2 están en scripts/, junto a este archivo.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import algoritmo_genetico as ag
import experimentos_ga as ex
from operadores_ga import CRUCES, MUTACIONES


# =====================================================================
# 1. PARÁMETROS (una sola tabla: alimenta la línea de comandos y el menú)
# =====================================================================
# nombre: (valor por defecto, tipo, descripción).
# El archivo de configuración puede sustituir cualquiera de estos valores por defecto.

PARAMS = {
    # Problema y salida
    "vehiculo": ("", str, "vehículo (texto informativo: el cálculo usa consumo y precio)"),
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
    # Colonia de hormigas (solo el comando aco)
    "hormigas": (40, int, "ACO: cantidad de hormigas por iteración"),
    "evaporacion": (0.05, float, "ACO: tasa de evaporación de feromonas (0 <= valor < 1)"),
    "intensificacion": (2.0, float, "ACO: depósito constante de feromonas"),
    "alfa": (2.0, float, "ACO: importancia de las feromonas"),
    "beta": (1.0, float, "ACO: importancia de la información heurística"),
    "beta_evaporacion": (0.0, float, "ACO: reducción de beta por iteración (0 a 1)"),
    "p_mejor": (0.1, float, "ACO: probabilidad de elegir la conexión de mayor peso (0 a 1)"),
    # Experimentos con varias corridas
    "repeticiones": (30, int, "nº de corridas independientes por configuración"),
    "tolerancia": (2.0, float, "éxito si el costo <= mejor del experimento x (1 + tolerancia/100)"),
    "procesos": (max(1, (os.cpu_count() or 2) // 2), int, "procesos en paralelo (no cambia los resultados, solo el tiempo)"),
    # Estudio del valor de la hora
    "vh_min": (0.0, float, "hora: valor de la hora mínimo (EUR/h)"),
    "vh_max": (60.0, float, "hora: valor de la hora máximo (EUR/h)"),
    "vh_paso": (5.0, float, "hora: paso entre valores (EUR/h)"),
    "rep_hora": (20, int, "hora: corridas por cada valor de la hora"),
}

# Parámetros que muestra el menú de cada comando
PARAMS_GA = ["poblacion", "generaciones", "padres", "elite", "k_torneo", "cruce", "mutacion", "p_mutacion"]
PARAMS_ACO = ["hormigas", "evaporacion", "intensificacion", "alfa", "beta", "beta_evaporacion", "p_mejor"]
PARAMS_COMANDO = {
    "ga": [*PARAMS_GA, "sin_gif"],
    "aco": PARAMS_ACO,
    "comparar": ["repeticiones", "tolerancia", "procesos",
                 "poblacion", "generaciones", "padres", "elite", "k_torneo", "p_mutacion"],
    "hora": ["vh_min", "vh_max", "vh_paso", "rep_hora", "procesos", *PARAMS_GA],
}
PARAMS_GLOBALES = ["valor_hora", "consumo", "precio", "vehiculo", "semilla", "version"]

TITULOS = {
    "ga": "Algoritmo genético: una corrida (JSON, mapa y GIF)",
    "comparar": "Comparar operadores de cruce y mutación (varias corridas)",
    "hora": "Estudio del valor de la hora (¿cambia la ruta? ¿desde qué valor?)",
}
TITULOS["aco"] = "Colonia de hormigas: una corrida (JSON, mapa, GIF y MP4)"
FUNCIONES_COMANDO = {
    "ga": ag.corrida,
    "comparar": ex.comparar_operadores,
    "hora": ex.estudio_valor_hora,
}
OPCIONES = {"cruce": sorted(CRUCES), "mutacion": sorted(MUTACIONES)}

# Archivo de configuración que se lee al arrancar (--config indica otro)
CONFIG = Path(__file__).resolve().parent / "config_ejemplo.json"

# Valor de partida de cada parámetro: el de la tabla, sustituido por el del archivo de configuración
DEFECTO = {n: v[0] for n, v in PARAMS.items()}


def leer_config(ruta):
    """Lee un archivo de configuración JSON y devuelve {parámetro: valor} ya validado.

    Cada clave es un nombre de PARAMS. Las claves que empiezan por "_" son comentarios.
    Lanza ValueError si hay una clave desconocida o un valor del tipo equivocado.
    """
    nombre = Path(ruta).name
    try:
        with open(ruta, encoding="utf-8-sig") as f:      # utf-8-sig: admite el archivo con o sin BOM
            datos = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{nombre}: no es un JSON válido (línea {e.lineno}: {e.msg}).") from None
    if not isinstance(datos, dict):
        raise ValueError(f"{nombre}: debe ser un objeto JSON con pares parámetro: valor.")

    config = {}
    for clave, valor in datos.items():
        if clave.startswith("_"):
            continue
        if clave not in PARAMS:
            raise ValueError(f"{nombre}: parámetro desconocido '{clave}'. Válidos: {', '.join(PARAMS)}.")
        tipo = PARAMS[clave][1]
        if tipo is bool:
            if not isinstance(valor, bool):
                raise ValueError(f"{nombre}: '{clave}' debe ser true o false.")
        elif tipo is int:
            if not isinstance(valor, int) or isinstance(valor, bool):
                raise ValueError(f"{nombre}: '{clave}' debe ser un número entero.")
        elif tipo is float:
            if not isinstance(valor, (int, float)) or isinstance(valor, bool):
                raise ValueError(f"{nombre}: '{clave}' debe ser un número.")
            valor = float(valor)
        elif not isinstance(valor, str):
            raise ValueError(f"{nombre}: '{clave}' debe ser un texto entre comillas.")
        if clave in OPCIONES and valor not in OPCIONES[clave]:
            raise ValueError(f"{nombre}: '{clave}' debe ser uno de: {', '.join(OPCIONES[clave])}.")
        config[clave] = valor
    return config


def cargar_config(ruta=None):
    """Aplica el archivo de configuración a DEFECTO. Devuelve la ruta leída, o None si no hay archivo.

    Sin `ruta` se usa CONFIG si existe. Con `ruta`, el archivo debe existir.
    """
    if ruta is None:
        if not CONFIG.is_file():
            return None
        ruta = CONFIG
    elif not Path(ruta).is_file():
        raise ValueError(f"no existe el archivo de configuración {ruta}.")
    config = leer_config(ruta)
    DEFECTO.update(config)
    print(f"Configuración leída de {Path(ruta).name}: " + (", ".join(f"{k} = {v}" for k, v in config.items()) or "sin parámetros"))
    return Path(ruta)


# =====================================================================
# 2. COMANDOS
# =====================================================================

def correr_aco(p):
    """Ejecuta el script ACO original con parámetros del orquestador.

    Usa el mismo intérprete de Python y la raíz del proyecto como directorio
    de trabajo. La semilla solo se modifica en el entorno del proceso hijo.
    Los comandos, módulos y archivos de salida del GA no se modifican.
    """
    for n in ["valor_hora", "consumo", "precio", *PARAMS_ACO]:
        if not math.isfinite(p[n]):
            raise ValueError(f"{n} debe ser un número finito.")
    if p["hormigas"] < 1:
        raise ValueError("hormigas debe ser al menos 1.")
    if not 0 <= p["semilla"] <= 2**32 - 1:
        raise ValueError("semilla para ACO debe estar entre 0 y 4294967295.")
    if not 0 <= p["evaporacion"] < 1:
        raise ValueError("evaporacion debe estar entre 0 (incluido) y 1 (excluido).")
    for n in ("beta_evaporacion", "p_mejor"):
        if not 0 <= p[n] <= 1:
            raise ValueError(f"{n} debe estar entre 0 y 1.")
    for n in ("intensificacion", "alfa", "beta"):
        if p[n] < 0:
            raise ValueError(f"{n} no puede ser negativo.")

    parte_2 = Path(__file__).resolve().parent
    script = parte_2 / "scripts" / "colonia_de_hormigas.py"
    if not script.is_file():
        raise ValueError(f"no existe el script ACO: {script}.")
    comando = [sys.executable, str(script)]
    for n in ["valor_hora", "consumo", "precio", *PARAMS_ACO, "version"]:
        comando.extend(["--" + n.replace("_", "-"), str(p[n])])
    entorno = os.environ.copy()
    entorno["ACO_RANDOM_SEED"] = str(p["semilla"])
    print("    ACO: máximo de 500 iteraciones; parada anticipada del optimizador.", flush=True)
    print(f"    Salidas: {parte_2.parent / 'outputs'}", flush=True)
    try:
        subprocess.run(comando, cwd=parte_2.parent, env=entorno, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"ACO terminó con código {e.returncode}; revisa el mensaje del script mostrado arriba.") from None


FUNCIONES_COMANDO["aco"] = correr_aco


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
    if p["vehiculo"]:
        print(f"    vehículo: {p['vehiculo']}")
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
        return tipo(texto)
    except ValueError:
        raise ValueError("se esperaba un número entero" if tipo is int else "se esperaba un número") from None


def mostrar(v):
    if v == "":
        return "(sin definir)"
    if isinstance(v, bool):
        return "sí" if v else "no"
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
            P[n] = DEFECTO[n] if texto == "" else convertir(n, texto)
        except ValueError as e:
            print(f"  Valor inválido ({e}); se conserva el actual.")


def menu():
    P = dict(DEFECTO)  # parámetros compartidos por todos los comandos
    comandos = list(TITULOS)

    def lanzar(c):
        try:
            correr(c, P)
        except KeyboardInterrupt:
            print("\n[Interrumpido]")
        except Exception as e:  # mostrar el error sin cerrar el menú
            print(f"\n[Error] {type(e).__name__}: {e}")

    print("\n=== Parte 2: recorrido de las 47 capitales (GA y ACO) ===")
    while True:
        print(f"\nConfig: valor de la hora={P['valor_hora']:g} EUR/h  consumo={P['consumo']:g} L/100 km  "
              f"precio={P['precio']:g} EUR/L  semilla={P['semilla']}  versión={P['version']}")
        for i, c in enumerate(comandos, 1):
            print(f"  {i}. {TITULOS[c]}")
        n_lim, n_cfg = len(comandos) + 1, len(comandos) + 2
        print(f"  {n_lim}. Borrar los archivos generados por el GA")
        print(f"  {n_cfg}. Configuración global (valor de la hora, consumo, precio, vehículo, semilla, versión)")
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
    # Primero se lee solo --config, porque el archivo fija los valores por defecto del resto.
    previo = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    previo.add_argument("--config", type=Path, default=None,
                        help=f"archivo de configuración JSON (por defecto, {CONFIG.name} junto a este archivo)")
    conocido, resto = previo.parse_known_args()
    try:
        cargar_config(conocido.config)
    except ValueError as e:
        sys.exit(f"[Error de configuración] {e}")

    if not resto:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
        return

    parser = argparse.ArgumentParser(description=__doc__, parents=[previo], allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", choices=[*TITULOS, "limpiar"],
                        help="comando a ejecutar, o 'limpiar' para borrar los archivos del GA")
    for n, (_, tipo, ayuda) in PARAMS.items():
        flag = "--" + n.replace("_", "-")
        if tipo is bool:
            parser.add_argument(flag, dest=n, action="store_true", default=DEFECTO[n], help=ayuda)
        elif n in OPCIONES:
            parser.add_argument(flag, dest=n, default=DEFECTO[n], choices=OPCIONES[n], help=ayuda)
        else:
            parser.add_argument(flag, dest=n, type=tipo, default=DEFECTO[n], help=ayuda)
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