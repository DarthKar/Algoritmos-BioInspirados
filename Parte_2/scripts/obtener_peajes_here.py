"""Consulta HERE Routing API v8 para cada arco dirigido (i -> j) y guarda
distancia, tiempo y costo de peajes en EUR.

IMPORTANTE (costos de HERE, plan Base, consultado 2026-10-10 en
https://developers.here.com/plans):
  - Routing (car): 30.000 transacciones/mes gratis.
  - Toll Cost: 2.500 transacciones/mes gratis. Pedir peajes cuenta como
    transacción adicional.
  47 x 46 = 2.162 arcos dirigidos caben en el cupo gratis de peajes UNA vez.
  Por eso cada respuesta se guarda en data/here_cache/ y nunca se vuelve a pedir.
  Prueben primero con --limite 3 y revisen la salida antes de correr todo.

Uso:
    set HERE_API_KEY=...            (Windows)   |  export HERE_API_KEY=... (Linux/Mac)
    python scripts/obtener_peajes_here.py --modo peaje --limite 3
    python scripts/obtener_peajes_here.py --modo peaje
    python scripts/obtener_peajes_here.py --modo sin_peaje      # ruta evitando autopistas de peaje

Salida: data/here_<modo>.csv con columnas
    i, j, origen, destino, dist_m, tiempo_s, peaje_eur, metodo_peaje, avisos
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import requests

load_dotenv()

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
URL = "https://router.hereapi.com/v8/routes"


def leer_capitales():
    with open(DATA / "capitales.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    filas.sort(key=lambda r: int(r["id"]))
    return filas


def parametros(origen, destino, modo, api_key):
    p = {
        "transportMode": "car",
        "routingMode": "fast",
        "origin": f"{float(origen['lat'])},{float(origen['lon'])}",
        "destination": f"{float(destino['lat'])},{float(destino['lon'])}",
        "departureTime": "any",          # sin tráfico: resultado reproducible
        "return": "summary",
        "apiKey": api_key,
    }
    if modo == "peaje":
        p["return"] = "summary,tolls"
        p["currency"] = "EUR"
        p["tolls[summaries]"] = "total"
    else:  # sin_peaje
        p["avoid[features]"] = "tollRoad"
    return p


def extraer(j, modo):
    """Devuelve (dist_m, tiempo_s, peaje_eur, metodo, avisos) de una respuesta de HERE."""
    if not j.get("routes"):
        return None, None, None, "sin_ruta", json.dumps(j.get("notices", []), ensure_ascii=False)
    secciones = j["routes"][0]["sections"]
    dist = sum(s["summary"]["length"] for s in secciones)
    tiempo = sum(s["summary"]["duration"] for s in secciones)
    avisos = [n.get("code", "") for s in secciones for n in s.get("notices", [])]
    avisos += [n.get("code", "") for n in j.get("notices", [])]
    if modo != "peaje":
        # Si HERE no pudo evitar el peaje, avisa con un código "violated...": ese arco NO es 0 EUR.
        if any("violated" in a.lower() for a in avisos):
            return dist, tiempo, None, "no_pudo_evitar_peaje", "|".join(avisos)
        return dist, tiempo, 0.0, "no_aplica", "|".join(avisos)

    # 1) Total oficial que calcula HERE (evita doble conteo entre sistemas de peaje)
    totales = [s["summary"].get("tolls", {}).get("total") for s in secciones]
    if all(t is not None for t in totales):
        if any(t.get("currency") != "EUR" for t in totales):
            avisos.append("moneda_no_EUR")
        return dist, tiempo, round(sum(t["value"] for t in totales), 2), "summary.tolls.total", "|".join(avisos)

    # 2) Respaldo: por cada peaje, la tarifa más baja entre sus 'fares'
    #    (un mismo peaje puede traer varias tarifas según medio de pago; sumarlas sería doble conteo)
    total = 0.0
    for s in secciones:
        for peaje in s.get("tolls", []):
            precios = []
            for fare in peaje.get("fares", []):
                precio = fare.get("convertedPrice") or fare.get("price")
                if precio and precio.get("currency") == "EUR":
                    precios.append(precio["value"])
            if precios:
                total += min(precios)
            elif peaje.get("fares"):
                avisos.append("peaje_sin_precio_EUR")
    return dist, tiempo, round(total, 2), "min_fares_por_peaje", "|".join(avisos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["peaje", "sin_peaje"], required=True)
    ap.add_argument("--limite", type=int, default=None, help="máximo de peticiones NUEVAS (para probar)")
    ap.add_argument("--pausa", type=float, default=0.5, help="segundos entre peticiones")
    args = ap.parse_args()

    api_key = os.environ.get("HERE_API_KEY")
    if not api_key:
        sys.exit("Falta la variable de entorno HERE_API_KEY")

    caps = leer_capitales()
    cache = DATA / "here_cache" / args.modo
    cache.mkdir(parents=True, exist_ok=True)
    pares = [(a, b) for a in caps for b in caps if a["id"] != b["id"]]
    nuevas = 0
    for a, b in pares:
        archivo = cache / f"{a['id']}_{b['id']}.json"
        if archivo.exists():
            continue
        if args.limite is not None and nuevas >= args.limite:
            break
        for intento in range(5):
            r = requests.get(URL, params=parametros(a, b, args.modo, api_key), timeout=60)
            if r.status_code == 429:          # demasiadas peticiones: esperar y reintentar
                time.sleep(2 ** intento)
                continue
            break
        if r.status_code != 200:
            sys.exit(f"Error HTTP {r.status_code} en {a['capital']} -> {b['capital']}: {r.text[:300]}")
        archivo.write_text(r.text, encoding="utf-8")
        nuevas += 1
        if nuevas <= 3:   # mostrar las primeras para verificar a mano
            print(a["capital"], "->", b["capital"], extraer(r.json(), args.modo))
        time.sleep(args.pausa)
    print(f"Peticiones nuevas: {nuevas}")

    salida = DATA / f"here_{args.modo}.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "origen", "destino", "dist_m", "tiempo_s", "peaje_eur", "metodo_peaje", "avisos"])
        faltan = 0
        for a, b in pares:
            archivo = cache / f"{a['id']}_{b['id']}.json"
            if not archivo.exists():
                faltan += 1
                continue
            res = extraer(json.loads(archivo.read_text(encoding="utf-8")), args.modo)
            w.writerow([a["id"], b["id"], a["capital"], b["capital"], *res])
    print(f"Escrito {salida} (faltan {faltan} arcos por consultar)")


if __name__ == "__main__":
    main()
