"""Descarga la matriz de distancias (m) y tiempos (s) por carretera entre las
47 capitales usando el servicio /table de OSRM (servidor demo público).

Uso:
    python scripts/obtener_osrm.py                 # descarga y guarda en data/
    python scripts/obtener_osrm.py --comparar      # descarga y compara con lo guardado, sin sobrescribir

Notas:
- Las coordenadas se leen de data/capitales.csv (columnas lat, lon).
- El servidor demo de OSRM (router.project-osrm.org) es gratuito y sin API key,
  pero su política de uso pide como máximo 1 petición por segundo. Este script
  hace UNA sola petición.
- OSRM usa datos de OpenStreetMap que cambian con el tiempo: si se vuelve a
  descargar otro día, pueden aparecer diferencias pequeñas. Por eso se guarda
  la fecha y hora de consulta en data/osrm_metadata.json.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
URL_BASE = "https://router.project-osrm.org/table/v1/driving/"


def leer_capitales():
    with open(DATA / "capitales.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    filas.sort(key=lambda r: int(r["id"]))
    return filas


def leer_matriz(ruta):
    with open(ruta, encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector)  # cabecera
        return [[int(x) for x in fila[1:]] for fila in lector]


def escribir_matriz(ruta, ids, matriz):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + ids)
        for i, fila in zip(ids, matriz):
            w.writerow([i] + fila)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparar", action="store_true",
                    help="no sobrescribe; compara la descarga con data/osrm_*.csv")
    args = ap.parse_args()

    caps = leer_capitales()
    coords = ";".join(f"{float(c['lon'])},{float(c['lat'])}" for c in caps)
    url = URL_BASE + coords + "?annotations=duration,distance&generate_hints=false"
    r = requests.get(url, timeout=60,
                     headers={"User-Agent": "proyecto-RNA-UNAL-TSP (uso academico)"})
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "Ok":
        sys.exit(f"OSRM respondió {j.get('code')}: {j.get('message')}")

    n = len(caps)
    if any(v is None for fila in j["distances"] for v in fila):
        sys.exit("Hay pares sin ruta (null). Revisar coordenadas.")
    D = [[round(v) for v in fila] for fila in j["distances"]]   # metros
    T = [[round(v) for v in fila] for fila in j["durations"]]   # segundos
    assert len(D) == n and all(len(f) == n for f in D)
    ids = [c["id"] for c in caps]

    if args.comparar:
        D0 = leer_matriz(DATA / "osrm_distancia_m.csv")
        T0 = leer_matriz(DATA / "osrm_tiempo_s.csv")
        dd = max(abs(D[i][k] - D0[i][k]) for i in range(n) for k in range(n))
        dt = max(abs(T[i][k] - T0[i][k]) for i in range(n) for k in range(n))
        print(f"Diferencia máxima con lo guardado: {dd} m, {dt} s")
        return

    escribir_matriz(DATA / "osrm_distancia_m.csv", ids, D)
    escribir_matriz(DATA / "osrm_tiempo_s.csv", ids, T)
    meta = {
        "fuente": "OSRM demo server (router.project-osrm.org), servicio table v1, perfil driving",
        "datos_mapa": "OpenStreetMap contributors (ODbL)",
        "fecha_consulta_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "unidades": {"distancia": "m (redondeado al metro)", "tiempo": "s (redondeado al segundo)"},
        "puntos_ajustados": [
            {"id": c["id"], "calle": s.get("name", ""), "dist_ajuste_m": round(s["distance"], 1),
             "lon": s["location"][0], "lat": s["location"][1]}
            for c, s in zip(caps, j["sources"])
        ],
    }
    with open(DATA / "osrm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("Guardado en data/: osrm_distancia_m.csv, osrm_tiempo_s.csv, osrm_metadata.json")


if __name__ == "__main__":
    main()