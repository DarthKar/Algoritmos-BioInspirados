# Datos del TSP: 47 capitales de provincia de la España peninsular

Carpeta pensada para copiarse tal cual dentro del repositorio del Trabajo 1 (Parte 2).
Todo lo que está en `data/` se puede usar sin internet.

## Archivos de `data/`

| Archivo | Contenido | Fuente | Consulta |
|---|---|---|---|
| `capitales.csv` | id 0 a 46, nombre, nombre y código INE, geonameid, lat, lon | INE tabla 67991; GeoNames | 2026-10-10 |
| `osrm_distancia_m.csv` | Matriz dirigida 47x47 de distancia por carretera (m) | OSRM, datos OpenStreetMap | 2026-10-10 06:07 UTC |
| `osrm_tiempo_s.csv` | Matriz dirigida 47x47 de tiempo sin tráfico (s) | OSRM | 2026-10-10 06:07 UTC |
| `osrm_metadata.json` | URL exacta de la consulta, método de verificación, punto de ajuste de cada ciudad | OSRM | 2026-10-10 |
| `vias_peaje_osrm.csv` | Por arco: vías candidatas a peaje que recorre la ruta OSRM y km | OSRM | 2026-10-10 |
| `estado_vias_peaje_2026.csv` | Estado 2026 de cada vía (vigente, liberada, verificar) con fuente | Ver columna fuente | 2026-10-10 |
| `combustible.csv` | Precio medio en España con impuestos | Comisión Europea, Weekly Oil Bulletin (datos al 2026-10-05) | 2026-10-10 |

Fila = origen, columna = destino. `[i, j]` puede ser distinto de `[j, i]`.

## Lo que falta

1. **Peajes en EUR**: correr `scripts/obtener_peajes_here.py` con una API key de HERE (ver el encabezado del script).
   No usar 0 EUR: 1.046 de los 2.162 arcos pasan por alguna vía que cobra peaje en 2026 (indicativo, ver `vias_peaje_osrm.csv`).
2. **Vehículo**: modelo, consumo homologado WLTP y combustible, con ficha citada (IDAE: https://coches.idae.es/base-datos).

## Scripts

```bash
pip install requests numpy
python scripts/costos.py --valor-hora 15 --consumo 5.5 --precio 1.849   # matriz de costos y prueba
python scripts/obtener_osrm.py --comparar                                # vuelve a consultar OSRM y compara
python scripts/obtener_peajes_here.py --modo peaje --limite 3            # prueba con 3 arcos
```

`obtener_osrm.py` y `obtener_peajes_here.py` no se pudieron probar contra los servidores reales desde el
entorno donde se prepararon: prueben primero con `--comparar` y `--limite 3` y revisen la salida a mano.

## Advertencias

- AP-68: la concesión termina el 10-11-2026 (Real Decreto 730/2026). Peajes consultados antes quedan viejos después.
- Autopistas SEITT (R-2, R-3, R-4, R-5, M-12, AP-36, AP-41): gratis de 0:00 a 6:00 en 2026.
- La ruta más rápida no mira el precio del peaje. Con peajes caros conviene considerar también la ruta sin peaje.
