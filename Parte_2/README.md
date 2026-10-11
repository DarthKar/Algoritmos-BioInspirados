## Parte 2

## Créditos y código de terceros

La implementación del algoritmo de optimización por colonia de hormigas (ACO) utilizada en este proyecto está basada en el código desarrollado por el usuario **johnberroa** y publicado en GitHub:

- **Repositorio original:** [Ant-Colony-Optimization](https://github.com/johnberroa/Ant-Colony-Optimization)
- **Archivo utilizado:** `AntColonyOptimizer.py`

Para su integración en este proyecto académico se realizaron adaptaciones que permiten registrar la evolución de la mejor solución encontrada y utilizar esta información para generar visualizaciones animadas de las rutas mediante GIF y MP4.

El modelo de costos aplicado al problema de las 47 capitales peninsulares, la integración de los componentes y las visualizaciones se documentan como parte del desarrollo del presente trabajo.

# Datos del TSP: 47 capitales de provincia de la España peninsular

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
| `here_peaje.csv` | Información de peajes por arco dirigido de una pareja de ciudades | Obtenido de la API HERE mediante el módulo obtener_peajes_here | 2026-10-10 |
Fila = origen, columna = destino. `[i, j]` puede ser distinto de `[j, i]`.

## En caso de querer actualizar los datos de peaje

**Peajes en EUR**: correr `scripts/obtener_peajes_here.py` con una API key de HERE (ver el encabezado del script).

## Vehículo seleccionado

Renault Clio TCe 115, consumo homologado de 5,3–5,5 L/100 km (tomado como 5,5 L/100 km). https://www.renault.es/preguntas-frecuentes-vehiculos/clio/esta-renault-clio-disponible-con-una-caja-de-cambios-manual-o-automatica.html Revisado 2026-10-10

## Scripts

```bash
pip install requests numpy
python Parte_2/scripts/costos.py --valor-hora 15 --consumo 5.5 --precio 1.849   # matriz de costos y prueba
python Parte_2/scripts/obtener_osrm.py --comparar                                # vuelve a consultar OSRM y compara
python Parte_2/scripts/obtener_peajes_here.py --modo peaje --limite 3            # prueba con 3 arcos
python Parte_2/scripts/colonia_de_hormigas.py --valor-hora 15 --consumo 5.5 --precio 1.849 --hormigas 100 --version 0001
```
