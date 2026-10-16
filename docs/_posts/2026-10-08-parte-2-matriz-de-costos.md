---
layout: post
title: "2.6 Construcción de la matriz de costos"
date: 2026-10-08
categories: [parte-2]
---

## 2.6 Construcción de la matriz de costos

El costo de viajar de la capital $$i$$ a la capital $$j$$ es la suma de los tres componentes descritos en la sección 2.1:

$$C_{ij} = v_h \cdot H_{ij} + P_{ij} + \frac{D_{ij}}{100} \cdot q \cdot p_f$$

donde:

- $$v_h$$: valor de la hora del vendedor, en EUR/h. Es el parámetro que se estudia en la sección 2.7.5.
- $$H_{ij}$$: duración del trayecto, en horas.
- $$P_{ij}$$: peajes del trayecto, en EUR.
- $$D_{ij}$$: distancia por carretera, en km.
- $$q$$: consumo del vehículo, 5,5 L/100 km (sección 2.2).
- $$p_f$$: precio del combustible, 1,849 EUR/L (sección 2.3).

La diagonal de la matriz se fija en cero. El cálculo está en el archivo `costos.py` del repositorio.

La distancia, la duración y el peaje de cada trayecto provienen de una misma consulta a HERE Routing API v8 (HERE Technologies, s. f.), hecha el 10 de octubre de 2026 para los 47 × 46 = 2162 pares ordenados de capitales. Se pidió la ruta más rápida en automóvil, sin tráfico. Las respuestas están guardadas en el repositorio (archivo `here_peaje.csv` y carpeta `here_cache`), de modo que la matriz se reconstruye sin conexión a internet. La Tabla 7 resume la fuente de cada dato.

| Dato | Fuente | Fecha de consulta |
|---|---|---|
| Distancia por carretera | HERE Routing API v8 (HERE Technologies, s. f.) | 2026-10-10 |
| Duración del trayecto | HERE Routing API v8 (HERE Technologies, s. f.) | 2026-10-10 |
| Peajes | HERE Routing API v8 (HERE Technologies, s. f.) | 2026-10-10 |
| Consumo del vehículo | Renault España (s. f.), sección 2.2 | 2026-10-12 |
| Precio del combustible | Comisión Europea (s. f.), sección 2.3; datos del 2026-10-05 | 2026-10-10 |

*Tabla 7. Fuentes de los datos de la matriz de costos. Fuente: elaboración propia.*

En una primera versión, la distancia y la duración se tomaban de OSRM (Luxen y Vetter, 2011) y solo el peaje de HERE. Al comparar ambas fuentes se encontró que no siempre eligen la misma carretera: en 53 de los 2162 trayectos la distancia difiere en más de 10 %, y los tiempos de HERE son en promedio 19,1 % menores que los de OSRM. Por ejemplo, entre Pontevedra y Ourense la ruta de OSRM mide 95 km y la de HERE 119 km, con 5,05 EUR de peaje. Mezclar las fuentes equivalía a cobrar el peaje de una ruta con el tiempo y la distancia de otra. Por eso se decidió tomar los tres datos de HERE; las matrices de OSRM se conservan en el repositorio sólo como verificación independiente.
