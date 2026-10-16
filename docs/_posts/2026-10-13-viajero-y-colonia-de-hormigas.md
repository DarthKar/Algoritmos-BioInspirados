---
layout: post
title: "El viajero por España: costos y colonia de hormigas"
date: 2026-10-13
categories: [parte-2]
---

## El problema

Visitar una sola vez las 47 capitales de provincia de la España peninsular (sin Palma, Las Palmas, Santa Cruz de Tenerife, Ceuta ni Melilla) y volver al inicio, minimizando el **costo económico**, no la distancia. Si $$C$$ es la matriz de costos de 47 × 47, buscamos la permutación $$\pi$$ que minimice

$$\sum_{k=1}^{46} C_{\pi_k,\pi_{k+1}} + C_{\pi_{47},\pi_1}$$

## La matriz de costos

$$C_{ij} = v_h \cdot H_{ij} + P_{ij} + \frac{D_{ij}}{100}\, q\, p_f$$

- $$v_h$$: valor de la hora del vendedor (EUR/h), el parámetro que estudiamos.
- $$H_{ij}$$, $$D_{ij}$$, $$P_{ij}$$: horas, kilómetros y peajes del trayecto.
- $$q = 5.5$$ L/100 km: consumo homologado de un Renault Clio TCe 115.
- $$p_f = 1.849$$ EUR/L: gasolina Euro-super 95 el 5 de octubre de 2026 (Boletín Petrolero de la Comisión Europea).

Duración, distancia y peaje salen de una misma consulta a la **HERE Routing API v8** (10 de octubre de 2026, 2162 pares ordenados). Al principio usábamos OSRM para distancia y tiempo, pero en 53 de los 2162 trayectos la distancia difería más de 10 % y los tiempos de HERE eran en promedio 19,1 % menores. Por ejemplo, Pontevedra–Ourense mide 95 km en OSRM y 119 km en HERE, con 5,05 EUR de peaje. Mezclar fuentes era cobrar el peaje de una ruta con el tiempo de otra, así que tomamos los tres datos de HERE.

## Colonia de hormigas (ACO)

Cada hormiga arma un recorrido eligiendo la siguiente ciudad según feromonas $$\tau_{ij}$$ y la heurística $$\eta_{ij} = 1/C_{ij}$$:

$$P_{ij}(t) = \frac{[\tau_{ij}(t)]^{\alpha}[\eta_{ij}]^{\beta}}{\sum_{k \in N_i}[\tau_{ik}(t)]^{\alpha}[\eta_{ik}]^{\beta}}$$

Con probabilidad $$q_0$$ elige directamente la mejor opción. Después de cada iteración, las feromonas se evaporan ($$\tau' = (1-\rho)\tau$$, con $$\rho = 0.05$$) y se refuerzan en el mejor recorrido. Partimos de la implementación de Jhon Berroa y le añadimos un historial para animar la evolución de la mejor ruta en GIF y MP4.

## Experimentos

Variamos el valor de la hora (5 a 25 EUR/h) y el número de hormigas (40 a 200). El mejor costo en cada caso:

| Valor de la hora (EUR) | Mejor costo (EUR) | Hormigas |
|---|---|---|
| 5 | 1152.84 | 80 |
| 10 | 1498.52 | 200 |
| 15 | 1950.38 | 200 |
| 20 | 2361.16 | 160 |
| 25 | 2750.34 | 160 |

Los recorridos no son variaciones de uno solo, pero hay patrones. Estas parejas quedaron siempre adyacentes: Badajoz–Cáceres, Bilbao–Vitoria-Gasteiz, Cádiz–Sevilla, Córdoba–Huelva, Girona–Barcelona, Granada–Jaén, Guadalajara–Madrid, Ourense–Pontevedra, Pamplona–San Sebastián, Santander–Bilbao, Toledo–Ciudad Real y Ávila–Segovia. Es probable que la solución óptima comparta esos rasgos.
