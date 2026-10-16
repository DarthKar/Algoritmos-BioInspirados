---
layout: post
title: "Algoritmo genético: operadores y valor de la hora"
date: 2026-10-14
categories: [parte-2]
---

## Diseño

Cada individuo es una permutación de las 47 capitales y su aptitud es $$-L(\pi)$$, el costo del recorrido con signo cambiado (PyGAD 3.8.1 maximiza). Como la matriz es asimétrica, un recorrido y su inverso son individuos distintos. En cada generación:

1. **Selección por torneo** de tamaño $$k = 3$$.
2. **Cruce** OX (de orden) o PMX (parcialmente mapeado), escritos por nosotros en `operadores_ga.py`.
3. **Mutación** con probabilidad $$p_m = 0.3$$: inversión, inserción, intercambio o mixta (inversión o inserción con probabilidad 1/2).
4. **Elitismo**: los $$E = 2$$ mejores pasan intactos.

Parámetros base: población 100, 1500 generaciones, 50 padres por generación, semilla 0.

## Comparación de operadores

Ocho combinaciones, 30 corridas cada una (semillas 0 a 29), a 15 EUR/h. Contamos como éxito costos de hasta 1707 EUR.

| Cruce + mutación | Media (EUR) | Mejor (EUR) | Éxito |
|---|---|---|---|
| OX + inserción | 1843.09 | 1723.30 | 0 % |
| OX + inversión | 1735.16 | 1677.08 | 23 % |
| **OX + mixta** | **1722.50** | **1673.53** | **50 %** |
| OX + swap | 1859.81 | 1709.67 | 0 % |
| PMX + inserción | 1923.22 | 1697.58 | 3 % |
| PMX + inversión | 1755.24 | 1688.07 | 13 % |
| PMX + mixta | 1727.96 | 1673.53 | 33 % |
| PMX + swap | 2304.56 | 1916.46 | 0 % |

OX con mutación mixta fue la mejor combinación; el intercambio fue la peor mutación, sobre todo con PMX.

## ¿Cambia la ruta óptima con el valor de la hora?

Sí. Como horas, peajes y combustible no dependen de $$v_h$$, el costo de un recorrido es una recta: $$L(\pi; v_h) = v_h\,H(\pi) + F(\pi)$$. Corrimos el genético 20 veces por cada valor de la hora entre 0 y 60 EUR/h (pasos de 5), reunimos los 230 recorridos distintos y, para cada valor, tomamos el más barato. El cambio de la ruta $$\pi_a$$ a la más rápida $$\pi_b$$ ocurre donde las rectas se cruzan:

$$v_h^{*} = \frac{F(\pi_b) - F(\pi_a)}{H(\pi_a) - H(\pi_b)}$$

La mejor ruta conocida cambia dos veces, en **14,57** y en **22,47 EUR/h**:

| Ruta | Valor de la hora (EUR/h) | Horas | Distancia (km) | Peajes (EUR) | Combustible (EUR) |
|---|---|---|---|---|---|
| R1 | 0 a 14,57 | 65,63 | 6313,7 | 47,26 | 642,07 |
| R2 | 14,57 a 22,47 | 65,18 | 6292,4 | 55,92 | 639,90 |
| R3 | 22,47 a 60 | 64,81 | 6174,0 | 76,32 | 627,86 |

Cuanto más vale la hora, más peajes conviene pagar (de 47,26 a 76,32 EUR) para ahorrar tiempo (de 65,63 a 64,81 horas). El estudio se reproduce con:

```
python Parte_2/orquestador.py hora --version informe
```
