---
layout: post
title: "2.1 a 2.4 Problema del vendedor viajero, modelo, gasolina y peajes"
date: 2026-10-06
categories: [parte-2]
---

## 2.1 Problema del vendedor viajero

Este es un problema de optimización combinatoria que consiste en determinar el recorrido de menor costo que permite visitar un conjunto de ciudades una única vez en un ciclo cerrado. El número de recorridos posibles aumenta factorialmente de acuerdo con la cantidad de ciudades, lo que dificulta encontrar una solución óptima a través de una búsqueda si el problema alcanza un tamaño considerable.

En este caso particular se aplica el problema a las 47 capitales de provincia de la España peninsular, excluyendo Palma, Las Palmas de Gran Canaria, Santa Cruz de Tenerife, Ceuta y Melilla; y se busca minimizar no la distancia, sino el costo económico total del desplazamiento, considerando tres componentes:

- **Costo del tiempo de viaje:** producto entre la duración del trayecto y el valor económico asignado a cada hora de trabajo del vendedor, que es un parámetro variable.
- **Costo de los peajes:** comprende los pagos asociados al uso de carreteras de peaje en los trayectos entre las distintas capitales.
- **Costo del combustible:** se calcula a partir de la distancia recorrida, el consumo homologado del vehículo seleccionado y el precio por litro del combustible utilizado. Para este trabajo se toma como referencia un Renault Clio TCe 115 automático de gasolina, con un consumo de 5,5 litros por cada 100 kilómetros.

La información de los desplazamientos se organiza en una matriz de costos de 47 × 47, en la cual cada elemento representa el costo de viajar directamente de una capital a otra, de acuerdo con los componentes anteriores. A partir de esta matriz se pretende encontrar una secuencia de visitas que minimice la suma de los costos de todos los desplazamientos, incluyendo el retorno a la ciudad inicial.

En términos matemáticos, si la matriz antes descrita es $$C$$, donde $$C_{ij}$$ representa el costo de viajar de la ciudad $$i$$ a la ciudad $$j$$, el objetivo del problema es hallar la lista $$\pi$$ que minimice:

$$\sum_{k=1}^{46} C_{\pi_k,\pi_{k+1}} + C_{\pi_{47},\pi_1}$$

## 2.2 Modelo seleccionado

Buscando trabajar sobre un consumo homologado de 5.5 litros por 100 km, se ha escogido el modelo Renault Clio TCe 115, que de acuerdo con Renault España (s. f.) cuenta con las siguientes especificaciones:

| Renault Clio TCe 115 | |
|---|---|
| Combustible | Gasolina |
| Potencia | 115 CV (85 kW nominales) |
| Transmisión | Automática / Manual |
| Homologación | WLTP, ciclo mixto |
| Consumo homologado | 5,3–5,5 L/100 km |

*Tabla 3. Especificaciones del modelo Renault Clio TCe 115. Fuente: elaboración propia.*

## 2.3 Precio del litro de gasolina en España

Se ha establecido un precio para el litro de gasolina de acuerdo con el precio del combustible Gasolina Euro-super 95, que estableció su valor en 1,849 EUR por litro el día 5 de octubre de 2026 (Comisión Europea, s. f.).

## 2.4 Comprobación de peajes

Los peajes presentes en todo recorrido, junto con sus precios, fueron obtenidos con la ayuda de la API HERE y la información acerca de ellos se encuentra en el documento `here_peaje.csv`, dentro del repositorio. Aquellos peajes que la API no pudo comprobar y sus estados se presentan a continuación:

| Vía | Estado | Fuente de confirmación | Fecha de consulta |
|---|---|---|---|
| AP-6 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-68 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-15 | vigente | Noticias de Navarra (2025) | 2026-10-10 |
| AP-9 | vigente | Europa Press (2025) | 2026-10-10 |
| R-4 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-66 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-51 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-71 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-61 | vigente | Europa Press (2025) | 2026-10-10 |
| R-3 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-46 | vigente | Europa Press (2025) | 2026-10-10 |
| M-12 | vigente | Europa Press (2025) | 2026-10-10 |
| R-2 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-7S | vigente | Europa Press (2025) | 2026-10-10 |
| AP-53 | vigente | Europa Press (2025) | 2026-10-10 |
| AP-8 | vigente | Diputación Foral de Bizkaia (2025) | 2026-10-10 |
| AP-1E | vigente | Bidegi (2026) | 2026-10-10 |
| C-32 | vigente | Departament de Territori (s. f.) | 2026-10-10 |
| AP-1/AP-8 | vigente | Bidegi (2026) | 2026-10-10 |
| C-33 | liberada | Infotrànsit (2021) | 2026-10-10 |
| AP-1B | liberada | Gasteiz Hoy (2018) | 2026-10-10 |
| AP-2 | liberada | Motorpasión (2019) | 2026-10-10 |
| AP-7N | liberada | Motorpasión (2021) | 2026-10-10 |
| AP-4 | liberada | Motorpasión (2019) | 2026-10-10 |

*Tabla 4. Comprobación de vías faltantes. Fuente: elaboración propia.*
