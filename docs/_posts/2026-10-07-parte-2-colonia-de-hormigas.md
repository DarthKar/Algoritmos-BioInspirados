---
layout: post
title: "2.5 Algoritmo de colonia de hormigas"
date: 2026-10-07
categories: [parte-2]
---

## 2.5 Algoritmo de colonia de hormigas

La optimización por colonia de hormigas (Ant Colony Optimization, ACO) es una técnica metaheurística inspirada en el comportamiento colectivo de las hormigas a la hora de buscar alimento, concretamente en el hecho de que desprenden feromonas sobre los caminos que recorren, permitiendo que otros miembros de su grupo los sigan. Con el tiempo, los caminos utilizados con mayor frecuencia pueden acumular mayores concentraciones de feromonas, favoreciendo su selección por parte de la colonia.

En el presente trabajo, el algoritmo opera sobre una matriz de costos de 47 × 47, cuyos elementos representan los costos económicos de desplazamiento entre las capitales de provincia de la España peninsular. Cada hormiga comienza en una ciudad seleccionada aleatoriamente y construye un recorrido visitando todas las capitales exactamente una vez, sin repetir ninguna hasta completar el circuito y regresar a la ciudad de partida.

La selección de la siguiente ciudad depende de dos factores: la concentración de feromonas asociada a la conexión y una información heurística definida como el inverso del costo de desplazamiento. De esta manera, las conexiones con mayor concentración de feromonas o menor costo pueden recibir una mayor preferencia durante la construcción de soluciones.

Una vez todas las hormigas han completado sus recorridos, se calcula el costo total de cada uno y se identifica la mejor solución de la iteración. Posteriormente se actualizan las feromonas mediante dos operaciones: la evaporación, que reduce su concentración sobre las conexiones, y la intensificación, que incrementa su valor sobre las conexiones pertenecientes al mejor recorrido de la iteración. Este procedimiento permite reforzar las soluciones prometedoras y orientar progresivamente la exploración del espacio de búsqueda.

El proceso se repite durante un número determinado de iteraciones o hasta alcanzar un criterio de parada anticipada. Durante la ejecución se conserva la mejor solución global encontrada, permitiendo obtener el recorrido de menor costo identificado por el algoritmo. Debido a su naturaleza metaheurística, ACO no garantiza la obtención de la solución óptima global; sin embargo, permite explorar eficientemente un espacio de búsqueda de gran tamaño y obtener soluciones de buena calidad en tiempos computacionales razonables.

## 2.5.1 Ecuaciones de actualización

**A. Información heurística.** La información heurística asociada al desplazamiento desde la ciudad $$i$$ hasta la ciudad $$j$$ se calcula como:

$$\eta_{ij} = \frac{1}{C_{ij}}$$

donde $$\eta_{ij}$$ es la conveniencia heurística de desplazarse de $$i$$ a $$j$$ y $$C_{ij}$$ es el costo económico de desplazamiento entre ambas ciudades.

**B. Probabilidad de transición.** Para determinar la siguiente ciudad, el algoritmo calcula el peso de cada conexión:

$$w_{ij}(t) = [\tau_{ij}(t)]^{\alpha}\,[\eta_{ij}]^{\beta}$$

La probabilidad de seleccionar una ciudad $$j$$ desde la ciudad actual $$i$$ es:

$$P_{ij}(t) = \frac{w_{ij}(t)}{\sum_{k \in N_i} [\tau_{ik}(t)]^{\alpha}\,[\eta_{ik}]^{\beta}}$$

donde:

- $$P_{ij}(t)$$: probabilidad de desplazarse de $$i$$ a $$j$$ en la iteración $$t$$.
- $$\tau_{ij}(t)$$: concentración de feromonas de la conexión.
- $$\eta_{ij}$$: información heurística.
- $$\alpha$$: importancia de las feromonas.
- $$\beta$$: importancia de la heurística.
- $$N_i$$: conjunto de ciudades que la hormiga todavía no ha visitado.

Para el caso de este algoritmo particular existe una probabilidad $$q_0$$ de elegir directamente la opción con mayor peso:

$$j^{*} = \arg\max_{j \in N_i} w_{ij}(t) \quad \text{si } r \sim U(0,1) < q_0$$

**C. Función de evaluación.** Cada hormiga construye un recorrido cerrado. Su costo total se determina mediante:

$$L_k = \sum_{s=1}^{47} C_{\pi_s^{(k)},\,\pi_{s+1}^{(k)}} \qquad \text{con la condición de que } \pi_1^{(k)} = \pi_{48}^{(k)}$$

donde $$L_k$$ es el costo total de la ruta construida por la hormiga $$k$$, $$\pi_s^{(k)}$$ es la ciudad visitada en la posición $$s$$ por dicha hormiga y $$C_{ij}$$ es el costo de desplazamiento entre dos ciudades. Una vez evaluadas todas las rutas, se selecciona la mejor de la iteración:

$$k^{*} = \arg\min_k\, (L_k)$$

De esta manera, se identifica el recorrido de menor costo entre las soluciones construidas por las hormigas durante esa iteración.

**D. Evaporación de feromonas.** Tras evaluar las soluciones, la concentración de feromonas disminuye según:

$$\tau'_{ij}(t) = (1 - \rho)\,\tau_{ij}(t)$$

donde $$\rho$$ es la tasa de evaporación, comprendida entre 0 y 1. Para el algoritmo implementado, $$\rho = 0.05$$.

## 2.5.2 Implementación del algoritmo de colonia de hormigas

Para resolver el problema del vendedor viajero mediante optimización por colonia de hormigas se utilizó como base la implementación en Python publicada por Jhon Berroa (s. f.) en el repositorio Ant-Colony-Optimization de GitHub.

La implementación original fue adaptada para incorporar un registro histórico de las mejores soluciones encontradas durante las iteraciones del algoritmo. Este registro permite representar gráficamente la evolución del recorrido y generar animaciones en formatos GIF y MP4. El algoritmo se integró con la matriz de costos del problema, construida a partir de los tiempos de desplazamiento, los costos de peajes y el consumo de combustible del vehículo seleccionado.

## 2.5.3 Experimentación con el algoritmo de colonia de hormigas

Se realizaron varias iteraciones del programa alterando el precio de la hora del vendedor y la cantidad de hormigas (agentes) involucradas en el algoritmo, para descubrir qué combinación de parámetros retorna un menor costo para el recorrido y cómo cambia este último en función de dichas variables. Los resultados se reportan en la tabla a continuación:

| Valor de la hora del vendedor (EUR) | Cantidad de hormigas | Costo total del viaje (EUR) |
|---|---|---|
| 5 | 40 | 1184.74 |
| 5 | 80 | 1152.84 |
| 5 | 120 | 1212.44 |
| 5 | 160 | 1265.52 |
| 5 | 200 | 1280.56 |
| 10 | 40 | 1799.59 |
| 10 | 80 | 1725.03 |
| 10 | 120 | 1533.95 |
| 10 | 160 | 1592.57 |
| 10 | 200 | 1498.52 |
| 15 | 40 | 2123.67 |
| 15 | 80 | 2072.80 |
| 15 | 120 | 2139.12 |
| 15 | 160 | 2056.73 |
| 15 | 200 | 1950.38 |
| 20 | 40 | 2568.51 |
| 20 | 80 | 2522.41 |
| 20 | 120 | 2502.64 |
| 20 | 160 | 2361.16 |
| 20 | 200 | 2407.22 |
| 25 | 40 | 3006.27 |
| 25 | 80 | 2894.41 |
| 25 | 120 | 2910.79 |
| 25 | 160 | 2750.34 |
| 25 | 200 | 2789.52 |

*Tabla 5. Resultados de los experimentos del algoritmo de colonia de hormigas. Fuente: elaboración propia.*

Se puede observar cierta correlación entre el valor de la hora del viajero y la cantidad de hormigas necesarias para optimizar el costo total del viaje; sin embargo, es difícil establecer una relación de causalidad, puesto que lo más probable es que las transformaciones en los recorridos óptimos dado el valor de la hora se deban al valor de los peajes. Dichos recorridos óptimos son los siguientes:

| Valor de la hora (EUR) | Recorrido |
|---|---|
| 5 | Murcia, Almería, Málaga, Granada, Jaén, Córdoba, Huelva, Cádiz, Sevilla, Badajoz, Cáceres, Salamanca, Valladolid, Zamora, León, Ourense, Pontevedra, A Coruña, Lugo, Oviedo, Santander, Bilbao, Vitoria-Gasteiz, Logroño, Pamplona, San Sebastián, Burgos, Palencia, Ávila, Segovia, Soria, Zaragoza, Castellón de la Plana, Tarragona, Girona, Barcelona, Lleida, Huesca, Teruel, Cuenca, Guadalajara, Madrid, Toledo, Ciudad Real, Albacete, Alicante, Valencia, Murcia |
| 10 | Vitoria-Gasteiz, Bilbao, Santander, Burgos, Guadalajara, Madrid, Toledo, Ciudad Real, Jaén, Granada, Almería, Málaga, Córdoba, Huelva, Sevilla, Cádiz, Badajoz, Cáceres, Salamanca, Zamora, Valladolid, Segovia, Ávila, Lugo, Ourense, Pontevedra, A Coruña, Oviedo, León, Palencia, Logroño, Soria, Teruel, Cuenca, Albacete, Murcia, Alicante, Valencia, Castellón de la Plana, Tarragona, Barcelona, Girona, Lleida, Zaragoza, Huesca, Pamplona, San Sebastián, Vitoria-Gasteiz |
| 15 | Burgos, León, Palencia, Valladolid, Ávila, Segovia, Cuenca, Castellón de la Plana, Tarragona, Lleida, Barcelona, Girona, Albacete, Murcia, Alicante, Valencia, Teruel, Huesca, Zaragoza, Soria, Guadalajara, Madrid, Toledo, Ciudad Real, Jaén, Granada, Almería, Málaga, Córdoba, Huelva, Sevilla, Cádiz, Cáceres, Badajoz, Salamanca, Zamora, Ourense, Pontevedra, Lugo, A Coruña, Oviedo, Santander, Bilbao, Vitoria-Gasteiz, Pamplona, San Sebastián, Logroño, Burgos |
| 20 | Zamora, Palencia, Valladolid, Burgos, Santander, Bilbao, Vitoria-Gasteiz, Logroño, Pamplona, San Sebastián, Oviedo, Lugo, A Coruña, Pontevedra, Ourense, León, Soria, Cuenca, Toledo, Ciudad Real, Cáceres, Badajoz, Sevilla, Cádiz, Huelva, Córdoba, Málaga, Jaén, Granada, Almería, Albacete, Murcia, Alicante, Valencia, Castellón de la Plana, Teruel, Barcelona, Girona, Tarragona, Lleida, Huesca, Zaragoza, Guadalajara, Madrid, Ávila, Segovia, Salamanca, Zamora |
| 25 | Ávila, Salamanca, Zamora, León, Oviedo, Lugo, A Coruña, Pontevedra, Ourense, Palencia, Valladolid, Santander, Bilbao, Vitoria-Gasteiz, Pamplona, San Sebastián, Logroño, Burgos, Soria, Zaragoza, Huesca, Lleida, Tarragona, Barcelona, Girona, Teruel, Valencia, Castellón de la Plana, Alicante, Murcia, Málaga, Córdoba, Huelva, Cáceres, Badajoz, Sevilla, Cádiz, Jaén, Granada, Almería, Albacete, Ciudad Real, Toledo, Cuenca, Guadalajara, Madrid, Segovia, Ávila |

*Tabla 6. Recorridos óptimos del algoritmo de colonia de hormigas. Fuente: elaboración propia.*

No se puede decir que los recorridos sean variaciones de uno mismo; sin embargo, sí existen ciertos patrones que se repiten de uno a otro. Por ejemplo, las siguientes parejas de capitales siempre permanecen adyacentes, aunque el sentido en que el recorrido las incluya puede variar:

- Badajoz y Cáceres
- Bilbao y Vitoria-Gasteiz
- Cádiz y Sevilla
- Córdoba y Huelva
- Girona y Barcelona
- Granada y Jaén
- Guadalajara y Madrid
- Ourense y Pontevedra
- Pamplona y San Sebastián
- Santander y Bilbao
- Toledo y Ciudad Real
- Ávila y Segovia

Además de ello, parejas de ciudades como las de Alicante y Valencia, Almería y Granada o Salamanca y Zamora siempre aparecen separadas por un máximo de una ciudad, por lo que se podría intuir que la solución óptima, si es que en ninguno de los casos es aquella que el algoritmo encontró, posee características como las mencionadas anteriormente; es decir, existen patrones de la solución óptima que se manifiestan en los resultados del algoritmo.
