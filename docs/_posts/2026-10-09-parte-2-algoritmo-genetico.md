---
layout: post
title: "2.7.1 a 2.7.3 Algoritmo genético"
date: 2026-10-09
categories: [parte-2]
---

## 2.7 Algoritmo genético

Los algoritmos genéticos son metaheurísticas inspiradas en la evolución natural. Mantienen una población de soluciones candidatas que mejora de una generación a la siguiente mediante tres mecanismos: la selección de los mejores individuos, el cruce de pares de padres para producir hijos y la mutación aleatoria de algunos hijos (Eiben y Smith, 2015).

En el problema del vendedor viajero cada individuo es un recorrido completo. Esto exige operadores diseñados para permutaciones, porque los operadores clásicos para cadenas binarias producirían recorridos con ciudades repetidas u omitidas (Larrañaga et al., 1999). Al igual que la colonia de hormigas, el algoritmo genético no garantiza encontrar el óptimo global.

## 2.7.1 Representación y función de aptitud

Cada individuo es una permutación $$\pi = (\pi_1, \dots, \pi_{47})$$ de las 47 capitales, que representa el recorrido cerrado que las visita en ese orden y regresa a la primera. Como la matriz de costos es asimétrica, un recorrido y su inverso son individuos distintos. El costo de un individuo es el definido en la sección 2.1:

$$L(\pi) = \sum_{k=1}^{46} C_{\pi_k,\pi_{k+1}} + C_{\pi_{47},\pi_1}$$

La librería utilizada maximiza, así que la aptitud es el costo con signo contrario, $$f(\pi) = -L(\pi)$$. La población inicial está formada por $$N$$ permutaciones generadas al azar.

## 2.7.2 Ecuaciones de actualización

En cada generación se seleccionan padres, se cruzan, se mutan los hijos y se forma la nueva población. Los pasos son los siguientes.

**A. Selección por torneo.** Cada uno de los $$P$$ padres se elige con un torneo: se sortean $$k$$ individuos de la población, con reemplazo, y gana el de mayor aptitud:

$$g = \arg\max_{i \in T} f(\pi_i)$$

donde $$T$$ es el conjunto de los $$k$$ individuos sorteados. El tamaño del torneo regula la presión de selección: con $$k = 1$$ la elección es completamente al azar, y cuanto mayor es $$k$$, más se favorece a los mejores.

**B. Cruce.** Los padres se emparejan de forma consecutiva y circular: el hijo $$m$$ se genera con los padres $$m$$ y $$m+1$$. En ambos operadores se eligen al azar dos posiciones de corte $$i < j$$ y el hijo $$h$$ conserva el tramo del primer padre $$a$$:

$$h_s = a_s, \qquad s = i, \dots, j$$

Los operadores difieren en cómo completan el resto con el segundo padre $$b$$:

- **Cruce de orden (OX)**, propuesto por Davis (1985, citado en Larrañaga et al., 1999). Se recorre el segundo padre empezando después del segundo corte, de forma circular, y sus ciudades se colocan en las posiciones libres, también empezando después del segundo corte y omitiendo las que ya están en el tramo copiado. Conserva un tramo contiguo del primer padre y el orden relativo de las demás ciudades en el segundo.
- **Cruce parcialmente mapeado (PMX)**, propuesto por Goldberg y Lingle (1985, citado en Larrañaga et al., 1999). Cada ciudad del tramo del segundo padre que no quedó copiada se ubica siguiendo la correspondencia entre los dos tramos, posición a posición, hasta encontrar una posición libre. Las demás ciudades se copian del segundo padre en su misma posición. Conserva más posiciones absolutas que OX.

La Tabla 8 muestra el resultado de ambos operadores para un mismo par de padres de nueve ciudades, con cortes en las posiciones 4 y 7 (en negrita, el tramo entre los cortes).

| | Recorrido |
|---|---|
| Padre 1 | 1 2 3 **4 5 6 7** 8 9 |
| Padre 2 | 9 3 7 **8 2 6 5** 1 4 |
| Hijo con OX | 3 8 2 **4 5 6 7** 1 9 |
| Hijo con PMX | 9 3 2 **4 5 6 7** 1 8 |

*Tabla 8. Ejemplo de los cruces OX y PMX; en negrita, el tramo entre los cortes. Fuente: elaboración propia.*

**C. Mutación.** Cada hijo muta con probabilidad $$p_m$$. Si muta, recibe una sola modificación, con dos posiciones $$i < j$$ elegidas al azar:

$$\pi' = \begin{cases} M(\pi), & \text{si } u < p_m \\ \pi, & \text{en otro caso} \end{cases} \qquad u \sim U(0,1)$$

Se implementaron cuatro operadores $$M$$:

- **Inversión.** Invierte el tramo entre las dos posiciones:

  $$\pi' = (\pi_1, \dots, \pi_{i-1}, \pi_j, \pi_{j-1}, \dots, \pi_i, \pi_{j+1}, \dots, \pi_{47})$$

  Cambia las dos conexiones de los extremos del tramo. El interior se recorre en sentido contrario, pero su costo casi no varía porque la asimetría media de la matriz es de 1,65 %.
- **Inserción.** Saca la ciudad de la posición $$i$$ y la inserta en la posición $$j$$; las ciudades intermedias se desplazan un lugar sin cambiar de orden. Cambia tres conexiones:

  $$\pi' = (\pi_1, \dots, \pi_{i-1}, \pi_{i+1}, \dots, \pi_j, \pi_i, \pi_{j+1}, \dots, \pi_{47})$$

  En la implementación la ciudad también puede moverse hacia atrás, de la posición $$j$$ a la $$i$$.
- **Intercambio (swap).** Intercambia las ciudades de las dos posiciones, $$\pi'_i = \pi_j$$ y $$\pi'_j = \pi_i$$. Cambia hasta cuatro conexiones.
- **Mixta.** Aplica la inversión o la inserción, cada una con probabilidad 1/2.

**D. Reemplazo con elitismo.** La nueva población se forma con los $$E$$ mejores individuos de la población actual, que pasan sin cambios, y con $$N - E$$ hijos:

$$\text{Población}_{t+1} = \text{Élite}_t \cup \text{Hijos}_t, \qquad |\text{Élite}_t| = E, \quad |\text{Hijos}_t| = N - E$$

Con $$E \geq 1$$ el mejor costo de la población nunca empeora de una generación a la siguiente. El algoritmo se detiene al completar las generaciones y devuelve el mejor recorrido encontrado.

## 2.7.3 Implementación del algoritmo genético

El algoritmo se implementó en Python con la librería PyGAD, versión 3.8.1 (Gad, 2024), que aporta el ciclo de generaciones, la selección por torneo y el elitismo. Los cruces y las mutaciones se escribieron aparte, en el archivo `operadores_ga.py`, por dos razones. Primero, los cruces que trae la librería están pensados para cadenas de valores independientes y repiten ciudades. Segundo, sus mutaciones para permutaciones, cuando se les indica una probabilidad, la aplican a cada posición por separado y operan sobre posiciones dispersas, por lo que no equivalen a los operadores descritos en la sección 2.7.2.

La Tabla 9 muestra los parámetros utilizados. Todos pueden cambiarse desde el menú o la línea de comandos del orquestador (`Parte_2/orquestador.py`) o desde el archivo de configuración, sin editar el código.

| Parámetro | Símbolo | Valor |
|---|---|---|
| Tamaño de la población | $$N$$ | 100 |
| Número de generaciones | $$G$$ | 1500 |
| Padres seleccionados por generación | $$P$$ | 50 |
| Individuos de élite | $$E$$ | 2 |
| Tamaño del torneo | $$k$$ | 3 |
| Operador de cruce | | OX |
| Operador de mutación | $$M$$ | Mixta |
| Probabilidad de mutación | $$p_m$$ | 0,3 |
| Semilla aleatoria | | 0 |

*Tabla 9. Parámetros del algoritmo genético. Fuente: elaboración propia.*

Para contar las evaluaciones, se entiende por una evaluación el cálculo del costo de un recorrido completo. El programa cuenta las llamadas reales a la función de aptitud. La librería no vuelve a evaluar a los individuos de élite ni a los hijos idénticos a uno de ellos, porque ya conoce su aptitud, así que el número real es menor o igual que el nominal:

$$N + G\,(N - E) = 100 + 1500 \cdot 98 = 147\,100$$

Los resultados son reproducibles: la población inicial y todas las decisiones aleatorias dependen de la semilla, de modo que la misma semilla con los mismos parámetros produce el mismo recorrido. Una corrida tarda unos 10 segundos en el equipo de desarrollo, y unos 25 segundos si además se genera la animación.
