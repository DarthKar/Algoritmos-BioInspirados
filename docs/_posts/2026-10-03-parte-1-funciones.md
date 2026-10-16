---
layout: post
title: "1.1 Funciones: Rastrigin y Schwefel"
date: 2026-10-03
categories: [parte-1]
---

## 1.1.1 Función de Rastrigin

![Superficie de la función de Rastrigin]({{ "/assets/figura1-rastrigin.png" | relative_url }})

*Figura 1. Superficie de la función de Rastrigin en tres dimensiones. Tomada de Surjanovic y Bingham (2013).*

La función de Rastrigin (Figura 1) es una función de prueba caracterizada por tener muchos mínimos locales: las montañas y valles de la superficie siguen un patrón cíclico y uniforme, a causa del coseno que se suma a un paraboloide. Esa combinación genera una cuadrícula regular de hoyos de tamaño casi idéntico, repartidos de manera simétrica alrededor del origen, como una especie de "caja de huevos" montada sobre un plato.

El dominio normalmente está dado por [-5.12, 5.12] en cada dimensión. A pesar de eso, el número de mínimos locales crece exponencialmente con la cantidad de dimensiones, lo que hace que sea muy exigente a la hora de optimizarla en más de dos dimensiones. El mínimo global se encuentra en el origen ($$x_i = 0$$ para todo $$i$$), con un valor de 0, en el centro de la gráfica, de modo que la simetría favorece algoritmos con sesgo hacia el centro del espacio de búsqueda.

$$f(x) = 10d + \sum_{i=1}^{d}\left[x_i^2 - 10\cos(2\pi x_i)\right], \qquad \mathrm{Dom}(f) = [-5.12,\ 5.12]$$

$$\text{min global} = (0, \dots, 0)$$

## 1.1.2 Función de Schwefel

![Superficie de la función de Schwefel]({{ "/assets/figura2-schwefel.png" | relative_url }})

*Figura 2. Superficie de la función de Schwefel en tres dimensiones. Tomada de Surjanovic y Bingham (2013).*

La función de Schwefel (Figura 2) tiene una característica particular: cuenta con múltiples mínimos locales y, a diferencia de Rastrigin, las montañas y valles que se forman en su superficie no son cíclicos ni uniformes, sino azarosos y de múltiples tamaños, a causa del término $$x$$ dentro de la función seno.

Además cuenta con un dominio mucho más amplio, [-500, 500], dependiendo de las dimensiones. Presenta anomalías en su superficie, como una indefinición en su derivada en el punto $$x_i = 0$$, lo que puede inutilizar los métodos por descenso de gradiente.

Es una función más caótica, debido a la irregularidad en sus picos y valles: las distancias pueden ser muy altas para algunos pasos definidos en los métodos y muy cortas para muchos otros, por ejemplo para la mutación gaussiana y uniforme.

$$f(x) = 418.9829\, d - \sum_{i=1}^{d} x_i \sin\left(\sqrt{\lvert x_i \rvert}\right), \qquad \mathrm{Dom}(f) = [-500,\ 500]$$

$$\text{min global} = (420.9687, \dots, 420.9687)$$
