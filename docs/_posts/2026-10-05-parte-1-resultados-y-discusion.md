---
layout: post
title: "1.3 Comparación justa, resultados y 1.4 Discusión"
date: 2026-10-05
categories: [parte-1]
---

## 1.3 Regla de comparación justa

Para comparar los métodos se contabilizaron por separado las evaluaciones de la función objetivo ($$f$$) y las evaluaciones del gradiente ($$\nabla f$$). El gradiente numérico se calculó mediante diferencias hacia adelante:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x)}{h}$$

En la implementación utilizada, cada coordenada requiere una evaluación en el punto perturbado y otra en el punto original. Por ello se consideró aproximadamente 1 evaluación de $$\nabla f \approx 2d$$ evaluaciones de $$f$$. Como los experimentos se realizaron con $$d = 2$$, 1 evaluación de $$\nabla f \approx 2(2) = 4$$ evaluaciones de $$f$$. Por lo tanto, el costo equivalente se calculó mediante

$$C_{eq} = n_f + 2d\, n_{\nabla f} \quad\Longrightarrow\quad C_{eq} = n_f + 4\, n_{\nabla f} \ \ (d = 2)$$

Se realizaron 30 corridas independientes por método y se calcularon los principales estadísticos.

## 1.3.1 Resultados para la función Rastrigin

La función de Rastrigin tiene su mínimo global en el punto (0, 0), donde el valor mínimo de la función es $$f(x^*) = 0$$.

| Método | Media | Desv. est. | Mejor | Peor | Éxito | Eval(f) | Eval(∇f) | Equiv. |
|---|---|---|---|---|---|---|---|---|
| GD analítico | 18.17 | 10.43 | 3.98 | 40.79 | 0 % | 1 | 10 | 40 |
| GD numérico | 18.17 | 10.43 | 3.98 | 40.79 | 0 % | 40 | 0 | 40 |
| Evolutivo uniforme | 0.03132 | 0.04221 | 0.00004871 | 0.1847 | 90 % | 10050 | 0 | 10050 |
| Evolutivo Gaussiano | 0.0001146 | 0.0001908 | 0.00000002151 | 0.0006299 | 100 % | 10050 | 0 | 10050 |
| PSO | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | 100 % | 10050 | 0 | 10050 |
| DE | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | 100 % | 10050 | 0 | 10050 |

*Tabla 1. Resultados para la función Rastrigin. Fuente: ejecución en Python.*

Los valores que aparecen como aproximadamente cero corresponden a soluciones muy cercanas al mínimo global. PSO y DE alcanzaron una tasa de éxito del 100 %, mientras que los métodos de descenso por gradiente no lograron alcanzar el criterio de éxito establecido.

## 1.3.2 Resultados para la función Schwefel

La función de Schwefel tiene su mínimo global aproximadamente cuando cada coordenada vale 420.9687. En ese punto, el valor mínimo de la función es aproximadamente cero.

| Método | Media | Desv. est. | Mejor | Peor | Éxito | Eval(f) | Eval(∇f) | Equiv. |
|---|---|---|---|---|---|---|---|---|
| GD analítico | 409.7 | 177 | 1.091e-10 | 711.1 | 3 % | 1 | 51 | 205 |
| GD numérico | 409.7 | 177 | 6.345e-8 | 711.1 | 3 % | 205 | 0 | 205 |
| Evolutivo uniforme | 0.246 | 0.3228 | 0.00002054 | 1.276 | 97 % | 10 050 | 0 | 10 050 |
| Evolutivo Gaussiano | 11.84 | 35.53 | 4.152e-8 | 118.4 | 90 % | 10 050 | 0 | 10 050 |
| PSO | 55.27 | 59.09 | 2.274e-13 | 118.4 | 53 % | 10 050 | 0 | 10 050 |
| DE | 2.274e-13 | 0 | 2.274e-13 | 2.274e-13 | 100 % | 10 050 | 0 | 10 050 |

*Tabla 2. Resultados para la función Schwefel. Fuente: ejecución en Python.*

DE obtuvo una tasa de éxito del 100 % y resultados prácticamente iguales a cero en las 30 corridas. El evolutivo uniforme también presentó un buen desempeño, con un 97 % de éxito. En cambio, el descenso por gradiente alcanzó solamente un 3 % de éxito, lo que refleja la dificultad de optimizar esta función mediante métodos basados en gradientes.

## 1.4 Discusión

Rastrigin y Schwefel son funciones no convexas y multimodales, por lo que presentan varios mínimos locales. Esta característica dificulta el desempeño de los métodos de descenso por gradiente.

En Rastrigin, el descenso analítico y el numérico obtuvieron una tasa de éxito de 0 %. Aunque utilizaron pocas evaluaciones equivalentes, quedaron atrapados en mínimos locales. En cambio, PSO y DE alcanzaron valores cercanos a cero en todas las corridas, y los algoritmos evolutivos obtuvieron tasas de éxito entre 90 % y 100 %. Esto muestra que los métodos heurísticos exploran mejor el dominio, aunque utilizan muchas más evaluaciones de $$f$$.

En Schwefel, el descenso por gradiente también tuvo un costo bajo, pero solamente alcanzó una tasa de éxito del 3 %. El algoritmo evolutivo con mutación uniforme logró 97 %, el evolutivo gaussiano 90 % y DE 100 %. PSO obtuvo un 53 %, por lo que fue menos estable que DE en esta configuración.

En conclusión, los métodos de descenso por gradiente son económicos y útiles para realizar refinamiento local, pero dependen mucho del punto inicial y pueden quedar atrapados en mínimos locales. Los métodos heurísticos requieren aproximadamente 10050 evaluaciones de la función, pero tienen una mayor capacidad de exploración global y alcanzan mejores resultados en funciones multimodales como Rastrigin y Schwefel.
