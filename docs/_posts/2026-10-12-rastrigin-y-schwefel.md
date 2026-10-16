---
layout: post
title: "Rastrigin y Schwefel: cuando el gradiente no alcanza"
date: 2026-10-12
categories: [parte-1]
---

## Las dos funciones

**Rastrigin** es una "caja de huevos" montada sobre un plato: una cuadrícula regular de mínimos locales de tamaño casi idéntico. Su mínimo global está en el origen, con valor 0, y su dominio habitual es $$[-5.12, 5.12]$$.

$$f(x) = 10d + \sum_{i=1}^{d}\left[x_i^2 - 10\cos(2\pi x_i)\right]$$

**Schwefel** es más caótica: sus valles no son uniformes ni cíclicos, el dominio es $$[-500, 500]$$ y el mínimo global queda lejos del centro, en $$x_i \approx 420.9687$$. Además su derivada no está definida en $$x_i = 0$$.

$$f(x) = 418.9829\,d - \sum_{i=1}^{d} x_i \sin\left(\sqrt{|x_i|}\right)$$

## Los métodos

- **Descenso de gradiente** (analítico y numérico): $$x^{k+1} = x^k - \mu f'(x^k)$$. Es barato, pero depende totalmente del punto inicial. Con una tasa de aprendizaje demasiado alta oscila y diverge.
- **Algoritmo evolutivo**: población, selección, cruce, mutación (uniforme o gaussiana) y elitismo.
- **PSO**: partículas con velocidad que se atraen hacia su mejor posición y hacia la mejor del enjambre.
- **Evolución diferencial (DE)**: mutante $$v_i = x_{r1} + F(x_{r2} - x_{r3})$$, cruce binomial y selección voraz.

## Comparación justa

Contamos por separado evaluaciones de $$f$$ y de $$\nabla f$$. Con diferencias hacia adelante, un gradiente cuesta $$2d$$ evaluaciones de $$f$$, así que para $$d=2$$ el costo equivalente es $$C_{eq} = n_f + 4\,n_{\nabla f}$$. Hicimos 30 corridas independientes por método.

## Resultados en Rastrigin

| Método | Media | Mejor | Éxito | Costo equiv. |
|---|---|---|---|---|
| GD analítico | 18.17 | 3.98 | 0 % | 40 |
| GD numérico | 18.17 | 3.98 | 0 % | 40 |
| Evolutivo uniforme | 0.03132 | 0.00004871 | 90 % | 10 050 |
| Evolutivo gaussiano | 0.0001146 | 2.151e-8 | 100 % | 10 050 |
| PSO | ≈ 0 | ≈ 0 | 100 % | 10 050 |
| DE | ≈ 0 | ≈ 0 | 100 % | 10 050 |

## Resultados en Schwefel

| Método | Media | Éxito | Costo equiv. |
|---|---|---|---|
| GD analítico | 409.7 | 3 % | 205 |
| GD numérico | 409.7 | 3 % | 205 |
| Evolutivo uniforme | 0.246 | 97 % | 10 050 |
| Evolutivo gaussiano | 11.84 | 90 % | 10 050 |
| PSO | 55.27 | 53 % | 10 050 |
| DE | ≈ 0 | 100 % | 10 050 |

## Qué aprendimos

El gradiente es barato y sirve para refinar localmente, pero queda atrapado en mínimos locales. Los métodos heurísticos gastan unas 10 050 evaluaciones y a cambio exploran todo el dominio. En Schwefel, DE fue el más estable (100 %), mientras que PSO (53 %) sufrió por perder diversidad demasiado pronto.
