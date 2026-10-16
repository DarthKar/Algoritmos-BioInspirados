---
layout: post
title: "1.2 Métodos de optimización"
date: 2026-10-04
categories: [parte-1]
---

## 1.2.1 Método por descenso de gradiente analítico

El descenso por gradiente es un método iterativo que consiste en aprovechar las herramientas del cálculo vectorial y la diferenciabilidad de la función a optimizar, para ir dando pequeños pasos hacia la dirección que nos indica el gradiente.

$$x^{k+1} = x^{k} - \mu f'(x^{k})$$

El paso que da el método está determinado por un parámetro $$\mu$$ llamado "tasa de aprendizaje", que indica cuánto nos movemos en la dirección del gradiente. Su magnitud es determinante para la convergencia del método: si es demasiado pequeño, el método demorará mucho en converger, pero convergerá con más seguridad; por el contrario, si es demasiado grande, el método convergerá más rápido, pero puede pasarse de la solución y quedarse oscilando sin converger nunca (véase la Figura 3).

![Divergencia del descenso de gradiente con η = 1,1]({{ "/assets/figura3-divergencia.png" | relative_url }})

*Figura 3. Divergencia del descenso de gradiente con una tasa de aprendizaje demasiado alta (η = 1,1). Nota. Figura generada con Claude (Anthropic, 2026) a partir de la función f(x) = x² y una tasa de aprendizaje η = 1,1.*

Dentro de nuestra implementación un parámetro importante es la tolerancia, que fija el primer criterio de parada del algoritmo: paramos cuando el valor absoluto de la diferencia entre dos iteraciones consecutivas es menor que la tolerancia que definimos,

$$\lvert x^{k+1} - x^{k} \rvert < tol$$

El segundo criterio de parada está dado por un `max_iter` que limita el máximo de iteraciones que puede realizar el algoritmo independientemente de la tolerancia definida; es decir, una vez llegado a esa iteración, el método se detiene se haya pasado o no el umbral. Por su naturaleza, conviene poner un `max_iter` acorde al método utilizado o a una cota teórica conocida.

Por último, el método exige un punto $$x_0$$ inicial, que será el punto de partida del algoritmo y será decisivo para la convergencia. Para una función como Rastrigin o Schwefel, con muchos mínimos locales, es esperable que si el punto inicial no cae exactamente en el valle donde está el mínimo global, el método no converja a la solución óptima y se quede con una solución errónea en un mínimo local.

El método por descenso de gradiente que utilizamos, `grad_desc`, está basado en la implementación utilizada por el profesor en clase y utiliza el gradiente de la función y no la función en sí misma. Adicionalmente, tiene un parámetro `limites` para pasar la tupla correspondiente al dominio de la función objetivo.

## 1.2.2 Método por descenso de gradiente numérico

Esta versión del mismo método utiliza la definición de la derivada de una función en un punto cualquiera para aproximar el gradiente en todas las direcciones/dimensiones en que esté definida. Se divide en tres funciones: `dif_marginal`, `gradiente_numerico` y `grad_desc_num`.

- `dif_marginal` calcula, por medio de la definición de derivada en una sola variable, moviendo el punto de evaluación un paso `h` a la derecha y calculando el límite cuando `h` se hace más pequeña hasta llegar a 0.
- `gradiente_numerico` utiliza la aproximación dada por `dif_marginal` para aproximar la tasa de cambio en todas las direcciones/dimensiones de la función objetivo en el punto dado y crear un gradiente aproximado en un punto $$x_0$$.
- `grad_desc_num` es análoga a `grad_desc`, pero utiliza como gradiente el valor aproximado en cada punto mediante `gradiente_numerico`.

Algunas anotaciones de la implementación general de ambos métodos, analítico y numérico:

- Utilizan la función `np.clip()` para asegurar que cada nueva iteración $$x^{k+1}$$ no se salga del dominio y, en caso de hacerlo, redirigirla.
- Los métodos están definidos de forma matricial: dependiendo de la dimensión del `x0` que se pase en el orquestador principal, el método calcula el gradiente y el resultado en n dimensiones.
- Los parámetros en ambas funciones son completamente modificables, aunque cuentan con los siguientes valores por defecto: `x0` obligatorio, `grad_func` / `f_obj` obligatorio, `max_iter = 250`, `learning_rate = 0,002`, tolerancia `1e-5`, `limites = None`.

La conclusión general del método por descenso de gradiente es que, en todos los casos excepto aquel en que su $$x_0$$ caiga justo en el valle donde está el mínimo global, tanto para Rastrigin como para Schwefel el método se quedará atrapado en un mínimo local y, aunque convergerá rápido, no convergerá casi nunca a la solución real, por la naturaleza engañosa de ambas funciones de prueba.

## 1.2.3 Algoritmo evolutivo

Es un método metaheurístico inspirado en la selección natural. No usa derivadas, sólo evaluaciones de la función objetivo en puntos que mantiene como una población de individuos, cada uno representando una solución candidata, y la hace evolucionar a lo largo de generaciones, favoreciendo a quienes más se acercan a la solución que buscamos mediante su "fitness", que en nuestra implementación corresponde a los valores más bajos, ya que buscamos minimizar la función.

El algoritmo construye cada generación con cuatro operaciones. Primero se eligen los individuos que serán padres de la siguiente generación, dando preferencia a los más aptos (mejor fitness). Luego se combinan dos padres seleccionados al azar para producir hijos que heredan características de ambos, lo que permite explotar las zonas más prometedoras del espacio de búsqueda. Para evitar que se formen los mismos caminos y facilitar el escape de los mínimos locales, se introducen mutaciones que perturban aleatoriamente alguna coordenada del individuo sumando ruido, para que haya más diversidad. Finalmente se usa elitismo para conservar sin modificaciones los mejores individuos de la generación actual, garantizando que nos acerquemos a las mejores soluciones sin descartarlas entre generaciones.

Los parámetros del método son:

- `tam_poblacion`: número de individuos por generación.
- `p_cruce`: probabilidad de que dos padres seleccionados se crucen en lugar de pasar directamente a la siguiente generación.
- `p_mutacion`: probabilidad de que cada coordenada de un individuo mute. Un valor demasiado alto convierte al algoritmo en una búsqueda aleatoria, y uno demasiado bajo provoca pérdida de diversidad y convergencia prematura.
- `sigma`: desviación estándar del ruido gaussiano de la mutación, que determina qué tan lejos puede saltar un individuo mutado.
- `n_elite`: cantidad de individuos que se conservan intactos en cada generación.

El criterio de parada principal es `max_generaciones`, complementado con la `tolerancia` sobre la mejora del mejor individuo: si durante un número determinado de generaciones consecutivas la mejora es menor al umbral, el algoritmo para. La población inicial se genera de manera aleatoria y uniforme dentro del dominio, lo que elimina la dependencia de un único punto inicial. Notas sobre la implementación:

- Utiliza `np.clip()` para mantener a los individuos dentro de los límites del dominio tras el cruce y la mutación.
- La población se representa como una matriz de dimensión (`tam_poblacion`, `n_dimensiones`), por lo que el método opera en n dimensiones.
- Es estocástico, por lo que se recomienda fijar una semilla y repetir los experimentos varias veces para reportar resultados promedio.
- Parámetros por defecto: `f_obj` y `limites` obligatorios, `tam_poblacion = 100`, `max_generaciones = 250`, `p_cruce = 0,8`, `p_mutacion = 0,1`, `sigma = 0,1 × amplitud del dominio`, `n_elite = 2`, tolerancia `1e-5`.

Este método es el más robusto frente a la multimodalidad de Rastrigin y Schwefel, ya que la mutación y la diversidad de la población le permiten escapar de los mínimos locales donde el descenso de gradiente queda atrapado. A cambio, requiere muchas más evaluaciones de la función objetivo y su convergencia final es más lenta; además, su precisión cerca del óptimo depende de un buen ajuste de la mutación y del valor de `sigma`.

## 1.2.4 Optimización por enjambre de partículas (PSO)

Es un método metaheurístico, al igual que el algoritmo evolutivo, inspirado en el comportamiento de bandadas de aves o bancos de peces. A diferencia del descenso de gradiente, no requiere derivadas ni diferenciabilidad de la función objetivo, porque trabaja únicamente evaluando la función en distintos puntos del dominio. Esto lo hace muy útil para funciones como Rastrigin y Schwefel, ya que tienen muchos mínimos locales que suelen atrapar los métodos basados en el gradiente.

El método mantiene un conjunto de "partículas", cada una representando una solución candidata en el espacio de búsqueda. Cada partícula tiene una posición $$X$$ y una velocidad $$V$$, y recuerda la mejor posición en la que ha estado a lo largo de todo el recorrido ($$P_{best}$$). Además, el enjambre completo comparte la mejor posición que ha encontrado cualquiera de sus integrantes ($$G_{best}$$). En cada iteración la velocidad y la posición de cada partícula se actualizan según:

$$V(k+1) = W\,V(k) + C_1 R_1\,(P_{best} - X(k)) + C_2 R_2\,(G_{best} - X(k))$$

$$X(k+1) = X(k) + V(k+1)$$

donde $$R_1$$ y $$R_2$$ son números aleatorios uniformes en [0,1] que se generan en cada iteración. La velocidad combina tres componentes: la inercia de la partícula (su movimiento previo), la atracción hacia la mejor posición personal (componente cognitivo) y la atracción hacia la mejor posición global (componente social). Los parámetros que dictan el comportamiento de este método son:

- `w` (inercia): controla cuánto se conserva la velocidad anterior. Un valor alto favorece la exploración del espacio, pero dificulta la convergencia. Un valor bajo favorece la explotación de la zona donde ya se encuentran las partículas.
- `c1` (coeficiente cognitivo): determina qué tanto pesa la memoria individual de cada partícula.
- `c2` (coeficiente social): determina qué tanto pesa la información compartida por el enjambre. Si es demasiado alto frente a `c1`, el enjambre colapsa rápidamente hacia el $$G_{best}$$ y puede quedar atrapado en un mínimo local.
- `n_particulas`: tamaño del enjambre. Más partículas implican mayor cobertura del dominio y mayor probabilidad de caer en el valle del mínimo global, a costa de más evaluaciones de la función por iteración.

Anotaciones de la implementación:

- Emplea `np.clip()` para garantizar que las coordenadas de cada partícula no sobrepasen las cotas del dominio y corregirlas en caso de exceso.
- Está estructurado bajo un enfoque matricial, representando las ubicaciones y vectores de desplazamiento en estructuras de orden (`n_particulas`, `n_dimensiones`), lo cual facilita su ejecución directa en múltiples dimensiones.
- Debido al carácter aleatorio del procedimiento, las corridas generan soluciones divergentes entre sí, por lo que conviene definir una semilla para replicar las pruebas.
- Parámetros por defecto: `x0` no aplica (se genera aleatoriamente), `f_obj` y `limites` obligatorios, `n_particulas = 50`, `w = 0,7`, `c1 = 1,5`, `c2 = 1,5`, `max_iter = 250`, tolerancia `1e-5`.

La conclusión general es que PSO explora el espacio de búsqueda de forma mucho más global que el descenso de gradiente, por lo que logra acercarse al mínimo global con mayor frecuencia tanto en Rastrigin como en Schwefel. Aun así, no garantiza convergencia exacta. En Schwefel, donde el mejor valle está lejos del centro y es engañoso, el enjambre puede converger prematuramente hacia un mínimo local si pierde diversidad, y su desempeño depende de un ajuste adecuado de `w`, `c1` y `c2`.

## 1.2.5 Evolución diferencial

La evolución diferencial es un método metaheurístico que trabaja con una población de soluciones candidatas y no requiere derivadas. Por esta razón puede utilizarse en funciones no convexas y multimodales como Rastrigin y Schwefel.

En cada generación, para cada individuo $$x_i$$, se seleccionan tres individuos diferentes $$x_{r1}$$, $$x_{r2}$$ y $$x_{r3}$$. Luego se genera un vector mutante mediante:

$$v_i = x_{r1} + F\,(x_{r2} - x_{r3})$$

donde $$F$$ es el factor de mutación. Esta operación utiliza la diferencia entre dos individuos para definir la dirección y el tamaño del desplazamiento.

Después se realiza un cruce binomial entre el individuo original y el vector mutante: cada coordenada se toma del mutante con probabilidad `CR`, y al menos una siempre viene de él, para que el individuo de prueba nunca sea igual al original.

Luego se hace la selección: el individuo de prueba reemplaza al original sólo si su valor de la función objetivo es menor o igual. Así el mejor valor de la población nunca empeora entre generaciones.

Los parámetros del método son:

- `N`: tamaño de la población (mínimo 4, para poder escoger tres individuos distintos al actual).
- `F` (factor de mutación): escala la diferencia entre individuos. Un valor alto favorece la exploración, y uno bajo la explotación.
- `CR` (probabilidad de cruce): cuántas coordenadas toma el individuo de prueba del mutante.
- `max_gen`: número máximo de generaciones, que es el criterio de parada.
- `semilla`: permite reproducir la ejecución.

La población inicial es aleatoria y uniforme dentro del dominio. Se usa `np.clip()` para mantener al mutante dentro de los límites, y la población se representa como una matriz (N, d), por lo que el método opera en d dimensiones. Al ser estocástico, se recomienda fijar una semilla y repetir los experimentos.

Los parámetros por defecto son: `N = 50`, `max_gen = 200`, `F = 0,8`, `CR = 0,9`; `f_obj`, `d`, `LB` y `UB` son obligatorios.

La evolución diferencial es muy robusta frente a la multimodalidad de Rastrigin y Schwefel: al no usar gradiente y adaptar el tamaño de sus pasos a la dispersión de la población, escapa de los mínimos locales donde el descenso de gradiente queda atrapado. A cambio, requiere muchas evaluaciones de la función objetivo, y su desempeño depende de un buen ajuste de `F` y `CR`; si la población pierde diversidad, puede converger prematuramente.
