---
layout: post
title: "2.7.4 y 2.7.5 Comparación de operadores y valor de la hora"
date: 2026-10-10
categories: [parte-2]
---

## 2.7.4 Comparación de operadores

Para elegir los operadores se compararon las ocho combinaciones posibles de los dos cruces con las cuatro mutaciones. Cada combinación se ejecutó 30 veces, con las semillas 0 a 29, un valor de la hora de 15 EUR/h y el resto de los parámetros de la Tabla 9. Estos últimos se fijaron en pruebas preliminares y se mantuvieron iguales en todas las combinaciones, para que la comparación dependiera solo de los operadores. La desviación estándar es la muestral. Los resultados se presentan en la Tabla 10 y en la Figura 4.

| Cruce + mutación | Media (EUR) | Desv. est. | Mejor | Peor | Éxito | Evaluaciones |
|---|---|---|---|---|---|---|
| OX + inserción | 1843,09 | 76,28 | 1723,30 | 2019,08 | 0 % | 71 472 |
| OX + inversión | 1735,16 | 37,15 | 1677,08 | 1810,62 | 23 % | 79 725 |
| OX + mixta | 1722,50 | 39,35 | 1673,53 | 1815,08 | 50 % | 76 519 |
| OX + swap | 1859,81 | 75,48 | 1709,67 | 2054,12 | 0 % | 67 893 |
| PMX + inserción | 1923,22 | 99,98 | 1697,58 | 2262,29 | 3 % | 80 623 |
| PMX + inversión | 1755,24 | 42,69 | 1688,07 | 1891,27 | 13 % | 68 488 |
| PMX + mixta | 1727,96 | 31,12 | 1673,53 | 1783,59 | 33 % | 80 825 |
| PMX + swap | 2304,56 | 176,19 | 1916,46 | 2566,15 | 0 % | 57 205 |

*Tabla 10. Resultados de 30 corridas por combinación de operadores (valor de la hora 15 EUR/h). Evaluaciones: media de las evaluaciones reales, de 147 100 nominales. Fuente: elaboración propia.*

![Costo del mejor recorrido en 30 corridas por combinación de operadores]({{ "/assets/figura4-operadores.png" | relative_url }})

*Figura 4. Costo del mejor recorrido en 30 corridas por combinación de operadores. Fuente: elaboración propia con el comando `comparar` del orquestador.*

## 2.7.5 Estudio del valor de la hora

El enunciado pregunta si la ruta óptima cambia al variar el valor de la hora y a partir de qué valor. Para responderlo se aprovecha que ni la duración ni los peajes ni el combustible de un trayecto dependen del valor de la hora. El costo de un recorrido es entonces una recta en $$v_h$$:

$$L(\pi;\, v_h) = v_h \cdot H(\pi) + F(\pi)$$

donde $$H(\pi)$$ son las horas totales del recorrido y $$F(\pi)$$ la suma de sus peajes y su combustible. Así, un recorrido encontrado con un valor de la hora puede evaluarse de forma exacta para cualquier otro valor. El procedimiento fue el siguiente:

1. Se ejecutó el algoritmo genético 20 veces (semillas 0 a 19) para cada valor de la hora entre 0 y 60 EUR/h, en pasos de 5 EUR/h: 260 corridas en total.
2. Se reunieron todos los recorridos distintos encontrados, 230 en total.
3. Para cada valor de la hora se tomó el recorrido más barato del conjunto. Como cada recorrido es una recta, el mejor cambia de uno a otro exactamente donde las rectas se cruzan.

El paso del recorrido $$\pi_a$$ al recorrido $$\pi_b$$, que tarda menos pero cuesta más en peajes y combustible, ocurre en:

$$v_h^{*} = \frac{F(\pi_b) - F(\pi_a)}{H(\pi_a) - H(\pi_b)}$$

Es decir, el valor de la hora a partir del cual conviene el recorrido más rápido es igual a su sobrecosto dividido por las horas que ahorra.

**Resultados.** La mejor ruta conocida sí cambia, dos veces: en 14,57 EUR/h y en 22,47 EUR/h. La Tabla 11 presenta la mejor ruta de cada tramo, la Figura 5 muestra cuánto más cuesta cada una frente a la mejor en todo el intervalo y la Tabla 12 lista sus recorridos.

| Ruta | Valor de la hora (EUR/h) | Horas | Distancia (km) | Peajes (EUR) | Combustible (EUR) |
|---|---|---|---|---|---|
| R1 | 0 a 14,57 | 65,63 | 6313,7 | 47,26 | 642,07 |
| R2 | 14,57 a 22,47 | 65,18 | 6292,4 | 55,92 | 639,90 |
| R3 | 22,47 a 60 | 64,81 | 6174,0 | 76,32 | 627,86 |

*Tabla 11. Mejor ruta conocida según el valor de la hora. Fuente: elaboración propia.*

![Sobrecosto de cada ruta frente a la mejor ruta conocida según el valor de la hora]({{ "/assets/figura5-valor-hora.png" | relative_url }})

*Figura 5. Sobrecosto de cada ruta frente a la mejor ruta conocida, según el valor de la hora; las líneas punteadas marcan los cambios de ruta. Fuente: elaboración propia con el comando `hora` del orquestador.*

| Ruta | Recorrido |
|---|---|
| R1 | A Coruña, Pontevedra, Ourense, Zamora, León, Palencia, Valladolid, Segovia, Ávila, Salamanca, Cáceres, Badajoz, Huelva, Cádiz, Sevilla, Córdoba, Jaén, Granada, Málaga, Almería, Murcia, Alicante, Albacete, Ciudad Real, Toledo, Madrid, Guadalajara, Cuenca, Teruel, Valencia, Castellón de la Plana, Tarragona, Girona, Barcelona, Lleida, Huesca, Zaragoza, Soria, Burgos, Vitoria-Gasteiz, Logroño, Pamplona, San Sebastián, Bilbao, Santander, Oviedo, Lugo, A Coruña |
| R2 | A Coruña, Pontevedra, Ourense, Zamora, León, Burgos, Palencia, Valladolid, Segovia, Ávila, Salamanca, Cáceres, Badajoz, Huelva, Cádiz, Sevilla, Córdoba, Jaén, Granada, Málaga, Almería, Murcia, Alicante, Albacete, Ciudad Real, Toledo, Madrid, Guadalajara, Cuenca, Teruel, Valencia, Castellón de la Plana, Tarragona, Girona, Barcelona, Lleida, Huesca, Zaragoza, Soria, Logroño, Pamplona, San Sebastián, Vitoria-Gasteiz, Bilbao, Santander, Oviedo, Lugo, A Coruña |
| R3 | A Coruña, Pontevedra, Ourense, Zamora, Salamanca, Cáceres, Badajoz, Huelva, Sevilla, Cádiz, Málaga, Almería, Granada, Jaén, Córdoba, Ciudad Real, Albacete, Murcia, Alicante, Valencia, Castellón de la Plana, Tarragona, Girona, Barcelona, Lleida, Huesca, Zaragoza, Teruel, Cuenca, Guadalajara, Madrid, Toledo, Ávila, Segovia, Valladolid, Palencia, Burgos, Soria, Logroño, Pamplona, San Sebastián, Vitoria-Gasteiz, Bilbao, Santander, Oviedo, León, Lugo, A Coruña |

*Tabla 12. Recorridos de las rutas R1, R2 y R3. Fuente: elaboración propia.*

La tendencia es la esperada: a medida que la hora del vendedor vale más, la mejor ruta paga más peajes (de 47,26 a 76,32 EUR) para tardar menos (de 65,63 a 64,81 horas). El estudio se reproduce con:

```
python Parte_2/orquestador.py hora --version informe
```
