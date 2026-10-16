---
layout: post
title: "4. Cacería de la alucinación"
date: 2026-10-12
categories: [ia]
---

## 4.1 Datos del mundo real

- En cuanto a las especificaciones del vehículo, al preguntarle por un carro con un consumo homologado de 5.5 litros por 100 km, el modelo de IA nos informó del Renault Clio TCe 115 y de sus especificaciones, entre las que se incluye la transmisión automática. No obstante, Renault España (s. f.) informa que el auto puede presentar tanto transmisión manual como automática según el tipo de motorización con el que el cliente lo necesite, lo que significa que, ante la ambigüedad de la afirmación, el modelo optó por mencionar solo uno de los dos tipos de transmisión.

## 4.2 Código

- Cuando se le pidió al modelo una alteración al programa para exportar un GIF y/o video en lugar de un gráfico del mejor recorrido, este no tomó en cuenta que la matriz de coordenadas contenía las latitudes de las capitales de España en su primera columna y las longitudes en la segunda, e interpretó lo contrario, de modo que en las primeras corridas del programa tras la actualización parecía no haber ciudades o recorrido sobre el mapa, puesto que las coordenadas se habían invertido.
- Se le pidió al modelo: "Busca exhaustivamente en la siguiente función (se le pasó la función `next_gen` del archivo `funciones.py`) algún error de implementación o lógica matemática y dame tu opinión sobre cómo implementarlo correctamente". La respuesta de la IA fue:

![Respuesta de la IA sobre la línea 52]({{ "/assets/alucinacion-respuesta-ia.png" | relative_url }})

  Revisando la línea 52 de la función vemos lo siguiente:

![Línea 52 de next_gen]({{ "/assets/alucinacion-codigo.png" | relative_url }})

  Sospechamos que existía una alucinación, ya que la función estaba dando los valores esperados y funcionaba perfectamente. Hicimos pruebas generando números aleatorios dentro del rango 0 a 3:

![Prueba de np.random.randint en la terminal]({{ "/assets/alucinacion-prueba-randint.png" | relative_url }})

  La respuesta fue 2, a pesar de generar un número exagerado de números aleatorios y varias corridas, por lo que vimos que no es inclusivo con el intervalo superior. También lo confirmamos con la documentación oficial de NumPy, donde se especifica que el intervalo que se entrega es de la forma [inferior, superior). La corrección fue no tomar en cuenta la corrección que hizo la IA. Si hubiéramos verificado desde un principio la documentación de la función `randint` del módulo `random` de NumPy, no habríamos dudado ni tenido que hacer pruebas para verificar si el intervalo era inclusivo o no.

## 4.3 Conceptos

- El prompt usado fue: "basándose en esos scripts y los .csv, ¿cómo puedo empezar a organizar mi parte del proyecto? dame un paso a paso, incluyendo librerías necesarias, qué tengo que leer y qué scripts crear. A mí me corresponde la parte del algoritmo evolutivo de la parte 2. Pienso hacerlo con pygad".

  La parte relevante de la respuesta del modelo fue: "Los cruces que trae PyGAD rompen permutaciones. single_point, two_points, uniform y scattered producen tours con ciudades repetidas. allow_duplicate_genes=False. Escribe tu propio cruce OX (o PMX) y pásalo como función en crossover_type."

  Si bien inmediatamente no hubo sospecha de alucinación, a los días surgió la duda porque el enunciado no pedía operadores propios, y en clase usamos PyGAD con sus parámetros por defecto en otros problemas sin necesidad de reescribir el cruce. La primera parte de la respuesta es cierta: si cada individuo es una lista de ciudades, el cruce por defecto produce ciudades repetidas. Lo que no era claro era que fuera obligatorio escribir un cruce propio.

  Consultando en internet, y con ayuda de otros LLM, resulta que ciertamente no era obligatoria la implementación de los cruces. En la literatura se presenta otra solución: claves aleatorias (Bean, 1994) con todos los parámetros por defecto de PyGAD, la cual no fue sugerida en ningún momento por la herramienta hasta que se le consultó su decisión y reconoció el error. Finalmente, al ya tener los cruces OX y PMX implementados, decidimos dejarlo así en el trabajo y añadir la posibilidad de comparación mediante los distintos tipos de cruces, sin dejar pasar por alto la alucinación de la IA de presentar su solución como la única posible.
