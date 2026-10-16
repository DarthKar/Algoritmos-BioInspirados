---
layout: post
title: "Uso de IA y cacería de alucinaciones"
date: 2026-10-15
categories: [ia]
---

Usamos IA para armar el menú interactivo (orquestador), revisar nuestro código, buscar patrones en los recorridos, generar las animaciones de las rutas y formular el estudio del valor de la hora. Lo que más nos enseñó fueron los errores.

## Datos del mundo real

Al pedir un auto de 5,5 L/100 km, el modelo propuso el Renault Clio TCe 115 y dijo que era automático. Renault España indica que según la motorización puede ser manual o automático, así que el modelo mencionó solo una de las dos opciones.

## Código

- **Latitud y longitud invertidas.** Al pedir un GIF del mejor recorrido, el modelo asumió el orden contrario al de nuestra matriz de coordenadas, y el mapa salía sin ciudades.
- **Un error que no existía.** Al revisar `next_gen`, el modelo dijo que `np.random.randint(0, N, ...)` podía devolver `N` por ser inclusivo en el extremo superior. Probamos con un millón de números en el rango 0 a 3 y el máximo fue 2. La documentación de NumPy lo confirma: el intervalo es [inferior, superior). Habríamos ahorrado la prueba leyendo la documentación primero.

## Conceptos

El modelo afirmó que, con PyGAD, había que escribir cruces propios porque los incluidos repiten ciudades. Lo primero es cierto, pero lo segundo no: otra solución de la literatura son las **claves aleatorias** (Bean, 1994) con los parámetros por defecto de PyGAD, y la herramienta nunca la mencionó hasta que le preguntamos. Como ya teníamos OX y PMX, los dejamos y aprovechamos para compararlos, pero nos quedó la lección de que una respuesta presentada como la única opción puede no serlo.
