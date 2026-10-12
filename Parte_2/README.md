## Parte 2

## Problema y organización del sistema

Se busca el circuito de costo más bajo posible o uno optimizado para tal fin que incluya las 47 capitales de provincia de la España peninsular y regrese a su punto de partida. El costo está dado por el tiempo de viaje valorado económicamente, los peajes y el combustible.

Los índices de las ciudades se obtienen ordenando `capitales.csv` por `id`. Las matrices son dirigidas: la fila identifica el origen y la columna el destino. No se presupone que viajar de una ciudad a otra tenga el mismo costo que realizar el trayecto inverso.

El programa sigue esta secuencia:

1. `obtener_osrm.py` genera las matrices de distancias y tiempos.
2. `obtener_peajes_here.py` genera el archivo de peajes por arco dirigido.
3. `costos.py` convierte unidades y combina los componentes económicos.
4. `AntColonyOptimizer.fit()` construye y evalúa circuitos completos.
5. `colonia_de_hormigas.py` guarda el resultado y genera el mapa y las animaciones.

Los datos ya incluidos en la carpeta `data/` permiten construir costos y ejecutar el optimizador sin repetir esas consultas. el ejecutor `colonia_de_hormigas.py` produce un gif, video, gráfico y json con los resultados de la ejecución.

### Estructura

```text
Algoritmos-BioInspirados-main/
    requirements.txt
    Parte_2/
        README.md
        config_ejemplo.json
        data/
            capitales.csv
            osrm_distancia_m.csv
            osrm_tiempo_s.csv
            osrm_metadata.json
            here_peaje.csv
            here_cache/
                peaje/
        scripts/
            obtener_osrm.py
            obtener_peajes_here.py
            costos.py
            optimizador_colonia_de_hormigas.py
            colonia_de_hormigas.py
    outputs/
```

### Dependencias directas

| Dependencia | Uso |
|---|---|
| `requests` | Consultas HTTP a OSRM y HERE. |
| `numpy` | Matrices de costos, selección aleatoria y operaciones del optimizador. |
| `python-dotenv` | Carga de variables de entorno desde un archivo `.env`. |
| `matplotlib` | Figuras y animaciones. |
| `geopandas` | Lectura y dibujo del fondo geográfico. |
| `Pillow` | Escritura de GIF mediante `PillowWriter`. |
| `imageio-ffmpeg` | Localización del ejecutable FFmpeg para exportar MP4. |

Instalación de las dependencias directas de los cinco programas:

```bash
python -m pip install -r requirements.txt
```

### Variables de entorno

| Variable | Programa | Propósito |
|---|---|---|
| `HERE_API_KEY` | `obtener_peajes_here.py` | Credencial utilizada para consultar HERE. Se exige incluso si las respuestas ya están en caché. |
| `ACO_RANDOM_SEED` | `colonia_de_hormigas.py` | Semilla entera del generador aleatorio de NumPy. |

## A. obtener_osrm.py

### A.1. Objetivo y procedimiento

Obtiene distancias y tiempos de desplazamiento entre todas las capitales mediante el servicio `table` de OSRM, usando el perfil `driving` del servidor configurado en el código. Realiza una consulta para el conjunto completo de coordenadas.

El programa lee las capitales, ordena sus registros por identificador y construye una cadena de coordenadas en orden `longitud,latitud`. Solicita las anotaciones `duration,distance` y desactiva la generación de `hints`. La petición tiene un tiempo de espera de 60 segundos y un encabezado que identifica el uso académico.

Después de verificar la respuesta HTTP y el código `Ok` del servicio, comprueba que no existan distancias nulas, redondea las distancias al metro y los tiempos al segundo y verifica las dimensiones de la matriz de distancias. Los resultados pueden guardarse o compararse con los archivos existentes.

### A.2. Funciones

| Función | Entradas | Resultado y comportamiento |
|---|---|---|
| `leer_capitales()` | `data/capitales.csv`. | Lista de diccionarios ordenada por `id`. |
| `leer_matriz(ruta)` | Ruta a un CSV matricial. | Lista de listas de enteros; omite la cabecera y la primera columna. |
| `escribir_matriz(ruta, ids, matriz)` | Ruta de salida, identificadores y valores. | Escribe una cabecera `id` seguida de los identificadores y una fila por origen. |
| `main()` | Argumentos de consola. | Coordina consulta, comprobación y guardado o comparación. |

### A.3. Entradas y salidas

El CSV de capitales debe contener, como mínimo, `id`, `capital`, `lat` y `lon`. El modo normal escribe:

| Archivo | Contenido |
|---|---|
| `osrm_distancia_m.csv` | Distancias dirigidas en metros. |
| `osrm_tiempo_s.csv` | Duraciones dirigidas en segundos. |
| `osrm_metadata.json` | Fuente, fecha UTC, URL de consulta, unidades y puntos ajustados a la red vial. |

Los puntos ajustados incluyen el identificador de la capital, el nombre de la calle, la distancia del ajuste y sus coordenadas. Este registro ayuda a identificar qué puntos utilizó el servicio.

### A.4. Uso

Desde la raíz del proyecto:

```bash
python Parte_2/scripts/obtener_osrm.py
python Parte_2/scripts/obtener_osrm.py --comparar
```

`--comparar` realiza igualmente una consulta en línea, pero no sobrescribe los archivos. Lee las matrices locales y muestra la mayor diferencia absoluta en metros y segundos. Requiere que los archivos previos existan y tengan una organización compatible.

### A.5. Limitaciones

Los datos pueden cambiar al actualizarse la red vial. La marca temporal permite identificar la consulta utilizada, pero no hace invariable el servicio. El código no implementa reintentos y no crea `data/`. Verifica distancias nulas, pero no comprueba explícitamente duraciones nulas antes de redondearlas. Tampoco comprueba de forma equivalente las dimensiones de ambas matrices. Las excepciones de conexión y lectura de archivos no se capturan dentro del script.

## B. obtener_peajes_here.py

### B.1. Objetivo y procedimiento

Consulta HERE Routing API v8 para cada par ordenado de capitales diferentes. Para 47 ciudades se consideran `47 × 46 = 2162` arcos. La diagonal no se consulta, porque representa desplazamientos de una ciudad a sí misma.

El programa utiliza transporte `car`, búsqueda de ruta `fast` y `departureTime=any`. Esta última opción evita fijar una hora concreta de salida; no garantiza que futuras consultas produzcan respuestas idénticas.

Cada respuesta se almacena en un JSON independiente bajo `data/here_cache/<modo>/<i>_<j>.json`. Si el archivo existe, la consulta de ese arco se omite. Al finalizar, las respuestas disponibles se procesan para reconstruir el CSV del modo seleccionado.

### B.2. Modos de consulta

| Modo | Configuración | Tratamiento de peajes |
|---|---|---|
| `peaje` | Solicita `summary,tolls`, moneda EUR y totales de peajes. | Extrae importes de la respuesta. No obliga a que la ruta tenga peajes. |
| `sin_peaje` | Solicita evitar `tollRoad`. | Registra cero si no aparecen avisos con `violated`; si aparecen, deja el importe sin determinar. |

El archivo `here_sin_peaje.csv` se genera solo al ejecutar el modo correspondiente. No sustituye automáticamente al archivo que lee `costos.py`.

### B.3. Funciones

| Función | Entradas | Salida |
|---|---|---|
| `leer_capitales()` | CSV de capitales. | Registros ordenados por `id`. |
| `parametros(origen, destino, modo, api_key)` | Registros de dos capitales, modo y credencial. | Diccionario de parámetros HTTP; HERE recibe coordenadas en orden `latitud,longitud`. |
| `extraer(j, modo)` | Respuesta JSON y modo. | Tupla `(dist_m, tiempo_s, peaje_eur, metodo, avisos)`. |
| `main()` | Opciones de consola y entorno. | Ejecuta consultas pendientes y escribe el CSV. |

### B.4. Extracción del importe

`extraer()` utiliza únicamente la primera ruta de la respuesta y suma las distancias y duraciones de sus secciones. Si no hay rutas, devuelve valores `None`, método `sin_ruta` y los avisos disponibles.

En modo `peaje`, intenta primero sumar `summary.tolls.total` de todas las secciones. Si todos los totales están presentes, redondea el resultado a dos decimales y registra el método `summary.tolls.total`. Si alguna moneda difiere de EUR, añade `moneda_no_EUR`, pero el código sigue sumando los valores sin convertirlos.

Cuando no dispone de todos los totales, recurre a los detalles de peajes. Para cada peaje busca `convertedPrice` o, en su ausencia, `price`, conserva las tarifas en EUR y suma la mínima. Esto evita sumar varias modalidades de pago del mismo peaje. El método registrado es `min_fares_por_peaje`.

Si un peaje contiene tarifas pero ninguna aporta un importe válido en EUR, se añade `peaje_sin_precio_EUR`; ese peaje no incrementa la suma. Por ello, un resultado numérico del método de respaldo puede ser incompleto y los avisos deben revisarse.

### B.5. Argumentos

| Argumento | Tipo | Valor predeterminado | Descripción |
|---|---|---|---|
| `--modo` | Cadena | Obligatorio | `peaje` o `sin_peaje`. |
| `--limite` | Entero | Sin límite | Máximo de peticiones nuevas; no limita las respuestas reutilizadas. |
| `--pausa` | Real | `0.5` | Segundos de espera después de guardar cada respuesta nueva. |

### B.6. Archivo de salida y reutilización

El archivo `data/here_<modo>.csv` contiene:

| Columna | Significado |
|---|---|
| `i`, `j` | Identificadores del origen y destino. |
| `origen`, `destino` | Nombres de capitales. |
| `dist_m`, `tiempo_s` | Distancia y duración de la ruta HERE. |
| `peaje_eur` | Importe extraído o campo vacío si no se pudo determinar. |
| `metodo_peaje` | Método aplicado o motivo de ausencia. |
| `avisos` | Códigos que requieren interpretación. |

El programa indica cuántos arcos carecen de un JSON en caché. Ese contador no incluye necesariamente respuestas existentes que no contienen una ruta o un peaje utilizable. Una ejecución limitada puede escribir un CSV parcial y sobrescribir el anterior del mismo modo.

La caché permite reanudar el trabajo conservando consultas completadas. No tiene fecha de caducidad ni identifica cambios de coordenadas o de parámetros; los archivos existentes se reutilizan por nombre. El encabezado del script menciona cupos comerciales consultados previamente: esos números no se validan durante la ejecución ni deben interpretarse como una garantía del plan disponible.

## C. costos.py

### C.1. Objetivo y modelo económico

Centraliza la lectura de datos y calcula el costo de cada arco dirigido. Para un desplazamiento de la capital `i` a la capital `j`:

C_ij= v_h * H_ij + P_ij + D_ij * q *p_f

| Símbolo | Unidad | Descripción |
|---|---|---|
| `v_h` | €/h | Valor de una hora del vendedor. |
| `H_ij` | h | Duración del viaje. |
| `P_ij` | € | Peajes. |
| `D_ij` | km | Distancia por carretera. |
| `q` | L/100 km | Consumo del vehículo. |
| `p_f` | €/L | Precio del combustible. |

Las distancias de OSRM se dividen por 1000 y sus tiempos por 3600. El consumo y el precio se consideran constantes durante todos los desplazamientos. La diagonal de la matriz económica se fija en cero.

### C.2. Funciones públicas

| Función | Parámetros | Retorno |
|---|---|---|
| `leer_capitales()` | Ninguno. | `(nombres, ubicaciones)`, con coordenadas NumPy de forma `(n, 2)` en orden `[lat, lon]`. |
| `leer_matriz(nombre)` | Nombre de un CSV en `data/`. | Matriz NumPy de valores reales, omitiendo cabecera e identificadores de fila. |
| `leer_peajes(n, archivo="here_peaje.csv")` | Número de ciudades y archivo. | Matriz `(n, n)` de peajes en EUR, con diagonal cero. |
| `matriz_costos(valor_hora, consumo_l_100km, precio_eur_l, peajes)` | Parámetros económicos y matriz de peajes. | `(C, partes)`. |
| `costo_tour(C, tour)` | Matriz económica y permutación de ciudades. | Costo del circuito cerrado como `float`. |

`partes` es un diccionario con las claves `km`, `horas`, `peajes` y `combustible`. Las dos primeras contienen magnitudes físicas; las dos últimas, importes en euros. El costo monetario del tiempo se obtiene multiplicando `horas` por `valor_hora`.

`leer_peajes()` comienza con valores `NaN` fuera de la diagonal. Solo reemplaza un valor si el campo `peaje_eur` no está vacío ni contiene `None`. Si queda algún `NaN`, lanza `ValueError` indicando cuántos arcos no tienen importe consultado. Esta comprobación evita tratar automáticamente la falta de información como peaje cero, aunque no detecta importes parciales acompañados de avisos.

### C.3. Costo de un circuito

Para una permutación `tour` de `n` ciudades:

\[
L(\pi)=\sum_{s=0}^{n-1} C_{\pi_s,\pi_{(s+1)\bmod n}}
\]

La implementación usa `np.roll(tour, -1)` para incluir el regreso al origen. La representación prevista es una permutación sin repetir la ciudad inicial al final. La función no verifica que cada ciudad aparezca exactamente una vez; esa validez corresponde al llamador.

Ejemplo de integración:

```python
from costos import leer_capitales, leer_peajes, matriz_costos, costo_tour

nombres, ubicaciones = leer_capitales()
P = leer_peajes(len(nombres))
C, partes = matriz_costos(25.0, 5.5, 1.849, P)
costo_referencia = costo_tour(C, list(range(len(nombres))))
```

Este ejemplo puede utilizarse desde otro script del mismo directorio. La ruta por orden de identificadores es una referencia de cálculo y no una solución optimizada.

### C.4. Ejecución independiente

```bash
python Parte_2/scripts/costos.py --valor-hora 25 --consumo 5.5 --precio 1.849
```

Los tres argumentos son obligatorios en este script. Se imprimen la suma de todos los peajes de la matriz, el número de arcos con peaje positivo, la forma de la matriz, la máxima diferencia entre sentidos opuestos y el costo del circuito por orden de `id`. La suma de peajes de toda la matriz no es el peaje de un recorrido. No se escribe una matriz económica a disco.

### C.5. Supuestos y limitaciones

Los identificadores deben corresponder a índices consecutivos de `0` a `n-1`. Las matrices deben conservar el mismo orden de ciudades: `leer_matriz()` descarta sus etiquetas sin comprobarlas. El programa no valida explícitamente dimensiones, finitud o signo de todos los parámetros.

La implementación usa distancia y tiempo de OSRM, y peajes de HERE. Aunque HERE también guarda distancia y duración, `matriz_costos()` no utiliza esos campos. Por tanto, la matriz combina dos fuentes que podrían elegir carreteras diferentes entre las mismas capitales. El costo es el del modelo integrado; no queda demostrado que sus tres componentes correspondan al mismo trayecto vial.

## D. colonia_de_hormigas.py

### D.1. Objetivo y flujo de ejecución

Es el punto de entrada del experimento ACO. Carga la semilla, configura FFmpeg, interpreta los argumentos, lee las capitales y peajes, construye la matriz económica y crea el optimizador.

Ejecuta `fit(problem, iterations=500)`, con minimización y parada anticipada predeterminadas. El número 500 es un máximo fijado dentro del programa: no existe un argumento de consola para modificarlo en esta versión.

Después imprime el mejor costo y guarda el resultado JSON. Descarga un GeoJSON de países de Natural Earth desde el repositorio `nvkelso/natural-earth-vector`, selecciona el registro `ADMIN == "Spain"` y genera las visualizaciones. Si falla la descarga del mapa, la optimización y el JSON pueden haberse completado, pero las imágenes y animaciones no se generarán.

### D.2. Argumentos de consola

| Argumento | Tipo | Predeterminado | Significado |
|---|---|---|---|
| `--valor-hora` | Real | Obligatorio | Valor de la hora del vendedor en EUR/h. |
| `--consumo` | Real | `5.5` | Consumo en L/100 km. |
| `--precio` | Real | `1.849` | Precio del combustible en EUR/L. |
| `--hormigas` | Entero | `40` | Hormigas por iteración. |
| `--evaporacion` | Real | `0.05` | Fracción de evaporación de feromonas. |
| `--intensificacion` | Real | `2` | Depósito constante de feromonas. |
| `--alfa` | Real | `2` | Exponente de feromonas. |
| `--beta` | Real | `1` | Exponente heurístico. |
| `--beta-evaporacion` | Real | `0` | Reducción del exponente heurístico. |
| `--p-mejor` | Real | `0.1` | Probabilidad de selección determinista del mejor peso. |
| `--version` | Cadena | Obligatorio | Sufijo de los archivos del experimento. |

`--version` identifica una ejecución; no representa el número de iteraciones del algoritmo. Reutilizar el mismo sufijo en el mismo directorio de trabajo puede sobrescribir las salidas anteriores.

Ejemplo mínimo con `ACO_RANDOM_SEED` ya configurada:

```bash
python Parte_2/scripts/colonia_de_hormigas.py --valor-hora 25 --version 0001
```

Ejemplo con parámetros explícitos:

```bash
python Parte_2/scripts/colonia_de_hormigas.py --valor-hora 25 --consumo 5.5 --precio 1.849 --hormigas 100 --evaporacion 0.05 --intensificacion 1.9 --alfa 2.1 --beta 0.9 --beta-evaporacion 0 --p-mejor 0.09 --version 0030
```

### D.3. Función crear_animacion

```python
crear_animacion(optimizer, ubicaciones, espana, version,
                intervalo=5, fps=5,
                exportar_gif=True, exportar_mp4=True)
```

| Parámetro | Descripción |
|---|---|
| `optimizer` | Optimizador con historiales de rutas y costos globales. |
| `ubicaciones` | Coordenadas `[lat, lon]` en el mismo orden de las ciudades de las rutas. |
| `espana` | Datos geográficos que se dibujan como fondo. |
| `version` | Sufijo de los archivos. |
| `intervalo` | Paso para seleccionar entradas del historial. |
| `fps` | Fotogramas por segundo de la reproducción y exportación. |
| `exportar_gif`, `exportar_mp4` | Activan las exportaciones desde la función; no hay opciones equivalentes en consola. |

La función selecciona los índices `0, intervalo, 2 × intervalo, ...` y añade el último si aún no estaba seleccionado. Por tanto, la animación incluye la solución final aunque el tamaño del historial no sea múltiplo del intervalo. Con los valores del ejecutor se representa aproximadamente una entrada de cada cinco iteraciones, a cinco fotogramas por segundo.

El mapa muestra capitales azules, una línea roja y un recuadro con el número de iteración y el mejor costo global. La función interna `actualizar(frame)` convierte los índices de la ruta a coordenadas y cambia la línea mediante `set_data()`. Como el circuito del optimizador ya incluye el origen al final, la línea queda cerrada.

Se dibujan segmentos rectos entre las coordenadas de las capitales. La visualización representa el orden de visitas; no reproduce la geometría de las carreteras que originaron los costos. Los límites del gráfico se fijan entre longitudes `-10` y `4.5` y latitudes `35.5` y `44.5` para centrar la península.

Los fotogramas muestran la mejor solución global, por lo que puede haber varios sin cambios si no se encontraron mejoras. Además, el muestreo puede omitir soluciones intermedias del historial completo.

### D.4. Exportaciones

| Archivo | Contenido |
|---|---|
| `outputs/resultados_<version>.json` | Mejor costo, valor de la hora y mejor ruta como cadena de nombres. |
| `outputs/Recorrido_ACO_<version>.png` | Figura de la mejor ruta final. |
| `outputs/Evolucion_ACO_<version>.gif` | Evolución de la mejor solución global, mediante `PillowWriter`, a 90 dpi. |
| `outputs/Evolucion_ACO_<version>.mp4` | La misma evolución, mediante FFmpeg, H.264 y formato de píxel `yuv420p`, a 120 dpi. |

El MP4 utiliza `libx264`, bitrate configurado en 1800 y el ejecutable obtenido por `imageio_ffmpeg.get_ffmpeg_exe()`. Su ruta se asigna a `mpl.rcParams["animation.ffmpeg_path"]`. El programa no incluye una función de descarga al navegador: escribe los archivos directamente en el equipo que ejecuta Python.

El PNG se guarda dentro de `actualizar()` al llegar al último índice. Si ambas exportaciones están activadas, ese fotograma puede guardarse durante ambas. Si ambas están desactivadas, no se debe asumir que el PNG se genera, porque no hay una llamada independiente que dibuje y guarde el último fotograma.

El JSON contiene exactamente las claves `Mejor costo`, `Valor de la hora` y `Mejor ruta`. No guarda por sí solo la semilla, todos los parámetros ACO, el tiempo de ajuste ni los historiales completos. La ruta exportada repite el nombre de la primera capital al final.

### D.5. Condiciones y limitaciones actuales

La variable `ACO_RANDOM_SEED` debe contener un entero. Si no existe, el código intenta ejecutar `sys.exit()`, pero el archivo no importa `sys`: en esa rama se producirá `NameError` en lugar del mensaje previsto. El uso normal con la semilla configurada evita esa rama; la corrección necesaria sería añadir `import sys`.

`crear_animacion()` presupone historiales no vacíos, `intervalo` entero positivo, `fps` positivo y coordenadas compatibles. No valida estas condiciones. El ejecutor no ofrece desactivar las animaciones ni cargar un mapa local desde consola. La exportación completa también presupone la disponibilidad del ejecutable FFmpeg.

## Créditos y código de terceros

La implementación del algoritmo de optimización por colonia de hormigas (ACO) utilizada en este proyecto está basada en el código desarrollado por el usuario **johnberroa** y publicado en GitHub:

- **Repositorio original:** [Ant-Colony-Optimization](https://github.com/johnberroa/Ant-Colony-Optimization)
- **Archivo utilizado:** `AntColonyOptimizer.py`

Para su integración en este proyecto se realizaron adaptaciones que permiten registrar la evolución de la mejor solución encontrada y utilizar esta información para generar visualizaciones animadas de las rutas mediante GIF y MP4.

El modelo de costos aplicado al problema de las 47 capitales peninsulares, la integración de los componentes y las visualizaciones se documentan como parte del desarrollo del presente trabajo.

# Datos del TSP: 47 capitales de provincia de la España peninsular

## Archivos de `data/`

| Archivo | Contenido | Fuente | Consulta |
|---|---|---|---|
| `capitales.csv` | id 0 a 46, nombre, nombre y código INE, geonameid, lat, lon | INE tabla 67991; GeoNames | 2026-10-10 |
| `osrm_distancia_m.csv` | Matriz dirigida 47x47 de distancia por carretera (m) | OSRM, datos OpenStreetMap | 2026-10-10 06:07 UTC |
| `osrm_tiempo_s.csv` | Matriz dirigida 47x47 de tiempo sin tráfico (s) | OSRM | 2026-10-10 06:07 UTC |
| `osrm_metadata.json` | URL exacta de la consulta, método de verificación, punto de ajuste de cada ciudad | OSRM | 2026-10-10 |
| `vias_peaje_osrm.csv` | Por arco: vías candidatas a peaje que recorre la ruta OSRM y km | OSRM | 2026-10-10 |
| `estado_vias_peaje_2026.csv` | Estado 2026 de cada vía (vigente, liberada, verificar) con fuente | Ver columna fuente | 2026-10-10 |
| `combustible.csv` | Precio medio en España con impuestos | Comisión Europea, Weekly Oil Bulletin (datos al 2026-10-05) | 2026-10-10 |
| `here_peaje.csv` | Información de peajes por arco dirigido de una pareja de ciudades | Obtenido de la API HERE mediante el módulo obtener_peajes_here | 2026-10-10 |
Fila = origen, columna = destino. `[i, j]` puede ser distinto de `[j, i]`.

## En caso de querer actualizar los datos de peaje

**Peajes en EUR**: correr `scripts/obtener_peajes_here.py` con una API key de HERE (ver el encabezado del script).

## Vehículo seleccionado

Renault Clio TCe 115, consumo homologado de 5,3–5,5 L/100 km (tomado como 5,5 L/100 km). https://www.renault.es/preguntas-frecuentes-vehiculos/clio/esta-renault-clio-disponible-con-una-caja-de-cambios-manual-o-automatica.html Revisado 2026-10-10

## Scripts

| Programa | Responsabilidad principal |
|---|---|
| `obtener_osrm.py` | Obtener matrices dirigidas de distancia y tiempo por carretera. |
| `obtener_peajes_here.py` | Consultar peajes y guardar respuestas reutilizables por trayecto. |
| `costos.py` | Construir la matriz económica y calcular el costo de circuitos cerrados. |
| `optimizador_colonia_de_hormigas.py` | Implementar el optimizador ACO y registrar su evolución. |
| `colonia_de_hormigas.py` | Integrar datos y optimizador, y exportar resultados y visualizaciones. |

```bash
pip install requests numpy
python Parte_2/scripts/costos.py --valor-hora 15 --consumo 5.5 --precio 1.849   # matriz de costos y prueba
python Parte_2/scripts/obtener_osrm.py --comparar                                # vuelve a consultar OSRM y compara
python Parte_2/scripts/obtener_peajes_here.py --modo peaje --limite 3            # prueba con 3 arcos
python Parte_2/scripts/colonia_de_hormigas.py --valor-hora 25 --hormigas 100 --intensificacion 1.9 --alfa 2.1 --beta 0.9 --beta-evaporacion 0 --p-mejor 0.09 --version 0030--version 0001
```
