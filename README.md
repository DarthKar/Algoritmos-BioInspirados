# Parte 1: optimización de Rastrigin y Schwefel

## Link al blog: https://darthkar.github.io/Algoritmos-BioInspirados/

## Objetivo y organización

La Parte 1 implementa y compara cinco métodos para minimizar las funciones de Rastrigin y Schwefel: descenso por gradiente analítico, descenso por gradiente numérico, algoritmo evolutivo, optimización por enjambre de partículas (PSO) y evolución diferencial (DE).

La implementación separa las responsabilidades en dos archivos:

| Archivo | Responsabilidad |
|---|---|
| `Parte 1/orquestador.py` | Define las funciones objetivo y sus parámetros, ofrece el menú y la interfaz de línea de comandos, ejecuta experimentos y prepara resultados, figuras y GIF. |
| `Parte 1/funciones.py` | Contiene los algoritmos de optimización, operadores del algoritmo evolutivo y herramientas de visualización. Recibe la función objetivo y los límites como argumentos, por lo que no está ligado a Rastrigin ni a Schwefel. |
| `Parte 1/salidas/` | Carpeta creada junto al orquestador para las figuras y GIF generados. |

El módulo `funciones.py` es la implementación común. Las funciones objetivo y gradientes se pasan como argumentos, mientras que `orquestador.py` selecciona la función, el dominio y los valores predeterminados apropiados.

## Instalación y ejecución

Desde la raíz del repositorio, instala las dependencias del proyecto:

```bash
python -m pip install -r requirements.txt
```

La Parte 1 necesita NumPy y Matplotlib. Para abrir el menú interactivo:

```bash
python "Parte 1/orquestador.py"
```

El menú permite elegir un método, editar sus parámetros y ejecutarlo con Enter. La configuración se conserva mientras el proceso siga abierto. La opción de configuración global permite cambiar la función, la dimensión, los límites, la semilla y la visualización.

También se puede ejecutar directamente desde la consola:

```bash
python "Parte 1/orquestador.py" gd --funcion rastrigin --d 2 --x0 3 -2 --iter-gd 250
python "Parte 1/orquestador.py" gdnum --funcion schwefel --d 3 --seed 12 --gif
python "Parte 1/orquestador.py" evo --tipo-mutacion gaussiana --N 80 --iter 300
python "Parte 1/orquestador.py" comparar --funcion ambas --d 2 --repeticiones 30 --seed 0
python "Parte 1/orquestador.py" limpiar
```

### Comandos

| Comando | Acción |
|---|---|
| `gd` | Descenso por gradiente analítico. |
| `gdnum` | Descenso por gradiente con diferencias finitas. |
| `evo` | Algoritmo evolutivo; admite mutación uniforme o gaussiana. |
| `pso` | Optimización por enjambre de partículas. |
| `de` | Evolución diferencial `DE/rand/1/bin`. |
| `comparar` | Ejecuta varias corridas de los métodos y variantes, presenta estadísticas y guarda un diagrama de cajas. Requiere al menos 30 repeticiones. |
| `limpiar` | Elimina los archivos `.png` y `.gif` directamente dentro de `Parte 1/salidas/`. |

Si se omite el comando y no se proporcionan argumentos, se abre el menú. `--funcion ambas` ejecuta el comando por separado para Rastrigin y Schwefel, aplicando a cada una sus límites y valores predeterminados.

### Parámetros de consola y del menú

Los parámetros siguientes están definidos en `orquestador.py`. Los nombres de consola se forman con `--` y guiones, por ejemplo `iter_gd` se escribe `--iter-gd`. El menú presenta los parámetros relevantes para cada comando; la línea de comandos admite los parámetros de la tabla.

| Parámetro | Predeterminado | Uso |
|---|---:|---|
| `funcion` | `rastrigin` | `rastrigin`, `schwefel` o `ambas`. |
| `d` | `2` | Número de variables. Los GIF admiten únicamente `d=2` o `d=3`. |
| `lb`, `ub` | Según la función | Límites inferior y superior; Rastrigin usa `[-5.12, 5.12]` y Schwefel `[-500, 500]`. Se pueden sobrescribir. |
| `seed` | `0` | Semilla aleatoria. La corrida `k` en `comparar` usa `seed + k`; una misma semilla y configuración reproducen la corrida. |
| `no_plot` | `False` | `--no-plot` evita mostrar las figuras en pantalla, pero estas se siguen guardando. También evita abrir el GIF en una ventana. |
| `gif` | `False` | `--gif` genera una animación de la búsqueda, además de la figura estática. |
| `x0` | Aleatorio | Punto inicial para `gd` y `gdnum`; debe tener `d` coordenadas. En consola: `--x0 1.0 -2.0`. |
| `iter_gd` | `250` | Máximo de actualizaciones de los métodos GD. |
| `lr` | Según la función | Tasa de aprendizaje. Predeterminada: `0.002` para Rastrigin y `1.0` para Schwefel. |
| `tol` | `1e-5` | GD se detiene cuando la norma del cambio entre puntos es menor que esta tolerancia. |
| `h` | `1e-3` | Tamaño de paso para aproximar el gradiente en `gdnum`. |
| `N` | `50` | Tamaño de población o de enjambre. Para DE debe ser al menos `4`. |
| `iter` | `200` | Generaciones de `evo` y `de`, o iteraciones de `pso`. |
| `elitismo` | `0.1` | Fracción nominal de élites del evolutivo. |
| `mutacion` | `0.2` | Fracción de individuos mutados en el evolutivo. |
| `tipo_mutacion` | `uniforme` | `uniforme` reemplaza una coordenada por un valor aleatorio; `gaussiana` perturba una coordenada según la dispersión de la población y recorta al dominio. |
| `w` | `0.7` | Coeficiente de inercia de PSO. |
| `c1`, `c2` | `1.5`, `1.5` | Coeficientes cognitivo y social de PSO. |
| `frac_vmax` | `0.2` | Fracción del rango del dominio que determina la velocidad máxima de PSO. |
| `F` | `0.8` | Factor de mutación de DE; el algoritmo valida que esté entre `0` y `2`. |
| `CR` | `0.9` | Probabilidad de cruce de DE; el algoritmo valida que esté entre `0` y `1`. |
| `repeticiones` | `30` | Número de corridas independientes por método en `comparar`; debe ser al menos `30`. |
| `umbral` | Según la función | Un resultado cuenta como éxito si `f_final < umbral`. Predeterminados: `0.1` para Rastrigin y `1.0` para Schwefel. |
| `costo_grad` | `2*d` | Peso asignado a cada evaluación de gradiente analítico en la métrica equivalente de `comparar`: `evaluaciones_f + costo_grad * evaluaciones_gradiente`. |

Para los parámetros booleanos, la línea de comandos los activa incluyendo la opción, por ejemplo `--gif` o `--no-plot`. En el menú se editan con respuestas afirmativas o negativas.

### Resultados

Las salidas se guardan en `Parte 1/salidas/`, una ruta calculada respecto al propio archivo del orquestador; por ello no depende del directorio desde el que se ejecute el comando. Se crea al guardar la primera figura o GIF.

| Salida | Nombre |
|---|---|
| Figura de convergencia y, si `d=2`, contorno de la búsqueda | `<metodo>_<funcion>_d<d>.png` |
| Animación, si se activa `gif` | `<metodo>_<funcion>_d<d>.gif` |
| Comparación de corridas | `comparar_<funcion>_d<d>.png` |

La figura PNG se guarda aunque se use `--no-plot`; esa opción solo controla su visualización. El GIF puede tardar más que la figura porque conserva cuadros de la trayectoria o de las poblaciones. El comando `limpiar` borra los PNG y GIF de esa carpeta, no archivos JSON ni subcarpetas. En el menú solicita confirmación; como comando de consola borra sin preguntar.

La comparación incluye GD analítico y numérico, evolutivo con mutación uniforme y gaussiana, PSO y DE. Presenta media, desviación estándar, mejor y peor fitness, porcentaje de éxito, evaluaciones de la función objetivo y del gradiente, costo equivalente y tiempo. El contador no incluye las evaluaciones realizadas únicamente para dibujar las curvas.

## API de optimización (`funciones.py`)

Las funciones siguientes son independientes de la función objetivo, siempre que esta reciba un vector individual y devuelva un valor escalar. En los métodos de población, cada fila de una matriz representa un individuo. Los métodos minimizan: un fitness menor es mejor.

### Descenso por gradiente

| Función | Parámetros y comportamiento |
|---|---|
| `grad_desc(x0, grad_func, max_iter=250, learning_rate=0.002, tolerancia=1e-5, limites=None)` | Usa el gradiente analítico `grad_func`; `x0` es el punto inicial; `limites=(LB, UB)` recorta cada nuevo punto si se proporciona. Devuelve la lista de puntos visitados. |
| `dif_marginal(x, f, i=0, h=0.001)` | Aproxima la derivada parcial de `f` en la coordenada `i` mediante diferencia hacia adelante. `h` es la perturbación. |
| `gradiente_numerico(x, f, h=0.001)` | Calcula una diferencia parcial por coordenada y devuelve un vector con la aproximación del gradiente. |
| `grad_desc_num(x0, f_obj, max_iter=250, learning_rate=0.002, tolerancia=1e-5, h=0.001, limites=None)` | Desciende usando `gradiente_numerico`; `h` controla la diferencia finita y `limites` recorta los puntos actualizados. Devuelve la trayectoria. |
| `historial_fitness(puntos, f)` | Evalúa la función objetivo `f` en cada punto de una trayectoria; devuelve un array con los valores para el análisis de convergencia. |

El recorte de `limites` ocurre después de calcular la actualización. En GD numérico, las perturbaciones usadas para estimar el gradiente no se recortan; si `f` no está definida fuera del dominio, debe tenerse en cuenta al elegir el punto y `h`.

### Algoritmo evolutivo

| Función | Parámetros y comportamiento |
|---|---|
| `pob_inicial(N=30, d=5, LB=None, UB=None)` | Genera `N` vectores de dimensión `d` uniformemente en los límites. Si se omiten, usa `[-1, 1]` por coordenada. |
| `pob_eval(Pob, f_obj)` | Evalúa cada fila con `f_obj` y devuelve una copia de la población y su fitness ordenados de menor a mayor. |
| `calcular_conteos(N, fr_elitismo, fr_mutacion)` | Calcula el número de élites, cruces y mutantes; fuerza un número par de descendientes de cruce para formar parejas. |
| `elitismo(Pob, n_elite)` | Copia las primeras `n_elite` filas de una población ya ordenada. |
| `cruzamiento(Pob_cruce)` | Cruza por un punto pares de individuos, emparejando filas de la primera mitad con las de la segunda. Se espera un número par de padres. |
| `mutacion(Pob, id_mutantes=None, coord_mut=None, LB=None, UB=None)` | En cada fila elegida en `id_mutantes`, reemplaza la coordenada indicada por `coord_mut` (o una aleatoria) con un valor uniforme en los límites. Devuelve las filas mutadas. |
| `mutacion_gaussiana(Poblacion, frac_m, lim_inf, lim_sup)` | Elige `int(N*frac_m)` individuos sin reemplazo y perturba una coordenada por individuo con una normal escalada por la desviación estándar de esa coordenada; recorta el resultado a los límites. |
| `next_gen(Pob, fr_elitismo=0.2, fr_mutacion=0.1, LB=None, UB=None, tipo_mutacion="uniforme")` | Construye una población nueva con élites, cruces y mutantes. `tipo_mutacion` admite `uniforme` o `gaussiana`. Requiere una población ordenada por fitness. |
| `actualizar_mejor(Pob, fitness, best_sol, best_fit)` | Compara el primer individuo (el mejor de la población ordenada) con el mejor histórico y conserva el mejor de ambos. |
| `mi_algoritmo_evolutivo(f_obj, d, LB, UB, N=50, max_gen=200, fr_elitismo=0.1, fr_mutacion=0.2, tipo_mutacion="uniforme", semilla=None, guardar_poblaciones=False)` | Ejecuta el evolutivo y devuelve `(mejor_solución, mejor_fitness, historial)`. Si `guardar_poblaciones=True`, añade al retorno la lista de poblaciones de la generación inicial a la final. `f_obj`, `LB` y `UB` se suministran para cada problema. |

Los porcentajes de elitismo y mutación determinan cuántas filas se reservan a cada operador. El resto se asigna al cruce, ajustando el conteo para que el número de padres sea par. Las fracciones deben elegirse de modo que los conteos sean válidos para el tamaño de población.

### PSO

| Función | Parámetros y comportamiento |
|---|---|
| `vel_inicial(N, d, v_max)` | Inicializa las velocidades de `N` partículas dentro de `[-v_max, v_max]`. |
| `evaluar_enjambre(Pob, f_obj)` | Evalúa las posiciones sin reordenar filas, porque cada fila corresponde a una partícula concreta. |
| `actualizar_velocidad(vel, pos, pbest_pos, gbest_pos, w, c1, c2, v_max)` | Combina inercia, mejor posición propia y mejor posición global; limita la velocidad a `[-v_max, v_max]`. |
| `actualizar_posicion(pos, vel, LB, UB)` | Suma velocidad a posición y recorta el resultado al dominio. |
| `actualizar_pbest(pos, fitness, pbest_pos, pbest_fit)` | Actualiza la mejor posición histórica individual cuando mejora estrictamente el fitness. |
| `actualizar_gbest(pbest_pos, pbest_fit, gbest_pos, gbest_fit)` | Actualiza la mejor posición global si alguna memoria individual la supera. |
| `mi_pso(f_obj, d, LB, UB, N=40, max_iter=200, w=0.7, c1=1.5, c2=1.5, frac_vmax=0.2, guardar_poblaciones=False)` | Ejecuta PSO y devuelve `(mejor_posición, mejor_fitness, historial)`. Con `guardar_poblaciones=True`, añade el enjambre inicial y el de cada iteración. |

### Evolución diferencial

`evolucion_diferencial(f_obj, d, LB, UB, N=50, max_gen=200, F=0.8, CR=0.9, semilla=None, guardar_poblaciones=False)` implementa `DE/rand/1/bin`: genera un vector mutante usando tres individuos distintos, aplica cruce binomial y conserva el candidato si su fitness no empeora. Recorta los mutantes al dominio y garantiza que al menos una coordenada proceda del mutante.

Devuelve `(mejor_solución, mejor_fitness, historial)`; con `guardar_poblaciones=True` agrega la población inicial y la de cada generación. Valida `d >= 1`, `N >= 4`, `max_gen >= 0`, `0 <= F <= 2`, `0 <= CR <= 1` y que todos los límites inferiores sean menores que los superiores.

### Visualización

| Función | Parámetros y comportamiento |
|---|---|
| `dibujar_contorno(ax, f, LB, UB, n=150)` | Dibuja sobre `ax` el contorno de una función vectorizada en una malla de `n × n`; se usa para problemas de dos dimensiones. |
| `animar_busqueda(cuadros, f, LB, UB, ruta, titulo="", estela=False, optimo=None, max_cuadros=60, fps=8, mostrar=False)` | Genera un GIF a partir de puntos de una trayectoria o poblaciones sucesivas. Admite `d=2` (contorno) o `d=3` (espacio de variables), submuestrea hasta `max_cuadros` conservando el último cuadro y puede reproducirlo en una ventana. |

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

Distancia, duración y peaje de cada arco se leen de `here_peaje.csv`, es decir, de la misma ruta de HERE. Las distancias se dividen por 1000 y los tiempos por 3600. El consumo y el precio se consideran constantes durante todos los desplazamientos. La diagonal de la matriz económica se fija en cero.

### C.2. Funciones públicas

| Función | Parámetros | Retorno |
|---|---|---|
| `leer_capitales()` | Ninguno. | `(nombres, ubicaciones)`, con coordenadas NumPy de forma `(n, 2)` en orden `[lat, lon]`. |
| `leer_matriz(nombre)` | Nombre de un CSV en `data/`. | Matriz NumPy de valores reales, omitiendo cabecera e identificadores de fila. |
| `leer_here(n, columna, archivo="here_peaje.csv")` | Número de ciudades, columna (`dist_m`, `tiempo_s` o `peaje_eur`) y archivo. | Matriz `(n, n)` con esa columna de HERE, con diagonal cero. |
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

Los tres argumentos son obligatorios en este script. Se imprimen la suma de todos los peajes de la matriz, el número de arcos con peaje positivo, la forma de la matriz, la máxima diferencia entre sentidos opuestos, el costo del circuito por orden de `id` y la diferencia de distancia y tiempo entre HERE y OSRM. La suma de peajes de toda la matriz no es el peaje de un recorrido. No se escribe una matriz económica a disco.

### C.5. Supuestos y limitaciones

Los identificadores deben corresponder a índices consecutivos de `0` a `n-1`. Las matrices deben conservar el mismo orden de ciudades: `leer_matriz()` descarta sus etiquetas sin comprobarlas. El programa no valida explícitamente dimensiones, finitud o signo de todos los parámetros.

La implementación usa distancia, tiempo y peajes de HERE, de modo que las tres componentes de un arco corresponden al mismo trayecto vial (la ruta más rápida en carro, sin tráfico). Hasta el 2026-10-12 la distancia y el tiempo se tomaban de OSRM; se cambió porque OSRM elige otra carretera en algunos arcos (53 arcos difieren más de 10 % en distancia) y eso cobraba peajes de vías que su ruta no recorre. Las matrices de OSRM no entran al costo y quedan como verificación independiente: sus tiempos son en promedio 19 % mayores que los de HERE.

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
| `here_peaje.csv` | Distancia (m), tiempo (s) y peaje (EUR) de la ruta de cada arco dirigido; son los tres datos que usa `costos.py` | Obtenido de la API HERE mediante el módulo obtener_peajes_here | 2026-10-10 |
| `espana.geojson` | Contorno de España (lon, lat) para dibujar el mapa sin conexión a internet | Natural Earth, Admin 0 Countries 1:50m (dominio público), entidad `ADMIN = Spain` de https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson, coordenadas redondeadas a 5 decimales | 2026-10-12 |
Fila = origen, columna = destino. `[i, j]` puede ser distinto de `[j, i]`.

## En caso de querer actualizar los datos de peaje

**Peajes en EUR**: correr `scripts/obtener_peajes_here.py` con una API key de HERE (ver el encabezado del script).

## Vehículo seleccionado

Renault Clio TCe 115 con caja automática (gasolina), consumo mixto WLTP homologado de 5,3–5,5 L/100 km (tomado como 5,5 L/100 km, el extremo superior). La misma página da 5,2–5,4 L/100 km para la caja manual. https://www.renault.es/preguntas-frecuentes-vehiculos/clio/esta-renault-clio-disponible-con-una-caja-de-cambios-manual-o-automatica.html Revisado 2026-10-10

## Scripts

| Programa | Responsabilidad principal |
|---|---|
| `obtener_osrm.py` | Obtener matrices dirigidas de distancia y tiempo por carretera. |
| `obtener_peajes_here.py` | Consultar peajes y guardar respuestas reutilizables por trayecto. |
| `costos.py` | Construir la matriz económica y calcular el costo de circuitos cerrados. |
| `optimizador_colonia_de_hormigas.py` | Implementar el optimizador ACO y registrar su evolución. |
| `colonia_de_hormigas.py` | Integrar datos y optimizador, y exportar resultados y visualizaciones. |
| `operadores_ga.py` | Cruces (OX, PMX) y mutaciones (inversión, inserción, swap, mixta) para permutaciones. |
| `algoritmo_genetico.py` | Algoritmo genético con PyGAD: una corrida, resultados, mapa y GIF. |
| `experimentos_ga.py` | Comparación de operadores con varias corridas y estudio del valor de la hora. |
| `../orquestador.py` | Menú interactivo y línea de comandos del algoritmo genético. |

```bash
pip install requests numpy
python Parte_2/scripts/costos.py --valor-hora 15 --consumo 5.5 --precio 1.849   # matriz de costos y prueba
python Parte_2/scripts/obtener_osrm.py --comparar                                # vuelve a consultar OSRM y compara
python Parte_2/scripts/obtener_peajes_here.py --modo peaje --limite 3            # prueba con 3 arcos
python Parte_2/scripts/colonia_de_hormigas.py --valor-hora 25 --hormigas 100 --intensificacion 1.9 --alfa 2.1 --beta 0.9 --beta-evaporacion 0 --p-mejor 0.09 --version 0030--version 0001
python Parte_2/orquestador.py                                                    # menú interactivo del algoritmo genético
```

## Algoritmo genético

### Orquestador

`Parte_2/orquestador.py` es el punto de entrada del algoritmo genético. Sin argumentos abre un menú interactivo en la consola: se elige el comando, se editan los parámetros por número y se ejecuta con Enter. También acepta los mismos parámetros por la línea de comandos.

```bash
python Parte_2/orquestador.py                                           # menú interactivo
python Parte_2/orquestador.py ga                                        # una corrida de demostración (JSON, mapa y GIF)
python Parte_2/orquestador.py ga --valor-hora 25 --elite 0 --sin-gif    # cambiar parámetros sin editar código
python Parte_2/orquestador.py comparar                                  # 30 corridas por cada pareja cruce + mutación
python Parte_2/orquestador.py hora                                      # estudio del valor de la hora
python Parte_2/orquestador.py limpiar                                   # borra los archivos del GA en outputs/
```

| Comando | Qué hace | Archivos en `outputs/` |
|---|---|---|
| `ga` | Una corrida del GA. | `resultados_GA_<v>.json`, `Recorrido_GA_<v>.png`, `Evolucion_GA_<v>.gif` |
| `comparar` | `repeticiones` corridas por cada pareja cruce + mutación (8 parejas). Imprime la tabla de media, desviación, mejor, peor y tasa de éxito. | `comparar_GA_<v>.png` (diagrama de cajas) |
| `hora` | Corre el GA para cada valor de la hora, imprime la mejor ruta de cada tramo y el valor en que cambia. | `hora_GA_<v>.png` |
| `limpiar` | Borra los archivos con `_GA_` en el nombre. No toca los de otros algoritmos. | |

### Archivo de configuración

Al arrancar, el orquestador lee `Parte_2/config_ejemplo.json`. Cada clave es un parámetro de la tabla siguiente, con el mismo nombre que en la línea de comandos pero con guion bajo (`valor_hora`, `p_mutacion`). Las claves que empiezan por `_` son comentarios. El archivo trae el vehículo, el valor de la hora, el consumo y el precio del combustible, con sus fuentes; se le puede agregar cualquier otro parámetro, por ejemplo `"elite": 0`.

Prioridad: línea de comandos, luego archivo de configuración, luego valores por defecto del código. Una clave desconocida o un valor del tipo equivocado detienen el programa con un mensaje que indica cuál es.

```bash
python Parte_2/orquestador.py --config otra.json ga    # usa otro archivo de configuración
```

### Parámetros

| Argumento | Por defecto | Significado |
|---|---|---|
| `--config` | `config_ejemplo.json` | Archivo de configuración JSON |
| `--vehiculo` | del archivo | Texto informativo; el cálculo usa `consumo` y `precio` |
| `--valor-hora` | del archivo (15) | EUR por hora del vendedor |
| `--consumo` | del archivo (5.5) | L/100 km del vehículo |
| `--precio` | del archivo (1.849) | EUR por litro de combustible |
| `--semilla` | 0 | Semilla aleatoria. La corrida `k` de un experimento usa `semilla + k` |
| `--version` | 0001 | Sufijo de los archivos de salida |
| `--poblacion` | 100 | Tamaño de la población |
| `--generaciones` | 1500 | Número de generaciones |
| `--padres` | 50 | Padres seleccionados por generación (torneo) |
| `--elite` | 2 | Mejores individuos que pasan sin cambios (0 = sin elitismo) |
| `--k-torneo` | 3 | Tamaño del torneo |
| `--cruce` | ox | `ox` o `pmx` |
| `--mutacion` | mixta | `inversion`, `insercion`, `swap` o `mixta` (inversión o inserción, al azar) |
| `--p-mutacion` | 0.3 | Probabilidad de que un hijo mute |
| `--sin-gif` | | No genera el GIF (solo JSON y PNG) |
| `--repeticiones` | 30 | `comparar`: corridas independientes por configuración |
| `--tolerancia` | 2 | `comparar`: éxito si el costo es menor o igual que el mejor del experimento más este porcentaje |
| `--procesos` | mitad de los núcleos | Procesos en paralelo. No cambia los resultados, solo el tiempo |
| `--vh-min`, `--vh-max`, `--vh-paso` | 0, 60, 5 | `hora`: rejilla de valores de la hora (EUR/h) |
| `--rep-hora` | 20 | `hora`: corridas por cada valor de la hora |

Con la misma semilla y los mismos parámetros el resultado se repite, con uno o con varios procesos.

### Conteo de evaluaciones

Una evaluación es un cálculo del costo de un recorrido completo. `evaluaciones_fitness` cuenta las llamadas reales a la función de aptitud; PyGAD no la llama para los élites ni para los hijos idénticos a un élite, porque ya conoce su aptitud. `evaluaciones_nominales` es el tope si se evaluara a todos: `poblacion + generaciones x (poblacion - elite)`.

### Estudio del valor de la hora

El costo de un recorrido es `valor_hora x horas + (peajes + combustible)`: una recta en el valor de la hora. El comando `hora` junta todos los recorridos que encontró el GA con los distintos valores, evalúa cada uno para cualquier valor de la hora y calcula la envolvente inferior de esas rectas. El resultado es la mejor ruta conocida en cada tramo y el valor exacto en que cambia, que es igual al sobrecosto en peajes y combustible dividido por las horas que ahorra. Las rutas son las mejores encontradas por el GA; no hay garantía de que sean óptimas.

### Ejecución directa del algoritmo

`scripts/algoritmo_genetico.py` conserva su propia línea de comandos para una corrida (`--valor-hora` y `--version` son obligatorios):

```bash
python Parte_2/scripts/algoritmo_genetico.py --valor-hora 15 --version 0001
```
