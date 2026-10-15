# Web, fase 7f: dianas, detalle de una elección y ranking

Fecha: 2026-10-15. Estado: cerrada el mismo día en `dev`.

## Qué pidió Luis

1. En todas las dianas: los puntos de las elecciones generales por encima de los demás (z-index mayor), puntos algo
   más pequeños y las dos etiquetas de los ejes, "Error Derecha" (X) y "Error Izquierda" (Y).
2. En la página de una elección (`/elecciones?scope=&date=`), reordenar los bloques: primera fila como en Promedios
   (hemiciclo a la izquierda, tabla de resultados a la derecha, sin "P(primero)"); segunda fila, las dos columnas del
   cuaderno `PollstersEvent` ("Desviación entre el porcentaje de votos estimados por las encuestas y los resultados"):
   a la izquierda las barras de "Error del último sondeo" y a la derecha el reparto entre Izquierda y Derecha; después
   la diana centrada, debajo la lista "Último sondeo de cada encuestadora" y el resto como estaba.
3. En Ranking Encuestadoras: primero la tabla y después las barras; la tabla al estilo de las de encuestas de
   Promedios y con los formatos de `Computer.print_ratings` del cuaderno `PollstersRatings`: barra de 0 a 100 en
   "Rating", barra roja a la izquierda o verde a la derecha (entre −2 y +2) en "Sesgo del margen" y un gradiente en
   "Error medio" y "Rebaño".

Diseño y UX quedan para después ("Aplica esto y luego ya hablamos de diseño y UX").

## Decisiones

- **Diana** (`charts/bullseye.js`): los ejes dejan el cero (`axisLine.onZero: false`) y van a los bordes, cada uno con
  su nombre "Error {bloque}" (`nameLocation: 'middle'`, en seminegrita), sin las etiquetas de los extremos (el alcance
  no es redondo); la cruz del cero se dibuja como dos series de línea, tras los tres anillos. Puntos de 8 px (antes
  10). Nueva opción `onTop` (`grupo => booleano`): las series de esos grupos llevan `z: 12` frente a `z: 10`, así sus
  puntos y etiquetas se pintan sobre los demás. La opción se construye en `bullseyeOption(rows, opts)`, exportada, y
  `renderBullseye` solo la monta: así `node` puede probarla. Usos: en el perfil, `onTop` = Generales; en el histórico
  con "Todos", el ámbito nacional (`es`, por su nombre); con un ámbito elegido no hay prioridad (todas las elecciones
  son del mismo ámbito); la diana de una elección tiene una sola serie.
- **Detalle de una elección**: orden de tarjetas: cabecera, fila `.hemi-grid` (Hemiciclo | Resultados por partido),
  fila `.split` (Error del último sondeo | Error del último sondeo por bloque), La diana, Último sondeo de cada
  encuestadora (con el glosario), Error de cada sondeo, Todas las encuestas, El modelo a cada horizonte, Descargas.
  - La cabecera (`#event`, fecha · nombre y la línea de cifras) se conserva encima: es el título de la página, el
    equivalente de la barra de evento de Promedios, no un bloque de análisis.
  - **Hemiciclo**: `renderHemicycle` con el resultado oficial, mayoría absoluta `n_seats // 2 + 1`
    (`pages.majority_of`), partidos ordenados por bloque y catálogo como en Promedios (`Catalog.order` del navegador).
    Para eso `initial.catalog.parties` pasa a ser la unión de los partidos de la tabla y de los resultados
    (`pages.chart_parties`: los del catálogo del run primero, en su orden, y el resto por aparición); `initial.results`
    lleva `n_seats`, `majority` y `rows` (`name`, `seats`). Sin resultado publicado, aviso en lugar del gráfico.
  - **Resultados por partido**: columnas Partido, % Voto (filtro `pct`) y Escaños (en negrita), como la tabla de
    Promedios sin "P(primero)"; conserva "Ver más" a partir de 20 filas.
  - **Reparto por bloque**: gráfico nuevo `charts/diverging.js` (`divergingOption`/`renderDiverging`): dos series de
    barras apiladas desde el cero, la de la izquierda en negativo y la de la derecha en positivo, ambas como magnitud
    (sin signo, como `plot_diverging` del cuaderno), valor al final de cada barra, leyenda con los dos bloques, eje
    simétrico que se lee en valor absoluto y línea del cero (`markLine`); los nombres de las casas en el borde
    izquierdo (`yAxis.axisLine.onZero: false`), no sobre el cero. Mismo orden de filas que las barras (de menor a mayor
    error de bloques). Qué bloque va a cada lado lo decide el servidor (`pages.split_sides`: "Izquierda" a la
    izquierda cuando es el primer bloque; si no, el segundo bloque a la izquierda y el primero a la derecha, como el
    cuaderno), en `split_blocks` del contexto e `initial.split`; los colores de bloque salen de los `bmaps` del run
    (`block_color`, `initial.block_colors`; sin partido en el mapa, el gris de los demás o el azul neutro).
  - La diana no cambia de sitio en el código (`#bullseye-chart.big`, 720 px, centrada); los textos "la tabla de
    arriba" pasan a "la tabla de abajo".
- **Ranking**: la tabla va antes que las barras (`#rating-chart`). Marcado `table.et.et-rank` dentro de `.et-wrap` (la
  misma base que "Todas las encuestas": diseño fijo, primera columna fija y en negrita, `colgroup`), con celdas de
  tres tipos, calculadas en el servidor por tres filtros nuevos de `pages.filters` (`bar`, `sides` y `shade`; sin JS):
  - `cell-bar` (Rating): `<i class="fill pos">` de anchura `bar(0, 100)` (verde claro, como `green-light` de
    `print_ratings`) bajo el valor.
  - `cell-bar cell-diverging` (Sesgo del margen): la barra parte de la línea central (`::before`) hacia la izquierda en
    rojo (negativo) o la derecha en verde (positivo), saturada en ±2 (`sides(2)`).
  - `cell-shade` (Error medio y Rebaño): fondo `rgba()` con una opacidad que crece con el valor (`shade(inicio, fin,
    color)`, tope 0,55): el error medio de 0 al mayor de la columna (`error_max` del contexto, como el `vmin=0` y el
    máximo de los datos de `background_gradient`) en rojo; el rebaño en rojo de 1 hacia 0 y en gris de 1 hacia 2,
    siguiendo las barras de `print_ratings` (rojo por debajo de 1, gris por encima). Sin valor, transparente.
  Los filtros se exportan en `pages` y se registran en `environment()`.
- Los colores de las celdas son los del cuaderno (`dataviz.colors`: rojo `#d62728`, verde `#2ca02c`, gris
  `#7f7f7f`), rebajados con opacidad para que el texto siga leyéndose.

## Comprobación

- `tests/test_js_charts.py` (nuevo, con `node`): `bullseyeOption` (nombres y posición de los ejes, cruz del cero,
  tamaño y `z` de los puntos, `onTop`) y `divergingOption` (lados, magnitudes, colores, línea del cero, eje
  simétrico, nombres en el borde). El arnés carga el ECharts empaquetado antes de definir `window`, para que zrender
  se sepa en node.
- `tests/test_pages.py`: orden de las tarjetas del detalle, `initial.results`/`split`/`block_colors`, catálogo
  ampliado, columnas de la tabla de resultados; tabla del ranking antes que las barras y las celdas con sus barras y
  sombreados; pruebas unitarias de `bar_width`, `bar_sides`, `shade`, `majority_of`, `split_sides` y
  `chart_parties`. 758 pruebas (75 de integración saltadas).
- Render a SVG con `node` (ECharts en modo servidor) de las tres dianas y del gráfico divergente, y capturas con
  Chrome sin cabeza de `/elecciones?date=`, `/casas`, `/casas?id=` y `/elecciones` con los fixtures: sin errores de
  consola.

## Ajustes 2026-10-15 (segunda tanda)

Luis, tras recorrer la primera tanda ("Está muy bien"):

1. **Ranking solo con historial**: "muestra sólo las que tengan al menos 2 encuestas en al menos 2 elecciones", en la
   tabla y en las barras. Los ratings traen `num_polls`/`num_events` acumulados, no por elección, y la parte
   `poll-errors` del análisis solo lleva el último sondeo de cada casa por elección (`n_last: 1`), así que el recuento
   sale de la base: consulta nueva `data.get_poll_counts()` (`elections.polls` agrupado por `pollster_id`,
   `event_scope` y `event_date`, contando los sondeos con `bias`, los que entran en los ratings; 1.155 filas en la base
   real), servida por `webapi.query_poll_counts()` (caché de consultas). `pages.eligible_pollsters(counts)` da los ids
   con al menos `RANKING_MIN_POLLS` (2) sondeos en al menos `RANKING_MIN_EVENTS` (2) elecciones de cualquier ámbito y
   `ranking_rows` descarta el resto (73 de las 125 encuestadoras con sondeos cumplen). El perfil y los enlaces de los
   demás paneles no cambian. La leyenda de la tabla y el "Cómo leerlo" lo dicen.
2. **Detalle de una elección**: el selector Partidos/Bloques de Promedios en la primera fila (`#results-view` en la
   tarjeta del hemiciclo; cambia el hemiciclo y muestra "Resultados por bloque", `#results-blocks`, oculta por
   defecto, con Bloque, % Voto y Escaños sumados). El bloque de cada partido sale de la primera fuente que lo lista
   (`pages.party_blocks`): `bmaps.blocks` de los parámetros de la elección, `blocks` (vs) del `meta` publicado, el
   catálogo del run; sin bloque, el partido no suma. La tabla de partidos enseña solo los de `parties.event` de los
   parámetros de la elección en `data/params.json` (`webapi.event_params(scope, date)`, leído del `data` del App con
   caché TTL; `pages.shown_parties`/`mark_shown`) y el resto tras "Ver los N restantes"; sin lista, o si ninguno de
   sus partidos está en ella, los 15 primeros (`RESULTS_SHOWN` baja de 20 a 15). La parte `results` publicada no
   cambia.
3. **Dianas**: los ejes se llaman "Derecha" e "Izquierda" (sin "Error "), y el tooltip solo se abre sobre los puntos:
   las etiquetas van en una serie gemela invisible y silenciosa de cada serie de puntos (mismo nombre, para que la
   leyenda apague las dos; `itemStyle.opacity: 0`, `silent: true`, `tooltip.show: false`), porque en ECharts la
   etiqueta de un punto responde al ratón como el punto. Los puntos atenuados (`alphaOf`) llevan su opacidad en la
   serie de puntos y el color de su etiqueta en la gemela.

Comprobación: `tests/test_js_charts.py` (gemelas, nombres de los ejes), `tests/test_pages.py` (ranking filtrado,
`event_params`, partidos mostrados, bloques, selector y tabla por bloques; la caché de parámetros se vacía por test
como la de consultas), 761 pruebas; consulta de recuentos ejecutada una vez contra la base real; capturas con Chrome
sin cabeza de las dos vistas del detalle, del ranking y del perfil, sin errores de consola.
