# Evolución de las publicaciones: un punto por día, como el gráfico del promedio

Diseño corto acordado con Luis el 2026-10-30 (chat). Afecta a la tarjeta «Evolución de las publicaciones» de
Promedios (`/{scope}`) y de Escaños (`/{scope}/escanos`).

## Intención y por qué

La tarjeta de hoy (un gráfico de 340 px, todos los partidos, eje desde cero, un punto por run) no deja ver los
cambios entre publicaciones, que son minúsculos: el PP se mueve entre 31,5 y 32,7 sobre un eje de 0 a 35.

Antes de rehacerla se comprobó que aporta algo que el gráfico del promedio no dice. Lo aporta: la curva del
promedio es una regresión de kernel gaussiano, así que el valor de cada día usa también los sondeos publicados
después y **revisa el pasado**; los puntos de la evolución (en vivo o retrospectivos) sólo usan los sondeos
conocidos ese día. Con los datos de producción del 2026-10-30: la curva de hoy dice que el PP bajó 0,2 en
octubre (31,7 → 31,5); lo que publicamos dice que bajó 1,2 (32,7 → 31,5). Sólo coinciden en los últimos días,
cuando aún no hay sondeos posteriores. Eso es lo que la tarjeta debe contar: lo que publicamos cada día.

Dos hallazgos que cambian la tarjeta:

- Los valores son idénticos en los dos modos (el voto del pronóstico es el de la estimación y, desde M13, los
  escaños también): el subtítulo «estimación/pronóstico de cada publicación» promete una diferencia que no existe.
- En Escaños los escaños por publicación no aparecen en ningún otro sitio y sí se mueven (PP 133 a 141).

## Objetivos

- Un gráfico como el del promedio: misma altura (`chart-tall`), vista Partidos/Bloques con selector, leyenda de
  pastillas, eje que sigue a lo seleccionado (el «zoom» se hace marcando partidos), botón de descarga.
- Un punto por día, los puntos ligados por una línea, tooltip al pasar por cada día.
- Escaños: el mismo gráfico con escaños.
- Un «Cómo leerlo» que explique la revisión: por qué estos puntos no son la curva del promedio.

## No objetivos y anti-objetivos

- No dibujar la curva revisada del promedio debajo de los puntos (Luis: sólo los puntos ligados).
- No añadir selector de ventana ni guardar la selección en el hash de la URL.
- No cambiar el modelo ni lo que se publica (`history.json` sigue igual; los bloques se suman en el navegador).
- Anti-objetivo: que el lector lea los puntos como una serie de sondeos o como el promedio.

## Restricciones

- Los titulares de los runs (`history.json`, parte `headline`) llevan `parties` con `pct` y `seats`, sin bloques.
- Los retrospectivos llevan `run_at` a las 12:00 convencionales y `backfill: true`; pueden convivir con un run en
  vivo el mismo día.
- Reutilizar las piezas del promedio: `legend-dom.js` (pastillas), `series-wire.js` (`fillLegend`,
  `chosenNames`), `blocks.js` (`REST_BLOCK`), `share.js` (`wireDownload`), `Catalog` (`bmaps.main`,
  `blockColor`, `orderBlocks`), `initial.inks` para las tintas de las pastillas.

## Enfoques

1. **Módulo hermano del promedio (elegido).** Reescribir `charts/evolution.js` como un gráfico de líneas con
   puntos y leyenda oculta gobernada por pastillas, con un cableado compartido `pages/evolution-wire.js` que usan
   las dos páginas. Reutiliza las piezas del promedio sin tocarlas. Trade-off: un módulo más que mantener.
2. Meter los runs como una capa de `renderSeries`. Descartado: `series.js` gira en torno a sondeos, promedio y
   proyección; forzar los runs ahí complica el módulo más usado del sitio.
3. Retocar el `evolution.js` actual (eje ajustado, más alto). Descartado: no da la vista de bloques ni las
   pastillas, y seguiría contando una cosa distinta en cada modo.

## Decisiones

- **Un punto por día.** Los runs se agrupan por su `as_of` (día que leyó el modelo; a falta de él, el día de
  `run_at`). Un run en vivo gana a uno retrospectivo; entre varios en vivo, el último `run_at`. (Agrupar por
  `as_of` y no por la hora de `run_at` es decisión mía: evita que un run de madrugada caiga en otro día.)
- **Puntos.** Los retrospectivos siguen huecos (borde del color, relleno blanco); los en vivo, llenos. Línea de
  2 px uniendo los puntos de cada nombre.
- **Tooltip por día** (eje): fecha (fecha y hora si el run es en vivo, «· estimación retrospectiva» si no) y el
  valor de cada nombre visible, de mayor a menor, como ahora.
- **Vistas.** Partidos: los partidos del titular, colores del catálogo, marcados por defecto los de `bmaps.main`
  (los mismos que el promedio), todos si no hay. Bloques (sólo si el ámbito tiene `bmaps.blocks`): cada bloque
  suma sus partidos y Otros (`REST_BLOCK`) es el resto hasta 100 (hasta la cámara en escaños), en el orden
  `orderBlocks`, colores `blockColor`; todos marcados por defecto. El cambio de vista reinicia la selección.
- **Eje vertical** sin cero fijo: mínimo y máximo redondeados a puntos enteros alrededor de lo visible (no a
  múltiplos de 5 como el promedio), con al menos un punto de recorrido; marcas enteras en escaños. El PP solo
  va de 31 a 33 y su bajada ocupa más de la mitad del alto.
- **Eje de fechas** con todos los días publicados; deslizador de zoom abajo como en el promedio.
- **Un solo gráfico para ambos modos:** se dibuja el titular `nowcast` siempre y el subtítulo de la tarjeta deja
  de nombrar el modo («· voto de cada publicación» / «· escaños de cada publicación»).
- **Descarga** como el promedio: el gráfico tal cual se ve (nombres marcados, zoom), título del evento, subtítulo
  de la tarjeta y firma; fichero `demoscopio-evolucion-{scope}-{fecha}.png` (`-escanos-` en Escaños).
- **Cómo leerlo** (ambas páginas, misma idea): cada punto es lo que el modelo publicó ese día con los sondeos
  conocidos hasta entonces; los huecos son retrospectivos (el modelo de hoy con los sondeos de aquel día); la
  curva del promedio revisa el pasado con los sondeos que llegan después, por eso no coinciden; los puntos se
  mueven porque entran sondeos, mira la tendencia; no es una serie de sondeos; pulsa la leyenda para ampliar.
  Sin cifras vivas. Pasa `assert_public_wording`.
- **Sin selector de ventana ni hash.** La selección vive en el gráfico, como la de la encuestadora.

## Lo que queda al constructor

- `charts/evolution.js`: `dailyRuns(runs)` (la regla de un punto por día, pura y probada), `evolutionOption(data)`
  (opción ECharts: `{runs, parties, selected, colors, field, formatter, legend}`), `evolutionImageOption` y
  `renderEvolution`. Bloques: `blockRuns(runs, blocks, field, total)` en `blocks.js`, con la misma regla de
  Otros que `blockPolls`.
- `pages/evolution-wire.js`: `wireEvolution(initial, catalog, {field, formatter, ids})` compartido por
  `promedios.js` y `escanos.js`; selector `#evolution-view` (radios `evview`), leyenda `#evolution-legend`,
  gráfico `#evolution`/`#seats-evolution`, botón `#evolution-download`.
- Plantillas: controles, leyenda, `chart chart-tall`, botón de descarga, nuevos subtítulos y «Cómo leerlo».
  Escaños necesita `inks` en sus datos iniciales (la lista de claves pinada en `tests/test_pages.py` cambia).
- Pruebas: `node` para `dailyRuns`, `blockRuns` y el eje de `evolutionOption`; de página para el marcado, los
  subtítulos sin modo y el texto; `assert_public_wording` en los textos nuevos.
- Sin cambios en Python salvo `inks` en Escaños. Sin republicar nada.

## Estado al cierre (2026-10-30)

- Construido en la rama `web-evolucion-publicaciones` (worktree `.worktrees/`, desde dev `bab1720`), plan
  `docs/superpowers/plans/2026-10-30-web-evolucion-publicaciones.md`: `blockRuns` (b2b3638), `charts/evolution.js`
  (e47cccc), `pages/evolution-wire.js` (201dad4), plantillas, `escanos.py` y páginas (520c8a3), círculos rellenos
  (84d57ff). Suite completa en verde tras la tarea 4: 1478 pasan, 76 saltadas; 7 pruebas nuevas en
  `tests/test_js_evolution.py`, 3 en `tests/test_js_blocks.py`, prueba de página reescrita.
- Comprobado en navegador (Chrome headless por CDP sobre un bundle local con el histórico de producción de 26
  runs): un punto por día, eje de 1 a 33 por defecto y de 26 a 27 con el PSOE solo, vista Bloques, descarga PNG de
  1600 × 1144 con la leyenda de lo marcado, sin errores de consola. Hallazgo arreglado: los puntos retrospectivos
  se dibujaban como huecos en la línea (el símbolo por defecto `emptyCircle` traza con `color`); ahora
  `symbol: 'circle'`, con prueba.
- Revisión independiente de toda la rama: sin críticos; corregido el docstring de `escanos_context`. Menores
  diferidos: (a) la vista Bloques a través de un cambio de nombre de partido (SUMAR → FA): los runs anteriores al
  cambio no tienen fila del nombre nuevo y esa parte cae en Otros, un escalón artificial; hoy latente porque todo
  el histórico se republicó el 30-10 con la candidatura actual, y una solución real necesita el linaje del smap;
  (b) con un run anclado (`?run=`), el fichero y el subtítulo de la descarga llevan la fecha del run anclado
  mientras el gráfico dibuja todo el histórico, como los demás botones de descarga; (c) `charts/evolution.js`
  importa tres ayudas de `series.js` (Escaños carga ahora ese módulo): subirlas a `base.js` como seguimiento.
- Queda a Luis: commitear su lote del árbol principal, fusionar la rama en dev (avance rápido) y desplegar. Nada
  que republicar.
