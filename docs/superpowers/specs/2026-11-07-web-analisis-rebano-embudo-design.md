# Web, Análisis del rebaño: el embudo por semanas en la pregunta 2

Añadido a la versión 2 del estudio `rebano` (`2026-11-06-web-analisis-rebano-v2-design.md`, fusionada en dev
ffb5c39 y publicada el 2026-11-06). Aprobado por Luis el 2026-11-07 («Ok»).

## Intención

Luis quiere recuperar en la tarjeta de la pregunta 2 («¿Es mayor según se acerca la elección?») el gráfico de líneas
de la versión 1: la dispersión por tramos de semanas hasta la elección, una línea por elección general, la media
discontinua y 2026 en azul, que avanza con cada actualización. Los cuadritos sólo enseñan dos saltos; el embudo
cuenta todo el recorrido. Decisiones de Luis: **las generales desde 2008, como antes, y un color por elección** en
lugar de las líneas grises.

## Hechos de partida (2026-11-07, sobre el parquet del spike)

Con la medida de la versión 2 (pares de sondeos de casas distintas a ≤ 7 días, `R = √(Σd²/Σv)` de todas las
comparaciones del tramo) y los tramos de abajo, las 7 generales pasadas dibujan: media 1,65 (27-52 sem.) → 1,31 →
1,42 → 1,42 → 1,35 → 1,19 → 1,07 (0-2 sem.); 2015 de 3,0 a 1,4; 2016 acaba en 0,70; abril de 2019 plana (1,3-1,8).
2026 tiene parejas desde el principio hasta el tramo de 3-4 semanas.

## Objetivos y límites

- El gráfico usa la medida de la versión 2, no la curva del promedio: es descriptivo (todas las parejas del tramo),
  mientras los recuentos de la tarjeta comparan las mismas parejas. La tarjeta lo dice.
- No cambia ningún recuento, veredicto ni subtítulo. No hay selectores. La imagen de la tarjeta incluye el gráfico.
- Un tramo con menos de dos parejas de sondeos distintas no se dibuja (la lección del 3,84 de la versión 1).
- Nada de «se copian», «mirar al vecino», «permite el azar», ni explicaciones de intenciones.

## Enfoques considerados

- **Líneas ECharts, recuperando `weeks.js` de la versión 1 (git c3aa20e) con la medida nueva** (elegido): el
  gráfico que Luis quiere, y la descarga de tarjeta ya redibuja los gráficos ECharts (sesgo del CIS).
- Barras HTML como el resto de la página: no dibuja curvas; descartado.
- Líneas indexadas a «lejos = 1»: contesta la pregunta 2 pero no es el embudo de antes; descartado.
- Todas las elecciones (45) o las 20 con cinco tramos: Luis eligió las generales desde 2008.

## D1. Parámetros y cálculo (`data/rebano/params.json`, `mtpy/lib/studies/herd/`)

- `params.json` gana `"weeks": {"scope": "es", "since": "2008-01-01", "min_pairs": 2, "buckets": [[0, 2], [3, 4],
  [5, 7], [8, 11], [12, 17], [18, 26], [27, 52]]}`. Un tramo `[lo, hi]` en semanas cubre los días `lo·7 ≤ días <
  (hi + 1)·7` de la comparación (la media de los días de sus dos sondeos): así `0-2` es el tramo final de la
  pregunta 2 (0-20 días) y `3-4` + `5-7` el de 3-8 semanas (21-55 días). Fila de `data/README.md` actualizada.
- `stats.py`: `bucket_of(days, buckets) -> Optional[tuple[int, int]]` y `by_buckets(rows, buckets, min_pairs) ->
  list[dict]`: por tramo, `{lo, hi, pairs, comparisons, ratio}` con `pairs` = parejas de sondeos distintas
  (`poll_i`, `poll_j`), `comparisons` = filas y `ratio` = `dispersion` de las filas del tramo sin redondear, o `None`
  si `pairs < min_pairs`.
- `study.py`: `HerdStudy.weeks = {'buckets': [{lo, hi, label}], 'rows': [...], 'mean': [...]}`. Entran las
  elecciones pasadas con `scope == weeks.scope` y `event_date >= weeks.since`, y la elección en curso si su ámbito es
  ese. `label` del tramo: `'{lo}-{hi}'`. `rows`: una fila por elección y tramo, `{event_date, label, current, lo,
  hi, pairs, comparisons, ratio}` con `ratio` a 2 decimales. `mean`: por tramo, `{lo, hi, k, ratio}` con la media
  sin redondear de los cocientes de las elecciones pasadas con cociente, redondeada a 2 (`None` sin ninguna).

## D2. Parte publicada (`bundle.py`, `publish/herd.py`, `publish/studies.py`, `pages/analisis.py`)

- `STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current', 'weeks')`; `STUDY_CSV['rebano'] =
  ('elections', 'weeks')`; `STUDY_CSV_OF['rebano']['weeks'] = ('weeks',)`; `SCHEMAS['rebano-weeks'] = {'buckets':
  list, 'rows': list, 'mean': list}`.
- `export_herd_weeks(study) -> (weeks, {'weeks': frame})`: el CSV en forma larga, las filas más las de la media
  (`event_date` nulo, `label` «Media», `current` falso), columnas `event_date, label, current, lo, hi, pairs,
  comparisons, ratio`.
- `CSV_LABELS['rebano']['weeks'] = 'Separación entre casas por elección y tramo de semanas (generales desde 2008 y
  la elección en curso)'`.
- `docs/web/contrato.md`: la parte y el CSV en las tablas de `rebano`.

## D3. La página

- `herd_context` gana `initial = {'weeks': parte weeks, 'state': ...}` (la isla que `base.html` imprime cuando
  `initial` existe) y `weeks_text` (`herd.weeks_text(weeks)`: la frase del aria-label y de la caja, «La media de las
  7 generales pasadas pasa de 1,65 a más de 27 semanas a 1,07 en las tres últimas; 2026 va en 1,05 a 3-4 semanas»,
  o `None` sin medias).
- Tarjeta 2, entre la entradilla y los cuadritos, a toda anchura: `ctitle` «Separación entre casas por semanas
  hasta la elección», `csub` «Generales desde {since_year} y {2026}, que se completa cada día · pares de sondeos de
  casas distintas con 7 días o menos de diferencia», `<div class="chart" id="weeks-chart" role="img"
  aria-label="{{ weeks_text }}">` y debajo un `.tnote`: «Cada línea es una elección general; la discontinua, la media
  de las pasadas; la azul, 2026, que avanza con cada actualización. La línea en 1 es lo que cabría esperar sólo por
  el azar de las muestras. La curva usa todas las parejas de cada tramo; los recuentos de abajo comparan las mismas
  parejas. Un tramo con menos de dos parejas de sondeos no se dibuja.» Sin elección en curso, la subtítulo y la nota
  omiten 2026. Sin filas, el gráfico no se pinta.
- `web/dist/js/charts/weeks.js` recuperado de c3aa20e y adaptado: `weeksOption(data, opts)` con **un color por
  elección pasada** (paleta por defecto, en este orden: `#e07b39`, `#2a9d8f`, `#8e44ad`, `#c0392b`, `#7f8c3a`,
  `#d4a017`, `#5d6d7e`, cíclica), líneas de 1,6 px con puntos, leyenda con su nombre; la media de las pasadas en
  negro discontinuo («Media de las {n}»); la elección en curso en azul (`THEME.accentColor`) de 3,5 px con
  `endLabel` «{label} · {valor}»; `markLine` fina en 1 con el rótulo «lo que cabría esperar sólo por el azar», sin
  banda; dos `markArea` claras sobre los tramos `3-4`+`5-7` («3-8 semanas») y `0-2` («tres últimas») con rótulo
  arriba; eje x con las etiquetas de los tramos y el nombre «semanas antes de la elección»; eje y desde 0 de 0,5 en
  0,5; tooltip por tramo con el valor y las parejas de cada línea. `connectNulls` en las pasadas.
- `web/dist/js/pages/analisis-rebano.js`: lee la isla, dibuja `#weeks-chart` con `renderWeeks` si hay filas y cablea
  las descargas. La descarga de la tarjeta redibuja el gráfico (`card-image.js`, como el sesgo del CIS).

## D4. Pruebas y aceptación

- `tests/test_herd_stats.py`: `bucket_of` en los bordes (20 → 0-2, 21 → 3-4, 55 → 5-7, 56 → 8-11, 365 → 27-52, 400
  → None) y `by_buckets` con un tramo de 1 pareja (`ratio` None) y otro de 2.
- `tests/test_herd_study.py` sobre `write_herd_bundle` (cifras a mano, ver el plan): Generales 2019 `5-7` 1,06 y
  `0-2` 0,35; Generales 2023 `0-2` 1,06 (3 parejas); los tramos con una sola pareja sin cociente; Madrid fuera;
  2026 `12-17` 1,41; media `0-2` 0,70 (k 2), `5-7` 1,06 (k 1).
- `tests/test_publish_herd.py`: la parte valida, el CSV lleva las filas y las de la media; fixture
  `tests/fixtures/bundle/rebano-weeks.json` con cifras como las reales.
- `tests/test_js_charts.py`: `weeksOption` da una serie por elección pasada con su color, la media discontinua, la
  actual con `endLabel`, la `markLine` en 1 y las dos `markArea`.
- `tests/test_pages.py`: `#weeks-chart` con su aria-label antes de los cuadritos, `initial-data` con `weeks`, la nota,
  y sin gráfico ni «2026» sin filas o sin elección en curso.
- Aceptación con datos reales (sólo lectura): las medias de arriba y 2026 con cociente hasta `3-4`.

## Pasos tras la fusión

`publish '{"what":["study"],"study":"rebano"}'` (lo hace quien integre, con el OK de Luis); la imagen desplegada
debe llevar este código antes del siguiente `update` diario, o republicará sin la parte `weeks`.

## Lo que queda al constructor

Nombres internos, el formato exacto del tooltip, la altura del gráfico (en torno a 320 px) y el estilo de las
`markArea` dentro del sistema del sitio.

## Cierre

Ejecutado el 2026-11-07 en la rama `rebano-embudo` (de `dev` 55d47ca, cabeza 6b9c7d8 antes de este cierre).

### Qué existe

- `mtpy/lib/studies/herd/stats.py` (`bucket_of`, `by_buckets`) y `HerdStudy.weeks` en `study.py`: la dispersión por
  tramos de semanas, con una pasada por elección general desde 2008 más la elección en curso.
- Parte publicada `weeks` del estudio `rebano` (exportador `export_herd_weeks`) y su CSV largo (`pairs` y
  `comparisons` enteros, filas de la media incluidas), con el segundo botón CSV en la página.
- Gráfico `web/dist/js/charts/weeks.js` (`weeksOption`/`renderWeeks`), cableado en
  `web/dist/js/pages/analisis-rebano.js` con descarga por `card-image.js`, y la tarjeta 2 de `/analisis/rebano`
  con `#weeks-chart`, su `aria-label`, la nota y el texto `weeks_text`.
- Pruebas en `test_herd_stats`, `test_herd_study`, `test_publish_herd`, `test_js_charts` y `test_pages`, y el
  fixture `tests/fixtures/bundle/rebano-weeks.json`.

### Cifras reales (bundle del día, sólo lectura)

- Cálculo completo del estudio (`compute_herd`): 29,4 s (límite 60 s); 7 pasadas.
- Media por tramo (27-52 → 0-2): 1,65 (k 5) / 1,31 / 1,42 / 1,42 / 1,35 / 1,19 / 1,07 (k 7), idénticas a las
  esperadas.
- Generales 2026: 0,99 / 1,01 / 1,33 / 0,92 / 1,05 / 1,24 a 3-4 semanas; 0-2 sin cociente (`None`).
- Tarjeta 2: «22 de 29 (1 entre 246)» en las tres últimas semanas y «9 de 17, como una moneda» hasta tres semanas
  antes; `aria-label`: «La media de las 7 generales pasadas pasa de 1,65 a más de 27 semanas a 1,07 en las tres
  últimas; Generales 2026 va en 1,24 a 3-4 semanas.»
- Renderizado con las partes reales (test temporal ya borrado) y capturado a 1300 px; sin errores de consola.

### Decisiones durante la ejecución

- Las zonas «3-8 semanas» y «tres últimas» usan `xAxis.splitArea`: una `markArea` sobre un eje de categorías no
  tiene anchura para un tramo único.
- `by_buckets` con comparaciones vacías: máscara booleana explícita (antes fallaba con un `KeyError`).
- `pairs` y `comparisons` se publican como `Int64` (las filas de la media llevan `None` y salían como `3.0` en el
  CSV).
- «N generales pasadas» en `weeks_text` cuenta las elecciones con algún cociente, no la k de un tramo; si la
  elección en curso no tiene cociente en ningún tramo, el texto termina en «en las tres últimas.».
- El tooltip de la media muestra k; la etiqueta final de 2026 va sobre el último punto; el icono de leyenda de 2026
  es plano.

### Lo que no está verificado

- Móvil (390 px) y modo oscuro del gráfico nuevo; sólo se vio a 1300 px y la descarga PNG de la tarjeta.
- Menores aplazados: la leyenda «Media de las N» cuenta elecciones con filas y `weeks_text` las que tienen cociente;
  las cláusulas de 2026 dependen de `has_run` aunque ninguna fila actual tenga cociente; el rótulo de la línea de
  referencia en 1 se pisa con las líneas en el borde izquierdo (se ve en la captura); `dispersion` sobre un marco
  vacío si `min_pairs` fuese 0 (el defecto es 2).
- La línea de 2015 (hasta 3,0) estira el eje y comprime el resto; no se ha decidido nada.

### Pasos tras la fusión

`publish '{"what":["study"],"study":"rebano"}'` (lo hace quien integre, con el OK de Luis). La imagen desplegada
debe llevar este código antes del siguiente `update` diario, o republicará sin la parte `weeks`.
