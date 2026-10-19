# Web: el promedio de sondeos y el modelo en el detalle de una elección

Fecha: 2026-10-19. Estado: aprobada por Luis el 2026-10-19 ("Ok, adelante") tras la revisión del constructor; las
tres preguntas del final quedan resueltas como se proponen (serie fuera de git, tarjetas tras los resultados,
cifras de la ejecución a horizonte). Sigue a las fases 7e y 7f (detalle de una elección) y a la 8 (rediseño
visual). Esta spec solo describe lo que cambia respecto de lo que hay en `dev`
(`3c3aacd`).

## Intención

En la página de una elección pasada (`/elecciones/{país}/{ámbito}/{AAAA-MM}`), igual que abajo se ven los sondeos
del ciclo ("Todas las encuestas"), Luis quiere ver también el **promedio de sondeos** de ese ciclo y **lo que el
modelo dijo** antes de votar. Hoy la página enseña lo segundo al final, en hasta 18 tablas verticales ("El modelo a
cada horizonte"), que no se leen como comparación con las encuestas; y lo primero no existe: el backtest guarda lo
que el modelo decía a 6, 14, 30, 60, 90 y 180 días, pero no la curva del promedio a lo largo del ciclo.

El horizonte de 6 días es el último día con sondeos antes de la veda: es "lo que el modelo habría publicado".

## Decisiones de Luis (2026-10-19)

1. Las dos cosas: **la curva con los puntos, igual que en el Promedio normal** (Promedios), y **lo que el modelo
   dijo, en porcentaje y escaños, a 6 días del evento**.

## Objetivos

- O1. El backtest guarda, para cada elección, la serie diaria del promedio del ajuste más cercano a la elección (el
  horizonte más corto), con su banda.
- O2. `publish event` publica esa serie como parte nueva de la elección, con su ruta en la API y su gemelo CSV.
- O3. El detalle dibuja el promedio como en Promedios: la línea y la banda de cada partido, los sondeos del ciclo
  como puntos y, al final, el resultado oficial.
- O4. Lo que el modelo dijo a 6 días está junto a las encuestas: dos filas fijas en "Todas las encuestas"
  (promedio del modelo y resultado, en % y en escaños) y la tarjeta del modelo compactada, con los 6 días delante.

## No objetivos

- `publish event` no calcula nada del modelo: sigue leyendo las tablas del backtest. La serie sale del mismo ajuste
  que el voto a 6 días ya publicado, así que las dos cifras coinciden por construcción.
- No cambian el protocolo del backtest (casos, horizontes, semilla, simulaciones) ni sus métricas.
- No se publica la serie en el paquete del backtest (`publish backtest`, `/metodo/backtest`): solo en la elección.
- Sin proyección hasta el día de la elección en la curva: la línea termina donde se congeló el modelo y los 6 días
  de veda quedan como hueco hasta el resultado.
- Promedios no cambia de aspecto ni de comportamiento (solo comparte código con el detalle).
- Anti-objetivo: que una elección publicada antes de esta fase deje de verse. Todo lo nuevo degrada a un aviso o
  desaparece sin ruido cuando falta su dato.

## Datos

### D1. La serie en el backtest (`mtpy/lib/backtest.py`, job `backtest`)

- `run_case` devuelve una tabla más, `series`: una fila por día y partido con `event_date`, `horizon`, `date`,
  `name`, `mean`, `lo`, `hi`. Es la serie diaria ajustada del Forecaster del caso (`sim.model`) **hasta el ancla
  `sim.as_of`** (el día en que se congeló el modelo: la elección menos el horizonte), con la banda de `fc_stat`
  (`cmin`/`cmax`): la misma forma que `export_series` del run (`mtpy/lib/publish/forecast.py`), que gana un
  parámetro `end` para cortar ahí (el run sigue cortando en `date_fit_last`). Cortar en `as_of` y no en
  `date_fit_last` importa: `date_fit_last` es el último sondeo más `max_fc` días, topado en la elección, así que la
  serie cruzaría la veda y su último día ya no sería el de la cifra a 6 días. Si importar `publish` desde
  `backtest` crea un ciclo de importaciones, la función se mueve a un módulo común (el constructor decide dónde).
  Redondeo a `PCT_DECIMALS`.
- `run_backtest` conserva solo la serie **del horizonte más corto que haya corrido** de cada elección (el 6 por
  defecto; si ese caso se omite por falta de sondeos, el siguiente). Las demás no se guardan: la curva que se dibuja
  es una sola y un fichero con las seis multiplicaría el tamaño por seis.
- `backtest_scope` escribe `series.csv` junto a las demás tablas de `backtest/results/` (o de la carpeta del
  ámbito); `BACKTEST_FILES` de `publish/backtest.py` no la incluye (no se publica en el paquete del backtest).
  `run.json` no cambia. `backtest/README.md` documenta la tabla y `deploy/README.md` ("Backtest del modelo") la
  cita en la salida del job.
- **`series.csv` queda fuera de git** (`.gitignore`): es un derivado pesado (unos 1,5 MB para `es`; del orden de
  20 MB con los 17 ámbitos autonómicos) que crecería el historial en cada ejecución del backtest. Consecuencia:
  `publish event` tiene que correr en la máquina donde corrió el backtest (hoy, la de Luis). Si algún día se
  publican elecciones desde ECS, habrá que llevar el fichero al contenedor o reconsiderarlo. (Decisión mía;
  pregunta P1.)
- Unos `results` sin `series.csv` (un backtest anterior a esta fase) publican la elección sin la parte.

### D2. La parte `series` de la elección (`mtpy/lib/bundle.py`, `mtpy/lib/publish/events.py`)

- `EVENT_PARTS` gana `series` (schema `event-series`): `{'horizon': int, 'dates': list, 'parties': list,
  'mean': dict, 'lo': dict, 'hi': dict}`; `dates`, `parties`, `mean`, `lo` y `hi` tienen exactamente la forma de
  la parte `series` de un run, para que el gráfico los consuma sin traducir. `horizon` es el del ajuste (6).
  Gemelo CSV `series.csv` en formato largo (`date`, `name`, `mean`, `lo`, `hi`), `EVENT_CSV` gana `series`.
- `export_event_series(source, event_date)` lee `series.csv` de la carpeta del backtest (`backtest_source`),
  filtra la elección y pivota. Sin fichero o sin filas para la elección no escribe la parte y borra un
  `series.json` y un `series.csv` de una publicación anterior si los hay (como hace con los gemelos `model-*`
  que faltan), para que no describan otro backtest.
- Orden de escritura en `publish_event`: tras `model` y antes de `meta`. `meta` no cambia.
- La ruta `/api/v1/events/{ámbito}/{fecha}/series` (y `?format=csv`) sale sola de `EVENT_PARTS`;
  `deploy/check-api.sh` la añade a las comprobaciones de rutas publicadas.
- Fixture `tests/fixtures/bundle/event-series.json` con una serie corta.

### D3. Lo que el modelo dijo a 6 días (dato de la página, `mtpy/lib/pages/elecciones.py`)

De la parte `model` ya publicada, `eleccion_context` toma el horizonte más corto (`horizons[0]`) y construye
`model_last = {'horizon', 'date' (la elección menos el horizonte), 'pct': {partido: shares.model},
'seats': {partido: seats.model_h, o seats.model si no hay ejecución a horizonte}, 'seats_valid'}`. Se embebe en
`initial.model_last`; `initial.results.rows` gana el `pct` oficial de cada partido. Sin `model`, `model_last` es
`None`.

## La página (`web/templates/eleccion.html`, `web/dist/js/pages/elecciones.js`)

### P1. Tarjeta nueva "Promedio de sondeos"

- **Dónde**: justo después de la rejilla del hemiciclo y los resultados, antes de la fila (barras | último sondeo).
  El lector ve el resultado, después cómo llegaron los sondeos hasta él y lo que dijo el modelo, y solo entonces
  las cuentas de cada encuestadora. (Decisión mía; pregunta P2.)
- **Título** "Promedio de sondeos"; texto de la tarjeta: "La intención de voto de cada partido a lo largo del
  ciclo según el promedio ponderado del modelo, congelado a {h} días de la elección, y el resultado oficial al
  final."
- **Controles**, los mismos de Promedios y con el mismo aspecto: Vista Partidos / Bloques (solo si la elección
  tiene bloques en su `meta`), Ventana (Todo el ciclo / 1 año / 6 meses / 3 meses, contada hacia atrás desde el
  día de la elección) y la leyenda de píldoras, con los partidos `main` de la elección encendidos por defecto.
  El estado va en el hash de la URL como en Promedios (`#view=…&window=…&…`): el detalle no usa el hash para
  nada más.
- **Gráfico**: `renderSeries` (`charts/series.js`) con `series` = la parte `series` de la elección, `polls` =
  `chart_polls(cycle)` (ya embebido en `initial.polls`), sin `projection`, `asOf` = último día de la serie,
  `when` y `anchor` = día de la elección. Novedad en `seriesOption`: una opción `result = {date, values:
  {nombre: pct}}` que dibuja el resultado oficial de cada partido (o bloque) como un punto marcado a la fecha de la
  elección: símbolo distinto y mayor que los puntos de los sondeos (p. ej. un rombo de 10 px), del color del
  partido con borde blanco, que la leyenda enciende y apaga con el partido; tooltip "Resultado oficial · {fecha}
  · {partido} {pct} %" (la forma concreta la decide el constructor). El eje de fechas llega hasta el día de la
  elección aunque la serie termine 6 días antes, así que el hueco de la veda se ve: `axisDatesOf` y `setWindow`
  cuentan con la fecha de `result` (hoy el eje y la ventana terminan en la última fecha de la serie, y el punto
  del resultado quedaría fuera en todas las ventanas). En la vista Bloques el resultado de cada bloque es la suma
  del voto de sus partidos (la página ya calcula `block_result_rows`).
- **Bloques y catálogo**: la vista Bloques de la curva usa **la misma asignación de bloques que ya usan en esta
  página el hemiciclo y "Resultados por bloque"** (`party_blocks`: `bmaps.blocks` de `data/params.json` si la
  elección está ahí, si no los dos bloques `vs` de la `meta`, si no el bloque del catálogo), calculada sobre la
  unión de los partidos de la serie y de los resultados, e invertida a `{bloque: [partidos]}` en
  `initial.catalog.bmaps.blocks`; `initial.catalog.bmaps.main` es el `main` de la `meta`. Así las tres vistas
  Bloques de la página cuadran entre sí y con Promedios en las elecciones de `params.json`, sin tocar el schema de
  la parte. Los partidos de la serie que no estén en el catálogo que el detalle embebe hoy se añaden a él
  (`chart_parties`), para que tengan color y nombre completo. (Decisión mía tras la revisión.)
- **Cableado compartido**: `seriesViews`, `chosenNames`, `fillLegend` y `wireAverage` de `pages/promedios.js` se
  extraen a `pages/series-wire.js` (como `polls-wire.js`) y Promedios y el detalle lo usan. Promedios sigue
  igual. El catálogo que embebe el detalle (`initial.catalog`) gana `bmaps` (`main` y `blocks` de la `meta` de
  la elección), que es lo que `seriesViews` lee para la vista Bloques y los partidos por defecto.
- **Sin la parte**: si la elección tiene modelo (`has_model`) pero no serie, la tarjeta se queda con el aviso "El
  promedio de este ciclo no está publicado todavía: hace falta volver a publicar la elección con un backtest que
  guarde la serie." Sin modelo, la tarjeta no aparece. (Decisión mía.)
- **"Cómo leerlo"** (tres párrafos, como las demás tarjetas): qué es cada cosa (puntos = sondeos, línea =
  promedio ponderado, banda = su intervalo de confianza, el punto marcado del final = el resultado oficial);
  cómo leerlo (el hueco entre la línea y el resultado es la veda; pulsa la leyenda; la vista Bloques suma); qué
  no es (la banda dice dónde estaba el consenso, no la incertidumbre del resultado; la línea se congeló a {h}
  días y no recoge lo que cambió después).
- **Descargas**: enlace nuevo "Promedio de sondeos del modelo (a {h} días)" al CSV de la parte, solo cuando está
  publicada.

### P2. "Lo que dijo el modelo" sustituye a "El modelo a cada horizonte"

- **Dónde**: justo después de P1. **Título** "Lo que dijo el modelo"; texto: "El voto y los escaños que el modelo
  habría publicado a cada distancia de la elección, frente al resultado; de entrada, a 6 días, el último día con
  sondeos."
- **Selector de horizonte**: un control segmentado "Horizonte" con una opción por horizonte de la parte `model`
  ("A 6 días", "A 14 días", …), el más corto marcado. Como los demás controles, el servidor lo pinta oculto y el
  JS lo enseña; sin JavaScript solo se ve el primer horizonte.
- **Una tabla por horizonte**, en lugar de las tres de hoy, con los partidos `main` como filas (los mismos de
  hoy): Partido | Promedio (%) | Resultado (%) | Error | Escaños modelo | Escaños reales. Error = promedio menos
  resultado, con signo, con el fondo del mismo convenio que las columnas de bloque de la tabla del último sondeo
  (`shade_around`: rojo si el modelo se quedó corto, verde si se pasó, más intenso cuanto mayor). Cuando los
  escaños del caso no son válidos (`seats_valid` falso), las dos columnas de escaños van con "—" y debajo el
  aviso de hoy ("Escaños no proyectables en esta elección…").
- **La tabla de bloques** de hoy se mantiene por horizonte (Bloque | Escaños reales | Escaños del modelo | Prob.
  de mayoría | Mayoría real).
- **Las cifras de escaños y de mayoría son las de la ejecución a horizonte** (`model_h`, `model_seats_h`,
  `p_majority_h`), no las del nowcast que enseña la tarjeta de hoy: son lo que el modelo habría publicado como
  pronóstico para el día de la elección y, según el README del backtest, "la comparación honesta a d días". A
  6 días apenas cambian; a 180 sí. Si una fila no trae `_h` (un backtest `nowcast_only`), se usa el nowcast. El
  voto no tiene esta distinción: su valor central es el mismo en las dos ejecuciones. (Decisión mía; pregunta
  P3.)
- **"Cómo leerlo"**: tres párrafos, reescritos sobre los de hoy (qué es cada columna; el error con signo y la
  probabilidad; "una probabilidad no se acierta ni se falla en una sola elección").
- **Descargas**: los tres CSV del modelo siguen igual. El texto de la tarjeta de Descargas pasa de "el modelo a
  cada horizonte" a "lo que dijo el modelo".

### P3. Dos filas fijas en "Todas las encuestas" (`web/templates/_polls_table.html`, `pages/polls-dom.js`)

- Solo cuando la elección tiene modelo. En el pie de la tabla, tras "Media simple", dos filas que no cambian
  con los filtros, con el mismo formato que las demás (una celda por partido, el líder sombreado, "–" si el
  partido no está) y las celdas de rating, fecha y muestra vacías:
  - **"Promedio de sondeos"**, con la línea pequeña "modelo · a {h} días" y, como `title`, "Congelado el
    {fecha}: solo los sondeos publicados hasta entonces". En la vista % de voto, `model_last.pct` a un decimal;
    en la vista Escaños, `model_last.seats` enteros, o "—" en todas las celdas si los escaños no son válidos.
  - **"Resultado"**, con la línea pequeña de la fecha de la elección: el `pct` oficial, o los escaños oficiales en
    la vista Escaños.
- El servidor pinta la vista % (`polls_table` o el contexto del detalle aportan las filas; el constructor decide
  dónde). `renderFoot` deja de reemplazar el pie entero: reconstruye solo la fila de la media y vuelve a pintar
  las filas fijas según la vista, desde `initial.model_last` y `initial.results`. En Promedios no hay filas fijas
  (no hay `model_last`) y el pie sigue como hoy.
- La regla del conmutador % de voto / Escaños no cambia: aparece cuando alguna encuestadora del ciclo publica
  escaños (`has_seats`). Si ninguna lo hace, los escaños del modelo se ven en P2.
- El "Cómo leerlo" de la tarjeta gana una frase sobre las dos filas: qué son y que los filtros no las tocan.

### P4. Orden de la página

Cabecera · Hemiciclo + Resultados · **Promedio de sondeos** · **Lo que dijo el modelo** · Error del último
sondeo | por bloque · La diana · Último sondeo de cada encuestadora · Error de cada sondeo · Todas las encuestas ·
Descargas.

## Operación (Luis, después de integrar)

1. Volver a correr el backtest para que escriba `series.csv`: `python job.py backtest '{"scopes":"all"}'`
   (unos 15 minutos para `es` con la base local según el README; más con todos los ámbitos).
2. Volver a publicar las elecciones: `python job.py publish '{"what":["event"],"scopes":"all","event_date":"all"}'`.
3. Desplegar (plantillas y JS de `web/dist`).

Hasta que se hagan 1 y 2, P1 enseña su aviso y P2 y P3 ya funcionan con la parte `model` publicada.

## Comprobación

- Python: `test_backtest_unit.py` (la tabla `series` de un caso; `run_backtest` conserva solo el horizonte más
  corto de cada elección), `test_jobs_backtest.py` (se escribe `series.csv`), `test_publish_unit.py` y
  `test_bundle_unit.py` (schema `event-series`, pivote, gemelo CSV, borrado de la parte antigua cuando no hay
  serie, orden de escritura), `test_webapi_unit.py` (ruta JSON y CSV), `test_pages.py` (`model_last`, las filas
  fijas, la tabla fusionada por horizonte con `_h` y con el nowcast de respaldo, el aviso y la ausencia de la
  tarjeta, los enlaces de Descargas).
- JS (con `node`, como en `test_js_charts.py` y `test_js_polls.py`): `seriesOption` con `result` (serie
  marcada, eje hasta la fecha del resultado, tooltip), el pie de la tabla con las filas fijas en las dos vistas,
  y que Promedios sigue produciendo la misma opción del gráfico tras extraer `series-wire.js`.
- Chrome sin cabeza sobre una elección de prueba, como en la fase 7e: la curva con puntos y resultado, el
  selector de horizonte, las filas fijas al filtrar y al cambiar de vista, sin errores de consola.
- Todas las pruebas existentes siguen pasando (952 hoy).

## Decisiones tomadas tras la revisión del constructor (2026-10-19)

El revisor contrastó la spec con el código y dejó una pregunta y tres avisos; los cuatro quedan resueltos arriba
como decisiones mías, sin cambiar lo que se construye:

1. La vista Bloques de la curva usa la asignación de bloques de la propia página (`party_blocks`), no el mapa `vs`
   de la `meta` a secas: con `vs` la curva enseñaría dos bloques distintos de los cuatro del hemiciclo de al lado y
   la izquierda de los sondeos anteriores a SUMAR quedaría sin su voto (P1, "Bloques y catálogo").
2. La serie del backtest se corta en `as_of`, no en `date_fit_last` (D1).
3. El eje y la ventana del gráfico incluyen la fecha del resultado (P1, "Gráfico").
4. Los partidos de la serie entran en el catálogo que embebe el detalle (P1, "Bloques y catálogo").

## Lo que queda al constructor

- Dónde vive la función que exporta la serie para que la usen el run y el backtest.
- La forma exacta del punto del resultado (símbolo, tamaño, borde) y de su tooltip, y cómo entra en la leyenda.
- La división concreta de `series-wire.js` y los nombres de sus exportaciones.
- Dónde se construyen las filas fijas del pie (contexto del detalle o `polls_table`) y los estilos de sus celdas.
- Los textos definitivos de "Cómo leerlo" dentro de lo que dice esta spec, y las fixtures de prueba.

## Preguntas para Luis

- P1. `series.csv` fuera de git (D1): publicar las elecciones desde tu máquina, donde corre el backtest. ¿Vale,
  o prefieres que vaya en el repositorio aunque pese?
- P2. La posición de las dos tarjetas, justo después de los resultados (P1, P2). ¿De acuerdo?
- P3. Escaños y mayoría de la ejecución a horizonte en la tarjeta del modelo y en la fila fija (P2): cambia las
  cifras que hoy enseña "El modelo a cada horizonte". ¿De acuerdo?
