# Vista Bloques del promedio: la línea del bloque se ajusta sobre la suma por sondeo (2026-11-09)

## Intención

En la curva del promedio del 29-N, con «Bloques» activo, Izquierda sube 2,1 puntos el 15-12-2023 (entra UP en las
encuestas) y Derecha 3,8 el 6-6-2024 (entra SALF). Luis señaló que muchos lo leen como un error en los datos, y lo
es en buena parte: la línea de un bloque es hoy la **suma de las series de sus partidos** (`blockSeries` en
`web/dist/js/blocks.js`), y cuando un partido entra a mitad de ciclo su serie nace ya a su nivel suavizado
(kernel bilateral: SALF arranca en 3,85 aunque su primer sondeo, el CIS, le daba 1,5) mientras las series de PP,
VOX o SUMAR siguen apoyándose en sondeos que no listan al nuevo. La suma de los promedios deja de ser el promedio de
la suma.

Medido con el run local `20261008-181100`: la suma de las seis series principales pasa de 87,3 a 91,1 el 6-6-2024 y
tarda tres o cuatro meses en volver a 88,7, mientras que los seis partidos suman entre 88 y 89 **en cada sondeo**
durante todo el periodo; con UP, 88,5 → 90,6 → 88,5 en abril de 2024. También hunde el bloque justo antes del
nacimiento (Derecha 46,1 el 5-6 frente a 47,5 ajustado). Ajustando el bloque sobre la suma por sondeo (kernel tosco
de 30 días) no hay salto: Derecha 47,49 → 47,50; Izquierda 41,57 → 41,56. El cambio real tras la transición es
pequeño (Derecha entre uno y dos puntos y medio arriba, mezclado con la tendencia).

Queremos que la línea de un bloque diga lo que las encuestas dicen del bloque, sin escalones de construcción, y que
el lector vea dónde entró cada partido nuevo.

## Objetivos

- La línea y la banda de Izquierda y Derecha salen de un ajuste del mismo estimador local sobre **lo que suman sus
  partidos en cada sondeo** (los puntos de la vista Bloques ya son eso: `blockPolls`). Otros sigue siendo
  100 menos los dos, con la banda en cuadratura, como hoy.
- Una línea discontinua marca, sólo en la vista Bloques, la entrada en las encuestas de cada partido de Izquierda o
  Derecha que nace a mitad de ciclo («Entra SALF», «Entra UP»).
- Lo mismo en el promedio del detalle de una elección pasada (`/{scope}/{date}`), cuando el backtest aporte la serie.
- Las series de bloque y las entradas van publicadas en el bundle (JSON y CSV), como el resto de la web: el
  navegador no recalcula nada.
- Los textos de la vista Bloques dicen cómo se construye la línea y por qué el desglose por partidos del tooltip
  puede no cuadrar con ella durante una transición.

## No objetivos

- Tocar el promedio por partido (la mezcla de sondeos con y sin Podemos sesga SUMAR en dic-2023..mar-2024; es
  inherente a cualquier promedio de encuestas y se queda).
- Cambiar los bloques del simulador (`groups.blocks` de la proyección), la evolución de las publicaciones
  (`blockRuns`), el hemiciclo, la tabla de resultados ni la tarjeta «Mayoría absoluta».
- Rehacer el backtest: la serie de bloque de las elecciones pasadas llegará cuando Luis lo vuelva a correr; hasta
  entonces el detalle sigue sumando partidos.
- Cambios en la API más allá de las claves nuevas de `series` y `event-series` (el contrato sigue en 1).

## Anti-objetivos

- Inventar datos: nada de suavizar o recortar el escalón a mano; sólo cambiar qué se ajusta.
- Romper runs ya publicados: un `series.json` sin `blocks` se dibuja como hoy (suma).
- Encarecer la publicación: son dos ajustes de kernel más por run (y por caso del backtest).

## Enfoques considerados

- **A. Ajustar el bloque sobre la suma por sondeo** (elegido): coherente con los puntos de la vista, banda propia del
  ajuste, sin escalón; sólo cambia lo que se publica.
- **B. Marcar las entradas** y dejar la suma: barato y honesto, pero deja el artefacto. Se hace **además** de A, para
  explicar el cambio real que quede.
- **C. Truncar la curva al último nacimiento**: descartado; pierde diez meses del 29-N y no arregla los cuatro meses
  de mezcla de UP.

Luis eligió A con B el 2026-11-09.

## Diseño

### Modelo (`mtpy/lib/forecaster.py`)

`Forecaster.fit_groups(groups: dict[str, list[str]], max_fc: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]`:

- `groups` es `{bloque: [nombres simulados]}` (los nombres de `self.names`, es decir, las columnas de `self.series`
  tras agrupar los sondeos con `poll_blocks`: FA ya lleva los sondeos de SUMAR, PP los de UPN).
- Por sondeo, el valor del bloque es la suma de sus miembros presentes (`sum(min_count=1)`): un sondeo sin ninguno
  no entra; uno con parte de ellos entra con lo que lista. Se toma de `self.fc_series`, que ya va neto de efectos
  de casa (M6) y lleva el peso definitivo (`weight`, con la muestra efectiva M12 y el rating).
- Cada bloque se ajusta con el mismo `LocalKernelEstimator`, `reg_params`, `alpha` y agrupación por encuestadora
  (M10) que `fit`, sobre el mismo `fc_index`; menos de tres sondeos útiles → el bloque no se ajusta (aviso, como
  `fit`).
- Devuelve `(forecast, fc_stat)` con una columna por bloque, con la misma forma que `self.forecast` y `self.fc_stat`
  para esas columnas. **No toca** `self.forecast`, `self.fc_stat` ni el residual `'-'`.
- `fit` se refactoriza lo justo para que ambos compartan el ajuste de una columna (misma salida que hoy).

`Forecaster.cycle_births(groups: dict[str, list[str]], min_days: int = 90) -> dict[str, str]`:

- Para cada nombre simulado que sea miembro de algún grupo, su primera fecha de sondeo es la **menor** entre las
  etiquetas de su bloque de sondeos (`self.blocks.parties`: FA toma la del primer sondeo de SUMAR, no la del primero
  que dijo «FA»), leídas de `party_first_polls` (sondeos computados).
- Se devuelve `{nombre: 'YYYY-MM-DD'}` sólo para los que entran al menos `min_days` días después del primer sondeo
  del ciclo (`date_first`). Con el 29-N: UP (2023-12-15) y SALF (2024-06-06); FA, PP, PSOE y VOX no.
- `min_days = 90` es un corte de Claude: separa «nace a mitad de ciclo» de «lo empezaron a listar unas semanas
  tarde». No se expone en la web.

### Publicación del run (`mtpy/lib/publish/forecast.py`, `mtpy/lib/bundle.py`)

- `export_series(fc, names, end=None, blocks=None)`: con `blocks` (la salida de `web_blocks`, sin `Otros`), ajusta
  los bloques con `fit_groups(blocks, max_fc=fc.max_fc)` (el mismo alcance tras el último sondeo que los partidos,
  para que la línea del bloque llegue a `as_of` como las suyas y no deje hueco antes de la proyección), corta en
  `end` como los partidos y añade a `data`:
  - `blocks`: `{'parties': ['Izquierda', 'Derecha'], 'mean': {...}, 'lo': {...}, 'hi': {...}}`, alineado a `dates`
    (`null` donde el bloque no tiene valor). Sólo los bloques que se han podido ajustar.
  - `first_polls`: `cycle_births(blocks)`; `{}` cuando no hay ninguno.
  - Sin `blocks` (ámbito sin Izquierda/Derecha), `blocks` es `{'parties': [], 'mean': {}, 'lo': {}, 'hi': {}}` y
    `first_polls` `{}`.
- El gemelo CSV `csv/series.csv` gana la columna `kind` (`party` | `block`) y las filas de los bloques detrás de las
  de los partidos: `date, name, kind, mean, lo, hi`.
- `publish_forecast` pasa `blocks=_web_bmap(sim)` (sin `Otros`).
- Esquema `series` en `bundle.SCHEMAS`: `+ blocks: dict, first_polls: dict`. Contrato en 1.

### Backtest y detalle de una elección (`mtpy/lib/backtest.py`, `mtpy/lib/publish/events.py`)

- `case_series(sim, event_date, horizon)` llama a `export_series` con los bloques de la web del caso
  (`web_blocks(sim.model.bmaps['vs'], sim.params['names'])` sin `Otros`): el `series.csv` del backtest gana `kind` y
  las filas de bloque. Las tablas de backtest ya guardadas no tienen `kind`: quien las lea trata la ausencia como
  `party`.
- `export_event_series` conserva `kind` y publica `blocks` con la misma forma que el run (vacío cuando el backtest
  no trae filas `block`), y `first_polls` calculado en la publicación del evento con la misma regla: la primera
  fecha de sondeo de cada partido de Izquierda/Derecha sobre las etiquetas de su bloque de sondeos
  (`Simulator.poll_blocks(names, smap)` con el `smap` del evento y `names` = los partidos simulados,
  `comp.event_params[event_date]['parties']['event']`, los mismos que `case_series` usa para las filas `block`;
  no `comp.names`, que marcaría partidos marginales del `vs` que el backtest no simula) a partir de los sondeos de
  `comp.series`, frente al primer sondeo del ciclo, 90 días. Una fila del backtest sin `kind` (tablas anteriores a
  este cambio, o `NaN` tras una fusión por elección) cuenta como `party`. Esquema `event-series`: `+ blocks: dict, first_polls: dict`.
- El CSV del evento `series.csv` lleva `kind` como el del run.
- Correr el backtest de nuevo y republicar los eventos es decisión de Luis, fuera de este trabajo.

### Web (`web/dist/js/blocks.js`, `web/dist/js/charts/series.js`, `web/dist/js/pages/series-wire.js`)

- `blockSeries(series, blocks)`: si `series.blocks` trae un bloque, su `mean`/`lo`/`hi` se leen de ahí; los bloques
  que no vengan (o un `series` sin `blocks`, runs antiguos) se suman como hoy. `Otros` sigue siendo el complemento
  con la banda en cuadratura de los otros dos.
- `seriesOption` acepta `marks: [{date, label}]` y los dibuja como líneas discontinuas en la serie de marcas, con la
  etiqueta arriba como «Estimación» y «Elección»; sin solaparse con éstas (si una entrada cae a menos de unos días
  de otra marca, la etiqueta va abajo; Claude decide el umbral).
- `seriesViews` pasa a la vista `bloques` `marks` con las entradas de `series.first_polls` («Entra {nombre}»),
  ordenadas por fecha; la vista `partidos` no lleva marcas (ahí la línea del partido ya empieza donde empieza).
- El desglose del tooltip (`breakdownTooltipLines`, db65a2d) no cambia: lista las series por partido y «Resto» sólo
  si sobra ≥ 0,05; cuando los partidos suman más que la línea (transición), no se añade nada.
- Las descargas de imagen (`share.js`) dibujan el gráfico tal cual: incluyen las marcas.

### Textos (`web/templates/promedios.html`, `web/templates/eleccion.html`, `docs/web/contrato.md`)

Sólo cambia el párrafo de la **tarjeta del promedio** (`promedios.html`, hoy línea 64, y su homólogo en
`eleccion.html`, línea 167). El de «Evolución de las actualizaciones» (`promedios.html`, línea 304) **no cambia**:
esa tarjeta sigue sumando los partidos de cada run (`blockRuns`) y su frase sigue siendo cierta.

El párrafo nuevo se pinta **sólo cuando la serie trae bloques ajustados**: el servidor pasa a la plantilla una
bandera (`series.blocks.parties` no vacío, en `mtpy/lib/pages/promedios.py` y en la página del detalle de una
elección) y, sin ella, la plantilla conserva la frase de hoy («cada bloque suma sus partidos…»): un run antiguo
(`?run=`), un ámbito aún no republicado o una elección pasada sin backtest nuevo siguen descritos como se dibujan.
Las marcas llegan con `first_polls`, en el mismo `series.json`, así que van siempre de la mano del texto nuevo.

Con la bandera, sustituir «cada bloque suma sus partidos, que se leen bajo la leyenda, y Otros es lo que falta
hasta 100» por una redacción de este tenor (Claude ajusta a cada sitio, con las reglas de redacción de Luis:
nosotros, todo a la vista, «sólo» con tilde, encuestadora/casa):

> En la vista Bloques, la línea de cada bloque es el promedio de lo que suman sus partidos en cada sondeo, no la
> suma de sus promedios: cuando un partido entra en las encuestas a mitad de ciclo (una línea discontinua lo marca),
> las dos cosas se separan durante unos meses y el desglose del bloque al pasar el ratón puede no cuadrar con la
> línea. Los partidos de cada bloque se leen bajo la leyenda; Otros es lo que falta hasta 100.

Contrato: enmienda 2026-11-09 en `series` y `event-series` (`blocks`, `first_polls`, columna `kind` del CSV).

### Pruebas

- `tests/test_forecaster_unit.py`: `fit_groups` coincide con el ajuste del partido cuando el grupo tiene un solo
  miembro; con todos los sondeos listando a todos los miembros, iguala la suma de los ajustes; con un miembro que
  entra a mitad de ciclo, no da escalón (comparar el día anterior y el del primer sondeo); menos de tres sondeos →
  sin columna; no muta `forecast` ni `fc_stat`. `cycle_births`: etiqueta heredada, umbral, vacío.
- `tests/test_publish_unit.py`: `export_series` con y sin `blocks` (claves, alineación a `dates`, `kind` y orden
  del CSV, corte en `end`); `export_event_series` conserva `blocks` y `first_polls`.
- `tests/test_backtest_unit.py`: `case_series` lleva `kind` y filas de bloque; `shortest_horizon` conserva columnas.
- `tests/test_js_blocks.py`: `blockSeries` lee `series.blocks` y cae a la suma sin él; Otros como complemento.
- `tests/test_js_charts.py`: marcas en `seriesOption`.
- Páginas: la bandera de bloques ajustados llega al contexto y la plantilla pinta una u otra frase.
- Redacción pública (`assert_public_wording`) sobre los textos nuevos.

### Despliegue (pasos de Luis)

Imagen y `update-service`; `update` de `es` (y de los ámbitos autonómicos cuando toque) para que el `series.json`
en vivo traiga `blocks`; sin ello la web sigue sumando. Backtest y eventos, cuando él decida.

## Lo que queda al constructor

- Cómo compartir el ajuste de una columna entre `fit` y `fit_groups` sin cambiar la salida de `fit`.
- Dónde poner la regla de las entradas para que el run y el evento la compartan (helper en `utils` o en
  `Forecaster`), y su prueba.
- El umbral de separación de etiquetas de las marcas y su posición.
- Las palabras exactas de los tres textos y de la enmienda del contrato.

## Decisiones tomadas tras la revisión del constructor (2026-11-09, Claude)

- Los bloques se ajustan con `max_fc = fc.max_fc`, no 0: misma longitud que las líneas de los partidos.
- El texto nuevo es condicional a que la serie traiga bloques ajustados; la Evolución no cambia (el «×2» de la
  primera versión contaba por error su párrafo).
- Los miembros de Izquierda/Derecha en el evento son los partidos simulados (`parties.event`), como en el backtest.
- Una fila del backtest sin `kind` cuenta como `party`.

## Riesgos aceptados

- La proyección por bloques sigue siendo la del simulador (suma de los partidos en `as_of`), así que puede quedar un
  pequeño desnivel entre el final de la línea ajustada y el inicio de la proyección; fuera de una transición es de
  centésimas (0,16 puntos en octubre de 2024, con SALF a medio asentar). No se corrige.
- Hasta que se vuelva a correr el backtest, el detalle de las elecciones pasadas suma partidos: igual que hoy.
