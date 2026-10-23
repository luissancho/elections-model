# Titular de escaños = reparto central

Fecha: 2026-10-23. Estado: aprobado por Luis el 2026-10-23 (revisión del constructor incorporada).

## Intención

La predicción de escaños pasa a ser un reparto real: D'Hondt en cada circunscripción sobre el voto central,
que es el promedio de sondeos proyectado a las provincias con el swing proporcional, sin sorteo ni deriva. Ese
reparto da los escaños de cada circunscripción; su suma por partido es el titular nacional; los bloques (`vs` y
`blocks`) son la suma de sus partidos. Las simulaciones quedan para lo que miden bien: intervalos, medias,
medianas y probabilidades.

Así la página de escaños cuenta un solo número: el hemiciclo, la tabla por partido, la barra de la calculadora,
«Resultados por territorio», el detalle de cada circunscripción y los bloques del resto de la web coinciden
por construcción.

## Por qué ahora

El 2026-10-23 Luis vio en `/es/escanos` que Madrid sumaba 38 en «Resultados por territorio» (PP 16, PSOE 10,
VOX 8, FA 4, sin UP) y 37 en el detalle por circunscripción (PP 15, PSOE 9, VOX 8, FA 3, UP 2). Madrid tiene
38 escaños el 29-N (RD 806/2026). La causa no era un error de datos: la página mezclaba tres cifras distintas
para «la predicción».

| Cifra | Qué es hoy | Problema |
|---|---|---|
| Titular nacional | Mediana de cada partido en las simulaciones, expandida a 350 por restos mayores (suman 346) | Correcta, pero no es ningún reparto |
| Detalle por circunscripción | Mediana cruda de cada partido en la provincia | No suma los escaños de la provincia (13 provincias de 52; Madrid 37 de 38) ni, sumada por provincias, da la mediana nacional (PP 130 frente a 132) |
| Resultados por territorio | Una simulación real, la más cercana en L1 a las medianas nacionales (`Simulator.scenario`) | Es un sorteo: en el run del 23-10 dejaba a UP a 0 en toda España con titular 2 y P(escaño en Madrid) 0,78, y la columna de UP desaparecía |

Luis decidió el 2026-10-23 que el titular sea también el reparto, igual a nivel provincial y nacional, incluidos
los bloques, y que el titular sea el mismo en nowcast y forecast (el voto central no depende de la deriva; sólo
los intervalos cambian).

## Objetivos

1. `Simulator.totals()` devuelve el reparto central, y todo lo que cuelga de él (resumen, bloques, titular,
   histórico, hemiciclos, deltas «vs 2023», orden de las crestas, backtest) lo hereda sin tocar a los
   consumidores.
2. Las partes `scenario` y `districts` publican ese mismo reparto: cada fila suma los escaños de su
   circunscripción, la suma de las circunscripciones es el titular y `summary.totals` es su fila nacional.
3. Intervalos del 95 %, `seats_mean`, `seats_median`, `p_seats`, `p_majority` y `p_first` siguen saliendo de
   las simulaciones, sin cambios.
4. El titular es idéntico en nowcast y forecast.
5. Los textos públicos y los gráficos marcan el reparto central y dejan de llamar «mediana» a la predicción.

## No objetivos

- Cambiar el voto central (la media del forecast) o el modelo de ruido, deriva o swing.
- Cambiar el contrato del bundle (sigue en 1) o republicar automáticamente los runs antiguos: siguen siendo
  legibles; Luis republica el backfill cuando pueda para que la evolución del titular no mezcle medianas con
  repartos.
- Un parámetro para elegir entre mediana y reparto en los trabajos (`Compute`, `Publish`, `Backtest`): se
  descartó (enfoque B).
- Los modos de regresión (`split=False`, sin provincias; hoy sólo en notebooks): sin reparto posible, `totals`
  mantiene allí la mediana expandida.
- El estudio CERA: tiene su propio reparto por escenarios y no usa `totals` ni `scenario`.
- Las figuras de los notebooks (`plot_dist_kde` marca la mediana en Python): fuera de la web, quedan como están.

## Lo que sería un fracaso

- La suma de las provincias de un partido no da su titular, o un bloque no suma sus partidos.
- El titular difiere entre nowcast y forecast.
- Un partido con escaños en el reparto no aparece como columna de la tabla por territorio.
- Calcular el reparto altera las simulaciones: cambia `results`, `dist()` o los intervalos, deja `params['random']`
  en otro estado o consume el generador aleatorio (los resultados con y sin llamar a `totals` deben ser idénticos).
- Un run antiguo publicado deja de validar o de renderizar.

## Restricciones y hechos del código

- `build_frame` con `params['random']` en `False` usa la media (`pct`) sin deriva y no toca `rng`;
  `build_umat` omite el ruido de swing en ese caso; `simulate(frame, umat)` reparte con `alloc_seats` y las
  barreras del ámbito (`threshold`, `threshold_scope`) y pone la fila `default_region` como suma. Las reglas del
  `smap` (Compromís, Podemos) pasan por `apply_source_rules` con el frame determinista. Un `simulate` sobre esa
  pasada es el reparto central: no hay que inventar reparto alguno.
- `summarize_seats` admite `headline`; `summary()` ya pasa `totals()` como titular y `group_results` suma el
  titular de los partidos de cada bloque.
- `set_params` reconstruye `rng` y `rng_swing` desde la semilla en cada llamada con argumentos (o con `reset`)
  cuando `random` es `True`, y los pone a `None` cuando es `False`: un `set_params(random=False)` destruiría los
  generadores del run en curso. El reparto se calcula con el flag puesto y restaurado a mano (o con una copia
  de `params`), nunca a través de `set_params`.
- `bundle.validate` exige las claves del esquema y admite claves de más: quitar `simulation` del esquema de
  `scenario` deja válidos los runs antiguos (la traen) y los nuevos (no la traen).
- El backtest toma `summary.loc[n, 'seats']` como predicción del modelo y la suma por bloque: hereda el
  reparto sin cambios de código, pero sus cifras cambian para todas las elecciones pasadas.
- Reglas de redacción pública: nosotros, «sólo» con tilde, encuestadora/casa, nada de rótulos fijos;
  `assert_public_wording` las comprueba.
- `tests/test_publish_unit.py::test_export_scenario_is_a_real_simulation` asume la simulación más cercana;
  `tests/test_publish_unit.py:82` exige `seats` flotante en `districts`; `tests/test_pages.py:3174` fija la frase
  «la suma de esas medianas no tiene por qué coincidir…» y `tests/test_pages.py:668` la etiqueta «Escenario
  central» del CSV. Los cuatro cambian con el diseño.
- Ninguna fixture unitaria ejecuta hoy la pasada determinista: `tests/fakes.py::synthetic_simulator` no tiene
  `prev_results`, `prev_totals` ni barreras, y las fixtures reales (`sim26`, `md`) viven en `tests/integration/`
  y se saltan sin base de datos. `tests/integration/test_model.py` (líneas 32, 148-151, 200, 255, 414) y
  `tests/integration/test_regional_model.py:105` fijan cifras de `totals()` con el método por defecto.
- El backtest también resume provincias (`_provinces`, `model_seats = np.median(seats)`), que no se publican en
  la web. Y el detalle de cada elección (`publish/events.py::export_event_model`) corta las tablas del backtest:
  tras relanzarlo hay que republicar los eventos.

## Enfoques

**A. Cambiar `totals` en el simulador (elegido).** `totals()` devuelve el reparto central, calculado una vez y
guardado; `method='median'` y `'mean'` siguen disponibles para la estadística antigua. Un solo punto de verdad y
la menor superficie de cambio: resumen, titular, histórico, backtest y páginas lo heredan.

**B. `central_result` más un parámetro `headline` que elija reparto o mediana.** Permitiría publicar o
backtestear con uno u otro, pero obliga a pasar el parámetro por `Compute`, `Publish` y `Backtest` y deja dos
titulares conviviendo, justo lo que se quiere evitar. Descartado.

**C. Calcular el D'Hondt sólo en la web al publicar.** Dejaría el backtest y el histórico con la mediana y el
titular fuera del modelo. Descartado.

Luis eligió A el 2026-10-23.

## Diseño

### Simulador (`mtpy/lib/simulator.py`)

- Nuevo método `central_result()`: tabla región × partido (mismo índice y columnas que `result()`, enteros) del
  reparto central: una pasada `build_frame` → `build_umat` → `simulate` sin ruido, con `names` y `regions` de los
  `params` vigentes. Exige `fit_forecast` hecho y `split=True`; con `split=False` devuelve `None`. Se guarda en
  caché (`self.central`) y la caché se vacía en `set_params` con argumentos y en `run`. Su cálculo no altera
  `params`, `rng`, `rng_swing`, `frames`, `units` ni `results`.
- `totals(sort=False, method='central')`: por defecto la fila `default_region` de `central_result()`;
  `'median'` y `'mean'` conservan el comportamiento actual (central de `dist()` expandida con
  `round_proportional`); con `split=False`, `'central'` cae a `'median'`. Docstring reescrito: el titular es el
  reparto central; la expansión proporcional sólo describe las alternativas.
- `summary(names, alpha, method='central')`: titular `totals(method)`; `seats_median` sigue en la tabla.
  `summarize_seats` no cambia de firma; su docstring deja de llamar «headline» a la expansión por defecto sin
  más matiz: dice que el titular lo da quien llama (el reparto central) y que sin él se expande la central.
- `unit_summary(region)`: `seats` pasa a ser el reparto central de esa región; nueva columna `seats_median`
  con lo que hoy es `seats`. El resto (`pct*`, `seats_mean`, `seats_lo/hi`, `p_seats`) igual.
- `scenario()` y `closest_simulation` se conservan como utilidades (notebooks), pero la publicación deja de
  usarlas.
- `plot_dist_kde` y `plot_forecast_output` siguen ordenando por `totals()` (ya lo hacen).

### Publicación (`mtpy/lib/publish/forecast.py`, `mtpy/lib/bundle.py`)

- `export_scenario(sim)`: exporta `central_result()`; `data = {'parties', 'rows'}` con las mismas filas
  (`region_id`, `region`, `seats`), sin `simulation`; el CSV conserva sus columnas. Docstring: reparto central.
- `SCHEMAS['scenario']` pierde la clave `simulation`. El contrato sigue en 1.
- `export_districts(sim, communities)`: `seats` es el reparto; `SUMMARY_COLS` incorpora `seats_median`
  (JSON y CSV).
- `export_summary`, `headline_mode`, `export_headline`, histórico y manifiesto: sin cambios de código; heredan.
- Etiqueta del CSV en `mtpy/lib/pages/escanos.py` (`CSV_PARTS`): «Escenario central» → «Reparto central».
- Invariantes que la publicación debe cumplir, por modo: `scenario.rows[region_id=0].seats` ==
  `summary.totals`; para cada región, `scenario` == `districts.seats`; y las partes `scenario` de nowcast y
  forecast son iguales.

### Web (`web/templates`, `web/dist/js`, `mtpy/lib/pages`)

- `district_table` (Python) y `territoryRows` (JS): la lógica no cambia; las columnas siguen siendo los partidos
  con algún escaño en el reparto, que ahora incluye a UP cuando lo tiene. Pies de la tabla (plantilla y
  `CAPTION` de `escanos.js`): «Reparto central: D'Hondt en cada circunscripción sobre el voto central; cada
  fila suma sus escaños. Pasa el ratón para ver el rango del 95 %» y la variante CCAA («…suma los escaños de las
  provincias de la comunidad»). El pie de «% Voto» no cambia.
- Detalle por circunscripción: la columna «Escaños» es el reparto (`districts.seats`), con su rango del 95 %
  y P(escaño) de las simulaciones. La fila ahora cuadra con el caption «{región} · N escaños».
- Tarjeta «Reparto de escaños» (crestas, `charts/ridge.js` y su alimentación en `escanos.js`): la marca gruesa
  y el rótulo usan el titular (`seats`) en vez de la mediana; el rótulo dice «Reparto central: N» y
  «Rango 95%: a–b». El texto «Cómo leerlo» cambia «la línea gruesa la mediana» por «la línea gruesa el reparto
  central».
- Calculadora de mayorías: la franja «N partidos · X escaños · P(mayoría absoluta)» sigue (X = suma del reparto);
  la línea del histograma y su pie pasan de «Mediana» a «Reparto central» con la misma X (`coalitionCaption`
  recibe la cifra central; `histogram.js` ya admite la etiqueta). El párrafo de «Cómo leerlo» que explica por
  qué la mediana no coincide se sustituye por uno que diga que la franja y la línea son el reparto central y
  que el rango del 95 % y la probabilidad salen de las simulaciones.
- Método, «Qué es cada cifra» (`metodo_escanos.html`): la viñeta de los escaños explica el reparto central
  (D'Hondt en cada circunscripción sobre el voto central, el promedio proyectado a las provincias con el swing
  proporcional, sin sorteo; la suma de circunscripciones es el titular y los bloques suman sus partidos; el
  mismo titular en nowcast y forecast, sólo cambian los rangos); la viñeta del escenario central desaparece o
  se funde con la anterior; las del intervalo y las probabilidades no cambian. Añadir una frase de que el
  reparto central de un partido pequeño puede quedar lejos del centro de su rango cuando a veces saca escaño y a
  veces no. Revisar el glosario (`_glosario.html`) por «escenario central» y «mediana».
- Inicio, Promedios, deltas «vs 2023», bloques, histórico: sin cambios de código.
- No queda ningún texto público que llame mediana al titular. La palabra puede seguir donde describa la
  mediana de las simulaciones como estadística secundaria (p. ej. una columna `seats_median` si se mostrara).

### Backtest y tareas de Luis

- `mtpy/lib/backtest.py`: las filas nacionales y de bloque heredan el reparto por `summary.seats`, sin cambios.
  En `_provinces`, `model_seats` pasa a ser el reparto central de la provincia (fila de `central_result()`) y la
  mediana queda en una columna `median_seats`, para que el CSV provincial diga lo mismo que la web.
- Tras integrar, en este orden: (1) relanzar el backtest y `publish backtest`; (2) republicar los eventos
  (`publish event` con `event_date: all`), porque «Lo que dijo el modelo» se corta del backtest; (3) republicar el
  backfill y un run nuevo, para que la evolución del titular no mezcle medianas con repartos; (4) desplegar la
  web. Con la republicación antes del despliegue no hay ventana en la que la web etiquete como reparto un run
  antiguo.
- Orientación (cálculo aproximado del 2026-10-23 con el voto mediano provincial y barrera del 3 %; el real usa
  la media proyectada): titular PP 135, PSOE 109, VOX 65, FA 9, UP 2, JxCat 6, ERC 8, EHB 7, PNV 5, BNG 2,
  CC 1, Compromís 1; Madrid PP 16, PSOE 9, VOX 8, FA 3, UP 2. Hoy el titular es PP 134, PSOE 108, VOX 67, FA 10.

## Decisiones tras la revisión del constructor (2026-10-23)

- Fixture: se amplía `tests/fakes.py::synthetic_simulator` con una tabla de resultados previos, totales
  (`votes`, `blank_pct`, `seats`) y barreras, lo justo para que `run()`, `build_umat`, `simulate` y
  `central_result()` ejecuten la pasada real en CI. Los tests de identidad nowcast/forecast y de «el RNG no se
  toca» van ahí, no a `tests/integration/`. El `summary.totals` del fake pasa a ser un D'Hondt de sus cifras.
- Tests de integración: las cifras fijadas de `totals()` que documentan la secuencia aleatoria cambian a
  `method='median'` y conservan su propósito; no se añade un paso contra la base en el plan. Luis fija las
  cifras del reparto central cuando corra la integración contra la RDS.
- `districts.rows[*].seats` pasa a entero, como `scenario`; el test de tipo se adapta.
- Runs antiguos: no se añade una rama de detección en la web; Luis republica todos los runs antes de desplegar
  (confirmado el 2026-10-23).

## Comprobación

Tests unitarios (simulador, con `split=True` sobre el fake ampliado):
- `totals()` == fila `default_region` de `central_result()` y suma `n_seats`; cada fila regional suma los escaños
  de su región (`reg_totals`).
- `summary('vs')['seats']` y `summary('blocks')['seats']` == suma de los `seats` de sus partidos.
- `totals()` idéntico tras `run(horizon=None)` y `run(horizon='deadline')`.
- `totals(method='median')` reproduce la expansión actual (test existente adaptado).
- `central_result()` no cambia `results`, `dist()`, `params['random']` ni el estado de `rng`: una ejecución que
  llama a `totals()` entre dos `run` con la misma semilla da los mismos `results` que una que no lo llama.
- `unit_summary` devuelve `seats` (reparto) y `seats_median`.

Tests de publicación y páginas:
- `export_scenario` sin `simulation`, filas == `central_result()`; `validate('scenario', …)` acepta datos con y
  sin `simulation`.
- `export_districts.rows[*].seats` == `scenario` por región; `summary.totals` == fila 0.
- `district_table` TOTAL == `summary.totals`; un partido con escaños sólo en una provincia aparece como columna.
- `assert_public_wording` sobre los textos nuevos; ningún texto de `escanos.html`, `metodo_escanos.html` ni
  `escanos.js`/`seats.js` llama mediana al titular; `tests/test_pages.py` deja de fijar la frase de las medianas
  (3174) y la etiqueta «Escenario central» (668) y fija las nuevas.
- Backtest: `_provinces` devuelve `model_seats` (reparto) y `median_seats` (mediana).

Manual (Luis): publicar un run en local (`dry_run`) y recorrer `/es/escanos`: Madrid suma 38 en las dos tablas
con las mismas cifras, UP aparece, TOTAL == hemiciclo, bloques == suma; `/es-md/escanos` igual con su barrera.

## Lo que queda al constructor

- Nombre y forma de la caché, y cómo poner y restaurar el flag `random` sin reconstruir los generadores.
- Dónde alimentar el rótulo de las crestas con el titular (hoy se construye en `escanos.js`).
- La redacción final de los textos, dentro de las reglas de Luis.
- Si `scenario()` debe seguir expuesto o pasar a privado (se recomienda dejarlo).
