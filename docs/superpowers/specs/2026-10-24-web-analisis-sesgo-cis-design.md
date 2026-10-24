# Web, Análisis: «El sesgo del CIS de Tezanos» (`/analisis/sesgo-cis`)

Fecha: 2026-10-24. Estado: **pendiente de aprobación** (brainstorm cerrado con Luis el 24 de octubre, maqueta v23
aprobada: `docs/superpowers/specs/2026-10-24-web-analisis-sesgo-cis-mockup.html`; revisión del constructor en la
sección final). Segundo estudio de la sección «Análisis», tras el del voto CERA
(`2026-10-19-web-analisis-voto-cera-design.md`). Esta spec describe lo que cambia respecto de `dev` (b740448 + los
cambios sin confirmar del 24-10).

## Intención

La tarjeta «El CIS: 38 de 41 elecciones en la misma dirección» de `/metodo/promedio` muestra que, desde que cambió
su método en septiembre de 2018, el último sondeo del CIS antes de cada elección ha dado al bloque de izquierdas más
voto del que obtuvo casi siempre. Luis quiere convertirla en un **análisis completo**, con más datos y más pruebas, que
demuestre dos cosas con datos públicos:

1. Que es prácticamente imposible que ese patrón se deba al azar o a un fallo general de las encuestas, y que la
   deriva coincide con la llegada de José Félix Tezanos a la presidencia del CIS (junio de 2018), un nombramiento
   político: el propio CIS, entre 2008 y 2016, se equivocaba como las demás casas y sin dirección dominante.
2. Que la leyenda de que «el CIS fue el único que acertó en 2023» es falsa: las demás casas fallaron hacia la derecha,
   el resultado se movió en la dirección del CIS, pero quedó más cerca del consenso de las otras casas que del CIS, cuyo
   error en los bloques fue el de siempre.

Es una **pieza de evidencia histórica**: no registra ninguna predicción y no se comprueba después del 29-N. Se
actualiza sola con cada elección que se incorpore a los análisis publicados.

Titular y entradilla, de Luis: **«El sesgo del CIS de Tezanos»** / «Desde el nombramiento de José Félix Tezanos en
2018, el CIS ha dado al bloque de izquierdas más voto del que obtuvo en 38 de 41 elecciones generales y autonómicas
celebradas. Es imposible que tal cosa pueda ocurrir por azar ni por un fallo general de las encuestas. Y no, en 2023
tampoco «acertó».» (el «38 de 41» es una cifra viva).

## Decisiones de Luis (brainstorm del 2026-10-24)

1. Evidencia histórica, sin predicción registrada.
2. Tezanos en el titular; la página debe mostrar que la deriva es suya (nombramiento político, cambio de método).
3. 2023 se analiza **sólo por bloques** (Izquierda frente a Derecha), no por partidos.
4. El control son los años **sin Tezanos, de 2008 a 2016**: mismo CIS, misma medida.
5. Tres preguntas con veredicto, como la página del voto CERA: ¿Se inclina a la izquierda desde que lo preside
   Tezanos? **SÍ** · ¿Puede ser azar, o un fallo general de las encuestas? **NO** · ¿Fue el CIS «el único que acertó»
   en 2023? **NO**.
6. Tres cifras arriba: «38 de 41» (elecciones con la izquierda sobreestimada), «8,0 puntos» (cuánto sobreestima a la
   izquierda frente a la derecha: el sesgo de bloques medio) y «39 de 41» (a la izquierda del consenso de las demás
   casas).
7. Pregunta 2: tres cajas de pruebas (moneda, por fecha electoral, frente al consenso), la **matriz de signos** (casas
   × elecciones) y, debajo, dos gráficos: el CIS frente a la media de las demás casas elección a elección, y el
   **ranking de puntos** (proporción de elecciones con la izquierda sobreestimada × sesgo de bloques medio). Descartados
   tras probarlos: la campana del azar, las barras «una entre cuántas», los ejes de probabilidad y de desviaciones
   típicas, el ranking frente al consenso y el conmutador antes/después.
8. El ranking y la matriz muestran las casas con 10 o más elecciones desde 2018 **y con sondeos en 2026**.
9. Pregunta 3: diana a la izquierda y tabla a la derecha al 50%, con las filas Izquierda / Derecha / Diferencia; debajo
   de la tabla, el ranking de distancia al resultado de las 21 casas.
10. Contenedor fluido, adaptado al ancho de pantalla, como el resto de la web.
11. Opción B de construcción (sección «Enfoques»): estudio publicado calculado desde las partes ya publicadas.

## Hechos verificados para los textos (fuentes en «Datos y método»)

- Tezanos fue secretario de Estudios y Programas de la Comisión Ejecutiva Federal del PSOE desde junio de 2017 y
  responsable de los sondeos del partido; el Consejo de Ministros lo nombró presidente del CIS el 29 de junio de 2018
  (tomó posesión el 30) y dejó la Ejecutiva en julio de 2018.
- El barómetro de septiembre de 2018 fue el primero con la nueva estimación de voto (dejó de aplicar el modelo
  tradicional de ajuste y pasó a una estimación basada en el voto directo más la simpatía por cada partido): es el
  mismo hecho que ya cuenta la tarjeta de Método.
- Las tres fuentes externas, con su URL exacta (van en `meta.sources` y en «Datos y método»):
  - «José Félix Tezanos, presidente del CIS, abandona su cargo en la Ejecutiva Federal del PSOE», El Independiente,
    9-7-2018: `https://www.elindependiente.com/politica/2018/07/09/jose-felix-tezanos-presidente-del-cis-abandona-cargo-la-ejecutiva-federal-del-psoe/`
  - «Los datos sorprendentes de la cocina del CIS de Tezanos», El Independiente, 25-9-2018:
    `https://www.elindependiente.com/politica/2018/09/25/los-datos-sorprendentes-cocina-cis-tezanos/`
  - Perfil de José Félix Tezanos en la UNED: `http://portal.uned.es/portal/page?_pageid=93%2C63197807&_dad=portal&_schema=PORTAL`
- Las cifras de la maqueta salen de las partes publicadas el 24-10-2026; las de esta spec son ilustrativas, la página
  las calcula en cada publicación.

## Objetivos

1. Un **estudio publicado** `sesgo-cis` (`site/v1/studies/sesgo-cis/`) con sus partes, gemelos CSV, esquemas,
   entrada en el manifiesto y rutas de la API, calculado **desde las partes ya publicadas** del bundle (D2, D3, D4).
2. Las declaraciones de los estudios (partes, esquemas, CSV, exportadores, títulos) **por estudio**, sin cambiar nada
   del contenido ni del contrato del `voto-cera` (D1).
3. La página `/analisis/sesgo-cis` con la estructura de la maqueta, textos con cifras vivas, gráficos propios y
   contenedor fluido; entra sola en el índice `/analisis` y en la banda de Inicio (D5, D6, D7).
4. Comprobación pública de redacción aplicada a la página nueva (D7).
5. Documentación: contrato, README de despliegue, y una nota en la tarjeta de Método (D8).

## No objetivos

- Ningún acceso a la base de datos: el estudio no instancia `Computer` ni lee `polls`. Si una cifra no está en las
  partes publicadas, no entra.
- Ningún cambio en los análisis publicados (`poll-errors`, `house-effects`), en el modelo, en el rating ni en la
  tarjeta de Método (salvo la nota de D8).
- Sin predicción registrada, sin «comprobación» tras el 29-N, sin pastillas de «Predicción registrada».
- Sin conmutador antes/después, sin gráfico de escaños publicados (PP + VOX frente a 170), sin campana del azar.
- Sin selector de ámbito ni barra de evento (el estudio es nacional; los ámbitos entran todos).
- Nada de nginx, Docker, CSP ni modo oscuro.

## Anti-objetivos (lo que haría fracasar la fase aunque funcionara)

- Que una cifra del texto esté escrita a mano en la plantilla: todo número sale de las partes publicadas, salvo los
  hechos del contexto (fechas y cargos de Tezanos, el cambio de método, el relato de 2023).
- Que las cifras del CIS de esta página **no coincidan** con las de la diana de su perfil (`/casas/cis`) y la tarjeta
  de Método: las filas son las mismas (D2), y un test lo fija con un bundle sintético.
- Que la página se vea en un contenedor fijo o que los gráficos no se redibujen al cambiar el ancho (decisión 10).
- Perder la lectura sin JavaScript: textos, veredictos, cifras y tablas salen del servidor; sólo los gráficos los dibuja
  el navegador.
- Que el estudio falle o publique cifras a medias cuando falte un ámbito: un ámbito sin análisis se omite y se anota en
  `meta.inputs`; si falta el ámbito nacional o el evento de 2023, la publicación se rechaza (D4).
- Romper el `voto-cera`: su publicación, su API, su página y sus tests no cambian de resultado.

## Enfoques considerados

- **A. Estudio calculado desde la base de datos** (como el CERA): un paquete que instancia el modelo de los 19 ámbitos
  y repite el cálculo de `poll-errors`. Descartado: duplica un cálculo pesado que ya se publica por ámbito y cualquier
  diferencia de parámetros dejaría la página y el perfil del CIS sin cuadrar.
- **B. Estudio calculado desde las partes publicadas** (elegido por Luis): el mismo patrón de publicación (partes,
  CSV, esquemas, manifiesto, API), pero el cálculo lee `poll-errors` de cada ámbito y las partes del evento de 2023 del
  propio bundle. Garantiza por construcción las mismas cifras que el perfil y Método, tarda segundos, no necesita base
  de datos y se encadena tras `publish analysis`.
- **C. Página calculada al servir** (como la tarjeta de Método): sin publicación. Descartado: dejaría la sección
  Análisis con dos mecánicas, sin CSV ni API, y leyendo más de veinte partes de S3 por trabajador en frío.

## D1. Declaraciones por estudio (`mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`)

Hoy `STUDY_PARTS`, `STUDY_SCHEMAS`, `STUDY_CSV`, `STUDY_CSV_OF` y `STUDY_EXPORTERS` son globales y describen el CERA.
Pasan a ser **diccionarios por slug**, con el CERA tal cual está:

```python
STUDIES = ('voto-cera', 'sesgo-cis')
STUDY_PARTS = {'voto-cera': ('meta', 'census', ..., 'check'),
               'sesgo-cis': ('meta', 'summary', 'elections', 'houses', 'matrix', 'election')}
STUDY_SCHEMAS = {'voto-cera': {part: 'study-' + part ...},            # nombres actuales, sin cambios
                 'sesgo-cis': {part: 'sesgo-cis-' + part ...}}
STUDY_CSV = {'voto-cera': (...los 11 de hoy...), 'sesgo-cis': ('elections', 'houses', 'matrix', 'election-houses')}
STUDY_CSV_OF = {'voto-cera': {...}, 'sesgo-cis': {'elections': ('elections',), 'houses': ('houses',),
                                                   'matrix': ('matrix',), 'election': ('election-houses',)}}
```

- `path_study(study, part)` y `path_study_csv(study, name)` validan contra las tuplas del estudio. `SCHEMAS` gana las
  seis entradas `sesgo-cis-*` (claves obligatorias en D3). `_check_study` no cambia.
- `publish/studies.py`: `STUDY_TITLES` gana el slug; `STUDY_EXPORTERS` pasa a `{study: {part: exportador}}`;
  `publish_study` recibe por estudio qué calcula y cómo: un pequeño registro `STUDY_RUNNERS = {study: {'compute':
  callable, 'registrable': bool}}` (nombre a elección del constructor). Para `voto-cera` todo sigue igual (registro,
  `checked`, lectura del `meta` anterior); para `sesgo-cis` no hay registro (`register` con este estudio es
  `PublishRefused('study: sesgo-cis has nothing to register')`), `checked` es falso y el `meta` no lleva las claves de
  registro. La entrada del manifiesto conserva su forma `{published_at, registered_at: null, checked: false}`.
- El exportador del `meta` también es por estudio: `export_study_meta` (CERA) se queda; el nuevo es
  `export_lean_meta` (D3).
- Consumidores que pasan a usar el estudio como clave: `Studies.part_action` (`STUDY_CSV_OF[study]`),
  `webapi.checks` (`check_part(part, STUDY_PARTS[study])`), `pages.analisis._csv_links(study)` (con un `CSV_LABELS`
  por estudio), `tests/test_webapi_unit.write_study` (escribe las partes y CSV del estudio que se le pasa) y el test
  `test_study_paths_follow_the_published_layout_and_reject_bad_input`.
- El paquete del CERA (`mtpy/lib/studies/cera`) y los pesados (`statsmodels`) siguen importándose dentro de las
  funciones: el test `test_importing_publish_does_not_load_the_study_package` sigue en pie y gana el paquete nuevo.

## D2. Las entradas: qué partes lee el estudio y con qué filtros

Todo se lee con un `BundleReader(fs, prefix=bundle.PREFIX)` sobre el `fs` del job: **siempre el prefijo publicado**,
también en `dry_run` (donde el escritor usa `site-dry/v1`), porque las entradas son la verdad publicada (decisión mía;
el CERA lee del escritor sólo su propio `meta` anterior). El catálogo de ámbitos y sus nombres salen de
`webapi.checks.catalogue()` (`data/es-scopes.csv`), en su orden.

**Filas del CIS (y de las demás casas).** Para cada ámbito del catálogo con `analysis/{scope}/meta.json` y
`poll-errors.json`: las `rows` de `poll-errors` cuyo `event_date` tenga carpeta publicada en `events/{scope}/`
(`reader.list_events(scope)`), exactamente el filtro de `pages.elecciones.error_rows`. Son el último sondeo de cada
casa con trabajo de campo entre 6 y 42 días antes de cada elección con resultado (`drange (6, 42)`, `n_last 1`, sin
agregadores, paneles online, wban ni exit), con los campos `event_date, event, pollster, days, sample_size, weight,
error_avg, bias_avg, error_blocks, bias_blocks, bias` y las columnas de bloque presentes (`Izquierda`, `Derecha`;
una puede ser nula, p. ej. Navarra 2019 sin `Derecha`). Un ámbito sin análisis se omite y se anota.

**Los dos periodos** (parámetros, D4): «con Tezanos», `event_date >= 2018-09-01`; «sin Tezanos»,
`2008-01-01 <= event_date < 2017-01-01`. Etiquetas públicas fijas: «2018 – 2026» (el año final es el de la última
elección con sondeo del CIS) y «2008 – 2016».

**Medidas por fila.** `Izquierda` (error en el bloque de izquierdas, positivo si el sondeo le dio más voto),
`Derecha`, y el **sesgo de bloques** `lean = -error_blocks` (= `Izquierda − Derecha`, positivo hacia la izquierda;
`error_blocks` publicado es `Derecha − Izquierda`, sondeo menos resultado). El consenso de una elección para una casa es
la **media simple del error `Izquierda` de las demás casas** con fila en esa elección (`n_others`); sin otras casas no
hay consenso para esa fila.

**El evento de 2023** (`events/es/2023-07-23/`): `meta` (`blocks`, `name`), `polls` (todos los sondeos con error,
`days` ascendente; se toma por casa el de menos `days` con `days <= 60`, criterio de la maqueta), `results` (porcentaje
de cada partido). Las cuotas por bloque de cada sondeo y del resultado se reconstruyen como `resultado del bloque +
error` (suma de `pct` de los partidos de `meta.blocks` para el resultado; para el sondeo, resultado + `Izquierda`/
`Derecha` de la fila). Fallo si falta cualquiera de las tres partes.

**Casas con sondeos en 2026:** la unión de (a) las casas con algún sondeo de fecha `>= 2026-01-01` en la parte
`polls` del último run nacional (`manifest.scopes.es.latest` → `runs/es/{run}/polls.json`), y (b) las casas con fila
de `poll-errors` en una elección de 2026. Sin manifiesto o sin run nacional, sólo (b), y se anota en `meta.inputs`.

## D3. Las partes de `sesgo-cis`

Envoltorio como todo estudio (`scope`, `run_id` y `mode` nulos; `schema` `sesgo-cis-{parte}@1`). Nombres de campo en
inglés (regla del repo); los textos públicos, en la página. Redondeos: errores y medias a 2 decimales, probabilidades
a 3 (`PROB_DECIMALS`) salvo las que se publican como «una entre N», que van enteras.

| Parte | Contenido (claves obligatorias en el esquema; el constructor fija el resto) |
|---|---|
| `meta` | `study`, `title`, `subtitle` (con el «38 de 41» vivo), `pollster` («CIS»), `periods` (`{with: {from, to, label, first_event, last_event, n}, without: {...}}`), `blocks` (`['Izquierda', 'Derecha']`), `params` (los de D4), `inputs` (por ámbito: `run_id` y `generated_at` del `poll-errors` leído, o `skipped`; el evento de 2023 con su `generated_at`; el run nacional usado para las casas de 2026), `sources`, `commit`, `dirty`, `versions`, `seconds`. Sin claves de registro ni `checked_at`. |
| `summary` | Las cifras que citan los textos, para cada periodo (`with`, `without`): `n`, `k_left` (filas con `Izquierda > 0`), `n_right`, `k_right` (filas con `Derecha < 0`), `mean_left`, `mean_right`, `n_lean`, `k_lean` (filas con `lean > 0`), `mean_lean`, `exceptions` (etiquetas de las elecciones con `Izquierda <= 0`), `consensus` (`n`, `k` = filas con el CIS a la izquierda del consenso, `mean_diff`), `dates` (`n`, `k` = fechas electorales con media del CIS > 0), `chance` (`coin`, `dates`, `consensus`: cada uno `{n, k, ways, total, p, odds, odds_exact}` como `metodo.sign_chance`), `t` (`all`, `dates`, `consensus`: `{n, mean, sd, t, p}`), y, sólo en `with`, `industry` (`{n_events, left, right, mean}`: el «fallo general de las encuestas» del texto de la pregunta 2, definido **sin el CIS**: elecciones del periodo con tres o más casas distintas del CIS con fila, de las que se cuentan las que tienen media del error `Izquierda` de esas casas mayor que cero (`left`), menor (`right`) y la media de esas medias (`mean`); es la misma medida de consenso que usa el resto de la página, y deja al CIS fuera para no inclinar la cifra hacia la propia tesis). Y `election` (D3, parte `election`) resume: `cis_dist`, `consensus_dist`, `closer` (casas más cerca del resultado que el CIS), `rank` (posición del CIS). |
| `elections` | Una fila por elección con sondeo del CIS en cualquiera de los dos periodos: `period`, `scope`, `scope_name`, `label` («Andalucía 2026», «Generales 2023»), `event`, `event_date`, `year`, `days`, `Izquierda`, `Derecha`, `lean`, `consensus` (media de las demás), `n_others`, `others_left` (cuántas de las demás sobreestimaron a la izquierda). CSV `elections`. |
| `houses` | El ranking: una fila por casa con 10 o más elecciones desde 2018 y sondeos en 2026 (la del CIS incluida): `pollster`, `slug`, `n`, `k_left`, `share_left`, `n_lean`, `mean_lean`, `mean_left`, `active_2026`, `cis` (bool). CSV `houses`. |
| `matrix` | La matriz de signos del periodo con Tezanos: `events` (`[{scope, event_date, label, short}]`, cronológico y en orden de catálogo a igual fecha; `short` es el código ISO del ámbito que ya lleva el `scode` sin el prefijo `es-`, en mayúsculas, más el año en dos cifras: «AN 18», «CB 23», «PV 24», y «GEN 23» para las generales; sin columna nueva en el catálogo ni diccionario de nombres; el nombre completo va en el tooltip), `houses` (las mismas que `houses`, el CIS primero y las demás por `share_left` descendente), `cells` (`[[x, y, Izquierda]]` sólo donde hay fila). CSV `matrix` en formato largo (`pollster, scope, event_date, Izquierda`). |
| `election` | 2023: `scope`, `date`, `name`, `blocks` (partidos de cada bloque), `max_days`, `result` (`{Izquierda, Derecha, gap}`), `cis` (`{date, days, Izquierda, Derecha, gap, e_left, e_right, e_gap, dist}`), `consensus` (lo mismo para la media de las otras casas, con `n`), `houses` (una fila por casa, la del CIS incluida: `pollster`, `slug`, `date`, `days`, `Izquierda`, `Derecha`, `gap`, `e_left`, `e_right`, `dist`), por `dist` ascendente. CSV `election-houses`. |

`dist` es la distancia al centro de la diana, `hypot(e_left, e_right)`. `gap` es `Izquierda − Derecha` en puntos de
voto. `meta.json` se escribe la última; el orden de escritura es el de `STUDY_PARTS['sesgo-cis'][1:]`.

## D4. El cálculo: `mtpy/lib/studies/lean/`

Paquete nuevo (nombre en inglés, «lean» = inclinación), ligero (`pandas` opcional, sin `statsmodels`):

- `data.py`: lectura de las entradas de D2 a partir de un `BundleReader` (`read_rows(reader, catalogue)`,
  `read_election(reader, scope, date)`, `active_pollsters(reader)`), devolviendo listas de dicts y las anotaciones de
  `inputs`. Lanza `MissingInput(parte)` cuando falta el ámbito nacional o el evento.
- `stats.py`: `sign_chance(n, k)` (la misma cuenta que `metodo.sign_chance`; Método pasa a importarla de aquí o se deja
  una copia idéntica con un test que compare ambas, a elección del constructor), `t_test(values)` (media, desviación
  típica muestral, t, p bilateral con `scipy.stats.t`), `by_date(rows)`, `consensus(rows)`, `house_rows(rows,
  active, min_n)`, `matrix(rows, houses)`, `industry(rows)`.
- `study.py`: `compute_lean(reader, catalogue, params=None) -> LeanStudy` (dataclass con los bloques de D3) y los
  parámetros por defecto `PARAMS`: `pollster 'CIS'`, `since '2018-09-01'`, `control ['2008-01-01', '2017-01-01']`,
  `blocks ['Izquierda', 'Derecha']`, `min_elections 10`, `max_days 60`, `active_from '2026-01-01'`, `election
  {scope 'es', date '2023-07-23'}`. Viven en `data/sesgo-cis/params.json` (como `data/cera/params.json`), con su fila
  en `data/README.md`; el estudio los lee del `app.data` y los publica en `meta.params`.
- La relación con la tarjeta de Método: `HOUSE_EXAMPLE` sigue en `metodo.py` (misma casa, misma fecha). Un test del
  estudio con un bundle sintético comprueba que `summary.with.n` y `k_left` coinciden con `house_effects_example()`
  sobre el mismo bundle (anti-objetivo 2).

## D5. La publicación y el job

- `publish.publish_study('sesgo-cis', writer)` → calcula (`compute_lean` con un `BundleReader` de prefijo `PREFIX`
  sobre `writer.fs`), exporta las partes de D3 con sus CSV, escribe `meta` la última, actualiza el manifiesto
  (`studies['sesgo-cis'] = {published_at, registered_at: None, checked: False}`) y devuelve `{study, registered_at:
  None, checked: False, seconds}`. Tiempos por paso como el CERA.
- Comando: `python job.py publish '{"what": ["study"], "study": "sesgo-cis"}'`. Encadenado tras los análisis:
  `python job.py publish '{"what": ["analysis", "study"], "scopes": [...], "study": "sesgo-cis"}'` (el orden de
  acciones del job ya es fijo: análisis antes que estudio). `register` con este estudio se rechaza (D1).
- Sin guardia LOREG: el estudio publica errores de elecciones pasadas, no sondeos de la convocatoria.
- `deploy/README.md` («### Estudios», tablas de `what` y parámetros) y `docs/web/contrato.md` («Estudios», «Rutas de
  estudios», árbol del paquete) ganan el estudio: partes, CSV, esquemas y el aviso de que las partes se declaran por
  estudio.

## D6. La página `/analisis/sesgo-cis`

**Rutas y registro.** `pages.analisis.STUDIES` gana `'sesgo-cis': {'template': 'analisis_sesgo_cis.html'}`;
`CONTEXTS` gana `lean_context`; el módulo de vista es `mtpy/lib/pages/lean.py` (y entra en la lista de
`test_pages_package_layout`). Sin cambios de rutas ni de menú: el índice `/analisis` y la banda de Inicio listan los
estudios publicados en el orden de `bundle.STUDIES` (CERA primero; decisión mía, fácil de invertir).

**Contexto** (`lean_context(state)`): lee las seis partes con `study_part`; cualquier parte o clave ausente da `None`
y la plantilla lo guarda (nunca un error, como `cera.py`). Entrega: `study_meta`, `verdicts` (SÍ / NO / NO con tonos
`q1`, `q3`, `q3`), `kpis` (3 filas de `m.kpis`), `summary`, `periods`, `elections` (filas del CIS por periodo, las de
«con Tezanos» primero y en orden de fecha descendente), `houses`, `election`, `csv_links`, `sources`, `notes`
(claves de la nota técnica) y la isla `initial` con sólo lo que dibujan los gráficos: `bull` (`{with: rows, without:
rows, blocks}`), `matrix`, `consensus` (`[{x, y, label, national}]`), `ranking` (`[{x, y, name, k, n, cis}]`),
`election` (`{houses: [...], consensus: {...}, blocks}`) y `distance` (`[{name, dist, cis, consensus}]`).

**Estructura** (ids de la maqueta; cada tarjeta con `m.card_start`/`qcard` como el CERA):

- `#study-head`: pastillas «Análisis», «Evidencia histórica · {n} elecciones desde 2018», «Actualizado el
  {published_at}»; `h1.study-title` «El sesgo del CIS de Tezanos»; `p.study-subtitle` con la entradilla y el
  «{k_left} de {n}» vivo.
- `nav.toc`: Resumen · 1 · ¿Se inclina a la izquierda? · 2 · ¿Puede ser azar? · 3 · ¿Acertó en 2023? · Datos y método.
- `m.kpis`: «Elecciones con la izquierda sobreestimada» `{k_left} de {n}` (apoyo: «último sondeo del CIS antes de cada
  elección analizada desde que lo preside Tezanos · {k_right} de {n_right} con la derecha subestimada»); «Cuánto
  sobreestima a la izquierda frente a la derecha» `{mean_lean} puntos` («de media en cada elección: {mean_left} de más
  al bloque de izquierdas y {|mean_right|} de menos al de derechas · sin Tezanos (2008 – 2016), el mismo CIS:
  {without.mean_lean}»); «A la izquierda del consenso de las demás encuestadoras» `{consensus.k} de {consensus.n}`
  («{mean_diff} puntos más a la izquierda que la media de las otras casas en la misma elección · antes, {without.k} de
  {without.n}»).
- `#resumen`: «Tres preguntas, tres respuestas», las tres tarjetas-enlace (`answer`) con una frase viva cada una y la
  nota azul («Medimos lo que dicen los sondeos frente a lo que dicen las urnas. No medimos intenciones…»).
- `#p1` (SÍ): entradilla con los hechos verificados (Tezanos, nombramiento, cambio de método) y el método de
  comparación; `grid2` con dos dianas cuadradas «Sin Tezanos: {without.n} elecciones (2008 – 2016)» y «Con Tezanos:
  {with.n} elecciones (2018 – 2026)» (`#bull-without`, `#bull-with`); las dos cajas de comparación (`.cmp`: elecciones
  con la izquierda sobreestimada, sesgo medio en cada bloque, a la izquierda del consenso); la tabla de las {n}
  elecciones con Tezanos (Elección, Fecha, Días antes, Izquierda, Derecha con `shade_around(0, 8)`), plegada a 8 filas
  con «Ver las {n} elecciones» (`wireMore`); `.sum` con el veredicto y las excepciones.
- `#p2` (NO): entradilla; `.tests` con las tres cajas (moneda «1 entre {odds}», fecha «1 entre {odds}», consenso
  «{k} de {n}», la tercera resaltada); la matriz (`#matrix`, título «Casa por casa y elección por elección: el CIS es
  la única casa prácticamente monocolor», subtítulo de lectura: rojo más a la izquierda, azul menos, intensidad por
  error, GEN en negrita, detalle al pasar el ratón); `grid2` con «El CIS, frente a la media de las demás casas en la
  misma elección» (`#consensus-chart`) y «Ninguna otra encuestadora se acerca» (`#ranking-chart`), subtítulos de la
  misma altura; `.sum`; la nota técnica (`.tech`) con las tres t de cada periodo.
- `#p3` (NO): entradilla con el relato de 2023 (mayorías absolutas publicadas por las demás, el CIS con el PSOE por
  delante, la mayoría no llegó); `grid2` al 50% con la diana de 2023 (`#bull-2023`: resultado en el centro, CIS en
  rojo, consenso como rombo, nombres de las casas en el tooltip y etiquetas sólo en los extremos) y, a la derecha, la
  tabla (CIS, Consenso, Resultado, Error CIS, Error consenso × Izquierda, Derecha, Diferencia; subtítulo con la
  composición de los bloques) y debajo el ranking «Quién estuvo más cerca del resultado en los bloques»
  (`#distance-chart`, barras por `dist`, CIS en rojo, consenso en negro, «el CIS fue la {rank}.ª de {n}»); `.sum`.
- `#datos`: las fuentes (análisis de errores por ámbito con enlace a `/elecciones`, perfil del CIS, tarjeta de
  Método, y las tres fuentes externas de «Hechos verificados» con sus URL), las definiciones (elecciones analizadas,
  error por bloque, sesgo de bloques, consenso, fallo general sin el CIS, periodos, pruebas) y las cuatro pastillas
  CSV.
- «Cómo leerlo» (`m.how`) en las tres tarjetas-pregunta, breve, como en el CERA (decisión mía: coherencia de la
  sección; sin él la página sigue completa).

**Textos.** Voz «nosotros», «sólo» con tilde, «encuestadora»/«casa», nunca «empresa», comillas «»; cifras con los
filtros `num`/`int`/`delta`; «Sobrestimó a la Izquierda en {k} de {n} elecciones ({share}) · {lean} puntos de media»
en el tooltip del ranking (texto de Luis). Los textos de la maqueta v23 son la referencia; el constructor los traslada
con las variables vivas. `assert_public_wording` se aplica a `/analisis/sesgo-cis` (y, de paso, el constructor puede
añadir `/analisis/voto-cera`, hoy sin comprobar; si la página del CERA no pasa, se deja fuera y se anota).

## D7. Los gráficos y el contenedor fluido

- **Dianas** (`charts/bullseye.js`, sin cambios de API salvo lo indicado): ejes `Derecha` (x) e `Izquierda` (y),
  anillos 2, 5 y 10, generales en azul y autonómicas en naranja con `onTop`, etiqueta sólo en las generales
  (`labelOf`), marcas de eje −10, −5, 0, +5, +10 (opción nueva `ticks` o `limit` fijo: el constructor elige; los
  puntos que superan 10 siguen dentro). La de 2023: anillos 2 y 5, marcas de 2 en 2, puntos azules, el CIS en rojo
  (`highlight`), el consenso como rombo negro (un segundo `highlight` con símbolo, o una serie extra), etiquetas de los
  extremos con `labelLayout.hideOverlap`.
- **Matriz** (`charts/matrix.js`, nuevo): `matrixOption({events, houses, cells}, opts)` y `renderMatrix`. Celdas
  `heatmap` con color por signo e intensidad por `|valor|` con tope 8 (rojo `#d62728` positivo, azul `#1d4f86`
  negativo, como la escala de las tablas), borde blanco, columnas arriba rotadas 90° con las generales en negrita,
  filas en el orden de la parte, alto `filas × 26 + 70`, tooltip «{casa} · {elección}: error en el bloque de
  izquierdas {valor}». En móvil, envoltorio con desplazamiento horizontal (`.chart-scroll`) y ancho mínimo (columnas ×
  18 px).
- **CIS frente al consenso** (`charts/scatter.js`): ejes −10..15 (x) y −5..15 (y) de 5 en 5, diagonal discontinua
  (opción nueva `diagonal: true`), puntos naranja (autonómicas) y azul (generales, etiquetadas con el año).
- **Ranking** (`charts/scatter.js` o módulo propio `charts/ranking.js` con `rankingOption`): x 0–100% (etiquetas con
  «%»), y −2..10 de 2 en 2, líneas discontinuas en 50% y 0 con el punto «Sin sesgo» (etiqueta bajo el punto, fuera de
  las líneas), recuadro punteado sin título alrededor de las casas que no son el CIS (mínimos y máximos ± margen), CIS
  en rojo con etiqueta «CIS · {k} de {n} · {lean} puntos» a su izquierda, las demás en gris con nombre; colocación de
  etiquetas sin solapes: `labelLayout: {hideOverlap: true}` con prioridad al CIS y a las casas más alejadas del centro
  (las ocultas quedan en el tooltip). Tooltip con el texto de Luis.
- **Distancia al resultado** (`charts/bars.js` con una opción exportable `rankOption`, o módulo propio): barras
  horizontales por `dist` ascendente, CIS en rojo, consenso en negro y en negrita, valor a la derecha, sin eje.
- Todos los módulos nuevos exportan su `*Option` para los tests con `node` (`tests/test_js_charts.py`).
- **Fluido**: la página usa el `main` sin ancho máximo del sitio; los gráficos se montan con `mountChart`
  (`ResizeObserver`); las dianas cuadradas se hacen con una regla por clase, `.chart.square { aspect-ratio: 1 / 1;
  height: auto; max-width: 480px; margin: 0 auto }` (y `max-width: 560px` para la de 2023), que el constructor puede
  aplicar también a los ids actuales sin cambiarles el comportamiento; `grid2` colapsa a una columna por debajo de
  900 px como el resto; la tabla de 2023 cabe en media anchura con las filas «Izquierda», «Derecha», «Diferencia».
- JS de página: `web/dist/js/pages/analisis-sesgo-cis.js` con el patrón de `analisis-cera.js` (`readInitial`,
  `tryDraw`, `wireMore`, `mountWhenReady`).

## D8. Documentación, nota en Método y memoria

- Contrato y README (D5). `data/README.md` gana `data/sesgo-cis/params.json`.
- La tarjeta «El CIS: … elecciones en la misma dirección» de `/metodo/promedio` gana una frase final con enlace: «El
  análisis completo, con el CIS de antes de 2018, las demás casas y 2023, está en /analisis/sesgo-cis» (sólo cuando el
  estudio está publicado; `house_effects_example` no cambia).
- Sin cambios en `AGENTS.md`.

## Lo que queda al constructor

- Nombres exactos de los campos dentro de la forma de D3, el reparto entre `data.py`/`stats.py`/`study.py`, y si el
  ranking y la distancia van en `scatter.js`/`bars.js` o en módulos propios (siempre con `*Option` exportado).
- Los textos finales a partir de la maqueta, con las variables vivas y las reglas de redacción.
- Fixtures sintéticos: un bundle con 2 o 3 ámbitos, 6 u 8 elecciones por periodo, 4 casas y un evento de 2023 mínimo,
  suficiente para fijar todas las cifras de `summary` a mano en los tests.
- Los detalles CSS de la matriz en móvil y de las cajas `.cmp`/`.tests`/`.tech` (estilos nuevos en `site.css`, con
  los tokens del sitio).

## Pruebas (mínimas)

- `tests/test_lean_*.py`: `sign_chance(41, 38)` (`ways 11522`, `odds 190000000`), `t_test` con valores conocidos, el
  consenso y las fechas sobre filas sintéticas, `compute_lean` sobre un bundle sintético (cifras de `summary`,
  exclusiones por `min_elections` y por «sin sondeos en 2026», Navarra sin `Derecha`), `MissingInput` sin el nacional o
  sin el evento, y la igualdad con `house_effects_example()` sobre el mismo bundle.
- `tests/test_publish_studies.py`: partes y CSV del `sesgo-cis` validan, `meta` la última, manifiesto, `register`
  rechazado, `voto-cera` intacto.
- `tests/test_bundle_unit.py`: rutas y esquemas por estudio.
- `tests/test_web_api.py`: índice con los dos estudios, partes y CSV del nuevo, errores.
- `tests/test_pages.py`: contexto (cifras desde las partes, KPIs, isla, enlaces CSV), página (pastillas, índice,
  veredictos, tres `kpi`, ids de los gráficos con `aria-label`, tablas, pastillas CSV = 4), `assert_public_wording`,
  índice y banda de Inicio con dos estudios, 404 sin publicar; `test_pages_package_layout` con `lean`.
- `tests/test_js_charts.py`: `matrixOption`, el ranking y la distancia (formas de las opciones), `scatter` con
  `diagonal`, `bullseye` con las marcas.
- Suite completa en verde (hoy, sin esta fase: la de `dev`).

## Cierre del constructor

Revisión de un constructor independiente (2026-10-24) sobre esta spec, la maqueta y el código. Tres preguntas, las
tres resueltas en el documento sin cambiar lo que se construye:

1. **Las URL de las tres fuentes externas.** Añadidas en «Hechos verificados»; van en `meta.sources` y en «Datos y
   método».
2. **Los códigos cortos de las columnas de la matriz.** Decisión: el código ISO que ya lleva el `scode` del catálogo
   (`AN`, `CB`, `CN`, `PV`…) más el año, y `GEN` para las generales; ni columna nueva en `data/es-scopes.csv` ni
   diccionario de etiquetas en el código (D3, parte `matrix`).
3. **La definición de `industry`** («fallo general de las encuestas»). Decisión: sin el CIS, con tres o más casas
   distintas y la media del error `Izquierda` de esas casas como criterio, la misma medida de consenso del resto de la
   página (D3, parte `summary`). La cifra pública de la maqueta («30 de 40», que incluía al CIS y usaba el sesgo de
   bloques) cambiará; la página la calcula.

Observaciones del revisor confirmadas: `lean = Izquierda − Derecha` equivale a `−error_blocks` en todos los ámbitos
(se calcula desde las columnas de bloque); el contenedor ya es fluido; `scipy` ya está en `requirements.txt`; el
criterio «menos `days` con `days <= 60`» de 2023 da las 21 casas de la maqueta.

Decisiones mías durante la redacción, para que Luis las revise al leer: las entradas se leen siempre del prefijo
publicado, también en `dry_run` (D2); el índice y la banda de Inicio listan los estudios en el orden de
`bundle.STUDIES`, el CERA primero (D6); «Cómo leerlo» breve en las tres tarjetas-pregunta (D6); la comprobación de
redacción pública se aplica a la página nueva y, si pasa, también a la del CERA (D6); los parámetros del estudio en
`data/sesgo-cis/params.json` (D4); la nota de enlace en la tarjeta de Método sólo con el estudio publicado (D8).
