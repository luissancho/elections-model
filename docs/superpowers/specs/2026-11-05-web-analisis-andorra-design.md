# Web, Análisis: «El mercado de la fruta en Andorra: ¿aciertan más los sondeos «clandestinos» de última hora?» (`/analisis/andorra`)

Fecha: 2026-11-05. Estado: **aprobada por Luis el 2026-11-05** (brainstorm cerrado el 5 de noviembre; maqueta v2
aprobada «con esta versión»: `docs/superpowers/specs/2026-11-05-web-analisis-andorra-mockup.html`; revisión del
constructor en la sección final, su tercera pregunta cerrada por Luis: «Sigue con la A»). Cuarto estudio de la
sección «Análisis», tras el voto CERA, el sesgo del CIS (`2026-10-24-web-analisis-sesgo-cis-design.md`) y el rebaño
(`2026-11-05-web-analisis-rebano-design.md`, aprobado y con plan, aún sin construir), cuyo patrón (estudio publicado,
calculado desde las partes del bundle) se repite. Esta spec describe lo que cambia respecto de `dev` (eaba7c9).

## Intención

Punto 3 de la lista de análisis para redes (memoria `ideas-analisis-redes`): una página de **evidencia histórica**
sobre los sondeos publicados fuera de España durante la veda (los «clandestinos», el «mercado de la fruta en Andorra»),
para publicarla la semana del 16 de noviembre, darle hilo el 23-24 cuando abra el mercado de 2026, y completarla el
30 de noviembre con la fila de 2026 y su comprobación.

Lo que dicen los datos el 5 de noviembre (reconocimiento sobre la base y las partes publicadas; scripts en el scratchpad
de la sesión): en 116 pares casa a casa (33 en generales desde 1996, 83 en 40 autonómicas) el sondeo clandestino estuvo
más cerca del resultado que el último legal de la misma casa en 81 (70 %; binomial p ≈ 1e-5; mejora media 0,36 puntos
por partido); en las cinco generales con modelo publicado (2015-2023) el consenso clandestino batió al promedio
congelado en cuatro (error medio 1,70 → 1,21) y quedó más cerca del resultado en 19 de las 24 cifras de partido; el
modelo recalculado el día de la votación con los sondeos clandestinos dentro bajó de 1,70 a 1,38 (mejor en las mismas
cuatro); los sondeos de la jornada con 5.000 entrevistas o más afinaron en 11 de 12. Todas esas cifras son vivas: el
estudio las calcula desde las partes y la página las lee.

Titular (de Luis) y entradilla (maqueta v2, con las cifras vivas): **«El mercado de la fruta en Andorra: ¿aciertan más
los sondeos «clandestinos» de última hora?»** / «Los cinco días anteriores a la votación la ley prohíbe publicar sondeos
en España. Se publican fuera, y en las redes circulan como precios de la fruta. En las cinco generales desde 2015, el
consenso de esos sondeos estuvo más cerca del resultado que el promedio congelado en cuatro, y quedó más cerca del
resultado en 19 de las 24 cifras de partido. Lo que no vio nadie, tampoco lo vio la fruta: en 2023 dejó al PSOE tres
puntos por debajo de lo que sacó.»

## Decisiones de Luis (brainstorm y maqueta, 2026-11-05)

1. **Opción A**: estudio en `/analisis` más el hilo (no sólo el hilo).
2. Titular con «mercado de la fruta» y «clandestinos»; **cabecera larga** que explica qué son estos sondeos, por qué
   existen y cómo funcionan, resumiendo tres artículos (El Confidencial 18-12-2017, Onda Cero 6-11-2019, Heraldo de
   Aragón 5-2-2026), enlazados como fuentes. Los cuatro párrafos de la maqueta v2 («La ley», «Por qué existen», «Cómo
   funcionan», «Qué hacemos con ellos») son el texto de referencia.
3. En la página se llaman **«sondeo clandestino»** y «sondeos clandestinos» (nunca «sondeo de la veda»); «consenso
   clandestino» es la media del último sondeo clandestino de cada casa; «veda» sólo nombra el periodo.
4. Pregunta 1: **«¿Aciertan más que el último sondeo legal?»**; su gráfico es el **«antes → después» por casa**
   (dumbbell), no las barras.
5. Maqueta v2 aprobada tal cual («adelante con esta versión»); las cajas naranjas restantes se cierran como dice
   «Decisiones mías» y Luis las corrige al leer esta spec.

## Decisiones mías (cajas naranjas de la maqueta, pendientes de veto)

- **Perímetro de P1**: todas las elecciones con sondeos clandestinos y partes publicadas (generales y autonómicas desde
  1996) en la cifra, la tabla por elección y la tabla de casas; las generales desde 2015 en el gráfico.
- **La columna «con la fruta dentro»** sale del backtest: una tabla nueva `ban.csv` por ámbito (el modelo recalculado
  el día de la votación con los sondeos clandestinos dentro, sólo nowcast), exportada en la parte `model` de cada
  elección bajo la clave `ban`. El estudio sólo lee (D3). Sin la tabla, la columna es nula y la página lo dice.
- **Umbrales de P3**: tres pares por casa para la tabla de casas; «los grandes» desde 5.000 entrevistas.
- **La tarjeta «El 29-N»**: cerrada mientras la elección en curso no esté publicada con resultado; al publicarse, lista
  sus sondeos clandestinos uno a uno, su consenso, el promedio congelado y el resultado. Los sondeos clandestinos
  siguen fuera del promedio, del rating y de las tablas del sitio, y la página no enseña ninguno de la elección en
  curso.
- **Sin predicción registrada**, como el sesgo del CIS y el rebaño.
- La ventana del «último legal» es la del rating: su último sondeo legal con **42 días o menos** (`params.max_days`);
  un par con el legal más viejo no cuenta (en el reconocimiento entraba Metroscopia 2023 con 216 días: ya no).
- El **consenso clandestino** se hace con las casas que entran en el promedio (sin agregadores ni paneles abiertos,
  `DROP_MTYPES`); los **pares** comparan cada casa consigo misma y sólo dejan fuera los agregadores (ElectoPanel se
  compara consigo mismo, como dice la maqueta). Por eso las cifras de 2019 pueden moverse una décima respecto de la
  maqueta: la página manda.

## Hechos del contexto (sin cifras a mano)

- La LOREG (artículo 69.7) prohíbe «la publicación y difusión o reproducción de sondeos electorales por cualquier medio
  de comunicación» durante los cinco días anteriores a la votación. El cargador marca `ctype = 'wban'` las filas que
  Wikipedia señala como publicadas durante la veda (`loader.py`, color `#FFEAEA`) y `exit` las de boca de urna. Hoy
  quedan **fuera del promedio, del rating y de la web**: `Computer.drop_ctypes = ['wban', 'exit']`,
  `webapi.queries.WEB_DROP_CTYPES`, y Método lo dice en la viñeta «Veda». Sus errores (`error_avg`, `error_blocks`,
  `bias_blocks`) sí están calculados en la base, como los de cualquier sondeo.
- El `Forecaster` **no filtra `ctype`** (`load_polls` llama a `get_poll_series` sin `drop_contexts`): los clandestinos
  no entran en el promedio porque su `date` cae dentro del horizonte (6 días) y de la congelación de la veda, no por
  un filtro. D3 lo hace explícito.
- Las partes de evento (`pollsters`, `polls`, `cycle`, `results`, `model`, `series`) sólo se publican para elecciones
  **con resultado oficial** (`publish_event` lo exige): por eso una parte nueva con los sondeos clandestinos nunca
  puede enseñar los de la elección en curso durante la veda.
- La parte `pollsters` de cada elección ya trae, por casa, su último sondeo legal (`last_days`, `last_error_avg`,
  `last_error_blocks`…: el de menos días entre los que entran en el rating, que excluye `wban` y `exit`). Desde 1985
  ningún sondeo legal tiene menos de 6 días (los únicos con 1-4 días son de 1977 y 1979, anteriores a la ley).
- El backtest publica en la parte `model` de cada elección las filas de `shares`, `seats` y `blocks` por horizonte
  (6, 14, 30…); la página de la elección lee el horizonte más corto como «Lo que dijo el modelo» y la `series` del
  backtest es la del horizonte más corto que corrió (`shortest_horizon`). **Nada de eso cambia**: la tabla nueva del
  backtest va aparte (D3).

## Objetivos

1. Una **parte de evento nueva `ban`** con los sondeos clandestinos de cada elección publicada con resultado, en
   todos los ámbitos, con la misma forma que `cycle` más sus errores (D2).
2. El **contrafactual** en el backtest: `ban.csv` por ámbito (horizonte 0 con los clandestinos dentro), exportado en
   la parte `model` como `ban` (D3). El `Forecaster` gana `drop_ctypes` explícito.
3. Un **estudio publicado `andorra`** (`site/v1/studies/andorra/`) con sus partes, gemelos CSV, esquemas, entrada en el
   manifiesto y rutas de la API, calculado desde las partes publicadas (D4, D5).
4. La página `/analisis/andorra` con la estructura de la maqueta v2, textos con cifras vivas y condicionados, el
   gráfico «antes → después», descarga de cada tarjeta como imagen, contenedor fluido; entra sola en `/analisis` y en
   la banda de Inicio (D7, D8).
5. Comprobación pública de redacción, documentación, un enlace en la viñeta «Veda» de Método y en la memoria (D9).

## No objetivos

- Ningún acceso a la base de datos desde el estudio ni desde la página. Si una cifra no está en las partes, no entra.
- Ningún cambio en el promedio, en el rating, en la API de sondeos en vivo ni en las páginas de elección: los
  sondeos clandestinos siguen fuera de todo eso.
- Sin predicción registrada; sin ámbitos en el gráfico de P1 (sólo generales desde 2015); sin el contrafactual en las
  autonómicas (la tabla `ban.csv` puede correr en cualquier ámbito, pero P2 es nacional).
- Nada de nginx, Docker, CSP ni modo oscuro.
- La acción `studies` del `update` (D5 del rebaño) no es de esta spec: si ya existe cuando se construya, este estudio
  se declara `live`; si no, se republica a mano tras cada `publish event` (D5).

## Anti-objetivos

- **Que la página publique un sondeo durante la veda.** La parte `ban` sólo existe para elecciones con resultado; el
  estudio no lee los `polls` del run ni la base; la tarjeta «El 29-N» no enseña nada de la elección en curso hasta
  que esté publicada con resultado. Un test lo fija: con el evento en curso sin `results`, ninguna parte del estudio
  ni el contexto de la página contiene sus sondeos.
- Que una cifra del texto esté escrita a mano: todo número sale de las partes; los textos cuyo sentido depende del
  signo o del recuento se condicionan (D7).
- Que la página rompa cuando falte algo: sin `ban.csv`, la columna «con la fruta dentro» es nula; sin parte `model`,
  la elección cuenta en P1 y no en P2; un ámbito sin eventos se omite y se anota en `meta.inputs`; sin ningún par, la
  publicación se rechaza (D4).
- Que el contrafactual altere el backtest que ya se publica: `ban.csv` no entra en `metrics`, `by_horizon`,
  `calibration` ni `series`; con `ban` apagado, los resultados del backtest son idénticos al centésimo a los de hoy.
- Perder la lectura sin JavaScript: textos, veredictos, cifras y tablas salen del servidor; sólo el dumbbell lo dibuja
  el navegador.

## Enfoques considerados

- **A. Estudio desde la base de datos**: un paquete que lee `elections.polls` (los `wban`), repite el ajuste del
  promedio y corre el `Forecaster` el día de la votación. Descartado: rompe el patrón de los estudios (sin base), y el
  contrafactual son siete minutos de modelo dentro de un `publish`.
- **B. Estudio desde las partes, con dos entradas nuevas** (elegido): una parte de evento `ban` (los sondeos
  clandestinos, con sus errores ya calculados) y una tabla del backtest `ban.csv` (el contrafactual), ambas publicadas
  por los jobs que ya existen (`publish event`, `backtest`). El estudio tarda segundos y se encadena tras cada
  publicación.
- **C. Estudio desde las partes sin contrafactual**: igual que B sin `ban.csv`; la columna «con la fruta dentro» no
  existe. Descartado por Luis en la maqueta (caja 2): el contrafactual es la respuesta a «¿movieron el promedio?».

## D1. Declaraciones (`mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`)

```python
EVENT_PARTS = ('meta', 'pollsters', 'polls', 'cycle', 'ban', 'results', 'model', 'series')
EVENT_CSV = (..., 'ban', 'ban-legal', ..., 'model-ban', ...)
SCHEMAS['event-ban'] = {'parties': list, 'columns': list, 'polls': list, 'legal': list}
# ``event-model`` no cambia sus claves obligatorias: ``ban`` es una clave opcional (lista, presente cuando hay tabla)

STUDIES = ('voto-cera', 'sesgo-cis', 'andorra')   # (más 'rebano' cuando se construya; el orden es el de /analisis)
STUDY_PARTS['andorra'] = ('meta', 'summary', 'pairs', 'elections', 'cells', 'houses', 'latest')
STUDY_SCHEMAS['andorra'] = {part: 'andorra-' + part for part in STUDY_PARTS['andorra']}
STUDY_CSV['andorra'] = ('pairs', 'elections', 'cells', 'houses', 'latest-polls')
STUDY_CSV_OF['andorra'] = {'pairs': ('pairs',), 'elections': ('elections',), 'cells': ('cells',),
                           'houses': ('houses',), 'latest': ('latest-polls',)}
```

- `SCHEMAS` gana las siete entradas `andorra-*` con las claves obligatorias de D4.
- `publish/studies.py`: `STUDY_TITLES['andorra']`, `STUDY_EXPORTERS['andorra']`, `STUDY_META['andorra'] =
  export_ban_meta`, `STUDY_RUNNERS['andorra'] = {'compute': compute_for_publish (de publish/ban.py), 'registrable':
  False}` (más `'live': True` si la clave existe ya).
- Los exportadores viven en `mtpy/lib/publish/ban.py` (patrón de `lean.py`: importa el paquete del estudio dentro de
  `compute_for_publish`; `test_importing_publish_does_not_load_the_study_package` gana el paquete nuevo).
- Consumidores por estudio (`Studies.part_action`, `webapi.checks`, `pages.analisis._csv_links` con
  `CSV_LABELS['andorra']`, `tests/test_webapi_unit.write_study`): sólo ganan el slug.

## D2. La parte de evento `ban`

- `publish_event` escribe, tras `cycle`, la parte `ban` con los sondeos de contexto `wban` de la elección:
  `comp.filter_polls('results', drop_ctypes=['exit'])` restringido a `ctype == 'wban'` (o el filtro equivalente sobre
  `comp.series`: lo fija el constructor), exportado con `export_event_cycle` (misma forma: `parties`, `columns`,
  `polls`, un sondeo por fila con una columna por partido, `weight` producto de los pesos) **más** las columnas
  `days` (días hasta la elección) y los errores del sondeo tal como están en la base: `error_avg`, `error_blocks`,
  `bias_blocks` (`columns` las lista; una función `export_event_ban(frame, names)` que envuelve a `export_event_cycle`
  o lo parametriza). `date` es el **fin del trabajo de campo** (el cargador pone `date = end_date`; `pub_date` también
  lo es, así que no se exporta ni se enseña como fecha de publicación).
- La parte lleva además `legal`: el **último sondeo legal de cada casa con clandestino** (misma forma de fila: una
  por casa, la de menos `days` entre sus sondeos sin `ctype`, con sus errores), calculado en `publish_event` desde
  `comp.filter_polls('results', drop_mtypes=[], drop_ctypes=['wban', 'exit'])` restringido a los `pollster_id` de
  `polls`. Es lo que permite el par de un panel abierto como ElectoPanel, que no tiene fila en `pollsters` ni en `cycle`
  (`DROP_MTYPES` los deja fuera): el estudio forma los pares con `ban.polls` × `ban.legal`, nunca con `pollsters`.
  Esquema `event-ban`: `{'parties': list, 'columns': list, 'polls': list, 'legal': list}`.
- Sin sondeos `wban`, la parte se escribe con `polls: []` y `legal: []` (y sus CSV vacíos: `ban` y `ban-legal`), para
  que el estudio distinga «sin clandestinos» de «sin parte».
- El filtro de la publicación no toca `pollsters`, `polls` ni `cycle`: siguen sin los `wban`.
- `publish event` de una elección sin resultado sigue rechazándose (`NothingToPublish`): es la garantía legal.
- `meta` del evento gana `n_ban` (entero) en su resumen, para que `/elecciones` pueda decir «y {n} publicados fuera
  durante la veda» si algún día se quiere (no en esta spec).
- Guía de Luis tras construir: `publish '{"what":["event"],"scopes":"all","event_date":"all"}'` (todas las elecciones
  con resultado de todos los ámbitos), que es lo que la parte nueva necesita; ~25 s por elección.

## D3. El contrafactual en el backtest

- `Forecaster.__init__` gana `drop_ctypes: Optional[list[str]] = ['wban', 'exit']` y `load_polls` lo pasa a
  `get_poll_series(drop_contexts=...)`; `Simulator` gana el mismo parámetro y lo pasa al `Forecaster`; `run_case` y
  `run_backtest` también. El default hace **estructural** lo que Método ya afirma: ningún clandestino ni boca de urna
  entra en el promedio. Hecho comprobado (base, 2026-11-05): `date` es el fin de campo, y 17 de los 257 `wban` tienen
  `days = 6` (publicados el primer día de la veda con el campo acabado el día anterior: GESOP en las generales de 2015
  y 2016, ElectoPanel en las autonómicas de 2019 y 2020); ninguno pesa hoy en un promedio (los de GESOP tienen
  `weight_over = 0` por solape con su tracking legal; ElectoPanel es `online`, fuera por `DROP_MTYPES`), así que los
  resultados del backtest y de los eventos **no deben moverse**: la repetición del backtest de D3 (que Luis corre de
  todas formas) lo verifica contra las tablas publicadas, y un test lo fija con el backtest sintético. Lo que sí
  cierra el default es el riesgo de 2026: un tracking publicado el primer día de la veda con campo del día anterior
  tendría `days = 6` y, sin solape, peso, y hoy entraría en la actualización de ese día.
- `run_backtest(..., ban: bool = False)`: con `ban`, por cada elección corre **un caso más**: `run_case(scope,
  event_date, horizon=0, nowcast_only=True, drop_ctypes=['exit'], …mismos parámetros…)`, cuyas filas de `shares`
  (con `horizon = 0`) van a una tabla aparte `ban` del resultado (`out['ban']`), **no** a `shares`, `metrics`,
  `by_horizon`, `calibration` ni `series`. Sin sondeos clandestinos en la elección (ninguna fila `wban` en su drange),
  el caso se salta y se anota en `meta` como los demás saltos. `run.json` registra `ban` (bool) y las elecciones con
  caso.
- `backtest_scope` escribe `ban.csv` cuando hay tabla (`RESULT_FILES` la gana); con `ban` apagado no la escribe y
  **borra** la que hubiera (una tabla vieja describiría otro modelo). `BACKTEST_FILES`/`BACKTEST_OPTIONAL` de
  `publish/backtest.py` la ganan como opcional: `publish backtest` no la publica como parte (no es métrica), pero
  `export_event_model` la lee: la parte `model` gana la clave `ban` con las filas de la elección (mismas columnas que
  `shares`, horizonte 0) y el CSV `model-ban`; sin tabla, sin clave (y el CSV viejo se retira como los demás). El
  horizonte 0 **no** se suma a `horizons` ni a `meta.horizons` del evento: la página de la elección sigue leyendo el
  horizonte más corto como «Lo que dijo el modelo» y no debe encontrar el contrafactual ahí.
- El job `backtest` gana el parámetro `ban` (bool, falso por defecto; `python job.py backtest
  '{"scopes":["es"],"ban":true}'`); el `push` sube `ban.csv` con el resto. Coste: ~80 s por elección nacional.
- Guía de Luis: `backtest {"scopes":["es"],"ban":true}` → `backtest push` → `publish event` de las cinco generales
  desde 2015 → `publish study andorra`. Si no corre el backtest con `ban`, la página funciona sin la columna.

## D4. Las entradas y el cálculo: `mtpy/lib/studies/ban/`

**Entradas** (todo con un `BundleReader(fs, prefix=bundle.PREFIX)` sobre el `fs` del job, también en `dry_run`):

- Los ámbitos del manifiesto (`manifest.scopes`) y, por ámbito, `reader.list_events(scope)`. Por elección: `meta`
  (`name`, `main`, `blocks`, `featured`), `ban` (`polls`, los clandestinos, y `legal`, el último legal de cada una de
  sus casas; una elección sin la parte se anota en `meta.inputs` y se omite), `pollsters` (`last_days`,
  `last_error_avg` por casa, para `error_legal`), `results`
  (`pct` por partido), `model` (filas de `shares` del horizonte `params.frozen_horizon` (6) y, si existe, `ban`).
- La elección en curso: `manifest.scopes.es.latest` → `meta.event_date` del run (`params.national`), y si
  `events/es/{event_date}/results` existe (publicada con resultado). Nada más del run.
- `data/andorra/params.json` (fila en `data/README.md`): `national 'es'`, `since '1996-01-01'`, `chart_since
  '2015-01-01'`, `max_days 42`, `frozen_horizon 6`, `min_pairs 3`, `big_sample 5000`, `exclude_mtypes_pairs
  ['aggr']`, `exclude_mtypes_consensus ['aggr', 'online']` (= `DROP_MTYPES`, copiado para no importar el modelo),
  `half 0.5`, `labels` (como `sesgo-cis`: «2019 abr.» cuando el año se repite).

**Medidas** (`stats.py`, `numpy`/`pandas`, `scipy.stats.binomtest` dentro de la función):

- **Error** de un sondeo: su `error_avg` tal como se publica (media del error absoluto en los partidos principales
  de la elección, el del rating). Error del promedio congelado: media de `|model − official|` sobre las filas `main`
  del horizonte 6; del consenso clandestino, lo mismo sobre su cifra por partido; «con la fruta dentro», lo mismo
  sobre las filas `ban`.
- **Último clandestino** de una casa: su fila de `ban.polls` con el mayor `date` (empate: la última fila). **Par**:
  esa fila con la fila de la casa en `ban.legal` cuando `legal.days <= max_days`; `d = error_ban − error_legal`;
  `closer = d < 0`; `big = sample_size >= big_sample`. Casas de `exclude_mtypes_pairs` fuera (por el `mtype` de la
  fila). `pollsters` no interviene en los pares: sólo en `error_legal` de la elección, que es la media de las casas que
  entran en el rating.
- **Prueba del azar** de los pares: `binomtest(k, n − ties, 0.5, alternative='greater').pvalue` con `k` los pares con
  `d < 0` y `ties` los `d == 0`; la página la cuenta como «una moneda lo hace {1 entre x}» o «menos de 1 entre
  10.000» por debajo de 1e-4.
- **Consenso clandestino** de una elección: media simple, por partido principal, del último clandestino de cada casa
  no excluida (`exclude_mtypes_consensus`); nulo con menos de un sondeo.
- **Celdas** (sólo elecciones del ámbito nacional con `model` y `event_date >= chart_since`): por partido principal,
  `frozen`, `consensus`, `official`, `move = consensus − frozen`, `gap = official − frozen`, `closer = |consensus −
  official| < |frozen − official|`, `direction = sign(move) == sign(gap)` con `|move| > 0.05`, `crossed =
  sign(consensus − official) != sign(frozen − official)` («pasó de largo»), `half = closer y (d0 − d1) / d0 >= half`.
- **Por elección**: `n_ban` (sondeos de la parte), `n_pairs`, `k_closer`, `mean_d`; con modelo: `error_legal`
  (media de `last_error_avg` de las casas de `pollsters` con `last_days <= max_days`), `error_frozen`,
  `error_consensus`, `error_with` (nulo sin `ban` en `model`), `beaten = error_consensus < error_frozen`, `beaten_with`.
- **Casas**: por `pollster_id` sobre todos los pares, `n`, `k`, `share`, `mean_d`; entran con `n >= min_pairs`.
- **Grandes**: los pares con `big`; `n`, `k`, `mean_d`.

**Validación** (`study.py`, `compute_ban(reader, data, params=None) -> BanStudy`): sin ningún par, `MissingInput`;
sin ninguna elección nacional con `model`, el estudio se publica con `cells` vacío y `summary.elections` nulo (la
página habla sólo de los pares).

**Pines de test**: un bundle sintético en `tests/fakes.py` (`write_ban_bundle`) con dos ámbitos, tres elecciones con
resultado (dos nacionales con `model`, una con `ban` en `model`; una autonómica sin `model`), tres casas más un
agregador y un panel, un par con el legal fuera de la ventana, un empate, un «pasó de largo», un «grande», y el evento
en curso sin `results` (con un `ban` que **no** debe existir: el test escribe la carpeta sin `results` y comprueba que
el estudio no la lee); cifras calculadas a mano en el test (`d` de un par, `p` de la binomial, consenso, celdas,
errores por elección).

## D5. Las partes de `andorra` y su publicación

Envoltorio como todo estudio (`scope`, `run_id` y `mode` nulos; `schema` `andorra-{parte}@1`). Nombres de campo en
inglés; errores y diferencias a 2 decimales, porcentajes de voto a 1, p-valores con tres cifras significativas (como
`sesgo-cis`).

| Parte | Contenido (claves obligatorias en el esquema) |
|---|---|
| `meta` | `study`, `title`, `subtitle` (con «{k_beaten} de {n_model}» y «{k_closer} de las {n_cells}» vivos; sin modelo, la frase de los pares), `params`, `inputs` (por ámbito: elecciones leídas con las partes encontradas, saltadas y por qué), `next` (`{event_date, open}`: la elección del run nacional y si está publicada con resultado; nulo sin manifiesto), `sources` (las tres noticias con fecha, `/elecciones`, `/metodo`, `/analisis`), `commit`, `dirty`, `versions`, `seconds`. Sin claves de registro. |
| `summary` | `pairs` (`{n, k, ties, share, mean_d, p, generals: {n, k, mean_d, n_events}, regional: {n, k, mean_d, n_events}, by_scope: [{scope, n, k, mean_d}]}`), `elections` (`{n, k_beaten, mean_legal, mean_frozen, mean_consensus, mean_with, k_with}`, los dos últimos nulos sin `ban`; nulo sin modelo), `cells` (`{n, k_closer, k_direction, k_crossed, k_half, missed: [{label, party}]}`), `big` (`{threshold, n, k, mean_d}`), `exception` (`{scope, n, k}` del ámbito con la menor `share` entre los de ≥ 10 pares, para la frase de Cataluña; nulo sin ninguno). |
| `pairs` | `rows`: `[{scope, event_date, label, national, pollster_id, pollster, slug, sponsor, sample_size, mtype, date_legal, days_legal, error_legal, date_ban, days_ban, error_ban, d, closer, big}]`, por `event_date` descendente y `d` ascendente. CSV `pairs`. |
| `elections` | `rows`: una por elección con parte `ban` y al menos un sondeo o un par: `{scope, event_date, label, national, n_ban, n_pairs, k_closer, mean_d, model: {error_legal, error_frozen, error_consensus, error_with, beaten, beaten_with} o nulo}`; `period` (`{since, k, n, mean_d}` de las generales anteriores a `chart_since`, la fila «1996 – 2011» de la tabla). CSV `elections` (plano: `model_error_frozen`…). |
| `cells` | `rows`: `[{event_date, label, party, frozen, consensus, official, move, gap, closer, direction, crossed, half}]` por elección cronológica y partido en el orden de `main`. CSV `cells`. |
| `houses` | `rows`: `[{pollster_id, pollster, slug, n, k, share, mean_d}]` con `n >= min_pairs`, por `share` descendente y `n` descendente; `big_rows`: los pares con `big`, por `sample_size` descendente (`{label, pollster, sponsor, sample_size, error_legal, error_ban, d}`). CSV `houses`. |
| `latest` | La última elección nacional con resultado y parte `ban` (hoy 2023; tras el `publish event` de 2026, 2026): `{scope, event_date, label, parties, polls: [{pollster, slug, sponsor, date (fin de campo), days, sample_size, mtype, error_avg, pct: {party: x}}], consensus: {party: x}, frozen: {party: x} o nulo, official: {party: x}, error: {legal, frozen, consensus, with}}`. CSV `latest-polls`. |

`meta.json` se escribe la última; el orden de escritura es el de `STUDY_PARTS['andorra'][1:]`.

- `publish.publish_study('andorra', writer)`: como el sesgo del CIS (`register` rechazado; manifiesto
  `{published_at, registered_at: null, checked: false}`). Comando: `python job.py publish '{"what":["study"],"study":"andorra"}'`.
- Republicación: tras cada `publish event` (y, cuando exista la acción `studies` del rebaño, a diario con `live`).
- `deploy/README.md` («### Estudios», `what`; «### Backtest», `ban`), `docs/web/contrato.md` (parte `ban` de evento,
  clave `ban` de `model`, estudio, rutas, árbol) ganan lo nuevo.

## D6. La guarda legal

- Ninguna ruta nueva sirve sondeos de una elección sin resultado: la parte `ban` sólo nace en `publish_event`, que
  exige resultado; `/api/v1/events/{scope}/{date}/ban` y su CSV siguen la regla de las demás partes de evento.
- El estudio no lee los `polls` del run ni la base; `latest` y la tarjeta «El 29-N» salen de `events/`.
- Test de guarda (D4): con el evento en curso sin `results`, el estudio no contiene ni su fecha ni sus sondeos, y
  `meta.next.open` es falso.
- `WEB_DROP_CTYPES` y `Computer.drop_ctypes` no cambian.

## D7. La página `/analisis/andorra`

**Rutas y registro.** `pages.analisis.STUDIES['andorra'] = {'template': 'analisis_andorra.html'}`; `CONTEXTS['andorra']
= ban_context`; módulo de vista `mtpy/lib/pages/ban.py` (en `test_pages_package_layout`). El índice `/analisis` y la
banda de Inicio lo listan en el orden de `bundle.STUDIES`.

**Contexto** (`ban_context(state)`): lee las siete partes con `study_part`; cualquier parte o clave ausente da `None`
y la plantilla lo guarda. Entrega `study_meta`, `summary`, `verdicts`, `kpis`, `tests` (las tres cajas de P2),
`elections` (filas de la tabla «Por elección» de P1: las generales por fecha más la fila `period`, luego las tres de
suma: generales, autonómicas, todas), `exception`, `cells` (agrupadas por elección para la tabla de P2),
`model_rows` (la tabla por elección de P2 con las columnas Sondeos · Último legal · Congelado · Clandestino · Con la
fruta y la fila «Media»), `houses`, `big_rows`, `latest`, `next` (`{event_date, open, veda_from}` = `event_date − 6`
días), `sources`, `csv_links`, `published_at` y la isla `initial` con sólo lo que dibuja el gráfico: `pairs` (las
filas con `national` y `event_date >= chart_since`: `label` («GESOP · 2015», «GAD3 · 19 abr.»), `error_legal`,
`error_ban`).

**Verdictos y textos condicionados** (cada frase fuerte con su condición; el constructor fija las demás):

- P1: «SÍ» (verde) si `pairs.share >= 0.6` y `pairs.p < 0.05`; «NO» (rojo) si `share < 0.5`; «A MEDIAS» (ámbar) en
  medio. `.sum`: «en {share} de cada 10 pares…», «una moneda lo hace {1 entre x}» sólo con `p < 0.05`; la frase de
  la excepción con `exception` (nombre del ámbito con `scope_name`).
- P2: «EN {k_beaten} DE {n}» (ámbar; verde «SÍ» si k = n, rojo «NO» si k = 0); sin modelo, la tarjeta dice que no hay
  generales con modelo publicado y no hay veredicto. Caja 1 nombra las elecciones batidas y la que no; caja 2
  «{mean_consensus} desde {mean_frozen}» y la comparación con `mean_legal` sólo si `mean_frozen < mean_legal`; caja 3
  «{k_closer} de {n}» y «en {k_half} de ellas recorrió más de la mitad del camino». La frase «Lo que no vio nadie,
  tampoco lo vio la fruta» de la entradilla sólo si alguna celda de la última elección con modelo tiene `|gap| >= 3` y
  `|move| < 1`; si no, una frase neutra. El `.sum` gana la frase del contrafactual sólo con `mean_with` no nulo.
- P3: «LOS GRANDES» (azul) si `big.n >= 5` y `big.k / big.n >= 0.75`; si no, «{k} DE {n} GRANDES»; sin ningún grande,
  la tarjeta omite esa tabla y el veredicto es el de la mejor casa («{casa}»). El `.sum` nombra las dos casas con
  mayor `share` entre las de `n >= 10` y la de menor, condicionado a que existan.
- KPIs: «Generales en que la fruta afinó el promedio congelado» `{k_beaten} de {n}` (o, sin modelo, la KPI pasa a
  «Elecciones con sondeos clandestinos» `{n}`); «Cifras de partido que quedaron más cerca del resultado» `{k_closer} de
  {n}` con «{k_direction} apuntaron en la dirección correcta y {k_crossed} pasaron de largo» y las `missed` nombradas;
  «Sondeos clandestinos más cerca que el último legal de su casa» `{k} de {n}` con las dos sumas y `mean_d`.
- Pastillas: «Análisis»; «Evidencia histórica · {n_model} generales con modelo · {n} sondeos clandestinos desde
  {since}, con las autonómicas»; «Se completa con {año} tras el cierre de urnas del {fecha}» mientras `next.open` es
  falso y «Completada con {año}» cuando es verdadero.

**Estructura** (ids de la maqueta v2; `m.card_start`/`qcard`; botón de descarga por tarjeta con `card_image_button`):

- `#study-head`: pastillas, `h1` (título de `meta`), entradilla viva y el bloque `.intro` con los cuatro párrafos
  («La ley», «Por qué existen», «Cómo funcionan», «Qué hacemos con ellos») y la línea de fuentes. Los párrafos son texto
  fijo de la plantilla (hechos, no cifras del estudio), salvo las casas y los medios que cita, que son los de la
  maqueta.
- `nav.toc`: Resumen · 1 · ¿Mejor que el último legal? · 2 · ¿Habrían afinado el promedio? · 3 · ¿Quién afina en el
  mercado? · El 29-N (o «{año}») · Datos y método.
- `m.kpis`, `#resumen` (tres tarjetas-enlace y la nota azul).
- `#p1`: entradilla; `grid2` con el dumbbell (`#c-dumb`) y la tabla «Por elección» (Elección · Pares · Más cerca ·
  Mejora media con flechas `.dlt`; la fila «1996 – 2011 ({n} generales)», las generales desde `chart_since`, y las
  sumas Generales / Autonómicas ({n} elecciones) / Todas); la nota de la excepción; `.sum`; «Cómo leerlo».
- `#p2`: entradilla; `.tests` (tres cajas); tabla de celdas a todo el ancho (Elección · Partido · Promedio congelado ·
  Consenso clandestino · Resultado · Movimiento, con las clases `okl`/`kol`/`ok`/`ko` y el texto del movimiento:
  «hacia el resultado · +0,3 de +1,0», «se alejó · −0,4», «pasó de largo · −2,4 cuando faltaban −0,8», «pasó de largo,
  pero más cerca · …», «clavado · +3,1 de +3,1» cuando `|consensus − official| < 0,05`); debajo `grid2` con la tabla
  por elección y los textos (`.sum` y `.tech` «Lo que se movió y lo que faltaba» con las tres celdas de mayor `|move|`
  y la de mayor `|gap|` sin recorrer); «Cómo leerlo».
- `#p3`: entradilla; `grid2` con la tabla de casas (Encuestadora con enlace al perfil · Vedas · Más cerca · Mejora
  media) y la de los sondeos de la jornada (Elección · Casa · medio · Entrevistas · Último legal · Clandestino ·
  Mejora); la nota de ElectoPanel sólo si aparece en la tabla; `.sum`; «Cómo leerlo».
- `#veintinueve`: «El 29-N: el mercado de la fruta de {año}» (título con el día y mes de `next.event_date`); cerrada:
  la entradilla legal y la caja `.locked` («Se abre tras el cierre de urnas del {fecha}»); abierta (`next.open` y
  `latest.event_date == next.event_date`): tabla sondeo a sondeo (Casa · Medio · Fin de campo · Entrevistas · partidos
  · Error),
  las filas Consenso clandestino · Promedio congelado · Resultado, y la frase de errores («el consenso clandestino erró
  {x} puntos por partido; el promedio congelado, {y}; con la fruta dentro, {z}»).
- `#datos`: datos, método (`.tech`: error, par, ventana de 42 días, consenso, promedio congelado, celdas, binomial,
  contrafactual), las cinco pastillas CSV y las fuentes.

**Textos.** Voz «nosotros», «sólo» con tilde, «encuestadora»/«casa», nunca «empresa», «actualización» para los runs,
comillas «» (también en «clandestinos»); cifras con `num`/`int`/`delta`. Los textos de la maqueta v2 son la referencia.
`assert_public_wording` se aplica a `/analisis/andorra`.

## D8. El gráfico y el contenedor fluido

- **Dumbbell** (`charts/dumbbell.js`, nuevo; `renderDumbbell(el, rows, opts)` y `dumbbellOption` exportado para
  `tests/test_js_charts.py`): eje y de categorías (`label`), inverso, de la elección más antigua a la más reciente y,
  dentro, por `d` ascendente; eje x de 0 al máximo redondeado a 0,5 (mínimo 3,5), de 0,5 en 0,5, título «error medio
  por partido (puntos)» con sitio (margen inferior ≥ 48); serie `custom`: línea de `error_legal` a `error_ban`, verde
  (`THEME` de éxito) si `error_ban < error_legal`, roja si no, gris si iguales; punto gris en el legal (r 5) y de color
  en el clandestino (r 6, borde blanco); tooltip «{label} · último legal {a} → clandestino {b} (afinó/empeoró {|d|})».
  Alto 22 px por fila (mínimo 320); montaje con `mountChart`; `grid2` colapsa en móvil y el gráfico ocupa el ancho.
- JS de página: `web/dist/js/pages/analisis-andorra.js` con el patrón de `analisis-sesgo-cis.js` (`readInitial`,
  `tryDraw`, `mountWhenReady`, `wireCardImages`).

## D9. Documentación, Método y memoria

- Contrato (parte `ban`, clave `ban` de `model`, estudio y rutas), README de despliegue (`ban` del backtest, estudio),
  `data/README.md` (parámetros).
- La viñeta «Veda» de `/metodo` (`metodo.html`) gana una frase final con enlace, sólo cuando el estudio está
  publicado: «Si aciertan más que los legales, elección a elección, está en /analisis/andorra».
- Memoria del proyecto: `project-analisis-andorra` al día y el punto 3 de `ideas-analisis-redes`.

## Lo que queda al constructor

- Los nombres internos (`BanStudy`, `ban_context`, `export_ban_*`, `export_event_ban`), el orden exacto de las
  columnas de los CSV y los redondeos no fijados.
- Cómo etiquetar las elecciones (`label`): reutilizar `_label`/`_short` de `studies/lean/data.py` moviéndolos a un
  módulo común de `studies/` o copiarlos; que lo fije.
- El desempate del «último clandestino» cuando una casa publica dos el mismo día (la última fila de `ban`, que va por
  `date` y `pollster_id`).
- Las frases neutras de respaldo de cada texto condicionado (D7), con el mismo tono.
- El tamaño mínimo de las tablas en móvil, los `scroll-hint` y el alto del dumbbell cuando haya más de 30 filas.
- Cómo sabe `run_backtest` si una elección tiene clandestinos en su `drange` para saltar el caso `ban` (una consulta
  aparte o `ctype` en las columnas que carga el `Forecaster`): que lo fije.

## Revisión del constructor (2026-11-05)

Un revisor independiente leyó la spec como constructor. Tres preguntas:

1. **El «último legal» de ElectoPanel.** `pollsters`, `polls` y `cycle` nacen de un `Computer` con `DROP_MTYPES`, así
   que un panel abierto no tiene fila y sus 21 pares no podían formarse. Resuelto como decisión mía: la parte `ban`
   lleva también `legal` (el último legal de cada casa con clandestino, sin filtro de `mtype`) y los pares salen de
   `ban.polls` × `ban.legal` (D2, D4). `pollsters` queda para `error_legal`.
2. **Clandestinos con `days ≥ 6`.** `date` es el fin de campo y hay 17 `wban` con `days = 6`; el reviewer temía que el
   default `['wban', 'exit']` moviera el backtest publicado. Comprobado en la base: ninguno pesa hoy (peso 0 por solape
   o `online`), así que el default se mantiene y la repetición del backtest lo verifica (D3). De paso, `pub_date` no
   se exporta como fecha de publicación y la tarjeta dice «Fin de campo» (D2, D7).
3. **El «promedio congelado» de 2026.** `DEFAULT_EVENTS` del backtest acaba en 2023, así que la fila de 2026 nacería
   sin `model` (ni P2 ni «con la fruta dentro») salvo que el backtest gane la elección; y cabe comparar con la
   predicción que de verdad se publicó (el run congelado) en vez de con la reconstrucción del backtest a horizonte 6.
   **Decidido por Luis (A, 2026-11-05)**: la reconstrucción del backtest a horizonte 6, como el resto de filas y como
   «Lo que dijo el modelo» de la página de la elección; el estudio no lee ningún run. Pasos tras el 29-N (guía de
   Luis, también en `deploy/README.md`): `DEFAULT_EVENTS` gana `2026-11-29` (como en cada elección), `backtest
   {"scopes":["es"],"ban":true}`, `backtest push`, `publish event` de `es/2026-11-29`, `publish study andorra`. Hasta
   entonces la fila de 2026 no existe y la tarjeta «El 29-N» sigue cerrada (D7).
