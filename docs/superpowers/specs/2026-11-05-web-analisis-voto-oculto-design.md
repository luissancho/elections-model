# Web, Análisis: «¿Voto oculto de quién?» (`/analisis/voto-oculto`)

Fecha: 2026-11-05. Estado: **revisada por el constructor y cerrada con las respuestas de Luis del 5 de noviembre (92
elecciones, linajes por suma); pendiente de su aprobación** (brainstorm cerrado el 5 de noviembre; maqueta v3 aprobada y
regenerada como v4 con esas cifras, misma estructura: `docs/superpowers/specs/2026-11-05-web-analisis-voto-oculto-mockup.html`). Cuarto
estudio de la sección «Análisis», tras el voto CERA, el sesgo del CIS (`2026-10-24-web-analisis-sesgo-cis-design.md`) y
el rebaño (`2026-11-05-web-analisis-rebano-design.md`), cuyo patrón de construcción (opción B: estudio publicado
calculado desde las partes del bundle) se repite. Esta spec describe lo que cambia respecto de `dev` **con la rama
`analisis-rebano` ya fusionada** (hoy `dev` = eaba7c9, `analisis-rebano` = 5362225 más cambios sin confirmar): la
clave `live` de `STUDY_RUNNERS`, la acción `studies` del job `publish` y su puesto final en `update`, y los patrones
`write_herd_bundle`, `herd_context` y `weeks.js` son el punto de partida, no parte de esta fase.

## Intención

Luis quiere una entrada en Análisis que conteste con datos a dos creencias muy extendidas: la de muchos politólogos,
«la gente de izquierdas contesta menos a las encuestas y por eso las casas infravaloran a la izquierda», y la de los
simpatizantes de Vox, «sus votantes se avergüenzan de decirlo y por eso las encuestas infravaloran a Vox». La página
debe enseñar la realidad, no una opinión, y servir de hilo en redes (punto 7 de la lista `ideas-analisis-redes`,
adelantado el 5 de noviembre).

Lo que dicen los datos el 5 de noviembre (reconocimiento sobre las partes publicadas; scripts `recon/all94.py`,
`recon/incumbent92.py` y `mockup/gen_v4.py` en el scratchpad de la sesión 0a867961): en las 92 elecciones publicadas
con sondeos finales y resultado (16 generales de 1977 a 2023 y 76 autonómicas de 17 comunidades, de 2009 a 2026), los
sondeos de las dos últimas semanas dieron a la izquierda menos voto del que sacó en 40 y más en 52 (sesgo medio +0,5);
frente a la derecha, la diferencia se equivoca de signo 43 veces en un sentido y 49 en el otro; a Vox, menos en 22 de 39
y −0,3 de media (p 0,28); ningún partido grande sale subestimado más veces de lo que sale sobreestimado (PSOE 49 de 92,
PP 46 de 91), ni el partido del Gobierno (11 de 16 en las generales, 34 de 76 en las autonómicas). Lo que se repite es
el **tamaño** del error: 2,0 puntos de media por partido grande, en cualquier sentido (299 pares elección-partido), y
más de 2 puntos en el 38 % de ellos. La única excepción es Podemos y sus marcas en las 32 autonómicas desde diciembre
de 2018 (de más en 30, +2,0), sin réplica en las generales (de menos en 2 de 3, −0,3) ni antes de 2018 (20 de 42 de
más, −0,1). Todas esas cifras son vivas: la página las calcula en cada publicación y añade 2026 cuando haya escrutinio.

Titular y entradilla (maqueta v3, con las cifras vivas): **«¿Voto oculto de quién?»** / «Se dice que la gente de
izquierdas contesta menos a las encuestas y que los votantes de Vox esconden su voto, y que por eso los sondeos
infravaloran a unos y a otros. En las 92 elecciones con sondeos de nuestra base, los sondeos finales han dado a la
izquierda menos voto del que sacó en 40 y más en 52; a Vox, menos en 22 de 39. Los sondeos se equivocan con cada partido
grande en 2,0 puntos de media, y el sentido del error no se puede adivinar.»

## Decisiones de Luis (brainstorm del 2026-11-05)

1. **Página en /analisis**, no hilo: un estudio publicado con cifras vivas, como el sesgo del CIS y el rebaño.
2. Tres preguntas con veredicto: ¿Subestiman las encuestas a la izquierda? **NO** · ¿Hay voto oculto de Vox? **NO** ·
   ¿De quién es el voto oculto, entonces? **DE NADIE**.
3. «Sistemático a Podemos y Sumar» **no se sostiene** (51 de 77 mezcla un periodo sin nada, 20 de 42 antes de 2018, con
   30 de 32 autonómicas desde finales de 2018): se cuenta como única excepción, en una nota, con la hipótesis del partido en caída al que los sondeos siguen
   con retraso. El PSOE en las generales (10 de 16) y el partido del Gobierno (11 de 16) tampoco son hallazgos: se
   declaran como creencias que no pasan la prueba, con las filas del Gobierno en la tabla de la pregunta 3.
4. La cifra que manda es el **tamaño del error** (2,0 puntos por partido grande con las 92; 1,9 con las 66), en la
   entradilla y en la tercera cifra.
5. Pregunta 1 con **barras cronológicas** de las 92 elecciones a todo el ancho (no la nube de puntos); la tabla por grupos
   debajo.
6. Pregunta 2 con la **cita de Tezanos** (marzo de 2022) marcada «Refutada» y una frase que la contesta.
7. El **voto oculto del PP** de los años 80 y 90 va en la pregunta 3 como nota azul: nació con razón (1982-1993, cuatro
   veces seguidas por encima de sus sondeos) y desde 1996 el signo va en contra.
8. Maqueta v3 aprobada tal cual («Aprobada esta versión»).
9. **Las 92 elecciones** (respuesta del 5 de noviembre a la revisión del constructor): todas las publicadas con sondeos
   finales y resultado, de los 18 ámbitos del catálogo, tengan o no run; no sólo las 66 de los ámbitos del manifiesto.
10. **Linajes** (ídem): PSC y PSE-EE cuentan como PSOE; IU, Podemos y Sumar son un solo espacio y, cuando concurren dos
    marcas (2015 Podemos e IU, Galicia y País Vasco 2024 Sumar y Podemos, 2026 Frente Amplio y Podemos), se **suman** las
    presentes, no se toma la primera.

## Decisiones mías (aceptadas por silencio al aprobar la maqueta; Luis las corrige al leer esta spec)

- **Sin predicción registrada**: la tarjeta «Después del 29-N» enseña lo que dicen hoy los sondeos de las dos últimas
  semanas y añade la fila de 2026 con el escrutinio; nada de `register`.
- **Ventana de 14 días** para los sondeos finales (con 7 ó 30 días, o con el último sondeo de cada casa, el cuadro no
  cambia). Entran todas las elecciones publicadas, también las que tienen 1 ó 2 sondeos finales (29, todas
  autonómicas); la cuenta de sondeos se publica por elección.
- La historia larga (1977-2023) sale de las generales; sin las autonómicas (36 de las 39 elecciones con Vox) no habría
  respuesta sobre Vox.
- **Media simple** de los sondeos, no el promedio del modelo (sólo existe desde 2015 y la pregunta va sobre las
  encuestas). Error = media − resultado; negativo = subestimado.
- **Linajes** por lista fija en los parámetros (D4), con AP y CD como PP; la regla de suma (decisión 10) se aplica a los
  cuatro linajes por igual: un linaje se calcula como un bloque.
- **Partido del Gobierno** por elección en una tabla curada (D4, apéndice), no deducido.
- URL `/analisis/voto-oculto`; paquete `mtpy/lib/studies/shy/` («shy voter» = voto oculto; nombres en inglés).

## Hechos del contexto

- La cita: José Félix Tezanos, marzo de 2022, al admitir que el CIS había estimado a Vox a la baja y anunciar un cambio
  de método: «probablemente hay personas que no quieren reconocer su opción» (The Objective, 2022-03-16; también Qué!).
  Va en `sources` con su URL.
- Referencias de la creencia para «Datos y método»: el «voto oculto del PP» (El Independiente, 2019-02-28), el artículo
  de REIS sobre ocultación del voto a PP y Vox en el recuerdo de voto (reis.cis.es/index.php/reis/article/view/2549), y
  Pavía y Larraz sobre la no respuesta como primera fuente de error de las predicciones en España (REIS 798). Sólo
  enlaces; la página no resume esos trabajos.
- La elección es el 29 de noviembre de 2026; la fecha sale del run (`meta.event_date`), no se escribe a mano.

## Objetivos

1. Un **estudio publicado** `voto-oculto` (`site/v1/studies/voto-oculto/`) con sus partes, gemelos CSV, esquemas, entrada en
   el manifiesto y rutas de la API, calculado desde las partes publicadas del bundle (D2, D3, D4).
2. Su **republicación con cada `update`** (`live`), para que la fila de 2026 y las cuentas aparezcan solas tras el
   `publish event` del 29-N (D5).
3. La página `/analisis/voto-oculto` con la estructura de la maqueta v3, textos con cifras vivas y condicionados, dos
   gráficos nuevos, descarga de cada tarjeta como imagen, contenedor fluido; entra sola en `/analisis` y en la banda de
   Inicio (D6, D7).
4. Comprobación pública de redacción, documentación, memoria (D8).

## No objetivos

- Ningún acceso a la base de datos: el estudio no instancia `Computer` ni `Forecaster`. Si una cifra no está en las
  partes publicadas (o en `data/`), no entra.
- Ningún cambio en el modelo, en el promedio, en los errores que publica `poll-errors`/`error-data`, en /casas ni en las
  páginas de elección.
- Sin registro de predicción, sin conmutadores de ventana, sin la nube de puntos en la pregunta 1, sin análisis por
  casa (ese es el del sesgo del CIS) ni por patrocinador.
- Nada de nginx, Docker, CSP ni modo oscuro.

## Anti-objetivos

- Que una cifra del texto esté escrita a mano en la plantilla: todo número sale de las partes; los textos cuyo sentido
  depende del signo o del recuento se condicionan (D6). La lista de elecciones, los linajes y el partido del Gobierno
  están en `data/`, no en el código.
- Que la página rompa cuando falte algo: sin run nacional, `current` es nulo y la tarjeta «Después del 29-N» habla sólo
  del pasado; una elección sin `cycle` o sin `results` se omite y se anota en `meta.inputs`; un linaje sin partido en
  una elección deja la celda nula; sin elecciones la publicación se rechaza (D4).
- Que la página afirme lo que la memoria del proyecto ya descartó: «sistemático a Podemos/Sumar», «el PSOE sale
  subestimado», «el Gobierno sale subestimado». Los textos fuertes van condicionados y los de la nota de Podemos, del
  PSOE y del Gobierno se escriben como creencias comprobadas (D6).
- Perder la lectura sin JavaScript: textos, veredictos, cifras y tablas salen del servidor; sólo los tres gráficos los
  dibuja el navegador.
- Que la republicación diaria toque los runs, los análisis, los eventos ni los otros estudios.

## Enfoques considerados

- **A. Estudio desde la base de datos**: descartado, como en los dos anteriores: los errores por sondeo ya publicados
  (`poll-errors`) no sirven porque no llevan el error por partido con signo (`error-data` lo publica en valor absoluto),
  y recalcular desde la base duplica lo que `cycle` + `results` ya tienen.
- **B. Estudio desde las partes publicadas** (elegido): lee `meta`, `cycle` y `results` de cada evento publicado de todos
  los ámbitos, y `meta` + `polls` del último run nacional; tarda segundos; se encadena tras cada publicación.
- **C. Página calculada al servir**: descartado por lo mismo que las veces anteriores (sin CSV ni API, lecturas de S3 en
  frío por trabajador).

## D1. Declaraciones del estudio (`mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`)

```python
STUDIES = ('voto-cera', 'sesgo-cis', 'rebano', 'voto-oculto')
STUDY_PARTS['voto-oculto'] = ('meta', 'summary', 'elections', 'groups')
STUDY_SCHEMAS['voto-oculto'] = {part: 'voto-oculto-' + part for part in STUDY_PARTS['voto-oculto']}
STUDY_CSV['voto-oculto'] = ('elections', 'groups')
STUDY_CSV_OF['voto-oculto'] = {'elections': ('elections',), 'groups': ('groups',)}
```

- `SCHEMAS` gana las cuatro entradas `voto-oculto-*` con las claves obligatorias de D3 (nulos como `(dict, type(None))`,
  como `rebano`). La línea del docstring de `bundle.py` pasa a «(voto-cera, sesgo-cis, rebano, voto-oculto)».
- `publish/studies.py`: `STUDY_TITLES['voto-oculto']` (título y plantilla de la entradilla con `{n}`, `{k_under}`,
  `{k_over}`, `{vox_under}`, `{vox_n}`, `{size}`), `STUDY_EXPORTERS['voto-oculto']`, `STUDY_META['voto-oculto'] =
  export_shy_meta`, `STUDY_RUNNERS['voto-oculto'] = {'compute': compute_shy_for_publish, 'registrable': False,
  'live': True}`. Tercer import tardío (`from .shy import (compute_for_publish as compute_shy_for_publish, export_shy_*)
  # noqa: E402`), con alias como el del rebaño.
- Los exportadores viven en `mtpy/lib/publish/shy.py` (patrón de `herd.py`: importa el paquete del estudio dentro de
  `compute_for_publish`; `test_importing_publish_does_not_load_the_study_package` gana `mtpy.lib.studies.shy`;
  `publish/__init__.py` reexporta los nombres nuevos; `test_publish_package_layout` gana `shy`).
- Consumidores ya preparados por estudio (`Studies.part_action`, `webapi.checks`, `pages.analisis._csv_links` con
  `CSV_LABELS['voto-oculto']`, `tests/test_webapi_unit.write_study` con fixtures `tests/fixtures/bundle/voto-oculto-*.json`,
  `test_study_registry_is_per_study`, `test_study_runners_declare_which_studies_register`): sólo ganan el slug.

## D2. Las entradas

Todo con un `BundleReader(fs, prefix=bundle.PREFIX)` sobre el `fs` del job, también en `dry_run` (como los dos
anteriores).

**Elecciones pasadas.** Para cada ámbito del catálogo (`data/es-scopes.csv`, los códigos `es` y `es-*` en su orden,
`es` primero; 18 hoy, tenga o no run el ámbito: `manifest.scopes` no interviene) y cada fecha de
`reader.list_events(scope)` (un ámbito sin carpeta de eventos se salta y se anota) con las tres partes `meta`, `cycle` y
`results` publicadas:

- `meta`: `name`, `blocks` (`{Izquierda: [...], Derecha: [...]}`; un ámbito sin bloques deja los bloques nulos y se anota).
- `cycle`: `parties` y `polls` (`date`, `pollster_id`, `pollster`, `start_date`, `end_date`, `sample_size`, `mtype`,
  `weight` y una columna por partido).
- `results`: `rows` (`party`, `pct`).

Una elección sin alguna de las tres partes, o **sin ningún sondeo final** en la ventana (hoy, Cantabria 2011 y 2019),
se omite y se anota en `meta.inputs[scope].skipped` con su motivo (`parts` o `no final polls`); no cuenta en `n_elections`. Las elecciones
**publicadas sin alimentar el modelo** (las de menos de 10 sondeos útiles, `featured` falso) entran igual si tienen las
tres partes: la pregunta va sobre las encuestas, no sobre el modelo (hoy, 14 de las 92).

**Ciclo actual.** El run `manifest.scopes.es.latest`: `meta` (`event_date`, `as_of`, `run_id`) y `polls` (misma forma
que `cycle`). Es «actual» mientras `event_date` no tenga carpeta publicada en `events/es/` con parte `results`; cuando la
tenga (tras el 29-N y `publish event`), esa elección entra en las pasadas por sus partes y `current` es nulo. Sin
manifiesto o sin run nacional: `current` nulo y anotación. Los bloques y los partidos principales del ciclo actual
salen del propio `meta` del run (`bmaps.blocks`, con Izquierda/Derecha/Otros, y `bmaps.main`), la misma fuente que
usará el evento de 2026 al publicarse; sin ellos, los bloques de 2026 quedan nulos y la tarjeta enseña sólo partidos.

**Filtros comunes.** Entran los sondeos con `weight > 0` (los que usa el promedio: sin agregadores ni paneles abiertos;
en el reconocimiento no había ninguno con peso nulo en las partes `cycle`) cuyo **fin de campo** (`end_date`, o `date`
sin ella) cae entre 0 y `params.window_days` (14) días antes de la elección (incluidos ambos extremos). Para el ciclo
actual, la ventana se cuenta hacia atrás desde `as_of`.

**Parámetros** (`data/voto-oculto/params.json`, fila en `data/README.md`) y **tabla del Gobierno**
(`data/voto-oculto/incumbents.json`, fila en `data/README.md`, contenido en el apéndice) y el **catálogo de ámbitos**
(`data/es-scopes.csv`, columnas `scode` y `name`): se leen con el `FileSystem` de `data/` que `publish_study` pasa a
`compute`.

## D3. Las partes de `voto-oculto`

Envoltorio como todo estudio (`scope`, `run_id` y `mode` nulos; `schema` `voto-oculto-{parte}@1`). Nombres de campo en
inglés; redondeos: errores y medias a 2 decimales, p-valores con tres cifras significativas (como `sesgo-cis`).

Un **bloque de cifras** (`figures`) es siempre el mismo dict: `n` (elecciones con dato), `under` (error < 0), `over`
(error > 0), `mean`, `median`, `p_t` (t de Student de una muestra sobre la media; nulo con menos de 3 datos), `p_sign`
(binomial de una cola: la probabilidad de que una moneda justa dé **`max(under, over)` o más** aciertos de signo entre
`n`; es la cifra que los textos escriben como «la probabilidad de {k} o más por azar», la misma cola que
`lean.stats.sign_chance`; para Vox, 22 de 39 da 0,26; la maqueta v3 escribía la bilateral y la v4 ya la de una cola), `size` (media de |error|), `size_over` (proporción con
|error| > `params.size_threshold`, 2 puntos).

| Parte | Contenido (claves obligatorias en el esquema; el constructor fija el resto) |
|---|---|
| `meta` | `study`, `title`, `subtitle` (plantilla de `STUDY_TITLES` con las cifras vivas), `scopes` (códigos leídos), `election` (`{date, days_left, as_of, run_id}` del ciclo actual o nulo), `params` (D4), `inputs` (por ámbito: `{events: n, skipped: [{date, reason}]}`; `run` con `run_id`/`as_of` o `skipped`; `incumbents` con las elecciones sin fila en la tabla), `sources` (las referencias de «Hechos del contexto», con URL), `commit`, `dirty`, `versions`, `seconds`. Sin claves de registro. |
| `summary` | `n_elections`, `n_general`, `n_regional`, `n_scopes`, `first_year`, `first_year_regional`, `last_year`; `left` (`{all, general, regional, recent}` → `figures` del bloque Izquierda; `recent` = elecciones desde `params.recent_from`, 2018-09-01); `right` (lo mismo para el bloque Derecha); `lean` (lo mismo para Izquierda − Derecha); `vox` (`figures` de Vox, más `extremes` = `{under: [{label, error}] × 3, over: [{label, error}] × 3}`, las tres mayores en cada sentido); `parties` (lista en el orden de la tabla de P3: `[{key, label, figures}]` para `PSOE`, `PP`, `VOX`, `UP`, `Izquierda`, `Derecha`, `incumbent_general`, `incumbent_regional`); `size` (`{n_pairs, mean, share_over}`: pares elección-partido de los cuatro linajes `params.size_parties`); `psoe_general` (`figures` del PSOE en las generales + `big`: etiquetas con error < `params.big_threshold`, −3); `pp` (`{early: {from, to, errors: [{label, error}]}, late: figures, all: figures}` con `params.pp_early`); `podemos` (`{before: figures, general_recent: figures, regional_recent: figures}` con `recent_from`); `current` (`{as_of, days_left, n_polls, from, to, values: {Izquierda, Derecha, PSOE, PP, VOX, …}, names: [...]}`: la media de los sondeos de la ventana actual para los partidos `main` del run y, para cada bloque de `params.blocks`, la suma de las medias de sus miembros con estimación, o nulo); `verdicts` (`{p1, p2, p3}`: `'NO'`/`'SÍ'`/`'DE NADIE'`/`'DE {label}'` según D6, para que la API y la página coincidan). |
| `elections` | Una fila por elección pasada: `scope`, `scope_name`, `event_date`, `label` («Generales abr. 2019», «Madrid 2021»), `short` («GEN 19A», «MD 21»), `general` (bool), `n_cycle`, `n_final`, `blocks` (`{Izquierda: [...], Derecha: [...]}` o nulo), `Izquierda`, `Derecha`, `lean`, `PSOE`, `PP`, `VOX`, `UP` (errores; nulo cuando el linaje no se presentó o ningún sondeo final lo estimó), `names` (`{PSOE: 'PSC', PP: 'AP', UP: 'UP+IU', ...}`: las marcas del linaje presentes en los resultados, unidas con `+`), `incumbent` (nombre o nulo), `incumbent_error`. Orden cronológico, y a igual fecha el orden del catálogo. CSV `elections` (plano: `blocks` como dos columnas con los partidos separados por `+`). |
| `groups` | Las filas de todas las tablas de la página en formato largo: `rows: [{table, key, label, n, under, over, mean, median, p_t, p_sign, size, size_over}]` con `table` ∈ `left`, `lean`, `vox`, `parties`, `podemos`, `pp`. CSV `groups`. |

`meta.json` se escribe la última; el orden de escritura es el de `STUDY_PARTS['voto-oculto'][1:]`.

## D4. El cálculo: `mtpy/lib/studies/shy/`

Paquete nuevo, ligero (`numpy`, `pandas`, `scipy.stats` dentro de funciones):

- `data.py`: `load_params(data)`, `load_incumbents(data)`, `MissingInput`, `load_catalogue(data)` (los `scode`/`name` de `data/es-scopes.csv`), `read_elections(reader, catalogue, params)`
  (lista de elecciones pasadas con sus sondeos finales, bloques y resultados, más `inputs`), `read_current(reader,
  data, params)` (el run, o `None`), `is_current(reader, scope, date)`, y las etiquetas `_label`/`_short` copiadas del
  sesgo del CIS (misma regla: mes abreviado cuando el año se repite en el ámbito).
- `stats.py`, con la medida:
  - **Error de un partido (linaje)** en una elección: un linaje se calcula como un bloque. Sus **marcas presentes** son
    las de su lista (`params.lineages`) que están en `results`; para cada sondeo final, la suma de las marcas presentes
    que el sondeo estima (no nulas) menos la suma de los resultados de esas mismas marcas; la media sobre los sondeos.
    Sin ninguna marca en `results`, o sin ningún sondeo que estime alguna, el linaje queda nulo en esa elección. Con una
    sola marca presente es la media simple de sus cifras menos su resultado (PSOE, PP y Vox siempre; IU/Podemos/Sumar
    en 52 de las 77). `names` lleva las marcas presentes unidas con `+`.
  - **Error de un bloque**: para cada sondeo final, la suma de los partidos del bloque (`meta.blocks`) que el sondeo
    estima (no nulos) menos la suma de los resultados de **esos mismos partidos**; la media sobre los sondeos. Sin bloques
    o sin ningún partido del bloque en el sondeo, nulo. `lean` = `Izquierda − Derecha` (nulo si falta uno).
  - **Partido del Gobierno**: el primer nombre de la lista de `incumbents.json` para `(scope, date)` presente en
    `results`; error como un partido. Sin fila o sin nombre presente, nulo y anotación.
  - `figures(values)`: el bloque de D3 (`scipy.stats.ttest_1samp` y `binomtest` binomial de una cola,
    `alternative='greater'` sobre `max(under, over)`; `p_t` nulo con n < 3).
  - `extremes(rows, k=3)`, `size(rows, parties, threshold)`.
- `study.py`: `compute_shy(reader, data, params=None) -> ShyStudy` (dataclass con los bloques de D3) y la validación:
  sin elecciones, `MissingInput('events')`. Los veredictos (D6) se calculan aquí, no en la página.
- Parámetros en `data/voto-oculto/params.json`: `window_days 14`, `recent_from '2018-09-01'`, `blocks ['Izquierda',
  'Derecha']`, `lineages {'PSOE': ['PSOE', 'PSC', 'PSE-EE'], 'PP': ['PP', 'AP', 'CD'], 'VOX': ['VOX'], 'UP': ['SUMAR', 'FA',
  'UP', 'IU', 'PCE']}` (el orden de la lista sólo manda en `names`; en 2026 se suman `FA` y `UP`), `labels
  {'UP': 'IU / Podemos / Sumar', 'PP': 'PP (AP, CD)'}`, `size_parties ['PSOE', 'PP', 'VOX', 'UP']`, `size_threshold 2`,
  `big_threshold -3`, `pp_early ['1982-01-01', '1996-01-01']`, `extremes 3`.
- `data/voto-oculto/incumbents.json`: `{scope: {date: [names]}}` con las 95 filas del apéndice (las 92 elecciones,
  Cantabria 2011 y 2019 por si ganan sondeos, y 2026; curación propia).
- Pines de test: un bundle sintético en `tests/fakes.py` (`write_shy_bundle`) con tres ámbitos del catálogo (`es` con
  tres elecciones, una de ellas sin `results`; un ámbito autonómico con dos, una de ellas con dos marcas del linaje
  `UP` presentes y un sondeo que estima sólo una; y un ámbito del catálogo sin run ni carpeta en el manifiesto), bloques,
  cuatro linajes con nombres distintos (`AP`, `PSC`), un run actual y un `data/` con parámetros, catálogo y la tabla del
  Gobierno; cifras calculadas a mano en el test (error de un partido, de un linaje con dos marcas, de un bloque con un
  partido ausente en un sondeo, `lean`, `figures` de una lista corta, `size`, los veredictos, `current` con la ventana
  sobre `as_of`, la elección omitida en `inputs`).

## D5. La publicación, el encadenado y el job

- `publish.publish_study('voto-oculto', writer)`: como los anteriores (`register` rechazado; manifiesto
  `{published_at, registered_at: null, checked: false}`).
- Comando suelto: `python job.py publish '{"what":["study"],"study":"voto-oculto"}'`.
- `live` verdadero: la acción `studies` del `publish` (y el último paso del `update`) lo republica cada día con los dos
  estudios vivos anteriores; nada nuevo que cablear. Tras el 29-N, `publish event` de `es/2026-11-29` y la siguiente
  republicación añaden la fila de 2026 y vacían `current`.
- `deploy/README.md` («### Estudios», un párrafo «**Voto oculto.**» y el comando), `docs/web/contrato.md` (estudios:
  «cuatro», sección `### voto-oculto` con partes/esquemas/CSV, fila de «Rutas de estudios») ganan el estudio.

## D6. La página `/analisis/voto-oculto`

**Rutas y registro.** `pages.analisis.STUDIES['voto-oculto'] = {'template': 'analisis_voto_oculto.html'}`;
`CONTEXTS['voto-oculto'] = shy_context` (en `analisis.py`, patrón de `herd_context`); módulo de vista `mtpy/lib/pages/shy.py`
(`SHY = 'voto-oculto'`, `flags`, `verdict_rows`, `kpis`, `tests`, `group_rows`, `party_rows`, `island`; en
`test_pages_package_layout` y en `pages/__init__.py`). El índice `/analisis` y la banda de Inicio lo listan en el orden
de `bundle.STUDIES` (el cuarto).

**Contexto** (`shy_context(state)`): lee las cuatro partes con `study_part`; cualquier parte o clave ausente da `None` y
la plantilla lo guarda. Entrega `study_meta`, `summary`, `verdicts`, `kpis`, `left_rows` (tabla de P1), `vox_tests` (las
tres cajas de P2), `party_rows` (tabla de P3), `current` (tarjeta «Después del 29-N»), `csv_links`, `sources`,
`published_at` y la isla `initial` con sólo lo que dibujan los gráficos: `left_bars` (`[{short, label, value, general}]`
cronológico), `vox_bars` (igual, sólo elecciones con Vox) y `party_strip` (`{rows: [{party, label, value}], means: {party:
mean}, order: [...]}`), más las medias para las rayas.

**Veredictos** (calculados en `summary.verdicts`, D3; la página sólo los pinta):

- P1 «NO» (rojo, `.q3`) salvo que `left.all.mean < 0` y `left.all.p_t < 0.05`: entonces «SÍ» (verde) y los textos de P1
  cambian a la frase de respaldo («la izquierda sale subestimada en {under} de {n}, {mean} de media: la creencia se
  sostiene»).
- P2 «NO» salvo que `vox.mean < 0` y `vox.p_t < 0.05`: «SÍ».
- P3 «DE NADIE» salvo que algún `parties[].figures` de `PSOE`, `PP`, `VOX`, `UP` tenga `mean < 0` y `p_t < 0.05`: «DE
  {label}» (varios, separados por « Y »).

**Textos condicionados** (cada frase fuerte con su condición; el constructor fija las demás y las frases neutras de
respaldo con el mismo tono):

- Entradilla y KPI 1: «menos voto del que sacó en {under} y más en {over}»; la coletilla «cuando se equivocan, más veces
  le dan de más que de menos» sólo con `over > under`; con `under > over`, «más veces de menos que de más».
- KPI 2: «con una moneda saldrían {n/2 redondeado} … la probabilidad de {under} o más por azar es del {p_sign}» (`p_sign`
  de una cola, en %, una cifra decimal por debajo del 10 %; la caja 1 de P2 repite la misma cifra).
- KPI 3: `size.mean` y `size.share_over` («en el {share} % de los casos, más de 2 puntos»).
- P1 `.sum`: «En las generales el signo se inclina un poco hacia la subestimación ({under} de {n}) por {k} elecciones en que
  el PSOE sacó más de 3 puntos por encima de sus sondeos ({lista}); una racha así sale por azar el {p_sign} % de las
  veces» (`under`, `n` y `p_sign` de `left.general`; `k` y la lista de `psoe_general.big`) sólo con `psoe_general.under > psoe_general.over` y `big` no vacío; «en las autonómicas pasa lo contrario ({over}
  de {n} de más), pero ahí los sondeos dan de más a los dos bloques ({left.regional.mean} a la izquierda y
  {right.regional.mean} a la derecha, porque reparten poco voto a los partidos pequeños): la cuenta que vale es
  Izquierda − Derecha, repartida {lean.regional.under} a {lean.regional.over}» sólo con `left.regional.over >
  left.regional.under` y las dos medias regionales positivas (hoy +0,7 y +0,6); en otro caso, frases neutras.
- P2: las tres cajas con `vox` (`under de n`, `mean` y mediana, `extremes`); la frase «lo que se espera de unos sondeos
  sin sesgo» sólo con `p_sign >= 0.05`; la cita de Tezanos con la pastilla «Refutada» sólo con veredicto «NO» (con «SÍ»,
  pastilla «Confirmada» y la frase de respaldo).
- P3 `.sum`: «las medias quedan a menos de un punto del cero en todos los casos salvo {k}» calculado sobre `parties`;
  la frase del PSOE y del Gobierno con sus cifras (`psoe_general`, `incumbent_general`, `incumbent_regional`) sólo cuando
  `incumbent_regional.p_sign >= 0.05` («en las {n_regional} autonómicas la racha desaparece»); si no, frase neutra.
- Nota azul del PP: `pp.early.errors` (lista con signo), «desde {año} el signo va {late.over} a {late.under} en contra»
  sólo con `late.over > late.under`; cierra con `pp.all`.
- Nota de Podemos: «Desde finales de 2018» es texto fijo ligado a `recent_from`; con `podemos.regional_recent.over ==
  podemos.regional_recent.n` «en las {n}»; si no, «en {over} de {n}»; «en las tres generales del mismo periodo pasó lo contrario dos veces» → «en las {n} generales del mismo periodo
  {frase según under/over}»; la hipótesis del partido en caída es texto fijo; los dos ejemplos (PP abril de 2019 y Cs
  noviembre de 2019) se leen de `elections` (PP en `es/2019-04-28`, y Cs no es linaje: cifra fija en el texto, marcada
  como tal en la plantilla con su fuente, o se omite si el constructor prefiere no escribir cifras a mano; decisión suya).
- «Después del 29-N»: con `current`, la fila «Sondeos finales, hoy» con `values` y el periodo («sondeos del {from} al
  {to} ({n_polls})»); la frase «Si la creencia tuviera razón, el 29-N la izquierda sacaría más de {Izquierda} y Vox más
  de {VOX}»; «cada una de esas cifras fallará en unos {size} puntos». Sin `current` y con la elección ya en `elections`,
  la fila de 2026 con resultado y error, y la frase «La fila de 2026 ya está en las cuentas de arriba».

**Estructura** (ids de la maqueta v3; `m.card_start`/`qcard`; botón de descarga por tarjeta con `card_image_button` como
los otros estudios, nombre `demoscopio-voto-oculto-{slug}.png`):

- `#study-head`: pastillas «Análisis», «Evidencia histórica · {n} elecciones con sondeos · {n_general} generales desde
  {first_year} y {n_regional} autonómicas desde {año}», «Página viva · añade la fila de 2026 en cuanto haya escrutinio»
  (tras el 29-N: «Página viva · con la fila de 2026»); `h1` y entradilla vivas.
- `nav.toc`: Resumen · 1 · ¿Subestiman a la izquierda? · 2 · ¿Voto oculto de Vox? · 3 · ¿De quién, entonces? · Después
  del 29-N · Datos y método.
- `m.kpis`: las tres cifras de la maqueta.
- `#resumen`: «Tres preguntas, tres respuestas», las tres tarjetas-enlace y la nota azul «Medimos lo que publicaron las
  encuestadoras, no lo que contestó la gente…».
- `#p1`: entradilla con la medida; gráfico de barras a todo el ancho (`#left-chart`); debajo, la tabla por grupos
  (Elecciones · N · Subestimada · De más · Sesgo medio · p) con las cinco filas (Generales, Autonómicas, Desde sept.
  2018, Todas, Izquierda − Derecha) y la nota de la última fila; `.sum`; «Cómo leerlo».
- `#p2`: la cita con la pastilla; entradilla; `.tests` (tres cajas); gráfico de barras de Vox (`#vox-chart`, alto 260);
  `.sum`; «Cómo leerlo».
- `#p3`: entradilla; `grid2` 1,3 / 1 con la tabla (Quién · N · Subest. · De más · Sesgo · p · Tamaño, ocho filas, la nota
  de los bloques en autonómicas debajo) y la nube de puntos por partido (`#parties-chart`); `.sum`; nota azul del PP;
  nota ámbar de Podemos; «Cómo leerlo».
- `#despues`: «Después del 29-N» con la tabla de tres filas (Sondeos finales, hoy · Resultado del 29-N · Error) y la
  frase condicionada.
- `#datos`: datos (los sondeos de las elecciones publicadas en Histórico, con enlace a `/elecciones`; los resultados y
  el bloque de cada partido, los de la página de la elección; los linajes y la tabla del Gobierno; CSV), método
  (`.tech`: sondeos finales, error de linaje y de bloque con la regla de suma, subestimado, p de la t y la binomial de
  una cola, la ventana y su robustez; «{n_regional} autonómicas de {n_scopes − 1} comunidades»), las dos pastillas CSV y las fuentes (la cita de Tezanos y las tres referencias).

**Textos.** Voz «nosotros», «sólo» con tilde, «encuestadora»/«casa», nunca «empresa», comillas «»; cifras con
`num`/`int`/`delta`. Los textos de la maqueta v3 son la referencia. `assert_public_wording` se aplica a
`/analisis/voto-oculto` (en la tupla compartida de `test_public_wording_…` y en su test de estructura).

## D7. Los gráficos y el contenedor fluido

Módulo nuevo `web/dist/js/charts/errors.js` con dos opciones y sus `render*`, ambas exportadas para
`tests/test_js_charts.py` (alias nuevo en `run_js`):

- **Barras con signo** (`errorBarsOption(rows, {mean, meanLabel, yMax, boldOf, rotate})`): eje x de categorías con
  `short` (rotadas 90° en P1 con sus 92 barras, 55° en P2; en negrita las generales, todas las etiquetas visibles, 9-10 px), eje y simétrico de 2 en 2 (tope
  ajustable al máximo |valor| redondeado hacia arriba, mínimo ±8 en P1 y ±5 en P2), barra roja por debajo de 0 y azul por
  encima, autonómicas con opacidad 0,55, raya naranja discontinua en la media con etiqueta «media {valor}», tooltip con
  `label` y el valor con signo. Alto 340 px en P1 (280 en móvil) y 260 en P2.
- **Nube por partido** (`errorStripOption({rows, means, order}, {xMax})`): eje y de categorías con las seis filas
  (`order`, de abajo arriba: Derecha, Izquierda, IU / Podemos / Sumar, Vox, PP, PSOE), eje x simétrico ±8 de 2 en 2, un
  punto por elección (rojo < 0, azul > 0, opacidad 0,5, borde blanco), línea negra en 0, un rectángulo naranja en la
  media de cada fila; tooltip «{label} · {party}: {valor}». Alto 340 px.
- Colores del tema (`THEME`), tipografía DM Sans, formato `+1,2`/`−1,2` (menos tipográfico), como el resto del sitio.
- JS de página: `web/dist/js/pages/analisis-voto-oculto.js` con el patrón de `analisis-rebano.js` (`readInitial`,
  `tryDraw`, `mountWhenReady`, `wireCardImages`). Montaje con `mountChart`; `grid2` colapsa en móvil; la tabla de P3 con
  `scroll-hint`.

## D8. Documentación, Método y memoria

- Contrato, README de despliegue y `data/README.md` (D5, D4).
- Opcional, decisión del constructor: una frase con enlace al final de la tarjeta de los errores de `/metodo/promedio`
  («De quién es el voto oculto, elección a elección, está en /analisis/voto-oculto»), sólo cuando el estudio está
  publicado; si no cabe sin tocar textos aprobados, no se añade.
- Memoria del proyecto: `project-analisis-voto-oculto` pasa a «construido» y el punto 7 de `ideas-analisis-redes` a hecho.

## Lo que queda al constructor

- Los nombres internos (`ShyStudy`, `shy_context`, `export_shy_*`), el orden exacto de las columnas de los CSV y los
  redondeos no fijados.
- Las frases neutras de respaldo de cada texto condicionado (D6), con el mismo tono; y si la cifra de Ciudadanos de la
  nota de Podemos se escribe a mano con su fuente o se omite.
- La forma exacta de `summary.current.values` cuando el run no trae alguno de los bloques o partidos (claves ausentes,
  la tarjeta las salta).
- El tamaño mínimo de las tablas en móvil y los `scroll-hint`; el tope del eje de barras cuando algún valor supere ±8.

## Calendario y rama

El rebaño se está construyendo en el worktree `analisis-rebano` y toca los mismos puntos compartidos (`STUDIES`, las
colas de los dicts de `bundle.py` y `studies.py`, el bloque de imports tardíos, `CONTEXTS`, `pages/__init__`, `run_js`,
las listas de los tests de layout). **La construcción de este estudio arranca cuando el rebaño esté en `dev`**, en un
worktree propio desde `dev`; hasta entonces sólo spec, maqueta y plan.

## Pruebas (mínimas)

- `tests/test_shy_*.py`: `figures` con valores conocidos (incluido n < 3), error de linaje y de bloque sobre filas
  sintéticas (partido ausente en un sondeo, linaje con dos marcas presentes, linaje sin partido, bloque nulo), el
  catálogo manda sobre el manifiesto (ámbito sin run), `lean`, `extremes`, `size`, los veredictos
  en los tres casos (NO, SÍ, DE {label}), `compute_shy` sobre el bundle sintético (cifras de `summary`, la elección
  omitida en `inputs`, `current` con la ventana sobre `as_of` y su vaciado cuando la elección tiene `results`),
  `MissingInput` sin elecciones.
- `tests/test_publish_shy.py`: partes y CSV validan, `meta` la última, manifiesto, `register` rechazado, los otros tres
  estudios intactos; la acción `studies` publica los tres estudios vivos (`sesgo-cis`, `rebano` y éste; `voto-cera` no).
- `tests/test_bundle_unit.py`, `test_publish_studies.py`, `test_publish_unit.py`: registro, esquemas, layout.
- `tests/test_web_api.py`: índice con los cuatro estudios, partes y CSV del nuevo, errores.
- `tests/test_pages.py`: contexto (cifras desde las partes, KPIs, isla, enlaces CSV), página (pastillas, índice,
  veredictos, tres `kpi`, ids de los tres gráficos con `aria-label`, tablas, pastillas CSV = 2, botones de descarga),
  `assert_public_wording`, índice y banda de Inicio con cuatro estudios, 404 sin publicar; `test_pages_package_layout`
  con `shy`.
- `tests/test_js_charts.py`: `errorBarsOption` (colores por signo, negritas, raya de la media, tope del eje) y
  `errorStripOption` (filas, medias, tooltip).
- Suite completa en verde (la de `dev` con el rebaño fusionado).

## Apéndice: la tabla del Gobierno (`data/voto-oculto/incumbents.json`)

El partido que presidía el Gobierno (de España o de la comunidad) el día de cada elección publicada; con varios
nombres, el primero presente en los resultados es el que cuenta. Curación propia del 5 de noviembre de 2026 (todas las filas casan con un nombre de los resultados publicados); Luis la
revisa.

| Ámbito | Elecciones y partido |
|---|---|
| `es` | 1977, 1979, 1982: UCD · 1986, 1989, 1993, 1996: PSOE · 2000, 2004: PP · 2008, 2011: PSOE · 2015, 2016: PP · 2019 abr., 2019 nov., 2023: PSOE · **2026-11-29: PSOE** (para que la fila de 2026 cuente al publicarse) |
| `es-an` | 2012, 2015, 2018: PSOE · 2022, 2026: PP |
| `es-ar` | 2011: PSOE · 2015: PP · 2019, 2023: PSOE · 2026: PP |
| `es-as` | 2011: PSOE · 2012: FAC · 2015, 2019, 2023: PSOE |
| `es-cb` | 2011: PRC · 2015: PP · 2019, 2023: PRC (2011 y 2019 sin sondeos finales hoy: la fila queda por si los ganan) |
| `es-cl` | 2011, 2015, 2019, 2022, 2026: PP |
| `es-cm` | 2011: PSOE · 2015: PP · 2019, 2023: PSOE |
| `es-cn` | 2011, 2015, 2019: CC · 2023: PSOE |
| `es-ct` | 2010: PSC · 2012: CiU · 2015: JxSí (CiU, CDC) · 2017: JxCat (PDeCAT) · 2021: JxCat · 2024: ERC |
| `es-ex` | 2011: PSOE · 2015: PP · 2019, 2023: PSOE · 2025: PP |
| `es-ga` | 2009: PSOE · 2012, 2016, 2020, 2024: PP |
| `es-ib` | 2011: PSOE · 2015: PP · 2019, 2023: PSOE |
| `es-mc` | 2011, 2015, 2019, 2023: PP |
| `es-md` | 2011, 2015, 2019, 2021, 2023: PP |
| `es-nc` | 2011, 2015: UPN · 2019: GBAI · 2023: PSOE (PSN) |
| `es-pv` | 2009: PNV · 2012: PSE-EE · 2016, 2020, 2024: PNV |
| `es-ri` | 2011, 2015, 2019: PP · 2023: PSOE |
| `es-vc` | 2011, 2015: PP · 2019, 2023: PSOE (PSPV) |

En 2017 y 2021 el Gobierno catalán estaba intervenido o en funciones; cuenta el partido del último presidente. Los
nombres entre paréntesis son alternativas por si el principal no está en los resultados del evento.

## Cierre (2026-11-06)

### Lo que existe, tarea a tarea

Nueve commits en la rama `analisis-voto-oculto` (`23b4fab..e945f86`), más el de este cierre, construidos con subagentes y revisión por tarea.

- **Tarea 1** (`8ff9874`, «Estudios: registro, esquemas y fixtures del estudio voto-oculto»): `mtpy/lib/bundle.py`,
  `mtpy/lib/publish/studies.py`, los cuatro fixtures `tests/fixtures/bundle/voto-oculto-*.json`, `tests/fakes.py`,
  `tests/test_bundle_unit.py`, `tests/test_publish_unit.py`.
- **Tarea 2** (`59ff8c3`, «Estudio voto-oculto: parámetros, tabla del Gobierno, catálogo y lectura de elecciones y run
  actual»): `data/voto-oculto/params.json`, `data/voto-oculto/incumbents.json`, `mtpy/lib/studies/shy/__init__.py`,
  `data.py`, `tests/test_shy_data.py`.
- **Tarea 3** (`4f9ec5b`, «Estudio voto-oculto: error de bloques y linajes con la regla de suma, cifras, extremos y
  tamaño»): `mtpy/lib/studies/shy/stats.py`, `tests/test_shy_stats.py`.
- **Tarea 4** (`e7ab8cc` y `2dd957b`, «Estudio voto-oculto: el estudio entero desde las partes publicadas (filas,
  resumen, grupos, veredictos)» y «Estudio voto-oculto: polls − results exacto en la fila, tamaño sobre errores sin
  redondear, docstring y tipos»): `mtpy/lib/studies/shy/study.py`, `tests/test_shy_study.py`.
- **Tarea 5** (`68786ac`, «Estudio voto-oculto: exportadores, meta con la entradilla viva, estudio vivo en el registro
  y documentación»): `mtpy/lib/publish/shy.py`, `mtpy/lib/publish/__init__.py`, `mtpy/lib/publish/studies.py`,
  `tests/test_publish_shy.py`, `tests/test_publish_studies.py`, `test_jobs_publish.py`, `test_pipelines_publish.py`,
  `test_publish_herd.py`, `data/README.md`, `deploy/README.md`, `docs/web/contrato.md`.
- **Tarea 6** (`15f3617` y `f8b051e`, «Web: página /analisis/voto-oculto con veredictos, textos vivos condicionados,
  tablas y la isla de los gráficos; enlace desde /metodo/promedio» y la ronda de correcciones «Web: /analisis/voto-oculto
  sólo dice que el PSOE y el Gobierno no pasan la prueba si no salen subestimados, …»): `mtpy/lib/pages/shy.py`,
  `pages/__init__.py`, `pages/analisis.py`, `pages/metodo.py`, `web/templates/analisis_voto_oculto.html`,
  `web/templates/metodo_promedio.html`, `web/dist/css/site.css`, `tests/test_pages.py`, `tests/test_web_api.py`.
- **Tarea 7** (`e945f86`, «Web: gráficos de errores con signo (barras cronológicas y nube por partido) y script de
  /analisis/voto-oculto»): `web/dist/js/charts/errors.js`, `web/dist/js/pages/analisis-voto-oculto.js`,
  `tests/test_js_charts.py`.
- **Tarea 8**: esta sección.

### Decisiones tomadas al construir

- P3: el veredicto es un «DE {SIGLAS}» por partido unidos con « Y » («DE PSOE Y DE IU / PODEMOS / SUMAR»); la cola del
  subtítulo une las siglas con « y ».
- `ShyStudy.scopes` son los códigos del catálogo con al menos una elección leída; `inputs[scope].events` cuenta
  elecciones leídas, no carpetas.
- `figures([])` devuelve `None` y las filas de partido guardan `figures: None`; la página pinta guiones.
- Entre las tareas 1 y 5 la suite estuvo en rojo a propósito (12 fallos de `STUDY_RUNNERS` sin el ejecutor de
  `voto-oculto`); la tarea 5 los dejó en verde. Lo mismo entre la 6 y la 7 (`test_referenced_assets_exist` por el
  script que crea la 7).
- Ajustes de texto de pruebas aceptados: manifiesto retirado por la ruta con prefijo del lector; `copy_event` desplaza
  las fechas de los sondeos con la de la elección.
- El paquete sintético de pruebas reescribe los linajes para añadir FA (el plan los dejaba sin ella).
- Las cifras se calculan sobre errores sin redondear y en las filas `polls − results == error` exactamente;
  `summary.size` también sale de los errores sin redondear.
- El subtítulo usa el marcador `{vox_clause}` para poder quitar la cláusula de Vox sin datos; la cola del subtítulo
  cambia según el veredicto.
- Sin datos de Vox, el veredicto de P2 es «SIN DATOS».
- La frase de Método va al final del primer párrafo de «Qué límites tiene el promedio», detrás de `shy_published`
  (la línea original quedaba dentro de una rama `if house_example`).
- Tres frases de la maqueta se reescribieron por ser falsas con las cifras: «ningún partido grande sale subestimado de
  forma sistemática» (PSOE 49/43), «Vox aparece en nuestra base en {año}» (no «se presenta desde 2018») y la respuesta
  sobre Tezanos sin «y el CIS tampoco desde entonces» (no está en las partes).
- Condiciones extra en las frases de la racha del PSOE y de «racha desaparece».
- Ejemplos del PP: generales desde 1996 con error del PP > 3.
- No se muestra la cifra de Ciudadanos; el ejemplo de abril de 2019 del PP se lee de `elections`.
- Bandera `generals_fail`; la caja 2 («no se distingue de cero») sólo sale con `not p2_yes`.
- Los partidos de P3 salen del veredicto, no se recalculan de las filas.
- Encabezado neutro de la nota de Podemos.
- Tarjeta «Después del 29-N» una vez publicada la elección: sin run y sin `meta.election`, muestra la fila de la última elección general sólo si su año coincide con `summary.last_year`, con sus tres filas (sondeos finales, resultado, error) leídas de los dicts `polls`/`results` de la fila de `elections`; si no, una frase neutra.
- La pastilla no usa `meta.election` (sin «añade la fila de –»).
- Aceptado sin tocar (menores): `_clean` copiado de `herd.py`, dos líneas de más de 120 columnas, `xMax` fijo en 8 en la
  nube, `signColor(null)` en azul.

### Lo que no se verificó

- No se ha ejecutado ningún job contra S3 ni ECS; tampoco la cadena `update` de ECS con el estudio vivo nuevo.
- No se ha visto la página en un móvil.
- Sí se verificó: la suite completa (2191 passed, 81 skipped, 1 aviso previo; la base en `dev` era 2146 passed) y un render con Chrome sin cabeza de la página con el paquete
  fixture (tres gráficos dibujados, sin errores en la consola).

### Lista de Luis

- `S3_BUCKET= python job.py publish '{"what":["study"],"study":"voto-oculto"}'` en local primero; después contra su S3
  (o esperar al `update` diario, cuyo paso `studies` lo publica en cuanto la imagen esté desplegada).
- Recorrer `/analisis/voto-oculto` (escritorio y móvil: las barras, el desplazamiento de la tabla de P3, los botones
  de descarga), `/analisis`, Inicio y `/metodo/promedio`.
- Desplegar la imagen.
- Tras el 29-N: `python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"2026-11-29"}'` y luego `python job.py publish '{"what":["studies"]}'` (la página añade sola la
  fila de 2026).
