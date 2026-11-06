# Web, Análisis: «¿Se copian las encuestadoras en la recta final?» (`/analisis/rebano`)

Fecha: 2026-11-05. Estado: **CONSTRUIDA el 2026-11-05 en la rama `analisis-rebano` (ver «Cierre»)**; aprobada por Luis el 2026-11-05 (brainstorm corto cerrado el 5 de noviembre; maqueta v1
aprobada: `docs/superpowers/specs/2026-11-05-web-analisis-rebano-mockup.html`; revisión del constructor en la sección
final, sus dos preguntas cerradas como decisiones mías y aprobadas con la spec). Tercer estudio de la sección «Análisis», tras el voto CERA (`2026-10-19-web-analisis-voto-cera-design.md`) y el
sesgo del CIS (`2026-10-24-web-analisis-sesgo-cis-design.md`), cuyo patrón de construcción (opción B: estudio publicado
calculado desde las partes del bundle) se repite. Esta spec describe lo que cambia respecto de `dev` (ea54e9b).

## Intención

Segundo análisis de la lista para redes antes del 29-N (memoria `ideas-analisis-redes`, 2026-11-05): una **página viva**
que conteste si las encuestadoras se copian en la recta final, con cifras que se recalculen con cada actualización del
promedio hasta la veda del 23 de noviembre, para publicarla hacia el 12-15 de noviembre y darle hilo en redes.

Lo que dicen los datos el 5 de noviembre (reconocimiento sobre las partes publicadas, scripts en el scratchpad de la
sesión): en las siete generales desde 2008 la dispersión de los sondeos alrededor del promedio, relativa al error de
sus muestras, baja en las últimas 8 semanas (media de 1,68 en el resto del ciclo a 1,20 en las últimas 8 semanas y 1,05
en las últimas 3); en 2008, 2016 y noviembre de 2019 bajó de 1 en las últimas 3 semanas (menos variación de la que
permiten las muestras); en 2026 la dispersión lleva todo el ciclo en 1,0-1,1 y, casa a casa, en las últimas 8 semanas,
SocioMétrica 0,36, NC Report 0,44 y Sigma Dos 0,57; el CIS, fuera del cálculo, se separa del promedio 10,7 errores de
su muestra. Todas esas cifras son vivas: la página las calcula en cada publicación.

Titular y entradilla (maqueta v1, con las cifras vivas): **«¿Se copian las encuestadoras en la recta final?»** / «En las
siete generales desde 2008, los sondeos se han parecido más entre sí cuanto más cerca estaba la elección: en las últimas
tres semanas, tanto como si todas las casas sacaran sus muestras de la misma urna. En 2026 ya van así desde el
principio, y las tres casas con más sondeos publican cifras que oscilan menos de lo que permite el azar de sus muestras.»

## Decisiones de Luis (brainstorm del 2026-11-05)

1. Eje de la página: la **convergencia entre casas semana a semana** (opción A), con el rebaño casa a casa en la recta
   final (opción B) como tercera pregunta. El CIS fuera del cálculo y visible aparte.
2. Maqueta v1 aprobada tal cual («Adelante»): titular, entradilla, tres cifras, tres preguntas con veredicto, P1 con el
   gráfico de semanas y la tabla, P2 con las tres cajas de prueba y el CIS aparte, P3 con la tabla viva de 2026 y la
   matriz casa × ciclo, «Después del 29-N», «Datos y método».
3. Las cajas naranjas de la maqueta quedaron sin respuesta expresa: las resuelvo yo como se dice en «Decisiones mías» y
   Luis las corrige al leer esta spec.

## Decisiones mías (propuestas en la maqueta, pendientes de veto)

- **Sin predicción registrada**, como el sesgo del CIS; la comprobación de después del 29-N es la tarjeta «Después del
  29-N» (D6), no un registro.
- **Gráfico de semanas** (no el dumbbell): es el que deja avanzar a 2026 a diario.
- Veredictos: P1 **«SÍ»** (verde) cuando todos los ciclos pasados se acercan, «EN {k} DE {n}» (ámbar) si no; P2 **«EN
  {k} DE {n}»** (ámbar; verde «SÍ» si k = n, rojo «NO» si k = 0); P3 **«{k} CASAS»** (azul; «NINGUNA» si k = 0, «1
  CASA» si k = 1).
- **Umbral de tres sondeos** con muestra publicada por casa y ventana; **ventana final de 8 semanas** (56 días);
  **últimas 3 semanas** = 21 días; **sólo generales**; ciclos desde 2008.
- La tarjeta «Después del 29-N» enseña ya la cuenta de los ciclos pasados (D3 `summary.accuracy`) y gana la fila de
  2026 cuando el evento se publique con resultado.
- Los estudios «vivos» (este y el sesgo del CIS) se **republican en cada `update`** (D5).

## Hechos del contexto (sin cifras a mano)

- La veda de sondeos (LOREG) empieza el 23 de noviembre de 2026; la elección es el 29. Ambas fechas salen del evento
  (`meta.event_date` del run) y de la guardia ya existente; la página no las escribe a mano.
- La medida del rebaño por casa es la de Método (`/metodo/rating#herding`, M12/M13 del código: `Forecaster.measure_herding`
  y `pool_herding`), acotada aquí a una ventana. La medida de dispersión de todos los sondeos es nueva (D4).

## Objetivos

1. Un **estudio publicado** `rebano` (`site/v1/studies/rebano/`) con sus partes, gemelos CSV, esquemas, entrada en el
   manifiesto y rutas de la API, calculado desde las partes publicadas del bundle (D2, D3, D4).
2. Su **republicación con cada `update`** y tras `publish analysis`/`event`, para que la página corra a diario hasta la
   veda y siga después (D5).
3. La página `/analisis/rebano` con la estructura de la maqueta, textos con cifras vivas y condicionados, gráficos
   propios, descarga de cada tarjeta como imagen, contenedor fluido; entra sola en `/analisis` y en la banda de Inicio
   (D6, D7).
4. Comprobación pública de redacción, documentación, una frase con enlace en la tarjeta «El rebaño, aparte» de Método
   (D8).

## No objetivos

- Ningún acceso a la base de datos: el estudio no instancia `Computer` ni `Forecaster`. Si una cifra no está en las
  partes publicadas, no entra.
- Ningún cambio en el modelo, en el peso del promedio, en la medida del rebaño del análisis (`herding`) ni en /casas.
- Sin ámbitos autonómicos (sólo `es`), sin registro de predicción, sin conmutadores de medida, sin dumbbell.
- Nada de nginx, Docker, CSP ni modo oscuro.

## Anti-objetivos

- Que una cifra del texto esté escrita a mano en la plantilla: todo número sale de las partes; los textos cuyo sentido
  depende del signo o del recuento se condicionan (D6).
- Que la página rompa cuando falte algo: sin run nacional, la parte `current` es nula y la página habla sólo de los
  ciclos pasados; un ciclo sin parte `cycle` se omite y se anota en `meta.inputs`; sin ciclos pasados la publicación se
  rechaza (D4).
- Que el rebaño por casa del ciclo entero no coincida con el de /casas y el perfil: esa cifra se **lee** de la parte
  `herding` del análisis nacional, no se recalcula (D2).
- Perder la lectura sin JavaScript: textos, veredictos, cifras y tablas salen del servidor; sólo los dos gráficos los
  dibuja el navegador.
- Que la republicación diaria toque los runs, los análisis o el `voto-cera`.

## Enfoques considerados

- **A. Estudio desde la base de datos**: un paquete que instancia el `Forecaster` por ciclo y repite el ajuste del
  promedio. Descartado: duplica un ajuste pesado que ya está publicado (la curva de cada elección y la del run) y
  cualquier diferencia de parámetros dejaría la página y las páginas de elección sin cuadrar.
- **B. Estudio desde las partes publicadas** (elegido): el patrón del sesgo del CIS. Lee `cycle` y `series` de cada
  generales, `polls` y `series` del último run nacional y `herding` del análisis nacional; tarda segundos; se encadena
  tras cada publicación.
- **C. Página calculada al servir**: descartado por lo mismo que la vez anterior (sin CSV ni API, lecturas de S3 en
  frío por trabajador).

## D1. Declaraciones del estudio (`mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`)

```python
STUDIES = ('voto-cera', 'sesgo-cis', 'rebano')
STUDY_PARTS['rebano'] = ('meta', 'summary', 'cycles', 'weeks', 'houses')
STUDY_SCHEMAS['rebano'] = {part: 'rebano-' + part for part in STUDY_PARTS['rebano']}
STUDY_CSV['rebano'] = ('cycles', 'weeks', 'houses')
STUDY_CSV_OF['rebano'] = {'cycles': ('cycles',), 'weeks': ('weeks',), 'houses': ('houses',)}
```

- `SCHEMAS` gana las cinco entradas `rebano-*` con las claves obligatorias de D3.
- `publish/studies.py`: `STUDY_TITLES['rebano']`, `STUDY_EXPORTERS['rebano']`, `STUDY_META['rebano'] = export_herd_meta`,
  `STUDY_RUNNERS['rebano'] = {'compute': compute_for_publish (de publish/herd.py), 'registrable': False, 'live': True}`.
  Clave nueva `live` en los tres estudios: `voto-cera` falso, `sesgo-cis` verdadero (D5).
- Los exportadores viven en `mtpy/lib/publish/herd.py` (patrón de `lean.py`: importa el paquete del estudio dentro de
  `compute_for_publish`; `test_importing_publish_does_not_load_the_study_package` gana el paquete nuevo).
- Consumidores ya preparados por estudio desde el sesgo del CIS (`Studies.part_action`, `webapi.checks`,
  `pages.analisis._csv_links` con `CSV_LABELS['rebano']`, `tests/test_webapi_unit.write_study`): sólo ganan el slug.

## D2. Las entradas

Todo con un `BundleReader(fs, prefix=bundle.PREFIX)` sobre el `fs` del job, también en `dry_run` (como el sesgo del CIS).
Ámbito fijo `es` (parámetro `scope`).

**Ciclos pasados.** Cada fecha de `reader.list_events('es')` con `event_date >= params.since` (2008-01-01) y parte
`cycle` publicada. De `cycle`: `parties` y `polls` (`date`, `pollster_id`, `pollster`, `start_date`, `end_date`,
`sample_size`, `weight` y una columna por partido). De `series`, si existe (hoy desde 2015, la del backtest): `dates`
y `mean` por partido, la curva del promedio que dibuja la página de la elección (la del backtest, horizonte 6, cubre el
ciclo entero: sólo los últimos días quedan fuera de ella). De `pollsters`, si existe: `last_error_avg` y
`last_error_blocks` por casa, el error de su último sondeo (para `houses` y `summary.accuracy`; `error_avg` es la media
de todos sus sondeos del año y no mide la recta final). Un ciclo sin `cycle` se omite y se anota.

**Ciclo actual.** El run `manifest.scopes.es.latest`: `meta` (`event_date`, `as_of`, `run_id`), `polls` (misma forma
que `cycle`) y `series` (la curva viva, cortada en `as_of`). Es «actual» mientras `event_date` no tenga carpeta
publicada en `events/es/` con parte `results`; cuando la tenga (tras el 29-N y `publish event`), ese ciclo entra en los
pasados por su parte `cycle` y `current` es nulo. Sin manifiesto o sin run nacional: `current` nulo y anotación.

**Rebaño del ciclo entero por casa.** `analysis/es/herding.json`, `events` (`event_date`, `pollster`, `n`, `ratio`,
`p_value`): se copia en `houses.h_cycle` por ciclo y casa cuando hay fila (la parte ya incluye el ciclo en curso, así
que 2026 también lo tiene). Sin la parte, `h_cycle` nulo.

**Filtros comunes.** Entran los sondeos con `sample_size > 0` y `weight > 0` (los que usa el promedio; sin muestra
publicada no hay error muestral que comparar), con `end_date` (o `date` sin ella) entre 0 y `params.max_days` (365)
días antes de la elección. Las **cuatro series principales** del ciclo son los cuatro partidos con mayor media de voto
en sus sondeos (`params.n_series`, como `measure_herding`). Las casas de `params.exclude` (`['CIS']`) quedan fuera de
todas las cifras agregadas y se calculan aparte.

## D3. Las partes de `rebano`

Envoltorio como todo estudio (`scope`, `run_id` y `mode` nulos; `schema` `rebano-{parte}@1`). Nombres de campo en
inglés; redondeos: cocientes y errores a 2 decimales, p-valores con tres cifras significativas (como `sesgo-cis`).

| Parte | Contenido (claves obligatorias en el esquema) |
|---|---|
| `meta` | `study`, `title`, `subtitle` (con «{k} de {n}» vivo y la frase de 2026 sólo con `current.like_final`), `scope`, `election` (`{date, days_left, as_of}` del ciclo actual o nulo), `params` (D4), `inputs` (por ciclo: parte `cycle`/`series`/`pollsters` leída con su `generated_at` o `skipped`; `current` con `run_id`, `as_of`; `herding` con su `generated_at` o `skipped`), `sources`, `commit`, `dirty`, `versions`, `seconds`. Sin claves de registro. |
| `summary` | `n_cycles`, `k_narrowed` (ciclos con `last.ratio < before.ratio`, comparados los cocientes ya redondeados de las filas), `narrowed` (etiquetas), `mean_before`, `mean_last`, `mean_final` (medias simples de los cocientes de los ciclos pasados), `k_below_one` y `below_one` (ciclos con `final.ratio < 1` redondeado a dos decimales, etiquetas: un 1,00 impreso nunca cuenta como «por debajo de 1»), `sign` (`{n, k, p}`: probabilidad binomial de `k` o más ciclos a la baja entre `n` con moneda justa; la «prueba 3» de P2), `current` (`{label, before, last, final, cycle}` con `{k, ratio}` cada uno, `days_left`, `rank_before` = puesto del cociente del resto del ciclo entre los de los ciclos pasados, 1 = el más bajo, empates a dos decimales con el mismo puesto, y `like_final` = `cycle.ratio <= mean_last`, «la dispersión de todo el ciclo ya está donde los pasados sólo llegaron en la recta final»; nulo sin run), `houses_now` (`{n, k, herd: [{pollster, h}], pending: [{pollster, k}]}`: casas medibles en la ventana actual, cuántas por debajo de 1 y cuáles, ordenadas por `h`, y las que tienen entre 1 y `min_polls − 1` sondeos con muestra), `cis` (`{current: {k, ratio}, cycle: {k, ratio}, own: {k, h, p}}` del ciclo actual, nulos sin sondeos), `accuracy` (`{n, herd: {k, error_avg, error_blocks}, rest: {...}}`: las casas de `houses` de ciclos con resultado, partidas por `h < 1`, con la media de `last_error_avg` y la media de `|last_error_blocks|` de cada grupo: el error de su **último sondeo**, el que usan el rating y el Histórico, no la media de todo el año; nulo si alguna mitad está vacía). |
| `cycles` | Una fila por ciclo, pasados y actual: `event_date`, `label` («2019 abr.» cuando el año se repite, como `_label` del sesgo del CIS), `current` (bool), `names` (las cuatro series), `consensus` (`'curve'` o `'local'`), `polls`, `with_sample`, `houses` (casas con sondeo con muestra), `before` (`{k, ratio}`, días 57-365), `last` (`{k, ratio}`, 0-56), `final` (`{k, ratio}`, 0-20), `change` (`last.ratio − before.ratio`), `cis` (`{last: {k, ratio}, cycle: {k, ratio}, own: {k, h, p}}`). CSV `cycles` (plano: `before_k`, `before_ratio`…). |
| `weeks` | `buckets` (`[{lo, hi, label}]` en semanas, de la más lejana a la más cercana: `27-52 … 0-1`), `rows` (`[{event_date, label, current, lo, hi, k, ratio}]` por ciclo y cubo, `ratio` nulo sin sondeos) y `mean` (`[{lo, hi, k, ratio}]`: media simple de los cocientes de los ciclos pasados con dato en el cubo, `k` ciclos). CSV `weeks`. |
| `houses` | Una fila por ciclo y casa con `k >= params.min_polls` sondeos con muestra en la ventana final (CIS incluido, con `cis` verdadero): `event_date`, `label`, `current`, `pollster`, `slug` (perfil), `k`, `h`, `p`, `cis`, `h_cycle` (de `herding`, nulo sin fila), `last_error_avg`, `last_error_blocks` (de `pollsters`: el error del último sondeo de la casa, el de bloques con el signo con que se publica; nulos en el ciclo actual). Orden: ciclo y `h` ascendente, nulos al final. CSV `houses`. |

`meta.json` se escribe la última; el orden de escritura es el de `STUDY_PARTS['rebano'][1:]`.

## D4. El cálculo: `mtpy/lib/studies/herd/`

Paquete nuevo (nombre en inglés, «herd» = rebaño), ligero (`numpy`, `pandas`, `scipy.stats.chi2` y `binom` dentro de
funciones):

- `data.py`: `load_params(data)`, `MissingInput`, `read_cycles(reader, params)` (lista de ciclos pasados con sus
  sondeos, curva y errores por casa, más `inputs`), `read_current(reader, params)` (el run, o `None`),
  `read_herding(reader, scope)` (dict `(event_date, pollster) → {n, ratio, p_value}`), `is_current(reader, scope, date)`.
- `stats.py`, con la medida:
  - **Residuo** de un sondeo en una serie: su cifra menos el **promedio en su fecha de fin de campo**: la curva
    (`series.mean[name]` en `end_date`; si la fecha cae fuera de la curva, el último día de la curva anterior a ella)
    o, en un ciclo sin curva (`consensus 'local'`), la media simple de los sondeos de las **demás** casas con fin de
    campo a ±`params.local_window` (10) días; sin dos sondeos ajenos no hay residuo. El sesgo del CIS no entra en el
    consenso local (está en `exclude`).
  - **Varianza esperada** del sondeo en la serie: `1e4 · p (1 − p) / n`, con `p` el promedio en tanto por uno
    (acotado a [0,005, 0,995]) y `n` la muestra publicada.
  - **Dispersión** de un grupo de sondeos: `sqrt(mean(residuo²) / mean(varianza esperada))` sobre todas las parejas
    sondeo × serie del grupo; `k` = sondeos distintos del grupo. Cubos por días hasta la elección: `before` 57-365,
    `last` 0-56, `final` 0-20 y los `buckets` de semanas (`lo·7 … hi·7+6` días).
  - **Rebaño por casa** en una ventana: la cuenta de `Forecaster.pool_herding` sobre sus sondeos de la ventana: por
    serie, `ss_obs = Σ (r_i − r̄)²`, `ss_exp = (1 − 1/k) · Σ var_i`; sumadas sobre las cuatro series, `h = sqrt(ss_obs /
    ss_exp)`, `p = chi2.cdf(h² · (k − 1), k − 1)`. Se diferencia de la del análisis en dos cosas, que la nota técnica
    dice: la referencia es la curva publicada (con los sondeos de la propia casa dentro, no dejada fuera) y la ventana.
  - `sign_test(n, k)`: `binom.sf(k − 1, n, 0.5)`; `accuracy(houses)`.
- `study.py`: `compute_herd(reader, data, params=None) -> HerdStudy` (dataclass con los bloques de D3) y la validación:
  sin ciclos pasados, `MissingInput('events/es')`.
- Parámetros en `data/rebano/params.json` (fila en `data/README.md`): `scope 'es'`, `since '2008-01-01'`, `max_days
  365`, `window_days 56`, `final_days 20`, `local_window 10`, `n_series 4`, `min_polls 3`, `exclude ['CIS']`, `buckets
  [[0, 1], [2, 3], [4, 5], [6, 7], [8, 11], [12, 17], [18, 26], [27, 52]]`.
- Pines de test: un bundle sintético en `tests/fakes.py` (`write_herd_bundle`) con dos ciclos pasados (uno con curva,
  otro sin ella), tres casas y el CIS, y el run actual; cifras calculadas a mano en el test (dispersión de un cubo,
  `h` y `p` de una casa con tres sondeos, el consenso local sin la casa ni el CIS, `k_narrowed`, `sign.p`).

## D5. La publicación, el encadenado diario y el job

- `publish.publish_study('rebano', writer)`: como el sesgo del CIS (`register` rechazado; manifiesto
  `{published_at, registered_at: null, checked: false}`).
- Comando suelto: `python job.py publish '{"what":["study"],"study":"rebano"}'`.
- **Acción nueva `studies`** del job `publish` (no troceada por ámbito, sin guardia LOREG): publica, en el orden de
  `bundle.STUDIES`, los estudios con `STUDY_RUNNERS[s]['live']`, y da el resultado de cada uno (`published`/`failed`);
  `what: ["analysis", "studies"]` las encadena. El `update` gana `'studies'` en `sequence['publish']` **en último
  lugar**, tras `'parties'`: la tarea diaria de ECS republica este estudio y el sesgo del CIS cada día, con el run recién
  publicado, y un estudio que falle con el `stop_on_error` por defecto nunca deja sin publicar el catálogo (un forecast
  `refused` por la veda no para la cadena: sólo cuenta `failed`). Con `stop_on_error` falso el fallo queda en el resumen
  y la alerta como cualquier fallo del `publish`.
- Guía de Luis tras construir: `publish '{"what":["study"],"study":"rebano"}'` contra su S3; a partir de ahí lo lleva
  el `update` diario. Tras el 29-N, `publish event` de `es/2026-11-29` y una republicación.
- `deploy/README.md` («### Estudios», `what`), `docs/web/contrato.md` (estudios, rutas, árbol) ganan el estudio y la
  acción `studies`.

## D6. La página `/analisis/rebano`

**Rutas y registro.** `pages.analisis.STUDIES['rebano'] = {'template': 'analisis_rebano.html'}`; `CONTEXTS['rebano'] =
herd_context`; módulo de vista `mtpy/lib/pages/herd.py` (en `test_pages_package_layout`). El índice `/analisis` y la banda
de Inicio lo listan en el orden de `bundle.STUDIES` (el tercero).

**Contexto** (`herd_context(state)`): lee las cinco partes con `study_part`; cualquier parte o clave ausente da `None`
y la plantilla lo guarda. Entrega `study_meta`, `summary`, `cycles` (filas, actual la última), `verdicts` (D6 «Verdictos»),
`kpis`, `tests` (las tres cajas de P2), `cis_rows` (tabla del CIS por ciclo), `houses_now` (tabla de 2026),
`matrix` (filas = casas con fila en algún ciclo, en el orden: las de 2026 primero por `h`, luego por número de ciclos
con `h < 1` descendente, el CIS el último; columnas = ciclos), `accuracy`, `csv_links`, `sources`, `published_at` y la
isla `initial` con sólo lo que dibujan los gráficos: `weeks` (`{buckets, rows, mean}`) y `matrix` (`{events, houses,
cells}` para `matrix.js`, `cells` como `[x, y, h, k, p]`: la columna, la fila, la medida, los sondeos y el p-valor, para el
tooltip).

**Verdictos y textos condicionados** (cada frase fuerte con su condición; el constructor fija las demás):

- P1: «SÍ» si `k_narrowed == n_cycles`; si no, «EN {k} DE {n}» en ámbar. `.sum`: «En los {n} ciclos…» sólo con k = n;
  si no, «En {k} de los {n} ciclos…». La frase «2026 es distinto: la dispersión de todo el ciclo ({cycle.ratio}) ya
  está donde los ciclos pasados sólo llegaron en la recta final ({mean_last})» sólo con `current.like_final`; si no,
  una frase neutra con la cifra y su puesto (`rank_before`). (El ranking solo no vale: 2008 está en 1,05 y 2026 en
  1,06, y la frase se apagaría el día del estreno; la comparación con la recta final de los pasados es la tesis.)
- P2: veredicto por `k_below_one`; la caja 3 con `sign.p` («1 entre {1/p}» redondeado; «menos de 1 entre 1.000» por
  debajo de 1e-3). La frase «Las diferencias entre casas … desaparecen en la recta final» sólo si `mean_final <= 1.1`.
- P3: «{k} CASAS» con `houses_now.k`; «NINGUNA» con 0; sin run, la tarjeta habla sólo de los ciclos pasados. «Las de
  siempre» (`.sum`) se construye con las casas de la matriz con dos o más ciclos por debajo de 1, nombradas; sin
  ninguna, la frase cambia a «Ninguna casa repite todavía».
- KPI 2 («1,06 hoy»): subtítulo «a {días} días del {election} · en el resto del ciclo, el {rank_before}.º más
  bajo de los {n + 1}» y, con `like_final`, «· ya como una recta final». Los días se cuentan **desde hoy** (Europe/Madrid,
  `pages.common.days_left(election.date)`), no desde el `as_of` del run: dentro de la veda la previsión se rechaza y la
  republicación diaria conserva el último run (el del 23-N, `days_left` 6), así que el `days_left` del estudio se queda
  como dato y la página no lo usa para los textos. Con 0 días o menos (elección celebrada, pendiente de `publish
  event`): etiqueta «Dispersión de los sondeos de {año}» (sin «hoy») y subtítulo «al cierre de los sondeos · …».
- KPI 3: «{k} de {n}» de `houses_now`; sin run, «Sin run» no: la KPI pasa a la casa más en rebaño del último ciclo pasado.

**Estructura** (ids de la maqueta; `m.card_start`/`qcard`; botón de descarga por tarjeta con `card_image_button` como
el sesgo del CIS):

- `#study-head`: pastillas «Análisis», «Página viva · se recalcula con cada actualización hasta la veda del {veda}»
  (la fecha de la veda = `election.date − 6` días; tras la veda, «se recalcula con cada actualización»; con la
  elección celebrada, «Página viva · elección celebrada, pendiente del escrutinio», sin fecha; sin run, «Evidencia
  histórica · {n} generales»), «{n} generales desde {since} · a {días} días del {election}» (sin los días una vez
  celebrada); `h1` y entradilla vivas. Los días, `after_veda` y la elección celebrada salen de hoy, como en el KPI 2;
  el título de la tabla de P3 cambia «a {días} días del …» por «al cierre de los sondeos» y la pregunta 3 sigue siendo
  «¿Quién va en rebaño en {año}?».
- `nav.toc`: Resumen · 1 · ¿Se acercan? · 2 · ¿Más de lo que permite el azar? · 3 · ¿Quién va en rebaño en {año}? ·
  Datos y método.
- `m.kpis`: «Ciclos en que los sondeos se acercaron en la recta final» `{k} de {n}` («la dispersión media pasa de
  {mean_before} en el resto del ciclo a {mean_last} en las últimas 8 semanas y a {mean_final} en las últimas 3 · 1 = lo
  que permite el azar de las muestras»); «Dispersión de los sondeos de {año}, hoy» `{current.cycle.ratio}` («a
  {days_left} días del {election}; {frase de rank_before}»); «Casas en rebaño en las últimas 8 semanas» `{k} de {n}`
  («{casa h} · … · el CIS, aparte: {cis.current.ratio}»).
- `#resumen`: «Tres preguntas, tres respuestas», las tres tarjetas-enlace con una frase viva cada una y la nota azul
  («Medimos cuánto varían las cifras publicadas alrededor del promedio…»).
- `#p1`: entradilla con la medida en una frase; `grid2` con el gráfico de semanas (`#weeks-chart`) y la tabla
  `cycles` (Ciclo · Sondeos · Resto del ciclo · Últimas 8 sem. · Últimas 3 sem. · Cambio con flecha `.dlt`, fila «Media
  de los {n}» y la fila de 2026 destacada `.now`); `.sum`; «Cómo leerlo».
- `#p2`: entradilla; `.tests` (ciclos por debajo de 1 en las últimas 3 semanas; media de los ciclos en las últimas 3
  semanas «desde {mean_before}»; probabilidad de una caída así por azar); `grid2` con la tabla «El CIS, aparte» (Ciclo ·
  Sondeos (8 sem.) · Distancia al promedio · Oscilación propia, de `cycles[].cis`) y «Qué significa la cifra» (`.tech`
  con las cuatro lecturas, el enlace a `/analisis/sesgo-cis`); `.sum`.
- `#p3`: entradilla; `grid2` con la tabla de 2026 (Encuestadora con enlace al perfil · Sondeos · Rebaño · p-valor ·
  barra; debajo, las casas de `houses_now.pending` nombradas como «aún sin sondeos suficientes» y el CIS «aparte»;
  pie con `h_cycle` de las tres primeras) y la matriz (`#matrix-chart`, leyenda de seis tramos: < 0,5 · 0,5-0,7 ·
  0,7-0,9 · 0,9-1,2 · 1,2-1,6 · > 1,6); `.sum` «Las de siempre»; «Cómo leerlo».
- `#despues`: «Después del 29-N»: la cuenta de `accuracy` sobre los ciclos pasados («las casas que iban en rebaño
  erraron {x} puntos de media en el voto; las demás, {y}») y, cuando 2026 tenga resultado, su fila.
- `#datos`: datos (sondeos de cada ciclo tal como se publicaron, con enlace a `/elecciones`; la curva del promedio de
  cada elección y la del run; sólo muestra publicada; el CIS aparte; `herding` de /casas), método (`.tech`: residuo,
  varianza esperada, dispersión, cubos, rebaño por casa, consenso local de 2008 y 2011, y las dos diferencias con la
  medida de Método), las tres pastillas CSV y las fuentes.

**Textos.** Voz «nosotros», «sólo» con tilde, «encuestadora»/«casa», nunca «empresa», «actualización» para los runs,
comillas «»; cifras con `num`/`int`/`delta`. Los textos de la maqueta v1 son la referencia. `assert_public_wording` se
aplica a `/analisis/rebano`.

## D7. Los gráficos y el contenedor fluido

- **Semanas** (`charts/line.js`, `renderLines`; el constructor añade lo que falte sin romper a Método): eje x de
  categorías con las etiquetas de los cubos («27-52 sem.» … «0-1»), eje y 0-3 de 0,5 en 0,5 (tope ajustable al máximo
  si algún ciclo lo supera), una línea fina gris por ciclo pasado (`connectNulls`), la media en negro discontinuo, 2026
  en azul grueso con puntos y etiqueta final «2026 · {ratio}», línea roja en 1 con la etiqueta «lo que permite el azar
  de las muestras» y la franja 0-1 sombreada; tooltip por cubo con todos los ciclos. Alto 340 px; en móvil, 280.
- **Matriz** (`charts/matrix.js`): mismo módulo que el sesgo del CIS con una **escala nueva** (opción `scale: 'ratio'`
  o `center: 1`): rojo por debajo de 1 con intensidad por `1 − h` (tope en 0,5), azul por encima con intensidad por
  `h − 1` (tope 1), gris claro entre 0,9 y 1,2, celda vacía sin fila; valor en la celda; tooltip «{casa} · {ciclo}: rebaño
  {h} ({k} sondeos, p {p})». Columnas = ciclos cronológicos, 2026 la última y en negrita.
- Ambos módulos exportan su `*Option` para `tests/test_js_charts.py`. Montaje con `mountChart`; `grid2` colapsa en
  móvil; la matriz con desplazamiento horizontal.
- JS de página: `web/dist/js/pages/analisis-rebano.js` con el patrón de `analisis-sesgo-cis.js` (`readInitial`,
  `tryDraw`, `mountWhenReady`, `wireCardImages`).

## D8. Documentación, Método y memoria

- Contrato, README de despliegue y `data/README.md` (D5, D4).
- La tarjeta «El rebaño, aparte» de `/metodo/rating` gana una frase final con enlace, sólo cuando el estudio está
  publicado: «Cómo se acercan las encuestadoras en la recta final, ciclo a ciclo y en vivo, está en
  /analisis/rebano».
- Memoria del proyecto: entrada nueva `project-analisis-rebano` y el punto 2 de `ideas-analisis-redes` en marcha.

## Lo que queda al constructor

- Los nombres internos (`HerdStudy`, `herd_context`, `export_herd_*`), el orden exacto de las columnas de los CSV y los
  redondeos no fijados.
- La interpolación de la curva cuando `end_date` cae en un hueco de `series.dates` (último día anterior, como dice D4,
  o el más cercano: que lo fije y lo documente).
- Las frases neutras de respaldo de cada texto condicionado (D6), con el mismo tono.
- El tamaño mínimo de las tablas en móvil y los `scroll-hint`.

## Revisión del constructor (2026-11-05)

Un revisor independiente leyó la spec como constructor. Dos preguntas, resueltas aquí como decisiones mías:

1. **La condición de «2026 es distinto»**: con el ranking del resto del ciclo (2008 en 1,05, 2026 en 1,06) la frase y
   la cláusula de la entradilla se apagarían el día del estreno. Resuelto con `current.like_final` (la dispersión de
   todo el ciclo ≤ la media de las últimas 8 semanas de los ciclos pasados), que es la tesis de la página; el puesto se
   enseña como cifra neutra (D3, D6).
2. **El error de «Después del 29-N»**: `pollsters.error_avg` es la media de todos los sondeos del año; el acierto de la
   recta final es el del último sondeo (`last_error_avg`, `|last_error_blocks|`), el mismo que usan el rating y el
   Histórico (D2, D3).

Menores anotados por el revisor y recogidos: `houses_now.pending` para las casas con sondeos insuficientes (D3, D6);
`studies` en último lugar del `update` (D5); la curva del backtest cubre el ciclo entero y `herding` ya trae el ciclo en
curso (D2); el filtro `weight > 0` vale igual en `cycle` (sin agregadores) y en `polls` del run.

## Cierre (2026-11-05)

Construida en la rama `analisis-rebano` (nueve tareas, de `eaba7c9` a este cierre), con una revisión por tarea limpia.

### Lo que existe

- **Tarea 1, registro y fixtures (`902f8bb`)**: `mtpy/lib/bundle.py` (el estudio `rebano` en `STUDIES` y sus esquemas),
  `mtpy/lib/pages/analisis.py` (la tarjeta en `/analisis`), los fixtures
  `tests/fixtures/bundle/rebano-{cycles,houses,meta,summary,weeks}.json` y sus tests.
- **Tarea 2, lectores (`2adffde`)**: `data/rebano/params.json`, `data/README.md`,
  `mtpy/lib/studies/herd/{__init__,data}.py` (parámetros y lectura de ciclos, run actual y rebaño desde el paquete),
  `tests/fakes.py`, `tests/test_herd_data.py`.
- **Tarea 3, la medida (`e33722f`, `217ab5f`)**: `mtpy/lib/studies/herd/stats.py` (dispersión frente a la curva, rebaño
  por casa; el consenso local deja fuera los sondeos sin peso) y `tests/test_herd_stats.py`.
- **Tarea 4, el estudio (`7acc576`)**: `mtpy/lib/studies/herd/study.py` (ciclos, semanas, casas y resumen) y
  `tests/test_herd_study.py`.
- **Tarea 5, exportadores (`8b76159`)**: `mtpy/lib/publish/herd.py` (`export_herd_*`, con la entradilla viva en `meta`),
  `mtpy/lib/publish/studies.py` y `__init__.py` (registro del estudio, clave `live`), tests `test_publish_herd.py`,
  `test_publish_studies.py`, `test_publish_unit.py`.
- **Tarea 6, acción `studies` (`5d776bf`)**: `mtpy/pipelines/publish/Studies.py`, `mtpy/jobs/Publish.py`,
  `mtpy/jobs/Update.py` (último paso de `update`), `deploy/README.md`, `docs/web/contrato.md` y sus tests.
- **Tarea 7, la página (`9cd5728`)**: `mtpy/lib/pages/herd.py` (textos vivos condicionados, tablas, isla de datos),
  `mtpy/lib/pages/{__init__,analisis}.py`, `web/templates/analisis_rebano.html`, reglas en `web/dist/css/site.css`, el
  script `web/dist/js/pages/analisis-rebano.js` y `tests/test_pages.py`.
- **Tarea 8, los gráficos (`46f64d4`, `6170ed7`)**: `web/dist/js/charts/weeks.js` (dispersión por semanas),
  `web/dist/js/charts/matrix.js` (matriz con la escala en torno a 1), el script de la página, el CSS y
  `tests/test_js_charts.py`.
- **Tarea 9, este cierre**: el enlace desde `/metodo/rating` (bandera `herd_published` en
  `mtpy/lib/pages/metodo.py`, frase en `web/templates/metodo_rating.html`, test en `tests/test_pages.py`) y esta sección.

### Decisiones tomadas durante la ejecución

- **Ventanas en días**: 0-55 (recta final), 56-365 (resto del ciclo) y 0-20 (últimas semanas, `final_days` 21: 21
  días), en vez de las 0-56/57-365 de D3; los tramos son `lo·7 … hi·7+6`.
- **`charts/weeks.js` es un módulo propio**, no una extensión de `line.js`.
- **La entradilla se compone una sola vez** en `export_herd_meta`, y el arranque de la página es `meta.subtitle` tal
  como se publicó (sin duplicar la redacción en español en la capa web).
- **El consenso local deja fuera los sondeos con peso 0** (el promedio nunca los usó; un sondeo descartado no debe
  definir el consenso).
- **Las filas de `houses` no llevan `slug`**: el enlace al perfil lo construye la página con `slugify`.
- **`studies` es el último paso de `update`** (tras `parties`).
- **Los muestrarios de la leyenda y de la tabla usan los colores del gráfico** (misma fórmula rgba, texto blanco sobre
  las celdas oscuras).
- **La columna del ciclo en curso va en negrita** y los rótulos de la matriz van rectos.
- **La imagen de la tarjeta de la pregunta 1 sale en una columna.**
- `Cycle.polls` conserva las fechas como texto; `stats.residual_rows` deriva `end` y `days`.
- **Los veredictos comparan los cocientes redondeados** (corrección tras la revisión final, contra la regla del plan de
  comparar sin redondear): `narrowed` y `below_one` usan los cocientes a dos decimales de las filas, como `rank_before`
  y como `h < 1` sobre el `h` redondeado, para que un 1,00 impreso nunca salga como «bajó de 1»; las medias siguen
  usando los cocientes sin redondear. Los pines no cambian (2019 final 0,96, 2023 1,14; los dos ciclos se acercan).

### Menores aparcados para la revisión final

- Tarea 1: valor constante del bloque del CIS en el fixture de ciclos; el CSV sin aserción de cuerpo.
- Tarea 2: `MissingInput` sin uso en `data.py`; `read_cycles` se fía de `list_events` para elecciones pasadas.
- Tarea 3: guarda de curva vacía/NaT en `curve_level`; `_is_number` frente a `pd.NA`; `dof` k−1 conservador (documentado
  en Método); sin test del respaldo con `end_date` nulo.
- Tarea 4: casa excluida con k >= min_polls sin test; empates de `rank_before` sin test; `cis.own` agrupa todas las casas
  excluidas (vale con `exclude=[CIS]`); `below_one`/`narrowed` comparan cocientes sin redondear (los textos no deben
  contradecir un 1,00 mostrado); líneas largas en tests.
- Tareas 5 y 6: guarda de elección en los exportadores; supervivencia de p sin test; etiquetas de fuente sin fijar;
  líneas largas de docstring/test; borde de lista `live` vacía.
- Tarea 7: valores de respaldo `or 4`/`or 10` en la plantilla (moverlos a `herd.flags`); enteros sin `fmt_int`; umbrales
  de color duplicados en la leyenda; P2 «NO»/P1 «EN 0 DE 0» con resumen vacío (teórico); etiqueta del KPI 3 cuando el
  menor h >= 1 y `KeyError` en `house[k]`; lógica en la plantilla (ml−mb, pasado/vivo/local); `after_day` en línea en
  `analisis.py`; tres formas de unir nombres; `flags()` recalculado; importación privada de `_get`; espaciado E303/E305
  en `test_pages.py`; huecos de cobertura (píldora tras la veda, P2 SÍ, 1 CASA, claves de la isla, límites del color,
  textos de acierto y resultado).
- Tarea 8: umbral de etiqueta blanca (alpha >= 0,6) frente al texto oscuro de h2; el límite 1,2 es inclusivo en el
  gráfico y exclusivo en la tabla; la serie actual sin `connectNulls`; cobertura de la matriz centrada; E303/E302 en
  `test_js_charts.py`.
- Ajena a la rama: la advertencia de matplotlib que ya estaba en `dev`.

### Ola de correcciones tras la revisión final

Doce puntos de la revisión de la rama entera, en una ola (todo el conjunto de tests en verde):

1. Los días de la página (pastilla, KPI 2, título de la tabla de P3) se cuentan desde hoy y no desde el `as_of` del run;
   con la elección celebrada, «Página viva · elección celebrada, pendiente del escrutinio» y «al cierre de los sondeos».
2. Test de la casa excluida con `k >= min_polls` (el CIS de 2023 real) en el estudio y en la página (último en la
   matriz, fuera de la tabla de 2026).
3. `narrowed` y `below_one` comparan los cocientes redondeados (un 1,00 impreso nunca «baja de 1»).
4. Texto blanco en las celdas `.h2` de la tabla de la matriz, como el gráfico desde la opacidad 0,6.
5. El tramo neutro de la tabla incluye 1,20 (`h4`), como el gráfico; límites fijados en un test.
6. La frase de 2026 de la entradilla exige `k > 0` y concuerda en número («la única casa», «las n casas», «k de las n»).
7. Sin run, el KPI 3 dice «Casa que menos oscila» si la menor `h` del último ciclo es ≥ 1 («Casa más en rebaño» sólo
   por debajo).
8. Tooltip de la matriz «{casa} · {ciclo}: rebaño {h} ({k} sondeos, p {p})»: la isla lleva `[x, y, h, k, p]` y
   `cellText` recibe la celda entera.
9. Fuera de la plantilla los valores tecleados `or 4`/`or 10`: `n_series` y `local_window` salen de `herd.flags`.
10. `final_days` 21 (no 20) en este cierre.
11. Líneas de más de 120 caracteres y líneas en blanco (E302/E303/E305) de los ficheros de la rama.
12. `/analisis/rebano` entra en el barrido de redacción pública de todas las páginas.

### Lo que NO se ha verificado

- Un recorrido real en navegador por Luis (alturas, scroll de la matriz, tabla oculta, imagen de la tarjeta, negritas y
  colores descansan en la comprobación con Chrome sin cabeza del implementador).
- Datos reales publicados: todo se ha probado con los fixtures.
- S3 y ECS: ningún `job.py` se ha ejecutado de verdad.

### Lista de Luis

1. Publicar el estudio: `python job.py publish '{"what":["study"],"study":"rebano"}'` contra su S3 (o
   `'{"what":["analysis","studies"],"scopes":["es"]}'`).
2. Recorrer `/analisis/rebano` en escritorio y móvil (el gráfico por semanas, el scroll de la matriz, los botones de
   descarga de cada tarjeta, la línea de casas pendientes), `/analisis`, la banda de Inicio y `/metodo/rating`.
3. Desplegar la imagen para que el `update` diario de ECS republique el estudio (su grupo `publish` ahora termina en
   `studies`).
4. Tras el 29-N: `publish event` de `es/2026-11-29` (con resultados) y después `publish '{"what":["studies"]}'`, para
   que 2026 entre en los ciclos pasados y «Después del 29-N» tenga su fila.
