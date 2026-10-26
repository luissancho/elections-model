# Contrato de datos del paquete publicado (`site/v1`, contrato 1)

Descripción del paquete que escribe `python job.py publish` y que leen la API y el sitio (fase 2; el análisis
de casas, fase 4, enmienda 2026-10-10; las elecciones pasadas y el backtest del modelo, fase 5, enmienda 2026-10-11;
la veda, `freeze.since`, `HEAD` y el paquete `mtpy/lib/publish/`, fase 6, enmienda 2026-10-13).
La fuente de verdad en código es `mtpy/lib/bundle.py` (rutas, sobre, `SCHEMAS`, validación) y
`mtpy/lib/publish/` (paquete de exportadores). Cada JSON se valida contra su esquema antes de escribirse.

## Árbol del paquete

El paquete vive bajo `site/v1/` de `app.fs` (`files/site/v1/` en local, `s3://{bucket}/site/v1/` en
producción). Los ensayos (`dry_run`) escriben bajo `site-dry/v1/` con la misma estructura y sin
`manifest.json` ni `history.json`.

```
site/v1/
    manifest.json                               último run de cada ámbito
    parties.json                                catálogo de partidos de la base (enmienda 2026-10-19; ver "Catálogo de partidos")
    analysis/{scope}/                           análisis de casas (enmienda 2026-10-10; ver "Análisis")
        meta.json house-effects.json herding.json poll-errors.json error-data.json
        csv/ house-effects.csv herding.csv herding-events.csv poll-errors.csv error-data.csv
    backtest/{scope}/                           backtest del modelo (fase 5; ver "Backtest")
        meta.json metrics.json by-horizon.json blocks.json shares.json seats.json
        csv/ metrics.csv by-horizon.csv blocks.csv shares.csv seats.csv
    events/{scope}/{date}/                      una elección pasada (fase 5; ver "Eventos")
        meta.json pollsters.json polls.json results.json model.json
        csv/ pollsters.csv polls.csv results.csv model-shares.csv model-seats.csv model-blocks.csv
    runs/{scope}/history.json                   headline de cada run, ascendente por run_id
    runs/{scope}/{run_id}/
        meta.json headline.json series.json polls.json fan.json house-effects.json dispersion.json weights.json
        {nowcast,forecast}/
            vote.json summary.json dist.json districts.json scenario.json projection.json
        csv/
            series.csv polls.csv fan.csv house-effects.csv dispersion.csv weights.csv
            {nowcast,forecast}-{vote,summary,dist,districts,scenario,projection}.csv
```

## Sobre común

Todo JSON es un sobre:

```
{"schema": "<nombre>@1", "contract": 1, "scope": "es"|null, "run_id": "20261008-181100"|null,
 "mode": "nowcast"|"forecast"|null, "generated_at": "2026-10-08T18:13:01Z", "data": {...}}
```

- `schema` es `{nombre}@{contract}`; `contract` es 1.
- `scope` y `run_id` son `null` en los ficheros que no pertenecen a un run (`manifest`, `history`; en
  `history` sí se informa `scope`).
- `mode` solo se informa en los seis ficheros por modo (`vote`, `summary`, `dist`, `districts`,
  `scenario`, `projection`), que viven en `nowcast/` y `forecast/`.
- `data` tiene las claves de cada esquema (secciones siguientes). Se exige que estén todas y con el tipo
  correcto; puede haber claves adicionales dentro de las filas.

## Convenciones

- Redondeos: porcentajes (`pct`, `lo`, `hi`, `mean`, `sd`, ...) a 2 decimales; probabilidades (`p_*`,
  `weight`, y las razones de `dispersion`) a 3; escaños estadísticos (`seats_mean`, `seats_lo`, ...) a 1.
- `NaN` e infinitos se escriben como `null`.
- Fechas `YYYY-MM-DD`; instantes ISO 8601 en UTC con `Z` (`2026-10-08T18:11:00Z`).
- `run_id` es `YYYYMMDD-HHMMSS` en UTC, uno por invocación (compartido por todos los ámbitos); ordena
  lexicográficamente y cabe en un alias de ruta (`[0-9a-z_-]+`).
- Los runs son inmutables: un `run_id` existente no se reescribe. Los punteros (`manifest.json`,
  `history.json`) sí se reescriben. Un relleno retroactivo (`backfill`, solo días anteriores a hoy) salta
  los `run_id` existentes y solo mueve el `latest` del manifest hacia delante; con `replace` (enmienda
  2026-10-17) los borra antes como `unpublish` y los vuelve a escribir con el mismo `run_id`.
- `headline.json` se escribe el último. Un run sin `headline.json` está incompleto y se ignora (no entra
  en `history` ni se puede apuntar a él).
- Los partidos se identifican por `name`; el catálogo (`id`, `fullname`, `color`, `block`, `regional`) está
  en `meta.parties`.
- JSON compacto, UTF-8, sin `NaN` literal. Los CSV son UTF-8, fin de línea `\n`, sin índice.

## Ficheros del run

### `meta`

Cómo y cuándo se produjo el run. Fuente: `Simulator`, `Forecaster` y `provenance()`.

`data`: `run_id`, `run_at`, `commit` (`null` si no hay git ni `GIT_COMMIT`), `dirty`, `versions` (python y
`numpy`, `pandas`, `scipy`, `statsmodels`), `scope`, `event_date`, `as_of`, `date_last` (último sondeo
usado), `date_fit_last` (o `null`), `horizon_max`, `n_sim`, `seed`, `drange`, `max_fc`, `alpha`,
`correctors`, `n_polls`, `n_pollsters` (sondeos y casas usados), `db_polls`, `db_last_poll` (sondeos del
evento en la base y fecha del último), `n_seats`, `majority`, `parties` (lista de `{name, id, fullname,
color, block, regional}` en el orden de la simulación), `bmaps`, `smap`, `regions` (`{id, name, seats}`;
incluye el total del ámbito, `id` 0, que es la cámara entera),
`diagnostics` (`drift_k`, `multiplier`, `ages`, `composition`, `clip_rate` por modo y, desde la enmienda 2026-10-25,
`error_events`, las fechas ISO ascendentes de las elecciones cuyos sondeos ajustan el estimador de error, y
`error_scope`, el ámbito estimador, `es` también en las autonómicas), `seconds` (`init`,
`fit`, `nowcast`, `forecast`, `export`, `total`, a 1 decimal; `nowcast` y `forecast` miden solo las
simulaciones y `export` suma la escritura de ambos modos y de los ficheros del ciclo), `freeze` y
`limit_date`.

`freeze` aquí significa que el run se publicó dentro de la ventana LOREG con `force`; no es el interruptor
editorial de `manifest.freeze`.

`limit_date` (enmienda 2026-10-09; cadena `YYYY-MM-DD` o `null`, clave obligatoria) es el día de corte de un
run retrospectivo: solo entraron los sondeos publicados hasta ese día y `as_of` se leyó ese día. Es `null` en
un run normal. Un lector trata la clave ausente (runs publicados antes de 2026-10-09) como `null`.

`drange` es la ventana efectiva de sondeos del run, en días antes de `event_date` (`[desde, hasta]`, `null`
sin tope). En un run retrospectivo el corte sale de `limit_date`: el límite inferior sube a los días entre
`limit_date` y `event_date`, así que un run del 2026-10-05 para el evento del 2026-11-29 registra
`[55, null]`.

`db_polls` y `db_last_poll` cuentan la base en el momento de la publicación: en un run retrospectivo, la base
de hoy (el día de la reconstrucción), no la de `limit_date`; los sondeos que entraron en el run son
`n_polls`.

### `headline`

Resumen del run, lo primero que muestra el sitio y la fila de `history`.

`data`: `run_id`, `run_at`, `event_date`, `as_of`, `date_last`, `n_polls`, `backfill`, `nowcast` y `forecast`. Cada
modo es `{parties: [...], p_majority: {bloque: p}}` y cada partido `{name, pct, lo, hi, seats, seats_lo,
seats_hi, p_first}` en el orden de `vote_forecast()`; `seats` es el titular entero (`totals()`).

`backfill` (enmienda 2026-10-09; booleano, clave obligatoria) es `true` en un run retrospectivo: una
reconstrucción con el modelo de hoy y solo los sondeos publicados hasta `meta.limit_date`, con `run_id`
`YYYYMMDD-120000` y `run_at` a las 12:00 UTC de ese día. Un lector trata la clave ausente (runs publicados
antes de 2026-10-09) como `false`. El contrato sigue en 1: son claves adicionales.

### `series`

Serie diaria ajustada de cada partido con sus cotas, hasta `meta.as_of`, el día en que se lee el promedio
(enmienda 2026-10-26; antes hasta `date_fit_last`, que puede quedar hasta diez días más allá y sigue en `meta`).
Orientación columnar. Fuente: `Forecaster.forecast` y `Forecaster.fc_stat` (`cmin`, `cmax`).

`data`: `dates`, `parties`, y `mean`, `lo`, `hi` (cada uno `{partido: [valor por fecha]}`; `lo`/`hi` son
`null` donde no hay estadística). CSV `csv/series.csv`: `date, name, mean, lo, hi` (largo).

### `polls`

Sondeos publicados y resultado anterior. Fuente: `Forecaster.fc_series_raw` y `Forecaster.nfc_series`.

`data`: `parties`, `columns` (`POLL_COLUMNS`), `polls` y `results`.

- `polls`: registros con `date, pollster_id, pollster, sponsor, start_date, end_date, sample_size, mtype,
  rating, weight` (10 columnas) más una columna por partido en puntos porcentuales. `weight` a 3
  decimales. Enmienda 2026-10-14 (fase 7c): el sondeo que publica escaños lleva además `seats`,
  `{partido: [mín, máx]}` (enteros; la encuestadora da un número, `mín = máx`, o una horquilla), solo con los
  partidos que los tienen; los sondeos sin escaños no llevan la clave. Son las cifras de la encuestadora
  (`polls_results.seats_min/seats_max`), no del modelo. Los runs anteriores no las tienen.
- `results`: registros `date` + una columna por partido (resultado de elecciones anteriores).

CSV `csv/polls.csv`: la tabla `polls` más, por partido, `seats_min_{partido}` y `seats_max_{partido}` (enteros
vacíos sin proyección; enmienda 2026-10-14).

### `fan`

Abanico de voto por partido y horizonte. Fuente: `sim.fan()`.

`data`: `horizons` (días, enteros ordenados; 0 es el nowcast) y `rows` (formato largo: `name`, `horizon`,
`mean`, `sd`, `lo`, `hi` y, desde la enmienda 2026-10-25, las tres piezas del margen en puntos, `err` (error
estándar del promedio, `s`), `pct_err` (error histórico de los sondeos de la última semana, `e`) y `drift` (deriva
de la opinión en ese horizonte, `d`, 0 en el horizonte 0), con `sd² = err² + pct_err² + drift²`, y `dof` (grados de
libertad de la t, entero); el partido va en `name`, como en el resto de ficheros; los runs anteriores a la enmienda
no traen las columnas nuevas y /metodo/escanos oculta entonces su tarjeta de los tres errores). CSV `csv/fan.csv`: las
mismas filas y columnas, las seis históricas primero.

### `house-effects`

Efecto de casa de cada encuestadora. Fuente: `Forecaster.house_effects`; `rows` vacío si no se ajustó.

`data`: `rows` con las 13 columnas `HE_COLUMNS`: `pollster_id, name, pollster, n, w, level, dev, dev_err,
prior, prior_err, effect, effect_err, center` (todas menos las cinco primeras a 2 decimales). CSV
`csv/house-effects.csv`.

### `dispersion`

Dispersión y efecto rebaño por encuestadora. Fuente: `Forecaster.dispersion`; `rows` vacío si no se
calculó.

`data`: `rows` con las 9 columnas `DISP_COLUMNS`: `pollster_id, pollster, n, ss_obs, ss_exp, ratio_raw,
ratio, herd_ratio, factor` (todas menos las tres primeras a 3 decimales). CSV `csv/dispersion.csv`.

### `weights`

Los sondeos que pesan en el promedio el día en que el run lee el voto estimado, con la recta local de ese día.
Enmienda 2026-10-15 (fase 9): los runs anteriores no la tienen. Fuente: `Forecaster.weights_at(name, date)`
leído en `sim.as_of` (desde la enmienda 2026-10-26, el día del cálculo, nunca después: el mínimo de `date_fit_last`,
la víspera de la veda y hoy; antes, `date_fit_last`, diez días por delante del último sondeo; en uno retrospectivo,
su día),
con la rejilla y el error agrupado de `fit`, así que la recta leída ese día es el promedio publicado.

`data`: `date` (el día leído), `parties` (los partidos con ajuste), `rows`, `line`, `kernel` y `stats`.

- `rows`: formato largo, una fila por partido y sondeo de la regresión de ese día, con las 13 columnas
  `WEIGHTS_COLUMNS`: `name, date, pollster_id, pollster, sponsor, sample_size, pct, pct_adj, weight,
  weight_kernel, weight_over, weight_sample, weight_rating`. `pct` es el voto publicado y `pct_adj` el neto del
  efecto de casa (el que ajusta la recta); `weight` es `weight_kernel` por el peso del sondeo, que es el producto
  de `weight_over`, `weight_sample` (con la dispersión de la casa dentro) y `weight_rating`. Porcentajes a 1
  decimal, pesos a 3.
- `line`: `{partido: {dates, mean, lo, hi}}`, la recta local del día y su banda del 95 %, día a día en la
  ventana (desde el primero entre los sondeos con kernel apreciable y los de `rows`, hasta el día leído). Columnar,
  como `series`.
- `kernel`: `{partido: {dates, values}}`, la campana del kernel centrada en el día leído, en `[0, 1]` (3
  decimales), en los mismos días que `line`.
- `stats`: `{partido: {date, mean, lo, hi, err, nobs, neff, bandwidth}}`: el día leído para ese partido (solo
  difiere de `date` si su rejilla acaba antes), el promedio, la banda, su error, los sondeos de la regresión
  (`nobs`, entero), a cuántos equivalen por sus pesos (`neff`, 1 decimal) y la anchura del kernel ese día en días
  (`bandwidth`, 1 decimal).

CSV `csv/weights.csv`: la tabla `rows`. La recta y la campana no tienen gemelo CSV.

## Ficheros por modo

Cada uno vive en `nowcast/` y en `forecast/` (`mode` informado en el sobre) y tiene gemelo CSV
`csv/{mode}-{nombre}.csv`.

### `vote`

Voto previsto. Fuente: `sim.vote_forecast()`.

`data`: `horizon` (días), `when` (fecha del modo) y `rows` con `name, pct, sd, lo, hi`.

### `summary`

Escaños y probabilidades por partido, por bloques de voto (`vs`) y por bloques de escaños (`blocks`).
Fuente: `summary()`, `summary('vs')`, `summary('blocks')`, `probabilities('vs')` y `totals()`.

`data`: `n_seats`, `majority`, `parties`, `vs` y `blocks` (listas de registros), `p_majority`
(`{bloque: p}`, de `probabilities('vs')`) y `totals` (`{partido: escaños enteros}`).

Cada registro de `parties`, `vs` y `blocks` tiene 15 columnas: `name`, `pct`, `pct_mean`, `pct_lo`,
`pct_hi`, `seats`, `seats_mean`, `seats_median`, `seats_lo`, `seats_hi`, `seats_min`, `seats_max`,
`p_seats`, `p_majority`, `p_first`. Aquí `seats` es el titular entero (`totals()`), `null` en un bloque
sin partidos. CSV: las tres tablas apiladas con una columna inicial `group` (`parties`, `vs`, `blocks`).

### `dist`

Distribución de escaños por simulación (base de histogramas y coaliciones en el navegador). Fuente:
`sim.dist()`.

`data`: `n_seats`, `parties` y `seats` (matriz simulaciones x partidos, enteros). CSV: la matriz con una
columna por partido.

### `districts`

Votos y escaños por circunscripción, sin el total del ámbito. Fuente: `sim.unit_summary(r)` por región.

`data`: `parties`, `regions` (`{id, name, seats, community, votes}`, solo las circunscripciones: a diferencia de
`meta.regions`, no incluye el total del ámbito) y `rows`. `community` es `{code, name}` de la comunidad autónoma de la
circunscripción (`null` en los ámbitos autonómicos) y `votes` los votos de la elección anterior
(`sim.prev_totals`: los totales de la elección futura, `sim.reg_totals`, no traen votos). Enmienda 2026-10-14 (fase 7c): los runs anteriores no llevan `community` ni `votes`. Cada fila tiene 11 columnas: `region_id`,
`region`, `name`, `pct`, `pct_lo`, `pct_hi`, `seats`, `seats_mean`, `seats_lo`, `seats_hi`, `p_seats`.
Aquí `seats` es la mediana de los escaños simulados redondeada a 1 decimal (un estadístico, como
`seats_mean`), a diferencia del titular entero de `summary` y `headline`. CSV: las mismas filas.

### `scenario`

La simulación más cercana a los escaños centrales, coherente en todas las circunscripciones. No es
aleatoria: `sim.scenario()` elige, de forma determinista, la simulación con menor distancia L1 a la
mediana de escaños de cada partido (empates por L2 y después por el menor índice). Fuente:
`sim.result(sim.scenario())`.

`data`: `simulation` (índice de la simulación), `parties` y `rows` (`{region_id, region, seats: [escaños
por partido]}`). CSV: `region_id, region` y una columna por partido.

### `projection`

Proyección del voto en el tiempo. Fuente: `sim.projection()` (también con `vs` y `blocks`).

`data`: `dates` y `groups`, con las claves `parties`, `vs` y `blocks`; cada grupo es `{names, mean, lo,
hi}` y cada estadística `{nombre: [valor por fecha]}`. CSV largo: `date, group, name, mean, lo, hi`.

## Análisis

Enmienda 2026-10-10 (fase 4). El análisis de casas de un ámbito vive en `analysis/{scope}/` (junto a `runs/`, en
el mismo `site/v1/`, o en `site-dry/v1/` con `dry_run`) y lo escribe `python job.py publish
'{"what":["analysis"],...}'`. A diferencia de un run, **se sobrescribe** en cada publicación: no hay `run_id`
ni histórico, no entra en el manifest ni en `history.json`, y la lectura no depende del manifest. Cada
ámbito tiene su análisis (`es` y los autonómicos); un ámbito sin sondeos del siguiente evento queda `skipped`
y no lo tiene. La fuente de verdad en código es `bundle.ANALYSIS_PARTS`, `ANALYSIS_SCHEMAS` y `ANALYSIS_CSV`
(`path_analysis`, `path_analysis_csv`) y `publish.publish_analysis`.

Mismo sobre que el resto de ficheros, con `scope` informado y `run_id`, `mode` a `null`; `schema` es
`{nombre}@1` y las claves obligatorias de `data` están en `SCHEMAS`. Cinco partes:

| Parte | `schema` | `data` (claves obligatorias) |
|---|---|---|
| `meta` | `analysis-meta@1` | `scope`, `event_date`, `events` (fechas de las elecciones del ámbito), `n_pollsters`, `params` (parámetros del cálculo: `bmap`, `drange`, `n_last`, `main`, `blocks` y los sondeos del evento en la base, `db_polls`, y la fecha del último, `db_last_poll`), `commit`, `dirty`, `versions`, `seconds` |
| `house-effects` | `analysis-house-effects@1` | `parties`, `blocks`, `rows` |
| `herding` | `analysis-herding@1` | `rows`, `events` |
| `poll-errors` | `poll-errors@1` | `blocks`, `rows` |
| `error-data` | `error-data@1` | `rows` |

- `house-effects.rows`: una fila por casa y partido o bloque (`pollster`, `name`, `n_events`, `hist`, `trend`,
  `cycle`, `n_cycle`; fuente `Computer.print_house_effects()`). `parties` son los partidos `main` del evento y
  `blocks` los bloques `vs`; los efectos van redondeados a 2 decimales.
- `herding.rows`: una fila por casa (`pollster`, `herding`, `herding_last`, `herding_p`, `herding_n`,
  `herding_events`; fuente `Computer.herding_summary()`); `herding.events`: una fila por ciclo y casa
  (`event_date`, `pollster`, `n`, `ratio`, `p_value`; fuente `get_herding(scope)`). Razones y valores p a 3
  decimales.
- `poll-errors.rows`: error de cada sondeo (`get_polls_metric('error', bmap='vs', drange=(6, 42), n_last=1)`)
  con `event_date` ISO, `event`, `pollster`, `days`, `sample_size`, `weight`, `error_avg`, `bias_avg`,
  `error_blocks`, `bias_blocks` y `bias`; `blocks` son los bloques presentes en las filas.
- `error-data.rows`: los datos del estimador de error (`get_error_estimator_data()`): `date`, `pollster`,
  `party`, `regional`, `weeks`, `weight`, `pct`, `error`.

`meta.json` se escribe el último: un análisis sin él está incompleto y la API lo trata como no publicado
(404 `analysis not published`). Si una publicación falla a medias, quedan partes nuevas junto a partes
viejas y manda `meta.json`. Los CSV (`csv/{nombre}.csv`) son los gemelos de `house-effects`, `herding`,
`poll-errors` y `error-data`, más `herding-events.csv` (las filas de `herding.events`); `meta` no tiene
gemelo.

## Backtest

Enmienda 2026-10-11 (fase 5). El backtest del modelo de un ámbito vive en `backtest/{scope}/` (en `site/v1/`, o en
`site-dry/v1/` con `dry_run`) y lo escribe `python job.py publish '{"what":["backtest"],...}'`, que **convierte** las
tablas que deja el job `backtest` (antes `backtest/run_backtest.py`) en `backtest/results/` (`es`) o `backtest/results/{scope}/` (los demás);
no calcula nada ni toca la base. Como el análisis, **se sobrescribe** en cada publicación: sin `run_id`, fuera del
manifest y de `history.json`. Un ámbito sin `metrics.csv` queda `skipped` (`no backtest results for {scope}`). La fuente
de verdad en código es `bundle.BACKTEST_PARTS`, `BACKTEST_SCHEMAS` y `BACKTEST_CSV` (`path_backtest`,
`path_backtest_csv`) y `publish.publish_backtest`.

Mismo sobre que el resto (`scope` informado, `run_id` y `mode` a `null`, `schema` `{nombre}@1`). Seis partes:

| Parte | `schema` | `data` (claves obligatorias) |
|---|---|---|
| `meta` | `backtest-meta@1` | `scope`, `run_at`, `commit` (de la ejecución del backtest, `null` sin `run.json`), `params` (el resto de `run.json`: `n_sim`, `seed`, `max_fc`, correctores, `db_polls`, `db_last_poll`...), `events`, `horizons`, `n_cases`, `n_seats_cases`, `source` (carpeta de origen: relativa al repositorio si está dentro de él, absoluta si no), `published` (`commit`, `dirty`, `versions` de la publicación), `seconds` |
| `metrics` | `backtest-metrics@1` | `columns`, `rows` |
| `by-horizon` | `backtest-by-horizon@1` | `columns`, `rows` |
| `blocks` | `backtest-blocks@1` | `columns`, `rows` |
| `shares` | `backtest-shares@1` | `columns`, `rows` |
| `seats` | `backtest-seats@1` | `columns`, `rows` |

- Las cinco tablas son `metrics.csv` (una fila por elección y horizonte), `by_horizon.csv` (agregado por horizonte),
  `blocks.csv`, `shares.csv` y `seats.csv` de `backtest/results`; `columns` conserva el orden de las columnas del CSV
  y `rows` son registros `{columna: valor}`. El significado de cada métrica está en `backtest/README.md`.
- Redondeos: una columna decimal cuyos valores son todos enteros (p. ej. `official`) y cuyo nombre no empieza por
  `cov_`, `brier`, `log_score` o `p_majority` se publica como entero nullable (`int` en el JSON, sin `.0` en el CSV);
  el resto de decimales va a 2 decimales, salvo las que empiezan por esos prefijos, a 3. `seats_valid`, `majority` y
  `main` van como booleano y `NaN` como `null`.
- `meta.events` y `meta.horizons` salen de `run.json` (sin él, de los valores únicos de `metrics`);
  `n_cases = len(metrics)` y `n_seats_cases` son los casos con escaños válidos (`seats_valid`).

`meta.json` se escribe el último: un backtest sin él está incompleto. Los CSV (`csv/{nombre}.csv`) son los gemelos
redondeados de las cinco tablas; `meta` no tiene gemelo.

## Eventos

Enmienda 2026-10-11 (fase 5). Una elección pasada de un ámbito vive en `events/{scope}/{date}/` (`date` = `YYYY-MM-DD`;
en `site/v1/`, o en `site-dry/v1/` con `dry_run`) y la escribe `python job.py publish '{"what":["event"],"scopes":["es"],"event_date":...}'`,
donde `event_date` es una fecha, una lista de fechas o `"all"` (las elecciones `featured` del ámbito anteriores a hoy).
Como el análisis y el backtest, **se sobrescribe** en cada publicación: sin `run_id`, fuera del manifest y de
`history.json`, sin cortafuegos LOREG; `event` no se combina con `forecast`. Cada fecha queda `published`, `skipped`
(`{date} is not a featured election of {scope}`, `no official results for {scope} {date}` o sin sondeos) o `failed`.
La fuente de verdad en código es `bundle.EVENT_PARTS`, `EVENT_SCHEMAS`, `EVENT_CSV` (`path_event`, `path_event_csv`,
`BundleReader.list_events`, `event_exists`) y `publish.publish_event`.

Mismo sobre que el resto (`scope` informado, `run_id` y `mode` a `null`, `schema` `{nombre}@1`). Seis partes
(`cycle` desde la enmienda 2026-10-14; una elección publicada antes no la tiene y la página lo avisa hasta que se
republica):

| Parte | `schema` | `data` (claves obligatorias) |
|---|---|---|
| `meta` | `event-meta@1` | `scope`, `event_date`, `name` (o `null`), `n_polls`, `n_pollsters`, `blocks` (partidos de cada bloque `vs`), `main`, `summary`, `has_model`, `horizons`, `commit`, `dirty`, `versions`, `seconds` |
| `pollsters` | `event-pollsters@1` | `blocks`, `rows` |
| `polls` | `event-polls@1` | `blocks`, `rows` |
| `cycle` | `event-cycle@1` | `parties`, `columns`, `polls` |
| `results` | `event-results@1` | `n_seats`, `rows` |
| `model` | `event-model@1` | `horizons`, `shares`, `seats`, `blocks` |
| `series` | `event-series@1` | `horizon`, `dates`, `parties`, `mean`, `lo`, `hi` |

- `meta.summary`: `n_polls`, `n_pollsters`, `error_avg`, `error_blocks`, `bias_blocks` (medias de los sondeos ponderadas
  por `weight`, 2 decimales) y `bias` = `{bloque: media ponderada de su desviación con signo}` (positivo: los
  sondeos dieron de más a ese bloque). `bias_blocks` no tiene signo: es una magnitud en la escala del error. `has_model` es
  `true` si `horizons` no está vacío; `horizons`, los horizontes del modelo.
- `pollsters.rows`: una fila por casa, de menor a mayor `error_blocks` (nulos al final): `pollster_id` (o `null`),
  `pollster`, `n_polls`, `sample_size` (mediana de las muestras conocidas, `null` si ninguna), `error_avg`,
  `error_blocks`, `bias_blocks` (medias, 2 decimales) y los `last_*` de su sondeo más cercano a la elección
  (`last_days`, `last_error_avg`, `last_error_blocks`, `last_bias_blocks` y `last_{bloque}`, la desviación con signo
  de cada bloque; `blocks` los lista).
- `polls.rows`: las filas de `poll-errors` (ver "Análisis") de todos los sondeos de la elección más `pollster_id`
  (tras `pollster`), por `days` ascendente (el más reciente primero); `blocks` lista las columnas de bloque.
- `cycle`: los sondeos del ciclo tal como se publicaron, con la forma del `polls` de un run (`parties`, `columns` =
  `POLL_COLUMNS` y `polls`, cada uno con `seats` `{partido: [mín, máx]}` cuando la casa publicó escaños; sin
  `results`), para que la tabla "Todas las encuestas" del detalle sea la misma que la de Promedios. Son los mismos
  sondeos que `polls.rows` (`Computer.filter_polls(featured=True)`: cada fila de la tabla tiene su error), por
  fecha y casa ascendentes, con `sample_size` nulo cuando no se conoce y `weight` el producto del peso global y el
  de muestra; los escaños salen de `forecaster.published_seats`. El gemelo CSV añade `seats_min_*`/`seats_max_*`.
- `results.rows`: `party_id`, `party`, `votes` (entero), `pct` (2 decimales) y `seats` (entero) del resultado oficial
  nacional (región 0), por votos descendente; `n_seats` es la suma de `seats`.
- `model` es el **corte del backtest** (ver "Backtest") a esta elección: `shares`, `seats` y `blocks` son las filas
  (`{columna: valor}`, mismos redondeos) de `shares.csv`, `seats.csv` y `blocks.csv` con ese `event_date`, y
  `horizons` los horizontes que aparecen, ascendentes. Sin backtest, las tres listas y `horizons` van vacías.
- `series` es la serie diaria del promedio del modelo del horizonte más corto del backtest (`series.csv` de
  `backtest/results`, normalmente a 6 días), hasta el día en que se congeló el modelo: `horizon` y, con la forma de la
  parte `series` de un run, `dates`, `parties`, `mean`, `lo` y `hi` (2 decimales, `null` en los huecos de la banda).
  Solo existe si el backtest guardó la serie: una elección puede tener `model` y no `series` (la publicada antes de
  2026-10-19, o con un backtest anterior), y entonces su detalle avisa en lugar de dibujar la curva.

`meta.json` se escribe el último: una elección sin él está incompleta. Los CSV (`csv/{nombre}.csv`) son los gemelos de
`pollsters`, `polls`, `cycle` y `results`, los tres cortes del modelo (`model-shares`, `model-seats`, `model-blocks`, solo los
que tienen tabla en el backtest; al republicar se borran los gemelos `model-*` de las tablas que ya no están) y
`series` (formato largo `date`, `name`, `mean`, `lo`, `hi`; al republicar sin serie se borran `series.json` y
`series.csv`); `meta` no tiene gemelo.

## Estudios

Enmienda 2026-10-19 (estudio del voto CERA). Un estudio vive en `studies/{study}/` (en `site/v1/`, o en `site-dry/v1/` con
`dry_run`) y lo escribe el publicador de estudios (`publish.publish_study`). Hoy hay dos: `voto-cera` y `sesgo-cis`.
Como el análisis, **se sobrescribe** en cada publicación: sin `run_id` y fuera de `history.json`. Su única huella en
el manifest es su entrada en `manifest.studies` (ver «Punteros»).

```
studies/{study}/{parte}.json
studies/{study}/csv/{nombre}.csv
```

La fuente de verdad en código es `bundle.STUDIES`, `STUDY_PARTS`, `STUDY_SCHEMAS`, `STUDY_CSV`, `STUDY_CSV_OF`
(`path_study`, `path_study_csv`, `BundleReader.study_exists`) y `publish.publish_study`. El registro es **por estudio**:
`STUDY_PARTS[estudio]`, `STUDY_SCHEMAS[estudio][parte]`, `STUDY_CSV[estudio]` y `STUDY_CSV_OF[estudio][parte]`; en la
publicación, `publish.STUDY_RUNNERS` (cómo se calcula y si se puede registrar), `STUDY_EXPORTERS` y `STUDY_META`.

Mismo sobre que el resto, con `scope` y `run_id` a `null` (y `mode` a `null`). Las partes de `voto-cera` llevan `schema`
`study-{parte}@1`; las de `sesgo-cis`, `sesgo-cis-{parte}@1`.

### `voto-cera`

Ocho partes:

| Parte | `schema` | `data` (claves obligatorias) |
|---|---|---|
| `meta` | `study-meta@1` | `study`, `title`, `subtitle`, `election` (dict), `registered_at`, `registered_commit`, `registered` (dict o `null`), `registered_changed_at`, `checked_at`, `params` (dict), `sources` (lista), `series_last`, `regression` (dict), `commit`, `dirty`, `versions` (dict), `seconds` (dict) |
| `census` | `study-census@1` | `series` (dict), `marks`, `periods` (dict), `provinces`, `countries`, `origins`, `exposure`, `election_provinces`, `election_countries` |
| `participation` | `study-participation@1` | `origin`, `generales`, `autonomicas` |
| `vote` | `study-vote@1` | `processes`, `ratios`, `ecological` (dict) |
| `placement` | `study-placement@1` | `sensitive` (dict), `acceleration`, `correlations` (dict), `madrid` |
| `prediction` | `study-prediction@1` | `scenarios` (dict), `last_seat`, `report` (dict) |
| `sensitivity` | `study-sensitivity@1` | `grid` (dict), `scenarios`, `placement` (dict) |
| `check` | `study-check@1` | `elections`, `rows`, `verdict` (texto o `null`) |

Los valores sin tipo son listas. `registered_at`, `registered_commit`, `registered_changed_at`, `checked_at` y `commit`
son texto o `null`.

Gemelos CSV (`studies/{study}/csv/{nombre}.csv`); el primero de cada parte es el CSV por defecto:

| Parte | CSV |
|---|---|
| `census` | `census-series`, `census-provinces`, `census-countries` |
| `participation` | `participation` |
| `vote` | `vote`, `ecological` |
| `prediction` | `last-seat` |
| `sensitivity` | `sensitivity-grid`, `sensitivity-scenarios`, `placement-grid` |
| `check` | `check` |

`meta`, `placement` no tienen gemelo.

### `sesgo-cis`

«El sesgo del CIS de Tezanos». Seis partes, sin `slug`; se calcula sólo con las partes ya publicadas del paquete
(análisis y elecciones) y no se registra:

| Parte | `schema` | `data` (claves obligatorias) |
|---|---|---|
| `meta` | `sesgo-cis-meta@1` | `study`, `title`, `subtitle`, `pollster`, `periods` (dict), `blocks` (lista), `params` (dict), `inputs` (dict), `sources` (lista de `{label, url}`), `commit` (texto o `null`), `dirty`, `versions` (dict), `seconds` (dict) |
| `summary` | `sesgo-cis-summary@1` | `with` (dict), `without` (dict), `election` (dict) |
| `elections` | `sesgo-cis-elections@1` | `rows` (lista) |
| `houses` | `sesgo-cis-houses@1` | `rows` (lista) |
| `matrix` | `sesgo-cis-matrix@1` | `events` (lista), `houses` (lista), `cells` (lista de `[elección, casa, error]`) |
| `election` | `sesgo-cis-election@1` | `scope`, `date`, `name` (texto o `null`), `blocks` (dict), `max_days` (entero), `result`, `cis`, `consensus` (dicts), `houses` (lista) |

`summary.with` y `summary.without` no llevan las filas internas del periodo (son la parte `elections`). Los subtítulos
se formatean al publicar con las elecciones con sesgo a la izquierda y el total del periodo actual.

Gemelos CSV:

| Parte | CSV |
|---|---|
| `elections` | `elections` |
| `houses` | `houses` |
| `matrix` | `matrix` (largo: `pollster`, `scope`, `event_date`, `Izquierda`) |
| `election` | `election-houses` |

`meta`, `summary` no tienen gemelo.

### Escritura

`meta.json` se escribe **el último**: un estudio sin él cuenta como «no
publicado» (`study_exists` es falso y la API da 404 `study not published`), de modo que una publicación a medias
nunca se ve.

## Catálogo de partidos

Enmienda 2026-10-19. `parties.json` vive junto al manifest (en `site/v1/`, o en `site-dry/v1/` con `dry_run`) y lo
escribe `publish.publish_parties` (acción `parties` del job `publish`; `forecast` y `analysis` lo reescriben también,
una vez por ejecución). Como el análisis, **se sobrescribe** en cada publicación: sin `scope`, sin `run_id` y fuera
de `history.json` y del manifest. Es la tabla `elections.parties` completa, en su orden.

Mismo sobre que el resto, con `scope`, `run_id` y `mode` a `null` y `schema` `parties@1`. `data` tiene una clave:

| Clave | Contenido |
|---|---|
| `rows` | Una fila por partido: `id` (entero), `name`, `fullname` (texto o `null`), `parent_id` (entero o `null`), `color` (texto o `null`), `block` (texto o `null`) y `regional` (booleano o `null`) |

Las páginas lo leen (`webapi.Site.parties`, caché TTL como el análisis) para completar el catálogo del run
(`meta.parties`): un partido que el run no simula (uno de una elección pasada, o una columna de sondeo que se pliega
en otro partido) toma de aquí su color, su nombre completo y su bloque; sin el fichero, esos partidos salen con el
gris de «otros», como antes de la enmienda. No tiene ruta en la API.

## Punteros

### `history.json`

Evolución del pronóstico publicado. Los runs reales son instantáneas no regenerables; desde 2026-10-09
también puede haber runs retrospectivos (`backfill: true`, reconstrucciones marcadas). Se reconstruye desde los `headline.json` de los runs
completos del ámbito tras cada publicación y tras `unpublish`.

`data`: `scope` y `runs` (lista de `headline`, ascendente por `run_id`; cada fila lleva `backfill`). Los
retrospectivos, con `run_id` a las 12:00 UTC de un día pasado, quedan antes de los reales de ese día
posteriores a esa hora y, por tanto, normalmente al principio de la lista.

### `manifest.json`

Único puntero del paquete; se reescribe al final de cada publicación. Si no existe, se parte de uno vacío.

`data`:

- `contract` (1) y `updated_at` (instante de la última escritura).
- `scopes`: `{ámbito: entrada}`, con la entrada `{latest, run_at, event_date, as_of, date_last, n_polls}`
  (el `run_id` publicado y los datos de su `headline`). Un ámbito fallido conserva su entrada anterior.
- `studies`: `{estudio: {published_at, registered_at, checked}}` (instante de publicación, instante del registro
  previo del estudio y si pasó la comprobación); `{}` si no hay ninguno, y un manifest anterior sin la clave se lee
  con `{}`.
- `freeze`: `{active, message, since}`; interruptor editorial que fija el job con `what: ["manifest"]`. `since`
  es la fecha ISO (Europe/Madrid) en que pasó de inactivo a activo, se conserva mientras siga activo y es `null`
  al desactivarlo; un manifest anterior sin la clave se lee con `since: null`.
- `attribution`: textos de atribución (`polls`, `results`, `model`). Provisionales; cada escritura del
  manifest la refresca desde `ATTRIBUTION`.

## Gemelos CSV

Cada JSON de datos tiene un CSV en `csv/`: `csv/{nombre}.csv` para los ficheros del run y
`csv/{modo}-{nombre}.csv` para los de modo. `meta`, `headline`, `history` y `manifest` no tienen gemelo.
Las columnas se describen en la sección de cada fichero.

## Rutas de la API (`/api/v1`, fase 2)

Solo `GET`. La API lee `manifest.json` para resolver el último run de un ámbito y sirve los ficheros del
paquete tal cual (los bytes escritos por `publish`, sin volver a serializar). La fuente de verdad en
código es `ROUTES` en `mtpy/lib/webapi.py` y los controladores `mtpy/controllers/Base.py` y
`Forecast.py`. (Enmienda 2026-10-11, fase 5: `webapi` es ahora el paquete `mtpy/lib/webapi/`; `ROUTES` vive en su
`__init__.py`, y las rutas de las fases 4 y 5 tienen sus controladores `Data`, `Analysis`, `Events` y `Backtest`.)
(Enmienda 2026-10-13, fase 6: toda ruta `GET` acepta también `HEAD`, que responde con las mismas cabeceras,
`content-length` incluido, y sin cuerpo; "Solo `GET`" en este documento quiere decir `GET` y `HEAD`.)

| Ruta | Fichero servido | `Cache-Control` |
|---|---|---|
| `/api/v1/health` | `{"status":"ok","contract":1}`, sin tocar el paquete | `no-store` |
| `/api/v1/manifest` | `manifest.json` | `public, max-age=60` |
| `/api/v1/scopes` | catálogo `data/es-scopes.csv` cruzado con el manifest (ver abajo) | `public, max-age=60` |
| `/api/v1/forecast/{scope}` | `meta.json` del run (atajo de `/forecast/{scope}/meta`) | 60 s, o inmutable con `?run=` |
| `/api/v1/forecast/{scope}/runs` | `runs/{scope}/history.json` | `public, max-age=60` |
| `/api/v1/forecast/{scope}/{part}` | `part` ∈ `meta`, `headline`, `series`, `polls`, `fan`, `house-effects`, `dispersion`, `weights` (enmienda 2026-10-15) | 60 s, o inmutable con `?run=` |
| `/api/v1/forecast/{scope}/{mode}/{part}` | `mode` ∈ `nowcast`, `forecast`; `part` ∈ `vote`, `summary`, `dist`, `districts`, `scenario`, `projection` | 60 s, o inmutable con `?run=` |

Enmienda 2026-10-10 (fase 4): diez rutas más, de dos controladores. `Data` lee las tablas de la base de
datos (`/parties`, `/pollsters`, `/pollsters/{id}`, `/ratings`, `/polls`, `/results/{scope}/{date}`,
`/house-effects`, `/herding`, `/drift`) y `Analysis` sirve el análisis publicado (`/analysis/{scope}/{part}`);
ver "Rutas de base y de análisis". Se registran antes que las de `forecast`.

Inmutable es `public, max-age=31536000, immutable`: solo cuando el run se pide de forma explícita con
`?run=`, porque un run publicado no se reescribe. Sin `?run=` se sirve el último run del manifest y la
respuesta caduca a los 60 s (`web.cache_ttl`). Las rutas fijas (`health`, `manifest`, `scopes`, `runs`) se
registran antes que las genéricas.

Las rutas de páginas no forman parte del contrato de la API: las renderizan los controladores de
`mtpy/controllers/` con plantillas Jinja2, y el contrato solo cubre `/api/v1` (enmienda 2026-10-13, fase 7: mapa por
eventos). Hay dos familias. Las **de evento** viven bajo el código del ámbito: `/{scope}` (Promedios, controlador
`Promedios`) y `/{scope}/escanos` (Escaños, `Escanos`), con `mode` y `run` (sin selector de modo desde la fase 7b.
Desde el 2026-10-25 el modo por defecto lo decide sólo el horizonte del run mostrado: Elección cuando está a 60 días o
menos de su elección, `event_date − as_of` del `meta` o de la entrada del manifest, `pages.run_horizon` y
`pages.FORECAST_HORIZON`, porque convocada la elección la pregunta es el día de la votación, y Hoy más lejos; la veda
congela la web a 6 días, ya dentro del plazo, y la congelación sólo decide la palabra «definitivo» y las fechas de los
títulos, no el modo, así que una autonómica lejana sigue en Hoy durante la veda de las generales; las tarjetas de Inicio
aplican la misma regla a su propio run; `mode` sigue aceptado en la query para enlaces y pruebas); un código que no es
del catálogo da 404 ("Ámbito desconocido.") y uno del catálogo sin pronóstico, el 404 "Este ámbito no tiene
pronóstico publicado.".
`/{scope}/escanos` acepta además `region` (solo de la página, no de la API): un entero de `districts.regions`; mal
formado da 400 y desconocido 404 en HTML. Las **globales** son `/` (Inicio, `Index`: el índice de eventos), `/elecciones`
(Histórico Errores, `Elecciones`), `/casas` (Ranking Encuestadoras, `Casas`) y `/metodo` (`Metodo`) con sus
cuatro subpáginas `/metodo/promedio`, `/metodo/escanos`, `/metodo/rating` y `/metodo/backtest`; una subpágina
desconocida da 404 ("Página del método no encontrada."). `/elecciones`, `/casas` y `/metodo/backtest` aceptan `scope`
(`es` por defecto). `/casas` acepta `id` (solo de la página): sin él, el
ranking del ámbito; con él, el perfil de la casa. `id` mal formado da 400 ("Casa no válida.") y desconocido 404 ("Casa
no encontrada."), ambos en HTML; sin base de datos, `/casas` da 503 ("La base de datos no está disponible."); sin
análisis publicado, el ranking se sirve y las secciones que dependen de él muestran "Análisis no publicado todavía".
`/elecciones` lee solo el paquete (`events/{scope}/` y `analysis/{scope}/poll-errors`, sin base) y acepta `date` (solo de
la página): sin él, el histórico de las elecciones publicadas; con él, el detalle de una. `date` mal formada da 400
("Fecha no válida.") y una elección no publicada 404 ("Elección no publicada."), ambos en HTML. `/metodo/backtest` lee
`backtest/{scope}/` y, sin backtest publicado, responde 200 con un aviso; `/metodo` y las otras tres subpáginas son
texto con las cifras vivas de las generales. Las URL anteriores a la fase 7 redirigen con un 301 (`cache-control:
no-store`, sin cuerpo) conservando el resto de la query: `/?scope=X` a `/X`, `/promedio` a `/X` (`/es` sin `scope`),
`/escanos` a `/X/escanos` y `/modelo` a `/metodo/backtest` (sin `mode` ni `run`). Los fragmentos (`#coalition=PP,VOX`
de la calculadora, los filtros del histórico y los controles del promedio) son solo del navegador: no llegan al
servidor ni crean entradas en la caché. Una ruta desconocida fuera de `/api/` devuelve un 404 en HTML; dentro de
`/api/`, el 404 JSON de la API. Los recursos de `/dist/` los sirve nginx.

Las páginas del sitio las resuelve el controlador en cada petición: toma `?run=` de la URL o, si no
está, el `latest` del ámbito en el manifest, y lee todas las partes con ese run (salvo `runs`, que sigue
siendo un puntero), de modo que una página nunca mezcla partes de dos runs aunque se publique uno nuevo
entre peticiones.

### Parámetros

- `run`: `YYYYMMDD-HHMMSS` (UTC). Vacío o ausente equivale al último run del ámbito.
- `format`: `json` (por defecto) o `csv`. Solo hay gemelo CSV para `series`, `polls`, `fan`,
  `house-effects`, `dispersion` y `weights`, y para las seis partes de cada modo; `meta` y `headline` no lo tienen.
  El CSV se sirve como `text/csv` con `Content-Disposition: attachment; filename="..."`, con nombre
  `{scope}-{run}-{part}.csv` o `{scope}-{run}-{mode}-{part}.csv`.

### `/scopes`

Un objeto `{"contract": 1, "scopes": [...]}` con una fila por ámbito del catálogo, en su orden:
`code`, `name`, `parent` (`null` si no tiene), `seats`, `simulable`, `latest` y los datos de la entrada del
manifest `run_at`, `event_date`, `as_of`, `date_last`, `n_polls`. `simulable` es verdadero si el ámbito
tiene entrada en el manifest; en caso contrario `latest` y los campos del run son `null`.

### Errores

Cuerpo JSON `{"status":"error","message":"..."}` con `Cache-Control: no-store`.

| Estado | `message` | Causa |
|---|---|---|
| 400 | `invalid scope` | ámbito mal formado o fuera del catálogo |
| 400 | `invalid run` | `run` no cumple `YYYYMMDD-HHMMSS` |
| 400 | `invalid mode` | `mode` no es `nowcast` ni `forecast` |
| 400 | `invalid format` | `format` no es `json` ni `csv` |
| 404 | `unknown part` | `part` fuera de la lista de la ruta |
| 404 | `scope not published` | el ámbito no tiene entrada en el manifest (sin `?run=`) |
| 404 | `no runs published` | el ámbito no tiene `history.json` |
| 404 | `run not found` | el fichero del run no existe |
| 404 | `no csv for this part` | `format=csv` en una parte sin gemelo CSV (`meta`, `headline`) |
| 503 | `no bundle published yet` | no existe `manifest.json` |
| 503 | `no file system configured` | la aplicación no tiene sistema de ficheros |
| 503 | `no data directory configured` | la aplicación no tiene directorio `data/` (lo lee el catálogo de ámbitos; lo puede dar cualquier ruta con `{scope}`) |

Las rutas desconocidas y los métodos distintos de GET reciben el 404 JSON genérico del router (`404 Not Found`).
(Enmienda 2026-10-13: `HEAD` se trata como `GET` sin cuerpo; `POST` y el resto siguen dando ese 404.)

### Cabeceras

- `ETag`: md5 de los bytes servidos, en todo contenido servido salvo `/health` (también `/scopes`).
- `Access-Control-Allow-Origin: *` en todas las respuestas de los controladores de la API, errores incluidos
  (solo `GET`, sin preflight). La pone el controlador, no el router: el 404 genérico del router (ruta
  desconocida o método distinto de GET) no la lleva.
- `x-freeze: active` cuando `manifest.freeze.active` está activo (en `scopes`, `runs`, `forecast/...`).
  Es solo un aviso: la API sigue sirviendo los datos. Las páginas lo muestran con un banner (`#freeze-banner`,
  enlazado a `/metodo#limitaciones`), el título «Pronóstico definitivo» en modo `forecast` y «congelado desde»
  en el pie.

## Rutas de base y de análisis (fase 4)

Enmienda 2026-10-10. Solo `GET`. Las nueve primeras leen la base de datos en vivo (controlador `Data`); la
última sirve el análisis publicado (controlador `Analysis`, `site/v1/analysis/`). Respuestas con
`Cache-Control: public, max-age=60` y `ETag`; nunca inmutables.

| Ruta | Parámetros | Respuesta JSON | CSV (`?format=csv`) |
|---|---|---|---|
| `/api/v1/parties` | | `{contract, rows}` | `parties.csv` |
| `/api/v1/pollsters` | | `{contract, rows}` | `pollsters.csv` |
| `/api/v1/pollsters/{id}` | `scope` (por defecto `es`) | `{contract, scope, pollster, ratings, house_effects, herding, quality}` | no (404 `no csv for this part`) |
| `/api/v1/ratings` | `scope`, `event` (fecha; por defecto el siguiente evento) | `{contract, scope, event, rows}` | `ratings-{scope}[-{event}].csv` |
| `/api/v1/polls` | `scope`, `event`, `pollster` (id), `limit` (1000 por defecto, máximo 5000) | `{contract, scope, event, pollster, limit, rows}` (un sondeo por fila, una columna por partido, los más recientes primero; sin los sondeos de contexto `wban` —publicados durante la veda— ni `exit`) | `polls-{scope}[-{event}][-{pollster}].csv` |
| `/api/v1/results/{scope}/{date}` | `region` (0, el ámbito entero, por defecto) | `{contract, scope, date, region, rows}` | `results-{scope}-{date}[-{region}].csv` |
| `/api/v1/house-effects` | `scope` | `{contract, scope, rows}` | `house-effects-{scope}.csv` |
| `/api/v1/herding` | `scope` | `{contract, scope, rows}` | `herding-{scope}.csv` |
| `/api/v1/drift` | `scope` | `{contract, scope, rows}` | `drift-{scope}.csv` |
| `/api/v1/analysis/{scope}/{part}` | `part` ∈ `meta`, `house-effects`, `herding`, `poll-errors`, `error-data`; `table=events` (solo CSV de `herding`) | el fichero del análisis tal cual (sobre completo) | `{scope}-{nombre}.csv`, con `nombre` ∈ `house-effects`, `herding`, `herding-events`, `poll-errors`, `error-data` |

- `rows` son registros JSON de la tabla de la base (`NaN` a `null`, fechas ISO). `scope` se valida contra el
  catálogo; `event` y `{date}` han de ser una fecha real `YYYY-MM-DD` y además una elección del ámbito
  (`event not found`): `event`, una de sus ratings o la próxima elección; `{date}`, una de sus elecciones
  (`events`), comprobada antes de leer los resultados. Los lectores corren sobre una caché de 60 s por ámbito, con un
  tope de 128 entradas (LRU); los filtros (`event`, `pollster`, `limit`, `region`) se aplican fuera de la caché:
  `results` se lee una vez por elección, con todas sus regiones.
- `ratings` y `polls` devuelven en `event` la fecha resuelta: la pedida o, sin `event`, la de la próxima elección
  (`null` si el ámbito no tiene ninguna).
- `ratings` y `pollster` devuelven `rating` y `quality` en escala 0-100.
- `pollster.quality` es el registro de `elections.pollsters_quality` de la casa: los cinco criterios de la rúbrica (`fieldwork`, `transparency`, `independence`, `identity`, `standards`) con su fuente (`*_source`), `notes`, `status` (`borrador`/`revisada`), `checked_at` (ISO), `points` y `quality`; `null` si la casa no está evaluada (su `quality` del catálogo es entonces el mínimo, 5). Ver `data/README.md`, «Calidad previa de las encuestadoras».
- `format=csv` se sirve como `text/csv` con `Content-Disposition: attachment`, igual que en `forecast`.
  `meta` y `pollsters/{id}` no tienen gemelo CSV (404 `no csv for this part`); `table=events` solo vale con
  `part=herding` y `format=csv`, y cualquier otro valor de `table` da el mismo 404.
- `analysis` lee `analysis/{scope}/{part}.json` (o `csv/{nombre}.csv`) del paquete a través de la misma
  caché de punteros (TTL 60 s), porque se sobrescribe: sin `meta.json` o sin el fichero, 404 `analysis not
  published`. No usa la base ni `manifest.json`.
- Sin base configurada (sin `app.db` o con `DB_HOST` vacío), las nueve rutas de base dan 503 `no database
  configured` antes de leer nada; un error al
  consultar la base da 503 `database unavailable` (con la traza en el log, no en la respuesta).

Errores nuevos (mismo cuerpo `{"status":"error","message":"..."}` y `Cache-Control: no-store`):

| Estado | `message` | Causa |
|---|---|---|
| 400 | `invalid date` | `event` o `{date}` no es una fecha real `YYYY-MM-DD` |
| 400 | `invalid pollster` | `pollster` o `{id}` no son de uno a seis dígitos |
| 400 | `invalid limit` | `limit` no es un entero entre 1 y 5000 |
| 400 | `invalid region` | `region` no son de uno a seis dígitos (ya existía para `/escanos`) |
| 404 | `pollster not found` | `{id}` no está en el catálogo de casas |
| 404 | `event not found` | `event` o `{date}` no es una elección del ámbito |
| 404 | `analysis not published` | el ámbito no tiene análisis completo (falta el fichero o `meta.json`) |
| 404 | `no csv for this part` | `format=csv` en `meta` del análisis, en `pollsters/{id}` o con un `table` no válido |
| 503 | `no database configured` | la aplicación no tiene base de datos o su `DB_HOST` está vacío |
| 503 | `database unavailable` | la consulta a la base falló |

`invalid scope`, `invalid format`, `unknown part` y `no file system configured` valen también aquí.

## Rutas de eventos y backtest (fase 5)

Solo `GET`. `events/{scope}` cruza la base de datos con el paquete; las otras dos sirven ficheros publicados
(controladores `Events` y `Backtest`, `site/v1/events/` y `site/v1/backtest/`). Respuestas con
`Cache-Control: public, max-age=60` y `ETag`; nunca inmutables, porque se sobrescriben en cada publicación.

| Ruta | Parámetros | Respuesta JSON | CSV (`?format=csv`) |
|---|---|---|---|
| `/api/v1/events/{scope}` | | `{contract, scope, rows}`; cada fila `{date, name, featured, has_page}`, por fecha ascendente | `events-{scope}.csv` |
| `/api/v1/events/{scope}/{date}/{part}` | `part` ∈ `meta`, `pollsters`, `polls`, `cycle`, `results`, `model`, `series`; `table` ∈ `shares`, `seats`, `blocks` (obligatoria en el CSV de `model`) | el fichero de la elección tal cual (sobre completo) | `{scope}-{date}-{nombre}.csv`, con `nombre` ∈ `pollsters`, `polls`, `cycle`, `results`, `model-shares`, `model-seats`, `model-blocks`, `series` |
| `/api/v1/backtest/{scope}/{part}` | `part` ∈ `meta`, `metrics`, `by-horizon`, `blocks`, `shares`, `seats` | el fichero del backtest tal cual (sobre completo) | `{scope}-backtest-{nombre}.csv`, con `nombre` ∈ `metrics`, `by-horizon`, `blocks`, `shares`, `seats` |

- `events/{scope}` lee la tabla de elecciones de la base (caché de consultas de 60 s) y marca `has_page` con las
  fechas que tienen `meta.json` en el paquete (`events/{scope}/{date}/`, caché de punteros de 60 s). Sin base
  configurada da 503 `no database configured`.
- `events/{scope}/{date}/{part}` valida en este orden `scope`, `date`, `part` y `format`; si la elección no está
  en el paquete da 404 `event not published` antes de leer ningún fichero. Los ficheros y sus gemelos CSV pasan
  por la misma caché de punteros que el análisis. Una elección publicada sin backtest no tiene gemelos
  `model-*`: pedirlos da el mismo 404 `event not published`.
- `meta` no tiene gemelo CSV en ninguna de las dos rutas. En `events`, `model` exige `table` y las demás partes no
  admiten `table`: cualquier otra combinación con `format=csv` da 404 `no csv for this part`.
- `backtest` no usa la base ni `manifest.json`: sin el fichero (o sin `meta.json`) da 404 `backtest not published`.

Errores nuevos (mismo cuerpo `{"status":"error","message":"..."}` y `Cache-Control: no-store`):

| Estado | `message` | Causa |
|---|---|---|
| 404 | `event not published` | la elección no tiene página publicada en el paquete (o falta el fichero pedido) |
| 404 | `backtest not published` | el ámbito no tiene backtest publicado (o falta el fichero pedido) |
| 404 | `no csv for this part` | `format=csv` en `meta`, o un `table` que no corresponde a la parte |

`invalid scope`, `invalid date`, `invalid format`, `unknown part`, `no database configured` y `no file system
configured` valen también aquí.

## Rutas de estudios

Solo `GET`. Sirven los ficheros publicados de cada estudio (controlador `Studies`, `site/v1/studies/{study}/`),
sin base de datos. Respuestas con `Cache-Control: public, max-age=60` y `ETag`; nunca inmutables, porque se
sobrescriben en cada publicación.

| Ruta | Parámetros | Respuesta JSON | CSV (`?format=csv`) |
|---|---|---|---|
| `/api/v1/studies` | | `{contract, studies}`; cada estudio publicado `{study, title, subtitle, registered_at, checked_at}`, leído de su `meta` | |
| `/api/v1/studies/{study}/{part}` | `study` ∈ `voto-cera`, `sesgo-cis`; `part` ∈ las partes del estudio (`voto-cera`: `meta`, `census`, `participation`, `vote`, `placement`, `prediction`, `sensitivity`, `check`; `sesgo-cis`: `meta`, `summary`, `elections`, `houses`, `matrix`, `election`); `table`: un CSV de la parte (el primero por defecto) | el fichero del estudio tal cual (sobre completo, `scope` nulo) | `{study}-{nombre}.csv`, con `nombre` ∈ `census-series`, `census-provinces`, `census-countries` (`census`); `participation`; `vote`, `ecological` (`vote`); `last-seat` (`prediction`); `sensitivity-grid`, `sensitivity-scenarios`, `placement-grid` (`sensitivity`); `check`; de `sesgo-cis`: `elections`, `houses`, `matrix`, `election-houses` (`election`) |

- `meta`, `placement` y `summary` no tienen gemelo CSV: `format=csv` da 404 `no csv for this part`, igual que un `table` que
  no pertenece a la parte.
- Un estudio está publicado cuando tiene `meta.json`; `studies` lista solo los publicados.

| Estado | `message` | Causa |
|---|---|---|
| 404 | `unknown study` | `study` no es un estudio conocido |
| 404 | `study not published` | el fichero pedido no está publicado |

`unknown part`, `no csv for this part` e `invalid format` valen también aquí.
