# Contrato de datos del paquete publicado (`site/v1`, contrato 1)

Descripción del paquete que escribe `python job.py publish` y que leerán la API y el sitio (fase 2). La
fuente de verdad en código es `mtpy/lib/bundle.py` (rutas, sobre, `SCHEMAS`, validación) y
`mtpy/lib/publish.py` (exportadores). Cada JSON se valida contra su esquema antes de escribirse.

## Árbol del paquete

El paquete vive bajo `site/v1/` de `app.fs` (`files/site/v1/` en local, `s3://{bucket}/site/v1/` en
producción). Los ensayos (`dry_run`) escriben bajo `site-dry/v1/` con la misma estructura y sin
`manifest.json` ni `history.json`.

```
site/v1/
    manifest.json                               último run de cada ámbito
    runs/{scope}/history.json                   headline de cada run, ascendente por run_id
    runs/{scope}/{run_id}/
        meta.json headline.json series.json polls.json fan.json house-effects.json dispersion.json
        {nowcast,forecast}/
            vote.json summary.json dist.json districts.json scenario.json projection.json
        csv/
            series.csv polls.csv fan.csv house-effects.csv dispersion.csv
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
  `history.json`) sí se reescriben.
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
`diagnostics` (`drift_k`, `multiplier`, `ages`, `composition`, `clip_rate` por modo), `seconds` (`init`,
`fit`, `nowcast`, `forecast`, `export`, `total`, a 1 decimal; `nowcast` y `forecast` miden solo las
simulaciones y `export` suma la escritura de ambos modos y de los ficheros del ciclo) y `freeze`.

`freeze` aquí significa que el run se publicó dentro de la ventana LOREG con `force`; no es el interruptor
editorial de `manifest.freeze`.

### `headline`

Resumen del run, lo primero que muestra el sitio y la fila de `history`.

`data`: `run_id`, `run_at`, `event_date`, `as_of`, `date_last`, `n_polls`, `nowcast` y `forecast`. Cada
modo es `{parties: [...], p_majority: {bloque: p}}` y cada partido `{name, pct, lo, hi, seats, seats_lo,
seats_hi, p_first}` en el orden de `vote_forecast()`; `seats` es el titular entero (`totals()`).

### `series`

Serie diaria ajustada de cada partido con sus cotas, hasta `date_fit_last` (o el último día del
pronóstico). Orientación columnar. Fuente: `Forecaster.forecast` y `Forecaster.fc_stat` (`cmin`, `cmax`).

`data`: `dates`, `parties`, y `mean`, `lo`, `hi` (cada uno `{partido: [valor por fecha]}`; `lo`/`hi` son
`null` donde no hay estadística). CSV `csv/series.csv`: `date, name, mean, lo, hi` (largo).

### `polls`

Sondeos publicados y resultado anterior. Fuente: `Forecaster.fc_series_raw` y `Forecaster.nfc_series`.

`data`: `parties`, `columns` (`POLL_COLUMNS`), `polls` y `results`.

- `polls`: registros con `date, pollster_id, pollster, sponsor, start_date, end_date, sample_size, mtype,
  rating, weight` (10 columnas) más una columna por partido en puntos porcentuales. `weight` a 3
  decimales.
- `results`: registros `date` + una columna por partido (resultado de elecciones anteriores).

CSV `csv/polls.csv`: la tabla `polls`.

### `fan`

Abanico de voto por partido y horizonte. Fuente: `sim.fan()`.

`data`: `horizons` (días, enteros ordenados; 0 es el nowcast) y `rows` (formato largo: `name`, `horizon`,
`mean`, `sd`, `lo`, `hi`; el partido va en `name`, como en el resto de ficheros). CSV `csv/fan.csv`: las
mismas filas.

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

`data`: `parties`, `regions` (`{id, name, seats}`, solo las circunscripciones: a diferencia de
`meta.regions`, no incluye el total del ámbito) y `rows`. Cada fila tiene 11 columnas: `region_id`,
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

## Punteros

### `history.json`

Evolución del pronóstico publicado (no regenerable). Se reconstruye desde los `headline.json` de los runs
completos del ámbito tras cada publicación y tras `unpublish`.

`data`: `scope` y `runs` (lista de `headline`, ascendente por `run_id`).

### `manifest.json`

Único puntero del paquete; se reescribe al final de cada publicación. Si no existe, se parte de uno vacío.

`data`:

- `contract` (1) y `updated_at` (instante de la última escritura).
- `scopes`: `{ámbito: entrada}`, con la entrada `{latest, run_at, event_date, as_of, date_last, n_polls}`
  (el `run_id` publicado y los datos de su `headline`). Un ámbito fallido conserva su entrada anterior.
- `freeze`: `{active, message}`; interruptor editorial que fija el job con `what: ["manifest"]`.
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
`Forecast.py`.

| Ruta | Fichero servido | `Cache-Control` |
|---|---|---|
| `/api/v1/health` | `{"status":"ok","contract":1}`, sin tocar el paquete | `no-store` |
| `/api/v1/manifest` | `manifest.json` | `public, max-age=60` |
| `/api/v1/scopes` | catálogo `data/es-scopes.csv` cruzado con el manifest (ver abajo) | `public, max-age=60` |
| `/api/v1/forecast/{scope}` | `meta.json` del run (atajo de `/forecast/{scope}/meta`) | 60 s, o inmutable con `?run=` |
| `/api/v1/forecast/{scope}/runs` | `runs/{scope}/history.json` | `public, max-age=60` |
| `/api/v1/forecast/{scope}/{part}` | `part` ∈ `meta`, `headline`, `series`, `polls`, `fan`, `house-effects`, `dispersion` | 60 s, o inmutable con `?run=` |
| `/api/v1/forecast/{scope}/{mode}/{part}` | `mode` ∈ `nowcast`, `forecast`; `part` ∈ `vote`, `summary`, `dist`, `districts`, `scenario`, `projection` | 60 s, o inmutable con `?run=` |

Inmutable es `public, max-age=31536000, immutable`: solo cuando el run se pide de forma explícita con
`?run=`, porque un run publicado no se reescribe. Sin `?run=` se sirve el último run del manifest y la
respuesta caduca a los 60 s (`web.cache_ttl`). Las rutas fijas (`health`, `manifest`, `scopes`, `runs`) se
registran antes que las genéricas.

### Parámetros

- `run`: `YYYYMMDD-HHMMSS` (UTC). Vacío o ausente equivale al último run del ámbito.
- `format`: `json` (por defecto) o `csv`. Solo hay gemelo CSV para `series`, `polls`, `fan`,
  `house-effects` y `dispersion`, y para las seis partes de cada modo; `meta` y `headline` no lo tienen.
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

### Cabeceras

- `ETag`: md5 de los bytes servidos, en todo contenido servido (también `/scopes`).
- `Access-Control-Allow-Origin: *` en todas las respuestas, errores incluidos (solo `GET`, sin preflight).
- `x-freeze: active` cuando `manifest.freeze.active` está activo (en `scopes`, `runs`, `forecast/...`).
  Es solo un aviso: la API sigue sirviendo los datos.

## Rutas de base (`/parties`, `/pollsters`, `/polls`, ...): fase 4

Pendientes de la fase 4 (las que leen la base de datos); su definición está en la sección "Contrato de
API" de la spec. Hasta entonces la API solo necesita el paquete publicado, no la base.
