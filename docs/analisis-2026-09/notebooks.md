# Módulo: Notebooks, salidas y flujo de uso

Repositorio: `/Users/luiss/HD/Proyectos/Code/elections-model` (rama `master`, HEAD `d2cd9d5 Core update`, 2026-09-23).
Análisis en modo solo lectura. Todas las referencias `fichero:línea` se han comprobado sobre el código actual.

---

## 1. Visión general

Los notebooks son la **única interfaz de uso** del modelo: no hay CLI, script de pipeline ni tests. Toda la operación (carga de datos, cálculo de ratings, forecast, simulación, figuras) se lanza desde Jupyter, y la lógica pesada vive en `mtpy/lib/` (`Computer`, `Forecaster`, `Simulator`, loaders). Los notebooks hacen tres cosas: (a) fijar parámetros en una celda de configuración, (b) invocar métodos de la librería en el orden correcto y (c) construir figuras y tablas, en parte con lógica propia (pandas) que no está en la librería.

### 1.1 Inventario (31 notebooks + 1 checkpoint)

| Carpeta | Notebook | Estado | Última ejecución (mtime) | Propósito |
|---|---|---|---|---|
| `notebooks/` | `PollstersRatings.ipynb` | operativo | 2025-12-29 | Diagnóstico de errores/desviaciones de encuestas en todos los eventos y ratings de encuestadoras |
| `notebooks/` | `PollstersEvent.ipynb` | operativo (rompe si `show_fc=True`) | 2025-12-29 | Errores de las encuestadoras en un evento concreto (2023-07-23) |
| `notebooks/` | `PollsErrors.ipynb` | operativo | 2025-12-29 | Distribución de errores por semana/porcentaje de voto |
| `notebooks/` | `PollsForecast.ipynb` | **roto** (`np.NaN`, numpy 2) | 2025-12-29 | Forecast del evento 2027-08-22, figuras `2708-analysis-*` y CSV `fc/fc*.csv` |
| `notebooks/` | `PollsSimulations.ipynb` | operativo | 2025-12-29 | Simulación de escaños (4 modos), figuras `2708-simulations-*`, CSV `fc/sim*.csv` |
| `data-load/` | `ResultsLoadInfoElectoral.ipynb` | operativo | 2025-12-29 | Carga de resultados oficiales por provincia (XLSX de infoelectoral) a `events_data`/`events_results` |
| `data-load/` | `PollsLoadWikipedia.ipynb` | operativo, **con efectos secundarios en BD** | 2026-09-22 | Scraping de encuestas de Wikipedia a `polls`/`polls_results` y cálculo de pesos del evento |
| `data-load/` | `PollsCompute.ipynb` | operativo, **con efectos secundarios en BD** | 2026-09-22 | Recalcula pesos, errores, desviaciones y ratings de todos los eventos (`save=True`) |
| `lab/` | `EventsSeats.ipynb` | operativo | 2026-09-23 | Diagnóstico del estimador votos→escaños (`v2s`) |
| `lab/` | `PollsOptimize.ipynb` | operativo | 2026-09-23 | Elección de ancho de banda (scott/silverman/isj, fijo vs adaptativo) para un partido |
| `lab/` | `PollstersOptimize.ipynb` | operativo | 2026-09-23 | Reimplementación paso a paso de `compute_ratings` y del estimador de sesgo |
| `events/es-201911/` | `PollsAnalysis`, `PollsSimulations`, `PollstersEvent` | **rotos** (imports `mtpy.lib.elections.*`, `sys.path` a 4 niveles, `path='elections'`) | 2025-03-27 | Copias antiguas de los notebooks raíz parametrizadas para 2019-11-10 |
| `events/es-202307/` | idem | **rotos** (mismos motivos) | 2025-03-27 | Copias para 2023-07-23 |
| `build/` (untracked, `.gitignore`) | 14 notebooks (`Build`, `ElectionsTest`, `Electocracia*`, `EventParams`, `EventsData`, `EventsResults`, `Fandom`, `Infoelectoral`, `Parties`, `Pipeline`, `Test`, `Test2`, `BiasEstimator`) | scratch; varios importan `mtpf`/`mtpy.lib.elections` (rotos) | 2023-05 → 2025-06 | Exploración, migraciones de datos y pruebas; `Pipeline.ipynb` usa un framework anterior (`mtpf`, `pipelines.elections.EventsResults`) |

Los tres notebooks de `events/*` para cada evento son versiones antiguas de `PollsForecast` (llamado allí `PollsAnalysis`), `PollsSimulations` y `PollstersEvent`, con `event_date` fijado y `n_sim=10000` en lugar de 1000. La única diferencia funcional relevante es que `PollsAnalysis` incluye dos figuras extra por encuestadora agrupadas por bloques (`group_results(..., blocks=build_blocks(fc.bmaps['vs']))`, celdas 17-18) que el `PollsForecast` actual no tiene.

### 1.2 Cabecera común y resolución de rutas

Todos los notebooks empiezan igual:

```python
%matplotlib inline
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import sys; sys.path.append('..')          # '../..' en data-load y lab, '../../../..' en events, '../../..' en build
from mtpy import mtpy
SEED = 42; np.random.seed(SEED)
mtpy.run()
```

- `sys.path.append(<relativo>)` depende de que el kernel arranque con CWD = carpeta del notebook. Hay cuatro variantes distintas (`'..'` ×5, `'../..'` ×6, `'../../..'` ×14, `'../../../..'` ×6). En `events/*` el nivel es incorrecto para la estructura actual (apunta al padre del repositorio).
- `mtpy.run()` (`mtpy/mtpy.py:19-91`) fija `shpath = <repo>` (o `<repo>/../../shared` si existe), carga `<repo>/.env`, `config.json`, conecta a PostgreSQL y crea `App.fs = FileSystem(<repo>/files)` (`mtpy/mtpy.py:36,83`). Verificado en ejecución: `fs.path = /Users/luiss/.../elections-model/mtpy/../files`.
- **Todas las escrituras de ficheros son relativas a `files/`**, no al notebook: `Computer.get_path`, `Forecaster.get_path` y `Simulator.get_path` devuelven `'{self.path}/{name}'` (`computer.py:2611-2625`, `forecaster.py:1183-1185`, `simulator.py:666-668`) y `FileSystem.get_path` antepone `files/` salvo ruta absoluta (`io.py:38-58`). Con `path='.'` una figura pedida como `img/2708-x.png` acaba en `files/./img/2708-x.png`. Las funciones de `dataviz` (`plot_histogram`, `plot_scatter`, `plot_series`, `plot_scores`, `plot_figure`) también guardan vía `App.get_().fs.write_bytes` (`dataviz.py:727-747`), que crea directorios (`io.py:416`).
- En `events/*` se usa `path='elections'`, que escribiría en `files/elections/img/...` (no existe hoy; las figuras `1911-*`/`2307-*` presentes en `files/img/` proceden de ejecuciones con `path='.'`).

### 1.3 Ficheros de entrada (todos en `files/`, **ignorados por git**)

| Fichero | Usado por | Contenido |
|---|---|---|
| `files/params.json` | `Forecaster` (bmaps), `Simulator` (`parties.event`, `smap`) (`simulator.py:98-101`) | Por scope y evento: `parties.event/polls`, `bmaps` {`max`, `min`, `main`, `blocks`, `vs`}, `smap` (reglas `agg`/`sub`/`split`) |
| `files/wikipedia/wp-urls.json`, `wp-maps.json` | `WikipediaLoader` (`loader.py:305-307`) | URLs de Wikipedia por evento (`results`, `polls`, `years`), y mapas de alias → nombre canónico de `parties`/`pollsters`/`sponsors` |
| `files/infoelectoral/es/PROV_02_YYYYMM_1.xlsx` + `.json` | `InfoElectoralLoader` (`loader.py:25,52,63-67,71`) | Excel oficial por provincia (17 eventos 1977-2023) y mapa de columnas de partidos |
| `files/es-provinces.csv`, `es-regions.csv` | loaders (`loader.py:57-58`), `PollsSimulations` celda 1 | Códigos de provincia/CCAA |

Sin estos ficheros y sin la BD PostgreSQL (credenciales en `.env`, también ignorado) **ningún notebook se puede ejecutar desde un clon del repositorio**.

---

## 2. Flujo end-to-end que sigue el usuario

```
[0] Preparar files/ (params.json, wp-*.json, XLSX infoelectoral, CSV provincias) + BD PostgreSQL + .env
[1] data-load/ResultsLoadInfoElectoral   (por evento pasado)  -> events_data, events_results
[2] data-load/PollsLoadWikipedia         (evento en curso)    -> polls, polls_results (+ compute_weights del evento)
[3] data-load/PollsCompute               (todos los eventos)  -> polls (pesos, errores, desviaciones, rating), polls_results, pollsters_ratings
[4] PollstersRatings / PollsErrors / PollstersEvent   (diagnóstico del Computer, figuras ratings-*, 2307-event-*)
[5] PollsForecast                        (evento 2027-08-22)  -> files/fc/fc.csv, fc-stat.csv, figuras 2708-analysis-*
[6] PollsSimulations                     (evento 2027-08-22)  -> files/fc/sim.csv, sim-stat.csv, figuras 2708-simulations-*
[lab] EventsSeats / PollsOptimize / PollstersOptimize   (diagnóstico de estimadores y bandwidths, sin salida a disco salvo seats-regression.png)
```

### 2.1 Etapa 1 — `ResultsLoadInfoElectoral.ipynb`

Parámetros: `scope='es'`, `event_date='2023-07-23'`, `verbose=1`, `path='.'`, `save=False`.

1. `InfoElectoralLoader(scope, event_date, verbose, path)` (`loader.py:27-68`): fija `fname = 'PROV_02_' + YYYYMM + '_1'`, lee `es-regions.csv`, `es-provinces.csv`, la tabla `parties` y el mapa de columnas `PROV_02_YYYYMM_1.json`.
2. `read_data()` (`loader.py:84-…`): abre el XLSX (`read_file`, `loader.py:70-82`), detecta cabeceras, mapea columnas (`Población`→`population`, `Votos válidos`→`votes`, …) y produce `loader.data` indexado por `region_id` (0 = total nacional) con `totals` y votos por partido. Informa `parties_missing`.
3. `build_series()` → `loader.totals` (53 × 13: `date, scope, region_id, region, seats, population, stations, registered, counted, votes, blank, invalid, …`) y `loader.results` (2 968 × 9: `date, scope, region_id, party_id, region, party, votes, pct, seats`).
4. `save_totals()` / `save_results()` sólo si `save=True` (`loader.py:181-218`) → tablas `elections.events_data` y `events_results`.
5. `show_summary()` imprime cuadre de votos y escaños (salida: `Total: 24.487.414 | Votes: 24.487.414 | Diff: 0 | Seats: 350`); `print_df(loader.seats / votes / pcts)`.

`build/Test.ipynb` (untracked) recorre todos los `get_event_dates(scope)` haciendo esto en bucle con `save=True`; es el único sitio donde existe la "carga masiva", y está roto (`mtpy.lib.elections.loader`).

### 2.2 Etapa 2 — `PollsLoadWikipedia.ipynb`

Parámetros: `scope='es'`, `event_date='2027-08-22'`, `years=None`, `exclude=None`, `verbose=1`, `path='.'`, **`save=True`**, `overwrite=False`.

1. `WikipediaLoader(...)` (`loader.py:~270-323`): lee `wp-urls.json`/`wp-maps.json`, obtiene la fila del evento y los diccionarios `colmap`/`idmap` de partidos, encuestadoras y patrocinadores.
2. `read_data()` (`loader.py:555-601`): descarga las tablas `wikitable` de la URL `polls` (User-Agent `ManyThings/1.0`), filtra por `years` (`["2023","2024","2025","2026"]` en `wp-urls.json`), y parsea filas (`read_rows`). Encuestadora no mapeada → `pollsters_missing` y la fila se descarta (`loader.py:396-408`): en la última ejecución se descartaron todas las variantes `CIS (Target Point)`, `CIS (Sondaxe)`, … y `Sumar`. Salida: 489 encuestas.
3. `build_series()` (`loader.py:603-650`): construye `polls` (489 × 32; añade `pub_date=end_date`, `mtype` desde `pollsters`, `computed=False`, `days = event_date - date`) y `results` (5 251 × 13).
4. `select_series(overwrite=False)` (`loader.py:652-658`): elimina las encuestas ya existentes en BD (clave `m_polls.key`) → quedan 6 nuevas (42 resultados).
5. `save_polls()` / `save_results()` (`loader.py:668-708`): `stage_write` + `upsert` + `vacuum`; con `overwrite=True` hace `DELETE` previo del evento.
6. `print_df(loader.parties_checks)`: tabla partido × {`event`, `polls`} para detectar partidos presentes sólo en encuestas o sólo en resultados.
7. `compute_series(save=True, overwrite=True)` (`loader.py:710-724`): instancia `Computer(event_dates=[event_date])` y ejecuta **sólo** `compute_weights` → 602 filas del evento 2027 con `mtype, proc_sample, weight_over, weight_sample, weight_rating, computed`. Los errores/ratings no se recalculan aquí (no tiene sentido para un evento sin resultado).

### 2.3 Etapa 3 — `PollsCompute.ipynb`

Parámetros: `scope='es'`, `event_dates=None` (todos), **`save=True`**, `path='.'`.

```python
comp = Computer(scope, event_dates, verbose=1, path).build_series()   # 3 842 × 511
comp.compute_weights(save=save, overwrite=True)   # mtype, proc_sample, overweights, merge ratings -> 3 825 filas
comp.compute_errors(save=save)                    # error_avg/blocks, bias_* (log-odds) -> 22 590 polls_results + 3 825 polls
comp.compute_deviations(save=save)                # ajusta estimador de sesgo por evento (11 eventos) -> bias_dev_adj, bias_dev_err
comp.compute_ratings(save=save)                   # ratings por evento (1996 … 2027) -> 1 012 filas pollsters_ratings + polls.rating/weight_rating
comp.print_ratings()
```

Es el paso que fija los pesos que después usa el `Forecaster` (`series['weight'] = weight_over * weight_sample * weight_rating`, `forecaster.py:265-266`). La celda de ratings recorre eventos en orden y para cada uno usa sólo las encuestas de eventos anteriores (`polls.loc[:edts[i-1]]`, `computer.py:928-948`), es decir, el rating aplicado a un evento se calcula *sin mirar* ese evento (out-of-sample por construcción). Parámetros del `Computer` que afectan (defaults en `computer.py:32-53`): `drop_mtypes=['aggr','online']`, `drop_ctypes=['wban','exit']`, `alpha=.05`, `ol_dev=3`, `ol_max=5000`, `wspan=4`, `pos_decay=.5`, `week_decay=.7`, `year_decay=.9`, `bias_dev_tau=.01`, `min_polls=3`, `ratings_margin=.05` (no usado en el código), `error_weights={'avg':.7,'blocks':.3}`.

### 2.4 Etapa 4 — Diagnóstico del Computer

**`PollstersRatings.ipynb`** — `scope='es'`, `event_dates=None`, `pollster=None`, `drange=(6, 42)`, `n_last=1`, `bmap='vs'`, `prefix='ratings'`.

- `df = comp.get_polls_metric(metric='error', bmap='vs', drange=(6,42), n_last=1)` (`computer.py:1390-1459`): última encuesta de cada encuestadora entre 6 y 42 días antes de cada evento, con columnas `sample_size, proc_sample, weight (= weight_over·weight_sample), error_avg, bias_avg, error_blocks, bias_blocks, bias` más una columna de error por bloque (`Derecha`, `Izquierda`). Índice `(event, pollster, days)`; `event` es una etiqueta `'{año}-{día}{inicial del mes}'` (ej. `1993-6J`). Resultado: 166 × 11.
- Figuras (todas en `files/img/`): `ratings-bullseye.png` (`plot_deviations`, diana 2D error bloque derecha vs izquierda), `ratings-bullseye-events.png` (sólo si `event_dates` no es `None`), `ratings-bullseye-pollsters.png` (rejilla 2×2 para `['CIS','GAD3','GESOP','40dB']`, encuestadoras hard-coded), `ratings-histogram.png` y `ratings-histogram-split.png` (histogramas ponderados por `weight` con KDE, media y std), `ratings-deviation.png` sólo con `pollster`.
- Lógica propia del notebook (celdas 10-16): construye `weeks = (days+1)//7`, `period = (n_weeks+1) - weeks` recortado a `[1, n_weeks]` y una secuencia `seq` evento-periodo; tabla `error_avg` por evento/semana con barras y gradientes (`print_df`); scatter `days` vs `|error_blocks|` con regresión local (`reg_type='lk'`, `reg_bw=7`), `seq` vs error (`reg_bw=n_weeks`) y `proc_sample` vs error (`reg_bw=1000`); barras de recuento/media por `days`, `period` y `event`. Nada de esto se guarda en disco ni está en la librería.
- `comp.ratings`, `comp.print_ratings()`, `comp.plot_ratings()` (tabla y gráfico de ratings por evento/encuestadora).

**`PollsErrors.ipynb`** — igual que el anterior con `bmap='main'` (6 partidos), `prefix='errors'`. Histograma de `error_avg` (stat `prob`), tabla evento × semana, y `dr = comp.get_error_estimator_data(drange, n_last)` (`computer.py:1510`; 1 339 × 9: `date, pollster, party, regional, weeks, weight, color, pct, error`) que es el conjunto de entrenamiento del estimador de error `v2err` usado por el simulador. Scatter `pct` vs `error` con regresión local (`reg_bw=5`) y media ponderada por bins de 10 puntos. **No guarda ninguna figura** (ningún `path=`).

**`PollstersEvent.ipynb`** — `event_date='2023-07-23'`, `pollster=None`, `show_fc=False`, `drange=None`, `n_last=1`, `bmap='vs'`, `prefix='2307-event'`.

- `dp`: `get_polls_metric(metric='error', bmap='vs')` **sin** `drange`/`n_last` → todas las encuestas del evento (incluye CEMOP a 431 días) agregadas por encuestadora (`polls`, `sample_size` mediana, `error_blocks` medio).
- `df`: con `n_last=1` → última encuesta de cada casa (30 × 10).
- Si `show_fc=True` (rama no ejecutada y **rota** por `np.NaN`): ajusta el `Forecaster`, evalúa el forecast en `event_date - drange[0]` días y lo añade como pseudo-encuestadora `'[MT]'` con `comp.poll_errors(...)` para comparar el modelo con las casas.
- Figuras: `pollsters-event-metrics.png` (4 paneles `plot_scores`: nº encuestas, muestra mediana, MAE, error última encuesta), `2307-event-deviation.png` (`plot_errors`), `2307-event-bullseye.png` (`plot_deviations`), `2307-event-histogram.png`, `2307-event-histogram-split.png`.

### 2.5 Etapa 5 — `PollsForecast.ipynb`

Parámetros: `scope='es'`, `event_date='2027-08-22'`, `drange=None`, `bmap='max'`, `fc_fit=True`, `dt_from=None`, `dt_to=None`, `prefix='2708-analysis'` (`edkey = event_date.replace('-','')[2:6]` → `YYMM`).

1. `fc = Forecaster(scope, event_date, drange, bmap, verbose=1, path).build_series()` (`forecaster.py:221-311`): concatena resultados del evento anterior + encuestas del ciclo (235 × 31), calcula `tfs` (días desde `date_start`) y `tte` (días hasta `date_end`), `weight_rating` (rellenando con `quality` de la encuestadora si no hay rating), `weight = weight_over·weight_sample·weight_rating`, agrupa partidos según `bmap` (`build_blocks`/`group_results`) y crea `forecast` (índice diario `date_start…date_end`, columnas `names + ['-']`) y `fc_stat` (mismo índice, una celda-dict por partido).
2. `dt_max` = última fecha de encuesta; `dt_from = dt_max - 180 días` por defecto.
3. `fc.fit_forecast(max_fc=10, fillna=False)` (`forecaster.py:362-403`): para cada partido llama `fit(name, max_fc, ret_stat=True)` (regresión local con kernel), escribe `forecast[name]` y `fc_stat[name] = {mean, cmin, cmax, err, nobs, neff}`, y recalcula `'-' = 100 - suma`. `max_fc=10` extrapola 10 días más allá de la última encuesta.
4. `fc.save_forecast('fc')` (`forecaster.py:451-461`) → `files/fc/fc.csv` (columnas `date, PP, PSOE, VOX, SUMAR, UP, SALF, -`; 1 492 filas diarias 2023-07-23 → 2027-08-22) y `files/fc/fc-stat.csv` (misma rejilla; cada celda es el `repr` de un dict Python, p. ej. `{'mean': 33.83, 'cmin': 32.77, 'cmax': 34.90, 'err': 0.53, 'nobs': 79.0, 'neff': 42.99}`, recuperado con `literal_eval` en `load_forecast`, `forecaster.py:463-476`). Con `fc_fit=False` se carga desde CSV.
5. Figuras `files/img/2708-analysis-…`: `polls-monthly.png` (encuestas por mes, `ts_resample(freq='M')`), `polls-pollster.png` (encuestas por casa), `polls-pollster-race.png` (sólo si hay encuestas con `days∈[6,41]`), `polls-pollster-sizes.png` (muestra mediana por casa), `forecast-max.png` y `forecast-blocks-max.png` (series completas, mensual), `forecast-race-max.png` y `forecast-race-blocks-max.png` (últimos 180 días, semanal, con IC), `forecast-results-max.png` y `forecast-results-blocks-max.png` (barra apilada del último valor), `pollsters-max.png` y `pollsters-race-max.png` (rejilla 2×2 por encuestadora `['CIS','GAD3','GESOP','40dB']` → ids `[1,2,17,15]`).
6. Lógica propia del notebook: conteo mensual de encuestas y de encuestas con muestra (celda 5), resolución nombre↔id de encuestadora (celda 15, `is_number`), cálculo de `ylim` común para las rejillas.

### 2.6 Etapa 6 — `PollsSimulations.ipynb`

Parámetros: `scope='es'`, `event_date='2027-08-22'`, `drange=6`, `n_sim=1000`, `fc_fit=True`, `seed=42`, `prefix='2708-simulations'`. Carga `parties` y `provinces` (`es-provinces.csv`) que **no se usan** después.

1. `sim = Simulator(scope, event_date, drange, seed, verbose=1, path)` (`simulator.py:26-200`): lee `params.json` (partidos del evento y `smap`), construye internamente un `Forecaster` con `bmap=names` y `drange=(6, None)`, un `Computer` sobre los eventos pasados para los estimadores `v2seats` (votos→escaños) y `v2err` (error esperado), totales por región y resultados previos. `limit_date = event_date - 6 días`. `default_params = {n_sim:1, split:False, random:False, names:None, regions:None}` (`simulator.py:178-184`).
2. `sim.fit_forecast(names=sim.params['names'], max_fc=10, fillna=True)` → `Forecaster.fit_forecast` + `build_forecast()` (`simulator.py:310-336`): tabla partido × `{mean, regional, err, nobs, error}` evaluada en `limit_date`; `error` = predicción de `v2err(mean, regional, weeks)` si existe, si no `err`.
3. `sim.save_forecast('sim')` → `files/fc/sim.csv` (13 partidos + `-`) y `sim-stat.csv`.
4. Cuatro modos (`run(split, random, n_sim)`, `simulator.py:625-664`):
   - **LS** `split=False, random=False`: `vpred = mean`; escaños nacionales por partido con el estimador `v2seats.predict([[pct, regional]])` (`simulator.py:616-618`), sin restricción de suma 350.
   - **MC** `split=False, random=True, n_sim=1000`: `vpred = clip(pct + std_err · t_{nobs-2}, 0)` con `std_err = sqrt(err² + pct_err²)` (`simulator.py:521-542`), luego `v2seats`.
   - **DH** `split=True, random=False`: reparte `vpred` por provincia a partir de los resultados previos (`build_umat`, con reglas `smap`), y aplica D'Hondt por provincia (`alloc_dhondt`, `simulator.py:482-494`); fila `es` = suma de provincias.
   - **MT** `split=True, random=True, n_sim=1000`: DH con ruido.
   Para cada modo `sim.totals(sort=True)` (`simulator.py:397-440`): mediana por partido de `dist()` (escaños nacionales por simulación), `floor`, y reparto de la diferencia hasta 350 por "ciclos" + mayores restos.
5. Figuras `files/img/2708-simulations-…`: `pct.png`, `pct-blocks.png` (forecast %), `seats-{ls,mc,dh,mt}.png` y `-blocks` (barras apiladas de escaños), `dist-{mc,mt}-{nat,reg}.png` (KDE ridge de la distribución de escaños por partido, `plot_dist_kde`, `simulator.py:462-479`), `seats.png` y `seats-blocks.png` (rejilla 4×1 comparando modos). Tabla resumen `LS|MC|DH|MT` con `TOTAL`.
6. Tras `run(split=True, ...)` el notebook muestra `sim.result()` (`simulator.py:370-378`), que devuelve **la simulación `loc=0`** región × partido, no la mediana.

### 2.7 Laboratorio

- **`EventsSeats`**: `event_dates = get_event_dates(scope, date_from='1980-01-01', skip=1)` (`data.py:463-521`; `skip=1` excluye el último evento); `df = comp.get_seats_estimator_data()` (`computer.py:1608`; columnas `party, regional, pct, seats, pos, color, years, weight, ratio`); `v2s = comp.get_seats_estimator()`; `v2s.predict([[34.6, 0]])` → 138,3 escaños; rejilla 2×2 (`seats`/`ratio` vs `pct`, nacional/regional, regresión local `reg_bw='scott'`) → `files/img/seats-regression.png`; derivada de la curva (`ts_fitreg(...).diff()`).
- **`PollsOptimize`**: `event_date='2027-08-22'`, `drange=6`, `bmap='max'`, `party='PP'`, `bw='isj'`; compara bandwidths fijos (`Kernel(x, weights, bw).get_bw_fixed()` → scott 109,6 / silverman 93,1 / isj 71,0 días), convergencia del adaptativo (`get_bw_adaptive(n_iter=3, min_delta=0.1)`), `fc.plot_bws` en rejilla 3×3, `fc.print_weights`/`plot_weights`, `fc.fit(party, ret_stat=True)` (última fila: `mean 31.91, cmin 30.56, cmax 33.25, err 0.67, nobs 100, neff 48`), y comparación polinómica (`reg_type='ls'`) vs local fija vs adaptativa con IC `alpha=0.05`. No guarda figuras.
- **`PollstersOptimize`**: `comp.print_rating_weights('GAD3')`; distribución de `bias_dev_adj` por evento con `Stat(..., weights)` (media ≈ 0, std 1-2); reproduce `fit_bias_estimator` para el último evento con `LocalKernelEstimator(d['lor'], weights, **comp.reg_params).fit(px, alpha)` y `lor_to_bias`; y **reimplementa línea a línea** `Computer.pollster_ratings` (celdas 17-18 ≡ `computer.py:1094-1190`): `rating_adj = Φ(-bias_dev_adj / τ)` con `τ = bias_dev_tau`, `rating = (num_polls_w·rating_adj + min_polls·quality)/(num_polls_w + min_polls)`, `weight_rating = log1p(rating)/log1p(mean(rating))`. No guarda figuras.

---

## 3. Salidas producidas

### 3.1 CSV (`files/fc/`, 29-dic-2025)

| Fichero | Productor | Forma | Notas |
|---|---|---|---|
| `fc.csv` | `PollsForecast` (`save_forecast('fc')`) | 1 492 × 8 (`date`, 6 partidos `max`, `-`) | valores diarios del forecast; NaN antes de la primera encuesta |
| `fc-stat.csv` | idem | 1 492 × 7 | dicts serializados como texto Python (no JSON) |
| `sim.csv`, `sim-stat.csv` | `PollsSimulations` (`save_forecast('sim')`) | 1 492 × 15 / × 14 | 13 partidos de `params.parties.event` |

No se guardan a disco ni las matrices de simulación (`sim.frames/units/results`) ni la tabla resumen de escaños ni los ratings (estos últimos van a BD).

### 3.2 Figuras (`files/img/`, 142 PNG en raíz + subcarpetas `2307/` (34, ene-2025) y `7706/`)

Patrón de nombre: `{edkey}-{etapa}-{figura}[-{bmap}].png` con `edkey = YYMM` del evento (`2708` → 2027-08-22, `2307`, `1911`, `7706` → 1977-06-15) y etapa ∈ {`analysis`, `event`, `simulations`}; sin evento: `ratings-*`, `pollsters-event-metrics`, `seats-regression`. Existen variantes `-full`, `-min`, `-vs` de ejecuciones con otros `bmap` (mar-2025) que ya no se generan con la configuración actual, y 12 ficheros `fig*_*.png` (censo, tilt, canibalización, …) que no produce ningún notebook del repositorio. `files/img/2307/` duplica las `2307-*` de la raíz con fecha anterior.

### 3.3 Base de datos

`PollsLoadWikipedia` (polls, polls_results, y `compute_weights`), `PollsCompute` (polls, polls_results, pollsters_ratings) y `ResultsLoadInfoElectoral` (events_data, events_results, con `save=True`). No hay versionado ni export de estas tablas.

---

## 4. Cuánta lógica vive en notebooks vs librería

- **Librería**: toda la estadística (pesos, errores, log-odds, estimador de sesgo, ratings, regresión local, KDE, simulación, D'Hondt) y las figuras "de modelo" (`plot_deviations`, `plot_errors`, `plot_ratings`, `plot_forecast_series`, `plot_forecast_output`, `plot_dist_kde`, `plot_bws`, `plot_weights`).
- **Notebooks**: (1) la orquestación (orden de llamadas y parámetros), (2) agregaciones ad hoc de diagnóstico (semanas/periodos/secuencias en `PollstersRatings` c.10-16, bins de % en `PollsErrors` c.9, conteos mensuales/por casa en `PollsForecast` c.5-8, métricas por casa en `PollstersEvent` c.4-6), (3) composición de rejillas (`create_figure`/`adjust_figure`/`plot_figure`) y cálculo de `ylim` comunes, (4) la tabla resumen de escaños por modo (`PollsSimulations` c.19), (5) resolución de ids de encuestadoras (`PollsForecast` c.15), y (6) en `lab/PollstersOptimize` una **copia** completa de `pollster_ratings`. Estimo que un 25-30 % de las celdas de código contienen lógica de análisis no encapsulada.

Aspectos positivos de diseño: celda de parámetros única y homogénea al principio de cada notebook (fácil de parametrizar con papermill), separación clara `data-load / análisis / lab`, `fc_fit`/`load_forecast` para no reajustar, semilla explícita, nombres de figura sistemáticos, y `verbose=1` con trazas de cada etapa.

---

## 5. Pipeline (pasos ordenados)

| # | Paso | Dónde | Qué hace | Parámetros clave |
|---|---|---|---|---|
| 0 | Configuración | `files/params.json`, `files/wikipedia/*.json`, `.env`, `mtpy/mtpy.py:19-91` | Rutas, BD, mapas de partidos/bloques | `scope='es'`, evento |
| 1 | Resultados oficiales | `notebooks/data-load/ResultsLoadInfoElectoral.ipynb` c.1-7; `loader.py:27-218` | XLSX infoelectoral → `events_data`, `events_results` | `event_date`, `save=False` |
| 2 | Encuestas | `notebooks/data-load/PollsLoadWikipedia.ipynb` c.1-8; `loader.py:270-724` | Wikipedia → `polls`, `polls_results`; `compute_weights` del evento | `event_date='2027-08-22'`, `save=True`, `overwrite=False` |
| 3 | Pesos/errores/ratings | `notebooks/data-load/PollsCompute.ipynb` c.2-6; `computer.py:401-1008` | `build_series` → `compute_weights(overwrite=True)` → `compute_errors` → `compute_deviations` → `compute_ratings` | `event_dates=None`, `save=True` |
| 4a | Diagnóstico global | `notebooks/PollstersRatings.ipynb`, `PollsErrors.ipynb` | `get_polls_metric(metric='error')`, dianas, histogramas, ratings | `drange=(6,42)`, `n_last=1`, `bmap='vs'`/`'main'` |
| 4b | Diagnóstico de evento | `notebooks/PollstersEvent.ipynb` | Errores por encuestadora en un evento pasado | `event_date='2023-07-23'`, `show_fc=False` |
| 5 | Forecast | `notebooks/PollsForecast.ipynb` c.2-17; `forecaster.py:221-476` | Regresión local ponderada por partido, CSV `fc/`, figuras `*-analysis-*` | `bmap='max'`, `max_fc=10`, `fillna=False` |
| 6 | Simulación | `notebooks/PollsSimulations.ipynb` c.2-21; `simulator.py:292-664` | Forecast con `bmap=names`, 4 modos LS/MC/DH/MT, CSV `fc/sim*`, figuras `*-simulations-*` | `drange=6`, `n_sim=1000`, `seed=42`, `max_fc=10`, `fillna=True` |
| 7 | Laboratorio | `notebooks/lab/*.ipynb` | Estimador de escaños, bandwidths, ratings paso a paso | — |

---

## 6. Hallazgos

Ordenados por severidad. "Cómo comprobar" indica una forma concreta de reproducirlo.

### Alta

**H1. Reproducibilidad: nada del flujo es ejecutable desde el repositorio.** `files/` (con `params.json`, `wp-urls.json`, `wp-maps.json`, XLSX de infoelectoral, `es-provinces.csv`) está en `.gitignore` (`.gitignore:2`), la BD PostgreSQL y `.env` son locales, y `Simulator.__init__` y los loaders leen esos ficheros sin alternativa (`simulator.py:98-101`, `loader.py:57-58,305-307`, `loader.py:71`). *Comprobar*: `git clone` limpio + `pip install -r requirements.txt` + abrir `PollsForecast.ipynb` → `FileNotFoundError`/error de conexión en `mtpy.run()`.

**H2. `PollsForecast.ipynb` no se ejecuta con el entorno actual (numpy 2.4.4).** Celdas 6, 7 y 8 usan `np.NaN` (`.replace(0, np.NaN)`), eliminado en NumPy 2.0 (verificado: `AttributeError: np.NaN was removed in the NumPy 2.0 release`). `requirements.txt` fija `numpy==1.25.2`, pero el intérprete `/Users/luiss/.pyenv/versions/3.11.9` tiene 2.4.4 y los notebooks de `lab/` ejecutados hoy muestran reprs `np.float64(...)` (numpy 2). Mismo problema en `PollstersEvent` c.5 (sólo si `show_fc=True`) y en los seis `events/*`. *Comprobar*: `jupyter nbconvert --execute notebooks/PollsForecast.ipynb`.

**H3. `Simulator.totals()` inventa escaños para partidos con 0 escaños en todas las simulaciones.** `simulator.py:405-427`: cuando `diff = 350 - Σ floor(mediana)` es ≥ número de partidos, el bloque "cycles" suma `diff // n_partidos` a **todos** los partidos antes de repartir los restos, ignorando que la mediana (y el máximo) sea 0. En LS/MC el estimador `v2seats` no está restringido a sumar 350 (`simulator.py:616-618`), así que el déficit es sistemático. *Verificado* con `n_sim=30`: medianas suman 329 (déficit 21 = 1 ciclo + 8 restos) → SALF y UPN reciben 1 escaño pese a `max=0` en las 30 simulaciones; CC pasa de mediana 1 a 2. La tabla resumen del notebook (`SALF 1, CC 1, UPN 1` en LS y MC) es consecuencia de este bug, y también explica que LS y MC coincidan exactamente. *Comprobar*: ejecutar `sim.run(split=False, random=True, n_sim=30); sim.dist().max(); sim.totals()`.

### Media

**M1. Los notebooks de `events/*` (6) y varios de `build/` están rotos.** Importan `mtpy.lib.elections.forecaster/simulator/computer/utils` (verificado: `ModuleNotFoundError: No module named 'mtpy.lib.elections'`), usan `sys.path.append('../../../..')` (apunta fuera del repo) y `path='elections'` (escribirían en `files/elections/`). Son copias del 27-mar-2025 de los notebooks raíz. *Comprobar*: ejecutar la celda 0 de `notebooks/events/es-202307/PollsAnalysis.ipynb`.

**M2. `sim.result()` en el modo MT muestra una única simulación aleatoria como si fuera "el resultado".** `Simulator.result(loc=0)` devuelve `self.results[0]` (`simulator.py:370-378`); en `PollsSimulations` c.15 se muestra tras `run(split=True, random=True, n_sim=1000)` como tabla provincia × partido. *Verificado*: `result(0).es` PP=136, `result(1).es` PP=149, `totals()` PP=136. No existe un método que devuelva la mediana por provincia. En el notebook publicado, la fila `es` de esa tabla (PP 124 / PSOE 127) contradice la tabla resumen (PP 126 / PSOE 118).

**M3. Efectos secundarios silenciosos en BD por "Run All".** `PollsCompute` (`save = True`, c.1) y `PollsLoadWikipedia` (`save = True`, c.1, más `compute_series(save=save, overwrite=True)`, c.8) escriben/borran en PostgreSQL (`loader.py:668-708` hace `DELETE` con `overwrite=True`; `computer.py:476-…` con `overwrite=True`). No hay confirmación, dry-run ni copia de seguridad. *Comprobar*: `grep -n "save = True" notebooks/data-load/*.ipynb`.

**M4. Docstring de `compute_ratings` describe una fórmula distinta a la implementada (y a la copiada en `PollstersOptimize`).** `computer.py:895-905` habla de `prior_weight = quality * num_polls / (num_polls + mean)`, `dev_reverted` y "rating scaled to [5-95]"; el código real (`computer.py:1173-1186`, idéntico a `PollstersOptimize` c.18) hace `rating_adj = norm.cdf(-bias_dev_adj, scale=bias_dev_tau)`, `rating = (num_polls_w·rating_adj + min_polls·quality)/(num_polls_w + min_polls)` y `weight_rating = log1p(rating)/log1p(mean)`. Además `ratings_margin=.05` (`computer.py:50,128`) no se usa en ninguna parte. Para un repositorio divulgativo esta discrepancia es crítica. *Comprobar*: `grep -n "ratings_margin\|prior_weight\|dev_reverted" mtpy/lib/computer.py`.

**M5. Duplicación de lógica estadística en `lab/PollstersOptimize.ipynb` c.17-18.** ~60 líneas que replican `Computer.pollster_ratings` (`computer.py:1094-1190`); cualquier cambio en la librería deja el notebook desincronizado sin aviso. *Comprobar*: diff manual entre la celda 18 y `computer.py:1140-1186`.

**M6. Salidas comprometidas y stale.** Los notebooks se guardan con outputs embebidos (1-4,7 MB; hasta 16 PNG base64) y vía Git LFS (`.gitattributes: *.ipynb filter=lfs`), lo que impide ver diffs de código en GitHub y el renderizado en nbviewer. Además, los outputs de los notebooks raíz (29-dic-2025: última encuesta 2025-12-15) no reflejan la BD actual (encuestas hasta 2026-07-09 cargadas el 22-sep-2026; `PollsOptimize` de hoy muestra 2026-09-11), y `files/img/2708-*` mezcla figuras de mar-2025 y dic-2025.

**M7. Deprecaciones de pandas que romperán con pandas 3.** `forecaster.py:400-401` (`fillna(method='ffill')`, usado por `Simulator.fit_forecast` con `fillna=True`), `forecaster.py:470` (`applymap`, usado por `load_forecast` cuando `fc_fit=False`), `PollstersOptimize` c.14 (`applymap`). `warnings.filterwarnings('ignore')` en todos los notebooks oculta los `FutureWarning`. *Comprobar*: ejecutar `fc.load_forecast('fc')` sin filtrar warnings.

### Baja

**B1. Versionado de dependencias incoherente.** `requirements.txt` fija `numpy==1.25.2`, `scipy==1.9.3`, `matplotlib==3.8.0`, `nbconvert==6.4.4`, `ipython==7.29.0`; el entorno real tiene numpy 2.4.4, scipy 1.17.0, matplotlib 3.10.8. No hay `pyproject.toml`, lockfile ni `environment.yml`; el kernel de los notebooks es el genérico `python3`.

**B2. `sys.path.append` relativo en lugar de paquete instalable.** Cuatro variantes según la carpeta; falla si se ejecuta con papermill/nbconvert desde otro CWD. Solución: `pip install -e .` con `pyproject.toml` (o `PYTHONPATH`).

**B3. Constantes hard-coded en notebooks.** Encuestadoras destacadas `['CIS','GAD3','GESOP','40dB']` (`PollstersRatings` c.7, `PollsForecast` c.15), ids `[1,2,3,15]` en `events/*`, nota de autor `'Luis Sancho - @luis_sancho'`, ventana `days.between(6, 41)` (`PollsForecast` c.7) frente a `drange=(6,42)` en otros notebooks, `vmax=350` (escaños del Congreso, disponible en `sim.reg_totals`).

**B4. Código muerto / inconsistencias menores.** `PollsSimulations` c.1 carga `parties` y `provinces` y no los usa; c.5 `'Promedio de sondeos'.format(...)` sin marcador; `PollstersEvent` importa `Forecaster`, `norm_range`, `group_results` sólo para la rama `show_fc`; `PollsForecast` c.5 `d = fc.fc_series[['sample_size']]; d['poll'] = 1` (SettingWithCopy); `events/*/PollsAnalysis` tiene dos figuras por bloques (c.17-18) que se perdieron en `PollsForecast`.

**B5. `PollstersEvent` c.4 mezcla ventanas temporales.** `dp` (media de error por encuestadora) usa todas las encuestas del ciclo (CEMOP a 431 días) mientras `df` usa `n_last=1`; el panel "Mean Absolute Error" no es comparable con "Last Poll Absolute Error". Es una decisión de análisis que convendría explicitar.

**B6. `print_df(..., path=)` requiere `dataframe_image`** (`dataviz.py:762-…`), ausente en `requirements.txt`; ningún notebook pasa `path` a `print_df`, así que hoy no afecta.

**B7. Figuras huérfanas en `files/img/`.** `fig1_censo_mensual.png` … `fig_prima_bloque.png` (12 ficheros) y las variantes `-full`/`-min`/`-vs` no las genera ningún notebook del repositorio; `files/img/2307/` y `files/img/7706/` son restos de ejecuciones anteriores.

---

## 7. Recomendaciones para convertirlo en material reproducible y publicable

1. **Empaquetar y fijar el entorno**: `pyproject.toml` con `mtpy` instalable (`pip install -e .`), extras `[notebooks]`; lock (`uv lock`/`pip-compile`); kernel con nombre propio (`python -m ipykernel install --name elections-model`). Sustituir `np.NaN`→`np.nan`, `fillna(method=)`→`.ffill()`, `applymap`→`.map`; quitar `warnings.filterwarnings('ignore')` o limitarlo a categorías concretas.
2. **Desacoplar de la BD para lectura**: exportar las tablas del esquema `elections` (parties, pollsters, polls, polls_results, events*, pollsters_ratings) a Parquet/CSV versionados en `data/` (o en un release/Zenodo con DOI si pesan), y dar a `mtpy.lib.data` un backend de ficheros. Versionar `params.json`, `wp-*.json`, `es-provinces.csv` y los XLSX de infoelectoral (son públicos) fuera de `.gitignore`. Con ello los notebooks de análisis (etapas 4-7) se ejecutan sin PostgreSQL; los de carga (etapas 1-3) quedan como "mantenimiento".
3. **Un notebook canónico por etapa, parametrizado**: `01_load_results`, `02_load_polls`, `03_compute_ratings`, `04_pollsters_diagnostics`, `05_forecast`, `06_simulate`, `07_lab_*`. Marcar la celda de configuración con la etiqueta `parameters` (papermill) y ejecutar por evento con `papermill 05_forecast.ipynb out/05_forecast_2027-08-22.ipynb -p event_date 2027-08-22 -p bmap max`. Eliminar `events/*` (se regeneran con papermill) y `build/`.
4. **Separar escritura y lectura de BD**: parámetro `save=False` por defecto y un flag explícito (`--confirm-write`), o mover la carga a scripts CLI (`python -m mtpy.lib.loader wikipedia --event 2027-08-22 --save`).
5. **Mover la lógica de análisis a la librería**: agregaciones por semana/periodo, bins de %, conteos mensuales, tabla resumen de modos, resolución de ids de encuestadoras, y el bloque duplicado de `PollstersOptimize` (que debería importar `Computer.pollster_ratings` y sólo mostrar pasos intermedios). Añadir `Simulator.result_median()` (mediana por provincia) y corregir `totals()` (repartir el déficit sólo entre partidos con restos > 0 o renormalizar `v2seats`).
6. **Regeneración de figuras y CSV con un script** (`make figures` / `scripts/build_outputs.py`) que ejecute los notebooks con `nbclient`/papermill y escriba en `outputs/{event}/` con nombres deterministas; versionar sólo las figuras finales pequeñas (o ninguna) y el resto en releases.
7. **Publicación**: Jupyter Book o Quarto sobre los notebooks ejecutados (`_toc.yml` = las etapas; capítulos de metodología con las fórmulas de `stat.py` y `computer.py`), con `nbstripout` en pre-commit para no versionar outputs, y sin Git LFS para `.ipynb` (GitHub no renderiza notebooks LFS). Un `README` con el diagrama del flujo de la sección 2, y `CITATION.cff` ya existente.
8. **Tests mínimos** para que el material sea fiable: `alloc_dhondt` contra un caso conocido (p. ej. 2023 Madrid), `totals()` suma 350 y nunca asigna a partidos con `max=0`, `Kernel.get_bw_fixed` contra `scipy.stats.gaussian_kde` (scott/silverman), `save_forecast`/`load_forecast` round-trip.

---

## 8. Preguntas abiertas para el autor

1. ¿Los notebooks de `events/*` se conservan intencionadamente como registro histórico (backtest de 2019-11 y 2023-07) o pueden regenerarse con la versión actual? ¿Qué diferencias de modelo había en marzo-2025 respecto a hoy (p. ej. `max_fc=0` y `n_sim=10000` allí frente a `max_fc=10` y `n_sim=1000` ahora)?
2. ¿El déficit sistemático de escaños en LS/MC (Σ medianas ≈ 329) es conocido? ¿Prefieres renormalizar `v2seats` para que sume 350 o repartir sólo entre partidos con escaños > 0?
3. ¿La tabla provincia × partido que se quiere publicar en el modo MT debe ser la mediana por provincia, la simulación modal, o la de la simulación cuyo total nacional coincide con la mediana?
4. ¿Es aceptable versionar públicamente los datos de encuestas scrapeados de Wikipedia y los XLSX de infoelectoral (licencias), o prefieres un dump con DOI en Zenodo?
5. ¿Las figuras `fig1_censo_mensual.png` … `fig_prima_bloque.png` en `files/img/` proceden de otro proyecto y deben salir del repositorio?
6. ¿Qué encuestadoras deben aparecer como "destacadas" en las rejillas (`['CIS','GAD3','GESOP','40dB']`)? ¿Debería ser un campo de `pollsters` (p. ej. `featured`) en lugar de una constante en el notebook?
7. ¿Cuánto tarda hoy `PollsSimulations` con `n_sim=1000` en MT (52 provincias × 1000)? En mi prueba `n_sim=5` tardó 0,1 s, lo que sugiere que 10 000 es viable; ¿por qué se bajó de 10 000 a 1 000?
8. ¿`ratings_margin` (`computer.py:50`) es un resto de una versión anterior del rating "[5-95]" descrita en el docstring, o está pendiente de implementar?
9. ¿El objetivo divulgativo incluye que un lector reproduzca el forecast completo (necesita BD) o basta con que reproduzca las figuras a partir de CSV congelados?
