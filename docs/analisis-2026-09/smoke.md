# Smoke test del modelo electoral (ejecución real contra la BD local)

Fecha: 2026-09-23. Python `/Users/luiss/.pyenv/versions/3.11.9/bin/python` (3.11.9), numpy 2.4.4, pandas 2.2.2, scipy 1.17.0.
Todo en modo **solo lectura**: ningún `save=True`, ningún `save_*`, ninguna escritura en el repo. Scripts y logs completos en
`/private/tmp/claude-501/-Users-luiss-HD-Proyectos-Code-elections-model/751e110f-ed09-4efb-a617-82a9991382fd/scratchpad/runs/` (`step1.py`…`step7.py`, `step5b.py` y sus `.log`).

## Resumen ejecutivo

- **El flujo completo Computer -> Forecaster -> Simulator funciona de punta a punta** con numpy 2.4 / pandas 2.2 sin ninguna excepción en el camino nominal. Tiempos: `build_series` del Computer 0.6 s; ajuste del Forecaster ~0.65 s por partido; Simulator 13 partidos ~8 s de ajuste + 5 s por cada 200 simulaciones MT (≈25 s / 1000).
- **Las únicas roturas encontradas son (a) de uso del parámetro `path` y (b) de API eliminada en numpy 2 / pandas 2 en código periférico y en notebooks** (`np.NaN` en `PollsForecast.ipynb`, `PollstersEvent.ipynb`; `Series.append` en `dataviz.plot_prevalence`; `np.in1d` en `nlp.py`). Nada de esto está en `mtpy/lib/*.py` ni en `stat.py`.
- Hay bastantes `FutureWarning` de pandas en el camino nominal (`groupby(axis=1)`, `fillna(method=)`, `applymap`, downcasting) que **quedan ocultos** por `warnings.filterwarnings('ignore')` en `mtpy/mtpy.py:15`. Funcionan hoy, pero romperán en pandas 3.
- `requirements.txt` fija `numpy==1.25.2` y `scipy==1.9.3`; el entorno real tiene 2.4.4 / 1.17.0. Hay que actualizar los pins (el código ya corre en las versiones nuevas).

## Paso 0. Arranque: `mtpy.run()` y resolución de rutas

- `mtpy.run()`: 1.59 s. Conecta a PostgreSQL local y crea `app.fs = FileSystem(fspath)` porque `S3_BUCKET` está vacío en `.env` (`mtpy/mtpy.py:80-85`).
- `fspath = <repo>/mtpy/../files` (`mtpy/mtpy.py:25-37`; `shpath` cae al propio repo porque `/Users/luiss/HD/Proyectos/shared` no existe).
- `FileSystem.get_path(name)` (`mtpy/core/io.py:38-52`): si `name` empieza por `/` lo devuelve tal cual; si no, lo prefija con `fspath`. Por eso:

| `path` pasado al modelo | archivo que se abre | resultado |
|---|---|---|
| `'.'` (como hacen los notebooks) | `<repo>/mtpy/../files/./params.json` | OK |
| `'/Users/luiss/HD/Proyectos/Code/elections-model/files'` | idem absoluto | OK |
| `None` (por defecto -> `os.getcwd()`, `forecaster.py:106`, `simulator.py:106`, `computer.py:130`) | `<cwd>/params.json` | **FileNotFoundError** si cwd no es `files/` |
| `'files'` desde la raíz del repo | `<repo>/files/files/params.json` | **FileNotFoundError** |

Conclusión: **`path` es relativo a la raíz del `FileSystem` de la app (`files/`), no al cwd**. El docstring ("Path to store model files") no lo dice, y el valor por defecto `os.getcwd()` es incoherente con esa semántica (solo funciona si se ejecuta desde `files/`). Lo mismo ocurre con las imágenes: `plot_figure(path='img/x.png')` -> `save_figure` -> `App.get_().fs.write_bytes(..., path)` (`dataviz.py:741-747`), así que `img/...` acaba en `files/img/` sea cual sea el cwd. Recomendación: eliminar `path` de las tres clases (o hacer que por defecto sea `''`/`'.'`) y documentar que todo es relativo a `app.fs`.

## Paso 1. `Computer(scope='es').build_series()`

- `Computer.__init__`: 0.39 s (incluye `get_event_params`, `computer.py:132-136`). `build_series`: **0.62 s**.
- `comp.series`: **(3842, 511)** = 3825 encuestas + 17 eventos; índice `['event_date','date','pollster_id','sponsor_id']`; 23 columnas de datos (`computer.py:437-443`) + 488 partidos (incluido un partido literal llamado `'-'`, que existe como fila en `elections.parties`).
- `comp.errors` (3825, 488), `comp.biases` (3825, 85), `comp.ratings` **(1012, 16)** (cargados de `pollsters_ratings`; columnas `quality, num_events, num_polls, error_avg, error_blocks, bias_*, rating_adj, rating, weight_rating`). Se ven ratings de 1996 con `num_events=0` y `rating = quality` (prior puro).
- Encuestas por evento: 1977: 11 … 2023: 891, 2027: 603.

## Paso 2. `get_polls_metric(metric='error', bmap='vs', drange=(6,42), n_last=1)`

- 0.01 s. Resultado **(166, 10)**, índice `(event, pollster, days)`, columnas `sample_size, proc_sample, weight, error_avg, bias_avg, error_blocks, bias_blocks, bias, Derecha, Izquierda`.
- `merge_bmaps('vs')` -> `Derecha = [UCD, EDC, AP, CD, CDS, PP, Cs, VOX, SALF]`, `Izquierda = [PSOE, PCE, PSP, IU, UP, MP, SUMAR]`.
- Descriptivos: `error_avg` media 2.28 (sd 1.11, máx 6.88); `bias` media 12.87; error de bloque `Derecha` media +1.10, `Izquierda` -0.73 (las encuestas sobreestiman a la derecha y subestiman a la izquierda en la última encuesta de cada casa, 1993-2023).
- Anomalía de datos: `Target Point` 2023 tiene `sample_size = 0` y `proc_sample = 1004` (el `fillna(0)` de `computer.py:1447` convierte un `NaN` en 0).

## Paso 3. Estimadores

### 3a. `get_error_estimator(drange=(6,None), n_last=1)` — 0.14 s
- Datos: **1634 filas** (`date, pollster, party, regional, weeks, weight, color, pct, error`), `weeks` de 1 a **192** (con `drange=(6,None)` entran encuestas de hasta casi 4 años antes), `error` media 1.41 (máx 20.1).
- Tipo `mtpy.core.utils.stat.LeastSquaresEstimator`, `poly_deg=1`, `cov_type='hac'`, `nobs=1634`, `neff=1472.8`, `serr=1.454`. Coeficientes `[intercepto, pct, regional, weeks] = [1.2585, 0.0432, -1.1944, 0.0131]`.
- `predict([[pct, regional, weeks]])` devuelve `pd.Series`:

| pct | regional | weeks | error previsto |
|---|---|---|---|
| 30 | 0 | 1 | 2.57 |
| 10 | 0 | 1 | 1.70 |
| 3 | 0 | 1 | 1.40 |
| 3 | 1 | 1 | 0.21 |
| 30 | 0 | 6 | 2.63 |

### 3b. `get_seats_estimator()` — 0.10 s
- Datos: **187 filas** (partido-evento con `pct >= 0.1` y `seats > 0`, `computer.py:1631-1633`), peso `0.97^years`.
- Coeficientes `[intercepto, pct, regional] = [-16.95, 4.51, 15.71]`, `serr = 4.72` escaños.
- Predicciones: 30% -> 118.3; 20% -> 73.2; 10% -> 28.1; **3% nacional -> -3.4** (negativo); 3% regional -> 12.3; 0.5% regional -> 1.0.
- Es una recta con intercepto negativo: subestima a partidos nacionales pequeños (escaños negativos, que `simulate()` recorta con `.clip(0)` en `simulator.py:630`) y da +15.7 escaños fijos a cualquier partido marcado `regional=1`. Sirve como aproximación rápida para los modos LS/MC pero no es un modelo de escaños creíble; el modo DH (D'Hondt por provincia) es el que tiene sentido.

## Paso 4. `Forecaster(scope='es', event_date='2027-08-22', bmap='max')`

- `bmap='max'` -> `['PP','PSOE','VOX','SUMAR','UP','SALF']`. `reg_params` por defecto: kernel gaussiano, ancho de banda adaptativo ISJ (`n_iter=3`), `poly_deg=1`, covarianza HAC (Bartlett, 1 lag) (`forecaster.py:147-181`).
- `build_series`: 0.13 s. `series` (308, 31); `fc_series` 306 filas (2 filas van a `nfc_series`: el resultado del evento anterior 2023-07-23 y el propio evento); `fc_index` 1492 días (2023-07-23 → 2027-08-22). Fechas: primera encuesta 2023-08-07, última **2026-09-11**.
- Peso de cada encuesta = `weight_over * weight_sample * weight_rating` (`forecaster.py:271`), ej. CIS 2026-09-04: 1.0 × 1.76 × 1.92 = 3.379. Distribución de `weight`: media 2.36, máx 5.26; `weight_rating` media 2.09.
- `fit('PP', max_fc=3, ret_stat=True)`: **0.64 s**. 1135 días no nulos. Cola: 31.899 (09-09) … 31.915 (09-14 = última encuesta + 3 días). `dstat` por día: `{'mean','cmin','cmax','err','nobs','neff'}`, p. ej. 2026-09-14: media 31.91, IC95 [30.56, 33.27], `err` 0.676, `nobs` 101, **`neff` 48.4**.
- `fit_forecast(names=['PP','PSOE'], max_fc=3)`: 1.27 s. PSOE 27.09. Columna `'-'` = 100 − suma de los fitted (`forecaster.py:394`): con solo dos partidos vale 40.99 y `get_forecast(sort=True)` la devuelve como primer "partido" (`-  40.99`). Es correcto por construcción, pero engañoso si el ajuste es parcial.
- `fit_forecast(..., fillna=True)`: 0.74 s; rellena con `fillna(method='ffill')` (`forecaster.py:400-401`, deprecado, hoy funciona) hasta 2027-08-22.

## Paso 5. `Simulator(scope='es', event_date='2027-08-22', drange=6, seed=42)`

- `__init__`: **1.05 s** (construye un `Forecaster` interno con `bmap=names` de 13 partidos, un `Computer` con 14 eventos 1982-2023, `v2seats`, `v2err(drange=(6,None), n_last=1)`, `reg_totals` de 53 regiones (`es` + 52 provincias, 350 escaños) y `prev_results` (53, 39) de 2023).
- `limit_date = 2027-08-16` (evento − 6 días). `smap = {'UP': sub SUMAR, 'SALF': sub VOX}`.
- `fit_forecast(names=13, max_fc=3, fillna=True)`: **8.14 s**. Resultado (`mean / err / error`): PP 31.91 / 0.68 / 2.65; PSOE 27.09 / 0.50 / 2.44; VOX 17.41 / 0.24 / 2.02; SUMAR 6.07; UP 2.94; SALF 2.07; ERC 2.25; EHB 1.33; JxCat 1.18; PNV 0.95; BNG 0.81; CC 0.41; UPN 0.18. `error` viene de `v2err` con `weeks = (6+1)//7 = 1` (`simulator.py:311`).

| Modo | Llamada | Tiempo | PP | PSOE | VOX | SUMAR | ERC | EHB | JxCat | PNV | BNG | CC | UP | SALF | UPN | Σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LS | `run(split=False, random=False)` | 0.00 s | 128 | 107 | 63 | 12 | 11 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 1 | 350 |
| MC | `run(split=False, random=True, n_sim=20)` | 0.09 s | 123 | 112 | 65 | 13 | 11 | 7 | 6 | 5 | 3 | 2 | 1 | 1 | 1 | 350 |
| DH | `run(split=True, random=False)` | 0.02 s | 139 | 112 | 61 | 8 | 8 | 7 | 4 | 5 | 2 | 1 | 2 | 0 | 1 | 350 |
| MT | `run(split=True, random=True, n_sim=20)` | 0.50 s | 134 | 115 | 62 | 9 | 8 | 7 | 4 | 5 | 2 | 1 | 2 | 0 | 1 | 350 |
| MT | `run(split=True, random=True, n_sim=200)` | 5.0 s | 137 | 112 | 62 | 8 | 9 | 7 | 4 | 5 | 2 | 1 | 2 | 0 | 1 | 350 |

- MT n=200, cuantiles 2.5/50/97.5 de `dist()`: PP 115/136/155; PSOE 94/111/134; VOX 42/61/81; SUMAR 2/8/20; UP 0/2/6; SALF 0/0/4; ERC 7/8/12; EHB 6/7/9; JxCat 3/4/7; PNV 3/5/6.
- `result(0)` (MT): matriz 53×13 por provincia, fila `es` = suma de provincias = 350. `unit(0, region='Madrid')`: `prev_pct` vs `vpred_pct` (PP 40.55→40.25, VOX 13.33→20.04, SUMAR 9.50→6.05).
- **Reproducibilidad (`step5b.py`)**: dos `run(split=True, random=True, n_sim=10)` con `seed=42` dan `results` idénticos (`np.array_equal` True); con `seed=7` distintos; DH es determinista. El `rng` se reinicia en cada `set_params` (`simulator.py:229`), lo que es correcto.
- Diferencia LS vs DH: LS reparte escaños a partidos que el estimador lineal predice ≤0 (SALF 1, UPN 1) porque `totals()` fuerza la suma a 350 repartiendo el déficit por "ciclos" sobre todos los partidos (`simulator.py:411-422`). En DH esos ceros se respetan.

## Paso 6. Uso de `path='.'` desde `notebooks/`

Reproducido con cwd = `notebooks/`: `Forecaster(..., path='.')` **funciona** (0.14 s) porque `'./params.json'` se resuelve contra `files/`; `Forecaster(..., path=None)` **falla** (`FileNotFoundError: .../notebooks/params.json`). Ver tabla del Paso 0. Los notebooks son coherentes con la semántica real (`path = '.'` en `PollsForecast.ipynb:88`, `PollsSimulations.ipynb:86`, `PollstersRatings.ipynb:89`, `data-load/PollsCompute.ipynb:53`), pero el valor por defecto del código no lo es.

## Paso 7. Compatibilidad numpy 2.4.4 / pandas 2.2.2

Probado ejecutando cada llamada (`step7.py`):

| Llamada | Estado real | Dónde aparece en el repo |
|---|---|---|
| `np.NaN`, `np.float`, `np.int`, `np.product`, `np.alltrue`, `np.trapz`, `np.in1d`, `np.infty`, `np.Inf`, `np.float_`, `np.string_`, `np.unicode_`, `np.asfarray`, `np.cumproduct` | **AttributeError** | `np.in1d`: `mtpy/core/utils/nlp.py:220` (no es del modelo). `np.NaN`: **notebooks** `PollsForecast.ipynb` (líneas 428, 450, 484), `PollstersEvent.ipynb` (1031-1036), `events/es-201911/*`, `events/es-202307/*`, `build/EventsData.ipynb`. Ningún uso en `mtpy/lib` ni en `stat.py`. |
| `np.row_stack` | DeprecationWarning | no usado |
| `DataFrame.append` / `Series.append` | **AttributeError** | `mtpy/core/utils/dataviz.py:2379` (`plot_prevalence`, llamada desde `dataviz.py:3799`); no está en el flujo del modelo |
| `DataFrame.iteritems` | **AttributeError** | no usado |
| `fillna(method='ffill')` | funciona + FutureWarning | `forecaster.py:400-401` (camino nominal cuando `fillna=True`, que el Simulator necesita); `dates.py:1023,1025` |
| `groupby(level=0, axis=1)` | funciona + FutureWarning | `lib/utils.py:91` (`group_results`, camino nominal); `loader.py:145,150` |
| `applymap` | funciona + FutureWarning | `forecaster.py:472` (`load_forecast`), `forecaster.py:1031-1032` (plots); `dataviz.py:3269` |
| `is_categorical_dtype` | funciona + DeprecationWarning | no encontrado en el repo |
| `stack(level=0, future_stack=True)` | OK | `simulator.py:650` (ya adaptado a pandas ≥2.1) |
| downcasting en `fillna(False).astype(bool)` | FutureWarning | `helpers.py:860` (camino nominal) |
| `groupby(...)` con categóricas sin `observed=` | FutureWarning | `lib/data.py:112` (camino nominal, se emite ~15 veces por `Simulator.__init__`) |
| `np.random.seed(config.app.seed)` | OK (seed es `42` int en `config.json:16`) | `mtpy/mtpy.py:90`; fallaría con un string |

Todas las advertencias anteriores están silenciadas globalmente por `warnings.filterwarnings('ignore')` en `mtpy/mtpy.py:15`, ejecutado al importar `mtpy.mtpy`. En `step5.py` reactivé los warnings tras `mtpy.run()` y aparecen (ver `step5.log` líneas 1-39).

## Excepciones registradas (todas provocadas a propósito; ninguna en el flujo nominal)

1. `Forecaster(..., path=None)` con cwd = `notebooks/` -> `FileNotFoundError: [Errno 2] No such file or directory: '/Users/luiss/HD/Proyectos/Code/elections-model/notebooks/params.json'`. Causa: `self.path = path or os.getcwd()` (`forecaster.py:106`) produce una ruta absoluta que `FileSystem.get_path` respeta tal cual. Arreglo: por defecto `''`/`'.'` (relativo a `app.fs`) o resolver `params.json` vía `app.fspath`.
2. `Forecaster(..., path='files')` con cwd = raíz -> `FileNotFoundError: .../mtpy/../files/files/params.json`. Misma causa; arreglo: documentar que `path` es relativo a `files/`.
3. `np.NaN` -> `AttributeError: np.NaN was removed in the NumPy 2.0 release` — afecta a celdas de `PollsForecast.ipynb` y `PollstersEvent.ipynb`. Arreglo: `np.nan`.
4. `pd.Series.append` -> `AttributeError` — `dataviz.py:2379`. Arreglo: `pd.concat([s, pd.Series({...})])`.
5. `np.in1d` -> `AttributeError` (eliminado en numpy 2.4) — `nlp.py:220`. Arreglo: `np.isin`.

## Recomendaciones derivadas de la ejecución

1. Actualizar `requirements.txt` a las versiones con las que realmente funciona (numpy ≥2, pandas 2.2, scipy 1.17) y corregir los 5 usos de API eliminada; sustituir `groupby(axis=1)`, `fillna(method=)`, `applymap` por sus equivalentes antes de pandas 3.
2. Quitar el `warnings.filterwarnings('ignore')` global (o limitarlo a un módulo) para que un repositorio científico muestre las deprecaciones.
3. Unificar la semántica de `path` (relativa a `app.fs`), eliminar el default `os.getcwd()` y documentarlo; es la única forma de que `Forecaster()`/`Simulator()` funcionen fuera de `files/` sin sorpresas.
4. Hacer explícito en `Simulator.fit_forecast` que `fillna=True` es necesario (el `build_forecast` lee `forecast.loc[limit_date]`, `simulator.py:315-317`, y `limit_date` está ~11 meses después de la última encuesta).
5. Documentar que los modos LS/MC usan un estimador lineal `seats ~ pct + regional` que da escaños negativos a partidos pequeños y +15.7 fijos a regionales; presentar DH/MT como el resultado del modelo y LS/MC como baseline.
6. Convertir estos scripts (`step1.py`…`step5b.py`) en tests de humo (`pytest`) con fixtures pequeñas (n_sim ≤ 20, 1-2 partidos): hoy no existe ningún test.
