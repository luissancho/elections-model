# Módulo `Computer`: construcción de la serie y cálculo de pesos

Fichero principal: `/Users/luiss/HD/Proyectos/Code/elections-model/mtpy/lib/computer.py` (líneas 1-640 y 1192-1330), con apoyo de `mtpy/lib/data.py`, `mtpy/lib/utils.py`, `mtpy/core/utils/stat.py` (clase `Stat`) y `mtpy/core/utils/helpers.py` (`apply_agg_func`).

Todas las afirmaciones sobre datos proceden de consultas de solo lectura a la base de datos local (esquema `elections`) y de una re-ejecución en memoria de `Computer(scope='es').build_series().compute_weights(save=False, overwrite=True)` realizada el 2026-09-23 (3.825 encuestas, 17 eventos 1977-2027, 95 encuestadoras).

---

## 1. Visión general

`Computer` es la primera pieza del modelo. Su responsabilidad en este módulo es:

1. **Construir una única tabla larga (`self.series`)** que contiene, con el mismo esquema de columnas, todas las encuestas publicadas para todas las elecciones del ámbito (`scope='es'`) y, como filas adicionales, el resultado final de cada elección.
2. **Calcular tres pesos por encuesta** (`weight_sample`, `weight_over`, `weight_rating`) que luego el `Forecaster` multiplica para obtener el peso final de cada encuesta en la regresión local (`forecaster.py:271`: `weight = weight_over * weight_sample * weight_rating`).
3. **Calcular tres pesos adicionales por encuesta** (`weight_pos`, `weight_week`, `weight_year`) que sólo se usan internamente para ponderar los errores históricos al calcular el rating de cada encuestadora (`computer.py:1116-1120`).

Los pesos "de posición temporal" no forman parte del peso que ve el `Forecaster`: la cercanía temporal la gestiona allí el kernel de la regresión local. Esto contradice la frase del docstring de clase ("proximity to election date", `computer.py:56-59`), que sugiere que el propio `Computer` pondera por proximidad a la elección.

### Comparación con FiveThirtyEight

El esquema es reconocible como el de Nate Silver: peso de encuesta = f(tamaño muestral, rating de la encuestadora, redundancia de encuestas de la misma casa), y rating de la encuestadora basado en el error relativo a las demás encuestas del mismo evento (el "Predictive Plus-Minus"). Diferencias sustantivas:

- 538 usa una ventana fija (encuestas de los últimos 21 días antes de la elección) para calificar; aquí se usa toda la serie pero con decaimientos exponenciales `0.5^pos · 0.7^semanas · 0.9^años` y un umbral de 0.01, lo que **en la práctica** reduce el conjunto a las 1-3 últimas encuestas de cada casa en las ~6 semanas finales (343 de 2.238 encuestas filtradas; ver §4.4).
- 538 penaliza el *herding* y ajusta por *house effects* al promediar; aquí no hay penalización por herding y el sesgo de casa se trata sólo a través del rating (no se corrige el nivel de la encuesta).
- 538 combina las encuestas repetidas de un mismo encuestador con un peso conjunto decreciente; aquí `weight_over` descarta las que se solapan en campo (trackings) y reparte `1/N` en una ventana de ±`wspan` días.
- El peso muestral es `sqrt(n/mediana)` sin tope explícito (el tope viene de la winsorización global, ver §4.1); 538 usa una forma cóncava similar con techo.

---

## 2. Constructor y parámetros por defecto (`computer.py:32-153`)

| Parámetro | Defecto | Uso real en el código |
|---|---|---|
| `scope` | — | filtro `event_scope`/`scope` en SQL |
| `event_dates` | `get_event_dates(scope)` → las 17 fechas | lista de eventos cargados |
| `drop_mtypes` | `['aggr','online']` | `filter_polls` (ratings/errores) y `msizes_all` (`computer.py:556`) |
| `drop_ctypes` | `['wban','exit']` | sólo `filter_polls` |
| `drange` | `None` → `(0, max(days))` | `filter_polls` (`norm_range`, `utils.py:105-131`) |
| `n_last` | `None` → 0 (desactivado) | `filter_polls` (`groupby.tail`) |
| `alpha` | 0.05 | no se usa en este módulo |
| `reg_params` | ver `set_reg_params` | estimadores de error (otro módulo) |
| `ol_dev` | 3 | nº de desviaciones típicas de la winsorización de `proc_sample` |
| `ol_max` | 5000 | `clip` superior del tamaño muestral antes de winsorizar |
| `wspan` | 4 | ventana en días para la penalización por sobrepublicación |
| `pos_decay` | 0.5 | `weight_pos = 0.5^seq_pos` |
| `week_decay` | 0.7 | `weight_week = 0.7^weeks` |
| `year_decay` | 0.9 | `weight_year = 0.9^years` |
| `bias_dev_tau` | 0.01 | prior del rating (otro módulo) |
| `min_polls` | 3 | pseudo-conteo del rating (otro módulo) |
| `ratings_margin` | 0.05 | no se usa en el código leído |
| `error_weights` | `{'avg': .7, 'blocks': .3}` | métrica `bias` (otro módulo) |
| `path` | `os.getcwd()` | ruta del `params.json` (leída vía `App.fs`, relativa a `files/`) |

Atributos de estado: `keys = ['event_date','date','pollster_id','sponsor_id']` (`computer.py:143`) es el índice de la serie; `series`, `errors`, `biases`, `ratings`, `parties`, `pollsters`, `names`.

`event_params` se construye en el constructor con `get_event_params` (`data.py:53-139`): lee `files/params.json[scope]` y, para cada evento, completa por defecto `parties.event/polls`, `bmaps.main` (partidos presentes en >95 % de las encuestas y en el resultado), `bmaps.blocks`, `bmaps.vs` y `smap` (partidos nuevos). Sólo los eventos presentes en `params.json` (2019-04, 2019-11, 2023, 2027) tienen `bmaps.max`/`bmaps.min`.

`set_reg_params` (`computer.py:214-248`) normaliza los parámetros del `LocalKernelEstimator`: kernel gaussiano, ancho de banda adaptativo con selector ISJ y 3 iteraciones, polinomio local de grado 1, covarianza HAC (Bartlett, 1 retardo). Nota: si el usuario pasa `bw_kwargs` o `cov_kwargs` parciales se sustituye el diccionario completo, no se mezcla.

---

## 3. Construcción de la serie

### 3.1 Cargas (`computer.py:308-399`)

- `load_events` → `get_event_series` (`data.py:195-250`): `Events ⋈ EventsData(region_id=0) ⋈ EventsResults(region_id=0)` pivotado a una columna por partido con el `metric` (`pct` por defecto). Añade `event_date = date`, `sample_size = votes`, `computed = featured`. Índice `date`.
- `load_polls` → `get_poll_series` (`data.py:277-327`): `Polls ⋈ PollsResults` pivotado por `party`. Índice `keys`. Las columnas de partidos son los **nombres** (`Parties.name`), no ids.
- `load_errors` / `load_biases`: la misma consulta con `metric='error'` / `'bias'` (columnas de `PollsResults`).
- `load_ratings` → `get_ratings` (`data.py:330-365`): añade automáticamente el **siguiente evento** (`get_next_event_date`) para que el evento futuro tenga sus ratings. Índice `(event_date, pollster_id)`.
- `get_parties` (`data.py:368-372`) mueve la fila 0 (partido `-` = "Otros") al final, de modo que `self.names` termina en `'-'`.

### 3.2 `build_series` (`computer.py:401-474`)

```
series = concat([polls.reset_index(), events.reset_index()])      # filas de eventos: pollster/sponsor NaN
pollsters = pollsters[name ∈ series.pollster]                     # 92 casas con encuestas
parties   = parties[name ∈ series.columns]                        # 488 partidos (incluido '-')
names     = parties.name.tolist()
series    = series.set_index(keys).sort_index()[data_cols + names]
```

- Forma resultante: **3.842 × 511** (3.825 encuestas + 17 eventos; 23 columnas de datos + 488 columnas de partidos, casi todas NaN). Índice `MultiIndex(event_date, date, pollster_id, sponsor_id)`; en las filas de eventos `date == event_date` y `pollster_id`/`sponsor_id` son NaN.
- `days` (columna de `Polls`) **no se calcula aquí**: viene del cargador, `loader.py:649`: `days = (event_date - date).days`, y `date` es la **fecha de fin de campo** (`loader.py:417`: `data['date'] = data['end_date']`; en la BD `date == end_date == pub_date` en el 100 % de las filas). Para las filas de eventos `days` es NaN.
- Partidos sin errores calculados (no concurrieron a ningún evento con resultado) se añaden a `self.errors` como NaN (`computer.py:466-472`) para evitar `KeyError`.
- Columna `'-'`: es un partido real de la tabla (`id=0`, nombre `-`, fullname "Otros"). En las encuestas es el resto no asignado; el `Forecaster` la recalcula como `100 - suma(bloques)` (`forecaster.py:298`).
- Propiedades: `events` = filas con `pollster` nulo, índice reducido a `event_date` (`computer.py:156-166`); `polls` = filas con `pollster` no nulo (`computer.py:168-177`).

### 3.3 `merge_bmaps` (`computer.py:179-212`)

Une los `bmaps[name]` de todos los eventos en un único diccionario `partido → bloque(s)`. Si el bmap es una lista se convierte en `{p: p}`. Tiene tres problemas verificados (ver F4): itera cadenas carácter a carácter, lanza `KeyError` si algún evento no define el bmap y muta `self.event_params` por aliasing de listas.

### 3.4 `filter_polls` (`computer.py:250-306`)

Selecciona encuestas para los cálculos de errores/ratings:

```
pollster not null  ∧  mtype ∉ drop_mtypes  ∧  ctype ∉ drop_ctypes
∧ weight_over > 0  ∧  days ∈ [drange_lo, drange_hi]
[∧ featured ∧ computed]        # featured=True por defecto
[groupby(event_date, pollster_id).tail(n_last)]
```

- `weight_over > 0` excluye también las encuestas con `weight_over` NaN (no calculadas) de forma silenciosa.
- Con `featured=True` se excluyen los eventos pre-1993 y el evento futuro 2027 (`featured=False`). Resultado actual: 2.238 encuestas de 11 eventos.
- `n_last` depende de que el índice esté ordenado por `date` dentro de cada evento (lo está por `keys`).

---

## 4. `compute_weights` (`computer.py:476-635`)

Entrada: `self.polls` (sin las columnas a recalcular). Con `overwrite=False` se restringe a `~computed` (`computer.py:530-531`); actualmente **todas** las encuestas de la BD están `computed=True`, y el notebook `PollsCompute` llama siempre `compute_weights(save=save, overwrite=True)`.

### 4.1 `mtype` y `proc_sample` (`computer.py:539-572`)

1. `mtype` se asigna **por encuestadora** (`Pollsters.mtype`), no por encuesta: `df['mtype'] = pollster.map(pollsters.mtype)`.
2. `sample_size <= 0 → NaN`. En la BD hay 744 nulos (19,5 %), ningún cero.
3. Imputación en cascada: mediana por `(event_date, pollster)` → mediana por `pollster` (todos los eventos) → mediana global de las encuestas cuyo `mtype ∉ drop_mtypes`. (Sólo el último escalón excluye `aggr`/`online`; los dos primeros no.)
4. Recorte y winsorización:

```
proc_sample = Stat(clip(proc_sample, 0, ol_max), dropna=True, outliers='group', n_dev=ol_dev).data
```

`Stat.winsorize(action='group')` (`stat.py:119-134`) calcula `μ ± n_dev·σ` sobre **toda la población agrupada (1977-2027)** con media y desviación típica no ponderadas y sustituye los valores fuera del intervalo por `max(data[dentro])`. Con los datos actuales: `μ = 1662.8`, `σ = 1054.7`, `μ+3σ = 4826.9`, `μ−3σ = −1501` (nunca activo). Consecuencia: el **tope efectivo es 4.715** (mayor valor observado ≤ 4826.9), no 5.000; 192 encuestas quedan en ese tope (153 de ellas tenían muestra bruta > 4.715). El tope es por tanto un estadístico de la muestra y cambiará al añadir encuestas.

### 4.2 `weight_sample` (`computer.py:576-583`)

```
weight_sample = round( sqrt( proc_sample / median_{event}(proc_sample) ), 2 )
```

La mediana es **por evento** (`groupby('event_date')`). Por construcción la mediana de `weight_sample` en cada evento es 1.0 (verificado en los 17 eventos). Rango observado: 0.58-2.17. Interpretación: el peso relativo es proporcional al error estándar inverso (∝ √n), como en 538, pero normalizado a la mediana del evento en lugar de a un valor absoluto, de manera que el nivel de tamaños muestrales de cada época no altera la escala.

Supuesto estadístico implícito: se trata a todas las encuestas como muestras aleatorias simples; no hay *design effect* para paneles online (que sólo se excluyen del cálculo de ratings, no del `Forecaster`).

### 4.3 `weight_over` (`computer.py:584-599` y `poll_overweights`, `computer.py:1275-1322`)

Preparación:

```
wrange = ((start_date − event_date).days, (end_date − event_date).days)   # enteros negativos
wtype  = factorize(mtype + '-' + ctype)                                    # como mtype es por casa, separa de hecho por ctype
```

`apply_agg_func(df, by=['event_date','pollster_id','wtype'], func=poll_overweights, columns='wrange', sort=['date','sponsor_id'])` (`helpers.py:1726-1830`) agrupa y aplica la función a la secuencia de tuplas de cada grupo. **El patrocinador no forma parte del grupo**: dos encuestas de la misma casa con distinto sponsor compiten entre sí.

Algoritmo `poll_overweights(x)`:

```
w = 1 para todas
sx = x ordenado por fecha de fin (np.argsort, no estable)
# (1) descarte por solapamiento
para i: si existe j>i (en orden de fin) con start_j <= end_i  →  w[i] = 0
# (2) penalización por sobrepublicación (si wspan no es None)
para i con w[i] > 0:
    n_prev = #{k<i : end_k >= end_i − wspan ∧ w[k]>0}
    n_post = #{j>i : end_j <= end_i + wspan ∧ start_j > end_i ∧ w[j]>0}
    w[i] = 1 / (n_prev + n_post + 1)
```

- (1) elimina los resultados parciales de trackings (cada publicación cuyo campo se solapa con una publicación posterior queda a 0). Distribución actual: 477 encuestas con `weight_over = 0` (110 en 2027), 3.253 con 1.0, y valores 1/2 … 1/9 en 94 casos.
- (2) es una ventana **centrada en cada encuesta** (±`wspan` días, usando fechas de fin de campo), por lo que la suma de pesos en un periodo no está normalizada a 1 (ej. tres encuestas en días 0, 3, 6 con `wspan=4` reciben 1/2, 1/3, 1/2).
- Empates exactos (misma casa, mismo campo, distinto sponsor): el orden de `np.argsort` decide cuál sobrevive; hay 3 casos en la BD (uno de ellos, `Data10` 2025-03-28, es en realidad el mismo sponsor escrito `Okdiario`/`OKDiario`).
- El docstring ("If N polls are published within a `wspan` days range, a weight of 1/N will be assigned") es coherente si N incluye la propia encuesta.

### 4.4 `weight_rating` (`computer.py:603-611`)

Simple *left join* de `self.ratings['weight_rating']` por `(event_date, pollster_id)`. El valor se calcula en `pollster_ratings` (`computer.py:1178-1181`):

```
weight_rating = log1p(rating) / log1p( mean_pollsters(rating) ),   rating ∈ [0,1]
```

(en la BD `rating` está reescalado a 0-100; `weight_rating` se guarda tal cual, rango observado 0-3.08, media ≈ 0.92 por la desigualdad de Jensen). Los ratings de un evento se calculan **sólo con encuestas de eventos anteriores** (`compute_ratings`, `computer.py:927-947`), por lo que:

- El primer evento con historial (1993) y los cinco pre-1993 no tienen rating → `weight_rating` NaN en 188 encuestas.
- `rating = 0` ⇒ `weight_rating = 0` ⇒ la encuesta se anula por completo en el `Forecaster` (producto). Ocurre en 33 filas de ratings (3 casas con `quality = 0`), incluidos 23 sondeos de `KeyData` para 2027.
- Para encuestadoras sin rating el `Forecaster` (`forecaster.py:270`) rellena `weight_rating` con `Pollsters.quality`, que está en escala **0-78** frente a un peso típico ≈ 1.

Finalmente `computed = True`, `featured = event.featured` y se devuelven/actualizan `['mtype','proc_sample','weight_sample','weight_over','weight_rating','computed','featured']` (`computer.py:616-633`). Con `save=True`, `save_model_data('Polls', …)` (`data.py:381-420`) primero pone a `NULL` esas columnas en **todas** las encuestas de los eventos afectados y luego hace `upsert` sólo de las filas de `df`.

Verificación de reproducibilidad: la re-ejecución con `overwrite=True` reproduce exactamente (`max |Δ| = 0`) los valores almacenados de `proc_sample`, `weight_sample`, `weight_over` y `weight_rating`.

---

## 5. Pesos para el rating: `poll_rating_weights` (`computer.py:1192-1231`)

Se aplica a las encuestas ya filtradas (`filter_polls`) dentro de `pollster_ratings` (`computer.py:1116`) y en `print_rating_weights` (`computer.py:1988`).

```
seq_pos     = posición desde el final dentro de (event_date, pollster_id)   # 0 = última encuesta de la casa
weight_pos  = pos_decay ^ seq_pos                = 0.5^seq_pos
dmin_e      = min_{encuestas del evento e}(days)  # 6-11 días por la veda electoral española
weeks       = max(0, floor((days − dmin_e + 1) / 7))
weight_week = week_decay ^ weeks                 = 0.7^weeks
years       = max_year(eventos incluidos) − year(event_date)
weight_year = year_decay ^ years                 = 0.9^years
todos redondeados a 2 decimales
```

Peso final de cada encuesta en el rating (`computer.py:1119-1123`): `w = weight_over · weight_sample · weight_pos · weight_week · weight_year`, y se eliminan las encuestas con `w < 0.01`.

Observaciones cuantitativas (datos actuales, 2.238 encuestas filtradas):

- Sólo **343 encuestas (15 %)** superan el umbral; conservan el 99,5 % del peso total exacto, así que el umbral es coherente, pero muestra que los decaimientos son muy agresivos: los supervivientes son `seq_pos ≤ 4` (170 con `seq_pos = 0`, 110 con 1) y `weeks ≤ 11` (126 en la semana 0).
- El redondeo a 2 decimales **no** es la causa (342 supervivientes sin redondear).
- 53 de las 68 casas del filtro conservan alguna encuesta; el rating de las otras 15 se apoya únicamente en el prior `quality`.
- `weeks` es relativo a la encuesta más cercana de cada evento, no al día de la elección: la "semana 0" abarca 6 días (`+1`) y su significado varía entre eventos (`dmin` 6-11).
- `years` usa el año natural: 2019-04 y 2019-11 reciben el mismo peso; 2015-12 y 2016-06 difieren en un factor 0.9. El año de referencia es el último evento **del subconjunto histórico**, no el evento objetivo (para el rating de 2027 las encuestas de 2023 pesan 1.0).

`lor_to_bias`/`bias_to_lor` (`computer.py:1233-1273`): conversión entre *log odds ratio* y "sesgo" porcentual, `bias = (e^x − 1)·100`, `x = log(bias/100 + 1)`. Se usan para almacenar sesgos en escala legible (%) y calcular en escala logarítmica.

---

## 6. Pipeline ordenado

1. `Computer.__init__` → `get_event_dates`, `get_event_params` (`params.json` + defaults) — `computer.py:107-153`, `data.py:53-139`.
2. `build_series` → cargas SQL (`load_events/polls/errors/biases/ratings`, `get_parties`, `get_pollsters`) — `computer.py:401-431`.
3. Concatenación encuestas+eventos, filtrado de `pollsters`/`parties`, ordenación de columnas — `computer.py:439-472`.
4. `compute_weights`: selección `~computed` si `overwrite=False` — `computer.py:525-534`.
5. `mtype` por casa; imputación de `sample_size` en cascada; `clip(0, ol_max)`; winsorización global `μ±3σ` → `proc_sample` — `computer.py:539-572`.
6. `weight_sample = sqrt(proc_sample / mediana_evento)` — `computer.py:576-583`.
7. `wrange`, `wtype`; `poll_overweights` por `(event, pollster, wtype)` → `weight_over` — `computer.py:585-599`, `1275-1322`.
8. Join de `ratings.weight_rating` — `computer.py:603-611`.
9. `computed/featured`; `save_model_data` opcional; actualización de `self.series` — `computer.py:616-635`.
10. (Módulo ratings) `filter_polls` → `poll_rating_weights` → `w = over·sample·pos·week·year`, umbral 0.01 → `pollster_ratings` → `weight_rating` — `computer.py:250-306`, `1192-1231`, `1116-1123`, `1178-1181`.

---

## 7. Hallazgos

### F1 (alta, bug) `save=True` con `overwrite=False` borra los pesos de las encuestas ya calculadas
`data.py:404-411` ejecuta `UPDATE polls SET mtype=NULL, proc_sample=NULL, weight_*=NULL, computed=NULL, featured=NULL WHERE event_date IN (...)` para **todas** las filas de los eventos presentes en `df`, y después `upsert(df)` sólo con las filas nuevas (`computer.py:618-625`). Las encuestas antiguas del mismo evento quedan con `computed=NULL` y sin pesos. Impacto: pérdida silenciosa de datos si alguien usa la opción incremental documentada. Cómo probar: sobre una copia de la BD, marcar 1 encuesta de 2027 como `computed=false`, ejecutar `compute_weights(save=True)` y contar `SELECT count(*) FROM elections.polls WHERE event_date='2027-08-22' AND weight_sample IS NULL`. Arreglo: restringir el `UPDATE` a las claves de `df`, o eliminarlo (el `upsert` ya sobrescribe), o forzar `overwrite=True` cuando `save=True`.

### F2 (media, bug/reproducibilidad) El modo incremental produce pesos distintos a los del recálculo completo
Con `overwrite=False`, las medianas de imputación, la winsorización, la mediana por evento y la detección de solapamientos se calculan **sólo sobre las encuestas nuevas** (`computer.py:531` y todo lo que sigue usa `df`). Verificado marcando las 60 últimas encuestas de 2027 como no calculadas: 22/59 `weight_sample` cambian (hasta 0.07), 1 `proc_sample` cambia en 153 unidades; además una encuesta nueva que solape con una ya calculada no la pone a 0. Cómo probar: script `recompute2.py` sección (b). Arreglo: calcular siempre los estadísticos sobre `self.polls` completo y escribir sólo las filas nuevas.

### F3 (media, bug en librería) `Stat.winsorize(action='group')` sustituye también los valores bajos por el máximo
`stat.py:131-132`: `np.where(x_in, data, np.max(data[x_in]))`. Un valor por debajo de `μ−nσ` se convierte en el máximo del rango. Verificado: `Stat([1,100,…,100], outliers='group', n_dev=1).data[0] == 100`. En `proc_sample` es latente (`μ−3σ < 0`), pero la función es de la librería estadística "divulgativa". Arreglo: `np.clip(data, min(data[x_in]), max(data[x_in]))`.

### F4 (media, bug) `merge_bmaps` itera cadenas carácter a carácter, falla si falta el bmap y muta `event_params`
`computer.py:200-211`. Verificado con `event_dates=['2019-11-10','2023-07-23','2027-08-22']`: `merge_bmaps('min')` devuelve `{'PP': ['PP','Cs','P'], …}` (partido fantasma `'P'` porque `'PP'` en 2027 es `str`); `merge_bmaps('max')` lanza `KeyError: 'max'` con el conjunto completo de eventos (sólo 2 eventos lo definen); y tras `merge_bmaps('vs')` el bloque `Derecha` de 1977 en `self.event_params` pasa de 3 a 9 partidos (aliasing `bm[party] = block`). Hoy es inocuo porque `compute_errors` sólo usa `main`/`vs` y `poll_errors` usa las claves, pero cualquier uso de `bmap='max'|'min'` en `Computer` fallará. Arreglo: normalizar `str → [str]`, copiar listas (`list(block)`) y usar `.get(name, {})`.

### F5 (media, estadística) `weight_rating = 0` excluye por completo a las casas con `rating = 0`
`computer.py:1181` (`log1p(0) = 0`) y `forecaster.py:271` (producto). 3 encuestadoras con `quality = 0` tienen rating 0 en 33 filas de ratings; en 2027, 23 encuestas de `KeyData` pesan 0 en el promedio. Es una exclusión dura no documentada. Cómo probar: `SELECT pollster_id, count(*) FROM elections.polls WHERE event_date='2027-08-22' AND weight_rating=0 GROUP BY 1`. Si es intencionado, documentarlo; si no, usar un suelo (`rating_floor`) o `weight_rating = (rating + ε) / (mean + ε)`.

### F6 (media, bug latente en consumidor) El `Forecaster` rellena `weight_rating` ausente con `quality` en escala 0-100
`forecaster.py:270`: `weight_rating.fillna(pollster_id.map(pollsters.quality))`. `quality` va de 0 a 78 (BD), `weight_rating` de 0 a 3.08. Hoy no se dispara para 2027 (todas las casas tienen rating porque `pollster_ratings` califica a todas las de `self.pollsters`, aunque sin historial reciban sólo el prior), pero afecta a las 188 encuestas pre-1996 y a cualquier casa nueva si el rating no se recalcula antes de pronosticar. Cómo probar: `Forecaster(scope='es', event_date='1993-06-06').build_series().series.weight_rating.describe()`. Arreglo: `fillna(log1p(quality/100) / log1p(mean_rating))` o recalcular ratings.

### F7 (baja, diseño estadístico) Los decaimientos del rating reducen el conjunto efectivo a ~15 % de las encuestas
`computer.py:1209-1226` y umbral `computer.py:1123`. 343 de 2.238 encuestas (99,5 % del peso) sobreviven; en la práctica el rating usa las 1-3 últimas encuestas de cada casa en las ~6 semanas finales de cada elección, y 15 de 68 casas quedan sin ninguna encuesta (rating = prior). Es defendible (538 usa 21 días) pero debería documentarse y, para un repositorio científico, justificarse con una calibración (¿`PollstersOptimize`?). Cómo probar: script `recompute2.py` sección (c).

### F8 (baja, diseño) `weeks` es relativo a la encuesta más cercana de cada evento y tiene un `+1` que acorta la semana 0
`computer.py:1217-1219`: `weeks = ((days − dmin_e + 1) // 7).clip(0)`. La semana 0 dura 6 días y su origen varía entre eventos (`dmin` 6-11). Cómo probar: `((np.arange(6,20) - 6 + 1)//7)` → `[0]*6 + [1]*7 + …`. Sugerencia: `weeks = (days − dmin_e) // 7` o, mejor, semanas absolutas hasta la elección.

### F9 (baja, diseño) `weight_year` usa el año natural, no la distancia entre eventos
`computer.py:1223-1226`. 2019-04 y 2019-11 tienen el mismo peso; 2015-12 vs 2016-06 difieren un 10 %. Además el año de referencia es el último evento del subconjunto histórico (para 2027, las encuestas de 2023 pesan 1.0 en vez de `0.9^4`). Cómo probar: `print_rating_weights()`. Alternativa: decaimiento por número de elecciones transcurridas o por días hasta el evento objetivo.

### F10 (baja, bug) Empates de campo entre sponsors se resuelven de forma arbitraria
`computer.py:1301-1310`: `np.argsort` (quicksort, no estable) decide qué encuesta sobrevive cuando dos publicaciones de la misma casa tienen idéntico `(start, end)`. 3 casos en la BD (1977 Metra Seis, 2011 Celeste Tel, 2027 Data10 `Okdiario`/`OKDiario`, este último un sponsor duplicado). Cómo probar: script `recompute2.py` sección (d). Arreglo: `np.argsort(..., kind='stable')` tras ordenar por `sort` y, para empates exactos, promediar o conservar la de sponsor conocido.

### F11 (baja, docs) Docstrings que no describen el código
- Docstring de clase (`computer.py:56-59`): el peso incluye "proximity to election date"; `compute_weights` no tiene componente temporal.
- `compute_ratings` (`computer.py:857-899`) describe fórmulas (`prior_weight = quality·num_polls/(num_polls+mean)`, `dev_reverted`, "scaled to the range [5-95]") que el código no implementa (`pollster_ratings` usa `norm.cdf(−dev, scale=tau)` y `(n·adj + min_polls·quality)/(n + min_polls)`, `computer.py:1166-1176`).
- `__init__`: `pos_decay`/`week_decay`/`year_decay` sólo dicen "Position decay" etc.; `ratings_margin` y `alpha` no se usan en el módulo.
- `compute_weights` docstring (`computer.py:483-486`) omite que la mediana de imputación global excluye `drop_mtypes` mientras las otras dos no (`computer.py:551-556`).

### F12 (baja, reproducibilidad) El tope de `proc_sample` es un estadístico de toda la población 1977-2027
`computer.py:567-572`. El tope efectivo (4.715) depende de `μ+3σ` sobre todas las encuestas de todos los eventos: cada carga de encuestas nuevas cambia los pesos de las antiguas al recalcular. Cómo probar: comparar `proc_sample.max()` antes y después de añadir encuestas ficticias. Sugerencia: tope fijo (`ol_max`) o winsorización por evento/época, documentando el valor.

### F13 (baja, edge case) `proc_sample` falla si no queda ninguna muestra
`computer.py:567-572`: si `msizes_all` es NaN (ningún tamaño muestral en `df`), `Stat(..., dropna=True).data` tiene menos filas que `df` y la asignación/`astype(int)` falla. Sólo en cargas nuevas con muestras desconocidas. Cómo probar: `Computer` sobre un evento cuyas encuestas no tengan `sample_size`.

### F14 (baja, rendimiento/diseño) Serie ancha y bucles fila a fila
`series` es 3.842 × 511 con 488 columnas de partidos casi vacías; `weight_sample` y `wrange` se calculan con `df.apply(axis=1)` (`computer.py:577-583`, `585-588`) y `poll_overweights` es O(n²) por grupo. Aceptable hoy (segundos) pero conviene vectorizar (`np.sqrt(df.proc_sample / df.event_date.map(medians))`) y almacenar resultados en formato largo para un repositorio público.

### F15 (baja, reproducibilidad) Dependencia de ficheros no versionados
`files/params.json` (bmaps `max`/`min`, `smap`) está en `.gitignore`; `Computer.path` por defecto es `os.getcwd()` y se resuelve a través de `App.fs` relativo a `files/` (`mtpy.py:35-83`). Sin ese fichero `get_event_params` falla y sin BD nada funciona. Para el repositorio científico conviene versionar `params.json` y publicar un volcado CSV/Parquet de `polls`, `polls_results`, `events_*`, `pollsters`.

---

## 8. Preguntas abiertas para el autor

1. ¿Es intencionado que una encuestadora con `rating = 0` (quality 0 y desviación muy positiva) quede excluida por completo del promedio (`weight_rating = 0`)?
2. ¿Se pretende soportar el modo incremental `overwrite=False` (el notebook siempre usa `overwrite=True`)? Si no, ¿eliminarlo?
3. ¿Cómo se asigna `Pollsters.quality` (0-78)? ¿Es un juicio manual, y con qué criterios? Es el prior del rating y el relleno del `Forecaster`.
4. ¿Los valores `pos_decay=0.5`, `week_decay=0.7`, `year_decay=0.9`, `wspan=4`, `ol_dev=3`, `ol_max=5000` están calibrados (¿`PollstersOptimize`/`PollsOptimize`?) o son elecciones a priori?
5. ¿Qué significan exactamente `ctype='wban'` (¿publicadas en la semana de veda?) y por qué se excluyen del rating pero no del `Forecaster`?
6. ¿Por qué `drop_mtypes=['aggr','online']` se aplica sólo al cálculo de ratings/errores y no al promedio de encuestas? ¿Deben los agregadores (`aggr`) entrar en el promedio?
7. ¿Debe la winsorización del tamaño muestral ser global (1977-2027) o por evento/época? ¿Debe `ol_max` ser el tope real?
8. ¿Se desea que `weeks` sea relativo a la última encuesta del evento (efecto de la veda) o a la fecha de la elección?
9. ¿Debe el año de referencia de `weight_year` ser el evento objetivo (las encuestas de 2023 pesarían `0.9^4` para 2027) o el último evento con resultados (como ahora)?
10. Datos: existen sponsors duplicados (`Okdiario`/`OKDiario`); ¿hay un proceso de normalización de nombres de sponsors/casas en el cargador?
