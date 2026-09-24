# Computer: errores, desviaciones, ratings y ajuste bayesiano

Módulo analizado: `mtpy/lib/computer.py` (clase `Computer`), métodos `compute_errors`, `compute_deviations`, `compute_ratings`, `poll_errors`, `pollster_ratings`, `poll_rating_weights`, `fit_bias_estimator`, `bayes_adjust`, `get_polls_metric`, `lor_to_bias`/`bias_to_lor`; persistencia en `mtpy/lib/data.py` (`save_model_data`, `save_ratings_data`, `get_ratings`, `get_next_event_date`) y modelos `Polls`, `PollsResults`, `PollstersRatings`, `Pollsters` en `mtpy/models/elections.py`.

Todo lo que sigue se ha reconstruido leyendo el código (no las docstrings) y se ha verificado en modo lectura contra la base de datos local (3 825 sondeos, 92 encuestadoras con sondeos, 11 eventos "featured" 1993-2023 más el evento futuro 2027-08-22). Los scripts de verificación están en el scratchpad (`q1.py`…`q5.py`).

---

## 1. Propósito

El `Computer` es el componente "retrospectivo" del modelo: toma todos los sondeos publicados para elecciones pasadas, los compara con el resultado real, y de ahí deriva (a) medidas de error por sondeo, (b) una desviación de cada sondeo respecto a "lo que decían las demás encuestas en ese momento", y (c) un rating por encuestadora que se convierte en el peso `weight_rating` que después usa el `Forecaster` en la regresión local ponderada. El rating de una encuestadora para el evento *i* se calcula **únicamente con sondeos de eventos anteriores a *i*** (bucle de `compute_ratings`, `computer.py:930-950`), de modo que el sistema de pesos no contiene fuga de información del propio evento al que se aplica.

La inspiración declarada es el "Predictive Plus-Minus" de FiveThirtyEight (`computer.py:860-862`), pero la implementación es distinta en casi todos los pasos (ver §7).

---

## 2. Datos y estructuras

### 2.1 `build_series` (`computer.py:401-474`)

Construye `self.series`, un único DataFrame que concatena sondeos y resultados finales:

- Índice: `self.keys = ['event_date', 'date', 'pollster_id', 'sponsor_id']` (`computer.py:145`). Las filas de resultados reales tienen `pollster` nulo, así que `self.events` = `series[pollster.isnull()]` con el índice reducido a `event_date` (`computer.py:155-166`) y `self.polls` = `series[pollster.notnull()]` (`computer.py:167-177`).
- Columnas de datos (`computer.py:441-447`): `start_date, end_date, pollster, sponsor, mtype, ctype, computed, featured, sample_size, parties, days, proc_sample, rating, error_avg, error_blocks, bias_avg, bias_blocks, bias, bias_dev_adj, bias_dev_err, weight_sample, weight_over, weight_rating` seguidas de una columna por partido (`self.names`, 488 partidos con la lista completa de eventos, incluyendo un partido especial `'-'`).
- `self.errors` y `self.biases` (`computer.py:351-383`): las mismas filas pero con el `error` y `bias` por partido cargados de `polls_results` (métrica `error`/`bias`).
- `self.ratings` (`computer.py:385-399`): tabla `pollsters_ratings` indexada por `['event_date', 'pollster_id']`; `get_ratings` añade automáticamente el siguiente evento futuro a la lista de fechas (`data.py:339-341`).
- `self.pollsters` se filtra a las encuestadoras que aparecen en la serie (`computer.py:456`); `quality` y `rating` vienen de la tabla `pollsters`.
- `self.event_params` (`computer.py:134-138` → `data.py:53-140`): por evento, `parties` (event/polls), `bmaps` (`main`, `blocks`, `vs`, y en `params.json` también `max`/`min`) y `smap`. Si no hay `params.json` para un evento, `main` se deriva automáticamente como los partidos presentes en >95 % de los sondeos de los últimos 42 días (`data.py:100-103`, `get_event_dmat` `data.py:37-45`) y `vs` agrupa esos partidos por el campo `block` de la tabla `parties` (`data.py:113-123`). `params.json` (gitignored, sólo cubre 2019-04, 2019-11, 2023 y 2027) sobreescribe por clave (`smap | params`, `data.py:137`).

Importante: `Computer.path` por defecto es `os.getcwd()` (`computer.py:132`) y `get_event_params` lee `f'{path}/params.json'` (`data.py:67`). Desde la raíz del repo la construcción falla con `FileNotFoundError`; hay que pasar `path='.../files'`.

### 2.2 `filter_polls` (`computer.py:250-306`)

Filtro común usado por deviations/ratings: sondeos con `pollster` no nulo, `mtype ∉ drop_mtypes` (por defecto `['aggr', 'online']`), `ctype ∉ drop_ctypes` (`['wban', 'exit']`), `weight_over > 0`, `days ∈ drange` (por defecto `[0, max]`), y si `featured=True`, sólo eventos `featured` y sondeos `computed`. Opcionalmente `n_last` últimos sondeos por (evento, encuestadora). Con los valores por defecto quedan 2 238 sondeos.

### 2.3 Parámetros por defecto (`computer.py:32-53`)

| Parámetro | Valor | Uso real |
|---|---|---|
| `drop_mtypes` | `['aggr','online']` | `filter_polls` |
| `drop_ctypes` | `['wban','exit']` | `filter_polls` |
| `alpha` | 0.05 | nivel de los IC del `LocalKernelEstimator` |
| `reg_params` | kernel gaussiano, bw adaptativo ISJ (3 iter.), grado 1, covarianza HAC (Bartlett, 1 lag) (`computer.py:214-248`) | `fit_bias_estimator` |
| `ol_dev`, `ol_max` | 3, 5000 | winsorización de `proc_sample` (compute_weights) |
| `wspan` | 4 | `poll_overweights` |
| `pos_decay` | 0.5 | `weight_pos` |
| `week_decay` | 0.7 | `weight_week` |
| `year_decay` | 0.9 | `weight_year` |
| `bias_dev_tau` | 0.01 | σ del prior N(0, τ²) en **ambos** `bayes_adjust` y escala de `norm.cdf` en `rating_adj` |
| `min_polls` | 3 | pseudo-recuento del prior `quality` en `rating` |
| `ratings_margin` | 0.05 | **no se usa en ningún sitio** |
| `error_weights` | `{'avg': .7, 'blocks': .3}` | sólo para combinar `bias_avg` y `bias_blocks` en `bias` |

---

## 3. Transformaciones de escala

`lor_to_bias(x) = (exp(x) − 1)·100` y `bias_to_lor(x) = log(x/100 + 1)` (`computer.py:1233-1273`). Es decir, lo que el código llama "bias" en escala de porcentaje es el **cambio porcentual de la odds** (OR − 1)·100, y su logaritmo es el log-odds-ratio (LOR). Todo el cálculo interno (medias, regresión, shrinkage) se hace en LOR; sólo se convierte a "%" para guardar y mostrar. Las dos funciones son inversas exactas, salvo por el redondeo a 2 decimales que se aplica al guardar (`computer.py:745, 838, 979`).

---

## 4. Métodos

### 4.1 `poll_errors(polls, events, error_type, bmap)` (`computer.py:1008-1092`)

Devuelve una matriz `dmat` con columnas MultiIndex `('poll', c)`, `('event', c)`, `('error', c)` para cada partido/bloque `c`, indexada como `polls`.

1. Si `bmap` es un nombre (`'main'`, `'vs'`), se llama a `merge_bmaps(name)` (`computer.py:179-212`), que **une** los bmaps de todos los eventos cargados (p. ej. `vs` completo = Derecha: UCD, EDC, AP, CD, CDS, PP, Cs, VOX, SALF; Izquierda: PSOE, PCE, PSP, IU, UP, MP, SUMAR). `build_blocks` (`utils.py:9-59`) construye la tabla de bloques y `group_results` (`utils.py:62-98`) suma las columnas de cada bloque con `sum(min_count=1)`: un bloque con algún partido ausente en el sondeo se suma con los que haya.
2. Para bloques nombrados, los bloques/partidos que **no** están en `event_params[dt]['bmaps'][bname]` de cada evento se ponen a NaN en `events` (`computer.py:1043-1046`), lo que anula el error de ese partido en ese evento (p. ej. `SUMAR` no cuenta como "main" en 2019).
3. Todo se divide por 100 (proporciones).
4. Error:
   - `'pct'`: `e = p − r` (proporción del sondeo menos resultado).
   - `'gap'`: `polls.diff(axis=1)` → una única columna `gap` = segundo bloque − primero (Izquierda − Derecha, por el orden del dict), y `e = gap_poll − gap_event`.
   - `'lor'`: `e = log[ (p/(1−p)) / (r/(1−r)) ]`.

### 4.2 `compute_errors(save=False)` (`computer.py:637-774`)

Se ejecutan seis `poll_errors` sobre **todos** los sondeos (`self.polls`, sin `filter_polls`):

| Llamada | error_type | bmap | Uso |
|---|---|---|---|
| `errors_pct` | pct | todos los partidos | `error` por partido → `polls_results.error` (×100) |
| `errors_lor` | lor | todos | `bias` por partido → `polls_results.bias` (`lor_to_bias`) |
| `errors_pct_main` | pct | main | `error_avg` |
| `errors_lor_main` | lor | main | `bias_avg` |
| `errors_gap_vs` | gap | vs | `error_blocks` |
| `errors_lor_vs` | lor | vs | `bias_blocks` |

Definiciones exactas por sondeo *k* (`computer.py:717-741`):

- `error_avg_k = 100 · Σ_{j∈main, e_kj≠NaN} r_j |e^{pct}_kj| / Σ r_j` — media de errores absolutos en puntos, **ponderada por la cuota real `r_j` de cada partido** (`weights=x['event']`), vía `Stat(...).mean()` (`stat.py:145-146`, que normaliza los pesos a `neff`).
- `error_blocks_k = 100 · (gap_poll − gap_event)` — **con signo** (positivo = el sondeo sobreestima Izquierda − Derecha).
- `bias_avg_k = Σ_{j∈main} r_j |LOR_kj| / Σ r_j` (en LOR, absoluto, ponderado por cuota).
- `bias_blocks_k = Σ_{b∈{Der,Izq}} r_b |LOR_kb| / Σ r_b` (en LOR, absoluto; NaN si falta un bloque completo).
- `bias_k = 0.7·bias_avg_k + 0.3·bias_blocks_k` (media ponderada `Stat` con `error_weights`; si `bias_blocks` es NaN se reduce a `bias_avg`).
- Finalmente `error_*` ×100, `bias_*` → `lor_to_bias`, redondeo a 2 decimales.

Nótese la inversión de nomenclatura: `error_blocks` es el único estadístico **con signo** (un sesgo real, de tipo *house effect*), mientras que todos los `bias_*` son **magnitudes absolutas** (errores).

Salida: DataFrame `[error_avg, error_blocks, bias_avg, bias_blocks, bias]` indexado por `self.keys`; con `save=True` se guardan `polls_results.error/bias` y `polls.<5 columnas>` vía `save_model_data` (`data.py:381-420`: `UPDATE ... SET col=NULL` para los eventos afectados y luego `upsert`). Valores empíricos (sondeos featured): `error_avg` mediana 3.6 pts, `bias` mediana 20.6 % de odds (máx. 155 %), `error_blocks` media −1.3 pts (los sondeos tienden a infraestimar Izquierda − Derecha… o sobreestimar la derecha).

Reproducibilidad verificada: recomputando con todos los eventos, los cinco valores coinciden exactamente con los almacenados (diferencia máxima 0.0 en 3 088 filas). Con sólo dos eventos cargados, 146 filas (todo 2016) cambian en `error_blocks`/`bias_blocks`/`bias` (ver hallazgo F7).

### 4.3 `fit_bias_estimator` (`computer.py:1461-1508`) y `compute_deviations` (`computer.py:776-851`)

Objetivo: obtener para cada sondeo la diferencia entre su error (`bias` en LOR) y el error medio de los sondeos contemporáneos del mismo evento.

1. `df = filter_polls(featured=True)`; `lor = bias_to_lor(bias)`; `weight = weight_over · weight_sample` (redondeado).
2. Para cada evento `dt`: `px` = todos los días desde el primer sondeo del evento hasta `dt`; se ajusta `LocalKernelEstimator(lor, weights, kernel gaussiano, bw adaptativo ISJ, grado 1, HAC).fit(px, alpha=.05)` (`stat.py:859-993`): para cada día, una regresión lineal local ponderada por kernel×peso, con `mean`, `cmin`, `cmax`, `err` (error estándar de la media local, `stat.py:837-841`), `nobs`, `neff`.
3. `bias_mean(t)`, `bias_err(t)` se unen a **todos** los sondeos (`self.polls`, incluidos agregadores y eventos no featured, que quedan NaN) por (`event_date`, `date`).
4. `bias_dev_k = lor_k − bias_mean(t_k)`.
5. `[bias_dev_adj, bias_dev_err] = bayes_adjust([bias_dev, bias_err], prior N(0, τ=0.01))`.
6. Conversión a "%" con `lor_to_bias` (incluida la desviación típica), redondeo a 2 decimales y guardado en `polls`.

Valores empíricos (dos campañas cortas, 2016 y 2019-11): `err` de la regresión mediana 0.0118 LOR con `neff` ≈ 18; desviación típica de `bias_dev` bruta 0.038 LOR (3.8·τ); factor de shrinkage `λ_dat/(λ_dat+λ_prior)` mediana 0.49 (rango p5–p95: 0.20–0.89). En la BD, `bias_dev_err` almacenado (post-shrinkage) es ≤ 0.0099 LOR (≤ τ, como debe ser por construcción).

### 4.4 `bayes_adjust(x, params)` (`computer.py:1324-1388`)

Actualización conjugada normal-normal por filas, correcta como fórmula:

```
λ_prior = 1/σ_prior²      λ_dat = 1/σ_dat²
λ_post  = λ_prior + λ_dat
μ_post  = (λ_prior·μ_prior + λ_dat·μ_dat) / λ_post
σ_post  = sqrt(1/λ_post)
```

`params` admite dict/list; `prior_mean`/`prior_std` pueden ser arrays. Se llama dos veces con `μ_prior=0, σ_prior=τ`: en `compute_deviations` (por sondeo, `σ_dat = bias_err`, el SE de la **media de regresión**) y en `pollster_ratings` (por encuestadora, `σ_dat = bias_dev_err` agregado). Ver hallazgos F1 y F2 sobre la especificación de `σ_dat`.

### 4.5 `poll_rating_weights(polls)` (`computer.py:1192-1231`)

Pesos adicionales sólo para el rating:

- `weight_pos = pos_decay^seq_pos`, con `seq_pos` = posición desde el final entre los sondeos de la misma (evento, encuestadora) ordenados por fecha (`cumcount(ascending=False)`): el último sondeo pesa 1, el anterior 0.5, etc. Suma máxima por evento y encuestadora ≈ 2.
- `weight_week = week_decay^weeks`, `weeks = ((days − min_days_evento + 1) // 7).clip(0)`. Con `drange=None`, `min_days ≈ 0`, así que `weeks ≈ days//7`.
- `weight_year = year_decay^(año_max − año_evento)`, donde `año_max` es el del evento más reciente **dentro del subconjunto pasado** (no el evento para el que se calcula el rating).

### 4.6 `pollster_ratings(polls)` (`computer.py:1094-1190`)

Entrada: sondeos ya filtrados y con `error_*` en [0,1] y `bias_*`, `bias_dev_*` en LOR (`computer.py:917-927`).

1. `weight = weight_over · weight_sample · weight_pos · weight_week · weight_year`; se eliminan los sondeos con `weight < 1e-2` (`computer.py:1103-1109`). Efecto empírico: sólo 343 de 2 238 sondeos sobreviven, todos a ≤ 82 días de la elección (0.7^13 ≈ 0.0097). **El rating se construye, en la práctica, con los sondeos de los últimos ~3 meses de cada campaña**, algo comparable a la ventana de 21 días de 538, aunque no está documentado.
2. Índice de salida: todas las encuestadoras de `self.pollsters` (`computer.py:1112-1114`). `quality = pollsters.quality/100`.
3. `num_events`, `num_polls` (recuento), `num_polls_w = Σ weight`. Como `pollster` es categórica, `groupby` produce filas con 0 para las encuestadoras sin sondeos (`observed=False`, ver F9).
4. Medias ponderadas por encuestadora `p`: `m_p(col) = Σ w_k col_k / Σ w_k` para `error_avg, error_blocks, bias_avg, bias_blocks, bias, bias_dev_adj` (`computer.py:1129-1138`).
5. Error estándar ponderado (`computer.py:1140-1144`):
   `se_p = sqrt( Σ w_k² · err_k² ) / Σ w_k · sqrt( (n_w+1)/n_w )`
   La primera parte es la varianza correcta de una media ponderada de estimadores independientes con desviación `err_k` (los pesos van al cuadrado en el numerador y el denominador es (Σw)², invariante a la escala de los pesos). El factor `sqrt((n_w+1)/n_w)` es una inflación *ad hoc* para pocos sondeos (n_w=0.01 → ×10).
6. Segundo `bayes_adjust` sobre `[bias_dev_adj_p, se_p]` con prior N(0, τ²).
7. `rating_adj_p = Φ(−bias_dev_adj_p / τ)` (`norm.cdf(scale=τ)`), rellenado con `quality` si es NaN (`computer.py:1173`). Una encuestadora media (desviación 0) obtiene 0.5; con desviación −τ (1 % de odds mejor que la media) 0.84; con −2.5τ, 0.99.
8. `rating_p = (n_w · rating_adj_p + min_polls · quality_p) / (n_w + min_polls)` (`computer.py:1178-1182`): media ponderada entre evidencia y prior subjetivo, con `min_polls=3` pseudo-sondeos. Con `n_w` ≈ 2 por evento como máximo, una encuestadora necesita ~2-3 elecciones con encuestas densas para que los datos pesen la mitad (40dB: n_w=4.33 → 59 % datos).
9. `weight_rating_p = log1p(rating_p) / log1p( mean_q rating_q )` (`computer.py:1186`): la media es sobre **todas** las encuestadoras del índice (92, la mayoría sin sondeos y con quality 0.05 → media 0.22), de modo que las buenas encuestadoras obtienen pesos ≈ 3.

### 4.7 `compute_ratings(save=False)` (`computer.py:853-1006`)

1. `polls = filter_polls()` con `error_*` /100 y `bias_*` → LOR.
2. Bucle `i = 1..len(edts)`: `event_date = edts[i]` (o `get_next_event_date` para el último, `data.py:524-543`), `iter_polls = polls.loc[:edts[i−1]]` (todos los sondeos de eventos **≤ i−1**, slicing inclusivo). Se salta el evento si no hay sondeos o si **algún** `bias` es nulo. Llama a `pollster_ratings` y añade `event_date`, `pollster_id`.
   - Por tanto: el rating de 1996 se calcula con 1993; el de 2027 con 1993…2023. El primer evento featured (1993) no recibe rating (sus 53 sondeos quedan con `rating`/`weight_rating` NaN en `polls`, verificado en la BD).
3. Reescalado ×100 de `quality, error_avg, error_blocks, rating_adj, rating`; `bias_*` → `lor_to_bias`; redondeo a 2 decimales.
4. `df` = `rating`, `weight_rating` asignados a cada sondeo por (`event_date`, `pollster_id`).
5. `save=True`: `save_ratings_data` (`data.py:423-460`: `DELETE` de los eventos afectados, `upsert` en `pollsters_ratings`, y actualización de `pollsters.rating` con el último rating de cada encuestadora, `astype(int)`), y `save_model_data('Polls', ...)` para `rating`/`weight_rating`.
6. Actualiza `self.pollsters['rating']`, `self.series[columns]` y `self.ratings[rparams]`.

Tabla `pollsters_ratings` (`elections.py:65-90`): clave (`event_date`, `event_scope`, `pollster_id`), 14 métricas. En la BD hay 11 fechas × 92 encuestadoras = 1 012 filas.

### 4.8 `get_polls_metric` (`computer.py:1390-1459`)

Utilidad de exploración (no forma parte del pipeline de cálculo): filtra sondeos, sustituye las columnas de partidos por `self.errors`/`self.biases` si `metric` es `'error'`/`'bias'`, agrupa por `bmap`, filtra por encuestadora y devuelve una tabla indexada por (`event` como "YYYY-DMmm", `pollster`, `days`) con `sample_size, proc_sample, weight(=over·sample), error_avg, bias_avg, error_blocks, bias_blocks, bias` + bloques. Es lo que usa `notebooks/PollstersRatings.ipynb` para las dianas y los histogramas.

### 4.9 Consumo aguas abajo

`Forecaster` (`forecaster.py:270-271`) usa `weight = weight_over · weight_sample · weight_rating`, rellenando `weight_rating` nulo con `pollsters.quality` **sin dividir por 100** (ver F11). No usa `bias`, `error_blocks` ni `bias_dev_*`: no hay corrección de *house effects* en la predicción.

---

## 5. Cifras empíricas de referencia (BD, sondeos featured & computed)

| Métrica | n | media | p50 | rango |
|---|---|---|---|---|
| `error_avg` (pts) | 3088 | 4.03 | 3.62 | 0.1–12.3 |
| `error_blocks` (pts, signo) | 3088 | −1.33 | −1.97 | −20–35 |
| `bias` (% odds) | 3088 | 23.1 | 20.6 | 0.9–155 |
| `bias_dev_adj` (% odds) | 3088 | −0.09 | −0.31 | −7–20 |
| `bias_dev_err` (% odds) | 3088 | 0.67 | 0.65 | 0.32–0.99 |
| `rating` por encuestadora (2027) | 92 | 22 | 8.9 | 0–84.8 |
| `weight_rating` (2027) | 92 | 0.92 | 0.42 | 0–3.08 |

`quality` en `pollsters`: 53 de 95 valen 5, tres valen 0, CIS = 10, máximo 78 (Ikerfel). Encuestadoras sin sondeos evaluables (p. ej. Ikerfel, YouGov como `aggr`) reciben `rating = quality` exactamente.

---

## 6. Notas positivas de diseño

- Separación temporal estricta en `compute_ratings`: el rating aplicado a los sondeos del evento *i* sólo usa eventos < *i*. Es la propiedad más importante para un backtest honesto y está bien resuelta.
- Trabajar en log-odds hace comparables los errores de partidos grandes y pequeños y evita asimetrías en los extremos; la pareja `lor_to_bias`/`bias_to_lor` es una inversa exacta.
- La fórmula del error estándar de la media ponderada (`Σw²σ²/(Σw)²`) es correcta y la actualización normal-normal está bien implementada como fórmula.
- El estimador local con ancho de banda adaptativo (ISJ) y covarianza HAC está escrito a mano y es legible: buen material divulgativo.
- Recomputar `compute_errors` con el conjunto completo de eventos reproduce exactamente lo almacenado en la BD.
- El peso de posición (`pos_decay`) y la eliminación de solapes (`weight_over`) resuelven de forma sencilla el problema de los *trackings* y de las encuestadoras que inundan el mercado.

---

## 7. Comparación con FiveThirtyEight

| Aspecto | 538 (pollster ratings) | Este modelo |
|---|---|---|
| Métrica de error | Error absoluto del margen entre los dos primeros candidatos, sondeos de los últimos 21 días | `bias` = 0.7·media ponderada de \|LOR\| de partidos principales + 0.3·\|LOR\| del bloque Der/Izq; ventana efectiva ~82 días vía `weight_week` |
| Plus-minus | Error del sondeo menos el error medio de los demás sondeos de la misma carrera, ajustado por tipo de elección, tamaño muestral y días | `bias_dev` = LOR del sondeo menos la media de regresión local del evento en la misma fecha; sin ajuste por tamaño muestral/días en la desviación (sí en los pesos) |
| Reversión a la media (Predictive Plus-Minus) | Shrinkage hacia una media que depende de criterios metodológicos objetivos (AAPOR/transparencia, live-caller, etc.), con peso según nº de sondeos | Dos shrinkages normal-normal con τ fijo hacia 0, y mezcla con `quality` subjetivo con `min_polls=3` |
| House effect (sesgo con signo) | Se calcula (“mean-reverted bias”) y en el forecast se corrige el sesgo partidista de cada encuestadora | `error_blocks` con signo se calcula por encuestadora pero **no se usa** ni en el rating ni en el forecast |
| Herding | Penalización a encuestadoras con menos varianza de la esperada | No existe |
| Nuevas encuestadoras | Rating provisional según metodología | `rating = quality` (subjetivo) |
| Salida | Nota A+…F y plus-minus | `rating` 0-100 y `weight_rating` ≈ log1p(rating)/log1p(media) |

---

## 8. Pipeline

1. `Computer(...).build_series()` (`computer.py:401-474`): carga eventos, sondeos, errores, sesgos, ratings, partidos, encuestadoras; construye `series`.
2. `compute_weights` (`computer.py:476-635`): `mtype`, `proc_sample` (imputación + winsorización), `weight_sample = sqrt(proc_sample / mediana_evento)`, `weight_over` (solapes y `wspan`), `weight_rating` (merge desde `self.ratings` de la BD).
3. `compute_errors` (`computer.py:637-774`): 6 × `poll_errors` → `error`/`bias` por partido y `error_avg, error_blocks, bias_avg, bias_blocks, bias` por sondeo.
4. `compute_deviations` (`computer.py:776-851`): `fit_bias_estimator` (regresión local del LOR por evento y día) → `bias_dev` → `bayes_adjust` → `bias_dev_adj, bias_dev_err`.
5. `compute_ratings` (`computer.py:853-1006`): para cada evento, `pollster_ratings` con los sondeos de eventos anteriores: pesos de rating → medias ponderadas → SE → `bayes_adjust` → `rating_adj` → `rating` → `weight_rating`; guarda `pollsters_ratings`, `polls.rating/weight_rating`, `pollsters.rating`.
6. `Forecaster` consume `weight_rating` (`forecaster.py:270-271`).

---

## 9. Hallazgos

Severidades: high = afecta a la validez de resultados o rompe el código en uso normal; medium = afecta a la interpretación estadística o a la reproducibilidad; low = mejora/aviso.

**F1 [stat, medium] La verosimilitud del shrinkage por sondeo está mal especificada.** `compute_deviations` pasa a `bayes_adjust` `σ_dat = bias_err` (`computer.py:822-830`), que es el error estándar de la **media local de regresión** (`stat.py:837-841`), no la dispersión del dato `bias_dev_k`. En un modelo normal-normal el dato es la desviación del sondeo, cuya varianza es `σ_sondeo² + SE_media(t)²`. Empíricamente la desviación típica de `bias_dev` bruta es 0.038 LOR frente a τ = 0.01 (3.8×): el prior declara que casi ningún sondeo se desvía más de un 2 % en odds cuando en realidad el 90 % central va de −6 % a +6 %. El shrinkage resultante (mediana 0.49, p5–p95 0.20–0.89) depende de la densidad de sondeos en la fecha, no del ruido del sondeo; esa es la intención declarada en la docstring (`computer.py:1327-1332`), pero es un regularizador *ad hoc*, no una actualización bayesiana. Test: `q3.py` (imprime `raw dev std`, `shrink factor quantiles`). Propuesta: modelo jerárquico `dev_k ~ N(θ_p, σ² + SE_t²)`, `θ_p ~ N(0, τ²)` con σ y τ estimados por Bayes empírico (o REML), en lugar de τ fijo.

**F2 [stat, medium] Doble shrinkage con incertidumbres ya posteriores.** `pollster_ratings` agrega `bias_dev_adj` (ya encogidos) y calcula `se_p` a partir de `bias_dev_err` (ya posteriores, ≤ τ) y vuelve a aplicar el mismo prior N(0, τ²) (`computer.py:1140-1170`). El prior se cuenta dos veces y `se_p` no refleja la dispersión real de la encuestadora (nunca se usa la varianza muestral de sus desviaciones). Test: comparar `se_p` con `std(bias_dev_p)/sqrt(n_p)` para encuestadoras con ≥ 5 sondeos.

**F3 [design, medium] `bias_dev_tau` está sobrecargado y satura el rating.** El mismo τ es (a) σ del prior en dos sitios, (b) la escala de `rating_adj = Φ(−dev/τ)` (`computer.py:1173`) y (c) los límites de los gráficos (`computer.py:2163, 2259-2280`). Con τ = 0.01, una desviación de −2.5 % de odds da `rating_adj ≈ 99`; en la BD Noxa (6 sondeos, dev −2.14) obtiene 98.5 y Encuestamos (6 sondeos) 95.0 antes de la mezcla con `quality`. Debería existir un parámetro propio para la escala del rating (p. ej. la desviación típica empírica de `bias_dev_adj_p`).

**F4 [docs, medium] La docstring de `compute_ratings` describe un algoritmo que no existe.** `computer.py:866-897` habla de `num_related`, `error_expected`, `error_adjusted = (num_related·error_related + min_related·error_expected)/(…)`, `prior_weight = quality·num_polls/(num_polls+mean)`, `dev_reverted` y "scaled to the range [5-95]". El código real (`computer.py:1094-1190`) es el de §4.6. También la docstring de `bias_dev_err`/`weight_rating` (“logarithm gives more weight to pollsters with a rating close to 1”, `computer.py:1183-1185`) es al revés: `log1p` es cóncavo y comprime las diferencias arriba.

**F5 [dead-code, low] `ratings_margin` no se usa.** Sólo aparece en `__init__` (`computer.py:100, 128`). Test: `grep -n ratings_margin mtpy/lib/*.py`.

**F6 [bug, medium] `plot_ratings_prior` referencia una columna inexistente.** Usa `df.rating_prior` (`computer.py:2348-2351`) pero `rparams`/`pollsters_ratings` no la contienen (`computer.py:911-915`, `elections.py:72-90`) → `AttributeError`. Test: `comp.plot_ratings_prior()` tras `build_series()`.

**F7 [reproducibility, medium] Las métricas de bloque dependen de qué eventos se cargan, y `merge_bmaps` muta `event_params`.** `merge_bmaps` une los bmaps de todos los eventos de la instancia (`computer.py:179-212`), así que `Izquierda` incluye `IU` sólo si se han cargado eventos en que IU era principal. Con `event_dates=['2016-06-26','2019-11-10']` los 146 sondeos de 2016 (72 de ellos publican IU aparte) obtienen `error_blocks`/`bias_blocks`/`bias` distintos de los almacenados (diferencias de hasta 7.8 pts / 17.8 %). Además `bm[party] = block` aliasa la lista del primer evento y los `append` posteriores modifican `self.event_params` (verificado: tras `merge_bmaps('vs')`, `event_params['2016-06-26']['bmaps']['vs']` pasa de `{Derecha:[PP,Cs], Izquierda:[PSOE,UP]}` a incluir VOX y MP). Test: `q3.py`/`q4.py`. Propuesta: definir los bloques por evento (ya está en `event_params[dt]`) y agrupar evento a evento; copiar listas en `merge_bmaps`.

**F8 [stat, medium] Sumas parciales de bloques.** `group_results` suma con `min_count=1` (`utils.py:96`): si a un sondeo le falta un partido del bloque que sí tiene resultado en el evento, el gap se calcula con el bloque incompleto y el "error" incluye el partido omitido (p. ej. 180 sondeos de 2019-04 sin VOX, 68 de 2011 sin IU). En 2015 y 2023 la mayoría de los casos son sustituciones legítimas (UP/SUMAR) que el bmap acepta, pero el código no distingue ambos casos. Test: `q5.py` (tabla `missing_parties` por evento). Propuesta: exigir que los partidos con resultado > umbral estén en el sondeo (o resolverlos con `smap`) y poner NaN en caso contrario.

**F9 [reproducibility, medium] El manejo de encuestadoras sin sondeos depende de `groupby(observed=False)` sobre una categórica.** `num_polls`, `num_polls_w` (`computer.py:1123-1126`) valen 0 para las ausentes sólo porque `pollster` es `category` y pandas 2.2 usa `observed=False` (emite `FutureWarning`; será `True` en pandas 3). Con `observed=True` serían NaN y `rating = (NaN·… + 3·q)/(NaN+3)` = NaN para todas las encuestadoras nuevas. Test: `comp.polls.groupby('pollster', observed=True)['days'].count()` tiene menos filas. Propuesta: `reindex(...).fillna(0)` explícito.

**F10 [design, low] `weight_rating` se normaliza con la media de rating de todas las encuestadoras registradas.** `computer.py:1186`: la media incluye ~50 encuestadoras inactivas con `quality=5`, por lo que dar de alta una encuestadora más cambia todos los pesos (media 0.22 en la BD; con sólo dos eventos, 0.39). Propuesta: normalizar sobre las encuestadoras con sondeos en la ventana, o sobre los sondeos ponderados.

**F11 [bug, medium] `Forecaster` rellena `weight_rating` nulo con `quality` en escala 0-100.** `forecaster.py:270`: `weight_rating.fillna(pollster.quality)`; `quality` es 5-78 mientras `weight_rating` es 0-3. Afecta a los 53 sondeos de 1993 (primer evento featured, sin rating previo) y a cualquier encuestadora sin fila en `pollsters_ratings`. Test: `Forecaster(scope='es', event_date='1993-06-06').build_series().series.weight_rating.describe()`.

**F12 [docs, low] Nomenclatura invertida respecto al uso estándar.** `bias_avg`, `bias_blocks`, `bias`, `bias_dev_*` son magnitudes absolutas (errores en odds); el único sesgo con signo es `error_blocks` (`computer.py:721-735`). Además `lor_to_bias` devuelve (OR−1)·100, un "cambio porcentual de odds", no un sesgo. Para un repositorio científico conviene renombrar (`abs_lor_error`, `gap_bias`, …) o documentarlo explícitamente.

**F13 [docs, low] `error_weights` sólo afecta a `bias`.** Se aplica con prefijo `bias_` (`computer.py:736-737`); `error_avg`/`error_blocks` nunca se combinan. La docstring lo describe como "weights for our custom error metric".

**F14 [stat, low] Transformación no lineal de una desviación típica y redondeo.** `bias_dev_err` (una σ en LOR) se pasa por `lor_to_bias` (`computer.py:836`) y se redondea a 2 decimales en "%": a valores ≈ 0.65 % el redondeo introduce errores relativos de ~1 %; y al volver a LOR (`computer.py:920-927`) la σ recuperada no es exactamente la original. Propuesta: guardar `bias_dev_*` en LOR.

**F15 [design, low] Auto-inclusión en la media de referencia.** `bias_mean(t)` se estima con todos los sondeos del evento, incluido el propio sondeo (y los posteriores dentro de la campaña) (`computer.py:1461-1508`, `815-820`). En fechas con pocos sondeos la desviación se sesga hacia 0. Propuesta: *leave-one-out* (o excluir la encuestadora) al calcular la media local, como hace 538 con "los demás sondeos".

**F16 [design, low] Ventana efectiva y pseudo-recuento no documentados.** El umbral `weight ≥ 1e-2` (`computer.py:1109`) con `week_decay=0.7` deja 343 de 2 238 sondeos (≤ 82 días). `num_polls_w` por evento y encuestadora es ≤ ~2 (`pos_decay=0.5`), luego `min_polls=3` equivale a ~1.5 elecciones de evidencia, no a 3 sondeos como sugiere la docstring (`computer.py:98-99`). Test: `q5.py`.

**F17 [reproducibility, low] `path` por defecto y `params.json`.** `self.path = path or os.getcwd()` (`computer.py:132`) y `get_event_params` lee `{path}/params.json` (`data.py:67`): desde la raíz del repo `Computer('es')` lanza `FileNotFoundError`; `params.json` está en `files/` y gitignored. Sin él, `main`/`vs` se derivan automáticamente (`data.py:100-123`) y cambian los resultados de §4.2.

**F18 [bug, low] Último evento sin sucesor y orden de `get_next_event_date`.** Si no existe un evento futuro en `events`, `event_date = pd.NaT` (`computer.py:931, 947`) y el guardado intentaría una clave nula; `get_next_event_date` (`data.py:524-543`) usa `get_var` con `limit=1` sin `ORDER BY`, por lo que con varios eventos futuros el elegido no está garantizado.

**F19 [design, low] Salto de evento completo por un `bias` nulo.** `compute_ratings` descarta la iteración si `iter_polls['bias'].isnull().any()` (`computer.py:938-943`) aunque `pollster_ratings` ya filtra nulos (`computer.py:1113`). Hoy no se activa (0 nulos en featured), pero un solo sondeo corrupto dejaría un evento sin ratings.

**F20 [stat, low] Ponderación por cuota electoral en `bias_avg`/`error_avg`.** Los pesos `x['event']` (`computer.py:717-720`) hacen que el error en partidos pequeños pese poco, mientras que el LOR amplifica precisamente esos errores (`bias_avg` máx. 243 %). Es una decisión razonable pero no documentada; conviene justificarla o usar pesos `sqrt(r_j)` / truncar.

**F21 [design, low] Sin *house effects* ni *herding*.** El sesgo con signo `error_blocks` por encuestadora se calcula (`computer.py:1129-1133`) pero ni el rating ni el `Forecaster` lo usan; no hay medida de varianza inter-sondeo por encuestadora. Frente a 538 son las dos ausencias más visibles.

**F22 [stat, low] Inflación `sqrt((n_w+1)/n_w)`.** `computer.py:1142-1144`: factor sin justificación estadística; para `n_w=0.01` multiplica por 10. Sugerencia: usar un `t` con `neff−1` grados de libertad o documentarlo como heurística.

**F23 [design, low] Prior `quality` subjetivo y no documentado.** 53/95 encuestadoras con 5, CIS con 10 (por debajo de encuestadoras privadas con 76), tres con 0. No hay criterio escrito ni fuente (`Electomania-Rankings.xlsx` en `files/` podría serlo). Para un repositorio científico el prior debería derivarse de criterios verificables (transparencia, muestra, método, historial) o al menos publicarse la rúbrica.

---

## 10. Preguntas abiertas para el autor

1. ¿Es intencionado que el shrinkage por sondeo dependa del SE de la media local (densidad de sondeos) y no de la dispersión de los sondeos? ¿Cómo se eligió τ = 0.01?
2. ¿Cuál es la rúbrica de `pollsters.quality`? ¿Procede de `files/Electomania-Rankings.xlsx`? ¿Por qué el CIS tiene 10?
3. ¿La docstring de `compute_ratings` (num_related, error_expected, [5-95]) corresponde a una versión anterior que se quiere recuperar, o hay que reescribirla?
4. ¿Se desea que `error_blocks` (sesgo con signo) actúe como *house effect* en el `Forecaster` (restándolo a cada sondeo), como hace 538?
5. ¿Deben los bloques (`vs`) definirse por evento (evitando la unión de `merge_bmaps`) y cómo se quiere tratar un sondeo que omite un partido con resultado (NaN vs suma parcial)?
6. ¿La ventana efectiva de ~82 días para el rating es una decisión consciente? ¿Y `min_polls=3` como pseudo-recuento (≈1.5 elecciones)?
7. ¿`ratings_margin` y `plot_ratings_prior`/`rating_prior` son restos de una versión anterior que se pueden eliminar?
8. ¿Cómo se mantiene el evento "futuro" (2027-08-22) en `events`? ¿Qué debe ocurrir con los ratings cuando no hay evento futuro registrado?
9. ¿Quieres que `bias_dev_*` se almacene en LOR (sin redondeo en %) para evitar la pérdida de precisión al ida-y-vuelta?
