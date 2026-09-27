# Forecaster: regresión local ponderada — informe de lectura en profundidad

Ficheros analizados (lectura completa): `mtpy/lib/forecaster.py` (1185 líneas), `mtpy/core/utils/stat.py` clases `Kernel` (249-520), `Estimator` (521-659), `LeastSquaresEstimator` (661-857) y `LocalKernelEstimator` (859-995); apoyo: `mtpy/lib/utils.py`, `mtpy/lib/data.py` (`get_poll_series`, `get_event_series`, `get_event_dates`), `mtpy/lib/simulator.py:100-330`, `mtpy/lib/computer.py:540-620, 1175-1190`, `notebooks/PollsForecast.ipynb`, `files/params.json`.

Todas las cifras "en vivo" de este informe proceden de ejecuciones de solo lectura sobre la base de datos local con `scope='es'`, `event_date='2027-08-22'`, `bmap='max'` (306 sondeos entre 2023-08-07 y 2026-09-11; partido de referencia PP).

---

## 1. Visión general

`Forecaster` construye, para un evento electoral concreto, una serie temporal diaria con la estimación del porcentaje de voto de cada partido (o bloque de partidos) a partir de todos los sondeos publicados desde las elecciones anteriores. El estimador es una **regresión polinómica local de grado 1 (local linear) con núcleo gaussiano en el tiempo**, en la que cada sondeo entra con un peso propio calculado previamente por `Computer` (`weight = weight_over * weight_sample * weight_rating`). Para cada día `t` de la rejilla se resuelve un problema de mínimos cuadrados ponderados con pesos `K_h(t - t_i) * w_i`, se toma la ordenada de la recta local en `t` como estimación y se deriva un error estándar (matriz sandwich tipo HAC/Newey-West) y un intervalo de confianza t-Student.

Es, en la terminología habitual, un **LOESS/LOWESS de grado 1 con pesos externos, ancho de banda adaptativo (tipo *balloon*, dependiente del punto de evaluación) y piloto por el método de Botev (ISJ)**. No hay ajuste explícito de *house effects* (sesgo por encuestadora): el sesgo histórico solo entra indirectamente a través del rating y, por tanto, del peso.

Decisiones de diseño acertadas que conviene mantener y documentar:

- **Grado 1 (local linear) por defecto** (`forecaster.py:173`). Es la decisión clave que hace que el estimador se comporte bien en la frontera derecha (último sondeo), donde Nadaraya-Watson (grado 0) tendría sesgo de orden `h` por la asimetría de la ventana. La corrección automática de sesgo de frontera del local linear (Fan & Gijbels, 1996) es lo que permite usar el valor en la última fecha como "promedio actual".
- **Pesos externos multiplicativos** (`forecaster.py:271`) combinados con el peso del núcleo (`stat.py:914`): estructura transparente y fácil de explicar.
- **Tamaño muestral efectivo de Kish** `n_eff = (Σw)²/Σw²` (`stat.py:278, 574, 709`) para los grados de libertad, en lugar de contar sondeos.
- **Varianza robusta tipo sandwich** en vez de la fórmula OLS clásica (`stat.py:753-778`).
- **Extrapolación acotada** por `max_fc` (`forecaster.py:343`): se evita extrapolar linealmente hasta la fecha electoral.
- API con `pd.Series` indexada por fecha: `Estimator.build_input` convierte fechas a días desde la primera observación (`stat.py:583-600`) y devuelve de nuevo fechas (`stat.py:978-989`).

---

## 2. Flujo de datos y estructuras

### 2.1 Entradas

- `files/params.json[scope][event_date]` (`forecaster.py:107-109`): listas `parties.polls`, `parties.event`, diccionario `bmaps` (`max`, `min`, `main`, `blocks`, `vs`) y `smap`. **Este fichero está ignorado por git** (`.gitignore:2 files/`), por lo que el módulo no es ejecutable desde un clon del repositorio.
- Tabla `elections.polls` + `elections.polls_results` vía `get_poll_series` (`data.py:277-334`): una fila por sondeo con columnas de metadatos (`pollster`, `sponsor`, `start_date`, `end_date`, `sample_size`, `days`, `mtype`, `computed`, …), los pesos calculados por `Computer` (`weight_over`, `weight_sample`, `weight_rating`) y una columna por partido con el `pct` (pivot). Filtros: `drange` sobre `days` (días antes de la elección) y `drop_mtypes` (por defecto excluye `aggr` y `online`, `forecaster.py:36`).
- Tabla `elections.events*` vía `get_event_series` (`data.py:195-247`): resultados finales del evento anterior y del actual (si existe), con `votes` copiado a `sample_size` y `computed=True` (`forecaster.py:202-203`).
- `get_parties()` / `get_pollsters()`: catálogo (colores, `quality` a priori de cada encuestadora).

### 2.2 `self.series` (salida de `build_series`)

`DataFrame` con `MultiIndex(date, pollster_id, sponsor_id)` ordenado por fecha (`forecaster.py:274-278, 289-294`). Filas = sondeos + 2 filas de eventos (anterior y actual, con `pollster` nulo). Columnas: `data_cols` (`forecaster.py:252-257`) + un bloque por nombre en `self.names` + la columna `'-'` con el resto hasta 100 (`forecaster.py:298`). En la ejecución de prueba: `(308, 31)`; `fc_series` `(306, 33)` (índice reducido a `date` por `reset_index(levels[1:])`, `forecaster.py:141`), `nfc_series` `(2, 33)`.

Variables temporales:

- `tfs = (date - date_start).days` (*time from start*, `forecaster.py:266`), `tte = (date_end - date).days` (`267`). Nota: **`fit()` no usa `tfs`**; `LocalKernelEstimator` recalcula sus propios deltas en días desde el primer sondeo (`stat.py:583-600`). `tfs` solo se usa en `print_weights`/`plot_bws`.
- `fc_index = date_range(date_start, date_end, freq='D')` (`forecaster.py:145`): 1492 días en el caso de prueba.

Pesos (`forecaster.py:270-271`):

```
weight_rating_i = weight_rating_i  (de Computer)  ó  quality(pollster_i) si es nulo
weight_i        = weight_over_i * weight_sample_i * weight_rating_i
```

donde, según `computer.py:573-577`, `weight_sample = sqrt(proc_sample / mediana_evento(proc_sample))`, `weight_over` ∈ [0,1] penaliza solapamiento/sobrepublicación por encuestadora (`computer.py:590-598`) y `weight_rating = log1p(rating)/log1p(mean(rating))` (`computer.py:1186`). Estadísticos observados: `weight` media 2.35, rango [0, 5.26]; `weight_rating` [0.24, 3.08]; `weight_over` mínimo 0 (sondeos anulados).

### 2.3 `self.forecast` y `self.fc_stat`

- `forecast`: `DataFrame(float)` índice diario `fc_index`, columnas `names + ['-']` (`forecaster.py:301-305`).
- `fc_stat`: `DataFrame(object)` mismo índice, columnas `names`; **cada celda es un `dict`** `{'mean','cmin','cmax','err','nobs','neff'}` o `None` (`forecaster.py:306-309, 356, 396`).

---

## 3. Métodos

### 3.1 `__init__` (`forecaster.py:32-129`)

Parámetros y valores por defecto: `scope`, `event_date`, `drop_mtypes=['aggr','online']`, `drange=None` (normalizado por `norm_range` a `(0, None)`, `utils.py:105-131`), `alpha=0.05`, `bmap=None`, `reg_params=None`, `verbose=0`, `path=os.getcwd()`. Resolución de `bmap` (`111-114`): si es `str` se toma `event_params['bmaps'][bmap]`; si no es `dict` se usa la lista `parties.polls`. El docstring de la clase ("Polls corresponding to a single election event", `45`) está desactualizado.

### 3.2 `set_reg_params` (`forecaster.py:147-181`)

Valores por defecto que se pasan tal cual a `LocalKernelEstimator`:

| clave | defecto | significado real en el código |
|---|---|---|
| `kernel` | `'gaussian'` | único núcleo implementado para la regresión (`stat.py:294-306`) |
| `bw_type` | `'adaptive'` | ancho de banda por punto de evaluación (`stat.py:507-514`) |
| `bw` | `'isj'` | piloto por Improved Sheather-Jones / Botev (`stat.py:378-401`); alternativas `'scott'`, `'silverman'` o un número |
| `bw_kwargs` | `{'n_iter': 3, 'min_delta': None}` | iteraciones del esquema adaptativo; `None` → 0 → nunca para antes de `n_iter` (`stat.py:458, 478`) |
| `poly_deg` | `1` | grado del polinomio local (`poly_deg=0` no funciona, ver hallazgo 7) |
| `cov_type` | `'hac'` | única opción implementada (`stat.py:759, 777`) |
| `cov_kwargs` | `{'hac_lags': 1, 'kernel': 'bartlett'}` | pesos de Bartlett `1 - l/(L+1)` (`stat.py:309-311`) |

### 3.3 `load_events` / `load_polls` (`forecaster.py:183-219`)

`date_end` = fecha del evento; `date_start` = fecha del evento anterior (`186-188`). Si solo hay un evento, `date_start = date_end - 12 meses - 6 días` (constante mágica, `191`). `load_polls` fija `date_first`/`date_last` con el mín/máx de `date` (`217`).

### 3.4 `build_series` (`forecaster.py:221-311`)

1. Concatena sondeos y eventos (`260-263`), calcula `tfs`/`tte` (`266-267`) y el peso (`270-271`).
2. `build_blocks(bmap, colors)` (`utils.py:10-64`) crea la tabla de bloques (`parties`, `color`); `group_results` (`utils.py:67-102`) suma los porcentajes de los partidos de cada bloque (`groupby(level=0, axis=1).sum(min_count=1)`, API deprecada).
3. Filtra sondeos con `computed == False` (`296`), añade `'-' = 100 - Σ bloques` (`298`) e inicializa `forecast`/`fc_stat` (`301-309`).

Supuestos: los porcentajes de los partidos de un bloque son aditivos; los sondeos sin `computed` no tienen pesos válidos; el color del bloque es el del primer partido (falla con `KeyError` si ese partido no aparece en los sondeos, `utils.py:59`).

### 3.5 `fit(name, max_fc=0, ret_stat=False)` (`forecaster.py:313-360`)

```
df  = fc_series[fc_series[name].notnull()]                # sondeos con dato para `name`, índice = date
px  = fc_index ∩ [min(df.date), max(df.date) + max_fc días]   # rejilla de evaluación (343)
reg = LocalKernelEstimator(df[name], weights=df.weight, **reg_params).fit(px, alpha).reindex(fc_index)
```

Si `df` está vacío devuelve `None` (`339-340`). Con `ret_stat=True` devuelve `(reg['mean'], Series de dicts)` (`354-358`). Tiempo medido: 0,7 s por partido (1135 días de rejilla, 306 sondeos).

### 3.6 Matemática de `LocalKernelEstimator` (con `Kernel` y `LeastSquaresEstimator`)

Notación: `t_i` día del sondeo `i` (delta en días desde el primer sondeo, `stat.py:583-600`), `y_i` porcentaje, `w_i` peso externo, `n` sondeos, rejilla de evaluación `p` (días).

**(a) Normalización de pesos (Kish)** — `stat.py:571-576` y de nuevo en `Kernel` (`278-279`) y en cada regresión local (`709-710`):

```
n_eff = (Σ w_i)² / Σ w_i²,     w_i ← w_i · n_eff / Σ w_i     (los pesos suman n_eff)
```

En el caso de prueba: `n=306`, `n_eff=248`.

**(b) Ancho de banda piloto fijo `h0`** (`stat.py:424-445`). Por defecto ISJ (Botev, Grotowski & Kroese 2010): histograma ponderado de 256 celdas sobre `[min−0.5σ, max+0.5σ]` (`362-375`), DCT-II, punto fijo `t* = ξγ^{[5]}(t*)` resuelto por `brentq` con un bracket que se duplica hasta 1 (`334-360`), `h0 = sqrt(t*)·(max−min)` (`398-400`). Si no converge o `t* < min_t`, `h0 = 2π·media(Δt ordenados)` (`384-386`, sin documentar). Alternativas: Scott `1.059·σ·n^{-1/5}` (`403-410`), Silverman `0.9·min(σ, IQR/1.349)·n^{-1/5}` (`412-422`). Valores observados para el eje temporal del PP: ISJ 71.0 días (truncado, ver hallazgo 1), Scott 109.6, Silverman 93.1.

**(c) Ancho de banda adaptativo `h_p`** (`stat.py:447-490`), tipo *balloon* (depende del punto de evaluación `p`, no del dato):

```
h_p^{(0)} = h0
para k = 1..n_iter:
    f̃_p = Σ_i w_i · exp(−(p − t_i)² / (2 h_p^{(k−1)2}))        # (471) densidad piloto NO normalizada
    G    = media geométrica de f̃_p sobre la rejilla p             # (473)
    h_p^{(k)} = h_p^{(k−1)} · sqrt(G / f̃_p)                      # (474)
    parar si RMS(h^{(k)} − h^{(k−1)}) < min_delta                 # (476-481); con min_delta=None nunca
```

Es la regla de Abramson (`λ ∝ f^{-1/2}`) solo en la primera iteración; en las siguientes `f̃` no se divide por `h_p`, por lo que la iteración converge hacia `h_p ∝ f_p^{-1/3}` (véase hallazgo 3) y **G se calcula sobre la rejilla de predicción, no sobre la muestra**, lo que hace que `h_p` dependa de qué días se pidan (hallazgo 2). Valores observados (rejilla completa): mín 60.8, mediana 69.9, **máx 128.6 días en la última fecha** (frontera derecha, baja densidad de sondeos futuros).

**(d) Pesos de núcleo y ventana local** (`stat.py:492-520`, `909-927`):

```
K_{p,i} = exp(−(p − t_i)² / (2 h_p²))          # sin constante de normalización (306)
ω_{p,i} = K_{p,i} · w_i                        # (914)
ventana(p) = { i : |ω_{p,i}| ≥ 1e−2 }          # (915)  ≈ |p − t_i| ≤ h_p·sqrt(2·ln(100·w_i))
```

Dentro de la ventana los `ω` se vuelven a normalizar a su `n_eff` local (`709-710`). En la última fecha: 100 sondeos en ventana, `n_eff=48.0`, `dof=46`, 19 encuestadoras distintas.

**(e) Regresión local ponderada** (`LeastSquaresEstimator`, `stat.py:693-731, 792-815`). Con `X = [1, t_i]` (grado 1), `W = diag(ω)`:

```
β̂_p = (XᵀWX)⁻¹ XᵀW y            # vía pinv de W^{1/2}X (795-796)
ŷ(p) = [1, p] · β̂_p              # (822)
```

Nota: cada modelo local se evalúa sobre toda la rejilla (`964`) y solo se conserva la fila `p` (`973`): coste O(N²).

**(f) Varianza y CI** (`stat.py:753-778, 817-831`). Con `u_i = ω_i · x_i · r_i` (residuo `r_i = y_i − x_iβ̂`, `763`):

```
Σ_HAC = Σ_i u_i u_iᵀ + 0.5·Σ_i (u_i u_{i−1}ᵀ + u_{i−1} u_iᵀ)     # Bartlett L=1: pesos [1, 0.5] (765-769)
V(β̂) = (XᵀWX)⁻¹ Σ_HAC (XᵀWX)⁻¹ · n_eff/dof                       # (772-775),  dof = n_eff − 2 (722)
err(p) = sqrt([1,p] V [1,p]ᵀ)                                     # (830)
CI = ŷ(p) ± t_{1−α/2, dof} · err(p)                               # (831)
```

El "lag" es la fila adyacente en el orden por fecha (varios sondeos el mismo día cuentan como lags consecutivos; huecos irregulares también). Verificado numéricamente: en la última fecha `err=0.668`, `cmax−mean=1.360`, `t_{0.975,46}=2.013`.

**Qué es y qué no es este intervalo**: es un IC de la **media condicional del suavizador** (error de estimación de la recta local), no un intervalo de predicción de un sondeo ni del resultado electoral; no incluye el sesgo de suavizado ni la incertidumbre del ancho de banda; y como se discute abajo, ignora la correlación intra-encuestadora.

### 3.7 `fit_forecast(names=None, max_fc=0, fillna=False)` (`forecaster.py:362-403`)

Bucle sobre `names` (por defecto `self.names`), escribe `forecast[name]` y `fc_stat[name]`, y **dentro del bucle** recalcula `forecast['-'] = 100 − Σ_{names} forecast` (`395`, usa el subconjunto `names`, no `self.names`). Con `fillna=True` aplica `fillna(method='ffill')` (`400-401`, deprecado) a ambos frames: el valor y el diccionario de estadísticos del último día estimado (`date_last + max_fc`) se repiten hasta `date_end`. No hay normalización a 100: los partidos se ajustan independientemente y `'-'` absorbe el resto (puede ser negativo).

Verificado: con `names=['PP','PSOE']`, `max_fc=3`, `fillna=True`: `forecast.loc['2027-08-22']` = PP 31.91, PSOE 27.09, `'-'` 40.99; `fc_stat['PP']['err']` idéntico (0.6756) el 2026-09-14 y el 2027-08-16.

### 3.8 `get_forecast(sort, date, prefix)` (`forecaster.py:405-449`)

Carga desde CSV si hay `prefix`; si `forecast` está vacío llama `fit_forecast()` con defectos (`max_fc=0`, sin ffill). `date=None` → última fila con algún dato (`442`), que tras `fillna=True` es la fecha electoral. Devuelve `Series` redondeada a 2 decimales.

### 3.9 `save_forecast` / `load_forecast` (`forecaster.py:451-477`)

Guarda `fc/{prefix}.csv` y `fc/{prefix}-stat.csv` (dicts serializados con `str()`); al cargar, `applymap(literal_eval)` (`472`). Funciona mientras ningún dict contenga `nan` (`literal_eval('nan')` lanza `ValueError`).

### 3.10 Consumo por `Simulator` (`simulator.py:118-128, 313-317`)

`Simulator` crea el `Forecaster` con `bmap=parties.event`, llama `fit_forecast(**kwargs)` y lee `forecast[n].loc[limit_date]`, `fc_stat[n].loc[limit_date]['err']` y `['nobs']`, con `limit_date = event_date − drange[0]`. Requiere `fillna=True` (de lo contrario la celda es `None`). Por tanto, el "error" que entra en la simulación es el `err` del último día estimado, congelado.

### 3.11 Métodos de visualización (`forecaster.py:479-1182`, lectura ligera)

- `print_weights(name, date)` (`479-577`): tabla de los sondeos que influyen en `date` con `weight_kernel`, `weight` y sus componentes; construye el estimador con `df.tfs` como "y" (`508`, solo se usan los pesos de núcleo); filtra `weight ≥ 1e-2` (`525`).
- `plot_weights` (`579-717`): dispersión de sondeos (tamaño/color por peso), recta local, CI y curva del núcleo en `date`; etiqueta fija "CI 95%" (`653`).
- `plot_bws` (`719-797`): serie del ancho de banda adaptativo por día frente al piloto fijo.
- `plot_forecast_output` (`799-953`): barras apiladas (o semicírculo polar) del último pronóstico, opcionalmente agrupado por bloques y con "Otros" hasta `vmax`.
- `plot_forecast_series` (`955-1178`): series de pronóstico, IC (`show_ci`), estrellas con resultados de eventos, puntos de sondeos, resaltado de una encuestadora. `block_map` es siempre `bool`, así que `if block_map is not None` (`1042, 1063, 1082, 1101`) es siempre cierto.
- `get_path` (`1180-1182`).

---

## 4. Valoración estadística

### 4.1 Frontera y extrapolación

Gracias al grado 1, en la última fecha la estimación (31.91) es coherente con los últimos sondeos (32.4, 31.1, 34.5, 25.5, 33.6, 31.5, 33.7 con pesos 3.8, 1.9, 1.7, 3.4, 1.0, 0.9, 0.9). Sin embargo el ancho de banda adaptativo **se duplica en la frontera** (128.6 frente a ~70 días en el interior) porque la densidad piloto cae al no haber sondeos futuros: el estimador en la fecha más importante es el más suavizado y el que más tarda en reflejar cambios recientes. Comparación en la última fecha: adaptativo-ISJ 31.91 (err 0.86 con rejilla corta), fijo-ISJ 31.87, fijo `h=30` 31.18 (err 1.21).

La extrapolación con `max_fc` es la prolongación de la recta local ajustada en la frontera: h=+10 d → 31.98 (err 0.71), +30 → 32.02 (0.76), +60 → 31.96 (0.80), +90 → 31.79 (0.82). El crecimiento del error es solo el del término `[1,p]V[1,p]ᵀ`; no hay componente de "deriva de la opinión" con el horizonte. Con `fillna=True` el valor y el error quedan además congelados desde `date_last + max_fc` hasta `date_end` (11 meses en el caso de prueba).

### 4.2 House effects y correlación intra-encuestadora

El modelo no resta el sesgo histórico por encuestadora (las columnas `bias`, `bias_dev_adj` existen en `series` pero `fit` solo usa `df[name]` y `df.weight`, `forecaster.py:337-352`). En la ventana de la última fecha los residuos medios por encuestadora tienen desviación típica 2.78 pp (rango −6.3 a +5.6) frente a una sd residual de 3.11: la mayor parte de la varianza residual es "de casa". El error estándar HAC (0.668) es **menor** que HC0 (0.744), HC1 (0.760) y que un sandwich agrupado por encuestadora (0.89–0.91, 19 clusters): el IC publicado está infraestimado ~35 %.

### 4.3 Comparación con FiveThirtyEight (polls-only) y con modelos de espacio de estados

- 538: media ponderada con pesos por rating, tamaño y **decaimiento exponencial por antigüedad**, más **ajuste de house effects** (estimado en el propio período comparando cada casa con la media) y ajuste por tipo de población. Aquí el decaimiento es un núcleo gaussiano simétrico en el tiempo (equivalente a un suavizado, no a una media "hacia atrás"), no hay ajuste de casa dentro del período y la tendencia (grado 1) sustituye al "trend line adjustment" de 538.
- Jackman (2005) / Linzer (2013): paseo aleatorio latente `μ_t = μ_{t−1} + ε_t`, observaciones `y_i ~ N(μ_{t_i} + δ_{casa(i)}, σ_i²)` con `σ_i²` derivada del tamaño muestral; estimación bayesiana. Ventajas frente al Forecaster actual: (i) house effects `δ` identificados y restados; (ii) la incertidumbre **crece con el horizonte** (`Var(μ_{T+h}) = Var(μ_T) + h·σ_ε²`), lo que resuelve el "congelado" de 4.1; (iii) suavizado adaptativo sin elegir ancho de banda; (iv) compatible con la filosofía divulgativa si se implementa con un filtro de Kalman + suavizador RTS "a mano" en `stat.py`. La regresión local puede mantenerse como método de referencia/visual, pero para el intervalo de la fecha electoral el enfoque de espacio de estados es más defendible.

### 4.4 Composición

Cada bloque se ajusta por separado; `Σ forecast` no está restringida y `'-'` puede ser negativo (`forecaster.py:298, 395`). Una transformación log-ratio (ALR/ILR) o al menos una renormalización final evitarían incoherencias, sobre todo cuando `Simulator` reparte escaños.

---

## 5. Pipeline (pasos ordenados)

1. `__init__` (`forecaster.py:32-129`): lee `params.json`, resuelve `bmap`, fija `reg_params` por defecto (`147-181`).
2. `load_events` (`183-205`): `date_start` = evento anterior, `date_end` = evento; filas de resultados.
3. `load_polls` (`207-219`): sondeos + pivot de porcentajes; filtros `drange`, `drop_mtypes`.
4. `build_series` (`221-311`): concat, `tfs/tte`, `weight = over·sample·rating`, bloques, filtro `computed`, columna `'-'`, frames vacíos.
5. `fit(name)` (`313-360`): rejilla `px`; `LocalKernelEstimator`: fechas→días y Kish (`stat.py:550-608`); piloto ISJ (`378-401`); `h_p` adaptativo (`447-490`); `ω = K·w` (`492-520, 914`); WLS local grado 1 por día (`693-731, 792-815`); sandwich HAC y CI t (`753-778, 817-831`); `reindex(fc_index)`.
6. `fit_forecast` (`362-403`): bucle por bloque, `'-'`, `ffill` opcional.
7. `get_forecast` / `save_forecast` / `load_forecast` (`405-477`).
8. `Simulator.build_forecast` (`simulator.py:311-330`) lee `mean/err/nobs` en `limit_date`.

---

## 6. Hallazgos

Severidad: alta / media / baja. Cada uno con evidencia, impacto y cómo comprobarlo.

### H1 [bug, media] El ancho de banda calculado por método se trunca a 3–5 caracteres — `stat.py:285-289`
`np.repeat('isj', kvar)` crea un array `dtype='<U3'`; la asignación `self.bw[i] = float` (289) convierte el número a cadena truncada a 3 caracteres y luego `np.array(..., dtype=float64)` (291). Evidencia: `Kernel.get_bw_isj(x, ones)` = 0.33630 → `Kernel(x, bw='isj').bw[0]` = 0.30000; para el eje temporal del PP el ISJ queda en 71.0 exactamente. Con `'scott'` (`<U5`) se trunca a 5 caracteres. Impacto: ancho de banda incorrecto (hasta −11 % en el ejemplo; si `h < 0.1` en datos a escala pequeña se convierte en `0.0` → división por cero en el núcleo). Prueba: `assert np.isclose(Kernel(x, bw='isj', bw_type='fixed').bw[0], Kernel.get_bw_isj(x, np.ones(len(x))))`. Corrección: `self.bw = [self.bw] * self.kvar` (lista Python) antes del bucle.

### H2 [stat/design, media] El ancho de banda adaptativo depende de la rejilla de predicción — `stat.py:471-474`
`G = gmean(f̃_p)` se calcula sobre los puntos `p` pedidos, no sobre la muestra. Evidencia: `h` en la última fecha = 128.6 (rejilla completa), 83.7 (últimos 41 días), 120.7 (rejilla +90 días); estimación 31.907/err 0.668 frente a 31.938/0.821. Impacto: `fit(max_fc=0)` y `fit(max_fc=10)` dan valores distintos para la misma fecha; `print_weights`/`plot_weights` (rejilla sin `max_fc`) muestran pesos que no son los usados en `fit_forecast(max_fc=10)`. Prueba: la del texto. Corrección: calcular `f̃` y `G` en los puntos muestrales `t_i` (Abramson clásico) y evaluar `h` en `p` por interpolación, o fijar `G` a partir de la muestra.

### H3 [stat, media] La densidad piloto no se normaliza en las iteraciones ≥ 2; el ancho de banda de frontera se infla con `n_iter` — `stat.py:471-474`
Como `f̃_p ∝ h_p·f(p)`, la actualización converge a `h_p ∝ f_p^{-1/3}` en lugar de `f_p^{-1/2}`, y solo la iteración 1 es Abramson. Evidencia: `h` en la última fecha 71 (n_iter 0) → 98.8 → 117.5 → 128.6 (defecto) → 141 (n_iter ≥ 10), mientras la mediana apenas cambia (70.4 → 69.6). Impacto: la fecha más relevante (última) usa una ventana ~2× la del interior; el pronóstico "actual" es el más rezagado. Prueba: `Kernel(x, weights=w, bw='isj', n_iter=k).get_bw_adaptive(p=z)[-1]` para `k=1..30`; regresar `log h_p` sobre `log f_p`. Corrección: dividir `f̃_p` por `h_p·√(2π)·Σw` y documentar que es un estimador *balloon*; considerar limitar `h_p ≤ c·h0`.

### H4 [stat, media] El error estándar ignora la correlación intra-encuestadora (house effects) — `stat.py:753-778`; `forecaster.py:337-352`
HAC con 1 lag en orden de filas (varios sondeos/día, huecos irregulares) no captura la dependencia por casa. Evidencia (ventana última fecha, 100 sondeos, 19 casas): SE modelo 0.668 < HC0 0.744 < cluster-por-encuestadora 0.89–0.91; sd de residuos medios por casa 2.78 pp frente a sd residual 3.11. Impacto: IC ~35 % demasiado estrecho; el `err` que alimenta `Simulator` hereda la infraestimación. Prueba: reproducir el sandwich agrupado (script de este informe). Corrección: `cov_type='cluster'` por `pollster_id`, o estimar y restar `δ_casa` antes del suavizado.

### H5 [stat, media] Con `fillna=True` el pronóstico y su error quedan congelados hasta la fecha electoral — `forecaster.py:399-401`; `simulator.py:313-317`
Evidencia: `fc_stat['PP']['err']` = 0.6756 tanto el 2026-09-14 como el 2027-08-16; `forecast` idéntico. Impacto: la incertidumbre no crece con el horizonte (11 meses), y `get_forecast()` devuelve como "última fecha" la electoral con un valor que en realidad es de `date_last + max_fc`. Prueba: la del texto. Corrección: modelar deriva con el horizonte (p. ej. `Var_h = err² + h·σ_ε²` estimando `σ_ε²` de la varianza de incrementos históricos) o, al menos, documentar que es un "snapshot".

### H6 [bug, media] `fit_forecast` revienta si un bloque no tiene datos; `build_series` revienta si el primer partido de un bloque no aparece en los sondeos — `forecaster.py:339-340, 391`; `utils.py:59`
Evidencia: columna toda `NaN` → `fit` devuelve `None` → `TypeError: cannot unpack non-iterable NoneType object`; `bmap={'PP':'PP','X':['CUP']}` → `KeyError: 'CUP'` en `build_blocks`. El docstring (`376`) promete "will be ignored". Corrección: `if res is None: continue` con aviso; en `build_blocks` usar `parties.get(...)` con color aleatorio.

### H7 [bug, baja] `poly_deg=0` (Nadaraya-Watson) no funciona — `stat.py:715-720, 740-748`
`exog = np.ones(n)` queda 1-D → `ValueError: shapes (41,) and (54,1) not aligned`. Impacto: no es posible comparar local linear con NW, comparación natural en un repositorio divulgativo. Corrección: `exog = np.ones((n, 1))`.

### H8 [bug, baja] `Kernel.get_bw_isj(x)` sin pesos falla — `stat.py:362-375`
`np.histogram(..., weights=None)` devuelve `int64` y `grid /= n` lanza `UFuncTypeError`. Solo afecta al uso estático (con instancia, `weights=ones` evita el fallo). Corrección: `grid = grid.astype(float) / n`.

### H9 [bug/design, baja] La columna `'-'` se calcula con el subconjunto `names` y dentro del bucle — `forecaster.py:395`
Con `names=['PP','PSOE']`, `'-'` = 40.99 (no "otros"); llamadas sucesivas con distintos `names` dejan `'-'` inconsistente. Corrección: calcular una vez tras el bucle con `self.names`.

### H10 [reproducibility, media] APIs de pandas deprecadas y versiones no fijadas — `forecaster.py:400-401, 472, 1046-1047`; `utils.py:99`
`fillna(method='ffill')`, `applymap`, `groupby(axis=1)` emiten `FutureWarning` en pandas 2.2.2 (capturados en la ejecución) y desaparecen en pandas 3. `requirements.txt` fija numpy 1.25.2/scipy 1.9.3 pero el entorno usa numpy 2.4.4/scipy 1.17. Corrección: `.ffill()`, `.map`, `T.groupby(level=0).sum().T`; fijar versiones probadas.

### H11 [reproducibility, alta] `files/params.json` (mapas de partidos/bloques) está fuera del control de versiones — `forecaster.py:107-109`; `.gitignore:2`
Sin ese fichero `Forecaster.__init__` falla; el notebook usa `path='.'` (`PollsForecast.ipynb` celda 1) y no hay `notebooks/params.json`, luego depende del directorio de trabajo. Corrección: versionar la configuración (p. ej. `config/params.json`) y resolver la ruta relativa al paquete.

### H12 [bug, baja] Ida y vuelta CSV de `fc_stat` falla si algún dict contiene `nan` — `forecaster.py:472`
`literal_eval("{'a': nan}")` → `ValueError`. En la prueba no apareció ningún `nan` (`err` siempre finito), pero es un fallo latente. Corrección: guardar `fc_stat` en formato largo (columnas `err, cmin, cmax, nobs, neff`) o JSON.

### H13 [design, baja] `fc_stat` almacena `dict` por celda — `forecaster.py:306-309, 356, 396`
Impide operaciones vectorizadas, obliga a `applymap` en los gráficos (`1046-1047`) y a H12. Corrección: `DataFrame` con `MultiIndex` de columnas `(name, stat)`.

### H14 [dead-code, baja] `if block_map is not None` siempre cierto — `forecaster.py:1042, 1063, 1082, 1101`
`block_map` se asigna a `True/False` (`1032-1040`). Inofensivo (`group_results` con bloques identidad), pero engañoso.

### H15 [docs, baja] Etiqueta "CI 95%" fija aunque `alpha` sea configurable — `forecaster.py:653`.

### H16 [design, baja] `print_weights` pasa `df.tfs` como variable dependiente — `forecaster.py:508`
Funciona porque `Estimator` toma el índice como `x`, pero el `endog` es absurdo; usar `df[name]` como en `plot_weights` (`641`).

### H17 [performance, baja] Cada regresión local se evalúa sobre toda la rejilla — `stat.py:964, 973`
O(N²) en días de rejilla; 0,7 s/partido hoy, pero crece cuadráticamente (ciclos de 4 años ≈ 1500 días). Corrección: `loc_est.fit(self.pred[pos:pos+1], alpha)`.

### H18 [stat, baja] Detalles de la implementación ISJ que la separan de Botev (2010) — `stat.py:362-401`
(i) el histograma se construye sobre `[min−0.5σ, max+0.5σ]` pero `h = sqrt(t*)·(max−min)` sin el relleno (368-369 vs 380, 398); (ii) el histograma ponderado se divide por `nobs` aunque los pesos suman `n_eff` (373), luego no integra 1; (iii) el retroceso `2π·media(Δt)` (384-386) y el recorte `n∈[50,1050]` del bracket (335) no están documentados. Prueba: comparar con `KDEpy.bw_selection.improved_sheather_jones` en muestras gaussianas.

### H19 [stat, baja] Sin restricción composicional — `forecaster.py:298, 395`
`Σ forecast` libre; `'-'` puede ser negativo. Corrección: ALR/ILR o renormalización final documentada.

### H20 [stat, baja] Semántica de los pesos y de los grados de libertad — `stat.py:574, 709-722, 915`
Los pesos se re-normalizan a Kish tres veces (global, núcleo, local) y `dof = n_eff_local − 2` es heurístico; el umbral `1e-2` (915) sobre `K·w` no normalizado hace que la ventana dependa de la escala de `w`. Documentar como convención o umbralizar solo `K`.

### H21 [docs, baja] Docstrings desactualizados — `forecaster.py:45` ("Polls corresponding to a single election event"), `376` (bloques ausentes "ignorados", ver H6), `set_reg_params` no documenta que `kernel`/`cov_type` solo admiten un valor.

### H22 [design, baja] Constante mágica cuando solo hay un evento — `forecaster.py:191` (`date_end − 12 meses − 6 días`).

---

## 7. Preguntas abiertas para el autor

1. ¿El ancho de banda ISJ = 71.0 exacto se había observado? ¿Se calibró `n_iter=3` sabiendo que infla el ancho de banda en la frontera, o fue accidental?
2. ¿Es intencionado que el "pronóstico" electoral sea el promedio en `date_last + max_fc` congelado (snapshot), dejando la deriva con el horizonte para `Simulator.v2err`? ¿Cómo se estima `v2err` y usa el horizonte real o `drange`?
3. ¿Por qué no se restan los sesgos por encuestadora calculados por `Computer` (`bias`, `bias_dev_adj`) antes del suavizado? ¿Se descartó por identificabilidad o por diseño?
4. ¿Debe `'-'` representar "otros partidos" (100 − Σ self.names) o el resto de los bloques ajustados? El código hace lo segundo cuando se pasa `names`.
5. ¿`weight_over = 0` pretende anular un sondeo (queda excluido por el umbral 1e-2) o es un artefacto?
6. ¿Se contempla alternar a un modelo de espacio de estados (Kalman/RTS implementado en `stat.py`) para la fecha electoral, manteniendo la regresión local como método explicativo?
7. ¿Qué `reg_params` usa `Simulator` en producción (el scout indica que pasa `self.reg_params`)? ¿Coinciden con los defectos de `set_reg_params`?
8. ¿Los `fc/*.csv` guardados se consideran artefactos de resultados a versionar (para reproducibilidad de publicaciones) o caché?
