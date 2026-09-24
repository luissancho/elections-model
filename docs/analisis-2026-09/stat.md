# Revisión matemática de `mtpy/core/utils/stat.py`

Fichero: `/Users/luiss/HD/Proyectos/Code/elections-model/mtpy/core/utils/stat.py` (1144 líneas, sin docstrings).
Entorno de verificación: Python 3.11.9, numpy 2.4.4, scipy 1.17.0, statsmodels (instalado). Todas las comprobaciones numéricas citadas se ejecutaron con scripts en `.../scratchpad/t1.py` a `t6.py` (solo lectura, sin tocar repo ni BD, salvo una lectura de `Forecaster('es','2027-08-22','max').build_series()` para medir el impacto sobre datos reales).

---

## 1. Propósito y mapa del módulo

`stat.py` es la librería estadística "hecha a mano" del proyecto. Todo el modelo (Computer, Forecaster, Simulator) y la capa de gráficos (`dataviz.py`) se apoyan en ella. Contiene cinco clases:

| Clase | Rol | Usada en |
|---|---|---|
| `Stat` (l. 69-247) | Estadísticos descriptivos ponderados (media, varianza, cuantiles, IC, winsorización, escalados) | `computer.py:564,717-737,2351`, `simulator.py:402`, `dataviz.py:2120` |
| `Kernel` (l. 249-519) | Kernel gaussiano, selectores de ancho de banda (Scott, Silverman, ISJ de Botev) y ancho adaptativo iterativo | `forecaster.py:740`, internamente por los estimadores |
| `Estimator` (l. 521-658) | Base: normaliza entradas (`exog`, `endog`, `weights`), convierte índices temporales a deltas en días, `r2_score` genérico | – |
| `LeastSquaresEstimator` (l. 661-856) | Mínimos cuadrados ponderados (WLS) vía pseudoinversa, covarianza HAC (Newey-West), bandas de confianza | `computer.py:1602,1674` (estimadores de error y de escaños), `dataviz.py:2689`, `dates.py:1079` |
| `LocalKernelEstimator` (l. 859-993) | Regresión local lineal ponderada por kernel (tipo LOESS con kernel gaussiano) | `forecaster.py:348,509,617`, `computer.py:1497,2412,2542`, `dataviz.py:2697` |
| `KernelDensityEstimator` (l. 996-1144) | KDE 1-D/2-D con pesos y tratamiento de outliers | solo `dataviz.py:1959,2171` |

Helpers externos que condicionan el comportamiento: `array_adjust`/`array_shape` (`helpers.py:1305-1348`), `is_timeseries` (`helpers.py:1283`), `ts_to_delta`/`ts_from_delta` (`dates.py:715-877`).

---

## 2. `Stat`: estadísticos descriptivos ponderados

### 2.1 Construcción y normalización de pesos (`__init__`, `set_params`, l. 71-111)

Entradas: `data` (1-D, se aplana con `array_adjust(·, 0)`), `weights` (opcional, 1-D; por defecto unos), `dist ∈ {'norm','t'}` (default `'norm'`), `dropna=False`, `outliers ∈ {'keep','group','remove'}` (default `'keep'`), `n_dev=3`.

`set_params` (l. 101-111):
- `vind = ~isnan(data)`; los pesos de los NaN se ponen a NaN.
- **Tamaño muestral efectivo de Kish**: `neff = (Σ w_i)² / Σ w_i²` sobre las observaciones válidas.
- Los pesos se reescalan para que **sumen `neff`**: `w_i ← w_i · neff / Σ w`. Esto implica `Σ w_i = neff` y `Σ w_i² = neff` (propiedad clave que se usa implícitamente después).
- `dof = neff − 1`.

Esta normalización es un buen diseño: convierte pesos arbitrarios (de fiabilidad) en pesos "equivalentes a frecuencias" de forma que las fórmulas clásicas con `ddof=1` producen el estimador insesgado para *reliability weights*.

### 2.2 Fórmulas implementadas

| Método | Fórmula (con `w` ya normalizados a `neff`) | Comprobación |
|---|---|---|
| `mean` (l. 145) | `μ = Σ w_i x_i / neff` = media ponderada | = `np.average(x, weights=w)` ✓ |
| `var` (l. 153) | `σ² = Σ w_i (x_i − μ)² / (neff − 1)` | Coincide exactamente con el estimador insesgado para reliability weights `Σ w̃(x−μ)² / (1 − Σ w̃²)` y con `DescrStatsW(x, w·neff/Σw, ddof=1).var` (t1 §1) ✓ |
| `std` | `√σ²` | ✓ |
| `std_mean` (l. 198) | `σ / √neff` | = `DescrStatsW.std_mean` ✓ |
| `quantile(q)` (l. 161-172) | Cuantil ponderado tipo "inverted CDF": ordena, `cw = cumsum(w)`, `tgt = q·neff`, `i = searchsorted(cw, tgt)`; si `tgt == cw[i]` (isclose) promedia `values[i], values[i+1]` | Sin pesos equivale a Hyndman-Fan tipo 2 (`np.quantile(method='averaged_inverted_cdf')`): `q=.25` de 1..8 → 2.5 (numpy default lineal daría 2.75); mediana coincide con `np.median` ✓ |
| `median` | `quantile(0.5)` | ✓ |
| `dev(n)` (l. 175) | `(μ − nσ, μ + nσ)` | ✓ (anotación de tipo `-> Self` incorrecta, devuelve tupla) |
| `ppf(q)` (l. 218) | `t.ppf(q, df=neff−1)` si `dist='t'`, si no `norm.ppf(q)` | ✓ |
| `conf(α)` (l. 182-188) | si `α<1`: `ppf(1−α/2)·σ`; si `α≥1` (múltiplo de sigma): `q = erf(α/√2)` y devuelve `ppf(q)·σ` | **Bug**: `erf(n/√2)` es la cobertura bilateral `P(|Z|<n)`, no `1−α/2`; `ppf(0.683)=0.475` ⇒ `conf(1)=0.475σ`, `conf(2)=1.69σ`, `conf(3)=2.78σ` (t1 §5). Debería ser `q = (1+erf(n/√2))/2` (⇒ `ppf(q)=n`) |
| `ci(α)` (l. 191) | `μ ± conf(α)`: intervalo para *observaciones individuales* bajo normalidad | ✓ (semántica distinta de `ci_mean`) |
| `conf_mean`, `ci_mean` (l. 202-215) | igual pero con `std_mean`: IC de la media | mismo bug con `α≥1` |
| `scale_standard` | `(x−μ)/σ` | ✓ |
| `scale_minmax(vmin,vmax)` | `(x−min)/(max−min)·(vmax−vmin)+vmin` | ✓ |
| `scale_sample` | `x · neff / Σx` | ✓ |
| `scale_weights` (l. 244) | `x ← x · [(Σx)²/Σx²] / Σx` = tratar `x` como pesos y normalizarlos a su propio `neff` | ✓ pero `*=` in-place falla con enteros (t1 §8) |

### 2.3 `winsorize` (l. 119-134)
- `n_dev ≥ 1`: límites `dev(n_dev)` (μ ± n·σ). `n_dev < 1`: límites `ci(1 − n_dev)`. Ojo: `n_dev=0.05` ⇒ `ci(0.95)` ⇒ `q = 0.525` ⇒ banda central del 5% (t1 §7). Probablemente la intención era `ci(n_dev)`.
- `action='remove'`: filtra y **recalcula `set_params`**, que sobrescribe `vind` con `~isnan(data_reducido)` (todo `True`, longitud reducida). Por tanto `vind` deja de ser una máscara sobre los datos originales.
- `action='group'`: `np.where(x_in, data, max(data[x_in]))` sustituye **ambas colas** por el máximo interior: `[-100, 10..13, 200]` → `[13, 10..13, 13]` (t1 §6). Los outliers bajos deberían ir a `vmin`. Llamado desde `computer.py:564` con `outliers='group', n_dev=ol_dev=3` sobre tamaños muestrales acotados en `[0, ol_max=5000]`; en la práctica `μ − 3σ` suele ser < 0, así que el bug es latente ahí.

### 2.4 `StatMeta` / `@pstatic` (l. 32-66)
Metaclase que permite `Stat.mean(x, weights=w)` sin instanciar: los args posicionales se mapean a los parámetros del **constructor** y solo los kwargs cuyo nombre coincida con la firma del método se pasan al método. Consecuencia: `Stat.quantile(x, 0.5)` interpreta `0.5` como `weights` y lanza `IndexError` (t1 §4); hay que escribir `Stat.quantile(x, q=0.5)`. Es una "magia" difícil de documentar para un repositorio divulgativo; una API explícita (funciones módulo `wmean(x, w)`, etc.) sería más clara.

---

## 3. `Kernel`: kernels y selección de ancho de banda

### 3.1 Construcción (l. 251-291)
Parámetros: `data` (n×k), `weights`, `kernel='gaussian'`, `bw_type='adaptive'`, `bw='isj'`, `n_iter=1`, `min_delta=None`. Los pesos se normalizan a `neff` como en `Stat`. Si `bw` no es array, `self.bw = np.repeat(bw, kvar)` (l. 285) y luego `self.bw[i] = get_bw_fixed(...)` (l. 289).

**Bug de truncado de cadenas (alto)**: `np.repeat('isj', 1)` crea un array `dtype='<U3'`; asignar un float lo convierte a cadena y lo **trunca a 3 caracteres** antes del `astype(float64)` de la l. 291. Comprobado: `Kernel(x, bw='isj').bw = [0.8]` frente al valor exacto `0.816132`; `'scott'` (5 chars) → `0.937` vs `0.937285`; `'silverman'` (9 chars) → exacto. Con datos reales del Forecaster (PP, 306 encuestas): ISJ `71.0` frente a `71.6196` días; Scott `109.6` frente a `109.6024`. La salida guardada en `notebooks/lab/PollsOptimize.ipynb:756-758` (`{'scott': 109.6, 'silverman': 93.146547, 'isj': 71.0}`) es la huella de este bug. Solución: `self.bw = [self.bw] * self.kvar` (lista) o `np.empty(kvar, dtype=float)`.

### 3.2 `kernel_gaussian(x, p, bw)` (l. 293-306)
`K(p, x) = exp(−‖p − x‖² / (2 bw²))`. **Sin la constante de normalización** `1/(√(2π)·bw)`: la integral sobre `p` vale `√(2π)·bw` (t2 K5). Para regresión local es irrelevante (los pesos se normalizan en el ajuste), pero para el KDE no (ver §6). Devuelve matriz `(n_p, n_x)`. El código para `k>1` (l. 300-304) no funciona por *broadcasting* de `bw`, pero nunca se ejecuta: `get_weights` llama siempre con una columna y multiplica (kernel producto).

### 3.3 `kernel_bartlett(bw)` (l. 309): `1 − l/bw` para `l = 0..bw−1`. Con `bw = hac_lags+1` da los pesos de Newey-West `1 − l/(L+1)` ✓.

### 3.4 Reglas fijas
- **Scott** (l. 403-409): `h = 1.059 · σ_w · n^{−1/5}` con `σ_w` ponderada (`Stat`) pero `n = nobs` (no `neff`). Constante `(4/3)^{1/5} ≈ 1.0592` ✓ (scipy usa factor `n^{−1/5}` sin 1.06, distinta convención).
- **Silverman** (l. 412-422): `h = 0.9 · min(σ_w, IQR/1.349) · n^{−1/5}` ✓ (IQR sin ponderar; `n = nobs`).

### 3.5 **Improved Sheather-Jones** (Botev, Grotowski & Kroese 2010) (l. 312-400)
Es un port casi literal de `KDEpy.bw_selection.improved_sheather_jones` / del `kde.m` de Botev:
1. `isj_grid`: histograma de 2⁸ = 256 bins en `[min − σ/2, max + σ/2]` (Botev/KDEpy: 2¹⁰ y *linear binning*), dividido por `n` (l. 374).
2. DCT tipo II (`fftpack.dct`), `I² = (1..N−1)²`, `a² = a[1:]²`. Botev usa `a2 = (a/2)²` y `f = 2π^{2l}Σ…`; aquí `a² = a²` y `f = ½π^{2l}Σ…` ⇒ idéntico ✓. `K0 = (2s−1)!!/√(π/2)` con `time = (k1·K0/(n f))^{2/(3+2s)}` equivale a `2·const·K0_botev/(N f)` ✓. Retorno `t − (2n√π f)^{−2/5}` ✓ (l. 331).
3. `isj_root`: `brentq` en `[0, 0.01(n−50)/1000]`, duplicando el extremo hasta ≥1 (KDEpy). **Diferencia**: KDEpy usa `10e−12 + 0.01(n−50)/1000`; aquí para `n ≤ 50` (tras el `clamp` a 50) `rb = 0`, `brentq(0, 0)` lanza `ValueError` en bucle, `rb` sigue siendo 0 y tras 100 iteraciones devuelve `bw = 0` ⇒ `get_bw_isj` cae al *fallback*. Comprobado: para `n = 20, 40, 50, 51` el resultado es exactamente el *fallback* (t2 K3).
4. `bw = √t* · x_range` (l. 398). `t*` está en unidades del **rango del histograma** (`x_range + σ`), no de `x_range`; Botev usa el rango extendido. Sesgo hacia abajo `x_range/(x_range+σ)` (0.86 en el test). KDEpy arrastra la misma inconsistencia.
5. *Fallback* `min_bw = 2π · mean(diff(sort(x)))` (l. 384-385), y también se aplica si `t* < (min_bw/x_range)²`. Heurística no documentada y ajena a Botev; con fechas de encuestas (muchos duplicados, espaciado medio 3.7 días) da 23.3 días.
6. **Pesos**: `np.histogram(x, weights=w)/n`. Como `Kernel` normaliza `w` a `neff`, el histograma suma `neff/n < 1`, y el algoritmo **no es invariante a la escala de los pesos**: `get_bw_isj(x, w) = 0.664`, `(x, 2w) = 0.366`, `(x, w·n/Σw) = 0.772`, sin pesos `0.816` (t2 K2). Debe normalizarse `grid /= grid.sum()`. Además `N` debería ser el número de valores únicos (Botev) y sin pesos el histograma es `int64` y el `/=` in-place falla (t2 K2).

### 3.6 Ancho de banda adaptativo (`get_bw_adaptive`, l. 447-490)
Partiendo de un piloto fijo `h₀` (ISJ/Scott/Silverman o número), para los puntos de evaluación `p`:
```
pdf_n(p_j) = Σ_i w_i · exp(−(p_j − x_i)² / (2 h_{n−1}(p_j)²))      (sin 1/h, sin normalizar)
h_n(p_j)   = h_{n−1}(p_j) · sqrt( gmean_j(pdf_n) / pdf_n(p_j) )
```
Parada: `n_iter` iteraciones (Forecaster/Computer usan 3) o RMS(`h_n − h_{n−1}`) < `min_delta` (en cuyo caso se **revierte** a `h_{n−1}`, l. 478).
Es una variante de Abramson (exponente ½), pero:
- el piloto se evalúa en los **puntos de predicción** (estimador "balloon": `h` depende de dónde se predice, no de cada dato) y la media geométrica se toma sobre `p`, no sobre la muestra;
- a partir de la 2ª iteración el "pdf" se calcula con `h` variable **sin dividir por `h`**, luego es `∝ h·f` y no una densidad; el esquema iterado no converge a nada definido (en el test los `h` siguen moviéndose: RMS 6.2 → 3.9 → 2.4 días con datos reales);
- si `pdf = 0` (punto lejos de los datos) se obtiene `h = 0` o `nan` (t4 M2).
En datos reales (PP, piloto ISJ 71.6 días) los anchos adaptativos van de 61 a 130 días (mediana 70).

### 3.7 `get_weights(p)` (l. 492-519)
Matriz `(n_p, n_obs)` de pesos kernel: producto sobre dimensiones de `kernel_gaussian(x_i, p_i, h_i)`; en modo adaptativo `h_i` es el vector por punto de predicción. **No incluye los pesos de observación** (se combinan fuera).

---

## 4. `Estimator` (base, l. 521-658)

`build_input` (l. 550-608): si `y is None` y `x` es pandas, `exog = x.index`, `endog = x.values`. `exog` se ajusta a `(n, kvar)`, `endog` a `(n,1)`, pesos a `(n,1)` normalizados a `neff`. Para cada columna detecta series temporales (`is_timeseries`), guarda `ranges[i] = (min, max)` (fechas) e infiere `ts_freq` (`Freq.infer`, default día); convierte a **deltas numéricos desde el mínimo** (`ts_to_delta`, en unidades de `ts_freq`) escribiendo floats sobre un array `datetime64` (funciona por la conversión float→ns→float; frágil) y trunca fracciones de la unidad (marcas horarias → días enteros, t6 M7).

`build_pred(p)` (l. 610-631): idem para los puntos de predicción; **con `p=None` copia `exog` (ya numérico) y vuelve a aplicar `ts_to_delta` ⇒ `ValueError`** para cualquier estimador con índice temporal (t4 M3). Es decir, `predict()` sin argumento y `LocalKernelEstimator.r2_score` no funcionan con los datos del modelo.

`r2_score` base (l. 634-644): `1 − Σ e² / Σ w (y − ȳ_w)²` (numerador sin pesos, denominador con pesos: inconsistente).

---

## 5. `LeastSquaresEstimator` (l. 661-856): WLS + HAC

Matriz de diseño (l. 726-733): `[1, x]` si `poly_deg=1`; si no, todas las combinaciones con repetición de grado ≤ `poly_deg` (términos cruzados incluidos). `dof = neff − poly_deg − kvar` (l. 736) — correcto solo si `kvar=1` o `poly_deg=1`; con `poly_deg=2, kvar=2` hay 6 columnas y `dof = n−4` en vez de `n−6` (t3 L3). Debe ser `neff − exog.shape[1]`. Si `dof ≤ 0` lanza `ValueError` (l. 738).

Ajuste (`fit`, l. 792-815), con `W̃ = diag(w)` normalizados a `neff`:
- `X_w = √W̃ X`, `y_w = √W̃ y`; `β = pinv(X_w) y_w` (= `(XᵀWX)⁻¹XᵀWy`, resuelto por SVD, numéricamente preferible a las ecuaciones normales) ✓ coincide con `sm.WLS` (t3 L1).
- `covb = pinv·pinvᵀ = (XᵀWX)⁻¹`; `scale = Σ w e² / dof`; `mcov = scale·covb` (no robusta). Nota: como `Σw = neff`, `scale` es la "media ponderada de e²" corregida por `neff − p`; difiere de statsmodels (que usa `n − p` y los pesos tal cual) por un factor de escala de pesos — convención interna coherente con `Stat.var`.
- `cov_type='hac'` (por defecto, `hac_lags=1`, Bartlett): `S = Γ₀ + Σ_l (1 − l/(L+1))(Γ_l + Γ_lᵀ)` con `Γ_l = Σ_t (w x e)_t (w x e)_{t−l}ᵀ`; `mcov = covb·S·covb · neff/dof` (l. 764-777). Comprobado frente a `sm.WLS(...).fit(cov_type='HAC', maxlags=1, use_correction=True)`: ratio 1.006, debido solo a la corrección `neff/(neff−p)` vs `n/(n−p)` ✓. No hay HC0–HC3 ni cluster; el "lag" es el orden de las filas (no el tiempo real), relevante con varias encuestas el mismo día y huecos.
- `serr = √scale`.

`predict(p, alpha)` (l. 817-856): `ŷ = Pβ`; si `alpha`: `perr = √diag(P·mcov·Pᵀ)` = **error estándar de la media condicional**, `cint = t_{1−α/2, dof}·perr`, columnas `mean, cmin, cmax, err`. Comprobado: `perr = 0.31` ≈ `se_mean` de statsmodels (0.285, difiere por HAC), mientras `se_obs` (intervalo de predicción) sería 2.43 (t3 L2). El comentario de la l. 688 ("Prediction intervals") y la docstring de `Forecaster.fit` ("standard error and confidence interval of the forecast") describen otra cosa: son **bandas de confianza de la tendencia suavizada**, no del resultado de una nueva encuesta ni de la elección. El Simulator combina ese `err` en cuadratura con `pct_err` (error histórico de encuestas, `simulator.py:530`), lo cual es coherente si se documenta como "incertidumbre de la media + error sistemático", pero no como intervalo de predicción de la regresión.

`r2_score` (l. 781-790): `1 − Σ(√w e)² / Σ(√w y − mean(√w y))²`. El centrado con la media de `√w·y` es incorrecto: da 0.445 donde el R² ponderado correcto (`statsmodels.rsquared`) es 0.211 (t3 L1/L4). Solo se usa en la leyenda de `plot_scatter`.

Índice del resultado: para `kvar=1` con fechas, se reconvierte con `ts_from_delta` y se hace `resample(freq).asfreq()`, por lo que el resultado queda **indexado por fecha con frecuencia diaria** (NaN en días sin predicción).

---

## 6. `LocalKernelEstimator` (l. 859-993): regresión local lineal ponderada

Parámetros por defecto: `kernel='gaussian'`, `bw_type='adaptive'`, `bw='isj'`, `bw_kwargs={}` (n_iter=1) — el modelo pasa `n_iter=3`, `poly_deg=1`, `cov_type='hac'` (`hac_lags=1`). `fit()` sin `p` no hace nada (l. 929-936); todo ocurre en `predict`:

1. `kws = Kernel(exog, weights, ...).get_weights(pred)` — matriz `(n_p, n)` de pesos kernel (§3.7); ISJ y el adaptativo se recalculan en **cada** llamada.
2. Para cada punto `p_j` (`get_local_estimator`, l. 909-927): `ω_i = kws[j,i] · w_i` (producto kernel × peso de encuesta, con `w` normalizados a media 1) y se **descartan** las observaciones con `ω_i < 1e−2` (soporte efectivamente compacto: con `h = 70` días y `w = 1`, `|p − x| ≤ h√(2 ln 100) ≈ 3.03h ≈ 212` días). Se construye un `LeastSquaresEstimator(exog[vind], endog[vind], weights=ω[vind], poly_deg, cov_type)` — es decir, dentro de cada ventana los `ω` se renormalizan a su `neff` local y el `dof` local es `neff_local − 2`.
3. `rloc = loc_est.fit(self.pred, alpha)` ajusta el WLS local y **predice en todos los puntos** `pred`, de los que solo se conserva la fila `j` (l. 966-975) — desperdicio O(n_p²) sin consecuencias numéricas.
4. Salida: `Series 'mean'` o `DataFrame ['mean','cmin','cmax','err','nobs','neff']` (con `nobs`/`neff` **locales**), reindexado a fechas diarias.

Con `poly_deg=1` es un **estimador local lineal** (no Nadaraya-Watson), equivalente a LOESS con kernel gaussiano truncado y ancho variable "balloon". Comportamientos observados:
- Si una ventana tiene `neff_local ≤ 2` (predicción lejos de los datos, inicio de serie con pocas encuestas, `bw` pequeño), `LeastSquaresEstimator` lanza `ValueError('Degrees of freedom...')` y **toda la predicción aborta** (t4 M1, M4 con 240 días). `get_local_estimator` solo protege el caso "todo NaN".
- `r2_score` (l. 886-897): correlación ponderada al cuadrado entre `y` e `ŷ` pero centrando ambas con la media de `ŷ`; llama a `self.predict()` (falla con fechas, §4) y **sobrescribe `self.result`** (t6 M9: pasa de 2 filas a 50).

### Nota estadística sobre la elección del ancho de banda
ISJ/Scott/Silverman son reglas para estimar la **densidad de `x`** (aquí, la distribución de las fechas de las encuestas), no para elegir el suavizado óptimo de `E[y|x]`. En regresión local el criterio habitual es validación cruzada (LOO/GCV) o AICc sobre los residuos de `y`. En datos reales el piloto ISJ resulta ≈ 72 días y el adaptativo 61–130 días; con `h = 70` días el kernel gaussiano promedia encuestas de ±4–7 meses (aunque el ajuste local lineal corrige parte del sesgo de retardo). Es la decisión con mayor impacto en las predicciones del modelo y no está justificada en el código.

---

## 7. `KernelDensityEstimator` (l. 996-1144)

Defaults: `bw_type='fixed'`, `bw='scott'`, `outliers='keep'`, `n_dev=3`. `build_input` apila `exog` (+ `endog` si 2-D) en `data` y llama a `process_outliers`. `build_pred`: si `p` es número, `linspace(min, max, p)` por dimensión; si array, lo recorta al rango.

`predict` (l. 1115-1144): `f̂(p) = Σ_i w_i K(p, x_i) / (n·h)` (1-D) o `/ n` (2-D, sobre `meshgrid`). Como `K` no está normalizado y `Σw = neff`, la integral vale `√(2π)·neff/n` en 1-D (2.49 y 2.23 en tests) y `2π·h₁h₂` en 2-D (0.58) (t2 K6, t6 M5). **No es una densidad**; solo proporcional. Con `bw_type='adaptive'` además se divide por el `h` fijo.

`process_outliers` (l. 1060-1089): usa `Stat(..., outliers=...)` y toma `x_samp.vind`; con `'remove'` `vind` ya tiene la longitud reducida (§2.3) ⇒ `IndexError` en cuanto se elimina algún outlier (t2 K7); con `'group'` ignora `x_samp.data` (los datos winsorizados) ⇒ **no tiene efecto**. Dos de los tres modos de outliers están rotos. `build_pred` con array y `kvar>1` tiene un error tipográfico (`data[:,0].max()` en l. 1052) y un `column_stack` de longitudes distintas.

---

## 8. Cómo lo usa el modelo (valores reales)

- `Forecaster.set_reg_params` (`forecaster.py:164-178`) y `Computer.set_reg_params` (`computer.py:231-245`): `gaussian / adaptive / 'isj' / n_iter=3 / min_delta=None / poly_deg=1 / hac (1 lag, bartlett)`; `alpha=0.05`.
- `Forecaster.fit` (`forecaster.py:348`): `LocalKernelEstimator(df[name], weights=df.weight).fit(px, alpha)`; `px` = índice diario hasta `max_fc` días tras la última encuesta; `err`, `cmin`, `cmax` se guardan en `fc_stat` y el Simulator usa `err` y `nobs` (`simulator.py:316-319, 530`).
- `Computer.get_error_estimator` / `get_seats_estimator` (`computer.py:1602, 1674`): `LeastSquaresEstimator(x=df[['pct','regional','weeks']], y=df['error'], weights=df['weight'])` con HAC por defecto (aquí el orden de filas no es temporal; la corrección HAC no tiene sentido y debería ser HC o cluster por evento/provincia).
- `Computer.compute_ratings` (`computer.py:717-737`): medias ponderadas de errores absolutos con `Stat(...).mean()`.
- `Simulator.totals` (`simulator.py:402`): medianas de escaños simulados con `Stat(x).median()`.
- Pesos reales (PP, 306 encuestas): `Σw = 720`, `neff = 248`, rango `[0, 5.26]`.

---

## 9. Pipeline (orden de operaciones dentro del módulo)

1. `Stat.__init__` → `set_params` (neff de Kish, pesos → neff) → `dropna` → `winsorize` (`stat.py:71-134`).
2. `Kernel.__init__`: normaliza pesos, resuelve `bw` por columna (`get_bw_fixed` → `get_bw_isj/scott/silverman`) (`stat.py:251-291, 377-445`).
3. `Kernel.get_bw_isj`: histograma → DCT → `isj_root(brentq)` sobre `isj_fixed_point` → `bw = √t*·range` o *fallback* (`stat.py:312-400`).
4. `Kernel.get_bw_adaptive`: iteración Abramson-like sobre puntos de predicción (`stat.py:447-490`).
5. `Kernel.get_weights`: matriz kernel `(n_p, n)` producto por dimensión (`stat.py:492-519`).
6. `Estimator.build_input` / `build_pred`: fechas → deltas, pesos → neff (`stat.py:550-631`).
7. `LeastSquaresEstimator.build_input` (diseño polinómico, dof, whitening) → `fit` (pinv, covb, scale, HAC) → `predict` (bandas t) (`stat.py:693-856`).
8. `LocalKernelEstimator.predict`: kernel × pesos, umbral 1e-2, WLS local por punto, reindexado diario (`stat.py:939-993`).
9. `KernelDensityEstimator`: outliers → kernel → suma ponderada / (n·h) (`stat.py:1023-1144`).

---

## 10. Hallazgos (ordenados por severidad)

| # | Sev. | Tipo | Dónde | Resumen |
|---|---|---|---|---|
| 1 | alta | bug | `stat.py:285-291` | `bw` como cadena ⇒ array `<U3` ⇒ el ancho de banda calculado se trunca a 3/5 caracteres (ISJ 71.62→71.0; Scott 109.602→109.6). Afecta a la configuración por defecto del Forecaster/Computer. |
| 2 | alta | bug | `stat.py:183-184, 203-204` | `conf/conf_mean(alpha≥1)`: `q=erf(n/√2)` ⇒ `conf(2)=1.69σ`. Latente (ningún call site con α≥1 en `mtpy/lib`; `dataviz.py:2156` usa α<1). |
| 3 | alta | bug | `stat.py:1060-1089` + `113-134` | KDE `outliers='remove'` lanza `IndexError`; `'group'` no hace nada. |
| 4 | alta | bug | `stat.py:131` | `winsorize('group')` manda los outliers bajos al máximo interior. Latente en `computer.py:564`. |
| 5 | media | stat/doc | `stat.py:833-838, 688`; `forecaster.py:331-334` | `err/cmin/cmax` son bandas de confianza de la media suavizada, no intervalos de predicción. Documentar y no llamarlos "prediction intervals". |
| 6 | media | stat | `stat.py:251-291, 402-422` | Selección de `h` por reglas de densidad (ISJ/Scott/Silverman sobre fechas) para una regresión; en datos reales ≈70 días. Sustituir por CV/GCV/AICc o justificar. |
| 7 | media | bug | `stat.py:335-338, 384-399` | ISJ con `n≤50` cae siempre al *fallback* `2π·spacing` sin aviso; *fallback* y umbral `min_t` no documentados. |
| 8 | media | stat | `stat.py:373-374, 398` | ISJ no invariante a la escala de pesos (histograma dividido por `n`, no por su suma) y `bw` usa `x_range` mientras `t*` está en unidades del rango del histograma. |
| 9 | media | stat | `stat.py:1135-1141, 294-306` | KDE no integra a 1 (falta `1/√(2π)`, `neff/n`, `2π h₁h₂` en 2-D). |
| 10 | media | bug | `stat.py:909-927, 736-738` | Ventana local con `neff≤2` ⇒ `ValueError` y aborta toda la predicción. Debería devolver NaN para ese punto. |
| 11 | media | bug | `stat.py:620-627, 886-897` | `predict()`/`r2_score` sin `p` fallan con índice temporal; `r2_score` sobrescribe `self.result`. |
| 12 | media | bug | `stat.py:781-790` | `LeastSquaresEstimator.r2_score` centra con `mean(√w·y)`: 0.445 vs 0.211 correcto. |
| 13 | media | stat | `stat.py:447-490` | Adaptativo: piloto sin `1/h` en iteraciones >1, gmean sobre puntos de predicción, `h=0/nan` si `pdf=0`, revierte al anterior al parar. |
| 14 | media | bug | `stat.py:161-172` | `quantile` con NaN y `dropna=False` indexa mal (mediana → NaN); guarda `i < neff−1` mezcla índice y neff. |
| 15 | media | diseño | `stat.py:32-66` | `@pstatic`: los posicionales van al constructor (`Stat.quantile(x, 0.5)` falla). |
| 16 | baja | bug | `stat.py:736` | `dof = neff − poly_deg − kvar` incorrecto para `poly_deg>1` y `kvar>1`. |
| 17 | baja | stat | `stat.py:753-779`; `computer.py:1602,1674` | HAC por defecto en datos sin orden temporal (estimadores de error/escaños) y con lags en índice de fila; sin HC/cluster. |
| 18 | baja | bug | `stat.py:374, 244` | `get_bw_isj(x)` sin pesos y `scale_weights` con enteros fallan por `/=`, `*=` in-place. |
| 19 | baja | diseño | `stat.py:120-123` | `winsorize(n_dev<1)` usa `ci(1−n_dev)`: `n_dev=0.05` ⇒ banda central del 5%. |
| 20 | baja | dead-code | `stat.py:300-304, 1049-1053` | Rama `k>1` de `kernel_gaussian` y rama array 2-D de `KDE.build_pred` no funcionan (typo `data[:,0].max()`). |
| 21 | baja | reproducibilidad | `stat.py:600-606`; `dates.py:801` | Floats asignados sobre `datetime64`; deltas fraccionarios truncados a la unidad (`0.5 días → 0`). |
| 22 | baja | rendimiento | `stat.py:944, 966-975` | ISJ+adaptativo recalculados en cada `predict`; cada WLS local predice en todos los puntos. |
| 23 | baja | docs | todo el fichero | Sin docstrings (AGENTS.md los exige); anotaciones erróneas (`dev -> Self`, `fit -> Self|Series|DataFrame`), nombres crípticos (`conf`, `dev`, `pstatic`, `perr/serr`). |
| 24 | baja | reproducibilidad | `requirements.txt` | Pines numpy 1.25.2/scipy 1.9.3 vs entorno 2.4.4/1.17; `scipy.fftpack` es legado (usar `scipy.fft`). |

---

## 11. Propuesta de suite de tests (verificación contra referencias, no dependencia)

`tests/test_stat.py` con pytest y `hypothesis` opcional:

- **Stat**: `mean/var/std_mean` vs `statsmodels.stats.weightstats.DescrStatsW(x, w·neff/Σw, ddof=1)`; sin pesos vs `np.mean/np.var(ddof=1)`. `quantile` sin pesos vs `np.quantile(method='averaged_inverted_cdf')`; con pesos enteros vs repetición de datos. `conf(n)` ⇒ `n·σ`; `ci_mean(0.05)` vs `DescrStatsW.tconfint_mean`. `winsorize` vs `scipy.stats.mstats.winsorize` / `np.clip`. Propiedades: invariancia a la escala de pesos, NaN con y sin `dropna`, tipos enteros.
- **Kernel**: `kernel_gaussian` vs `scipy.stats.norm.pdf·√(2π)h`; Scott/Silverman vs fórmulas cerradas y `scipy.stats.gaussian_kde.factor`; ISJ vs `KDEpy.bw_selection.improved_sheather_jones` (instalar solo en tests) para n∈{30, 100, 1000} y datos bimodales; invariancia de ISJ a `w` vs `c·w`; tamaño ≤50 no debe caer al *fallback* silenciosamente.
- **LeastSquaresEstimator**: `coef`, `covb`, `mcov` (no robusta y HAC) vs `sm.WLS(...).fit(cov_type='HAC', maxlags=L, use_correction=True)` ajustando la convención `neff`; `r2_score` vs `rsquared`; `perr` vs `get_prediction().se_mean`; `dof == n − exog.shape[1]` para varios `poly_deg/kvar`.
- **LocalKernelEstimator**: con `bw` grande y `poly_deg=1` debe coincidir con WLS global; con datos `y = a + b·x` exactos recupera la recta; vs `statsmodels.nonparametric.KernelReg(reg_type='ll')` con kernel gaussiano fijo; robustez: ventana vacía ⇒ NaN, no excepción; `predict()`/`r2_score` con índice de fechas; idempotencia de `result`.
- **KDE**: `∫f̂ = 1` (trapecio) en 1-D y 2-D; vs `scipy.stats.gaussian_kde` con el mismo `h` (`bw_method=h/σ`); `outliers` en los tres modos.
- **Regresión del modelo**: fijar semilla y datos sintéticos "tipo encuestas" (fechas duplicadas, pesos 0–5) y congelar las salidas de `Forecaster.fit` (snapshot) para detectar cambios numéricos al corregir los bugs 1, 7, 8.

---

## 12. Notas positivas de diseño

- La normalización de pesos a `neff` de Kish hace que media, varianza (`ddof=1`), `std_mean` y `dof` sean coherentes con el estimador insesgado para pesos de fiabilidad; es una elección elegante y verificada numéricamente.
- WLS por pseudoinversa (SVD) en vez de ecuaciones normales: estable ante colinealidad.
- El HAC (Bartlett) está bien implementado y coincide con statsmodels.
- El ISJ es un port fiel del algoritmo de Botev (misma parametrización que KDEpy), y el diseño "kernel × pesos de encuesta + regresión local lineal + t con dof local" es una formulación razonable y transparente de un promedio de encuestas tipo LOESS.
- La conversión automática de índices temporales a deltas y de vuelta permite usar `pd.Series` con fechas directamente.

---

## 13. Preguntas para el autor

1. ¿La intención de `bw='isj'` era realmente elegir el suavizado de la regresión con un selector de densidad sobre las fechas, o fue una aproximación provisional? ¿Se ha evaluado por validación cruzada la precisión de la predicción frente a `h`?
2. ¿Los intervalos `cmin/cmax` de `fc_stat` se presentan al público como "intervalo de la tendencia" o como "intervalo del resultado"? El Simulator suma `err²` + `pct_err²`: ¿es esa la descomposición prevista (incertidumbre del promedio + error sistemático histórico)?
3. ¿Por qué el *fallback* `2π·spacing` en ISJ y el umbral `min_t`? ¿Hay algún caso de uso documentado que lo motivara?
4. ¿Por qué el ancho adaptativo se calcula en los puntos de predicción (balloon) en lugar de por observación (Abramson)? ¿Se busca un efecto tipo "vecino más cercano" de LOESS?
5. ¿El HAC en los estimadores de error/escaños (`computer.py:1602, 1674`) es deliberado? Los datos no están ordenados en el tiempo; parece más adecuado HC o cluster por evento/provincia/encuestadora.
6. ¿Se quiere mantener la metaclase `@pstatic` en la versión divulgativa o conviene una API funcional explícita?
7. `winsorize(n_dev<1)`: ¿`n_dev=0.05` debía significar "recortar el 5% exterior" (`ci(0.05)`)?
8. ¿Qué versiones de numpy/scipy son las de referencia? Los pines de `requirements.txt` no coinciden con el entorno actual.
