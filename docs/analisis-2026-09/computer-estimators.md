# Computer: estimadores de sesgo, error y escaños

Módulo analizado: `mtpy/lib/computer.py` (clase `Computer`, 2625 líneas), métodos `fit_bias_estimator` (1461-1508), `get_error_estimator_data` (1510-1569), `get_error_estimator` (1571-1606), `get_seats_estimator_data` (1608-1650), `get_seats_estimator` (1652-1679), `get_path` (2611-2625), más las clases `LeastSquaresEstimator` (stat.py 661-857) y `LocalKernelEstimator` (stat.py 859-994) que los sustentan, y el uso que hace `Simulator` de `v2seats`/`v2err` (simulator.py 155-169, 311-330, 509-540, 627-629).

Todas las cifras numéricas que se citan (coeficientes, tamaños, residuos) se han obtenido ejecutando el código contra la base de datos local el 2026-09-23 con `Computer(scope='es', path='files').build_series()` (17 eventos 1977-2027, 3842 filas en `series`, 488 partidos, de ellos 309 regionales).

---

## 1. Propósito y encaje

Dentro de `Computer` (que calcula pesos, errores, desviaciones y ratings de encuestas a partir de elecciones pasadas) hay tres *estimadores* que convierten esa historia en funciones reutilizables:

| Estimador | Método | Modelo | Consumidor |
|---|---|---|---|
| Sesgo medio a lo largo del tiempo (por evento) | `fit_bias_estimator` | Regresión local lineal con kernel gaussiano (LOESS ponderado) sobre el log-ratio del "bias" de cada encuesta, evaluada en una rejilla diaria | `compute_deviations` (computer.py:810) para calcular `bias_dev_adj`/`bias_dev_err` de cada encuesta, que alimentan el rating de la encuestadora |
| Error absoluto esperado de una encuesta en función del % de voto | `get_error_estimator` | Mínimos cuadrados ponderados: `|e| = β0 + β1·pct + β2·regional + β3·weeks` | `Simulator.v2err` (simulator.py:164-169) para inflar la varianza del ruido Monte Carlo |
| Escaños esperados en función del % de voto nacional | `get_seats_estimator` | Mínimos cuadrados ponderados: `seats = γ0 + γ1·pct + γ2·regional` | `Simulator.v2seats` (simulator.py:156) cuando `split=False` (modos LS y MC) |

Los tres se apoyan en `self.series`, el DataFrame construido por `build_series` (401-475): concatenación de encuestas y resultados finales, índice `['event_date','date','pollster_id','sponsor_id']` (`self.keys`, línea 145), con columnas de metadatos (`data_cols`, 443-449: `days`, `featured`, `computed`, `weight_sample`, `weight_over`, `weight_rating`, `bias`, …) y una columna por partido (`self.names`). Las filas de resultados finales son las que tienen `pollster` nulo (propiedad `events`, 155-165); las encuestas, `pollster` no nulo (propiedad `polls`, 167-177). `self.errors` (351-366) es la misma estructura pero con el error signado `poll − resultado` por partido, en puntos porcentuales (lo calcula `compute_errors`, 637-774, y se guarda en `polls_results.error`).

### 1.1 Filtro común: `filter_polls` (250-306)

Todos los estimadores de encuestas parten de `filter_polls(featured=True, **kwargs)`:

- `drange` → `norm_range(drange, series.days.max())` (utils.py:99): `None` → `(0, dmax)`, entero `k` → `(k, dmax)`. Se filtra `days.between(lo, hi)` (línea 286).
- Se excluyen `mtype ∈ drop_mtypes` (por defecto `['aggr','online']`) y `ctype ∈ drop_ctypes` (por defecto `['wban','exit']`), y encuestas con `weight_over == 0` (282-284).
- `featured=True` exige `featured & computed` (289-290). En la práctica esto deja fuera los eventos 1977-1989 y 2027 (no tienen encuestas *featured*): el primer evento con datos es 1993-06-06.
- `n_last > 0` conserva las últimas `n_last` encuestas de cada `(event_date, pollster_id)` (292-293).

Ningún estimador incorpora `weight_rating` en sus pesos (sólo `weight_over × weight_sample`): decisión correcta, porque el rating se deriva precisamente de estas estimaciones y usarlo aquí sería circular.

---

## 2. `fit_bias_estimator(featured=True, **kwargs)` (1461-1508)

### 2.1 Qué hace

Para cada evento con encuestas filtradas, ajusta una curva suavizada del "bias" medio de las encuestas a lo largo del periodo pre-electoral y devuelve, día a día, la media estimada, su intervalo de confianza y su error estándar.

### 2.2 Datos

1. `df = filter_polls(featured, **kwargs)` (1475).
2. `df['lor'] = bias_to_lor(df['bias'])` (1477), con `bias_to_lor(x) = log(1 + x/100)` (1254-1273); inversa `lor_to_bias(x) = (e^x − 1)·100` (1233-1252). **Ojo**: la columna `bias` de `polls` no es un sesgo signado sino una magnitud: `compute_errors` la define (735-745) como media ponderada `0.7·bias_avg + 0.3·bias_blocks` (`error_weights`, línea 49), donde `bias_avg` es la media ponderada de `|error en log-ratio|` sobre los partidos del bmap `main` (727-730) y `bias_blocks` lo mismo sobre `vs`. Por tanto `lor` ≥ 0 siempre y lo que se suaviza es el *error absoluto medio en escala log-ratio* de cada encuesta, no un sesgo direccional.
3. `df['weight'] = round(weight_over × weight_sample, 2)` (1479).

### 2.3 Modelo (por evento `dt`, bucle 1487-1506)

- Serie de entrada: `d['lor']` indexada por `date` (se eliminan los niveles `pollster_id`, `sponsor_id`, 1490); puede haber varias encuestas el mismo día.
- Rejilla de predicción: `px = date_range(dt_start, dt, freq='D')` donde `dt_start` es la fecha de la **primera encuesta del evento en `self.polls` (sin filtrar)** (1492-1494).
- Estimador: `LocalKernelEstimator(d['lor'], weights=d['weight'], **self.reg_params).fit(px, alpha=self.alpha)` (1497-1503). `reg_params` por defecto (`set_reg_params`, 214-248): `kernel='gaussian'`, `bw_type='adaptive'`, `bw='isj'`, `bw_kwargs={'n_iter': 3, 'min_delta': None}`, `poly_deg=1`, `cov_type='hac'`, `cov_kwargs={'hac_lags': 1, 'kernel': 'bartlett'}`. `alpha=0.05` (línea 41).

Reconstrucción matemática de `LocalKernelEstimator.predict` (stat.py 939-994) con esos parámetros:

1. Las fechas se convierten en días desde la primera observación (`ts_to_delta`, stat.py 583-588), `t_i`.
2. Los pesos de las encuestas se reescalan al tamaño muestral efectivo de Kish (stat.py 572-575):
   `n_eff = (Σ w_i)² / Σ w_i²`, `w_i ← w_i · n_eff / Σ w_i`.
3. Ancho de banda piloto `h0` por el método *Improved Sheather-Jones* sobre `{t_i}` ponderado (stat.py 378-401, invocado en 285-287).
4. Ancho de banda adaptativo tipo Abramson pero evaluado en los puntos de predicción (*balloon estimator*), `get_bw_adaptive` (stat.py 447-490), 3 iteraciones:
   `f̂(p) = Σ_i w_i · exp(−(p − t_i)²/(2h²))`, `g = media geométrica de f̂`, `h(p) ← h(p)·sqrt(g / f̂(p))`.
   Es decir, el ancho de banda es mayor donde hay pocas encuestas y menor donde hay muchas.
5. Pesos kernel (no normalizados) `K_p(i) = exp(−(p − t_i)² / (2 h(p)²))` (stat.py 294-307).
6. En cada día `p` (stat.py 909-927): pesos locales `ω_i = K_p(i)·w_i`, se descartan los puntos con `ω_i < 0.01` y se ajusta un `LeastSquaresEstimator` de grado 1 (recta local) con esos pesos, esto es, WLS con `β̂ = (X'ΩX)^{-1} X'Ω y` vía pseudo-inversa de la matriz "blanqueada" `sqrt(ω)·X` (stat.py 795-798). La predicción en `p` es `ŷ(p) = β̂0 + β̂1·p`.
7. Con `alpha` se devuelven además `cmin`, `cmax`, `err` (= error estándar de la media local, `sqrt(x_p' M x_p)` con `M` la covarianza HAC Newey-West de 1 retardo y kernel de Bartlett, stat.py 753-780, 826-833; cuantil `t_{1−α/2, dof}` con `dof = n_eff − 2`), y `nobs`, `neff` locales.

Salida: `dreg` con `MultiIndex(event_date, date)` y columnas `['mean','cmin','cmax','err','nobs','neff']` en unidades log-ratio. Ejemplo real para 2016-06-26: 168 días, 0 NaN, `nobs` locales entre 19 y 45, `neff` entre 12.7 y 24.5, media final 0.1088 (= 11.5 % en unidades "bias"), `err` ≈ 0.012.

### 2.4 Uso posterior (`compute_deviations`, 776-851)

`bias_dev_i = lor_i − mean(date_i)` (811-816), y luego `bayes_adjust(df[['bias_dev','bias_err']], params={'mean': 0, 'std': bias_dev_tau=0.01})` (821-827) produce `bias_dev_adj` y `bias_dev_err`, que se devuelven en unidades "bias" (`lor_to_bias`, 832). En resumen: el estimador define "lo normal" para una encuesta publicada ese día, y cada encuesta se puntúa por cuánto se aleja de lo normal, encogiendo hacia 0 con un prior gaussiano cuya fuerza depende del error estándar de la curva (más encuestas ese día → `err` menor → menos encogimiento).

### 2.5 Notas de diseño

- El docstring de `compute_deviations` (782-783) habla de "polls with similar sample and time to the event", pero el kernel sólo actúa sobre el tiempo; el tamaño de muestra sólo entra como peso.
- `px` empieza en la primera encuesta *sin filtrar* (1492), mientras que el ajuste usa las filtradas; en 2008 (28 días) y 2023 (15 días) la curva se extrapola al inicio. Es inocuo (el kernel gaussiano sigue dando peso) pero es una inconsistencia.
- `plot_bias_weights` (2370-2515) y `print_bias_weights` (2517-2609) reconstruyen a mano el mismo preprocesado (`lor`, `weight`, `LocalKernelEstimator(**reg_params)`) en vez de reutilizar `fit_bias_estimator`, y además usan `drop_ctypes=None` (2394), así que el gráfico puede no corresponder exactamente a lo que se ajustó.

---

## 3. `get_error_estimator_data(featured=True, **kwargs)` (1510-1569)

Construye la tabla larga (una fila por encuesta × partido) del error absoluto cometido.

1. `polls = filter_polls(...)`; `polls['weight'] = round(weight_over × weight_sample, 2)` (1535); `errors = self.errors.loc[polls.index]` (1536).
2. `melt` de los % (`d_pct`, columnas `weight`, `party`, `pct`) y de los errores (`d_err`, `party`, `error`) sobre `names = [n for n in self.names if n in polls.columns]` (1538-1551); se concatenan por índice.
3. Filtro `(pct > 0) & (|error| > 0)` (1554); se une `days` desde `self.polls` (1555-1557).
4. `error ← |error|` (1559); `pollster` (nombre), `regional` (0/1, desde `parties.regional`), `color`; `weeks = (days + 1) // 7` (1563). Nota: en `poll_rating_weights` (1219) las semanas se definen distinto, restando el periodo de veda (`event_dlimits`); aquí no.
5. Devuelve `['date','pollster','party','regional','weeks','weight','color','pct','error']` **ordenado por `pct`** (1565-1567).

Con los filtros por defecto (`drange=None`, `n_last=None`): 14 716 filas, 43 % regionales, `pct ∈ [0.1, 49.3]`, `error ∈ [0.01, 21.1]`, media 2.46; `weeks ∈ [1, 209]`, mediana 52 (la mitad de las filas son encuestas a más de un año). Con lo que pasa el `Simulator` (`drange=(0,None)`, `n_last=1`): 1 634 filas, mediana de `weeks` = 1.

## 4. `get_error_estimator(featured=True, **kwargs)` (1571-1606)

```
LeastSquaresEstimator(x=df[['pct','regional','weeks']], y=df['error'], weights=df['weight']).fit()
```

Modelo: `|e| = β0 + β1·pct + β2·regional + β3·weeks + ε`, WLS con pesos Kish-normalizados (stat.py 705-711), `poly_deg=1`, `cov_type='hac'` por defecto (stat.py 669). Coeficientes reales:

| Datos | β0 | β1 (pct) | β2 (regional) | β3 (weeks) | R² | n |
|---|---|---|---|---|---|---|
| por defecto | 2.177 | 0.0437 | −2.661 | 0.0127 | 0.383 | 14 716 |
| como Simulator (`n_last=1`) | 1.259 | 0.0432 | −1.194 | 0.0131 | 0.394 | 1 634 |

Lectura: el error absoluto crece ~0.44 puntos por cada 10 puntos de voto, es 2.7 puntos menor para regionales (que tienen % nacionales pequeños) y crece 0.66 puntos por año de antelación. Problemas empíricos:

- Al ser lineal sobre una variable no negativa y heterocedástica, predice **valores negativos**: 2 565 de 14 716 filas in-sample (17 %); p. ej. `(pct=0.1, regional=1, weeks=1) → −0.47`, `(2, 1, 1) → −0.38`.
- La desviación típica del residuo crece con `pct` (1.07 en (0,1] → 3.99 en (30,50]), luego el error estándar único `serr = 2.46` no describe la incertidumbre en ninguna zona.
- El `R²` es 0.38: la mayor parte de la variabilidad del error de una encuesta no se explica por estas tres variables.

El docstring (1586, 1593) menciona los parámetros `computed` y `drop_contexts`, que no existen (son `featured` y `drop_ctypes`).

### 4.1 Uso en `Simulator`

- Se construye una sola vez con `drange=self.drange` y `n_last=self.n_last` (simulator.py 164-167; `n_last=1` por defecto, línea 35). 
- `weeks = (self.model.drange[0] + 1) // 7` (simulator.py 311 y 509): con el `drange=6` habitual → `weeks=1`, **independientemente de `limit_date`**. Es decir, aunque se simule con encuestas de hoy para unas elecciones dentro de 11 meses, la incertidumbre añadida es la de "una semana antes".
- `build_forecast` (325-330) guarda `fc['error'] = v2err.predict([mean, regional, weeks])`.
- `build_frame` (521-537), si `random=True`: `pct_err = v2err.predict([pct, regional, weeks])`; `std_err = sqrt(err² + pct_err²)`; `rand = clip(pct + std_err · t_{nobs−2}, 0)`. Es decir: se suma en cuadratura el error estándar de la regresión local del Forecaster (`err`) y el error absoluto medio *de una encuesta individual* predicho por este estimador, y se usa como escala de una t de Student. Dos objeciones: (i) `E|X| = σ·sqrt(2/π)` para una normal, así que usar el MAE como σ subestima la desviación típica ~20 %; (ii) la cantidad modelada es el error de una encuesta suelta, no el error del promedio ponderado: conceptualmente se debería calibrar contra el error histórico del propio Forecaster. El signo negativo de una predicción se pierde al elevar al cuadrado (una predicción de −0.47 aporta +0.47² de varianza).

---

## 5. `get_seats_estimator_data()` (1608-1650)

1. Para `metric ∈ ['pct','seats']`, `load_events(metric)` (308-329) trae los resultados nacionales (`region_id = 0`) por partido; en `seats` se elimina la columna `seats` del evento (total de la cámara, 350) para que no colisione con el `melt` (1621-1622).
2. `melt` por partido y concatenación → filas `(date, party)` con `pct` y `seats`.
3. Filtro **`(pct ≥ 0.1) & (seats > 0)`** (1635): de 349 combinaciones evento-partido con `pct ≥ 0.1`, se descartan 162 (46 %) por tener 0 escaños (84 nacionales, 78 regionales). Ejemplos excluidos: CDS 1993 1.76 %, PACMA 2019A 1.25 %, MUC 1986 1.14 %; regionales con hasta 0.7 % y 0 escaños.
4. `regional`, `color`, `year`, `years = year_max − year` (año máximo 2023; 2027 cae por NaN), `weight = round(0.97^years, 2)` (1642; de 1.00 en 2023 a 0.25 en 1977), `ratio = seats/pct`, `pos` = ranking dentro del evento (1643-1644; sólo los usa el notebook `lab/EventsSeats`).
5. Devuelve 187 filas ordenadas por `pct`: 68 nacionales (`pct` 1.19-48.11) y 119 regionales (`pct` 0.16-5.04).

Nótese que `0.97` está codificado a mano y difiere del atributo `year_decay=0.9` (47, 125, 1225) que usa el resto de la clase.

## 6. `get_seats_estimator(**kwargs)` (1652-1679)

```
LeastSquaresEstimator(x=df[['pct','regional']], y=df['seats'], weights=df['weight'], **kwargs).fit()
```

Modelo: `seats = γ0 + γ1·pct + γ2·regional + ε`. Coeficientes reales: `γ0 = −16.95`, `γ1 = 4.509`, `γ2 = +15.71`; `R² = 0.991` (inflado por los grandes partidos), `serr = 4.72`, `n_eff = 157`.

Implicaciones:

- Un partido nacional obtiene 0 escaños hasta `pct = 3.76 %` (raíz de la recta) y después 4.5 escaños por punto. Un regional tiene intercepto `−1.24`, así que se le asignan escaños desde `0.28 %` con la **misma pendiente** de 4.5 escaños/punto.
- Predicciones para nacionales: 10 % → 28.1; 20 % → 73.2; 33 % → 131.9; 45 % → 186.0. Para regionales: 1 % → 3.3; 3 % → 12.3.
- Residuos medios por tramo de `pct`: (1,3] +1.1; (3,10] −2.2; (10,20] −4.5; (20,30] +4.2; (30,50] +0.9. Es la convexidad de D'Hondt en 52 circunscripciones (los partidos medianos se llevan menos y los grandes más), que una recta no puede reproducir. Desviación típica del residuo: 8.1 escaños en nacionales, 1.3 en regionales.
- No hay restricción de suma: aplicado a los % reales de cada elección, la suma de escaños predichos oscila entre 316 (1977) y 376 (1996); para 2023 da 359.7 en lugar de 350.
- Validación *leave-one-event-out* (15 grupos): RMSE 4.94 / MAE 2.80 con grado 1; 4.92 / 2.67 con `poly_deg=2` (la interacción cuadrática apenas ayuda; el problema está en la forma funcional y en el truncamiento, no en el grado).

### 6.1 Uso en `Simulator`

- `v2seats = computer.get_seats_estimator()` (156), con el `Computer` construido sobre los eventos anteriores al simulado (`get_event_dates(..., date_to=event_date, skip=1)`, 130-135).
- `simulate` (627-629): si `split=False`, `result[scope] = v2seats.predict([vpred, regional]).clip(0)`; el resultado se redondea a entero en `run` (662). No se renormaliza a 350 por simulación.
- `totals` (397-436) toma la **mediana** por partido de las simulaciones, la trunca y reparte la diferencia hasta `n_seats` por el método de restos mayores (sumando o restando cíclicamente). Es decir, la restricción de 350 sólo se impone sobre el resumen, no sobre cada simulación, y las distribuciones (`dist`) siguen sin restricción.

### 6.2 Valoración: ¿sustituto razonable del modelo provincial?

Como aproximación rápida para los modos LS/MC es útil (y el ajuste in-sample es aceptable para partidos de 15-45 %), pero el supuesto implícito es fuerte: **la relación % nacional → escaños es afín, invariante en 46 años de sistema de partidos, independiente de cuántos y qué partidos compiten, con la misma pendiente para nacionales y regionales, y sin restricción de suma**. Los riesgos de extrapolación concretos son: (a) cualquier escenario con fragmentación distinta a la histórica (cuatro partidos nacionales entre 10 y 25 %); (b) partidos regionales con concentración territorial diferente (JxCat 1.6 % → 7 escaños, EHB 1.2 % → 6, PACMA 1.2 % → 0: la regresión sólo ve los dos primeros); (c) partidos por encima de 48 % (nunca observados); (d) partidos nacionales entre 1 y 4 %, donde la recta predice negativo y el clip fuerza 0 aunque históricamente IU 3.7 % obtuvo 2. El modo `split=True` (reparto provincial + D'Hondt) es el que respeta la mecánica electoral; convendría documentar los modos LS/MC como *baseline* explícitamente aproximado o sustituir la recta por un modelo monótono con restricción de suma (p. ej. isotónico por tipo de partido, o D'Hondt sobre una distribución provincial media).

---

## 7. `get_path(name=None)` (2611-2625)

Devuelve `'{path}/{name}'` si `name` es verdadero; si no, **devuelve `None`** implícitamente, aunque el docstring dice "the path to the current polls will be returned" y la anotación es `-> str`. `self.path` por defecto es `os.getcwd()` (131), y de ahí se lee `params.json` (`get_event_params`, data.py 67): desde la raíz del repo `Computer(scope='es')` lanza `FileNotFoundError` porque el fichero está en `files/` (directorio ignorado por git).

---

## 8. Métodos de visualización e impresión (1681-2610)

930 líneas de las 2625 (35 %) de la clase son gráficos/tablas, y el módulo importa `matplotlib`, `adjustText` y una docena de helpers de `dataviz` a nivel de módulo (1-19):

| Método | Líneas | Qué dibuja |
|---|---|---|
| `plot_deviations` | 1681-1824 | Diana 2D del error de cada encuesta sobre los dos bloques (`bmap`), con `adjust_text` |
| `plot_errors` | 1826-1976 | Errores de cada encuesta por bloques frente al resultado final (permite `drange`, `n_last`) |
| `print_rating_weights` | 1978-2041 | Tabla estilizada con los pesos (`weight_over`, `weight_sample`, `weight_pos`, `weight_week`, `weight_year`) de cada encuesta |
| `print_ratings` | 2043-2107 | Tabla de componentes del rating por encuestadora |
| `plot_ratings` | 2109-2189 | Barras ordenadas de `rating` / `bias_dev_adj` / `rating_adj` / `weight_rating` |
| `plot_ratings_grid` | 2191-2294 | Rejilla 2×2 de las métricas anteriores |
| `plot_ratings_prior` | 2296-2368 | Prior bayesiano del rating frente a nº de encuestas |
| `plot_bias_weights` | 2370-2515 | Pesos kernel×encuesta que afectan a la estimación de sesgo en una fecha, con la curva local |
| `print_bias_weights` | 2517-2609 | Tabla equivalente a la anterior |

Ninguno de ellos es necesario para calcular pesos, errores, ratings ni estimadores; separar `Computer` en un núcleo numérico (`computer.py`) y un módulo de presentación (`computer_plots.py` o funciones libres que reciban el `Computer`) reduciría el fichero a ~1 700 líneas, eliminaría dependencias gráficas del núcleo y facilitaría tests.

---

## 9. Pipeline

| Paso | Dónde | Qué | Matemática / parámetros |
|---|---|---|---|
| 1 | computer.py:401-475 | `build_series`: carga eventos, encuestas, errores, biases, ratings, partidos, encuestadoras; concatena en `series` | índice `keys`; `errors` alineado a `names` |
| 2 | computer.py:250-306 | `filter_polls`: `featured & computed`, `drange`, `n_last`, `drop_mtypes/ctypes`, `weight_over>0` | defaults: `drange=None→(0,dmax)`, `n_last=None`, `['aggr','online']`, `['wban','exit']` |
| 3 | computer.py:1461-1508 | `fit_bias_estimator`: por evento, LOESS lineal ponderado de `lor=log(1+bias/100)` sobre rejilla diaria | pesos `w_over·w_sample`; kernel gaussiano; bw ISJ + adaptativo (3 iter); HAC lag 1; `alpha=0.05` |
| 4 | computer.py:776-851 | `compute_deviations`: `bias_dev = lor − mean(t)`; `bayes_adjust` con prior N(0, 0.01) | salida en unidades bias |
| 5 | computer.py:1510-1569 | `get_error_estimator_data`: tabla larga (encuesta×partido) con `|error|`, `pct`, `regional`, `weeks=(days+1)//7`, `weight` | filtros `pct>0`, `|error|>0`; ordenada por `pct` |
| 6 | computer.py:1571-1606 | `get_error_estimator`: WLS `|e| ~ 1 + pct + regional + weeks` | β = (2.18, 0.044, −2.66, 0.013) con defaults; R² 0.38 |
| 7 | computer.py:1608-1650 | `get_seats_estimator_data`: resultados nacionales por evento-partido con `pct≥0.1 & seats>0`, `weight=0.97^years` | 187 filas, 68 nacionales / 119 regionales |
| 8 | computer.py:1652-1679 | `get_seats_estimator`: WLS `seats ~ 1 + pct + regional` | γ = (−16.95, 4.51, 15.71); R² 0.99; serr 4.7 |
| 9 | simulator.py:155-169 | `Simulator.__init__` instancia `v2seats` y `v2err` (`drange`, `n_last=1`) | |
| 10 | simulator.py:311-330, 509-537 | `build_forecast`/`build_frame`: `error = v2err.predict([pct, regional, weeks])`, `std_err = sqrt(err²+pct_err²)`, ruido `t_{nobs−2}` | `weeks = (drange[0]+1)//7` |
| 11 | simulator.py:627-629, 397-436 | `simulate` (split=False): `seats = clip(v2seats.predict([vpred, regional]), 0)`; `totals` ajusta la mediana a 350 por restos | |

---

## 10. Hallazgos

Ver la sección `findings` del resultado estructurado; se resumen aquí con severidad:

1. **[stat, media]** Truncamiento `seats > 0` (1635) elimina el 46 % de las observaciones y sesga al alza el intercepto, sobre todo en regionales.
2. **[stat, media]** Forma lineal sin restricción de suma para escaños: residuos sistemáticos por tramo, totales 316-376, 2023 → 359.7.
3. **[stat, media]** Regresión lineal sobre error absoluto: 17 % de predicciones negativas, heterocedasticidad fuerte, R² 0.38.
4. **[stat, media]** Simulator usa el MAE predicho como σ (subestima ~20 %) y mezcla error de encuesta individual con error del promedio.
5. **[design, media]** `weeks` en Simulator no depende de `limit_date` (siempre `(drange[0]+1)//7`).
6. **[design, media]** 35 % del fichero es visualización; dependencias gráficas en el núcleo.
7. **[reproducibility, media]** `path` por defecto = cwd y `params.json` en `files/` (ignorado por git): `Computer(scope='es')` falla desde la raíz.
8. **[design, baja]** `0.97` codificado (1642) frente a `year_decay=0.9`; el estimador de error no pondera por antigüedad.
9. **[stat, baja]** HAC Newey-West sobre datos de sección cruzada ordenados por `pct` (1567, 1646; stat.py 669).
10. **[bug, baja]** `dof = neff − poly_deg − kvar` (stat.py 722) incorrecto para `poly_deg > 1`.
11. **[docs, baja]** Docstrings con parámetros inexistentes (`computed`, `drop_contexts`, 1586/1593/1710/1725); `get_path` devuelve `None` en contra del docstring; `compute_deviations` habla de "similar sample size".
12. **[dead-code, baja]** `self.seats_estimator/error_estimator/bias_estimator` (151-153) nunca se asignan.
13. **[design, baja]** Lógica duplicada en `plot_bias_weights`/`print_bias_weights` frente a `fit_bias_estimator`; inicio de rejilla `px` desde encuestas sin filtrar (1492).
14. **[docs, baja]** `bias` es un error absoluto medio (no signado); el nombre induce a error en la documentación del estimador.
15. **[stat, baja]** Filtro `|error| > 0` (1554) descarta errores exactamente nulos (observaciones válidas).

## 11. Preguntas abiertas para el autor

- ¿El estimador de escaños se concibe como *baseline* pedagógico para LS/MC o como alternativa seria al reparto provincial? Determina si merece la pena sustituirlo por un modelo monótono con suma fija.
- ¿Por qué 0.97 de decaimiento anual en escaños y ninguno en errores, mientras `year_decay=0.9` gobierna los ratings?
- ¿`weeks=1` fijo en el Simulator es deliberado (simular "la última semana") o debería usar la distancia `limit_date → event_date`?
- ¿Se ha calibrado alguna vez el error del Forecaster (promedio) por evento, para usarlo en lugar del error de encuesta individual?
- ¿Qué criterio define `featured` en `events`/`polls`? Sólo hay datos desde 1993 y la documentación debería explicarlo.
- ¿`n_last=1` en el Simulator (una encuesta por encuestadora y evento para el estimador de error) es intencionado? Reduce la muestra de 14 716 a 1 634 filas.
- ¿Es intencionado que `bias` (y por tanto la curva de `fit_bias_estimator`) sea una magnitud absoluta y no un sesgo direccional?
