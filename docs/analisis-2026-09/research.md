# Dossier de referencia: métodos de agregación de encuestas, simulación electoral y estándares de repositorio científico

Documento de trabajo para comparar el modelo `elections-model` (paquete `mtpy`, módulos `Computer`, `Forecaster`, `Simulator` y librería estadística propia `mtpy/core/utils/stat.py`) con (1) los métodos de FiveThirtyEight/Nate Silver, (2) la literatura académica de agregación de encuestas y predicción electoral, (3) los modelos españoles públicos y (4) los estándares de repositorios científicos y divulgativos.

Convenciones: las rutas son absolutas dentro de `/Users/luiss/HD/Proyectos/Code/elections-model`; cada afirmación sobre el código cita `archivo:línea`. Cuando algo del código se aparta de lo que dice su docstring, se marca como **[DESAJUSTE]**. Los datos de la base de datos se leyeron en modo solo lectura el 2026-09-23 (95 encuestadoras, 17 eventos 1977-2027, 1.012 filas en `pollsters_ratings`).

---

## 0. Resumen de lo que el código hace realmente (base para la comparación)

Antes de comparar conviene fijar, con precisión y citando líneas, qué calcula cada componente. Esto es lo que un lector externo debería poder leer en un `METHODS.md` del futuro repositorio.

### 0.1 Computer (`mtpy/lib/computer.py`)

Parámetros por defecto (`mtpy/lib/computer.py:32-55`): `drop_mtypes=['aggr','online']`, `drop_ctypes=['wban','exit']`, `alpha=.05`, `ol_dev=3`, `ol_max=5000`, `wspan=4`, `pos_decay=.5`, `week_decay=.7`, `year_decay=.9`, `bias_dev_tau=.01`, `min_polls=3`, `ratings_margin=.05`, `error_weights={'avg':.7,'blocks':.3}`. Regresión local por defecto (`mtpy/lib/computer.py:214-248`): kernel gaussiano, ancho de banda adaptativo con semilla ISJ, 3 iteraciones, polinomio local de grado 1, covarianza HAC (Newey-West, 1 retardo, kernel Bartlett).

**Pesos por encuesta** (`compute_weights`, `mtpy/lib/computer.py:476-635`):
- `proc_sample`: imputación de tamaño muestral por mediana (encuestadora+evento → encuestadora → todas) y winsorización con `Stat(..., outliers='group', n_dev=3)` tras un `clip(0, ol_max=5000)` (`:543-568`). Nota: `Stat.winsorize(action='group')` sustituye los valores fuera de ±3σ por el máximo de los valores dentro del rango (`mtpy/core/utils/stat.py:119-135`), es decir, también los valores *por debajo* de −3σ se elevan al máximo; en la práctica no ocurre con muestras positivas, pero la semántica es asimétrica.
- `weight_sample = sqrt(proc_sample / mediana_del_evento)` (`:572-576`). Idéntico en forma al peso por tamaño muestral de 538 (ver §1).
- `weight_over` (`poll_overweights`, `:1275-1322`): descarta (peso 0) una encuesta cuyo campo se solapa con una posterior de la misma encuestadora/tipo (trackings), y asigna `1/(N+1)` cuando hay N encuestas de la misma casa en una ventana de ±`wspan=4` días. Análogo al "rapid-fire pollster adjustment" de 538, con ventana de 4 días en vez de 14.
- `weight_rating`: se toma de `self.ratings` (`:602-608`), calculado en `compute_ratings`.

**Errores** (`compute_errors`, `:637-773`): para cada encuesta y partido se calculan el error en puntos (`pct`), el *log odds ratio* entre encuesta y resultado (`lor`, `:1077`) y el error en el margen entre los dos bloques del mapa `vs` (`gap`). Los agregados por encuesta son:
- `error_avg`: media ponderada por el resultado real de cada partido del |error pct| sobre el mapa `main` (`:718-722`).
- `error_blocks`: error (con signo) en el margen entre bloques `vs` (`:723`).
- `bias_avg`, `bias_blocks`: medias ponderadas del **valor absoluto** del LOR sobre `main` y sobre `vs` (`:724-731`).
- `bias = 0.7·bias_avg + 0.3·bias_blocks` (`:733-734`, con `error_weights`).

**[DESAJUSTE terminológico]** La columna `bias` no es un sesgo direccional (house effect) sino un error absoluto en escala log-odds. Los LOR *con signo* por partido sí se guardan en `PollsResults.bias` (`:693-700`) pero **no se usan en ningún ajuste posterior** de las encuestas (no hay corrección de house effects; ver §1.3).

**Desviaciones** (`compute_deviations`, `:776-851`): se ajusta, para cada evento, una regresión kernel local del `bias` (en LOR) de todas las encuestas contra la fecha, ponderando por `weight_over·weight_sample` (`fit_bias_estimator`, `:1461-1508`). `bias_dev = bias − bias_mean(t)` es la desviación de cada encuesta respecto al error medio de sus contemporáneas; se contrae hacia 0 con un prior normal N(0, τ=0.01) por precisión (`bayes_adjust`, `:1324-1388`). Esto es el equivalente estructural del "Advanced Plus-Minus" de 538 (error relativo a las demás encuestas de la misma carrera).

**Ratings** (`compute_ratings` `:853-1006` y `pollster_ratings` `:1094-1190`): para cada evento *i* se usan solo encuestas de eventos ≤ *i−1* (`:920-925`), de modo que el rating aplicado a un evento es *out-of-sample* respecto a él (buena práctica, coherente con la lógica "predictive" de 538). Dentro de `pollster_ratings`:
- Peso por encuesta `weight = weight_over·weight_sample·weight_pos·weight_week·weight_year` (`:1116-1118`), con `weight_pos = 0.5^k` (k = posición desde la última encuesta de la casa en ese evento), `weight_week = 0.7^semanas` respecto a la encuesta más cercana al evento y `weight_year = 0.9^años` respecto al evento más reciente (`poll_rating_weights`, `:1192-1231`). Se eliminan pesos < 0.01 (`:1121`).
- `bias_dev_adj` = media ponderada de las desviaciones ajustadas; `bias_dev_err` = error estándar ponderado con corrección `(n_w+1)/n_w` (`:1146-1155`); nueva contracción bayesiana hacia 0 con τ (`:1163-1169`).
- `rating_adj = Φ(−bias_dev_adj / τ)` (`:1172`): probabilidad normal de que la casa sea mejor que la media; si no hay datos, se rellena con `quality`.
- `rating = (n_w·rating_adj + min_polls·quality) / (n_w + min_polls)` (`:1177-1181`): mezcla con un prior `quality` (0-1) con pseudo-conteo `min_polls=3`. Es la misma forma funcional que el "Predictive Plus-Minus" de 538 (`n/(n+n_shrink)`).
- `weight_rating = log1p(rating)/log1p(mean(rating))` (`:1185`). En la BD el rango observado es 0-3.08 (media 0.92).

**[DESAJUSTE docstring/código]** El docstring de `compute_ratings` (`:853-905`) describe fórmulas (`error_adjusted = (num_related·error_related + min_related·error_expected)/(...)`, `prior_weight = quality·num_polls/(num_polls+mean)`, "rating escalado a [5-95]") que **no coinciden** con lo implementado: el código usa una regresión kernel del error medio, contracción normal-normal y la mezcla anterior; `ratings_margin` se asigna (`:128`) pero **no se usa** en el cálculo. La referencia del docstring (`:861`, `fivethirtyeight.com/methodology/how-our-pollster-ratings-work/`) hoy redirige a `abcnews.com/politics` (enlace muerto).

**Origen de `quality`**: columna `pollsters.quality` en la BD (0-78, mediana 5, 95 casas). No hay código que la calcule ni documento que la justifique; en `files/` existe un `Electomania-Rankings.xlsx` (ignorado por git) que podría ser su origen, pero no hay referencia a él en `mtpy/` ni en los notebooks. Es el equivalente del "transparency score / AAPOR-Roper prior" de 538, pero **sin criterio documentado**. Las casas sin encuestas evaluadas reciben `rating == quality` (por ejemplo `pollster_id=54`, rating 78 con `num_polls=0`).

**Estimadores auxiliares para el simulador**: `get_error_estimator` (`:1571-1606`) ajusta una regresión lineal ponderada (WLS) `|error| ~ pct + regional + weeks`; `get_seats_estimator` (`:1652-1680`) ajusta `seats ~ pct + regional` con pesos `0.97^años` sobre datos históricos filtrados por `pct >= 0.1 & seats > 0` (`:1636`). **[SESGO DE SELECCIÓN]** al excluir los partidos con 0 escaños, el estimador nacional de escaños aprende solo de casos con representación.

### 0.2 Forecaster (`mtpy/lib/forecaster.py`)

- `build_series` (`:221-311`): concatena encuestas y resultado anterior; `weight = weight_over·weight_sample·weight_rating` (`:268`). **[BUG LATENTE]** `:267` rellena `weight_rating` ausente con `pollsters.quality`, que está en escala 0-100 mientras `weight_rating` está en escala ~0-3: una casa nueva sin fila de rating tendría un peso decenas de veces mayor. Hoy no se dispara porque `pollsters_ratings` tiene 92 filas para el evento 2027-08-22, pero es una trampa para reproducciones futuras.
- `fit` (`:313-360`): para cada partido/bloque, `LocalKernelEstimator(serie, weights).fit(px, alpha)` sobre un eje diario desde la primera encuesta hasta `última encuesta + max_fc` días. La extrapolación más allá de la última encuesta es la del polinomio local de grado 1 (tendencia lineal local) sin ninguna reversión a la media.
- `fit_forecast` (`:362-403`): ajusta cada partido **por separado**; el bloque `'-'` (otros) se calcula como `100 − Σ` (`:392`), por lo que el vector no está restringido a ser una composición válida (la suma de los ajustes puede exceder 100 y `'-'` ser negativo).
- Intervalo: `LocalKernelEstimator.predict` devuelve `mean, cmin, cmax, err, nobs, neff` (`mtpy/core/utils/stat.py:947-951`), donde `err` es el error estándar de la predicción de la WLS local con covarianza HAC (`stat.py:753-780, 836-837`) y `nobs` es el número de encuestas con peso kernel ≥ 0.01 (`stat.py:914`). Es un intervalo de la *media local* (incertidumbre de estimación), no un intervalo de *predicción* del resultado electoral.

### 0.3 Simulator (`mtpy/lib/simulator.py`)

- Fecha de corte `limit_date = event_date − drange[0]` días (`:116-121`). Construye un `Computer` con los eventos anteriores (`:139-152`) para obtener `v2seats` (`:163-166`) y `v2err` (`:171-177`).
- `build_forecast` (`:310-336`): toma `mean`, `err`, `nobs` del Forecaster en `limit_date` y `error = v2err.predict([mean, regional, weeks])`.
- `build_frame` (`:496-545`): `std_err = sqrt(err² + pct_err²)` (`:526`) y `rand = pct + std_err · t_{nobs−2}` (`:531-534`), **independiente por partido** (no hay matriz de correlación ni restricción de suma; no hay término de deriva temporal explícito, solo la covariable `weeks` en el estimador de error). Colas pesadas vía t con `nobs−2` grados de libertad, que para ventanas con muchas encuestas es casi normal.
- `build_umat` (`:547-596`): proyección provincial por **swing proporcional**: `fmul = votos_previstos / votos_anteriores` nacional por partido (`:587`) y `vpred_pct_provincia = prev_pct_provincia · fmul` (`:588`). Reglas `smap` (agg/sub/split) para cambios de oferta partidista (`:551-580`).
- `simulate` (`:598-631`): en cada provincia renormaliza a 100 (`:613`), convierte a votos con el censo de votos de la elección anterior y aplica `alloc_dhondt` (`:482-494`). **[DESAJUSTE legal]** no se aplica la barrera del 3 % de votos válidos por circunscripción (LOREG art. 163.1.a): un `grep` de `0.03`/`threshold`/`umbral` en `mtpy/lib/*.py` no devuelve nada. En la práctica solo importa en Madrid y Barcelona (las únicas circunscripciones donde el 3 % es inferior al umbral efectivo de D'Hondt), pero es una desviación del sistema electoral real.
- `totals` (`:397-442`): devuelve la **mediana por partido** de la distribución de escaños y luego reparte la diferencia hasta 350 por restos. Es un resumen razonable para titulares, pero no es el resultado de ninguna simulación concreta y mezcla medianas marginales; debería documentarse como tal y acompañarse de la distribución conjunta (`dist`, `:381-395`).
- Semilla: `np.random.default_rng(seed)` (`:225`), reproducible.

### 0.4 Librería estadística propia (`mtpy/core/utils/stat.py`)

`Kernel` (`:249-519`): tamaño muestral efectivo `neff = (Σw)²/Σw²` y reescalado de pesos (`:277-281`); anchos de banda Scott/Silverman/ISJ (`:313-422`); ancho adaptativo tipo Abramson `h_i = h·sqrt(gm(f)/f(x_i))` iterado (`:447-490`). `LeastSquaresEstimator` (`:661-857`): WLS por pseudo-inversa, `dof = neff − poly_deg − kvar` (`:720`), covarianza HAC Newey-West con kernel Bartlett (`:753-780`), errores de predicción `sqrt(diag(P Σ Pᵀ))` e intervalos t (`:836-837`). `LocalKernelEstimator` (`:859-994`): regresión local ponderada (LOESS con kernel gaussiano y pesos de encuesta multiplicados por los del kernel). Es una implementación seria y "transparente" (objetivo divulgativo), pero **no tiene tests** ni contraste numérico con `statsmodels` (que está en `requirements.txt` y podría usarse como oráculo en tests).

---

## 1. FiveThirtyEight: ratings de encuestadoras y promedio de encuestas

Fuentes primarias: la metodología de ratings 2024 de 538/ABC (Morris) [538-ratings-2024]; la metodología de promedios "How (most of) our polling averages work" [538-averages]; el codebook del repositorio `fivethirtyeight/data/pollster-ratings` (versiones pre-2024 y 2024) [538-data]; y la continuación del método "era Nate" en Silver Bulletin [SB-ratings]. Las páginas originales de fivethirtyeight.com (2014-2023) redirigen hoy a abcnews.com, y web.archive.org no es accesible desde este entorno, por lo que la descripción del método 2014-2023 se apoya en el codebook, en el datasette público y en Silver Bulletin, que declara aplicar "la misma metodología que los ratings de la era Nate de FiveThirtyEight".

### 1.1 Ratings de encuestadoras (método 2014-2023, continuado en Silver Bulletin)

Columnas del fichero histórico `pollster-ratings.csv` (datasette de 538): `Polls Analyzed`, `NCPP/AAPOR/Roper`, `Banned by 538`, `Predictive Plus-Minus`, `538 Grade`, `Mean-Reverted Bias`, `Races Called Correctly`, `Misses Outside MOE`, `Simple Average Error`, `Simple Expected Error`, `Simple Plus-Minus`, `Advanced Plus-Minus`, `Mean-Reverted Advanced Plus-Minus`, `# of Polls for Bias Analysis`, `Bias`, `House Effect`, `Average Distance from Polling Average (ADPA)`, `Herding Penalty`.

Cadena de cálculo (según [SB-ratings] y [538-data]):
1. **Simple Average Error**: error medio absoluto en el *margen* entre los dos primeros candidatos (encuesta − resultado), sobre encuestas cuya fecha mediana de campo cae en los últimos 21 días (31 en la versión 2024) antes de la elección.
2. **Simple Expected Error / Simple Plus-Minus**: el error esperado se modela por regresión con controles de tipo de elección, días hasta la elección y tamaño muestral; plus-minus = error observado − esperado (negativo = mejor).
3. **Advanced Plus-Minus**: incorpora el rendimiento de las demás casas en las mismas carreras (error relativo a la "dificultad" de la carrera) y pondera más los resultados recientes.
4. **Mean-Reverted Advanced Plus-Minus**: contracción hacia 0 en función del número de encuestas (menos encuestas → más contracción), con descuento de las antiguas.
5. **Predictive Plus-Minus**: la contracción no se hace hacia 0 sino hacia una media que depende de la calidad metodológica: bonus por pertenencia a la AAPOR Transparency Initiative o por depositar datos en el Roper Center (antes también NCPP). La versión 2024 formaliza: `Predictive Error = Adjusted Error · n/(n+n_shrink) + prior_grupo · n_shrink/(n+n_shrink)` [538-ratings-2024].
6. **Herding**: `ADPA` = distancia media de cada encuesta de la casa al promedio de las demás encuestas en campo; si ADPA es menor que el mínimo teórico dado el error muestral, se añade una **Herding Penalty** al Advanced Plus-Minus antes de calcular el Predictive Plus-Minus [SB-ratings].
7. **Bias / Mean-Reverted Bias / House Effect**: sesgo direccional medio (solo carreras D vs R) y su versión contraída; `House Effect` es la desviación sistemática respecto al promedio de encuestas.
8. **Grade**: letras (A+ … F) por umbrales de Predictive Plus-Minus; datos escasos → "A/B", "B/C"; casas vetadas → F.

Versión 2024 (Morris) [538-ratings-2024]: raw error/bias por encuesta → *excess error* restando el esperado de una regresión multinivel (tamaño, días, ciclo, tipo; descuento ~14 %/año) → *relative excess error* frente a las demás encuestas de la carrera (ponderado por tamaño) → mezcla en *adjusted error/bias* → contracción hacia el prior de grupo por transparencia → **POLLSCORE** = media de predictive error y predictive bias → **Transparency Score** (10 preguntas, 0/0.5/1 cada una; 70 % medido + 30 % implícito, donde AAPOR/Roper = 10) → ranking por Pareto sobre (POLLSCORE, transparencia) → estrellas 0.5-3.0; **bootstrap** de 1.000 réplicas para incorporar incertidumbre; la herding penalty se mantiene pero atenuada para casas transparentes.

Correspondencias con el código:
- "Simple Expected Error" ↔ `get_error_estimator` (`computer.py:1571-1606`, WLS de |error| sobre pct/regional/semanas): misma idea (error esperado condicionado), pero en 538 se usa para *ratings* y aquí solo para *simulación*. Los ratings usan en cambio la desviación respecto a la media kernel de contemporáneas (`:1461-1508`), que corresponde al "Advanced Plus-Minus".
- "Mean reversion + prior por calidad" ↔ `:1163-1181` (dos contracciones: normal-normal con τ y mezcla con `quality` con pseudo-conteo 3). 538 contrae una sola vez hacia un prior de grupo *documentado* (afiliación/transparencia); aquí el prior es un número manual por casa.
- "Time weighting" ↔ `poll_rating_weights` (`:1192-1231`): 0.5^pos, 0.7^semana, 0.9^año. 538 usa ~14 %/año (≈0.86^año) sin descuento intra-evento por posición.
- **No hay equivalente** a ADPA/herding penalty, a bias direccional mean-reverted, a bootstrap de incertidumbre del rating, ni a métricas de calibración tipo "Races Called Correctly"/"Misses Outside MOE".

### 1.2 Promedio de encuestas (538, versión 2023-2024) [538-averages]

- **Inclusión**: elección de población (LV > RV > A), exclusión de hipotéticos, eliminación dinámica de solapes en trackings, tope de tamaño muestral efectivo en 10.000 (winsorizado), imputación de tamaño desconocido por mediana de la casa o de todas.
- **Pesos multiplicativos**: (a) `sqrt(n / mediana_n_del_tipo)`; (b) *rapid-fire*: en ventanas de 14 días, todas las encuestas de una casa suman el peso de una; (c) *outlier downweighting* con k-NN y kernel (mínimo 0.05 para outliers al 95 %).
- **Ajustes**: house effects por regresión multinivel con contracción bayesiana hacia 0 (y 2.4 puntos de sobreestimación asumida para encuestas partidistas); conversión de población (±1/±2 puntos); ajuste de tendencia nacional a estados.
- **Promedio**: mezcla de (i) media móvil exponencial (decay con tope −0.2, ventana dura 30-60 días salvo <10 encuestas) y (ii) **regresión local polinómica con kernel** (grado 0-2 elegido por AIC), con parámetro de mezcla que favorece la regresión cuando hay muchas encuestas y la EWMA cuando hay pocas.
- **Hiperparámetros**: 7 hiperparámetros optimizados sobre histórico con tres criterios: verosimilitud de predecir encuestas futuras, autocorrelación de errores y autocorrelación temporal del promedio (evitar tendencias artificiales). Desde 2023, por tipo de promedio.
- **Bandas**: percentil 95 de la incertidumbre para *predecir encuestas futuras*, no el resultado electoral.
- En el forecast presidencial 2024, los ratings entran **escalando el tamaño muestral efectivo** (tope 1.500) en vez de como peso directo [538-forecast-2024].

Correspondencias con el código:
- (a) ≡ `computer.py:572-576`; tope 5.000 vs 10.000; imputación por mediana ≡ `:543-561`.
- (b) ≈ `poll_overweights` (`:1275-1322`), ventana ±4 días y regla `1/(N+1)` (538: 14 días, suma = 1).
- (c) **no existe** (no hay downweighting de outliers en el Forecaster).
- Regresión local ≡ `LocalKernelEstimator` (`stat.py:859-994`) con grado 1 fijo y ancho ISJ adaptativo; **no hay** EWMA ni mezcla adaptativa según densidad de encuestas, ni selección por AIC, ni optimización de hiperparámetros contra histórico (el notebook `notebooks/lab/PollsOptimize.ipynb` sugiere exploración manual).
- House effects: **no se aplican**.

### 1.3 Diferencias de fondo que conviene documentar

1. **Sesgo direccional vs error absoluto**: 538 separa *error* (absoluto) y *bias* (con signo) y corrige house effects antes de promediar. El modelo solo pondera por error absoluto; una casa sistemáticamente sesgada pero "consistente" no es corregida.
2. **Escala de errores**: 538 trabaja en puntos del margen; el modelo usa log-odds por partido (`:1077`) y una conversión `(e^x − 1)·100` (`:1233-1252`). Es defendible (errores relativos comparables entre partidos grandes y pequeños) pero debe explicarse, porque la etiqueta `bias` induce a confusión.
3. **Validación**: 538 valida el promedio por su capacidad de predecir encuestas futuras y calibra los ratings por bootstrap; el modelo no tiene validación retrospectiva sistemática (no hay tests ni un script de backtesting que recorra los 16 eventos con `limit_date` móvil).

---

## 2. Simulación electoral de 538 (errores correlacionados, colas pesadas)

Fuente primaria disponible: metodología del forecast presidencial 2024 [538-forecast-2024]; la de 2020 (fivethirtyeight.com) redirige y no fue accesible, pero la estructura es la misma y la conocida de la literatura (Silver 2020; Gelman/Morris en HDSR discuten sus supuestos [HDSR-2020], [HDSR-2024]).

- **Tres fuentes de error**: (i) error de la encuesta el día de la elección (media de ~2 puntos por candidato en estados competitivos), (ii) deriva entre hoy y la elección (≈6 puntos de cambio esperado del margen a 75 días, decreciente con los días restantes, estimado sobre 300+ días de histórico), (iii) error estatal idiosincrático.
- **Colas pesadas**: los errores se extraen de una **t de Student con 5 grados de libertad**.
- **Correlación**: matriz estado×estado estimada a partir de historia electoral 1948-2020, demografía (raza, educación, edad, renta) y regiones por *fuzzy c-means*; el error temporal también correlacionado.
- **Fundamentals**: 11 indicadores económicos + incumbencia, etc.; error esperado ~6.5 puntos; mezcla con encuestas por *Bayesian model stacking* según incertidumbre (73 % encuestas a 75 días, 98.5 % el día de la elección).
- **Escala**: decenas de miles de simulaciones (10.000 por predictor antes del stacking).

Correspondencias con el código:
- Colas pesadas: `simulator.py:531-534` usa `standard_t(nobs−2)`; con `nobs` grande la t converge a normal, es decir, las colas dependen de cuántas encuestas haya en la ventana kernel y no de un parámetro calibrado como en 538 (df=5). Recomendación: parametrizar `df` y calibrarlo con el histórico de errores (Jennings & Wlezien 2018; Shirani-Mehr et al. 2018 encuentran RMSE ≈ 3.5 puntos, el doble del margen de error nominal).
- Correlación: **ausente** entre partidos (deberían ser negativas por la restricción de suma y positivas dentro de bloques) y entre provincias (el swing proporcional aplica el *mismo* multiplicador nacional a todas las provincias, `:587-588`, por lo que la correlación interprovincial es 1 por construcción y no hay ruido provincial). El equivalente natural en un sistema multipartido es una perturbación en log-ratios con covarianza estimada de errores históricos por partido (Stoetzer et al. 2019) más un término provincial (Montalvo et al. 2019).
- Deriva temporal: en 538 es un término explícito que escala con los días restantes; en el modelo solo entra vía la covariable `weeks` de `v2err` (`computer.py:1571-1606`, `simulator.py:519-525`), que mide error medio histórico por semanas *antes* del evento, lo que es una aproximación razonable pero mezcla error de encuesta y deriva.
- Fundamentals: no hay componente estructural (en España, Magalhães/Aguiar-Conraria/Lewis-Beck 2012 y Montalvo et al. 2019 proponen priors económicos/provinciales).

---

## 3. Literatura académica de agregación de encuestas y predicción electoral

### 3.1 Jackman (2005), "Pooling the polls over an election campaign" (AJPS 40(4): 499-517) [Jackman-2005]
Modelo de espacio de estados: intención latente `ξ_t = ξ_{t−1} + ω_t` (paseo aleatorio con varianza estimada), medida `y_i ~ N(ξ_{t(i)} + δ_{casa(i)}, s_i²)` con `s_i = sqrt(p(1−p)/n_i)` y **house effects δ** identificados anclando el estado final al resultado electoral (o imponiendo Σδ = 0); estimación bayesiana (WinBUGS/Stan). Implementaciones de referencia: Stan (jrnold, freerangestats) [Jackman-Stan], [FRS-2017]. Hallazgo: house effects de hasta ~2.7 puntos en Australia 2004/2007.
*Relación con el código*: la regresión kernel local (`forecaster.py:313-360`) es un suavizador *frecuentista* de la misma señal latente; carece de house effects y de la propagación de incertidumbre del paseo aleatorio hacia el día de la elección. El histórico de 16 eventos permitiría estimar δ por casa (anclando a resultados), que es exactamente lo que `PollsResults.bias` (LOR con signo) ya contiene en bruto.

### 3.2 Linzer (2013), "Dynamic Bayesian Forecasting of Presidential Elections in the States" (JASA 108(501): 124-134) [Linzer-2013]
Jerárquico estado/nacional: `logit(π_{jt}) = β_{jt} + δ_t`, con paseos aleatorios **hacia atrás desde el día de la elección**, cuyo valor final tiene un prior informativo del modelo estructural (Time-for-Change). Verosimilitud binomial por encuesta. Genera probabilidades de victoria por estado y Colegio Electoral. Base del modelo de The Economist 2020 (Heidemanns/Gelman/Morris) y de zweitstimme.

### 3.3 Heidemanns, Gelman & Morris (2020), "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election" (HDSR 2.4) [HDSR-2020], código [Economist-2020]
Añade a Linzer: corrección de no respuesta partidista, modo de encuesta y población; priors estatales que se actualizan; **matriz de correlación estatal empírica** por variables políticas/demográficas; implementado en Stan, MIT + CC-BY 4.0; el README publica calibración y cobertura de intervalos en 2008/2012/2016. Gelman & Morris (HDSR 2024) [HDSR-2024] revisan los problemas de colas y correlaciones excesivas. Es el mejor ejemplo de *repositorio de modelo periodístico con estándares científicos*.

### 3.4 Fisher, Ford, Jennings, Pickup & Wlezien (2011), "From polls to votes to seats: Forecasting the 2010 British general election" (Electoral Studies 30(2): 250-257) [Fisher-2011]; Wlezien et al. (2013) "Polls and the Vote in Britain" (Political Studies) [Wlezien-2013]
Tres etapas explícitas: (1) ajustar y agregar encuestas (house effects), (2) proyectar el cambio hasta el día de la elección (regresión histórica del resultado sobre las encuestas a *d* días, que implica descuento de la ventaja actual hacia el resultado anterior), (3) traducir votos en escaños (swing uniforme, swing proporcional y modelos probabilísticos por circunscripción). Es la estructura conceptual más cercana a `Forecaster → Simulator`; su etapa (2) no tiene equivalente en el código (no hay descuento sistemático de la última estimación hacia el resultado previo o hacia un prior).

### 3.5 Stoetzer, Neunhoeffer, Gschwend, Munzert & Sternberg (2019), "Forecasting Elections in Multiparty Systems: A Bayesian Approach Combining Polls and Fundamentals" (Political Analysis 27(2): 255-262) [Stoetzer-2019]; modelo zweitstimme [ZS-2021], [ZS-2025], replicación [ZS-repo]
Transforma cuotas de partido a **log-ratios** (composición), paseo aleatorio multivariante hacia atrás desde la elección con prior de fundamentals, house effects por casa, matriz de covarianza de errores entre partidos; produce probabilidades de pluralidad y de mayorías de coalición; evaluado en RMSE y calibración. El repositorio de replicación 2025 documenta orden de scripts (`00_run-model.R`…), versiones de R y codebook. Es la referencia directa para un sistema multipartido como el español.

### 3.6 Otras referencias útiles
- Shirani-Mehr, Rothschild, Goel & Gelman (2018), "Disentangling Bias and Variance in Election Polls" (JASA 113(522): 607-614): error total ≈ 3.5 puntos RMSE, el doble del margen nominal [SM-2018]. Base empírica para calibrar `std_err` del simulador.
- Jennings & Wlezien (2018), "Election polling errors across time and space" (Nature Human Behaviour 2: 276-283), 30.000 encuestas, 351 elecciones, 45 países; datos en Dataverse [JW-2018]. Útil para un prior de error en función de días a la elección.
- Erikson & Wlezien (2012), *The Timeline of Presidential Elections*: poder predictivo de las encuestas en función del tiempo [EW-2012].
- Bailey, Pack & Mansillo (2022), PollBasePro / paquete `britpol` [britpol]: estimaciones diarias 1955-2022 por modelo de espacio de estados; buen ejemplo de paquete de datos + modelo + codebook (OSF/Dataverse).
- Montalvo, Papaspiliopoulos & Stumpf-Fétizon (2019), "Bayesian forecasting of electoral outcomes with new parties' competition" (EJPE 59: 52-70; arXiv 1612.03073) [Montalvo-2019]: modelo jerárquico provincia/CCAA/nación para España, agregación de encuestas con house effects, swing provincial jerárquico, D'Hondt por provincia con barrera y simulación de escaños; validado en 2015 con dos partidos nuevos (30 % del voto). **Es la referencia académica más cercana al Simulator.**
- Pavía, García-Cárceles & Badal (2016), "Estimating Representatives from Election Poll Proportions: The Spanish Case" (Statistica Applicata) [Pavia-2016]: dos enfoques basados en modelo para pasar de proporciones nacionales a escaños en elecciones multidistrito, reduciendo el MSE de la proyección directa. Pavía, Larraz & Montero (2008), "Election Forecasts Using Spatiotemporal Models" (JASA 103(483): 1050-1059) [Pavia-2008]: kriging/cokriging para predicción en noche electoral; Pavía & Larraz (2008), "Quick-counts from non-selected polling stations" (J. Applied Statistics 35: 383-405). Pavía es además autor del *Spanish Electoral Archive* (Scientific Data 2021) [SEA].
- Magalhães, Aguiar-Conraria & Lewis-Beck (2012), "Forecasting Spanish elections" (IJF 28: 769-776) [MAL-2012]: modelo estructural para España.
- Alaminos & Alaminos-Fernández (2023), *Métodos y Modelos para la Predicción Electoral: Una Guía Práctica* (Univ. Alicante, RUA) [Alaminos-2023]: guía en español que cubre agregación, transferencia de voto y reparto de escaños.

---

## 4. Modelos españoles

### 4.1 El País (Kiko Llaneras y Borja Andrino, desde 2016)
Descripción pública (artículos y entrevistas; no hay un documento técnico único): cuatro pasos: (1) agregar y promediar encuestas; (2) proyectar el promedio a cada provincia; (3) incorporar la incertidumbre esperada, calibrada con un registro histórico de errores de encuestas (en España ~2 puntos por partido de media, a veces 3 o más) que depende del diseño, del país y de la fecha; (4) simular 15.000 elecciones ("un Montecarlo simple", según Llaneras) para repartir escaños y calcular probabilidades [ElPais-modelo], [Electomania-Llaneras]. Los detalles de ponderación de casas y de la proyección provincial no son públicos.
*Comparación*: la arquitectura de `Simulator` es la misma (promedio → provincia → ruido → D'Hondt → distribución). La diferencia relevante es que El País calibra la incertidumbre con un error histórico *a nivel de partido y fecha* y presenta horquillas por partido; el modelo lo hace con `v2err` (WLS sobre pct/regional/semanas), que es una calibración análoga pero sin correlaciones.

### 4.2 Electomanía / ElectoPanel
Página de metodología [Electomania-metodologia]: panel online con ponderación sociodemográfica y por recuerdo de voto, 1.000-3.000 entrevistas; escaños por **D'Hondt por circunscripción provincial con barrera del 3 % y mínimo de escaños por provincia**; los desgloses territoriales se elaboran "a partir de resultados históricos y de la distribución del voto en convocatorias previas" (sin detallar matrices de transferencia). Advierten de la fragilidad de la asignación provincial en circunscripciones medianas con muchos restos. Su promedio ponderado "Pollwatch" (europeas) pondera a las casas por su desviación histórica en los tres últimos ciclos (peso 5-10 lineal) y usa participación media reciente para convertir a votos [Electomania-pollwatch]. Electomanía publica además rankings de encuestadoras (posible origen del `Electomania-Rankings.xlsx` en `files/`).

### 4.3 Electocracia
Portal de datos abiertos de sondeos y promedio de encuestas con barómetro de escaños; declara que sus proyecciones "incluyen un desglose de los métodos empleados", pero no se ha localizado un documento metodológico detallado [Electocracia].

### 4.4 Key Data (Público)
Descripción publicada por Público: "desk research" de toda la información disponible (comportamiento electoral y encuestas publicadas o no), a la que se aplican "las ponderaciones correspondientes" para llegar a una estimación de voto sobre la que se aplica la ley electoral para asignar escaños [KeyData]. No hay detalle de las ponderaciones.

### 4.5 Politikon
Blog colectivo (Galindo, Llaneras, Simón, etc.) que desde 2014-2016 publicó análisis de encuestas y errores (CIS, cocina electoral) y del que salió el modelo de Llaneras; no existe hoy un modelo Politikon separado con metodología pública [Politikon].

### 4.6 Reparto provincial: swing uniforme, proporcional, "arrastre", D'Hondt y 3 %
- **Swing uniforme** (UNS): sumar a cada provincia la misma variación nacional en puntos; **swing proporcional**: multiplicar la cuota provincial anterior por el cociente nacional (`nuevo/anterior`). El código implementa el **proporcional** (`simulator.py:587-588`), con renormalización a 100 por provincia (`:613`). Fisher et al. (2011) y Wikipedia [UNS] documentan el historial mixto del UNS; Pavía et al. (2016) y Montalvo et al. (2019) muestran que un swing jerárquico (provincia dentro de CCAA dentro de nación) reduce el error en España.
- **"Modelo de arrastre"/transferencia**: la práctica española habitual (CIS, Electomanía, Alaminos 2023) construye la estimación por matrices de transferencia desde el recuerdo de voto; a nivel de agregador, el equivalente es el swing por bloques o por CCAA usando encuestas autonómicas (Wikipedia "Sub-national opinion polling" recoge esas encuestas). El `smap` del código (`:551-580`) es un mecanismo *ad hoc* para reasignar cuotas cuando cambia la oferta (agg/sub/split), sin base empírica documentada.
- **Ley**: LOREG art. 162 (350 diputados; mínimo inicial de 2 por provincia; Ceuta y Melilla 1) y art. 163.1 (a: exclusión de candidaturas con < 3 % de votos válidos en la circunscripción; b-d: cocientes D'Hondt) [LOREG]. El código aplica los cocientes (`alloc_dhondt`, `:482-494`) pero **no la barrera del 3 %**, y toma los escaños por provincia de `events_data` (`get_reg_totals`, `:234-246`) en vez de recalcularlos por población (aceptable si la tabla está actualizada para 2027).

---

## 5. Estándares para un repositorio científico/divulgativo

### 5.1 JOSS (Journal of Open Source Software) [JOSS-checklist], [JOSS-criteria]
Checklist del revisor: repositorio público con **LICENSE OSI**; instalación reproducible según documentación; verificación de las afirmaciones funcionales; documentación con *statement of need*, lista de dependencias gestionada automáticamente, **ejemplos de uso reales**, documentación de API, **tests automatizados** (o pasos manuales verificables) y guías de contribución/soporte; el *paper* (~1.000 palabras) con "Statement of need", "State of the field", "Software design", "Research impact statement" y disclosure de uso de IA; evidencia de desarrollo sostenido (≥6 meses de historial público, releases, issues). Estado actual del repo: 3 commits, README de una línea, sin tests, sin CI, sin docs → lejos de JOSS; pyOpenSci (más orientado a paquetes Python científicos) sería un paso intermedio.

### 5.2 pyOpenSci Python Package Guide [pyOpenSci-guide], [pyOpenSci-submit]
Recomendaciones: `pyproject.toml` (PEP 621) con metadatos y dependencias; layout `src/`; build backend moderno (hatchling/flit/setuptools); publicación en PyPI/conda-forge; documentación con Sphinx o MyST en Read the Docs/GitHub Pages; tests con pytest y CI (GitHub Actions) en varias versiones de Python; linters/formateadores (ruff/black); versionado semántico y changelog; ficheros base README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT. El repo hoy mezcla el modelo con un framework genérico (`mtpy/core`, `api.py`, `worker.py`, `deploy/` con gunicorn/nginx/supervisord) que el modelo no necesita; `requirements.txt` fija `numpy==1.25.2`, `scipy==1.9.3` mientras el intérprete usado tiene numpy 2.4.4/scipy 1.17.0 (deriva de versiones no declarada).

### 5.3 Citabilidad: CITATION.cff + Zenodo DOI [CFF], [Zenodo-GitHub]
`CITATION.cff` (existe en el repo, `/Users/luiss/HD/Proyectos/Code/elections-model/CITATION.cff`) debe incluir `version`, `date-released`, `license`, `repository-code`, `identifiers`/`doi` y opcionalmente `preferred-citation` (artículo/preprint). Activar la integración GitHub→Zenodo para que cada *release* obtenga un DOI y exista un *concept DOI* estable; añadir badge y sección "How to cite" al README; herramientas `cffinit` y `cffconvert`.

### 5.4 FAIR y FAIR4RS [FAIR-2016], [FAIR4RS-2022]
Wilkinson et al. (2016) para datos; Chue Hong et al. (2022, RDA, DOI 10.15497/RDA00068) para software: identificadores persistentes, metadatos ricos (CFF/codemeta), licencias claras, dependencias declaradas, interoperabilidad (formatos abiertos), documentación de procedencia. Para este proyecto lo crítico es la **procedencia de los datos**: encuestas de Wikipedia (`WikipediaLoader`) y resultados de Infoelectoral (`InfoElectoralLoader`) viven en una BD local y en `files/` ignorado por git; un repositorio FAIR necesita un *snapshot* versionado y citable de la tabla de encuestas (CSV/Parquet + codebook + DOI en Zenodo/Dataverse, como PollBasePro o el Spanish Electoral Archive) y un script que reconstruya la BD desde él.

### 5.5 Ten Simple Rules for Reproducible Computational Research (Sandve et al. 2013, PLOS Comp Biol 9(10): e1003285) [Sandve-2013]
1) registrar cómo se produjo cada resultado; 2) evitar manipulación manual; 3) archivar versiones exactas de programas; 4) control de versiones de scripts; 5) guardar resultados intermedios en formatos estándar; 6) **anotar semillas aleatorias**; 7) guardar datos crudos detrás de cada gráfico; 8) salidas jerárquicas; 9) enlazar afirmaciones con resultados; 10) acceso público a scripts, ejecuciones y resultados. Aplicación directa: los notebooks raíz producen `files/fc/*.csv` y `files/img/*.png` no versionados; hace falta un *pipeline* (`make`/`dvc repro`/`snakemake`) que regenere todo desde datos versionados con semilla fija (`Simulator(seed=42)` ya existe, `simulator.py:225`).

### 5.6 Publicación ejecutable: Jupyter Book 2 / MyST, Quarto, Binder, nbval/papermill [JB2], [Quarto], [Binder], [nbval]
- **Jupyter Book 2** (distribución de MyST-MD) o **Quarto** para convertir los notebooks (`PollstersRatings`, `PollsErrors`, `PollsForecast`, `PollsSimulations`) en un sitio/PDF con narrativa, ecuaciones y figuras, publicado en GitHub Pages. Quarto tiene plantillas de revista y manuscritos reproducibles; Jupyter Book 2 está más integrado con el ecosistema Jupyter.
- **Binder/repo2docker** para ejecución en el navegador: requiere que los datos estén en el repo (o descargables) y un `environment.yml`/`requirements.txt` fiel; hoy el modelo depende de PostgreSQL local y `.env`, lo que impide Binder sin un backend de datos en ficheros (SQLite/Parquet).
- **nbval** (pytest plugin) o **nbmake/papermill** para ejecutar notebooks en CI y detectar regresiones; `papermill` permite parametrizar `event_date`, `drange`, `n_sim`.

### 5.7 DVC (Data Version Control) [DVC]
Versionar `files/` (encuestas crudas de Wikipedia, JSON/XLSX de Infoelectoral, `params.json`, salidas `fc/`, `img/`) con DVC apuntando a un remoto (S3/Zenodo/Drive), y definir un `dvc.yaml` con etapas `load → compute → forecast → simulate → report`, de modo que `dvc repro` reconstruya todo y `dvc metrics` registre errores de backtest por evento.

### 5.8 Ejemplos de repositorios modelo a imitar
- `TheEconomist/us-potus-model` [Economist-2020]: README con descripción del modelo, mejoras sobre Linzer, scripts por año, modelos Stan, resultados de calibración y cobertura por ciclo, MIT + CC-BY.
- `zweitstimme-org/prediction-2025-replication` [ZS-repo]: orden numerado de scripts, versiones de R, codebook, salidas (RMSE, tablas) y análisis de escenarios.
- `jackobailey/britpol` [britpol]: paquete R con datos, modelos, `tests/`, codebook en OSF, DOI en Dataverse.
- `fivethirtyeight/data/pollster-ratings` [538-data]: datos crudos + README/codebook por columna + notas de cambio metodológico por año (README_PRE2024 vs README).

---

## 6. Lista de comprobaciones concretas para el agente comparador

(En formato "tema → método de referencia → qué mirar en el código".) Ver también el campo `comparison_hooks` del resultado estructurado.

1. Peso por tamaño muestral: 538 `sqrt(n/mediana)` con tope 10.000 y winsorización ↔ `computer.py:543-576` (tope 5.000, `Stat.winsorize('group')` en `stat.py:119-135`).
2. Encuestas en ráfaga/trackings: 538 ventana 14 días, peso total = 1 ↔ `poll_overweights` `computer.py:1275-1322` (ventana ±4 días, `1/(N+1)`, descarte por solape).
3. Recencia y suavizado: 538 EWMA + regresión local (grado 0-2, AIC, mezcla adaptativa, 7 hiperparámetros optimizados) ↔ `forecaster.py:313-360` + `stat.py:859-994` (grado 1 fijo, ISJ adaptativo, sin EWMA, sin optimización).
4. House effects: 538 regresión multinivel con contracción y aplicación parcial; Jackman δ por casa ↔ **inexistente**; señal disponible en `PollsResults.bias` (`computer.py:693-700`).
5. Outliers: 538 k-NN + kernel (mínimo 0.05) ↔ inexistente en el Forecaster.
6. Rating: 538 Advanced/Predictive Plus-Minus, contracción `n/(n+n_shrink)` hacia prior de transparencia, herding penalty, bootstrap ↔ `computer.py:1094-1190` (desviación kernel, dos contracciones, prior `quality` manual, sin herding, sin bootstrap); docstring desactualizado `:853-905`; `ratings_margin` sin uso.
7. Ponderación temporal del rating: 538 ~14 %/año ↔ `poll_rating_weights` `computer.py:1192-1231` (0.5^pos · 0.7^semana · 0.9^año).
8. Rating → peso en el promedio: 538 escala el tamaño muestral efectivo ↔ `computer.py:1185` (`log1p(rating)/log1p(media)`) y `forecaster.py:267-268` (fallback a `quality` en escala 0-100: bug latente).
9. Incertidumbre del promedio: 538 bandas para predecir encuestas futuras ↔ `err` HAC de la WLS local (`stat.py:753-780, 836-837`), interpretado luego como error del resultado.
10. Deriva hasta la elección: 538 término explícito (≈6 pts a 75 días); Fisher et al. regresión con descuento ↔ solo covariable `weeks` en `v2err` (`computer.py:1571-1606`; `simulator.py:519-525`).
11. Colas y correlaciones: 538 t(5) + matriz estado×estado ↔ `simulator.py:531-534` t(nobs−2) independiente por partido, sin correlación ni restricción de suma; provincias con correlación 1 por swing proporcional (`:587-588`).
12. Proyección provincial: swing uniforme vs proporcional vs jerárquico (Montalvo 2019; Pavía 2016) ↔ `build_umat` `simulator.py:547-596` (proporcional, `smap` ad hoc).
13. D'Hondt: LOREG art. 163.1.a (3 %) y art. 162 ↔ `alloc_dhondt` `simulator.py:482-494` sin barrera; escaños por provincia de `events_data`.
14. Resumen de escaños: `totals` `simulator.py:397-442` (mediana marginal + ajuste a 350) frente a reportar la distribución conjunta y probabilidades de mayoría (Stoetzer 2019).
15. Estimador nacional de escaños sin provincias: `get_seats_estimator` `computer.py:1608-1680` con filtro `seats > 0` (sesgo de selección).
16. Validación: backtesting sobre los 16 eventos con `limit_date` móvil, RMSE/calibración (Stoetzer 2019; Economist README) ↔ no existe; `notebooks/lab/PollsOptimize.ipynb`, `PollstersOptimize.ipynb` son exploratorios.
17. Reproducibilidad: datos en PostgreSQL + `files/` ignorado; `requirements.txt` desfasado; sin tests; CITATION.cff sin versión/DOI/licencia; README de una línea ↔ JOSS/pyOpenSci/FAIR4RS/Sandve.

---

## 7. Referencias (URL)

- [538-ratings-2024] Morris, G. E. (2024). "How 538's pollster ratings work". ABC News/538. https://abcnews.com/538/538s-pollster-ratings-work/story?id=105398138
- [538-averages] 538 (2024). "How (most of) our polling averages work". https://abcnews.com/538/polling-averages-work/story?id=109364028
- [538-forecast-2024] 538 (2024). "How 538's 2024 presidential election forecast works". https://abcnews.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753
- [538-data] FiveThirtyEight, `data/pollster-ratings` (README y README_PRE2024, raw-polls.csv, pollster-ratings.csv). https://github.com/fivethirtyeight/data/tree/master/pollster-ratings
- [538-datasette] Columnas históricas de pollster-ratings (datasette). https://fivethirtyeight.datasettes.com/fivethirtyeight/pollster-ratings~2Fpollster-ratings
- [SB-ratings] Silver, N. "Silver Bulletin pollster ratings" (misma metodología que la era Nate de 538). https://www.natesilver.net/p/pollster-ratings-silver-bulletin
- [538-old-ratings] Silver, N. (2014). "How FiveThirtyEight Calculates Pollster Ratings" (redirige a ABC). https://fivethirtyeight.com/features/how-fivethirtyeight-calculates-pollster-ratings/
- [HDSR-2020] Heidemanns, M., Gelman, A., Morris, G. E. (2020). HDSR 2(4). https://hdsr.mitpress.mit.edu/pub/nw1dzd02/release/1
- [HDSR-2024] Gelman, A., Morris, G. E. et al. (2024). "Grappling with uncertainty in forecasting the 2024 U.S. presidential election". HDSR 6(4). https://hdsr.mitpress.mit.edu/pub/yoa73r1m/release/1
- [Economist-2020] TheEconomist/us-potus-model (R + Stan, MIT/CC-BY). https://github.com/TheEconomist/us-potus-model
- [Jackman-2005] Jackman, S. (2005). AJPS 40(4): 499-517. https://www.tandfonline.com/doi/abs/10.1080/10361140500302472 (PDF: https://www.uh.edu/hobby/eitm/_docs/past-lectures/2015-lectures/harold-clarke/pooling-the-polls-over-an-election-campaign.pdf)
- [Jackman-Stan] Arnold, J. "Simon Jackman's Bayesian Model Examples in Stan: campaign". https://jrnold.github.io/bugs-examples-in-stan/campaign.html
- [FRS-2017] Ellis, P. (2017). "State-space modelling of the Australian 2007 federal election". http://freerangestats.info/blog/2017/06/24/oz-polls-statespace
- [Linzer-2013] Linzer, D. (2013). JASA 108(501): 124-134. https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf ; https://ideas.repec.org/a/taf/jnlasa/v108y2013i501p124-134.html
- [Fisher-2011] Fisher, S., Ford, R., Jennings, W., Pickup, M., Wlezien, C. (2011). Electoral Studies 30(2): 250-257. https://www.sciencedirect.com/science/article/abs/pii/S0261379410000946
- [Wlezien-2013] Wlezien, C., Jennings, W., Fisher, S., Ford, R., Pickup, M. (2013). "Polls and the Vote in Britain". Political Studies. https://journals.sagepub.com/doi/10.1111/1467-9248.12008
- [Stoetzer-2019] Stoetzer, L. F. et al. (2019). Political Analysis 27(2): 255-262. https://www.cambridge.org/core/journals/political-analysis/article/forecasting-elections-in-multiparty-systems-a-bayesian-approach-combining-polls-and-fundamentals/CA929544F672A09A0E34C5529EBFA482 (PDF: https://www.marcel-neunhoeffer.com/pdf/papers/pa_forecast-multiparty.pdf)
- [ZS-2021] Stoetzer et al. (2021). "The Zweitstimme Model: A Dynamic Forecast of the 2021 German Federal Election". PS. https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/abs/zweitstimme-model-a-dynamic-forecast-of-the-2021-german-federal-election/3BF2DA3841C15602D7EA31B3872FD975
- [ZS-2025] Erfort, Stoetzer, Gschwend, Koch, Munzert, Rajski (2025). "The Zweitstimme Forecast for the German Federal Election 2025". PS. https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/zweitstimme-forecast-for-the-german-federal-election-2025-coalition-majorities-and-vacant-districts/9DD258D89C69F73D1AFC284B0CDBE54D
- [ZS-repo] zweitstimme-org/prediction-2025-replication. https://github.com/zweitstimme-org/prediction-2025-replication
- [britpol] Bailey, J. `britpol` / PollBasePro. https://github.com/jackobailey/britpol ; datos https://doi.org/10.7910/DVN/3POIQW
- [SM-2018] Shirani-Mehr, H., Rothschild, D., Goel, S., Gelman, A. (2018). JASA 113(522): 607-614. https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1448823 (PDF: https://sites.stat.columbia.edu/gelman/research/published/polling-errors.pdf)
- [JW-2018] Jennings, W., Wlezien, C. (2018). Nature Human Behaviour 2: 276-283. https://www.nature.com/articles/s41562-018-0315-6
- [EW-2012] Erikson, R. S., Wlezien, C. (2012). *The Timeline of Presidential Elections*. U. Chicago Press. https://press.uchicago.edu/ucp/books/book/chicago/T/bo13948250.html
- [Montalvo-2019] Montalvo, J. G., Papaspiliopoulos, O., Stumpf-Fétizon, T. (2019). EJPE 59: 52-70. https://arxiv.org/abs/1612.03073 ; https://www.sciencedirect.com/science/article/abs/pii/S0176268018302398
- [Pavia-2016] Pavía, J. M., García-Cárceles, B., Badal, E. (2016). "Estimating Representatives from Election Poll Proportions: The Spanish Case". Statistica Applicata. https://www.academia.edu/124882763/Estimating_Representatives_from_Election_Poll_Proportions_The_Spanish_Case
- [Pavia-2008] Pavía, J. M., Larraz, B., Montero, J. M. (2008). JASA 103(483): 1050-1059. https://www.tandfonline.com/doi/abs/10.1198/016214507000001427
- [Pavia-Larraz-2008] Pavía, J. M., Larraz, B. (2008). "Quick-counts from non-selected polling stations". J. Applied Statistics 35(4): 383-405. https://ideas.repec.org/a/taf/japsta/v35y2008i4p383-405.html
- [SEA] Pavía, J. M. et al. (2021). "Spanish electoral archive. SEA database". Scientific Data. https://www.nature.com/articles/s41597-021-00975-y
- [MAL-2012] Magalhães, P., Aguiar-Conraria, L., Lewis-Beck, M. (2012). "Forecasting Spanish elections". IJF 28: 769-776. https://www.sciencedirect.com/science/article/abs/pii/S0169207012000453
- [Alaminos-2023] Alaminos-Fernández, A., Alaminos, A. (2023). *Métodos y Modelos para la Predicción Electoral: Una Guía Práctica*. Univ. Alicante. https://rua.ua.es/bitstream/10045/138240/3/Modelos_y_Metodos_para_la_Prediccion_Electoral.pdf
- [ElPais-modelo] Descripción del modelo de El País (Llaneras/Andrino; 4 pasos, 15.000 simulaciones). https://www.mundiario.com/articulo/politica/kiko-llaneras-concede-opciones-tanto-gobierno-derechas-como-izquierdas/20230719093108274518.html ; https://blogs.uspceu.com/actualidad/quiero-ser-kiko-llaneras/
- [Electomania-Llaneras] "Modelo estadístico de Kiko Llaneras para el País Vasco". https://electomania.es/modelo-estadistico-de-kiko-llaneras-para-el-pais-vasco/
- [Electomania-metodologia] Electomanía, "Metodología". https://electomania.es/metodologia/
- [Electomania-pollwatch] Electomanía, "Pollwatch: promedio de encuestas ponderado para europeas". https://electomania.es/pollwatch-promedio-de-encuestas-ponderado-para-europeas/
- [Electomania-Andalucia] "ElectoPanel Andalucía (I): asignación de escaños en cada provincia". https://electomania.es/electopanel-andalucia-i-asignacion-de-escanos-en-cada-provincia/
- [Electocracia] https://electocracia.com/
- [KeyData] Público, descripción del estudio Key Data. https://www.publico.es/politica/partidos/psoe-retrocede-encuestas-pp-termina-despegar-vox-nuevo-auge.html
- [Politikon] https://politikon.es/author/kikollaneras/ ; https://politikon.es/2016/06/29/otra-vuelta-a-las-encuestas/
- [UNS] Wikipedia, "Uniform national swing". https://en.wikipedia.org/wiki/Uniform_national_swing
- [LOREG] Ley Orgánica 5/1985 (LOREG), arts. 162-163, Junta Electoral Central. https://www.juntaelectoralcentral.es/cs/jec/loreg/contenido?idContenido=2707171&p=1379061423059&template=Loreg%2FJEC_Contenido
- [Wiki-polls-ES] Wikipedia, "Opinion polling for the next Spanish general election". https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Spanish_general_election
- [JOSS-checklist] JOSS review checklist. https://joss.readthedocs.io/en/latest/review_checklist.html
- [JOSS-criteria] JOSS review criteria / submitting. https://joss.readthedocs.io/en/latest/review_criteria.html ; https://joss.readthedocs.io/en/latest/submitting.html
- [pyOpenSci-guide] pyOpenSci Python Package Guide. https://www.pyopensci.org/python-package-guide/index.html
- [pyOpenSci-submit] pyOpenSci author guide. https://www.pyopensci.org/software-peer-review/how-to/author-guide.html
- [CFF] Citation File Format. https://citation-file-format.github.io/
- [Zenodo-GitHub] Zenodo-GitHub integration docs. https://rue-a.github.io/github-zenodo-integration/documentation/
- [FAIR-2016] Wilkinson, M. D. et al. (2016). Scientific Data 3: 160018. https://www.nature.com/articles/sdata201618
- [FAIR4RS-2022] Chue Hong, N. P. et al. (2022). FAIR4RS Principles, RDA. https://www.nature.com/articles/s41597-022-01710-x ; https://doi.org/10.15497/RDA00068
- [Sandve-2013] Sandve, G. K. et al. (2013). PLOS Comp Biol 9(10): e1003285. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285
- [JB2] Jupyter Book 2 / MyST (SciPy 2025). https://proceedings.scipy.org/articles/hwcj9957 ; https://executablebooks.org/en/latest/blog/2024-05-20-jupyter-book-myst/
- [Quarto] Reproducible manuscripts with Quarto. https://jjallaire.quarto.pub/reproducible-manuscripts-with-quarto/
- [Binder] Binder / repo2docker. https://mybinder.readthedocs.io/en/latest/introduction.html ; https://repo2docker.readthedocs.io/en/latest/usage.html
- [nbval] Fangohr, H., Fauske, V. (2020). nbval (arXiv 2001.04808). https://arxiv.org/abs/2001.04808 ; https://github.com/computationalmodelling/nbval
- [DVC] Data Version Control. https://dvc.org/
