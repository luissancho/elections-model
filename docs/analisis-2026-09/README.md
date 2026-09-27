# Análisis en profundidad del modelo electoral (`elections-model`)

Fecha: 2026-09-23. Repositorio: `/Users/luiss/HD/Proyectos/Code/elections-model` (rama `master`, HEAD `d2cd9d5`).
Entorno de ejecución: Python 3.11.9 (pyenv), numpy 2.4.4, pandas 2.2.2, scipy 1.17.0; PostgreSQL local con el esquema `elections` (17 eventos 1977-2027, 3.825 encuestas, 95 encuestadoras).

**Cómo se ha hecho.** Nueve lecturas independientes por módulo (Computer ×3, Forecaster, Simulator, `stat.py`, capa de datos, infraestructura, notebooks), una prueba de ejecución real de punta a punta contra la base de datos (solo lectura), un dosier de referencias externas (FiveThirtyEight, literatura académica, modelos españoles, estándares de repositorios científicos) y, finalmente, verificación manual y empírica de todos los hallazgos de severidad alta y media que se citan aquí. Los informes detallados (≈380 KB) están en `docs/analisis-2026-09/*.md`; este documento los consolida.

---

## 1. Arquitectura y flujo real

```
[Wikipedia (tablas de sondeos)] --WikipediaLoader--> polls, polls_results
[Infoelectoral (XLSX por provincia)] --InfoElectoralLoader--> events_data, events_results
                                                      |
                     PostgreSQL (esquema elections) <-+   + files/params.json (bmaps, smap), pollsters.quality (manual)
                                                      |
  Computer.build_series() -> compute_weights() -> compute_errors() -> compute_deviations() -> compute_ratings()
        (pesos por encuesta: weight_sample, weight_over, weight_rating)        (ratings out-of-sample por evento)
                                                      |
  Forecaster(event, bmap).build_series().fit_forecast(max_fc)   -> regresión local lineal ponderada por partido/bloque
                                                      |
  Simulator(event, drange, seed) -> fit_forecast(fillna=True) -> run(split, random, n_sim)
        LS: regresión lineal escaños~pct   MC: LS + ruido t   DH: swing proporcional + D'Hondt   MT: DH + ruido t
                                                      |
  totals(), dist(), plot_* -> files/fc/*.csv, files/img/*.png
```

- Los **notebooks son la única interfaz** (`notebooks/data-load/*` escriben en la BD; `notebooks/Polls*`, `Pollsters*` analizan; `notebooks/lab/*` exploran).
- El modelo vive dentro de `mtpy`, un framework personal genérico de ≈27.000 líneas. El modelo usa ≈5.150 líneas de `mtpy/lib/`, las 1.144 de `mtpy/core/utils/stat.py` y subconjuntos pequeños de `app.py`, `io.py`, `data.py` (`Model`), `helpers.py`, `dates.py` y `dataviz.py`.
- Tiempos medidos: `Computer.build_series` 0,6 s; ajuste del Forecaster ≈0,65 s por partido; `Simulator.__init__` 1 s; `fit_forecast` de 13 partidos 8 s; 1.000 simulaciones MT ≈25 s.

---

## 2. Qué hace cada módulo (fiel al código, no a las docstrings)

### 2.1 Computer (`mtpy/lib/computer.py`, 2.625 líneas; 35 % son gráficos)

**Serie** (`build_series`, 401-474): una única tabla ancha (3.842 × 511: 3.825 encuestas + 17 resultados, 488 columnas de partido casi vacías), índice `(event_date, date, pollster_id, sponsor_id)`. `date` es la fecha de fin de campo; `days = event_date − date`.

**Pesos por encuesta** (`compute_weights`, 476-635):

| Peso | Fórmula real | Notas |
|---|---|---|
| `proc_sample` | imputación en cascada (mediana casa+evento → casa → global) → `clip(0, 5000)` → winsorización `μ±3σ` **global 1977-2027** | tope efectivo hoy 4.715, no 5.000; cambia al añadir encuestas |
| `weight_sample` | `sqrt(proc_sample / mediana_evento(proc_sample))` | misma forma que 538; mediana 1 por evento; rango 0,58-2,17 |
| `weight_over` | 0 si el campo se solapa con una encuesta posterior de la misma casa (trackings); si no, `1/(n_prev+n_post+1)` en ±`wspan=4` días | análogo al "rapid-fire" de 538 (ventana 14 días, suma 1) |
| `weight_rating` | `log1p(rating)/log1p(media_casas(rating))`, rating ∈ [0,1] | rango 0-3,08, media 0,92; `rating=0` ⇒ peso 0 (KeyData, 23 encuestas 2027) |

El Forecaster multiplica los tres (`forecaster.py:271`). **No hay peso temporal en el promedio**: la recencia la gestiona el kernel de la regresión local (la docstring de clase, 56-59, dice lo contrario).

**Errores** (`compute_errors`, 637-774), por encuesta:
- `error_avg`: media del |error en puntos| sobre los partidos `main`, **ponderada por la cuota real de cada partido**.
- `error_blocks`: error **con signo** del margen Izquierda − Derecha (`vs`). Es el único "sesgo" direccional que se calcula, y no se usa en ningún ajuste posterior.
- `bias_avg`, `bias_blocks`: media ponderada del **valor absoluto** del log-odds-ratio encuesta/resultado; `bias = 0,7·bias_avg + 0,3·bias_blocks`.
- Se almacenan en "% de odds" con `lor_to_bias(x) = (e^x − 1)·100` (inversa exacta de `bias_to_lor`).

Cifras (encuestas featured 1993-2023, n=3.088): `error_avg` media 4,03 pts (mediana 3,62); `error_blocks` media −1,33 pts (las encuestas tienden a sobreestimar a la derecha); `bias` mediana 20,6 % de odds.

**Desviaciones** (`compute_deviations` + `fit_bias_estimator`, 776-851, 1461-1508): por evento, regresión local (kernel gaussiano, ISJ adaptativo, grado 1, HAC) del `bias` en escala LOR frente a la fecha, pesos `weight_over·weight_sample`. `bias_dev = lor − media(t)`; después shrinkage normal-normal hacia 0 con prior τ = 0,01 y "σ del dato" = **error estándar de la media local** (no la dispersión del sondeo). Empíricamente, sd de `bias_dev` bruta = 0,038 (3,8·τ); factor de shrinkage mediana 0,49.

**Ratings** (`compute_ratings` + `pollster_ratings`, 853-1006, 1094-1190). Para el evento *i* solo se usan encuestas de eventos < *i* (sin fuga de información; el primer evento con historial, 1993, queda sin rating). Dentro:
1. `w = weight_over · weight_sample · 0,5^pos · 0,7^semanas · 0,9^años`; se eliminan `w < 0,01`. Efecto real: **sobreviven 343 de 2.238 encuestas (15 %)**, todas a ≤82 días de la elección; 15 de 68 casas quedan solo con el prior.
2. Media ponderada de `bias_dev_adj`; `se = sqrt(Σw²σ²)/Σw · sqrt((n_w+1)/n_w)` (la primera parte es correcta; el factor es ad hoc).
3. Segundo shrinkage con el mismo prior N(0, τ²) sobre valores ya posteriores.
4. `rating_adj = Φ(−dev/τ)` (una casa media obtiene 0,5; con −τ, 0,84; con −2,5τ, 0,99: **satura**).
5. `rating = (n_w·rating_adj + 3·quality)/(n_w + 3)`, con `quality` **manual** (0-78; 53 de 95 casas valen 5; CIS = 10; sin rúbrica documentada).
6. `weight_rating` como arriba, normalizado por la media de **todas** las casas registradas (incluidas ~50 inactivas).

La docstring de `compute_ratings` (866-905) describe otro algoritmo (`num_related`, `error_expected`, escala [5-95], `ratings_margin`) que **no existe** en el código; `ratings_margin` no se usa.

**Estimadores para el simulador** (1510-1680):
- `v2err`: WLS `|error| = β0 + β1·pct + β2·regional + β3·semanas`. Con lo que pasa el Simulator (`n_last=1`, 1.634 filas): β = (1,26; 0,043; −1,19; 0,013), R² 0,39; predice valores negativos para partidos pequeños (17 % de las filas en los datos por defecto) y es fuertemente heterocedástico.
- `v2seats`: WLS `escaños = γ0 + γ1·pct + γ2·regional`, pesos `0,97^años`, entrenado **solo con pares (partido, elección) con escaños > 0** (se descartan 162 de 349, el 46 %). γ = (−16,95; 4,51; 15,71), R² 0,99, error 4,7 escaños. Un nacional recibe 0 hasta el 3,76 % y después 4,5 escaños/punto; un regional recibe +15,7 fijos. Aplicado a los % reales de cada elección suma entre 316 y 376 escaños (2023: 359,7).

### 2.2 Forecaster (`mtpy/lib/forecaster.py`, 1.185 líneas)

- `build_series`: encuestas del ciclo + resultado anterior; `weight = weight_over·weight_sample·weight_rating` (si `weight_rating` es nulo se rellena con `quality` **en escala 0-100**: bug latente, hoy no se dispara para 2027 pero sí para 1993); bloques según `bmap` (`build_blocks`/`group_results`); columna `'-'` = 100 − Σ bloques.
- `fit(name)`: `LocalKernelEstimator(serie, pesos).fit(rejilla diaria hasta última encuesta + max_fc)`. Es un **LOESS de grado 1 con kernel gaussiano, pesos externos multiplicativos, ancho de banda piloto ISJ (Botev) y ancho adaptativo tipo "balloon"** (depende del punto de evaluación), con varianza sandwich HAC (Bartlett, 1 retardo) y bandas t con `dof = n_eff_local − 2`.
- Valores reales (PP, 306 encuestas 2023-2026): piloto ISJ 71,0 días (Scott 109,6; Silverman 93,1); ancho adaptativo entre 61 y 130 días, **máximo en la última fecha** (128,6, por la caída de densidad en la frontera); en la última fecha: media 31,91, IC95 [30,56; 33,27], `err` 0,676, `nobs` 101, `neff` 48.
- `fit_forecast(fillna=True)` arrastra hacia delante el último valor **y su error** hasta la fecha electoral (11 meses en el caso actual): el "pronóstico" es una foto congelada de `última encuesta + max_fc`.
- **No hay ajuste de house effects** (las columnas `bias`, `bias_dev_adj` existen en la serie pero `fit` no las usa), ni restricción composicional (Σ libre, `'-'` puede ser negativo).
- Las bandas `cmin/cmax/err` son **intervalos de confianza de la media suavizada**, no intervalos de predicción del resultado.

### 2.3 Simulator (`mtpy/lib/simulator.py`, 668 líneas)

- Construye internamente un `Forecaster` (`bmap = parties.event`, 13 partidos en 2027) y un `Computer` con las elecciones anteriores (1982-2023) para `v2seats` y `v2err`. `limit_date = event_date − drange[0]` (6 días: la veda de la LOREG). `weeks = (drange[0]+1)//7 = 1` **siempre**, aunque el último sondeo esté a 49 semanas.
- `build_frame`: `std_err = sqrt(err² + pct_err²)`; `vpred = max(0, pct + std_err · t_{nobs−2})`, **sorteo independiente por partido**, sin restricción de suma (Σ `vpred` entre 84 y 102 en 30 simulaciones; corr(PP, VOX) ≈ 0).
- `build_umat`: reglas `smap` (`agg`, `sub`, `split`, `regions`) sobre los % previos por provincia; después **swing proporcional**: `vpred_pct[r,p] = prev_pct[r,p] · vpred[p]/prev_pct_es[p]`. Sin ruido provincial (correlación 1 entre provincias por construcción).
- `simulate` (`split=True`): renormaliza a 100 entre los partidos incluidos por provincia (el voto a "otros" se redistribuye), convierte a votos con el total de la elección anterior y aplica `alloc_dhondt` (D'Hondt puro, correcto, **sin la barrera del 3 %** del art. 163.1.a LOREG). Con `split=False`: `clip(v2seats.predict, 0)` sin restricción a 350.
- `totals()`: mediana por partido y ajuste a 350 repartiendo el déficit **"por ciclos" a todos los partidos** y luego por restos.

Resultados medidos (2027-08-22, `max_fc=3`, `seed=42`):

| Modo | PP | PSOE | VOX | SUMAR | ERC | EHB | JxCat | PNV | BNG | CC | UP | SALF | UPN | Σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LS bruto | 126 | 105 | 61 | 10 | 9 | 5 | 4 | 3 | 2 | 1 | 0 | 0 | 0 | 326 |
| LS `totals()` | 128 | 107 | 63 | 12 | 11 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 1 | 350 |
| MC n=30 (máximo por partido) | 148 | 125 | 79 | 27 | 11 | 7 | 6 | 4 | 4 | 2 | 10 | **0** | **0** | 291-358 por simulación |
| MC `totals()` | 128 | 108 | 65 | 12 | 11 | 7 | 6 | 5 | 3 | 2 | 1 | **1** | **1** | 350 |
| DH | 139 | 112 | 61 | 8 | 8 | 7 | 4 | 5 | 2 | 1 | 2 | 0 | 1 | 350 |
| MT n=200 (mediana) | 137 | 112 | 62 | 8 | 9 | 7 | 4 | 5 | 2 | 1 | 2 | 0 | 1 | 350 |

MT n=200, cuantiles 2,5/50/97,5: PP 115/136/155; PSOE 94/111/134; VOX 42/61/81; SUMAR 2/8/20.

### 2.4 Librería estadística (`mtpy/core/utils/stat.py`, 1.144 líneas, sin docstrings)

| Clase | Qué implementa | Verificación |
|---|---|---|
| `Stat` | media/varianza/cuantiles/IC ponderados con tamaño efectivo de Kish `neff=(Σw)²/Σw²` y pesos reescalados a `neff`; winsorización; escalados; metaclase `@pstatic` | media, var (ddof=1), `std_mean` coinciden con `DescrStatsW`; cuantil = Hyndman-Fan tipo 2. Bugs en `winsorize('group')`, `conf(α≥1)`, `quantile` con NaN, `@pstatic` con posicionales |
| `Kernel` | kernel gaussiano sin constante; Scott, Silverman, **ISJ de Botev (port fiel de KDEpy)**; ancho adaptativo iterado | Bug de truncado de `bw` por dtype `<U3`; ISJ cae al *fallback* `2π·espaciado` para n≤50; no invariante a la escala de pesos; adaptativo evaluado en los puntos de predicción y sin normalizar por `h` en iteraciones ≥2 |
| `LeastSquaresEstimator` | WLS por pseudoinversa, HAC Newey-West (Bartlett), bandas t | coeficientes = `sm.WLS`; HAC = statsmodels (ratio 1,006); `dof` incorrecto si `poly_deg>1` y `kvar>1`; `r2_score` mal centrado (0,445 vs 0,211) |
| `LocalKernelEstimator` | regresión local lineal `kernel × pesos`, umbral 1e-2, WLS por punto | correcto como LOESS; ventana con `neff≤2` aborta toda la predicción; `predict()`/`r2_score` sin `p` fallan con fechas; O(N²) |
| `KernelDensityEstimator` | KDE 1-D/2-D | no integra a 1; `outliers='remove'` lanza `IndexError`, `'group'` no hace nada |

### 2.5 Datos y cargadores (`mtpy/lib/loader.py`, `mtpy/lib/data.py`, `mtpy/models/elections.py`)

- **Wikipedia** (`WikipediaLoader`): scraping en vivo de las tablas "Opinion polling for the … Spanish general election" (inglés), sin archivar HTML ni `oldid`; `date = end_date = pub_date` (no existe fecha real de publicación); el contexto `exit`/`wban` se detecta por color de fila pero se guarda en la clave `context`, que `format_data` descarta: **`ctype` nunca se persiste** (la serie 2027 tiene 0 `ctype`; los 84 existentes son de una versión anterior); casas con nota (`CIS (Sondaxe)`, `CIS (Target Point)`…) se descartan en silencio.
- **Infoelectoral** (`InfoElectoralLoader`): XLSX `PROV_02_YYYYMM_1` + JSON manual de mapeo de candidaturas a siglas (decisiones de curación no documentadas). `pct` sobre votos válidos (incluye blancos).
- **BD**: 13 tablas; sin claves foráneas, sin DDL versionado, sin volcado. Columnas crudas y calculadas mezcladas en `polls`/`polls_results`; `save_model_data` hace `UPDATE … SET col = NULL` sobre **todo el evento** antes del `upsert`. `events_data` de 2027 es copia manual de 2023.
- **Curación manual sin rúbrica**: `pollsters.quality`, `parties.block`, `params.json` (solo 4 eventos), `wp-maps.json`.
- Todo `files/` está en `.gitignore`.

---

## 3. Fortalezas reales del modelo

1. **Separación temporal estricta en los ratings**: el rating aplicado al evento *i* solo usa eventos anteriores. Es la propiedad clave para un backtest honesto y está bien resuelta.
2. **Trabajar en log-odds** hace comparables los errores de partidos grandes y pequeños; la pareja `lor_to_bias`/`bias_to_lor` es una inversa exacta.
3. **Regresión local de grado 1** (no Nadaraya-Watson): corrige automáticamente el sesgo de frontera en la última fecha, que es la que importa.
4. **Tamaño muestral efectivo de Kish** en `Stat`, `Kernel` y WLS: las fórmulas ponderadas resultan coherentes con el estimador insesgado para pesos de fiabilidad. HAC y WLS coinciden con statsmodels.
5. **ISJ de Botev portado a mano** y legible: excelente material divulgativo.
6. **Cadena Forecaster → error histórico → ruido → D'Hondt materializada en arrays inspeccionables** (`frames`, `units`, `results`) y semilla reproducible (`default_rng(seed)` en cada `run`).
7. La regla `sub` del `smap` (reparto del voto previo entre origen y escisión según las estimaciones actuales) está bien pensada.
8. `weight_over` resuelve de forma simple trackings y casas que inundan el mercado.
9. Notebooks con celda de parámetros homogénea (fácil de parametrizar con papermill), separación `data-load / análisis / lab`, nombres de figuras sistemáticos.

---

## 4. Hallazgos verificados

Estado: **E** = confirmado ejecutando código; **L** = confirmado por lectura del código; **P** = plausible (cuantificado por un lector, no reproducido de nuevo). Solo se listan alta y media; las ~130 de severidad baja están en los informes por módulo.

### 4.1 Bugs de código

| # | Sev. | Dónde | Qué ocurre | Est. | Arreglo |
|---|---|---|---|---|---|
| B1 | alta | `stat.py:285-291` | `np.repeat('isj', k)` crea un array `<U3`; el ancho de banda calculado se **trunca a 3 caracteres** al asignarlo (0,3363 → 0,3; ISJ real del PP 71,62 → 71,0; `'scott'` a 5 caracteres). Afecta a la configuración por defecto del Forecaster y del Computer | E | `self.bw = [self.bw] * self.kvar` (lista) antes del bucle |
| B2 | alta | `simulator.py:404-435` | `totals()` reparte el déficit hasta 350 sumando `diff // n` a **todos** los partidos: en LS (bruto 326) UP 0→2, SALF 0→1, UPN 0→1; en MC, SALF y UPN reciben 1 escaño con **máximo 0 en todas las simulaciones**. Son las cifras que se publican | E | Para DH/MT publicar media o cuantiles (ya suman 350 por simulación); para LS/MC restos mayores puros y nunca a partidos con predicción 0; o renormalizar `v2seats` por simulación |
| B3 | alta | `simulator.py:550-561` | `build_umat` recorre todas las reglas `smap` con `frame.loc[key]`: `run(names=['PP','PSOE','VOX'])` → `KeyError: 'UP'`. El parámetro `names` es inutilizable en eventos con `smap` | E | saltar reglas cuyo `key`/`names` no estén en `params['names']` |
| B4 | alta | `simulator.py:572-578` + `params.json` | La regla `regions` compara **códigos INE** (`['3','13','46']`, `['34']`) con **nombres** de provincia. Efecto real en 2019-04-28 (modo DH): COMPROMIS 0, NA+ 0, JxCat 0 escaños (reales 1, 2, 7); JxCat además por no tener voto previo aplicable (B5) | E | mapear con `es-provinces.csv` o indexar `prev_results` por `region_id` |
| B5 | media | `simulator.py:580-588` | Partido sin voto previo y sin regla `smap` aplicable → `prev_pct` NaN → 0 escaños en DH/MT sin aviso | E | lanzar aviso/excepción; reparto por defecto (uniforme o por censo) |
| B6 | alta | `loader.py:519` vs `elections.py:128` | El contexto se guarda como `context`; el modelo espera `ctype`; `format_data` lo descarta → `drop_ctypes=['wban','exit']` no puede actuar sobre datos nuevos | E | renombrar a `ctype` |
| B7 | alta | `data.py:404-411` | `save_model_data` pone a NULL las columnas de **todas** las encuestas del evento y luego hace `upsert` solo de `df`: con `overwrite=False` se pierden los pesos de las encuestas ya calculadas | L | restringir el `UPDATE` a las claves de `df` o eliminarlo (el `upsert` ya sobrescribe) |
| B8 | media | `computer.py:531-599` | Modo incremental (`overwrite=False`): medianas, winsorización y solapes se calculan solo sobre las encuestas nuevas → pesos distintos al recálculo completo | P | estadísticos siempre sobre `self.polls` completo |
| B9 | media | `computer.py:200-211` | `merge_bmaps` itera cadenas carácter a carácter (partido fantasma `'P'`), lanza `KeyError` si un evento no define el bmap (`'max'` con todos los eventos) y **muta `event_params`** por aliasing | P | normalizar `str→[str]`, copiar listas, `.get()` |
| B10 | media | `forecaster.py:270` | `weight_rating` nulo se rellena con `quality` en escala 0-100 (peso ≈ 1) | E (latente: 0 nulos en 2027; 53 encuestas de 1993 afectadas) | rellenar con `log1p(quality/100)/log1p(media)` o recalcular ratings |
| B11 | alta | `stat.py:131` | `winsorize('group')` manda los outliers **bajos** al máximo interior (`[-100,10..13,200]` → `[13,…,13]`) | E (latente en `proc_sample`) | `np.clip(data, min_in, max_in)` |
| B12 | alta | `stat.py:183-184, 203-204` | `conf(α≥1)` usa `erf(n/√2)` como cuantil: `conf(1)=0,48σ`, `conf(2)=1,69σ` | E (latente) | `q = (1+erf(n/√2))/2` |
| B13 | alta | `stat.py:1060-1089` | KDE `outliers='remove'` → `IndexError`; `'group'` sin efecto | L | usar `x_samp.data` y máscara sobre datos originales |
| B14 | media | `stat.py:335-338, 384-399` | ISJ con n≤50 cae **siempre y en silencio** al *fallback* `2π·espaciado` | E | bracket `1e-12 + …` como KDEpy; avisar |
| B15 | media | `stat.py:736` | `dof = neff − poly_deg − kvar` incorrecto con `poly_deg>1` y `kvar>1` (196 vs 194) | E | `neff − exog.shape[1]` |
| B16 | media | `stat.py:781-790` | `r2_score` de WLS centra con `mean(√w·y)` | P | R² ponderado estándar |
| B17 | media | `stat.py:909-927, 736-738` | Ventana local con `neff≤2` lanza `ValueError` y aborta toda la predicción | P | devolver NaN en ese punto |
| B18 | media | `stat.py:620-627, 886-897` | `predict()`/`r2_score` sin `p` fallan con índice temporal; `r2_score` sobrescribe `result` | P | no re-aplicar `ts_to_delta`; no mutar `result` |
| B19 | media | `stat.py:161-172` | `quantile` con NaN y `dropna=False` indexa mal; usa `neff` como longitud | P | filtrar NaN antes; usar `len` |
| B20 | media | `simulator.py:537` | `standard_t(nobs−2)`: `nobs≤2` → `ValueError`; NaN → NaN; `nobs=3` → Cauchy | L | `df = max(neff−2, 3)` o normal |
| B21 | media | `data.py:305-306` | `drop_contexts` filtra por `t.context`, columna inexistente → error SQL | L | `t.ctype` |
| B22 | media | `loader.py:384-401` | Casas con nota (`CIS (Sondaxe)`) no están en el mapa → fila descartada en silencio | L | separar nota/subcontratista; avisar |
| B23 | media | `computer.py:2348` | `plot_ratings_prior` usa `rating_prior`, columna inexistente | L | eliminar o recalcular |
| B24 | media | `forecaster.py:339-340, 391`; `utils.py:59` | `fit_forecast` revienta si un bloque no tiene datos; `build_blocks` si el primer partido del bloque no aparece | P | `continue` con aviso; `.get()` |
| B25 | media | notebooks `PollsForecast`, `PollstersEvent`, `events/*` | `np.NaN` (eliminado en NumPy 2) → `AttributeError`; `PollsForecast.ipynb` **no se ejecuta** en el entorno actual | E | `np.nan` |
| B26 | media | `forecaster.py:400-401, 472`; `utils.py:91`; `loader.py:145,150`; `simulator.py:421,435` | `fillna(method=)`, `applymap`, `groupby(axis=1)`, `ds[i]` posicional: funcionan con FutureWarning en pandas 2.2 y **rompen en pandas 3**; ocultos por `warnings.filterwarnings('ignore')` en `mtpy/mtpy.py:15` | E | `.ffill()`, `.map`, `.T.groupby().sum().T`, `.iloc` |

### 4.2 Debilidades metodológicas (afectan a la validez, no rompen nada)

| # | Sev. | Dónde | Qué | Est. |
|---|---|---|---|---|
| M1 | alta | `simulator.py:481-494, 616-623` | **No se aplica la barrera del 3 %** por circunscripción (art. 163.1.a LOREG); además D'Hondt se aplica sobre % renormalizados entre partidos incluidos, no sobre votos válidos (necesita conservar "otros" y blancos). Solo importa en Madrid y Barcelona, donde se juegan 69 escaños | E |
| M2 | alta | `simulator.py:536-539` | Errores por partido **independientes**, sin restricción de suma ni correlación (PP/VOX, PSOE/SUMAR): las mayorías de bloque, que es lo que se publica, no tienen la incertidumbre correcta. En DH/MT la renormalización provincial absorbe el exceso; en LS/MC nada (suma 291-358) | E |
| M3 | alta | `computer.py:1608-1680` | Estimador de escaños (modos LS/MC) **lineal, sin restricción de suma, entrenado solo con `seats>0`** (46 % de casos excluidos): residuos sistemáticos por tramo, negativos por debajo del 3,76 %, +15,7 fijos a regionales | E |
| M4 | media | `simulator.py:587-588` | Swing proporcional **determinista**: toda la varianza de escaños proviene de 13 números nacionales; no hay ruido provincial ni correlación regional; ERC/EHB/PNV apenas varían | E |
| M5 | media | `forecaster.py:337-352`; `computer.py:1129` | **Sin house effects**: el sesgo con signo por casa (`error_blocks`, `PollsResults.bias`) se calcula pero no se resta antes de promediar. Es la diferencia principal con 538 y con Jackman/Linzer | L |
| M6 | media | `stat.py:753-778` | El error estándar HAC (1 retardo, orden de filas) **ignora la correlación intra-casa**: en la última fecha SE 0,668 frente a 0,89-0,91 con cluster por encuestadora (IC ≈35 % demasiado estrecho); la sd de residuos medios por casa (2,78) explica casi toda la sd residual (3,11) | P |
| M7 | media | `stat.py:447-490` | Ancho de banda adaptativo: (a) la media geométrica se calcula sobre la **rejilla de predicción**, luego `fit(max_fc=0)` y `fit(max_fc=10)` dan valores distintos para la misma fecha; (b) el piloto no se normaliza por `h` en iteraciones ≥2, así que solo la primera es Abramson y el ancho de frontera crece con `n_iter` (71 → 99 → 118 → 129 → 141) | L+P |
| M8 | media | `stat.py:251-291` | El ancho de banda de una **regresión** se elige con selectores de **densidad** (ISJ/Scott/Silverman sobre las fechas), no por validación cruzada del error de predicción. Es la decisión con más impacto en el promedio (≈70 días) y no está justificada | L |
| M9 | media | `forecaster.py:399-401`; `simulator.py:313-317` | Con `fillna=True` el pronóstico y su error quedan **congelados** desde `date_last+max_fc` hasta la elección: la incertidumbre no crece con el horizonte | E |
| M10 | media | `simulator.py:311, 509` | `weeks` = 1 siempre; el error histórico se evalúa "a una semana" aunque el último sondeo esté a 49 | L |
| M11 | media | `computer.py:822-830, 1140-1170` | Shrinkage bayesiano mal especificado: σ del dato = SE de la media local (no la dispersión del sondeo); τ = 0,01 es 3,8× menor que la dispersión real; **doble shrinkage** con incertidumbres ya posteriores; `rating_adj = Φ(−dev/τ)` satura (Noxa 98,5 con 6 sondeos) | P |
| M12 | media | `computer.py:1118, 1178` | Prior `quality` **subjetivo y sin rúbrica**; `min_polls=3` equivale a ≈1,5 elecciones de evidencia (n_w ≤ 2 por evento) | L |
| M13 | media | `computer.py:1559`; `simulator.py:530` | Se usa el **MAE** predicho como σ (para una normal `E|e| = 0,8σ`: infraestima ≈20 %) y se mezcla el error de una encuesta individual con el del promedio; el estimador puede ser negativo y se eleva al cuadrado | L |
| M14 | media | `computer.py:1602, 1674` | Covarianza HAC en estimadores de sección cruzada ordenados por `pct` (no hay serie temporal) | L |
| M15 | media | `computer.py:179-212`; `utils.py:96` | Las métricas de bloque dependen de qué eventos se cargan (`merge_bmaps` une bmaps de todos) y los bloques se suman con partidos ausentes (`min_count=1`) | P |
| M16 | media | `forecaster.py:298, 395` | Sin restricción composicional; `'-'` se recalcula con el subconjunto `names` (con `names=['PP','PSOE']` vale 41 %) | E |

### 4.3 Reproducibilidad, infraestructura y datos

| # | Sev. | Qué | Est. |
|---|---|---|---|
| R1 | alta | **Nada del flujo es ejecutable desde un clon**: `files/` (params.json, mapas Wikipedia, CSV de provincias, XLSX/JSON Infoelectoral) está en `.gitignore`; la BD y `.env` son locales; no hay DDL ni volcado; `mtpy.run()` falla además porque `log/` no existe (`FileHandler` en `app.py:173`) | E/L |
| R2 | alta | Scraping de Wikipedia en vivo sin snapshot ni `oldid`: dataset no reproducible ni citable | L |
| R3 | alta | `requirements.txt` fija numpy 1.25.2 / scipy 1.9.3 pero el código corre con 2.4.4 / 1.17; faltan `lxml`, `openpyxl`, `xgboost`; ≈25 paquetes ajenos al modelo; sin lockfile ni `pyproject.toml` | E |
| R4 | alta | Importar el modelo carga 211 módulos, incluidos `xgboost`, `mlxtend`, `sklearn`, `dask`, por `dataviz.py:34-43 → learning.py`, que el modelo no usa | E |
| R5 | media | Notebooks en **Git LFS** con outputs embebidos (1-5 MB) y desfasados respecto a la BD; GitHub no los renderiza | L |
| R6 | media | Credenciales reales (AWS, BD, Pushover) en `deploy/*.env` y `.env` dentro del árbol (ignoradas, nunca en el historial); sin `.env.example` | L |
| R7 | media | `warnings.filterwarnings('ignore')` global al importar `mtpy` | E |
| R8 | media | Código muerto/roto: `pipelines/events/*` importa `sources.Events` (módulo vacío) y `config.elections.*` inexistente; `jobs/Update` referencia stripe/aemet; `crontab` llama a un job `rep` inexistente; `Worker` usa `app.qs` nunca registrado; tablas `provinces`, `pollsters_parties`, `forecasts*` sin uso | L |
| R9 | media | Los 6 notebooks de `events/*` importan `mtpy.lib.elections.*` (inexistente) con `sys.path` a 4 niveles; `lab/PollstersOptimize` duplica ~60 líneas de `pollster_ratings` | L |
| R10 | media | "Run All" en `data-load/*` escribe y borra en PostgreSQL sin confirmación (`save = True`, `overwrite=True` en `compute_series`) | L |
| R11 | media | `Simulator(seed=None)` por defecto; `np.random.seed(42)` del notebook no afecta a `default_rng` | L |
| R12 | media | `path` es relativo a la raíz de `app.fs` (`files/`), pero su valor por defecto es `os.getcwd()`: `Forecaster()` sin `path` falla fuera de `files/`; los notebooks funcionan porque pasan `path='.'` | E |
| R13 | media | Columnas crudas y calculadas mezcladas en `polls`; `overwrite=True` del loader destruye columnas curadas (`ctype`, `notes`, `url`) | L |

### 4.4 Documentación y nomenclatura

- `bias*` son **errores absolutos** en escala de odds; el único sesgo con signo se llama `error_blocks`. Para un repositorio científico hay que renombrar o documentar de forma explícita (`abs_lor_error`, `gap_bias`).
- Docstring de `compute_ratings` describe un algoritmo inexistente; `ratings_margin` no se usa; `get_error_estimator` cita parámetros inexistentes; docstring de clase `Computer` habla de "proximity to election date"; `Forecaster` "Polls corresponding to a single election event"; `alpha` del Simulator no produce ningún IC; `stat.py` sin docstrings.
- Enlace de 538 en la docstring (`computer.py:861`) está muerto (redirige a ABC).

---

## 5. Comparación con FiveThirtyEight y con la literatura

| Componente | FiveThirtyEight (2014-2024) | Literatura (Jackman 2005; Linzer 2013; Stoetzer et al. 2019; Montalvo et al. 2019) | Este modelo |
|---|---|---|---|
| Peso por muestra | `sqrt(n/mediana)`, tope 10.000 winsorizado | varianza binomial `p(1−p)/n` en la verosimilitud | `sqrt(n/mediana_evento)`, tope 5.000 (efectivo 4.715) ✔ |
| Trackings / ráfagas | ventana 14 días, suma de pesos = 1 | — | solape → 0; `1/(N+1)` en ±4 días ✔ (ventana centrada, no normalizada) |
| Recencia / suavizado | EWMA + regresión local (grado 0-2 por AIC), mezcla adaptativa, 7 hiperparámetros optimizados contra histórico | paseo aleatorio latente (Kalman/Stan); incertidumbre crece con el horizonte | LOESS grado 1, ISJ adaptativo, sin optimización, foto congelada |
| House effects | regresión multinivel con shrinkage, se restan | δ por casa identificados anclando al resultado | **no** (señal disponible en `PollsResults.bias`) |
| Outliers | k-NN + kernel (mínimo 0,05) | — | no |
| Ratings | error relativo a la carrera → shrinkage `n/(n+n_shrink)` hacia prior de **transparencia documentado**; herding; bootstrap | — | desviación frente a media kernel → doble shrinkage con τ fijo → mezcla con `quality` manual ✔ estructura; sin herding, sin bootstrap, sin rúbrica |
| Ventana de evaluación | 21-31 días antes de la elección | — | ≈82 días efectivos (por decaimientos), sin documentar |
| Rating → peso | escala el tamaño muestral efectivo | — | `log1p(rating)/log1p(media)` (0-3) |
| Incertidumbre del promedio | bandas para predecir **encuestas futuras** | posterior del estado latente | IC de la media suavizada (HAC), ≈35 % estrecho |
| Deriva hasta la elección | término explícito ≈6 pts a 75 días | varianza del paseo aleatorio × horizonte; Fisher et al.: descuento hacia el resultado previo | solo la covariable `weeks` de `v2err`, fijada en 1 |
| Colas y correlaciones | t(5) + matriz estado×estado (demografía + región) | log-ratios con Σ entre partidos (Stoetzer); jerarquía provincia/CCAA/nación (Montalvo) | t(nobs−2) independiente por partido; provincias con correlación 1 |
| Proyección territorial | — | swing jerárquico (Montalvo 2019; Pavía 2016) | swing proporcional + `smap` ad hoc |
| D'Hondt | — | D'Hondt con barrera 3 % (Montalvo 2019; Electomanía) | D'Hondt **sin** barrera |
| Fundamentals | 11 indicadores + incumbencia, stacking | prior estructural (Time-for-Change; MAL 2012 para España) | no |
| Validación | predicción de encuestas futuras; calibración por bootstrap | RMSE y cobertura out-of-sample publicados (Economist, zweitstimme) | ninguna sistemática |

Modelos españoles: El País (Llaneras/Andrino) sigue la misma arquitectura (promedio → provincia → ruido calibrado con errores históricos → 15.000 simulaciones → D'Hondt); Electomanía aplica D'Hondt con barrera y mínimo por provincia; Montalvo, Papaspiliopoulos & Stumpf-Fétizon (2019) es la referencia académica más cercana al Simulator (jerárquico provincia/CCAA/nación, house effects, partidos nuevos, validado en 2015).

---

## 6. Mejoras metodológicas priorizadas

Orden por (impacto en lo que se publica) × (esfuerzo):

1. **Composición y correlación en la simulación** (M2). Muestrear en el espacio de log-ratios (ALR/ILR) con una normal multivariante cuya Σ se estime de los errores históricos conjuntos por partido de las 11 elecciones featured (el Computer ya tiene `PollsResults.error` por partido y encuesta) o, más simple, una Dirichlet con concentración calibrada. Garantiza Σ = 100, correlación negativa entre partidos y positiva dentro de bloques. *Validación*: cobertura de los intervalos de escaños por bloque en el backtest.
2. **Barrera del 3 % y base de votos válidos** (M1). Conservar "otros" y blancos en `umat`, calcular `pct` sobre válidos, filtrar `< 3` antes de `alloc_dhondt`. Test unitario contra Madrid/Barcelona 2023 con los votos oficiales (debe reproducir exactamente los 37 y 32 escaños).
3. **`totals()` y resúmenes** (B2). Publicar media y cuantiles de `dist()` (en DH/MT ya suman 350 por simulación), probabilidades de mayoría absoluta y de bloques, y la simulación mediana por provincia (`result_median()`); documentar LS/MC como línea base didáctica o sustituir `v2seats` por un modelo monótono con restricción de suma (M3).
4. **Ruido provincial** (M4). `ε_{r,p}` con varianza estimada de los residuos históricos del swing proporcional (la BD tiene 14 elecciones por provincia); para regionales, modelar el error sobre su % en su comunidad.
5. **House effects** (M5). Estimar δ por casa en la escala LOR con shrinkage (o directamente el `error_blocks` medio con signo, ya calculado) y restarlo antes del suavizado, como hacen 538 y Jackman. Validar que reduce el MAE del promedio a 6 días en 2015-2023.
6. **Incertidumbre del promedio y del horizonte** (M6, M9, M10). (a) `cov_type='cluster'` por `pollster_id` en la WLS local; (b) sustituir la foto congelada por un término de deriva `Var_h = err² + h·σ_ε²` con `σ_ε²` estimada de la varianza histórica de incrementos del promedio, o implementar un **filtro de Kalman + suavizador RTS a mano en `stat.py`** (encaja con la vocación divulgativa y resuelve house effects, deriva y ancho de banda a la vez, manteniendo el LOESS como método explicativo); (c) `weeks` real = `(event_date − date_last)//7`.
7. **Ancho de banda** (M7, M8, B1). Corregir el truncado; calcular la media geométrica del piloto sobre la muestra (Abramson clásico) o documentar el estimador *balloon*; elegir `h` por validación cruzada leave-one-out sobre las encuestas (o AICc) y comparar con ISJ en los 11 eventos.
8. **Ratings** (M11-M13). Un modelo jerárquico `dev_k ~ N(θ_casa, σ² + SE_t²)`, `θ_casa ~ N(0, τ²)` con σ, τ por Bayes empírico; una sola contracción; escala del rating con parámetro propio; rúbrica escrita para `quality` (transparencia, método, tamaño, historial) al estilo del *Transparency Score* de 538; ventana de evaluación documentada (o fija: 6-42 días).
9. **Error del promedio, no de la encuesta** (M13). Calibrar `std_err` con el error histórico **del propio Forecaster** a `d` días (backtest), no con el de encuestas individuales; ajustar sobre `e²` o `log|e|`, o escalar por `√(π/2)`.
10. **Fundamentals y descuento** (Fisher et al. 2011): regresión histórica del resultado sobre el promedio a `d` días para descontar la ventaja actual hacia el resultado previo; a largo plazo un prior económico (Magalhães et al. 2012).

## 7. Plan de validación / backtesting (no existe hoy)

- **Protocolo**: para cada evento featured 2008-2023 (y 2019-04/2019-11 como casos con partidos nuevos), congelar `limit_date = event_date − d` con `d ∈ {6, 14, 30, 60, 90, 180}`, ratings calculados solo con eventos anteriores (ya garantizado), ajustar Forecaster y Simulator con semilla fija y comparar con el resultado.
- **Métricas**: MAE y RMSE por partido y por bloque (%), cobertura empírica de los IC 50/80/95 %, log score o CRPS de la distribución de escaños por partido, error absoluto de escaños por partido y bloque, probabilidad asignada a la mayoría de bloque ganadora, Brier. Comparar contra tres líneas base: última encuesta, media simple de las últimas 4 semanas, resultado anterior.
- **Artefactos**: `backtest/` con un script CLI que produzca `metrics.csv` por evento y horizonte, figuras de calibración (bullseye ya existente + reliability diagrams) y un capítulo del libro. Publicar las cifras en el README como hace The Economist.

---

## 8. Hoja de ruta hacia un repositorio científico y divulgativo

### 8.1 Diagnóstico para un tercero

Hoy un tercero **no puede** ejecutar nada: necesita PostgreSQL con esquema y datos que no existen en el repo, `files/`, `.env`, crear `log/`, instalar 42 paquetes con pines desfasados (uno de ellos, `xgboost`, ni siquiera listado) y aun así `PollsForecast.ipynb` falla por `np.NaN`. Tampoco puede leer los notebooks en GitHub (Git LFS). En cambio, la lógica científica está bien localizada (≈6.300 líneas) y depende de una interfaz mínima del framework, lo que hace la extracción viable sin reescribir Computer/Forecaster/Simulator.

### 8.2 Arquitectura objetivo

```
<paquete>/                      # nombre propio, p. ej. "sondeos" o "esforecast" (mtpy es el framework)
├── pyproject.toml              # PEP 621; requires-python >= 3.11; extras [db], [ingest], [notebooks], [dev]
├── README.md · LICENSE (MIT) · CITATION.cff (completo) · CHANGELOG.md · CONTRIBUTING.md
├── src/<paquete>/
│   ├── stats/                  # stat.py troceado: descriptive.py (Stat), kernels.py (Kernel + bw), wls.py, local.py, kde.py
│   ├── ratings.py              # Computer: errores, desviaciones, ratings (sin gráficos)
│   ├── weights.py              # compute_weights, poll_overweights
│   ├── aggregate.py            # Forecaster (regresión local; futuro: state-space)
│   ├── seats.py                # D'Hondt con barrera, swing, smap, estimadores
│   ├── simulate.py             # Simulator
│   ├── blocks.py               # lib/utils.py
│   ├── data/                   # schema.py (meta), repository.py (getters), backends/{parquet,sqlite,postgres}.py
│   ├── ingest/                 # loader.py (Wikipedia con snapshot HTML/oldid; Infoelectoral)
│   ├── viz/                    # subconjunto de dataviz (≈33 funciones), sin `learning`
│   └── _compat/                # helpers.py, dates.py (subconjuntos)
├── data/                       # params.json, es-provinces.csv, wikipedia/*.json, infoelectoral/*.json + CSV/Parquet de las tablas
├── docs/                       # Jupyter Book o Quarto (ver 8.5)
├── notebooks/                  # 01_load_results … 06_simulate, 07_lab_*, parametrizados (papermill), sin outputs (nbstripout)
├── backtest/                   # run_backtest.py, metrics.csv, figuras
├── tests/
└── .github/workflows/ci.yml    # ruff + pytest + nbmake, matriz 3.11/3.12, numpy 1.26 y 2.x
```

**Keep / extract / drop** (de `mtpy`): mantener `lib/*`, `models/elections.py`, `stat.py`; extraer subconjuntos de `app.py` (→ `settings` + `logging`), `io.py` (→ `pathlib`), `core/data.py` (solo `Model.get_results/get_agg/get_var/format_data/upsert` + `build_select_query`), `helpers.py` (38 de 61 funciones), `dates.py` (10 de 22), `dataviz.py` (33 de 87, sin `learning`); descartar `services/*`, `dal/{BigQuery,DynamoDB,Mongo,MySQL,Redshift,Salesforce}`, `api.py`, `worker.py`, `job.py`, `pipeline.py`, `jobs/`, `pipelines/`, `controllers/`, `learners/`, `models/sources`, `strings.py`, `learning.py`, `nlp.py`, `Dockerfile`, `deploy/`.

**Desacoplar de PostgreSQL**: exportar `elections.*` a CSV/Parquet (`polls` 3.825 filas, `polls_results` 32.876, `events_results` 55.862) con un `datapackage.json` (Frictionless Table Schema generado desde `Model.meta`); backend de ficheros por defecto en `lib/data.py`; PostgreSQL como extra opcional para el mantenimiento. Separar columnas crudas y derivadas (`polls_weights`, `polls_errors`).

### 8.3 Datos

- **Licencias**: Wikipedia CC BY-SA 4.0 (citar página y `oldid`; atribuir casa y medio); Infoelectoral reutilizable con atribución ("Origen de los datos: Ministerio del Interior") según Ley 37/2007; curación propia (quality, bloques, mapas) CC BY 4.0; `Electomania-Rankings.xlsx` no redistribuir.
- **Procedencia**: `source_url`, `source_revision`, `retrieved_at` en `polls.csv`; guardar el HTML crudo en `data/raw/wikipedia/<evento>/<oldid>.html` y permitir `read_data(from_file=…)`.
- **Publicación**: release en GitHub sincronizada con Zenodo (concept DOI + DOI por versión); `CITATION.cff` con `version`, `license`, `repository-code`, `doi`, ORCID; README de datos con diccionario de columnas y changelog.
- **Mantenimiento**: CLI `python -m <paquete>.ingest wikipedia --event 2027-08-22 --snapshot` y `… infoelectoral --file PROV_02_202307_1.xlsx`; escritura en BD solo con `--write` explícito.

### 8.4 Calidad

- **Tests** (`pytest`): `stats/` contra oráculos (`DescrStatsW`, `np.quantile(method='averaged_inverted_cdf')`, `sm.WLS(cov_type='HAC')`, `KDEpy.improved_sheather_jones`, `statsmodels KernelReg(ll)`, `scipy.gaussian_kde`); propiedades (invariancia a la escala de pesos, NaN, enteros, ventana vacía → NaN); `alloc_dhondt` contra los 52 repartos oficiales de 2023 (con barrera); `totals()` nunca asigna a partidos con máximo 0; `build_blocks`/`group_results`; round-trip `save_forecast`/`load_forecast`; snapshots numéricos de `Forecaster.fit` con datos sintéticos; smoke tests de punta a punta con `n_sim ≤ 20` (los scripts `step1-5.py` de la prueba de ejecución son convertibles directamente).
- **Tooling**: `ruff` (lint + formato), `pre-commit` (ruff, nbstripout), `mypy` opcional, GitHub Actions, `uv`/`pip-tools` para lock, cobertura.
- **Higiene inmediata**: quitar el filtro global de warnings; `np.nan`; `.ffill()`, `.map`, `.iloc`; actualizar pines a numpy ≥ 2, pandas 2.2, scipy 1.17; `.env.example`; rotar y sacar del árbol las credenciales de `deploy/*.env`; salir de Git LFS para `.ipynb`.

### 8.5 Documentación y divulgación

- **Jupyter Book 2 (MyST) o Quarto** con capítulos: (1) el problema y las fuentes; (2) pesos y ratings (con las fórmulas de §2.1 y un capítulo sobre "por qué log-odds"); (3) el promedio: kernels, ISJ, regresión local, bandas; (4) de votos a escaños: swing, D'Hondt, barrera; (5) simulación y colas; (6) backtesting y calibración; (7) glosario y comparación con 538. Cada capítulo = un notebook parametrizado ejecutado en CI (`nbmake`/`papermill`) sobre los CSV versionados; Binder/Colab posibles una vez sin PostgreSQL.
- `stats/` como subpaquete didáctico con docstrings estilo NumPy, referencias (Botev 2010; Fan & Gijbels 1996; Newey-West 1987; Kish 1965) y un notebook "la librería estadística explicada" que compare cada función con su equivalente de scipy/statsmodels.
- README con el diagrama de §1, resultados del backtest, cómo citar, badge de DOI. Esbozo de *paper* JOSS/pyOpenSci (statement of need, state of the field, diseño, validación).

### 8.6 Plan por fases (un solo autor)

| Fase | Entregables | Esfuerzo |
|---|---|---|
| 0. Higiene y fixes verificados | B1-B7, B10-B15, B20-B26; `np.nan`; pines; warnings; `.env.example`; `log/` creado por código; `path` relativo documentado; borrar `events/*` y `build/`; salir de LFS | 2-3 días |
| 1. Extracción del paquete | `pyproject.toml`, layout `src/`, subconjuntos de `core`, sin `learning`/`services`; notebooks con `pip install -e .`; separar gráficos de `Computer` | 1-2 semanas |
| 2. Datos y tests | export CSV/Parquet + `datapackage.json`, backend de ficheros, snapshot Wikipedia, DDL, suite de tests + CI, smoke tests | 1-2 semanas |
| 3. Método y backtest | M1-M4 (barrera, composición/correlación, `totals`, ruido provincial), M6/M9/M10, script de backtest y métricas | 2-4 semanas |
| 4. Docs y publicación | Jupyter Book/Quarto, capítulo de `stats/`, README, CITATION.cff completo, Zenodo DOI, release 1.0, esbozo JOSS | 1-2 semanas |
| 5. (opcional) Modelo de espacio de estados y house effects | Kalman/RTS en `stats/`, δ por casa, comparación con LOESS en el backtest | 2-4 semanas |

---

## 9. Reutilización para sondeos propios y trackings digitales

Piezas del modelo directamente aplicables: la estadística ponderada con tamaño efectivo de Kish (`Stat`), el suavizado por regresión local con pesos y bandas (`LocalKernelEstimator`) para series de tracking diario, el shrinkage normal-normal (`bayes_adjust`) para estimaciones con pocas observaciones (por segmento, por día), el sistema de pesos por solape y ráfagas (`poll_overweights`) para muestras que se acumulan, y el esquema "error histórico por horizonte" para intervalos honestos.

Lo que falta para muestras no probabilísticas (paneles online, redes): **post-estratificación / raking** (Deming-Stephan, con `neff` de Kish tras calibrar), **MRP** (regresión multinivel + post-estratificación sobre el censo, con `files/es-regions-ages.csv` como punto de partida), *design effects*, detección de fraude/duplicados, ajuste por recuerdo de voto (transferencias, lo que Alaminos 2023 describe para España) y un modelo de espacio de estados para el tracking (que también resuelve el problema de la deriva del §6.6). El subpaquete `stats/` es el lugar natural para añadir `raking.py` y `mrp.py`.

---

## 10. Preguntas para el autor (decisiones que cambian el diseño)

1. ¿Cuál de los cuatro modos es "el modelo" de cara a publicación? ¿LS/MC son solo referencias didácticas?
2. ¿La ausencia de la barrera del 3 % es deliberada? ¿Se quiere modelar "otros" y blancos por provincia?
3. ¿Se acepta publicar media/cuantiles y probabilidades en lugar de medianas forzadas a 350?
4. ¿Cómo se asigna `pollsters.quality`? ¿Deriva de `Electomania-Rankings.xlsx`? ¿Por qué el CIS tiene 10?
5. ¿Por qué no se restan los sesgos por casa (`error_blocks`, `PollsResults.bias`) antes de promediar? ¿Se descartó por identificabilidad o por diseño?
6. ¿Los decaimientos (0,5/0,7/0,9), `wspan=4`, `τ=0,01`, `min_polls=3` y `n_iter=3` están calibrados o son elecciones a priori? ¿Se sabía que el ISJ salía exactamente 71,0?
7. ¿El pronóstico es una foto de `última encuesta + max_fc` congelada (deriva delegada en `v2err`) o debería crecer con el horizonte?
8. ¿Se contempla un modelo de espacio de estados (Kalman/RTS a mano) manteniendo el LOESS como método explicativo?
9. ¿Qué parte de los datos se puede redistribuir (Wikipedia CC BY-SA, Infoelectoral con atribución)? ¿Licencia deseada para la curación propia?
10. ¿Se mantendrá PostgreSQL o basta un backend de ficheros con volcado versionado?
11. ¿Los notebooks de `events/*` son registro histórico de predicciones publicadas o pueden regenerarse?
12. ¿Nombre del paquete y versión mínima de Python (el código ya exige ≥3.10; `typing.Self` sugiere 3.11)?
13. ¿Las credenciales de `deploy/*.env` siguen activas? Conviene rotarlas y sacarlas del árbol.
14. ¿`ctype` (`wban`/`exit`) se quiere mantener con detección por color de fila? ¿Las encuestas del CIS con campo subcontratado se registran como CIS o como el subcontratista?

---

## Apéndice A. Referencias clave

- FiveThirtyEight: Morris (2024) "How 538's pollster ratings work"; "How (most of) our polling averages work"; "How 538's 2024 presidential election forecast works" (abcnews.com/538); `fivethirtyeight/data/pollster-ratings` (codebook); Silver Bulletin pollster ratings.
- Jackman (2005) AJPS 40(4); Linzer (2013) JASA 108(501); Heidemanns, Gelman & Morris (2020) HDSR 2(4) y repo `TheEconomist/us-potus-model`; Fisher, Ford, Jennings, Pickup & Wlezien (2011) Electoral Studies 30(2); Stoetzer et al. (2019) Political Analysis 27(2) y `zweitstimme-org/prediction-2025-replication`; Shirani-Mehr, Rothschild, Goel & Gelman (2018) JASA 113(522); Jennings & Wlezien (2018) Nature Human Behaviour 2.
- España: Montalvo, Papaspiliopoulos & Stumpf-Fétizon (2019) EJPE 59 (arXiv 1612.03073); Pavía, García-Cárceles & Badal (2016) Statistica Applicata; Pavía et al. (2021) Spanish Electoral Archive, Scientific Data; Magalhães, Aguiar-Conraria & Lewis-Beck (2012) IJF 28; Alaminos & Alaminos-Fernández (2023) *Métodos y Modelos para la Predicción Electoral* (RUA); LOREG arts. 162-163; Electomanía "Metodología".
- Estándares: JOSS review checklist; pyOpenSci Python Package Guide; Citation File Format + Zenodo-GitHub; FAIR (Wilkinson 2016) y FAIR4RS (Chue Hong 2022); Sandve et al. (2013) *Ten Simple Rules*; Jupyter Book 2/MyST; Quarto; Binder; nbval/nbmake/papermill; DVC.

## Apéndice B. Informes detallados

`docs/analisis-2026-09/`: `computer-weights.md`, `computer-ratings.md`, `computer-estimators.md`, `forecaster.md`, `simulator.md`, `stat.md`, `data.md`, `infra.md`, `notebooks.md`, `research.md` (dosier con URLs), `smoke.md` (prueba de ejecución con tiempos y tablas). Lista completa de los 194 hallazgos brutos en `docs/analisis-2026-09/findings.json`.
