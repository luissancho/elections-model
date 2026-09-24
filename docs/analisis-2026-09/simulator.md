# Simulator: Monte Carlo, reparto provincial y D'Hondt

Módulo analizado: `mtpy/lib/simulator.py` (668 líneas) y sus dependencias directas: `Computer.get_seats_estimator` / `get_error_estimator` (`mtpy/lib/computer.py:1510-1679`), `Forecaster.fit_forecast` / `fit` (`mtpy/lib/forecaster.py:313-403`), `mtpy/lib/data.py` (`get_event_data:142`, `get_event_results:169`, `get_event_dates:463`), `LeastSquaresEstimator` y `LocalKernelEstimator` (`mtpy/core/utils/stat.py:661-856, 859-980`) y `Stat.median/quantile` (`stat.py:149-172`).

Todas las afirmaciones empíricas provienen de ejecuciones de solo lectura contra la base de datos local (evento `es/2027-08-22`, `drange=6`, `seed=42`, `max_fc=3`, `fillna=True`, `n_sim` ≤ 30) y del evento `es/2019-04-28` para comprobar las reglas `smap` con `regions`.

---

## 1. Visión general (overview)

### 1.1 Propósito

`Simulator` es la tercera pieza del modelo. Recibe el promedio ponderado de sondeos (`Forecaster`) en una fecha de corte, le añade una perturbación aleatoria cuyo tamaño procede del error histórico de las encuestas (`Computer.get_error_estimator`) y traduce el porcentaje nacional de cada partido a escaños por dos vías alternativas:

* **Regresión (`split=False`)**: un ajuste lineal `escaños ~ pct + regional` entrenado con todas las elecciones generales desde 1982 (`Computer.get_seats_estimator`).
* **Reparto provincial (`split=True`)**: un *swing proporcional uniforme* que proyecta el porcentaje nacional a cada una de las 52 provincias a partir de los resultados de la elección anterior, seguido del método D'Hondt en cada provincia.

Combinado con el interruptor `random`, esto produce las cuatro "modalidades" que usa el notebook `notebooks/PollsSimulations.ipynb`:

| Modo | `split` | `random` | `n_sim` | Qué hace realmente |
|---|---|---|---|---|
| **LS** | False | False | 1 | Escaños = regresión lineal aplicada al promedio de sondeos. Determinista. |
| **MC** | False | True | 1000 | Ruido t-Student independiente por partido sobre el promedio; regresión lineal a escaños en cada simulación. |
| **DH** | True | False | 1 | Swing proporcional por provincia + D'Hondt sobre el promedio de sondeos. Determinista. |
| **MT** | True | True | 1000 | Ruido t-Student por partido + swing proporcional + D'Hondt en cada simulación. Es el único modo que respeta simultáneamente la aleatoriedad y la restricción de 350 escaños por simulación. |

### 1.2 Flujo de datos y estructuras

El constructor (`simulator.py:27-200`) construye y encadena todo:

1. Lee `files/params.json[scope][event_date]` (`:98-100`); de ahí toma `names = parties.event` (los 13 partidos del evento 2027: PP, PSOE, VOX, SUMAR, UP, SALF, JxCat, ERC, EHB, PNV, BNG, CC, UPN) y `smap` (el "mapa de fuentes" que explica cómo derivar el resultado previo de partidos nuevos o fusionados).
2. `limit_date = event_date − drange[0]` días (`:104-109`); con `drange=6` la fecha de corte es 6 días antes de la elección (2027-08-16). Todo el forecast se lee en esa fecha.
3. Instancia `Forecaster(bmap=self.names, drange, drop_mtypes=['aggr','online'], alpha=0.05)` y llama a `build_series()` (`:118-128`). El Simulator **no** usa bloques: pasa la lista de partidos como `bmap`, de modo que cada partido es su propio bloque.
4. Instancia `Computer` con todas las elecciones entre 1980-01-01 y `event_date` excluyendo la última (`skip=1`, `:130-143`), es decir, las 14 elecciones 1982-2023; de él obtiene dos estimadores por mínimos cuadrados ponderados:
   * `v2seats = computer.get_seats_estimator()` (`:155-158`).
   * `v2err = computer.get_error_estimator(drange=self.drange, n_last=self.n_last)` (`:163-169`), solo si `add_errors=True`.
5. `reg_totals` (`get_reg_totals`, `:234-246`): tabla `region → (votes, seats)` de `events_data` para el evento a simular; la fila con `region_id=0` (nacional, `region=NaN`) se renombra a `default_region = scope = 'es'`. Para 2027 `votes` es NaN (no se usa) y `seats` es 350 nacional y 1-37 por provincia (suma 350).
6. `regions = reg_totals.index` → `['es', 'Álava', 'Albacete', …, 'Melilla']` (53 entradas, nombres de provincia, no códigos).
7. `prev_results` (`get_prev_results`, `:248-291`): resultados por provincia de la elección anterior (2023-07-23), ver §1.4.
8. `set_params()` inicializa `params = {n_sim:1, split:False, random:False, names:<todos>, regions:['es']}` y `rng=None`.

Arrays de salida tras `run()` (`:636-662`):

* `frames`: `(n_sim, n_names, 10)` con columnas `cols_frame = ['mean','regional','err','nobs','error','pct','pct_err','std_err','rand','vpred']`, redondeado a 2 decimales.
* `units`: `(n_sim, n_regions, n_names, 2)` con `cols_unit = ['prev_pct','vpred_pct']`.
* `results`: `(n_sim, n_regions, n_names)` escaños enteros (redondeados).

Accesores: `frame(loc)`, `unit(loc, region)`, `result(loc)` devuelven DataFrames etiquetados; `dist()` devuelve la matriz `(n_sim × partidos)` de escaños nacionales; `totals()` la mediana por partido ajustada a 350.

### 1.3 `set_params` (`:202-232`)

Con `reset=True` (valor por defecto de `run`) parte de `default_params` y superpone los `kwargs`. Normaliza tipos; si `split=False` fuerza `regions=['es']` (solo nivel nacional), si `split=True` y no se especifican regiones usa las 53. **La semilla se aplica aquí**: si `random=True` crea `rng = np.random.default_rng(self.seed)` en cada llamada (`:227-230`). Consecuencias: (a) dos `run(random=True)` sucesivos con la misma semilla reproducen exactamente los mismos sorteos (comprobado: `np.array_equal` = True); (b) los modos MC y MT consumen la misma secuencia aleatoria, lo que hace sus resultados comparables sorteo a sorteo; (c) con `seed=None` (valor por defecto del constructor) nada es reproducible; el `np.random.seed(SEED)` del notebook **no** afecta a `default_rng`.

### 1.4 `get_prev_results` (`:248-291`)

* Fecha previa: `model.nfc_series.index[0]`, es decir, la primera fila "no sondeo" de la serie del Forecaster = la elección anterior (`date_start`).
* Lee `events_results` de esa fecha, descarta `party_id ≤ 0` (la fila "-" de otros), renombra `region=NaN` a `'es'`.
* Lista de columnas `names`: los partidos del evento presentes en los resultados previos, en el orden de `df['party'].unique()`, más las claves del `smap` y todos los partidos citados en sus reglas (`:263-275`). Así, en 2027, `SALF` y `UP` (ausentes de 2023) entran vía las claves `smap`.
* Agrega por `(region, party)` → `votes, pct, seats`, pivota a un `MultiIndex` de columnas `(métrica, partido)`, rellena con 0 los pares ausentes y convierte todo lo `≤ 0` a `NaN` (`:277-289`). Resultado: `prev_results` de forma `53 × (3·len(names))` indexado por nombre de región.
* Comprobado en la BD: `pct` en `events_results` está expresado sobre **votos válidos** (candidaturas + blanco): en 2023 la suma de `pct` por provincia es ≈ 99.2 y coincide con `Σ votos_partido / events_data.votes`. Es la base que la LOREG usa para la barrera del 3 %.

### 1.5 `fit_forecast` / `build_forecast` (`:293-336`)

`fit_forecast(**kwargs)` delega en `Forecaster.fit_forecast(names, max_fc, fillna)` y luego llama a `build_forecast()`, que construye una tabla `partido × ['mean','regional','err','nobs','error']` leyendo **una sola fecha**, `limit_date`:

* `mean = model.forecast[n].loc[limit_date]`: la estimación de la regresión local (kernel gaussiano, ancho de banda adaptativo ISJ, polinomio local de grado 1, covarianza HAC) en la fecha de corte.
* `err = fc_stat[n].loc[limit_date]['err']`: el error estándar de la media local, `perr = sqrt(x' Σ_HAC x)` (`stat.py:830`).
* `nobs = fc_stat[n].loc[limit_date]['nobs']`: número de sondeos con peso kernel ≥ 0.01 en ese punto (`stat.py:912-913, 962`). En la ejecución de prueba, `nobs` ≈ 73-104.
* `regional`: bandera 0/1 de la tabla `parties`.
* `error = v2err.predict([mean, regional, weeks])` con `weeks = (drange[0]+1)//7` (`:311`), es decir, **1 semana** con `drange=6`, independientemente de cuánto tiempo haya pasado desde el último sondeo real (en la prueba, el último sondeo es de 2026-09-11 y el evento 2027-08-22: 49 semanas; con `fillna=True` el forecast se arrastra hacia delante hasta la fecha de corte).

Modelo de error histórico (`computer.py:1510-1606`): para cada elección pasada "featured", cada encuestadora aporta su último sondeo publicado a ≥ `drange[0]` días (`filter_polls`, `n_last=1`) y, para cada partido, el par `(pct del sondeo, |error| = |pct sondeo − pct real|)`. Se ajusta por WLS con pesos `weight_over·weight_sample`:

`E|error| = β0 + β1·pct + β2·regional + β3·weeks`

Coeficientes obtenidos en la prueba (1634 observaciones, R² = 0.39): `β = (1.259, 0.0432, −1.194, 0.0131)`. Un partido nacional con el 32 % tiene un error esperado de ≈ 2.65 puntos; uno regional con el 1 % ≈ 0.12 puntos.

### 1.6 `build_frame` (`:496-545`) — cómo se genera la aleatoriedad

Parte de `forecast.loc[names]`, copia `pct = mean` y define `vind = pct > 0` (partidos con estimación positiva). Si `random=False`, `vpred = pct` y termina. Si `random=True`:

1. `pct_err = v2err.predict([pct, regional, weeks])` (idéntico a la columna `error` ya calculada en `build_forecast`; se recalcula en cada simulación).
2. `std_err = sqrt(err² + pct_err²)`: combinación en cuadratura del error estadístico de la regresión local y del error histórico esperado.
3. `rand = clip( pct + std_err · T_{nobs−2}, 0, ∞ )` (`:536-539`), donde `T_ν` es una t-Student estándar con `ν = nobs − 2` grados de libertad, **muestreada de forma independiente para cada partido** (`rng.standard_t` recibe un vector de `df`, un sorteo por partido).
4. `vpred = rand`.

Propiedades que se deducen del código y se verificaron:

* La distribución es **t-Student por partido, independiente entre partidos, truncada solo por abajo en 0** (masa puntual en 0 para partidos pequeños: SALF 2.07 ± 1.36 tiene ≈ 6 % de probabilidad de caer a 0).
* No hay restricción de suma: `Σ vpred` en 30 simulaciones tuvo media 94.6 y desviación típica 4.3 (rango 84-102). El bloque `'-'` (resto hasta 100) que el Forecaster calcula **no se usa en ningún punto del Simulator**.
* Los grados de libertad `nobs − 2` son el número de sondeos del entorno local; con `nobs` ≈ 100 la t es prácticamente normal. Si `nobs ≤ 2` numpy lanza `ValueError df <= 0`; si `nobs` es NaN el sorteo es NaN; con `nobs = 3` la t es una Cauchy (varianza infinita).
* `std_err` mezcla un error estándar de la media (0.2-0.7 puntos) con un error absoluto esperado (1.4-2.7 puntos); el segundo domina.

### 1.7 `build_umat` (`:547-596`) — de porcentaje nacional a provincias

`prev_pcts` = `prev_results['pct'].fillna(0)` (53 regiones × partidos). Primero aplica las reglas `smap` en orden, cada una de tipo:

* **`agg`**: `prev[key] = prev[key] + Σ prev[names]` (el partido `key` hereda el voto previo de `names`; por ejemplo, en 2023 `PP` heredó el de `Cs`). Las columnas `names` no se eliminan, solo se descartan más tarde si no están en `params['names']`.
* **`sub`** (partido nuevo que sale de otro): con `fc_k = vpred[key]`, `fc_v = Σ vpred[names]`, `prev_v = Σ prev[names]` (por región):
  `prev[key] = round( prev_v / (fc_v/fc_k + 1), 2 ) = prev_v · fc_k / (fc_k + fc_v)`
  y después `prev[names] −= prev[key] / len(names)`. Es decir, el voto previo del partido origen se reparte entre origen y escisión **en proporción a sus estimaciones actuales** (que, en modo aleatorio, cambian en cada simulación). Como consecuencia, el factor de swing (ver abajo) del origen y de la escisión coincide y es igual al swing conjunto del bloque: `fmul = (fc_k + fc_v)/prev_v(es)`. Es una decisión de diseño razonable y coherente.
* **`split`**: `prev[name] += prev[key] / len(names)` para cada `name` (el voto previo de una marca desaparecida, p. ej. `NA+` → PP y UPN, se reparte a partes iguales).
* **`regions`** (opcional): mantiene `prev[key]` solo en las regiones listadas (o excluye las de `exclude`), siempre conservando `'es'` (`:572-578`). La comparación es `r in regions` sobre el índice de `prev_pcts`, que contiene **nombres** de provincia.

Después: `prev_pcts = prev_pcts.where(>0, NaN)[names]` (los ceros pasan a NaN, lo que más tarde se traduce en 0 votos) y el swing proporcional:

* `total_votes[r] = Σ_partidos prev_results['votes'][r]` (solo partidos incluidos, **no** el total de votos válidos).
* `prev_votes = prev_pcts · total_votes / 100` (redondeado); `n_votes = total_votes['es']`.
* `fc_votes[p] = round(vpred[p] · n_votes / 100)`.
* `fmul[p] = fc_votes[p] / prev_votes['es'][p]` ≈ `vpred[p] / prev_pct_es[p]`: **ratio nacional previsto/previo**.
* `vpred_pct[r, p] = prev_pct[r, p] · fmul[p]`.

Es un *proportional swing* (multiplicativo), no un *uniform swing* (aditivo): un partido que sube del 12.4 % al 17.4 % (VOX) multiplica por 1.41 su porcentaje en todas las provincias. Verificado: la media provincial ponderada por votos de `vpred_pct` reproduce `vpred` (PP 32.02 vs 31.91; ERC 2.18 vs 2.25; las diferencias vienen del redondeo a 2 decimales y de la base de votos). Un partido sin voto previo en una provincia (NaN) queda en NaN → 0 votos allí, lo que es correcto para partidos regionales pero fatal para partidos nuevos sin regla `smap` (ver hallazgos). La salida es un DataFrame `regiones × (['prev_pct','vpred_pct'] × partidos)` redondeado a 2 decimales.

### 1.8 `alloc_dhondt` (`:481-494`)

Método de las mayores medias: mientras queden escaños, asigna el siguiente al partido con mayor cociente `votes / (seats + 1)` y actualiza su cociente. Es una implementación correcta del D'Hondt puro. Empates: `max` sobre un `dict` devuelve la primera clave en orden de inserción (orden de `params['names']`), no el criterio LOREG de mayor número de votos (con votos iguales es indiferente). **No aplica la barrera del 3 % de votos válidos por circunscripción** (art. 163.1.a LOREG), ni distingue Ceuta/Melilla (mayoría simple con 1 escaño; D'Hondt con 1 escaño equivale, así que no importa).

### 1.9 `simulate` (`:598-631`)

* `split=True`: para cada provincia (salta `'es'`): `d_pcts = umat[r]['vpred_pct'].fillna(0)`, `d_adj = d_pcts·100/Σ d_pcts` (renormaliza a 100 **entre los partidos incluidos**, absorbiendo el voto a "otros"), `d_votes = round(d_adj · total_votes[r]/100)` y `alloc_dhondt(d_votes, seats[r])`. La fila `'es'` es la suma de provincias (`:625`) y por construcción suma 350 en cada simulación (verificado en MT: `dist().sum(axis=1)` = 350 siempre).
* `split=False`: `result['es'] = clip(v2seats.predict([vpred, regional]), 0)` (`:627-629`). El modelo de escaños (`computer.py:1608-1679`) es una regresión WLS `seats = γ0 + γ1·pct + γ2·regional` entrenada con los pares (partido, elección) de 1982-2023 con `pct ≥ 0.1` **y `seats > 0`**, pesos `0.97^(años desde la última elección)`. Coeficientes obtenidos: `γ = (−16.80, 4.48, 15.59)`, R² = 0.99, 161 observaciones. Predicciones: 0.5 % → −14.6 (recortado a 0), 5 % → 5.6, 12 % → 37, 33 % → 131; un regional con 1 % → −1.2 → 0 y con 2.25 % (ERC) → 8.9. El resultado por partido no está restringido a 350 (LS: suma bruta 326; MC: media 328, rango 291-358).

### 1.10 `run` (`:633-664`)

Reserva los tres arrays, aborta si `split=False` y no hay `v2seats`, y para `i in range(n_sim)`: `frame = build_frame()`, `umat = build_umat(frame)` (**siempre**, aunque `split=False`), reordena `umat` a `(regiones, partidos, 2)` (verificado que `unit()` coincide con `umat`), `result = simulate(frame, umat)` y guarda. Al final redondea `frames`/`units` a 2 decimales y `results` a entero. Rendimiento: 20 simulaciones MT en 0.5 s; 1000 ≈ 25 s.

### 1.11 `dist` y `totals` (`:381-442`)

`dist()` devuelve `(n_sim × partidos)` escaños nacionales (en `split=True` suma las provincias, que equivale a la fila `'es'`). `totals()` calcula la **mediana** por partido (`Stat.median` = cuantil ponderado 0.5; con `n_sim` par promedia los dos centrales) y luego fuerza que sumen `reg_totals['es'].seats = 350`:

`ds = floor(medians)`, `diff = 350 − Σ ds`, `rem = medians − ds`; si `diff > 0` reparte `diff // n_partidos` escaños **a todos los partidos** ("ciclos") y el resto de uno en uno por mayor resto; si `diff < 0` lo simétrico. Esto no es el método de restos mayores clásico (que solo reparte el resto fraccionario): con 13 partidos y `diff = 24` (modo LS) cada partido recibe +1 aunque su mediana sea 0. Resultado en la prueba LS: raw `UP=0, SALF=0, UPN=0` → `totals()` = `UP 2, SALF 1, UPN 1`; y en MC, `SALF 1, UP 1`.

`totals()` es lo que se publica en las gráficas (`plot_forecast_output`). No existe ningún método que devuelva intervalos de confianza como datos; los IC solo aparecen en `plot_dist_kde` (`:462-479`, parámetro `ci_alpha` de `dataviz.plot_kde_1d`), y el `alpha` del constructor solo se reenvía al Forecaster.

### 1.12 Supuestos estadísticos implícitos (resumen)

1. El error de predicción de cada partido es t-Student independiente, centrado en el promedio (sin sesgo), con escala `sqrt(err² + E|e|²)`.
2. La distribución geográfica del voto de un partido es la de la elección anterior escalada multiplicativamente (swing proporcional uniforme, sin componente provincial aleatoria, sin correlación negativa entre partidos, sin participación).
3. Los partidos no incluidos ("otros") desaparecen y su voto se reparte proporcionalmente entre los incluidos en cada provincia.
4. La relación escaños-porcentaje nacional (modos LS/MC) es lineal y estable en 40 años.
5. El forecast a `limit_date` equivale a "una semana antes" a efectos de error, cualquiera que sea la fecha del último sondeo.

### 1.13 Notas positivas de diseño

* La cadena Forecaster → error histórico → ruido → D'Hondt es transparente y cada paso está materializado en arrays inspeccionables (`frames`, `units`, `results`), lo que es ideal para un repositorio divulgativo.
* La regla `sub` del `smap` está bien pensada: reparte el voto previo del partido origen según las estimaciones actuales, lo que hace que origen y escisión compartan el mismo swing conjunto; en modo aleatorio ese reparto también es estocástico.
* El uso de la misma semilla para MC y MT permite comparaciones sorteo a sorteo.
* `alloc_dhondt` es una implementación mínima, legible y correcta del algoritmo de mayores medias.
* La combinación en cuadratura de error estadístico y error empírico es una forma sencilla y defendible de evitar la infra-cobertura típica de los IC puramente muestrales.

---

## 2. Pipeline

| # | Paso | Dónde | Qué hace | Matemática / parámetros |
|---|---|---|---|---|
| 1 | Cargar parámetros del evento | `simulator.py:98-116` | `params.json[scope][event_date]` → `names`, `smap` normalizado a listas de reglas | `limit_date = event_date − drange[0]` |
| 2 | Construir Forecaster | `:118-128`, `forecaster.py:221-311` | Series de sondeos + elección previa; cada partido es su bloque | `bmap=names`, `drop_mtypes=['aggr','online']`, kernel gaussiano, bw adaptativo ISJ, HAC(1) |
| 3 | Construir Computer y estimadores | `:130-169`, `computer.py:1571-1606, 1652-1679` | 14 elecciones 1982-2023; WLS de error y de escaños | `E|e| = β0+β1·pct+β2·reg+β3·weeks` (pesos `w_over·w_sample`); `seats = γ0+γ1·pct+γ2·reg` (pesos `0.97^años`, solo `seats>0`) |
| 4 | Totales por región | `:234-246` | `events_data` del evento → `seats` por provincia (350) | `region NaN → 'es'` |
| 5 | Resultados previos por provincia | `:248-291` | `events_results` de la elección anterior, pivotado `(votes,pct,seats) × partido` | `≤0 → NaN`; `pct` sobre votos válidos |
| 6 | Ajustar forecast | `:293-336`, `forecaster.py:362-403` | Regresión local por partido; lee `mean, err, nobs` en `limit_date`; `error = v2err(mean, reg, weeks)` | `weeks=(drange[0]+1)//7 = 1` |
| 7 | Frame por simulación | `:496-545` | Ruido t-Student independiente por partido | `vpred = max(0, pct + sqrt(err²+E|e|²)·T_{nobs−2})` |
| 8 | Matriz unitaria | `:547-596` | Reglas `smap` (`agg`,`sub`,`split`,`regions`) y swing proporcional | `vpred_pct[r,p] = prev_pct[r,p]·vpred[p]/prev_pct_es[p]` |
| 9 | Escaños | `:598-631` | `split`: renormalizar a 100 por provincia, votos = `pct·total_prev[r]`, D'Hondt; no `split`: `clip(v2seats(vpred, reg), 0)` | sin barrera 3 % |
| 10 | Bucle Monte Carlo | `:633-664` | `n_sim` iteraciones; arrays `frames`, `units`, `results` | `rng = default_rng(seed)` en `set_params` |
| 11 | Agregación | `:381-442` | `dist()` escaños nacionales por simulación; `totals()` mediana por partido forzada a 350 | heurístico de "ciclos" + restos |
| 12 | Salida gráfica | `:444-479` | `plot_forecast_output` (barras) y `plot_dist_kde` (KDE con mediana/IC) | IC solo en la gráfica |

---

## 3. Hallazgos

### F1 — `build_umat` falla con `KeyError` si `names` no incluye todos los partidos del `smap` (bug, alta)
* `simulator.py:550-561`: el bucle recorre todas las reglas `smap` y hace `frame.loc[key]` / `frame.loc[names]`, pero `frame` solo tiene las filas de `params['names']`.
* Reproducido: `sim.run(split=True, random=False, names=['PP','PSOE','VOX'])` → `KeyError: 'UP'`. Como `run()` llama a `build_umat` siempre (`:649`), también falla con `split=False`.
* Impacto: el parámetro `names` documentado en `default_params` es inutilizable con cualquier evento que tenga `smap`.
* Arreglo: saltar reglas cuyo `key` o `names` no estén en `params['names']` (o restringir `smap` a esos partidos al construir).

### F2 — La regla `regions` del `smap` compara códigos con nombres de provincia (bug, media)
* `simulator.py:572-578` evalúa `r in regions` sobre `prev_pcts.index`, que son nombres (`get_prev_results:261`, `get_reg_totals:244`; comprobado: `['es','Álava','Albacete',…]`). En `params.json` las reglas de 2019-04-28 usan códigos: `COMPROMIS: regions ['3','13','46']`, `NA+: regions ['34']`.
* Reproducido con `Simulator('es','2019-04-28')`: `umat['vpred_pct']['COMPROMIS']` y `['NA+']` solo son no nulos en `'es'`; en modo DH ambos obtienen 0 escaños (reales: 1 y 2).
* Arreglo: mapear códigos ↔ nombres (p. ej. con `files/es-provinces.csv`) o guardar `region_id` en `prev_results`.

### F3 — Partidos sin voto previo y sin regla `smap` aplicable reciben 0 escaños silenciosamente en DH/MT (bug/diseño, media)
* `simulator.py:580-588`: `prev_pct = NaN` → `fmul = fc_votes/NaN` → `vpred_pct = NaN` → `fillna(0)` en `:616` → 0 votos en todas las provincias, aunque el forecast nacional sea alto.
* Reproducido: en 2019-04-28 `JxCat` (forecast 1.31 %, regla `agg` con `PDeCAT`, pero en 2016 la marca era otra) → 0 escaños en DH (reales: 7). Ocurriría con cualquier partido nuevo sin `sub`.
* Arreglo: avisar/lanzar excepción cuando `prev_pct_es[p]` sea NaN para un partido con `vpred > 0`; ofrecer un reparto por defecto (p. ej. uniforme o proporcional al censo).

### F4 — No se aplica la barrera legal del 3 % provincial (stat, alta)
* `alloc_dhondt` (`:481-494`) y `simulate` (`:616-623`) reparten entre todos los partidos con votos. La LOREG (art. 163.1.a) excluye las candidaturas con < 3 % de votos válidos en la circunscripción; en la práctica solo afecta a Madrid y Barcelona, pero justo ahí se juegan 37 y 32 escaños.
* Además la base de D'Hondt es `d_adj` (renormalizado a 100 entre partidos incluidos, `:617`), no votos válidos: un partido con el 2.9 % de válidos aparece con > 3 % tras renormalizar. La BD sí tiene la base correcta (`pct` sobre válidos, `events_data.blank`).
* Test: `Simulator.alloc_dhondt({'A':60,'B':37,'C':2.9}, 37)` da 1 escaño a `C`; con la barrera no debería. En MT, contar simulaciones en que un partido con `vpred_pct < 3` en Madrid recibe escaño.
* Arreglo: conservar el bloque "otros" y el voto en blanco en `umat`, calcular `pct` sobre válidos y filtrar `< 3` antes de D'Hondt.

### F5 — `totals()` reparte escaños "por ciclos" a partidos con mediana 0 (stat, alta)
* `simulator.py:404-435`: si `diff > n_partidos`, `ds += diff // n` para **todos**. En LS la suma bruta fue 326 (`diff=24`, 13 partidos) → +1 a todos: `UP 0→2`, `SALF 0→1`, `UPN 0→1`. En MC (`n_sim=30`) `SALF 0→1`, `UP 0→1`.
* Además, forzar a 350 la suma de medianas marginales no tiene justificación probabilística: el vector de medianas no es un escenario. Estas cifras son las que se publican en los gráficos de barras.
* Test: `sim.run(split=False, random=False); sim.results.sum()` vs `sim.totals().sum()`.
* Arreglo: para DH/MT publicar la media (que sí suma 350 en esperanza) o la simulación más cercana a la mediana; para LS/MC, si se mantiene el ajuste, usar restos mayores puro (`diff < n`) y nunca dar escaños a partidos con predicción 0.

### F6 — El estimador de escaños (modos LS/MC) es lineal, entrenado solo con `seats>0` y no suma 350 (stat, media)
* `computer.py:1634-1636` filtra `seats > 0` (sesgo de selección: excluye PACMA, CUP…); `computer.py:1674-1679` ajusta `seats = −16.8 + 4.48·pct + 15.59·regional`. Predice negativos para nacionales < 3.75 % (recortado a 0 en `simulator.py:629`) y +15.6 de intercepto a todo regional.
* La suma de escaños por simulación no está restringida: MC `dist().sum(axis=1)` media 328, sd 17.5, rango 291-358 → los IC por partido de MC incluyen escenarios imposibles.
* Test: `sim.v2seats.predict(np.array([[0.5,0],[1,1],[33,0]]))` → `[-14.6, -1.2, 131]`.
* Sugerencia: documentar LS/MC como línea base didáctica; si se mantienen, ajustar con un término no lineal (`pct²` o log) e incluir los ceros; o renormalizar a 350 por simulación.

### F7 — Errores independientes por partido, sin restricción de suma ni correlación (stat, media)
* `simulator.py:536-539`: `rng.standard_t(nobs−2)` por partido; `'-'` no se usa. `Σ vpred` en 30 simulaciones: media 94.6, sd 4.3, rango 84-102.
* En DH/MT la renormalización provincial (`:617`) absorbe el exceso/defecto proporcionalmente (equivale a redistribuir "otros"), pero las correlaciones negativas entre partidos del mismo bloque (PP/VOX, PSOE/SUMAR), que son las que determinan las mayorías, no se modelan; en LS/MC no hay ninguna corrección.
* Test: correlación de `frames[:,:,vpred]` entre PP y VOX ≈ 0.
* Sugerencia: muestrear en el espacio de composiciones (Dirichlet, o normal multivariante sobre logits con Σ estimada de los errores históricos conjuntos, que el Computer ya calcula por bloques).

### F8 — Sin componente provincial aleatoria ni correlación regional (stat, media)
* `simulator.py:587-588`: el swing es determinista dado el sorteo nacional; toda la varianza de escaños proviene de 13 números. Las distribuciones provinciales son "demasiado estrechas" y no admiten sorpresas locales; para regionales, el modelo de error da `E|e|` ≈ 0.12-0.18 puntos (coeficiente `regional = −1.19`), así que ERC/EHB/PNV apenas varían (MT sd 1.5/0.7/1.3 escaños).
* Test: `sim.units[:, r, p, 1] / sim.frames[:, p, 9]` es constante entre simulaciones para toda `r`.
* Sugerencia: añadir ruido provincial `ε_{r,p}` con varianza estimada de los residuos históricos del swing proporcional (la BD tiene resultados provinciales de 14 elecciones), o modelar el error de los regionales sobre su porcentaje en su comunidad, no el nacional.

### F9 — `weeks` se fija en 1 aunque el último sondeo sea muy anterior (stat, media)
* `simulator.py:311, 509`: `weeks = (drange[0]+1)//7`. En la prueba, último sondeo 2026-09-11, evento 2027-08-22 (≈ 49 semanas), forecast arrastrado con `fillna`; el error se evalúa como a 1 semana (`β3 = 0.013/semana` → +0.6 puntos si se usaran 49). Es coherente si se interpreta como "si la elección fuera ya", pero no se documenta.
* Test: comparar `v2err.predict([[32,0,1]])` con `[[32,0,49]]`.
* Sugerencia: usar `weeks = (event_date − date_last)//7` o exponer un parámetro `as_of`.

### F10 — Escala del ruido: `E|e|` no es σ y se combina con un error estándar (stat, baja)
* `computer.py:1559` ajusta el **error absoluto** esperado; `simulator.py:530` lo usa como escala de una t. Si los errores fueran normales, `E|e| = 0.798·σ` (comprobado numéricamente), luego σ se infraestima ≈ 20 %. Además la regresión lineal puede devolver valores negativos para `pct` pequeños (se elevan al cuadrado sin comprobar el signo).
* Test: comparar la cobertura empírica de los IC del modelo con los errores reales de 2019/2023 (backtest con `Simulator('es','2023-07-23')`).
* Sugerencia: ajustar la regresión sobre `e²` (o `log|e|`) y usar `sqrt` como σ, o escalar por `sqrt(π/2)`.

### F11 — `standard_t(nobs−2)` puede fallar o degenerar (bug, media)
* `simulator.py:537`: `nobs ≤ 2` → `ValueError: df <= 0` (comprobado en numpy); `nobs = NaN` (fecha sin `fc_stat`) → `vpred = NaN`; `nobs = 3` → Cauchy. Ocurre con partidos poco sondeados o `drange` grandes. `nobs` es además un recuento de sondeos con peso ≥ 0.01, no la muestra efectiva (`neff` está disponible en `fc_stat`).
* Test: `Simulator(..., drange=(6,20))` con un partido regional y `random=True`, o inspeccionar `sim.forecast['nobs'].min()`.
* Sugerencia: `df = max(neff − 2, 3)` o una normal si `df` es grande, y `fillna` explícito.

### F12 — Reproducibilidad: `seed=None` por defecto y dependencia del estado de la BD (reproducibility, media)
* `simulator.py:39, 227-230`: sin semilla nada es reproducible; el `np.random.seed(42)` del notebook no afecta a `default_rng`. Los resultados dependen del contenido de la BD (sondeos cargados desde Wikipedia) y de `files/params.json`, que está en `.gitignore`, así que un tercero no puede reproducir ni el forecast ni la simulación.
* Test: dos `Simulator(seed=None).run(random=True)` dan resultados distintos; `git check-ignore files/params.json`.
* Sugerencia: exigir semilla, registrar en la salida (`sim.csv`) semilla, versión de datos y parámetros; versionar `params.json` y un volcado CSV de sondeos/resultados.

### F13 — `totals()` usa indexación posicional en una `Series` etiquetada (bug latente, baja)
* `simulator.py:421, 435`: `ds[i] += 1` con `i` entero de `argsort` → `FutureWarning: Series.__getitem__ treating keys as positions is deprecated` (comprobado en pandas 2.2.2); en pandas 3 será búsqueda por etiqueta → `KeyError`.
* Arreglo: `ds.iloc[i] += 1`. (En `forecaster.py:400-401, 472` hay deprecaciones análogas: `fillna(method='ffill')`, `applymap`.)

### F14 — Documentación desajustada / ausente (docs, baja)
* `frame()` (`:338-358`) dice devolver "the simulation results" pero devuelve el frame de entrada. `alpha` se documenta como "Confidence interval" (`:59-60`) pero el Simulator no calcula ningún IC ni expone cuantiles: los IC solo existen en `plot_dist_kde` (`:462-479`). `build_umat`, `simulate`, `run`, `dist`, `totals`, `alloc_dhondt` carecen de docstring. `get_reg_totals` promete "total votes" que para el evento futuro son NaN y no se usan.
* Sugerencia: añadir `summary(alpha)` que devuelva media, mediana y cuantiles de `dist()` y de `frames[...,'vpred']`.

### F15 — Cálculo redundante y código muerto (performance/dead-code, baja)
* `run()` construye `umat` en cada iteración también con `split=False` (`:649`); `pct_err` se recalcula en cada simulación y es idéntico a `forecast['error']` (`:325-330` vs `:525-529`); `reg_totals['votes']` y `prev_results['seats']` no se usan; `smap` del usuario se muta in place (`:114-116`); `load_forecast(prefix=None)` (`:304-308`) reajusta sin `max_fc`/`fillna` y deja `forecast.loc[limit_date]` en NaN.

### F16 — "Otros" desaparecen y la participación no se modela (design, baja)
* `simulator.py:582-585, 616-620`: `total_votes[r]` es la suma de votos de los partidos incluidos en la elección previa; el voto a partidos no incluidos y el blanco se redistribuye proporcionalmente por la renormalización. La participación solo actúa como escala (D'Hondt es invariante), así que cambios diferenciales de participación por provincia no tienen efecto. Aceptable como simplificación, pero conviene documentarlo y es un prerrequisito para F4.

### F17 — `get_event_dates(..., skip=1)` supone que `event_date` existe en `events` (design, baja)
* `simulator.py:130-135`: si se simula una fecha que no está en la tabla `events`, `skip=1` descartaría la elección anterior real del Computer. `get_prev_results` (`:257`) depende de que el Forecaster haya encontrado dos eventos.
* Test: `Simulator('es','2027-01-01')` (fecha inexistente) → `KeyError` en `params.json` antes, pero si se añadiera la clave, el Computer perdería 2023.

---

## 4. Preguntas abiertas para el autor

1. ¿La omisión de la barrera del 3 % es deliberada (simplificación) o un olvido? ¿Se quiere modelar el bloque "otros" y el voto en blanco para poder aplicarla?
2. ¿Cuál de los cuatro modos se considera "el modelo" de cara a publicación? ¿LS/MC son solo referencias didácticas?
3. ¿Las reglas `regions` del `smap` deben expresarse con códigos INE (como en 2019-04) o con nombres? ¿Debería `prev_results` indexarse por `region_id`?
4. ¿Por qué `df = nobs − 2` para la t-Student y no `neff`? ¿Se pretende que la t modele la incertidumbre de la media local o el error histórico?
5. ¿`drange=6` responde a la prohibición de publicar sondeos en los 5 días previos (art. 69.7 LOREG)? ¿Se quiere que `weeks` refleje la distancia real desde el último sondeo cuando se simula "hoy"?
6. ¿Se desea modelar correlación entre partidos (bloques) y ruido provincial? El Computer ya calcula errores por bloques (`error_blocks`), que podrían alimentar una Σ.
7. ¿El ajuste a 350 en `totals()` es un requisito de presentación? ¿Aceptaría publicar media/cuantiles en lugar de medianas forzadas?
8. ¿Por qué el estimador de escaños usa todas las elecciones desde 1982 (sistemas de partidos muy distintos) y excluye `seats = 0`?
9. `files/params.json` y los datos están fuera del repositorio: ¿qué parte de los datos (sondeos de Wikipedia, resultados de Infoelectoral) se puede redistribuir para que el modelo sea reproducible por terceros?
