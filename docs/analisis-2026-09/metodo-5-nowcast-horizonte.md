# Método 5: nowcast, deriva de la opinión y fecha incierta (24-09-2026)

Cambios: `Simulator` (`mtpy/lib/simulator.py`: ancla `as_of`, `horizon_max`, `durations`, params `horizon` y `date_prior` de `run()`, `build_horizons`, `horizon_candidates`, `horizon_sampler`, columna `drift` del frame, `fan()`, `TERMINAL_WEEKS`); `DriftEstimator` y `Computer.get_drift_data / compute_drift / load_drift / get_drift_estimator` (`mtpy/lib/computer.py`); `Forecaster.date_fit_last` (`mtpy/lib/forecaster.py`); tabla nueva `elections.drift` (modelo `Drift` en `mtpy/models/elections.py`, `get_drift` y `save_drift_data` en `mtpy/lib/data.py`, `Model.table_exists` en `mtpy/core/data.py`); backtest en dos modos (`mtpy/lib/backtest.py`, `backtest/run_backtest.py --nowcast-only`); notebooks `data-load/PollsCompute` (celda `compute_drift`) y `PollsSimulations` (sección "Pronóstico a fecha"). Tests: `tests/test_simulator_unit.py` (M5), `tests/test_backtest_unit.py`, `tests/integration/test_model.py` (M5), `tests/integration/test_backtest.py`. Lo motivó el backtest (`metodo-3-backtest.md`): coberturas del 50 % de 0,33-0,38 a 60-180 días.

## Problema: dos preguntas con un solo número

El `Simulator` respondía a "¿quién ganaría si las elecciones fueran ya?" con un único ruido por partido: el error histórico de las encuestas a una semana de la elección (`v2err` con `weeks = (drange[0] + 1) // 7 = 1`), sumado en cuadratura al error estadístico del promedio. El backtest muestra que ese ruido está bien calibrado para esa pregunta: a 6 días los intervalos del 95 % cubren el 96 % de los resultados y los del 50 % el 57 %.

Pero la salida se anclaba en `2027-08-22`, que no es la fecha de la elección sino el **límite de la legislatura**: `limit_date = 2027-08-22 − 6`, el promedio se arrastraba once meses con `fillna` hasta esa fecha y la tabla se presentaba como pronóstico del 22 de agosto de 2027 con la incertidumbre de la última semana de campaña. Faltaba todo lo que la opinión puede moverse hasta la fecha real, que además no se conoce: el Gobierno puede convocar en cualquier momento.

En el backtest ese mismo mecanismo hacía otra cosa: con `drange = d` días, `weeks = d // 7`, y el error de las encuestas publicadas `d` días antes incluye la deriva hasta la elección. La deriva estaba escondida dentro del error de encuesta, mezclada con él y sin forma de separarla del horizonte.

## Solución: separar el error de las encuestas de la deriva de la opinión

Tres cambios:

1. **Ancla `as_of`**. El pronóstico se lee en el último día realmente ajustado (`Forecaster.date_fit_last` = última encuesta + `max_fc`) o en `limit_date`, lo que ocurra antes; `event_date` pasa a ser `deadline` y `horizon_max = deadline − as_of` los días que quedan. Para 2027: `as_of = 2026-09-14`, `horizon_max = 342`. En el backtest, donde las encuestas llegan hasta `limit_date`, el ancla coincide con `limit_date` y nada cambia.
2. **Error terminal fijo**. `weeks` es la constante `TERMINAL_WEEKS = 1` en `build_forecast` y `build_frame`: `σ_enc` es siempre el error de las encuestas de la última semana, calibrado por el backtest a 6 días. La deriva ya no se cuela por ahí, así que no hay doble conteo al añadirla.
3. **Deriva explícita por horizonte**. En cada simulación el ruido de un partido con cuota `p` a `h` días es `sqrt(err² + pct_err² + σ_der(h, p)²)`, con `σ_der(h, p) = p · sqrt(k · h)`: un **paseo aleatorio relativo**. El `run()` admite `horizon=None` (nowcast, `h = 0`), un entero, `'deadline'` (`h = horizon_max`) o `'random'` (cada simulación sortea `h` de un prior sobre la fecha); `fan()` da la tabla analítica de intervalos por horizonte.

Con `horizon=None` y semilla 42 la simulación es idéntica a la anterior (los horizontes solo se sortean en modo `'random'`, una vez y antes de los demás sorteos): la regresión MT n=200 (PP 137, PSOE 112, VOX 62, SUMAR 8) no cambia.

## La deriva empírica

`Computer.get_drift_data` ajusta el promedio diario de cada partido encuestado en cada ciclo pasado (`Forecaster` con `drange=None`, `max_fc=0`) y mide los incrementos `μ(t+d) − μ(t)` sobre todos los días `t`: por ciclo, partido y horizonte guarda `n`, `rms`, `bias` y el nivel medio del partido (`level`). Once ciclos (1993-2023), 34 partidos, 860 filas, persistidas en `elections.drift` (`compute_drift(save=True)`, ≈ 70 s).

### En puntos, para los partidos principales (ciclos desde 2011)

| d (días) | 7 | 14 | 30 | 60 | 90 | 180 | 365 |
|---|---|---|---|---|---|---|---|
| RMS de `μ(t+d) − μ(t)` (pp) | 0,25 | 0,47 | 0,92 | 1,61 | 2,13 | 3,19 | 4,48 |

Por debajo de 60 días la varianza crece como `d²`: la serie ajustada es localmente lineal (artefacto del suavizado), no una propiedad de la opinión. De 60 días en adelante crece como un paseo aleatorio (`Var ∝ d`) y no se satura hasta un año. El error terminal del promedio (último día ajustado frente al resultado oficial) es 1,46 pp de RMSE: es lo que `v2err` ya captura.

### La deriva es proporcional al nivel del partido

Entre los partidos principales (10-40 %) la deriva en puntos parece homogénea, pero al incluir todos los partidos encuestados la elasticidad de la varianza respecto al nivel es 2,06 (regresión `log(rms²/d)` sobre `log(level)`, ciclos desde 2011): la deriva **en puntos es proporcional a la cuota**. Una regional al 1 % se mueve centésimas al mes; un partido al 30 % se mueve puntos. Aplicar a UPN (0,2 %) la deriva en puntos del PP daba intervalos de 0-9 % y le quitaba escaños al PNV o a EHB en las simulaciones a un año; el modelo relativo lo evita.

Deriva relativa (`rms / level`) a 90 días por tamaño, ciclos desde 2015:

| Partidos | n partido-ciclo | media geométrica | media cuadrática |
|---|---|---|---|
| pequeños y regionales (< 3 %) | 38 | 0,070 | 0,205 |
| medianos (3-10 %) | 9 | 0,131 | 0,239 |
| grandes (> 10 %) | 20 | 0,085 | 0,110 |

La media geométrica es parecida en los tres grupos (7-13 % de la cuota a 90 días); la cuadrática está dominada por los partidos nacidos o disueltos dentro del ciclo (VOX en 2019, de 2 a 10 %; Cs en 2015; UP en 2015; IU en 2016 al fundirse en UP; SUMAR y MP en 2023), que derivan varias veces más. `DriftEstimator` usa la media geométrica: es la deriva del partido establecido típico, y la salvedad de los partidos nuevos se declara.

### Depende de la época

| Ciclo | Encuestas | Partidos | `k` (×10⁻⁵) | relativa 30 d | 90 d | 180 d | 365 d |
|---|---|---|---|---|---|---|---|
| 1993 | 47 | 16 | 0,9 | 0,010 | 0,027 | 0,044 | 0,071 |
| 1996 | 66 | 12 | 1,5 | 0,015 | 0,034 | 0,057 | 0,091 |
| 2000 | 103 | 10 | 0,9 | 0,010 | 0,028 | 0,049 | 0,065 |
| 2004 | 123 | 8 | 0,8 | 0,010 | 0,026 | 0,043 | 0,060 |
| 2008 | 197 | 9 | 0,7 | 0,011 | 0,026 | 0,038 | 0,047 |
| 2011 | 306 | 10 | 2,9 | 0,023 | 0,049 | 0,078 | 0,116 |
| 2015 | 353 | 15 | 12,1 | 0,036 | 0,097 | 0,168 | 0,266 |
| 2016 | 111 | 11 | 4,6 | 0,034 | 0,062 | — | — |
| 2019-04 | 296 | 14 | 5,5 | 0,029 | 0,072 | 0,109 | 0,140 |
| 2019-11 | 79 | 15 | 9,4 | 0,054 | 0,091 | 0,156 | — |
| 2023 | 557 | 17 | 5,7 | 0,034 | 0,075 | 0,106 | 0,130 |

Los ciclos bipartidistas (hasta 2011) derivan de 3 a 10 veces menos que los del sistema fragmentado desde 2015 (y tienen pocas encuestas, lo que suaviza más las series). Dar el mismo peso a todos daría `k = 2,7·10⁻⁵`, una deriva que el sistema actual desmiente. En vez de fijar una fecha de corte, `get_drift_estimator` pondera cada ciclo con `year_decay = 0,9` por año de antigüedad, el mismo decaimiento que el `Computer` usa para los ratings de las encuestadoras: `k = 5,0·10⁻⁵`, es decir, una deriva relativa de `0,0071·√h`: 3,9 % de la cuota a 30 días, 6,7 % a 90, 9,5 % a 180 y 13,1 % a 342 (el límite de la legislatura). El ajuste usa solo los horizontes ≥ 60 días (`d_min`), por el artefacto de suavizado; en los cortos la ley `√(k·h)` queda por encima de la curva empírica (1,9 % frente a 0,8 % a 7 días): conservadora.

En el backtest cada elección usa solo los ciclos anteriores (`Computer` con `skip=1`): `k` fuera de muestra de 1,4·10⁻⁵ en 2015 (solo ciclos bipartidistas: el modelo de 2015 habría subestimado la deriva, como cualquier modelo de entonces), 4,0·10⁻⁵ en 2016 y 2019-04, 4,4·10⁻⁵ en 2019-11 y 4,5·10⁻⁵ en 2023.

## La fecha de la elección

Quince legislaturas cerradas desde 1977, con duraciones (días entre elecciones) de 624, 1337, 1333, 1225, 1316, 1001, 1470, 1463, 1456, 1351, 1491, 189, 1036, 196 y 1351; el máximo legal es 4 años más los 30 días de convocatoria (1491). Nueve de quince acabaron antes del 90 % del máximo, dos fueron repeticiones. La actual lleva 1149 días en `as_of` y quedan 342.

El prior `'historical'` de `date_prior` (`horizon_candidates`) toma las legislaturas pasadas que duraron al menos los días transcurridos y les resta esos días, acotando por `horizon_max`: diez candidatos equiprobables, 76, 167, 184, 188, 202, 202, 307, 314, 321 y 342 días (mediana ≈ 200: la historia dice que a estas alturas lo normal es que la elección llegue en unos seis meses, no al agotar la legislatura). `'uniform'` es U{0..342}; también se admite un array de días. El prior es una hipótesis declarada, no una predicción de la fecha: el Gobierno adelanta cuando las encuestas le favorecen, y esa dependencia entre fecha y resultado no se modela.

## Modos de uso

- `Simulator(..., as_of=None)`: ancla por defecto; `as_of` explícito lee la serie en otra fecha (con aviso si es anterior a la última encuesta: el kernel es bilateral, no es una congelación; para eso está `limit_date`).
- `run(horizon=None)`: **nowcast**, "si las elecciones fueran mañana". Es el titular.
- `run(horizon=h)` o `run(horizon='deadline')`: pronóstico a `h` días o al límite de la legislatura.
- `run(horizon='random', date_prior='historical')`: un solo número a fecha desconocida, con el prior declarado; `sim.horizons` guarda el horizonte de cada simulación.
- `fan(horizons=None, alpha=None, wide=False)`: intervalos analíticos de voto por horizonte, `mean ± t · sqrt(err² + pct_err² + k·h·mean²)`, el gemelo de la simulación antes del reparto de escaños.

## Efecto en 2027 (MT, n = 200, semilla 42, `max_fc = 3`)

| | nowcast | `deadline` (342 días) | `random` (prior histórico) |
|---|---|---|---|
| PP, voto (IC 95 %) | 26,6-36,8 | 22,2-40,8 | 22,6-42,1 |
| PP, escaños | 137 (115-157) | 137 (99-170) | 139 (100-171) |
| PSOE, escaños | 112 (94-134) | 112 (81-148) | 111 (78-144) |
| VOX, escaños | 62 (42-81) | 62 (31-95) | 61 (38-94) |
| PNV, escaños | 5 (3-6) | 5 (2,5-6) | 5 (3-6) |
| P(PP primero) | 0,92 | 0,78 | 0,85 |
| P(mayoría absoluta Derecha) | 0,975 | 0,90 | 0,955 |
| P(mayoría absoluta Izquierda) | 0,00 | 0,00 | 0,00 |

Las medianas no cambian (la deriva es simétrica); los intervalos de los grandes se ensanchan un 40-70 % a un año y los de las regionales apenas (la deriva relativa de una cuota del 1 % son centésimas). Desviación típica del abanico (`fan`, pp) por horizonte:

| Partido | 0 | 30 | 90 | 180 | 342 |
|---|---|---|---|---|---|
| PP | 2,74 | 3,00 | 3,47 | 4,08 | 4,99 |
| PSOE | 2,49 | 2,71 | 3,09 | 3,58 | 4,33 |
| VOX | 2,04 | 2,15 | 2,35 | 2,62 | 3,05 |
| SUMAR | 1,54 | 1,56 | 1,59 | 1,64 | 1,73 |
| ERC | 0,18 | 0,20 | 0,24 | 0,28 | 0,35 |
| PNV | 0,13 | 0,13 | 0,14 | 0,16 | 0,18 |

## Backtest: antes y después

Cinco elecciones (2015-2023), seis horizontes, 500 simulaciones por caso y modo (`python backtest/run_backtest.py`, ≈ 15 min). "Antes" es la ejecución guardada tras el método 4; "nowcast" la nueva con `weeks = 1`; `_h` la ejecución a horizonte con la deriva. Coberturas sobre los partidos principales:

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| Cobertura voto 50 %: antes | 0,57 | 0,50 | 0,44 | 0,38 | 0,33 | 0,37 |
| nowcast | 0,57 | 0,50 | 0,44 | 0,38 | 0,33 | 0,30 |
| `_h` | 0,57 | 0,50 | 0,44 | 0,38 | 0,38 | 0,37 |
| Cobertura voto 95 %: antes | 0,96 | 0,88 | 0,84 | 0,84 | 0,84 | 0,87 |
| nowcast | 0,96 | 0,88 | 0,84 | 0,84 | 0,84 | 0,87 |
| `_h` | 0,96 | 0,88 | 0,84 | 0,84 | 0,89 | 0,87 |
| Cobertura escaños 50 %: antes | 0,62 | 0,59 | 0,42 | 0,31 | 0,31 | 0,35 |
| `_h` | 0,62 | 0,59 | 0,42 | 0,37 | 0,37 | 0,35 |
| Cobertura escaños 95 %: antes | 1,00 | 1,00 | 0,96 | 0,92 | 0,92 | 0,80 |
| `_h` | 1,00 | 1,00 | 0,96 | 0,92 | 0,92 | 0,80 |
| CRPS escaños: antes | 4,46 | 5,69 | 6,33 | 8,28 | 8,93 | 9,37 |
| `_h` | 4,46 | 5,69 | 6,32 | 8,26 | 8,90 | 9,28 |

Tres lecturas:

1. **El nowcast nuevo coincide con el anterior** aunque `weeks` haya pasado de `d // 7` a 1: el coeficiente de `weeks` en el estimador de error es 0,012 pp por semana (3,17 pp de error para un partido al 30 % a una semana, 3,45 a 25 semanas). La covariable nunca capturó la deriva; el modelo siempre usó, de hecho, el error terminal a cualquier horizonte, y por eso sus intervalos eran demasiado estrechos lejos de la elección.
2. **La deriva mejora las coberturas a 90-180 días, pero poco** (95 %: 0,84 → 0,89 a 90 días; 50 %: 0,33 → 0,38 y 0,30 → 0,37), y el CRPS de escaños baja un 1 %. Para un partido al 25 % la deriva fuera de muestra (`k` = 4-4,5·10⁻⁵) añade 0,9 pp a 30 días, 1,3 a 60, 1,6 a 90 y 2,2 a 180, en cuadratura sobre un error terminal de unos 3,2 pp: los intervalos se ensanchan un 4 %, 8 %, 12 % y 23 %.
3. **El error que falta no es deriva del partido típico.** El RMSE del promedio frente al resultado crece de 1,94 pp (6 días) a 2,88 (30), 3,40 (60), 3,33 (90) y 3,62 (180): en cuadratura sobre el terminal, 2,1, 2,8, 2,7 y 3,1 pp, el doble de lo que da `k`. Dos causas, ambas visibles en `metrics.csv`: (a) **sesgos de las encuestas** presentes ya a 6 días (2015: cobertura del 95 % de 0,80 en todos los horizontes; 2019-11: 0,67 desde 60 días) que ningún término de horizonte corrige y que son el objeto del siguiente paso (efectos de casa, `pollsters_parties`); (b) **partidos nacidos en el ciclo** entre los principales (Cs y UP en 2015, VOX en 2019, MP en 2019-11, SUMAR en 2023), que derivan tres veces más que el partido establecido que representa la media geométrica. `DriftEstimator.fit(agg='quadratic')` da la media cuadrática (`k` ≈ 2,5 veces mayor, dominada por esos partidos) para quien prefiera calibrar la cobertura media a costa de ensanchar de más a los partidos establecidos; el multiplicador por edad del partido es la extensión natural.

Para 2027 la consecuencia práctica: el abanico y los modos a horizonte son una cota inferior honesta de la incertidumbre de PP, PSOE y VOX (establecidos), y una subestimación para SALF y para cualquier reconfiguración de SUMAR y UP.

## Salvedades

- **Error terminal supuesto igual "hoy" que en campaña**: `σ_enc` es el error de las encuestas de la última semana, la única calibración verificable. Las encuestas de campaña suelen ser algo mejores, así que el nowcast a un año vista es, si acaso, algo estrecho.
- **Partidos nuevos o en disolución** derivan varias veces más que el partido establecido típico que representa `k` (media geométrica). En 2027 afecta a SALF (nacido en 2024) y a la evolución de UP y SUMAR; sus intervalos a horizonte son demasiado estrechos. Una extensión natural: un multiplicador por edad del partido.
- **Fecha y resultado no son independientes**: el prior histórico es una hipótesis declarada.
- **Artefacto de suavizado** a menos de 60 días: la ley `√(k·h)` es conservadora ahí; el nowcast (h = 0) no se ve afectado.
- **`k` en muestra para los primeros ciclos del backtest** (2015 solo ve ciclos bipartidistas): es la evolución fuera de muestra honesta, no un fallo del estimador.
- **La tabla `elections.drift`** hay que recalcularla al añadir un ciclo (`compute_drift(save=True)` en `PollsCompute`); si falta, el `Simulator` la calcula al vuelo (≈ 70 s) con aviso.
