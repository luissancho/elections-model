# Método 15: calibración de la incertidumbre (01-11-2026)

Cambios:
- Diagnóstico de la calibración: cuantiles de 5 en 5 (`q05`…`q95`) en `shares.csv` y `seats.csv`, `calibration.csv` con el PIT de cada celda y `summarize_calibration` por población y grupo de horizontes (`mtpy/lib/backtest.py`).
- Tabla nueva `elections.average_errors`, que se crea al guardar: modelo `AverageErrors`, `get_average_errors` y `save_average_errors_data` (`mtpy/models/elections.py`, `mtpy/lib/data.py`); `Computer.get_average_error_data`, `compute_average_errors` y `load_average_errors`, y el paso del job `compute` (`mtpy/lib/computer.py`).
- `AverageErrorEstimator` y `Computer.get_average_error_estimator` (`mtpy/lib/computer.py`).
- `Simulator(error_model='polls' | 'average')`, `AVERAGE_ERROR_HORIZON`, `error_model_used`; `fit_error_estimator` y `error_events` con los dos modelos (`mtpy/lib/simulator.py`); `diagnostics.error_model` en el `meta` de cada actualización (`mtpy/lib/publish/forecast.py`).
- Backtest con `error_model`: `run_case`, `run_backtest` y `backtest_scope`, columna `error_model` de `meta.csv` (el usado) y clave de `run.json` (el pedido) (`mtpy/lib/backtest.py`); job `backtest` (`mtpy/jobs/Backtest.py`), pipeline (`mtpy/pipelines/backtest/Backtest.py`) y `--error-model polls|average` (`backtest/run_backtest.py`).

Tests: `tests/test_backtest_unit.py` (calibración, `run.json`, script), `tests/test_computer_unit.py` (`average_errors` y su estimador), `tests/test_simulator_unit.py` (`error_model`), `tests/test_publish_unit.py` (`diagnostics.error_model`), `tests/test_jobs_backtest.py` y `tests/test_pipelines_backtest.py`; `tests/integration/test_average_errors.py` y la guarda de regresión `tests/integration/test_backtest.py` (con `error_model='polls'` el backtest reproduce la referencia de `backtest/results/`).

## Problema

El margen de cada partido es `sqrt(s² + e² + d²)`: `s` es el error estándar de la regresión local del promedio, `e` el error terminal y `d` la deriva hasta el horizonte. Hasta ahora `e` sale de los sondeos sueltos (`Computer.get_error_estimator`): el error absoluto de la última encuesta de cada casa a una semana de la elección, por nivel y por si el partido es regional. Pero lo que se sortea alrededor no es una encuesta, es el promedio, y el error del promedio no tiene por qué ser el de una encuesta suelta: suele ser menor, porque los errores propios de cada casa se compensan, pero arrastra el error común a todas. Y la calibración de hoy es desigual. En las generales (`backtest/results/`, cinco elecciones), a 6 días las bandas cubren lo que prometen (0,56 la del 50 %, 0,78 la del 80 %, 0,96 la del 95 %); a 14 días la del 95 % sigue en 0,96, pero las del 50 y el 80 % se quedan en 0,47 y 0,66; a 30 días o más todas se quedan cortas (la del 95 %, entre 0,72 y 0,84 en el nowcast). Una banda que acierta más de lo que promete está tan mal calibrada como una que acierta menos: la del 50 % debería acertar una de cada dos veces.

## Método

Se cambia lo que alimenta cada pieza de la fórmula, no la fórmula que explica la web. Cada paso tiene su backtest completo (`es` y autonómicas) con su tabla antes / después, y se juzga con el criterio 17 de la especificación (`docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md`): **coberturas y PIT más cerca del nominal donde faltaban, sin empeorar donde ya cubrían, y log-score y CRPS no peores más allá del ruido**. El PIT (la fracción de simulaciones por debajo del resultado oficial) se mira en tres poblaciones separadas, cada elección con el mismo peso dentro de la suya: las generales y las autonómicas al nivel de partido deciden; las provincias sólo informan de M9.

Ningún paso cambia el modelo vivo: todo entra como opción con el valor de hoy por defecto. La decisión sobre los valores por defecto es de Luis.

## El error del propio promedio (`error_model='average'`)

**Qué cambia.** `e` deja de ser el error de una encuesta suelta y pasa a ser el del propio promedio. `Computer.get_average_error_estimator` ajusta, con pesos iguales, una regresión lineal de `|e_net|` sobre el nivel del promedio y la marca de partido regional (`AverageErrorEstimator`, la misma forma que el estimador de sondeos, sin las semanas). Usa las filas de `average_errors` a 6 días (`Simulator.AVERAGE_ERROR_HORIZON`) de los partidos principales. `e_net² = max(error² − s², 0)` es el error del promedio en el ancla sin la parte que ya pone `s`, para no contarla dos veces. Las filas son las de las elecciones **anteriores** a la simulada del ámbito estimador (`est_scope`: `es` también para las autonómicas, la regla de M11), así que en el backtest el estimador está fuera de muestra por construcción. Con menos de 12 filas (o sin la tabla), el Simulator avisa y vuelve al error de sondeos. `meta.csv` y el `meta` de cada actualización (`diagnostics.error_model`) registran el modelo que se usó de verdad.

**Cómo se midió.** La tabla `elections.average_errors` todavía no existe en la base: la tiene que rellenar el job `compute` (`compute_average_errors(save=True)`) antes de que `error_model='average'` pueda usarse en una actualización. Para este backtest se calcularon en memoria las filas de `es` (14 generales desde 1982, 59 filas a 6 días), con el mismo código, y el backtest las leyó de un CSV con la semántica exacta de `load_average_errors`: sólo las elecciones anteriores a cada caso. Todo lo demás, como en la referencia: 1.000 simulaciones, semilla 42, configuración por defecto. Las generales se comparan con `backtest/results/`, regenerada el 01-11-2026 con este código y esta base. Las autonómicas, con una ejecución gemela con `error_model='polls'` (`backtest/results_polls/`, sin versionar). Los 28 casos de las generales usaron el error del promedio: para 2015 ya había 35 filas.

**Cómo se mueve `e`.** Para 2023 a 6 días (estimadores con las elecciones 1982-2019):

| Partido | Promedio | `s` | `e` sondeos | `e` promedio | Error real |
|---|---|---|---|---|---|
| PP | 34,3 | 0,35 | 2,61 | 2,39 | 1,19 |
| PSOE | 28,3 | 0,16 | 2,41 | 2,15 | 3,34 |
| SUMAR | 13,1 | 0,27 | 1,90 | 1,51 | 0,81 |
| VOX | 13,1 | 0,40 | 1,90 | 1,51 | 0,73 |

`e` baja entre un 8 y un 21 % en los partidos principales, como se esperaba: el promedio se equivoca menos que una encuesta suelta. Pero en los partidos pequeños, y sobre todo en los regionales, baja mucho más. La regresión del promedio sólo ve partidos principales, y los únicos regionales que lo han sido en `es` son CiU y PNV entre 1993 y 2004 (7 filas, con errores de décimas). Su coeficiente regional (−0,98 para 2023, con una constante de 0,96) deja a ERC, Junts, el PNV, Bildu, el BNG o CC con un `e` de centésimas. El estimador de sondeos, en cambio, aprende de cientos de encuestas de esos partidos.

### Generales (`es`, 5 elecciones, 28 casos)

Coberturas del nowcast de los partidos principales, antes → después:

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| Voto 50 % | 0,56 → 0,52 | 0,47 → 0,42 | 0,44 → 0,39 | 0,33 → 0,29 | 0,25 → 0,25 | 0,30 → 0,30 |
| Voto 80 % | 0,78 → 0,78 | 0,66 → 0,66 | 0,68 → 0,68 | 0,66 → 0,46 | 0,62 → 0,42 | 0,57 → 0,57 |
| Voto 95 % | 0,96 → 0,92 | 0,96 → 0,92 | 0,84 → 0,84 | 0,84 → 0,75 | 0,76 → 0,76 | 0,72 → 0,72 |
| Escaños 50 % | 0,61 → 0,50 | 0,53 → 0,48 | 0,46 → 0,30 | 0,27 → 0,27 | 0,22 → 0,22 | 0,35 → 0,25 |
| Escaños 80 % | 0,89 → 0,78 | 0,68 → 0,68 | 0,74 → 0,68 | 0,60 → 0,42 | 0,59 → 0,53 | 0,55 → 0,55 |
| Escaños 95 % | 1,00 → 1,00 | 1,00 → 0,96 | 0,90 → 0,90 | 0,85 → 0,79 | 0,80 → 0,74 | 0,68 → 0,68 |
| Voto 95 % con horizonte | 0,96 → 0,92 | 0,96 → 0,92 | 0,84 → 0,84 | 0,84 → 0,80 | 0,84 → 0,76 | 0,87 → 0,80 |
| Voto provincial 95 % | 0,94 → 0,92 | 0,94 → 0,91 | 0,92 → 0,88 | 0,88 → 0,85 | 0,88 → 0,84 | 0,85 → 0,80 |

Ninguna cobertura sube. Las bandas se estrechan en todos los horizontes y sólo se acercan al nominal donde sobraban: la del 95 % del voto a 6-14 días y la del 50 % de escaños a 6 días (0,61 → 0,50). Las que ya faltaban faltan más: el 50 % del voto a 14 días baja a 0,42 y el 80 % a 60-90 días, de 0,62-0,66 a 0,42-0,46.

PIT de los partidos principales (`summarize_calibration`; colas = las cajas 0 y 9, nominal 0,20; L1 = distancia a la uniforme):

| Grupo | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| 6-14 | 0,28 → 0,28 | 0,35 → 0,36 | 0,24 → 0,29 | 0,38 → 0,49 |
| 30-60 | 0,33 → 0,43 | 0,45 → 0,52 | 0,33 → 0,48 | 0,42 → 0,60 |
| 90-180 | 0,40 → 0,54 | 0,64 → 0,70 | 0,48 → 0,53 | 0,72 → 0,77 |

Con todos los partidos, la cola del voto a 6-14 días pasa de 0,26 a 0,53 (L1 de 0,22 a 0,69): el colapso de los regionales.

Puntuaciones por caso, sobre los 22 casos con escaños válidos (más bajo es mejor; casos mejor / peor). «Nowcast» es la ejecución con la elección en el ancla (columnas sin sufijo de `metrics.csv`); «horizonte», la ejecución con la deriva hasta la elección (columnas `_h`):

| Grupo | CRPS escaños, nowcast | CRPS escaños, horizonte | Log-score mayorías, nowcast | Log-score mayorías, horizonte | Brier mayorías, nowcast | CRPS escaños provinciales, nowcast |
|---|---|---|---|---|---|---|
| 6-14 (8 casos) | 5,19 → 5,19 (3 / 5) | 5,17 → 5,15 (3 / 5) | 0,171 → 0,149 (8 / 0) | 0,175 → 0,156 (8 / 0) | 0,042 → 0,038 (8 / 0) | 0,146 → 0,146 (4 / 4) |
| 30-60 (8) | 7,55 → 7,70 (1 / 7) | 7,44 → 7,53 (2 / 6) | 0,267 → 0,259 (6 / 2) | 0,282 → 0,271 (6 / 2) | 0,083 → 0,082 (6 / 2) | 0,182 → 0,183 (3 / 5) |
| 90-180 (6) | 9,56 → 9,77 (0 / 6) | 9,32 → 9,44 (0 / 6) | 0,343 → 0,344 (4 / 2) | 0,356 → 0,350 (4 / 2) | 0,118 → 0,120 (4 / 2) | 0,215 → 0,217 (1 / 5) |

Lo único que mejora es la probabilidad de mayoría a 6-14 días: un 13 % de log-score en el nowcast y un 11 % con horizonte, en los 8 casos. Pero es una mejora por estrechar. En ninguna de las cinco elecciones hubo mayoría absoluta de bloque, y con bandas más estrechas baja casi siempre la probabilidad de una mayoría que no llegó (la Izquierda en 2019-04, del 17 al 13 % a 6 días). En esta muestra, más confianza puntúa mejor; no dice nada de una elección con la mayoría en el aire, como la de 2015 para la Derecha (del 64 al 66 %). El CRPS de escaños queda igual a 6-14 días en las dos ejecuciones (3 casos mejor y 5 peor en cada una) y empeora a 30 días o más en casi todos los casos: 7 de 8 en el nowcast y 6 de 8 con horizonte a 30-60 días, 6 de 6 en las dos a 90-180.

### Autonómicas (17 ámbitos, 167 casos)

Medias por grupo de horizontes, ponderadas por casos; nowcast de los partidos principales, antes → después:

| Grupo | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 95 % | Voto provincial 95 % |
|---|---|---|---|---|---|---|
| 6-14 | 0,51 → 0,41 | 0,77 → 0,68 | 0,95 → 0,88 | 0,60 → 0,55 | 0,96 → 0,90 | 0,94 → 0,88 |
| 30-60 | 0,46 → 0,34 | 0,70 → 0,61 | 0,90 → 0,82 | 0,52 → 0,48 | 0,89 → 0,84 | 0,88 → 0,82 |
| 90-180 | 0,41 → 0,36 | 0,63 → 0,55 | 0,86 → 0,78 | 0,48 → 0,43 | 0,84 → 0,79 | 0,86 → 0,79 |

PIT de los partidos principales (colas nominal 0,20; L1 a la uniforme):

| Grupo | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| 6-14 | 0,24 → 0,32 | 0,13 → 0,25 | 0,28 → 0,34 | 0,33 → 0,39 |
| 30-60 | 0,31 → 0,39 | 0,22 → 0,37 | 0,32 → 0,38 | 0,26 → 0,39 |
| 90-180 | 0,33 → 0,41 | 0,32 → 0,42 | 0,35 → 0,41 | 0,35 → 0,43 |

Puntuaciones por caso, sobre los 147 casos con escaños válidos (casos mejor / peor; nowcast y horizonte como en las generales):

| Grupo | CRPS escaños, nowcast | CRPS escaños, horizonte | Log-score mayorías, nowcast | Log-score mayorías, horizonte | Brier mayorías, nowcast | CRPS escaños provinciales, nowcast |
|---|---|---|---|---|---|---|
| 6-14 (49 casos) | 1,294 → 1,309 (18 / 31) | 1,294 → 1,306 (18 / 31) | 0,172 → 0,173 (24 / 9) | 0,173 → 0,173 (24 / 9) | 0,057 → 0,059 (25 / 9) | 0,564 → 0,567 (18 / 31) |
| 30-60 (51) | 1,794 → 1,830 (13 / 38) | 1,777 → 1,804 (15 / 36) | 0,157 → 0,160 (29 / 8) | 0,161 → 0,163 (29 / 8) | 0,050 → 0,053 (30 / 8) | 0,864 → 0,876 (14 / 37) |
| 90-180 (47) | 2,358 → 2,410 (11 / 36) | 2,309 → 2,342 (13 / 34) | 0,124 → 0,123 (27 / 6) | 0,142 → 0,140 (27 / 8) | 0,036 → 0,038 (28 / 6) | 1,348 → 1,370 (10 / 37) |

En las autonómicas no hay ningún grupo en el que la calibración mejore. Las bandas, que a 6-14 días cubrían casi el nominal (0,95 la del 95 %, 0,77 la del 80 %), se quedan cortas en todos los horizontes. El PIT se aleja de la uniforme: a 6-14 días, el L1 del voto pasa de 0,13 a 0,25. El CRPS de escaños empeora en dos de cada tres casos o más, en el nowcast y con horizonte. Las mayorías ganan más casos de los que pierden, pero la media apenas se mueve (de 0,172 a 0,173 a 6-14 días, de 0,142 a 0,140 con horizonte a 90-180): muchos casos con una ganancia pequeña y unos pocos con una pérdida mayor. El MAE del voto es el mismo antes y después, como debe ser: el promedio no cambia, sólo cambian las bandas.

### Por qué no cubre

`e` es la predicción de una regresión del error **absoluto**, la media de `|e_net|`, y el Simulator la usa como desviación típica. Para una normal, `E|X| = σ · √(2/π) ≈ 0,8 σ`. Con las colas que tienen los errores del promedio la distancia es mayor: en las filas de `es`, la raíz del cuadrado medio de `e_net` es 1,45 veces su media absoluta a 6 días (2,39 frente a 1,66) y 1,3-1,5 veces en los demás horizontes. El estimador de sondeos tiene el mismo sesgo de forma, pero lo compensa sin querer, porque el error de una encuesta suelta es mayor que el del promedio. Al cambiar el error de la encuesta por el del promedio desaparece esa compensación y queda el sesgo a la vista.

A eso se suma, en las generales, el colapso de los partidos regionales descrito arriba: la tabla sólo tiene partidos principales, y casi ningún regional lo ha sido.

### Decisión (criterio 17)

**No se cumple el criterio.** Las coberturas no se acercan al nominal donde faltaban: a 30 días o más, en las generales y en las autonómicas, bajan más. Donde ya cubrían, a 6-14 días, empeoran: en las autonómicas la del 95 % del voto pasa de 0,95 a 0,88 y la del 50 % de 0,51 a 0,41. El PIT se aleja de la uniforme en las dos poblaciones que deciden y en todos los grupos. El CRPS de escaños empeora más allá del ruido a 30 días o más, en el nowcast y con horizonte: en las generales, 1 caso mejor y 7 peor a 30-60 días en el nowcast (2 y 6 con horizonte), 0 y 6 a 90-180 en las dos; en las autonómicas, unos dos de cada tres casos peor en todos los grupos. La única mejora, el log-score de las mayorías en las generales a 6-14 días (−13 % en el nowcast y −11 % con horizonte, 8 de 8 casos en las dos), sale de estrechar las bandas en una muestra en la que ningún bloque llegó a la mayoría absoluta, y no se repite en las autonómicas.

`error_model='average'` queda como opción documentada y `'polls'` sigue siendo el valor por defecto. **La decisión sobre los valores por defecto del modelo vivo está pendiente y es de Luis.** Lo que el backtest señala para los pasos siguientes:

- La pieza que falta es la **escala**. Habría que llevar `e` del error absoluto medio a una desviación típica, ajustando sobre `e_net²` y tomando la raíz, o multiplicando por un factor medido fuera de muestra. Con eso, el error del promedio (`'average'`) es el candidato que hay que comparar otra vez con este backtest.
- Los partidos que no están en la tabla (los no principales, y en `es` los regionales) necesitan un suelo: el `e` de sondeos o el de un partido nacional de su nivel. Con siete filas de CiU y PNV, la regresión no puede decir nada de ellos.
- El horizonte (paso 13) y las colas (paso 14) se miden mejor después de la escala. Con `e` corto, la deriva tendría que tapar también el hueco del error terminal.

Para usar `error_model='average'` en una actualización, el job `compute` tiene que rellenar antes la tabla `average_errors` en la base (`compute_average_errors(save=True)`, con `es` y cada ámbito autonómico). Este backtest usó las filas de `es` calculadas en memoria. Sin la tabla, el Simulator avisa y usa el error de sondeos.

Tiempos de la ejecución autonómica con `error_model='average'` (1.000 simulaciones, en un Mac con otra ejecución del backtest en paralelo durante la primera mitad): 10.410 s en total (2 h 53 min). Las generales tardaron 2.941 s (49 min).

## El crecimiento con el horizonte (`drift_scale`)

Cambios:
- `DriftEstimator(scale=1.)`: `var` y `sigma` multiplican la varianza de la deriva por `scale`; `scale_rows` cuenta las filas del ajuste (0 si no se ajustó). `DriftEstimator.fit_scale` y `check_drift_scale` (`mtpy/lib/computer.py`).
- `Computer.get_drift_estimator(scale=None | 'auto' | número)`: con `'auto'` ajusta la escala sobre `load_average_errors()` sin las filas de los partidos jóvenes; un número la fija; `None` deja 1 sin leer la tabla.
- `Simulator(drift_scale=None)`, validado antes de tocar la base, y `fit_drift_estimator` (`mtpy/lib/simulator.py`); `diagnostics.drift_scale` y `diagnostics.drift_scale_rows` en el `meta` de cada actualización (`mtpy/lib/publish/forecast.py`).
- Backtest con `drift_scale`: `run_case`, `run_backtest` y `backtest_scope`, columnas `drift_scale` (la usada) y `drift_scale_rows` de `meta.csv`, clave `drift_scale` de `run.json` (la pedida) (`mtpy/lib/backtest.py`); job `backtest` (`mtpy/jobs/Backtest.py`), pipeline (`mtpy/pipelines/backtest/Backtest.py`) y `--drift-scale auto|1.5` (`backtest/run_backtest.py`).
- `summarize_calibration(main_only=True)`: en las poblaciones de generales y autonómicas sólo cuentan las celdas de los partidos principales, como dice la especificación (punto 11); las provinciales, todas. Las tablas de PIT de esta sección y de la anterior son de partidos principales.

Tests: `tests/test_computer_unit.py` (`fit_scale`, la escala en `var` y `sigma`, `get_drift_estimator` con `None`, un número, `'auto'`, la tabla vacía, los valores no válidos, la exclusión de los jóvenes y la edad desconocida de los partidos con sondeos desde el principio de la base), `tests/test_simulator_unit.py` (`drift_scale` hasta el estimador, valores no válidos), `tests/test_publish_unit.py` (`diagnostics`), `tests/test_backtest_unit.py` (`run.json`, script, `main_only`), `tests/test_jobs_backtest.py` y `tests/test_pipelines_backtest.py`; la guarda de regresión `tests/integration/test_backtest.py` (con `drift_scale=None` el backtest reproduce `backtest/results/`).

**Qué cambia.** La deriva `d` es un paseo aleatorio relativo, `d² = k · días · p²`, con `k` ajustada sobre cómo se mueve la serie del promedio entre una fecha y otra. Lo que importa es otra cosa: cuánto crece el error del promedio frente al resultado cuando el ancla se aleja de la elección. `fit_scale` lo mide con la tabla `average_errors`. Para cada elección anterior y partido con fila a 6 días y a 30 días o más, el exceso de varianza del promedio sobre su propio error estándar, `(error_d² − s_d²) − (error_6² − s_6²)`, se compara con lo que ya pone la deriva en ese tramo, `k · (d − 6) · p_d²` (`p` es el nivel del promedio a `d` días, en puntos, como en `var`). La escala es la suma de los excesos entre la suma de lo esperado, recortada a [1, 4] con un aviso que dice el valor sin recortar.

No entran los partidos con menos de `age_max` años (cuatro) en su elección: su deriva ya va multiplicada por el multiplicador de edad (M8), y contarlos aquí también duplicaría su efecto. La edad sale de las mismas fechas del primer sondeo que usa la tabla de deriva. Un partido sin fecha cuenta como maduro, y también uno cuyo primer sondeo cae a menos de cuatro años del primer sondeo de la base. Los sondeos de la base empiezan en 1979, y la base no distingue un partido fundado en 1977 de uno de 1879: sin esa regla, todos los de 1982 (PSOE, AP, UCD, PCE) contarían como jóvenes y la elección de 1982 saldría entera del ajuste. Los jóvenes de verdad (CDS en 1986, IU en 1989, UP y Cs en 2015-2016, VOX en 2019, SUMAR en 2023) siguen fuera. Como el Computer estimador sólo tiene las elecciones anteriores a la simulada, el ajuste está fuera de muestra.

**Cómo se midió.** Igual que en la sección anterior: la tabla `average_errors` sigue vacía en la base, así que el backtest leyó las filas de `es` calculadas en memoria con la semántica de `load_average_errors`. 1.000 simulaciones, semilla 42. Se midieron dos configuraciones, cada una contra su propio «antes»: (a) `error_model='average'` con `drift_scale='auto'` frente a `backtest/results_avg`, y (b) el modelo que se publica, `error_model='polls'` con `drift_scale='auto'`, frente a `backtest/results` (generales) y `backtest/results_polls` (autonómicas). Las generales son los 28 casos de `es`. Las autonómicas son un subconjunto fijo de cinco ámbitos, los que más casos tienen (Madrid, Andalucía, Cataluña, Comunitat Valenciana y Galicia: 11 elecciones, 66 casos). Las otras doce comunidades quedan para una ejecución completa.

La deriva sólo entra en la ejecución con horizonte. El nowcast no cambia por construcción, y así sale: las columnas sin sufijo de `metrics.csv` son idénticas antes y después en los 28 casos de las generales y en los 66 de las autonómicas, en las dos configuraciones. Por eso no se tabulan. Todo lo que sigue es de la ejecución con horizonte (columnas `_h`, PIT del modo `h`).

**La escala ajustada.** La misma en las dos configuraciones (no depende de `e`):

| Elección simulada | 2015-12 | 2016-06 | 2019-04 | 2019-11 | 2023-07 |
|---|---|---|---|---|---|
| Escala | 4,00 (recortada) | 2,25 | 2,22 | 2,23 | 2,49 |
| Filas | 136 | 148 | 154 | 170 | 182 |

En 2015 sólo había siete generales detrás, con el ciclo de 1982 pesando mucho, y el valor se recorta a 4. Desde 2016 la escala es estable, entre 2,2 y 2,5: el error del promedio crece con el horizonte algo más del doble de lo que pone la deriva de la serie. Cada autonómica toma la escala de `es` con las generales anteriores a ella: entre 2,22 y 2,49 en los cinco ámbitos (entre 154 y 194 filas).

Dos ajustes alternativos, como sensibilidad:
- Con los partidos jóvenes dentro del ajuste, la escala sale 2,4-3,1 (4,0 recortada en 2015). En (a) eso sube la cobertura del 95 % con horizonte a 90-180 días del voto de 0,78 a 0,85 y la de escaños de 0,80 a 0,87, casi lo mismo que la regla elegida.
- Sin 1982 (todos sus partidos cuentan como jóvenes para la base), la escala sale 1,69 / 1,00 recortada / 1,00 recortada / 1,00 / 1,34. Apenas mueve nada: en (b), la del 80 % del voto a 30-60 días de 0,72 a 0,77 y poco más. El CRPS de escaños mejora en unos pocos casos y el log-score de las mayorías autonómicas ya empeora (0 casos mejor y 11 peor a 90-180 días).

### (a) El error del promedio con la escala

Coberturas con horizonte de los partidos principales, antes → después (media por grupo de horizontes, ponderada por casos):

| Grupo | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 | 0,49 → 0,51 | 0,72 → 0,75 | 0,92 → 0,94 | 0,49 → 0,54 | 0,73 → 0,79 | 0,98 → 0,98 |
| Generales 30-60 | 0,38 → 0,43 | 0,68 → 0,75 | 0,82 → 0,84 | 0,34 → 0,44 | 0,69 → 0,78 | 0,88 → 0,88 |
| Generales 90-180 | 0,30 → 0,38 | 0,58 → 0,70 | 0,78 → 0,85 | 0,31 → 0,41 | 0,57 → 0,70 | 0,80 → 0,87 |
| Autonómicas 6-14 | 0,46 → 0,47 | 0,73 → 0,73 | 0,87 → 0,87 | 0,55 → 0,57 | 0,74 → 0,75 | 0,88 → 0,88 |
| Autonómicas 30-60 | 0,31 → 0,36 | 0,60 → 0,66 | 0,78 → 0,80 | 0,46 → 0,50 | 0,65 → 0,69 | 0,83 → 0,84 |
| Autonómicas 90-180 | 0,34 → 0,34 | 0,59 → 0,67 | 0,79 → 0,79 | 0,42 → 0,47 | 0,65 → 0,73 | 0,85 → 0,87 |

PIT del modo `h`, partidos principales (colas = las cajas 0 y 9, nominal 0,20; L1 = distancia a la uniforme):

| Grupo | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 | 0,28 → 0,25 | 0,38 → 0,29 | 0,29 → 0,26 | 0,55 → 0,34 |
| Generales 30-60 | 0,32 → 0,25 | 0,39 → 0,32 | 0,31 → 0,22 | 0,37 → 0,35 |
| Generales 90-180 | 0,41 → 0,29 | 0,60 → 0,45 | 0,41 → 0,29 | 0,62 → 0,50 |
| Autonómicas 6-14 | 0,28 → 0,28 | 0,25 → 0,25 | 0,29 → 0,29 | 0,35 → 0,35 |
| Autonómicas 30-60 | 0,38 → 0,32 | 0,41 → 0,32 | 0,38 → 0,34 | 0,38 → 0,32 |
| Autonómicas 90-180 | 0,37 → 0,27 | 0,42 → 0,33 | 0,42 → 0,30 | 0,50 → 0,34 |

Puntuaciones por caso con horizonte, sobre los casos con escaños válidos (más bajo es mejor; casos mejor / peor):

| Grupo | CRPS escaños, generales | Log-score mayorías, generales | CRPS escaños, autonómicas | Log-score mayorías, autonómicas |
|---|---|---|---|---|
| 6-14 | 5,155 → 5,129 (4 / 4; 8 casos) | 0,156 → 0,163 (0 / 7) | 1,783 → 1,778 (15 / 7; 22 casos) | 0,107 → 0,109 (2 / 11) |
| 30-60 | 7,525 → 7,427 (7 / 1; 8) | 0,271 → 0,282 (2 / 6) | 2,692 → 2,654 (16 / 6; 22) | 0,075 → 0,086 (0 / 16) |
| 90-180 | 9,443 → 9,301 (5 / 1; 6) | 0,350 → 0,360 (2 / 4) | 3,464 → 3,405 (14 / 8; 22) | 0,082 → 0,104 (0 / 16) |

La escala no arregla lo que le falta a `'average'`: el `e` corto y los partidos ausentes de la tabla (sección anterior) siguen ahí. El `e` de los regionales sigue en centésimas porque `AverageErrorEstimator.predict` no tiene suelo, y la escala, que sólo toca la deriva, no lo cambia. A 6-14 días, donde la deriva es pequeña, lo que se ve es sobre todo el hueco del `e` corto tapado en parte por la deriva.

### (b) El modelo que se publica con la escala

Coberturas con horizonte de los partidos principales, antes → después:

| Grupo | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 | 0,51 → 0,54 | 0,75 → 0,75 | 0,96 → 0,96 | 0,60 → 0,63 | 0,85 → 0,88 | 1,00 → 1,00 |
| Generales 30-60 | 0,40 → 0,48 | 0,72 → 0,77 | 0,84 → 0,86 | 0,41 → 0,47 | 0,72 → 0,81 | 0,88 → 0,90 |
| Generales 90-180 | 0,33 → 0,38 | 0,67 → 0,75 | 0,85 → 0,85 | 0,34 → 0,45 | 0,66 → 0,75 | 0,83 → 0,87 |
| Autonómicas 6-14 | 0,55 → 0,55 | 0,78 → 0,78 | 0,94 → 0,94 | 0,67 → 0,67 | 0,77 → 0,78 | 0,93 → 0,93 |
| Autonómicas 30-60 | 0,40 → 0,41 | 0,67 → 0,69 | 0,86 → 0,87 | 0,52 → 0,55 | 0,69 → 0,73 | 0,87 → 0,89 |
| Autonómicas 90-180 | 0,36 → 0,39 | 0,62 → 0,68 | 0,83 → 0,85 | 0,45 → 0,51 | 0,68 → 0,76 | 0,87 → 0,89 |

PIT del modo `h`, partidos principales:

| Grupo | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 | 0,25 → 0,25 | 0,26 → 0,28 | 0,21 → 0,18 | 0,25 → 0,27 |
| Generales 30-60 | 0,28 → 0,23 | 0,29 → 0,30 | 0,28 → 0,19 | 0,37 → 0,40 |
| Generales 90-180 | 0,33 → 0,24 | 0,47 → 0,43 | 0,35 → 0,26 | 0,56 → 0,45 |
| Autonómicas 6-14 | 0,22 → 0,22 | 0,22 → 0,21 | 0,26 → 0,26 | 0,35 → 0,36 |
| Autonómicas 30-60 | 0,31 → 0,27 | 0,26 → 0,25 | 0,34 → 0,31 | 0,27 → 0,25 |
| Autonómicas 90-180 | 0,33 → 0,26 | 0,34 → 0,25 | 0,35 → 0,28 | 0,40 → 0,28 |

Puntuaciones por caso con horizonte (casos mejor / peor):

| Grupo | CRPS escaños, generales | Log-score mayorías, generales | CRPS escaños, autonómicas | Log-score mayorías, autonómicas |
|---|---|---|---|---|
| 6-14 | 5,167 → 5,154 (3 / 5; 8 casos) | 0,175 → 0,182 (1 / 7) | 1,766 → 1,764 (12 / 10; 22 casos) | 0,115 → 0,116 (0 / 13) |
| 30-60 | 7,435 → 7,380 (6 / 2; 8) | 0,282 → 0,292 (3 / 5) | 2,641 → 2,613 (16 / 6; 22) | 0,082 → 0,091 (1 / 14) |
| 90-180 | 9,321 → 9,227 (5 / 1; 6) | 0,356 → 0,367 (2 / 4) | 3,408 → 3,368 (14 / 9; 22) | 0,086 → 0,107 (0 / 16) |

El Brier de las mayorías va en la misma dirección que el log-score: en las autonómicas, 1 caso mejor y 16 peor a 90-180 días (de 0,020 a 0,025).

### Decisión (criterio 17)

**No se cumple el criterio.** En las dos configuraciones, coberturas y PIT con horizonte se acercan al nominal a 30 días o más, donde faltaban: en el modelo que se publica, la del 80 % de escaños a 90-180 días pasa de 0,66 a 0,75 en las generales y de 0,68 a 0,76 en las autonómicas, y las colas del PIT del voto bajan de 0,33 a 0,24-0,26. El CRPS de escaños mejora en la mayoría de los casos a 30 días o más (generales 6 / 2 y 5 / 1; autonómicas 16 / 6 y 13 / 9). Pero la probabilidad de mayoría absoluta empeora más allá del ruido. En las autonómicas, el log-score empeora en casi todos los casos que cambian: 0 mejor y 13 peor a 6-14 días, 1 y 14 a 30-60, 0 y 16 a 90-180, con la media de 0,086 a 0,107 (+24 %). En las generales es 1 / 7, 3 / 5 y 2 / 4. Y donde ya cubrían, a 6-14 días, algunas bandas de las generales pasan del nominal (la del 50 % de escaños de 0,60 a 0,63, la del 80 % de 0,85 a 0,88). Las cifras de la ejecución sin 1982 dicen lo mismo con un efecto más pequeño.

La lectura es que la escala ensancha bien cada partido por separado, pero la deriva de los partidos se sortea de forma independiente. Con bandas más anchas, la suma de un bloque se abre y su probabilidad de mayoría se va hacia el medio, también en las comunidades en las que la mayoría estaba clara. El ancho de cada partido mejora y la probabilidad conjunta empeora: la escala necesita la composición del ruido (paso 15, M7) para que la deriva de un bloque no crezca como la de partidos independientes.

`drift_scale` queda como opción documentada; `None` (1) sigue siendo el valor por defecto y nada cambia en el modelo vivo. **La decisión es de Luis.** Lo que el backtest señala:

- La escala medida es estable, 2,2-2,5 desde 2016: la deriva de la serie se queda en menos de la mitad de lo que crece el error del promedio con el horizonte.
- Antes de adoptarla, hay que probarla junto con la composición del ruido (paso 15), mirando el log-score y el Brier de las mayorías.
- Una ejecución completa (las 17 comunidades) de la configuración que se elija; esta se hizo sobre cinco.

Tiempos (1.000 simulaciones, en un Mac con cuatro ejecuciones a la vez): generales 3.037 s en (a) y 3.104 s en (b); autonómicas, 66 casos en cada configuración, entre 770 y 977 s por ámbito (4.263 s y 4.429 s en total).
