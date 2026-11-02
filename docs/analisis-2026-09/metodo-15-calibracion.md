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

No entran los partidos con menos de `age_max` años (cuatro) en su elección: su deriva ya va multiplicada por el multiplicador de edad (M8), y contarlos aquí también duplicaría su efecto. La edad sale de las mismas fechas del primer sondeo que usa la tabla de deriva. Un partido sin fecha cuenta como maduro, y también uno cuyo primer sondeo cae a menos de cuatro años del primer sondeo de la base. Los sondeos de la base empiezan el 18-06-1979, y la base no distingue un partido fundado en 1977 de uno de 1879: sin esa regla, todos los de 1982 (PSOE, AP, UCD, PCE) contarían como jóvenes y la elección de 1982 saldría entera del ajuste. La ventana de cuatro años llega hasta el 18-06-1983, y deja dentro también al CDS: se fundó en julio de 1982, su primer sondeo es del 30-08-1982 y en las generales de 1986 tenía 3,8 años. Para la regla su edad es desconocida, y sus filas de 1986 se quedan. Son 20 filas más en el ajuste: 16 de 1982 (cuatro horizontes de 30 días o más de cada uno de los cuatro partidos) y 4 del CDS en 1986. Siguen fuera, como jóvenes, IU en 1989, UP y Cs en 2015 y 2016, VOX en abril y noviembre de 2019 y Más País en noviembre de 2019. El caso del CDS es un punto para Luis: la ventana es una elección y se puede estrechar, y la edad de M8 usa la misma aproximación. Como el Computer estimador sólo tiene las elecciones anteriores a la simulada, el ajuste está fuera de muestra.

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
- Sin 1982 ni el CDS de 1986 (todos cuentan como jóvenes para la base), la escala sale 1,69 / 1,00 recortada / 1,00 recortada / 1,00 / 1,34. Apenas mueve nada: en (b), la del 80 % del voto a 30-60 días de 0,72 a 0,77 y poco más. El CRPS de escaños mejora en unos pocos casos y el log-score de las mayorías autonómicas ya empeora (0 casos mejor y 11 peor a 90-180 días).

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
| 90-180 | 9,321 → 9,227 (5 / 1; 6) | 0,356 → 0,367 (2 / 4) | 3,408 → 3,368 (13 / 9; 22) | 0,086 → 0,107 (0 / 16) |

El Brier de las mayorías va en la misma dirección que el log-score: en las autonómicas, 1 caso mejor y 16 peor a 90-180 días (de 0,020 a 0,025).

### Decisión (criterio 17)

**No se cumple el criterio.** En las dos configuraciones, coberturas y PIT con horizonte se acercan al nominal a 30 días o más, donde faltaban: en el modelo que se publica, la del 80 % de escaños a 90-180 días pasa de 0,66 a 0,75 en las generales y de 0,68 a 0,76 en las autonómicas, y las colas del PIT del voto bajan de 0,33 a 0,24-0,26. El CRPS de escaños mejora en la mayoría de los casos a 30 días o más (generales 6 / 2 y 5 / 1; autonómicas 16 / 6 y 13 / 9). Pero la probabilidad de mayoría absoluta empeora más allá del ruido. En las autonómicas, el log-score empeora en casi todos los casos que cambian: 0 mejor y 13 peor a 6-14 días, 1 y 14 a 30-60, 0 y 16 a 90-180, con la media de 0,086 a 0,107 (+24 %). En las generales es 1 / 7, 3 / 5 y 2 / 4. Y donde ya cubrían, a 6-14 días, algunas bandas de las generales pasan del nominal (la del 50 % de escaños de 0,60 a 0,63, la del 80 % de 0,85 a 0,88). Las cifras de la ejecución sin 1982 ni el CDS de 1986 dicen lo mismo con un efecto más pequeño.

La lectura es que la escala ensancha bien cada partido por separado, pero la deriva de los partidos se sortea de forma independiente. Con bandas más anchas, la suma de un bloque se abre y su probabilidad de mayoría se va hacia el medio, también en las comunidades en las que la mayoría estaba clara. El ancho de cada partido mejora y la probabilidad conjunta empeora: la escala necesita la composición del ruido (paso 15, M7) para que la deriva de un bloque no crezca como la de partidos independientes.

`drift_scale` queda como opción documentada; `None` (1) sigue siendo el valor por defecto y nada cambia en el modelo vivo. **La decisión es de Luis.** Lo que el backtest señala:

- La escala medida es estable, 2,2-2,5 desde 2016: la deriva de la serie se queda en menos de la mitad de lo que crece el error del promedio con el horizonte.
- Antes de adoptarla, hay que probarla junto con la composición del ruido (paso 15), mirando el log-score y el Brier de las mayorías.
- Una ejecución completa (las 17 comunidades) de la configuración que se elija; esta se hizo sobre cinco.

Tiempos (1.000 simulaciones, en un Mac con cuatro ejecuciones a la vez): generales 3.037 s en (a) y 3.104 s en (b); autonómicas, 66 casos en cada configuración, entre 770 y 977 s por ámbito (4.263 s y 4.429 s en total).

## Las colas (`tail_dof`)

Cambios:
- `Simulator(tail_dof=None)`, validado antes de tocar la base (`_check_tail_dof`: `None` o un entero de al menos 2; 0, 1, negativos, decimales, booleanos y cadenas dan `ValueError`). `degrees_of_freedom` da los grados de libertad de la t de cada partido, los mismos en el sorteo (`build_frame`) y en el abanico (`_fan`): `tail_dof` para todos cuando se fija y, si no, `nobs − 2` con mínimo 3, como hasta ahora. El frame del sorteo guarda los que usó en la columna nueva `dof` de `cols_frame`, que va al final y no cambia el orden del sorteo (`mtpy/lib/simulator.py`). `diagnostics.tail_dof` en el `meta` de cada actualización (`mtpy/lib/publish/forecast.py`).
- Backtest con `tail_dof`: `run_case`, `run_backtest` y `backtest_scope`, columna `tail_dof` de `meta.csv` (vacía sin fijar) y clave de `run.json` (`mtpy/lib/backtest.py`); job `backtest` con su validación (`mtpy/jobs/Backtest.py`), pipeline (`mtpy/pipelines/backtest/Backtest.py`) y `--tail-dof 5` (`backtest/run_backtest.py`).

Tests: `tests/test_simulator_unit.py` (`tail_dof` fija los grados de libertad del sorteo y del abanico; con `None`, los `nobs − 2` y el mismo sorteo; valores no válidos en el fixture y en el constructor), `tests/test_publish_unit.py` (`diagnostics`), `tests/test_backtest_unit.py` (`run.json`, script), `tests/test_jobs_backtest.py` y `tests/test_pipelines_backtest.py`; la guarda de regresión `tests/integration/test_backtest.py` (con `tail_dof=None` el backtest reproduce `backtest/results/`).

**Qué cambia.** Cada partido se sortea como `p + σ · t(ν)`, con `σ` el margen `sqrt(s² + e² + d²)` y `ν` sus grados de libertad. Hasta ahora `ν = nobs − 2` (mínimo 3). En las generales, con cientos de sondeos por ciclo, es casi una normal; en las autonómicas lejos de la elección, no (ver abajo). La especificación (punto 14) pide probar unas colas fijas para todos los partidos, `ν` = 3, 5, 8 o 15, frente a `None`, elegir por el log-score del voto dejando fuera cada elección por turno y, si las diferencias quedan dentro del ruido, quedarse con 5 (el precedente de 538) y decirlo.

Hay un detalle del código que pesa en la lectura. `σ` es la escala de la t, no su desviación típica: la varianza de una t de `ν` grados es `ν / (ν − 2)`. Con `ν` fijo, el sorteo no sólo tiene colas más gruesas, también es más ancho. Frente a una normal, la desviación típica crece ×1,73 con 3 grados, ×1,29 con 5, ×1,15 con 8 y ×1,07 con 15. Y la banda del 95 % crece más que la del 50 %: con 5 grados, ×1,31 frente a ×1,08. Lo que se compara aquí son colas y ancho a la vez.

Esa comparación con una normal vale para `None` en las generales, no en todas las autonómicas. Los grados que usó `None` están acotados por los sondeos de cada caso: los `nobs` de un partido no pasan de los sondeos del ciclo (`n_polls` de `meta.csv`), así que `ν ≤ n_polls − 2`, con mínimo 3.
- Generales: la cota es de al menos 18 en todos los casos, y la mediana va de 258 a 311 según el grupo de horizontes. Ningún caso queda en 15 o menos.
- Autonómicas: la mediana es de 44,5 a 6-14 días, 19 a 30-60 y 12 a 90-180. A 30-60 días, 7 de los 22 casos no pasan de 15 (1 no pasa de 8); a 90-180, 12 de 22 (3 no pasan de 8). En las gallegas de 2020, a 180 días, había 5 sondeos, así que `ν` era 3.

Donde `None` ya usaba pocos grados, fijar 8 o 15 da un sorteo más estrecho y de colas más finas que `None`, no más ancho.

**Cómo se midió.** Sobre la configuración de la sección anterior, `error_model='average'` con `drift_scale='auto'`. 1.000 simulaciones, semilla 42, y la tabla `average_errors` leída del mismo CSV calculado en memoria (sigue vacía en la base). El candidato `None` es esa misma ejecución, `backtest/results_avg_scale2` (sin repetirla). Los otros cuatro van en `backtest/results_dof3`, `_dof5`, `_dof8` y `_dof15`. Las generales son los 28 casos de `es` (22 con escaños válidos). Las autonómicas son los cinco ámbitos fijos de la sección anterior: Madrid, Andalucía, Cataluña, Comunitat Valenciana y Galicia, 11 elecciones y 66 casos. La escala de la deriva ajustada es idéntica en las cinco ejecuciones, porque no depende de `ν`. Los grados de libertad entran en los dos sorteos, el nowcast y el de horizonte, así que se tabulan los dos modos.

El log-score del voto no está en `metrics.csv`, y lo calculamos aparte con las mismas tablas para los cinco candidatos. Para cada partido principal es menos el logaritmo de la densidad del sorteo en su resultado oficial. La densidad se reconstruye con los 21 cuantiles exportados en `shares.csv` (2,5 %, de 5 en 5 hasta el 95 % y 97,5 %). La función de distribución es lineal a trozos entre ellos, así que la densidad es constante en cada tramo, con un ancho mínimo de 0,01 puntos por tramo para los cuantiles repetidos. Fuera del 2,5 % y del 97,5 %, una cola exponencial lleva ese 2,5 % de masa y empieza con la densidad del último tramo. Es una aproximación: más allá del 97,5 % la cola es la misma forma para todos los candidatos, y lo que distingue a cada uno es el reparto de su masa dentro de ese intervalo. Por caso se hace la media de sus partidos principales; más bajo es mejor, como en el log-score de las mayorías.

### Coberturas

Partidos principales, media por grupo de horizontes ponderada por casos. En cada celda, los cinco candidatos en este orden: None · 3 · 5 · 8 · 15. Con horizonte:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,51 · 0,58 · 0,54 · 0,48 · 0,51 | 0,75 · 0,85 · 0,77 · 0,70 · 0,75 | 0,94 · 1,00 · 0,98 · 0,96 · 0,96 | 0,54 · 0,65 · 0,63 · 0,51 · 0,51 | 0,79 · 0,98 · 0,93 · 0,76 · 0,76 | 0,98 · 1,00 · 1,00 · 1,00 · 1,00 |
| Generales 30-60 (10) | 0,43 · 0,45 · 0,45 · 0,38 · 0,40 | 0,75 · 0,75 · 0,75 · 0,75 · 0,75 | 0,84 · 0,96 · 0,86 · 0,84 · 0,84 | 0,44 · 0,50 · 0,44 · 0,39 · 0,42 | 0,78 · 0,83 · 0,83 · 0,81 · 0,78 | 0,88 · 0,98 · 0,93 · 0,93 · 0,91 |
| Generales 90-180 (8) | 0,38 · 0,45 · 0,41 · 0,38 · 0,35 | 0,70 · 0,81 · 0,75 · 0,72 · 0,72 | 0,85 · 0,95 · 0,95 · 0,88 · 0,88 | 0,41 · 0,52 · 0,44 · 0,44 · 0,34 | 0,70 · 0,82 · 0,78 · 0,75 · 0,75 | 0,87 · 0,94 · 0,88 · 0,88 · 0,88 |
| Autonómicas 6-14 (22) | 0,47 · 0,51 · 0,47 · 0,50 · 0,48 | 0,73 · 0,80 · 0,76 · 0,73 · 0,73 | 0,86 · 0,97 · 0,94 · 0,90 · 0,88 | 0,57 · 0,68 · 0,61 · 0,62 · 0,59 | 0,75 · 0,82 · 0,77 · 0,76 · 0,74 | 0,88 · 0,97 · 0,93 · 0,89 · 0,89 |
| Autonómicas 30-60 (22) | 0,36 · 0,43 · 0,38 · 0,37 · 0,33 | 0,66 · 0,71 · 0,66 · 0,65 · 0,64 | 0,80 · 0,94 · 0,88 · 0,81 · 0,79 | 0,50 · 0,54 · 0,53 · 0,51 · 0,50 | 0,69 · 0,76 · 0,72 · 0,71 · 0,69 | 0,84 · 0,96 · 0,89 · 0,86 · 0,83 |
| Autonómicas 90-180 (22) | 0,34 · 0,42 · 0,37 · 0,37 · 0,33 | 0,67 · 0,71 · 0,67 · 0,66 · 0,66 | 0,79 · 0,91 · 0,84 · 0,78 · 0,77 | 0,47 · 0,52 · 0,50 · 0,48 · 0,48 | 0,73 · 0,79 · 0,75 · 0,73 · 0,72 | 0,87 · 0,95 · 0,87 · 0,85 · 0,86 |

Nowcast:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,47 · 0,56 · 0,52 · 0,48 · 0,47 | 0,72 · 0,77 · 0,72 · 0,68 · 0,72 | 0,92 · 1,00 · 0,98 · 0,96 · 0,96 | 0,49 · 0,63 · 0,57 · 0,49 · 0,51 | 0,73 · 0,89 · 0,78 · 0,73 · 0,71 | 0,98 · 1,00 · 1,00 · 1,00 · 1,00 |
| Generales 30-60 (10) | 0,34 · 0,38 · 0,37 · 0,34 · 0,36 | 0,57 · 0,70 · 0,68 · 0,61 · 0,59 | 0,80 · 0,92 · 0,84 · 0,84 · 0,82 | 0,28 · 0,35 · 0,31 · 0,31 · 0,28 | 0,55 · 0,75 · 0,75 · 0,60 · 0,61 | 0,84 · 0,96 · 0,90 · 0,88 · 0,82 |
| Generales 90-180 (8) | 0,27 · 0,30 · 0,30 · 0,27 · 0,27 | 0,47 · 0,61 · 0,58 · 0,58 · 0,47 | 0,75 · 0,91 · 0,80 · 0,80 · 0,77 | 0,23 · 0,30 · 0,27 · 0,26 · 0,23 | 0,53 · 0,57 · 0,53 · 0,53 · 0,47 | 0,72 · 0,91 · 0,81 · 0,73 · 0,73 |
| Autonómicas 6-14 (22) | 0,46 · 0,49 · 0,47 · 0,47 · 0,46 | 0,73 · 0,77 · 0,75 · 0,73 · 0,73 | 0,86 · 0,97 · 0,92 · 0,88 · 0,86 | 0,55 · 0,67 · 0,58 · 0,59 · 0,57 | 0,73 · 0,79 · 0,76 · 0,73 · 0,73 | 0,87 · 0,96 · 0,92 · 0,89 · 0,88 |
| Autonómicas 30-60 (22) | 0,27 · 0,36 · 0,32 · 0,30 · 0,28 | 0,55 · 0,63 · 0,58 · 0,57 · 0,57 | 0,74 · 0,89 · 0,80 · 0,76 · 0,74 | 0,44 · 0,49 · 0,45 · 0,44 · 0,44 | 0,64 · 0,69 · 0,66 · 0,64 · 0,65 | 0,78 · 0,89 · 0,83 · 0,78 · 0,75 |
| Autonómicas 90-180 (22) | 0,29 · 0,32 · 0,30 · 0,30 · 0,27 | 0,47 · 0,56 · 0,51 · 0,50 · 0,48 | 0,69 · 0,81 · 0,73 · 0,68 · 0,67 | 0,37 · 0,43 · 0,38 · 0,39 · 0,37 | 0,57 · 0,64 · 0,60 · 0,58 · 0,57 | 0,74 · 0,84 · 0,79 · 0,73 · 0,72 |

### PIT

Partidos principales (colas = las cajas 0 y 9, nominal 0,20; L1 = distancia a la uniforme). Entre paréntesis, las elecciones del voto y de los escaños. Con horizonte:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,25 · 0,15 · 0,23 · 0,30 · 0,25 | 0,29 · 0,24 · 0,30 · 0,46 · 0,34 | 0,26 · 0,02 · 0,13 · 0,24 · 0,26 | 0,34 · 0,44 · 0,46 · 0,30 · 0,34 |
| Generales 30-60 (5 / 4) | 0,25 · 0,25 · 0,25 · 0,25 · 0,25 | 0,32 · 0,29 · 0,28 · 0,28 · 0,33 | 0,22 · 0,17 · 0,17 · 0,22 · 0,25 | 0,35 · 0,48 · 0,50 · 0,40 · 0,33 |
| Generales 90-180 (5 / 4) | 0,29 · 0,21 · 0,25 · 0,27 · 0,27 | 0,45 · 0,43 · 0,48 · 0,47 · 0,52 | 0,29 · 0,20 · 0,23 · 0,26 · 0,26 | 0,50 · 0,39 · 0,38 · 0,45 · 0,38 |
| Autonómicas 6-14 (11 / 11) | 0,27 · 0,20 · 0,24 · 0,27 · 0,27 | 0,25 · 0,18 · 0,22 · 0,24 · 0,28 | 0,29 · 0,22 · 0,26 · 0,28 · 0,29 | 0,35 · 0,25 · 0,27 · 0,28 · 0,30 |
| Autonómicas 30-60 (11 / 11) | 0,32 · 0,26 · 0,32 · 0,33 · 0,34 | 0,32 · 0,25 · 0,27 · 0,27 · 0,36 | 0,34 · 0,26 · 0,31 · 0,33 · 0,34 | 0,32 · 0,13 · 0,30 · 0,32 · 0,33 |
| Autonómicas 90-180 (11 / 11) | 0,27 · 0,23 · 0,28 · 0,30 · 0,30 | 0,33 · 0,24 · 0,27 · 0,31 · 0,32 | 0,30 · 0,25 · 0,29 · 0,32 · 0,35 | 0,34 · 0,24 · 0,32 · 0,34 · 0,40 |

Nowcast:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,28 · 0,23 · 0,28 · 0,32 · 0,29 | 0,36 · 0,27 · 0,30 · 0,45 · 0,26 | 0,29 · 0,14 · 0,24 · 0,27 · 0,32 | 0,49 · 0,29 · 0,45 · 0,43 · 0,44 |
| Generales 30-60 (5 / 4) | 0,43 · 0,30 · 0,32 · 0,39 · 0,41 | 0,52 · 0,41 · 0,57 · 0,58 · 0,54 | 0,48 · 0,25 · 0,33 · 0,42 · 0,45 | 0,60 · 0,38 · 0,53 · 0,59 · 0,64 |
| Generales 90-180 (5 / 4) | 0,54 · 0,42 · 0,43 · 0,43 · 0,51 | 0,70 · 0,69 · 0,69 · 0,73 · 0,66 | 0,53 · 0,44 · 0,47 · 0,51 · 0,55 | 0,77 · 0,67 · 0,67 · 0,72 · 0,70 |
| Autonómicas 6-14 (11 / 11) | 0,27 · 0,22 · 0,25 · 0,27 · 0,27 | 0,25 · 0,21 · 0,22 · 0,26 · 0,29 | 0,29 · 0,23 · 0,27 · 0,30 · 0,29 | 0,36 · 0,27 · 0,29 · 0,31 · 0,32 |
| Autonómicas 30-60 (11 / 11) | 0,43 · 0,35 · 0,40 · 0,41 · 0,41 | 0,52 · 0,34 · 0,41 · 0,42 · 0,46 | 0,40 · 0,34 · 0,36 · 0,39 · 0,40 | 0,40 · 0,31 · 0,37 · 0,40 · 0,39 |
| Autonómicas 90-180 (11 / 11) | 0,50 · 0,40 · 0,46 · 0,47 · 0,49 | 0,60 · 0,47 · 0,52 · 0,54 · 0,58 | 0,50 · 0,40 · 0,44 · 0,49 · 0,49 | 0,59 · 0,46 · 0,52 · 0,59 · 0,58 |

### Puntuaciones por caso

Más bajo es mejor. En cada celda, la media de los cinco candidatos (None · 3 · 5 · 8 · 15) y, entre corchetes, los casos en que 5 mejora / empeora frente a None y los casos con valor. Con horizonte:

| Grupo | Log-score voto | CRPS escaños | Log-score mayorías | Brier mayorías |
|---|---|---|---|---|
| Generales 6-14 | 2,226 · 2,266 · 2,236 · 2,317 · 2,205 [4 / 6; 10] | 5,129 · 5,280 · 5,208 · 5,326 · 5,120 [3 / 5; 8] | 0,163 · 0,225 · 0,187 · 0,178 · 0,166 [2 / 6; 8] | 0,041 · 0,054 · 0,045 · 0,044 · 0,041 [2 / 6; 8] |
| Generales 30-60 | 2,807 · 2,646 · 2,607 · 2,696 · 2,709 [7 / 3; 10] | 7,427 · 7,442 · 7,396 · 7,543 · 7,320 [4 / 4; 8] | 0,282 · 0,328 · 0,297 · 0,299 · 0,280 [1 / 7; 8] | 0,086 · 0,097 · 0,088 · 0,091 · 0,085 [3 / 5; 8] |
| Generales 90-180 | 3,026 · 2,783 · 2,817 · 2,863 · 2,849 [5 / 3; 8] | 9,301 · 9,173 · 9,071 · 9,319 · 9,098 [5 / 1; 6] | 0,360 · 0,396 · 0,370 · 0,374 · 0,359 [2 / 4; 6] | 0,119 · 0,127 · 0,120 · 0,123 · 0,118 [2 / 4; 6] |
| Generales todos | 2,662 · 2,549 · 2,535 · 2,608 · 2,569 [16 / 12; 28] | 7,102 · 7,128 · 7,057 · 7,221 · 7,005 [12 / 10; 22] | 0,260 · 0,309 · 0,277 · 0,275 · 0,260 [5 / 17; 22] | 0,079 · 0,090 · 0,081 · 0,083 · 0,078 [7 / 15; 22] |
| Autonómicas 6-14 | 2,304 · 2,227 · 2,243 · 2,268 · 2,275 [10 / 12; 22] | 1,778 · 1,777 · 1,776 · 1,768 · 1,786 [10 / 12; 22] | 0,109 · 0,134 · 0,120 · 0,116 · 0,109 [1 / 15; 22] | 0,032 · 0,037 · 0,034 · 0,033 · 0,031 [1 / 15; 22] |
| Autonómicas 30-60 | 2,698 · 2,656 · 2,663 · 2,695 · 2,742 [15 / 7; 22] | 2,654 · 2,612 · 2,629 · 2,632 · 2,654 [11 / 11; 22] | 0,086 · 0,113 · 0,095 · 0,090 · 0,082 [1 / 15; 22] | 0,019 · 0,026 · 0,021 · 0,020 · 0,017 [1 / 16; 22] |
| Autonómicas 90-180 | 2,871 · 2,830 · 2,874 · 2,871 · 2,945 [10 / 12; 22] | 3,405 · 3,375 · 3,383 · 3,391 · 3,408 [12 / 10; 22] | 0,104 · 0,138 · 0,115 · 0,109 · 0,099 [1 / 16; 22] | 0,024 · 0,032 · 0,027 · 0,026 · 0,023 [1 / 18; 22] |
| Autonómicas todos | 2,624 · 2,571 · 2,593 · 2,611 · 2,654 [35 / 31; 66] | 2,613 · 2,588 · 2,596 · 2,597 · 2,616 [33 / 33; 66] | 0,100 · 0,129 · 0,110 · 0,105 · 0,097 [3 / 46; 66] | 0,025 · 0,032 · 0,027 · 0,026 · 0,024 [3 / 49; 66] |

Nowcast:

| Grupo | Log-score voto | CRPS escaños | Log-score mayorías | Brier mayorías |
|---|---|---|---|---|
| Generales 6-14 | 2,258 · 2,296 · 2,236 · 2,303 · 2,226 [5 / 5; 10] | 5,191 · 5,265 · 5,231 · 5,362 · 5,175 [4 / 4; 8] | 0,149 · 0,212 · 0,176 · 0,166 · 0,155 [2 / 6; 8] | 0,038 · 0,051 · 0,042 · 0,041 · 0,039 [2 / 6; 8] |
| Generales 30-60 | 3,047 · 2,680 · 2,765 · 2,916 · 2,916 [6 / 4; 10] | 7,697 · 7,481 · 7,565 · 7,746 · 7,579 [4 / 4; 8] | 0,259 · 0,293 · 0,263 · 0,266 · 0,248 [3 / 5; 8] | 0,082 · 0,087 · 0,079 · 0,083 · 0,077 [3 / 5; 8] |
| Generales 90-180 | 3,490 · 2,843 · 2,943 · 3,196 · 3,209 [6 / 2; 8] | 9,767 · 9,286 · 9,407 · 9,667 · 9,566 [5 / 1; 6] | 0,344 · 0,359 · 0,339 · 0,349 · 0,328 [2 / 4; 6] | 0,120 · 0,119 · 0,114 · 0,121 · 0,113 [2 / 4; 6] |
| Generales todos | 2,892 · 2,589 · 2,627 · 2,777 · 2,753 [17 / 11; 28] | 7,350 · 7,168 · 7,219 · 7,403 · 7,247 [13 / 9; 22] | 0,242 · 0,282 · 0,252 · 0,252 · 0,236 [7 / 15; 22] | 0,076 · 0,083 · 0,075 · 0,078 · 0,073 [7 / 15; 22] |
| Autonómicas 6-14 | 2,340 · 2,238 · 2,235 · 2,273 · 2,293 [15 / 7; 22] | 1,788 · 1,780 · 1,784 · 1,777 · 1,795 [10 / 12; 22] | 0,106 · 0,129 · 0,116 · 0,111 · 0,106 [1 / 13; 22] | 0,031 · 0,036 · 0,033 · 0,032 · 0,031 [1 / 15; 22] |
| Autonómicas 30-60 | 2,979 · 2,724 · 2,802 · 2,909 · 3,050 [19 / 3; 22] | 2,739 · 2,676 · 2,709 · 2,721 · 2,744 [13 / 9; 22] | 0,064 · 0,090 · 0,073 · 0,068 · 0,063 [2 / 14; 22] | 0,013 · 0,019 · 0,015 · 0,014 · 0,012 [2 / 14; 22] |
| Autonómicas 90-180 | 3,354 · 2,941 · 3,139 · 3,273 · 3,523 [17 / 5; 22] | 3,571 · 3,484 · 3,531 · 3,562 · 3,583 [16 / 6; 22] | 0,060 · 0,083 · 0,067 · 0,063 · 0,060 [0 / 14; 22] | 0,015 · 0,020 · 0,016 · 0,016 · 0,015 [0 / 14; 22] |
| Autonómicas todos | 2,891 · 2,634 · 2,725 · 2,818 · 2,956 [51 / 15; 66] | 2,699 · 2,647 · 2,675 · 2,687 · 2,708 [39 / 27; 66] | 0,077 · 0,101 · 0,085 · 0,081 · 0,076 [3 / 41; 66] | 0,019 · 0,025 · 0,021 · 0,021 · 0,019 [3 / 43; 66] |

### Dejando fuera cada elección (punto 14)

Cada celda es el log-score medio del voto del candidato en los casos de las otras cuatro elecciones (o de los otros cuatro ámbitos), y en negrita el que gana la fila. A la derecha, el candidato que ganaría la fila con el CRPS de escaños y con el log-score de las mayorías. Debajo de cada tabla, la diferencia con 5 elección a elección (o ámbito a ámbito): la media, la desviación entre las cinco y en cuántas gana a 5. Una diferencia media menor que esa desviación cuenta como ruido. En las generales, 2015 no tiene casos con escaños válidos. En su fila, las dos columnas de la derecha salen por eso de las cinco elecciones, con las mismas cifras que con todas.

Generales, con horizonte:

| Fuera | None | 3 | 5 | 8 | 15 | Gana | CRPS escaños, gana | Log-score mayorías, gana |
|---|---|---|---|---|---|---|---|---|
| 2015-12 | 2,650 | 2,568 | **2,525** | 2,592 | 2,549 | 5 | 15 | None |
| 2016-06 | 2,705 | 2,544 | **2,522** | 2,616 | 2,596 | 5 | 5 | 15 |
| 2019-04 | 2,661 | 2,555 | **2,543** | 2,628 | 2,584 | 5 | 15 | 15 |
| 2019-11 | 2,627 | 2,544 | **2,530** | 2,581 | 2,537 | 5 | 15 | None |
| 2023-07 | 2,668 | **2,535** | 2,554 | 2,626 | 2,578 | 3 | 15 | None |

None +0,124 ± 0,150 (1 de 5); 3 +0,014 ± 0,082 (2 de 5); 8 +0,074 ± 0,075 (1 de 5); 15 +0,032 ± 0,116 (1 de 5).

Generales, nowcast:

| Fuera | None | 3 | 5 | 8 | 15 | Gana | CRPS escaños, gana | Log-score mayorías, gana |
|---|---|---|---|---|---|---|---|---|
| 2015-12 | 2,875 | **2,601** | 2,607 | 2,763 | 2,728 | 3 | 3 | 15 |
| 2016-06 | 2,997 | **2,583** | 2,640 | 2,807 | 2,819 | 3 | 3 | 15 |
| 2019-04 | 2,880 | **2,599** | 2,639 | 2,811 | 2,775 | 3 | 3 | 15 |
| 2019-11 | 2,841 | **2,578** | 2,616 | 2,739 | 2,701 | 3 | 3 | 15 |
| 2023-07 | 2,864 | **2,586** | 2,633 | 2,766 | 2,744 | 3 | 3 | None |

None +0,257 ± 0,241 (1 de 5); 3 −0,035 ± 0,075 (4 de 5); 8 +0,152 ± 0,089 (0 de 5); 15 +0,125 ± 0,157 (1 de 5).

Autonómicas, con horizonte:

| Fuera | None | 3 | 5 | 8 | 15 | Gana | CRPS escaños, gana | Log-score mayorías, gana |
|---|---|---|---|---|---|---|---|---|
| Andalucía | 2,686 | **2,621** | 2,647 | 2,678 | 2,726 | 3 | 3 | 15 |
| Cataluña | 2,667 | **2,601** | 2,628 | 2,642 | 2,705 | 3 | 8 | 15 |
| Galicia | 2,459 | **2,454** | 2,455 | 2,455 | 2,488 | 3 | 8 | 15 |
| Madrid | 2,618 | **2,576** | 2,600 | 2,617 | 2,624 | 3 | 3 | 15 |
| C. Valenciana | 2,690 | **2,602** | 2,637 | 2,666 | 2,724 | 3 | 3 | 15 |

None +0,028 ± 0,085 (3 de 5); 3 −0,023 ± 0,056 (4 de 5); 8 +0,018 ± 0,057 (2 de 5); 15 +0,051 ± 0,113 (3 de 5).

Autonómicas, nowcast:

| Fuera | None | 3 | 5 | 8 | 15 | Gana | CRPS escaños, gana | Log-score mayorías, gana |
|---|---|---|---|---|---|---|---|---|
| Andalucía | 3,009 | **2,710** | 2,828 | 2,936 | 3,101 | 3 | 3 | 15 |
| Cataluña | 2,936 | **2,678** | 2,772 | 2,861 | 3,029 | 3 | 3 | 15 |
| Galicia | 2,743 | **2,522** | 2,591 | 2,649 | 2,794 | 3 | 3 | 15 |
| Madrid | 2,741 | **2,584** | 2,620 | 2,717 | 2,745 | 3 | 3 | 15 |
| C. Valenciana | 3,010 | **2,673** | 2,804 | 2,916 | 3,085 | 3 | 3 | 15 |

None +0,154 ± 0,116 (1 de 5); 3 −0,076 ± 0,139 (3 de 5); 8 +0,094 ± 0,097 (0 de 5); 15 +0,202 ± 0,220 (1 de 5).

### Decisión (punto 14 y criterio 17)

**Por la regla del punto 14 sale 5, y por ruido.** Con horizonte, 5 gana cuatro de las cinco filas de las generales (3 gana la de 2023) y 3 gana las cinco de las autonómicas. En el nowcast, 3 gana todas las filas en las dos poblaciones. Pero ningún candidato mejora a 5 más allá del ruido. 3, el único que le gana filas, queda en +0,014 ± 0,082 frente a 5 en las generales con horizonte y en −0,023 ± 0,056 en las autonómicas. En el nowcast queda en −0,035 ± 0,075 y −0,076 ± 0,139.

Al revés, en el nowcast 5 sí mejora a algunos más allá del ruido: a `None` en las generales (+0,257 ± 0,241) y en las autonómicas (+0,154 ± 0,116), y a 8 en las generales (+0,152 ± 0,089). Con horizonte, ninguna diferencia con 5 sale del ruido (`None`, +0,124 ± 0,150 y +0,028 ± 0,085). Ningún candidato mejora a 5 más allá del ruido, así que la especificación dice 5, y lo decimos.

**El criterio 17 no se cumple con 3, 5 ni 8.** Lo que el log-score del voto premia a 30 días o más es sobre todo el ancho. El nowcast no lleva deriva y se queda corto a esos horizontes; cualquier ensanchamiento le ayuda, y por eso 3 gana todas sus filas. Al lado empeoran tres cosas:
- La probabilidad de mayoría absoluta empeora más allá del ruido en las autonómicas.
  - En los dos modos y en todos los grupos, y más cuanto menos grados. Con horizonte y 5 grados, el log-score de las mayorías pasa de 0,100 a 0,110 (+10 %), con 3 casos mejor y 46 peor de 66; el Brier, de 0,025 a 0,027, con 3 y 49. En el nowcast, el log-score pasa de 0,077 a 0,085, con 3 y 41.
  - En las generales con horizonte empeora en la media de los tres grupos: de 0,260 a 0,277 en total, con 5 casos mejor y 17 peor de 22. A 90-180 días son 2 y 4 de 6, dentro del ruido.
  - En el nowcast de las generales empeora a 6-14 días (de 0,149 a 0,176, 2 y 6 de 8). A 30 días o más queda plano o algo mejor: el log-score, de 0,259 a 0,263 a 30-60 días y de 0,344 a 0,339 a 90-180; el Brier, de 0,082 a 0,079 y de 0,120 a 0,114 (de 0,076 a 0,075 en total).
  - Con 3 grados y horizonte, el log-score de las mayorías sube a 0,129 en las autonómicas y a 0,309 en las generales.
- Donde ya cubrían, a 6-14 días, las bandas pasan del nominal. En las generales con horizonte, la del 80 % de escaños va de 0,79 a 0,93 con 5 grados (0,98 con 3) y la del 95 % del voto de 0,94 a 0,98 (1,00). Las colas del PIT de escaños se vacían: de 0,26 a 0,13 con 5 grados y a 0,02 con 3, frente a un nominal de 0,20.
- El CRPS de escaños no mejora más allá del ruido. Con horizonte y 5 grados, 12 casos mejor y 10 peor en las generales, y 33 y 33 en las autonómicas.

15 grados queda cerca de `None` en las generales, donde los dos son casi una normal. Coberturas y PIT apenas se mueven, y el log-score de las mayorías queda igual (0,260 con horizonte).

En las autonómicas no es neutro. A 90-180 días, en 12 de los 22 casos los grados de `None` no pasaban de 15, y ahí 15 da un sorteo más estrecho. A 90-180 días:
- con horizonte, el log-score del voto empeora de 2,871 a 2,945, con 9 casos mejor y 13 peor;
- con horizonte, la cobertura del 95 % del voto baja de 0,79 a 0,77;
- en el nowcast, el log-score del voto pasa de 3,354 a 3,523.

El log-score de las mayorías mejora algo (de 0,100 a 0,097 en total con horizonte, 29 casos mejor y 17 peor). Tampoco gana nada en el voto.

`tail_dof` queda como opción documentada; `None` sigue siendo el valor por defecto y nada cambia en el modelo vivo. **La decisión es de Luis.** Lo que el backtest señala:

- Por la regla de la especificación sale 5, porque ningún candidato le mejora más allá del ruido en el log-score del voto. Con el resto a la vista (mayorías, bandas cortas que se pasan, colas del PIT vacías), 5 no pasa el criterio 17 en esta configuración.
- Esta prueba mezcla colas y ancho, porque `σ` es la escala de la t. Para probar sólo las colas, el sorteo tendría que usar `σ · sqrt((ν − 2) / ν) · t(ν)`, con la misma varianza que la normal y sólo otra forma. Es un cambio del sorteo, no de esta opción, y queda propuesto.
- El ensanchamiento que el voto agradece a 30 días o más ya lo pedía la sección anterior. Es el mismo hueco del horizonte, y una t más ancha no lo arregla mejor que la escala de la deriva: empeora también las mayorías.

Límites de la prueba:
- Las autonómicas son cinco de los diecisiete ámbitos, los que más casos tienen; las otras doce quedan para una ejecución completa de la configuración que se elija.
- La tabla `average_errors` está vacía en la base: el backtest la leyó de un CSV calculado en memoria con el mismo código y la semántica de `load_average_errors` (sólo las elecciones anteriores a cada caso).
- La prueba es acumulativa: se hizo sobre `error_model='average'` con `drift_scale='auto'`, la configuración de las dos secciones anteriores, que no cumplió el criterio 17. Si el modelo vivo sigue con `error_model='polls'` y `drift_scale=None`, este resultado puede no trasladarse y habría que repetir la comparación sobre esa configuración.
- El log-score del voto sale de los cuantiles exportados, no de las simulaciones (ver «Cómo se midió»).

Tiempos (1.000 simulaciones, en un Mac con cuatro ejecuciones a la vez): generales entre 3.082 y 3.200 s por candidato; autonómicas, 66 casos por candidato, entre 843 y 1.105 s por ámbito (entre 4.653 y 4.960 s en total).

## La correlación entre partidos (`composition`)

Cambios: ninguno en el código. `Simulator(composition=None | 'auto' | r)`, `Computer.get_composition_ratio` y `--composition auto` existen desde el método 7 (`docs/analisis-2026-09/metodo-7-composicion.md`). Este paso vuelve a medir la opción sobre la escala y las colas de las dos secciones anteriores.

Tests: ninguno nuevo; los de M7 siguen en `tests/test_simulator_unit.py`, `tests/test_computer_unit.py` y `tests/integration/test_backtest.py`.

**Qué cambia.** Con `composition='auto'`, los partidos nacionales (`regional = 0`) se sortean juntos, con una correlación común negativa `ρ`. `ρ` sale de una sola razón, `r = Var(Σ error) / Σ Var(error)`, estimada con el sesgo del sector por partido de las elecciones anteriores (`Computer.get_composition_ratio`), así que en el backtest está fuera de muestra. La cópula gaussiana conserva la distribución de cada partido. Lo que cambia es la estructura del sorteo: la suma de todos los nacionales se contrae hacia `√r` y las diferencias entre partidos se ensanchan. Los regionales siguen sorteándose por separado.

**Por qué se repite.** La sección de la escala dejó una hipótesis. La deriva de cada partido se sortea por separado, así que con bandas más anchas la suma de un bloque se abre y su probabilidad de mayoría se va hacia el medio. La composición del ruido debía evitar que la deriva de un bloque creciera como la de partidos independientes. El método 7 dejó la composición desactivada con un criterio fijado de antemano: **entra si no empeora el CRPS de escaños, el Brier ni el log-score de las mayorías a 6-14 días.** Aquí se juzga con ese criterio y con el 17.

**Cómo se midió.** Sobre las dos configuraciones de las secciones anteriores, cada una frente a sí misma sin composición:
- (a) `error_model='average'`, `drift_scale='auto'` y `tail_dof=5`: de `backtest/results_dof5` a `backtest/results_comp2`;
- (b) lo mismo con los grados de la regresión (`tail_dof=None`): de `backtest/results_avg_scale2` a `backtest/results_comp2_none`.

No se repitió ningún «antes». 1.000 simulaciones, semilla 42, y la tabla `average_errors` leída del mismo CSV calculado en memoria (sigue vacía en la base). Las generales son los 28 casos de `es` (22 con escaños válidos; 2015 no tiene ninguno). Las autonómicas son los cinco ámbitos fijos de las secciones anteriores: Madrid, Andalucía, Cataluña, Comunitat Valenciana y Galicia, 11 elecciones y 66 casos. `meta.csv` confirma la configuración de cada caso: `error_model` `average` en todos, la misma escala de la deriva (entre 2,22 y 4,00 en `es`, entre 2,22 y 2,49 en los ámbitos), los grados pedidos, y `composition_ratio` 1 con `ρ` 0 en los dos «antes». La composición entra en los dos sorteos, el nowcast y el de horizonte, así que se tabulan los dos modos.

**La razón ajustada.** `r` y `ρ` son las mismas en las dos configuraciones, caso a caso: `r` no depende de `e`, de la deriva ni de los grados, y `ρ` es la del último sorteo del nowcast (`composition_rho` de `meta.csv`), que no lleva deriva ni depende de los grados. `ρ` depende de los márgenes de cada partido y cambia algo con el horizonte.

| Elección simulada | 2015-12 | 2016-06 | 2019-04 | 2019-11 | 2023-07 |
|---|---|---|---|---|---|
| `r` | 0,140 | 0,082 | 0,109 | 0,207 | 0,160 |
| `ρ` | −0,18 | −0,24 | −0,18 a −0,19 | −0,16 a −0,17 | −0,29 |

Cada autonómica toma la `r` de `es` con las generales anteriores a ella, como la escala: entre 0,109 y 0,223 en los cinco ámbitos. Su `ρ` va de −0,105 a −0,291 (la mediana por ámbito, de −0,12 en Cataluña a −0,20 en Galicia). El resto recortado (`clip_rate`, la fracción de sorteos en que los partidos suman más de 100) baja en todos los ámbitos: en `es`, de 0,29 a 0,11 de media en (a) y de 0,26 a 0,07 en (b); en las autonómicas, de entre 0,17 y 0,30 por ámbito a entre 0,02 y 0,16.

### (a) Sobre `tail_dof=5`

Coberturas de los partidos principales, media por grupo de horizontes ponderada por casos, antes → después. Con horizonte:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,54 → 0,54 | 0,77 → 0,77 | 0,98 → 0,98 | 0,63 → 0,60 | 0,93 → 0,93 | 1,00 → 1,00 |
| Generales 30-60 (10) | 0,45 → 0,45 | 0,75 → 0,75 | 0,86 → 0,86 | 0,44 → 0,47 | 0,83 → 0,83 | 0,93 → 0,96 |
| Generales 90-180 (8) | 0,41 → 0,38 | 0,75 → 0,72 | 0,95 → 0,95 | 0,44 → 0,44 | 0,78 → 0,78 | 0,88 → 0,94 |
| Autonómicas 6-14 (22) | 0,47 → 0,50 | 0,76 → 0,76 | 0,94 → 0,93 | 0,61 → 0,65 | 0,77 → 0,80 | 0,93 → 0,92 |
| Autonómicas 30-60 (22) | 0,38 → 0,39 | 0,66 → 0,67 | 0,88 → 0,88 | 0,53 → 0,54 | 0,72 → 0,75 | 0,89 → 0,92 |
| Autonómicas 90-180 (22) | 0,37 → 0,40 | 0,67 → 0,68 | 0,84 → 0,86 | 0,50 → 0,50 | 0,75 → 0,79 | 0,87 → 0,92 |

Nowcast:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,52 → 0,52 | 0,72 → 0,72 | 0,98 → 0,96 | 0,57 → 0,54 | 0,78 → 0,81 | 1,00 → 1,00 |
| Generales 30-60 (10) | 0,37 → 0,38 | 0,68 → 0,68 | 0,84 → 0,84 | 0,31 → 0,35 | 0,75 → 0,75 | 0,90 → 0,90 |
| Generales 90-180 (8) | 0,30 → 0,30 | 0,58 → 0,58 | 0,80 → 0,80 | 0,27 → 0,27 | 0,53 → 0,57 | 0,81 → 0,87 |
| Autonómicas 6-14 (22) | 0,47 → 0,49 | 0,75 → 0,74 | 0,92 → 0,93 | 0,58 → 0,62 | 0,76 → 0,76 | 0,92 → 0,92 |
| Autonómicas 30-60 (22) | 0,32 → 0,32 | 0,58 → 0,59 | 0,80 → 0,83 | 0,45 → 0,48 | 0,66 → 0,66 | 0,83 → 0,84 |
| Autonómicas 90-180 (22) | 0,30 → 0,30 | 0,51 → 0,51 | 0,73 → 0,75 | 0,38 → 0,41 | 0,60 → 0,63 | 0,79 → 0,82 |

Escaños de los bloques (`blocks.csv`: la Izquierda y la Derecha de cada caso con escaños válidos). Cobertura con horizonte y ancho medio de la banda del 80 %, en escaños:

| Grupo (filas) | Bloque 50 %, horizonte | Bloque 80 %, horizonte | Bloque 95 %, horizonte | Ancho 80 %, horizonte | Ancho 80 %, nowcast |
|---|---|---|---|---|---|
| Generales 6-14 (16) | 0,62 → 0,81 | 1,00 → 1,00 | 1,00 → 1,00 | 34,0 → 37,5 | 32,0 → 35,0 |
| Generales 30-60 (16) | 0,50 → 0,56 | 0,88 → 0,94 | 1,00 → 1,00 | 40,8 → 44,7 | 32,0 → 34,9 |
| Generales 90-180 (12) | 0,50 → 0,50 | 0,92 → 0,92 | 1,00 → 1,00 | 47,4 → 53,3 | 31,2 → 34,5 |
| Autonómicas 6-14 (44) | 0,57 → 0,61 | 0,91 → 0,89 | 0,93 → 0,93 | 9,7 → 9,9 | 9,2 → 9,5 |
| Autonómicas 30-60 (44) | 0,57 → 0,55 | 0,75 → 0,80 | 0,91 → 0,93 | 12,2 → 12,7 | 10,0 → 10,4 |
| Autonómicas 90-180 (44) | 0,55 → 0,66 | 0,86 → 0,89 | 0,95 → 0,95 | 16,0 → 16,9 | 10,2 → 10,7 |

PIT de los partidos principales (`summarize_calibration(main_only=True)`; colas = las cajas 0 y 9, nominal 0,20; L1 = distancia a la uniforme). Entre paréntesis, las elecciones del voto y de los escaños. Con horizonte:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,23 → 0,23 | 0,30 → 0,27 | 0,13 → 0,12 | 0,46 → 0,45 |
| Generales 30-60 (5 / 4) | 0,25 → 0,25 | 0,28 → 0,28 | 0,17 → 0,17 | 0,50 → 0,42 |
| Generales 90-180 (5 / 4) | 0,25 → 0,27 | 0,48 → 0,42 | 0,23 → 0,23 | 0,38 → 0,44 |
| Autonómicas 6-14 (11 / 11) | 0,24 → 0,24 | 0,22 → 0,24 | 0,26 → 0,24 | 0,27 → 0,29 |
| Autonómicas 30-60 (11 / 11) | 0,32 → 0,30 | 0,27 → 0,26 | 0,31 → 0,31 | 0,30 → 0,24 |
| Autonómicas 90-180 (11 / 11) | 0,28 → 0,27 | 0,27 → 0,27 | 0,29 → 0,28 | 0,32 → 0,30 |

Nowcast:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,28 → 0,28 | 0,30 → 0,35 | 0,24 → 0,22 | 0,45 → 0,40 |
| Generales 30-60 (5 / 4) | 0,32 → 0,32 | 0,57 → 0,46 | 0,33 → 0,25 | 0,53 → 0,42 |
| Generales 90-180 (5 / 4) | 0,43 → 0,43 | 0,69 → 0,69 | 0,47 → 0,44 | 0,67 → 0,67 |
| Autonómicas 6-14 (11 / 11) | 0,25 → 0,26 | 0,22 → 0,22 | 0,27 → 0,27 | 0,29 → 0,36 |
| Autonómicas 30-60 (11 / 11) | 0,40 → 0,39 | 0,41 → 0,45 | 0,36 → 0,34 | 0,37 → 0,31 |
| Autonómicas 90-180 (11 / 11) | 0,46 → 0,45 | 0,52 → 0,51 | 0,44 → 0,40 | 0,52 → 0,49 |

Puntuaciones por caso. Más bajo es mejor; entre paréntesis, casos mejor / peor. En la primera columna, los casos con escaños válidos / con voto. El log-score del voto es el de la sección anterior, reconstruido con los cuantiles exportados. El Brier va con cuatro decimales.

| Grupo | CRPS escaños, nowcast | CRPS escaños, horizonte | Log-score mayorías, nowcast | Log-score mayorías, horizonte | Brier mayorías, nowcast | Brier mayorías, horizonte | Log-score voto, nowcast | Log-score voto, horizonte |
|---|---|---|---|---|---|---|---|---|
| Generales 6-14 (8 / 10) | 5,231 → 5,221 (3 / 5) | 5,208 → 5,209 (3 / 5) | 0,176 → 0,194 (0 / 8) | 0,187 → 0,206 (0 / 8) | 0,0420 → 0,0461 (0 / 8) | 0,0449 → 0,0491 (0 / 8) | 2,236 → 2,202 (8 / 2) | 2,236 → 2,236 (7 / 3) |
| Generales 30-60 (8 / 10) | 7,565 → 7,464 (5 / 3) | 7,396 → 7,347 (5 / 3) | 0,263 → 0,274 (3 / 5) | 0,297 → 0,305 (4 / 4) | 0,0789 → 0,0815 (3 / 5) | 0,0881 → 0,0900 (4 / 4) | 2,765 → 2,667 (7 / 3) | 2,607 → 2,603 (5 / 5) |
| Generales 90-180 (6 / 8) | 9,407 → 9,330 (4 / 2) | 9,071 → 9,134 (2 / 4) | 0,339 → 0,341 (3 / 3) | 0,370 → 0,375 (3 / 3) | 0,1144 → 0,1132 (3 / 3) | 0,1201 → 0,1195 (3 / 3) | 2,943 → 2,936 (6 / 2) | 2,817 → 2,810 (5 / 3) |
| Generales todos (22 / 28) | 7,219 → 7,157 (12 / 10) | 7,057 → 7,057 (10 / 12) | 0,252 → 0,263 (6 / 16) | 0,277 → 0,288 (7 / 15) | 0,0752 → 0,0773 (6 / 16) | 0,0811 → 0,0832 (7 / 15) | 2,627 → 2,578 (21 / 7) | 2,535 → 2,531 (17 / 11) |
| Autonómicas 6-14 (22 / 22) | 1,784 → 1,778 (14 / 8) | 1,776 → 1,770 (14 / 8) | 0,116 → 0,117 (5 / 11) | 0,120 → 0,121 (7 / 9) | 0,0329 → 0,0332 (6 / 11) | 0,0339 → 0,0342 (7 / 11) | 2,235 → 2,243 (8 / 14) | 2,243 → 2,232 (13 / 9) |
| Autonómicas 30-60 (22 / 22) | 2,709 → 2,686 (16 / 6) | 2,629 → 2,614 (15 / 7) | 0,073 → 0,076 (3 / 13) | 0,095 → 0,098 (8 / 8) | 0,0147 → 0,0152 (3 / 13) | 0,0209 → 0,0210 (9 / 9) | 2,802 → 2,774 (11 / 11) | 2,663 → 2,620 (15 / 7) |
| Autonómicas 90-180 (22 / 22) | 3,531 → 3,507 (13 / 9) | 3,383 → 3,367 (12 / 10) | 0,067 → 0,070 (2 / 14) | 0,115 → 0,118 (6 / 10) | 0,0164 → 0,0167 (2 / 14) | 0,0269 → 0,0270 (8 / 11) | 3,139 → 3,082 (13 / 9) | 2,874 → 2,799 (18 / 4) |
| Autonómicas todos (66 / 66) | 2,675 → 2,657 (43 / 23) | 2,596 → 2,583 (41 / 25) | 0,085 → 0,088 (10 / 38) | 0,110 → 0,112 (21 / 27) | 0,0214 → 0,0217 (11 / 38) | 0,0272 → 0,0274 (24 / 31) | 2,725 → 2,700 (32 / 34) | 2,593 → 2,550 (46 / 20) |

El ruido se mide como en la sección anterior: la diferencia media por elección (generales) o por ámbito (autonómicas), ± su desviación entre ellos, y entre paréntesis en cuántos mejora. Una diferencia media menor que esa desviación cuenta como ruido.
- Generales a 6-14 días: CRPS de escaños −0,009 ± 0,207 (1 de 4) en el nowcast y +0,001 ± 0,209 (1 de 4) con horizonte; log-score de las mayorías +0,018 ± 0,009 y +0,018 ± 0,010 (0 de 4 en los dos); Brier +0,0042 ± 0,0020 y +0,0042 ± 0,0022 (0 de 4 en los dos).
- Autonómicas a 6-14 días: CRPS de escaños −0,008 ± 0,016 (4 de 5) y −0,008 ± 0,017 (3 de 5); log-score de las mayorías +0,001 ± 0,003 (1 de 5) en los dos; Brier +0,0001 ± 0,0014 (1 de 5) y +0,0001 ± 0,0015 (2 de 5).
- Todos los casos: CRPS de escaños −0,072 ± 0,188 y −0,012 ± 0,172 en las generales, −0,018 ± 0,029 y −0,012 ± 0,032 en las autonómicas; log-score del voto −0,051 ± 0,040 (5 de 5) y −0,008 ± 0,091 (3 de 5) en las generales, −0,025 ± 0,025 (4 de 5) y −0,047 ± 0,037 (5 de 5) en las autonómicas.

El centro no se mueve: el MAE del voto es idéntico en los 94 casos. Las bandas de escaños se acercan al nominal donde faltaban: con horizonte, la del 95 % pasa de 0,88 a 0,94 en las generales a 90-180 días y de 0,87 a 0,92 en las autonómicas. En el nowcast de las generales, las colas del PIT de escaños bajan de 0,33 a 0,25 a 30-60 días. El CRPS de escaños mejora en la mayoría de los casos de las autonómicas (43 mejor y 23 peor en el nowcast, 41 y 25 con horizonte), dentro del ruido. El log-score del voto mejora más allá del ruido en el nowcast de las generales y con horizonte en las autonómicas. La cópula conserva la distribución de cada partido, así que el voto sólo cambia porque el resto ya casi no se recorta y el sorteo es otro.

A 6-14 días, las mayorías empeoran. En las generales empeoran en los 8 casos y en los dos modos, más allá del ruido: el log-score pasa de 0,176 a 0,194 en el nowcast y de 0,187 a 0,206 con horizonte. En las autonómicas empeoran algo y dentro del ruido (de 0,116 a 0,117 y de 0,120 a 0,121). Y donde ya pasaba del nominal, la banda del 50 % de escaños de las autonómicas a 6-14 días pasa más: de 0,61 a 0,65 con horizonte y de 0,58 a 0,62 en el nowcast.

### (b) Sobre `tail_dof=None`

Coberturas, antes → después. Con horizonte:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,51 → 0,54 | 0,75 → 0,72 | 0,94 → 0,96 | 0,54 → 0,57 | 0,79 → 0,85 | 0,98 → 1,00 |
| Generales 30-60 (10) | 0,43 → 0,45 | 0,75 → 0,75 | 0,84 → 0,84 | 0,44 → 0,44 | 0,78 → 0,78 | 0,88 → 0,93 |
| Generales 90-180 (8) | 0,38 → 0,38 | 0,70 → 0,70 | 0,85 → 0,85 | 0,41 → 0,44 | 0,70 → 0,75 | 0,87 → 0,91 |
| Autonómicas 6-14 (22) | 0,47 → 0,50 | 0,73 → 0,73 | 0,86 → 0,88 | 0,57 → 0,60 | 0,75 → 0,75 | 0,88 → 0,89 |
| Autonómicas 30-60 (22) | 0,36 → 0,35 | 0,66 → 0,64 | 0,80 → 0,81 | 0,50 → 0,52 | 0,69 → 0,70 | 0,84 → 0,87 |
| Autonómicas 90-180 (22) | 0,34 → 0,37 | 0,67 → 0,66 | 0,79 → 0,79 | 0,47 → 0,49 | 0,73 → 0,74 | 0,87 → 0,87 |

Nowcast:

| Grupo (casos) | Voto 50 % | Voto 80 % | Voto 95 % | Escaños 50 % | Escaños 80 % | Escaños 95 % |
|---|---|---|---|---|---|---|
| Generales 6-14 (10) | 0,47 → 0,49 | 0,72 → 0,70 | 0,92 → 0,96 | 0,49 → 0,54 | 0,73 → 0,73 | 0,98 → 1,00 |
| Generales 30-60 (10) | 0,34 → 0,34 | 0,57 → 0,52 | 0,80 → 0,82 | 0,28 → 0,31 | 0,55 → 0,63 | 0,84 → 0,88 |
| Generales 90-180 (8) | 0,27 → 0,27 | 0,47 → 0,49 | 0,75 → 0,75 | 0,23 → 0,23 | 0,53 → 0,53 | 0,72 → 0,76 |
| Autonómicas 6-14 (22) | 0,46 → 0,48 | 0,73 → 0,70 | 0,86 → 0,88 | 0,55 → 0,58 | 0,73 → 0,73 | 0,87 → 0,89 |
| Autonómicas 30-60 (22) | 0,27 → 0,30 | 0,55 → 0,57 | 0,74 → 0,76 | 0,44 → 0,44 | 0,64 → 0,66 | 0,78 → 0,80 |
| Autonómicas 90-180 (22) | 0,29 → 0,29 | 0,47 → 0,48 | 0,69 → 0,70 | 0,37 → 0,39 | 0,57 → 0,60 | 0,74 → 0,79 |

Escaños de los bloques:

| Grupo (filas) | Bloque 50 %, horizonte | Bloque 80 %, horizonte | Bloque 95 %, horizonte | Ancho 80 %, horizonte | Ancho 80 %, nowcast |
|---|---|---|---|---|---|
| Generales 6-14 (16) | 0,62 → 0,62 | 1,00 → 1,00 | 1,00 → 1,00 | 29,2 → 31,9 | 27,2 → 29,9 |
| Generales 30-60 (16) | 0,56 → 0,56 | 0,88 → 0,88 | 1,00 → 1,00 | 34,5 → 38,0 | 27,2 → 29,6 |
| Generales 90-180 (12) | 0,33 → 0,50 | 0,92 → 0,92 | 1,00 → 1,00 | 39,6 → 45,0 | 26,0 → 29,1 |
| Autonómicas 6-14 (44) | 0,52 → 0,57 | 0,84 → 0,89 | 0,91 → 0,93 | 8,3 → 8,8 | 7,9 → 8,3 |
| Autonómicas 30-60 (44) | 0,52 → 0,55 | 0,68 → 0,70 | 0,91 → 0,91 | 10,7 → 11,3 | 8,8 → 9,2 |
| Autonómicas 90-180 (44) | 0,50 → 0,57 | 0,84 → 0,84 | 0,93 → 0,93 | 14,4 → 15,2 | 9,2 → 9,8 |

PIT de los partidos principales. Con horizonte:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,25 → 0,28 | 0,29 → 0,39 | 0,26 → 0,21 | 0,34 → 0,33 |
| Generales 30-60 (5 / 4) | 0,25 → 0,25 | 0,32 → 0,28 | 0,22 → 0,22 | 0,35 → 0,29 |
| Generales 90-180 (5 / 4) | 0,29 → 0,29 | 0,45 → 0,38 | 0,29 → 0,26 | 0,50 → 0,45 |
| Autonómicas 6-14 (11 / 11) | 0,27 → 0,27 | 0,25 → 0,28 | 0,29 → 0,28 | 0,35 → 0,35 |
| Autonómicas 30-60 (11 / 11) | 0,32 → 0,34 | 0,32 → 0,33 | 0,34 → 0,33 | 0,32 → 0,29 |
| Autonómicas 90-180 (11 / 11) | 0,27 → 0,29 | 0,33 → 0,27 | 0,30 → 0,30 | 0,34 → 0,33 |

Nowcast:

| Grupo (elecciones) | Voto, colas | Voto, L1 | Escaños, colas | Escaños, L1 |
|---|---|---|---|---|
| Generales 6-14 (5 / 4) | 0,28 → 0,30 | 0,36 → 0,31 | 0,29 → 0,29 | 0,49 → 0,53 |
| Generales 30-60 (5 / 4) | 0,43 → 0,48 | 0,52 → 0,59 | 0,48 → 0,40 | 0,60 → 0,64 |
| Generales 90-180 (5 / 4) | 0,54 → 0,51 | 0,70 → 0,66 | 0,53 → 0,47 | 0,77 → 0,67 |
| Autonómicas 6-14 (11 / 11) | 0,27 → 0,30 | 0,25 → 0,29 | 0,29 → 0,29 | 0,36 → 0,35 |
| Autonómicas 30-60 (11 / 11) | 0,43 → 0,41 | 0,52 → 0,44 | 0,40 → 0,36 | 0,40 → 0,36 |
| Autonómicas 90-180 (11 / 11) | 0,50 → 0,48 | 0,60 → 0,57 | 0,50 → 0,44 | 0,59 → 0,54 |

Puntuaciones por caso:

| Grupo | CRPS escaños, nowcast | CRPS escaños, horizonte | Log-score mayorías, nowcast | Log-score mayorías, horizonte | Brier mayorías, nowcast | Brier mayorías, horizonte | Log-score voto, nowcast | Log-score voto, horizonte |
|---|---|---|---|---|---|---|---|---|
| Generales 6-14 (8 / 10) | 5,191 → 5,208 (3 / 5) | 5,129 → 5,165 (3 / 5) | 0,149 → 0,162 (2 / 6) | 0,163 → 0,176 (2 / 6) | 0,0381 → 0,0395 (2 / 6) | 0,0407 → 0,0430 (2 / 6) | 2,258 → 2,251 (6 / 4) | 2,226 → 2,244 (5 / 5) |
| Generales 30-60 (8 / 10) | 7,697 → 7,577 (6 / 2) | 7,427 → 7,355 (4 / 4) | 0,259 → 0,255 (3 / 5) | 0,282 → 0,285 (4 / 4) | 0,0822 → 0,0780 (3 / 5) | 0,0862 → 0,0855 (4 / 4) | 3,047 → 2,945 (6 / 4) | 2,807 → 2,708 (6 / 4) |
| Generales 90-180 (6 / 8) | 9,767 → 9,548 (6 / 0) | 9,301 → 9,154 (5 / 1) | 0,344 → 0,332 (3 / 3) | 0,360 → 0,354 (3 / 3) | 0,1201 → 0,1126 (3 / 3) | 0,1192 → 0,1142 (3 / 3) | 3,490 → 3,366 (5 / 3) | 3,026 → 2,895 (5 / 3) |
| Generales todos (22 / 28) | 7,350 → 7,253 (15 / 7) | 7,102 → 7,049 (12 / 10) | 0,242 → 0,242 (8 / 14) | 0,260 → 0,264 (9 / 13) | 0,0765 → 0,0734 (8 / 14) | 0,0787 → 0,0779 (9 / 13) | 2,892 → 2,817 (17 / 11) | 2,662 → 2,596 (16 / 12) |
| Autonómicas 6-14 (22 / 22) | 1,788 → 1,794 (11 / 11) | 1,778 → 1,784 (11 / 11) | 0,106 → 0,107 (3 / 10) | 0,109 → 0,112 (3 / 10) | 0,0308 → 0,0310 (3 / 12) | 0,0316 → 0,0322 (3 / 11) | 2,340 → 2,365 (9 / 13) | 2,304 → 2,324 (10 / 12) |
| Autonómicas 30-60 (22 / 22) | 2,739 → 2,723 (13 / 9) | 2,654 → 2,636 (12 / 10) | 0,064 → 0,066 (4 / 11) | 0,086 → 0,086 (7 / 9) | 0,0126 → 0,0128 (4 / 12) | 0,0186 → 0,0179 (7 / 9) | 2,979 → 2,992 (11 / 11) | 2,698 → 2,724 (9 / 13) |
| Autonómicas 90-180 (22 / 22) | 3,571 → 3,536 (14 / 8) | 3,405 → 3,374 (14 / 8) | 0,060 → 0,064 (2 / 11) | 0,104 → 0,106 (5 / 11) | 0,0149 → 0,0155 (2 / 11) | 0,0242 → 0,0242 (5 / 11) | 3,354 → 3,350 (12 / 10) | 2,871 → 2,861 (10 / 12) |
| Autonómicas todos (66 / 66) | 2,699 → 2,684 (38 / 28) | 2,613 → 2,598 (37 / 29) | 0,077 → 0,079 (9 / 32) | 0,100 → 0,101 (15 / 30) | 0,0194 → 0,0198 (9 / 35) | 0,0248 → 0,0247 (15 / 31) | 2,891 → 2,902 (32 / 34) | 2,624 → 2,636 (29 / 37) |

Ruido:
- Generales a 6-14 días: CRPS de escaños +0,016 ± 0,077 (1 de 4) en el nowcast y +0,036 ± 0,064 (1 de 4) con horizonte; log-score de las mayorías +0,013 ± 0,028 y +0,013 ± 0,020 (1 de 4 en los dos); Brier +0,0015 ± 0,0098 y +0,0024 ± 0,0077 (1 de 4 en los dos).
- Autonómicas a 6-14 días: CRPS de escaños +0,004 ± 0,031 y +0,003 ± 0,032 (2 de 5 en los dos); log-score de las mayorías +0,001 ± 0,002 (1 de 5) y +0,002 ± 0,003 (0 de 5); Brier +0,0001 ± 0,0012 y +0,0004 ± 0,0012 (1 de 5 en los dos).
- Todos los casos: CRPS de escaños −0,096 ± 0,075 (4 de 4) y −0,050 ± 0,078 (3 de 4) en las generales, −0,015 ± 0,028 y −0,013 ± 0,027 (4 de 5 en los dos) en las autonómicas; log-score del voto −0,076 ± 0,146 (3 de 5) y −0,071 ± 0,090 (4 de 5) en las generales, +0,010 ± 0,052 y +0,012 ± 0,031 (2 de 5 en los dos) en las autonómicas.

Sobre `None` el patrón es el mismo, con otro tamaño. El CRPS de escaños del nowcast de las generales mejora más allá del ruido, sobre todo a 90-180 días (6 casos mejor y 0 peor). Las bandas de escaños se acercan al nominal donde faltaban: con horizonte, la del 95 % de las generales va de 0,88 a 0,93 a 30-60 días y de 0,87 a 0,91 a 90-180. En el nowcast de las autonómicas, las colas del PIT de escaños bajan de 0,50 a 0,44 a 90-180 días.

Pero donde ya cubrían, a 6-14 días, las bandas pasan del nominal. En las generales con horizonte, la del 50 % de escaños va de 0,54 a 0,57 y la del 80 % de 0,79 a 0,85; en las autonómicas, la del 50 % de 0,57 a 0,60. El L1 del PIT del voto de las generales con horizonte a 6-14 días sube de 0,29 a 0,39. Las mayorías a 6-14 días empeoran en las dos poblaciones y en los dos modos, dentro del ruido. En las generales, el log-score pasa de 0,149 a 0,162 en el nowcast y de 0,163 a 0,176 con horizonte, con 2 casos mejor y 6 peor en los dos; en las autonómicas, de 0,106 a 0,107 y de 0,109 a 0,112, con 3 y 10. El log-score del voto de las autonómicas empeora algo, dentro del ruido.

### Los bloques se ensanchan

La composición no estrecha los bloques: los ensancha. El ancho medio de la banda del 80 % de los escaños de un bloque crece en todos los grupos, en los dos modos y sobre las dos configuraciones. En las generales crece entre un 8,7 y un 13,6 %: con horizonte, sobre 5 grados, de 34,0 a 37,5 escaños a 6-14 días y de 47,4 a 53,3 a 90-180; sobre `None`, de 29,2 a 31,9 y de 39,6 a 45,0. En las autonómicas crece entre un 1,6 y un 6,9 %. Con las bandas más anchas, la cobertura de los bloques sube o queda igual en 34 de las 36 celdas, también donde ya pasaba del nominal: sobre 5 grados, la del 50 % con horizonte de las generales a 6-14 días va de 0,62 a 0,81.

Es lo que el método 7 ya había medido para 2027. La correlación común se reparte entre todos los nacionales. Contrae su suma, y por eso el resto deja de recortarse, pero apenas contrae la de un par (PP + VOX, un 7 %). Y ensancha las diferencias entre partidos: la desviación de Derecha − Izquierda pasaba de 4,45 a 4,86 puntos. Los escaños de un bloque dependen de su diferencia con el otro, así que su banda se abre y la probabilidad de mayoría se va un poco más hacia el medio. La composición, tal como está construida, hace lo contrario de lo que pedía la sección de la escala.

### ¿Se recuperan las mayorías?

Para saber si la composición devuelve a las mayorías lo que les quitaron la escala y las colas, cada ejecución se compara también con la referencia sin escala, sin colas fijas y sin composición. Es la del error del promedio, `backtest/results_avg`, de la primera sección, en los mismos casos. Log-score de las mayorías con horizonte; entre paréntesis, casos mejor / peor que la referencia:

| Grupo (casos) | Referencia | + escala | + escala + 5 | + escala + composición | + escala + 5 + composición |
|---|---|---|---|---|---|
| Generales 6-14 (8) | 0,156 | 0,163 (0 / 7) | 0,187 (2 / 6) | 0,176 (2 / 6) | 0,206 (2 / 6) |
| Generales 30-60 (8) | 0,271 | 0,282 (2 / 6) | 0,297 (2 / 6) | 0,285 (3 / 5) | 0,305 (3 / 5) |
| Generales 90-180 (6) | 0,350 | 0,360 (2 / 4) | 0,370 (2 / 4) | 0,354 (3 / 3) | 0,375 (2 / 4) |
| Generales todos (22) | 0,251 | 0,260 (4 / 17) | 0,277 (6 / 16) | 0,264 (8 / 14) | 0,288 (7 / 15) |
| Autonómicas 6-14 (22) | 0,107 | 0,109 (2 / 11) | 0,120 (1 / 15) | 0,112 (2 / 12) | 0,121 (1 / 15) |
| Autonómicas 30-60 (22) | 0,075 | 0,086 (0 / 16) | 0,095 (0 / 16) | 0,086 (2 / 14) | 0,098 (0 / 16) |
| Autonómicas 90-180 (22) | 0,082 | 0,104 (0 / 16) | 0,115 (1 / 17) | 0,106 (0 / 17) | 0,118 (0 / 18) |
| Autonómicas todos (66) | 0,088 | 0,100 (2 / 43) | 0,110 (2 / 48) | 0,101 (4 / 43) | 0,112 (1 / 49) |

Brier de las mayorías con horizonte:

| Grupo (casos) | Referencia | + escala | + escala + 5 | + escala + composición | + escala + 5 + composición |
|---|---|---|---|---|---|
| Generales 6-14 (8) | 0,0391 | 0,0407 (0 / 7) | 0,0449 (2 / 6) | 0,0430 (2 / 6) | 0,0491 (2 / 6) |
| Generales 30-60 (8) | 0,0840 | 0,0862 (2 / 6) | 0,0881 (2 / 6) | 0,0855 (3 / 5) | 0,0900 (3 / 5) |
| Generales 90-180 (6) | 0,1180 | 0,1192 (2 / 4) | 0,1201 (2 / 4) | 0,1142 (3 / 3) | 0,1195 (3 / 3) |
| Generales todos (22) | 0,0769 | 0,0787 (4 / 17) | 0,0811 (6 / 16) | 0,0779 (8 / 14) | 0,0832 (8 / 14) |
| Autonómicas 6-14 (22) | 0,0310 | 0,0316 (2 / 11) | 0,0339 (1 / 15) | 0,0322 (2 / 13) | 0,0342 (1 / 17) |
| Autonómicas 30-60 (22) | 0,0152 | 0,0186 (0 / 16) | 0,0209 (0 / 17) | 0,0179 (2 / 14) | 0,0210 (0 / 17) |
| Autonómicas 90-180 (22) | 0,0191 | 0,0242 (1 / 16) | 0,0269 (1 / 19) | 0,0242 (1 / 16) | 0,0270 (1 / 17) |
| Autonómicas todos (66) | 0,0218 | 0,0248 (3 / 43) | 0,0272 (2 / 51) | 0,0247 (5 / 43) | 0,0274 (2 / 51) |

**No se recuperan.** Con horizonte, ninguna de las dos ejecuciones con composición vuelve al log-score de las mayorías de la referencia, en ningún grupo de ninguna de las dos poblaciones. Con el Brier pasa lo mismo, salvo en una celda: las generales a 90-180 días sobre `None` (0,1142 frente a 0,1180, con 3 casos mejor y 3 peor de 6). La composición no deshace el empeoramiento del log-score; lo aumenta un poco.
- Sobre 5 grados, el log-score de todos los casos pasa de 0,277 a 0,288 en las generales y de 0,110 a 0,112 en las autonómicas; la referencia es 0,251 y 0,088.
- Sobre `None`, de 0,260 a 0,264 y de 0,100 a 0,101.
- El Brier sobre `None` baja algo con horizonte, de 0,0787 a 0,0779 en las generales y de 0,0248 a 0,0247 en las autonómicas, dentro del ruido. Queda lejos de la referencia (0,0769 y 0,0218).

En el nowcast la escala no entra, así que la referencia coincide con el «antes» de (b). Ahí la composición sobre `None` deja la media del log-score de todos los casos de las generales donde estaba (0,242, con 8 casos mejor y 14 peor), pero la empeora a 6-14 días (de 0,149 a 0,162).

### Decisión (criterio de M7 y criterio 17)

**No se cumple ninguno de los dos criterios, y las mayorías no se recuperan.**
- **Criterio de M7.** A 6-14 días, el log-score y el Brier de las mayorías empeoran sobre las dos configuraciones, en las dos poblaciones y en los dos modos: suben las 16 medias de las tablas. En las generales sobre 5 grados, el empeoramiento va más allá del ruido, con 0 casos mejor de 8 en las cuatro celdas (log-score +0,018 ± 0,009 en el nowcast y +0,018 ± 0,010 con horizonte). En el resto queda dentro del ruido. El CRPS de escaños a 6-14 días queda dentro del ruido: sobre 5 grados mejora algo o queda igual (de 5,208 a 5,209 en las generales con horizonte), y sobre `None` empeora algo. Es el resultado del método 7 (25-09-2026) y de su revisión del 07-10-2026: el sorteo conjunto pierde a corto plazo.
- **Criterio 17.** La composición acerca al nominal algunas bandas de escaños a 30 días o más, y a esos horizontes el CRPS de escaños mejora en más casos de los que empeora. La excepción son las generales con horizonte sobre 5 grados: 7 casos mejor y 7 peor, y a 90-180 días la media empeora (de 9,071 a 9,134, con 2 mejor y 4 peor). Pero empeora las mayorías a 6-14 días: más allá del ruido en las generales sobre 5 grados. Y lleva más allá del nominal bandas de escaños que ya cubrían: la del 50 % de las autonómicas a 6-14 días sobre las dos configuraciones, y la del 80 % de las generales con horizonte sobre `None` (de 0,79 a 0,85). Ninguna de las dos combinaciones lo cumple.
- **Las mayorías.** Con horizonte, ninguna combinación con composición vuelve al log-score de las mayorías de la referencia sin escala: en las autonómicas, 43 de 66 casos peor sobre `None` y 49 de 66 sobre 5 grados. La razón está en los bloques: la correlación común los ensancha en escaños, entre un 8,7 y un 13,6 % en las generales, en vez de estrecharlos.

`composition` sigue como opción documentada y `None` sigue siendo el valor por defecto; nada cambia en el modelo vivo. **Es un hallazgo para Luis (02-11-2026), y la decisión es suya.** Lo que el backtest señala:
- La prueba que pedía la sección de la escala, la escala junto con la composición, está hecha, y sale negativa: la composición no arregla las mayorías de la escala.
- Si se quiere que la escala no abra las probabilidades de mayoría, hace falta otra pieza. Una correlación común entre todos los nacionales no la da. La candidata natural es una estructura por bloques, pero el método 7 la descartó porque la correlación dentro de un bloque cambia de un ciclo a otro (PP-VOX, +0,66 en 2023 y −0,69 en 2027). Queda como pregunta abierta, no como propuesta.
- Sobre `None`, a 30 días o más, la composición mejora el CRPS de escaños en más casos de los que empeora: en las generales, 12 mejor y 2 peor en el nowcast, 9 y 5 con horizonte; en las autonómicas, 27 y 17, y 26 y 18. También sube o deja igual las coberturas de escaños, que faltaban. Pero las mayorías tampoco mejoran ahí, todo dentro del ruido:
  - En las autonómicas empeoran en más casos de los que mejoran. Con horizonte, el log-score queda en 0,086 a 30-60 días (7 casos mejor y 9 peor) y pasa de 0,104 a 0,106 a 90-180 (5 y 11). En el nowcast pasa de 0,064 a 0,066 (4 y 11) y de 0,060 a 0,064 (2 y 11).
  - En las generales el resultado es mixto: con horizonte, de 0,282 a 0,285 a 30-60 días (4 y 4) y de 0,360 a 0,354 a 90-180 (3 y 3).

  Si Luis quisiera la composición para el modo Elección lejos de la elección, ganaría en los escaños de cada partido sin ganar en las mayorías, y las empeoraría a 6-14 días.

Límites de la prueba:
- Las autonómicas son cinco de los diecisiete ámbitos, los que más casos tienen; las otras doce quedan para una ejecución completa de la configuración que se elija.
- La tabla `average_errors` está vacía en la base: el backtest la leyó de un CSV calculado en memoria con el mismo código y la semántica de `load_average_errors`.
- La prueba es acumulativa. Se hizo sobre `error_model='average'` con `drift_scale='auto'` (y con 5 grados o sin ellos), configuraciones que no cumplieron el criterio 17 y que Luis puede no adoptar. Sobre el modelo que se publica (`'polls'`, sin escala ni colas), la referencia de la composición siguen siendo las medidas del método 7.
- Las generales con escaños válidos son cuatro elecciones: el ruido de las mayorías se mide con cuatro diferencias.
- El log-score del voto sale de los cuantiles exportados, no de las simulaciones (ver la sección anterior), y `ρ` es la del último sorteo del nowcast de cada caso.

Tiempos (1.000 simulaciones, en un Mac con cuatro ejecuciones a la vez): generales 3.065 s en (a) y 3.136 s en (b); autonómicas, 66 casos en cada configuración, entre 792 y 986 s por ámbito (4.408 y 4.322 s en total).

## La tarjeta «Fiabilidad» y la parte `calibration` (02-11-2026)

Cambios:
- Parte nueva `calibration` del backtest de cada ámbito (`bundle.BACKTEST_PARTS`, esquema `backtest-calibration@1`, gemelo `csv/calibration.csv`): las filas de `calibration.csv` tal cual, con el PIT sin redondear (redondeado, una celda podría cruzar el borde de una décima). Una carpeta de resultados anterior al PIT, sin `calibration.csv`, se publica sin la parte y borra la de una publicación anterior (`publish.backtest.BACKTEST_OPTIONAL`).
- Las 38 columnas `q*` de los cuantiles (`q05`…`q95` y sus `_h`) salen de las partes publicadas `shares`, `seats` y `blocks` (y del corte `model` de cada elección), en `export_backtest_table`: la web no las lee y el PIT va en su parte. Los CSV de `backtest/results` las conservan.
- `meta.params` del backtest lleva siempre `error_model`, `drift_scale` y `tail_dof` cuando hay `run.json`; uno escrito antes de ellos los lleva con el comportamiento de entonces, `'polls'`, `None` y `None` (`BACKTEST_RUN_DEFAULTS`).
- `pages.backtest.reliability_context`: lee las partes `calibration` y `blocks` de todos los ámbitos con backtest publicado, pone la población (`generales` para `es`, `autonomicas` para los demás; las celdas `prov_*`, `provincias`) y resume con `summarize_calibration` y `majority_reliability`.
- Tarjeta «Fiabilidad de las bandas» en `/metodo/backtest`, tras la de cobertura: un gráfico por población con las diez barras de cada grupo de horizontes y la línea de 0,10, el diagrama de fiabilidad de las mayorías con la diagonal, el enlace al CSV y «Cómo leerlo». Sin la parte en ningún ámbito (paquetes anteriores), no hay tarjeta. Las barras con línea de referencia son la opción `references` de `groupedOption` (`web/dist/js/charts/grouped.js`), como la de `renderLines`.

Qué mide la tarjeta:
- **Generales y autonómicas:** el voto de los partidos principales en modo Elección (la ejecución con horizonte, `mode == 'h'`), en los tres grupos de horizontes (6-14, 30-60 y 90-180 días). Cada elección pesa lo mismo dentro de su población, tenga los partidos que tenga.
- **Provincias:** el voto de cada partido en cada provincia, de todos los ámbitos. El backtest sólo tiene el nowcast de las provincias, que es una previsión de la elección sólo cerca de ella, así que entran sólo a 6-14 días.
- **Mayorías:** la probabilidad de mayoría absoluta de cada bloque en modo Elección frente a la frecuencia con que la hubo, en las cajas de `majority_reliability` con casos, juntando todos los ámbitos.

Tests: `tests/test_bundle_unit.py` (la parte y su esquema), `tests/test_publish_unit.py` (la parte con su CSV, la carpeta sin `calibration.csv`, los `q*` fuera, `meta.params`), `tests/test_pages.py` (`reliability_context`, la tarjeta con el fixture `backtest-calibration.json`, sin ella sin la parte, redacción pública) y `tests/test_js_charts.py` (barras con referencia). Comprobada en Chrome sin cabeza sobre el paquete de pruebas: la tarjeta se dibuja sin errores de consola, y sin la parte no aparece.

## Decisión sobre los valores por defecto (02-11-2026)

**No entra ninguno.** El criterio 17 no se cumplió en ninguno de los tres pasos medidos: el error del propio promedio empeora coberturas y PIT, la escala de la deriva empeora las mayorías, y las colas fijas empeoran las mayorías y pasan del nominal donde ya cubrían. Los valores por defecto siguen siendo los de antes del método 15 y nada cambia en el modelo vivo: ni `backtest/results/` se reejecuta ni el job `compute` tiene que rellenar `average_errors` para el modelo publicado (sólo la lee `error_model='average'`). El paso 15 (`composition`) terminó después y no la cambia: tampoco cumple el criterio 17 ni el suyo del método 7 (ver su sección).

Las cuatro piezas quedan como opciones, con el valor de hoy por defecto:

| Opción | Valores | Por defecto | Dónde se elige |
|---|---|---|---|
| `error_model` | `'polls'` (el error de los sondeos sueltos), `'average'` (el del propio promedio, tabla `average_errors`) | `'polls'` | `Simulator(error_model=...)`; job `backtest` `"error_model"`; `--error-model` |
| `drift_scale` | `None` (1), `'auto'` (ajustada sobre `average_errors`), un número positivo | `None` | `Simulator(drift_scale=...)`; job `backtest` `"drift_scale"`; `--drift-scale` |
| `tail_dof` | `None` (los grados de la regresión, `nobs − 2`), un entero de al menos 2 | `None` | `Simulator(tail_dof=...)`; job `backtest` `"tail_dof"`; `--tail-dof` |
| `composition` | `None` (ruido independiente), `'auto'`, una razón en (0, 1] | `None` | `Simulator(composition=...)`; job `backtest` `"composition"` |

**Revisión: tras el 29-N**, con la elección general como caso nuevo del backtest. La decisión es de Luis, que puede cambiarla en un commit suyo; para adoptar una configuración hacen falta, por este orden, el backtest completo de las 17 comunidades con ella, `backtest/results/` reejecutado, el job `compute` con el paso `average_errors` (si usa `'average'` o `drift_scale='auto'`) y las constantes de `tests/integration/test_model.py`.

## Salvedades

- **Pocas generales.** El backtest de `es` tiene cinco elecciones (28 casos); cada barra del PIT de las generales son cinco elecciones, y una sola elección rara la mueve. Las autonómicas tienen más casos, pero la escala, las colas y la correlación (pasos 13 a 15) se midieron sobre cinco de los diecisiete ámbitos.
- **El estimador del error del promedio no ve las autonómicas.** Se ajusta sobre las filas del ámbito estimador (`es` para todos, la regla de M11), así que el `e` de una autonómica sale de las generales; y sólo ve partidos principales, por lo que los regionales de `es` (CiU y PNV en 1993-2004, siete filas) quedan casi sin error.
- **`s` neto.** `e_net² = max(e² − s², 0)` quita del error del promedio la parte que ya pone `s`; cuando `s` es mayor que el error, el neto es cero y la fila empuja el estimador hacia abajo.
- **Unidades.** Los dos estimadores de error predicen el error **absoluto** medio y el Simulator lo usa como desviación típica: para una normal, la media del valor absoluto es 0,8 veces la desviación típica, y con las colas del error del promedio la raíz del cuadrado medio es 1,3-1,5 veces la media absoluta. Es la causa de que `'average'` no cubra, y cualquier nueva prueba del error del promedio tiene que empezar por esa escala.
