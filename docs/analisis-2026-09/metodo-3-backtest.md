# Método 3: backtest del modelo (24-09-2026)

Código: `mtpy/lib/backtest.py` (`run_case`, `summarize_case`, `run_backtest`) y `backtest/run_backtest.py` (CLI). Resultados en `backtest/results/` (`run.json` registra commit, fecha, semilla y estado de la base de datos). Tests: `tests/test_backtest_unit.py` (métricas) y `tests/integration/test_backtest.py` (un caso real). Protocolo y definición de las métricas en `backtest/README.md`.

## Qué mide y qué no

El `Computer` ya guarda el error de **cada encuesta** frente al resultado (`polls_results.error`, `polls.error_avg`, ratings): es el backtest de las encuestadoras. Este es el backtest del **modelo**: el promedio ponderado a `d` días de la elección, sus intervalos, los escaños simulados y las probabilidades de mayoría, en cinco elecciones (2015-12, 2016-06, 2019-04, 2019-11, 2023-07) y seis horizontes (6, 14, 30, 60, 90 y 180 días). Cada caso es fuera de muestra: las encuestas se cortan en `limit_date = elección − d` y los ratings y estimadores usan solo elecciones anteriores. 500 simulaciones MT por caso, semilla 42; 28 casos válidos (2016 a 180 días no tiene encuestas y 2019-11 a 180 días tiene una sola).

Cambios que ha exigido: `Forecaster` y `Simulator` leen los parámetros del evento con `get_event_params` (derivados de los datos y sobrescritos por `data/params.json`), así que 2015 y 2016 funcionan sin entrada en el JSON; `Forecaster.fit` omite con aviso los partidos con menos de tres encuestas con peso positivo (NA+ en 2019-04 a 30 días rompía la selección del ancho de banda); `build_forecast` tolera partidos sin ajuste. En 2015, UP y Cs no tienen resultado previo ni regla `smap`, así que sus escaños no se pueden proyectar: el caso solo cuenta para porcentajes (`seats_valid = False`).

## Resultados por horizonte (medias de las elecciones; partidos principales)

Tabla de la primera ejecución, anterior a la sucesión de partidos en las encuestas (método 4); `backtest/results/` contiene la ejecución más reciente y `metodo-4-sucesion-encuestas.md` la comparación.

| d | n | MAE promedio | MAE última enc. | MAE media 4 sem. | MAE result. anterior | Cob. 50 % | Cob. 80 % | Cob. 95 % | Cob. IC Forecaster 95 % | MAE escaños | Cob. escaños 95 % | CRPS escaños | Brier mayoría bloque |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 5 (4) | **1,61** | 1,74 | 1,86 | 5,77 | 0,57 | 0,78 | 0,96 | 0,36 | 6,3 | 1,00 | 4,4 | 0,05 |
| 14 | 5 (4) | 2,04 | 2,13 | 1,97 | 5,77 | 0,50 | 0,71 | 0,88 | 0,21 | 8,2 | 1,00 | 5,8 | 0,05 |
| 30 | 5 (4) | 2,14 | 2,25 | 2,14 | 5,77 | 0,49 | 0,76 | 0,84 | 0,31 | 8,4 | 0,96 | 5,9 | 0,08 |
| 60 | 5 (4) | 2,81 | 2,69 | 2,80 | 5,77 | 0,38 | 0,77 | 0,81 | 0,13 | 10,0 | 0,88 | 7,2 | 0,13 |
| 90 | 5 (4) | 2,98 | 2,78 | 2,93 | 5,77 | 0,32 | 0,73 | 0,86 | 0,24 | 13,0 | 0,88 | 9,0 | 0,16 |
| 180 | 3 (2) | 3,35 | 3,24 | 3,24 | 7,75 | 0,28 | 0,57 | 0,78 | 0,07 | 15,9 | 0,68 | 12,6 | 0,22 |

`n` = elecciones con porcentajes (con escaños). MAE en puntos porcentuales sobre votos válidos; escaños en escaños. Con 4-5 elecciones y 4-6 partidos principales cada una, las coberturas se estiman sobre 20-30 comprobaciones: son orientativas.

## Lectura

1. **El promedio bate a las encuestas sueltas en la última semana, y empata con la media simple después.** A 6 días el error es 1,61 puntos frente a 1,74 de la última encuesta y 1,86 de la media de cuatro semanas. A 14-30 días el promedio y la media simple son indistinguibles (2,0-2,1), y a 60-90 días la última encuesta es ligeramente mejor (2,7-2,8 frente a 2,8-3,0): el ancho de banda de ~70 días del suavizado introduce retardo cuando la opinión se mueve. El resultado anterior (5,8) queda lejos: las encuestas aportan información.
2. **La incertidumbre está bien calibrada a una semana y es demasiado estrecha a partir de dos.** Los intervalos del 95 % de las cuotas cubren el 96 % a 6 días, pero el 84-88 % a 14-30 días y el 78-86 % a 60-180; los del 50 % pasan del 57 % al 28-38 %. El error histórico que alimenta la simulación depende poco del horizonte (`v2err` con `weeks = (d+1)//7`), y el pronóstico congelado no añade deriva (M9, M10). Este backtest es justo lo que hace falta para calibrar esa dependencia.
3. **El intervalo propio del Forecaster no es un intervalo del resultado**: cubre el 36 % a 6 días y el 7 % a 180. Es el error estadístico de la media local, como ya decía el análisis (M6).
4. **Escaños**: 6,3 de error medio en los principales a 6 días, 8-8,5 a 14-30, 10-13 a 60-90, con intervalos del 95 % que cubren siempre a 6-14 días (algo anchos) y el 88 % a 60-90. Los mayores fallos son los de las encuestas: PSOE 2023 a 30 días (97 escaños frente a 121; −6,3 puntos de voto), PP 2019-04 a 14 días (88 frente a 66; +5,0), VOX 2019-11 a 30 días (30 frente a 52; −4,6), UP 2016 a 14 días (90 frente a 71).
5. **Probabilidades de mayoría de bloque** (`vs`): Brier 0,05 a 6 días, 0,08 a 30, 0,16 a 90. En ninguna de las cinco elecciones el bloque de derecha alcanzó los 176, y el modelo lo captó en 2016, 2019-04 y 2019-11 (probabilidades de 0,03 a 0,19 a ≤ 30 días); falló en 2015 a 6 días (0,81, por la sobreestimación de Cs: 18,4 frente a 13,9) y en 2023 (0,52 a 6 días, 0,74 a 30), el error colectivo de las encuestas de aquel año.
6. **Hallazgo para el modelo de datos: la sucesión de partidos en las encuestas.** En 2023 a 60-90 días la suma de los principales queda 9,7 puntos por debajo del resultado (`bias_sum`), y en 2015 UP aparece 6 puntos por debajo a 30 días. La causa no es solo demoscópica: SUMAR no existía en las encuestas hasta junio de 2023 (aparecían UP y MP) y `parties.event` no recoge esos porcentajes, que van al residuo `'-'`. El `smap` resuelve la herencia de **resultados** previos, pero no la de **encuestas** previas. Es el primer punto a corregir en la capa de datos (equivalente al `bmaps.min` aplicado dentro del Forecaster).

## Implicaciones para los pasos siguientes

- Calibrar la escala del ruido por horizonte con los errores de este backtest (el error del promedio, no el de encuestas sueltas) y hacer que la incertidumbre crezca con `d` (M9/M10, M13).
- Resolver la sucesión de partidos en las encuestas (SUMAR 2023, UP 2015) antes de juzgar cualquier mejora del promedio.
- Revisar el ancho de banda a horizontes largos (validación cruzada frente a ISJ, M8).
- Con la composición y correlación entre partidos (M2), el ruido provincial (M4) y los house effects (M5), volver a ejecutar `python backtest/run_backtest.py` y comparar `by_horizon.csv`: es el criterio de aceptación de cada cambio.

## Limitaciones

- Cinco elecciones (cuatro con escaños): las medias por horizonte son sensibles a cada una; conviene mirar `metrics.csv` caso a caso.
- Las encuestas cargadas son las de Wikipedia con su cobertura actual; un cambio en la base de datos cambia los resultados (`run.json` guarda el estado).
- El `Computer` de cada caso usa todas las elecciones anteriores, pero la curación (`quality`, mapas) es la de hoy: no es una reconstrucción histórica del conocimiento disponible en cada fecha.
