# Método 14: afinar el promedio con los sondeos futuros (02/03-11-2026)

Cambios:
- `mtpy/lib/tuning.py`: cortes por ciclo (`cycle_cuts`), predicción y puntuación de los sondeos futuros (`predict_polls`, `score_polls`, `autocorrelation`, `roughness`), un corte (`run_cut`), la búsqueda por coordenadas (`coordinate_descent`, `run_search`), la orquestación por ámbito (`run_tuning`, `tuning_scope`), el pool de procesos (`workers`, con un vigilante de 15 minutos por corte), las partes de la fase 2 y su unión (`merge_parts`), la etiqueta de la configuración elegida (`best_label`) y la copia entre el repositorio y S3 (`copy_tuning`).
- Job `tune` (`mtpy/jobs/Tune.py`, `mtpy/pipelines/tune/Tune.py`) con las acciones `run`, `push` y `pull`, y su envoltorio `backtest/run_tuning.py` (`--workers`, `--mode merge`).
- Regresión de grado 0 (`poly_deg=0`) arreglada en `mtpy/core/utils/stat.py`: la matriz de diseño era de una dimensión y todo ajuste de grado 0 fallaba (`tuple index out of range`), también fuera del afinado; el grado 1 queda idéntico.
- Keepalives de TCP en las conexiones de PostgreSQL (`mtpy/core/dal/PostgreSQL.py`): un enlace muerto falla en unos 2 minutos en vez de colgar (la primera fase 1 se quedó colgada más de una hora en un enlace móvil).

Tests: `tests/test_tuning_unit.py`, `tests/test_jobs_tune.py`, `tests/test_pipelines_tune.py`, `tests/test_stat.py`, `tests/test_dal_unit.py` (sin base). Spec: `docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md` (parte A).

## Problema

Los parámetros del promedio (ancho de banda de la regresión local, grado, prolongación, ventana de sobrepublicación, contracción de los efectos de casa y del rebaño) se fijaron a mano en cuadernos, sin un criterio fuera de muestra. El backtest de elecciones sólo tiene cinco generales: no basta para elegir entre configuraciones que difieren en décimas. Los sondeos publicados después de cada fecha son cientos de pruebas fuera de muestra por ciclo: un buen promedio a fecha de hoy predice bien la cifra que publicará la próxima casa (más su efecto de casa).

## Método

- **Cortes.** Uno cada 7 días (14 en la fase 2) en los 365 días anteriores a cada elección, hasta hoy en el ciclo en curso, con al menos 5 sondeos útiles en los 90 días anteriores. En cada corte se congela el Forecaster como en el backtest (`drange` = días hasta la elección) con sólo los sondeos publicados hasta el corte.
- **Predicción.** Cada sondeo que entraría en el promedio, publicado en los k días siguientes y con `weight_over` > 0, se predice con el promedio del corte leído en su fecha más el efecto de casa de su encuestadora. Se puntúa el RMSE a k = 7 y k = 28 días, cada celda (sondeo y partido) con el mismo peso; cada ciclo pesa lo mismo en la búsqueda.
- **Búsqueda.** Descenso por coordenadas sobre la rejilla fijada de antemano (spec A5): se elige por RMSE a k = 7 si k = 28 no empeora más de un 2 % y la rugosidad no pasa de 1,5 veces la de partida. Fase 1 (suavizado: `bw`, `bw_cap`, `poly_deg`, `max_fc`, `wspan`), dos vueltas con efectos de casa, muestra efectiva y rebaño apagados, cortes semanales en los seis ciclos de búsqueda (generales de 2008 a 2019-11). Fase 2 (contracciones: `tau`, `he_n_iter`, `herd_prior`), una vuelta con todo encendido desde el `best` de la fase 1, cortes cada 14 días en los ciclos de búsqueda desde 2015. Comprobación: el `best` de la fase 2 frente a la configuración vigente, las dos con todo encendido, en los ciclos reservados (2023-07-23 y el ciclo en curso, 2026-11-29). La «configuración vigente» es la de hoy normalizada: los pesos de sobrepublicación se recalculan en cada corte con la ventana viva (`wspan` 4) en lugar de leerse de la base, que los guarda calculados con todos los sondeos (mirarían al futuro).
- **Ejecución.** Las dos fases y la comprobación corrieron en el portátil con un pool de 8 procesos (`--workers 8`): cada proceso arranca su app y su motor de base de datos, y los cortes de una evaluación se reparten entre ellos; las tablas son idénticas a las de la ejecución secuencial (mismo orden por configuración, ciclo y corte). La fase 2 puede repartirse además por ciclos en tareas separadas (ECS): `shrinkage` con `cycles` es una parte, que no desciende sino que evalúa el producto entero de `tau` × `he_n_iter` × `herd_prior` desde el punto de partida (18 configuraciones con la rejilla de la spec; escribe `trials-shrinkage-{ciclos}.csv` y compañía, con el punto de partida en `run-shrinkage-{ciclos}.json`, sin `best.json`), y `merge` junta las partes y repite sobre la tabla junta el mismo descenso de la fase: como toda configuración que ese descenso puede visitar está en todas las partes, el resultado es exactamente el de la fase entera de una vez (si cada parte descendiera por su cuenta, los ciclos se separarían tras `tau` —con estos datos 2015 prefiere 0,08, 2016 y 2019-04 prefieren 0,04 y 2019-11 0,16— y la unión no podría seguir). `merge` parte del punto de partida que registraron las partes, así que repetirla da lo mismo, y escribe sus propias `cuts.csv`, `scores.csv` y `trials.csv`; `push` y `pull` copian las tablas entre el repositorio y S3. Aquí no hizo falta: la fase 2 corrió entera en el portátil, y sus resultados son los de esa ejecución.

### Tablas (`backtest/tuning/`, `es` en la raíz)

| Fichero | Qué tiene |
|---|---|
| `trials-smoothing.csv`, `scores-smoothing.csv` | Fase 1: cada prueba del descenso (vuelta, clave, valor, RMSE a 7 y 28, rugosidad, elegida, número de evaluación) y las puntuaciones por configuración, ciclo y horizonte (`evaluation`) |
| `trials-shrinkage.csv`, `scores-shrinkage.csv` | Fase 2, lo mismo |
| `scores-check.csv`, `trials-check.csv` | Comprobación: puntuaciones de la configuración afinada y de la vigente en 2023 y 2026; `trials-check.csv` sólo tiene la cabecera (la comprobación no busca) |
| `cuts.csv`, `scores.csv`, `trials.csv`, `run.json` | Los de la **última ejecución, la comprobación** (cada modo los reescribe): un corte por fila con sus segundos, las puntuaciones, y el registro de la ejecución (commit, estado de la base: 3.745 sondeos, el último del 25-10-2026) |
| `best.json` | La configuración afinada, con efectos de casa, muestra efectiva y rebaño encendidos, etiquetada con los ajustes que difieren de la vigente |
| `polls.csv` | Una fila por sondeo futuro y partido de la última ejecución; no se versiona (`.gitignore`) |

## Coste

**Medida previa** en `es` con una sola configuración, cortes cada 28 días en todos los ciclos (95 cortes), 8 procesos en el portátil (10 núcleos, 24 GB), contra la base de AWS (sólo lectura). Los segundos por corte son los de la columna `seconds` de `cuts.csv`, medidos dentro de cada proceso (equivalentes a la ejecución secuencial, con 8 procesos compitiendo por la CPU):

| Configuración | Cortes | Media por corte | Búsqueda | Comprobación | Suma de los cortes | Reloj con 8 procesos |
|---|---|---|---|---|---|---|
| Todo encendido (vigente) | 95 | 43,5 s | 37,3 s | 59,2 s | 4.136 s | 581 s (7,1 ×) |
| Efectos de casa, muestra efectiva y rebaño apagados | 95 | 6,6 s | 6,4 s | 7,3 s | 629 s | 123 s (5,1 ×) |

**Ejecución real** (fase 1 repetida tras arreglar el grado 0; las tres sin un solo corte fallido ni avisos):

| Fase | Evaluaciones × cortes | Cortes | Media por corte | Suma de los cortes | Reloj con 8 procesos | Spec A10 |
|---|---|---|---|---|---|---|
| 1, suavizado | 33 × 258 | 8.514 | 6,1 s | 52.000 s (14,4 h) | 6.659 s (1,85 h; 7,8 ×) | 700 × 25, segundos por corte: ≈ 15 h de CPU |
| 2, contracciones | 6 × 77 | 462 | 29,6 s | 13.657 s (3,8 h) | 1.759 s (0,5 h; 7,8 ×) | 350 × 8 a 35-55 s: ≈ 40 h de CPU, en ECS |
| Comprobación | 2 × 102 | 204 | 53,4 s | 10.897 s (3,0 h) | 1.393 s (0,4 h; 7,8 ×) | — |
| Total | | 9.180 | | 76.551 s (21,3 h) | 9.811 s (2,7 h) | |

La fase 1 evaluó 33 configuraciones distintas (no 25: dos vueltas sobre 21 valores, con las repetidas memorizadas); la fase 2, 6 de 8 (dos son la vigente). El tamaño de la spec era una cota alta: con los ciclos reales de `es` (258 cortes semanales en la fase 1, 77 quincenales desde 2015 en la 2) y el pool de 8 procesos, todo cupo en menos de 3 horas de reloj en el portátil y nada fue a ECS. Medias por corte: 6,1 s en la fase 1 (efectos apagados, frente a los «segundos» de la spec); con todo encendido, 29,6 s en la fase 2 (por debajo de los 35-55 s de la spec: va de 6,8 s en 2016 y 7,4 s en 2019-11, ciclos con pocos sondeos, a 45,8 s en 2015), 53,4 s en la comprobación y 43,5 s en la medida previa (las dos dentro de los 35-55 s).

Repartir la fase 2 por ciclos en ECS costaría el triple de CPU, porque cada parte evalúa las 18 configuraciones del producto en vez de las 6 del descenso: con los cortes y segundos de esta ejecución, 18 × (27 × 45,8 + 12 × 6,8 + 27 × 32,4 + 11 × 7,4) ≈ 40.900 s (11,4 h) frente a 13.657 s, con la parte más larga (2015) en unas 6,2 h de CPU.

## Resultados (`es`)

### Fase 1: suavizado (efectos apagados, 6 ciclos de búsqueda, k = 7 / k = 28)

| Paso | Configuración en vigor | RMSE k = 7 | RMSE k = 28 | Rugosidad |
|---|---|---|---|---|
| Partida (vigente) | adaptativo, 3 iteraciones, sin tope, grado 1, `max_fc` 10, `wspan` 4 | 1,729 | 1,855 | 0,0219 |
| Vuelta 1, `bw` | adaptativo, 2 iteraciones | 1,727 | 1,849 | 0,0221 |
| Vuelta 1, `bw_cap` | tope ×1,5 | 1,726 | 1,875 | 0,0237 |
| Vuelta 1, `poly_deg` | grado 0 probado (1,773; 1,917) y descartado | 1,726 | 1,875 | 0,0237 |
| Vuelta 1, `max_fc` | **0** | **1,688** | **1,804** | 0,0219 |
| Vuelta 1, `wspan` | 4 (2: 1,689; 7: 1,701; 14: 1,701) | 1,688 | 1,804 | 0,0219 |
| Vuelta 2, `bw` | adaptativo, 3 iteraciones (otra vez la vigente) | 1,688 | 1,804 | 0,0219 |
| Vuelta 2, `bw_cap` | **tope ×1,0** | **1,681** | **1,791** | 0,0232 |
| Vuelta 2, resto | sin cambios (grado 0: 1,740; `max_fc` 10: 1,755 / 2,010) | 1,681 | 1,791 | 0,0232 |

Resultado: `bw_cap=1.0; max_fc=0`. El RMSE a 7 días baja de 1,729 a 1,681 (−2,8 %) y a 28 días de 1,855 a 1,791 (−3,5 %), con 1,06 veces la rugosidad de partida (el límite era 1,5). La mayor ganancia es la prolongación: `max_fc` 0 (sostener el último valor del promedio en lugar de prolongar la recta de la regresión hasta 10 días) baja el RMSE a 7 días de 1,726 a 1,688; la segunda, limitar el ancho de banda adaptativo en la frontera al del piloto (×1,0), de 1,688 a 1,681. Los anchos fijos (ISJ, 30, 45, 60, 90 días) son peores que el adaptativo en las dos vueltas, y el grado 0 suaviza más (rugosidad 0,013-0,015) pero predice peor (+0,05 a 7 días). La ventana de sobrepublicación vigente (4 días) es la mejor.

### Fase 2: contracciones (todo encendido, 4 ciclos desde 2015, cortes cada 14 días)

| Paso | RMSE k = 7 | RMSE k = 28 |
|---|---|---|
| Partida (`best` de la fase 1, todo encendido) | 1,6002 | 1,8163 |
| `tau` 0,04 / 0,08 / 0,16 | 1,6059 / **1,6002** / 1,6034 | — |
| `he_n_iter` **1** | 1,5995 | 1,8153 |
| `herd_prior` **2** | 1,5986 | 1,8145 |

Resultado: `bw_cap=1.0; max_fc=0; he_n_iter=1; herd_prior=2.0`. La fase 2 movió dos ajustes, `he_n_iter` de 3 a 1 y `herd_prior` de 5 a 2, por ganancias del nivel del ruido: el RMSE a 7 días baja 0,0007 y 0,0009 puntos de voto (≈ 0,05 % cada uno; 0,0016 puntos y −0,10 % entre los dos) sobre 77 cortes, y `tau` se queda en 0,08. Esos dos cambios viajan con la configuración afinada a la comprobación y a cualquier adopción por A8 sin que nada los respalde por sí solos; por eso la tarea 9b hace el backtest también de la candidata de la fase 1 sola (`bw_cap=1.0; max_fc=0`), para que Luis decida con las dos.

### Comprobación: ciclos reservados, todo encendido

| Ciclo | Cortes | Configuración | RMSE k = 7 | RMSE k = 28 | MAE k = 7 | Autocorrelación | Rugosidad |
|---|---|---|---|---|---|---|---|
| 2023-07-23 | 53 | vigente | 1,320 | 1,476 | 1,010 | 0,319 | 0,0117 |
| 2023-07-23 | 53 | afinada | **1,283** (−2,8 %) | **1,442** (−2,3 %) | 0,967 | 0,244 | 0,0121 |
| 2026-11-29 (en curso) | 49 | vigente | 1,076 | 1,117 | 0,787 | 0,158 | 0,0065 |
| 2026-11-29 (en curso) | 49 | afinada | **1,058** (−1,7 %) | **1,096** (−1,8 %) | 0,771 | 0,126 | 0,0068 |

(200 y 98 sondeos a 7 días; 795 y 377 a 28.) La configuración afinada mejora el RMSE a 7 y a 28 días en los dos ciclos reservados, con una rugosidad 1,03-1,05 veces la vigente y menos autocorrelación de los residuos de un mismo corte (0,24 frente a 0,32 en 2023, 0,13 frente a 0,16 en 2026): el promedio afinado va menos retrasado respecto a los sondeos que vienen. El «skill» es negativo en las dos (el error supera el ruido de muestreo de los sondeos), como cabe esperar con efectos de casa, diseños y redondeos que el muestreo no recoge.

### Qué decide esto

Spec A8 pide dos condiciones a la vez para cambiar los valores por defecto. **La primera se cumple**: la configuración afinada mejora el RMSE a k = 7 en los dos ciclos de comprobación (2023 y 2026). **La segunda** (que el backtest de elecciones de `es` no empeore MAE de voto, CRPS de escaños ni Brier de mayorías más de un 2 % ni baje la cobertura del 95 %) es la tarea 9b. Aquí no cambia ningún valor por defecto: la configuración queda en `backtest/tuning/best.json` como opción documentada hasta ese juicio.
