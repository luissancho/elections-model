# Método 14: afinar el promedio con los sondeos futuros (02/03-11-2026)

Cambios:
- `mtpy/lib/tuning.py`: cortes por ciclo (`cycle_cuts`), predicción y puntuación de los sondeos futuros (`predict_polls`, `score_polls`, `autocorrelation`, `roughness`), un corte (`run_cut`), la búsqueda por coordenadas (`coordinate_descent`, `run_search`), la orquestación por ámbito (`run_tuning`, `tuning_scope`), el pool de procesos (`workers`, con un vigilante de 15 minutos por corte), las partes de la fase 2 y su unión (`merge_parts`), la etiqueta de la configuración elegida (`best_label`) y la copia entre el repositorio y S3 (`copy_tuning`).
- Job `tune` (`mtpy/jobs/Tune.py`, `mtpy/pipelines/tune/Tune.py`) con las acciones `run`, `push` y `pull`, y su envoltorio `backtest/run_tuning.py` (`--workers`, `--mode merge`).
- Regresión de grado 0 (`poly_deg=0`) arreglada en `mtpy/core/utils/stat.py`: la matriz de diseño era de una dimensión y todo ajuste de grado 0 fallaba (`tuple index out of range`), también fuera del afinado; el grado 1 queda idéntico.
- Backtest con una configuración del promedio: `--config` en `backtest/run_backtest.py` y `config` en el job `backtest` (`mtpy/jobs/Backtest.py`, validada antes de correr), enhebrada hasta `run_case` (`mtpy/lib/backtest.py`), que construye el `Simulator` con sus parámetros y, con `weights`, los pesos recalculados sobre los sondeos del corte; `meta.csv` registra su etiqueta (`config`) y los sondeos del promedio con peso recalculado (`n_overrides`), y `run.json` la configuración entera.
- Keepalives de TCP en las conexiones de PostgreSQL (`mtpy/core/dal/PostgreSQL.py`): un enlace muerto falla en unos 2 minutos en vez de colgar (la primera fase 1 se quedó colgada más de una hora en un enlace móvil).

Tests: `tests/test_tuning_unit.py`, `tests/test_jobs_tune.py`, `tests/test_pipelines_tune.py`, `tests/test_stat.py`, `tests/test_dal_unit.py`, `tests/test_backtest_unit.py`, `tests/test_jobs_backtest.py`, `tests/test_pipelines_backtest.py` (sin base) y `tests/integration/test_backtest.py::test_run_case_with_a_tuned_config` (con base, sólo lectura). Spec: `docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md` (parte A).

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

Spec A8 pide dos condiciones a la vez para cambiar los valores por defecto. **La primera se cumple**: la configuración afinada mejora el RMSE a k = 7 en los dos ciclos de comprobación (2023 y 2026). **La segunda** (que el backtest de elecciones de `es` no empeore MAE de voto, CRPS de escaños ni Brier de mayorías más de un 2 % ni baje la cobertura del 95 %) **no se cumple** (ver «Backtest antes / después» y «Decisión»). No cambia ningún valor por defecto: la configuración queda en `backtest/tuning/best.json` como opción documentada.

## Backtest antes / después (`es`, 03-11-2026)

Tres ejecuciones del backtest de elecciones de `es` con `--config`. Cada una cubre las cinco generales desde 2015 a los seis horizontes, con 1.000 simulaciones por caso y modo y semilla 42, contra la base de AWS (sólo lectura: 3.745 sondeos, el último del 25-10-2026, el mismo estado que la referencia). Corrieron a la vez en el portátil, cada una en su carpeta (fuera de git, `backtest/results_*/`):

| Ejecución | Configuración (`--config`) | Carpeta | Reloj |
|---|---|---|---|
| (T) afinada | `backtest/tuning/best.json` (`bw_cap=1.0; max_fc=0; he_n_iter=1; herd_prior=2.0`) | `backtest/results_tuned` | 47 min 18 s |
| (P) sólo la fase 1 | la vigente normalizada con `bw_cap=1.0` y `max_fc=0` (`bw_cap=1.0; max_fc=0`) | `backtest/results_phase1` | 49 min 57 s |
| (D) vigente con pesos recalculados | `normalize_config(default_config())` (`default; pesos recalculados`): el modelo de hoy con los pesos de sobrepublicación recalculados en cada corte con `wspan` 4 | `backtest/results_default_rw` | 49 min 14 s |

```
python backtest/run_backtest.py --config backtest/tuning/best.json --out backtest/results_tuned
python backtest/run_backtest.py --config <vigente normalizada>.json --out backtest/results_default_rw
python backtest/run_backtest.py --config <vigente normalizada + bw_cap=1.0 + max_fc=0>.json --out backtest/results_phase1
```

Hay dos «antes»:

- **(a)**, la referencia commiteada en `backtest/results/` (commit `45aaa8b`), es el modelo tal como se publica. Usa los pesos guardados en la base, calculados con todos los sondeos del ciclo: miran al futuro. Es el «antes» que nombra A8.
- **(D)** sólo cambia, respecto a (a), los pesos, que se recalculan en cada corte.

Así:

- T frente a (a) mezcla el afinado con la retirada de esa mirada al futuro.
- T frente a (D) aísla el afinado.
- P frente a (D) aísla la fase 1.
- T frente a (P) aísla los dos ajustes de la fase 2.

En las tres ejecuciones `meta.csv` lleva la etiqueta de la configuración (`config`). `n_overrides` es igual a `n_polls` en los 28 casos: todos los sondeos del promedio llevan el peso recalculado, también en (D), que tiene `weights`.

Los casos son los mismos 28 de la referencia, porque en las cuatro ejecuciones faltan 2016-06-26 y 2019-11-10 a 180 días. Esos dos ciclos duraron 189 y 196 días, y 180 días antes de la elección sólo había 0 y 1 sondeos del ciclo. La referencia lo anota en `meta.error` («No polls…», «Not enough polls…»). Las tres nuevas anotan «zero-size array to reduction operation minimum…», porque el recálculo de los pesos falla antes que el `Forecaster`: cambia el motivo, no el caso omitido.

Columnas: los valores de `by_horizon.csv`, la media de las elecciones de cada horizonte, del nowcast. La fila «Todos» es la media de los casos de `metrics.csv`. El MAE, el CRPS y el Brier se comparan en cambio relativo (negativo es mejor) y las coberturas en diferencia absoluta. En negrita, lo que incumple A8: un empeoramiento de más del 2 % o una cobertura del 95 % más baja. Los casos con escaños son 4 por horizonte (2 a 180 días), porque 2015 no proyecta escaños (`seats_valid`).

**MAE de voto (puntos, partidos principales)**

| Horizonte | Casos | (a) publicada | (D) por defecto, pesos recalculados | (P) fase 1 | (T) afinada | T frente a (a) | T frente a (D) | P frente a (D) |
|---|---|---|---|---|---|---|---|---|
| 6 | 5 | 1,699 | 1,681 | 1,674 | 1,676 | −1,3 % | −0,3 % | −0,4 % |
| 14 | 5 | 1,976 | 1,900 | 1,889 | 1,896 | −4,0 % | −0,2 % | −0,6 % |
| 30 | 5 | 2,168 | 2,070 | 2,041 | 2,046 | −5,6 % | −1,2 % | −1,4 % |
| 60 | 5 | 2,803 | 2,832 | 2,871 | 2,859 | **+2,0 %** | +1,0 % | +1,4 % |
| 90 | 5 | 2,838 | 2,813 | 2,795 | 2,770 | −2,4 % | −1,5 % | −0,6 % |
| 180 | 3 | 3,015 | 2,973 | 3,017 | 3,013 | −0,1 % | +1,4 % | +1,5 % |
| Todos | 28 | 2,374 | 2,335 | 2,336 | 2,331 | −1,8 % | −0,2 % | +0,0 % |

**CRPS de escaños (partidos principales)**

| Horizonte | Casos | (a) publicada | (D) por defecto, pesos recalculados | (P) fase 1 | (T) afinada | T frente a (a) | T frente a (D) | P frente a (D) |
|---|---|---|---|---|---|---|---|---|
| 6 | 4 | 4,828 | 4,803 | 4,797 | 4,794 | −0,7 % | −0,2 % | −0,1 % |
| 14 | 4 | 5,544 | 5,239 | 5,156 | 5,184 | −6,5 % | −1,0 % | −1,6 % |
| 30 | 4 | 6,228 | 6,087 | 5,970 | 5,987 | −3,9 % | −1,6 % | −1,9 % |
| 60 | 4 | 8,865 | 9,074 | 9,281 | 9,233 | **+4,1 %** | +1,7 % | **+2,3 %** |
| 90 | 4 | 9,455 | 9,514 | 9,408 | 9,283 | −1,8 % | −2,4 % | −1,1 % |
| 180 | 2 | 9,755 | 9,773 | 10,379 | 10,365 | **+6,3 %** | **+6,1 %** | **+6,2 %** |
| Todos | 22 | 7,236 | 7,201 | 7,237 | 7,212 | −0,3 % | +0,2 % | +0,5 % |

**Brier de la mayoría absoluta de los bloques**

| Horizonte | Casos | (a) publicada | (D) por defecto, pesos recalculados | (P) fase 1 | (T) afinada | T frente a (a) | T frente a (D) | P frente a (D) |
|---|---|---|---|---|---|---|---|---|
| 6 | 4 | 0,0428 | 0,0416 | 0,0416 | 0,0419 | −2,1 % | +0,6 % | −0,0 % |
| 14 | 4 | 0,0416 | 0,0469 | 0,0461 | 0,0469 | **+12,8 %** | −0,1 % | −1,9 % |
| 30 | 4 | 0,0724 | 0,0709 | 0,0689 | 0,0688 | −4,9 % | −2,9 % | −2,8 % |
| 60 | 4 | 0,0926 | 0,0934 | 0,0834 | 0,0834 | −9,9 % | −10,7 % | −10,7 % |
| 90 | 4 | 0,1085 | 0,1082 | 0,1056 | 0,1058 | −2,5 % | −2,3 % | −2,5 % |
| 180 | 2 | 0,1368 | 0,1373 | 0,1496 | 0,1497 | **+9,4 %** | **+9,1 %** | **+9,0 %** |
| Todos | 22 | 0,0775 | 0,0781 | 0,0764 | 0,0767 | −1,1 % | −1,9 % | −2,2 % |

**Cobertura 95 del voto**

| Horizonte | Casos | (a) publicada | (D) por defecto, pesos recalculados | (P) fase 1 | (T) afinada | T frente a (a) | T frente a (D) | P frente a (D) |
|---|---|---|---|---|---|---|---|---|
| 6 | 5 | 0,960 | 1,000 | 1,000 | 1,000 | +0,040 | = | = |
| 14 | 5 | 0,960 | 0,920 | 0,920 | 0,920 | **−0,040** | = | = |
| 30 | 5 | 0,837 | 0,837 | 0,837 | 0,837 | = | = | = |
| 60 | 5 | 0,843 | 0,843 | 0,793 | 0,793 | **−0,050** | **−0,050** | **−0,050** |
| 90 | 5 | 0,763 | 0,763 | 0,803 | 0,803 | +0,040 | +0,040 | +0,040 |
| 180 | 3 | 0,717 | 0,717 | 0,783 | 0,783 | +0,067 | +0,067 | +0,067 |
| Todos | 28 | 0,856 | 0,856 | 0,861 | 0,861 | +0,005 | +0,005 | +0,005 |

**Cobertura 95 de los escaños**

| Horizonte | Casos | (a) publicada | (D) por defecto, pesos recalculados | (P) fase 1 | (T) afinada | T frente a (a) | T frente a (D) | P frente a (D) |
|---|---|---|---|---|---|---|---|---|
| 6 | 4 | 1,000 | 1,000 | 1,000 | 1,000 | = | = | = |
| 14 | 4 | 1,000 | 1,000 | 1,000 | 1,000 | = | = | = |
| 30 | 4 | 0,896 | 0,896 | 0,896 | 0,896 | = | = | = |
| 60 | 4 | 0,854 | 0,854 | 0,792 | 0,792 | **−0,062** | **−0,062** | **−0,062** |
| 90 | 4 | 0,804 | 0,804 | 0,804 | 0,804 | = | = | = |
| 180 | 2 | 0,675 | 0,675 | 0,675 | 0,675 | = | = | = |
| Todos | 22 | 0,889 | 0,889 | 0,878 | 0,878 | **−0,011** | **−0,011** | **−0,011** |

A horizonte (la ejecución `_h`, con la deriva hasta la elección), los cambios relativos y las diferencias de cobertura son:

| Horizonte | CRPS T/(a) | CRPS T/(D) | Brier T/(a) | Brier T/(D) | Cob. 95 voto T−(a) | Cob. 95 voto T−(D) | Cob. 95 escaños T−(a) | Cob. 95 escaños T−(D) |
|---|---|---|---|---|---|---|---|---|
| 6 | −0,6 % | −0,1 % | −2,6 % | +0,3 % | +0,040 | = | = | = |
| 14 | −6,4 % | −1,0 % | **+11,0 %** | −0,8 % | **−0,040** | = | = | = |
| 30 | −3,8 % | −1,7 % | −3,4 % | −1,3 % | = | = | = | = |
| 60 | **+3,8 %** | +1,7 % | −6,8 % | −7,1 % | = | = | = | = |
| 90 | −1,2 % | −1,7 % | −2,7 % | −3,1 % | = | = | = | = |
| 180 | **+5,6 %** | **+5,4 %** | **+8,6 %** | **+8,6 %** | = | = | = | = |

Casos mejores / peores, comparando cada caso (elección × horizonte) de `metrics.csv` con el mismo caso de la referencia:

| Métrica | Casos | T frente a (a) | T frente a (D) | P frente a (D) | T frente a (P) | D frente a (a) |
|---|---|---|---|---|---|---|
| MAE de voto | 28 | 18 / 10 | 16 / 12 | 17 / 11 | 14 / 14 | 16 / 12 |
| RMSE de voto | 28 | 16 / 12 | 14 / 14 | 19 / 9 | 12 / 16 | 14 / 14 |
| CRPS de escaños | 22 | 9 / 13 | 12 / 10 | 14 / 8 | 11 / 11 | 8 / 14 |
| Brier | 22 | 12 / 10 | 13 / 9 | 13 / 7 | 5 / 14 | 9 / 10 |
| CRPS de escaños, a horizonte | 22 | 9 / 13 | 13 / 9 | 14 / 8 | 9 / 12 | 9 / 13 |
| Brier, a horizonte | 22 | 14 / 8 | 13 / 8 | 14 / 7 | 6 / 11 | 10 / 9 |
| Cobertura 95 del voto | 28 | 3 / 2 | 2 / 1 | 2 / 1 | 0 / 0 | 1 / 1 |
| Cobertura 95 de los escaños | 22 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 0 |

Lo que dicen las tablas:

- **El afinado mejora el voto a corto plazo y en conjunto.**
  - Frente a (a), el MAE de voto baja en 5 de 6 horizontes: −1,8 % en todos los casos, con 18 casos mejores y 10 peores.
  - Frente a (D), el MAE de voto baja −0,2 %.
  - De 6 a 30 días el MAE de voto y el CRPS de escaños mejoran frente a las dos referencias, y el Brier a 30 días también.
- **Los incumplimientos se concentran en 60 y 180 días.** Son los mismos en T y en P, así que vienen de la fase 1.
  - **A 60 días**, en 2023 el promedio de Sumar sube de 16,00 a 16,75 frente a un resultado de 12,33, y el resultado sale por poco de la banda del 95 % (12,39-21,04).
    - Eso baja la cobertura del voto (−0,050) y de los escaños (−0,062), y empeora el CRPS de escaños.
    - El Brier de ese horizonte mejora un 10 %.
  - **A 180 días** sólo quedan 3 casos (2 con escaños), y uno decide: 2019-04 (el PSOE, 28,67, pasa de un promedio de 27,47 a 26,24; el PP, 16,69, se queda en torno a 24).
    - El CRPS de escaños empeora un 6 %, el Brier un 9 % y el MAE un 1,4 % frente a (D).
  - En los dos, sostener el último valor del promedio (`max_fc` 0) en lugar de prolongar su tendencia pierde cuando la tendencia sigue hasta la elección. Es la otra cara de la ganancia a 7 días de la búsqueda.
- **El Brier a 14 días (+12,8 % frente a (a)) y la cobertura del voto a 14 días (−0,040) no son del afinado.** (D) frente a (a) da lo mismo: +12,9 % y −0,040 (en 2023 el Brier pasa de 0,116 a 0,134, y en 2015 un partido sale de la banda). Es el precio de quitar la mirada al futuro de los pesos guardados. T frente a (D) a 14 días mejora el MAE, el CRPS y el Brier y no toca las coberturas.
- **La fase 2 no se nota.**
  - T frente a (P) queda dentro de ±1,8 % en todas las métricas y horizontes del nowcast: MAE −0,2 % y CRPS −0,3 % en todos los casos, 14 / 14 casos de MAE.
  - El Brier es algo peor en 14 casos de 22, con diferencias de milésimas: +0,3 % en todos los casos.
  - A horizonte, la cobertura del 95 % de los escaños a 30 días baja 0,042: un partido en un caso.
  - Es lo esperable de dos ajustes elegidos por ganancias del 0,05 % sobre 77 cortes.

## Decisión

**A8 no se cumple.** La primera mitad sí: la configuración afinada mejora el RMSE a 7 días en los dos ciclos de comprobación. La segunda no:

- **Frente a (a)**, la referencia publicada y el «antes» de A8, horizonte a horizonte:
  - a 14 días el Brier empeora un 12,8 % y la cobertura del voto baja 0,040;
  - a 60 días el MAE de voto empeora un 2,0 %, el CRPS de escaños un 4,1 % y bajan las dos coberturas del 95 %;
  - a 180 días el CRPS empeora un 6,3 % y el Brier un 9,4 %.
  - Ni siquiera en todos los casos juntos se cumple: el MAE (−1,8 %), el CRPS (−0,3 %) y el Brier (−1,1 %) mejoran, pero la cobertura del 95 % de los escaños baja (0,889 → 0,878, un caso).
- **Frente a (D)**, que aísla el afinado, quedan los incumplimientos de 60 días (coberturas) y de 180 días (CRPS +6,1 %, Brier +9,1 %), los mismos que trae (P) sola.

Recomendación para Luis: no cambiar los valores por defecto antes del 29-N. `backtest/tuning/best.json` queda como opción documentada: se puede evaluar con `--config` en cualquier backtest. Ninguno de los dos candidatos (T ni P) cumple A8, y el que se adopte debería ser el de la fase 1 sola, porque los dos ajustes de la fase 2 no se distinguen del ruido.

**Siguiente revisión: tras el 29-11-2026.** El resultado añadirá una sexta general al backtest y cerrará el ciclo en curso, que hoy sólo cuenta como comprobación hasta el 25-10. Antes de esa revisión merece la pena una búsqueda que no premie sólo 7 días:

- `max_fc` entre 0 y 10, o sólo en horizontes cortos;
- o la regla de A8 aplicada a todos los casos juntos en vez de horizonte a horizonte, con 3 casos a 180 días.

Las dos son decisiones de Luis.

## Salvedades

- **Los dos «antes» difieren en los pesos.**
  - La referencia (a) usa los pesos de sobrepublicación guardados en la base, calculados con todos los sondeos del ciclo: miran al futuro.
  - La tarea 4 midió en un corte (todo encendido) un RMSE a 7 días de 1,389 con los pesos guardados y de 1,240 con los recalculados en el corte.
  - En el backtest el recálculo, (D) frente a (a), mejora el MAE de voto (−1,6 % en todos los casos) pero empeora el Brier a 14 días un 12,9 %.
  - Por eso T se compara también con (D). Mientras los valores por defecto no cambien, el modelo publicado sigue usando los pesos guardados.
- **Las muestras son pequeñas.**
  - Hay 5 elecciones, y sólo 4 con escaños (2015 no los proyecta).
  - A 180 días son 3 casos, 2 con escaños.
  - El Brier de la mayoría vale 0,04-0,15: un 2 % es una diezmilésima, y una sola probabilidad que se mueve unas décimas lo rompe.
  - Las coberturas se mueven de 0,04 a 0,06 cuando un solo partido entra o sale de la banda en un solo caso.
- **Los casos omitidos son los mismos**, pero con otro motivo en `meta.error` (ver arriba): con `weights` el recálculo de los pesos falla con un error de numpy cuando no hay sondeos a 180 días. El caso se omite igual.
- **`config` en ECS.**
  - El job `backtest` lee la ruta de `config` del disco del contenedor, es decir, la copia del repositorio que lleva la imagen.
  - No la lee del sistema de ficheros de la app (S3), que es donde `tune` escribe `best.json`.
  - Para correr en ECS una configuración nueva hay que commitear `best.json` y reconstruir la imagen, o pasar la configuración como diccionario en el JSON del job.
- **Los keepalives de TCP valen para toda la app.** Están en `mtpy/core/dal/PostgreSQL.py`: un enlace muerto falla en unos 2 minutos en cualquier job y en la web, no sólo en el afinado.
- **El arreglo de la regresión de grado 0 vive en el modelo** (`mtpy/core/utils/stat.py`). Antes todo ajuste con `poly_deg=0` fallaba; el grado 1, el que se usa, queda idéntico.
- **La fase 2 movió dos ajustes por ganancias del nivel del ruido**: `he_n_iter` de 3 a 1 y `herd_prior` de 5 a 2, ≈ 0,05 % cada uno. El backtest lo confirma: T y P no se distinguen.
- **La guarda de rugosidad (1,5 veces) compara con el punto de partida de cada fase.** En la fase 2 ese punto es el `best` de la fase 1, no la configuración vigente. En la comprobación la afinada queda en 1,03-1,05 veces la vigente.
- **`mtpy/lib/tuning.py` tiene unas 2.200 líneas**: la búsqueda, los cortes, el pool, las partes y la copia a S3 en un solo módulo. Partirlo es una deuda pendiente, no un cambio de esta tarea.
