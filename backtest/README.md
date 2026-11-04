# Backtest del modelo

Evaluación fuera de muestra del modelo completo (promedio de encuestas, intervalos, escaños y probabilidades de mayoría) sobre las elecciones generales con resultado oficial desde 2015. Es distinto del backtest de las encuestadoras que hace el `Computer` (errores de cada encuesta, ratings): aquí se evalúa la **salida del modelo**.

## Protocolo

Para cada elección `E` y cada horizonte `d ∈ {6, 14, 30, 60, 90, 180}` días:

1. Se congela el modelo en `limit_date = E − d`: solo entran las encuestas publicadas al menos `d` días antes (`Simulator(drange=d)`), y los ratings, el estimador de error y el estimador de escaños del `Computer` se calculan solo con las elecciones anteriores a `E` (por construcción del `Simulator`, `get_event_dates(..., skip=1)`). Cada caso es genuinamente fuera de muestra.
2. Se ajusta el promedio (`fit_forecast(max_fc=10, fillna=True)`) y se simulan 500 elecciones en modo MT (`split=True`, `random=True`, `seed=42`) **dos veces con la misma semilla**: como *nowcast* (la elección celebrada en el ancla `as_of` del pronóstico, solo con el error de las encuestas; columnas sin sufijo) y **a horizonte** (`horizon='deadline'`: en una elección pasada el límite es la propia elección, así que se añade la deriva de la opinión desde `as_of` hasta ella; columnas con sufijo `_h`). Ver `docs/analisis-2026-09/metodo-5-nowcast-horizonte.md`.
3. Se compara con el resultado oficial nacional (`events_results`, `pct` sobre votos válidos, escaños).

Desde el método 6 el `Simulator` resta a cada encuesta el efecto de su casa (`house_effects=True`): el efecto del ciclo se estima solo con las encuestas hasta `limit_date` y su prior con las elecciones anteriores a `E` (tabla `pollsters_parties`), así que sigue siendo fuera de muestra. Las líneas base (`last_poll`, `mean_4w`) se calculan siempre con las encuestas brutas. `--no-house-effects` desactiva la corrección y `--industry-bias` añade el desplazamiento por el sesgo del sector (ver `docs/analisis-2026-09/metodo-6-house-effects.md`). El método 7 permite sortear juntos los partidos nacionales con una correlación negativa común fijada por la razón de composición estimada en las elecciones anteriores (`--composition auto`, o un valor fijo); por defecto se sortean independientes, porque el backtest no lo favorece a corto plazo (ver `metodo-7-composicion.md`). Desde el método 9 la proyección provincial lleva oscilaciones por comunidad autónoma y por provincia (`regional_noise=True`, por defecto), estimadas con las elecciones anteriores a cada una; `--no-regional-noise` deja el swing proporcional determinista (ver `metodo-9-oscilaciones-autonomicas.md`).

Elecciones: 2015-12-20, 2016-06-26, 2019-04-28, 2019-11-10 y 2023-07-23. Los parámetros de 2015 y 2016 se derivan automáticamente de los datos (`get_event_params`), porque no tienen entrada en `data/params.json`. En 2015, UP y Cs no tienen resultado previo ni regla de herencia (`smap`), así que sus escaños no pueden proyectarse: el caso queda marcado `seats_valid = False` y solo cuenta para las métricas de porcentaje de voto.

## Métricas

Por caso (elección × horizonte), sobre los partidos principales (`bmaps.main`):

| Métrica | Qué mide |
|---|---|
| `mae_shares`, `rmse_shares` | Error absoluto medio y cuadrático del promedio (`forecast['mean']`) frente al porcentaje oficial |
| `mae_last_poll`, `mae_mean_4w`, `mae_prev_result` | Lo mismo para las tres líneas base: la última encuesta publicada, la media simple de las encuestas de las cuatro semanas anteriores al corte, y el resultado de la elección anterior |
| `bias_sum` | Error con signo de la suma de los partidos principales (negativo: el promedio dejó en `'-'` voto que fue a esos partidos) |
| `cov_fc95` | Cobertura del intervalo del 95 % del propio `Forecaster` (`fc_stat`: solo el error estadístico de la media local, con cluster por casa desde el método 10). Mide dónde está el consenso de las casas, no el resultado: es baja por construcción y no es una prueba de calibración (ver `metodo-10-error-estandar-cluster.md`) |
| `cov_shares50/80/95` | Cobertura de los intervalos empíricos de las cuotas simuladas (`shares()`, que incluyen el error histórico) |
| `mae_seats`, `mae_seats_all` | Error absoluto medio de escaños (titular `totals()`) en los principales y en todos los partidos |
| `cov_seats50/80/95` | Cobertura de los intervalos de escaños simulados |
| `crps_seats` | CRPS medio de la distribución de escaños de los principales (`E|X − y| − E|X − X'|/2`; es el error absoluto cuando la distribución es un punto) |
| `brier_vs`, `log_score_vs` | Calidad de la probabilidad de mayoría absoluta de los bloques `vs` (Derecha/Izquierda) frente a lo ocurrido |
| `cov_shares50/80/95_h`, `cov_seats50/80/95_h` | Coberturas de la ejecución a horizonte (con la deriva de la opinión hasta la elección) |
| `mae_seats_h`, `crps_seats_h`, `brier_vs_h`, `log_score_vs_h` | Las mismas métricas de escaños y de mayoría para la ejecución a horizonte |
| `cov_prov_shares50/80/95`, `mae_prov_shares`, `rmse_prov_shares`, `n_prov_cells`, `n_prov_missing` | Calibración y error de las **cuotas por provincia** (celdas provincia × partido con cuota oficial o simulada ≥ 1 % que el modelo proyecta; `n_prov_missing` cuenta las que no proyecta, por falta de geografía previa), nowcast; solo en los casos con escaños válidos |
| `cov_prov_seats95`, `crps_prov_seats`, `mae_prov_seats` | Lo mismo para los escaños por provincia |

`meta.csv` registra además si se aplicaron efectos de casa (`house_effects`, `industry_bias`), cuántas casas se corrigieron entre los partidos principales (`he_pollsters`) y el tamaño de los efectos (`he_mean_abs`, `he_max_abs`, puntos), la razón de composición y la correlación usadas (`composition_ratio`, `composition_rho`) y la fracción de simulaciones con el residuo `'-'` recortado a 0 (`clip_rate`).

### Calibración (PIT, método 15)

`calibration.csv` (desde el 2026-11-01) guarda, para cada celda del backtest, el **PIT**: la fracción de simulaciones por debajo del resultado oficial, más la mitad de las iguales (`backtest.pit`). Si las bandas dicen la verdad, el PIT es uniforme: el resultado cae en cada décima de la banda una de cada diez veces; PIT amontonados en los extremos son bandas demasiado estrechas, y en el centro, demasiado anchas. Columnas: `scope`, `event_date`, `horizon`, `kind` (`shares` y `seats` nacionales, `prov_shares` y `prov_seats` por provincia), `mode` (`now`, el nowcast, o `h`, la ejecución a horizonte; las provincias sólo `now`), `party`, `region_id` (la provincia; vacío en las celdas nacionales), `main`, `official`, `model` y `pit` (vacío sin resultado oficial). Las celdas de escaños sólo existen con escaños válidos. `shares.csv`, `seats.csv` y `blocks.csv` llevan además los cuantiles de 5 en 5 (`q05`…`q95` y sus `_h`), que no se publican.

`summarize_calibration` resume el PIT por **población**, tipo, modo y grupo de horizontes (6-14, 30-60 y 90-180 días, `HORIZON_GROUPS`): las generales (`es`) y las autonómicas (los demás ámbitos), con sus partidos principales, deciden; las provincias (las celdas `prov_*` de cualquier ámbito, todos los partidos) sólo informan del reparto territorial (M9). Cada elección (ámbito y fecha) pesa lo mismo dentro de su población. `majority_reliability` compara la probabilidad de mayoría absoluta de los bloques con la frecuencia con que la hubo. La web lo enseña en la tarjeta «Fiabilidad de las bandas» de `/metodo/backtest`, con la parte `calibration` del backtest publicado (ver `docs/web/contrato.md`).

Parámetros de la incertidumbre (método 15), todos con el valor de antes por defecto (decisión del 02-11-2026, revisión tras el 29-N; ver `docs/analisis-2026-09/metodo-15-calibracion.md`): `error_model` (`polls`, el error de los sondeos sueltos, o `average`, el del propio promedio de la tabla `average_errors`; `--error-model`), `drift_scale` (`None`, `auto` o un número que multiplica la varianza de la deriva; `--drift-scale`) y `tail_dof` (`None` o los grados de libertad fijos de la t del sorteo; `--tail-dof`). `run.json` guarda los pedidos y `meta.csv` los usados (`error_model`, `drift_scale`, `drift_scale_rows` y `tail_dof`) en las ejecuciones desde el 01-11-2026.

Rating plano (fase C): `flat_ratings` (`--flat-ratings`) pone a 1 el peso por rating de todos los sondeos (también de las casas sin rating, que sin él llevan su calidad previa), de modo que el promedio corre sin el rating. Es excluyente con `--recompute-ratings` (`ValueError` antes de evaluar nada) y queda registrado en `run.json` y en la columna `flat_ratings` de `meta.csv`. Las corridas de la fase C viven en `backtest/results_flat` y `backtest/results_rated` y se leen en el método 16, [`docs/analisis-2026-09/metodo-16-rating.md`](../docs/analisis-2026-09/metodo-16-rating.md) (sección «¿Mejora el promedio? (modelo vivo, prueba 1a, 04-11-2026)»: sin el rating, el MAE de voto sube un 0,8 % en `es` y un 0,4 % en los cinco autonómicos, dentro del ±2 % del criterio 17).

La referencia de `es` commiteada en `backtest/results/` es anterior a esos tres parámetros (corrió con `45aaa8b`, el primer paso del método 15): trae `calibration.csv`, pero su `meta.csv` no tiene las columnas `error_model`, `drift_scale`, `drift_scale_rows` ni `tail_dof` y su `run.json` no tiene las claves, así que el `meta.params` que publica `publish backtest` las toma de `BACKTEST_RUN_DEFAULTS` (`polls`, `None`, `None`), que son justo la configuración con que corrió. Las carpetas de los autonómicos (`backtest/results/es-*`) son anteriores al PIT: no traen `calibration.csv` ni las claves, y su `meta.params` también sale de `BACKTEST_RUN_DEFAULTS`. Las tendrán tras `backtest` con `"scopes":"all"` en ECS, `backtest push` y `publish backtest`; hasta entonces la tarjeta «Fiabilidad de las bandas» sólo enseña las generales y las provincias de `es`.

Los casos con menos de 5 encuestas útiles se omiten (`meta.csv` registra el motivo). `by_horizon.csv` promedia las métricas de las elecciones para cada horizonte e indica cuántos casos entran (`n_cases`, `n_seats_cases`). Un modelo bien calibrado tiene coberturas cercanas al nivel nominal (0,50, 0,80, 0,95); si son menores, los intervalos son demasiado estrechos. El nowcast responde a "si las elecciones fueran hoy" y solo está calibrado a horizontes cortos; la comparación honesta a `d` días es la ejecución `_h`. `meta.csv` guarda además el ancla (`as_of`), los días hasta la elección (`horizon_max`) y la constante de deriva usada (`drift_k`, ajustada solo con los ciclos anteriores a cada elección).

## Ejecutar

```
python job.py backtest                                   # todo (es); el script de abajo es su envoltorio
python job.py backtest '{"scopes":"all","horizons":[6,30]}'   # es y todos los autonómicos, cada uno en su carpeta
python backtest/run_backtest.py                          # todo (unos 15 minutos con la base de datos local)
python backtest/run_backtest.py --events 2023-07-23 --horizons 6 30 --n-sim 200
python backtest/run_backtest.py --nowcast-only           # sin la ejecución a horizonte (la mitad de tiempo)
python backtest/run_backtest.py --no-house-effects       # sin restar los efectos de casa (referencia del método 6)
python backtest/run_backtest.py --industry-bias --out backtest/results_ib   # con el sesgo del sector, en otra carpeta
python backtest/run_backtest.py --composition auto --out backtest/results_comp   # sorteo conjunto (método 7)
python backtest/run_backtest.py --scope es-md                                    # un ámbito autonómico (método 11)
python backtest/run_backtest.py --scopes all --horizons 6 30                     # todos los autonómicos, cada uno en su carpeta
python backtest/run_backtest.py --scopes all --min-events 1 --out files/backtest/own       # estimadores propios de cada ámbito en todos los casos
```

Desde 2026-10-17 el trabajo lo hace el job `backtest` (`mtpy/jobs/Backtest.py`; parámetros y salida en
`deploy/README.md`, "Backtest del modelo"), con las mismas opciones en JSON y los ámbitos aislados (uno que falla no
detiene a los demás); `backtest/run_backtest.py` solo traduce sus opciones al job. Ojo: en el job `"scopes": "all"`
incluye `es`; en el script `--scopes all` son solo los autonómicos.

Dónde quedan las tablas: `backtest/run_backtest.py` escribe en la copia del repositorio (`backtest/results`, o
`--out`), mientras que `publish backtest` y `publish event` leen el sistema de ficheros de la app (S3 con
`S3_BUCKET`; `files/` sin él). Tras una ejecución local, hay que copiarlas con `push` (el script imprime la orden al
terminar):

```
set -a; . deploy/elections.env; set +a; python job.py backtest '{"action":"push","scopes":["es"]}'
```

o publicar desde la copia del repositorio con `"source":"backtest/results"`. Sin eso, `publish event` rehúsa
(`refused`) republicar sin modelo una elección que ya lo tiene publicado. El job `backtest` lanzado en ECS escribe
directamente en S3 (ver `deploy/README.md`, "Backtest del modelo").

Desde el método 11 el backtest vale para cualquier ámbito del catálogo (`data/es-scopes.csv`). En un ámbito autonómico los estimadores (error de sondeo, deriva) se ajustan con las elecciones de `es`; con `--min-events k` se ajustan con las propias en los casos con al menos `k` elecciones `featured` anteriores (la validación del método 11 no encontró ventaja en los propios). `meta.csv` registra el ámbito usado (`est_scope`) y el recuento (`n_featured`); `--min-events 1` fuerza los propios en todos los casos, lo que permite comparar las dos variantes caso a caso. `--scope es` (por defecto) escribe en `backtest/results/`, como siempre; cada ámbito autonómico escribe en `backtest/results/{scope}/` y, sin `--events`, evalúa sus elecciones `featured` (resultado oficial y al menos 10 sondeos útiles) celebradas desde 2019. Un ámbito sin elecciones evaluables se anuncia y se salta. En los ámbitos autonómicos las "provincias" de `provinces.csv` son sus circunscripciones (provincias, islas o zonas), y la cuota oficial por circunscripción de las candidaturas sin escaño es una estimación plana (ver `metodo-11-ambitos-autonomicos.md`).

La deriva se lee de la tabla `elections.drift`; si está vacía, cada `Simulator` la calcula al vuelo (más lento). Se rellena con `Computer.compute_drift(save=True)` (notebook `data-load/PollsCompute`).

Escribe en `backtest/results/` (la copia del repositorio con el script; el sistema de ficheros de la app con el job sin `out`): `shares.csv`, `seats.csv`, `blocks.csv` (una fila por partido o bloque y caso; desde el 2026-10-24 `blocks.csv` lleva también los intervalos de escaños de cada bloque, `lo50`…`hi95` y sus `_h`, cuantiles de la suma de sus partidos simulación a simulación, que el detalle de la elección enseña como «Rango 95%»), `provinces.csv` (una fila por provincia, partido y caso), `series.csv` (una fila por día y partido: la serie diaria del promedio, con su banda, del ajuste del horizonte más corto de cada elección, hasta el día en que se congeló el modelo; fuera de git por tamaño, la lee `publish event` para la curva del detalle de la elección: vive con las demás tablas en el sistema de ficheros de la app, así que ya no ata `publish event` a la máquina que corrió el backtest), `meta.csv` (cortes, encuestas usadas, tiempos), `metrics.csv` (una fila por caso), `by_horizon.csv`, `calibration.csv` (el PIT de cada celda, ver «Calibración») y `run.json` (commit, fecha, estado de la base de datos). La lógica está en `mtpy/lib/backtest.py` (`run_case`, `summarize_case`, `run_backtest` y `backtest_scope`, que ejecuta un ámbito y escribe su carpeta), reutilizable desde los notebooks.

## Sondeos futuros (afinado del promedio, método 14)

El backtest de elecciones sólo tiene cinco generales, pocas para elegir entre configuraciones del promedio que difieren en décimas. El afinado usa como prueba fuera de muestra los sondeos publicados después de cada fecha. Lógica en `mtpy/lib/tuning.py`, job `tune` (`mtpy/jobs/Tune.py`) y envoltorio `backtest/run_tuning.py`. Método, resultados y decisión en `docs/analisis-2026-09/metodo-14-sondeos-futuros.md`; spec en `docs/superpowers/specs/2026-11-01-sondeos-futuros-y-calibracion-design.md`, parte A.

### Protocolo (spec A2-A7)

- **Cortes.**
  - Hay uno cada 7 días (14 en la fase 2) en los 365 días anteriores a cada elección, y en el ciclo en curso hasta hoy.
  - Cada corte necesita al menos 5 sondeos útiles en los 90 días anteriores.
  - En cada corte el `Forecaster` se congela como en el backtest (`drange` = días hasta la elección), con sólo los sondeos publicados hasta el corte.
  - Los pesos de sobrepublicación se recalculan con esos sondeos (`tuning.reweighted_overrides`), sin los guardados en la base, que miran al futuro.
- **Predicción y puntuación.**
  - Cada sondeo publicado en los `k` días siguientes que entraría en el promedio se predice: promedio del corte en su fecha más el efecto de casa de su encuestadora.
  - Se puntúa el RMSE a `k` = 7 y 28 días, con todas las celdas (sondeo × partido) iguales y todos los ciclos iguales.
- **Particiones.**
  - Búsqueda: las generales de 2008 a 2019-11.
  - Comprobación: 2023-07-23 y el ciclo en curso (2026-11-29).
- **Búsqueda.**
  - Es un descenso por coordenadas sobre una rejilla fija (`tuning.GRID`).
  - Se elige por RMSE a 7 días, siempre que el de 28 no empeore más de un 2 % y la rugosidad no pase de 1,5 veces la del punto de partida de la fase.
  - Fase 1, `smoothing` (`bw`, `bw_cap`, `poly_deg`, `max_fc`, `wspan`): efectos de casa, muestra efectiva y rebaño apagados.
  - Fase 2, `shrinkage` (`tau`, `he_n_iter`, `herd_prior`): todo encendido, desde el `best.json` de la fase 1.
- **Comprobación** (`check`): el `best.json` frente a la configuración vigente en los ciclos reservados.
- **Criterio A8**: los valores por defecto sólo cambian si se cumplen las dos condiciones.
  - El RMSE a 7 días mejora en la comprobación.
  - El backtest de elecciones (`--config`, abajo) no empeora MAE de voto, CRPS de escaños ni Brier más de un 2 % ni baja la cobertura del 95 %.

### Tablas (`backtest/tuning/`, `es` en la raíz, los demás ámbitos en `{ámbito}/`)

| Fichero | Qué tiene |
|---|---|
| `cuts.csv`, `scores.csv`, `trials.csv`, `run.json` | La última ejecución (cada modo los reescribe): un corte por fila con sus segundos, las puntuaciones por configuración, ciclo y horizonte, las pruebas del descenso y el registro (commit, estado de la base) |
| `trials-{modo}.csv`, `scores-{modo}.csv` | Las pruebas y puntuaciones de cada modo de búsqueda (`smoothing`, `shrinkage`) y de la comprobación (`check`), que no se pisan |
| `best.json` | La configuración elegida, etiquetada con los ajustes que difieren de la vigente; la escriben `smoothing`, `shrinkage` y `merge` |
| `cuts-`, `scores-`, `trials-shrinkage-{ciclos}.csv`, `run-shrinkage-{ciclos}.json` | Las partes de la fase 2 repartida por ciclos (abajo) |
| `polls.csv` | Una fila por sondeo futuro y partido; no se versiona |

### Ejecutar

```
python backtest/run_tuning.py --mode smoothing --workers 8      # fase 1 (es)
python backtest/run_tuning.py --mode shrinkage --workers 8      # fase 2, desde el best.json de la fase 1
python backtest/run_tuning.py --mode check --workers 8          # best.json frente a la vigente en 2023 y 2026
python backtest/run_tuning.py --config default --config bw=fixed,45 bw_cap=1.5   # evaluar configuraciones (modo evaluate)
python backtest/run_tuning.py --mode shrinkage --cycles 2019-11-10              # una parte de la fase 2
python backtest/run_tuning.py --mode merge                                       # unir las partes
```

El job `tune` hace el trabajo con los mismos parámetros en JSON:

```
python job.py tune '{"scopes":["es"],"mode":"smoothing","workers":8}'
python job.py tune '{"scopes":["es"],"mode":"evaluate","configs":["default",{"bw":["fixed",45]}]}'
python job.py tune '{"action":"push","scopes":["es"]}'
python job.py tune '{"action":"pull","scopes":["es"]}'
```

- **Modos**:
  - `smoothing`, `shrinkage` y `check`: las fases y la comprobación;
  - `evaluate`: puntúa las configuraciones dadas. Su `"default"` es `default_config()` tal cual, con los pesos guardados en la base (que miran al futuro), mientras que la fila `default` de `check` es el default normalizado, con los pesos recalculados en el corte: las dos filas se llaman igual y no coinciden;
  - `merge`: une las partes.
- **Dónde escribe**:
  - `backtest/run_tuning.py` escribe en la copia del repositorio.
  - El job sin `out` escribe en el sistema de ficheros de la app (S3 con `S3_BUCKET`).
  - `push` copia las tablas de la copia del repositorio a la app y `pull` las trae de vuelta.
- **`--workers`** reparte los cortes de cada evaluación entre procesos. Cada proceso tiene su app y su motor de base de datos, y un vigilante de 15 minutos por corte. Las tablas son las de la ejecución secuencial.
- **Fase 2 por ciclos.**
  - `shrinkage` con `cycles` es una parte: evalúa el producto entero de `tau` × `he_n_iter` × `herd_prior` (18 configuraciones) desde el punto de partida y nunca escribe `best.json`.
  - `merge` repite el descenso sobre la tabla junta y da exactamente el resultado de la fase entera.
  - Cuesta el triple de CPU que la fase de una vez, porque cada parte evalúa las 18 configuraciones en vez de las 6 del descenso.

**Coste medido** en el portátil, con un pool de 8 procesos y la base de AWS:

- Medida previa, sólo en `es`:
  - 6,6 s por corte con efectos, muestra efectiva y rebaño apagados;
  - 43,5 s con todo encendido.
- Ejecución real de `es`:
  - fase 1: 8.514 cortes a 6,1 s;
  - fase 2: 462 a 29,6 s;
  - comprobación: 204 a 53,4 s;
  - en total, 21,3 h de CPU en 2,7 h de reloj.

### Backtest con una configuración

`--config PATH` (`config` en el job `backtest`) corre el backtest de elecciones con una configuración del promedio, la del `best.json` u otra con las mismas claves:

- Sus `reg_params`, `he_params` y `disp_params` van al `Simulator`.
- Sus interruptores y su `max_fc` mandan sobre los indicadores sueltos; el job rechaza uno que la contradiga.
- Con `weights`, los pesos se recalculan sobre los sondeos de cada corte.

`run.json` guarda la configuración entera, y `meta.csv` su etiqueta (`config`) y los sondeos del promedio con peso recalculado (`n_overrides`; 0 sin `weights`, vacío sin configuración). Hay que escribirlo siempre en otra carpeta:

```
python backtest/run_backtest.py --config backtest/tuning/best.json --out backtest/results_tuned
```

Resultado del 03-11-2026 (tablas en el método 14):

- La configuración afinada (`bw_cap=1.0; max_fc=0; he_n_iter=1; herd_prior=2.0`) mejora el voto de 6 a 30 días.
- Empeora el CRPS de escaños y las coberturas del 95 % a 60 días, y el CRPS y el Brier a 180 días: **no cumple A8**.
- Sobre todos los casos juntos, el resultado depende del modo.
  - En la ejecución a horizonte (`_h`) se cumple frente a las dos referencias.
  - En el nowcast falla sólo por la cobertura del 95 % de los escaños: un partido en un caso, Sumar en 2023 a 60 días.
  - La regla y el modo que deciden son de Luis.
- Los valores por defecto no cambian. `best.json` queda como opción, con la siguiente revisión tras el 29-11-2026.
