# Backtest del modelo

Evaluación fuera de muestra del modelo completo (promedio de encuestas, intervalos, escaños y probabilidades de mayoría) sobre las elecciones generales con resultado oficial desde 2015. Es distinto del backtest de las encuestadoras que hace el `Computer` (errores de cada encuesta, ratings): aquí se evalúa la **salida del modelo**.

## Protocolo

Para cada elección `E` y cada horizonte `d ∈ {6, 14, 30, 60, 90, 180}` días:

1. Se congela el modelo en `limit_date = E − d`: solo entran las encuestas publicadas al menos `d` días antes (`Simulator(drange=d)`), y los ratings, el estimador de error y el estimador de escaños del `Computer` se calculan solo con las elecciones anteriores a `E` (por construcción del `Simulator`, `get_event_dates(..., skip=1)`). Cada caso es genuinamente fuera de muestra.
2. Se ajusta el promedio (`fit_forecast(max_fc=10, fillna=True)`) y se simulan 500 elecciones en modo MT (`split=True`, `random=True`, `seed=42`) **dos veces con la misma semilla**: como *nowcast* (la elección celebrada en el ancla `as_of` del pronóstico, solo con el error de las encuestas; columnas sin sufijo) y **a horizonte** (`horizon='deadline'`: en una elección pasada el límite es la propia elección, así que se añade la deriva de la opinión desde `as_of` hasta ella; columnas con sufijo `_h`). Ver `docs/analisis-2026-09/metodo-5-nowcast-horizonte.md`.
3. Se compara con el resultado oficial nacional (`events_results`, `pct` sobre votos válidos, escaños).

Elecciones: 2015-12-20, 2016-06-26, 2019-04-28, 2019-11-10 y 2023-07-23. Los parámetros de 2015 y 2016 se derivan automáticamente de los datos (`get_event_params`), porque no tienen entrada en `data/params.json`. En 2015, UP y Cs no tienen resultado previo ni regla de herencia (`smap`), así que sus escaños no pueden proyectarse: el caso queda marcado `seats_valid = False` y solo cuenta para las métricas de porcentaje de voto.

## Métricas

Por caso (elección × horizonte), sobre los partidos principales (`bmaps.main`):

| Métrica | Qué mide |
|---|---|
| `mae_shares`, `rmse_shares` | Error absoluto medio y cuadrático del promedio (`forecast['mean']`) frente al porcentaje oficial |
| `mae_last_poll`, `mae_mean_4w`, `mae_prev_result` | Lo mismo para las tres líneas base: la última encuesta publicada, la media simple de las encuestas de las cuatro semanas anteriores al corte, y el resultado de la elección anterior |
| `bias_sum` | Error con signo de la suma de los partidos principales (negativo: el promedio dejó en `'-'` voto que fue a esos partidos) |
| `cov_fc95` | Cobertura del intervalo del 95 % del propio `Forecaster` (`fc_stat`: solo error estadístico de la media local) |
| `cov_shares50/80/95` | Cobertura de los intervalos empíricos de las cuotas simuladas (`shares()`, que incluyen el error histórico) |
| `mae_seats`, `mae_seats_all` | Error absoluto medio de escaños (titular `totals()`) en los principales y en todos los partidos |
| `cov_seats50/80/95` | Cobertura de los intervalos de escaños simulados |
| `crps_seats` | CRPS medio de la distribución de escaños de los principales (`E|X − y| − E|X − X'|/2`; es el error absoluto cuando la distribución es un punto) |
| `brier_vs`, `log_score_vs` | Calidad de la probabilidad de mayoría absoluta de los bloques `vs` (Derecha/Izquierda) frente a lo ocurrido |
| `cov_shares50/80/95_h`, `cov_seats50/80/95_h` | Coberturas de la ejecución a horizonte (con la deriva de la opinión hasta la elección) |
| `mae_seats_h`, `crps_seats_h`, `brier_vs_h`, `log_score_vs_h` | Las mismas métricas de escaños y de mayoría para la ejecución a horizonte |

Los casos con menos de 5 encuestas útiles se omiten (`meta.csv` registra el motivo). `by_horizon.csv` promedia las métricas de las elecciones para cada horizonte e indica cuántos casos entran (`n_cases`, `n_seats_cases`). Un modelo bien calibrado tiene coberturas cercanas al nivel nominal (0,50, 0,80, 0,95); si son menores, los intervalos son demasiado estrechos. El nowcast responde a "si las elecciones fueran hoy" y solo está calibrado a horizontes cortos; la comparación honesta a `d` días es la ejecución `_h`. `meta.csv` guarda además el ancla (`as_of`), los días hasta la elección (`horizon_max`) y la constante de deriva usada (`drift_k`, ajustada solo con los ciclos anteriores a cada elección).

## Ejecutar

```
python backtest/run_backtest.py                          # todo (unos 15 minutos con la base de datos local)
python backtest/run_backtest.py --events 2023-07-23 --horizons 6 30 --n-sim 200
python backtest/run_backtest.py --nowcast-only           # sin la ejecución a horizonte (la mitad de tiempo)
```

La deriva se lee de la tabla `elections.drift`; si está vacía, cada `Simulator` la calcula al vuelo (más lento). Se rellena con `Computer.compute_drift(save=True)` (notebook `data-load/PollsCompute`).

Escribe en `backtest/results/`: `shares.csv`, `seats.csv`, `blocks.csv` (una fila por partido o bloque y caso), `meta.csv` (cortes, encuestas usadas, tiempos), `metrics.csv` (una fila por caso), `by_horizon.csv` y `run.json` (commit, fecha, estado de la base de datos). La lógica está en `mtpy/lib/backtest.py` (`run_case`, `summarize_case`, `run_backtest`), reutilizable desde los notebooks.
