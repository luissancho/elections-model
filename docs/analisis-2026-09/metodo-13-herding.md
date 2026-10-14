# Método 13: herding (07-10-2026)

Cambios:
- `Forecaster.measure_herding`, `pool_herding`, `shrink_ratio` y `dispersion_factor`; `disp_params` `herding` y `herd_series`; columna `herd_ratio` de `fit_dispersion` (`mtpy/lib/forecaster.py`).
- `Simulator(disp_params=...)` (`mtpy/lib/simulator.py`).
- `Computer.get_herding_data`, `compute_herding`, `load_herding`, `herding_summary` y la columna `herding` de `print_ratings` (`mtpy/lib/computer.py`).
- Tabla nueva `elections.pollsters_herding`, que se crea al guardar: modelo `PollstersHerding`, `get_herding` y `save_herding_data` (`mtpy/models/elections.py`, `mtpy/lib/data.py`).
- Backtest con `herding` y `--herding` (`mtpy/lib/backtest.py`, `backtest/run_backtest.py`); paso `compute` de `load/run_load.py`.

Tests: `tests/test_forecaster_unit.py` y `tests/test_computer_unit.py` (M13), `tests/integration/test_herding.py`. Notebooks: `data-load/PollsCompute` (`compute_herding`) y `PollstersRatings` (sección "Herding").

## Problema

La muestra efectiva (M12, ver `metodo-6-house-effects.md`) baja el peso de las casas cuyas cifras oscilan **más** de lo que permite su muestra, pero nunca toca a las que oscilan **menos**. Eso es el herding en sentido estricto: cifras pegadas al consenso. Con M12 ya se veía que casi todas las casas, salvo el CIS, estaban por debajo de 1, pero la medida de M12 no sirve para afirmarlo. Se calcula frente a un promedio que incluye las encuestas de la propia casa, y una casa que publica mucho arrastra el promedio hacia sí y parece más estable de lo que es.

## Medida

`Forecaster.measure_herding` compara, para cada casa con 5 o más encuestas que publican su muestra:

- **El residuo de sus cifras publicadas frente al promedio calculado sin sus encuestas.** Es un leave-one-out: se pone a 0 el peso de la casa y se reajusta el promedio. Al residuo se le resta su media, que es el efecto de la casa.
- **La varianza de muestreo de su muestra publicada**, `p (1 − p) / n`, al nivel del promedio. Se descartan las muestras imputadas (`proc_sample` rellena las que faltan y recorta las grandes; con una muestra recortada, una casa grande parecería rebaño sin serlo).

El cálculo usa las cuatro series principales del ciclo, por nivel medio. `ratio = √(Σ residuo² / Σ varianza esperada)`, sumando sobre las series. El `p_value` es la cola inferior de una χ² con `n − 1` grados de libertad: es conservador, porque trata las cuatro series de una encuesta como si fueran una sola (comparten los encuestados).

Un ratio por debajo de 1 puede tener cuatro causas que la medida no distingue:
1. seguir al consenso;
2. entrevistar a un panel fijo;
3. ponderar por recuerdo de voto;
4. suavizar las propias oleadas.

Para el promedio las cuatro tienen el mismo efecto: sus encuestas aportan menos información independiente de la que sugiere su muestra. Que el promedio no recoja del todo los movimientos reales de la opinión y el redondeo de las cifras solo pueden añadir dispersión. Si algo, la medida subestima el herding.

Se comprobó también contra el consenso publicado antes de cada encuesta: la media de las demás casas en los 21 días anteriores, que es lo que una casa puede copiar. Da los mismos resultados.

## Resultados (`es`, 4 series principales)

Ratio por ciclo de las casas con 5 o más encuestas con muestra publicada:

| Casa | 2015 | 2016 | 2019-04 | 2019-11 | 2023 | 2026 (en curso) |
|---|---|---|---|---|---|---|
| Hamalgama | — | — | — | 0,37 | 0,63 | **0,49** |
| Celeste Tel | 1,37 | 0,43 | 1,15 | 0,62 | 0,80 | **0,51** |
| NC Report | 1,19 | 0,74 | 1,01 | — | 0,75 | **0,58** |
| SocioMétrica | — | — | 1,03 | 0,77 | 0,76 | **0,78** |
| Simple Lógica | 2,23 | 1,37 | 1,22 | 1,19 | 1,15 | **0,56** |
| GAD3 | 1,24 | 0,79 | 1,28 | 0,98 | 0,98 | 0,74 |
| 40dB | 1,62 | — | 0,94 | — | 1,02 | 0,82 |
| Sigma Dos | 1,94 | 0,53 | 1,16 | 0,67 | 1,17 | 0,88 |
| GESOP | 1,43 | 0,26 | 1,16 | — | 1,36 | 1,00 |
| Metroscopia | 2,90 | 1,30 | 1,77 | — | 1,51 | — |
| CIS | 2,11 | — | 1,93 | — | 2,53 | **3,16** |

Hay cuatro casas con herding persistente: Hamalgama, Celeste Tel desde 2016, NC Report y SocioMétrica. El ciclo actual está por debajo de 1 en casi todas las casas; Simple Lógica cambia de perfil, aunque no publica desde septiembre de 2024. El CIS y Metroscopia oscilan siempre de más; el CIS, en el ciclo actual, como una muestra de unas 400 entrevistas siendo de 4.000. Los ciclos cortos (2016, 2019-11) dan ratios más bajos en casi todas las casas, porque en pocas semanas la opinión se mueve menos de lo que el promedio deja sin explicar en un ciclo largo.

## En el ranking

`Computer.herding_summary(event_date)` agrega los ciclos hasta `event_date`, incluido el ciclo en curso, porque el herding no necesita el resultado. Suma los residuos al cuadrado con un decaimiento por antigüedad `year_decay = 0,9` por año, como el rating: `herding = √(Σ d · ss_obs / Σ d · ss_exp)`. `print_ratings` lo muestra como columna centrada en 1. **Es informativo: no entra en el rating**, por decisión de Luis, para no penalizar dos veces; la penalización, si la hay, va por el peso en el promedio.

## M12 simétrico (`disp_params={'herding': True}`)

`dispersion_factor` deja el multiplicador de `weight_sample` así:

- `1 / ratio` si la casa oscila de más, como hasta ahora;
- su `herd_ratio` si oscila de menos. Es el ratio leave-one-out encogido hacia 1 con `n / (n + 5)`, con suelo `1 / max_ratio`.

Con `weight_sample ∝ √n`, los dos casos equivalen a una muestra efectiva `n · factor²`. En el ciclo actual: Hamalgama ×0,59, Celeste Tel ×0,64, NC Report ×0,66, Simple Lógica ×0,68, SocioMétrica ×0,81, GAD3 ×0,82, 40dB ×0,84, DYM ×0,88, Target Point ×0,89 y Sigma Dos ×0,89. El PP pasa de 32,75 a 32,66 y el resto se mueve menos de 0,1 puntos. El cálculo cuesta unos 30 s más por ajuste (un ajuste sin cada casa en las cuatro series principales).

### Backtest (cinco elecciones, seis horizontes, 500 simulaciones, semilla 42)

La referencia reproduce exactamente `backtest/results/`.

| Métrica (media de 28 casos) | Referencia | M12 simétrico | Cambio | Casos mejor / peor |
|---|---|---|---|---|
| MAE voto | 2,399 | 2,403 | +0,2 % | 15 / 6 |
| RMSE voto | 2,936 | 2,938 | +0,1 % | 14 / 7 |
| CRPS escaños | 7,088 | 7,101 | +0,2 % | 12 / 7 |
| Brier mayorías | 0,0722 | 0,0708 | −2,0 % | 13 / 4 |
| Log-score mayorías | 0,246 | 0,242 | −1,2 % | 13 / 4 |
| CRPS escaños con horizonte | 7,026 | 7,037 | +0,2 % | 12 / 7 |
| Cobertura voto 95 % | 0,879 | 0,879 | = | — |
| MAE provincial | 2,754 | 2,757 | +0,1 % | 13 / 6 |

Lectura: es neutro, dentro del ruido. El promedio de los partidos principales se mueve 0,03 puntos de media; los casos mejoran más a menudo de lo que empeoran, y las probabilidades de mayoría mejoran un 1-2 %. Las medias de voto y escaños empeoran por un solo caso: 2016 a 14 días, con MAE de 2,41 a 2,63 y CRPS de 6,97 a 7,74. En ese ciclo de seis meses las casas estables acertaban (el PP subestimado pasa de 3,44 a 3,57). Sin ese caso, el M12 simétrico mejora o empata en todas las métricas. En 2023 mejora en todos los horizontes, aunque poco: el PSOE pasa de 5,51 a 5,47.

**Decisión (Luis, 07-10-2026): activado por defecto** (`disp_params['herding'] = True` en `Forecaster.set_disp_params`; `Simulator(disp_params={'herding': False})` y `--no-herding` lo desactivan), con el mismo criterio que M12: coherencia (cada casa pesa por la información que demuestra, en los dos sentidos) con un backtest neutro. `backtest/results/` contiene la ejecución con el herding activado.

Los backtests autonómicos (`backtest/results/es-*`, horizontes de 6 y 30 días) se reejecutaron con el herding activado. El efecto es prácticamente nulo: en 10 de las 17 comunidades las métricas no cambian, porque pocas casas llegan a 5 encuestas con muestra publicada en un ciclo autonómico. La media de las 17 queda así:

| Métrica | Antes | Con herding |
|---|---|---|
| MAE voto | 2,125 | 2,126 |
| CRPS escaños | 1,379 | 1,379 |
| Brier | 0,061 | 0,062 |
