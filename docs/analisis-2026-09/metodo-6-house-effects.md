# Método 6: efectos de casa (25-09-2026)

Cambios: `Forecaster.fit_house_effects` y sus funciones puras (`house_deviations`, `house_prior`, `normal_update`, `center_effects`, `apply_house_effects`, `industry_bias`), `series_raw`, parámetros `house_effects`/`he_params` (`mtpy/lib/forecaster.py`, `mtpy/lib/utils.py`); `Computer.get_house_effects_data / compute_house_effects / house_effects_summary / print_house_effects`, y en M6b `within_block_bias`, `block_error_decomposition` y el nuevo `error_weights` (`mtpy/lib/computer.py`); tabla `pollsters_parties` con columnas (`mtpy/models/elections.py`, `get_house_effects`/`save_house_effects_data` en `mtpy/lib/data.py`); `Simulator(house_effects=True, industry_bias=False)`; backtest con `--no-house-effects` y `--industry-bias`. Tests: `tests/test_forecaster_unit.py`, `tests/test_computer_unit.py`, `tests/test_utils.py`, `tests/integration/test_house_effects.py`, `tests/integration/test_model.py` (M6), `tests/integration/test_backtest.py`. Notebooks: `data-load/PollsCompute` (`compute_house_effects`), `PollstersRatings` (sección "Efectos de casa"), `PollsSimulations` (sección "Efectos de casa").

## Problema

Cada encuesta tenía su error con signo por partido frente al resultado (`polls_results.error`), pero nunca se agregaba por casa, y el rating de cada casa se construía con `bias_dev`, una medida por encuesta sin signo y sin partido. Nada se restaba: el promedio usaba las encuestas tal cual, así que su nivel dependía de qué casas habían publicado en las últimas semanas. Y hay casas con inclinaciones grandes y estables: en el ciclo 2023-2027 el CIS da al PP 5,2 puntos menos que el consenso de las demás casas y al PSOE 3,7 más (error típico 0,4 con 35 encuestas); DYM, SocioMétrica, InvyMark y NC Report dan al PP 2,2-2,7 más; GAD3, GESOP y NC Report al PSOE 1,5-1,9 menos. La dispersión entre casas es de 2,4 puntos para el PP y 1,7 para el PSOE, muy por encima del error de muestreo.

Tres hechos medidos guiaron el diseño (14 ciclos, 43 casas):

- Las inclinaciones **persisten a medias** de una elección a la siguiente (correlación 0,4-0,7 en los pares recientes; 2023 → 2027: 0,61): la historia es un buen prior, no un sustituto de la medida del ciclo.
- **El consenso suele tener más razón que la casa que se aparta**: la desviación frente al consenso correlaciona 0,35 con el error frente al resultado, y las casas a más de un punto del consenso fallan más (2,4 frente a 1,9 puntos).
- **El sesgo del sector** (media de todas las casas frente al resultado) es pequeño e inconsistente entre elecciones salvo para el PSOE (subestimado en las cinco últimas generales, −1,0 ± 2,4 puntos de media decaída).

## Método

### Medida en cada elección cerrada (`Computer.compute_house_effects`)

Por elección con resultado, casa y partido, la tabla `pollsters_parties` guarda:

- `dev_result`: media ponderada (`weight_over · weight_sample`) del error con signo de las encuestas de la casa en los últimos 90 días (`n_days`), y su error típico; `dev_result_c` la misma **centrada entre casas** por elección y partido (pesos: número de encuestas): la inclinación de la casa respecto al sector; `industry`: lo restado, el sesgo del sector.
- `dev_cycle`: la desviación de la casa frente al consenso del ciclo (todas sus encuestas, sin prior), medida con el propio `Forecaster`.
- `level`: la cuota oficial del partido.

Once elecciones (1993-2023), 1963 filas, 97 s. Los efectos por bloque no se guardan: en puntos son la suma de los efectos de los partidos del bloque (`house_effects_summary`).

### Prior y actualización (`Forecaster.fit_house_effects`)

El efecto de la casa `h` en la serie `b` para el ciclo actual es un posterior normal-normal (`normal_update`):

- **Prior** (`house_prior`): media de sus desviaciones relativas pasadas `dev_result_c / nivel` (nivel con suelo de 2 puntos, desviación relativa acotada a ±0,5), con peso `year_decay^años · min(n, 10)` y encogida hacia 0 con el peso de una elección "vacía" (`prior_events = 1`); en puntos, multiplicada por el nivel actual. Desviación típica del prior: `tau = 0,08` × nivel (la dispersión relativa observada entre casas es 0,03-0,13). Sin historia, prior 0.
- **Dato**: la desviación de sus encuestas brutas frente al promedio (`house_deviations`), con los pesos de las encuestas en el promedio (solapamiento, muestra y rating; desde el 07-10-2026, ver más abajo: antes pesos de precisión sin rating) decaídos por la antigüedad de la encuesta dentro del ciclo, y su error típico (suelo: un cuarto de la desviación típica del prior; casas con menos de 5 encuestas no informan).
- **Backfitting** (3 iteraciones): promedio sobre las encuestas corregidas → desviaciones de las brutas frente a él → posterior → **recentrado** por serie con el peso de cada casa en esa serie (`center_effects`: la media ponderada de los efectos es 0 con los mismos pesos que usa el promedio, así que el nivel medio del promedio en el ciclo no depende de la corrección; localmente sí cambia, que es justo lo que se corrige: qué casas publican alrededor de cada fecha) → resta a las encuestas brutas (`apply_house_effects`; las filas de resultados nunca se tocan) → repetir. `series_raw` conserva la serie sin corregir, y los gráficos (`plot_forecast_series`) dibujan siempre los sondeos publicados (`fc_series_raw`): la corrección sólo entra en la media.

La corrección opera sobre las series que de verdad se promedian, los bloques de `bmap` (en 2023, PP = PP+Cs y SUMAR = SUMAR+UP+MP; en 2027 cada partido es su bloque), y el prior de un bloque es la historia de su partido cabecera.

### Identificación y sesgo del sector

El dato del ciclo (encuesta − consenso) está centrado entre casas por construcción; el prior histórico se centra también (`dev_result_c`) para que mida lo mismo. Sin ese centrado, una casa con pocas encuestas absorbería el sesgo del sector y una con muchas no. El sesgo del sector es otro término, explícito y opcional (`Simulator(industry_bias=True)`, `Forecaster.industry_bias`): media decaída entre elecciones de `industry / nivel`, encogida hacia 0, con la dispersión entre elecciones como incertidumbre (la propia magnitud si solo hay una elección); desplaza el consenso y ensancha su error. Desactivado por defecto: el backtest lo juzga.

### El rating con el error descompuesto (M6b)

Una encuesta que reparte mal los votos entre PSOE y SUMAR pero clava la suma de la izquierda falla menos que una que se equivoca de bloque. `compute_errors` mide ahora, en la escala log-odds del rating, `bias_within`: el error de la cuota de cada partido principal **dentro de su bloque** (encuesta frente a resultado, ponderado por la cuota oficial), que es 0 si cada bloque se reparte como en el resultado sea cual sea el tamaño del bloque. El error de bloque ya existía (`bias_blocks`). El `bias` que alimenta `bias_dev` y el rating pasa de `0,7·bias_avg + 0,3·bias_blocks` (que contaba dos veces los bloques, porque `bias_avg` ya los contiene) a `0,7·bias_blocks + 0,3·bias_within` (`error_weights`; en log-odds `bias_within` es unas tres veces mayor que `bias_blocks`, porque la cuota del socio menor dentro de su bloque se mueve mucho en términos relativos, así que con 0,7/0,3 las dos componentes pesan de hecho parecido). `bias_within` y `error_within` se guardan como las demás medidas, por encuesta en `polls` y agregados por casa y elección en `pollsters_ratings` (columnas `numeric(16,2)` creadas a mano en ambas tablas); `bias_avg` se conserva. La misma descomposición en puntos (`block_error_decomposition`: `error_main = error_between + error_within`) da `error_within`, que se guarda junto a `error_blocks` por encuesta (`polls`) y por casa (`pollsters_ratings`).

## Efecto en 2027 (`as_of` 2026-09-14, MT n = 200, semilla 42; medido antes de M6b, ver más abajo los valores con el rating nuevo)

Efectos de casa aplicados (puntos, partidos principales; `n` encuestas en el ciclo):

| Casa | n | PP | PSOE | VOX | SUMAR | Derecha | Izquierda |
|---|---|---|---|---|---|---|---|
| CIS | 35 | −4,93 | +3,52 | −0,42 | +0,24 | −5,35 | +3,76 |
| DYM | 18 | +2,44 | −0,74 | −0,68 | −0,47 | +1,76 | −1,20 |
| InvyMark | 18 | +2,33 | +0,51 | −0,26 | −0,87 | +2,07 | −0,36 |
| NC Report | 22 | +2,21 | −1,51 | +0,13 | −0,71 | +2,34 | −2,22 |

Para el CIS el prior histórico (−0,72 al PP, +0,97 al PSOE, con desviación típica 2,5 y 2,2) pesa poco frente a 35 encuestas con error típico 0,4: el efecto es casi la medida del ciclo. El recentrado es de centésimas.

| | sin efectos | con efectos | con efectos y sesgo del sector |
|---|---|---|---|
| PP, promedio | 31,91 | 31,69 | 30,96 |
| PSOE, promedio | 27,10 | 27,32 | 28,32 |
| Escaños PP / PSOE / VOX / SUMAR | 137 / 112 / 62 / 8 | 137 / 113 / 61 / 8 | 132 / 118 / 63 / 8 |
| P(mayoría absoluta Derecha) | 0,975 | 0,97 | 0,87 |
| P(PP primero) | 0,92 | 0,895 | 0,72 |

Con efectos de casa el promedio de 2027 cambia poco (dos décimas en PP y PSOE): el recentrado conserva el nivel medio del ciclo (con los pesos de precisión; localmente el promedio sigue dependiendo de qué casas publican cerca de cada fecha, que es justo lo que se corrige) y el kernel de 70 días ya diluía la mezcla de casas; la diferencia máxima entre el promedio bruto y el corregido del PP a lo largo del ciclo es 0,29 puntos. Lo que cambia es la interpretación (cada encuesta se lee neta de su inclinación) y la robustez ante rachas en las que solo publica una casa. El sesgo del sector, en cambio, mueve mucho (PP −0,7, PSOE +1,0) con una incertidumbre enorme (±3,5 y ±2,4 puntos, que se suman al error): por eso va aparte y desactivado.

## Backtest

Cinco elecciones (2015-2023), seis horizontes, 500 simulaciones por caso y modo (nowcast y a horizonte), tres ejecuciones con los ratings anteriores a M6b: sin efectos de casa (referencia), con efectos (por defecto) y con efectos más sesgo del sector. Partidos principales:

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| MAE voto: sin efectos | 1,61 | 2,00 | 2,26 | 2,83 | 2,72 | 3,00 |
| con efectos | 1,63 | 1,92 | 2,15 | 2,82 | 2,79 | 2,99 |
| efectos + sector | 1,83 | 2,09 | 2,38 | 3,03 | 2,85 | 3,05 |
| RMSE voto: sin efectos | 1,94 | 2,49 | 2,88 | 3,40 | 3,33 | 3,62 |
| con efectos | 1,94 | 2,37 | 2,75 | 3,36 | 3,32 | 3,55 |
| efectos + sector | 2,30 | 2,66 | 3,05 | 3,57 | 3,42 | 3,68 |
| Cobertura voto 95 %: sin efectos | 0,96 | 0,88 | 0,84 | 0,84 | 0,84 | 0,87 |
| con efectos | 0,96 | 0,96 | 0,84 | 0,84 | 0,84 | 0,87 |
| efectos + sector | 0,96 | 0,96 | 0,89 | 0,93 | 0,93 | 0,87 |
| MAE escaños: sin efectos | 6,40 | 8,02 | 8,74 | 11,72 | 12,56 | 12,73 |
| con efectos | 6,43 | 7,38 | 8,64 | 11,71 | 12,82 | 12,60 |
| efectos + sector | 8,84 | 9,35 | 10,35 | 13,53 | 13,83 | 13,50 |
| CRPS escaños: sin efectos | 4,46 | 5,69 | 6,33 | 8,30 | 8,99 | 9,43 |
| con efectos | 4,57 | 5,32 | 5,98 | 8,38 | 9,13 | 9,28 |
| efectos + sector | 6,18 | 6,57 | 7,20 | 9,33 | 9,54 | 10,06 |
| Brier mayoría `vs`: sin efectos | 0,046 | 0,050 | 0,082 | 0,097 | 0,119 | 0,111 |
| con efectos | 0,039 | 0,045 | 0,076 | 0,097 | 0,110 | 0,101 |
| efectos + sector | 0,093 | 0,086 | 0,108 | 0,138 | 0,134 | 0,095 |

Lecturas:

1. **Los efectos de casa mejoran de forma modesta y consistente** a 14-30 días (MAE 2,00 → 1,92 y 2,26 → 2,15; RMSE 2,49 → 2,37 y 2,88 → 2,75; cobertura del 95 % a 14 días 0,88 → 0,96; CRPS de escaños 5,69 → 5,32 y 6,33 → 5,98) y las probabilidades de mayoría en todos los horizontes (Brier 0,046 → 0,039 a 6 días, 0,119 → 0,110 a 90). A 6 días son neutros (MAE 1,61 → 1,63): en la última semana publican todas las casas y el kernel ya las promedia. Por elección, a 6 días mejoran 2023 (1,43 → 1,28) y 2019-04 (1,65 → 1,50) y empeora 2016 (2,16 → 2,52), el ciclo corto en el que muchas casas cambiaron de métodos tras el fallo de 2015. En cada caso hay de media 11-12 casas con encuestas suficientes para informar su efecto (`he_pollsters`, de 1 a 19), con un efecto medio de 0,56 puntos y máximo de 4,7; las demás quedan en su prior. Se deja **activado por defecto**.
2. **El sesgo del sector empeora todo salvo la cobertura**: MAE 1,61 → 1,83 a 6 días, escaños 6,4 → 8,8, Brier 0,046 → 0,093; la cobertura sube (0,84 → 0,93 a 60-90 días) solo porque su incertidumbre (±2-3 puntos) ensancha los intervalos. El sesgo histórico del sector no predice el de la siguiente elección: cambia de signo entre elecciones para PP y VOX y se atenúa para el PSOE. Se deja **desactivado**, disponible (`industry_bias=True`, `--industry-bias`) como hipótesis explícita.
3. El coste real es menor que el estimado: 32 s por caso frente a 25 (7 s de backfitting).

### M6b: el rating con el error descompuesto

Tras recalcular el pipeline (`compute_errors` → `compute_deviations` → `compute_ratings`) con `bias = 0,7·bias_blocks + 0,3·bias_within`, los ratings apenas cambian (correlación 0,995 con los anteriores; entre las casas con 20 o más encuestas evaluadas, Sigma Dos 73,8 → 77,5, Celeste Tel 63,0 → 65,3, InvyMark 21,2 → 24,3 y NC Report 46,3 → 51,0, la casa que más gana: acierta los bloques mejor que el reparto). El backtest con efectos de casa y el nuevo rating es **neutro** frente al anterior (MAE de voto 1,63 → 1,63 a 6 días y 1,92 → 1,91 a 14; CRPS de escaños 5,32 → 5,28 a 14 días y 5,98 → 5,95 a 30; Brier ±0,002). Se probó también la mezcla 0,6/0,4: indistinguible (MAE 1,92 a 14 días, CRPS 5,29). Se mantiene: mide lo que se quería medir (bloque frente a reparto entre aliados), elimina la doble cuenta de los bloques y no cuesta nada. `backtest/results/` contiene esta ejecución (efectos de casa, rating nuevo).

Al cambiar `weight_rating` cambia también el promedio, y con él la regresión numérica de referencia (MT n = 200, semilla 42, sin efectos de casa): pasa de PP 137 / PSOE 112 / VOX 62 / SUMAR 8 a PP 139 / PSOE 111 / VOX 62 / SUMAR 8 (promedio del PP 32,16, PSOE 26,88); con efectos de casa, PP 137 / PSOE 113 / VOX 62 / SUMAR 8 (PP 31,72, PSOE 27,25; P(mayoría absoluta Derecha) 0,98 → 0,97; P(PP primero) 0,935 → 0,90). Los tests de integración usan ahora esta referencia.

Al corregir la revisión de este método se encontró y arregló un bug latente en `Stat` (`mtpy/core/utils/stat.py`): reescalaba **in situ** el array de pesos que recibía, así que reutilizar la misma serie de pesos fila a fila la corrompía en cuanto un dato era NaN. No se había manifestado porque `bias_avg` y `bias_blocks` nunca eran nulos; `bias_within` sí puede serlo (bloques de un solo partido), y dejaba `bias` nulo en 2995 encuestas. Test: `test_stat_does_not_mutate_the_callers_weights`.

## Salvedades

- **Momento frente a inclinación**: una casa que solo publica en un periodo mezcla su inclinación con el momento del ciclo; el backfitting contra el promedio de las demás lo atenúa y el decaimiento por antigüedad privilegia la metodología actual, pero no lo elimina.
- **Autoinfluencia**: el consenso incluye a la propia casa; una casa con muchas encuestas ve su desviación algo encogida. Dejar fuera la casa multiplicaría el coste por el número de casas.
- **Doble cuenta parcial**: la casa sesgada se corrige y además pesa menos (`weight_rating`), como en 538. Desde el 07-10-2026 las desviaciones y el recentrado usan los pesos del promedio, rating incluido (antes, pesos de precisión sin rating: ver el apartado final).
- **Partidos nuevos y sucesiones**: sin historia, prior 0; SUMAR no hereda la historia de UP.
- **Sesgo del sector**: inconsistente entre elecciones salvo el PSOE; solo debe activarse si el backtest lo respalda.
- **Coste**: unos 25 s más por `Simulator` (13 series × 3 iteraciones); `compute_house_effects` ≈ 100 s en el pipeline.

## 07-10-2026: recentrado con los pesos del promedio y muestra efectiva (M12)

Cambios: `Forecaster.fit_house_effects` (pesos del dato y del recentrado), `Forecaster.fit_dispersion` / `house_dispersion` / `apply_dispersion` / `set_disp_params`, parámetros `dispersion` / `disp_params` (`Forecaster`, `Simulator`), `fc_series_raw` y `plot_forecast_series` (`mtpy/lib/forecaster.py`, `mtpy/lib/simulator.py`); backtest con `--no-dispersion` (`mtpy/lib/backtest.py`, `backtest/run_backtest.py`). Tests: `tests/test_forecaster_unit.py` (5 nuevos), regresión de `tests/integration/test_model.py` re-fijada a 141 / 107 / 62 / 8. Notebook `PollsForecast` con `house_effects=True`.

### Problema

Con los ratings globales (M11) el promedio del PP con efectos de casa quedaba **1,0 puntos por debajo** del bruto a 13-10-2026 (32,96 → 31,95) y el PSOE 0,7 por encima, no las dos décimas de la sección anterior, y la diferencia era estructural: entre −0,35 y −0,85 en todos los trimestres del ciclo. La causa era el recentrado: los efectos sumaban cero con pesos de precisión (solapamiento × muestra, sin rating), un consenso en el que el CIS pesaba el 17,6 % por sus 35 barómetros de 4.000 entrevistas, mientras que en el promedio, con su rating (0,89 frente a 1,8–3,3 de las casas grandes), pesaba el 7 % en el ciclo y el 5 % en la ventana. El CIS, 5 puntos por debajo en el PP, arrastraba el cero: el resto de casas quedaba de media 1,06 por encima y, al corregirlas, el promedio bajaba. Es decir, con los efectos de casa activados el nivel del promedio dejaba de obedecer al rating y el CIS recuperaba por el recentrado el peso que el rating le quitaba. Descomposición medida con un contrafactual: −0,57 por el recentrado y −0,44 por la mezcla de casas que han publicado desde agosto (las más inclinadas al PP: DYM +2,4, InvyMark +2,3, NC Report +2,2, SocioMétrica +1,9).

### Cambios

1. **Un solo juego de pesos.** El dato (`house_deviations`) y el recentrado (`center_effects`) usan los pesos de las encuestas en el promedio (`weight`, rating incluido) decaídos por antigüedad. La diferencia media en el ciclo entre el promedio corregido y el bruto pasa de −0,64 a 0,00 en el PP (y a cero en todos los partidos); a 13-10-2026 queda −0,44, que oscila entre −0,43 y +0,34 a lo largo del ciclo: la mezcla de casas. El efecto del CIS en el PP pasa de −4,97 a −5,65 y el de GAD3 de +0,80 a +0,12; la cuota del CIS en el consenso, del 17,6 % al 7 %.
2. **Muestra efectiva** (`fit_dispersion`, antes de los efectos de casa). Para cada casa con 5 o más encuestas se compara la dispersión de sus encuestas alrededor del promedio, neta de su efecto (residuos centrados por partido, sumados sobre las series), con la varianza muestral `p (1 − p) / n` de sus muestras al nivel del promedio. La ratio, encogida hacia 1 con `n / (n + 5)` y acotada a [1, 10], divide `weight_sample` (que es proporcional a la raíz de la muestra): una casa cuyas cifras oscilan más de lo que su muestra permite pesa como la muestra que produciría esa dispersión. Nunca se premia la poca dispersión. Los pesos cambian en `series` y `series_raw`, de modo que los efectos de casa los usan. Medido en el ciclo actual: el CIS tiene una ratio de 2,95 (2,71 encogida; sus barómetros de 4.000 oscilan como una muestra de unas 460) y pesa un 37 %; **todas las demás casas están entre 0,46 y 0,98**, es decir, menos dispersas que su muestra: herding o cocina generalizados, con Hamalgama (0,51), NC Report (0,56), Celeste Tel (0,57) y Simple Lógica (0,59) como las más estables. La cuota del CIS en el promedio desde junio baja del 5 % al 2 %.

Promedio a 13-10-2026 con efectos de casa: PP 31,95 → 32,52 → 32,75 y PSOE 26,88 → 26,44 → 26,29 (bruto: 32,96 y 26,16).

### Backtest

Cinco elecciones, seis horizontes, 500 simulaciones, semilla 42, ratings actuales. Las referencias calculadas sobre sondeos brutos coinciden exactamente entre ejecuciones.

| Media de los 28 casos | MAE voto | RMSE voto | CRPS escaños | Brier mayoría `vs` |
|---|---|---|---|---|
| Sin efectos de casa | 2,355 | 2,887 | 6,98 | 0,072 |
| Recentrado anterior (pesos de precisión) | 2,322 | 2,823 | 6,89 | 0,064 |
| Recentrado con los pesos del promedio | 2,367 | 2,890 | 7,01 | 0,069 |
| Recentrado con los pesos del promedio + muestra efectiva | 2,399 | 2,936 | 7,09 | 0,072 |

| MAE de voto por horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| Sin efectos de casa | 1,62 | 2,02 | 2,24 | 2,82 | 2,72 | 2,96 |
| Recentrado anterior | 1,64 | 1,91 | 2,11 | 2,81 | 2,76 | 2,94 |
| Recentrado con los pesos del promedio | 1,68 | 1,95 | 2,18 | 2,86 | 2,80 | 3,00 |
| + muestra efectiva | 1,68 | 1,96 | 2,21 | 2,86 | 2,88 | 3,07 |

Lecturas:

1. **El backtest no premia ninguno de los dos cambios.** Cada uno empeora alrededor de un 1,5–2 % (acumulado, MAE 2,32 → 2,40 y CRPS 6,89 → 7,09); el primero en 20 de 28 casos y en 4 de 5 elecciones, el segundo en 20 de 25 casos con diferencia. La cobertura al 95 % no cambia salvo a 180 días con la muestra efectiva (0,87 → 0,80).
2. **Casi todo es el PSOE de 2023**, que el modelo subestimaba 5,1 puntos con el recentrado anterior, 5,4 con el nuevo y 5,5 con la muestra efectiva (5,5 sin efectos): las casas con mal rating daban más al PSOE y acertaron. El PP queda igual (2,17) y con menos sesgo. Las tres manipulaciones quitan voz al CIS en el nivel, y las tres pierden un poco: la ganancia medida de los efectos de casa en la sección anterior venía del consenso sin rating, no de corregir quién publica, y con el rating sólo queda a 14–30 días.
3. **Ámbitos autonómicos** (`backtest/results/{scope}/`, 6 y 30 días, 58 casos, mismos casos y mismas referencias brutas que la ejecución del 06-10-2026): neutro. Media por ámbito del MAE de voto 2,10 → 2,13 y del CRPS de escaños 1,37 → 1,38; sólo Castilla-La Mancha (dos casos) empeora de forma visible (2,49 → 3,01).
4. **Decisión: los dos cambios quedan activados por defecto**, por coherencia (un solo juego de pesos decide el nivel; los efectos de casa corrigen quién publica; cada casa pesa por la precisión que demuestra), aceptando que el backtest no lo respalda: las diferencias están dentro del ruido de cinco elecciones, aunque su signo es consistente. La muestra efectiva se desactiva con `Simulator(dispersion=False)` o `--no-dispersion`; el recentrado anterior no tiene interruptor (es una línea de `fit_house_effects`: los pesos del dato). `backtest/results/` contiene la ejecución con los dos cambios.

### Pendientes

- **Herding**: la medida de dispersión dice que todas las casas menos el CIS son más estables de lo que su muestra permite. 538 lo penaliza en el rating (ADPA frente al mínimo teórico); aquí el rating lo premia, porque mide el error relativo al consenso. Penalizarlo iría en contra de lo que el backtest acaba de premiar (más peso al nivel del CIS); no se ha probado.
- **Nivel del CIS**: tres experimentos consistentes sugieren que el nivel del CIS aportaba información que el consenso ponderado por rating no tiene, sobre todo en el PSOE. Puede ser el sesgo del sector sobre el PSOE (que `industry_bias` no logra capturar) o casualidad de 2023.
- **Varianza esperada**: usa muestreo aleatorio simple; el efecto de diseño real (1,2–1,5) y el error del propio promedio harían las ratios algo menores. No cambia el orden de las casas.
- `dev_cycle` de `pollsters_parties` se recalcula con los nuevos pesos en la próxima ejecución del pipeline (`compute_house_effects`); es diagnóstico, no alimenta el modelo.
