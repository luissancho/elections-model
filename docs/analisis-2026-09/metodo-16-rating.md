# Método 16: validación del rating de encuestadoras (04-11-2026)

Cambios:
- `Forecaster(weight_ratings='flat')`: pone a 1 el peso por rating de todos los sondeos de la serie, también de las casas sin fila de rating, que sin él llevan su calidad previa (`mtpy/lib/forecaster.py`, ebef322).
- Backtest con rating plano: `flat_ratings` en `run_case`, `run_backtest` y `backtest_scope`, columna `flat_ratings` de `meta.csv` y clave de `run.json`, job `backtest` (`"flat_ratings": true`) y `--flat-ratings` en `backtest/run_backtest.py`, excluyente con `--recompute-ratings` (ca8aacb).
- Módulo puro `mtpy/lib/rating_check.py`: correlación de rango, pendiente de calibración, acierto relativo frente a nada, a la calidad previa y al rating, cuartos, promedio simple con y sin rating, intervalos por bootstrap y el resumen de la parte (2a26f5f, e5068ae).
- `Computer.rating_check_data(scopes='all')`: reúne los pares casa-elección y los sondeos de la última campaña de cada elección valorada, sólo lectura (`mtpy/lib/computer.py`, 6dc0b09, b50b1a6).
- Parte `rating-check` del análisis nacional (`analysis-rating-check@1`, CSV `rating-check.csv` y `rating-check-elections.csv`), sólo con `rating_check` en el job `publish` (`mtpy/lib/publish/analysis.py::export_rating_check`, contrato en `docs/web/contrato.md`; 7055255).
- Tarjeta «Cómo lo comprobamos» de /metodo/rating con cifras vivas de la parte: KPIs, nube del diferencial previsto frente al observado, cuartos, promedio con y sin rating, «Cómo leerlo» y la nota `comprobacion` (`mtpy/lib/pages/metodo.py`, `web/templates/metodo_rating.html`, `web/dist/js/pages/metodo.js`; a3d6865, 64b6233 y los textos definitivos de este documento).

Tests: `tests/test_rating_check_unit.py`, `tests/test_forecaster_unit.py`, `tests/test_backtest_unit.py`, `tests/test_jobs_backtest.py`, `tests/test_pipelines_backtest.py`, `tests/test_computer_unit.py`, `tests/test_publish_unit.py`, `tests/test_bundle_unit.py`, `tests/test_jobs_publish.py`, `tests/test_pipelines_publish.py`, `tests/test_web_api.py`, `tests/test_pages.py` (la tarjeta con el fixture, sin la parte, con una parte degenerada y las frases de cada signo) y `tests/test_js_charts.py`. Spec: `docs/superpowers/specs/2026-11-04-rating-validacion-design.md` (fase C).

## Problema

El rating de cada encuestadora entra en el peso de sus sondeos en el promedio (`weight_rating`) y ordena el ranking público de /casas. Nunca se había comprobado por lo que hace. Lo medido antes apuntaba a poco contenido predictivo: la desviación de una casa varía entre elecciones (σ = 0,014 en log odds ratio) más de lo que las casas se separan entre sí (τ = 0,01, y una dispersión real de las medias de casa de unos 0,0034 descontado el ruido); la correlación del error de una casa entre generales salió 0,02 y de autonómicas a generales 0,04 (spec del 2026-10-20); y los cambios del rating no movieron el backtest.

La fase C no calibra parámetros contra el backtest de cinco generales, que sería perseguir 2023. **Valida**: mide fuera de muestra si el rating ordena a las casas de modo que anticipe quién se desviará menos, si la desviación que prevé se cumple en su medida, y si el promedio mejora al ponderarlo con él. Y publica la respuesta en /metodo/rating sea cual sea.

## Método

La regla común a las tres pruebas: el rating de una casa en una elección *e* es la fila (*e*, casa) de `pollsters_ratings`, calculada sólo con las elecciones de cualquier ámbito celebradas **estrictamente antes** de *e* (`rate_events`). Una elección es un ámbito y una fecha: dos autonómicas del mismo día son dos elecciones, y ninguna ve a la otra. Las casas de `drop_mtypes` (agregadores y paneles en línea abiertos) no tienen rating y no entran.

### 1a. Rating plano en el backtest del modelo vivo

El backtest completo (`backtest_scope`) dos veces con el mismo código, la misma base, la misma semilla y el mismo `n_sim`: con el rating (`backtest/results_rated`) y con `flat_ratings` (`backtest/results_flat`), en `es` (5 generales, 28 casos) y en los cinco autonómicos de la parte B del método 15 (`es-md`, `es-an`, `es-ct`, `es-vc`, `es-ga`). La referencia commiteada `backtest/results` no sirve de par: es de otro código (45aaa8b) y de otra base, y con un criterio del ±2 % eso confunde.

Juicio: el criterio 17 (método 15) caso a caso, con mejor / peor contados: MAE de voto, CRPS de escaños, Brier de mayorías y cobertura del 95 %; además, la comparación con `mae_last_poll` y `mae_mean_4w`, que ya son promedios sin rating. Lectura declarada de antemano: si plano ≈ con rating dentro del ±2 %, el rating es cosmético para la predicción.

### 1b. Rating plano en un promedio simple, en todas las elecciones valoradas

La prueba con potencia: todas las elecciones con resultado y rating, no cinco. Para cada elección valorada *e*, los sondeos útiles (`computed`, sin `drop_mtypes` ni `drop_ctypes`, `weight_over` > 0) de los **30 días** anteriores, con al menos **3 casas** distintas. Para cada partido principal del ciclo (`bmaps.main`; un principal hecho de varios partidos suma sus errores), el promedio ponderado del error publicado con dos pesos:

- `w_plano = weight_over · weight_sample`
- `w_rating = w_plano · weight_rating`, con el `weight_rating` de la fila (*e*, casa) o, para las casas sin fila, su calidad previa `log1p(q) / log1p(media de q)` (ver Salvedades).

El error del promedio de cada partido es `|Σ w·error / Σ w|`, y el MAE de la elección, la media sobre los principales con peso en los dos promedios. Resumen por población (todas, generales, autonómicas): la diferencia emparejada `con rating − plano` en puntos de voto, su media con IC del 95 % por bootstrap **de elecciones** (2.000 réplicas, semilla fija) y el recuento de elecciones mejor / peor con el rating (`simple_average_errors`, `summarize_simple_average`).

### 2. Discriminación y calibración por pares casa-elección

Un par es una casa con sondeos útiles en una elección valorada *e* y fila de rating en ella:

- **Desviación observada** `d_obs` (en la spec, `d_e`): la evidencia que el propio rating usa, `bias_dev` de `_event_deviations` sobre los sondeos de *e*, en log odds ratio. Es la media, con los pesos de posición del rating, de las desviaciones de sus sondeos frente al error medio de los sondeos de su fecha (`fit_bias_estimator`, regresión local), cada una contraída con N(0, τ). El error de un sondeo es `0,7·bias_blocks + 0,3·bias_within`: mide error absoluto, no sesgo con signo. Cada elección se pasa sola por `_rating_polls` (peso por año 1), así que `d_obs` no depende de la fecha de la comprobación.
- **Desviación prevista** `d_prev`: el `bias_dev_adj` de la fila (*e*, casa), la media contraída de sus elecciones anteriores, guardada en escala «bias» y devuelta a log odds ratio (`bias_to_lor`). Sin valor si la casa no tenía elecciones anteriores (`num_events_w = 0`).
- **Rating antes de *e***: `rating` (*R*, la mezcla con la calidad previa), `rating_adj` y `quality` (*Q*), guardados ×100 y aquí en [0, 1]; `num_events_w`, `weight_rating`.

Medidas (`rating_check.discrimination`, `skill`, `quarters`), con IC del 95 % por bootstrap **de casas** (todas las filas de una casa entran o salen juntas; 2.000 réplicas, semilla 0 más un desplazamiento por medida):

1. **Correlación de rango** de Spearman entre *R* y `d_obs`: negativa si las casas con más rating se desvían menos. También en dos cortes de evidencia, `num_events_w` ≥ 2 y < 2.
2. **Pendiente de calibración** β: mínimos cuadrados de `d_obs = a + β·d_prev` en los pares con `d_prev`. β = 1, la desviación prevista se cumple en su medida; 0, no anticipa nada; entre 0 y 1, lo previsto exagera las diferencias (la contracción hacia la media se queda corta); mayor que 1, se queda corto. También sobre las elecciones con 10 o más sondeos (`slope_polled`), como comprobación de la atenuación (Salvedades).
3. **Acierto relativo** (*skill*) frente a predecir `d_obs` con nada (0), con la calidad previa sola (−τ·Φ⁻¹(*Q*), la desviación que implicaría `rating_adj = Q`) y con el rating (−τ·Φ⁻¹(*R*)), con las probabilidades recortadas a [0,001, 0,999]: `skill = 1 − ECM(predictor) / ECM(0)`. Si *Q* sola iguala a *R*, la evidencia no añade nada al prior; si 0 iguala a los dos, nada predice.
4. **Cuartos**: dentro de cada elección con 4 o más casas, las casas ordenadas por *R* y partidas en cuartos (`pd.qcut` del rango); el error medio por partido (`error_avg`, en puntos de voto, ponderado por `weight_over · weight_sample · weight_pos`) de cada cuarto. Es la lectura para el público: «las casas mejor valoradas fallaron X frente a Y».
5. **Recuentos**: pares, casas, elecciones y fechas por población.

Los pares pesan igual (cada elección pesa por sus casas); no se aplica `weight_event`, porque la pregunta es predictiva. Las escalas: τ = 0,01, σ = 0,014, k = (σ/τ)² = 1,96. En la nube de la tarjeta, `d_prev` y `d_obs` van ×100 (log odds ratio por 100), muy cerca del «Diferencial» del ranking para desviaciones de este tamaño (la escala «bias» es (e^x − 1)·100).

Coste: `rating_check_data` sobre 18 ámbitos, 188 s, y el resumen con sus bootstraps, 58 s (medida del 04-11-2026 en el portátil, contra la base de AWS, sólo lectura). Por eso la parte sólo se recalcula con `rating_check: true` del job `publish`, nunca desde `update`.

## ¿Mejora el promedio? (modelo vivo, prueba 1a)

**Pendiente de las corridas.** Las cuatro corridas están en marcha en el portátil (04-11-2026), en dos carpetas hermanas ignoradas por git:

```bash
python backtest/run_backtest.py --scope es --out backtest/results_rated
python backtest/run_backtest.py --scope es --flat-ratings --out backtest/results_flat
python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --out backtest/results_rated
python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --flat-ratings --out backtest/results_flat
```

Cuando acaben: `run.json` de `results_flat` con `flat_ratings: true` y el de `results_rated` con `false`; las tablas de `by_horizon.csv` y `metrics.csv` emparejadas por (`event_date`, `horizon`), con las diferencias de `mae_shares`, `crps_seats`, `brier_vs` y `cov_shares95` en % y en puntos o escaños, el recuento mejor / peor por métrica, la comparación con `mae_last_poll` / `mae_mean_4w` y el juicio del criterio 17, por ámbito y horizonte. Esas tablas y la línea con fecha de la tarjeta («También repetimos las pruebas históricas del modelo completo…», hoy con los marcadores `{fecha}` y `{cifras}`) son la Task 8b.

## Resultados (pruebas 1b y 2, 04-11-2026)

Medidos con el código b50b1a6/e5068ae sobre la base del 04-11-2026, con `Computer('es').build_series().rating_check_data('all')` y `rating_check_summary`, y repetidos con el mismo resultado en una segunda pasada. El Spearman se reprodujo con un script independiente que lee los ratings de `get_ratings` directamente: −0,1848, igual.

### Recuentos

| Población | Pares | Con `d_prev` | Casas | Elecciones | Fechas | Elecciones del promedio simple |
|---|---|---|---|---|---|---|
| Todas | 942 | 832 | 112 | 78 | 22-06-1986 a 17-05-2026 | 71 |
| Generales | 238 | 187 | 75 | 13 | 22-06-1986 a 23-07-2023 | 13 |
| Autonómicas | 704 | 645 | 78 | 65 | 01-03-2009 a 17-05-2026 | 58 |

Los 110 pares sin `d_prev` son casas en su primera elección valorada: entran en la correlación, el acierto relativo y los cuartos, no en la pendiente. El promedio simple tiene 58 autonómicas, no 65: no entran las que no reúnen 3 casas con sondeos útiles en sus 30 días anteriores. La primera general con pares es la de 1986.

### Prueba 2: ¿ordena el rating? ¿Prevé la desviación en su medida?

| Población | Spearman (IC 95 %) | Spearman, `num_events_w` ≥ 2 | Spearman, `num_events_w` < 2 | β (IC 95 %) | Acierto del rating | Acierto de la calidad previa |
|---|---|---|---|---|---|---|
| Todas | −0,18 (−0,24 a −0,08) | −0,19 (−0,28 a −0,03), 578 | −0,18 (−0,29 a −0,04), 364 | 0,44 (−0,02 a 0,72) | −3,7 % | −16,1 % |
| Generales | −0,12 (−0,23 a +0,02) | −0,07 (−0,22 a +0,12), 117 | −0,11 (−0,29 a +0,06), 121 | 0,06 (−0,45 a 0,50) | −13,1 % | −24,4 % |
| Autonómicas | −0,21 (−0,28 a −0,08) | −0,22 (−0,32 a −0,02), 461 | −0,19 (−0,34 a −0,03), 243 | 0,55 (−0,01 a 0,92) | +0,5 % | −12,4 % |

`slope_polled` (β sólo con las elecciones de 10 o más sondeos) sale **idéntico** a β en las tres poblaciones (0,44, 0,06 y 0,55, con 832, 187 y 645 pares): todas las elecciones con pares tienen 10 o más sondeos, así que la comprobación no separa nada con estos datos y la tarjeta no la muestra (la nota lo dice). Sus intervalos sólo difieren por la semilla del bootstrap (−0,05 a 0,73 en todas).

La recta ajustada, en la escala de la nube (×100): de (−1,29, −0,53) a (1,30, 0,61); la constante es prácticamente 0. Las desviaciones previstas van de −1,29 a 1,30 y las observadas de −4,71 a 4,84: lo previsto ya está muy contraído y aun así exagera.

Los ECM de las tres predicciones salen 0,0002-0,0003 (log odds ratio al cuadrado) porque la parte redondea a 4 decimales: se publica el acierto relativo, no los ECM.

### Cuartos del rating (error medio por partido de cada par, en puntos)

| Población | Cuarto con más rating | 2.º | 3.º | Cuarto con menos rating | Pares (1.º / 2.º / 3.º / 4.º) |
|---|---|---|---|---|---|
| Todas | 2,71 | 2,85 | 2,83 | 3,28 | 262 / 227 / 206 / 247 |
| Generales | 2,68 | 2,86 | 2,95 | 3,23 | 63 / 59 / 54 / 62 |
| Autonómicas | 2,72 | 2,85 | 2,78 | 3,29 | 199 / 168 / 152 / 185 |

Los cuartos no tienen intervalo.

### Prueba 1b: el promedio simple con y sin rating (MAE por partido principal, en puntos)

| Población | Elecciones | Sin rating | Con rating | Diferencia (IC 95 %) | En % del error sin rating | Mejor / peor con rating |
|---|---|---|---|---|---|---|
| Todas | 71 | 1,929 | 1,940 | +0,011 (−0,020 a +0,048) | +0,5 % (−1,0 % a +2,5 %) | 37 / 34 |
| Generales | 13 | 1,657 | 1,661 | +0,004 (−0,058 a +0,066) | +0,2 % | 7 / 6 |
| Autonómicas | 58 | 1,990 | 2,002 | +0,012 (−0,020 a +0,060) | +0,6 % | 30 / 28 |

Las generales, una a una (diferencia con rating − sin rating, en puntos): 1986 −0,193; 1989 +0,202; 1993 −0,233; 1996 −0,028; 2000 −0,010; 2004 +0,126; 2008 +0,091; 2011 −0,022; 2015 +0,005; 2016 −0,017; abril de 2019 +0,013; noviembre de 2019 −0,006; 2023 +0,125. Las tres mayores diferencias son de campañas con pocas casas (4 en 1989, 7 en 1993, 9 en 1986); desde 2015, con 16 a 21 casas, ninguna pasa de dos centésimas salvo 2023 (+0,125).

## Lectura

- **El rating ordena, poco.** La correlación de rango es negativa y su intervalo no llega a 0 en el conjunto (−0,18) y en las autonómicas (−0,21): las casas con más rating antes de una elección se desvían, en conjunto, algo menos en ella. Es una correlación débil. En las generales solas (−0,12, 13 elecciones) el intervalo cruza el 0: no se distingue un orden en la población que más importa al ranking, y casi tres cuartos de los pares son autonómicos.
- **La evidencia no ordena mejor que el prior.** Las casas con más evidencia (`num_events_w` ≥ 2) no se ordenan mejor que las de menos (−0,19 frente a −0,18): el orden no crece con las elecciones acumuladas.
- **Los cuartos lo dicen en puntos.** El cuarto con más rating falló 2,71 puntos por partido y el de menos 3,28: 0,57 puntos, un 17 % menos. Los dos cuartos centrales casi no se distinguen (2,85 y 2,83). La mayor parte del error, unos 2,7 puntos, la tienen también las mejores casas: es el error compartido (M6), que ningún rating corrige.
- **La contracción se queda corta.** β = 0,44 y el intervalo no llega a 1 (0,72 como mucho): la desviación prevista exagera las diferencias, aunque ya va muy contraída. El intervalo toca el 0 (−0,02), así que no se puede descartar que la desviación prevista no anticipe nada. En las generales β = 0,06: no anticipa nada.
- **El acierto relativo es nulo.** Predecir la desviación con el rating es un 3,7 % peor que predecir 0 (un 13 % peor en las generales, un 0,5 % mejor en las autonómicas). No contradice al Spearman: el rating ordena algo, pero exagera el tamaño de las desviaciones (β < 1), y lo que gana con el orden lo pierde con la exageración. La calidad previa sola es **peor que nada** (−16 %; −24 % en las generales): leída como desviación con −τ·Φ⁻¹(*Q*), la rúbrica predice mal. Una explicación probable, no medida: la escala de *Q* tiene su punto neutro en 0,5 y la media de los pares es 0,42, así que el prior predice en promedio desviaciones positivas, y grandes para las casas de calidad baja (con *Q* = 0,05, +0,016, más que cualquier desviación prevista por el rating, que no pasa de ±0,013).
- **El promedio no mejora con el rating.** En 71 elecciones, el promedio simple de la última campaña con el rating yerra 0,011 puntos más que sin él, un 0,5 %, con un intervalo de −0,020 a +0,048 que incluye el 0, y 37 elecciones mejoran frente a 34 que empeoran. La diferencia media queda dentro del ±2 % en las tres poblaciones. Es lo esperable con un orden débil: aunque los pesos por rating difieren bastante (en 2023, 0,57 el CIS y 2,36 GAD3), el error de cada promedio lo domina lo que fallan todas las casas a la vez, y el orden del rating explica una parte pequeña de lo que se separan.

En una frase: el rating discrimina un poco entre casas, sobre todo en las autonómicas, con una contracción que se queda corta; no sirve para predecir el tamaño de la desviación de una casa y no mejora el promedio. Defiende el ranking como orden aproximado, no como predicción.

## Decisión (propuesta)

La fase C2 (afinar parámetros del rating con la verosimilitud fuera de muestra de la prueba 2, con cientos de pares en vez de cinco generales) es una decisión de Luis. Lo que las cifras permiten:

- **La pendiente dice en qué dirección**: β = 0,44 con el intervalo por debajo de 1 pide una contracción más fuerte, es decir, un τ menor (y con él k = (σ/τ)² mayor). Coincide con lo que ya dice la tarjeta «Qué límites tiene el rating» (τ = 0,01 parece ancha frente a la dispersión real de 0,0034-0,006). Es el único parámetro sobre el que hay una cifra que apunte.
- **El acierto negativo de la calidad previa cuestiona el peso de la rúbrica** (k·*Q* en `R = (W·R_aj + k·Q)/(W + k)`, unas 2 elecciones de evidencia) o su traducción a desviación. Pero el Spearman de las casas con poca evidencia (−0,18), en cuyo rating el prior pesa más de la mitad, dice que la rúbrica ordena algo: el problema es de escala, no de orden.
- **Nada dice que el promedio fuera a mejorar.** El rating mueve el promedio simple un 0,5 %, dentro del ruido. Un C2 cambiaría el orden y las notas del ranking, no la predicción. La mezcla 0,7/0,3, el 0,5 autonómico y el 0,9 por año no tienen ninguna cifra aquí: afinarlos sería una búsqueda a ciegas que perseguiría ruido.

Recomendación, sin decidir:

1. **Nada antes del 29-N.** Cambiar el orden del ranking en campaña es un anti-objetivo de la spec, y las cifras no dan una mejora de la predicción que lo justifique.
2. **Tras el 29-N**, con la sexta general en las cifras (`compute '{"what":["compute","ratings"],"scopes":"all"}'` y `publish analysis` de `es` con `rating_check`), un C2 **acotado** a τ (o k) y, como mucho, al peso de la calidad previa, con este juez: el ECM fuera de muestra de `d_obs` predicho con −τ·Φ⁻¹(*R*) sobre todos los pares (o su verosimilitud normal), bootstrap por casa, y como condiciones que el intervalo de β incluya el 1 y que el acierto relativo del rating no sea negativo. Antes de afinar τ con β hay que medir cuánto de β < 1 es la atenuación por la contracción de `d_obs` sondeo a sondeo (Salvedades): por ejemplo β por tramos del número de sondeos de la elección (10 no separa nada; 30 o más sí separaría).
3. **El peso del rating en el promedio se queda como está** mientras la prueba 1a no diga otra cosa: si el backtest plano queda dentro del ±2 %, el rating es cosmético para la predicción, y quitarlo o dejarlo es una decisión de presentación, no de acierto.

## Salvedades

- **Pocas generales.** 13 elecciones con pares (1986-2023) y 238 pares; en la prueba 1a, 5 generales (28 casos). Todos los intervalos de las generales cruzan el 0, y una sola campaña con pocas casas (1989, 1993) mueve su promedio simple. Las cifras del conjunto las deciden las autonómicas.
- **Autonómicas.** 704 de los 942 pares y 58 de las 71 elecciones del promedio simple, desde 2009. Su rating incluye evidencia de todos los ámbitos con el peso autonómico (0,5), y dos autonómicas del mismo día son dos elecciones que no se ven entre sí. En la prueba 1a sólo entran las cinco de la parte B y las elecciones `featured` desde 2019.
- **`d_obs` contraída por sondeo y atenuación de β.** `d_obs` es la evidencia del rating, no la desviación cruda (que no se guarda): cada sondeo se contrae con N(0, τ) frente al error de la regresión local, más en las fechas con pocos sondeos. β compara en esa misma escala, la de `d_prev` y la de σ y τ, así que 1 sigue siendo la referencia; pero la contracción variable atenúa β hacia abajo en las elecciones poco sondeadas. La comprobación prevista (`slope_polled`, elecciones con 10 o más sondeos) no separa nada, porque todas los tienen; un corte más alto está en la recomendación.
- **El consenso de cada fecha incluye a la propia casa.** La línea del error medio frente a la que se mide cada sondeo incluye los sondeos de la casa: en las fechas con pocos sondeos su desviación se acerca a 0 por partida doble (por la línea y por la contracción). Afecta igual a `d_obs` y a la evidencia del rating.
- **El diferencial mide error absoluto, no sesgo con signo.** `bias = 0,7·bias_blocks + 0,3·bias_within`: una casa que yerra mucho en cualquier dirección tiene desviación positiva. La prueba no dice nada del sesgo hacia un bloque.
- **La calidad media del respaldo.** Para las casas sin fila de rating, la prueba 1b usa `log1p(q) / log1p(media de q)` con la media de las **casas de esa elección**, mientras que el `Forecaster` usa la media de **todo el catálogo** (`self.pollsters.quality.mean()`). La diferencia sólo afecta a las casas sin rating, que en las elecciones valoradas son pocas, pero el promedio simple no es exactamente el peso del modelo.
- **Intervalos.** El bootstrap de la prueba 2 remuestrea casas, no elecciones: los pares de una misma elección comparten el error común y el consenso, y esa dependencia no entra en el intervalo, que puede quedarse estrecho. El de la prueba 1b remuestrea elecciones.
- **`slope_polled` idéntico a `slope`.** Ver arriba: con los datos de hoy es redundante; se conserva en la parte por si una elección futura con pares tuviera menos de 10 sondeos.
- **Referencia emparejada.** La prueba 1a compara `results_flat` con `results_rated`, corridas el mismo día con el mismo código y la misma base. `backtest/results` (45aaa8b, 01-11-2026) es otro código y otra base (sondeos, ratings, Compromís): se cita sólo como contexto, nunca como par.

## Reproducción

```bash
# Cifras de 1b y 2 sin escribir en el sitio real (site-dry/v1/analysis/es/rating-check.json)
python job.py publish '{"what":["analysis"],"scopes":["es"],"rating_check":true,"dry_run":true}'
# Publicación real (Luis), tras un compute de ratings
python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
python job.py publish '{"what":["analysis"],"scopes":["es"],"rating_check":true}'
```

En Python: `Computer(scope='es').build_series().rating_check_data('all')` devuelve `pairs` y `polls`; `rating_check.rating_check_summary(pairs, polls, comp.bias_dev_tau, comp.bias_dev_sigma, {})` el `data` de la parte, y `rating_check.simple_average_errors(polls)` la tabla por elección. La prueba 1a, con los comandos de su sección.
