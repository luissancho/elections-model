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

## ¿Mejora el promedio? (modelo vivo, prueba 1a, 04-11-2026)

Las cuatro corridas del 04-11-2026 en el portátil, en dos carpetas hermanas ignoradas por git, con la misma base (`es`: 3.753 sondeos hasta el 01-11-2026), `n_sim` 1.000 y semilla 42:

```bash
python backtest/run_backtest.py --scope es --out backtest/results_rated
python backtest/run_backtest.py --scope es --flat-ratings --out backtest/results_flat
python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --out backtest/results_rated
python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --flat-ratings --out backtest/results_flat
```

Comprobaciones: `run.json` con `flat_ratings: true` en las seis carpetas de `results_flat` y `false` en las seis de `results_rated`; 28 casos en `es` (5 generales; 2016 y noviembre de 2019 sin el horizonte de 180 días) y 66 en los autonómicos (18 de `es-md`, 12 de cada uno de los otros cuatro; 11 elecciones); ningún `OperationalError` en el log. Los `run.json` llevan los commits 64b6233, 6b97648 y 6e9d2b9 según la hora de cada corrida: entre ellos sólo cambian documentos, la web y sus tests, no el modelo. La corrida de `es` con rating se rehízo entera al final tras un corte de red. Las líneas base `mae_last_poll`, `mae_mean_4w` y `mae_prev_result` salen idénticas en las dos carpetas, caso a caso (no dependen del rating).

Las métricas son las del modo Hoy (`brier_vs`, `crps_seats`, `cov_shares95`); las del modo Elección dan la misma lectura: en las generales, Brier_h −7,4 % (11 / 11; +13,1 % sin 2023), cobertura_h igual y CRPS_h +0,8 %; en las autonómicas, Brier_h +2,5 % (20 / 19), cobertura_h −0,8 puntos porcentuales (1 / 3) y CRPS_h +0,6 %.

Emparejado por (ámbito, `event_date`, `horizon`), la diferencia es plano − con rating (positiva: quitar el rating empeora), media sobre los casos con la métrica; «mejor / peor» cuenta los casos en que el plano gana o pierde (en la cobertura, gana si cubre más); los empates aparte.

### Por población

| Población | Métrica | Casos | Con rating | Plano | Diferencia | Plano mejor / peor (empates) |
|---|---|---|---|---|---|---|
| Generales (5 elecciones) | MAE de voto (puntos) | 28 | 2,3737 | 2,3926 | +0,019 (+0,8 %) | 13 / 15 |
| | CRPS de escaños | 22 | 7,2361 | 7,3061 | +0,070 escaños (+1,0 %) | 10 / 12 |
| | Brier de mayorías | 22 | 0,0775 | 0,0706 | −0,0069 (−8,9 %) | 10 / 12 |
| | Cobertura del 95 % | 28 | 0,8560 | 0,8631 | +0,7 pp | 1 / 0 (27) |
| Autonómicas (11 elecciones) | MAE de voto (puntos) | 66 | 2,5102 | 2,5208 | +0,011 (+0,4 %) | 32 / 33 (1) |
| | CRPS de escaños | 66 | 2,6430 | 2,6612 | +0,018 escaños (+0,7 %) | 30 / 33 (3) |
| | Brier de mayorías | 66 | 0,0209 | 0,0215 | +0,0007 (+3,3 %) | 16 / 20 (30) |
| | Cobertura del 95 % | 66 | 0,8515 | 0,8470 | −0,45 pp | 3 / 4 (59) |
| Todas (94 casos) | MAE de voto (puntos) | 94 | 2,4695 | 2,4826 | +0,013 (+0,5 %) | 45 / 48 (1) |
| | CRPS de escaños | 88 | 3,7913 | 3,8225 | +0,031 escaños (+0,8 %) | 40 / 45 (3) |
| | Brier de mayorías | 88 | 0,0350 | 0,0338 | −0,0012 (−3,4 %) | 26 / 32 (30) |
| | Cobertura del 95 % | 94 | 0,8528 | 0,8518 | −0,1 pp | 4 / 4 (86) |

**El Brier de las generales lo decide 2023.** El plano gana en menos casos de los que pierde (10 / 12) y aun así su media mejora un 8,9 %: los seis casos de 2023 son los seis mayores a su favor (de −0,028 a −0,052 cada uno; Brier medio de la elección, 0,187 → 0,146), y de los otros 16 pierde 12, con el mayor en noviembre de 2019 a 90 días (+0,044). Sin 2023 (22 casos, 16 con Brier), el plano empeora en voto, escaños y mayorías (la cobertura sube 0,9 puntos porcentuales): MAE de voto 2,370 → 2,407 (+1,6 %, 8 / 14), CRPS 7,42 → 7,65 (+3,1 %, 4 / 12) y Brier 0,036 → 0,042 (+16,4 %, 4 / 12). En 2023 el plano también acierta más en voto (2,389 → 2,340) y en escaños (CRPS 6,75 → 6,39). Por elección, el MAE de voto con rating / plano: 2015 2,496 / 2,447; 2016 2,482 / 2,584; abril de 2019 2,197 / 2,243; noviembre de 2019 2,312 / 2,378; 2023 2,389 / 2,340. La de 2015 no tiene escaños válidos ni Brier.

**El Brier de las autonómicas es pequeño y lo mueven pocos casos.** 30 de los 66 casos empatan (probabilidades de mayoría de 0 o 1 que aciertan con los dos pesos: casi todos los de `es-ct` y `es-an`); la media sube de 0,0209 a 0,0215, con las mayores diferencias en `es-md` 2021 a 30 días (+0,027) y 2019 (−0,041 a 14 días, +0,020 a 6), Galicia 2024 y Valencia 2023. La cobertura baja 0,45 puntos porcentuales con sólo 7 casos que cambian (3 / 4), el mayor en Galicia 2020 (0,750 → 0,667 de media en la elección).

### Por ámbito (autonómicos)

| Ámbito | Casos | MAE de voto (con / plano, %) | CRPS de escaños (con / plano, %) | Brier (con / plano) | Cobertura del 95 % (con / plano) |
|---|---|---|---|---|---|
| `es-md` | 18 | 2,889 / 2,927 (+1,3 %, 6 / 12) | 3,671 / 3,746 (+2,0 %, 4 / 14) | 0,0392 / 0,0380 | 0,815 / 0,806 |
| `es-an` | 12 | 1,754 / 1,739 (−0,8 %, 6 / 6) | 1,964 / 1,976 (+0,6 %, 5 / 7) | 0,0000 / 0,0000 | 0,911 / 0,942 |
| `es-ct` | 12 | 2,162 / 2,165 (+0,1 %, 6 / 6) | 2,632 / 2,603 (−1,1 %, 8 / 4) | 0 / 0 | 0,833 / 0,833 |
| `es-vc` | 12 | 1,947 / 1,926 (−1,0 %, 10 / 2) | 1,956 / 1,937 (−1,0 %, 9 / 3) | 0,0333 / 0,0359 | 0,958 / 0,958 |
| `es-ga` | 12 | 3,610 / 3,644 (+0,9 %, 4 / 7) | 2,477 / 2,503 (+1,0 %, 4 / 5) | 0,0227 / 0,0256 | 0,758 / 0,717 |

Ningún ámbito pasa del ±2 % en el MAE de voto; en el CRPS, `es-md` llega al +2,0 % (4 / 14).

### Por horizonte (MAE de voto, en puntos)

| Días | Generales: con rating / plano | % | Plano mejor / peor | Autonómicas: con rating / plano | % | Plano mejor / peor |
|---|---|---|---|---|---|---|
| 6 | 1,699 / 1,702 | +0,2 % | 2 / 3 | 1,559 / 1,558 | −0,0 % | 4 / 7 |
| 14 | 1,976 / 1,929 | −2,4 % | 4 / 1 | 1,985 / 2,008 | +1,2 % | 5 / 6 |
| 30 | 2,168 / 2,198 | +1,4 % | 2 / 3 | 2,309 / 2,372 | +2,7 % | 2 / 9 |
| 60 | 2,803 / 2,911 | +3,8 % | 1 / 4 | 2,943 / 2,928 | −0,5 % | 8 / 3 |
| 90 | 2,838 / 2,859 | +0,7 % | 2 / 3 | 2,878 / 2,913 | +1,2 % | 6 / 5 |
| 180 | 3,015 / 2,999 | −0,6 % | 2 / 1 | 3,389 / 3,346 | −1,2 % | 7 / 3 |

Con cinco casos por horizonte en las generales (tres a 180 días) y once en las autonómicas, ningún horizonte separa nada: el signo no es el mismo en todos los horizontes. El CRPS de las generales va igual (+1,1 %, −2,4 %, +1,2 %, +3,6 %, +1,5 %, −1,4 %) y su Brier mejora con el plano en todos los horizontes salvo 90 días, por 2023.

### Frente a los promedios sin rating

| Población | Modelo con rating | Modelo plano | Último sondeo | Media de 4 semanas | Casos en que gana al último sondeo (con / plano) | A la media de 4 semanas (con / plano) |
|---|---|---|---|---|---|---|
| Generales | 2,374 | 2,393 | 2,452 | 2,382 | 14 / 15 de 28 | 17 / 15 de 28 |
| Autonómicas | 2,510 | 2,521 | 2,717 | 2,634 | 43 / 44 de 66 | 42 / 42 de 66 |

Las dos líneas base ya son promedios sin rating. En las autonómicas el modelo les gana con y sin rating, por 0,1-0,2 puntos. En las generales el modelo con rating queda 0,009 puntos por debajo de la media de 4 semanas y el plano 0,010 por encima: la ventaja del modelo sobre esa media en las generales es del mismo tamaño que lo que mueve el rating, una centésima de punto, y con 28 casos ninguna de las dos se distingue de 0 (17 / 11 y 15 / 13 casos).

### Juicio (criterio 17)

Leído como el cambio «quitar el rating»: en el MAE de voto y el CRPS de escaños, el plano queda dentro del ±2 % en las dos poblaciones (+0,8 % y +1,0 % en las generales; +0,4 % y +0,7 % en las autonómicas), con los casos mejor / peor casi a la par. El Brier se mueve en sentidos opuestos: −8,9 % en las generales (por 2023; sin ella, +16 %) y +3,3 % en las autonómicas, fuera del ±2 % pero sobre un error de 0,02 con la mitad de los casos empatados. La cobertura del 95 % sube 0,7 puntos porcentuales en las generales y baja 0,45 en las autonómicas. Estrictamente, el plano no pasa el criterio 17 en las autonómicas (Brier y cobertura); con este tamaño de las diferencias, ni lo pasa ni lo suspende por algo que se distinga del ruido.

La lectura declarada de antemano se cumple: **plano ≈ con rating dentro del ±2 % en voto y escaños, así que el rating es cosmético para la predicción del modelo vivo.** El signo, si acaso, favorece al rating por muy poco (+0,5 % de error de voto sin él en los 94 casos), salvo en 2023, la general en la que también el promedio simple con rating falló más que sin él (+0,125 puntos en la prueba 1b). Es del mismo tamaño que en la prueba 1b (+0,5 % en el promedio simple, con rating peor; aquí, sin rating peor): en los dos casos, el rating mueve el error del promedio un 0,5 %, en un sentido o en otro.

Las tablas salen de un script de scratch (no commiteado) que lee `metrics.csv` de las dos carpetas y empareja los casos; en `es`, el `by_horizon.csv` de cada carpeta da las mismas medias por horizonte (en los autonómicos, `by_horizon.csv` es por ámbito).

## ¿Y con otra banda? (fase C2a, 06-11-2026)

La prueba 1a quitó el rating, que es la banda de anchura 0 entre la peor y la mejor encuestadora, y el error de voto del modelo vivo subió un 0,5 %: una señal pequeña a favor del rating. La fase C2a pregunta si **ensanchar la banda** (o estrecharla) acierta más. Spec: `docs/superpowers/specs/2026-11-05-rating-exponent-design.md`.

Cambios:
- `rating_exponent` (γ, número ≥ 0, 1 por defecto) de `Forecaster` y `Simulator`: eleva a γ el `weight_rating` de cada sondeo, después del relleno por calidad y de los `weight_overrides`; los resultados oficiales no se tocan. γ = 1 es el modelo tal cual y γ = 0, el rating plano; un valor negativo o no finito da `ValueError` (`mtpy/lib/forecaster.py`, c0370a4).
- Backtest: `rating_exponent` en `run_case`, `run_backtest`, `backtest_scope`, el pipeline y el job `backtest`, y `--rating-exponent` en `backtest/run_backtest.py`; queda en `run.json` y en la columna `rating_exponent` de `meta.csv` (81306fa).
- `backtest/compare_results.py A B`: empareja los `metrics.csv` de dos carpetas por (ámbito, `event_date`, `horizon`) y escribe la tabla por población en markdown (`pair_results`, `compare_paired` y `compare_markdown` en `mtpy/lib/backtest.py`). Sustituye al script de scratch de la prueba 1a (0c5276a).
- `rating_check.simple_average_curve(polls, exponents)`: la prueba 1b con el peso por rating elevado a cada γ, emparejada por elección frente a γ = 1 (a6aaf98).

Tests: `tests/test_forecaster_unit.py`, `tests/test_simulator_unit.py`, `tests/test_backtest_unit.py`, `tests/test_jobs_backtest.py`, `tests/test_pipelines_backtest.py`, `tests/test_launchers.py` y `tests/test_rating_check_unit.py`.

### La rejilla y la banda

γ ∈ {0, 0,5, 1, 1,5, 2, 3}, fijada antes de mirar y simétrica alrededor de 1: si lo mejor estuviera por debajo de 1, una rejilla sólo hacia arriba no lo vería. γ no cambia la forma del mapa del rating al peso (sigue casi lineal en *R*); sólo estira o encoge la banda. En el ciclo nacional en curso (05-11-2026, 24 casas, 337 sondeos), a γ = 1 el peso por rating va de 0,58 (CIS) a 2,25 (GAD3), 3,9 veces:

| γ | 0 | 0,5 | 1 | 1,5 | 2 | 3 |
|---|---|---|---|---|---|---|
| Peso de la mejor casa frente a la peor | 1× (todos los sondeos pesan 1) | 2× | 3,9× | 7,7× | 15× | 59× |

Los jueces, en este orden (spec, decisión 4): la prueba 1b como cribado, en minutos, y el backtest emparejado del modelo vivo, que es el que decide. El arnés de sondeos futuros (método 14) no juzga: con una banda más ancha el promedio se acerca a las casas mejor valoradas, y predecir sus sondeos futuros lo premiaría por construcción.

La regla para cambiar el valor por defecto, fijada antes de las corridas (spec, decisión 6): un γ sólo sustituye a 1 si mejora el MAE de voto y el CRPS de escaños frente a γ = 1 más de un 2 % en las dos poblaciones, con más casos mejor que peor, sin empeorar el Brier más de un 2 % ni bajar la cobertura. Lectura declarada de antemano: una curva plana dentro del ±2 % confirma que la anchura de la banda es cosmética.

### El promedio simple por γ (prueba 1b, cribado)

`simple_average_curve` sobre el frame `polls` de `Computer('es').build_series().rating_check_data('all')`, guardado el 05-11-2026 (4.457 filas): 71 elecciones (13 generales y 58 autonómicas), los sondeos útiles de los 30 días anteriores de cada una, 2.000 réplicas por bootstrap de elecciones, semilla 0; 9 s. La diferencia es el MAE con γ menos el MAE con γ = 1, emparejado por elección y promediado (negativa: γ acierta más); «mejor / peor» cuenta las elecciones en que γ gana o pierde frente a γ = 1.

| γ | Población | Elecciones | MAE (puntos) | Diferencia frente a γ = 1 (IC 95 %) | Mejor / peor que γ = 1 |
|---|---|---|---|---|---|
| 0 | Todas | 71 | 1,9292 | −0,0106 (−0,0478 a +0,0197) | 34 / 37 |
| | Generales | 13 | 1,6572 | −0,0041 (−0,0664 a +0,0580) | 6 / 7 |
| | Autonómicas | 58 | 1,9902 | −0,0121 (−0,0601 a +0,0197) | 28 / 30 |
| 0,5 | Todas | 71 | 1,9369 | −0,0029 (−0,0175 a +0,0085) | 31 / 40 |
| | Generales | 13 | 1,6617 | +0,0004 (−0,0230 a +0,0296) | 6 / 7 |
| | Autonómicas | 58 | 1,9986 | −0,0036 (−0,0212 a +0,0098) | 25 / 33 |
| 1 | Todas | 71 | 1,9398 | referencia | |
| | Generales | 13 | 1,6613 | referencia | |
| | Autonómicas | 58 | 2,0023 | referencia | |
| 1,5 | Todas | 71 | 1,9401 | +0,0002 (−0,0077 a +0,0090) | 38 / 33 |
| | Generales | 13 | 1,6623 | +0,0010 (−0,0137 a +0,0128) | 5 / 8 |
| | Autonómicas | 58 | 2,0023 | +0,0001 (−0,0099 a +0,0108) | 33 / 25 |
| 2 | Todas | 71 | 1,9390 | −0,0008 (−0,0159 a +0,0142) | 40 / 31 |
| | Generales | 13 | 1,6633 | +0,0020 (−0,0203 a +0,0192) | 5 / 8 |
| | Autonómicas | 58 | 2,0008 | −0,0015 (−0,0195 a +0,0181) | 35 / 23 |
| 3 | Todas | 71 | 1,9340 | −0,0058 (−0,0299 a +0,0189) | 38 / 33 |
| | Generales | 13 | 1,6615 | +0,0002 (−0,0322 a +0,0266) | 5 / 8 |
| | Autonómicas | 58 | 1,9951 | −0,0071 (−0,0356 a +0,0235) | 33 / 25 |

Las filas de γ = 0 son la prueba 1b de la fase C con el signo al revés (allí, con rating − sin rating: +0,011 y 37 / 34).

**La curva es plana.** En el conjunto, el MAE va de 1,929 a γ = 0 a 1,940 a γ = 1 y 1,934 a γ = 3; ninguna diferencia frente a γ = 1 pasa de 0,011 puntos (en las autonómicas, 0,012 a γ = 0), un 0,5 % del error, y todos los intervalos cruzan el 0. Los recuentos tampoco separan: las bandas anchas ganan algo más elecciones de las que pierden en el conjunto (de 38 / 33 a 40 / 31), la estrecha (γ = 0,5) pierde más de las que gana (31 / 40), y en las generales todos los γ distintos de 1 pierden más de las que ganan (6 / 7 o 5 / 8), siempre por milésimas de punto de media. En el promedio simple, la anchura de la banda es cosmética.

### El backtest emparejado por γ (el que decide)

Las corridas, encadenadas en segundo plano en el portátil del 05-11-2026 a las 14:15 al 06-11-2026 a las 01:50, con γ = 1 primero para comprobar la base cuanto antes; por γ, `es` (28 casos, 5 generales) y luego los cinco autonómicos de la prueba 1a (66 casos: 18 de `es-md` y 12 de cada uno de los otros cuatro), en `backtest/results_exp{γ}` (ignoradas por git), con los parámetros por defecto (`n_sim` 1.000, semilla 42). Tiempos (`es` + los cinco): γ = 1, 56 min + 1 h 21 (2 h 17); γ = 0,5, 50 min + 1 h 19; γ = 1,5, 1 h 15 + 1 h 45; γ = 2, 50 min + 1 h 07; γ = 3, 59 min + 1 h 10.

**La base no cambió.** γ = 1 se rehízo como referencia (`backtest/results_exp1`) y reproduce `results_rated` de la prueba 1a al diezmilésimo en las cuatro métricas, con los 94 casos empatados (`compare_results.py backtest/results_rated backtest/results_exp1`). Por eso `results_flat` vale como γ = 0 y no se rehízo (spec, decisión 4). Comprobaciones: `run.json` con `rating_exponent` = γ en las seis carpetas de cada γ; 28 / 18 / 12 / 12 / 12 / 12 casos por ámbito en todos los γ salvo 1,5 (ver Salvedades); ningún `OperationalError` en el log. Los `run.json` llevan los commits a6aaf98, ea54e9b, eaba7c9, a3f905b y 4049057 según la hora de cada corrida: entre ellos sólo cambian la web, `publish`, `update` y los estudios, no el modelo ni el backtest. Las cinco rejillas acabaron (01:49) antes del recálculo de los ratings guardados del 06-11-2026 a las 01:50: todas corrieron sobre la misma base que `results_rated` y `results_flat`.

Emparejado por (ámbito, `event_date`, `horizon`), la diferencia es γ − (γ = 1) en % de la media de γ = 1 (positiva: γ yerra más; en la cobertura, en puntos porcentuales, positiva: cubre más); «mejor / peor» cuenta los casos en que γ gana o pierde (en la cobertura, gana si cubre más), con los empates entre paréntesis. «Casos» son los de voto y cobertura; el CRPS y el Brier sólo cuentan los casos con escaños válidos (22 de los 28 de las generales: 2015 no los tiene). Métricas del modo Hoy, como en la prueba 1a.

**Generales** (5 elecciones, 28 casos)

| γ | Casos | MAE de voto | CRPS de escaños | Brier de mayorías | Cobertura del 95 % |
|---|---|---|---|---|---|
| 0 | 28 | +0,8 % (13 / 15) | +1,0 % (10 / 12) | −8,9 % (10 / 12) | +0,7 pp (1 / 0, 27) |
| 0,5 | 28 | +0,2 % (12 / 16) | +0,2 % (12 / 10) | −4,5 % (11 / 11) | +0,0 pp (0 / 0, 28) |
| 1,5 | 28 | +0,8 % (15 / 13) | +0,8 % (10 / 12) | +5,9 % (12 / 10) | −0,9 pp (0 / 1, 27) |
| 2 | 28 | +1,3 % (14 / 14) | +1,1 % (10 / 12) | +8,3 % (8 / 13, 1) | −0,2 pp (1 / 1, 26) |
| 3 | 28 | +1,8 % (14 / 14) | +1,4 % (12 / 10) | +12,5 % (7 / 15) | −0,9 pp (1 / 2, 25) |

**Autonómicas** (11 elecciones, 66 casos)

| γ | Casos | MAE de voto | CRPS de escaños | Brier de mayorías | Cobertura del 95 % |
|---|---|---|---|---|---|
| 0 | 66 | +0,4 % (32 / 33, 1) | +0,7 % (30 / 33, 3) | +3,3 % (16 / 20, 30) | −0,5 pp (3 / 4, 59) |
| 0,5 | 66 | +0,3 % (26 / 40) | +0,4 % (28 / 35, 3) | +1,2 % (16 / 19, 31) | −0,8 pp (2 / 5, 59) |
| 1,5 | 64 | −0,1 % (34 / 30) | −0,3 % (31 / 30, 3) | −0,5 % (15 / 20, 29) | −0,6 pp (0 / 2, 62) |
| 2 | 66 | −0,3 % (32 / 34) | −0,6 % (32 / 31, 3) | −0,6 % (16 / 24, 26) | −0,4 pp (1 / 2, 63) |
| 3 | 66 | −0,8 % (33 / 32, 1) | +2,4 % (28 / 35, 3) | −7,1 % (21 / 20, 25) | −1,2 pp (1 / 5, 60) |

**Todas** (94 casos)

| γ | Casos | MAE de voto | CRPS de escaños | Brier de mayorías | Cobertura del 95 % |
|---|---|---|---|---|---|
| 0 | 94 | +0,5 % (45 / 48, 1) | +0,8 % (40 / 45, 3) | −3,4 % (26 / 32, 30) | −0,1 pp (4 / 4, 86) |
| 0,5 | 94 | +0,3 % (38 / 56) | +0,3 % (40 / 45, 3) | −2,0 % (27 / 30, 31) | −0,6 pp (2 / 5, 87) |
| 1,5 | 92 | +0,1 % (49 / 43) | +0,3 % (41 / 42, 3) | +3,0 % (27 / 30, 29) | −0,7 pp (0 / 3, 89) |
| 2 | 94 | +0,1 % (46 / 48) | +0,2 % (42 / 43, 3) | +4,3 % (24 / 37, 27) | −0,4 pp (2 / 3, 89) |
| 3 | 94 | −0,1 % (47 / 46, 1) | +1,9 % (40 / 45, 3) | +3,8 % (28 / 35, 25) | −1,1 pp (2 / 7, 85) |

Las medias de γ = 1, con las que se calculan los %: en las generales, MAE de voto 2,3737, CRPS 7,2361, Brier 0,0775 y cobertura 0,8560; en las autonómicas, 2,5102, 2,6430, 0,0209 y 0,8515 (las de γ = 1,5, sobre 64 casos: 2,5404, 2,6728, 0,0215 y 0,8469). Las tablas completas de `compare_results.py`, con las medias de cada γ, al final de la sección.

### Juicio (regla 6 de la spec)

**Ningún γ pasa la regla.** Ninguno mejora el MAE de voto y el CRPS de escaños más de un 2 % en las dos poblaciones con más casos mejor que peor; en las generales, ningún γ mejora siquiera el voto o el CRPS de γ = 1:

- **Las bandas anchas empeoran las generales, más cuanto más anchas**: a γ = 1,5, 2 y 3, el voto sube un 0,8, un 1,3 y un 1,8 %, el CRPS un 0,8, un 1,1 y un 1,4 %, y el Brier un 5,9, un 8,3 y un 12,5 %. En las autonómicas el voto se mueve dentro del ±1 % (−0,1, −0,3 y −0,8 %), y lo mejor que dan es −0,8 % en voto a γ = 3, con el CRPS un 2,4 % peor, o −0,6 % en el CRPS a γ = 2: muy lejos del 2 %.
- **La cobertura del 95 % baja en todos los γ > 1**, en las dos poblaciones: entre −0,4 y −1,1 puntos porcentuales sobre todos los casos.
- **La banda estrecha pierde en más casos de los que gana**: γ = 0,5 empeora el MAE de voto en 56 de los 94 casos (38 / 56; 26 / 40 en las autonómicas), con medias apenas peores (+0,3 %). El rating plano (γ = 0) es la prueba 1a: +0,5 % de error de voto, 45 / 48.
- **El Brier de las generales sigue a la anchura en todo el tramo** (−8,9, −4,5, 0, +5,9, +8,3 y +12,5 % de γ = 0 a 3), pero sobre 22 casos y lo mueven pocas elecciones: en la prueba 1a, los seis casos de 2023 decidían el de γ = 0. No es una señal para estrechar la banda, porque el voto y los escaños de las generales también empeoran hacia γ = 0.

La lectura declarada de antemano se cumple: **la curva es plana dentro del ±2 % en el voto, en todos los γ y las dos poblaciones (lo más, +1,8 % en las generales a γ = 3), y en los escaños salvo el CRPS de las autonómicas a γ = 3 (+2,4 %, a peor), así que la anchura de la banda es cosmética para la predicción.** Los recuentos añaden lo que las medias no dicen: γ = 0,5 pierde 56 de los 94 casos y γ = 1,5 gana 49 de 92 (con una media un 0,1 % peor). El rating ayuda un poco, y alrededor de 1 una banda acierta lo mismo que otra: la de hoy está en el tramo plano de la curva.

Decisión: el valor por defecto se queda en γ = 1 y el modelo vivo no cambia (no hay commit de ese tipo). `rating_exponent` queda en el código como palanca del backtest.

### Salvedades

- **Cinco generales.** 28 casos, 22 con escaños y Brier. El Brier de las generales lo domina 2023 (prueba 1a), así que su pendiente frente a γ dice poco; el voto y el CRPS de las generales se mueven menos de un 2 % en todos los γ.
- **La referencia γ = 1 se rehízo** con el código de hoy y la misma base que los demás puntos, y se comprobó que reproduce `results_rated` caso a caso: las cinco carpetas `results_exp{γ}` son comparables entre sí y con `results_flat`.
- **γ = 1,5 sobre 92 casos.** `es-an` 2022-06-19 a 90 y 180 días se perdieron por un corte de red con la base: el arnés los registró como «skipped», sin la palabra `OperationalError`. No se pudieron repetir sobre la misma base, porque los ratings guardados se recalcularon el 06-11-2026 a las 01:50, después de la cadena entera. El emparejamiento por caso los deja fuera también de γ = 1 en esa comparación (64 autonómicas; MAE de voto de referencia 2,4897 en 92 casos, frente a 2,4695 en 94), sin tocar el resto. Lección para las corridas largas: contar los casos de cada ámbito frente a la referencia, no sólo buscar `OperationalError` en el log.
- **La prueba 1b sólo pesa la última campaña de cada elección**: los sondeos de los 30 días anteriores, sin horizontes, sin efectos de casa y sin el resto del modelo. Tiene la potencia de 71 elecciones, pero mide otra cosa que el backtest; es el cribado, no el juez.
- **El arnés de sondeos futuros no juzgó** (spec, decisión 4): con una banda más ancha el promedio se acerca a las casas mejor valoradas, y predecir sus sondeos futuros lo premiaría por construcción. Queda como diagnóstico posterior.
- **Sin intervalos en el backtest.** Como en la prueba 1a, con 5 generales y 11 autonómicas los recuentos mejor / peor son la medida de su ruido.

### Reproducción

```bash
# La rejilla, encadenada en segundo plano (≈ 2 h por γ en el portátil); γ = 0 es results_flat de la prueba 1a
nohup sh -c 'for g in 1 0.5 1.5 2 3; do python backtest/run_backtest.py --scope es --rating-exponent $g --out backtest/results_exp$g && python backtest/run_backtest.py --scopes es-md es-an es-ct es-vc es-ga --rating-exponent $g --out backtest/results_exp$g || exit 1; done' > backtests.log 2>&1 &
# La base no cambió: diferencia 0 y 94 casos empatados
python backtest/compare_results.py backtest/results_rated backtest/results_exp1
# Cada punto de la curva frente a γ = 1
python backtest/compare_results.py backtest/results_exp1 backtest/results_flat --label-a "γ = 1" --label-b "γ = 0"
python backtest/compare_results.py backtest/results_exp1 backtest/results_exp0.5 --label-a "γ = 1" --label-b "γ = 0,5"
python backtest/compare_results.py backtest/results_exp1 backtest/results_exp1.5 --label-a "γ = 1" --label-b "γ = 1,5"
python backtest/compare_results.py backtest/results_exp1 backtest/results_exp2 --label-a "γ = 1" --label-b "γ = 2"
python backtest/compare_results.py backtest/results_exp1 backtest/results_exp3 --label-a "γ = 1" --label-b "γ = 3"
```

En Python, la prueba 1b: `data = Computer(scope='es').build_series().rating_check_data('all')` (≈ 4 min contra la base, sólo lectura) y `rating_check.simple_average_curve(data['polls'], [0, 0.5, 1, 1.5, 2, 3])` (9 s).

### Tablas completas (`compare_results.py`)

Tal como las escribe el script (punto decimal; «mejor / peor» desde γ).

**γ = 0** (`results_flat`)

| Población | Métrica | Casos | γ = 1 | γ = 0 | Diferencia (%) | Mejor / peor (empates) |
|---|---|---:|---:|---:|---:|---:|
| Generales | MAE de voto (puntos) | 28 | 2.3737 | 2.3926 | +0.0188 (+0.8%) | 13 / 15 (0) |
| Generales | CRPS de escaños | 22 | 7.2361 | 7.3061 | +0.0699 (+1.0%) | 10 / 12 (0) |
| Generales | Brier de mayorías | 22 | 0.0775 | 0.0706 | -0.0069 (-8.9%) | 10 / 12 (0) |
| Generales | Cobertura del 95 % | 28 | 0.8560 | 0.8631 | +0.0071 (+0.8%) | 1 / 0 (27) |
| Autonómicas | MAE de voto (puntos) | 66 | 2.5102 | 2.5208 | +0.0106 (+0.4%) | 32 / 33 (1) |
| Autonómicas | CRPS de escaños | 66 | 2.6430 | 2.6612 | +0.0183 (+0.7%) | 30 / 33 (3) |
| Autonómicas | Brier de mayorías | 66 | 0.0209 | 0.0215 | +0.0007 (+3.3%) | 16 / 20 (30) |
| Autonómicas | Cobertura del 95 % | 66 | 0.8515 | 0.8470 | -0.0045 (-0.5%) | 3 / 4 (59) |
| Todas | MAE de voto (puntos) | 94 | 2.4695 | 2.4826 | +0.0130 (+0.5%) | 45 / 48 (1) |
| Todas | CRPS de escaños | 88 | 3.7913 | 3.8225 | +0.0312 (+0.8%) | 40 / 45 (3) |
| Todas | Brier de mayorías | 88 | 0.0350 | 0.0338 | -0.0012 (-3.4%) | 26 / 32 (30) |
| Todas | Cobertura del 95 % | 94 | 0.8528 | 0.8518 | -0.0011 (-0.1%) | 4 / 4 (86) |

**γ = 0,5** (`results_exp0.5`)

| Población | Métrica | Casos | γ = 1 | γ = 0,5 | Diferencia (%) | Mejor / peor (empates) |
|---|---|---:|---:|---:|---:|---:|
| Generales | MAE de voto (puntos) | 28 | 2.3737 | 2.3794 | +0.0057 (+0.2%) | 12 / 16 (0) |
| Generales | CRPS de escaños | 22 | 7.2361 | 7.2494 | +0.0133 (+0.2%) | 12 / 10 (0) |
| Generales | Brier de mayorías | 22 | 0.0775 | 0.0740 | -0.0035 (-4.5%) | 11 / 11 (0) |
| Generales | Cobertura del 95 % | 28 | 0.8560 | 0.8560 | +0.0000 (+0.0%) | 0 / 0 (28) |
| Autonómicas | MAE de voto (puntos) | 66 | 2.5102 | 2.5176 | +0.0074 (+0.3%) | 26 / 40 (0) |
| Autonómicas | CRPS de escaños | 66 | 2.6430 | 2.6543 | +0.0113 (+0.4%) | 28 / 35 (3) |
| Autonómicas | Brier de mayorías | 66 | 0.0209 | 0.0211 | +0.0002 (+1.2%) | 16 / 19 (31) |
| Autonómicas | Cobertura del 95 % | 66 | 0.8515 | 0.8432 | -0.0083 (-1.0%) | 2 / 5 (59) |
| Todas | MAE de voto (puntos) | 94 | 2.4695 | 2.4764 | +0.0069 (+0.3%) | 38 / 56 (0) |
| Todas | CRPS de escaños | 88 | 3.7913 | 3.8031 | +0.0118 (+0.3%) | 40 / 45 (3) |
| Todas | Brier de mayorías | 88 | 0.0350 | 0.0343 | -0.0007 (-2.0%) | 27 / 30 (31) |
| Todas | Cobertura del 95 % | 94 | 0.8528 | 0.8470 | -0.0059 (-0.7%) | 2 / 5 (87) |

**γ = 1,5** (`results_exp1.5`, 92 casos)

| Población | Métrica | Casos | γ = 1 | γ = 1,5 | Diferencia (%) | Mejor / peor (empates) |
|---|---|---:|---:|---:|---:|---:|
| Generales | MAE de voto (puntos) | 28 | 2.3737 | 2.3916 | +0.0179 (+0.8%) | 15 / 13 (0) |
| Generales | CRPS de escaños | 22 | 7.2361 | 7.2969 | +0.0607 (+0.8%) | 10 / 12 (0) |
| Generales | Brier de mayorías | 22 | 0.0775 | 0.0821 | +0.0046 (+5.9%) | 12 / 10 (0) |
| Generales | Cobertura del 95 % | 28 | 0.8560 | 0.8470 | -0.0089 (-1.0%) | 0 / 1 (27) |
| Autonómicas | MAE de voto (puntos) | 64 | 2.5404 | 2.5371 | -0.0033 (-0.1%) | 34 / 30 (0) |
| Autonómicas | CRPS de escaños | 64 | 2.6728 | 2.6649 | -0.0079 (-0.3%) | 31 / 30 (3) |
| Autonómicas | Brier de mayorías | 64 | 0.0215 | 0.0214 | -0.0001 (-0.5%) | 15 / 20 (29) |
| Autonómicas | Cobertura del 95 % | 64 | 0.8469 | 0.8410 | -0.0059 (-0.7%) | 0 / 2 (62) |
| Todas | MAE de voto (puntos) | 92 | 2.4897 | 2.4928 | +0.0032 (+0.1%) | 49 / 43 (0) |
| Todas | CRPS de escaños | 86 | 3.8402 | 3.8498 | +0.0097 (+0.3%) | 41 / 42 (3) |
| Todas | Brier de mayorías | 86 | 0.0358 | 0.0369 | +0.0011 (+3.0%) | 27 / 30 (29) |
| Todas | Cobertura del 95 % | 92 | 0.8496 | 0.8428 | -0.0068 (-0.8%) | 0 / 3 (89) |

**γ = 2** (`results_exp2`)

| Población | Métrica | Casos | γ = 1 | γ = 2 | Diferencia (%) | Mejor / peor (empates) |
|---|---|---:|---:|---:|---:|---:|
| Generales | MAE de voto (puntos) | 28 | 2.3737 | 2.4042 | +0.0305 (+1.3%) | 14 / 14 (0) |
| Generales | CRPS de escaños | 22 | 7.2361 | 7.3172 | +0.0811 (+1.1%) | 10 / 12 (0) |
| Generales | Brier de mayorías | 22 | 0.0775 | 0.0839 | +0.0064 (+8.3%) | 8 / 13 (1) |
| Generales | Cobertura del 95 % | 28 | 0.8560 | 0.8542 | -0.0018 (-0.2%) | 1 / 1 (26) |
| Autonómicas | MAE de voto (puntos) | 66 | 2.5102 | 2.5018 | -0.0084 (-0.3%) | 32 / 34 (0) |
| Autonómicas | CRPS de escaños | 66 | 2.6430 | 2.6277 | -0.0152 (-0.6%) | 32 / 31 (3) |
| Autonómicas | Brier de mayorías | 66 | 0.0209 | 0.0207 | -0.0001 (-0.6%) | 16 / 24 (26) |
| Autonómicas | Cobertura del 95 % | 66 | 0.8515 | 0.8472 | -0.0043 (-0.5%) | 1 / 2 (63) |
| Todas | MAE de voto (puntos) | 94 | 2.4695 | 2.4727 | +0.0032 (+0.1%) | 46 / 48 (0) |
| Todas | CRPS de escaños | 88 | 3.7913 | 3.8001 | +0.0088 (+0.2%) | 42 / 43 (3) |
| Todas | Brier de mayorías | 88 | 0.0350 | 0.0365 | +0.0015 (+4.3%) | 24 / 37 (27) |
| Todas | Cobertura del 95 % | 94 | 0.8528 | 0.8493 | -0.0035 (-0.4%) | 2 / 3 (89) |

**γ = 3** (`results_exp3`)

| Población | Métrica | Casos | γ = 1 | γ = 3 | Diferencia (%) | Mejor / peor (empates) |
|---|---|---:|---:|---:|---:|---:|
| Generales | MAE de voto (puntos) | 28 | 2.3737 | 2.4176 | +0.0439 (+1.8%) | 14 / 14 (0) |
| Generales | CRPS de escaños | 22 | 7.2361 | 7.3355 | +0.0994 (+1.4%) | 12 / 10 (0) |
| Generales | Brier de mayorías | 22 | 0.0775 | 0.0872 | +0.0097 (+12.5%) | 7 / 15 (0) |
| Generales | Cobertura del 95 % | 28 | 0.8560 | 0.8470 | -0.0089 (-1.0%) | 1 / 2 (25) |
| Autonómicas | MAE de voto (puntos) | 66 | 2.5102 | 2.4889 | -0.0213 (-0.8%) | 33 / 32 (1) |
| Autonómicas | CRPS de escaños | 66 | 2.6430 | 2.7069 | +0.0640 (+2.4%) | 28 / 35 (3) |
| Autonómicas | Brier de mayorías | 66 | 0.0209 | 0.0194 | -0.0015 (-7.1%) | 21 / 20 (25) |
| Autonómicas | Cobertura del 95 % | 66 | 0.8515 | 0.8391 | -0.0124 (-1.5%) | 1 / 5 (60) |
| Todas | MAE de voto (puntos) | 94 | 2.4695 | 2.4677 | -0.0019 (-0.1%) | 47 / 46 (1) |
| Todas | CRPS de escaños | 88 | 3.7913 | 3.8641 | +0.0728 (+1.9%) | 40 / 45 (3) |
| Todas | Brier de mayorías | 88 | 0.0350 | 0.0363 | +0.0013 (+3.8%) | 28 / 35 (25) |
| Todas | Cobertura del 95 % | 94 | 0.8528 | 0.8415 | -0.0113 (-1.3%) | 2 / 7 (85) |

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

`slope_polled` (β sólo con las elecciones de 10 o más sondeos) sale **idéntico** a β en las tres poblaciones (0,44, 0,06 y 0,55, con 832, 187 y 645 pares): todas las elecciones con pares tienen 10 o más sondeos, así que la comprobación no separa nada con estos datos y la tarjeta no la muestra (la nota lo dice). Sus intervalos sólo difieren por la semilla del bootstrap (−0,05 a 0,73 en el conjunto).

La recta ajustada, en la escala de la nube (×100): de (−1,29, −0,53) a (1,30, 0,61); la constante es prácticamente 0. Las desviaciones previstas van de −1,29 a 1,30 y las observadas de −4,71 a 4,84: lo previsto ya está muy contraído y aun así exagera.

Los ECM de las tres predicciones (log odds ratio al cuadrado) son del orden de 1e-4 y a los 4 decimales de la parte saldrían 0,0002-0,0003: la parte publica sólo el acierto relativo (`skill_prior`, `skill_rating` y `n`); `skill` sigue devolviendo los ECM para quien los quiera en Python.

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
- **Los cuartos lo dicen en puntos.** El cuarto con más rating falló 2,71 puntos por partido y el de menos 3,28: 0,57 puntos, un 17 % menos. Los dos cuartos centrales casi no se distinguen (2,85 y 2,83). La mayor parte del error la tienen también las mejores casas. No todo es error compartido: el error de cada par incluye el ruido de muestra de sus sondeos, que el promedio quita. La estimación de estos datos del suelo compartido (M6) es el error del promedio simple de cada campaña, 1,93 puntos por partido, un 71 % de los 2,71 del mejor cuarto; ese error lo cometen todas las casas a la vez y ningún rating lo corrige.
- **La contracción se queda corta.** β = 0,44 y el intervalo no llega a 1 (0,72 como mucho): la desviación prevista exagera las diferencias, aunque ya va muy contraída. El intervalo toca el 0 (−0,02), así que no se puede descartar que la desviación prevista no anticipe nada. En las generales β = 0,06 (−0,45 a 0,50): no se distingue de no anticipar nada, y el intervalo descarta el 1.
- **El acierto relativo es nulo.** Predecir la desviación con el rating es un 3,7 % peor que predecir 0 (un 13 % peor en las generales, un 0,5 % mejor en las autonómicas). No contradice al Spearman: el rating ordena algo, pero exagera el tamaño de las desviaciones (β < 1), y lo que gana con el orden lo pierde con la exageración. La calidad previa sola es **peor que nada** (−16 %; −24 % en las generales): leída como desviación con −τ·Φ⁻¹(*Q*), la rúbrica predice mal. Una explicación probable, no medida: la previsión de la calidad previa es asimétrica. *Q* va de 0,05 a 0,64, así que −τ·Φ⁻¹(*Q*) va de +0,016 a −0,0036: prevé desviaciones positivas, a veces grandes, y nunca negativas de más de 0,0036. *R* también baja a 0,05 (+0,016), pero sube hasta 0,85 (−0,010), así que el rating prevé a los dos lados de 0. Las medias no lo explican: 0,42 la de *Q* y 0,43 la de *R*.
- **El promedio no mejora con el rating.** En 71 elecciones, el promedio simple de la última campaña con el rating yerra 0,011 puntos más que sin él, un 0,5 %, con un intervalo de −0,020 a +0,048 que incluye el 0, y 37 elecciones mejoran frente a 34 que empeoran. La diferencia media queda dentro del ±2 % en las tres poblaciones. Es lo esperable con un orden débil: aunque los pesos por rating difieren bastante (en 2023, 0,57 el CIS y 2,36 GAD3), el error de cada promedio lo domina lo que fallan todas las casas a la vez, y el orden del rating explica una parte pequeña de lo que se separan.
- **El modelo vivo tampoco cambia (prueba 1a).** Sin el rating, el MAE de voto del backtest sube un 0,8 % en las generales y un 0,4 % en las autonómicas, y el CRPS de escaños un 1,0 % y un 0,7 %, con los casos mejor / peor casi a la par: dentro del ±2 %. El Brier de mayorías lo mueven pocas elecciones: baja un 8,9 % en las generales por 2023, la general más reciente, y las otras tres con Brier lo suben (un 16 % sin 2023, sobre una base de 0,036), y sube un 3,3 % en las autonómicas, sobre un error de 0,02 con la mitad de los casos empatados. Es la prueba 1b con el modelo entero: el rating mueve el error del promedio un 0,5 % (aquí a su favor, allí en contra), sin un signo que se distinga del ruido.

En una frase: el rating discrimina un poco entre casas, sobre todo en las autonómicas, con una contracción que se queda corta; no sirve para predecir el tamaño de la desviación de una casa y no mejora el promedio, ni el simple ni el del modelo. Defiende el ranking como orden aproximado, no como predicción.

## Decisión (propuesta)

La fase C2 (afinar parámetros del rating con la verosimilitud fuera de muestra de la prueba 2, con cientos de pares en vez de cinco generales) es una decisión de Luis. Lo que las cifras permiten:

- **La pendiente dice en qué dirección**: β = 0,44 con el intervalo por debajo de 1 pide una contracción más fuerte, es decir, un τ menor (y con él k = (σ/τ)² mayor). Coincide con lo que ya dice la tarjeta «Qué límites tiene el rating» (τ = 0,01 parece ancha frente a la dispersión real de las medias de casa: unos 0,0034 descontado el ruido, como en el Problema; 0,006 sin descontarlo, en las casas con cinco o más elecciones). Es el único parámetro sobre el que hay una cifra que apunte.
- **El acierto negativo de la calidad previa cuestiona el peso de la rúbrica** (k·*Q* en `R = (W·R_aj + k·Q)/(W + k)`, unas 2 elecciones de evidencia) o su traducción a desviación. Pero el Spearman de las casas con poca evidencia (−0,18), en cuyo rating el prior pesa cerca de la mitad o más (k = 1,96 frente a W < 2), dice que la rúbrica ordena algo: el problema es de escala, no de orden.
- **Nada dice que el promedio fuera a mejorar.** El rating mueve el promedio simple un 0,5 % y el error de voto del modelo vivo otro 0,5 % (en sentido contrario), dentro del ruido las dos veces. La fase C2a (sección «¿Y con otra banda?») tampoco lo encuentra al ensanchar o estrechar la banda: la curva es plana dentro del ±2 % en el voto. Un C2 cambiaría el orden y las notas del ranking, no la predicción. La mezcla 0,7/0,3, el 0,5 autonómico y el 0,9 por año no tienen ninguna cifra aquí: afinarlos sería una búsqueda a ciegas que perseguiría ruido.

Recomendación, sin decidir:

1. **Nada antes del 29-N.** Cambiar el orden del ranking en campaña es un anti-objetivo de la spec, y las cifras no dan una mejora de la predicción que lo justifique (tampoco al cambiar la banda del rating en el promedio, fase C2a).
2. **Tras el 29-N**, con la sexta general en las cifras (`compute '{"what":["compute","ratings"],"scopes":"all"}'` y `publish analysis` de `es` con `rating_check`), un C2 **acotado** a τ (o k) y, como mucho, al peso de la calidad previa, con este juez: el ECM fuera de muestra de `d_obs` predicho con −τ·Φ⁻¹(*R*) sobre todos los pares (o su verosimilitud normal), bootstrap por casa, y como condiciones que el intervalo de β incluya el 1 y que el acierto relativo del rating no sea negativo. Antes de afinar τ con β hay que medir cuánto de β < 1 es la atenuación por la contracción de `d_obs` sondeo a sondeo (Salvedades): por ejemplo β por tramos del número de sondeos de la elección (10 no separa nada; 30 o más sí separaría).
3. **El peso del rating en el promedio se queda como está.** La prueba 1a dio lo declarado de antemano: el backtest plano queda dentro del ±2 % en voto y escaños, así que el rating es cosmético para la predicción, y quitarlo o dejarlo es una decisión de presentación, no de acierto. Quitarlo tampoco pasa limpio el criterio 17 (en las autonómicas, Brier +3,3 % y cobertura −0,45 puntos), y sin 2023 el plano empeora en las generales en voto, escaños y mayorías: no hay cifra que pida el cambio. Tampoco hay que buscar en C2 una mejora del promedio: el juez de C2 es la prueba 2, no el backtest.
4. **La banda del rating se queda en γ = 1** (fase C2a, 06-11-2026, hecha). Ningún punto de la rejilla γ ∈ {0, 0,5, 1,5, 2, 3} pasa la regla 6 de su spec: las bandas anchas empeoran las generales (a γ = 3, voto +1,8 %, CRPS +1,4 %, Brier +12,5 %) y bajan la cobertura en todos los γ > 1; la estrecha (γ = 0,5) pierde en 56 de los 94 casos; en las autonómicas el voto se mueve dentro del ±1 %. La anchura de la banda es cosmética, como lo es el rating entero, y el valor por defecto no cambia. `rating_exponent` queda como palanca del backtest: si un C2 tras el 29-N cambiara los ratings, la curva se repite con los comandos de esa sección (unas 12 h de portátil).

## Salvedades

- **Pocas generales.** 13 elecciones con pares (1986-2023) y 238 pares; en la prueba 1a, 5 generales (28 casos). Todos los intervalos de las generales cruzan el 0, y una sola campaña con pocas casas (1989, 1993) mueve su promedio simple. En la prueba 1a, una sola elección (2023) da la vuelta al Brier y reduce la diferencia del CRPS de las generales. Las cifras del conjunto las deciden las autonómicas.
- **Autonómicas.** 704 de los 942 pares y 58 de las 71 elecciones del promedio simple, desde 2009. Su rating incluye evidencia de todos los ámbitos con el peso autonómico (0,5), y dos autonómicas del mismo día son dos elecciones que no se ven entre sí. En la prueba 1a sólo entran las cinco de la parte B y las elecciones `featured` desde 2019.
- **`d_obs` contraída por sondeo y atenuación de β.** `d_obs` es la evidencia del rating, no la desviación cruda (que no se guarda): cada sondeo se contrae con N(0, τ) frente al error de la regresión local, más en las fechas con pocos sondeos. β compara en esa misma escala, la de `d_prev` y la de σ y τ, así que 1 sigue siendo la referencia; pero la contracción variable atenúa β hacia abajo en las elecciones poco sondeadas. La comprobación prevista (`slope_polled`, elecciones con 10 o más sondeos) no separa nada, porque todas los tienen; un corte más alto está en la recomendación.
- **El consenso de cada fecha incluye a la propia casa.** La línea del error medio frente a la que se mide cada sondeo incluye los sondeos de la casa: en las fechas con pocos sondeos su desviación se acerca a 0 por partida doble (por la línea y por la contracción). Afecta igual a `d_obs` y a la evidencia del rating.
- **El diferencial mide error absoluto, no sesgo con signo.** `bias = 0,7·bias_blocks + 0,3·bias_within`: una casa que yerra mucho en cualquier dirección tiene desviación positiva. La prueba no dice nada del sesgo hacia un bloque.
- **La calidad media del respaldo.** Para las casas sin fila de rating, la prueba 1b usa `log1p(q) / log1p(media de q)` con la media de las **casas de esa elección**, mientras que el `Forecaster` usa la media de **todo el catálogo** (`self.pollsters.quality.mean()`). La diferencia sólo afecta a las casas sin rating, que en las elecciones valoradas son pocas, pero el promedio simple no es exactamente el peso del modelo. Además, esa `q` es la calidad del catálogo de hoy, no la que tenía la casa en la fecha de la elección (los pares sí usan la calidad de esa fecha): una pequeña mirada al futuro, en pocos sondeos, que el backtest del `Forecaster` también tiene.
- **Intervalos.** El bootstrap de la prueba 2 remuestrea casas, no elecciones: los pares de una misma elección comparten el error común y el consenso, y esa dependencia no entra en el intervalo, que puede quedarse estrecho. El de la prueba 1b remuestrea elecciones.
- **`slope_polled` idéntico a `slope`.** Ver arriba: con los datos de hoy es redundante; se conserva en la parte por si una elección futura con pares tuviera menos de 10 sondeos.
- **Referencia emparejada.** La prueba 1a compara `results_flat` con `results_rated`, corridas el mismo día con el mismo modelo (commits 64b6233 a 6e9d2b9, que sólo difieren en documentos y web) y la misma base. La prueba 1a no tiene intervalos: con 5 generales y 11 autonómicas, los recuentos mejor / peor son la medida de su ruido. `backtest/results` (45aaa8b, 01-11-2026) es otro código y otra base (sondeos, ratings, Compromís): se cita sólo como contexto, nunca como par.

## Reproducción

```bash
# Cifras de 1b y 2 sin escribir en el sitio real (site-dry/v1/analysis/es/rating-check.json)
python job.py publish '{"what":["analysis"],"scopes":["es"],"rating_check":true,"dry_run":true}'
# Publicación real (Luis), tras un compute de ratings
python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
python job.py publish '{"what":["analysis"],"scopes":["es"],"rating_check":true}'
```

En Python: `Computer(scope='es').build_series().rating_check_data('all')` devuelve `pairs` y `polls`; `rating_check.rating_check_summary(pairs, polls, comp.bias_dev_tau, comp.bias_dev_sigma, {})` el `data` de la parte, y `rating_check.simple_average_errors(polls)` la tabla por elección. La prueba 1a, con los comandos de su sección.
