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

Con cinco casos por horizonte en las generales (tres a 180 días) y once en las autonómicas, ningún horizonte separa nada: el signo cambia de un horizonte al siguiente. El CRPS de las generales va igual (+1,1 %, −2,4 %, +1,2 %, +3,6 %, +1,5 %, −1,4 %) y su Brier mejora con el plano en todos los horizontes salvo 90 días, por 2023.

### Frente a los promedios sin rating

| Población | Modelo con rating | Modelo plano | Último sondeo | Media de 4 semanas | Casos en que gana al último sondeo (con / plano) | A la media de 4 semanas (con / plano) |
|---|---|---|---|---|---|---|
| Generales | 2,374 | 2,393 | 2,452 | 2,382 | 14 / 15 de 28 | 17 / 15 de 28 |
| Autonómicas | 2,510 | 2,521 | 2,717 | 2,634 | 43 / 44 de 66 | 42 / 42 de 66 |

Las dos líneas base ya son promedios sin rating. En las autonómicas el modelo les gana con y sin rating, por 0,1-0,2 puntos. En las generales el modelo con rating queda 0,009 puntos por debajo de la media de 4 semanas y el plano 0,010 por encima: la ventaja del modelo sobre esa media en las generales es del mismo tamaño que lo que mueve el rating, una centésima de punto, y con 28 casos ninguna de las dos se distingue de 0 (17 / 11 y 15 / 13 casos).

### Juicio (criterio 17)

Leído como el cambio «quitar el rating»: en el MAE de voto y el CRPS de escaños, el plano queda dentro del ±2 % en las dos poblaciones (+0,8 % y +1,0 % en las generales; +0,4 % y +0,7 % en las autonómicas), con los casos mejor / peor casi a la par. El Brier se mueve en sentidos opuestos: −8,9 % en las generales (por 2023; sin ella, +16 %) y +3,3 % en las autonómicas, fuera del ±2 % pero sobre un error de 0,02 con la mitad de los casos empatados. La cobertura del 95 % sube 0,7 puntos porcentuales en las generales y baja 0,45 en las autonómicas. Estrictamente, el plano no pasa el criterio 17 en las autonómicas (Brier y cobertura); con este tamaño de las diferencias, ni lo pasa ni lo suspende por algo que se distinga del ruido.

La lectura declarada de antemano se cumple: **plano ≈ con rating dentro del ±2 % en voto y escaños, así que el rating es cosmético para la predicción del modelo vivo.** El signo, si acaso, favorece al rating por muy poco (+0,5 % de error de voto sin él en los 94 casos), salvo en 2023, la general en la que también el promedio simple con rating falló más que sin él (+0,125 puntos en la prueba 1b). Coincide con la prueba 1b (+0,5 % en el promedio simple, con rating peor; aquí, sin rating peor): en los dos casos, el rating mueve el error del promedio un 0,5 %, en un sentido o en otro.

Las tablas salen de un script de scratch (no commiteado) que lee `metrics.csv` de las dos carpetas y empareja los casos; en `es`, el `by_horizon.csv` de cada carpeta da las mismas medias por horizonte (en los autonómicos, `by_horizon.csv` es por ámbito).

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
- **Los cuartos lo dicen en puntos.** El cuarto con más rating falló 2,71 puntos por partido y el de menos 3,28: 0,57 puntos, un 17 % menos. Los dos cuartos centrales casi no se distinguen (2,85 y 2,83). La mayor parte del error la tienen también las mejores casas. No todo es error compartido: el error de cada par incluye el ruido de muestra de sus sondeos, que el promedio quita. La estimación de estos datos del suelo compartido (M6) es el error del promedio simple de cada campaña, 1,93 puntos por partido, un 71 % de los 2,71 del mejor cuarto; ese error lo cometen todas las casas a la vez y ningún rating lo corrige.
- **La contracción se queda corta.** β = 0,44 y el intervalo no llega a 1 (0,72 como mucho): la desviación prevista exagera las diferencias, aunque ya va muy contraída. El intervalo toca el 0 (−0,02), así que no se puede descartar que la desviación prevista no anticipe nada. En las generales β = 0,06 (−0,45 a 0,50): no se distingue de no anticipar nada, y el intervalo descarta el 1.
- **El acierto relativo es nulo.** Predecir la desviación con el rating es un 3,7 % peor que predecir 0 (un 13 % peor en las generales, un 0,5 % mejor en las autonómicas). No contradice al Spearman: el rating ordena algo, pero exagera el tamaño de las desviaciones (β < 1), y lo que gana con el orden lo pierde con la exageración. La calidad previa sola es **peor que nada** (−16 %; −24 % en las generales): leída como desviación con −τ·Φ⁻¹(*Q*), la rúbrica predice mal. Una explicación probable, no medida: la previsión de la calidad previa es asimétrica. *Q* va de 0,05 a 0,64, así que −τ·Φ⁻¹(*Q*) va de +0,016 a −0,0036: prevé desviaciones positivas, a veces grandes, y nunca negativas de más de 0,0036. *R* también baja a 0,05 (+0,016), pero sube hasta 0,85 (−0,010), así que el rating prevé a los dos lados de 0. Las medias no lo explican: 0,42 la de *Q* y 0,43 la de *R*.
- **El promedio no mejora con el rating.** En 71 elecciones, el promedio simple de la última campaña con el rating yerra 0,011 puntos más que sin él, un 0,5 %, con un intervalo de −0,020 a +0,048 que incluye el 0, y 37 elecciones mejoran frente a 34 que empeoran. La diferencia media queda dentro del ±2 % en las tres poblaciones. Es lo esperable con un orden débil: aunque los pesos por rating difieren bastante (en 2023, 0,57 el CIS y 2,36 GAD3), el error de cada promedio lo domina lo que fallan todas las casas a la vez, y el orden del rating explica una parte pequeña de lo que se separan.
- **El modelo vivo tampoco cambia (prueba 1a).** Sin el rating, el MAE de voto del backtest sube un 0,8 % en las generales y un 0,4 % en las autonómicas, y el CRPS de escaños un 1,0 % y un 0,7 %, con los casos mejor / peor casi a la par: dentro del ±2 %. El Brier de mayorías baja un 8,9 % en las generales sólo por 2023 (sin ella, sube un 16 %) y sube un 3,3 % en las autonómicas, sobre un error de 0,02 con la mitad de los casos empatados. Es la prueba 1b con el modelo entero: el rating mueve el error del promedio un 0,5 % (aquí a su favor, allí en contra), sin un signo que se distinga del ruido.

En una frase: el rating discrimina un poco entre casas, sobre todo en las autonómicas, con una contracción que se queda corta; no sirve para predecir el tamaño de la desviación de una casa y no mejora el promedio, ni el simple ni el del modelo. Defiende el ranking como orden aproximado, no como predicción.

## Decisión (propuesta)

La fase C2 (afinar parámetros del rating con la verosimilitud fuera de muestra de la prueba 2, con cientos de pares en vez de cinco generales) es una decisión de Luis. Lo que las cifras permiten:

- **La pendiente dice en qué dirección**: β = 0,44 con el intervalo por debajo de 1 pide una contracción más fuerte, es decir, un τ menor (y con él k = (σ/τ)² mayor). Coincide con lo que ya dice la tarjeta «Qué límites tiene el rating» (τ = 0,01 parece ancha frente a la dispersión real de las medias de casa: unos 0,0034 descontado el ruido, como en el Problema; 0,006 sin descontarlo, en las casas con cinco o más elecciones). Es el único parámetro sobre el que hay una cifra que apunte.
- **El acierto negativo de la calidad previa cuestiona el peso de la rúbrica** (k·*Q* en `R = (W·R_aj + k·Q)/(W + k)`, unas 2 elecciones de evidencia) o su traducción a desviación. Pero el Spearman de las casas con poca evidencia (−0,18), en cuyo rating el prior pesa cerca de la mitad o más (k = 1,96 frente a W < 2), dice que la rúbrica ordena algo: el problema es de escala, no de orden.
- **Nada dice que el promedio fuera a mejorar.** El rating mueve el promedio simple un 0,5 % y el error de voto del modelo vivo otro 0,5 % (en sentido contrario), dentro del ruido las dos veces. Un C2 cambiaría el orden y las notas del ranking, no la predicción. La mezcla 0,7/0,3, el 0,5 autonómico y el 0,9 por año no tienen ninguna cifra aquí: afinarlos sería una búsqueda a ciegas que perseguiría ruido.

Recomendación, sin decidir:

1. **Nada antes del 29-N.** Cambiar el orden del ranking en campaña es un anti-objetivo de la spec, y las cifras no dan una mejora de la predicción que lo justifique.
2. **Tras el 29-N**, con la sexta general en las cifras (`compute '{"what":["compute","ratings"],"scopes":"all"}'` y `publish analysis` de `es` con `rating_check`), un C2 **acotado** a τ (o k) y, como mucho, al peso de la calidad previa, con este juez: el ECM fuera de muestra de `d_obs` predicho con −τ·Φ⁻¹(*R*) sobre todos los pares (o su verosimilitud normal), bootstrap por casa, y como condiciones que el intervalo de β incluya el 1 y que el acierto relativo del rating no sea negativo. Antes de afinar τ con β hay que medir cuánto de β < 1 es la atenuación por la contracción de `d_obs` sondeo a sondeo (Salvedades): por ejemplo β por tramos del número de sondeos de la elección (10 no separa nada; 30 o más sí separaría).
3. **El peso del rating en el promedio se queda como está.** La prueba 1a dio lo declarado de antemano: el backtest plano queda dentro del ±2 % en voto y escaños, así que el rating es cosmético para la predicción, y quitarlo o dejarlo es una decisión de presentación, no de acierto. Quitarlo tampoco pasa limpio el criterio 17 (en las autonómicas, Brier +3,3 % y cobertura −0,45 puntos), y sin 2023 el plano empeora en las generales en voto, escaños y mayorías: no hay cifra que pida el cambio. Tampoco hay que buscar en C2 una mejora del promedio: el juez de C2 es la prueba 2, no el backtest.

## Salvedades

- **Pocas generales.** 13 elecciones con pares (1986-2023) y 238 pares; en la prueba 1a, 5 generales (28 casos). Todos los intervalos de las generales cruzan el 0, y una sola campaña con pocas casas (1989, 1993) mueve su promedio simple. En la prueba 1a, una sola elección (2023) da la vuelta al Brier y al CRPS de las generales. Las cifras del conjunto las deciden las autonómicas.
- **Autonómicas.** 704 de los 942 pares y 58 de las 71 elecciones del promedio simple, desde 2009. Su rating incluye evidencia de todos los ámbitos con el peso autonómico (0,5), y dos autonómicas del mismo día son dos elecciones que no se ven entre sí. En la prueba 1a sólo entran las cinco de la parte B y las elecciones `featured` desde 2019.
- **`d_obs` contraída por sondeo y atenuación de β.** `d_obs` es la evidencia del rating, no la desviación cruda (que no se guarda): cada sondeo se contrae con N(0, τ) frente al error de la regresión local, más en las fechas con pocos sondeos. β compara en esa misma escala, la de `d_prev` y la de σ y τ, así que 1 sigue siendo la referencia; pero la contracción variable atenúa β hacia abajo en las elecciones poco sondeadas. La comprobación prevista (`slope_polled`, elecciones con 10 o más sondeos) no separa nada, porque todas los tienen; un corte más alto está en la recomendación.
- **El consenso de cada fecha incluye a la propia casa.** La línea del error medio frente a la que se mide cada sondeo incluye los sondeos de la casa: en las fechas con pocos sondeos su desviación se acerca a 0 por partida doble (por la línea y por la contracción). Afecta igual a `d_obs` y a la evidencia del rating.
- **El diferencial mide error absoluto, no sesgo con signo.** `bias = 0,7·bias_blocks + 0,3·bias_within`: una casa que yerra mucho en cualquier dirección tiene desviación positiva. La prueba no dice nada del sesgo hacia un bloque.
- **La calidad media del respaldo.** Para las casas sin fila de rating, la prueba 1b usa `log1p(q) / log1p(media de q)` con la media de las **casas de esa elección**, mientras que el `Forecaster` usa la media de **todo el catálogo** (`self.pollsters.quality.mean()`). La diferencia sólo afecta a las casas sin rating, que en las elecciones valoradas son pocas, pero el promedio simple no es exactamente el peso del modelo.
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
