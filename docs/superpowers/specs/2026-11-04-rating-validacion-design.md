# Fase C: ¿sirve el rating de encuestadoras? Validación fuera de muestra y publicación

Fecha: 2026-11-04. Estado: APROBADA por Luis el 2026-11-04 (enfoque B; tarjeta «Cómo lo comprobamos» fusionada).

## Intención

Las fases A (afinar el promedio con los sondeos futuros, método 14) y B (calibrar la incertidumbre, método 15)
no encontraron mejoras que pasaran sus criterios. La fase C natural sería calibrar los parámetros del rating de
encuestadoras. Luis quiere la fase C para dos cosas a la vez: **mejorar la predicción** y **poder defender en
público el ranking de encuestadoras**.

Lo medido hasta hoy dice que el rating tiene poco contenido predictivo que afinar: la desviación de una casa
varía entre elecciones (σ = 0,014 en log odds ratio) más de lo que las casas se separan entre sí (τ = 0,01;
dispersión real de las medias de casa ≈ 0,0034 descontado el ruido); la correlación del error de una casa entre
generales salió 0,02 y de autonómicas a generales 0,04 (spec del 2026-10-20); los cambios del rating no movieron
el backtest (rating por elección: voto igual, CRPS +0,6 %, Brier +2,9 %); ningún criterio de la rúbrica de la
calidad previa fue concluyente dentro de cada elección. Y el error que importa es el compartido por todas las
casas (M6), que ningún rating corrige.

Por eso la fase C **no calibra parámetros contra el backtest de cinco generales** (sería perseguir 2023, el
anti-objetivo de la fase A). La fase C **valida**: mide fuera de muestra si el rating mejora el promedio y si
ordena a las casas de modo que anticipe quién se acercará más, y **publica la respuesta que salga** en
/metodo/rating, con cifras vivas, sea cual sea. Si el rating no discrimina, calibrar lo que no predice sería
maquillaje; si discrimina, la verosimilitud fuera de muestra de esta misma prueba es el juez con el que afinar
después (fase C2, aplazada al 29-N como sexto caso).

## Qué hay hoy

- **Rating** (`Computer.pollster_ratings`, `rate_events`, `compute_ratings(scopes='all')`): por casa y elección
  valorada, con los sondeos de **todas las elecciones de cualquier ámbito celebradas estrictamente antes** de
  ella (fuera de muestra por construcción). El diferencial de cada sondeo (`bias_dev_adj`) es su error absoluto
  (`bias = 0,7·bias_blocks + 0,3·bias_within`, log odds ratio) menos el error medio de los sondeos de la misma
  elección en su fecha (`fit_bias_estimator`, regresión local), contraído con prior N(0, τ). La evidencia se
  agrupa por elección (`_event_deviations`: `bias_dev` y `weight_event = weight_year · weight_scope` por casa y
  elección), se contrae con σ/√n_eff, `rating_adj = Φ(−d/τ)`, `R = (W·R_aj + k·Q)/(W + k)` con k = (σ/τ)² ≈ 1,96
  y `weight_rating = log1p(R)/log1p(mean R)`. Parámetros del `Computer`: `bias_dev_tau` 0,01, `bias_dev_sigma`
  0,014, `error_weights` {blocks 0,7, within 0,3}, `year_decay` 0,9, `pos_decay` 0,5, `rating_weight` 0,5 de las
  autonómicas (catálogo). La calidad previa `Q` es la rúbrica (`quality.py`, «Confianza»), a la fecha de cada
  elección (Implantación cuenta sólo las generales anteriores).
- **Tabla `elections.pollsters_ratings`**: una fila por (`event_scope`, `event_date`, `pollster_id`) con
  `quality`, `num_events`, `num_events_w`, `num_national_events`, las medias descriptivas, `bias_dev_adj`,
  `bias_dev_err`, `rating_adj`, `rating`, `weight_rating`. La fila de la elección e del ámbito s es el rating
  **antes** de e. La escribe `compute ratings` (`compute.rate_scope`), la lee `data.get_ratings` y la API
  (`webapi.query_ratings`, caché) para el ranking de /casas y los perfiles.
- **Promedio** (`Forecaster`): `weight = weight_over · weight_sample · weight_rating`; el rating de cada sondeo es
  el de su propia elección (fuera de muestra); `Forecaster(weight_ratings=Series)` sustituye en memoria los
  ratings por (`event_date`, `pollster_id`), y el backtest lo usa con `--recompute-ratings`
  (`backtest.rating_overrides`). Las casas sin rating llevan su calidad previa.
- **Backtest** (`backtest_scope`, job `backtest`, `backtest/run_backtest.py`): 5 generales × 6 horizontes en `es`
  (28 casos) y las autonómicas `featured` desde 2019 por ámbito; MAE de voto frente a `last_poll` y `mean_4w`
  (líneas base **sin rating**), CRPS de escaños, Brier y log-score de mayorías, coberturas, PIT. Referencias:
  `backtest/results` (es, 17 autonómicos; la de `es` anterior al PIT pero con todas las métricas). Carpetas
  `backtest/results_*/` ignoradas por git (scratch de experimentos). Criterio 17 (método 15): un cambio entra si
  no empeora MAE de voto, CRPS de escaños ni Brier de mayorías más de un 2 % ni baja la cobertura del 95 %, con
  los casos mejor / peor contados.
- **Publicación**: `publish analysis` escribe por ámbito las partes `house-effects`, `herding`, `poll-errors`,
  `error-data` y `meta` (`bundle.ANALYSIS_PARTS`, esquemas en `bundle.SCHEMAS`, contrato en
  `docs/web/contrato.md`); se sobrescribe en cada publicación, y el job `update` la encadena **a diario** tras el
  forecast (`mtpy/jobs/Update.py`). La API (`mtpy/controllers/Analysis.py`) sirve una parte con `?format=csv`
  y deriva el nombre del CSV de la parte; `&table=events` sólo vale para `herding`; `bundle.path_analysis_csv`
  valida contra la lista cerrada `ANALYSIS_CSV`. /metodo/rating (`pages/metodo.py`,
  `web/templates/metodo_rating.html`) tiene diez tarjetas (resumen, qué se mide, diferencial, calidad previa,
  evidencia por elecciones, rating final, ranking global, rebaño, **«Cómo lo comprobamos»** (`comprobacion`) y
  «Qué límites tiene el rating» (`limites`)) con cifras vivas y notas desplegables. `comprobacion` dice hoy que
  «el rating no se comprueba por sí solo, porque no pronostica nada: es un peso», publica la tabla fija del
  backtest por sondeos / por elecciones del 20-10-2026 (`rating-backtest-table`) y las correlaciones 0,04 / 0,02
  entre elecciones; la prueba 2 de esta fase mide justo si pronostica, y la 1a repite ese backtest. El
  patrón de una tarjeta con datos medidos es «Fiabilidad de las bandas» en /metodo/backtest
  (`reliability_context`, parte `calibration`, gráficos en `web/dist/js/pages/metodo.js`, caché por TTL del
  sitio). Hay `charts/scatter.js` (nube con diagonal, recta ajustada y puntos nombrados).
- **Pruebas ya hechas** reutilizables como método: bootstrap por casa con 2.000 réplicas y semilla fija (spec del
  2026-10-20, «Prueba por criterio»); desviaciones centradas en la media de la elección. Y la lección de la fase
  A: una referencia almacenada no está emparejada con una corrida nueva (código y base distintos), así que se
  corre también la base con el mismo código (`results_default_rw` en la parte A).
- **Escalas guardadas**: `pollsters_ratings` guarda `bias_dev_adj` en escala «bias» ((e^x − 1)·100, 2
  decimales) y `rating`/`rating_adj` ×100; `rating_pool` devuelve log odds ratio (`bias_to_lor`). El
  `bias_dev_adj` **por sondeo** que escribe `compute_deviations` ya está contraído (`bayes_adjust` con N(0, τ)
  frente al error de la regresión local); la desviación cruda por sondeo no se guarda. Las casas sin fila de
  rating llevan en el promedio su calidad previa normalizada por la calidad media de las casas de la elección
  (`Forecaster.build_series`).

## Restricciones

- Nada de lo que se construye cambia el modelo vivo ni el orden del ranking: el rating, sus parámetros y su
  peso en el promedio se quedan como están. El único código que toca el modelo es una opción nueva del
  backtest/Forecaster para anular el rating, inerte por defecto.
- Ninguna escritura en la RDS, S3 ni ECS por mi cuenta: los backtests corren en el portátil en carpetas propias;
  la publicación de la parte nueva la lanza Luis (o yo con su autorización explícita).
- Sin cambios de esquema en la base (Luis hace los ALTER a mano y no hacen falta: todo se publica en el bundle).
- Calendario: la veda empieza el 23-11 y la general es el 29-11. El diagnóstico (pruebas 1 y 2) puede correr
  esta semana en paralelo a la tarea de `average_errors`, porque sólo lee. La tarjeta se despliega antes del
  17-11 si Luis quiere (no toca el modelo) o después del 29-N. Tras el 29-N, un `compute ratings` + `publish
  analysis` añade la sexta general a las cifras vivas sin tocar código.
- Redacción pública: las reglas de Luis (nosotros con firma, todo a la vista, notas desplegables sólo para
  expertos, «sólo» con tilde, encuestadora/casa nunca empresa, «actualización» para los runs). La tarjeta
  publica la cifra que salga, también si es neutra: es lo aprobado para esta fase y lo que defiende el ranking
  en público (enseñar la prueba, no el argumento). La regla de /metodo/promedio de no decir que el backtest no
  mejora no se aplica aquí.
- Nombres de código en inglés neutro (`rating_check`, no calcos); los textos públicos en español.
- Una elección es un ámbito y una fecha; dos autonómicas el mismo día son dos elecciones, y ninguna ve a la
  otra en su rating (como en `rate_events`).

## Enfoques

- **A. Diagnóstico y documento.** Las pruebas 1 y 2 en scripts de scratch y un `metodo-16-rating.md` con las
  tablas. Un día de portátil, sin código en el modelo. No deja nada en la web: no sirve al segundo objetivo.
- **B. Diagnóstico publicado (elegido).** Las pruebas como funciones con tests junto al backtest y al rating,
  una parte `rating-check` del bundle y una tarjeta en /metodo/rating con cifras vivas, al estilo de
  «Fiabilidad». Dos o tres días de SDD. Sirve a los dos objetivos y se actualiza sola con cada elección.
- **C. B más calibración.** Afinar la mezcla 0,7/0,3, el 0,5 autonómico y el decaimiento por año con la
  verosimilitud fuera de muestra de la prueba 2 (cientos de pares casa-elección, no cinco generales). Defendible,
  pero cambiaría el orden del ranking en campaña y, por lo medido, no movería el promedio. Queda como fase C2,
  sólo si la prueba 2 muestra que el rating discrimina, y después del 29-N.

Luis eligió **B** el 2026-11-04, con A como su primera tarea para que las cifras decidan el texto de la tarjeta
antes de escribirla.

## Objetivos

1. Saber si el rating mejora el promedio: en el modelo vivo (backtest) y en un promedio simple sobre todas las
   elecciones valoradas.
2. Saber si el rating discrimina: si el rating antes de una elección anticipa la desviación de la casa en ella,
   y si está calibrado (la desviación prevista se cumple en la proporción que dice).
3. Publicar las dos respuestas en /metodo/rating con cifras vivas y descargables, y documentar el método y sus
   resultados (`docs/analisis-2026-09/metodo-16-rating.md`).
4. Dejar decidido, con cifras, si hay una fase C2 de calibración y con qué juez.

## No objetivos

- Afinar ningún parámetro del rating ni de la rúbrica (fase C2, si procede).
- Meter el rebaño en el rating, cambiar la calidad previa o el peso autonómico (decisiones cerradas de Luis).
- Afinar el rating con sondeos futuros (mediría cercanía al consenso, no a la verdad; no objetivo de la fase A).
- Tocar el ranking de /casas, los perfiles o el método de /metodo/promedio.
- Backtests autonómicos de los 17 ámbitos para la prueba 1: bastan `es` y los cinco de la parte B.
- Reproducir las pruebas de la rúbrica del 2026-10-20 (ya publicadas en el método).

## Anti-objetivos

- Una tarjeta que sólo aparece, o sólo dice algo, cuando el resultado favorece al rating.
- Vender un resultado neutro como mejora, o uno negativo como «el rating ordena, luego sirve».
- Contar la misma elección varias veces (autonómicas del mismo día) o mirar al futuro en algún par
  casa-elección (un rating calculado con la elección que se juzga).
- Cambiar k, τ o la mezcla «a ojo» a raíz de las cifras: eso es C2, con su juez y su decisión aparte.
- Añadir minutos a cada `publish analysis` sin medirlos.

## Diseño

### D1. Prueba 1a: rating plano en el backtest del modelo vivo

- Opción nueva `flat_ratings` del backtest (`backtest_scope`/`run_backtest`, job `backtest` como parámetro
  JSON `flat_ratings`, CLI `--flat-ratings`, excluyente con `--recompute-ratings`), registrada en `run.json` y
  en `meta.csv` como las demás. Cómo llega al Forecaster: `Forecaster(weight_ratings='flat')` pone
  `weight_rating = 1` en **todos** los sondeos de la serie (también en las casas sin rating, que hoy llevan su
  calidad previa); el `Simulator` y `run_case` lo enhebran como enhebran `weight_ratings`. Con una `Series` el
  comportamiento actual no cambia. (Decisión mía: la cadena `'flat'` en vez de construir una `Series` de unos,
  porque la `Series` dejaría fuera a las casas sin fila de rating.)
- Corridas, en el portátil, **dos carpetas hermanas** ignoradas por git: `backtest/results_flat` (rating
  plano) y `backtest/results_rated` (con rating, mismo código y misma base, la referencia emparejada), para
  `es` (28 casos) y los cinco autonómicos de la parte B (`es-md`, `es-an`, `es-ct`, `es-vc`, `es-ga`), con los
  parámetros por defecto del modelo vivo (misma semilla, mismo `n_sim`). La referencia almacenada
  `backtest/results` no sirve de par: es de otro código (45aaa8b, 2026-11-01) y de otra base (sondeos, ratings,
  Compromís), y con un criterio del ±2 % eso confunde; se cita sólo como contexto. Coste: ≈ 2 h `es` y 2-4 h los
  cinco, en segundo plano.
- Juicio: criterio 17, caso a caso (mejor / peor contados): MAE de voto, CRPS de escaños, Brier de mayorías,
  cobertura del 95 %; además la comparación con `mae_last_poll` y `mae_mean_4w`, que ya son promedios sin
  rating. Lectura esperada y declarada de antemano: si plano ≈ con rating (dentro del ±2 %), el rating es
  cosmético para la predicción.
- Resultado: tablas antes / plano por horizonte y ámbito en `metodo-16-rating.md`, sección «¿Mejora el
  promedio? (modelo vivo)». No entra en el bundle (es una corrida de experimento, como las del método 15).

### D2. Prueba 1b: rating plano en un promedio simple, en todas las elecciones valoradas

La prueba con potencia estadística: ~40 elecciones en vez de 5. Para cada elección valorada e (ámbito y fecha
con resultado y rating calculado), los sondeos útiles (`computed`, sin `drop_mtypes`, sin `wban`/`exit`,
`weight_over > 0`) publicados en los **30 días** anteriores, con al menos **3 casas** distintas:

- Para cada partido principal del ciclo (`bmaps.main` de la elección, con la sucesión de partidos que aplica el
  Computer): el promedio ponderado de la cifra publicada con dos pesos, `w_plano = weight_over · weight_sample` y
  `w_rating = w_plano · weight_rating`, donde `weight_rating` es el de la fila (e, casa) de
  `pollsters_ratings` (antes de e) y la calidad previa para las casas sin fila. Error absoluto de cada promedio
  frente al resultado oficial, MAE sobre los partidos principales por elección.
- Resumen: diferencia emparejada por elección (con rating − plano, puntos de voto), media, IC 95 % por
  bootstrap de elecciones (2.000 réplicas, semilla fija) y recuento mejor / peor, para todas las elecciones y
  por población (generales, autonómicas).
- También los dos pesos de la mezcla de referencia: `weight_sample` solo (sin sobrepublicación) no se prueba;
  el objeto es el rating, no el resto de los pesos.
- Funciones puras en un módulo nuevo `mtpy/lib/rating_check.py` (`simple_average_errors(polls, ratings,
  results, ...)` y `summarize_simple_average(frame)`); los datos los reúne `Computer.rating_check_data()` (ver
  D4). Entra en la parte `rating-check` (D5) y en la tarjeta (D6) como «el promedio de la última campaña, con y
  sin rating».

### D3. Prueba 2: discriminación y calibración fuera de muestra

Pares (casa, elección): para cada elección valorada e y cada casa con sondeos útiles en e, la **desviación
observada** `d_e` de la casa en e (`_event_deviations` sobre los sondeos de e: `bias_dev` en log odds ratio,
centrada por fecha frente al consenso y contraída por sondeo, **exactamente la evidencia que el propio rating
usa**; no se recalcula la desviación cruda) y, de la fila (e, casa) de `pollsters_ratings`,
el **rating antes de e** (`rating`, `rating_adj`, `weight_rating`), la **desviación prevista** `d_prev =
bias_dev_adj` antes de e (la media contraída de sus elecciones anteriores; NaN si no tenía ninguna) y su
evidencia (`num_events_w`, `num_national_events`, `quality`). Las casas de `drop_mtypes` no entran (no tienen
rating). Población de cada par: generales o autonómicas, por el ámbito de e.

Medidas (módulo `rating_check.py`, funciones puras sobre el frame de pares; las cifras con IC por bootstrap por
casa, 2.000 réplicas, semilla fija, como la prueba por criterio del 2026-10-20):

1. **Correlación de rango** (Spearman) entre el rating antes de e y `d_e`, sobre todos los pares, por población,
   y en dos cortes de evidencia: casas con `num_events_w ≥ 2` (más evidencia que prior) y con menos. Signo
   esperado si el rating sirve: negativo (más rating, menos desviación).
2. **Pendiente de calibración**: regresión de `d_e` sobre `d_prev` (pares con `d_prev` no nulo), pendiente β
   con IC. β ≈ 1 calibrado; 0 inútil; entre medias, la contracción (τ frente a σ) es demasiado floja o fuerte.
   Es la cifra que diría con qué juez afinar k y τ en C2. Lectura declarada: β compara la predicción del rating
   con la misma magnitud (contraída por sondeo) con la que el rating se alimenta y con la que se midieron σ y τ,
   así que 1 sigue siendo la referencia en esa escala; la contracción por sondeo varía con el número de sondeos
   de cada fecha (más en fechas con pocos), lo que **atenúa** β hacia abajo en las elecciones poco sondeadas.
   Esa salvedad va al método 16 y a la nota de la tarjeta; se publica además β sobre los pares de elecciones
   con 10 o más sondeos en la ventana del consenso como comprobación.
3. **Skill frente a tres predictores** de `d_e`: nada (0), sólo calidad previa (`Φ⁻¹`-equivalente: la
   desviación que implica `rating_adj = Q`, es decir `−τ·Φ⁻¹(Q)`) y el rating completo (`−τ·Φ⁻¹(R)`): error
   cuadrático medio de cada uno y `skill = 1 − MSE(R)/MSE(0)`, por población. Si Q sola iguala a R, la evidencia
   no añade nada al prior; si 0 iguala a ambos, nada predice.
4. **Lectura para el público**: cuartos del rating antes de e dentro de cada elección (primer cuarto frente al
   último): error medio por partido en e (`error_avg`, puntos de voto) de cada cuarto, por población, con el
   número de pares. Es lo que un lector entiende: «las casas mejor valoradas fallaron X frente a Y».
5. **Recuentos**: pares, casas, elecciones por población, y el rango de fechas.

Decisiones mías: los pares pesan igual (cada elección pesa por sus casas; se publica además por población);
no se aplica `weight_event` (la pregunta es predictiva, no de estimación); las autonómicas del mismo día son
pares distintos y sus ratings no se ven entre sí (`rate_events`); la desviación observada se calcula con los
pesos de posición del rating dentro de e (`weight_pos`), como la usa el propio rating.

### D4. Reunir los datos

`Computer.rating_check_data(scopes='all')` devuelve el frame de pares de D3 y el de sondeos de D2: la bolsa del
rating (`rating_pool`, todos los ámbitos con `rating_weight > 0`, hoy 18 Computers) cruzada con las filas de
`pollsters_ratings` de cada elección valorada (`data.get_ratings` por ámbito, deshaciendo sus escalas con
`bias_to_lor` y ÷100) y con los resultados oficiales (`events`). Sólo lectura. El coste se mide en la primera
tarea (`rating_pool` de 18 ámbitos ya lo paga `compute ratings`; estimación: 1-3 min) y se registra en
`meta.seconds`.

### D5. Publicación: parte `rating-check` del análisis nacional

- Parte nueva `rating-check` en `bundle.ANALYSIS_PARTS`, esquema `analysis-rating-check@1` en
  `bundle.SCHEMAS`, CSV gemelos `rating-check-pairs` (un par casa-elección por fila: ámbito, fecha, casa,
  población, rating, `rating_adj`, `d_prev`, `d_e`, `error_avg` en e, `num_events_w`, `quality`) y
  `rating-check-elections` (una elección por fila: MAE plano y con rating de D2, casas, sondeos). JSON con
  `summary` (las medidas 1-3 y 5 de D3 y el resumen de D2, por población: `all`, `generales`, `autonomicas`),
  `quarters` (medida 4), `scatter` (los pares con `d_prev` y `d_e` para la nube, en la escala del diferencial
  del ranking: ×100 del log odds ratio, como «Diferencial») y `params` (ventana de 30 días, mínimo de casas,
  réplicas, semilla, τ, σ, k).
- La escribe `publish analysis` **sólo para `NATIONAL_SCOPE`** (el ranking es nacional y la prueba reúne todos
  los ámbitos), tras `error-data` y antes de `meta`, y **sólo cuando se pide**: parámetro `rating_check`
  (`false` por defecto) del job `publish` (`{"what":["analysis"],"scopes":["es"],"rating_check":true}`) y del
  pipeline, que el job `update` pone a `true` cuando su `compute` fue `full` (los ratings cambian sólo entonces:
  tras un resultado o con `overwrite`). Así la publicación diaria no paga la bolsa de 18 ámbitos y «se actualiza
  sola con cada elección» se cumple por el `update` full que sigue al 29-N. Cuando no se pide, la parte anterior
  **se conserva** (no se borra como la `house-effects` rancia: sus cifras no caducan con el ciclo); `meta.params`
  registra `rating_check` (true cuando se recalculó en esa publicación) y `meta.seconds` su coste. En los
  ámbitos autonómicos la parte no existe (404 `analysis part not published`, como hoy una parte ausente).
  Enmiendas en `docs/web/contrato.md` (tabla de partes del análisis), en el README de publish y en el de jobs.
- CSV gemelos como `herding`: el de los pares se llama como la parte (`rating-check.csv`,
  `/api/v1/analysis/es/rating-check?format=csv`) y el de las elecciones `rating-check-elections.csv`
  (`?format=csv&table=elections`): `table` pasa a admitir `elections` también para `rating-check` en el
  controlador, y los dos nombres entran en `ANALYSIS_CSV`. (Decisión mía: lo que menos toca; se pierde el
  sufijo `-pairs`.)

### D6. Tarjeta «Cómo lo comprobamos» rehecha en /metodo/rating

- **Decidido por Luis (2026-11-04)**: se **funde** la prueba en la tarjeta
  existente `comprobacion` («Cómo lo comprobamos»), que hoy dice lo contrario de lo que esta fase mide («no
  pronostica nada: es un peso») y publica una tabla fija del 20-10. Se conserva su principio (el rating de cada
  elección se calcula sólo con las anteriores, también en las pruebas), se quita la frase «no pronostica nada»,
  la tabla fija `rating-backtest-table` pasa a la línea del backtest plano (D1, con fecha) y el párrafo de las
  correlaciones 0,04 / 0,02 lo sustituye la correlación viva de la prueba 2. Lead nuevo: «El rating de cada
  elección se calcula sólo con las elecciones anteriores a ella, y lo comprobamos por lo que hace: si anticipa
  qué casas se acercarán más al resultado y si el promedio que pondera con él acierta más.» El índice de la
  página y los anclajes no cambian; «Qué límites tiene el rating» no cambia.
- Contenido, en este orden, con cifras vivas de la parte (sin la parte, la tarjeta no se dibuja, como
  «Fiabilidad» sin `calibration`):
  1. **KPIs** (`m.kpis`): pares casa-elección, elecciones (generales + autonómicas), correlación de rango con
     su IC, pendiente de calibración con su IC, skill del rating frente a nada.
  2. **Nube** (`charts/scatter.js`): desviación prevista (x) frente a observada (y), escala del Diferencial,
     diagonal y recta ajustada; sin nombres (son cientos de puntos); tooltip con casa y elección. Botón de
     descarga de imagen como en las demás tarjetas con gráfico (`share.js`).
  3. **Tabla de cuartos**: por población, error medio por partido del primer cuarto y del último, pares.
  4. **El promedio con y sin rating** (D2): por población, MAE plano, MAE con rating, diferencia con IC y
     mejor / peor.
  5. **«Cómo leerlo»** (`m.how`): tres o cuatro frases escritas **a partir de las cifras** cuando estén
     (primera tarea), en el tono del método: qué significa cada cifra y qué no (un rating que no mejora el
     promedio sigue ordenando casas si la correlación es negativa; una pendiente menor que 1 es una contracción
     corta, no un rating falso). Sin condicionales que cambien el mensaje según el signo: el texto describe las
     cifras que se muestran; sólo las frases que dependen de un signo se escriben con la plantilla (`{% if %}`)
     para que el 29-N no deje un texto falso.
  6. Enlaces a los dos CSV y a una nota desplegable nueva `comprobacion` (fórmulas de las medidas, bootstrap,
     ventana, escalas, la salvedad de β) en las notas de la página, y la línea del backtest del modelo vivo (D1)
     con la fecha de la corrida, cuyas cifras no están en el bundle (van al método 16; cuando cambie, se cambia la
     línea). Las frases que hoy comprueba `assert_public_wording`/los claims de la página se revisan con el texto
     nuevo.
- Contexto en `pages/metodo.py` (`rating_check_context()`, caché por TTL del sitio como `reliability_context`),
  dibujo en `web/dist/js/pages/metodo.js` (subpágina `rating`), datos iniciales en la isla JSON de la página.
- `national_figures` no cambia. /casas y los perfiles no cambian; el glosario gana «pendiente de calibración»
  sólo si se usa en la tarjeta.

### D7. Documento del método y pruebas

- `docs/analisis-2026-09/metodo-16-rating.md`: problema, método (D1-D3), resultados (tablas de D1 por
  horizonte y ámbito; las de D2 y D3 por población), decisión sobre C2 (con qué juez y qué parámetros, o por qué
  no), salvedades (pocas generales; las autonómicas desde 2019; el diferencial mide error absoluto, no sesgo
  con signo; el consenso de cada fecha incluye a la propia casa). Enlace desde el README de `docs/analisis-2026-09`
  y desde `backtest/README.md` (opción `--flat-ratings`).
- Tests: unitarios de `rating_check.py` con frames pequeños construidos a mano (una correlación perfecta, una
  nula, un par sin `d_prev`, dos autonómicas el mismo día); del Forecaster con `weight_ratings='flat'` (todos los
  pesos de rating a 1, series idénticas con `Series`); del job/CLI (`flat_ratings` en `run.json`, excluyente con
  `recompute_ratings`); de `publish analysis` con la parte nueva sólo en `es` y `meta.params`; del contexto y
  la plantilla de la tarjeta (con parte y sin ella) y del esquema; fixtures del bundle en `tests/fixtures/bundle`.
  Regla del proyecto: los tests nunca con `out=None` en el backtest, y ninguno escribe en `backtest/results`.

## Comprobación

- Las cifras de D2 y D3 reproducidas en un cuaderno o script de scratch independiente de `rating_check.py`
  (mismos pares, mismas medidas) en la primera tarea: las dos vías deben coincidir.
- La tarjeta comprobada en el navegador con el bundle real (técnica del scratchpad: ASGI + Chrome por CDP) y
  con el fixture, en escritorio y móvil.
- Suite completa en verde; `assert_public_wording` sobre los textos nuevos.
- `publish analysis` de `es` medido antes y después (segundos de la parte en `meta`).

## Calendario

| Cuándo | Qué |
|---|---|
| 4-6 nov | Tarea 1: datos y pruebas 1b y 2 en scratch + `rating_check.py` con tests; backtests plano y con rating de `es` y los cinco autonómicos en el portátil (≈ 2 h `es`, 2-4 h los cinco, en segundo plano), en paralelo a la tarea de `average_errors` de Luis |
| 6-8 nov | Parte del bundle, tarjeta, documento 16, textos a partir de las cifras |
| antes del 17-11 (opcional) | Luis publica `analysis` de `es` y despliega; o se espera al 29-N |
| tras el 29-N | el `update` full (o `publish analysis` con `rating_check`): la tarjeta suma la sexta general; decisión sobre C2 |

## Lo que queda al constructor

- La forma exacta de enhebrar `'flat'` por `Simulator`/`run_case` y de registrarlo en `meta.csv`.
- Cómo partir `rating_check_data` entre el `Computer` y el módulo puro, y dónde cachear la bolsa si hace falta.
- El esquema JSON concreto de la parte y los nombres de columna de los CSV, dentro de las convenciones del
  contrato (minúsculas con guiones en nombres de parte, `snake_case` en columnas).
- Los textos finales de la tarjeta y la nota, escritos con las cifras reales delante.
- El tamaño y las guardas del bootstrap (semilla, réplicas) como constantes del módulo.

## Decisiones mías que Luis puede cambiar

- Prueba 1a sólo en `es` y los cinco autonómicos de la parte B, en dos carpetas hermanas (`results_flat` y
  `results_rated`, misma base y mismo código), sin usar `backtest/results` como par.
- `d_e` es la evidencia contraída por sondeo que usa el propio rating; β se lee en esa escala con su salvedad.
- La parte se recalcula sólo con `rating_check: true` (y en el `update` full); si no, se conserva la anterior.
- CSV gemelos nombrados como en `herding` (`rating-check` y `rating-check-elections`, `table=elections`).
- Rating plano = `weight_rating = 1` para todas las casas, también las que hoy llevan su calidad previa.
- Ventana de 30 días y mínimo de 3 casas en la prueba 1b; pares con el mismo peso en la prueba 2.
- La parte se publica sólo en el análisis nacional; la tarjeta va en /metodo/rating tras «Un ranking global».
- Las cifras del backtest del modelo vivo (D1) van al método 16 y a una línea de la tarjeta, no al bundle.
- La tarjeta publica el resultado aunque sea neutro o negativo (lo aprobado en el enfoque B).
