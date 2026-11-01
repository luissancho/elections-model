# Afinar el promedio con los sondeos futuros y calibrar la incertidumbre del resultado

Fecha: 2026-11-01. Estado: borrador para Luis, con sus decisiones de la conversación del 2026-11-01 («Quiero hacerlo
ya»: el proyecto empieza ahora, antes del 29-N). Escrita sin acceso a la base de datos: las cifras salen de
`backtest/results/` y de los documentos de método. Revisada por un constructor independiente el mismo día; sus
nueve preguntas están resueltas en «Decisiones tras la revisión» al final, como decisiones mías que Luis puede
corregir. Planes de ejecución: `docs/superpowers/plans/2026-11-01-sondeos-futuros.md` (parte A) y
`docs/superpowers/plans/2026-11-01-calibracion-incertidumbre.md` (parte B).

Origen: la pregunta de Luis sobre qué cabe esperar del modelo frente a las encuestadoras («supera al 64 % de las
encuestadoras y sus errores no son tan diferentes a los del resto»). La respuesta, medida sobre el propio backtest:
a 6 días el error del modelo y el de la media simple de cuatro semanas tienen una correlación de 0,97 y se separan
0,61 puntos de RMSE, con un error de 2,07; alrededor del 90 % de la varianza del error del modelo es el error que
comparte todo el sector esa elección, que ninguna ponderación puede quitar. Lo que depende del método son esas seis
décimas, y cinco elecciones no pueden afinarlas. Y la incertidumbre, el producto que ninguna encuesta da, está mal
calibrada a partir de dos semanas. Son los dos objetivos de este proyecto.

## Intención

Dos partes, en este orden, porque la segunda se calibra sobre el promedio que salga de la primera:

**Parte A. Afinar el promedio con los sondeos futuros.** El promedio de un día es una afirmación sobre dónde está el
consenso de las encuestas ese día; los sondeos que salen después, que el promedio no ha visto, son la prueba más
limpia de esa afirmación, y hay miles. Cada parámetro del promedio que hoy está fijado por razonamiento (ancho de
banda, prolongación, ventana de sobrepublicación, tope de la muestra, contracción de los efectos de casa y del
rebaño) pasa a elegirse por su capacidad de predecir los sondeos de los días siguientes a cada corte, en todos los
ciclos con sondeos, fuera de muestra y con un ciclo de comprobación aparte. Es la validación de FiveThirtyEight para
su promedio (ver `docs/analisis-2026-09/research.md`, §1.2).

**Parte B. Calibrar la incertidumbre del resultado.** Que la banda «del 50 %» acierte una de cada dos veces y la
«del 95 %» diecinueve de cada veinte, a cualquier horizonte. Hoy, en modo Elección: 0,56 / 0,78 / 0,96 a 6 días y
0,30 / 0,67 / 0,84 a 90 (`by_horizon.csv`, coberturas del 50 / 80 / 95 %). Las piezas: la escala del error terminal
medida sobre el propio promedio y no sobre sondeos sueltos, el crecimiento con el horizonte medido igual, colas con un
parámetro propio, y un diagnóstico de fiabilidad que lo enseñe en la web.

Éxito: (A) una configuración que predice mejor los sondeos futuros en los ciclos de comprobación que la actual y no
empeora el backtest de elecciones; (B) coberturas y PIT compatibles con el nivel nominal en todos los horizontes
juntando generales, autonómicas y provincias, sin ensanchar donde ya cubre.

## Qué hay hoy

- **Backtest de elecciones** (`mtpy/lib/backtest.py`, job `backtest`): 5 generales × 6 horizontes (28 casos) y 29
  autonómicas × 6 horizontes; por caso, MAE del voto frente a tres líneas base, coberturas del 50 / 80 / 95 %,
  escaños, CRPS, Brier y log-score, celdas provinciales. Exporta sólo esos tres cuantiles (`lo50`…`hi95`, y `_h`).
  Criterio de aceptación vigente (M12, M13): un cambio entra si no empeora MAE de voto, CRPS de escaños y Brier
  más allá del ruido de cinco elecciones, con los casos mejor / peor contados.
- **Forecaster** (`mtpy/lib/forecaster.py`): regresión local de grado 1, kernel gaussiano, ancho de banda piloto
  ISJ y adaptativo con 3 iteraciones; el análisis de septiembre (`forecaster.md`, §4) midió que el ancho adaptativo
  se duplica en la frontera (128 frente a 70 días) y dejó tres hallazgos sin verificar sobre el piloto (F056, F057,
  F106). Acepta un corte `drange` en días antes de la elección y una sustitución de ratings en memoria
  (`weight_ratings`). Los pesos `weight_over` (1 / N en ventanas de `wspan` = 4 días), `weight_sample` (raíz de la
  muestra tratada, tope `ol_max` = 5000, por la muestra efectiva de M12/M13) y `weight_rating` se precalculan en la
  base (`Computer.compute_weights`) y el Forecaster los lee. Efectos de casa con backfitting de 3 iteraciones y
  prior contraído (M6); rebaño con contracción `n / (n + 5)` (M13); `max_fc` = 10 días de prolongación.
- **Simulator** (`mtpy/lib/simulator.py`): la desviación de cada partido es `sqrt(s² + e² + d²)`: `s` el error
  estándar de la regresión local (cluster por casa, M10), `e` el estimador de error terminal (`get_error_estimator`:
  error absoluto de la última encuesta de cada casa en las generales desde 1982, sobre nivel, regional y semanas,
  con `weeks` = 1 desde M5) y `d` la deriva `p · m · sqrt(k · h)` (M5, M8). El sorteo es una t con `nobs − 2`
  grados de libertad (mínimo 3): con muchos sondeos es una normal. La copula gaussiana entre partidos existe
  (composición, M7) y está desactivada. `e` es el error de una encuesta suelta, no el del promedio: incluye el
  ruido que el promedio elimina y no incluye el crecimiento con el horizonte que el backtest mide (M5, lectura 3:
  el error que falta a 30-180 días «es el doble de lo que da k»).
- **`Computer.get_drift_data`** ya recorre los ciclos con sondeos ajustando un Forecaster por ciclo: es el patrón
  del bucle que la parte A necesita.
- **Cuaderno `notebooks/lab/PollsOptimize.ipynb`**: exploración manual de parámetros, sin criterio fuera de muestra.
- **Jobs** (`job.py`, `mtpy/jobs/Backtest.py`, `deploy/ecs/job.py`): el backtest corre por ámbitos aislados,
  escribe en `backtest/results/` o en S3 y se lanza desde el portátil o en ECS. Un job nuevo entra por el mismo
  camino.

## Restricciones

- **El 29-N está a cuatro semanas** y el modelo publica en campaña. La veda de sondeos empieza el 24-11 (últimos
  sondeos el 23-11). Ningún cambio en los valores por defecto del modelo vivo después del **20-11**, y sólo con el
  criterio de aceptación de abajo. Nada de lo que se construye obliga a cambiar el modelo vivo: todo entra como
  opción documentada hasta que el criterio lo respalde.
- **Base de datos local**: el bucle de sondeos futuros y los backtests necesitan la base (`elections`), que no está
  en esta sesión. Los tests unitarios no la necesitan; los de integración se saltan sin ella.
- **Coste**: un ajuste del Forecaster sin efectos de casa tarda segundos; el backfitting añade 7-25 s. Con 14 ciclos
  y cortes semanales en el último año son unos 700 ajustes por configuración. La búsqueda se diseña para no
  multiplicar eso por decenas (ver A6, A10).
- **Cinco generales** para aceptar cambios en lo que va del promedio al resultado: las diferencias son del 1-2 % y
  están dentro del ruido; el criterio se fija antes y se cuenta por casos, como hasta ahora.
- **Fuera de muestra siempre**: cada corte sólo ve sondeos anteriores; los estimadores sólo elecciones anteriores
  a la evaluada; los ciclos de comprobación no entran en la búsqueda.
- **Reproducibilidad**: semilla, `run.json` con commit y estado de la base, tablas versionadas como las del
  backtest.
- **Redacción pública**: reglas de Luis (`assert_public_wording`) en cualquier texto de la web.

## Enfoques

### Parte A: el promedio

- **A1. Afinar la regresión local actual con validación de origen móvil sobre los sondeos futuros.** Un bucle
  alrededor del Forecaster, sin tocar el método: cortes en cada ciclo, predicción de los sondeos de los k días
  siguientes, puntuación, búsqueda por coordenadas sobre una rejilla fijada de antemano, comprobación en ciclos
  reservados. Cambia parámetros, no el modelo; cuesta un módulo y un job; sirve después para juzgar cualquier otro
  método. **Elegido.**
- **A2. Sustituir la regresión local por un modelo de espacio de estados** (paseo aleatorio latente con filtro de
  Kalman y suavizador, Jackman 2005 / Linzer 2013): sin retardo en la frontera, efectos de casa identificados y una
  incertidumbre que crece con el horizonte por construcción. Es lo más fundado a medio plazo, pero son dos a cuatro
  semanas por sí solo y necesitaría exactamente el arnés de A1 para demostrar que mejora. **Aplazado a después del
  29-N**; A1 construye lo que lo juzgará.
- **A3. Afinar contra las elecciones** (el backtest actual como función objetivo). 28 casos no separan seis
  décimas; es la razón de este proyecto. **Descartado.**

### Parte B: la incertidumbre

- **B1. Recalibrar el modelo de error actual con los errores del propio promedio.** Sustituir el error terminal
  de sondeos sueltos por el del promedio a seis días, medido fuera de muestra en los ciclos anteriores; medir el
  crecimiento con el horizonte sobre la misma tabla y escalar con él la deriva; dar a las colas un parámetro propio
  elegido por el log-score; reevaluar la correlación entre partidos. Mantiene la fórmula `sqrt(s² + e² + d²)` que
  la web explica, cambia lo que alimenta cada pieza. **Elegido.**
- **B2. Inflar las bandas por horizonte** con un factor sacado de la tabla de coberturas. Es lo más barato y lo que
  queda como red si B1 no llega antes del 20-11, pero no distingue colas de escala ni partidos grandes de pequeños,
  y la web tendría que explicar un factor sin modelo detrás. **Reserva**, no destino.
- **B3. Modelo bayesiano completo** (Linzer, HDSR 2020): fuera del alcance y del calendario.

## Decisiones

### Parte A

1. **Ciclos.** Las generales con sondeos suficientes (las `featured` del Computer: diez o más sondeos útiles),
   desde 2008; las autonómicas `featured`; y los ciclos en curso (2026 nacional y autonómicos), que no necesitan
   resultado. Mismo filtro de sondeos que el modelo (`drop_mtypes`, `computed`).
2. **Cortes.** Uno cada 7 días en los 365 días anteriores a cada elección (hasta hoy en los ciclos en curso); se
   descarta un corte con menos de `min_polls` = 5 sondeos útiles en los 90 días anteriores. Un corte es un
   Forecaster con `drange` = días entre el corte y la elección, exactamente como congela el backtest.
3. **Predicción de un sondeo futuro.** Para cada sondeo publicado en los k días siguientes al corte y cada partido
   principal del ciclo (`bmaps.main`, con la sucesión de partidos que ya aplica el Forecaster): el promedio del
   corte leído en la fecha del sondeo (la recta prolongada hasta `max_fc` días y después el último valor), más el
   efecto de casa de esa casa cuando los efectos están activados. Se compara cifra publicada con cifra publicada.
   **Sondeos objetivo**: los que entrarían en el promedio (`computed`, sin `drop_mtypes`) y además con
   `weight_over` > 0 en la base, es decir, sin las oleadas solapadas de los trackings (477 sondeos): el descarte
   por solapamiento no depende de `wspan`, así que el conjunto objetivo es el mismo para todas las
   configuraciones; el rating no filtra (un sondeo publicado es un sondeo a predecir).
4. **Puntuación.** Dos horizontes de prueba, k = 7 y k = 28 días, puntuados por separado: RMSE y MAE sobre los
   sondeos objetivo con el mismo peso cada uno, por partido y en total; y una versión normalizada por el ruido
   mínimo del sondeo (`p (1 − p) / n`) como «skill», informativa. Dos diagnósticos al estilo 538: la
   autocorrelación de orden 1 de los residuos de los sondeos futuros de un mismo corte (señal de retardo) y la
   rugosidad del promedio (cambio medio absoluto de un día al siguiente de la serie ajustada, en puntos por día;
   señal de ruido). La rugosidad se juzga **relativa a la de la configuración actual** sobre los mismos cortes
   (con el suavizado de hoy vale centésimas de punto al día, muy por debajo del ruido de un sondeo, así que un
   umbral absoluto no frenaría nada): una candidata no puede pasar de 1,5 veces la actual.
5. **Parámetros y rejilla**, fijados antes de la primera ejecución. Ancho de banda: piloto fijo ISJ, fijo de 30,
   45, 60 y 90 días, adaptativo con 1, 2 y 3 iteraciones, y adaptativo con el ancho de frontera limitado a ×1,
   ×1,5 y ×2 del piloto. Grado de la regresión: 0 y 1. Ventana de sobrepublicación `wspan`: 2, 4, 7 y 14 días.
   Desviación del prior de los efectos de casa `tau` (hoy 0,08 del nivel: cuánto se confía en la inclinación
   medida en el ciclo frente al prior): 0,04, 0,08 y 0,16; iteraciones del backfitting 1 y 3. Contracción del
   rebaño `n / (n + c)` con un parámetro propio `herd_prior` (hoy comparte `n_prior` = 5 con la muestra efectiva;
   se separa, con el mismo valor por defecto): c = 2, 5 y 10. Prolongación `max_fc`: 0, 3 y 10. No se afinan: el
   exponente del peso por muestra (raíz, como en 538), el tope de la muestra `ol_max` (la winsorización posterior
   corta en unos 4.800 y lo haría casi inerte), `prior_events` y el rating (ver No objetivos). Los pesos que hoy
   se precalculan en la base (`weight_over`, `weight_sample`) se recalculan en memoria por configuración con las
   funciones del Computer y se inyectan en el Forecaster, como ya se hace con `weight_ratings`.
6. **Búsqueda.** Descenso por coordenadas: un parámetro cada vez sobre sus valores de la rejilla, partiendo de los
   valores por defecto actuales; se elige por RMSE a k = 7 y se exige que no empeore k = 28 más de un 2 % ni pase
   de 1,5 veces la rugosidad actual. **Fase 1, el suavizado** (ancho, tope de frontera, grado, `max_fc`,
   `wspan`): dos vueltas con efectos de casa, muestra efectiva y rebaño **desactivados**, porque el rebaño es un
   leave-one-out de unos 30 s por ajuste y el backfitting 7-25 s más, y sin ellos un corte son segundos. **Fase
   2, las contracciones** (`tau`, iteraciones, `herd_prior`): una vuelta con todo activado y el suavizado ya
   fijado, con cortes cada 14 días y sólo los ciclos desde 2015, repartida por ciclos en ECS. La comprobación
   final de `best` frente a la configuración actual, en los ciclos de comprobación, se hace **con todo activado**:
   la cifra que decide es la del estimador vivo. Son unas 25 configuraciones en la fase 1 y 8 en la 2, no un
   producto cartesiano.
7. **Ciclos de comprobación.** La búsqueda usa las generales hasta 2019-11 y las autonómicas anteriores a 2020. La
   configuración elegida se comprueba en 2023, en el ciclo en curso de 2026 y en las autonómicas desde 2020, y se
   publican los dos resultados.
8. **Aceptación para cambiar los valores por defecto** (los dos a la vez): (i) la configuración mejora el RMSE a
   k = 7 en los ciclos de comprobación; (ii) el backtest de elecciones de `es` (28 casos) no empeora MAE de voto,
   CRPS de escaños ni Brier de mayorías más de un 2 % ni baja la cobertura del 95 %, con los casos mejor / peor
   contados, como en M12 y M13. Si no se cumple, la configuración queda como opción documentada (`config` del job
   y del Simulator) y se vuelve a juzgar con el 29-N como sexto caso.
9. **Infraestructura**, gemela del backtest: `mtpy/lib/tuning.py` con funciones puras (`cycle_cuts`,
   `predict_polls`, `score_polls`, `diagnostics`) y de orquestación (`run_cut`, `run_tuning`, `tuning_scope`);
   job `tune` (`SequenceJob`, ámbitos aislados, parámetros en JSON: `scopes`, `cycles`, `cut_days`, `horizons`,
   `configs` o `grid`, `out`); `backtest/run_tuning.py` como envoltorio; salida en `backtest/tuning/{scope}/`
   con `cuts.csv` (un corte por fila: sondeos usados, segundos), `polls.csv` (un sondeo futuro y partido por fila:
   predicción, cifra, residuo, configuración), `scores.csv` (por configuración, ámbito y k) y `run.json`. El
   lanzador de ECS no cambia.
10. **Coste.** Antes de la rejilla, una ejecución de una sola configuración en `es` para medir segundos por corte y
    extrapolar. Estimación previa: un corte sin efectos de casa, muestra efectiva ni rebaño son segundos (fase 1:
    unos 700 cortes × 25 configuraciones ≈ 15 h de CPU en el portátil, menos con la caché de abajo); con todo
    activado un corte son unos 35-55 s (rebaño ≈ 30 s, backfitting 7-25 s), así que la fase 2 (8 configuraciones
    × unos 350 cortes) son unas 40 h de CPU y va a ECS repartida por ciclos. La serie de sondeos de un corte no
    depende de los parámetros de regresión: se construye una vez por corte y se reutiliza entre configuraciones
    dentro de una ejecución. El job parte por ámbitos y, dentro de un ámbito, por ciclos (`chunks`).

### Parte B

11. **Diagnóstico primero.** El backtest exporta los cuantiles de 5 en 5 (`q05`…`q95`) de las cuotas simuladas
    (nowcast y `_h`) y de los escaños, además de los tres intervalos de hoy, y escribe `calibration.csv` con el
    PIT de cada partido y caso (la fracción de simulaciones por debajo del resultado oficial). Una función
    `summarize_calibration` agrupa los PIT en diez cajas por horizonte en **tres poblaciones separadas**: las
    generales (partidos principales; 20-30 celdas por horizonte: las que importan el 29-N), las autonómicas
    (partidos principales; cientos de celdas, con los estimadores prestados de `es`, así que informan del mismo
    modelo de error) y las celdas provinciales (cientos por caso; informan de M9, no del voto nacional). Cada
    elección pesa lo mismo dentro de su población (cada celda, 1 / celdas de su elección), para que una elección
    con muchos partidos no mande. Da también la fiabilidad de las probabilidades de mayoría de `blocks.csv`. Con
    500 simulaciones el PIT tiene una resolución de 0,002, de sobra. **El juez de los pasos 12-15** son las
    generales y las autonómicas al nivel de partido, miradas por separado y sin que una contradiga a la otra; las
    provincias se publican como diagnóstico de M9 y no deciden.
12. **El error terminal del promedio.** Tabla nueva `elections.average_errors` (gemela de `drift`), rellenada por
    `Computer.compute_average_errors(save=True)` desde el job `compute`: para cada elección `featured` del ámbito
    y cada horizonte del backtest, el error del promedio nowcast (modelo menos resultado) de cada partido
    principal, con su nivel y con el error estándar `s` de la regresión local en el ancla (`fc_stat`), obtenido
    con un ajuste sin simulación (sólo `fit_forecast`) **con la configuración viva del Simulator** (efectos de
    casa, muestra efectiva y rebaño activados, la del backtest; el Forecaster suelto los trae apagados). En `es`
    entran todas las generales `featured` con sondeos (desde 1982: unas 14, 40-60 filas a 6 días para una elección
    reciente, no sólo las cinco del backtest), y cada ámbito autonómico calcula las suyas. Estimador
    `Computer.get_average_error_estimator`, de la forma de `get_error_estimator` (error absoluto sobre nivel y
    regional), ajustado sobre las filas del horizonte más corto de las elecciones **anteriores** a la evaluada
    **del ámbito estimador** (`est_scope`: `es` para todos salvo con `min_events`, la regla de M11; las
    autonómicas no entran en el estimador de `es`), y sobre el error **neto de `s`** (`e_net² = max(e² − s², 0)`),
    para que `sqrt(s² + e² + d²)` no cuente `s` dos veces (es un 3-8 % de varianza a 6-14 días, justo donde la
    cobertura ya es buena). El Simulator gana `error_model='average' | 'polls'`: con `'average'`, `e` es ese
    estimador; `'polls'` es el de hoy. Por defecto `'polls'` hasta que el criterio de aceptación lo cambie.
13. **El crecimiento con el horizonte.** De la misma tabla, por horizonte: la varianza del error del promedio a
    `d` días menos la terminal, comparada con `k · d · p²`, da un factor de escala de la deriva `drift_scale`,
    ajustado con las elecciones anteriores a la evaluada (fuera de muestra). `DriftEstimator` lo aplica como
    multiplicador de la varianza; `meta.csv` lo registra. Si el factor sale cerca de 1, la deriva actual basta y se
    documenta.
14. **Las colas.** `Simulator(tail_dof=None)`: con un entero, los grados de libertad de la t del sorteo son fijos
    para todos los partidos (con `None`, los `nobs − 2` de hoy). Candidatos 3, 5, 8, 15 y `None`, elegidos por el
    log-score y la uniformidad del PIT del backtest completo, dejando fuera cada elección por turno. Con diferencias
    dentro del ruido, 5 (el precedente de 538), y se dice.
15. **La correlación entre partidos.** Con la escala y las colas nuevas se reevalúa `composition='auto'` (M7) con
    el mismo criterio de entonces: entra si no empeora escaños ni mayorías a 6-14 días.
16. **Web.** Una tarjeta nueva en `/metodo/backtest`, «Fiabilidad»: el histograma del PIT por grupo de horizontes
    (6-14, 30-60, 90-180) juntando ámbitos, y el diagrama de fiabilidad de las probabilidades de mayoría; «Cómo
    leerlo» en dos párrafos; enlace al CSV. Nada más cambia en la web.
17. **Orden de B.** 11 (diagnóstico, sin tocar el modelo) → 12 (escala) → 13 (horizonte) → 14 (colas) → 15
    (correlación), cada paso con su backtest completo (`es` y autonómicas) y su tabla antes / después, como los
    métodos anteriores. El criterio de aceptación de cada paso: coberturas y PIT más cerca del nominal en los
    horizontes donde faltaban **sin** empeorar donde ya cubrían, y log-score y CRPS no peores más allá del ruido.

### Comunes

18. **Documentación**: `docs/analisis-2026-09/metodo-14-sondeos-futuros.md` y `metodo-15-calibracion.md`, con el
    formato de los métodos anteriores (problema, método, resultados, backtest antes / después, decisión con fecha,
    salvedades); `backtest/README.md` (tablas nuevas, `calibration.csv`, `tuning/`); `deploy/README.md` (job
    `tune`, parámetro `error_model` y `tail_dof` del backtest).
19. **Guardas de regresión**: con `error_model='polls'`, `tail_dof=None`, `drift_scale` = 1 y la configuración
    actual del promedio, el backtest reproduce `backtest/results/` exactamente (mismas tablas, misma semilla);
    `tests/integration/test_backtest.py` lo comprueba en un caso.

## Calendario frente al 29-N

| Semana | Parte A | Parte B |
|---|---|---|
| 2-8 nov | Módulo, job y tests; medida de coste; fase 1 (suavizado) en los ciclos de búsqueda | Cuantiles y PIT en el backtest; backtest completo actual; tarjeta de fiabilidad |
| 9-13 nov | Fase 2 (contracciones) en ECS; comprobación con todo activado; backtest de elecciones con la candidata; **decisión A8 el 13-11 como tarde** | Tabla de errores del promedio sobre el promedio actual; estimador terminal; backtest |
| 16-20 nov | Si A8 se cumplió: los defaults nuevos entran y B recalcula su tabla y sus backtests sobre el promedio nuevo (unos 50 min la tabla, 1-2 h los backtests); si no, A espera al 29-N | Horizonte y colas; decisión (criterio 17) el 20-11 |
| 21-29 nov | Modelo vivo congelado; documentación | Igual |
| Después del 29-N | La elección entra como sexto caso; se reevalúa todo; se abre A2 | Correlación (15) si no cupo |

Regla de encaje entre A y B (decisión mía tras la revisión): B se calibra sobre el promedio con el que vaya a
publicar el modelo. Si A cambia los valores por defecto, cambia antes del 13-11 y B rehace su tabla y sus
backtests en la semana del 16-20; si A no está decidida el 13-11, el modelo vivo sigue con el promedio actual, B se
calibra sobre él y A entra después del 29-N. Así nunca sale a la web un estimador de error ajustado a un promedio
distinto del publicado. La decisión de cambiar el modelo vivo antes del 29-N es de Luis; el criterio y las fechas
límite son los de arriba.

## No objetivos

- **Predecir el signo del error compartido** del sector: medido y descartado en M6 (cambia de signo entre
  elecciones; forzado, añade ruido). Vive en la incertidumbre, no en el punto.
- **El modelo de espacio de estados** (A2): después del 29-N, con el arnés de A1 como juez.
- **Afinar el rating con sondeos futuros**: mediría si las casas bien valoradas están cerca del consenso futuro,
  no de la verdad. Su peso en el promedio se queda como está.
- **El indicador «frente a las encuestadoras» y el techo del error** (consenso frente a residuo del modelo) en
  la web: trabajo aparte y pequeño, después.
- **El estimador de escaños, el ruido provincial y la deriva por edad**: no se tocan; reciben la incertidumbre
  del voto que salga de B.
- **Configuraciones por ámbito**: la configuración es una para todos; las autonómicas son datos de la búsqueda,
  no destinos de ajustes propios.

## Anti-objetivos

- Un promedio que zigzaguea o que corre detrás de un sondeo (la guarda de rugosidad de A6).
- Perseguir el PSOE de 2023: ciclos de comprobación reservados y rejilla fijada antes de mirar.
- Cambiar el modelo vivo la última semana, o sin que los dos criterios lo respalden.
- Ensanchar bandas «por si acaso»: sólo donde el PIT lo pida, y estrechar donde sobre.

## Partes

- `mtpy/lib/tuning.py` (nuevo): cortes, predicción, puntuación, diagnósticos, orquestación por ámbito.
- `mtpy/lib/forecaster.py`: inyección de pesos recalculados (`weight_overrides` o equivalente), límite del ancho de
  frontera (`bw_cap`), `herd_prior` separado de `n_prior` en `disp_params` (mismo valor por defecto), lectura del
  promedio en una fecha con el efecto de casa sumado.
- `mtpy/lib/computer.py`: funciones puras de `weight_over` y `weight_sample` reutilizables en memoria;
  `get_average_error_data`, `compute_average_errors`, `get_average_error_estimator`; `DriftEstimator` con
  `drift_scale`.
- `mtpy/models/elections.py`, `mtpy/lib/data.py`: tabla `average_errors` y sus lectura y escritura.
- `mtpy/lib/simulator.py`: `error_model`, `tail_dof`, `drift_scale` en `meta`; cuantiles de 5 en 5 en `shares()`.
- `mtpy/lib/backtest.py`: cuantiles, `calibration.csv`, `summarize_calibration`, parámetros nuevos de `run_case`.
- `mtpy/jobs/Tune.py`, `mtpy/pipelines/tune/`, `backtest/run_tuning.py`, `job.py` (registro del job).
- `mtpy/jobs/Compute.py`: paso `average_errors` del `compute`.
- `mtpy/lib/publish/backtest.py` o equivalente y `mtpy/lib/bundle.py`: parte `calibration` del paquete;
  `docs/web/contrato.md`.
- `mtpy/lib/pages/backtest.py`, `web/templates/metodo_backtest.html`, `web/dist/js/pages/metodo.js` y un gráfico:
  la tarjeta «Fiabilidad».
- Tests: `tests/test_tuning_unit.py` (cortes, predicción con efecto de casa, puntuación, diagnósticos, búsqueda
  por coordenadas sobre datos sintéticos), `test_forecaster_unit.py` (inyección de pesos, límite de frontera),
  `test_computer_unit.py` (estimador del error del promedio, `drift_scale`), `test_simulator_unit.py`
  (`tail_dof`, `error_model`), `test_backtest_unit.py` (PIT, cajas de fiabilidad), `test_jobs_tune.py`,
  `test_pages.py` (tarjeta y redacción); integración (con base): un corte de 2023 y la guarda de regresión (19).
- Documentación de 18.

## Comprobación

- Suite unitaria completa en verde sin base de datos.
- Con base: la guarda de regresión (19) reproduce `backtest/results/`; un corte de 2023 a 30 días predice los
  sondeos de la semana siguiente con un RMSE del orden del ruido de un sondeo más la dispersión entre casas (2-3
  puntos); `scores.csv` de la configuración actual coincide entre dos ejecuciones.
- La tarjeta de fiabilidad renderizada con Chrome sin cabeza sobre el paquete de pruebas, sin errores de consola y
  con `assert_public_wording`.
- Cada decisión de A8 y de 17 con su tabla antes / después en el documento del método correspondiente.

## Lo que queda al constructor

- Los valores exactos de la rejilla dentro de lo fijado en A5 y el orden de los parámetros en el descenso.
- Cómo se inyectan los pesos recalculados (parámetro del Forecaster o series en memoria) y cómo se cachea la serie
  por corte.
- La forma del estimador del error del promedio (lineal en el nivel como `v2err`, o proporcional) y si
  `drift_scale` multiplica `k` o añade un término: lo que mejor ajuste la tabla, documentado.
- Formatos de las tablas, nombres de columnas y del gráfico de la tarjeta (echarts, como el resto).
- Cómo se reparte el trabajo entre el portátil y ECS según la medida de coste de A10.

## Decisiones tras la revisión del constructor (2026-11-01)

Un constructor independiente leyó el borrador con el código delante y dejó nueve preguntas. Luis había pedido
avanzar sin más preguntas, así que las resuelvo yo; cada una está ya incorporada arriba y aquí queda la razón.
Las dos primeras son las que más conviene que Luis mire.

1. **Sobre qué población se juzga la calibración (B11, criterio 17).** Tres poblaciones separadas (generales,
   autonómicas, provincias), cada elección con el mismo peso dentro de la suya; deciden las dos primeras al nivel de
   partido, las provincias sólo informan de M9. Juntarlo todo dejaría el voto nacional, que es lo que importa el
   29-N, ahogado por cientos de celdas provinciales.
2. **Encaje de A y B en el calendario (B12, intención).** B se calibra sobre el promedio que vaya a publicar: si A
   cambia los defaults, lo hace el 13-11 como tarde y B rehace tabla y backtests en la semana del 16-20; si no, A
   espera al 29-N. Y `compute_average_errors` usa la configuración viva del Simulator, no la del Forecaster suelto.
3. **Qué elecciones alimentan el estimador del error del promedio (B12).** Las del ámbito estimador, según la
   regla de M11 (las autonómicas no entran en el de `es`); todas las generales `featured` desde 1982, no sólo las
   del backtest, para tener 40-60 filas. Agrupar autonómicas con un peso de ámbito queda como experimento posterior.
4. **Fase 1 de la búsqueda con muestra efectiva y rebaño (A6).** Apagados en la fase 1 (el rebaño cuesta 30 s por
   ajuste y haría inviable el calendario), activados en la fase 2 y en la comprobación final, que es la cifra que
   decide. El coste de A10 se corrige con esa cuenta.
5. **Sondeos objetivo (A3).** Los que entrarían en el promedio y además con `weight_over` > 0: fuera las oleadas
   solapadas de los trackings, que dominarían k = 7 y la autocorrelación; el conjunto no depende de `wspan`.
6. **Guarda de rugosidad (A4, A6).** Relativa a la configuración actual (≤ 1,5 ×): con el suavizado de hoy la
   rugosidad son centésimas de punto al día, y un umbral «el ruido de un sondeo» no frenaría nunca.
7. **Doble cuenta de `s` en `e` (B12).** La tabla guarda `s` y el estimador se ajusta sobre el error neto,
   `e_net² = max(e² − s², 0)`; la fórmula de la web no cambia.
8. **Qué contracción se afina (A5).** `tau` (la desviación del prior, lo que decide cuánto se cree la inclinación
   medida) y no `prior_events`; y `herd_prior` separado de `n_prior` para no mover la muestra efectiva al mover
   el rebaño, con el mismo valor por defecto (la guarda de regresión lo exige).
9. **`ol_max` frente a la winsorización (A5).** Fuera de la rejilla: la winsorización posterior corta en unos
   4.800 y el tope sería inerte. Con ello la función en memoria puede ser exactamente la de la base.
