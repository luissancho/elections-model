# Banda del rating en el peso del promedio: el exponente `rating_exponent` (fase C2a)

Fecha: 2026-11-05. Estado: CONSTRUIDA en dev el 2026-11-06 (aprobada por Luis el 2026-11-05; enfoque y diseño acordados
en conversación; corre en el portátil). Resultados al final y en `docs/analisis-2026-09/metodo-16-rating.md`, sección
«¿Y con otra banda? (fase C2a, 06-11-2026)». Sigue a la fase C (`2026-11-04-rating-validacion-design.md`, método 16).

## Intención

La prueba 1a del método 16 midió que quitar el rating del peso del promedio (rating plano) empeora el voto y los escaños
del modelo vivo un 0,8 / 1,0 % en las generales y un 0,4 / 0,7 % en las autonómicas, dentro del ±2 % del criterio 17; las
probabilidades de mayoría van al revés en las generales (−8,9 % por 2023) y el promedio simple de la prueba 1b es neutro
(+0,5 % peor con rating, 37 / 34). Luis lee esa señal pequeña como «el rating mejora, no mucho» y quiere saber si
**ensanchar la banda entre la peor y la mejor encuestadora** mejora más el acierto frente al resultado. Esta fase mide esa
curva, en las dos direcciones, y sólo cambia el valor por defecto si un punto de la curva pasa el criterio 17 con holgura.

## Qué hay hoy

- El peso de un sondeo es `weight_over · weight_sample · weight_rating`; `weight_rating = log1p(R) / log1p(mean R)` con R
  en 0-1 (`Computer.pollster_ratings`), casi lineal en R; las casas sin fila de rating llevan `log1p(q) / log1p(mean q)` por
  su calidad previa (`Forecaster.build_series`). En el ciclo nacional en curso (05-11-2026, 24 casas, 337 sondeos) la banda
  va de 0,58 (CIS) a 2,25 (GAD3), 3,9×; el CIS pasa del 17,3 % del peso sin rating al 6,5 % con él, Sigma Dos del 12,6 al 17,3 %.
- `Forecaster(weight_ratings='flat')` y `backtest --flat-ratings` (fase C) ponen el rating a 1: es la banda de anchura 0.
- Las corridas emparejadas de la fase C (`backtest/results_rated`, `backtest/results_flat`, ignoradas por git; `es` + es-md,
  es-an, es-ct, es-vc, es-ga; `n_sim` 1.000, semilla 42) y su tabla de comparación, hecha con un script de scratch no
  commiteado; el procedimiento está en el método 16 («¿Mejora el promedio?»).
- `rating_check.simple_average_errors(polls)` y `summarize_simple_average` (prueba 1b) sobre el frame `polls` de
  `Computer.rating_check_data('all')` (≈ 4 min contra la base), con la columna `weight_rating` por sondeo.

## Decisiones

1. **Una palanca**: `rating_exponent` (γ, float ≥ 0, 1.0 por defecto) de `Forecaster.__init__`, aplicada en `build_series`
   al `weight_rating` de las filas de sondeos **después** del relleno por calidad (`weight_rating ** γ`); los resultados
   oficiales no se tocan; γ = 1 deja el modelo exactamente como está; γ = 0 equivale a `weight_ratings='flat'` (que se
   conserva); valores negativos o no finitos → `ValueError`. γ no cambia la forma del mapa (sigue siendo casi lineal en R),
   sólo estira o encoge la banda: con la banda de hoy, γ 0,5 → 2×, 1 → 3,9×, 1,5 → 7,7×, 2 → 15×, 3 → 59×.
2. **Enhebrado como `flat_ratings`**: `Simulator(rating_exponent=1.0)` lo pasa al Forecaster; `backtest.run_case`,
   `run_backtest`, `backtest_scope(rating_exponent=1.0)`, el pipeline y el job `backtest` (parámetro JSON), el CLI
   `--rating-exponent`; registrado en `run.json` y en `meta.csv` de cada caso; documentado en `backtest/README.md`,
   `deploy/README.md` (fila de la tabla) y `README.md`. El modelo vivo no lo recibe: el default de la firma es lo que rige.
3. **Rejilla fijada antes de mirar**: γ ∈ {0, 0,5, 1, 1,5, 2, 3}. Simétrica: si el óptimo estuviera por debajo de 1, una
   rejilla sólo hacia arriba no lo vería.
4. **Jueces, en este orden**:
   - **Prueba 1b (cribado, minutos)**: `rating_check.simple_average_curve(polls, exponents)` → una fila por γ y población
     con `mae_rated`, `diff` frente a γ = 1, IC por bootstrap de elecciones y mejor / peor; sobre el `polls` de
     `rating_check_data('all')` elevado a γ. 71 elecciones.
   - **Backtest emparejado (el que decide)**: `es` (28 casos) y los cinco autonómicos (66 casos) por γ, en carpetas
     `backtest/results_exp{γ}` (ignoradas por git), con los parámetros por defecto (`n_sim` 1.000, semilla 42). γ = 1 se
     **rehace** como referencia (misma base y mismo código que los demás); γ = 0 se rehace salvo que la referencia nueva
     reproduzca `results_rated` caso a caso (entonces la base no cambió y `results_flat` vale). Criterio 17 frente a γ = 1:
     MAE de voto, CRPS de escaños, Brier de mayorías (±2 %), cobertura del 95 % no menor, casos mejor / peor (empates aparte),
     por población y en total; la curva de las cuatro métricas frente a γ.
   - El arnés de sondeos futuros (método 14) **no** juzga: con una banda más ancha el promedio se acerca a las casas mejor
     valoradas y predecir sus sondeos futuros lo premiaría por construcción. Queda como diagnóstico posterior, fuera de
     esta fase.
5. **Script de comparación commiteado**: `backtest/compare_results.py A B [--scopes …]` empareja los `metrics.csv` de dos
   carpetas de resultados por (ámbito, `event_date`, `horizon`) y escribe la tabla por población (como la del método 16:
   casos, media A, media B, diferencia y %, mejor / peor / empates para las cuatro métricas) en markdown; funciones puras en
   `mtpy/lib/backtest.py` (`pair_results`, `compare_paired`) con tests sobre CSV diminutos. Sustituye al script de scratch
   de la fase C.
6. **Regla para cambiar el default**: sólo si un γ mejora MAE de voto y CRPS de escaños frente a γ = 1 más del 2 % en las
   dos poblaciones, con más casos mejor que peor, sin empeorar el Brier más del 2 % ni bajar la cobertura; commit aparte,
   antes del 17-11 o después del 29-N como sexto caso. Si ningún γ lo cumple (lo esperable con señales del 0,5 %), nada
   cambia y la curva se publica en el método 16 como sección «¿Y con otra banda?» (16b), con la lectura declarada de
   antemano: una curva plana dentro del ±2 % confirma que la anchura de la banda es cosmética.
7. **Coste y calendario**: cada γ son ≈ 2 h 20 de portátil (65 min `es`, 75 min los cinco); cinco corridas encadenadas en
   segundo plano ≈ 12 h (noche del 05 al 06-11); la regla de la fase C: una corrida sólo vale si su log no tiene
   `OperationalError`, 28 casos en `es` y 66 en los autonómicos.

## No objetivos

- Cambiar la forma del mapa rating → peso (otra familia, p. ej. ponderación por precisión del modelo jerárquico, `log`
  sobre 0-100): una fase aparte si la curva de γ lo sugiere.
- Tocar el rating, la calidad previa, la parte `rating-check` publicada o la tarjeta de /metodo/rating.
- Afinar γ con sondeos futuros; ECS; cambiar `n_sim` o la semilla de las corridas.

## Anti-objetivos

- Elegir γ por un solo horizonte, una sola elección (2023) o una sola métrica.
- Cambiar el modelo vivo sin la regla 6 o en la última semana antes de la veda (23-11).
- Comparar contra las carpetas de la fase C sin comprobar que la base no cambió.

## Comprobación

- Tests: Forecaster (γ = 2 eleva al cuadrado los pesos de sondeos, resultados intactos; γ = 0 ≡ `'flat'`; γ negativo →
  `ValueError`), backtest/pipeline/job/CLI (γ en `run.json` y `meta.csv`; default 1,0), `pair_results`/`compare_paired`
  (empates, casos sin métrica, un caso que falta en una carpeta), `simple_average_curve` (γ = 1 reproduce
  `summarize_simple_average`; γ = 0 reproduce `mae_flat`).
- Las curvas de 1b y del backtest en `docs/analisis-2026-09/metodo-16-rating.md` (sección nueva) con los comandos; la spec
  cierra con «Resultados» y la decisión.

## Lo que queda al constructor

- Dónde cae el exponente respecto a `weight_overrides` (después del relleno y antes del producto `weight`; los overrides
  de `weight_rating` también se elevan).
- La forma exacta de la tabla markdown de `compare_results.py` (la del método 16 como modelo).
- El lanzador encadenado de las cinco corridas (`nohup sh -c '… && …'`, log en el espacio del plan).

## Resultados (2026-11-06)

Detalle, tablas completas de `compare_results.py` y salvedades en `docs/analisis-2026-09/metodo-16-rating.md`, sección
«¿Y con otra banda? (fase C2a, 06-11-2026)». Código: c0370a4 (`rating_exponent` en `Forecaster` y `Simulator`), 81306fa
(backtest, job y CLI), 0c5276a (`compare_results.py`) y a6aaf98 (`simple_average_curve`).

**Cribado 1b (promedio simple de la última campaña, 71 elecciones, 2.000 réplicas por bootstrap de elecciones, semilla
0).** MAE − MAE de γ = 1, en puntos, todas las elecciones; mejor / peor que γ = 1.

| γ | MAE (puntos) | Diferencia frente a γ = 1 (IC 95 %) | Mejor / peor |
|---|---|---|---|
| 0 | 1,9292 | −0,0106 (−0,0478 a +0,0197) | 34 / 37 |
| 0,5 | 1,9369 | −0,0029 (−0,0175 a +0,0085) | 31 / 40 |
| 1 | 1,9398 | referencia | |
| 1,5 | 1,9401 | +0,0002 (−0,0077 a +0,0090) | 38 / 33 |
| 2 | 1,9390 | −0,0008 (−0,0159 a +0,0142) | 40 / 31 |
| 3 | 1,9340 | −0,0058 (−0,0299 a +0,0189) | 38 / 33 |

Curva plana: en el conjunto ninguna diferencia pasa de 0,011 puntos (0,012 en las autonómicas a γ = 0) y todos los
intervalos, también los de las generales y las autonómicas, cruzan el 0.

**Backtest emparejado frente a γ = 1** (`es` 28 casos + `es-md`, `es-an`, `es-ct`, `es-vc`, `es-ga` 66; `n_sim` 1.000,
semilla 42; del 05-11 a las 14:15 al 06-11 a las 01:50). γ = 1 se rehízo (`backtest/results_exp1`) y reproduce
`results_rated` al diezmilésimo, 94 / 94 empates: la base no cambió y `results_flat` vale como γ = 0. Diferencia relativa
γ − (γ = 1) (positiva: γ yerra más), cobertura en puntos porcentuales; entre paréntesis, casos mejor / peor en el voto.

| γ | Población | Casos | MAE de voto | CRPS de escaños | Brier de mayorías | Cobertura del 95 % |
|---|---|---|---|---|---|---|
| 0 | Generales | 28 | +0,8 % (13 / 15) | +1,0 % | −8,9 % | +0,7 pp |
| 0,5 | Generales | 28 | +0,2 % (12 / 16) | +0,2 % | −4,5 % | +0,0 pp |
| 1,5 | Generales | 28 | +0,8 % (15 / 13) | +0,8 % | +5,9 % | −0,9 pp |
| 2 | Generales | 28 | +1,3 % (14 / 14) | +1,1 % | +8,3 % | −0,2 pp |
| 3 | Generales | 28 | +1,8 % (14 / 14) | +1,4 % | +12,5 % | −0,9 pp |
| 0 | Autonómicas | 66 | +0,4 % (32 / 33) | +0,7 % | +3,3 % | −0,5 pp |
| 0,5 | Autonómicas | 66 | +0,3 % (26 / 40) | +0,4 % | +1,2 % | −0,8 pp |
| 1,5 | Autonómicas | 64 | −0,1 % (34 / 30) | −0,3 % | −0,5 % | −0,6 pp |
| 2 | Autonómicas | 66 | −0,3 % (32 / 34) | −0,6 % | −0,6 % | −0,4 pp |
| 3 | Autonómicas | 66 | −0,8 % (33 / 32) | +2,4 % | −7,1 % | −1,2 pp |
| 0 | Todas | 94 | +0,5 % (45 / 48) | +0,8 % | −3,4 % | −0,1 pp |
| 0,5 | Todas | 94 | +0,3 % (38 / 56) | +0,3 % | −2,0 % | −0,6 pp |
| 1,5 | Todas | 92 | +0,1 % (49 / 43) | +0,3 % | +3,0 % | −0,7 pp |
| 2 | Todas | 94 | +0,1 % (46 / 48) | +0,2 % | +4,3 % | −0,4 pp |
| 3 | Todas | 94 | −0,1 % (47 / 46) | +1,9 % | +3,8 % | −1,1 pp |

γ = 1,5 va sobre 92 casos: `es-an` 2022-06-19 a 90 y 180 días se perdieron por un corte de red con la base (registrados
como «skipped», sin `OperationalError`) y no se pudieron repetir en la misma base, porque los ratings guardados se
recalcularon el 06-11 a las 01:50, tras la cadena; el emparejamiento por caso los deja fuera.

**Decisión (regla 6).** Ningún γ pasa: ninguno mejora el MAE de voto y el CRPS de escaños más de un 2 % en las dos
poblaciones con más casos mejor que peor. Las bandas anchas empeoran las generales (γ = 3: voto +1,8 %, CRPS +1,4 %,
Brier +12,5 %) mientras las autonómicas se mueven dentro del ±1 % en el voto; la cobertura baja en todos los γ > 1 (de
−0,4 a −1,1 puntos sobre todos los casos); la banda estrecha (γ = 0,5) pierde en más casos de los que gana (38 / 56 en
el MAE de voto). Lectura declarada de antemano, cumplida: la curva es plana dentro del ±2 % en el voto (y en los escaños
salvo el CRPS de las autonómicas a γ = 3, +2,4 %, a peor), así que **la anchura de la banda es cosmética**; los recuentos
(γ = 0,5 pierde 56 de 94 casos, γ = 1,5 gana 49 de 92) dicen que el rating ayuda un poco y que alrededor de 1 una banda
acierta lo mismo que otra. El valor por defecto se queda en 1 y el modelo vivo no cambia: no hay commit de default.

**Pasos de Luis que quedan.** Integrar `dev` (el código de la palanca y del script de comparación). No hay nada que publicar ni
desplegar por esta fase.
