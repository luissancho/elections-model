# Método 17: tope de la calidad previa para las casas sin elecciones (07-11-2026)

Cambios:
- `Computer(quality_cap='median')`: estadístico de los ratings de la primera pasada que hace de tope de la calidad previa (`'median'` por defecto, `'mean'`, o `None` para el rating de antes); segunda pasada en `Computer.pollster_ratings` con la calidad efectiva `Q_ef` (`mtpy/lib/computer.py`). Sin columnas nuevas en `pollsters_ratings` ni parámetro en `params.json`.
- Textos: la tarjeta «La evidencia se cuenta por elecciones» y la nota técnica de la mezcla de /metodo/rating (`web/templates/metodo_rating.html`), la intro del ranking de /casas (`web/templates/casas.html`) y el tooltip de «Confianza» (`mtpy/lib/pages/casas.py`).
- Documentación: este método y la fórmula vigente en `computer-ratings.md` (punto 8).

Tests: `tests/test_computer_unit.py` (sección «tope de la calidad previa»: casa sin elecciones por encima de la mediana, calidad baja intacta, tramo sobrante acreditado en `W/(W + k)`, `None` reproduce la mezcla de antes, `'mean'`, valor desconocido, una sola casa con rating, varias casas sin evidencia, casa excluida fuera del listón), `tests/test_pages.py` (frases nuevas de /metodo/rating y /casas, frases viejas ausentes). Spec: `docs/superpowers/specs/2026-11-07-rating-tope-calidad-previa-design.md`; plan: `docs/superpowers/plans/2026-11-07-rating-tope-calidad-previa.md`.

## Problema

El rating de una encuestadora es `R = (W · R_aj + k · Q) / (W + k)`, con `W` sus elecciones ponderadas, `R_aj` lo medido, `Q` su calidad previa y `k = (σ/τ)² = (0,014/0,01)² = 1,96` (método 16 y spec del 2026-10-20). Una casa sin elecciones tiene `W = 0` y su rating es su calidad previa entera. El 2026-11-07, Opina 360 (64 puntos de rúbrica, cero elecciones) pesaba 2,05 en el promedio de las generales, lo mismo que Sigma Dos con 17 elecciones ponderadas (2,11), y por encima de 40dB (2,04) o Celeste Tel (2,03). Nueve casas con rating no habían sondeado ninguna elección; sus 12 sondeos eran el 2,9 % del peso del ciclo de `es` y el 10,7 % del de los últimos 60 días.

La rúbrica no predice el acierto: entre las 20 casas con `W ≥ 3`, la pendiente del rating medido (`rating_adj`) sobre la calidad es 0,14 y la correlación 0,12 (con `W ≥ 5`, −0,13 y −0,05). No hay razón para creerse los 64 puntos antes de ver a la casa sondear una elección, y sí la hay para no hacerlo: una casa de un actor político puede publicitar sus métodos precisamente para pesar en los promedios.

## Método

Dos pasadas en `pollster_ratings`:

1. La mezcla de siempre, `R0 = (W · R_aj + k · Q) / (W + k)`, para las casas con rating (las de `drop_mtypes` siguen fuera, NaN).
2. El listón `m = mediana(R0)` (o la media con `'mean'`; los NaN no cuentan) y la calidad efectiva `Q_ef = min(Q, m) + max(Q − m, 0) · W / (W + k)`: el tramo de la calidad por debajo del listón entra entero; el tramo por encima, en la proporción de elecciones que la casa ya tiene. Rating final `R = (W · R_aj + k · Q_ef) / (W + k)` y `weight_rating = log1p(R) / log1p(media de R)` como antes, con la media nueva.

El listón se calcula en cada cálculo, sobre la tabla de esa elección valorada: el histórico de ratings y el backtest usan la mediana de su época. Se eligió la mediana y no la media porque la media de la tabla baja cada vez que entra una casa diminuta sin evaluar (hoy 121 casas con rating, 61 por debajo de 20 y 32 por debajo de 10, casi todas con calidad 5-8) y la mediana apenas se mueve; y porque «una casa sin elecciones pesa como la casa típica de la tabla hasta que demuestre algo» se explica sola. El listón es un dato de la tabla (19,5 el 2026-11-07) y subirá si se evalúan las casas que siguen en 5.

Lo que la regla dice, en una frase: la calidad previa puede quitar peso desde el primer sondeo, pero sólo lo da cuando hay elecciones que la respalden.

## Alternativas consideradas

| | Regla | Casa nueva (Q 48) | Pega |
|---|---|---|---|
| Antes | `R = Q` sin elecciones | 1,63 | 64 puntos de papeleo valen 17 elecciones de Sigma Dos |
| B | Tope en la media de los ratings (27,2) | 1,06 | La media baja con cada casa diminuta del catálogo; «peso 1» hay que explicarlo |
| **B2 (elegida)** | **Tope en la mediana de los ratings (19,5)** | **0,81** | El listón se mueve si se evalúan las 60 casas que siguen en 5 |
| C | Calidad contraída hacia la media con la pendiente medida (0,14) | 0,99 | Diluye la calidad de todas (CEMOP 2,14 → 1,53) |
| D | Peso × `W/(W + k)` («casi nulo sin historial», otros agregadores) | 0,00 | Castiga a las autonómicas honradas con pocas elecciones (CEMOP 1,02, Sondaxe 1,31) y contrae dos veces |

B y B2 son indistinguibles a partir de la segunda elección; sólo difieren en la casa sin elecciones, que es donde está el riesgo.

## Resultados

Sobre la tabla viva de `es` del 2026-11-07 (ratings para las generales del 29-N, tras poner a Opina 360 en independencia 0 y a SW Demoscopia en campo 1 y transparencia 1):

| Casa | Q | W | Rating antes | Peso antes | Rating con el tope | Peso con el tope |
|---|---|---|---|---|---|---|
| Opina 360 | 48 | 0 | 48,0 | 1,63 | 19,5 | 0,81 |
| Ateneo del Dato | 40 | 0 | 40,0 | 1,40 | 19,5 | 0,81 |
| Winston | 16 | 0 | 16,0 | 0,62 | 16,0 | 0,68 |
| Grupo ANA | 8 | 0 | 8,0 | 0,32 | 8,0 | 0,35 |
| SW Demoscopia | 24 | 2,8 | 28,8 | 1,05 | 28,8 | 1,12 |
| CEMOP | 56 | 1,8 | 67,3 | 2,14 | 65,2 | 2,06 |
| Sondaxe | 48 | 4,3 | 57,8 | 1,89 | 56,9 | 2,00 |
| GAD3 | 56 | 14,5 | 72,2 | 2,26 | 71,8 | 2,46 |
| Sigma Dos | 64 | 17,1 | 66,4 | 2,12 | 66,0 | 2,30 |

Las casas con calidad por debajo del listón no cambian de rating; su peso sube una o dos centésimas porque baja la media que normaliza. Las veteranas recuperan casi toda su calidad (`W/(W + k)` es 0,88 con 15 elecciones) y suben una décima por el mismo normalizador. El peso de las nueve casas sin elecciones en los últimos 60 días de `es` pasa del 10,7 % al 7,7 %.

Cómo crece Opina 360 con cada elección, si midiera como una casa media (desviación 50), peso con la regla de antes → con el tope: 0 elecciones 1,63 → 0,81; 1 elección 1,65 → 1,40; 2 elecciones 1,66 → 1,60; 3 elecciones 1,67 → 1,68; 10 elecciones 1,68 → 1,81.

**Backtest emparejado de `es`:** pendiente (Tarea 4 del plan).

## Fuera de alcance

El peso de reserva del `Forecaster` para una casa con sondeos pero sin fila de rating (aparecida después del último `ratings`) sigue siendo su calidad normalizada por la media de calidades, sin tope (`forecaster.py`, `build_series`). En la práctica una casa nueva está sin evaluar (calidad 5, peso ≈ 0,3) hasta que se la evalúa a mano, y `catalogues` y `ratings` se corren juntos; si algún día se evaluara alta una casa sin recalcular los ratings, ese hueco existe. Se decide aparte.
