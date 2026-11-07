# Rating: la calidad previa de una casa sin elecciones no supera la mediana (2026-11-07)

## Qué y por qué

Hoy el rating de una encuestadora es `R = (W · R_aj + k · Q) / (W + k)`, con `W` sus elecciones ponderadas,
`R_aj` lo medido, `Q` su calidad previa y `k = (σ/τ)² = 1,96`. Una casa sin elecciones tiene `W = 0` y su rating
es su calidad previa entera: Opina 360, con 64 puntos de papeleo y cero elecciones, pesaba 2,05 en el promedio,
lo mismo que Sigma Dos con 17 elecciones. Luis quiere evitar que una casa nueva, en particular la de un actor
político, entre con más peso que una veterana sólo por publicitar sus métodos. La rúbrica no predice el acierto
(entre las 20 casas con `W ≥ 3`, pendiente 0,14 y correlación 0,12 del rating medido sobre la calidad), así que
no hay razón para creerse los 64 puntos antes de verla sondear una elección.

**Regla acordada (B2).** La calidad previa puede quitar peso desde el primer sondeo, pero sólo lo da cuando hay
elecciones que la respalden: una casa sin elecciones pesa como la casa típica de la tabla (la mediana de los
ratings), haga lo que haga con su papeleo.

## Objetivos, no objetivos

- Objetivo: ninguna casa sin elecciones pesa más que la casa mediana de la tabla; la mala calidad (independencia 0,
  sin ficha) sigue restando desde el primer día; las casas con historial no cambian.
- No objetivo: calibrar nada contra el backtest (decisión de robustez frente al juego, no de ajuste; nunca se
  re-propone calibrar el rating por backtest). No se tocan `k`, `σ`, `τ`, los efectos de casa (M6) ni la rúbrica.
- No objetivo: el peso de reserva del `Forecaster` para una casa que aún no tiene fila de rating (su calidad
  normalizada por la media de calidades, `forecaster.py:600-608`). Queda fuera, anotado abajo.

## Alternativas consideradas

| | Regla | Casa nueva (Q 48) | Pega |
|---|---|---|---|
| Hoy | `R = Q` sin elecciones | 1,63 | 64 puntos de papeleo = 17 elecciones de Sigma Dos |
| B | Tope en la media de los ratings (27,2) | 1,06 | La media baja con cada casa diminuta que entra en el catálogo; «peso 1» hay que explicarlo |
| **B2 (elegida)** | **Tope en la mediana de los ratings (19,5)** | **0,81** | El listón es un dato de la tabla y se mueve si se evalúan las 60 casas que siguen en 5 |
| C | Calidad contraída hacia la media con la pendiente medida (0,14) | 0,99 | Diluye la calidad de todas (CEMOP 2,14 → 1,53): más invasiva |
| D | Peso × `W/(W + k)` («casi nulo sin historial», otros agregadores) | 0,00 | Castiga a las autonómicas honradas con pocas elecciones (CEMOP 1,02, Sondaxe 1,31) y contrae dos veces |

B y B2 son indistinguibles a partir de la segunda elección; sólo difieren en la casa sin elecciones, que es donde
está el riesgo. Luis eligió B2 el 2026-11-07.

## Diseño

**Fórmula.** En `Computer.pollster_ratings` (`mtpy/lib/computer.py`, hoy ll. 2326-2348), dos pasadas:

1. La de hoy: `R0 = (W · R_aj + k · Q) / (W + k)` para todas las casas con rating (las de `drop_mtypes` siguen
   fuera, NaN).
2. El listón `c = mediana(R0)` (NaN fuera) y la calidad efectiva
   `Q_ef = min(Q, c) + max(Q − c, 0) · W / (W + k)`: el tramo de la calidad por debajo del listón entra entero; el
   tramo por encima, en la proporción de elecciones que la casa ya tiene. Rating final
   `R = (W · R_aj + k · Q_ef) / (W + k)`, y `weight_rating = log1p(R) / log1p(media de R)` como hoy, con la media
   nueva.

Consecuencias comprobadas sobre la tabla viva de `es` (2026-11-07): casa nueva con cualquier calidad ≥ 19,5 →
rating 19,5, peso 0,81; con calidad 8 o 16 no cambia (0,34 / 0,68); SW Demoscopia (Q 24, W 2,8) 1,05 → 1,12;
CEMOP (Q 56, W 1,8) 2,14 → 2,06; GAD3 2,26 → 2,46 y Sigma Dos 2,12 → 2,30, por el normalizador. Opina 360 con
una elección medida como una casa media pasa a 1,40 (hoy 1,65) y con tres, 1,68 (hoy 1,67).

**Parámetro.** `Computer(quality_cap='median')`, con valores `'median'`, `'mean'` y `None` (desactivado:
reproduce el rating de hoy). Atributo `self.quality_cap`. No se guarda ninguna columna nueva en
`pollsters_ratings` (nada de `ALTER TABLE`): `rating_adj` sigue siendo lo medido (o la calidad cruda sin
evidencia) y `quality` la calidad previa de la rúbrica; el listón de cada cálculo se imprime con `verbose > 0`.

**Textos públicos.**
- `web/templates/metodo_rating.html`, tarjeta «La evidencia se cuenta por elecciones»: la frase «Una encuestadora
  sin sondeos evaluables tiene como rating su calidad previa» pasa a decir que tiene como rating su calidad previa
  **o la mediana de los ratings de la tabla si la suya es mayor**: la calidad previa puede quitar peso desde el
  primer sondeo, pero sólo lo da cuando hay elecciones que la respalden. Fórmula de `Q_ef` en la nota técnica
  `evidencia` (sólo para expertos), no en el cuerpo. La caption de la tabla del 20 de octubre no cambia (son datos
  de ese día).
- `web/templates/casas.html` l. 58 (intro del ranking): añadir «y que, sin elecciones, no puede superar la mediana
  de la tabla».
- `mtpy/lib/pages/casas.py`, tooltip `quality`: «pesa en el rating mientras ha sondeado pocas elecciones; sin
  ninguna, no cuenta por encima de la mediana de los ratings».
- Reglas de redacción: «sólo» con tilde, encuestadora/casa, nunca empresa (`assert_public_wording`).

**Documentación.** `docs/analisis-2026-09/metodo-17-tope-calidad-previa.md` (formato de los métodos 15 y 16:
Cambios, Tests, Problema, Método, Resultados con la tabla de arriba) y corrección de la fórmula desfasada de
`computer-ratings.md` l. 151 (`min_polls = 3` → `k = (σ/τ)²` y el tope).

**Tests** (`tests/test_computer_unit.py`, sobre `bare_computer`, que gana `quality_cap`):
- sin elecciones y calidad por encima de la mediana → rating = mediana de la primera pasada;
- sin elecciones y calidad por debajo → rating = calidad (no cambia);
- con elecciones, la calidad sobrante se acredita en `W/(W + k)` (fórmula exacta);
- `quality_cap=None` reproduce el rating de hoy (`test_rating_follows_the_grouped_formula` se mantiene con
  `None` o se adapta con el listón);
- `'mean'` usa la media; las casas de `drop_mtypes` no entran en el listón.
`tests/test_pages.py`: la frase nueva de /metodo/rating y la redacción pública.

**Comprobación.** Ratings de `all` recalculados con la regla y backtest de `es` comparado con la referencia
(criterio 17 caso a caso), como en la fase C2a. Afecta al 2,9 % del peso del ciclo de `es` (12 sondeos de 9 casas
sin elecciones): se espera plano; sirve para comprobar que no empeora, no para decidir.

**Despliegue** (Luis): ratings de todos los ámbitos, imagen y update. En vivo, Ateneo del Dato pasaría de 1,40 a
0,81 y Opina 360 de 1,63 a 0,81.

## Fuera de alcance, anotado

El peso de reserva del `Forecaster` para una casa con sondeos pero sin fila de rating (aparecida después del
último `ratings`) sigue siendo su calidad normalizada por la media de calidades, sin tope. En la práctica una casa
nueva está sin evaluar (calidad 5, peso ≈ 0,3) hasta que Luis la evalúa, y `catalogues` y `ratings` se corren
juntos; si algún día se evalúa una casa alta sin recalcular ratings, ese hueco existe. Se decide aparte.

## Decisiones mías (a revisar al leer)

1. El listón se calcula sobre los ratings de la primera pasada de **ese** cálculo (cada elección valorada tiene
   su propia tabla y su propia mediana), no sobre un valor fijo: así el backtest y el histórico de ratings usan
   el listón de su época.
2. `'median'` por defecto en el constructor, sin parámetro en `params.json`: es una regla del método, no un
   ajuste por ámbito.
3. Sin columnas nuevas en la base; el listón sólo se imprime.
4. La fórmula de `Q_ef` va en la nota técnica, y en el cuerpo una frase en palabras.
