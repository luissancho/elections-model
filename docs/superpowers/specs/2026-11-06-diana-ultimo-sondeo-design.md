# Diana del perfil: el último sondeo de cada elección, sin tope de días, y la casa frente a las demás

**Fecha:** 2026-11-06 · **Estado:** aprobado por Luis (camino 1; segundo gráfico pedido en la misma sesión) ·
**Tamaño:** cambio pequeño

## Intención

1. La diana de errores del perfil de una encuestadora (`/casas/{slug}`) debe tener un punto por cada elección
   analizada en la que la casa publicó sondeos, con su último sondeo, aunque ese sondeo sea de más de 42 días
   antes de la votación. Hoy sólo pinta las elecciones con un sondeo entre 6 y 42 días antes, y por eso SW
   Demoscopia sale con 7 elecciones en el ranking y 3 puntos en la diana.
2. Junto a la diana, en escritorio una al lado del otro, el gráfico de puntos de la pregunta 2 del análisis del
   sesgo del CIS («El CIS, frente a la media de las demás casas en la misma elección»), hecho para la casa del
   perfil: cada punto una elección; horizontal, la media del error de las demás casas en el bloque de izquierdas;
   vertical, el error de la casa; la diagonal separa «dio al bloque más voto que las demás» de «menos».

## Cómo funciona hoy

La diana no consulta la base: lee la parte `poll-errors` del análisis de cada ámbito, publicada con
`get_polls_metric('error', bmap='vs', drange=(6, 42), n_last=1)`: el último sondeo de cada casa con fin de campo
entre 6 y 42 días antes de cada elección con resultado. La misma parte alimenta:

- el perfil: la diana, su tabla y la tabla «Frente al resultado» (`last_poll_records`);
- el histórico de `/elecciones`: histograma, diana de todos los sondeos y tabla por semanas (`elecciones_context`);
- la tarjeta «Un ejemplo: el CIS desde 2018» de `/metodo/promedio` (`house_effects_example`), cuyo texto dice
  «entre 6 y 42 días»;
- el estudio «El sesgo del CIS de Tezanos» (`studies.lean.data.read_error_rows`), cuyo texto dice lo mismo.

La página de cada elección no la lee: usa su propia parte `polls`.

## Objetivos, no objetivos y anti-objetivos

- **Objetivo:** que el perfil muestre todas las elecciones sondeadas por la casa, cada una con su último sondeo.
- **No objetivo:** cambiar el rating, el promedio ni la página de cada elección.
- **Anti-objetivo:** que cambie una sola cifra del histórico de `/elecciones`, de la tarjeta del CIS o del estudio
  del sesgo del CIS: esas páginas hablan de la recta final y lo dicen en su texto.
- **Anti-objetivo:** que un sondeo de 2020 parezca una predicción del 23-J: el punto tiene que distinguirse.

## Restricciones

- El límite inferior de 6 días se queda: es la veda electoral (los sondeos de veda, `ctype = 'wban'`, y los de
  pie de urna ya quedan fuera por `drop_ctypes`).
- Para las casas con un sondeo dentro de la ventana no cambia nada: con `n_last = 1` su último sondeo es el mismo.
- La web no cambia hasta que Luis republique el análisis de los 18 ámbitos; `meta.params.drange` pasa a
  `[6, null]` y el contrato (`docs/web/contrato.md`) lo recoge.

## Caminos considerados

1. **Abrir la parte y filtrar donde importa la ventana** (elegido). `POLL_ERRORS_PARAMS['drange']` pasa a
   `(6, None)`. Las filas ya llevan `days`, así que el histórico, la tarjeta del CIS y el estudio del sesgo filtran
   `days <= 42` y no cambian ninguna cifra; el perfil usa todas las filas.
2. **Una parte nueva sólo para el perfil** (`last-polls`). No toca a nadie, pero duplica esquema, contrato,
   publicación y tests para la misma información. Descartado.

## Decisiones

- La ventana de la recta final vive en una sola constante, `POLL_ERRORS_WINDOW = (6, 42)` en `mtpy/lib/bundle.py`
  (el módulo del contrato, que ya importan la publicación, las páginas y el estudio sin ciclos). La publicación
  usa su mínimo como límite inferior de `drange`; las páginas y el estudio, su máximo como tope.
- Filtran por `days <= POLL_ERRORS_WINDOW[1]`: `elecciones_context` (el histórico entero: filtros, histograma,
  diana y semanas), `house_effects_example` y `read_error_rows`. `history_rows` no filtra: es el lector común y el
  perfil lo usa entero.
- En el perfil, cada fila de la diana lleva `late` (`days > POLL_ERRORS_WINDOW[1]`). El gráfico pinta huecos esos
  puntos (`bullseyeOption` admite `hollowOf: row => boolean`: relleno transparente y borde del color de su serie,
  con la misma opacidad por antigüedad); su etiqueta y su tooltip (que ya dice «N días antes») no cambian. La
  tabla ya trae la columna «Días antes».
- Textos del perfil: el subtítulo de la tarjeta y «Cómo leerlo» explican que un sondeo de más de seis semanas va
  hueco. Los textos de `/metodo/promedio` y del estudio del sesgo del CIS siguen siendo ciertos y no se tocan.
- La ventana de la tarjeta del CIS y del estudio no se menciona en el perfil: ahí ya no hay ventana.

### El gráfico «frente a las demás casas»

- Misma tarjeta que la diana (`#bullseye`), en un `.grid2`: a la izquierda la diana, a la derecha el gráfico de
  puntos, cada uno con su `.ctitle` y `.csub`; en móvil se apilan (la regla general de `.grid2`). La tabla de la
  diana sigue debajo. Los filtros de ámbito y años de la tarjeta mandan sobre los dos gráficos y la tabla.
- Datos: `casa_context` añade a cada fila de la diana `others` (la media del error de las demás casas en el bloque
  de izquierdas de la pareja, `table_blocks(blocks)[0]`, en esa misma elección: sus filas de `poll-errors` con
  `days <= POLL_ERRORS_WINDOW[1]`, la recta final, como en el estudio) y `n_others` (cuántas; `others` es `None`
  sin ninguna). El propio punto de la casa es el de la diana (hueco si es `late`). `initial` lleva `left`, el
  nombre de ese bloque.
- Gráfico: `scatterOption` (`charts/scatter.js`), como en el estudio: `diagonal`, ejes fijos y simétricos
  calculados en el cableado (el primer múltiplo de 5 por encima del mayor valor absoluto de ambas coordenadas, 5
  como mínimo, paso 5), color por tipo de elección (generales en el acento, autonómicas en el naranja de la diana),
  etiqueta `evento-ámbito` en todos los puntos con `hideOverlap`, y la misma opción `hollowOf` que la diana para
  los puntos `late`. Eje horizontal «Media de las demás casas, error en {bloque}»; en el tooltip, «demás casas» y
  el nombre de la casa. Sin punto cuando falta `others` o el error de la casa en ese bloque.
- Descarga: el botón de la tarjeta exporta los dos gráficos apilados (`charts` de `wireDownload`), con los mismos
  filtros y el recuento en el subtítulo.
- Textos: subtítulo de la tarjeta, títulos de columna y «Cómo leerlo» explican los dos gráficos y los puntos
  huecos; redacción con las reglas públicas de Luis (`sólo`, encuestadora/casa).

## Lo que queda al constructor

Los tests concretos (publicación, páginas, estudio, opciones de ECharts), la redacción exacta de los textos del
perfil, el alto y la proporción de los dos gráficos en la tarjeta y la nota del contrato. La republicación del
análisis es de Luis.
