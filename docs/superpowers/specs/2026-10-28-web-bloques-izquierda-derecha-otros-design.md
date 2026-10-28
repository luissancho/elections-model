# Web: «Bloques» = Izquierda, Derecha y Otros

Fecha: 2026-10-28. Estado: diseño pendiente de aprobación de Luis.

## Intención

Hoy la vista «Bloques» de la web enseña los cuatro bloques de `bmaps.blocks` (Derecha, Izquierda, Regionalista,
Separatista). El modelo, en cambio, mide los bloques con `bmaps.vs`, que desde el relleno de `params.json` del
2026-10-28 lleva el bloque completo (en 2026: Derecha = PP, VOX, SALF; Izquierda = PSOE, SUMAR, FA, UP): de `vs`
salen los errores por bloque, el rating, los efectos de casa, el «Sesgo bloques», las dianas y Derecha frente a
Izquierda en Inicio. Luis quiere que la web hable el mismo idioma: tres bloques, **Izquierda** y **Derecha** de
`vs` y **Otros** con todo lo demás.

## Objetivos

- En toda vista «Bloques» de la web (Promedios, Escaños y detalle de una elección, pasada o futura) hay sólo tres
  bloques: Izquierda, Derecha y Otros.
- Izquierda y Derecha son los partidos de `bmaps.vs` de esa elección.
- **Otros es el resto hasta el total**: en voto, 100 − Izquierda − Derecha (entran los partidos fuera de `vs`,
  el residuo del modelo `'-'` y cualquier candidatura sin modelar); en escaños, total de la cámara − Izquierda −
  Derecha. En la run la resta se hace **en cada simulación**, de modo que Otros tiene rango 95 % propio.
- Otros no tiene P(mayoría): se escribe «–» (no es una coalición posible).
- La columna «vs {año}» de los bloques **sigue el linaje del smap, como la de los partidos** (decidido por Luis):
  el resultado anterior se lleva primero a los partidos de hoy (`inherited_results`: PP 2026 frente a PP + UPN
  de 2023, FA frente a Sumar) y luego se agrupa con el `vs` de la elección de **hoy**; Otros anterior = total −
  I − D anterior. Así la delta de Derecha cuadra con las de sus partidos (el escaño de UPN de 2023 ya era de
  Derecha).
- La regla vale para **todos los ámbitos**, también los autonómicos (decidido por Luis), aunque allí Otros sea a
  menudo el bloque mayor (País Vasco: PNV + EH Bildu; Cataluña: Junts + ERC + CUP + AC + Comuns; Galicia: BNG;
  Navarra: UPN + Geroa Bai + EH Bildu; Canarias: CC + ASG + AHI + NC) y salga con «–» en P(mayoría). Es
  coherente con el modelo, que mide los bloques de cada ámbito con su `vs`.
- Los sondeos siguen la misma regla: en la curva del promedio en vista Bloques (Promedios y detalle de una
  elección) y en la vista Bloques de «Una encuestadora frente al promedio», Otros de cada sondeo (y de cada
  fecha de la serie) es 100 − I − D. «Todas las encuestas» no tiene vista Bloques y no se le añade.
- Regionalista y Separatista desaparecen de la web.

## No objetivos

- Cambiar el modelo, `params.json` ni `bmaps.blocks`: el mapa de cuatro bloques sigue en el fichero y en el
  modelo para quien lo use; sólo deja de enseñarse.
- Tocar lo que ya sale de `vs`: Derecha frente a Izquierda en Inicio, la Calculadora de mayorías (partidos
  marcados de entrada), las dianas, el «Sesgo bloques», el Histórico de errores, el rating y el reparto por
  bloque del último sondeo (`charts/diverging.js`).
- «Lo que dijo el modelo» en el detalle de una elección: sigue con Izquierda y Derecha del backtest. Añadir Otros
  exigiría repetir el backtest entero.
- El orden de los **partidos** en el hemiciclo y en las tablas de partidos (`HEMICYCLE_BLOCKS`, `catalog.order`
  por `parties.block`): no cambia; sólo cambia la vista de bloques.
- Republicar las runs antiguas (relleno, `?run=` fijado): conservan los bloques que publicaron (cuatro). La web
  no debe romperse con ellas (nombres desconocidos al final, como hoy), pero no las convierte.

## Anti-objetivos

- Que los tres bloques no sumen el total (100 en voto, el total de escaños en escaños) en una misma vista.
- Que Otros salga sin rango en la tabla de Escaños mientras Izquierda y Derecha lo tienen.
- Que la página y la run digan cosas distintas de un mismo bloque (la página no recalcula lo que la run publica).

## Enfoques considerados

- **A. La run publica los tres bloques (ELEGIDO por Luis).** Al publicar, el grupo `blocks` del resumen y de la
  proyección se calcula con un mapa nuevo de tres bloques; la suma por simulación da rangos de verdad. Las
  páginas apenas cambian. Coste: hay que republicar la run actual para verlo; en elecciones pasadas el resultado
  real se agrupa en la página con el `vs` de `params.json`.
- B. Calcularlo en la página por diferencia a partir de `summary['vs']`: sin republicar nada, pero Otros se queda
  sin rango 95 % (los intervalos no se restan) y la tabla de Escaños tendría un hueco. Descartado.
- C. A más B como respaldo para runs antiguas: más código para un caso casi invisible (Promedios y Escaños leen
  la run más reciente salvo `?run=`). Descartado.

## Diseño

### 1. Publicación de la run (`mtpy/lib/publish/forecast.py`)

- Un mapa de tres bloques derivado de `bmaps.vs` del simulador: `{'Izquierda': vs['Izquierda'], 'Derecha':
  vs['Derecha'], 'Otros': [todas las demás categorías simuladas, incluido el residuo Simulator.OTHERS]}`. Se
  construye con una función pura y probada (p. ej. `web_blocks(vs, categories)`), en el orden Izquierda,
  Derecha, Otros.
- Ese mapa sustituye a `'blocks'` en `export_summary` (tabla `blocks` del resumen) y en `export_projection`
  (grupo `blocks` de la proyección nacional diaria, la que el navegador dibuja en modo Elección como línea
  discontinua y barra de rango 95 % en la fecha de la elección), y se escribe en el meta de la run como
  `bmaps.blocks`, para que el navegador agrupe sondeos y serie con el mismo mapa. `bmaps.vs` y `bmaps.main` del
  meta no cambian. (`export_districts` no tiene grupo de bloques y no se toca.)
- **Otros en la proyección es el complemento** (decisión mía, se sigue de «Otros = resto hasta 100»): media
  100 − I − D en cada fecha, y su banda es la de I + D (Var(100 − I − D) = Var(I + D)), calculada como la de
  I + D en la proyección (con la correlación que ya use el simulador para sumar bloques). Así la línea
  discontinua de Otros empalma en `as_of` con la curva histórica, que el navegador calcula como 100 − I − D, y
  I + D + Otros = 100 también en la proyección. Si `Simulator.projection` no sabe hacer un bloque resto, el
  constructor lo añade allí o lo deriva en el publicador; el invariante es lo que cuenta.
- `p_majority` de Otros: nulo en la tabla `blocks` (no se publica una probabilidad de un bloque que no es
  coalición). `_p_majority` sigue saliendo de `vs` y no cambia.
- Invariantes que deben comprobar los tests: en cada simulación I + D + Otros = total de escaños; `pct`,
  `pct_mean` y la cuota simulada de I + D + Otros = 100 (con tolerancia de redondeo); en la proyección, la
  media de I + D + Otros = 100 en cada fecha. Si `Simulator.summary` no
  incluye el residuo en `pct` (hoy `pct` sale de `forecast.loc[parties]`, sin `'-'`), el constructor se asegura
  de que Otros lo incluya, sea pasando el residuo o calculando el `pct` de Otros como complemento; lo que
  importa es el invariante, no la vía.
- Si `vs` de la elección está vacío o no tiene Izquierda y Derecha, el grupo `blocks` se publica como hoy (con
  `bmaps.blocks` del modelo) y la vista Bloques se comporta como ahora; es un caso de ámbito raro, no se
  inventa nada.

### 2. Páginas Python

- **Orden de los bloques en tablas**: Izquierda, Derecha, Otros (y nombres desconocidos detrás, en el orden de
  entrada, para las runs antiguas). Sustituye a `BLOCK_ORDER` de `promedios.py` (hoy Derecha primero) y a su uso
  en `elecciones.event_block_rows`.
- **Color de Otros**: `OTHERS_COLOR` (#9e9e9e), que `block_color` ya devuelve para un bloque sin partido en
  `vs`. Hay que comprobar que `block_color('Otros')` no coge el color del primer partido de Otros a través del
  nuevo `bmaps.blocks` del meta (hoy busca en `vs` y luego en `blocks`): Otros va gris siempre.
- **P(mayoría) de Otros**: «–» allí donde la tabla de bloques tiene esa columna.
- **Promedios** (`block_result_rows`, deltas «vs {año}»): sin cambios de lógica más allá del orden; las cifras
  vienen de la run.
- **Último resultado y elecciones pasadas** (`elecciones.last_results`, `party_blocks`, `event_block_rows` y el
  detalle de una elección): cada partido del resultado va a Izquierda o Derecha si está en `bmaps.vs` de los
  parámetros de esa elección (`event_params`; de respaldo, `blocks` del meta del evento, que es su `vs`), y a
  Otros en cualquier otro caso (ya no al bloque del catálogo). Otros: `pct` = 100 − I − D, `seats` = total − I
  − D (no la suma de los partidos listados, que deja fuera votos sin candidatura con escaño). Esto sirve para
  la tabla del detalle de esa elección, con su propio `vs`.
- **Deltas «vs {año}» de los bloques en Promedios y Escaños**: no se toman de `last_results['blocks']` agrupado
  con el `vs` de entonces, sino de los resultados heredados por partido (`inherited_results`, el mismo linaje
  que las deltas de partidos) agrupados con el `vs` de la elección actual, con Otros por diferencia (ver
  Objetivos).
- **Curva del promedio en el detalle de una elección** (`curve_blocks`, `curve_block_colors`, `bmaps.blocks`
  del contexto): el mapa de bloques que se pasa al navegador es el de tres de esa elección, construido con su
  `vs`; Otros sin lista de partidos (ver 3).
- `run_less_colours` (ámbitos sin previsión publicada): con Otros siempre gris, sólo necesita los colores de
  Izquierda y Derecha; se simplifica en consecuencia.

### 3. Navegador (`web/dist/js`)

- **Otros como resto** en `blocks.js`: `blockSeries`, `blockPolls` y `blockResult` aceptan un bloque «resto»
  (Otros) cuyo valor es 100 − la suma de los demás bloques. Se calcula sólo cuando Izquierda y Derecha tienen
  valor en ese sondeo o fecha; si falta alguno, Otros no tiene valor (no se inventa). La banda de Otros en la
  serie usa la de I + D (Var(100 − I − D) = Var(I + D)), sumando en cuadratura las semibandas de los partidos de
  I y D, como hoy los demás bloques. Cómo se marca el bloque resto en el mapa (nombre `Otros`, lista vacía o
  marcador explícito) es decisión del constructor, siempre que el servidor y el navegador usen la misma regla.
- **Orden en el hemiciclo y en la barra apilada de Escaños**: Izquierda a la izquierda, Otros en el centro,
  Derecha a la derecha (`catalog.orderBlocks`). `BLOCK_ORDER` de `catalog.js` sigue sirviendo para ordenar
  **partidos** (`catalog.order`); la vista de bloques usa su propio orden.
- **Escaños**: el reparto por bloques, los «ridges» de bloques y la tabla «Bloques ideológicos» leen
  `summary.blocks` y `bmaps.blocks` de la run nueva; Otros con su rango 95 %.
- **Color**: `catalog.blockColor('Otros')` devuelve `OTHERS_COLOR` aunque el nuevo `bmaps.blocks` liste partidos
  en Otros.

### 4. Textos públicos

- Escaños: el pie «Bloques ideológicos, con todos los partidos.» pasa a decir qué es cada bloque, p. ej.
  «Izquierda y Derecha, con todos sus partidos; Otros es el resto de la cámara.»
- Promedios y detalle de una elección: donde dice «En la vista Bloques, cada bloque suma sus partidos», añadir
  que Otros es lo que falta hasta 100.
- Revisar con `grep` otros textos que nombren Regionalista, Separatista o los cuatro bloques (plantillas y
  `/metodo`).
- Reglas de redacción de Luis: «sólo» con tilde, «encuestadora» o «casa», nunca «empresa»
  (`assert_public_wording`).

### 5. Después de integrar (Luis)

- `publish forecast` (o la run del día) de `es` y de los ámbitos autonómicos con previsión, para que Promedios y
  Escaños enseñen los tres bloques. No hace falta republicar eventos ni el backtest: las elecciones pasadas se
  agrupan en la página (también las autonómicas, que cambian en cuanto se despliegue).

## Decisiones del diseño

Tomadas con Luis el 2026-10-28:

- Desaparecen los cuatro bloques de la web; quedan Izquierda, Derecha y Otros.
- Otros = resto hasta 100 (voto) o hasta el total (escaños), residuo incluido; sin P(mayoría); «vs {año}» por
  diferencia.
- Enfoque A: la run publica los tres bloques.
- La regla vale en todos los ámbitos, autonómicos incluidos.
- «vs {año}» de los bloques con el linaje del smap, como los partidos.

Decisiones mías (para revisar): Otros de la proyección como complemento con la banda de I + D; la vista Bloques
de sondeos es la de la curva y la de «Una encuestadora frente al promedio» («Todas las encuestas» no gana una);
si `vs` no tiene Izquierda y Derecha, se publica como hoy.
- Orden en tablas Izquierda, Derecha, Otros; en el hemiciclo Izquierda, Otros, Derecha; Otros en gris
  #9e9e9e; lo que ya sale de `vs` no se toca; `bmaps.blocks` sigue en el fichero y en el modelo.

## Queda para el constructor

- La vía concreta para que Otros incluya el residuo en `pct` (ver 1).
- Cómo se marca el bloque resto en el mapa del navegador (ver 3).
- Los nombres de las funciones nuevas (en inglés neutro, como el resto del código).
- Los tests: invariantes de suma en la publicación; orden, color y «–» en las páginas; Otros por diferencia en
  `blocks.js` (sondeos sin I o D, serie con huecos) y en el resultado de una elección pasada.
