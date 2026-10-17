# Web: degradados y barras en las tablas, tipo de encuestadora y filtro de años (2026-10-17)

Seguimiento del recorrido de Luis por la web con datos reales, tras la fase 8 y los ajustes del 16 y 17 de octubre
(«Confianza», ranking reordenado con tooltips, diana del perfil sombreada). Una tanda de cambios de presentación en
las tablas de cuatro páginas; no cambia ningún dato publicado ni ningún contrato del bundle.

## Qué pidió Luis

Histórico Errores, tarjeta «Elecciones publicadas»:

1. Quitar las columnas «Error de bloques» y «Modelo».
2. «Error medio» con degradado rojo de 0 a 8.
3. «Sesgo bloques» con barra divergente de -8 a 8, roja a la izquierda y azul a la derecha.

Detalle de una elección, tarjeta «Último sondeo de cada encuestadora»:

4. «Error medio» con degradado rojo de 0 a 8.
5. «Error de bloques» pasa a ser `error_blocks` (con signo), se llama «Sesgo bloques» y lleva la misma barra
   divergente de -8 a 8.
6. Las columnas «Derecha» e «Izquierda» cambian de orden (Izquierda a la izquierda) y llevan degradado
   rojo-neutro-verde de -8 a 8.

Perfil de una encuestadora:

7. El KPI «Tipo» en castellano (`state` → «Pública», `aggr` → «Agregadora», `online` → «Online», sin tipo →
   «Estándar»; acepta sugerencias).
8. «Diana de errores»: el filtro de años ofrece todos los años desde su primer evento hasta el año actual, no solo
   los de elecciones; en la tabla, Izquierda antes que Derecha y degradado rojo-neutro-verde de -8 a 8.
9. «Último sondeo por elección»: el degradado rojo de cada celda pasa a rojo-neutro-verde de -8 a 8.
10. «Efectos de casa por elección»: degradado rojo-neutro-verde de -8 a 8.
11. «Rebaño por ciclo»: en «Rebaño», doble degradado rojo de 0 a 1 y gris de 1 a 2.

Promedios, tarjetas «Resultados por partido» y «Resultados por bloque»:

12. Quitar las columnas P(*), que la pestaña Escaños detalla.

## Decisiones

- **Un filtro para los dobles degradados.** `shade_around(x, centre, span, below, above)` en `pages/filters.py`
  (registrado como `shade_around`): `below` crece del centro hacia `centre - span` y `above` hacia
  `centre + span`; `'transparent'` sin valor o sin `span` positivo. Sustituye las parejas de `shade` con
  condicional que ya había en las plantillas (cambio del camino al rating, rebaño del ranking, diana del perfil).
  `format.js` lleva `shade` y `shadeAround`, espejo de los filtros, con un test de paridad en node
  (`tests/test_js_format.py`); el JS del perfil los usa al reconstruir la tabla de la diana con cada filtro, que
  antes perdía el sombreado del servidor.
- **La barra del sesgo toma los colores de los bloques**, como en el ranking y en el camino al rating: positivo
  favorece al primer bloque (Derecha, azul) y crece hacia la derecha; negativo al segundo (Izquierda, rojo) y crece
  hacia la izquierda. Es lo que Luis pidió («roja a la izquierda y azul a la derecha») sin fijar colores que no
  valdrían en un ámbito con otros bloques. `bias_colors(blocks, meta, catalog)` en `pages/elecciones.py`; en el
  histórico, sin análisis publicado, el par sale de los dos bloques del run (`bmaps.vs`).
- **El orden de las columnas de bloque es solo de las tablas.** `table_blocks(blocks)` pone `LEFT_BLOCK`
  («Izquierda») primero, con la misma regla que `split_sides`; `blocks` y los ejes de la diana siguen en el orden
  publicado (Derecha en X, Izquierda en Y), y el JSON inicial no cambia. El JS lee el orden de la cabecera de la
  tabla, no de los datos.
- **Etiquetas del tipo** (mi propuesta, sobre la de Luis; cerrada con él el mismo día): `state` → «Pública», `aggr` →
  «Agregador» (el sustantivo que ya usa la prosa del sitio, en vez de «Agregadora»), `online` →
  «Panel Online» (dice qué es: un panel en línea autoseleccionado, no «tiene web»; Luis fijó las dos iniciales en
  mayúscula), sin tipo (nulo en la base de
  datos, que el catálogo imprime como «-») → «Privada», la contraparte natural de «Pública» y la palabra que ya
  usaba la web. Un valor desconocido sale tal cual. `MTYPE_LABELS`, `MTYPE_DEFAULT` y `mtype_label` en
  `pages/casas.py`. Si Luis prefiere «Online» y «Estándar», son dos cadenas.
- **Los años del filtro de la diana** salen de `year_range(years, today)`, el mismo cálculo que ya hacía el
  histórico (desde el primer año hasta hoy, o hasta el último año con elección si es posterior); el JS toma los
  extremos de las opciones del `select`, no de los datos.
- **Las barras necesitan sitio** en una tabla sin columnas fijas: `.cell-bar { min-width: 120px }`.
- **Textos.** Cada «Cómo leerlo» cuenta el degradado o la barra nuevos con las palabras de los demás («más rojo
  cuanto mayor, de 0 a 8 puntos», «crece hacia el lado del bloque al que favoreció… hasta 8 puntos de margen», «rojo
  si se quedó corto y verde si se pasó»). La tarjeta de Promedios remite a la pestaña Escaños para las
  probabilidades; la tira del líder del hemiciclo conserva su P(primero). La línea resumen de la cabecera de una
  elección sigue dando el error de bloques sin signo, y el glosario de esa página lo define.
- **Fuera de alcance.** El orden de las filas de «Último sondeo de cada encuestadora» sigue siendo el error de
  bloques sin signo (como las barras y la diana); la pestaña Escaños conserva sus P(*); la ranking de `/casas` no
  cambia salvo el filtro del rebaño, que da el mismo color.

## Comprobación

- `tests/test_pages.py`: tests nuevos de `shade_around`, `year_range`, `table_blocks` y `bias_colors`,
  `mtype_label`, la tabla de elecciones publicadas, la del último sondeo de una elección, las tablas del perfil
  (diana, último sondeo, efectos, rebaño y años del filtro) y las tablas de Promedios; los existentes actualizados
  (cabeceras, etiquetas del tipo, opciones del filtro, signo del sesgo, sombreado del último sondeo).
- `tests/test_js_format.py`: paridad de `shade` y `shadeAround` con los filtros de Python.
- Pendiente de Luis: ver las páginas con datos reales (sobre todo la anchura de las barras en «Elecciones
  publicadas» y «Último sondeo de cada encuestadora», y el nombre de los tipos).

## Segunda petición del mismo día: Promedios frente a la última elección

Luis quiere comparar el promedio con las anteriores elecciones del ámbito, en porcentaje y en escaños, también por
bloques. Promedios ya lo hacía solo en las tarjetas de partido (`party_cards`: «−25 esc. vs 2023», «+0,0 pt»,
«nuevo») y en el chip de Inicio. Propuesta aceptada en el chat:

- En «Resultados por partido» y «Resultados por bloque», una columna «vs {año}» tras «% Voto» (puntos) y otra tras
  «Escaños», en el hueco de las P(*). La diferencia va con signo (`delta`) y con el doble degradado rojo-verde
  (`shade_around`): ±5 puntos en el voto (`DELTA_PCT_SPAN`) y una décima de la cámara en escaños
  (`delta_seats_span`, 35 en el Congreso), para que una cámara pequeña no salga saturada. La cabecera lleva como
  tooltip la fecha exacta («Diferencia en puntos frente al resultado oficial del 23 jul 2023») y el lead de la
  tarjeta dice «y cuánto cambian frente a 2023».
- Un partido que no concurrió entonces lleva un guion, el «–» de las celdas sin valor (Luis lo prefirió a «nuevo»;
  las tarjetas de partido conservan su «nuevo»). Un partido que desapareció no tiene fila, pero cuenta en su bloque.
- La elección es la última publicada del ámbito con resultados, la misma de las tarjetas (`last_results`), así que la
  comparación aparece en cada ámbito autonómico con su elección publicada y no aparece sin ella. Los bloques del
  resultado anterior salen de sumar cada partido de entonces en el bloque que le dan los parámetros de esa elección,
  su meta o el catálogo (`party_blocks` + `event_block_rows`, la regla del detalle de elección): `last_results`
  acepta ahora el catálogo y devuelve también `blocks`.
- `result_rows` y `block_result_rows` llevan `prev_pct`, `prev_seats`, `delta_pct` y `delta_seats` (`None` sin
  resultados o sin presencia entonces); el contexto añade `results_date` y `delta_spans`.
- El hemiciclo, las series y la pestaña Escaños no cambian. Queda como posible paso posterior una tarjeta de
  subidas y bajadas con barras divergentes.

Comprobación: tests de `last_results` con catálogo, de las filas con diferencias y de la escala de escaños, y de las
dos tablas con y sin elección publicada (`tests/test_pages.py`).

## Ajustes de la tarde (recorrido de Luis)

Luis vio las columnas «vs 2023» con datos reales y pidió tres cosas:

1. **Tarjetas de partido.** La fila de comparación quedaba pegada a «Último año»; ahora va justo bajo «Escaños» y
   «Voto», alineada con esas etiquetas (rejilla de dos columnas de 56 px) y en fuente mucho más pequeña: dos insignias
   «▼ 25» y «–» (la de voto con un decimal) en blanco translúcido, como la píldora del puesto, con la fecha de la
   elección en el tooltip de la fila. Un partido sin resultado anterior lleva «–» (ya no «nuevo»; `party_cards` da
   las diferencias numéricas).
2. **Tablas de resultados.** Sin degradados en «vs 2023»: una insignia pequeña (`_macros.badge`, clase `.dlt`) con
   «▼ 1» en rojo, «▲ 6» en verde o «–» en gris neutro, para la diferencia de voto (un decimal) y la de escaños. La
   magnitud se redondea como `fmt_num` y la insignia es neutra cuando redondea a cero («+0,04» → «–»). Se retiran
   `DELTA_PCT_SPAN`, `delta_seats_span` y `delta_spans`.
3. **Hemiciclos y barra de Escaños.** Junto a los escaños de cada sector del hemiciclo (Inicio y Promedios, vistas
   Partidos y Bloques) y de cada segmento de la barra apilada de Escaños, la misma flecha con la magnitud («135 ▼ 25»)
   en el blanco de la etiqueta, porque rojo y verde no se leerían sobre el color del partido; el tooltip la repite
   con el año («▼ 25 frente a 2023»). Un partido o bloque sin resultado anterior no lleva nada en el gráfico (la
   tabla sí lleva su guion). El servidor pasa `deltas` (`seat_deltas`: `{parties, blocks, year}`) en el JSON inicial
   de Promedios y Escaños y `deltas` + `results_year` en el `hero` de Inicio; `hemicycleData`/`hemicycleLabel`
   (hemicycle.js) y `stackedOption` (stacked.js, extraído de `renderStacked`) escriben la etiqueta.

El filtro `badge` (`delta_badge`) tiene su espejo `fmtDelta` en `format.js`, con test de paridad en node.

Tercera tanda (misma tarde), tras ver las insignias:

4. **Tarjetas de partido:** la flecha va al lado de cada cifra («136 ▼ 1 · 32,1% ▼ 1,0»), en pequeño y en blanco, y
   «escaños» / «voto» siguen alineados debajo gracias a una rejilla de dos columnas (`.pc .fig`); desaparece la fila
   aparte. Riesgo que queda por ver con datos reales: en tarjetas estrechas (seis por fila en un portátil) las dos
   flechas pueden rozar el borde; si pasa, bajar la cifra grande de 38 a 34 px.
5. **Hemiciclos:** la flecha va debajo de los escaños, en una tercera línea de la etiqueta.
6. **Tablas:** la flecha va en su color (verde, rojo o gris) pero sin píldora: `.dlt` es solo texto en negrita. El
   nombre `badge` del filtro y la macro se conserva.

Cuarta tanda (misma tarde):

7. **Hemiciclo:** un poco de aire entre la cifra de escaños y la flecha (3 px de relleno superior en el estilo
   `delta` de la etiqueta; afecta también al de Inicio, que comparte el componente).
8. **Barra de Escaños:** «PSOE 106» en negrita y «▼ 15» en peso normal (etiqueta con texto enriquecido `seats` /
   `delta` en `stackedOption`).
9. **Tablas de Escaños:** las dos tablas de debajo de la barra llevan la columna «vs {año}» tras «Escaños», con la
   misma flecha (solo escaños: no hay voto en ellas); el contexto de Escaños añade `deltas`, `results_year` y
   `results_date`, y el «Cómo leerlo» de la tarjeta explica la flecha.
