# Detalle de un sondeo: el cambio frente a la entrega anterior de la misma casa y medio — diseño corto

Fecha: 2026-11-09. Amplía `docs/superpowers/specs/2026-11-09-web-detalle-sondeo-design.md` (página
`/{scope}/sondeos/{fecha}/{casa}[/{medio}]`, fusionada en `dev` en 874d9b1).

## Intención

Luis quiere que la página de un sondeo, además de compararlo con el promedio del día, lo compare con **la entrega
anterior de la misma casa para el mismo medio en el ciclo actual** (40dB para Prisa frente al 40dB para Prisa
anterior): cuánto ha movido cada partido. Y lo quiere **en la imagen que se comparte** (opción 2, elegida por Luis el
2026-11-09): el cambio junto a cada barra, porque es lo primero que se pregunta quien ve un sondeo nuevo.

## Objetivos

1. Encontrar la entrega anterior: entre los sondeos del run (el ciclo actual), los de la misma casa (`pollster_id`, o
   `pollster` si el sondeo no tiene id, como el gráfico de la casa) y el mismo medio (`sponsor`, comparado por su slug
   como `find_poll` y las URL) con `date` estrictamente anterior; la de
   fecha más reciente (empate: la última publicada en la parte, como en la tabla).
2. **Tarjeta (imagen):** junto a la cifra de cada barra, el cambio frente a la anterior con el formato de las flechas
   de la web (`fmtDelta` a un decimal: «31,9 % ▲ 1,2», «▼ 0,4», «–» si no cambia), verde si sube y rojo si baja
   (los colores de `.dlt`), sólo en los partidos con cifra en las dos entregas; la fila «Otros» no lleva flecha (su
   resto depende de qué partidos lista cada entrega). La nota de la tarjeta añade «Flechas: frente al sondeo anterior de
   {casa}{ para {medio}} (campo {fechas de campo})», más «, con días de campo en común» si el último día de campo de
   la anterior (o su `date`) no es anterior al primero de este, y el punto final; sin fechas de campo en la anterior,
   «(campo hasta el {fecha})»; sin anterior, «Primer sondeo de {casa}{ para {medio}} en este ciclo.» y
   ninguna flecha.
3. **Tabla «Voto y escaños»:** dos columnas tras «Diferencia»: «Anterior» (la cifra de la entrega anterior) y
   «Cambio» (`m.badge(…, 1)`, vacía si la anterior no da cifra para el partido); con escaños en las dos entregas, la columna «Escaños» muestra además el cambio del
   punto medio de la horquilla (`m.badge(…, 0)`, omitido si es 0). Pie: «Anterior: sondeo de {casa}{ para {medio}},
   campo {fechas de campo}» con enlace a su página. Sin
   anterior, las columnas no aparecen y el pie dice «Primer sondeo de … en este ciclo».
4. «Cómo leerlo» de la tarjeta y de la tabla explica las flechas, el guion (no cambia) frente a la ausencia de flecha
   (la anterior no daba cifra) y, en la tabla, la flecha de los escaños.

## No objetivos

- Comparar con entregas de ciclos anteriores o de la misma casa para otro medio.
- Flechas en el hemiciclo.
- Cambiar la página del evento ni el gráfico de la casa (ya marca el sondeo con un anillo).

## Enfoques

1. **Etiqueta compuesta en `bars.js`** (elegido): `renderBars` gana una opción `labelExtra(row) -> {text, color}|null`
   que dibuja, tras la cifra, un segundo texto con su color, dentro del mismo `renderItem` (dos textos de zrender
   seguidos, el segundo colocado tras medir el primero o con `rich`). Las demás llamadas no la pasan y no cambian.
   La imagen la recoge sola, porque `card-image.js` redibuja la opción del gráfico.
2. Superponer las flechas como HTML sobre el gráfico: se descolocaría al redibujar a 1200 px y html2canvas las
   pondría en otro sitio. Descartado.
3. Una segunda serie de texto en ECharts: duplica la lógica de posición de la etiqueta. Descartado.

## Diseño

- `mtpy/lib/pages/sondeos.py`: `previous_poll(polls, poll) -> Optional[dict]` (objetivo 1); `vote_rows` gana
  `previous` (opcional) y añade a cada fila `prev` (cifra o `None`), `change` (`round(value − prev, 1)` o `None`) y,
  con escaños en las dos, `seats_change` (diferencia de puntos medios redondeados, como `pollSeats`);
  `sondeo_context` añade `previous` (`{'date', 'href', 'title'}` o `None`, `href` con `poll_detail_href`) y la isla
  `initial` añade `previous` (`{'date', 'title', 'values': {partido: cifra}}` o `None`).
- `web/dist/js/poll-card.js`: `voteRows(poll, parties, average, previous = null)` añade `change` a cada fila con cifra
  en las dos entregas (`null` en «Otros» y sin anterior).
- `web/dist/js/charts/bars.js`: opción `labelExtra` (enfoque 1), con JSDoc y test en node (con y sin la opción).
- `web/dist/js/pages/sondeo.js`: pasa `labelExtra: (row) => row.change == null ? null : {text: fmtDelta(row.change,
  1), color}` con los colores de `.dlt` (`#2ca02c`, `#d62728`, el gris apagado para «–»).
- `web/templates/sondeo.html`: la nota de la tarjeta, las dos columnas, el pie y los «Cómo leerlo».
- Tests: `previous_poll` (misma casa y medio, otro medio no cuenta, sin medio frente a con medio, fecha igual o
  posterior no cuenta, primero del ciclo), `vote_rows` con anterior, la página (columnas, pie con enlace, nota de la
  tarjeta, sin anterior), `voteRows` y `labelExtra` en node; comprobación en Chrome de la imagen.

## Lo que queda al constructor

La posición exacta del segundo texto (separación, tamaño) y si la flecha va en negrita; que no se salga del gráfico
con el partido más votado (el eje ya deja un 15 % de margen: ampliarlo si hace falta sólo cuando hay flechas).

## Cambio de Luis, 2026-11-09 (tras la fusión)

Las flechas salen de la tarjeta y de su imagen: la comparación con la entrega anterior queda sólo en la tabla «Voto y
escaños» (columnas «vs Anterior» del voto y de los escaños) y en su pie, que da el campo de la anterior y avisa de los
días en común. Ver el cambio en `2026-11-09-web-detalle-sondeo-design.md`.
