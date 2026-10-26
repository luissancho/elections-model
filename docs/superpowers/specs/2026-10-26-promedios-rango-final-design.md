# Promedios: fuera la banda del histórico, el rango del 29-N como barra al final

Fecha: 2026-10-26. Estado: aprobada por Luis y CONSTRUIDA el 2026-10-26 en `dev` (plan
`docs/superpowers/plans/2026-10-26-promedios-rango-final.md`, ejecución nativa, sin commits: los hace Luis).
Desviación: el detalle de una elección conserva la palabra «banda» en la explicación de la celda «Rango 95%» de «Lo
que dijo el modelo», que es la franja de escaños de la tabla, no la del gráfico. Mockups:
`docs/superpowers/specs/2026-10-26-promedios-banda-mockups.html` (variante elegida: **B1**).

## Intención

El gráfico «Promedio de sondeos y proyección» mezcla hoy dos incertidumbres distintas en una misma franja: en el
histórico, la banda es la precisión del promedio (±0,9 en el PP); en la proyección, el margen del pronóstico (±5,2
que crece a ±5,7). Vistas seguidas parecen la misma cosa con un escalón inexplicable (Luis, 26-10: «así no se
entiende»). Decisión: el histórico va sin banda, sólo la línea y los sondeos; la proyección es la línea discontinua
sin franja; y el rango del 95 % del voto el día de la votación se dibuja como una barra vertical por partido en la
fecha de la elección, con el punto del voto central, sin etiquetas fijas: las cifras salen al pasar por el punto, y
al pasar la barra del partido viene al frente y se ve completa. El rango en porcentaje queda a la vista, cosa que
las tarjetas, siempre en escaños, no dan.

## Qué hay hoy

- `seriesOption` (`web/dist/js/charts/series.js`) dibuja por partido la banda `lo–hi` de `series` (la precisión del
  promedio, `fc_stat`), la línea `mean`, los sondeos y, en Elección, la línea discontinua de `projection` con su
  franja `lo–hi` (el abanico: `s² + e² + d(h)²`) desde `as_of` hasta `when`. Lo usan Promedios (partidos y, por
  `blockProjection`, bloques), «Una encuestadora frente al promedio» y el detalle de una elección pasada (con el
  resultado como rombo y sin proyección). El tooltip de la línea y de la proyección da «valor (lo–hi) · fecha»
  (`meanTooltipLine`).
- `weights.js` (ejemplo vivo de /metodo/promedio) dibuja la banda de la regresión del ejemplo con `bandSeries`.
- Textos que hablan de la banda: `promedios.html` (aria-label, «Cómo leerlo» de la serie y de la encuestadora),
  `eleccion.html` (aria-label y «Cómo leerlo»), `metodo_escanos.html` («s … el de su banda», tarjeta de los tres
  errores), `metodo_promedio.html` (intro, navegación «La banda», tarjeta «Qué significa la banda del promedio»).

## Enfoques

- **A.** Banda fina del promedio en el histórico + barra del 29-N al final. Conserva lo que /metodo/promedio explica,
  pero mantiene dos lecturas de «banda» en un mismo gráfico.
- **B1.** Sin banda en el histórico + barras del 29-N superpuestas en la vertical de la elección, con realce al pasar.
  Lo menos confuso; el solape se resuelve con la interacción.
- **B2.** Igual que B1 con un carril por partido a la derecha del 29-N. Sin solape, pero come anchura y añade un
  elemento nuevo que explicar.
- **C.** El error completo siempre (franja de ±5 en todo el ciclo). Sin escalón, pero se traga los sondeos y la
  tendencia.
- **Selector «Ver intervalos»** con B2 oculto por defecto. Descartado: una proyección desnuda se lee como predicción
  sin margen y un botón pide saber qué es un intervalo antes de pulsarlo.

Elegido **B1** (Luis, 2026-10-26).

## Decisiones (mías, con su razón)

1. **La banda del histórico desaparece de todos los gráficos que pasan por `seriesOption`**: Promedios (partidos y
   bloques), la encuestadora frente al promedio y el detalle de una elección pasada. Un mismo gráfico con banda en
   una página y sin ella en otra volvería a confundir. `bandSeries` sigue existiendo para el ejemplo de
   /metodo/promedio, donde la banda es el objeto de la explicación. Las partes publicadas no cambian: `lo`/`hi` de
   `series` siguen en la API y el CSV.
2. **El tooltip del histórico pierde el paréntesis** `(lo–hi)`: si la banda no se dibuja, su cifra tampoco se
   enseña. El de la proyección conserva el rango, que es el del pronóstico, con su cabecera «Proyección».
3. **La barra**: en la fecha de la elección (`when`), por partido visible, un segmento vertical de `lo` a `hi` del
   último día de la proyección, extremos redondeados, 4 px, color del partido, opacidad 0,45, y un punto blanco con
   borde del color en `mean`. Dibujada por encima de las líneas. Al pasar por el punto o por la barra, la del partido
   pasa a opacidad 1 y al frente y las demás se atenúan (realce por serie con desenfoque del resto, el mecanismo de
   ECharts que el constructor elija); tooltip «Elección · {fecha}» y debajo «PP: 31,4% (25,9–37,0)». Comparte
   `name` con la línea del partido: la leyenda la oculta y la muestra con él.
4. **Cuándo se dibuja**: sólo con proyección de más de un día (modo Elección); en Hoy el gráfico queda en línea y
   sondeos. Pertenece a la capa del promedio: visible con «Promedio» y «Ambos», oculta con «Sondeos». La vista
   Bloques dibuja la barra de cada bloque con `blockProjection`. El detalle de una elección pasada no tiene
   proyección y no la dibuja; sus rombos del resultado no cambian.
5. **Textos**. `promedios.html`: aria-label «Promedio de sondeos{{ y la proyección }}» sin «banda»; «Cómo leerlo»
   de la serie: «la línea es el promedio ponderado del modelo. En el modo Elección, la línea discontinua lleva el
   voto estimado hasta el día de la votación y la barra del final es el rango del 95 % de cada partido ese día: pasa
   por el punto para ver las cifras»; el selector «deja a la vista sólo los puntos, sólo la línea, o las dos cosas»;
   la frase «La banda dice dónde está el consenso…» pasa a «La línea dice dónde está el consenso de las
   encuestadoras; la barra del final, dónde puede estar el voto el día de la elección»; tarjeta de la encuestadora:
   «la línea es el promedio de todas». `eleccion.html`: aria-label y «Cómo leerlo» sin banda; «La línea dice dónde
   estaba el consenso de las encuestadoras, no la incertidumbre del resultado». `metodo_escanos.html`: «s es el
   error del propio promedio: dice cuánto puede equivocarse la línea al resumir los sondeos» (sin «banda»), y en la
   tarjeta de los tres errores «s es la precisión del promedio» / «vale varias veces el error del promedio».
   `metodo_promedio.html`: la tarjeta pasa a «Qué significa la precisión del promedio» con la entrada de navegación
   «La precisión del promedio», y añade un párrafo: desde octubre de 2026 los gráficos de la web no la dibujan como
   banda, porque se confundía con el margen del pronóstico; es `s`, la primera de las tres piezas del margen
   (enlace a `/metodo/escanos#errores`), y el ejemplo de más abajo sí la dibuja. La intro (líneas 40 y 46) deja de
   anunciar «una banda» en la curva y habla de «una medida de la precisión». Las notas para expertos conservan
   «banda» como término técnico.
6. **Redacción**: reglas de Luis («sólo» con tilde, encuestadora/casa, nunca empresa; `assert_public_wording`).

## No objetivos

- Cambiar el modo por defecto, el abanico de Escaños, las tarjetas o la imagen descargable (que sigue capturando
  el gráfico tal cual, barras incluidas).
- Publicar datos nuevos: todo sale de `series` y `projection`, ya publicadas.
- Carriles, selector o etiquetas fijas.

## Partes

- `web/dist/js/charts/series.js`: sin `bandSeries` en el histórico ni en la proyección; serie de barras por partido
  en `when` con realce; roles del tooltip: `mean` sin rango, `projection` y `bar` con rango y cabecera.
- `web/dist/js/polls-tooltip.js`: `meanTooltipLine` sin paréntesis para el histórico (nueva variante o parámetro).
- `web/dist/js/pages/series-wire.js` / `promedios.js`: nada, salvo que el bloque pase `lo`/`hi` a la barra (ya lo
  hace `blockProjection`).
- Plantillas: `promedios.html`, `eleccion.html`, `metodo_escanos.html`, `metodo_promedio.html` (textos arriba).
- Tests: `tests/test_js_charts.py` (sin bandas; barras sólo con proyección; ocultas con `sondeos`; una por bloque en
  la vista Bloques; realce configurado; tooltip del histórico sin rango), `tests/test_pages.py` (textos nuevos,
  `assert_public_wording` en /es, el detalle de una elección y las dos páginas del método).

## Comprobación

- `python -m pytest tests/test_js_charts.py tests/test_pages.py tests/test_pages_ejemplo.py -q` y la suite completa.
- Paseo de Luis por /es en Elección y en Hoy (`?mode=nowcast`), vista Bloques, selector Sondeos / Promedio / Ambos,
  la encuestadora frente al promedio y el detalle de 2023; en el móvil, el realce al tocar un punto.
