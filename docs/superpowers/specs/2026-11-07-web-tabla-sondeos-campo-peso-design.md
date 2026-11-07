# Web, «Todas las encuestas»: fechas de campo, columna «Peso» y desglose del peso

Cambio pequeño en la tarjeta «Todas las encuestas», compartida por Promedios (`/{scope}`) y el detalle de una
elección pasada (`/elecciones/...`). Aprobado por Luis el 2026-11-07 («Aprobado»), en tres mensajes: las fechas de
campo, la columna «Peso» y el tooltip con el desglose.

## Intención

Luis quiere que la celda «Fecha» enseñe los días del trabajo de campo («29–30 Oct», «1 Sep–1 Nov») en vez del fin
del campo en `dd/mm/aa`; una columna «Peso» a la derecha de «Rating (0-100)» con lo que el sondeo pesa en el
promedio, con barra y cifra como las del ranking; y, al pasar por la cifra, un tooltip con el desglose del peso por
factores.

## Hechos de partida (2026-11-07)

- `date` de cada sondeo es ya el fin del campo (`loader`: `data['date'] = data['end_date']`). `start_date` y
  `end_date` viajan en la parte `polls` y en los datos del navegador (`chart_polls`); hoy sólo salen en el tooltip
  «Trabajo de campo: 28 sept 2026 – 30 sept 2026».
- La tarjeta se pinta dos veces: el servidor renderiza las 15 filas más recientes (`polls_table`,
  `web/templates/_polls_table.html`) y el navegador repinta al filtrar, ordenar o paginar (`polls-table.js`,
  `pages/polls-dom.js`, `pages/polls-wire.js`). Todo cambio va en los dos sitios con el mismo texto.
- La parte `polls` lleva `weight` (3 decimales): `weight_over · weight_sample · weight_rating` de
  `Forecaster.fc_series_raw`, con el factor de dispersión de la casa ya dentro de `weight_sample`
  (`apply_dispersion` al ajustar). Es el «peso base» del método, sin la cercanía, que depende del día leído. En
  `es` (run 20261008) va de 0 a 5,06, mediana 1,71; cuatro sondeos pesan 0 (trackings pisados por una entrega
  posterior, `weight_over = 0`). No viaja al navegador ni hay desglose en la parte.
- La parte `event-cycle` (ciclo de una elección pasada) tiene la misma forma, pero su `weight` es
  `weight_over · weight_sample` del `Computer` (sin rating ni dispersión); lo leen los estudios.
- Los selectores Desde/Hasta de la misma tarjeta escriben los meses con mayúscula y «Sep» (`MONTH_LABELS` de
  `promedios.py`, `MONTHS` de `polls-table.js`); el resto de la web usa `oct`, `sept`.
- El ranking (`/casas`) pinta el rating como `td.cell-bar` con `i.fill.pos` y la cifra; el ejemplo vivo del método
  pinta el peso con `i.fill.w` (azul) de 0 a 3 y dos decimales.

## Objetivos y límites

- Mismo aspecto y mismo texto en las filas del servidor y en las del navegador; la ordenación por fecha sigue
  siendo por fin de campo (`date`).
- Las descargas CSV ganan las columnas nuevas y no pierden ninguna. Ninguna imagen descargable incluye esta tabla.
- No cambia ningún peso ni promedio: sólo se publica y se enseña lo que el modelo ya calcula. El `weight` de
  `event-cycle` no cambia de definición.
- Los runs y eventos ya publicados no llevan las columnas nuevas: en ellos la columna Peso sale con barra y cifra
  pero sin tooltip, y nada falla.
- Redacción pública: «encuestadora» o «casa», nunca «empresa»; «sólo» con tilde.

## Enfoques considerados

- **Servidor y navegador en paralelo, desglose como enmienda de la parte `polls`** (elegido): es la arquitectura
  de la tarjeta y las componentes del peso están en la serie que ya exporta la parte.
- Desglose calculado en la página a partir de las partes `dispersion` y `weights`: `weights` sólo trae los sondeos
  de la ventana de un día y por partido; descartado.
- Tres factores con la dispersión dentro de la muestra (como la tabla del ejemplo vivo): más simple, pero Luis
  aprobó los cuatro de la fórmula «Peso base = Muestra × Dispersión × Frecuencia × Rating» del método; el
  exportador conoce el factor por casa y puede separarlo.
- Escala fija de la barra (0 a 3, como el método): en `es` saturaría los pesos de 3 a 5; elegida la escala 0 al
  mayor peso del ciclo, sobre todos los sondeos de la parte, para que las barras no salten al filtrar.

## D1. Fechas de campo en la celda «Fecha»

- Etiqueta `fieldwork_label(start, end, date)` en `pages/promedios.py` y `fieldworkLabel(start, end, date)` en
  `polls-table.js`, con los meses de `MONTH_LABELS`/`MONTHS` («Sep», «Oct») y el guión corto «–» sin espacios:
  - inicio y fin válidos, mismo día: «30 Oct»; mismo mes: «29–30 Oct»; meses distintos: «1 Sep–1 Nov», también al
    cambiar de año («28 Dic–3 Ene»);
  - sólo una de las dos: esa; ninguna: `date`; nada válido: el guión `–`. Una fecha ISO inválida cuenta como
    ausente. Sin año en la celda: el tooltip «Trabajo de campo: …» con los años se conserva tal cual.
- `polls_table` escribe la etiqueta en `row['date']`; `renderBody` la usa en lugar de `shortDate`. `short_date` y
  `shortDate` desaparecen si nadie más los usa (el constructor lo comprueba).
- La cabecera «Fecha» gana un `title`: «Días del trabajo de campo del sondeo». `col.c-d` pasa de 80 a 104 px (la
  etiqueta más larga, «30 Sep–31 Oct»).

## D2. Columna «Peso»

- `chart_polls` añade a las claves que viajan al navegador `weight`, `weight_sample`, `weight_disp`,
  `weight_over` y `weight_rating` (sólo las que el sondeo tiene, como hoy).
- `polls_table` calcula `wmax`, el mayor `weight` de todos los `polls` (no sólo de las filas mostradas), y cada fila
  lleva `weight` (dos decimales, `fmt_num`, o el guión), `weight_bar` (`bar_width(weight, 0, wmax)`; 0 sin `wmax`)
  y `weight_title` (D4, o `None`). En el navegador, `maxWeight(polls)` en `polls-table.js`, calculado una vez en
  `wirePollsTable` y pasado a `renderBody`.
- Plantilla: `<col class="c-w">` tras `c-r`; cabecera `<th scope="col" data-key="weight" title="…"><button
  type="button" class="sort">Peso</button></th>` tras el rating; celda `<td class="cell-bar"
  title="…"><i class="fill w" style="left: 0; width: X%"></i><span>1,71</span></td>`; con peso 0 la celda lleva
  además `na` (cifra en gris, barra vacía); sin peso, `<td class="na">—</td>`. Las filas del pie (media simple y
  fijas) ganan un `<td></td>` más. Mismo DOM en `polls-dom.js` (`FIXED_COLUMNS` con `['weight', 'Peso']`,
  `renderFoot` con cuatro celdas vacías); `FIXED_KEYS` de `polls-wire.js` y las claves fijas de `sortPolls` ganan
  `weight` (ordena como `rating`: numérico, ausentes al final).
- Título de la cabecera: «Peso del sondeo en el promedio: muestra × dispersión × frecuencia × rating de la
  encuestadora, sin la cercanía al día estimado. La barra va de 0 al mayor peso del ciclo.» CSS: `.et col.c-w {
  width: 96px; }`.
- El párrafo de lectura de la tabla (en `promedios.html` y `eleccion.html`) gana una frase sobre la columna: en
  Promedios, que es lo que el sondeo pesa en el promedio del modelo, el producto de los cuatro factores del método
  antes de la cercanía, y que la cifra enseña el desglose al pasar por ella; en el detalle de una elección pasada,
  que el peso del ciclo recoge la muestra y la frecuencia, y el rating de entonces va aparte en su columna.

## D3. Enmienda de las partes `polls` y `event-cycle` (`publish/forecast.py`, `publish/events.py`)

- `POLL_COLUMNS` pasa de 10 a 14 columnas: `…, rating, weight, weight_sample, weight_disp, weight_over,
  weight_rating`, las cuatro nuevas a 3 decimales como `weight` (`export_poll_rows`; una columna ausente sigue
  saliendo nula). `weight_sample` conserva el sentido que tiene en la parte `weights` (con la dispersión de la
  casa dentro); `weight_disp` es el factor de dispersión de la casa; `weight = weight_sample · weight_over ·
  weight_rating` y la muestra sin la dispersión es `weight_sample / weight_disp`.
- Run (`export_polls`): `weight_over`, `weight_sample` y `weight_rating` de `fc.fc_series_raw`; `weight_disp` de
  `fc.dispersion['factor']` por `pollster_id`, 1 para una casa sin factor o cuando `fc.dispersion` es `None`.
- Ciclo (`export_event_cycle`): `weight_over` y `weight_sample` del frame; `weight_disp` y `weight_rating` siempre
  nulos, aunque el frame traiga `weight_rating`, porque su `weight` no los lleva.
- `docs/web/contrato.md`: enmienda 2026-11-07 en `### polls` (14 columnas, su sentido, los runs anteriores no las
  tienen) y en la fila de `event-cycle`; gemelos CSV ampliados.

## D4. Tooltip del desglose

- Texto `weight_breakdown(poll)` en `promedios.py` y `weightBreakdown(poll)` en `polls-table.js`: una línea por
  factor, en este orden y sólo los que el sondeo trae (valor numérico): «Muestra» (`weight_sample / weight_disp`
  si `weight_disp` es un número positivo, si no `weight_sample`), «Dispersión» (`weight_disp`), «Frecuencia»
  (`weight_over`), «Rating» (`weight_rating`); formato «{Factor} ×{valor a 2 decimales}», líneas separadas por
  `\n`, en el atributo `title` de la celda. Sin ningún factor, sin `title`. Ejemplo:

  ```
  Muestra ×1,41
  Dispersión ×1,00
  Frecuencia ×1,00
  Rating ×2,40
  ```

  Un sondeo pisado enseña «Frecuencia ×0,00». En un ciclo pasado salen sólo «Muestra» y «Frecuencia».

## D5. Pruebas y comprobación

- Python: `fieldwork_label` (los seis casos), `weight_breakdown` (cuatro factores, parciales, ninguno, cero),
  `polls_table` (etiqueta, peso, barra sobre todos los sondeos, título), la página (`data-key="weight"`, cuatro
  `<td></td>` en el pie, claves nuevas en los datos del navegador, frase del párrafo), `export_polls` y
  `export_event_cycle` (columnas, `weight_disp` de la dispersión y nulos del ciclo), contrato.
- JS con `node` (`tests/test_js_polls.py`): `fieldworkLabel`, `weightBreakdown`, `maxWeight`, `sortPolls` por
  `weight`, y el DOM de `renderBody`/`renderFoot` si ya hay pruebas de él.
- Navegador: Promedios `/es` y una elección pasada con el bundle local: columnas, barras, tooltip, filtros,
  orden por Peso, paginación; nada cambia en las descargas de imágenes.

## Lo que queda al constructor

Dónde viven exactamente los helpers nuevos, los fixtures que ganan las claves nuevas, los anchos finales de las
columnas y la redacción exacta de las dos frases del párrafo de lectura, dentro de las reglas de redacción.

## Después, para Luis

Desplegar la imagen. El siguiente `update` de ECS republica `es` con las columnas nuevas (tooltip completo);
`publish --what events` vuelve a publicar los ciclos de las elecciones pasadas (tooltip con muestra y frecuencia).
Los runs anteriores fijados por `?run=` siguen sin tooltip.
