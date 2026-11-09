# Detalle de un sondeo y su tarjeta para compartir — diseño

Fecha: 2026-11-09. Páginas: nueva `/{scope}/sondeos/{fecha}/{casa}[/{medio}]` (`controllers/Sondeos.py`,
`lib/pages/sondeos.py`, `web/templates/sondeo.html`, `web/dist/js/pages/sondeo.js`, `web/dist/js/poll-card.js`);
tocadas: la tabla «Todas las encuestas» de `/{scope}` (`_polls_table.html`, `polls-dom.js`, `promedios.py`), el
hemiciclo (`charts/hemicycle.js`) y la parte `polls` del contrato (`publish/forecast.py`, `docs/web/contrato.md`).

## Intención

Luis quiere que cada sondeo tenga su propia página en la web, con todo su detalle (casa, medio, rating, pesos, voto
por partido, los escaños que publicó la casa y la evolución de los sondeos de esa casa frente al promedio), y que el
lector pueda descargarla como **una sola imagen** para compartirla en redes, al estilo de la tarjeta de sondeo de
Electomanía (cabecera con casa, medio, fecha y nota; barras de voto; hemiciclo de escaños). Hoy un sondeo sólo se ve
como una fila de la tabla «Todas las encuestas» y como un punto del gráfico del promedio.

## Objetivos

1. Una URL por sondeo, estable y compartible, para cualquier ámbito publicado.
2. Una **tarjeta** al principio de la página que resume el sondeo y se descarga como una única imagen PNG.
3. Debajo, el detalle completo en tarjetas: voto frente al promedio del día, escaños de la casa, el peso del sondeo
   en el promedio y sus factores, la casa en este ciclo frente al promedio (con el sondeo resaltado) y la ficha.
4. Un enlace a la página desde cada fila de la tabla «Todas las encuestas».
5. Construirlo con lo que ya existe: tarjetas Jinja2, `card-image.js` para la imagen, los gráficos de ECharts
   (`bars.js`, `hemicycle.js`, `series.js`) y los datos del run publicado.

## No objetivos

- **Escaños del modelo para cada sondeo.** Decidido por Luis el 2026-11-09: sólo se muestran los escaños que
  publicó la propia encuestadora (`seats` de la parte `polls`). Si no los publicó, la página lo dice y no inventa.
- Página para los sondeos de una elección pasada (`events/{scope}/{date}/cycle.json`). El resolutor se escribe de
  forma que pueda leer esa parte más adelante, pero ahora sólo sirve los sondeos del run del evento en curso.
- `og:image` con la imagen del sondeo: la imagen se genera en el navegador del lector; no hay renderizado en el
  servidor (ECS no tiene navegador). El lector descarga el PNG y lo adjunta.
- Lista de los sondeos de una casa en su perfil (`/casas/{slug}`) con enlaces al detalle. Es natural, pero es
  otra tarea.
- Modo oscuro (pendiente en toda la web).

## Anti-objetivos

- Que la tarjeta parezca una proyección nuestra de escaños: el hemiciclo lleva siempre «escaños según
  {casa}» y los textos lo repiten.
- Que la imagen dependa del tamaño de la pantalla: sale igual desde el móvil y desde el ordenador.
- Ocultar el detalle tras controles: todo a la vista, como el resto de la web; los «Cómo leerlo» plegados.

## Restricciones

- Identificadores en inglés neutro; textos públicos en español con «sólo» con tilde, «encuestadora/casa» y
  «medio», nunca «empresa» ni «patrocinador» visible (`assert_public_wording`).
- Sin FastAPI ni scripts sueltos: controlador en `mtpy/controllers`, contexto en `mtpy/lib/pages`.
- nginx sólo sirve `/dist/`; la página es servida por la app como las demás.
- El 2026-11-09 Luis tiene cambios sin commit en 16 ficheros (rótulos de los bloques en las imágenes):
  `web/dist/js/share.js`, `charts/series.js`, `blocks.js`, `polls-tooltip.js`, `pages/promedios.js`,
  `pages/series-wire.js`, `pages/legend-dom.js`, `pages/evolution-wire.js`, `pages/elecciones.js`,
  `web/dist/css/site.css`, `web/templates/promedios.html`, `eleccion.html` y cuatro tests de JS
  (`test_js_charts.py`, `test_js_blocks.py`, `test_js_evolution.py`, `test_js_share.py`). Este trabajo **no toca
  ninguno de sus módulos de JS ni sus plantillas**: lo que necesite de ellos lo usa desde fuera (el anillo que
  resalta el sondeo se añade a la opción que devuelve `seriesOption` en el módulo de la página, sin cambiar
  `series.js`). Sí necesita escribir en dos ficheros que él también tiene abiertos: `site.css` (el CSS de la
  tarjeta) y `tests/test_js_charts.py` (los tests del hemiciclo). **La construcción arranca cuando Luis haya
  commiteado lo suyo**; si prefiere que sigamos encima, los commits de esas dos tareas incluyen sólo los trozos
  propios (`git diff` del fichero antes de añadirlo) y nunca los suyos.
- La `date` de un sondeo es **el último día de su trabajo de campo** (`loader.py`: `date = end_date`; no hay
  fecha de publicación distinta). La página nunca dice «publicado»: habla de «trabajo de campo» y de «fecha del
  sondeo», y la URL lleva esa fecha.
- `html2canvas` 1.4.1 no entiende `color-mix()`: el CSS nuevo de la tarjeta usa colores planos o `rgba`.
- Luis commitea en paralelo: `git status` antes de cada commit; nunca reescribir sus commits.

## Enfoques considerados

1. **Página servida + tarjeta `.card.shot` fotografiada con `card-image.js`** (elegido). La página se diseña de
   modo que su primera tarjeta *es* la imagen: html2canvas la dibuja a 1200 px de ancho, redibuja sus gráficos
   desde las opciones de ECharts y añade la firma. Un solo código ya probado (las tarjetas de los análisis), una
   sola imagen con todo el detalle, y la misma tarjeta se ve en la página.
2. **Imagen compuesta en canvas con `share.js`** (barras y hemiciclo apilados, tabla al lado, chips). Más nítida y
   sin html2canvas, pero cada pieza de la tarjeta (cabecera con medio y nota, dos columnas, fila de cifras) se
   maqueta a mano en canvas y no se corresponde con nada que el lector vea en la página. Descartado por coste, y
   porque `share.js` está en obras.
3. **Sin página: un botón por fila de la tabla** que genera la tarjeta al vuelo con los datos embebidos. Ahorra el
   servidor, pero no hay URL compartible ni sitio para la evolución de la casa ni para los pesos. Descartado: Luis
   quiere la página.
4. **Render de la imagen en el servidor** (una `og:image` real por sondeo). Exige un navegador sin cabeza en la
   tarea de ECS; fuera de alcance.

## Diseño

### URL y resolución

- Rutas nuevas en `pages.common.ROUTES`, después de `/{scope}`:
  `('/{scope}/sondeos/{date}/{pollster}', 'sondeos', 'detail')` y
  `('/{scope}/sondeos/{date}/{pollster}/{sponsor}', 'sondeos', 'detail')`. `date` es la fecha del sondeo
  (`date` de la parte `polls`, ISO: `2026-11-09`), `pollster` el slug de la casa (`slugify`, el mismo de
  `/casas/{slug}`) y `sponsor` el slug del medio (`slugify(sponsor)`). Ejemplo:
  `/es/sondeos/2026-11-09/nc-report/la-razon`; un sondeo sin medio: `/es/sondeos/2026-11-06/cis`.
- `pages.slugs.poll_detail_href(scope, poll)` construye la ruta a partir de un registro de la parte `polls`
  (`None` sin `date` o sin `pollster`). Es la única función que conoce la forma de la URL.
- `pages.sondeos.find_poll(polls, date, pollster, sponsor)`: entre los registros de la parte `polls`, los que
  tienen esa `date`, `slugify(pollster) == pollster` y `slugify(sponsor or '') == (sponsor or '')`. Si la URL no
  trae medio y ningún sondeo de la casa ese día carece de medio, pero hay **exactamente uno** con medio, se
  sirve ése (la canonical lleva la URL completa). Si hay varios candidatos, o ninguno, 404 `poll not found`
  («Sondeo no encontrado.», nueva entrada de `MESSAGES`). Una `date` que no cumple `check_date` es un 400
  `invalid date`.
- La página es una **página del evento** a efectos de estado: `common_context(state, None, event=True)`, de modo
  que `mode` y `run` (un run anclado, `?run=`) viajan como en `/{scope}`, la barra inferior enciende «Encuestas»
  y la barra del evento muestra la elección, la fecha y la pastilla del run. El índice de la barra del evento
  **enlaza a las secciones de `/{scope}`**: el contexto reescribe `event_bar['sections'][i]['href']` como
  `'/' + scope + query + '#' + id`, porque las anclas no existen en esta página.
- Canonical: `site_url + poll_detail_href(scope, poll)` (sin `mode` ni `run`, como las demás páginas).

### Datos

Todo sale del run de `state` (el último del ámbito o el anclado), como en la página del evento; la página no lee la
base de datos, así que funciona también en el paquete local sin `DB_*`:

- `polls.json`: el sondeo (`date, pollster_id, pollster, sponsor, start_date, end_date, sample_size, mtype,
  rating, weight`, los porcentajes por partido y `seats` cuando los publicó) y los demás sondeos de la misma casa
  en el ciclo (para el gráfico).
- `series.json`: el promedio de cada partido **el día del sondeo** (`series_at(series, date)`: el último día de
  `dates` ≤ `date`; `None` para un partido sin serie o una fecha anterior a la primera).
- `weights.json` (si el run la tiene): las filas del sondeo (misma `date`, `pollster_id` y `sponsor`) dan su
  `weight_kernel` y su `weight` en el promedio del día leído (`weights.date`). Se muestra el del partido con más
  voto en el sondeo que tenga fila; si el sondeo no tiene fila, está fuera de la ventana del kernel ese día.
- `house-effects.json` (si el run la tiene y `rows` no está vacío): `effect` de la casa por partido.
- `meta.json`: catálogo, `n_seats`, `majority`, `event_date`, `run_at`, `as_of`.

**Enmienda del contrato (parte `polls`, 2026-11-09):** `POLL_COLUMNS` pasa de 10 a 14 columnas: tras `weight` se
añaden `weight_over`, `weight_sample`, `weight_rating` (a 3 decimales, como `weight`; `weight` sigue siendo su
producto) y `url` (la fuente del sondeo tal como la guarda la tabla `polls`; `null` si no la hay). La serie del
`Forecaster` se recorta a sus `data_cols` (`forecaster.py`, ~línea 578), que hoy no incluyen `url`: se añade ahí,
como una columna de texto más (igual que `pollster` y `sponsor`), para que `fc_series_raw` la lleve; los tres
pesos ya están. `export_poll_rows` rellena con `null` cualquier columna que la serie no traiga, así que la
publicación no se rompe aunque falte alguna; `csv/polls.csv` y la parte `event-cycle` (que comparte
`POLL_COLUMNS`) ganan las mismas columnas. Los runs anteriores no las tienen: la página enseña entonces sólo el peso
total y una fuente genérica. Se actualizan `docs/web/contrato.md` (la frase «10 columnas» y la enmienda) y los
tests que fijan las columnas.

### La página

Orden de las tarjetas (ids entre paréntesis), todas con `card_start`/`card_end` y su «Cómo leerlo» plegado:

1. **La tarjeta** (`poll-card`, clase `shot`): la imagen. Boceto a 1200 px:

   ```
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ NC Report  para La Razón                                            [⤓]  │
   │ Trabajo de campo 5 – 7 nov 2026 · 1.000 entrevistas · Mixta               │
   │ Elecciones generales · 29 nov 2026                                        │
   │                                                                           │
   │  PP     ████████████████████▏32,6         │   Escaños según NC Report     │
   │  PSOE   ███████████████▏25,6              │        ╭───────────╮          │
   │  Vox    ███████████▏19,3                  │      ╱  PSOE  PP     ╲        │
   │  FA     ████▏6,9                          │     │ 99–101  136–138 │       │
   │  Otros  ██▏3,7                            │          350 · mayoría 176    │
   │  …                                        │                               │
   │  (marca oscura en cada barra: promedio del día)                           │
   │                                                                           │
   │  RATING 64,2 │ PESO EN EL PROMEDIO 0,84 │ DISTANCIA AL PROMEDIO 1,3 pts   │
   │                                                 demoscopio.es · @luis_sancho │
   └───────────────────────────────────────────────────────────────────────────┘
   ```

   - Cabecera: la casa en grande (enlazada a su perfil en la página; el enlace no se nota en la imagen), «para
     {medio}» si lo hay, y debajo una línea con el trabajo de campo (las dos fechas completas, `fmt_date`; una
     sola si coinciden o sólo se conoce una; «Fecha {date}» cuando no se conoce el campo), la muestra (`fmt_int`,
     «entrevistas») y el tipo (`mtype_label`); una segunda línea con el evento (título y fecha de `event_bar`).
   - Columna izquierda: **barras horizontales** de voto (`bars.js`, `renderBars`) en orden descendente del
     sondeo, color del catálogo, etiqueta a un decimal, y en cada barra la **marca del promedio del día** (el
     intervalo `[lo, hi]` de `bars.js` con `lo = hi = promedio`, que dibuja una línea con tapas; sin promedio no
     hay marca). Una fila «Otros» gris con `100 − suma` cuando ese resto es ≥ 0,5 (para que la tarjeta se lea
     como la de la casa). Un partido sin valor en el sondeo no tiene barra.
   - Columna derecha, **sólo si el sondeo trae `seats`**: el hemiciclo (`hemicycle.js`) con los escaños de la casa,
     titulado «Escaños según {casa}». Cada partido entra con el **punto medio** de su horquilla redondeado y su
     arco se rotula con el texto de la horquilla («136–138», o el número si mín = máx): `hemicycleOption` gana
     una opción `labels` (`{nombre: texto}`) que sustituye al número en el rótulo del arco (`hemicycleData`
     guarda el texto en cada ítem y `hemicycleLabel` lo prefiere al valor); sin `labels`, nada cambia. Si la suma
     de los puntos medios no llega a la cámara, un arco gris «Otros» completa el total para que la
     proporción sea la de la cámara (2026-11-09, revisión final: no son escaños sin asignar, sino sobre todo los
     que la casa da a partidos sin cifra propia en la tabla, más el redondeo de las horquillas; su tooltip lo dice y
     una línea pequeña dentro de la tarjeta, sobre las cifras, explica la marca de las barras y el gris; antes se
     llamaba «Sin asignar»); si se pasa, se dibuja tal cual (la tarta normaliza). La línea de la mayoría
     y el total del centro son los de `meta`. Sin `seats`, la columna no existe: las barras ocupan todo el ancho
     y una línea pequeña dice «{casa} no publicó escaños con este sondeo».
   - Fila de cifras (`m.kpis`): **Rating** (el `rating` del sondeo, a un decimal, en el color de la insignia de la
     tabla; un guion sin rating), **Peso en el promedio** (`weight`, a 2 decimales), **Muestra** (`fmt_int`) y
     **Distancia al promedio** (media de |sondeo − promedio del día| sobre los partidos con las dos cifras, a un
     decimal, «pts»; un guion si no hay promedio). Un guion va sin unidad (2026-11-09, revisión final). En la
     tarjeta no hay más texto que la línea de la marca (2026-11-09, revisión final): el resto va debajo.
   - Botón `card_image_button` en la esquina (fichero `demoscopio-sondeo-{scope}-{fecha}-{casa}[-{medio}].png`,
     firma `{site_host} · {author_handle}`; la firma la añade `card-image.js` como `.shot-sign`).
   - En el móvil las dos columnas se apilan; la imagen siempre se dibuja a 1200 px (html2canvas evalúa las
     media queries a `windowWidth`), así que sale igual desde cualquier pantalla. El hemiciclo calcula su radio,
     su centro y sus textos en píxeles del contenedor **en pantalla**, y `card-image.js` redibuja hoy cada
     gráfico con la opción de la pantalla (`getOption()`): desde un móvil saldría pequeño y descentrado.
     `renderHemicycle` deja en el contenedor `el.__imageOption = (width, height) => hemicycleOption(...)` y
     `card-image.js` (`chartOptions`, `prepareClone`) usa esa función, cuando existe, con el tamaño que la copia
     da al contenedor; sin ella, sigue como hoy (las barras y las series son relativas y no la necesitan).

2. **Voto y escaños** (`vote`): tabla servida con una fila por partido con valor en el sondeo, en el orden de las
   barras: Partido · Sondeo (%) · Promedio del día (%) · Diferencia (insignia `m.badge` con signo y color, como
   los «vs 2023» de la página del evento) · Efecto de casa (si el run tiene la parte; un guion para el partido
   sin fila) · Escaños (texto de la horquilla; la columna sólo existe si el sondeo trae `seats`). Pie: la fecha
   del promedio con que se compara («Promedio del {fecha}», que puede ser anterior al día del sondeo si la serie
   acaba antes).

3. **Peso en el promedio** (`weight`): `m.kpis` con los cuatro números del peso propio: **Solape**
   (`weight_over`), **Muestra** (`weight_sample`), **Rating** (`weight_rating`) y **Peso** (`weight`, el
   producto), a 2 decimales; un guion en los tres factores si el run es anterior a la enmienda. Debajo, una
   línea según el caso (sólo si el run trae la parte `weights`):
   - el sondeo tiene fila en `weights` → «En el promedio del {weights.date}: kernel {weight_kernel} · peso
     {weight} ({partido})»;
   - su `weight` es 0 → «Este sondeo no entra en el promedio: su peso es 0» y, si `weight_over` es 0, «: se
     solapa con otro sondeo de la misma casa publicado pocos días antes» (el `Forecaster` descarta los sondeos de
     peso 0; `weights_at` no los lista);
   - sin fila y con su fecha anterior al primer día de la recta de cualquier partido (`weights.line[p].dates[0]`)
     → «Este sondeo ya no pesa en el promedio del {weights.date}: su fecha está fuera de la ventana del kernel»;
   - sin fila en cualquier otro caso (un partido con menos de tres sondeos útiles, por ejemplo) → «Este sondeo
     no entra en la recta del promedio del {weights.date}».
   Frases propuestas; Luis las revisa con el documento. «Cómo leerlo» con los nombres que usa el ejemplo vivo
   del método (F(solape), F(muestra), F(rating), F(kernel)) y el enlace a `/metodo/promedio`.

4. **{Casa} en este ciclo** (`house`): el gráfico «Una encuestadora frente al promedio» de la página del evento con
   la casa fija: los sondeos de la casa como puntos sobre la línea del promedio, **este sondeo resaltado** con un
   anillo (una serie `scatter` más por partido, símbolo `circle` hueco de borde grueso en el color del partido,
   añadida a la opción que devuelve `seriesOption`/`seriesImageOption` en el módulo de la página; fuera de la
   leyenda). Controles: ventana (Todo el ciclo / 1 año / 6 meses / 3 meses) y, con bloques, Partidos / Bloques;
   la leyenda de pastillas (`fillLegend`); la línea, la proyección y la barra del final son las del modo de la
   página, exactamente como en el gráfico de la casa de `/{scope}` (la isla lleva las partes `projection` y
   `vote` del modo, como allí). Botón de descarga
   del gráfico con `wireDownload` (título «{casa} · {evento}», subtítulo «frente al promedio · {title_suffix}»,
   fichero `demoscopio-encuestadora-{scope}-{casa}-{fecha}.png`). Enlaces al perfil de la casa y al ranking.

5. **Ficha** (`sheet`): tabla de dos columnas: Encuestadora (enlace al perfil) · Medio · Tipo · Trabajo de campo
   (las fechas completas; sólo la fecha del sondeo, que es su último día, cuando no se conoce el campo) · Muestra
   · Rating · Peso · Fuente (enlace `url` con `rel="noopener"` si lo hay; si no, «Wikipedia, lista de sondeos») ·
   Elección · Actualización del modelo (`run_at`, y «run anclado» si `pinned`).

6. **Cierre** (`m.closing`): «¿Cómo pesamos cada sondeo?» → `/metodo/promedio`, con enlaces a «Todas las
   encuestas» (`/{scope}{query}#tabla`) y al perfil de la casa.

Arriba del todo, `<p class="back">← Todas las encuestas</p>` a `/{scope}{query}#tabla`. `<title>`: «{casa}
{para medio} · {fecha del sondeo} · {evento}»; descripción: «Sondeo de {casa}{ para medio} con trabajo de campo
hasta el {fecha} para {evento}: voto por partido frente al promedio, escaños publicados, rating y peso en el
promedio.»

### El módulo de la página

`web/dist/js/pages/sondeo.js` lee `#initial-data` (`state`, `meta`, `series` (`round_series`), `polls`
(`chart_polls` **sólo con los sondeos de la casa**), `projection` y `vote` (las partes del modo, para la línea y
la barra del final como en `/{scope}`), `catalog`, `inks`, `poll` (`date`, `pollster_id`, `sponsor`, los
porcentajes), `average` (`{partido: promedio del día}`), `seats` (`{partido: [mín, máx]}` o `null`), `n_seats`,
`majority`), dibuja las barras, el hemiciclo y el gráfico de la casa, cablea los controles del gráfico, su descarga
(`wireDownload`) y la de la tarjeta (`wireCardImages`). La lógica pura va en `web/dist/js/poll-card.js`, probada con
`node`: `voteRows(poll, parties, average)` (filas ordenadas con «Otros»), `pollSeats(seats, nSeats)` (`{seats,
labels, rest}`: puntos medios, textos de horquilla y el resto sin asignar), `distanceToAverage(poll, average,
parties)` y `highlightSeries(poll, parties, colors)` (las series del anillo).

### Enlaces de entrada

- `promedios.chart_polls` añade a cada sondeo `detail_href` (`poll_detail_href(scope, poll)`; hace falta pasarle
  el `scope`) y `polls_table` lo añade a cada fila. En `_polls_table.html` y en `polls-dom.js` (`renderBody`) la
  celda **Fecha** pasa a ser un enlace al detalle cuando el sondeo lo trae; los sondeos de una elección pasada
  (`eleccion.html` comparte la tabla y la función) no lo traen y la celda queda como hoy. El `title` de la celda
  (las fechas del campo) se conserva.
- `readTable` de `share.js` lee `textContent`, así que el enlace no cambia la imagen de ninguna tabla.

### Comprobación de despliegue

`deploy/check-api.sh` añade la ruta a sus comprobaciones (`/es/sondeos/2020-01-01/nadie` → 404 html;
`/es/sondeos/abc/nadie` → 400 html). No hay ruta nueva en la API ni en nginx.

## Pruebas

- `tests/test_pages.py`: la página de un sondeo de la fixture (con y sin medio, con y sin `seats`) se sirve con
  cabeceras de caché y canonical; sus tarjetas e ids; la isla `initial` con sólo los sondeos de la casa, `average`
  y `poll`; el botón de la tarjeta con su fichero y firma; la fila de cifras; la tabla de voto con la diferencia
  y los escaños; la tarjeta del peso con los tres factores (y con guiones en un run sin ellos); la ficha con la
  fuente; el índice de la barra del evento apuntando a `/{scope}…#…`; `?run=` anclado conservado en los enlaces;
  404 para un sondeo que no existe y para un slug ambiguo; 400 para una fecha inválida; la celda Fecha enlazada
  en `/{scope}` y sin enlace en la elección pasada; `assert_public_wording` sobre la página.
- `tests/test_pages.py` (unitarios de `pages.sondeos`): `find_poll` (exacto, sin medio con uno solo con medio,
  ambiguo, ninguno), `series_at` (día exacto, día anterior, fecha anterior a la serie, partido sin serie),
  `poll_detail_href` (con y sin medio, casa sin slug → id), `distance` y las filas de la tabla.
- `tests/test_publish_unit.py`: las 14 columnas de `POLL_COLUMNS` en `polls.json`, en `csv/polls.csv` y en
  `event-cycle`; `weight == weight_over · weight_sample · weight_rating` redondeado; `url` nulo cuando la serie no
  lo trae.
- `tests/test_js_charts.py`: `hemicycleOption` con `labels` rotula el texto y sin `labels` no cambia; el arco
  «Otros» (antes «Sin asignar»), su tooltip y el anillo de la mayoría que sigue sumando la cámara; los rótulos de
  horquilla encogen con el radio.
- `tests/test_js_card_image.py`: `chartOptions` devuelve la función `imageOption` del contenedor que la tiene y
  `null` en los demás; la copia la usa con el tamaño de la imagen.
- `tests/test_js_poll_card.py` (nuevo, `node`): `voteRows`, `pollSeats`, `distanceToAverage`, `highlightSeries`.
- `tests/test_forecaster_unit.py`: `fc_series_raw` lleva `url` cuando los sondeos la traen.
- Comprobación en Chrome sin cabeza (como la del 2026-10-27): abrir la página de un sondeo con escaños del
  bundle local, pulsar el botón de la tarjeta y comprobar el PNG (2400 px de ancho, dos columnas, barras,
  hemiciclo rotulado, cifras y firma) sin errores de consola; repetir con un sondeo sin escaños y en una ventana
  de 390 px.

## Lo que queda al constructor

- La forma exacta del anillo de resalte (grosor, tamaño) y de la marca del promedio en las barras.
- Si `fc_series_raw` no trae `url`, cómo llega del `Polls` a la serie (o se deja nulo y se anota en el contrato).
- El ancho de la columna del hemiciclo en la tarjeta a 1200 px (orientativo: 60 / 40).
- Los textos de los «Cómo leerlo», con las reglas de redacción de Luis.

## Pasos de Luis después de la fusión

1. `update`/`compute`+`publish` de los ámbitos para que los runs lleven las cuatro columnas nuevas (los runs de
   relleno anteriores siguen sirviendo la página con el peso total).
2. Pasear la página de dos o tres sondeos (con y sin escaños) en el ordenador y en el móvil, y probar la descarga.
3. Imagen y `update-service` en ECS.

## Cambio de Luis, 2026-11-09 (tras la fusión)

- La tarjeta del sondeo (y su imagen) sólo da lo que dice el sondeo: sin la marca del promedio en las barras (ni en su
  tooltip), sin flechas y sin la fila de cifras. La nota sólo explica el arco gris de los escaños, o dice que la casa
  no publicó escaños.
- Rating y peso en el promedio pasan a la cabeza de la tarjeta «Peso en el promedio», sobre F(solape), F(muestra) y
  F(rating). La muestra ya está en la entradilla; la distancia al promedio desaparece.
- La tabla «Voto y escaños» queda en: % Voto · vs Anterior · Escaños (con horquilla) · vs Anterior (puntos medios de
  las horquillas) · Efecto de casa (con su explicación en la cabecera y en «Cómo leerlo»). Las comparaciones con el
  promedio se ven en el gráfico de la casa en el ciclo.
- El anillo del sondeo en ese gráfico pasa de 16 a 11 px (borde de 2 px) para no tapar los puntos vecinos.
- Se retiran `series_at`, `series_day`, `distance_to_average`, las opciones `intervalText` y `labelExtra` de
  `renderBars` y el `average`/`previous` de la isla de datos.
