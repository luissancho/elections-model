# Página del evento: «Los cambios de cada casa» (cada encuestadora frente a su entrega anterior) — diseño corto

Fecha: 2026-11-11. Tarjeta nueva en la sección Promedio de la página del evento (`/{scope}`,
`web/templates/promedios.html`). Maqueta aprobada por Luis:
`.superpowers/brainstorm/97250-1794407404/content/matriz-v3.html` (fuera de git).

## Intención

Luis quiere otra manera de leer la tendencia: lo que hace cada casa **frente a sí misma**. Si la mayoría de las
encuestadoras, también las que dan a un partido cifras bajas, lo suben frente a su entrega anterior, eso dice algo
que el promedio puede tapar cuando entran casas nuevas que lo dejan más bajo (cambia la mezcla, no la opinión). El
modelo corrige el nivel de cada casa (M6) sólo con 5 sondeos en el ciclo o 3 a 90 días o menos de una elección pasada; las
casas con pocos sondeos entran sin corregir.

Lo quiere **sólo mostrado, sin estadística**: para cada casa con al menos dos sondeos en las últimas cuatro
semanas, el cambio de cada partido frente a su entrega inmediatamente anterior, tenga un día o veinte de distancia
(«sin comparaciones, simplemente mostrar»), en voto y en escaños.

## Objetivos

1. **Tarjeta** `#house-changes` dentro de `<section id="promedio">`, justo después de la tarjeta «Promedio de
   sondeos». Igual en los modos Hoy y Elección. Si ninguna casa cumple la regla, la tarjeta no se pinta.
2. **Filas.** Una por *línea* (misma casa y mismo medio, la definición de `previous_poll` del detalle de un sondeo)
   con **al menos dos sondeos con `date` dentro de los 28 días que acaban en `meta.as_of`** (`date >= as_of − 28
   días`; la `date` de un sondeo es su último día de campo). La fila compara la última entrega de la línea con la
   inmediatamente anterior (`previous_poll`), sin distancia mínima ni máxima: las tabulaciones diarias comparan días
   consecutivos y los sondeos con peso 0 por solape cuentan como cualquier otro. Orden: rating descendente (sin
   rating, al final), empate por nombre.
3. **Primera columna** (fija al desplazar): nombre de la casa enlazado a su perfil, su rating con la insignia de
   siempre (un decimal, color de `gradient` sobre `RATING_COLORS`, como «Todas las encuestas») y debajo
   «{medio} · {fecha anterior} → {fecha última} ({n} días)», con cada fecha enlazada al detalle de su sondeo.
4. **Celdas.** Una columna por partido, en el orden de «Todas las encuestas» (`table_parties`), sólo los partidos
   con alguna cifra en las filas de la vista. Arriba el cambio y debajo las dos cifras, «31,1 → 31,0»:
   - sube o baja: la flecha de la web (`m.badge`: «▲ 0,6», «▼ 0,1», verde y roja de `.dlt`), todas del mismo tamaño;
   - no cambia: «=» en gris;
   - la última entrega da cifra y la anterior no: sólo la cifra, sin flecha;
   - la última no da cifra: una raya.
5. **Selector Voto / Escaños** (`.seg`, como el de Evolución), sólo si alguna fila tiene escaños. En Escaños las
   cifras son **las que publica la propia casa** (`poll['seats']`, nunca un reparto del modelo), con su horquilla
   («133–139 → 131–137», `seats_text`), y el cambio es la diferencia de los puntos medios exactos de las dos
   horquillas: el medio escaño cuenta y se muestra redondeado a un escaño entero. Las líneas cuya última entrega no
   publica escaños no salen en esa vista.
6. **Fila final** «Suben · bajan · igual»: por partido, cuántas filas lo suben, lo bajan y lo dejan igual
   («▲ 3 ▼ 2 = 2»), contando sólo las celdas con cambio calculable.
7. **Imagen.** Botón de descargar la tarjeta (`m.card_image_button`, clase `shot`), con la vista que esté a la
   vista; la imagen recoge la tabla entera, sin columnas cortadas.
8. **«Cómo leerlo»** con el texto de abajo (provisional: lo revisa Luis).

## No objetivos

- Media ponderada de los cambios, línea del promedio, pruebas de significación o cualquier resumen más allá del
  recuento de la fila final.
- Distancia mínima o máxima entre entregas, o ventana fija (opciones 2 y 3, descartadas por Luis).
- Triángulos proporcionales al cambio y los gráficos «De dónde a dónde» y «Suben y bajan» de las maquetas
  (descartados por Luis: «me basta con la A»).
- Cambiar el modelo, el contrato del bundle o exigir una republicación: todo sale de la parte `polls` del run.
- La tarjeta en el detalle de una elección pasada, en `/casas` o en el detalle de un sondeo.
- Guardar la vista (Voto / Escaños) en la URL.

## Anti-objetivos

- Que una fila compare entregas de medios distintos o de otro ciclo.
- Que una cifra que falta se lea como «no cambia» (la raya y el «=» no se confunden, ni en la celda ni en el recuento).
- Que la imagen descargada corte partidos.

## Enfoques

1. **En el servidor, al pintar la página** (elegido): una función de `mtpy/lib/pages` construye las filas con los
   sondeos que `promedios_context` ya tiene (`poll_rows`) y reutiliza `previous_poll`, `_seats_mid`, `seats_text`,
   `gradient` y `m.badge`. Funciona sin JavaScript (el navegador sólo cambia de vista) y las reglas de «entrega
   anterior» viven en un único sitio.
2. En el navegador, con `initial.polls`: los datos están, pero habría que duplicar en JavaScript `previous_poll` y
   los puntos medios, y mantener los dos espejos. Descartado.
3. Una parte nueva del bundle (`house-changes`): cambia el contrato y obliga a republicar para algo que se deduce
   de `polls`. Descartado.

## Diseño

- `mtpy/lib/pages/sondeos.py`: `HOUSE_CHANGES_DAYS = 28` y
  `house_changes(polls, parties, colors, as_of, scope, days=HOUSE_CHANGES_DAYS) -> Optional[dict]`, junto a
  `previous_poll`. Devuelve `None` sin filas; si no,
  `{'days', 'pct': vista, 'seats': vista | None}` con
  `vista = {'parties': [{'name', 'color'}], 'rows': [...], 'counts': {partido: {'up', 'down', 'flat'}}}` y cada fila
  `{'pollster', 'href', 'rating', 'rating_color', 'sponsor', 'prev_date', 'last_date', 'gap_days', 'prev_href',
  'last_href', 'cells': [{'prev', 'value', 'change'} | None]}` (textos ya formateados salvo `change`, que va a
  `m.badge` con 1 decimal en voto y 0 en escaños). El cambio de voto es `round(última − anterior, 1)`; el de
  escaños, la diferencia de los puntos medios exactos de las dos horquillas,
  `((bajo2 + alto2) − (bajo1 + alto1)) / 2`, que puede acabar en ,5: el medio escaño cuenta y se muestra redondeado
  a un escaño entero. Con docstring y nombres en inglés.
- `promedios_context` añade `context['house_changes']` (importación local de `sondeos`, como ya hace con
  `elecciones`: `sondeos` importa `promedios`).
- `web/templates/promedios.html` (o un parcial `_house_changes.html`): la tarjeta con `m.card_start(...,
  classes='shot')`, el selector oculto hasta que el navegador lo active, las dos tablas (la de escaños con `hidden`)
  dentro de un contenedor con desplazamiento lateral, y `m.how()`.
- `web/dist/css/site.css`: tabla de la matriz (primera columna `position: sticky`, cifras tabulares, línea pequeña
  bajo el cambio, pie con el recuento); la insignia `.rt` hoy sólo existe bajo `.et`: generalizarla o repetirla.
- `web/dist/js/pages/promedios.js`: mostrar el selector, alternar las dos tablas y cablear el botón de la imagen
  (`wireCardImages`).
- Tests: unitarios de `house_changes` (borde de la ventana, línea por medio, tabulación diaria con un día de
  distancia, cifra ausente en una de las dos entregas, puntos medios de horquillas, sin filas → `None`, orden por
  rating, recuento) y de la página (la tarjeta está con filas y no está sin ellas; sin escaños no hay selector);
  los textos pasan `assert_public_wording`. `tests/test_pages.py` tiene cambios de Luis sin commit: no tocarlos.

### Textos (provisionales)

- Título: «Los cambios de cada casa». Entradilla: «Cuánto cambia cada partido entre las dos últimas entregas de
  cada encuestadora con al menos dos sondeos en las últimas cuatro semanas.»
- Cómo leerlo:
  1. «Cada fila es una encuestadora, con el medio para el que publica, que ha terminado al menos dos sondeos en las
     últimas cuatro semanas. La flecha dice cuánto ha movido a cada partido entre su entrega anterior y la última, y
     debajo van las dos cifras: verde si sube, roja si baja y un signo igual si repite. Una raya quiere decir que
     la última entrega no da cifra para ese partido; una cifra sola, sin flecha, que la anterior no la daba.»
  2. «La comparación es siempre de una casa consigo misma. No importa que una encuestadora suela dar a un partido
     más o menos que las demás: aquí sólo se ve hacia dónde lo mueve. Si casi todas lo mueven en la misma
     dirección, es una señal que no depende de qué casas hayan publicado esa semana.»
  3. «Cada cambio por separado dice poco: entre dos sondeos de mil entrevistas, un punto arriba o abajo cabe en el
     margen de error. Y los periodos no son iguales: bajo el nombre van las fechas de las dos entregas y los días
     que las separan. Las encuestadoras que publican a diario sobre muestras que se solapan suelen moverse muy
     poco de un día a otro. Mira el conjunto, no una celda.»
  4. «En Escaños, las cifras son las que publica cada encuestadora, no un reparto nuestro. Cuando da una horquilla,
     la flecha compara sus puntos medios. Las que no publican escaños no aparecen.»

## Criterios de aceptación

Con los datos de `es` del 2026-11-11 (run con `as_of` 2026-11-11):

- Voto: 7 filas, en este orden: Sigma Dos (65,9), IMOP, SocioMétrica, Sondaxe, NC Report, SW Demoscopia, Ateneo
  del Dato. Sigma Dos / PP: «▼ 0,1» y «31,1 → 31,0». SW Demoscopia compara el 8 con el 9 de noviembre (1 día).
  Recuento del PP «▲ 3 ▼ 2 = 2»; del PSOE «▲ 5 ▼ 1 = 1».
- Escaños: 6 filas (sin Sondaxe). Sigma Dos / PP: «▼ 2» y «133–139 → 131–137»; IMOP / PSOE: «▼ 7» y «113 → 106».
- GAD3, 40dB y las demás casas con una sola entrega en la ventana no salen.
- En un ámbito sin ninguna línea con dos entregas en 28 días, la página no tiene `#house-changes`.
- La imagen descargada contiene los 13 partidos en las dos vistas.

## Decisiones

De Luis (2026-11-11): la matriz (A); entrega inmediatamente anterior sin límites de distancia; casas con dos o más
sondeos en 3-4 semanas; las dos cifras con flecha bajo cada cambio; rating con su color; flechas de un solo tamaño;
versión de escaños. Aprobó además, sobre la maqueta: ventana de 28 días hasta el día de la actualización; línea =
casa y medio; escaños de la propia casa con puntos medios y sin filas vacías; todos los partidos con la columna de
la casa fija; orden por rating y fila de recuento; debajo de «Promedio de sondeos»; enlaces al perfil y al detalle;
botón de imagen; título provisional.

Mías, por coherencia con el resto de la web (corregibles): la flecha usa el formato de `m.badge` («▲ 0,6», sin
signo; la maqueta ponía «▲ +0,6»); las columnas de partidos sin ninguna cifra en la vista se omiten; la vista no
va a la URL; sin JavaScript se ve sólo la tabla de voto.

## Queda a quien lo construya

- Parcial aparte o tarjeta dentro de `promedios.html`; nombres de las clases CSS.
- Cómo consigue la imagen la tabla entera: `cardImage` admite `width`, así que puede pedirse más ancha que los
  1.200 px habituales cuando la tabla no quepa, o compactar las columnas en la copia.
- Anchuras en el móvil (primera columna estrecha, como los 124 px de «Todas las encuestas» bajo 720 px).
