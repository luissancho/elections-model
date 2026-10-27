# Descarga de cada tarjeta entera como imagen (análisis)

**Fecha:** 2026-10-27 · **Páginas:** `/analisis/sesgo-cis` y `/analisis/voto-cera` · **Estado:** construido en `dev`.

## Intención

Luis quiere compartir cada tarjeta de los dos análisis (resumen, las tres preguntas, debate, comprobación, datos) como
una imagen: la tarjeta entera, tal como se ve en la web, con un icono cuadrado pequeño en la esquina superior derecha
de cada una que la descarga. Los botones de descarga que ya existen (`share.js`) fotografían un gráfico, no una
tarjeta: vuelven a dibujar la opción de ECharts a tamaño fijo bajo un título y una firma. Una tarjeta de estas páginas
lleva párrafos largos, cajas, una matriz, dos o tres gráficos, tablas y el resumen.

## Enfoques considerados

1. **Componer la tarjeta con las piezas de `share.js`** (texto envuelto en canvas, cajas, gráficos lado a lado). Sin
   dependencias y con el acabado de las imágenes actuales, pero cada tarjeta es una maqueta a mano y cualquier cambio
   de la tarjeta obliga a retocar la imagen. Descartado por coste.
2. **Rasterizar el DOM tal cual** con html2canvas. Un solo código para todas las tarjetas, pero la imagen depende del
   ancho de la ventana (estrecha en el móvil) y se cuelan el botón y el «Cómo leerlo».
3. **Rasterizar una copia de la tarjeta a ancho fijo** (elegido por Luis el 2026-10-27): html2canvas dibuja la página
   clonada a 1200 px de ancho de ventana (`windowWidth`), y en la copia se quitan los botones y el «Cómo leerlo», se
   vuelven a dibujar los gráficos al tamaño que les da esa anchura y se añade la firma. La imagen sale igual desde
   cualquier pantalla.

## Diseño

- **Librería vendorizada:** `web/dist/vendor/html2canvas-1.4.1.min.js` (MIT, `LICENSE-html2canvas.txt`), 194 KB, cargada
  sólo al primer clic (`loadHtml2canvas` inyecta el `<script>`); las páginas no pesan más si nadie descarga.
- **Módulo `web/dist/js/card-image.js`:** `wireCardImages(root, onError)` revela cada `button.card-img-btn` (oculto sin
  JS) y al pulsarlo guarda el PNG de su `.card` (`cardImage`): `html2canvas(card, {scale: 2, windowWidth: 1200,
  backgroundColor: '#fff', onclone})`. En `onclone` (`prepareClone`): fuera `.card-img-btn, .tool-btn, button.more,
  details.how` (`STRIPPED`); cada `.chart` con gráfico montado (`el.__chart`, `mountChart`) se vacía, se mide en la
  copia y se sustituye por un `<img>` de su opción (`chart.getOption()`) dibujada con `chartDataUrl` de `share.js` a
  ese tamaño y a 2x; se añade `<p class="shot-sign">` con la firma; se espera a `doc.fonts.ready`. Las filas ocultas
  de las tablas («Ver más») quedan como en pantalla.
- **Botón:** macro `card_image_button(file, sign)` de `_macros.html`, cuadrado de 28 px con el icono de descarga,
  absoluto en la esquina superior derecha (`.card.shot { position: relative }`; la cabecera de la tarjeta deja 36 px
  a la derecha). Cada plantilla define `shot(slug)` con el fichero `demoscopio-{estudio}-{slug}.png` y la firma
  «{título del estudio} · {site_host} · {author_handle}». Tarjetas: resumen, pregunta-1/2/3 y datos en las dos
  páginas; debate y comprobacion además en la del CERA.
- **CSS:** el fondo de `.sum` deja `color-mix()` (html2canvas 1.4.1 no lo entiende) por `--qc-soft`, el mismo tinte
  del 9 % escrito en hexadecimal por tono; `.shot-image` sin sombra en la copia; `.shot-sign` pequeña, a la derecha.
- **Despliegue:** `deploy/check-api.sh` y `deploy/smoke.sh` comprueban el nuevo fichero del vendor como el de ECharts.

## Pruebas

- `tests/test_pages.py::test_study_cards_carry_a_card_image_button`: un botón por tarjeta en las dos páginas, ficheros y
  firma, la clase `shot`, el módulo cableado en los dos scripts de página, el vendor y su licencia.
- `tests/test_js_card_image.py` (node): `CARD_IMAGE`, `STRIPPED` y `chartOptions`.
- Comprobación en Chrome sin cabeza (scripts en el scratchpad de la sesión, `check_app.py` + `check2.mjs`): las 12
  tarjetas descargadas a 2272 px de ancho sin errores de consola; gráficos, cajas, tablas y firma correctos.

## Lo que queda

- Luis pasea las dos páginas con datos reales y prueba las descargas en su navegador (y en el móvil).
- El rótulo «Izquierda» del eje vertical de las dianas sale igual que en pantalla, algo pegado al borde del lienzo;
  es cosa de `bullseye.js`, no de la imagen.
