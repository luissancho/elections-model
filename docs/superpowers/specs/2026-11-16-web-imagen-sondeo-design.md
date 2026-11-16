# Web: imagen propia en el enlace de un sondeo

Séptimo paso de la ronda de SEO y redes sociales, pedido por Luis el 2026-11-16 («Vamos con la 2»: la imagen con
cifras en las páginas que no cambian, empezando por los sondeos). Lo que eligió él y lo que propuse yo están
separados más abajo.

## Intención

Durante la campaña del 29-N lo que más se comparte en X son los sondeos. Hoy el enlace a un sondeo sale con la
imagen genérica de la marca, igual que cualquier otra página. Queremos que salga con **sus cifras**: quien ve la
tarjeta en X, Facebook o WhatsApp lee el sondeo sin entrar, con la marca de Demoscopio encima.

La imagen genérica se eligió por la caché de X (guarda la vista previa de cada URL unos siete días), y Luis la
quiso «al menos para el Promedio». Un sondeo publicado no cambia, salvo corrección: ahí la caché no molesta.

## Hechos de partida (2026-11-16)

- Toda página declara `og:image` con la imagen de marca (`pages.og_image_url()`); `Page.render` admite otra en
  `og_image=`. En `base.html` el tipo, el tamaño (1200×675) y el texto alternativo de `og:image` están fijos.
- La tarjeta que se descarga de la página de un sondeo se dibuja en el navegador (html2canvas + ECharts) y su alto
  crece con los partidos. Las redes no ejecutan JavaScript al leer un enlace.
- El servidor tiene Pillow 10.3 (`requirements.txt`, la misma imagen de Docker para la web y los jobs). La imagen
  de marca se dibujó con Pillow en `web/brand/build.py`, que no es código de la web.
- Las fuentes del sitio sólo existen como woff2 variables (`web/dist/vendor/fonts/`), que Pillow no lee sin
  `brotli`, que no es dependencia de la web. Son SIL OFL.
- Las rutas son expresiones regulares y los segmentos de un sondeo (`[0-9a-z_\-]+`) no admiten puntos.
- Un controlador puede devolver `bytes`.
- nginx guarda en caché las respuestas 200 de la aplicación (60 s por defecto; respeta el `Cache-Control` de la
  respuesta; suelta lo que nadie pide en 10 minutos) y limita a 20 peticiones por segundo por IP.
- En las generales hay 365 sondeos publicados, 109 sin escaños; un sondeo trae entre 3 y 13 partidos; el nombre
  más largo es «Ateneo del Dato para elDiario.es». Algún sondeo no trae muestra.
- Medido en el Mac de Luis: dibujar y codificar una imagen de 1200×675 cuesta entre 25 y 75 ms y pesa entre 70 y
  270 kB.

## Enfoques

1. **El servidor dibuja la imagen cuando se la piden** (elegido por Luis, tras preguntar por el rendimiento
   frente a la 2 y ver las medidas). Sin almacén ni republicación; siempre de acuerdo con la página; si un sondeo
   se corrige, la imagen cambia sola. Sólo la piden las redes, una vez por enlace.
2. Dibujarla al publicar y guardarla en S3. Descartado: nginx sólo sirve `/dist/`, así que servirla costaría una
   lectura a S3 por petición, lo mismo que dibujarla; y dibuja y sube todos los sondeos de los 18 ámbitos en cada
   publicación, cuando casi ninguno se comparte. El dibujo es una función que recibe datos: pasar a esta opción
   más adelante reutiliza el mismo código.
3. Capturar la tarjeta real con un navegador sin cabeza. Descartado: Chromium en la imagen de Docker, uno o dos
   segundos por sondeo, y la tarjeta no es 16:9.

## Diseño

### Qué se ve (aspecto «A», elegido por Luis sobre tres bocetos)

Bocetos, sin versionar, redibujados con todo lo decidido: `.superpowers/brainstorm/poll-image/`
`A1-trece-partidos.png`, `A2-horquillas.png`, `A3-sin-escanos.png` y `A4-nombre-largo.png` (los cuatro juntos en
`A-final.png`). La imagen construida debe parecerse a ellos.

Imagen PNG de 1200×675 sobre el fondo claro de la web:

- **Cabecera**: «{casa} para {medio}» (o sólo la casa) en la fuente de los títulos; debajo, en gris, «{elección} ·
  Campo: {fechas} · {muestra} entrevistas». Las fechas van compactas y con el año, como en los bocetos («3–5 nov
  2026», «29 oct–2 nov 2026», «5 nov 2026» si es un solo día): es un formato nuevo, porque `fieldwork_text`
  repite el año («3 nov 2026 – 5 nov 2026») y `fieldwork_label` no lo lleva. La marca (el logotipo con la
  lente), arriba a la derecha.
- **Filas**, hasta seis, una por partido: sus siglas, una barra de su color proporcional a su voto (la mayor de la
  imagen ocupa el ancho máximo), el voto con el formato del sitio («31,9%», el signo pegado: `fmt_pct`, decisión
  de Luis del 2026-10-15) y, detrás, el cambio frente a la
  entrega anterior de la misma casa y medio (triángulo verde hacia arriba, rojo hacia abajo, «=» sin cambio).
- **Columna «Escaños»** a la derecha, con los que publicó la propia casa: una cifra («136») o una horquilla
  («125–136»).
- **Pie**, abajo a la derecha, en gris pequeño: «y N partidos más en demoscopio.es», si el sondeo trae más de
  seis.
- La esquina inferior izquierda queda libre: X superpone ahí el título del enlace. Nada importante en los 24 px
  de arriba y de abajo, que Facebook recorta.

### Casos

| Caso | Qué hace |
|---|---|
| Más de seis partidos | Los seis con más voto y el pie con los que faltan |
| Menos de seis | Las filas que haya; el resto queda vacío |
| Sin escaños | La misma imagen sin la columna «Escaños» |
| Un partido sin escaños publicados en un sondeo que sí los trae | Su celda queda vacía |
| Sin entrega anterior, o el partido no estaba en ella | Sin cambio en esa fila |
| Título largo | Se encoge hasta caber a la izquierda de la marca, hasta un mínimo; por debajo, se corta con «…» |
| Sin muestra | Se omite « · N entrevistas» |
| Sin fechas de campo | El día del sondeo |

Las cifras, los nombres y el cambio son los de la página del sondeo: salen de las mismas funciones (`vote_rows`,
`previous_poll`, `poll_title`, `seats_text`), no de un cálculo aparte.

### Qué partidos y en qué orden (elegido por Luis)

**Los seis con más voto en el sondeo, de mayor a menor**, como la página del sondeo (el orden de `vote_rows`, sin
recalcularlo). En un sondeo con trece partidos entran, por ejemplo, ERC y AA y se quedan fuera UP y SALF: los
partidos cambian de una imagen a otra según lo que diga cada sondeo.

Luis lo eligió sobre otras dos opciones: el orden fijo de las columnas de «Todas las encuestas»
(`polls['parties']`), con el que dibujé por error la primera ronda de bocetos, y esos mismos seis ordenados por
voto, que era mi recomendación.

### Dirección y etiquetas

- La imagen de un sondeo vive en la dirección de su página acabada en `.png`:
  `/es/sondeos/2026-11-05/gad3/vocento.png`. Responde 404 o 400 en los mismos casos que la página, sin dibujar
  nada. Pedida por el identificador de la casa en vez de su nombre, responde igual (no redirige).
- Siempre dibuja el sondeo de **lo último publicado**: `mode` y `run` no cambian la imagen.
- La página del sondeo declara como `og:image` esa dirección, absoluta, con `?v=` y una huella corta de lo que la
  imagen pinta (los datos de la tarjeta y una versión del diseño): si el sondeo se corrige o el diseño cambia, la
  dirección cambia y las redes piden la nueva. El servidor no usa `v` para nada.
- `og:image:alt` de esa página dice las cifras: «Sondeo de GAD3 para Vocento (campo 3–5 nov 2026): PP 31,9%,
  PSOE 25,9%, VOX 18,6%, FA 7,5%, ERC 2,7%, AA 1,6%.» El tamaño y el tipo no cambian (1200×675, PNG).
- El resto de páginas sigue con la imagen de marca y su texto alternativo.
- La respuesta lleva `image/png`, ETag y `Cache-Control: public, max-age=86400`; nginx la guarda mientras se
  siga pidiendo. No hay que tocar `nginx.conf`.
- La imagen no entra en el sitemap ni se enlaza desde la página.

### Código

- El dibujo vive dentro de `mtpy` (web en `mtpy/controllers` + `mtpy/lib`): un módulo nuevo de `mtpy/lib/pages`
  con las piezas de dibujo (lienzo, textos, barras, la marca) y la tarjeta del sondeo, y una acción nueva en el
  controlador `Sondeos`. `web/brand/build.py` no se importa desde la web.
- Lo que la imagen pinta se compone primero como datos (título, subtítulo, filas con siglas, voto, color, cambio y
  escaños, pie) en una función sin Pillow; el dibujo sólo recibe eso. La página usa los mismos datos para la
  huella y el texto alternativo.
- Las fuentes del dibujo se versionan como TrueType junto al código (generadas una vez desde las woff2 del sitio
  con las herramientas de `web/brand`, que ya piden `fonttools` y `brotli`); en ejecución sólo hace falta Pillow.
- Las piezas de dibujo se escriben pensando en una segunda tarjeta (la de una elección pasada), sin construirla.

### Pruebas

- Los datos de la tarjeta: cada caso de la tabla, con los valores exactos.
- La imagen: PNG de 1200×675; los mismos datos dan los mismos bytes; cada caso de la tabla se dibuja sin error
  (13 partidos, sin escaños, horquillas, título más largo que el máximo, nombres con caracteres raros).
- La ruta: 200 con `image/png` y su caché; 404 de un sondeo que no existe y 400 de una fecha inválida sin dibujar;
  por nombre y por identificador de la casa; con medio y sin él.
- Las etiquetas: la página del sondeo declara su imagen con `?v=`, que cambia si cambia una cifra; su texto
  alternativo; las demás páginas siguen con la de marca.
- A ojo: yo miro las imágenes de varios sondeos reales (los cuatro de los bocetos y un ámbito autonómico) antes
  de dar el paso por hecho.

## Límites conocidos

- Un enlace a un sondeo ya compartido en X sigue con la imagen genérica hasta que caduque su caché (unos siete
  días).
- La primera petición de cada imagen la dibuja Python (décimas de segundo en el servidor, sin medir allí).
- Las siglas y los colores son los del catálogo: un color muy claro (BNG, CC) da una barra poco visible, como en
  la web.

## Fuera de este paso

- La imagen de una elección pasada: después, con las mismas piezas de dibujo.
- Las descripciones con cifras (el «nivel 1»).
- Imágenes para las páginas que cambian (portada, evento, ranking): siguen con la genérica.

## Qué eligió Luis y qué propuse yo

De Luis: hacer la imagen con cifras para las páginas que no cambian; el enfoque 1 (dibujar al pedir); los seis
partidos con más voto, por voto; el aspecto A (vio también el B «marcador» y el C «hemiciclo», y la A y la C en
cuatro sondeos reales).

Mías, por revisar:

- **Las fechas de campo compactas** («3–5 nov 2026»), un formato que la web no tenía.
- **El pie «y N partidos más en demoscopio.es».**
- **El cambio frente a la entrega anterior** en la imagen, como en la tarjeta de la página.
- **El texto alternativo con las cifras.**
- **La dirección acabada en `.png`** y que la imagen sea siempre la de lo último publicado.
- **Un día de caché.**
- **Sólo sondeos en este paso.**

## Lo que queda al constructor

- El nombre del módulo y de sus funciones, y el reparto exacto de píxeles, tamaños de letra y márgenes, con los
  bocetos como guía.
- Cómo se espacian las letras (con el motor de Pillow que haya o colocando cada letra): la imagen tiene que salir
  bien con y sin `raqm`, y las anchuras se miden con el mismo motor con que se dibuja.
- Si el dibujo se hace al doble de tamaño y se reduce, o directamente.
- Qué fuentes TrueType se versionan (una variable por familia o una por peso) y dónde.
- Cómo se calcula la huella de `?v=`.
