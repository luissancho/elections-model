# Web: imagen de marca para los enlaces compartidos (`og:image`)

Cambio pequeño, primer paso de una ronda de SEO y redes sociales. Aprobado por Luis el 2026-11-16 («Aprobado,
adelante»). Plan: `docs/superpowers/plans/2026-11-16-web-imagen-enlaces.md`.

## Intención

Luis quiere que, al compartir un enlace de demoscopio.es en X, Facebook o una aplicación de mensajería, la vista
previa salga con una imagen. Hoy sale sin ninguna.

## Hechos de partida (2026-11-16)

- `web/templates/base.html` ya envía título, descripción, canónica y `og:site_name`, `og:title`,
  `og:description`, `og:url`, `og:type` y `og:locale` en todas las páginas. No hay `og:image`, y la tarjeta de X es
  `summary` (miniatura pequeña). `error.html` no lleva etiquetas `og:`.
- Los gráficos se dibujan en el navegador; X y Facebook no ejecutan JavaScript al leer un enlace, así que la imagen
  tiene que ser un fichero ya hecho.
- Los ficheros de marca (`logo.svg`, favicons) los genera `web/brand/build.py` con fontTools y Pillow a partir de
  las fuentes de `web/dist/vendor/fonts`, se guardan en `web/dist/img` y van al repositorio; nginx los sirve bajo
  `/dist/`. `tests/test_deploy_files.py` comprueba que existen y lo que miden.
- X y Facebook enseñan la imagen grande a 1,91:1 aproximadamente y recortan desde el centro: a un 16:9 de
  1200×675 le quitan unos 23 px arriba y otros tantos abajo. (Guías de terceros, que no coinciden del todo.)
- X guarda la vista previa de cada URL unos 7 días y no ofrece ya ninguna herramienta para refrescarla; los posts
  publicados no se actualizan.
- `robots.txt` y `sitemap.xml` devuelven 404 (queda para la ronda de SEO).

## Lo que decidió Luis

- **Imagen genérica, sin cifras**, también en el Promedio: una página que cambia cada día, con una caché de 7 días
  que no controlamos, acabaría enseñando cifras viejas con la marca.
- **16:9**, 1200×675.
- **Aspecto: la variante «D1 · Cristal blanco»** de los bocetos (tres rondas en el navegador): el reparto de la B
  (logotipo grande, lema y dominio a la izquierda) sobre el fondo claro de la web, con la lente de la C a la
  derecha.

## Enfoques

1. **Una imagen fija de marca, generada por `web/brand/build.py`** (elegido). No caduca, sirve a todas las páginas
   y no se separa de la marca: sale del mismo logotipo, colores y fuentes.
2. **Un resumen con cifras por página, generado en el servidor.** Era la idea inicial de Luis; descartado para las
   páginas vivas por la caché de X. Sigue teniendo sentido donde la página no cambia (un sondeo, una elección
   pasada, una actualización fijada con `?run=`): segundo paso, con su propio diseño.
3. **Una imagen hecha a mano** (Canva, Figma). Descartado: deja de ser reproducible y se desfasa si cambia el
   logotipo.

## Diseño

### La imagen (`web/dist/img/og-image.png`)

PNG de 1200×675, en coordenadas de la imagen:

- Fondo `#f4f5f8` (`--color-bg`).
- Logotipo: el mismo de `logo.svg` (tinta `#1a1c22`, lente `#1d4f86`), esquina superior izquierda en (84, 226) y
  580 px de ancho.
- Lema en dos líneas, DM Sans 500 de 28 px, `#5a5f72`, desde x = 88 con líneas base en 398 y 438: «Promedio de
  encuestas, previsión de escaños» / «y ranking de encuestadoras» (`SITE_TAGLINE`).
- Dominio «demoscopio.es», Outfit 700 de 24 px, `#1d4f86`, en x = 88 y línea base 590.
- Lente: centro (958, 338), radio exterior 218, aro de 46 px en `#1d4f86`, cristal blanco. Dentro, recortados al
  cristal, 150 puntos (los sondeos) en `#1d4f86` al 22 % y una curva suave (el promedio) en `#1d4f86`. Sin cifras
  ni colores de partido. Los puntos salen de una semilla fija, para que la imagen sea la misma en cada generación.
- Nada importante en las franjas de 24 px de arriba y de abajo.

`build.py` la dibuja con Pillow, a varias veces su tamaño y reducida con Lanczos (como ya hace con la lente del
favicon), con las fuentes de la web instanciadas al peso que toca. Sin dependencias nuevas.

### Las etiquetas (`base.html`)

Tras `og:locale`:

- `og:image`: URL absoluta, `<site_url>/dist/img/og-image.png?v=<8 hex del MD5 del fichero>`.
- `og:image:type` (`image/png`), `og:image:width` (1200), `og:image:height` (675).
- `og:image:alt`: «Demoscopio: promedio de encuestas, previsión de escaños y ranking de encuestadoras».
- `twitter:card` pasa de `summary` a `summary_large_image`. X toma la imagen de `og:image`.

La URL llega a la plantilla como `og_image`: `Page.render` la pone por defecto, igual que `canonical`, y un
controlador podrá pasar otra en el segundo paso. El `?v=` cambia cuando cambia el fichero, para que las
plataformas que guardan la imagen por URL pidan la nueva.

### Pruebas

- `tests/test_pages.py`: las etiquetas nuevas en una página normal, con la URL absoluta según `site_url` (también
  con el host cambiado) y el `?v=`.
- `tests/test_deploy_files.py`: el fichero existe, es un PNG de 1200×675 y pesa menos de 300 kB.

### Comprobación tras desplegar

1. `curl` de una página: las etiquetas; `curl` de la imagen: 200 e `image/png`.
2. Facebook Sharing Debugger con la portada.
3. Un post de prueba de Luis en X. Mirar el recorte y si el título que X superpone abajo a la izquierda tapa el
   dominio; si lo tapa, se sube o se quita.

## Fuera de este paso

- El resumen con cifras en las páginas que no cambian, y una imagen por elección («Elecciones generales · 29 de
  noviembre de 2026») si la genérica se queda corta en el Promedio.
- `robots.txt`, `sitemap.xml`, datos estructurados y `twitter:site`: ronda de SEO.
- La página de error sigue sin etiquetas `og:`.

## Decisiones mías, por revisar

- El nombre del fichero (`og-image.png`) y el `?v=` con el hash.
- El texto alternativo.
- No añadir `twitter:image`: X usa `og:image` cuando falta.
