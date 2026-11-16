# Web: `robots.txt` y `sitemap.xml`

Cambio pequeño, segundo paso de la ronda de SEO y redes sociales. Luis dio paso a la tarea el 2026-11-16 («Sigue con
eso»); el diseño no lo ha revisado: todo lo que no viene de él está en «Decisiones mías, por revisar». Plan:
`docs/superpowers/plans/2026-11-16-web-robots-sitemap.md`.

## Intención

Que los buscadores encuentren todas las páginas de demoscopio.es sin depender de seguir enlaces, y que sepan qué no
les interesa rastrear. Hoy `/robots.txt` y `/sitemap.xml` responden 404.

## Hechos de partida (2026-11-16)

- nginx sólo sirve `/dist/`; todo lo demás lo responde la aplicación (`pages.ROUTES`, la primera ruta que casa gana).
  Los dos nombres llevan un punto, que `/{scope}` no admite: hoy caen en la página de 404.
- Las páginas se pintan en el servidor y cada una declara su URL canónica (`Page.canonical_url`: sin `mode` ni
  `run`). El origen público es `WEB_SITE_URL` (`https://demoscopio.es` por defecto).
- Qué páginas hay depende de lo publicado: los ámbitos con previsión y sus sondeos (el manifiesto y la parte `polls`
  de la última actualización de cada ámbito), las elecciones pasadas (`site().events`), los estudios
  (`site().studies`) y las encuestadoras (la base de datos, `pollster_slugs`).
- El navegador sólo pide a `/api/` dos cosas, y las dos son añadidos interactivos: las simulaciones de la calculadora
  de escaños y el ejemplo de pesos de `/metodo/promedio`. El resto de `/api/` son descargas en JSON y CSV enlazadas
  desde las páginas («Ver los datos», el manifiesto del pie); varias consultan la base de datos.
- En producción hay 18 ámbitos publicados: del orden de un millar largo de páginas de sondeos, un centenar de
  elecciones pasadas y otro de encuestadoras. Muy por debajo del tope de un sitemap (50.000 URL).

## Enfoques

1. **Dos rutas de la aplicación** (elegido): `/robots.txt` y `/sitemap.xml` las responde un controlador, y el
   sitemap se calcula en cada petición con las mismas funciones que usan las páginas para enlazarse. Nunca se
   desfasa de lo publicado y no añade ningún paso al despliegue ni a los jobs.
2. **Ficheros generados por el job `publish`.** Descartado: el sitemap dependería de acordarse de publicar tras
   cada cambio de rutas, nginx no lee de S3 (la aplicación tendría que servirlo igualmente) y las encuestadoras
   vienen de la base, no del paquete.
3. **`robots.txt` estático en `web/dist`, servido por nginx como el favicon.** Descartado por poco: la línea
   `Sitemap:` lleva la URL absoluta, que depende de `WEB_SITE_URL`, y serían dos sitios donde mirar.

## Diseño

### `robots.txt`

```
User-agent: *
Disallow: /api/

Sitemap: <site_url>/sitemap.xml
```

`text/plain`. Todo abierto menos `/api/`: son los mismos datos de las páginas en JSON y CSV, y que un rastreador
recorra esos enlaces sólo carga S3 y la base. El coste es que Google, al pintar la página, no verá las dos cosas que
el navegador pide a `/api/`; el contenido de las páginas no depende de ellas.

### `sitemap.xml`

Un `<urlset>` del protocolo sitemaps.org con una `<url>` por página, en este orden:

1. `/`, con `<lastmod>` = `updated_at` del manifiesto.
2. `/{ámbito}` de cada ámbito publicado, con `<lastmod>` = el `run_at` de su última actualización.
3. `/elecciones` y cada elección pasada publicada (`event_href`).
4. `/casas` y el perfil de cada encuestadora con slug (`pollster_slugs`), por orden alfabético.
5. `/metodo` y sus subpáginas con texto.
6. `/analisis` y cada estudio publicado con página (`study_cards`).
7. La página de cada sondeo de la última actualización de cada ámbito publicado (`poll_detail_href`), si su página
   lo encuentra (`find_poll`): dos registros con el mismo día, casa y patrocinador comparten una URL que responde
   404, y no se lista.

Sólo URL canónicas, absolutas con `site_url` y sin parámetros. Sin `<changefreq>` ni `<priority>` (Google los
ignora) y con `<lastmod>` sólo donde hay una fecha cierta y barata (los dos primeros grupos).

Fuera del sitemap: `/api/`, las rutas que sólo redirigen (`/promedio`, `/{ámbito}/escanos`…), las variantes con
`?scope=` de las páginas globales y `/privacidad`.

Sin base de datos, el sitemap responde 503 con la página de error de siempre (sin caché), igual que `/casas`: un
rastreador reintenta, y así nunca se guarda una lista a medias.

### Cabeceras

Las dos respuestas llevan ETag y `cache-control: public, max-age=3600`: la caché de nginx respeta la cabecera de la
aplicación y las guarda como mucho una hora, pero suelta antes lo que nadie pide en 10 minutos (`inactive=10m`), así
que la visita de un rastreador casi siempre las recalcula. Un sondeo nuevo tarda como mucho una hora en aparecer.

### Código

- `mtpy/lib/pages/crawlers.py`: `robots_txt()`, `sitemap_entries()` (la lista de rutas con su `lastmod`) y
  `sitemap_xml()`.
- `mtpy/controllers/Crawlers.py`: `robots_action` y `sitemap_action`.
- `pages.ROUTES`: las dos rutas, antes de `/{scope}`, con el punto escapado (los patrones son expresiones regulares).

### Pruebas

`tests/test_pages_crawlers.py`:

- `robots.txt`: 200, `text/plain`, sus líneas, y la de `Sitemap:` sigue a `site_url`.
- `sitemap.xml`: XML válido con las URL esperadas de cada grupo y sus `lastmod`; ni `/api/` ni parámetros.
- Cada URL del sitemap responde 200 y su canónica es ella misma.
- Sin base de datos, 503 sin caché.

### Comprobación tras desplegar

1. `curl` de `/robots.txt` y de `/sitemap.xml` (también en `deploy/check-api.sh`).
2. Dar de alta `https://demoscopio.es/sitemap.xml` en Google Search Console y en Bing Webmaster Tools (paso de
   Luis: hay que verificar el dominio).

## Corregido tras la revisión

- El sitemap listaba la URL que comparten dos registros de sondeo con el mismo día, casa y patrocinador, y esa
  página responde 404 (hay un par así en el paquete local del 8 de octubre: Data10, 2025-03-28, «Okdiario» y
  «OKDiario»). Ahora cada URL de sondeo pasa por la misma regla que su página. La tabla de sondeos de la página
  del evento sigue enlazando esa URL: es anterior a este paso y queda fuera.
- La hora de caché de nginx estaba contada de más (`inactive=10m`).

## Fuera de este paso

- Comprimir el XML en nginx (`gzip_types` no incluye `application/xml`): son unos 150 kB para un rastreador.
- Un índice de sitemaps: hace falta a partir de 50.000 URL.
- Etiquetas `noindex`, datos estructurados, títulos de las páginas y `twitter:site`: siguen en la ronda.

## Decisiones mías, por revisar

- **Los rastreadores de IA (GPTBot, ClaudeBot, Google-Extended…) quedan permitidos**, como cualquier otro: la web es
  divulgativa y gana con que la citen. Cerrarlos son unas líneas más en `robots.txt`.
- **Las páginas de los sondeos entran en el sitemap.** Son la mayoría de las URL y las que responden a búsquedas
  concretas («encuesta GAD3 noviembre»). Tras una elección, las del ciclo que acaba dejan de existir, como hoy.
- **`/api/` cerrado** en `robots.txt`.
- **`/privacidad` no entra**: es una página legal, sin interés para un buscador; se puede rastrear igualmente.
- **Entran todas las encuestadoras con slug**, también las que no tienen rating (agregadores, paneles en línea).
- La hora de caché y el 503 sin base de datos.
