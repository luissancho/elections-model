<!-- Especificación de diseño acordada con Luis el 2026-10-09 (sesión de brainstorming con Claude Code).
     Enmienda las secciones "Frontend", "Infraestructura" y "Pruebas" de
     docs/superpowers/specs/2026-10-07-web-publicacion-design.md para las páginas web. -->

# Web servida desde los controladores: enrutado en Python y plantillas Jinja2

## Contexto

La fase 2 dejó la web como ficheros estáticos: nginx servía `web/` con `root` y `try_files $uri
$uri.html`, de modo que la URL `/promedio` la fabricaba nginx y las páginas eran cascarones HTML que
cargaban los datos con `fetch` a `/api/v1`. Luis quiere el modelo que ya usaba con mtpy: nginx delante
como proxy, gunicorn ejecutando los controladores de Python, y el controlador renderizando la plantilla
con Jinja2 y sus variables. Las rutas de la web se definen en Python junto a las de la API; nginx solo
sirve los recursos (`css`, `js`, imágenes, librerías) con una `location` y no conoce ninguna URL de página.

## Decisiones de Luis (2026-10-09)

- Las páginas las renderiza el servidor con **Jinja2** desde los controladores de `mtpy/controllers/`:
  todo el control en la app, nginx con una configuración aséptica.
- Los recursos estáticos los sirve **nginx** con `location`, bajo el prefijo **`/dist/`** (carpeta
  `web/dist/`).
- `nginx.conf` conserva su **estructura original** (proxy a gunicorn de todo lo que no sea estático) más
  el **endurecimiento** ya escrito y probado en la fase 2: `proxy_cache` 60 s, `limit_req` con `realip`,
  cabeceras de seguridad, `gzip_types`, logs al stdout del contenedor.
- En el Mac, Luis usa su propia configuración local de nginx añadiéndole la misma `location /dist/`; no
  hacen falta scripts de desarrollo ni el truco `?api=`.
- `jinja2==3.1.2` se mantiene salvo decisión de Luis (sus avisos de seguridad afectan al filtro
  `xmlattr`, que no se usa).

## Arquitectura

```
navegador ──► nginx :8042 ──► /dist/*            ficheros de web/dist/ (alias; vendor 1 año, resto revalidación)
                          ──► /healthz           200 sin tocar gunicorn
                          ──► todo lo demás ──► gunicorn :8000 ──► Router de mtpy
                                                   /api/v1/...  → Forecast (JSON/CSV, como hoy)
                                                   /            → Index(Page)    → Jinja2 → HTML
                                                   /promedio    → Promedio(Page) → Jinja2 → HTML
                                                   otra ruta    → 404 HTML (bajo /) o 404 JSON (bajo /api/)
```

Una petición de página: nginx (caché 60 s por URL con query) → gunicorn → `Page.dispatch` valida la
query (`scope`, `mode`, `run`) con los validadores de `webapi` → el constructor de contexto de
`mtpy/lib/pages.py` lee el paquete con `webapi.site()` (manifest, meta, headline, partes) → Jinja2
renderiza la plantilla con el HTML completo y un bloque `<script type="application/json"
id="initial-data">` con los datos de los gráficos → el JS de la página lee ese bloque y dibuja con
ECharts. Cero `fetch` al cargar; la API queda para el enlace CSV y para consumidores externos.

## Componentes

| Fichero | Papel |
|---|---|
| `web/templates/` (nuevo) | Plantillas Jinja2: `base.html` (documento, cabecera, pie, banner de veda, bloques `title`, `content`, `scripts`), parciales `_header.html` y `_footer.html`, páginas `index.html` y `promedio.html`, y `error.html`. Autoescape activado; ningún `|safe` sobre datos. |
| `web/dist/` (nuevo, por traslado) | Recursos: `css/site.css`, `js/**`, `vendor/echarts-5.6.0.min.js` + licencia, `img/`. URLs `/dist/...`. |
| `mtpy/lib/pages.py` (nuevo) | `ROUTES` de páginas (`/promedio` → `promedio`; `/` lo registra `mtpy.api()`), `NOT_FOUND` (`/api/` → `base` JSON, `/` → `page` HTML), entorno Jinja2 (`FileSystemLoader(web/templates)`, autoescape, `StrictUndefined`) cacheado en la `App`, filtros de formato (`pct`, `prob`, `num`, `date`, `datetime`, `range`, mismas reglas que `format.js`), `parse_state(query)` y los constructores de contexto por página sobre `webapi.site()`. |
| `mtpy/controllers/Page.py` (nuevo) | `Page(Controller)`: `render(template, **context) -> str` (`content-type: text/html`, `ETag` md5, `Cache-Control: public, max-age=60`), `dispatch` que convierte `HttpError` en `error.html` con su código y cualquier otra excepción en una página 500 (con traza en el log), ambas `no-store`; `not_found_action` HTML. |
| `mtpy/controllers/Index.py` (cambia) | `Index(Page)` renderiza la portada. La "API Home" JSON desaparece (`/api/v1/health` cumple esa función). |
| `mtpy/controllers/Promedio.py` (nuevo) | `Promedio(Page)` renderiza `/promedio`. Las páginas de las fases 3-5 (`Escanos`, `Casas`, `Elecciones`, `Modelo`, `Metodo`) seguirán este patrón: un controlador, una plantilla, una entrada en `pages.ROUTES`. |
| `mtpy/mtpy.py` (cambia) | `api(routes=None, not_found=None)`: registra también las reglas `not_found` con `router.add_not_found`. |
| `api.py` (cambia) | `api = mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)`. |
| `deploy/docker/nginx.conf` (cambia) | Sin `root` ni `try_files`: `location ^~ /dist/vendor/` (alias, `expires 1y`), `location ^~ /dist/` (alias, `expires -1`), `location = /healthz`, `location /` → proxy a gunicorn con el endurecimiento actual. |
| `web/js/**` (cambia) | `pages/*.js` leen `initial-data`; `state.js` y `layout.js` se eliminan; `api.js` conserva `forecastUrl` para el CSV y pierde `apiBase`/`?api=`; `format.js`, `catalog.js` y `charts/*` no cambian. Los controles (ámbito, modo) envían el formulario al cambiar. |
| `deploy/check-api.sh`, `deploy/smoke.sh`, `deploy/nginx-check.sh`, `deploy/README.md` (cambian) | Comprueban `/` y `/promedio` como HTML y `/dist/...`; el README documenta la `location /dist/` para la configuración local del Mac. |
| `tests/test_pages.py` (nuevo), `tests/test_web_routes.py` (cambia), `tests/test_api_asgi.py` (cambia) | Ver Pruebas. |

## Contexto de las plantillas

Común a todas las páginas (`pages.common_context(state)`):

- `site_title` ("Pronóstico electoral"), `pages` (nav: `href`, `label`, `active`), `state`
  (`scope`, `mode`, `run`), `scopes` (ámbitos con entrada en el manifest: `code`, `name`,
  seleccionado el actual), `manifest` (`freeze`, `attribution`, `updated_at`), `meta` (`run_id`,
  `as_of`, `date_last`, `event_date`, `commit`, `dirty`), `mode_label` ("Elección" / "Hoy") y
  `title_suffix` ("Pronóstico para el 29 nov 2026" / "Estimación a 13 oct 2026").
- El formulario de controles: `<form method="get">` con `<select name="scope">`, radios `name="mode"`
  (`nowcast`, `forecast`) y un campo oculto `run` cuando está fijado; un botón "Ver" visible sin JS.

Por página (`pages.index_context`, `pages.promedio_context`):

- Portada: tabla del titular renderizada en HTML (partido con punto de color, voto con intervalo,
  escaños con intervalo, probabilidad de ser primero) y `initial` = `{state, meta, headline, vote,
  summary, runs}` (los `data` de cada fichero del paquete) para los gráficos.
- Promedio: tabla de los últimos 40 sondeos en HTML (fecha, casa, patrocinador, muestra y columnas de
  `bmaps.main`), enlace CSV a `/api/v1/forecast/{scope}/polls?run=...&format=csv`, e `initial` =
  `{state, meta, series, polls, projection, vote}`.
- `initial` se emite con el filtro `tojson` dentro de `<script type="application/json"
  id="initial-data">`; el JS lo lee con `JSON.parse(textContent)`. La CSP (`script-src 'self'`) no lo
  bloquea: un bloque JSON no se ejecuta.

## Parámetros, validación y caché

- `scope`: `check_scope`; por defecto el primer ámbito del manifest. Un ámbito del catálogo sin publicar
  → 404 HTML "ámbito sin pronóstico publicado".
- `mode`: `check_mode`; por defecto `forecast`. `run`: `check_run`; sin él, el `latest` del manifest.
  Todas las partes de una página se leen con el mismo `run`, así que una página nunca mezcla runs.
- Sin paquete publicado → 503 HTML "todavía no hay ningún pronóstico publicado". Error inesperado →
  500 HTML genérico y traza en `app.logger`.
- Cabeceras de página: `Cache-Control: public, max-age=60`, `ETag`; errores `no-store`. nginx cachea
  cada URL (query incluida) 60 s con `proxy_cache_use_stale`; las retiradas (`point`/`unpublish`) se ven en
  ≤ 2 min más la caché del navegador.

## nginx

Estructura original de Luis (`worker_processes`, `events`, `http` con `mime.types`, `upstream app_server`,
`server` en 8042 con `charset utf-8`) sin `root` ni `try_files`, con:

- `location ^~ /dist/vendor/ { alias /app/web/dist/vendor/; expires 1y; }` y `location ^~ /dist/ { alias
  /app/web/dist/; expires -1; }`.
- `location = /healthz { return 200 'ok'; }`.
- `location / { ... proxy_pass http://app_server; }` para páginas y `/api/`, con el endurecimiento actual:
  `limit_req` (20 r/s, ráfaga 40, 429) sobre la IP real (`realip` desde `X-Forwarded-For`), `proxy_cache`
  con `proxy_cache_path /var/lib/nginx/api_cache` y clave por defecto (URL completa con query),
  `proxy_cache_valid 200 60s`, `use_stale`, `X-Cache`, cabeceras `X-Forwarded-*`; cabeceras de seguridad
  (`X-Content-Type-Options`, `Referrer-Policy`, `Content-Security-Policy` con `frame-ancestors`,
  `base-uri`, `object-src`) repetidas en este `location`; `gzip_types` para HTML, JSON, JS, CSS, CSV y
  SVG con `gzip_proxied any` y `gzip_vary on`; `server_tokens off`; logs a `/proc/1/fd/1` y
  `/proc/1/fd/2`.
- Sin `client_max_body_size 4G`, `proxy_buffering off` ni `ssl_protocols` (TLS lo termina el proxy del
  servidor), como ya decidió la spec general.

## Pruebas

- `tests/test_pages.py` (arnés ASGI de `tests/test_api_asgi.py`, fixture con el paquete de fixtures de
  `tests/test_webapi_unit.py`): `/` y `/promedio` responden 200 `text/html; charset=utf-8` con `ETag` y
  `max-age=60`; el título y el `h1` contienen el sufijo del modo; el nav marca la página activa
  (`aria-current`); el `<select>` lista los ámbitos publicados con el actual seleccionado; `?mode=nowcast`
  marca su radio; el bloque `initial-data` se parsea y trae las claves de la página con el `run_id`
  correcto; `?run=` fija el run en todas las partes; `?scope=es-md` → 404 HTML; `?run=mal` → 400 HTML;
  `/nope` → 404 HTML; `/api/v1/nope` sigue siendo 404 JSON; sin paquete → 503 HTML `no-store`; sin JS la
  tabla del titular está en el HTML; ningún `<script>` inline salvo el bloque JSON.
- Test de plantillas: el entorno Jinja2 compila todas las plantillas de `web/templates/`
  (`env.get_template`), con `StrictUndefined`.
- `tests/test_web_routes.py` adaptado: cada `src`/`href` que empieza por `/dist/` en las plantillas
  existe en `web/dist/`; cada enlace del nav está en `pages.ROUTES` (o es `/`); cada literal `/api/v1` del
  JS casa con `webapi.ROUTES`; los `import` del JS resuelven; sin `onclick=` ni `<script>` inline en las
  plantillas.
- `tests/test_api_asgi.py`: el test de `Index` cambia de "API Home" a "página HTML".
- `deploy/check-api.sh` añade `/` y `/promedio` (200, `text/html`) y `/dist/vendor/echarts-5.6.0.min.js`;
  `deploy/smoke.sh` comprueba las cabeceras de caché de `/dist/vendor/` y `/dist/js/`;
  `deploy/nginx-check.sh` reescribe también los `alias` al `web/dist` del repositorio.
- Comprobación local sin Docker: uvicorn con `S3_BUCKET=` más el nginx local con la `location /dist/`;
  `node --check` y renderizado SSR de los gráficos como en la fase 2.

## Migración y limpieza

1. Mover `web/css`, `web/js`, `web/vendor` a `web/dist/`; crear `web/dist/img/` (vacío salvo favicon si
   se añade).
2. Convertir `web/index.html` y `web/promedio.html` en plantillas que extienden `base.html`; borrar los
   originales.
3. Reescribir `pages/index.js` y `pages/promedio.js` para leer `initial-data`; borrar `state.js` y
   `layout.js`; recortar `api.js`.
4. `Index` deja de devolver JSON; `mtpy.api()` acepta `not_found`; `api.py` registra páginas y API.
5. `nginx.conf` según la sección anterior; scripts y README.
6. Enmendar la spec general: "Frontend" (renderizado en servidor, sin router JS ni `state.js`;
   controles por formulario), "Infraestructura" (nginx sin `root`, `location /dist/`), "Pruebas"
   (`test_pages.py`), y anotar en "Estado al cierre de la fase 2" que las páginas pasan a controladores.

## Fuera de alcance

- Páginas nuevas (fases 3-5) y la ocultación de vistas en veda: siguen en sus fases, ya sobre este
  patrón.
- Cambiar `Dockerfile`, `supervisord.conf` o `init.sh` (imagen de Luis), o la versión de jinja2.
- Servir los recursos desde Python o un modo de desarrollo propio: Luis usa su nginx local.

## Verificación

`python -m pytest -m "not integration" -q` en verde; `bash deploy/nginx-check.sh`; uvicorn (`S3_BUCKET=`)
más nginx local: `/` y `/promedio` devuelven HTML con la tabla del titular y el bloque `initial-data`,
`/dist/vendor/echarts-5.6.0.min.js` sale de nginx con un año de caché, `/api/v1/manifest` sigue
igual; `curl /nope` → 404 HTML, `curl /api/v1/nope` → 404 JSON; `docker logs` muestra nginx (smoke de
Luis); recorrido visual de Luis en escritorio y móvil.

## Estado al cierre (2026-10-09)

Implementado en `dev` con el plan `docs/superpowers/plans/2026-10-09-web-enrutado-controladores.md`
(plan c1db892, spec 307ec38). Commits: 93ed27f (recursos a `web/dist/`), 7aad529 (núcleo de páginas y
`/`), f27925a (`/promedio`), 2e7ba70 y 0258ef3 (JS que solo dibuja y su endurecimiento), 6ba4430 y
965f891 (nginx y scripts), más el de esta documentación.

Ficheros:

- `mtpy/lib/pages.py`: `ROUTES`, `PAGES`, `NOT_FOUND`, entorno Jinja2 (`StrictUndefined`, autoescape,
  filtros `num/int/pct/prob/range/date/datetime` según `format.js`), `parse_state`, `common_context`,
  `index_context`, `promedio_context`, `table_rows`, `api_url`.
- `mtpy/controllers/Page.py` (`render`, `error_page`, `dispatch`, `not_found_action`), `Index.py` y
  `Promedio.py`; `api.py` registra `pages.ROUTES + webapi.ROUTES` con `not_found=pages.NOT_FOUND`;
  `mtpy.api()` acepta `not_found`; `mtpy/core/api.py` gana `Router.controller_class(name)`.
- `web/templates/` (`base`, `_header`, `_footer`, `error`, `index`, `promedio`) y `web/dist/{css,js,vendor,img}`
  (`js/pages/common.js` nuevo; `state.js`, `layout.js` y `api.js` borrados).
- `deploy/docker/nginx.conf`, `nginx-check.sh`, `check-api.sh`, `smoke.sh`, `deploy/README.md`.

Pruebas: 320 unitarias en verde (`python -m pytest -m "not integration" -q`; 72 de integración sin
ejecutar): `tests/test_pages.py` 17, `tests/test_api_asgi.py` 31, `tests/test_web_routes.py` 4. Verificado
el 2026-10-09 con el nginx 1.29.5 local (puerto 8043) delante de uvicorn (8000): `/` primero `MISS` y
luego `HIT`, cabeceras como las esperadas, `check-api.sh` todo OK. El smoke con Docker no se ha
ejecutado (Docker apagado).

Decisiones tomadas durante la ejecución:

- `Router.controller_class(name)` en el núcleo: resuelve la clase cuando el módulo de un controlador está
  enlazado en el paquete por `from .Page import Page`; con test de regresión.
- `#status` vive en `base.html` y el bloque `initial-data` usa `tojson` con `sort_keys` desactivado
  (`env.policies['json.dumps_kwargs'] = {'sort_keys': False}`) para conservar el orden de las claves.
- `fmt_num` agrupa los números de 4 cifras ("4.000"), a diferencia de `Intl` es-ES, porque el plan lo
  exige para la columna de muestra; se mantiene.
- `wireControls` (`common.js`) envía el formulario al cambiar un control y descarta el `run` fijado
  cuando cambia el ámbito; `pageshow` restablece los controles para la caché de ida y vuelta (bfcache) y
  el formulario lleva `autocomplete="off"`.
- `check-api.sh` decide una sola vez si hay nginx a partir de `/healthz`, antes de las comprobaciones
  de `/dist/`.
- Efectos a tener en cuenta: nginx cachea las páginas 60 s y las limita con la misma zona `limit_req` que
  la API.

Pendiente de Luis:

- `bash deploy/smoke.sh` con Docker (paquete local y `deploy/elections.env`).
- Recorrer `/` y `/promedio` en el navegador con y sin JS: el botón "Ver" debe funcionar sin JS; con JS,
  el cambio de ámbito o de modo recarga la página.
- Añadir las dos `location` de `/dist/` y el proxy a su nginx local (ver `deploy/README.md`).
- Primera publicación a S3 y despliegue.
- Decidir si se sube `jinja2` de 3.1.2 a 3.1.6.
- Accesibilidad: el envío al cambiar del `<select>` se dispara con cada flecha del teclado en
  Chrome y Firefox (WCAG 3.2.2); la spec lo exige, revisar si molesta.

Menores aplazados: el `<select>` no muestra opción seleccionada si el ámbito por defecto queda fuera del
catálogo; `table_rows` indexa `poll['date']` directamente; `Page.dispatch` duplica `Controller.dispatch`;
`common_context` analiza el manifest varias veces por petición.
