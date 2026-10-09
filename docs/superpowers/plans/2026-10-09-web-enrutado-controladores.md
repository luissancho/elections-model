# Web servida desde los controladores (Jinja2, `/dist/`): plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que `/` y `/promedio` las sirvan controladores de Python que renderizan plantillas Jinja2 con
los datos del paquete ya embebidos, que nginx solo sirva los recursos de `web/dist/` y haga de proxy de
todo lo demás, y que el JS se limite a dibujar los gráficos a partir de un bloque JSON embebido.

**Architecture:** `mtpy/lib/pages.py` (rutas de páginas, entorno Jinja2 cacheado en la `App`, filtros
de formato con las reglas de `format.js`, lectura de la query y constructores de contexto sobre
`webapi.site()`), `mtpy/controllers/Page.py` (clase base: `render`, errores en HTML, 404 HTML),
`Index(Page)` y `Promedio(Page)`; plantillas en `web/templates/` y recursos en `web/dist/`;
`mtpy.api()` acepta `not_found` para separar el 404 HTML de las páginas del 404 JSON de la API. Los
controles de ámbito y modo son un formulario GET: cambiar de estado es una navegación. nginx vuelve a la
estructura original de Luis (proxy de todo) más el endurecimiento de la fase 2 y dos `location` para
`/dist/`.

**Tech Stack:** Python 3.11 (`python` del venv), Jinja2 3.1.2 (ya en `requirements.txt`), mtpy (ASGI
propio), pytest 9; HTML + módulos ES + ECharts 5.6.0 vendorizado; nginx (imagen de Luis; Homebrew 1.29.5
en local para `nginx -t` y la comprobación en el puerto 8043); Node 25 solo para `node --check` y
renderizados SSR de ECharts.

**Spec:** `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md` (todas sus secciones),
que enmienda `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Frontend",
"Infraestructura", "Pruebas"). Contrato del paquete: `docs/web/contrato.md`.

## Global Constraints

- PEP8 y docstring numpydoc en inglés en todo lo nuevo de `mtpy/`; español en `tests/`, `docs/`,
  `deploy/*.md` y `deploy/*.sh`; plantillas y textos visibles en español; JS: módulos ES,
  identificadores y comentarios en inglés, sin inline scripts ni `on*=`, sin `innerHTML` con datos.
- `python -m pytest -m "not integration" -q` sin base (hoy 301 tests); `FutureWarning` es error.
- Árbol limpio al empezar (rama `dev`, `307ec38`); `git add <ruta>` explícito por tarea; mensajes cortos
  en inglés con el trailer `Co-Authored-By` del modelo que escribe el commit.
- **Nunca arrancar `mtpy.run()` sin `S3_BUCKET=` (vacío) delante** (`uvicorn`, scripts): el `.env` de la
  raíz apunta al bucket real y a la RDS. Nunca `python job.py`. El paquete local `files/site/v1` (run
  `20261008-181100`) alimenta todas las comprobaciones manuales.
- Jinja2: `Environment(loader=FileSystemLoader(web/templates), autoescape=select_autoescape(('html',)),
  undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)`; ningún `|safe` sobre datos; el bloque
  de datos se emite con `{{ initial|tojson }}` dentro de `<script type="application/json"
  id="initial-data">`. Filtros registrados: `pct`, `prob`, `num`, `int`, `date`, `datetime`, `range`, con
  las reglas de `web/js/format.js` (miles `.`, decimales `,`, `'–'` para nulos, meses `ene feb mar abr
  may jun jul ago sept oct nov dic`, `prob`: `0 %`, `100 %`, `> 99 %` si `p ≥ 0,995`, `< 1 %` si
  `0 < p ≤ 0,005`).
- Estado de página: `scope` (`webapi.check_scope`; por defecto el primer ámbito publicado del manifest en
  el orden del catálogo; un ámbito del catálogo sin publicar → 404), `mode` (`check_mode`; por defecto
  `forecast`), `run` (`check_run`; por defecto el `latest` del manifest). Todas las partes de una página
  se leen con el mismo `run`.
- Cabeceras de página: `content-type: text/html; charset=utf-8`, `cache-control: public, max-age=60`,
  `etag` (md5 de los bytes); errores (400/404/500/503) en HTML con `no-store`; textos de error en español.
- Recursos bajo `/dist/` (`web/dist/{css,js,vendor,img}`); URLs absolutas en las plantillas
  (`/dist/css/site.css`, `/dist/vendor/echarts-5.6.0.min.js`, `/dist/js/pages/<página>.js`).
- nginx (`deploy/docker/nginx.conf`): sin `root`, `index` ni `try_files`; `location ^~ /dist/vendor/`
  (`alias /app/web/dist/vendor/; expires 1y;`), `location ^~ /dist/` (`alias /app/web/dist/; expires -1;`),
  `location = /healthz`, `location /` → proxy de todo lo demás con el bloque actual de `limit_req`,
  `proxy_cache` (clave por defecto), cabeceras `X-Forwarded-*`, `X-Cache` y las tres de seguridad
  repetidas. Nada más cambia en `deploy/docker` (`Dockerfile`, `supervisord.conf`, `init.sh`, `crontab`
  son de Luis).
- Fuera de alcance: páginas nuevas, ocultación en veda, versión de jinja2, modo de desarrollo propio
  (Luis usa su nginx local con la `location /dist/` documentada en el README).

## Review Focus

1. Petición sin paquete publicado o con `?scope=es-md` (catálogo sin publicar) → página de error HTML
   503/404 con `no-store`, nunca JSON ni traza; `/api/v1/nope` sigue siendo 404 JSON (tests en Tarea 2).
2. Plantilla con una variable no definida: `StrictUndefined` la convierte en 500 HTML con traza en el log,
   no en un hueco silencioso; el test que compila todas las plantillas no basta, así que cada contexto se
   renderiza en los tests con el paquete de fixtures (Tareas 2 y 3).
3. `?run=` fijado: todas las partes de la página y el enlace CSV llevan ese run, y el formulario lo
   conserva como campo oculto (Tareas 2, 3 y 4).
4. Recursos: `/dist/vendor/...` sale de nginx con un año de caché y `/dist/js/...` con revalidación; una
   URL de página bajo `/dist/` inexistente da 404 de nginx, no de Python; `/nope` → 404 HTML de Python
   (Tarea 5, nginx local).
5. Sin JavaScript la página informa: tabla del titular y tabla de sondeos en el HTML, botón "Ver" del
   formulario visible; con JS el botón se oculta y el cambio de un control envía el formulario (Tareas 2-4;
   comprobación visual de Luis).

---

### Task 1: Recursos a `web/dist/` y test estático adaptado

**Files:**
- Move: `web/css/` → `web/dist/css/`, `web/js/` → `web/dist/js/`, `web/vendor/` → `web/dist/vendor/`
  (`git mv`); Create: `web/dist/img/.gitkeep`
- Modify: `web/index.html`, `web/promedio.html` (rutas `/dist/...`; se convierten en plantillas en las
  Tareas 2-3), `tests/test_web_routes.py`

**Interfaces:**
- Produces: layout `web/dist/{css,js,vendor,img}`; `tests/test_web_routes.py` con `DIST =
  os.path.join(WEB, 'dist')`, `js_files()` sobre `DIST/js`, `html_files()` sobre `WEB/templates` si
  existe y si no sobre `WEB` (transición), y la regla de recursos: un `src`/`href` que empieza por
  `/dist/` debe existir en `web/dist/`; las páginas (`/`, `/promedio`) se comprueban contra
  `pages.ROUTES` a partir de la Tarea 2 (hasta entonces, contra `web/<nombre>.html`).

- [ ] **Step 1: Mover los recursos y actualizar las referencias**

```bash
mkdir -p web/dist web/dist/img && touch web/dist/img/.gitkeep
git mv web/css web/dist/css && git mv web/js web/dist/js && git mv web/vendor web/dist/vendor
sed -i '' -e 's#"/css/#"/dist/css/#g; s#"/vendor/#"/dist/vendor/#g; s#"/js/#"/dist/js/#g' web/index.html web/promedio.html
```

Los `import` relativos del JS no cambian (la estructura interna de `js/` es la misma).

- [ ] **Step 2: Adaptar `tests/test_web_routes.py`**

`WEB = .../web`, `DIST = os.path.join(WEB, 'dist')`, `TEMPLATES = os.path.join(WEB, 'templates')`.
`js_files()` recorre `DIST/js`; `html_files()` devuelve los `.html` de `TEMPLATES` si el directorio
existe y, si no, los de `WEB`. En `test_referenced_assets_exist`: `ref.startswith('/dist/')` →
`os.path.isfile(os.path.join(WEB, ref.lstrip('/')))`; `ref.startswith('/api/')` → se ignora; cualquier
otro `href` sin extensión es una página y se acepta si existe `web/<nombre>.html` **o** si casa con una
ruta de `mtpy.lib.pages.ROUTES` o es `/` (el import de `pages` se hace dentro de un `try` hasta la
Tarea 2: `pages_routes()` devuelve `[]` si el módulo no existe). `test_vendor_is_pinned_and_licensed`
apunta a `DIST/vendor`.

- [ ] **Step 3: Ejecutar los tests**

Run: `python -m pytest tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: 4 passed; 301 passed.

- [ ] **Step 4: Commit**

```bash
git add web/dist web/index.html web/promedio.html tests/test_web_routes.py
git commit -m "$(printf 'Move the web assets under web/dist\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

(`git add web/dist` recoge los renombrados; comprobar con `git status --short` que no queda nada en
`web/css`, `web/js` ni `web/vendor`.)

---

### Task 2: Núcleo de páginas: `pages.py`, `Page`, plantillas base y portada

**Files:**
- Create: `mtpy/lib/pages.py`, `mtpy/controllers/Page.py`, `web/templates/base.html`,
  `web/templates/_header.html`, `web/templates/_footer.html`, `web/templates/error.html`,
  `web/templates/index.html`, `tests/test_pages.py`
- Modify: `mtpy/mtpy.py` (`api(routes=None, not_found=None)`), `mtpy/controllers/Index.py`, `api.py`,
  `tests/test_api_asgi.py` (test de `Index`), `tests/test_web_routes.py` (si hace falta por
  `html_files()`)
- Delete: `web/index.html`

**Interfaces:**
- Consumes: `webapi.site()` (`manifest_data()`, `latest_run(scope)`, `run_file(scope, run, part,
  mode=None)` → bytes, `scopes()`, `freeze()`), `webapi.check_scope/check_mode/check_run`,
  `bundle.loads`, `HttpError`, `Controller`/`Response` (`set_content_type`, `set_etag`, `set_cache`,
  `set_header`, `set_status_code`, `_set_error`), `Router.add_not_found`.
- Produces `mtpy/mtpy.py`: `api(routes=None, not_found=None)`; tras las rutas, `for pattern, controller
  in not_found or []: router.add_not_found(pattern, controller)`.
- Produces `mtpy/lib/pages.py`:

```python
REPO_ROOT, TEMPLATES_DIR = ..., os.path.join(REPO_ROOT, 'web', 'templates')
SITE_TITLE = 'Pronóstico electoral'
PAGES = ({'href': '/', 'label': 'Portada'}, {'href': '/promedio', 'label': 'Promedio'})
ROUTES = [('/promedio', 'promedio', 'index', ['GET'])]          # '/' lo registra mtpy.api()
NOT_FOUND = [('/api/', 'base'), ('/', 'page')]
MONTHS = ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic')
DASH = '–'
MESSAGES = {'no bundle published yet': 'Todavía no hay ningún pronóstico publicado.',
            'scope not published': 'Este ámbito no tiene pronóstico publicado.',
            'run not found': 'Ese run no existe.', 'no runs published': 'Este ámbito no tiene runs publicados.',
            'invalid scope': 'Ámbito no válido.', 'invalid run': 'Run no válido.', 'invalid mode': 'Modo no válido.'}
TABLE_ROWS = 40

def fmt_num(x, digits=0) -> str; fmt_int(x) -> str; fmt_pct(x, digits=1) -> str; fmt_prob(p, digits=0) -> str
def fmt_range(lo, hi, digits=1) -> str; fmt_date(iso) -> str; fmt_datetime(iso) -> str   # Europe/Madrid (zoneinfo; UTC+1 si falta tzdata)
def environment() -> jinja2.Environment      # cacheado en App.get_().templates (app.set('templates', env)); filtros 'num','int','pct','prob','range','date','datetime'
def parse_state(query: dict) -> dict         # {'scope': str|None (check_scope si viene), 'mode': check_mode(query.get('mode') or 'forecast'), 'run': check_run(query.get('run'))}
def published_scopes() -> list[dict]         # filas de site().scopes() con simulable True (orden del catálogo)
def resolve_scope(scope: str|None) -> str    # scope si está en el manifest; None → primer publicado (HttpError(503,'no bundle published yet') si no hay); dado y sin publicar → HttpError(404,'scope not published')
def part(scope, run, name, mode=None) -> dict   # bundle.loads(site().run_file(...))['data']
def query_string(state) -> str               # '?scope=es&mode=forecast[&run=...]' (urlencode)
def api_url(scope, part, mode=None, run=None, fmt=None) -> str   # '/api/v1/forecast/{scope}[/{mode}]/{part}[?run=..&format=..]'
def common_context(state, active: str) -> dict          # ver abajo
def index_context(state, active: str = '/') -> dict     # common + headline (data[mode]), majority (lista de {block, p, width, color}), initial
```

  `common_context(state, active)` resuelve `scope` y `run` (`state` devuelto ya completo: `scope`, `mode`,
  `run`, `pinned: bool`), lee `meta`, y devuelve: `site_title`, `pages` (lista de `{href, label,
  active}`; `active` es el `href` de la página actual), `state`, `query` (`query_string`
  sin `run` cuando no está fijado), `scopes` (`published_scopes()`), `manifest` (`manifest_data()`),
  `meta`, `catalog` (`{name: {fullname, color, block}}` desde `meta.parties`), `mode_label`
  (`'Hoy'`/`'Elección'`), `title_suffix` (`'Estimación a {date(as_of)}'` en nowcast, `'Pronóstico para
  el {date(event_date)}'` en forecast). `index_context` añade `headline` (`headline.data[mode]`),
  `majority` (por bloque de `p_majority`: `{'block', 'p', 'width': round(p*100, 1), 'color': color del
  primer partido del bloque en `meta.bmaps.vs` o `blocks`, gris `#9e9e9e` si no}`) e `initial = {'state',
  'meta', 'headline' (data completo), 'vote', 'summary', 'runs' (history data)}`.

- Produces `mtpy/controllers/Page.py`:

```python
class Page(Controller):
    active = '/'                      # href de la página en el nav; las subclases lo fijan
    def state(self) -> dict            # pages.parse_state(self.request.query)
    def render(self, template: str, **context) -> str   # env.get_template(template).render(**context); text/html; etag md5; set_cache(60)
    def error_page(self, status: int, message: str) -> str  # error.html con {site_title, status, message (MESSAGES.get(message, message) o genérico)}; set_status_code; text/html; no-store; sin etag
    async def dispatch(self, action, **kwargs)           # como Controller.dispatch, pero HttpError → error_page(status, message); Exception → logger.error(traceback) + error_page(500, 'Error interno'); luego send()
    async def not_found_action(self) -> str              # error_page(404, 'Página no encontrada')
```

- Produces `mtpy/controllers/Index.py`: `class Index(Page)`, `active = '/'`, `async def
  index_action(self) -> str: return self.render('index.html', **pages.index_context(self.state(),
  active=self.active))`.
- Nota para plantillas y tests: con autoescape, un `&` dentro de un atributo se emite como `&amp;`
  (`href="/promedio?scope=es&amp;mode=nowcast"`); es HTML válido y los tests lo esperan así.
- Produces plantillas (textos en español, sin inline scripts ni `on*=`):
  - `base.html`: `<!doctype html><html lang="es">`, `<head>` con charset, viewport, `<title>{{ site_title
    }} · {% block title %}{% endblock %}</title>`, `<link rel="stylesheet" href="/dist/css/site.css">`,
    `<script defer src="/dist/vendor/echarts-5.6.0.min.js"></script>`, `{% block scripts %}{% endblock %}`;
    `<body>`: `{% if manifest.freeze.active %}<div id="freeze-banner">{{ manifest.freeze.message or
    'Publicación congelada.' }}</div>{% endif %}`, `{% include '_header.html' %}`, `<main id="main">{%
    block content %}{% endblock %}</main>`, `{% include '_footer.html' %}`, `{% if initial is defined
    %}<script type="application/json" id="initial-data">{{ initial|tojson }}</script>{% endif %}`.
  - `_header.html`: `<header id="site-header">` con `<h1><a href="/{{ query }}">{{ site_title }}</a></h1>`,
    `<nav aria-label="Secciones">` con `<a href="{{ page.href }}{{ query }}"{% if page.active %}
    aria-current="page"{% endif %}>` por página, y `<form id="controls" class="controls" method="get"
    action="">` con `<label>Ámbito <select name="scope" id="scope">` (`<option value="{{ s.code }}"{% if
    s.code == state.scope %} selected{% endif %}>{{ s.name }}</option>`), `<fieldset id="mode"><legend>
    Fecha</legend>` con dos radios `name="mode"` (`nowcast` "Hoy", `forecast` "Elección", `checked` según
    `state.mode`), `{% if state.pinned %}<input type="hidden" name="run" value="{{ state.run }}">{% endif
    %}` y `<button type="submit" class="nojs-only">Ver</button>`.
  - `_footer.html`: `<footer id="site-footer">` con `Run {{ meta.run_id }} · Estimación a {{
    meta.as_of|date }} · último sondeo {{ meta.date_last|date }} · commit {{ meta.commit or '–' }}{% if
    meta.dirty %} (sucio){% endif %}`, las tres líneas de `manifest.attribution` (`polls`, `results`,
    `model`) y `<a href="/api/v1/manifest">Manifiesto de datos</a>`.
  - `error.html`: documento independiente (no extiende `base.html`): mismo `<head>` mínimo (sin ECharts),
    `<header>` con el título enlazado a `/`, `<main><p class="status error">{{ message }}</p><p
    class="muted">Error {{ status }}</p></main>`.
  - `index.html`: `{% extends 'base.html' %}`; `title` "Portada"; `scripts`: `<script type="module"
    src="/dist/js/pages/index.js"></script>`; `content`: las secciones del `web/index.html` actual con los
    mismos `id` (`headline`, `headline-table`, `vote-chart`, `seats-chart`, `hemicycle`, `majority`,
    `evolution`) y `h2` con `{{ title_suffix }}`; la tabla del titular renderizada (`<caption>`, `<th
    scope="col">` Partido/Voto/Escaños/Primero; por fila: punto de color `<span class="dot"
    style="background-color: {{ catalog[row.name].color }}"></span>` — el único `style` inline, permitido
    por la CSP (`style-src 'unsafe-inline'`) —, `fullname` con `title="{{ row.name }}"`, `{{ row.pct|pct
    }} <span class="muted">{{ row.lo|range(row.hi) }}</span>`, `{{ row.seats|int }} <span class="muted">{{
    row.seats_lo|range(row.seats_hi, 0) }}</span>`, `{{ row.p_first|prob }}`); el bloque `#majority`
    renderizado (`.majority-row` con `<strong>`, `.majority-track` + `.majority-bar` con `style="width:
    {{ m.width }}%; background-color: {{ m.color }}"` y `.majority-text` "{{ m.p|prob }} de probabilidad de
    mayoría absoluta"); el párrafo "Qué es esto" con `<a href="/promedio{{ query }}">`.
- Produces `api.py`: `from mtpy.lib import pages, webapi`; `api = mtpy.api(routes=pages.ROUTES +
  webapi.ROUTES, not_found=pages.NOT_FOUND)`.
- Produces en `tests/test_pages.py`: `site_api(app)` = `mtpy.api(routes=pages.ROUTES + webapi.ROUTES,
  not_found=pages.NOT_FOUND)` sobre un `fresh_app` con `fs`, `data` y el paquete de `write_bundle` (de
  `tests.test_webapi_unit`); `html(body) -> str` decodifica.

- [ ] **Step 1: Crear `tests/test_pages.py`**

```python
"""Tests de las páginas servidas por los controladores (`Page`, `Index`, `Promedio`) con el arnés ASGI."""
import json
import os
import re

import pytest

from mtpy import mtpy
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, pages, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import ROOT, RUN, fixture_data, write_bundle


def site_api(app):
    """API completa (páginas + /api/v1 + not_found) registrada en `app`."""
    return mtpy.api(routes=pages.ROUTES + webapi.ROUTES, not_found=pages.NOT_FOUND)


@pytest.fixture
def api(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return site_api(fresh_app)


def html(body):
    return body.decode('utf-8')


def initial(body):
    match = re.search(r'<script type="application/json" id="initial-data">(.*?)</script>', html(body), re.S)
    assert match, 'sin bloque initial-data'
    return json.loads(match.group(1))


def test_home_is_rendered_html_with_cache_headers(api):
    status, headers, body = call(api, '/')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=60' and headers['etag'].startswith('"')
    assert '<title>Pronóstico electoral · Portada</title>' in page
    assert 'Pronóstico para el' in page and 'aria-current="page"' in page
    assert '<option value="es" selected>' in page and 'name="mode" value="forecast" checked' in page.replace('\n', ' ')
    assert 'id="headline-table"' in page and '<th scope="col">Partido</th>' in page
    assert fixture_data('headline')['forecast']['parties'][0]['name'] in page
    assert '/dist/css/site.css' in page and '/dist/vendor/echarts-5.6.0.min.js' in page and '/dist/js/pages/index.js' in page
    assert 'onclick=' not in page and '<script>' not in page
    assert page.count('<script') == 3  # ECharts (defer), el módulo de la página y el bloque JSON


def test_home_initial_data_has_the_page_parts_with_one_run(api):
    data = initial(call(api, '/')[2])
    assert set(data) == {'state', 'meta', 'headline', 'vote', 'summary', 'runs'}
    assert data['state'] == {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert data['meta']['run_id'] == RUN and data['headline']['run_id'] == RUN
    assert data['runs']['runs'][0]['run_id'] == RUN


def test_nowcast_and_pinned_run_are_reflected(api):
    status, headers, body = call(api, '/', query='mode=nowcast&run=' + RUN)
    page = html(body)
    assert status == 200 and 'Estimación a' in page
    assert 'name="mode" value="nowcast" checked' in page.replace('\n', ' ')
    assert 'type="hidden" name="run" value="{}"'.format(RUN) in page
    assert initial(body)['state'] == {'scope': 'es', 'mode': 'nowcast', 'run': RUN, 'pinned': True}
    assert 'href="/promedio?scope=es&amp;mode=nowcast&amp;run={}"'.format(RUN) in page


def test_freeze_banner_and_attribution(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda electoral'})
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert 'id="freeze-banner"' in page and 'Veda electoral' in page
    assert 'Sondeos: Wikipedia' in page and 'href="/api/v1/manifest"' in page


@pytest.mark.parametrize('path, query, status, text', [
    ('/', 'scope=es-md', 404, 'no tiene pronóstico publicado'),
    ('/', 'scope=es-xx', 400, 'Ámbito no válido'),
    ('/', 'run=2026-10-08', 400, 'Run no válido'),
    ('/', 'run=20200101-000000', 404, 'Ese run no existe'),
    ('/', 'mode=tomorrow', 400, 'Modo no válido'),
    ('/nope', '', 404, 'Página no encontrada'),
])
def test_page_errors_are_html_and_not_cached(api, path, query, status, text):
    got, headers, body = call(api, path, query=query)
    assert got == status and headers['content-type'] == 'text/html; charset=utf-8'
    assert headers['cache-control'] == 'no-store' and 'etag' not in headers
    assert text in html(body) and 'Pronóstico electoral' in html(body)


def test_api_not_found_stays_json(api):
    status, headers, body = call(api, '/api/v1/nope')
    assert status == 404 and json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_without_bundle_the_home_is_a_503_page(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    status, headers, body = call(site_api(fresh_app), '/')
    assert status == 503 and 'Todavía no hay ningún pronóstico publicado' in html(body)
    assert headers['cache-control'] == 'no-store'


def test_default_scope_is_the_first_published(api, fresh_app):
    """Con `es` publicado, `?scope=` ausente lleva a `es`; un ámbito publicado adicional se lista en el selector."""
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert '<option value="es" selected>' in page and '<option value="es-md">Madrid</option>' in page


def test_templates_compile_with_strict_undefined():
    env = pages.environment()
    names = sorted(f for f in os.listdir(pages.TEMPLATES_DIR) if f.endswith('.html'))
    assert {'base.html', '_header.html', '_footer.html', 'error.html', 'index.html'} <= set(names)
    for name in names:
        env.get_template(name)


def test_filters_follow_the_js_rules():
    assert pages.fmt_pct(32.66) == '32,7 %' and pages.fmt_pct(None) == '–'
    assert pages.fmt_num(1234567) == '1.234.567' and pages.fmt_num(3.14159, 1) == '3,1'
    assert pages.fmt_prob(0.995) == '> 99 %' and pages.fmt_prob(0.003) == '< 1 %' and pages.fmt_prob(0) == '0 %'
    assert pages.fmt_prob(1) == '100 %' and pages.fmt_prob(0.968) == '97 %' and pages.fmt_prob(None) == '–'
    assert pages.fmt_range(27.31, 38.02) == '27,3–38,0' and pages.fmt_range(118.0, 159.5, 0) == '118–160'
    assert pages.fmt_date('2026-10-13') == '13 oct 2026' and pages.fmt_date('2026-09-05') == '5 sept 2026'
    assert pages.fmt_date('garbage') == '–' and pages.fmt_date(None) == '–'
    assert pages.fmt_datetime('2026-10-08T18:11:02Z') == '8 oct 2026, 20:11'
    assert pages.api_url('es', 'polls', run=RUN, fmt='csv') == '/api/v1/forecast/es/polls?run={}&format=csv'.format(RUN)
    assert pages.api_url('es', 'vote', mode='forecast') == '/api/v1/forecast/es/forecast/vote'
```

  (`write_bundle(fs, scope, run_id, manifest)` ya acepta `scope` y `run_id`; con `manifest=True` añade la
  entrada al manifest existente vía `update_manifest`, que fusiona.)

- [ ] **Step 2: Cambiar el test de `Index` en `tests/test_api_asgi.py`**

El test de la línea ~356 que espera `{'status': 'ok', 'message': 'API Home'}` pasa a comprobar que `/`
con un `fresh_app` sin paquete devuelve 503 y `text/html; charset=utf-8` con el texto "pronóstico"
(la portada es ahora una página). Si el test también cubría `mtpy.api(routes=...)`, mantener esa parte.

- [ ] **Step 3: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_pages.py tests/test_api_asgi.py -q`
Expected: `ImportError: cannot import name 'pages' from 'mtpy.lib'` en la recogida de `test_pages.py`; el
test de `Index` falla (sigue devolviendo JSON).

- [ ] **Step 4: Implementar**

`mtpy/mtpy.py` (`not_found`), `mtpy/lib/pages.py`, `mtpy/controllers/Page.py`, `Index.py`, las cinco
plantillas, `api.py`; `git rm web/index.html`. `fmt_datetime` usa `zoneinfo.ZoneInfo('Europe/Madrid')`
con respaldo a UTC+1 (misma lógica que `publish._madrid_tz`, sin importar `publish`). En `Page.render`
el md5 se calcula sobre los bytes UTF-8 del HTML.

- [ ] **Step 5: Ejecutar los tests y la suite**

Run: `python -m pytest tests/test_pages.py tests/test_api_asgi.py tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: todos PASS (`test_pages.py`: 10 + 6 parametrizados); suite ≈ 317 passed.

- [ ] **Step 6: Comprobación rápida con uvicorn**

Run: `S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning & sleep 8; curl -s -D - http://127.0.0.1:8000/ -o /tmp/home.html | head -8; grep -c 'initial-data' /tmp/home.html; curl -s -o /dev/null -w '%{http_code} %{content_type}\n' http://127.0.0.1:8000/nope; curl -s -o /dev/null -w '%{http_code} %{content_type}\n' http://127.0.0.1:8000/api/v1/nope; kill %1`
Expected: `200`, `content-type: text/html; charset=utf-8`, `cache-control: public, max-age=60`; `1`;
`404 text/html; charset=utf-8`; `404 application/json; charset=utf-8`.

- [ ] **Step 7: Commit**

```bash
git add mtpy/mtpy.py mtpy/lib/pages.py mtpy/controllers/Page.py mtpy/controllers/Index.py api.py web/templates tests/test_pages.py tests/test_api_asgi.py tests/test_web_routes.py
git rm -q web/index.html
git commit -m "$(printf 'Render the home page from a Jinja2 controller\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 3: Página `/promedio` en servidor

**Files:**
- Create: `mtpy/controllers/Promedio.py`, `web/templates/promedio.html`
- Modify: `mtpy/lib/pages.py` (`promedio_context`, `table_rows`), `tests/test_pages.py`
- Delete: `web/promedio.html`

**Interfaces:**
- Produces `pages.table_rows(polls: list[dict], parties: list[str], n: int = TABLE_ROWS) -> list[list[str]]`:
  los `n` últimos sondeos, los más recientes primero (orden estable: invertir y ordenar por `date`
  descendente), celdas ya formateadas: `fmt_date(date)`, `pollster or DASH`, `sponsor or DASH`,
  `fmt_int(sample_size)`, y `fmt_num(valor, 1)` por partido.
- Produces `pages.promedio_context(state, active='/promedio') -> dict`: `common_context` +
  `series`, `polls`, `projection` (del modo), `vote` (del modo) como `data`; `table_parties`
  (`meta.bmaps.main` ∩ `series.parties`, o todas si vacío), `table_rows`, `n_polls` (`meta.n_polls`),
  `csv_href` (`api_url(scope, 'polls', run=run, fmt='csv')`), `series_subtitle` (`'a {date(as_of)}'` en
  nowcast, `'{date(as_of)} → {date(vote.when)}'` en forecast), e `initial = {'state', 'meta', 'series',
  'polls', 'projection', 'vote'}`.
- Produces `mtpy/controllers/Promedio.py`: `class Promedio(Page)`, `active = '/promedio'`,
  `index_action` → `render('promedio.html', **pages.promedio_context(state))`.
- Produces `web/templates/promedio.html`: extiende `base.html`; `title` "Promedio de sondeos"; `scripts`
  → `/dist/js/pages/promedio.js`; `content`: sección `#series` (`h2` "Promedio de sondeos y proyección
  <span class="muted">{{ series_subtitle }}</span>", `<div class="chart chart-tall" id="series-chart">`,
  el párrafo explicativo con `<a href="/{{ query }}">portada</a>`) y sección `#polls` (`h2` "Sondeos del
  ciclo", `<div class="table-wrap">` con `<table>`: `<caption>Últimos {{ table_rows|length }} de {{
  n_polls|int }} sondeos · <a href="{{ csv_href }}">Descargar CSV</a></caption>`, `<th scope="col">`
  Fecha/Casa/Patrocinador/Muestra y un `<th scope="col" title="{{ catalog[name].fullname }}">{{ name }}</th>`
  por partido, filas de `table_rows`).

- [ ] **Step 1: Añadir los tests a `tests/test_pages.py`**

```python
def test_promedio_is_rendered_with_the_polls_table_and_csv_link(api):
    status, headers, body = call(api, '/promedio', query='scope=es&mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Pronóstico electoral · Promedio de sondeos</title>' in page
    assert 'href="/promedio?scope=es&amp;mode=forecast" aria-current="page"' in page.replace('\n', ' ')
    assert 'id="series-chart"' in page and 'Sondeos del ciclo' in page
    assert 'href="/api/v1/forecast/es/polls?run={}&amp;format=csv"'.format(RUN) in page
    assert '<th scope="col">Fecha</th>' in page and '/dist/js/pages/promedio.js' in page
    data = initial(body)
    assert set(data) == {'state', 'meta', 'series', 'polls', 'projection', 'vote'}
    assert data['projection']['dates'] and data['vote']['when']


def test_table_rows_are_newest_first_and_formatted():
    polls = [
        {'date': '2026-09-05', 'pollster': 'CIS', 'sponsor': None, 'sample_size': 4000, 'PP': 41.0, 'PSOE': 29.0},
        {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': None, 'PP': None, 'PSOE': 30.5},
        {'date': '2026-10-01', 'pollster': 'CIS', 'sponsor': None, 'sample_size': 4000, 'PP': 40.5, 'PSOE': 29.5},
    ]
    rows = pages.table_rows(polls, ['PP', 'PSOE'], n=2)
    assert rows == [['1 oct 2026', 'CIS', '–', '4.000', '40,5', '29,5'], ['1 oct 2026', 'GAD3', 'ABC', '–', '–', '30,5']]
    assert len(pages.table_rows(polls, ['PP'])) == 3
```

  Y en `test_templates_compile_with_strict_undefined` añadir `'promedio.html'` al conjunto exigido; en
  `test_page_errors_are_html_and_not_cached` añadir el caso `('/promedio', 'scope=es-md', 404, 'no tiene
  pronóstico publicado')`.

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "promedio or table_rows"`
Expected: `/promedio` → 404 HTML "Página no encontrada" (sin ruta) y `AttributeError: module
'mtpy.lib.pages' has no attribute 'table_rows'`.

- [ ] **Step 3: Implementar y borrar `web/promedio.html`**

- [ ] **Step 4: Ejecutar los tests y la suite**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: PASS; suite ≈ 320 passed.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages.py mtpy/controllers/Promedio.py web/templates/promedio.html tests/test_pages.py
git rm -q web/promedio.html
git commit -m "$(printf 'Render the poll average page from a Jinja2 controller\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 4: JavaScript: dibujar desde `initial-data`, controles por formulario

**Files:**
- Create: `web/dist/js/pages/common.js`
- Modify: `web/dist/js/pages/index.js`, `web/dist/js/pages/promedio.js`, `web/dist/css/site.css`,
  `tests/test_web_routes.py`
- Delete: `web/dist/js/state.js`, `web/dist/js/layout.js`, `web/dist/js/api.js`

**Interfaces:**
- Consumes: el bloque `#initial-data` (claves por página, Tareas 2-3), los `id` de las plantillas, los
  módulos `format.js`, `catalog.js`, `charts/*.js` (sin cambios).
- Produces `web/dist/js/pages/common.js`: `readInitial() -> object` (`JSON.parse` del `textContent` de
  `#initial-data`), `wireControls()` (añade la clase `js` al `form#controls`, y en `change` del `select`
  o de los radios llama a `form.requestSubmit()`), `mapNames(names, fn)`, `showError(message)` (escribe
  en `#status` con la clase `error` y lo muestra), `mountWhenReady(fn)` (ejecuta `fn` cuando
  `window.echarts` existe: tras `DOMContentLoaded` el script `defer` ya cargó; si no, espera al evento
  `load`).
- Produces `pages/index.js`: lee `initial`, construye `Catalog(meta)`, dibuja `renderBars` (voto y
  escaños), `renderHemicycle`, `renderEvolution` con los mismos datos y opciones que hoy (la tabla del
  titular y el bloque de mayorías ya vienen del servidor: no se renderizan en JS); sin `fetch`, sin
  `state.js`/`layout.js`; errores → `showError('No se pudieron dibujar los gráficos.')` y
  `console.error`.
- Produces `pages/promedio.js`: lee `initial`, dibuja `renderSeries` con las mismas opciones que hoy
  (`selected` = `meta.bmaps.main ∩ series.parties`, `anchor` = `meta.date_last`); la tabla y el CSV
  vienen del servidor. Conserva `export function tableRows` solo si sigue usándose (no: se elimina; la
  lógica está en `pages.table_rows`).
- Produces CSS: `.controls.js .nojs-only { display: none; }`, `.status.error` (si no existe), y nada
  más.
- Produces `tests/test_web_routes.py`: `test_every_api_literal...` pasa a buscar los literales `/api/v1`
  en las plantillas (`web/templates/*.html`, hoy solo `/api/v1/manifest` en el pie), con la sustitución
  de `{{ ... }}` por `x`; el JS ya no debe contener literales `/api/v1` (`assert` de ausencia); las URL que
  construye `pages.api_url` quedan cubiertas por `test_filters_follow_the_js_rules`;
  `test_referenced_assets_exist` sobre las plantillas
  (`/dist/` existe; páginas en `pages.ROUTES` o `/`; sin `onclick=`; sin `<script>` inline salvo
  `type="application/json"`).

- [ ] **Step 1: Reescribir los módulos de página, crear `common.js`, borrar los tres módulos**

Mantener los `renderX` y sus opciones tal cual están en los módulos actuales (copiar los bloques de
`paint()` que construyen `colors`, `fullnames`, filas ordenadas, `nSeats`, `majority`, `selected`,
`anchor`). El `#content` deja de estar `hidden`: el HTML ya está pintado por el servidor.

- [ ] **Step 2: Comprobaciones**

Run: `for f in $(find web/dist/js -name '*.js'); do node --check "$f" || echo "FAIL $f"; done; python -m pytest tests/test_web_routes.py tests/test_pages.py -q`
Expected: sin `FAIL`; tests PASS. Renderizado SSR opcional de `series.js` con los datos de
`files/site/v1` (como en la fase 2) para confirmar que las opciones no cambiaron.

- [ ] **Step 3: Comprobación funcional sin nginx (solo HTML)**

Run: `S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning & sleep 8; curl -s http://127.0.0.1:8000/promedio | grep -c 'initial-data'; curl -s http://127.0.0.1:8000/ | grep -o 'class="controls[^"]*"'; kill %1`
Expected: `1`; `class="controls"` (la clase `js` la añade el navegador).

- [ ] **Step 4: Commit**

```bash
git add web/dist/js/pages web/dist/css/site.css tests/test_web_routes.py
git rm -q web/dist/js/state.js web/dist/js/layout.js web/dist/js/api.js
git commit -m "$(printf 'Draw the charts from the embedded page data and submit the controls form\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 5: nginx (`/dist/` + proxy de todo), scripts y comprobación local

**Files:**
- Modify: `deploy/docker/nginx.conf`, `deploy/nginx-check.sh`, `deploy/check-api.sh`, `deploy/smoke.sh`

**Interfaces:**
- Produces `nginx.conf` (a partir del actual): quitar `root`, `index`, `location ^~ /vendor/`, `location
  ~* \.(js|css)$` y el `try_files` de `location /`; renombrar `location ^~ /api/` a `location /` (es el
  proxy de todo, páginas incluidas) conservando su contenido; añadir antes:

```nginx
        location ^~ /dist/vendor/ {
            alias /app/web/dist/vendor/;
            expires 1y;
        }

        location ^~ /dist/ {
            alias /app/web/dist/;
            expires -1;
        }
```

  (`/healthz` se mantiene; un fichero inexistente bajo `/dist/` da el 404 de nginx.)
- Produces `nginx-check.sh`: reescribe también `alias /app/web/dist/vendor/` → `<repo>/web/dist/vendor/`
  y `alias /app/web/dist/` → `<repo>/web/dist/`.
- Produces `check-api.sh`: tipo `html` en `check` (content-type `text/html`); nuevas comprobaciones `check
  / 200 html`, `check "/promedio?scope=es&mode=nowcast" 200 html`, `check /nope 404 html`, y dos solo
  con nginx (`n/a sin nginx` si el 404 viene con `text/html` de Python): `/dist/vendor/echarts-5.6.0.min.js`
  200 `application/javascript`, `/dist/css/site.css` 200 `text/css`.
- Produces `smoke.sh`: `check_static /dist/vendor/echarts-5.6.0.min.js application/javascript`,
  `check_header /dist/vendor/echarts-5.6.0.min.js 'max-age=31536000'`, `check_header /dist/js/pages/index.js
  'no-cache'`, `check_header / 'max-age=60'`, `check_static /promedio text/html`, y que `/` contenga
  `initial-data` (`curl -s ... | grep -q initial-data`).

- [ ] **Step 1: Editar los cuatro ficheros**

- [ ] **Step 2: Sintaxis y comprobación local completa**

Run: `bash deploy/nginx-check.sh`
Expected: `syntax is ok` / `test is successful`.

Después, el nginx local (Homebrew) en 8043 con la conf reescrita como hace `nginx-check.sh` (más
`listen 8043`, logs y pid en el temporal, sin `daemon off;`) y uvicorn en 8000:

```bash
S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning & sleep 8
# arrancar nginx con la copia reescrita; luego:
for p in / "/promedio?scope=es&mode=nowcast" /dist/vendor/echarts-5.6.0.min.js /dist/js/pages/index.js /dist/css/site.css /dist/nope /nope /api/v1/manifest /healthz; do
  printf '%-45s ' "$p"; curl -s -o /dev/null -D - "http://127.0.0.1:8043$p" | grep -i '^HTTP\|^content-type\|^cache-control\|^x-cache' | tr '\n' ' '; echo
done
bash deploy/check-api.sh http://127.0.0.1:8043
# parar nginx y uvicorn; comprobar con lsof que 8043 y 8000 quedan libres
```
Expected: `/` y `/promedio` 200 `text/html` con `max-age=60` y `X-Cache` (MISS y luego HIT al repetir);
vendor 200 `application/javascript` `max-age=31536000`; `index.js` `no-cache`; `site.css` 200 `text/css`;
`/dist/nope` 404 **sin** `text/html` de Python (404 de nginx); `/nope` 404 `text/html` (Python);
`/api/v1/manifest` como hasta ahora; `/healthz` 200; `check-api.sh` todo OK.

- [ ] **Step 3: Commit**

```bash
git add deploy/docker/nginx.conf deploy/nginx-check.sh deploy/check-api.sh deploy/smoke.sh
git commit -m "$(printf 'Proxy the pages to gunicorn and serve only /dist from nginx\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 6: Documentación y enmiendas de la spec general

**Files:**
- Modify: `deploy/README.md`, `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`,
  `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md` (estado de cierre),
  `docs/superpowers/plans/2026-10-09-web-enrutado-controladores.md` (cabecera "Estado"),
  `docs/web/contrato.md` (nota en "Rutas de la API": las páginas no son rutas de la API)

- [ ] **Step 1: README, sección "La web"**

Reescribir: nginx proxya `/` y `/api/` a gunicorn y sirve solo `/dist/`; las páginas las renderizan
`Index` y `Promedio` con Jinja2 (`web/templates/`), los recursos viven en `web/dist/`; cómo añadir una
página (controlador + plantilla + entrada en `pages.ROUTES`/`PAGES`); desarrollo en el Mac con el nginx
local de Luis: añadir a su configuración las dos `location` de `/dist/` (con `alias` al `web/dist` del
repositorio) y el proxy a `127.0.0.1:8000`, y arrancar `S3_BUCKET= python -m uvicorn api:api --port
8000`; `deploy/nginx-check.sh`, `check-api.sh` (nuevas rutas) y `smoke.sh`; retirar las instrucciones de
`http.server` y `?api=`.

- [ ] **Step 2: Enmiendas de la spec general**

En "Frontend": sustituir el párrafo del estado en la query string y `state.js` por el renderizado en
servidor (controlador + Jinja2, formulario GET, `initial-data`, JS solo dibuja); en "Infraestructura":
nginx sin `root`, `location /dist/`, proxy de páginas; en "Pruebas": `tests/test_pages.py`; añadir al
final de "Estado al cierre de la fase 2" un párrafo "Enmienda del 2026-10-09: páginas desde los
controladores" que remite a la spec nueva. En la spec nueva, añadir "## Estado al cierre (fecha)" con
ficheros, commits, tests y lo pendiente de Luis (smoke Docker, navegador, `location /dist/` en su nginx
local). En el plan, la cabecera `> **Estado:** completado ...`.

- [ ] **Step 3: Suite y commit**

Run: `python -m pytest -m "not integration" -q`
Expected: ≈ 320 passed.

```bash
git add deploy/README.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md docs/superpowers/plans/2026-10-09-web-enrutado-controladores.md docs/web/contrato.md
git commit -m "$(printf 'Document the controller-rendered pages and the /dist layout\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Tareas de Luis (fuera de la sesión)

- [ ] Añadir las dos `location /dist/` y el proxy a su nginx local y recorrer `/` y `/promedio` (con y
  sin JS: el botón "Ver" debe funcionar sin JS; con JS el cambio de ámbito/modo recarga la página).
- [ ] Con Docker: `bash deploy/smoke.sh` (paquete local) y con `deploy/elections.env`.
- [ ] Decidir si se sube `jinja2` a 3.1.6.
