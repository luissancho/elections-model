# Web, fase 3: escaños, autonómicos e histórico — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir la página `/escanos` (escaños por partido y bloque con sus distribuciones, calculadora
de coaliciones sobre `dist`, circunscripciones con escenario central y detalle por circunscripción,
abanico de voto por horizonte, evolución de los escaños publicados y descargas CSV), dejar las tres
páginas listas para los ámbitos autonómicos publicados con `scopes: "all"`, y cerrar los flecos de las
fases 2 y del rediseño que se aplazaron a esta fase.

**Architecture:** Mismo patrón que `/` y `/promedio` (rediseño del 2026-10-09): `Escanos(Page)` en
`mtpy/controllers/Escanos.py` renderiza `web/templates/escanos.html` con el contexto de
`pages.escanos_context` (tablas en HTML; `initial-data` con `summary`, `dist`, `fan` y `runs`); el
módulo `web/dist/js/pages/escanos.js` dibuja histogramas, barra apilada, abanico y evolución con tres
gráficos nuevos (`charts/histogram.js`, `charts/stacked.js`, `charts/fan.js`) y calcula las coaliciones
en el navegador con `web/dist/js/seats.js` (estado en el fragmento `#coalition=`). Las circunscripciones
se eligen con `?region=` (formulario GET como el de ámbito y modo). No cambian el paquete, el job de
publicación, la API, nginx ni la imagen Docker.

**Tech Stack:** Python 3.11 (`python` del venv), Jinja2 3.1.6, mtpy (ASGI propio), pytest 9; HTML +
módulos ES + ECharts 5.6.0 vendorizado; nginx local (Homebrew 1.29.5, puerto 8043) y Node 25 solo para
`node --check`, un test de paridad y renderizados SSR.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` (tabla "Fases", fila 3;
"Frontend" con sus enmiendas; "Contrato de API"; "Pruebas"; "Verificación" fases 3-5; "Estado al
cierre de la fase 2" para los flecos diferidos) y
`docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md` (patrón de página; "Estado al
cierre" para los menores aplazados). Contrato del paquete: `docs/web/contrato.md`.

## Decisiones de diseño de la fase 3 (supuestos; Luis puede corregirlos antes de ejecutar)

- **D1. Una página nueva, `/escanos`**, en el menú como "Escaños" (Portada · Promedio · Escaños), con
  el mismo estado `scope`/`mode`/`run` y los mismos controles de cabecera que las otras dos.
- **D2. Secciones, en este orden:** (1) escaños por partido: tabla de `summary.parties` y un
  histograma por partido a partir de `dist` (V6); (2) bloques: tablas de `summary.blocks` y
  `summary.vs` con la probabilidad de mayoría absoluta y una barra apilada de los bloques; (3)
  calculadora de coaliciones; (4) circunscripciones (V7): tabla general con el escenario central
  (`scenario`, coherente: cada fila suma los escaños de la circunscripción) y detalle de una
  circunscripción con medianas, intervalos y probabilidad de escaño (`districts`); (5) abanico de voto
  por horizonte (V8, `fan`); (6) evolución de los escaños de cada publicación (`history`); (7)
  descargas CSV de las cinco partes.
- **D3. Calculadora solo en JavaScript** (`seats.js`, como fija la spec: "de `dist` a histogramas,
  cuantiles y P(suma ≥ mayoría)"). Las casillas de los partidos las renderiza el servidor; la selección
  vive en el fragmento de la URL (`#coalition=PP,VOX`, `history.replaceState`), así que los enlaces se
  comparten sin crear entradas en la caché de nginx. Por defecto, los partidos del primer bloque de
  `summary.vs`. Sin JS, la sección muestra las casillas y una nota en `<noscript>`; las probabilidades
  de los bloques fijos ya están en las tablas del servidor. Los cuantiles siguen la regla de
  `Stat.quantile` (cuantil empírico que promedia los dos valores centrales cuando `q·n` es entero), la
  misma de `summary`, para que una "coalición" de un solo partido dé el intervalo de la tabla.
- **D4. `region` es un parámetro de query propio de `/escanos`** (no entra en `parse_state` ni en los
  enlaces del menú): entero de la lista `districts.regions`; mal formado → 400 "Circunscripción no
  válida"; desconocido → 404 "Circunscripción no encontrada"; ausente → la circunscripción con más
  escaños (la primera en caso de empate). Un ámbito sin circunscripciones (`districts.regions == []`,
  comunidades uniprovinciales) no muestra la sección y `?region=` da 404.
- **D5. `initial-data` de `/escanos` = `{state, meta, summary, dist, fan, runs}`** (~50 KB más el
  histórico). `districts` (112 KB) y `scenario` solo se renderizan en HTML.
- **D6. Evolución de escaños** reutilizando `charts/evolution.js` con un campo configurable (`pct` en la
  portada, `seats` aquí).
- **D7. Fuera de alcance:** selector de agrupación (`group`; la página muestra todos los partidos, que
  es lo que cuenta para los escaños, y los bloques en sus tablas), ocultación de vistas durante la veda
  (fase 6; decisión editorial de Luis antes del 2026-11-24), `Page.dispatch` duplicado y soporte `HEAD`
  (núcleo, fase 6), publicar `scopes: "all"` contra la RDS (lo hace Luis; aquí se prueba con un ámbito
  autonómico sintético), y los tres menores que se quedan como están: el formulario sin JS con run fijado
  y cambio de ámbito acaba en 404 (caso límite documentado), el `<select>` que envía al pulsar flechas
  (decisión de Luis) y `X-Forwarded-For` repetido (cosmético, nginx de Luis).
- **D8. Flecos que sí entran** (tareas 1 y 2): recorte del `polls` embebido en `/promedio`;
  `resolve_scope` sin la vuelta al orden del manifest (503 si no hay ámbito del catálogo publicado);
  `table_rows` tolerante a sondeos sin fecha; `cache_ttl` con basura; ayudante `serve()` en
  `Forecast`; colores de los gráficos en `THEME`; decimales de los ticks; título "y proyección" solo en
  `forecast`; redacción del `caption`; `aria-label` en todos los gráficos.

## Global Constraints

- PEP8 y docstring numpydoc en inglés en todo lo nuevo de `mtpy/`; español en `tests/`, `docs/`,
  `deploy/*.md` y `deploy/*.sh`; plantillas y textos visibles en español; JS: módulos ES, identificadores
  y comentarios en inglés, sin scripts en línea ni `on*=`, sin `innerHTML` con datos (los nodos que
  cambian se rellenan con `textContent` o se crean con `createElement`).
- `python -m pytest -m "not integration" -q` sin base (hoy 323 tests; `FutureWarning` es error);
  `tests/test_web_routes.py` exige módulo `type="module"` en cada página, recursos existentes, imports
  que resuelven y ningún literal `/api/v1` en el JS.
- Árbol limpio al empezar (rama `dev`, `68522d2`); `git add <ruta>` explícito por tarea; mensajes cortos
  en inglés con el trailer `Co-Authored-By` del modelo que escribe el commit.
- **Nunca arrancar `mtpy.run()` ni uvicorn sin `S3_BUCKET=` (vacío) delante**; nunca `python job.py`;
  nunca tocar el nginx de Luis en el 8080. El paquete local `files/site/v1` (run `20261008-181100`, 13
  partidos, 52 circunscripciones, horizontes `[0, 7, 14, 30, 47]`) alimenta las comprobaciones manuales.
- Patrón de página (spec del rediseño): contexto sobre `webapi.site()` con un solo `run` por página;
  `StrictUndefined`; ningún `|safe`; `initial` con `{{ initial|tojson }}`; cabeceras
  `text/html; charset=utf-8`, `public, max-age=60`, `ETag`; errores HTML `no-store` con texto en
  español (`pages.MESSAGES`).
- Formato: filtros `num/int/pct/prob/range/date/datetime` ya registrados; escaños enteros con `int`,
  intervalos de escaños con `range(…, 0)`, probabilidades con `prob`, porcentajes con `pct`.
- Paquete: `summary.parties/vs/blocks` (15 columnas; `seats` entero o `null` en un bloque vacío),
  `dist` (`n_seats`, `parties`, `seats` matriz simulaciones × partidos), `districts` (`parties`,
  `regions` sin el total, `rows` con `region_id, region, name, pct, pct_lo, pct_hi, seats, seats_mean,
  seats_lo, seats_hi, p_seats`), `scenario` (`simulation`, `parties`, `rows` con `region_id, region,
  seats[]`; la fila `region_id` 0 es el total), `fan` (`horizons`, `rows` con `name, horizon, mean, sd,
  lo, hi`), `history` (`runs`: headlines ascendentes). Mayoría absoluta = `summary.majority`.
- nginx, Docker, `requirements.txt`, el job y la API no cambian (salvo el ayudante interno de
  `Forecast.py`, sin efecto en las rutas).

## Review Focus

1. Ámbito autonómico publicado con `meta.bmaps` sin `max` y `districts.regions == []`: las tres páginas
   responden 200 con `?scope=es-md`, `/escanos` oculta la sección de circunscripciones y la calculadora
   arranca con el primer bloque de `vs` (test en la tarea 4).
2. `?region=abc` → 400 HTML `no-store`; `?region=99` (fuera de `districts.regions`) → 404 HTML; sin
   `?region=` se abre la circunscripción con más escaños (tarea 4).
3. Un partido de `summary` o de un bloque `vs` ausente de `dist.parties` (o al revés): el servidor
   intersecta la coalición por defecto, y el JS salta el histograma de un partido sin columna y
   descarta los nombres desconocidos del fragmento `#coalition=` sin lanzar (tareas 3 y 5).
4. Bloque sin partidos (`seats` `null` en `summary.vs/blocks`) y partido sin escaños en ninguna
   simulación: las tablas muestran `–`, `seat_rows` deja los nulos al final y el histograma del
   partido sigue dibujándose con una sola barra en 0 (tareas 3 y 5).
5. Run fijado (`?run=`): los cinco enlaces CSV, el campo oculto del formulario de circunscripción y los
   enlaces de las filas de la tabla general llevan ese `run` (tareas 3 y 4).

---

### Task 1: Flecos de servidor (`pages.py`, `webapi.py`, `Forecast.py`)

**Files:**
- Modify: `mtpy/lib/pages.py` (`published_scopes`, `resolve_scope`, `table_rows`, `promedio_context`)
- Modify: `mtpy/lib/webapi.py` (`settings`)
- Modify: `mtpy/controllers/Forecast.py` (`part_action`, `mode_part_action`)
- Test: `tests/test_pages.py`, `tests/test_webapi_unit.py`, `tests/test_web_api.py`

**Interfaces:**
- Consumes: `pages.part`, `pages.common_context`, `webapi.site()` (existentes).
- Produces: `pages.chart_polls(polls: dict) -> dict` (`{'parties': list, 'polls': [{'date', 'pollster',
  <partido>: valor, ...}]}`, sin `columns` ni `results`); `pages.resolve_scope(None)` levanta 503 cuando
  ningún ámbito del catálogo está publicado; `Forecast.serve(scope, part, mode=None) -> bytes`.

- [ ] **Step 1: Tests que fallan en `tests/test_pages.py`**

```python
def test_promedio_initial_polls_carry_only_the_chart_columns(api):
    data = initial(call(api, '/promedio')[2])
    parties = set(data['polls']['parties'])
    assert set(data['polls']) == {'parties', 'polls'}
    for poll in data['polls']['polls']:
        assert set(poll) == {'date', 'pollster'} | parties


def test_without_a_published_catalogue_scope_the_home_is_a_503_page(fresh_app, tmp_path):
    """Un manifest cuyo único ámbito no está en el catálogo no se puede pedir por URL: 503, no un `<select>` vacío."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs, scope='zz')
    status, headers, body = call(site_api(fresh_app), '/')
    assert status == 503 and 'Todavía no hay ningún pronóstico publicado' in html(body)


def test_table_rows_tolerate_a_poll_without_date():
    polls = [{'date': '2026-10-01', 'pollster': 'CIS', 'PP': 40.0}, {'pollster': 'GAD3', 'PP': 41.0}]
    rows = pages.table_rows(polls, ['PP'])
    assert rows[0][0] == '1 oct 2026' and rows[1][0] == '–' and rows[1][1] == 'GAD3'
```

Y en `tests/test_webapi_unit.py`:

```python
def test_settings_fall_back_to_the_default_ttl_on_garbage(monkeypatch):
    fake = SimpleNamespace(config=SimpleNamespace(web=SimpleNamespace(prefix='', cache_ttl='abc')))
    monkeypatch.setattr(webapi.App, 'get_', staticmethod(lambda: fake))
    with pytest.warns(UserWarning):
        assert webapi.settings() == {'prefix': bundle.PREFIX, 'cache_ttl': 60.0}
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py tests/test_webapi_unit.py -q -k "chart_columns or published_catalogue or without_date or garbage"`
Expected: 4 FAIL (claves de más en `polls`, 200 en vez de 503, `KeyError: 'date'`, `ValueError` de `float`).

- [ ] **Step 3: Implementar**

- `pages.chart_polls(polls)`: copia `parties` y, por sondeo, `date`, `pollster` y las claves de
  `parties`; `promedio_context` embebe `chart_polls(polls)` en `initial['polls']` (la tabla sigue usando
  `polls['polls']` completo).
- `pages.resolve_scope`: sin `scope`, `codes = [row['code'] for row in published_scopes()]`; si está
  vacío, `HttpError(503, 'no bundle published yet')` (se elimina `or list(manifest['scopes'])`).
- `pages.table_rows`: clave de orden `poll.get('date') or ''` y `fmt_date(poll.get('date'))`.
- `webapi.settings`: `cache_ttl` que no convierte a `float` → `warnings.warn('invalid web.cache_ttl
  {!r}; using {}'.format(ttl, DEFAULT_TTL))` y el valor por defecto.
- `Forecast.serve(self, scope, part, mode=None)`: `check_format`, `resolve_run`, `notice_freeze` y la
  rama CSV/JSON (con el 404 `no csv for this part` cuando `mode is None and part not in CSV_PARTS`);
  `part_action` y `mode_part_action` solo validan `scope`, `mode`, `part` y llaman a `serve`.

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_pages.py tests/test_webapi_unit.py tests/test_web_api.py -q`
Expected: todo PASS (los tests existentes de `Forecast` cubren el refactor: `test_csv_twins`,
`test_meta_and_parts_resolve_latest_or_an_explicit_run`, `test_errors_are_json_and_not_cached`).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages.py mtpy/lib/webapi.py mtpy/controllers/Forecast.py tests/test_pages.py tests/test_webapi_unit.py
git commit -m "Trim the embedded polls, drop the manifest-order scope fallback and harden settings"
```

---

### Task 2: Flecos de JavaScript y plantillas

**Files:**
- Modify: `web/dist/js/charts/base.js` (`THEME`), `web/dist/js/charts/bars.js`,
  `web/dist/js/charts/series.js`, `web/dist/js/pages/index.js`
- Modify: `web/templates/index.html`, `web/templates/promedio.html`
- Modify: `mtpy/lib/pages.py` (`promedio_context`)
- Test: `tests/test_pages.py`, `tests/test_web_routes.py`

**Interfaces:**
- Produces: `THEME.intervalColor = '#333'`, `THEME.markColor = '#555'` en `charts/base.js`;
  `renderBars(el, rows, {…, axisFormatter = formatter})` (los ticks del eje usan `axisFormatter`);
  `promedio_context` añade `series_title` (`'Promedio de sondeos'` en `nowcast`, `'Promedio de
  sondeos y proyección'` en `forecast`).

- [ ] **Step 1: Tests que fallan**

En `tests/test_pages.py`:

```python
def test_promedio_title_mentions_the_projection_only_in_forecast(api):
    assert 'Promedio de sondeos y proyección' in html(call(api, '/promedio', query='mode=forecast')[2])
    page = html(call(api, '/promedio', query='mode=nowcast')[2])
    assert 'Promedio de sondeos' in page and 'y proyección' not in page


def test_promedio_caption_counts_the_polls(api, fresh_app):
    """El fixture tiene 1 sondeo de 120 usados; con n_polls == sondeos mostrados el caption cambia."""
    assert 'Últimos 1 de 120 sondeos' in html(call(api, '/promedio')[2])
    meta = fixture_data('meta') | {'n_polls': 1}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    assert 'Todos los sondeos del ciclo (1)' in html(call(api, '/promedio')[2])
```

(El segundo `call` es el primero que lee `meta` en ese `App`, así que la sobrescritura se ve; el
`api` fixture crea un `App` nuevo por test.)

En `tests/test_web_routes.py`:

```python
def test_chart_containers_have_aria_labels():
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for tag in re.findall(r'<div[^>]*class="chart[^"]*"[^>]*>', text):
            assert 'aria-label="' in tag and 'role="img"' in tag, (os.path.basename(path), tag)
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q -k "projection_only or all_polls or aria_labels"`
Expected: 3 FAIL.

- [ ] **Step 3: Implementar**

- `base.js`: `THEME` gana `intervalColor` y `markColor`; `bars.js` y `series.js` dejan de declarar
  `INTERVAL_COLOR`/`MARK_COLOR` y usan `THEME`.
- `bars.js`: opción `axisFormatter` (por defecto `formatter`) para `xAxis.axisLabel.formatter`;
  `pages/index.js` pasa `axisFormatter: (x) => fmtPct(x, 0)` al gráfico de voto (el de escaños ya usa
  `fmtInt`).
- `promedio.html`: `<h2>{{ series_title }} <span class="muted">…</span></h2>`; `caption`: `{% if n_polls
  > table_rows|length %}Últimos {{ table_rows|length }} de {{ n_polls|int }} sondeos{% else %}Todos los
  sondeos del ciclo ({{ n_polls|int }}){% endif %} · <a …>Descargar CSV</a>`.
- `index.html` y `promedio.html`: cada `<div class="chart…">` lleva `role="img"` y un `aria-label`
  descriptivo (voto previsto por partido con intervalo del 95 %; escaños previstos por partido con
  intervalo; hemiciclo; evolución de la estimación de cada publicación; promedio de sondeos con banda y
  proyección).

- [ ] **Step 4: Tests y sintaxis en verde**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q && for f in web/dist/js/charts/*.js web/dist/js/pages/*.js; do node --check "$f" || exit 1; done`
Expected: PASS y ningún error de `node --check`.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/base.js web/dist/js/charts/bars.js web/dist/js/charts/series.js web/dist/js/pages/index.js web/templates/index.html web/templates/promedio.html mtpy/lib/pages.py tests/test_pages.py tests/test_web_routes.py
git commit -m "Move the chart colours to THEME, fix the axis ticks and the average page copy, label the charts"
```

---

### Task 3: Página `/escanos` en servidor: tablas de partidos y bloques, calculadora, descargas, ruta y menú

**Files:**
- Create: `mtpy/controllers/Escanos.py`, `web/templates/escanos.html`
- Modify: `mtpy/lib/pages.py` (`PAGES`, `ROUTES`, `seat_rows`, `block_rows`, `default_coalition`,
  `csv_links`, `escanos_context`)
- Modify: `web/dist/css/site.css`
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `common_context`, `part`, `api_url`, `block_color`, `Catalog`, `bundle.loads`,
  `site().history` (existentes).
- Produces:
  - `pages.PAGES` añade `{'href': '/escanos', 'label': 'Escaños'}` (tercero); `pages.ROUTES` añade
    `('/escanos', 'escanos', 'index', ['GET'])`.
  - `seat_rows(rows: list) -> list`: filas de `summary` ordenadas por `seats` descendente, `null` al
    final, orden estable.
  - `block_rows(rows: list, meta: dict, parties: Catalog) -> list`: cada fila con `color`
    (`block_color`), en el orden de entrada.
  - `default_coalition(meta: dict, summary: dict, names: list) -> list`: partidos del primer bloque de
    `summary['vs']` (si está vacío, de `summary['blocks']`) según `meta['bmaps']`, intersectados con
    `names` (orden de `names`); `[]` sin bloques.
  - `csv_links(scope: str, mode: str, run: str) -> list`: cinco `{'label', 'href'}` en este orden:
    `summary` "Resumen por partido y bloque", `dist` "Escaños por simulación", `districts`
    "Circunscripciones", `scenario` "Escenario central" (los cuatro con `mode`), `fan` "Abanico por
    horizonte" (sin `mode`); todos con `run` y `fmt='csv'` (`api_url`).
  - `escanos_context(state: dict, region=None, active: str = '/escanos') -> dict`: el contexto común
    más `summary`, `party_rows` (`seat_rows(summary['parties'])`), `block_rows`, `vs_rows`,
    `calculator` (`[{'name', 'fullname', 'color', 'checked'}]` en el orden de `dist['parties']`),
    `csv_links`, `subtitle` (como en `index.html`), y, hasta la tarea 4, `district_table=None`,
    `regions=[]`, `region=None`, `region_rows=[]`; `initial = {'state', 'meta', 'summary', 'dist',
    'fan', 'runs'}` (en ese orden). El parámetro `region` se acepta y se ignora hasta la tarea 4.
  - `Escanos(Page)` con `active = '/escanos'` e `index_action` que renderiza `escanos.html` con
    `pages.escanos_context(self.state(), region=self.request.query.get('region'), active=self.active)`.
  - Plantilla `escanos.html` (extiende `base.html`; título "Escaños"; módulo
    `/dist/js/pages/escanos.js`) con estos identificadores, de los que dependen el JS y los tests:
    `section#parties` (`table#parties-table`, cabeceras Partido · Escaños · Intervalo 95 % · Mín–máx ·
    Media · P(escaño) · P(primero) · P(mayoría)); `div#histograms` con un `div.chart.chart-small`
    por fila de `party_rows` con `data-party="{{ row.name }}"`; `section#blocks` (`div.chart#blocks-chart`,
    `table#blocks-table` y `table#vs-table` con Bloque · Escaños · Intervalo 95 % · P(mayoría absoluta));
    `section#coalition` (`fieldset#coalition-parties` con un `<label>` por entrada de `calculator`:
    `<input type="checkbox" name="coalition" value="{{ name }}"{% if checked %} checked{% endif %}>`,
    punto de color y `fullname`; `p#coalition-result` con el texto "Marca partidos para calcular su
    mayoría."; `div.chart#coalition-chart`; `<noscript><p class="muted">La calculadora necesita
    JavaScript.</p></noscript>`); `section#districts` solo si `district_table` (tarea 4);
    `section#fan` (`div.chart#fan-chart`); `section#evolution` (`div.chart#seats-evolution`);
    `section#downloads` (`<ul>` con los `csv_links`). Todos los `div.chart` con `role="img"` y
    `aria-label`. Textos "Qué es esto" breves en cada sección (`p.about`), como en la portada.
  - CSS: `.grid-small { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px; }`, `.chart.chart-small { height: 160px; }`, `.coalition-parties { display: flex;
    flex-wrap: wrap; gap: 8px 16px; }`, `.coalition-result { font-size: 1.1rem; margin: 8px 0; }`.

- [ ] **Step 0: Alinear los fixtures con la forma real del contrato**

`tests/fixtures/bundle/meta.json`: `bmaps` pasa de `{"Derecha": ["PP"]}` a `{"max": ["PP"], "main":
["PP"], "vs": {"Derecha": ["PP"]}, "blocks": {"Derecha": ["PP"]}}` (la forma de `meta.bmaps` del paquete
real; el esquema solo exige `dict`). `tests/fixtures/bundle/summary.json`: las filas de `vs` y `blocks`
se llaman `Derecha` (hoy `PP`), con los mismos valores. Nada más cambia en los fixtures.

Run: `python -m pytest -m "not integration" -q`
Expected: todo PASS (ningún test depende del contenido de `bmaps` ni del nombre de los bloques del
fixture; `promedio_context` ya caía a todos los partidos cuando faltaba `main`).

- [ ] **Step 1: Tests que fallan en `tests/test_pages.py`**

```python
def test_escanos_is_rendered_with_tables_links_and_initial(api):
    status, headers, body = call(api, '/escanos', query='scope=es&mode=forecast')
    page = html(body)
    assert status == 200 and headers['content-type'] == 'text/html; charset=utf-8'
    assert '<title>Pronóstico electoral · Escaños</title>' in page
    assert 'href="/escanos?scope=es&amp;mode=forecast" aria-current="page"' in page.replace('\n', ' ')
    assert 'id="parties-table"' in page and '<th scope="col">P(mayoría)</th>' in page
    assert 'id="blocks-table"' in page and 'id="vs-table"' in page and 'id="blocks-chart"' in page
    assert 'name="coalition" value="PP" checked' in page and 'id="coalition-result"' in page
    assert 'id="fan-chart"' in page and 'id="seats-evolution"' in page
    assert 'href="/api/v1/forecast/es/forecast/summary?run={}&amp;format=csv"'.format(RUN) in page
    assert 'href="/api/v1/forecast/es/fan?run={}&amp;format=csv"'.format(RUN) in page
    assert page.count('format=csv') == 5 and 'id="districts"' not in page  # sin circunscripciones hasta la tarea 4
    assert '/dist/js/pages/escanos.js' in page
    data = initial(body)
    assert list(data) == ['state', 'meta', 'summary', 'dist', 'fan', 'runs']
    assert data['state'] == {'scope': 'es', 'mode': 'forecast', 'run': RUN, 'pinned': False}
    assert data['dist']['parties'] == ['PP'] and data['summary']['majority'] == 176


def test_escanos_nowcast_and_pinned_run(api):
    page = html(call(api, '/escanos', query='mode=nowcast&run=' + RUN)[2])
    assert 'Estimación a' in page and 'type="hidden" name="run" value="{}"'.format(RUN) in page
    assert page.count('run={}&amp;format=csv'.format(RUN)) == 5


def test_seat_rows_sort_by_seats_desc_with_nulls_last():
    rows = [{'name': 'A', 'seats': 5}, {'name': 'B', 'seats': None}, {'name': 'C', 'seats': 9}, {'name': 'D', 'seats': 5}]
    assert [r['name'] for r in pages.seat_rows(rows)] == ['C', 'A', 'D', 'B']


def test_default_coalition_is_the_first_vs_block_present_in_dist():
    meta = {'bmaps': {'vs': {'Derecha': ['PP', 'VOX', 'SALF'], 'Izquierda': ['PSOE']}}}
    summary = {'vs': [{'name': 'Derecha'}, {'name': 'Izquierda'}], 'blocks': []}
    assert pages.default_coalition(meta, summary, ['PSOE', 'VOX', 'PP']) == ['VOX', 'PP']
    assert pages.default_coalition({'bmaps': {}}, {'vs': [], 'blocks': []}, ['PP']) == []


def test_csv_links_cover_the_five_parts_in_order():
    links = pages.csv_links('es', 'nowcast', RUN)
    assert [l['label'] for l in links] == ['Resumen por partido y bloque', 'Escaños por simulación', 'Circunscripciones', 'Escenario central', 'Abanico por horizonte']
    assert links[0]['href'] == '/api/v1/forecast/es/nowcast/summary?run={}&format=csv'.format(RUN)
    assert links[4]['href'] == '/api/v1/forecast/es/fan?run={}&format=csv'.format(RUN)
```

Además: en `test_page_errors_are_html_and_not_cached` añadir `('/escanos', 'scope=es-md', 404, 'no
tiene pronóstico publicado')`; en `test_templates_compile_with_strict_undefined` añadir
`'escanos.html'`; en `test_bundle_strings_are_escaped_and_the_json_island_stays_safe` recorrer también
`'/escanos'`; en `tests/test_web_routes.py` nada (el enlace `/escanos` resuelve por `pages.ROUTES`).

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "escanos or seat_rows or default_coalition or csv_links"`
Expected: FAIL (404 en `/escanos`, atributos inexistentes en `pages`).

- [ ] **Step 3: Implementar `pages.py`, `Escanos.py`, `escanos.html` y el CSS según Interfaces**

`seat_rows` ordena con `key=lambda r: -1 if r.get('seats') is None else r['seats']` y `reverse=True`
(estable); `default_coalition` toma `members = (meta.get('bmaps') or {}).get('vs' | 'blocks', {}).get(first)`.

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q`
Expected: PASS (`test_web_routes` fallará por `/dist/js/pages/escanos.js` inexistente hasta la tarea 5:
crear en esta tarea un módulo mínimo `web/dist/js/pages/escanos.js` con `wireControls()` y
`mountWhenReady(() => {})`, que la tarea 5 completa).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages.py mtpy/controllers/Escanos.py web/templates/escanos.html web/dist/js/pages/escanos.js web/dist/css/site.css tests/test_pages.py tests/fixtures/bundle/meta.json tests/fixtures/bundle/summary.json
git commit -m "Add the seats page: party and block tables, coalition form and CSV links"
```

---

### Task 4: Circunscripciones: tabla general, detalle `?region=`, validador y ámbito autonómico

**Files:**
- Modify: `mtpy/lib/webapi.py` (`check_region`), `mtpy/lib/pages.py` (`MESSAGES`, `resolve_region`,
  `district_table`, `region_rows`, `escanos_context`), `web/templates/escanos.html` (`section#districts`)
- Test: `tests/test_webapi_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Consumes: `escanos_context` de la tarea 3; `fmt_range`.
- Produces:
  - `webapi.check_region(region) -> Optional[int]`: `None` o `''` → `None`; `re.fullmatch(r'\d{1,6}',
    region)` → `int`; otro → `HttpError(400, 'invalid region')`.
  - `pages.MESSAGES` añade `'invalid region': 'Circunscripción no válida.'` y `'region not found':
    'Circunscripción no encontrada.'`.
  - `resolve_region(region: Optional[int], regions: list) -> Optional[dict]`: `regions` vacío → `None`
    (con `region` dado → 404 `region not found`); `region` `None` → la de más `seats` (primera en
    empate); `region` fuera de los `id` → 404 `region not found`.
  - `district_table(districts: dict, scenario: dict) -> Optional[dict]`: `None` si
    `districts['regions']` está vacío; si no, `{'parties': [nombres de districts['parties'] con algún
    escaño > 0 en scenario o seats_hi > 0 en districts, en ese orden], 'rows': [{'id', 'name', 'seats',
    'cells': [{'seats': int | None, 'range': str}]}] en el orden de districts['regions'], 'total':
    {'seats': int, 'cells': [...]} | None}`; `cells[i].seats` = escaños del escenario para (región,
    partido i) o `None` si la región no está en `scenario.rows`; `range` = `fmt_range(seats_lo,
    seats_hi, 0)` de la fila de `districts` o `DASH`; `total` sale de la fila `region_id == 0` de
    `scenario` (`seats` = suma de sus escaños).
  - `region_rows(districts: dict, region_id: int) -> list`: filas de esa región ordenadas por `seats`
    desc y después `pct` desc (estable), sin formatear.
  - `escanos_context`: `region_id = check_region(region)`; lee `districts` y `scenario`; `current =
    resolve_region(region_id, districts['regions'])`; contexto `district_table`, `regions`
    (`districts['regions']`), `region` (`current`), `region_rows` (`[]` sin `current`).
  - Plantilla, `section#districts` (solo si `district_table`): `form#district-form` (`method="get"`,
    campos ocultos `scope`, `mode` y, si `state.pinned`, `run`; `<select name="region" id="region">`
    con una opción por `regions` y la actual `selected`; botón "Ver" `class="nojs-only"`);
    `table#region-table` (Partido · Voto · Intervalo 95 % · Escaños · Intervalo 95 % · P(escaño), con
    `caption` "{{ region.name }} · {{ region.seats }} escaños"); `table#district-table` (primera columna
    Circunscripción con `<a href="/escanos{{ query }}&amp;region={{ row.id }}">`, segunda Escaños,
    una columna por partido con `cells[i].seats|int` y `title="{{ cells[i].range }}"`, fila final
    "Total" si `district_table.total`), `caption` "Escenario central: la simulación más cercana a la
    mediana de escaños; cada fila suma los escaños de su circunscripción. Pasa el ratón para ver el
    intervalo del 95 %."

- [ ] **Step 1: Tests que fallan**

En `tests/test_webapi_unit.py`:

```python
def test_check_region_accepts_integers_only():
    assert webapi.check_region(None) is None and webapi.check_region('') is None
    assert webapi.check_region('28') == 28
    for bad in ('abc', '-1', '28.0', '1' * 7, '28;'):
        with pytest.raises(HttpError) as err:
            webapi.check_region(bad)
        assert err.value.status == 400 and err.value.message == 'invalid region'
```

En `tests/test_pages.py`, un ayudante que escribe partes de circunscripciones explícitas (dos partidos,
dos regiones, escenario con total) sobre el paquete de fixtures:

```python
DISTRICTS = {
    'parties': ['PP', 'PSOE'],
    'regions': [{'id': 8, 'name': 'Barcelona', 'seats': 32}, {'id': 28, 'name': 'Madrid', 'seats': 37}],
    'rows': [
        {'region_id': 8, 'region': 'Barcelona', 'name': 'PP', 'pct': 20.0, 'pct_lo': 17.0, 'pct_hi': 23.0, 'seats': 7.0, 'seats_mean': 7.1, 'seats_lo': 6.0, 'seats_hi': 8.0, 'p_seats': 1.0},
        {'region_id': 8, 'region': 'Barcelona', 'name': 'PSOE', 'pct': 25.0, 'pct_lo': 22.0, 'pct_hi': 28.0, 'seats': 9.0, 'seats_mean': 9.2, 'seats_lo': 8.0, 'seats_hi': 10.0, 'p_seats': 1.0},
        {'region_id': 28, 'region': 'Madrid', 'name': 'PP', 'pct': 39.0, 'pct_lo': 34.0, 'pct_hi': 44.0, 'seats': 16.0, 'seats_mean': 15.8, 'seats_lo': 14.0, 'seats_hi': 18.0, 'p_seats': 1.0},
        {'region_id': 28, 'region': 'Madrid', 'name': 'PSOE', 'pct': 24.0, 'pct_lo': 20.0, 'pct_hi': 28.0, 'seats': 9.0, 'seats_mean': 9.1, 'seats_lo': 8.0, 'seats_hi': 11.0, 'p_seats': 1.0},
    ],
}
SCENARIO = {'simulation': 3, 'parties': ['PP', 'PSOE'], 'rows': [
    {'region_id': 0, 'region': 'es', 'seats': [140, 106]},
    {'region_id': 8, 'region': 'Barcelona', 'seats': [7, 9]},
    {'region_id': 28, 'region': 'Madrid', 'seats': [16, 9]},
]}


def write_districts(fs, districts=DISTRICTS, scenario=SCENARIO, scope='es', run_id=RUN):
    writer = bundle.BundleWriter(fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part(scope, run_id, 'districts', mode), 'districts', districts, scope, run_id=run_id, mode=mode)
        writer.write_json(bundle.path_part(scope, run_id, 'scenario', mode), 'scenario', scenario, scope, run_id=run_id, mode=mode)


def test_escanos_districts_default_to_the_largest_and_link_the_rows(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/escanos', query='scope=es&mode=forecast')[2])
    assert 'id="districts"' in page and '<option value="28" selected>Madrid</option>' in page
    assert 'Madrid · 37 escaños' in page  # caption del detalle
    assert 'href="/escanos?scope=es&amp;mode=forecast&amp;region=8"' in page
    assert '<td title="6–8">7</td>' in page and '<td title="14–18">16</td>' in page
    assert '<td>Total</td>' in page and '140' in page
    page = html(call(api, '/escanos', query='scope=es&mode=forecast&region=8')[2])
    assert '<option value="8" selected>Barcelona</option>' in page and 'Barcelona · 32 escaños' in page


def test_escanos_district_form_keeps_the_pinned_run(api, fresh_app):
    write_districts(fresh_app.fs)
    page = html(call(api, '/escanos', query='scope=es&run=' + RUN)[2])
    assert page.count('type="hidden" name="run" value="{}"'.format(RUN)) == 2  # cabecera y formulario de circunscripción


@pytest.mark.parametrize('query, status, text', [
    ('region=abc', 400, 'Circunscripción no válida'),
    ('region=99', 404, 'Circunscripción no encontrada'),
])
def test_escanos_region_errors(api, fresh_app, query, status, text):
    write_districts(fresh_app.fs)
    got, headers, body = call(api, '/escanos', query=query)
    assert got == status and headers['cache-control'] == 'no-store' and text in html(body)


def test_escanos_without_districts_hides_the_section(api, fresh_app):
    write_districts(fresh_app.fs, districts={'parties': ['PP'], 'regions': [], 'rows': []},
                    scenario={'simulation': 0, 'parties': ['PP'], 'rows': [{'region_id': 0, 'region': 'es-md', 'seats': [70]}]})
    status, headers, body = call(api, '/escanos')
    assert status == 200 and 'id="districts"' not in html(body)
    assert call(api, '/escanos', query='region=28')[0] == 404


def test_regional_scope_renders_every_page(api, fresh_app):
    """Ámbito autonómico sintético: `bmaps` sin `max`, sin circunscripciones."""
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    meta = fixture_data('meta') | {'scope': 'es-md', 'n_seats': 135, 'majority': 68}
    meta['bmaps'] = {k: v for k, v in meta['bmaps'].items() if k != 'max'}
    bundle.BundleWriter(fresh_app.fs).write_json(bundle.path_part('es-md', '20261009-120000', 'meta'), 'meta', meta, 'es-md', run_id='20261009-120000')
    write_districts(fresh_app.fs, districts={'parties': ['PP'], 'regions': [], 'rows': []},
                    scenario={'simulation': 0, 'parties': ['PP'], 'rows': [{'region_id': 0, 'region': 'es-md', 'seats': [70]}]},
                    scope='es-md', run_id='20261009-120000')
    webapi.site().manifest_cache_clear()
    for path in ('/', '/promedio', '/escanos'):
        status, headers, body = call(api, path, query='scope=es-md')
        assert status == 200 and '<option value="es-md" selected>Madrid</option>' in html(body), path
    assert 'name="coalition" value="PP" checked' in html(call(api, '/escanos', query='scope=es-md')[2])


def test_district_table_and_region_rows():
    table = pages.district_table(DISTRICTS, SCENARIO)
    assert table['parties'] == ['PP', 'PSOE'] and [r['name'] for r in table['rows']] == ['Barcelona', 'Madrid']
    assert table['rows'][1]['cells'] == [{'seats': 16, 'range': '14–18'}, {'seats': 9, 'range': '8–11'}]
    assert table['total'] == {'seats': 246, 'cells': [{'seats': 140, 'range': '–'}, {'seats': 106, 'range': '–'}]}
    assert pages.district_table({'parties': [], 'regions': [], 'rows': []}, SCENARIO) is None
    assert [r['name'] for r in pages.region_rows(DISTRICTS, 28)] == ['PP', 'PSOE']
    assert pages.resolve_region(None, DISTRICTS['regions'])['id'] == 28
```

(Actualizar la aserción `'id="districts"' not in page` del test de la tarea 3: sin `write_districts`,
el fixture `districts` de `tests/fixtures/bundle/` tiene una región, así que la sección sí aparece; la
aserción pasa a `'id="districts"' in page`.)

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_webapi_unit.py tests/test_pages.py -q -k "region or district or regional_scope"`
Expected: FAIL (`check_region` inexistente, sección ausente, 200 en vez de 400/404).

- [ ] **Step 3: Implementar según Interfaces**

`district_table` indexa `scenario['rows']` por `region_id` y `districts['rows']` por `(region_id,
name)`; la posición del partido en `scenario['parties']` da el escaño de la celda.

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_webapi_unit.py tests/test_pages.py tests/test_web_routes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/webapi.py mtpy/lib/pages.py web/templates/escanos.html tests/test_webapi_unit.py tests/test_pages.py
git commit -m "Add the districts table with the central scenario and the per-district detail"
```

---

### Task 5: JavaScript: histogramas, barra apilada, abanico, calculadora y evolución de escaños

**Files:**
- Create: `web/dist/js/seats.js`, `web/dist/js/charts/histogram.js`, `web/dist/js/charts/stacked.js`,
  `web/dist/js/charts/fan.js`, `tests/test_js_seats.py`
- Modify: `web/dist/js/pages/escanos.js` (completo), `web/dist/js/charts/evolution.js`,
  `web/dist/js/pages/common.js`, `web/dist/js/catalog.js`, `web/dist/css/site.css` (si hace falta)

**Interfaces:**
- Consumes: `readInitial`, `wireControls`, `mountWhenReady`, `mapNames`, `showError` (`common.js`);
  `Catalog`, `BLOCK_ORDER`, `OTHERS_COLOR` (`catalog.js`); `bandSeries` (`charts/series.js`);
  `mountChart`, `baseOption`, `escapeHtml`, `THEME` (`charts/base.js`); `fmtInt`, `fmtPct`,
  `fmtProb`, `fmtRange`, `fmtDateShort`, `fmtDateTime` (`format.js`); los identificadores de la
  plantilla (tareas 3 y 4); `initial = {state, meta, summary, dist, fan, runs}`.
- Produces (`seats.js`, funciones puras, sin DOM):
  - `column(dist, name) -> number[]` (`[]` si el partido no está en `dist.parties`).
  - `coalitionSeats(dist, names) -> number[]`: suma por simulación de las columnas presentes; `[]`
    si ninguna lo está.
  - `quantile(sorted, q) -> number`: `tgt = q · n`; `i = clamp(ceil(tgt) − 1, 0, n − 1)`; si `tgt` es
    entero (`|tgt − (i + 1)| < 1e-9`) e `i < n − 1`, `(sorted[i] + sorted[i + 1]) / 2`; si no,
    `sorted[i]` (regla de `Stat.quantile` con pesos unitarios).
  - `summarize(values, majority, alpha = 0.05) -> {n, median, lo, hi, min, max, pMajority} | null`
    (`null` con `values` vacío; `median = quantile(0.5)`, `lo = quantile(alpha/2)`, `hi = quantile(1 −
    alpha/2)`, `pMajority` = proporción de `values ≥ majority`).
  - `binWidth(values, maxBins = 40) -> number`: `max(1, ceil((max − min + 1) / maxBins))`.
  - `histogram(values, width) -> {starts: number[], counts: number[]}`: bins `[start, start + width)`
    desde `floor(min / width) · width` hasta cubrir `max`, sin huecos (bins vacíos con 0).
  - `parseCoalition(hash, parties) -> string[] | null`: de `'#coalition=PP,VOX'` devuelve los nombres
    presentes en `parties` (orden de `parties`), `null` si el fragmento no lleva `coalition=`; nombres
    desconocidos se descartan (decodificados con `decodeURIComponent`).
  - `coalitionHash(names) -> string`: `'#coalition=' + names.map(encodeURIComponent).join(',')`
    (`''` sin nombres).
- Produces (gráficos):
  - `renderHistogram(el, values, {color, majority = null, median = null, lo = null, hi = null,
    width = null, formatter = fmtInt})`: barras de la proporción de simulaciones por bin (eje y en %,
    `fmtPct(x, 0)`), eje x categórico con el inicio de cada bin (`width` de `binWidth` si es `null`),
    `markLine` continua en `median` y discontinua en `majority` cuando `majority ≤ último inicio +
    width`, tooltip "`a–b` escaños: p %" (o "`a` escaños" con `width == 1`); `lo`/`hi` en el subtítulo
    del tooltip del cabezal. Devuelve la instancia.
  - `renderStacked(el, rows, {colors, fullnames, majority, total})`: `rows = [{name, seats}]` en
    orden; una sola categoría horizontal, una serie `bar` apilada por fila (`stack: 'seats'`), etiqueta
    "`nombre` `escaños`" dentro del segmento si `seats / total ≥ 0.05`, eje x `[0, total]`, `markLine`
    en `majority`, tooltip con `fullname` y escaños.
  - `renderFan(el, fan, {parties, selected = null, colors, fullnames, markAt, markLabel})`: eje x de
    valor (días, `'{value} d'`), por partido línea `mean` y banda `lo–hi` (`bandSeries` con los horizontes
    como x), leyenda `scroll` con `selected`, `markLine` vertical en `markAt` con `markLabel`, tooltip por
    horizonte "`partido`: `mean %` (`lo–hi`)" para los visibles, ordenados por `mean` desc.
  - `evolution.js`: `evolutionSeries(runs, mode, parties, field = 'pct')` y `renderEvolution(el, runs,
    mode, {colors, parties, fullnames, field = 'pct', formatter = fmtPct})`; el eje y y el tooltip usan
    `formatter`; la portada no cambia (valores por defecto).
  - `catalog.js`: `Catalog.orderBlocks(names)` ordena nombres de bloque por `BLOCK_ORDER` (desconocidos
    al final, en el orden de entrada).
  - `common.js`: `wireForm(id)`: envía el formulario `#id` al cambiar cualquiera de sus controles
    (`requestSubmit`/`submit`); `wireControls()` se mantiene (tiene la lógica del `run` fijado).
- `pages/escanos.js` (`paint()`): `const {state, meta, summary, dist, fan, runs} = readInitial()`;
  `Catalog(meta)`; por cada `div[data-party]` de `#histograms`, `values = column(dist, name)` (si está
  vacío, deja el div sin dibujar y sigue), `summarize(values, majority)` y `renderHistogram` con el
  color del partido; `renderStacked(#blocks-chart, rows)` con `rows` = `summary.blocks` ordenadas por
  `orderBlocks` (`seats` `null` → 0) y colores `catalog.blockColor`; `renderFan(#fan-chart, fan, …)` con
  `selected` = `bmaps.main` presentes, `markAt = state.mode === 'nowcast' ? 0 : meta.horizon_max`,
  `markLabel = 'Hoy' | 'Elección'`; `renderEvolution(#seats-evolution, runs.runs, state.mode, {field:
  'seats', formatter: fmtInt, parties: summary.parties.map(name)})`; calculadora: `parseCoalition(location.hash,
  dist.parties)` → si no es `null`, marca las casillas; `update()` lee las casillas marcadas,
  `coalitionSeats`, `summarize`, escribe en `#coalition-result` (`textContent`) "`N` partidos ·
  mediana `median` escaños · intervalo 95 % `lo–hi` · mayoría absoluta (`majority`): `pMajority`" (o
  "Marca partidos para calcular su mayoría." sin selección, y vacía el gráfico con `clear()`),
  `renderHistogram(#coalition-chart, values, {majority, median, lo, hi, color: THEME.markColor})` y
  `history.replaceState(null, '', location.pathname + location.search + coalitionHash(names))`;
  `#coalition-parties` escucha `change` → `update()`; `wireControls()`, `wireForm('district-form')`,
  `mountWhenReady(paint)`.

- [ ] **Step 1: Test de paridad que falla (`tests/test_js_seats.py`)**

```python
"""Paridad entre `web/dist/js/seats.js` y la regla de cuantiles de `Stat` (se salta sin `node`)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from mtpy.core.utils.stat import Stat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEATS_JS = os.path.join(ROOT, 'web', 'dist', 'js', 'seats.js')
DIST = {'n_seats': 350, 'parties': ['PP', 'PSOE', 'VOX'],
        'seats': [[140, 106, 65], [137, 110, 62], [145, 100, 70], [133, 112, 61], [150, 95, 72], [139, 108, 60], [141, 104, 66], [136, 109, 63]]}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as s from "file://{}"; const dist = {}; console.log(JSON.stringify({}));'.format(SEATS_JS, json.dumps(DIST), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_summarize_matches_stat_quantiles():
    values = np.array([row[0] + row[2] for row in DIST['seats']], dtype=float)
    stat = Stat(values, dropna=True)
    got = run_js('s.summarize(s.coalitionSeats(dist, ["PP", "VOX"]), 176)')
    assert got == {'n': 8, 'median': stat.median(), 'lo': stat.quantile(0.025), 'hi': stat.quantile(0.975),
                   'min': 193.0, 'max': 222.0, 'pMajority': 1.0}


def test_column_and_coalition_ignore_unknown_parties():
    assert run_js('s.column(dist, "PSOE")') == [row[1] for row in DIST['seats']]
    assert run_js('s.column(dist, "ERC")') == []
    assert run_js('s.coalitionSeats(dist, ["ERC"])') == []
    assert run_js('s.coalitionSeats(dist, ["PP", "ERC"])') == [row[0] for row in DIST['seats']]
    assert run_js('s.summarize([], 176)') is None


def test_histogram_bins_cover_the_range_without_gaps():
    assert run_js('s.binWidth([0, 0, 1], 40)') == 1 and run_js('s.binWidth([100, 180], 40)') == 3
    assert run_js('s.histogram([3, 4, 4, 9], 2)') == {'starts': [2, 4, 6, 8], 'counts': [1, 2, 0, 1]}
    assert run_js('s.histogram([0, 0, 0], 1)') == {'starts': [0], 'counts': [3]}


def test_coalition_hash_round_trip():
    assert run_js('s.parseCoalition("#coalition=VOX,PP,ERC", dist.parties)') == ['PP', 'VOX']
    assert run_js('s.parseCoalition("#other=1", dist.parties)') is None
    assert run_js('s.parseCoalition("#coalition=", dist.parties)') == []
    assert run_js('s.coalitionHash(["PP", "VOX"])') == '#coalition=PP,VOX' and run_js('s.coalitionHash([])') == ''
```

- [ ] **Step 2: Comprobar que falla**

Run: `python -m pytest tests/test_js_seats.py -q`
Expected: FAIL (`seats.js` no existe; `node` termina con error).

- [ ] **Step 3: Implementar `seats.js` y comprobar la paridad**

Run: `python -m pytest tests/test_js_seats.py -q`
Expected: PASS.

- [ ] **Step 4: Implementar `charts/histogram.js`, `charts/stacked.js`, `charts/fan.js`, los cambios de `evolution.js`, `catalog.js` y `common.js`, y `pages/escanos.js` según Interfaces**

- [ ] **Step 5: Sintaxis, tests estáticos y renderizado SSR**

Run: `for f in web/dist/js/*.js web/dist/js/charts/*.js web/dist/js/pages/*.js; do node --check "$f" || exit 1; done && python -m pytest tests/test_web_routes.py tests/test_pages.py -q`
Expected: sin errores de sintaxis; PASS (imports que resuelven, módulo en la plantilla, sin `/api/v1`).

Render SSR de los cuatro gráficos nuevos con ECharts en Node sobre el run local (patrón de la fase 2:
`echarts.init(null, null, {renderer: 'svg', ssr: true, width: 800, height: 360})` con un `window`
mínimo y `renderToSVGString()`), uno por gráfico: `renderHistogram` con `column(dist, 'PP')`,
`renderStacked` con `summary.blocks`, `renderFan` con `fan.json`, `renderEvolution` con `history.json`
y `field: 'seats'`; comprobar que cada SVG contiene `<path` o `<rect` y el texto de la `markLine`
("Mayoría" / "Elección"). Script en el scratchpad, no en el repositorio.

- [ ] **Step 6: Commit**

```bash
git add web/dist/js/seats.js web/dist/js/charts/histogram.js web/dist/js/charts/stacked.js web/dist/js/charts/fan.js web/dist/js/charts/evolution.js web/dist/js/catalog.js web/dist/js/pages/common.js web/dist/js/pages/escanos.js web/dist/css/site.css tests/test_js_seats.py
git commit -m "Draw the seats page: histograms, stacked blocks, fan, seat evolution and the coalition calculator"
```

---

### Task 6: Scripts, documentación, enmiendas de las specs y verificación local

**Files:**
- Modify: `deploy/check-api.sh`, `deploy/smoke.sh`, `deploy/README.md`, `docs/web/contrato.md`,
  `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`,
  `docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: scripts que comprueban `/escanos`; documentación al día; "Estado al cierre de la fase 3".

- [ ] **Step 1: Scripts**

`deploy/check-api.sh`: tras la línea de `/promedio`, `check "/escanos?scope=es&mode=forecast" 200 html`
y `check "/escanos?region=abc" 400 html`. `deploy/smoke.sh`: `check_static /escanos text/html` tras
la de `/promedio`.

Run: `bash -n deploy/check-api.sh deploy/smoke.sh`
Expected: sin salida.

- [ ] **Step 2: Documentación**

- `deploy/README.md`, "La web": las páginas son `Index` (`/`), `Promedio` (`/promedio`) y `Escanos`
  (`/escanos`, con `?region=`); lista de `check-api.sh` con las dos rutas nuevas; "Cómo añadir una
  página": un parámetro propio de la página (como `region`) se valida en `webapi` y se lee en el
  `index_action`, fuera de `parse_state`; "Pendiente de Luis": retirar la primera publicación a S3 (hecha
  el 2026-10-08, run `20261008-202843`) y la decisión sobre `jinja2` (3.1.6 desde `68522d2`).
- `docs/web/contrato.md`: el párrafo de páginas nombra `/escanos` y su parámetro `region` (fuera del
  contrato de la API); el fragmento `#coalition=` es solo del navegador.
- Spec general: nueva sección "Estado al cierre de la fase 3 (fecha)" con ficheros, tests, decisiones
  D1-D8 y lo pendiente de Luis (recorrido en navegador de `/escanos` en escritorio y móvil, publicación
  de `scopes: "all"` y revisión de cada ámbito en la web, decisión sobre la veda antes del 24-11,
  agrupación `group`); en "Estado al cierre de la fase 2" y en el "Estado al cierre" de la spec del
  rediseño, anotar que la primera publicación a S3 y el pin de `jinja2` ya están resueltos.

- [ ] **Step 3: Verificación local completa**

Run: `python -m pytest -m "not integration" -q`
Expected: todo PASS (≈ 323 + los nuevos de las tareas 1-5).

Run: `bash deploy/nginx-check.sh`
Expected: `syntax is ok`.

Con uvicorn y el nginx local de comprobación (como en el rediseño; nunca el 8080 de Luis):

```bash
S3_BUCKET= python -m uvicorn api:api --port 8000 &
bash deploy/check-api.sh http://127.0.0.1:8043
curl -s 'http://127.0.0.1:8043/escanos?scope=es&mode=forecast' | grep -c 'id="districts"'
curl -s 'http://127.0.0.1:8043/escanos?scope=es&region=28' | grep -o 'Madrid · 38 escaños'
curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8043/escanos?region=x'
```

Expected: `check-api.sh` todo OK; `1`; `Madrid · 38 escaños` (reparto del decreto: Madrid 38); `400`.
Comprobar también que el HTML de `/escanos` con el run local pesa menos de 250 KB sin comprimir
(`curl -s … | wc -c`) y que `initial-data` no contiene `"rows"` de `districts` (`grep -c region_id` = 0
dentro del bloque JSON).

- [ ] **Step 4: Commit**

```bash
git add deploy/check-api.sh deploy/smoke.sh deploy/README.md docs/web/contrato.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md docs/superpowers/specs/2026-10-09-web-enrutado-controladores-design.md
git commit -m "Check the seats page in the scripts and document the phase 3 closing state"
```

---

## Pendiente de Luis tras la fase 3 (no lo hace el plan)

- Recorrer `/escanos` en escritorio y móvil: histogramas pequeños legibles, barra apilada con etiquetas,
  calculadora (marcar y desmarcar, enlace con `#coalition=` compartido), formulario de circunscripción
  con y sin JS, abanico con la leyenda, evolución con un solo run.
- Publicar `scopes: "all"` contra la RDS y revisar cada ámbito en las tres páginas (nombres de bloques
  distintos de `BLOCK_ORDER` quedan al final del hemiciclo y de la barra apilada).
- Decidir la política de veda (qué se oculta con `manifest.freeze.active`) antes del 2026-11-24 (fase 6)
  y si se añade el selector de agrupación `group`.

## Esfuerzo estimado

Tareas 1-2: 0,5 d · Tarea 3: 1 d · Tarea 4: 0,75 d · Tarea 5: 1,25 d · Tarea 6: 0,5 d → 4 días
(la spec estimaba 3-4).
