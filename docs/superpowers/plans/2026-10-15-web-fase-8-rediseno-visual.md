# Web fase 8 (rediseño visual al estilo de Objetivo176 y Election Simulator) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar a toda la web el sistema visual de la maqueta aprobada (DM Sans + Outfit, tarjetas de radio 18, controles en tinta, tablas con barras y rangos, KPIs y tarjetas de partido), una portada nueva con hero, una cabecera compacta con barra inferior en móvil, y el hemiciclo con los números dentro.

**Architecture:** Las páginas siguen saliendo del servidor (Jinja2 + contexto de `mtpy/lib/pages`) con todas las cifras en el HTML; `site.css` se reescribe con los tokens y componentes de la maqueta; los módulos JS solo dibujan (ECharts con el tema nuevo, sparklines en SVG, leyenda de píldoras HTML que gobierna a ECharts). Las fuentes se vendorizan como woff2 y el wordmark se regenera con Outfit.

**Tech Stack:** Python 3.11 (Jinja2 3.1), CSS sin preprocesador, JS ES modules sin build, ECharts 5.6 vendido, pytest + node por subproceso, fonttools + Pillow para la marca (solo en desarrollo).

**Spec:** `docs/superpowers/specs/2026-10-15-web-fase-8-rediseno-visual-design.md` (y la maqueta `docs/superpowers/specs/2026-10-15-web-fase-8-mockups.html`, que es la referencia visual de cada componente: ábrela en el navegador y copia sus medidas).

## Global Constraints

- Textos en español; siglas de partido; "encuestadora/s" y "medio/s"; docstrings numpy/JSDoc en inglés; CSP `default-src 'self'` (sin scripts ni estilos inline, sin recursos externos: nada de Google Fonts en producción); controles solo-JS renderizados `hidden` y mostrados por el JS; DOM con `createElement`/`textContent`.
- Suite `python -m pytest -m "not integration" -q` en verde (758 al empezar), sin avisos nuevos; tests de paridad en node (`tests/test_js_charts.py`) para las opciones de ECharts que cambien.
- Nunca ejecutar `job.py`, `mtpy.run()` ni uvicorn sin `S3_BUCKET=` y `DB_HOST=` en el entorno.
- Commits en inglés sin prefijo `feat:`, con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `git add` solo de los ficheros tocados. Rama `dev`.
- Tokens, escala tipográfica y medidas: los de D1 y D3 de la spec, copiados tal cual (no "parecidos"); puntos de corte: móvil `max-width: 719px`, tableta `max-width: 1099px` (el `899px` actual de `.hemi-grid`/`.split` pasa a `1099px`), escritorio el resto; contenedor fluido (`main` sin `max-width`, `--band-pad` 32/16 px) como ya está.
- Los estados seleccionados (`.seg`, pestañas del evento, barra inferior activa en acento solo en la barra) van en tinta `--color-text`; `--color-accent` queda para marca, enlaces, barra de título de tarjeta y entrada activa del menú.
- Las decisiones 1-6 de "Decisiones del autor tras la revisión del constructor" de la spec obligan a todas las tareas (leyenda de píldoras solo en los dos gráficos de series de Promedios; paginación de "Todas las encuestas" intacta; hero = primer ámbito publicado; tarjetas de cierre intactas; capturas manuales; KPIs del perfil = los tres de la cabecera).
- Nada de la publicación, los jobs, la API, nginx ni Docker cambia.

## Review Focus

1. Un run publicado sin la parte `polls` (o sin `summary` del modo): la portada responde 200 con el hero sin barra de escaños ni "Últimas encuestas", y Promedios sin KPI de último sondeo (test en la tarea 6 y en la 5).
2. Un partido del resumen que no está en `meta.parties`: color `OTHERS_COLOR` en la barra, la tarjeta de partido y el hemiciclo, nombre completo = siglas, sin excepción de `StrictUndefined` (test en la tarea 5).
3. `seats_lo`/`seats_hi` ausentes en una fila: la celda de rango pinta "–" sin barra y sin error (test del filtro `rng` en la tarea 2).
4. La página de error (`error.html`) y la portada sin ningún ámbito publicado (503) se renderizan sin barra inferior y sin `StrictUndefined` (tests en las tareas 3 y 6).
5. Un `meta` sin `n_seats` ni `majority`: la barra de evento omite la cámara y el hero lee `summary.n_seats`/`summary.majority`; sin ninguno de los dos, no hay barra de escaños (test en la tarea 6).

---

### Task 1: fuentes vendidas, tokens y escala tipográfica

**Files:**
- Create: `web/brand/fonts.py` (descarga de desarrollo), `web/dist/vendor/fonts/DMSans-400.woff2`, `DMSans-500.woff2`, `DMSans-600.woff2`, `DMSans-700.woff2`, `Outfit-700.woff2`, `Outfit-800.woff2`
- Modify: `web/dist/vendor/fonts/LICENSE` (entradas de DM Sans y Outfit, OFL; fuera las de Source), `web/dist/css/site.css` (bloque `@font-face`, tokens, base), `tests/test_deploy_files.py`
- Delete: `web/dist/vendor/fonts/SourceSerif4-4.005-Semibold.woff2`, `SourceSans3-3.052-Regular.woff2`, `SourceSans3-3.052-Semibold.woff2` (al final de la tarea 8, cuando el wordmark ya no las necesite; hasta entonces se quedan sin referenciar)

**Interfaces:**
- Produces: `web/brand/fonts.py` con `FAMILIES = {'DM Sans': (400, 500, 600, 700), 'Outfit': (700, 800)}` y `fetch(out_dir: str) -> list[str]`: pide `https://fonts.googleapis.com/css2?family={familia}:wght@{pesos}&display=swap` con un `User-Agent` de Chrome reciente (así la API sirve woff2 por peso), toma la URL del bloque `/* latin */` de cada `@font-face`, la descarga a `{Familia sin espacios}-{peso}.woff2` y devuelve los nombres. Se ejecuta una vez (`python web/brand/fonts.py`) y los ficheros se commitean; no es dependencia de la web. Si un fichero pasa de 40 KB, el constructor lo subconjunta con `pyftsubset --unicodes="U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD" --flavor=woff2` (el rango "latin" de Google Fonts).
- Produces (`site.css`): seis `@font-face` (`font-family: 'DM Sans'` con `font-weight` 400/500/600/700, `'Outfit'` 700/800, `font-display: swap`, `src: url('/dist/vendor/fonts/…')`); tokens de D1 con esos nombres exactos (`--color-soft2`, `--color-accent-soft`, `--color-hover`, `--color-ok`, `--color-ok-soft`, `--color-bad`, `--color-bad-soft`, `--font-head`, `--font-body`, `--radius: 18px`, `--radius-sm: 10px`, `--shadow`, `--shadow-up`); `body` a 15 px / 1,5 DM Sans con `font-variant-numeric: tabular-nums`; `h1, h2, h3` en `--font-head`; `.lead` 13,5 px; `th` 10,5 px mayúsculas `.04em` en `--color-muted`; `td, th` 13 px (12,5 px en móvil). `--color-faint` deja de usarse en texto (se sustituye por `--color-muted` en las reglas que lo usaban).
- Produces (`base.html`): `<link rel="preload" href="/dist/vendor/fonts/DMSans-400.woff2" as="font" type="font/woff2" crossorigin>` y lo mismo para `Outfit-800.woff2`, antes del `<link rel="stylesheet">`.

- [ ] **Step 1: Test que falla** (en `tests/test_deploy_files.py`):

```python
def test_site_fonts_are_vendored_and_licensed():
    """Las fuentes que `site.css` declara existen en `web/dist/vendor/fonts`, pesan lo que una subconjunto latino,
    están en `LICENSE` y las antiguas ya no se referencian."""
    css = read('web/dist/css/site.css')
    faces = re.findall(r"@font-face\s*{[^}]*font-family:\s*'([^']+)'[^}]*font-weight:\s*(\d+)[^}]*url\('/dist/vendor/fonts/([\w.-]+)'\)", css, re.S)
    assert {(family, weight) for family, weight, _ in faces} == {
        ('DM Sans', '400'), ('DM Sans', '500'), ('DM Sans', '600'), ('DM Sans', '700'), ('Outfit', '700'), ('Outfit', '800')}
    licence = read('web/dist/vendor/fonts/LICENSE')
    for _, _, name in faces:
        path = ROOT / 'web' / 'dist' / 'vendor' / 'fonts' / name
        assert path.is_file() and path.stat().st_size < 60_000, name
        assert name in licence, name
    assert 'Source Serif' not in css and 'Source Sans' not in css
    assert 'Indian Type Foundry' in licence and 'Outfit Project Authors' in licence
    assert "--font-body: 'DM Sans'" in css and "--font-head: 'Outfit'" in css
    assert 'font-variant-numeric: tabular-nums' in css
```

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_deploy_files.py -q -k fonts_are_vendored` → FAIL (no hay `@font-face` de DM Sans).
- [ ] **Step 3: Escribir `web/brand/fonts.py`** (`fetch`, `if __name__ == '__main__': print(fetch(OUT))`, `OUT = …/web/dist/vendor/fonts`) y ejecutarlo: `python web/brand/fonts.py` → seis ficheros. Comprobar con `python -c "from fontTools.ttLib import TTFont; f=TTFont('web/dist/vendor/fonts/DMSans-400.woff2'); print(f['name'].getDebugName(1), len(f.getGlyphOrder()))"` que son DM Sans y Outfit con glifos latinos (ñ, acentos: `'ntilde' in f.getGlyphOrder()`).
- [ ] **Step 4: `LICENSE`**: cabeceras "DM Sans (DMSans-400.woff2 … DMSans-700.woff2): Copyright 2014-2017 Indian Type Foundry (info@indiantypefoundry.com), with Reserved Font Name 'DM Sans'." y "Outfit (Outfit-700.woff2, Outfit-800.woff2): Copyright 2021 The Outfit Project Authors (https://github.com/Outfitio/Outfit-Fonts), with Reserved Font Name 'Outfit'."; el texto de la OFL 1.1 que ya está se conserva; las entradas de Source se retiran.
- [ ] **Step 5: `site.css`**: sustituir el bloque de `@font-face` y los tokens; `body`, `h1-h3`, `.lead`, `th`/`td` con la escala de D1; `base.html` con los dos `preload`. Nada más de CSS en esta tarea (los componentes son la tarea 2).
- [ ] **Step 6: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS (el test de la marca sigue pasando: el wordmark todavía es el SVG antiguo).
- [ ] **Step 7: Commit**: `git commit -m "Vendor DM Sans and Outfit and set the type scale and tokens of the redesign"`.

---

### Task 2: componentes de `site.css` y el filtro de rango

**Files:**
- Modify: `web/dist/css/site.css` (una sección por componente, en el orden de D3), `mtpy/lib/pages/filters.py`, `mtpy/lib/pages/common.py` (`environment()` registra `rng`), `tests/test_pages.py` (tests unitarios de los filtros, junto a los de `bar_width`/`bar_sides`/`shade`)

**Interfaces:**
- Produces (`filters.py`): `rng_cell(lo, hi, median, total, majority=None) -> Optional[dict]`: `{'left': %, 'width': %, 'mark': %, 'line': % | None}` con un decimal, cada valor `100 * x / total` recortado a [0, 100] antes de restar (`left` = `lo`, `width` = `hi` recortado menos `left`, `mark` = la mediana, `line` = la mayoría o `None`); `None` cuando `lo`, `hi`, `median` o `total` faltan, `total <= 0` o `lo > hi`. Registrado en `environment()` como `rng` (al lado de `bar`, `sides`, `shade`) y exportado en `pages/__init__.py`.
- Produces (`_macros.html`): `card_start` pinta el subtítulo de modo como `<span class="pill">{{ subtitle|replace('· ', '', 1) }}</span>` dentro del `h2`/`h3` (en vez de `<span class="mode-subtitle">`); `.mode-subtitle` desaparece del CSS. Test: `test_card_subtitle_is_a_pill(api)`: en `/es`, `'<span class="pill">estimación a ' in page` y `'mode-subtitle' not in page`.
- Produces (`#freeze-banner`): `position: sticky; top: 0; z-index: 21` (encima de la cabecera, que pasa a `top` igual a la altura de la pancarta cuando existe: `body:has(#freeze-banner) .hdr { top: var(--banner-h, 44px); }`, con la pancarta de una sola línea a 44 px).
- Produces (`site.css`, clases exactas, que las plantillas de las tareas siguientes usan): `.card`, `.card-head`, `.card-title` (ya `.card-head h2/h3`: se conserva el selector actual y la barra `::before` de 4×18 px), `.pill` (`.ok`, `.brand`), `.chip` (`.ok`), `.seg` (`.lg`; activo en `--color-text`), `.btn` (`.pri`), `.kpis > .kpi` (`.l`, `.v`, `.s`, `.s.ok`), `.pcards > .pc` (`.rk`, `.nm`, `.fl`, `.big`, `.lb`, `.dl`, `.sl`, `svg.spark`), `.rows > .row` (`.n`, `.d`, `.maj`, `.chev`) y `.row-foot`, `.rows-2`, `.bar > i`, `.rng > i/b/u`, `.pn`, `.lead-strip`, `.legend > .lg` (`.on`, `--c`), `.card-link` (`.ic`, `.go`), `.steps > .step` (`.n`), `.past > a`, `.hero`, `.eyebrow`, `.hero-t` (`em`), `.hero-p`, `.hcard`, `.hcard-top`, `.hcard-id`, `.flag`, `.bignum`, `.seatbar` (`.trk`, `.mk`, `.ends`), `.chips`, `.cta`, `.sec-h`, `.side`, `.hemi-lg`, `.plist > .pr` (`.dt`, `.who`, `.vals`), `.evbar` (sustituye a `#event-bar`: título, `.seg.lg`, `.run`), `.tabbar`, `.hdr` (sustituye el estilo de `#site-header`, que conserva el `id`), `.ico-btn`. Medidas, colores y pesos: los de la maqueta (`mock-template` dentro de `2026-10-15-web-fase-8-mockups.html`, bloque "Sistema de diseño propuesto"), transcritos con `@media` en vez de `@container`.
- Las clases que las páginas actuales ya usan y se conservan con el estilo nuevo: `.seg`, `.pills`/`.pill` de selección múltiple (pasan al estilo `.lg`: borde 1,5 px del color, activo relleno), `.chip` del selector de encuestas (`.dd-panel .chip`), `.table-wrap`, `.et`, `.et-rank`, `.geo`, `.cell-bar`, `.cell-shade`, `.figures` (se mantiene hasta la tarea 7), `.how`, `.lead`, `.stack`, `.grid`, `.grid-three`, `.split`, `.hemi-grid`, `.controls`, `.bar` (la barra de controles de la tabla de encuestas se renombra a `.ctrl-bar` para liberar `.bar` para la celda: cambio en `_polls_table.html` y en `polls-wire.js` si la selecciona por clase; comprobar con `grep -rn '"bar"\|\.bar\b' web/dist/js web/templates`).

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`, junto a `test_bar_width…`):

```python
def test_rng_cell_places_the_range_the_median_and_the_majority():
    assert pages.rng_cell(118, 160, 140, 350, 176) == {'left': 33.7, 'width': 12.0, 'mark': 40.0, 'line': 50.3}
    assert pages.rng_cell(0, 4, 0, 350) == {'left': 0.0, 'width': 1.1, 'mark': 0.0, 'line': None}
    assert pages.rng_cell(300, 420, 350, 350, 176)['width'] == 14.3  # recortado a 100
    assert pages.rng_cell(None, 160, 140, 350) is None and pages.rng_cell(118, 160, None, 350) is None
    assert pages.rng_cell(118, 160, 140, 0) is None and pages.rng_cell(160, 118, 140, 350) is None
    assert pages.environment().filters['rng'] is pages.rng_cell
```

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py -q -k rng_cell` → FAIL.
- [ ] **Step 3: Implementar `rng_cell`** en `filters.py` (con `_fraction`/`_is_missing` ya existentes), importarlo en `common.py` y registrarlo; exportarlo en `pages/__init__.py`.
- [ ] **Step 4: `site.css`**: escribir las secciones de componentes de D3 (y las reglas de móvil/tableta de cada uno) transcribiendo la maqueta. Renombrar `.bar` (barra de controles) a `.ctrl-bar` en CSS, `_polls_table.html` y JS. No tocar todavía las plantillas (salvo ese renombrado).
- [ ] **Step 5: Comprobación visual rápida**: con el paquete local y sin base (`S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8000`, los `/dist/` servidos como en "Desarrollo sin Docker" del `deploy/README.md`), abrir `/es` y `/casas`: tipografía nueva, tarjetas de radio 18, segmentados en tinta, nada roto. Parar uvicorn.
- [ ] **Step 6: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 7: Commit**: `git commit -m "Add the components of the redesign to the stylesheet and the range cell filter"`.

---

### Task 3: cabecera compacta, barra inferior y barra de evento

**Files:**
- Create: `web/templates/_svg.html` (sprite de iconos), `web/templates/_tabbar.html`
- Modify: `mtpy/lib/pages/common.py` (`TABBAR`, `tabbar()`, `chamber_text()`, `event_info`, `common_context`), `mtpy/lib/pages/__init__.py` (exportar `tabbar`, `chamber_text`, `TABBAR`), `web/templates/base.html` (sprite al principio del `body`, `_tabbar.html` al final), `web/templates/_header.html`, `web/templates/_brand.html` (sin `brand-tagline`), `web/templates/_event_bar.html`, `web/templates/error.html` (sin barra), `web/dist/css/site.css` (`.hdr`, `.tabbar`, `.evbar` ya escritos en la tarea 2: solo ajustes), `tests/test_pages.py`

**Interfaces:**
- Produces (`common.py`):
  - `TABBAR = ({'key': 'inicio', 'label': 'Inicio', 'icon': 'home'}, {'key': 'promedios', 'label': 'Promedios', 'icon': 'line'}, {'key': 'escanos', 'label': 'Escaños', 'icon': 'hemi'}, {'key': 'historico', 'label': 'Histórico', 'icon': 'target'}, {'key': 'ranking', 'label': 'Ranking', 'icon': 'trophy'})`.
  - `tabbar(scope: str, active: Optional[str], active_suffix: str, event: bool, query: str) -> list`: cinco dicts `{'key', 'label', 'icon', 'href', 'active'}`; `href`: `/`, `/{scope}{query}`, `/{scope}/escanos{query}` (`query` = `event_query(state)` en un evento, `''` fuera), `/elecciones{scope_query(scope)}`, `/casas`; `active`: en un evento, `promedios` si `active_suffix == ''` y `escanos` si `'/escanos'`; fuera, `inicio` con `active == '/'`, `historico` con `'/elecciones'`, `ranking` con `'/casas'`; ninguna con `/metodo`.
  - `chamber_text(scope: str, meta: dict) -> Optional[str]`: `'Congreso, {n} escaños'` para `es`, `'Parlamento, {n} escaños'` para el resto, con `n = meta.get('n_seats')` (`fmt_int`); `None` sin `n_seats`.
  - `event_info` añade `chamber` (`chamber_text`) y `run_at` (`meta.get('run_at')`), `n_polls` (`meta.get('n_polls')`); `common_context` añade `tabbar` (`tabbar(scope, active, active_suffix, event, query)`), `method_icon_href = '/metodo'`.
- Produces (plantillas): `_svg.html`: `<svg hidden aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><symbol id="i-home" viewBox="0 0 24 24">…</symbol>…</svg>` con `i-home`, `i-line`, `i-hemi`, `i-target`, `i-trophy`, `i-book`, `i-help`, `i-chevron`, trazo 2 px `currentColor`, `fill="none"`, `stroke-linecap="round"` (los `path` de la maqueta). `_tabbar.html`: `<nav class="tabbar" aria-label="Secciones">` con un `<a href class="on"?>` por entrada, cada uno `<svg><use href="#i-{icon}"/></svg>{label}`; se incluye solo `{% if tabbar is defined %}`. `_header.html`: `<header id="site-header" class="hdr">`, la marca, `<nav aria-label="Secciones">` (igual que hoy) y `<a class="ico-btn" href="/metodo" aria-label="Método"><svg><use href="#i-help"/></svg></a>`. `_event_bar.html`: `<div id="event-bar" class="evbar">` con `<h2>{{ event_bar.title }}<small>{{ event_bar.date|date }}{% if event_bar.days_text %} · {{ event_bar.days_text|lower }}{% endif %}{% if event_bar.chamber %} · {{ event_bar.chamber }}{% endif %}</small></h2>`, `<nav class="seg lg event-tabs">` con `<a><svg><use href="#i-line"/></svg>Promedios</a>` y `#i-hemi` Escaños (`aria-current="page"` → clase `on`), y `<div class="run"><span class="pill brand">{{ title_suffix }}</span>{% if state.pinned %}<span class="pill event-run">Run del {{ meta.run_at|date }}</span>{% endif %}<small>Publicación del {{ event_bar.run_at|date }} · {{ event_bar.n_polls|int }} sondeos</small></div>`. `base.html`: `{% include '_svg.html' %}` justo tras `<body>`, `{% include '_tabbar.html' %}` tras el pie; `error.html` no cambia de contexto, así que `_tabbar.html` no sale (el `is defined`).

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`):

```python
def test_tabbar_links_the_five_sections_and_marks_the_active_one():
    bar = pages.tabbar('es', None, '/escanos', True, '?mode=forecast')
    assert [row['href'] for row in bar] == ['/', '/es?mode=forecast', '/es/escanos?mode=forecast', '/elecciones?scope=es', '/casas']
    assert [row['active'] for row in bar] == [False, False, True, False, False]
    assert [row['icon'] for row in bar] == ['home', 'line', 'hemi', 'target', 'trophy']
    bar = pages.tabbar('es-md', '/casas', '', False, '')
    assert bar[1]['href'] == '/es-md' and bar[3]['href'] == '/elecciones?scope=es-md' and bar[4]['active']
    assert not any(row['active'] for row in pages.tabbar('es', '/metodo', '', False, ''))
    assert pages.chamber_text('es', {'n_seats': 350}) == 'Congreso, 350 escaños'
    assert pages.chamber_text('es-md', {'n_seats': 135}) == 'Parlamento, 135 escaños'
    assert pages.chamber_text('es', {}) is None


def test_header_is_compact_and_the_pages_carry_the_tabbar_and_the_icons(api):
    page = html(call(api, '/es')[2])
    assert 'brand-tagline' not in page and '<meta name="description"' in page
    assert '<svg hidden aria-hidden="true"' in page and page.index('<symbol id="i-home"') < page.index('<main')
    assert 'class="ico-btn" href="/metodo" aria-label="Método"' in page
    bar = re.search(r'<nav class="tabbar" aria-label="Secciones">(.*?)</nav>', page, re.S).group(1)
    assert bar.count('<a ') == 5 and 'href="/es/escanos" ' in bar and '<use href="#i-trophy"' in bar
    assert re.search(r'<a class="on" href="/es"[^>]*><svg[^>]*><use href="#i-line"', bar)
    assert page.index('</footer>') < page.index('<nav class="tabbar"')


def test_event_bar_names_the_chamber_and_the_publication(api):
    page = html(call(api, '/es')[2])
    assert '<h2>Elecciones generales<small>29 nov 2026 · ' in page and 'Congreso, 350 escaños</small></h2>' in page
    assert '<span class="pill brand">Estimación a ' in page
    assert 'Publicación del 8 oct 2026 · 120 sondeos' in page
    assert 'class="seg lg event-tabs"' in page and re.search(r'<a class="on" href="/es"[^>]*><svg', page)


def test_error_page_has_no_tabbar(api):
    status, _, body = call(api, '/nope')
    assert status == 404 and 'class="tabbar"' not in html(body) and '<svg hidden aria-hidden="true"' not in html(body)
```

(Los tests existentes que cambian: `test_nowcast_and_pinned_run_are_reflected` (`<span class="pill event-run">Run del 8 oct 2026</span>`), los que buscan `aria-current="page"` en las pestañas del evento (ahora `class="on"` y `aria-current` a la vez: conservar `aria-current="page"` en el `<a>` activo para no tocarlos), `test_brand_assets_match_the_templates_and_nginx_serves_the_favicon` (sigue: `_brand.html` conserva `width`/`height`).)

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py -q -k "tabbar or header_is_compact or event_bar_names or error_page_has_no_tabbar"` → FAIL.
- [ ] **Step 3: Implementar** `TABBAR`, `tabbar`, `chamber_text`, `event_info`, `common_context`; las plantillas; los iconos; `error.html` sin cambios de contexto. Ajustar los tests existentes que fijaban el marcado antiguo.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit**: `git commit -m "Make the header compact, add the mobile tab bar and name the chamber in the event bar"`.

---

### Task 4: tema de los gráficos, hemiciclo nuevo y leyenda de píldoras

**Files:**
- Create: `web/dist/js/pages/legend-dom.js`
- Modify: `web/dist/js/charts/base.js` (`THEME`, `tooltipStyle()`), cada `web/dist/js/charts/*.js` con `tooltip:` (añadir `...tooltipStyle()`), `web/dist/js/charts/hemicycle.js` (`hemicycleOption`, `renderHemicycle`), `web/dist/js/charts/series.js` (`legend: 'hidden'`), `web/dist/js/pages/promedios.js` (leyenda de píldoras en los dos gráficos de series), `web/templates/promedios.html` (`<div class="legend" id="series-legend" hidden>` y `id="pollster-legend"` encima de cada gráfico; contenedor `#hemicycle` con `class="chart hemi"`), `web/dist/css/site.css` (`.chart.hemi { height: auto; aspect-ratio: 2 / 1.1; max-width: 620px; margin: 0 auto; }`), `tests/test_js_charts.py`

**Interfaces:**
- Produces (`base.js`): `THEME = {fontFamily: "'DM Sans', system-ui, sans-serif", headFamily: "'Outfit', 'DM Sans', sans-serif", textColor: '#1a1c22', mutedColor: '#5a5f72', gridColor: 'rgba(0,0,0,.08)', axisColor: 'rgba(0,0,0,.15)', intervalColor: '#333', markColor: '#555', accentColor: '#1d4f86', barColor: '#3a6ea5', animationDuration}`; `export function tooltipStyle()` → `{backgroundColor: '#fff', borderColor: 'rgba(0,0,0,.08)', borderWidth: 1, borderRadius: 10, padding: [8, 12], textStyle: {color: THEME.textColor, fontSize: 12.5}, extraCssText: 'box-shadow: 0 8px 24px rgba(16,24,40,.08)'}`; `baseOption()` añade `textStyle.fontSize: 11` nada más. Los ejes de valores de todos los gráficos usan `axisLabel.color: THEME.mutedColor`, `axisLabel.fontSize: 10.5`, `splitLine.lineStyle.color: THEME.gridColor` y `axisLine.lineStyle.color: THEME.axisColor` (donde ya ponían `gridColor` para el eje, pasa a `axisColor`).
- Produces (`hemicycle.js`): `export function hemicycleOption(seats, {colors, order, nSeats, majority, fullnames, width, height})` → opción completa: serie `pie` con `startAngle: 180, endAngle: 0`, `radius: [0.52 * R, R]` y `center: [width / 2, height - 12]` en px, `R = Math.max(40, Math.min(width / 2 - 10, height - 36))`; `label: {show: seats >= 0.05 * total, position: 'inside', formatter: '{name|{b}}\n{seats|{c}}', rich: {name: {fontFamily: THEME.headFamily, fontWeight: 700, fontSize: 15, color: '#fff', align: 'center'}, seats: {fontFamily: THEME.headFamily, fontWeight: 800, fontSize: 24, color: '#fff', align: 'center'}}}`, `labelLine: {show: false}`, `itemStyle: {borderColor: '#fff', borderWidth: 1.5}`; una segunda serie `pie` silenciosa (`silent: true`, `tooltip: {show: false}`, `label: {show: false}`, mismo centro, `radius: [0.5 * R, R + 10]`) con tres datos `[majority - 0.35, 0.7, total - majority - 0.35]` en escaños, el del medio `#1a1c22` y los otros `transparent`, que dibuja la marca de la mayoría; `graphic`: píldora `rect` de 32×16 con `r: 8` y `fill: '#1a1c22'` centrada en `[width / 2, height - 12 - R - 24]` con el texto `String(majority)` (10 px, 700, blanco) encima, el total `String(total)` (`headFamily`, 800, 34 px) en `[width / 2, height - 12 - 30]` y el subtítulo `ESCAÑOS · MAYORÍA ABSOLUTA {majority}` (10,5 px, 600, `mutedColor`, `letterSpacing` 1) en `[width / 2, height - 14]` (ambos `textAlign: 'center'`, `textVerticalAlign: 'bottom'`); sin `majority`, ni marca ni píldora y el subtítulo `ESCAÑOS`. `renderHemicycle(el, seats, opts)` mide `el.clientWidth/clientHeight`, monta con `mountChart` y vuelve a aplicar `hemicycleOption` (no solo `resize`) desde un `ResizeObserver` propio guardado en `el.__hemiObserver` (uno por elemento).
- Produces (`series.js`): `legend` admite `'hidden'`: `legend: {show: false, selectedMode: true, data, selected}` (el componente existe y `dispatchAction` funciona) y `grid.top: 24`.
- Produces (`legend-dom.js`): `export function renderLegend(container, items, onToggle)`: vacía `container`, crea un `<button type="button" class="lg" data-name>` por `item` (`{name, color, on}`) con `style.setProperty('--c', color)`, `<i class="dot">` y el nombre, clase `on` cuando `item.on`; al pulsar llama `onToggle(name)`; `container.hidden = false`. `export function markLegend(container, selected)`: pone/quita `on` según `selected` (`{name: boolean}`, lo que ECharts reporta en `legendselectchanged`).
- Produces (`promedios.js`): en `wireAverage` y `wireHouse`, `render(...)` pasa `legend: 'hidden'`; tras el primer dibujo, `renderLegend(document.getElementById('series-legend'), current.parties.map((name) => ({name, color: current.colors[name], on: !state.selected || state.selected.includes(name)})), (name) => chart.dispatchAction({type: 'legendToggleSelect', name}))` y en el `legendselectchanged` existente, además de `state.selected`, `markLegend(legendEl, event.selected)`; al cambiar de vista (Partidos/Bloques) se vuelve a `renderLegend` con los nombres de la vista. El `hash` `sel=` sigue saliendo de `chosenNames` sin cambios.

- [ ] **Step 1: Tests que fallan** (en `tests/test_js_charts.py`; ampliar `run_js` para importar también `hemicycle.js` como `h` y `series.js` como `s`, y `base.js` como `base`):

```python
def test_hemicycle_puts_the_labels_inside_and_marks_the_majority():
    option = run_js('h.hemicycleOption({PP: 140, PSOE: 107, VOX: 63, ERC: 9, BNG: 2}, {colors: {PP: "#1d84ce"}, nSeats: 350, majority: 176, width: 600, height: 330})')
    pies = [s for s in option['series'] if s['type'] == 'pie']
    assert len(pies) == 2 and pies[0]['startAngle'] == 180 and pies[0]['endAngle'] == 0
    # R = max(40, min(600 / 2 - 10, 330 - 36)) = 290; centro en [width / 2, height - 12]
    assert pies[0]['center'] == [300, 318] and pies[0]['radius'] == [150.8, 290]
    assert pies[1]['center'] == [300, 318] and pies[1]['radius'] == [145, 300]
    labels = {row['name']: row['label']['show'] for row in pies[0]['data']}
    assert labels == {'PP': True, 'PSOE': True, 'VOX': True, 'ERC': False, 'BNG': False}
    assert pies[0]['label']['position'] == 'inside' and pies[0]['labelLine']['show'] is False
    assert pies[0]['label']['rich']['seats']['fontSize'] == 24 and 'Outfit' in pies[0]['label']['rich']['seats']['fontFamily']
    assert pies[1]['silent'] is True and [round(d['value'], 2) for d in pies[1]['data']] == [175.65, 0.7, 173.65]
    texts = [g for g in option['graphic'] if g['type'] == 'text']
    assert [g['style']['text'] for g in texts] == ['176', '350', 'ESCAÑOS · MAYORÍA ABSOLUTA 176']
    assert [g for g in option['graphic'] if g['type'] == 'rect'][0]['shape'] == {'x': 284, 'y': 4, 'width': 32, 'height': 16, 'r': 8}
    assert pies[0]['data'][0]['itemStyle']['color'] == '#1d84ce' and pies[0]['data'][3]['itemStyle']['color'] == '#9e9e9e'
    assert option['tooltip']['borderRadius'] == 10


def test_hemicycle_without_majority_has_no_mark():
    option = run_js('h.hemicycleOption({A: 10, B: 5}, {nSeats: 15, width: 400, height: 220})')
    assert len([s for s in option['series'] if s['type'] == 'pie']) == 1
    assert [g['style']['text'] for g in option['graphic'] if g['type'] == 'text'] == ['15', 'ESCAÑOS']
    assert [g for g in option['graphic'] if g['type'] == 'rect'] == []


SERIES_ARGS = ("{series: {dates: ['2026-10-01', '2026-10-02'], parties: ['PP'], mean: {PP: [30, 31]}, lo: {PP: [29, 30]}, "
               "hi: {PP: [31, 32]}}, polls: {parties: ['PP'], polls: []}, projection: null, parties: ['PP'], selected: null, "
               "colors: {PP: '#1d84ce'}, asOf: '2026-10-02', when: '2026-10-02', window: 'all', legend: %s}")


def test_series_hidden_legend_keeps_the_component_and_the_theme():
    option = run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))
    assert option['legend'] == {'show': False, 'selectedMode': True, 'type': 'scroll', 'top': 0, 'data': ['PP'], 'selected': {'PP': True}}
    assert option['grid']['top'] == 24 and option['yAxis']['axisLabel']['color'] == '#5a5f72'
    assert option['yAxis']['splitLine']['lineStyle']['color'] == 'rgba(0,0,0,.08)' and option['tooltip']['borderRadius'] == 10
    shown = run_js('s.seriesOption(%s)' % (SERIES_ARGS % 'true'))
    assert shown['legend']['show'] is True and shown['grid']['top'] == 40
    theme = run_js('base.THEME')
    assert theme['fontFamily'].startswith("'DM Sans'") and theme['headFamily'].startswith("'Outfit'") and theme['mutedColor'] == '#5a5f72'
```

(`seriesOption(args)` recibe el mismo objeto que `renderSeries(el, args)` sin `el` y devuelve la opción; `renderSeries` pasa a ser `mountChart(el, seriesOption(args))` más lo que ya hace con `setWindow`. Ajustar `SERIES_ARGS` a la firma real si `renderSeries` exige alguna clave más.)

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_js_charts.py -q -k "hemicycle or hidden_legend"` → FAIL (`hemicycleOption` no existe).
- [ ] **Step 3: Implementar** `base.js` (tema, `tooltipStyle`), `hemicycle.js` (`hemicycleOption` + `renderHemicycle` con su observer), `series.js` (`seriesOption` extraída y `'hidden'`), `legend-dom.js`, el cableado en `promedios.js`, los `tooltip` de todos los `charts/*.js` con `...tooltipStyle()` y los colores de ejes del tema; `promedios.html` con los dos `<div class="legend" … hidden>` y `#hemicycle` con `class="chart hemi"`; CSS de `.chart.hemi`.
- [ ] **Step 4: Tests en verde**: `python -m pytest tests/test_js_charts.py tests/test_js_blocks.py -q` → PASS; suite completa → PASS.
- [ ] **Step 5: Comprobación en el navegador** (uvicorn con el paquete local, como en la tarea 2): en `/es`, la leyenda de píldoras enciende y apaga partidos en los dos gráficos, el `hash` `sel=` cambia al pulsar, recargar con `#sel=PP,PSOE` deja solo esos dos encendidos; el hemiciclo llena la tarjeta con los números dentro y la marca en 176; redimensionar la ventana lo redibuja; sin errores en consola. Capturas en `.superpowers/` o en el informe.
- [ ] **Step 6: Commit**: `git commit -m "Restyle the charts, draw the hemicycle with the seats inside and add the pill legend of the series"`.

---

### Task 5: Promedios: KPIs, tarjetas de partido, tira del líder y tablas con barras

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`kpi_rows`, `party_cards`, `lead_strip`, `result_rows`/`block_result_rows` con `seats_lo`/`seats_hi`, `promedios_context`), `mtpy/lib/pages/common.py` (`last_results`, `hemicycle_order`), `mtpy/lib/pages/__init__.py`, `web/templates/promedios.html`, `web/templates/_polls_table.html` (botones `.btn`, "Descargar CSV" al lado de la paginación), `web/dist/js/pages/promedios.js` (`paintCards`: sparklines), `web/dist/js/sparkline.js` (nuevo, puro), `tests/test_pages.py`, `tests/test_js_charts.py` (o `tests/test_js_sparkline.py`)

**Interfaces:**
- Produces (`common.py`): `HEMICYCLE_BLOCKS = ('Izquierda', 'Separatista', 'Regionalista', 'Derecha')`; `hemicycle_order(names: list, parties: Catalog) -> list` (por bloque en ese orden, desconocido al final, y dentro del bloque por orden del catálogo, el espejo de `Catalog.order` de `catalog.js`); `last_results(scope: str) -> dict`: `{party: {'seats', 'pct'}}` de la elección publicada más reciente del ámbito con parte `results` (`published_events(scope)[0]['date']` y `event_part(scope, date, 'results')['rows']`), `{}` sin elecciones o sin `results`; `delta_text(value, unit) -> str`: `'+3'`/`'−14'`/`'0'` (signo menos tipográfico U+2212) seguido de `' esc.'` o `' pt'` con un decimal en los puntos (`fmt_num(x, 1)`).
- Produces (`promedios.py`):
  - `kpi_rows(summary: dict, meta: dict, parties: Catalog, polls: list, mode: str) -> list`: cinco dicts `{'label', 'value', 'unit', 'support', 'ok': bool, 'color': str | None}`: ("Primer partido", siglas del partido de mayor `seats` (empate: mayor `pct`), `'{pct} del voto'`, `'P(primero) {p_first}'`, False, color); ("Escaños del líder", `seats`, `'de {n_seats}'`, `'Rango 95 %: {lo}–{hi}'` (`fmt_range(seats_lo, seats_hi, 0)`), False, None); ("Mayoría absoluta", `fmt_prob(p)` del bloque de `summary.vs` con mayor `p_majority`, el nombre del bloque, `'{partidos de bmaps.vs[bloque] unidos por " + "}: {seats} escaños'`, `p >= 0.5`, None); ("Sondeos del ciclo", `fmt_int(meta.n_polls)`, `'de {meta.n_pollsters} encuestadoras'`, `'Desde el {fecha del sondeo más antiguo}'`, False, None); ("Último sondeo", `fmt_date_short(date)` (nuevo: `'3 oct'`), la encuestadora, `'para {sponsor} · {sample} entrevistas'` (sin sponsor: `'{sample} entrevistas'`; sin muestra, solo el sponsor), False, None). La muestra de un sondeo es `poll.get('sample_size', poll.get('sample'))` (el paquete real lleva `sample_size`; la fixture, `sample`). Sin `polls`, el último KPI no sale (cuatro); sin `vs`, el tercero no sale.
  - `party_cards(summary: dict, meta: dict, parties: Catalog, results: dict) -> list`: los partidos de `meta.bmaps.main` (en ese orden) presentes en `summary.parties`, cada uno `{'name', 'fullname', 'color', 'rank' (1 + posición por seats desc, pct desc), 'seats', 'pct', 'delta_seats' (`delta_text`, o `None`), 'delta_pct' (o `None`), 'new': bool (no está en `results`)}`; sin `bmaps.main`, los seis primeros por escaños.
  - `lead_strip(summary: dict, majority: int, with_first: bool = True) -> dict`: `{'name', 'seats', 'short': majority - seats (o 0 si lo supera), 'over': seats >= majority, 'p_first'}` del partido con más escaños.
  - `result_rows` añade `seats_lo`, `seats_hi`; `block_result_rows` también; `promedios_context` añade `kpis`, `cards`, `lead`, `pct_max` (el mayor `pct` de `result_rows`), `n_seats`, `majority` (del `summary`, con `meta` como respaldo) y en `initial`: `cards` (`[{'name', 'color'}]`, para los sparklines) — `series` ya está.
- Produces (`promedios.html`): tras la barra de evento y antes de `#series`: `<div class="kpis" id="kpis">` con `<div class="kpi"><div class="l">…</div><div class="v"{% if kpi.color %} style="color: …"{% endif %}>{{ value }}<small>{{ unit }}</small></div><div class="s{% if kpi.ok %} ok{% endif %}">{{ support }}</div></div>`; `<div class="pcards" id="pcards">` con `<article class="pc" style="--c: {{ color }}" data-name="{{ name }}"><span class="rk">{{ rank }}.º</span><div class="nm">…</div><div class="fl">…</div><div class="big"><b>{{ seats }}</b><span>{{ pct|pct }}</span></div><div class="lb"><span>escaños</span><span>voto</span></div><div class="dl"><span>{{ delta_seats or 'nuevo' }}{% if delta_seats %} <i>vs {{ results_year }}</i>{% endif %}</span><span>{{ delta_pct }} <i>pt</i></span></div><span class="sl">Último año</span><svg class="spark" viewBox="0 0 180 28" preserveAspectRatio="none" aria-hidden="true"></svg></article>` (`results_year` = el año de la última elección, en el contexto; sin resultados, el `.dl` no sale); el hemiciclo con `<p class="lead-strip">Líder: <b>{{ lead.name }}</b> · <b>{{ lead.seats }} escaños</b> · {% if lead.over %}supera la mayoría en <b>{{ lead.seats - majority }}</b>{% else %}le faltan <b>{{ lead.short }}</b> para la mayoría{% endif %} · P(primero) <b>{{ lead.p_first|prob }}</b></p>` bajo `#hemicycle`; `#parties-table` con columnas Partido (`.pn` con `.dot`), % Voto (`{{ row.pct|pct }}`), `<td><span class="bar"><i style="width: {{ row.pct|bar(0, pct_max) }}%; background: {{ color }}"></i></span></td>`, Escaños (`<b>`), Rango 95 % (`{% set r = row.seats_lo|rng(row.seats_hi, row.seats, n_seats, majority) %}{% if r %}<span class="rng"><i style="left: {{ r.left }}%; width: {{ r.width }}%; background: …"></i><b style="left: calc({{ r.mark }}% - 1px); background: …"></b>{% if r.line is not none %}<u style="left: {{ r.line }}%"></u>{% endif %}</span> <span class="mut">{{ row.seats_lo|range(row.seats_hi, 0) }}</span>{% else %}–{% endif %}`), P(primero); `#blocks-result-table` igual con P(mayoría). Los botones de `#polls-pag` con `class="btn"`; `Descargar CSV` como `<a class="mut">` al final de `.pag` (y fuera de `.ctrl-bar`).
- Produces (`sparkline.js`): `export function sparklinePoints(values, width = 180, height = 28)` → `[[x, y], …]` para los valores no nulos (índices equiespaciados en `width`, `y` de `height - 3` (mínimo) a `3` (máximo); un solo valor distinto → `y = height / 2`), `[]` sin valores; `export function sparklineSvg(svg, values)`: pinta en `svg` (SVG ya en el DOM) una `polyline` (`points`, `fill="none"`, `stroke="#fff"`, `stroke-width="2"`, `stroke-linejoin="round"`, `vector-effect="non-scaling-stroke"`) y un `circle` final (`r="2.6"`, `fill="#fff"`) con `createElementNS`. `promedios.js`: `paintCards(initial)` → para cada `article.pc[data-name]`, los últimos 365 días de `initial.series.mean[name]` muestreados cada 7 (`series.dates` ya es diario) → `sparklineSvg`.

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`; la fixture `api` tiene un partido, `PP`, con `summary` de un partido; para los deltas, la fixture `events_site` publica la elección de 2023 cuyo `event-results` da PP 160 escaños y 33,06 %):

```python
def test_promedios_opens_with_the_kpis_the_party_cards_and_the_lead_strip(events_site):
    page = html(call(events_site, '/es')[2])
    assert page.index('id="kpis"') < page.index('id="pcards"') < page.index('id="series"')
    kpis = re.findall(r'<div class="kpi"><div class="l">([^<]+)</div>', page)
    assert kpis == ['Primer partido', 'Escaños del líder', 'Mayoría absoluta', 'Sondeos del ciclo', 'Último sondeo']
    assert '<div class="v" style="color: #1d84ce">PP<small>33,1 % del voto</small></div>' in page.replace('\n', '')
    assert '135<small>de 350</small>' in page and 'Rango 95 %: 120–150' in page
    assert '<div class="s">PP: 135 escaños</div>' in page and '41 %<small>Derecha</small>' in page
    assert '120<small>de 15 encuestadoras</small>' in page and '5 oct<small>Sigma Dos</small>' in page
    card = re.search(r'<article class="pc" style="--c: #1d84ce" data-name="PP">(.*?)</article>', page, re.S).group(1)
    assert '<span class="rk">1.º</span>' in card and '<b>135</b><span>33,1 %</span>' in card
    assert '−25 esc. <i>vs 2023</i>' in card and '+0,0 <i>pt</i>' in card  # 135 − 160; 33,1 − 33,06 = +0,04 → +0,0
    assert 'class="spark"' in card and 'Último año' in card
    assert 'Líder: <b>PP</b> · <b>135 escaños</b> · le faltan <b>41</b> para la mayoría · P(primero) <b>62 %</b>' in page.replace('\n', '')


def test_promedios_party_table_draws_the_vote_bar_and_the_seat_range(api):
    page = html(call(api, '/es')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert '<span class="bar"><i style="width: 100.0%; background: #1d84ce"></i></span>' in table
    assert 'class="rng"' in table and 'left: 34.3%; width: 8.6%' in table and 'calc(38.6% - 1px)' in table and '<u style="left: 50.3%"></u>' in table
    assert '120–150' in table and '<th scope="col">Rango 95 %</th>' in table


def test_promedios_without_polls_part_and_with_an_unknown_party_still_renders(api, fresh_app, tmp_path):
    summary = fixture_data('summary')
    summary['parties'].append(dict(summary['parties'][0], name='XX', seats=3, pct=1.0, seats_lo=0, seats_hi=5, p_first=0))
    summary['totals']['XX'] = 3
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'summary', 'nowcast'), 'summary', summary, 'es', run_id=RUN)
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'polls')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/es')
    page = html(body)
    assert status == 200 and 'Último sondeo' not in page and 'id="kpis"' in page
    assert 'data-name="XX"' not in page  # fuera de bmaps.main
    assert 'background: #9e9e9e' in re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
```

(Comprobar primero en `bundle.path_part` la firma real de las partes de modo y si `promedios_context` tolera la falta de `polls`: hoy `part()` lanza 404; el contexto pasa a leerla con `bundle_part(site().run_file, scope, run, 'polls')` y, sin ella, `poll_rows = []`, sin tarjeta "Todas las encuestas" (`{% if table_total %}`) ni "Una encuestadora frente al promedio" y sin KPI de último sondeo; `initial.polls = {'parties': [], 'polls': []}`.)

Tests en node (`tests/test_js_sparkline.py`, con el arnés de `test_js_charts.py` sin ECharts):

```python
def test_sparkline_points_scale_the_values_and_skip_the_gaps():
    assert run_js('sparklinePoints([1, null, 3, 2], 180, 28)') == [[0, 25], [120, 3], [180, 14]]
    assert run_js('sparklinePoints([5, 5], 180, 28)') == [[0, 14], [180, 14]]
    assert run_js('sparklinePoints([], 180, 28)') == []
```

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py tests/test_js_sparkline.py -q -k "promedios_opens or party_table_draws or without_polls_part or sparkline"` → FAIL.
- [ ] **Step 3: Implementar** las funciones de contexto, la plantilla, `sparkline.js`, `paintCards`, y `fmt_date_short` en `filters.py` (`'3 oct'`, sin año; registrado como `date_short`). Adaptar los tests existentes de Promedios que fijan la tabla (`test_promedios_central_block_is_three_cards_without_ranges` queda invertido: ahora hay rango; renombrarlo a `…_with_ranges`).
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Navegador**: `/es` con el paquete local: KPIs, seis tarjetas con sparkline, tira del líder, barras y rangos; en 390 px el carrusel de tarjetas desliza y queda a 16 px de los KPIs. Sin errores de consola.
- [ ] **Step 6: Commit**: `git commit -m "Open Promedios with the KPI row, the party cards and the lead strip, and draw bars and ranges in the tables"`.

---

### Task 6: la portada

**Files:**
- Create: `web/dist/js/pages/inicio.js`
- Modify: `mtpy/lib/pages/inicio.py` (`INTRO`, `hero_context`, `hero_segments`, `latest_polls`, `inicio_context`), `mtpy/lib/pages/__init__.py`, `web/templates/index.html` (reescrita), `tests/test_pages.py`
- Delete: `web/templates/_event_card.html` (solo la usaba `index.html`)

**Interfaces:**
- Produces (`inicio.py`):
  - `INTRO = 'Demoscopio promedia los sondeos de cada elección, simula el reparto de escaños y puntúa a las encuestadoras según lo que acertaron. Sin cookies y con el método a la vista.'`
  - `hero_context(card: dict, scopes: list, today=None) -> Optional[dict]`: `None` si `card['published']` es falso; si no, lee `meta`, `headline[mode]`, `summary` del modo (`bundle_part(site().run_file, code, run, 'summary', mode)`) y `last_results(code)`, y devuelve `{'code', 'href', 'href_seats', 'title' (`event_title`), 'flag': code == 'es', 'chamber': 'Congreso de los Diputados' | 'Parlamento', 'n_seats', 'majority', 'date', 'days_text', 'eyebrow': 'Generales' | nombre corto del ámbito (el `name` del catálogo), 'mode_label' ('Si hoy hubiera elecciones' | 'Pronóstico' | 'Pronóstico definitivo', la regla de `_event_card.html`), 'side_title' ('Si hoy hubiera generales' | 'Si hoy hubiera elecciones en {name}' | 'Pronóstico para {…}' en modo forecast), 'leader': {'name', 'fullname', 'color', 'seats', 'pct'}, 'segments': hero_segments(...), 'majority_pct': 100 * majority / n_seats (un decimal), 'ends': [{'name', 'seats'}] (los bloques de `summary.vs`, "Izquierda" primero si existe), 'majority_chip': {'block', 'seats', 'p', 'clear': p >= .5} | None, 'deltas': [{'name', 'text'}] (los dos partidos con más escaños que estén en `last_results`, `delta_text(seats - last, 'esc.')`), 'results_year', 'run_at', 'n_polls', 'n_pollsters', 'as_of'}`; sin `summary`, `segments = []`, `ends = []`, `majority_chip` sale de `headline[mode]['p_majority']` (el bloque mayor, sin `seats`) y el `leader` de `headline[mode]['parties']` (el de mayor `pct`, como `card_headline`).
  - `hero_segments(summary: dict, parties: Catalog) -> list`: `[{'name', 'color', 'seats', 'width'}]` (anchura `100 * seats / n_seats` con un decimal) de los partidos con `seats > 0` de `summary.totals` (o `summary.parties`) en `hemicycle_order`.
  - `latest_polls(code: str, run: str, parties: Catalog, top: list, n: int = 5) -> list`: `[{'date', 'day', 'month' ('oct'), 'pollster', 'sponsor', 'sample', 'values': [{'name', 'color', 'pct'}]}]` de las `n` últimas filas de la parte `polls` por `date` desc (y `pollster` asc a igual fecha), `values` para `top` (las siglas de los tres partidos con más `pct` del `summary` del hero); `[]` sin parte.
  - `inicio_context` devuelve además `hero`, `autonomics` (las tarjetas publicadas distintas del hero, orden de `event_cards`), `unpublished` (los `title` sin la parte "Elecciones autonómicas · " de las no publicadas), `latest_polls`, `past_shown` (las 12 primeras de `past`), `past_more` (las restantes), `intro`, y `initial = {'hero': {'seats': {name: seats}, 'colors': {name: color}, 'fullnames': {…}, 'order': [names], 'n_seats', 'majority'}}` solo con hero y `segments`; `cards` desaparece del contexto (y `card_headline` se queda para el `leader` de las filas).
- Produces (`index.html`): el hero de D4 con estos `id`: `#hero`, `#hero-card`, `#hero-seatbar`, `#auto-rows`, `#hemicycle-mini` (`<div class="chart hemi" …>`, dibujado por JS; `{% if initial is defined %}`), `#hemi-legend`, `#latest-polls`, `#more` (las tres tarjetas, con `<span class="ic"><svg><use href="#i-target"/></svg></span>` etc.), `#how` (los cuatro pasos, texto fijo de la maqueta), `#past`; `<script type="module" src="/dist/js/pages/inicio.js">` en `scripts` solo con hero. Las tarjetas grises desaparecen; la frase de no publicados: `<p class="row-foot">Sin pronóstico todavía: {{ unpublished|join(', ') }}.</p>`.
- Produces (`inicio.js`): lee `initial`, `renderHemicycle(document.getElementById('hemicycle-mini'), hero.seats, {colors: hero.colors, order: (names) => hero.order.filter((n) => names.includes(n)), nSeats: hero.n_seats, majority: hero.majority, fullnames: hero.fullnames})` dentro de `tryDraw`, con `mountWhenReady`.

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`; sustituyen a `test_home_lists_every_catalogue_scope_and_marks_the_unpublished_ones`, `test_home_puts_the_published_regional_scope_before_the_unpublished_ones`, `test_home_lists_the_published_events_with_their_headline`, `test_home_cards_are_links`, `test_home_cards_follow_the_default_mode_while_frozen` y `test_event_cards_of_the_home_are_h3_under_the_section_h2`, que se reescriben contra el marcado nuevo manteniendo lo que comprueban: ámbitos publicados y no publicados, orden, enlaces, modo en veda):

```python
def test_home_hero_is_the_national_scope_with_its_figures(api):
    page = html(call(api, '/')[2])
    assert '<h1 class="hero-t">' in page and 'un solo <em>promedio</em>.' in page and pages.inicio.INTRO in page
    assert 'Generales · 29 nov 2026 · ' in page and 'id="hero-card"' in page and 'class="flag"' in page
    assert 'Congreso de los Diputados · 350 escaños · 29 nov 2026' in page
    assert '<b style="color: #1d84ce">135</b>' in page.replace('\n', '') and 'primer partido con el <strong>33,1 %</strong>' in page
    bar = re.search(r'<div class="seatbar" id="hero-seatbar">(.*?)</div>\s*<div class="chips">', page, re.S).group(1)
    assert '<i style="width: 38.6%; background: #1d84ce" title="PP 135"></i>' in bar and 'left: 50.3%' in bar
    assert 'Derecha 135' in bar
    # con p < 0,5 el chip va en gris: "Sin mayoría clara · Derecha 41 %" (D4); con p >= 0,5, `.chip.ok` "Derecha 203 escaños · > 99 % de mayoría absoluta"
    assert '<span class="chip">Sin mayoría clara <small>· Derecha 41 %</small></span>' in page.replace('\n', '')
    assert 'href="/es">Ver el promedio' in page and 'class="btn" href="/es/escanos">Escaños' in page
    assert 'Última publicación: 8 oct 2026 · 120 sondeos de 15 encuestadoras' in page
    assert 'id="hemicycle-mini"' in page and '/dist/js/pages/inicio.js' in page
    data = initial(call(api, '/')[2])
    assert data == {'hero': {'seats': {'PP': 135}, 'colors': {'PP': '#1d84ce'}, 'fullnames': {'PP': 'Partido Popular'}, 'order': ['PP'], 'n_seats': 350, 'majority': 176}}
    polls = re.search(r'<div class="plist" id="latest-polls">(.*?)</div>\s*</section>', page, re.S).group(1)
    assert polls.count('class="pr"') == FIXTURE_POLLS and '<b>5</b>oct' in polls and 'Sigma Dos' in polls and '33,1 %' in polls


def test_home_lists_the_regional_scopes_as_rows_and_names_the_unpublished_ones(api, fresh_app):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    rows = re.search(r'<div class="card rows" id="auto-rows">(.*?)</div>\s*</div>\s*<aside', page, re.S).group(1)
    assert rows.count('class="row"') == 1 and 'href="/es-md"' in rows and '<div class="n">Madrid</div>' in rows
    assert 'PP 33,1 % · 135 esc.' in rows and 'Derecha: <b>41 %</b> de mayoría' in rows
    assert 'Sin pronóstico todavía: Andalucía, Aragón' in rows and rows.count(',') >= 15 and 'Madrid,' not in rows
    assert 'card-muted' not in page and 'Pronóstico no publicado todavía' not in page


def test_home_hero_falls_back_to_the_first_published_scope(api, fresh_app, tmp_path):
    write_bundle(fresh_app.fs, scope='es-md', run_id='20261009-120000', manifest=True)
    manifest = json.loads((tmp_path / bundle.PREFIX / bundle.path_manifest()).read_text())
    del manifest['data']['scopes']['es']
    (tmp_path / bundle.PREFIX / bundle.path_manifest()).write_text(json.dumps(manifest))
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert re.search(r'<span class="eyebrow"[^>]*>Madrid · \d+ \w+ \d{4} · ', page)
    assert 'class="flag"' not in page and 'Parlamento · 350 escaños · ' in page and 'Si hoy hubiera elecciones en Madrid' in page
    assert 'id="auto-rows"' not in page  # el único ámbito publicado es el hero
    assert 'href="/es-md/escanos"' in re.search(r'<nav class="tabbar".*?</nav>', page, re.S).group(0)


def test_home_without_summary_or_polls_keeps_the_hero_without_the_bar(api, fresh_app, tmp_path):
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'polls')))
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'summary', 'nowcast')))
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/')
    page = html(body)
    assert status == 200 and 'id="hero-card"' in page and 'id="hero-seatbar"' not in page
    assert 'id="latest-polls"' not in page and 'initial-data' not in page
    assert '<b style="color: #1d84ce">135</b>' in page.replace('\n', '')  # el líder sale del headline
    assert 'Sin mayoría clara <small>· Derecha 41 %</small>' in page.replace('\n', '')


def test_home_hero_reads_the_chamber_from_the_summary_without_meta_seats(api, fresh_app):
    meta = fixture_data('meta')
    del meta['n_seats'], meta['majority']
    writer = write_bundle(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().manifest_cache_clear()
    page = html(call(api, '/')[2])
    assert 'Congreso de los Diputados · 350 escaños · ' in page and 'id="hero-seatbar"' in page and 'left: 50.3%' in page
    assert initial(call(api, '/')[2])['hero']['majority'] == 176


def test_home_closing_sections_link_the_model_pages_and_the_past(events_site):
    page = html(call(events_site, '/')[2])
    more = page[page.index('id="more"'):page.index('id="how"')]
    for href, icon in (('/elecciones', 'i-target'), ('/casas', 'i-trophy'), ('/metodo', 'i-book')):
        assert re.search(r'<a class="card card-link" href="%s">\s*<span class="ic"><svg[^>]*><use href="#%s"' % (href, icon), more), href
    assert page.count('<div class="card step">') == 4 and 'Recoge los sondeos' in page
    past = page[page.index('id="past"'):]
    assert 'href="/elecciones?scope=es&amp;date=2023-07-23"' in past and past.index('2023') < past.index('2019')
```

(Para `test_home_hero_falls_back…` comprobar cómo `published_scopes()` y `resolve_scope(None)` leen el manifest: si `es` fuera obligatorio en algún punto, el test se ajusta a quitar `es` del catálogo con `monkeypatch` sobre `webapi.checks.catalogue` como hacen otros tests; el comportamiento exigido es el de D4.)

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py -q -k "home_"` → FAIL.
- [ ] **Step 3: Implementar** `inicio.py`, `index.html`, `inicio.js`; borrar `_event_card.html`; reescribir los tests antiguos de la portada contra el marcado nuevo (misma intención).
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Navegador**: `/` con el paquete local a 1440, 1000 y 390 px: hero, hemiciclo pequeño, últimas encuestas, autonómicas bajo la tarjeta destacada, secciones finales; en móvil, la columna lateral entre la tarjeta y las autonómicas; sin desplazamiento horizontal (`document.documentElement.scrollWidth <= window.innerWidth`).
- [ ] **Step 6: Commit**: `git commit -m "Rebuild the home with the hero of the next general election, the regional rows and the closing sections"`.

---

### Task 7: el resto de páginas

**Files:**
- Modify: `web/templates/escanos.html` (`.rng` en `#parties-table`/`#blocks-table`, píldoras `.lg` de la calculadora, `#coalition-result` como `<p class="chip" …>`), `web/dist/js/pages/escanos.js` (`rowHeight: 64` en `renderRidge`; el chip de resultado con clase `ok` cuando `p >= .5`), `web/templates/elecciones.html` (`#figures-list` como `<div class="kpis">` con cuatro `.kpi` y los mismos `id` en `.v`), `web/dist/js/pages/history-dom.js` (`fillFigures` escribe en los `.v`; `hidden` del contenedor igual), `web/templates/casa.html` (KPIs de la cabecera), `web/templates/eleccion.html` (tira del líder bajo el hemiciclo, `.bar` en `#results-table`; `#hemicycle` con `class="chart hemi"`), `mtpy/lib/pages/elecciones.py` (`eleccion_context` añade `lead` = `lead_strip` del resultado, sin `p_first`, y `pct_max`), `mtpy/lib/pages/casas.py` (`casa_context` añade `kpis`: tres dicts como los de Promedios), `web/templates/_metodo_nav.html` (`class="seg metodo-nav"` con `<a class="on">` en la activa), `web/templates/metodo.html` (índice como `.card-link` sin icono), `web/dist/css/site.css` (lo que falte: `.figures` fuera, `.metodo-nav`, prosa a 72ch), `tests/test_pages.py`

**Interfaces:**
- Consumes: `lead_strip` (tarea 5), `rng`/`bar` (tarea 2), `.kpis` (tarea 2), `renderHemicycle` (tarea 4).
- Produces: `casa_context['kpis']` = `[{'label': 'Rating', 'value': fmt_num(rating, 1), 'unit': '', 'support': 'entra así en el promedio', 'ok': False, 'color': None}, {'label': 'Calidad previa', 'value': fmt_num(quality, 1), …, 'support': ''}, {'label': 'Tipo', 'value': mtype or '–', …}]`; la plantilla pinta `.kpis` en la tarjeta `profile` en vez de `<p class="muted">Tipo: …</p>`. `eleccion_context['lead']` = `lead_strip(results, majority, with_first=False)` con `results` = `{'parties': [{'name': row['party'], 'seats': row['seats']}]}` (sin filas, `None`).

- [ ] **Step 1: Tests que fallan** (en `tests/test_pages.py`):

```python
def test_escanos_table_draws_the_seat_range_and_the_ridge_is_compact(api):
    page = html(call(api, '/es/escanos')[2])
    table = re.search(r'<table id="parties-table">(.*?)</table>', page, re.S).group(1)
    assert 'class="rng"' in table and '<u style="left: 50.3%"></u>' in table
    assert '<fieldset class="pills lg" id="coalition-parties">' in page
    assert '<p class="chip" id="coalition-result">' in page


def test_history_figures_are_a_kpi_row(history_site):
    page = html(call(history_site, '/elecciones')[2])
    kpis = re.search(r'<div class="kpis" id="figures-list" hidden>(.*?)</div>\s*<ul class="complements"', page, re.S).group(1)
    assert kpis.count('<div class="kpi">') == 4
    for id_ in ('figure-polls', 'figure-events', 'figure-mean', 'figure-bias'):
        assert 'class="v" id="%s"' % id_ in kpis, id_
    assert 'id="figure-mean-label"' in kpis


def test_profile_opens_with_three_kpis_and_keeps_the_path_table(houses):
    page = html(call(houses, '/casas', query='scope=es&id=2')[2])
    profile = page[page.index('id="profile"'):page.index('id="path"')]
    assert re.findall(r'<div class="kpi"><div class="l">([^<]+)</div>', profile) == ['Rating', 'Calidad previa', 'Tipo']
    assert 'Tipo: ' not in profile and 'id="path-table"' in page and 'id="rating-path"' in page


def test_election_detail_has_the_lead_strip_without_p_first(events_site):
    page = html(call(events_site, '/elecciones', query='scope=es&date=2023-07-23')[2])
    strip = re.search(r'<p class="lead-strip">(.*?)</p>', page, re.S).group(1).replace('\n', '')
    assert 'Líder: <b>PP</b> · <b>160 escaños</b>' in strip and 'P(primero)' not in strip
    assert 'le faltan <b>16</b> para la mayoría' in strip
    assert 'class="chart hemi"' in page


def test_method_nav_is_a_segmented_control(api):
    page = html(call(api, '/metodo/promedio')[2])
    nav = re.search(r'<nav id="metodo-nav" class="seg metodo-nav"[^>]*>(.*?)</nav>', page, re.S).group(1)
    assert 'class="on" href="/metodo/promedio"' in nav or re.search(r'<a class="on"[^>]*href="/metodo/promedio"', nav)
```

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_pages.py -q -k "seat_range_and_the_ridge or kpi_row or three_kpis or lead_strip_without or method_nav"` → FAIL.
- [ ] **Step 3: Implementar** plantillas, contextos y JS; CSS restante (`.metodo-nav`, prosa `max-width: 72ch` en `.how-body`, `#glosario dd`, las tarjetas de texto del método; `.figures` retirado). Ajustar los tests existentes que fijaban `<p class="muted">Tipo:` o `<dl class="figures"`.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Navegador**: `/es/escanos`, `/elecciones`, `/casas`, `/casas?id=`, `/elecciones?date=`, `/metodo` y las cuatro subpáginas con el paquete local y la base (`.env` del repositorio con `DB_STATEMENT_TIMEOUT=10000`; sin base, `/casas` da 503 y se comprueba con las capturas de fase 7f): todo con el sistema nuevo, sin errores de consola.
- [ ] **Step 6: Commit**: `git commit -m "Restyle Escaños, the error history, the pollster pages, the election detail and the method"`.

---

### Task 8: marca con Outfit, limpieza y cierre

**Files:**
- Modify: `web/brand/build.py` (`FONT`, constantes de la lente calculadas, caja del wordmark desde los glifos), `web/dist/img/logo.svg` (regenerado), `web/dist/img/favicon.svg`, `favicon.ico`, `apple-touch-icon.png` (regenerados, iguales), `web/templates/_brand.html` (`width`/`height` nuevos), `web/dist/vendor/fonts/LICENSE` (ya sin Source), `tests/test_deploy_files.py`, `deploy/README.md` (apartado "La web": fuentes y marca; "Desarrollo sin Docker" no cambia), `docs/superpowers/specs/2026-10-15-web-fase-8-rediseno-visual-design.md` (estado al cierre)
- Delete: los tres `woff2` de Source

**Interfaces:**
- Produces (`build.py`): `FONT = …/Outfit-800.woff2`; `def glyph_box(font, char) -> tuple` (`(xMin, yMin, xMax, yMax)` con `fontTools.pens.boundsPen.BoundsPen`); las constantes de la lente se calculan en `wordmark()` a partir de la "o" de Outfit: `height = yMax - yMin`, `LENS_RADIUS = height / 2`, `LENS_CENTER = ((xMin + xMax) / 2, (yMin + yMax) / 2)`, `LENS_STROKE = round(0.19 * height)`, `LENS_DOT = round(0.28 * LENS_RADIUS)`; `top`/`bottom` de la caja del wordmark desde `glyph_box(font, 'D')[3]` (o la "i" si es más alta) y `glyph_box(font, 'p')[1]`; `ICON_*` sin cambios (el favicon no cambia). El módulo sigue sin dependencias de la web.
- Produces (`README`): en "La web", un párrafo "Tipografías y marca": DM Sans y Outfit vendidas en `web/dist/vendor/fonts/` (`python web/brand/fonts.py` las descarga de Google Fonts, OFL, subconjunto latino), `python web/brand/build.py` regenera el wordmark con Outfit.

- [ ] **Step 1: Test que falla** (en `tests/test_deploy_files.py`, ampliando `test_brand_assets_match_the_templates_and_nginx_serves_the_favicon` o nuevo):

```python
def test_wordmark_is_set_in_outfit_and_the_old_fonts_are_gone():
    source = read('web/brand/build.py')
    assert "'Outfit-800.woff2'" in source and 'SourceSerif4' not in source
    fonts = ROOT / 'web' / 'dist' / 'vendor' / 'fonts'
    assert not list(fonts.glob('Source*')) and 'Source' not in read('web/dist/vendor/fonts/LICENSE')
    logo = read('web/dist/img/logo.svg')
    assert logo.count('<path d=') == 9  # D e m o s c o p i: nueve glifos, la lente aparte
    assert read('web/dist/img/favicon.svg').startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">')
```

(El favicon regenerado debe ser byte a byte el actual: comprobar con `git diff --stat web/dist/img/` que solo cambia `logo.svg`.)

- [ ] **Step 2: Ejecutar y ver que falla**: `python -m pytest tests/test_deploy_files.py -q -k wordmark_is_set` → FAIL.
- [ ] **Step 3: Implementar** `build.py`, ejecutar `python web/brand/build.py`, actualizar `_brand.html` con el `width`/`height` que imprime el SVG (el test de la marca lo exige), borrar los Source, README.
- [ ] **Step 4: Suite en verde**: `python -m pytest -m "not integration" -q` → PASS, sin avisos nuevos (anotar el número de pruebas).
- [ ] **Step 5: Capturas finales** (manuales): Chrome sin cabeza (`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=… --window-size=W,H URL`) de `/`, `/es`, `/es/escanos`, `/casas`, `/casas?id=2`, `/elecciones`, `/elecciones?scope=es&date=2023-07-23` y `/metodo` a 390, 1000, 1440 y 1920 px contra uvicorn con el paquete local y la base del `.env`; `--dump-dom` + un `grep` del `scrollWidth` no es posible sin JS: comprobar el desplazamiento horizontal en la consola del navegador normal (`document.documentElement.scrollWidth <= window.innerWidth`) en `/` y `/es` a 390 px. Guardar las capturas en `.superpowers/` y adjuntar la lista al informe.
- [ ] **Step 6: Cerrar la spec**: sección "Estado al cierre de la fase 8" con los commits, el número de pruebas, lo comprobado en el navegador y lo pendiente (p. ej. el hueco bajo la columna lateral de la portada, que Luis rellenará en otra fase; el modo oscuro).
- [ ] **Step 7: Commit**: `git commit -m "Set the wordmark in Outfit, drop the old fonts and close the phase 8 spec"`.

---

## Notas para el ejecutor

- La maqueta es la verdad visual: ante una duda de medida, color o espaciado, abrir `docs/superpowers/specs/2026-10-15-web-fase-8-mockups.html` y leer su CSS (`.page { … }` y siguientes). Las `@container` de la maqueta son `@media` en el sitio.
- Las clases nuevas conviven con las antiguas durante las tareas 2-7; al cerrar la tarea 7, `grep` de las que ya no usa ninguna plantilla ni módulo (`.tile`, `.tile-hero`, `.card-muted`, `.brand-tagline`, `.figures`, `.coalition-result`, `.event-title`, `.event-run` si se renombró) y retirarlas de `site.css`.
- `StrictUndefined`: toda clave nueva del contexto debe existir en todas las páginas que incluyan la plantilla que la usa (`tabbar` con `is defined` en `_tabbar.html`; `initial` con `is defined` en `base.html`, como hoy).
