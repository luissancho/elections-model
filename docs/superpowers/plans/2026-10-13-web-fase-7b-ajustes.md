# Web, fase 7b: ajustes tras el recorrido de Luis — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aplicar las notas de Luis tras recorrer la web de la fase 7 (2026-10-13): portada con tarjetas
clicables, Promedios con controles en tres líneas y píldoras de partidos, tooltip por sondeo, tabla de sondeos
con el nombre del sondeo y carga progresiva, Escaños simplificado (tabla, histogramas con mediana y rango,
un solo bloque, píldoras en la calculadora), diana del histórico más grande, etiquetas "Histórico Errores" y
"Ranking Encuestadoras", sin selector Hoy / Elección (Hoy por defecto, Elección en veda), fecha de hoy en los
títulos del modo Hoy, "Cómo leerlo" plegados, siglas en vez de nombres completos, barras azules y la
terminología "encuestadora" y "medio" en toda la web.

**Architecture:** Sin cambios de arquitectura ni de paquete: todo son plantillas, CSS, JS de página, módulos
puros y contextos de página. El nombre del sondeo se compone en la página como lo hace el cargador
(`pollster + ' / ' + sponsor`, `mtpy/lib/loader.py:588`), así que `polls.json` no cambia. Un componente de
píldoras (HTML + CSS, casillas ocultas con etiquetas coloreadas) sustituye a las casillas con punto en Promedios
y en la calculadora de coaliciones; las leyendas de ECharts de los dos gráficos de Promedios se ocultan porque
las píldoras son el selector. El tooltip del promedio pasa a `trigger: 'item'`.

**Tech Stack:** el de la fase 7 (Jinja2, ECharts 5.6 vendorizado, JS sin build, CSS con tokens, pytest, node
para los tests de paridad).

**Spec:** las notas de Luis del 2026-10-13 (reproducidas en "Notas de Luis") sobre la spec de la fase 7
`docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md`; la fase 7 cerró en `ef93060` (645 tests).

## Notas de Luis (2026-10-13, tras el recorrido)

Portada: botones o tarjetas (como Objetivo176) para acceder a cada ámbito, y tarjetas también para Errores,
Rating y Método. Promedios: el selector de partidos igual que el de "Una casa frente al promedio"; los tres
selectores en tres líneas, en el orden Vista, Ventana y Partidos (este sin título y sin Todos/Ninguno); por cada
punto del promedio, un recuadro con el sondeo (nombre, fecha, muestra, siglas de cada partido en su color con su
valor, en orden descendente) en lugar del día; quitar el selector Hoy / Elección (Hoy por defecto, Elección si
está congelado); en "Sondeos del ciclo", cargar más hasta tener todos y una columna "Sondeo" con el `name` en
vez de Casa y Patrocinador. Escaños: quitar "Mín–máx" y "Media", "Intervalo 95 %" → "Rango 95 %"; en los
histogramas, líneas de mediana (gruesa) y 95 % izquierda y derecha (finas), sin el valor escrito; dejar solo
"Bloques ideológicos, con todos los partidos"; calculadora con el mismo selector de partidos. Histórico de
errores: llamarlo "Histórico Errores"; la diana más grande y centrada. Rating: llamarlo "Ranking Encuestadoras".
Toda la web: "Estimación a …" muestra la fecha de hoy en modo Hoy y la del evento en Elección; todos los "Cómo
leerlo" plegados; siempre siglas en vez del nombre completo; barras por defecto en azul neutro en vez de gris;
"casa/s" → "encuestadora/s" y "patrocinador/es" → "medio/s".

## Global Constraints

- Los de la fase 7 (patrón `Page` + Jinja2 + isla + JS que solo dibuja; sin `fetch`; `StrictUndefined`;
  textos en español; sin cifras a mano; nginx, Docker, `publish`, el paquete y la API no cambian).
- El nombre de un sondeo es `pollster + ' / ' + sponsor` cuando hay `sponsor`, si no `pollster` (regla del
  cargador); se calcula en `pages/promedios.py`, nunca a mano en la plantilla.
- Terminología pública: "encuestadora/s" (nunca "casa/s" para una empresa de sondeos; "efectos de casa" pasa a
  "efectos de encuestadora"), "medio/s" (nunca "patrocinador/es"); las siglas (`name`) siempre, con el nombre
  completo solo en `title=`.
- Modo: sin control en la interfaz; `mode` lo resuelve `default_mode(frozen)`; `?mode=` sigue aceptado en la
  query (enlaces y tests) pero ninguna página lo pinta como control.
- Títulos: modo Hoy → "Estimación a {hoy en Europe/Madrid}" si el run es el último, "Estimación del {run_at}"
  si es un run fijado anterior; modo Elección → "Pronóstico para el {event_date}" / "Pronóstico definitivo
  para el {event_date}" / "Pronóstico publicado el {run_at}" como hasta ahora.
- Suite unitaria en verde (645 al empezar), `node --check` de cada JS tocado, PEP8/docstrings numpy/JSDoc.

## Review Focus

1. Un sondeo sin `sponsor` o sin `sample_size` en el tooltip y en la tabla: nombre = encuestadora sola, muestra
   "–"; nunca "None" ni "undefined". Test de node en la tarea 3.
2. Un ámbito congelado: el título dice "Pronóstico definitivo", las tarjetas del Inicio enseñan `forecast`, y
   la barra de evento no ofrece ningún control; sin JS la página sigue completa. Test en la tarea 1.
3. Las píldoras sin JS: las casillas ocultas siguen siendo accesibles con teclado (`:focus-visible` visible) y
   el formulario de coaliciones no depende de JS para marcar (el cálculo sí). Comprobación en el navegador en la
   tarea 4.
4. "Cargar más" cuando ya están todos los sondeos: el botón desaparece; con menos de 40 sondeos nunca aparece.
   Test de node en la tarea 3.
5. La sustitución de "casa" no debe tocar palabras que la contienen ("casar", "escasa", "casación") ni los
   nombres de ficheros, ids, clases CSS, rutas (`/casas`, `casa.html`, `casas.js`) o claves de datos: solo texto
   visible. Test en la tarea 1 (ninguna plantilla contiene " casa" o " casas" como palabra en texto visible; las
   rutas y los ids intactos).

---

### Task 1: Terminología, siglas, modo sin control, fechas de título, "Cómo leerlo" plegado, barras azules

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`PAGES` labels: "Histórico Errores", "Ranking Encuestadoras"; `title_of`
  con `today`; `mode_text`), `mtpy/lib/pages/promedios.py`, `escanos.py`, `elecciones.py`, `casas.py`,
  `inicio.py`, `metodo.py` (textos de titular con "casa"/"patrocinador" y nombres completos)
- Modify: `web/templates/*.html` (todas: terminología, siglas, `open=True` fuera, `_event_bar.html` sin
  formulario ni `<details>`), `web/dist/js/charts/base.js` (`THEME.barColor`), `charts/bars.js`,
  `charts/histogram.js` (color por defecto), `web/dist/js/pages/*.js` (`fullnames` → siglas en leyendas y
  tooltips; cadenas con "casa"), `web/dist/css/site.css` (`.segmented`, `.mode-help` fuera)
- Test: `tests/test_pages.py`, `tests/test_js_*.py` que fijen cadenas

**Interfaces:**
- Produces: `title_of(mode, meta, frozen, pinned, is_latest, today=None)`: `nowcast` → `'Estimación a {today}'`
  (hoy en Europe/Madrid, `filters._madrid_tz`) cuando `is_latest`, si no `'Estimación del {run_at}'`; `forecast`
  como hasta ahora. `common_context` ya no pone `mode_text` del control; conserva `mode_text` como frase de
  lectura ("Si las elecciones fueran hoy…" / "Lo que puede pasar el …") para la barra de evento.
- `THEME.barColor = '#3a6ea5'` (azul neutro) en `charts/base.js`; `renderBars` lo usa cuando `colors[name]`
  no existe (`OTHERS_COLOR` sigue siendo el gris del partido "otros" en los gráficos por partido); el
  histograma de la coalición y cualquier barra sin color de partido usan `THEME.barColor`.
- `PAGES` = Inicio, Histórico Errores, Ranking Encuestadoras, Método; `<title>` y titulares de esas páginas con
  las etiquetas nuevas.

- [ ] **Step 1: Tests que fallan**

```python
def test_public_wording_uses_encuestadora_and_medio_and_short_names(events_site):
    for path, query in (('/', ''), ('/es', ''), ('/es/escanos', ''), ('/elecciones', ''),
                        ('/elecciones', 'scope=es&date=2023-07-23'), ('/metodo', ''), ('/metodo/promedio', ''),
                        ('/metodo/escanos', ''), ('/metodo/rating', ''), ('/metodo/backtest', '')):
        page = html(call(events_site, path, query=query)[2])
        text = re.sub(r'<[^>]+>', ' ', page)
        assert not re.search(r'\b[Cc]asas?\b', text), path
        assert not re.search(r'\b[Pp]atrocinador(es)?\b', text), path
        assert 'open>' not in page and '<details class="how" open' not in page, path
    page = html(call(events_site, '/es')[2])
    assert 'Partido Popular' not in re.sub(r'<[^>]+>', ' ', page) and 'title="Partido Popular"' in page


def test_menu_labels_and_titles(api):
    page = html(call(api, '/elecciones')[2])
    assert '>Histórico Errores</a>' in page and '>Ranking Encuestadoras</a>' in page
    assert '<title>Pronóstico electoral · Histórico Errores</title>' in page


def test_event_bar_has_no_mode_control_and_the_title_shows_today(api, fresh_app):
    from datetime import date
    page = html(call(api, '/es')[2])
    assert 'name="mode"' not in page and 'mode-help' not in page and 'id="event-controls"' not in page
    assert 'Estimación a ' + pages.fmt_date(date.today().isoformat()) in page  # hoy en Europe/Madrid
    assert 'Estimación del ' in html(call(api, '/es', query='run=' + OLDER_RUN)[2])  # run fijado anterior


def test_frozen_site_has_no_control_and_shows_the_forecast(api, fresh_app):
    ...  # manifest congelado como en test_frozen_site_defaults_to_the_forecast…: 'Pronóstico definitivo' en /es, sin name="mode"
```

(`OLDER_RUN`: escribir un segundo run con `write_bundle(..., run_id='20261001-120000', manifest=False)` como en
el test de la fase 7. Usar la fecha de hoy en Europe/Madrid, no `date.today()` UTC, si el test corre de noche.)

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py -q -k "wording or labels or mode_control or frozen"` → FAIL.

- [ ] **Step 3: Implementar** — sustitución palabra a palabra en el texto visible de todas las plantillas y en
los textos de Python/JS (incluidos los tres textos del método: "efectos de casa" → "efectos de encuestadora",
"cluster por casa" → "por encuestadora", "la casa" → "la encuestadora"; concordancias femeninas intactas);
rutas, ids, clases, nombres de fichero y claves de datos intactos. Siglas: `catalog[name].fullname` →
`name` con `title="{{ fullname }}"` en tablas, tarjetas del Inicio y leyendas/tooltips de los gráficos (los
`fullnames` de los módulos de página pasan a devolver la sigla; dejar `fullnames` solo en `title`). `open=True`
fuera de las siete plantillas. `_event_bar.html`: sin `<form>`, sin `<details>`; conserva título, fecha, días,
pestañas, chip del run fijado y `mode_text`; el `region` oculto desaparece con el formulario (el formulario de
circunscripción de Escaños ya lleva `mode` y `run` ocultos). `title_of` con `today`. `THEME.barColor`.
Etiquetas del menú y `<title>`s.

- [ ] **Step 4: Suite y node** — Run: `python -m pytest -m "not integration" -q && for f in web/dist/js/charts/base.js web/dist/js/charts/bars.js web/dist/js/charts/histogram.js web/dist/js/pages/*.js; do node --check $f; done` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Use encuestadora and medio, short party names, today's date in the nowcast title, no mode control and blue bars"`

---

### Task 2: Inicio con tarjetas clicables

**Files:**
- Modify: `web/templates/index.html`, `web/templates/_event_card.html`, `web/dist/css/site.css`, `mtpy/lib/pages/inicio.py` (si hace falta `href` de las secciones)
- Test: `tests/test_pages.py`

**Interfaces:**
- Cada tarjeta de ámbito es un enlace entero: `<a class="card card-link" href="{{ card.href }}">` con el
  contenido de hoy dentro (h3 sin enlace propio, cifras, "Última publicación"), y debajo del bloque un enlace
  secundario "Escaños" que no anida dentro del `<a>` (un `<p class="card-actions">` fuera del enlace o dos
  enlaces hermanos dentro de un `<article>`: elegir el que valide HTML; nada de `<a>` dentro de `<a>`).
- La tarjeta "Para entender el modelo" pasa a tres tarjetas clicables en una `grid`: "Histórico Errores"
  (`/elecciones`), "Ranking Encuestadoras" (`/casas`), "Método" (`/metodo`), cada una con una frase.
- CSS: `.card-link` (bloque, sin subrayado, borde que se oscurece al `:hover`/`:focus-visible`, cursor
  pointer), `.card-actions`.

- [ ] **Step 1: Test que falla**

```python
def test_home_cards_are_links(api):
    page = html(call(api, '/')[2])
    assert 'class="card card-link" href="/es"' in page.replace("'", '"')
    assert page.count('card-link') >= 4  # el ámbito + las tres secciones
    assert 'href="/elecciones"' in page and 'href="/casas"' in page and 'href="/metodo"' in page
    assert '<a' not in re.search(r'<a class="card card-link"[^>]*>(.*?)</a>', page, re.S).group(1)
```

- [ ] **Step 2: Ejecutar y ver que falla** — Run: `python -m pytest tests/test_pages.py -q -k home_cards_are_links` → FAIL.
- [ ] **Step 3: Implementar** (ver "Interfaces").
- [ ] **Step 4: Suite** — Run: `python -m pytest -m "not integration" -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Make the home cards clickable tiles for the events and the global sections"`

---

### Task 3: Promedios: píldoras, tres líneas, tooltip por sondeo, tabla con "Sondeo" y carga progresiva

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`poll_name`, `chart_polls` con `name` y `sample_size`, `table_rows`
  con "Sondeo"), `web/templates/promedios.html`, `web/dist/js/pages/promedios.js`,
  `web/dist/js/charts/series.js` (tooltip `item`, leyenda ocultable), `web/dist/css/site.css` (`.pills`)
- Create: `web/dist/js/pills.js` (puro: estado de un grupo de píldoras) — opcional si el cableado cabe en
  `promedios.js` sin lógica nueva; `web/dist/js/polls-table.js` (puro: paginación de filas)
- Test: `tests/test_pages.py`, `tests/test_js_blocks.py` o nuevo `tests/test_js_polls.py`

**Interfaces:**
- `pages/promedios.py`: `poll_name(poll: dict) -> str` (`pollster` + `' / '` + `sponsor` si hay `sponsor`);
  `chart_polls` conserva además `name` (calculado), `sample_size`; `table_rows(polls, parties, n)` devuelve
  filas `[fecha, nombre, muestra, …partidos]`; `promedios_context` añade `table_total = len(polls)` y en
  `initial` la lista completa de filas de tabla ya formateadas (`table_rows_all`) solo si `table_total > 40`
  (si no, nada que cargar).
- Píldoras (HTML, mismo patrón en Promedios y en la calculadora de la tarea 4): `<fieldset class="pills"
  id="series-parties">` con `<label class="pill" style="--pill-color: {{ color }}" title="{{ fullname }}"><input
  type="checkbox" name="party" value="{{ name }}" checked> {{ name }}</label>`; CSS: la casilla visualmente
  oculta pero enfocable, la etiqueta como píldora con borde del color y fondo del color cuando `:checked`
  (texto blanco o negro según contraste simple: usar `color-mix` no; fijar texto blanco sobre el color y gris
  sobre fondo claro cuando no está marcada), `:focus-visible` con anillo. Sin `<legend>` visible (`legend`
  con clase `sr-only` para accesibilidad), sin botones Todos/Ninguno.
- Orden y disposición: tres líneas `Vista` (solo con bloques), `Ventana`, píldoras (`#series-parties` en la
  vista Partidos, `#series-blocks` en Bloques), cada una en su fila (`.series-controls` en columna).
- `renderSeries` gana `legend = true` (opción); Promedios pasa `legend: false` en los dos gráficos; las píldoras
  sincronizan `selected` como hacían las casillas (el código de legendselectchanged se retira).
- Tooltip: `tooltip.trigger = 'item'`. Punto de sondeo → recuadro: `<strong>{name}</strong>` (nombre del
  sondeo), `{fecha} · muestra {sample_size}` (o "muestra –"), y una línea por partido con valor, en orden
  descendente, con marcador de color y sigla (`{sigla} {valor}`); en la vista Bloques, los bloques. Punto de la
  línea media → `{sigla}: {media} ({lo}–{hi}) · {fecha}`; proyección → igual con "Proyección". Las bandas
  siguen `silent` sin tooltip. Lógica pura del recuadro en `series.js` como `pollTooltipLines(poll, parties,
  colors)` exportada y testeada en node.
- Tabla: cabeceras Fecha, Sondeo, Muestra, partidos; `polls-table.js`: `nextRows(all, shown, step = 40)` →
  filas a añadir; botón `<button type="button" id="polls-more" hidden>Cargar más</button>` que el JS muestra
  cuando `table_total > filas pintadas`, añade 40 por clic con `textContent`, y oculta cuando no quedan; el
  pie "Últimos 40 de N" se actualiza a "{mostrados} de N"; sin JS, 40 filas + CSV como hoy.

- [ ] **Step 1: Tests de node que fallan** (`tests/test_js_polls.py`, patrón de `test_js_range.py`):

```python
def test_poll_tooltip_lines_sorts_parties_and_tolerates_missing_fields():
    poll = {'date': '2026-10-01', 'name': 'GAD3 / ABC', 'sample_size': None, 'PP': 31.2, 'PSOE': 28.9, 'VOX': 12.0}
    lines = run_js('s.pollTooltipLines(poll, ["PSOE", "PP", "VOX", "ERC"], {PP: "#1", PSOE: "#2", VOX: "#3"})')
    assert lines[0].startswith('<strong>GAD3 / ABC') and 'muestra –' in lines[1]
    assert [l for l in lines[2:]] and 'PP 31,2' in lines[2] and 'PSOE 28,9' in lines[3] and 'VOX 12,0' in lines[4] and len(lines) == 5

def test_next_rows_pages_by_forty_and_stops():
    assert run_js('p.nextRows(Array.from({length: 95}, (_, i) => i), 40)') == list(range(40, 80))
    assert run_js('p.nextRows(Array.from({length: 95}, (_, i) => i), 80)') == list(range(80, 95))
    assert run_js('p.nextRows(Array.from({length: 30}, (_, i) => i), 30)') == []
```

- [ ] **Step 2: Tests de página que fallan**

```python
def test_promedios_controls_in_three_rows_with_pills_and_the_polls_table_by_name(api):
    page = html(call(api, '/es')[2])
    assert page.index('id="series-view"') < page.index('id="series-window"') < page.index('id="series-parties"')
    assert 'class="pill"' in page and 'data-all' not in page and 'data-none' not in page
    assert '<th scope="col">Sondeo</th>' in page and '<th scope="col">Casa</th>' not in page
    assert 'id="polls-more"' in page
    data = initial(call(api, '/es')[2])
    assert data['polls']['polls'][0]['name'] and 'sample_size' in data['polls']['polls'][0]

def test_poll_name_follows_the_loader_rule():
    assert pages.poll_name({'pollster': 'GAD3', 'sponsor': 'ABC'}) == 'GAD3 / ABC'
    assert pages.poll_name({'pollster': 'GAD3', 'sponsor': None}) == 'GAD3'
```

- [ ] **Step 3: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_js_polls.py tests/test_pages.py -q -k "poll or pills"` → FAIL.
- [ ] **Step 4: Implementar** (ver "Interfaces"); medir `/es` con el paquete local como en la fase 7 (la isla crece con las filas de tabla: anotar el tamaño).
- [ ] **Step 5: Suite y node** — Run: `python -m pytest -m "not integration" -q && for f in web/dist/js/charts/series.js web/dist/js/pages/promedios.js web/dist/js/polls-table.js; do node --check $f; done` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "Promedios: party pills in three control rows, a tooltip per poll, the poll name column and load-more rows"`

---

### Task 4: Escaños: tabla, histogramas, un solo bloque, píldoras en la calculadora

**Files:**
- Modify: `web/templates/escanos.html`, `web/dist/js/charts/histogram.js`, `web/dist/js/pages/escanos.js`,
  `mtpy/lib/pages/escanos.py` (contexto sin `vs_rows` si ya nada lo usa)
- Test: `tests/test_pages.py`, `tests/test_js_histogram.py`

**Interfaces:**
- Tabla `#parties-table`: Partido, Escaños, Rango 95 %, P(escaño), P(primero), P(mayoría) (sin Mín–máx ni
  Media; `seat_rows` puede conservar los campos, la plantilla no los pinta).
- `renderHistogram` gana `bounds = false` y `markLabels = true`: con `bounds` y `lo`/`hi`, dos líneas sólidas
  finas (`lineStyle.width: 1`) en los bins de `lo` y `hi`; la mediana sólida gruesa (`width: 3`); con
  `markLabels: false` ninguna línea lleva etiqueta; la de mayoría sigue discontinua. Escaños pasa `bounds:
  true, markLabels: false` en los histogramas por partido y en la coalición; los valores por defecto dejan el
  histórico (tarea 6 de la fase 7) como está.
- Sección "Bloques": gráfico apilado + tabla `#blocks-table`; `#vs-table` desaparece de la plantilla.
- Calculadora: `<fieldset class="pills" id="coalition-parties">` con el mismo patrón de píldora de la tarea 3
  (`name="coalition"`, `--pill-color`, `title=fullname`), sin `<legend>` visible; `escanos.js` no cambia su
  lógica (lee las casillas).

- [ ] **Step 1: Tests que fallan**

```python
def test_escanos_table_histograms_blocks_and_pills(api):
    page = html(call(api, '/es/escanos')[2])
    assert '<th scope="col">Rango 95 %</th>' in page and 'Mín–máx' not in page and '<th scope="col">Media</th>' not in page
    assert 'id="vs-table"' not in page and 'id="blocks-table"' in page
    assert 'id="coalition-parties" class="pills"' in page.replace('class="pills" id="coalition-parties"', 'id="coalition-parties" class="pills"')
```

y en `tests/test_js_histogram.py`: con `bounds: true, lo: 120, hi: 150, median: 135, markLabels: false`, el
`markLine.data` tiene tres entradas (lo, mediana, hi) con anchuras 1, 3, 1 y sin `label.formatter`; sin opciones, la
salida es la de hoy.

- [ ] **Step 2: Ejecutar y ver que fallan** — Run: `python -m pytest tests/test_pages.py tests/test_js_histogram.py -q -k "escanos_table or histogram"` → FAIL.
- [ ] **Step 3: Implementar** (ver "Interfaces"); comprobar en el navegador local que las píldoras se marcan con teclado y que la calculadora recalcula.
- [ ] **Step 4: Suite y node** — Run: `python -m pytest -m "not integration" -q && node --check web/dist/js/charts/histogram.js web/dist/js/pages/escanos.js` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "Escaños: simpler seats table, median and range lines in the histograms, one block table and party pills"`

---

### Task 5: Diana del histórico, documentación y verificación

**Files:**
- Modify: `web/dist/css/site.css` (`#errors-bullseye { max-width: 640px; margin: 0 auto; }`),
  `web/templates/elecciones.html` (si la tarjeta necesita centrar el gráfico), `deploy/README.md` (etiquetas,
  píldoras, sin selector de modo, columna "Sondeo"), `docs/web/contrato.md` (solo si nombra el selector de
  modo), `docs/superpowers/specs/2026-10-13-web-fase-7-diseno-design.md` ("Ajustes tras el recorrido de Luis
  (fase 7b)": lista de cambios, decisiones, verificación)
- Test: verificación

- [ ] **Step 1: Diana** — regla CSS y comprobación de que `#bullseye-chart` (detalle y perfil) no cambia.
- [ ] **Step 2: Docs** — README y spec; `bash -n deploy/check-api.sh deploy/smoke.sh` (sin cambios de rutas; solo si alguna etiqueta aparece en los scripts).
- [ ] **Step 3: Verificación** — `python -m pytest -m "not integration" -q`; `node --check` de todo `web/dist/js`; `S3_BUCKET= DB_HOST= python -m uvicorn api:api --port 8001 &`, `bash deploy/check-api.sh http://127.0.0.1:8001`, `curl -s http://127.0.0.1:8001/es | wc -c`, recorrido en Chrome sin cabeza de `/`, `/es`, `/es/escanos`, `/elecciones`, `/metodo/*` a 1280 y 360 px (píldoras, tooltip por sondeo, "Cargar más", histogramas, tarjetas clicables), parar uvicorn.
- [ ] **Step 4: Commit** — `git commit -m "Document the phase 7b adjustments and enlarge the history bullseye"`
