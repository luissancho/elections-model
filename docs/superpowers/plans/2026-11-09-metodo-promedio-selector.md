# Selector de ámbito y partido en «Un cálculo completo» — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La tarjeta `#ejemplo` de `/metodo/promedio` deja elegir ámbito (generales + autonómicos con run en vivo) y
partido; el gráfico y la tabla se cargan por JS y las cifras del texto se reescriben al cambiar.

**Architecture:** El servidor pinta el texto con las cifras del líder nacional (marcadas con `data-w`) y manda en la
isla la lista de ámbitos con sus `href` fijados al run (del manifiesto, sin leer partes). Un módulo JS puro
(`weights-example.js`) carga `weights` + `meta` de un ámbito con `fetch` (patrón `loadDist` de `seats.js`) y construye
la vista, los textos y las filas de la tabla; `metodo.js` lo cablea al DOM.

**Tech Stack:** Python (Jinja2, `mtpy/lib/pages`), JS ES modules sin build, ECharts (`charts/weights.js`), pytest +
node para los tests de JS.

**Spec:** `docs/superpowers/specs/2026-11-09-metodo-promedio-selector-design.md`

## Global Constraints

- Sin cambios en la publicación (`mtpy/lib/publish/*`), en `bundle.py` ni en la API.
- Elecciones pasadas fuera; nada de estado en la URL.
- Sin JS: el texto se lee entero con las cifras del líder nacional; controles, gráfico y tabla quedan `hidden`.
- Los `href` de `weights`, `meta` y CSV van **fijados al run** con `api_url(..., run=run)`.
- Las cifras que escribe JS deben ser idénticas carácter a carácter a las de Jinja (`fmtPct`↔`pct`, `fmtNum`↔`num`,
  `fmtDate`↔`date`, `fmtDateShort`↔`date_short`, `fmtInt`↔`int`, guion `–` para ausentes).
- Redacción pública: «sólo» con tilde, «encuestadora/casa», nunca «empresa» (`assert_public_wording`).
- Identificadores en inglés; textos en español.
- `site.css` tiene cambios de Luis sin commit: no tocarlo salvo que sea imprescindible, y entonces sólo añadir.
- Luis hace commits en paralelo con `git commit -a`: antes de cada commit, `git status` y `git log -3`; añadir sólo los
  ficheros de la tarea; nunca reescribir sus commits. Los mensajes terminan con
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Desviación respecto a la spec, decidida en el plan: el subtítulo (`lead`) de la tarjeta pasa a genérico en vez de
  llevar `data-w`, porque `card_start` lo escapa.

## Review Focus

1. **Respuestas que llegan desordenadas**: el lector cambia de ámbito dos veces seguidas y la primera respuesta llega
   la última. Debe quedar pintado el último ámbito elegido. (Tarea 3: token de petición; comprobación en navegador.)
2. **Nombres de casa o patrocinador con `<`, `&` o comillas** en la tabla construida por JS: deben salir escapados,
   nunca como HTML. (Tarea 2: `test_table_rows_escape_names`.)
3. **Ámbito cuyo último run no tiene `weights`** (publicado antes de la parte) o con `parties` vacío: aviso en la
   línea de estado, se conserva el caso anterior y los selectores siguen vivos. (Tarea 2: `test_load_example_rejects…`;
   Tarea 3: comprobación.)
4. **Paridad de cifras servidor/JS**: al volver a «Generales» + líder, el texto debe quedar exactamente igual que el
   servido. (Tarea 2: `test_texts_match_the_served_page`.)
5. **Partido sin sondeo de peso positivo o sin `sponsor`/`sample_size`**: «Ver los N sondeos» y la tabla siguen
   coherentes; sin sondeo que más pesa, la frase desaparece. (Tarea 2: `test_top_and_texts_without_rows`.)

---

### Task 1: Servidor — texto marcado, isla ligera y lista de ámbitos

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (`weights_example` ~244-303, `metodo_section_context` ~759-840)
- Modify: `web/templates/metodo_promedio.html:715-808` (tarjeta `#ejemplo`)
- Test: `tests/test_pages_ejemplo.py`

**Interfaces:**
- Produces:
  - `weights_example() -> Optional[dict]`: igual que hoy **sin** `rows`, `line` ni `kernel` (sigue `name`,
    `fullname`, `color`, `date`, `published`, `run`, `stats`, `n_rows`, `top`).
  - `example_scopes() -> list[dict]` en `metodo.py`: una fila por ámbito `simulable` de `site().scopes()['scopes']`,
    `es` primero y el resto en el orden del catálogo, con
    `{'code', 'label', 'name', 'run', 'published', 'weights_href', 'meta_href', 'csv_href'}`; `label` es
    `'Generales'` para `es` y `name` para el resto; `published` = `run_at[:10]`;
    `weights_href = api_url(code, 'weights', run=run)`, `meta_href = api_url(code, 'meta', run=run)`,
    `csv_href = api_url(code, 'weights', run=run, fmt='csv')`.
  - Isla: `initial['example'] = {'scope': 'es', 'party': example['name'], 'scopes': example_scopes()}`.
  - `context['example_csv']` pasa a ser el `csv_href` de `es` (fijado al run).
  - Marcado de la plantilla que lee la tarea 3 (ids y atributos exactos):
    - `<span data-w="KEY">` con `KEY` ∈ `published`, `subject`, `date`, `mean`, `lo`, `hi`, `nobs`, `bandwidth`,
      `neff` (aparece dos veces), `n_rows`; la frase del sondeo que más pesa entera en `<span data-w="top">`.
    - Párrafo «Las fechas describen…» con `id="weights-dates"`, siempre presente, `hidden` cuando
      `example.date <= example.published`.
    - `<div class="series-controls" id="weights-controls">` contiene `<form id="weights-pick" class="filters-bar" hidden>`
      con `<label>Ámbito <select name="scope">` (una `<option value="{code}">{label}</option>` por ámbito, `es`
      seleccionado) y `<label>Partido <select name="party">` (sin opciones: las pone JS), y el `#weights-field` de hoy.
    - `#weights-chart` y `#weights-list` llevan `hidden`; `#weights-table` con su `<thead>` actual y `<tbody></tbody>`
      vacío.
    - Enlace CSV con `id="weights-csv"`.
    - El `lead` de la tarjeta pasa a genérico: «Los sondeos que pesan en el promedio de un partido en la última
      actualización y la recta que se ajusta a ellos.» (decisión del plan: el `lead` es texto escapado y no admite
      `data-w`).
  - `subject` en el servidor es `example.name` (el caso por defecto es siempre `es`).

- [ ] **Step 1: Reescribir los tests de `tests/test_pages_ejemplo.py`**

  - `test_weights_example_picks_the_leader_and_its_slice`: quitar las aserciones de `rows`, `line` y `kernel`; añadir
    `assert not {'rows', 'line', 'kernel'} & set(example)`.
  - Nuevo `test_example_scopes_lists_the_live_scopes_pinned_to_their_run(api, monkeypatch)`: parchear
    `type(metodo.site()).scopes` para devolver `{'contract': 1, 'scopes': [...]}` con `es-an` (simulable,
    `latest='20261101-090000'`, `run_at='2026-11-01T09:00:00'`), `es-xx` (no simulable) y `es` (simulable,
    `latest=RUN`), en ese orden; esperar
    `[row['code'] for row in rows] == ['es', 'es-an']`, `rows[0]['label'] == 'Generales'`,
    `rows[1]['label'] == rows[1]['name']`, `rows[1]['published'] == '2026-11-01'`,
    `rows[1]['weights_href'] == '/api/v1/forecast/es-an/weights?run=20261101-090000'`,
    `rows[1]['meta_href'] == '/api/v1/forecast/es-an?run=20261101-090000'`,
    `rows[1]['csv_href'] == '/api/v1/forecast/es-an/weights?run=20261101-090000&format=csv'`.
  - `test_promedio_context_carries_the_example_and_its_island`: `set(island) == {'scope', 'party', 'scopes'}`,
    `island['scope'] == 'es'`, `island['party'] == 'PP'`, `island['scopes'][0]['code'] == 'es'`,
    `context['example_csv'] == '/api/v1/forecast/es/weights?run={}&format=csv'.format(RUN)`.
  - `test_promedio_page_shows_the_example_block`: mantener las frases del texto (todas las de hoy siguen siendo
    válidas) y cambiar lo de la tabla y la isla:
    `'<tbody></tbody>' in table` (sin filas), `'id="weights-chart" ' in page` con `hidden`,
    `'<details class="weights-list" id="weights-list" hidden>' in page`, `'<form id="weights-pick" class="filters-bar" hidden>' in card`,
    `'<option value="es" selected>Generales</option>' in card`, `'<select name="party"></select>' in card`,
    `'data-w="mean">33,1%</span>' in page`, `page.count('data-w="neff"') == 2`,
    `'<span data-w="top"> El sondeo que más pesa es el de Sigma Dos del 1 oct, con 1,14.</span>' in re.sub(r'\s+', ' ', page)`,
    `'id="weights-dates" hidden' in page`, `'id="weights-csv"' in page`,
    `'Los sondeos que pesan en el promedio de un partido en la última actualización' in page`,
    `set(initial(body)['example']) == {'scope', 'party', 'scopes'}`, y `assert_public_wording(page)` si el test vecino
    ya lo usa en la página.

- [ ] **Step 2: Ejecutar y ver que fallan**

  Run: `pytest tests/test_pages_ejemplo.py -q`
  Expected: FAIL en los cuatro tests tocados (`example_scopes` no existe, isla con filas, tabla con filas).

- [ ] **Step 3: Implementar en `metodo.py` y en la plantilla**

  `example_scopes()` con docstring numpy como las vecinas; no lee partes, sólo `site().scopes()`. Si `es` no está
  simulable la lista empieza por los demás (no pasa en la práctica: sin run nacional no hay tarjeta). Actualizar el
  docstring de `metodo_section_context` (isla `example`). Plantilla según el marcado de **Interfaces**; las filas del
  `{% for row in example.rows %}` desaparecen.

- [ ] **Step 4: Ejecutar los tests de páginas**

  Run: `pytest tests/test_pages_ejemplo.py tests/test_pages.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git status --short && git log --oneline -3
  git add mtpy/lib/pages/metodo.py web/templates/metodo_promedio.html tests/test_pages_ejemplo.py
  git commit -m "Ejemplo del promedio: texto marcado, isla sin filas y ámbitos con run en vivo"
  ```

### Task 2: Módulo JS puro `weights-example.js`

**Files:**
- Create: `web/dist/js/weights-example.js`
- Test: `tests/test_js_weights_example.py` (harness `run_js` como `tests/test_js_seats.py`, saltado sin `node`; datos
  de `tests/fixtures/bundle/weights.json` y `meta.json`, el `data` del sobre)

**Interfaces:**
- Consumes: la fila de ámbito de la tarea 1 (`{code, label, name, run, published, weights_href, meta_href, csv_href}`);
  `fmtPct`, `fmtNum`, `fmtInt`, `fmtDate`, `fmtDateShort`, `DASH` de `format.js`.
- Produces (todas exportadas):
  - `async loadExample(scope, fetchImpl = fetch) -> Promise<{weights, meta}>`: pide `scope.weights_href` y
    `scope.meta_href` en paralelo con `Accept: application/json`; rechaza si alguna respuesta no es OK, si falta `data`
    o si `weights.parties` no es una lista no vacía.
  - `partyOptions(weights) -> string[]`: `weights.parties` ordenados por `stats[name].mean` de mayor a menor (los sin
    media al final, en su orden).
  - `defaultParty(weights) -> string|null`: el primero de `partyOptions`.
  - `exampleView(weights, meta, party) -> object|null`: `{name, color, date, stats, rows, line, kernel, n_rows, top}`
    con `rows` las de `party` ordenadas por fecha descendente, `color` de `meta.parties[].color`, `top` el sondeo de
    mayor `weight` (el más reciente en empate; `null` sin filas o con todos los pesos 0/nulos) como
    `{pollster, sponsor, date, weight}`; `null` si `party` no está en `weights.parties`. Es el objeto que
    `renderWeights(el, view, {field})` ya sabe dibujar.
  - `subjectOf(name, scope) -> string`: `name` si `scope.code === 'es'`, si no `` `${name} en ${scope.name}` ``.
  - `exampleTexts(view, scope) -> object`: cadenas para cada `data-w`: `published` (`fmtDate(scope.published)`),
    `subject`, `date` (`fmtDate(view.date)`), `mean`/`lo`/`hi` (`fmtPct`), `nobs` (`fmtInt`), `bandwidth`
    (`fmtNum(.., 0)`), `neff` (`fmtNum(.., 1)`), `n_rows` (`String`), `top` (`` ` El sondeo que más pesa es el de ${pollster} del ${fmtDateShort(date)}, con ${fmtNum(weight, 2)}.` `` o `''`),
    y `dates_hidden` (`view.date <= scope.published`).
  - `tableRowsHtml(rows) -> string`: las `<tr>` con el mismo marcado que tenía la plantilla
    (`<td class="f l">{fecha corta}</td>`, `<td><b>{casa}</b>{ / patrocinador}</td>`, entrevistas con `fmtInt`,
    publicada y corregida con `fmtPct`, peso con
    `<td class="cell-bar"><i class="fill w" style="left: 0; width: {w}%"></i><span>{fmtNum(weight, 2)}</span></td>`
    donde `w` replica `bar_width(weight, 0, 3)` de `pages/filters.py` (0-100, un decimal, 0 si falta), y los cuatro
    factores con `fmtNum(.., 2)`); todo texto de la fila escapado (`&`, `<`, `>`, `"`, `'`).

- [ ] **Step 1: Escribir los tests**

  - `test_party_options_and_default_order_by_mean`: con `weights` de prueba de tres partidos con medias 20, 33,1 y
    `null` → `['B', 'A', 'C']` y `defaultParty` `'B'`.
  - `test_example_view_slices_and_sorts_the_party`: sobre la fixture, `view.name == 'PP'`,
    `[r.date for r in view.rows] == ['2026-10-05', '2026-10-01', '2026-09-24']`, `view.color == '#1d84ce'`,
    `view.n_rows == 3`, `view.top == {'pollster': 'Sigma Dos', 'sponsor': <el de la fixture>, 'date': '2026-10-01', 'weight': 1.138}`,
    y `exampleView(weights, meta, 'ERC') is None`.
  - `test_texts_match_the_served_page`: `exampleTexts` de la fixture con `scope = {code: 'es', name: 'España', published: '2026-10-08'}` →
    `published == date == '8 oct 2026'`, `mean == '33,1%'`, `lo == '32,4%'`, `hi == '33,8%'`, `nobs == '3'`,
    `bandwidth == '42'`, `neff == '2,6'`, `subject == 'PP'`, `dates_hidden is True`,
    `top == ' El sondeo que más pesa es el de Sigma Dos del 1 oct, con 1,14.'` (las mismas cadenas que asserta
    `test_promedio_page_shows_the_example_block`).
  - `test_subject_names_the_scope_outside_the_generals`: `subjectOf('PSOE', {code: 'es-an', name: 'Andalucía'}) == 'PSOE en Andalucía'`.
  - `test_top_and_texts_without_rows`: partido con filas de peso 0 → `top is None` y `texts.top == ''`;
    `n_rows == '2'`.
  - `test_table_rows_match_the_old_markup`: sobre la fixture, `'<td class="f l">5 oct</td>' in html`,
    `'<td>32,4%</td>' in html and '<td>32,9%</td>' in html`, `'<b>GAD3</b> / ABC' in html`, `'width: 17.1%' in html`,
    `html.count('class="cell-bar"') == 3`, `'<td>0,64</td>' in html`, y `'>–<' in html` para una entrevista ausente.
  - `test_table_rows_escape_names`: fila con `pollster: 'A<b>&"'` → `'A&lt;b&gt;&amp;&quot;' in html` y
    `'<b>A<b>' not in html`.
  - `test_load_example_rejects_missing_parts`: `fetchImpl` de prueba (función async que devuelve
    `{ok, status, json}` según la URL): 404 en `weights` → rechaza con mensaje que contiene `HTTP 404`;
    `weights` con `parties: []` → rechaza; ambos OK → resuelve con `{weights, meta}` iguales a los `data`.

- [ ] **Step 2: Ejecutar y ver que fallan**

  Run: `pytest tests/test_js_weights_example.py -q`
  Expected: FAIL (el módulo no existe).

- [ ] **Step 3: Implementar `web/dist/js/weights-example.js`**

  Cabecera de comentario como los demás módulos; JSDoc en cada export. Sin DOM: sólo datos y cadenas.

- [ ] **Step 4: Ejecutar los tests**

  Run: `pytest tests/test_js_weights_example.py tests/test_js_format.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git status --short && git log --oneline -3
  git add web/dist/js/weights-example.js tests/test_js_weights_example.py
  git commit -m "Ejemplo del promedio: módulo de carga, vista, textos y tabla por ámbito y partido"
  ```

### Task 3: Cableado en `metodo.js` y comprobación en navegador

**Files:**
- Modify: `web/dist/js/pages/metodo.js` (`wireWeightsField` ~47-66, `paint` ~150-160)
- Test: comprobación en la app local (sin test de DOM: el proyecto no tiene entorno de DOM para node)

**Interfaces:**
- Consumes: el marcado de la tarea 1 y todos los exports de la tarea 2; `renderWeights` de `charts/weights.js`;
  `showError`, `tryDraw`, `readInitial` de `pages/common.js`.
- Produces: `wireWeightsExample(initial)` en `metodo.js` (exportada, como `wireWeightsField`), llamada desde `paint()`
  cuando existen `#weights-chart` e `initial.example`.

- [ ] **Step 1: Implementar `wireWeightsExample(initial)`**

  - Caché en memoria `Map` por `code` de las promesas de `loadExample`.
  - Estado: `{scope, party, field}`; `field` empieza en `'pct'` y lo mantiene `#weights-field` (sustituir
    `wireWeightsField(el, example)` por la lectura del estado actual; el conmutador redibuja la vista vigente).
  - Token de petición: cada cambio de ámbito incrementa un contador; al resolver, si el contador cambió, se descarta.
  - Al cargar un ámbito: rellenar `select[name=party]` con `partyOptions` (seleccionado `initial.example.party` en la
    primera carga de `es`, si no `defaultParty`); pintar.
  - Pintar: `exampleView` → `renderWeights(el, view, {field})` dentro de `tryDraw`; cada `[data-w=KEY]` recibe
    `textContent = texts[KEY]`; `#weights-dates.hidden = texts.dates_hidden`; `#weights-table tbody.innerHTML =
    tableRowsHtml(view.rows)`; `summary` de `#weights-list` → `Ver los ${n_rows} sondeos`; `aria-label` del gráfico
    con `subject` y `date`; `#weights-csv.href = scope.csv_href`.
  - Revelar `#weights-pick`, `#weights-field`, `#weights-chart` y `#weights-list` tras la primera pintura correcta.
  - Error de carga: `showError('No se pudieron cargar los sondeos de este ámbito.')`, devolver el `select[name=scope]`
    al ámbito pintado y no tocar nada más.

- [ ] **Step 2: Tests existentes**

  Run: `pytest tests/test_pages_ejemplo.py tests/test_js_weights_example.py tests/test_js_charts.py -q`
  Expected: PASS.

- [ ] **Step 3: Comprobar en la app local** (skill `run`; la web local usa el bundle configurado en `.env`)

  En `/metodo/promedio#ejemplo`:
  - Carga: el gráfico aparece con el líder nacional; el texto no cambia respecto al servido.
  - Partido: elegir otro partido de las generales → gráfico, tabla, cifras, «Ver los N sondeos» y CSV cambian.
  - Ámbito: elegir una autonómica → lista de partidos nueva, texto «del {partido} en {ámbito}», fecha de su run.
  - Publicada/Corregida sigue tras cambiar de partido y de ámbito.
  - Volver a Generales + líder → texto idéntico al servido.
  - Dos cambios de ámbito muy seguidos (red lenta en DevTools) → queda el último.
  - Con JS desactivado → texto completo con cifras, sin controles, gráfico ni tabla.
  - Consola sin errores.

- [ ] **Step 4: Suite completa**

  Run: `pytest -q`
  Expected: PASS (salvo los saltos de siempre sin `node`/red).

- [ ] **Step 5: Commit**

  ```bash
  git status --short && git log --oneline -3
  git add web/dist/js/pages/metodo.js
  git commit -m "Ejemplo del promedio: selectores de ámbito y partido en /metodo/promedio"
  ```
