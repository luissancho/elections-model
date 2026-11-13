# «Escaños que bailan» dentro de la página del evento — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** «Escaños que bailan» deja de ser una página aparte y pasa a ser una tarjeta de la página del evento (`/{scope}`), en una sección «Escaños» del índice, con el selector Provincias / Partidos: mapa y ficha lado a lado en Provincias y sólo las columnas en Partidos.

**Architecture:** Sólo web. La tarjeta se pinta en el servidor con las piezas ya hechas (funciones puras de `mtpy/lib/pages/circunscripciones.py` y `seat_charts.py`, macros de `web/templates/_seat_odds.html`), bajo una sola clave del contexto del evento (`seat_odds`) para no chocar con las suyas; el selector de vista, los filtros y el cambio de ficha siguen siendo radios y `:has()`. La página aparte se queda en dos rutas que responden 301, y se borra lo que deja de usarse (hemiciclo de opciones, ficha de partido, pestañas, direcciones por circunscripción).

**Tech Stack:** Python 3.11, Jinja2, `web/dist/css/site.css`, módulos ES de `web/dist/js`, pytest (arnés ASGI; `node` para los tests de JavaScript).

**Spec:** `docs/superpowers/specs/2026-11-13-escanos-que-bailan-en-el-evento-design.md` (aprobada por Luis el 2026-11-13). Lo que no cambia de la tarjeta viene de `docs/superpowers/specs/2026-11-13-escanos-que-bailan-mapas-design.md` (estados, signo, textos, «Retoques de Luis»). Boceto elegido: `mock-lado-1280.png`, en `scratchpad/retoques/shots/` de la sesión.

## Global Constraints

- Índice de la página del evento: Promedio · Resultados · Escaños · Evolución · Tabla, en ese orden. Anclajes: `#promedio`, `#resultados` (Hemiciclo, Resultados, Calculadora de mayorías), `#escanos` (la tarjeta «Escaños que bailan» y, debajo, «Resultados por territorio»), `#evolucion`, `#tabla`. `#territorio` y `seccion=territorio` dejan de existir. No se añade ningún apaño para los enlaces viejos de fuera.
- Estados y signo, sin cambios: seguro ≥ 99%, probable ≥ 75%, en juego ≥ 25%; nada por debajo del 25%; círculo sólido / más claro con borde continuo / borde discontinuo. Palabras retiradas de los textos de la tarjeta: «posible», «en el aire», «aspirantes», «reparto central», «Central», «reñidos», «disputados».
- Lado a lado desde 1.200 px de ventana (mapa hasta 760 px a la izquierda, ficha a la derecha, sin el resumen de cada fila); apilado por debajo; por debajo de 720 px, lo «estrecho» de siempre.
- La página del evento se sigue leyendo y manejando sin JavaScript: el selector de vista, los dos filtros y el cambio de ficha van con radios y `:has()`; sin `:has()` se ve todo. Nada de scripts en línea (CSP `default-src 'self'`); un bloque `<style>` generado está permitido.
- La tarjeta sólo se pinta si el run mostrado tiene la parte `seat-odds` en su modo, con simulaciones y dos circunscripciones o más (`published_seat_odds`).
- No se toca el modelo, la parte `seat-odds`, la API, `mtpy/lib/publish`, el Hemiciclo del reparto central, la Calculadora ni «Resultados por territorio» (salvo su enlace «Escaño a escaño en…», que se va, y su `seccion`).
- Web en `mtpy/controllers` y `mtpy/lib`; identificadores en inglés neutro; docstring numpy en cada función y JSDoc en cada función de JavaScript. Textos públicos en español, «sólo» con tilde, sin «empresa» ni «patrocinador», ningún «Cómo leerlo» abierto, porcentajes pegados.
- Trabajo en un worktree con su rama (`git worktree add .worktrees/escanos-evento -b escanos-evento dev`); nunca un fichero sin commit en el checkout principal; nunca `git stash`, `git commit -a` ni reescribir commits; antes de cada commit, `git status --short` y `git log -1 --oneline`. Tests con `~/.pyenv/versions/3.11.9/bin/pytest -p no:cacheprovider` (en adelante `pytest`). Commits en español con el trailer `Co-Authored-By` del modelo que los escribe. Nunca `.env`, S3, la base, `publish_*` ni jobs.

## Review Focus

1. **Runs a medias**: con la parte y sin tabla de territorio, con tabla y sin la parte (un run fijado anterior al 13-11), y sin ninguna de las dos: la sección «Escaños» y su entrada del índice salen sólo si hay algo que enseñar, y nunca queda un hueco. Tareas 1 y 2.
2. **Choques dentro de una página grande**: ningún `id` repetido en la página del evento y ningún `name` de los radios de la tarjeta usado por otro control; las claves del contexto de la tarjeta no pisan las del evento (`cards`, `parties`, `districts`, `totals` ya existen ahí). Tarea 2.
3. **Enlaces internos a los anclajes de antes**: la portada, el método, la calculadora (`#…&coalition=`), la tabla y el formulario del territorio, las rutas viejas de escaños y la barra de las subpáginas apuntan al anclaje que les toca; ninguno emite `territorio`. Tarea 1.
4. **El borde de los 1.200 px y la ficha estrecha**: a 1.199 apilado y a 1.200 lado a lado sin desbordes con Madrid y Barcelona elegidas; un ámbito sin posiciones de mapa (rejilla seguida) también lado a lado; la copia de la imagen, que se dibuja a 1.200 px. Tareas 2, 3 y 5.
5. **El script de la tarjeta dentro del de la página**: sin tarjeta no falla ni frena el resto del cableado del evento, y un fallo suyo no tumba los gráficos; tras volver atrás, la línea del filtro y el nombre de la imagen coinciden con los radios. Tarea 3.

## Decisiones del plan (para que Luis las vea)

1. La tarjeta se llama `seat-odds` por dentro y vive bajo una sola clave del contexto, `seat_odds`.
2. El módulo `mtpy/lib/pages/circunscripciones.py` pasa a llamarse `seat_odds.py` (ya no hay página con esa URL), y sus tests, `tests/test_pages_seat_odds.py`.
3. La bajada a la ficha al tocar una casilla sólo ocurre con puntero o dedo (no con el teclado), sólo en apilado y sólo si la ficha no está a la vista.
4. La imagen de Provincias se llama `demoscopio-escanos-{scope}-{circunscripción}[-{partido}][-{estado}].png`.

---

### Task 1: El índice y los anclajes de la página del evento

Sin tarjeta todavía: cambia el índice, el orden y cada referencia a los anclajes de antes.

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`EVENT_SECTIONS`, `subpage_event_bar`), `mtpy/lib/pages/promedios.py` (secciones que faltan), `mtpy/lib/pages/inicio.py` (`href_seats`), `mtpy/controllers/Redirect.py` (`seats_location`)
- Modify: `web/templates/promedios.html` (secciones, su orden, `seccion` y el `action` del formulario del territorio), `web/templates/metodo_escanos.html` (el enlace `/es#escanos` del cierre)
- Modify: `web/dist/js/seats.js` (`withSection(..., 'escanos')`), `web/dist/js/pages/seats-wire.js` (`seccion=territorio#territorio`), `web/dist/js/pages/promedios.js` (comentarios)
- Test: `tests/test_pages.py`, `tests/test_js_url_state.py`, `tests/test_js_territory.py`, `tests/test_pages_ejemplo.py`, `tests/test_web_routes.py`, `tests/test_pages_circunscripciones.py`

**Interfaces:**
- Produces: `EVENT_SECTIONS = (('promedio', 'Promedio'), ('resultados', 'Resultados'), ('escanos', 'Escaños'), ('evolucion', 'Evolución'), ('tabla', 'Tabla'))`.
- Produces: en `promedios.html`, `<section class="event-section" id="resultados">` (lo que hoy lleva `id="escanos"`) y, justo detrás y antes de `id="evolucion"`, `<section class="event-section" id="escanos">` con «Resultados por territorio» (lo que hoy lleva `id="territorio"`). La Tarea 2 mete la tarjeta al principio de esa sección.
- Produces: la sección `escanos` falta del índice cuando el run no tiene tabla de territorio (la Tarea 2 añade «ni tarjeta»).

- [ ] **Step 1: Tests que fallan** — en `tests/test_pages.py`, junto a `test_event_page_title_and_index`:

```python
def test_the_event_page_index_says_results_and_seats_and_no_anchor_of_before_is_left(events_site):
    status, _, body = call(events_site, '/es', query='mode=forecast')
    page = re.sub(r'\s+', ' ', html(body))
    index = page[page.index('class="seg lg event-index"'):]
    assert re.findall(r'<a href="#([a-z]+)"[^>]*>([^<]+)</a>', index[:index.index('</nav>')]) == [
        ('promedio', 'Promedio'), ('resultados', 'Resultados'), ('escanos', 'Escaños'), ('evolucion', 'Evolución'),
        ('tabla', 'Tabla')]
    assert re.findall(r'<section class="event-section" id="([a-z]+)"', page) == [
        'promedio', 'resultados', 'escanos', 'evolucion', 'tabla']
    results = page[page.index('id="resultados"'):page.index('<section class="event-section" id="escanos"')]
    assert all('id="{}"'.format(card) in results for card in ('hemicycle-card', 'seats', 'coalition'))
    seats = page[page.index('<section class="event-section" id="escanos"'):page.index('id="evolucion"')]
    assert 'id="districts"' in seats and 'seccion=escanos#escanos' in seats and 'action="/es#escanos"' in seats
    assert not [old for old in ('#territorio', 'id="territorio"', 'seccion=territorio') if old in page]
```

  (Hace falta un run con tabla de territorio: tomar la fixture y la preparación de `test_event_page_territory_table_has_hidden_controls_and_the_island_carries_the_districts`, que ya la tiene.) Y, en los tests que ya existen, los valores nuevos:
  - rutas viejas (`test_old_urls_redirect_permanently`): sin `region`, `seccion=resultados`; con `region`, `seccion=escanos`;
  - portada: el enlace a los escaños de un ámbito es `/{scope}#resultados`;
  - método: el enlace del cierre de `/metodo/escanos` es `/es#resultados`;
  - `tests/test_js_url_state.py` y `tests/test_js_territory.py`: la calculadora comparte `#resultados&coalition=…`, y el enlace de una circunscripción es `…&region={id}&seccion=escanos#escanos`;
  - barra de una subpágina (`subpage_event_bar`): enlaza `#resultados` y `#escanos`, y sin circunscripciones no enlaza `#escanos`.
  Añadir a `tests/test_web_routes.py` un test que recorra `web/templates`, `web/dist/js` y `mtpy` y falle si algún fichero contiene `#territorio`, `seccion=territorio` o `'territorio'` como identificador de sección.

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages.py tests/test_web_routes.py tests/test_js_url_state.py tests/test_js_territory.py -q`.

- [ ] **Step 3: El cambio** — según las interfaces; cada referencia interna pasa al anclaje que le toca: lo que hablaba del Hemiciclo, la tabla de resultados o la calculadora, a `resultados`; lo que hablaba del territorio, a `escanos`. Docstrings y comentarios al día.

- [ ] **Step 4: Verde** — `pytest tests/test_pages.py tests/test_pages_ejemplo.py tests/test_web_routes.py tests/test_pages_circunscripciones.py tests/test_js_url_state.py tests/test_js_territory.py -q`.

- [ ] **Step 5: Commit** — «Página del evento: el índice pasa a Promedio, Resultados, Escaños, Evolución y Tabla; los anclajes se renombran (#resultados, #escanos) y la sección del territorio sube detrás de Resultados».

---

### Task 2: La tarjeta «Escaños que bailan» en la página del evento

Pintada en el servidor y manejable sin JavaScript. La página aparte sigue funcionando hasta la Tarea 4.

**Files:**
- Modify: `mtpy/lib/pages/circunscripciones.py`, `mtpy/lib/pages/__init__.py`, `mtpy/lib/pages/escanos.py` (`seats_context`), `mtpy/lib/pages/promedios.py`
- Modify: `web/templates/_seat_odds.html`, `web/templates/promedios.html`, `web/templates/circunscripciones.html` (sólo para pasar parámetros a las macros), `web/templates/_macros.html` (fuera `seat_odds_link`), `web/dist/css/site.css`, `web/dist/js/pages/seats-wire.js` (fuera `#region-odds` y `odds_hrefs`)
- Test: `tests/test_pages_circunscripciones.py`, `tests/test_pages.py`, `tests/test_js_territory.py`

**Interfaces:**
- Consumes: `published_seat_odds`, `district_views`, `default_district`, `scope_totals`, `options_sentence`, `scope_options`, `drawn_token`, `drawn_row`, `district_sentence`, `districts_are_provinces`, `option_title`, `map_tiles`, `party_columns`, `column_ticks`, `district_positions`, `result_rows` (ya existen).
- Produces: `pages.seat_odds_card(scope: str, run: str, mode: str, meta: dict, parties: Catalog, summary: dict, sign: str) -> Optional[dict]`: `None` sin la parte (`published_seat_odds`). Si no, un dict con todo lo de la tarjeta y nada más: `n_sim`, `totals`, `foot`, `thresholds`, `state_words`, `district_word`, `view_label` («Provincias» o «Circunscripciones»), `parties` (los del filtro: `name`, `slug`, `fullname`, `color`), `grid` (`map_tiles`, opciones dibujadas con `party_slug`), `cards` (una por circunscripción, por nombre: `slug`, `name`, `seats`, sus recuentos por estado, `sentence`, `rows` dibujadas), `chosen` (el `slug` de `default_district`), `columns`, `ticks` y `rest` (como `_party_charts`, en el orden de `result_rows(summary)`, con el título de cada círculo nombrando su circunscripción), `image_base` (`'demoscopio-escanos-{scope}'`), `image_file` (`'{image_base}-{chosen}.png'`) e `image_sign`.
- Produces: el contexto del evento lleva `seat_odds` (ese dict o `None`) en lugar de `odds`; `initial_extra` pierde `odds_hrefs`; el CSV de `seat-odds` se sigue ofreciendo sólo si la parte está publicada.
- Produces: macros de `_seat_odds.html` con parámetros, sin leer el contexto: `legend(thresholds)`, `how(thresholds)`, `chips(item, state_words)`, `columns(charts, district_word, thresholds)`; y una nueva, `card(odds, stamp)`, con toda la tarjeta.
- Produces (marcado que usa la Tarea 3): `<section class="card shot so-card" id="seat-odds" data-file-base="{image_base}">` con su `card_image_button(image_file, image_sign)`; selector de vista `<fieldset class="seg so-filter so-views" id="so-view">`, radios `name="sv"`, `id="sv-map"` (marcado, `data-slug=""`) e `id="sv-cols"` (`data-slug="columnas"`); bloque `<div class="so-view-map">` con los filtros (`#map-party`, radios `mp`; `#map-state`, radios `ms`; con sus `data-slug` y `data-label` de hoy), la línea `#map-filter-line`, y `<div class="so-split">` con el mapa, la leyenda y el pie a un lado y `<div class="so-fichas" id="fichas">` al otro; casillas `<label class="so-tile">` con `<input type="radio" name="d" value="{slug}" id="d-{slug}" aria-label="…">`; fichas `<div class="so-ficha" id="ficha-{slug}">`; desplegable `<select id="district-select" hidden>` (sin formulario); bloque `<div class="so-view-cols">` con las columnas; el «Cómo leerlo», una vez, al pie.

- [ ] **Step 1: Tests que fallan** — en `tests/test_pages_circunscripciones.py`:

```python
def event_page(site, query='mode=forecast'):
    status, _, body = call(site, '/es', query=query)
    return status, re.sub(r'\s+', ' ', html(body))


def test_the_event_page_carries_the_card_of_the_seats_that_dance(odds_site, fresh_app):
    write_event_districts(fresh_app.fs)
    status, page = event_page(odds_site)
    section = page[page.index('<section class="event-section" id="escanos"'):page.index('id="evolucion"')]
    assert status == 200 and section.index('id="seat-odds"') < section.index('id="districts"')
    card = section[section.index('id="seat-odds"'):section.index('id="districts"')]
    assert 'Escaños que bailan' in card and 'según las 10.000 simulaciones del modelo.' in card
    assert re.findall(r'name="sv" value="(map|cols)"[^>]*?( checked)?>([^<]+)<', card) == [
        ('map', ' checked', 'Provincias'), ('cols', '', 'Partidos')]
    assert re.findall(r'name="d" value="([a-z-]+)"', card) == ['la-coruna', 'guadalajara', 'cadiz']
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', card) == ['la-coruna']  # most options in play
    assert re.findall(r'class="so-ficha" id="ficha-([a-z-]+)"', card) == ['cadiz', 'guadalajara', 'la-coruna']
    assert re.findall(r'name="mp" value="([a-z-]+)"', card) == ['all', 'pp', 'psoe', 'vox', 'aa', 'bng']
    assert re.findall(r'class="so-col" data-col="([^"]+)"', card) == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert 'Hay 22 opciones para 19 escaños' in card and card.count('<details class="how"') == 1
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1).' in card
    # the rules of the run, in the head: one per party for the filter, one per district for its card
    assert '#seat-odds:has(#mp-psoe:checked) .so-dots i:not(.p-psoe){opacity:.08}' in page
    assert '#seat-odds:has(#d-cadiz:checked) #ficha-cadiz{display:block}' in page
    # what did not come along, and what it replaces
    gone = ('so-hemi', 'so-pills', 'id="partidos"', '/circunscripciones', 'region-odds', 'odds_hrefs', 'reñid')
    assert not [piece for piece in gone if piece in page]
    text = visible_text(card)
    assert not [word for word in RETIRED + ('disputados',) if word in text]
    assert_public_wording(page, 'evento con la tarjeta')
    # nothing of the card collides with the rest of the page
    ids = re.findall(r'\sid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), [name for name in set(ids) if ids.count(name) > 1]
    for name in ('sv', 'mp', 'ms', 'd'):
        assert page.count('name="{}"'.format(name)) == card.count('name="{}"'.format(name)) > 0


def test_the_card_and_its_section_come_only_with_something_to_show(odds_site, api, fresh_app, tmp_path):
    # the part without a territory table: the section with the card alone, and its entry in the index
    status, page = event_page(odds_site)
    section = page[page.index('<section class="event-section" id="escanos"'):page.index('id="evolucion"')]
    assert 'id="seat-odds"' in section and 'id="districts"' not in section and '<a href="#escanos"' in page
    # a territory table without the part (a pinned run older than the part): the table alone, no rule of the card
    write_event_districts(fresh_app.fs)
    remove_odds(tmp_path)
    status, page = event_page(api)
    assert status == 200 and 'id="districts"' in page and 'id="seat-odds"' not in page and '#seat-odds:has' not in page
```

  (Si la página del evento ya tenía algún `id` repetido antes de esta tarea, la comprobación de los `id` se limita a los de la tarjeta y se anota en el informe. Si el bundle de prueba sin `write_event_districts` trae tabla de territorio, la primera mitad del segundo test se monta quitando la parte `districts`; lo que se fija es el comportamiento: con una de las dos cosas hay sección, con ninguna no la hay — esto último ya lo prueba `test_event_page_index_skips_missing_sections`, que se adapta.) Los tests que comprobaban los enlaces del evento a la página aparte (`…links_the_page_and_the_card_of_the_district_shown`, `…results_card_links_the_page_at_the_foot_of_its_tables`, `…links_of_the_event_page_keep_a_pinned_run`, `…link_of_a_district_without_card_is_hidden…`, `…do_not_compute_the_tokens`) se sustituyen por los de arriba; se conservan, adaptados al contexto nuevo, `…a_part_without_simulations_takes_down_no_page` y `…two_renders…_read_it_once`. Test de hojas de estilo (expresiones regulares sobre `site.css`, como los que hay): la vista (`#seat-odds:has(#sv-cols:checked) .so-view-map` y `#seat-odds:not(:has(#sv-cols:checked)) .so-view-cols`, `display: none`), el lado a lado (`@media (min-width: 1200px)` con `.so-split` en rejilla `minmax(0, 760px) minmax(0, 1fr)` y `.so-split .so-note { display: none }`), las tres reglas de estado con `#seat-odds`, `.so-ficha { display: none }` y la salida sin `:has()` (todas las fichas y las dos vistas a la vista).

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py -q -k "event_page or card_and_its_section"`.

- [ ] **Step 3: `seat_odds_card` y el contexto** — según las interfaces. `seats_context` ya tiene el `summary` del modo: el orden de las columnas sale de `result_rows(summary)` sin leer nada más. `seat_odds_context` (la página aparte) puede reutilizar `seat_odds_card` o quedarse como está: se borra en la Tarea 4.

- [ ] **Step 4: Macros, plantilla y estilos**
  - Macros con parámetros; `circunscripciones.html` les pasa los suyos y no cambia de aspecto.
  - `card(odds, stamp)`: título «Escaños que bailan»; entradilla «Qué escaños tiene seguros cada partido, cuáles son probables y cuáles están en juego, según las {n_sim} simulaciones del modelo.»; selector de vista; en la vista del mapa, la frase «Cada círculo es una opción de escaño: sólido si es seguro, más claro si es probable y con el borde discontinuo si está en juego. Toca una {provincia|circunscripción} para ver su detalle.» (la última, en `<span class="so-hint">`), los dos filtros, `stamp`, la línea del filtro, el mapa, la leyenda, el pie y las fichas (cabecera con nombre, escaños y pastillas; frase; filas; sin leyenda ni «Cómo leerlo» propios); en la de las columnas, su frase («Cada círculo es una opción de escaño en alguna {provincia|circunscripción}. Abajo, los seguros; encima, los probables y los que están en juego.»), las columnas, la leyenda y su pie; al final, `how(thresholds)`.
  - `promedios.html`: la sección `escanos` se pinta si hay `seat_odds` o `district_table`, con `s.card(seat_odds, stamp)` primero; el bloque `<style>` de las reglas del run, en el bloque `scripts` y sólo con tarjeta; fuera los dos `m.seat_odds_link` y el párrafo de `#region-odds`. La línea `stamp` es la de la página aparte (elección · estimación · fecha), `shot-only`.
  - `promedios.py`: la sección `escanos` falta del índice sólo si no hay ni tarjeta ni tabla.
  - CSS: las reglas de las interfaces y del test; dentro de la tarjeta las fichas no son tarjetas (sin borde, sombra ni relleno propios) y en lado a lado llevan una raya a su izquierda; el desplegable, sólo por debajo de 720 px y sólo cuando el script le quita `hidden`.

- [ ] **Step 5: Verde** — `pytest tests/test_pages_circunscripciones.py tests/test_pages.py tests/test_pages_seat_charts.py tests/test_js_territory.py tests/test_web_routes.py -q`.

- [ ] **Step 6: Commit** — «Página del evento: tarjeta «Escaños que bailan» en la sección Escaños, con el selector Provincias / Partidos, el mapa y la ficha lado a lado y las columnas de partidos».

---

### Task 3: El script y la imagen de la tarjeta

**Files:**
- Create: `web/dist/js/pages/seat-odds-wire.js`
- Modify: `web/dist/js/seat-odds.js`, `web/dist/js/pages/promedios.js`, `web/dist/css/site.css`
- Test: `tests/test_js_seat_odds.py`, `tests/test_pages_circunscripciones.py`, `tests/test_web_routes.py`

**Interfaces:**
- Consumes: el marcado de la Tarea 2; `wireCardImages` de `card-image.js`; `tryDraw` y `showError` de `pages/common.js`; `imageFile` y `filterLine` de `seat-odds.js`.
- Produces: `seat-odds.js` gana `outOfView(rect: {top: number, bottom: number}, height: number): boolean` (cierto si la caja no asoma nada en una ventana de esa altura). La Tarea 4 le quita `canonicalFor`.
- Produces: `seat-odds-wire.js` exporta `wireSeatOdds(): void`, que no hace nada sin `#seat-odds`; `promedios.js` la llama dentro de su propio `tryDraw`.

- [ ] **Step 1: Tests que fallan** — en `tests/test_js_seat_odds.py`:

```python
def test_a_card_is_out_of_view_only_when_none_of_it_shows():
    assert run_js("[s.outOfView({top: 900, bottom: 1300}, 800), s.outOfView({top: 700, bottom: 1300}, 800), "
                  "s.outOfView({top: -500, bottom: -10}, 800), s.outOfView({top: -500, bottom: 10}, 800)]") == [
        True, False, True, False]
    assert run_js("s.imageFile('demoscopio-escanos-es', ['madrid', 'psoe', 'en-juego'])") == (
        'demoscopio-escanos-es-madrid-psoe-en-juego.png')
```

  En `tests/test_pages_circunscripciones.py`: el botón de la tarjeta nace con `data-file="demoscopio-escanos-es-la-coruna.png"` y la sección con `data-file-base="demoscopio-escanos-es"`; `#map-filter-line` nace `hidden` y vacía; el `<select id="district-select" hidden>` lleva las tres circunscripciones por nombre con «Cádiz · 8 escaños» y la elegida marcada; y el test de contrato entre script y marcado pasa a leer `seat-odds-wire.js` contra la página del evento. Regla de la copia de la imagen en `site.css`: `.shot-image .so-filter { display: none }` (cubre el selector de vista y los dos filtros) y la casilla elegida sin borde oscuro, como hoy.

- [ ] **Step 2: Ejecutarlos y ver que fallan.**

- [ ] **Step 3: El script** — `seat-odds-wire.js`, cada cosa en su `tryDraw`:
  - al cambiar cualquier radio `sv`, `mp`, `ms` o `d`, y una vez al montar y en `pageshow` con `persisted`: la línea del filtro (`filterLine`, oculta si queda vacía o si la vista es la de las columnas) y el `data-file` del botón: en el mapa, `imageFile(base, [circunscripción, partido, estado])`; en las columnas, `imageFile(base, ['columnas'])`;
  - el desplegable: se le quita `hidden`; al cambiar, marca el radio de esa circunscripción y lanza su `change`; al cambiar un radio `d`, toma su valor;
  - la bajada a la ficha: en el `click` de una casilla con puntero (`event.detail > 0`), si la ventana mide menos de 1.200 px de ancho y la ficha elegida está fuera de la vista (`outOfView`), `scrollIntoView({block: 'nearest'})`, suave salvo con `prefers-reduced-motion`;
  - `wireCardImages` sobre la tarjeta.
  En la copia de la imagen el lado a lado tiene que salir (se dibuja a 1.200 px de ventana): comprobarlo y, si la media query no basta, forzarlo bajo `.shot-image`.

- [ ] **Step 4: Verde** — `pytest tests/test_js_seat_odds.py tests/test_js_card_image.py tests/test_pages_circunscripciones.py tests/test_web_routes.py -q`, y una mirada en Chrome sin ventana sobre el bundle de prueba: tocar una casilla cambia la ficha; el filtro escribe la línea y el nombre; el desplegable a 390 px elige casilla; la imagen sale con mapa y ficha lado a lado, sin selectores.

- [ ] **Step 5: Commit** — «Página del evento: script e imagen de «Escaños que bailan» (nombre e imagen según lo que está a la vista, desplegable del móvil y bajada a la ficha al tocar una casilla en apilado)».

---

### Task 4: Retirar la página aparte y lo que sobra

**Files:**
- Modify: `mtpy/controllers/Circunscripciones.py` (dos acciones que redirigen)
- Rename: `mtpy/lib/pages/circunscripciones.py` → `mtpy/lib/pages/seat_odds.py`; `tests/test_pages_circunscripciones.py` → `tests/test_pages_seat_odds.py` (con `git mv`)
- Delete: `web/templates/circunscripciones.html`, `web/dist/js/pages/circunscripciones.js`
- Modify: `mtpy/lib/pages/seat_odds.py`, `mtpy/lib/pages/seat_charts.py`, `mtpy/lib/pages/__init__.py`, `mtpy/lib/pages/escanos.py`, `web/templates/_seat_odds.html`, `web/dist/css/site.css`, `web/dist/js/seat-odds.js`, `web/dist/js/card-image.js`, `docs/web/contrato.md`
- Test: `tests/test_pages_seat_odds.py`, `tests/test_pages_seat_charts.py`, `tests/test_js_seat_odds.py`, `tests/test_js_card_image.py`, `tests/test_web_routes.py`, `tests/test_pages.py`

**Interfaces:**
- Produces: `GET /{scope}/circunscripciones` y `GET /{scope}/circunscripciones/{district}` responden 301 a `/{scope}{query}#escanos`, donde `query` conserva todo menos `partido`, `vista` y `circunscripcion`; un ámbito desconocido sigue dando 404.
- Produces: desaparecen de `pages` `seat_odds_context`, `seat_odds_links`, `district_href`, `party_view`, `party_sentence`, `hemicycle_layout` y lo que sólo usaban ellas; de `seat-odds.js`, `canonicalFor`; de `card-image.js`, `freeFluidSvgs` y su llamada; de `_seat_odds.html`, la macro `hemicycle`.

- [ ] **Step 1: Tests que fallan** — en el fichero de tests de la tarjeta:

```python
@pytest.mark.parametrize('path, query, location', [
    ('/es/circunscripciones', '', '/es#escanos'),
    ('/es/circunscripciones/cadiz', 'mode=nowcast', '/es?mode=nowcast#escanos'),
    ('/es/circunscripciones', 'partido=psoe&vista=partidos&circunscripcion=cadiz&mode=nowcast&run=' + RUN,
     '/es?mode=nowcast&run=' + RUN + '#escanos'),
    ('/es/circunscripciones', 'vista=renidos', '/es#escanos'),
])
def test_the_addresses_of_the_page_that_was_go_to_the_card(odds_site, path, query, location):
    status, headers, _ = call(odds_site, path, query=query)
    assert status == 301 and headers['location'] == location


def test_nothing_of_the_page_that_was_is_left():
    gone = ('seat_odds_context', 'seat_odds_links', 'district_href', 'party_view', 'party_sentence', 'hemicycle_layout')
    assert not [name for name in gone if hasattr(pages, name)]
    css = site_css()
    assert not [rule for rule in ('.so-hemi', '.so-tabs', '.so-pills', '.so-by-party', '.so-settled', '#vistas',
                                  '#partidos', '#grafico', '#mapa', '#provincias') if rule in css]
    root = os.path.join(os.path.dirname(__file__), '..')
    assert not os.path.exists(os.path.join(root, 'web', 'templates', 'circunscripciones.html'))
    assert not os.path.exists(os.path.join(root, 'web', 'dist', 'js', 'pages', 'circunscripciones.js'))
```

  `tests/test_js_card_image.py`: fuera los tests de `freeFluidSvgs`; `prepareClone` vuelve a no nombrarla. `tests/test_web_routes.py`: las dos rutas siguen resolviéndose al controlador.

- [ ] **Step 2: Ejecutarlos y ver que fallan.**

- [ ] **Step 3: La retirada** — el controlador se queda en las dos redirecciones (un ámbito desconocido, 404, por `pages.resolve_event_scope`); se borran la plantilla, el script, las funciones, las macros y las reglas de la lista, y los tests que sólo probaban lo retirado (los de la página aparte: vistas, canónica, títulos, 404 de circunscripción y partido, pestañas, ficha de partido, hemiciclo); se conservan, apuntando a la tarjeta, las comprobaciones que siguen teniendo sentido (estados y bordes exactos, frases, orden de las columnas, eje, «provincia» o «circunscripción», posiciones del catálogo, rejilla sin posiciones, pie sin más opciones que escaños, parte sin simulaciones, una sola lectura de la parte). El renombrado de los dos ficheros va en un commit propio, sin otros cambios, antes del resto.

- [ ] **Step 4: El contrato** — en `docs/web/contrato.md`, el párrafo de «Escaños que bailan» pasa a decir lo que hay: una tarjeta de la página del evento en `#escanos`, cuándo se pinta, que funciona sin JavaScript, las dos rutas viejas con su 301 y los anclajes nuevos de la página del evento.

- [ ] **Step 5: Toda la batería** — `pytest -q`. Esperado: todo verde, con los mismos tests saltados que en `dev` (81) y sin avisos nuevos.

- [ ] **Step 6: Commits** — «Escaños que bailan: el módulo y sus tests pasan a llamarse seat_odds» y «Escaños que bailan: fuera la página aparte (sus direcciones redirigen a la tarjeta del evento), el hemiciclo de las opciones y la ficha de partido».

---

### Task 5: Comprobación en el navegador, peso y cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-11-13-escanos-que-bailan-en-el-evento-design.md` (sección «Estado al cierre», al final); lo que la comprobación obligue a corregir, con su test.

- [ ] **Step 1: Datos reales** — un bundle temporal fuera del repositorio con el run publicado de `es` (partes `meta`, `summary`, `seat-odds`, `districts`, `scenario`, `dist` y las demás que pida la página del evento, leídas de la API pública de demoscopio.es: sólo GET, sin credenciales) y la página del evento servida en local o volcada a ficheros estáticos, como en las comprobaciones anteriores (`scratchpad/retoques/` de la sesión tiene el volcado, el servidor estático y el ayudante de Chrome). Para el ámbito sin posiciones, las ocho provincias andaluzas con `district_positions` devolviendo `{}`.

- [ ] **Step 2: Chrome sin ventana**, a 360, 390, 768, 1024, 1199, 1200, 1280 y 1440 px. Pasa si:
  - el índice dice Promedio · Resultados · Escaños · Evolución · Tabla y cada entrada baja a su sección;
  - a 1.200 px o más el mapa y la ficha van lado a lado sin desbordes con Madrid y con Barcelona elegidas, y a 1.199 o menos, apilados; el ámbito sin posiciones también;
  - el selector de vista, los dos filtros y el cambio de ficha funcionan sin recargar, con JavaScript y sin él; nada más de la página del evento deja de funcionar (el Hemiciclo, la calculadora, el detalle del territorio, los gráficos);
  - a 360 y 768 px, al tocar una casilla del norte la página baja hasta la ficha; con el teclado, no;
  - la imagen de Provincias sale con el mapa y la ficha lado a lado, sin selectores, con la línea del filtro y el nombre que toca; la de Partidos, con las columnas;
  - tras marcar un filtro, ir a otra página y volver, la línea y el nombre del fichero coinciden con los radios;
  - las direcciones viejas (`/es/circunscripciones`, `/es/circunscripciones/cadiz`) acaban en la tarjeta;
  - con un run sin la parte `seat-odds`, la página del evento carga sin la tarjeta y sin errores en la consola.

- [ ] **Step 3: Peso** — tamaño en gzip y número de elementos de la página del evento con el run real, antes (en `dev`) y después. Si sube más de 30 KB en gzip, parar y consultarlo.

- [ ] **Step 4: «Estado al cierre»** en la spec: lo construido, lo comprobado y cómo, lo que no se pudo comprobar, el peso, lo que difiere del diseño (incluidas las cuatro decisiones de este plan) y lo que queda para Luis. En español llano, sin etiquetas internas y sin afirmar nada que no se haya medido.

- [ ] **Step 5: Toda la batería y commit** — `pytest -q` en verde; «Escaños que bailan en la página del evento: comprobación en el navegador, peso y cierre de la spec».

## Después del plan (Luis)

Fusionar en `dev`, push e imagen nueva de **web**. No hay que republicar nada. Tras desplegar: abrir `/es#escanos` en un móvil, tocar una provincia y descargar la imagen de la tarjeta.
