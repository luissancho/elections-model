# «Los cambios de cada casa» — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La página del evento (`/{scope}`) gana, bajo «Promedio de sondeos», una tarjeta con una matriz casa × partido: cuánto cambia cada partido entre las dos últimas entregas de cada encuestadora con dos o más sondeos en 28 días, en voto y en escaños.

**Architecture:** `house_changes` (en `mtpy/lib/pages/sondeos.py`, junto a `previous_poll`) construye las dos vistas con los sondeos de la parte `polls` que `promedios_context` ya lee; un parcial Jinja pinta las dos tablas en el servidor. El navegador sólo alterna la vista y descarga la tarjeta como imagen, que se ensancha lo que desborde la tabla.

**Tech Stack:** Python 3.11 (Jinja2), pytest; ES modules probados con `node` desde `tests/test_js_*.py`; html2canvas vendorizado (`card-image.js`).

**Spec:** `docs/superpowers/specs/2026-11-11-web-cambios-por-casa-design.md` (maqueta: `.superpowers/brainstorm/97250-1794407404/content/matriz-v3.html`).

## Global Constraints

- Línea = misma casa (`pollster_id`, o `pollster` si el sondeo no tiene id) y mismo medio (`slugify(sponsor or '')`): la clave de `previous_poll`. La entrega anterior es `previous_poll(polls, última)`, sin distancia mínima ni máxima.
- Ventana: `HOUSE_CHANGES_DAYS = 28`; entra un sondeo si `date[:10] >= as_of − 28 días` (`as_of` = `meta['as_of']`). Una línea sale si tiene **dos o más** sondeos en la ventana.
- Cambio de voto `round(última − anterior, 1)`; de escaños `_seats_mid(última) − _seats_mid(anterior)`; escaños sólo los de `poll['seats']`.
- Celdas: sube o baja `m.badge(cambio, 1)` (voto) o `m.badge(cambio, 0)` (escaños), es decir «▲ 0,6» / «▼ 2» sin signo; cambio 0 → `<span class="dlt flat">=</span>`; sin cifra en la última → `<td class="na">—</td>`; con cifra en la última y no en la anterior → sólo la cifra. Bajo el cambio, `<small>{anterior} → {última}</small>`.
- El selector dice «% Voto» y «Escaños» (regla de Luis del 2026-11-05 para toda la página del evento; la spec lo llama «Voto / Escaños»).
- Textos públicos en español: «sólo» con tilde, «encuestadora / casa / medio», nunca «empresa» ni «patrocinador»; la `date` de un sondeo es su último día de campo, nunca «publicado el». Título, entradilla y «Cómo leerlo»: los de la spec, literales.
- Identificadores en inglés neutro; docstrings numpy; JSDoc en inglés con sangría de 2 espacios; PEP8, líneas ≤ 120.
- No se toca el modelo, el bundle ni su contrato; no se ejecuta ningún job ni se publica nada.
- `tests/test_pages.py` y `web/templates/metodo_escanos.html` tienen cambios de Luis sin commit: **no tocarlos**. Los tests nuevos de página van en `tests/test_pages_house_changes.py`. Trabajar en un worktree (rama `feat/cambios-por-casa` desde `dev`).
- Tests con el `pytest` de pyenv, nunca `python3 -m pytest`. Commits en español, por rutas explícitas, con el trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; `git status --short` y `git log -1 --oneline` antes de cada commit.

## Review Focus

1. Dos entregas de la misma línea el mismo día (una oleada corregida): la anterior es la previa en la parte, la fila dice «(mismo día)» y no «(0 días)». Tareas 1 y 2.
2. Un run sin `as_of` válido en `meta`: no hay tarjeta y la página se pinta igual. Tarea 1.
3. Una casa sin id, sin medio y sin rating: fila sin enlace al perfil, sin insignia y al final; nada revienta. Tareas 1 y 2.
4. Un partido con cifra en una sola de las dos entregas: raya (falta en la última) o cifra sin flecha (falta en la anterior), y fuera del recuento; nunca «=». Tareas 1 y 2.
5. Una tabla más ancha que la tarjeta (13 partidos, horquillas de escaños): en pantalla se desplaza con la casa fija; la imagen se ensancha y no corta columnas. Tarea 3.

---

### Task 1: `house_changes`, las dos vistas de la matriz

**Files:**
- Modify: `mtpy/lib/pages/sondeos.py` (junto a `previous_poll`), `mtpy/lib/pages/__init__.py` (re-exportar `HOUSE_CHANGES_DAYS` y `house_changes`)
- Create: `tests/test_pages_house_changes.py`

**Interfaces:**
- Consumes: `previous_poll`, `_seats_mid`, `seats_text`, `_has_value` (sondeos.py); `poll_href`, `fieldwork_label`, `RATING_COLORS` (promedios.py); `gradient`, `fmt_num`, `DASH` (filters.py); `poll_detail_href`, `slugify` (slugs.py).
- Produces: `HOUSE_CHANGES_DAYS = 28` y
  `house_changes(polls: list, parties: list, colors: dict, as_of: Optional[str], scope: Optional[str] = None, days: int = HOUSE_CHANGES_DAYS) -> Optional[dict]`.
  `None` sin filas o sin `as_of` ISO válido; si no, `{'days': int, 'pct': view, 'seats': view | None}` con
  `view = {'parties': [{'name', 'color'}], 'rows': [row], 'counts': {name: {'up', 'down', 'flat'}}}`,
  `row = {'pollster', 'href', 'rating', 'rating_color', 'sponsor', 'prev_date', 'last_date', 'gap_days', 'prev_href', 'last_href', 'cells'}` y
  `cells` alineada con `view['parties']`: `None` (la última entrega no da cifra) o `{'prev': str | None, 'value': str, 'change': number | None}`.

- [ ] **Step 1: Tests que fallan** (`tests/test_pages_house_changes.py`)

```python
"""La tarjeta «Los cambios de cada casa» de la página del evento: `pages.house_changes` y su matriz en `/{scope}`."""
import re

from mtpy.lib import bundle, pages, webapi
from mtpy.lib.pages.filters import DASH, gradient
from mtpy.lib.pages.slugs import poll_detail_href
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data

AS_OF = '2026-11-11'
NAMES = ['PP', 'PSOE', 'UP']
COLORS = {'PP': '#1d84ce', 'PSOE': '#ef1c27', 'UP': '#7b4977'}


def poll(date, pollster_id, pollster, sponsor, rating, seats=None, **values):
    """Registro de la parte `polls`: la casa, el medio, el rating, los escaños que publica y una cifra por partido."""
    record = {'date': date, 'pollster_id': pollster_id, 'pollster': pollster, 'sponsor': sponsor, 'rating': rating,
              **values}
    return record | ({'seats': seats} if seats else {})


SIGMA_SEATS = ({'PP': [133, 139], 'PSOE': [97, 103]}, {'PP': [131, 137], 'PSOE': [101, 106]})
SW_SEATS = ({'PP': [116, 116]}, {'PP': [113, 119], 'PSOE': [127, 132]})
POLLS = [
    poll('2026-10-13', 1, 'Sigma Dos', 'El Mundo', 65.9, PP=30.0, PSOE=26.0),  # before the window
    poll('2026-10-30', 1, 'Sigma Dos', 'El Mundo', 65.9, SIGMA_SEATS[0], PP=31.1, PSOE=24.6, UP=1.9),
    poll('2026-11-06', 1, 'Sigma Dos', 'El mundo', 65.9, SIGMA_SEATS[1], PP=31.0, PSOE=25.2, UP=1.6),
    poll('2026-10-14', 2, 'GAD3', 'Mediaset', 71.0, PP=31.4, PSOE=25.0),  # one delivery per outlet: no line with two
    poll('2026-11-05', 2, 'GAD3', 'ABC', 71.0, PP=31.9, PSOE=24.5),
    poll('2026-11-08', 3, 'SW Demoscopia', 'La Séptima', 25.5, SW_SEATS[0], PP=28.5, PSOE=31.1, UP=1.2),
    poll('2026-11-09', 3, 'SW Demoscopia', 'La Séptima', 25.5, SW_SEATS[1], PP=28.5, PSOE=31.2),
    poll('2026-10-28', None, 'Sondaxe', None, None, PP=29.2, PSOE=26.9),
    poll('2026-11-04', None, 'Sondaxe', None, None, PP=29.7, PSOE=27.3, UP=2.2),
    poll('2026-10-14', 5, 'IMOP', 'El Confidencial', 62.8, PP=28.0),  # the first day of the window
    poll('2026-11-06', 5, 'IMOP', 'El Confidencial', 62.8, PP=29.8),
    poll('2026-10-13', 6, '40dB', 'Prisa', 60.0, PP=30.0),  # the day before the window
    poll('2026-11-04', 6, '40dB', 'Prisa', 60.0, PP=30.5),
]


def test_house_changes_compare_the_last_two_deliveries_of_each_line_in_the_window():
    changes = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es')
    view = changes['pct']
    assert changes['days'] == 28 == pages.HOUSE_CHANGES_DAYS
    assert view['parties'] == [{'name': name, 'color': COLORS[name]} for name in NAMES]
    assert [row['pollster'] for row in view['rows']] == ['Sigma Dos', 'IMOP', 'SW Demoscopia', 'Sondaxe']
    sigma, imop, sw, sondaxe = view['rows']
    assert sigma['cells'] == [{'prev': '31,1', 'value': '31,0', 'change': -0.1},
                              {'prev': '24,6', 'value': '25,2', 'change': 0.6},
                              {'prev': '1,9', 'value': '1,6', 'change': -0.3}]
    assert (sigma['rating'], sigma['rating_color']) == ('65,9', gradient(65.9, 0, 100, pages.RATING_COLORS))
    assert (sigma['sponsor'], sigma['prev_date'], sigma['last_date']) == ('El mundo', '30 Oct', '6 Nov')
    assert sigma['gap_days'] == 7
    assert sigma['href'] == pages.poll_href(POLLS[2])
    assert sigma['prev_href'] == poll_detail_href('es', POLLS[1])
    assert sigma['last_href'] == poll_detail_href('es', POLLS[2])
    assert imop['cells'] == [{'prev': '28,0', 'value': '29,8', 'change': 1.8}, None, None]
    assert sw['cells'] == [{'prev': '28,5', 'value': '28,5', 'change': 0.0},
                           {'prev': '31,1', 'value': '31,2', 'change': 0.1}, None]  # UP: gone from the last delivery
    assert sw['gap_days'] == 1
    assert sondaxe['cells'][2] == {'prev': None, 'value': '2,2', 'change': None}  # UP: new in the last delivery
    assert (sondaxe['href'], sondaxe['rating'], sondaxe['rating_color'], sondaxe['sponsor']) == (None, DASH, None, None)
    assert view['counts'] == {'PP': {'up': 2, 'down': 1, 'flat': 1}, 'PSOE': {'up': 3, 'down': 0, 'flat': 0},
                              'UP': {'up': 0, 'down': 1, 'flat': 0}}


def test_house_changes_seats_are_the_houses_own_ranges_by_their_midpoints():
    view = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es')['seats']
    assert [party['name'] for party in view['parties']] == ['PP', 'PSOE']
    assert [row['pollster'] for row in view['rows']] == ['Sigma Dos', 'SW Demoscopia']
    assert view['rows'][0]['cells'] == [{'prev': f'133{DASH}139', 'value': f'131{DASH}137', 'change': -2},
                                        {'prev': f'97{DASH}103', 'value': f'101{DASH}106', 'change': 4}]
    assert view['rows'][1]['cells'] == [{'prev': '116', 'value': f'113{DASH}119', 'change': 0},
                                        {'prev': None, 'value': f'127{DASH}132', 'change': None}]
    assert view['counts'] == {'PP': {'up': 0, 'down': 1, 'flat': 1}, 'PSOE': {'up': 1, 'down': 0, 'flat': 0}}
    without = [{key: value for key, value in record.items() if key != 'seats'} for record in POLLS]
    assert pages.house_changes(without, NAMES, COLORS, AS_OF, 'es')['seats'] is None


def test_house_changes_without_two_deliveries_in_the_window_or_without_the_day_is_none():
    assert pages.house_changes([], NAMES, COLORS, AS_OF, 'es') is None
    assert pages.house_changes(POLLS[:2] + POLLS[3:5] + POLLS[11:], NAMES, COLORS, AS_OF, 'es') is None
    for day in (None, '', 'ayer'):
        assert pages.house_changes(POLLS, NAMES, COLORS, day, 'es') is None
    week = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es', days=7)
    assert [row['pollster'] for row in week['pct']['rows']] == ['SW Demoscopia'] and week['days'] == 7


def test_house_changes_same_day_deliveries_rating_ties_and_no_scope():
    polls = [poll('2026-11-05', 9, 'Zeta', 'M', 50.0, PP=30.0), poll('2026-11-05', 9, 'Zeta', 'M', 50.0, PP=30.4),
             poll('2026-11-01', 8, 'Alfa', None, 50.0, PP=29.0),
             poll('2026-11-03', 8, 'Alfa', None, 50.0, PP=29.0, PSOE=27.0)]
    view = pages.house_changes(polls, NAMES, COLORS, AS_OF)['pct']
    assert [row['pollster'] for row in view['rows']] == ['Alfa', 'Zeta']  # the same rating: by name
    zeta = view['rows'][1]
    assert zeta['gap_days'] == 0 and zeta['cells'][0] == {'prev': '30,0', 'value': '30,4', 'change': 0.4}
    assert (zeta['prev_href'], zeta['last_href']) == (None, None)
    assert [party['name'] for party in view['parties']] == ['PP', 'PSOE']  # UP has no figure in any row
    assert view['rows'][0]['cells'][1] == {'prev': None, 'value': '27,0', 'change': None}
    assert view['counts']['PSOE'] == {'up': 0, 'down': 0, 'flat': 0}
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_pages_house_changes.py -q` → FAIL (`module 'mtpy.lib.pages' has no attribute 'house_changes'`).

- [ ] **Step 3: Implementar `HOUSE_CHANGES_DAYS` y `house_changes` en `mtpy/lib/pages/sondeos.py`** y re-exportarlos en `mtpy/lib/pages/__init__.py`.
  - La última entrega de una línea es la de mayor `(date[:10], posición en la parte)`, el orden de `previous_poll`.
  - Filas por rating descendente; sin rating al final; empate por `pollster.casefold()` y luego por medio.
  - `rating` = `fmt_num(rating, 1)`; `rating_color` = `gradient(rating, 0, 100, RATING_COLORS)`, `None` si da `'transparent'`; los datos de la fila (rating, medio, enlaces) son los de la última entrega.
  - `prev_date` / `last_date` = `fieldwork_label(None, None, date)` («30 Oct»).
  - Vista de escaños: sólo las líneas cuya última entrega tiene alguna horquilla válida (`seats_text`) de `parties`; sin ninguna, `'seats': None`.
  - `view['parties']`: los de `parties`, en su orden, con alguna celda no `None` en las filas de la vista.

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_pages_house_changes.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add mtpy/lib/pages/sondeos.py mtpy/lib/pages/__init__.py tests/test_pages_house_changes.py` · «Cambios por casa: house_changes compara las dos últimas entregas de cada casa y medio en 28 días, en voto y en escaños».

---

### Task 2: La tarjeta en la página del evento

**Files:**
- Modify: `mtpy/lib/pages/promedios.py` (`promedios_context`, tras `context['table'] = …`), `web/templates/promedios.html` (tras el `{{ m.card_end() }}` de la tarjeta `series`, dentro de `<section id="promedio">`), `web/dist/css/site.css`
- Create: `web/templates/_house_changes.html`
- Test: `tests/test_pages_house_changes.py`

**Interfaces:**
- Consumes: `house_changes(...)` de la Tarea 1.
- Produces: `context['house_changes']` (`None` o el dict de la Tarea 1) y este marcado, del que dependen los tests y la Tarea 3:
  - tarjeta `m.card_start('house-changes', 'Los cambios de cada casa', <entradilla de la spec>, classes='shot')` seguida de `m.card_image_button(site_title|lower ~ '-cambios-' ~ state.scope ~ '-' ~ meta.run_at[:10] ~ '.png', 'Los cambios de cada casa · ' ~ event_bar.title ~ ' · ' ~ site_host ~ ' · ' ~ author_handle)`;
  - sólo con vista de escaños: `<fieldset id="changes-metric" class="seg" hidden>` con `<legend>Mostrar</legend>` y los radios `name="hcmetric"`, `value="pct"` (marcado, «% Voto») y `value="seats"` («Escaños»);
  - `<div class="hc-wrap" id="changes-pct" data-shot-fit>` y, con escaños, `<div class="hc-wrap" id="changes-seats" data-shot-fit hidden>`, cada uno con una `<table class="hc">`;
  - primera celda de cada fila: `<th scope="row">` con el nombre en `<b>` (enlazado a `row.href` si lo hay), la insignia `<span class="rt" style="background: …">` sólo con `rating_color`, y `<small>{medio} · <a href="{prev_href}">{prev_date}</a> → <a href="{last_href}">{last_date}</a> ({distancia})</small>` (sin «{medio} · » sin medio; fechas sin enlace sin `href`; distancia «mismo día», «1 día» o «{n} días»);
  - pie `<tfoot>` con `<th scope="row">Suben · bajan · igual</th>` y por partido `<span class="dlt up">▲ {up}</span> <span class="dlt down">▼ {down}</span> <span class="dlt flat">= {flat}</span>`, o `<td class="na">—</td>` si los tres son 0;
  - «Cómo leerlo» (`m.how()`) con los cuatro párrafos de la spec.

- [ ] **Step 1: Tests que fallan** (añadir a `tests/test_pages_house_changes.py`; el `meta` del fixture tiene `as_of` 2026-10-08 y el catálogo conoce PP y PSOE)

```python
PAGE_POLLS = [
    poll('2026-09-29', 7, 'Sigma Dos', 'El Mundo', 65.9, {'PP': [133, 139]}, PP=31.1),
    poll('2026-10-06', 7, 'Sigma Dos', 'El Mundo', 65.9, {'PP': [131, 137]}, PP=31.0, PSOE=25.2),
    poll('2026-10-06', 3, 'GAD3', 'Mediaset', 71.0, PP=31.4, PSOE=25.0),
    poll('2026-10-07', 3, 'GAD3', 'Mediaset', 71.0, PP=31.4),
    poll('2026-10-02', None, 'Sondaxe', None, None, PP=29.2, PSOE=26.9),
    poll('2026-10-02', None, 'Sondaxe', None, None, PP=29.7, PSOE=27.3),
]


def write_polls(fresh_app, polls, parties=('PP', 'PSOE')):
    """Sustituye los sondeos de la parte `polls` del run de pruebas."""
    part = fixture_data('polls') | {'parties': list(parties), 'polls': polls}
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', part, 'es', run_id=RUN)
    webapi.site().files.clear()


def changes_card(api, query=''):
    """La tarjeta de `/es` en una línea, y la página entera."""
    page = re.sub(r'\s+', ' ', html(call(api, '/es', query=query)[2]))
    start = page.index('<section class="card shot" id="house-changes"')
    return page[start:page.index('</section>', start)], page


def test_event_page_has_no_changes_card_without_a_line_with_two_deliveries(api):
    assert 'id="house-changes"' not in html(call(api, '/es')[2])


def test_event_page_changes_card_follows_the_average_with_both_views(api, fresh_app):
    write_polls(fresh_app, PAGE_POLLS)
    card, page = changes_card(api)
    assert page.index('id="series"') < page.index('id="house-changes"') < page.index('id="escanos"')
    assert '<h2>Los cambios de cada casa</h2>' in card
    assert 'class="card-img-btn" hidden data-file="demoscopio-cambios-es-2026-10-08.png"' in card
    assert '<fieldset id="changes-metric" class="seg" hidden>' in card
    radios = re.findall(r'<label><input type="radio" name="hcmetric" value="(\w+)"( checked)?> ([^<]+)</label>', card)
    assert radios == [('pct', ' checked', '% Voto'), ('seats', '', 'Escaños')]
    pct, seats = card.split('id="changes-seats" data-shot-fit hidden>')
    assert '<div class="hc-wrap" id="changes-pct" data-shot-fit>' in pct
    # by rating, the house without one last
    assert pct.index('<b>GAD3</b>') < pct.index('<b>Sigma Dos</b>') < pct.index('<b>Sondaxe</b>')
    assert re.search(r'<span class="rt" style="background: #[0-9a-f]{6}">65,9</span>', pct)
    assert ('El Mundo · <a href="/es/sondeos/2026-09-29/sigma-dos/el-mundo">29 Sep</a> → '
            '<a href="/es/sondeos/2026-10-06/sigma-dos/el-mundo">6 Oct</a> (7 días)') in pct
    assert '(1 día)' in pct and '(mismo día)' in pct
    sondaxe = pct[pct.index('<b>Sondaxe</b>'):]
    sondaxe = sondaxe[:sondaxe.index('</th>')]
    assert '<a href="/casas/' not in sondaxe and 'class="rt"' not in sondaxe  # no id, no rating
    assert re.search(r'<span class="dlt down">▼ 0,1</span>\s*<small>31,1 → 31,0</small>', pct)
    assert re.search(r'<span class="dlt flat">=</span>\s*<small>31,4 → 31,4</small>', pct)
    assert re.search(r'<td>\s*<small>25,2</small>\s*</td>', pct)  # Sigma Dos, PSOE: new in the last delivery
    assert '<td class="na">—</td>' in pct  # GAD3, PSOE: no figure in the last delivery
    foot = pct[pct.index('<tfoot>'):]
    assert 'Suben · bajan · igual' in foot
    count = '<span class="dlt up">▲ {}</span> <span class="dlt down">▼ {}</span> <span class="dlt flat">= {}</span>'
    assert foot.count(count.format(1, 1, 1)) == 1 and count.format(1, 0, 0) in foot  # PP, then PSOE
    assert '<b>GAD3</b>' not in seats and '<b>Sondaxe</b>' not in seats
    assert re.search(r'<span class="dlt down">▼ 2</span>\s*<small>133–139 → 131–137</small>', seats)
    for phrase in ('La comparación es siempre de una casa consigo misma', 'Mira el conjunto, no una celda',
                   'no un reparto nuestro', 'aquí sólo se ve hacia dónde lo mueve'):
        assert phrase in card, phrase
    assert_public_wording(page, '/es con los cambios de cada casa')
    assert 'id="house-changes"' in changes_card(api, 'mode=nowcast')[1]


def test_event_page_changes_card_without_seats_has_no_switch(api, fresh_app):
    write_polls(fresh_app, [{key: value for key, value in record.items() if key != 'seats'}
                            for record in PAGE_POLLS])
    card, _ = changes_card(api)
    assert 'id="changes-metric"' not in card and 'id="changes-seats"' not in card and 'id="changes-pct"' in card
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_pages_house_changes.py -q -k event_page` → FAIL (el primero pasa: aún no hay tarjeta).

- [ ] **Step 3: Contexto, parcial y estilos**
  - `promedios_context`: `context['house_changes'] = house_changes(poll_rows, context['table_parties'], colors, meta.get('as_of'), scope)`, con la importación local de `.sondeos` (ese módulo importa `promedios`).
  - `_house_changes.html`: una macro para la tabla de una vista (`view`, `digits`: 1 en voto, 0 en escaños) y la tarjeta con el marcado de **Interfaces**; `promedios.html` lo incluye con `{% if house_changes %}`. Cabecera de cada partido como en `_polls_table.html` (`<i class="pd">` con su color y `title` con el nombre completo del catálogo).
  - `site.css`: `.hc-wrap` con `overflow-x: auto`; `table.hc` con celdas centradas, `white-space: nowrap`, cifras tabulares y el `<small>` en línea aparte y apagado; primera columna alineada a la izquierda y `position: sticky; left: 0` con el fondo de la tarjeta; la regla de la insignia `.et .rt` y la del punto `.pd` pasan a valer también bajo `.hc`; bajo 720 px la primera columna se estrecha (el `<small>` puede partirse).

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_pages_house_changes.py tests/test_pages.py tests/test_pages_ejemplo.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add mtpy/lib/pages/promedios.py web/templates/promedios.html web/templates/_house_changes.html web/dist/css/site.css tests/test_pages_house_changes.py` · «Página del evento: tarjeta «Los cambios de cada casa» bajo el promedio, con las vistas de voto y de escaños».

---

### Task 3: El selector, la imagen entera y la comprobación en el navegador

**Files:**
- Create: `web/dist/js/pages/house-changes-wire.js`, `tests/test_js_house_changes.py`
- Modify: `web/dist/js/card-image.js` (`fitOverflow`, llamada desde `prepareClone`), `web/dist/js/pages/promedios.js` (importar y llamar), `web/dist/css/site.css` (la copia de la imagen), `tests/test_js_card_image.py`

**Interfaces:**
- Consumes: los ids `house-changes`, `changes-metric`, `changes-pct`, `changes-seats` y el atributo `data-shot-fit` de la Tarea 2; `wireCardImages(root, onError)` de `card-image.js`.
- Produces: `fitOverflow(card: HTMLElement) -> number` (exportada por `card-image.js`) y `wireHouseChanges(doc = document, onError = null) -> void`.

- [ ] **Step 1: Tests que fallan**

En `tests/test_js_card_image.py`:

```python
def test_fit_overflow_widens_the_card_by_what_its_marked_scrollers_hide():
    """Una tabla más ancha que la tarjeta (`data-shot-fit`) ensancha la copia lo que desborda; las ocultas (ancho 0) y
    las que caben no cuentan."""
    card = ('{style: {}, getBoundingClientRect: () => ({width: 1136}), querySelectorAll: (sel) => '
            'sel === "[data-shot-fit]" ? %s : []}')
    fit = '(() => { const card = %s; return [c.fitOverflow(card), card.style.width || null]; })()'
    wide = card % '[{scrollWidth: 1500, clientWidth: 1100}, {scrollWidth: 0, clientWidth: 0}]'
    assert run_js(fit % wide) == [400, '1536px']
    assert run_js(fit % (card % '[{scrollWidth: 900, clientWidth: 1100}]')) == [0, None]
```

`tests/test_js_house_changes.py` (nuevo; mismo arnés que `tests/test_js_card_image.py`, con `m` como módulo):

```python
"""El selector % Voto / Escaños de «Los cambios de cada casa» (`web/dist/js/pages/house-changes-wire.js`), comprobado
con `node` y un documento de mentira (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
MODULE = pathlib.Path(DIST, 'js', 'pages', 'house-changes-wire.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

FAKES = (
    'const el = (hidden) => ({hidden, listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }, '
    'querySelectorAll: () => []}); '
    'const nodes = {"house-changes": el(false), "changes-metric": el(true), "changes-pct": el(false), '
    '"changes-seats": el(true)}; '
    'const doc = {getElementById: (id) => nodes[id] || null}; '
    'const ids = ["changes-metric", "changes-pct", "changes-seats"]; '
    'const shown = () => ids.map((id) => (nodes[id] ? nodes[id].hidden : null)); '
    'const pick = (value) => nodes["changes-metric"].listeners.change({target: {value}}); '
)


def run_js(steps):
    """Ejecuta `steps` (sentencias que acaban en `return`) con `m` (el módulo) y el documento de `FAKES`; da su JSON."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const m = await import("{}"); console.log(JSON.stringify((() => {{ {} {} }})())); process.exit(0);'
    ).format(json.dumps(ECHARTS), MODULE, FAKES, steps)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True,
                         cwd=ROOT)

    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_switch_shows_one_table_at_a_time():
    steps = ('m.wireHouseChanges(doc); const start = shown(); pick("seats"); const seats = shown(); pick("pct"); '
             'return [start, seats, shown()];')
    assert run_js(steps) == [[False, False, True], [False, True, False], [False, False, True]]


def test_wiring_without_the_card_or_without_seats_does_nothing():
    assert run_js('m.wireHouseChanges({getElementById: () => null}); return shown();') == [True, False, True]
    steps = 'delete nodes["changes-metric"]; delete nodes["changes-seats"]; m.wireHouseChanges(doc); return shown();'
    assert run_js(steps) == [None, False, None]
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_js_card_image.py tests/test_js_house_changes.py -q` → FAIL.

- [ ] **Step 3: Implementar**
  - `fitOverflow(card)`: el mayor `scrollWidth − clientWidth` de los `[data-shot-fit]` de la tarjeta; si es positivo, `card.style.width = Math.ceil(ancho actual + exceso) + 'px'`; devuelve el exceso (0 si no hay). `prepareClone` la llama tras quitar `STRIPPED` y antes de redibujar los gráficos; html2canvas mide la copia después de `onclone`, así que la imagen sale con ese ancho.
  - `site.css`: en la copia, `.shot-image .hc-wrap { overflow: visible; }`.
  - `wireHouseChanges(doc, onError)`: sin `#house-changes`, nada; muestra `#changes-metric` si existe y, en su `change`, deja a la vista `#changes-pct` o `#changes-seats` según `event.target.value`; cablea el botón con `wireCardImages(card, onError)`.
  - `promedios.js`: `tryDraw(() => wireHouseChanges(document, showError), 'the changes of each house');` tras el promedio.

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_js_card_image.py tests/test_js_house_changes.py tests/test_js_event_page.py -q` → PASS.

- [ ] **Step 5: Comprobar en un navegador de verdad** (Chrome headless o a mano) la página `/es` servida en local, con datos reales (sólo lectura) o con un bundle de prueba de 13 partidos. Pasa si:
  - el selector alterna las dos tablas y, sin JavaScript, sólo se ve la de voto;
  - a 390 px de ancho la tabla se desplaza de lado y la columna de la casa queda fija y legible;
  - la imagen descargada (capturar el blob de `saveBlob`) contiene todos los partidos en las dos vistas, con la firma;
  - con el run del 2026-11-11 de `es`, las cifras de «Criterios de aceptación» de la spec coinciden; con otro run, tres celdas comprobadas a mano contra «Todas las encuestas».
  Anotar en el informe qué se comprobó y con qué datos.

- [ ] **Step 6: Suite completa y commit** — `pytest -q -x` → PASS · `git add web/dist/js/card-image.js web/dist/js/pages/house-changes-wire.js web/dist/js/pages/promedios.js web/dist/css/site.css tests/test_js_card_image.py tests/test_js_house_changes.py` · «Cambios por casa: selector % Voto / Escaños e imagen de la tarjeta con la tabla entera».
