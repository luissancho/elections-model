# «Escaños que bailan» — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cada run publica, por circunscripción y partido, en cuántas simulaciones se consigue cada escaño (parte `seat-odds`), y la página `/{scope}/circunscripciones` lo enseña escaño a escaño: seguros (99% o más) y en el aire, por circunscripción, por partido y en una lista de los más reñidos.

**Architecture:** El simulador resume `results` en recuentos enteros (`Simulator.seat_reach`) y `export_seat_odds` los escribe como una parte más por modo. Un módulo de página nuevo (`mtpy/lib/pages/circunscripciones.py`) convierte la parte en fichas con funciones puras y un controlador pinta en el servidor las tres vistas con Jinja2; el navegador sólo envía el desplegable y descarga la imagen. La página del evento gana dos enlaces de entrada.

**Tech Stack:** Python 3.11 (numpy, pandas), Jinja2, CSS de `web/dist/css/site.css`, módulos ES de `web/dist/js`, pytest (arnés ASGI de `tests/test_api_asgi.py`).

**Spec:** `docs/superpowers/specs/2026-11-12-escanos-que-bailan-design.md`. Maqueta: `.superpowers/brainstorm/53417-1794493344/content/pagina-v3.html` del checkout principal (fuera de git; leer su CSS y su script, no los blobs base64).

## Global Constraints

- Seguro: probabilidad ≥ 99% (`SAFE_SHARE = 0.99`). En el aire: ≥ 1% y < 99% (`OPEN_FLOOR = 0.01`). Por debajo del 1% no se dibuja ni se cuenta. La probabilidad es `reach[k-1] / n_sim`, sin redondear antes de comparar. Los umbrales son constantes de la página, nunca de la parte.
- Escaños en el aire de una circunscripción: `max(escaños − seguros, 0)`.
- La etiqueta de una ficha en el aire es un porcentaje entero pegado («80%»), limitado a entre 1% y 98%. Ninguna ficha en el aire dice 99%, 100% ni 0%.
- La página sigue el modo y el run de la página del evento (`?mode=`, o `default_mode`; `?run=`). Sin selector de modo.
- La página y sus enlaces de entrada sólo existen si el run mostrado tiene la parte con dos circunscripciones o más.
- No se toca el modelo, el reparto central, la tabla «Resultados por territorio» ni su detalle, salvo los enlaces de la Tarea 5.
- La página se lee sin JavaScript; el JavaScript sólo envía el desplegable al cambiar y descarga la imagen. Nada de scripts en línea (CSP `default-src 'self'`); los estilos en línea están permitidos.
- Web y API en `mtpy/controllers` y `mtpy/lib`; nada de FastAPI ni scripts sueltos. El módulo de la página se llama como su URL (`circunscripciones`); el resto de identificadores, en inglés neutro. Docstrings numpy en todas las funciones.
- Textos públicos en español, «sólo» con tilde, sin «empresa» ni «patrocinador», ningún «Cómo leerlo» abierto (`assert_public_wording`). Porcentajes pegados, sin espacio. Todos los textos de este plan son provisionales: los revisa Luis.
- Luis y otras sesiones trabajan en paralelo en `dev` y lanzan jobs en el checkout principal: el trabajo va en un worktree con su rama (`superpowers:using-git-worktrees`); nunca se crea ni se deja un fichero sin commit en el checkout principal (un run publicado guarda `dirty = git status --porcelain`), nunca se revierte nada ajeno.
- Tests con el `pytest` de pyenv (Python 3.11), nunca `python3`, nunca con `out=None` ni escribiendo en `backtest/results`. Commits en español con el trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; `git status --short` y `git log -1 --oneline` antes de cada commit; sólo los ficheros de la tarea, nunca `git commit -a`.
- Fecha: `loreg_guard` rechaza publicar desde el 24-N. La Tarea 1 es la que tiene que estar desplegada antes; por eso va primero y se puede desplegar sola.

## Review Focus

1. Un run sin la parte (un run fijado anterior al despliegue): la página responde 404 con su mensaje, y la página del evento no pinta los enlaces ni la descarga CSV; Tareas 3 y 5.
2. Un ámbito de una sola circunscripción (es-md; el bundle de prueba): la parte existe con una región, la página responde 404 y no hay enlaces; Tareas 3 y 5.
3. Un partido al que el reparto central da un escaño que ninguna simulación da (`reach` vacío, `seats` 1), y uno con fichas y `seats` 0: la fila sale, sin fichas en el primer caso, y nada divide por cero; Tareas 1 y 2.
4. Los bordes de los umbrales: exactamente 99% es seguro; 98,99% está en el aire y se rotula «98%»; exactamente 1% se dibuja con «1%»; 0,99% no se dibuja; Tarea 2.
5. El modo: sin `mode` y a menos de 60 días la página lee la parte `forecast`; con `?mode=nowcast`, la otra; todos los enlaces internos conservan `mode` y `run`; Tareas 3 y 4.

---

### Task 1: La parte `seat-odds`

**Files:**
- Modify: `mtpy/lib/simulator.py` (método nuevo tras `unit_summary`)
- Modify: `mtpy/lib/bundle.py` (`MODE_PARTS`, `SCHEMAS`)
- Modify: `mtpy/lib/publish/forecast.py` (`export_seat_odds`, `MODE_EXPORTERS`), `mtpy/lib/publish/__init__.py` (exportarlo)
- Create: `tests/fixtures/bundle/seat-odds.json`
- Modify: `docs/web/contrato.md`
- Test: `tests/test_simulator_unit.py`, `tests/test_publish_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Produces: `Simulator.seat_reach(region: int | str) -> dict[str, list[int]]`: para cada partido de `params['names']`, en su orden, cuántas simulaciones le dan al menos `k` escaños en la región, para `k = 1..máximo`; lista vacía si nunca saca ninguno. `ValueError` sin simulaciones, como `unit_summary`.
- Produces: `publish.export_seat_odds(sim) -> tuple[dict, pandas.DataFrame]` (filas y CSV por región, partido y `k`, en ese orden). `data = {'n_sim': int, 'parties': list, 'regions': [{'id', 'name', 'seats'}], 'rows': [{'region_id', 'name', 'seats', 'reach'}]}`; `seats` es el entero de `sim.central_result()` y `reach` la lista de `seat_reach`. Regiones sin el total del ámbito, en el orden de `export_districts`. Sólo las filas con `reach` no vacío o `seats > 0`. CSV: `['region_id', 'region', 'name', 'seats', 'k', 'sims', 'p']`, una fila por `k`, `p = round(sims / n_sim, 4)`.
- Produces: la parte `seat-odds` en `MODE_PARTS` (detrás de `scenario`), con esquema `{'n_sim': int, 'parties': list, 'regions': list, 'rows': list}`.

- [ ] **Step 1: Tests que fallan** — en `tests/test_simulator_unit.py`:

```python
def test_seat_reach_counts_the_simulations_that_reach_each_seat():
    sim = synthetic_simulator()
    sim.results = np.array([  # 4 simulations x [total, Madrid, Barcelona] x [PP, PSOE, VOX]
        [[5, 4, 1], [3, 2, 1], [2, 2, 0]],
        [[6, 3, 1], [4, 2, 0], [2, 1, 1]],
        [[5, 5, 0], [3, 3, 0], [2, 2, 0]],
        [[5, 4, 1], [3, 2, 1], [2, 2, 0]],
    ])
    assert sim.seat_reach(28) == {'PP': [4, 4, 4, 1], 'PSOE': [4, 4, 1], 'VOX': [2]}
    assert sim.seat_reach('Barcelona') == {'PP': [4, 4], 'PSOE': [4, 3], 'VOX': [1]}
    assert all(isinstance(n, int) for counts in sim.seat_reach(28).values() for n in counts)
    assert sum(sum(counts) for counts in sim.seat_reach(28).values()) == 4 * 6  # every simulation fills Madrid
    sim.results[:, 1, 2] = 0
    assert sim.seat_reach(28)['VOX'] == []
    sim.results = None
    with pytest.raises(ValueError):
        sim.seat_reach(28)
```

  En `tests/test_publish_unit.py`:

```python
SEAT_ODDS_COLUMNS = ['region_id', 'region', 'name', 'seats', 'k', 'sims', 'p']


def central(sim, madrid, barcelona):
    """Fija el reparto central de la caché del simulador sintético."""
    total = [a + b for a, b in zip(madrid, barcelona)]
    sim.central_table = pd.DataFrame([total, madrid, barcelona], columns=NAMES,
                                     index=pd.Index(['es', 'Madrid', 'Barcelona'], name='region'))


def test_export_seat_odds_counts_by_district_and_party():
    sim = synthetic_simulator(n_sim=50)
    central(sim, [3, 2, 1], [2, 2, 0])
    data, frame = publish.export_seat_odds(sim)
    data = valid('seat-odds', data)
    assert data['n_sim'] == 50 and data['parties'] == NAMES
    assert data['regions'] == [{'id': 28, 'name': 'Madrid', 'seats': 6}, {'id': 8, 'name': 'Barcelona', 'seats': 4}]
    assert [(r['name'], r['seats']) for r in data['rows'] if r['region_id'] == 28] == [('PP', 3), ('PSOE', 2), ('VOX', 1)]
    for row in data['rows']:
        assert set(row) == {'region_id', 'name', 'seats', 'reach'} and isinstance(row['seats'], int)
        assert row['reach'] == sim.seat_reach(row['region_id'])[row['name']]
    assert sum(sum(r['reach']) for r in data['rows'] if r['region_id'] == 28) == 50 * 6
    assert list(frame.columns) == SEAT_ODDS_COLUMNS and len(frame) == sum(len(r['reach']) for r in data['rows'])
    first = frame.iloc[0]
    assert (first['region'], first['name'], first['k']) == ('Madrid', 'PP', 1) and first['p'] == round(first['sims'] / 50, 4)


def test_export_seat_odds_keeps_a_central_seat_no_simulation_gives_and_drops_the_seatless():
    sim = synthetic_simulator(n_sim=50)
    sim.results[:, 2, 2] = 0  # VOX never wins a seat in Barcelona
    central(sim, [3, 2, 1], [2, 2, 0])
    assert (8, 'VOX') not in {(r['region_id'], r['name']) for r in publish.export_seat_odds(sim)[0]['rows']}
    central(sim, [3, 2, 1], [2, 1, 1])
    rows = {(r['region_id'], r['name']): r for r in valid('seat-odds', publish.export_seat_odds(sim)[0])['rows']}
    assert rows[(8, 'VOX')] == {'region_id': 8, 'name': 'VOX', 'seats': 1, 'reach': []}


def test_export_seat_odds_is_empty_without_a_central_allocation():
    sim = synthetic_simulator(n_sim=20)
    sim.params['split'] = False  # the regression mode: `central_result()` is None
    data, frame = publish.export_seat_odds(sim)
    assert valid('seat-odds', data) == {'n_sim': 20, 'parties': NAMES, 'regions': [], 'rows': []}
    assert frame.empty and list(frame.columns) == SEAT_ODDS_COLUMNS


def test_seat_odds_is_a_mode_part_written_with_the_others():
    assert bundle.MODE_PARTS[-3:] == ('scenario', 'seat-odds', 'projection')
    assert [name for name, _ in publish.forecast.MODE_EXPORTERS] == list(bundle.MODE_PARTS)
```

  Y en `tests/test_pages.py` (la API sirve cualquier parte de `MODE_PARTS`; el test fija que ésta lo es):

```python
def test_api_serves_the_seat_odds_part_of_a_mode(api):
    status, _, body = call(api, '/api/v1/forecast/es/forecast/seat-odds')
    payload = json.loads(body)
    assert status == 200 and (payload['schema'], payload['mode']) == ('seat-odds@1', 'forecast')
    assert payload['data'] == fixture_data('seat-odds')
```

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_simulator_unit.py tests/test_publish_unit.py tests/test_pages.py -k "seat_reach or seat_odds" -q`. Esperado: fallan por `seat_reach` y `export_seat_odds` inexistentes y por la parte desconocida.

- [ ] **Step 3: Implementar** `Simulator.seat_reach`, `export_seat_odds` (y su entrada en `MODE_EXPORTERS`, en la posición de `MODE_PARTS`), `MODE_PARTS`, `SCHEMAS['seat-odds']` y la exportación en `mtpy/lib/publish/__init__.py`. El recuento de `k` es `int((seats >= k).sum())` sobre la columna de la región en `self.results`; sin comprobación del invariante de la suma (hay simulaciones que no reparten todos los escaños cuando nadie pasa la barrera).

- [ ] **Step 4: El bundle de prueba** — crear `tests/fixtures/bundle/seat-odds.json` con el sobre de `districts.json` (`"schema": "seat-odds@1"`) y `data`: `n_sim` 1000, `parties` `["PP"]`, `regions` `[{"id": 28, "name": "Madrid", "seats": 37}]` y una fila `{"region_id": 28, "name": "PP", "seats": 15, "reach": [1000 trece veces, 870, 540, 210, 30]}`. Una sola circunscripción, a propósito (Review Focus 2).

- [ ] **Step 5: Contrato** — en `docs/web/contrato.md`: `seat-odds` en las tres enumeraciones de ficheros por modo (el árbol del paquete, la nota de `mode` y la tabla de rutas de la API) y una sección `### seat-odds` tras `### scenario` con la fuente, el `data`, el significado de `reach`, por qué son recuentos, el CSV, el caso de una sola circunscripción y una línea «Enmienda 2026-11-12: los runs anteriores no llevan esta parte».

- [ ] **Step 6: Ver que pasa todo** — `pytest tests/test_simulator_unit.py tests/test_publish_unit.py tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_web_api.py tests/test_pages.py -q`. Esperado: todo verde. Si `test_csv_links_cover_the_five_parts_in_order` u otro test enumera las partes por modo, no se toca aquí: `pages.escanos.CSV_PARTS` cambia en la Tarea 5.

- [ ] **Step 7: Commit** — «Parte `seat-odds`: en cuántas simulaciones consigue cada partido cada escaño de cada circunscripción (recuentos enteros, por modo, con su CSV)».

### Task 2: De la parte a las fichas (funciones puras)

**Files:**
- Create: `mtpy/lib/pages/circunscripciones.py`
- Modify: `mtpy/lib/pages/__init__.py` (exportar lo de abajo)
- Test: `tests/test_pages_circunscripciones.py` (nuevo)

**Interfaces:**
- Consumes: el `data` de la parte `seat-odds` (Tarea 1); `slugify` de `pages/slugs.py`; `_sort_name` de `pages/escanos.py` (nombre sin acentos para ordenar).
- Produces, en `mtpy.lib.pages` (constantes `SAFE_SHARE = 0.99`, `OPEN_FLOOR = 0.01`, `CLOSEST_SHOWN = 40`):
  - `token_label(share: float) -> str`: porcentaje entero, redondeo de la mitad hacia arriba, limitado a 1–98, con «%» pegado.
  - `seat_tokens(name: str, reach: list, seats: int, n_sim: int) -> list`: una ficha por `k` con `share >= OPEN_FLOOR`, en orden: `{'k', 'share', 'safe': bool, 'central': bool (k <= seats), 'label': str | None (None en las seguras), 'alpha': float | None (round(0.08 + 0.42 * share, 3); None en las seguras), 'title': str}`. `title`: «2.º de PP: seguro» o «3.º de PP: lo consigue en el 80% de las simulaciones».
  - `district_views(data: dict) -> list`: una por región, en el orden de la parte: `{'id', 'name', 'slug', 'seats', 'safe', 'open', 'contenders', 'rows'}`; `open = max(seats - safe, 0)`, `contenders` = fichas en el aire; cada fila `{'name', 'seats', 'tokens', 'safe', 'open', 'expected'}` (`open` = sus fichas en el aire, `expected = sum(reach) / n_sim`), sólo los partidos con alguna ficha o `seats > 0`, por `seats` descendente y después `expected` descendente.
  - `district_order(districts: list) -> list`: más `open` primero; a igualdad, más `seats`, y después el nombre sin acentos.
  - `scope_totals(districts: list) -> dict`: `{'seats', 'safe', 'open', 'contenders', 'live', 'districts'}` (`live` = circunscripciones con `open > 0`).
  - `district_sentence(district: dict) -> str` (textos del Step 3).
  - `party_view(districts: list, name: str) -> dict`: `{'name', 'seats', 'safe', 'open', 'rows', 'settled'}`; `rows` = las circunscripciones donde tiene alguna ficha en el aire, `{'district': {'id', 'name', 'slug', 'seats'}, 'seats', 'tokens', 'safe', 'open'}`, ordenadas por la distancia al 50% de su ficha en el aire más cercana al 50%; `settled` = `[{'name', 'slug', 'safe'}]` donde sólo tiene seguras, por `safe` descendente.
  - `closest_seats(districts: list, limit: int = CLOSEST_SHOWN) -> list`: las fichas en el aire más cercanas al 50%: `{'district', 'slug', 'party', 'k', 'share', 'label', 'central'}`.

- [ ] **Step 1: Tests que fallan** — crear `tests/test_pages_circunscripciones.py`:

```python
"""La página «Escaños que bailan» (`/{scope}/circunscripciones`): de la parte `seat-odds` a las fichas y sus vistas."""
import pytest

from mtpy.lib import pages


def odds_row(region_id, name, seats, *reach):
    return {'region_id': region_id, 'name': name, 'seats': seats, 'reach': list(reach)}


ODDS = {
    'n_sim': 10000, 'parties': ['PP', 'PSOE', 'VOX', 'AA', 'BNG'],
    'regions': [{'id': 11, 'name': 'Cádiz', 'seats': 8}, {'id': 19, 'name': 'Guadalajara', 'seats': 3},
                {'id': 15, 'name': 'La Coruña', 'seats': 8}],
    'rows': [
        odds_row(11, 'PP', 3, 10000, 10000, 8010, 730, 10), odds_row(11, 'PSOE', 2, 10000, 9950, 4110, 80),
        odds_row(11, 'VOX', 2, 10000, 8530, 570), odds_row(11, 'AA', 1, 7580),
        odds_row(19, 'PP', 1, 10000, 60), odds_row(19, 'PSOE', 1, 9990), odds_row(19, 'VOX', 1, 9950),
        odds_row(15, 'PP', 3, 10000, 10000, 9900, 3000),  # the third, exactly at 99%: safe
        odds_row(15, 'PSOE', 2, 10000, 9899, 2400),       # the second, at 98,99%: open, labelled 98%
        odds_row(15, 'VOX', 1, 3657),                     # a central seat that comes out in 37% of the simulations
        odds_row(15, 'BNG', 2, 10000, 9950, 100, 99),     # the third, exactly at 1%: drawn; the fourth is not
    ],
}


def by_name(district):
    return {row['name']: row for row in district['rows']}


def test_tokens_follow_the_two_thresholds_to_the_exact_count():
    pp = pages.seat_tokens('PP', [10000, 10000, 8010, 730, 10], 3, 10000)
    assert [(t['k'], t['safe'], t['central'], t['label']) for t in pp] == [
        (1, True, True, None), (2, True, True, None), (3, False, True, '80%'), (4, False, False, '7%')]
    assert (pp[2]['alpha'], pp[0]['alpha']) == (round(0.08 + 0.42 * 0.801, 3), None)
    assert pp[1]['title'] == '2.º de PP: seguro'
    assert pp[2]['title'] == '3.º de PP: lo consigue en el 80% de las simulaciones'
    edges = pages.seat_tokens('X', [9900, 9899, 100, 99], 0, 10000)
    assert [(t['k'], t['safe'], t['label']) for t in edges] == [(1, True, None), (2, False, '98%'), (3, False, '1%')]
    assert pages.seat_tokens('X', [], 1, 10000) == []


@pytest.mark.parametrize('share, label', [(0.9899, '98%'), (0.985, '98%'), (0.805, '81%'), (0.5, '50%'),
                                           (0.0149, '1%'), (0.01, '1%')])
def test_token_label_is_a_whole_percentage_between_1_and_98(share, label):
    assert pages.token_label(share) == label


def test_district_views_count_the_safe_and_the_open_seats():
    cadiz, guadalajara, coruna = pages.district_views(ODDS)
    assert (cadiz['name'], cadiz['slug'], cadiz['seats']) == ('Cádiz', 'cadiz', 8) and coruna['slug'] == 'la-coruna'
    assert (cadiz['safe'], cadiz['open'], cadiz['contenders']) == (5, 3, 6)
    assert [row['name'] for row in cadiz['rows']] == ['PP', 'PSOE', 'VOX', 'AA']  # central seats, then expected seats
    assert [(row['safe'], row['open']) for row in cadiz['rows']] == [(2, 2), (2, 1), (1, 2), (0, 1)]
    assert by_name(cadiz)['PSOE']['expected'] == pytest.approx(2.414)
    assert (guadalajara['safe'], guadalajara['open'], guadalajara['contenders']) == (3, 0, 0)
    assert (coruna['safe'], coruna['open'], coruna['contenders']) == (6, 2, 5)
    vox = by_name(coruna)['VOX']['tokens']
    assert [(t['safe'], t['central'], t['label']) for t in vox] == [(False, True, '37%')]


def test_district_views_keep_a_party_with_a_central_seat_and_no_tokens_and_never_count_below_zero():
    data = {'n_sim': 100, 'parties': ['A', 'B', 'C'], 'regions': [{'id': 1, 'name': 'Uno', 'seats': 2}],
            'rows': [odds_row(1, 'A', 1, 100, 100, 100), odds_row(1, 'B', 1), odds_row(1, 'C', 0, 40)]}
    uno = pages.district_views(data)[0]
    assert [(row['name'], len(row['tokens'])) for row in uno['rows']] == [('A', 3), ('B', 0), ('C', 1)]
    assert (uno['safe'], uno['open'], uno['contenders']) == (3, 0, 1)  # three safe seats for two: never minus one


def test_district_order_totals_and_sentences():
    districts = pages.district_views(ODDS)
    assert [d['name'] for d in pages.district_order(districts)] == ['Cádiz', 'La Coruña', 'Guadalajara']
    tie = [{'name': 'Ávila', 'open': 1, 'seats': 3}, {'name': 'Zamora', 'open': 1, 'seats': 3},
           {'name': 'Soria', 'open': 1, 'seats': 2}, {'name': 'Madrid', 'open': 1, 'seats': 38}]
    assert [d['name'] for d in pages.district_order(tie)] == ['Madrid', 'Ávila', 'Zamora', 'Soria']
    assert pages.scope_totals(districts) == {'seats': 19, 'safe': 14, 'open': 5, 'contenders': 11, 'live': 2, 'districts': 3}
    cadiz, guadalajara, _ = districts
    assert pages.district_sentence(cadiz) == ('De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). '
                                              'Los otros 3 se los disputan 6 aspirantes.')
    assert pages.district_sentence(guadalajara) == 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).'


def test_party_view_lists_the_most_doubtful_districts_first():
    view = pages.party_view(pages.district_views(ODDS), 'PSOE')
    assert (view['name'], view['seats'], view['safe'], view['open']) == ('PSOE', 5, 4, 3)
    assert [row['district']['name'] for row in view['rows']] == ['Cádiz', 'La Coruña']
    assert view['rows'][0]['district'] == {'id': 11, 'name': 'Cádiz', 'slug': 'cadiz', 'seats': 8}
    assert view['settled'] == [{'name': 'Guadalajara', 'slug': 'guadalajara', 'safe': 1}]
    assert pages.party_view(pages.district_views(ODDS), 'UP') == {
        'name': 'UP', 'seats': 0, 'safe': 0, 'open': 0, 'rows': [], 'settled': []}


def test_closest_seats_are_the_open_ones_nearest_to_a_coin_toss():
    closest = pages.closest_seats(pages.district_views(ODDS))
    assert len(closest) == 11
    assert closest[0] == {'district': 'Cádiz', 'slug': 'cadiz', 'party': 'PSOE', 'k': 3, 'share': 0.411,
                          'label': '41%', 'central': False}
    assert [(c['district'], c['party'], c['k']) for c in closest[1:4]] == [
        ('La Coruña', 'VOX', 1), ('La Coruña', 'PP', 4), ('Cádiz', 'AA', 1)]
    assert (closest[-1]['party'], closest[-1]['label']) == ('BNG', '1%')
    assert len(pages.closest_seats(pages.district_views(ODDS), limit=4)) == 4
```

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py -q`. Esperado: `AttributeError` de `pages`.

- [ ] **Step 3: Implementar** el módulo y exportarlo. Lo que los tests no fijan:
  - Seguro si `count / n_sim >= SAFE_SHARE`; dibujado si `count / n_sim >= OPEN_FLOOR` (la división, sin redondear).
  - `district_sentence`, con la lista de seguros «SIGLAS n» en el orden de las filas:
    - todos seguros: «Los {n} escaños son seguros ({lista}).»; con un solo escaño, «El escaño es seguro ({lista}).»
    - algunos: «De los {n} escaños, {s} son seguros ({lista}). Los otros {o} se los disputan {c} aspirantes.»; con `s == 1`, «1 es seguro»; con `o == 1`, «El otro se lo disputan {c} aspirantes.»
    - ninguno: «Ninguno de los {n} escaños es seguro: se los disputan {c} aspirantes.»; con un solo escaño, «El escaño no es seguro: se lo disputan {c} aspirantes.»
    - `c == 1` se escribe «1 aspirante».

- [ ] **Step 4: Ver que pasan** — `pytest tests/test_pages_circunscripciones.py -q`. Esperado: todo verde.

- [ ] **Step 5: Commit** — «Escaños que bailan: de la parte `seat-odds` a las fichas (seguros al 99%, en el aire desde el 1%), con sus recuentos, órdenes y frases».

### Task 3: La página y la vista por circunscripción

**Files:**
- Create: `mtpy/controllers/Circunscripciones.py`, `web/templates/circunscripciones.html`, `web/templates/_seat_odds.html` (macros de las fichas y la leyenda), `web/dist/js/pages/circunscripciones.js`
- Modify: `mtpy/lib/pages/common.py` (`ROUTES`, `MESSAGES`), `mtpy/lib/pages/circunscripciones.py`, `mtpy/lib/pages/__init__.py`, `web/dist/css/site.css`
- Test: `tests/test_pages_circunscripciones.py`, `tests/test_web_routes.py`

**Interfaces:**
- Consumes: todo lo de la Tarea 2; `common_context(state, None, event=True)`, `bundle_part(site().run_file, scope, run, 'seat-odds', mode)`, el reenlace de `event_bar['sections']` de `sondeo_context` (`pages/sondeos.py`), `m.kpis`, `m.headline`, `m.card_image_button`, `m.how`, `wireForm` (`pages/common.js`), `wireCardImages` (`card-image.js`), `Page.redirect`.
- Produces: rutas `('/{scope}/circunscripciones', 'circunscripciones', 'index', ['GET'])` y `('/{scope}/circunscripciones/{district}', 'circunscripciones', 'detail', ['GET'])`, tras la de `/{scope}/escanos`.
- Produces: `pages.seat_odds_context(state: dict, district: Optional[str] = None) -> dict`: el contexto común del evento más `view` (`'district'`), `totals`, `kpis` (filas de `m.kpis`), `districts` (en `district_order`, cada una con `href` y `rank_name`, su posición alfabética), `district` (la elegida; la primera sin `district`), `sentence`, `tabs` (`[{'key', 'label', 'href', 'on'}]` de las tres vistas), `n_sim`, `page_title`, `image_file`, `image_sign`. La Tarea 4 le añade `party` y `view`.
- Produces: `MESSAGES` nuevos: `'seat odds not published'` → «Este run no tiene el desglose escaño a escaño.», `'party not found'` → «Partido no encontrado.», `'invalid view'` → «Vista no válida.».
- Produces: macros `tokens(row, color)` y `legend()` en `_seat_odds.html`; clases CSS con prefijo `so-`.

- [ ] **Step 1: Tests que fallan** — añadir a `tests/test_pages_circunscripciones.py` (con `import copy, os, re`, `from mtpy.lib import bundle, webapi`, `from tests.test_api_asgi import call`, `from tests.test_pages import api, assert_public_wording, html, initial  # noqa: F401`, `from tests.test_webapi_unit import RUN, fixture_data`). `call` recibe la query aparte y `event_query` escribe siempre `?mode=…`, también en el modo por defecto:

```python
def write_odds(fs, data=ODDS, modes=bundle.MODES):
    """Sobrescribe la parte `seat-odds` del run del bundle de prueba."""
    writer = bundle.BundleWriter(fs)
    for mode in modes:
        writer.write_json(bundle.path_part('es', RUN, 'seat-odds', mode), 'seat-odds', data, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()


@pytest.fixture
def odds_site(api, fresh_app):
    write_odds(fresh_app.fs)
    return api


def page_of(site, path, query=''):
    status, _, body = call(site, path, query=query)
    return status, re.sub(r'\s+', ' ', html(body))


def test_the_page_opens_on_the_district_with_most_open_seats(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones')
    assert status == 200 and '<h1 class="headline">Escaños que bailan</h1>' in page
    kpis = page[page.index('class="kpis"'):page.index('class="seg lg so-tabs"')]
    assert [int(n) for n in re.findall(r'class="v"[^>]*>(\d+)', kpis)] == [14, 5, 11, 2]
    assert 'de 19' in kpis and 'de 3' in kpis
    items = re.findall(r'class="so-item[^"]*" href="([^"#?]+)', page)
    assert items == ['/es/circunscripciones/cadiz', '/es/circunscripciones/la-coruna', '/es/circunscripciones/guadalajara']
    card = page[page.index('id="ficha"'):]
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan 6 aspirantes.' in card
    assert re.findall(r'class="so-token[^"]*"[^>]*title="([^"]+)"', card)[:3] == [
        '1.º de PP: seguro', '2.º de PP: seguro', '3.º de PP: lo consigue en el 80% de las simulaciones']
    assert '<link rel="canonical" href="' in page and page.count('<details class="how"') >= 1
    assert_public_wording(page, 'circunscripciones')


def test_a_district_has_its_own_address_and_never_labels_an_open_seat_as_sure(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones/la-coruna')
    card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
    assert status == 200 and 'La Coruña' in card and '<title>Escaños que bailan en La Coruña' in page
    labels = re.findall(r'<span class="so-pct[^"]*">([^<]+)</span>', card)
    assert set(labels) == {'30%', '98%', '24%', '37%', '1%'} and not {'99%', '100%', '0%'} & set(labels)
    assert len(re.findall(r'class="so-token safe"', card)) == 6
    assert len(re.findall(r'class="so-token open out"', card)) == 3 and len(re.findall(r'class="so-token open"', card)) == 2
    closed = page_of(odds_site, '/es/circunscripciones/guadalajara')[1]
    assert 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).' in closed and 'decidida' in closed


def test_the_page_follows_the_mode_and_keeps_it_in_its_links(odds_site, fresh_app):
    nowcast = copy.deepcopy(ODDS)
    nowcast['rows'][3] = odds_row(11, 'AA', 1, 2500)  # AA's seat in Cádiz: 76% in the election mode, 25% today
    write_odds(fresh_app.fs, nowcast, modes=('nowcast',))
    assert '>76%<' in page_of(odds_site, '/es/circunscripciones/cadiz')[1]  # less than 60 days away: forecast
    today = page_of(odds_site, '/es/circunscripciones/cadiz', 'mode=nowcast')[1]
    assert '>25%<' in today and '>76%<' not in today
    assert 'href="/es/circunscripciones/la-coruna?mode=nowcast#ficha"' in today
    assert 'href="/es?mode=nowcast#territorio"' in today  # the index of the event bar goes back to the event page


def test_the_page_does_not_exist_without_the_part_or_with_a_single_district(api, odds_site, fresh_app, tmp_path):
    assert call(odds_site, '/es/circunscripciones/no-existe')[0] == 404
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='circunscripcion=cadiz&mode=nowcast')
    # the form without JavaScript: on to the address of the district, keeping the mode
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast'
    write_odds(fresh_app.fs, fixture_data('seat-odds'))
    status, page = page_of(api, '/es/circunscripciones')
    assert status == 404 and 'Este run no tiene el desglose escaño a escaño.' in page  # one district
    for mode in bundle.MODES:
        os.remove(os.path.join(str(tmp_path), bundle.path_part('es', RUN, 'seat-odds', mode)))
    webapi.site().files.clear()
    assert call(api, '/es/circunscripciones')[0] == 404  # a run published before the part existed
```

  En `tests/test_web_routes.py`, junto a las comprobaciones de `page_target`: `page_target('/es/circunscripciones') == ('circunscripciones', 'index')`, `page_target('/es-md/circunscripciones/la-coruna') == ('circunscripciones', 'detail')` y `page_target('/es/circunscripciones/a/b') is None`.

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py tests/test_web_routes.py -q`. Esperado: 404 de ruta desconocida en los tests nuevos.

- [ ] **Step 3: Implementar el contexto y el controlador.**
  - `seat_odds_context`: `common_context(state, None, event=True)`; la parte con `bundle_part` en el modo del estado; `HttpError(404, 'seat odds not published')` si falta o tiene menos de dos regiones; `HttpError(404, 'region not found')` si `district` no es el `slug` de ninguna; el índice de la barra del evento reenlazado a `/{scope}{query}#id`, como en `sondeo_context`.
  - `kpis`: cuatro filas, en este orden: «Escaños seguros» (valor `safe`, unidad «de {seats}»), «Escaños en el aire» (`open`), «Aspirantes» (`contenders`, unidad «a esos {open}»), «Circunscripciones con algo en juego» (`live`, unidad «de {districts}»).
  - `href` de cada circunscripción: `/{scope}/circunscripciones/{slug}{query}#ficha`. `tabs`: «Por circunscripción» (`/{scope}/circunscripciones{query}`), «Por partido» y «Los más reñidos» (sus `href` son los de la Tarea 4: `?partido={slug del primer partido de la parte con alguna ficha}` y `?vista=renidos`, añadidos al `query`).
  - `page_title`: «Escaños que bailan en {circunscripción}». `image_file`: `escanos-{slug}`; `image_sign`: la firma que usa la tarjeta de un sondeo (`sondeo_context`).
  - `Circunscripciones(Page)`, `active = None`: `index_action(scope)` y `detail_action(scope, district)`. `index_action` con `circunscripcion` en la query responde `self.redirect` a `/{scope}/circunscripciones/{valor}` más el resto de la query (`mode`, `run`). La canónica, la de `Page.canonical_url()` (quita `mode` y `run`).

- [ ] **Step 4: Implementar la plantilla, el CSS y el script.**
  - `circunscripciones.html` extiende `base.html`: `title` «{{ page_title }} · {{ event_bar.title }} {{ event_bar.year }}»; `description` «Los escaños de {circunscripción}, uno a uno: cuáles tiene seguros cada partido y cuáles están en el aire, según las simulaciones del modelo para {título del evento} {año}.» (en las otras vistas, «de {siglas}, circunscripción a circunscripción» y «Los escaños más reñidos de {título del evento} {año}»); `scripts` con `/dist/js/pages/circunscripciones.js`; contenido en `<div id="content" class="stack">`: `m.headline('Escaños que bailan', 'h1')`, la entradilla «Qué escaños tiene seguros cada partido en cada circunscripción y cuáles están en el aire, según las {n_sim con punto de millar} simulaciones del modelo.», `m.kpis(kpis)`, `<nav class="seg lg so-tabs">` con los tres enlaces (`class="on"` en el actual) y la vista.
  - Vista por circunscripción: `<div class="so-md">` con la lista (`<aside class="card so-list">`) y la ficha (`<section class="card shot" id="ficha">`).
    - Lista: un `fieldset.seg.so-sort` con dos radios `name="orden"` («Más en el aire», marcado; «A–Z», `value="az"`) y un enlace `a.so-item` por circunscripción (`class="so-item on"` la actual) con el nombre, sus escaños y una pastilla «{n} en el aire» o «decidida». Cada enlace lleva `style="--by-open: {posición}; --by-name: {rank_name}"`; el orden lo cambia el CSS (`.so-item { order: var(--by-open) }` y `.so-list:has(input[value="az"]:checked) .so-item { order: var(--by-name) }`), sin JavaScript.
    - Desplegable para pantallas estrechas: `<form id="district-pick" class="so-pick" method="get" action="/{scope}/circunscripciones">` con los ocultos `mode` y `run` del formulario de circunscripción de `promedios.html`, un `select name="circunscripcion"` (valor = `slug`; texto «Cádiz · 8 escaños · 3 en el aire») y el botón «Ver» con `class="nojs-only"`. Bajo 720 px se ve el desplegable y no la lista; por encima, al revés.
    - Ficha: `m.card_image_button(image_file, image_sign)`; cabecera con el nombre (`h2`), «{n} escaños», pastilla «{safe} seguros» y, si `open > 0`, «{open} en el aire»; la frase; una rejilla `so-grid` con una fila por partido: punto de color y siglas (título = nombre completo), los escaños del reparto central bajo la cabecera «Central», las fichas (`tokens`) y la nota «2 seguros · 2 en el aire» (`so-note`, oculta bajo 720 px; «—» si no hay nada); la leyenda; y un `m.how()` con el «Cómo leerlo» de la spec.
  - `_seat_odds.html`: `tokens(row, color)` pinta `<span class="so-tokens" role="img" aria-label="…">` (la fila leída: «PP: 2 seguros; 3.º, 80%; 4.º, 7%») con un `<span class="so-token safe" title="…" style="background: {color}">` por ficha segura y, por cada una en el aire, `<span class="so-token open[ out]" title="…"><i style="border-color: {color}; background: {color con la opacidad alpha}"></i><span class="so-pct">{label}</span></span>` (`out` si el central no se la da). `legend()` pinta `div.so-legend` con los tres estados y «por debajo del 1% no se dibuja», con los umbrales tomados de las constantes.
  - CSS (`so-`), con las medidas de la spec: ficha de 22 px, esquinas de 6 px, 4 px de separación, borde de 2 px continuo (`open`) o discontinuo (`out`), etiqueta de 11 px en negrita (`open`) o en `--color-muted` sin negrita (`out`), salto de línea de las fichas. El resto, de la maqueta y de los componentes que ya hay (`.card`, `.seg`, `.pill`, `.dot`, `.kpis`).
  - `circunscripciones.js`: `wireForm('district-pick')` y `wireCardImages(document, showError)`, dentro de `mountWhenReady`/`tryDraw` como `sondeo.js`.

- [ ] **Step 5: Ver que pasan** — `pytest tests/test_pages_circunscripciones.py tests/test_web_routes.py tests/test_pages.py -q`. Esperado: todo verde.

- [ ] **Step 6: Commit** — «Página «Escaños que bailan» (`/{scope}/circunscripciones`): cabecera con los seguros y los que están en el aire y la ficha de cada circunscripción, escaño a escaño».

### Task 4: Las vistas por partido y «Los más reñidos»

**Files:**
- Modify: `mtpy/lib/pages/circunscripciones.py`, `mtpy/controllers/Circunscripciones.py`, `web/templates/circunscripciones.html`, `web/dist/css/site.css`
- Test: `tests/test_pages_circunscripciones.py`

**Interfaces:**
- Consumes: `party_view`, `closest_seats`, las macros `tokens` y `legend`, `seat_odds_context` (Tarea 3).
- Produces: `pages.seat_odds_context(state, district=None, party=None, view=None)`. Con `party` (un `slug` de siglas): `view = 'party'`, más `parties` (`[{'name', 'fullname', 'color', 'href', 'on'}]`, los partidos de la parte con alguna ficha, en el orden de la parte) y `party` (`party_view` con `fullname`, `color` y un `href` por fila). Con `view == 'renidos'`: `view = 'closest'` y `closest` (`closest_seats`, cada uno con `color` y `href`). `district` gana a `party`, y `party` a `view`.

- [ ] **Step 1: Tests que fallan** — añadir:

```python
def test_the_party_view_lists_its_doubtful_districts_and_sums_the_settled_ones(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')
    assert status == 200 and '<title>Escaños que bailan: PSOE' in page
    assert re.findall(r'class="so-party[^"]*" href="[^"]*partido=([a-z]+)', page) == ['pp', 'psoe', 'vox', 'aa', 'bng']
    card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
    assert re.findall(r'href="/es/circunscripciones/([a-z-]+)\?mode=forecast#ficha"', card) == ['cadiz', 'la-coruna']
    assert 'Tiene 4 escaños seguros y aspira a otros 3 en 2 circunscripciones.' in card
    assert 'Sin nada en el aire (1):' in card and 'Guadalajara 1' in card
    assert 'rel="canonical" href="' in page and 'circunscripciones?partido=psoe"' in page
    assert_public_wording(page, 'circunscripciones por partido')
    assert call(odds_site, '/es/circunscripciones', query='partido=nadie')[0] == 404
    kept = page_of(odds_site, '/es/circunscripciones', 'partido=pp&mode=nowcast')[1]
    assert 'href="/es/circunscripciones/cadiz?mode=nowcast#ficha"' in kept


def test_the_closest_view_ranks_the_open_seats_by_their_distance_to_a_coin_toss(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones', 'vista=renidos')
    table = page[page.index('id="closest"'):]
    rows = re.findall(r'<tr>(.*?)</tr>', table[table.index('<tbody>'):])
    assert status == 200 and len(rows) == 11
    assert 'Cádiz' in rows[0] and '3.º de PSOE' in rows[0] and '41%' in rows[0] and 'no se lo da' in rows[0]
    assert 'La Coruña' in rows[1] and '1.º de VOX' in rows[1] and 'se lo da' in rows[1] and 'no se lo da' not in rows[1]
    assert 'circunscripciones?vista=renidos"' in page
    assert_public_wording(page, 'circunscripciones, los más reñidos')
    assert call(odds_site, '/es/circunscripciones', query='vista=otra')[0] == 400
```

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py -q`. Esperado: fallan los dos nuevos.

- [ ] **Step 3: Implementar.**
  - Contexto: `HttpError(404, 'party not found')` si `party` no es el `slugify` de las siglas de ningún partido con alguna ficha; `HttpError(400, 'invalid view')` si `view` no es `renidos`. El controlador pasa `partido` y `vista` de la query. `page_title`: «Escaños que bailan: {siglas}» y «Los escaños más reñidos». `image_file` de la vista por partido: `escanos-{slug del partido}`.
  - Vista por partido: los botones `a.so-party` (`.seg`) con punto de color y siglas; la tarjeta (`section.card.shot#ficha`) con el nombre completo, «{seats} escaños en el reparto central», las pastillas «{safe} seguros» y «{open} en el aire», la frase «Tiene {safe} escaños seguros y aspira a otros {open} en {n} circunscripciones. Primero, las más dudosas.» (singulares: «1 escaño seguro», «1 circunscripción»; sin fichas en el aire, «Tiene {safe} escaños seguros y ninguno en el aire.»), una fila por circunscripción (nombre enlazado a su ficha, escaños de la circunscripción —columna «Escaños», oculta bajo 720 px—, «Central», fichas, nota) y, si hay, la línea «Sin nada en el aire ({n}): Guadalajara 1 · …»; la leyenda y el mismo «Cómo leerlo».
  - Vista «Los más reñidos»: tarjeta con la frase «Los {mostrados} escaños más cercanos al cara o cruz, de los {contenders} que están en el aire. La raya marca el 50%.» y una tabla `#closest` con la cabecera en `<thead>` y las filas en `<tbody>`: circunscripción (enlazada a su ficha), «{k}.º de {siglas}» con su punto, una barra de 0 a 100 con la raya del 50% y la etiqueta, y «se lo da» o «no se lo da» bajo la cabecera «Reparto central» (columna oculta bajo 720 px).

- [ ] **Step 4: Ver que pasan** — `pytest tests/test_pages_circunscripciones.py tests/test_pages.py -q`. Esperado: todo verde.

- [ ] **Step 5: Commit** — «Escaños que bailan: vista por partido (sus circunscripciones, de más a menos dudosa) y «Los más reñidos»».

### Task 5: Las entradas desde la página del evento

**Files:**
- Modify: `mtpy/lib/pages/circunscripciones.py` (`seat_odds_links`), `mtpy/lib/pages/__init__.py`, `mtpy/lib/pages/escanos.py` (`seats_context`, `csv_links`, `CSV_PARTS`), `web/templates/promedios.html` (tarjeta «Resultados por territorio»), `web/dist/js/pages/seats-wire.js` (donde llama a `renderRegion`)
- Test: `tests/test_pages_circunscripciones.py`, `tests/test_pages.py` (`test_csv_links_cover_the_five_parts_in_order`)

**Interfaces:**
- Produces: `pages.seat_odds_links(scope: str, run: str, mode: str, query: str) -> Optional[dict]`: `{'page': '/{scope}/circunscripciones{query}', 'districts': {region_id: '/{scope}/circunscripciones/{slug}{query}#ficha'}}`, o `None` si el run no tiene la parte o tiene menos de dos regiones.
- Produces: `seats_context` añade `odds` (ese dict o `None`) y, en `initial_extra`, `odds_hrefs` (`odds['districts']` o `{}`); `csv_links(scope, mode, run, skip=())` deja fuera las partes de `skip`, y `seats_context` le pasa `('seat-odds',)` cuando el run no tiene la parte; `CSV_PARTS` gana `('seat-odds', 'Escaños uno a uno por circunscripción', True)` tras `scenario`.

- [ ] **Step 1: Tests que fallan** — en `tests/test_pages_circunscripciones.py`:

```python
DISTRICTS = {  # the districts of `ODDS` in the `districts` part, so that the detail of the event page shows one of them
    'parties': ['PP'],
    'regions': [dict(region, community=None, votes=1000) for region in ODDS['regions']],
    'rows': [{'region_id': region['id'], 'region': region['name'], 'name': 'PP', 'pct': 30.0, 'pct_lo': 27.0,
              'pct_hi': 33.0, 'seats': 3, 'seats_median': 3.0, 'seats_mean': 3.1, 'seats_lo': 2, 'seats_hi': 4,
              'p_seats': 1.0} for region in ODDS['regions']],
}


def test_the_event_page_links_the_page_and_the_card_of_the_district_shown(odds_site, api, fresh_app, tmp_path):
    writer = bundle.BundleWriter(fresh_app.fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part('es', RUN, 'districts', mode), 'districts', DISTRICTS, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()
    status, _, body = call(odds_site, '/es', query='mode=nowcast')
    page = re.sub(r'\s+', ' ', html(body))
    territory = page[page.index('id="territorio"'):page.index('id="tabla"')]
    assert status == 200 and 'href="/es/circunscripciones?mode=nowcast"' in territory and 'Escaños que bailan' in territory
    assert re.search(r'<a id="region-odds"[^>]*href="/es/circunscripciones/[a-z-]+\?mode=nowcast#ficha"', territory)
    assert initial(body)['odds_hrefs']['11'] == '/es/circunscripciones/cadiz?mode=nowcast#ficha'
    assert 'seat-odds' in page[page.index('id="escanos"'):]  # its CSV among the downloads
    for mode in bundle.MODES:
        os.remove(os.path.join(str(tmp_path), bundle.path_part('es', RUN, 'seat-odds', mode)))
    webapi.site().files.clear()
    status, bare = page_of(api, '/es')
    assert status == 200 and '/es/circunscripciones' not in bare and 'region-odds' not in bare and 'seat-odds' not in bare
```

  En `tests/test_pages.py`, `test_csv_links_cover_the_five_parts_in_order` pasa a seis (`seat-odds` tras `scenario`) y comprueba que `pages.csv_links('es', 'nowcast', RUN, skip=('seat-odds',))` devuelve las cinco de antes.

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py -k event_page tests/test_pages.py -k csv_links -q`.

- [ ] **Step 3: Implementar.**
  - En la tarjeta `districts` de `promedios.html`, sólo `{% if odds %}`: bajo la tabla del detalle, `<p class="more"><a id="region-odds" class="more-link" href="{{ odds.districts[region.id] }}">Escaño a escaño en {{ region.name }} ›</a></p>` (sin el párrafo si la región mostrada no está en `odds.districts`), y al final de la tarjeta, antes del «Cómo leerlo», `<p class="more"><a class="more-link" href="{{ odds.page }}">Escaños que bailan: los seguros y los que están en el aire ›</a></p>`.
  - En `seats-wire.js`, al redibujar el detalle de una circunscripción: `#region-odds` toma el `href` de `initial.odds_hrefs[id]` y el texto «Escaño a escaño en {nombre} ›»; se oculta si no hay `href`. Las claves de `odds_hrefs` llegan como texto.

- [ ] **Step 4: Ver que pasan** — `pytest tests/test_pages_circunscripciones.py tests/test_pages.py tests/test_js_territory.py tests/test_js_event_page.py -q`. Esperado: todo verde.

- [ ] **Step 5: Commit** — «Página del evento: enlaces a «Escaños que bailan» y a la ficha de la circunscripción del detalle, y la descarga de la parte `seat-odds`».

### Task 6: Método, contrato de las rutas y comprobación en el navegador

**Files:**
- Modify: `web/templates/metodo_escanos.html` (tarjeta `cifras`, «Qué es cada cifra»), `docs/web/contrato.md` (rutas de las páginas), `docs/superpowers/specs/2026-11-12-escanos-que-bailan-design.md` (sección «Estado al cierre»)
- Test: `tests/test_pages.py`

- [ ] **Step 1: Test que falla** — en `tests/test_pages.py`, junto a los de `/metodo/escanos`:

```python
def test_method_seats_explains_the_safe_and_the_open_seats(api):
    page = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    block = page[page.index('<h3 id="seguros">'):page.index('id="comprobacion"')]
    assert 'Escaños seguros y en el aire' in block and '99%' in block and '1%' in block
    assert 'no es una garantía' in block
    assert_public_wording(page, 'metodo/escanos')
```

- [ ] **Step 2: Ejecutarlo y ver que falla** — `pytest tests/test_pages.py -k safe_and_the_open -q`.

- [ ] **Step 3: El texto** — al final de la tarjeta `cifras` de `metodo_escanos.html`, con los umbrales tomados de `pages.SAFE_SHARE` y `pages.OPEN_FLOOR` si el contexto de la página ya recibe constantes así, o escritos si no:

  > `<h3 id="seguros">Escaños seguros y en el aire</h3>` En cada simulación repartimos los escaños de cada circunscripción. Al terminar contamos, para cada partido y cada escaño (el primero, el segundo, el tercero…), en cuántas simulaciones lo consigue. Llamamos seguro al escaño que sale en el 99% o más de las simulaciones, y decimos que está en el aire el que sale entre el 1% y el 99%; por debajo del 1% lo damos por descartado.
  >
  > «Seguro» no es una garantía: es lo que dicen nuestras simulaciones, que heredan los errores del promedio y los de la proyección del voto a las circunscripciones. El reparto central es otra cosa: reparte el voto central, y puede dar a un partido un escaño que sale en menos de la mitad de las simulaciones. Cuando ocurre, lo marcamos.

- [ ] **Step 4: Contrato** — en la sección de rutas de las páginas de `docs/web/contrato.md`: las dos rutas, los parámetros `partido`, `vista`, `circunscripcion`, `mode` y `run`, y los 404 y el 400.

- [ ] **Step 5: Toda la batería** — `pytest -q`. Esperado: todo verde, con el mismo número de tests saltados que en `dev`.

- [ ] **Step 6: Comprobar en un navegador de verdad** (Chrome headless o a mano), con la aplicación servida en local sobre un bundle de prueba cuya parte `seat-odds` salga de `export_seat_odds` sobre un `Simulator` real de `es` (sólo lectura contra la base, 1.000 simulaciones, escrito en una carpeta temporal fuera del repositorio, nunca en S3), o, si la base no responde, sobre una parte sintética de al menos diez circunscripciones con una de 38 escaños. Pasa si, a 1280 px y a 390 px:
  - Madrid (o la de 38) muestra sus fichas en varias líneas sin desbordar la tarjeta ni pisar las etiquetas;
  - el conmutador «Más en el aire / A–Z» reordena la lista sin recargar, y a 390 px se ve el desplegable y no la lista, y cambiarlo lleva a la ficha elegida;
  - ninguna ficha en el aire dice 99%, 100% ni 0%;
  - la vista por partido de un partido grande (unas 50 filas) y «Los más reñidos» no tienen desplazamiento horizontal;
  - el botón de imagen descarga la ficha entera, con su leyenda y sin el botón;
  - en `/es`, al cambiar de circunscripción en «Resultados por territorio», el enlace «Escaño a escaño en…» cambia con ella.

  Anotar en la spec, en «Estado al cierre», lo comprobado y lo que no se haya podido comprobar.

- [ ] **Step 7: Commit** — «Escaños que bailan: apartado del método, rutas en el contrato y cierre de la spec».

## Después del plan (Luis)

Push de `dev`, imagen nueva de web y de jobs, y un `update` de cada ámbito antes del 24-N: el primer run con la imagen nueva ya lleva la parte y la página aparece sola. No hace falta relleno.
