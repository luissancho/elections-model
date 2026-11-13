# «Escaños que bailan», segunda versión — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** La página `/{scope}/circunscripciones` enseña cada opción de escaño como un círculo en tres estados (seguro, probable, en juego; nada por debajo del 25%), con un mapa de provincias con selectores y fichas que cambian en el sitio, un hemiciclo y unas columnas por partido, y sin rastro del reparto central.

**Architecture:** Sólo web: la parte `seat-odds` no cambia y no se republica nada. Las funciones puras de `mtpy/lib/pages/circunscripciones.py` pasan de dos estados a tres, y un módulo nuevo (`mtpy/lib/pages/seat_charts.py`) calcula en Python la colocación de los tres dibujos (casillas del mapa, plazas del hemiciclo, columnas). Jinja2 lo pinta todo en el servidor; los selectores, el conmutador y el cambio de ficha funcionan con radios y `:has()`. El JavaScript sólo cambia la dirección al tocar una casilla, pone nombre a las imágenes y las descarga.

**Tech Stack:** Python 3.11 (pandas para leer el catálogo), Jinja2, CSS de `web/dist/css/site.css`, módulos ES de `web/dist/js`, pytest (arnés ASGI de `tests/test_api_asgi.py`; `node` para los tests de JavaScript).

**Spec:** `docs/superpowers/specs/2026-11-13-escanos-que-bailan-mapas-design.md`. Maqueta aprobada: `.superpowers/brainstorm/16860-1794563945/content/mapa-partidos-columnas-bloques.html` del checkout principal (fuera de git; leer su `<style>`, líneas 7-112, y su `<script>`, líneas 130-332, no los blobs base64). El CSS de cada dibujo sale de ahí, con las clases `so-` de este plan.

## Global Constraints

- Estados: seguro si la probabilidad es ≥ 99% (`SAFE_SHARE = 0.99`), probable si ≥ 75% y < 99% (`LIKELY_SHARE = 0.75`), en juego si ≥ 25% y < 75% (`PLAY_SHARE = 0.25`). Por debajo del 25% no se dibuja ni se cuenta, y ningún texto lo dice. La probabilidad es `reach[k-1] / n_sim`, comparada sin redondear. `OPEN_FLOOR` deja de existir.
- La unidad es la «opción de escaño». Palabras retiradas de los textos de la página y de sus tests: «posible», «en el aire», «aspirantes», «reparto central», «Central» (los nombres de circunscripción no cuentan: Asturias tiene una que se llama «Central»). Nada del reparto central en la página: ni columnas, ni marcas, ni menciones; el `seats` de la parte se queda sin uso.
- El signo es un círculo en el color del partido: sólido si es seguro; probable, opacidad `0,42 + 0,26 × (p − 0,75) / 0,24` y borde continuo; en juego, opacidad `0,06 + 0,20 × (p − 0,25) / 0,50` y borde **discontinuo**. Nunca rayados ni cuadrados. Borde de 1 px en los círculos pequeños y de 2 px en los de las fichas (22 px). La etiqueta es un porcentaje entero pegado («78%») entre 25% y 98%, sólo en las fichas. Cada círculo lleva su `title` («3.º de PP: lo consigue en el 78% de las simulaciones», «2.º de PP: seguro»).
- No cambian: las direcciones (`/{scope}/circunscripciones`, `/{slug}`, `?partido=`, `?vista=renidos`), el 301 del desplegable, la canónica, los 404 y el 400, el modo y el run, la barra del evento, la línea de la imagen (`so-stamp`) y `published_seat_odds` / `seat_odds_links`. Se añade `?vista=partidos`.
- La página se lee y se maneja sin JavaScript: selectores, conmutador y cambio de ficha con radios y `:has()`. Nada de scripts en línea (CSP `default-src 'self'`); los estilos en línea y un bloque `<style>` generado están permitidos (`style-src 'unsafe-inline'`). Ninguna librería de gráficos.
- Web en `mtpy/controllers` y `mtpy/lib`; nada de FastAPI ni scripts sueltos. Identificadores en inglés neutro, docstrings numpy en todas las funciones. Textos públicos en español, «sólo» con tilde, sin «empresa» ni «patrocinador», ningún «Cómo leerlo» abierto (`assert_public_wording`), porcentajes pegados. Todos los textos son provisionales: los revisa Luis.
- No se toca el modelo, la parte `seat-odds`, la API ni `mtpy/lib/publish`. No se publica, no se lanza ningún job, no se escribe en S3 ni en la base. El `.env` no se lee, no se imprime y no se copia.
- Luis y otras sesiones trabajan y lanzan jobs en el checkout principal: todo el trabajo va en un worktree con su rama (`git worktree add .worktrees/escanos-mapas -b escanos-mapas dev`); nunca se deja un fichero sin commit en el checkout principal, nunca `git stash`, nunca `git commit -a`, nunca se reescribe un commit ajeno. Antes de cada commit, `git status --short` y `git log -1 --oneline`; sólo los ficheros de la tarea.
- Tests con `~/.pyenv/versions/3.11.9/bin/pytest` (en adelante `pytest`), nunca `python3`. Commits en español con el trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Puntos de corte: lo «estrecho» es por debajo de 720 px, como en la primera versión; la comprobación final es a 360, 390, 768, 1024 y 1280 px. Todo destino de un ancla lleva `scroll-margin` bajo la cabecera fija (regla de `#ficha, #vistas` en `site.css`).

## Review Focus

1. **Un ámbito sin posiciones de mapa**, o con alguna circunscripción sin posición (los autonómicos; un catálogo sin las columnas nuevas): el mapa sale en rejilla seguida, por nombre y con el nombre completo, y nada falla. Tarea 2.
2. **Una circunscripción sin opciones que disputar o sin ninguna opción** (el escaño que falta repartido entre partidos por debajo del 25%): casilla y ficha salen, la frase no dice «se los disputan», no hay filas rotas; y el pie de las opciones no sale si no superan a los escaños. Tareas 1 y 2.
3. **Un run sin `bmaps.blocks` o con bloques incompletos** (el bundle de prueba sólo trae `Derecha`), y un partido con opciones fuera de los tres bloques: columnas por partido sin «Otros» en el primer caso, a «Otros» en el segundo; ninguna opción se pierde. Tarea 3.
4. **Muy pocos o muchos círculos**: un hemiciclo de 3 opciones y uno de 390; una casilla de 40 opciones (Madrid) crece hacia abajo sin pisar a las vecinas ni desbordar a 360 px. Tareas 3 y 6.
5. **El estado de los radios fuera del camino feliz**: la imagen descargada recoge el partido y el estado marcados (la copia no puede perder los radios), y tras volver atrás o recargar con el formulario restaurado, la línea y el nombre del fichero coinciden con lo marcado. Tareas 4 y 6.

## Decisiones del plan (para que Luis las vea al revisarlo)

1. La pestaña Partidos tiene la dirección `?vista=partidos`; los botones de partido aterrizan en el selector (`#partidos`), no arriba de las pestañas.
2. El hemiciclo es SVG en línea con atributos de presentación (como la maqueta) y sus recuentos del centro son HTML encima; si la imagen descargada pierde el borde discontinuo, en la Tarea 6 se pasa a círculos HTML posicionados.
3. En la imagen del mapa la casilla elegida no lleva el borde oscuro (la ficha a la que se refiere no sale en esa imagen).
4. Al tocar una casilla la página no se desplaza sola hasta la ficha (como en la maqueta). En la Tarea 6 se anota si la ficha queda fuera de la vista, para decidirlo con datos.
5. Con JavaScript, al tocar una casilla cambian la dirección (`replaceState`, sin llenar el historial) y el título de la pestaña.
6. Títulos, descripciones y la frase de «Los más reñidos» de la Tarea 1.

## Estructura de ficheros

| Fichero | Responsabilidad |
|---|---|
| `mtpy/lib/pages/circunscripciones.py` | Estados, opciones, vistas de circunscripción y de partido, frases y el contexto de la página (se adapta) |
| `mtpy/lib/pages/seat_charts.py` (nuevo) | Colocación pura de los tres dibujos: casillas del mapa, plazas del hemiciclo, columnas |
| `mtpy/lib/webapi/checks.py` | Lector con TTL de las posiciones de mapa del catálogo |
| `data/es-districts.csv` | Abreviatura y casilla de las 52 circunscripciones de `es` |
| `web/templates/circunscripciones.html`, `web/templates/_seat_odds.html` | La página y las macros del signo, la leyenda y el «Cómo leerlo» |
| `web/dist/css/site.css` (bloque `so-`) | El signo, el mapa, el hemiciclo, las columnas |
| `web/dist/js/seat-odds.js` (nuevo), `web/dist/js/pages/circunscripciones.js` | Nombres de las imágenes (puro) y el cableado de la página |
| `tests/test_pages_circunscripciones.py`, `tests/test_pages_seat_charts.py` (nuevo), `tests/test_js_seat_odds.py` (nuevo), `tests/test_data_files.py`, `tests/test_pages.py` | Pruebas |

---

### Task 1: Tres estados, el círculo y el vocabulario

La página queda entera con el vocabulario nuevo y navegable (desplegable + ficha, vista de partido, «Los más reñidos»); el mapa y el hemiciclo llegan en las Tareas 2 y 3.

**Files:**
- Modify: `mtpy/lib/pages/circunscripciones.py`, `mtpy/lib/pages/__init__.py` (exportaciones)
- Modify: `web/templates/circunscripciones.html`, `web/templates/_seat_odds.html`, `web/dist/css/site.css` (bloque `so-`, desde la línea 1121)
- Test: `tests/test_pages_circunscripciones.py`

**Interfaces (Produces, todas exportadas en `pages`):**
- `SAFE_SHARE = 0.99`, `LIKELY_SHARE = 0.75`, `PLAY_SHARE = 0.25`, `STATES = ('safe', 'likely', 'play')`. Fuera `OPEN_FLOOR` y `district_order`.
- `seat_state(share: float) -> Optional[str]`: `'safe'`, `'likely'`, `'play'` o `None` por debajo de `PLAY_SHARE`.
- `token_label(share: float) -> str`: como hoy, limitado a entre 25% y 98%.
- `token_alpha(state: str, share: float) -> Optional[float]`: `None` si es seguro; las dos fórmulas de arriba, redondeadas a 3 decimales.
- `seat_tokens(name: str, reach: list, n_sim: int) -> list`: una ficha por opción (sin el parámetro `seats`), `{'k', 'share', 'state', 'label', 'alpha', 'title'}`; `label` y `alpha` son `None` si es segura.
- `district_views(data: dict) -> list`: `{'id', 'name', 'slug', 'seats', 'safe', 'likely', 'play', 'rows'}`; `rows` sólo con los partidos con alguna opción, `{'name', 'tokens', 'safe', 'likely', 'play', 'expected'}`, donde `expected` es la suma de las probabilidades de sus opciones dibujadas; ordenadas por `expected` descendente (a igualdad, el orden de la parte).
- `default_district(districts: list) -> dict`: la de más opciones en juego; a igualdad, más escaños y después el nombre (`_sort_name`).
- `scope_totals(districts: list) -> dict`: `{'seats', 'safe', 'likely', 'play', 'options'}`.
- `options_sentence(options: int, seats: int) -> str`: «Hay 390 opciones para 350 escaños: a un mismo escaño pueden optar varios partidos.» si `options > seats`; `''` si no.
- `state_text(state: str, count: int) -> str`: «1 seguro», «5 seguros», «1 probable», «3 probables», «2 en juego».
- `district_sentence(district: dict) -> str`: la de hoy; «los otros N» son `max(seats − safe, 0)` y los partidos que se los disputan, los que tienen alguna opción probable o en juego.
- `party_view(districts: list, name: str) -> dict`: `{'name', 'safe', 'likely', 'play', 'rows', 'settled'}`; `rows` son las circunscripciones donde tiene alguna opción no segura (cada una con `district`, `tokens`, `safe`, `likely`, `play`), primero la que tiene una opción no segura más cerca del 50%.
- `party_sentence(party: dict) -> str`: sólo los estados con alguna opción, «escaño(s)» tras la primera cifra.
- `closest_seats(districts: list, limit: int = CLOSEST_SHOWN) -> list`: las opciones no seguras más cercanas al 50%, `{'district', 'slug', 'party', 'k', 'share', 'label', 'state'}`.
- `closest_sentence(shown: int) -> str`: 0 → «No hay ningún escaño probable ni en juego.»; 1 → «El único escaño que no es seguro. La raya marca el 50%.»; n → «Los n escaños más cercanos al cara o cruz. La raya marca el 50%.»
- `tokens_note(row: dict) -> str` («2 seguros · 1 probable»), `tokens_reading(name: str, row: dict) -> str` («PP: 2 seguros; 3.º, 80%»).
- `drawn_token(token: dict, color: str) -> dict`: copia con `color` y `fill` (`tint(color, alpha)`; `None` si es segura). `drawn_row` lo usa. Las Tareas 2 y 3 lo consumen.
- Macro `s.circle(token, extra='')` de `_seat_odds.html`: `<i class="so-o {state}[ {extra}]" style="--c: {color}[; --f: {fill}]" title="{title}"></i>`.
- `thresholds` del contexto: `{'safe': '99%', 'likely': '75%', 'play': '25%'}` (`threshold_label` de las tres constantes).

- [ ] **Step 1: Tests que fallan (funciones puras)** — en `tests/test_pages_circunscripciones.py`, sustituir `ODDS` y los tests de las funciones puras:

```python
ODDS = {
    'n_sim': 10000, 'parties': ['PP', 'PSOE', 'VOX', 'AA', 'BNG', 'UP'],
    'regions': [{'id': 11, 'name': 'Cádiz', 'seats': 8}, {'id': 19, 'name': 'Guadalajara', 'seats': 3},
                {'id': 15, 'name': 'La Coruña', 'seats': 8}],
    'rows': [
        odds_row(11, 'PP', 3, 10000, 10000, 8010, 730, 10), odds_row(11, 'PSOE', 2, 10000, 9950, 4110, 80),
        odds_row(11, 'VOX', 2, 10000, 8530, 570), odds_row(11, 'AA', 1, 7580),
        odds_row(11, 'UP', 0, 2400),                      # never reaches 25%: no option anywhere
        odds_row(19, 'PP', 1, 10000, 60), odds_row(19, 'PSOE', 1, 9990), odds_row(19, 'VOX', 1, 9950),
        odds_row(15, 'PP', 3, 10000, 10000, 9900, 3000),  # the third, exactly at 99%: safe; the fourth, in play
        odds_row(15, 'PSOE', 2, 10000, 9899, 2499),       # 98,99%: likely, labelled 98%; 24,99%: not drawn
        odds_row(15, 'VOX', 1, 3657),
        odds_row(15, 'BNG', 2, 10000, 9950, 2500, 99),    # the third, exactly at 25%: in play
    ],
}


def test_states_follow_the_three_thresholds_to_the_exact_count():
    edges = pages.seat_tokens('X', [9900, 9899, 7500, 7499, 2500, 2499], 10000)
    assert [(t['k'], t['state'], t['label'], t['alpha']) for t in edges] == [
        (1, 'safe', None, None), (2, 'likely', '98%', 0.68), (3, 'likely', '75%', 0.42),
        (4, 'play', '75%', 0.26), (5, 'play', '25%', 0.06)]
    assert [pages.seat_state(share) for share in (1.0, 0.99, 0.9899, 0.75, 0.7499, 0.25, 0.2499, 0.0)] == [
        'safe', 'safe', 'likely', 'likely', 'play', 'play', None, None]
    assert (pages.token_alpha('likely', 0.801), pages.token_alpha('play', 0.5), pages.token_alpha('safe', 1.0)) == (
        0.475, 0.16, None)
    assert edges[0]['title'] == '1.º de X: seguro'
    assert edges[3]['title'] == '4.º de X: lo consigue en el 75% de las simulaciones'
    assert pages.seat_tokens('X', [], 10000) == [] and 'central' not in edges[0] and not hasattr(pages, 'OPEN_FLOOR')


def test_district_views_count_the_options_of_each_state():
    cadiz, guadalajara, coruna = pages.district_views(ODDS)
    assert (cadiz['name'], cadiz['slug'], cadiz['seats']) == ('Cádiz', 'cadiz', 8) and coruna['slug'] == 'la-coruna'
    assert [(d['safe'], d['likely'], d['play']) for d in (cadiz, guadalajara, coruna)] == [(5, 3, 1), (3, 0, 0), (6, 1, 3)]
    # only the parties with some option (UP has none), by the expected seats among the drawn ones
    assert [row['name'] for row in cadiz['rows']] == ['PP', 'PSOE', 'VOX', 'AA']
    assert [row['name'] for row in coruna['rows']] == ['PP', 'BNG', 'PSOE', 'VOX']
    assert [(r['safe'], r['likely'], r['play']) for r in cadiz['rows']] == [(2, 1, 0), (2, 0, 1), (1, 1, 0), (0, 1, 0)]
    assert by_name(cadiz)['PSOE']['expected'] == pytest.approx(2.406) and 'seats' not in by_name(cadiz)['PSOE']


def test_the_default_district_the_totals_and_the_foot_sentence():
    districts = pages.district_views(ODDS)
    assert pages.default_district(districts)['name'] == 'La Coruña'  # three options in play
    tie = [{'name': 'Zamora', 'play': 1, 'seats': 3}, {'name': 'Ávila', 'play': 1, 'seats': 3},
           {'name': 'Soria', 'play': 1, 'seats': 2}]
    assert pages.default_district(tie)['name'] == 'Ávila'
    assert pages.default_district(tie + [{'name': 'Madrid', 'play': 1, 'seats': 38}])['name'] == 'Madrid'
    assert pages.scope_totals(districts) == {'seats': 19, 'safe': 14, 'likely': 4, 'play': 4, 'options': 22}
    assert pages.options_sentence(22, 19) == ('Hay 22 opciones para 19 escaños: a un mismo escaño pueden optar '
                                              'varios partidos.')
    assert pages.options_sentence(19, 19) == '' and pages.options_sentence(17, 19) == ''


def test_the_party_view_the_closest_seats_and_the_texts_of_the_states():
    districts = pages.district_views(ODDS)
    view = pages.party_view(districts, 'PSOE')
    assert (view['safe'], view['likely'], view['play']) == (4, 1, 1) and 'seats' not in view
    assert [row['district']['name'] for row in view['rows']] == ['Cádiz', 'La Coruña']
    assert view['settled'] == [{'name': 'Guadalajara', 'slug': 'guadalajara', 'safe': 1}]
    closest = pages.closest_seats(districts)
    assert len(closest) == 8 and closest[0] == {'district': 'Cádiz', 'slug': 'cadiz', 'party': 'PSOE', 'k': 3,
                                                'share': 0.411, 'label': '41%', 'state': 'play'}
    assert [(c['district'], c['party'], c['k']) for c in closest[1:4]] == [
        ('La Coruña', 'VOX', 1), ('La Coruña', 'PP', 4), ('La Coruña', 'BNG', 3)]
    assert (closest[-1]['party'], closest[-1]['label'], closest[-1]['state']) == ('PSOE', '98%', 'likely')
    assert [pages.state_text(s, n) for s, n in [('safe', 1), ('safe', 5), ('likely', 1), ('likely', 3), ('play', 1),
                                                ('play', 2)]] == ['1 seguro', '5 seguros', '1 probable',
                                                                  '3 probables', '1 en juego', '2 en juego']
    cadiz = by_name(districts[0])
    assert pages.tokens_note(cadiz['PP']) == '2 seguros · 1 probable' and pages.tokens_note(cadiz['AA']) == '1 probable'
    assert pages.tokens_reading('PP', cadiz['PP']) == 'PP: 2 seguros; 3.º, 80%'
    assert [pages.closest_sentence(n) for n in (0, 1, 8)] == [
        'No hay ningún escaño probable ni en juego.', 'El único escaño que no es seguro. La raya marca el 50%.',
        'Los 8 escaños más cercanos al cara o cruz. La raya marca el 50%.']


@pytest.mark.parametrize('safe, likely, play, text', [
    (66, 30, 25, 'Tiene 66 escaños seguros, 30 probables y 25 en juego.'), (2, 0, 0, 'Tiene 2 escaños seguros.'),
    (1, 0, 0, 'Tiene 1 escaño seguro.'), (3, 0, 2, 'Tiene 3 escaños seguros y 2 en juego.'),
    (0, 2, 1, 'Tiene 2 escaños probables y 1 en juego.'), (0, 0, 1, 'Tiene 1 escaño en juego.'),
])
def test_party_sentence_names_only_the_states_with_some_option(safe, likely, play, text):
    assert pages.party_sentence({'safe': safe, 'likely': likely, 'play': play}) == text
```

  El test parametrizado de `token_label` cambia sus bordes: `(0.25, '25%')` y `(0.2549, '25%')` en lugar de `(0.0149, '1%')` y `(0.01, '1%')`. El de `district_sentence` se conserva con todas sus variantes: `sentence_of` construye las filas con `'likely': 0, 'play': int(name in possible)` y sin `'open'`, y gana el caso `(1, {}, [], 'El escaño no es seguro.')` (una circunscripción sin ninguna opción). Se borran los tests de `district_order`, del partido con escaño central y sin fichas, y de `OPEN_FLOOR`.

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_pages_circunscripciones.py -q -k "states or district_views or default_district or party_view or party_sentence or token_label or district_sentence"`. Esperado: fallan por `seat_state`, `state`, `default_district`.

- [ ] **Step 3: Las funciones puras** — en `circunscripciones.py`, según el bloque de interfaces. `seat_odds_context` deja de devolver `kpis` y `rank_name`; `districts` va ordenada por nombre (`_sort_name`) con su `href`; sin circunscripción, partido ni vista, la elegida es `default_district`; `party` pierde `seats`; el docstring del módulo y los de cada función se reescriben (ninguno menciona ya el reparto central). Actualizar las exportaciones de `pages/__init__.py`.

- [ ] **Step 4: Tests que fallan (la página)** — sustituyen a `test_the_page_opens_on_the_district_with_most_open_seats`, `test_a_district_has_its_own_address_and_never_labels_an_open_seat_as_sure` y `test_the_list_sorts_itself_without_javascript_and_the_rows_read_aloud` (importar `visible_text` de `tests.test_pages`):

```python
RETIRED = ('posible', 'en el aire', 'aspirante', 'reparto central', 'Central')


def test_the_three_views_speak_of_three_states_and_never_of_the_central_allocation(odds_site):
    landing = page_of(odds_site, '/es/circunscripciones')[1]
    assert '<h2>La Coruña</h2>' in landing and ('Qué escaños tiene seguros cada partido, cuáles son probables y '
                                                'cuáles están en juego, según las 10.000 simulaciones del modelo.') in landing
    assert re.findall(r'<a href="[^"]*"[^>]*>(Provincias|Partidos|Los más reñidos)</a>', landing) == [
        'Provincias', 'Partidos', 'Los más reñidos']
    assert 'id="seat-kpis"' not in landing and 'so-list' not in landing and 'name="orden"' not in landing
    assert '<title>Demoscopio · Escaños que bailan · Elecciones generales 2026</title>' in landing
    cadiz = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    for chip in ('<span class="chip so-chip safe"><b>5</b> seguros</span>',
                 '<span class="chip so-chip likely"><b>3</b> probables</span>',
                 '<span class="chip so-chip play"><b>1</b> en juego</span>'):
        assert chip in cadiz
    assert ('<i class="so-o likely" style="--c: #1d84ce; --f: rgba(29, 132, 206, 0.475)" '
            'title="3.º de PP: lo consigue en el 80% de las simulaciones"></i>') in cadiz
    assert '<span class="so-pct">80%</span>' in cadiz and '>7%<' not in cadiz  # under 25%: neither drawn nor counted
    assert 'aria-label="PP: 2 seguros; 3.º, 80%"' in cadiz and '<div class="so-note">2 seguros · 1 probable</div>' in cadiz
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan PP, PSOE, VOX y AA.' in cadiz
    legend = cadiz[cadiz.index('class="so-legend"'):]
    assert 'Seguro: 99% o más' in legend and 'Probable: del 75% al 99%' in legend and 'En juego: del 25% al 75%' in legend
    assert 'probable si lo consigue entre el 75% y el 99% y está en juego entre el 25% y el 75%' in cadiz
    form = cadiz[cadiz.index('id="district-pick"'):cadiz.index('</form>')]
    assert re.findall(r'<option value="[a-z-]+"( selected)?>([^<]+)</option>', form) == [
        (' selected', 'Cádiz · 8 escaños'), ('', 'Guadalajara · 3 escaños'), ('', 'La Coruña · 8 escaños')]
    party = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')[1]
    assert 'Tiene 4 escaños seguros, 1 probable y 1 en juego.' in party
    assert '<b>Sólo seguros (1):</b> Guadalajara 1.' in party
    closest = page_of(odds_site, '/es/circunscripciones', 'vista=renidos')[1]
    assert 'Los 8 escaños más cercanos al cara o cruz. La raya marca el 50%.' in closest
    assert closest.count('<tr role="row">') == 1 + 8
    for label, page in (('landing', landing), ('cadiz', cadiz), ('party', party), ('closest', closest)):
        text = visible_text(page[page.index('id="content"'):])
        assert not [word for word in RETIRED if word in text], label
        assert_public_wording(page, label)
    assert call(odds_site, '/es/circunscripciones', query='partido=up')[0] == 404  # a party without options
```

  Los demás tests de página del fichero se adaptan a las cifras nuevas sin perder ninguna comprobación que siga teniendo sentido (modo, run fijado, canónica, 301, 404 y 400, barra del evento, imagen con su línea, enlaces de la página del evento, lecturas de la parte). El `<option>` del desplegable ya no dice «en el aire» ni «decidida».

- [ ] **Step 5: Plantillas y CSS**
  - `_seat_odds.html`: macro `circle` (interfaces); `tokens(row)` pinta por opción `<span class="so-token {state}">` con el círculo y, si no es segura, `<span class="so-pct">{label}</span>` (en negrita si es probable, en gris si está en juego); `chips(item)` pinta una pastilla por estado con opciones (`<span class="chip so-chip {state}"><b>{n}</b> {texto sin la cifra}</span>`); `legend()`: tres muestras en gris con «Seguro: {safe} o más», «Probable: del {likely} al {safe}», «En juego: del {play} al {likely}»; `how()` con el texto de la spec («Textos»), umbrales de `thresholds` y comillas angulares.
  - `circunscripciones.html`: fuera la tira de cifras, la lista y su conmutador; entradilla nueva; pestañas «Provincias», «Partidos», «Los más reñidos»; el desplegable, ordenado por nombre y con el texto «Cádiz · 8 escaños», visible por ahora en todos los anchos, encima de la ficha; fichas sin columna «Central» (cabeceras «Partido», «Escaño a escaño»; en la de partido, «Circunscripción», «Escaños», «Escaño a escaño»), con `s.chips`; la de partido sin «en el reparto central»; «Los más reñidos» sin la columna «Reparto central» ni la línea «Central: ».
  - Descripciones: portada, «Qué escaños tiene seguros cada partido en cada circunscripción, cuáles son probables y cuáles están en juego, según las simulaciones del modelo para {elección}.»; circunscripción, «Los escaños de {nombre}, uno a uno: cuáles tiene seguros cada partido, cuáles son probables y cuáles están en juego, según…»; partido, «Los escaños de {siglas}, circunscripción a circunscripción: cuáles tiene seguros, cuáles son probables y cuáles están en juego, según…»; «Los más reñidos», la de hoy.
  - CSS: `.so-o` (círculo de `--o`, 22 px por defecto, borde `--ob`, 2 px por defecto, `solid var(--c)`; fondo `var(--f, var(--c))`; `.so-o.play { border-style: dashed }`); se borran las reglas que ya nadie usa (`.so-list`, `.so-sort`, `.so-item*`, `.so-md`, `.so-air`, `.so-key*`, `.so-token.open/.out`, `.so-central`, `.so-narrow`, `.so-num` si queda sin uso).

- [ ] **Step 6: Verde** — `pytest tests/test_pages_circunscripciones.py tests/test_pages.py -q`. Esperado: todo verde (el test del método y el del enlace del evento no cambian hasta la Tarea 5).

- [ ] **Step 7: Commit** — «Escaños que bailan v2: tres estados por probabilidad (seguro, probable, en juego), el círculo como signo y fuera el reparto central y la tira de cifras».

---

### Task 2: El catálogo y la pestaña Provincias

**Files:**
- Modify: `data/es-districts.csv`, `mtpy/lib/data.py` (docstring de `get_districts`), `mtpy/lib/webapi/checks.py`, `mtpy/lib/webapi/__init__.py` (exportación)
- Create: `mtpy/lib/pages/seat_charts.py`, `tests/test_pages_seat_charts.py`
- Modify: `mtpy/lib/pages/circunscripciones.py`, `mtpy/lib/pages/__init__.py`, `web/templates/circunscripciones.html`, `web/dist/css/site.css`
- Test: `tests/test_data_files.py`, `tests/test_pages_seat_charts.py`, `tests/test_pages_circunscripciones.py`

**Interfaces:**
- Consumes: `district_views`, `drawn_token`, `scope_totals`, `options_sentence`, `default_district`, `STATES`, `s.circle`, `s.chips`, `s.tokens` (Tarea 1).
- Produces: `webapi.district_positions(scope: str) -> dict`: `{region_id: {'abbr': str, 'col': int, 'row': int, 'w': int, 'h': int}}` de las circunscripciones del ámbito con `abbr`, `map_col` y `map_row` rellenos (`w` y `h` valen 1 si están vacíos); `{}` si la aplicación no tiene directorio de datos, el fichero no trae las columnas o ninguna está rellena. Con el TTL de `catalogue()` (misma caché, clave `('districts', scope)`). Lee el CSV con `keep_default_na=False, na_values=['']`: la abreviatura de Navarra es «NA», que pandas toma por un valor vacío si no se le dice lo contrario.
- Produces: `pages.scope_options(districts: list) -> list`: todas las opciones del ámbito, cada ficha con `party` y `district` (nombre), en el orden de las circunscripciones y de sus filas.
- Produces: `pages.map_tiles(districts: list, positions: dict, order: list) -> dict`: `{'placed': bool, 'tiles': list}`. `placed` sólo si **todas** las circunscripciones tienen posición. Casilla: `{'id', 'name', 'slug', 'seats', 'label', 'col', 'row', 'w', 'h', 'options'}`; `options` son sus fichas con `party`, ordenadas por partido (posición en `order`), estado (`STATES`) y `k`. Con posiciones: `label` es la abreviatura y las casillas van por fila y columna. Sin ellas: `label` es el nombre, `col`, `row`, `w` y `h` son `None` y van por nombre (`_sort_name`).
- Produces (marcado que usan las Tareas 3 y 4): envoltorio `<div id="provincias">`; tarjeta del mapa `<section class="card shot so-card so-map-card" id="mapa">`; selectores `<fieldset class="seg so-filter" id="map-party">` (radios `name="mp"`, `id="mp-all"`, `id="mp-{slug del partido}"`) y `<fieldset class="seg so-filter" id="map-state">` (radios `name="ms"`, `id="ms-all|safe|likely|play"`); casilla `<label class="so-tile">` con su radio `<input type="radio" name="d" value="{slug}" id="d-{slug}" aria-label="{nombre}, {n} escaños">`; fichas `<section class="card shot so-card so-ficha" id="ficha-{slug}">` dentro de `<div id="ficha" class="so-fichas">`.

- [ ] **Step 1: Test del catálogo que falla** — en `tests/test_data_files.py`:

```python
def test_the_districts_of_es_have_a_map_tile_each_and_no_two_overlap():
    # «NA» (Navarra) is an abbreviation, not a missing value
    d = pd.read_csv(os.path.join(DATA, 'es-districts.csv'), keep_default_na=False, na_values=[''])
    es, tiled = d.loc[d['scope'] == 'es'], ['abbr', 'map_col', 'map_row', 'map_w', 'map_h']
    assert es[tiled[:3]].notnull().all().all() and es['abbr'].is_unique
    assert es.set_index('region_id').loc[31, 'abbr'] == 'NA'
    assert d.loc[d['scope'] != 'es', tiled].isnull().all().all()
    cells = {}
    for row in es.itertuples():
        width, height = (int(v) if pd.notnull(v) else 1 for v in (row.map_w, row.map_h))
        for col in range(int(row.map_col), int(row.map_col) + width):
            for line in range(int(row.map_row), int(row.map_row) + height):
                assert 0 <= col < 10 and (col, line) not in cells, (row.name, cells.get((col, line)))
                cells[(col, line)] = row.name
    tile = es.set_index('region_id')
    assert tuple(tile.loc[28, tiled]) == ('M', 4, 3, 2, 2) and tuple(tile.loc[8, tiled]) == ('B', 8, 1, 2, 2)
    assert tuple(tile.loc[46, tiled]) == ('V', 7, 5, 2, 1) and tuple(tile.loc[11, tiled[:3]]) == ('CA', 3, 8)
    assert tuple(tile.loc[15, tiled[:3]]) == ('C', 0, 0) and tuple(tile.loc[52, tiled[:3]]) == ('ML', 6, 8)
```

- [ ] **Step 2: El catálogo** — añadir a `data/es-districts.csv` las columnas `abbr,map_col,map_row,map_w,map_h` al final, rellenas sólo en `es` con la tabla «Datos» de la spec (por código INE, que en `es` es el `region_id`); `map_w` y `map_h` sólo en Barcelona, Madrid (2 y 2) y Valencia (2 y 1). Editar el fichero como texto, sin reescribirlo con pandas: cada una de sus 119 líneas sólo gana su cola (`,,,,,` fuera de `es`) y ninguna otra celda cambia (comprobarlo con `git diff --word-diff`). Actualizar el docstring de `get_districts`. `pytest tests/test_data_files.py tests/test_loader_unit.py -q` en verde.

- [ ] **Step 3: Tests que fallan (colocación)** — `tests/test_pages_seat_charts.py`:

```python
"""La colocación de los dibujos de «Escaños que bailan»: casillas del mapa, plazas del hemiciclo y columnas."""
from mtpy.lib import pages
from tests.test_pages_circunscripciones import ODDS

POSITIONS = {11: {'abbr': 'CA', 'col': 3, 'row': 8, 'w': 1, 'h': 1}, 19: {'abbr': 'GU', 'col': 6, 'row': 3, 'w': 1, 'h': 1},
             15: {'abbr': 'C', 'col': 0, 'row': 0, 'w': 1, 'h': 1}}


def test_map_tiles_place_each_district_and_sort_its_options():
    views = pages.district_views(ODDS)
    grid = pages.map_tiles(views, POSITIONS, ODDS['parties'])
    assert grid['placed'] and [tile['label'] for tile in grid['tiles']] == ['C', 'GU', 'CA']  # by row, then column
    cadiz = grid['tiles'][2]
    assert (cadiz['name'], cadiz['slug'], cadiz['col'], cadiz['row'], cadiz['w'], cadiz['h']) == ('Cádiz', 'cadiz', 3, 8, 1, 1)
    assert [(o['party'], o['state'], o['k']) for o in cadiz['options']] == [
        ('PP', 'safe', 1), ('PP', 'safe', 2), ('PP', 'likely', 3), ('PSOE', 'safe', 1), ('PSOE', 'safe', 2),
        ('PSOE', 'play', 3), ('VOX', 'safe', 1), ('VOX', 'likely', 2), ('AA', 'likely', 1)]
    assert len(pages.scope_options(views)) == 22 and pages.scope_options(views)[0]['district'] == 'Cádiz'


def test_a_scope_without_every_position_draws_its_tiles_in_a_row_by_name():
    views = pages.district_views(ODDS)
    for positions in ({}, {11: POSITIONS[11]}):  # no catalogue columns; one district placed and two not
        grid = pages.map_tiles(views, positions, ODDS['parties'])
        assert not grid['placed'] and [tile['label'] for tile in grid['tiles']] == ['Cádiz', 'Guadalajara', 'La Coruña']
        assert all(tile['col'] is None and tile['w'] is None for tile in grid['tiles'])
    empty = dict(ODDS, rows=[row for row in ODDS['rows'] if row['region_id'] != 19])  # a district without options
    tiles = pages.map_tiles(pages.district_views(empty), POSITIONS, ODDS['parties'])['tiles']
    assert [(tile['label'], len(tile['options'])) for tile in tiles] == [('C', 9), ('GU', 0), ('CA', 9)]
```

- [ ] **Step 4: `district_positions`, `scope_options` y `map_tiles`** — según las interfaces. `seat_charts.py` no importa nada de `circunscripciones.py` (sería un ciclo): `STATES` se muda a `seat_charts.py` y `circunscripciones.py` lo importa de ahí, junto a las funciones de los dibujos (`pages` lo sigue exportando); `scope_options` vive en `circunscripciones.py`. El contexto importa `district_positions` con `from ..webapi import district_positions, site`. Test de la lectura, en `tests/test_pages_circunscripciones.py`:

```python
def test_the_positions_of_the_map_come_from_the_catalogue(api):
    positions = webapi.district_positions('es')
    assert len(positions) == 52 and positions[31] == {'abbr': 'NA', 'col': 6, 'row': 0, 'w': 1, 'h': 1}
    assert positions[28] == {'abbr': 'M', 'col': 4, 'row': 3, 'w': 2, 'h': 2} and webapi.district_positions('es-an') == {}
    assert all(type(positions[28][key]) is int for key in ('col', 'row', 'w', 'h'))
```

  `pytest tests/test_pages_seat_charts.py tests/test_pages_circunscripciones.py -q -k "tiles or scope_without or positions"` en verde.

- [ ] **Step 5: Tests que fallan (la página)** — en `tests/test_pages_circunscripciones.py`:

```python
def test_the_map_draws_every_district_as_the_label_of_a_radio_and_every_card_is_on_the_page(odds_site):
    page = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert '<h2>Dónde bailan los escaños, provincia a provincia</h2>' in page
    grid = page[page.index('class="so-map"'):page.index('class="so-legend"')]
    assert re.findall(r'name="d" value="([a-z-]+)"', grid) == ['la-coruna', 'guadalajara', 'cadiz']  # by row and column
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', grid) == ['cadiz']
    assert 'title="Cádiz · 8 escaños" style="grid-column: 4 / span 1; grid-row: 9 / span 1"' in grid
    assert '<span class="so-ab">CA</span>' in grid and 'aria-label="Cádiz, 8 escaños"' in grid
    assert grid.count('class="so-o ') == 22 and grid.count('class="so-o play p-') == 4
    assert '<i class="so-o safe p-pp" style="--c: #1d84ce" title="1.º de PP: seguro"></i>' in grid
    assert re.findall(r'name="mp" value="([a-z-]+)"', page) == ['all', 'pp', 'psoe', 'vox', 'aa', 'bng']  # never UP
    states = page[page.index('id="map-state"'):page.index('class="so-map"')]
    assert re.findall(r'>([^<>]+) <small>(\d+)</small>', states) == [
        ('Todos', '22'), ('Seguros', '14'), ('Probables', '4'), ('En juego', '4')]
    # the rules of the run: one per party for the filter, one per district for its card
    assert '#mapa:has(#mp-psoe:checked) .so-dots i:not(.p-psoe){opacity:.08}' in page
    assert '#provincias:has(#d-cadiz:checked) #ficha-cadiz{display:block}' in page
    assert re.findall(r'class="card shot so-card so-ficha" id="ficha-([a-z-]+)"', page) == ['cadiz', 'guadalajara', 'la-coruna']
    assert 'Hay 22 opciones para 19 escaños: a un mismo escaño pueden optar varios partidos.' in page
    assert '<title>Demoscopio · Escaños que bailan en Cádiz · Elecciones generales 2026</title>' in page
    landing = page_of(odds_site, '/es/circunscripciones')[1]
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', landing) == ['la-coruna']
    text = visible_text(page[page.index('id="content"'):])
    assert not [word for word in RETIRED if word in text]
    assert_public_wording(page, 'provincias')


def test_the_map_without_positions_and_the_foot_without_more_options_than_seats(odds_site, fresh_app, monkeypatch):
    monkeypatch.setattr('mtpy.lib.pages.circunscripciones.district_positions', lambda scope: {})
    page = page_of(odds_site, '/es/circunscripciones')[1]
    loose = page[page.index('class="so-map so-map-loose"'):page.index('class="so-legend"')]
    assert re.findall(r'<span class="so-ab">([^<]+)</span>', loose) == ['Cádiz', 'Guadalajara', 'La Coruña']
    assert 'grid-column' not in loose
    monkeypatch.undo()
    safe_only = dict(ODDS, rows=[odds_row(row['region_id'], row['name'], row['seats'], *[n for n in row['reach'] if n >= 9900])
                                 for row in ODDS['rows']])
    write_odds(fresh_app.fs, safe_only)  # 14 options for 19 seats, and districts where nobody disputes what is left
    page = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert 'opciones para' not in page and 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1).' in page
    assert 'se los disputan' not in page
```

- [ ] **Step 6: La pestaña** — en `seat_odds_context`: el mapa (`map_tiles` con las opciones pasadas por `drawn_token`), las fichas de todas las circunscripciones (la tarjeta de hoy, una por circunscripción, por nombre), los selectores (partidos con alguna opción, en el orden de la parte, con `slug`, siglas, nombre completo y color; los cuatro estados con los recuentos de `scope_totals`) y `options_sentence`. En la plantilla, vista `district`:
  - el bloque `<style>` generado va en el bloque `scripts` de la plantilla (está en el `<head>`), sin espacios, con las dos reglas del test por partido y por circunscripción;
  - tarjeta `#mapa`: botón de imagen (`demoscopio-escanos-{scope}-mapa.png`), título, entradilla de la spec, los dos selectores (`<legend>` «Partido» y «Estado»; «Todos» marcado en ambos; cada partido con su punto de color, sus siglas y su nombre completo en el `title`; cada estado con su recuento en `<small>`), la línea `so-stamp`, el mapa (`<div class="so-map" role="radiogroup" aria-label="Circunscripciones">`; con `placed` falso, `class="so-map so-map-loose"` y sin `grid-column`), la leyenda y el pie (`<p class="so-foot">`, sólo si la frase no está vacía);
  - casilla: `<label class="so-tile" title="{nombre} · {n} escaño(s)" style="grid-column: {col+1} / span {w}; grid-row: {row+1} / span {h}">` con su radio (marcado el de la circunscripción de la dirección o la de `default_district`), `<span class="so-ab">{label}</span>` y `<span class="so-dots" aria-hidden="true">` con un `s.circle(token, 'p-' ~ slug del partido)` por opción;
  - el desplegable entre el mapa y las fichas, visible sólo por debajo de 720 px;
  - `<div id="ficha" class="so-fichas">` con todas las fichas; cada una, completa (botón de imagen con su fichero, cabecera, `so-stamp`, frase, filas, leyenda y «Cómo leerlo»). Título y descripción, los de la circunscripción de la dirección.
  - CSS, de la maqueta: `.so-map` (rejilla de 10 columnas, máximo 760 px, centrada; `.so-map-loose`: `repeat(auto-fill, minmax(150px, 1fr))`), `.so-tile` (radio oculto como en `.seg input`; `:has(input:checked)` con borde oscuro de 2 px; `:has(input:focus-visible)` con el contorno de foco), `.so-dots` (`--o: 11px; --ob: 1px`, separación de 2 px; por debajo de 720 px, 7 px y 1,5 px), `.so-ficha { display: none }`, y las tres reglas fijas del selector de estado (`#mapa:has(#ms-safe:checked) .so-dots i:not(.safe) { opacity: .08 }` y sus dos hermanas). `#mapa` entra en la regla de `scroll-margin`.

- [ ] **Step 7: Verde** — `pytest tests/test_pages_circunscripciones.py tests/test_pages_seat_charts.py tests/test_data_files.py tests/test_pages.py -q`.

- [ ] **Step 8: Commit** — «Escaños que bailan v2: mapa de provincias con selector de partido y de estado, fichas que cambian en el sitio y posiciones de las 52 circunscripciones en el catálogo».

---

### Task 3: La pestaña Partidos: hemiciclo, columnas y ficha del partido

**Files:**
- Modify: `mtpy/lib/pages/seat_charts.py`, `mtpy/lib/pages/circunscripciones.py`, `mtpy/lib/pages/__init__.py`, `mtpy/controllers/Circunscripciones.py` (docstrings: `vista` admite `partidos`)
- Modify: `web/templates/circunscripciones.html`, `web/dist/css/site.css`
- Test: `tests/test_pages_seat_charts.py`, `tests/test_pages_circunscripciones.py`

**Interfaces:**
- Consumes: `scope_options`, `drawn_token`, `state_text`, `options_sentence`, `hemicycle_order` (de `common.py`), `context['meta']['bmaps'].get('blocks')`.
- Produces: `pages.hemicycle_layout(options: list, order: list) -> dict`: `{'rows': int, 'radius': float, 'seats': list, 'parties': list}`; cada plaza es la ficha de su opción más `x`, `y` (1 decimal), `row` (0 la interior) y `angle`; `parties` son los de `order` con alguna opción, `{'name', 'safe', 'likely', 'play'}`.
- Produces: `pages.party_columns(options: list, order: list, blocks: Optional[dict]) -> list`: `{'name', 'members', 'safe', 'likely', 'play', 'options'}` por columna; `options` por estado, partido (posición en `order`) y probabilidad descendente. Con `blocks` válido (trae `Izquierda` **y** `Derecha`): una columna por partido de esos dos bloques con alguna opción, en ese orden, y una última `'Otros'` con todos los demás partidos con opciones (`members`), si los hay. Sin `blocks` válido: una columna por partido con opciones, en el orden de `order`.
- Produces: `pages.column_ticks(columns: list) -> list`: `[50, 100, …]` hasta el tamaño de la columna más alta.
- Produces (marcado que usa la Tarea 4): tarjeta `<section class="card shot so-card so-chart-card" id="grafico">`; conmutador `<fieldset class="seg so-filter" id="chart-form">` con radios `name="pf"`, `id="pf-hemi"` (marcado) e `id="pf-cols"`; todo lo que pertenece a un dibujo lleva la clase `so-form-hemi` o `so-form-cols`.

- [ ] **Step 1: Tests que fallan (colocación)** — en `tests/test_pages_seat_charts.py`:

```python
from collections import Counter

ORDER = ['PSOE', 'PP', 'VOX', 'AA', 'BNG']  # the order of the hemicycle of the web for the test catalogue


def test_the_hemicycle_seats_every_option_once_party_by_party_and_from_the_inside_out():
    options = pages.scope_options(pages.district_views(ODDS))
    layout = pages.hemicycle_layout(options, ORDER)
    seats = layout['seats']
    assert len(seats) == 22 == len({(seat['x'], seat['y']) for seat in seats})
    assert (layout['rows'], layout['radius'], Counter(seat['row'] for seat in seats)) == (2, 22.0, {0: 6, 1: 16})
    assert [(p['name'], p['safe'], p['likely'], p['play']) for p in layout['parties']] == [
        ('PSOE', 4, 1, 1), ('PP', 6, 1, 1), ('VOX', 2, 1, 1), ('AA', 0, 1, 0), ('BNG', 2, 0, 1)]
    wedges = [[seat for seat in seats if seat['party'] == name] for name in ORDER]
    for left, right in zip(wedges, wedges[1:]):  # each party sits to the left of the next one
        assert min(seat['angle'] for seat in left) >= max(seat['angle'] for seat in right)
    for wedge in wedges:  # no option sits in a row further in than a likelier one of its party
        assert all(a['share'] >= b['share'] for a in wedge for b in wedge if a['row'] < b['row'])


def test_the_hemicycle_holds_with_three_options_and_with_390():
    option = {'k': 1, 'share': 1.0, 'state': 'safe', 'label': None, 'alpha': None, 'title': '', 'district': 'X'}
    few = pages.hemicycle_layout([dict(option, party='A')] * 2 + [dict(option, party='B')], ['A', 'B'])
    assert (few['rows'], len(few['seats']), len({(s['x'], s['y']) for s in few['seats']})) == (1, 3, 3)
    assert pages.hemicycle_layout([], ['A'])['seats'] == []
    many = pages.hemicycle_layout([dict(option, party='A')] * 200 + [dict(option, party='B')] * 190, ['A', 'B'])
    assert (many['rows'], many['radius']) == (10, 9.7)
    assert Counter(seat['row'] for seat in many['seats']) == {0: 23, 1: 26, 2: 30, 3: 34, 4: 37, 5: 41, 6: 44, 7: 48,
                                                              8: 52, 9: 55}
    assert all(0 <= seat['x'] <= 1000 and 20 <= seat['y'] <= 540 for seat in many['seats'])


def test_the_columns_follow_the_blocks_of_the_run_and_fall_back_to_one_per_party():
    options = pages.scope_options(pages.district_views(ODDS))
    blocks = {'Izquierda': ['PSOE', 'AA', 'FA'], 'Derecha': ['PP', 'VOX'], 'Otros': []}  # BNG is in no block; FA has no option
    columns = pages.party_columns(options, ODDS['parties'], blocks)
    assert [(c['name'], c['safe'], c['likely'], c['play']) for c in columns] == [
        ('PSOE', 4, 1, 1), ('AA', 0, 1, 0), ('PP', 6, 1, 1), ('VOX', 2, 1, 1), ('Otros', 2, 0, 1)]
    assert columns[-1]['members'] == ['BNG'] and sum(len(c['options']) for c in columns) == 22
    assert [o['state'] for o in columns[2]['options']] == ['safe'] * 6 + ['likely', 'play']
    bare = pages.party_columns(options, ODDS['parties'], None)
    assert [c['name'] for c in bare] == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert pages.party_columns(options, ODDS['parties'], {'Derecha': ['PP']}) == bare  # the blocks of the test bundle
    assert pages.column_ticks(columns) == [] and pages.column_ticks([{'options': [0] * 121}, {'options': [0] * 49}]) == [50, 100]
```

- [ ] **Step 2: `hemicycle_layout`, `party_columns`, `column_ticks`** — el hemiciclo, con esta colocación (centro `(500, 520)`, radios de 190 a 460, `viewBox="0 20 1000 520"`):

```python
total = len(options)
rows = max(1, min(12, round((total * 0.26) ** 0.5)))
radii = [460.0] if rows == 1 else [190 + 270 * index / (rows - 1) for index in range(rows)]
exact = [total * radius / sum(radii) for radius in radii]
per_row = [int(value) for value in exact]
# largest remainders; on a tie, the outer row
for row in sorted(range(rows), key=lambda row: (per_row[row] - exact[row], -row))[:total - sum(per_row)]:
    per_row[row] += 1
places = [{'row': row, 'r': radii[row], 'angle': math.pi - (index + 0.5) * math.pi / per_row[row]}
          for row in range(rows) for index in range(per_row[row])]
places.sort(key=lambda place: (-place['angle'], place['row']))  # left to right
# each party of `order` takes the next len(mine) places; inside its wedge, sorted by (row, -angle), they are
# filled with its options sorted by (STATES.index(state), -share): safe at the core, in play at the rim
# x = 500 + r * cos(angle), y = 520 - r * sin(angle)
arcs = [math.pi * radii[row] / per_row[row] for row in range(rows) if per_row[row]]
gap = min(arcs + ([270 / (rows - 1)] if rows > 1 else []))
radius = round(min(0.375 * gap, 22), 1)
```

  Sin opciones devuelve `{'rows': 0, 'radius': 0.0, 'seats': [], 'parties': []}`.

- [ ] **Step 3: Verde** — `pytest tests/test_pages_seat_charts.py -q`.

- [ ] **Step 4: Tests que fallan (la página)** — en `tests/test_pages_circunscripciones.py`:

```python
def write_blocks(fs, blocks):
    """Sobrescribe la parte `meta` del run del bundle de prueba con estos bloques de la web."""
    meta = fixture_data('meta')
    meta = meta | {'bmaps': meta['bmaps'] | {'blocks': blocks}}
    bundle.BundleWriter(fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()


def test_the_parties_tab_draws_the_hemicycle_and_the_columns_and_opens_on_the_hemicycle(odds_site, fresh_app):
    status, page = page_of(odds_site, '/es/circunscripciones', 'vista=partidos')
    assert status == 200 and '<title>Demoscopio · Escaños que bailan, partido a partido · Elecciones generales 2026</title>' in page
    assert re.findall(r'<a href="[^"]*"( class="on" aria-current="page")?>(Provincias|Partidos|Los más reñidos)</a>', page) == [
        ('', 'Provincias'), (' class="on" aria-current="page"', 'Partidos'), ('', 'Los más reñidos')]
    assert 'href="/es/circunscripciones?mode=forecast&amp;vista=partidos#vistas"' in page
    assert re.findall(r'name="pf" value="(hemi|cols)"[^>]*?( checked)?>', page) == [('hemi', ' checked'), ('cols', '')]
    assert '<h2 class="so-form-hemi">El hemiciclo que baila</h2>' in page
    assert '<h2 class="so-form-cols">Las opciones de escaño de cada partido</h2>' in page
    svg = page[page.index('<svg class="so-hemi-svg"'):page.index('</svg>')]
    assert 'viewBox="0 20 1000 520" role="img" aria-label="Hemiciclo con las opciones de escaño de cada partido según su probabilidad"' in svg
    assert svg.count('<circle') == 22 and svg.count('stroke-dasharray') == 4 and svg.count('<title>') == 22
    for count in ('<b>14</b> seguros', '<b>4</b> probables', '<b>4</b> en juego'):
        assert count in page[page.index('class="so-hemi-core"'):]
    assert re.findall(r'<b>([A-Z]+)</b><em>([^<]+)</em>', page) == [
        ('PSOE', '4 · 1 · 1'), ('PP', '6 · 1 · 1'), ('VOX', '2 · 1 · 1'), ('AA', '0 · 1 · 0'), ('BNG', '2 · 0 · 1')]
    assert ('Tras cada partido: seguros · probables · en juego. Hay 22 opciones para 19 escaños: a un mismo escaño '
            'pueden optar varios partidos.') in page
    # the blocks of the test bundle are incomplete: one column per party and no «Otros»
    assert re.findall(r'class="so-col" data-col="([^"]+)"', page) == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert 'href="/es/circunscripciones?mode=forecast&amp;partido=psoe#partidos"' in page and 'id="ficha"' not in page
    assert '350' not in visible_text(page[page.index('id="grafico"'):page.index('id="partidos"')])  # never the chamber
    assert 'mayoría' not in page[page.index('id="grafico"'):page.index('id="partidos"')]
    write_blocks(fresh_app.fs, {'Izquierda': ['PSOE', 'AA'], 'Derecha': ['PP', 'VOX'], 'Otros': ['BNG']})
    page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')[1]
    assert re.findall(r'class="so-col" data-col="([^"]+)"', page) == ['PSOE', 'AA', 'PP', 'VOX', 'Otros']
    assert 'Bajo cada partido: seguros · probables · en juego. Otros:' in page
    assert 'id="ficha"' in page and 'Tiene 4 escaños seguros, 1 probable y 1 en juego.' in page
    text = visible_text(page[page.index('id="content"'):])
    assert not [word for word in RETIRED if word in text]
    assert_public_wording(page, 'partidos')
    assert call(odds_site, '/es/circunscripciones', query='vista=otra')[0] == 400
```

  Añadir a la tabla del test de la canónica el caso `('/es/circunscripciones', 'vista=partidos&mode=nowcast', 'https://demoscopio.es/es/circunscripciones?vista=partidos')`.

- [ ] **Step 5: La pestaña** — `seat_odds_context` admite `view == 'partidos'` (la vista `'party'` sin partido; cualquier otro valor distinto de `renidos`, 400 `invalid view`); título «Escaños que bailan, partido a partido» y descripción «Las opciones de escaño de cada partido en un hemiciclo y en columnas: seguras, probables y en juego, según las simulaciones del modelo para {elección}.»; la pestaña Partidos enlaza `?vista=partidos#vistas` y está marcada con o sin partido; los botones de partido enlazan `?partido={slug}#partidos`. El orden del hemiciclo es `hemicycle_order(partidos con opciones, catalog)`. En la plantilla, vista `party`:
  - tarjeta `#grafico`: botón de imagen (`demoscopio-escanos-{scope}-hemiciclo.png`); los dos títulos y las dos entradillas de la spec, cada uno con su clase de dibujo; el conmutador («Hemiciclo», «Columnas»); `so-stamp`;
  - hemiciclo (`so-form-hemi`): `<div class="so-hemi">` con el `<svg class="so-hemi-svg">` y, encima, `<div class="so-hemi-core">` con los tres recuentos (`s.circle` en gris, `<b>{n}</b>` y la palabra de `state_text`); cada plaza es un `<circle>` con atributos de presentación y su `<title>`: segura, `fill` del color; probable, `fill` de `fill`, `stroke` del color; en juego, lo mismo con `stroke-dasharray`; el radio de las no seguras, 0,9 menos; grosor y trazo proporcionales al radio (con 9,7: 1,6, 1,9 y «3.1 2.5», los de la maqueta). Debajo, `<div class="so-plist">` con una pastilla por partido (`<b>{siglas}</b><em>{seguros} · {probables} · {en juego}</em>`) y el pie;
  - columnas (`so-form-cols`): `<div class="so-cols" data-shot-fit>` (desplazamiento horizontal si no caben) con las rayas de `column_ticks` (`bottom: calc({valor} / var(--wn) * (var(--wsq) + var(--wgap)))`) y una `<div class="so-col" data-col="{name}">` por columna, apilada de abajo arriba con `flex-wrap: wrap-reverse` y `flex: none`, 7 por fila (`--wn: 7`; 5 por debajo de 720 px), cada círculo en el color de su partido; debajo, las siglas con su punto (sin punto en «Otros») y `<small>{seguros} · {probables} · {en juego}</small>`; la leyenda y el pie («Bajo cada partido: seguros · probables · en juego.» y, sólo con columna «Otros», « Otros: » con el punto y las siglas de sus miembros);
  - `<nav class="seg so-pills" id="partidos" aria-label="Partidos">` y, sólo con `?partido=`, la ficha del partido (`id="ficha"`, la de la Tarea 1);
  - CSS: `.so-form-cols { display: none }` y su inversa con `#grafico:has(#pf-cols:checked)`; `.so-hemi` (relativo, máximo 900 px), `.so-hemi-core` (centrado abajo, dentro del hueco; letra de 12 px, 11 px por debajo de 720 px); columnas de la maqueta (`.waffle`, `.wcol`, `.tick`, `.wlabels`). `#partidos` y `#grafico` entran en la regla de `scroll-margin`.

- [ ] **Step 6: Verde** — `pytest tests/test_pages_circunscripciones.py tests/test_pages_seat_charts.py tests/test_pages.py -q`.

- [ ] **Step 7: Commit** — «Escaños que bailan v2: pestaña Partidos con el hemiciclo de las opciones de escaño, las columnas por bloque y su conmutador (`?vista=partidos`)».

---

### Task 4: Las imágenes y el script de la página

**Files:**
- Create: `web/dist/js/seat-odds.js`, `tests/test_js_seat_odds.py`
- Modify: `web/dist/js/pages/circunscripciones.js`, `web/templates/circunscripciones.html`, `web/dist/css/site.css`
- Test: `tests/test_js_seat_odds.py`, `tests/test_pages_circunscripciones.py`

**Interfaces:**
- Consumes: el marcado de las Tareas 2 y 3 (`#mapa`, `#map-party`, `#map-state`, radios `mp`, `ms`, `d` y `pf`, `#grafico`, `#district-pick`, `#district-select`); `wireCardImages` y `STRIPPED` de `card-image.js` (no se tocan).
- Produces: `seat-odds.js`, sin importaciones: `imageFile(base: string, parts: string[]): string` (une con `-` las partes no vacías y añade `.png`) y `filterLine(party: string, state: string): string` (une con « · » las no vacías).

- [ ] **Step 1: Tests que fallan** — `tests/test_js_seat_odds.py` (con el arnés de `tests/test_js_card_image.py`, sin ECharts: el módulo no importa nada):

```python
def test_the_image_is_named_after_what_is_on_view():
    assert run_js("s.imageFile('demoscopio-escanos-es-mapa', ['psoe', 'en-juego'])") == 'demoscopio-escanos-es-mapa-psoe-en-juego.png'
    assert run_js("s.imageFile('demoscopio-escanos-es-mapa', ['', ''])") == 'demoscopio-escanos-es-mapa.png'
    assert run_js("s.imageFile('demoscopio-escanos-es', ['columnas'])") == 'demoscopio-escanos-es-columnas.png'
    assert run_js("[s.filterLine('PSOE', 'En juego'), s.filterLine('PSOE', ''), s.filterLine('', 'Seguros'), "
                  "s.filterLine('', '')]") == ['PSOE · En juego', 'PSOE', 'Seguros', '']
```

  En `tests/test_pages_circunscripciones.py`:

```python
def test_the_cards_carry_what_the_script_needs_to_name_their_images_and_change_the_address(odds_site):
    page = page_of(odds_site, '/es/circunscripciones')[1]
    assert 'id="mapa" data-file-base="demoscopio-escanos-es-mapa"' in page
    assert page.count('data-file="demoscopio-escanos-es-mapa.png"') == 1
    assert re.findall(r'data-file="demoscopio-escanos-es-([a-z-]+)\.png"', page) == ['mapa', 'cadiz', 'guadalajara', 'la-coruna']
    assert '<p class="so-filter-line shot-only" id="map-filter-line" hidden></p>' in page
    assert 'id="mp-all" data-slug="" data-label=""' in page and 'id="mp-psoe" data-slug="psoe" data-label="PSOE"' in page
    assert 'id="ms-play" data-slug="en-juego" data-label="En juego"' in page
    assert ('id="d-cadiz" data-href="/es/circunscripciones/cadiz?mode=forecast" '
            'data-title="Demoscopio · Escaños que bailan en Cádiz · Elecciones generales 2026"') in page
    assert page.count('class="so-stamp shot-only"') == 1 + 3  # the map and every district card
    parties = page_of(odds_site, '/es/circunscripciones', 'vista=partidos')[1]
    assert 'id="grafico" data-file-base="demoscopio-escanos-es"' in parties
    assert 'data-file="demoscopio-escanos-es-hemiciclo.png"' in parties
    assert 'id="pf-hemi" data-slug="hemiciclo"' in parties and 'id="pf-cols" data-slug="columnas"' in parties
    pinned = page_of(odds_site, '/es/circunscripciones', 'mode=nowcast&run=' + RUN)[1]
    assert 'data-href="/es/circunscripciones/cadiz?mode=nowcast&amp;run={}"'.format(RUN) in pinned
    # the copy the image is drawn from hides the selectors and never removes them: the filter lives in their radios
    css = open(os.path.join(os.path.dirname(__file__), '..', 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert re.search(r'\.shot-image \.so-filter\s*\{\s*display:\s*none', css)
```

- [ ] **Step 2: Ejecutarlos y ver que fallan** — `pytest tests/test_js_seat_odds.py tests/test_pages_circunscripciones.py -q -k "image or cards_carry"`.

- [ ] **Step 3: Marcado y CSS** — los atributos del test, en ese orden tras el `id` (los slugs de estado son `seguros`, `probables`, `en-juego`; sus etiquetas, «Seguros», «Probables», «En juego»); la línea `#map-filter-line` bajo el `so-stamp` del mapa. CSS de la copia de la imagen: `.shot-image .so-filter { display: none }` (se **ocultan**, no se quitan: el estado de `:has()` vive en los radios; por eso no entran en `STRIPPED`), `.shot-image .so-tile:has(input:checked)` sin el borde oscuro, `.shot-image .so-pick { display: none }`.

- [ ] **Step 4: El script** — `circunscripciones.js`, cada parte en su `tryDraw`:
  - radios `d` (`change`): `history.replaceState(null, '', radio.dataset.href)`, `document.title = radio.dataset.title` y `#district-select` con su valor;
  - `#district-pick`: clase `js` (oculta «Ver»), sin envío; al cambiar el desplegable, marca el radio de ese valor y lanza su `change`. Ya no se usa `wireForm`;
  - radios `mp` y `ms` (`change`, y una vez al montar): `#map-filter-line` con `filterLine(etiqueta del partido, etiqueta del estado)` y `hidden` si queda vacía; el botón de imagen de `#mapa` con `data-file = imageFile(base, [slug del partido, slug del estado])`;
  - radios `pf` (`change`, y una vez al montar): el botón de `#grafico` con `imageFile(base, [slug del dibujo])`;
  - `pageshow` con `event.persisted`: repetir las dos sincronizaciones;
  - `wireCardImages(document, showError)`, como hoy.

- [ ] **Step 5: Verde** — `pytest tests/test_js_seat_odds.py tests/test_js_card_image.py tests/test_pages_circunscripciones.py -q`.

- [ ] **Step 6: Commit** — «Escaños que bailan v2: imágenes de las cuatro tarjetas con lo que está a la vista (sin selectores, con el partido y el estado elegidos) y dirección que cambia al tocar una casilla».

---

### Task 5: Los textos de fuera de la página: enlace del evento, método y contrato

**Files:**
- Modify: `web/templates/_macros.html` (`seat_odds_link`), `web/templates/metodo_escanos.html` (apartado `#seguros`), `docs/web/contrato.md` (párrafo de «Escaños que bailan», hacia la línea 783)
- Test: `tests/test_pages_circunscripciones.py`, `tests/test_pages.py`

- [ ] **Step 1: Tests que fallan** — en `test_the_results_card_links_the_page_at_the_foot_of_its_tables`, el texto del enlace pasa a `'Escaños que bailan: seguros, probables y en juego ›</a></p>'`. En `tests/test_pages.py`, las líneas del apartado `#seguros` (hacia la 3575) pasan a:

```python
    block = page[page.index('<h3 id="seguros">'):page.index('id="comprobacion"')]
    assert '<h3 id="seguros">Escaños seguros, probables y en juego</h3>' in block
    assert 'probable al que sale entre el 75% y el 99%' in block and 'el que sale entre el 25% y el 75%' in block
    assert 'las opciones suman más que los escaños' in block and 'no es una garantía' in block
    assert not [word for word in ('posible', 'en el aire', 'reparto central', 'descartado') if word in block]
```

- [ ] **Step 2: Ejecutarlos y ver que fallan.**

- [ ] **Step 3: Los textos** — el enlace y los dos párrafos del método, tal cual en la spec («Textos»), con comillas angulares. En el contrato: la portada abre en la circunscripción con más opciones en juego; `vista` admite `partidos` y `renidos`; la pestaña Provincias pinta todas las fichas y cambia entre ellas sin recargar; la página lee `data/es-districts.csv` para las posiciones del mapa (sin ellas, rejilla seguida); la canónica conserva `vista=partidos`.

- [ ] **Step 4: Toda la batería** — `pytest -q`. Esperado: todo verde, con los mismos tests saltados que en `dev`.

- [ ] **Step 5: Commit** — «Escaños que bailan v2: enlace del evento, apartado del método y contrato con los tres estados».

---

### Task 6: Comprobación en el navegador, peso y cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-11-13-escanos-que-bailan-mapas-design.md` (sección «Estado al cierre», al final); lo que la comprobación obligue a corregir en plantilla, CSS o script (con su test si es de comportamiento).

- [ ] **Step 1: Datos reales en un bundle temporal** — fuera del repositorio (carpeta temporal), el bundle de prueba (`tests/test_webapi_unit.write_bundle`) con la parte `meta` y las `seat-odds` de los dos modos del run publicado de `es`, leídas de la API pública de demoscopio.es (sólo GET, sin credenciales), y servido en local con uvicorn con `data` apuntando al `data/` del worktree. Nunca `publish_forecast`, nunca S3, nunca el `.env`. Para el ámbito autonómico: un ámbito con la parte publicada y varias circunscripciones (`es-an`, `es-cl`); si ninguno la tiene aún, la parte de `es` reducida a las ocho provincias andaluzas con `district_positions` devolviendo `{}`.

- [ ] **Step 2: Comprobar en Chrome**, a 360, 390, 768, 1024 y 1280 px. Pasa si:
  - el borde discontinuo se distingue del continuo en el mapa (11 px y 7 px), en el hemiciclo y en las columnas; ningún círculo por debajo del 99% se ve sólido;
  - nada se desborda ni se pisa: Madrid y Barcelona crecen hacia abajo, el mapa no pasa de 760 px, las columnas del run caben sin barra a 360 px, el hueco del hemiciclo contiene sus tres recuentos;
  - los dos selectores se combinan y atenúan sin recargar; el conmutador cambia de dibujo; al tocar una casilla cambian la ficha, la dirección y el título; el desplegable hace lo mismo en el móvil;
  - con JavaScript desactivado: selectores, conmutador y cambio de ficha siguen funcionando; el desplegable envía su formulario y llega a la ficha;
  - las cuatro imágenes se descargan enteras, con los bordes discontinuos, sin selectores ni botones; con «PSOE» y «En juego» marcados, la del mapa lleva la línea «PSOE · En juego», los círculos atenuados y el nombre `…-mapa-psoe-en-juego.png`; con «Columnas», `…-columnas.png`. Si la del hemiciclo pierde el borde discontinuo, pasar el hemiciclo a círculos HTML posicionados (mismas posiciones en porcentaje) y repetir;
  - tras marcar un filtro, ir a otra página y volver atrás, la línea y el nombre del fichero coinciden con los radios;
  - el ámbito autonómico se ve bien en rejilla seguida, con nombres completos, y su hemiciclo sale con menos filas;
  - anotar si, al tocar una casilla a 360 y a 1280 px, la ficha queda fuera de la vista (decisión 4 del plan).

- [ ] **Step 3: Peso** — tamaño en gzip de `/es/circunscripciones` (las 52 fichas) y de `?vista=partidos`, junto a los ~84 KB de la página del evento. Si la primera pasa de 150 KB en gzip, parar y consultarlo antes de fusionar.

- [ ] **Step 4: «Estado al cierre»** en la spec: lo comprobado, lo que no se haya podido comprobar, el peso y lo que difiera del diseño (incluidas las seis decisiones de este plan).

- [ ] **Step 5: Toda la batería y commit** — `pytest -q` en verde; «Escaños que bailan v2: comprobación en el navegador, peso y cierre de la spec».

## Después del plan (Luis)

Fusionar en `dev`, push e imagen nueva de **web** (la de jobs no cambia de comportamiento). No hay que republicar ni rellenar: la página lee la parte `seat-odds` que ya llevan los runs. Conviene desplegarlo antes del 24-N. Los textos públicos son provisionales hasta su revisión.
