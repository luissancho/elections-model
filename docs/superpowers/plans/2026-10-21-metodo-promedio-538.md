# /metodo/promedio al estilo de 538 — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar en `/metodo/promedio` el texto aprobado en la spec, con notas al pie desplegables, dos cifras vivas nuevas del ciclo y los cambios de la tarjeta del ejemplo del PP.

**Architecture:** El texto vive en la plantilla Jinja2 `web/templates/metodo_promedio.html`; las cifras vivas salen de `mtpy/lib/pages/metodo.py` (partes del run nacional publicado, sin cambiar el run). Las notas son dos macros de `_macros.html` (llamada y tarjeta «Notas», numeradas desde una única lista ordenada de claves en la plantilla) y un módulo ES `web/dist/js/notes.js` que despliega una copia de la nota bajo el bloque que la contiene.

**Tech Stack:** Python 3, Jinja2 3.1, pytest, JS ES modules sin dependencias (tests con `node`), MathML.

**Spec:** `docs/superpowers/specs/2026-10-21-metodo-promedio-538-design.md` — el texto de su sección 5 y las notas de su sección 6 se copian tal cual; este plan no los repite.

**Desviación de la spec (7.1):** los tests de `node` del repo no tienen DOM, así que el test JS cubre las funciones puras de `notes.js` y el comportamiento (abrir, cerrar con la «×» y con Esc, dos notas a la vez) se comprueba en el navegador en Task 5.

## Global Constraints

- Texto público sin «casa/s» ni «patrocinador/es» (`assert_public_wording`), salvo la cabecera «Casa» de la tabla del ejemplo (`WEIGHTS_HOUSE_HEADER`, se mantiene).
- Nada del cuerpo plegado; las notas, además de desplegables, siempre visibles en la tarjeta «Notas».
- No decir en público que una corrección no mejora el backtest.
- Sin rótulos fijos repetidos («El problema», «La idea», «Los campos», «Un caso real»).
- Ids conservados: `que-entra`, `peso`, `efectos-de-casa`, `ejemplo-cis`, `regresion`, `banda`, `ejemplo`, `voto-estimado`; nuevos: `resumen` (ya existe), `muestra`, `dispersion`, `sobrerrepresentacion`, `rating`, `cuentas`, `anchura`, `consecuencias`, `notas`, y `nota-{clave}` / `llamada-{clave}` de cada nota.
- PEP8, docstrings numpydoc en toda función, identificadores en inglés neutro, textos públicos en español; JSDoc en el JS, como el resto de `web/dist/js`.
- Tipografía de la página actual: «», coma decimal, punto de miles.
- **No commitear** `tests/test_pages.py` ni `web/dist/css/site.css` (tienen cambios de Luis sin commit): se editan en su sitio, sin tocar sus cambios, y se quedan sin commit. Antes de cada commit, `git status` y `git log -1`; añadir solo ficheros propios por nombre; nunca `git commit -a`.
- No tocar la base de datos ni lanzar jobs.
- Cifras: la foto es el run `site/v1/runs/es/20261020-180003` (copia local en `/private/tmp/claude-501/-Users-luiss-HD-Proyectos-Code-elections-model/b8051675-099d-4b34-b414-7423519d26eb/scratchpad/run/`: `weights.json`, `dispersion.json`, `house-effects.csv`, `meta.json`) y `backtest/results/by_horizon.csv`; se pueden corregir contra esos ficheros (anotándolo en el commit), nunca contra la base.

## Review Focus

- Página sin run nacional ni análisis (fixture `api` vacío): las notas siguen numeradas 1…14 y enlazadas, la frase de {n_polls}/{n_pollsters} desaparece entera y las tarjetas de ejemplo no se pintan. Test en Task 3.
- Sin JavaScript: cada llamada es un `<a href="#nota-{clave}">` que lleva a una nota existente, y cada nota enlaza de vuelta a `#llamada-{clave}`. Test en Task 1 (macros) y Task 4 (página entera: todo `href="#nota-…"` tiene su `id`).
- Una llamada dentro de un `<li>`: el recuadro se inserta tras el `<li>`, dentro de la lista, no tras el primer `<p>` ni fuera de la lista. Test en Task 1 (`noteAnchorBlock`).
- Fórmulas anchas en un móvil de 360 px: no producen scroll horizontal de la página (contenedor con `overflow-x: auto`). Comprobación en Task 5 (navegador).
- Ruta del CIS con muchas elecciones (afirma el efecto) y página entera con datos: `assert_public_wording` pasa. Test en Task 4 (reutiliza `test_house_example_claims_the_effect_when_chance_cannot_explain_it`, ya existente, más la página completa).

---

### Task 1: Notas desplegables (macros, módulo JS, estilo)

**Files:**
- Modify: `web/templates/_macros.html` (al final)
- Create: `web/dist/js/notes.js`
- Modify: `web/dist/js/pages/metodo.js` (import y llamada a nivel de módulo, junto a `wireControls()`)
- Modify: `web/dist/css/site.css` (sin commit)
- Test: `tests/test_notes.py` (nuevo: macros con `pages.environment()`), `tests/test_js_notes.py` (nuevo: funciones puras con `node`)

**Interfaces:**
- Produces:
  - Macro `note(key, keys)`: `keys` es la lista ordenada de claves de la página; produce `<sup class="note-ref"><a id="llamada-{key}" href="#nota-{key}" aria-controls="nota-{key}-open" aria-expanded="false" data-note="{key}">{n}</a></sup>` con `n = keys.index(key) + 1`. Clave desconocida: error de plantilla (`keys.index` lanza), no un número vacío.
  - Macro `notes_card(keys)` con `{% call %}`: abre `m.card_start('notas', 'Notas', lead)` con lead «Detalles técnicos, comprobaciones y referencias para quien quiera ir más allá.», y dentro un `<ol class="notes">`; el cuerpo del `call` contiene los `<li>`; el `<li>` de cada nota se escribe con la macro `note_item(key)` (`<li id="nota-{key}">…cuerpo… <a class="note-back" href="#llamada-{key}" aria-label="Volver al texto">↩</a></li>`), en el mismo orden que `keys`.
  - `notes.js` exporta `noteKeyOf(href: string) -> string|null` (`'#nota-tope'` → `'tope'`; otra cosa → `null`), `noteAnchorBlock(ancestors: string[]) -> number` (índice, en la lista de etiquetas de los ancestros de la llamada desde el más cercano, del bloque tras el que se inserta el recuadro: el primer `li`, `p`, `td`, `dd` o `blockquote`; `-1` si no hay) y `wireNotes(root = document) -> void`.
  - Comportamiento de `wireNotes`: clic en una llamada → `preventDefault`; si su recuadro está abierto lo cierra; si no, inserta tras el bloque de `noteAnchorBlock` un `<div class="note-pop" id="nota-{key}-open" role="note">` con una copia del contenido del `<li id="nota-{key}">` sin el enlace de vuelta y un `<button class="note-close" aria-label="Cerrar la nota">×</button>`; actualiza `aria-expanded`. La «×» y Esc (con el foco dentro del recuadro) lo cierran y devuelven el foco a la llamada. Varios recuadros pueden estar abiertos a la vez.
- Consumes: nada.

- [ ] **Step 1: Write the failing tests**

`tests/test_notes.py`:

```python
KEYS = ['tope', 'diseno']
SOURCE = ("{% import '_macros.html' as m %}<p>Uno{{ m.note('tope', keys) }}</p><ul><li>Dos{{ m.note('diseno', keys) }}</li></ul>"
          "{% call m.notes_card(keys) %}{% call m.note_item('tope') %}Nota A{% endcall %}"
          "{% call m.note_item('diseno') %}Nota B{% endcall %}{% endcall %}")

def test_note_calls_are_numbered_by_the_order_of_the_keys_and_link_both_ways():
    page = pages.environment().from_string(SOURCE).render(keys=KEYS)
    assert re.findall(r'<a id="llamada-([a-z]+)" href="#nota-\1"[^>]*>(\d+)</a>', page) == [('tope', '1'), ('diseno', '2')]
    assert 'id="nota-tope"' in page and 'href="#llamada-tope"' in page and 'id="nota-diseno"' in page
    assert page.index('id="nota-tope"') < page.index('id="nota-diseno"') and 'id="notas"' in page

def test_an_unknown_note_key_fails_the_render():
    with pytest.raises(Exception):
        pages.environment().from_string("{% import '_macros.html' as m %}{{ m.note('x', ['a']) }}").render()
```

`tests/test_js_notes.py` (patrón de `tests/test_js_polls.py`, se salta sin `node`):

```python
def test_note_key_of_reads_only_note_hrefs():
    assert run_js('[n.noteKeyOf("#nota-tope"), n.noteKeyOf("#otra"), n.noteKeyOf("")]') == ['tope', None, None]

def test_the_pop_goes_after_the_list_item_or_the_paragraph():
    assert run_js('n.noteAnchorBlock(["sup", "li", "ul", "section"])') == 1
    assert run_js('n.noteAnchorBlock(["sup", "strong", "p", "section"])') == 2
    assert run_js('n.noteAnchorBlock(["sup", "section"])') == -1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_notes.py tests/test_js_notes.py -v`
Expected: FAIL (macros y módulo inexistentes).

- [ ] **Step 3: Implement the macros, `notes.js` and the wiring**

`metodo.js`: `import {wireNotes} from '../notes.js';` y `wireNotes();` tras `wireControls();` (fuera de `paint`, que sale pronto sin `#initial-data`). `notes.js` no importa nada y solo toca el DOM dentro de `wireNotes`, para que `node` pueda importarlo.

- [ ] **Step 4: Style in `site.css`**

`.note-ref` (volado, sin subrayado, área de pulsación ≥ 24 px), `.note-pop` (recuadro con el gris de las tarjetas, borde a la izquierda con el color de acento del sitio, `position: relative`, la «×» arriba a la derecha), `ol.notes` (texto algo menor que el cuerpo) y `.math-wrap { overflow-x: auto; }` para las fórmulas (Task 3 y 4 envuelven cada `<math display="block">` en él).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_notes.py tests/test_js_notes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit** (sin `site.css`)

```bash
git add web/templates/_macros.html web/dist/js/notes.js web/dist/js/pages/metodo.js tests/test_notes.py tests/test_js_notes.py
git commit -m "Notas al pie desplegables: macros de llamada y tarjeta, notes.js"
```

### Task 2: Cifras vivas del controlador

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (`weights_example`, nueva `cycle_counts`, `metodo_section_context` rama `promedio`)
- Test: `tests/test_pages_ejemplo.py`

**Interfaces:**
- Produces:
  - `cycle_counts() -> Optional[dict]`: `{'n_polls': int, 'n_pollsters': int}` del ciclo de `NATIONAL_SCOPE`, leído de la parte `polls` del último run (`site().latest_run`, `part(NATIONAL_SCOPE, run, 'polls')['polls']`): `n_polls` cuenta las filas con `pollster_id` no nulo (las que no son resultados), incluidas las de peso 0; `n_pollsters`, sus `pollster_id` distintos. `None` sin run, sin parte (404) o sin filas; otro error del bundle se propaga, como en `weights_example`.
  - `weights_example()` añade `'top'`: `{'pollster', 'sponsor', 'date', 'weight'}` de la fila del líder con mayor `weight` (en empate, la más reciente); `None` sin filas.
  - Contexto de `promedio`: `context['cycle'] = cycle_counts()`. Docstrings de `metodo_section_context` y `weights_example` al día.
- Consumes: nada de otras tareas.

- [ ] **Step 1: Write the failing tests** en `tests/test_pages_ejemplo.py`

`test_cycle_counts_counts_the_polls_and_pollsters_of_the_cycle`: con el fixture de run que ya usan los tests del ejemplo (`write_weights` o equivalente; si la parte `polls` de la fixture no tiene filas suficientes, escribir una con tres sondeos de dos encuestadoras, uno de peso 0, y una fila de resultado sin `pollster_id`) → `{'n_polls': 3, 'n_pollsters': 2}`. `test_cycle_counts_is_none_without_a_run` con el fixture `api` vacío → `None`. `test_weights_example_names_the_heaviest_poll`: `example['top']` es la fila de mayor `weight` de las de la fixture (comprobar `pollster` y `weight` concretos de la fixture).

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_pages_ejemplo.py -k "cycle_counts or heaviest" -v`
Expected: FAIL.

- [ ] **Step 3: Implement `cycle_counts` and `top`**

- [ ] **Step 4: Run the file**

Run: `python -m pytest tests/test_pages_ejemplo.py -v`
Expected: PASS salvo `test_promedio_page_shows_the_example_block` (ya falla en `dev`; lo reescribe Task 4).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/metodo.py tests/test_pages_ejemplo.py
git commit -m "/metodo/promedio: sondeos y encuestadoras del ciclo, y el sondeo que más pesa del ejemplo"
```

### Task 3: Texto, primera mitad (titular a efectos de encuestadora)

**Files:**
- Modify: `web/templates/metodo_promedio.html` (de `{% block description %}` a la tarjeta `efectos-de-casa` incluida; la tarjeta `ejemplo-cis` no cambia)
- Modify: `tests/test_pages.py` (`test_metodo_promedio_text`; sin commit)
- Test: `tests/test_pages_ejemplo.py` (nuevo `test_promedio_text_first_half`)

**Interfaces:**
- Consumes: macros `note`, `notes_card`, `note_item` y la clase `.math-wrap` (Task 1); `cycle` (Task 2).
- Produces: al principio del bloque `content`, `{% set note_keys = ['tope', 'diseno', 'umbrales', 'ventana', 'suelo', 'comun', 'convenciones', 'referencias', 'corte', 'isj', 'abramson', 'cobertura', 'limites', 'escala'] %}` (las notas 1…14 de la spec, en su orden); Task 4 escribe la tarjeta «Notas» con estas claves.

- [ ] **Step 1: Write the failing test**

`test_promedio_text_first_half` (con `house_site` o el fixture que tenga run y análisis): el titular de la spec; el índice enlaza, en orden, `#que-entra`, `#peso`, `#efectos-de-casa`, `#ejemplo-cis`, `#regresion`, `#banda`, `#ejemplo`, `#voto-estimado`, `#notas`; existen los h3 con ids `muestra`, `dispersion`, `sobrerrepresentacion`, `rating`, `cuentas` dentro de `#peso`, en ese orden; aparecen las frases literales «Por qué una raíz y no otra cosa», «Cuando falta la muestra», «de los 2.935 sondeos de generales», «Por qué el rebaño se mide de otra manera», «forman un grupo que pesa como un único sondeo», «Por qué un logaritmo», «pesa 3,7 veces más que el barómetro del CIS», «Por qué fiarnos del consenso», «Por qué los efectos suman cero»; con `cycle` presente, «entran {n} sondeos de {m} encuestadoras» con las cifras de la fixture; ninguna de «El problema.», «La idea.», «Los campos», «Un caso real», «backtest»; las llamadas 1 a 7 (`llamada-tope` … `llamada-convenciones`) en ese orden en la página; `assert_public_wording`. Variante sin run (fixture `api`): la frase «entran … sondeos de … encuestadoras» no aparece y la página responde 200.

En `tests/test_pages.py::test_metodo_promedio_text`, sustituir la aserción de `ln(1 + <var>R</var>/100)…` por la presencia de un `<math` con `aria-label` que contenga «logaritmo» dentro de `id="rating"`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_pages_ejemplo.py::test_promedio_text_first_half -v`
Expected: FAIL.

- [ ] **Step 3: Write the template**

Secciones 5 «Titular», 5.1, 5.2, 5.3 (con sus h3 y h4/h5 de la spec, todos con título propio) y 5.4 de la spec; cada fórmula en `<div class="math-wrap"><math display="block" aria-label="…">`; tablas con `<div class="table-wrap"><table class="et">` y pie de fecha en `<caption>`; las llamadas `{{ m.note('tope', note_keys) }}` … donde la spec dice `[nota N]`. Antes de escribir cada cifra, comprobarla contra los ficheros de la foto (Global Constraints); anotar en el commit las que se corrijan. El `{% block description %}` se reescribe a una frase que cubra el contenido nuevo.

- [ ] **Step 4: Run the page tests**

Run: `python -m pytest tests/test_pages_ejemplo.py tests/test_pages.py -k "promedio or metodo or wording" -v`
Expected: PASS salvo `test_promedio_page_shows_the_example_block` (Task 4).

- [ ] **Step 5: Commit** (sin `tests/test_pages.py`)

```bash
git add web/templates/metodo_promedio.html tests/test_pages_ejemplo.py
git commit -m "/metodo/promedio: qué sondeos entran, cuánto pesa cada sondeo y los efectos de encuestadora, al estilo de 538"
```

### Task 4: Texto, segunda mitad (regresión a firma) y tarjeta del ejemplo

**Files:**
- Modify: `web/templates/metodo_promedio.html` (de `regresion` al final)
- Test: `tests/test_pages_ejemplo.py` (reescribir `test_promedio_page_shows_the_example_block`; nuevo `test_promedio_text_second_half`)

**Interfaces:**
- Consumes: `note_keys` y las macros (Tasks 1 y 3); `example.top`, `example.published`, `example.date` (Task 2 y existentes); `author_url`, `author_handle` (contexto común).

- [ ] **Step 1: Verify the references**

Comprobar en Crossref o en la web del editor autores, año, título, revista y volumen de Cleveland (1979), Fan y Gijbels (1996), Sheather y Jones (1991), Botev, Grotowski y Kroese (2010) y Abramson (1982), y corregir las notas 8, 10 y 11 si algo no cuadra.

- [ ] **Step 2: Write the failing tests**

`test_promedio_text_second_half`: h3 `anchura` y `consecuencias` dentro de `#regresion`; al menos cuatro `<math` dentro de `#regresion` (kernel, mínimos cuadrados, forma de la solución, anchura adaptativa) y dos dentro de `#banda`; frases «regresión polinómica local ponderada por kernel», «Por qué una recta y no una media», «1,70 puntos por partido», «Por qué agrupamos por encuestadora», «Por qué una t de Student»; la tabla de la campana con «93 %» y «54 %»; la tarjeta `#notas` después de `#voto-estimado` con 14 `<li id="nota-…">` en el orden de `note_keys` y numeradas 1…14 en las llamadas; todo `href="#nota-X"` de la página tiene su `id="nota-X"`; la firma «Por Luis Sancho» con `href="https://x.com/luis_sancho"` y `rel="me"` entre `#notas` y `#closing`; `assert_public_wording`.

`test_promedio_page_shows_the_example_block` reescrito sobre la tarjeta nueva: las dos fechas («En el último cálculo publicado, el del 8 oct 2026, el promedio del PP leído el …» con las fechas de la fixture; la frase condicional solo si `date > published`); cabeceras `['Fecha', 'Casa', 'Entrevistas', 'Publicada', 'Corregida', 'Peso', 'Cercanía', 'Solape', 'Muestra', 'Rating']`; la columna Corregida muestra `pct_adj` de la fixture; la frase del sondeo que más pesa con su encuestadora; el resto de aserciones actuales que sigan siendo ciertas (orden gráfico → lista plegada → tabla, barra de peso, CSV).

- [ ] **Step 3: Run them to verify they fail**

Run: `python -m pytest tests/test_pages_ejemplo.py -k "second_half or example_block" -v`
Expected: FAIL.

- [ ] **Step 4: Write the template**

Secciones 5.6 a 5.9, la tarjeta «Notas» (`{% call m.notes_card(note_keys) %}` con los 14 `note_item` y los textos de la sección 6 de la spec) y la firma (sección 5 «Firma»), con el mismo marcado que Task 3. Tabla del ejemplo con las columnas nuevas y sus `title`. Comprobar las cifras contra la foto antes de escribirlas.

- [ ] **Step 5: Run the page tests**

Run: `python -m pytest tests/test_pages_ejemplo.py tests/test_pages.py tests/test_notes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/templates/metodo_promedio.html tests/test_pages_ejemplo.py
git commit -m "/metodo/promedio: regresión local, banda, ejemplo del PP, voto estimado, notas y firma"
```

### Task 5: Comprobación completa y en el navegador

**Files:** ninguno nuevo (correcciones donde haga falta).

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: todo pasa (los `skipped` de siempre).

- [ ] **Step 2: Browser check**

Servir la app con la fixture de datos de los tests (envoltorio ASGI con `tests.test_api_asgi`, técnica de la memoria del proyecto) o con el gunicorn local de Luis en 8000 si está levantado, y con Chrome sin cabeza por CDP comprobar en `/metodo/promedio`: clic en la llamada 1 abre el recuadro bajo su párrafo y `aria-expanded="true"`; clic en la llamada 2 abre otro sin cerrar el primero; la «×» cierra y el foco vuelve a la llamada; Esc dentro del recuadro lo cierra; una llamada dentro de un `<li>` abre el recuadro tras el `<li>`; con JS desactivado la llamada salta a `#nota-…`; a 360 px de ancho, `document.documentElement.scrollWidth <= 360` (las fórmulas no desbordan la página). Capturas a 1280 y 360 px.

- [ ] **Step 3: Commit any fixes** (solo ficheros propios; `site.css` y `tests/test_pages.py` sin commit)

- [ ] **Step 4: Report to Luis** qué ficheros quedan sin commit para él (`tests/test_pages.py`, `web/dist/css/site.css`), las cifras corregidas contra la foto y que tiene que lanzar `compute` con `overwrite` y publicar (pendiente de `2b3d13a`).
