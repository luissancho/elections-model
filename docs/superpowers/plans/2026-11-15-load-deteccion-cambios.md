# Job `load`: detección de cambios en los sondeos ya cargados — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que `load` compare lo leído de Wikipedia con la base y liste los sondeos cambiados, sustituidos y retirados junto a los nuevos, sin escribir nada distinto de lo que hoy escribe.

**Architecture:** una función pura `poll_changes` en `mtpy/lib/load.py` compara cuatro frames (sondeos y resultados leídos, sondeos y resultados de la base). `WikipediaLoader.select_series` deja esos cuatro frames en el cargador; `load_polls` llama a la función y añade tres listas al resultado; el pipeline `load.Polls` las pinta bajo la línea del ámbito.

**Tech Stack:** Python 3.11, pandas, pytest. Sin dependencias nuevas ni cambios de esquema.

**Spec:** `docs/superpowers/specs/2026-11-15-load-deteccion-cambios-design.md`

## Global Constraints

- `load` escribe exactamente lo mismo que hoy, con y sin `overwrite`: ninguna tarea toca `save_polls`, `save_results` ni el filtrado de `self.polls`/`self.results` que hace `select_series`.
- Clave de un sondeo: `Polls.key` = `['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id']`; `sponsor_id` puede ser nulo.
- Campos comparados del sondeo: `start_date`, `sample_size`, `ctype`. De cada partido: `pct` (redondeado a 2 decimales), `seats_min`, `seats_max`. Nada más.
- Dos nulos son iguales (`None`, `NaN`, `NaT`, `pd.NA`).
- Si la comparación falla, `warnings.warn('poll changes failed: {tipo}: {mensaje}')` y la carga sigue con las tres listas vacías.
- Identificadores en inglés; el detalle de la línea `~` va en español (`muestra`, `inicio`, `tipo`, `escaños`), con coma decimal y `—` para un nulo.
- PEP8, docstrings en todas las funciones (estilo numpydoc del módulo), líneas ≤ 120.
- Las pruebas nunca tocan la RDS ni escriben en `backtest/results`.
- Luis hace commits en paralelo en el mismo árbol: antes de cada commit, `git status` y `git log -1`; añadir sólo los ficheros de la tarea, nunca `git commit -a` ni reescribir sus commits.

## Review Focus

1. **Tipos distintos en los dos lados** (la base devuelve categóricas, `Int64`, fechas como `datetime64`; lo leído puede traer `float`, `object`): un 2000 frente a 2000.0 o un `NaN` frente a `pd.NA` no es un cambio. → Tarea 1, `test_poll_changes_ignores_dtype_differences`.
2. **`sponsor_id` nulo en la clave**: un sondeo sin medio debe casar con el suyo de la base, no salir como nuevo y retirado a la vez. → Tarea 1, `test_poll_changes_matches_polls_without_sponsor`.
3. **Base vacía** (primera carga de un ciclo) o **nada leído con resultados**: todo es nuevo, ninguna lista falla. → Tarea 1, `test_poll_changes_with_empty_database`.
4. **Clave repetida en el artículo** (dos filas con la misma casa, medio y fin de campo): no revienta; se compara la primera. → Tarea 1, `test_poll_changes_with_duplicate_keys`.
5. **Falsos avisos con datos reales** justo después de un `overwrite`: la lista de cambiados debe salir vacía. → Tarea 3, paso de verificación con `dry_run` contra la base.

---

### Task 1: `poll_changes`, la comparación pura

**Files:**
- Modify: `mtpy/lib/load.py` (junto a `new_poll_list`)
- Test: `tests/test_load_unit.py`

**Interfaces:**
- Consumes: `new_poll_list(polls)` (ya existe: dicts `{'pollster', 'sponsor', 'start_date', 'end_date'}` ordenados por fin de campo, casa y medio) y `Polls.key`.
- Produces:

```python
POLL_FIELDS = ('start_date', 'sample_size', 'ctype')

def poll_changes(read_polls, read_results, cur_polls, cur_results, removed=True) -> dict:
    """{'changed': [...], 'replaced': [...], 'removed': [...]}"""
```

  Los cuatro argumentos son `DataFrame` (o `None`/vacíos) con las columnas de `Polls` y `PollsResults`: los sondeos traen la clave, `pollster`, `sponsor`, `start_date`, `end_date`, `sample_size`, `ctype`; los resultados traen la clave, `party`, `pct`, `seats_min`, `seats_max`.

  - `changed`: un dict de `new_poll_list` por sondeo con la misma clave y algún dato distinto, más `'changes'`: lista de `{'field', 'party', 'old', 'new'}` en este orden: los campos de `POLL_FIELDS` que cambian (`party` `None`; fechas como `YYYY-MM-DD`, `sample_size` como `int`), y después, por partido en el orden de lo leído y luego los que sólo están en la base: `field='party'` para uno añadido (`old=None`, `new=pct`) o quitado (`old=pct`, `new=None`), `field='pct'` (`old`, `new` como `float` redondeado a 2), `field='seats'` (`old`, `new` como `[min, max]` de `int` o `None` si no hay escaños). Un partido añadido o quitado no lleva además líneas de `pct` ni `seats`.
  - `replaced`: dict de `new_poll_list` del sondeo nuevo más `'old'`: dict de `new_poll_list` del retirado al que sustituye.
  - `removed`: dicts de `new_poll_list` de los sondeos de la base que lo leído no trae y no están emparejados.
  - Con `removed=False`, `replaced` y `removed` son `[]`.
  - Las tres listas, ordenadas como `new_poll_list`.

- [ ] **Step 1: Write the failing tests** en `tests/test_load_unit.py`, con dos constructores de frames locales al test (`poll_frame(rows)` y `result_frame(rows)`) que rellenan `event_date='2026-11-29'`, `event_scope='es'`:

```python
def test_poll_changes_without_differences_is_empty():
    # mismos dos sondeos y resultados en ambos lados
    assert poll_changes(read, read_res, cur, cur_res) == {'changed': [], 'replaced': [], 'removed': []}

def test_poll_changes_reports_poll_fields():
    # base: sample_size None, start_date 2026-11-10, ctype None; leído: 2000, 2026-11-09, 'exit'
    assert out['changed'][0]['changes'] == [
        {'field': 'start_date', 'party': None, 'old': '2026-11-10', 'new': '2026-11-09'},
        {'field': 'sample_size', 'party': None, 'old': None, 'new': 2000},
        {'field': 'ctype', 'party': None, 'old': None, 'new': 'exit'}]

def test_poll_changes_reports_parties_shares_and_seats():
    # base: PP 31.0 sin escaños, PSOE 29.9 (117-123), UP 1.3; leído: PP 31.4 (125-136), PSOE 29.9 (117-123), SALF 1.8
    assert out['changed'][0]['changes'] == [
        {'field': 'pct', 'party': 'PP', 'old': 31.0, 'new': 31.4},
        {'field': 'seats', 'party': 'PP', 'old': None, 'new': [125, 136]},
        {'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8},
        {'field': 'party', 'party': 'UP', 'old': 1.3, 'new': None}]

def test_poll_changes_ignores_dtype_differences():
    # base: sample_size Int64 2000, pct 31.4, seats_min pd.NA, pollster categórica, fechas datetime64
    # leído: sample_size 2000.0, pct 31.400000001, seats_min float('nan'), pollster str, fechas date
    assert out['changed'] == []

def test_poll_changes_matches_polls_without_sponsor():
    # sponsor_id pd.NA en la base y None en lo leído, mismo sondeo
    assert out == {'changed': [], 'replaced': [], 'removed': []}

def test_poll_changes_new_and_removed_polls():
    # leído trae uno que la base no tiene (otra casa); la base, uno que lo leído no trae (otra casa)
    assert out['changed'] == [] and out['replaced'] == []
    assert out['removed'] == [{'pollster': 'Hamalgama', 'sponsor': 'OKdiario',
                               'start_date': '2026-10-20', 'end_date': '2026-10-22'}]

def test_poll_changes_pairs_a_sponsor_change():
    # misma casa y mismo fin de campo, medio distinto
    assert out['replaced'][0]['sponsor'] == 'Prisa' and out['replaced'][0]['old']['sponsor'] == 'El País'
    assert out['removed'] == []

def test_poll_changes_pairs_a_longer_fieldwork():
    # misma casa, mismo medio y mismo inicio; fin 2026-11-06 en la base, 2026-11-13 en lo leído
    assert out['replaced'][0]['end_date'] == '2026-11-13' and out['replaced'][0]['old']['end_date'] == '2026-11-06'

def test_poll_changes_leaves_ambiguous_pairs_apart():
    # dos nuevos de la misma casa con el mismo fin de campo que un retirado: ninguno se empareja
    assert out['replaced'] == [] and len(out['removed']) == 1

def test_poll_changes_without_removed():
    # el caso de test_poll_changes_pairs_a_sponsor_change con removed=False
    assert out['replaced'] == [] and out['removed'] == []

def test_poll_changes_with_empty_database():
    for cur, cur_res in ((None, None), (read.iloc[:0], read_res.iloc[:0])):
        assert poll_changes(read, read_res, cur, cur_res) == {'changed': [], 'replaced': [], 'removed': []}
    assert poll_changes(None, None, None, None) == {'changed': [], 'replaced': [], 'removed': []}

def test_poll_changes_with_duplicate_keys():
    # lo leído repite una clave (dos filas iguales salvo sample_size): no lanza y compara la primera
    assert len(out['changed']) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_load_unit.py -k poll_changes -q`
Expected: FAIL con `ImportError: cannot import name 'poll_changes'`.

- [ ] **Step 3: Implement `POLL_FIELDS` y `poll_changes` en `mtpy/lib/load.py`**

Normalizar los dos lados a valores planos de Python antes de comparar (fechas a `YYYY-MM-DD`, enteros a `int`, `pct` a `round(float, 2)`, cualquier nulo a `None`, la clave con `sponsor_id` nulo a `None`) y comparar dicts por clave: así los tipos de pandas no intervienen. Emparejado: para cada retirado, los candidatos son los nuevos con el mismo `pollster_id` y (el mismo `date`) o (el mismo `sponsor_id` y el mismo `start_date` no nulo); se empareja sólo si el retirado tiene un candidato y ese candidato no lo es de otro retirado.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_load_unit.py -q`
Expected: PASS (los nuevos y los que ya había).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/load.py tests/test_load_unit.py
git commit -m "Load: poll_changes compara los sondeos leídos con los de la base (cambiados, sustituidos y retirados)"
```

---

### Task 2: el cargador guarda los cuatro frames y `load_polls` devuelve las listas

**Files:**
- Modify: `mtpy/lib/loader.py` (`WikipediaLoader.__init__`, junto a `self.new_polls = None`, y `select_series`)
- Modify: `mtpy/lib/load.py` (`load_polls`: resultado inicial, `select_and_save`, docstring)
- Test: `tests/test_load_unit.py` (y su `FakeLoader`), `tests/test_loader_unit.py`

**Interfaces:**
- Consumes: `poll_changes(read_polls, read_results, cur_polls, cur_results, removed=True)` de la Tarea 1.
- Produces:
  - `WikipediaLoader.read_polls`, `.read_results`: los frames de `build_series` antes del filtrado de `select_series`; `.cur_polls`, `.cur_results`: las filas del ciclo en la base (`m_polls`/`m_results.get_results(query={'filters': self.filters}, formatted=True)`, sin índice). Los cuatro, `None` hasta `select_series`.
  - El resultado de `load_polls` gana `'changed'`, `'replaced'` y `'removed'` (listas de la Tarea 1, `[]` por defecto y en los resultados fallidos); `'new'` no cambia (los sustitutos siguen en ella).

- [ ] **Step 1: Write the failing tests**

En `tests/test_load_unit.py`, `FakeLoader` acepta `changes=None` (un dict con los cuatro frames) y su `select_series` pone `read_polls`, `read_results`, `cur_polls`, `cur_results` (frames vacíos por defecto). Pruebas:

```python
def test_load_polls_returns_the_changes(...):
    # FakeLoader con un sondeo cambiado (muestra None → 2000) en sus cuatro frames
    assert result['changed'][0]['changes'] == [{'field': 'sample_size', 'party': None, 'old': None, 'new': 2000}]
    assert result['replaced'] == [] and result['removed'] == []

def test_load_polls_with_years_skips_removed(...):
    # frames con un retirado; years=['2026'] → result['removed'] == []; sin years → un retirado

def test_load_polls_survives_a_failing_comparison(..., monkeypatch):
    monkeypatch.setattr(load_module, 'poll_changes', boom)   # lanza ValueError('x')
    with pytest.warns(UserWarning, match='poll changes failed: ValueError: x'):
        result = load_polls(...)
    assert result['status'] == 'loaded' and result['changed'] == []
    assert ('save_polls', False) in loader.calls

def test_load_polls_failed_results_carry_empty_change_lists(...):
    # nombres sin mapear → status failed y las tres listas []
```

En `tests/test_loader_unit.py`, siguiendo el patrón de las pruebas existentes de `select_series` (modelos de pega):

```python
def test_select_series_keeps_the_read_and_current_frames(...):
    # tras select_series(False): read_polls tiene todos los leídos aunque polls quede sólo con los nuevos;
    # cur_polls y cur_results son los de la base
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_load_unit.py tests/test_loader_unit.py -q`
Expected: FAIL en las pruebas nuevas (`KeyError: 'changed'`, `AttributeError: read_polls`).

- [ ] **Step 3: Implement**

`select_series`: guardar `read_polls`/`read_results` antes de filtrar y `cur_polls`/`cur_results` al leer la base; el resto del método, igual. `load_polls`: en `select_and_save`, tras `result['new'] = ...`, llamar a `poll_changes(wiki.read_polls, wiki.read_results, wiki.cur_polls, wiki.cur_results, removed=not years)` dentro de un `try/except Exception` que avisa con `'poll changes failed: {}: {}'.format(type(e).__name__, e)`; actualizar el docstring de `Returns`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_load_unit.py tests/test_loader_unit.py tests/test_loader_marks.py tests/test_jobs_load.py tests/test_jobs_update.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/loader.py mtpy/lib/load.py tests/test_load_unit.py tests/test_loader_unit.py
git commit -m "Load: el cargador conserva lo leído y lo de la base, y load_polls devuelve los sondeos cambiados, sustituidos y retirados"
```

---

### Task 3: las líneas del pipeline, el resumen y la alerta

**Files:**
- Modify: `mtpy/pipelines/load/Polls.py` (`line_of`, `poll_line` y formateadores nuevos)
- Modify: `mtpy/jobs/Load.py` (docstring de la clase: las líneas nuevas)
- Test: `tests/test_pipelines_load.py`, `tests/test_jobs_update.py`

**Interfaces:**
- Consumes: `result['new']`, `result['changed']`, `result['replaced']`, `result['removed']` de la Tarea 2 (usar `result.get(...) or []`: los resultados de cargadores de pega antiguos no las traen).
- Produces:

```python
def poll_label(poll) -> str            # 'Casa / Medio, 2026-11-10 → 2026-11-14' (lo que hoy arma poll_line)
def poll_line(scope, poll) -> str      # sin cambios: 'es: + ' + poll_label(poll)
def replaced_line(scope, poll) -> str  # 'es: + {label} (sustituye a {fechas del viejo})'
def changed_line(scope, poll) -> str   # 'es: ~ {label}: {detalle}; {detalle}'
def removed_line(scope, poll) -> str   # 'es: - {label}'
def change_text(change) -> str         # un detalle de changed_line
```

  Textos exactos de `change_text`: `start_date` → `inicio A → B`; `sample_size` → `muestra A → B`; `ctype` → `tipo A → B`; `party` añadido → `+ PARTIDO 1,8`; `party` quitado → `- PARTIDO`; `pct` → `PARTIDO 31,0 → 31,4`; `seats` → `PARTIDO escaños A → B` con `125–136` (raya corta) o `125` si mínimo y máximo coinciden. Nulo: `—`. Los % con coma decimal y los decimales justos (`31`, `31,4`, `1,85`).

  En `(sustituye a …)` van sólo las fechas del viejo si la casa y el medio son los mismos; si cambió el medio, `poll_label` entero del viejo.

  Orden de las líneas bajo la del ámbito: nuevos (un sustituto sale una sola vez, con su `(sustituye a …)`), cambiados, retirados. La línea del ámbito añade, tras los recuentos actuales y antes de `notes_suffix`, ` · N changed` y ` · N removed` (cada uno sólo si N > 0, unidos por `, ` en un único ` · ` como hace hoy `nouns`).

- [ ] **Step 1: Write the failing tests** en `tests/test_pipelines_load.py`:

```python
def test_change_text_formats():
    assert change_text({'field': 'sample_size', 'party': None, 'old': None, 'new': 2000}) == 'muestra — → 2000'
    assert change_text({'field': 'start_date', 'party': None, 'old': '2026-11-10', 'new': '2026-11-09'}) \
        == 'inicio 2026-11-10 → 2026-11-09'
    assert change_text({'field': 'ctype', 'party': None, 'old': None, 'new': 'exit'}) == 'tipo — → exit'
    assert change_text({'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8}) == '+ SALF 1,8'
    assert change_text({'field': 'party', 'party': 'UP', 'old': 1.3, 'new': None}) == '- UP'
    assert change_text({'field': 'pct', 'party': 'PP', 'old': 31.0, 'new': 31.4}) == 'PP 31 → 31,4'
    assert change_text({'field': 'seats', 'party': 'PP', 'old': None, 'new': [125, 136]}) == 'PP escaños — → 125–136'
    assert change_text({'field': 'seats', 'party': 'CC', 'old': [1, 1], 'new': [0, 1]}) == 'CC escaños 1 → 0–1'

def test_changed_replaced_and_removed_lines():
    poll = {'pollster': '40dB', 'sponsor': 'Prisa', 'start_date': '2026-11-10', 'end_date': '2026-11-14'}
    assert changed_line('es', poll | {'changes': [
        {'field': 'sample_size', 'party': None, 'old': None, 'new': 2000},
        {'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8}]}) \
        == 'es: ~ 40dB / Prisa, 2026-11-10 → 2026-11-14: muestra — → 2000; + SALF 1,8'
    assert removed_line('es', poll) == 'es: - 40dB / Prisa, 2026-11-10 → 2026-11-14'
    assert replaced_line('es', poll | {'old': poll | {'end_date': '2026-11-12'}}) \
        == 'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 2026-11-10 → 2026-11-12)'
    assert replaced_line('es', poll | {'old': poll | {'sponsor': 'El País'}}) \
        == 'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 40dB / El País, 2026-11-10 → 2026-11-14)'

def test_polls_lists_changes_below_the_line_and_counts_them(fresh_app, events, capsys):
    # FakeLoader con 1 nuevo (que además es sustituto), 1 cambiado y 1 retirado
    # la línea del ámbito termina en ' · 1 changed, 1 removed'; debajo, en orden: '+ … (sustituye a …)', '~ …', '- …'
    # el sustituto no aparece dos veces

def test_polls_without_changes_keeps_the_line_as_it_was(fresh_app, events, capsys):
    # sin cambios: la línea y las '+' son las de test_polls_lists_the_new_polls_below_the_line_from_verbose_1
```

En `tests/test_jobs_update.py`, junto a `test_update_lists_the_new_polls_in_the_alert`:

```python
def test_update_lists_the_changed_polls_in_the_alert(...):
    # la alerta contiene la línea '~' del sondeo cambiado
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipelines_load.py tests/test_jobs_update.py -q`
Expected: FAIL con `ImportError: cannot import name 'change_text'`.

- [ ] **Step 3: Implement** los formateadores y el cambio de `line_of` en `mtpy/pipelines/load/Polls.py`, y añadir al docstring de `Load` (párrafo de la salida de consola) las tres líneas nuevas con un ejemplo de cada una.

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: PASS, sin avisos nuevos.

- [ ] **Step 5: Verify against the real data (sólo lectura)**

Run: `python job.py load '{"scopes":["es"],"dry_run":true,"verbose":1}'`
Expected: `es: polls checked (… dry run)` y, como el ciclo se reescribió hoy con `overwrite`, ninguna línea `~` salvo las de sondeos que Wikipedia haya tocado desde entonces. Cada `~` o `-` que salga se coteja a mano con el artículo; uno que no corresponda a un cambio real es un fallo de la comparación (tipos, redondeo, correcciones del CSV) y se arregla con su prueba en la Tarea 1 antes de seguir. Apuntar en el commit lo que salió.

- [ ] **Step 6: Commit**

```bash
git add mtpy/pipelines/load/Polls.py mtpy/jobs/Load.py tests/test_pipelines_load.py tests/test_jobs_update.py
git commit -m "Load: las líneas ~ (cambiado), - (retirado) y «sustituye a» bajo el resultado del ámbito, en el resumen y en la alerta"
```
