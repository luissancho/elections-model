# Calidad previa: rúbrica, evidencia del rating y publicación — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sustituir la calidad previa manual de las encuestadoras por una rúbrica pública versionada en `data/`, replicada en la base y publicada en el perfil, el método y la API, y quitar del rating el descuento por semanas para que el prior deje de pesar cuando hay sondeos de sobra.

**Architecture:** Un módulo puro `mtpy/lib/quality.py` lee y valida el catálogo JSON de la rúbrica y el CSV de evaluaciones y calcula puntos y calidad; `save_catalogues` replica el CSV en `elections.pollsters_quality` y escribe `pollsters.quality` de todas las casas; `webapi` sirve la réplica cacheada al perfil y a la API; las plantillas del perfil y del método leen el catálogo para que texto, validación y datos salgan de un solo sitio. En `Computer` desaparece `week_decay`.

**Tech Stack:** Python 3, pandas, PostgreSQL (vía `mtpy.core.data.Model`), Jinja2, pytest. Sin JavaScript nuevo.

**Spec:** `docs/superpowers/specs/2026-10-16-calidad-previa-design.md`

## Global Constraints

- Rúbrica: cinco criterios con claves `fieldwork` (0-3), `transparency` (0-3), `independence` (0-2), `identity` (0-1), `standards` (0-1), en ese orden; textos de los niveles los de la tabla de D1 de la spec, literales.
- `quality = max(5, 8 · puntos)`; casa sin fila en el CSV = «sin evaluar» = 5. Los valores 8 y 5 viven en el JSON (`scale.per_point`, `scale.floor`) y nunca se escriben en código ni en plantillas.
- CSV `data/pollsters-quality.csv`, UTF-8, columnas exactas y en este orden: `pollster_id, pollster, fieldwork, fieldwork_source, transparency, transparency_source, independence, independence_source, identity, identity_source, standards, standards_source, notes, status, checked_at`. `status` ∈ {`borrador`, `revisada`}; `checked_at` ISO `YYYY-MM-DD`; un criterio con puntos > 0 exige fuente.
- La base es réplica del CSV: `save_catalogues` borra las filas de casas retiradas y pone el mínimo a las no evaluadas. `pollsters.quality` no se edita a mano.
- Rating: sin `week_decay` ni `weight_week` en ningún sitio; `pos_decay = 0,5`, `year_decay = 0,9`, `min_polls = 3`, `bias_dev_tau = 0,01` y el umbral de descarte de peso `< 1e-2` no cambian. `Forecaster` no se toca.
- Las páginas (`mtpy/lib/pages`) y `webapi` no importan el modelo ni `publish`; `quality.py` solo importa pandas y `App`.
- Textos públicos en español, con el tono de las fases 7-9; nada de cifras de casas concretas en el método (órdenes de magnitud: «en torno al 10 %», «entre 3 y 4»).
- Commits en inglés, imperativo, una línea, con la firma `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Todos los tests unitarios (`pytest tests -m "not integration"` o el `pytest.ini` del proyecto) verdes en cada commit.

## Review Focus

Entradas que la spec implica y que ningún texto suyo menciona; cada línea tiene su test en la tarea que la posee:

1. Una fila del CSV con un `pollster_id` que no existe en `pollsters` → `save_catalogues` debe fallar nombrando los ids, no crear calidad para una casa fantasma (Tarea 1, `test_pollsters_quality_values_fills_the_floor_and_checks_ids`).
2. El nombre `pollster` del CSV no coincide con el de la base para ese id (síntoma de id equivocado) → aviso con los dos nombres (Tarea 1, `test_pollsters_quality_values_warns_on_name_mismatch`).
3. Catálogo JSON incompleto (falta un nivel, `max` sin niveles, criterio sin `name`) o CSV con puntos fuera del máximo, fuente vacía con puntos, estado desconocido o fecha futura → `ValueError` con el motivo, nunca una calidad silenciosa (Tarea 1, `test_read_quality_rejects_*`).
4. La tabla `pollsters_quality` aún no existe (antes del paso 1 de D6) → el perfil y la API tratan a todas las casas como «sin evaluar» sin error 503 (Tarea 2, `test_get_pollsters_quality_without_table_is_empty`; Tarea 4, `test_query_pollster_without_quality_rows_gives_none`).
5. Una casa `aggr` u `online` evaluada → su tarjeta recuerda que sus sondeos no entran en el promedio; una casa con `quality` nula en `pollsters` → KPI «–» con apoyo «sin evaluar» (Tarea 5, `test_casa_profile_quality_card_of_an_informative_house`, `test_casa_profile_of_a_house_without_ratings`).

---

### Task 1: Catálogo de la rúbrica, CSV de evaluaciones y módulo `quality`

**Files:**
- Create: `data/pollsters-quality-criteria.json`
- Create: `data/pollsters-quality.csv`
- Create: `mtpy/lib/quality.py`
- Create: `tests/test_quality_unit.py`
- Modify: `tests/test_data_files.py` (añadir al final)
- Modify: `data/README.md` (tabla de ficheros, l. 8-20, y una sección nueva antes de «Resultados autonómicos», l. 23)

**Interfaces:**
- Consumes: `App.get_().data.read(name) -> str` y `App.get_().data.read_csv(name) -> pd.DataFrame` (lectores del directorio `data/`, ya usados por `get_scopes`).
- Produces (en `mtpy/lib/quality.py`):
  - `CRITERIA_FILE = 'pollsters-quality-criteria.json'`, `QUALITY_FILE = 'pollsters-quality.csv'`, `STATUSES = ('borrador', 'revisada')`, `COLUMNS` (la tupla de las 15 columnas del CSV, en orden).
  - `load_criteria() -> dict`: el JSON validado, `{'scale': {'per_point': int, 'floor': int}, 'criteria': {key: {'name': str, 'max': int, 'levels': {str: str}}}}`.
  - `criteria_keys(criteria: dict) -> list[str]`: las claves en el orden del JSON.
  - `max_points(criteria: dict) -> int`: suma de los `max` (10).
  - `quality_from_points(points: int | float, scale: dict) -> float`: `max(floor, per_point * points)`.
  - `level_text(criteria: dict, key: str, points: int) -> str`: el texto del nivel.
  - `read_quality(criteria: Optional[dict] = None, today: Optional[date] = None) -> pd.DataFrame`: el CSV validado con `COLUMNS` más `points` (int) y `quality` (float); `checked_at` como `datetime64`; índice 0..n-1. Lanza `ValueError` con el motivo.
  - `pollsters_quality_values(pollsters: pd.DataFrame, quality: pd.DataFrame, scale: dict) -> pd.Series`: la calidad de cada fila de `pollsters` (`id`, `name`), alineada a su índice: la del CSV para las evaluadas y `scale['floor']` para las demás; `ValueError` si el CSV trae ids que no están en `pollsters`; `warnings.warn` (UserWarning) por cada nombre que no coincida.

- [ ] **Step 1: Escribir el catálogo JSON**

`data/pollsters-quality-criteria.json`, con los textos literales de la tabla D1 de la spec:

```json
{
  "scale": {"per_point": 8, "floor": 5},
  "criteria": {
    "fieldwork": {"name": "Trabajo de campo", "max": 3, "levels": {
      "0": "Sin trabajo de campo propio (agregadores, estimaciones sobre otros sondeos) o modo no declarado.",
      "1": "Muestra autoseleccionada (panel voluntario, captación en web o redes) con ponderación.",
      "2": "Panel con reclutamiento controlado y cuotas, propio o contratado.",
      "3": "Entrevistas propias con marco probabilístico: teléfono, presencial o mixto con teléfono."}},
    "transparency": {"name": "Transparencia", "max": 3, "levels": {
      "0": "No publica ficha técnica.",
      "1": "Ficha técnica mínima: muestra, fechas, modo y error.",
      "2": "Ficha técnica completa, con intención directa de voto y método de estimación declarado.",
      "3": "Publica microdatos o las tablas completas de cada sondeo."}},
    "independence": {"name": "Independencia", "max": 2, "levels": {
      "0": "Depende de un gobierno, un partido o un actor político: organismo público, fundación o encuesta de partido.",
      "1": "Un único cliente, un medio.",
      "2": "Varios clientes de distinta línea, o autofinanciada, sin vínculo con un actor político."}},
    "identity": {"name": "Identidad", "max": 1, "levels": {
      "0": "Marca sin entidad identificable.",
      "1": "Entidad identificable: razón social o institución, sede, responsables, web."}},
    "standards": {"name": "Estándares", "max": 1, "levels": {
      "0": "Ni asociada ni certificada.",
      "1": "Miembro de Insights+Analytics España o de ESOMAR, o certificada ISO 20252."}}
  }
}
```

- [ ] **Step 2: Escribir el CSV con solo la cabecera**

`data/pollsters-quality.csv`: una línea con las 15 columnas de las Global Constraints, sin filas (la Tarea 7 las añade).

- [ ] **Step 3: Escribir los tests del módulo**

`tests/test_quality_unit.py` (usa `fresh_app`; el `App` lee `data/` sin base, como `webapi.catalogue()` en `tests/test_pages.py`). Los dos ficheros se leen como texto con `App.get_().data.read(name)`; para los casos inválidos se parchea ese `read` con contenidos sintéticos, dejando el otro fichero en su versión real:

```python
import os
import pandas as pd
import pytest

from mtpy.lib.quality import (COLUMNS, CRITERIA_FILE, QUALITY_FILE, criteria_keys, level_text, load_criteria,
                              max_points, pollsters_quality_values, quality_from_points, read_quality)

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
GOOD_ROW = {'pollster_id': 1, 'pollster': 'CIS', 'fieldwork': 3, 'fieldwork_source': 'https://www.cis.es/ficha',
            'transparency': 3, 'transparency_source': 'https://www.cis.es/microdatos', 'independence': 0,
            'independence_source': '', 'identity': 1, 'identity_source': 'https://www.cis.es', 'standards': 0,
            'standards_source': '', 'notes': 'Organismo público', 'status': 'revisada', 'checked_at': '2026-10-16'}


def csv_of(rows):
    return pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(index=False)


def patch_data(monkeypatch, fresh_app, files):
    """`files`: {nombre: texto}; los nombres que no estén se leen de `data/`."""
    def read(name):
        return files[name] if name in files else open(os.path.join(DATA, name), encoding='utf-8').read()
    monkeypatch.setattr(fresh_app.data, 'read', read)


def patch_csv(monkeypatch, fresh_app, text):
    patch_data(monkeypatch, fresh_app, {QUALITY_FILE: text})


def test_catalogue_has_the_five_criteria_in_order_and_ten_points(fresh_app):
    criteria = load_criteria()
    assert criteria_keys(criteria) == ['fieldwork', 'transparency', 'independence', 'identity', 'standards']
    assert max_points(criteria) == 10 and criteria['scale'] == {'per_point': 8, 'floor': 5}
    assert level_text(criteria, 'independence', 1) == 'Un único cliente, un medio.'
    for key, entry in criteria['criteria'].items():
        assert sorted(entry['levels']) == [str(i) for i in range(entry['max'] + 1)], key


def test_quality_from_points_applies_the_floor_and_the_scale(fresh_app):
    scale = load_criteria()['scale']
    assert [quality_from_points(p, scale) for p in (0, 1, 7, 10)] == [5, 8, 56, 80]


def test_read_quality_adds_points_and_quality(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW]))
    frame = read_quality(today=pd.Timestamp('2026-10-16').date())
    assert list(frame.columns) == list(COLUMNS) + ['points', 'quality']
    assert frame.loc[0, 'points'] == 7 and frame.loc[0, 'quality'] == 56.0
    assert frame.loc[0, 'checked_at'] == pd.Timestamp('2026-10-16')


@pytest.mark.parametrize('change, reason', [
    ({'fieldwork': 4}, 'fieldwork'),                 # por encima del máximo
    ({'transparency': -1}, 'transparency'),          # negativo
    ({'fieldwork_source': ''}, 'fieldwork_source'),  # puntos sin fuente
    ({'status': 'ok'}, 'status'),                    # estado desconocido
    ({'checked_at': '2099-01-01'}, 'checked_at'),    # fecha futura
    ({'checked_at': 'ayer'}, 'checked_at'),          # fecha no ISO
])
def test_read_quality_rejects_invalid_rows(fresh_app, monkeypatch, change, reason):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | change]))
    with pytest.raises(ValueError, match=reason):
        read_quality(today=pd.Timestamp('2026-10-16').date())


def test_read_quality_rejects_duplicated_ids_and_wrong_columns(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW, GOOD_ROW]))
    with pytest.raises(ValueError, match='pollster_id'):
        read_quality()
    patch_csv(monkeypatch, fresh_app, 'pollster_id,pollster\n1,CIS\n')
    with pytest.raises(ValueError, match='columns'):
        read_quality()


def test_load_criteria_rejects_a_missing_level(fresh_app, monkeypatch):
    broken = '{"scale": {"per_point": 8, "floor": 5}, "criteria": {"fieldwork": {"name": "Campo", "max": 2, "levels": {"0": "a", "2": "c"}}}}'
    patch_data(monkeypatch, fresh_app, {CRITERIA_FILE: broken})
    with pytest.raises(ValueError, match='fieldwork'):
        load_criteria()


def test_pollsters_quality_values_fills_the_floor_and_checks_ids(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW]))
    criteria = load_criteria()
    evaluated = read_quality(criteria, today=pd.Timestamp('2026-10-16').date())
    pollsters = pd.DataFrame({'id': [1, 2], 'name': ['CIS', 'GAD3'], 'quality': [10.0, 76.0]})
    values = pollsters_quality_values(pollsters, evaluated, criteria['scale'])
    assert values.tolist() == [56.0, 5.0] and values.index.equals(pollsters.index)
    with pytest.raises(ValueError, match='7'):
        pollsters_quality_values(pollsters, evaluated.assign(pollster_id=7), criteria['scale'])


def test_pollsters_quality_values_warns_on_name_mismatch(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | {'pollster': 'C.I.S.'}]))
    criteria = load_criteria()
    evaluated = read_quality(criteria, today=pd.Timestamp('2026-10-16').date())
    with pytest.warns(UserWarning, match='C.I.S.'):
        pollsters_quality_values(pd.DataFrame({'id': [1], 'name': ['CIS']}), evaluated, criteria['scale'])
```

Y en `tests/test_data_files.py`, al final, dos tests que leen los ficheros reales sin `App` (con `os.path.join(DATA, ...)`, como los demás del módulo):

```python
def test_quality_criteria_catalogue_is_complete():
    criteria = json.load(open(os.path.join(DATA, 'pollsters-quality-criteria.json')))
    assert list(criteria['criteria']) == ['fieldwork', 'transparency', 'independence', 'identity', 'standards']
    assert sum(c['max'] for c in criteria['criteria'].values()) == 10
    assert criteria['scale'] == {'per_point': 8, 'floor': 5}
    for key, entry in criteria['criteria'].items():
        assert entry['name'] and sorted(entry['levels']) == [str(i) for i in range(entry['max'] + 1)], key


def test_quality_evaluations_follow_the_rubric():
    criteria = json.load(open(os.path.join(DATA, 'pollsters-quality-criteria.json')))['criteria']
    frame = pd.read_csv(os.path.join(DATA, 'pollsters-quality.csv'), dtype={'notes': str}, keep_default_na=False)
    assert list(frame.columns)[:2] == ['pollster_id', 'pollster'] and frame['pollster_id'].is_unique
    for key, entry in criteria.items():
        points = frame[key].astype(int)
        assert points.between(0, entry['max']).all(), key
        assert (frame.loc[points > 0, key + '_source'].str.len() > 0).all(), key
    assert frame['status'].isin(['borrador', 'revisada']).all()
    assert pd.to_datetime(frame['checked_at'], format='%Y-%m-%d').le(pd.Timestamp.today()).all()
```

- [ ] **Step 4: Ejecutar los tests y ver que fallan**

Run: `pytest tests/test_quality_unit.py tests/test_data_files.py -q`
Expected: FAIL con `ModuleNotFoundError: mtpy.lib.quality` y `FileNotFoundError` de los ficheros nuevos (los de `test_data_files` pasan en cuanto existan los dos ficheros).

- [ ] **Step 5: Implementar `mtpy/lib/quality.py`**

Docstring de módulo en inglés como el resto de `mtpy/lib`. `load_criteria` lee con `json.loads(App.get_().data.read(CRITERIA_FILE))` y valida: `scale` con `per_point` y `floor` enteros positivos; cada criterio con `name`, `max` ≥ 1 y `levels` con exactamente las claves `"0"..str(max)`. `read_quality` lee con `pd.read_csv(io.StringIO(App.get_().data.read(QUALITY_FILE)), dtype=str, keep_default_na=False)`, exige `tuple(frame.columns) == COLUMNS` (`ValueError('columns ...')`), convierte los cinco criterios a int, y valida en este orden con mensajes que contengan la columna afectada: ids únicos (`'pollster_id'`), rango `0..max`, fuente no vacía cuando hay puntos (`'<key>_source'`), `status` en `STATUSES`, `checked_at` con `pd.to_datetime(format='%Y-%m-%d')` y no posterior a `today` (hoy por defecto). Añade `points = suma de los criterios` y `quality = quality_from_points(points, scale)`. `pollsters_quality_values` mapea `pollsters['id']` sobre `quality.set_index('pollster_id')['quality']`, rellena con `scale['floor']`, y antes comprueba `set(quality.pollster_id) <= set(pollsters.id)` (el mensaje lista los ids sobrantes) y avisa por cada id cuyo `pollster` del CSV difiera del `name` de la base (mensaje con ambos nombres).

- [ ] **Step 6: Ejecutar los tests y ver que pasan**

Run: `pytest tests/test_quality_unit.py tests/test_data_files.py -q`
Expected: PASS

- [ ] **Step 7: Documentar los ficheros en `data/README.md`**

Dos filas nuevas en la tabla (tras `es-districts.csv`): `pollsters-quality-criteria.json` («La rúbrica de la calidad previa de las encuestadoras: cinco criterios con sus niveles y la escala de puntos a calidad») y `pollsters-quality.csv` («La evaluación de cada encuestadora según la rúbrica: puntos y fuente por criterio, notas, estado (`borrador`/`revisada`) y fecha de comprobación; `pollster_id` es el `id` de `elections.pollsters`»), ambas «Curación propia». Una sección nueva `## Calidad previa de las encuestadoras` antes de «Resultados autonómicos» con: qué mide (cómo trabaja la casa, no cuánto acierta), la tabla de criterios y puntos (sin repetir los textos de los niveles: remitir al JSON), la regla `quality = max(floor, per_point · puntos)` con sus valores actuales (5 y 8) y el tope (80), «sin evaluar» = `floor`, que la base (`elections.pollsters_quality`, `pollsters.quality`) es réplica escrita por `save_catalogues()` (Tarea 2), y la norma de fuentes (URL o referencia citable; la base como fuente se cita «base de datos del proyecto, consulta de patrocinadores a {fecha}»).

- [ ] **Step 8: Commit**

```bash
git add data/pollsters-quality-criteria.json data/pollsters-quality.csv mtpy/lib/quality.py tests/test_quality_unit.py tests/test_data_files.py data/README.md
git commit -m "Add the public rubric of the prior quality of the pollsters and its reader"
```

---

### Task 2: Modelo `PollstersQuality`, réplica en `save_catalogues` y lector

**Files:**
- Modify: `mtpy/models/elections.py` (después de `Pollsters`, l. 21-33)
- Modify: `mtpy/lib/data.py` (imports l. 1-12; `save_catalogues` l. 131-154; nuevo `get_pollsters_quality` junto a `get_pollsters`, l. 608)
- Modify: `tests/integration/test_scopes.py`
- Modify: `tests/test_data_unit.py` (añadir al final; `FakeTable` ya existe en l. 40)

**Interfaces:**
- Consumes: Tarea 1 (`read_quality`, `load_criteria`, `pollsters_quality_values`, `COLUMNS`); `Model.table_exists()`, `Model.create(replace=False)`, `Model.format_data(df, int_type='nullable', bin_type='nullable', sort=True)`, `Model.upsert(df)`, `Model.execute(sql)`, `Model.get_results(formatted=True)`; `get_pollsters()`.
- Produces:
  - `PollstersQuality(Model)`: `table = 'elections.pollsters_quality'`, `key = ['pollster_id']`, `sort = ['pollster_id']`, `meta` con `pollster_id ['int', 3]`, `pollster ['str', None, False]`, los cinco criterios `['int', 1]`, las cinco fuentes `'str'`, `notes 'str'`, `status ['cat', 10]`, `checked_at 'dtd'`, `points ['int', 2]`, `quality 'num'`.
  - `save_catalogues() -> dict[str, int]`: ahora `{'scopes': 18, 'districts': 118, 'quality': <filas del CSV>, 'pollsters': <casas actualizadas>}`.
  - `get_pollsters_quality() -> pd.DataFrame`: la réplica formateada, o un frame vacío con las columnas del `meta` si la tabla no existe.

- [ ] **Step 1: Escribir los tests unitarios del lector y de la réplica**

En `tests/test_data_unit.py`, al final. El `FakeTable` existente sirve para `get_results`; añade un doble mínimo para `save_catalogues` que anote llamadas:

```python
def test_get_pollsters_quality_without_table_is_empty(monkeypatch):
    class NoTable(FakeTable):
        meta = {'pollster_id': 1, 'pollster': 1, 'points': 1, 'quality': 1}
        def table_exists(self):
            return False
    monkeypatch.setattr(data, 'PollstersQuality', NoTable)
    frame = data.get_pollsters_quality()
    assert frame.empty and list(frame.columns) == ['pollster_id', 'pollster', 'points', 'quality']


def test_save_catalogues_replicates_the_csv_and_writes_every_pollster(monkeypatch):
    """La réplica borra las casas retiradas del CSV, hace upsert de las evaluadas y escribe `pollsters.quality`
    de todas (el mínimo para las no evaluadas)."""
    calls = []

    class Table:
        def __init__(self, name):
            self.name = name
            self.table = 'elections.pollsters_quality' if name == 'quality' else 'elections.' + name
        def table_exists(self):
            return False
        def create(self, replace=True):
            calls.append((self.name, 'create', replace))
        def format_data(self, df, **kwargs):
            return df
        def execute(self, sql):
            calls.append((self.name, 'execute', sql))
        def upsert(self, df):
            calls.append((self.name, 'upsert', df.copy()))
            return len(df)

    evaluated = pd.DataFrame({'pollster_id': [1], 'pollster': ['CIS'], 'points': [7], 'quality': [56.0]})
    monkeypatch.setattr(data, 'Scopes', lambda: Table('scopes'))
    monkeypatch.setattr(data, 'Districts', lambda: Table('districts'))
    monkeypatch.setattr(data, 'PollstersQuality', lambda: Table('quality'))
    monkeypatch.setattr(data, 'Pollsters', lambda: Table('pollsters'))
    monkeypatch.setattr(data, 'get_scopes', lambda: pd.DataFrame({'name': ['España']}, index=pd.Index(['es'], name='scode')))
    monkeypatch.setattr(data, 'get_districts', lambda: pd.DataFrame({'scope': ['es'], 'region_id': [0]}))
    monkeypatch.setattr(data, 'read_quality', lambda criteria=None: evaluated)
    monkeypatch.setattr(data, 'load_criteria', lambda: {'scale': {'per_point': 8, 'floor': 5}, 'criteria': {}})
    monkeypatch.setattr(data, 'get_pollsters', lambda: pd.DataFrame({'id': [1, 2], 'name': ['CIS', 'GAD3'], 'quality': [10.0, 76.0], 'rating': [16.0, 73.0]}))
    out = data.save_catalogues()
    assert out == {'scopes': 1, 'districts': 1, 'quality': 1, 'pollsters': 2}
    sql = [c[2] for c in calls if c[0] == 'quality' and c[1] == 'execute']
    assert len(sql) == 1 and 'DELETE FROM elections.pollsters_quality' in sql[0] and 'NOT IN (1)' in sql[0]
    written = [c[2] for c in calls if c[0] == 'pollsters' and c[1] == 'upsert'][0]
    assert written['quality'].tolist() == [56.0, 5.0] and written['rating'].tolist() == [16.0, 73.0]
```

(`save_catalogues` compone el `DELETE` con el atributo `table` de la instancia, `'DELETE FROM {} WHERE pollster_id NOT IN ({})'.format(qmodel.table, ids)`, con los ids separados por `, `; con el CSV vacío, `'DELETE FROM {}'.format(qmodel.table)` sin `WHERE`.)

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_data_unit.py -q`
Expected: FAIL (`data.PollstersQuality`, `data.get_pollsters_quality`, `data.read_quality` no existen).

- [ ] **Step 3: Añadir el modelo y el lector**

`PollstersQuality` en `mtpy/models/elections.py` con la docstring: «Replica of `data/pollsters-quality.csv` (see `mtpy.lib.data.save_catalogues`): the evaluation of each pollster with the public rubric of `data/pollsters-quality-criteria.json`, its `points` and the resulting `quality`». En `data.py`: importar `PollstersQuality` y, de `.quality`, `load_criteria`, `pollsters_quality_values`, `read_quality`; `get_pollsters_quality()` con docstring.

- [ ] **Step 4: Extender `save_catalogues`**

Tras `scopes` y `districts`: `criteria = load_criteria()`; `quality = read_quality(criteria)`; crear la tabla si falta; `format_data` + `DELETE ... NOT IN` + `upsert`; luego `pollsters = get_pollsters()`, `pollsters['quality'] = pollsters_quality_values(pollsters, quality, criteria['scale'])`, `Pollsters().upsert(Pollsters().format_data(pollsters, int_type='nullable', bin_type='nullable', sort=True))`; `out['quality'] = len(quality)`, `out['pollsters'] = len(pollsters)`. Actualiza la docstring (réplica completa; qué devuelve).

- [ ] **Step 5: Ejecutar los unitarios y ver que pasan**

Run: `pytest tests/test_data_unit.py tests/test_quality_unit.py -q`
Expected: PASS

- [ ] **Step 6: Actualizar el test de integración**

En `tests/integration/test_scopes.py`:

```python
    out = save_catalogues()
    assert out['scopes'] == 18 and out['districts'] == 118
    from mtpy.lib.quality import read_quality
    from mtpy.models.elections import Pollsters, PollstersQuality
    assert out['quality'] == read_quality().shape[0] == PollstersQuality().get_results(formatted=True).shape[0]
    assert out['pollsters'] == Pollsters().get_results(formatted=True).shape[0]
```

Run: `pytest tests/integration/test_scopes.py -q` (se salta sin base; con la RDS del `.env` escribe la tabla nueva: **no lo ejecutes contra la RDS sin el visto bueno de Luis**, es el paso 1 de D6).
Expected: SKIPPED sin base.

- [ ] **Step 7: Commit**

```bash
git add mtpy/models/elections.py mtpy/lib/data.py tests/test_data_unit.py tests/integration/test_scopes.py
git commit -m "Replicate the pollster quality evaluations into the database and derive pollsters.quality"
```

---

### Task 3: El rating sin el descuento por semanas

**Files:**
- Modify: `mtpy/lib/computer.py` (`__init__` l. 372-374 y 416-421 y 451-453; `pollster_ratings` l. 1596-1690; `poll_rating_weights` l. 1693-1741; `print_rating_weights` l. 3340-3370)
- Modify: `tests/test_computer_unit.py` (`bare_computer` l. 188-195 y tests nuevos tras l. 250)
- Modify: `notebooks/lab/PollstersOptimize.ipynb` (celdas 6-10; una celda nueva tras la 10)

**Interfaces:**
- Produces: `Computer.__init__` sin `week_decay`; `poll_rating_weights(polls, rating_weights=None) -> pd.DataFrame` con exactamente las columnas `['weight_pos', 'weight_year', 'weight_scope']`; `pollster_ratings` con `df['weight'] = prod(weight_over, weight_sample, weight_pos, weight_year, weight_scope)`.

- [ ] **Step 1: Escribir los tests**

En `tests/test_computer_unit.py`: en `bare_computer`, `c.pos_decay, c.year_decay = .5, .9` (sin `week_decay`). Añade a `pool(rows)` un parámetro `days=10` que fije `date = event_date − days` y `'days': days`. Tests nuevos:

```python
def test_rating_weights_have_no_week_factor():
    c = bare_computer()
    out = c.poll_rating_weights(pool(ES19))
    assert list(out.columns) == ['weight_pos', 'weight_year', 'weight_scope']
    assert not hasattr(c, 'week_decay')


def test_a_distant_poll_counts_as_much_as_a_close_one():
    """D3: la desviación se mide contra el consenso de la misma fecha, así que la distancia a la elección no
    descuenta evidencia: mismo `num_polls_w`, misma desviación y mismo rating."""
    c = bare_computer()
    close = c.rate_events(pool(ES19, days=10), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    far = c.rate_events(pool(ES19, days=80), [JULY23], {'es': 1.}, c.pollsters).set_index('pollster_id')
    for col in ('num_polls_w', 'bias_dev_adj', 'rating_adj', 'rating'):
        assert close[col].tolist() == pytest.approx(far[col].tolist()), col
    assert close.loc[1, 'num_polls_w'] == pytest.approx(1.0)
```

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_computer_unit.py -q -k "week_factor or distant_poll"`
Expected: FAIL (`week_decay` existe; `bare_computer` sin `week_decay` rompe `poll_rating_weights`).

- [ ] **Step 3: Quitar el descuento del `Computer`**

Elimina el parámetro, la línea de la docstring y el atributo. En `poll_rating_weights`, borra las líneas de `event_dlimits`, `weeks` y `weight_week` y devuelve `['weight_pos', 'weight_year', 'weight_scope']`; actualiza su docstring (posición, antigüedad y ámbito). En `pollster_ratings`, las dos listas de columnas (l. 1611 y 1615) sin `weight_week`; reescribe el comentario y la docstring del método para que digan lo que hace: media ponderada de las desviaciones con los pesos de posición, antigüedad y ámbito (más muestra y sobrerrepresentación); contracción de la desviación de la casa hacia 0 por su precisión; `rating_adj = Φ(−d/τ)`; mezcla `R = (n·R_aj + k·Q)/(n + k)` con `k = min_polls`, «the prior weight of a hierarchical model, `k = σ²_house / τ²`: the poll-to-poll variance of a house beyond its sampling error over the variance between houses, measured at 3-4 on the national polls»; `weight_rating = log1p(R)/log1p(mean R)` (la docstring actual dice lo contrario: log1p comprime arriba). En `print_rating_weights` (l. 3351-3364), asigna las tres columnas (`[['weight_pos', 'weight_year', 'weight_scope']]`, hoy asigna cuatro a tres nombres) y quita `weight_week` de `weight_cols` y del producto. `grep -n "week" mtpy/lib/computer.py` debe quedar sin resultados relativos al rating.

- [ ] **Step 4: Ejecutar todos los tests del Computer y ver que pasan**

Run: `pytest tests/test_computer_unit.py -q`
Expected: PASS (los tests M11 existentes siguen iguales: sus pools usan un solo sondeo por casa y evento).

- [ ] **Step 5: Actualizar el notebook de laboratorio**

Con `NotebookEdit` sobre `notebooks/lab/PollstersOptimize.ipynb`: celda 6, `df[['weight_pos', 'weight_year', 'weight_scope']] = comp.poll_rating_weights(df)`; celdas 7, 8, 9 y 10, `weights = ['weight_sample', 'weight_over', 'weight_pos', 'weight_year']`. Inserta tras la celda 10 una celda markdown «## Medición de k = σ²_casa / τ²» y una celda de código que reproduzca el procedimiento de la spec (D3, «Cómo se midió k») sobre `comp.series` (sondeos de `es` con `bias` no nulo, `computed`, `ctype` fuera de `wban`/`exit`, escala de puntos de `bias_dev_adj`/`bias_dev_err`):

```python
d = comp.filter_polls()
d = d[d['bias'].notnull()]
rows = []
for name, x in d.groupby('pollster', observed=True):
    if len(x) >= 20:
        rows.append((name, len(x), x['bias_dev_adj'].var(), (x['bias_dev_err'] ** 2).mean()))
v = pd.DataFrame(rows, columns=['pollster', 'n', 'var_dev', 'mean_err2']).set_index('pollster')
v['sigma_house'] = np.sqrt((v['var_dev'] - v['mean_err2']).clip(lower=0))
sigma = np.sqrt(((v['var_dev'] - v['mean_err2']) * v['n']).sum() / v['n'].sum())
tau = comp.lor_to_bias(comp.bias_dev_tau)
print(v.round(3).to_string())
print('sigma_house = {:.2f} pp, tau = {:.3f} pp, k = {:.1f} (min_polls = {})'.format(sigma, tau, (sigma / tau) ** 2, comp.min_polls))
```

No ejecutes el notebook (necesita la base); déjalo sin salidas en las celdas editadas.

- [ ] **Step 6: Verificar las cifras de D3 con la réplica de solo lectura**

Exporta `elections.polls` (columnas `event_date, event_scope, date, pollster_id, pollster, days, mtype, ctype, computed, weight_over, weight_sample, bias, bias_dev_adj, bias_dev_err`, filas con `bias` no nulo) y `elections.scopes` (`scode, rating_weight`) con `psql \copy` desde el `.env` del proyecto, al directorio temporal de la sesión, y reproduce con pandas la tabla «Efecto esperado» de la spec: filtra `event_date < 2026-11-29`, `computed`, `ctype ∉ {wban, exit}`, `mtype ∉ {aggr, online}`, `rating_weight > 0`; `w_pos = 0.5^rank` (rango por fecha descendente dentro de `(event_date, event_scope, pollster_id)`), `w_year = 0.9^(max(año) − año)`, `w_scope`; «hoy» añade `w_week = 0.7^((days − min(days) del evento + 1) // 7)` y descarta pesos `< 1e-2`; `n = Σ w`, `prior = 3/(3 + n)`, `dev = Σ w·bias_dev_adj / Σ w`, `err = sqrt(Σ w²·bias_dev_err²)/Σw · sqrt((n+1)/n)`, contracción `dev · (1/err²)/(1/err² + 1/τ²)` con `τ = 1.005`, `R_aj = Φ(−dev/τ)·100`. Comprueba que CIS, Sigma Dos, GAD3, Metroscopia, Simple Lógica, Ikerfel y Cluster17 caen a menos de un punto de la tabla de la spec en «hoy» y en «nuevo» (n, prior, R_aj), y que `k` del Step 5 (misma fórmula sobre el export, casas `es` con ≥ 20 sondeos) sale entre 3 y 4. Anota los valores obtenidos en el mensaje del commit del Step 7 si difieren en más de un punto, y en ese caso corrige la tabla de la spec.

- [ ] **Step 7: Commit**

```bash
git add mtpy/lib/computer.py tests/test_computer_unit.py notebooks/lab/PollstersOptimize.ipynb
git commit -m "Drop the week decay from the pollster ratings so the evidence counts every poll"
```

---

### Task 4: La réplica en `webapi` y en la API

**Files:**
- Modify: `mtpy/lib/webapi/queries.py` (lectores l. 44-57; `query_pollsters` l. 272-281; `query_pollster` l. 284-326)
- Modify: `mtpy/lib/webapi/__init__.py` (exports l. 23-30)
- Modify: `tests/test_webapi_unit.py` (l. 396-440 y tests nuevos)
- Modify: `tests/test_web_api.py` (`fake_readers` l. 150-175 y el test de `/api/v1/pollsters/3`, l. 232)
- Modify: `docs/web/contrato.md` (fila de `/api/v1/pollsters/{id}`, l. 534, y viñeta l. 551)

**Interfaces:**
- Consumes: Tarea 2 (`data.get_pollsters_quality`).
- Produces: `webapi.queries.get_pollsters_quality` (proxy `_reader('get_pollsters_quality')`); `query_pollsters_quality() -> pd.DataFrame` (clave de caché `('pollsters-quality',)`); `query_pollster(...)` devuelve `{'pollster', 'ratings', 'house_effects', 'herding', 'quality'}` con `quality` el registro JSON (`frame_to_records`) de la fila de la casa o `None`.

- [ ] **Step 1: Escribir los tests**

En `tests/test_webapi_unit.py`: en `test_query_pollster_profile_and_unknown_pollster` parchea además `get_pollsters_quality` con un frame de una fila (`pollster_id 3`, los cinco criterios, fuentes, `notes`, `status 'revisada'`, `checked_at` `2026-10-16`, `points 7`, `quality 56.0`) y cambia la aserción de claves a `['pollster', 'ratings', 'house_effects', 'herding', 'quality']`; añade `assert profile['quality']['points'] == 7 and profile['quality']['checked_at'] == '2026-10-16'`. En `test_query_pollster_reuses_the_cached_scope_tables` parchea `get_pollsters_quality` con `counted('quality', pd.DataFrame({'pollster_id': [3], 'quality': [56.0]}))`, añade `webapi.query_pollsters_quality()` a las lecturas previas, cambia la aserción a `sorted(calls) == ['herding', 'house_effects', 'pollsters', 'quality', 'ratings']` y el recuento final a `len(calls) == 5`. Test nuevo:

```python
def test_query_pollster_without_quality_rows_gives_none(fresh_app, monkeypatch):
    """Antes de replicar el CSV la tabla está vacía (o no existe): el perfil lleva `quality` a None."""
    use_database(fresh_app)
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pd.DataFrame({'id': [3], 'name': ['CIS']}))
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame())
    monkeypatch.setattr(webapi.queries, 'get_house_effects', lambda scope: pd.DataFrame({'pollster_id': []}))
    monkeypatch.setattr(webapi.queries, 'get_herding', lambda scope: pd.DataFrame({'pollster_id': []}))
    monkeypatch.setattr(webapi.queries, 'get_pollsters_quality', lambda: pd.DataFrame(columns=['pollster_id', 'quality']))
    assert webapi.query_pollster(3, 'es')['quality'] is None
```

En `tests/test_web_api.py`: `fake_readers` parchea `get_pollsters_quality` con el mismo frame de una fila para el id 3; en el test de `/api/v1/pollsters/3` añade `assert data['quality']['status'] == 'revisada' and data['quality']['quality'] == 56.0` y, para el id 2, `assert json.loads(call(db_api, '/api/v1/pollsters/2')[2])['quality'] is None`.

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_webapi_unit.py tests/test_web_api.py -q -k pollster`
Expected: FAIL (`get_pollsters_quality` no es atributo de `webapi.queries`).

- [ ] **Step 3: Implementar**

`get_pollsters_quality = _reader('get_pollsters_quality')`; `query_pollsters_quality()` con `_query(('pollsters-quality',), lambda: get_pollsters_quality())` y docstring; en `query_pollster`, `rows = _rows_of(query_pollsters_quality(), 'pollster_id', pollster_id)` y `'quality': frame_to_records(rows)[0] if not rows.empty else None`; docstring actualizada. Exporta `get_pollsters_quality` y `query_pollsters_quality` en `webapi/__init__.py`. El controlador `Data.pollster_action` no cambia (hace `**profile`); actualiza su docstring (l. 141).

- [ ] **Step 4: Ejecutar y ver que pasan**

Run: `pytest tests/test_webapi_unit.py tests/test_web_api.py -q`
Expected: PASS

- [ ] **Step 5: Documentar el contrato**

`docs/web/contrato.md`: la fila de `/api/v1/pollsters/{id}` pasa a `{contract, scope, pollster, ratings, house_effects, herding, quality}`; nueva viñeta tras la l. 551: «`pollster.quality` es el registro de `elections.pollsters_quality` de la casa: los cinco criterios de la rúbrica (`fieldwork`, `transparency`, `independence`, `identity`, `standards`) con su fuente (`*_source`), `notes`, `status` (`borrador`/`revisada`), `checked_at` (ISO), `points` y `quality`; `null` si la casa no está evaluada (su `quality` del catálogo es entonces el mínimo, 5). Ver `data/README.md`, «Calidad previa de las encuestadoras».»

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/webapi/queries.py mtpy/lib/webapi/__init__.py mtpy/controllers/Data.py tests/test_webapi_unit.py tests/test_web_api.py docs/web/contrato.md
git commit -m "Serve the quality evaluation of a pollster in its profile and in the API"
```

---

### Task 5: La tarjeta «Calidad previa» del perfil

**Files:**
- Modify: `mtpy/lib/pages/casas.py` (`profile_kpis` l. 496-519; `casa_context` l. 521-700)
- Modify: `mtpy/lib/pages/__init__.py` (export de `quality_card`, l. 18-20)
- Modify: `web/templates/casa.html` (KPIs l. 14-16; tarjeta nueva antes de `card_start('path', ...)`, l. 17)
- Modify: `web/templates/_glosario.html` (entrada nueva bajo `with_rating`)
- Modify: `tests/test_pages.py` (`fake_houses` l. 883-950; tests l. 1042-1060, 1142-1152, 2484-2488; tests nuevos)

**Interfaces:**
- Consumes: Tarea 1 (`quality.load_criteria`, `criteria_keys`, `max_points`, `level_text`); Tarea 4 (`profile['quality']`, `profile['pollster']['quality']`).
- Produces:
  - `profile_kpis(current: dict, mtype: Optional[str], quality_support: str) -> list`: igual que hoy, con `support` del KPI «Calidad previa» = `quality_support`.
  - `quality_card(record: Optional[dict], pollster: dict, criteria: dict) -> dict`: `{'evaluated': bool, 'rows': [{'key', 'name', 'points', 'max', 'level', 'source', 'parts'}], 'points': int|None, 'max_points': int, 'quality': float|None, 'status': str|None, 'draft': bool, 'checked_at': str|None, 'notes': str|None, 'informative': bool, 'support': str, 'floor': int, 'per_point': int}`; `informative` cuando `pollster['mtype'] in ('aggr', 'online')`; `support` = `'{points} de {max_points} puntos'` o `'sin evaluar'`; `parts` es la fuente partida en texto y enlaces por `source_parts(source) -> list` (`[{'text': str, 'url': str|None}]`, cada URL sin el `)`, `.`, `;`, `,` o `:` que cierra la frase; corrección de la revisión final, sustituye a `is_url`).
  - Contexto del perfil: `context['quality'] = quality_card(...)`; `context['current']['quality'] = profile['pollster'].get('quality')` (la calidad de hoy).

- [ ] **Step 1: Ampliar la fixture y escribir los tests**

En `fake_houses`: `pollsters.quality` pasa a `[56.0, 76.0, 40.0, 5.0, 5.0]` y `mtype` de `Cero` a `'online'`; añade el frame de la réplica y su parche:

```python
    quality = pd.DataFrame({
        'pollster_id': [1, 3], 'pollster': [name, 'Hamalgama'],
        'fieldwork': [3, 2], 'fieldwork_source': ['https://www.cis.es/es/metodologia', 'Ficha técnica del sondeo de 2026-10-01'],
        'transparency': [3, 1], 'transparency_source': ['https://www.cis.es/es/microdatos', 'https://hamalgama.example/ficha'],
        'independence': [0, 1], 'independence_source': ['', 'base de datos del proyecto, consulta de patrocinadores a 2026-10-16'],
        'identity': [1, 1], 'identity_source': ['https://www.cis.es', 'https://hamalgama.example'],
        'standards': [0, 0], 'standards_source': ['', ''],
        'notes': ['Organismo público adscrito al Ministerio de la Presidencia', ''],
        'status': ['revisada', 'borrador'], 'checked_at': pd.to_datetime(['2026-10-16', '2026-10-16']),
        'points': [7, 5], 'quality': [56.0, 40.0],
    })
    monkeypatch.setattr(webapi.queries, 'get_pollsters_quality', lambda: quality)
```

Tests nuevos (y ajusta los existentes que lean la calidad de la fixture: `test_profile_opens_with_three_kpis_and_keeps_the_path_table` sigue igual; `test_casa_profile_of_a_house_without_ratings` pasa a esperar un solo «–», el del rating, y `'<div class="v">5,0<small></small></div>'` con `sin evaluar`):

```python
def test_casa_profile_quality_card_lists_the_rubric_with_sources(houses):
    page = html(call(houses, '/casas/cis')[2])
    kpis = page[page.index('id="profile"'):page.index('id="quality"')]
    assert '<div class="v">56,0<small></small></div>' in kpis.replace('\n', '') and '7 de 10 puntos' in kpis
    card = page[page.index('id="quality"'):page.index('id="path"')]
    rows = re.findall(r'<tr>\s*<th scope="row">([^<]+)</th>\s*<td>(\d) / (\d)</td>', card)
    assert [r[0] for r in rows] == ['Trabajo de campo', 'Transparencia', 'Independencia', 'Identidad', 'Estándares']
    assert [(r[1], r[2]) for r in rows] == [('3', '3'), ('3', '3'), ('0', '2'), ('1', '1'), ('0', '1')]
    assert 'Entrevistas propias con marco probabilístico' in card and 'href="https://www.cis.es/es/microdatos"' in card
    assert 'Comprobado el ' in card and 'Organismo público' in card and 'provisional' not in card
    assert 'cada punto vale 8' in card and 'mínimo de 5' in card  # el «Cómo leerlo» lee la escala del catálogo


def test_casa_profile_quality_card_marks_drafts_and_plain_sources(houses):
    card = html(call(houses, '/casas/hamalgama')[2])
    card = card[card.index('id="quality"'):card.index('id="path"')]
    assert 'Evaluación provisional, pendiente de revisión' in card
    assert 'Ficha técnica del sondeo de 2026-10-01' in card and 'href="Ficha' not in card
    assert 'consulta de patrocinadores a 2026-10-16' in card


def test_casa_profile_quality_card_of_an_unrated_house(houses):
    page = html(call(houses, '/casas/nueva')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'Sin evaluar' in card and 'calidad mínima, 5' in card and 'href="/metodo/rating#calidad-previa"' in card
    assert '<th scope="row">' not in card and 'sin evaluar' in page[page.index('id="profile"'):page.index('id="quality"')]


def test_casa_profile_quality_card_of_an_informative_house(houses):
    page = html(call(houses, '/casas/cero')[2])
    card = page[page.index('id="quality"'):page.index('id="path"')]
    assert 'no entran en el promedio' in card and 'Sin evaluar' in card
```

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_pages.py -q -k "quality_card or without_ratings or three_kpis"`
Expected: FAIL (`id="quality"` no existe).

- [ ] **Step 3: Implementar el contexto**

En `casas.py`: importar `from ..quality import criteria_keys, level_text, load_criteria, max_points`; `quality_card` según Interfaces (`checked_at` como `str` ISO del registro, la plantilla la formatea con `|date`); en `casa_context`, `context['current'] = {'rating': <rating de la última fila o None>, 'quality': profile['pollster'].get('quality'), 'event_date': ...}`, `context['quality'] = quality_card(profile.get('quality'), profile['pollster'], load_criteria())`, `context['kpis'] = profile_kpis(context['current'], context['mtype'], context['quality']['support'])`. Actualiza la docstring de `casa_context` (`current`: la calidad de hoy; `quality`). Exporta `quality_card` en `pages/__init__.py`.

- [ ] **Step 4: Escribir la plantilla**

En `casa.html`, tras la tarjeta `profile`: `{{ m.card_start('quality', 'Calidad previa', 'Lo que se sabe de cómo trabaja la encuestadora antes de mirar si acierta: cinco criterios comprobables, con su fuente.') }}`. Si `quality.evaluated`: `<div class="table-wrap"><table id="quality-table"><caption>…</caption><thead><tr><th scope="col">Criterio</th><th scope="col">Puntos</th><th scope="col">Nivel</th><th scope="col">Fuente</th></tr></thead><tbody>` con una fila por `quality.rows`: `<th scope="row">{{ row.name }}</th><td>{{ row.points }} / {{ row.max }}</td><td>{{ row.level }}</td><td>{% if row.is_url %}<a href="{{ row.source }}" rel="noopener">{{ row.source }}</a>{% else %}{{ row.source or '–' }}{% endif %}</td>`; pie `<p class="muted">{{ quality.points }} de {{ quality.max_points }} puntos: calidad previa {{ quality.quality|num(1) }}. Comprobado el {{ quality.checked_at|date }}.{% if quality.notes %} {{ quality.notes }}{% endif %}</p>`; si `quality.draft`, `<p class="notice">Evaluación provisional, pendiente de revisión.</p>`. Si no evaluada: `<p class="notice">Sin evaluar: tiene la calidad mínima, {{ quality.floor }}, hasta que alguien compruebe cómo trabaja según <a href="/metodo/rating#calidad-previa">la rúbrica</a>.</p>`. Si `quality.informative`: `<p class="muted">Sus sondeos no entran en el promedio ni en el rating (agregadores y paneles en línea quedan fuera): la calidad previa es informativa.</p>`. «Cómo leerlo» (`m.how()`): la rúbrica puntúa cómo trabaja la casa, no cuánto acierta; `cada punto vale {{ quality.per_point }} de calidad, con un mínimo de {{ quality.floor }}`; que la calidad del KPI es la de hoy y la que usó cada rating pasado está en «Camino al rating». Añade al «Cómo leerlo» de «Camino al rating» la frase «La calidad previa que se usó en cada elección fue la de entonces; la del KPI es la de hoy.» En `_glosario.html`, bajo `with_rating`, antes de «Rating»: `<dt>Calidad previa</dt><dd>Lo que se supone de la encuestadora antes de mirar sus sondeos, de 0 a 100, según una rúbrica pública de cinco criterios sobre cómo trabaja; pesa en el rating mientras tiene pocos sondeos.</dd>`.

- [ ] **Step 5: Ejecutar toda la suite de páginas y ver que pasa**

Run: `pytest tests/test_pages.py -q`
Expected: PASS (ajusta las aserciones existentes que dependan del `quality` antiguo de la fixture, p. ej. `test_casa_profile_header_uses_the_scope_rating`, l. 1198, sin cambiar lo que comprueban).

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/pages/casas.py mtpy/lib/pages/__init__.py web/templates/casa.html web/templates/_glosario.html tests/test_pages.py
git commit -m "Show the quality rubric of a pollster in its profile"
```

---

### Task 6: El método: la rúbrica y los pesos del rating

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (nueva `quality_rubric()`; `metodo_section_context` l. 226-262)
- Modify: `mtpy/lib/pages/__init__.py` (export de `quality_rubric`)
- Modify: `web/templates/metodo_rating.html` (resumen l. 38; tarjeta `calidad-previa` l. 86-130; descripción l. 3)
- Modify: `web/templates/metodo.html` (sección «Los datos», l. 71-74)
- Modify: `tests/test_pages.py` (`test_metodo_rating_text` l. 1843-1856; test nuevo)

**Interfaces:**
- Consumes: Tarea 1 (`load_criteria`, `criteria_keys`, `max_points`).
- Produces: `quality_rubric() -> dict`: `{'rows': [{'key', 'name', 'max', 'levels': [{'points': int, 'text': str}]}], 'max_points': int, 'per_point': int, 'floor': int, 'cap': int}` (`cap = per_point · max_points`); en el contexto de la subpágina `rating`, `context['rubric'] = quality_rubric()`.

- [ ] **Step 1: Escribir los tests**

Reemplaza en `test_metodo_rating_text` la aserción `'el denominador es la calidad previa media…'` (se mantiene) y añade:

```python
    assert 'Trabajo de campo' in text and 'Estándares' in text and '<table id="rubric-table">' in page
    assert 'cada punto vale 8 de calidad, con un mínimo de 5' in text and 'nadie pasa de 80 antes de los datos' in text
    assert 'entre 3 y 4' in text and 'en torno al 10' in text
    assert 'Por su cercanía a la votación' not in text and 'multiplica el peso por 0,7' not in text
    assert 'puesta a mano' not in text and 'rúbrica pública' in text
    assert 'calidad previa según una rúbrica pública' in re.sub(r'\s+', ' ', html(call(api, '/metodo')[2]))


def test_metodo_rating_reads_the_rubric_from_the_catalogue(api, monkeypatch):
    changed = pages.quality_rubric() | {'per_point': 10, 'floor': 1, 'cap': 100}
    monkeypatch.setattr(pages.metodo, 'quality_rubric', lambda: changed)
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/rating')[2]))
    assert 'cada punto vale 10 de calidad, con un mínimo de 1' in text and 'nadie pasa de 100 antes de los datos' in text
```

- [ ] **Step 2: Ejecutar y ver que fallan**

Run: `pytest tests/test_pages.py -q -k metodo_rating`
Expected: FAIL.

- [ ] **Step 3: Implementar el contexto**

`quality_rubric()` en `metodo.py` a partir de `load_criteria()`; en `metodo_section_context`, `if section == 'rating': context['rubric'] = quality_rubric()`; docstring. Export en `pages/__init__.py`.

- [ ] **Step 4: Reescribir los textos**

`metodo_rating.html`:
- Resumen (l. 38): «Con pocos sondeos manda una calidad previa según una rúbrica pública; con muchos, lo medido. El rating va de 0 a 100.»
- Tarjeta `calidad-previa`: lead «Antes de ver sus sondeos, cada encuestadora tiene una calidad previa de 0 a 100: lo que se sabe de cómo trabaja, no de cuánto acierta.» Párrafo 1: qué mide y qué no (acierto, coherencia y longevidad quedan fuera porque los miden los datos). Tabla `<table id="rubric-table">` desde `rubric.rows`: columnas Criterio, Puntos (`0-{{ row.max }}`), Niveles (una lista `<ul>` con `{{ level.points }}: {{ level.text }}` de mayor a menor). Párrafo 2: «La calidad es la suma de puntos, hasta {{ rubric.max_points }}: cada punto vale {{ rubric.per_point }} de calidad, con un mínimo de {{ rubric.floor }}; nadie pasa de {{ rubric.cap }} antes de los datos, los últimos se ganan acertando. Lo que no se puede comprobar vale 0 en su criterio, así que una encuestadora de la que no se sabe nada queda cerca del mínimo; una sin evaluar tiene exactamente el mínimo.» Párrafo 3: la evaluación es pública y revisable, con una fuente por criterio, en el fichero `data/pollsters-quality.csv` del repositorio; cada perfil de <a href="/casas{{ scope_query }}">Ranking Encuestadoras</a> muestra el desglose. Conserva la fórmula de la mezcla y su explicación; sustituye «<var>k</var> vale 3: la calidad previa pesa como tres sondeos. Con <var>n</var> = 3, mitad y mitad…» por: «<var>k</var> vale 3: la calidad previa pesa como tres sondeos. Es, redondeado, lo que mide la relación entre lo que una encuestadora varía de sondeo a sondeo, más allá de su error muestral, y lo que varían las encuestadoras entre sí: entre 3 y 4. Con <var>n</var> = 3, mitad y mitad; con menos, manda la calidad previa; con muchos más, lo medido.» En la lista de pesos de <var>n</var>: borra el punto «Por su cercanía a la votación»; en el de posición añade «Los sondeos con un peso total por debajo de una centésima se descartan.»; sustituye el párrafo final «Con estos pesos, cada elección aporta pocos sondeos efectivos…» por «Con estos pesos, una encuestadora con decenas de sondeos en varias elecciones tiene una calidad previa que pesa en torno al 10 % de su rating, y una recién llegada está dominada por ella.»
- Descripción (l. 3): «…la calidad previa (una rúbrica pública)…».

`metodo.html` (l. 71-74): «…y cada encuestadora recibe una calidad previa según una rúbrica pública (ver <a href="/metodo/rating#calidad-previa">El rating</a>).»

- [ ] **Step 5: Ejecutar y ver que pasan**

Run: `pytest tests/test_pages.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/pages/metodo.py mtpy/lib/pages/__init__.py web/templates/metodo_rating.html web/templates/metodo.html tests/test_pages.py
git commit -m "Explain the quality rubric and the evidence of the rating in the method"
```

---

### Task 7: Evaluación inicial de las casas activas (D5)

**Files:**
- Modify: `data/pollsters-quality.csv` (las filas)

**Interfaces:**
- Consumes: Tarea 1 (formato y validación: `pytest tests/test_data_files.py -q` debe pasar al final).

Las 72 casas con algún sondeo desde el 1 de enero de 2019 (consulta a la base del 2026-10-16; `id|nombre|mtype|sondeos|patrocinadores desde 2019 (nº)`):

```
28|ElectoPanel|online|613|Electomanía 690
93|EM Analytics|online|446|Electomanía 305; El Plural 91; GMG 25; VilaWeb 11; Nós Diario 10; Crónica Vasca 8
3|Sigma Dos||253|El Mundo 204; Antena 3 33; sin patrocinador 7; RTVCyL 6; Las Provincias 3; EITB 2; RTVE 2; La Verdad 2; Telemadrid 2; Ical 1; El Diario Montañés 1; OKDiario 1; Hoy 1
7|NC Report||199|La Razón 195; sin patrocinador 5; Camps 3; La Tribuna 1; EPDA 1
10|SocioMétrica||161|El Español 171; sin patrocinador 2
2|GAD3||148|ABC 90; sin patrocinador 19; NIUS 15; La Vanguardia 8; RTVE 7; Mediaset 7; Vocento 5; El Norte de Castilla 4; Las Provincias 2; FORTA 1; El Debate 1; Telecinco 1; EITB 1; Diario La Rioja 1; COPE 1
1|CIS|state|134|sin patrocinador 134
8|KeyData|aggr|124|Público 124
6|Data10||102|OKDiario 102; sin patrocinador 1
4|InvyMark||101|La Sexta 94; Telemadrid 5; sin patrocinador 2
12|Celeste Tel||98|Onda Cero 55; elDiario.es 34; sin patrocinador 12
15|40dB||92|Prisa 86; El País 8
5|Simple Lógica||87|elDiario.es 51; sin patrocinador 36
14|Hamalgama||85|VozPópuli 51; OKDiario 28; UNED 4; La Provincia 2; Espiral21 1
11|DYM||73|Henneo 66; El Independiente 3; Prensa Ibérica 2; La Nueva España 1; sin patrocinador 1
17|GESOP||72|El Periódico 34; Prensa Ibérica 27; CEO 15; The Adelaide Review 10; sin patrocinador 4; Faro de Vigo 1; La Nueva España 1
27|Sondaxe||69|La Voz de Galicia 69; sin patrocinador 1
9|IMOP||63|El Confidencial 61; COPE 2; CIS 2
18|Target Point||63|El Debate 65
23|SyM Consulting||50|sin patrocinador 40; EPDA 6; Valencia Plaza 3; La Nueva Crónica 3
13|Metroscopia||46|sin patrocinador 39; Henneo 6; El Confidencial 2; Periodismo 2030 1; El País 1; Ical 1
36|Demoscopia||41|ESdiario 37; OKDiario 4; sin patrocinador 2; esRadio 1
24|CEMOP||25|sin patrocinador 25
16|SW Demoscopia||22|sin patrocinador 13; Grupo Viva 4; InfoLibre 2; Cadena SER 1; El Plural 1; Publicaciones Sur 1
99|CENTRA||21|CEA 21
19|Ágora Integral||19|Canarias Ahora 17; El Periódico 2
98|Gizaker||15|EITB 12; Grupo Noticias 4; sin patrocinador 1
54|Ikerfel||15|GPS 12; El Diario Vasco 2; Vocento 1
40|Ipsos||13|La Vanguardia 8; sin patrocinador 3; Henneo 3
47|A+M||12|Heraldo de Aragón 13
50|IBES||10|Última Hora 10; sin patrocinador 2
108|Idus3||9|sin patrocinador 9
102|Invest Group||9|CEA 4; Prensa Ibérica 4; Levante EMV 1
52|Feedback||8|El Nacional 8; Diari d'Andorra 6; The National 5
41|Opinometre||8|CEO 6; Ara 2
29|Cluster17||6|Agenda Pública 3; El Grand Continent 2; sin patrocinador 1
118|Social Data||6|Grupo Viva 6
106|CIES||5|Diario de Navarra 3; sin patrocinador 2
123|UCAM||5|sin patrocinador 5
42|YouGov|aggr|5|sin patrocinador 3; Ara 3
104|Aztiker||4|Gara 4
30|Top Position||4|sin patrocinador 4
116|TSA||4|Canarias7 4
97|Ateneo del Dato||3|elDiario.es 2; sin patrocinador 1
107|GfK||3|sin patrocinador 3; FORTA 3; CRTVG 1
103|Ikertalde||3|GPS 3
131|Lápiz Estratégico||3|Prensa Ibérica 3
95|More in Common||3|sin patrocinador 3
128|Murcia Electoral||3|sin patrocinador 3
115|NueveCuatroUno||3|sin patrocinador 3
136|OBEDE||3|UCAM 3
49|Deimos|online|2|sin patrocinador 1; VozPópuli 1
134|DemosGal||2|La Capital 2
109|Gadeso||2|sin patrocinador 2
22|GIPEyOP|online|2|sin patrocinador 2
113|inPactos||2|sin patrocinador 2
73|Intercampo||2|CIS 2
37|Llorente & Cuenca|aggr|2|sin patrocinador 2
94|Opina 360||2|Antena 3 1; Telecinco 1
119|UPNA||2|Parliament of Navarre 2
51|Asturbarómetro||1|La Nueva España 1
48|AtlasIntel||1|sin patrocinador 1
105|Commentia||1|Grupo Joly 1
101|CPS||1|EHU 1
65|Demométrica||1|sin patrocinador 1
39|GI Internacional||1|Diario16 1
25|Merca2||1|sin patrocinador 1
114|Nexo||1|CEA 1
96|ODEC||1|Mediaflows UV 1
124|Perfiles||1|La Provincia 1
26|Sináptica||1|Público 1
137|Winston||1|Artículo 14 1
```

Datos de la base útiles para dos criterios (misma consulta): la proporción de sondeos con muestra registrada es del 80-100 % en casi todas las casas; excepciones KeyData 0 %, Llorente & Cuenca 0 %, GI Internacional 0 %, InvyMark 16 %, NueveCuatroUno 33 %, Metroscopia 59 %, ElectoPanel 65 %. «GPS» es el Gabinete de Prospección Sociológica del Gobierno Vasco; «CEO» el Centre d'Estudis d'Opinió de la Generalitat; «CEA» la Junta de Andalucía (Centro de Estudios Andaluces); «EHU» la Universidad del País Vasco.

- [ ] **Step 1: Investigar y redactar las filas**

Para cada casa, en el orden de la lista (se puede repartir en tres lotes de 24 entre subagentes con búsqueda web, cada uno escribiendo sus filas en un fichero temporal del directorio de la sesión que después se concatena), comprobar en este orden y anotar la fuente de cada criterio como URL (web de la casa, ficha técnica de un sondeo reciente, directorio de asociados de Insights+Analytics España, directorio de ESOMAR, registro público) o como referencia citable: (1) `identity`: existe una entidad identificable (razón social, institución, sede, responsables, web). (2) `fieldwork`: modo declarado en la ficha de su último sondeo publicado (CATI/presencial/mixto → 3; panel con cuotas → 2; panel voluntario o captación en web/redes → 1; agregador, estimación o modo no declarado → 0). (3) `transparency`: microdatos o tablas completas → 3; ficha completa con intención directa y método de estimación → 2; ficha mínima → 1; sin ficha → 0 (para una casa cuyos sondeos de la base tienen muestra y fechas en ≥ 80 % de los casos, el nivel 1 se puede citar con la base: «base de datos del proyecto, consulta de muestras a 2026-10-16»). (4) `independence`: la columna de patrocinadores de arriba es la fuente («base de datos del proyecto, consulta de patrocinadores a 2026-10-16»): organismo público o cliente gubernamental o de partido → 0; un único medio → 1; varios clientes o autofinanciada → 2. (5) `standards`: listado de asociados de Insights+Analytics España, directorio de ESOMAR o certificado ISO 20252 en su web → 1; si no se encuentra → 0. Lo que no se encuentre vale 0 con `notes` «no comprobado: <criterio>». `status = borrador`, `checked_at` = la fecha de la comprobación. En `pollster` el nombre exacto de la lista (es el de la base).

- [ ] **Step 2: Validar el fichero**

Run: `pytest tests/test_data_files.py tests/test_quality_unit.py -q`
Expected: PASS; además `python -c "from mtpy import mtpy; mtpy.run(); from mtpy.lib.quality import read_quality; print(read_quality()[['pollster', 'points', 'quality', 'status']].to_string())"` lista las 72 filas (si `mtpy.run()` exige base, usa `App` con el directorio `data/` como en los tests). Comprueba a mano que CIS suma 7 (56) y ElectoPanel 4 (32) como dice la spec, o anota en el commit por qué no.

- [ ] **Step 3: Commit**

```bash
git add data/pollsters-quality.csv
git commit -m "Add the draft quality evaluation of the pollsters with polls since 2019"
```

---

### Task 8: Cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-10-16-calidad-previa-design.md` (línea de estado)

- [ ] **Step 1: Suite completa**

Run: `pytest -q` (los de integración se saltan sin base)
Expected: PASS; anota el número de tests.

- [ ] **Step 2: Cerrar la spec**

Línea de estado: «Estado: cerrada el {fecha} en `dev` (plan `docs/superpowers/plans/2026-10-16-calidad-previa.md`, commits desde `9e83989`; N tests). Pendiente de Luis: D6 (réplica, recálculo de ratings, publicación, comprobación de la tabla de D3 y revisión de las filas `borrador`).»

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-16-calidad-previa-design.md
git commit -m "Close the prior quality spec"
```
