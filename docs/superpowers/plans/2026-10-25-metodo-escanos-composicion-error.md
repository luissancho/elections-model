# Los tres errores con cifras vivas en /metodo/escanos — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar en la parte `fan` del run las tres piezas del error de cada partido (`err`, `pct_err`, `drift`) y
enseñarlas en una tarjeta nueva de /metodo/escanos con los números del último cálculo de las generales.

**Architecture:** `Simulator._fan` ya calcula las piezas y las tira; se conservan en `fan()` y `export_fan` las
publica junto a `dof`. `export_meta` añade a `diagnostics` las elecciones que ajustan el estimador de error. La
página lee el último run nacional con un `error_example()` gemelo de `weights_example()` y la plantilla pinta la
tabla; sin las columnas (run anterior a este cambio) la tarjeta se oculta.

**Tech Stack:** Python 3.11, pandas, Jinja2 (plantillas en `web/templates`), pytest (`python -m pytest`).

**Spec:** `docs/superpowers/specs/2026-10-25-metodo-escanos-composicion-error-design.md`

## Global Constraints

- Redacción pública (reglas de Luis): «sólo» (solamente) siempre con tilde; «encuestadora» o «casa», nunca
  «empresa» ni «patrocinador»; ningún `<details class="how" open`; comprobado por `assert_public_wording`.
- En /metodo todo a la vista: prosa visible, sin «Cómo leerlo» plegado (regla de Luis para /metodo, 2026-10-21);
  la spec dice «Cómo leerlo» con tres ideas y aquí van como párrafos visibles.
- Nombres de código en inglés neutro; textos públicos en español. Docstrings en todas las funciones (AGENTS.md).
- `mtpy/lib/pages` no importa el modelo (los workers web van ligeros).
- Esquemas del paquete (`bundle.SCHEMAS`): `fan` y `meta` sólo validan las claves de primer nivel; no cambian.
- Sólo las generales (`NATIONAL_SCOPE`) y sólo el último run, como el resto de cifras vivas del método.

## Review Focus

- Un run publicado antes de este cambio (su `fan` sin `pct_err`, su `meta.diagnostics` sin `error_events`): la
  página debe renderizar sin la tarjeta ni su entrada de navegación, nunca un 500 (Tarea 3 y Tarea 4).
- Un `fan` con un solo horizonte (run del día de la elección, `horizon_max` 0): `drift` 0, `sd_election` igual a
  `sd_now`, la tabla sigue saliendo (Tarea 3).
- Un partido de `bmaps.main` sin fila en el `fan` (sin ajuste, `mean` 0): se omite sin romper (Tarea 3).
- `Simulator.fan(wide=True)`: la tabla ancha no debe cambiar al añadir columnas largas (Tarea 1).
- `error_events()` en un simulador sin estimador de error (`v2err` None, ámbito sin elecciones pasadas): lista
  vacía, y `export_meta` publica `[]` (Tareas 1 y 2).

---

### Task 1: Las piezas del error en `Simulator.fan` y las elecciones del estimador

**Files:**
- Modify: `mtpy/lib/simulator.py:1014-1091` (`fan`, `_fan`) y un método nuevo `error_events` junto a
  `fit_error_estimator` (`:433`).
- Test: `tests/test_simulator_unit.py` (al final, donde ya se importa `synthetic_simulator` de `tests.fakes`).

**Interfaces:**
- Consumes: `tests.fakes.synthetic_simulator()` (Simulator por `__new__`, `v2err = v2drift = None`, `forecast`
  con `err` 1,5/1,2/1,0 y `nobs` 40/40/30 para PP/PSOE/VOX, `horizon_max` 55).
- Produces: `Simulator._fan(horizons, alpha=None) -> pd.DataFrame` con columnas
  `['party', 'horizon', 'mean', 'err', 'pct_err', 'drift', 'sd', 'lo', 'hi', 'dof']`; `Simulator.fan()` (largo)
  devuelve las mismas columnas (ya no tira `dof`); `fan(wide=True)` sin cambios;
  `Simulator.error_events() -> list[str]` (fechas ISO ascendentes, `[]` sin `v2err`).

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_fan_carries_the_three_errors_and_the_dof():
    sim = synthetic_simulator()
    sim.v2err = types.SimpleNamespace(predict=lambda p: np.array([2.0]))
    sim.v2drift = types.SimpleNamespace(var=lambda d, level, age=None: 1e-4 * d * level ** 2)
    fan = sim.fan(horizons=[0, 55])
    assert list(fan.columns) == ['party', 'horizon', 'mean', 'err', 'pct_err', 'drift', 'sd', 'lo', 'hi', 'dof']
    pp0 = fan[(fan.party == 'PP') & (fan.horizon == 0)].iloc[0]
    pp55 = fan[(fan.party == 'PP') & (fan.horizon == 55)].iloc[0]
    assert (pp0.err, pp0.pct_err, pp0.drift, pp0.dof) == (1.5, 2.0, 0.0, 38.0)
    assert pp0.sd == pytest.approx(math.sqrt(1.5 ** 2 + 2.0 ** 2))
    assert pp55.drift == pytest.approx(40 * math.sqrt(1e-4 * 55))
    assert pp55.sd == pytest.approx(math.sqrt(pp55.err ** 2 + pp55.pct_err ** 2 + pp55.drift ** 2))
    wide = sim.fan(horizons=[0, 55], wide=True)
    assert list(wide.columns.get_level_values(0).unique()) == ['lo', 'hi']  # the wide table does not change


def test_error_events_lists_the_elections_of_the_error_estimator():
    sim = synthetic_simulator()
    assert sim.error_events() == []  # no estimator
    sim.v2err = types.SimpleNamespace(predict=lambda p: np.array([2.0]))
    sim.n_last = 1
    index = pd.MultiIndex.from_tuples([(pd.Timestamp('2023-07-23'), pd.Timestamp('2023-07-16'), 1),
                                       (pd.Timestamp('2019-11-10'), pd.Timestamp('2019-11-03'), 1),
                                       (pd.Timestamp('2023-07-23'), pd.Timestamp('2023-07-15'), 2)],
                                      names=['event_date', 'date', 'pollster_id'])
    seen = {}
    def filter_polls(featured=True, **kwargs):
        seen.update(kwargs, featured=featured)
        return pd.DataFrame({'PP': [1., 2., 3.]}, index=index)
    sim.computer = types.SimpleNamespace(filter_polls=filter_polls)
    assert sim.error_events() == ['2019-11-10', '2023-07-23']
    assert seen == {'featured': True, 'drange': Simulator.ERROR_DRANGE, 'n_last': 1}
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_simulator_unit.py -q -k "three_errors or error_events"`
Expected: FAIL (`list(fan.columns)` sin `err`; `AttributeError: 'Simulator' object has no attribute 'error_events'`).

- [ ] **Step 3: Implementar en `mtpy/lib/simulator.py`**

`_fan`: en cada fila añadir `'err': float(np.nan_to_num(err))`, `'pct_err': pct_err`, `'drift': sqrt(drift_var)`;
columnas en el orden de Interfaces. `fan`: quitar el `.drop(columns='dof')`; actualizar el docstring (Returns).
`error_events(self) -> list[str]`: `[]` si `self.v2err is None`; si no, `self.computer.filter_polls(featured=True,
drange=self.ERROR_DRANGE, n_last=self.n_last)` y las fechas distintas del nivel `event_date` del índice (o la columna
`event_date` si no es nivel, como hace `get_drift_data`), ordenadas, en `'%Y-%m-%d'`. Docstring: «las elecciones
cuyos sondeos ajustan el estimador de error (`fit_error_estimator`)».

- [ ] **Step 4: Comprobar que pasan y que nada más se rompe**

Run: `python -m pytest tests/test_simulator_unit.py tests/test_publish_unit.py -q`
Expected: los dos nuevos en PASS; `test_export_fan_house_effects_and_dispersion` FALLA ahora por las columnas
nuevas (se arregla en la Tarea 2).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/simulator.py tests/test_simulator_unit.py
git commit -m "Simulator: el abanico conserva s, e, d y dof por horizonte; error_events() del estimador de error"
```

---

### Task 2: Publicar las piezas en `fan` y el estimador en `meta.diagnostics`; contrato

**Files:**
- Modify: `mtpy/lib/publish/forecast.py:370-385` (`export_fan`), `:555-556` (`diagnostics` en `export_meta`).
- Modify: `tests/fakes.py:113` (`synthetic_simulator`: añadir `sim.est_scope = 'es'`, `sim.computer = None`,
  `sim.n_last = 1`).
- Modify: `docs/web/contrato.md:86` (`diagnostics`) y `:145-151` (`### fan`).
- Test: `tests/test_publish_unit.py:267-273` y `:351-367`.

**Interfaces:**
- Consumes: `Simulator.fan()` y `Simulator.error_events()` de la Tarea 1; `sim.est_scope` (ámbito estimador).
- Produces: parte `fan` con filas `name, horizon, mean, sd, lo, hi, err, pct_err, drift, dof` (`err`, `pct_err`,
  `drift` redondeados como `mean`, `dof` entero); CSV con esas columnas en ese orden; `meta.diagnostics` con
  `error_events: list[str]` y `error_scope: str`.

- [ ] **Step 1: Actualizar los tests para que fallen**

En `test_export_fan_house_effects_and_dispersion`:

```python
    assert set(fan['rows'][0]) == {'name', 'horizon', 'mean', 'sd', 'lo', 'hi', 'err', 'pct_err', 'drift', 'dof'}
    assert fan['rows'][0]['dof'] == 38 and fan['rows'][0]['drift'] == 0.0 and fan['rows'][0]['err'] == 1.5
    assert list(publish.export_fan(sim)[1].columns) == ['name', 'horizon', 'mean', 'sd', 'lo', 'hi', 'err', 'pct_err', 'drift', 'dof']
```

En `test_export_meta_and_headline`, el `diagnostics` esperado gana `'error_events': [], 'error_scope': 'es'`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py -q -k "export_fan or export_meta_and_headline"`
Expected: FAIL (claves y columnas que faltan; `diagnostics` sin `error_events`).

- [ ] **Step 3: Implementar**

`export_fan`: `round_cols(sim.fan(), pct=('mean', 'sd', 'lo', 'hi', 'err', 'pct_err', 'drift'))`, `dof` a `int`,
columnas en el orden de Interfaces (las seis antiguas primero). `export_meta`: en `diagnostics`,
`'error_events': sim.error_events()` y `'error_scope': sim.est_scope`. `tests/fakes.py`: los tres atributos del
sintético. Docstrings de los exportadores con las columnas nuevas.

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_simulator_unit.py tests/test_jobs_publish.py -q`
Expected: PASS.

- [ ] **Step 5: Enmendar `docs/web/contrato.md`**

En `### fan`: «`rows` … `mean`, `sd`, `lo`, `hi` y, desde la enmienda 2026-10-25, `err` (error estándar del
promedio, `s`), `pct_err` (error histórico de los sondeos, `e`), `drift` (deriva en ese horizonte, `d`, 0 en el
horizonte 0) y `dof` (grados de libertad de la t), en puntos: `sd² = err² + pct_err² + drift²`; los runs anteriores
no las traen». En `diagnostics`: añadir «`error_events` (fechas ISO ascendentes de las elecciones cuyos sondeos
ajustan el estimador de error) y `error_scope` (el ámbito estimador, `es` también en las autonómicas), enmienda
2026-10-25».

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish/forecast.py tests/fakes.py tests/test_publish_unit.py docs/web/contrato.md
git commit -m "publish: el fan publica s, e, d y dof; meta.diagnostics lleva las elecciones del estimador de error"
```

---

### Task 3: `error_example()` y el contexto de /metodo/escanos

**Files:**
- Modify: `tests/fixtures/bundle/fan.json` y `tests/fixtures/bundle/meta.json`.
- Modify: `mtpy/lib/pages/metodo.py` (función nueva junto a `weights_example`, `:162`; `metodo_section_context`,
  `:448`).
- Test: `tests/test_pages_ejemplo.py` (junto a los tests de `weights_example`).

**Interfaces:**
- Consumes: `part(NATIONAL_SCOPE, run, name)`, `site().latest_run`, `catalog(meta)` y `bundle_part` ya usados en
  `metodo.py`; `HttpError` 404 como «no publicado».
- Produces: `metodo.error_example() -> Optional[dict]` con `published` (`meta.run_at`), `event_date`, `horizon`
  (el mayor de `fan.horizons`), `n_events`, `first_year`, `last_year`, `scope` (`error_scope`) y `rows`
  (`name`, `fullname`, `color`, `pct`, `err`, `pct_err`, `drift`, `sd_now`, `sd_election`, `lo`, `hi`, `dof`;
  partidos de `meta.bmaps.main` por `pct` descendente); el contexto de la subpágina `escanos` lo expone como
  `errors`.

- [ ] **Step 1: Ampliar los fixtures**

`fan.json`: `horizons` `[0, 52]`; filas de PP
`{"name": "PP", "horizon": 0, "mean": 33.1, "sd": 1.0, "lo": 31.0, "hi": 35.2, "err": 0.5, "pct_err": 0.87, "drift": 0.0, "dof": 38}`
y `{"name": "PP", "horizon": 52, "mean": 33.1, "sd": 1.28, "lo": 30.5, "hi": 35.7, "err": 0.5, "pct_err": 0.87, "drift": 0.8, "dof": 38}`.
`meta.json`: en `diagnostics`, `"error_events": ["2015-12-20", "2016-06-26", "2019-04-28", "2019-11-10", "2023-07-23"]`
y `"error_scope": "es"`.

- [ ] **Step 2: Escribir los tests que fallan**

```python
def test_error_example_reads_the_three_errors_of_the_main_parties(api):
    example = metodo.error_example()
    pp = pages.catalog(fixture_data('meta'))['PP']
    assert {k: example[k] for k in ('published', 'event_date', 'horizon', 'n_events', 'first_year', 'last_year', 'scope')} == {
        'published': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'horizon': 52, 'n_events': 5,
        'first_year': '2015', 'last_year': '2023', 'scope': 'es'}
    assert example['rows'] == [{'name': 'PP', 'fullname': pp['fullname'], 'color': pp['color'], 'pct': 33.1, 'err': 0.5,
                                'pct_err': 0.87, 'drift': 0.8, 'sd_now': 1.0, 'sd_election': 1.28, 'lo': 30.5, 'hi': 35.7,
                                'dof': 38}]


def test_error_example_is_none_without_the_components_and_skips_parties_without_a_fit(api, monkeypatch):
    fan, meta = fixture_data('fan'), fixture_data('meta')
    old = {**fan, 'rows': [{k: v for k, v in row.items() if k not in ('err', 'pct_err', 'drift', 'dof')} for row in fan['rows']]}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: old if name == 'fan' else fixture_data(name))
    assert metodo.error_example() is None  # a run published before the columns existed
    without = {**meta, 'diagnostics': {k: v for k, v in meta['diagnostics'].items() if k != 'error_events'}}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: without if name == 'meta' else fixture_data(name))
    assert metodo.error_example() is None
    wider = {**meta, 'bmaps': {**meta['bmaps'], 'main': ['PP', 'XX']}}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: wider if name == 'meta' else fixture_data(name))
    assert [row['name'] for row in metodo.error_example()['rows']] == ['PP']  # XX has no fit: no row
    nowcast_only = {**fan, 'horizons': [0], 'rows': [row for row in fan['rows'] if row['horizon'] == 0]}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: nowcast_only if name == 'fan' else fixture_data(name))
    row = metodo.error_example()['rows'][0]
    assert metodo.error_example()['horizon'] == 0 and row['drift'] == 0.0 and row['sd_election'] == row['sd_now']


def test_error_example_lets_other_errors_through_and_is_none_without_a_run(api, monkeypatch):
    def boom(scope, run, name, mode=None):
        raise HttpError(503, 'bundle unavailable')
    monkeypatch.setattr(metodo, 'part', boom)
    with pytest.raises(HttpError):
        metodo.error_example()
    def not_found(scope, run, name, mode=None):
        raise HttpError(404, 'run not found')
    monkeypatch.setattr(metodo, 'part', not_found)
    assert metodo.error_example() is None
```

- [ ] **Step 3: Comprobar que fallan**

Run: `python -m pytest tests/test_pages_ejemplo.py -q -k error_example`
Expected: FAIL con `AttributeError: module ... has no attribute 'error_example'`.

- [ ] **Step 4: Implementar `error_example()` en `mtpy/lib/pages/metodo.py`**

Mismo patrón que `weights_example`: `run = site().latest_run(NATIONAL_SCOPE)`, `meta`, `fan` por `part`, 404 → `None`,
otros `HttpError` se propagan. `None` si ninguna fila del `fan` trae `pct_err` o si `diagnostics` no trae
`error_events`. `horizon = max(fan['horizons'])`; por cada nombre de `meta['bmaps']['main']` con fila en el
horizonte 0: `err`, `pct_err`, `dof`, `sd_now` de esa fila; `drift`, `sd_election`, `lo`, `hi` de la fila del
horizonte mayor (la misma fila cuando `horizon` es 0); `fullname` y `color` de `catalog(meta)`; ordenar por `pct`
descendente. `first_year`/`last_year`: `[:4]` del primero y último de `error_events`. En
`metodo_section_context`, `if section == 'escanos': context['errors'] = error_example()`, y añadirlo al docstring.

- [ ] **Step 5: Comprobar que pasan**

Run: `python -m pytest tests/test_pages_ejemplo.py tests/test_pages.py -q`
Expected: PASS (si algún test del abanico de Escaños fijaba `horizons == [0]`, actualizarlo a `[0, 52]`).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/bundle/fan.json tests/fixtures/bundle/meta.json mtpy/lib/pages/metodo.py tests/test_pages_ejemplo.py
git commit -m "Método: error_example() lee s, e y d del último run nacional para /metodo/escanos"
```

---

### Task 4: La tarjeta «Un ejemplo con datos publicados: los tres errores del …»

**Files:**
- Modify: `web/templates/metodo_escanos.html` (`:19-24` navegación; nueva tarjeta tras el `card_end` de
  `hoy-y-eleccion`, `:121`).
- Test: `tests/test_pages.py` (junto a `test_metodo_escanos_text`, `:2471`).

**Interfaces:**
- Consumes: `errors` del contexto (Tarea 3); macros `m.card_start(id, title, lead)`, `m.card_end()`; filtros
  `date`, `pct`, `num`, `range`, `int`.
- Produces: tarjeta `id="errores"` con la tabla `id="errors-table"` (clase `et et-fit`) y el enlace
  `href="#errores"` en la navegación, ambos sólo con `errors`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_metodo_escanos_shows_the_three_errors_of_the_latest_run(api):
    page = html(call(api, '/metodo/escanos')[2])
    assert 'href="#errores">Los tres errores del último cálculo</a>' in page and 'id="errores"' in page
    assert 'Un ejemplo con datos publicados: los tres errores del 8 oct 2026' in page
    table = re.search(r'<table[^>]*id="errors-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', table) == [
        'Partido', 'Voto', 'Promedio (s)', 'Histórico (e)', 'Deriva (d), 52 días', 'Hoy (σ)', 'Elección (σ)',
        'Rango 95% en Elección']
    row = re.sub(r'\s+', ' ', re.findall(r'<tr>(.*?)</tr>', table, re.S)[1])
    for cell in ('PP', '33,1%', '0,50', '0,87', '0,80', '1,00', '1,28', '30,5–35,7'):
        assert cell in row, cell
    text = re.sub(r'\s+', ' ', page)
    assert 'las 5 elecciones generales con sondeos celebradas de 2015 a 2023' in text
    assert 'No tiene dirección' in text and 'los 52 días que faltan hasta el 29 nov 2026' in text
    assert_public_wording(page, '/metodo/escanos')


def test_metodo_escanos_hides_the_errors_card_on_an_old_run(api, monkeypatch):
    monkeypatch.setattr(pages.metodo, 'error_example', lambda: None)
    page = html(call(api, '/metodo/escanos')[2])
    assert 'id="errores"' not in page and '#errores' not in page and 'los tres errores' not in page
```

Y en `test_metodo_escanos_text`, añadir `'errores'` a la tupla de anclas.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_pages.py -q -k "metodo_escanos"`
Expected: FAIL (`id="errores"` ausente).

- [ ] **Step 3: Implementar la plantilla**

Navegación: tras `<a href="#hoy-y-eleccion">`, `{% if errors %}<a href="#errores">Los tres errores del último cálculo</a>{% endif %}`.
Tarjeta, tras el `card_end` de «Hoy y Elección», dentro de `{% if errors %}`:

```jinja
{{ m.card_start('errores', 'Un ejemplo con datos publicados: los tres errores del ' ~ (errors.published|date),
                'Las tres piezas del margen de cada partido principal en el último cálculo de las generales, en puntos de voto.') }}
  <div class="table-wrap">
    <table class="et et-fit" id="errors-table">
      <caption class="sr-only">Error del promedio, error histórico y deriva de cada partido principal, con el margen total de cada modo.</caption>
      <thead><tr>
        <th scope="col">Partido</th><th scope="col">Voto</th><th scope="col">Promedio (s)</th>
        <th scope="col">Histórico (e)</th><th scope="col">Deriva (d), {{ errors.horizon|int }} días</th>
        <th scope="col">Hoy (σ)</th><th scope="col">Elección (σ)</th><th scope="col">Rango 95% en Elección</th>
      </tr></thead>
      <tbody>
        {% for row in errors.rows %}
        <tr><th scope="row" style="--c: {{ row.color }}">{{ row.name }}</th><td>{{ row.pct|pct(1) }}</td>
            <td>{{ row.err|num(2) }}</td><td>{{ row.pct_err|num(2) }}</td><td>{{ row.drift|num(2) }}</td>
            <td>{{ row.sd_now|num(2) }}</td><td>{{ row.sd_election|num(2) }}</td><td>{{ row.lo|range(row.hi) }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <p>
    Cada fila descompone el margen de un partido: <var>s</var> es la banda del promedio; <var>e</var>, lo que se
    equivocaron los sondeos de la última semana en las elecciones pasadas para un partido de ese tamaño; y
    <var>d</var>, lo que la opinión puede moverse en los {{ errors.horizon|int }} días que faltan hasta el
    {{ errors.event_date|date }}. Hoy suma las dos primeras y Elección las tres, siempre en cuadratura: la raíz de
    la suma de los cuadrados.
  </p>
  <p>
    <strong>De dónde sale <var>e</var>.</strong> De la última encuesta de cada encuestadora antes de cada una de
    las {{ errors.n_events|int }} elecciones generales con sondeos celebradas de {{ errors.first_year }} a
    {{ errors.last_year }}, todas con el mismo peso y con sus fallos grandes dentro. Por eso vale varias veces la
    banda del promedio: la banda dice cuánto puede equivocarse la línea al resumir los sondeos; <var>e</var>,
    cuánto se han equivocado los sondeos frente a las urnas.
  </p>
  <p>
    <strong>No tiene dirección.</strong> El modelo no sabe hacia qué lado se equivocarán los sondeos esta vez,
    sólo cuánto suelen hacerlo: el sorteo reparte ese error a los dos lados del voto estimado.
  </p>
  <p>
    <strong>Las colas.</strong> El sorteo usa una t de Student cuyos grados de libertad salen del número de
    sondeos del partido, muchos en los grandes, así que sus colas son casi las de una campana normal.
  </p>
{{ m.card_end() }}
```

Si la plantilla usa otra forma de colorear la celda del partido en sus tablas (`thresholds-table`), seguirla.

- [ ] **Step 4: Comprobar que pasan**

Run: `python -m pytest tests/test_pages.py tests/test_pages_ejemplo.py -q`
Expected: PASS, incluido `assert_public_wording`.

- [ ] **Step 5: Commit**

```bash
git add web/templates/metodo_escanos.html tests/test_pages.py
git commit -m "Método: tarjeta con los tres errores del último cálculo en /metodo/escanos"
```

---

### Task 5: Cierre

**Files:**
- Modify: `docs/superpowers/specs/2026-10-25-metodo-escanos-composicion-error-design.md` (Estado).

- [ ] **Step 1: Suite completa**

Run: `python -m pytest -q -m "not integration" -p no:cacheprovider`
Expected: todo en verde (1345 antes de este plan, más los nuevos).

- [ ] **Step 2: Cerrar la spec**

Estado: «construido el 2026-10-25 en `dev`; la tarjeta aparece al publicar un run de las generales con el
código nuevo» y las desviaciones (prosa visible en vez de «Cómo leerlo»; `dof` entero).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-25-metodo-escanos-composicion-error-design.md
git commit -m "Spec: cierre de los tres errores con cifras vivas"
```
