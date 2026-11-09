# Tercera tarjeta «¿Podría ser azar?»: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir al estudio `rebano` la prueba de hipótesis nula «las tres últimas semanas = el resto del ciclo» con
20.000 barajas, su distribución nula, su p a dos colas y el intervalo del efecto, publicarla como parte `chance` y
pintarla como tercera tarjeta de la página con un histograma.

**Architecture:** `stats.py` gana `pooled_terms`, `bootstrap_pooled`, `null_distribution` e `histogram`; `study.py`
compone `HerdStudy.chance`; el paquete declara la parte y su CSV; la página gana el veredicto `q3`, los textos y el
gráfico `chance.js`.

**Tech Stack:** Python 3.11, pandas, numpy; Jinja2; ECharts 5.6 (vendor); pytest con `node` para los gráficos.

**Spec:** `docs/superpowers/specs/2026-11-09-web-analisis-rebano-azar-design.md`.

## Global Constraints

- Nada cambia en las tarjetas 1 y 2, los KPIs, el subtítulo, `summary`, `elections`, `current` ni `weeks`.
- Parámetros exactos: `"chance": {"periods": {"final": [0, 21], "rest": [21, 365]}, "permutations": 20000, "bin_width":
  0.01, "resamples": 2000}`; `min_rows` 3 y `seed` 0 del estudio.
- La p publicada es a dos colas con la corrección `(k + 1)/(B + 1)`; `p_one` también corregida.
- Redondeos: `r_a`/`r_b` 2 decimales, `change`/`lo`/`hi` 3, `null.mean/sd/min/max` 4, `p` con `_sig3`.
- La página condiciona la tarjeta 3, su `answer` y la frase del método a que exista la parte `chance`.
- Textos públicos exactamente los de la spec; «sólo» con tilde, casa/encuestadora, nunca empresa; nada de «se copian»,
  «vecino», «permite el azar»; `assert_public_wording` pasa. «Hipótesis nula» y «p» se permiten en esta tarjeta.
- Código en inglés neutro, docstrings numpy, PEP8 a 120 columnas; `python -m pytest -q` verde en cada tarea.

## Review Focus

1. Un remuestreo del bootstrap puede dar `ΣA` 0 → división por cero: el cambio de ese remuestreo se descarta y, si
   quedan menos de 3 remuestreos válidos, `None`. → Tarea 1.
2. `histogram` con un solo valor o todos iguales: una caja, no un bucle infinito ni un rango vacío. → Tarea 1.
3. `null_distribution` con `observed` 0 exacto (cambio nulo): `count_two` cuenta todas las barajas (|x| ≥ 0) y `p_two`
   ≈ 1; sin error. → Tarea 1.
4. La página con `chance` presente pero `ci` o `null.sd` nulos: textos sin paréntesis y sin «desviaciones», nunca
   error. → Tarea 4.
5. El eje x del gráfico debe incluir siempre el observado aunque esté fuera de las cajas (si no, la línea no se ve).
   → Tarea 4 (test JS del rango).

---

### Task 1: Estadística: términos por elección, bootstrap del cambio conjunto, distribución nula e histograma

**Files:**
- Modify: `mtpy/lib/studies/herd/stats.py` (tras `pooled_p`), `data/rebano/params.json`, `data/README.md:32`
- Test: `tests/test_herd_stats.py`

**Interfaces:**
- Consumes: `_shared_pairs`, `_weighted_sums`, `_permuted_terms`, `PERMUTATION_TOLERANCE`, `compare`.
- Produces: `pooled_terms(rows_by_election, a, b, min_rows) -> list[tuple[float, float]]` (`(A_e, B_e)` por elección
  con parejas compartidas); `bootstrap_pooled(terms, resamples, seed) -> Optional[dict]` (`{lo, hi}` sin redondear;
  `None` con menos de 3 elecciones o menos de 3 remuestreos válidos); `null_distribution(rows_by_election, a, b,
  min_rows, permutations, seed) -> Optional[dict]` (`{observed, permuted (np.ndarray), count_one, count_two, p_one,
  p_two}`); `histogram(values, bin_width) -> list[dict]` (`{lo, hi, count}`).

- [ ] **Step 1: Write the failing tests**

```python
def test_pooled_terms_are_the_weighted_sums_of_each_election():
    e1, e2 = two_elections()
    terms = pooled_terms([e1, e2], 'far', 'final', 3)
    assert terms[0] == pytest.approx((6.0, 1.5)) and terms[1] == pytest.approx((24 / 7, 12 / 7))
    assert pooled_terms([e1.iloc[:0]], 'far', 'final', 3) == []
    # the pooled change follows from the terms: sqrt(ΣB/ΣA) − 1
    assert (sum(b for _, b in terms) / sum(a for a, _ in terms)) ** 0.5 - 1 == pytest.approx(-0.4161, abs=1e-4)


def test_bootstrap_pooled_interval_is_deterministic_and_brackets_the_change():
    terms = [(6.0, 1.5), (24 / 7, 12 / 7), (4.0, 3.0), (2.0, 2.5)]
    ci = bootstrap_pooled(terms, 2000, 0)
    change = (sum(b for _, b in terms) / sum(a for a, _ in terms)) ** 0.5 - 1
    assert ci['lo'] <= change <= ci['hi'] and bootstrap_pooled(terms, 2000, 0) == ci
    assert bootstrap_pooled(terms[:2], 2000, 0) is None
    assert bootstrap_pooled([(0.0, 0.0)] * 4, 50, 0) is None          # every resample divides by zero


def test_null_distribution_matches_the_exact_enumeration():
    # the two-election, four-block case of test_pooled_p_matches_the_exact_enumeration: enumerate the 36 arrangements
    # (statistic Σw·r²_b / Σw·r²_a), p_one = share ≤ observed, p_two = share with |√ratio − 1| ≥ |observed change|
    ...
    out = null_distribution(rows, 'far', 'final', 3, 4000, 0)
    assert out['permuted'].shape == (4000,) and isinstance(out['count_one'], int)
    assert abs(out['p_one'] - exact_one) < 0.03 and abs(out['p_two'] - exact_two) < 0.03
    assert out['p_one'] == pytest.approx((out['count_one'] + 1) / 4001)
    assert null_distribution([], 'far', 'final', 3, 100, 0) is None


def test_null_distribution_with_a_zero_change_has_p_two_near_one():
    # identical far and final rows: observed change 0, every arrangement gives |change| ≥ 0
    rows = pd.DataFrame({'pair': ['A|B'] * 8, 'poll_i': ['a%d' % i for i in range(8)], 'poll_j': ['b%d' % i for i in range(8)],
                         'period': ['far'] * 4 + ['final'] * 4, 'd2': [2.0] * 8, 'v': [1.0] * 8})
    out = null_distribution([rows], 'far', 'final', 3, 500, 0)
    assert out['observed'] == pytest.approx(0.0) and out['p_two'] == pytest.approx(1.0)


def test_histogram_bins_cover_the_values_with_fixed_width():
    assert histogram([-0.012, 0.004, 0.031], 0.01) == [
        {'lo': -0.02, 'hi': -0.01, 'count': 1}, {'lo': -0.01, 'hi': 0.0, 'count': 0}, {'lo': 0.0, 'hi': 0.01, 'count': 1},
        {'lo': 0.01, 'hi': 0.02, 'count': 0}, {'lo': 0.02, 'hi': 0.03, 'count': 0}, {'lo': 0.03, 'hi': 0.04, 'count': 1}]
    assert histogram([0.05, 0.05], 0.01) == [{'lo': 0.05, 'hi': 0.06, 'count': 2}]
    assert histogram([], 0.01) == []
```

(`histogram([0.05, 0.05], 0.01)`: `floor(0.05/0.01)·0.01 = 0.05`, `ceil(0.05/0.01)·0.01 = 0.05` → una sola caja de
`0.05` a `0.06` con los dos valores; cuidado con los flotantes: redondear los bordes a los decimales de `bin_width` y
usar `np.histogram` con esos bordes, con la última caja cerrada por la derecha.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_herd_stats.py`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

`pooled_terms` reutiliza la suma por pareja de `pooled` (puede refactorizarse `pooled` para que llame a
`pooled_terms` y conserve sus cifras: `pairs`, `rows_*` siguen saliendo de las parejas compartidas). `bootstrap_pooled`
con `np.random.default_rng(seed).integers`, descarta remuestreos con `ΣA` 0. `null_distribution` suma los
`_permuted_terms` de cada elección con un generador y calcula los cambios permutados `np.sqrt(perm_b / perm_a) − 1`
(con `np.errstate`), los dos recuentos y las p corregidas. `histogram` con `np.histogram`.

- [ ] **Step 4: Parámetros**

`data/rebano/params.json`: añadir `"chance": {...}` (tras `robustness`, antes de `weeks`). `data/README.md:32`:
añadir «tramos, barajas, anchura de caja y remuestreos de la prueba de hipótesis nula».

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q tests/test_herd_stats.py tests/test_herd_study.py tests/test_publish_herd.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/studies/herd/stats.py data/rebano/params.json data/README.md tests/test_herd_stats.py
git commit -m "Rebaño: términos por elección, bootstrap del cambio conjunto, distribución nula e histograma"
```

---

### Task 2: El estudio: `HerdStudy.chance`

**Files:**
- Modify: `mtpy/lib/studies/herd/study.py` (`HerdStudy`, `compute_herd`, función nueva `_chance(elections, params)`)
- Test: `tests/test_herd_study.py`

**Interfaces:**
- Consumes: `comparisons`, `compare`, `pooled`, `pooled_terms`, `bootstrap_pooled`, `null_distribution`, `histogram`,
  `_sig3` (Tarea 1).
- Produces: `HerdStudy.chance` (dict o `None`) con `periods`, `permutations`, `elections`, `falls`, `pairs`, `rows_a`,
  `rows_b`, `r_a`, `r_b`, `change`, `ci`, `count_one`, `count_two`, `p_one`, `p_two`, `null` (`mean`, `sd`, `min`, `max`, `bin_width`, `bins`).

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_herd_chance(fresh_app, tmp_path):
    c = compute(tmp_path).chance
    assert c['periods'] == {'final': [0, 21], 'rest': [21, 365]} and c['permutations'] == 20000
    assert (c['elections'], c['falls'], c['pairs'], c['rows_a'], c['rows_b'], c['r_a'], c['r_b']) == (2, 1, 2, 12, 10, 0.94, 0.77)
    assert c['change'] == pytest.approx(-0.1775, abs=6e-4) and c['ci'] is None
    assert abs(c['p_one'] - 22 / 150) < 0.01 and abs(c['p_two'] - 65 / 150) < 0.015      # exact: 150 arrangements
    null = c['null']
    assert abs(null['min'] - (-0.542)) < 0.01 and abs(null['max'] - 1.015) < 0.01 and null['bin_width'] == 0.01
    assert sum(b['count'] for b in null['bins']) == 20000 and null['bins'][0]['lo'] <= null['min'] < null['bins'][0]['hi']
    assert set(c) == {'periods', 'permutations', 'elections', 'falls', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b',
                      'change', 'ci', 'count_one', 'count_two', 'p_one', 'p_two', 'null'}
    assert isinstance(c['count_one'], int) and c['count_two'] >= c['count_one'] >= 0


def test_compute_herd_chance_is_none_without_shared_pairs(fresh_app, tmp_path):
    params = {'chance': {'periods': {'final': [0, 1], 'rest': [1, 2]}, 'permutations': 100, 'bin_width': 0.01, 'resamples': 100}}
    assert compute(tmp_path, params=params).chance is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_herd_study.py -k chance`
Expected: FAIL (`AttributeError: chance`).

- [ ] **Step 3: Implement**

`_chance(elections, params)`: comparaciones con `params | {'periods': chance['periods']}` por elección pasada;
`compare(rows, 'rest', 'final', min_rows)` por elección para `falls` y `elections`; `pooled` para las cifras;
`pooled_terms` → `bootstrap_pooled(terms, chance['resamples'], seed)`; `null_distribution(..., chance['permutations'],
seed)`; `null` con `mean`, `sd` (`ddof` 0), `min`, `max` de `permuted` a 4 decimales y `histogram(permuted,
bin_width)`. `HerdStudy` gana el campo `chance` y su docstring. `compute_herd` lo rellena (la elección en curso no
entra).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_herd_study.py tests/test_publish_herd.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/study.py tests/test_herd_study.py
git commit -m "Rebaño: prueba de hipótesis nula (tres últimas semanas frente al resto del ciclo) en el estudio"
```

---

### Task 3: La parte `chance`: esquema, exportador, CSV, contrato, fixture

**Files:**
- Modify: `mtpy/lib/bundle.py` (`STUDY_PARTS`, `STUDY_CSV`, `STUDY_CSV_OF`, `SCHEMAS`), `mtpy/lib/publish/herd.py`
  (`CHANCE_COLUMNS`, `export_herd_chance`), `mtpy/lib/publish/studies.py` (`STUDY_EXPORTERS`),
  `mtpy/lib/publish/__init__.py`, `mtpy/lib/pages/analisis.py` (`CSV_LABELS`), `docs/web/contrato.md`
- Create: `tests/fixtures/bundle/rebano-chance.json`
- Test: `tests/test_publish_herd.py`, `tests/test_bundle_unit.py` (si enumera partes o CSV), `tests/test_pages.py`
  (la cuenta de píldoras CSV pasa de 3 a 4, con comentario)

**Interfaces:**
- Consumes: `HerdStudy.chance` (Tarea 2).
- Produces: `STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current', 'weeks', 'chance')`;
  `SCHEMAS['rebano-chance'] = {'chance': (dict, type(None))}`; `STUDY_CSV['rebano'] = ('elections', 'weeks',
  'robustness', 'chance')`; `STUDY_CSV_OF['rebano']['chance'] = ('chance',)`; `CHANCE_COLUMNS = ['lo', 'hi', 'count']`;
  `export_herd_chance(study) -> ({'chance': ...}, {'chance': frame})`; `CSV_LABELS['rebano']['chance']` verbatim de la
  spec.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_herd_chance_validates_and_flattens_the_bins(study):
    chance, frames = publish.export_herd_chance(study)
    assert valid('rebano-chance', chance) and chance['chance']['permutations'] == 20000
    frame = frames['chance']
    assert list(frame.columns) == CHANCE_COLUMNS and str(frame['count'].dtype) == 'int64'
    assert len(frame) == len(chance['chance']['null']['bins']) and int(frame['count'].sum()) == 20000


def test_export_herd_chance_without_shared_pairs_is_null(study_without_pairs):   # o construir el estudio con params
    chance, frames = publish.export_herd_chance(study_without_pairs)
    assert valid('rebano-chance', chance) and chance['chance'] is None and list(frames['chance'].columns) == CHANCE_COLUMNS and frames['chance'].empty
```

Y `test_publish_study_rebano_writes_every_part` ya recorre `STUDY_PARTS`: añadir `fs.exists(path_study_csv('rebano',
'chance'))`. Fixture `rebano-chance.json` (sobre `rebano-chance@1`, `scope` nulo, `generated_at` 2026-11-09):

```
chance: periods {final [0, 21], rest [21, 365]}, permutations 20000, elections 34, falls 25, pairs 475, rows_a 4530,
rows_b 2654, r_a 1.58, r_b 1.17, change -0.263, ci {lo -0.383, hi -0.036}, count_one 0, count_two 0, p_one 0.00005,
p_two 0.00005,
null {mean 0.0, sd 0.031, min -0.119, max 0.148, bin_width 0.01, bins: [lo, hi, count] =
[-0.12,-0.11,2], [-0.11,-0.1,8], [-0.1,-0.09,24], [-0.09,-0.08,61], [-0.08,-0.07,140], [-0.07,-0.06,289],
[-0.06,-0.05,538], [-0.05,-0.04,901], [-0.04,-0.03,1362], [-0.03,-0.02,1856], [-0.02,-0.01,2282], [-0.01,0.0,2544],
[0.0,0.01,2530], [0.01,0.02,2282], [0.02,0.03,1856], [0.03,0.04,1362], [0.04,0.05,901], [0.05,0.06,538],
[0.06,0.07,289], [0.07,0.08,140], [0.08,0.09,61], [0.09,0.1,24], [0.1,0.11,8], [0.11,0.12,2], [0.12,0.13,0],
[0.13,0.14,0], [0.14,0.15,0], [0.15,0.16,0]}   (28 cajas, suman 20.000)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_publish_herd.py`
Expected: FAIL.

- [ ] **Step 3: Implement**

Declaraciones en `bundle.py`; `export_herd_chance` en `publish/herd.py` (frame de las cajas con `count` `int64`;
vacío con las columnas sin `chance`); registro en `STUDY_EXPORTERS['rebano']['chance']` y en `publish/__init__.py`;
`CSV_LABELS`; `contrato.md` (fila de la parte `chance`: `chance` (dict o `null`) con sus claves; la lista de partes
«Seis partes»; las rutas con `chance` y su CSV). `grep -n rebano tests/test_bundle_unit.py` por si enumera.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_publish_herd.py tests/test_bundle_unit.py tests/test_pages.py -k "herd or rebano or bundle"`
Expected: PASS (actualizar la cuenta de píldoras CSV a 4 con `# elections + weeks + robustness + chance CSVs`).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/herd.py mtpy/lib/publish/studies.py mtpy/lib/publish/__init__.py mtpy/lib/pages/analisis.py docs/web/contrato.md tests/fixtures/bundle/rebano-chance.json tests/test_publish_herd.py tests/test_pages.py tests/test_bundle_unit.py
git commit -m "Rebaño: parte chance (prueba de hipótesis nula) con su CSV y contrato"
```

---

### Task 4: La página: tarjeta 3, veredicto, textos, histograma

**Files:**
- Modify: `mtpy/lib/pages/herd.py` (`verdicts` → `q3`; `p_text`, `null_text`, `null_summary_text`, `sigmas`),
  `mtpy/lib/pages/analisis.py` (`herd_context`), `web/templates/analisis_rebano.html`,
  `web/dist/js/pages/analisis-rebano.js`, `web/dist/css/site.css` (altura de `#chance-chart`, 260 px / 220 en móvil)
- Create: `web/dist/js/charts/chance.js`
- Test: `tests/test_pages.py`, `tests/test_js_charts.py` (`CHANCE_JS`)

**Interfaces:**
- Consumes: la parte `chance` (fixture de la Tarea 3); `percent`, `fmt_int`, `fmt_num`, `ci_text`, `_get`, `_is_num`,
  `DASH`; `mountChart`, `baseOption`, `axisLabelStyle`, `tooltipStyle`, `THEME` de `base.js`.
- Produces: `verdicts(summary, chance=None)` con `q3`; `p_text(p)`; `null_text(chance)`; `null_summary_text(chance)`;
  `sigmas(chance)`; `chanceOption(chance, opts)` y `renderChance(el, chance, opts)`; contexto `chance`, `null_text`,
  `null_summary`, `sigmas`, `initial.chance`.

- [ ] **Step 1: Write the failing tests**

```python
def test_herd_p_text_and_null_texts():
    assert p_text(0.00005) == 'p < 0,0001' and p_text(0.0004) == 'p < 0,001' and p_text(0.047) == 'p = 0,047' and p_text(None) is None
    chance = fixture_data('rebano-chance')['chance']
    assert null_text(chance) == (
        'Ninguna de las 20.000 barajas bajó del −12 %; lo observado es un −26 %, de 1,58 a 1,17 veces el error de las '
        'muestras: p < 0,0001 a dos colas. Las separaciones de las tres últimas semanas son un 26 % menores que en el '
        'resto del ciclo (entre el 4 y el 38 %). En 25 de las 34 elecciones las mismas casas se acercan.')
    some = dict(chance, count_two=240, p_two=0.012, ci=None)
    assert null_text(some).startswith('240 de las 20.000 barajas llegaron tan lejos como lo observado (−26 %), que va de 1,58 a '
                                      '1,17 veces el error de las muestras: p = 0,012 a dos colas. Las separaciones')
    assert '(entre' not in null_text(some)
    assert null_summary_text(chance) == 'Barajando 20.000 veces, ninguna baraja llega a lo observado: p < 0,0001.'
    assert null_summary_text(some) == '240 de 20.000 barajas llegan a lo observado: p = 0,012.'
    assert sigmas(chance) == 8 and sigmas(dict(chance, null=dict(chance['null'], sd=0))) is None
    assert null_text(None) is None and null_summary_text({}) is None
    assert verdicts(fixture_data('rebano-summary'), chance)['q3'] == {'label': 'NO', 'tone': 'q1'}
    assert verdicts({}, dict(chance, p_two=0.03))['q3'] == {'label': 'DIFÍCILMENTE', 'tone': 'q2'}
    assert verdicts({}, dict(chance, p_two=0.3))['q3'] == {'label': 'SÍ, PODRÍA', 'tone': 'q3'}
    assert verdicts({}, None)['q3'] == {'label': 'SIN DATOS', 'tone': 'q2'}
```

(`null_text` lee `count_two` y `permutations` del bloque; en el fixture `count_two` es 0.)

En `tests/test_js_charts.py`:

```python
def test_chance_option_draws_the_null_histogram_and_marks_the_observed():
    chance = {...fixture...}
    option = run_js('ch.chanceOption(%s)' % json.dumps(chance))
    bars = option['series'][0]
    assert len(bars['data']) == 28 and option['xAxis']['min'] <= -26.3 and option['xAxis']['max'] >= 16
    line = option['series'][0]['markLine']['data'][0] if 'markLine' in option['series'][0] else option['series'][1]['markLine']['data'][0]
    assert line['xAxis'] == pytest.approx(-26.3) and 'observado −26 %' in json.dumps(option, ensure_ascii=False)
    assert option['xAxis']['name'] == 'cambio de la separación entre el resto del ciclo y las tres últimas semanas'
```

En `tests/test_pages.py`: la página con el fixture tiene `'<section class="card qcard shot q1" id="p3">'`, tres
veredictos `['SÍ, EN CONJUNTO', 'SÍ, EN LAS TRES ÚLTIMAS SEMANAS', 'NO']`, `text.count('¿Podría ser azar?') == 3`
(resumen, tarjeta, y… ajustar al conteo real: resumen + h2 = 2 si el título no se repite), `'id="chance-chart"' in
page`, el `lead`, el `csub` («20.000 barajas de las mismas 475 parejas de casas en 34 elecciones»), `null_text`, la
nota («Qué se baraja y qué p damos», «p = 1/20.001», «a unas 8 desviaciones típicas»), la frase del `tech`
(«La prueba de la tarjeta 3 baraja 20.000 veces»), `'pregunta-3' in slugs`, `initial-data` con `chance`; y un test
sin la parte (`publish_study(..., parts={'chance': {'chance': None}})`): sin `id="p3"`, sin «Podría ser azar», sin
«prueba de la tarjeta 3», 200 y `assert_public_wording`. Actualizar los contadores existentes que cambien (capturas,
`answer`s).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_pages.py -k herd tests/test_js_charts.py -k chance`
Expected: FAIL.

- [ ] **Step 3: Implement the view models, the context, the chart and the template**

`herd.py`: `verdicts(summary, chance=None)` (los dos primeros como hoy); `p_text`; `null_text` (frases de la spec D4;
«menores»/«mayores» por el signo; el intervalo con la lógica de `ci_text` sobre `{'ci': chance['ci']}`; números con
`fmt_int`/`fmt_num(…, 2)`/`percent`/`_whole`); `null_summary_text`; `sigmas`. `herd_context`: `chance = _get(parts.get
('chance'), 'chance')`, `verdicts=herd.verdicts(summary, chance)`, `null_text`, `null_summary`, `sigmas`,
`initial={'weeks': …, 'chance': chance, 'state': …}`. `chance.js`: `chanceOption(chance, {color = THEME.mutedColor,
observed = THEME.accentColor} = {})` y `renderChance`; el eje x en % con `min`/`max` que cubren cajas y observado
(margen de 2 cajas), barras que ocupan su caja (constructor: `custom`/`renderItem` o `bar` sobre eje de valores),
`markLine` en `observed·100` con `label` «observado {percent}», tooltip «de {lo} a {hi} %: {count} barajas», eje y
«barajas». `analisis-rebano.js`: `drawChance(readInitial().chance)` si existe `#chance-chart`. Plantilla: la tercera
`answer` en el resumen; la tarjeta 3 entre la 2 y «en directo», con `shot('pregunta-3')`, cabecera, `lead`, `ctitle`,
`csub`, gráfico, `tnote`, caja `sum` (`<b>` en la primera frase: partir `null_text` en la primera «. » o pasar dos
campos: constructor), nota `details.how`; la frase del `tech`. CSS: `#chance-chart { height: 260px; }` (220 en
móvil), como `#weeks-chart`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_pages.py -k "herd or rebano or analisis" tests/test_js_charts.py && python -m pytest -q`
Expected: PASS; batería completa verde.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/herd.py mtpy/lib/pages/analisis.py web/templates/analisis_rebano.html web/dist/js/charts/chance.js web/dist/js/pages/analisis-rebano.js web/dist/css/site.css tests/test_pages.py tests/test_js_charts.py
git commit -m "Rebaño: tarjeta 3 «¿Podría ser azar?» con el histograma de la distribución nula, la p a dos colas y el intervalo"
```

---

### Aceptación (controlador, tras la Tarea 4, sólo lectura con el `.env` del checkout principal)

Calcular el estudio con datos reales: `chance` con 34 elecciones, 25 bajan, 475 parejas, cambio ≈ −0,263, `p_two`
0,00005, `null.min` ≈ −0,12, `null.max` ≈ +0,15, intervalo ≈ [−0,38, −0,04]; tiempo total < 50 s. La página con las
partes reales en Chrome headless (tarjeta 3 en escritorio y móvil, la nota abierta, la imagen de la tarjeta).
