# Refuerzo estadístico del análisis del rebaño: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir al estudio `rebano` una prueba de permutación conjunta (todas las parejas de casas de todas las
elecciones, barajando por parejas de sondeos), un intervalo bootstrap de la bajada mediana y una rejilla de robustez,
y escribirlos en la página con una frase por pregunta y una nota desplegable para expertos.

**Architecture:** `stats.py` gana los bloques (una pareja de sondeos = un bloque), la permutación por bloques
(compartida por la prueba por elección y la conjunta), `pooled`/`pooled_p` y `bootstrap_median`; `study.py` los
encadena y añade `pooled`, `ci` y `robustness` al `summary`; la parte `summary` crece (superconjunto) con un CSV nuevo
`robustness`; `pages/herd.py` redacta las frases y la tabla, la plantilla las coloca.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (ya en uso), Jinja2; pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-web-analisis-rebano-refuerzo-design.md` (dev b69eea8). Las anteriores:
`2026-11-06-web-analisis-rebano-v2-design.md` y `2026-11-07-web-analisis-rebano-embudo-design.md`.

## Global Constraints

- La medida no cambia: `comparisons`, `compare` (peso fijo por pareja `w = 1/(1/Σv_a + 1/Σv_b)`), `sign_test`,
  `count`, `by_buckets`, los tramos y la regla de los veredictos (`herd._significant` lee sólo la prueba del signo).
- La unidad de permutación es la **pareja de sondeos** (`poll_i`, `poll_j`): sus filas (una por partido) van juntas,
  en `permutation_p` y en `pooled_p`. El barajado mantiene cuántos bloques tiene cada periodo dentro de cada pareja de
  casas; `p` = proporción de permutaciones con `perm_b/perm_a ≤ observado·(1 + PERMUTATION_TOLERANCE)`.
- Parámetros exactos: `"resamples": 2000`; `"robustness": {"pair_days": [3, 5, 7, 10], "min_rows": [3, 6]}`; la celda
  por defecto es `(params['pair_days'], params['min_rows'])`, hoy (7, 3). `permutations` 2000 y `seed` 0 se reutilizan.
- Redondeos como en el estudio: `r_a`/`r_b` 2 decimales, `change`/`median_change`/`lo`/`hi` 3, `p` con `_sig3`.
- La parte `summary` nueva es un superconjunto de la actual; la página condiciona cada texto nuevo a que su dato exista
  (sin `pooled`, `ci` o `robustness` la página queda como hoy).
- Textos públicos: «sólo» con tilde, «casa»/«encuestadora», nunca «empresa»; nada de «se copian», «vecino», «permite el
  azar», ni intenciones; `assert_public_wording` pasa. El azar se escribe con `herd.chance_text` / `herd._odds`.
- `sign_flip_p` NO entra (decisión registrada en la spec).
- Tests: `python -m pytest -q` verde en cada tarea; ningún test escribe fuera de `tmp_path`.
- Código en inglés neutro, docstrings numpy, PEP8 (120 columnas como el resto del paquete).

## Review Focus

1. Una pareja de casas con un solo bloque en un periodo: la permutación tiene pocos arreglos y `p` ≥ 1/C(n, k); el
   resultado debe ser una `p` válida (nunca error, nunca NaN). → Tarea 1 (`test_permutation_p_one_block_per_period`).
2. Celda de la rejilla sin elecciones (p. ej. mínimo 6 con 3 días): la fila existe con `n` 0, `k` 0 y `None` en el
   resto, y la página escribe «–». → Tareas 2 y 4.
3. `p` conjunta exactamente 0 (ninguna permutación iguala la bajada): `_sig3(0.0)` es `0.0`, `chance_text(0.0)` es
   «menos de 1 entre 1.000» y el CSV lo escribe como `0.0`. → Tareas 2, 3 y 4.
4. Intervalo cuyos extremos redondean al mismo entero, o con los dos extremos positivos: `ci_text` no escribe «entre el
   15 y el 15 %». → Tarea 4.
5. Cambio conjunto que redondea a 0 %: la frase dice «es igual que», no «un 0 % menor». → Tarea 4.

---

### Task 1: Bloques, permutación por bloques, prueba conjunta y bootstrap (`stats.py`, parámetros)

**Files:**
- Modify: `mtpy/lib/studies/herd/stats.py` (tras `_weighted_sums`, antes de `compare`; `permutation_p`)
- Modify: `data/rebano/params.json`, `data/README.md:32`
- Test: `tests/test_herd_stats.py`

**Interfaces:**
- Consumes: `_shared_pairs(rows, a, b, min_rows)`, `_weighted_sums(d2_a, v_a, d2_b, v_b)`, `PERMUTATION_TOLERANCE`,
  `_sig3` (existentes).
- Produces:
  - `_blocks(group: pd.DataFrame, a: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `(d2, v, is_a)` por pareja de
    sondeos (`groupby(['poll_i', 'poll_j'], sort=False)`: `d2` y `v` sumados, `is_a` = su `period == a`; todas las
    filas de una pareja de sondeos comparten periodo).
  - `_permuted_terms(shared: pd.DataFrame, a: str, permutations: int, generator: np.random.Generator) -> tuple`:
    `(perm_a, perm_b, obs_a, obs_b)`; por pareja de casas (`groupby('pair')`), bloques de `_blocks`, barajado como el
    actual de `permutation_p` (argsort de uniformes por permutación; los `int(is_a.sum())` primeros toman la etiqueta
    `a`), `_weighted_sums` sobre los bloques; `perm_*` arrays de `permutations`, `obs_*` floats.
  - `permutation_p(rows, a, b, min_rows, permutations, seed) -> Optional[float]`: misma firma; usa `_permuted_terms`
    con `np.random.default_rng(seed)`; misma `p` final.
  - `pooled(rows_by_election: list[pd.DataFrame], a: str, b: str, min_rows: int) -> Optional[dict]`:
    `{'elections', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b', 'change', 'falls'}` sin redondear (ver spec D1).
  - `pooled_p(rows_by_election: list[pd.DataFrame], a: str, b: str, min_rows: int, permutations: int, seed: int)
    -> Optional[float]`.
  - `bootstrap_median(changes: list[float], resamples: int, seed: int) -> Optional[dict]`: `{'lo', 'hi'}` sin
    redondear (percentiles 2,5 y 97,5 con `np.percentile`, remuestreo con `generator.integers`); `None` con menos de 3.

- [ ] **Step 1: Write the failing tests**

En `tests/test_herd_stats.py`, importar también `_blocks, bootstrap_median, pooled, pooled_p`. Las filas construidas
de los tests de permutación ganan `poll_i`/`poll_j` (en `two_pairs()` y en los frames de los tests pequeños: una
pareja de sondeos distinta por fila, `poll_i = 'p%d' % i`, `poll_j = 'q%d' % i`, para que cada fila siga siendo su
propio bloque y los valores exactos de hoy no cambien).

```python
def test_blocks_sum_the_rows_of_each_pair_of_polls():
    group = pd.DataFrame({'pair': ['A|B'] * 4, 'poll_i': ['a1', 'a1', 'a2', 'a2'], 'poll_j': ['b1', 'b1', 'b2', 'b2'],
                          'period': ['far', 'far', 'final', 'final'], 'd2': [1.0, 3.0, 0.5, 0.5], 'v': [2.0, 2.0, 1.0, 1.0]})
    d2, v, is_a = _blocks(group, 'far')
    assert d2.tolist() == [4.0, 1.0] and v.tolist() == [4.0, 2.0] and is_a.tolist() == [True, False]


def test_permutation_p_keeps_the_series_of_a_pair_of_polls_together():
    # 2 pairs of polls far (d2 4 per series) and 2 final (d2 1), two series each: C(4,2) = 6 arrangements of the
    # blocks, the observed fall is the extreme one: p = 1/6 (by rows it would be 1/C(8,4) = 1/70)
    rows = pd.DataFrame({'pair': ['A|B'] * 8, 'poll_i': ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4'],
                         'poll_j': ['b1'] * 2 + ['b2'] * 2 + ['b3'] * 2 + ['b4'] * 2,
                         'period': ['far'] * 4 + ['final'] * 4, 'd2': [4.0] * 4 + [1.0] * 4, 'v': [2.0] * 8})
    p = permutation_p(rows, 'far', 'final', 3, 3000, 0)
    assert abs(p - 1 / 6) < 0.025


def test_permutation_p_one_block_per_period():
    rows = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['a1'] * 3 + ['a2'] * 3, 'poll_j': ['b1'] * 3 + ['b2'] * 3,
                         'period': ['far'] * 3 + ['final'] * 3, 'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    p = permutation_p(rows, 'far', 'final', 3, 2000, 0)
    assert abs(p - 0.5) < 0.03                   # two arrangements: the observed one and the swap


def two_elections():
    """Two elections with one pair of houses each; by hand: R_a = √2, R_b² = 22,5/33 (see the plan)."""
    e1 = pd.DataFrame({'pair': ['A|B'] * 6, 'poll_i': ['a%d' % i for i in range(6)], 'poll_j': ['b%d' % i for i in range(6)],
                       'period': ['far'] * 3 + ['final'] * 3, 'd2': [4.0] * 3 + [1.0] * 3, 'v': [2.0] * 6})
    e2 = pd.DataFrame({'pair': ['C|D'] * 7, 'poll_i': ['c%d' % i for i in range(7)], 'poll_j': ['d%d' % i for i in range(7)],
                       'period': ['far'] * 4 + ['final'] * 3, 'd2': [2.0] * 4 + [1.0] * 3, 'v': [1.0] * 7})
    return [e1, e2]


def test_pooled_weighs_the_pairs_of_every_election():
    e1, e2 = two_elections()
    one = pooled([e1], 'far', 'final', 3)
    same = compare(e1, 'far', 'final', 3)
    assert (one['elections'], one['pairs'], one['rows_a'], one['rows_b']) == (1, 1, 3, 3)
    assert one['r_a'] == pytest.approx(same['r_a']) and one['r_b'] == pytest.approx(same['r_b'])
    both = pooled([e1, e2], 'far', 'final', 3)
    # w1 = 3 (r²: 2 → 0,5), w2 = 12/7 (r²: 2 → 1): R_a² = (6 + 24/7)/(33/7) = 2, R_b² = (1,5 + 12/7)/(33/7) = 22,5/33
    assert (both['elections'], both['pairs'], both['rows_a'], both['rows_b']) == (2, 2, 7, 6)
    assert both['r_a'] == pytest.approx(2 ** 0.5) and both['r_b'] == pytest.approx((22.5 / 33) ** 0.5)
    assert both['change'] == pytest.approx(-0.4161, abs=1e-4) and both['falls'] is True
    assert pooled([e1.iloc[:0]], 'far', 'final', 3) is None and pooled([], 'far', 'final', 3) is None


def test_pooled_p_matches_the_exact_enumeration():
    # two elections with 4 blocks each (2 + 2): 6 · 6 = 36 arrangements enumerated with itertools over the
    # statistic Σ w·r²_b / Σ w·r²_a of the blocks (compute it in the test as `statistic` does for rows)
    ...
    assert abs(pooled_p(rows, 'far', 'final', 3, 4000, 0) - exact) < 0.03


def test_pooled_p_small_with_a_strong_fall_and_none_without_pairs():
    # each election: 4 far blocks with d2 10 and 4 final blocks with d2 1 (two series each): the observed is the
    # extreme of 70 · 70 arrangements
    ...
    assert pooled_p(rows, 'far', 'final', 3, 2000, 0) < 0.01
    assert pooled_p([], 'far', 'final', 3, 200, 0) is None


def test_bootstrap_median_interval_contains_the_median_and_is_deterministic():
    changes = [-0.5, -0.1, -0.2, 0.4]
    ci = bootstrap_median(changes, 2000, 0)
    assert ci['lo'] <= -0.15 <= ci['hi'] and min(changes) <= ci['lo'] and ci['hi'] <= max(changes)
    assert bootstrap_median(changes, 2000, 0) == ci and bootstrap_median([-0.1, 0.2], 2000, 0) is None
```

Los tests existentes cambian así: `test_permutation_p_shuffles_within_each_pair_with_its_weights` y
`test_permutation_p_small_exact_case` conservan sus valores (una fila = un bloque) tras añadir `poll_i`/`poll_j`;
`test_compare_without_differences_in_a_is_none` idem.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_herd_stats.py`
Expected: FAIL (`ImportError` de `_blocks`, `pooled`, `pooled_p`, `bootstrap_median`).

- [ ] **Step 3: Implement `_blocks`, `_permuted_terms`, `permutation_p` por bloques, `pooled`, `pooled_p`, `bootstrap_median`**

`permutation_p` conserva la firma y el docstring actualizado («dentro de cada pareja de casas se barajan las
etiquetas de sus parejas de sondeos, con sus series juntas»). `pooled` reutiliza la suma por pareja de `compare`
(`groupby(['pair', 'period'])[['d2', 'v']].sum().unstack('period')`) elección a elección y acumula `Σ w`, `Σ w·r²_a`,
`Σ w·r²_b`, filas y parejas. `pooled_p` crea un generador y suma los `_permuted_terms` de cada elección con parejas
compartidas. Módulo: añadir la frase de los bloques al docstring del módulo.

- [ ] **Step 4: Parámetros**

`data/rebano/params.json`: añadir `"resamples": 2000` y `"robustness": {"pair_days": [3, 5, 7, 10], "min_rows": [3,
6]}` (tras `seed`, antes de `weeks`). `data/README.md:32`: añadir a la descripción «remuestreos del intervalo y rejilla
de robustez (días entre sondeos y mínimos de comparaciones)».

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q tests/test_herd_stats.py tests/test_herd_study.py tests/test_publish_herd.py`
Expected: `test_herd_stats.py` PASS; en `test_herd_study.py` fallan sólo las aserciones de `p` (0,214 → 1/6) que la
Tarea 2 actualiza; `test_publish_herd.py` PASS.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/studies/herd/stats.py data/rebano/params.json data/README.md tests/test_herd_stats.py
git commit -m "Rebaño: permutación por parejas de sondeos (bloques), prueba conjunta y bootstrap de la mediana"
```

---

### Task 2: El estudio: `pooled`, `ci` y la rejilla de robustez (`study.py`)

**Files:**
- Modify: `mtpy/lib/studies/herd/study.py` (`compute_herd`, `HerdStudy` docstring)
- Test: `tests/test_herd_study.py`

**Interfaces:**
- Consumes: `stats.pooled`, `stats.pooled_p`, `stats.bootstrap_median`, `stats.comparisons`, `stats.compare`,
  `stats.count`, `_sig3` (Tarea 1).
- Produces: `summary['comparisons'][clave]` gana `pooled` (`{elections, pairs, rows_a, rows_b, r_a, r_b, change, p}`
  o `None`) y `ci` (`{lo, hi}` o `None`); `summary['robustness'] = {'pair_days': list, 'min_rows': list, 'rows':
  list}` con una fila por celda y comparación `{pair_days, min_rows, default, comparison, n, k, median_change, change,
  p}` (orden: `pair_days` externo, `min_rows` interno, comparaciones en el orden de `COMPARISONS`); una celda sin
  elecciones lleva `n` 0, `k` 0 y `None` en `median_change`, `change`, `p`. `_results(election, params, rows)` deja de
  recalcular las comparaciones (recibe `rows`). Función nueva `_robustness(elections, params) -> dict`.

- [ ] **Step 1: Write the failing tests**

Cifras del paquete sintético (`write_herd_bundle`; enumeración exacta del 2026-11-07): Generales 2019 tiene una pareja
de casas con 2 parejas de sondeos en cada tramo (4 bloques, 6 arreglos) y la bajada observada es la extrema en las tres
comparaciones: `p` = 1/6; Generales 2023 lejos → final tiene 5 bloques (2 + 3) y la subida observada es la máxima:
`p` = 1,0. Conjunta lejos → final: 2 estratos, filas 8/10, `r_a` 0,98, `r_b` 0,81, `change` −0,173, `p` exacta 22/60.

```python
def test_compute_herd_rows(fresh_app, tmp_path):
    ...
    assert abs(g19['p'] - 1 / 6) < 0.03            # exact 1/6: 4 blocks (2 pairs of polls per period)
    ...
    assert abs(g23['p'] - 1.0) < 1e-9              # the observed rise is the largest of the 10 arrangements


def test_compute_herd_summary(fresh_app, tmp_path):
    s = compute(tmp_path).summary
    ff = s['comparisons']['far_final']
    assert {k: ff[k] for k in ('n', 'k', 'sign_p', 'clear', 'median_change', 'since_year')} == {
        'n': 2, 'k': 1, 'sign_p': 0.75, 'clear': 0, 'median_change': 0.62, 'since_year': '2019'}
    pooled = ff['pooled']
    assert {k: pooled[k] for k in ('elections', 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b', 'change')} == {
        'elections': 2, 'pairs': 2, 'rows_a': 8, 'rows_b': 10, 'r_a': 0.98, 'r_b': 0.81, 'change': -0.173}
    assert abs(pooled['p'] - 22 / 60) < 0.03 and ff['ci'] is None       # 2 elections: no interval
    fm = s['comparisons']['far_mid']
    assert fm['pooled']['change'] == -0.248 and abs(fm['pooled']['p'] - 1 / 6) < 0.03 and fm['ci'] is None
    assert s['comparisons']['mid_final']['pooled']['change'] == -0.667


def test_compute_herd_robustness(fresh_app, tmp_path):
    r = compute(tmp_path).summary['robustness']
    assert r['pair_days'] == [3, 5, 7, 10] and r['min_rows'] == [3, 6] and len(r['rows']) == 24
    assert [(row['pair_days'], row['min_rows'], row['comparison']) for row in r['rows'][:4]] == [
        (3, 3, 'far_final'), (3, 3, 'far_mid'), (3, 3, 'mid_final'), (3, 6, 'far_final')]
    default = {row['comparison']: row for row in r['rows'] if row['default']}
    s = compute(tmp_path).summary['comparisons']
    assert set(default) == set(COMPARISONS) and all(row['pair_days'] == 7 and row['min_rows'] == 3 for row in default.values())
    for key, row in default.items():
        assert (row['n'], row['k'], row['median_change'], row['change']) == (
            s[key]['n'], s[key]['k'], s[key]['median_change'], s[key]['pooled']['change'])
    d3 = next(row for row in r['rows'] if (row['pair_days'], row['min_rows'], row['comparison']) == (3, 3, 'far_final'))
    assert (d3['n'], d3['k'], d3['median_change'], d3['change']) == (2, 1, 0.618, -0.234)
    empty = next(row for row in r['rows'] if (row['pair_days'], row['min_rows'], row['comparison']) == (3, 6, 'far_final'))
    assert empty == {'pair_days': 3, 'min_rows': 6, 'default': False, 'comparison': 'far_final', 'n': 0, 'k': 0,
                     'median_change': None, 'change': None, 'p': None}
    ten = next(row for row in r['rows'] if (row['pair_days'], row['min_rows'], row['comparison']) == (10, 6, 'mid_final'))
    assert (ten['n'], ten['k'], ten['change']) == (1, 1, -0.667)
```

(`compute` puede llamarse una vez y guardarse; el paquete sintético tarda poco.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_herd_study.py`
Expected: FAIL (`KeyError: 'pooled'`, `'robustness'`; las `p` antiguas).

- [ ] **Step 3: Implement**

En `compute_herd`: calcular `rows = comparisons(...)` una vez por elección (ya se hace para `_weeks`; pasar también a
`_results`), guardar `rows_by_election` de las pasadas; tras el bucle, por clave: `pooled` = `stats.pooled(...)`
redondeado + `p` = `_sig3(pooled_p(..., params['permutations'], params['seed']))`; `ci` = `bootstrap_median([r['change']
sin redondear de raw[key]], params['resamples'], params['seed'])` redondeado a 3. `_robustness(elections, params)`:
por `d` en `params['robustness']['pair_days']` recalcular las comparaciones de cada elección con `params |
{'pair_days': d}` (una vez por `d`), por `m` en `min_rows` y por clave: `compare` por elección → `count` sin `p` por
elección (pasar `p` 1,0 a `count` para que `clear` no cuente, o un `count` que ignore `clear`: la fila no publica
`clear`), `pooled` y `pooled_p`. Docstring de `HerdStudy.summary` actualizado.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_herd_study.py tests/test_herd_stats.py tests/test_publish_herd.py`
Expected: PASS salvo `test_publish_herd.py::test_export_herd_parts_validate` si `validate` rechaza claves nuevas (no
lo hace: el esquema mira las claves de primer nivel, y `robustness` se declara en la Tarea 3; si falla, dejarlo para la
Tarea 3 y anotarlo).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/studies/herd/study.py tests/test_herd_study.py
git commit -m "Rebaño: prueba conjunta, intervalo de la mediana y rejilla de robustez en el resumen del estudio"
```

---

### Task 3: La parte publicada: esquema, CSV `robustness`, contrato (`bundle.py`, `publish/herd.py`, `analisis.py`)

**Files:**
- Modify: `mtpy/lib/bundle.py:89-110, 197` (`STUDY_CSV`, `STUDY_CSV_OF`, `SCHEMAS['rebano-summary']`)
- Modify: `mtpy/lib/publish/herd.py` (`ROBUSTNESS_COLUMNS`, `export_herd_summary`)
- Modify: `mtpy/lib/pages/analisis.py:64` (`CSV_LABELS['rebano']['robustness']`)
- Modify: `docs/web/contrato.md:555, 890`
- Modify: `tests/fixtures/bundle/rebano-summary.json`
- Test: `tests/test_publish_herd.py`, `tests/test_bundle_unit.py` (si enumera los CSV de los estudios)

**Interfaces:**
- Consumes: `summary['robustness']['rows']` (Tarea 2).
- Produces: `STUDY_CSV['rebano'] = ('elections', 'weeks', 'robustness')`; `STUDY_CSV_OF['rebano']['robustness'] =
  ('summary',)`; `SCHEMAS['rebano-summary']` con `'robustness': dict`; `ROBUSTNESS_COLUMNS = ['pair_days', 'min_rows',
  'default', 'comparison', 'n', 'k', 'median_change', 'change', 'p']`; `export_herd_summary(study) -> (summary,
  {'robustness': frame})` con `n`, `k` como `int64` (sin nulos: 0 en las celdas vacías) y `median_change`, `change`,
  `p` como float con NaN donde `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_herd_summary_validates_and_flattens_the_robustness_grid(study):
    summary, frames = publish.export_herd_summary(study)
    assert valid('rebano-summary', summary)
    frame = frames['robustness']
    assert list(frame.columns) == publish.ROBUSTNESS_COLUMNS and len(frame) == 24
    assert str(frame['n'].dtype) == 'int64' and frame['default'].sum() == 3
    assert frame.loc[(frame.pair_days == 3) & (frame.min_rows == 6), 'change'].isna().all()


def test_publish_study_rebano_writes_the_robustness_csv(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    path = f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'robustness')}"
    assert fs.exists(path)
    text = <leer el fichero como hace test_export_herd_weeks_csv_counts_are_integers>
    assert text.splitlines()[0] == ','.join(publish.ROBUSTNESS_COLUMNS) and ',3.0,' not in text   # counts as integers
```

Y en `test_export_herd_parts_validate`: `frames` de `export_herd_summary` ya no es `{}` → `set(frames) ==
{'robustness'}`. El fixture `rebano-summary.json` gana, en cada comparación, `pooled` y `ci`, y `robustness` con las 24
filas; cifras (del spike del 2026-11-07, `p` por bloques):

| clave | `pooled` | `ci` |
|---|---|---|
| far_final | `{elections 23, pairs 309, rows_a 2501, rows_b 1750, r_a 1.42, r_b 1.03, change -0.274, p 0.0}` | `{lo -0.348, hi -0.032}` |
| far_mid | `{elections 17, pairs 185, rows_a 1701, rows_b 1284, r_a 1.33, r_b 1.25, change -0.062, p 0.0575}` | `{lo -0.281, hi 0.186}` |
| mid_final | `{elections 29, pairs 312, rows_a 2027, rows_b 1820, r_a 1.36, r_b 1.04, change -0.232, p 0.0}` | `{lo -0.274, hi -0.022}` |

`robustness`: `pair_days [3, 5, 7, 10]`, `min_rows [3, 6]`, filas `(pair_days, min_rows, comparison): n, k,
median_change, change, p` (la fila por defecto de cada comparación repite `n`, `k`, `median_change` del fixture y
`change` de `pooled`):

```
(3,3) far_final 18 11 -0.057 -0.226 0.0 | far_mid 13 6 0.030 -0.033 0.289 | mid_final 23 14 -0.045 -0.217 0.0
(3,6) far_final 3 2 -0.020 -0.057 0.316 | far_mid 2 0 0.203 0.189 0.994 | mid_final 3 3 -0.198 -0.324 0.0
(5,3) far_final 20 14 -0.157 -0.249 0.0 | far_mid 16 10 -0.221 -0.083 0.0325 | mid_final 28 18 -0.163 -0.210 0.0
(5,6) far_final 4 4 -0.302 -0.433 0.0 | far_mid 4 2 -0.099 -0.154 0.015 | mid_final 6 3 -0.041 -0.272 0.0
(7,3) far_final 23 19 -0.15 -0.274 0.0 (default) | far_mid 17 9 -0.1 -0.062 0.0575 (default) | mid_final 29 23 -0.15 -0.232 0.0 (default)
(7,6) far_final 5 4 -0.465 -0.368 0.0 | far_mid 6 3 -0.064 -0.118 0.013 | mid_final 9 7 -0.113 -0.310 0.0
(10,3) far_final 27 22 -0.214 -0.263 0.0 | far_mid 17 8 0.015 -0.072 0.0125 | mid_final 30 22 -0.109 -0.211 0.0
(10,6) far_final 8 6 -0.259 -0.325 0.0 | far_mid 6 3 -0.025 -0.119 0.0005 | mid_final 11 8 -0.103 -0.260 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_publish_herd.py`
Expected: FAIL (`ROBUSTNESS_COLUMNS` no existe; `frames == {}`).

- [ ] **Step 3: Implement**

`bundle.py`: las tres declaraciones. `publish/herd.py`: `ROBUSTNESS_COLUMNS` y el frame en `export_herd_summary`
(`pd.DataFrame(_clean(rows), columns=ROBUSTNESS_COLUMNS)`, `frame[['n', 'k']] = frame[['n', 'k']].astype('int64')`).
`analisis.py`: `CSV_LABELS['rebano']['robustness'] = 'Robustez: recuentos y cambio conjunto con otros días entre
sondeos y otro mínimo de comparaciones'`. `contrato.md`: la fila de `summary` describe `comparisons` (cada una con
`n, k, sign_p, clear, median_change, since_year, pooled` (dict o `null`: `elections, pairs, rows_a, rows_b, r_a, r_b,
change, p`) y `ci` (`{lo, hi}` o `null`)) y `robustness` (`pair_days`, `min_rows`, `rows`); la línea 890 añade
`robustness` (`summary`) a los CSV de `rebano`; donde el contrato describa la `p` de `elections`, decir «permutación
de las parejas de sondeos dentro de cada pareja de casas». Comprobar `grep -n "rebano" tests/test_bundle_unit.py`
por si enumera los CSV (actualizar la tupla).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest -q tests/test_publish_herd.py tests/test_bundle_unit.py tests/test_pages.py -k "herd or rebano or bundle"`
Expected: PASS (`test_herd_page_structure_and_wording` cuenta 2 píldoras CSV y ahora hay 3: actualizar ese `== 2` a
`== 3` en esta tarea con un comentario `# elections + weeks + robustness CSVs`).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py mtpy/lib/publish/herd.py mtpy/lib/pages/analisis.py docs/web/contrato.md tests/fixtures/bundle/rebano-summary.json tests/test_publish_herd.py tests/test_pages.py tests/test_bundle_unit.py
git commit -m "Rebaño: parte summary con pooled, ci y robustness, CSV robustness y contrato"
```

---

### Task 4: La página: frases con intervalo y prueba conjunta, nota de robustez, método (`herd.py`, plantilla)

**Files:**
- Modify: `mtpy/lib/pages/herd.py` (tras `chance_text`/`_odds`: `ci_text`, `pooled_text`, `pooled_both_text`,
  `robustness_text`, `robustness_rows`)
- Modify: `mtpy/lib/pages/analisis.py:674-729` (`herd_context`)
- Modify: `web/templates/analisis_rebano.html` (cajas `sum` de las tarjetas 1 y 2, «Cómo lo medimos», `tech`)
- Test: `tests/test_pages.py` (tests `herd_*`)

**Interfaces:**
- Consumes: `summary['comparisons'][k]['pooled' | 'ci']`, `summary['robustness']` (Tarea 2, fixture de la Tarea 3);
  `chance_text`, `_odds`, `percent`, `fmt_int`, `DASH`, `_get`, `_is_num`.
- Produces:
  - `ci_text(comparison: Optional[dict]) -> Optional[str]`: `lo`/`hi` → enteros `round(abs(x) * 100)`; los dos
    negativos → «entre el {menor} y el {mayor} %»; cruza el cero → «entre una subida del {hi} % y una bajada del
    {|lo|} %»; los dos positivos → «entre el {menor} y el {mayor} % de subida»; enteros iguales → «en torno al {x} %»
    (con «de subida» si positivos); `None` sin `ci`.
  - `pooled_text(comparison, later: str, earlier: str) -> Optional[str]`: «Con todas las parejas de todas las
    elecciones juntas, la separación {later} es un {x} % {menor|mayor} que {earlier}: {_odds(p)}.»; `x` =
    `round(abs(change) * 100)`; `x` 0 → «es igual que {earlier}: {_odds(p)}.»; `None` sin `pooled`.
  - `pooled_both_text(mid_final, far_mid) -> Optional[str]`: «Con todas las parejas juntas: un {x} % {menos|más} en las
    tres últimas semanas que entre 3 y 8 semanas antes ({_odds}), y un {y} % {menos|más} hasta tres semanas antes que
    lejos ({_odds}).» con las cláusulas que tengan `pooled` (una sola → sin «, y»); cambio 0 → «lo mismo en las tres
    últimas semanas que…»; `None` sin ninguna.
  - `robustness_text(summary) -> Optional[str]`: sobre las filas con `min_rows == params min_rows por defecto` (la
    fila `default` dice cuál) y las comparaciones `far_final` y `mid_final`: «Con {3, 5, 7 o 10} días entre los dos
    sondeos, la bajada conjunta al final queda entre el {min} y el {max} %, y en las tres últimas semanas entre el
    {min} y el {max} %; el recuento de elecciones que se acercan al final va de {k} de {n} con {d} días a {k} de {n}
    con {d} (la fracción k/n menor y la mayor; con empate, la de menos días). Con un mínimo de {6} comparaciones por pareja de casas (dos parejas de sondeos) quedan pocas elecciones:
    la tabla las enseña.» (`min`/`max` de `round(abs(change) * 100)` sobre filas con `change` no nulo; el recuento
    por la fracción `k/n` mínima y máxima; `None` sin filas con `change`).
  - `robustness_rows(summary) -> list[dict]`: `{'pair_days', 'min_rows', 'default', 'far_final': {'count', 'change',
    'chance'}, 'mid_final': {...}}`, `count` «k de n» (o `DASH` con `n` 0), `change` = `percent(change)` (o `DASH`),
    `chance` = `chance_text(p)` (o `DASH`); la fila por defecto primero y sin repetir en la rejilla; `[]` sin filas.
  - `herd_context` añade `ci={'far_final': ..., 'mid_final': ...}`, `pooled={'far_final': pooled_text(ff, 'al
    final', 'lejos de la votación'), 'both': pooled_both_text(mf, fm)}`, `robustness_text`, `robustness_rows`.

- [ ] **Step 1: Write the failing tests**

```python
def test_herd_ci_and_pooled_texts():
    assert ci_text({'ci': {'lo': -0.348, 'hi': -0.032}}) == 'entre el 3 y el 35 %'
    assert ci_text({'ci': {'lo': -0.281, 'hi': 0.186}}) == 'entre una subida del 19 % y una bajada del 28 %'
    assert ci_text({'ci': {'lo': 0.05, 'hi': 0.12}}) == 'entre el 5 y el 12 % de subida'
    assert ci_text({'ci': {'lo': -0.152, 'hi': -0.148}}) == 'en torno al 15 %' and ci_text({}) is None
    ff = {'pooled': {'change': -0.274, 'p': 0.0}}
    assert pooled_text(ff, 'al final', 'lejos de la votación') == (
        'Con todas las parejas de todas las elecciones juntas, la separación al final es un 27 % menor que lejos de '
        'la votación: por azar, menos de 1 entre 1.000.')
    assert pooled_text({'pooled': {'change': 0.004, 'p': 0.5}}, 'al final', 'lejos') == (
        'Con todas las parejas de todas las elecciones juntas, la separación al final es igual que lejos: como una moneda.')
    assert pooled_text({}, 'al final', 'lejos') is None
    both = pooled_both_text({'pooled': {'change': -0.232, 'p': 0.0}}, {'pooled': {'change': -0.062, 'p': 0.0575}})
    assert both == ('Con todas las parejas juntas: un 23 % menos en las tres últimas semanas que entre 3 y 8 semanas '
                    'antes (por azar, menos de 1 entre 1.000), y un 6 % menos hasta tres semanas antes que lejos '
                    '(por azar, 1 entre 17).')
    assert pooled_both_text({'pooled': {'change': 0.08, 'p': 0.9}}, {}) == (
        'Con todas las parejas juntas: un 8 % más en las tres últimas semanas que entre 3 y 8 semanas antes (como una moneda).')
    assert pooled_both_text({}, {}) is None


def test_herd_robustness_text_and_rows():
    summary = fixture_data('rebano-summary')
    assert robustness_text(summary) == (
        'Con 3, 5, 7 o 10 días entre los dos sondeos, la bajada conjunta al final queda entre el 23 y el 27 %, y en las '
        'tres últimas semanas entre el 21 y el 23 %; el recuento de elecciones que se acercan al final va de 11 de 18 '
        'con 3 días a 19 de 23 con 7. Con un mínimo de 6 comparaciones por pareja de casas (dos parejas de sondeos) '
        'quedan pocas elecciones: la tabla las enseña.')
    rows = robustness_rows(summary)
    assert len(rows) == 8 and rows[0]['default'] is True and (rows[0]['pair_days'], rows[0]['min_rows']) == (7, 3)
    assert rows[0]['far_final'] == {'count': '19 de 23', 'change': '−27 %', 'chance': 'menos de 1 entre 1.000'}
    assert rows[1]['far_final'] == {'count': '11 de 18', 'change': '−23 %', 'chance': 'menos de 1 entre 1.000'}
    assert rows[2]['far_final'] == {'count': '2 de 3', 'change': '−6 %', 'chance': 'como una moneda'}   # p 0,316 > COIN
    assert [(r['pair_days'], r['min_rows']) for r in rows] == [(7, 3), (3, 3), (3, 6), (5, 3), (5, 6), (7, 6), (10, 3), (10, 6)]
    assert robustness_text({}) is None and robustness_rows({}) == []
    # a cell without elections: dashes
    empty = dict(summary, robustness={'pair_days': [3], 'min_rows': [6], 'rows': [
        {'pair_days': 3, 'min_rows': 6, 'default': False, 'comparison': 'far_final', 'n': 0, 'k': 0,
         'median_change': None, 'change': None, 'p': None}]})
    assert robustness_rows(empty)[0]['far_final'] == {'count': '–', 'change': '–', 'chance': '–'}
    assert robustness_text(empty) is None
```

En `test_herd_page_structure_and_wording`: añadir `'bajada mediana del 15 % (entre el 3 y el 35 %)' in text`,
`'un 27 % menor que lejos de la votación: por azar, menos de 1 entre 1.000' in text`, `'Con todas las parejas juntas:
un 23 % menos en las tres últimas semanas' in text`, `'(entre el 2 y el 27 %)' in text`, `'id="robustness-table"' in
page`, `'Con otros días entre sondeos y otro mínimo de comparaciones' in text`, `page.count('<tr class="is-default">')
== 1`, `'en los recuentos, una elección con muchos sondeos no pesa más' in text`; cambiar la aserción de la
permutación a `'baraja 2.000 veces el tramo de cada pareja de sondeos, con sus partidos juntos, dentro de cada pareja
de casas' in text`; añadir `'prueba conjunta' in text and 'remuestrear las elecciones 2.000 veces' in text and 'con 3,
5, 7 y 10 días entre sondeos' in text`. Test nuevo `test_herd_page_without_pooled_ci_or_robustness`: publicar el
fixture con `parts={'summary': {'comparisons': <las tres sin pooled ni ci>, 'robustness': {}}}` y comprobar que la
página responde 200, no contiene «Con todas las parejas», «entre el 3 y el 35 %», `robustness-table` ni «Con otros
días», y sí «bajada mediana del 15 %» y «1 entre 769».

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_pages.py -k herd`
Expected: FAIL (`ImportError` de `ci_text` etc.; textos ausentes).

- [ ] **Step 3: Implement the view models and the context**

En `herd.py` (con docstrings numpy) y en `herd_context`.

- [ ] **Step 4: Edit the template**

- Tarjeta 1, `sum`: `… del {{ (med|abs * 100)|round|int }} %{% if ci.far_final %} ({{ ci.far_final }}){% endif %}` …
  tras la frase del azar: `{% if pooled.far_final %} {{ pooled.far_final }}{% endif %}`.
- Tarjeta 2, `sum`: tras `({{ chance.mid_final }})`: `{% set medm = mf.get('median_change') %}{% if medm is number and
  medm != 0 %}, con {{ 'una bajada mediana' if medm < 0 else 'una subida mediana' }} del {{ (medm|abs * 100)|round|int
  }} %{% if ci.mid_final %} ({{ ci.mid_final }}){% endif %}{% endif %}`; tras la frase de «Hasta tres semanas antes»:
  `{% if pooled.both %} {{ pooled.both }}{% endif %}`; la frase «Sólo en…» sigue al final.
- «Cómo lo medimos»: la frase pasa a «…y contamos por elección, sin mezclar sondeos de elecciones distintas: así, en
  los recuentos, una elección con muchos sondeos no pesa más que otra con pocos; en la prueba conjunta sí pesa más, y
  por eso damos las dos.» Tras el párrafo: `{% if robustness_rows %}<details class="how"><summary>Con otros días entre
  sondeos y otro mínimo de comparaciones</summary><div class="how-body">{% if robustness_text %}<p>{{ robustness_text
  }}</p>{% endif %}<div class="table-wrap"><table id="robustness-table"><caption class="sr-only">Recuentos y cambio
  conjunto con otros días entre sondeos y otro mínimo de comparaciones.</caption><thead><tr><th scope="col">Días</th>
  <th scope="col">Mínimo</th><th scope="col">Se acercan (lejos → final)</th><th scope="col">Cambio conjunto</th><th
  scope="col">Por azar</th><th scope="col">Se acercan (3-8 sem. → final)</th><th scope="col">Cambio conjunto</th><th
  scope="col">Por azar</th></tr></thead><tbody>{% for row in robustness_rows %}<tr{% if row.default %} class="is-default"
  {% endif %}>…</tr>{% endfor %}</tbody></table></div></div></details>{% endif %}` (la fila por defecto con sus celdas
  en `<b>`; CSS: `.is-default td { font-weight: 600; }` en `site.css` si no existe una regla equivalente).
- `tech`: la frase de la permutación → «La <b>permutación</b> baraja {{ permutations }} veces el tramo de cada pareja
  de sondeos, con sus partidos juntos, dentro de cada pareja de casas, y dice cuántas veces saldría una bajada tan
  grande sólo por azar: es lo que llamamos clara por sí sola.» Tras la prueba del signo: «La <b>prueba conjunta</b>
  baraja a la vez las parejas de sondeos de todas las parejas de casas de todas las elecciones y mide cuánto baja la
  separación juntándolas: cada pareja de casas pesa según sus comparaciones, así que las elecciones con muchos sondeos
  pesan más que en la prueba del signo. Un sondeo que entra en varias parejas cuenta en todas. El <b>intervalo</b> de
  la bajada mediana sale de remuestrear las elecciones {{ (params.get('resamples') or 2000)|int|num }} veces. La
  <b>robustez</b> repite los recuentos y la prueba conjunta con {{ pair_days|join(', ') con « y » final }} días entre
  sondeos y con un mínimo de {{ min_rows }} comparaciones por pareja.» (los `|num`/`fmt_int` que use la plantilla
  para «2.000»; `params.robustness` leído de `study_meta.params`, con los valores por defecto de la spec si falta).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest -q tests/test_pages.py -k "herd or rebano or analisis" && python -m pytest -q`
Expected: PASS; la batería completa verde.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/pages/herd.py mtpy/lib/pages/analisis.py web/templates/analisis_rebano.html web/dist/css/site.css tests/test_pages.py
git commit -m "Rebaño: intervalo y prueba conjunta en las cajas, nota de robustez y método en la página"
```

---

### Aceptación (controlador, tras la Tarea 4, sólo lectura con el `.env` del checkout principal)

Calcular el estudio con datos reales y comprobar: `pooled` lejos → final ≈ −0,27 con `p` < 0,001, 3-8 → final ≈ −0,23
con `p` < 0,001, lejos → 3-8 ≈ −0,06 con `p` ≈ 0,06; intervalos [−0,35, −0,03], [−0,27, −0,02], [−0,28, +0,19]; «claras»
3 / 2 / 4; 24 filas de robustez; tiempo total < 45 s. Después, la página con las partes reales en Chrome headless
(`check_app_real.py` del scratchpad) para leer las frases y abrir la nota.
