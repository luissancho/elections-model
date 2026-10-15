# Web, fase 9: el ejemplo vivo del promedio — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un bloque nuevo en `/metodo/promedio` que muestre, para el partido líder del último run nacional, la tabla de los sondeos que pesan en el promedio del día de la estimación y un gráfico interactivo con esos sondeos, la recta local, su banda y la campana del kernel, alimentado por una parte nueva del run, `weights`.

**Architecture:** El Forecaster gana un método único `weights_at(name, date)` del que tiran `print_weights`, `plot_weights` y un export nuevo `export_weights`, que `publish_forecast` escribe como parte `weights` del run (schema, CSV, contrato). La página del método lee la parte del último run nacional en `pages/metodo.py` (`weights_example`), la plantilla renderiza tabla y textos en el servidor, y un módulo ECharts nuevo (`charts/weights.js`) dibuja el gráfico desde la isla `#initial-data`.

**Tech Stack:** Python 3.11 (pandas, numpy, Jinja2), ECharts 5.6 en JS vanilla (módulos ES), pytest 9 (`pytest` del shim de pyenv) y node 25 para las opciones ECharts.

**Spec:** `docs/superpowers/specs/2026-10-15-web-fase-9-ejemplo-promedio-design.md`

## Global Constraints

- Rama `dev`. El árbol tiene cambios sin commit de Luis en `mtpy/lib/pages/__init__.py`, `mtpy/lib/pages/casas.py`, `tests/test_pages.py`, `web/dist/js/pages/casas.js`, `web/templates/casa.html` y la spec de la fase 8: **no se tocan ni se añaden al índice**. Cada commit añade solo los ficheros del paso con `git add <rutas>`; nunca `git add -A` ni `git add .`. Los tests nuevos de páginas van en un fichero nuevo, `tests/test_pages_ejemplo.py`, no en `tests/test_pages.py`.
- Tests: `pytest` (pyenv, Python 3.11.9); `pytest.ini` convierte `FutureWarning` en error. Suite actual: 759 tests, todos en verde antes de empezar (`pytest -q`).
- Nombres técnicos fijos: parte `weights`, schema `weights` (`weights@1`), columnas `WEIGHTS_COLUMNS = ['name', 'date', 'pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight', 'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating']`, método `Forecaster.weights_at`, export `export_weights(fc, names, date)`, función de página `weights_example()`, constante `EXAMPLE_SHOWN = 20`, ids HTML `ejemplo`, `weights-table`, `weights-more`, `weights-chart`, módulo `web/dist/js/charts/weights.js` con `weightsOption(example)` y `renderWeights(el, example)`.
- Umbral de pesos `1e-2`, el del estimador local (`mtpy/core/utils/stat.py:986`), sobre los pesos reescalados por `Estimator.build_input` (suman `neff`), no sobre los brutos.
- Redondeos del paquete: porcentajes y `err` a `PCT_DECIMALS`, pesos y `kernel.values` a `PROB_DECIMALS` (3), `bandwidth` y `neff` a 1 decimal, `nobs` entero. Fechas ISO `YYYY-MM-DD`.
- Textos públicos en castellano, en el tono de `metodo_promedio.html`; la palabra pública del herding es «rebaño». Sin `script` inline ni recursos externos (CSP `default-src 'self'`). La página sigue leyéndose sin JavaScript: solo el gráfico lo dibuja el navegador.
- Docstrings en todas las funciones nuevas (AGENTS.md), estilo numpydoc como el resto del paquete.

## Review Focus

1. Un run antiguo sin `weights.json` (todos los publicados hasta hoy): `/metodo/promedio` debe salir exactamente como hoy, sin bloque ni isla, con 200 → test en la tarea 6.
2. Un partido que no está en los últimos sondeos (su rejilla acaba antes de `date`): `weights_at` lee al final de su rejilla y lo registra en `stats['date']`, sin desbordar ni devolver NaN → test en la tarea 1.
3. Un partido con menos de tres sondeos útiles, o con `date` anterior a su primer sondeo: `weights_at` devuelve `None` y `export_weights` lo omite de `parties` sin romper el run → tests en las tareas 1 y 4.
4. Más de `EXAMPLE_SHOWN` sondeos (lo normal, un centenar en las generales): las filas sobrantes salen ocultas y el botón «Cargar más» con `data-batch` → test en la tarea 6.
5. Un sondeo sin `sponsor` o sin `sample_size` (frecuente): la tabla imprime guion y el tooltip no rompe → tests en las tareas 6 y 7 (el fixture y `EXAMPLE` llevan `sponsor: null`).

---

### Task 1: `Forecaster.weights_at` y la máscara del estimador

**Files:**
- Modify: `mtpy/core/utils/stat.py:986-1009` (`LocalKernelEstimator.get_local_estimator`)
- Modify: `mtpy/lib/forecaster.py:217` (`__init__`), `:581-640` (`fit_forecast`), nuevo método tras `fit_forecast`
- Test: `tests/test_forecaster_unit.py` (`_synthetic_forecaster` en `:105-142` y tests nuevos)

**Interfaces:**
- Consumes: `LocalKernelEstimator(x, weights, groups, **reg_params)`, `.build_pred(ix)`, `.get_kernel()`, `.get_local_estimator(kws, pos)`, `.pred`, `.ranges`, `.ts_freq`, `.weights`; `Kernel.get_weights(p)`, `Kernel.get_bw_adaptive(x, weights, bw, p=...)`, `Kernel.kernel_gaussian(x, p, bw)` (estático), `Kernel.bw`, `Kernel.bw_type`; `ts_from_delta`, `add_delta` ya importados en `forecaster.py`.
- Produces:
  - `LocalKernelEstimator.local_mask(self, kws: np.ndarray, pos: int) -> np.ndarray` (bool, una entrada por observación): `np.abs(kws[pos] * self.weights.squeeze()) >= 1e-2`; `get_local_estimator` pasa a usarla.
  - `Forecaster.max_fc: int` (0 en `__init__`; `fit_forecast` guarda el `max_fc` con que ajusta).
  - `Forecaster.weights_at(self, name: str, date: Optional[str | pd.Timestamp] = None) -> Optional[dict]` con las claves `polls` (DataFrame indexado por fecha, columnas `pollster_id, pollster, sponsor, sample_size, pct, pct_adj, weight, weight_kernel, weight_over, weight_sample, weight_rating`, orden de fecha ascendente), `line` (DataFrame diario, columnas `mean, cmin, cmax, err`), `kernel` (Series diaria en `[0, 1]`, mismo índice que `line`) y `stats` (dict `date` (Timestamp), `mean, cmin, cmax, err` (float), `nobs` (int), `neff` (float), `bandwidth` (float, días)). `None` si hay menos de tres sondeos útiles o `date` es anterior al primero.

- [ ] **Step 1: Write the failing tests**

En `tests/test_forecaster_unit.py`, añadir `fc.max_fc = 0` a `_synthetic_forecaster` (junto a `fc.he_enabled, ... = ...`) y, al final del fichero:

```python
# --- Fase 9: los pesos de los sondeos en un día (`weights_at`) ---
def test_weights_at_matches_the_fitted_average_and_its_band_on_the_same_grid():
    fc, _, _ = _synthetic_forecaster()
    fc.fit_forecast(names=['PP'], max_fc=10)
    when = fc.date_fit_last
    out = fc.weights_at('PP')                       # date=None → date_fit_last
    assert fc.max_fc == 10 and out['stats']['date'] == when
    cell = fc.fc_stat.loc[when, 'PP']
    assert out['stats']['mean'] == pytest.approx(fc.forecast.loc[when, 'PP'])
    assert out['stats']['cmin'] == pytest.approx(cell['cmin']) and out['stats']['cmax'] == pytest.approx(cell['cmax'])
    assert out['stats']['err'] == pytest.approx(cell['err']) and out['stats']['nobs'] == cell['nobs']
    assert out['stats']['neff'] == pytest.approx(cell['neff']) and out['stats']['bandwidth'] > 0
    polls = out['polls']
    assert len(polls) == cell['nobs'] and polls.index.is_monotonic_increasing
    assert list(polls.columns) == ['pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight',
                                   'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating']
    assert np.allclose(polls['weight'], polls['weight_kernel'] * polls['weight_over'] * polls['weight_sample'] * polls['weight_rating'])
    assert np.allclose(polls['pct'], fc.fc_series_raw.loc[polls.index, 'PP'])     # lo publicado
    assert np.allclose(polls['pct_adj'], fc.fc_series.loc[polls.index, 'PP'])     # lo que ajusta la recta
    assert not np.allclose(polls['pct'], polls['pct_adj']) and polls['sponsor'].isnull().all()
    line, kernel = out['line'], out['kernel']
    assert line.index[-1] == when and (line.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert line.loc[when, 'mean'] == pytest.approx(out['stats']['mean'])
    assert line.index[0] <= polls.index[0]                       # la ventana empieza en el primer sondeo con kernel apreciable
    assert kernel.index.equals(line.index) and kernel.loc[when] == pytest.approx(1.0) and kernel.max() <= 1.0
    assert kernel.iloc[0] < kernel.loc[when]


def test_weights_at_reads_a_party_without_recent_polls_at_the_end_of_its_grid():
    fc, _, _ = _synthetic_forecaster()
    dates = fc.series.index.get_level_values('date')
    cut = fc.date_last - pd.Timedelta(days=60)
    fc.series.loc[dates > cut, 'PSOE'] = np.nan
    fc.series_raw.loc[dates > cut, 'PSOE'] = np.nan
    fc.fit_forecast(names=['PP', 'PSOE'], max_fc=10, fillna=True)
    last = fc.series.loc[fc.series['PSOE'].notnull()].index.get_level_values('date').max()
    end = last + pd.Timedelta(days=10)
    out = fc.weights_at('PSOE')
    assert end < fc.date_fit_last and out['stats']['date'] == end
    assert out['stats']['mean'] == pytest.approx(fc.forecast.loc[end, 'PSOE'])
    assert out['line'].index[-1] == end and out['polls'].index.max() <= last


def test_weights_at_returns_none_without_three_polls_or_before_the_first():
    fc, _, _ = _synthetic_forecaster()
    assert fc.weights_at('PP', date='2025-12-01') is None
    dates = fc.series.index.get_level_values('date')
    fc.series.loc[dates > dates.min() + pd.Timedelta(days=15), 'PSOE'] = np.nan   # quedan dos sondeos
    with pytest.warns(UserWarning):
        assert fc.weights_at('PSOE') is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_forecaster_unit.py -k weights_at -v`
Expected: 3 FAIL con `AttributeError: 'Forecaster' object has no attribute 'weights_at'`

- [ ] **Step 3: Implement `LocalKernelEstimator.local_mask` in `mtpy/core/utils/stat.py`**

Añadir el método justo antes de `get_local_estimator` y sustituir en este las dos líneas `weights = kws[pos] * self.weights.squeeze()` / `vind = np.abs(weights) >= 1e-2` por `weights = kws[pos] * self.weights.squeeze()` / `vind = self.local_mask(kws, pos)`. Docstring: la máscara de observaciones de la regresión local en el punto `pos`.

- [ ] **Step 4: Implement `Forecaster.max_fc` and `Forecaster.weights_at` in `mtpy/lib/forecaster.py`**

- `__init__`: `self.max_fc = 0  # Days forecast after the last poll by the last `fit_forecast` (the grid of `fit`)` junto a `self.date_fit_last`.
- `fit_forecast`: `self.max_fc = max_fc` antes del bucle de nombres.
- `weights_at`, tras `fit_forecast`. Pasos que la firma no fija:
  1. `df = self.fc_series[self.fc_series[name].notnull() & (self.fc_series['weight'] > 0)]`; si `len(df) < 3`, `warnings.warn` como `fit` y `return None`.
  2. `date`: `pd.Timestamp(date)` si se da; si no, `self.date_fit_last` si no es `None`, si no `min(pd.to_datetime('now').floor('d'), df.index.max())`. Si `date < df.index.min()`, `return None`.
  3. Rejilla de `fit`: `ix = self.fc_index`; `px = ix[(ix >= df.index.min()) & (ix <= add_delta(df.index.max(), f'P{self.max_fc}D'))]`. Si `date > px.max()`, `date = px.max()`.
  4. Estimador como `fit`: `groups` por `pollster_id` si `cov_type == 'cluster'`; `est = LocalKernelEstimator(df[name], weights=df.weight.values, groups=groups, **self.reg_params)`; `kws = est.build_pred(px).get_kernel().get_weights(est.pred)`; `pos = (date - px.min()).days`.
  5. Máscara `mask = est.local_mask(kws, pos)`; `loc_est = est.get_local_estimator(kws, pos)`; si `None`, `return None`.
  6. `polls`: `df[mask]` con `weight_kernel = kws[pos][mask]`, `weight = weight_kernel * df.weight[mask]`, `pct = self.fc_series_raw.loc[...]` alineado por posición de fila (misma indexación que `fc_series`: `self.fc_series_raw[self.fc_series[name].notnull() & (self.fc_series['weight'] > 0)][name].values[mask]`), `pct_adj = df[name][mask]`, `sponsor = df['sponsor'] if 'sponsor' in df else None`, `sample_size = df.get('sample_size')`; columnas en el orden de Interfaces; `sort_index()`.
  7. Ventana: `kernel_all = kws[pos]`; `dt_from = df.index[kernel_all >= 1e-2].min()`; `dt_to = max(df.index[kernel_all >= 1e-2].max(), date)`; `days = pd.date_range(dt_from, dt_to, freq='D')`.
  8. `line`: `deltas = np.asarray(ts_to_delta(days, dt_from=est.ranges[0][0], freq=est.ts_freq), dtype=float)` (importar `ts_to_delta` de `..core.utils.dates` junto a `add_delta` y `ts_from_delta`; es la conversión de `Estimator.build_pred`, así que `deltas` está en la unidad de `est.pred`); `line = loc_est.fit(deltas, alpha=self.alpha)`; `line.index = days`; columnas `mean, cmin, cmax, err`. No llamar a `est.build_pred(days)`: sobrescribiría `est.pred`, del que dependen `kws` y `pos`.
  9. `bandwidth`: `kernel = est.get_kernel()`; si `kernel.bw_type == 'fixed'`, `float(kernel.bw[0])`; si no, `float(kernel.get_bw_adaptive(kernel.data[:, 0], kernel.weights, kernel.bw[0], p=est.pred[:, 0])[pos])`.
  10. `kernel`: `pd.Series(Kernel.kernel_gaussian(deltas, est.pred[pos, 0], bandwidth).squeeze(), index=days)`.
  11. `stats`: `{'date': date, 'mean', 'cmin', 'cmax', 'err'` de `line.loc[date]`, `'nobs': int(loc_est.nobs), 'neff': float(loc_est.neff), 'bandwidth': bandwidth}`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_forecaster_unit.py tests/test_stat.py -q`
Expected: todos PASS (los 3 nuevos incluidos)

- [ ] **Step 6: Commit**

```bash
git add mtpy/core/utils/stat.py mtpy/lib/forecaster.py tests/test_forecaster_unit.py
git commit -m "Compute the polls that weigh on the average of one day with the grid and the band of the fit (weights_at)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `print_weights` y `plot_weights` sobre `weights_at`

**Files:**
- Modify: `mtpy/lib/forecaster.py:1488-1592` (`print_weights`), `:1593-1717` (`plot_weights`)
- Test: `tests/test_forecaster_unit.py`

**Interfaces:**
- Consumes: `Forecaster.weights_at` (tarea 1).
- Produces: las mismas firmas de hoy. `print_weights(name, date=None, show=True)` devuelve con `show=False` un DataFrame con las columnas `['date', 'pollster', 'weight', 'tfs', 'tte', 'sample_size', 'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating']`, `date` formateada `%d-%b`, `tfs` = días desde el primer sondeo de la tabla, `tte` = días hasta el último. `plot_weights` pinta la campana diaria (`kernel`) en el eje gemelo y los puntos con `pct_adj` como hoy (`df[name]`).

- [ ] **Step 1: Write the failing test**

```python
def test_print_weights_keeps_its_columns_and_plot_weights_draws_the_daily_kernel():
    fc, _, _ = _synthetic_forecaster()
    out = fc.weights_at('PP', date='2026-06-01')
    table = fc.print_weights('PP', date='2026-06-01', show=False)
    assert list(table.columns) == ['date', 'pollster', 'weight', 'tfs', 'tte', 'sample_size', 'weight_kernel',
                                   'weight_over', 'weight_sample', 'weight_rating']
    assert len(table) == len(out['polls']) and table['tfs'].min() == 0 and table['tte'].min() == 0
    assert table['date'].iloc[0] == out['polls'].index[0].strftime('%d-%b')
    assert np.allclose(table['weight'], out['polls']['weight'])
    _, ax = matplotlib.pyplot.subplots()
    fc.plot_weights('PP', date='2026-06-01', show=False, ax=ax)
    twin = [a for a in ax.figure.axes if a is not ax][0]
    assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == len(out['polls'])
    assert len(twin.lines) == 1 and len(twin.lines[0].get_xdata()) == len(out['kernel'])
    matplotlib.pyplot.close('all')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_forecaster_unit.py -k print_weights_keeps -v`
Expected: FAIL (`AttributeError: 'DataFrame' object has no attribute 'tfs'`, porque la serie sintética no trae `tfs`/`tte`, o las longitudes no coinciden)

- [ ] **Step 3: Rewrite `print_weights` and `plot_weights` on top of `weights_at`**

- `print_weights`: `out = self.weights_at(name, date)`; si `None`, `return None`. `df = out['polls'].reset_index()`; `df['tfs'] = (df.date - df.date.min()).dt.days`; `df['tte'] = (df.date.max() - df.date).dt.days`; `df['date'] = df.date.dt.strftime('%d-%b')`; seleccionar las diez columnas; el resto (`bars`, `get_df_styler`, `print_styler`) igual que hoy.
- `plot_weights`: `out = self.weights_at(name, date)`; `df = out['polls']`, `dr = out['line']` (con `dr['cint'] = dr['cmax'] - dr['mean']`), `date = out['stats']['date']`; puntos en `(df.index, df['pct_adj'])` con el tamaño y el color de hoy; `plot_series(dr[['mean', 'cmin', 'cmax']], dt_from=dr.index[0], dt_to=dr.index[-1], ...)`; `ax2.plot(out['kernel'].index.values, out['kernel'].values, ...)`; leyenda, título, nota y `plot_figure` como hoy. Las etiquetas `lbl_reg`/`lbl_score` usan `out['stats']`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_forecaster_unit.py -q`
Expected: todos PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/forecaster.py tests/test_forecaster_unit.py
git commit -m "Draw and print the weights of the polls from weights_at, with the cluster band and the daily kernel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: la parte `weights` del paquete: schema, fixture y contrato

**Files:**
- Modify: `mtpy/lib/bundle.py:60` (`RUN_PARTS`), `:78-123` (`SCHEMAS`)
- Create: `tests/fixtures/bundle/weights.json`
- Modify: `docs/web/contrato.md:29-34` (árbol), `:159-166` (tras `### dispersion`), `:404` (ruta `/api/v1/forecast/{scope}/{part}`), `:452-453` (gemelos CSV)
- Test: `tests/test_bundle_unit.py` (parametrizado sobre `SCHEMAS`, sin cambios), `tests/test_web_api.py`

**Interfaces:**
- Produces: `bundle.RUN_PARTS == ('meta', 'headline', 'series', 'polls', 'fan', 'house-effects', 'dispersion', 'weights')`; `bundle.SCHEMAS['weights'] == {'date': str, 'parties': list, 'rows': list, 'line': dict, 'kernel': dict, 'stats': dict}`; el fixture `weights.json` que `write_bundle` (`tests/test_webapi_unit.py:24`) escribe en el run de pruebas `20261008-120000`.

- [ ] **Step 1: Write the failing test**

En `tests/test_web_api.py`, tras `test_csv_twins`:

```python
def test_weights_part_is_served_with_the_run(api):
    status, headers, body = call(api, '/api/v1/forecast/es/weights')
    assert status == 200 and headers['cache-control'] == 'public, max-age=60'
    data = json.loads(body)
    assert data['schema'] == 'weights@1' and data['data']['parties'] == ['PP'] and data['data']['date'] == '2026-10-08'
    assert data['data']['stats']['PP']['nobs'] == 3 == len(data['data']['rows'])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_web_api.py -k weights_part -v`
Expected: FAIL (400 `invalid part` o 404)

- [ ] **Step 3: Declare the part and write the fixture**

- `bundle.py`: añadir `'weights'` al final de `RUN_PARTS` y la entrada `'weights'` en `SCHEMAS` tras `'dispersion'`. Actualizar el docstring del módulo (`:37`) si enumera las partes.
- `tests/fixtures/bundle/weights.json`, con el sobre de `dispersion.json` (`"schema": "weights@1"`, `run_id` `20261008-120000`, `generated_at` `2026-10-08T12:00:00Z`) y `data`:
  - `"date": "2026-10-08"`, `"parties": ["PP"]`.
  - `rows` (tres, por fecha): `{"name": "PP", "date": "2026-09-24", "pollster_id": 3, "pollster": "GAD3", "sponsor": "ABC", "sample_size": 1200, "pct": 32.4, "pct_adj": 32.9, "weight": 0.512, "weight_kernel": 0.64, "weight_over": 1.0, "weight_sample": 0.8, "weight_rating": 1.0}`, `{"name": "PP", "date": "2026-10-01", "pollster_id": 1, "pollster": "Sigma Dos", "sponsor": null, "sample_size": 2000, "pct": 33.5, "pct_adj": 33.2, "weight": 1.138, "weight_kernel": 0.9, "weight_over": 1.0, "weight_sample": 1.265, "weight_rating": 1.0}`, `{"name": "PP", "date": "2026-10-05", "pollster_id": 7, "pollster": "Sigma Dos", "sponsor": null, "sample_size": null, "pct": 33.1, "pct_adj": 32.8, "weight": 0.98, "weight_kernel": 0.98, "weight_over": 1.0, "weight_sample": 1.0, "weight_rating": 1.0}`.
  - `line.PP`: `dates` los 15 días del `2026-09-24` al `2026-10-08`; `mean` lineal de 32.8 a 33.1 (paso 0.3/14, 2 decimales); `lo = mean − 0.7`; `hi = mean + 0.7`.
  - `kernel.PP`: los mismos `dates`; `values[i] = round(exp(−((i − 14) / 42)² / 2), 3)` (termina en 1.0).
  - `stats.PP`: `{"date": "2026-10-08", "mean": 33.1, "lo": 32.4, "hi": 33.8, "err": 0.35, "nobs": 3, "neff": 2.6, "bandwidth": 42.0}`.
- `docs/web/contrato.md`: `weights.json` y `csv/weights.csv` en el árbol del run; sección `### weights` tras `### dispersion`: fuente `Forecaster.weights_at` leído en `sim.as_of` (enmienda 2026-10-15, fase 9; los runs anteriores no la tienen), `data` con `date`, `parties`, `rows` (las 13 `WEIGHTS_COLUMNS`, formato largo, una fila por partido y sondeo de la regresión de ese día; porcentajes a 1 decimal, pesos a 3), `line` (`{partido: {dates, mean, lo, hi}}`, la recta local del día en la ventana), `kernel` (`{partido: {dates, values}}`, campana en `[0, 1]`) y `stats` (`{partido: {date, mean, lo, hi, err, nobs, neff, bandwidth}}`); CSV `csv/weights.csv` = `rows`. Añadir `weights` a la lista de `part` de `/api/v1/forecast/{scope}/{part}` y a la de gemelos CSV.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bundle_unit.py tests/test_web_api.py tests/test_webapi_unit.py -q`
Expected: todos PASS (`test_validate_accepts_every_fixture_and_rejects_a_missing_key[weights]` incluido)

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/bundle.py tests/fixtures/bundle/weights.json docs/web/contrato.md tests/test_web_api.py
git commit -m "Declare the weights part of the run in the bundle, its fixture and the contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `export_weights` en la publicación

**Files:**
- Modify: `mtpy/lib/publish/forecast.py` (constante `WEIGHTS_COLUMNS` junto a `HE_COLUMNS`/`DISP_COLUMNS`; `export_weights` tras `export_dispersion`; la tupla `cycle` de `publish_forecast`, `:770-773`)
- Modify: `mtpy/lib/publish/__init__.py:42-46` (re-exportar `export_weights`, `WEIGHTS_COLUMNS`)
- Modify: `tests/fakes.py:23-63` (`synthetic_forecaster`)
- Test: `tests/test_publish_unit.py`

**Interfaces:**
- Consumes: `fc.weights_at(name, date)` (tarea 1; en los tests, el doble de `tests/fakes.py`); `sim.as_of` (Timestamp); `round_cols`, `PCT_DECIMALS`, `PROB_DECIMALS` del módulo.
- Produces: `export_weights(fc, names, date) -> tuple[dict, pd.DataFrame]` con `data = {'date': 'YYYY-MM-DD', 'parties': [...], 'rows': [...], 'line': {partido: {'dates', 'mean', 'lo', 'hi'}}, 'kernel': {partido: {'dates', 'values'}}, 'stats': {partido: {'date', 'mean', 'lo', 'hi', 'err', 'nobs', 'neff', 'bandwidth'}}}` y `frame` con `WEIGHTS_COLUMNS`. `publish_forecast` escribe `weights.json` y `csv/weights.csv` tras `dispersion`.

- [ ] **Step 1: Give the test double a `weights_at`**

En `synthetic_forecaster` de `tests/fakes.py`, antes del `return`, una función local que el `SimpleNamespace` exporta como `weights_at=weights_at`:

```python
    def weights_at(name, date=None):
        """Pesos sintéticos: kernel creciente con la fecha, recta plana en el promedio y campana de 30 días."""
        if name not in NAMES:
            return None
        when = pd.Timestamp(date) if date is not None else pd.Timestamp(date_fit_last)
        frame = polls[polls[name].notnull()]
        out = pd.DataFrame({
            'pollster_id': frame['pollster_id'], 'pollster': frame['pollster'], 'sponsor': frame['sponsor'],
            'sample_size': frame['sample_size'], 'pct': frame[name], 'pct_adj': frame[name] - 0.3,
            'weight_kernel': np.linspace(0.4, 1., len(frame)), 'weight_over': 1., 'weight_sample': frame['weight'],
            'weight_rating': 1.,
        }, index=frame.index)
        out['weight'] = out['weight_kernel'] * out['weight_over'] * out['weight_sample'] * out['weight_rating']
        out = out[['pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight', 'weight_kernel',
                   'weight_over', 'weight_sample', 'weight_rating']]
        days = pd.date_range(frame.index.min(), max(frame.index.max(), when), freq='D')
        mean = float(forecast[name].dropna().iloc[0])
        line = pd.DataFrame({'mean': mean, 'cmin': mean - 2., 'cmax': mean + 2., 'err': 1.}, index=days)
        kernel = pd.Series(np.exp(-np.square((days - when).days / 30.) / 2), index=days)
        stats = {'date': when, 'mean': mean, 'cmin': mean - 2., 'cmax': mean + 2., 'err': 1., 'nobs': len(frame),
                 'neff': 0.8 * len(frame), 'bandwidth': 30.}
        return {'polls': out, 'line': line, 'kernel': kernel, 'stats': stats}
```

- [ ] **Step 2: Write the failing tests**

En `tests/test_publish_unit.py`, tras `test_export_fan_house_effects_and_dispersion`:

```python
def test_export_weights_reads_the_as_of_day_and_writes_the_long_csv():
    sim = synthetic_simulator()
    data, frame = publish.export_weights(sim.model, NAMES, sim.as_of)
    data = valid('weights', data, mode=None)
    assert data['date'] == '2026-10-05' and data['parties'] == NAMES
    assert list(frame.columns) == publish.WEIGHTS_COLUMNS and len(frame) == 6 + 6 + 5
    row = data['rows'][0]
    assert row['name'] == 'PP' and row['date'] == '2026-09-05' and row['sponsor'] is None and row['pct'] == 41.0
    assert row['pct_adj'] == 40.7 and row['weight'] == round(0.4 * 1.2, 3) and row['weight_kernel'] == 0.4
    assert data['rows'][6]['name'] == 'PSOE' and {r['name'] for r in data['rows']} == set(NAMES)
    line = data['line']['PP']
    assert line['dates'][0] == '2026-09-05' and line['dates'][-1] == '2026-10-05' and len(line['dates']) == 31
    assert line['mean'][0] == 40.0 and line['lo'][0] == 38.0 and line['hi'][0] == 42.0
    kernel = data['kernel']['PP']
    assert kernel['dates'] == line['dates'] and kernel['values'][-1] == 1.0 and kernel['values'][0] < 1.0
    assert data['stats']['PP'] == {'date': '2026-10-05', 'mean': 40.0, 'lo': 38.0, 'hi': 42.0, 'err': 1.0, 'nobs': 6,
                                   'neff': 4.8, 'bandwidth': 30.0}
    assert data['stats']['VOX']['nobs'] == 5


def test_export_weights_skips_a_party_without_fit():
    sim = synthetic_simulator()
    original = sim.model.weights_at
    sim.model.weights_at = lambda name, date=None: None if name == 'VOX' else original(name, date)
    data, frame = publish.export_weights(sim.model, NAMES, sim.as_of)
    assert data['parties'] == ['PP', 'PSOE'] and set(data['stats']) == {'PP', 'PSOE'} and len(frame) == 12
```

Y en `test_publish_forecast_backfill_freezes_the_polls_and_stamps_the_day`, tras leer `head`:

```python
    weights = writer.read_json(bundle.path_part('es', '20261005-120000', 'weights'))['data']
    assert weights['date'] == head['as_of'] == '2026-10-05'
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_publish_unit.py -k "weights or backfill_freezes or whole_layout" -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'export_weights'`; `whole_layout` falla porque `weights.json` no se escribe)

- [ ] **Step 4: Implement `export_weights` and wire it into `publish_forecast`**

- `WEIGHTS_COLUMNS` como en Global Constraints.
- `export_weights(fc, names, date)`: `iso = pd.Timestamp(date).strftime('%Y-%m-%d')`; por cada `name`, `out = fc.weights_at(name, date=date)`, omitido si `None`. Filas: `out['polls'].reset_index()` con `name` delante, `date` ISO, `round_cols(pct=('pct', 'pct_adj'), prob=('weight', 'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'))`, `sample_size` entero o `None`. `line`: fechas ISO, `mean`/`lo`/`hi` desde `mean`/`cmin`/`cmax` a `PCT_DECIMALS` (`None` donde NaN, como `_stat_bound`). `kernel`: `values` a 3 decimales. `stats`: `date` ISO de `stats['date']`, `mean`/`lo`/`hi`/`err` a `PCT_DECIMALS`, `nobs` int, `neff` y `bandwidth` a 1 decimal. `frame = pd.DataFrame(rows, columns=WEIGHTS_COLUMNS)`.
- `publish_forecast`: `('weights', export_weights(fc, names, sim.as_of))` al final de la tupla `cycle`.
- `publish/__init__.py`: re-exportar `WEIGHTS_COLUMNS` y `export_weights`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_publish_unit.py tests/test_jobs_publish.py -q`
Expected: todos PASS

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish/forecast.py mtpy/lib/publish/__init__.py tests/fakes.py tests/test_publish_unit.py
git commit -m "Publish the weights of the polls of the estimation day as a part of the run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: el contexto del ejemplo en `pages/metodo.py`

**Files:**
- Modify: `mtpy/lib/pages/metodo.py` (constante `EXAMPLE_SHOWN`, `weights_example`, `metodo_section_context`)
- Create: `tests/test_pages_ejemplo.py`

**Interfaces:**
- Consumes: `part(scope, run, name)` y `site().latest_run(scope)` ya importados en `metodo.py`; `api_url(scope, part, fmt=...)` de `.common` (importarla); `HttpError` (404 cuando la parte no existe); `common_context`.
- Produces:
  - `EXAMPLE_SHOWN = 20`.
  - `weights_example() -> Optional[dict]` con `name`, `fullname`, `color` (de `meta['parties']`), `date` (`weights['date']`), `run` (id del run), `stats` (los del líder), `rows` (las filas del líder, fecha ascendente, tal como vienen en la parte), `n_rows`, `weight_max` (`max(3, max(row['weight']))`), `line`, `kernel`. El líder es el partido de mayor `pct` de `headline['nowcast']['parties']`. `None` si la parte `weights` da 404, si `headline` no trae partidos o si el líder no está en `weights['parties']`.
  - En `metodo_section_context(state, 'promedio')`: `context['example'] = weights_example()`, `context['example_shown'] = EXAMPLE_SHOWN`, `context['example_hidden'] = example is not None and example['n_rows'] > EXAMPLE_SHOWN`, `context['example_csv'] = api_url(NATIONAL_SCOPE, 'weights', fmt='csv')` y, si `example` no es `None`, `context['initial'] = {'state': context['state'], 'example': {'name', 'color', 'date', 'stats', 'rows' (solo `date, pollster, sponsor, sample_size, pct, pct_adj, weight, weight_kernel, weight_over, weight_sample, weight_rating`), 'line', 'kernel'}}`. Las demás secciones no llevan ninguna de estas claves.

- [ ] **Step 1: Write the failing tests**

`tests/test_pages_ejemplo.py`:

```python
"""El ejemplo vivo del promedio en `/metodo/promedio` (fase 9): contexto, plantilla e isla de datos."""
import os
import re

import pytest

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages
from mtpy.lib.pages import metodo
from tests.test_api_asgi import call
from tests.test_pages import api, html, initial  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data

STATE = {'scope': 'es', 'mode': None, 'run': None}


def without_weights(tmp_path):
    """Quita la parte `weights` del run de pruebas: un run publicado antes de la fase 9."""
    os.remove(os.path.join(str(tmp_path), bundle.path_part('es', RUN, 'weights')))


def test_weights_example_picks_the_leader_and_its_slice(api):
    example = metodo.weights_example()
    assert example['name'] == 'PP' and example['fullname'] == fixture_data('meta')['parties'][0]['fullname']
    assert example['color'] == '#1d84ce' and example['date'] == '2026-10-08' and example['run'] == RUN
    assert example['n_rows'] == 3 == example['stats']['nobs'] and example['weight_max'] == 3
    assert [row['date'] for row in example['rows']] == ['2026-09-24', '2026-10-01', '2026-10-05']
    assert example['line']['dates'][-1] == '2026-10-08' and len(example['kernel']['values']) == 15


def test_weights_example_is_none_without_the_part_or_the_leader(api, tmp_path, monkeypatch):
    weights = fixture_data('weights')
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None:
                        {**weights, 'parties': ['PSOE']} if name == 'weights' else fixture_data(name))
    assert metodo.weights_example() is None
    monkeypatch.undo()
    without_weights(tmp_path)
    assert metodo.weights_example() is None


def test_weights_example_lets_other_errors_through(api, monkeypatch):
    def boom(scope, run, name, mode=None):
        raise HttpError(503, 'bundle unavailable')
    monkeypatch.setattr(metodo, 'part', boom)
    with pytest.raises(HttpError) as error:
        metodo.weights_example()
    assert error.value.status == 503


def test_promedio_context_carries_the_example_and_its_island(api):
    context = pages.metodo_section_context(STATE, 'promedio')
    assert context['example']['name'] == 'PP' and context['example_shown'] == 20 and context['example_hidden'] is False
    assert context['example_csv'] == '/api/v1/forecast/es/weights?format=csv'
    assert set(context['initial']) == {'state', 'example'}
    island = context['initial']['example']
    assert set(island) == {'name', 'color', 'date', 'stats', 'rows', 'line', 'kernel'}
    assert set(island['rows'][0]) == {'date', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight',
                                      'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'}
    other = pages.metodo_section_context(STATE, 'escanos')
    assert 'example' not in other and 'example_csv' not in other and 'initial' not in other


def test_promedio_context_without_the_part_has_no_island(api, tmp_path):
    without_weights(tmp_path)
    context = pages.metodo_section_context(STATE, 'promedio')
    assert context['example'] is None and context['example_hidden'] is False and 'initial' not in context
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pages_ejemplo.py -v`
Expected: 5 FAIL (`AttributeError: module ... has no attribute 'weights_example'`, `KeyError: 'example'`)

- [ ] **Step 3: Implement `EXAMPLE_SHOWN`, `weights_example` and the context keys in `mtpy/lib/pages/metodo.py`**

`weights_example` sigue el patrón de `national_figures`: `run = site().latest_run(NATIONAL_SCOPE)`; `headline = part(NATIONAL_SCOPE, run, 'headline')`; `parties = headline.get('nowcast', {}).get('parties') or []`; sin partidos → `None`; `leader = max(parties, key=lambda p: p['pct'])['name']`; `weights = part(..., 'weights')` en un `try` que devuelve `None` solo si `error.status == 404`; `leader not in weights['parties']` → `None`; `meta = part(..., 'meta')` para `fullname`/`color` (`next` sobre `meta['parties']` por `name`; `color`/`fullname` `None` si no está). `rows = sorted((r for r in weights['rows'] if r['name'] == leader), key=lambda r: r['date'])`. Actualizar el docstring del módulo (la página `promedio` lleva el ejemplo).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pages_ejemplo.py tests/test_pages.py -q`
Expected: todos PASS

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages/metodo.py tests/test_pages_ejemplo.py
git commit -m "Read the weights of the leading party of the latest national run for the method page

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: el bloque «Un ejemplo» en la plantilla

**Files:**
- Modify: `web/templates/metodo_promedio.html:13-20` (índice), `:202-203` (entre la tarjeta `banda` y la tarjeta `voto-estimado`)
- Modify: `web/dist/css/site.css:734-738` (`.cell-bar .fill.w`)
- Test: `tests/test_pages_ejemplo.py`

**Interfaces:**
- Consumes: `example`, `example_shown`, `example_hidden`, `example_csv` del contexto (tarea 5); filtros `date`, `date_short`, `pct`, `num`, `int`, `bar`; macros `m.card_start`, `m.card_end`, `m.how`.
- Produces: tarjeta `#ejemplo` con el párrafo de cifras, `table#weights-table` (clase `et`, dentro de `.table-wrap`, primera columna `class="f l"`), `p.scroll-hint`, `button#weights-more.more[data-batch]` (solo con `example_hidden`), `div.chart#weights-chart[role=img]`, «Cómo leerlo» y el enlace al CSV.

- [ ] **Step 1: Write the failing tests**

Añadir a `tests/test_pages_ejemplo.py`:

```python
def test_promedio_page_shows_the_example_block(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' in page and 'Un ejemplo: el promedio del PP hoy' in page
    assert 'href="#ejemplo"' in page and page.index('id="banda"') < page.index('id="ejemplo"') < page.index('id="voto-estimado"')
    assert 'id="weights-table"' in page and 'id="weights-chart"' in page and 'id="weights-more"' not in page
    assert len(re.findall(r'<tr>\s*<td class="f l">', page)) == 3 and '<tr hidden>' not in page
    text = re.sub(r'\s+', ' ', page)
    assert 'se leía en 33,1%, con una banda de 32,4% a 33,8%' in text and 'Entraban en la recta 3 sondeos' in text
    assert 'unos 2,6 sondeos de igual peso' in text and 'unos 42 días de anchura' in text and '8 oct 2026' in text
    assert '<b>GAD3</b> / ABC' in text and '24 sept' in text and '>–<' in page     # muestra ausente → guion
    assert 'class="fill w"' in page and 'width: 17.1%' in page                      # peso 0,512 sobre 3
    assert 'Descargar los pesos de todos los partidos (CSV)' in page and '/api/v1/forecast/es/weights?format=csv' in page
    data = initial(body)
    assert data['example']['name'] == 'PP' and len(data['example']['rows']) == 3 and 'fullname' not in data['example']


def test_promedio_page_hides_the_rows_beyond_the_batch_and_offers_more(api, monkeypatch):
    monkeypatch.setattr(metodo, 'EXAMPLE_SHOWN', 2)
    page = html(call(api, '/metodo/promedio')[2])
    assert page.count('<tr hidden>') == 1 and 'id="weights-more"' in page and 'data-batch="2"' in page


def test_promedio_page_without_the_part_renders_as_before(api, tmp_path):
    without_weights(tmp_path)
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' not in page and 'href="#ejemplo"' not in page
    assert 'initial-data' not in page and 'id="voto-estimado"' in page and 'id="banda"' in page
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pages_ejemplo.py -k promedio_page -v`
Expected: 2 FAIL (`'id="ejemplo"' in page`), el tercero PASS

- [ ] **Step 3: Write the block in `web/templates/metodo_promedio.html` and the bar colour in `site.css`**

- Índice: tras `<a href="#banda">La banda del promedio</a>`, `{% if example %}<a href="#ejemplo">Un ejemplo: el {{ example.name }} hoy</a>{% endif %}`.
- Entre `{{ m.card_end() }}` de `banda` y `m.card_start('voto-estimado', ...)`, dentro de `{% if example %}`:
  - `{{ m.card_start('ejemplo', 'Un ejemplo: el promedio del ' ~ example.name ~ ' hoy', 'Los sondeos que pesan en el promedio del ' ~ example.name ~ ' leído el ' ~ example.date|date ~ ' y la recta que ese día se ajusta a ellos: lo que las secciones anteriores explican, con los datos de hoy.') }}`
  - Párrafo: `El {{ example.date|date }} el promedio del {{ example.name }} se leía en {{ example.stats.mean|pct }}, con una banda de {{ example.stats.lo|pct }} a {{ example.stats.hi|pct }}. Entraban en la recta {{ example.stats.nobs|int }} sondeos, que por sus pesos equivalen a unos {{ example.stats.neff|num(1) }} sondeos de igual peso, y la campana que los pesaba por su cercanía tenía unos {{ example.stats.bandwidth|num(0) }} días de anchura a cada lado.`
  - Tabla en `<div class="table-wrap"><table class="et" id="weights-table">` con `<caption>` («Sondeos que entran en la recta del {{ example.date|date }} y sus pesos»), cabeceras `<th scope="col" class="f l">Fecha</th>`, `Encuestadora`, `Muestra` (`title="Entrevistas del sondeo"`), `{{ example.name }}` (`title="Voto publicado"`), `Peso` (`title="Peso total en la recta de ese día"`), `Cercanía` (`title="Peso por cercanía al día: la campana del kernel"`), `Sobrerrep.` (`title="Factor de sobrerrepresentación de la encuestadora"`), `F. muestra` (`title="Factor de muestra, con la dispersión de la encuestadora dentro"`), `Rating` (`title="Factor del rating de la encuestadora"`). Filas `{% for row in example.rows %}<tr{% if loop.index > example_shown %} hidden{% endif %}>`: `<td class="f l">{{ row.date|date_short }}</td>`, `<td><b>{{ row.pollster }}</b>{% if row.sponsor %} / {{ row.sponsor }}{% endif %}</td>`, `<td>{{ row.sample_size|int }}</td>`, `<td>{{ row.pct|pct }}</td>` y cinco celdas `<td class="cell-bar"><i class="fill w" style="left: 0; width: {{ row.weight|bar(0, example.weight_max) }}%"></i><span>{{ row.weight|num(2) }}</span></td>` con escalas `weight_kernel` 0–1, `weight_over` 0–1, `weight_sample` 0–2, `weight_rating` 0–2.
  - `<p class="scroll-hint">Desliza para ver más →</p>`; `{% if example_hidden %}<button type="button" class="more" id="weights-more" data-batch="{{ example_shown }}" hidden>Cargar más</button>{% endif %}`.
  - `<div class="chart" id="weights-chart" role="img" aria-label="Sondeos que pesan en el promedio del {{ example.name }} el {{ example.date|date }}, con la recta local del día, su banda y la campana del kernel"></div>`
  - `{% call m.how() %}` con cuatro párrafos: (1) qué es cada columna: la fecha de fin de campo, la encuestadora y el medio, la muestra, el voto publicado, el peso total y sus cuatro factores, los de `<a href="#peso">Cuánto pesa cada sondeo</a>` más la cercanía al día; el factor de muestra lleva dentro la dispersión de la encuestadora (el rebaño). (2) el gráfico: cada punto es un sondeo, más grande y más intenso cuanto más pesa; la recta es la de ese día, con su banda del 95 %, y la campana (eje de la derecha, de 0 a 1) es el peso por cercanía de `<a href="#regresion">La regresión local</a>`; las guías marcan el día y el valor leído. (3) la recta es solo la de ese día: cada día tiene la suya y el promedio publicado une sus valores; los puntos muestran la cifra publicada y la recta se ajusta a las cifras netas del efecto de encuestadora (`<a href="#efectos-de-casa">Los efectos de encuestadora</a>`), por lo que no pasa «por el medio» de los puntos. (4) la banda es la del promedio, con el error agrupado por encuestadora (`<a href="#banda">La banda del promedio</a>`); la tabla enseña los {{ example_shown }} primeros sondeos y «Cargar más» sigue.
  - `<p class="notice"><a href="{{ example_csv }}">Descargar los pesos de todos los partidos (CSV)</a></p>` y `{{ m.card_end() }}`.
- `site.css`, junto a `.cell-bar .fill.neg`: `.cell-bar .fill.w { background: rgba(29, 79, 134, 0.18); }`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pages_ejemplo.py tests/test_pages.py -q`
Expected: todos PASS (incluido `test_templates_compile_with_strict_undefined` y `test_metodo_promedio_text`)

- [ ] **Step 5: Commit**

```bash
git add web/templates/metodo_promedio.html web/dist/css/site.css tests/test_pages_ejemplo.py
git commit -m "Show the polls that weigh on the average of the leading party as a live example of the method

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: el gráfico `charts/weights.js` y su enganche en `metodo.js`

**Files:**
- Create: `web/dist/js/charts/weights.js`
- Modify: `web/dist/js/pages/metodo.js:1-6` (imports), `:24-30` (`paint`), `:160-161` (enganche)
- Modify: `tests/test_js_charts.py:19-25` (`WEIGHTS_JS`), `:30-45` (`run_js`)
- Test: `tests/test_js_charts.py`

**Interfaces:**
- Consumes: `baseOption`, `mountChart`, `THEME`, `tooltipStyle`, `axisLabelStyle` de `charts/base.js`; `bandSeries(name, times, lo, hi, {color, opacity, stack, z})` y `dayTime(iso)` de `charts/series.js`; `pollTooltipLines(poll, parties, colors)`, `meanTooltipLine(name, mean, lo, hi, date, color)`, `escapeText` de `polls-tooltip.js`; `fmtNum`, `fmtPct`, `fmtDateShort` de `format.js`; `readInitial`, `tryDraw`, `wireMore` de `pages/common.js`.
- Produces:
  - `weightsOption(example) -> object`: la opción ECharts. `example` es la isla (`name, color, date, stats, rows, line, kernel`). Series en este orden y con estos nombres: `Recta del día` (línea, color del partido, ancho 2, sin símbolos, z 3), las dos de `bandSeries('Banda 95 %', ...)` (color del partido, opacidad 0.15, stack `band:weights`), `Sondeos` (scatter, z 4, `data` de objetos `{value: [dayTime(date), pct, index], symbolSize: Math.round(6 + 14 * weight / wmax), itemStyle: {color, opacity: 0.25 + 0.75 * weight / wmax}}` con `wmax = max(weight)`), `Kernel` (línea en `yAxisIndex: 1`, color `THEME.mutedColor`, discontinua, ancho 1.5, sin símbolos, z 2), `marks` (serie vacía con `markLine` silenciosa, discontinua, `THEME.markColor`, sin símbolo, `label: {show: false}`, `data: [{xAxis: dayTime(date)}, {yAxis: stats.mean}]`). `xAxis` de tipo `time` con `min`/`max` = `dayTime` del primer y último `line.dates`, etiquetas `fmtDateShort(isoDay)` con `axisLabelStyle()`. `yAxis`: `[{type: 'value', scale: true, axisLabel: fmtPct(x, 0)}, {type: 'value', min: 0, max: 1, show: false}]`. `legend: {data: ['Sondeos', 'Recta del día', 'Banda 95 %', 'Kernel'], top: 0}`, `grid: {top: 40, left: 48, right: 16, bottom: 28, containLabel: false}`, `tooltip: {trigger: 'item', ...tooltipStyle(), formatter}`.
  - `pollLines(row, name) -> string[]`: las líneas del tooltip de un sondeo: `pollTooltipLines({name: row.sponsor ? `${row.pollster} / ${row.sponsor}` : row.pollster, date: row.date, sample_size: row.sample_size}, [], {})` (dos líneas) más `Publicado {fmtPct(pct)} · corregido {fmtPct(pct_adj)}`, `Peso {fmtNum(weight, 2)}` y `cercanía {k} · sobre. {o} · muestra {s} · rating {r}` (`fmtNum(x, 2)`), escapadas con `escapeText`.
  - `renderWeights(el, example)`: `mountChart(el, weightsOption(example))`.
  - El formatter del tooltip: `Sondeos` → `pollLines(row, name).join('<br>')`; `Recta del día` → `meanTooltipLine(name, mean[i], lo[i], hi[i], line.dates[i], color)`; `Kernel` → `Peso por cercanía {fmtNum(v, 2)}`; el resto `''`.
  - `metodo.js`: importa `renderWeights` y `wireMore`; `paint()` lee la isla una vez (`const data = readInitial()`), dibuja `#weights-chart` con `tryDraw(() => renderWeights(el, data.example), 'the weights example')` cuando existen el contenedor y `data.example`, y mantiene los gráficos del backtest con `const {horizons = [], mae = {}, coverage = {}, seats = {}} = data`; a nivel de módulo, `wireMore('weights-more', 'weights-table')`.

- [ ] **Step 1: Write the failing tests**

En `tests/test_js_charts.py`: `WEIGHTS_JS = pathlib.Path(DIST, 'js', 'charts', 'weights.js').as_uri()` junto a las demás rutas; en `run_js`, añadir `const w = await import("{}"); ` tras `const r = await import("{}"); ` y `WEIGHTS_JS` tras `RIDGE_JS` en `.format(...)`. Al final del fichero:

```python
EXAMPLE = {
    'name': 'PP', 'color': '#1d84ce', 'date': '2026-10-08',
    'stats': {'date': '2026-10-08', 'mean': 33.1, 'lo': 32.4, 'hi': 33.8, 'err': 0.35, 'nobs': 2, 'neff': 1.8, 'bandwidth': 42.0},
    'rows': [
        {'date': '2026-09-24', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1200, 'pct': 32.4, 'pct_adj': 32.9,
         'weight': 0.5, 'weight_kernel': 0.64, 'weight_over': 1.0, 'weight_sample': 0.8, 'weight_rating': 1.0},
        {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'sponsor': None, 'sample_size': None, 'pct': 33.1, 'pct_adj': 33.0,
         'weight': 1.0, 'weight_kernel': 0.98, 'weight_over': 1.0, 'weight_sample': 1.0, 'weight_rating': 1.0},
    ],
    'line': {'dates': ['2026-09-24', '2026-10-01', '2026-10-08'], 'mean': [32.8, 32.95, 33.1],
             'lo': [32.1, 32.25, 32.4], 'hi': [33.5, 33.65, 33.8]},
    'kernel': {'dates': ['2026-09-24', '2026-10-01', '2026-10-08'], 'values': [0.946, 0.986, 1.0]},
}


def test_weights_option_sizes_the_polls_by_weight_and_draws_the_line_the_band_the_kernel_and_the_guides():
    option = run_js('w.weightsOption(%s)' % json.dumps(EXAMPLE))
    assert [s['name'] for s in option['series']] == ['Recta del día', 'Banda 95 %', 'Banda 95 %', 'Sondeos', 'Kernel', 'marks']
    line, band, points, kernel, marks = option['series'][0], option['series'][2], option['series'][3], option['series'][4], option['series'][5]
    assert line['color'] == '#1d84ce' and line['lineStyle']['width'] == 2 and line['data'][2][1] == 33.1
    assert band['areaStyle'] == {'color': '#1d84ce', 'opacity': 0.15} and band['stack'] == 'band:weights'
    assert [p['symbolSize'] for p in points['data']] == [13, 20]
    assert [p['itemStyle']['opacity'] for p in points['data']] == [0.625, 1] and points['data'][1]['value'][1:] == [33.1, 1]
    assert kernel['yAxisIndex'] == 1 and kernel['lineStyle']['type'] == 'dashed' and kernel['data'][2][1] == 1.0
    assert option['yAxis'][1]['max'] == 1 and option['yAxis'][1]['show'] is False and option['yAxis'][0]['scale'] is True
    day = run_js('s.dayTime("2026-10-08")')
    assert marks['markLine']['data'] == [{'xAxis': day}, {'yAxis': 33.1}] and marks['markLine']['label'] == {'show': False}
    assert option['xAxis']['min'] == run_js('s.dayTime("2026-09-24")') and option['xAxis']['max'] == day
    assert option['legend']['data'] == ['Sondeos', 'Recta del día', 'Banda 95 %', 'Kernel'] and option['tooltip']['trigger'] == 'item'


def test_weights_poll_lines_name_the_sponsor_and_the_weight_factors():
    lines = run_js('w.pollLines(%s, "PP")' % json.dumps(EXAMPLE['rows'][0]))
    assert lines[0] == '<strong>GAD3 / ABC</strong>' and lines[1].endswith('· muestra 1.200')
    assert lines[2:] == ['Publicado 32,4% · corregido 32,9%', 'Peso 0,50', 'cercanía 0,64 · sobre. 1,00 · muestra 0,80 · rating 1,00']
    lines = run_js('w.pollLines(%s, "PP")' % json.dumps(EXAMPLE['rows'][1]))
    assert lines[0] == '<strong>Sigma Dos</strong>' and lines[1].endswith('· muestra –')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_js_charts.py -k weights -v`
Expected: 2 FAIL (`Cannot find module .../charts/weights.js`)

- [ ] **Step 3: Write `web/dist/js/charts/weights.js` and wire `metodo.js`**

Cabecera del módulo en el estilo de `series.js` (comentario de una línea con lo que dibuja). Exportar `pollLines`, `weightsOption` y `renderWeights`. El resto queda fijado por Interfaces.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_js_charts.py tests/test_pages_ejemplo.py -q`
Expected: todos PASS

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/weights.js web/dist/js/pages/metodo.js tests/test_js_charts.py
git commit -m "Draw the weights example: polls sized by weight, the day's line and band and the kernel bell

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: comprobación de conjunto, notebook y cierre de la spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-web-fase-9-ejemplo-promedio-design.md:3-6` (Estado)
- Verify: `notebooks/lab/PollsOptimize.ipynb` (celdas 7 y 8, sin cambios)

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: todo en verde; 759 + 15 tests nuevos aproximadamente (3 + 1 forecaster, 1 web api, 2 + 1 publish, 5 + 3 páginas, 2 JS). Sin `FutureWarning`.

- [ ] **Step 2: Render the page with the fixture and look at it**

Run (desde la raíz, con el arnés de páginas): `pytest tests/test_pages_ejemplo.py -k shows_the_example -q` ya cubre el HTML. Para verlo en el navegador, `gunicorn` local de Luis en 8000 apunta a su paquete real, que no trae la parte: el bloque no sale hasta que publique un run (comportamiento esperado, Review Focus 1). Dejarlo anotado en el informe final.

- [ ] **Step 3: Check the notebook cells against the local database (if it answers)**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from mtpy import mtpy; from mtpy.lib.forecaster import Forecaster
mtpy.run()
fc = Forecaster(scope='es', event_date='2026-11-29', drange=6, bmap='max', verbose=0, path='.').build_series()
print(fc.print_weights('PP', show=False).head()); fc.plot_weights('PP', show=False, path=None)
print('ok')"
```
Expected: la tabla con las diez columnas y `ok`. Si la base no responde (`QueryCanceled` u otro error de conexión, ver `.env`), anotarlo en el informe para que Luis ejecute las celdas 7 y 8 él mismo; no es un fallo del código.

- [ ] **Step 4: Close the spec**

En la cabecera de la spec, el Estado pasa a: `Estado: cerrada el 2026-10-15 en dev (commits de la fase 9; N tests). Pendiente de Luis: publicar un run (publish forecast) para que el bloque aparezca con datos reales.`

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-10-15-web-fase-9-ejemplo-promedio-design.md docs/superpowers/plans/2026-10-15-web-fase-9-ejemplo-promedio.md
git commit -m "Close the spec of the live example of the average and add its plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
