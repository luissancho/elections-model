# Web, fase 3b: relleno retroactivo de publicaciones y ciclo completo en el promedio

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el job de publicación pueda producir runs "a fecha de" un día pasado (solo con los sondeos
publicados hasta ese día), marcados como retrospectivos, para que la evolución de las publicaciones de
la portada y de `/escanos` tenga puntos desde la convocatoria de las elecciones (5 de octubre de 2026);
que esos puntos se distingan visualmente; y que `/promedio` muestre todo el ciclo por defecto.

**Architecture:** `Simulator` ya acepta `limit_date` (solo entran los sondeos publicados hasta esa fecha y
`as_of` se lee ese día; lo usa el backtest). `publish_forecast` lo expone junto a `run_at`, y anota
`meta.limit_date` y `headline.backfill`; `Publish` acepta `backfill: {from, to}` y publica un run por día
con `run_id` `YYYYMMDD-120000`, saltando los días ya publicados; el `latest` del manifest pasa a ser el
run más nuevo del `history` (nunca retrocede). El gráfico de evolución dibuja huecos los puntos con
`backfill`. No cambia la API, nginx ni la imagen Docker; el contrato sigue en 1 (claves adicionales).

**Tech Stack:** Python 3.11, pytest 9 (dobles `tests/fakes.py`); módulos ES + ECharts 5.6.0; Node 25 para
`node --check` y un renderizado SSR.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Paquete publicado", "Comando de
publicar", "Frontend"; la historia "no regenerable" se matiza: un run retrospectivo es una reconstrucción
marcada) y `docs/web/contrato.md` (`meta`, `headline`, `history`). Diseño aprobado por Luis en chat el
2026-10-09.

## Decisiones

- **D1.** Corte por `limit_date` = fecha de publicación del sondeo (columna `date`), no de carga en la
  base: un run al 5 de octubre usa lo que era público ese día. Ratings, deriva y catálogo son los de hoy;
  los efectos de casa se reajustan con esos sondeos (como en el backtest).
- **D2.** `run_id` retroactivo = `YYYYMMDD-120000` y `run_at` = `YYYY-MM-DDT12:00:00Z`: ordena antes que
  cualquier run real del mismo día posterior a las 12 UTC y el eje temporal lo sitúa en su día.
- **D3.** Un día cuyo `run_id` ya existe se salta (`skipped`, "run exists"): repetir el relleno es
  idempotente. Un run retroactivo nunca mueve `latest` hacia atrás: tras reconstruir el `history`, la
  entrada del manifest sale del último run del `history` (también en las publicaciones normales).
- **D4.** `headline.backfill` (bool) y `meta.limit_date` (str o null) son claves nuevas obligatorias en
  los esquemas; el JS trata la ausencia como `false` (runs publicados antes de este cambio).
- **D5.** Puntos retrospectivos: símbolo hueco (relleno blanco, borde del color del partido) y en el
  tooltip "estimación retrospectiva". Las tarjetas se titulan "Evolución de las publicaciones" con una
  nota: "Los puntos huecos son estimaciones retrospectivas: el modelo de hoy con los sondeos publicados
  hasta ese día."
- **D6.** `/promedio` abre con todo el ciclo (desde el primer sondeo hasta la elección); el deslizador
  queda para acercar. Se elimina la ventana de 180 días.
- **D7.** Fuera de alcance: una tarjeta de serie de sondeos en la portada; rellenar antes de la
  convocatoria (Luis decide el rango al lanzar el job); ejecutar el relleno real (lo lanza Luis contra la
  RDS).

## Global Constraints

- PEP8 y docstring numpydoc en inglés en `mtpy/`; español en `tests/`, `docs/`, `deploy/*.md`; JS con
  módulos ES, identificadores en inglés, sin `innerHTML` con datos.
- `python -m pytest -m "not integration" -q` en verde (hoy 349); `FutureWarning` es error.
- Nunca `python job.py`; nunca `mtpy.run()`/uvicorn sin `S3_BUCKET=` delante. Los tests usan los dobles.
- Árbol limpio (rama `dev`, `c0c7225`); `git add <ruta>` explícito; mensajes cortos en inglés con el
  trailer `Co-Authored-By` del modelo que escribe el commit.
- Contrato 1: los runs son inmutables; `history` ascendente por `run_id`; `headline.json` se escribe el
  último. Fixtures de `tests/fixtures/bundle/` validan contra `SCHEMAS` (`test_bundle_unit.py`).

## Review Focus

1. Relleno repetido: los días ya publicados quedan `skipped` y el `history`/`manifest` no cambian
   (Tarea 1).
2. `latest` nunca retrocede: un run normal de hoy seguido de un relleno de días anteriores deja `latest`
   en el de hoy; un relleno sin run normal previo deja `latest` en el día más reciente del relleno
   (Tarea 1).
3. Rango inválido (`from` > `to`, fecha no ISO, `to` posterior a hoy) → `ValueError` antes de tocar el
   paquete (Tarea 1).
4. Un run antiguo sin la clave `backfill` en el `history` se dibuja como punto normal (Tarea 2).
5. `/promedio` con un solo día de serie (ciclo recién empezado) sigue dibujando sin ventana (Tarea 2).

---

### Task 1: `limit_date`/`run_at` en `publish_forecast`, claves nuevas del contrato y `backfill` en el job

**Files:**
- Modify: `mtpy/lib/publish.py` (`export_meta`, `export_headline`, `publish_forecast`), `mtpy/lib/bundle.py`
  (`SCHEMAS`), `mtpy/jobs/Publish.py`, `tests/fixtures/bundle/meta.json`, `tests/fixtures/bundle/headline.json`,
  `tests/fixtures/bundle/history.json`
- Test: `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`, `tests/test_bundle_unit.py`

**Interfaces:**
- Consumes: `Simulator(..., limit_date=...)` (existente), `FakeSimulator` (anota los `kwargs` de la
  construcción), `bundle.iso_utc`, `BundleWriter.begin_run` (`FileExistsError` si el run existe),
  `publish.rebuild_history`, `publish.latest_entry`, `publish.loreg_guard`.
- Produces:
  - `export_meta(..., freeze: bool, limit_date: Optional[str] = None) -> dict`: añade `'limit_date':
    limit_date`.
  - `export_headline(sim, run_id, run_at, nowcast, forecast, backfill: bool = False) -> dict`: añade
    `'backfill': bool(backfill)`.
  - `bundle.SCHEMAS['meta']['limit_date'] = NULLABLE_STR`; `bundle.SCHEMAS['headline']['backfill'] = bool`.
  - `publish_forecast(..., limit_date: Optional[str] = None, run_at: Optional[str] = None)`: con
    `limit_date`, el simulador se construye con `limit_date=limit_date`; `run_at` sustituye a
    `bundle.iso_utc(writer.now())` cuando se da; `meta.limit_date` y `headline.backfill = limit_date is
    not None`.
  - `publish.backfill_days(spec: dict, today: Optional[str] = None) -> list[str]`: valida `{'from',
    'to'}` (`to` opcional, por defecto `from`; ambos `YYYY-MM-DD` ISO; `from <= to`; `to` no posterior a
    `today`, hoy en Europe/Madrid por defecto, como `loreg_guard`) y devuelve los días ISO ascendentes;
    `ValueError` con mensaje `'publish: backfill ...'` si no cumple.
  - `publish.backfill_run_id(day: str) -> str` = `day.replace('-', '') + '-120000'`;
    `publish.backfill_run_at(day: str) -> str` = `day + 'T12:00:00Z'`.
  - `Publish.run(..., backfill=None, ...)`: con `backfill`, la acción `forecast` recorre por ámbito los
    días de `backfill_days(backfill, today)` y llama a `_forecast_scope` con `run_id=backfill_run_id(day)`
    y los parámetros `limit_date=day, run_at=backfill_run_at(day)`; un `FileExistsError` de
    `publish_forecast` → `{'status': 'skipped', 'reason': 'run exists'}`; `results[scope]` es la lista de
    resultados por día (`[{'day', 'status', ...}]`) y la línea de resumen por día es
    `'{scope}: published {run_id} (as_of ..., N polls, backfill {day}) in X s'` o
    `'{scope}: skipped {day} (run exists)'`.
  - Manifest: tanto en publicación normal como en relleno, tras `rebuild_history` la entrada del ámbito
    es `publish.latest_entry(history['runs'][-1])` (el run más nuevo), no la del run recién producido.
    `dry_run` sigue sin escribir punteros.

- [ ] **Step 1: Tests que fallan**

En `tests/test_publish_unit.py` (junto a `test_export_meta_and_headline` y a los de `publish_forecast`,
reutilizando `make_writer`, `FakeSimulator`, `fake_stats`, `fake_prov` del fichero):

```python
def test_export_meta_and_headline_carry_the_backfill_keys():
    sim = synthetic_simulator()
    meta = publish.export_meta(sim, RUN, '2026-10-05T12:00:00Z', 50, 10, publish.DEFAULT_CORRECTORS, {}, {}, 0, None,
                               {'commit': None, 'dirty': False, 'versions': {}}, False, limit_date='2026-10-05')
    assert meta['limit_date'] == '2026-10-05' and meta['run_at'] == '2026-10-05T12:00:00Z'
    head = publish.export_headline(sim, RUN, '2026-10-05T12:00:00Z', {}, {}, backfill=True)
    assert head['backfill'] is True
    assert publish.export_headline(sim, RUN, '2026-10-05T12:00:00Z', {}, {})['backfill'] is False
    assert bundle.validate(bundle.envelope('headline', head, 'es', run_id=RUN), 'headline')  # claves nuevas en el esquema


def test_publish_forecast_backfill_freezes_the_polls_and_stamps_the_day(tmp_path):
    writer, factory = make_writer(tmp_path), FakeSimulator()
    out = publish.publish_forecast('es', writer, '20261005-120000', '2026-11-29', n_sim=50, simulator=factory,
                                   stats=fake_stats, prov=fake_prov, limit_date='2026-10-05', run_at='2026-10-05T12:00:00Z')
    assert factory.calls[0][3]['limit_date'] == '2026-10-05'
    meta = writer.read_json(bundle.path_part('es', '20261005-120000', 'meta'))['data']
    head = writer.read_json(bundle.path_part('es', '20261005-120000', 'headline'))['data']
    assert meta['limit_date'] == '2026-10-05' and meta['run_at'] == '2026-10-05T12:00:00Z'
    assert head['backfill'] is True and out['entry']['run_at'] == '2026-10-05T12:00:00Z'
    normal = publish.publish_forecast('es', writer, RUN, '2026-11-29', n_sim=50, simulator=factory, stats=fake_stats, prov=fake_prov)
    assert 'limit_date' not in factory.calls[-1][3]
    assert writer.read_json(bundle.path_part('es', RUN, 'headline'))['data']['backfill'] is False
    assert writer.read_json(bundle.path_part('es', RUN, 'meta'))['data']['limit_date'] is None


@pytest.mark.parametrize('spec, days', [
    ({'from': '2026-10-05', 'to': '2026-10-07'}, ['2026-10-05', '2026-10-06', '2026-10-07']),
    ({'from': '2026-10-05'}, ['2026-10-05']),
])
def test_backfill_days_expands_the_range(spec, days):
    assert publish.backfill_days(spec, today='2026-10-09') == days
    assert publish.backfill_run_id('2026-10-05') == '20261005-120000'
    assert publish.backfill_run_at('2026-10-05') == '2026-10-05T12:00:00Z'


@pytest.mark.parametrize('spec', [
    {'from': '2026-10-07', 'to': '2026-10-05'}, {'from': '05/10/2026'}, {'to': '2026-10-05'}, 'ayer',
    {'from': '2026-10-05', 'to': '2026-10-10'},
])
def test_backfill_days_rejects_bad_ranges(spec):
    with pytest.raises(ValueError, match='publish: backfill'):
        publish.backfill_days(spec, today='2026-10-09')
```

En `tests/test_jobs_publish.py` (con las fixtures `fresh_app`, `patched` y el patrón de los tests
existentes; `patched` inyecta `FakeSimulator`, `db_stats` y `provenance`):

```python
def test_backfill_publishes_one_run_per_day_and_never_moves_latest_back(fresh_app, tmp_path, patched, capsys):
    fs = FileSystem(str(tmp_path))
    writer = bundle.BundleWriter(fs)
    Publish().run(what=['forecast'], scopes=['es'], fs=fs)
    today_run = publish.read_manifest(writer)['scopes']['es']['latest']
    result = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert [r['status'] for r in result['es']] == ['published', 'published']
    assert writer.list_runs('es') == ['20261005-120000', '20261006-120000', today_run]
    history = writer.read_json(bundle.path_history('es'))['data']['runs']
    assert [r['run_id'] for r in history] == ['20261005-120000', '20261006-120000', today_run]
    assert [r['backfill'] for r in history] == [True, True, False]
    assert history[0]['run_at'] == '2026-10-05T12:00:00Z'
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == today_run
    out = capsys.readouterr().out
    assert 'es: published 20261005-120000' in out and 'backfill 2026-10-05' in out
    again = Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert [r['status'] for r in again['es']] == ['skipped', 'skipped'] and 'run exists' in capsys.readouterr().out
    assert publish.read_manifest(writer)['scopes']['es']['latest'] == today_run


def test_backfill_alone_points_latest_to_its_newest_day(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-05', 'to': '2026-10-06'}, today='2026-10-09')
    assert publish.read_manifest(bundle.BundleWriter(fs))['scopes']['es']['latest'] == '20261006-120000'


def test_backfill_range_errors_fail_before_touching_the_bundle(fresh_app, tmp_path, patched):
    fs = FileSystem(str(tmp_path))
    with pytest.raises(ValueError, match='publish: backfill'):
        Publish().run(what=['forecast'], scopes=['es'], fs=fs, backfill={'from': '2026-10-07', 'to': '2026-10-05'})
    assert not fs.exists(bundle.path_manifest())
```

`tests/test_bundle_unit.py`: el test que valida los fixtures sigue pasando una vez añadidas las claves
(`meta.json`: `"limit_date": null`; `headline.json`: `"backfill": false`; `history.json`: `"backfill":
false` en cada run), y un test nuevo comprueba que un `headline` sin `backfill` ya no valida:

```python
def test_headline_requires_the_backfill_flag():
    data = fixture_data('headline')
    del data['backfill']
    with pytest.raises(ValueError, match='missing keys'):
        bundle.validate(bundle.envelope('headline', data, 'es', run_id=RUN), 'headline')
```

(`fixture_data` y `RUN` como los usa ya ese fichero; si no existen allí, importarlos de
`tests.test_webapi_unit`.)

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py -q -k "backfill or carry_the"`
Expected: FAIL (`TypeError` por argumentos desconocidos, `AttributeError` de `backfill_days`, claves
ausentes).

- [ ] **Step 3: Implementar según Interfaces**

En el job, extraer el bucle actual de ámbitos a un método que reciba `(scope, run_id, extra_params)` y
recorrer días cuando hay `backfill`; `_line` admite el sufijo `backfill {day}` y el estado `skipped` con
`day`. La entrada del manifest se calcula desde el `history` reconstruido.

- [ ] **Step 4: Tests en verde**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py tests/test_pages.py tests/test_web_api.py -q && python -m pytest -m "not integration" -q`
Expected: PASS (los tests de páginas y API leen los fixtures con las claves nuevas).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish.py mtpy/lib/bundle.py mtpy/jobs/Publish.py tests/fixtures/bundle/meta.json tests/fixtures/bundle/headline.json tests/fixtures/bundle/history.json tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_bundle_unit.py
git commit -m "Publish retrospective runs with a poll cutoff date and keep the manifest on the newest run"
```

---

### Task 2: Puntos retrospectivos en la evolución y ciclo completo en `/promedio`

**Files:**
- Modify: `web/dist/js/charts/evolution.js`, `web/dist/js/charts/series.js`, `web/dist/js/pages/promedio.js`,
  `web/templates/index.html`, `web/templates/escanos.html`, `web/dist/css/site.css` (si hace falta)
- Test: `tests/test_pages.py` (textos), `tests/test_web_routes.py` (estáticos, sin cambios), renderizado
  SSR en el scratchpad

**Interfaces:**
- Consumes: `history.runs[].backfill` (ausente en runs antiguos → `false`), `history.runs[].run_at`.
- Produces:
  - `evolutionSeries(runs, mode, parties, field = 'pct')`: cada punto es `[run_at, value, backfill]`
    con `backfill = Boolean(run.backfill)`.
  - `renderEvolution`: los puntos con `backfill` se emiten como `{value: [t, v, true], itemStyle:
    {color: '#fff', borderColor: <color del partido>, borderWidth: 2}}` (símbolo hueco); el tooltip
    añade " · estimación retrospectiva" a la cabecera de la fecha cuando el primer punto del eje lo es.
  - `renderSeries`: desaparece la ventana inicial de 180 días (`WINDOW_DAYS`, `lastWindow`); el
    `dataZoom` deslizador cubre todo el eje (sin `startValue`/`endValue`); `promedio.js` deja de pasar
    `anchor`.
  - Plantillas: las dos tarjetas se titulan "Evolución de las publicaciones" (subtítulo actual) y llevan
    un `p.about`: "Un punto por cada publicación. Los puntos huecos son estimaciones retrospectivas: el
    modelo de hoy con los sondeos publicados hasta ese día."; `aria-label` "Evolución de las
    publicaciones: estimación (o escaños) de cada run".

- [ ] **Step 1: Tests que fallan en `tests/test_pages.py`**

```python
def test_evolution_cards_explain_the_retrospective_points(api):
    for path in ('/', '/escanos'):
        page = html(call(api, path)[2])
        assert 'Evolución de las publicaciones' in page and 'estimaciones retrospectivas' in page
```

- [ ] **Step 2: Comprobar que falla**

Run: `python -m pytest tests/test_pages.py -q -k retrospective`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Verificar**

Run: `python -m pytest tests/test_pages.py tests/test_web_routes.py -q && for f in web/dist/js/charts/evolution.js web/dist/js/charts/series.js web/dist/js/pages/promedio.js; do node --check "$f" || exit 1; done`
Expected: PASS y sin errores de sintaxis.

SSR (script en el scratchpad, patrón de la fase 3: `window.echarts` con `init` SSR en SVG,
`ResizeObserver` falso): `renderEvolution` con un `history` sintético de tres runs (dos con `backfill:
true`, uno sin la clave) → el SVG contiene tres círculos por partido y dos de ellos con `fill="#fff"`;
`renderSeries` con el run local (`files/site/v1/runs/es/20261008-181100`) → el `dataZoom` arranca en
`2023-08-07`; `renderSeries` con una serie de un solo día no lanza.

- [ ] **Step 5: Commit**

```bash
git add web/dist/js/charts/evolution.js web/dist/js/charts/series.js web/dist/js/pages/promedio.js web/templates/index.html web/templates/escanos.html tests/test_pages.py
git commit -m "Draw retrospective runs as hollow points and open the poll average on the whole cycle"
```

---

### Task 3: Documentación y enmienda de la spec

**Files:**
- Modify: `deploy/README.md` ("Publicar": parámetro `backfill`, ejemplo, estados y líneas de resumen),
  `docs/web/contrato.md` (`meta.limit_date`, `headline.backfill`, nota en `history.json`: ascendente por
  `run_id`, los retrospectivos quedan antes), `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`
  (enmienda fechada en "Paquete publicado"/"Comando de publicar" y una nota al final del "Estado al cierre
  de la fase 3": relleno retroactivo, D1-D7, commits)

- [ ] **Step 1: Escribir**

README, ejemplo exacto:

```
S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-05","to":"2026-10-08"}}'
set -a; . deploy/elections.env; set +a; python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-05","to":"2026-10-08"}}'
```

con la advertencia: un run retrospectivo es una reconstrucción (ratings, deriva y catálogo de hoy; solo
los sondeos publicados hasta ese día), `es` tarda unos 2 min por día, los días ya publicados se saltan y
`latest` no retrocede.

- [ ] **Step 2: Verificar y commit**

Run: `python -m pytest -m "not integration" -q`
Expected: PASS.

```bash
git add deploy/README.md docs/web/contrato.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md
git commit -m "Document the retrospective publish runs and the whole-cycle average chart"
```

## Pendiente de Luis tras el plan

- Lanzar el relleno real contra la RDS (`backfill` del 5 al 8 de octubre, o desde la fecha que decida) y
  comprobar en la portada que los puntos huecos aparecen desde la convocatoria.
- Recorrer `/promedio` con todo el ciclo: si tres años resultan densos, pedir el control "Últimos 6
  meses / Todo el ciclo".
