# Web de resultados, fase 2 (API `/api/v1` y sitio mínimo): plan de implementación

> **Estado:** completado el 2026-10-08 en `dev` (commits 207cd46..8e0a116 más el commit de documentación; humo Docker y recorrido en navegador pendientes de Luis). Ver el estado de cierre en la spec.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Servir el paquete publicado en la fase 1 por HTTP (`/api/v1`, solo GET, JSON y CSV, con caché y
cabeceras estables) y publicar un sitio mínimo de dos páginas (`/` portada y `/promedio`) que lo dibuja con
ECharts, listo para el primer despliegue con `es` en el servidor de Luis (nginx + gunicorn en su imagen
Docker actual).

**Architecture:** la capa de servicio `mtpy/lib/webapi.py` (validadores en listas cerradas, cachés TTL y
LRU por proceso, lector del paquete sobre `app.fs`, tabla `ROUTES`) no sabe de HTTP; los controladores
`mtpy/controllers/Base.py` y `Forecast.py` solo validan, llaman al servicio y devuelven bytes o dicts; la
API sirve los ficheros del paquete tal cual (sin reserializar) y resuelve `latest` por `manifest.json`.
El frontend `web/` es HTML + módulos ES sin build, con ECharts 5.6.0 vendorizado; nginx sirve `web/`
estático y hace de proxy con caché de `/api/`. Nada de esta fase toca el modelo ni el job `publish`,
salvo tres flecos aparcados por la revisión final de la fase 1 (Tarea 1).

**Tech Stack:** Python 3.11 (`python` del venv), mtpy (ASGI propio en `mtpy/core/api.py`), pandas 2.2.2,
pytest 9; HTML5, CSS, JavaScript ES2020 (módulos nativos), ECharts 5.6.0 (`echarts.min.js`, 1,0 MB,
Apache 2.0); nginx (imagen `python:3.11-slim` de Luis, nginx de Homebrew en local solo para `nginx -t`).

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md`, secciones "Contrato de API",
"Servicio y caché", "Frontend", "Infraestructura" y "Seguridad" (con las salvedades del 2026-10-08: imagen
Docker de Luis sin cambios, configuración única `deploy/elections.env`), "Pruebas", fase 2 de "Fases",
"Verificación" y "Estado al cierre de la fase 1". Contrato del paquete: `docs/web/contrato.md`.

## Global Constraints

- PEP8 y docstring en toda función, clase y método nuevos (AGENTS.md); docstrings numpydoc en inglés
  dentro de `mtpy/`; español en `tests/`, `docs/`, `deploy/*.md` y `deploy/*.sh`. JavaScript: módulos ES
  (`import`/`export`), `const`/`let`, sin dependencias salvo ECharts, comentarios en inglés, nombres en
  inglés; textos visibles para el usuario en español.
- `python -m pytest -m "not integration" -q` sin base de datos (hoy 262 tests); `pytest.ini` convierte
  `FutureWarning` en error. Los tests de la API usan el arnés ASGI de `tests/test_api_asgi.py`
  (`call`) y la fixture `fresh_app`; nunca arrancan `mtpy.run()`.
- Árbol limpio al empezar (rama `dev`, `ee16575`). Cada commit añade solo los ficheros de su tarea con
  `git add <ruta>`; nunca `-A`, `.` ni `-a`. Mensajes cortos en inglés con el trailer `Co-Authored-By`
  del modelo que escribe el commit.
- **Nunca ejecutar `python job.py`, `uvicorn api:api` ni ningún comando que arranque `mtpy.run()` sin
  `S3_BUCKET=` (vacío) delante**: el `.env` de la raíz apunta `app.fs` al bucket real y la base de datos a
  la RDS. Con `S3_BUCKET=` la API lee el paquete local `files/site/v1` (existe: run `20261008-181100`).
- Contrato de API fijado por la spec: `/api/v1`, solo GET; rutas fijas registradas antes que las
  genéricas (primera coincidencia gana); toda entrada pasa por listas cerradas antes de tocar `app.fs`
  (ámbito en `get_scopes()`, `run` con `^\d{8}-\d{6}$`, `part`/`mode`/`format` en tuplas); errores
  `{"status":"error","message"}` con 400 (entrada inválida), 404 (recurso inexistente), 503 (paquete sin
  publicar) y 500; `Cache-Control: public, max-age=60` en punteros y en ficheros de run resueltos por
  `latest`, `public, max-age=31536000, immutable` con `?run=` explícito, `no-store` en `/health` y en
  errores; `Access-Control-Allow-Origin: *` en toda respuesta de la API; CSV `text/csv; charset=utf-8` con
  `Content-Disposition: attachment; filename="..."`; `ETag` en todo contenido servido.
- La API sirve los ficheros del paquete **sin transformarlos** (bytes del JSON tal cual, sobre incluido);
  `?format=csv` sirve el gemelo `csv/` donde existe (`series`, `polls`, `fan`, `house-effects`,
  `dispersion` y los seis de modo); `meta`, `headline`, `runs`, `manifest` y `scopes` no tienen CSV (404).
- Caché por proceso en `webapi.py`: `manifest.json`, `history.json` y el catálogo de ámbitos con TTL
  `web.cache_ttl` (60 s por defecto); ficheros de run inmutables en LRU de 200 entradas sin caducidad.
  Configuración: sección `"web": {"prefix": "${WEB_PREFIX}", "cache_ttl": "${WEB_CACHE_TTL}"}` en
  `config.json`; valores vacíos → `site/v1` y 60. No hay `WEB_FREEZE`: el aviso editorial es
  `manifest.freeze` (decisión de este plan).
- Infraestructura: **solo cambia `deploy/docker/nginx.conf`** (más los scripts nuevos `deploy/smoke.sh`,
  `deploy/check-api.sh` y `deploy/nginx-check.sh`). `Dockerfile`, `supervisord.conf`, `init.sh`,
  `crontab`, `requirements.txt` y `.dockerignore` no se tocan (imagen de Luis). `COPY . .` ya incluye
  `web/`.
- Frontend: multipágina, estado en la query string (`?scope=es&mode=forecast&run=...`), sin router JS,
  sin inline scripts ni inline event handlers (CSP `script-src 'self'`), un `<script type="module">` por
  página, `fetch` solo a través de `js/api.js`, colores de partido desde `meta.parties`, una columna por
  debajo de 720 px, `chart.resize()` con `ResizeObserver`, `lang="es"`, `Intl` `es-ES`. ECharts
  vendorizado como `web/vendor/echarts-5.6.0.min.js` (versión en el nombre; caché inmutable en nginx) con
  `web/vendor/LICENSE-echarts.txt`.
- Decisiones de este plan que Luis puede cambiar: modo por defecto `forecast` (el notebook publica así);
  la portada muestra todos los partidos de `vote.rows` (el selector de agrupación es de la fase 3);
  `simulable` en `/scopes` = el ámbito tiene entrada en el manifest; orden del hemiciclo por bloque
  `Izquierda, Separatista, Regionalista, Derecha` (constante `BLOCK_ORDER` en `catalog.js`); cabecera
  `x-freeze: active` en las respuestas de la API cuando `manifest.freeze.active` es verdadero.
- Docker: el demonio no está arrancado en la máquina de ejecución; las tareas construyen y documentan
  `deploy/smoke.sh`, y lo ejecutan solo si `docker info` responde; si no, queda para Luis. La verificación
  visual en navegador (escritorio y móvil, sin errores de consola) es de Luis (spec, Verificación fase 2).

## Review Focus

1. `?run=` con mayúsculas, puntos, barras o longitud distinta (`2026-10-08`, `20261008-181100/..`) →
   400 antes de tocar `app.fs`; un `run` bien formado pero inexistente → 404 (tests en Tarea 3).
2. Paquete sin publicar (sin `manifest.json`) → 503 JSON con mensaje, nunca 500 con traza; ámbito del
   catálogo sin publicar (`es-md`) → 404; ámbito fuera del catálogo o con caracteres raros → 400 o el 404
   JSON del router (tests en Tarea 3).
3. Un fichero de run servido vía `latest` nunca lleva `immutable`; el mismo fichero con `?run=` sí; el
   CSV sigue la misma regla y los errores son `no-store` (tests en Tarea 3).
4. `history.json` con un solo run y `polls` con `null` en partidos: la portada dibuja la evolución con un
   punto y la tabla de sondeos muestra guiones, sin excepciones en consola (comprobación de Luis, Tarea 9;
   los datos de prueba del arnés replican el caso en `tests/test_web_routes.py` solo a nivel estático).
5. nginx: `location /api/` tiene prioridad sobre `/` (un `index.html` nunca responde a `/api/...`),
   `/healthz` responde 200 sin gunicorn, y `try_files $uri $uri.html $uri/ =404` sirve `/promedio` como
   `promedio.html` (comprobación con `deploy/nginx-check.sh` y `deploy/smoke.sh`, Tarea 7).

---

### Task 1: Flecos de la fase 1 aparcados por la revisión final

**Files:**
- Modify: `mtpy/lib/publish.py` (`_check_run`), `mtpy/jobs/Publish.py` (`run`)
- Test: `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Produces: `publish._check_run` usa `fullmatch` (un `run_id` o `scope` con salto de línea final se
  rechaza); `publish_forecast` convierte en `NothingToPublish` también el `ValueError` de `fit_forecast`
  (ya lo hace; se añade el test que faltaba); en el job, una excepción de `rebuild_history`/`update_manifest`
  al escribir el manifest no impide imprimir el resumen: se captura, se añade la línea `manifest: failed
  ({Type}: {msg})`, cuenta como fallo (`RuntimeError` final) y las líneas de los ámbitos se imprimen igual.

- [ ] **Step 1: Añadir los tests**

En `tests/test_publish_unit.py`:

```python
def test_check_run_rejects_a_trailing_newline(tmp_path):
    writer, _, _ = publish_es(tmp_path)
    for bad in ('20261008-120000\n', '20261008-120000 '):
        with pytest.raises(ValueError, match='invalid run id'):
            publish.point(writer, 'es', bad)
    with pytest.raises(ValueError, match='invalid scope'):
        publish.unpublish(writer, 'es\n', RUN)
    assert writer.exists(bundle.path_run('es', RUN))


def test_fit_forecast_value_error_is_nothing_to_publish(tmp_path):
    factory = FakeSimulator()

    def failing(scope, event_date, **kwargs):
        sim = factory(scope, event_date, **kwargs)

        def fit_forecast(**kw):
            raise ValueError('Not enough polls to fit the average of es 2026-11-29')
        sim.fit_forecast = fit_forecast
        return sim
    with pytest.raises(publish.NothingToPublish, match='Not enough polls'):
        publish.publish_forecast('es', make_writer(tmp_path), RUN, '2026-11-29', n_sim=5, simulator=failing,
                                 stats=fake_stats, prov=fake_prov)
```

En `tests/test_jobs_publish.py`:

```python
def test_manifest_write_failure_keeps_the_summary(fresh_app, tmp_path, patched, capsys, monkeypatch):
    from mtpy.jobs.Publish import Publish

    def boom(writer, scopes=None, freeze=None):
        raise OSError('s3 down')
    monkeypatch.setattr(publish, 'update_manifest', boom)
    with pytest.raises(RuntimeError, match='manifest'):
        Publish().run(what=['forecast'], scopes=['es'], fs=FileSystem(str(tmp_path)))
    out = capsys.readouterr().out
    assert 'es: published' in out and 'manifest: failed (OSError: s3 down)' in out
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py -q -k "trailing or nothing_to_publish or manifest_write"`
Expected: 2 FAIL (`'20261008-120000\n'` pasa la validación y llega a `not found`; `OSError` escapa sin
resumen) y 1 PASS (`fit_forecast` ya está dentro del `try`).

- [ ] **Step 3: Implementar**

`_check_run`: `RUN_ID_RE.fullmatch` y `ROUTE_ALIAS_RE.fullmatch`. En `Publish.run`, envolver la llamada
a `publish.update_manifest` (ambas: tras `forecast` y en la acción `manifest`) en `try/except Exception`
que añade `('manifest', None, reason)` a `failures`, registra la traza con el logger si existe y añade la
línea `manifest: failed (...)`; el `RuntimeError` final incluye `manifest` cuando falló.

- [ ] **Step 4: Ejecutar los tests y la suite**

Run: `python -m pytest -m "not integration" -q`
Expected: 265 passed.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish.py mtpy/jobs/Publish.py tests/test_publish_unit.py tests/test_jobs_publish.py
git commit -m "$(printf 'Tighten run id checks and report manifest write failures\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 2: `mtpy/lib/webapi.py`: validadores, cachés y lector del paquete

**Files:**
- Create: `mtpy/lib/webapi.py`, `tests/test_webapi_unit.py`
- Modify: `config.json` (sección `web`), `.env.example` (`WEB_PREFIX=`, `WEB_CACHE_TTL=`)

**Interfaces:**
- Consumes: `bundle.BundleReader(fs, prefix)` (`read_json`, `exists`, `path`), `bundle.loads`,
  `bundle.path_manifest/path_history/path_part/path_csv`, `bundle.RUN_ID_RE`, `bundle.ROUTE_ALIAS_RE`,
  `bundle.MODES`, `bundle.RUN_PARTS`, `bundle.MODE_PARTS`, `bundle.PREFIX`, `bundle.CONTRACT`; el
  catálogo de ámbitos `data/es-scopes.csv` leído con `App.get_().data.read_csv('es-scopes.csv').set_index('scode')`
  (lo mismo que hace `mtpy.lib.data.get_scopes()`, replicado aquí a propósito: importar `mtpy.lib.data`
  arrastra los modelos, scipy y statsmodels a cada worker de gunicorn, y la web de la fase 2 no usa la
  base); `HttpError` de `mtpy/core/api.py`; `App.get_()` (`fs`, `data`, `config`). `webapi.py` no importa
  `mtpy.lib.data`, `mtpy.lib.publish` ni `mtpy.lib.simulator`.
- Produces (constantes): `DEFAULT_TTL = 60`, `LRU_SIZE = 200`, `IMMUTABLE_MAX_AGE = 31536000`,
  `FORMATS = ('json', 'csv')`, `CSV_PARTS = bundle.RUN_PARTS[2:]` (los cinco con gemelo CSV),
  `MODE_PARTS = bundle.MODE_PARTS`, `RUN_PARTS = bundle.RUN_PARTS`, `MODES = bundle.MODES`.
- Produces: `settings() -> dict` con `prefix` y `cache_ttl` leídos de `App.get_().config.web` cuando
  existe la sección (`getattr(config, 'web', None)`; `config` puede ser `None` en tests); valores vacíos →
  `bundle.PREFIX` y `DEFAULT_TTL`; `cache_ttl` se convierte con `float`.
- Produces: `class TTLCache` (`__init__(self, ttl: float, clock: Callable[[], float] = time.monotonic)`;
  `get(self, key, loader: Callable[[], Any]) -> Any` devuelve el valor cacheado si `clock() - stored <
  ttl`, si no llama a `loader()`, guarda y devuelve; una excepción del `loader` se propaga sin guardar
  nada; `clear()`); `class LRUCache` (`__init__(self, maxsize: int = LRU_SIZE)`; `get(key, loader)` con
  `OrderedDict`: acierto → `move_to_end`; fallo → `loader()`, inserta y expulsa el más antiguo si supera
  `maxsize`; `__len__`).
- Produces (validadores; todos devuelven el valor limpio o lanzan `HttpError`): `check_scope(scope: str)
  -> str` (400 `'invalid scope'` si no casa `ROUTE_ALIAS_RE.fullmatch` o no está en el índice del catálogo
  `scope_codes()`); `check_run(run: Optional[str]) -> Optional[str]` (`None`/`''` → `None`; 400 `'invalid
  run'` si no casa `RUN_ID_RE.fullmatch`); `check_mode(mode: str) -> str` (400 `'invalid mode'` si no está
  en `MODES`); `check_part(part: str, allowed: Sequence[str]) -> str` (404 `'unknown part'`);
  `check_format(fmt: Optional[str]) -> str` (`None`/`''` → `'json'`; 400 `'invalid format'`).
  `catalogue() -> pd.DataFrame` (el CSV indexado por `scode`, cacheado en un `TTLCache` de módulo con el
  TTL de `settings()`) y `scope_codes() -> list[str]` (`catalogue().index.tolist()`).
- Produces: `class Site` (servicio sobre el paquete; una instancia por proceso, guardada en la `App`):

```python
class Site:
    def __init__(self, reader: BundleReader, ttl: float = DEFAULT_TTL, lru: int = LRU_SIZE,
                 clock: Callable[[], float] = time.monotonic): ...
    def manifest(self) -> bytes          # TTL; FileNotFoundError → HttpError(503, 'no bundle published yet')
    def manifest_data(self) -> dict      # bundle.loads(self.manifest())['data']
    def freeze(self) -> dict             # manifest_data()['freeze']
    def latest_run(self, scope: str) -> str   # manifest scopes[scope]['latest']; ausente → HttpError(404, 'scope not published')
    def history(self, scope: str) -> bytes    # TTL; FileNotFoundError → HttpError(404, 'no runs published')
    def run_file(self, scope: str, run_id: str, part: str, mode: Optional[str] = None) -> bytes  # LRU; FileNotFoundError → HttpError(404, 'run not found')
    def run_csv(self, scope: str, run_id: str, name: str, mode: Optional[str] = None) -> bytes   # LRU; ídem
    def scopes(self) -> dict             # ver abajo; catálogo + manifest
```

  `scopes()` devuelve `{'contract': bundle.CONTRACT, 'scopes': [...]}` con una fila por ámbito del
  catálogo en su orden: `{'code', 'name', 'parent' (str|None), 'seats' (int), 'simulable' (bool: tiene
  entrada en el manifest), 'latest' (run_id|None), 'run_at', 'event_date', 'as_of', 'date_last',
  'n_polls'}` (los seis últimos `None` sin entrada). Lee bytes con `reader.fs.read_bytes(reader.path(name))`
  (sin reserializar); los `FileNotFoundError` se traducen a `HttpError` donde se indica y cualquier otra
  excepción se propaga (500 por `Controller.dispatch`).
- Produces: `site() -> Site`: devuelve `App.get_().site` si existe, si no crea
  `Site(BundleReader(app.fs, prefix=settings()['prefix']), ttl=settings()['cache_ttl'])`, lo guarda con
  `app.set('site', ...)` y lo devuelve; `app.fs` `None` → `HttpError(503, 'no file system configured')`.
  (`Site` no define `__call__`: `App.set` guardaría el resultado de llamarlo.)

- [ ] **Step 1: Crear `tests/test_webapi_unit.py`**

```python
"""Tests de `mtpy/lib/webapi.py` sin servidor ni base de datos: validadores, cachés y lector del paquete."""
import json
import os

import pandas as pd
import pytest

from mtpy.core.api import HttpError
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'bundle')
RUN = '20261008-120000'


def fixture_data(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)['data']


def write_bundle(fs, scope='es', run_id=RUN, manifest=True):
    """Paquete mínimo a partir de los fixtures: un run completo, su history y el manifest."""
    writer = bundle.BundleWriter(fs)
    for part in bundle.RUN_PARTS:
        writer.write_json(bundle.path_part(scope, run_id, part), part, fixture_data(part), scope, run_id=run_id)
    for mode in bundle.MODES:
        for part in bundle.MODE_PARTS:
            writer.write_json(bundle.path_part(scope, run_id, part, mode), part, fixture_data(part), scope,
                              run_id=run_id, mode=mode)
    writer.write_csv(bundle.path_csv(scope, run_id, 'series'), pd.DataFrame({'date': ['2026-10-05'], 'name': ['PP']}))
    writer.write_csv(bundle.path_csv(scope, run_id, 'vote', 'forecast'), pd.DataFrame({'name': ['PP'], 'pct': [40.0]}))
    publish.rebuild_history(writer, scope)
    if manifest:
        publish.update_manifest(writer, {scope: publish.latest_entry(fixture_data('headline') | {'run_id': run_id})})
    return writer


@pytest.fixture
def site_app(fresh_app, tmp_path):
    """`App` con `fs` local, el catálogo de `data/` y un paquete mínimo publicado para `es`."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return fresh_app


def test_validators_accept_the_catalogue_and_reject_the_rest(site_app):
    assert webapi.check_scope('es') == 'es' and webapi.check_scope('es-md') == 'es-md'
    for bad in ("es'", 'es;', 'es-xx', 'ES', 'es/..', ''):
        with pytest.raises(HttpError) as err:
            webapi.check_scope(bad)
        assert err.value.status == 400
    assert webapi.check_run(None) is None and webapi.check_run('') is None
    assert webapi.check_run(RUN) == RUN
    for bad in ('2026-10-08', RUN + '\n', RUN + '/..', 'latest'):
        with pytest.raises(HttpError) as err:
            webapi.check_run(bad)
        assert err.value.status == 400
    assert webapi.check_mode('nowcast') == 'nowcast'
    with pytest.raises(HttpError) as err:
        webapi.check_mode('tomorrow')
    assert err.value.status == 400
    assert webapi.check_part('series', webapi.RUN_PARTS) == 'series'
    with pytest.raises(HttpError) as err:
        webapi.check_part('nope', webapi.RUN_PARTS)
    assert err.value.status == 404
    assert webapi.check_format(None) == 'json' and webapi.check_format('csv') == 'csv'
    with pytest.raises(HttpError) as err:
        webapi.check_format('xml')
    assert err.value.status == 400


def test_ttl_cache_with_an_injected_clock():
    now = [0.0]
    calls = []
    cache = webapi.TTLCache(ttl=60, clock=lambda: now[0])

    def loader():
        calls.append(1)
        return len(calls)
    assert cache.get('k', loader) == 1 and cache.get('k', loader) == 1
    now[0] = 59.9
    assert cache.get('k', loader) == 1
    now[0] = 60.0
    assert cache.get('k', loader) == 2

    def boom():
        raise FileNotFoundError('x')
    with pytest.raises(FileNotFoundError):
        cache.get('other', boom)
    assert cache.get('other', lambda: 'ok') == 'ok'


def test_lru_cache_evicts_the_oldest():
    cache = webapi.LRUCache(maxsize=2)
    assert cache.get('a', lambda: 1) == 1 and cache.get('b', lambda: 2) == 2
    assert cache.get('a', lambda: 99) == 1          # acierto: `a` pasa a ser el más reciente
    assert cache.get('c', lambda: 3) == 3           # expulsa `b`
    assert cache.get('b', lambda: 20) == 20 and len(cache) == 2


def test_settings_defaults_without_config(fresh_app):
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}
    from mtpy.core.app import Config
    fresh_app.set('config', Config({'web': {'prefix': 'site-dry/v1', 'cache_ttl': '5'}}))
    assert webapi.settings() == {'prefix': 'site-dry/v1', 'cache_ttl': 5.0}
    fresh_app.set('config', Config({'web': {'prefix': '', 'cache_ttl': ''}}))
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}


def test_site_reads_the_bundle_and_resolves_latest(site_app):
    site = webapi.site()
    assert site is webapi.site()
    manifest = bundle.loads(site.manifest())
    assert manifest['schema'] == 'manifest@1' and site.latest_run('es') == RUN
    assert bundle.loads(site.history('es'))['data']['runs'][0]['run_id'] == RUN
    assert bundle.loads(site.run_file('es', RUN, 'meta'))['schema'] == 'meta@1'
    assert bundle.loads(site.run_file('es', RUN, 'vote', 'forecast'))['mode'] == 'forecast'
    assert site.run_csv('es', RUN, 'series') == b'date,name\n2026-10-05,PP\n'
    assert site.run_csv('es', RUN, 'vote', 'forecast').startswith(b'name,pct\n')
    assert site.freeze() == {'active': False, 'message': None}
    for call, status in (
        (lambda: site.latest_run('es-md'), 404), (lambda: site.history('es-md'), 404),
        (lambda: site.run_file('es', '20200101-000000', 'meta'), 404), (lambda: site.run_csv('es', RUN, 'meta'), 404),
    ):
        with pytest.raises(HttpError) as err:
            call()
        assert err.value.status == status


def test_site_scopes_crosses_the_catalogue_with_the_manifest(site_app):
    data = webapi.site().scopes()
    codes = [row['code'] for row in data['scopes']]
    assert codes[:2] == ['es', 'es-an'] and len(codes) == 18 and data['contract'] == 1
    es = data['scopes'][0]
    assert es['simulable'] is True and es['latest'] == RUN and es['seats'] == 350 and es['parent'] is None
    md = [row for row in data['scopes'] if row['code'] == 'es-md'][0]
    assert md['simulable'] is False and md['latest'] is None and md['parent'] == 'es' and md['name'] == 'Madrid'


def test_site_without_manifest_is_a_503_and_caches_nothing(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    with pytest.raises(HttpError) as err:
        webapi.site().manifest()
    assert err.value.status == 503
    write_bundle(fresh_app.fs)
    assert bundle.loads(webapi.site().manifest())['data']['scopes']['es']['latest'] == RUN


def test_site_pointers_expire_with_the_ttl(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    writer = write_bundle(fresh_app.fs)
    now = [0.0]
    site = webapi.Site(bundle.BundleReader(fresh_app.fs), ttl=60, clock=lambda: now[0])
    assert site.latest_run('es') == RUN
    publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert site.latest_run('es') == RUN
    now[0] = 61.0
    assert site.latest_run('es') == '20261009-120000'


def test_site_requires_a_file_system(fresh_app):
    with pytest.raises(HttpError) as err:
        webapi.site()
    assert err.value.status == 503


def test_webapi_import_stays_light():
    """Los workers de la web no deben cargar el modelo ni la pila científica: `webapi` solo importa `bundle`."""
    import subprocess
    import sys
    code = ("import sys, mtpy.lib.webapi; "
            "print(sorted(m for m in ('mtpy.lib.data', 'mtpy.lib.publish', 'mtpy.lib.simulator', 'scipy', 'statsmodels') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == '[]'
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_webapi_unit.py -q`
Expected: `ImportError: cannot import name 'webapi' from 'mtpy.lib'` en la recogida.

- [ ] **Step 3: Implementar `mtpy/lib/webapi.py` y la configuración**

Según Interfaces (imports: `time`, `collections.OrderedDict`, `typing`, `pandas`, `from ..core.api import
HttpError`, `from ..core.app import App`, `from ..core.utils.helpers import is_empty`, `from . import
bundle`, `from .bundle import BundleReader`; **no** `from .data import ...`). Docstring de módulo:
capa de servicio de la API web (fase 2: rutas del paquete; las rutas de base llegan en la fase 4). En
`config.json` añadir `"web": {"prefix": "${WEB_PREFIX}", "cache_ttl": "${WEB_CACHE_TTL}"}` tras `s3`; en
`.env.example` añadir `WEB_PREFIX=` y `WEB_CACHE_TTL=` tras `S3_BUCKET=`.

- [ ] **Step 4: Ejecutar los tests**

Run: `python -m pytest tests/test_webapi_unit.py -q && python -m pytest -m "not integration" -q`
Expected: 10 passed; 275 passed en la suite.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/webapi.py tests/test_webapi_unit.py config.json .env.example
git commit -m "$(printf 'Add the web API service layer: validators, caches and bundle reader\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 3: `ROUTES`, controladores `Base` y `Forecast`, `api.py`

**Files:**
- Modify: `mtpy/lib/webapi.py` (`ROUTES`, `add_routes`), `api.py` (raíz)
- Create: `mtpy/controllers/Base.py`, `mtpy/controllers/Forecast.py`, `tests/test_web_api.py`

**Interfaces:**
- Consumes: Tarea 2; `Controller`, `HttpError`, `Response.set_cache/set_etag/set_content_type/set_header`
  (`mtpy/core/api.py`); el router resuelve `'forecast'` → `mtpy/controllers/Forecast.py` clase `Forecast`
  (`to_camel`); `mtpy.api(routes=)`; arnés `call` de `tests/test_api_asgi.py`.
- Produces en `webapi.py`:

```python
ROUTES = [
    ('/api/v1/health', 'forecast', 'health', ['GET']),
    ('/api/v1/manifest', 'forecast', 'manifest', ['GET']),
    ('/api/v1/scopes', 'forecast', 'scopes', ['GET']),
    ('/api/v1/forecast/{scope}', 'forecast', 'meta', ['GET']),
    ('/api/v1/forecast/{scope}/runs', 'forecast', 'runs', ['GET']),
    ('/api/v1/forecast/{scope}/{part}', 'forecast', 'part', ['GET']),
    ('/api/v1/forecast/{scope}/{mode:str}/{part}', 'forecast', 'mode_part', ['GET']),
]

def add_routes(router) -> None: for route in ROUTES: router.add_route(*route)
```

- Produces `mtpy/controllers/Base.py`:

```python
class Base(Controller):
    """Common behaviour of the /api/v1 controllers: CORS, cache headers, JSON/CSV bytes."""
    def before_dispatch(self): self.response.set_header('access-control-allow-origin', '*')
    @property
    def query(self) -> dict: return self.request.query
    def cache(self, immutable: bool = False) -> None   # set_cache(IMMUTABLE_MAX_AGE, immutable=True) o set_cache(int(settings()['cache_ttl']))
    def no_store(self) -> None                         # set_header('cache-control', 'no-store')
    def json_bytes(self, content: bytes, immutable: bool = False) -> bytes   # content-type application/json; etag md5 hex del contenido; cache(immutable); devuelve content
    def csv_bytes(self, content: bytes, filename: str, immutable: bool = False) -> bytes  # text/csv; content-disposition attachment; filename="..."; etag; cache
    def notice_freeze(self) -> None                    # set_header('x-freeze', 'active') si site().freeze()['active']; un HttpError del manifest aquí se ignora (sin manifest no hay aviso)
```

- Produces `mtpy/controllers/Forecast.py`, `class Forecast(Base)`: `health_action()` → `no_store()`,
  `{'status': 'ok', 'contract': bundle.CONTRACT}` sin tocar `app.fs`; `manifest_action()` →
  `json_bytes(site().manifest())`; `scopes_action()` → `cache()`, `notice_freeze()`, `site().scopes()`;
  `meta_action(scope)` → `await self.part_action(scope, 'meta')`; `runs_action(scope)` → `check_scope`,
  `notice_freeze`, `json_bytes(site().history(scope))`; `part_action(scope, part)` → `check_scope`,
  `check_part(part, RUN_PARTS)`, `run, explicit = self.resolve_run(scope)`, `fmt =
  check_format(self.query.get('format'))`, `notice_freeze`; CSV: `part in CSV_PARTS` o `HttpError(404,
  'no csv for this part')`, `csv_bytes(site().run_csv(scope, run, part), f'{scope}-{run}-{part}.csv',
  explicit)`; JSON: `json_bytes(site().run_file(scope, run, part), explicit)`; `mode_part_action(scope,
  mode, part)` igual con `check_mode`, `MODE_PARTS`, `site().run_file(..., mode)`, CSV
  `f'{scope}-{run}-{mode}-{part}.csv'` (`run_csv(scope, run, part, mode)`). `resolve_run(scope) ->
  tuple[str, bool]`: `check_run(self.query.get('run'))` → `(run, True)` o `(site().latest_run(scope),
  False)`.
- Produces `api.py` (raíz): `from mtpy.lib import webapi` y `api = mtpy.api(routes=webapi.ROUTES)`.
- Produces en `tests/test_web_api.py`: `make_web_api(app) -> Api` (router con el namespace por defecto
  `mtpy.controllers`, `webapi.add_routes(router)`, `app.set('router', router)`, `Api()`), importando
  `call` de `tests.test_api_asgi` y `write_bundle`/`ROOT`/`RUN`/`fixture_data` de `tests.test_webapi_unit`.

- [ ] **Step 1: Crear `tests/test_web_api.py`**

```python
"""Tests de la API `/api/v1` (controladores `Base` y `Forecast`) con el arnés ASGI, sin servidor ni base."""
import hashlib
import json
import os

import pytest

from mtpy.core.api import Api, Router
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi
from tests.test_api_asgi import call
from tests.test_webapi_unit import ROOT, RUN, fixture_data, write_bundle


def make_web_api(app):
    """Router real de la API (namespace `mtpy.controllers`) registrado en `app`; devuelve el ASGI."""
    router = Router()
    webapi.add_routes(router)
    app.set('router', router)
    return Api()


@pytest.fixture
def api(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return make_web_api(fresh_app)


def test_health_needs_nothing_and_is_not_cached(fresh_app):
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/health')
    assert status == 200 and json.loads(body) == {'status': 'ok', 'contract': 1}
    assert headers['cache-control'] == 'no-store' and headers['access-control-allow-origin'] == '*'


def test_manifest_is_served_verbatim_with_pointer_cache(api, tmp_path):
    status, headers, body = call(api, '/api/v1/manifest')
    assert status == 200 and headers['content-type'] == 'application/json; charset=utf-8'
    assert body == (tmp_path / 'site' / 'v1' / 'manifest.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60'
    assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert 'x-freeze' not in headers


def test_without_bundle_the_api_answers_503(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/manifest')
    assert status == 503 and json.loads(body)['message'] == 'no bundle published yet'
    assert headers['cache-control'] == 'no-store'
    status, headers, body = call(make_web_api(fresh_app), '/api/v1/forecast/es/headline')
    assert status == 503


def test_scopes_cross_catalogue_and_manifest(api):
    status, headers, body = call(api, '/api/v1/scopes')
    data = json.loads(body)
    assert status == 200 and data['scopes'][0]['code'] == 'es' and data['scopes'][0]['latest'] == RUN
    assert headers['cache-control'] == 'public, max-age=60'


def test_meta_and_parts_resolve_latest_or_an_explicit_run(api, tmp_path):
    run_dir = tmp_path / 'site' / 'v1' / 'runs' / 'es' / RUN
    status, headers, body = call(api, '/api/v1/forecast/es')
    assert status == 200 and body == (run_dir / 'meta.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='run=' + RUN)
    assert status == 200 and body == (run_dir / 'series.json').read_bytes()
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    status, headers, body = call(api, '/api/v1/forecast/es/runs')
    assert status == 200 and json.loads(body)['schema'] == 'history@1'
    status, headers, body = call(api, '/api/v1/forecast/es/forecast/vote')
    assert status == 200 and json.loads(body)['mode'] == 'forecast' and headers['cache-control'] == 'public, max-age=60'


def test_csv_twins(api):
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='format=csv')
    assert status == 200 and headers['content-type'] == 'text/csv; charset=utf-8'
    assert headers['content-disposition'] == 'attachment; filename="es-{}-series.csv"'.format(RUN)
    assert body == b'date,name\n2026-10-05,PP\n' and headers['cache-control'] == 'public, max-age=60'
    status, headers, body = call(api, '/api/v1/forecast/es/forecast/vote', query='format=csv&run=' + RUN)
    assert status == 200 and headers['content-disposition'].endswith('-forecast-vote.csv"')
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    status, headers, body = call(api, '/api/v1/forecast/es/meta', query='format=csv')
    assert status == 404 and json.loads(body)['message'] == 'no csv for this part'
    status, headers, body = call(api, '/api/v1/forecast/es/series', query='format=xml')
    assert status == 400 and json.loads(body)['message'] == 'invalid format'


@pytest.mark.parametrize('path, query, status, message', [
    ('/api/v1/forecast/es-xx', '', 400, 'invalid scope'),
    ('/api/v1/forecast/es-md', '', 404, 'scope not published'),
    ('/api/v1/forecast/es-md/runs', '', 404, 'no runs published'),
    ('/api/v1/forecast/es', 'run=2026-10-08', 400, 'invalid run'),
    ('/api/v1/forecast/es', 'run=20200101-000000', 404, 'run not found'),
    ('/api/v1/forecast/es/nope', '', 404, 'unknown part'),
    ('/api/v1/forecast/es/tomorrow/vote', '', 400, 'invalid mode'),
    ('/api/v1/forecast/es/forecast/nope', '', 404, 'unknown part'),
    ('/api/v1/forecast/es/forecast/vote/extra', '', 404, '404 Not Found'),
    ("/api/v1/forecast/es'/meta", '', 404, '404 Not Found'),
    ('/api/v1/nope', '', 404, '404 Not Found'),
])
def test_errors_are_json_and_not_cached(api, path, query, status, message):
    got, headers, body = call(api, path, query=query)
    assert got == status and json.loads(body) == {'status': 'error', 'message': message}
    assert headers['cache-control'] == 'no-store' and headers['content-type'] == 'application/json; charset=utf-8'


def test_post_is_not_allowed(api):
    status, headers, body = call(api, '/api/v1/manifest', method='POST')
    assert status == 404


def test_freeze_header_when_the_manifest_is_frozen(api, fresh_app):
    publish.update_manifest(bundle.BundleWriter(fresh_app.fs), freeze={'active': True, 'message': 'Veda'})
    webapi.site().manifest_cache_clear()
    status, headers, body = call(api, '/api/v1/forecast/es/headline')
    assert status == 200 and headers['x-freeze'] == 'active'
    status, headers, body = call(api, '/api/v1/health')
    assert 'x-freeze' not in headers


def test_mtpy_api_registers_the_web_routes(fresh_app):
    from mtpy import mtpy
    api = mtpy.api(routes=webapi.ROUTES)
    patterns = [route['pattern'] for route in fresh_app.router.routes]
    assert patterns[0] == '/' and len(patterns) == 1 + len(webapi.ROUTES)
    assert all(route['methods'] == {'GET'} for route in fresh_app.router.routes[1:])
```

  Nota: `Site` necesita `manifest_cache_clear()` (vacía la caché TTL del manifest; la usa el test del
  `x-freeze`). Añadirlo a `Site` en este paso (`def manifest_cache_clear(self) -> None`).

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_web_api.py -q`
Expected: `AttributeError: module 'mtpy.lib.webapi' has no attribute 'add_routes'` en la fixture / en la
recogida.

- [ ] **Step 3: Implementar `ROUTES`/`add_routes`, `Base`, `Forecast`, `Site.manifest_cache_clear` y `api.py`**

Según Interfaces. `Base.json_bytes` calcula `hashlib.md5(content).hexdigest()` como ETag. En
`Forecast`, `notice_freeze()` se llama después de validar la entrada y antes de leer el fichero.

- [ ] **Step 4: Ejecutar los tests y la suite**

Run: `python -m pytest tests/test_web_api.py tests/test_webapi_unit.py -q && python -m pytest -m "not integration" -q`
Expected: 10 + 11 parametrizados + 10 → 31 passed; 296 passed en la suite (sin `FutureWarning`).

- [ ] **Step 5: Arranque local de la API contra el paquete real y recorrido de rutas**

Run (en segundo plano, con el `.env` anulado para S3):
```bash
S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning &
sleep 8
for p in health manifest scopes forecast/es forecast/es/runs forecast/es/headline "forecast/es/series?run=20261008-181100" forecast/es/forecast/vote "forecast/es/forecast/summary?format=csv" forecast/es-md forecast/es/nope; do
  printf '%-55s ' "$p"; curl -s -o /dev/null -w '%{http_code} %{content_type} cache=%{header_json}\n' "http://127.0.0.1:8000/api/v1/$p" | cut -c1-60
done
kill %1
```
Expected: `200` en las nueve primeras (JSON o `text/csv`), `404` en `forecast/es-md` (ámbito sin
publicar) y en `forecast/es/nope`; `uvicorn` arranca sin tocar la base (el router importa `webapi`, no
`simulator`). Si `curl` muestra `503 no bundle published yet`, el `S3_BUCKET=` no ha surtido efecto:
parar y revisar antes de seguir.

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/webapi.py mtpy/controllers/Base.py mtpy/controllers/Forecast.py api.py tests/test_web_api.py
git commit -m "$(printf 'Serve the published bundle at /api/v1 with cache headers and CSV twins\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 4: Cimientos del frontend: ECharts, CSS, `api.js`, `state.js`, `format.js`, `catalog.js`, `layout.js` y `test_web_routes.py`

**Files:**
- Create: `web/vendor/echarts-5.6.0.min.js`, `web/vendor/LICENSE-echarts.txt`, `web/css/site.css`,
  `web/js/api.js`, `web/js/state.js`, `web/js/format.js`, `web/js/catalog.js`, `web/js/layout.js`,
  `web/js/charts/base.js`, `web/index.html` (esqueleto; la Tarea 5 lo completa), `tests/test_web_routes.py`

**Interfaces:**
- Consumes: `webapi.ROUTES` (patrones), `Router.add_route`/`Router.parse_route` (para casar literales en
  el test), las respuestas de la Tarea 3 (sobres del contrato: `manifest.data.scopes`, `freeze`,
  `attribution`; `meta.data.parties`, `bmaps`, `as_of`, `date_last`, `commit`, `dirty`, `run_id`).
- Produces `web/js/api.js`:

```js
export function apiBase()                       // '' por defecto; `?api=http://host:8000` (solo desarrollo) → ese origen
export async function getJSON(path)             // fetch(apiBase() + path) con caché Map por página; lanza ApiError {status, message} si !ok (lee el JSON de error si puede)
export async function getManifest()             // /api/v1/manifest → envelope
export async function getScopes()               // /api/v1/scopes
export function forecastUrl(scope, part, {mode = null, run = null, format = null} = {})
                                                // '/api/v1/forecast/{scope}[/{mode}]/{part}' + query (run, format) ; part 'runs' → '/api/v1/forecast/{scope}/runs'; part 'meta' → '/api/v1/forecast/{scope}'
export async function getForecast(scope, part, opts)   // getJSON(forecastUrl(...))
export class ApiError extends Error { constructor(status, message) }
```

- Produces `web/js/state.js`: `readState()` → `{scope: 'es', mode: 'forecast', run: null}` desde
  `URLSearchParams` (valores validados: `mode` ∈ `nowcast|forecast`, `run` que case
  `/^\d{8}-\d{6}$/`, si no se ignoran); `writeState(patial)` → `history.replaceState` con la query
  actualizada (sin `run` cuando es `null`) y dispara el evento `statechange` en `document` con el nuevo
  estado; `onStateChange(handler)`; `DEFAULT_MODE = 'forecast'`.
- Produces `web/js/format.js`: `fmtPct(x, digits = 1)` → `"32,7 %"` (`Intl.NumberFormat('es-ES')`;
  `null` → `"–"`), `fmtNum(x, digits = 0)`, `fmtInt(x)`, `fmtDate(iso)` → `"13 oct 2026"` (`Intl.DateTimeFormat('es-ES', {day: 'numeric', month: 'short', year: 'numeric'})`), `fmtDateShort(iso)` → `"13 oct"`, `fmtRange(lo, hi, digits)` → `"27,3–38,0"`, `fmtDateTime(iso)` para `run_at`.
- Produces `web/js/catalog.js`: `OTHERS_COLOR = '#9e9e9e'`, `BLOCK_ORDER = ['Izquierda', 'Separatista',
  'Regionalista', 'Derecha']`, `class Catalog { constructor(meta) ; color(name) ; fullname(name) ;
  block(name) ; blockColor(block) /* color del primer partido del bloque en bmaps.vs, luego bmaps.blocks; OTHERS_COLOR si no */ ; order(names) /* ordena por BLOCK_ORDER y, dentro, por el orden de meta.parties */ }`.
- Produces `web/js/layout.js`: `renderHeader(root, {pages, active, scopes, state})` (título del sitio,
  nav con `Portada` → `/` y `Promedio` → `/promedio` conservando la query; `<select id="scope">` con los
  ámbitos `simulable` del `/scopes`, `<fieldset id="mode">` con dos radios `nowcast`/`forecast` con
  etiquetas "Hoy" y "Elección"; los cambios llaman a `writeState`); `renderFooter(root, {meta, manifest})`
  (`Run {run_id} · Estimación a {as_of} · último sondeo {date_last} · commit {commit}[ (sucio)]` más las
  tres líneas de `manifest.data.attribution` y un enlace a `/api/v1/manifest`); `renderFreeze(root,
  manifest)` (muestra `#freeze-banner` con `manifest.data.freeze.message` cuando `active`); `renderError(root, error)` (mensaje en español con el `status`: 503 → "Todavía no hay ningún pronóstico publicado").
- Produces `web/js/charts/base.js`: `mountChart(el, option)` → `echarts.init(el, null, {renderer: 'canvas'})`, `setOption(option, true)`, `ResizeObserver` → `chart.resize()`, devuelve `chart`; `THEME = {fontFamily: 'system-ui, sans-serif', textColor: '#333', gridColor: '#e5e5e5', animationDuration: 300}`; `import * as echarts from '../../vendor/echarts-5.6.0.min.js'` **no funciona** (el bundle UMD no es un módulo ES): ECharts se carga con `<script src="/vendor/echarts-5.6.0.min.js">` clásico antes del módulo de página y `base.js` usa `window.echarts` (`const echarts = window.echarts`).
- Produces `web/css/site.css`: variables (`--color-text`, `--color-muted`, `--color-bg`, `--color-line`,
  `--max-width: 1100px`), `body` sin margen con `font-family: system-ui`, `header`/`main`/`footer`
  centrados a `--max-width`, rejilla `.grid` de dos columnas que pasa a una por debajo de 720 px, `.card`
  con título, `.chart` con `height: 360px` (`240px` en móvil), `.table-wrap` con `overflow-x: auto`,
  `#freeze-banner` destacado, `.status` para mensajes.
- Produces `web/index.html` (esqueleto; contenido definitivo en la Tarea 5): `<!doctype html>`,
  `lang="es"`, `<meta charset>`, `<meta name="viewport">`, `<title>Pronóstico electoral · Portada</title>`,
  `<link rel="stylesheet" href="/css/site.css">`, `<script src="/vendor/echarts-5.6.0.min.js"></script>`,
  `<script type="module" src="/js/pages/index.js"></script>`, `<header id="site-header">`, `<main
  id="main">` con `<p class="status" id="status">Cargando…</p>`, `<footer id="site-footer">`, `<div
  id="freeze-banner" hidden>`.
- Produces `tests/test_web_routes.py` (código abajo).

- [ ] **Step 1: Descargar ECharts y su licencia**

Run:
```bash
mkdir -p web/vendor && curl -fsSL -o web/vendor/echarts-5.6.0.min.js https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js && curl -fsSL -o web/vendor/LICENSE-echarts.txt https://cdn.jsdelivr.net/npm/echarts@5.6.0/LICENSE && wc -c web/vendor/echarts-5.6.0.min.js && head -c 60 web/vendor/echarts-5.6.0.min.js && head -2 web/vendor/LICENSE-echarts.txt
```
Expected: `1034102 web/vendor/echarts-5.6.0.min.js`; el fichero empieza por un comentario de licencia o
`!function` (UMD); la licencia es Apache 2.0.

- [ ] **Step 2: Crear `tests/test_web_routes.py`**

```python
"""Tests estáticos del frontend `web/`: literales de la API frente a `ROUTES`, recursos referenciados e imports."""
import os
import re

import pytest

from mtpy.core.api import Router
from mtpy.lib import webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEB = os.path.join(ROOT, 'web')


def js_files():
    for folder, _, files in os.walk(os.path.join(WEB, 'js')):
        for name in files:
            if name.endswith('.js'):
                yield os.path.join(folder, name)


def html_files():
    return [os.path.join(WEB, f) for f in os.listdir(WEB) if f.endswith('.html')]


def router():
    r = Router()
    webapi.add_routes(r)
    return r


def matches_a_route(path):
    return any(Router.parse_route(route, path, 'GET') is not False for route in router().routes)


def test_every_api_literal_in_the_js_matches_a_route():
    literals = []
    for path in js_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for match in re.finditer(r"""['"`](/api/v1/[^'"`?]*)""", text):
            literal = re.sub(r'\$\{[^}]*\}', 'x', match.group(1)).rstrip('/')
            literals.append((os.path.relpath(path, ROOT), literal))
    assert literals, 'ningún literal /api/v1 en web/js'
    bad = [(f, lit) for f, lit in literals if not matches_a_route(lit)]
    assert bad == []


def test_referenced_assets_exist():
    missing = []
    for path in html_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for ref in re.findall(r'(?:src|href)="(/[^"]+)"', text):
            if not ref.startswith('/api/') and not os.path.isfile(os.path.join(WEB, ref.lstrip('/'))):
                missing.append((os.path.basename(path), ref))
        assert 'type="module"' in text and '<script' in text
        assert 'onclick=' not in text and '<script>' not in text  # CSP: sin inline
    assert missing == []


def test_js_imports_resolve():
    missing = []
    for path in js_files():
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for ref in re.findall(r"""from\s+['"](\.{1,2}/[^'"]+)['"]""", text):
            if not os.path.isfile(os.path.normpath(os.path.join(os.path.dirname(path), ref))):
                missing.append((os.path.relpath(path, ROOT), ref))
    assert missing == []


def test_vendor_is_pinned_and_licensed():
    assert os.path.getsize(os.path.join(WEB, 'vendor', 'echarts-5.6.0.min.js')) > 900_000
    with open(os.path.join(WEB, 'vendor', 'LICENSE-echarts.txt'), encoding='utf-8') as fh:
        assert 'Apache License' in fh.read(400)
```

- [ ] **Step 3: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_web_routes.py -q`
Expected: `test_every_api_literal_in_the_js_matches_a_route` falla (`ningún literal /api/v1 en web/js`),
`test_referenced_assets_exist` falla por `index.html` ausente o sin módulo; el de vendor pasa.

- [ ] **Step 4: Escribir los módulos, el CSS y el esqueleto de `index.html`**

Según Interfaces. `api.js` construye URLs solo con las plantillas
`` `/api/v1/forecast/${scope}/${part}` ``, `` `/api/v1/forecast/${scope}/${mode}/${part}` ``,
`` `/api/v1/forecast/${scope}/runs` ``, `` `/api/v1/forecast/${scope}` ``, `'/api/v1/manifest'`,
`'/api/v1/scopes'` (el test las casa con `ROUTES`). `apiBase()` lee `new URLSearchParams(location.search).get('api')` y lo acepta solo si empieza por `http://localhost` o `http://127.0.0.1` (desarrollo).

- [ ] **Step 5: Ejecutar los tests**

Run: `python -m pytest tests/test_web_routes.py -q && python -m pytest -m "not integration" -q`
Expected: 4 passed (el de literales pasa con los de `api.js`); 300 passed en la suite.

- [ ] **Step 6: Commit**

```bash
git add web/vendor web/css/site.css web/js/api.js web/js/state.js web/js/format.js web/js/catalog.js web/js/layout.js web/js/charts/base.js web/index.html tests/test_web_routes.py
git commit -m "$(printf 'Add the web front-end foundation with vendored ECharts\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 5: Portada (`/`): titular, voto, escaños, hemiciclo, mayorías y evolución

**Files:**
- Create: `web/js/pages/index.js`, `web/js/charts/bars.js`, `web/js/charts/hemicycle.js`,
  `web/js/charts/evolution.js`
- Modify: `web/index.html`

**Interfaces:**
- Consumes: Tarea 4; datos: `manifest`, `scopes`, `meta` (`parties`, `bmaps`, `n_seats`, `majority`,
  `as_of`, `date_last`, `event_date`, `run_id`, `commit`, `dirty`), `headline` (`nowcast`/`forecast`:
  `parties[{name, pct, lo, hi, seats, seats_lo, seats_hi, p_first}]`, `p_majority`), `{mode}/vote`
  (`rows[{name, pct, sd, lo, hi}]`, `when`, `horizon`), `{mode}/summary` (`parties[...]`, `vs`,
  `p_majority`, `totals`, `n_seats`, `majority`), `runs` (`history.data.runs[]`).
- Produces `web/js/charts/bars.js`: `renderBars(el, rows, {value, lo, hi, colors, formatter, majority})`
  → barras horizontales ordenadas como `rows` (una por `name`, color de `colors[name]`), con el intervalo
  `[lo, hi]` dibujado como una línea con topes mediante una serie `custom` (`renderItem` sobre el eje de
  valor), etiqueta del valor al final de la barra con `formatter`, `markLine` vertical en `majority`
  cuando se da (escaños), tooltip con nombre completo, valor e intervalo; devuelve el chart.
- Produces `web/js/charts/hemicycle.js`: `renderHemicycle(el, seats, {colors, order, nSeats,
  majority})` → `pie` con `startAngle: 180`, `endAngle: 0`, `radius: ['45%', '85%']`, `center: ['50%',
  '75%']`, datos `seats` filtrados a `> 0` y ordenados con `order` (izquierda a derecha), etiquetas
  `{name}\n{seats}` solo si `seats >= nSeats * 0.03`, texto central `graphic` con `"{majority} escaños
  para la mayoría"`, tooltip con nombre, escaños y porcentaje de la cámara.
- Produces `web/js/charts/evolution.js`: `renderEvolution(el, runs, mode, {colors, parties})` → una
  línea por partido (`pct` del `headline[mode]`) sobre el eje temporal `run_at`; con un único run dibuja
  los puntos (`showSymbol: true`); tooltip con fecha de publicación y `pct` por partido; `dataZoom`
  cuando hay más de 20 runs.
- Produces `web/js/pages/index.js` (módulo de página): al cargar, `renderHeader` con `Cargando…` →
  `Promise.all([getManifest(), getScopes()])` → `state = readState()`; si `state.scope` no está en el
  manifest, usa el primero publicado y `writeState`; carga `meta` (con `run`), `headline`, `vote` y
  `summary` del `mode`, y `runs`; pinta: `#headline` tabla (partido con punto de color, `pct` con
  `[lo, hi]`, `seats` con `[seats_lo, seats_hi]`, `p_first` en %; todas las filas de `headline[mode].parties`,
  orden de llegada), `#vote-chart` (`renderBars` con `pct/lo/hi`, filas ordenadas por `pct` desc),
  `#seats-chart` (`renderBars` con `summary.parties` `seats/seats_lo/seats_hi`, filas con `seats > 0 ||
  seats_hi > 0`, `majority`), `#hemicycle` (`summary.totals` o `headline seats`), `#majority` (dos barras
  horizontales a escala 0-100 % con `headline[mode].p_majority` por bloque, color `blockColor`, y la
  frase "{bloque}: {p} % de probabilidad de mayoría absoluta"), `#evolution` (`renderEvolution`), footer
  y banner; títulos de sección con el modo: "Pronóstico para el {event_date}" (forecast) o "Estimación a
  {as_of}" (nowcast); en `statechange` vuelve a cargar lo que depende de `scope`/`mode`/`run` (todo salvo
  manifest y scopes) y repinta; cualquier `ApiError` → `renderError` en `#status`.
- Produces `web/index.html` definitivo: secciones `<section class="card" id="headline">`, `.grid` con
  `#vote-chart` y `#seats-chart` (`<div class="chart">`), `.grid` con `#hemicycle` y `#majority`,
  `#evolution`; cada `.card` con `<h2>` (texto fijo en español; el modo lo completa JS en un `<span>`).
  Un párrafo "Qué es esto" de dos líneas bajo el titular (texto provisional; `metodo.html` llega en la
  fase 5) con enlace a `/promedio`.

- [ ] **Step 1: Escribir los tres módulos de gráficos y la página**

Según Interfaces. Sin `fetch` directo: solo `api.js`. Sin `innerHTML` con datos de la API sin escapar:
construir nodos con `document.createElement`/`textContent` (los nombres de partido vienen del paquete,
pero el hábito evita XSS si en el futuro llegan de la base).

- [ ] **Step 2: Comprobación estática**

Run: `python -m pytest tests/test_web_routes.py -q`
Expected: 4 passed (los literales nuevos casan con `ROUTES`; los imports resuelven; `index.html`
referencia ficheros existentes).

- [ ] **Step 3: Comprobación funcional sin navegador**

Run (API + estático en local, con el paquete real):
```bash
S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning &
sleep 8
python -m http.server 8080 -d web >/dev/null 2>&1 &
sleep 1
curl -s -o /dev/null -w 'index %{http_code}\n' http://127.0.0.1:8080/index.html
curl -s -o /dev/null -w 'page js %{http_code}\n' http://127.0.0.1:8080/js/pages/index.js
curl -s http://127.0.0.1:8000/api/v1/forecast/es/headline | python -c "import json,sys; d=json.load(sys.stdin)['data']; print('headline parties', len(d['forecast']['parties']), 'p_majority', d['forecast']['p_majority'])"
kill %1 %2
```
Expected: `200`, `200`, y la línea con 13 partidos y dos bloques. La comprobación visual (abrir
`http://127.0.0.1:8080/?api=http://127.0.0.1:8000` con los dos servidores arrancados) es de Luis; anotar
en el informe que no se pudo hacer en la sesión si no hay navegador.

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/js/pages/index.js web/js/charts/bars.js web/js/charts/hemicycle.js web/js/charts/evolution.js
git commit -m "$(printf 'Add the home page: headline, vote and seat bars, hemicycle, majorities and evolution\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 6: Promedio (`/promedio`): serie con sondeos y proyección, tabla de sondeos

**Files:**
- Create: `web/promedio.html`, `web/js/pages/promedio.js`, `web/js/charts/series.js`

**Interfaces:**
- Consumes: Tarea 4; datos: `meta`, `series` (`dates`, `parties`, `mean/lo/hi` columnares), `polls`
  (`columns`, `polls[]` con `date`, `pollster`, `sponsor`, `sample_size`, `weight` y un valor o `null`
  por partido; `results[]`), `{mode}/projection` (`dates`, `groups.parties.{names, mean, lo, hi}`),
  `{mode}/vote` (`when`).
- Produces `web/js/charts/series.js`: `renderSeries(el, {series, polls, projection, parties, colors,
  asOf, when})` → eje X temporal; por partido: banda `lo–hi` (dos series `line` apiladas con `stack` y
  `areaStyle` transparente abajo, color del partido con `opacity 0.15`), línea `mean` (`lineWidth 2`,
  `showSymbol: false`), puntos de sondeos `scatter` (`symbolSize 6`, `opacity 0.6`, tooltip con casa,
  fecha y valor), banda de proyección desde `asOf` hasta `when` (`projection.groups.parties`, mismo
  partido, trazo discontinuo), `markLine` vertical en `asOf` ("Estimación") y en `when` ("Elección" solo
  en forecast); `dataZoom` tipo `slider` con ventana inicial de los últimos 180 días; leyenda con
  selección por partido (por defecto solo `bmaps.main`); `connectNulls: false`.
- Produces `web/js/pages/promedio.js`: carga `manifest`, `scopes`, `state`, `meta`, `series`, `polls`,
  `projection` del modo y `vote` del modo; pinta `#series-chart` con `renderSeries` y `#polls-table`
  (las 40 últimas filas de `polls.polls`, más recientes primero: fecha, casa, patrocinador, muestra y una
  columna por partido de `bmaps.main` con `fmtNum(x, 1)` o `"–"`; `<caption>` con `n_polls` del `meta` y
  enlace "Descargar CSV" a `forecastUrl(scope, 'polls', {format: 'csv', run})`); header, footer, banner,
  errores y `statechange` como en la portada.
- Produces `web/promedio.html`: mismo esqueleto que `index.html` (`<title>Pronóstico electoral ·
  Promedio de sondeos</title>`, módulo `/js/pages/promedio.js`), `.card#series` con `.chart` alto
  (`480px`), `.card#polls` con `.table-wrap`.

- [ ] **Step 1: Escribir la página, el módulo y el gráfico**

Según Interfaces. Las fechas del eje X se construyen con `new Date(iso)` (los ISO `YYYY-MM-DD` del
paquete se interpretan como UTC: usar `iso + 'T00:00:00'` para evitar el desplazamiento de un día en
zonas negativas).

- [ ] **Step 2: Comprobación estática y funcional**

Run: `python -m pytest tests/test_web_routes.py -q` y el mismo arranque local de la Tarea 5 Step 3
pidiendo `promedio.html`, `js/pages/promedio.js` y
`curl -s "http://127.0.0.1:8000/api/v1/forecast/es/polls" | python -c "import json,sys; d=json.load(sys.stdin)['data']; print(len(d['polls']), d['columns'])"`.
Expected: 4 passed; `200`, `200`, `315 [...]`.

- [ ] **Step 3: Commit**

```bash
git add web/promedio.html web/js/pages/promedio.js web/js/charts/series.js
git commit -m "$(printf 'Add the poll average page with the series chart and the polls table\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 7: nginx (estático + proxy con caché) y scripts de comprobación

**Files:**
- Modify: `deploy/docker/nginx.conf`
- Create: `deploy/nginx-check.sh`, `deploy/check-api.sh`, `deploy/smoke.sh`

**Interfaces:**
- Produces `deploy/docker/nginx.conf` (fichero completo):

```nginx
daemon off;
worker_processes auto;
pid /run/nginx.pid;
error_log /dev/stderr warn;

include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;
}

http {
    sendfile on;
    tcp_nopush on;
    types_hash_max_size 2048;
    server_tokens off;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    access_log /dev/stdout;
    keepalive_timeout 60;

    gzip on;
    gzip_min_length 1024;
    gzip_types application/json application/javascript text/css text/csv image/svg+xml;

    limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;
    limit_req_status 429;
    proxy_cache_path /var/cache/nginx/api levels=1:2 keys_zone=api:10m max_size=100m inactive=10m use_temp_path=off;

    upstream app_server {
        server 127.0.0.1:8000 fail_timeout=0;
    }

    server {
        listen 8042;
        server_name localhost;
        charset utf-8;
        root /app/web;
        index index.html;

        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy strict-origin-when-cross-origin always;
        add_header Content-Security-Policy "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'" always;

        location = /healthz {
            access_log off;
            default_type text/plain;
            return 200 'ok';
        }

        location /api/ {
            limit_req zone=api burst=40 nodelay;
            proxy_cache api;
            proxy_cache_valid 200 60s;
            proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
            proxy_cache_lock on;
            add_header X-Cache $upstream_cache_status always;
            add_header X-Content-Type-Options nosniff always;
            add_header Referrer-Policy strict-origin-when-cross-origin always;
            add_header Content-Security-Policy "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'" always;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Host $http_host;
            proxy_redirect off;
            proxy_pass http://app_server;
        }

        location /vendor/ {
            expires 1y;
        }

        location ~* \.(js|css)$ {
            expires -1;
        }

        location / {
            try_files $uri $uri.html $uri/ =404;
        }
    }
}
```

  Notas: `proxy_pass http://app_server;` sin barra final conserva `/api/...` íntegro; nginx respeta el
  `Cache-Control` de gunicorn (`no-store` nunca se cachea; `max-age=60`/`immutable` sí) y
  `proxy_cache_valid` solo actúa cuando falta la cabecera. Un `add_header` dentro de un `location` anula
  los del `server`, por eso los tres de seguridad se repiten en `/api/` (el único `location` con
  `add_header`); `/vendor/` y los `.js`/`.css` usan `expires` (`1y` → `max-age=31536000`; `-1` →
  `no-cache`), que no rompe la herencia. `/var/cache/nginx` existe en la imagen (`chown docker` en el
  `Dockerfile`). El exceso de peticiones responde 429.
- Produces `deploy/nginx-check.sh`: comprueba la sintaxis con el nginx local (macOS Homebrew o Linux):
  copia `nginx.conf` a un directorio temporal sustituyendo `/etc/nginx/mime.types` por el `mime.types`
  del nginx instalado (`nginx -V` → `--conf-path` → su directorio, o `/opt/homebrew/etc/nginx` /
  `/etc/nginx`), `include /etc/nginx/modules-enabled/*.conf;` por una línea vacía, `/run/nginx.pid` y
  `/var/cache/nginx/api` por rutas bajo el temporal, y ejecuta `nginx -t -c <copia> -p <temporal>`;
  sale con el código de `nginx -t`. Uso: `bash deploy/nginx-check.sh`.
- Produces `deploy/check-api.sh BASE_URL`: con `curl` y `python` recorre `/healthz` (solo si `BASE_URL`
  es nginx: se acepta 404 cuando se apunta a gunicorn), `/api/v1/health`, `/api/v1/manifest`,
  `/api/v1/scopes`, `/api/v1/forecast/es`, `/api/v1/forecast/es/runs`, `/api/v1/forecast/es/headline`,
  `/api/v1/forecast/es/forecast/vote`, `/api/v1/forecast/es/forecast/summary?format=csv`,
  `/api/v1/forecast/es-xx` (espera 400), `/api/v1/forecast/es/nope` (espera 404); comprueba código,
  `content-type` y que los JSON se parsean; imprime una línea por ruta y termina con error si alguna falla.
- Produces `deploy/smoke.sh [ENV_FILE]`: `set -euo pipefail`; `docker build --platform linux/amd64 -t
  elections-web:smoke .`; `docker run -d --name elections-smoke -p 127.0.0.1:8042:8042` con `--env-file
  ENV_FILE` si se da, y si no `-e APP_API=api -e DB_ADAPTER=PostgreSQL -e S3_BUCKET= -v "$PWD/files:/app/files"`
  (paquete local); espera hasta 90 s a `/healthz`; `docker run --rm elections-web:smoke bash -c 'find /app
  -name "*.env" | wc -l'` debe dar `0`; `bash deploy/check-api.sh http://127.0.0.1:8042`; `curl` de `/`,
  `/promedio` y `/vendor/echarts-5.6.0.min.js` (200, `text/html`/`application/javascript`); mide `docker
  stop` (< 10 s con `date +%s`); limpia el contenedor en un `trap`.

- [ ] **Step 1: Escribir los tres scripts y el `nginx.conf`**

Según Interfaces; `chmod +x` a los scripts; comentarios de cabecera en español explicando el uso.

- [ ] **Step 2: Comprobar la sintaxis de nginx en local**

Run: `bash deploy/nginx-check.sh`
Expected: `nginx: the configuration file ... syntax is ok` y `test is successful`.

- [ ] **Step 3: Recorrer la API con `check-api.sh` contra uvicorn**

Run: `S3_BUCKET= python -m uvicorn api:api --port 8000 --log-level warning & sleep 8; bash deploy/check-api.sh http://127.0.0.1:8000; kill %1`
Expected: todas las rutas `OK` (el `/healthz` marcado como "n/a sin nginx").

- [ ] **Step 4: Humo Docker (solo si el demonio responde)**

Run: `docker info >/dev/null 2>&1 && bash deploy/smoke.sh || echo "docker no disponible: smoke.sh queda para Luis"`
Expected: con Docker, `smoke.sh` termina en verde (imagen sin `*.env`, rutas OK, `/promedio` 200,
`docker stop` en < 10 s); sin Docker, el mensaje. Anotar el resultado en el informe.

- [ ] **Step 5: Commit**

```bash
git add deploy/docker/nginx.conf deploy/nginx-check.sh deploy/check-api.sh deploy/smoke.sh
git commit -m "$(printf 'Serve web/ with nginx, proxy /api with cache and add smoke scripts\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 8: Documentación: rutas en el contrato, runbook y cierre de la fase

**Files:**
- Modify: `docs/web/contrato.md` (sección "Rutas de la API"), `deploy/README.md`,
  `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Estado al cierre de la fase 2"),
  `docs/superpowers/plans/2026-10-08-web-fase-2-api-sitio.md` (cabecera "Estado")

- [ ] **Step 1: Completar "Rutas de la API" en `docs/web/contrato.md`**

Tabla ruta → fichero servido → caché (`max-age=60` / `immutable` con `?run=` / `no-store`), los
parámetros `run` y `format`, los errores (400/404/503) y las cabeceras (`ETag`, `Access-Control-Allow-Origin`,
`x-freeze`); `/scopes` con sus campos; nota de que la API sirve los ficheros tal cual; "Rutas de base
(`/parties`, `/pollsters`, ...): fase 4".

- [ ] **Step 2: Actualizar `deploy/README.md`**

Título "Despliegue de la web (fases 0-2)". Sección nueva "La web" tras "Arrancar el contenedor":
qué sirve nginx (`web/` en `/`, `/api/` → gunicorn con caché 60 s y `limit_req`, `/healthz`), comprobación
tras el arranque (`curl /healthz`, `bash deploy/check-api.sh http://127.0.0.1:8042`, abrir `/` y
`/promedio`), desarrollo sin Docker (`S3_BUCKET= python -m uvicorn api:api --port 8000` + `python -m
http.server 8080 -d web` + `http://127.0.0.1:8080/?api=http://127.0.0.1:8000`), `deploy/nginx-check.sh`,
`deploy/smoke.sh` (con y sin `deploy/elections.env`), variables `WEB_PREFIX`/`WEB_CACHE_TTL`, y la
retirada (`point`/`unpublish` visibles en ≤ 2 min: TTL 60 s de la API + 60 s de nginx). Mencionar que la
web solo necesita S3 (no la base) hasta la fase 4.

- [ ] **Step 3: Añadir "Estado al cierre de la fase 2" a la spec y la cabecera al plan**

Tras "Estado al cierre de la fase 1": ficheros creados, rango de commits, recuento de tests, decisiones
(modo por defecto `forecast`; sin `WEB_FREEZE`; `simulable` = publicado; ficheros servidos tal cual;
ECharts 5.6.0 vendorizado con versión en el nombre; `x-freeze`; `BLOCK_ORDER`; infraestructura limitada a
`nginx.conf`), lo verificado en la sesión (arnés, `check-api.sh` contra uvicorn, `nginx-check.sh`,
`smoke.sh` si hubo Docker), lo pendiente de Luis (recorrido visual en escritorio y móvil, `smoke.sh` con
Docker, despliegue en el servidor con `deploy/elections.env`, dominio/TLS del proxy delante de 8042,
primera publicación a S3 para que la web tenga datos en producción) y lo diferido a la fase 6 de los
flecos de la fase 0 (`HEAD`, `send()` fuera del `try` de `dispatch`, docstring de `Api.__call__`). En el
plan, la cabecera `> **Estado:** completado ...` como en los planes anteriores.

- [ ] **Step 4: Suite y commit**

Run: `python -m pytest -m "not integration" -q`
Expected: 300 passed.

```bash
git add docs/web/contrato.md deploy/README.md docs/superpowers/specs/2026-10-07-web-publicacion-design.md docs/superpowers/plans/2026-10-08-web-fase-2-api-sitio.md
git commit -m "$(printf 'Document the API routes, the web runbook and the phase 2 closing state\n\nCo-Authored-By: <modelo> <noreply@anthropic.com>')"
```

---

### Task 9: Verificación de Luis (fuera de la sesión; no bloquea las Tareas 1-8)

- [ ] Arrancar Docker y ejecutar `bash deploy/smoke.sh` (paquete local) y `bash deploy/smoke.sh
  deploy/elections.env` (S3 real, lectura).
- [ ] Recorrer `/` y `/promedio` en escritorio y móvil sin errores de consola, con `?mode=nowcast`,
  `?scope=es&run=20261008-181100` y tras una retirada (`point`) comprobar que el cambio se ve en ≤ 2 min.
- [ ] Publicar `es` a S3 (`set -a; . deploy/elections.env; set +a; python job.py publish
  '{"what":["forecast"],"scopes":["es"]}'`) para que el sitio en producción tenga datos; desplegar la
  imagen en el servidor con `--env-file deploy/elections.env`; decidir el proxy/TLS delante de 8042.
- [ ] Decidir si el modo por defecto es `forecast`, si la portada debe agrupar partidos (fase 3) y el
  orden del hemiciclo (`BLOCK_ORDER`).
