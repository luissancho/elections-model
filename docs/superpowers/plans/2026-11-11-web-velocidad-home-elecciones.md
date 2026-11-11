# Velocidad de la home y de `/elecciones` — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Con la copia de 60 s caducada, la home y `/elecciones` dejan de tardar unos 6 s: la home pierde la tira «Elecciones pasadas», las elecciones pasadas se leen de S3 en paralelo y sin `exists`, y nginx sirve `/elecciones` desde la copia guardada mientras la refresca por detrás.

**Architecture:** `Site` (`mtpy/lib/webapi/published.py`) gana un pool de hilos y lee por oleadas planas: una de listados y otra de metas, siempre lanzadas desde el hilo de la petición (una tarea del pool nunca espera a otra). `Site.events` conserva su entrada de caché y su resultado; `Site.event_metas` da a `published_events` las metas ya leídas; `Site.warm_events` y `Site.warm_analysis` precargan lo que `/elecciones` recorre ámbito a ámbito. En nginx, un `location = /elecciones` repite el bloque de `location /` con `proxy_cache_background_update on`.

**Tech Stack:** Python 3.11 (`concurrent.futures.ThreadPoolExecutor`, s3fs detrás de `mtpy.core.services.s3.S3`), Jinja2, pytest; nginx (`proxy_cache`).

**Spec:** `docs/superpowers/specs/2026-11-11-web-velocidad-home-elecciones-design.md`.

## Global Constraints

- Ninguna página muestra algo distinto de lo de hoy salvo la tira de la home: mismas elecciones, mismo orden, mismos 404.
- El contrato del bundle no cambia y no hay que republicar. `BundleReader.list_events` y `event_exists` conservan su comportamiento (los usan los jobs de los estudios).
- El TTL de 60 s no se toca. Una elección está publicada si su `meta.json` existe.
- El refresco por detrás de nginx va sólo en `location = /elecciones`; `location /` queda como está.
- Una tarea del pool nunca llama a `Site._each` ni a nada que espere a otra tarea del pool.
- `POOL_SIZE = 16` (medido el 2026-11-11 desde el portátil: 1, 8, 16 y 32 hilos → 18,4, 2,4, 1,5 y 1,05 s).
- Identificadores en inglés neutro; docstrings numpy en todas las funciones; textos públicos en español.
- Luis trabaja en paralelo en `dev` (cambios suyos sin commit en `tests/test_pages.py` y `web/templates/metodo_escanos.html` el 2026-11-11): el trabajo va en un worktree con su rama (`superpowers:using-git-worktrees`), nunca se revierte ni se reescribe nada suyo.
- Tests con el `pytest` de pyenv (Python 3.11), nunca `python3`, y nunca con `out=None` ni escribiendo en `backtest/results`. Commits en español con el trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; `git status --short` y `git log -1 --oneline` antes de cada commit; sólo los ficheros de la tarea, nunca `git commit -a`.

## Review Focus

1. Una carpeta de fecha sin `meta.json` (elección a medio publicar): no aparece en `events`, no rompe y no dispara lecturas en serie; cubierto en la Tarea 2.
2. Un error de S3 que no es un 404 en mitad de una oleada: se propaga como hoy (página 5xx), no se convierte en «no publicada» y no deja nada en caché; cubierto en la Tarea 2.
3. Caducidad parcial (la lista de fechas sigue vigente y las metas han caducado): mismo resultado, relectura en paralelo y ningún `exists`; cubierto en la Tarea 2.
4. Más ámbitos que hilos (18 ámbitos, 16 hilos): `warm_events` termina sin bloquearse; cubierto en la Tarea 3 con `workers=2` y cinco ámbitos.
5. `/elecciones?scope=es-md` y el detalle `/elecciones/es/…`: la primera entra en el bloque nuevo, el segundo sigue en `location /`; cubierto en la Tarea 4 con nginx real.

---

### Task 1: La home sin la tira «Elecciones pasadas»

**Files:**
- Modify: `web/templates/index.html` (bloque `{% if past_shown %}…{% endif %}`)
- Modify: `mtpy/lib/pages/inicio.py` (`past_events`, `PAST_SHOWN`, `inicio_context`, docstring del módulo e imports que queden sin uso)
- Modify: `web/dist/css/site.css` (las cuatro reglas `.past`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `inicio_context` sin las claves `past_shown` y `past_more`; `mtpy.lib.pages.inicio` sin `past_events` ni `PAST_SHOWN`.

- [ ] **Step 1: Tests que fallan** — en `tests/test_pages.py`:

```python
def test_home_has_no_past_elections_strip(history_two_scopes):
    page = html(call(history_two_scopes, '/')[2])
    assert 'id="past"' not in page and 'Elecciones pasadas' not in page and 'class="past"' not in page
    assert 'href="/elecciones/es/generales/2023-07"' not in page and 'href="/elecciones/es/madrid/2021-05"' not in page
    more = page[page.index('id="more"'):page.index('id="how"')]
    assert '<a class="card card-link" href="/elecciones">' in more  # the link to the history that stays
    assert not hasattr(pages.inicio, 'past_events') and not hasattr(pages.inicio, 'PAST_SHOWN')
    assert '.past' not in open(os.path.join(ROOT, 'web/dist/css/site.css')).read()
```

  Y adaptar los cuatro tests que miran la tira:
  - `test_home_closing_sections_link_the_model_pages_and_the_past`: quitar las dos líneas de `past` (se queda el resto, incluido el chip «PP −25 escaños vs 2023») y renombrarlo a `test_home_closing_sections_link_the_model_pages`.
  - `test_home_past_elections_include_scopes_without_a_run`: borrarlo.
  - `test_history_and_home_link_the_elections_by_slug`: quitar las tres últimas líneas (las de `home`) y renombrarlo a `test_history_links_the_elections_by_slug`.
  - `test_inicio_shows_the_studies_band_only_with_a_study`: quitar `assert page.index('id="past"') < page.index('id="analisis"')` y poner `assert page.index('id="how"') < page.index('id="analisis"')`; el comentario de `write_event` pasa a `# a past election, for the «vs 2023» of the hero`.

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_pages.py -q -k "home or inicio or history_links"` → FAIL (`id="past"` sigue en la página).

- [ ] **Step 3: Implementar** — fuera el bloque de la plantilla, `past_events`, `PAST_SHOWN`, las dos claves del contexto, su mención en los docstrings y las reglas `.past`; quitar de `inicio.py` los imports que queden sin uso (`published_events`, `event_href`).

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_pages.py -q` → PASS.

- [ ] **Step 5: Commit** — «Inicio: fuera la tira de elecciones pasadas (188 lecturas a S3 por render)».

---

### Task 2: Las elecciones de un ámbito, en paralelo y sin `exists`

**Files:**
- Modify: `mtpy/lib/webapi/caches.py` (`TTLCache.fresh`)
- Modify: `mtpy/lib/bundle.py` (`BundleReader.list_event_dates`; `list_events` lo reutiliza)
- Modify: `mtpy/lib/webapi/published.py` (`POOL_SIZE`, `Site.__init__`, `_each`, `_event_meta`, `_load_events`, `events`, `event_metas`)
- Modify: `mtpy/lib/pages/elecciones.py` (`published_events`)
- Modify: `tests/fakes.py` (`CountingFileSystem`)
- Test: `tests/test_webapi_unit.py`, `tests/test_bundle_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Produces:
  - `TTLCache.fresh(key) -> bool`: la clave tiene una entrada sin caducar (no carga ni reordena).
  - `BundleReader.list_event_dates(scope: str) -> list[str]`: las carpetas de `events/{scope}` con forma de fecha, ascendentes, sin mirar metas; `[]` sin carpeta; valida el ámbito como `list_events`.
  - `Site(reader, ttl=DEFAULT_TTL, lru=LRU_SIZE, clock=time.monotonic, workers=POOL_SIZE)` con `self.pool: ThreadPoolExecutor`.
  - `Site._each(call: Callable, items: Sequence) -> list`: `call(item)` de cada elemento, en orden; con menos de dos elementos, en el hilo que llama; si no, en `self.pool`. La primera excepción se propaga.
  - `Site._event_meta(scope: str, date: str) -> Optional[bytes]`: `self.event(scope, date, 'meta')`, o `None` con su 404; cualquier otro error se propaga.
  - `Site._load_events(scopes: Sequence[str]) -> dict`: `{scope: fechas publicadas ascendentes}` en dos oleadas de `_each`: los listados (`reader.list_event_dates`) y las metas (`_event_meta`) de todos los pares `(scope, date)`.
  - `Site.events(scope: str) -> list[str]`: como hoy (entrada `('events', scope)` bajo el TTL), cargada con `_load_events([scope])[scope]`.
  - `Site.event_metas(scope: str) -> dict[str, bytes]`: `{fecha: bytes de su meta}` de `events(scope)`, ascendente; las metas cuya entrada `('event', bundle.path_event(scope, date, 'meta'))` no está vigente (`pointers.fresh`) se releen juntas con `_each`; una meta que ya no existe se omite.
  - `tests.fakes.CountingFileSystem(path, delay=0.0)`: `FileSystem` local que anota bajo un `Lock` cada `read_bytes`, `listdir` y `exists` en `calls: Counter` (por método) y `names: list[tuple[str, str]]` (`(método, nombre)`), antes de delegar (la llamada cuenta aunque lance), duerme `delay` segundos en `read_bytes` y guarda en `peak` el máximo de `read_bytes` simultáneos; `reset()` vacía `calls` y `names` y pone `peak` a cero.
- Consumes: nada de la Tarea 1.

- [ ] **Step 1: Tests que fallan**

  En `tests/test_bundle_unit.py`:

```python
def test_list_event_dates_lists_the_date_folders_without_reading_metas(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    assert writer.list_event_dates('es') == []
    data = load_fixture('event-results')['data']
    for date in ('2023-07-23', '2019-11-10'):
        writer.write_json(bundle.path_event('es', date, 'results'), 'event-results', data, 'es')
    assert writer.list_event_dates('es') == ['2019-11-10', '2023-07-23'] and writer.list_events('es') == []
    with pytest.raises(ValueError):
        writer.list_event_dates('../x')
```

  En `tests/test_webapi_unit.py` (`CountingFileSystem` importado de `tests.fakes`, `Counter` de `collections`):

```python
def counting_site(tmp_path, delay=0.0, workers=16):
    """Site con reloj inyectado sobre un `CountingFileSystem`: `(site, fs, now)`."""
    fs = CountingFileSystem(str(tmp_path), delay=delay)
    now = [0.0]
    return webapi.Site(bundle.BundleReader(fs), ttl=60, clock=lambda: now[0], workers=workers), fs, now


def test_ttl_cache_fresh_tells_an_unexpired_entry():
    now = [0.0]
    cache = webapi.TTLCache(ttl=60, clock=lambda: now[0])
    assert not cache.fresh('k')
    cache.get('k', lambda: None)
    assert cache.fresh('k')
    now[0] = 60.0
    assert not cache.fresh('k')


def test_site_events_read_the_metas_in_parallel_and_never_ask_exists(tmp_path):
    site, fs, now = counting_site(tmp_path, delay=0.02)
    for date in ('2011-11-20', '2015-12-20', '2019-11-10', '2023-07-23'):
        write_event(fs, date=date, model=False)
    fs.reset()
    assert site.events('es') == ['2011-11-20', '2015-12-20', '2019-11-10', '2023-07-23'] == site.reader.list_events('es')
    assert fs.peak >= 2
    fs.reset()
    assert site.events('es-cb') == [] and site.events('es') == site.events('es')
    metas = site.event_metas('es')
    assert list(metas) == site.events('es') and metas['2023-07-23'] == site.event('es', '2023-07-23', 'meta')
    assert fs.calls['exists'] == 0 and fs.calls['read_bytes'] == 0 and fs.calls['listdir'] == 1  # only `es-cb`


def test_site_events_skip_a_date_folder_without_meta(tmp_path):
    site, fs, now = counting_site(tmp_path)
    write_event(fs)
    writer = bundle.BundleWriter(fs)
    writer.write_json(bundle.path_event('es', '2019-11-10', 'results'), 'event-results', fixture_data('event-results'), 'es')
    assert site.events('es') == ['2023-07-23'] == site.reader.list_events('es')
    fs.reset()
    assert site.events('es') == ['2023-07-23'] and list(site.event_metas('es')) == ['2023-07-23']
    assert sum(fs.calls.values()) == 0  # the folder without meta is not asked again within the TTL


def test_site_events_propagate_an_error_that_is_not_a_404_and_cache_nothing(tmp_path, monkeypatch):
    site, fs, now = counting_site(tmp_path)
    write_event(fs)
    write_event(fs, date='2019-11-10', model=False)
    inner = fs.read_bytes
    broken = [True]

    def read(name):
        if broken[0] and name.endswith('2019-11-10/meta.json'):
            raise OSError('S3 down')
        return inner(name)
    monkeypatch.setattr(fs, 'read_bytes', read)
    with pytest.raises(OSError):
        site.events('es')
    broken[0] = False
    assert site.events('es') == ['2019-11-10', '2023-07-23']


def test_site_event_metas_reread_in_parallel_what_expired(tmp_path):
    site, fs, now = counting_site(tmp_path, delay=0.02)
    dates = ['2011-11-20', '2015-12-20', '2019-11-10', '2023-07-23']
    for date in dates:
        write_event(fs, date=date, model=False)
    for date in dates:
        site.event('es', date, 'meta')  # the metas enter the cache at t=0
    now[0] = 30.0
    assert site.events('es') == dates  # the list enters at t=30 and outlives the metas
    now[0] = 61.0
    fs.reset()
    assert list(site.event_metas('es')) == dates
    assert fs.calls == Counter(read_bytes=4) and fs.peak >= 2
```

  En `tests/test_pages.py`:

```python
def counting(fresh_app):
    """Cambia el `fs` del lector de la web por un `CountingFileSystem` sobre la misma carpeta y vacía los punteros."""
    fs = CountingFileSystem(fresh_app.fs.path)
    webapi.site().reader.fs = fs
    webapi.site().manifest_cache_clear()
    return fs


def test_home_reads_only_the_elections_of_its_scope_and_never_asks_exists(history_two_scopes, fresh_app):
    call(history_two_scopes, '/')  # the run files stay in the LRU
    fs = counting(fresh_app)
    assert call(history_two_scopes, '/')[0] == 200
    events = [name for method, name in fs.names if method == 'read_bytes' and '/events/' in name]
    assert fs.calls['exists'] == 0 and fs.calls['listdir'] == 1
    assert events and all('/events/es/' in name for name in events) and len(events) == len(set(events))
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_pages.py -q -k "list_event_dates or fresh or site_events or event_metas or home_reads"` → FAIL (`CountingFileSystem`, `fresh`, `list_event_dates` y `workers` no existen).

- [ ] **Step 3: Implementar** con las firmas de arriba. `list_events` pasa a filtrar `list_event_dates` con `event_exists` (mismo resultado). `published_events` recorre `sorted(site().event_metas(scope).items(), reverse=True)` y saca la meta con `bundle.loads(raw)['data']`; su docstring y su resultado no cambian. Para guardar en caché un valor ya calculado, `self.pointers.get(key, lambda: value)`.

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_bundle_unit.py tests/test_webapi_unit.py tests/test_pages.py tests/test_web_api.py -q` → PASS (incluidos `test_site_events_expire_with_the_ttl` y `test_site_events_and_backtest_read_the_published_files`, sin tocar).

- [ ] **Step 5: Commit** — «Web: las elecciones de un ámbito se leen en paralelo y sin exists (Site.events, Site.event_metas)».

---

### Task 3: Precarga de todos los ámbitos en `/elecciones` y estudios en paralelo

**Files:**
- Modify: `mtpy/lib/webapi/published.py` (`warm_events`, `warm_analysis`, `studies`)
- Modify: `mtpy/lib/pages/elecciones.py` (`elecciones_context`)
- Test: `tests/test_webapi_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Consumes: `Site._each`, `Site._load_events`, `Site.events`, `TTLCache.fresh`, `CountingFileSystem`, `counting_site` y `counting` de la Tarea 2.
- Produces:
  - `Site.warm_events(scopes: Sequence[str]) -> None`: carga con un solo `_load_events` los ámbitos cuya entrada `('events', scope)` no está vigente y guarda cada lista en esa entrada; sin ámbitos caducados no hace nada.
  - `Site.warm_analysis(scopes: Sequence[str], parts: Sequence[str]) -> None`: lee con un `_each` `self.analysis(scope, part)` de todo el producto; un 404 se ignora, cualquier otro error se propaga.
  - `Site.studies() -> list`: igual que hoy, con las metas leídas por `_each`.

- [ ] **Step 1: Tests que fallan**

  En `tests/test_webapi_unit.py`:

```python
def test_site_warm_events_loads_every_scope_in_two_waves_without_deadlock(tmp_path):
    site, fs, now = counting_site(tmp_path, delay=0.02, workers=2)
    scopes = ['es', 'es-md', 'es-an', 'es-ct', 'es-pv', 'es-cb']  # the last one has no elections
    for scope in scopes[:5]:
        write_event(fs, scope=scope, date='2019-11-10', model=False)
        write_event(fs, scope=scope, date='2023-07-23', model=False)
    fs.reset()
    site.warm_events(scopes)
    assert fs.calls == Counter(listdir=6, read_bytes=10) and fs.peak == 2
    fs.reset()
    site.warm_events(scopes)
    assert [site.events(scope) for scope in scopes] == [['2019-11-10', '2023-07-23']] * 5 + [[]]
    assert sum(fs.calls.values()) == 0


def test_site_warm_analysis_ignores_the_unpublished_parts(tmp_path):
    site, fs, now = counting_site(tmp_path)
    write_analysis(fs)
    fs.reset()
    site.warm_analysis(['es', 'es-md'], ('meta', 'poll-errors'))  # nothing of `es-md` is published
    assert fs.calls == Counter(read_bytes=4)
    fs.reset()
    assert site.analysis('es', 'meta') and sum(fs.calls.values()) == 0


def test_site_studies_reads_the_metas_once_and_in_parallel(tmp_path):
    site, fs, now = counting_site(tmp_path, delay=0.02)
    write_study(fs)
    write_study(fs, study='sesgo-cis')
    fs.reset()
    assert site.studies() == ['voto-cera', 'sesgo-cis']
    assert fs.calls == Counter(read_bytes=len(bundle.STUDIES)) and fs.peak >= 2
```

  En `tests/test_pages.py`:

```python
def test_elecciones_reads_every_scope_once_and_never_asks_exists(history_two_scopes, fresh_app):
    before = html(call(history_two_scopes, '/elecciones')[2])
    fs = counting(fresh_app)
    status, _, body = call(history_two_scopes, '/elecciones')
    assert status == 200 and html(body) == before
    scopes = webapi.site().scopes()['scopes']
    events = [name for method, name in fs.names if method == 'read_bytes' and '/events/' in name]
    assert fs.calls['exists'] == 0 and fs.calls['listdir'] == len(scopes)
    assert events and len(events) == len(set(events))  # no file of an election is read twice in one render
```

- [ ] **Step 2: Verlos fallar** — `pytest tests/test_webapi_unit.py tests/test_pages.py -q -k "warm or studies_reads or elecciones_reads"` → FAIL (`warm_events` no existe).

- [ ] **Step 3: Implementar** con las firmas de arriba. En `elecciones_context`, justo después de `common_context(...)`: `codes` = los códigos de `site().scopes()['scopes']`; `site().warm_events(codes)`; `site().warm_analysis([code for code in codes if site().events(code)], ('meta', 'poll-errors'))`. El resto de la función no cambia.

- [ ] **Step 4: Verlos pasar** — `pytest tests/test_webapi_unit.py tests/test_pages.py tests/test_web_api.py -q` → PASS.

- [ ] **Step 5: Medir contra S3** (sólo lectura) con los dos guiones de la sesión del 2026-11-11, `cold_home.py` y `profile_pages.py`, ejecutados con el `mtpy` del worktree (imprimir `mtpy.__file__` para comprobarlo). Esperado con los punteros caducados: home ≈ 25 lecturas y `/elecciones` ≈ 150 (antes 214 y 244), ningún `exists`. Anotar llamadas y segundos para el informe final.

- [ ] **Step 6: Commit** — «Elecciones: precarga en paralelo de las elecciones y del análisis de todos los ámbitos; estudios en paralelo».

---

### Task 4: nginx sirve `/elecciones` desde la copia guardada y la refresca por detrás

**Files:**
- Modify: `deploy/docker/nginx.conf`
- Create: `deploy/nginx-stale-check.sh`
- Modify: `deploy/README.md` (el párrafo de `/` y `/api/` de la sección de nginx y la lista de guiones)
- Test: `tests/test_deploy_files.py`

**Interfaces:**
- Produces: `location = /elecciones { … }` delante de `location / { … }`, con todas las directivas de éste más `proxy_cache_background_update on;`. `deploy/nginx-stale-check.sh [nginx.conf]` sale con 0 si el comportamiento es el esperado.

- [ ] **Step 1: Test que falla** — en `tests/test_deploy_files.py`:

```python
def nginx_location(conf, header):
    """Directivas (líneas sin espacios alrededor) del bloque `location` cuya cabecera es `header`."""
    body = conf[conf.index(header + ' {') + len(header) + 2:]
    return [line.strip() for line in body[:body.index('}')].splitlines() if line.strip() and not line.strip().startswith('#')]


def test_nginx_refreshes_only_the_history_page_in_the_background():
    """`/elecciones` se sirve desde la copia guardada mientras se refresca por detrás; el resto, como siempre."""
    conf = read('deploy/docker/nginx.conf')
    everything, history = nginx_location(conf, 'location /'), nginx_location(conf, 'location = /elecciones')
    assert 'proxy_cache_background_update on;' in history and 'proxy_cache_background_update' not in ' '.join(everything)
    assert set(everything) <= set(history) and len(history) == len(everything) + 1
    assert any(line.startswith('proxy_cache_use_stale') and ' updating ' in line for line in history)
    assert conf.index('location = /elecciones {') < conf.index('location / {')
    assert 'inactive=10m' in conf  # the bound of the age of the copy that is served
    assert subprocess.run(['bash', '-n', 'deploy/nginx-stale-check.sh'], cwd=ROOT).returncode == 0
```

- [ ] **Step 2: Verlo fallar** — `pytest tests/test_deploy_files.py -q -k background` → FAIL.

- [ ] **Step 3: Implementar el bloque** en `nginx.conf`, con un comentario de dos líneas encima: qué hace, que la copia servida tiene como mucho unos 10 minutos (`inactive=10m`) y que el resto de páginas no lo llevan para no enseñar cifras viejas tras una publicación.

- [ ] **Step 4: Escribir `deploy/nginx-stale-check.sh`** (bash, `set -euo pipefail`, cabecera de uso como `nginx-check.sh`). Adapta la configuración con los mismos `sed` que `nginx-check.sh`, más `listen 8042` → `listen 18042` y `127.0.0.1:8000` → `127.0.0.1:18000`; arranca un servidor de Python en el 18000 que tarda 2 s en responder cualquier ruta con `Cache-Control: public, max-age=5` y, en el cuerpo, cuántas veces le han pedido esa ruta; arranca nginx con esa configuración; y comprueba, con `curl -w '%{time_total}'` y la cabecera `X-Cache`:

  | Paso | Petición | Esperado |
  |---|---|---|
  | 1 | `/elecciones?scope=es-md` | `MISS`, ≥ 2 s, cuerpo 1 |
  | 2 | a los 7,5 s del paso 1, la misma | `STALE`, < 0,5 s, cuerpo 1 |
  | 3 | en la misma conexión que el paso 2 (`curl` con dos URL), `/healthz` | < 0,5 s |
  | 4 | 3 s después, la misma del paso 1 | `HIT`, cuerpo 2 |
  | 5 | `/elecciones/es/generales/2023-07`, y otra vez 7,5 s después | `MISS` y luego `EXPIRED` con ≥ 2 s |

  Para nginx y el servidor al salir (`trap`), imprime `OK`/`FAIL` por paso y sale con 1 si alguno falla. Como root (dentro de un contenedor) los workers de nginx no son root: dar permisos al directorio temporal (`chmod -R a+rwX`).

- [ ] **Step 5: Ejecutarlo** — `bash deploy/nginx-check.sh` → `syntax is ok`; `bash deploy/nginx-stale-check.sh` con el nginx local → cinco `OK`. Después, con el nginx de la imagen: `docker run --rm -v "$PWD":/app -w /app python:3.11-slim bash -c "apt-get update -qq && apt-get install -y -qq nginx curl >/dev/null && nginx -v && bash deploy/nginx-stale-check.sh"` → cinco `OK`. Si Docker no está arrancado, decirlo en el informe: el comportamiento queda comprobado sólo con el nginx local y falta la versión de la imagen.

- [ ] **Step 6: Documentar** en `deploy/README.md`: en el párrafo de `/` y `/api/`, que `/elecciones` tiene bloque propio con `proxy_cache_background_update` (copia de hasta unos 10 minutos, refresco por detrás) y por qué sólo ella; y el guion nuevo junto a `nginx-check.sh`.

- [ ] **Step 7: Verlo pasar** — `pytest tests/test_deploy_files.py -q` → PASS.

- [ ] **Step 8: Commit** — «nginx: /elecciones se sirve desde la copia guardada y se refresca por detrás».

---

### Cierre

- [ ] `pytest -q` completo → PASS (los ≈ 2.471 de `dev` menos el test borrado, más los nuevos).
- [ ] Informe para Luis: lecturas y segundos antes y después (Tarea 3, paso 5), resultado de los dos nginx (Tarea 4, paso 5) y los pasos que le quedan: imagen nueva, `update-service` y, ya en producción, repetir la medición con `curl` (home y `/elecciones` con `x-cache: EXPIRED`/`STALE`).
