# Web, fase 6: veda, despliegue en ECS y endurecimiento — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar la web lista para la campaña y la veda del 24 al 28 de noviembre: guarda LOREG para todos los
ámbitos, congelación con pronóstico definitivo visible en la web, sondeos vetados fuera de la API en vivo,
despliegue reproducible en AWS ECS detrás del ALB (web y jobs), ensayos de operación documentados y ejecutados en
local, la deuda del núcleo que afecta a la operación (`HEAD`, despacho local del router, arranque de los
workers, manifest parseado una vez) y a la mantenibilidad (`publish.py` como paquete), y los flecos de la fase 5
que no son de diseño.

**Architecture:** Sin cambios de arquitectura: la misma imagen (nginx + gunicorn, `init.sh`) corre en ECS, el
paquete sigue en S3 y la base en la RDS. La veda se apoya en lo que ya existe (`loreg_guard`, `manifest.freeze`,
`x-freeze`, el banner) y lo amplía: ventana nacional para todos los ámbitos, `freeze.since`, títulos y pie,
filtro de `ctype` en `/api/v1/polls`. ECS solo necesita que nginx confíe en los rangos privados, que s3fs pueda
usar el rol de tarea, workers y logs por variable, y plantillas de definición de tarea. La deuda del núcleo se
paga con cambios pequeños y cubiertos por tests en `mtpy/core/api.py`, `worker.py` y `helpers.py`; `publish.py`
se divide con el mismo patrón de fachada que `webapi` y `pages`. Nada de esto cambia una respuesta de la API ni
una página fuera de lo que la spec fija (títulos y pie congelados, columna "Sesgo del margen").

**Tech Stack:** Python 3.11, pandas, Jinja2 3.1.6, pytest 9 (dobles sin base; integración marcada), nginx
(Homebrew local para `nginx-check.sh`), Docker (si el demonio está arriba), AWS ECS/ALB/ECR (plantillas JSON;
la infraestructura la crea Luis), `hey` o `ab` para la carga.

**Spec:** `docs/superpowers/specs/2026-10-13-web-fase-6-endurecimiento-design.md` (autoridad: decisiones 1-7 de
Luis, supuestos, "La veda", "Despliegue en AWS ECS", "Ensayos de operación", "Deuda del núcleo",
"Mantenibilidad", "Flecos de la fase 5", "Verificación"). Spec general:
`docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Servicio y caché", "Cambios en el núcleo" 4,
"Infraestructura", "Verificación" fase 6, "Estado al cierre de la fase 5" para los menores aplazados). Contrato:
`docs/web/contrato.md`. Runbooks: `deploy/README.md`.

## Estado de partida

Rama `dev` en `04a7a33` (spec de la fase 6 confirmada), árbol limpio, 532 tests unitarios en verde
(`python -m pytest -m "not integration" -q`). La fase 5 está publicada y desplegada. Hechos que las tareas
usan: `publish.loreg_guard(scope, event_date, today=None, force=False)` solo guarda `es`;
`Publish._forecast_scope` la llama con `today` y `force` y `publish_forecast(..., freeze=in_window)` marca
`meta.freeze`; `_analysis_scope` no pasa por la guarda; `update_manifest(writer, scopes=None, freeze=None)`
escribe `freeze: {active, message}`; `Site.freeze()` lo devuelve; `Base.notice_freeze` pone `x-freeze`;
`base.html` pinta `#freeze-banner`; `common_context` pone `title_suffix` "Pronóstico para el {fecha}" o
"Estimación a {as_of}". `webapi.queries.query_polls` llama a `get_poll_series(scope, date)` sin
`drop_contexts` (sirve `wban` y `exit`); el cargador asigna esos `ctype` por el color de la fila de Wikipedia y
`Computer`/`data.py` los excluyen por defecto. `Router.handle` guarda `controller`/`action`/`params` en el
router y `Router.dispatch` los lee; `Controller.dispatch` y `Page.dispatch` duplican el `try/except`;
`add_route` acepta `methods` (por defecto `['GET', 'POST']`) y `parse_route` rechaza el resto (un `HEAD`
acaba en `not_found`). `supervisord.conf` arranca `gunicorn -w 4` con `stdout_logfile=/dev/stdout` bajo un
supervisord demonizado; `init.sh` no exporta nada. `S3.connection()` pasa `aws_params['key']`/`['secret']` tal
cual a `S3FileSystem`. `nginx.conf` confía en `127.0.0.1` y `172.16.0.0/12` para `real_ip`.
`mtpy/core/utils/helpers.py` importa sklearn, scipy y statsmodels al cargar; `stat.py` y `dataviz.py` importan
scipy/statsmodels. `publish.py` tiene 1.894 líneas (definiciones listadas en la tarea 8).

## Decisiones de ejecución (supuestos del plan; Luis puede corregirlos)

- **E1.** La ventana nacional la calcula el job una vez por ejecución (`publish.next_event_date('es')`) y la
  pasa a la guarda como `national`; en los tests, la fixture `patched` ya parchea `next_event_date`.
- **E2.** `freeze.since` lo pone `update_manifest` al activar (hoy en Europe/Madrid, parámetro `today` para
  tests), lo conserva al reactivar con otro mensaje y lo borra al desactivar. Contrato 1 sin cambio de versión.
- **E3.** `asyncio.to_thread` NO entra en esta fase: el despacho local del router sí (bug latente), y el
  estado de cierre deja escrito cómo activar `to_thread` si la prueba de carga en ECS lo pide.
- **E4.** `HEAD` se resuelve en `Router.add_route` (toda ruta `GET` acepta también `HEAD`) y en
  `Response.send_content` (sin cuerpo cuando la petición es `HEAD`), no en cada controlador.
- **E5.** El ensayo de `point`/`unpublish` en local se hace sobre una copia (`files/site.bak`) que se restaura
  al terminar: el paquete local de Luis no se toca.
- **E6.** La comprobación de que el ALB entrega la IP real se hace sin servicios externos: `/healthz` devuelve
  la cabecera `X-Client-IP: $remote_addr` y `check-api.sh` avisa (WARN, no FAIL) si esa IP es privada y la URL
  base no es local.
- **E7.** En el detalle de una elección y en la lista, "Error de bloques" muestra `|error_blocks|`
  (`error_blocks_abs`, calculado en la página, no en el paquete) y la columna nueva "Sesgo del margen" el valor
  con signo; el paquete publicado no cambia.
- **E8.** `publish.py` se divide en `common`, `forecast`, `analysis`, `backtest`, `events` y `pointers`; el
  fichero original se mueve con `git mv` a `forecast.py` (el bloque mayor). Los tests que rompen una función
  interna para probar la atomicidad (`export_error_data`, `export_backtest_table`, `export_event_model`)
  parchean el submódulo que la define.
- **E9.** Orden de ejecución = orden de las tareas; si el plazo aprieta, las tareas 7 y 8 pueden quedar para
  después de la elección (la spec lo permite para `publish.py`).

## Global Constraints

- PEP8 y docstring numpydoc en inglés en `mtpy/`; español en `tests/`, `docs/`, `deploy/*.md`, mensajes de
  los scripts, plantillas y textos visibles; JS: módulos ES, identificadores en inglés, sin `fetch`, sin
  literales `/api/v1`, sin scripts en línea ni `on*=`, sin `innerHTML` con datos.
- `python -m pytest -m "not integration" -q` en verde sin base (hoy 532); `FutureWarning` es error; los tests
  unitarios parchean los lectores en `webapi.queries` y los dobles de `tests/fakes.py`; los de integración
  (marcados) solo leen la RDS y escriben en `FileSystem(tmp_path)`.
- Nunca `python job.py` ni `mtpy.run()`/uvicorn sin `S3_BUCKET=` delante, y `DB_HOST=` vacío para servir en
  local; `point`/`unpublish` solo sobre `files/site.bak`; nunca el nginx local de Luis en el 8080 (usar 8043
  para nginx local y 8001 para uvicorn); nunca `aws` CLI contra la cuenta real.
- Ninguna respuesta de la API ni página cambia salvo lo que fija la spec: títulos y pie con el manifest
  congelado, el enlace del banner, la columna "Sesgo del margen", el orden por `error_blocks_abs`, y la
  supresión de `initial.seats` en `/modelo`. `check-api.sh` sin FAIL antes y después de cada tarea.
- Contrato 1 se mantiene: `freeze` gana `since` (lector tolerante), nada más cambia en los esquemas.
- Un commit por tarea (dos si la tarea lo dice), `git add <rutas>` explícito, mensaje corto en inglés con el
  trailer `Co-Authored-By` del modelo que escribe el commit.
- Rama `dev`, árbol limpio (`04a7a33` o posterior).

## Review Focus

1. **Guarda LOREG** (T1): un autonómico dentro de la ventana de las generales → `refused` para `forecast`,
   relleno y `analysis`; `event`, `backtest`, `manifest`, `point` y `unpublish` siguen; `force` publica y
   marca `meta.freeze` (solo `forecast`); el motivo nombra la elección que cierra la ventana.
2. **Sondeos vetados** (T2): `/api/v1/polls` nunca devuelve filas `wban` ni `exit`, con y sin `pollster`/`limit`;
   la caché por evento no cambia de clave.
3. **Manifest congelado** (T2): todas las páginas muestran el banner con enlace; el modo "Elección" dice
   "Pronóstico definitivo", el modo "Hoy" sigue diciendo "Estimación a"; el pie dice desde cuándo; al
   descongelar, todo vuelve y `since` es `null`.
4. **`HEAD`** (T5): en toda ruta `GET`, las mismas cabeceras y sin cuerpo; en una ruta inexistente, 404 sin
   cuerpo; `POST` sigue como hoy; dos peticiones concurrentes nunca se intercambian `params`.
5. **ECS** (T3): con `AWS_KEY`/`AWS_SECRET` vacíos, `S3FileSystem(key=None, secret=None)`; con valores,
   como hoy; sin `GUNICORN_WORKERS`, 4 workers; nginx confía en los tres rangos privados y `/healthz` enseña
   la IP que ha visto.

---

### Task 1: guarda LOREG para todos los ámbitos, `freeze.since` y runbook de la veda

**Files:**
- Modify: `mtpy/lib/publish.py` (`loreg_guard`, `read_manifest`, `update_manifest`), `mtpy/jobs/Publish.py`
  (`run`, `_forecast_scope`, `_forecast_runs`, `_analysis_scope`, `_analysis_line`), `docs/web/contrato.md`
  ("Punteros" → `manifest.freeze`), `deploy/README.md` ("Guarda LOREG" y nueva sección "Veda" tras "Publicar
  en S3 contra la RDS")
- Test: `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Consumes: `publish._today(today)`, `publish.next_event_date(scope)`, `PublishRefused`, la fixture `patched`
  (parchea `next_event_date` con `{'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08', 'es-ri':
  '2027-05-23'}`) y `test_loreg_window_refuses_es_unless_forced` como modelo.
- Produces:
  - `publish.loreg_guard(scope, event_date, today=None, force=False, national=None) -> bool`: `in_window` si
    `0 <= (event - today).days <= 5` para la elección propia **o** para `national` (fecha ISO de la próxima
    elección de `es`, o `None`); sin `force` lanza `PublishRefused('{scope} {event_date}: inside the LOREG window
    (art. 69.7) of {guard_scope} {guard_date}; pass force=true to publish')` con `guard_scope`/`guard_date` la
    elección que cierra la ventana (la propia tiene prioridad); devuelve `in_window`.
  - `publish.update_manifest(writer, scopes=None, freeze=None, today=None) -> dict`: `freeze` se escribe como
    `{'active', 'message', 'since'}`; `since` = `_today(today).isoformat()` al pasar de inactivo a activo, se
    conserva si ya estaba activo, `None` al desactivar; `read_manifest` devuelve por defecto `{'active': False,
    'message': None, 'since': None}` y añade `'since': None` a un manifest antiguo sin la clave.
  - `Publish.run`: calcula `national = publish.next_event_date('es')` una vez si `what` incluye `forecast` o
    `analysis` (E1) y lo pasa a `_forecast_runs` → `_forecast_scope(..., national=national)` y a
    `_analysis_scope(scope, writer, verbose, today, force, national)`, que llama a `loreg_guard(scope,
    publish.next_event_date(scope), today, force, national)` antes de `publish_analysis` (sin evento próximo:
    `skipped` "no upcoming event", como `forecast`); `refused` con `reason` y línea `'{scope}: analysis refused
    ({reason})'` (`_analysis_line`); `force` no marca nada en el análisis (supuesto de la spec).
  - Contrato: `freeze: {active, message, since}` en "Punteros"; README "Guarda LOREG" reescrita (todos los
    ámbitos, ventana propia o nacional, `forecast`/relleno/`analysis`, `force`) y sección "Veda": runbook del
    23 de noviembre (cargar sondeos → `publish forecast all` → `publish manifest freeze` con el mensaje →
    `check-api.sh https://<host>`), noche del 29 (`freeze` a `false`), después (`publish event`), "Sondeos
    durante la veda" (cargar es seguro: el cargador marca `wban`/`exit`, el modelo y la API los excluyen; no
    recalcular ratings ni publicar hasta el 29) y "Ensayo" (los comandos de T2, paso 5).

- [ ] **Step 1: Tests que fallan**

`tests/test_publish_unit.py`:

```python
def test_loreg_guard_covers_the_national_window_for_every_scope():
    with pytest.raises(publish.PublishRefused, match=r'es-md 2027-05-23: inside the LOREG window \(art\. 69\.7\) of es 2026-11-29'):
        publish.loreg_guard('es-md', '2027-05-23', today='2026-11-25', national='2026-11-29')
    assert publish.loreg_guard('es-md', '2027-05-23', today='2026-11-25', national='2026-11-29', force=True) is True
    assert publish.loreg_guard('es-md', '2027-05-23', today='2026-11-20', national='2026-11-29') is False
    with pytest.raises(publish.PublishRefused, match='of es-md 2026-11-27'):
        publish.loreg_guard('es-md', '2026-11-27', today='2026-11-25')
    with pytest.raises(publish.PublishRefused):
        publish.loreg_guard('es', '2026-11-29', today='2026-11-29')


def test_update_manifest_stamps_since_on_activation(tmp_path):
    writer = make_writer(tmp_path)
    assert publish.read_manifest(writer)['freeze'] == {'active': False, 'message': None, 'since': None}
    on = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda'}, today='2026-11-23')['freeze']
    assert on == {'active': True, 'message': 'Veda', 'since': '2026-11-23'}
    again = publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda (2)'}, today='2026-11-24')['freeze']
    assert again['since'] == '2026-11-23' and again['message'] == 'Veda (2)'
    assert publish.update_manifest(writer, freeze={'active': False})['freeze'] == {'active': False, 'message': None, 'since': None}
```

`tests/test_jobs_publish.py`:

```python
def test_national_window_refuses_regional_forecast_and_analysis_but_not_events(fresh_app, tmp_path, patched, capsys):
    from mtpy.jobs.Publish import Publish
    fs = FileSystem(str(tmp_path))
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-25')
    assert result['es-cb']['status'] == 'refused' and 'of es 2026-11-29' in result['es-cb']['reason'] and patched == []
    result = Publish().run(what=['analysis'], scopes=['es-cb'], fs=fs, today='2026-11-25')
    assert result['es-cb']['status'] == 'refused' and 'es-cb: analysis refused (' in capsys.readouterr().out
    result = Publish().run(what=['forecast'], scopes=['es-cb'], fs=fs, today='2026-11-25', force=True)
    assert result['es-cb']['status'] == 'published' and patched[-1][3]['freeze'] is True
    write_backtest_source(str(tmp_path / 'bt'))
    assert Publish().run(what=['backtest'], scopes=['es'], fs=fs, today='2026-11-25', source=str(tmp_path / 'bt'))['es']['status'] == 'published'
```

(`patched` guarda las llamadas de `publish_forecast` como `(scope, run_id, event_date, kwargs)`; ajustar el
índice a su forma real.) El test existente `test_manifest_merges_scopes_and_freeze` pasa a esperar `since`.

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_publish_unit.py tests/test_jobs_publish.py -q -k "loreg or national or since or manifest"`
Expected: FAIL (`national` desconocido, `since` ausente).

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Suite en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todo en verde.

- [ ] **Step 5: Contrato y README** (secciones descritas en Interfaces)

- [ ] **Step 6: Commit**

```bash
git add mtpy/lib/publish.py mtpy/jobs/Publish.py tests/test_publish_unit.py tests/test_jobs_publish.py docs/web/contrato.md deploy/README.md
git commit -m "Guard every scope inside the national LOREG window and stamp when the manifest was frozen"
```

---

### Task 2: web congelada y sondeos vetados fuera de la API

**Files:**
- Modify: `mtpy/lib/webapi/queries.py` (`WEB_DROP_CTYPES`, `query_polls`), `mtpy/lib/pages/common.py`
  (`common_context`), `web/templates/base.html` (banner), `web/templates/_footer.html`, `deploy/check-api.sh`
  (`check_header`, comprobación de congelación), `docs/web/contrato.md` (fila de `/api/v1/polls` y nota de
  `x-freeze`), `deploy/README.md` ("Veda" → "Ensayo")
- Test: `tests/test_web_api.py`, `tests/test_webapi_unit.py`, `tests/test_pages.py`

**Interfaces:**
- Consumes: `update_manifest(..., freeze=, today=)` (T1), `Site.freeze()`, `Site.manifest_cache_clear()`,
  `test_freeze_banner_and_attribution` (patrón de manifest congelado en `tests/test_pages.py`), los dobles
  `get_poll_series(scope, event)` de `tests/test_web_api.py::fake_readers` y `tests/test_webapi_unit.py`.
- Produces:
  - `queries.WEB_DROP_CTYPES = ('wban', 'exit')`; el cargador de `query_polls` llama a
    `get_poll_series(scope, date, drop_contexts=list(WEB_DROP_CTYPES))`; clave de caché sin cambios. Los
    dobles de los tests aceptan `drop_contexts=None`.
  - `common_context` añade `freeze` (el dict del manifest) y `frozen` (`bool(freeze['active'])`); con
    `frozen` y modo `forecast`, `title_suffix = 'Pronóstico definitivo para el {fecha}'`; `nowcast` no cambia.
  - `base.html`: `#freeze-banner` = mensaje + `<a href="/metodo{{ query }}#limitaciones">Más información</a>`.
    `_footer.html`: con `frozen` y `freeze.since`, añade ` · congelado desde {{ freeze.since|date }}`.
  - `check-api.sh`: `check_header RUTA CABECERA VALOR` (curl `-sI`, FAIL si falta o difiere); tras
    `/api/v1/manifest`, si `freeze.active` es `true` (lectura con `$PY`) comprueba `check_header
    /api/v1/scopes x-freeze active` y que `/` contiene `freeze-banner`; si no, imprime `n/a  congelación: inactiva`.
  - README "Veda" → "Ensayo": `S3_BUCKET= python job.py publish '{"what":["forecast"],"scopes":["es"],"today":"2026-11-25","dry_run":true}'`
    → `refused`; con `"force":true` → publicado en `site-dry` con `meta.freeze: true`; `S3_BUCKET= python
    job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"Ensayo"}}'` sobre el paquete
    local; `S3_BUCKET= DB_HOST= uvicorn api:api --port 8001` + `bash deploy/check-api.sh http://127.0.0.1:8001`
    (cabecera y banner); `freeze` a `false`.

- [ ] **Step 1: Tests que fallan**

`tests/test_web_api.py`:

```python
def test_polls_route_never_serves_banned_contexts(db_api, fresh_app, monkeypatch):
    seen = []
    def poll_series(scope, event, drop_contexts=None):
        seen.append(drop_contexts)
        return pd.DataFrame({'date': ['2026-10-01'], 'pollster_id': [1], 'ctype': [None], 'PP': [30.0]})
    monkeypatch.setattr(webapi.queries, 'get_poll_series', poll_series)
    webapi.queries._QUERIES.clear()
    status, _, body = call(db_api, '/api/v1/polls', query='scope=es&limit=5')
    assert status == 200 and seen == [['wban', 'exit']]
```

`tests/test_pages.py`:

```python
def test_frozen_manifest_marks_every_page(api, fresh_app):
    writer = bundle.BundleWriter(fresh_app.fs)
    publish.update_manifest(writer, freeze={'active': True, 'message': 'Veda electoral'}, today='2026-11-23')
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/', query='scope=es')
    page = html(body)
    assert status == 200 and 'Pronóstico definitivo para el' in page and 'congelado desde 23 nov 2026' in page
    assert 'freeze-banner' in page and '/metodo?scope=es&amp;mode=forecast#limitaciones' in page
    status, _, body = call(api, '/', query='scope=es&mode=nowcast')
    assert 'Estimación a' in html(body) and 'Pronóstico definitivo' not in html(body)
    status, _, body = call(api, '/metodo', query='scope=es')
    assert 'freeze-banner' in html(body)
    publish.update_manifest(writer, freeze={'active': False})
    webapi.site().manifest_cache_clear()
    status, _, body = call(api, '/', query='scope=es')
    assert 'freeze-banner' not in html(body) and 'Pronóstico para el' in html(body) and 'congelado desde' not in html(body)
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_web_api.py tests/test_pages.py -q -k "banned or frozen"`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces** (ajustar los dobles existentes de `get_poll_series` a la firma con `drop_contexts`)

- [ ] **Step 4: Suite y ensayo local**

Run: `python -m pytest -m "not integration" -q` → verde. Después el ensayo de la veda del README (paso 5 de la
sección "Ensayo", en local con `S3_BUCKET=`): anotar en el informe la salida de `check-api.sh` con el manifest
local congelado y descongelado (`files/site/v1/manifest.json` queda como estaba: `freeze` inactivo).

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/webapi/queries.py mtpy/lib/pages/common.py web/templates/base.html web/templates/_footer.html deploy/check-api.sh docs/web/contrato.md deploy/README.md tests/test_web_api.py tests/test_webapi_unit.py tests/test_pages.py
git commit -m "Show the definitive forecast while frozen and keep banned polls out of the live API"
```

---

### Task 3: despliegue en AWS ECS

**Files:**
- Modify: `deploy/docker/nginx.conf` (`set_real_ip_from`, `/healthz`), `deploy/docker/supervisord.conf`
  (`[program:gunicorn]`), `deploy/docker/init.sh` (`GUNICORN_WORKERS`), `mtpy/core/services/s3.py`
  (`connection`), `.env.example` (`GUNICORN_WORKERS=`), `deploy/check-api.sh` (IP del cliente),
  `deploy/README.md` ("Fichero de configuración", "Despliegue en ECS", "Publicar desde ECS")
- Create: `deploy/ecs/task-web.example.json`, `deploy/ecs/task-job.example.json`,
  `deploy/ecs/run-task.example.json`, `tests/test_s3_unit.py`, `tests/test_deploy_files.py`
- Test: los dos nuevos; `bash deploy/nginx-check.sh`; `bash -n deploy/docker/init.sh`

**Interfaces:**
- Consumes: `S3(aws_params, bucket)` (`mtpy/mtpy.py:83` pasa `config.aws.to_dict()` = `{'key', 'secret',
  'region'}` con cadenas vacías cuando las variables no están), `nginx-check.sh` (comprueba la sintaxis con el
  nginx local), `check` y `check_nginx_only` de `check-api.sh`.
- Produces:
  - `S3.connection()`: `S3FileSystem(key=self.aws_params.get('key') or None, secret=self.aws_params.get('secret') or None)`.
  - `supervisord.conf`: `command=gunicorn -b 127.0.0.1:8000 -w %(ENV_GUNICORN_WORKERS)s -k uvicorn.workers.UvicornWorker --access-logfile /proc/1/fd/1 --error-logfile /proc/1/fd/2 api:api`;
    `init.sh`: `export GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"` antes de `supervisord -c`.
  - `nginx.conf`: `set_real_ip_from 127.0.0.1; 10.0.0.0/8; 172.16.0.0/12; 192.168.0.0/16` (comentario: ALB
    en la VPC); `location = /healthz` añade `add_header X-Client-IP $remote_addr always;`.
  - `check-api.sh`: con nginx (`/healthz` 200) lee `X-Client-IP`, imprime `info cliente según nginx: <ip>` y
    `WARN` (no FAIL) si la IP es privada o de loopback y el host de `BASE_URL` no es `127.0.0.1`/`localhost`.
  - `deploy/ecs/task-web.example.json`: `family` `elections-web`, Fargate, `cpu` `1024`, `memory` `2048`,
    contenedor `web` con la imagen `<account>.dkr.ecr.<region>.amazonaws.com/elections-web:<tag>`, `portMappings`
    8042, `environment` con `APP_ENV`, `APP_API=1`, `APP_VERBOSE`, `DB_ADAPTER`, `DB_HOST`, `DB_PORT`,
    `DB_USERNAME`, `DB_DATABASE`, `DB_STATEMENT_TIMEOUT=10000`, `AWS_REGION`, `S3_BUCKET`, `CWL_GROUP`,
    `GUNICORN_WORKERS=2` (sin `AWS_KEY`/`AWS_SECRET`: rol de tarea), `secrets` `DB_PASSWORD` con `valueFrom`
    `<secret-arn>`, `logConfiguration` `awslogs`, `taskRoleArn` `<task-role-read>`, `executionRoleArn`;
    `task-job.example.json`: `elections-job`, `2048`/`4096`, sin `APP_API`, `RUN_JOB` vacío, rol
    `<task-role-write>`; `run-task.example.json`: `overrides.containerOverrides[0].environment` con
    `RUN_JOB=publish {"what":["forecast"],"scopes":"all"}` (JSON sin espacios). Placeholders entre `<>`;
    comentarios no (JSON puro); README explica cada placeholder.
  - README "Despliegue en ECS": qué debe existir (cluster, ECR, servicio, target group con health check
    `/api/v1/health` cada 30 s y 2 fallos, listener del ALB, grupos de seguridad con acceso a la RDS, rol de
    tarea de lectura `s3:GetObject`/`s3:ListBucket` sobre `site/*`, rol de ejecución, grupo de logs,
    secreto de `DB_PASSWORD`), el procedimiento (`docker build --platform linux/amd64`, `docker tag`,
    `docker push`, registrar la definición desde la plantilla, `aws ecs update-service --force-new-deployment`,
    `bash deploy/check-api.sh https://<host>`), y que `deploy/elections.env` sigue siendo el entorno del
    portátil, no el de la web. "Publicar desde ECS": `aws ecs run-task` con `run-task.example.json`, dónde ver
    la salida (`awslogs`), que el rol de jobs escribe en `site/*` y `site-dry/*`, y el aviso de medir la memoria
    de `publish forecast all` (T4 lo mide en local).

- [ ] **Step 1: Tests que fallan**

`tests/test_s3_unit.py`:

```python
def test_empty_credentials_fall_back_to_the_botocore_chain(monkeypatch):
    seen = {}
    monkeypatch.setattr('mtpy.core.services.s3.S3FileSystem', lambda **kw: seen.update(kw) or object())
    S3({'key': '', 'secret': '', 'region': 'eu-west-1'}, 'bucket').connect()
    assert seen == {'key': None, 'secret': None}
    S3({'key': 'AK', 'secret': 'SK', 'region': 'eu-west-1'}, 'bucket').connect()
    assert seen == {'key': 'AK', 'secret': 'SK'}
```

`tests/test_deploy_files.py`:

```python
def test_ecs_templates_are_valid_and_consistent():
    web = load('deploy/ecs/task-web.example.json'); job = load('deploy/ecs/task-job.example.json'); run = load('deploy/ecs/run-task.example.json')
    env = {e['name']: e['value'] for e in web['containerDefinitions'][0]['environment']}
    assert env['GUNICORN_WORKERS'] == '2' and env['DB_STATEMENT_TIMEOUT'] == '10000' and 'AWS_KEY' not in env
    assert web['containerDefinitions'][0]['portMappings'][0]['containerPort'] == 8042
    assert job['memory'] == '4096' and all(e['name'] != 'APP_API' for e in job['containerDefinitions'][0]['environment'])
    assert run['overrides']['containerOverrides'][0]['environment'][0]['name'] == 'RUN_JOB'


def test_supervisord_and_init_take_the_workers_from_the_environment():
    conf = read('deploy/docker/supervisord.conf'); init = read('deploy/docker/init.sh')
    assert '-w %(ENV_GUNICORN_WORKERS)s' in conf and '/proc/1/fd/1' in conf
    assert 'GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"' in init
    assert subprocess.run(['bash', '-n', 'deploy/docker/init.sh'], cwd=ROOT).returncode == 0


def test_nginx_trusts_the_private_ranges_and_reports_the_client():
    conf = read('deploy/docker/nginx.conf')
    for cidr in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'):
        assert 'set_real_ip_from {};'.format(cidr) in conf
    assert 'X-Client-IP $remote_addr always' in conf
```

- [ ] **Step 2: Comprobar que fallan**

Run: `python -m pytest tests/test_s3_unit.py tests/test_deploy_files.py -q`
Expected: FAIL.

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Verificación**

Run: `python -m pytest -m "not integration" -q` → verde; `bash deploy/nginx-check.sh` → OK; `bash
deploy/check-api.sh http://127.0.0.1:8043` con el nginx local (si está configurado como en la fase 2) → línea
`info cliente según nginx`; `docker build --platform linux/amd64 -t elections-web:f6 .` y `bash deploy/smoke.sh`
solo si el demonio está arriba (si no, "no verificado: Docker" en el informe).

- [ ] **Step 5: Commit**

```bash
git add deploy/docker/nginx.conf deploy/docker/supervisord.conf deploy/docker/init.sh mtpy/core/services/s3.py .env.example deploy/check-api.sh deploy/ecs deploy/README.md tests/test_s3_unit.py tests/test_deploy_files.py
git commit -m "Prepare the image for ECS: private ranges, task-role credentials, workers and logs by environment, task templates"
```

---

### Task 4: ensayos de operación

**Files:**
- Create: `deploy/load-test.sh`
- Modify: `deploy/README.md` (sección "Ensayos de operación": carga, `point`/`unpublish`, congelación, copia
  y restauración; "Duraciones medidas" gana las cifras locales)
- Test: `bash -n deploy/load-test.sh`; `tests/test_deploy_files.py` (el script existe y es ejecutable con `bash -n`)

**Interfaces:**
- Consumes: `hey` (si está instalado) o `ab`; `check-api.sh`; el runbook de `point`/`unpublish` del README
  ("Retirada de un run"); `pg_dump`/`pg_restore` de Homebrew.
- Produces:
  - `deploy/load-test.sh BASE_URL [CONCURRENCIA=50] [SEGUNDOS=20]`: con `hey` (`-z ${SEGUNDOS}s -c $CONCURRENCIA
    -q 5`) o, si no, `ab -t $SEGUNDOS -c $CONCURRENCIA`, sobre `/api/v1/forecast/es/nowcast/summary` y `/`;
    imprime p95 y códigos de respuesta; después 50 `curl -sI` a la ruta de la API contando `X-Cache: HIT`; sale
    con 1 si hay 5xx o si `BASE_URL` responde 429 a la primera petición. Mensajes en español. `-q 5` por
    petición mantiene cada cliente por debajo del límite de 20 r/s por IP; el README lo explica.
  - README "Ensayos de operación": (1) carga: cómo lanzarlo contra ECS (`https://<host>`) y contra local
    (`http://127.0.0.1:8043`), criterios de la spec (p95 API < 200 ms, páginas < 500 ms, mayoría `HIT`, sin
    5xx) y qué hacer si no se cumplen (`to_thread`, E3); (2) `point`/`unpublish` en local sobre
    `files/site.bak` (E5): copiar `files/site` → `files/site.bak`, `S3_BUCKET= python job.py publish
    '{"what":["point"],...}'` y `unpublish` sobre un run de la copia, comprobar manifest e `history.json`,
    restaurar; (3) congelación: enlace a "Veda" → "Ensayo"; (4) copia y restauración: `pg_dump -Fc -n
    elections -h <rds> -U <user> elections > elections-$(date +%F).dump`, subida con `aws s3 cp` a
    `s3://<bucket>/backups/`, `pg_restore -d <local> --clean --if-exists elections-<fecha>.dump`, y
    `python -m pytest -m integration -q` contra la base restaurada como criterio.
  - "Duraciones medidas": p95 y HIT del ensayo local (uvicorn :8001 sin caché y, si hay nginx local, :8043),
    y el RSS de `S3_BUCKET= python job.py publish '{"what":["backtest"],"scopes":["es"]}'` (único job sin
    base; `/usr/bin/time -l`) como referencia; la memoria de `publish forecast all` la mide Luis desde ECS
    (README lo dice).

- [ ] **Step 1: Test que falla**

`tests/test_deploy_files.py`:

```python
def test_load_test_script_is_valid_bash():
    assert os.path.isfile(os.path.join(ROOT, 'deploy', 'load-test.sh'))
    assert subprocess.run(['bash', '-n', 'deploy/load-test.sh'], cwd=ROOT).returncode == 0
```

- [ ] **Step 2: Comprobar que falla** — Run: `python -m pytest tests/test_deploy_files.py -q -k load_test` → FAIL.

- [ ] **Step 3: Script, README y ensayos locales** (`point`/`unpublish` sobre la copia; carga contra uvicorn
  :8001 con `S3_BUCKET= DB_HOST=`; anotar cifras; restaurar `files/site`)

- [ ] **Step 4: Verificación** — Run: `python -m pytest tests/test_deploy_files.py -q` → verde; `bash
  deploy/load-test.sh http://127.0.0.1:8001 10 5` termina con código 0 y cifras.

- [ ] **Step 5: Commit**

```bash
git add deploy/load-test.sh deploy/README.md tests/test_deploy_files.py
git commit -m "Document and script the operation rehearsals: load, pointers, freeze, backup"
```

---

### Task 5: núcleo: `HEAD`, despacho local del router y gancho de errores

**Files:**
- Modify: `mtpy/core/api.py` (`Api.__call__`, `Response`, `Router.add_route`, `Router.handle`,
  `Router.dispatch`, `Controller.dispatch`), `mtpy/controllers/Page.py` (`dispatch` → `error_result`)
- Test: `tests/test_api_asgi.py`, `tests/test_pages.py` (los tests de error existentes), `tests/test_web_api.py`

**Interfaces:**
- Consumes: `Router.parse_route(route, uri, method)`, `Response.send_content(result)`, `Controller.dispatch`,
  `Page.error_page(status, message)`, el arnés `call(api, path, query='', method='GET')`.
- Produces:
  - `Router.add_route(pattern, controller, action='index', methods=None)`: si `GET` está en `methods`, añade
    `HEAD` (E4); `ROUTES` de `webapi`/`pages` no cambian.
  - `Response.head: bool` (lo fija `Api.__call__` con `request.method == 'HEAD'`); `send_content` envía las
    mismas cabeceras (`content-type`, `content-length` del cuerpo que habría enviado, `etag`, `cache-control`)
    y un cuerpo vacío cuando `head` es `True`.
  - `Router.handle(uri=None) -> tuple[Controller, str, dict]` devuelve `(controller, action, params)` sin
    guardarlos en el router (los atributos `controller`/`action`/`params` desaparecen o quedan sin usar);
    `Router.dispatch(target)` recibe la tupla; `Api.__call__`: `await router.dispatch(await router.handle())`.
  - `Controller.dispatch(action, **kwargs)`: un solo `try/except` que, ante `HttpError`, hace `self.result =
    self.error_result(error.status, error.message)` y, ante otra excepción, registra la traza (como hoy hace
    `Page`) y `self.result = self.error_result(500, 'Error interno')`; `Controller.error_result(status,
    message)` por defecto `self.response._set_error(status, message)` y devuelve `{'status': 'error',
    'message': message}` (para 500: el mensaje genérico de hoy, `'500 Internal Server Error'`); `Page.error_result`
    devuelve `self.error_page(status, message)`; `Page.dispatch` se elimina.

- [ ] **Step 1: Tests que fallan**

`tests/test_api_asgi.py`:

```python
def test_head_returns_the_get_headers_without_a_body(fresh_app):
    api = make_api()  # el arnés del fichero: Echo en /echo/{id:num}
    get_status, get_headers, get_body = call(api, '/echo/7')
    status, headers, body = call(api, '/echo/7', method='HEAD')
    assert status == get_status == 200 and body == b'' and headers['content-type'] == get_headers['content-type']
    assert headers['content-length'] == str(len(get_body))
    status, _, body = call(api, '/nope', method='HEAD')
    assert status == 404 and body == b''
    status, _, _ = call(api, '/echo/7', method='POST')
    assert status == 404


def test_concurrent_requests_keep_their_own_params(fresh_app):
    # dos peticiones a /echo/{id} entrelazadas con asyncio.gather; la acción cede el control con asyncio.sleep(0)
    results = asyncio.run(gather_calls(api, ['/echo/1', '/echo/2']))
    assert [json.loads(body)['id'] for _, _, body in results] == ['1', '2']
```

(`gather_calls` se añade al arnés: construye los `scope`/`receive`/`send` de cada petición y ejecuta
`asyncio.gather(api(...), api(...))`. Comprobar con `git stash` que el segundo test falla o es inestable
ANTES del cambio del router; si no se puede provocar la intercalación, el test queda como guarda del diseño y
el informe lo dice.)

`tests/test_pages.py`: los tests de error existentes (`test_page_errors_are_html_and_not_cached`,
`test_unexpected_error_is_a_generic_500_page_and_is_logged`) siguen en verde sin cambios; `tests/test_web_api.py`:
`test_errors_are_json_and_not_cached` ídem.

- [ ] **Step 2: Comprobar que fallan** — Run: `python -m pytest tests/test_api_asgi.py -q -k "head or concurrent"` → FAIL.

- [ ] **Step 3: Implementar según Interfaces**

- [ ] **Step 4: Verificación** — Run: `python -m pytest -m "not integration" -q` → verde; `S3_BUCKET= DB_HOST=
  uvicorn api:api --port 8001` + `curl -sI http://127.0.0.1:8001/api/v1/manifest` → `200` con `etag` y sin
  cuerpo; `bash deploy/check-api.sh http://127.0.0.1:8001` sin FAIL.

- [ ] **Step 5: Commit**

```bash
git add mtpy/core/api.py mtpy/controllers/Page.py tests/test_api_asgi.py
git commit -m "Accept HEAD, dispatch on local state and share the error handling of the controllers"
```

---

### Task 6: núcleo: arranque de los workers, manifest parseado una vez, `worker.py`

**Files:**
- Modify: `mtpy/core/utils/helpers.py` (imports perezosos), `mtpy/core/utils/io.py` (si importa `helpers`
  o pesados al cargar), `mtpy/lib/webapi/published.py` (`Site.manifest`, `manifest_data`), `mtpy/core/worker.py`
  (`job_class`), `deploy/README.md` ("Duraciones medidas": RSS y arranque)
- Test: `tests/test_webapi_unit.py` (`test_webapi_import_stays_light`, nuevo test del manifest),
  `tests/test_worker_unit.py` (nuevo)

**Interfaces:**
- Consumes: `Router.controller_class` (patrón: si el atributo del paquete es un módulo, tomar su atributo
  homónimo), `TTLCache.get(key, loader)`, `bundle.loads`.
- Produces:
  - `helpers.py`: `sklearn`, `scipy.signal`, `scipy.stats`, `statsmodels` se importan dentro de las funciones
    que los usan (una importación local por función, sin cambiar firmas); si `io.py` arrastra `helpers` para
    algo que no lo necesita al cargar, lo mismo. Medida: `python -X importtime -c "import api" 2>&1 | tail -1`
    y `python -c "import api, resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"` antes y
    después (anotar en README).
  - `test_webapi_import_stays_light` comprueba además que `sklearn` y `statsmodels` no están en `sys.modules`
    tras `import mtpy.lib.webapi` y `import mtpy.lib.pages` (si `scipy` sigue cargando por `stat.py`, el
    informe lo dice y la aserción no lo incluye).
  - `Site.manifest()` y `Site.manifest_data()` comparten una entrada `'manifest'` de `pointers` que guarda
    `(bytes, data)`; `manifest()` devuelve `[0]`, `manifest_data()` `[1]`; `bundle.loads` se ejecuta una vez
    por TTL (test con `monkeypatch` que cuenta las llamadas: dos `manifest_data()` y un `manifest()` seguidos →
    una llamada).
  - `Worker.job_class(name) -> type`: `getattr(self.namespace, name)` y, si el resultado es un módulo,
    `getattr(module, name)`; `run`/`dispatch` del worker lo usan donde hoy hacen `getattr(self.namespace,
    name)()`. Test con un `SimpleNamespace` cuyo atributo `Publish` es un módulo falso con una clase
    `Publish` dentro.

- [ ] **Step 1: Tests que fallan**

`tests/test_webapi_unit.py`:

```python
def test_manifest_is_parsed_once_per_ttl(site_app, monkeypatch):
    calls = []
    real = bundle.loads
    monkeypatch.setattr(webapi.published.bundle, 'loads', lambda raw: calls.append(1) or real(raw))
    s = webapi.site()
    s.manifest_cache_clear()
    s.manifest_data(); s.manifest_data(); s.manifest()
    assert len(calls) == 1 and s.manifest_data()['scopes'] and isinstance(s.manifest(), bytes)


def test_webapi_import_stays_light():  # ampliado
    ...  # además de data/publish/simulator: 'sklearn' y 'statsmodels' ausentes de sys.modules tras
    ...  # `import mtpy.lib.webapi, mtpy.lib.pages` (scipy solo si el informe confirma que no carga)
```

`tests/test_worker_unit.py`:

```python
def test_job_class_unwraps_a_module_bound_on_the_namespace():
    """Como `Router.controller_class`: si el paquete `jobs` tiene enlazado el submódulo en vez de la clase."""
    class Publish: ...
    module = types.ModuleType('mtpy.jobs.Publish'); module.Publish = Publish
    worker = Worker.__new__(Worker); worker.namespace = types.SimpleNamespace(Publish=module, Other=Publish)
    assert worker.job_class('Publish') is Publish and worker.job_class('Other') is Publish
```

- [ ] **Step 2: Comprobar que fallan** — Run: `python -m pytest tests/test_webapi_unit.py tests/test_worker_unit.py -q -k "manifest_is_parsed or job_class or stays_light"` → FAIL.

- [ ] **Step 3: Implementar según Interfaces** y medir (antes/después)

- [ ] **Step 4: Verificación** — Run: `python -m pytest -m "not integration" -q` → verde; `check-api.sh`
  local sin FAIL; cifras en el README.

- [ ] **Step 5: Commit**

```bash
git add mtpy/core/utils/helpers.py mtpy/core/utils/io.py mtpy/lib/webapi/published.py mtpy/core/worker.py tests/test_webapi_unit.py tests/test_worker_unit.py deploy/README.md
git commit -m "Lighter workers: lazy heavy imports, one manifest parse per TTL, module-safe job lookup"
```

---

### Task 7: flecos de la fase 5

**Files:**
- Modify: `mtpy/lib/pages/common.py` (`bundle_part`), `mtpy/lib/pages/casas.py`, `elecciones.py`, `modelo.py`
  (usar `bundle_part`; `error_blocks_abs`; sin `initial.seats`), `web/templates/eleccion.html`,
  `elecciones.html` (columna y textos), `web/dist/js/pages/elecciones.js` (barras por `error_blocks_abs`),
  `web/dist/js/pages/modelo.js` (nada de `seats`), `web/templates/casas.html` (texto del rating),
  `mtpy/lib/webapi/__init__.py` (imports ≤ 120 columnas, alfabéticos), `deploy/check-api.sh` (comentario de
  `check_published`), `tests/test_pages.py`, `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`,
  `tests/test_web_api.py`, `tests/test_webapi_unit.py`

**Interfaces:**
- Consumes: `analysis_part`/`event_part`/`backtest_part` (mismo cuerpo "None en 404"), `pollster_rows`/
  `poll_rows`/`published_events` de `elecciones.py`, `renderBars` (campo `value` configurable), `fmt_num`.
- Produces:
  - `pages.common.bundle_part(read, *args) -> Optional[dict]`: `bundle.loads(read(*args))['data']`, `None`
    si `read` lanza `HttpError` 404, cualquier otra excepción sube; `analysis_part(scope, name)`,
    `event_part(scope, date, name)` y `backtest_part(scope, name)` quedan como envoltorios de una línea.
  - `elecciones.py`: cada fila de `pollster_rows` y `poll_rows` y cada evento de `published_events` gana
    `error_blocks_abs` (`abs(error_blocks)` o `None`); `pollster_rows` se ordena por `error_blocks_abs`
    ascendente con nulos al final; `initial.pollsters` lleva `error_blocks_abs` y `elecciones.js` dibuja las
    barras con `value: 'error_blocks_abs'`; `initial.sector` lleva `error_blocks_abs`. Plantillas: "Error de
    bloques" muestra `error_blocks_abs|num(1)` y la columna nueva "Sesgo del margen" `error_blocks|num(1)` (con
    signo) en la tabla de casas y en la de sondeos del detalle; el texto "Qué es esto" explica que el sesgo
    del margen es la diferencia con signo entre el margen de bloques del sondeo y el del resultado.
  - `casas.html` (texto del rating): "combina el sesgo de sus sondeos frente al resultado, comparado con el de
    las demás casas en las mismas semanas (un fallo compartido por todo el sector no penaliza a una casa), con
    una calidad previa…" (alineado con `/metodo`).
  - `modelo_context` sin `initial['seats']`; `modelo.js` sin referencia.
  - Tests nuevos: `test_eleccion_detail_orders_houses_by_the_absolute_block_error` (patched `event_part` con
    `error_blocks` `[-3.0, 1.0, 2.0]` → orden `[1.0, 2.0, -3.0]` por nombre, columna "Sesgo del margen" con
    `-3,0`), `test_casas_explains_the_relative_bias` (`houses` fixture: la frase "comparado con el de las demás
    casas"), `test_prob_columns_stay_float_when_integral` (`export_backtest_table` con `cov_seats95` todo
    `1.0` → `1.0` flotante, no `1` entero), `test_event_all_and_invalid_list_through_the_job` (`event_date='all'`
    → `published`/`skipped` por fecha; lista con `'2023-13-01'` → una entrada `failed` con `invalid`),
    `test_model_csv_of_an_event_without_backtest_is_404` (`write_event(fs, model=False)` → `model?format=csv&table=seats`
    → 404 `event not published`), `test_site_events_expire_with_the_ttl` (reloj inyectado), `test_pollster_row_without_id_is_plain_text`
    (`pollster_id` `None` → sin enlace); `headers` sin usar eliminados; `patched` deja de parchear `provenance`
    globalmente (lo hacen los tests de `backtest`/`event` que lo necesitan).

- [ ] **Step 1: Tests que fallan** (los siete anteriores más la ausencia de `initial.seats` en el test de `/modelo`)

- [ ] **Step 2: Comprobar que fallan** — Run: `python -m pytest tests/test_pages.py tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_web_api.py tests/test_webapi_unit.py -q -k "absolute or relative_bias or stay_float or invalid_list or without_backtest_is_404 or events_expire or plain_text"` → FAIL.

- [ ] **Step 3: Implementar según Interfaces**; `node --check web/dist/js/pages/elecciones.js web/dist/js/pages/modelo.js`

- [ ] **Step 4: Verificación** — Run: `python -m pytest -m "not integration" -q` → verde; `check-api.sh` local sin FAIL.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/pages mtpy/lib/webapi/__init__.py web/templates web/dist/js/pages deploy/check-api.sh tests
git commit -m "Close the phase 5 leftovers: absolute block error, shared bundle_part, missing tests"
```

---

### Task 8: `publish.py` como paquete `mtpy/lib/publish/`

**Files:**
- Move: `mtpy/lib/publish.py` → `mtpy/lib/publish/forecast.py` (`git mv`)
- Create: `mtpy/lib/publish/__init__.py`, `common.py`, `analysis.py`, `backtest.py`, `events.py`, `pointers.py`
- Modify: `tests/test_publish_unit.py`, `tests/test_jobs_publish.py`, `tests/test_webapi_unit.py` (destinos de
  `monkeypatch`), `deploy/README.md` ("Publicar": el paquete), `docs/web/contrato.md` (cabecera: fuente de
  verdad en `mtpy/lib/publish/`)
- Test: `tests/test_publish_unit.py` (disposición), toda la suite

**Interfaces:**
- Consumes: la lista de definiciones de `publish.py` (línea: nombre): 43 `REPO_ROOT`, 45-68 constantes de
  redondeo y columnas, 71 `records`, 89 `round_cols`, 112-273 exportadores del run (`_summary_table`,
  `_p_majority`, `export_vote`…`headline_mode`), 295-419 (`_stat_bound`, `export_series`, `export_polls`,
  `export_fan`, `export_house_effects`, `export_dispersion`, `n_polls`), 435 `_git`, 443 `provenance`, 468
  `_meta_parties`, 476 `export_meta`, 533 `export_headline`, 557 `_iso_dates`, 562-702 exportadores del
  análisis y `export_analysis_meta`, 702-805 backtest (`BACKTEST_FILES`…`export_backtest_meta`), 805
  `ATTRIBUTION`, 809 `PublishRefused`, 813 `NothingToPublish`, 817 `_madrid_tz`, 834 `_today`, 856
  `loreg_guard`, 891-965 `backfill_*`, 965 `resolve_scopes`, 999 `next_event_date`, 1015 `db_stats`, 1035
  `MODE_EXPORTERS`, 1039 `publish_mode`, 1064 `latest_entry`, 1081 `publish_forecast`, 1198
  `publish_analysis`, 1287 `publish_backtest`, 1350-1722 eventos (`EVENT_POLLSTER_COLUMNS`…`publish_event`),
  1722 `rebuild_history`, 1743 `read_manifest`, 1772 `_write_manifest`, 1781 `update_manifest`, 1812
  `_check_run`, 1831 `point`, 1858 `unpublish` (las líneas son las de `04a7a33`; T1 añade `national`/`since`).
- Produces:
  - `common.py`: `REPO_ROOT`, `PCT_DECIMALS`, `PROB_DECIMALS`, `SEATS_DECIMALS`, `ISO_DAY_RE`,
    `PROVENANCE_PACKAGES`, `records`, `round_cols`, `_git`, `provenance`, `_iso_dates`, `ATTRIBUTION`,
    `PublishRefused`, `NothingToPublish`, `_madrid_tz`, `_today`, `loreg_guard`, `resolve_scopes`,
    `next_event_date`, `db_stats`, `_weighted_mean`, `_iso_day`.
  - `forecast.py`: `PCT_COLS`, `PROB_COLS`, `SEATS_COLS`, `POLL_COLUMNS`, `HE_COLUMNS`, `DISP_COLUMNS`,
    `DEFAULT_CORRECTORS`, `SUMMARY_COLS`, los exportadores del run, `export_meta`, `export_headline`,
    `backfill_days`, `backfill_run_id`, `backfill_run_at`, `MODE_EXPORTERS`, `publish_mode`, `latest_entry`,
    `publish_forecast`.
  - `analysis.py`: `ANALYSIS_*_COLUMNS`, `POLL_ERRORS_COLUMNS`, `ERROR_DATA_COLUMNS`, `POLL_ERRORS_PARAMS`,
    `export_analysis_*`, `export_poll_errors`, `export_error_data`, `publish_analysis`.
  - `backtest.py`: `BACKTEST_FILES`, `PROB_PREFIXES`, `BACKTEST_BOOL_COLS`, `BACKTEST_RUN_EXCLUDED`,
    `backtest_source`, `export_backtest_table`, `export_backtest_meta`, `publish_backtest`.
  - `events.py`: `EVENT_POLLSTER_COLUMNS`, `EVENT_MODEL_PARTS`, `_records`, `_present_blocks`,
    `export_event_*`, `event_summary`, `event_dates_to_publish`, `publish_event`.
  - `pointers.py`: `rebuild_history`, `read_manifest`, `_write_manifest`, `update_manifest`, `_check_run`,
    `point`, `unpublish`.
  - `__init__.py`: docstring, reexportación explícita de todos los nombres públicos (los tests usan además
    `publish.REPO_ROOT`); ningún submódulo se llama como un nombre exportado (`forecast` ≠ `publish_forecast`,
    `events` ≠ `event_summary`…; comprobar `backtest` frente a `backtest_source` y `common` — no hay colisión).
    Test de disposición como `test_webapi_package_layout` (submódulos `['analysis', 'backtest', 'common',
    'events', 'forecast', 'pointers']`, `inspect.ismodule`, `__module__` de cuatro nombres).
  - `monkeypatch`: `publish.publish_forecast`/`publish_analysis`/`publish_event`/`publish_backtest`,
    `resolve_scopes`, `next_event_date`, `provenance` siguen parcheándose en el paquete (el job los busca
    ahí); `export_error_data` → `publish.analysis.export_error_data`, `export_backtest_table` →
    `publish.backtest.export_backtest_table`, `export_event_model` → `publish.events.export_event_model`
    (E8; grep `monkeypatch.setattr(publish,` para encontrarlos todos).
  - `test_webapi_import_stays_light` sigue comprobando `mtpy.lib.publish` (nombre del paquete).

- [ ] **Step 1: Referencia byte a byte ANTES de tocar nada**

Script en el espacio de trabajo (no se commitea): con `S3_BUCKET=`, `publish_forecast('es', writer, run_id='20260101-000000', event_date='2026-11-29', simulator=FakeSimulator(), stats=…, prov=…)` sobre un `FileSystem(tmp)` con `fixed_clock`, más `publish_analysis` con `FakeComputer`, `publish_backtest` desde `write_backtest_source` y `publish_event` con `synthetic_event`, guardando el árbol en `ref/`.

- [ ] **Step 2: Test de disposición que falla** — Run: `python -m pytest tests/test_publish_unit.py -q -k layout` → FAIL.

- [ ] **Step 3: Dividir** (`git mv` + recorte literal; imports entre submódulos solo de `common` hacia arriba:
  `forecast`, `analysis`, `backtest`, `events`, `pointers` importan `common`; `events` importa `backtest`
  (`export_backtest_table`, `backtest_source`) y `analysis` (`export_poll_errors`); nada importa `__init__`)

- [ ] **Step 4: Verificación** — Run: `python -m pytest -m "not integration" -q` → verde; repetir el script
  del paso 1 en `new/` y `diff -r ref new` → sin diferencias (mismo reloj fijo, mismos dobles); `S3_BUCKET=
  python job.py publish '{"what":["backtest"],"scopes":["es"],"dry_run":true}'` → `published`.

- [ ] **Step 5: Commit**

```bash
git add mtpy/lib/publish tests/test_publish_unit.py tests/test_jobs_publish.py tests/test_webapi_unit.py deploy/README.md docs/web/contrato.md
git commit -m "Split publish into a package: common, forecast, analysis, backtest, events and pointers"
```

---

### Task 9: cierre de la fase 6

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` ("Estado al cierre de la fase 6";
  fila 6 de "Fases"; "Infraestructura" nota ECS), `docs/superpowers/specs/2026-10-13-web-fase-6-endurecimiento-design.md`
  ("Estado al cierre" breve con enlaces), `deploy/README.md` (índice y repaso final: "Despliegue en ECS",
  "Veda", "Ensayos de operación", "Duraciones medidas" coherentes), `docs/web/contrato.md` (repaso)

**Interfaces:**
- Produces: "Estado al cierre de la fase 6 (fecha)": commits, recuento de tests, decisiones E1-E9, resoluciones
  de la ejecución (del libro de registro), menores aplazados, "Verificado en la sesión" (ensayo de veda local,
  `check-api.sh`, `nginx-check.sh`, cifras de carga y RSS, Docker si hubo), "No verificado" (ECS real, ALB,
  rol de tarea, carga contra ECS, `pg_restore`), "Pendiente de Luis" (crear la infraestructura de ECS y
  desplegar con las plantillas; `run-task` de `publish backtest` como prueba; medir `publish forecast all`
  en la tarea de jobs; carga contra el ALB y decidir `to_thread` (E3); `pg_dump` semanal; ensayar la veda la
  semana del 17; el 23: runbook; la noche del 29: descongelar; después: `publish event`), y la lista de lo
  que queda para el chat de diseño.

- [ ] **Step 1: Documentación** — [ ] **Step 2: Verificación final** (`python -m pytest -m "not integration"
  -q`; `bash deploy/check-api.sh http://127.0.0.1:8001` con `S3_BUCKET= DB_HOST=`; `bash deploy/nginx-check.sh`)
  — [ ] **Step 3: Commit** (`git add docs deploy/README.md && git commit -m "Document phase 6: veda, ECS, rehearsals and the closing state"`)
