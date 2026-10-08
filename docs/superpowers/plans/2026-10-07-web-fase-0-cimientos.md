# Web de resultados, fase 0 (cimientos y seguridad): plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar el núcleo HTTP de mtpy y la imagen Docker listos para la web: respuestas JSON correctas y
errores controlados en `mtpy/core/api.py`, rutas inyectables en `mtpy.api()`, imagen sin credenciales y
una comprobación temprana de `app.fs` sobre S3.

**Architecture:** cambios pequeños y compatibles en `mtpy/core/api.py` y `mtpy/mtpy.py`, cubiertos por un
arnés ASGI que no arranca servidor ni base de datos; `Dockerfile` con `COPY` explícito y `.dockerignore`
con semántica de Docker; un job `check_s3` del framework que ejercita `app.fs` para destapar el riesgo de
`s3fs==0.4.2`. Ninguna tarea toca `mtpy/lib`.

**Tech Stack:** Python 3.11, mtpy (ASGI propio en `mtpy/core/api.py`), pytest, Docker (`python:3.11-slim`,
nginx, supervisord), gunicorn 23.0.0 + uvicorn 0.18.3, s3fs 0.4.2.

**Spec:** `docs/superpowers/specs/2026-10-07-web-publicacion-design.md` (secciones "Cambios en el núcleo",
"Infraestructura", "Seguridad", "Pruebas" y fase 0 de "Fases").

## Global Constraints

- PEP8 y docstring en toda función, clase y método nuevos (AGENTS.md). Idioma: docstrings y comentarios
  en inglés dentro de `mtpy/`, estilo numpydoc como `mtpy/core/io.py` (`Parameters` / `Returns`); en
  español en `tests/`, `deploy/README.md` y `deploy/sql/*.sql`; comentarios del `Dockerfile` y de
  `.dockerignore` en inglés, como hoy.
- Tests con pytest; `pytest.ini` convierte `FutureWarning` en error. Los tests unitarios no arrancan
  `mtpy.run()` ni tocan la base; el arnés ASGI crea un `App` nuevo y restaura el anterior (fixture
  `fresh_app`).
- Antes de empezar, `git status --short` debe estar limpio (M13 ya está commiteado: `5ec4dbb`). **Cada
  commit añade solo los ficheros de su tarea** con `git add <ruta>` explícito; nunca `git add -A`,
  `git add .` ni `git commit -a`. Rama `dev`.
- Nada de credenciales en ficheros versionados ni en la imagen. Los `deploy/*.env` reales siguen fuera
  de git (`.gitignore` ya tiene `*.env`); solo se versionan `*.env.example` con valores vacíos.
- Compatibilidad: `Index.index_action` (`/` → `{"status": "ok", "message": "API Home"}`) sigue
  funcionando; `init.sh` y `supervisord` siguen arrancando nginx + gunicorn con `APP_API=api`.
- Versiones: base `python:3.11-slim`; `gunicorn==23.0.0`; `uvicorn==0.18.3` (conserva
  `uvicorn.workers.UvicornWorker`). `s3fs==0.4.2` se mantiene en esta fase: el job `check_s3` (Tarea 7)
  solo diagnostica; si Luis lo ejecuta y falla, la fase 1 aplica la regla de decisión de `deploy/README.md`.
- La imagen se construye para `linux/amd64` (el servidor); la máquina de Luis es arm64, así que
  `docker build` lleva `--platform linux/amd64` y corre bajo emulación (más lento, no distinto).
- Mensajes de commit cortos en inglés, como el historial del repositorio, terminados con
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. Query string con claves repetidas o vacías (`?a=1&a=3&c=`): `Request.query` devuelve el primer valor
   y conserva las vacías, nunca lanza (tests en Tarea 1).
2. Valores numpy anidados y `pd.NaT` en una respuesta: `json_default` convierte arrays en listas con
   `null` por NaN y `NaT` en `null`; un NaN en un `float` nativo (o `np.float64`, que es subclase de
   `float`) produce un 500 JSON legible, no un cuerpo que `JSON.parse` rechace (tests en Tarea 2).
3. Una acción que devuelve un entero fuera de `status_codes` (418) o un `bool`, o que lanza `HttpError`
   sin mensaje: respuesta JSON con ese código y un mensaje por defecto, nunca `KeyError` (tests en
   Tareas 2 y 3).
4. Arranque con `lifespan=auto` de uvicorn: el scope `lifespan` no trae `path`; el ciclo
   `startup`/`shutdown` recibe `*.complete` y termina; un scope `websocket` se ignora sin excepción
   (tests en Tarea 4, con scopes desnudos como los de uvicorn).
5. La imagen sigue arrancando con la semántica de los env actuales (`APP_API=api`, `S3_BUCKET` vacío) y
   responde en `/`; ningún `*.env`, `.git` ni notebook dentro (verificación en Tarea 6).

---

### Task 1: Arnés ASGI de tests y `Request.query`

**Files:**
- Modify: `tests/conftest.py` (añadir la fixture `fresh_app`)
- Create: `tests/test_api_asgi.py`
- Modify: `mtpy/core/api.py:25-43` (`Request`)

**Interfaces:**
- Produces: fixture `fresh_app` (en `tests/conftest.py`): `App` recién creado, restaurado al terminar.
- Produces: en `tests/test_api_asgi.py`, `make_api(app, routes=ROUTES, not_found=()) -> Api` y
  `call(api, path, query='', method='GET', scope_type='http', incoming=None)`, que devuelve
  `(status, headers: dict, body: bytes)` para scopes `http` y la lista de mensajes enviados para otros.
  Los scopes distintos de `http` solo llevan `type` y `headers`, como los de uvicorn.
- Produces: `Request.query -> dict[str, str]` (primer valor de cada clave; las claves vacías se conservan).

- [ ] **Step 1: Añadir la fixture `fresh_app` a `tests/conftest.py`**

```python
@pytest.fixture
def fresh_app():
    """`App` nuevo para el test; al terminar se restaura el anterior (si `mtpy.run()` ya había arrancado)."""
    from mtpy.core.app import App

    saved = App._app
    App._app = None
    app = App.get_()
    yield app
    App._app = saved
```

- [ ] **Step 2: Crear `tests/test_api_asgi.py` con el arnés, el controlador de pruebas y los dos primeros tests**

```python
"""Tests del núcleo HTTP de mtpy (`mtpy/core/api.py`) con un arnés ASGI sin servidor ni base de datos."""
import asyncio
import json
import types

import numpy as np
import pandas as pd

from mtpy.core.api import Api, Controller, Router


class Echo(Controller):
    """Controlador de pruebas: una acción por comportamiento verificado."""

    async def list_action(self):
        return [1, 2, 3]

    async def types_action(self):
        return {
            'n': np.int64(3), 'f': np.float64(1.5), 'b': np.bool_(True), 'd': pd.Timestamp('2026-10-07'),
            'nat': pd.NaT, 'arr': np.array([1.0, np.nan]), 'none': None,
        }

    async def nan_action(self):
        return {'x': float('nan')}

    async def text_action(self):
        return 'hola'

    async def csv_action(self):
        self.response.set_content_type('text/csv')
        return 'a,b\n1,2\n'

    async def query_action(self):
        return self.request.query

    async def item_action(self, id):
        return {'id': id}

    async def teapot_int_action(self):
        return 418

    async def bool_action(self):
        return True

    async def boom_action(self):
        raise RuntimeError('kaboom')

    async def cached_action(self):
        self.response.set_cache(60)
        return {}

    async def immutable_action(self):
        self.response.set_cache(31536000, immutable=True).set_etag('abc')
        return {}


ROUTES = [
    ('/list', 'echo', 'list'), ('/types', 'echo', 'types'), ('/nan', 'echo', 'nan'), ('/text', 'echo', 'text'),
    ('/csv', 'echo', 'csv'), ('/query', 'echo', 'query'), ('/item/{id}', 'echo', 'item'),
    ('/teapot-int', 'echo', 'teapot_int'), ('/bool', 'echo', 'bool'), ('/boom', 'echo', 'boom'),
    ('/cached', 'echo', 'cached'), ('/immutable', 'echo', 'immutable'),
]


def make_api(app, routes=ROUTES, not_found=()):
    """Router con los controladores de este módulo registrado en `app`; devuelve el callable ASGI."""
    router = Router().set_namespace(types.SimpleNamespace(Echo=Echo))
    for pattern, controller, action in routes:
        router.add_route(pattern, controller, action, methods=['GET'])
    for pattern, controller in not_found:
        router.add_not_found(pattern, controller)
    app.set('router', router)
    return Api()


def call(api, path, query='', method='GET', scope_type='http', incoming=None):
    """Una petición ASGI; devuelve (status, cabeceras, cuerpo) en `http` y los mensajes enviados en otros scopes.

    Los scopes distintos de `http` solo llevan `type` y `headers`, como los que envía uvicorn.
    """
    scope = {'type': scope_type, 'headers': []}
    if scope_type == 'http':
        scope.update({'path': path, 'method': method, 'query_string': query.encode('latin-1')})
    pending = list(incoming) if incoming is not None else [{'type': 'http.request', 'body': b'', 'more_body': False}]
    sent = []

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(api(scope, receive, send))
    if scope_type != 'http':
        return sent
    start = [m for m in sent if m['type'] == 'http.response.start'][0]
    body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
    headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in start['headers']}
    return start['status'], headers, body


def test_query_returns_first_value_of_each_key_and_keeps_blank_values(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/query', query='a=1&b=2&a=3&c=')
    assert status == 200
    assert json.loads(body) == {'a': '1', 'b': '2', 'c': ''}


def test_query_is_empty_without_query_string(fresh_app):
    assert json.loads(call(make_api(fresh_app), '/query')[2]) == {}
```

Las acciones `types`, `nan`, `csv`, `teapot_int`, `bool`, `boom`, `cached` e `immutable` se usan en las
Tareas 2-4; se definen ya para que el fichero no cambie de forma. `Echo` no redefine `not_found_action`:
la hereda de `Controller` (`mtpy/core/api.py:321-322`).

- [ ] **Step 3: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: FAIL en los dos tests con `AttributeError: 'Request' object has no attribute 'query'`.

- [ ] **Step 4: Implementar `Request.query` en `mtpy/core/api.py`**

En `Request.get_request` (`:37-43`) usar `parse_qsl(qs, keep_blank_values=True)`. Añadir la propiedad
`query(self) -> dict` que recorre `self.request` (lista de tuplas) y rellena un dict con `setdefault`, de
modo que gane el primer valor de cada clave. Docstring en ambos.

- [ ] **Step 5: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Comprobar que el resto de tests unitarios sigue en verde**

Run: `python -m pytest -m "not integration" -q`
Expected: todos PASS (hoy 158 tests más los nuevos), ningún error de recogida.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_api_asgi.py mtpy/core/api.py
git commit -m "$(printf 'Add ASGI test harness and Request.query\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 2: `Response.set_content` (listas, numpy, NaN, enteros y content-type explícito) y cabeceras de caché

**Files:**
- Modify: `mtpy/core/api.py:71-151` (`Response`) y cabecera del módulo (imports y `json_default`)
- Test: `tests/test_api_asgi.py`

**Interfaces:**
- Produces: `json_default(obj) -> Any`, función de módulo en `mtpy/core/api.py`, con este orden de ramas
  (importa: `pd.NaT` es subclase de `datetime`, así que su comprobación va antes que la de fechas):
  1. `obj is pd.NaT or obj is pd.NA` → `None` (identidad: `pd.NaT == pd.NaT` es `False`).
  2. `isinstance(obj, np.ndarray)` → `obj.tolist()` limpiado recursivamente (todo `float` NaN → `None`).
  3. `isinstance(obj, np.generic)` → `obj.item()`, y `None` si el resultado es un `float` NaN.
  4. `isinstance(obj, datetime.datetime)` (incluye `pd.Timestamp`) → `'%Y-%m-%d'` si `obj.time() ==
     datetime.time()` y `obj.tzinfo is None`, si no `obj.isoformat()`.
  5. `isinstance(obj, datetime.date)` → `obj.isoformat()`.
  6. `isinstance(obj, set)` → `list(obj)`.
  7. En otro caso, `TypeError`.
  (`np.float64` es subclase de `float`: `json.dumps` no llama a `default` y, con `allow_nan=False`, un
  NaN acaba en el 500 de abajo; `tuple` se serializa de forma nativa y no pasa por `default`.)
- Produces: `Response.set_cache(seconds: int, immutable: bool = False) -> Response` (cabecera
  `cache-control: public, max-age=<seconds>[, immutable]`) y `Response.set_etag(value: str) -> Response`
  (cabecera `etag: "<value>"`); `Response.status_codes` incluye `422: 'Unprocessable Entity'` y
  `503: 'Service Unavailable'`; `set_header` y `get_header` normalizan la clave con `key.lower()`.
- Produces: `set_content` trata `bool` antes que `int` (un `bool` se serializa como JSON `true`/`false`);
  la rama `int` usa `Response.status_codes.get(content, 'Error')`; `dict`, `list` y `tuple` van como JSON
  con `ensure_ascii=False`, `allow_nan=False` y `default=json_default`; un `TypeError`/`ValueError` al
  serializar produce estado 500 y cuerpo `{"status": "error", "message": "Invalid content"}`; `str` y
  `bytes` respetan un `content-type` ya fijado por la acción y, si no lo hay, siguen siendo `text/plain`.

- [ ] **Step 1: Añadir los tests a `tests/test_api_asgi.py`**

```python
def test_list_is_serialised_as_json(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/list')
    assert status == 200
    assert headers['content-type'] == 'application/json; charset=utf-8'
    assert json.loads(body) == [1, 2, 3]


def test_numpy_and_pandas_values_are_converted(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/types')
    assert status == 200
    assert json.loads(body) == {
        'n': 3, 'f': 1.5, 'b': True, 'd': '2026-10-07', 'nat': None, 'arr': [1.0, None], 'none': None,
    }


def test_nan_in_native_float_gives_a_json_500_not_an_invalid_body(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/nan')
    assert status == 500
    assert headers['content-type'] == 'application/json; charset=utf-8'
    assert json.loads(body) == {'status': 'error', 'message': 'Invalid content'}


def test_unknown_int_status_and_bool_do_not_raise(fresh_app):
    api = make_api(fresh_app)
    status, headers, body = call(api, '/teapot-int')
    assert status == 418
    assert json.loads(body) == {'status': 'error', 'message': '418 Error'}
    status, headers, body = call(api, '/bool')
    assert status == 200
    assert json.loads(body) is True


def test_str_without_explicit_type_is_text_plain(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/text')
    assert headers['content-type'] == 'text/plain; charset=utf-8'
    assert body == b'hola'


def test_explicit_content_type_is_kept_for_str(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/csv')
    assert status == 200
    assert headers['content-type'] == 'text/csv; charset=utf-8'
    assert body == b'a,b\n1,2\n'


def test_header_keys_are_case_insensitive(fresh_app):
    from mtpy.core.api import Response

    response = Response(None).set_header('Content-Type', 'text/csv')
    assert response.headers == {'content-type': 'text/csv'}
    assert response.get_header('CONTENT-TYPE') == 'text/csv'


def test_cache_headers(fresh_app):
    api = make_api(fresh_app)
    assert call(api, '/cached')[1]['cache-control'] == 'public, max-age=60'
    headers = call(api, '/immutable')[1]
    assert headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert headers['etag'] == '"abc"'


def test_status_codes_include_422_and_503():
    from mtpy.core.api import Response

    assert Response.status_codes[422] == 'Unprocessable Entity'
    assert Response.status_codes[503] == 'Service Unavailable'
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: fallan los 9 nuevos: `/list` devuelve 500 `Invalid content-type` (assert); `/types` FAILED por
`TypeError: Object of type int64 is not JSON serializable` (la excepción escapa de `asyncio.run`);
`/nan` devuelve 200 con cuerpo `{"x": NaN}` (assert 500); `/teapot-int` FAILED por `KeyError: 418`;
`/csv` llega como `text/plain; charset=utf-8` (assert); el test de cabeceras falla porque la clave queda
como `Content-Type`; `/cached` FAILED por `AttributeError: 'Response' object has no attribute
'set_cache'`; 422/503 FAILED por `KeyError: 422`. Los dos de la Tarea 1 siguen en verde.

- [ ] **Step 3: Implementar en `mtpy/core/api.py`**

`import datetime`, `import math`, `import numpy as np`, `import pandas as pd` en la cabecera.
`json_default(obj)` como función de módulo con el orden de ramas de Interfaces. En `Response`: añadir
422 y 503 a `status_codes`; `set_header`/`get_header` con `key = key.lower()`; `set_cache` y `set_etag`
con `set_header`; en `set_content`, primero `isinstance(content, bool)` (va a la rama JSON), después la
rama `int` con `Response.status_codes.get(content, 'Error')`, después `isinstance(content, (dict, list,
tuple))` con `json.dumps(content, ensure_ascii=False, allow_nan=False, default=json_default)` dentro de
`try/except (TypeError, ValueError)` que fija 500 y el cuerpo `{"status": "error", "message": "Invalid
content"}`; en las ramas `str` y `bytes`, `if 'content-type' not in self.headers:
self.set_content_type('text/plain')`.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_asgi.py mtpy/core/api.py
git commit -m "$(printf 'JSON lists, numpy values and cache headers in core API responses\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 3: `HttpError` y errores controlados en `Controller.dispatch`

**Files:**
- Modify: `mtpy/core/api.py` (nueva clase `HttpError` tras `Response`; `Controller.dispatch`, `:300-310`)
- Test: `tests/test_api_asgi.py`

**Interfaces:**
- Produces: `class HttpError(Exception)` con `__init__(self, status: int, message: Optional[str] = None)`,
  atributos `status` y `message`; sin mensaje, `message` = `'<status> <texto de status_codes o "Error">'`.
- Produces: `Controller.dispatch` captura `HttpError` (estado = `error.status`, cuerpo
  `{"status": "error", "message": error.message}`) y cualquier otra excepción (registro con
  `self.app.logger.error(...)` si `self.app.logger` no es `None`, y respuesta `500` por el mecanismo `int`
  de `set_content`). `before_dispatch` y `after_dispatch` quedan dentro del `try`.

- [ ] **Step 1: Añadir a `tests/test_api_asgi.py` el import, dos acciones, un controlador y cinco tests**

En el import: `from mtpy.core.api import Api, Controller, HttpError, Router`. En `Echo`:

```python
    async def http_error_action(self):
        raise HttpError(404, 'no such thing')

    async def teapot_action(self):
        raise HttpError(418)
```

Tras `Echo`, un segundo controlador, y en `make_api` el namespace pasa a
`types.SimpleNamespace(Echo=Echo, Guarded=Guarded)`:

```python
class Guarded(Echo):
    """Controlador cuyo `before_dispatch` rechaza toda petición."""

    def before_dispatch(self):
        raise HttpError(401, 'login required')
```

En `ROUTES`: `('/http-error', 'echo', 'http_error'), ('/teapot', 'echo', 'teapot'), ('/guarded', 'guarded', 'list')`.
Tests:

```python
def test_http_error_becomes_a_json_error_with_its_status(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/http-error')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': 'no such thing'}


def test_http_error_with_unknown_status_and_no_message(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/teapot')
    assert status == 418
    assert json.loads(body) == {'status': 'error', 'message': '418 Error'}


def test_http_error_in_before_dispatch_is_handled(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/guarded')
    assert status == 401
    assert json.loads(body) == {'status': 'error', 'message': 'login required'}


def test_unhandled_exception_is_logged_and_returns_500(fresh_app):
    errors = []
    fresh_app.set('logger', types.SimpleNamespace(error=errors.append))
    status, headers, body = call(make_api(fresh_app), '/boom')
    assert status == 500
    assert json.loads(body) == {'status': 'error', 'message': '500 Internal Server Error'}
    assert len(errors) == 1 and 'kaboom' in errors[0]


def test_unhandled_exception_without_logger_still_returns_500(fresh_app):
    assert call(make_api(fresh_app), '/boom')[0] == 500
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: `ImportError: cannot import name 'HttpError'` (todo el módulo falla en la recogida).

- [ ] **Step 3: Implementar `HttpError` y el `try/except` de `Controller.dispatch` en `mtpy/core/api.py`**

`HttpError` tras la clase `Response` (usa `Response.status_codes.get(status, 'Error')` para el mensaje
por defecto). En `dispatch`: `try` alrededor de `before_dispatch`, la acción y `after_dispatch`;
`except HttpError as error` → `self.response.set_status_code(error.status)` y `self.result = {'status':
'error', 'message': error.message}`; `except Exception` → si `self.app.logger` no es `None`,
`self.app.logger.error('Unhandled error in {}.{}: {}'.format(type(self).__name__, action,
traceback.format_exc()))`, y `self.result = 500`. `await self.send()` fuera del `try`. `import traceback`.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: PASS (16 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_asgi.py mtpy/core/api.py
git commit -m "$(printf 'Add HttpError and controlled error responses to the core API\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 4: `Router.handle` sin parámetros residuales y scopes ASGI no HTTP

**Files:**
- Modify: `mtpy/core/api.py:10-22` (`Api.__call__`) y `:251-277` (`Router.handle`)
- Test: `tests/test_api_asgi.py`

**Interfaces:**
- Produces: `Api.__call__` responde el ciclo `lifespan` (`lifespan.startup` → `lifespan.startup.complete`;
  `lifespan.shutdown` → `lifespan.shutdown.complete` y fin) y vuelve sin hacer nada en cualquier scope
  distinto de `http` y `lifespan`, antes de leer `scope['path']`.
- Produces: `Router.handle` deja `self.params = {}` también en la rama `not_found` registrada con
  `add_not_found` (`:267-273`).

- [ ] **Step 1: Añadir los tests a `tests/test_api_asgi.py`**

```python
def test_prefix_not_found_does_not_reuse_previous_params(fresh_app):
    api = make_api(fresh_app, not_found=[('/x', 'echo')])
    assert json.loads(call(api, '/item/42')[2]) == {'id': '42'}
    status, headers, body = call(api, '/x/anything')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_unknown_route_is_a_json_404(fresh_app):
    status, headers, body = call(make_api(fresh_app), '/nope')
    assert status == 404
    assert json.loads(body) == {'status': 'error', 'message': '404 Not Found'}


def test_lifespan_scope_is_acknowledged(fresh_app):
    sent = call(make_api(fresh_app), '', scope_type='lifespan',
                incoming=[{'type': 'lifespan.startup'}, {'type': 'lifespan.shutdown'}])
    assert sent == [{'type': 'lifespan.startup.complete'}, {'type': 'lifespan.shutdown.complete'}]


def test_other_scopes_are_ignored(fresh_app):
    assert call(make_api(fresh_app), '', scope_type='websocket', incoming=[]) == []
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: `test_prefix_not_found_does_not_reuse_previous_params` falla con estado 500 (los parámetros
`{'id': '42'}` de la petición anterior llegan a `not_found_action`, provocan `TypeError` y el `except`
de la Tarea 3 lo convierte en 500); `test_lifespan_scope_is_acknowledged` y `test_other_scopes_are_ignored`
fallan con `KeyError: 'path'` (el scope desnudo llega a `Request.__init__`, `:33`);
`test_unknown_route_is_a_json_404` pasa ya (rama final de `handle`).

- [ ] **Step 3: Implementar en `mtpy/core/api.py`**

En `Api`: método `async def lifespan(self, receive, send)` con el bucle descrito en Interfaces; en
`__call__`, `if scope['type'] == 'lifespan': await self.lifespan(receive, send); return` y
`if scope['type'] != 'http': return` antes de crear `Request`. En `Router.handle`, añadir
`self.params = {}` en la rama de `self.not_found` (`:267-273`).

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: PASS (20 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_asgi.py mtpy/core/api.py
git commit -m "$(printf 'Handle ASGI lifespan and reset stale route params in the core API\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 5: `mtpy.api(routes=None)`

**Files:**
- Modify: `mtpy/mtpy.py:100-111` (`api`)
- Test: `tests/test_api_asgi.py`

**Interfaces:**
- Produces: `mtpy.api(routes: Optional[list[tuple]] = None) -> Optional[Api]`: registra `/` → `index`
  como hoy y después cada tupla de `routes` con `router.add_route(*route)`; las tuplas son
  `(pattern, controller, action)` o `(pattern, controller, action, methods)`. Sin `App` arrancado sigue
  devolviendo `None`. Es el punto donde la fase 2 inyectará `webapi.ROUTES` desde `api.py`.

- [ ] **Step 1: Añadir los tests a `tests/test_api_asgi.py`**

```python
def test_mtpy_api_registers_extra_routes(fresh_app):
    from mtpy import mtpy

    api = mtpy.api(routes=[('/api/v1/ping', 'index', 'index', ['GET'])])
    assert isinstance(api, Api)
    assert [route['pattern'] for route in fresh_app.router.routes] == ['/', '/api/v1/ping']
    assert fresh_app.router.routes[1]['methods'] == {'GET'}


def test_mtpy_api_without_app_returns_none():
    from mtpy import mtpy
    from mtpy.core.app import App

    saved = App._app
    App._app = None
    try:
        assert mtpy.api() is None
    finally:
        App._app = saved
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_api_asgi.py -v -k mtpy_api`
Expected: `test_mtpy_api_registers_extra_routes` falla con `TypeError: api() got an unexpected keyword
argument 'routes'`; el otro pasa ya.

- [ ] **Step 3: Implementar `api(routes=None)` en `mtpy/mtpy.py`**

Firma `def api(routes=None):` con docstring; tras `router.add_route('/', 'index')`, `for route in routes
or []: router.add_route(*route)`.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_api_asgi.py -v`
Expected: PASS (22 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_asgi.py mtpy/mtpy.py
git commit -m "$(printf 'Accept extra routes in mtpy.api()\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 6: Imagen Docker sin credenciales ni cron, gunicorn 23

Precondición: daemon de Docker en marcha (`docker info` responde).

**Files:**
- Modify: `.dockerignore`, `Dockerfile`, `deploy/docker/supervisord.conf:14-20`, `deploy/docker/init.sh:8-11`,
  `requirements.txt` (línea `gunicorn==20.1.0`)
- Delete: `deploy/docker/crontab`

**Interfaces:**
- Produces: imagen `elections-web:<tag>` para `linux/amd64` que contiene solo `api.py job.py worker.py
  pipeline.py config.json mtpy/ data/` más `files/` y `log/` vacíos; `ENV GIT_COMMIT` desde
  `--build-arg`; arranca nginx + gunicorn con `APP_API=api`; `HEALTHCHECK` sobre
  `http://127.0.0.1:8042/` (la fase 2 lo cambiará a `/api/v1/health`).

- [ ] **Step 1: Reescribir `.dockerignore`**

```
# Docker anchors patterns at the context root: use **/ for nested paths.
.git
.gitignore
.gitattributes
.pytest_cache/
.superpowers/
backtest/
docs/
files/
load/
log/
notebooks/
tests/
deploy/*.env
**/*.env
**/env.*
**/__pycache__/
**/.ipynb_checkpoints/
**/.DS_Store
*.code-workspace
AGENTS.md
venv/
.idea/
.vscode/
```

- [ ] **Step 2: Reescribir `Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PATH=/app

# Runtime only: nginx + supervisord serve the API, curl runs the healthcheck (every pin ships wheels;
# psycopg2-binary bundles its own libpq)
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl nginx supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /bin/bash /bin/sh

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY deploy/docker/nginx.conf /etc/nginx/nginx.conf
COPY deploy/docker/supervisord.conf /etc/supervisord.conf
COPY deploy/docker/init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh

RUN addgroup --gid 10001 docker \
    && adduser --disabled-password --uid 10001 --ingroup docker docker \
    && mkdir -p /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chown -R docker:docker /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor \
    && chmod -R g=u /run /var/log /var/cache /var/lib /etc/nginx /etc/supervisor

WORKDIR ${APP_PATH}

# Explicit list: credentials, git history, notebooks and runtime files never enter the image
COPY --chown=docker:docker api.py job.py worker.py pipeline.py config.json ./
COPY --chown=docker:docker mtpy/ mtpy/
COPY --chown=docker:docker data/ data/
RUN mkdir -p files log && chown -R docker:docker files log

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

USER docker
EXPOSE 8042
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8042/ || exit 1

CMD ["/bin/bash", "-c", "init.sh"]
```

- [ ] **Step 3: Quitar el cron muerto y subir gunicorn**

En `deploy/docker/supervisord.conf` borrar el bloque `[program:crontab]` (`:14-20`). En
`deploy/docker/init.sh` borrar el bloque `APP_CRONTAB` (`:8-11`, desde `APP_CRONTAB=$(echo ...)` hasta
su `fi`). `git rm deploy/docker/crontab` (deja la eliminación preparada en el índice). En
`requirements.txt`, `gunicorn==20.1.0` → `gunicorn==23.0.0`.

- [ ] **Step 4: Construir la imagen**

Run: `docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t elections-web:fase0 .`
Expected: termina con `naming to docker.io/library/elections-web:fase0` (bajo emulación tarda varios
minutos). `husl==4.0.3` se instala desde sdist (`Building wheel for husl`): es Python puro y no necesita
compilador; no es un fallo. Solo si `pip install` falla compilando otro paquete, añadir `gcc libpq-dev` a
la línea de `apt-get install`, reconstruir y anotarlo en el commit.

- [ ] **Step 5: Verificar el contenido de la imagen**

Run:
```bash
docker run --rm elections-web:fase0 bash -c 'find /app \( -name "*.env" -o -name ".git" -o -name "*.ipynb" \) | wc -l; ls -1 /app | sort; echo "commit=$GIT_COMMIT"; gunicorn --version'
```
Expected: `0`; nueve líneas exactas: `api.py`, `config.json`, `data`, `files`, `job.py`, `log`, `mtpy`,
`pipeline.py`, `worker.py`; `commit=<sha>`; `gunicorn (version 23.0.0)`.

- [ ] **Step 6: Verificar que la API arranca**

Run:
```bash
docker run -d --rm --name ew -p 8042:8042 -e APP_API=api -e DB_ADAPTER=PostgreSQL -e APP_VERBOSE=1 elections-web:fase0
for i in $(seq 1 45); do curl -fsS http://localhost:8042/ && break; sleep 2; done; echo; docker stop ew
```
Expected: `{"status": "ok", "message": "API Home"}` y el contenedor para.

- [ ] **Step 7: Commit**

```bash
git status --short   # debe mostrar "D  deploy/docker/crontab" entre los cambios
git add .dockerignore Dockerfile deploy/docker/supervisord.conf deploy/docker/init.sh requirements.txt
git commit -m "$(printf 'Docker image with explicit COPY, no credentials and no cron; gunicorn 23\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Task 7: Job `check_s3`, ficheros de entorno de ejemplo y rol de solo lectura

**Files:**
- Create: `mtpy/jobs/CheckS3.py`, `tests/test_jobs_check_s3.py`, `deploy/web.env.example`,
  `deploy/publish.env.example`, `deploy/sql/web_reader.sql`, `deploy/README.md`

**Interfaces:**
- Produces: `class CheckS3(Job)` con `run(self, prefix='site/v1/_smoke', fs=None, **kwargs) ->
  dict[str, bool]`; `fs` por defecto `self.app.fs`; pasos en orden `write` (`fs.write_bytes(payload,
  name)` con `name = prefix + '/check.json'`), `read` (`fs.read_bytes(name) == payload`), `exists`
  (`fs.exists(name)`), `listdir` (`'check.json' in fs.listdir(prefix)`), `remove` (`fs.remove(name)`),
  `gone` (`not fs.exists(name)`); cada paso imprime `<paso> OK` o `<paso> FAIL: <error>`; si alguno
  falla, al final `raise RuntimeError('check_s3 failed: <pasos fallidos>')`. Invocación:
  `python job.py check_s3` (el nombre `check_s3` resuelve a `mtpy/jobs/CheckS3.py` vía `to_camel`).

- [ ] **Step 1: Crear `tests/test_jobs_check_s3.py`**

```python
"""Tests del job `check_s3` (`mtpy/jobs/CheckS3.py`) sobre un sistema de ficheros local."""
import pytest

from mtpy.core.io import FileSystem


def test_check_s3_round_trip_on_local_fs(fresh_app, tmp_path, capsys):
    from mtpy.jobs.CheckS3 import CheckS3

    result = CheckS3().run(fs=FileSystem(str(tmp_path)))
    assert result == {'write': True, 'read': True, 'exists': True, 'listdir': True, 'remove': True, 'gone': True}
    assert 'write OK' in capsys.readouterr().out
    assert not (tmp_path / 'site' / 'v1' / '_smoke' / 'check.json').exists()


def test_check_s3_raises_when_a_step_fails(fresh_app, tmp_path):
    from mtpy.jobs.CheckS3 import CheckS3

    class Broken(FileSystem):
        def read_bytes(self, name):
            raise IOError('boom')

    with pytest.raises(RuntimeError, match='read'):
        CheckS3().run(fs=Broken(str(tmp_path)))
```

- [ ] **Step 2: Ejecutar los tests para verificar que fallan**

Run: `python -m pytest tests/test_jobs_check_s3.py -v`
Expected: `ModuleNotFoundError: No module named 'mtpy.jobs.CheckS3'`.

- [ ] **Step 3: Implementar `mtpy/jobs/CheckS3.py`**

`from ..core.worker import Job` (patrón de `mtpy/jobs/Test.py`). `payload` = JSON con `checked_at` en
UTC. Cada paso en su `try/except Exception` que convierte el fallo en `False` y guarda el mensaje;
`print('{} {}'.format(step, 'OK' if ok else 'FAIL: ' + error))`; `self.app.logger.info(...)` solo si
`self.app.logger` no es `None`. Al final, si hay fallos, `RuntimeError` con la lista de pasos fallidos.

- [ ] **Step 4: Ejecutar los tests para verificar que pasan**

Run: `python -m pytest tests/test_jobs_check_s3.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Crear los ficheros de despliegue**

`deploy/web.env.example` (las claves que `config.json` sustituye más las de `init.sh`, valores vacíos
salvo los no secretos; `GIT_COMMIT` no va aquí porque lo fija la imagen en el build y un `--env-file` lo
pisaría):

```
APP_ENV=pro
APP_VERBOSE=1
APP_API=api
APP_QUEUE=
LOC_TIMEZONE=Europe/Madrid
DB_ADAPTER=PostgreSQL
DB_HOST=
DB_PORT=5432
DB_USERNAME=web_reader
DB_PASSWORD=
DB_STAGE_DIR=
DB_DATABASE=manythings
AWS_KEY=
AWS_SECRET=
AWS_REGION=eu-west-1
S3_BUCKET=
CWL_GROUP=
PUSHOVER_USER=
PUSHOVER_TOKEN=
```

Añadir al final un comentario: `# Fase 2: WEB_PREFIX, WEB_CACHE_TTL, WEB_FREEZE y GUNICORN_WORKERS (sección web de config.json)`.
`deploy/publish.env.example`: igual pero `DB_USERNAME=` (usuario normal de la base), `DB_STAGE_DIR=stage`
y comentario de que las claves AWS son las del usuario IAM con escritura en `site/*`.

`deploy/sql/web_reader.sql`:

```sql
-- Rol de solo lectura para el contenedor web. Lo ejecuta Luis con el usuario maestro de la RDS.
-- La contraseña no puede contener comillas dobles ni barras invertidas (config.json se rellena por sustitución de texto).
CREATE ROLE web_reader LOGIN PASSWORD '<cambiar>';
GRANT CONNECT ON DATABASE manythings TO web_reader;
GRANT USAGE ON SCHEMA elections TO web_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA elections TO web_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA elections GRANT SELECT ON TABLES TO web_reader;
ALTER ROLE web_reader SET statement_timeout = '5s';
```

`deploy/README.md` (versión fase 0, en español): cómo construir la imagen (`docker build --platform
linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t elections-web:<tag> .`), cómo arrancarla
(`docker run -d --restart unless-stopped -p 127.0.0.1:8042:8042 --env-file
~/.config/elections-model/web.env elections-web:<tag>`), dónde viven los `.env` reales
(`~/.config/elections-model/`, modo 600, nunca en el repo; sin `GIT_COMMIT`, que lo fija la imagen), y
cómo ejecutar la comprobación de S3 desde el portátil (`python job.py check_s3`, con `.env` apuntando al
bucket) y desde la imagen (`docker run --rm --env-file ~/.config/elections-model/publish.env -e
RUN_JOB=check_s3 elections-web:<tag>`). Regla de decisión si `check_s3` falla: subir `s3fs` a una versión
de 2024 o posterior (con `aiobotocore` compatible) y repetir; si sigue fallando, la fase 1 reimplementa
`read_bytes`, `write_bytes`, `exists`, `listdir` y `remove` de `mtpy/core/services/s3.py` sobre `boto3`.

- [ ] **Step 6: Comprobar que los ejemplos no están ignorados por git y que los tests siguen en verde**

Run: `git check-ignore -v deploy/web.env.example deploy/publish.env.example; python -m pytest -m "not integration" -q`
Expected: `git check-ignore` no imprime nada (código de salida 1); todos los tests PASS.

- [ ] **Step 7: Commit**

```bash
git add mtpy/jobs/CheckS3.py tests/test_jobs_check_s3.py deploy/web.env.example deploy/publish.env.example deploy/sql/web_reader.sql deploy/README.md
git commit -m "$(printf 'Add check_s3 job, env examples and read-only DB role\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>')"
```

---

### Tareas de Luis (fuera del código; no bloquean las tareas 1-7)

- [ ] Ejecutar `deploy/sql/web_reader.sql` en la RDS con el usuario maestro (contraseña sin `"` ni `\`).
- [ ] Restringir el grupo de seguridad de la RDS al servidor web y al portátil.
- [ ] Crear los usuarios IAM `elections-web` (`s3:GetObject` y `s3:ListBucket` sobre `site/*`) y
  `elections-publish` (lo mismo más `s3:PutObject` y `s3:DeleteObject`); rotar el par de claves actual, compartido por
  `deploy/docker.env` y `deploy/elections.env`, y la contraseña de la RDS.
- [ ] Mover los `.env` reales a `~/.config/elections-model/` y ejecutar `python job.py check_s3` con el
  entorno de publicación; anotar el resultado (decide si la fase 1 debe cambiar `s3fs`).
- [ ] Confirmar que el esquema `elections` existe y está al día en la RDS; si no, `pg_dump
  --schema=elections` de la base local y `pg_restore` una sola vez.
- [ ] TLS hacia la RDS: el adaptador (`mtpy/core/data.py`) no expone `sslmode` y psycopg2 negocia con
  `sslmode=prefer`; activar `rds.force_ssl` solo tras comprobar que la conexión usa TLS (si no, la fase 2
  añade `sslmode` al adaptador).
