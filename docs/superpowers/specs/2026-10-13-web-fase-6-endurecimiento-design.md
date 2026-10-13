<!-- Especificación de diseño de la fase 6 de la web, acordada con Luis el 2026-10-13 (sesión de brainstorming con
     Claude Code). Enmienda la fila 6 de "Fases", "Infraestructura", "Seguridad" y "Verificación" de
     docs/superpowers/specs/2026-10-07-web-publicacion-design.md. -->

# Web, fase 6: veda, despliegue en AWS ECS y endurecimiento

## Contexto

Las fases 0 a 5 están publicadas y desplegadas (estado al cierre de la fase 5, 2026-10-11): la web sirve el
pronóstico, el promedio, los escaños, las casas, las elecciones pasadas, el modelo y el método desde el paquete
`site/v1` de S3 y la RDS. La elección general es el 2026-11-29 y la veda de sondeos (LOREG art. 69.7) va del 24 al
28 de noviembre. Hoy es 2026-10-13: quedan seis semanas. Luis va a mover la web de su servidor con Docker a AWS ECS
detrás de su gateway. La fase 6 deja la web lista para la campaña: qué pasa en la veda, cómo se despliega y opera
en ECS, los ensayos de operación que la spec general reservó para esta fase y la deuda técnica acumulada.

Los cambios de diseño o de contenido de las páginas (lo que Luis vio al recorrerlas) quedan fuera: van a un chat
aparte sobre diseño de la web, y entrarán como fase 7 o como tareas sueltas.

## Decisiones de Luis (2026-10-13)

1. **Veda: congelar y publicar el pronóstico definitivo.** El 23 de noviembre, con los sondeos del día cargados,
   se hace la última publicación de todos los ámbitos y se activa la congelación. La web muestra ese pronóstico,
   sin actualizarse, hasta después de la votación.
2. **Los ámbitos autonómicos se congelan igual** que `es`, sin reglas distintas por ámbito.
3. **`error_blocks` en el detalle de una elección**: las casas se ordenan y las barras se dibujan por el valor
   absoluto; el valor con signo va en una columna aparte, "Sesgo del margen" (el notebook PollstersEvent dibuja
   `.abs()`).
4. **Despliegue en AWS ECS detrás de un ALB** (respuesta del revisor-constructor, 2026-10-13); no hace falta nada
   más delante: el ALB termina TLS y reenvía `X-Forwarded-For` desde una IP privada de la VPC.
5. **Durante la veda sí se cargan sondeos** en la RDS; el cargador ya los marca con `ctype` `wban` (sondeos
   publicados durante la prohibición) o `exit` (a pie de urna) por el color de la fila de Wikipedia, y el modelo
   los excluye por defecto (`drop_ctypes=['wban', 'exit']` en `Computer` y `data.py`). La web no debe difundirlos.
6. **Se pueden tocar `supervisord.conf` e `init.sh`** para lo mínimo que ECS necesita (workers por variable,
   logs al proceso 1); la estructura de la imagen sigue siendo la de Luis.
7. **`publish` también se lanzará desde ECS** como tarea puntual, además de desde el portátil.

Supuestos míos que Luis no ha discutido (marcados como tales; cualquiera puede corregirse antes de planificar):

- Los sondeos publicados antes del 24 siguen visibles en `/promedio` y en el detalle de las elecciones durante la
  veda: la ley prohíbe publicar y difundir sondeos nuevos, no mantener los ya publicados, y es la práctica de los
  medios. La web no oculta vistas; solo deja de actualizarse y lo avisa.
- Con `force`, un `analysis` publicado dentro de la veda no deja marca en su `meta` (el esquema `analysis-meta`
  no tiene `freeze` y no se amplía): el aviso público es el del manifest.
- Durante la veda el job rechaza publicar `forecast`, `analysis` y rellenos retroactivos de cualquier ámbito
  (todos derivan de los sondeos de la elección en curso); `event` y `backtest` (elecciones pasadas) y las acciones
  operativas (`manifest`, `point`, `unpublish`) siguen permitidos.
- El ensayo de la congelación se hace sobre el paquete local, no sobre el público: el interruptor real se toca una
  vez, el 23 de noviembre, y otra para descongelar.

## Objetivos, no objetivos y anti-objetivos

Objetivos: (1) una veda sin sorpresas: un runbook de tres órdenes, un aviso claro en la web, una guarda que impida
publicar por error; (2) un despliegue en ECS reproducible, con el límite de peticiones y los logs funcionando
detrás del gateway; (3) los ensayos de operación de la spec general (carga, `point`/`unpublish`, congelación,
restauración de una copia); (4) pagar la deuda del núcleo que afecta a la operación (`HEAD`, bloqueo de la base
dentro de acciones `async`, arranque de los workers) y la que afecta a la mantenibilidad (`publish.py`); (5) cerrar
los flecos de la fase 5 que no son de diseño.

No objetivos: cambios de diseño o de contenido de las páginas (chat aparte); nuevas vistas; automatizar la
publicación con cron (sigue siendo manual, decisión de 2026-10-07); separar la web de `mtpy` (más adelante); un
rol de solo lectura en la RDS (decisión de 2026-10-08).

Anti-objetivos (fallos aunque todo "funcione"): publicar un run de `es` o de un autonómico dentro de la veda sin
`force`; que el gateway haga que todos los visitantes compartan un mismo cubo de `limit_req` (429 en masa el día
de la elección); que un cambio del núcleo (`HEAD`, `to_thread`, imports perezosos) cambie una sola respuesta de la
API o una página (la suite de 532 tests y `check-api.sh` son el contrato); que la división de `publish.py` cambie
un solo byte del paquete publicado.

## La veda

### Comportamiento

- **Runbook del 23 de noviembre** (README, sección "Veda"): cargar los sondeos del día como siempre → `python
  job.py publish '{"what":["forecast"],"scopes":"all"}'` → comprobar el manifest → `python job.py publish
  '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'` con el texto que decida Luis (por ejemplo:
  "Pronóstico definitivo, publicado el 23 de noviembre. Durante la veda electoral no se publican sondeos nuevos
  ni se actualiza el pronóstico.") → `bash deploy/check-api.sh https://<host>` comprueba la cabecera `x-freeze`
  y el banner. **Noche del 29**: `freeze` a `false`. Después, con los resultados oficiales cargados, `publish event`
  de la elección.
- **Web congelada**: el banner actual (`base.html`, `manifest.freeze.message`) se mantiene como único aviso global
  y pasa a tener un enlace a `/metodo#limitaciones`, donde ya se explica la veda. En la cabecera de las páginas
  del pronóstico (`/`, `/promedio`, `/escanos`), cuando el manifest está congelado, el sufijo del título dice
  "Pronóstico definitivo" en vez de "Pronóstico para el 29 nov 2026" y el pie añade "congelado desde {fecha}"
  (fecha de `freeze.since`, ver abajo). Los dos modos ("Hoy" y "Elección") siguen disponibles: son dos lecturas
  del mismo run definitivo. La API no cambia salvo la cabecera `x-freeze`, que ya existe.
- **Manifest**: `freeze` gana `since` (ISO, la fecha en que se activó; `null` si no está activo). El esquema
  `manifest` del contrato pasa a `freeze: {active, message, since}`; `update_manifest` pone `since` al activar y
  lo borra al desactivar. Contrato 1 se mantiene (clave nueva, lector tolerante).
- **Guarda LOREG ampliada** (`publish.loreg_guard`): hoy rechaza `es` en los 5 días previos a su elección y el
  día de la elección. Pasa a rechazar cualquier ámbito dentro de la ventana de **su propia** elección o de la
  elección **general** (`get_next_event_date('es')`), para `forecast` (incluido `backfill`) y `analysis`; `force`
  sigue publicando y marcando `meta.freeze: true`. `event`, `backtest`, `manifest`, `point` y `unpublish` no pasan
  por la guarda. El motivo del rechazo nombra la elección que lo causa ("es 2026-11-29: inside the LOREG window
  (art. 69.7)…", como hoy).
- **Sondeos cargados durante la veda**: las rutas en vivo que devuelven sondeos no sirven nunca los de `ctype`
  `wban` ni `exit`: `webapi.queries.query_polls` llama a `get_poll_series(scope, date, drop_contexts=WEB_DROP_CTYPES)`
  con `WEB_DROP_CTYPES = ('wban', 'exit')` (hoy los devuelve), con test. `/promedio` y el detalle de las
  elecciones leen el paquete (el run congelado del 23 y `get_polls_metric`, que ya los excluye), y `/casas` lee
  ratings que no se recalculan en la veda. El runbook lo dice: cargar sondeos durante la veda es seguro, no hay
  que recalcular ratings ni publicar hasta el 29. El constructor comprueba que el `Forecaster` también excluye
  esos `ctype` al construir `polls.json` (en `data.py` el filtro por defecto está en los lectores del modelo).
- **Ensayo** (README "Veda", apartado "Ensayo"): en local (`S3_BUCKET=`), con `today` simulado, `publish
  forecast` de `es` y de un autonómico con `"today":"2026-11-25"` → `refused`; con `force` y `dry_run` → publicado
  en `site-dry` con `meta.freeze: true`; `publish manifest` con `freeze` activo sobre el paquete local, uvicorn +
  `check-api.sh` (cabecera y banner), y desactivar. Nada de esto toca S3.

### Mecánica

| Fichero | Cambio |
|---|---|
| `mtpy/lib/publish.py` (`loreg_guard`, `update_manifest`, `read_manifest`) | ventana por ámbito propio y por `es`; `freeze.since`; aplicar la guarda a `analysis` |
| `mtpy/jobs/Publish.py` | `_analysis_scope` y los rellenos pasan por la guarda; línea `refused` con el motivo |
| `mtpy/lib/webapi/queries.py` | `WEB_DROP_CTYPES`; `query_polls` no sirve `wban` ni `exit` |
| `mtpy/lib/bundle.py` | `SCHEMAS['manifest']['freeze']` documentado con `since` (sigue siendo `dict`) |
| `mtpy/lib/pages/common.py`, `web/templates/base.html`, `_footer.html` | sufijo "Pronóstico definitivo", pie "congelado desde", enlace del banner |
| `deploy/check-api.sh` | con el manifest congelado comprueba `x-freeze: active` en una ruta de la API y el banner en `/` |
| `deploy/README.md` | sección "Veda" (runbook del 23, descongelar, ensayo) |

## Despliegue en AWS ECS

Luis despliega la misma imagen (nginx + gunicorn dentro del contenedor, puerto 8042, `init.sh` con `APP_API`)
como servicio de ECS detrás de un ALB, en la raíz de su propio host (sin prefijo de ruta: `/dist/` y `/api/v1`
no cambian). Lo que cambia respecto al servidor propio:

- **Dirección real del cliente**: nginx solo confía en `127.0.0.1` y `172.16.0.0/12` para `real_ip`. Detrás del
  ALB la petición llega desde una IP privada de la VPC, que puede ser `10.0.0.0/8`. `nginx.conf` confía en
  los tres rangos privados (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) con `real_ip_recursive on`, de modo
  que `limit_req` y los logs usan la IP del visitante que trae `X-Forwarded-For`. Sin esto, el día de la elección
  todos los visitantes compartirían un cubo de 20 r/s (anti-objetivo).
- **Credenciales de S3**: con un rol de tarea de ECS no hacen falta claves. `mtpy/core/services/s3.py` pasa
  `key`/`secret` explícitos a `S3FileSystem`; con `AWS_KEY`/`AWS_SECRET` vacíos debe pasar `None` para que s3fs use
  la cadena de credenciales de botocore (rol de tarea). Se mantiene el modo con claves (portátil, `job.py`).
  El rol de la tarea web solo necesita Get/List sobre `site/*`; el de la tarea de jobs, Get/List/Put/Delete sobre
  `site/*` y `site-dry/*`. `job.py` desde el portátil sigue con las claves de `deploy/elections.env`.
- **Base de datos**: la RDS debe aceptar conexiones desde el grupo de seguridad del servicio; `DB_STATEMENT_TIMEOUT=10000`
  en la definición de tarea (pendiente desde la fase 4).
- **Comprobación de salud**: el target group del gateway apunta a `/api/v1/health` (atraviesa gunicorn; `/healthz`
  solo prueba nginx), intervalo 30 s, 2 fallos para marcar la tarea como no sana. La imagen no necesita
  `HEALTHCHECK` propio.
- **Workers y logs** (cambio mínimo en `deploy/docker/supervisord.conf` e `init.sh`, decisión 6): gunicorn
  arranca con `-w %(ENV_GUNICORN_WORKERS)s` (por defecto 4, como hoy; en ECS 2) y escribe sus logs en
  `/proc/1/fd/1` y `/proc/1/fd/2` como ya hace nginx (hoy apunta a `/dev/stdout` de un supervisord demonizado y
  se pierde); `init.sh` exporta `GUNICORN_WORKERS` con su valor por defecto. Los tracebacks de la app
  (`core/app.Log`) van a CloudWatch con `CWL_GROUP` en la definición de tarea, que ya existe en `mtpy`; sin
  cambios de código. El driver `awslogs` recoge la salida del proceso 1.
- **Tamaño**: `GUNICORN_WORKERS=2` → 1 vCPU y 2 GB de memoria por tarea; una tarea basta (la caché de nginx y la
  de la API son por proceso, así que más tareas no comparten caché; escalar solo si el p95 lo pide).
- **`publish` desde ECS** (decisión 7): una segunda definición de tarea, `elections-job`, con la misma imagen,
  2 vCPU y 4 GB (un `forecast` de todos los ámbitos con 1.000 simulaciones; el constructor mide el RSS real de
  `publish forecast es` y ajusta), sin `APP_API`, con `RUN_JOB` vacío por defecto y un rol de tarea con
  Get/List/Put/Delete sobre `site/*` y `site-dry/*`; se lanza con `aws ecs run-task` sobreescribiendo
  `RUN_JOB` (`publish {"what":["forecast"],"scopes":"all"}`, JSON sin espacios porque `init.sh` no entrecomilla),
  sale por `awslogs` y termina. El rol de la tarea web queda de solo lectura (`Get`/`List`). Plantillas de ambas
  definiciones y del `run-task` en `deploy/ecs/` y README "Publicar desde ECS".
- **Procedimiento** (README "Despliegue en ECS"): `docker build --platform linux/amd64 -t <ecr>/elections-web:<sha>
  .` → `docker push` → actualizar la definición de tarea (imagen y variables) → `aws ecs update-service
  --force-new-deployment` → `bash deploy/check-api.sh https://<host>` → `bash deploy/smoke.sh` ya no aplica al
  servidor (se mantiene para el Docker local). Plantilla de definición de tarea sin secretos en
  `deploy/ecs/task-web.example.json` (familia, puerto 8042, variables de entorno con los nombres de
  `.env.example` más `GUNICORN_WORKERS=2`, `DB_STATEMENT_TIMEOUT=10000` y `CWL_GROUP`, secretos de `DB_PASSWORD`
  desde Secrets Manager o SSM, `awslogs`, rol de tarea de lectura, 1 vCPU / 2 GB), `deploy/ecs/task-job.example.json`
  y `deploy/ecs/run-task.example.json`. El cluster, el repositorio ECR, el servicio, el target group, los roles y
  el grupo de logs los crea Luis en la consola (el README lista lo que debe existir; no hay infraestructura como
  código en esta fase).
- **`deploy/elections.env` deja de ser el entorno de la web** (sigue siendo el de `job.py` en el portátil): las
  mismas variables viven en la definición de tarea. El README lo dice en "Fichero de configuración".

## Ensayos de operación

- **Carga**: 200 peticiones concurrentes a `/api/v1/forecast/es/nowcast/summary` y a `/` contra el servicio en
  ECS con `hey` o `ab`; criterio: `X-Cache: HIT` en la mayoría, p95 < 200 ms en la API y < 500 ms en las
  páginas, sin 5xx ni 429 desde una sola IP de prueba (el límite es por visitante; la prueba desde una IP sí
  puede dar 429 a partir de 20 r/s: se documenta y se mide con `hey -q` por debajo del límite). Si el p95 de las
  páginas lo pide, `asyncio.to_thread` entra (ver "Deuda del núcleo"); si no, se deja documentado como no
  necesario.
- **`point` y `unpublish`**: ensayo en local sobre un paquete con dos runs (README ya lo documenta en
  producción); criterio: el manifest cambia, `history.json` se reconstruye, la web lo refleja en ≤ 2 min.
- **Congelación**: el ensayo de la sección "La veda".
- **Copia y restauración**: `pg_dump -Fc -n elections` de la RDS a un fichero local y a `s3://<bucket>/backups/`
  (comando documentado, lo ejecuta Luis semanalmente en campaña), y `pg_restore` en la base local; criterio: los
  tests de integración pasan contra la base restaurada. Sin job nuevo (YAGNI).

## Deuda del núcleo

Cambios en `mtpy/core/api.py` y `mtpy/core/utils`, cada uno con tests y sin cambiar ninguna respuesta:

1. **`HEAD`**: el router acepta `HEAD` en toda ruta `GET` y responde con las mismas cabeceras y sin cuerpo (hoy,
   404 en uvicorn sin nginx; nginx lo convertía). Útil para los monitores del gateway.
2. **Lecturas bloqueantes dentro de acciones `async`**: `webapi.queries._query` y las lecturas del paquete pasan
   por `asyncio.to_thread` solo si la prueba de carga lo pide; antes, `Router.handle` debe despachar sobre
   variables locales (hoy guarda `params`/`action` en el router: con `to_thread` dos peticiones concurrentes se
   pisarían). Si la carga no lo pide, se hace igualmente el cambio del router (bug latente, "Cambios en el
   núcleo" 4 de la spec general) y `to_thread` se deja documentado.
3. **Arranque de los workers**: `mtpy/core/utils/helpers.py` e `io.py` importan scipy y statsmodels; pasarlos a
   imports perezosos dentro de las funciones que los usan, medir el RSS de un worker y el tiempo de arranque
   antes y después (README "Duraciones medidas").
4. **`Page.dispatch` duplica `Controller.dispatch`**: `Controller.dispatch` gana un gancho `error_result(status,
   message)` que `Page` sobrescribe; una sola implementación.
5. **`worker.py`, enlace de módulo latente** (de la lista diferida del rediseño): el mismo patrón que se corrigió
   en `Router.controller_class`.
6. **El manifest se parsea ~4 veces por petición**: `Site.manifest_data` cachea el dict parseado junto a los
   bytes (misma clave y TTL).

## Mantenibilidad: `publish.py`

`mtpy/lib/publish.py` tiene ~1.900 líneas. Pasa al paquete `mtpy/lib/publish/` con el mismo patrón de `webapi` y
`pages` (fachada `__init__` que reexporta todo; ningún submódulo con el nombre de un símbolo exportado; los tests
y `jobs/Publish.py` siguen usando `publish.X`; los `monkeypatch` cambian al submódulo que define la función):
`common.py` (constantes de redondeo, `records`, `round_cols`, `provenance`, `_madrid_tz`, `_today`,
`loreg_guard`, `resolve_scopes`, `next_event_date`, `db_stats`, excepciones), `forecast.py` (exportadores del
run, `publish_mode`, `publish_forecast`, `backfill_*`, `latest_entry`), `analysis.py`, `events.py`,
`backtest.py`, `pointers.py` (`rebuild_history`, `read_manifest`, `update_manifest`, `point`, `unpublish`,
`ATTRIBUTION`). Criterio: el paquete publicado por `publish forecast`, `analysis`, `event` y `backtest` en
`site-dry` es byte a byte igual antes y después (test de integración marcado, o comparación de los fixtures).

## Flecos de la fase 5 que entran

De la lista "Menores aplazados" del estado de cierre de la fase 5, los que no son de diseño de las páginas:

- `error_blocks` por valor absoluto en el detalle de la elección, con la columna "Sesgo del margen" (decisión 3).
- `casas.html`: el texto del rating dice lo mismo que `/metodo` (sesgo relativo al consenso de las demás casas).
- Helper compartido `pages.common.bundle_part(loader, *args)` para `analysis_part`, `event_part`, `backtest_part`.
- Tests que faltan: columna `PROB_PREFIXES` entera que sigue siendo flotante; `event_date='all'` y lista con una
  fecha inválida a través del job; ruta de un `model-*` CSV de un evento sin backtest; TTL de `Site.events`;
  fila de casa sin `pollster_id`.
- Orden y anchura de los imports de `webapi/__init__.py`; `headers` sin usar en tres tests; el `patched`
  global de `provenance`; el comentario de `check_published` sobre `scope not published`.
- `initial.seats` de `/modelo`: se usa (una tercera gráfica pequeña de error de escaños por horizonte) o se quita
  del contexto; mi propuesta: quitarlo, porque la tabla "Escaños y mayorías" ya lo muestra.

Quedan fuera, para el chat de diseño: el histograma de errores del detalle, las nimiedades de redacción de
`/metodo` (1.000 simulaciones a mano, la frase de calibración), los matices de las barreras autonómicas y
cualquier cambio de maquetación.

## Verificación

- Fase 6 de la spec general: 200 peticiones concurrentes con `X-Cache: HIT`; `point` a un run anterior visible
  en ≤ 2 min; `freeze` activado y desactivado (en local); restauración de un `pg_dump` en la base local.
- `python -m pytest -m "not integration" -q` en verde (532 hoy) y los tests de integración marcados contra la RDS
  (solo lectura) en verde; `bash deploy/check-api.sh https://<host>` contra ECS sin FAIL, incluida la cabecera
  `x-freeze` cuando toque; `HEAD /api/v1/manifest` → 200 sin cuerpo; RSS de un worker medido antes y después.
- El paquete publicado en `site-dry` es idéntico antes y después de dividir `publish.py`.
- El runbook de la veda ensayado en local el mismo día que se escribe, y de nuevo la semana del 17 de noviembre.

## Riesgos

- Plazo: seis semanas; la veda y ECS van primero, la deuda del núcleo después, `publish.py` lo último (si no
  cabe, se queda para después de la elección).
- ALB: si la VPC usa un rango fuera de los privados o aparece un CDN delante, `limit_req` degrada;
  `check-api.sh` lo detecta comparando la IP que nginx registra con la del cliente (comprobación nueva).
- Memoria de la tarea de jobs: si `publish forecast all` no cabe en 4 GB, se sube la tarea o se publica por
  ámbitos; medir antes del 23.
- Rol de tarea: si s3fs no toma el rol con `key=None`, se cae al modo con claves en la definición de tarea
  (sin cambio de código).
- Congelación pública: el interruptor se toca dos veces; un error de texto se corrige con otro `manifest`
  (≤ 2 min de caché).

## Lo que queda al constructor

El texto exacto del banner y del pie en la congelación (en español, siguiendo las plantillas), la forma de las
plantillas de `deploy/ecs/`, el orden de los submódulos del paquete `publish`, las medidas de la prueba de carga
(herramienta y cifras), la medida del RSS de `publish forecast es` y los detalles de los tests. El plan de tareas se escribe con `writing-plans` tras
la aprobación de esta spec, con la veda y ECS como primeras tareas.
