# Jobs como pipelines de MTpy y tarea de jobs en ECS

Fecha: 2026-10-31. Estado: aprobado por Luis el 2026-10-31. Planes: uno por fase, en `docs/superpowers/plans/`. Fase 1 construida el 2026-10-31 (8e30900..2bbd057, 1.653 tests; revisión final de la rama limpia; libro de a bordo en `docs/superpowers/plans/2026-10-31-jobs-pipelines-fase-1-ledger.md`). Fase 2 construida el 2026-10-31 (9b2f849..bc35a30, 1.683 tests; revisión final de la rama limpia; libro de a bordo en `docs/superpowers/plans/2026-10-31-jobs-pipelines-fase-2-ecs-ledger.md`).

Revisión del constructor (2026-10-31): cinco preguntas. Tres resueltas en este documento (el manifest se
escribe una vez al final, como hoy, con un gancho `finish`; `SIGTERM` pasa por el aviso; el worker de cola
captura el fallo) y dos contestadas el mismo día: `load params` se queda en el portátil y `backtest` corre en ECS con sus
resultados en S3 (ver "Backtest en S3"); la red de la tarea se leyó de la cuenta con la CLI (ver
"Infraestructura real"): subredes públicas con IP pública, así que Pushover y Wikipedia llegan.
Decidido por Luis (2026-10-31): la copia de `backtest/results` del repositorio sigue siendo el registro que se
commitea; `pull` es un paso del flujo.

## Intención

Luis sigue lanzando él los jobs (`load`, `compute`, `backtest`, `publish`), pero quiere que corran en AWS, donde
ya vive la web: como tareas puntuales de ECS Fargate, con los logs en CloudWatch y un aviso por Pushover al
terminar, haya ido bien o mal. Para eso los jobs pasan a usar el sistema de pipelines de MTpy (`SequenceJob` y
`Pipeline` de `mtpy/core/worker.py`, descrito en el README de MTpy), que ya trae la ejecución por secuencia, el
log por etapas y el aviso final; y después se completa la tarea de jobs de ECS que la fase 6 dejó en plantilla
(`deploy/ecs/task-job.example.json`, `RUN_JOB` en `deploy/docker/init.sh`).

Dos fases, en este orden:

1. **Adaptación de los jobs al sistema de pipelines.** Cada acción de hoy (`what`) es un `Pipeline`; cada job,
   un `SequenceJob` sobre su grupo.
2. **Tarea de ECS** que arranca el contenedor con un job dado (`RUN_JOB`) y procedimiento para lanzarla desde el
   portátil.

## Por qué ahora

Campaña hasta el 23 de noviembre: carga de sondeos y dos publicaciones diarias (`es` con 10.000 simulaciones y
los autonómicos). `es` tarda 11 minutos contra la RDS desde el portátil y depende de la conexión (ya ha habido
cuelgues silenciosos). La fase 6 dejó la tarea de jobs en plantilla, pero lanzarla es editar un JSON a mano,
buscar el log en CloudWatch y acordarse de `GIT_COMMIT` con cada imagen; y si el job revienta no hay aviso.

## Objetivos

1. Un pipeline por acción en `mtpy/pipelines/<job>/<Accion>.py` y los cuatro jobs como `SequenceJob` de su
   grupo. Desde el job se lanza cualquier combinación de acciones (`what`) sobre cualquier lista de ámbitos
   (`scopes`), y un pipeline suelto con `job.py pipeline` o `pipeline.py`.
2. Un único aviso al final por Pushover con el resumen (los fallos delante), también cuando el job revienta
   fuera del bucle por ámbitos; y salida distinta de cero si hubo fallos, para que ECS y el portátil lo vean.
3. Un parámetro del job, `stop_on_error`, para elegir entre parar al primer fallo o seguir con todos e informar
   al final.
4. Los comandos y runbooks de hoy siguen valiendo: `python job.py publish '{"what":[...],"scopes":[...]}'`,
   `RUN_JOB='publish {...}'` y los parámetros documentados en `deploy/README.md`.
5. La tarea de ECS se lanza desde el portátil con un comando, con los logs en CloudWatch (una sola copia) y el
   aviso de Pushover.
6. El backtest corre en ECS: sus tablas viven en S3 (el sistema de ficheros de la app), de donde las leen
   `publish backtest` y `publish event` sin reconstruir la imagen; el repositorio conserva una copia para el
   registro.

## No objetivos

- Programar los jobs (EventBridge Scheduler, cron del contenedor): Luis los lanza. La decisión de 2026-10-07 de
  que la publicación es manual sigue.
- El worker de cola (`APP_QUEUE`, SQS): no se toca.
- Partir las funciones por ámbito de `mtpy/lib` (`publish_forecast`, `publish_analysis`, `publish_event`,
  `compute_scope`, `rate_scope`, `load_polls`, `load_params`, `run_backtest`) en fases: siguen siendo la
  unidad de trabajo, con sus tests unitarios y de integración (ver "Enfoques").
- Cambiar el modelo, el contenido de los paquetes publicados, el orden de las acciones o el formato de las
  líneas de resumen por ámbito.
- Crear la infraestructura en AWS (definición de tarea, roles, secretos): es de Luis; aquí van las plantillas y
  el runbook.
- Los pipelines `events` (`Events.py`, `EventsResults.py`): sin tocar.
- Correr `load params` en ECS: reescribe `data/params.json`, que se revisa y se commitea; se queda en el
  portátil (decisión de Luis, 2026-10-31).
- Separar la web de `mtpy` (más adelante, como está decidido).

## Lo que sería un fracaso

- Un job que publica bien y termina en error (hoy pasaría con `publish forecast all`: el resumen supera los
  1.024 caracteres de Pushover, `Pushover.send` lanza y el job muere después de publicar), o un job que falla y
  cuya tarea de ECS aparece como bien acabada (hoy `SequenceJob` vuelve con salida 0 tras avisar del error).
- En modo continuar, un ámbito roto deja sin publicar a los demás.
- Cada línea del job dos veces en CloudWatch (hoy la plantilla lleva `CWL_GROUP` y el driver `awslogs`).
- Una tarea de ECS sin nada que hacer que se queda en `tail -f /dev/null` facturando 2 vCPU y 4 GB.
- Perder los escenarios de los tests de los jobs de hoy (es-ar rompe con `KeyError`, es-ri falla con
  `ValueError` del paquete, es-cb sin sondeos → `skipped`, veda → `refused`, `backfill` con `replace`, relleno
  por días, `event` con varias fechas, `point`/`unpublish` sin `run` → `ValueError`...).
- Que `python job.py load '{"scopes":["es"],"overwrite":true}'` o cualquier otro comando del README deje de
  funcionar o cambie de significado.
- Un job que corre en ECS y escribe en el árbol del contenedor (`data/params.json`, o `backtest/results` sin
  pasar por S3) como si fuera a persistir.
- `publish event` desde ECS sin las tablas del backtest (hoy publicaría el evento sin la lámina del modelo,
  `has_model: false`, pisando el evento vivo).

## Restricciones

- Estructura de MTpy, tal cual la describe su README: `Pipeline` con `pipes = {acción: {etapa: [tareas]}}`,
  etapas `Setup`, `Extract`, `Transform`, `Load`, `Build`, `Set`, tareas como métodos `<tarea>_task`;
  `SequenceJob` con `sequence = {grupo: [pipelines]}`, `default_action`, `select`, `action` y `chunks`; un
  pipeline se ejecuta una vez por chunk. `mtpy/core/worker.py` es código del framework: todo cambio ha de ser
  compatible hacia atrás (`Update.py`, `Pipeline.py` y los pipelines `events` siguen funcionando igual).
- Identificadores en inglés (regla de Luis); textos de consola y del resumen en castellano, como hoy.
- Los pipelines y los jobs siguen aceptando las fábricas inyectables que usan los tests (`computer`, `loader`,
  `params_loader`, `catalogues`, `featured`, `fs`, `runner`, `stats`, `simulator`...): llegan por `**kwargs` a las
  tareas.
- `job.py` y `RUN_JOB` no cambian de interfaz: el nombre del job y un JSON. `init.sh` sigue pasando el JSON como
  un solo argumento.
- `tests/test_deploy_files.py` vigila las plantillas de `deploy/ecs/`, `init.sh` y `supervisord.conf`: cambia
  con ellas.
- Hoy hay 1.557 tests; al cerrar cada fase, todos en verde.
- Luis trabaja en el mismo árbol y hace commits en paralelo: comprobar `git status` antes de cada commit.

## Enfoques considerados

**Fase 1.**

- A. Sólo el ciclo de vida: `job.py` pasa por `dispatch`, la clase `Job` envuelve `run` con duración, aviso
  final recortado y salida distinta de cero; los cuatro jobs quedan como están. Descartado por Luis: quiere
  los jobs en el sistema de pipelines.
- B. **Elegido.** Un `Pipeline` por acción, con las funciones por ámbito de `mtpy/lib` como unidad de trabajo
  de una tarea, y lo que hoy hace el job alrededor (resolver parámetros, guardas, estados `skipped`/`refused`,
  entrada del manifest, puntero) repartido en `Setup` y `Set`. Los jobs quedan como `SequenceJob` finos que
  traducen `what`/`scopes` a `select`/`chunks`.
- C. Como B, pero con las fases del modelo como tareas (`init`, `fit`, `nowcast`, `forecast`, `export`), partiendo
  `publish_forecast` y `compute_scope`. Descartado ahora: el `Simulator` exporta cada modo antes de cambiar al
  siguiente (una sola simulación en memoria), así que separar `Transform` de `Load` obliga a reescribir
  `forecast.py` o a duplicar la memoria; y esas funciones tienen sus propios tests unitarios y de integración.
  Es un paso posible después del 29-N.

**Fase 2.**

- Tarea puntual de ECS con `RUN_JOB` (`run-task` sobre la definición `elections-model-job`). **Elegido**: ya
  está en plantilla y es lo que pide Luis.
- Ejecutar dentro del contenedor web (ECS Exec) o con el cron del contenedor: descartados, la tarea web tiene
  1 vCPU, 2 GB, `DB_STATEMENT_TIMEOUT` de 10 s y rol de sólo lectura en S3.

## Diseño de la fase 1

### Estructura

```
mtpy/pipelines/
  load/      Polls.py  Params.py
  compute/   Catalogues.py  Compute.py  Ratings.py
  backtest/  Backtest.py
  publish/   Forecast.py  Analysis.py  Event.py  Backtest.py  Study.py  Parties.py  Manifest.py  Point.py  Unpublish.py
  events/    (como está)
```

Cada job de `mtpy/jobs/` pasa a `SequenceJob` con su grupo en `sequence` (los pipelines en el orden canónico de
hoy), `default_action` y `stop_on_error = False` (el comportamiento de hoy: seguir y resumir). Su `run` conserva
la firma documentada (`what`, `scopes` y los parámetros de cada job), hace la validación que cruza pipelines
(`event` no se combina con `forecast`, `backfill` necesita `forecast`, `run` es obligatorio para `point` y
`unpublish`, `event_date` para `event`, `study` para `study`, `years` no va con `overwrite`), resuelve
`scopes` (`"all"` consulta la base) y llama a `super().run(select=..., chunks=..., **params)`. Devuelve los
resultados con la forma que documenta hoy cada job (`{ámbito: resultado}`, listas en `backfill` y `event`,
por acción en `load` con varios `what`).

| Job | Grupo | Pipelines (orden) | Acción por defecto |
| --- | --- | --- | --- |
| `load` | `load` | Polls, Params | `update`; Polls con `overwrite` corre `rebulk` |
| `compute` | `compute` | Catalogues, Compute, Ratings | `update` |
| `backtest` | `backtest` | Backtest | `run` |
| `publish` | `publish` | Forecast, Analysis, Event, Backtest, Study, Parties, Manifest, Point, Unpublish | `publish` |

`what` selecciona pipelines del grupo (`select` con entradas `grupo.pipeline`, ver "Núcleo"). Como hoy,
`forecast` y `analysis` añaden `parties` a la selección (el catálogo de partidos se reescribe una vez por run
del job). Decisiones mías: las acciones de `Polls` (`update` incremental, `rebulk` con `overwrite`, que es
donde las etapas difieren de verdad) y una sola acción en los demás pipelines, con `dry_run` y `force` como
parámetros, como hoy.

### Pipelines

Cada pipeline tiene una acción con las etapas que hoy se puedan separar. La unidad de trabajo es la función
por ámbito de `mtpy/lib`; lo que hoy hace el job alrededor va a `Setup` (resolver `event_date`, guardas,
comprobar el run existente, el plan de días del relleno) y a `Set` (entrada del manifest, puntero, registro).
Un pipeline que decide en `Setup` que el chunk se salta o se rechaza (`skipped`, `refused`) termina ahí (ver
"Núcleo"); un fallo de la unidad de trabajo se captura dentro del pipeline, como hoy en `_x_scope`, y deja
`status: failed` con la razón y el traceback en el log.

| Pipeline | Unidad de trabajo | Por chunk (ámbito) | Notas |
| --- | --- | --- | --- |
| load.Polls | `load.load_polls` | sí | `update` lee y selecciona lo nuevo; `rebulk` todo el ciclo (`overwrite`) |
| load.Params | `load.load_params` | sí | |
| compute.Catalogues | `compute.replicate_catalogues` | no | una vez por secuencia |
| compute.Compute | `compute.compute_scope` | sí | |
| compute.Ratings | `compute.rate_scope` | sí | |
| backtest.Backtest | `backtest.backtest_scope` | sí | acciones `run`, `push` y `pull`; las tablas van al sistema de ficheros de la app (ver "Backtest en S3") |
| publish.Forecast | `publish.publish_forecast` | sí | el relleno (`backfill`) itera los días dentro del chunk; las entradas del manifest se escriben una vez, al acabar todos los chunks (`finish`) |
| publish.Analysis | `publish.publish_analysis` | sí | |
| publish.Event | `publish.publish_event` | sí | varias fechas dentro del chunk (`event_date` lista o `"all"`); el corte del modelo sale del mismo sistema de ficheros |
| publish.Backtest | `publish.publish_backtest` | sí | lee las tablas del sistema de ficheros de la app (`source` local opcional) |
| publish.Study | `publish.publish_study` | no | el chunk no aplica; `study` y `register` por parámetro |
| publish.Parties | `publish.publish_parties` | no | |
| publish.Manifest | `publish.update_manifest` | no | `freeze` por parámetro |
| publish.Point | `publish.point` | sí | |
| publish.Unpublish | `publish.unpublish` | sí | |

El manifest se escribe como hoy: una vez, al acabar todos los ámbitos, con las entradas de los publicados (un
ámbito fallido no entra; en `dry_run` no se escribe), en el gancho `finish` del pipeline (ver "Núcleo", punto
11). Así el resultado de un ámbito publicado sigue siendo `published` aunque falle el manifest, el resumen
lleva una sola línea `manifest: ...` y el fallo del manifest es un fallo aparte del job (`manifest`), como en
`test_manifest_write_failure_keeps_the_summary`. Para el runbook: el manifest es un leer-modificar-escribir
sin bloqueo, así que no se lanzan dos publicaciones a la vez (hoy tampoco: las dos llamadas diarias van una
tras otra).

Las líneas de progreso (`ScopeReporter`, `ScopeProgress`: `[i/n] ámbito: empezando`, fases con segundos,
avisos, línea de resultado) se quedan dentro de las tareas con los mismos niveles de verbosidad (0 a 3). Las
cabeceras de etapa de `Pipeline.run` (`Run publish.Forecast.es pipeline...`, `--> Setup...`) son las de MTpy y
van por `app.logger.info`.

### Núcleo (`mtpy/core/worker.py`), cambios compatibles hacia atrás

1. `SequenceJob.stop_on_error`: atributo de clase, `True` por defecto (el comportamiento de MTpy de hoy: parar
   al primer fallo); `run(..., stop_on_error=None)` lo sobreescribe por ejecución (`{"stop_on_error": true}` en
   el JSON del job). Los cuatro jobs de elecciones lo ponen a `False`.
2. Un chunk falla cuando su pipeline lanza una excepción o deja `status: failed`. En modo parar: se registra,
   se avisa y se relanza. En modo continuar: se registra con el traceback en el log y la secuencia sigue con el
   resto de chunks y pipelines.
3. `select` admite entradas `grupo` (todos sus pipelines) y `grupo.pipeline`; el orden es siempre el declarado
   en `sequence`.
4. `Pipeline.chunked = True` (atributo de clase); con `False`, `SequenceJob` lo ejecuta una vez por secuencia,
   con `chunk=None`.
5. Un pipeline deja en `self.result` el resultado del chunk (diccionario con `status` y lo demás como hoy) y
   en `self.line` su línea de resumen; `SequenceJob` los recoge en `self.results` y `self.lines`, y `run` los
   devuelve. `Pipeline.run` los pone a `None` al empezar cada chunk.
6. Una tarea puede terminar el chunk antes de tiempo con un estado que no es fallo (`skipped`, `refused`):
   `Pipeline.run` no ejecuta las etapas restantes y el resultado lleva ese estado y la razón. Nombre y
   mecanismo (método `skip(status, reason)` o excepción de control), a elección del constructor.
7. Aviso final único: duración del job, número de fallos y las líneas de resumen, con los fallos delante.
   `Job.alert` manda a Pushover como mucho 1.024 caracteres (si recorta, lo dice en la última línea y remite al
   log) y nunca lanza: un error de Pushover va a `logger.error`. El texto completo va siempre al logger.
8. Tras el aviso, en modo continuar, `run` lanza `RuntimeError('<job>: failed ...')` si hubo fallos; en modo
   parar relanza la excepción del chunk. En los dos casos el proceso termina con salida distinta de cero.
9. `job.py` llama a `dispatch(**params)` (entran `before_dispatch` y `after_dispatch`) y termina con salida 1 si
   el job lanza. El job `Pipeline` (`mtpy/jobs/Pipeline.py`), que es el lanzador de un pipeline suelto, se
   arregla de paso: con un `select` sin punto `module` no está definido.
10. `SequenceJob` pasa a cada ejecución del pipeline la posición del chunk (índice y total) para el prefijo
    `[i/n]`; el nombre del parámetro, a elección del constructor.

11. `Pipeline.finish(**kwargs)`: gancho que `SequenceJob` llama una vez al acabar el último chunk de un
    pipeline (también si es `chunked = False`, tras su única ejecución); no hace nada por defecto. Un fallo en
    `finish` es un fallo del job con su propia línea de resumen (como hoy `manifest (failed)`), no un estado
    de ningún chunk. `publish.Forecast` escribe ahí las entradas del manifest.
12. `SIGTERM` (el `stop-task` de ECS, que con `exec` llega a Python): `job.py` instala un manejador que lo
    convierte en una excepción, de modo que el job pasa por el mismo camino que un fallo (línea `interrumpido`
    en el log, aviso de Pushover, salida distinta de cero). El `stopTimeout` de ECS (30 s por defecto) da de
    sobra para el aviso. El run a medias queda incompleto y se retira con `unpublish` (runbook).
13. `Worker.handle_message` captura y registra la excepción de `dispatch`, para que el bucle de la cola no
    muera con el primer job que falle (hoy `SequenceJob` no lanzaba; con el punto 8 sí). Aquí `APP_QUEUE` no
    se usa; es sólo compatibilidad del framework.

Sin fallos, ningún job de MTpy que ya exista cambia de comportamiento.

### Tests

- `tests/test_worker_unit.py` crece con el núcleo: `select` con `grupo.pipeline`, parar y continuar, `chunked`,
  el fin anticipado del chunk, el recorte y la resistencia del aviso, la excepción final.
- Los escenarios por ámbito de `tests/test_jobs_*.py` pasan a tests por pipeline (`tests/test_pipelines_*.py`,
  un fichero por grupo o por pipeline, a elección del constructor), con las mismas fábricas de
  `tests/fakes.py`.
- `tests/test_jobs_*.py` se quedan con lo que es del job: la traducción `what`/`scopes`, las validaciones
  cruzadas, la forma de los resultados y el contenido del aviso.
- Los tests de `mtpy/lib` (unitarios e integración) no se tocan.

## Diseño de la fase 2

### Infraestructura real (leída de la cuenta el 2026-10-31, sólo lectura)

Las plantillas de `deploy/ecs/` describen una infraestructura que no es la desplegada. Lo que hay:

- Clúster `elections-model-cluster`, servicio `elections-model-service`, definición de tarea
  `elections-model-cluster-elections-model-service` (revisión 33), contenedor `Main`, imagen
  `elections-model:latest` de ECR, Fargate 1.4.0, 1 vCPU y 2 GB, `enableExecuteCommand` activo.
- Rol de tarea y de ejecución: los dos `ecsTaskExecutionRole` (políticas `AmazonECSTaskExecutionRolePolicy`,
  `ecsExecPolicy`, `envSecretsPolicy` y `s3EnvFilePolicy`).
- La configuración entra por **fichero de entorno en S3** (`environmentFiles`, `s3://elections-s3-bucket/elections.env`):
  la definición no lleva ni `environment` ni `secrets`. Ese fichero es `deploy/elections.env`, con `DB_PASSWORD`,
  `PUSHOVER_USER`, `PUSHOVER_TOKEN`, `AWS_KEY` y `AWS_SECRET` (la app escribe en S3 con esas claves, no con el
  rol), `APP_VERBOSE=1` y `CWL_GROUP` vacío.
- Logs: driver `awslogs`, grupo `/ecs/elections-model`, prefijo `ecs`.
- Red: VPC por defecto (`vpc-53085337`, 172.31.0.0/16), las tres subredes por defecto (`subnet-bb3b4de3`,
  `subnet-48794c2c`, `subnet-1e072968`), todas públicas (tabla de rutas con `0.0.0.0/0 → igw-d38f52b7`, IP
  pública automática), `assignPublicIp: ENABLED` en el servicio, grupo de seguridad `sg-01b9001143bbdbd57`
  (entrada 8042 sólo desde el del ALB, salida libre). Sin NAT ni endpoints de VPC. La tarea web sale a
  Internet por su IP pública: Pushover y Wikipedia llegan si la tarea de jobs va igual.
- No existe ninguna definición de tarea de jobs.

Consecuencias para el diseño:

- **La definición `elections-model-job` se deriva de la viva**, no de una plantilla con marcadores: el lanzador
  (`--register`) lee la definición actual del servicio web con `describe-task-definition` y registra la de jobs
  con lo mismo (imagen, roles, fichero de entorno, grupo de logs) y lo que cambia: familia, 2 vCPU y 4 GB, sin
  puertos, prefijo de logs `job`, `RUN_JOB` vacío. Así nunca se desvía de la web y no hay marcadores que
  rellenar. Las plantillas de `deploy/ecs/` se reescriben como la salida de ese paso (referencia), con la
  nota de que la configuración va por fichero de entorno.
- **Red y seguridad del `run-task`**: el lanzador toma las subredes, el grupo de seguridad y `assignPublicIp`
  del despliegue actual del servicio (`describe-services`, `deployments[0].networkConfiguration`); `ecs.env`
  sólo necesita el clúster y el servicio, y admite sobreescribirlos.
- **Pushover y la verbosidad** ya están en el fichero de entorno: la fase 2 no toca secretos ni variables.
  `RUN_JOB` del override gana al fichero de entorno (ECS: `environment` y los overrides tienen prioridad sobre
  `environmentFiles`).
- `GIT_COMMIT` sigue horneado en la imagen con `--build-arg`: la definición viva no lleva variables.
- El README ("Despliegue en ECS", marcadores) se corrige para describir esto.

### Qué corre en ECS

- **En ECS**: `load polls`; `compute` entero; `backtest` (ver "Backtest en S3"); `publish` entero
  (`forecast`, `analysis`, `event`, `backtest`, `study`, `parties`, `manifest`, `point`, `unpublish`), con
  las tablas del backtest leídas de S3.
- **En el portátil**: `load params` (reescribe `data/params.json`, que se commitea). El lanzador rechaza `load`
  con `params` en `what` con un mensaje; el código de los jobs no sabe nada de ECS.

### Backtest en S3

Hoy `backtest_scope` escribe las ocho tablas y `run.json` con rutas locales en `backtest/results/` del
repositorio (`es` en la raíz, cada autonómico en su carpeta; 162 ficheros, 14 MB; `series.csv` fuera de git),
y `publish_backtest`, `export_event_model` y `export_event_series` las leen con `pd.read_csv` de ese mismo
árbol. En Fargate ese árbol muere con la tarea.

Diseño:

- **Las tablas viven en el sistema de ficheros de la app** (`app.fs`: S3 cuando `S3_BUCKET` tiene valor, en ECS
  y en el portátil con el `.env` de Luis; `files/` en local sin bucket), bajo el prefijo `backtest/results/`
  con la misma estructura que hoy (`es` en la raíz, los demás en `backtest/results/{ámbito}/`). Es un prefijo
  nuevo del bucket, al lado de `site/` y `site-dry/`; la web no lo lee.
- **Escritura y lectura por `FileSystem`**: `backtest_scope` escribe por `fs.write_csv` y `fs.write_json`
  (`FileSystem` local o `S3`, misma API), y `backtest_source` devuelve un `FileSystem` y una carpeta, de la
  que `publish_backtest`, `export_event_model` y `export_event_series` leen con `fs.read_csv` y `fs.exists`.
  `out` (del backtest) y `source` (de `publish`) siguen aceptando una carpeta local, envuelta en
  `FileSystem(carpeta)`: el árbol del repositorio sigue sirviendo en el portátil, en los tests y para las
  variantes del método (`"out":"backtest/results_comp"`). Sin `out`/`source`, el destino y el origen son
  `app.fs` + `backtest/results`.
- **Acciones del pipeline `backtest.Backtest`**: `run` (calcular y escribir el ámbito en `app.fs`), `push`
  (subir la carpeta del ámbito del árbol del repositorio a `app.fs`: la siembra inicial con los resultados de
  hoy y las ejecuciones del portátil) y `pull` (bajar la carpeta del ámbito de `app.fs` al árbol del
  repositorio, para revisar `by_horizon.csv` contra la ejecución anterior y commitear). `push` y `pull`
  sobrescriben los nueve ficheros del ámbito y no tocan los demás ámbitos.
- **Flujo tras un cambio del método**: `deploy/ecs/job.py backtest '{"scopes":"all"}'` en ECS → S3; en el portátil
  `python job.py backtest '{"action":"pull","scopes":"all"}'`, revisar el `diff` y commitear; después
  `deploy/ecs/job.py publish '{"what":["backtest","event"],...}'` lee S3 directamente. La limitación de hoy (`series.csv`
  sólo en la máquina del backtest) desaparece.
- **Decidido (Luis, 2026-10-31)**: la copia del repositorio sigue siendo el registro que se commitea y `pull`
  es un paso del flujo, porque el repositorio es el registro científico y los `diff` de `by_horizon.csv`
  son el criterio de aceptación de cada cambio del modelo. `backtest/results` no sale de git y `series.csv`
  sigue ignorado.
- Permisos: la app escribe en S3 con las claves del fichero de entorno, las mismas de `site/`; no hace falta
  ningún permiso nuevo.

1. **Definición de tarea de jobs** derivada de la viva (ver "Infraestructura real"): `deploy/ecs/job.py --register`
   la registra o la actualiza; las plantillas de `deploy/ecs/` pasan a ser la referencia de lo que genera.
   `GIT_COMMIT` se hornea en la imagen (`ARG GIT_COMMIT` → `ENV GIT_COMMIT` en el `Dockerfile`, y el `docker
   build` del README lleva `--build-arg GIT_COMMIT=$(git rev-parse --short HEAD)`).
   `tests/test_deploy_files.py` y la tabla de marcadores del README cambian con ello.
2. **`deploy/docker/init.sh`**: `exec python /app/job.py ...` (la señal de `stop-task` llega a Python y el
   código de salida del job es el del contenedor); sin `RUN_JOB` y sin ningún servicio, mensaje y salida 1 en
   vez de `tail -f /dev/null`.
3. **Lanzador `python deploy/ecs/job.py <job> '<json>' [--follow]`** (en Python con boto3, decidido por Luis el
   2026-10-31 en lugar del `deploy/run-job.sh` en bash que se había previsto; misma función): lee el clúster y el servicio de
   `deploy/ecs.env` (ignorado por git y por `.dockerignore` como todo `*.env`; plantilla
   `deploy/ecs.env.example`; subredes, grupo de seguridad e IP pública salen del servicio y se pueden
   sobreescribir ahí), rechaza un job vacío y `load params` (ver "Qué corre en ECS"), construye el `run-task` con el
   override `RUN_JOB='<job> <json>'`, lo lanza e imprime el id de la tarea, el comando `aws logs tail` del flujo
   y el enlace a la consola. Con `--follow` sigue el log hasta que la tarea para y termina con el código de
   salida del contenedor. `run-task.example.json` se queda como referencia de lo que genera.
4. **README** (`deploy/README.md`): sección «Jobs desde ECS» en lugar de «Publicar desde ECS» (lanzador, dónde
   está el log, el aviso, códigos de salida, `stop_on_error`, publicaciones una tras otra, una tarea parada a
   medias deja un run incompleto que hay que retirar con `unpublish`, coste del orden de 10 céntimos por hora
   de tarea), el `build` con `GIT_COMMIT`, y los pasos de «Veda» y «Campaña» apuntando al lanzador.
5. **Pasos de Luis** (fuera de la construcción): `deploy/ecs/job.py --register` la primera vez, `backtest push` de
   todos los ámbitos para sembrar S3 con los resultados de hoy, y medir la memoria de `publish forecast es`
   con 10.000 simulaciones en la primera tarea (4 GB sin medir).

## Lo que queda al constructor

- El reparto del cuerpo de cada `_x_scope` de los jobs de hoy entre `Setup`, la tarea de trabajo y `Set`; los
  nombres de las tareas.
- Los nombres del parámetro de posición del chunk, del mecanismo de fin anticipado y de `result`/`line`.
- `Pipeline.__init__` lee `self.app.config.app.compression` y `get_model`: los tests de pipelines necesitan un
  `App` con `config` (o el núcleo tolera su ausencia; cambio compatible).
- El reparto de los tests por pipeline y qué se queda en los del job.
- El texto exacto de la cabecera del aviso (nombre del job, duración, fallos).
- Los detalles del lanzador (construcción del JSON con `python` o `jq`, `aws logs tail`, un bucle de
  `describe-tasks` para el código de salida; `--register` con `describe-task-definition` y
  `register-task-definition`).
- Cómo `backtest_source` devuelve sistema de ficheros y carpeta, y cómo `push`/`pull` listan los nueve
  ficheros (`BACKTEST_FILES` más `series.csv`, `metrics.csv`, `provinces.csv`, `by_horizon.csv`, `run.json`).

## Tamaño

Proyecto. La fase 1 es grande (cuatro jobs a quince pipelines, cambios en el núcleo y los tests); la fase 2 es
pequeña. Un plan por fase; la fase 2 no depende de la 1 salvo por el código de salida y el aviso, así que puede
construirse en cuanto la 1 esté en verde.
