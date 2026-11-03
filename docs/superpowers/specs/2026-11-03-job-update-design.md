# Job `update`: carga, compute y publicación en una sola tarea

Fecha: 2026-11-03. Estado: aprobada por Luis el 2026-11-03; construida el 2026-11-03 (plan docs/superpowers/plans/2026-11-03-job-update.md).

Pendiente de Luis: el `ALTER TABLE elections.events ADD COLUMN loaded_at timestamp, ADD COLUMN computed_at timestamp;` en el RDS y en cualquier base local, antes de desplegar la imagen y antes de usar `dev` contra cualquier base (local incluida), porque sin las columnas falla toda lectura de `events`, la web incluida (el orden inverso es inocuo); desplegar la imagen con el código nuevo; y el primer `update` o `compute` sin `overwrite`, que computa una vez todos los ámbitos pedidos porque toda elección cuenta como pendiente.

Desviaciones de la construcción: ninguna en las decisiones. Dos añadidos en las pruebas: las suites de los pipelines y de los jobs que ejecutan `compute_scope` con `save=True` llevan un fixture autouse que sustituye `db_now` y `mark_computed`, de modo que ninguna prueba toca la base; y las precisiones de documentación (parada en el ámbito que falla, excepción original, «sólo»).

## Intención

Un job `update` que encadene los tres comandos que hoy Luis lanza uno tras otro:

```
python job.py load '{"scopes":["es"],"verbose":3}'
python job.py compute '{"what":["compute"],"scopes":["es"],"verbose":3}'
python job.py publish '{"what":["forecast","analysis"],"scopes":["es"],"n_sim":10000,"verbose":3}'
```

en una sola llamada, `python job.py update '{"scopes":["es"],"n_sim":10000,"verbose":3}'`, y por tanto en una
sola tarea de ECS (`python deploy/ecs/job.py update '{...}'`), con un solo resumen y un solo aviso de Pushover.
El compute de un ámbito sólo corre cuando hace falta: cuando la base guarda algo cargado para ese ámbito después
de su último compute. Para saberlo, la base lleva dos marcas de tiempo por elección.

## Por qué ahora

Cada actualización diaria del forecast de las generales del 29-N son hoy tres tareas de ECS (tres arranques de
contenedor, tres avisos) o tres comandos a mano. El compute es la parte lenta (los errores del promedio ajustan
una vez por elección y horizonte) y la mayoría de los días no hay sondeos nuevos. Luis ya ha escrito la
`sequence` de los tres grupos en `mtpy/jobs/Update.py` (sin commit): la pregunta era si bastaba con eso.

## Objetivos

1. Un job `update` con la secuencia load (polls) → compute (compute) → publish (forecast, analysis, parties)
   sobre los `scopes` pedidos (también `all`), con los mismos parámetros que hoy reciben los tres jobs, un
   resumen y un aviso.
2. Saber en la base, por ámbito, si hay algo cargado desde el último compute: marcas `loaded_at` y
   `computed_at` en `elections.events`, escritas por los cargadores y por `compute.compute_scope`.
3. Saltar el compute de un ámbito sin nada pendiente, con una línea de resumen que lo diga (`skipped`).
4. Un único interruptor de «rehacerlo todo», `overwrite`: la carga reemplaza el ciclo entero (como en `load`) y
   el compute corre aunque no haya nada pendiente.
5. El mismo criterio en el job `compute`: con `overwrite` falso (por defecto) sólo computa los ámbitos con algo
   pendiente, lo que abarata `scopes: "all"`.

## No objetivos

- Un `select` de grupos en `update`: siempre corre la cadena entera (Luis, 2026-11-03). Para un grupo suelto
  están los jobs `load`, `compute` y `publish`.
- Meter en `update` las demás acciones de los tres jobs (`params`, `catalogues`, `ratings`, `event`, `backtest`,
  `study`, `point`, `unpublish`, `backfill`): siguen en su job.
- Un planificador: Luis lanza la tarea.
- Marcas de tiempo en `polls` o en las tablas del compute, o cambiar el significado de `polls.computed`
  («pesos calculados», que la propia carga pone a verdadero).
- Marcar el compute hecho desde el cuaderno `PollsCompute.ipynb`: sólo marca `compute_scope` (el job). Un
  compute hecho a mano deja el ámbito como estaba: pendiente o no; a lo sumo el siguiente `update` recalcula de
  más.
- Cambios en `mtpy/core/worker.py`: no hacen falta.

## Lo que sería un fracaso

- Publicar un forecast sin computar sondeos cargados fuera de la propia ejecución (del portátil o del cuaderno),
  que es el caso que descarta la regla «computar si esta carga guardó algo».
- Tres avisos de Pushover y tres resúmenes en vez de uno.
- Publicar el forecast sobre un compute que se quedó a medias.
- Que el job `compute` salte un ámbito en silencio: la línea de resumen ha de decir que lo saltó y por qué.
- Que las marcas se pierdan con los upserts de `events` que ya existen (`save_event` del cargador de resultados,
  `update_featured`).

## Restricciones

- El `ALTER TABLE` lo hace Luis a mano en la RDS (regla del proyecto); el código no crea ni altera tablas, sólo
  actualiza el modelo de `mtpy/models` y escribe en las columnas.
- Identificadores en inglés; textos de consola y de resumen como hoy (cabeceras en castellano, estados y razones
  en inglés, como `skipped (no upcoming event)`).
- Los jobs y los pipelines siguen aceptando las fábricas inyectables de los tests (`loader`, `computer`,
  `featured`, `catalogues`, `fs`, `simulator`...) y no tocan la base en los tests. Ningún test escribe en
  `backtest/results` ni corre contra la base del `.env`.
- `job.py`, `RUN_JOB` y el lanzador `deploy/ecs/job.py` no cambian de interfaz: nombre del job y un JSON.
- `SequenceJob` pasa los mismos kwargs a todos los pipelines de la secuencia, que ignoran los que no conocen.
  `force` llega a la vez al cargador (altas de casas y medios) y a la publicación (salta la guarda de la LOREG);
  ver la decisión 6.

## Alternativas consideradas

- **Saber lo pendiente con la regla de la ejecución**: computar un ámbito si la carga de esta misma ejecución
  guardó sondeos (`n_saved > 0` u `overwrite`). Sin cambio de esquema, pero ciega a lo cargado desde el portátil
  o el cuaderno, que es el uso habitual de Luis antes de publicar desde ECS. Descartada por él (2026-11-03).
- **Reutilizar `polls.computed`**: la fase de pesos de la carga lo pone a verdadero en todo el ciclo
  (`compute_weights(overwrite=True)`), así que no distingue «pesos calculados» de «compute del job hecho».
  Compararlo con los `n` de rebaño o de efectos de casa dependería de los filtros de tipo y de serie.
- **Marcas en `polls`** (`loaded_at` por fila): la tabla grande, un `ALTER` más caro, y el cargador reescribe las
  filas con upsert (borrado e inserción desde la stage), con lo que la marca sería la del último upsert, no la
  de la carga nueva. Las marcas por elección en `events` bastan y los upserts de `events` van por frame
  (`ON CONFLICT ... DO UPDATE` sólo de las columnas del frame), que respeta las columnas que no van en él.
- **Una tabla nueva de marcas** (`scope`, `action`, `ts`): un `CREATE TABLE`, un modelo y una réplica más por
  dos columnas; `events` ya es la tabla por elección y ámbito que leen el cargador y el compute.
- **Un gancho en `SequenceJob`** para que el job filtre los chunks de cada pipeline: cambia el framework y el
  ámbito saltado no pasa por el progreso del pipeline (la línea la inventaría el job y el contador `[i/n]` del
  compute dejaría de cuadrar con el de la carga). Con la marca en la base, el salto cabe en el propio pipeline
  de compute.
- **Tres llamadas a `super().run`**, una por grupo: cada una manda su aviso y su resumen.
- **Un parámetro `pending_only`** en vez de `overwrite`: Luis prefiere el mismo `overwrite` de `load`
  (2026-11-03), un solo interruptor con el mismo sentido en la carga, el compute y `update`.

## Decisiones

### 1. Esquema y modelo

```
ALTER TABLE elections.events ADD COLUMN loaded_at timestamp, ADD COLUMN computed_at timestamp;
```

`mtpy/models/elections.py`, `Events.meta`: `'loaded_at': 'dts'` y `'computed_at': 'dts'`. Las dos admiten
nulo. `events` no es un catálogo versionado (`save_catalogues` no lo toca), así que no hay réplica CSV que
actualizar.

Los dos escritores que ya hacen upsert de `events` pisarían las marcas si no se les dice lo contrario:
`Model.format_data` añade como nulas las columnas del modelo que faltan en el frame y `build_upsert_query`
escribe en el `ON CONFLICT` todas las columnas del frame (hallazgo 1 de la revisión del constructor). Se
resuelve con el patrón que ya usa el código para `featured`, arrastrar el valor actual:

- `WikipediaResultsLoader.save_event` ya lee la fila actual para conservar `featured`; conserva igual
  `loaded_at` y `computed_at` (una fila nueva las deja nulas).
- `update_featured` lee las elecciones con `get_events` (que con el modelo ampliado devuelve las marcas) y las
  **deja en el frame** del upsert, de modo que vuelven a escribirse con el valor leído. No se quitan del frame:
  `format_data` las repondría como nulas.

Queda una ventana de milisegundos en `update_featured` (una carga que marcase `loaded_at` entre su lectura y su
upsert perdería la marca); cargar y computar el mismo ámbito a la vez ya no está soportado (los dos escriben los
pesos de `polls`), así que se documenta y no se cierra. Cualquier escritor futuro de `events` ha de arrastrar las
marcas como arrastra `featured`; el docstring del modelo lo dice.

### 2. Marca de carga (`loaded_at`)

Un helper `mark_loaded(scope, date)` en `mtpy/lib/loader.py` ejecuta
`UPDATE elections.events SET loaded_at = now() WHERE scope = ... AND date = ...` con el modelo `Events`
(`execute`, como ya hace el `DELETE` del overwrite). El reloj es el de la base en las dos marcas, así se
comparan entre sí sin zonas horarias de por medio. Lo llaman, al final de su guardado y sólo si escribieron
filas:

- `WikipediaLoader.save_polls` (sondeos nuevos o ciclo entero con `overwrite`). Si el frame de sondeos está
  vacío no marca: con `n_read == 0` o sin sondeos nuevos la carga no toca la base.
- `WikipediaResultsLoader` e `InfoElectoralLoader` al guardar los resultados oficiales (`save_results`, tras
  `save_totals`): cargar el resultado de una elección también deja compute pendiente (errores, destacados).

La fila de `events` existe antes (el próximo evento se lee de ella para resolver `event_date`; el cargador de
resultados la crea en `save_event` antes de guardar). Si no existe, el `UPDATE` no marca nada y el ámbito queda
como estaba. Con `dry_run` (`save=False`) no se guarda ni se marca.

### 3. Marca de compute (`computed_at`)

`compute.compute_scope`, con `save`, marca al terminar (después de `average_errors`) **todas** las elecciones
del ámbito: `UPDATE elections.events SET computed_at = <inicio> WHERE scope = ...`. Todas y no sólo las de la
serie del `Computer`, porque el compute es del ámbito entero y una elección con resultado cargado pero sin
sondeos quedaría pendiente para siempre. La marca lleva el instante de **inicio** del compute, leído de la base
(`SELECT now()`) antes de `build_series`, no el del final: lo que se cargue mientras el compute corre no ha
entrado en la serie y debe seguir pendiente (sugerencia del constructor). Lo hace a través de un argumento
inyectable `mark` (`mark(scope, started_at)`), con el real `compute.mark_computed` por defecto, como
`featured`; el instante de inicio se obtiene con un helper inyectable del mismo módulo. Un fallo en una fase
lanza antes de marcar; `dry_run` no marca; `rate_scope` (ratings) ni `replicate_catalogues` marcan.

### 4. Lo pendiente

`compute.pending_events(scope) -> list[str]`: fechas de las elecciones del ámbito con `computed_at` nulo, o con
`loaded_at` posterior a `computed_at`. Lee `Events` (sin base inyectable, como `get_events`). Un ámbito tiene
compute pendiente si la lista no está vacía.

Consecuencia de la regla «`computed_at` nulo es pendiente»: tras el `ALTER`, todas las elecciones están
pendientes y el primer `compute` o `update` sin `overwrite` computa todos los ámbitos que se le pidan una vez,
y los marca. Después sólo computa lo cargado.

### 5. El salto en el pipeline y en el job `compute`

`pipelines/compute/Compute.prepare_task(chunk, position, overwrite=False, pending=None, **kwargs)`: cuando el
pipeline es el de la serie (`self.ACTION == 'compute'`; `Ratings` hereda de `Compute` y no salta nunca) y
`overwrite` es falso, llama a `pending(chunk)` (el real, `compute.pending_events`, por defecto) y, si no hay
nada, termina el ámbito con

```
{'status': 'skipped', 'action': 'series', 'reason': 'nothing loaded since the last compute', 'warnings': []}
```

y la línea `es: series skipped (nothing loaded since the last compute)` (`line_of` ya formatea los estados que no
son `computed`). La cabecera `compute series: n ámbitos` cuenta todos los ámbitos pedidos; con `overwrite` lleva
`· overwrite`, como la de la carga. `catalogues` y `ratings` no cambian.

Job `compute`: parámetro nuevo `overwrite=False` (documentado en la clase y en `run`), que pasa a los pipelines
junto con `pending` (inyectable para los tests). Con `dry_run` el criterio se aplica igual (un ámbito sin nada
pendiente se salta también en seco). El resultado de un ámbito saltado es el diccionario de arriba; si corren
también los ratings, el del ámbito es el de los ratings, como hoy.

Cambio de comportamiento, decisión mía a revisar por Luis: hasta ahora `compute` computaba siempre; a partir de
ahora, sin `overwrite`, salta los ámbitos sin nada cargado desde su último compute. Para recalcular tras un
cambio de código o de parámetros (`params.json`, calidad previa) hay que pasar `"overwrite": true`. La línea de
resumen lo deja a la vista.

### 6. El job `update` (`mtpy/jobs/Update.py`)

```python
sequence = {'load': ['polls'], 'compute': ['compute'], 'publish': ['forecast', 'analysis', 'parties']}
default_action = 'update'
stop_on_error = True
chunk_noun = 'scopes'
```

`run(scopes=('es',), overwrite=False, dry_run=False, verbose=1, stop_on_error=None, **kwargs)`:

- Comprueba primero que hay sistema de ficheros (`self.app.fs`, o el `fs` inyectado), como `Publish.run`, para
  no descubrir tras media hora de compute que no hay bucket (sugerencia del constructor).
- Resuelve `scopes` con `publish.resolve_scopes` (`'all'` consulta la base); sin ámbitos, `ValueError`
  (`update: no scopes to run`). Con `'all'` pasa `skip_missing` a la carga, como `load`.
- `years` con `overwrite` se rechaza (`load.YEARS_WITH_OVERWRITE`), como en `load`.
- `event_date` se rechaza (`ValueError`: `update: "event_date" is not accepted (use load)`): en la carga es
  el ciclo a leer, pero en `publish.forecast` es la elección forzada del forecast, y `SequenceJob` lo entregaría
  a los dos; recargar un ciclo pasado publicaría su «forecast» como run vivo (hallazgo 3 del constructor).
  `update` trabaja siempre con la próxima elección de cada ámbito.
- Llama a `super().run(select=None, action='rebulk' if overwrite else 'update', chunks=targets,
  stop_on_error=..., overwrite=overwrite, dry_run=dry_run, verbose=verbose, **kwargs)`. La acción `rebulk`
  sólo existe en `load.polls`; los demás pipelines caen a su primera acción (`update`, `publish`), que es lo que
  hace hoy `Pipeline.run` con una acción que no está en `pipes`. `overwrite` llega al compute como kwarg (el de
  la carga va en la acción).
- El resto de parámetros pasa tal cual a los tres grupos: `force`, `years` (carga), `n_sim`, `seed`,
  `drange`, `max_fc`, `alpha`, `correctors`, `today` (publicación) y las fábricas de los tests. `fs` lo toman
  los pipelines de publish de la app cuando no llega.
- `parties` va al final del grupo publish, como lo añade el job `publish` con forecast y analysis, para que las
  páginas nombren y coloreen a todos los partidos.
- Devuelve `{scope: {'polls': ..., 'compute': ..., 'forecast': ..., 'analysis': ...}, 'parties': ...}` (los
  resultados de `self.results`, reagrupados por ámbito y nombrados por pipeline).
- **Para al primer fallo por defecto** (`stop_on_error = True`, el de `SequenceJob`, al contrario que los
  otros tres jobs; decisión de Luis, 2026-11-03): el primer ámbito que falle en cualquier grupo detiene la
  cadena, el job avisa con lo hecho hasta ahí y sale con error (salida 1 de `job.py`), y nada de lo que venía
  detrás se publica. Así un compute fallido a medias (`compute_scope` guarda fase a fase: un fallo en `drift`,
  `house_effects`, `herding` o `average_errors` deja errores nuevos con efectos de casa o rebaño viejos) nunca
  llega al forecast, y una carga fallida (una casa sin mapear) no publica ese día: Luis arregla el mapeo y
  relanza. Un fallo en `parties` o en el `finish` del forecast (el manifest) se trata igual.
- Con `"stop_on_error": false` en el JSON sigue como los otros jobs: un ámbito cuya carga falla no se computa
  (no hay marca nueva, salvo que ya estuviera pendiente) y se publica con los sondeos que tenía; uno cuyo
  compute falla se publica sobre lo que haya en la base; el job avisa y sale con error al final (`RuntimeError`
  tras el resumen).
- `force` llega a la carga (da de alta casas y medios sin mapear) y a la publicación (publica dentro de la
  ventana de la LOREG). Se documenta en la clase y en el README: para dar de alta casas, mejor `load` con
  `force` aparte; en `update`, `force` es «publicar aunque estemos en veda».

Salida por consola: las cabeceras y líneas de cada pipeline como hoy (`load polls: 1 ámbito`,
`compute series: 1 ámbito`, `publish forecast: 1 ámbito · run <id>`, `publish analysis: ...`, `parties: ...`),
con los cuatro niveles de `verbose`; un resumen y un aviso al final con las líneas de todos los pipelines, los
fallos delante.

Docker y ECS: `RUN_JOB='update {"scopes":["es"],"n_sim":10000}'` y
`python deploy/ecs/job.py update '{"scopes":["es"],"n_sim":10000}'`. El lanzador no cambia: `update` nunca
corre `load params` ni toca las copias del backtest. La imagen de ECS ha de llevar el código nuevo (despliegue
normal), sin volver a registrar la definición de tarea.

### 7. Pasos de Luis

1. `ALTER TABLE elections.events ADD COLUMN loaded_at timestamp, ADD COLUMN computed_at timestamp;` en la RDS
   (y en cualquier base local que use).
2. Desplegar la imagen con el código nuevo.
3. Primer `update` (o `compute`) sin `overwrite`: computa una vez todos los ámbitos pedidos (todo está
   pendiente) y deja las marcas. Desde entonces el compute sólo corre con algo cargado.

### 8. Tests y documentación

- `tests/test_jobs_update.py`: la cadena entera con las fábricas falsas de `load`, `compute` y `publish`
  (`loader`, `computer`, `featured`, `pending`, `fs` y las de `publish_fixtures.py`); el compute saltado sin
  nada pendiente y corrido con `overwrite` (acción `rebulk` en la carga); por defecto, una carga o un compute
  fallidos paran la cadena sin publicar y el job sale con error; con `stop_on_error` falso, un ámbito con la
  carga fallida no se computa y se publica con lo que tenía; `all` con `skip_missing`; `years` con `overwrite`
  y `event_date` rechazados; sin sistema de ficheros falla antes de cargar; el resultado reagrupado; un solo
  aviso.
- `tests/test_pipelines_compute.py`: el salto (resultado y línea), `overwrite`, `Ratings` no salta, cabecera
  con `· overwrite`.
- `tests/test_jobs_compute.py`: `overwrite` por defecto falso con `pending` inyectado; `catalogues` y
  `ratings` no cambian.
- `tests/test_compute_unit.py`: `compute_scope` marca al final con `save` y no en seco ni tras un fallo
  (`mark` falso); `pending_events` sobre un frame de `events` con el modelo sustituido.
- `tests/test_loader_unit.py` (o el que toque): `save_polls` y los `save_results` marcan sólo si escribieron.
- `tests/test_ecs_job.py`: `run_job_value('update', ...)` pasa.
- `deploy/README.md`: sección «Job update» junto a las de `load`, `compute` y `publish` (uso, parámetros, el
  salto del compute, `overwrite`, `force`, pasos del `ALTER`), y `overwrite` en la sección de `compute`.
  `README.md` de la raíz si lista los jobs. Docstrings de `Update`, `Compute`, `compute_scope`, `pending_events`,
  `mark_loaded`, `mark_computed`.

## Hallazgos de la revisión del constructor (2026-11-03)

1. Los upserts de `events` pisarían las marcas (`format_data` repone como nulas las columnas ausentes del
   frame y el `ON CONFLICT` escribe todas las del frame). Resuelto en la decisión 1 arrastrando las marcas como
   se arrastra `featured`.
2. La publicación de un ámbito con el compute fallido. Resuelto por Luis: `update` para al primer fallo
   (decisión 6).
3. `event_date` significa cosas distintas en la carga y en la publicación. Resuelto: `update` lo rechaza.
4. Sugerencias adoptadas: `computed_at` con el instante de inicio del compute (decisión 3) y comprobar el
   sistema de ficheros al principio de `update` (decisión 6).

## Queda al constructor

- Nombres exactos de los helpers y del SQL; dónde vive `mark_loaded` si los tres cargadores lo comparten.
- Cómo inyectar `pending`, `mark` y el instante de inicio en los tests de `compute_scope` y del pipeline sin
  tocar la base.
- Comprobar que el upsert de `events` escribe `NULL` en una marca nula (`NaT` tras `format_data`) y no la
  cadena `NaT`.
- Las fixtures de `publish` que reutiliza el test del job `update`.
