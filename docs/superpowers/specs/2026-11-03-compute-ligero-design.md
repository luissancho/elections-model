# Compute ligero: sólo lo que cambia con los sondeos del ciclo, y `average_errors` como acción aparte

Fecha: 2026-11-03. Estado: la propuesta general la aprobó Luis el 2026-11-03 («Construye eso que propones», tras la
pregunta «¿Hace falta correrlo entero cada vez que hay un sondeo nuevo?»); el detalle de las decisiones es mío, para
que lo corrija. Revisada por un constructor independiente el mismo día (sus preguntas, resueltas, al final).
Construida el 2026-11-03 (plan `docs/superpowers/plans/2026-11-03-compute-ligero.md`, cinco tareas, suite completa en
verde). Desviaciones de la construcción: ninguna en las decisiones; en el pipeline, el salto por «nada pendiente»
(`PipelineEnd`) se deja pasar explícitamente por el `try` que envuelve las dos consultas, como hace `collect` en la
base. Pendiente de Luis: desplegar la imagen, medir el `light` de `es` y lanzar `average_errors` cuando lo necesite
la tarea de la semana.

Origen: el `compute` de `es` tarda unos 45 minutos tras cargar unos pocos sondeos, con el comando
`python job.py compute '{"what":["compute"],"scopes":["es"]}'`. De las ocho fases de `compute_scope`, sólo dos miran
el ciclo en curso; la última, `average_errors`, cuesta unos 37 minutos en `es` y no la lee nadie en vivo.

## Intención

Que el compute que sigue a una carga de sondeos recalcule sólo lo que esos sondeos cambian, y que lo demás se
recalcule cuando cambia lo que lo alimenta: un resultado oficial nuevo, datos de ciclos pasados, los catálogos o el
método. Y sacar del compute diario la tabla `average_errors`, que sólo hace falta para las opciones del método 15.

Lo que mira cada fase, comprobado en el código el 2026-11-03:

| Fase | ¿Lee los sondeos del ciclo en curso? | Quién la usa |
| --- | --- | --- |
| `weights` | Sí: `weight_over` (solape) depende de los vecinos del ciclo. El cargador sin `overwrite` sólo pesa los sondeos nuevos entre sí (`Loader.compute_series`, `Computer` de un solo evento con `overwrite=False`): la ola anterior de un tracking conserva su peso hasta que la fase recorre el ciclo entero | El Forecaster vivo, que además descarta los sondeos con `computed` falso |
| `featured` | No por sí sola: `update_featured` exige resultado oficial y 10 sondeos útiles, así que un ciclo sin resultado nunca es destacado; cambia cuando entra un resultado o sondeos de un ciclo con resultado | Todo |
| `errors`, `deviations` | No: sondeo contra resultado oficial | Ratings y web |
| `drift` | No: ciclos destacados con resultado | El Simulator |
| `house_effects` | No: la tabla guarda los ciclos pasados con resultado; el efecto del ciclo en curso se ajusta en vivo con ese prior | Prior del Forecaster |
| `herding` | Sí: todos los ciclos con sondeos, el actual incluido | Sólo la web (`analysis`, perfil de casas); el Forecaster vivo lo mide al vuelo en `fit_dispersion` |
| `average_errors` | No: elecciones destacadas con resultado, un Forecaster por elección y horizonte (≈ 80 ajustes en `es`, cada uno releyendo su ciclo en la base) | Nadie en vivo: sólo `error_model='average'` y `drift_scale='auto'`, que no son los valores por defecto (método 15) |

## Objetivos

1. Dos modos del compute de la serie: `light` (pesos y rebaño) y `full` (lo de hoy sin `average_errors`: pesos,
   destacados, pesos otra vez, errores, desviaciones, deriva, efectos de casa y rebaño).
2. Que el pipeline elija el modo solo (`auto`, por defecto): `full` con `overwrite`, o cuando alguna de las elecciones
   pendientes del ámbito tiene resultado oficial; `light` cuando lo pendiente son sólo elecciones sin resultado (el
   ciclo en curso). Con `"mode": "light"` o `"mode": "full"` se fuerza.
3. `average_errors` como acción propia del job `compute` (`what: ["average_errors"]`), fuera de `compute` y fuera de
   `update`.
4. Que la línea de resumen de cada ámbito diga el modo y por qué.
5. Documentación: READMEs, docstrings, el método 15 (que dice que el job `compute` rellena la tabla) y la spec del
   job `update`.

## No objetivos

- Restringir el `Computer` del modo `light` a las elecciones pendientes: los pesos sobre toda la serie y el rebaño
  de todos los ciclos son más lentos de lo estrictamente necesario, pero dan exactamente lo mismo que el `full` en
  las fases que corren (los rellenos de `proc_sample` miran todos los ciclos) y las filas de rebaño de los ciclos
  pasados se reemplazan por idénticas. Queda como optimización posible, con medidas, si el `light` resultara lento.
- Marcas por fase o por tabla: la marca `computed_at` sigue siendo del ámbito y la escribe cualquier compute de la
  serie, `light` o `full`.
- Cambiar `load/run_load.py` (la carga por lotes de los autonómicos), que sigue llamando a `compute_scope` en modo
  `full` por defecto.
- Un paso `average_errors` en `update`: si algún día `error_model='average'` pasa a ser el valor por defecto, se
  añadirá a la secuencia tras `compute`.
- Mover a `light`/`full` el job `ratings` o los catálogos: no cambian.

## Lo que sería un fracaso

- Publicar un pronóstico con un sondeo cargado y no pesado (el Forecaster lo descartaría) o con el solape del ciclo
  sin revisar: el modo `light` ha de correr siempre la pasada de pesos sobre el ámbito entero (`overwrite=True`).
- Que un resultado oficial recién cargado se compute en `light` por defecto: sus errores, destacados, deriva y
  efectos de casa quedarían sin calcular. La regla `auto` ha de mirar si alguna elección pendiente tiene resultado.
- Que el primer compute tras el `ALTER` de las marcas (todo pendiente) salga `light`: como las elecciones con
  resultado también están pendientes, la regla da `full`.
- Que `average_errors` desaparezca sin forma de lanzarlo: la tarea de la semana del 3 al 8 de noviembre lo necesita
  (`what: ["average_errors"]`).
- Que la línea de resumen no diga el modo: Luis vería un compute de tres minutos sin saber si fue por diseño.

## Restricciones

- Identificadores en inglés (`mode`, `light`, `full`, `auto`, `mode_reason`, `result_dates`,
  `average_errors_scope`); cabeceras en castellano y estados y razones en inglés, como en el resto de los jobs.
- Las fábricas inyectables de los tests siguen funcionando (`computer`, `featured`, `catalogues`, `pending` y, nueva,
  `results`); ningún test toca la base ni `backtest/results`.
- `job.py`, `RUN_JOB` y `deploy/ecs/job.py` no cambian de interfaz (el lanzador no valida `what` ni `mode`).
- Sin cambios de esquema: todo se lee de `events`, `events_data` y las tablas que ya existen.
- Sin cambios en `mtpy/core/worker.py`.

## Alternativas consideradas

- **Dejar el compute como está y sólo sacar `average_errors`**: quita 37 de los 45 minutos de `es`, pero los otros
  8-13 minutos siguen recalculando errores, desviaciones, deriva y efectos de casa que no cambian con los sondeos
  del ciclo; y en los autonómicos (`scopes: "all"`) el ahorro proporcional es menor. El modo `light` es poco código
  encima de la misma secuencia.
- **Modo decidido por el job `update` (por lo que cargó esta ejecución)**: ciego a lo cargado desde el portátil o
  el cuaderno, la misma objeción que descartó la regla «computar si esta carga guardó algo» en la spec del job
  `update`. Con las marcas en `events` y los resultados en `events_data`, el pipeline lo sabe solo.
- **`pending_events` devolviendo también si cada elección tiene resultado** (un `DataFrame` o un `dict`): cambia la
  firma que ya usan el pipeline, los tests y la spec del `update`. Una función aparte, `result_dates(scope)`, que
  también sirve a `get_events` (hoy repite esa consulta), es más pequeña y deja `pending_events` como está.
- **Una lista de fases (`steps`) en el JSON en vez de `mode`**: más flexible, pero deja al usuario componer una
  secuencia inválida (deriva sin errores). Dos modos cerrados bastan; las fases sueltas siguen en el cuaderno.
- **Decidir el modo por `polls.computed`** (hay sondeos sin pesar): no distingue sondeos nuevos de un resultado
  nuevo, y el cargador ya los pone a verdadero.

## Decisiones

### 1. Librería: modos de `compute_scope` y `average_errors_scope` (`mtpy/lib/compute.py`)

```python
COMPUTE_STEPS = ('weights', 'featured', 'errors', 'deviations', 'drift', 'house_effects', 'herding')
MODES = {
    'light': ('weights', 'herding'),
    'full': COMPUTE_STEPS,
}
AVERAGE_ERRORS_STEP = 'average_errors'
RATINGS_STEP = 'ratings'
```

`compute_scope(scope, save=True, verbose=0, progress=None, computer=None, featured=None, mark=None, clock=None,
mode='full')`:

- `mode` es `'light'` o `'full'`; otro valor, `ValueError` (`compute: unknown mode "x" (expected light or full)`).
- `full`: la secuencia de hoy sin `average_errors`: pesos, destacados (con `save`), pesos, errores, desviaciones y,
  con `save` y una serie fresca, deriva, efectos de casa y rebaño; marca `computed_at` al terminar.
- `light`: una pasada de pesos sobre el ámbito entero (`compute_weights(save, overwrite=True)`, que copia a cada
  sondeo nuevo el destacado de su elección) y, con `save` y una serie fresca, el rebaño; marca `computed_at` al
  terminar. Sin `featured` ni segunda pasada: `update_featured` exige resultado oficial, y en `auto` el modo `light`
  sólo sale cuando ninguna elección pendiente lo tiene (pregunta 2 del constructor). En seco (`save=False`) sólo los
  pesos (el rebaño lee de la base los sondeos con los pesos que acaban de guardarse, como la deriva y los efectos de
  casa leen los errores; sin escritura no los tendría). Dos `Computer` en vez de tres.
- El resultado añade `'mode': mode` a `n_polls`, `n_events`, `featured` (`None` en `light`, como en seco) y
  `seconds` (`seconds` sólo lleva las fases que corrieron, más `total`).
- `progress` recibe las fases que corren, en orden (`weights`, `herding` en `light`).

`average_errors_scope(scope, save=True, verbose=0, progress=None, computer=None) -> dict`: un `Computer` fresco y
la fase `average_errors` (`comp.compute_average_errors(save=save)`), que reporta a `progress` con la clave
`AVERAGE_ERRORS_STEP`. Devuelve `n_rows` (filas de la tabla devuelta: elección × partido × horizonte), `n_events`
(elecciones distintas en ella) y `seconds` (`average_errors` y `total`). No marca `computed_at` (no es un compute de
lo pendiente) y en seco calcula sin guardar, como `rate_scope`. Su prior de efectos de casa es la tabla que haya en la
base: hay que lanzarlo después de un compute `full` del ámbito (documentado, no comprobado por código).

`resolve_mode(mode, overwrite, pending, results) -> tuple[str, str]`, la regla `auto`, pura y probada sin base:

| `mode` | `overwrite` | pendientes con resultado | Resultado |
| --- | --- | --- | --- |
| `light` / `full` | cualquiera | cualquiera | `(mode, 'requested')` |
| `auto` | `True` | cualquiera | `('full', 'overwrite')` |
| `auto` | `False` | alguna (`set(pending) & set(results)`) | `('full', 'results pending')` |
| `auto` | `False` | ninguna | `('light', 'polls pending')` |

`mode` fuera de `auto`, `light`, `full`: `ValueError` (`compute: unknown mode "x" (expected auto, light or full)`).
Con `overwrite` la lista de pendientes no se consulta (puede ser `None`); con `pending` vacío y sin `overwrite`
(caso al que el pipeline no llega, porque salta antes) devuelve `('light', 'polls pending')`. `check_mode(mode)`
hace sólo la validación, para que los jobs fallen antes de empezar.

`mode` no evita el salto por «nada pendiente»: ese interruptor sigue siendo sólo `overwrite` (decisión de Luis en la
spec del job `update`: un solo interruptor con el mismo sentido en la carga, el compute y `update`). Un `full`
forzado en un ámbito sin nada pendiente termina `skipped`; para recalcularlo sin recargar el ciclo,
`python job.py compute '{"what":["compute"],"scopes":["es"],"overwrite":true}'` (pregunta 1 del constructor).

`data.result_dates(scope) -> list[str]`: fechas (`YYYY-MM-DD`, ordenadas) de las elecciones del ámbito con resultado
oficial, la consulta que hoy hace `get_events` por dentro (`events_data` con `region_id = 0` y `votes IS NOT NULL`);
`get_events` pasa a usarla (`has_results = date.isin(pd.to_datetime(result_dates(scope)))`, fechas contra fechas),
así hay una sola definición de «tiene resultado». `compute.py` la reexporta junto a `pending_events`.

### 2. Pipeline `compute.Compute` (`mtpy/pipelines/compute/Compute.py`)

`prepare_task(chunk, position, overwrite=False, mode='auto', pending=None, results=None, **kwargs)`, sólo para
`ACTION == 'compute'` (`Ratings` y `AverageErrors` nunca saltan ni eligen modo):

1. Sin `overwrite`, `pending(chunk)` (el real, `compute.pending_events`, por defecto); vacío → `skipped` con
   `nothing loaded since the last compute`, como hoy y sea cual sea `mode` (el salto va antes que el modo).
2. `results(chunk)` (el real, `compute.result_dates`) sólo cuando hace falta: `mode == 'auto'`, sin `overwrite` y
   con pendientes.
3. `self.mode, self.mode_reason = compute.resolve_mode(mode, overwrite, pending, results)`.
4. Un fallo en cualquiera de las dos consultas (o un `mode` inválido) termina el ámbito como `failed` con su razón,
   como hoy con `pending`.

`compute_task` pasa `self.mode` a la unidad (`compute_scope(..., mode=self.mode)`) y el resultado lleva `mode` y
`mode_reason`. Cabecera: `compute series: n ámbitos[ · overwrite][ · light | · full][ · dry run]`, el modo sólo
cuando se fuerza. Línea de resumen:

```
es: series computed (light, polls pending: 4010 polls, 15 events) in 180 s
es: series computed (full, results pending: 4010 polls, 15 events, 7 featured) in 900 s
es: series computed (full, overwrite: 4010 polls, 15 events, 7 featured) in 900 s
es-md: series computed (light, requested: 310 polls, 11 events, dry run) in 20 s
es-md: series computed (full, requested: 310 polls, 11 events, dry run) in 60 s
```

El tramo de destacados sólo va cuando la fase corrió (`featured` no nulo); en seco, `dry run`.

Etiquetas de las fases a `verbose` 2: las de hoy; `errores del promedio` sigue en `STEPS` para la acción nueva.

### 3. Pipeline `compute.AverageErrors` (`mtpy/pipelines/compute/AverageErrors.py`)

Como `Ratings`: hereda de `Compute`, `ACTION = 'average_errors'`, `label = 'errores del promedio'`
(`ACTIONS['average_errors'] = 'errores del promedio'`), unidad `compute.average_errors_scope`. Cabecera
`compute errores del promedio: n ámbitos[ · dry run]`; línea
`es: average errors computed (59 rows, 14 events) in 2200 s` y, en seco, `(59 rows, 14 events, dry run)`. Nunca
salta; un fallo es `es: errores del promedio failed (...)`, con el patrón de `catálogos failed`.

### 4. Job `compute` (`mtpy/jobs/Compute.py`)

- `WHAT = ('catalogues', 'compute', 'ratings', 'average_errors')` y
  `sequence = {'compute': ['catalogues', 'compute', 'ratings', 'average_errors']}`: el orden fijo es catálogos,
  series, ratings y errores del promedio (los últimos leen la tabla de efectos de casa que escribe la serie).
- Parámetro nuevo `mode='auto'`, validado con `check_mode` antes de empezar; va a los pipelines con `overwrite`,
  `pending` y `results` (inyectable nuevo, para los tests).
- `what` por defecto sigue siendo `["compute", "ratings"]`.
- Docstring: la tabla de fases por modo, la regla `auto`, cuándo forzar `full` (tras cambiar código, `params.json` o
  la calidad previa, junto con `overwrite`), y `average_errors` como acción aparte con su coste.

### 5. Job `update` (`mtpy/jobs/Update.py`)

Sin cambios de secuencia. `mode` pasa a los pipelines como el resto de los kwargs y se valida al empezar con
`check_mode` (un valor inválido falla antes de la carga). Por defecto (`auto`) la actualización diaria hace `light`;
hace `full` con `overwrite` o cuando una elección con resultado quedó pendiente (tras cargar un resultado con `load`
o desde el cuaderno). `"mode": "full"` sirve para un día con sondeos nuevos en el que se quiere la serie entera; sin
nada pendiente el compute se salta igual (ver la decisión 1), y para rehacerlo todo sigue `overwrite`. Docstring y
README con el ejemplo `python job.py update '{"scopes":["es"],"mode":"full"}'` y esa salvedad.

### 6. Documentación

- `README.md`: sección `compute` (tabla de `what` con `average_errors`, fila `mode`, la regla `auto` y los
  ejemplos) y sección `update` (una frase sobre el modo).
- `deploy/README.md`: «Calcular las series y los ratings» (tabla de acciones con `average_errors` como acción,
  fases por modo, ejemplos, parámetros con `mode`, cabeceras y líneas de resumen nuevas, el párrafo de
  `average_errors` reescrito como acción), «Actualizar» (punto 2 con el modo, parámetros, ejemplo de salida) y
  «Duraciones medidas» (la estimación del `light` de `es`, pendiente de medir por Luis).
- `docs/analisis-2026-09/metodo-15-calibracion.md`: donde dice que el job `compute` rellena `average_errors`, pasa a
  decir `python job.py compute '{"what":["average_errors"],"scopes":...}'`.
- `docs/superpowers/specs/2026-11-03-job-update-design.md`: una enmienda al principio que remite a esta spec
  (modo `light`/`full` y `average_errors` fuera de `compute`).
- Docstrings de `compute_scope`, `Compute`, `AverageErrors`, `Update`, módulo `compute.py`, `run_load.compute_scope`
  (ya no menciona `average_errors`) y `mtpy/jobs/Backtest.py` (donde dice que la tabla la rellena el job `compute`,
  ahora su acción `average_errors`).

### 7. Tests

- `tests/test_compute_unit.py`: `compute_scope` en `light` (fases, `progress`, marca, en seco, dos `Computer`) y en
  `full` (sin `average_errors`), `mode` inválido, `average_errors_scope` (fases, cifras, no marca, en seco calcula
  sin guardar), `resolve_mode` (las cuatro filas de la tabla, `pending` vacío y los errores), `check_mode`.
- `tests/test_data_marks.py`: `result_dates` (consulta y fechas) y `get_events` usándola con fechas contra fechas.
- `tests/test_pipelines_compute.py`: el modo por defecto con pendientes sin resultado es `light` y con resultado
  `full`; `overwrite` → `full`; `mode` forzado y su cabecera; `mode` forzado sin nada pendiente → `skipped`;
  `results` sólo se consulta cuando hace falta; un `results` que falla es `failed`; las líneas de resumen;
  `AverageErrors` (cabecera, línea, en seco, nunca salta).
- `tests/test_jobs_compute.py`: `mode` inválido falla antes; `average_errors` corre el último y sólo con `what`;
  las líneas y los `inits` del `Computer` por modo.
- `tests/test_jobs_update.py`: la actualización diaria (pendiente sin resultado) hace `light`; con un resultado
  pendiente, `full`; `mode` inválido falla antes de la carga.
- Los helpers de los tests de pipelines, jobs y `update` reciben `results=lambda scope: ['2023-07-23']` junto al
  `pending` de hoy (`['2026-11-29']`), de modo que el caso por defecto de los tests es el `light` de la rutina diaria.

### 8. Pasos de Luis

1. Nada que alterar en la base.
2. Desplegar la imagen con el código nuevo.
3. La tarea de la semana (rellenar `average_errors`) pasa a ser
   `python deploy/ecs/job.py compute '{"what":["average_errors"],"scopes":"all"}'`, tras un compute `full` de cada
   ámbito (el primero tras el `ALTER` ya lo es).
4. Medir el `light` de `es` y anotarlo en «Duraciones medidas».

## Decisiones tras la revisión del constructor

Dos preguntas, resueltas como decisiones mías (Luis puede corregirlas):

1. **`mode: "full"` forzado sin `overwrite` y sin nada pendiente: `skipped`.** El salto por «nada pendiente» va
   antes que el modo y sólo lo evita `overwrite`, el único interruptor de rehacerlo todo que Luis quiso. Un `update`
   con `mode: full` un día sin sondeos nuevos salta el compute y publica sobre la serie que hay; para recalcular sin
   recargar el ciclo está `compute` con `overwrite`. Documentado en las decisiones 1 y 5 y en los READMEs.
2. **`light` sin `featured` ni segunda pasada de pesos.** `update_featured` exige resultado oficial, y en `auto` el
   `light` sólo sale cuando ninguna elección pendiente lo tiene, así que esas dos fases no podían cambiar nada en
   el caso que importa. `light` = pesos (una pasada, ámbito entero) y rebaño: dos `Computer`. La tabla de fases de
   la intención corregida en consecuencia.

Decisiones menores del constructor, aceptadas: `get_events` compara fechas con `pd.to_datetime(result_dates(scope))`
(con cadenas, pandas 3 daría todo falso y `update_featured` desmarcaría todas las elecciones); la variable local
`results` de `Compute.run` se renombra para no sombrear el inyectable; `resolve_mode` con `pending` vacío y sin
`overwrite` devuelve `('light', 'polls pending')`; `n_rows` y `n_events` de `average_errors_scope` son las filas del
`DataFrame` devuelto y sus `event_date` distintos; el docstring de `mtpy/jobs/Backtest.py` nombra la acción nueva.
