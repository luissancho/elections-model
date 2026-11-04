# MT Elections Model

Modelo de promedio de encuestas y simulación de escaños de las elecciones españolas (generales y autonómicas), con la
web que lo publica. El código vive en `mtpy/`, los datos de entrada versionados en `data/` (ver `data/README.md`) y la
documentación del método y de las fases en `docs/`.

## Puesta en marcha

```bash
pip install -r requirements.txt
```

La configuración (base de datos PostgreSQL, sistema de ficheros, S3) se lee de `config.json` y de las variables de
entorno de `.env`.

## Jobs

Los trabajos por lotes se lanzan con `job.py`, con el nombre del job y sus parámetros en JSON:

```bash
python job.py <job> '{"param": "valor", ...}'
```

En Docker, con la variable `RUN_JOB` (el nombre del job y después el JSON):
`docker run --rm --env-file deploy/elections.env -e RUN_JOB='compute {"what":["ratings"],"scopes":"all"}' elections-model:<tag>`;
ver `deploy/README.md`. Como tarea de ECS Fargate se lanzan desde el portátil con
`python deploy/ecs/job.py <job> '{...}' [--follow]` (ver "Jobs desde ECS" en `deploy/README.md`).

Cada acción de un job es un pipeline de MTpy en `mtpy/pipelines/<job>/` y los jobs son `SequenceJob`s. Con
`"stop_on_error": true` el job para en el primer ámbito que falla; por defecto sigue con todos, salvo `update`, que
para al primer fallo por defecto (`"stop_on_error": false` para seguir). Al final envía un
único aviso con el resumen y el proceso sale con código 1 si algo falló. Una acción se puede lanzar sola con
`python job.py pipeline '{"select":"publish.backtest","chunks":"es,es-md"}'`; los detalles están en
`deploy/README.md`.

### `update`: carga, series y publicación de una vez

Encadena `load.polls`, `compute.compute` y `publish.forecast`, `analysis` y `parties` de los ámbitos pedidos en una
sola tarea. La serie sólo se computa en los ámbitos con algo cargado desde su último compute (`overwrite` recarga el
ciclo y lo computa todo), a diario en modo `light` (pesos y rebaño) y en `full` con `overwrite` o cuando una elección
con resultado quedó pendiente (`"mode": "full"` fuerza la serie entera cuando hay algo pendiente), y para al primer
fallo. Parámetros y pasos en "Actualizar" de `deploy/README.md`.

```bash
python job.py update '{"scopes":["es"],"n_sim":10000}'
python job.py update '{"scopes":["es"],"overwrite":true}'
python job.py update '{"scopes":["es"],"mode":"full"}'
```

### `compute`: series de sondeos y ratings

Calcula en la base de datos las tablas del modelo de cada ámbito (la secuencia de
`notebooks/data-load/PollsCompute.ipynb`): es el paso previo a `publish`, que las lee.

| Parámetro | Valores | Por defecto |
|---|---|---|
| `what` | `compute` (la serie de cada ámbito, en modo `light`, pesos y rebaño, o `full`, pesos, eventos destacados, errores, desviaciones, deriva, efectos de casa y rebaño), `ratings` (los ratings de las encuestadoras de cada ámbito, con los sondeos de todos los ámbitos) y/o `average_errors` (los errores del promedio en las elecciones pasadas, la tabla que sólo leen las opciones del método 15; unos 37 minutos en `es`, tras un compute `full`). Siempre en este orden: la serie de todos los ámbitos, los ratings de todos y los errores del promedio de todos | `["compute", "ratings"]` |
| `scopes` | Lista de ámbitos (`es`, `es-md`, …) o `"all"`: `es` y todos los autonómicos | `["es"]` |
| `overwrite` | `true` computa la serie de todos los ámbitos aunque no tengan nada cargado desde su último compute (tras un cambio de código o de parámetros), en modo `full` | `false` |
| `mode` | `auto` elige el modo de la serie según lo pendiente: `full` con `overwrite` o cuando una elección pendiente tiene resultado oficial; `light` cuando lo pendiente son sólo sondeos del ciclo en curso (la rutina diaria). `light` o `full` fuerzan las fases, pero no evitan el salto de un ámbito sin nada pendiente; un `light` con una elección pendiente con resultado se rechaza (`failed`), porque dejaría ese resultado sin computar y el ámbito marcado | `auto` |
| `dry_run` | `true` calcula sin escribir nada; la serie `full` se detiene tras las desviaciones y la `light` tras los pesos (deriva, efectos y rebaño leen lo guardado); `ratings` y `average_errors` calculan sin guardar | `false` |
| `verbose` | 0 cabecera de cada acción y resumen final; 1 además `[i/n] ámbito: empezando`, la línea de resultado y los avisos del modelo de cada ámbito; 2 además una línea por fase con sus segundos; 3 además la salida propia del `Computer` | 1 |

Un ámbito que falla no detiene a los demás: queda `failed` con su motivo en el resumen y el job termina con error.
Sin `overwrite`, la serie sólo corre en los ámbitos con algo cargado desde su último compute (marcas `loaded_at` y
`computed_at` de `events`); el resto termina `skipped`, sea cual sea `mode`. Tras un cambio de código o de
parámetros, `"overwrite": true`. La línea de cada ámbito dice el modo y su razón:
`es: series computed (light, polls pending: 4010 polls, 15 events) in 180 s`.

```bash
python job.py compute '{"what":["ratings"],"scopes":"all"}'
python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
python job.py compute '{"what":["compute"],"scopes":["es-md"],"dry_run":true,"verbose":2}'
python job.py compute '{"what":["compute"],"scopes":"all","overwrite":true}'
python job.py compute '{"what":["compute"],"scopes":["es"],"mode":"full"}'
python job.py compute '{"what":["average_errors"],"scopes":"all"}'
```

### `publish`: el paquete de la web

Publica los resultados del modelo en el paquete que leen la API y el sitio (`site/v1`): `forecast` (el pronóstico de
cada ámbito, un run inmutable), `analysis` (el análisis de encuestadoras), `event` (las elecciones pasadas),
`backtest`, `parties` (el catálogo de partidos de la base, `parties.json` junto al manifest; `forecast` y `analysis`
lo reescriben también), y las acciones de punteros `manifest`, `point` y `unpublish`. Las acciones se ejecutan
siempre en ese orden, sea cual sea el de la lista. Mismos niveles de `verbose` que `compute`; `dry_run` escribe bajo `site-dry/v1`.
El detalle de cada acción y del paquete está en `docs/web/contrato.md` y en la docstring de `mtpy/jobs/Publish.py`.

```bash
python job.py publish '{"what":["forecast","analysis"],"scopes":["es"]}'
python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"all"}'
python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
```

Con `"rating_check": true`, `analysis` escribe además en `es` la parte `rating-check`: la validación del rating de las
encuestadoras sobre las elecciones de todos los ámbitos (¿predice el rating que tenía una casa antes de una elección su
desviación en ella?, ¿mejora el rating un promedio simple?). Es cara y sólo cambia cuando cambian los ratings, así que
se pide aparte: sin el parámetro (y en el job `update`, que nunca lo pone) la parte publicada se queda como estaba, y
en otro ámbito se ignora. Tras una elección, cuando ya están cargados sus resultados, se recalculan los ratings de
todos los ámbitos y se republica la validación:

```bash
python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
python job.py publish '{"what":["analysis"],"scopes":["es"],"rating_check":true}'
```

### `backtest`: el backtest del modelo

Ejecuta el backtest del modelo sobre las elecciones pasadas y escribe sus tablas en `backtest/results/` (`es`) o
`backtest/results/{ámbito}/` (los demás), que después convierte `publish backtest`. Los mismos parámetros que
`backtest/run_backtest.py` (hoy un envoltorio del job), en JSON; `"scopes": "all"` es `es` y todos los autonómicos y
los ámbitos están aislados como en `compute`. Protocolo y métricas en `backtest/README.md`; parámetros y salida en
`deploy/README.md` ("Backtest del modelo").

```bash
python job.py backtest
python job.py backtest '{"scopes":"all","horizons":[6,30]}'
python job.py backtest '{"scopes":["es"],"config":"backtest/tuning/best.json","out":"backtest/results_tuned"}'
```

`config` corre el backtest con una configuración del promedio, como el `best.json` de `tune`, y conviene escribirlo
en otra carpeta (`out`). `flat_ratings` (`--flat-ratings` en `backtest/run_backtest.py`) corre el modelo con el rating plano (todos los sondeos con peso por rating 1, fase C); ver `backtest/README.md`.

### `tune`: afinar el promedio con los sondeos futuros

Busca la configuración del promedio (ancho de banda, prolongación, ventana de sobrepublicación, contracciones de los
efectos de casa y del rebaño) que mejor predice los sondeos publicados después de cada corte, y escribe sus tablas en
`backtest/tuning/` (`es`) o `backtest/tuning/{ámbito}/`. El destino es el sistema de ficheros de la app, o la copia del
repositorio con `backtest/run_tuning.py`, su envoltorio. Lee la base y no escribe en ella.

Modos:

- `smoothing` (fase 1);
- `shrinkage` (fase 2, también por partes con `cycles`);
- `merge` (une las partes);
- `check` (la afinada frente a la vigente en los ciclos reservados);
- `evaluate` (configuraciones dadas).

Las acciones son `run`, `push` y `pull`, y `workers` reparte los cortes entre procesos. Los valores por defecto del
modelo sólo cambian si se cumple el criterio A8, y a 03-11-2026 no se cumple. Protocolo y tablas en `backtest/README.md`
("Sondeos futuros"); método y decisión en `docs/analisis-2026-09/metodo-14-sondeos-futuros.md`; parámetros en
`deploy/README.md` ("Afinar el promedio").

```bash
python job.py tune '{"scopes":["es"],"mode":"smoothing","workers":8}'
python job.py tune '{"scopes":["es"],"mode":"check","workers":8}'
python backtest/run_tuning.py --mode shrinkage --workers 8
```

### Carga de los ámbitos autonómicos

`load/run_load.py` descarga de Wikipedia los sondeos y resultados de las elecciones autonómicas y los carga en la
base; sus pasos `compute` y `ratings` son los del job `compute`, al que conviene llamar directamente para recalcular
(`run_load.py` no admite `es`).

```bash
python load/run_load.py --scopes all --what urls polls results --save
```

## Tests

```bash
pytest -q
```

Los tests unitarios no necesitan base de datos; los de `tests/integration/` se saltan sin ella.
