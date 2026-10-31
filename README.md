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
ver `deploy/README.md`.

Cada acción de un job es un pipeline de MTpy en `mtpy/pipelines/<job>/` y los jobs son `SequenceJob`s. Con
`"stop_on_error": true` el job para en el primer ámbito que falla; por defecto sigue con todos. Al final envía un
único aviso con el resumen y el proceso sale con código 1 si algo falló. Una acción se puede lanzar sola con
`python job.py pipeline '{"select":"publish.backtest","chunks":"es,es-md"}'`; los detalles están en
`deploy/README.md`.

### `compute`: series de sondeos y ratings

Calcula en la base de datos las tablas del modelo de cada ámbito (la secuencia de
`notebooks/data-load/PollsCompute.ipynb`): es el paso previo a `publish`, que las lee.

| Parámetro | Valores | Por defecto |
|---|---|---|
| `what` | `compute` (pesos, eventos destacados, errores, desviaciones, deriva, efectos de casa y rebaño de cada ámbito) y/o `ratings` (los ratings de las encuestadoras de cada ámbito, con los sondeos de todos los ámbitos). Con los dos, primero la serie de todos los ámbitos y después los ratings de todos | `["compute", "ratings"]` |
| `scopes` | Lista de ámbitos (`es`, `es-md`, …) o `"all"`: `es` y todos los autonómicos | `["es"]` |
| `dry_run` | `true` calcula sin escribir nada; la serie se detiene tras las desviaciones (deriva, efectos y rebaño leen los errores guardados) | `false` |
| `verbose` | 0 cabecera de cada acción y resumen final; 1 además `[i/n] ámbito: empezando`, la línea de resultado y los avisos del modelo de cada ámbito; 2 además una línea por fase con sus segundos; 3 además la salida propia del `Computer` | 1 |

Un ámbito que falla no detiene a los demás: queda `failed` con su motivo en el resumen y el job termina con error.

```bash
python job.py compute '{"what":["ratings"],"scopes":"all"}'
python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
python job.py compute '{"what":["compute"],"scopes":["es-md"],"dry_run":true,"verbose":2}'
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

### `backtest`: el backtest del modelo

Ejecuta el backtest del modelo sobre las elecciones pasadas y escribe sus tablas en `backtest/results/` (`es`) o
`backtest/results/{ámbito}/` (los demás), que después convierte `publish backtest`. Los mismos parámetros que
`backtest/run_backtest.py` (hoy un envoltorio del job), en JSON; `"scopes": "all"` es `es` y todos los autonómicos y
los ámbitos están aislados como en `compute`. Protocolo y métricas en `backtest/README.md`; parámetros y salida en
`deploy/README.md` ("Backtest del modelo").

```bash
python job.py backtest
python job.py backtest '{"scopes":"all","horizons":[6,30]}'
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
