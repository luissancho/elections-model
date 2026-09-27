# Módulo: Infraestructura, empaquetado y código muerto

Repositorio: `/Users/luiss/HD/Proyectos/Code/elections-model` (rama `master`, 3 commits). Análisis en modo solo lectura sobre el código, el entorno `/Users/luiss/.pyenv/versions/3.11.9` y la base de datos PostgreSQL local (solo `SELECT`).

## 1. Visión general

El modelo electoral (Computer / Forecaster / Simulator) no es un paquete independiente: vive dentro de `mtpy`, un framework personal genérico ("ManyThings") que incluye un contenedor de dependencias (`App`), configuración por JSON + variables de entorno, logging, sistema de ficheros abstracto (local o S3), una capa de acceso a datos multi-motor (PostgreSQL, MySQL, Redshift, BigQuery, DynamoDB, Mongo, Salesforce), un ORM mínimo (`Model`), un runtime de API ASGI, un worker de colas SQS, jobs/pipelines ETL, integraciones SaaS (Stripe, Pipedrive, CleverTap, Facebook Ads, SendGrid, Simplecast, Google, AWS...) y utilidades de estadística, NLP, ML y visualización.

De todo ese framework (≈27.000 líneas en `mtpy/`), el modelo electoral sólo necesita una fracción pequeña. El grafo de importaciones real, construido con `grep`/`ast` sobre `mtpy/lib/*`, `mtpy/models/elections.py` y los notebooks raíz/`data-load`/`lab`, es el siguiente:

| Consumidor | Importa de `mtpy.core` |
|---|---|
| `mtpy/lib/computer.py:12-19` | `app.Core`; `utils.helpers.{apply_agg_func, format_number}`; `utils.dates.ts_from_delta`; `utils.stat.{LeastSquaresEstimator, LocalKernelEstimator, Stat}`; `utils.dataviz` (13 funciones) |
| `mtpy/lib/forecaster.py:13-21` | `app.Core`; `utils.dates.{add_delta, ts_from_delta}`; `utils.helpers.format_number`; `utils.stat.{LocalKernelEstimator, Kernel}`; `utils.dataviz` (13 funciones) |
| `mtpy/lib/simulator.py:10-13` | `app.Core`; `utils.stat.Stat`; `utils.helpers.unset_categorical`; `utils.dataviz.plot_kde_1d` |
| `mtpy/lib/loader.py:12-18` | `app.Core`; `models.elections` (8 modelos); `utils.helpers.{array_shift, format_number, is_number}` |
| `mtpy/lib/data.py:6-11` | `app.App`; `worker.Model` (re-export de `core.data.Model`); `models.elections` (8 modelos) |
| `mtpy/lib/utils.py:6` | `utils.dataviz.{get_color, palette}` |
| `mtpy/models/elections.py:1` | `core.data.Model` |
| Notebooks | `mtpy.mtpy.run`, `core.app.App`, `core.utils.dataviz` (≈20 funciones), `core.utils.dates.{ts_resample, ts_fitreg}`, `core.utils.helpers.{format_number, is_number}`, `core.utils.stat.{Kernel, LocalKernelEstimator, Stat}` |

Y las llamadas efectivas al runtime desde `mtpy/lib` se limitan a `self.app.fs.{read, read_csv, write_csv, exists}`, `self.app.fspath` y a los métodos de `Model`: `get_results` (17 usos), `get_agg` (2), `get_var` (1), `columns`, `format_data`, `execute` y `upsert` (sólo en las rutas de guardado `save_model_data`, `save_ratings_data` y en los loaders).

### Prueba de humo (entorno real)

`import mtpy.lib.simulator` funciona en 2,1 s pero carga 211 módulos de terceros/stdlib, incluidos `xgboost`, `mlxtend`, `sklearn`, `statsmodels`, `seaborn`, `dask`, `sqlalchemy`, `sshtunnel`, `PIL`, `joblib`, `husl`, `adjustText`, `IPython`. La causa es la cadena `computer.py → core/utils/dataviz.py:34-43 (from .learning import ...) → core/utils/learning.py:5-13 (mlxtend, sklearn, statsmodels, xgboost)`. Ninguna de las funciones de `learning` es alcanzable desde el modelo: dentro de `dataviz` sólo las usan `plot_collinearity_grid` (l. 3848), `plot_feature_rank` (l. 3941) y `print_feature_split` (l. 4550), que nadie del modelo llama.

`mtpy.run()` conecta con PostgreSQL correctamente (`adapter=PostgreSQL`, `fs=FileSystem`, `fspath=<repo>/files`, `verbose=1`).

## 2. Arranque: `mtpy.run()` paso a paso (`mtpy/mtpy.py:19-93`)

1. Singleton: si `App` ya existe se devuelve (`mtpy.py:20-21`; `App.get_()`/`has_()` en `core/app.py:10-19`).
2. Rutas: `abspath = dirname(mtpy/mtpy.py) + '/..'` (raíz del repo). `shpath = abspath + '/../../shared'` si existe ese directorio hermano, si no `abspath` (`mtpy.py:29-33`). `fspath = fspath or shpath + '/files'` (`mtpy.py:35`). Es decir, el modelo espera `files/` en la raíz del repo (o en un directorio compartido `../../shared`).
3. Carga `.env` desde `shpath` con `python-dotenv` si existe (`mtpy.py:39-40`).
4. Lee `config.json` y sustituye cada `${VAR}` por `os.getenv(VAR, '')` mediante regex (`mtpy.py:42-47`); una variable ausente se convierte silenciosamente en cadena vacía. Construye `Config` (objeto recursivo por atributos; `core/app.py:62-134`).
5. Adaptador de BD: `getattr(dal, config.db.adapter)` (`mtpy.py:51`). `mtpy/core/dal/__init__.py:5-12` implementa un `__getattr__` de paquete que importa el submódulo por nombre (lazy). Para `PostgreSQL` se instancia `PostgreSQL(params, database)` (`dal/PostgreSQL.py:13`), subclase de `SQLAdapter` (`core/data.py:343`). La conexión real (`create_engine('postgresql+psycopg2://...', pool_pre_ping=True)`, opcional túnel SSH con `sshtunnel`) se crea perezosamente en `_build_client` (`core/data.py:435-484`).
6. Bases secundarias `config.dbs` (vacío en `config.json:13`).
7. `verbose` (`mtpy.py:69-71`), `Log(shpath + '/log', verbose)` (`mtpy.py:73`) y, si `cwl.group` no está vacío, un `CloudWatchLogHandler` (`mtpy.py:74-76`).
8. Sistema de ficheros: `S3(...)` si `s3.bucket` no está vacío, si no `FileSystem(fspath)` (`mtpy.py:80-83`).
9. `Pushover` (notificaciones push) siempre instanciado (`mtpy.py:87`).
10. `np.random.seed(config.app.seed)` con `seed = 42` (`config.json:16`, `mtpy.py:91`): siembra el RNG global legacy de NumPy (comprobado: primer `np.random.uniform()` tras `run()` = 0.37454, el valor canónico de semilla 42).

Efecto colateral de importación: `mtpy/mtpy.py:16` ejecuta `warnings.filterwarnings('ignore')` a nivel de módulo, silenciando todas las advertencias de Python (incluidas `FutureWarning`/`DeprecationWarning` de pandas y NumPy) para cualquier proceso que importe `mtpy.mtpy`.

## 3. Núcleo mínimo del framework

### `App` (`core/app.py:6-59`)
Contenedor de dependencias tipo service-locator. `set(key, value)` guarda el valor (o lo llama si es callable, o instancia una clase por nombre). `__getattr__` devuelve `self.deps[key]` y, si no existe, **devuelve `None` sin error** (`app.py:25-27`; verificado: `App.get_().some_unknown_dependency is None`). `class_exists`/`get_class` con nombres con punto hacen `hasattr('modulo', 'Clase')` sobre una **cadena**, no sobre el módulo (`app.py:46-59`), por lo que nunca funcionan (`App.class_exists('os.path') → False`); no se usan en el modelo.

### `Config` (`core/app.py:62-134`)
Diccionario recursivo expuesto como atributos; `to_dict()`, `merge()`, `offset_is_empty()` (no recursivo: `Config('{"db":{"adapter":""}}').offset_is_empty('db') → False`).

### `Core` (`core/app.py:137-140`)
Clase base que sólo guarda `self.app = App.get_()`. Es la superclase de `Computer`, `Forecaster`, `Simulator`, los loaders, `Model` y los adaptadores.

### `Log` (`core/app.py:143-202`)
Crea tres loggers estándar con nombres globales `'debug'`, `'info'`, `'error'` (`app.py:163`), con `StreamHandler` a stdout si `verbose > 0` y `FileHandler` a `<path>/info.log` y `<path>/error.log` (`app.py:172-176`). `FileHandler` abre el fichero al construir el objeto: si el directorio no existe, `mtpy.run()` falla con `FileNotFoundError` (verificado con un directorio inexistente). `log/` está en `.gitignore:3`, luego un clon limpio no puede arrancar sin crear `log/` a mano (el `Dockerfile:57` lo hace con `mkdir -p files log`).

### `FileSystem` (`core/io.py:25-854`)
Envoltorio sobre `os`/`pandas`: `get_path(name)` devuelve `name` tal cual si empieza por `/`, si no `'{fspath}/{name}'` (`io.py:50-59`). Los notebooks pasan `path = '.'` a `Computer/Forecaster/Simulator`, con lo que `self.app.fs.read(f'{self.path}/params.json')` resuelve a `files/./params.json`, y las figuras a `files/./img/...`; `fc/*.csv` idem (`forecaster.py:452-471`). Métodos usados por el modelo: `read`, `read_csv`, `write_csv`, `exists` (firma en `io.py:262-268`, `634-640`, `668-676`). El resto (`read_url`, `save_image`, `save_object` con `joblib`, `read_excel`/`write_excel`, `read_bytes`...) no se usa desde el modelo salvo `pd.ExcelFile` directo en `loader.py:70`.

### `Model` (`core/data.py:1930-3050`) y `SQLAdapter`/`PostgreSQL`
ORM declarativo mínimo: cada subclase (`mtpy/models/elections.py`) declara `table`, `key`, `sort`, `autokey` y `meta` = `{columna: tipo | [tipo, tamaño, nullable]}` con tipos abreviados `'int'`, `'num'`, `'str'`, `'cat'`, `'bin'`, `'dtd'`, `'dts'`, `'obj'` (parseados por `helpers.parse_meta`, `data.py:2009-2024`). En `__init__` resuelve el adaptador (`app.db` por defecto, `data.py:1999-2010`). `get_results(query, formatted)` (`data.py:2158-2190`) rellena `columns`/`sort` por defecto, delega en `SQLAdapter.build_select_query` y, si `formatted=True`, aplica `format_data` (casting por `meta` y opcionalmente `relations`). `upsert(df)` formatea y delega en `SQLAdapter.upsert(table, df, key=...)` (`data.py:2409-2443`). `PostgreSQL` (`dal/PostgreSQL.py`) añade `COPY` para carga/descarga por *stage* (`execute_load`, `build_load_query`), metadatos vía `information_schema` y gestión de secuencias. Todo el motor de *staging* (dask, gzip, `stage_dir`) no lo usa el modelo.

### `mtpy/lib/data.py`
Capa de acceso del modelo (getters `get_event_*`, `get_poll_series`, `get_pollsters`, `get_ratings`, `get_event_params`) y dos rutas de escritura: `save_model_data` (`UPDATE ... SET col = NULL` + `upsert`, `data.py:413-420`) y `save_ratings_data` (`DELETE` + `upsert` + cast a `int` del rating, `data.py:442-458`). Las sentencias se construyen por interpolación de cadenas con las fechas/ámbitos del propio DataFrame.

## 4. Inventario keep / extract / drop para un paquete científico independiente

| Componente | Líneas | Decisión | Motivo |
|---|---|---|---|
| `mtpy/lib/{computer,forecaster,simulator,data,utils}.py` | ≈5.150 | **KEEP** (núcleo) | El modelo. |
| `mtpy/lib/loader.py` | 752 | **KEEP** (extra `ingest`) | Carga Wikipedia/Infoelectoral; requiere `lxml`, `requests`, `openpyxl`. |
| `mtpy/models/elections.py` | 274 | **KEEP** | Esquema de tablas del modelo. |
| `mtpy/core/utils/stat.py` | 1.144 | **KEEP** (100 % alcanzable) | Librería estadística propia. Depende de `dates` y `helpers`. |
| `mtpy/core/app.py` (`App`, `Config`, `Core`, `Log`) | 202 | **EXTRACT/simplificar** | Sustituible por un `settings` (pydantic/dataclass) + `logging` estándar. |
| `mtpy/core/io.py` `FileSystem` | 854 | **EXTRACT** subset (`get_path`, `read`, `read_csv`, `write_csv`, `exists`, `read_json`/`write_json`) | O reemplazar por `pathlib`. |
| `mtpy/core/data.py` (`Adapter`, `DBAdapter`, `SQLAdapter`, `Model`) | 3.050 | **EXTRACT** subset | Sólo `Model.get_results/get_agg/get_var/format_data/upsert/execute` + `SQLAdapter.build_select_query/get_results/upsert`. Sin staging/dask/ssh. |
| `mtpy/core/dal/PostgreSQL.py` | 320 | **EXTRACT** (opcional) | Alternativa: SQLite/Parquet para reproducibilidad. |
| `mtpy/core/utils/helpers.py` | 1.825 | **EXTRACT** 38/61 funciones (≈1.150 líneas alcanzables por nombre) | Predicados `is_*`, `format_*`, `parse_meta`, `apply_agg_func`, `filter_smooth`, `get_histogram`... Quitar `sklearn` (`DictVectorizer`, `StandardScaler`, `IterativeImputer`, l. 16-24) y `statsmodels` si no se usan en el subconjunto. |
| `mtpy/core/utils/dates.py` | 1.413 | **EXTRACT** 10/22 (`Freq`, `Delta`, `Format`, `add_delta`, `ts_diff`, `ts_fitreg`, `ts_from_delta`, `ts_range`, `ts_resample`, `ts_to_delta`) | `ts_fitreg` importa `stat` de forma diferida (`dates.py:1072`). |
| `mtpy/core/utils/dataviz.py` | 5.137 | **EXTRACT** 33/87 (≈2.230 líneas) | Quitar import de `learning` (l. 34-43) y las 3 funciones que lo usan. Depende de `seaborn`, `husl`, `adjustText`, `IPython.display`, `PIL`. |
| `mtpy/core/utils/strings.py` | 958 | **DROP** (5 nombres coinciden sólo por homonimia: `split`, `join`, `process`...) | `contractions`, `emoji`, `ftfy`, `sacremoses` innecesarios. |
| `mtpy/core/utils/learning.py`, `nlp.py` | 925 | **DROP** | 0 funciones alcanzables; requieren `xgboost`, `mlxtend`, `torch`, `spacy`, `stanza`, `transformers`, `pke`, `enchant`. |
| `mtpy/core/dal/{BigQuery,DynamoDB,Mongo,MySQL,Redshift,Salesforce}.py` | 845 | **DROP** | Otros motores. |
| `mtpy/core/services/*` (aws, s3, stripe, pipedrive, clevertap, facebook, google, sendgrid, simplecast, smtp, apilayer, pushover) | 2.750 | **DROP** | `mtpy.run()` los importa (`mtpy.py:11-13`) pero el modelo no los usa; `S3`/`CloudWatch` sólo si `S3_BUCKET`/`CWL_GROUP` están definidos. |
| `mtpy/core/{api,worker}.py`, `mtpy/controllers`, `mtpy/jobs`, `mtpy/pipelines`, `mtpy/learners`, `mtpy/models/sources` | 1.400 | **DROP** | Runtime API/worker/ETL. `pipelines/events/*` está roto (ver hallazgos). |
| `api.py`, `worker.py`, `job.py`, `pipeline.py`, `Dockerfile`, `deploy/docker/*` (nginx, supervisord, supercronic, crontab) | — | **DROP** | Despliegue de servicios que el modelo no necesita; `crontab` invoca un job `rep` inexistente. |
| `notebooks/{PollstersRatings,PollstersEvent,PollsErrors,PollsForecast,PollsSimulations}.ipynb`, `data-load/*`, `lab/*` | — | **KEEP** (como `examples/` o `docs/`) | Convertir a `jupytext`/`nbstripout`. |
| `notebooks/events/es-2019*/es-2023*` | 14 MB | Archivar | Copias antiguas. |
| `files/` (params.json, es-provinces.csv, es-regions.csv, wikipedia/*.json, infoelectoral/es/*, fc/, img/) | 37 MB | **Versionar los inputs** (`params.json`, CSV, JSON de mapas: ≈60 KB) como `data/`; `img/` y `fc/` son salidas | Hoy todo está ignorado (`.gitignore:2`), lo que hace irreproducible el repo. |

## 5. Dependencias y deriva de versiones

`requirements.txt` (42 paquetes) es el del framework, no el del modelo. Comparación pin vs. entorno pyenv 3.11.9 y uso real:

- Deriva: `numpy 1.25.2 → 2.4.4`, `scipy 1.9.3 → 1.17.0`, `scikit-learn 1.2.0 → 1.8.0`, `statsmodels 0.13.5 → 0.14.6`, `matplotlib 3.8.0 → 3.10.8`, `boto3 1.22 → 1.42`, `requests 2.31 → 2.33`, `typing_extensions 4.6 → 4.15`, `joblib 1.2 → 1.5`, `gunicorn 20.1 → 25.3`, `dnspython 1.16 → 2.8`. El código funciona con NumPy 2.x (no hay `np.NaN`/`np.float`/`np.int`), pero nadie lo ha fijado.
- **Faltan** en `requirements.txt` y son necesarios para importar el modelo hoy: `xgboost` (vía `learning.py:13`), `lxml` (`loader.py:2`), `python-dateutil` (implícito por pandas), `openpyxl` (para `pd.ExcelFile` en `loader.py:70`), `psycopg2-binary` sí está.
- **No usados** por el modelo (candidatos a eliminar): `boto3`, `s3fs`, `dnspython`, `jinja2`, `oauth2client`, `pdf2image`, `proto-plus`, `protobuf`, `PyJWT`, `pyzbar`, `unidecode`, `xlsxwriter`, `cmake`, `cython`, `gunicorn`, `uvicorn`, `nbconvert`, `mlxtend`, `contractions`, `emoji`, `ftfy`, `sacremoses`, `dask`, `sshtunnel`, `ipython` (sólo `IPython.display` en dataviz/helpers).
- Dependencias mínimas del modelo tras la extracción: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `adjustText`, `husl`, `Pillow`, `typing_extensions` (o Python ≥ 3.11 y `typing.Self`), `python-dotenv`; opcionales: `sqlalchemy` + `psycopg2-binary` (`db`), `lxml` + `requests` + `openpyxl` (`ingest`), `statsmodels` (`DescrStatsW` en `dataviz.py:20` y `helpers.py:20`, revisar si el subset lo usa), `jupyter`/`ipython` (`notebooks`).

APIs de pandas obsoletas (eliminadas en pandas 3.0), todas verificadas por grep: `fillna(method='ffill')` en `forecaster.py:400-401` y `dates.py:1023-1025`; `DataFrame.applymap` en `forecaster.py:472, 1031-1032` y `dataviz.py:3269`; `groupby(level=0, axis=1)` en `lib/utils.py:91` y `loader.py:145,150`; alias de frecuencia `freq='M'` en `forecaster.py:723, 961`. En ejecución (`Computer(scope='es').build_series()` con warnings activados) aparecen además 338 `PerformanceWarning` por DataFrame fragmentado (`computer.py:469-471`), 17 `FutureWarning` por `groupby` categórico sin `observed=` (`lib/data.py:112`) y 2 `FutureWarning` por downcasting en `fillna` (`helpers.py:860`); todas ocultas por el filtro global.

## 6. Empaquetado, licencia, cita y secretos

- No hay `pyproject.toml`/`setup.py`; los notebooks hacen `sys.path.append('..')` (o `'../..'`) para importar `mtpy`.
- `README.md` tiene una línea. `AGENTS.md` está en `.gitignore:4` (no versionado).
- `LICENSE`: MIT, © 2019 Luis Sancho. Adecuada para un repositorio científico; conviene actualizar el año.
- `CITATION.cff` (v1.2.0) sólo incluye `title`, `authors`, `date-released` (2025-02-18) y `url`; faltan `version`, `license`, `repository-code`, `abstract`, `keywords`, `identifiers`/`doi` (Zenodo) y ORCID.
- Notebooks en **Git LFS** (`.gitattributes:1`): los cinco notebooks raíz están almacenados como punteros LFS (verificado con `git lfs ls-files`); clonar exige `git-lfs`, GitHub no renderiza ni difiere su contenido y consume cuota LFS.
- Secretos: `.env`, `deploy/docker.env` y `deploy/elections.env` **no están versionados ni en el historial** (`git ls-files`, `git log --all -- '*.env'` vacío; `.gitignore:16 *.env`). Sin embargo `deploy/elections.env` contiene valores reales de `AWS_KEY` (20 caracteres), `AWS_SECRET` (40), `DB_PASSWORD`, `PUSHOVER_TOKEN` y un `DB_HOST` remoto, y `deploy/docker.env` también credenciales AWS. Viven dentro del árbol del repo: un `git add -f`, un cambio de `.gitignore` o un `zip` del directorio los expondría. No se detectó ningún secreto en ficheros rastreados.
- `Dockerfile` instala nginx, supervisor, supercronic, cmake y `COPY . .` (con `.dockerignore` excluyendo `files/`, `log/`, `*.env`); `deploy/docker/crontab:1` ejecuta `python /app/job.py rep` y no existe ningún job `Rep` (`mtpy/jobs/` sólo tiene `Pipeline`, `Test`, `Update`).

## 7. Layout objetivo recomendado

```
elections-model/
├── pyproject.toml            # nombre p.ej. "electoral" o "esforecast"; requires-python >= 3.11
├── README.md, LICENSE, CITATION.cff, CHANGELOG.md, CONTRIBUTING.md
├── src/<pkg>/
│   ├── __init__.py           # __version__
│   ├── config.py             # settings (dataclass/pydantic-settings) + .env
│   ├── stats/                # kernel.py, estimators.py, stat.py  (de core/utils/stat.py)
│   ├── viz/                  # subset de dataviz (33 funciones) sin `learning`
│   ├── data/                 # repository.py (getters), schema.py (Model meta), backends/{postgres,sqlite,parquet}.py
│   ├── ingest/               # loader.py (Wikipedia, Infoelectoral)  [extra "ingest"]
│   ├── computer.py, forecaster.py, simulator.py, blocks.py (lib/utils.py)
│   └── _compat/helpers.py, dates.py (subset)
├── data/                     # params.json, es-provinces.csv, es-regions.csv, wikipedia/*.json (versionados)
├── examples/ o docs/notebooks/   # notebooks raíz (jupytext .py pareado + nbstripout)
├── tests/                    # pytest: stat (kernels, ISJ, LOESS vs. statsmodels/sklearn), blocks, D'Hondt, formato de datos
└── .github/workflows/ci.yml  # ruff + pytest + nbmake (opcional)
```

Tooling: `ruff` (lint + format, PEP8), `pytest` + `pytest-cov`, `pre-commit` (ruff, nbstripout, end-of-file), `mypy` opcional, GitHub Actions (matriz 3.11/3.12, numpy 1.26 y 2.x), `pip-tools`/`uv` para un lock (`requirements.lock`), `zenodo` + `CITATION.cff` completo para DOI, `mkdocs-material` + `mkdocstrings` para documentación divulgativa. Extras en `pyproject`: `[db]` (sqlalchemy, psycopg2-binary), `[ingest]` (lxml, requests, openpyxl), `[notebooks]` (jupyter, ipywidgets), `[dev]`.

Para la reproducibilidad científica es clave desacoplar los datos de PostgreSQL: exportar las tablas `elections.*` a Parquet/CSV (≈4.000 encuestas) y hacer que `lib/data.py` acepte un backend de ficheros; así cualquier lector puede ejecutar `Forecaster`/`Simulator` sin base de datos.

## 8. Notas positivas de diseño

- El modelo depende de una interfaz pequeña del framework (`Core`, `app.fs`, `Model.get_results/upsert`), lo que hace la extracción viable sin reescribir Computer/Forecaster/Simulator.
- `stat.py` no tiene dependencias del framework más allá de `dates`/`helpers`, y es alcanzable al 100 %.
- `Model.meta` documenta explícitamente el esquema de cada tabla (`models/elections.py`), útil como base de un `schema.md`.
- La sustitución `${VAR}` en `config.json` + `.env` separa bien configuración de código; `.env` y credenciales nunca han entrado en git.
- `.dockerignore`/`.gitignore` coherentes; el `Dockerfile` corre como usuario no root.

## 9. Pipeline (flujo de infraestructura)

1. `import mtpy.mtpy` — `mtpy/mtpy.py:1-16`: importa `dal`, `api`, `app`, `io`, `services.aws/pushover/s3`, `helpers`; `warnings.filterwarnings('ignore')` global.
2. `mtpy.run()` — `mtpy/mtpy.py:19-47`: rutas `abspath/shpath/fspath`, carga `.env`, lee `config.json`, sustituye `${VAR}` (ausente → `''`), construye `Config`.
3. Adaptador BD — `mtpy/mtpy.py:51-58`, `core/dal/__init__.py:5-12`, `core/data.py:343-484`: `getattr(dal, 'PostgreSQL')` (import perezoso), `create_engine` diferido, túnel SSH opcional.
4. Logging/FS/notificaciones/semilla — `mtpy/mtpy.py:69-91`: `Log(shpath/log)` (falla si no existe `log/`), `S3` o `FileSystem(fspath)`, `Pushover`, `np.random.seed(42)`.
5. Instanciación de modelos — `core/data.py:1982-2024`: `_get_dal()` → `app.db`; `parse_meta(meta)`.
6. Lectura — `core/data.py:2158-2190`: `Model.get_results(query={columns, filters, sort, relations}, formatted)` → `SQLAdapter.build_select_query` → `pd.DataFrame` → `format_data` (casting por `meta`).
7. Ficheros del modelo — `core/io.py:38-59, 262-380, 634-711`: `fs.read(f'{path}/params.json')` con `path='.'` → `files/./params.json`; salidas `files/fc/*.csv`, `files/img/*.png`.
8. Escritura (excluida en este análisis) — `lib/data.py:413-420, 442-458`, `loader.py:187-211, 675-700`: `execute(UPDATE/DELETE)` + `Model.upsert(df)` → `SQLAdapter.upsert` (`INSERT ... ON CONFLICT`).
9. Runtime no utilizado — `api.py`, `worker.py`, `job.py`, `pipeline.py`, `Dockerfile`, `deploy/`: API ASGI + gunicorn/nginx, worker SQS (`app.qs` nunca se define), jobs `Update` (stripe/aemet) y `pipelines/events` (rotos).

## 10. Hallazgos

| # | Sev. | Tipo | Dónde | Hallazgo |
|---|---|---|---|---|
| 1 | alta | reproducibilidad | `core/app.py:173`, `mtpy.py:73`, `.gitignore:3` | `mtpy.run()` lanza `FileNotFoundError` en un clon limpio porque `Log` abre `log/info.log` y `log/` está ignorado. Verificado. |
| 2 | alta | diseño/deps | `core/utils/dataviz.py:34-43`, `core/utils/learning.py:5-13` | Importar el modelo carga `xgboost`, `mlxtend`, `sklearn`, `statsmodels`... por un import de `learning` que sólo usan 3 funciones de trazado no alcanzables (`dataviz.py:3848, 3941, 4550`). `xgboost` ni siquiera está en `requirements.txt`. |
| 3 | alta | reproducibilidad | `requirements.txt` | Pines desfasados frente al entorno (numpy 1.25.2 vs 2.4.4, scipy 1.9.3 vs 1.17, sklearn 1.2 vs 1.8...), faltan `xgboost`, `lxml`, `openpyxl`; ≈25 paquetes no usados por el modelo. |
| 4 | media | bug latente | `forecaster.py:400-401,472,723,961,1031-1032`; `lib/utils.py:91`; `loader.py:145,150`; `dates.py:1023-1025`; `dataviz.py:3269` | APIs de pandas eliminadas en 3.0 (`fillna(method=)`, `applymap`, `groupby(axis=1)`, `freq='M'`). Ocultas por el filtro global de warnings. |
| 5 | media | diseño | `mtpy.py:16` | `warnings.filterwarnings('ignore')` a nivel de módulo silencia todas las advertencias del proceso (incl. las de NumPy/pandas/scipy sobre estadística). |
| 6 | media | código muerto/roto | `pipelines/events/Events.py:33,36-38,144`; `models/sources/elections.py` (vacío); `jobs/Update.py:6-13`; `deploy/docker/crontab:1`; `core/worker.py:32` | `sources.Events()` no existe; `config.elections.*` no está en `config.json`; `self.target.dal.query` no existe (`Model` sólo tiene `_dal`/`execute`); `Update` referencia pipelines stripe/aemet inexistentes; cron llama a `job.py rep` sin job `Rep`; `Worker` usa `app.qs` que nunca se registra. |
| 7 | media | seguridad | `deploy/elections.env`, `deploy/docker.env`, `.env` | Credenciales reales (AWS key/secret, DB password, Pushover token) dentro del árbol del repo aunque ignoradas; no hay `.env.example`. Recomendado rotar y mover fuera del repo. |
| 8 | media | reproducibilidad | `.gitattributes:1` | Notebooks en Git LFS: punteros en GitHub, no renderizables ni diffables; cuota LFS. |
| 9 | media | empaquetado | notebooks (`sys.path.append('..')`), sin `pyproject.toml` | El paquete no es instalable; los notebooks dependen de la ruta relativa. |
| 10 | media | reproducibilidad | `.gitignore:2` (`files/`) | Los inputs imprescindibles (`files/params.json`, `es-provinces.csv`, `es-regions.csv`, `wikipedia/wp-*.json`) no están versionados; nadie puede ejecutar el modelo desde el repo. |
| 11 | baja | bug | `lib/data.py:455` | `dp['rating'] = ...astype(int)` trunca un rating `numeric(16,2)` (valores hasta 84.77 en `pollsters_ratings`) al guardarlo en `pollsters.rating`. Verificado en BD: `pollsters.rating` sólo tiene enteros. |
| 12 | baja | diseño | `lib/data.py:7` | `from ..core.worker import Model` (re-export) en lugar de `from ..core.data import Model`; arrastra `worker.py` (signal, jobs) sin necesidad. |
| 13 | baja | diseño | `core/app.py:25-27` | `App.__getattr__` devuelve `None` para dependencias no registradas → errores tardíos y opacos (`'NoneType' has no attribute ...`). |
| 14 | baja | bug latente | `core/app.py:46-59` | `class_exists`/`get_class` aplican `hasattr` a una cadena; nunca resuelven nombres con punto. No usado por el modelo. |
| 15 | baja | diseño | `mtpy.py:44-46`, `core/app.py:91-103` | Variables de entorno ausentes se sustituyen por `''` sin aviso; `getattr(dal, '')` produce un error críptico. `offset_is_empty` no es recursivo. |
| 16 | baja | diseño | `core/app.py:163` | Loggers con nombres globales `'info'`, `'error'`, `'debug'` que pueden colisionar con otras librerías; el logger `debug` no tiene handler si `verbose=0`. |
| 17 | baja | reproducibilidad | `mtpy.py:91`, `config.json:16` | Siembra del RNG global legacy (`np.random.seed(42)`) en el arranque; el estado depende del orden de ejecución de celdas. Preferible `np.random.default_rng(seed)` inyectado. |
| 18 | baja | diseño | `lib/data.py:413-418,442-446`; `loader.py:187,206,675,695` | SQL por interpolación de cadenas (fechas/ámbitos). Sin riesgo externo, pero frágil; usar parámetros de SQLAlchemy. |
| 19 | baja | rendimiento | `computer.py:469-471`; `lib/data.py:112`; `helpers.py:860` | 338 `PerformanceWarning` (inserción columna a columna), `groupby` categórico sin `observed=`, downcasting en `fillna`. |
| 20 | baja | docs | `CITATION.cff`, `README.md`, `LICENSE` | Cita incompleta (sin `version`, `license`, `repository-code`, DOI, ORCID); README de una línea; año de licencia 2019. |
| 21 | baja | empaquetado | `Dockerfile`, `deploy/docker/*` | Infraestructura de servicios (nginx, supervisor, supercronic, gunicorn/uvicorn) innecesaria para el modelo; `.gitignore:4` excluye `AGENTS.md`. |
| 22 | baja | diseño | `core/dal/__init__.py:5-12`, `jobs/__init__.py`, `pipelines/__init__.py`, `learners/__init__.py`, `controllers/__init__.py` | Carga dinámica por `__getattr__` de paquete: oculta dependencias al análisis estático y a los linters. |

## 11. Preguntas abiertas para el autor

1. ¿El repositorio científico será independiente de ManyThings o seguirá compartiendo `mtpy` (y el directorio hermano `../../shared`) con otros proyectos?
2. ¿Se necesita mantener el backend PostgreSQL, o es aceptable publicar un volcado (Parquet/CSV/SQLite) de `elections.*` y un backend de ficheros como modo por defecto?
3. ¿Qué licencia/permiso tienen los datos de entrada (tablas de Wikipedia, ficheros de Infoelectoral, `Electomania-Rankings.xlsx`) para redistribuirlos en el repo?
4. ¿Se quieren conservar los notebooks históricos `notebooks/events/es-201911` y `es-202307` como registro de predicciones publicadas (congelados) o eliminarlos?
5. ¿Las credenciales de `deploy/elections.env` y `deploy/docker.env` siguen activas? ¿Se pueden rotar y retirar del árbol del repo?
6. ¿Versión mínima de Python objetivo? El código ya usa `str | tuple` (3.10+) y `typing.Self` en `services/google.py` (3.11+).
7. ¿Se quiere soportar NumPy 2.x oficialmente (hoy funciona con 2.4.4 pero el pin dice 1.25.2)?
8. ¿Qué nombre tendrá el paquete (`mtpy` es el framework; el modelo necesita un nombre propio para PyPI/Zenodo)?
9. ¿`pollsters.rating` debe ser entero (ranking) o decimal? El cast en `save_ratings_data` sugiere lo primero, el esquema `numeric(16,2)` lo segundo.
