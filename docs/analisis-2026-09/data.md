# Módulo: Capa de datos, cargadores y reproducibilidad

Ficheros analizados (rutas absolutas bajo `/Users/luiss/HD/Proyectos/Code/elections-model`):
`mtpy/lib/data.py` (542 líneas), `mtpy/lib/loader.py` (752), `mtpy/models/elections.py` (274), `mtpy/core/data.py` (clase `Model`, l. 1930+; `SQLAdapter.build_select_query` l. 1059+), `mtpy/core/utils/helpers.py::format_data` (l. 357+), `mtpy/mtpy.py::run`, `files/params.json`, `files/wikipedia/wp-urls.json`, `files/wikipedia/wp-maps.json`, `files/es-provinces.csv`, `files/es-regions.csv`, `files/infoelectoral/es/PROV_02_202307_1.json`, `files/infoelectoral/es/PROV_02_197706_1.json`, `notebooks/data-load/*.ipynb`. Se ha consultado la base de datos PostgreSQL en modo sólo lectura y se ha ejecutado `WikipediaLoader.read_data()`/`build_series()` en seco (sin guardar) para verificar el parser.

---

## 1. Visión general

La capa de datos del modelo tiene tres piezas:

1. **Modelos ORM ligeros** (`mtpy/models/elections.py`): 13 clases que declaran tabla, clave primaria, orden y `meta` (tipos de columna) sobre el esquema PostgreSQL `elections`. Toda lectura/escritura pasa por `mtpy.core.data.Model` (`get_results`, `get_row`, `get_var`, `get_agg`, `upsert`, `stage_write`, `format_data`).
2. **Cargadores** (`mtpy/lib/loader.py`): `InfoElectoralLoader` (resultados oficiales por provincia desde XLSX del Ministerio del Interior) y `WikipediaLoader` (encuestas desde las tablas "Opinion polling for the … Spanish general election" de la Wikipedia inglesa).
3. **Funciones de acceso** (`mtpy/lib/data.py`): getters que devuelven DataFrames "anchos" (una columna por partido) para `Computer`, `Forecaster` y `Simulator`, más dos funciones de guardado de resultados calculados (`save_model_data`, `save_ratings_data`) y una derivación automática de parámetros por evento (`get_event_params`).

Además existen ficheros de configuración/curación manual en `files/` (ignorado por git, `.gitignore` l. 2): `params.json`, `wikipedia/wp-urls.json`, `wikipedia/wp-maps.json`, `es-provinces.csv`, `es-regions.csv`, `infoelectoral/es/PROV_02_YYYYMM_1.{xlsx,json}`, además de `Electomania-Rankings.xlsx`, `es-regions-ages.csv` (no referenciados por el código) y las salidas `fc/*.csv`, `img/*.png`.

### 1.1 Bootstrap y raíz del sistema de ficheros

`mtpy.run()` (`mtpy/mtpy.py` l. 19-95) fija `abspath` = raíz del repo, `shpath` = `abspath` (salvo que exista `../../shared`), **`fspath = shpath + '/files'`** (l. 35) y carga `.env` desde `shpath` (l. 39-40). `config.json` se interpola con variables de entorno (`DB_ADAPTER`, `DB_HOST`, …, l. 42-47). Si `S3_BUCKET` está definido, `app.fs` es un `S3`; si no, `FileSystem(fspath)` (l. 80-85). Consecuencia importante: **todos los argumentos `path` de `Computer`, `Forecaster`, `Simulator` y los loaders son relativos a `files/`**; los notebooks pasan `path='.'` (`notebooks/data-load/PollsLoadWikipedia.ipynb` celda 1). El valor por defecto `path or os.getcwd()` (`loader.py` l. 39 y 289; `computer.py` l. 132) produce una ruta absoluta concatenada a `fspath`, que no existe (comprobado: con `path='files'` se busca `mtpy/../files/files/wikipedia/wp-urls.json`).

---

## 2. Esquema de datos (`mtpy/models/elections.py`)

Tipos `meta`: `'dtd'` fecha, `'cat'` categórica, `'str'`, `'int'`/`['int', n]`, `'num'`, `'bin'`, `'obj'` (JSON). Estado real en la BD (consulta a `information_schema` y recuento de filas, 2026-09-23):

| Tabla | Clave | Filas | Contenido y origen |
|---|---|---|---|
| `parties` (l. 4-18) | `id` autokey | 515 | Catálogo manual: `name` (sigla), `fullname`, `parent_id`, `color`, `block` ∈ {Derecha 79, Izquierda 104, Regionalista 226, Separatista 105, Otros 1}, `regional`. El id 0 es `'-'` = "Otros". |
| `pollsters` (l. 21-33) | `id` | 95 | Manual: `name`, `mtype` (sólo 10 informados: state 2, online 4, aggr 4), **`quality`** (manual, 0-78; 53 de 95 valen 5, 3 valen 0), `rating` (escrito por `save_ratings_data`). |
| `sponsors` (l. 36-45) | `id` | 81 | Manual (medios que encargan la encuesta). |
| `provinces` (l. 48-62) | `id` | 52 | `country, name, slug, ncode, scode, seats`. **No la usa ningún módulo de `mtpy/lib`** (grep sin resultados). |
| `pollsters_ratings` (l. 65-90) | `(event_date, event_scope, pollster_id)` | 1012 = 92 × 11 eventos (1996-2027) | Salida de `Computer.compute_ratings`. |
| `pollsters_parties` (l. 93-105) | 4 cols | 0 | Vacía, sin uso. |
| `polls` (l. 108-148) | `(event_date, event_scope, date, pollster_id, sponsor_id)` | 3825 | Encuestas: columnas "crudas" (`pollster`, `sponsor`, `name`, `start_date`, `end_date`, `pub_date`, `mtype`, `ctype`, `sample_size`, `parties`, `days`, `featured`, `computed`, `notes`, `url`) mezcladas con columnas **calculadas** por `Computer` (`proc_sample`, `rating`, `error_*`, `bias_*`, `weight_*`). |
| `polls_results` (l. 151-172) | clave de `polls` + `party_id` | 32 876 | `party`, `pct`, `seats`, `seats_min`, `seats_max` (crudas) + `error`, `bias` (calculadas; 22 590 informadas). |
| `forecasts`, `forecasts_results` (l. 175-215) | | 0 | Vacías: `Forecaster` escribe CSV en `files/fc/` (`forecaster.py` l. 452-470), no en estas tablas. |
| `events` (l. 218-230) | `(date, scope)` | 17 | Manual: 1977-06-15 … 2023-07-23 + placeholder `2027-08-22` "Elecciones Generales Próximas". `featured` = True sólo 1993-2023 (11 eventos). |
| `events_data` (l. 233-254) | `(date, scope, region_id)` | 901 = 53 × 17 | Totales por provincia (`region_id` 0 = nacional): `seats, population, stations, registered, counted, votes, abstentions, blank, invalid`. |
| `events_results` (l. 257-274) | `+ party_id` | 55 862 | Votos, `pct` y escaños por provincia y partido (incluye `party_id` 0 "-"). |

Restricciones en BD: sólo PRIMARY KEY y NOT NULL; **no hay claves foráneas** (`pg_constraint`), ni DDL/migración versionada en el repo (la creación se hace con `Model.create()`, `core/data.py` l. 2467+, o a mano).

---

## 3. Acceso a datos: `mtpy/lib/data.py`, función por función

Todas las consultas se construyen concatenando strings de filtro SQL (`"scope = '{}'".format(scope)`) que `SQLAdapter.build_select_query` (`core/data.py` l. 1059-1121) envuelve en `SELECT "t"."col"… FROM tabla AS "t" [LEFT JOIN …] WHERE f1 AND f2 ORDER BY sort`. Con `formatted=True`, `Model.format_data` → `helpers.format_data` (l. 357-424) **reindexa el DataFrame a las columnas de `meta`** (l. 85-89: añade las que faltan y descarta las que sobran) y aplica tipos.

### `get_event_dates(scope='es', date_from, date_to, skip=0, min_polls=0, featured=False)` (l. 463-521)
Devuelve fechas de elección (`'YYYY-MM-DD'`) ordenadas. Si `min_polls > 0` agrega `polls` por `event_date` y exige `count >= min_polls` (l. 498-503); si no, lista `events` (l. 505-508). `skip` elimina las *k* últimas. El docstring dice `min_polls` "default 1" (l. 484) pero el valor real es 0.

### `get_next_event_date(scope='es', date_from=None)` (l. 524-542)
`Events().get_var(filters=[scope, date > date_from])` → `SELECT "t".* … LIMIT 1` **sin ORDER BY** (`Model.get_var`, `core/data.py` l. 2221-2238, no añade `sort` a diferencia de `get_results`/`get_row`). Devuelve la primera columna de la primera fila (`date`, por orden físico de columnas). Correcto hoy porque sólo hay un evento futuro.

### `get_event_series(scope, event_dates=None, metric='pct')` (l. 195-250)
`events ⋈ events_data[region_id=0]` (inner, por `date`) ⋈ pivote de `events_results[region_id=0]` con `party` en columnas y `metric` ∈ {pct, votes, seats} como valor (left). Salida: una fila por elección con `date, scope, name, featured, region_id, region, seats, population, …, invalid` + una columna por partido. Es la "verdad terreno" para `Computer.load_events` (`computer.py` l. 320-330).

### `get_poll_series(scope, event_dates=None, metric='pct', drange=None, drop_mtypes=None, drop_contexts=None)` (l. 277-327)
`polls` filtrado por `t.event_scope`, `t.event_date IN (...)`, opcionalmente `t.days BETWEEN a AND b` (l. 296-302; nótese que `drange=(0, x)` cae en la rama `elif drange[1]` porque `0` es falsy, y `(x, 0)` en `elif drange[0]`), `t.mtype NOT IN` (l. 304) y **`t.context NOT IN` (l. 306): la columna `context` no existe en `polls` (la columna real es `ctype`), así que usar `drop_contexts` produce un error SQL**. Los resultados se leen con `relations=[{'model': Polls(), 'name': 'p'}]` (LEFT JOIN por la clave compuesta) y los filtros se reescriben `t.`→`p.` (l. 317). Se pivota `polls_results[metric]` (`pct`, `seats`, `error`, `bias`) por `party` y se une a `polls` **por `['date','pollster_id','sponsor_id']` sin `event_date`** (l. 322). Hoy no hay ninguna tripleta compartida por dos `event_date` (consulta: 0 casos), pero la unión duplicaría filas si la hubiese.

### `get_ratings(scope, event_dates=None)` (l. 330-365)
Lee `pollsters_ratings` para los eventos pedidos **más el siguiente evento** a `event_dates[-1]` (l. 341-343, para disponer del rating "vigente" calculado *para* la próxima elección), une el nombre del encuestador y ordena por `(event_date, pollster_id)`.

### `get_parties()` (l. 368-372) / `get_pollsters()` (l. 375-378)
`get_parties` mueve la fila 0 (`id` 0, `'-'` Otros) al final (`rename(index={0: n})`), de modo que el orden por defecto es `PSOE, PP, VOX, …, '-'`.

### `get_event_data`, `get_event_results`, `get_poll_results` (l. 142-274)
Filtros `scope`, `date IN (...)` y opcional `region_id`. A diferencia de `get_event_series`/`get_poll_series`, **no contemplan `event_dates=None`** (l. 154, 181, 264 → `TypeError` en `join(None)`), aunque la firma lo declare `Optional`.

### `get_event_dmat(scope, event_date, polls, events, parties)` (l. 14-51)
Construye la "matriz del evento": toma las encuestas del evento con las columnas de partido, ordena partidos por media en encuestas y por resultado final, y si hay encuestas con `days ∈ [1, 42]` se restringe a las **6 últimas semanas** (l. 40-41). Añade filas agregadas con `days=0`:
- `count` = 100 · nº encuestas que incluyen al partido / nº encuestas (l. 43-44),
- `mean`, `std` de `pct`,
- `result` = resultado oficial.
Salida: índice `(days, pollster)`, columnas = partidos (primero los que obtuvieron resultado, luego el resto).

### `get_event_params(scope, event_dates=None, path=None)` (l. 53-139)
Deriva automáticamente por evento la estructura que consumen `Computer`/`Forecaster`/`Simulator` y la **sobrescribe con `files/params.json`**:
- `parties.event` = partidos con resultado oficial; `parties.polls` = todos los que aparecen en encuestas o resultado (l. 92-95).
- `bmaps.main` = partidos presentes en **> 95 %** de las encuestas de las últimas 6 semanas (l. 97-98).
- `bmaps.blocks` = partidos de `parties.polls` agrupados por `parties.block` (sólo Derecha/Izquierda/Regionalista/Separatista); `bmaps.vs` = ídem sólo con `main` y sólo Derecha/Izquierda (l. 107-119).
- `smap` = `{partido_nuevo: []}` para partidos con resultado que no existían en la elección anterior (l. 100-105, 131).
- **Fusión superficial** `smap | params` (l. 134): si `params.json` trae `bmaps`, sustituye el diccionario completo (p. ej. para 2019-04-28 no hay `max` ni `min`; `Computer.merge_bmaps('max')` (`computer.py` l. 195-221) daría `KeyError` en eventos sin esa entrada).
`params.json` (sólo 4 eventos: 2027-08-22, 2023-07-23, 2019-11-10, 2019-04-28) define además `bmaps.max` (lista de partidos individuales), `bmaps.min` (fusiones, p. ej. `"VOX+SALF": ["VOX","SALF"]`) y `smap` con reglas `{'type': 'agg'|'sub'|'split', 'names': [...], 'regions': [...]}` para trasladar votos entre partidos entre elecciones (consumido por `Simulator`, `simulator.py` l. 96-113).

### `save_model_data(model, data)` (l. 381-420)
Para `Polls`/`PollsResults`: formatea con `int_type='nullable'`, **pone a NULL todas las columnas no clave del DataFrame para los `event_scope`/`event_date` afectados** (`UPDATE … SET col = NULL`, l. 413-418) y hace `upsert` (`ON CONFLICT (key) DO UPDATE`, `core/data.py` l. 2409-2434). Es el mecanismo por el que los cálculos de `Computer` viven mezclados con los datos crudos en la misma tabla.

### `save_ratings_data(data)` (l. 423-460)
`DELETE` de `pollsters_ratings` para los eventos afectados + `upsert`; después escribe en `pollsters.rating` el último rating de cada encuestador con `fillna(0).astype(int)` (l. 453): **truncamiento, no redondeo** (BD: `Sigma Dos` 73 vs 73.92; 56 discrepancias).

---

## 4. `InfoElectoralLoader` (`loader.py` l. 23-260)

Entrada: `files/infoelectoral/es/PROV_02_YYYYMM_1.xlsx` (fichero "02 = Congreso, resultados por provincia" del portal infoelectoral.interior.gob.es; 16 elecciones 1977-2023 presentes) y el mapa manual `PROV_02_YYYYMM_1.json` {nombre de candidatura en el XLSX → sigla de `parties.name` | null}.

1. `__init__` (l. 27-67): lee `es-regions.csv` (→ `self.regions`, **nunca usado**) y `es-provinces.csv` (`code → province`, usado para el nombre de región), catálogo `parties`, el JSON de mapeo (si existe) y `parties_idmap`.
2. `read_file` (l. 69-81): abre el XLSX **con la ruta local `app.fspath`** (l. 70; el resto del código usa `app.fs`, que podría ser S3), toma la primera hoja, localiza la fila de cabeceras (primera con la 2.ª columna no nula), la de totales (última con la 2.ª columna nula) y la fila de nombres de candidatura **dos filas por encima** de las cabeceras (l. 77). Devuelve las filas de datos incluida la de totales.
3. `read_data` (l. 83-153): `col_map` traduce las 16 cabeceras oficiales ("Censo CERA", "Votos válidos", …) a `state, region_id, region, population, stations, registered_*, counted_*, votes, votes_candidates, blank, invalid`. Para cada candidatura: mapeada → sigla; `null` → `'-'` (Otros); ausente → `parties_missing` **y no se añade a `party_cols`**, lo que desalinea `votes.columns = party_cols` (l. 141-142) y provoca un `ValueError` de longitud (falla ruidosamente, pero con un mensaje poco explicativo). Si ninguna candidatura mapea a `'-'` se añaden dos columnas artificiales a 0 (l. 123-126). Las columnas de partidos van intercaladas (pares = votos, impares = escaños, l. 140/145); se suman por sigla con `groupby(level=0, axis=1).sum()` (l. 143, 148; **`axis=1` está deprecado en pandas 2.x y desaparece en 3.0**). El índice `region_id` es el código de provincia, con la fila de totales → 0. Salida `self.data` con MultiIndex de columnas `totals | votes | seats`.
4. `build_series` (l. 155-178): `totals` → filas de `events_data` (`abstentions = registered − counted`, `region` por `es-provinces.csv`, `seats` = suma de escaños de partidos por región, l. 172, alineada por posición); `results` → filas largas de `events_results` con `pct = round(100 · votes / votes_válidos_región, 2)` (l. 168; el denominador incluye votos en blanco: la suma nacional de `pct` es 98.4-99.8 %).
5. `save_totals`/`save_results` (l. 180-218): `DELETE … WHERE scope AND date` + `stage_write` (gz) + `upsert` + `vacuum`.
6. `show_summary`, `votes`, `seats`, `pcts` (l. 220-260): comprobaciones (diferencia entre votos válidos − blancos y suma de votos por candidatura; nº de candidaturas con voto/escaño).

Decisiones de curación incrustadas en los JSON: se agrupan candidaturas bajo una misma sigla con criterio político (p. ej. 1977: `"FALANGE ESPAÑOLA DE LAS JONS": "AN18"`, `"ALIANZA FORAL NAVARRA": "AP"`, `"UNION NAVARRA DE IZQUIERDA": "EE"`; 2023: `"SORIA ¡YA!"`, `"ARAGÓN EXISTE"`, `"ESPAÑA VACIADA"` → `"EV"`). Son decisiones legítimas pero no están documentadas en ningún sitio del repositorio.

Los datos de `events_data` para el evento placeholder `2027-08-22` no salen de este loader (no hay XLSX): son una copia manual de 2023 (`seats`, `votes`, `registered` idénticos en las 53 filas; `population` actualizada). Es una hipótesis implícita del `Simulator` (reparto de escaños por provincia = 2023).

---

## 5. `WikipediaLoader` (`loader.py` l. 263-752)

Entrada: `wp-urls.json` (17 eventos; clave `polls` = 1..3 URL de la Wikipedia inglesa; `years` opcional; la clave `results` **no se usa**), `wp-maps.json` (alias → nombre canónico para `parties` (slug del `href` de Wikipedia), `pollsters` y `sponsors`), y los catálogos en BD (`parties`, `pollsters`, `sponsors`, `events`).

1. `__init__` (l. 265-323): carga el evento (`Events.get_row`), URLs, mapas y diccionarios `colmap` (alias→canónico) e `idmap` (nombre→id).
2. `read_data` (l. 553-604): `requests.get` con User-Agent `ManyThings/1.0 …` **sin caché ni archivo del HTML**; para cada `table.wikitable` toma el año del encabezado precedente (`./preceding-sibling::div[1]/*[h3|h4|h5]`, l. 574; depende del marcado `mw-heading` actual de Wikipedia), lo filtra por `years`, y **si no hay `years` sólo procesa la primera tabla** (`break`, l. 603). Deduplica por `(date, pollster_id, sponsor_id)` (l. 598; la última tabla sobrescribe).
3. `read_table` (l. 535-551): partidos = `th[4:-1]` de la primera fila → `parse_party` (l. 330-337, extrae el slug tras `/wiki/`); filas desde `tr[2:]`. Verificado en vivo (2026): cabecera = `Polling firm/Commissioner, Fieldwork date, Sample size, Turnout, <15 partidos>, Lead`.
4. `read_rows` (l. 470-533): reconstruye celdas expandiendo `rowspan`/`colspan` (cola `remainder`); detecta el **contexto por color de fondo de la fila** (`#EAFFEA` → `exit`, `#FFEAEA` → `wban`, l. 480-484) y lo guarda en la clave **`context`**.
5. `read_cols` (l. 363-468): descarta filas cuya primera celda está en negrita (resultado electoral, l. 368-371); `pollster/sponsor` separando por `/`; si no hay `/`, concatena la nota `<span>` al nombre (l. 385-386) → p. ej. `"CIS (Target Point)"`; aplica alias y **si el encuestador no existe en `pollsters` la fila se descarta en silencio** (l. 400-401; en el dry-run de 2026 se pierden las etiquetas `CIS (Ateneo del Dato)`, `CIS (Logoslab)`, `CIS (Sondaxe)`, `CIS (Target Point)`, `Sumar`); sponsor desconocido → `sponsor_id = 0`. Fechas: `parse_dates` (l. 339-361) interpreta `"28 Feb–3 Mar 2024"`, `"8–12 Jan 2024"`, `"15 Jan"` y cruces de año (`"30 Dec–2 Jan 2024"` → 2023-12-30/2024-01-02, verificado); `date = end_date`. `sample_size` numérico o `None` (`"?"` → `None`). Si la celda `Lead` no es numérica la fila se descarta (l. 424-426). Por partido: `pct` (se omite si ≤ 0 o no numérico), escaños del `<span>` `"a/b"` → `seats_min`, `seats_max`, `seats = floor(mean)` (l. 448-455).
6. `build_series` (l. 606-651): aplica `exclude`; `parties` = nº de resultados válidos; `event_date/scope`; **`pub_date = end_date`** (l. 648; en BD 3825/3825 filas cumplen `pub_date = end_date = date`: no existe fecha real de publicación); `mtype` = `pollsters.mtype` del encuestador (l. 649: es un atributo del encuestador, no de la encuesta); `computed = internal = partisan = False` (las dos últimas no están en `meta` y se descartan); `days = event_date − date`. Después `format_data` **reindexa a las columnas de `Polls.meta`, por lo que `context` se descarta y `ctype` queda siempre NaN** (verificado: dry-run 2023 detecta 13 filas `wban`, `loader.polls['ctype'].notna().sum() == 0`; la serie 2027 tiene 0 `ctype` en BD).
7. `select_series(overwrite=False)` (l. 653-659): modo append-only: elimina las claves ya presentes en BD. Con `overwrite=True`, `save_polls`/`save_results` (l. 661-697) hacen `DELETE` del evento completo y recargan, **perdiendo cualquier columna curada a mano** (`ctype`, `notes`, `url`) que el loader no produce.
8. `compute_series` (l. 699-714): lanza `Computer(...).build_series().compute_weights(save)` para el evento.
9. `parties_checks` (l. 717-752): tabla partidos-en-resultado × partidos-en-encuestas.

---

## 6. Flujo operativo real (notebooks `data-load/`)

1. `ResultsLoadInfoElectoral.ipynb`: descargar XLSX → generar plantilla JSON (celda 3, comentada: imprime `"nombre": "",` por candidatura) → rellenar a mano → `read_data` → `build_series` → `save_totals`/`save_results` (`save=False` en el notebook guardado).
2. `PollsLoadWikipedia.ipynb`: `WikipediaLoader('es','2027-08-22', path='.')` → `read_data` (489 encuestas 2023-2026; 9 etiquetas de encuestador perdidas) → `build_series` → `select_series(overwrite=False)` (6 nuevas) → `save_polls`/`save_results` → `parties_checks` → `compute_series(save=True, overwrite=True)`.
3. `PollsCompute.ipynb`: `Computer(scope='es', event_dates=None, path='.').build_series()` (3842 × 511) → `compute_weights(save=True, overwrite=True)` → `compute_errors(save=True)` → `compute_deviations(save=True)` → `compute_ratings(save=True)` (1012 ratings, 3825 polls actualizadas).

Prerrequisitos manuales no versionados: alta de nuevos `pollsters`/`sponsors`/`parties` en BD (por SQL o notebook `lab/`), edición de `wp-maps.json`, `params.json`, `pollsters.quality`, `events` (fecha placeholder y `featured`), `events_data` del evento futuro.

---

## 7. Caracterización de los datos (BD, sólo lectura)

- **Eventos**: 17 (16 elecciones 1977-2023 + placeholder 2027-08-22). `featured` 1993-2023.
- **Encuestas por evento** (`n_polls`, % con `sample_size`, % con `mtype`, encuestadores): 1977: 10/90 %/0/6 · 1979: 7/100/0/5 · 1982: 31/45/0/4 · 1986: 43/81/2/13 · 1989: 44/70/30/13 · 1993: 53/60/13/12 · 1996: 77/82/34/13 · 2000: 123/81/26/14 · 2004: 213/68/47/16 · 2008: 327/71/32/23 · 2011: 332/79/6/24 · 2015: 404/81/5/25 · 2016: 146/86/3/24 · 2019-04: 371/77/14/26 · 2019-11: 152/78/31/23 · 2023: 890/83/32/34 · 2027: 602/92/55/25. Rango de fechas: la serie de cada evento empieza justo tras la elección anterior (p. ej. 2023: 2019-11-12 → 2023-07-22; 2027: 2023-07-25 → 2026-09-18).
- `sample_size` nulo en 744/3825 (19.5 %); mín. 402, máx. 210 000 (agregadores/paneles). `mtype`: NULL 2787, online 541, aggr 295, state 202. `ctype`: NULL 3741, wban 67, exit 17 (ninguno en 2027). `computed` = True en las 3825; `featured` = 3 597 (eventos 1993-2023). `rating` informado en el 100 % desde 1996. `notes`/`url`: 0 filas. `pollster` denormalizado coherente con `pollsters.name` (0 discrepancias).
- Nº de partidos por encuesta: 2 → 166 encuestas (suma de `pct` ≈ 52-60 %, p. ej. InvyMark/La Sexta 2012-13 sólo PP/PSOE), moda 6 y 16. Sumas de `pct` por encuesta: media 82-95 % según evento, ninguna > 101.
- `polls_results`: 32 876 filas, 57 partidos distintos, `seats` en 23 595, `error`/`bias` en 22 590.
- `events_results`: 486 partidos distintos, 46-97 partidos por elección, 56 filas por provincia en 1977.
- `pollsters_ratings`: 92 encuestadores × 11 eventos (1996-2027), rating medio ≈ 22, `weight_rating` medio ≈ 0.92, `quality` media 20.9.

---

## 8. Reproducibilidad y procedencia: diagnóstico

Un tercero **no puede** ejecutar hoy el pipeline: (a) necesita una PostgreSQL con el esquema `elections` y no hay DDL ni volcado; (b) los catálogos (`parties`, `pollsters`, `sponsors`, `events`, `provinces`) y las curaciones (`pollsters.quality`, `events_data` 2027, `ctype`) sólo existen en la BD del autor; (c) `files/` está en `.gitignore` (l. 2), así que `params.json`, `wp-*.json`, `es-*.csv` y los XLSX/JSON de Infoelectoral no están versionados; (d) el scraping de Wikipedia es en vivo, sin guardar el HTML ni el `oldid` de la revisión, por lo que dos ejecuciones no dan el mismo resultado; (e) `requirements.txt` fija `numpy==1.25.2`, `scipy==1.9.3` mientras el intérprete usado tiene numpy 2.4.4 / scipy 1.17.0, e incluye decenas de dependencias del framework genérico (boto3, dask, gunicorn, sklearn, statsmodels, pyzbar…) ajenas al modelo; (f) no hay tests; el README es de una línea.

### Licencias de las fuentes
- **Wikipedia** (tablas de encuestas): texto bajo **CC BY-SA 4.0** (y GFDL). Obligaciones: atribución (enlace a la página/revisión), indicar modificaciones, **share-alike** para obras derivadas del texto. Las cifras de encuestas son hechos (no protegibles como tales), pero la selección/estructura de la tabla y, en la UE, el derecho *sui generis* sobre bases de datos aconsejan publicar el dataset de encuestas bajo CC BY-SA 4.0 citando la página y el `oldid`, y atribuir a cada encuestador/medio (`pollster`, `sponsor`).
- **Infoelectoral (Ministerio del Interior)**: aviso legal consultado: reutilización permitida para fines comerciales y no comerciales con las condiciones de la Ley 37/2007: citar "Origen de los datos: Ministerio del Interior", mencionar la fecha de última actualización, no desnaturalizar la información ni sugerir patrocinio, conservar metadatos. Compatible con redistribuir los XLSX/CSV.
- **Curación propia** (`pollsters.quality`, `parties.block`, mapas JSON, `params.json`): obra del autor; recomendable **CC BY 4.0** para datos (el código ya es MIT, `LICENSE`). `Electomania-Rankings.xlsx` es contenido de un tercero: no redistribuir sin permiso; si `quality` deriva de él, documentar la metodología y la fecha.

### Estrategia de publicación propuesta
1. **Separar datos crudos de derivados**: mover `proc_sample`, `rating`, `error_*`, `bias_*`, `weight_*` de `polls` (y `error`, `bias` de `polls_results`) a tablas/CSV `polls_weights`, `polls_errors`. Así `save_model_data` deja de anular columnas de la tabla cruda y el dataset publicado es estable.
2. **Paquete de datos versionado en el repo** (`data/`), formato CSV UTF-8 + `datapackage.json` (Frictionless Table Schema generado desde `Model.meta`) con: `events.csv`, `provinces.csv`, `parties.csv`, `pollsters.csv` (con `quality` y su justificación), `sponsors.csv`, `events_data.csv`, `events_results.csv`, `polls.csv`, `polls_results.csv`, `params.json`, `wp-maps.json`, `wp-urls.json`, `infoelectoral/*.json` y los XLSX originales (permitido con atribución). Parquet opcional para `events_results` (56 k filas: CSV basta).
3. **Snapshot ejecutable**: un `elections.sqlite` construido por script desde los CSV, y un adaptador `SQLite` en `mtpy/core/dal/` (existen MySQL/PostgreSQL) para que `mtpy.run()` funcione sin PostgreSQL; alternativamente `pg_dump --schema-only -n elections` + `COPY` desde CSV.
4. **Procedencia**: en `polls.csv` añadir `source_url`, `source_revision` (`oldid` de Wikipedia) y `retrieved_at`; en el loader, guardar el HTML crudo en `data/raw/wikipedia/<event>/<oldid>.html` y permitir `read_data(from_file=...)`; en Infoelectoral registrar la fecha de descarga.
5. **Publicación con DOI**: release en GitHub sincronizada con Zenodo (o el dataset en Zenodo/HF Datasets), y actualizar `CITATION.cff` con el DOI del dataset y la versión; README de datos con diccionario de columnas, licencias por tabla y changelog.
6. **Pipeline reproducible**: `make data` = cargar XLSX → CSV, parsear snapshots HTML → CSV, construir SQLite; tests con fixtures (un HTML de tabla y un XLSX pequeño) para `parse_dates`, `read_rows` (rowspan), `read_data` de Infoelectoral; `requirements.txt` mínimo (numpy, pandas, scipy, lxml, requests, openpyxl, matplotlib, psycopg2 opcional).

---

## 9. Pipeline (pasos ordenados)

1. `mtpy.run()` — `mtpy/mtpy.py:19-95` — carga `.env`, `config.json`, adaptador BD, `fs` con raíz `files/`.
2. Resultados oficiales — `loader.py:69-178` — XLSX Infoelectoral + JSON de mapeo → `events_data`, `events_results` (`pct = 100·votes/votes_válidos`).
3. Encuestas — `loader.py:553-651` — scraping Wikipedia → `polls`, `polls_results` (`date = end_date`, `days = event_date − end_date`, `seats = floor((min+max)/2)`).
4. Alta append-only y guardado — `loader.py:653-697`.
5. Pesos del evento — `loader.py:699-714` → `Computer.compute_weights`.
6. Parámetros por evento — `data.py:53-139` — auto (`main` = presencia > 95 % en 6 semanas) ⊕ `params.json`.
7. Series anchas para los modelos — `data.py:195-250`, `277-327`, `330-365`.
8. Persistencia de cálculos — `data.py:381-460` — `UPDATE … NULL` + `upsert`; ratings `DELETE` + `upsert` + `pollsters.rating` truncado.

---

## 10. Hallazgos

| # | Sev. | Tipo | Dónde | Descripción / cómo comprobar |
|---|---|---|---|---|
| 1 | alta | bug | `loader.py:519` vs `models/elections.py:128`, `helpers.py:85-89` | El contexto `exit`/`wban` se guarda como `context`; el modelo espera `ctype`; `format_data` lo descarta. `ctype` nunca se persiste: la serie 2027 tiene 0 `ctype` y `Computer` (`drop_ctypes=['wban','exit']`) no puede excluirlos. Test: `WikipediaLoader('es','2023-07-23',years=['2023'],path='.').read_data().build_series()` → 13 `context='wban'` en `loader.data`, `loader.polls.ctype.notna().sum()==0`. |
| 2 | alta | reproducibility | `loader.py:562-566, 598` | Scraping en vivo sin archivar HTML ni `oldid`; el dataset no es reproducible ni citable. |
| 3 | alta | reproducibility | `.gitignore:2`, `files/*` | `params.json`, mapas, CSV de provincias y ficheros Infoelectoral no están versionados. |
| 4 | alta | reproducibility | `models/elections.py` (sin DDL); BD sin FK | No hay DDL, volcado ni seeds de `parties/pollsters/sponsors/events/provinces`; un tercero no puede recrear la BD. |
| 5 | media | bug | `data.py:305-306` | `drop_contexts` filtra por `t.context`, columna inexistente → error SQL. Test: `get_poll_series('es','2023-07-23',drop_contexts=['wban'])`. |
| 6 | media | bug | `loader.py:384-386, 400-401` | Encuestas cuyo encuestador lleva nota (`CIS (Target Point)`, `CIS (Sondaxe)`, …) se descartan en silencio porque el nombre compuesto no está en `pollsters` ni en `wp-maps.json`. Test: `loader.pollsters_missing` tras `read_data()` (2026: 5 etiquetas). |
| 7 | media | design | `loader.py:653-697` | `overwrite=True` borra el evento y recarga sin `ctype`/`notes`/`url`: se pierde la curación manual. |
| 8 | media | reproducibility | `loader.py:143,148`; `lib/utils.py:91` | `groupby(level=0, axis=1)` deprecado (FutureWarning en pandas 2.2.2; eliminado en 3.0). Test: `warnings.simplefilter('error')`. |
| 9 | media | reproducibility | `requirements.txt` | Versiones fijadas (numpy 1.25.2, scipy 1.9.3) distintas de las usadas (2.4.4 / 1.17.0); dependencias ajenas al modelo. |
| 10 | media | design | `models/elections.py:108-148`; `data.py:413-418` | Columnas crudas y calculadas mezcladas en `polls`; `save_model_data` anula columnas de toda la serie antes de reescribir. |
| 11 | baja | bug | `data.py:532-537`; `core/data.py:2221-2238` | `get_next_event_date` sin ORDER BY (`get_var` no añade `sort`): resultado indefinido con >1 evento futuro. |
| 12 | baja | bug | `data.py:154, 181, 264` | `get_event_data/get_event_results/get_poll_results` con `event_dates=None` → `TypeError`. |
| 13 | baja | bug | `data.py:453` | `astype(int)` trunca el rating guardado en `pollsters.rating` (56 discrepancias frente a `pollsters_ratings`). |
| 14 | baja | docs | `loader.py:648`; `models/elections.py:126` | `pub_date` es siempre `end_date` (3825/3825): la columna no representa la publicación real. |
| 15 | baja | design | `loader.py:649` | `mtype` se hereda del encuestador (73 % nulo), no describe la metodología de cada encuesta. |
| 16 | baja | bug | `loader.py:39, 289`; `computer.py:132` | `path` por defecto `os.getcwd()` es incompatible con la raíz `files/` de `app.fs`. Test: `WikipediaLoader('es','2027-08-22')` sin `path` → `FileNotFoundError`. |
| 17 | baja | design | `loader.py:70` | `read_file` usa `app.fspath` local mientras el resto usa `app.fs` (rompe con S3). |
| 18 | baja | bug | `loader.py:600-603` | Sin `years`, sólo se procesa la primera `wikitable` de la página. Test: `WikipediaLoader('es','2019-11-10',path='.').read_data()` y comparar con el nº de tablas de la página. |
| 19 | baja | bug | `loader.py:598` | Deduplicación por `(date,pollster_id,sponsor_id)`: dos encuestas del mismo encuestador/medio con la misma fecha final se pisan. |
| 20 | baja | bug | `data.py:322` | Unión `polls ⋈ results` sin `event_date`; duplicaría filas si una clave se repitiese entre eventos (hoy 0 casos). |
| 21 | baja | design | `data.py:134` | Fusión superficial `smap \| params`: `bmaps` de `params.json` sustituye por completo a los automáticos; `merge_bmaps('max')` puede fallar en eventos sin `max`. |
| 22 | baja | bug | `loader.py:117-121, 141-142` | Candidatura ausente del JSON → columnas desalineadas → `ValueError` críptico en lugar de mensaje claro. |
| 23 | baja | design | `events_data` 2027 (BD) | Reparto de escaños provincial de 2027 copiado a mano de 2023 (verificado idéntico en `seats/votes/registered`); hipótesis no documentada. |
| 24 | baja | docs | `data.py:484` | Docstring `min_polls` "default 1" vs valor real 0; la mayoría de funciones de `data.py` carecen de docstring (AGENTS.md lo exige). |
| 25 | baja | dead-code | `loader.py:57`; `wp-urls.json` clave `results`; `models/elections.py:48-62, 93-105, 175-215`; `models/sources/elections.py` (vacío) usado por `pipelines/events/*.py` | `self.regions`, `es-regions.csv`, URL `results`, tabla `provinces`, `pollsters_parties`, `forecasts*`, `Electomania-Rankings.xlsx`, `es-regions-ages.csv` sin uso; `pipelines/events` importa clases inexistentes. |
| 26 | baja | design | `data.py:153-155, 292-306` | SQL por concatenación de strings (sin parámetros); frágil ante comillas en nombres. |
| 27 | baja | stat | BD `polls.parties` | 166 encuestas con sólo 2 partidos (suma ≈ 52-60 %) conviven con encuestas completas; conviene documentar cómo las trata `Computer`. |

## 11. Preguntas para el autor

1. ¿De dónde sale `pollsters.quality` (escala 0-78, 53 valores en 5)? ¿Deriva de `files/Electomania-Rankings.xlsx`? ¿Con qué regla y fecha?
2. ¿Cómo se rellenaron los 84 `ctype` (`wban`/`exit`) existentes (versión anterior del loader, SQL manual)? ¿Se quiere mantener la detección por color de fila?
3. ¿Deben incluirse las encuestas del CIS con trabajo de campo subcontratado (`CIS (Sondaxe)`, …) como CIS o como el subcontratista?
4. ¿La fecha `2027-08-22` es la fecha legal límite? ¿Se prevé un mecanismo para actualizar el placeholder y `events_data` cuando se convoque?
5. ¿Qué criterio se siguió en las fusiones de candidaturas de los JSON de Infoelectoral (p. ej. `FALANGE → AN18`, `SORIA ¡YA! → EV`)?
6. ¿Se usa en algún flujo `mtpy/pipelines/events/*` o la tabla `provinces`? ¿Se pueden eliminar junto con `forecasts*` y `pollsters_parties`?
7. ¿La BD se ha creado con `Model.create()`? ¿Existe algún volcado que sirva como punto de partida para el paquete de datos?
8. ¿Qué licencia desea para los datos curados (CC BY 4.0 vs CC BY-SA 4.0 por herencia de Wikipedia)?
