# Método 11: ámbitos autonómicos (spec, 28-09-2026)

Estado: **diseño aprobado, pendiente de implementación**. Este documento es la especificación; el plan de
implementación por tareas se escribirá a partir de él. Cuando se implemente, la cabecera pasará al formato
de los métodos anteriores (cambios, tests, impacto numérico).

## Objetivo

Que el modelo trate cualquier proceso autonómico igual que el general:

1. **Recopilar** encuestas y resultados de las 17 comunidades autónomas, con un `scope` por comunidad
   definido en una tabla de ámbitos.
2. **Alimentar el ranking** de casas con los errores cometidos en las autonómicas, con menor peso que los
   nacionales.
3. **Pronosticar y simular** una elección autonómica con el `Forecaster` y el `Simulator` actuales:
   promedio, intervalos, escaños por circunscripción y probabilidades.

## Decisiones tomadas

| Id | Decisión |
|---|---|
| D1 | Códigos de ámbito ISO 3166-2 en minúsculas: `es-md` Madrid, `es-cl` Castilla y León, `es-ct` Cataluña… El ámbito nacional sigue siendo `es` |
| D2 | Un partido conserva su sigla nacional cuando la candidatura es su federación regional (PP, PSOE, VOX, UP, SUMAR, Cs). Las marcas propias son partidos con `parent_id` hacia el nacional cuando lo tienen (PSC, PSE-EE, Más Madrid, Comuns, Compromís) o sin él (CHA, PRC, UPL, CC, BNG…) |
| D3 | Peso inicial de los errores autonómicos en el rating: `rating_weight = 0.5`; el nacional vale 1. Se calibra después con el backtest de ratings |
| D4 | Votos válidos por circunscripción: híbrido. Se reparten por censo al cargar (marcados como estimados) y se sustituyen por la cifra oficial en los eventos donde sea fácil obtenerla |
| D5 | Alcance temporal: elecciones autonómicas desde 2009-2011, que es donde la Wikipedia inglesa tiene tablas de sondeos |
| D6 | La spec vive aquí, en la serie de métodos, no en `docs/superpowers/specs/` |

## Contexto: lo que hay y lo que no

Hallazgos de la exploración (27-09-2026) que condicionan el diseño:

- **Sondeos.** En la Wikipedia inglesa no existen páginas "Opinion polling for the … regional election". Las
  tablas están dentro del artículo de cada elección, `https://en.wikipedia.org/wiki/{año}_{gentilicio}_regional_election`
  y `Next_{gentilicio}_regional_election`, sección *Opinion polls › Voting intention estimates*. El formato
  es el mismo que el nacional: `Polling firm/Commissioner | Fieldwork date | Sample size | Turnout | un th
  por partido (enlace) | Lead`, con las filas de resultado electoral en negrita y fondo `#EFEFEF`.
  Inventario 2009-2026: **3.479 filas** en 17 comunidades (Cataluña 587, Andalucía 326, Madrid 339,
  Galicia 343, País Vasco 289, Valencia 294); Asturias, Cantabria, La Rioja y Murcia sólo tienen tablas
  desde 2023. En algunos artículos pequeños de 2011-2019 la tabla cuelga del `h2` "Opinion polls" sin
  `h3`, y en `Next_Madrilenian` el `th` "Lead" no está en la primera fila de cabecera (SALF ocupa la última
  posición), así que el cargador no puede asumir "Lead" por posición.
- **Resultados.** No están en Infoelectoral (competencia de cada comunidad). El mismo artículo trae
  *Results › Overall* (votos, %, escaños de cada candidatura; votos válidos, nulos, emitidos, abstención y
  censo del conjunto) y *Results › Distribution by constituency* (% y escaños por circunscripción, **sin
  votos**; cabecera de tres filas con `colspan=2` por partido y subcolumnas `%`/`S`; celdas vacías con
  `rowspan`/`colspan` para candidaturas que no concurren). Archivo Histórico Electoral de la Generalitat
  (Argos, `http://www.argos.gva.es/ahe/`, sólo http) tiene resultados por provincia de todas las
  comunidades, pero el servidor devolvía *Service Unavailable* durante la exploración: no puede ser la
  fuente primaria.
- **Circunscripciones.** No siempre son provincias: Asturias tiene tres zonas (Central, Occidental,
  Oriental), Canarias siete islas más una lista autonómica desde 2019, Baleares cuatro islas. Madrid,
  Murcia (desde 2019), Navarra, La Rioja y Cantabria son distrito único.
- **Umbral legal.** Varía por comunidad y por reforma (3 % en Cataluña, Andalucía o Castilla y León; 5 % en
  Madrid, Galicia o Extremadura; Canarias combina un umbral autonómico con uno insular). El `Simulator`
  tiene hoy el 3 % de la LOREG fijo.
- **Códigos.** `data/es-regions.csv` y `es-regions-ages.csv` usan la numeración de comunidades del
  Ministerio del Interior (07 Castilla-La Mancha, 08 Castilla y León, 12 Madrid…) mientras
  `es-provinces.csv.reg_code` usa la del INE (07 Castilla y León, 08 Castilla-La Mancha, 13 Madrid…).
  La tabla `elections.provinces` (`ncode`, `scode`, `seats`) no la usa ningún módulo de `mtpy/lib`.

## Enfoque

Se descartan dos alternativas. **A**, un `Computer` por ámbito con ratings separados, es barato pero los
errores autonómicos nunca llegan al ranking nacional. **C**, un modelo jerárquico conjunto de errores por
casa y ámbito, exige rehacer el `Computer` y no hay datos para estimarlo bien todavía.

Se adopta **B**: la capa de datos, los pesos, los errores y las desviaciones siguen siendo **por ámbito y
por evento**, exactamente como hoy. Sólo el paso de ratings se hace **transversal**: para cada evento reúne
los sondeos de todos los ámbitos con elección anterior a la suya y multiplica el peso de cada sondeo por el
`rating_weight` de su ámbito. Cambia un método (`compute_ratings`) y las consultas que lo alimentan.

## Diseño

### 1. Catálogos

**Tabla `scopes`** (`mtpy/models/elections.py`, `Scopes`, clave `scode`), versionada en `data/es-scopes.csv`
y cargada con el mismo mecanismo que el resto de tablas manuales:

| Columna | Tipo | Contenido |
|---|---|---|
| `scode` | cat | `es`, `es-an`, … (ISO 3166-2 en minúsculas) |
| `name` | str | Nombre de la comunidad |
| `ine_code` | int | Código INE de la comunidad (0 para `es`) |
| `parent` | cat | Ámbito padre (`es` para las comunidades, nulo para `es`) |
| `demonym` | str | Gentilicio inglés que usa Wikipedia en los títulos (`Madrilenian`, `Castilian-Leonese`…) |
| `threshold` | num | Umbral legal vigente, % de votos válidos de la circunscripción |
| `threshold_scope` | num | Umbral alternativo sobre el conjunto de la comunidad (Canarias); nulo si no aplica |
| `rating_weight` | num | Peso de sus errores en el rating (1 nacional, 0,5 autonómicas) |
| `seats` | int | Escaños del parlamento actual (informativo; los de cada evento van en `events_data`) |

Catálogo inicial (los umbrales son los vigentes en 2026 según la ley de cada comunidad y **deben
verificarse ley por ley en la fase R0**; los históricos que cambiaron se registran como override por evento
en `params.json`, ver §4):

| `scode` | Comunidad | INE | Gentilicio | Circunscripciones | Umbral |
|---|---|---|---|---|---|
| es-an | Andalucía | 01 | Andalusian | 8 provincias | 3 |
| es-ar | Aragón | 02 | Aragonese | 3 provincias | 3 |
| es-as | Asturias | 03 | Asturian | 3 zonas | 3 |
| es-ib | Baleares | 04 | Balearic | 4 islas | 5 |
| es-cn | Canarias | 05 | Canarian | 7 islas + lista autonómica (desde 2019) | 15 insular ó 4 autonómico (6/30 antes de 2019) |
| es-cb | Cantabria | 06 | Cantabrian | 1 | 5 |
| es-cl | Castilla y León | 07 | Castilian-Leonese | 9 provincias | 3 |
| es-cm | Castilla-La Mancha | 08 | Castilian-Manchegan | 5 provincias | 3 |
| es-ct | Cataluña | 09 | Catalan | 4 provincias | 3 |
| es-vc | Comunidad Valenciana | 10 | Valencian | 3 provincias | 5 autonómico (verificar la reforma de 2022) |
| es-ex | Extremadura | 11 | Extremaduran | 2 provincias | 5 autonómico |
| es-ga | Galicia | 12 | Galician | 4 provincias | 5 |
| es-md | Madrid | 13 | Madrilenian | 1 | 5 |
| es-mc | Murcia | 14 | Murcian | 1 (5 hasta 2015) | 3 (5 hasta 2015) |
| es-nc | Navarra | 15 | Navarrese | 1 | 3 |
| es-pv | País Vasco | 16 | Basque | 3 provincias | 3 |
| es-ri | La Rioja | 17 | Riojan | 1 | 5 |

**Tabla `districts`** sustituye a `provinces` (que se elimina del modelo; nadie la lee). Clave
`(scope, region_id)`: `name`, `slug`, `ine_code` (código INE de la provincia cuando la circunscripción es
una provincia; nulo si no), `group` (comunidad INE a la que pertenece, para el método 9). Convención de
`region_id`:

- `0` es siempre el total del ámbito (como hoy el total nacional).
- Las provincias conservan su **código INE** (1-52) en cualquier ámbito, de modo que `es-provinces.csv`
  y las reglas `regions` del `smap` siguen valiendo.
- Las circunscripciones no provinciales reciben códigos **a partir de 100**, únicos dentro del ámbito:
  zonas de Asturias 101-103, islas de Baleares 101-104, islas de Canarias 101-107 y lista autonómica
  canaria 100. Las cinco circunscripciones de Murcia hasta 2015 usan 101-105.

`data/es-districts.csv` recoge las circunscripciones de los 17 ámbitos con su población, para el reparto
por censo de D4. `es-regions.csv` y `es-regions-ages.csv` pasan a códigos INE (una sola numeración en
`data/`); `es-provinces.csv` no cambia.

### 2. Parámetros y URLs por ámbito

- `data/params.json` ya está indexado por ámbito (`{"es": {...}}`); se añaden las entradas de cada
  `es-*` que las necesite. `get_event_params` deja de fallar cuando el ámbito no tiene entrada
  (`params.get(scope, {})`).
- `data/wikipedia/wp-urls.json` se indexa igual: `{"es": {...}, "es-md": {"2023-05-28": {"results": url,
  "polls": url}, "2027-05-30": {...}}}`. Una única URL sirve para sondeos y resultados en las autonómicas,
  porque están en el mismo artículo; la fecha del próximo evento es el límite legal de la legislatura,
  como en `es`.
- `data/wikipedia/wp-maps.json` gana un nivel opcional por ámbito:
  `{"parties": {...}, "pollsters": {...}, "sponsors": {...}, "scopes": {"es-md": {"parties": {"PP":
  ["People%27s_Party_of_the_Community_of_Madrid"], "MM": ["M%C3%A1s_Madrid"]}}}}`. El cargador resuelve
  primero el alias del ámbito y después el global. Con ello el enlace regional del PP se mapea a `PP`
  (D2) y el de Más Madrid a su propio partido.
- Umbrales históricos: `params.json` admite `"threshold"` y `"threshold_scope"` por evento, que
  prevalecen sobre `scopes` (Murcia hasta 2015, Canarias hasta 2019, Valencia según la verificación).

### 3. Ingesta

**`WikipediaLoader`** (sondeos), cambios mínimos:

- Selección de la tabla: la primera `wikitable` cuyo encabezado más cercano (h2, h3 o h4) contenga
  "Voting intention"; si no hay ninguna, la primera `wikitable` bajo el `h2` "Opinion polls" cuya primera
  fila empiece por "Polling firm". Hoy sólo mira `h3/h4/h5` en el `div` hermano anterior.
- Columnas: los partidos son los `th` de la primera fila con enlace a un artículo que no sea `File:`;
  "Lead" se detecta por texto (o por ausencia de enlace en el último `th`), no por posición `[4:-1]`.
- Alias por ámbito (§2). El resto (rowspan, fechas, tamaño de muestra, contexto `exit`/`wban`, `exclude`)
  no cambia.

**`WikipediaResultsLoader`** (nuevo, `mtpy/lib/loader.py`), misma interfaz que `InfoElectoralLoader`
(`read_data → build_series → save_totals/save_results`, `parties_missing`, `show_summary`):

- Lee *Results › Overall*: por candidatura, votos, % y escaños; del pie, votos válidos, nulos, emitidos,
  abstención y censo. Las candidaturas se mapean a partidos con los alias del ámbito y los globales, con
  el nombre entre paréntesis de la celda ("People's Party (PP)") como clave alternativa; las no mapeadas
  suman a `'-'` (id 0), como en Infoelectoral. Con ello rellena `events_data` y `events_results` de
  `region_id = 0`, exactos.
- Lee *Results › Distribution by constituency*: reutiliza la lógica de `rowspan`/`colspan` de
  `WikipediaLoader.read_rows`; cada fila da `pct` y `seats` por partido y circunscripción. El nombre de
  la circunscripción se resuelve contra `districts` por nombre o alias (`Biscay` → Vizcaya, `Gipuzkoa` →
  Guipúzcoa, `Girona` → Gerona; los alias van en `es-districts.csv`).
- Votos por circunscripción (D4): `votes_i = votes_total · census_i / Σ census` con el censo de
  `es-districts.csv` (o el censo por provincia del INE cuando la circunscripción es una provincia);
  `votes` de cada partido en la circunscripción = `pct · votes_i / 100`. `events_data` gana una columna
  `estimated` (bin) que marca esas filas. Si existe `data/results/{scope}/{fecha}.csv` con votos válidos
  oficiales por circunscripción, se usan y `estimated = False`. El artículo en Wikipedia de la comunidad
  y Argos son las fuentes para curar esos CSV.
- Distrito único: sólo la fila `region_id = 0` y una fila de circunscripción idéntica con `region_id`
  = código INE de la provincia (Madrid 28, Murcia 30, Navarra 31, La Rioja 26, Cantabria 39), para que
  el `Simulator` tenga al menos una unidad de reparto.
- Lista autonómica canaria: circunscripción `100` cuyo `pct` es el del total; los escaños los da la
  tabla si los publica y, si no, la diferencia entre el total y la suma insular.
- Evento próximo: crea la fila de `events`, y las de `events_data` con los escaños de cada circunscripción
  del artículo `Next_…` (tabla *Electoral system*) o de `es-districts.csv`, sin votos.

**Carga por lotes**: `load/run_load.py --scopes es-md es-cl … --what polls results --since 2009` recorre
`wp-urls.json`, ejecuta ambos cargadores por evento e imprime por ámbito los partidos, casas y
patrocinadores sin mapear, para completar `wp-maps.json` y `parties` en dos o tres pasadas. Las doce
autonómicas del 28-05-2023 se cargan en la misma pasada. `computed`/`featured`: un evento autonómico es
`featured` cuando tiene resultado oficial y al menos 10 sondeos útiles.

### 4. Computer

- `build_series`, `compute_weights`, `compute_errors`, `compute_deviations`, `get_house_effects_data`,
  `get_drift_data` y `get_swing_residuals` **no cambian**: ya trabajan por ámbito y evento, y los bloques
  (`main`, `vs`, `blocks`) salen de `get_event_params` con los partidos presentes en cada ámbito.
- **`compute_ratings(save, scopes=None)`** se hace transversal. `scopes=None` conserva el comportamiento
  actual (sólo el ámbito del `Computer`); `scopes='all'` toma de `scopes` los ámbitos con
  `rating_weight > 0`. Algoritmo:
  1. Carga los sondeos con `bias` calculado de todos los ámbitos seleccionados (`get_poll_series` por
     ámbito, concatenados con un nivel `event_scope` en el índice; `self.keys` pasa a incluirlo).
  2. Para cada evento `(fecha, ámbito)` de los ámbitos del `Computer` (y el próximo de cada uno), toma
     los sondeos de los eventos con **fecha estrictamente anterior**, de cualquier ámbito. Las doce
     autonómicas de un mismo día no se ven entre sí; la general de julio de 2023 sí ve las de mayo.
  3. `poll_rating_weights` agrupa por `(event_date, event_scope, pollster_id)` para `weight_pos`, por
     `(event_date, event_scope)` para `weight_week`, y calcula `weight_year` con el año máximo del
     conjunto. Se añade `weight_scope = rating_weight[event_scope]` al producto que forma `weight`.
  4. `pollster_ratings` no cambia. `num_events` cuenta eventos de cualquier ámbito.
  5. Guarda `pollsters_ratings` con el `event_scope` de cada evento y `polls.rating/weight_rating` de
     los sondeos de ese evento; `pollsters.rating` guarda el último rating del ámbito `es`.
- Invariante de regresión: con `rating_weight = 0` en todos los `es-*`, los ratings de `es` coinciden con
  los actuales (test de integración).
- Estimadores de error, escaños y deriva: se ajustan con los eventos del ámbito cuando tiene al menos
  **3 eventos featured**; si no, con los del ámbito padre (`es`). La variable `regional` de los
  estimadores vale 0 para todos los partidos en los ámbitos autonómicos (dentro de una comunidad no hay
  partidos "de una parte del territorio" en el sentido que usa el estimador). El estimador de escaños
  pasa a regresar la **cuota de escaños** (`seats / seats_total` del evento) y a multiplicar por los
  escaños del ámbito al predecir, para que sea comparable entre parlamentos de 33 y 350 escaños.
- Efectos de casa (M6): prior del propio ámbito; con menos de 3 eventos el prior se anula y sólo actúa
  el efecto del ciclo, como ya prevé `Forecaster.fit_house_effects`. Prior compartido por `parent_id`
  queda como mejora posterior.

### 5. Forecaster y Simulator

- Ambos reciben `scope` como hoy; `get_event_dates`, `get_event_series`, `get_poll_series` y
  `get_event_params` filtran por él, así que el promedio, la deriva (M5), la edad del partido (M8) y los
  intervalos con cluster (M10) funcionan sin cambios.
- `Simulator.threshold` por defecto pasa a `None` = "el del ámbito": lee `threshold` y `threshold_scope`
  de `scopes` con el override por evento de `params.json`. `alloc_seats` acepta el umbral alternativo:
  una candidatura participa si supera el umbral de la circunscripción **o** el autonómico (Canarias).
- Circunscripciones: `get_reg_totals` y `get_prev_results` ya se indexan por `region_id`; con
  `events_data` del ámbito el `Simulator` reparte por sus circunscripciones sin cambios. `region_names`
  sale de `districts`.
- Ruido del método 9: en los ámbitos autonómicos el componente por comunidad no existe (todas las
  circunscripciones son de la misma). `Simulator` usa `region_groups` de `districts.group`; cuando el
  ámbito no es `es`, `_add_swing_noise` aplica sólo el choque por circunscripción, con la curva
  provincial estimada en `es` (`SwingNoise.sigma_province`) mientras el ámbito no tenga 3 pares de
  elecciones propias.
- `smap` por evento y ámbito en `params.json`, como hoy (herencias de Cs → PP, UP → SUMAR, etc.).

### 6. Backtest, notebooks y documentación

- `run_backtest(scope=...)` ya está parametrizado; `backtest/run_backtest.py --scope es-md` y un bucle
  `--scopes all` escriben en `backtest/results/{scope}/`. Elecciones evaluables: las autonómicas con
  resultado oficial y al menos 5 sondeos útiles a cada horizonte (2019-2026, sobre todo las doce de
  2023 y Cataluña, Galicia y País Vasco 2024).
- Notebooks `data-load/*` parametrizados por `scope` (una celda); `PollsForecast` y `PollsSimulations`
  igual. Nuevo `notebooks/events/es-md-202305/` como ejemplo autonómico.
- `data/README.md` documenta `es-scopes.csv`, `es-districts.csv` y `results/`; `backtest/README.md`
  los ámbitos; este documento pasa a describir los cambios hechos y su impacto.

## Fases

Orden por dependencias. Cada fase cierra con sus tests en verde y un commit.

| Fase | Contenido | Tests y criterio de cierre |
|---|---|---|
| R0 | `Scopes` y `Districts` en el modelo y en `data/`; `es-regions*.csv` a códigos INE; `params.json`, `wp-urls.json` y `wp-maps.json` por ámbito; `get_event_params` tolerante; eliminación de `Provinces` | `test_data_files`: códigos ISO válidos, `region_id` únicos por ámbito, todo ámbito de `wp-urls` existe en `scopes`, umbrales en (0, 100]; suite actual en verde |
| R1 | `WikipediaLoader` generalizado; alias por ámbito; carga de sondeos de los 17 ámbitos 2009-2026 | Unit: detección de "Lead" y de la tabla con fixtures HTML (Madrid 2023, Next Madrid, CyL 2022, Asturias 2019). Integración: recuento por ámbito ≥ 95 % del inventario (3.479 filas) y cero casas sin mapear entre las que tienen ≥ 3 sondeos |
| R2 | `WikipediaResultsLoader`; `events_data.estimated`; eventos próximos; `load/run_load.py` | Unit: parseo de Overall y de la tabla por circunscripción con fixtures (CyL 2022, Canarias 2023, Asturias 2023). Integración: por evento, `Σ votes` de partidos = válidos − blancos ± 0,1 % y `Σ seats` por circunscripción = escaños del parlamento |
| R3 | `compute_ratings` transversal con `weight_scope` | Unit: con sondeos sintéticos, peso 0 reproduce el rating de un solo ámbito y peso 1 iguala a mezclar los eventos. Integración: ratings de `es` con `rating_weight = 0` idénticos a los actuales; con 0,5, informe de las 15 casas cuyo rating más cambia |
| R4 | Umbral por ámbito y alternativo, `districts` en `Simulator`, ruido M9 sólo provincial, estimadores con caída al padre, cuota de escaños | Unit: `alloc_seats` con umbral doble; lookup de umbral con override. Integración: `Simulator('es-md', '2023-05-28', drange=6)` y `('es-cl', '2022-02-13')` en modo determinista reproducen los escaños reales con error absoluto medio ≤ 1,5 por partido principal |
| R5 | Backtest de las autonómicas 2019-2026, notebooks por ámbito, `data/README.md`, este documento | `by_horizon.csv` por ámbito; coberturas del 50/80/95 % en cuotas dentro de ±0,15 del nominal a 6 días en los ámbitos con ≥ 3 eventos |

## Riesgos y salvedades

- **Formato de Wikipedia.** Las tablas antiguas (2009-2015) de comunidades pequeñas tienen variantes
  (sin columna Turnout, sin `h3`); R1 las acomoda con fixtures y acepta perder algunas filas, dentro del
  criterio del 95 %.
- **Identidad de los partidos.** D2 exige curar alias por ámbito y crear partidos con `parent_id`. Es
  trabajo manual en `wp-maps.json` y `parties`; el cargador por lotes lo hace iterativo listando lo que
  falta.
- **Votos por circunscripción estimados.** No afectan al reparto de escaños ni al umbral (ambos operan
  con porcentajes dentro de la circunscripción), sólo al peso relativo entre circunscripciones al
  agregar y al voto en blanco. El flag `estimated` permite medir su efecto y sustituirlos.
- **Ratings.** El peso 0,5 es un prior; el `Computer` pierde la propiedad de que un evento sólo ve sondeos
  de su ámbito, y hay que comprobar que el rating de casas puramente regionales (Sondaxe, GESOP,
  Ikertalde) no se dispara ni se hunde por tener pocos eventos.
- **Rendimiento.** `compute_ratings` sobre ~7.300 sondeos y ~100 eventos sigue en el rango de un minuto;
  `get_house_effects_data` y `get_drift_data` se ejecutan por ámbito y crecen linealmente.
- **Argos e Infoelectoral** quedan como fuentes de verificación, no de carga.

## Fuera de alcance (mejoras posteriores)

Prior de efectos de casa compartido por `parent_id` entre ámbitos; transferencia de las encuestas
autonómicas al pronóstico provincial de las generales (swing por comunidad informado por sondeos
autonómicos); calibración de `rating_weight` por comunidad; elecciones municipales y europeas.
