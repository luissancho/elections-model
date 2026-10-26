# Job `load`: la carga de sondeos de Wikipedia como un job

Fecha: 2026-10-26. Estado: aprobado por Luis el 2026-10-26; construido el 2026-10-26 (plan docs/superpowers/plans/2026-10-26-job-load.md)

Comprobación: dry run contra la base del `.env` (`python job.py load '{"scopes":["es"],"dry_run":true,"verbose":2}'`,
commit f3adccd): `es: polls checked (521 read, 0 new, 2026-11-29, dry run) in 2 s`, con los tres conjuntos
`Missing` vacíos; `Skipped pollsters` = las «CIS (…)» (Ateneo del Dato, Demócrata, Jaime Miquel, JM, Logoslab,
Opina 360, Sondaxe, SocioMétrica, Target Point), `Sumar` y `PSOE`; `Skipped sponsors` = `{'PP'}`.

Pendiente de Luis: la primera carga real (`python job.py load '{"scopes":["es"]}'`) y, con `force`, revisar el `mtype` de las casas nuevas.

Desviaciones de la construcción: `skip.parties` se comprueba después de los alias (así `PP+Cs` sigue mapeado a PP
en `es-pv` y se silencia donde no hay alias), y `load/run_load.py` no se ha tocado.

Enmiendas de la revisión final (2026-10-26): `years` con `overwrite` se rechaza; con `"all"`, un ámbito sin
próximo evento o sin artículo de sondeos termina `skipped` (no `failed`); un fallo de `parties_checks` es un aviso;
una casa o un medio desconocidos sólo se anotan como sin mapear si la fila es un sondeo (fechas y `Lead` numérico).

Dry run de todos los ámbitos tras la ola de correcciones (2026-10-26, commit 7dd320a, sólo lectura):
`es` y nueve autonómicos `checked` (es-vc con 4 sondeos nuevos); `es-an`, `es-ar`, `es-cl` y `es-ex` `skipped`
(su próximo evento no tiene artículo de sondeos); cinco `failed` por nombres sin mapear aparecidos en
Wikipedia desde la última carga autonómica: es-cn (partidos `Drago_Canarias`, `Municipalists_Canaries_First`;
casas «Areópago Genuina 26», «GCEMOP»; medio «Atlántico Hoy»), es-cm (casa «The Place Public Affairs», medio
«Diario Lanza»), es-ct (casa y medio «The Objective»), es-mc (medio «La 7»), es-nc (medios «CoCiudadana»,
«EiTB»). Las casas y los medios entran con `force`; los dos partidos canarios piden alias o `skip.parties`.

## Intención

Convertir el cuaderno `notebooks/data-load/PollsLoadWikipedia.ipynb` en un job de `mtpy/jobs`, al estilo de
`compute` y `publish`: `python job.py load '{"scopes":["es"],"overwrite":true}'` desde la línea de comandos y
`RUN_JOB='load {...}'` en Docker, con cabecera, línea de resultado por ámbito, resumen final, avisos y fallos
aislados por ámbito. La carga deja de depender de abrir un cuaderno y de editar a mano `event_date` y
`overwrite`.

## Por qué ahora

Hoy la secuencia de Luis tras un cambio de sondeos es: cuaderno con `overwrite` → `compute` → `publish`. El
cuaderno es el único paso manual y el único que no corre en Docker. Esta mañana la recarga del ciclo se hizo
copiando sus celdas a un script.

## Objetivos

1. Un job `load` con la secuencia del cuaderno: leer los artículos de `wp-urls.json`, construir las series
   (con las correcciones CSV), seleccionar lo nuevo o todo el ciclo, guardar sondeos y resultados y recalcular
   los pesos del evento.
2. Avisar de partidos, casas y medios sin mapear, con el mismo bloque de texto del cuaderno, y **parar el
   ámbito antes de guardar nada** si hay alguno; con `force` dar de alta en la base las casas y los medios
   nuevos y seguir.
3. Callar (no avisar, no dar de alta) a las casas descartadas: las de la lista `skip` de `wp-maps.json`, las
   que se llaman como un partido y las «CIS (…)».
4. Reutilizar el andamiaje de `compute` (reporter, verbosidad 0-3, `dry_run`, resumen, `alert`, tests con
   fábricas falsas) y dejar la lógica en `mtpy/lib`, como manda el reparto del framework.

## No objetivos

- Cargar resultados de elecciones (`ResultsLoadWikipedia`, `ResultsLoadInfoElectoral`), reconstruir el
  inventario de artículos (`urls`) ni el CERA: siguen en sus cuadernos y en `load/run_load.py`. El job nace con
  una sola acción, `polls`, y el nombre de las demás queda reservado.
- Dar de alta partidos automáticamente: `parties_missing` trae la clave del artículo de Wikipedia y un partido
  necesita siglas, color, bloque y a veces `parent_id`; un partido nuevo para siempre, con o sin `force`.
- Mapear las «CIS (Sondaxe)», «CIS (Demócrata)», etc. al CIS: son estimaciones de terceros sobre los datos
  brutos del CIS y duplicarían el barómetro; se descartan como hasta ahora, pero sin ruido.
- Gestionar la caché de artículos de `files/stage/wikipedia/` (eso es de `run_load.py`).
- Tocar el cuaderno: se queda como herramienta exploratoria.

## Lo que sería un fracaso

- El job guarda sondeos de un ciclo con una casa o un medio sin mapear sin haber avisado (hoy el cuaderno
  descarta el sondeo en silencio y sigue).
- `force` da de alta como casa a «Sumar», «PP» o una «CIS (…)».
- Una recarga con `overwrite` que falla a medias deja el ciclo borrado: el borrado y la reinserción de
  `save_polls`/`save_results` son los de hoy; el job debe haber resuelto todos los nombres **antes** de
  llamar a ninguno de los dos.
- El job cambia la secuencia de cálculo: los pesos del evento se recalculan como en el cuaderno
  (`compute_series`), ni más ni menos; el `compute` completo sigue siendo su job.

## Restricciones

- Estructura de los jobs: `mtpy/jobs/<Nombre>.py` con `class Nombre(ScopeReporter, Job)` y `run(**kwargs)`,
  lógica en `mtpy/lib/<nombre>.py` con funciones puras de ámbito que reciben fábricas inyectables (ver
  `compute.compute_scope` y `jobs/Compute.py`). `job.py` convierte `load` en `Load`.
- `WikipediaLoader` (`mtpy/lib/loader.py`) hace el trabajo: `__init__(scope, event_date, years, exclude,
  verbose, path)`, `read_data`, `build_series`, `select_series(overwrite)`, `save_polls(overwrite)`,
  `save_results(overwrite)`, `parties_checks`, `compute_series(save, overwrite)`; listas `parties_missing`,
  `pollsters_missing`, `sponsors_missing`, `fixes_missing`. `read_cols` descarta ya el sondeo de una casa sin
  id y el de las casas de `skip`, y deja a `sponsor_id` 0 un medio sin id.
- Catálogos: `elections.pollsters (id, name, mtype, quality, rating)` y `elections.sponsors (id, name)`,
  modelos `Pollsters` y `Sponsors` con `autokey` (id `GENERATED BY DEFAULT AS IDENTITY`). `pollsters.quality`
  la deriva `save_catalogues` de la calidad previa; una casa sin evaluar lleva el valor «sin evaluar» de la
  rúbrica.
- La salida por consola sigue los cuatro niveles de `compute`/`publish` (`progress.py`).
- Nombres de código en inglés neutro; textos públicos en español. Sin scripts sueltos nuevos.

## Alternativas consideradas

1. **Job nuevo `load`** con la lógica en `mtpy/lib/load.py`, calcado de `compute`. Separa la red y el
   parseo de Wikipedia del cálculo y reutiliza todo el andamiaje. **Elegida por Luis.**
2. Acción `load` dentro de `compute` (`what: ["load","compute","ratings"]`): una sola orden, pero mete
   `overwrite`, `force` y los avisos de Wikipedia en un job que hoy es puro cálculo y hace crecer sus tests y
   su docstring. Descartada.
3. Secuencia con `SequenceJob` (`Update`, `Pipeline`): pensada para los pipelines de `ancla`/`meteo`, sin
   reporter ni resumen por ámbito. Descartada.

## Decisiones

### 1. Interfaz del job

```
python job.py load '{"scopes":["es"],"overwrite":true}'
python job.py load '{"scopes":["es"],"force":true}'
python job.py load '{"scopes":["es"],"event_date":"2023-07-23","overwrite":true}'
python job.py load '{"scopes":"all","dry_run":true,"verbose":2}'
```

| Parámetro | Por defecto | Significado |
| --- | --- | --- |
| `what` | `["polls"]` | Acciones; hoy sólo `polls`. Una desconocida falla con `ValueError` antes de empezar |
| `scopes` | `["es"]` | Ámbitos, o `"all"` (`publish.resolve_scopes`) |
| `event_date` | próximo evento de cada ámbito | `publish.next_event_date(scope)`; también una fecha ISO para cargar un ciclo pasado. Un ámbito sin próximo evento, o cuyo evento no tiene `polls` en `wp-urls.json`, no lee nada y termina `skipped` si se pidió `"all"` (como los ámbitos sin sondeos suficientes de `publish forecast`) o `failed` si se nombró en `scopes` |
| `overwrite` | `false` | `false`: sólo los sondeos que no están en la base (`select_series`); `true`: borra el ciclo y lo reinserta entero |
| `force` | `false` | Dar de alta las casas y los medios nuevos y seguir, en vez de parar el ámbito |
| `years` | `null` | Lista de años (`"2026"`; los números se convierten) que se pasa al loader (como en el cuaderno). Con `overwrite` falla con `ValueError` antes de empezar: el borrado del ciclo se llevaría los demás años |
| `dry_run` | `false` | Leer, resolver nombres e informar sin escribir nada (ni altas, ni sondeos, ni pesos) |
| `verbose` | `1` | Niveles abajo |
| `loader`, `computer` | reales | Fábricas para los tests: `loader(scope, event_date, years=None, exclude=None, verbose=0, path=None)` y `Computer(scope=..., event_dates=..., verbose=..., path=...)` (ésta ya la usa el loader; la del job sirve para sustituir `compute_series`) |

`exclude` no se expone: nadie lo usa. `path` tampoco: es `'.'`.

### 2. Secuencia por ámbito (`load.load_polls`)

Función `load_polls(scope, event_date, overwrite=False, force=False, years=None, save=True, verbose=0,
progress=None, loader=None, computer=None) -> dict` en `mtpy/lib/load.py`, con fases que `progress` recibe con
sus segundos:

1. `read`: `loader.read_data()`. Si no hay sondeos, el ámbito termina `loaded` con 0 leídos (no es un fallo:
   un ciclo recién abierto).
2. `names`: clasificar los nombres sin mapear (sección 3). Imprimir el bloque del cuaderno (nivel 1):

   ```
   # Polls: 520

   Missing Parties: set()
   Missing Pollsters: {'Sumar'}
   Missing Sponsors: set()
   ```

   con los descartados ya quitados de los conjuntos. Si queda un partido (ver la cuestión abierta de la
   sección 3), o queda una casa o un medio y no hay `force`, el ámbito termina `failed` aquí, antes de
   `build_series`, con la razón `unmapped parties: X; unmapped pollsters: Y, Z; unmapped sponsors: W`
   (sólo las clases con nombres). Con `force`, las casas y los medios se dan de alta (sección 4) y la
   lectura se resuelve de nuevo; si además queda un partido, no se da de alta nada antes de fallar.
3. `series`: `build_series()`; `fixes_missing` se convierte en aviso
   (`Poll fixes without a poll read: …`) como los avisos del modelo en `publish`.
4. `save` (sólo con `save`): `select_series(overwrite)`, `save_polls(overwrite)`, `save_results(overwrite)`.
   Sin `overwrite` y sin sondeos nuevos, nada se escribe y la línea lo dice (`0 saved`).
5. `weights` (sólo con `save` y si se guardó algo o hay `overwrite`): `compute_series(save=True,
   overwrite=True)`, los pesos del evento, como el cuaderno.

`parties_checks` (la tabla del cuaderno) se imprime al nivel 2, después de guardar; si su consulta falla, es un
aviso (`parties checks failed: …`) y los pesos se calculan igual.

Resultado: `{'status': 'loaded' | 'checked' | 'failed', 'action': 'polls', 'event_date', 'n_read',
'n_saved', 'overwrite', 'missing': {'parties': [...], 'pollsters': [...], 'sponsors': [...]}, 'added':
{'pollsters': [...], 'sponsors': [...]}, 'seconds': {fase: s, 'total': s}, 'warnings': [...]}`;
`checked` es el estado de un `dry_run` que llega al final. Un fallo lleva `reason` (`TipoDeError: mensaje`,
o el texto de los nombres sin mapear).

### 3. Nombres descartados y nombres sin mapear

Regla única, en el loader, para que valga también en el cuaderno y en `run_load.py`. **Sólo afecta a los
nombres que no están en la base ni en los alias de `wp-maps.json`**: un nombre con id conserva su id y su
sondeo, como hoy (la casa «IMC», con 8 sondeos de `es-an`, comparte nombre con un partido y debe seguir
igual). Entre los que no tienen id, una casa se descarta (su sondeo no entra, sin aviso, nunca se da de alta)
cuando

- está en `skip.pollsters` de `wp-maps.json` (como hoy: esta lista sigue yendo antes del idmap, es decir,
  descarta aunque la casa exista), o
- su nombre coincide, sin distinguir mayúsculas ni acentos, con el `name` o el `fullname` de un partido de
  `elections.parties`, o
- encaja en `^CIS \(.+\)$`.

Un medio sin id se descarta (el sondeo entra con `sponsor_id` 0, sin aviso, nunca se da de alta) cuando
está en `skip.sponsors` o coincide con el nombre de un partido (encargos de partidos: D7). La precedencia de
hoy para los medios no cambia: si el nombre está en `elections.sponsors`, el sondeo toma su id.

`read_cols` deja de añadir los descartados a `pollsters_missing`/`sponsors_missing` y los anota en dos listas
nuevas, `pollsters_skipped` y `sponsors_skipped`, que el job imprime al nivel 2 (`Skipped pollsters: {...}`)
para poder auditar la regla.

Lo que queda en `*_missing` es lo que no está en la base ni en los alias ni en las reglas de descarte: eso es
lo que se avisa y, salvo `force`, lo que para.

**Partidos sin mapear: lista `skip.parties` (decidido por Luis el 2026-10-26).** Hoy una columna de
partido sin alias se descarta en silencio y el sondeo entra sin ella; según `files/stage/load_report.csv`,
27 de los 95 eventos autonómicos tienen columnas así a propósito (partidos menores sin ficha:
`Somos_Región`, `Por_La_Rioja`, `Navarre_Platform`…), entre ellos el próximo evento de `es-mc`. Para que la
parada por partidos valga en todos los ámbitos, `wp-maps.json` gana `skip.parties`, una lista global de
claves de artículo (lo que `parties_missing` trae hoy) cuya columna se ignora sin aviso; `read_cols` las
anota en `parties_skipped` y el job las imprime al nivel 2 como las casas. La lista se cura de una vez con
los nombres de `parties_missing` del informe (los 27 eventos), y a partir de ahí un partido nuevo que no
esté en ella **para el ámbito**, con o sin `force`. El `exclude` del loader queda como está (actúa en
`build_series`, después de la lectura) y no sustituye a la lista.

### 4. `force`: altas

Con `force`, antes de `build_series`:

- cada casa de `pollsters_missing` se inserta en `elections.pollsters` con `name` tal cual viene de
  Wikipedia, `mtype` nulo, `quality` el valor «sin evaluar» de la rúbrica (el mismo que `save_catalogues`
  daría a una casa sin evaluación) y `rating` nulo;
- cada medio de `sponsors_missing` se inserta en `elections.sponsors` con su `name`.

Los ids los pone la base (`IDENTITY`). Después el loader recarga sus catálogos e idmaps y vuelve a resolver
las filas leídas **sin descargar otra vez el artículo** (los datos crudos ya están en memoria o en la caché
del loader; cómo se reaprovecha queda al constructor). La línea de resultado cuenta las altas
(`· 2 pollsters added, 1 sponsor added`) y el resultado las lista en `added`. Con `dry_run`, `force` no
inserta nada: el ámbito termina `checked` y la línea dice lo que se habría dado de alta.

Una casa nueva nace sin `mtype`: si es un agregador o un panel online hay que marcarlo a mano después, porque
`DROP_MTYPES` la excluiría del promedio. El job lo recuerda con un aviso por cada alta
(`new pollster «X» without mtype: check it`).

### 5. Salida por consola y resumen

| Nivel | Qué sale |
| --- | --- |
| 0 | Cabecera (`load polls: 1 ámbito`, `· overwrite`, `· force`, `· dry run`) y el resumen final |
| 1 | Más `[i/n] es: empezando`, el bloque de nombres sin mapear, los avisos en vivo (`· aviso:`) y la línea de resultado |
| 2 | Más una línea por fase con sus segundos (`lectura`, `nombres`, `series`, `guardado`, `pesos`), los descartados y `parties_checks` |
| 3 | Más la salida propia del loader y del `Computer` (reciben `verbose - 2`: los contadores por año, `Save...`, `n rows updated...`) |

Líneas de resumen:

```
es: polls loaded (520 read, 4 saved, 2026-11-29) in 38 s
es: polls loaded (520 read, 520 saved, 2026-11-29, overwrite) in 61 s · 2 pollsters added, 1 sponsor added · 3 avisos
es: polls checked (520 read, 4 new, 2026-11-29, dry run) in 12 s
es: polls checked (520 read, 520 new, 2026-11-29, overwrite, dry run) in 12 s
es: polls failed (unmapped pollsters: Sumar; unmapped sponsors: Artículo 14)
es-md: polls failed (no polls article for 2027-05-23)
es-an: polls skipped (no polls article for 2030-06-16)
es-ex: polls skipped (no upcoming event)
```

Los ámbitos están aislados y el job termina con `RuntimeError: load: failed scopes: es (polls)` si alguno
falló, después del resumen y del `alert`, como `compute`; los `skipped` no cuentan.

### 6. Relación con `compute` y con `run_load.py`

- El job recalcula sólo los pesos del evento, como el cuaderno. Después hay que lanzar `compute` (series y
  ratings) y `publish`; el README lo dice en el mismo sitio donde hoy lo dice de los cuadernos. No se añade
  una opción para encadenar `compute`: una orden, una cosa.
- `load/run_load.py` no se toca (decisión del diseño tras la revisión): su paso `polls` es otra cosa, un
  dry run por defecto que recorre todos los eventos del ámbito, tolera los nombres sin mapear y los cuenta
  con frecuencias en su informe. Unificarlo con `load.load_polls` (un `strict` o un `--force`) queda como
  trabajo posterior, igual que mover `urls`, `results` y `cera` al job.

### 7. Tests y documentación

- `tests/test_load_unit.py`: `load_polls` con un `FakeLoader` (anota las llamadas y simula nombres sin
  mapear, descartados y `fixes_missing`) y un `FakeComputer`: lo nuevo y el `overwrite`, parada por nombres,
  `force` con altas sólo de casas y medios (nunca partidos ni descartados), `dry_run` sin escrituras,
  resultado y segundos.
- `tests/test_jobs_load.py`: el job como `test_jobs_compute.py`: cabeceras, niveles de verbosidad, líneas de
  resumen, aislamiento de fallos y `RuntimeError` final, `event_date` por defecto y por parámetro, ámbito sin
  artículo.
- `tests/test_loader_unit.py`: las reglas de descarte (listas `skip` de casas, medios y partidos, nombre de
  partido, «CIS (…)»), la precedencia frente al idmap y las listas `*_skipped`.
- `tests/test_data_files.py`: `skip.parties` de `wp-maps.json` es una lista de claves sin duplicados y
  ninguna coincide con un alias de `parties`.
- `deploy/README.md`: sección «Cargar los sondeos» entre «Comprobar S3» y «Calcular las series», con la
  misma estructura que la de `compute` (qué hace, ejemplos, parámetros, niveles, estados y resumen, Docker).
  Nota: en ECS el contenedor necesita salida HTTPS a Wikipedia.
- El docstring de `Load` lleva los ejemplos de línea de comandos y de Docker como `Compute`.

## Hallazgos de la revisión del constructor (resueltos)

- Altas: el framework inserta con `IDENTITY` (`Model.format_query` pone `DEFAULT` en la clave `autokey`
  cuando viene vacía); hay que llamar a `insert(df, truncate=False)`, porque el `insert` del adaptador
  trunca la tabla por defecto.
- Calidad «sin evaluar» = `scale['floor']` de la rúbrica (`quality_from_points(0, scale)`), lo mismo que
  `pollsters_quality_values` da a una casa sin evaluación y sin generales.
- `select_series` corre también con `dry_run` (es lectura) para poder decir `4 new`.
- La fábrica `computer` del job sustituye las tres líneas de `compute_series`
  (`Computer(scope, event_dates=[event_date], ...).build_series().compute_weights(save=True, overwrite=True)`).
- Comprobar que el evento tiene `polls` en `wp-urls.json` **antes** de construir el loader: su `__init__`
  hace `KeyError` y consulta la base.
- Re-resolver tras las altas: `read_cols` descarta las filas de casas sin id, así que `self.data` no sirve;
  hay que conservar los documentos parseados (o un `cache_dir`) y volver a leer de ahí.

## Queda al constructor

- Nombre exacto de las fases y de los helpers de `mtpy/lib/load.py`; forma del `FakeLoader`.
- Cómo re-resolver las filas tras las altas sin otra descarga (reaprovechar `self.data` y los mapas, o la
  caché del loader).
- Dónde vive la regla de descarte dentro de `read_cols` y cómo se cargan los nombres de los partidos para la
  comparación (ya están en `self.parties`).
- Texto exacto de los avisos y de las razones de fallo, respetando el formato de las líneas de arriba.
- La curación inicial de `skip.parties`: sacar los nombres de la columna `parties_missing` de
  `files/stage/load_report.csv` (y de un dry run de `es` si saliera alguno), dejarlos en el plan a la vista
  de Luis y escribirlos en `wp-maps.json` ordenados.
