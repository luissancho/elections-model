# Datos de entrada

Todo lo que el modelo **lee** vive aquí y está versionado. Lo que el modelo **escribe** en tiempo de ejecución
(pronósticos `fc/`, figuras `img/`) va a `files/`, que está ignorado por git. La raíz de este directorio se
resuelve en `mtpy.run()` (`app.data`, `app.datapath`); se puede cambiar con la variable de entorno `DATA_PATH`.

| Fichero | Contenido | Origen y licencia |
|---|---|---|
| `params.json` | Por ámbito y evento: partidos con resultado y en encuestas, mapas de bloques (`max`, `min`, `main`, `blocks`, `vs`) y `smap` (herencia de votos entre elecciones; `regions` son códigos de provincia `region_id`) | Curación propia |
| `es-provinces.csv` | Provincias con su código oficial (`code` = `region_id` = código INE), comunidad y población | Elaboración propia a partir del INE |
| `es-regions.csv`, `es-regions-ages.csv` | Comunidades autónomas y población por edad | Elaboración propia a partir del INE |
| `es-scopes.csv` | Catálogo de ámbitos (`scode` ISO 3166-2 en minúsculas; `es` es el nacional): comunidad, código INE, ámbito padre, gentilicio inglés de Wikipedia, umbral legal, peso en el rating y escaños que elige la próxima elección | Curación propia (ver "Ámbitos y circunscripciones") |
| `es-districts.csv` | Circunscripciones de cada ámbito (`scope`, `region_id`): nombre, código INE de la provincia cuando lo es, comunidad (`reg_code`), población, escaños vigentes y alias en inglés (`\|`) | Elaboración propia a partir del INE y de las leyes electorales |
| `pollsters-quality-criteria.json` | La rúbrica de la calidad previa de las encuestadoras: cinco criterios con sus niveles y la escala de puntos a calidad | Curación propia |
| `pollsters-quality.csv` | La evaluación de cada encuestadora según la rúbrica: puntos y fuente por criterio, notas, estado (`borrador`/`revisada`) y fecha de comprobación; `pollster_id` es el `id` de `elections.pollsters` | Curación propia |
| `wikipedia/wp-urls.json` | Por ámbito y evento, URL de los sondeos (`polls`) y de los resultados (`results`) en la Wikipedia inglesa. En `es` son las páginas "Opinion polling for the … Spanish general election"; en los `es-*`, el artículo de cada elección autonómica, que trae ambas cosas (`python load/run_load.py --what urls` lo reconstruye) | Curación propia |
| `wikipedia/wp-maps.json` | Alias de partidos, encuestadoras y patrocinadores hacia los nombres canónicos de la base de datos. `scopes` añade alias de partidos por ámbito (el enlace del PP de Madrid es `PP`), que prevalecen sobre los globales; `skip` lista las "casas" y patrocinadores que se dejan fuera a propósito (sondeos internos de los partidos) | Curación propia |
| `results/{scope}/{fecha}.csv` | Opcional: votos válidos oficiales por circunscripción (`region_id`, `votes`, `blank`) de una elección autonómica; sustituyen al reparto por población y dejan `events_data.estimated` en falso | Fuentes oficiales de cada comunidad |
| `results/{scope}/{fecha}-seats.csv` | Opcional: erratas de la tabla por circunscripción de Wikipedia (`region_id`, `key`, `seats`). Hoy sólo Cataluña 2010 (CiU en Lleida: 9 escaños, no 8) | Curación propia |
| `infoelectoral/es/PROV_02_YYYYMM_1.xlsx` | Resultados oficiales del Congreso por provincia, un fichero por elección (1977-2023) | Ministerio del Interior, portal Infoelectoral; reutilización permitida citando "Origen de los datos: Ministerio del Interior" |
| `infoelectoral/es/PROV_02_YYYYMM_1.json` | Mapeo manual de cada candidatura del XLSX a la sigla usada en el modelo (`null` = "Otros") | Curación propia |
| `cera/ine/cerapropais_{yy}.xls[x]` | Estudio del voto CERA: censo CERA cerrado el día 1 de cada mes por provincia de inscripción y país de residencia, un libro por año (2006–2014 `.xls`, 2015 en adelante `.xlsx`; el de 2026 llega al 1-9-2026) | INE, Oficina del Censo Electoral (`ine.es/oficina_censo/censo_cerrado/`); reutilización permitida citando la fuente |
| `cera/ine/elecgen2026_tab{6,7,8}.xlsx` | Censo de consulta de las generales del 29-11-2026: electores residentes en el extranjero por país (total nacional), por comunidad y por provincia × país | INE, Oficina del Censo Electoral (`ine.es/oficina_censo/elecgen2026/`); reutilización permitida citando la fuente |
| `cera/escrutinio/es-{fecha}.csv` | Escrutinio del voto CERA de las generales de 2019-11 y 2023-07 por provincia (`region_id`, `label` de Interior, `electors`, `voters`, `invalid`, `blank` y una columna por candidatura con la etiqueta de Interior). Electores y votantes de Infoelectoral (`Censo CERA` y `Total votantes CERA` de `infoelectoral/es/PROV_02_YYYYMM_1.xlsx`); votos por candidatura, nulos y blancos del fichero de Interior, cuyas cifras de electores y votantes contienen erratas (electores de Albacete en 2023; de Granada, Lleida y Santa Cruz de Tenerife en 2019; votantes de Barcelona en 2023) | Ministerio del Interior, «Resultados voto CERA» y portal Infoelectoral; reutilización permitida citando "Origen de los datos: Ministerio del Interior" |
| `cera/escrutinio/candidatures.json` | Por elección, cada columna de candidatura del escrutinio CERA → nombre completo de la candidatura en la fila 3 de `infoelectoral/es/PROV_02_YYYYMM_1.xlsx` (el PSC va con el PSOE, como en Infoelectoral) | Curación propia |
| `cera/provinces.json` | Etiqueta de provincia normalizada (mayúsculas, sin acentos, solo `[A-Z0-9]`) → `region_id`, para las etiquetas del INE (`tab8`), de Interior y de `es-provinces.csv` | Curación propia |
| `cera/countries.csv` | Países del INE (`code`, `name`) con su región (`latam`, `europe`, `usa_canada`, `other`) y su exposición a la Ley de Nietos (`alta`, `media`, `baja`, ajustada a la tabla 2 del artículo) | Curación propia a partir del INE |
| `cera/params.json` | Parámetros del estudio del voto CERA (convocatoria, inscripciones en litigio, participación por origen, periodos, bloques, umbral de la comprobación) | Curación propia |
| `cera/autonomicas.csv` | Voto CERA de seis autonómicas (Castilla y León 2022/2026, Andalucía 2022/2026, Aragón 2023/2026): censo, votantes, PP, PSOE, VOX, otras izquierdas y bloques izquierda/derecha/resto | Juntas electorales de cada comunidad (BOCYL, BOJA, BOA) |
| `cera/consulados/{proceso}.csv` | Censo y votantes CERA por país de residencia (consulados sumados por país): Castilla y León 2026, Extremadura 2023 y 2025, Aragón 2026 | Juntas electorales de cada comunidad (BOCYL, DOE, BOA) |

Las tablas de encuestas de Wikipedia se descargan en vivo con `WikipediaLoader`, y los resultados autonómicos con `WikipediaResultsLoader`; su contenido es CC BY-SA 4.0. `load/run_load.py` hace la carga por lotes de los ámbitos autonómicos y guarda una copia de los artículos en `files/stage/wikipedia/`.

## Calidad previa de las encuestadoras

Mide **cómo trabaja la casa**, no cuánto acierta (eso lo miden el rating y el sesgo de casa). La rúbrica son cinco criterios; los textos de cada nivel están en `pollsters-quality-criteria.json`.

| Criterio | Clave | Puntos |
|---|---|---|
| Trabajo de campo | `fieldwork` | 0-3 |
| Transparencia | `transparency` | 0-3 |
| Independencia | `independence` | 0-2 |
| Identidad | `identity` | 0-1 |
| Estándares | `standards` | 0-1 |

La calidad es `quality = max(floor, per_point · puntos)`, con `floor = 5` y `per_point = 8` (valores actuales de `scale` en el JSON), de modo que el tope, con 10 puntos, es 80. Una encuestadora sin evaluar (sin fila en el CSV) vale `floor`.

`pollsters-quality.csv` es la fuente; la base (`elections.pollsters_quality` y `pollsters.quality`) es una réplica que escribe `save_catalogues()`. `pollster_id` es el `id` de `elections.pollsters`.

La tabla réplica se puede borrar y recrear cuando se quiera (`DROP TABLE elections.pollsters_quality` y después `save_catalogues()`): sus datos salen enteros del CSV, y `create(replace=False)` solo crea la tabla si falta, nunca cambia los tipos de una que ya existe. Las fuentes y las notas son `TEXT`.

Cada punto lleva su fuente en la columna `{criterio}_source`: una URL o una referencia citable. Si la fuente es la propia base, se cita como «base de datos del proyecto, consulta de patrocinadores a {fecha}».

## Resultados autonómicos

No están en Infoelectoral. Se leen del artículo de cada elección: la tabla *Overall* da votos, porcentaje y escaños de cada candidatura y los totales del conjunto (exactos, `region_id = 0`); la tabla *Distribution by constituency* da porcentaje y escaños por circunscripción, **sin votos**. Por eso en `events_data` y `events_results` de los ámbitos `es-*`:

- Los votos válidos de cada circunscripción son los del conjunto repartidos en proporción a su población (`events_data.estimated = true`), salvo que exista `results/{scope}/{fecha}.csv`. Los votos de un partido en la circunscripción son su porcentaje de esos votos. No afecta al reparto de escaños ni a la barrera, que operan con porcentajes.
- Las candidaturas con resultado en el conjunto pero sin columna en la tabla por circunscripción (no sacaron escaño en ninguna) llevan en cada circunscripción su porcentaje del conjunto, limitado al voto que dejan las demás: es una estimación plana, la base del swing si después crecen.
- Las candidaturas sin alias en `wp-maps.json` suman a `'-'` (otros).
- Un distrito único (Madrid, Murcia desde 2019, Navarra, La Rioja, Cantabria) tiene una fila de circunscripción igual a la del total, con el código INE de la provincia. La lista autonómica canaria (`region_id` 100) la vota todo el archipiélago.

## Ámbitos y circunscripciones

Una sola numeración de comunidades en todo `data/`: la del INE (`es-regions*.csv`, `es-provinces.csv.reg_code`,
`es-scopes.csv.ine_code`, `es-districts.csv.reg_code`).

`region_id` en `es-districts.csv` y en la base de datos: `0` es el total del ámbito; las provincias conservan
su código INE (1-52) en cualquier ámbito; las circunscripciones no provinciales empiezan en 100 (zonas de
Asturias 101-103, islas de Baleares 101-104, islas de Canarias 101-107 y lista autonómica canaria 100,
distritos de Murcia hasta 2015 101-105). `seats` vacío marca una circunscripción extinguida.

Población: provincias, de `es-provinces.csv`; islas, padrón del INE a 1-1-2021 (tabla 2910); zonas de
Asturias y distritos antiguos de Murcia, suma del padrón municipal del INE a 1-1-2025 (tablas 2886 y 2883)
con la composición municipal de cada circunscripción. Sólo se usa como peso relativo para repartir los votos
válidos entre circunscripciones cuando no hay cifra oficial.

Umbrales (`threshold`: % de los votos válidos, incluidos los blancos, de la circunscripción;
`threshold_scope`: % del conjunto de la comunidad). Una candidatura entra en el reparto si supera uno
**o** el otro; un umbral vacío no se puede superar. Comprobados el 1-10-2026 contra la sección
*Electoral system* del artículo de la última elección de cada comunidad en la Wikipedia inglesa, que cita
el estatuto y la ley electoral autonómica:

| Ámbito | Circunscripción | Conjunto | Notas |
|---|---|---|---|
| `es` | 3 | | LOREG, art. 163.1.a |
| `es-an`, `es-ar`, `es-as`, `es-cl`, `es-cm`, `es-ct`, `es-pv` | 3 | | |
| `es-ib`, `es-ga` | 5 | | |
| `es-cb`, `es-md`, `es-ri` | 5 | | Distrito único |
| `es-nc` | 3 | | Distrito único |
| `es-mc` | 3 | | Distrito único desde 2019; hasta 2015, cinco distritos y sólo 5 % regional |
| `es-vc` | | 5 | Sólo el umbral autonómico |
| `es-ex` | 5 | 5 | El autonómico exige concurrir en las dos provincias (no se modela) |
| `es-cn` | 15 | 4 | Hasta 2015: 30 insular o 6 autonómico, y 60 escaños sin lista autonómica |

Los umbrales históricos distintos del vigente se registran por evento en `params.json` (`threshold`,
`threshold_scope`). `seats` es el tamaño de la cámara que elige la próxima elección (Madrid: 143, por
población; la actual tiene 135).
