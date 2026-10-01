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
| `wikipedia/wp-urls.json` | URLs de las páginas "Opinion polling for the … Spanish general election" por evento | Curación propia |
| `wikipedia/wp-maps.json` | Alias de partidos, encuestadoras y patrocinadores hacia los nombres canónicos de la base de datos | Curación propia |
| `infoelectoral/es/PROV_02_YYYYMM_1.xlsx` | Resultados oficiales del Congreso por provincia, un fichero por elección (1977-2023) | Ministerio del Interior, portal Infoelectoral; reutilización permitida citando "Origen de los datos: Ministerio del Interior" |
| `infoelectoral/es/PROV_02_YYYYMM_1.json` | Mapeo manual de cada candidatura del XLSX a la sigla usada en el modelo (`null` = "Otros") | Curación propia |

Las tablas de encuestas de Wikipedia se descargan en vivo con `WikipediaLoader`; su contenido es CC BY-SA 4.0.

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
