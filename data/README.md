# Datos de entrada

Todo lo que el modelo **lee** vive aquí y está versionado. Lo que el modelo **escribe** en tiempo de ejecución
(pronósticos `fc/`, figuras `img/`) va a `files/`, que está ignorado por git. La raíz de este directorio se
resuelve en `mtpy.run()` (`app.data`, `app.datapath`); se puede cambiar con la variable de entorno `DATA_PATH`.

| Fichero | Contenido | Origen y licencia |
|---|---|---|
| `params.json` | Por ámbito y evento: partidos con resultado y en encuestas, mapas de bloques (`max`, `min`, `main`, `blocks`, `vs`) y `smap` (herencia de votos entre elecciones; `regions` son códigos de provincia `region_id`) | Curación propia |
| `es-provinces.csv` | Provincias con su código oficial (`code` = `region_id` = código INE), comunidad y población | Elaboración propia a partir del INE |
| `es-regions.csv`, `es-regions-ages.csv` | Comunidades autónomas y población por edad | Elaboración propia a partir del INE |
| `wikipedia/wp-urls.json` | URLs de las páginas "Opinion polling for the … Spanish general election" por evento | Curación propia |
| `wikipedia/wp-maps.json` | Alias de partidos, encuestadoras y patrocinadores hacia los nombres canónicos de la base de datos | Curación propia |
| `infoelectoral/es/PROV_02_YYYYMM_1.xlsx` | Resultados oficiales del Congreso por provincia, un fichero por elección (1977-2023) | Ministerio del Interior, portal Infoelectoral; reutilización permitida citando "Origen de los datos: Ministerio del Interior" |
| `infoelectoral/es/PROV_02_YYYYMM_1.json` | Mapeo manual de cada candidatura del XLSX a la sigla usada en el modelo (`null` = "Otros") | Curación propia |

Las tablas de encuestas de Wikipedia se descargan en vivo con `WikipediaLoader`; su contenido es CC BY-SA 4.0.
