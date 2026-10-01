# Método 11: ámbitos autonómicos (01-10-2026)

Cambios: catálogos `data/es-scopes.csv` y `data/es-districts.csv` (modelos `Scopes` y `Districts`, que sustituyen a `Provinces`); `WikipediaLoader` generalizado y `WikipediaResultsLoader` nuevo (`mtpy/lib/loader.py`); carga por lotes `load/run_load.py`; `Computer.compute_ratings(scopes=...)`, `rating_pool`, `rate_events` y `regional_flags` (`mtpy/lib/computer.py`); umbral por ámbito, circunscripciones, ruido sólo por circunscripción y estimadores con caída al ámbito padre en el `Simulator` (`mtpy/lib/simulator.py`); `get_scopes`, `get_districts`, `get_thresholds`, `update_featured` (`mtpy/lib/data.py`); backtest por ámbito (`backtest/run_backtest.py --scopes`). Tests: `tests/test_loader_unit.py`, `tests/test_data_unit.py`, `tests/test_data_files.py`, `tests/test_computer_unit.py`, `tests/test_simulator_unit.py`, `tests/integration/test_regional_load.py`, `test_ratings_scopes.py`, `test_regional_model.py`, `test_scopes.py`. Notebooks: `data-load/ResultsLoadWikipedia`, `events/es-md-202305/`. Plan de implementación: `metodo-11-plan.md`.

## Problema

El modelo sólo conocía las elecciones generales (`scope = 'es'`). Las autonómicas tienen sus propios sondeos, sus propias circunscripciones y su propia barrera legal, y además son una fuente de información sobre las casas: quien se equivoca en Madrid o en Cataluña dice algo de cómo trabaja. El objetivo es que cualquier elección autonómica se trate igual que la general:

1. **Recopilar** sondeos y resultados de las 17 comunidades.
2. **Alimentar el ranking** de casas con sus errores en las autonómicas, con menos peso que los nacionales.
3. **Pronosticar y simular** una elección autonómica con el `Forecaster` y el `Simulator` de siempre: promedio, intervalos, escaños por circunscripción y probabilidades.

## Decisiones

| Id | Decisión |
|---|---|
| D1 | Códigos de ámbito ISO 3166-2 en minúsculas: `es-md` Madrid, `es-cl` Castilla y León, `es-ct` Cataluña… El nacional sigue siendo `es` |
| D2 | Un partido conserva su sigla nacional cuando la candidatura es su federación regional (PP, PSOE, VOX, UP, SUMAR, Cs). Las marcas propias son partidos aparte, con `parent_id` cuando lo tienen (PSC, PSE-EE, Más Madrid) o sin él (CHA, PRC, UPL, CC, BNG…) |
| D3 | Peso de los errores autonómicos en el rating: `rating_weight = 0,5`; el nacional vale 1. Pendiente de calibrar |
| D4 | Votos válidos por circunscripción: se reparten por población al cargar (`estimated`) y se sustituyen por la cifra oficial donde se cure un CSV |
| D5 | Alcance temporal: elecciones autonómicas desde 2009 |
| D6 | Esta especificación vive en la serie de métodos |

Se adoptó el enfoque en el que datos, pesos, errores y desviaciones siguen siendo **por ámbito y por evento**, como antes, y sólo el paso de ratings se hace **transversal**. Se descartaron un `Computer` por ámbito con ratings separados (los errores autonómicos nunca llegarían al ranking nacional) y un modelo jerárquico conjunto de errores por casa y ámbito (no hay datos para estimarlo todavía).

## Datos

**Fuente.** La Wikipedia inglesa no tiene páginas de sondeos autonómicos aparte: las tablas están dentro del artículo de cada elección (`{año}_{gentilicio}_regional_election`, sección *Opinion polls*), con el mismo formato que las nacionales. El mismo artículo trae los resultados: *Results › Overall* (votos, porcentaje y escaños de cada candidatura y los totales del conjunto) y *Distribution by constituency* (porcentaje y escaños por circunscripción, sin votos). Infoelectoral no cubre las autonómicas.

**Inventario.** `load/run_load.py --what urls` recorre los artículos desde el de la próxima elección hacia atrás por el enlace "elección anterior" de cada ficha: 95 eventos, 78 celebrados desde 2009 y 17 próximos. Los 78 tienen tabla de sondeos. Aragón, Castilla y León y Extremadura no tienen todavía artículo de la próxima elección (votaron entre diciembre de 2025 y marzo de 2026): su evento próximo se fecha cuatro años después y no tiene sondeos; Andalucía, que votó en mayo de 2026, tampoco.

**Qué se ha cargado.**

| Ámbito | Circunscripciones | Umbral | Elecciones (featured) | Sondeos | Casas | Próxima (sondeos) |
|---|---|---|---|---|---|---|
| `es-an` Andalucía | 8 provincias | 3 | 5 (5) | 325 | 39 | 2030-06-16 (0) |
| `es-ar` Aragón | 3 provincias | 3 | 5 (3) | 133 | 24 | 2030-02-08 (0) |
| `es-as` Asturias | 3 zonas | 3 | 5 (2) | 93 | 20 | 2027-05-23 (10) |
| `es-ib` Baleares | 4 islas | 5 | 4 (3) | 97 | 13 | 2027-06-27 (6) |
| `es-cn` Canarias | 7 islas y lista autonómica | 15 insular o 4 autonómico | 4 (3) | 101 | 20 | 2027-06-27 (6) |
| `es-cb` Cantabria | 1 | 5 | 4 (3) | 73 | 14 | 2027-05-23 (3) |
| `es-cl` Castilla y León | 9 provincias | 3 | 5 (3) | 144 | 21 | 2030-03-15 (0) |
| `es-cm` Castilla-La Mancha | 5 provincias | 3 | 4 (3) | 124 | 24 | 2027-05-23 (17) |
| `es-ct` Cataluña | 4 provincias | 3 | 6 (6) | 481 | 43 | 2028-06-26 (19) |
| `es-vc` C. Valenciana | 3 provincias | 5 autonómico | 4 (4) | 210 | 34 | 2027-06-27 (34) |
| `es-ex` Extremadura | 2 provincias | 5 provincial o 5 autonómico | 5 (4) | 116 | 24 | 2029-12-21 (0) |
| `es-ga` Galicia | 4 provincias | 5 | 5 (5) | 255 | 31 | 2028-03-25 (10) |
| `es-md` Madrid | 1 | 5 | 5 (5) | 271 | 31 | 2027-05-23 (14) |
| `es-mc` Murcia | 1 (5 distritos hasta 2015) | 3 (5 autonómico hasta 2015) | 4 (4) | 123 | 20 | 2027-05-23 (23) |
| `es-nc` Navarra | 1 | 3 | 4 (4) | 103 | 18 | 2027-05-23 (8) |
| `es-pv` País Vasco | 3 provincias | 3 | 5 (5) | 214 | 32 | 2028-05-21 (8) |
| `es-ri` La Rioja | 1 | 5 | 4 (3) | 79 | 15 | 2027-05-23 (4) |
| **Total** | 66 | | 78 (65) | 2.942 | 88 | |

Una elección es `featured` cuando tiene resultado oficial y al menos 10 sondeos útiles (computados, con peso de solapamiento positivo y sin agregadores, paneles online, sondeos en veda ni a pie de urna). Quedan fuera 13, nueve de ellas de mayo de 2019, donde dominan los trackings de ElectoPanel.

**Cobertura de sondeos.** Las tablas de los 78 artículos suman 3.687 filas: 347 son filas de resultado (la propia elección, la anterior y las generales intercaladas), 204 son sondeos que sólo publican escaños y 3.100 tienen estimación de voto y ventaja numérica. De éstas, 31 son sondeos internos de partido (la "casa" es PP, PSOE, PSPV, ERC…), que no se cargan. De las 3.069 restantes están cargadas 2.942, el **95,9 %**. Lo que falta: 92 sondeos de 81 casas con uno o dos sondeos, que no se han dado de alta; unos 20 con la fecha sin día ("Dec 2019"); 15 con empate en cabeza ("Tie"), que el cargador descartaba ya en las generales; y unos 20 con la ventaja dada como rango. El inventario de la especificación inicial (3.479 filas) contaba todas las filas de las tablas, también las de resultado.

**Curación.** 151 enlaces de partido de Wikipedia: 103 van a partidos que ya existían (las federaciones a la sigla nacional, D2), 14 a 13 partidos nuevos (PSC, PSE-EE, MM, PorA, JxSí, SCI, ASG, AHI, DO, El Pi, MxMe, GxF, Sa Unió) y 34 se quedan en "otros" (sin escaños en ninguna elección). 39 casas nuevas, las que tienen tres o más sondeos, con `quality` 5 (casa sin historial); 53 patrocinadores nuevos. Los alias de partido son por ámbito (`wp-maps.json › scopes`), porque el enlace del "People's Party of the Community of Madrid" es `PP` sólo en Madrid.

## Método

### Catálogos

`data/es-scopes.csv` da a cada ámbito su código, comunidad, ámbito padre, gentilicio inglés (para construir los títulos de Wikipedia), umbral legal, peso en el rating y escaños que elige la próxima elección. `data/es-districts.csv` da las circunscripciones de cada ámbito, también las 52 de `es`. Las provincias conservan su código INE (1-52) en cualquier ámbito, de modo que las reglas `regions` del `smap` siguen valiendo; las circunscripciones que no son provincias empiezan en 100 (zonas de Asturias 101-103, islas de Baleares 101-104, islas de Canarias 101-107 y lista autonómica 100, distritos de Murcia hasta 2015 101-105). El código lee los CSV; las tablas `scopes` y `districts` son su réplica en la base de datos (`save_catalogues()`). `data/` usa ya una sola numeración de comunidades, la del INE.

Los umbrales se comprobaron contra la sección *Electoral system* del artículo de la última elección de cada comunidad, que cita el estatuto y la ley electoral (tabla en `data/README.md`). Una candidatura entra en el reparto si supera el umbral de su circunscripción **o** el del conjunto del ámbito; un umbral vacío no se puede superar. Así caben los tres casos que existen: sólo circunscripción (la mayoría), sólo conjunto (Comunidad Valenciana; Murcia hasta 2015) y los dos (Canarias, 15 % insular o 4 % autonómico; Extremadura). Los umbrales históricos distintos del vigente van por evento en `params.json` (Canarias 2011 y 2015: 30 y 6; Murcia 2011 y 2015: 5 autonómico).

### Ingesta

**Sondeos** (`WikipediaLoader`). La tabla es la primera `wikitable` cuya primera fila empieza por "Polling firm" bajo un encabezado que contenga "Voting intention"; si no hay, la que cuelga directamente de "Opinion polls" (artículos pequeños sin subsecciones). Las columnas de partido son las celdas de cabecera con enlace a un artículo, y "Lead" se localiza por su texto, no por posición. Los enlaces se normalizan (llegan unas veces codificados y otras no). Dos columnas del mismo sondeo que acaban en el mismo partido (miembros de una coalición posterior) se suman. Una celda de fecha sin día descarta la fila.

**Resultados** (`WikipediaResultsLoader`). El total del ámbito (`region_id = 0`) sale de *Overall* y es exacto; las filas de los miembros de una coalición se ignoran, porque sus votos y escaños ya están en la fila de la coalición. La tabla por circunscripción da porcentaje y escaños, así que:

- los votos válidos de cada circunscripción son los del conjunto repartidos en proporción a su población (`events_data.estimated`), salvo que `data/results/{scope}/{fecha}.csv` traiga la cifra oficial. No afecta al reparto de escaños ni a la barrera, que operan con porcentajes dentro de la circunscripción;
- las candidaturas con resultado en el conjunto pero sin columna en esa tabla (no sacaron escaño en ninguna parte) llevan en cada circunscripción su porcentaje del conjunto, limitado al voto que dejan las demás. Es una estimación plana, y es lo que permite proyectar a un partido que después crece (VOX antes de 2019);
- un distrito único tiene una fila de circunscripción igual a la del total; la lista autonómica canaria usa su propio porcentaje y la vota todo el archipiélago;
- una circunscripción cuyo nombre no está en el catálogo (ni como alias) detiene la carga con un error que la nombra;
- las erratas de la tabla se corrigen con `data/results/{scope}/{fecha}-seats.csv`. Hay una: Cataluña 2010 da a CiU 8 escaños en Lleida y fueron 9.

Para el evento próximo se crean el evento y sus circunscripciones con los escaños del catálogo, sin votos.

**Comprobación.** En las 78 elecciones, los votos de las candidaturas suman los válidos menos los blancos (la mayor diferencia, 952 votos de 2,6 millones en la Comunidad Valenciana de 2019, es del propio artículo) y los escaños por circunscripción suman los del parlamento. Y, sobre todo, **D'Hondt con los umbrales del catálogo reproduce los escaños oficiales en 287 de 291 circunscripciones**; las cuatro restantes bailan un escaño en el último cociente porque los porcentajes de la tabla llevan un decimal.

### Ratings transversales

`Computer.compute_ratings(scopes='all')` reúne los sondeos filtrados de todos los ámbitos con peso positivo y, para cada evento del ámbito del `Computer` (y el próximo), puntúa a las casas con los sondeos de las elecciones celebradas **estrictamente antes**, de cualquier ámbito. Las doce autonómicas de un mismo día no se ven entre sí; la general de julio de 2023 sí ve las de mayo. El peso de cada sondeo se multiplica por el `rating_weight` de su ámbito; la posición del sondeo y la semana se cuentan por ámbito y evento, y `num_events` cuenta pares de ámbito y fecha. Los ámbitos con peso 0 se descartan antes de calcular ningún peso. `scopes=None` (por defecto) conserva el comportamiento anterior.

Invariantes comprobados: con `scopes=None`, los 1.012 ratings de `es` son idénticos a los guardados; con peso 0 en todos los autonómicos, `scopes='all'` da lo mismo que `scopes=None`.

### Simulator

- **Umbral.** `Simulator(threshold=None)` toma los umbrales del ámbito y del evento; un número fija el de circunscripción y `0` lo desactiva.
- **Estimadores.** El error de sondeo, los escaños por regresión, la deriva y la razón de composición se ajustan con las elecciones del propio ámbito cuando tiene al menos 3 `featured` anteriores, y con las del ámbito padre (`es`) si no. El estimador de escaños regresa la **cuota** de escaños de la cámara, para que parlamentos de 33 y 350 escaños sean comparables. El prior de los efectos de casa se anula con menos de 3 elecciones previas.
- **Partidos "regionales".** En `es` la marca es la curada en la tabla de partidos. Dentro de una comunidad se deriva de los resultados: un partido es regional cuando sólo concurrió en circunscripciones que suman menos de la mitad del voto válido (UPL, Por Ávila, los partidos de una isla). La especificación decía cero para todos, y con eso un partido del 0,4 % recibía el error de sondeo de uno de ámbito completo (1,3 puntos): Sa Unió perdía el escaño de Formentera en el 37 % de las simulaciones, dejando 52 de 300 con un escaño sin repartir, y Por Ávila lo perdía en el 43 %. Con la marca derivada, ninguna simulación de Baleares queda corta y Por Ávila conserva el suyo en más del 80 %.
- **Ruido del método 9.** Dentro de una comunidad no hay componente por comunidad: sólo el choque por circunscripción, con la curva provincial estimada en `es`. En un distrito único no hay nada que redistribuir.
- **Partidos sin base.** Un partido simulado sin resultado previo ni regla de herencia se queda sin escaños, pero no rompe la proyección. Las reglas (`smap` en `params.json`) cubren los casos claros: SALF hereda de VOX, SUMAR de UP, Más Madrid de UP en 2019, VOX en Cataluña 2021 de PP y Cs, Aliança Catalana de Junts, Por Andalucía de la Adelante de 2018, y Navarra Suma y su ruptura.
- **Errores claros.** Un evento sin sondeos, o con tan pocos que el promedio no se puede ajustar, falla con un mensaje que lo dice, en vez de devolver ceros.

## Resultados

**Reparto determinista a 6 días.** Error absoluto medio de escaños de los partidos principales: Castilla y León 2022, 1,25; Asturias 2023, 0,6; Canarias 2023, 1,4. Madrid 2023 da 3,2: los sondeos ponían a Podemos-IU en el 5,1 % y sacó el 4,76 %, por debajo de la barrera, así que el modelo le da 7 escaños que no tuvo. Sin ese partido el error es 0,5. Es un fallo de pronóstico en el borde de la barrera, no del reparto.

**Próximas elecciones.** De los 17 ámbitos, 11 se pueden simular hoy; 4 no tienen sondeos todavía (Andalucía, Aragón, Castilla y León, Extremadura) y 2 tienen demasiado pocos para ajustar el promedio (Cantabria 3, La Rioja 4).

**Ratings.** Con los sondeos autonómicos al 0,5, las casas con sondeos evaluables en el último rating de `es` pasan de 53 a 84. Entre las que ya tenían, el rating cambia 4,9 puntos de media en valor absoluto y la correlación con el anterior es 0,96. Los mayores cambios: GAD3 +18,9 (de 15 a 74 sondeos evaluables), CIS −16,9 (de 16 a 80), Target Point +14,5, 40dB −14,2, Data10 +13,6, CEMOP +13,4, Simple Lógica −10,8, InvyMark +10,8, Sigma Dos −10,6, Ipsos −10,6 y GESOP −10,4. Sondaxe no se mueve (+0,1). Las casas nuevas entran entre 5 y 30 (Gizaker 30,4 con 13 sondeos). La tabla completa está en `files/stage/m11_ratings_impact.csv`. **Los ratings transversales están guardados para los eventos autonómicos; los de `es` siguen siendo los de antes** hasta que se revise ese cambio.

**Backtest.** BACKTEST_PLACEHOLDER

## Salvedades

- **Formato de Wikipedia.** El cargador depende de la estructura de los artículos. Los tests usan una instantánea del 01-10-2026 de seis artículos (`tests/fixtures/wikipedia/`); `load/run_load.py` sin `--save` enseña qué ha cambiado antes de escribir nada.
- **Votos y cuotas estimados.** Los votos por circunscripción y las cuotas por circunscripción de las candidaturas sin escaño son estimaciones. Pesan en la agregación entre circunscripciones y en el voto en blanco, no en el reparto.
- **Partidos locales sin sondeos.** Los sondeos de la próxima elección canaria no listan a ASG ni a AHI, así que el modelo no les da los escaños de La Gomera y El Hierro. Lo mismo vale para cualquier partido de una sola circunscripción que las casas no publiquen.
- **Recién llegados territoriales.** Teruel Existe en 2023, Soria ¡Ya! en 2022, Democracia Ourensana en 2024 y AHI en 2023 no tenían geografía previa ni partido del que heredarla: la proyección no les da escaños.
- **Estimadores propios con pocos datos.** Con tres o cuatro elecciones, el estimador de error del ámbito es ruidoso; en Baleares deja a un partido regional del 1,5 % con un error de 0,04 puntos, demasiado estrecho.
- **Peso 0,5.** Es un valor de partida (D3). El cambio de rating de las casas grandes es considerable y conviene revisarlo antes de adoptarlo en `es`.
- **Sondeos con empate en cabeza.** Se descartan, también en las generales. Son 15 en las autonómicas.
- **Extremadura.** El umbral autonómico alternativo exige concurrir en las dos provincias; no se modela.

## Fuera de alcance

Prior de efectos de casa compartido por `parent_id` entre ámbitos; transferencia de los sondeos autonómicos al pronóstico provincial de las generales; calibración de `rating_weight` por comunidad; curva de ruido por circunscripción propia de cada comunidad; elecciones municipales y europeas.
