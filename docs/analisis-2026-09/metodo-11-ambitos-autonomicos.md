# Método 11: ámbitos autonómicos (01-10-2026)

Cambios: catálogos `data/es-scopes.csv` y `data/es-districts.csv` (modelos `Scopes` y `Districts`, que sustituyen a `Provinces`); `WikipediaLoader` generalizado y `WikipediaResultsLoader` nuevo (`mtpy/lib/loader.py`); carga por lotes `load/run_load.py`; `Computer.compute_ratings(scopes=...)`, `rating_pool`, `rate_events` y `regional_flags` (`mtpy/lib/computer.py`); umbral por ámbito, circunscripciones, ruido sólo por circunscripción y estimadores con caída al ámbito padre en el `Simulator` (`mtpy/lib/simulator.py`); `get_scopes`, `get_districts`, `get_thresholds`, `update_featured` (`mtpy/lib/data.py`); prior global de efectos de casa (`Forecaster.load_house_history`, `house_prior`, `industry_bias`; `party_roots` en `mtpy/lib/utils.py`); backtest por ámbito (`backtest/run_backtest.py --scopes`). Tests: `tests/test_loader_unit.py`, `tests/test_data_unit.py`, `tests/test_data_files.py`, `tests/test_computer_unit.py`, `tests/test_simulator_unit.py`, `tests/integration/test_regional_load.py`, `test_ratings_scopes.py`, `test_regional_model.py`, `test_scopes.py`. Notebooks: `data-load/ResultsLoadWikipedia`, `events/es-md-202305/`. Plan de implementación: `metodo-11-plan.md`.

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
| D7 | Los sondeos encargados por un partido a una casa real ("Sigma Dos / PP") se cargan sin patrocinador, como siempre se hizo en `es`; los sondeos internos, cuya "casa" es el partido, se descartan (`skip`). Ratificado el 06-10-2026 |

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
| **Total** | 61 vigentes y 5 de Murcia hasta 2015 | | 78 (65) | 2.942 | 88 | |

Una elección es `featured` cuando tiene resultado oficial y al menos 10 sondeos útiles (computados, con peso de solapamiento positivo y sin agregadores, paneles online, sondeos en veda ni a pie de urna). Quedan fuera 13, nueve de ellas de mayo de 2019, donde dominan los trackings de ElectoPanel.

**Cobertura de sondeos.** Las tablas de los 78 artículos suman 3.687 filas: 347 son filas de resultado (la propia elección, la anterior y las generales intercaladas), 204 son sondeos que sólo publican escaños y 3.100 tienen estimación de voto y ventaja numérica. De éstas, 31 son sondeos internos de partido (la "casa" es PP, PSOE, PSPV, ERC…), que no se cargan. De las 3.069 restantes están cargadas 2.942, el **95,9 %**. Lo que falta: 92 sondeos de 81 casas con uno o dos sondeos, que no se han dado de alta; unos 20 con la fecha sin día ("Dec 2019"); 15 con empate en cabeza ("Tie"), que el cargador descartaba ya en las generales; y unos 20 con la ventaja dada como rango. El inventario de la especificación inicial (3.479 filas) contaba todas las filas de las tablas, también las de resultado.

**Curación.** 151 enlaces de partido de Wikipedia: 103 van a partidos que ya existían (las federaciones a la sigla nacional, D2), 14 a 13 partidos nuevos (PSC, PSE-EE, MM, PorA, JxSí, SCI, ASG, AHI, DO, El Pi, MxMe, GxF, Sa Unió) y 34 se quedan en "otros" (sin escaños en ninguna elección). 39 casas nuevas, las que tienen tres o más sondeos, con `quality` 5 (casa sin historial); 53 patrocinadores nuevos. Los alias de partido son por ámbito (`wp-maps.json › scopes`), porque el enlace del "People's Party of the Community of Madrid" es `PP` sólo en Madrid.

## Método

### Catálogos

`data/es-scopes.csv` da a cada ámbito su código, comunidad, ámbito padre, gentilicio inglés (para construir los títulos de Wikipedia), umbral legal, peso en el rating y escaños que elige la próxima elección. `data/es-districts.csv` da las circunscripciones de cada ámbito, también las 52 de `es`. Las provincias conservan su código INE (1-52) en cualquier ámbito, de modo que las reglas `regions` del `smap` siguen valiendo; las circunscripciones que no son provincias empiezan en 100 (zonas de Asturias 101-103, islas de Baleares 101-104, islas de Canarias 101-107 y lista autonómica 100, distritos de Murcia hasta 2015 101-105). El código lee los CSV; las tablas `scopes` y `districts` son su réplica en la base de datos (`save_catalogues()`). `data/` usa ya una sola numeración de comunidades, la del INE.

Los umbrales se comprobaron contra la sección *Electoral system* del artículo de la última elección de cada comunidad, que cita el estatuto y la ley electoral (tabla en `data/README.md`). Una candidatura entra en el reparto si supera el umbral de su circunscripción **o** el del conjunto del ámbito; un umbral vacío no se puede superar. Así caben los tres casos que existen: sólo circunscripción (la mayoría), sólo conjunto (Comunidad Valenciana; Murcia hasta 2015) y los dos (Canarias, 15 % insular o 4 % autonómico; Extremadura). Los umbrales históricos distintos del vigente van por evento en `params.json` (Canarias 2011 y 2015: 30 y 6; Murcia 2011 y 2015: 5 autonómico).

### Ingesta

**Sondeos** (`WikipediaLoader`). La tabla es la primera `wikitable` cuya primera fila empieza por "Polling firm" bajo un encabezado que contenga "Voting intention"; si no hay, la que cuelga directamente de "Opinion polls" (artículos pequeños sin subsecciones). Las columnas de partido son las celdas de cabecera con enlace a un artículo y, entre ellas, las que sólo traen un texto o un logo (su clave es ese texto, para que aparezcan como pendientes de mapear en vez de perderse); "Lead" se localiza por su texto, no por posición. Los enlaces se normalizan (llegan unas veces codificados y otras no). Dos columnas del mismo sondeo que acaban en el mismo partido (miembros de una coalición posterior) se suman. Una celda de fecha sin día descarta la fila.

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
- **Estimadores.** El error de sondeo, los escaños por regresión, la deriva y la razón de composición se ajustan con las elecciones del ámbito padre (`es`). `Simulator(min_events=k)` los ajusta con las del propio ámbito cuando tiene al menos `k` `featured` anteriores; la especificación fijaba `k = 3`, y la validación del 06-10-2026 (ver *Resultados*) dejó por defecto los del padre en todos los casos. El estimador de escaños regresa la **cuota** de escaños de la cámara, para que parlamentos de 33 y 350 escaños sean comparables.
- **Efectos de casa globales** (decisión del 04-10-2026). El prior del efecto de cada casa sobre cada partido se construye con sus desviaciones frente al resultado en **todas** las elecciones anteriores, de cualquier ámbito, con el peso de cada ámbito (`rating_weight`: 1 nacional, 0,5 autonómico) multiplicando el de cada elección; el partido se resuelve por su raíz (`parent_id`: PSC y PSE-EE comparten historia con el PSOE, Más Madrid con Más País, Junts con CiU). Así lo medido con el PSOE en las generales corrige al PSC en Cataluña, y Galicia 2024 informa al PSOE de 2027. El mínimo de tres elecciones cuenta pares ámbito-fecha, y el sesgo del sector (opcional) usa la misma historia con los mismos pesos. Efecto: en Madrid 2027 casi todas las casas tienen ya prior (antes ninguna), y el sesgo del sector con el PSOE, que en las generales era de −1 punto, queda cerca de 0 con las autonómicas.
- **Partidos "regionales".** En `es` la marca es la curada en la tabla de partidos. Dentro de una comunidad se deriva de los resultados: un partido es regional cuando sólo concurrió en circunscripciones que suman menos de la mitad del voto válido (UPL, Por Ávila, los partidos de una isla). La especificación decía cero para todos, y con eso un partido del 0,4 % recibía el error de sondeo de uno de ámbito completo (1,3 puntos): Sa Unió perdía el escaño de Formentera en el 37 % de las simulaciones, dejando 52 de 300 con un escaño sin repartir, y Por Ávila lo perdía en el 43 %. Con la marca derivada, ninguna simulación de Baleares queda corta y Por Ávila conserva el suyo en más del 80 %.
- **Ruido del método 9.** Dentro de una comunidad no hay componente por comunidad: sólo el choque por circunscripción, con la curva provincial estimada en `es`. En un distrito único no hay nada que redistribuir, y la lista autonómica canaria, que vota todo el archipiélago, queda fuera del choque.
- **Partidos sin base.** Un partido simulado sin resultado previo ni regla de herencia se queda sin escaños, pero no rompe la proyección. Las reglas (`smap` en `params.json`) cubren los casos claros: SALF hereda de VOX, SUMAR de UP, Más Madrid de UP en 2019, VOX en Cataluña 2021 de PP y Cs, Aliança Catalana de Junts, Por Andalucía de la Adelante de 2018, y Navarra Suma y su ruptura.
- **Errores claros.** Un evento sin sondeos, o con tan pocos que el promedio no se puede ajustar, falla con un mensaje que lo dice, en vez de devolver ceros.

## Resultados

**Reparto determinista a 6 días.** Error absoluto medio de escaños de los partidos principales: Castilla y León 2022, 1,25; Asturias 2023, 0,6; Canarias 2023, 1,4. Madrid 2023 da 3,2: los sondeos ponían a Podemos-IU en el 5,1 % y sacó el 4,76 %, por debajo de la barrera, así que el modelo le da 7 escaños que no tuvo. Sin ese partido el error es 0,5. Es un fallo de pronóstico en el borde de la barrera, no del reparto.

**Próximas elecciones.** De los 17 ámbitos, 11 se pueden simular hoy; 4 no tienen sondeos todavía (Andalucía, Aragón, Castilla y León, Extremadura) y 2 tienen demasiado pocos para ajustar el promedio (Cantabria 3, La Rioja 4).

**Ratings.** Con los sondeos autonómicos al 0,5, las casas con sondeos evaluables en el último rating de `es` pasan de 53 a 84. Entre las que ya tenían, el rating cambia 4,9 puntos de media en valor absoluto y la correlación con el anterior es 0,96. Los mayores cambios: GAD3 +18,9 (de 15 a 74 sondeos evaluables), CIS −16,9 (de 16 a 80), Target Point +14,5, 40dB −14,2, Data10 +13,6, CEMOP +13,4, Simple Lógica −10,8, InvyMark +10,8, Sigma Dos −10,6, Ipsos −10,6 y GESOP −10,4. Sondaxe no se mueve (+0,1). Las casas nuevas entran entre 5 y 30 (Gizaker 30,4 con 13 sondeos). La tabla completa está en `files/stage/m11_ratings_impact.csv`. Luis decidió el 04-10-2026 adoptar el **ranking global**: todas las elecciones alimentan los errores, sesgos y rating de cada casa (nacionales con peso 1, autonómicas con 0,5) y ese rating se aplica a todas. Está guardado para los 17 ámbitos y para `es`, con `pollsters.rating` actualizado; `PollsCompute` y `run_load` lo calculan así por defecto. La regresión de referencia de `tests/integration/test_model.py` pasa de 140 / 109 / 61 / 8 a 141 / 107 / 62 / 8 (PP / PSOE / VOX / SUMAR, MT 200 simulaciones). Con el mismo criterio, el prior de los efectos de casa (método 6) también es global (ver *Simulator › Efectos de casa globales*).

**Backtest.** `python backtest/run_backtest.py --scopes all --horizons 6 30 --n-sim 500` evalúa las elecciones `featured` de cada ámbito desde 2019: 29 casos por horizonte en los 17 ámbitos (`backtest/results/{scope}/`). Entre uno y tres casos por ámbito es poco para juzgar cada comunidad por separado; el conjunto sí dice algo. Ejecución de referencia del 06-10-2026 con el código de los métodos 1 a 11 y los estimadores de `es` en todos los casos (ver *Validación del mínimo de elecciones*); la columna de `es` es la ejecución del mismo día.

| Conjunto de los 17 ámbitos | 6 días | 30 días | `es` a 6 días (referencia) |
|---|---|---|---|
| MAE de cuotas, modelo / último sondeo / media de 4 semanas | 1,84 / 1,84 / 1,89 | 2,38 / 2,57 / 2,32 | 1,64 / 1,74 / 1,86 |
| Cobertura de cuotas al 50 / 80 / 95 % | 0,46 / 0,76 / 0,95 | 0,45 / 0,76 / 0,95 | 0,48 / 0,78 / 0,96 |
| MAE de escaños por partido principal (casos válidos) | 1,90 (24) | 2,32 (26) | 6,27 (sobre 350) |
| Cobertura de escaños al 50 / 80 / 95 % | 0,59 / 0,81 / 0,95 | 0,55 / 0,78 / 0,91 | 0,61 / 0,84 / 1,00 |
| Cobertura de cuotas por circunscripción al 50 / 80 / 95 % | 0,50 / 0,78 / 0,94 | 0,47 / 0,74 / 0,92 | 0,58 / 0,87 / 0,94 |
| Brier de la mayoría absoluta de bloque | 0,055 | 0,048 | 0,033 |

Los intervalos de cuotas están tan bien calibrados como en las generales, y los de escaños también. El promedio no mejora al último sondeo a 6 días (en las autonómicas hay menos sondeos y más recientes), y a 30 días queda entre las dos líneas base. Un caso queda sin métricas de escaños cuando un partido con escaño carecía de base para proyectarse (Teruel Existe 2023, Soria ¡Ya! 2022) y también en Navarra 2023, donde el modelo sí proyecta a UPN y PP con la regla `split` de Navarra Suma pero la comprobación de huérfanos del backtest no reconoce ese tipo de regla.

Por ámbito, a 6 días:

| Ámbito | Elecciones | MAE cuotas | Cobertura 50 / 80 / 95 | MAE escaños |
|---|---|---|---|---|
| `es-an` | 2022, 2026 | 1,74 | 0,50 / 0,70 / 1,00 | 2,8 |
| `es-ar` | 2023, 2026 | 1,34 | 0,66 / 0,80 / 1,00 | 1,1 |
| `es-as` | 2023 | 1,42 | 0,60 / 0,80 / 1,00 | 0,6 |
| `es-cb` | 2023 | 1,45 | 0,60 / 1,00 / 1,00 | 1,2 |
| `es-cl` | 2022, 2026 | 1,31 | 0,40 / 0,74 / 0,87 | 0,9 |
| `es-cm` | 2023 | 2,07 | 0,00 / 1,00 / 1,00 | 1,3 |
| `es-cn` | 2023 | 1,83 | 0,40 / 0,80 / 1,00 | 2,0 |
| `es-ct` | 2021, 2024 | 1,09 | 0,69 / 0,88 / 0,94 | 2,1 |
| `es-ex` | 2023, 2025 | 1,73 | 0,62 / 0,75 / 1,00 | 1,5 |
| `es-ga` | 2020, 2024 | 2,09 | 0,38 / 0,75 / 0,75 | 2,4 |
| `es-ib` | 2023 | 2,51 | 0,17 / 0,67 / 0,83 | 2,2 |
| `es-mc` | 2019, 2023 | 3,82 | 0,00 / 0,62 / 1,00 | 3,5 |
| `es-md` | 2019, 2021, 2023 | 1,21 | 0,71 / 0,83 / 1,00 | 2,5 |
| `es-nc` | 2019, 2023 | 2,52 | 0,29 / 0,36 / 0,93 | — |
| `es-pv` | 2020, 2024 | 1,29 | 0,75 / 0,83 / 1,00 | 1,0 |
| `es-ri` | 2023 | 2,10 | 0,40 / 0,80 / 1,00 | 0,4 |
| `es-vc` | 2019, 2023 | 2,22 | 0,17 / 0,83 / 0,92 | 2,7 |

El criterio de la especificación (coberturas dentro de ±0,15 del nominal en los ámbitos con tres o más elecciones) sólo es aplicable a Madrid, el único con tres casos desde 2019: cumple al 80 y al 95 % (0,83 y 1,00) y se pasa de cobertura al 50 % (0,71). Murcia es el peor ámbito (MAE 3,82, ninguna cuota dentro del intervalo del 50 %): en 2019 el promedio daba a Cs seis puntos de más (18,1 frente a 12,0) y al PP cinco de menos (27,1 frente a 32,4).

**Validación del mínimo de elecciones (06-10-2026).** La especificación fijaba en 3 las elecciones `featured` anteriores a partir de las cuales un ámbito ajusta sus estimadores (error de sondeo y deriva; en modo MT el de escaños no interviene) con sus propias elecciones en vez de con las de `es`. Para validarlo, cada uno de los 29 casos del backtest se ejecutó dos veces con el mismo código y la misma semilla (500 simulaciones, 6 y 30 días): forzando los estimadores propios (`--min-events 1`) y forzando los de `es` (`--min-events 99`). Con las dos variantes de cada caso, cualquier política "propios a partir de k elecciones" se evalúa sin volver a simular. El promedio puntual es el mismo en las dos (los estimadores sólo cambian la anchura de los intervalos y la deriva), así que el MAE de cuotas no distingue; se comparan el CRPS de escaños y las coberturas. Métricas agregadas sobre los partidos principales, propios / `es`:

| Elecciones previas | Casos | CRPS escaños, 6 d | Cobertura cuotas 95 %, 6 d | Anchura 95 % cuotas, 6 d | CRPS escaños, horizonte 30 d | Cobertura cuotas 95 %, horizonte 30 d |
|---|---|---|---|---|---|---|
| 1 | 3 | 0,61 / 0,46 | 0,82 / 0,94 | 9,3 / 7,8 | 1,48 / 1,40 | 0,76 / 0,76 |
| 2 | 12 | 1,10 / 1,11 | 0,98 / 0,95 | 9,4 / 8,9 | 1,21 / 1,19 | 0,95 / 1,00 |
| 3 | 8 | 1,68 / 1,66 | 0,93 / 0,93 | 9,4 / 8,7 | 2,49 / 2,45 | 0,83 / 0,93 |
| 4 o 5 | 6 | 1,26 / 1,25 | 0,97 / 0,97 | 8,1 / 8,1 | 1,83 / 1,81 | 0,94 / 0,94 |
| Todos | 29 | 1,26 / 1,25 | 0,95 / 0,95 | 9,1 / 8,6 | 1,68 / 1,66 | 0,90 / 0,94 |

A 6 días la única diferencia es el estimador de error: el CRPS es el mismo (diferencia media +0,015 escaños, intervalo bootstrap del 95 % por casos [−0,012, +0,042]) y los intervalos propios son un 6 % más anchos sin ganar cobertura. A 30 días entra la deriva, y la ajustada con uno a cinco ciclos autonómicos es demasiado pequeña: los intervalos propios son un 3 % más estrechos, la cobertura del 95 % de las cuotas baja de 0,94 a 0,90 (diferencia −0,046, intervalo [−0,104, 0,000]) y el CRPS empeora (+0,035, intervalo [+0,007, +0,064]). Los propios ganan en CRPS en 10 de 24 casos a 6 días y en 9 de 26 a 30 días. Por tramos, ningún valor de k cumple el criterio fijado de antemano (no empeorar el CRPS ni alejar las coberturas del nominal en los casos con k o más elecciones): falla para k de 1 a 4 y sólo lo cumple k = 5, con un único caso (Cataluña 2024). Agregando las políticas sobre los 29 casos, el CRPS a 6 días va de 1,264 (k = 1) a 1,253 (siempre `es`) y a 30 días de 1,682 a 1,655, con la cobertura del 95 % a horizonte de 0,895 a 0,941: la configuración "siempre `es`" es la mejor o empata en todas las métricas. **Decisión:** `Simulator(min_events=None)` por defecto, es decir, los estimadores del padre en todos los ámbitos autonómicos; `--min-events` del backtest queda para repetir la comparación cuando haya más elecciones. La opción híbrida (error propio con suelo en el del padre) no hace falta: el estimador de error propio no aporta nada medible. Tablas completas de la comparación en `files/backtest/min_events_summary/`.

## Salvedades

- **Formato de Wikipedia.** El cargador depende de la estructura de los artículos. Los tests usan una instantánea del 01-10-2026 de seis artículos (`tests/fixtures/wikipedia/`); `load/run_load.py` sin `--save` enseña qué ha cambiado antes de escribir nada.
- **Votos y cuotas estimados.** Los votos por circunscripción y las cuotas por circunscripción de las candidaturas sin escaño son estimaciones. Pesan en la agregación entre circunscripciones y en el voto en blanco, no en el reparto.
- **Partidos locales sin sondeos.** Los sondeos de la próxima elección canaria no listan a ASG ni a AHI, así que el modelo no les da los escaños de La Gomera y El Hierro. Lo mismo vale para cualquier partido de una sola circunscripción que las casas no publiquen.
- **Recién llegados territoriales.** Teruel Existe en 2023, Soria ¡Ya! en 2022, Democracia Ourensana en 2024 y AHI en 2023 no tenían geografía previa ni partido del que heredarla: la proyección no les da escaños.
- **Estimadores propios con pocos datos.** Con tres o cuatro elecciones, el estimador de error del ámbito es ruidoso (en Baleares deja a un partido regional del 1,5 % con un error de 0,04 puntos) y la deriva ajustada con sus ciclos es demasiado pequeña. Por eso no se usan por defecto; `--min-events` del backtest permite repetir la comparación cuando haya más elecciones.
- **Peso 0,5.** Es un valor de partida (D3). El cambio de rating de las casas grandes al adoptar el ranking global es considerable (GAD3 +19, CIS −17); conviene calibrarlo con el backtest de ratings.
- **Sondeos con empate en cabeza.** Se descartan, también en las generales. Son 15 en las autonómicas.
- **Umbrales.** En Extremadura el umbral autonómico alternativo exige concurrir en las dos provincias; no se modela. En la Comunidad Valenciana la base legal del 5 % son los votos emitidos, no los válidos. En Canarias, hasta 2015, también entraba la lista más votada de cada isla; tampoco se modela.
- **Sondeos encargados por partidos** (D7). Se descartan los sondeos cuya "casa" es un partido (31). Los encargados por un partido a una casa (Celeste Tel/PSOE, Sigma Dos/PP…) se cargan como cualquier otro, sin patrocinador: 76 en los ámbitos autonómicos (57 entran en los ratings, 14 son de eventos próximos) y 13 en `es`, donde siempre se cargaron así.
- **Resultados de referencia.** `backtest/results/` (`es`, seis horizontes) y `backtest/results/{scope}/` (6 y 30 días) son las ejecuciones del 06-10-2026 con el código de los métodos 1 a 11, los estimadores de `es` en los ámbitos autonómicos y la base de datos de ese día (3.836 sondeos nacionales). Los documentos de los métodos anteriores conservan las cifras de la ejecución de cada uno; con los métodos 10 y 11 el backtest de `es` cambia menos de 0,05 puntos de MAE de cuotas y de 0,07 de CRPS de escaños en todos los horizontes.
- **Caché del cargador por lotes.** `load/run_load.py` en seco lee la copia local de los artículos; con `--save` los descarga de nuevo salvo que se pida `--cached`.

## Fuera de alcance

Transferencia de los sondeos autonómicos al pronóstico provincial de las generales; calibración de `rating_weight` por comunidad (afecta al rating y al prior de efectos de casa); curva de ruido por circunscripción propia de cada comunidad; elecciones municipales y europeas.
