# Web, Análisis: la pregunta 3 del voto CERA, sobre la predicción del modelo para el 29-N

Proyecto mediano. Luis lo pidió el 2026-11-24, primer día de la veda («Quiero actualizar el análisis del Voto CERA,
partiendo de las predicciones del modelo para el 29-N, en lugar de los resultados de 2023») y decidió el resto en la
conversación y sobre una maqueta. Lo que es suyo está en «Decisiones de Luis»; lo que no, en «Decisiones mías, por
revisar». Maqueta (décima versión: escaños y probabilidad a la vez, con los cambios de Luis sobre la novena;
sólo de escritorio; cifras reales de la prueba):
`docs/superpowers/specs/2026-11-24-web-analisis-voto-cera-mayoria-mockup.html`. Diseño original del estudio:
`docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-design.md`.

## Intención

En palabras de Luis: «Quiero que vea qué probabilidades hay de que, con el voto CERA, la Izquierda pueda revertir la
mayoría absoluta de la Derecha. Por varias vías: más participación, más voto a la Izquierda, colocación de votos en
circunscripciones clave, aunque esta última es más difícil.»

La pregunta 3 de `/analisis/voto-cera` suma hoy el voto de los electores nuevos a los **resultados de 2023** y
cuenta escaños que cambian. A cinco días de la elección el lector ya no se pregunta por 2023: se pregunta si el voto
exterior puede quitarle a la derecha la mayoría absoluta que el modelo le da. La tarjeta pasa a responder eso, con
las mismas tres palancas, y deja de hablar de escaños de 2023.

Sobre la maqueta, Luis añadió la otra mitad: «es muy interesante ver qué probabilidades tiene la Izquierda de darle
la vuelta según el escenario de dos dimensiones, pero también la gente entiende muy bien el concepto "podría
arrebatar X escaños a la Derecha en tal y cual caso"». Por eso cada escenario dice dos cosas a la vez: cuántos
escaños le quita a la derecha partiendo del reparto central del modelo y con qué probabilidad la derecha conserva
la mayoría en las 10.000 simulaciones.

## Hechos de partida (2026-11-24)

- **El estudio** (`mtpy/lib/studies/cera/`, `compute_cera(data)`) se calcula sólo con ficheros de `data/`, sin base
  de datos, y `publish study` lo publica en ocho partes (`bundle.STUDY_PARTS['voto-cera']`). La pregunta 3 sale de
  `prediction` (modelo de incremento sobre 2023 en dos censos, último escaño) y `sensitivity` (rejilla de todo el
  censo exterior, tabla de censo × participación × sesgo, colocación óptima), y la pintan `pages/cera._question_3`,
  `web/templates/analisis_voto_cera.html` (tarjeta `#p3`) y `web/dist/js/pages/analisis-cera.js` (gráfico de
  brechas y radar).
- **La predicción está registrada** desde el 2026-10-20 (commit 23b02c5): «El voto CERA de las generales del 29 de
  noviembre de 2026 no dará al PSOE ni a la izquierda un saldo neto de más de 1 escaño». Se comprueba con la parte
  `check` cuando exista el escrutinio CERA del 29-N. «Lo que dice el debate», «Comprobación», el caso de Madrid y
  la fila de KPI leen también `prediction` y `sensitivity`.
- **El modelo.** El run publicado de `es` (hoy `20261124-102045`, datos hasta el 2026-11-23, 10.000 simulaciones,
  semilla 42) da a la derecha (PP, VOX y SALF, los bloques de la web: `meta.bmaps.blocks`) 190 escaños de reparto
  central y mayoría absoluta en 8.734 simulaciones (87,3 %). El bundle publica medianas, reparto central,
  probabilidades (redondeadas a tres decimales) y, en la parte `dist`, los escaños nacionales de cada partido en
  cada simulación; **no** publica ni los votos ni los escaños por provincia de cada simulación.
- **El run se puede rehacer en local** simulación a simulación (78 s): el `Simulator` con los parámetros del
  `meta` del run y `today` igual a su `as_of` (el `meta` no guarda `today`, y sólo sirve para fijar ese ancla),
  `fit_forecast`, modo `forecast` y `run(split=True, random=True, n_sim=10000, horizon='deadline')`, el mismo
  camino de `publish.publish_forecast`. Saltarse el run del modo `nowcast` no cambia nada: cada `run` vuelve a
  sembrar. Cada simulación guarda el voto por provincia y partido (`units[..., vpred_pct]`) y sus escaños
  (`results`). Los votos que el simulador reparte son ese porcentaje por la base de la provincia,
  `sim.prev_totals['votes']`, que **no** son los votos válidos de 2023 tal cual: desde el método 19 van escalados
  por el cambio del censo de residentes (`apply_census`).
- El run del 23-N no está en el bundle local (`files/site/v1/`): hay que leerlo del bundle publicado.
- **El reparto central** que la web da como titular de los escaños (`Simulator.central_result()`, la parte
  `scenario`, con el total y todas las provincias) es una sola pasada sin sorteo: D'Hondt sobre el voto central de
  cada provincia (`build_frame` → `build_umat` → `simulate` con `random` apagado). Hoy: derecha 190, izquierda 129.
- **La web da sus rangos al 95 %** («Rango 95%»): cuantiles empíricos a `alpha / 2` y `1 − alpha / 2`, con `alpha`
  0,05 (`quantile`).
- **Prueba hecha** (scripts de usar y tirar): un reparto D'Hondt vectorizado sobre esos votos reproduce las 10.000
  simulaciones guardadas; aplicar 63 escenarios de voto exterior a las 10.000 lleva 12 s. El mismo reparto sobre el
  voto central (el `vpred_pct` de ese `build_umat` sin sorteo por la base de la provincia) da los escaños de
  `central_result()` en las 52 provincias. Las cifras están en «Cifras de la prueba».
- Con los resultados del 29-N cargados en la base, o con un sondeo anterior al 23-N corregido o cargado tarde, no
  hay garantía de poder rehacer ese run.

## Decisiones de Luis

1. La pregunta parte de la predicción del modelo para el 29-N, no de 2023.
2. Lo que se mide: la probabilidad de que el voto CERA le quite la mayoría absoluta a la derecha, por tres vías:
   más participación, más voto a la izquierda y colocación en circunscripciones clave.
3. Colocación: «deja sólo la lectura a ciegas, aplicando el histórico». No hay lectura «sabiendo el resultado».
4. «Quiero dejarlo ya fijo con el run del 23-N, no se cambia tras las elecciones, más allá de mostrar el resultado
   real.»
5. Construcción: «Vamos con la A», la foto fija de los resultados (ver «Enfoques»).
6. Titular de la tarjeta: «¿Puede el voto CERA arrebatar la mayoría absoluta a la derecha el 29-N?».
7. Las casillas de las rejillas dicen la probabilidad de mayoría que le queda a la derecha.
8. La primera tabla enseña el censo CERA total, además de votantes y votos.
9. Las barras van en porcentaje de las simulaciones, no en «tantas de 10.000».
10. Se mantiene la tabla de «las tres dimensiones a la vez», reducida a dos: participación y sesgo.
11. Aceptado con «lo demás ok»: la medida se toma sobre las 10.000 simulaciones del run; los ejes de participación
    y voto a la izquierda son los que la página ya usa; el registro del 20-10 no se toca; se publica en la veda con
    el run del 23-N, como el resto de la web.
12. Vio en la maqueta lo que desaparece de la tarjeta (tabla de escaños 2023 → 29-N, gráfico de brechas de 2023,
    radar, colocación «sabiendo el resultado») y sólo pidió recuperar el censo y la tabla de dimensiones.
13. Sobre la cuarta maqueta: el embudo no lleva la barra «Simulaciones», que siempre es el 100 %.
14. No hay selector de censo: sólo el de la convocatoria, «con el aviso de los del litigio».
15. La tabla de las dos dimensiones sólo lleva la probabilidad de mayoría, sin los cambios medios de escaños.
16. La colocación no lleva tabla («creo que sólo estorba»): queda como un comentario.
17. Preguntó qué significa que la derecha «gane» la mayoría en 2 simulaciones. No es un error: el voto exterior que
    se espera no va sólo a la izquierda (ver D6). La forma de contarlo es mía.
18. Cambio de concepto, sobre la quinta maqueta: enseñar a la vez los escaños que cada escenario podría mover hacia
    la izquierda partiendo de la simulación central del modelo y la probabilidad de cada escenario en las 10.000
    simulaciones. Matiza la 7 y la 15: las casillas y la tabla de dos dimensiones llevan las dos cifras.
19. Propuso calcular los escaños también desde otros percentiles, «por ejemplo, desde el 25 % más hacia la
    Izquierda». Visto con cifras, lo que un escenario quita casi no depende de dónde parte la derecha (ver «Cifras
    de la prueba»); lo que cambia es cuántos escaños le hace falta perder. El percentil se usa para eso: 15 desde
    el reparto central y 6 desde el percentil 25. La propuesta es mía; Luis la aceptó con la sexta maqueta.
20. «Sí, me encaja, rehaz el diseño con la v6»: aceptó el gráfico que une las dos lecturas y que el titular de cada
    escenario sean los escaños desde el reparto central, con el rango de las simulaciones al lado.
21. Sobre la séptima maqueta: en «¿Qué haría falta?» quedan sólo dos gráficos, el de las barras rojas y la rejilla,
    uno al lado del otro. La tabla de las dos dimensiones sale de la tarjeta (revierte la 10 y la 15).
22. En el gráfico de las barras rojas, títulos cortos: «30 % voto + 70 % izq.» mejor que «Todo el censo exterior».
23. Ese gráfico lleva un eje con una marca cada 2 escaños hasta 16, y sus dos líneas verticales se llaman «si la
    derecha suma 181» y «si la derecha suma 190».
24. El aviso del litigio ocupa todo el ancho de la tarjeta y se limita a decir que el censo incluye las 163.020
    inscripciones en litigio, «y es que hemos querido tener en cuenta todos los escenarios, incluidos los más
    extremos». Ya no da las cifras del censo sin litigio (matiza la 14).
25. La primera tabla sólo lleva el censo total y los votantes; fuera los votos a candidaturas (matiza la 8).
26. Las barras del embudo no le acaban de gustar: «Busquemos otra manera de mostrar los porcentajes de simulaciones
    que cambian escaños». La octava maqueta trajo tres maneras: 10.000 puntos, tres cifras y cuadrados anidados.
27. Sobre la octava maqueta: «Me quedo con la B, las 3 cifras, pero en la última diciendo que sólo en el 0,07% se
    cambia el sentido de la mayoría […], sin dar los componentes». Cierra la 26, descarta las otras dos maneras y
    matiza la 17: la tarjeta ya no dice en cuántas simulaciones la derecha pierde la mayoría y en cuántas la
    consigue.
28. En «¿Qué haría falta?»: «daría más anchura al de la izquierda y menos al de las casillas, acortando su columna
    de títulos».
29. Sobre la novena maqueta («Vamos por mejor camino»), los escenarios de los electores nuevos en el gráfico de las
    barras: el primero, «en vez de lo que se espera», es «9 % voto y X % izq. (como 2023)», y el del centro, en
    vez del triple de voto con el 90 %, «vota el 50 % de ellos y el 80 % lo hace a la izquierda». La X la pongo
    yo: 50 % (en 2023 fue el 49,6 %).
30. En la parte de todo el censo, los dos primeros son «30 % voto + 60 % izq.» y «30 % voto + 80 % izq.»; los otros
    dos, iguales.

## Objetivos

- La tarjeta `#p3` responde con cifras del modelo y de cada escenario dice dos cosas: los escaños que le quita a la
  derecha desde el reparto central y la probabilidad de que la derecha conserve la mayoría en las simulaciones.
  Lo dice del voto CERA que se espera y de cada escenario de las tres palancas.
- Todo lo que enseña sale de una foto fija tomada con el run del 23-N, publicada como una parte más del estudio.
- Después del 29-N la tarjeta enseña además lo que pasó, sin recalcular nada.
- `publish study` sigue sin tocar la base de datos y da siempre el mismo resultado.

## No objetivos

- No se cambia el pronóstico: el voto CERA no entra en el modelo ni en `/es`. Es un análisis de sensibilidad sobre
  las simulaciones ya publicadas.
- No se tocan las preguntas 1 y 2, «Lo que dice el debate», «Comprobación» ni las partes `prediction`,
  `sensitivity` y `check` del bundle, que siguen sosteniendo la predicción registrada.
- No se vuelve a registrar nada: `registered_at`, `registered_commit` y `registered` quedan como están.
- No se guardan las simulaciones (enfoque B, descartado por Luis).
- No se añaden ejes ni escenarios después de hacer la foto.

## Anti-objetivos

- Una cifra de la tarjeta escrita a mano o calculada fuera de la foto.
- Una foto tomada sobre un run que no es el publicado: si el run rehecho no coincide con el del bundle, no se
  escribe nada.
- Que la tarjeta cambie de cifras después del 29-N, o que una publicación posterior del estudio las mueva.
- Que el cambio altere la predicción registrada o su comprobación.
- Presentar el resultado como una previsión del voto CERA: son escenarios («si votaran así»), y la colocación es
  un límite, no un plan que alguien pueda ejecutar.
- Dar los escaños del reparto central como si fueran seguros: es una sola configuración de las muchas que el
  modelo simula, y por eso va siempre con el rango de las simulaciones al lado.

## Restricciones

- Código dentro de `mtpy` (lib, pipelines, jobs, controladores): ni scripts sueltos ni FastAPI. PEP8, docstrings
  en todas las funciones, identificadores en inglés neutro; el español queda en los textos públicos.
- Textos públicos: «sólo» con tilde, «encuestadora» o «casa», nunca «empresa»; pasan `assert_public_wording`.
- El estudio queda fuera de la guardia de la LOREG (no publica sondeos), como hoy.
- La foto hay que hacerla **antes del 29-N** y en local, porque lee la base (los sondeos) y escribe un fichero del
  repositorio que se versiona.
- Luis trabaja y lanza jobs en el mismo árbol: antes de tocar ficheros compartidos o de hacer commit se mira
  `git status`, `git log` y los jobs en marcha. El despliegue y la publicación los hace él.

## Enfoques

- **A. Foto fija de los resultados (elegido por Luis).** Una acción de job rehace el run del 23-N, comprueba que
  coincide con el publicado, aplica todos los escenarios y escribe un fichero pequeño y versionado en `data/cera/`
  con los recuentos. El estudio lo lee como el resto de sus entradas. A cambio, los ejes quedan fijados al hacer la
  foto: una casilla distinta ya no se podrá calcular después del 29-N.
- **B. Foto fija de las simulaciones.** Guardar las 10.000 simulaciones por provincia y calcular los escenarios en
  cada publicación. Permite añadir o auditar escenarios más adelante, a cambio de un binario de 10–13 MB en el
  repositorio. Luis no lo quiso: lo quiere fijo y la página sólo necesita las casillas.
- **Descartado: recalcular desde la base en cada publicación.** Tras el 29-N no rehace el run del 23-N y haría
  depender al estudio de la base, que hoy no necesita.

## Diseño

### D1. La medida: las mismas simulaciones, con y sin el voto exterior

Para cada simulación `s`, provincia `r` y partido `p` del modelo, el simulador reparte unos votos `V[s, r, p]`.
Un escenario es una matriz `Δ[r, c]` de votos que se suman (52 provincias × los partidos del modelo más el resto
sin escaño). Se recalcula D'Hondt en cada provincia con `max(V + Δ, 0)`, con el umbral del 3 % sobre los votos
válidos que usa el simulador más los votos añadidos, y se compara con los escaños **de esa misma simulación** sin
añadir nada. Es una comparación emparejada: las diferencias no llevan ruido de simulación.

Cada escenario se lee de dos maneras, con la misma `Δ`:

- **Sobre el reparto central**: `Δ` se suma a los votos centrales de cada provincia (los que hay detrás de
  `central_result()`: su voto central por la base de la provincia) y se repite D'Hondt con las mismas reglas. Da un
  número de escaños, «le quita 3 a la derecha», que es el titular del escenario.
- **Sobre las 10.000 simulaciones**: la comparación emparejada de arriba. Da la probabilidad de mayoría y el rango
  de lo que le quita.

Las dos no tienen por qué coincidir. El reparto central es una sola configuración de restos: un escaño que en ella
queda lejos puede estar cerca en la mayoría de las simulaciones, y al revés. Con el 50 % de participación y el
70 % del voto a la izquierda, el reparto central pierde 3 escaños y la mediana de las simulaciones 5; con el 70 % y
el 70 %, 9 y 7. En la prueba difieren en 24 de 62 escenarios, por uno o dos escaños, y el reparto central queda
siempre dentro del rango del 95 %. Por eso el rango va siempre al lado del titular.

- Bloques: los del run (`meta.bmaps.blocks`), derecha = PP, VOX y SALF; izquierda = PSOE, FA y UP. Mayoría: la del
  run (`meta.majority`, 176).
- De 2023 al modelo: el voto exterior se reparte entre candidaturas de 2023 y hay que llevarlo a los partidos del
  29-N. Cada candidatura va al partido del modelo con su misma sigla; SUMAR va a FA y UPN a PP; las siglas que el
  modelo no simula van al resto, que cuenta para el umbral y no recibe escaños. UP y SALF no reciben voto exterior:
  en 2023 no tuvieron voto CERA propio.
- Antes de aplicar ningún escenario, el reparto vectorizado con `Δ = 0` tiene que dar los escaños guardados en las
  10.000 simulaciones, y el reparto central con `Δ = 0`, los de `central_result()` en cada provincia. Si no, la
  foto no se hace.
- Precisiones que la prueba dejó fijadas: la base de votos y del umbral es `sim.prev_totals['votes']`, nunca los
  totales de 2023 del estudio; `Δ` se suma sin redondear; la base del umbral es esa base más la suma neta de `Δ` en
  la provincia (con el resto, antes de recortar a cero); el sesgo (`_biased_split`) se aplica a las candidaturas
  de 2023 con los bloques del estudio y sólo después se pasa a los partidos del modelo; una candidatura sin sigla
  va al resto.

### D2. Los escenarios

Todos reutilizan las funciones y los parámetros del estudio (`seats.py`, `stress.py`, `data/cera/params.json`).

1. **Voto que se espera** (`expected`), en los dos censos de hoy (`convocatoria` y `sin_litigio`): `Δ` es el
   `extra` del modelo de incremento (`new_electors` + `increment_votes`: los electores nuevos desde 2023, con la
   participación por origen de `params.participation` y el reparto del escrutinio CERA de 2023 de cada provincia).
   Además del total, se guarda por provincia: en cuántas simulaciones cambia su reparto, en cuántas la derecha
   pierde escaño allí y en cuántas lo gana. La página sólo enseña el censo de la convocatoria; el otro queda en la
   foto y en los CSV.
2. **Sólo los electores nuevos** (`new_electors`): la rejilla de todo el censo, con sus mismos ejes
   (`GRID_PARTICIPATIONS` × `GRID_LEFT_FRACS`), aplicada sólo a los electores nuevos desde 2023 (censo de la
   convocatoria). `Δ` es `increment_votes` con la misma participación para los dos orígenes y el reparto
   `_biased_split(empirical_split(cera_2023), f)`: la parte del voto a izquierda o derecha que va a la izquierda.
   Veinte casillas. El gráfico de escenarios enseña dos: 9 % y 50 % («como 2023»: votó el 9 % del censo exterior
   y a la izquierda fue el 49,6 % de lo que se repartieron los dos bloques) y 50 % y 80 %. Las veinte quedan en
   la foto y en los CSV. Sustituye a las nueve filas de múltiplos de la participación que se espera (× 1, × 2 y
   × 3 con sesgo empírico, 70 % y 90 %), que la tarjeta ya no usa.
3. **Todo el censo exterior** (`grid`): `GRID_PARTICIPATIONS` × `GRID_LEFT_FRACS`, los ejes de hoy. `Δ` quita el
   voto CERA de 2023 de cada candidatura y pone el del escenario: censo de la provincia × participación × 0,99 ×
   `_biased_split(cera_2023, f)`. Es la definición de `stress.exterior_grid`, sobre las simulaciones.
4. **Colocación a ciegas** (`placement`): `PLACEMENT_PARTICIPATIONS` × `PLACEMENT_FRACS`. El plan se decide con
   los resultados oficiales de 2023, con el algoritmo de `stress._greedy_seats` (los votos útiles, bolsa ×
   participación × fracción × 0,99, van de 500 en 500 a la provincia donde al PSOE le costaba menos el siguiente
   escaño), y da unos votos por provincia. `Δ` son esos votos, sumados al PSOE en cada simulación del 29-N. La
   bolsa es la de `placement_pool` (181.032 electores). `_greedy_seats` hoy sólo devuelve escaños: hay que sacar de
   él los votos colocados por provincia sin cambiar lo que `optimal_placement` publica.

De cada escenario se guardan recuentos enteros sobre las 10.000 simulaciones: votos añadidos, simulaciones en las
que la derecha pierde la mayoría (`lost`), en las que la gana (`gained`), en las que cambia algún escaño
(`moved`), y la distribución del cambio de escaños de la derecha y de la izquierda (cuántas simulaciones con −2,
−1, 0, +1…), de la que salen las medias y el «cambian el total de la derecha». Y, del reparto central, el cambio
de escaños de la derecha y de la izquierda.

El rango que la tarjeta enseña no se guarda aparte: son los cuantiles 2,5 % y 97,5 % del cambio de escaños de la
derecha, que salen de la distribución guardada con la misma cuenta que los rangos del resto de la web.

### D3. La foto fija: `data/cera/majority.json`

Un JSON legible, escrito sólo por la acción de D4, con:

- `run`: ámbito, fecha de la elección, id del run, `as_of`, `n_sim`, semilla, y el commit y el día de la foto.
- `blocks`, `majority` y el mapa de siglas de 2023 a partidos del modelo que se aplicó.
- `base`: simulaciones con mayoría de la derecha, los escaños de los dos bloques en el reparto central y el
  histograma de los escaños de la derecha en las 10.000 simulaciones. De aquí salen las dos referencias del
  gráfico (cuántos escaños tendría que perder la derecha desde el reparto central, 15, y desde su percentil 25,
  6) y las simulaciones en las que no tiene mayoría ni antes de contar el voto exterior.
- `expected` (por censo): las cifras de la tabla (censo, electores nuevos, votantes y votos de 2023 y
  proyectados), los recuentos de D2, el cambio del reparto central y las 52 provincias.
- `new_electors` (veinte casillas), `grid` y `placement` (ejes, recuentos y cambio del reparto central de cada
  escenario), con la bolsa de la colocación.

La tarjeta entera sale de este fichero, también la tabla de censo, votantes y votos, para que un cambio posterior
de las entradas del estudio no deje la tabla y las casillas hablando de cosas distintas. No lleva probabilidades
redondeadas: la página las calcula de los recuentos.

**Esto es todo lo que quedará del run del 23-N.** Lo que no esté aquí no se podrá calcular después del 29-N.

### D4. La acción que hace la foto

Una acción de job dentro de `mtpy` (propongo el job `load`, que ya tiene el precedente de `params`: escribe un
fichero versionado de `data/` a partir de la base), que recibe el id del run (por defecto, el `latest` del ámbito)
y:

1. Lee del bundle publicado el `meta` del run y rehace el run con sus parámetros, por el mismo camino que
   `publish.publish_forecast` construye y corre el simulador (no una copia de la receta).
2. Comprueba que el run rehecho es el publicado comparando, con los mismos exportadores, la parte `dist` del modo
   `forecast` (los escaños nacionales de cada partido en cada una de las 10.000 simulaciones, iguales una a una) y
   el `scenario` (el reparto central). El `summary` no sirve para esto: va redondeado. De la `dist` publicada sale
   también la base: hoy, 8.734 simulaciones con mayoría de la derecha y 190 escaños de reparto central.
3. Comprueba el reparto vectorizado de D1 y el reparto central con `Δ = 0`, provincia a provincia.
4. Toma del estudio censos, repartos, bolsa y parámetros, y aplica D2. La acción no puede depender del fichero que
   va a escribir: las entradas del estudio se calculan sin la foto.
5. Escribe el fichero. Pide árbol limpio (apunta el commit), no pisa un fichero existente sin `overwrite` y con
   `dry_run` hace todo menos escribir.

Si 2 o 3 fallan, termina `refused` con el motivo y no escribe. El cálculo emparejado es un módulo puro del
paquete del estudio (matrices de votos y de escaños dentro, recuentos fuera), que se prueba sin base de datos; lo
que toca la base y el bundle queda en la acción.

### D5. La publicación: una parte nueva, `majority`

- `compute_cera` lee y valida el fichero y lo lleva en `CeraStudy`. El fichero pasa a ser una entrada obligatoria
  del estudio: sin él, `publish study` falla con un mensaje claro. Por eso la foto entra en la rama antes que este
  paso (ver «Orden de construcción»).
- `bundle.STUDY_PARTS['voto-cera']` gana `majority` (esquema `study-majority`), con su exportador y sus gemelos
  CSV (rejilla de todo el censo, rejilla de los electores nuevos, colocación y provincias). Cada escenario lleva
  en su CSV el cambio del reparto central y el rango del 95 %. `docs/web/contrato.md` lo recoge.
- La parte lleva lo de D3 más `outcome`, nulo hasta que exista la comprobación del 29-N (D7).
- Las siete partes actuales no cambian, ni en contenido ni en CSV.

### D6. La página: la tarjeta de la pregunta 3

Como la maqueta, de arriba abajo:

- **Cabecera**: el titular nuevo y el veredicto «NO». El titular cambia también en la tarjeta de «Tres preguntas,
  tres respuestas», con una frase que sale de la foto («con el voto que se espera no cambia ningún escaño del
  reparto central del modelo, y sólo en el 0,07 % de sus simulaciones cambia el sentido de la mayoría»), y en el
  índice de secciones.
- **Entradilla** con el método en tres líneas y la pastilla «Foto fija · datos hasta el 23 de noviembre».
- **Tres cifras**, con los escaños en grande y la probabilidad debajo:
  - la derecha en el reparto central del modelo: 190 escaños; tendría que perder 15 para quedarse sin mayoría, y
    la conserva en el 87,3 % de las simulaciones;
  - con el voto CERA que se espera: 0 escaños, sigue en 190; mayoría en el 87,3 % (3 simulaciones menos de 10.000);
  - en el extremo de la rejilla: −10 escaños, se queda en 180; mayoría en el 61,0 %.
- **«Con el voto que se espera»**, sólo con el censo de la convocatoria (sin selector), en dos columnas:
  - una entradilla con las dos lecturas: sobre el reparto central, los 30.790 votantes nuevos no mueven ningún
    escaño; en las simulaciones sí mueven alguno, pero casi nunca el que decide la mayoría;
  - a la izquierda, la tabla 2023 → 29-N con dos filas: censo CERA y votantes CERA;
  - debajo, **las simulaciones que cambian escaños**, en tres casillas en fila que se leen como frases, con el
    porcentaje en grande: «En el 8,97 % se mueve algún escaño», «En el 5,71 % cambia el total de escaños de la
    derecha» (con «3,75 % a la baja y 1,96 % al alza» debajo) y «Sólo en el 0,07 % cambia el sentido de la
    mayoría». La tercera no lleva desglose: la tarjeta no dice en cuántas simulaciones la derecha pierde la
    mayoría (5) y en cuántas la consigue (2);
  - una línea que explica por qué la mayoría puede cambiar en los dos sentidos, sin cifras: el voto exterior que se
    espera no va sólo a la izquierda, y en provincias como Orense o Lugo suma más a la derecha;
  - a la derecha, las barras «Dónde mueve escaños»: las diez provincias donde más veces cambia un escaño, en
    porcentaje de las simulaciones, partidas en «la derecha pierde», «la derecha gana» y «cambia entre otros
    partidos»;
  - al final del bloque y a todo el ancho de la tarjeta, el aviso del litigio, que sólo dice: «El censo incluye las
    163.020 inscripciones en litigio. Hemos querido tener en cuenta todos los escenarios, incluidos los más
    extremos».
- **«¿Qué haría falta para que importara?»**, con dos gráficos, uno al lado del otro; el de la izquierda ocupa
  tres quintos del ancho y la rejilla, dos:
  - a la izquierda, **el gráfico «Lo que le quita cada escenario y lo que haría falta»**, que une las dos
    lecturas. Una fila por escenario: una barra con los escaños que la derecha pierde desde el reparto central, una
    línea fina con lo que pierde en el 95 % de las simulaciones y, a la derecha, los escaños y la probabilidad de
    mayoría. El eje va de 0 a 16 escaños, con una marca cada 2. Dos líneas verticales marcan lo que haría falta:
    «si la derecha suma 181» (6 escaños; 181 es su percentil 25 en las simulaciones) y «si la derecha suma 190»
    (15; su reparto central). Siete filas con títulos cortos, en dos grupos. «Electores nuevos desde 2023»: «9 %
    voto + 50 % izq.» (con «como 2023» debajo), «50 % voto + 80 % izq.» y «Colocación a ciegas» (con «100 % voto +
    100 % PSOE» debajo). «Todo el censo exterior»: «30 % voto + 60 % izq.», «30 % voto + 80 % izq.», «50 % voto +
    80 % izq.» y «70 % voto + 80 % izq.». Sus cifras están en «Cifras de la prueba». El voto que se espera ya no
    tiene fila: está en el bloque de arriba;
  - dos notas bajo ese gráfico. Qué marcan las líneas: 15 escaños si la derecha suma 190 y 6 si suma 181; con 181
    o menos empieza en algo más de una de cada cuatro simulaciones (27,7 %), y en el 12,7 % no tiene mayoría ni
    antes de contar el voto exterior. Y qué quieren decir «voto» (la participación) e «izq.» (la parte del voto a
    izquierda o derecha que va a la izquierda), con una frase que enlaza con el bloque de arriba: la primera fila
    aplica a los electores nuevos lo que hizo el censo exterior en 2023, y el voto que se espera de ellos es algo
    menor, un 7 %, con el mismo resultado;
  - a la derecha, la rejilla de todo el censo exterior, participación × voto a la izquierda, con dos cifras por
    casilla: arriba los escaños que pierde la derecha desde el reparto central y debajo la probabilidad de que
    conserve la mayoría. Su columna de títulos es corta y repite los de las barras: las filas son «9 % voto»,
    «20 % voto», «30 % voto», «50 % voto» y «70 % voto», y las columnas, de «50 % izq.» a «80 % izq.», sin rótulo
    en la esquina. Que la primera casilla (9 % y 50 %) es lo que pasó en 2023 lo dice la nota de debajo;
  - debajo de los dos, un comentario sobre la colocación, sin tabla: por qué es a ciegas (quien colocara electores
    sólo podía guiarse por 2023) y entre qué cifras quedaría, las de los extremos de la rejilla de colocación: de
    1 a 5 escaños menos en el reparto central, y la mayoría entre el 86,5 % (con la participación de 2023 y la
    mitad del voto al PSOE) y el 75,1 % (si votaran todos, y todos al PSOE). Es un límite, no una previsión.
- **Resumen**, que empieza por los escaños («Con el voto que se espera, ni un escaño: la derecha sigue en 190») y
  explica por qué la probabilidad baja aunque ningún escenario llegue a los 15 del reparto central: baja en las
  simulaciones en las que la derecha ya parte con poco margen.
- **«Cómo leerlo»**.

Las casillas se sombrean por lo que baja la probabilidad respecto a la de hoy. Las probabilidades de las casillas
van con un decimal; las cifras de las simulaciones que cambian y las barras de provincias, con dos (ver
«Decisiones mías»). En la rejilla y en las cifras de cabecera los escaños son el cambio de la derecha, con signo
(«−3»); el gráfico dibuja escaños que pierde. Su línea fina se corta en cero si en una punta del rango la derecha
gana escaños. Pasa en dos de las siete filas, «9 % voto + 50 % izq.» y «30 % voto + 60 % izq.»: en las dos el rango
llega a un escaño ganado por la derecha, que el gráfico no enseña y el CSV sí.

De la tarjeta salen la tabla de escaños 2023 → 29-N, el selector de censo, el gráfico de brechas, el radar y la
rejilla de colocación «óptima» con escaños. Sus datos siguen en el bundle y en los CSV de «Datos y método». Los
escaños que la tarjeta enseña ahora son otros: los del reparto central del modelo para el 29-N, no los de 2023.
Tampoco enseña la rejilla de los electores nuevos (el gráfico toma dos de sus casillas), los votos a candidaturas,
las cifras del censo sin litigio ni el desglose de las simulaciones que cambian la mayoría (5 en las que la
derecha la pierde y 2 en las que la consigue): quedan en la foto y en los CSV de la parte nueva.

### D7. Después del 29-N

Nada de la foto se recalcula. Cuando el estudio tenga la comprobación del 29-N (el mismo interruptor que hoy
enciende «Comprobada con el escrutinio»), la parte lleva `outcome` y la tarjeta enseña arriba una banda «Qué pasó
el 29-N»: los escaños de la derecha, si tuvo mayoría, cuántos escaños decidió el voto CERA y si la mayoría habría
sido otra sin él. Debajo, una línea dice que lo que sigue es la foto del 23 de noviembre tal como se publicó.

`outcome` sale de los resultados oficiales y de `check.with_without`, con los bloques de la foto, y se construye
ahora, para que la banda aparezca al cargar resultados y escrutinio sin tocar código. Con los datos reales de hoy
es nulo; el estado con resultados se prueba con unos resultados del 29-N inventados.

Las siglas oficiales del 29-N pueden no coincidir con los nombres del modelo (una coalición, otra grafía). Para
eso hay un mapa de alias en `data/cera/params.json`, de sigla oficial a partido del modelo, vacío hasta entonces.
Si algún partido de los dos bloques de la foto no aparece en los resultados, ni por su nombre ni por un alias, el
estudio no se publica y dice cuál falta: una banda con la mayoría mal contada es peor que no publicar.

### D8. Los textos

Los de la maqueta valen de punto de partida. «Cómo leerlo» explica: qué es una simulación y qué se compara; las
dos lecturas (el reparto central es el titular de escaños de la web; el rango y la probabilidad salen de las
simulaciones) y por qué pueden no coincidir; por qué hay dos referencias de lo que haría falta, 15 y 6; el paso de
las candidaturas de 2023 a los partidos del 29-N; qué mueve cada tabla (sólo los electores nuevos, todo
el censo exterior, la bolsa colocada); por qué la colocación es a ciegas y es un límite; y por qué es una foto
fija. Todas las cifras de los textos salen de la parte. No se dice que las simulaciones están en votos de 2023: la
base de cada provincia va escalada por el censo. La tarjeta habla de los 30.790 votantes nuevos, la cifra de su
tabla; «Cómo leerlo» dice que 30.482 de ellos son votos a candidaturas, que es lo que se suma en el reparto. Los
textos son provisionales hasta que Luis los revise.

## Cifras de la prueba (run 20261124-102045), para contrastar la construcción

- Base: derecha ≥ 176 en 8.734 de 10.000; reparto central: derecha 190, izquierda 129. Escaños de la derecha en las
  simulaciones: percentil 5, 170; 10, 174; 25, 181; 50, 188; 75, 195; 95, 205. Con 181 o menos, el 27,70 % de las
  simulaciones; con 180 o menos, el 24,48 %; sin mayoría de partida (175 o menos), el 12,66 %.
- Voto que se espera, convocatoria (30.482 votos; censo 2.327.388 → 2.769.645; votantes 208.431 → 239.221; votos
  203.152 → 233.634): pierde la mayoría en 5, la gana en 2; cambia algún escaño en 897; cambio de la derecha
  −2: 8, −1: 367, +1: 196. Sin litigio (22.412 votos; censo 2.606.625; votantes 231.070; votos 225.564): 4 y 2;
  676; −2: 5, −1: 283, +1: 135. En los dos censos el reparto central no cambia (derecha 0, izquierda 0) y el rango
  del 95 % del cambio de la derecha va de −1 a 0.
- Provincias, convocatoria (cambia / derecha pierde / derecha gana): Madrid 172/59/39, Barcelona 100/27/20, Santa
  Cruz de Tenerife 65/22/15, Almería 37/36/0, Alicante 36/25/0, Baleares 34/23/0, Orense 30/0/26, Granada 29/9/10,
  Valencia 28/18/0, Lugo 26/0/17. Cambia alguna vez en 50 de las 52, en los dos censos.
- Rejilla de los electores nuevos (442.257; votos añadidos con el 9 %, 39.405; 20 %, 87.567; 30 %, 131.350; 50 %,
  218.917; 70 %, 306.484), simulaciones en las que la derecha pierde y gana la mayoría (filas: participación;
  columnas: voto a la izquierda):

  | | 50 % | 60 % | 70 % | 80 % |
  |---|---|---|---|---|
  | 9 % | 8 / 4 | 18 / 0 | 27 / 0 | 44 / 0 |
  | 20 % | 16 / 9 | 39 / 3 | 61 / 0 | 94 / 0 |
  | 30 % | 22 / 10 | 63 / 5 | 95 / 2 | 137 / 0 |
  | 50 % | 40 / 23 | 86 / 8 | 152 / 3 | 218 / 1 |
  | 70 % | 49 / 35 | 118 / 17 | 208 / 5 | 321 / 1 |

- La rejilla de los electores nuevos, cambio de la derecha en el reparto central y, entre paréntesis, su rango
  del 95 %:

  | | 50 % | 60 % | 70 % | 80 % |
  |---|---|---|---|---|
  | 9 % | 0 (−1 a +1) | 0 (−1 a 0) | 0 (−1 a 0) | 0 (−1 a 0) |
  | 20 % | 0 (−1 a +1) | 0 (−1 a +1) | 0 (−2 a 0) | 0 (−2 a 0) |
  | 30 % | 0 (−1 a +1) | 0 (−2 a +1) | 0 (−2 a 0) | 0 (−3 a 0) |
  | 50 % | 0 (−1 a +1) | 0 (−2 a +1) | −1 (−3 a 0) | −2 (−3 a 0) |
  | 70 % | 0 (−2 a +1) | −1 (−3 a +1) | −1 (−3 a 0) | −2 (−4 a 0) |

  En las veinte casillas la izquierda gana en el reparto central lo que la derecha pierde. En 2023, con los
  bloques del run, a la izquierda fueron 89.689 votos CERA y a la derecha 91.009: el 49,6 %.

- Rejilla de todo el censo exterior, «pierde / gana» por casilla (filas: participación; columnas: voto a la
  izquierda):

  | | 50 % | 60 % | 70 % | 80 % |
  |---|---|---|---|---|
  | 9 % | 42 / 17 | 99 / 2 | 179 / 0 | 260 / 0 |
  | 20 % | 77 / 47 | 217 / 7 | 429 / 1 | 647 / 0 |
  | 30 % | 100 / 61 | 335 / 12 | 663 / 3 | 987 / 0 |
  | 50 % | 146 / 81 | 582 / 10 | 1.136 / 0 | 1.739 / 0 |
  | 70 % | 195 / 85 | 843 / 9 | 1.647 / 0 | 2.635 / 0 |

- La misma rejilla, cambio de la derecha en el reparto central y, entre paréntesis, su rango del 95 % en las
  simulaciones:

  | | 50 % | 60 % | 70 % | 80 % |
  |---|---|---|---|---|
  | 9 % | 0 (−1 a +1) | 0 (−2 a 0) | 0 (−3 a 0) | −2 (−4 a 0) |
  | 20 % | 0 (−2 a +2) | −1 (−4 a +1) | −2 (−5 a 0) | −3 (−6 a 0) |
  | 30 % | −1 (−3 a +2) | −2 (−5 a +1) | −3 (−7 a 0) | −4 (−8 a −1) |
  | 50 % | −1 (−4 a +2) | −3 (−7 a 0) | −3 (−9 a −1) | −8 (−12 a −3) |
  | 70 % | −2 (−4 a +3) | −3 (−8 a 0) | −9 (−12 a −2) | −10 (−15 a −5) |

  En el extremo la izquierda gana 11 en el reparto central (y 10, 3 y 2 en el resto de su fila, de derecha a
  izquierda).

- Colocación a ciegas, simulaciones en las que la derecha pierde la mayoría (no la gana en ninguna; filas:
  participación; columnas: voto al PSOE):

  | | 50 % | 60 % | 70 % | 80 % | 100 % |
  |---|---|---|---|---|---|
  | 9 % | 83 | 88 | 98 | 111 | 155 |
  | 20 % | 182 | 223 | 255 | 283 | 333 |
  | 50 % | 527 | 549 | 610 | 666 | 694 |
  | 100 % | 694 | 737 | 777 | 950 | 1.223 |

- La misma rejilla, cambio de la derecha en el reparto central y su rango del 95 %:

  | | 50 % | 60 % | 70 % | 80 % | 100 % |
  |---|---|---|---|---|---|
  | 9 % | −1 (−1 a 0) | −1 (−1 a 0) | −1 (−2 a 0) | −1 (−2 a 0) | −1 (−2 a 0) |
  | 20 % | −1 (−2 a 0) | −1 (−3 a 0) | −1 (−3 a 0) | −1 (−3 a 0) | −1 (−3 a 0) |
  | 50 % | −2 (−4 a 0) | −2 (−4 a 0) | −2 (−5 a 0) | −3 (−5 a 0) | −3 (−5 a 0) |
  | 100 % | −3 (−5 a 0) | −4 (−6 a 0) | −4 (−6 a 0) | −5 (−7 a 0) | −5 (−8 a 0) |

- Las siete filas del gráfico (escaños que pierde la derecha: reparto central y rango del 95 %; probabilidad de
  mayoría):

  | Escenario | Central | Rango 95 % | Mayoría |
  |---|---|---|---|
  | Electores nuevos, 9 % y 50 % (como 2023) | 0 | 0 a 1 (sin cortar, de −1 a 1) | 87,3 % |
  | Electores nuevos, 50 % y 80 % | 2 | 0 a 3 | 85,2 % |
  | Colocación a ciegas, al máximo (100 % y 100 %) | 5 | 0 a 8 | 75,1 % |
  | Todo el censo exterior, 30 % y 60 % | 2 | 0 a 5 (sin cortar, de −1 a 5) | 84,1 % |
  | Todo el censo exterior, 30 % y 80 % | 4 | 1 a 8 | 77,5 % |
  | Todo el censo exterior, 50 % y 80 % | 8 | 3 a 12 | 70,0 % |
  | Todo el censo exterior, 70 % y 80 % | 10 | 5 a 15 | 61,0 % |

- Lo que un escenario quita casi no depende de dónde parte la derecha: en las 902 simulaciones en las que parte de
  180 a 182 escaños (su percentil 25 ± 1), la mediana de lo que le quita cada escenario es la de las 10.000 en 48
  de los 62 escenarios y se aparta un escaño en los otros 14 (en la rejilla de todo el censo quita uno menos; en
  la colocación y en una casilla de los electores nuevos, uno más).

Si el run publicado al hacer la foto es otro, las cifras serán las suyas; estas sirven mientras sea el mismo.

## Decisiones mías, por revisar

1. **El plan de la colocación se decide con 2023 y se aplica al 29-N.** Es mi lectura de «a ciegas, aplicando el
   histórico». Luis la vio en la maqueta y no la corrigió.
2. **SUMAR → FA, UPN → PP, y UP y SALF sin voto exterior.** Todo el voto exterior de SUMAR va a FA; repartirlo
   entre FA y UP pediría un supuesto más y le quitaría eficacia a la izquierda.
3. **La rejilla de todo el censo quita el voto CERA de 2023** del voto del modelo antes de poner el del escenario,
   como hace hoy sobre 2023.
4. **Cómo se cuenta que la derecha también puede ganar la mayoría**: la línea que dice que puede cambiar en los
   dos sentidos, sin cifras desde la decisión 27 de Luis, y «3 simulaciones menos de 10.000» en la cifra de
   cabecera, que es el saldo y no el desglose.
5. **Los electores nuevos se guardan como rejilla**, con los ejes de la de todo el censo (veinte casillas), aunque
   el gráfico sólo enseña dos: es todo lo que quedará si Luis cambia otra vez las filas. Sustituye a las nueve
   filas de múltiplos, que ya no enseña nadie. El censo sin litigio sigue en la foto y en los CSV.
6. **Dos decimales en las cifras de las simulaciones que cambian y en las barras de provincias**, uno en las
   casillas. Con uno, casi todas las provincias empataban en 0,3 % y las partes no sumaban el total.
7. **Gráfico nuevo de provincias** en el sitio del de brechas de 2023.
8. **La tabla de censo, votantes y votos sale de la foto**, no de `prediction`.
9. **Qué guarda la foto** (D3). Guarda más de lo que la tarjeta enseña, porque después del 29-N no se podrá
   calcular: el histograma de la base, las distribuciones del cambio de escaños, las provincias del censo sin
   litigio y la rejilla entera de la colocación (la tarjeta sólo cita sus dos extremos). Van también en los CSV.
10. **La banda del 29-N aparece con la comprobación**, no la noche electoral, y qué dice (D7).
11. **La acción vive en el job `load`**, pide árbol limpio y no pisa la foto sin `overwrite`.
12. **Texto corto del índice de secciones**: «3 · ¿Quitan la mayoría?» en lugar de «3 · ¿Mueven escaños?».
13. **Los CSV antiguos de la pregunta 3 siguen listados** en «Datos y método», con una etiqueta que diga que son
    sobre 2023.
14. **Todos los textos** de la tarjeta.
15. **La foto la hace quien construye**, en local y nada más existir la acción, y entra en la rama. Lee la base
    (sólo lectura, como la prueba) y escribe un fichero del repositorio.
16. **Si falta un partido de los bloques en los resultados del 29-N, el estudio no se publica** (D7).
17. **El run rehecho se comprueba contra la parte `dist`**, simulación a simulación.
18. **El rango es el del 95 %**, como el «Rango 95%» del resto de la web. La sexta maqueta, la que Luis aceptó, lo
    traía al 90 %: con el 95 % las líneas del gráfico salen algo más largas y la del extremo de la rejilla llega a
    los 15 escaños.
19. **El gráfico**: los escenarios de seis de las siete filas son de Luis (decisiones 22, 29 y 30). Míos, la fila
    de la colocación (su máximo), los nombres de los dos grupos, para que el título corto no pierda de qué
    electores habla, y «como 2023» en una línea aparte bajo el título de la primera fila.
20. **La segunda referencia es el percentil 25** de los escaños de la derecha (181, 6 escaños); los nombres de las
    dos líneas son de Luis. Y cómo lo dice la nota: «en algo más de una de cada cuatro simulaciones (27,7 %)»,
    porque con 181 o menos hay más del 25 %.
21. **Las tres cifras de cabecera** llevan los escaños en grande y la probabilidad debajo.
22. **Signos**: las tablas y las cifras dan el cambio de la derecha («−3»); el gráfico, escaños que pierde. La
    línea del rango se corta en cero, y ahora pasa en dos filas (ver D6).
23. **La rejilla no lleva rango**, sólo escaños y probabilidad: el rango va en el gráfico y en los CSV.
24. **La foto guarda el cambio del reparto central de la izquierda** además del de la derecha, aunque la tarjeta
    sólo enseña el de la derecha.
25. **El cambio de concepto es posterior a la revisión del constructor.** Lo he contrastado yo con el código
    (`central_result`, `export_scenario`, los cuantiles de los rangos) y con la prueba; no ha pasado por un
    revisor.
26. **Los textos hablan de «30.790 votantes nuevos»**, no de «30.482 votos nuevos», para que cuadren con la tabla,
    que ya no lleva los votos a candidaturas.
27. **El aviso del litigio va al final del bloque «Con el voto que se espera»**, bajo sus dos columnas, y su
    redacción exacta a partir de la frase de Luis.
28. **Las tres casillas se leen como frases** («En el…», «Sólo en el…»). La de la tercera sale de la de Luis; la
    segunda conserva su desglose (3,75 % a la baja y 1,96 % al alza), que Luis vio en la opción B y no tocó.
29. **«Cambia el sentido de la mayoría» en todos los sitios**: el resumen y la tarjeta de «Tres preguntas, tres
    respuestas» decían «la mayoría cambia de manos»; ahora usan la fórmula de Luis.
30. **Cómo se acorta la columna de títulos de la rejilla**: filas y columnas con los títulos de las barras
    («9 % voto», «50 % izq.»), sin rótulo en la esquina, y «la de 2023» en la nota. Y el reparto del ancho, tres
    quintos y dos.
31. **La primera fila es un escenario nuevo, no el voto que se espera con otro nombre**: los electores nuevos
    votando como el censo exterior en 2023 (9 % y 50 %, 39.803 votantes). El voto que se espera (5 % y 10 % según
    el origen, un 7 %, 30.790 votantes) sigue siendo el del bloque de arriba. Dan lo mismo: ningún escaño y el
    87,3 %. La nota bajo el gráfico lo dice.
32. **La X de «como 2023» es 50 %**, la casilla de la rejilla más cercana al 49,6 % de 2023, y no el reparto
    provincia a provincia de 2023 (con él, el voto de los electores nuevos a la izquierda sería el 49,1 %). Así
    la fila es una casilla de la rejilla, como las demás.
33. **El resumen cita el escenario del centro**: «Si votara la mitad de los electores nuevos y el 80 % lo hiciera a
    la izquierda, dos escaños», en lugar del triple de participación con el 90 %.

## Lo que queda a quien lo construya

- Nombres de módulos, funciones, claves del fichero y de la parte, y de los CSV.
- Cómo se reutiliza el camino de `publish_forecast` (sacar de él la construcción del simulador o pasarle una
  fábrica que guarde el simulador) y cómo se vectoriza el reparto, siempre que pase la comprobación de D1.
- Si las barras de provincias se pintan en la plantilla o con la librería de gráficos; la imagen descargable de
  la tarjeta tiene que seguir saliendo. Las simulaciones que cambian son tres casillas de texto.
- Qué se retira de `pages/cera.py`, de la isla de datos y de `analisis-cera.js` al dejar de pintarse el gráfico de
  brechas y el radar, sin romper lo que otras tarjetas leen.
- La escala de sombreado de las casillas.
- Cómo se dibuja el gráfico de los escenarios (plantilla y CSS, como en la maqueta, o la librería de gráficos) y
  cómo se lee a 390 px, donde las etiquetas de las filas no caben a la izquierda de las barras. La maqueta sólo
  está hecha para escritorio: en el móvil las columnas se apilan, como en el resto de la página.
- De dónde toma el módulo puro el voto central: tiene que ser el del mismo `build_umat` sin sorteo que usa
  `central_result()`, sin tocar los generadores ni cambiar lo que `run` da.
- Qué se hace si un cuantil del rango no sale entero. En la prueba salen enteros todos.
- Buscar en la página y en sus metadatos (descripción, imagen del enlace, índice de `/analisis`) las frases que
  resumen la pregunta 3 con «cero escaños». Las de «Lo que dice el debate» y «Comprobación» hablan de la
  predicción registrada y se quedan.
- Una línea al principio del diseño del 19-10 que remita a éste.

## Pruebas

- El módulo puro, con simulaciones pequeñas hechas a mano: el reparto vectorizado coincide con
  `Simulator.alloc_seats`, también en empates y con el umbral; un escenario con `Δ = 0` no mueve nada; `lost`,
  `gained`, `moved` y las distribuciones salen como se calculan a mano; el cambio del reparto central de un
  escenario sale como se calcula a mano, y puede no coincidir con la mediana de las simulaciones.
- La acción, con dobles del simulador y del bundle: se niega si la `dist` del run rehecho difiere de la publicada
  en una sola simulación, si el reparto no reproduce las simulaciones o si el reparto central con `Δ = 0` no da
  el de `central_result()`; no pisa la foto; `dry_run` no escribe.
- `outcome`: nulo con los datos reales; con resultados inventados, los escaños y la mayoría con y sin voto CERA;
  con un partido de los bloques ausente, el estudio se niega; con un alias, lo encuentra.
- La foto real: forma y coherencia (recuentos entre 0 y `n_sim`, ejes del tamaño de sus matrices, 52 provincias,
  la rejilla de los electores nuevos con los ejes de la de todo el censo, el cambio del reparto central de cada
  escenario dentro de su rango del 95 %, los escaños centrales de la base iguales a los del `scenario` del run)
  y, mientras no haya resultados del 29-N, que sus votos coinciden con los de `prediction`.
- Publicación: la parte y sus CSV; las siete partes actuales, byte a byte como antes; el registro intacto.
- Página, con la parte de ejemplo: los dos estados (antes y después de la comprobación), el titular nuevo en los
  tres sitios, las tres casillas de las simulaciones que cambian (la tercera, sin desglose), las siete filas del
  gráfico con sus dos referencias, las casillas de la rejilla con sus dos cifras, y `assert_public_wording`.
- En el navegador, a 1280 y a 390 px: tablas, barras y gráfico de escenarios sin cortar (con su eje y sus dos
  líneas), los dos gráficos de «¿Qué haría falta?» uno al lado del otro a 1280 (el de las barras, más ancho) y
  apilados a 390, el aviso del litigio a todo el ancho, imagen descargable.

## Orden de construcción

La foto tiene fecha de caducidad (un sondeo anterior al 23-N corregido o cargado tarde ya impediría rehacer el
run), así que va primero:

1. El módulo puro del cálculo emparejado, con sus pruebas. Se puede contrastar sin base de datos con el material
   de la prueba, guardado fuera del repositorio en `files/stage/cera-majority-spike/` (scripts, las 10.000
   simulaciones, el voto central y sus resultados; las dos lecturas de cada escenario, en `concept.py` y
   `out/concept95.json`; la rejilla de los electores nuevos, en `chart_rows.py` y `out/chart_rows.json`).
2. La acción, y con ella **la foto**: se lanza en local, se comprueban sus cifras contra «Cifras de la prueba» y
   el fichero entra en la rama.
3. La lectura del fichero en el estudio, la parte nueva y sus CSV.
4. La página, `outcome` y los textos.

## Lo que hace Luis al final

1. Revisar la foto (su resumen y el fichero) y los textos.
2. Desplegar y publicar el estudio (`publish study`, sin `register`).
3. Tras el escrutinio CERA: cargar resultados y escrutinio como ya estaba previsto, añadir los alias de siglas que
   hagan falta y volver a publicar el estudio.

## Revisión del constructor (2026-11-24)

Un revisor sin la conversación leyó el diseño contra el código. No dejó preguntas para Luis. Cambió esto:

- La comprobación del run rehecho era floja (`summary` redondeado, `scenario` sin sorteo): ahora es la `dist`.
- La base de votos no son los válidos de 2023 sino `sim.prev_totals['votes']`, escalada por el censo.
- El fichero obligatorio chocaba con que la foto se hiciera al final: ahora se hace al principio y entra en la rama.
- `outcome` se construye y se prueba ahora, con alias de siglas y negándose si falta un partido.

El cambio de concepto (decisiones 18 a 20 de Luis) es posterior a esta revisión; ver la decisión mía 25.

## Visto por el camino, fuera del alcance

Los bloques del estudio (`data/cera/params.json`) son los de 2023: izquierda = PSOE, SUMAR, UP y MP; derecha = PP,
VOX y Cs. La comprobación de la predicción registrada cuenta el saldo «de la izquierda» con ellos, así que con los
resultados del 29-N dejaría a FA y a SALF en «Otros» si no se añaden. No lo toca este cambio; hay que decidirlo
antes de publicar la comprobación.

## Después del 29-N: pasos

1. Cargar los resultados y el escrutinio CERA del 29-N como dice la spec del 2026-10-19. El fichero de siglas de
   los resultados sirve a dos lecturas: «Comprobación», con los `blocks` de `data/cera/params.json` (hoy sin FA ni
   SALF), y la banda «Qué pasó el 29-N», con los bloques de la foto a través de `majority.aliases`.
2. Ejecutar `publish study`. Si se niega, el mensaje nombra las siglas que hay que añadir a `majority.aliases` de
   `data/cera/params.json`: cada sigla oficial con escaño va a un partido del modelo, o a `"-"` si es una
   candidatura que el modelo no simula (sus escaños van al resto). También se niega si un alias apunta a algo que
   no es un partido del modelo ni `"-"`, o si un partido de la derecha o de la izquierda no tiene ninguna sigla.
3. Leer la banda una vez, a mano, contra los totales de Interior: los escaños de la derecha, si tiene mayoría
   absoluta y los escaños que movió el voto exterior (contados provincia a provincia, en las provincias decididas
   de «Comprobación»; una provincia puede mover más de un escaño).
4. La foto no se vuelve a tomar: `data/cera/majority.json` y el código que la midió se quedan como están.
5. Las pruebas que habrá que tocar entonces, porque fijan el estado de antes del escrutinio con los datos reales:
   en `tests/test_cera_study.py`, `test_compute_cera_runs_end_to_end` (las comprobaciones son sólo 2019 y 2023 y
   no hay `verdict_input`) y `test_read_scrutinies_skips_the_elections_without_results_and_needs_their_candidatures`
   (con el escrutinio del 29-N y sin sus resultados en la lista, ese escrutinio sale como omitido); en
   `tests/test_publish_studies.py`, `test_export_study_figures` (las elecciones de la comprobación son sólo 2019 y
   2023, sin veredicto),
   `test_export_study_majority_carries_the_snapshot_the_ranges_and_the_csv` (`outcome` nulo) y
   `test_the_majority_fixture_is_what_the_exporter_writes` (el fixture `study-majority.json` lleva `outcome` nulo:
   hay que regenerarlo). Las dos pruebas del resultado y de la tabla congelada ya valen antes y después.
