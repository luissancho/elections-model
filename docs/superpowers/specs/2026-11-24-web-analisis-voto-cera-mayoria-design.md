# Web, Análisis: la pregunta 3 del voto CERA, sobre la predicción del modelo para el 29-N

Proyecto mediano. Luis lo pidió el 2026-11-24, primer día de la veda («Quiero actualizar el análisis del Voto CERA,
partiendo de las predicciones del modelo para el 29-N, en lugar de los resultados de 2023») y decidió el resto en la
conversación y sobre una maqueta. Lo que es suyo está en «Decisiones de Luis»; lo que no, en «Decisiones mías, por
revisar». Maqueta (cuarta versión: las correcciones de Luis a la segunda, con cifras reales de la prueba):
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
  central y mayoría absoluta en 8.734 simulaciones (87,3 %). El bundle publica medianas, reparto central y
  probabilidades; **no** publica los votos por provincia de cada simulación.
- **El run se puede rehacer en local** simulación a simulación (78 s): el `Simulator` con los parámetros del
  `meta` del run y `today='2026-11-23'`, `fit_forecast`, modo `forecast` y
  `run(split=True, random=True, n_sim=10000, horizon='deadline')`, el mismo camino de `publish.publish_forecast`.
  Cada simulación guarda el voto por provincia y partido (`units[..., vpred_pct]`, que por los votos válidos de la
  elección anterior da los votos que el simulador reparte) y sus escaños (`results`).
- **Prueba hecha** (scripts de usar y tirar): un reparto D'Hondt vectorizado sobre esos votos reproduce las 10.000
  simulaciones guardadas; aplicar 63 escenarios de voto exterior a las 10.000 lleva 12 s. Las cifras están en
  «Cifras de la prueba».
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

## Objetivos

- La tarjeta `#p3` responde con cifras del modelo: probabilidad de mayoría de la derecha hoy, con el voto CERA que
  se espera y bajo cada escenario de las tres palancas.
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

- Bloques: los del run (`meta.bmaps.blocks`), derecha = PP, VOX y SALF; izquierda = PSOE, FA y UP. Mayoría: la del
  run (`meta.majority`, 176).
- De 2023 al modelo: el voto exterior se reparte entre candidaturas de 2023 y hay que llevarlo a los partidos del
  29-N. Cada candidatura va al partido del modelo con su misma sigla; SUMAR va a FA y UPN a PP; las siglas que el
  modelo no simula van al resto, que cuenta para el umbral y no recibe escaños. UP y SALF no reciben voto exterior:
  en 2023 no tuvieron voto CERA propio.
- Antes de aplicar ningún escenario, el reparto vectorizado con `Δ = 0` tiene que dar los escaños guardados en las
  10.000 simulaciones. Si no, la foto no se hace.

### D2. Los escenarios

Todos reutilizan las funciones y los parámetros del estudio (`seats.py`, `stress.py`, `data/cera/params.json`).

1. **Voto que se espera** (`expected`), en los dos censos de hoy (`convocatoria` y `sin_litigio`): `Δ` es el
   `extra` del modelo de incremento (`new_electors` + `increment_votes`: los electores nuevos desde 2023, con la
   participación por origen de `params.participation` y el reparto del escrutinio CERA de 2023 de cada provincia).
   Además del total, se guarda por provincia: en cuántas simulaciones cambia su reparto, en cuántas la derecha
   pierde escaño allí y en cuántas lo gana.
2. **Dos dimensiones, sólo los electores nuevos** (`new_electors`): participación × 1, × 2 y × 3 (las dos tasas
   por origen) cruzada con sesgo empírico, 70 % y 90 % (`_biased_split`: la parte del voto a izquierda o derecha que
   va a la izquierda). Nueve filas, censo de la convocatoria. La fila (× 1, empírico) es el voto que se espera.
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
−1, 0, +1…), de la que salen las medias y el «cambian el total de la derecha».

### D3. La foto fija: `data/cera/majority.json`

Un JSON legible, escrito sólo por la acción de D4, con:

- `run`: ámbito, fecha de la elección, id del run, `as_of`, `n_sim`, semilla, y el commit y el día de la foto.
- `blocks`, `majority` y el mapa de siglas de 2023 a partidos del modelo que se aplicó.
- `base`: simulaciones con mayoría de la derecha, sus escaños en el reparto central y el histograma de sus escaños
  en las 10.000 simulaciones.
- `expected` (por censo): las cifras de la tabla (censo, electores nuevos, votantes y votos de 2023 y
  proyectados), los recuentos de D2 y las 52 provincias.
- `new_electors` (nueve filas), `grid` y `placement` (ejes y recuentos), con la bolsa de la colocación.

La tarjeta entera sale de este fichero, también la tabla de censo, votantes y votos, para que un cambio posterior
de las entradas del estudio no deje la tabla y las casillas hablando de cosas distintas. No lleva probabilidades
redondeadas: la página las calcula de los recuentos.

**Esto es todo lo que quedará del run del 23-N.** Lo que no esté aquí no se podrá calcular después del 29-N.

### D4. La acción que hace la foto

Una acción de job dentro de `mtpy` (propongo el job `load`, que ya tiene el precedente de `params`: escribe un
fichero versionado de `data/` a partir de la base), que recibe el id del run (por defecto, el `latest` del ámbito)
y:

1. Lee el `meta` del run publicado y rehace el run con sus parámetros, por el mismo camino que
   `publish.publish_forecast` construye y corre el simulador (no una copia de la receta).
2. Comprueba que el `summary` y el `scenario` del modo `forecast` que salen del run rehecho, pasados por los mismos
   exportadores, son iguales a los publicados. Hoy: 190 escaños y 8.734 simulaciones para la derecha.
3. Comprueba el reparto vectorizado de D1.
4. Calcula el estudio (`compute_cera`) para tomar de él censos, repartos, bolsa y parámetros, y aplica D2.
5. Escribe el fichero. Pide árbol limpio (apunta el commit), no pisa un fichero existente sin `overwrite` y con
   `dry_run` hace todo menos escribir.

Si 2 o 3 fallan, termina `refused` con el motivo y no escribe. El cálculo emparejado es un módulo puro del
paquete del estudio (matrices de votos y de escaños dentro, recuentos fuera), que se prueba sin base de datos; lo
que toca la base y el bundle queda en la acción.

### D5. La publicación: una parte nueva, `majority`

- `compute_cera` lee y valida el fichero y lo lleva en `CeraStudy`. El fichero pasa a ser una entrada obligatoria
  del estudio: sin él, `publish study` falla con un mensaje claro.
- `bundle.STUDY_PARTS['voto-cera']` gana `majority` (esquema `study-majority`), con su exportador y sus gemelos
  CSV (rejilla, dos dimensiones, colocación y provincias). `docs/web/contrato.md` lo recoge.
- La parte lleva lo de D3 más `outcome`, nulo hasta que exista la comprobación del 29-N (D7).
- Las siete partes actuales no cambian, ni en contenido ni en CSV.

### D6. La página: la tarjeta de la pregunta 3

Como la maqueta, de arriba abajo:

- **Cabecera**: el titular nuevo y el veredicto «NO». El titular cambia también en la tarjeta de «Tres preguntas,
  tres respuestas», con una frase que sale de la foto («con el voto que se espera, la mayoría cambia de manos en
  el 0,07 % de las simulaciones del modelo»), y en el índice de secciones.
- **Entradilla** con el método en tres líneas y la pastilla «Foto fija · datos hasta el 23 de noviembre».
- **Tres cifras**: mayoría de la derecha hoy (87,3 %, 8.734 de 10.000); con el voto CERA que se espera (8.731: la
  pierde en 5 y la gana en 2); en el extremo de la rejilla (61,0 %).
- **«Con el voto que se espera»**, con el selector de censo de hoy (convocatoria / sin litigio):
  - tabla 2023 → 29-N con tres filas: censo CERA, votantes CERA y votos a candidaturas;
  - embudo en porcentaje de las simulaciones: mueven algún escaño (8,97 %), cambian el total de la derecha
    (5,71 %; 3,75 % a la baja y 1,96 % al alza), cambian la mayoría (0,07 %; aquí sí en simulaciones, 5 y 2);
  - nota del litigio, con sus cifras;
  - barras «Dónde mueve escaños»: las diez provincias donde más veces cambia un escaño, en porcentaje de las
    simulaciones, partidas en «la derecha pierde», «la derecha gana» y «cambia entre otros partidos».
- **«¿Qué haría falta para que importara?»**:
  - tabla de las dos dimensiones (participación, sesgo, probabilidad de mayoría, cambio medio de escaños de la
    derecha y de la izquierda), sólo los electores nuevos;
  - rejilla de todo el censo exterior, participación × voto a la izquierda, con la probabilidad en cada casilla;
  - rejilla de la colocación a ciegas, participación × voto al PSOE, y a su lado cómo se hace la cuenta y por qué
    es a ciegas.
- **Resumen** y **«Cómo leerlo»**.

Las casillas se sombrean por lo que baja la probabilidad respecto a la de hoy. Las probabilidades de las casillas
van con un decimal; el embudo y las barras, con dos (ver «Decisiones mías»).

De la tarjeta salen la tabla de escaños 2023 → 29-N, el gráfico de brechas, el radar y la rejilla de colocación
«óptima» con escaños. Sus datos siguen en el bundle y en los CSV de «Datos y método».

### D7. Después del 29-N

Nada de la foto se recalcula. Cuando el estudio tenga la comprobación del 29-N (el mismo interruptor que hoy
enciende «Comprobada con el escrutinio»), la parte lleva `outcome` y la tarjeta enseña arriba una banda «Qué pasó
el 29-N»: los escaños de la derecha, si tuvo mayoría, cuántos escaños decidió el voto CERA y si la mayoría habría
sido otra sin él. Debajo, una línea dice que lo que sigue es la foto del 23 de noviembre tal como se publicó.

`outcome` sale de los resultados oficiales y de `check.with_without`, con los bloques de la foto. La
correspondencia entre las siglas oficiales del 29-N y los partidos del modelo se cierra al cargar esos
resultados; hasta entonces `outcome` es nulo y las pruebas fijan ese estado.

### D8. Los textos

Los de la maqueta valen de punto de partida. «Cómo leerlo» explica: qué es una simulación y qué se compara; el
paso de las candidaturas de 2023 a los partidos del 29-N; qué mueve cada tabla (sólo los electores nuevos, todo
el censo exterior, la bolsa colocada); por qué la colocación es a ciegas y es un límite; y por qué es una foto
fija. Todas las cifras de los textos salen de la parte. Los textos son provisionales hasta que Luis los revise.

## Cifras de la prueba (run 20261124-102045), para contrastar la construcción

- Base: derecha ≥ 176 en 8.734 de 10.000; reparto central 190.
- Voto que se espera, convocatoria (30.482 votos; censo 2.327.388 → 2.769.645; votantes 208.431 → 239.221; votos
  203.152 → 233.634): pierde la mayoría en 5, la gana en 2; cambia algún escaño en 897; cambio de la derecha
  −2: 8, −1: 367, +1: 196. Sin litigio (22.412 votos; censo 2.606.625; votantes 231.070; votos 225.564): 4 y 2;
  676; −2: 5, −1: 283, +1: 135.
- Provincias, convocatoria (cambia / derecha pierde / derecha gana): Madrid 172/59/39, Barcelona 100/27/20, Santa
  Cruz de Tenerife 65/22/15, Almería 37/36/0, Alicante 36/25/0, Baleares 34/23/0, Orense 30/0/26, Granada 29/9/10,
  Valencia 28/18/0, Lugo 26/0/17. Cambia alguna vez en 50 de las 52, en los dos censos.
- Dos dimensiones, sólo los electores nuevos (simulaciones en las que la derecha pierde y gana la mayoría; cambio
  medio de escaños):

  | Participación | Sesgo | Votos | Pierde | Gana | Media derecha | Media izquierda |
  |---|---|---|---|---|---|---|
  | × 1 | empírico | 30.482 | 5 | 2 | −0,019 | +0,031 |
  | × 1 | 70 % | 30.482 | 20 | 0 | −0,105 | +0,118 |
  | × 1 | 90 % | 30.482 | 42 | 0 | −0,198 | +0,212 |
  | × 2 | empírico | 60.964 | 9 | 4 | −0,033 | +0,064 |
  | × 2 | 70 % | 60.964 | 43 | 0 | −0,209 | +0,240 |
  | × 2 | 90 % | 60.964 | 82 | 0 | −0,400 | +0,429 |
  | × 3 | empírico | 91.446 | 17 | 6 | −0,050 | +0,098 |
  | × 3 | 70 % | 91.446 | 63 | 0 | −0,314 | +0,360 |
  | × 3 | 90 % | 91.446 | 126 | 0 | −0,592 | +0,639 |

- Rejilla de todo el censo exterior, «pierde / gana» por casilla (filas: participación; columnas: voto a la
  izquierda):

  | | 50 % | 60 % | 70 % | 80 % |
  |---|---|---|---|---|
  | 9 % | 42 / 17 | 99 / 2 | 179 / 0 | 260 / 0 |
  | 20 % | 77 / 47 | 217 / 7 | 429 / 1 | 647 / 0 |
  | 30 % | 100 / 61 | 335 / 12 | 663 / 3 | 987 / 0 |
  | 50 % | 146 / 81 | 582 / 10 | 1.136 / 0 | 1.739 / 0 |
  | 70 % | 195 / 85 | 843 / 9 | 1.647 / 0 | 2.635 / 0 |

- Colocación a ciegas, simulaciones en las que la derecha pierde la mayoría (no la gana en ninguna; filas:
  participación; columnas: voto al PSOE):

  | | 50 % | 60 % | 70 % | 80 % | 100 % |
  |---|---|---|---|---|---|
  | 9 % | 83 | 88 | 98 | 111 | 155 |
  | 20 % | 182 | 223 | 255 | 283 | 333 |
  | 50 % | 527 | 549 | 610 | 666 | 694 |
  | 100 % | 694 | 737 | 777 | 950 | 1.223 |

Si el run publicado al hacer la foto es otro, las cifras serán las suyas; estas sirven mientras sea el mismo.

## Decisiones mías, por revisar

1. **El plan de la colocación se decide con 2023 y se aplica al 29-N.** Es mi lectura de «a ciegas, aplicando el
   histórico». Luis la vio en la maqueta y no la corrigió.
2. **SUMAR → FA, UPN → PP, y UP y SALF sin voto exterior.** Todo el voto exterior de SUMAR va a FA; repartirlo
   entre FA y UP pediría un supuesto más y le quitaría eficacia a la izquierda.
3. **La rejilla de todo el censo quita el voto CERA de 2023** del voto del modelo antes de poner el del escenario,
   como hace hoy sobre 2023.
4. **Las rejillas y la tabla de dos dimensiones sólo con el censo de la convocatoria**; el selector de censo sólo
   mueve el bloque «Con el voto que se espera».
5. **Nueve filas en la tabla de dos dimensiones** (los tres niveles de cada una) y sus columnas: probabilidad de
   mayoría y cambio medio de escaños de derecha e izquierda. Los votos nuevos van en la nota.
6. **Dos decimales en el embudo y en las barras**, uno en las casillas. Con uno, casi todas las provincias
   empataban en 0,3 % y las partes del embudo no sumaban el total.
7. **Gráfico nuevo de provincias** en el sitio del de brechas de 2023.
8. **La tabla de censo, votantes y votos sale de la foto**, no de `prediction`.
9. **Qué guarda la foto** (D3), incluidos el histograma de la base y las distribuciones del cambio de escaños, que
   la página no enseña enteros.
10. **La banda del 29-N aparece con la comprobación**, no la noche electoral, y qué dice (D7).
11. **La acción vive en el job `load`**, pide árbol limpio y no pisa la foto sin `overwrite`.
12. **Texto corto del índice de secciones**: «3 · ¿Quitan la mayoría?» en lugar de «3 · ¿Mueven escaños?».
13. **Los CSV antiguos de la pregunta 3 siguen listados** en «Datos y método», con una etiqueta que diga que son
    sobre 2023.
14. **Todos los textos** de la tarjeta.

## Lo que queda a quien lo construya

- Nombres de módulos, funciones, claves del fichero y de la parte, y de los CSV.
- Cómo se reutiliza el camino de `publish_forecast` (sacar de él la construcción del simulador o pasarle una
  fábrica que guarde el simulador) y cómo se vectoriza el reparto, siempre que pase la comprobación de D1.
- Si las barras y el embudo se pintan en la plantilla o con la librería de gráficos; la imagen descargable de la
  tarjeta tiene que seguir saliendo.
- Qué se retira de `pages/cera.py`, de la isla de datos y de `analisis-cera.js` al dejar de pintarse el gráfico de
  brechas y el radar, sin romper lo que otras tarjetas leen.
- La escala de sombreado de las casillas.
- Buscar en la página y en sus metadatos (descripción, imagen del enlace, índice de `/analisis`) las frases que
  resumen la pregunta 3 con «cero escaños». Las de «Lo que dice el debate» y «Comprobación» hablan de la
  predicción registrada y se quedan.
- Una línea al principio del diseño del 19-10 que remita a éste.

## Pruebas

- El módulo puro, con simulaciones pequeñas hechas a mano: el reparto vectorizado coincide con
  `Simulator.alloc_seats`, también en empates y con el umbral; un escenario con `Δ = 0` no mueve nada; `lost`,
  `gained`, `moved` y las distribuciones salen como se calculan a mano.
- La acción, con dobles: se niega si el run rehecho no coincide con el publicado o si el reparto no reproduce las
  simulaciones; no pisa la foto; `dry_run` no escribe.
- La foto real: forma y coherencia (recuentos entre 0 y `n_sim`, ejes del tamaño de sus matrices, 52 provincias,
  la fila × 1 empírico igual al voto que se espera en la convocatoria) y, mientras no haya resultados del 29-N,
  que sus votos coinciden con los de `prediction`.
- Publicación: la parte y sus CSV; las siete partes actuales, byte a byte como antes; el registro intacto.
- Página, con la parte de ejemplo: los dos estados (antes y después de la comprobación), los dos censos, el
  titular nuevo en los tres sitios, y `assert_public_wording`.
- En el navegador, a 1280 y a 390 px: selector de censo, tablas sin cortar, imagen descargable.

## Lo que hace Luis al final

1. Lanzar la acción de la foto en local, antes del 29-N, y revisar su resumen.
2. Hacer commit del fichero.
3. Desplegar y publicar el estudio (`publish study`, sin `register`).
4. Tras el escrutinio CERA: cargar resultados y escrutinio como ya estaba previsto y volver a publicar el estudio.

## Visto por el camino, fuera del alcance

Los bloques del estudio (`data/cera/params.json`) son los de 2023: izquierda = PSOE, SUMAR, UP y MP; derecha = PP,
VOX y Cs. La comprobación de la predicción registrada cuenta el saldo «de la izquierda» con ellos, así que con los
resultados del 29-N dejaría a FA y a SALF en «Otros» si no se añaden. No lo toca este cambio; hay que decidirlo
antes de publicar la comprobación.
