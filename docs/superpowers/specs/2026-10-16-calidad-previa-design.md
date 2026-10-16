# Calidad previa de las encuestadoras: rúbrica pública, evidencia del rating y publicación

Fecha: 2026-10-16. Estado: aprobada por Luis el 2026-10-16 ("Adelante"); en construcción. Descripción acordada en el
chat del 16 de octubre (rúbrica, almacenamiento completo y cambio del recuento de evidencia del rating: "Ok, procede"). Revisión del
constructor hecha el mismo día: tres preguntas (cómo se midió k, alcance de la verificación contra la base, origen del
KPI «Calidad previa»), resueltas como decisiones mías 9-11 e incorporadas a D3, D4, D5 y D6. Esta spec solo describe
lo que cambia respecto de lo que hay en `dev` (`9e83989`). Cierra el hallazgo F23/M12 del análisis de
septiembre (`docs/analisis-2026-09/computer-ratings.md`: "prior `quality` subjetivo y no documentado") y la
recomendación de `data.md` de versionar `pollsters.csv` "con `quality` y su justificación".

## Intención

La calidad previa (`pollsters.quality`, 0-100) es el prior del rating de cada encuestadora: manda mientras la casa
tiene pocos sondeos evaluados y es el peso con el que entran en el promedio las casas sin rating. Hoy es un número
manual sin rúbrica. Luis lo construyó a partir de la tabla PollCheck de Electomanía (`files/Electomania-Rankings.xlsx`):
cuatro de sus cinco subnotas (serie histórica, societario, difusión y coherencia, con pesos 3:2:2:6, la desviación
histórica con peso 0) reescaladas a 10-90, con retoques a mano (CIS 31 → 10 por politización reciente, ElectoPanel
90 → 10, EM Analytics 10, agregadores 0) y 5 para las casas que no estaban en la tabla. Luis quiere "hacerlo algo más
objetivo": un criterio escrito y comprobable, guardado con su desglose y sus fuentes, y publicado en la web.

Al revisar cómo pesa ese prior en el rating apareció un segundo problema, que esta spec también resuelve: con la
fórmula actual el prior del CIS pesa un 20 % pese a sus 234 sondeos evaluables, porque la "cantidad de sondeos" que
entra en la mezcla descuenta un 30 % por cada semana de distancia a la elección. Luis: "con la cantidad de sondeos que
tenemos del CIS debería contar solo su rendimiento; el prior solo debe servir cuando no hay suficientes sondeos".

## Decisiones de Luis (2026-10-16)

1. La calidad previa deja de medir acierto pasado (eso ya lo mide el rating) y pasa a medir lo que los datos no
   ven: cómo trabaja la casa y de quién depende. Electomanía sale del prior.
2. Una casa de la que no se sabe nada debe pesar poco mientras no se la pueda evaluar: la calidad mínima sigue siendo
   5 y es la de las casas sin evaluar.
3. Se guarda todo: el desglose por criterio, la fuente y la fecha de comprobación de cada casa, y se publica en el
   perfil de la encuestadora y en el método.
4. El prior no debe pesar cuando hay sondeos de sobra. Se quita el descuento por semanas del rating y se mantienen
   los otros tres pesos (posición, antigüedad y ámbito) y `k = 3`.
5. El 0 deja de existir como calidad (hoy anula los sondeos de la casa en silencio).

## Objetivos

1. **D1** Una rúbrica de cinco criterios comprobables, 10 puntos, con niveles escritos y una regla fija de puntos a
   calidad.
2. **D2** La evaluación de cada casa versionada en `data/` y replicada en la base de datos; `pollsters.quality` se
   deriva de ella y deja de editarse a mano.
3. **D3** El rating cuenta la evidencia sin el descuento por semanas, en la media ponderada y en el recuento.
4. **D4** La web muestra el desglose en el perfil de cada casa, explica la rúbrica en el método y la API lo sirve.
5. **D5** Una primera evaluación de las casas activas, con fuentes, en estado borrador hasta que Luis la revise.

## No objetivos

- Ningún otro cambio en el rating: ni `k` (`min_polls = 3`), ni τ (`bias_dev_tau`), ni la posición (0,5), ni la
  antigüedad (0,9), ni el peso de las autonómicas (0,5), ni la fórmula de mezcla `R = (n·R_aj + k·Q)/(n + k)`, ni la
  contracción por precisión de la desviación de la casa. Quitar también la antigüedad bajaría el prior del CIS al 4 %,
  pero diluiría al CIS reciente con treinta años del CIS antiguo: descartado.
- Ningún cambio en el Forecaster: el relleno del peso de una casa sin rating con su calidad (en la escala del peso,
  B10) sigue igual.
- No se excluye ninguna casa por su calidad. Los agregadores (`mtype = aggr`) y los paneles abiertos (`online`) ya
  quedan fuera del rating y del promedio por `drop_mtypes` en `Computer` y `Forecaster`; su calidad previa pasa a ser
  informativa.
- No se evalúan ahora las casas sin sondeos recientes (unas 75): quedan "sin evaluar" con calidad 5, que es lo que
  tienen hoy casi todas. Se podrán evaluar después añadiendo filas al CSV.
- No se publican las notas de Electomanía ni se redistribuye su XLSX.
- La columna «Calidad» del ranking de `/casas` no cambia de forma.
- Nada de nginx, Docker ni CSP.

## Anti-objetivos (lo que haría fracasar el cambio aunque funcionara)

- Una evaluación publicada sin fuente citable: cada punto por encima de 0 lleva su fuente.
- Que la calidad de la base y la del CSV difieran: la base es réplica, nunca se edita a mano.
- Que una casa sin evaluar salga en blanco en la web: debe decir "sin evaluar" y por qué vale 5.
- Que el texto del método describa pesos que el código ya no aplica (hoy pasa al revés con la docstring de
  `compute_ratings`): el método, la docstring y el código deben decir lo mismo.
- Romper `notebooks/data-load/PollsCompute.ipynb` o `notebooks/lab/PollstersOptimize.ipynb`.

## D1. La rúbrica

Cinco criterios, 10 puntos. Cada criterio tiene niveles cerrados; se asigna el nivel más alto cuyo texto se cumple
y se puede citar. Lo que no se puede comprobar vale 0 en ese criterio: así una casa desconocida puntúa cerca del
mínimo por construcción, sin decisión manual.

| Clave | Criterio | Puntos | Niveles |
|---|---|---|---|
| `fieldwork` | Trabajo de campo | 0-3 | 3: entrevistas propias con marco probabilístico (teléfono, presencial o mixto con teléfono). 2: panel con reclutamiento controlado y cuotas (propio o contratado). 1: muestra autoseleccionada (panel voluntario, captación en web o redes) con ponderación. 0: sin trabajo de campo propio (agregadores, estimaciones sobre otros sondeos) o modo no declarado. |
| `transparency` | Transparencia | 0-3 | 3: publica microdatos o las tablas completas de cada sondeo. 2: ficha técnica completa con intención directa de voto y método de estimación declarado. 1: ficha técnica mínima (muestra, fechas, modo, error). 0: no publica ficha. |
| `independence` | Independencia | 0-2 | 2: varios clientes de distinta línea, o autofinanciada, sin vínculo con un actor político. 1: un único cliente, un medio. 0: depende de un gobierno, un partido o un actor político (organismo público, fundación o encuesta de partido). |
| `identity` | Identidad | 0-1 | 1: entidad identificable (razón social o institución, sede, responsables, web). 0: marca sin entidad identificable. |
| `standards` | Estándares | 0-1 | 1: miembro de Insights+Analytics España o de ESOMAR, o certificada ISO 20252. 0: ninguna de las tres. |

**De puntos a calidad.** `quality = max(5, 8 · puntos)`: 0 puntos → 5, 1 → 8, 7 → 56, 10 → 80. Nadie pasa de 80 antes
de los datos; los últimos 20 se ganan acertando (el rating ajustado de las mejores casas está en 73-81). Una casa sin
fila en el CSV está "sin evaluar" y vale 5, igual que una con 0 puntos.

**Normas de aplicación.**

- Se evalúa la casa tal como trabaja hoy, no por épocas: una fila por casa. Si cambia, se edita la fila y la fecha.
- La fuente de cada criterio es una URL o una referencia citable (web de la casa, ficha técnica de un sondeo
  concreto, lista de miembros de la asociación, registro público). La base de datos sirve de fuente para dos cosas:
  los patrocinadores de la casa (independencia) y si sus sondeos traen muestra y fechas (transparencia mínima).
- Lo que queda fuera a propósito: la longevidad (si una casa tiene veinte años de sondeos, los datos ya mandan), el
  tamaño muestral (ya pesa sondeo a sondeo en `weight_sample`), la coherencia (la mide el rebaño) y cualquier nota de
  acierto.

**Consecuencias conocidas, aceptadas por Luis.** CIS: 3 + 3 + 0 + 1 + 0 = 7 puntos, calidad 56 (hoy 10); con D3 su
rating pasa de 16 a unos 18. ElectoPanel: 1 + 1 + 1 + 1 + 0 = 4, calidad 32 (hoy 10), informativa porque es `online`.
Una casa seria pero desconocida sube de 5 a 35-45 en cuanto se comprueba que existe, declara el modo y publica ficha.

## D2. Datos versionados y réplica en la base

Dos ficheros nuevos en `data/`, curación propia, CC BY 4.0 como el resto de la curación:

**`data/pollsters-quality-criteria.json`.** El catálogo de la rúbrica, única fuente de verdad de los textos: por
clave de criterio, `name` (nombre público), `max` (puntos) y `levels` (texto de cada nivel, de `"0"` a `max`), más
`scale` con `per_point: 8` y `floor: 5`. Lo leen la función de puntuación, la web (`/metodo/rating` y el perfil) y
el test del CSV.

**`data/pollsters-quality.csv`.** Una fila por casa evaluada, UTF-8, columnas:
`pollster_id, pollster, fieldwork, fieldwork_source, transparency, transparency_source, independence,
independence_source, identity, identity_source, standards, standards_source, notes, status, checked_at`.
`pollster_id` es la clave (el `id` de `elections.pollsters`; `pollster` es su nombre, solo para leer el fichero);
`status` ∈ {`borrador`, `revisada`}; `checked_at` es la fecha de la comprobación (`YYYY-MM-DD`). Un criterio con
puntos > 0 exige fuente. Los puntos y la calidad no se guardan en el CSV: se derivan.

**Puntuación.** Una función pura en `mtpy/lib/data.py` (o un módulo pequeño `mtpy/lib/quality.py`, a criterio del
constructor) que lee los dos ficheros, valida (ids únicos, puntos dentro del máximo de cada criterio, fuente
presente cuando hay puntos, `status` y `checked_at` válidos) y devuelve una tabla con los criterios, `points` (suma)
y `quality` (regla de D1).

**Réplica.** Nuevo modelo `PollstersQuality` (`elections.pollsters_quality`, clave `pollster_id`) con las columnas del
CSV más `points` y `quality`. `save_catalogues()` la escribe como hace con `scopes` y `districts` (crea la tabla si
falta, upsert por clave) y además **borra las filas de casas que ya no estén en el CSV** (es una réplica), y después
escribe `pollsters.quality` de **todas** las casas: la calidad calculada para las evaluadas y 5 para las demás.
Devuelve también los recuentos (`quality` filas, `pollsters` actualizadas). El test de integración
`tests/integration/test_scopes.py` recoge los nuevos recuentos.

Con esto `pollsters.quality` deja de ser una columna manual. El cargador de ámbitos autonómicos sigue dando de alta
las casas nuevas con 5: es exactamente "sin evaluar".

**Documentación.** Filas nuevas en la tabla de `data/README.md` y una sección corta "Calidad previa de las
encuestadoras" con la rúbrica resumida, la regla de puntos y el paso de réplica. `docs/web/contrato.md` recoge el
campo nuevo de la API (D4).

## D3. El rating cuenta la evidencia: fuera el descuento por semanas

**Diagnóstico.** `Computer.pollster_ratings` (`computer.py:1596-1690`) mezcla `rating = (n_w · rating_adj +
min_polls · quality)/(n_w + min_polls)`, con `n_w` la suma de `weight_over · weight_sample · weight_pos ·
weight_week · weight_year · weight_scope`. Esa fórmula es un modelo jerárquico: el peso del prior es `k/(k + n)` con
`k = σ²_casa / τ²`, la varianza de una casa de sondeo a sondeo más allá de su error muestral, dividida por la varianza
entre casas. Medido sobre los sondeos nacionales de la base: σ_casa ≈ 1,9 puntos, τ ≈ 1 punto, `k ≈ 3,7`. El 3 actual
es correcto. Lo que falla es `n`: `weight_week = 0,7^semanas` es herencia de cuando el error se medía contra el
resultado; la desviación se mide hoy contra el consenso de la misma fecha, así que un sondeo lejano informa igual de
la casa. Con ese descuento los 234 sondeos evaluables del CIS valen 12.

**Cómo se midió k** (para reproducirlo; el constructor lo deja como celda en `notebooks/lab/PollstersOptimize.ipynb`,
que ya reimplementa el rating, junto con la réplica de la tabla de abajo). Sobre `elections.polls` con `bias` no nulo,
`event_scope = 'es'`, `computed`, `ctype` fuera de `wban`/`exit`, sin pesos, en la escala de puntos en que están
guardadas `bias_dev_adj` y `bias_dev_err` (la desviación de cada sondeo respecto del consenso de su fecha, ya contraída
por su precisión, y su error): para cada casa con 20 sondeos o más, `σ²_casa = var(bias_dev_adj) − media(bias_dev_err²)`
(acotado en 0); se agrupa ponderando por el número de sondeos: σ_casa = 1,93 puntos (entre 1,2 y 2,7 según la casa).
τ no se estima: es el `bias_dev_tau = 0,01` log-odds del modelo, 1,005 puntos en esa escala. `k = 1,93² / 1,005² ≈ 3,7`.
Es una medición aproximada, suficiente para decir que el 3 está bien; por eso el método habla de «unas 3-4» y no de
3,7. Si al reproducirla sale fuera de 3-4, se publica el intervalo que salga, no se quita la frase.

**Cambio.** Desaparecen el parámetro `week_decay` de `Computer.__init__`, el atributo, la columna `weight_week` de
`poll_rating_weights` y su factor en `pollster_ratings` (`df['weight']` pasa a ser el producto de `weight_over`,
`weight_sample`, `weight_pos`, `weight_year` y `weight_scope`). Se quita de la media ponderada y del recuento a la vez:
declarar más precisión que la del estimador sería inconsistente. El umbral de descarte de sondeos de peso < 0,01 se
mantiene (con el cambio descarta menos). Las docstrings de `pollster_ratings`, `poll_rating_weights` y de los
parámetros del constructor describen lo que el código hace, incluida la lectura jerárquica de `min_polls` (`k ≈ σ²/τ²
≈ 3-4 medido). `print_rating_weights` y cualquier salida que liste `weight_week` pierden la columna. El notebook
`notebooks/lab/PollstersOptimize.ipynb` (celdas que listan `weight_week` en `weights` y en la tabla) se actualiza para
que vuelva a ejecutar.

**Efecto esperado** (réplica del cálculo sobre la base, para la próxima general: una exportación de `elections.polls`
y `elections.scopes` y el recuento en pandas con los mismos pesos de `poll_rating_weights`; la columna «hoy» coincide
con `pollsters_ratings` a menos de un punto, lo que valida la réplica). El constructor reproduce esta tabla con la
misma réplica (consultas de solo lectura contra la RDS, que responden en segundos con `psql`; no hace falta ejecutar
`Computer`), y Luis la confirma con los ratings recalculados en el paso 2 de D6:

| Casa | n hoy | n nuevo | Prior hoy | Prior nuevo | R_aj hoy | R_aj nuevo |
|---|---|---|---|---|---|---|
| CIS | 12,2 | 31,9 | 20 % | 9 % | 17 | 15 |
| Sigma Dos | 17,7 | 29,0 | 15 % | 9 % | 65 | 63 |
| GAD3 | 14,1 | 22,4 | 18 % | 12 % | 72 | 77 |
| Metroscopia | 6,6 | 15,0 | 31 % | 17 % | 79 | 77 |
| Simple Lógica | 3,8 | 8,9 | 44 % | 25 % | 60 | 50 |
| Ikerfel | 2,3 | 5,0 | 56 % | 37 % | 58 | 58 |
| Cluster17 | 1,5 | 1,8 | 67 % | 62 % | 91 | 91 |

Las casas grandes bajan al 9-12 % de prior, las medianas al 15-25 % y las nuevas siguen dominadas por él.

**Tests.** `tests/test_computer_unit.py` (fixture de la l. 192 fija `week_decay`): se adapta y se añade una
comprobación de que, a igualdad de lo demás, un sondeo lejano pesa lo mismo que uno cercano en `n_w` y en la media.

## D4. Web y API

**Perfil de la casa (`/casas/{slug}`, `pages/casas.py`, `casa.html`).**

- El KPI «Calidad previa» pasa a mostrar la calidad **actual** de la casa (`pollsters.quality`, que ya viaja en
  `profile.pollster`), no la de su última fila de ratings, y añade un apoyo (`support`): «7 de 10 puntos» para una
  casa evaluada, «sin evaluar» para las demás. Así el KPI y la tarjeta nueva dicen siempre lo mismo, también entre
  el paso 1 y el paso 2 de D6 o tras editar el CSV sin recalcular. La calidad con la que se calculó cada rating sigue
  en las filas de `pollsters_ratings` y la explica el «Cómo leerlo» de «Camino al rating» («la calidad previa que se
  usó en cada elección fue la de entonces; la del KPI es la de hoy»).
- Una tarjeta nueva «Calidad previa», justo después de la tarjeta del perfil y antes de «Camino al rating». Para una
  casa evaluada: una tabla con una fila por criterio (Criterio, Puntos sobre el máximo, Nivel: el texto del nivel
  asignado según el catálogo, Fuente: enlace si es una URL, texto si no), una línea de pie con «Comprobado el
  {fecha}» y, si `status = borrador`, la nota «Evaluación provisional, pendiente de revisión»; `notes` se muestra si
  existe. Para una casa sin evaluar: un aviso «Sin evaluar: tiene la calidad mínima, 5, hasta que alguien compruebe
  cómo trabaja», con enlace a la rúbrica del método. Para una casa `aggr` u `online`, una frase que recuerde que sus
  sondeos no entran en el promedio y que la calidad es informativa. Un «Cómo leerlo» con la regla de puntos a
  calidad y la referencia al método.
- Los datos llegan por `webapi.query_pollster`, que añade la clave `quality` (el registro de
  `elections.pollsters_quality` de la casa, o `None`) leída de una tabla cacheada nueva `pollsters-quality` (misma
  caché de 60 s que las demás, `get_pollsters_quality()` en `data.py`). Sin base configurada, la página se comporta
  como hoy (la ruta ya exige base).

**Método.**

- `/metodo/rating`, tarjeta «La calidad previa»: se reescribe. Qué mide (cómo trabaja la casa, no cuánto acierta);
  la tabla de la rúbrica generada desde el catálogo JSON (criterio, puntos, niveles); la regla `max(5, 8 ·
  puntos)` y por qué el tope es 80; qué significa «sin evaluar»; que la evaluación es pública y revisable (fichero
  `data/pollsters-quality.csv` del repositorio) y que cada punto lleva su fuente. Se mantiene la fórmula de la
  mezcla y la lectura de `k = 3`, con una frase nueva: es la relación medida entre lo que una casa varía de sondeo a
  sondeo y lo que varían las casas entre sí (unas 3-4).
- Misma página, lista de pesos de `n`: desaparece el punto «Por su cercanía a la votación»; la frase sobre el
  descarte de peso < 0,01 pasa al punto de posición o al párrafo final; el párrafo «Con estos pesos, cada elección
  aporta pocos sondeos efectivos…» se sustituye por uno que diga que una casa con decenas de sondeos en varias
  elecciones tiene un prior que pesa en torno al 10 % y una nueva está dominada por él. Los textos del método dan
  órdenes de magnitud, no cifras de casas concretas: las cifras exactas quedan en la tabla de D3, que Luis confirma
  tras recalcular.
- `/metodo`, sección «Los datos»: «cada encuestadora recibe una calidad previa» → «cada encuestadora recibe una
  calidad previa según una rúbrica pública (ver El rating)».
- `_glosario.html`: entrada «Calidad previa» (si falta) con una definición de una línea y la regla de puntos.
- Las afirmaciones con cifras («unas 3-4», «en torno al 10 %») son comprobables: el constructor las verifica con
  la réplica de D3 contra la base (solo lectura) antes de publicarlas, como en las fases 7 y 9 (*claim check*). El
  recálculo completo de los ratings (`Computer.compute_ratings`) no es tarea del constructor: es el paso 2 de D6.

**API.** `/api/v1/pollsters/{id}` añade la clave `quality` al sobre (`{contract, scope, pollster, ratings,
house_effects, herding, quality}`): el registro de la réplica o `null`. `/api/v1/pollsters` no cambia (sus filas ya
llevan `quality`). `docs/web/contrato.md` documenta el campo y la tabla.

**Tests.** `tests/test_pages.py` (perfil con y sin evaluación, texto «sin evaluar», tabla de criterios, nota
provisional), `tests/test_webapi_unit.py` (la clave `quality` del perfil y la caché), `tests/test_data_files.py`
(validez del CSV y del catálogo contra la rúbrica), `tests/test_web_api.py` (el sobre de la API). Los textos nuevos
del método pasan por el *claim check* de la fase 7.

## D5. Evaluación inicial

Se evalúan las casas con algún sondeo desde el 1 de enero de 2019 en cualquier ámbito (unas 60, incluidas las
`aggr` y `online`, cuya calidad es informativa). El listado de casas, sus patrocinadores y la proporción de sus
sondeos con muestra y fechas salen de consultas de solo lectura a la base (`psql` contra la RDS funciona desde el
intérprete de comandos; `DB_STATEMENT_TIMEOUT` solo afecta a los jobs). Claude redacta el borrador de cada fila con sus
fuentes (webs de las casas, fichas técnicas de sondeos recientes, listas de miembros de Insights+Analytics España y
ESOMAR, patrocinadores y muestras de la base) y lo deja en `status = borrador`; Luis revisa cada fila, corrige lo que
haga falta y la pasa a `revisada`. Cuando la fuente es la base, se cita como «base de datos del proyecto, consulta de
patrocinadores a {fecha}» (o «de muestras»), para que un lector sepa qué se miró. Lo que no se encuentre vale 0 en su criterio, con la nota «no comprobado». Las casas sin sondeos desde
2019 no entran en el CSV: "sin evaluar", calidad 5.

## D6. Operación (pasos de Luis tras integrar `dev`)

1. `save_catalogues()` (desde el notebook `PollsCompute` o un intérprete), que crea `pollsters_quality` y
   reescribe `pollsters.quality`.
2. Recalcular los ratings como hoy: `PollsCompute.ipynb` con `compute_ratings(save=True, scopes='all')` para `es` y
   para cada ámbito autonómico publicado (el cambio de D3 afecta a todos los eventos de `pollsters_ratings`).
3. Publicar como siempre (`python job.py publish …`: `forecast`, `analysis`, `event`) para que el promedio y las
   páginas usen los ratings nuevos; el perfil y la API leen la réplica en vivo y no necesitan publicación.
4. Comprobar la tabla «Efecto esperado» de D3 contra `pollsters_ratings` recalculada (n, prior y rating ajustado de
   las siete casas) y revisar las filas `borrador` del CSV (D5).

Hasta el paso 1, la tabla `pollsters_quality` no existe: el perfil y la API deben tratar la tabla ausente como «sin
evaluar» para todas las casas, sin error.

## Decisiones mías (a revisar al leer)

1. **Tope 80 y 8 puntos por punto.** Alternativa descartada: 10 por punto (tope 100, una casa nueva perfecta pesaría
   más que GAD3).
2. **Evaluar la casa de hoy, no por épocas.** Una fila por casa; la historia de los ratings guarda la calidad que se
   usó en cada evento (`pollsters_ratings.quality`), así que el pasado publicado no se reescribe hasta que se recalcula.
3. **Fuente por criterio** en el CSV, no una lista única: cada punto cita lo suyo.
4. **El catálogo de la rúbrica en JSON**, no en la plantilla: los textos del método, del perfil y de la validación
   salen de un solo sitio.
5. **Se elimina `week_decay`** en lugar de dejarlo a 1: un parámetro que no debe usarse es una trampa para el
   siguiente que lo lea.
6. **`save_catalogues` borra las filas de casas retiradas del CSV** y pone 5 a las no evaluadas: la base es una réplica
   completa, no un acumulado.
7. **Las casas `aggr` y `online` se evalúan también**, con una frase en su perfil que recuerde que no entran en el
   promedio.
8. **La tarjeta de calidad va antes de «Camino al rating»**: es lo primero que se pregunta un lector al ver la
   calidad en el KPI.
9. **El KPI «Calidad previa» muestra la calidad de hoy** (`pollsters.quality`), no la de la última fila de ratings:
   KPI y tarjeta no pueden contradecirse. La histórica se queda en «Camino al rating».
10. **El constructor verifica las cifras con la réplica en pandas, no ejecutando `Computer`**; el recálculo real es de
    Luis (D6). Los textos del método dan órdenes de magnitud; las cifras exactas, la tabla de D3.
11. **El procedimiento con que se midió k queda escrito y reproducible** (celda del notebook de laboratorio) y el
    método dice «unas 3-4»: es una medición aproximada que respalda el 3, no un ajuste.

## Lo que se deja al constructor

- Nombres exactos de funciones, módulo de la puntuación, tipos de las columnas en `PollstersQuality`.
- Maquetación de la tarjeta y de la tabla del método dentro del sistema de estilos actual; sin JavaScript nuevo.
- El orden de los commits y el detalle de los tests, siguiendo las convenciones de las fases anteriores.
- Las búsquedas de fuentes de D5 y su registro, siempre que cada fila cumpla las normas de D1.
