# Rating: evidencia por elección, experiencia en generales y análisis autonómicos sin ciclo

Fecha: 2026-10-20. Estado: aprobada por Luis el 2026-10-20 («Ok»). La descripción se acordó en el chat del 20 de octubre
(«Me gusta», «¿Y meterlo como un criterio más en el quality?», «Ok, adelante» y la fusión de estándares y generales en
un solo criterio). Revisión del constructor hecha el mismo día: tres preguntas (recuento de generales de agregadores y
paneles, Opina 360, medias descriptivas por elección), resueltas como decisiones mías 4, 7 y 8; y dos hechos del código
incorporados a D1 y D5. Esta spec solo describe lo que cambia
respecto de `dev` en `922161b`. Se apoya en la spec de la calidad previa
(`docs/superpowers/specs/2026-10-16-calidad-previa-design.md`), que deja el rating y la rúbrica como están hoy.

## Intención

Luis vio que CENTRA (Centro de Estudios Andaluces) es la primera del ranking de /casas, con un rating de 75,3, después
de sondear solo dos elecciones, Andalucía 2022 y 2026, y ninguna general. Además, su perfil dice «Esta encuestadora no
tiene sondeos en las elecciones analizadas». Quiere entender por qué y reforzar el cálculo del rating, dando más
peso a la experiencia en elecciones generales.

El análisis del 20 de octubre encontró dos causas independientes:

1. **El rating cuenta la evidencia por sondeos, como si fueran independientes.** `Computer.pollster_ratings` promedia
   las desviaciones de todos los sondeos de una casa (12 en el caso de CENTRA) y calcula su error típico como si cada
   sondeo fuera una prueba independiente (0,32 en la escala de sesgo). Pero los sondeos de una misma elección
   comparten la suerte de esa elección. Medido sobre la bolsa global de sondeos (todos los ámbitos, sondeos con
   `bias` evaluable antes de 2026-11-29), en log odds ratio:

   | Medida | Valor |
   |---|---|
   | σ: desviación típica de la desviación de una casa entre elecciones (casas con ≥ 3 elecciones, ponderada por antigüedad × ámbito) | 0,0137 |
   | τ (`bias_dev_tau`) | 0,0100 |
   | Desviación típica de las medias de las casas (≥ 5 elecciones) | 0,0060 |

   El ruido de una sola elección es mayor que la diferencia real entre casas. Con dos elecciones, el error típico de
   CENTRA es ≈ 0,99, no 0,32. La contracción bayesiana apenas actúa y su `rating_adj` llega a 96. Después, la mezcla
   con la calidad previa (k = 3 sondeos frente a 2,8 sondeos ponderados) solo lo deja a medias.
   El mismo fallo hace que Opina 360, sin ningún sondeo evaluable, sea tercera (72) solo por su calidad previa.
2. **El análisis de un ámbito recién votado no se publica.** `publish_analysis` llama a
   `Computer.print_house_effects`, que ajusta el promedio del ciclo actual. En Andalucía (próxima elección 2030-06-16)
   ese ciclo aún no tiene sondeos, y salta con `NothingToPublish` («No polls for es-an 2030-06-16: nothing to
   forecast»). Sin `analysis/es-an/` en el bundle no hay `poll-errors` de sus elecciones pasadas, así que la diana y las
   tablas del perfil de CENTRA quedan vacías. En S3 faltan también `es-ar`, `es-cl` y `es-ex`.

Luis añadió que quiere dar más prominencia a las casas con experiencia en generales («las autonómicas son más
sencillas muchas veces de pronosticar»). Medido con el último sondeo de cada casa a 30 días o menos de la elección
(65 autonómicas, 14 generales):

| | Autonómicas | Generales |
|---|---|---|
| Error medio por partido | 2,36 | 2,30 |
| Error en el margen entre bloques | 4,34 | 4,72 |
| Error común de la industria (margen) | 4,46 | 4,09 |
| Dispersión entre casas en una misma elección (`bias_dev_adj`) | 1,28 | 1,68 |

Las autonómicas no son más fáciles. Lo que sí pasa es que en ellas las casas se separan menos, así que distinguen
peor a unas de otras. Sobre si acertar en autonómicas predice acertar en generales: la correlación es 0,04 (13 casas
con ≥ 3 de cada), pero generales contra generales da 0,02. Con 14 generales los datos no deciden ni a favor ni en
contra.

Tampoco hay base empírica fuerte para ningún criterio de la rúbrica. Prueba del 20 de octubre: desviación de cada casa en
cada elección, centrada en la media de esa elección, diferencia entre grupos de la rúbrica e intervalo del 95 % por
remuestreo de encuestadoras (59 casas evaluadas con elecciones, escala de sesgo, negativo = más cerca del resultado
que el consenso):

| Comparación | Diferencia | IC 95 % | Casas |
|---|---|---|---|
| Estándares 1 − 0 | −0,31 | [−0,66, +0,07] | 11 / 48 |
| Generales previas 1 − 0 | −0,01 | [−0,32, +0,32] | 31 / 59 |
| Implantación 2 − 0 | −0,24 | [−0,50, +0,07] | 9 / 59 |
| Implantación 1 − 0 | +0,12 | [−0,33, +0,52] | 22 / 59 |
| Independencia alta − baja | −0,34 | [−0,70, +0,16] | 23 / 36 |
| Transparencia ≥ 2 − < 2 | +0,72 | [−0,78, +1,03] | 7 / 52 (sin el CIS: +0,05) |

Ninguna diferencia es concluyente con tan pocas casas. Por eso la calidad previa se presenta como lo que es: un prior
con criterios observables y documentados, no un predictor validado del acierto, cuyo peso (k ≈ 2 elecciones) se
diluye a medida que hay evidencia. El método lo dice con esta tabla.

Luis decidió fusionar estándares y generales en un solo criterio porque ninguno de los dos demuestra por sí solo, en
España, que una casa sea más fiable (las más transparentes, como GAD3 o 40dB, no tienen estándares), aunque eso
penalice a GfK y ODEC.

Bajar el peso de las autonómicas (0,5 → 0,25) no da prominencia a las casas con generales, porque lo que hace es
acercar a todas a su calidad previa. Las que más suben son las que no tienen generales y tienen una calidad previa
alta (GfK, CPS, inPactos, CIES). Por eso la experiencia en generales entra como criterio de la calidad previa, que es
el prior que manda justo cuando hay poca evidencia.

## Decisiones de Luis (2026-10-20)

1. La evidencia del rating se cuenta por elecciones, no por sondeos.
2. El peso de las autonómicas se queda en 0,5 (`rating_weight` de `data/es-scopes.csv`).
3. La experiencia en generales entra en la calidad previa, fusionada con los estándares en un solo criterio de 0 a 2
   (propuesta de Luis): 0 si la casa no ha sondeado ninguna general, 1 si ha sondeado al menos una y 2 si además está
   asociada o certificada. Nombre: «Implantación» (confirmado por Luis).
4. La rúbrica sigue sumando 10 puntos: trabajo de campo 3, transparencia 3, independencia 2 e implantación 2.
   «Identidad» deja de ser un criterio con punto propio y pasa a ser condición de la transparencia: sin entidad
   identificable, la transparencia vale 0. No hay reescala: 8 puntos de calidad por punto de rúbrica, suelo 5.
5. El ranking de /casas muestra cuántas generales ha sondeado cada casa, con una columna «Generales» y un filtro para
   ver solo las que tienen experiencia en generales.
6. Se arregla la publicación del análisis de los ámbitos sin sondeos en el ciclo actual.
7. Antes de dar el cambio por bueno, se pasa el backtest antes y después.

## Objetivos

- **D1. Análisis autonómico sin ciclo.** `publish analysis` publica el análisis de un ámbito aunque su próxima
  elección no tenga sondeos todavía: los errores, los datos de error y el rebaño de sus elecciones pasadas. Solo se
  omiten los efectos de casa del ciclo actual. Tras republicar, el perfil de CENTRA muestra la diana con Andalucía
  2022 y 2026.
- **D2. Rating por elección.** `pollster_ratings` agrupa las desviaciones por elección, calcula el error típico con la
  variación entre elecciones y mezcla con la calidad previa usando la evidencia medida en elecciones.
- **D3. Criterio «Implantación» y rúbrica reorganizada.** La rúbrica pasa a ser: trabajo de campo 0-3, transparencia
  0-3 (0 sin entidad identificable), independencia 0-2 e implantación 0-2 (generales y estándares). La parte de
  generales se calcula a la fecha de cada elección.
- **D4. Web.** Columna «Generales» y filtro en el ranking de /casas. La tarjeta «Confianza» del perfil, el método
  (/metodo/rating y la explicación de la rúbrica), el glosario y la API reflejan los cambios.
- **D5. Backtest antes y después**, sin tocar los ratings de producción ni `backtest/results/`.

## No objetivos

- Sin cambios en τ (`bias_dev_tau = 0,01`), el decaimiento por posición (0,5), el decaimiento por antigüedad (0,9), el
  peso de los ámbitos (`rating_weight`) ni la fórmula del peso en el promedio
  (`weight_rating = log1p(R) / log1p(mean R)`). Se observó que τ podría ser ancho frente a la dispersión real entre
  casas (0,006). Queda anotado para otra vez, sin cambiarlo aquí.
- Sin regla de mínimos para entrar en el ranking: Luis retiró una así el 2026-10-15. La prominencia la dan el criterio
  y el filtro.
- No se quita ninguna casa del ranking.
- No se separa el ranking en generales y autonómicas. El ranking sigue siendo único (M11).
- Sin cambios en los efectos de casa (M6), el rebaño (M13) ni la muestra efectiva (M12).
- No se reevalúa a mano ninguna casa: las notas de `data/pollsters-quality.csv` se conservan tal cual, incluida la
  columna `identity`.

## Anti-objetivos (lo que haría fracasar el cambio aunque funcione)

- Que el criterio «Implantación» mire al futuro: el rating de una elección pasada, el backtest o el camino al rating del
  perfil no pueden contar generales posteriores a esa elección.
- Que una casa con muchas elecciones vea su rating dominado por la calidad previa: la calidad previa sigue siendo el
  prior y debe perder peso con la evidencia.
- Que el backtest «después» escriba en las tablas de ratings de la RDS o en `backtest/results/`, que son de los jobs
  de Luis (incidente del 2026-10-19).
- Que se publique un ámbito con un análisis a medias sin que la web lo sepa: si falta una parte, las páginas deben
  mostrar el aviso de siempre en esa tarjeta, no un error.

## Diseño

### D1. Análisis sin sondeos en el ciclo actual

Hoy `mtpy/lib/publish/analysis.py::publish_analysis` envuelve todo el análisis en un `try` y convierte en
`NothingToPublish` el `ValueError` de `print_house_effects` (línea ≈ 224). El cambio:

- Si el ciclo actual no tiene sondeos, el análisis se publica igual, con todas las partes que dependen de elecciones
  pasadas (`poll-errors`, `error-data`, `herding`, `herding-events` y sus CSV) y sin la de efectos de casa del ciclo
  (`house-effects`).
- `meta.json` lo dice explícitamente, por ejemplo con `house_effects: false` o un motivo. El nombre exacto lo decide
  el constructor.
- `NothingToPublish` queda solo para cuando no hay nada que publicar: un ámbito sin ninguna elección pasada con
  sondeos.
- Los consumidores de `house-effects` (la tarjeta de efectos de /casas con su selector de ámbito y lo que lo lea en
  `mtpy/lib/pages` y `mtpy/lib/webapi`) tratan la parte ausente como ya tratan un análisis no publicado: aviso en la
  tarjeta, sin error. El constructor localiza todos los lectores.
- El docstring de `Publish._analysis_scope` ya promete este comportamiento para los ámbitos sin próxima elección. Se
  amplía a los ámbitos con próxima elección pero sin sondeos.

Comprobación: `publish analysis` para `es-an`, `es-ar`, `es-cl` y `es-ex` debe dar `published`, no `skipped`. Ojo:
con `S3_BUCKET` definido, el modo en seco escribe en S3 bajo `site-dry/`. La comprobación del constructor se hace con
un escritor sobre el sistema de archivos local (sin `S3_BUCKET`, o con el escritor local que usen los tests).

### D2. Rating con la evidencia agrupada por elección

En `Computer.pollster_ratings`. Una «elección» es un par ámbito y fecha (`event_key`, como hoy).

1. **Peso de cada sondeo dentro de su elección:** `w = weight_over · weight_sample · weight_pos`. Se mantiene el
   descarte de los sondeos cuyo peso total, el de hoy (incluidos antigüedad y ámbito), es menor que 0,01.
2. **Desviación de la casa en cada elección:** `d_e = Σ w·bias_dev_adj / Σ w`, en log odds ratio.
3. **Peso de cada elección:** `W_e = weight_year · weight_scope`. Lo calcula `poll_rating_weights` como hoy y es el
   mismo para todos los sondeos de la elección.
4. **Desviación media de la casa:** `d = Σ W_e·d_e / Σ W_e`. Evidencia: `W = Σ W_e`, que se guarda como
   `num_events_w`. Número efectivo de elecciones: `n_eff = W² / Σ W_e²`.
5. **Error típico:** `se = σ / √n_eff`, con `σ = bias_dev_sigma`, un parámetro nuevo de `Computer` con valor por
   defecto 0,014 (lo medido, redondeado). Se guarda en `bias_dev_err`, que conserva su significado: el error típico de
   la desviación de la casa.
6. **Contracción hacia 0** con el prior N(0, τ²): `d_post = d · τ² / (τ² + se²)`. Es la misma `bayes_adjust`, ahora
   con el error agrupado. Se guarda en `bias_dev_adj`.
7. **Rating de la evidencia:** `rating_adj = Φ(−d_post / τ)`, como hoy. Las casas sin evidencia toman su calidad
   previa, también como hoy.
8. **Mezcla con la calidad previa:** `R = (W·rating_adj + k·Q) / (W + k)`, con `k = (σ/τ)²` (≈ 1,96 con 0,014 y
   0,01): el peso del prior en un modelo jerárquico, ahora contado en elecciones. `Q` es la calidad previa de la casa
   a la fecha de la elección que se puntúa (D3).
9. **`min_polls`** deja de usarse en el rating. Si no lo usa nada más, se elimina del constructor de `Computer` y de
   sus docstrings. El constructor lo comprueba.
10. **Medir σ:** un método de `Computer`, por ejemplo `measure_event_sigma(pool)`, devuelve σ con el mismo
    procedimiento de la medición del 20 de octubre: desviación típica ponderada por `W_e` de `d_e` alrededor de la
    media de su casa, solo con casas de ≥ 3 elecciones. Sirve para documentar y revisar el valor por defecto, no
    para recalcularlo en cada ejecución.

**Columnas de `pollsters_ratings`:**
- Se mantienen `num_events`, `num_polls` y `num_polls_w`. Esta última sigue siendo la suma de los pesos de los
  sondeos y queda como dato informativo.
- Las medias descriptivas por casa (`error_avg`, `error_blocks`, `error_within`, `bias_avg`, `bias_blocks`,
  `bias_within`, `bias`) siguen ponderadas por sondeo con el peso de hoy: describen sus sondeos, no son el rating.
  Solo cambian `bias_dev_adj` («Diferencial» en el ranking), `bias_dev_err`, `rating_adj`, `rating` y
  `weight_rating`.
- Se añaden `num_events_w` (W) y `num_generals`: el número de generales con sondeos de la casa celebradas antes de la
  elección puntuada, con la regla de recuento de D3 (no la bolsa del rating).
- Luis añade las columnas a mano en la RDS (prefiere hacer él los cambios de esquema). El constructor deja el
  `ALTER TABLE` escrito en la sección de pasos de Luis y actualiza el modelo de la tabla en `mtpy/models` y la réplica
  CSV si la hay.

Resultado esperado con los datos del 20 de octubre (prototipo con k ≈ 1,9 y la calidad previa de hoy, sin el criterio
«Implantación»): GAD3 77,3 (1ª), Metroscopia 70,9, CEMOP 67,3, IMOP 66,5, CENTRA 64,5 (5ª, antes 1ª), Simple Lógica de
17ª a 29ª. Con el criterio de D3 (prototipo de +1 por generales), CENTRA queda alrededor del 8º puesto. Las cifras finales saldrán del código real y
se comprobarán con un script de verificación contra la base, en modo lectura.

### D3. Rúbrica con «Implantación» y sin «Identidad» como criterio

`data/pollsters-quality-criteria.json`:
- **Salen** `identity` y `standards` como criterios puntuables.
- **`transparency`** mantiene sus niveles 0-3 y añade la condición: sin entidad identificable (`identity = 0` en el
  CSV), la transparencia cuenta 0. Los textos de los niveles lo dicen («Ficha técnica mínima, publicada por una
  entidad identificable: …»).
- **Entra `implantation`**, «Implantación», máximo 2:
  - 0: «Solo ha sondeado elecciones autonómicas o locales.»
  - 1: «Ha sondeado al menos una elección general.»
  - 2: «Ha sondeado al menos una elección general y es miembro de Insights+Analytics España o de ESOMAR, o está
    certificada ISO 20252.»
  - Los textos hablan de lo que se comprueba, no del tamaño: GfK, una multinacional, tiene 0 porque en la base solo
    tiene autonómicas.
  - El criterio es mixto: la parte de generales se calcula y la de estándares se evalúa a mano, con su fuente. El JSON
    lo marca (por ejemplo `"computed": "general"`) para que el validador no busque la columna en el CSV y la web
    presente cada parte con su origen.

`data/pollsters-quality.csv` no cambia de columnas. `identity`, `identity_source`, `standards` y `standards_source` se
siguen validando (0-1, con fuente si vale 1), porque son la evidencia de la condición de transparencia y del nivel 2.

`mtpy/lib/quality.py`:
- Puntos manuales = `fieldwork + transparency·[identity = 1] + independence` (0-8).
- Implantación a la fecha de una elección = `general · (1 + standards)`, donde `general` vale 1 si la casa tiene al
  menos una general en el recuento (abajo) celebrada estrictamente antes de esa fecha.
- Calidad a esa fecha = `max(floor, per_point · (puntos manuales + implantación))`.
- Las casas sin evaluar tienen 0 puntos manuales y `standards = 0`: valen 5, u 8 si tienen generales.
- **Recuento de generales:** elecciones con `event_scope = 'es'` en las que la casa tiene al menos un sondeo con `bias`
  no nulo, sin contar los de `ctype` `wban` y `exit` (no son sondeos de la casa en el sentido de la rúbrica) y sin
  filtrar por `mtype`, para que el dato público de paneles y agregadores también sea cierto (ElectoPanel y KeyData
  tienen 3 generales cada una). Se obtiene de los sondeos de `es` por una vía propia, no de la bolsa del rating, que
  quita paneles y agregadores y que puede no incluir `es`.
- La regla vive en una sola función con dos usos:
  - En el rating, la calidad de cada elección puntuada: `rate_events` pasa a `pollster_ratings` la calidad a la fecha
    del objetivo; para `NaT` o la próxima elección, todas las generales pasadas.
  - La calidad de hoy en `pollsters.quality`, que escribe `save_catalogues()` y que usan el KPI del perfil, la columna
    «Confianza» y el relleno del Forecaster para las casas sin rating. Se calcula con las generales celebradas hasta
    hoy.
- Las casas excluidas del rating por `drop_mtypes` siguen fuera del rating y del promedio. Su calidad es informativa.

Efecto con las notas actuales: 43 de las 72 casas evaluadas tienen generales. Cambian de calidad 31, todas a la baja
salvo dos:

| Encuestadora | Generales | Antes | Después | Por qué |
|---|---|---|---|---|
| Opina 360 | 0 | 72 | 64 | sin generales |
| CPS | 0 | 64 | 56 | sin generales |
| GfK | 0 | 64 | 48 | sin generales y pierde estándares |
| CENTRA, Nexo, inPactos, CIES, UCAM, OBEDE | 0 | 56 | 48 | sin generales |
| Gizaker, Invest Group, Ateneo del Dato | 0 | 48 | 40 | sin generales |
| EM Analytics, Ikertalde | 0 | 40 | 32 | sin generales |
| More in Common, Idus3, Gadeso | 0 | 32 | 24 | sin generales |
| Murcia Electoral | 0 | 32 | 24 | sin entidad identificable |
| ODEC | 0 | 32 | 16 | sin generales y pierde estándares |
| Perfiles, NueveCuatroUno, Aztiker, DemosGal, Commentia, Winston | 0 | 24 | 16 | sin generales |
| Social Data, TSA, Lápiz Estratégico | 0 | 16 | 8 | sin entidad identificable |
| UPNA | 0 | 16 | 8 | sin generales |
| KeyData | 3 | 8 | 16 | gana generales; sin entidad (transparencia ya 0) |
| GI Internacional | 1 | 8 | 16 | gana generales; sin entidad (transparencia ya 0) |

El resto (CIS, GAD3, Sigma Dos, ElectoPanel, etc.) conserva su calidad. El constructor recalcula la tabla con el
código real y la deja en Resultados.

### D4. Web

- **Ranking de /casas:**
  - Columna «Generales» (`num_generals` de la última fila nacional de ratings: el número, no el criterio), justo después de «Elecciones», con su
    `title` y su entrada en el glosario desde la misma fuente (`pages.casas.table_columns`, `COLUMN_LABELS`).
  - Filtro «Con experiencia en generales» en la barra de filtros del ranking, apagado por defecto, con el mismo
    mecanismo y estilo que los filtros existentes. Si los filtros se guardan en el hash o en la URL, este también.
- **Perfil, tarjeta «Confianza»:**
  - El criterio «Implantación» aparece como un criterio más, con su nivel, y separa sus dos partes: las generales
    («calculado a partir de los sondeos publicados», sin fuente) y los estándares (con su fuente, como hoy).
  - La transparencia explica la condición de identidad cuando se aplica (`identity = 0`).
  - El KPI usa la calidad de hoy.
- **/metodo/rating:**
  - Se reescribe el párrafo de la evidencia y de k («k vale 3: la calidad previa pesa como tres sondeos…», en
    `web/templates/metodo_rating.html`): la evidencia se cuenta por elecciones, el porqué con las cifras medidas (σ,
    τ y k ≈ 2 elecciones) y lo que significa (una casa con dos autonómicas pesa menos de una elección general
    completa).
  - Se explica el criterio «Implantación» y por qué no se bajó el peso de las autonómicas, con las cifras de la tabla de
    la Intención.
  - En la explicación de la rúbrica se dice que la calidad previa es un prior de criterios observables, no un predictor
    validado, con la tabla de la prueba por criterio de la Intención (la cifra la vuelve a calcular el constructor con
    el código final y la deja en Resultados).
  - Como en fases anteriores, cada cifra y afirmación del texto se comprueba contra el código o los datos.
- **Rúbrica en el método:** se genera desde el JSON, así que debe salir bien sin tocarla. El constructor lo comprueba
  y añade la frase de la condición de identidad si la plantilla no la muestra.
- **API** `/api/v1/pollsters/{id}` y la de ratings: incluyen `num_generals` y `num_events_w`, y el desglose de la
  calidad incluye `implantation` con sus dos partes.
- Palabras: «encuestadora» (nunca «casa» en los textos de /casas, `test_houses_pages_use_encuestadora`), «Confianza»
  como nombre público de la calidad previa, «Implantación» para el criterio y «Generales» para la columna.

### D5. Backtest antes y después

- El backtest del modelo (`backtest/run_backtest.py`, 5 generales, todos los horizontes) usa los ratings guardados en
  la base. El `Forecaster` toma `weight_rating` de la tabla `polls` (no de `Computer.load_ratings`).
- Para el «después», el constructor añade una forma de recalcular los ratings en memoria con el código nuevo
  (`compute_ratings(save=False, scopes='all')`) y de inyectar el `weight_rating` resultante en el camino
  `Simulator` → `Forecaster`, activada por una opción del backtest, por ejemplo `--recompute-ratings`. Así no se
  escribe en la RDS.
- Los ratings recalculados respetan la regla de cada caso del backtest: solo elecciones anteriores a la evaluada.
- El «antes» se ejecuta con la misma opción sobre el código de `922161b`, o con los ratings actuales de la base si
  ya son los de ese código. El constructor elige y lo dice en el informe.
- Las salidas van a un directorio aparte, por ejemplo `files/backtest/rating-por-eleccion/{antes,despues}`, nunca a
  `backtest/results/`.
- Ámbito: `es` obligatorio. Los autonómicos con resultados ya en `backtest/results/es-*`, si el tiempo de ejecución es
  razonable (el constructor mide uno y decide).
- Informe: tabla antes/después de `mae_shares`, `mae_seats`, `crps_seats`, `brier_vs` y coberturas por horizonte, en
  la sección de resultados de esta spec. El cambio no se condiciona al resultado: Luis decide con el informe delante.
  Si empeora de forma clara y consistente, se señala.

## Restricciones

- Rama `dev`. Luis hace commits en paralelo en el mismo árbol (`git commit -a`): revisar `git status` y `git log`
  antes de cada commit y no reescribir nunca sus commits.
- Ninguna escritura en la RDS ni en S3 por parte del constructor. Los recálculos, las publicaciones y los
  `ALTER TABLE` son de Luis (sección de pasos). Las comprobaciones contra la base son de solo lectura.
- `backtest/results/` y los CSV que escriben los jobs de Luis no se tocan ni se revierten.
- Tests con `pytest` (1134 hoy): todos en verde al terminar. Cada cambio de comportamiento lleva su test, incluidos
  el criterio a la fecha (sin mirar al futuro), la condición de identidad, el agrupado por elección (una casa con
  muchos sondeos en una sola elección no puede tener más evidencia que una elección) y el análisis sin ciclo.
- La regresión fijada de `test_model.py` ([141, 107, 62, 8]) puede moverse, porque cambia `weight_rating`. Si se
  mueve, se vuelve a fijar y se anota en el informe con las cifras antiguas y nuevas.
- PEP8, docstrings en todas las funciones, el mismo estilo que el código de alrededor (AGENTS.md).

## Decisiones mías (Luis puede cambiarlas)

1. **σ como parámetro fijo** (0,014) con un método para medirlo, en lugar de estimarlo en cada ejecución: estable y
   reproducible, como se hizo con k.
2. **La mezcla usa W = Σ W_e** (la suma de los pesos de las elecciones), no `n_eff`, para conservar el significado
   actual de «cantidad de evidencia ponderada».
3. **Las casas sin evaluar con generales valen 8, no 5:** la parte de generales es un dato, no una evaluación.
4. **Recuento de generales:** sin filtrar por `mtype`, sin `wban`/`exit` y por una vía propia, no desde la bolsa del
   rating. La misma cifra alimenta el criterio, `num_generals` y la columna del ranking, así que no pueden
   contradecirse. Agregadores como KeyData cuentan sus generales aunque no hagan campo: su campo ya vale 0.
5. **`num_polls_w` se conserva** como dato informativo. Se añaden `num_events_w` y `num_generals`, en lugar de
   reutilizar columnas con otro significado.
6. **Directorio del backtest:** `files/backtest/rating-por-eleccion/` (está en `files/`, ignorado por git).
7. **Opina 360 no necesita arreglo aparte:** el ranking ya deja fuera a las casas sin sondeos evaluables
   (`num_polls = 0`, `mtpy/lib/pages/casas.py`), así que no aparece en /casas. Su rating sigue siendo su calidad previa
   (64), que es el peso de sus sondeos en el promedio, como decidió Luis el 2026-10-16 para las casas sin evidencia.
8. **Las medias descriptivas por casa no se agrupan por elección** (ver D2).

## Lo que queda al constructor

- El nombre de los campos nuevos de `meta.json` del análisis y cómo marca la web la ausencia de `house-effects`.
- La forma de la opción del backtest que recalcula los ratings en memoria.
- Dónde vive la función del criterio (`mtpy/lib/quality.py` es el sitio natural) y cómo recibe las generales de cada
  casa (de la bolsa del rating en `compute_ratings` y de una consulta en `save_catalogues`).
- El texto final de /metodo, dentro de lo que dice D4.
- El diseño del filtro, dentro del estilo de la barra de filtros existente.

## Pasos de Luis tras la integración

1. Añadir las columnas en la RDS:
   `ALTER TABLE elections.pollsters_ratings ADD COLUMN num_events_w double precision, ADD COLUMN num_generals integer;`
   (ajustar al tipo de las demás columnas numéricas de la tabla).
2. `save_catalogues()`, para que `pollsters.quality` sea la calidad de hoy con el criterio nuevo.
3. Recalcular los ratings: `python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'`.
4. Publicar: `publish analysis` con `scopes: all` (ahora incluye `es-an`, `es-ar`, `es-cl` y `es-ex`), y después los
   eventos y la previsión, como de costumbre.
5. Revisar /casas (CENTRA, GfK, CPS, Opina 360), el perfil de CENTRA (con la diana de Andalucía) y /metodo/rating.

## Resultados

(Lo rellena el constructor: tabla del backtest antes/después, ranking resultante, regresión de `test_model.py`.)
