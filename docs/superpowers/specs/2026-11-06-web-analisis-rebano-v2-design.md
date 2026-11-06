# Web, Análisis: «¿Se comportan las encuestadoras como un rebaño?» (`/analisis/rebano`, versión 2)

Sustituye por completo el contenido de la versión 1 (`2026-11-05-web-analisis-rebano-design.md`, construida y
publicada el 2026-11-05): misma dirección y mismo identificador de estudio (`rebano`), cálculo, partes y página nuevos.

## Intención

Luis quiere un análisis que conteste **con datos y con todas las elecciones que tengamos** dos preguntas:

1. ¿Existe realmente un «efecto rebaño» entre las encuestadoras?
2. En caso afirmativo, ¿es mayor según se acercan las elecciones?

Es contenido para redes (hilo hacia el 12-15 de noviembre, antes del 29-N) y para la web: cada tarjeta tiene que
poder publicarse sola y entenderse de un vistazo. La página sigue «viva» con las generales de 2026 hasta su escrutinio.

La versión 1 no contestaba esto: medía sólo las generales desde 2008 (7 ciclos), su pregunta 2 se apoyaba en «3 de 7
ciclos por debajo del azar» (con todas las elecciones y la banda del azar, sólo 2016 lo está de verdad), y su pregunta
3 era una tabla de casas de 2026 con un único sondeo medible. Luis las consideró irrelevantes («la 2 habla sólo de la
anomalía del CIS», «la 3 da información sobre casas con rebaño a estas alturas»).

## Decisiones de Luis (2026-11-06)

- **Sin selectores** en el análisis; cada tarjeta, publicable de un vistazo.
- **Vocabulario**: nada de «se copian» ni «mirar al vecino»; el marco es **«se comportan como un rebaño»** («se
  inspiran», «se protegen» valen como matiz). **No explicar intenciones** (revirtió el párrafo «Por qué» el mismo día):
  la página dice lo que pasa, no por qué.
- **Todas las elecciones publicadas**, generales y autonómicas, de todas las fechas con sondeos.
- **Respuesta 1 = el recuento conjunto**: elección a elección la bajada no supera al azar, pero que ocurra en 19 de 23
  es muy improbable. **Fuera** la prueba «se parecen más de lo que daría el puro azar» (5 de 38 elecciones: «no son
  muchas»).
- Estructura aprobada («Perfecto, adelante»): cabecera con tres KPIs, tarjeta 1 (barras por elección), tarjeta 2 (los
  dos recuentos de cuadritos), tarjeta 3 (2026 en directo), «Cómo lo medimos», «Datos y método».

## Hechos de partida (spike del 2026-11-06, scripts en el scratchpad de la sesión a00d6fa2)

Sobre el bundle publicado de Luis el 2026-11-06 (12 ámbitos en el manifiesto, 66 elecciones con parte `cycle`, 2.017
sondeos a 0-365 días de su elección sin agregadores ni paneles abiertos, 1.658 con muestra; CIS fuera):

| Comparación de la misma pareja de casas en la misma elección | Elecciones en que difieren menos | Signo (una cola) |
|---|---|---|
| lejos (56-365 días) → últimas 3 semanas (0-20) | 19 de 23 | p 0,0013 (1 entre 769) |
| lejos → 3-8 semanas (21-55) | 9 de 17 | p 0,5 |
| 3-8 semanas → últimas 3 semanas | 23 de 29 | p 0,0012 (1 entre 864) |

Elección a elección (permutación dentro de cada pareja de casas), lejos → últimas 3 semanas: sólo 3 de las 23 caen de
forma clara por sí solas (generales 2015 y 2016, Madrid 2021); una sube de forma clara. Las generales aportan la mayor
parte de las comparaciones (280 de 322 parejas de casas con las dos épocas). Mezclar elecciones sin emparejar engaña:
muchas autonómicas sólo tienen sondeos al final, y la dispersión bruta de «las tres últimas semanas» sale *más* alta
que la de lejos. Por eso toda comparación se hace **dentro de cada elección y con las mismas parejas de casas**.

Descartado en el spike: la prueba de la «flecha del tiempo» (¿se parece un sondeo más a los publicados antes que a
los de después?) no distingue en simulación un rebaño fuerte de ninguno; no se usa ni se menciona.

## Objetivos

- Contestar las dos preguntas con recuentos por elección y su probabilidad por azar, con todas las elecciones.
- Cifras vivas en todos los textos: nada escrito a mano en la plantilla.
- 2026 en directo: cómo van las mismas parejas de casas en las generales en curso; al publicarse su escrutinio entra
  sola en los recuentos.
- Cada tarjeta, una imagen descargable con su titular (el botón de tarjeta de las demás páginas de Análisis).

## No objetivos

- Medir casas individuales, el CIS o la precisión frente al resultado.
- Explicar por qué se acercan (indecisos, prudencia, inspiración): ni en el texto ni en notas.
- Cambiar la medida de rebaño del modelo (`herding`, /metodo/rating, /casas): es otra cosa y no se toca.

## Anti-objetivos

- Afirmar que se copian, o que una elección concreta «demuestra» el rebaño.
- Mezclar elecciones sin emparejar (la trampa de composición de arriba).
- Un selector, una pestaña o cualquier cosa que haya que manipular para ver el resultado.
- Un titular que diga más que el recuento: «en conjunto» y «muy improbable por azar», no «siempre».

## Enfoques considerados

- **A. Estrechez frente al azar** (¿se parecen más de lo que darían muestras independientes?): 5 de 38 elecciones en
  las tres últimas semanas, 0 antes. Conservadora (las diferencias de método empujan hacia arriba) y con poco que
  contar. **Descartada por Luis.**
- **B. Convergencia de las mismas parejas de casas, contada por elección** (elegida): fija el método de cada casa y la
  elección; contesta la pregunta 1 con el recuento lejos → final y la 2 con los dos tramos.
- **C. Flecha del tiempo**: falló la validación con simulación (dirección no identificable). Descartada.
- Formato: **tarjetas publicables sueltas** (elegido) frente a un único póster (pierde la segunda respuesta) y frente
  al formato de tres preguntas de los otros estudios (Luis pidió salir de él).

## D1. Declaraciones (`mtpy/lib/bundle.py`, `mtpy/lib/publish/studies.py`)

```python
STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current')
STUDY_SCHEMAS['rebano'] = {part: 'rebano-' + part for part in STUDY_PARTS['rebano']}  # versión @2
STUDY_CSV['rebano'] = ('elections',)
STUDY_CSV_OF['rebano'] = {'elections': ('elections',)}
```

- `SCHEMAS`: fuera `rebano-cycles`, `rebano-weeks`, `rebano-houses`; dentro `rebano-elections` y `rebano-current`, con
  las claves obligatorias de D3; versión de esquema 2 para las cuatro.
- `STUDY_TITLES['rebano']`: título «¿Se comportan las encuestadoras como un rebaño?» y la plantilla del subtítulo de
  D3. `STUDY_RUNNERS['rebano']` sigue `{'compute': …, 'registrable': False, 'live': True}`: la acción `studies` del
  `update` diario lo republica igual que hoy.
- `pages/analisis.py`: `CSV_LABELS['rebano'] = {'elections': 'Comparaciones por elección: las mismas parejas de casas
  en cada tramo'}`.
- Los ficheros viejos del estudio en S3 (`cycles`, `weeks`, `houses` y sus CSV) quedan huérfanos: el constructor
  decide si `publish_study` borra los ficheros de partes que ya no declara; si no, se anota para que Luis los borre.

## D2. Entradas y filtros

Con un `BundleReader(fs, prefix=bundle.PREFIX)` sobre el `fs` del job (también en `dry_run`), como ahora.

- **Elecciones pasadas**: todos los ámbitos de `manifest.scopes` y, en cada uno, cada fecha de `reader.list_events`
  con parte `cycle` **y** parte `results` (una elección sin escrutinio no es pasada). Sin fecha mínima (`since` nulo
  en los parámetros; el parámetro existe por si Luis quiere acotarlo).
- **Elección en curso**: sólo el ámbito `params.live_scope` (`'es'`): el run `manifest.scopes.es.latest` mientras su
  `event_date` no tenga `results` publicado (la regla de `read_current` de hoy). Los runs de otros ámbitos no entran.
- **Sondeos**: fin de campo (`end_date`, o `date` sin ella) entre 0 y `max_days` (365) días antes de la elección;
  `sample_size > 0`, `weight > 0`, `mtype` fuera de `DROP_MTYPES`, `pollster` fuera de `exclude` (`['CIS']`).
- **Series**: las `n_series` (4) con mayor media de voto en los sondeos de la elección (`main_series` de hoy, con los
  sondeos ya filtrados de la ventana, antes de quitar los que no tienen muestra: igual que el spike).

## D3. La medida, el cálculo y las partes

**Comparación.** Dos sondeos de casas distintas de la misma elección cuyo fin de campo difiere `pair_days` (7) días o
menos, en una serie en la que ambos tienen cifra. Su diferencia al cuadrado `d² = (x_i − x_j)²` frente a la que daría
el azar de sus dos muestras, `v = 1e4 · p (1 − p) (1/n_i + 1/n_j)`, con `p` la media de las dos cifras en tanto por
uno acotada a [0,005, 0,995]. Cada comparación cae en un **tramo** por la media de los días de sus dos sondeos, con
tramos semiabiertos para que una media en ,5 no se pierda: `final` [0, 21), `mid` [21, 56), `far` [56, 365]
(`params.periods`, límites inferiores incluidos y superiores excluidos salvo el último). La **pareja de casas** es el par no ordenado de sus dos
encuestadoras.

**Dispersión de un conjunto de comparaciones**: `R = sqrt(Σ d² / Σ v)` (1 = lo que daría el azar de las muestras).

**Comparación de dos tramos A → B en una elección**: se quedan las parejas de casas con al menos `min_rows` (3)
comparaciones en A **y** en B. Cada pareja de casas `g` tiene su `r²_a,g = Σ d²_a,g / Σ v_a,g` y su `r²_b,g`, y un
peso, el mismo en los dos tramos, `w_g = 1 / (1/Σ v_a,g + 1/Σ v_b,g)`; `R_a = sqrt(Σ w_g r²_a,g / Σ w_g)` y `R_b`
igual; `change = R_b / R_a − 1`; `falls = R_b < R_a`; `pairs` = parejas de casas que entran. Sin ninguna pareja, la
elección no entra en esa comparación. `p` de la elección: permutación de las etiquetas A/B de las comparaciones
**dentro de cada pareja de casas**, `permutations` (2.000) veces con semilla fija `seed`, `p = P(cociente permutado ≤
observado)` (cociente = `R_b² / R_a²`, que da el mismo orden que `change`, recalculado en cada barajado con sus
pesos).

Revisión final (2026-11-06): la primera versión sumaba todas las filas de las parejas (`R = sqrt(Σ d² / Σ v)` por
tramo), con lo que cada pareja pesaba por sus filas, distintas en cada tramo; ese cambio de mezcla sesgaba `falls`
(probabilidad nula de bajar 0,59 en `far_final`, hasta 0,99 en generales grandes). Con el peso fijo por pareja la
probabilidad nula media es 0,50 en las tres comparaciones.

Las tres comparaciones: `far_final` (pregunta 1), `far_mid` y `mid_final` (pregunta 2). Recuento de cada una:
`n` elecciones, `k` con `falls`, `sign_p = binom.sf(k − 1, n, 0.5)` y `clear` = elecciones con `p < 0,05`.

Nota: el spike contó `min_rows` sobre filas pareja de sondeos × serie y asignó el tramo por la media de días; la
especificación lo mantiene así. Con los tramos semiabiertos entran 102 comparaciones con media en 20,5 o 55,5 días que
el spike perdía; recontado el 2026-11-06, los tres recuentos no cambian (19 de 23, 9 de 17, 23 de 29).

**Parámetros** (`data/rebano/params.json`, sustituye al actual; fila en `data/README.md`): `scopes` (`"all"` = los del
manifiesto), `since` (nulo), `live_scope` `"es"`, `max_days` 365, `pair_days` 7, `periods` `{"final": [0, 21],
"mid": [21, 56], "far": [56, 365]}` (semiabiertos), `min_rows` 3, `n_series` 4, `exclude` `["CIS"]`, `permutations` 2000, `seed` 0.

**Partes** (envoltorio de estudio; nombres de campo en inglés; cocientes a 2 decimales y `change` a 3 para la página,
sin redondear en las cuentas; p con tres cifras significativas):

| Parte | Contenido |
|---|---|
| `meta` | `study`, `title`, `subtitle`, `scope` nulo, `election` (`{scope, date, days_left, as_of}` de la elección en curso o nulo), `params`, `inputs` (elecciones leídas y omitidas con su motivo; el run en curso), `sources`, `commit`, `dirty`, `versions`, `seconds`. |
| `summary` | `n_elections` (pasadas con al menos una comparación), `n_scopes`, `since_year` (año de la primera elección que entra), `comparisons` (`{far_final, far_mid, mid_final}`, cada uno `{n, k, sign_p, clear, median_change, since_year}`, con el año de la primera elección de esa comparación), `kinds` (los mismos recuentos de `far_final` partidos en `generales` y `autonómicas`). |
| `elections` | Una fila por elección pasada y comparación con dato: `scope`, `event_date`, `label` (el de la página, D4), `kind`, `comparison`, `pairs`, `rows_a`, `rows_b`, `r_a`, `r_b`, `change`, `falls`, `p`. CSV `elections`. |
| `current` | La elección en curso (nulo sin ella): `scope`, `event_date`, `label`, `days_left`, `as_of`, `windows` (fechas de inicio de `mid` y `final`), y `far_mid` y `mid_final` con la forma de una fila de `elections` (nulos mientras el tramo B no tenga ninguna pareja). |

**Subtítulo** (compuesto al publicar, como hoy, con cláusulas condicionadas a las cifras): «En {k} de las {n}
elecciones con sondeos de las mismas encuestadoras lejos y al final, desde {since_year de far_final}, las mismas
encuestadoras se parecen más entre sí en las tres últimas semanas que lejos de la votación{: por azar, 1 entre X}.
Hasta tres semanas antes, no: {k2} de {n2}, como una moneda.» (revisión final: antes «elecciones con datos desde
{since_year}» y «por azar pasaría una vez de cada X»). La segunda frase sólo si `far_mid` no es significativa y `mid_final` sí; las demás combinaciones las
redacta el constructor con el mismo tono (D5, textos condicionados).

**Paquete**: se reescribe `mtpy/lib/studies/herd/` (`data.py` con la lectura multiámbito, `stats.py` con
comparaciones, dispersión, comparación de tramos, permutación y signo, `study.py` con `compute_herd`). Lo que hoy
sólo sirve a la versión 1 (residuos frente a la curva, consenso local, rebaño por casa, semanas, precisión) se borra
con sus tests. Para leer y etiquetar elecciones de varios ámbitos, reutilizar lo del estudio de Andorra
(`mtpy/lib/studies/ban/data.py`, `mtpy/lib/studies/labels.py`).

**Tiempo**: el cálculo corre cada día dentro del `update` de ECS. Objetivo: menos de 2 minutos con el bundle real
(66 lecturas de `cycle`; la permutación vectorizada con numpy, no en bucles de Python por permutación).

## D4. La página `/analisis/rebano`

Plantilla `web/templates/analisis_rebano.html` reescrita; `pages/herd.py` con las funciones de vista nuevas (las de
la versión 1 se borran); `analisis.herd_context` con el contexto nuevo. Reglas de redacción de siempre: «sólo» con
tilde, «encuestadora» y «casa», nunca «empresa» (`assert_public_wording`); nada de «se copian», «copiar», «mirar al
vecino»; ninguna explicación de intenciones.

**Etiquetas de elección**: «Generales 2016», «Generales 2019 abr.» (mes cuando el año se repite en el ámbito),
«Cataluña 2024», «Madrid 2021»: nombre del ámbito como en el resto del sitio, más el año.

**Cabecera**: píldoras («Análisis»; «Página viva · se actualiza cada día hasta la veda del {fecha}» con elección en
curso, «elección celebrada, pendiente del escrutinio» tras la jornada, «Evidencia histórica» sin ella; «{n} elecciones
desde {since_year}»), título, subtítulo (`meta.subtitle`). Tres KPIs:

1. «Elecciones en que las encuestadoras se acercan al final» — `{k} de {n}` (`far_final`); apoyo «por azar, 1 vez de
   cada {X}».
2. «En las tres últimas semanas» — `{k} de {n}` (`mid_final`); apoyo «hasta tres semanas antes, {k} de {n}: como una
   moneda» (o la cifra con su probabilidad si fuera significativa).
3. «Generales {año}, en directo» — el `change` de `mid_final` de `current` en porcentaje con «a {d} días»; antes de
   que empiece el tramo final, la fecha en que empieza; sin elección en curso, el KPI se sustituye por «Generales y
   autonómicas» — `{n_generales} y {n_autonómicas}` elecciones.

**Tarjeta 1 · «¿Se comportan las encuestadoras como un rebaño?»** Veredicto: «SÍ, EN CONJUNTO» si `far_final.k > n/2`
y `sign_p < 0,05`; «NO» si `k ≤ n/2`; «NO SE PUEDE DECIR» en otro caso. Entradilla de dos o tres frases: qué se
compara (las mismas dos casas, lejos y al final, en la misma elección). **Gráfico en HTML y CSS** (sin ECharts: sale
igual en la imagen de la tarjeta y sin JavaScript): una barra horizontal por elección de `far_final`, ordenadas de la
que más baja a la que más sube, con la etiqueta de la elección a la izquierda y el porcentaje a la derecha; hacia la
izquierda (color de «se acercan») las que bajan, hacia la derecha las que suben; eje en 0 = sin cambio; las barras se
recortan en ±100 % con el valor real impreso. Debajo, la caja de conclusión: «En {k} de {n} elecciones…; en cada una
por separado podría ser azar (sólo en {clear} la bajada es clara por sí sola); que pase {k} de {n} veces por azar,
1 de cada {X}.»

**Tarjeta 2 · «¿Es mayor según se acerca la elección?»** Veredicto: «SÍ, EN LAS TRES ÚLTIMAS SEMANAS» si `mid_final`
es significativa y `far_mid` no; «SÍ, DESDE ANTES» si ambas lo son; «NO» si `mid_final` no lo es. Dos bloques lado a
lado (uno debajo del otro en el móvil), cada uno con un cuadrito por elección de la comparación (lleno = bajan, vacío
= no), el recuento grande y su probabilidad: «Hasta tres semanas antes · {k} de {n} · como una moneda» y «En las tres
últimas semanas · {k} de {n} · 1 entre {X}». Caja de conclusión con la frase de las dos cifras.

**Tarjeta 3 · «Generales {año}, en directo»** (sólo con elección en curso o celebrada sin escrutinio): las mismas dos
comparaciones de `current` con sus cifras (`pairs`, `r_a`, `r_b`, `change`) y una frase: «Por ahora, las mismas
parejas de casas difieren un {x} % menos/más que entre 3 y 8 semanas antes»; antes del tramo final, «el tramo de las
tres últimas semanas empieza el {fecha}». Nota: «Tras el escrutinio, {año} entra en la cuenta como la elección
número {n + 1}.»

**«Cómo lo medimos»**: texto llano con la idea de D3 (dos sondeos de casas distintas que acaban su campo con menos de
una semana de diferencia; su diferencia frente a la que darían sus muestras; las mismas parejas en los dos tramos, para
que el método de cada casa no cuente; por qué se cuenta por elección y no mezclando: el aviso de composición), y los
límites: las generales pesan más porque tienen más sondeos tempranos; muchas autonómicas sólo tienen una pareja de
casas comparable; sin el CIS ni los agregadores ni los paneles abiertos; «medimos cuánto se acercan, no por qué».

**«Datos y método»**: fuentes (Histórico, cada elección, el promedio), parámetros con sus valores vivos, la
permutación y la prueba del signo en una frase cada una, el CSV `elections`, fecha de actualización.

**Fuera**: el gráfico de semanas, la tabla de ciclos, el CIS, la tabla de casas de 2026, la matriz casa × ciclo,
«Después del 29-N», el script `web/dist/js/pages/analisis-rebano.js` y `charts/weeks.js` si nadie más los usa
(`matrix.js` lo usa el sesgo del CIS: queda, sin la opción `center` si sólo era para este estudio).

## D5. Otras páginas y documentación

- `/metodo/rating` (l. 612-615): el enlace dice el título nuevo y una frase que encaje («Si las encuestadoras se
  acercan en la recta final, elección a elección, está en …»).
- `/analisis` (tarjeta del estudio) e Inicio, si muestran título, subtítulo o veredictos del estudio: los nuevos.
- `docs/web/contrato.md` y `deploy/README.md`: las partes nuevas del estudio.
- Memoria del proyecto: `project-analisis-rebano.md` (versión 2, cifras reales al publicar).

## D6. Pruebas y aceptación

- Bundle sintético nuevo en `tests/fakes.py` (sustituye a `write_herd_bundle`): dos ámbitos, tres elecciones pasadas
  con resultado (una sin parejas en algún tramo), CIS y un agregador que deben quedar fuera, y un run en curso de `es`;
  cifras calculadas a mano en el test (una dispersión, un `change`, un recuento con su `sign_p`, una `p` de
  permutación con semilla en un caso pequeño, el reparto por tramo de una comparación que cruza 21 días).
- Tests de página con las partes de fixture: veredictos en sus tres ramas, KPIs, barras ordenadas y recortadas,
  cuadritos, tarjeta 3 en sus estados (antes del tramo final, durante, celebrada, sin elección), redacción
  (`assert_public_wording`, y que no aparezcan «copi», «vecino», «rebaño por casa»).
- **Aceptación con datos reales** (la hace el constructor contra el bundle de Luis, sin publicar): `compute_herd`
  reproduce el spike: `far_final` 19 de 23, `far_mid` 9 de 17, `mid_final` 23 de 29 (o la diferencia explicada si
  el bundle ha cambiado desde el 2026-11-06), y el tiempo de cálculo.
- Suite completa en verde.

## Lo que queda al constructor

- Nombres de funciones, organización interna de `stats.py`, forma exacta de los esquemas más allá de las claves.
- Las redacciones de las ramas condicionadas no escritas aquí, con el tono de las escritas.
- El estilo de las barras y cuadritos dentro del sistema de diseño del sitio (colores de «se acercan» / «se
  separan» de la paleta existente).
- Si el publicador borra los ficheros de partes no declaradas (D1).

## Revisión del constructor (2026-11-06)

Un revisor leyó la especificación como constructor. Lo que planteó y cómo queda:

- **Comparaciones con media de días en ,5** (caían fuera de los tramos del spike): decisión mía, tramos semiabiertos
  (D3); recontado, las cifras de aceptación no cambian.
- **Resueltas por el propio revisor, anotadas como decisiones mías**: 2026 no entra en los recuentos pasados hasta
  tener `results` (el spike ya no tenía parejas suyas); 2.000 permutaciones en vez de las 5.000 del spike (puede mover
  en uno el número de «claras»: se comprueba en la aceptación y se acepta la cifra que salga); entran también las
  elecciones con menos de 10 sondeos, como en el spike; los datos de 2026 vienen del run `latest` de `es` con la regla
  de `read_current` de hoy.
- **Cambios sin commit de la versión 1 en el árbol** (arreglos del 2026-11-06: umbral de semanas, «cabría esperar sólo
  por el azar», pregunta 3): pendiente de Luis.

## Cierre (2026-11-06)

Construido en la rama `analisis-rebano-v2` (`c6f6529..fff4310` por tareas, `ae87064` primer cierre y la ronda de la
revisión final `3bde00b..`), base `c3aa20e` de `dev`, con revisión por tarea y revisión final.

**Qué existe**

- Paquete `mtpy/lib/studies/herd/`: `stats.py` (comparaciones entre parejas de casas, tramos semiabiertos, dispersión
  con peso fijo por pareja de casas, permutación de 2.000 barajados dentro de cada pareja, prueba del signo),
  `data.py` (lectura de todas las elecciones publicadas) y `study.py` (`compute_herd`).
- Exportadores `mtpy/lib/publish/herd.py`: partes `meta`, `summary`, `elections` y `current` (+ CSV
  `elections`); las partes de la versión 1 (`cycles`, `weeks`, `houses`) ya no se escriben. `docs/web/contrato.md`
  describe las partes y claves de la versión 2.
- Página `mtpy/lib/pages/herd.py` + `web/templates/analisis_rebano.html` + `web/dist/js/pages/analisis-rebano.js`
  (barras por elección, cuadritos, tabla en directo), ajustes en `web/dist/css/site.css` (la píldora de página viva
  y las etiquetas de los KPI parten línea sólo en esta página); fixtures y pruebas nuevas.

**Cifras con el bundle real (2026-11-06, estadística con peso fijo por pareja, `compute_herd` en 31,8 s)**

| Comparación | Elecciones a favor | `sign_p` | Claras por sí solas | Desde |
|---|---|---|---|---|
| lejos → final (`far_final`) | 19 de 23 | 0,0013 (1 entre 769) | 3 | 1996 |
| lejos → medio (`far_mid`) | 9 de 17 | 0,5 (moneda) | 4 | 1989 |
| medio → final (`mid_final`) | 22 de 29 | 0,00407 (1 entre 246) | 8 | 1982 |

Bajada mediana de `far_final`: −15 % (`far_mid` −12 %, `mid_final` −13 %). Generales 8 de 10 (p 0,055, mediana
−7 %), autonómicas 11 de 13 (p 0,011, mediana −20 %). 34 elecciones, 9 ámbitos, desde 1982 (la píldora; el subtítulo
cuenta desde 1996, el primer año de `far_final`). Generales 2026 en directo: `far_mid` +6 % (37 parejas, 1,02 →
1,08, p 0,783), a 23 días; el tramo final empieza el 9-11. Probabilidad nula media de bajar (permutaciones dentro de
cada pareja): 0,501 / 0,502 / 0,499 en `far_final` / `far_mid` / `mid_final` (antes, sumando filas, 0,59 en
`far_final`); con ella la cola de Poisson-binomial casi coincide con la moneda (0,00134 frente a 0,0013; 0,00394
frente a 0,00407). Frente al primer cierre cambia `mid_final` (23 → 22 de 29) y varias barras (Generales 2023 sube
+8 %); los veredictos no cambian: pregunta 1 «SÍ, EN CONJUNTO», pregunta 2 «SÍ, EN LAS TRES ÚLTIMAS SEMANAS».
Capturas a 390 y 1280 px sin desborde horizontal (`v3_390.png`, `v3_1280.png` en el scratchpad de la sesión
a00d6fa2; informe en `final-fix-report.md` del espacio de trabajo SDD).

**Decisiones tomadas en la ejecución**

- `main_series` se retiró de `stats.py` y se recoloca en `data.py` con la definición de la versión 1; las
  distancias de empate de la permutación llevan tolerancia de 1e-9.
- La pareja extra de es/2023 (A3-B8, 5 días) se acepta tal cual: las cifras comprobadas no cambian.
- Subtítulo: segunda frase sólo si `far_mid` no es significativa y `mid_final` sí; con `k*2 <= n` la frase es «En
  sólo {k} de las {n} …; en las demás, no»; guarda con `n == 0`.
- Texto del KPI 2 («hasta tres semanas antes, {k} de {n}: como una moneda» bajo «En las tres últimas semanas»):
  es el de D4, aprobado por Luis, y se mantiene.
- «a 1 día» en singular; sin frase de directo si falta la fecha del tramo final; «bajada mediana del 15 %» sin
  doble negativo; «como una moneda» sin «por azar».
- Se restauró la clase `.qb` del CSS compartido (la usaban `/analisis/voto-oculto` y `/analisis/andorra`).
- En la prueba de palabras prohibidas, «copi» se comprueba como inicio de palabra (choca con «Demoscopio»).
- Revisión final: peso fijo por pareja de casas (D3); el lado de cada barra sale de `falls` y un cambio que
  redondea a 0 % lleva un decimal; `since_year` propio de cada comparación; una sola forma de la probabilidad
  («1 entre N»); las dos preguntas de Luis en todas partes; «Datos y método» corregido (3 comparaciones por pareja
  de casas y tramo, permutación de comparaciones, sin promedios ni enlace al método del promedio); los sondeos sin
  encuestadora quedan fuera de las comparaciones.

**Menores aplazados** (no bloquean): `n <= 0` sin guardar en `comparisons`; `p` puede ser 0,0; `main_series` sin
prueba de orden; importes privados `_is_number`/`_sig3`; orden de columnas de `elections` sin prueba; KPI 3 con
fallbacks sin probar; «tres últimas semanas» fijo en la página y no leído de `params`; «subida mediana del 0 %»;
guion de la tabla en directo sin prueba a nivel HTML; el pie muestra el id de ejecución crudo («20261008-120000»)
con la procedencia de los fixtures; en 390 px la tabla en directo se desplaza dentro de su envoltorio.

**Sin verificar**: los botones de descarga de las tarjetas no se han probado en un navegador; la página con
Chrome sin cabeza sólo se ha visto estática; el pie «Actualizado / Último sondeo» se vio con una procedencia
ficticia (la de los fixtures), no con la de un `publish` real.

**Pasos de Luis** (en este orden: `rebano` ya está publicado y el `update` diario con la imagen ANTIGUA republica
las partes de la versión 1)

1. Desplegar la imagen nueva.
2. Después, `python job.py publish '{"what":["study"],"study":"rebano"}'` (o dejar que lo haga el `update` diario).
3. Sólo con la imagen nueva ya en marcha, borrar de su S3 `studies/rebano/{cycles,weeks,houses}.json` y
   `studies/rebano/csv/{cycles,weeks,houses}.csv`.
4. Tras el 29-N, `publish event` de `es/2026-11-29` y una republicación.
