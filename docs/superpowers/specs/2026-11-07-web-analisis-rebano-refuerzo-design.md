# Web, Análisis del rebaño: refuerzo estadístico (prueba conjunta, intervalo y robustez)

Añadido a la versión 2 del estudio `rebano` (`2026-11-06-web-analisis-rebano-v2-design.md`, fusionada en dev ffb5c39)
y al embudo por semanas (`2026-11-07-web-analisis-rebano-embudo-design.md`, fusionado en dev 5d655f6). Pedido por Luis
el 2026-11-07 («¿Cómo podemos reforzar estadísticamente este cálculo?» → «Venga, adelante» a los puntos 1, 2 y 4 de la
propuesta; el 5, la calibración por simulación, queda para después).

## Intención

La página contesta con dos niveles de prueba: por elección, una permutación dentro de cada pareja de casas (casi nunca
concluyente: la mediana es de 3 o 4 parejas de casas por elección y muchas autonómicas tienen una), y entre elecciones,
una prueba del signo que cuenta cada elección como una moneda, igual la de 80 parejas que la de 1, y que ignora cuánto
baja la separación. Luis quiere que el recuento (19 de 23, 22 de 29) vaya acompañado de una prueba que use la magnitud y
el tamaño, de un intervalo para la bajada típica, y de una comprobación de que el resultado no depende de los 7 días
entre sondeos ni del mínimo de 3 comparaciones.

## Hechos de partida (partes publicadas del 2026-11-07 y spike del mismo día, `spike_pooled*.py` y `spike_blocks.py`
en el scratchpad de la sesión a00d6fa2)

- Lejos → tres últimas semanas: 23 elecciones, 309 parejas de casas, 2.501 filas lejos y 1.750 cerca; por elección
  entre 1 y 80 parejas (mediana 4); 19 bajan (signo 0,0013), 3 claras, bajada mediana −15 %.
- Lejos → 3-8 semanas: 17 elecciones, 185 parejas, 9 bajan (0,5), mediana −12 %. 3-8 semanas → tres últimas: 29
  elecciones, 312 parejas, 22 bajan (0,0041), 8 claras, mediana −13 %.
- **Prueba conjunta** (estratos elección × pareja de casas, barajando por parejas de sondeos): lejos → final **−27 %**,
  p < 0,0005; 3-8 → final **−23 %**, p < 0,0005; lejos → 3-8 **−6 %**, p 0,058. Barajando fila a fila (como hoy la
  prueba por elección) la p de lejos → 3-8 baja a 0,025: el barajado por filas es anticonservador, porque las cuatro
  filas de una pareja de sondeos salen de los mismos dos sondeos.
- **Intervalo bootstrap de la mediana** (2.000 remuestreos de elecciones): lejos → final [−35 %, −3 %]; 3-8 → final
  [−27 %, −2 %]; lejos → 3-8 [−28 %, +19 %].
- **Prueba de signos al azar sobre la magnitud** (cada elección igual): lejos → final p 0,09; 3-8 → final 0,006; lejos →
  3-8 0,27. Las cuatro elecciones que suben en lejos → final suben mucho y muchas de las que bajan, poco.
- **Rejilla**: la bajada conjunta al final va de −23 % (3 días) a −27 % (7 días) y −26 % (10 días), siempre p < 0,0005;
  en las tres últimas semanas de −21 % a −23 %. El recuento sí se tambalea: con 3 días entre sondeos quedan 18
  elecciones y bajan 11 (moneda); con 10 días, 22 de 27. La bajada temprana (lejos → 3-8) crece con los días entre
  sondeos (−3 % con 3 días, −7 % con 10) y sólo es clara con 10: compatible con la deriva del voto dentro de la pareja,
  no con un rebaño temprano. Con un mínimo de 6 comparaciones (dos parejas de sondeos por pareja de casas y tramo)
  quedan entre 3 y 11 elecciones por celda, y la bajada conjunta al final es mayor (−32 a −44 %). Un mínimo de 2 da lo
  mismo que 3 (una pareja de sondeos ya tiene 3 o 4 filas) y 8 casi lo mismo que 6.
- Coste: las comparaciones con otro `pair_days` cuestan 0,9 s por valor; la rejilla completa con permutaciones por
  bloques, unos 6 s. La lectura de las partes es el grueso de los 30 s actuales.

## Objetivos

- Una **prueba conjunta** de la bajada con todas las parejas de todas las elecciones a la vez, con una p de permutación
  honesta (por parejas de sondeos).
- Un **intervalo** para la bajada mediana entre elecciones.
- Una **tabla de robustez**: los recuentos y la prueba conjunta con 3, 5, 7 y 10 días entre sondeos y con 3 y 6
  comparaciones mínimas por pareja de casas.
- Todo en la página, a la vista lo que importa (una frase por pregunta) y lo de expertos en una nota desplegable; todo
  descargable.

## No objetivos y anti-objetivos

- No cambia la medida (parejas de casas distintas a ≤ `pair_days`, `R = √(Σd²/Σv)`, peso fijo por pareja), ni los
  tramos, ni la prueba del signo, ni la regla de los veredictos, ni los KPIs, ni el subtítulo. Las cifras actuales de
  la página siguen saliendo igual salvo las «claras por sí solas» (D1), que bajan al barajar por parejas de sondeos.
- Nada por casa (sin referencia no se atribuye el acercamiento a una casa), nada de dirección («quién sigue a quién»).
- La calibración por simulación (punto 5) no entra. El método de Fisher sobre las p por elección no entra: con una
  pareja y tres filas la p de permutación es muy gruesa y el combinado engaña.
- Nada de «se copian», «permite el azar», ni explicaciones de intenciones. Reglas de redacción de siempre («sólo» con
  tilde, casa/encuestadora, nunca empresa).

## Enfoques considerados

- **Permutación estratificada conjunta** (elegido): el mismo estadístico de `compare` con los estratos «elección ×
  pareja de casas» de todas las elecciones, y su nulo barajando dentro de cada estrato a la vez. Reutiliza la máquina
  actual y pesa cada pareja por su información.
- **Unidad del barajado = pareja de sondeos** (elegido tras la revisión del constructor), en la prueba conjunta y
  también en la de cada elección: las filas de una pareja de sondeos (una por partido) van juntas. Barajar fila a fila
  (lo de hoy) trata como independientes cuatro filas que salen de los mismos dos sondeos; con 4.000 filas juntas daría
  p = 0 casi con cualquier bajada. Queda sin corregir la dependencia de segundo orden de un sondeo que entra en varias
  parejas de sondeos de la misma pareja de casas; se dice en el método.
- Combinar las p por elección (Fisher, Stouffer): descartado, p discretas y gruesas en las elecciones con una pareja.
- Prueba del signo ponderada: descartado, no es estándar y mezcla recuento y tamaño sin un nulo limpio.
- Intervalo por bootstrap de elecciones (elegido) frente a bootstrap de parejas: la elección es la unidad de los
  recuentos, y así el intervalo habla de lo mismo que la mediana.
- Robustez como rejilla de parámetros (elegido) frente a corregir la deriva con una regresión de `d²` sobre los días de
  separación: la rejilla no añade supuestos y se explica en una frase. Eje de mínimos 3 y 6 (una y dos parejas de
  sondeos), no 2 y 3 (indistinguibles).
- **Propuesta mía, opcional (Luis decide al aprobar):** la prueba de signos al azar sobre la magnitud (`sign_flip_p`:
  media del logaritmo del cambio de cada elección, signos al azar 2.000 veces, cada elección pesa igual). Contesta a la
  objeción de que la prueba conjunta la dominan las generales grandes, y hoy diría que lejos → final «podría ser azar»
  (0,09) mientras 3-8 → final es clara (0,006). Si entra, va sólo en la parte y en la nota de expertos, no en las cajas.

## D1. Parámetros y estadística (`data/rebano/params.json`, `mtpy/lib/studies/herd/stats.py`)

- `params.json` gana `"resamples": 2000` (remuestreos del bootstrap) y `"robustness": {"pair_days": [3, 5, 7, 10],
  "min_rows": [3, 6]}` (la rejilla; la celda por defecto es `(pair_days, min_rows)` de los parámetros, hoy (7, 3)).
  Fila de `data/README.md` actualizada.
- `_blocks(group) -> (d2, v, is_a)`: los bloques de una pareja de casas, uno por pareja de sondeos (`poll_i`, `poll_j`):
  `Σ d2`, `Σ v` y si su periodo es `a` (todas las filas de una pareja de sondeos comparten periodo, el de la media de
  sus días).
- `_permuted_terms(shared, a, permutations, generator) -> (perm_a, perm_b, obs_a, obs_b)`: por pareja de casas, barajar
  las etiquetas de sus **bloques** manteniendo cuántos bloques tiene cada periodo, y acumular `Σ w·r²` de cada periodo
  por permutación (`_weighted_sums` sobre los bloques) más los términos observados. `permutation_p` pasa a usarlo: sus
  valores cambian (hoy baraja filas); «claras por sí solas» con datos reales pasan de 3 / 4 / 8 a 3 / 2 / 4.
- `pooled(rows_by_election: list[pd.DataFrame], a, b, min_rows) -> Optional[dict]`: la fórmula de `compare` con los
  estratos «elección × pareja de casas» de todas las elecciones (cada estrato con sus filas de `_shared_pairs`):
  `R_a = √(Σ w_g r²_a,g / Σ w_g)` y `R_b` igual. Devuelve `{elections (con algún estrato), pairs (estratos), rows_a,
  rows_b, r_a, r_b, change (= r_b/r_a − 1), falls}` sin redondear; `None` sin estratos o con `r_a` 0.
- `pooled_p(rows_by_election, a, b, min_rows, permutations, seed) -> Optional[float]`: un solo generador con `seed`;
  para cada elección, `_permuted_terms` de sus parejas compartidas, sumando `perm_a`, `perm_b`, `obs_a`, `obs_b` entre
  elecciones; `p` = proporción de permutaciones con `perm_b/perm_a ≤ observado·(1 + PERMUTATION_TOLERANCE)`, como en
  `permutation_p`. `None` cuando `pooled` es `None`.
- `bootstrap_median(changes: list[float], resamples: int, seed: int) -> Optional[dict]`: remuestrea las elecciones con
  reemplazo `resamples` veces (generador con `seed`), mediana de cada remuestreo, `{lo, hi}` = percentiles 2,5 y 97,5
  sin redondear; `None` con menos de 3 cambios.
- Opcional (si Luis lo aprueba): `sign_flip_p(changes, permutations, seed) -> Optional[float]`: estadístico = media de
  `log(1 + change)`; nulo = signos al azar; `p` = proporción de medias permutadas ≤ la observada; `None` con menos de 3.

## D2. El cálculo (`mtpy/lib/studies/herd/study.py`)

- `compute_herd` conserva las comparaciones (`rows`) de cada elección pasada ya calculadas y, por cada clave de
  `COMPARISONS`, añade a `summary['comparisons'][clave]`:
  - `pooled`: `{elections, pairs, rows_a, rows_b, r_a, r_b, change, p}` con `r_a`/`r_b` a 2 decimales, `change` a 3 y
    `p` = `_sig3(pooled_p(...))` (`permutations` y `seed` de los parámetros); `None` sin estratos.
  - `ci`: `{lo, hi}` de `bootstrap_median` sobre los cambios sin redondear de las elecciones de esa comparación, a 3
    decimales; `None` con menos de 3 elecciones.
  - Opcional: `flip_p` (`_sig3`).
- `summary['robustness'] = {'pair_days': [...], 'min_rows': [...], 'rows': [...]}`: por cada celda de la rejilla (en el
  orden `pair_days` × `min_rows`), se recalculan las comparaciones de cada elección con ese `pair_days`
  (`comparisons(polls, names, params | {'pair_days': d})`) y, por cada clave de `COMPARISONS`, una fila
  `{pair_days, min_rows, default (bool), comparison, n, k, median_change, change, p}`: `n`, `k` y `median_change` (3
  decimales) de `count` sobre los `compare` por elección (sin permutaciones por elección: `p` por elección no se
  calcula en la rejilla), `change` (3 decimales) y `p` (`_sig3`) de `pooled` y `pooled_p`. La celda por defecto da las
  mismas cifras que el resumen (prueba de ello en D5). Sin elecciones, `rows` vacío.
- `elections`, `current` y `weeks` no cambian de forma; en `elections` cambian las `p` (D1). Coste medido: la rejilla
  unos 6 s; el cálculo completo queda por debajo de 45 s con datos reales (hoy 30 s).

## D3. Parte publicada (`bundle.py`, `publish/herd.py`, `publish/studies.py`, `pages/analisis.py`, `docs/web/contrato.md`)

- `SCHEMAS['rebano-summary']` gana `'robustness': dict`. Las claves nuevas dentro de `comparisons` no cambian el
  esquema. `meta.params` lleva los parámetros nuevos sin más.
- `STUDY_CSV['rebano'] = ('elections', 'weeks', 'robustness')`; `STUDY_CSV_OF['rebano']['robustness'] = ('summary',)`;
  `export_herd_summary` devuelve `(summary, {'robustness': frame})` con `ROBUSTNESS_COLUMNS = ['pair_days', 'min_rows',
  'default', 'comparison', 'n', 'k', 'median_change', 'change', 'p']` (`n`, `k` enteros).
- `CSV_LABELS['rebano']['robustness'] = 'Robustez: recuentos y cambio conjunto con otros días entre sondeos y otro mínimo
  de comparaciones'`.
- `contrato.md`: las claves nuevas de `summary` y el CSV en las tablas de `rebano`; la permutación por parejas de
  sondeos donde describa la `p` de `elections`.
- Compatibilidad: la parte nueva es un superconjunto de la actual, así que la imagen desplegada hoy la pinta sin
  cambios; y la página nueva con una parte antigua no escribe las frases nuevas (todo condicionado a que existan).

## D4. La página (`mtpy/lib/pages/herd.py`, `web/templates/analisis_rebano.html`)

- `herd.ci_text(comparison) -> Optional[str]`: el intervalo de la bajada mediana en palabras, con los extremos
  redondeados a enteros de porcentaje: «entre el 3 y el 35 %» cuando los dos extremos son bajadas; «entre una subida
  del 19 % y una bajada del 28 %» cuando el intervalo cruza el cero; `None` sin `ci`.
- `herd.pooled_text(comparison, later, earlier) -> Optional[str]`: «Con todas las parejas de todas las elecciones
  juntas, la separación {al final} es un 27 % menor que {lejos de la votación}: por azar, menos de 1 entre 1.000.»
  («mayor» si el cambio es positivo; «igual» si redondea a 0 %; el azar con `chance_text(pooled.p)`); `None` sin
  `pooled`.
- Tarjeta 1, caja `sum`: la frase actual «En 19 de 23 elecciones las mismas casas se acercan al final, con una bajada
  mediana del 15 %» gana el intervalo entre paréntesis: «… del 15 % (entre el 3 y el 35 %).» Tras la frase del azar
  del recuento, una frase nueva: el `pooled_text` de `far_final` con «al final» y «lejos de la votación».
- Tarjeta 2, caja `sum`: «En las tres últimas semanas, 22 de 29 elecciones se acercan (1 entre 246)» gana «, con una
  bajada mediana del 13 % (entre el 2 y el 27 %)» (`mid_final`); tras «Hasta tres semanas antes, 9 de 17: como una
  moneda.», una frase con las dos pruebas conjuntas, que se reconcilia sola por la magnitud: «Con todas las parejas
  juntas: un 23 % menos en las tres últimas semanas que entre 3 y 8 semanas antes (por azar, menos de 1 entre 1.000),
  y un 6 % menos hasta tres semanas antes que lejos (1 entre 17).» La frase «Sólo en 4 de las 29 la bajada es clara por
  sí sola.» sigue cerrando (con la cifra nueva de D1).
- **Si la prueba conjunta contradice al recuento** (decisión mía, tras la revisión del constructor: opción b): se
  publican las dos cifras tal cual, y la reconciliación está en la propia frase conjunta, que da las dos magnitudes
  (6 % frente a 23 %). Los veredictos siguen leyendo sólo la prueba del signo (`_significant` no cambia); hoy no hay
  contradicción (lejos → 3-8 da p 0,058).
- KPIs, veredictos, barras, cuadritos, embudo y «en directo»: sin cambios.
- Tarjeta «Cómo lo medimos»: la frase «así una elección con muchos sondeos no pesa más que otra con pocos» pasa a «así,
  en los recuentos, una elección con muchos sondeos no pesa más que otra con pocos; en la prueba conjunta sí pesa más,
  y por eso damos las dos». Tras el párrafo, una nota desplegable `<details class="how">` con `summary` «Con otros días
  entre sondeos y otro mínimo de comparaciones» y dentro: una frase (`herd.robustness_text(summary)`: «Con 3, 5, 7 o 10
  días entre los dos sondeos y con 3 o 6 comparaciones por pareja, la bajada conjunta al final queda entre el 23 y el
  44 %, y en las tres últimas semanas entre el 21 y el 32 %; el recuento de elecciones que se acercan al final va de
  11 de 18 con 3 días a 22 de 27 con 10.», con los mínimos y máximos de `change` y los extremos de `k de n` de las
  celdas; `None` sin filas) y una tabla (`herd.robustness_rows(summary)`) con una fila por celda, la celda por defecto
  en negrita y primera: columnas «Días», «Mínimo», y para «Lejos → tres últimas» y «3-8 semanas → tres últimas» las
  tres de «Se acercan» (`k de n`), «Cambio conjunto» (`percent`) y «Por azar» (`chance_text`). `far_mid` va sólo en el
  CSV. Si entra `flip_p`, una línea más en la nota: «Contando cada elección igual y con su magnitud (signos al azar):
  lejos → final {chance}; 3-8 → final {chance}; lejos → 3-8 {chance}.» La nota cerrada no sale en la imagen de la
  tarjeta; abierta, sí (como las demás `.how`).
- «Datos y método», texto `tech`: la frase de la permutación pasa a «La <b>permutación</b> baraja {permutations} veces
  el tramo de cada pareja de sondeos, con sus partidos juntos, dentro de cada pareja de casas…» y tres frases nuevas:
  «La <b>prueba conjunta</b> baraja a la vez las parejas de sondeos de todas las parejas de casas de todas las
  elecciones y mide cuánto baja la separación juntándolas: cada pareja de casas pesa según sus comparaciones, así que
  las elecciones con muchos sondeos pesan más que en la prueba del signo. Un sondeo que entra en varias parejas cuenta
  en todas.» «El <b>intervalo</b> de la bajada mediana sale de remuestrear las elecciones {resamples} veces.» «La
  <b>robustez</b> repite los recuentos y la prueba conjunta con {pair_days de la rejilla} días entre sondeos y con un
  mínimo de {min_rows} comparaciones por pareja.»
- `assert_public_wording` sobre los textos nuevos.

## D5. Pruebas y aceptación

- `tests/test_herd_stats.py`: `_blocks` agrupa las filas de una pareja de sondeos; `permutation_p` con una pareja de
  casas y un bloque por periodo da `p` ≥ 0,5 (sólo dos arreglos), y con cifras a mano sobre filas construidas da el
  valor esperado (los tests actuales se actualizan a los valores por bloques); `pooled` con una sola elección da las
  cifras de `compare`; con dos elecciones, el cálculo a mano de `R_a`, `R_b` y `change` sobre filas construidas (cuatro
  parejas, pesos distintos); `pooled_p` ≈ 0,5 (entre 0,3 y 0,7) con dos periodos intercambiables y pequeña (< 0,05) con
  una bajada construida; `bootstrap_median` determinista con la semilla, `lo ≤ mediana ≤ hi`, `None` con dos cambios;
  opcional: `sign_flip_p` ≈ 0,5 con cambios simétricos y pequeña con todos negativos.
- `tests/test_herd_study.py` sobre `write_herd_bundle`: `pooled`, `ci` y `robustness` presentes con cifras a mano (ver
  el plan); la fila por defecto de `robustness` coincide con `summary['comparisons']` en `n`, `k`, `median_change` y
  `pooled.change`; `elections`, `current` y `weeks` no cambian de forma.
- `tests/test_publish_herd.py`: la parte valida con `robustness`; el CSV `robustness` lleva las columnas y los enteros;
  fixture `tests/fixtures/bundle/rebano-summary.json` ampliado con cifras como las reales.
- `tests/test_pages.py`: las frases con intervalo y la prueba conjunta en las dos cajas; la nota desplegable con la
  frase y la tabla y la fila por defecto en negrita; sin `pooled`/`ci`/`robustness` en la parte, las cajas y la tarjeta
  quedan como hoy; `assert_public_wording`.
- Aceptación con datos reales (sólo lectura): `pooled` de las tres comparaciones con su p (−27 %, −23 %, −6 %; p
  < 0,0005, < 0,0005, ≈ 0,06), los intervalos, la rejilla completa, las «claras» 3 / 2 / 4 y el tiempo total (< 45 s).

## Pasos tras la fusión

Desplegar la imagen y `publish '{"what":["study"],"study":"rebano"}'` justo después (lo lanza quien integre, con el OK
de Luis). Con la imagen antigua la parte nueva se pinta igual que hoy; con la imagen nueva y la parte antigua no salen
las frases nuevas ni la nota.

## Revisión del constructor (2026-11-07)

Tres preguntas: (1) qué se baraja, filas o parejas de sondeos → parejas de sondeos, en las dos pruebas (medido: la p
conjunta de lejos → 3-8 pasa de 0,025 a 0,058; las «claras» de 3 / 4 / 8 a 3 / 2 / 4; las dos pruebas principales siguen
en p < 0,0005); (2) qué dice la página si la prueba conjunta contradice al recuento → se publican las dos con sus
magnitudes y la frase de los pesos en «Cómo lo medimos» lo explica, los veredictos no cambian de regla; (3) si entra
`sign_flip_p` → Luis aprobó la spec el 2026-11-07 («Ok, aceptado») sin pronunciarse: queda FUERA, como dice la
spec para ese caso; se puede añadir después (D1 y D4 opcionales quedan escritos).

## Lo que queda al constructor

Nombres internos, el formato exacto de la tabla de robustez dentro del sistema del sitio, el orden de las filas (por
defecto primero, luego la rejilla), la redacción de un intervalo o un rango con subida en los dos extremos, y si las
comparaciones de la rejilla se calculan una vez con 10 días y se filtran por `pair_days`, siempre que den lo mismo.

## Cierre (2026-11-08)

Construido con subagentes en la rama `rebano-refuerzo` (desde dev d894e20): 29016d6 (bloques, `pooled`, `pooled_p`,
`bootstrap_median`, parámetros), 9c039ad (resumen del estudio con `pooled`, `ci` y `robustness`), 0af5982 (esquema,
CSV `robustness`, contrato, fixture), 165f1e6 (página) y 14693c7 (ola de correcciones tras la revisión final). Batería
2333 passed / 81 skipped. Con datos reales (2026-11-07): conjunta lejos → final −27 % (p < 0,0005), 3-8 → final −23 %
(p < 0,0005), lejos → 3-8 −6 % (p 0,058); intervalos [−35 %, −3 %], [−27 %, −2 %], [−28 %, +19 %]; claras 3 / 2 / 4;
cálculo completo 41 s (antes 30).

Decisiones tomadas en la ejecución: `sign_flip_p` fuera; extremos del recuento en la frase de robustez por la fracción
k/n (empate, menos días); una celda sin elecciones es una fila con `n` 0 y nulos; el `median_change` del resumen sigue
a 2 decimales y la rejilla a 3 (el test compara a 2); el mapa de CSV es parte → nombres (`STUDY_CSV_OF['rebano']
['summary'] = ('robustness',)`); el intervalo se rotula «(95 %)» en el método y el contrato; la frase de robustez calla
si alguna celda que resume sube; los textos del método se condicionan a los datos (la página queda como antes sin
`pooled`/`ci`/`robustness`); la nota de expertos va al final de «Cómo lo medimos»; una sola fila en la tabla con
`.is-default`; «(dos parejas de sondeos)» sólo cuando el mínimo mayor es 6 y `larger` usa `>`; `herd.names` gana
`conjunction`. Aparcado (menores, sin segunda ola): sumas por pareja duplicadas entre `compare` y `pooled`; `_blocks`
no asegura un periodo por bloque (se cumple por construcción); `pooled_p` desempaqueta por índice y repite
`_shared_pairs`; la rejilla recalcula las comparaciones del `pair_days` por defecto y `pooled` dos veces por celda
(margen de 4 s sobre los 45 del objetivo: medir de nuevo cuando entre 2026); `ci` no nulo sin test en el estudio;
«bajada del 0 %» en `ci_text` cuando un extremo redondea a 0; sin fila `default` si los parámetros sacan la celda de
la rejilla; docstring de `robustness_text` sin el `None` por subida; la cláusula de la mediana de la tarjeta 2 sale
aunque no haya `ci`. Recomendación del revisor final para más adelante: p de Phipson–Smyth `(k+1)/(B+1)` en las dos
permutaciones (nunca 0,0; cambia las «claras» cercanas a 0,05) y una cláusula en la nota de expertos sobre el peso de
las generales grandes en la prueba conjunta.
