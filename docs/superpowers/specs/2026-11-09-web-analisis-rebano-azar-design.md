# Web, Análisis del rebaño: tercera tarjeta «¿Podría ser azar?» (prueba de hipótesis nula)

Añadido a la versión 2 del estudio `rebano` (`2026-11-06-web-analisis-rebano-v2-design.md`), al embudo
(`2026-11-07-web-analisis-rebano-embudo-design.md`) y al refuerzo estadístico
(`2026-11-07-web-analisis-rebano-refuerzo-design.md`, fusionado en dev d5317a3). Pedido por Luis el 2026-11-09:
«¿podemos abordar una tercera tarjeta como una prueba de hipótesis nula? … negando la hipótesis de que las desviaciones
de las últimas 3 semanas son iguales que las del resto del ciclo, con un p-value resultante». Aprobada la propuesta
el mismo día («Ok, con tu propuesta»).

## Intención

Una tarjeta que presente la prueba como lo que es: una hipótesis nula enunciada en una frase, la distribución de lo
que daría el azar (las barajas) con el valor observado marcado fuera, y una p. No añade evidencia nueva: es la
máquina de la prueba conjunta del refuerzo, con dos tramos en vez de tres y 20.000 barajas, contada con su
distribución nula a la vista. Lo que añade es claridad.

## Hechos de partida (spike del 2026-11-09, `spike_h0.py`, `spike_h0_ci.py`, `synth_h0.py` en el scratchpad de la
sesión a00d6fa2)

- Tramos «tres últimas semanas» (0-20 días) frente a «resto del ciclo» (21-365 días), mismas parejas de casas con al
  menos 3 comparaciones en cada tramo: 34 elecciones (25 bajan), 475 parejas de casas, 4.530 filas en el resto y 2.654
  al final; R de 1,58 a 1,17, un −26,3 %.
- Permutación por parejas de sondeos (bloques) con 20.000 barajas: ninguna baja del −11,9 % ni sube del +14,8 %; la
  distribución nula del cambio tiene media 0,0 % y desviación típica 3,1 %; lo observado está a unas 8 desviaciones.
  p a una cola 0 de 20.000, a dos colas 0 de 20.000; con la corrección (k + 1)/(B + 1), 1/20.001 = 0,00005.
- Intervalo del 95 % del cambio conjunto por bootstrap de elecciones (2.000 remuestreos): [−38 %, −4 %]. Calculado con
  los términos por elección precomputados cuesta menos de un segundo; recomputando `pooled` cada vez, 63 s (no).
- Coste: las comparaciones con los dos tramos 0,9 s; las 20.000 barajas 0,9 s. El cálculo del estudio pasa de 41 a
  unos 44 s.
- Histograma con cajas de 1 punto porcentual: 28 cajas de −12 % a +16 %.
- Paquete sintético de los tests (`write_herd_bundle`): Generales 2019 1 pareja, 8/4 filas, R 1,2435 → 0,3521
  (−0,7168); Generales 2023 1 pareja, 4/6 filas, 0,3538 → 1,0563 (+1,9858); Madrid 2021 sin pareja compartida.
  Conjunta: 2 elecciones, 2 parejas, filas 12/10, R 0,9356 → 0,7696 (−0,1775); bloques 6 y 5 → 150 arreglos exactos:
  p a una cola 22/150 = 0,1467, a dos colas 65/150 = 0,4333; nulo de −0,542 a +1,015.

## Objetivos

- Tarjeta 3 «¿Podría ser azar?» con veredicto, la hipótesis en una frase, el histograma de la distribución nula con el
  observado marcado, la p a dos colas, el tamaño del efecto con intervalo, el recuento de elecciones y una nota para
  expertos.
- Parte publicada `chance` con todo lo que la tarjeta pinta, y su CSV (las cajas del histograma).
- Las tarjetas 1 y 2, los KPIs y el subtítulo no cambian.

## No objetivos y anti-objetivos

- No cambia la medida ni las pruebas de las tarjetas 1 y 2 (tres tramos, 2.000 barajas por elección).
- Nada de «se copian», «vecino», «permite el azar», ni intenciones. Reglas de redacción («sólo» con tilde,
  casa/encuestadora, nunca empresa). Se puede decir «hipótesis nula» y «p» en esta tarjeta: Luis lo pide así.
- No se calcula p por casa ni dirección alguna.

## Enfoques considerados

- **Permutación estratificada conjunta sobre dos tramos, con histograma de la distribución nula** (elegido): es la
  prueba que ya sostiene la página, con la referencia literal de la hipótesis de Luis (resto del ciclo, 21-365 días),
  más barajas para que la p tenga suelo bajo, y el dibujo clásico de una prueba de hipótesis.
- Reutilizar los tres tramos de hoy (lejos 56-365): descartado, Luis formuló «el resto del ciclo» y con 21-365 entran
  34 elecciones y 475 parejas en vez de 23 y 309.
- Una prueba paramétrica (F sobre los cocientes de sumas de cuadrados): descartada, las filas no son independientes
  y la permutación ya está.
- p a una cola (sólo bajada): descartada, la hipótesis nula es «igual» y la alternativa «distinto»; con estos datos
  dan lo mismo.
- Bootstrap recomputando `pooled` en cada remuestreo: descartado por coste (63 s); los términos por elección se
  precomputan.

## D1. Parámetros y estadística (`data/rebano/params.json`, `mtpy/lib/studies/herd/stats.py`)

- `params.json` gana `"chance": {"periods": {"final": [0, 21], "rest": [21, 365]}, "permutations": 20000, "bin_width":
  0.01, "resamples": 2000}`. `min_rows`, `seed`, `n_series`, `pair_days`, `exclude` y `max_days` se reutilizan. Fila
  de `data/README.md` actualizada.
- `pooled_terms(rows_by_election, a, b, min_rows) -> list[tuple[float, float]]`: por elección con parejas compartidas,
  `(A_e, B_e) = (Σ_g w_g r²_a,g, Σ_g w_g r²_b,g)` sobre sus parejas de casas (las mismas sumas de `pooled`); las
  elecciones sin parejas no entran. `pooled` puede construirse sobre ella (o no: constructor).
- `bootstrap_pooled(terms, resamples, seed) -> Optional[dict]`: remuestrea las elecciones con reemplazo, cambio de cada
  remuestreo `√(ΣB/ΣA) − 1`, `{lo, hi}` = percentiles 2,5 y 97,5 sin redondear; `None` con menos de 3 elecciones.
- `null_distribution(rows_by_election, a, b, min_rows, permutations, seed) -> Optional[dict]`: una pasada de la
  permutación estratificada (`_permuted_terms` por elección, un generador) y devuelve `{'observed': cambio observado
  (√(obs_b/obs_a) − 1), 'permuted': ndarray de cambios permutados, 'count_one': barajas con cociente ≤ observado·(1 +
  PERMUTATION_TOLERANCE), 'count_two': barajas con |cambio| ≥ |observado|·(1 − PERMUTATION_TOLERANCE), 'p_one':
  (count_one + 1)/(B + 1), 'p_two': (count_two + 1)/(B + 1)}`; `None` sin parejas compartidas o con `obs_a` 0.
- `histogram(values, bin_width) -> list[dict]`: cajas `{lo, hi, count}` de anchura `bin_width` desde
  `floor(min/bw)·bw` hasta `ceil(max/bw)·bw` (bordes redondeados a los decimales de `bin_width`), `count` enteros, con
  la caja que incluye el máximo cerrada por la derecha; `[]` sin valores.

## D2. El cálculo (`mtpy/lib/studies/herd/study.py`)

- `HerdStudy.chance`: `None` si ninguna elección tiene parejas compartidas con los tramos de `chance`; si no,
  `{'periods': los tramos, 'permutations': B, 'elections': n con parejas compartidas, 'falls': elecciones cuyo
  `compare(rest, final)` baja, 'pairs', 'rows_a', 'rows_b', 'r_a', 'r_b' (2 decimales), 'change' (3), 'ci': {lo, hi}
  (3) o None, 'count_one', 'count_two' (enteros), 'p_one', 'p_two' (`_sig3`), 'null': {'mean', 'sd', 'min', 'max' (4 decimales), 'bin_width', 'bins'}}`.
  Las comparaciones de los dos tramos se calculan una vez por elección pasada (`comparisons(polls, names, params |
  {'periods': chance.periods})`), y de ellas salen `pooled` (cifras), `pooled_terms` → `bootstrap_pooled` (intervalo)
  y `null_distribution` (p y nulo); el histograma sobre `permuted`.
- `summary`, `elections`, `current` y `weeks` no cambian.

## D3. Parte publicada (`bundle.py`, `publish/herd.py`, `publish/studies.py`, `publish/__init__.py`, `pages/analisis.py`, `docs/web/contrato.md`)

- `STUDY_PARTS['rebano'] = ('meta', 'summary', 'elections', 'current', 'weeks', 'chance')`; `SCHEMAS['rebano-chance'] =
  {'chance': (dict, type(None))}`; `STUDY_CSV['rebano'] += 'chance'`; `STUDY_CSV_OF['rebano']['chance'] = ('chance',)`.
- `export_herd_chance(study) -> (data, {'chance': frame})`: `data = {'chance': study.chance}` limpio; el CSV con una
  fila por caja, `CHANCE_COLUMNS = ['lo', 'hi', 'count']` (`count` entero); frame vacío con las columnas sin `chance`.
- `CSV_LABELS['rebano']['chance'] = 'Distribución nula del cambio de la separación (barajas) de «¿Podría ser azar?»'`.
- `contrato.md`: la parte, sus claves y el CSV; las rutas.
- Compatibilidad: la imagen antigua ignora la parte; la imagen nueva sin la parte no pinta la tarjeta 3 ni su entrada
  en el resumen.

## D4. La página (`mtpy/lib/pages/herd.py`, `pages/analisis.py`, `web/templates/analisis_rebano.html`,
`web/dist/js/charts/chance.js`, `web/dist/js/pages/analisis-rebano.js`, `tests/test_js_charts.py`)

- `herd.verdicts` devuelve también `q3`: con `chance.p_two` < 0,01 → («NO», tono `q1`); < 0,05 → («DIFÍCILMENTE»,
  `q2`); si no → («SÍ, PODRÍA», `q3`); sin `chance` → («SIN DATOS», `q2`).
- `herd.p_text(p) -> Optional[str]`: «p < 0,0001» si `p` < 0,0001; «p < 0,001» si < 0,001; «p = 0,047» (tres
  decimales, coma) si no; `None` sin `p`.
- `herd.chance_text(chance) -> Optional[str]` (ojo: el nombre `chance_text` ya existe para la prueba del signo; el
  nuevo se llama `null_text`): «Ninguna de las {20.000} barajas bajó del {−12 %}; lo observado es un {−26 %}, de {1,58}
  a {1,17} veces el error de las muestras: {p < 0,0001} a dos colas. Las separaciones de las tres últimas semanas son
  un {26} % menores que en el resto del ciclo{ (entre el 4 y el 38 %)}. En {25} de las {34} elecciones las mismas
  casas se acercan.» Con `count_two` > 0: la primera frase pasa a «{k} de las {20.000} barajas llegaron tan lejos como
  lo observado ({−26 %}), que va de {1,58} a {1,17} veces el error de las muestras: {p = 0,012} a dos colas.» Con cambio
  ≥ 0: «… son un {x} % mayores …». El intervalo con la lógica de `ci_text` (sin él, sin paréntesis). `None` sin
  `chance`.
- `herd.null_summary_text(chance) -> Optional[str]` (una frase para la tarjeta del resumen): «Barajando {20.000} veces,
  ninguna baraja llega a lo observado: {p < 0,0001}.» / «{k} de {20.000} barajas llegan a lo observado: {p = 0,012}.»
- `herd.sigmas(chance) -> Optional[int]`: `round(|observed − null.mean| / null.sd)`; `None` sin `sd` o `sd` 0.
- Resumen de la página: tercera `answer` «PREGUNTA 3 · ¿Podría ser azar?» con el veredicto `q3` y `null_summary_text`
  (sin `chance`, «Aún no hay datos para esta prueba.»).
- Tarjeta 3 (`id="p3"`, tras la tarjeta 2, antes de «en directo»; `shot('pregunta-3')`): cabecera con el número 3, el
  título «¿Podría ser azar?» y el veredicto. `lead`: «Si en las tres últimas semanas las mismas casas se separasen
  igual que en el resto del ciclo, ¿qué bajada veríamos por puro azar? Para saberlo barajamos {20.000} veces qué
  parejas de sondeos de cada pareja de casas caen en las tres últimas semanas y cuáles en el resto, y medimos la bajada
  en cada baraja.» `ctitle` «Lo que daría el azar, frente a lo observado»; `csub` «{20.000} barajas de las mismas {475}
  parejas de casas en {34} elecciones · cambio de la separación entre el resto del ciclo (de 3 semanas a un año antes)
  y las tres últimas semanas»; `<div class="chart" id="chance-chart" role="img" aria-label="{{ null_text }}">`;
  `.tnote` «Cada barra cuenta las barajas que dieron ese cambio; la línea marca el observado.»; caja `sum` con
  `null_text` en negrita la primera frase; nota `<details class="how">` con `summary` «Qué se baraja y qué p damos» y
  el texto: «Dentro de cada pareja de casas barajamos qué parejas de sondeos caen en las tres últimas semanas y cuáles
  en el resto, manteniendo cuántas hay en cada tramo y con las cifras de los partidos de cada pareja de sondeos
  juntas. La p es a dos colas: cuenta las barajas con un cambio tan grande como el observado en cualquier dirección, y
  suma una a las barajas y al total para no dar nunca cero; con ninguna de {20.000}, p = 1/{20.001}. A una cola,
  {p < 0,0001}. Un sondeo que entra en varias parejas cuenta en todas, lo que hace la p algo optimista; con lo
  observado a unas {8} desviaciones típicas del azar, no cambia la conclusión. Las tarjetas 1 y 2 usan {2.000} barajas
  por elección.» Sin `chance`, la tarjeta no se pinta.
- `web/dist/js/charts/chance.js`: `chanceOption(chance, opts)` y `renderChance(el, chance, opts)`: eje x de valores en
  % (cambio × 100, etiquetas «−10 %»), que cubre las cajas y el observado con un margen de dos cajas; una serie con una
  barra por caja que ocupa exactamente su anchura (`custom` con `renderItem`, o `bar` sobre eje de valores: constructor);
  eje y de recuentos («barajas»); `markLine` vertical en el observado con rótulo «observado {−26 %}» y el color del
  acento; tooltip «de {lo} a {hi} %: {count} barajas»; nombre del eje x «cambio de la separación entre el resto del
  ciclo y las tres últimas semanas». La página lo dibuja desde la isla `initial.chance` (`analisis-rebano.js`); la
  descarga de la tarjeta lo redibuja como los demás.
- `herd_context`: `chance` (la parte), `null_text`, `null_summary`, `sigmas`, `initial.chance`; `verdicts.q3`.
- «Datos y método», `tech`: una frase: «La <b>prueba de la tarjeta 3</b> baraja {20.000} veces, dentro de cada pareja
  de casas de cada elección, qué parejas de sondeos caen en las tres últimas semanas y cuáles en el resto del ciclo
  ({21} a {365} días), y da la p a dos colas con la corrección (k + 1)/(B + 1).»
- `assert_public_wording` sobre los textos nuevos.

## D5. Pruebas y aceptación

- `tests/test_herd_stats.py`: `pooled_terms` sobre `two_elections()` da `[(6, 1.5), (24/7, 12/7)]`; `bootstrap_pooled`
  determinista, `lo ≤ cambio ≤ hi`, `None` con dos elecciones; `null_distribution` sobre el caso de las 36
  permutaciones exactas (`p_one` y `p_two` contra la enumeración, ±0,03 con 4.000 barajas; `permuted` de longitud B;
  `count_*` enteros); `histogram([−0.012, 0.004, 0.031], 0.01)` → cajas de −0,02 a 0,04 con cuentas `[1, 0, 1, 0, 0, 1]`
  y `histogram([], 0.01) == []`.
- `tests/test_herd_study.py` sobre `write_herd_bundle`: `chance` con `elections` 2, `falls` 1, `pairs` 2, `rows_a` 12,
  `rows_b` 10, `r_a` 0,94, `r_b` 0,77, `change` ≈ −0,1775 (±0,0006), `ci` None, `permutations` 20000, `p_one` ≈ 0,147
  (±0,01), `p_two` ≈ 0,433 (±0,015), `null.min` ≈ −0,542 y `null.max` ≈ 1,015 (±0,01), `bins` con `sum(count) ==
  20000`; `chance` None con un paquete sin parejas compartidas (p. ej. `params={'chance': {..., 'periods':
  {'final': [0, 1], 'rest': [1, 2]}}}`).
- `tests/test_publish_herd.py`: la parte valida; el CSV con `lo, hi, count` enteros; `publish_study` escribe `chance`;
  fixture `tests/fixtures/bundle/rebano-chance.json` con las cifras reales (ver el plan).
- `tests/test_js_charts.py`: `chanceOption` da una serie con tantas barras como cajas, `markLine` en el observado × 100,
  eje x que incluye el observado y las cajas, rótulo «observado −26 %».
- `tests/test_pages.py`: tercera `answer` en el resumen, tarjeta `#p3` con veredicto «NO», los textos (`null_text`
  exacto con el fixture, `lead`, `csub`, nota), `id="chance-chart"`, `initial-data` con `chance`, el `tech`; sin la
  parte: ni tarjeta ni `answer` ni «Podría ser azar»; `assert_public_wording`; los contadores existentes (preguntas,
  capturas `pregunta-3`, píldoras CSV 4).
- Aceptación con datos reales (sólo lectura): las cifras de «Hechos de partida» y el tiempo total (< 50 s); la página
  con la parte real en Chrome (tarjeta 3 y su imagen).

## Pasos tras la fusión

Desplegar la imagen y publicar el estudio justo después (quien integre, con el OK de Luis). La imagen antigua ignora
la parte nueva; la nueva sin la parte no pinta la tarjeta.

## Lo que queda al constructor

Nombres internos, el tipo de serie del histograma, el formato del tooltip, el color de las barras dentro del sistema
del sitio, si `pooled` se reescribe sobre `pooled_terms`.

## Cierre (2026-11-09)

Construido con subagentes en la rama `rebano-azar` (desde dev 0373fee): 6a96e21 + 65b14f0 + 7d3f4f4 (estadística:
`pooled_terms`, `bootstrap_pooled`, `null_distribution`, `histogram`, parámetros), 13d5f05 (`HerdStudy.chance`),
2429dc5 + d955785 (parte `chance`, CSV, contrato, fixture), febb1af (tarjeta 3, veredicto, textos, `chance.js`) y
c15089e (ola de correcciones tras la revisión final). Con datos reales (2026-11-09): 34 elecciones, 25 bajan, 475
parejas, R 1,58 → 1,17 (−26,3 %), intervalo [−38 %, −4 %], ninguna de 20.000 barajas llega (p = 1/20.001), nulo con
desviación típica del 3,1 % y mínimo −11,9 %; cálculo 50,4 s (antes 41; la lectura de S3 manda).

Decisiones tomadas en la ejecución: `count_two` con tolerancia aditiva en la escala del cambio (la multiplicativa
daría p ≈ 0 con un cambio observado que fuese ruido en torno a 0); `count_one` sobre cocientes, igual que `pooled_p`;
`histogram` ignora no finitos, amplía una caja si un extremo cae fuera y toma los decimales de `Decimal`; clave de
contexto `chance_test` (la clave `chance` ya era la de las palabras de la prueba del signo); el resumen pasa a «Tres
preguntas, tres respuestas» cuando hay parte; sin parte se oculta también su CSV; el borde del azar se redondea hacia
fuera para que «ninguna bajó del −12 %» sea siempre cierto; espacios duros en las cifras de la tarjeta; la p a una
cola de la nota sólo si el cambio es una bajada; `_chance` reutiliza las comparaciones del estudio reetiquetadas.
Arreglado de paso: dos espacios que faltaban en el texto del método («en todas. El intervalo», «veces. La robustez»).
Aparcado: el bucle de barajas duplicado entre `null_distribution` y `pooled_p`; `pooled()` como guarda; ruido de coma
flotante en los extremos del eje para anchuras de caja no binarias; la celda por defecto de la rejilla de robustez
recalcula las comparaciones (≈1 s); la píldora CSV se filtra por el texto del enlace; «(0 %)» para cambios mínimos.
Recomendación del revisor final, para Luis: el «NO» se apoya en la permutación dentro de cada elección; el intervalo
entre elecciones, que no sufre la dependencia de un sondeo en varias parejas, también excluye el cero pero por poco
(−4 %); valorar que el veredicto baje a «DIFÍCILMENTE» si el intervalo cruza el cero.
