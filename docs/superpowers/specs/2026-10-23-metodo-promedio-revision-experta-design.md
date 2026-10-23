# /metodo/promedio con el texto de la revisión externa («Cómo calculamos el promedio de encuestas», propuesta 2)

Fecha: 2026-10-23. Rama: `dev`. Página: `web/templates/metodo_promedio.html`. Decidido con Luis en conversación.
Estado: implementado el 23-10 (plantilla, `metodo.py`, `weights.js`, `metodo.js`, tests), sin commit, para que Luis revise el
diff. Sin backtest publicado, la tarjeta «Cómo comprobamos si funciona» describe la comparación sin cifras; sin el análisis
de errores, la tarjeta del CIS y su nota (la cuarta) no se renderizan y las demás se numeran sin hueco.

## 1. Intención

Luis pidió a un experto externo una redacción alternativa de `/metodo/promedio` (`demoscopio-promedio-propuesta-2.md`:
diez apartados, seis notas técnicas y una lista de observaciones «para Luis»). Quiere **ese texto en la página**,
contestando a sus observaciones finales y aprovechando los gráficos de los dos ejemplos (la diana del CIS y los pesos
del PP). El experto no auditó el código; antes de escribir se contrastó cada afirmación suya con `forecaster.py`,
`computer.py`, `core/utils/stat.py`, `simulator.py`, `publish/forecast.py` y `backtest/results/`: todas las reglas que
describe son las del modelo.

## 2. Enfoques considerados

1. **Fusión** del texto del experto con la página actual (hecha el 23-10 por la tarde y descartada por Luis: quería el
   texto del experto, no un injerto).
2. **El texto del experto como página (elegido)**, con sus seis notas en el componente de notas, sus fórmulas en MathML
   y sus dos ejemplos sobre las tarjetas vivas que ya existen. Las cifras que él copió de la web vuelven a ser variables.
3. Sustituir el componente de notas por sus seis notas al pie en texto plano: descartado, el componente ya hace lo que
   él propone (enlaces desde el texto, notas al final, desplegables).

## 3. Decisiones de Luis (23-10)

- **Casos reales.** Se conservan las tablas fechadas de la página actual (dispersión de nueve encuestadoras, ratings,
  efecto CIS-PP, peso de la campana por antigüedad, cuentas CIS/GAD3) y las mediciones de una frase (CIS como una
  muestra de 400-460; dispersión de 2,4 puntos de las inclinaciones en el PP; persistencia del 40-70 % y 61 %; consenso
  frente a encuestadora en 14 ciclos y 43 encuestadoras, 2,4 frente a 1,9; la ilustración de la raíz). Se prescinde de
  las dos historias largas (el seguimiento de GAD3 para ABC en 2023 y el solape de Sigma Dos entre Antena 3 y El Mundo).
- **Backtest.** Se mantiene la regla de Luis: no se publica la valoración del experto («la diferencia es pequeña: no
  basta…»); se dan las cifras y su contexto (cinco generales, 2015-2023) de forma descriptiva.
- **Criterio del experto para el cuerpo**: texto ligero, las fórmulas de detalle en las notas. Deroga para esta página la
  regla del 21-10 de «cada fórmula en el cuerpo».

## 4. Estructura

| id | Título | Fuente |
|---|---|---|
| — | h1 «¿Cómo promediamos las encuestas?» + índice | igual |
| `resumen` | Qué nos dice el conjunto de las encuestas | §0 + §1 del experto (la imagen de la curva) |
| `que-entra` | Qué encuestas incorporamos | §2, con la política real de paneles online y el corte por fin de campo |
| `peso` | Cuánto pesa cada encuesta (h3 `muestra`, `dispersion`, `sobrerrepresentacion`, `rating`, `cuentas`) | §3 + tablas conservadas |
| `efectos-de-casa` | Cómo ajustamos las diferencias entre encuestadoras | §4 (sin «El CIS: 38 de 41») + tabla CIS-PP |
| `ejemplo-cis` | El CIS: N de M elecciones en la misma dirección | §4-CIS sobre la tarjeta viva (diana + tabla), cifras vivas |
| `regresion` | Cómo estimamos la tendencia de cada día (h3 `anchura`, `consecuencias`) | §5 |
| `banda` | Qué significa la banda del promedio | §6 |
| `ejemplo` | Un cálculo completo, con datos publicados | §7 sobre la tarjeta viva (gráfico + tabla + CSV) |
| `voto-estimado` | Del promedio a la estimación del voto | §8 |
| `comprobacion` | Cómo comprobamos si funciona | §9 descriptivo, cifras vivas del backtest |
| `limites` | Qué límites tiene el promedio | §10 |
| `notas` | Notas (6: `muestra`, `pesos`, `regresion`, `efectos`, `binomial`, `intervalos`) | sus seis notas |

Las claves de las notas son letras minúsculas (lo exige la numeración del componente). Las llamadas van en el texto
donde él las puso.

## 5. Respuestas a las observaciones «antes de publicar» (y dónde quedan)

1. **Paneles online.** `mtype = 'online'` marca los paneles abiertos de participación voluntaria (ElectoPanel, con
   cuotas y ponderación, pero autoseleccionado). Las encuestas por internet con muestra elegida por la encuestadora
   (40dB, DYM) no llevan esa marca y entran. → en `que-entra`.
2. **Corte temporal.** La regla es sobre el fin del campo: `days` = días entre el fin del campo y la elección,
   `drange = 6` en producción (`publish/forecast.py`). El backtest aplica la misma regla, así que solo usa sondeos que
   ya eran públicos a la fecha límite. La LOREG (art. 69.7) es contexto, no la regla. → en `que-entra`.
3. **Dispersión y solapamientos.** Redacción del experto (comportamiento, no causa; coincidir en fechas no prueba
   compartir entrevistados). La χ² del rebaño usa n − 1 grados de libertad, la cuenta conservadora, porque los cuatro
   partidos de un sondeo no son independientes (`pool_herding`). La suma recorre todos los partidos del promedio con el
   k de cada uno; los residuos son frente a la tendencia estimada. → cuerpo y nota `muestra`.
4. **Correcciones de encuestadora.** Especificado desde `fit_house_effects`: en cada una de las tres iteraciones la
   desviación es la de los sondeos publicados frente al promedio ajustado con los sondeos corregidos de la iteración
   anterior; su error es √(varianza ponderada insesgada / n_ef de Kish); W_c es el peso total, decaído por antigüedad,
   de los sondeos de la encuestadora en esa serie; la antigüedad del ciclo se cuenta desde el último sondeo y la de la
   historia desde la elección; la historia es la de todos los ámbitos (peso 1 generales, 0,5 autonómicas), por linaje de
   partido (PSOE ↔ PSC), con al menos tres elecciones, desviación relativa acotada a ±50 % y tamaño no menor de 2
   puntos. No se llama «bayesiana» a la contracción. → nota `efectos`.
5. **Anchura.** Especificado desde `Kernel`: ISJ sobre las fechas con sus pesos (reescalados para sumar n_ef),
   histograma de 256 celdas entre el mínimo y el máximo ampliados media desviación típica, DCT, punto fijo de Botev
   (ℓ = 7) con `brentq`; suelo y respaldo 2π · separación media entre sondeos ordenados; adaptativo sobre la malla diaria
   con f_k(t) = Σ w_i K(t_i − t; h_k(t)) sin normalizar, g_k su media geométrica, tres iteraciones, sin cotas; las fechas
   repetidas simplemente suman. No hay validación cruzada ni se afirma. → nota `regresion`.
6. **Intervalos.** c = G/(G − 1) · (n_ef − 1)/(n_ef − 2) con el n_ef de Kish; ν = min(n_ef − 2, G − 1); con menos de
   dos encuestadoras en la ventana se usa HC1 (`get_robust_cov`). El bootstrap de septiembre remuestreó encuestadoras
   con los efectos fijados (100 réplicas): 0,38 frente a 0,25 en el PP; reestimando los efectos sale 1,18, que mide otra
   cosa (`docs/analisis-2026-09/metodo-10`). → cuerpo y nota `intervalos`.
7. **MAE, escala y t.** e es el error absoluto medio que predice la regresión y se usa tal cual como escala; la t tiene
   ν = n_obs − 2 (mínimo 3) y se multiplica directamente por σ_voto, así que su desviación típica real es
   √(ν/(ν − 2)) veces la escala (1,4 % con ν ≈ 70; 73 % con ν = 3); e y s pueden solaparse. La comprobación es la
   cobertura (96 % a seis días). → nota `intervalos`.
8. **Pruebas históricas.** Cinco generales (20-12-2015, 26-6-2016, 28-4-2019, 10-11-2019, 23-7-2023), horizontes 6, 14,
   30, 60, 90 y 180 días; el error es la media del error absoluto por partido; las líneas base son el último sondeo
   (media de los sondeos de la última fecha) y la media simple de los sondeos de los 28 días anteriores; ratings, pesos
   y efectos se recalculan con la información anterior a la fecha límite. Las cifras pasan a ser vivas
   (`national_figures`). → `comprobacion`, enlace a `/metodo/backtest`.
9. **Las elecciones del CIS.** Bloques Izquierda/Derecha del mapa `vs`; el último sondeo del CIS con fin de campo entre
   6 y 42 días antes de cada elección analizada (`POLL_ERRORS_PARAMS`); un error exactamente 0 no cuenta en ningún
   sentido y tampoco en n; el periodo empieza en septiembre de 2018 por un hecho externo (el cambio de estimación del
   CIS), declarado en el texto; se eligió el CIS por ser el caso más llamativo, y se dice; la probabilidad se calcula
   para un solo bloque. La nota muestra la cuenta binomial con cifras vivas (`sign_chance` da los casos favorables, el
   total y la razón exacta). → `ejemplo-cis` y nota `binomial`.
10. **Casos especiales.** Un partido con menos de tres sondeos utilizables no se ajusta y queda en «otros»; «otros» es
    100 menos la suma de los partidos seguidos (puede ser negativo si un sondeo suma más de 100); los cambios de
    candidatura se resuelven por elección con el mapa de linaje de `params.json`. → `limites`.

Presentación: se conservan los dos gráficos; el de pesos gana un conmutador Publicada/Corregida para los puntos (los
datos ya viajan en la isla); las dos tarjetas de ejemplo dicen fecha de cálculo, último sondeo y fecha leída; las notas
tienen llamadas desde el texto.

## 6. Cambios de código

- `metodo.py`: `sign_chance` devuelve además `ways` (casos favorables), `total` (2ⁿ) y `odds_exact`;
  `national_figures` añade `backtest` (horizonte más corto: `n_cases`, `mae`, `mae_last_poll`, `mae_mean_4w`,
  `cov95`, `cov_band95`) y los años de las elecciones evaluadas.
- `metodo.js` + `weights.js`: `weightsOption(example, {field})` dibuja los puntos con `pct` o `pct_adj`; un control
  `#weights-field` en la plantilla, oculto sin JS, redibuja. El eje Y tiene los mismos límites en las dos vistas
  (`valueRange`: la unión de las dos cifras y de la banda, medio punto más ancha y redondeada a la unidad) y el redibujo
  se fusiona en el gráfico ya montado, así que solo se mueven los puntos (Luis, 23-10).
- Tests: `tests/test_pages_ejemplo.py` (frases, ids, notas), `tests/test_pages.py` (figuras), `tests/test_js_charts.py`
  si el conmutador tiene función pura.

## 7. Lo que no entra

- Las frases de aviso del revisor («no se ha auditado el código», «para una reproducción exacta deben especificarse…»):
  se sustituyen por la especificación real.
- Los párrafos-placeholder (paneles online, corte temporal).
- La valoración del backtest (decisión de Luis, punto 3).
- Las cifras literales de los ejemplos vivos.
