# Las tres piezas del error, con las cifras del último cálculo (/metodo/escanos)

Fecha: 2026-10-25. Estado: aprobada por Luis y CONSTRUIDA el 2026-10-25 en `dev` (plan
`docs/superpowers/plans/2026-10-25-metodo-escanos-composicion-error.md`, ejecución nativa, sin commits: los hace Luis);
la tarjeta aparece al publicar un run de las generales con el código nuevo. Desviaciones: los tres puntos de «Cómo
leerlo» van como prosa visible, por la regla de /metodo de tenerlo todo a la vista; `dof` se publica entero; el
nombre del partido va sin color en la tabla, como la de barreras de la misma página; los tests de página viven en
`tests/test_pages_ejemplo.py`. Origen: su pregunta del 2026-10-25 sobre en
qué se descompone la incertidumbre de las predicciones y si tiene en cuenta las elecciones anteriores (2004, 2023);
la respuesta existe en el código y en la fórmula de /metodo/escanos, pero no en cifras. Alcance acordado: «sólo la
composición del error», sin la lectura de qué haría falta para que fallara la mayoría.

## Intención

Que quien dude de las horquillas vea en /metodo/escanos, con los números del último cálculo publicado de las
generales, cuánto pesa cada una de las tres piezas de la fórmula que la página ya explica: el error del promedio
`s`, el error histórico de los sondeos `e` y la deriva de la opinión `d`; y que lea de dónde sale `e`: la última
encuesta de cada casa en todas las generales con sondeos desde 1982, con el mismo peso, 2004 y 2023 incluidas, y
sin dirección. Hoy la página dice que `e` «suele ser el mayor de los dos» y que `e` se mide «en las elecciones
pasadas»; con el run del 24-10, para el PP, `s` = 0,5, `e` = 2,6 y `d` = 1,2 puntos a 34 días: `e` es el 81 % de la
varianza. Eso es lo que falta enseñar.

## Qué hay hoy

- `Simulator._fan` calcula por partido y horizonte `err` (`s`, error estándar de la regresión local, cluster por
  casa), `pct_err` (`e`, el estimador de error terminal: regresión del error absoluto de la última encuesta de
  cada casa, publicada seis o más días antes, en las generales `featured` del ámbito estimador desde 1982, sin
  descuento por antigüedad, sobre tamaño del partido, regional y semanas) y la varianza de la deriva
  (`v2drift.var(h, mean, age)`, `d = p·m·sqrt(k·h)`), y devuelve `mean`, `sd = sqrt(s² + e² + d²)`, `lo`, `hi`
  y `dof` (los grados de libertad de la t, `nobs − 2`). `fan()` tira `dof`; `export_fan` publica `name, horizon,
  mean, sd, lo, hi` en la parte `fan` del run (`horizons`, `rows`) y su CSV.
- `meta.diagnostics` del run ya lleva `drift_k`, `multiplier` (partidos jóvenes) y `ages`; no lleva qué
  elecciones ajustan el estimador de error ni el ámbito estimador (`est_scope`, `es` también en las autonómicas).
- `/metodo/escanos` lee cifras vivas del run nacional y del backtest por `national_figures()`
  (`mtpy/lib/pages/metodo.py`: `n_sim`, coberturas) y el ejemplo vivo del promedio por `weights_example()`, que
  devuelve `None` y oculta su tarjeta cuando el run no trae la parte. Las tarjetas «Del promedio al sorteo del
  voto» y «Hoy y Elección» dan las fórmulas de `σ` con `s`, `e`, `d` y la t de Student, y la nota de que cada
  partido se sortea por su lado.

## Enfoques

- **A. Ampliar la parte `fan`** con las piezas por fila y el estimador de error en `meta.diagnostics`. Mismas
  magnitudes que ya viajan en el abanico, en el mismo sitio; tres columnas, dos campos y una republicación.
  El `fan` deja de ser sólo un abanico, pero el gráfico de Escaños ignora las columnas nuevas.
- **B. Parte nueva `errors`** con una fila por partido principal y las elecciones del estimador en su cabecera.
  Más limpia de leer, pero una parte más en paquete, API, contrato y tests para los mismos números.
- **C. Deducirlo en la página** de la banda de la serie y de los totales del `fan` por diferencia de cuadrados.
  Sin republicar, pero la banda lleva un cuantil t con pocos grados de libertad y el número no sería el del
  modelo. Descartado: la tarjeta existe para enseñar las cifras exactas.

Elegido **A** (Luis, 2026-10-25).

## Decisiones (mías, con su razón)

1. **Columnas nuevas del `fan`**: `err`, `pct_err`, `drift` y `dof` en cada fila (nombres del `frame` del
   simulador, en puntos; `drift` es la desviación en ese horizonte, 0 en el horizonte 0). `Simulator.fan()` las
   conserva en su salida larga (la ancha no cambia); `export_fan` las publica en la parte y en el CSV gemelo.
   Ninguna página las lee salvo la tarjeta nueva.
2. **`meta.diagnostics`** gana `error_events` (fechas ISO ascendentes de las elecciones cuyos sondeos ajustan el
   estimador de error: las fechas de evento distintas de los sondeos que entran en `get_error_estimator_data`
   con `ERROR_DRANGE` y `n_last`, leídas con `filter_polls` con los mismos argumentos) y `error_scope`
   (`est_scope`). Viven en el run para que la cuenta no se quede en 14 cuando pase el 29-N.
3. **Tarjeta nueva en /metodo/escanos**, id `errores`, justo después de «Hoy y Elección» (necesita `d`), con
   entrada en la navegación de la página cuando existe. Título: «Un ejemplo con datos publicados: los tres errores
   del {run_at}». Entradilla: los números del último cálculo de las generales, en puntos de voto. Tabla por partido
   principal (`bmaps.main`, por voto descendente): Partido · Voto · Promedio `s` · Histórico `e` · Deriva `d` ({h}
   días) · Hoy `σ` · Elección `σ` · Rango 95 % en Elección. Dos decimales; el rango con el formato de la web
   (`fmt_range`).
4. **«Cómo leerlo»** con tres ideas, en el orden de la pregunta de Luis: (a) de dónde sale `e`: la última
   encuesta de cada casa en las {n} generales de {año primero} a {año último}, todas con el mismo peso, con sus
   fallos grandes dentro, y por eso vale varias veces la banda del promedio; (b) `e` no tiene dirección: el
   modelo no sabe hacia qué lado se equivocarán los sondeos, sólo cuánto suelen hacerlo; (c) las colas: la t
   toma sus grados de libertad del número de sondeos, así que son casi normales. Sin cifras del día en la prosa
   salvo las de la tabla; sin comparar con 2004 ni con 2023 (fuera del alcance).
5. **Sólo las generales y sólo el último run**, como el resto de cifras vivas del método (`NATIONAL_SCOPE`); la
   página no tiene selector de run. `error_example()` en `metodo.py`, gemelo de `weights_example()`: devuelve
   `None` sin run nacional, cuando el `fan` del run no trae `pct_err` (publicado antes de esta spec) o cuando el
   `meta` no trae `error_events`; la tarjeta y su entrada de navegación se ocultan.
6. **Horizonte de la tabla**: el mayor del `fan` (`horizon_max`, el de la elección) frente al 0; los demás
   horizontes del abanico no se enseñan aquí (ya están en el gráfico de Escaños).
7. **Redacción**: reglas de Luis (`sólo` con tilde, encuestadora/casa, nunca empresa; `assert_public_wording`);
   tono de /metodo/promedio (nosotros, todo a la vista).

## No objetivos

- La lectura de «qué haría falta para que fallara»: colchón de escaños del bloque y puntos de fallo equivalentes
  (Luis, 2026-10-25: «sólo la composición del error»).
- Cifras por ámbito autonómico o por run fijado; el gráfico del abanico de Escaños; el modelo y sus estimadores.
- El sorteo conjunto (composición M7): la página ya lo explica y no cambia.

## Partes

- `mtpy/lib/simulator.py`: `_fan` devuelve también `err`, `pct_err` y `drift`; `fan()` deja de tirar `dof`; helper
  `error_events()` (fechas de evento del conjunto de entrenamiento del estimador de error; lista vacía sin
  `v2err`).
- `mtpy/lib/publish/forecast.py`: `export_fan` publica las columnas nuevas (parte y CSV); `export_meta` añade
  `error_events` y `error_scope` a `diagnostics`.
- `mtpy/lib/bundle.py`: esquema del `fan` y del `meta` si validan claves de fila; `docs/web/contrato.md`:
  enmienda 2026-10-25 con las columnas y los campos.
- `mtpy/lib/pages/metodo.py`: `error_example()`; el contexto de `/metodo/escanos` lo expone como `errors`.
- `web/templates/metodo_escanos.html`: la tarjeta, su «Cómo leerlo» y la entrada de navegación condicionales.
- Tests: `test_simulator_unit.py` o el equivalente del `fan` (columnas y `drift` = 0 en el horizonte 0),
  `test_publish*.py` (exportador y meta), `test_pages.py` (tarjeta con el fixture ampliado, oculta con un `fan`
  sin `pct_err`, orden y formato de la tabla, navegación, `assert_public_wording`); fixtures
  `tests/fixtures/bundle/fan.json` y `meta.json` ampliados.

## Comprobación

- Suite unitaria completa en verde; la página del método con el fixture ampliado y con el antiguo.
- Chrome sin cabeza sobre el bundle de pruebas (envoltorio ASGI con `/dist/`): la tarjeta renderizada, sin
  errores de consola.
- Con un run real tras desplegar: las cifras del PP coinciden con `sd` del `fan` (`sqrt(s² + e² + d²)` = `sd`
  en cada horizonte) y con la banda de la serie.

## Operación

Desplegar y publicar un run de las generales (`publish forecast es`, el diario) para que la tarjeta aparezca;
los runs anteriores no se tocan y la página sólo lee el último. Los runs autonómicos ganan las columnas cuando se
republiquen, sin efecto en la web.
