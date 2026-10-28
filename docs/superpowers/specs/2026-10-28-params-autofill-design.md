# Relleno automático de `parties` y `bmaps` en `data/params.json`

Fecha: 2026-10-28. Estado: aprobado y CONSTRUIDO en dev el 2026-10-28 (acción `params` del job `load`, 18 ámbitos,
108 elecciones); `data/params.json` reescrito y pendiente de revisión de Luis antes de su commit.

## Intención

`data/params.json` pasa a ser la fuente única de `parties` y `bmaps` de cada elección de cada ámbito. Hoy el
fichero sólo cubre unas pocas elecciones (6 de 17 en `es`, 1 de 6 en Madrid) y el resto se deriva en caliente
en `get_event_params` (`mtpy/lib/data.py`), con una regla de `vs` que deja fuera a los partidos que polean poco.
Luis quiere que `vs` recoja todos los partidos de su bloque y que todo quede escrito y revisable.

Es el primer paso de otro cambio ya acordado: que la web agrupe «Bloques» en Izquierda, Derecha y Otros a
partir de `vs` (no forma parte de este diseño).

## Objetivos

- Una acción de job que recorra todas las elecciones de los ámbitos pedidos y escriba, por elección,
  `parties.event`, `parties.polls`, `bmaps.main`, `bmaps.blocks` y `bmaps.vs`.
- Sobrescribe siempre esas cinco claves, también en las elecciones escritas a mano (2026, 2023, 2019); Luis las
  comprueba al final con el diff.
- Conserva intactas las demás claves de la elección (`smap`, `threshold`, `threshold_scope`, `max`, `min` y
  cualquier otra) y las elecciones de ámbitos no pedidos.

## No objetivos

- Cambiar el modelo, la web o los publicadores. Sólo el fichero y la derivación de respaldo.
- Mantener `max` y `min` automáticamente: siguen siendo manuales.
- Tocar `data/sesgo-cis/params.json` ni `data/cera/params.json`.

## Anti-objetivos

- Que fichero y respaldo en caliente digan cosas distintas para la misma elección.
- Un script suelto: Luis quiere jobs (`job.py`).

## Reglas de derivación (decididas con Luis)

Por elección, con la matriz del ciclo de `get_event_dmat` (sondeos del ciclo + resultado):

- `parties.polls`: todo partido con columna en los sondeos del ciclo o en el resultado, con las candidaturas
  primero (en su orden) y después los nombres sólo encuestados (Sumar, UPN, AC en 2026), como Luis los ordena a
  mano (añadido 2026-10-28 tarde).
- Listas manuales (añadidas 2026-10-28 tarde, tras la retirada de AC y la vuelta de Compromís): `parties.exclude`
  (nombres que nunca son candidatura aunque tengan sondeos; siguen en `polls`) y `parties.include` (candidaturas
  aunque aún no tengan sondeos; se añaden a `event` y `polls`). El job las conserva y la derivación las aplica,
  en elecciones pasadas y futuras.
- `parties.event`: los partidos con resultado. En una elección sin resultado (futura), `polls` menos los
  nombres que el `smap` de esa elección absorbe con una regla `agg` (en 2026 quita SUMAR y UPN y reproduce la
  lista manual de Luis).
- `bmaps.main` (regla revisada el 2026-10-28 a petición de Luis, criterio estadístico en vez de editorial): partidos
  de `event` encuestados por casi todas las casas en el tramo final y con peso medible. Ventana: los sondeos de
  los últimos 42 días, ampliada a los 30 más recientes cuando hay menos (elección en curso). Presencia ≥ 90 % de
  esos sondeos, contando como presente al partido o a los nombres que absorbe con una regla `agg` nacional: sin
  `regions` o con `regions.exclude` (FA hereda los sondeos de SUMAR aunque no se presente en Valencia); una regla
  que lista sus provincias es regional y no cuenta, y media de voto en la ventana ≥ 1 %. Sustituye al «> 95 % de los sondeos a
  1-42 días», que con 6 sondeos a un mes del 29-N dejaba fuera a UP (5 de 6) y bailaba con cada sondeo. El rating
  no usa `main`; sí el error y sesgo medios publicados, las tarjetas, la parrilla de efectos de casa y las
  métricas por provincia del backtest, siempre dentro de cada elección (el `Computer` anula en cada elección los
  partidos que no están en su propio `main`).
- `bmaps.blocks`: Derecha, Izquierda, Regionalista y Separatista según `parties.block` del catálogo, sobre los
  partidos de `polls`. Sin bloques vacíos. El partido comodín de bloque «Otros» (id 528, nombre `-`) no entra.
- `bmaps.vs`: Derecha e Izquierda completas, idénticas a las de `blocks`.
- Orden de los partidos en todas las listas: por resultado de la elección descendente; sin resultado, por la
  media de los sondeos del ciclo descendente. El primero de cada bloque da el color del bloque en la web.
- Se saltan las elecciones sin sondeos ni resultado; las demás se escriben aunque `polls` quede vacío.

## Enfoques considerados

1. **Acción `params` del job `load`** (elegido). `python job.py load '{"what":"params","scopes":"all"}'`, con
   `dry_run` que enseña el diff sin escribir. Encaja con el job que ya lee datos externos hacia el repo y con la
   preferencia de Luis por jobs.
2. Acción en el job `update`. Descartado: `update` encadena cargas y cómputos de datos, y esto escribe un fichero
   versionado del repo.
3. Dejar la derivación sólo en caliente y cambiar la regla de `vs`. Descartado: Luis quiere ver y revisar las
   listas, y las elecciones escritas a mano quedarían con reglas distintas.

## Cómo encaja

- La función de derivación vive en `mtpy/lib/data.py` (o `load.py`), compartida por la acción del job y por la
  derivación de respaldo de `get_event_params`, que se queda para una elección aún no escrita y pasa a la misma
  regla de `vs`.
- El fichero conserva su formato: sangría de 4, mismas claves en el mismo orden (`parties`, `bmaps`, `smap`...),
  ámbitos y elecciones en el orden actual; las elecciones nuevas, por fecha descendente como hoy.
- `dry_run` imprime, por ámbito y elección, qué claves cambian (lista antigua → nueva) y nada más.
- Tests: la derivación con una matriz de ciclo de fixtures (pasado y futuro, smap con `agg`, bloque vacío,
  partido «Otros»); la reescritura del fichero con un JSON temporal que conserva claves manuales y orden; la
  acción del job en seco.

## Consecuencias que Luis asume

- `vs` entra en el modelo (errores por bloque, rating, efectos de casa, backtest). Al ampliar `vs` en los
  autonómicos y en las generales anteriores a 2019 cambiarán algo esas cifras: tras aceptar el fichero habrá que
  `compute` + `publish analysis` + `publish event` (todas) + backtest + `publish backtest`.
- Las listas manuales de 2026 y 2023 se pierden si la derivación difiere; el diff lo enseña.

## Lo que queda al constructor

- Nombre y firma exactos de la función de derivación y dónde vive.
- Cómo obtener la media de sondeos del ciclo para ordenar en elecciones futuras (de la misma matriz).
- Formato de las líneas de `dry_run` y del resumen final del job, en la línea de las acciones existentes.
