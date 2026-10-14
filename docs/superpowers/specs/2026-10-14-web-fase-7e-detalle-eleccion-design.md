# Web, fase 7e: el detalle de una elección

Fecha: 2026-10-14. Estado: cerrada el mismo día en `dev`.

## Qué pidió Luis

En la página de una elección pasada (`/elecciones?scope=&date=`):

1. "Resultados" enseña los 20 primeros partidos y un "Ver más".
2. "Último sondeo de cada encuestadora": la diana más grande y centrada, con los puntos en azul.
3. Ya no interesa la tabla de errores de todos los sondeos de cada encuestadora: solo el error del último sondeo de
   cada una.
4. Debajo de los resultados, una fila con dos tarjetas a dos columnas: a la izquierda las barras horizontales y a la
   derecha la lista de errores, las dos con el último sondeo de cada encuestadora. Columnas: Encuestadora, Muestra,
   Días antes, Error medio, Error de bloques, y los bloques (Derecha, Izquierda).
5. En la fila siguiente, la diana grande con todos esos errores.
6. Por último, la lista "Todas las encuestas" del ciclo, igual que la de Promedios (con paginación).

## Decisiones

- **Parte nueva `cycle` de la elección** (`event-cycle@1`: `parties`, `columns`, `polls`, con la forma del `polls` de
  un run). El detalle solo lee el paquete y el `polls` de la elección trae errores, no lo que publicó cada sondeo, así
  que `publish event` escribe además los sondeos del ciclo tal como se publicaron, con los escaños de las casas que
  los dan (`forecaster.published_seats`, extraído de `Forecaster.load_poll_seats`). Son los mismos sondeos que
  `polls.rows` (`Computer.filter_polls(featured=True)`): cada fila de la tabla tiene su error. Gemelo CSV `cycle.csv`,
  ruta `/api/v1/events/{scope}/{date}/cycle` y enlace en Descargas.
- **Compatibilidad**: una elección publicada antes de la parte enseña un aviso en lugar de la tabla hasta que se
  republica (`"event_date":"all"`). Lo demás de la página no depende de `cycle`.
- **El último sondeo de cada casa sale de `polls.rows`** (`pages.last_poll_rows`: el de menos `days` por casa), no de
  los `last_*` de la parte `pollsters`, porque la tabla pide la muestra del último sondeo y `pollsters` solo trae la
  mediana. La parte `pollsters` sigue publicándose (CSV y API) pero la página ya no la lee.
- **Orden**: barras y tabla de menor a mayor error de bloques (sin signo); los bloques con signo.
- **Diana**: `#bullseye-chart.big` (720 px, centrada), puntos en el azul del sitio (`color` de `renderBullseye`) y
  `labelLayout.hideOverlap` para que las etiquetas no se amontonen cuando hay muchas casas.
- **"Ver más"**: el servidor pinta todas las filas, oculta las que pasan de 20 y el botón; el JS enseña el botón y
  destapa las filas. Sin JS se ven 20 y el CSV las tiene todas.
- **Tabla compartida**: el marcado de "Todas las encuestas" pasa a `_polls_table.html` y su cableado a
  `pages/polls-wire.js` (`wirePollsTable(polls, catalog)`), usados por Promedios y por el detalle; el detalle embebe el
  catálogo de los partidos de la tabla (`initial.catalog`) porque su `meta` es la de la elección, no la del run.
- **Orden de las tarjetas**: cabecera, Resultados, fila (barras | tabla del último sondeo), La diana, Error de cada
  sondeo (histograma, que Luis no mencionó y se conserva), Todas las encuestas, El modelo a cada horizonte,
  Descargas.

## Comprobación

Pruebas (`tests/test_pages.py`, `test_publish_unit.py`, `test_bundle_unit.py`, `test_webapi_unit.py`,
`test_jobs_publish.py`) y captura con Chrome sin cabeza de una elección de prueba con 14 casas, 29 resultados y los
315 sondeos reales del run como ciclo: "Ver más" destapa las filas, la tabla pagina (16–30 de 315), sin errores de
consola.
