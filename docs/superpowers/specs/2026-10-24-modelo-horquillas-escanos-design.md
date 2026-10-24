# Horquillas de escaños en «Lo que dijo el modelo»

Fecha: 2026-10-24. Estado: petición de Luis del 2026-10-24 («añadir las horquillas de escaños tanto en partidos
como en bloques»); construido el mismo día en `dev` sin ronda de preguntas (cambio pequeño, decisiones abajo);
backtest y eventos republicados el mismo día (ver Operación). Falta desplegar la web.

## Intención

En el detalle de una elección, la tarjeta «Lo que dijo el modelo» enseña por horizonte el voto y los escaños que
el modelo habría publicado frente al resultado. Hoy los escaños van en seco (un número); Luis quiere ver también
la horquilla del modelo, en la tabla de partidos y en la de bloques, para juzgar si el resultado cayó dentro.

## Qué hay hoy

- La parte `model` del evento (`export_event_model`) copia las tablas `shares`, `seats` y `blocks` del backtest.
  `seats` ya trae por partido los cuantiles `lo50/hi50/lo80/hi80/lo95/hi95` del nowcast y, con ejecución a
  horizonte, los mismos con sufijo `_h`. `blocks` sólo trae `model_seats`, `p_majority` (y sus `_h`).
- `horizon_tables` (`mtpy/lib/pages/elecciones.py`) funde las tres tablas por horizonte; la plantilla
  `eleccion.html` pinta las dos tablas; el JS sólo cambia de horizonte.
- Promedios y Escaños enseñan las horquillas como «Rango 95%»: banda del 95 % sobre el hemiciclo (`rng_cell`),
  marca en el escenario central, línea discontinua en la mayoría absoluta y el texto `lo–hi` (`fmt_range`).

## Decisiones (mías, con su razón)

1. **Rango 95 %**, como en el resto de la web; los demás niveles del backtest (50, 80) no se enseñan.
2. **Horquilla de los bloques desde las simulaciones**: el cuantil de la suma de escaños de los partidos del
   bloque, simulación a simulación (`now['dist']`, y `fwd['dist']` para `_h`). Sumar las horquillas de los
   partidos sería incorrecto (los cuantiles no se suman; la suma ignora la correlación). Esto cambia
   `run_case` y obliga a repetir el backtest y republicar los eventos para que las horquillas de bloques
   aparezcan; hasta entonces esa celda muestra «–» (una parte antigua no las trae).
3. **Misma regla que los escaños**: la horquilla es la de la ejecución a horizonte (`lo95_h`/`hi95_h`) cuando la
   fila la trae; si no, la del nowcast.
4. **Columna «Rango 95%» tras «Escaños reales»** en las dos tablas (en bloques, antes de «Prob. de mayoría»),
   con el mismo marcado que Promedios: banda, marca (escaños del modelo), línea de mayoría y texto. Con
   `seats_valid` falso, raya como las demás celdas de escaños.
5. «Cómo leerlo» explica la columna en una frase.

## No objetivos

- No se tocan las filas fijas de «Todas las encuestas» ni la curva.
- No se cambia la métrica de cobertura del backtest (`cov_seats95` ya existe por partido).

## Partes

- `mtpy/lib/backtest.py`: en el bucle de bloques de `run_case`, los cuantiles de la suma de escaños del bloque
  (`_block_quantiles`, helper puro) sin sufijo y con `_h`. README del backtest: `blocks.csv` lleva intervalos.
- `mtpy/lib/pages/elecciones.py`: `horizon_tables` añade `seats_lo` y `seats_hi` a las filas de partidos y de
  bloques.
- `web/templates/eleccion.html`: la columna nueva en las dos tablas y la frase de «Cómo leerlo».
- Tests: `test_backtest_unit.py` (helper), `test_pages.py` (`horizon_tables`, la página), fixture
  `tests/fixtures/bundle/event-model.json` y `synthetic_backtest` con los cuantiles de bloques.

## Comprobación

- Tests unitarios: helper `_block_quantiles`, `horizon_tables` con horquilla (ejecución a horizonte, nowcast,
  parte sin ella), la página (cabeceras, celda de rango de partidos y bloques, raya con escaños no válidos y con
  una parte sin horquilla de bloques); `tests/integration/test_backtest.py` comprueba las columnas nuevas de
  `blocks` contra la base.
- Chrome sin cabeza sobre el bundle de pruebas (envoltorio ASGI con `/dist/`): la tarjeta a 6 y a 30 días, sin
  errores de consola.

## Operación

1. HECHO 2026-10-24 (17:56-21:27): `python job.py backtest '{"scopes":"all"}'`, 18 ámbitos completados, sin fallos;
   `backtest/results` (y las carpetas por ámbito) llevan ya los intervalos de bloques.
2. HECHO 2026-10-24 (21:28-21:49): `python job.py publish '{"what":["event"],"scopes":"all","event_date":"all"}'`,
   282 elecciones publicadas, sin fallos; la API pública ya sirve `lo95`/`hi95` en los bloques.
3. Pendiente de Luis: commit de los resultados del backtest y del código, y desplegar la web (la plantilla nueva).
