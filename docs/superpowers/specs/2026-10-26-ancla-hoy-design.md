# El promedio se lee hoy, nunca en el futuro (ancla `as_of`)

Fecha: 2026-10-26. Estado: aprobada por Luis («Sí, adelante») y CONSTRUIDA el 2026-10-26 en `dev`, sin commits (los hace él). Las constantes de regresión de `tests/integration/test_model.py` se re-fijaron contra la base del 26-10 (mediana 135/109/68/8, reparto central 134/111/66/9: un escaño cambia por leer hoy en vez de tres días por delante en esa fixture). Hay que republicar un run de las generales para que la marca «Estimación» caiga en el día del cálculo.

## Intención

La marca «Estimación» del gráfico de Promedios cae diez días después del último sondeo (el 5 de noviembre con un
sondeo del 26 de octubre), porque el modelo lee el promedio en `as_of = min(víspera de la veda, último sondeo +
max_fc)`. Luis: «eso no tiene sentido». El backtest congela y lee el día del corte, y los rellenos retroactivos
leen su propio día: producción es el único sitio que lee en el futuro, y lo que validó el backtest es la lectura
«a hoy». Decisión: el ancla nunca pasa de hoy.

## Decisiones

1. `Simulator(today=None)`: `_set_anchor` toma `as_of = min(limit_date, date_fit_last, today)` cuando no se da
   `as_of`; `today` es la fecha dada o la de hoy en Europe/Madrid. Un `as_of` explícito sigue mandando. Backtest y
   rellenos no cambian (su `limit_date` es anterior a hoy). `max_fc` se mantiene: la recta se ajusta igual, sólo
   se lee antes.
2. `publish_forecast(today=None)` pasa `today` al Simulator (el job ya lo tiene para la guardia de la LOREG y lo
   enhebra) y corta la parte `series` en `as_of` (`export_series(..., end=sim.as_of)`, como el backtest): la línea
   continua acaba en la marca y la discontinua arranca ahí.
3. Textos: /metodo/escanos («la elección simulada se celebra en la fecha de la estimación, el día del cálculo») y
   /metodo/promedio (la prolongación de diez días existe en el ajuste, pero el promedio publicado se lee el día del
   cálculo); contrato (`weights`/`meta.as_of`); nota de enmienda en `docs/analisis-2026-09/metodo-5-nowcast-horizonte.md`.
4. Efecto en cifras: décimas (PP 31,49 frente a 31,41 hoy); el horizonte crece de 24 a 34 días. Hay que republicar
   un run de las generales para verlo.

## Partes y comprobación

`mtpy/lib/simulator.py`, `mtpy/lib/publish/forecast.py`, `mtpy/jobs/Publish.py`, plantillas y docs de arriba;
tests: `test_simulator_unit.py` (ancla con `today`), `test_publish_unit.py` (`today` al Simulator, serie cortada),
`test_jobs_publish.py` (`today` enhebrado), `tests/integration/test_model.py` (regla nueva del ancla, contra la base),
`test_pages.py` (textos). Suite completa en verde.
