# Método 19: el censo de la convocatoria en el peso de las circunscripciones (21-11-2026)

Cambios:
- `data/es-districts.csv`: escaños de `es` para el 29-N como en la base y en el decreto de convocatoria (Madrid 38, Cádiz 8; antes 37 y 9, los de 2023).
- `data/census/es.csv`: censo electoral de las generales por provincia, de 1986 a la del 29-11-2026 (`date`, `region_id`, `residents`, `abroad`); ver `data/README.md`.
- `data.get_census` (`mtpy/lib/data.py`) y tres funciones puras en `mtpy/lib/utils.py`: `census_factors` (cambio del censo residente de cada circunscripción entre dos elecciones), `census_totals` (votos válidos anteriores reescalados) y `census_shares` (resultado anterior del conjunto recalculado con esos pesos).
- `Simulator(census=True)`, `Simulator.apply_census` y los atributos `census_applied` y `census_pcts`; `build_umat` parte de `census_pcts` cuando existe (`mtpy/lib/simulator.py`). Activo por defecto; `census=False` conserva los pesos anteriores.
- Backtest: `census` en `run_case`, `run_backtest` y `backtest_scope` (`mtpy/lib/backtest.py`), en la pipeline y el job, y `--no-census` en `backtest/run_backtest.py`; se guarda en `run.json` (lo pedido) y en la columna `census` de `meta.csv` (lo aplicado).
- Publicación: `DEFAULT_CORRECTORS['census'] = True`, que llega al `Simulator` y queda en `meta.correctors` de cada run (`mtpy/lib/publish/forecast.py`); `docs/web/contrato.md`.
- Base de datos: `elections.events_data.registered` de `es` 2026-11-29 (52 provincias y el total, 38.728.356), antes vacío. Ningún cálculo lo lee.

Tests: `tests/test_census.py` (las tres funciones, `apply_census` y `build_umat` con el censo, el fichero de `es`), `tests/test_backtest_unit.py`, `tests/test_pipelines_backtest.py` (la opción llega al runner y a `run.json`) y `tests/test_publish_unit.py` (activo salvo que `correctors` lo apague).

## Problema

El simulador pesa cada circunscripción con sus votos válidos de la elección anterior (`prev_totals['votes']`). Ese peso decide cuánto cuenta cada provincia cuando el swing lleva la cuota del conjunto a las provincias (la suma ponderada de las provincias tiene que dar la cuota del conjunto) y es la cifra de votos por circunscripción que se publica. Entre julio de 2023 y el 29-N el censo no ha cambiado igual en todas partes, y el modelo no lo sabía.

La Oficina del Censo Electoral publica el censo de consulta de cada convocatoria por circunscripción. El del 29-N (tabla 5, bajada el 21-11-2026):

| | 23-07-2023 | 29-11-2026 | Cambio |
|---|---|---|---|
| Residentes en España | 35.141.198 | 35.958.711 | +2,3 % |
| Residentes en el extranjero (CERA) | 2.328.260 | 2.769.645 | +19,0 % |
| Total | 37.469.458 | 38.728.356 | +3,4 % |

## Método

Con `census=True`, y sólo en un ámbito con el censo de las dos elecciones en `data/census/{scope}.csv`:

1. **Pesos.** Los votos válidos anteriores de cada circunscripción se multiplican por el cambio de su censo **residente** (`residents` de la convocatoria entre `residents` de la anterior). El total del ámbito crece como la suma de sus circunscripciones.
2. **Base del swing.** La cuota anterior de cada partido en el conjunto se recalcula como la media de sus cuotas provinciales con los pesos nuevos: el resultado que habría dado la elección anterior con el censo de hoy. La razón de cada partido en el conjunto (previsión entre cuota anterior) se mide contra esa base, de modo que la suma ponderada de las provincias sigue dando la previsión del conjunto. Sin este paso, un partido fuerte en las provincias que crecen recibiría en el conjunto más voto del que dicen los sondeos.
3. Lo demás no cambia: las reglas del `smap`, la barrera, el reparto D'Hondt y `prev_results`, que conserva el resultado oficial (es el que usan las líneas base del backtest).

Por qué el censo residente y no el total: el voto exterior participa muy poco (en 2023 votó el 9 % de su censo) y su censo ha crecido un 19 %. El censo total de Madrid crece un 6,0 % y el de Orense un 1,3 %; el residente, un 3,9 % y un −1,5 %. La diferencia entre las dos opciones es del tamaño del propio ajuste.

## Evidencia

Con las generales de 1986 a 2023 (doce pares de elecciones consecutivas con censo por provincia), sin sondeos de por medio:

**El peso de cada provincia.** Se compara la parte de los votos válidos del conjunto que tuvo cada provincia en la elección nueva con la que tenía en la anterior, tal cual y reescalada por el censo.

| | Votos anteriores | Con el censo |
|---|---|---|
| Error relativo medio del peso de una provincia | 3,02 % | 2,63 % |
| Voto del conjunto mal situado (puntos) | 1,30 | 1,16 |
| Pares en que queda más cerca | 3 | 9 |

El censo mejora el peso en nueve de doce pares. Empeora en 1989, en 2000 y en 2023, el último: el cambio de la participación entre provincias pesa tanto como el del censo.

**El voto por provincia.** Con el resultado real del conjunto y el swing logístico, error absoluto medio de la cuota de cada partido en cada provincia (2.253 celdas):

| | Votos anteriores | Con el censo |
|---|---|---|
| Error absoluto medio (puntos) | 1,812 | 1,812 |
| Raíz del error cuadrático medio | 2,744 | 2,746 |

No hay diferencia: el peso de la provincia mueve muy poco la cuota que el swing le asigna.

**El run del día.** Previsión de `es` del 21-11-2026 con y sin el censo, 300 simulaciones emparejadas (semilla 42): el peso de cada provincia en el conjunto cambia entre el −5,0 % de Zamora (Lugo −3,9 %, Orense −3,7 %) y el +3,2 % de Guadalajara (Toledo +2,3 %, Madrid +1,6 %); la media de escaños de cada partido se mueve como mucho 0,2 (Vox, −0,2); los rangos del 95 % son los mismos salvo un escaño en algún extremo; el reparto central cambia un escaño, en La Coruña, del PSOE al BNG. El BNG sube porque Galicia pesa menos en el conjunto: la misma cuota nacional es una cuota gallega algo mayor.

## Backtest

Dos corridas de `es` con el modelo completo (el promedio de sondeos de cada momento, con su error), con y sin el censo: cinco elecciones (2015 a 2023), horizontes de 6 y 30 días, 1.000 simulaciones, semilla 42 (`backtest/results_census_off` y `backtest/results_census_on`, carpetas locales sin versionar; `python backtest/run_backtest.py --scope es --horizons 6 30 [--no-census] --out ...`, con la opción de hoy). El voto del conjunto es idéntico por construcción. Las métricas de escaños y de provincias, sobre los ocho casos con escaños válidos (2015 no los tiene):

| Métrica | Votos anteriores | Con el censo | Diferencia | Casos mejor / peor |
|---|---|---|---|---|
| MAE de la cuota provincial (puntos) | 2,361 | 2,365 | +0,2 % | 3 / 5 |
| CRPS de los escaños provinciales | 0,1492 | 0,1497 | +0,3 % | 5 / 3 |
| MAE de los escaños provinciales | 0,2099 | 0,2125 | +1,3 % | 0 / 2 |
| Cobertura de la cuota provincial al 50 / 80 / 95 % | 0,535 / 0,831 / 0,935 | 0,534 / 0,832 / 0,934 | | |
| CRPS de los escaños del conjunto | 5,560 | 5,573 | +0,2 % | 5 / 3 |
| MAE de los escaños del conjunto (partidos principales) | 8,17 | 8,19 | +0,3 % | 1 / 1 |
| Brier de las mayorías absolutas de los bloques | 0,0587 | 0,0585 | −0,4 % | 5 / 1 |
| Cobertura de los escaños al 50 / 95 % | 0,510 / 0,948 | 0,510 / 0,948 | | |

Todas las diferencias quedan dentro del ±2 % que el proyecto trata como ruido (criterio 17). El único caso que se mueve algo es abril de 2019 (CRPS de escaños 5,91 → 5,99 a 6 días y 4,78 → 4,87 a 30); 2023 mejora unas centésimas a 30 días (7,56 → 7,52). El backtest no distingue las dos opciones.

## Decisión

Luis pidió el 21-11-2026 guardar el censo del 29-N y usarlo en el simulador, y lo activó ese mismo día tras ver la evidencia, que es neutra: el peso de las provincias mejora en nueve pares de doce, y ni el voto por provincia ni los escaños cambian fuera del ruido. El motivo no es el backtest sino que el dato es el oficial de esta convocatoria. Es el valor por defecto del `Simulator`, del backtest y de la publicación; `census=False`, `--no-census` y `correctors={"census": false}` conservan los pesos anteriores para comparar, también con el resultado del 29-N. La serie publicada se rehace con un relleno.

## Salvedades

- **Censo de consulta, no definitivo.** El del 29-N es anterior a las reclamaciones; en 2023 la diferencia con el definitivo fue de unos miles de electores. Cuando Infoelectoral publique el fichero de la elección, sus filas de `data/census/es.csv` se sustituyen.
- **El backtest usa el censo definitivo** de cada elección evaluada, que se conoce semanas antes de votar salvo ese margen.
- **Sólo `es`.** Las autonómicas no tienen fichero de censo y el interruptor no hace nada en ellas (`census_applied` queda en `False`, y la columna `census` de `meta.csv`).
- **La participación no se modela.** El peso sigue suponiendo que cada provincia vota en la misma proporción que la vez anterior; es lo que falló en los tres pares en que el censo empeora.
- **Las oscilaciones del método 9** (`SwingNoise`) se siguen midiendo con los votos anteriores sin reescalar.
- **El fichero se construyó a mano** (script suelto sobre los XLSX de Infoelectoral y la tabla del INE); no hay paso de `load` que lo rehaga.
- **La historia publicada.** Los runs anteriores al cambio no llevan `census` en `meta.correctors`; para una serie coherente hay que rellenar con `replace`.
- **El backtest publicado** (`backtest/results`) se hizo sin el censo; las diferencias medidas aquí están dentro del ruido, pero no coincide al decimal con el modelo en vivo hasta que se rehaga.
