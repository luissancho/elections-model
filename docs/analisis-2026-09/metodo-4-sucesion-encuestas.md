# Método 4: sucesión de partidos en las encuestas (24-09-2026)

Cambios: `Simulator.poll_blocks` (`mtpy/lib/simulator.py`) y su uso al construir el `Forecaster` interno; reglas en `data/params.json` (2019-04-28: JxCat hereda también `PDeCAT`; 2016-06-26: UP hereda IU; 2015-12-20: CDC hereda CiU y EHB hereda AMAIUR). Tests: `tests/test_simulator_unit.py` (`poll_blocks`) y `tests/integration/test_model.py` (2023 a 90 días). Lo detectó el backtest (`metodo-3-backtest.md`, hallazgo 6).

## Problema

El `Simulator` construía su promedio con `bmap = parties.event`: cada partido del evento era su propio bloque y cualquier otra etiqueta de las encuestas iba al residuo `'-'`. Pero las encuestas de un ciclo cambian de etiquetas cuando los partidos se fusionan o se refundan: en el ciclo 2019-2023 hay 506 encuestas con `UP`, 419 con `MP` y 498 con `Cs` (hasta junio de 2023), y solo 168 con `SUMAR`; en 2016-2019, 145 con `PDeCAT` y ninguna con `CDC` (la etiqueta de los resultados); en 2016, 71 con `IU` antes de Unidos Podemos. El `smap` ya describía esas herencias, pero solo se aplicaba a los **resultados** de la elección anterior (reparto provincial), no a las **encuestas**. Consecuencia medida en el backtest: en 2023 a 60-90 días la suma de los partidos principales quedaba 9,7 puntos por debajo del resultado, con SUMAR muy infraestimado y un residuo `'-'` enorme.

## Solución

`poll_blocks(names, smap)` construye los bloques del promedio: cada partido simulado más los nombres de las reglas `agg` de su `smap` que no tengan restricción regional (las regionales, como COMPROMIS ← UP en Valencia o NA+ ← PP+Cs en Navarra, describen herencias de voto provincial y no deben aplicarse a encuestas nacionales). Las reglas `sub` y `split` no intervienen. Un partido fuente que también se simula conserva sus propias encuestas. Con esos bloques, `group_results` suma en cada encuesta los porcentajes del sucesor y de sus predecesores (`min_count=1`: basta con que aparezca uno), y el resultado de la elección anterior se agrega igual, como ya hacía el `smap`.

Los nombres de las reglas `agg` sirven así a la vez para resultados y para encuestas: un predecesor sin resultado previo (PDeCAT en 2016) no aporta nada al reparto provincial, y uno sin encuestas (MES, CHA) no aporta nada al promedio.

Cuando una encuesta lista a la vez al sucesor y a un predecesor (23 encuestas de abril a junio de 2023 con SUMAR y UP, cuando aún eran opciones separadas) se suman: es el tratamiento honesto de una coalición que luego concurrió unida, aunque la suma de las partes tienda a sobreestimar a la coalición.

## Efecto (casos medidos con 100 simulaciones antes de reejecutar el backtest completo)

| Caso | MAE cuotas antes → después | `bias_sum` antes → después | Detalle |
|---|---|---|---|
| 2023 a 90 días | 3,44 → 2,75 | −9,84 → −2,27 | SUMAR pasa de quedar en el residuo a 14,7 (oficial 12,3) |
| 2023 a 30 días | 2,34 → 2,56 | −3,22 → −2,34 | SUMAR 14,3 frente a 12,3: la suma de partes sobreestima a la coalición |
| 2016 a 90 días | 2,61 → 2,32 | −4,73 → +0,66 | UP incorpora las encuestas de IU |
| 2019-04 a 90 días | 2,88 → 2,88 | 4,38 → 4,38 | JxCat no es partido principal; su serie gana 145 encuestas de PDeCAT |

El pronóstico de 2027 no cambia: su `smap` solo tiene reglas `sub` (UP desde SUMAR, SALF desde VOX), que no afectan a las encuestas.

## Reglas añadidas a `data/params.json`

- 2019-04-28: `JxCat: agg [CDC, PDeCAT]`.
- 2016-06-26 (nueva entrada, solo `smap`; partidos y bloques siguen derivándose de los datos): `UP: agg [IU]`. IU obtuvo el 3,7 % en 2015 y concurrió dentro de Unidos Podemos en 2016: la regla afecta también al reparto provincial.
- 2015-12-20 (nueva entrada, solo `smap`): `CDC: agg [CiU]` y `EHB: agg [AMAIUR]` (sus antecesores en 2011); `UP` y `Cs` siguen sin regla (partidos nuevos), así que el caso sigue sin escaños válidos en el backtest.

## Pendiente

- Reglas `sub` para encuestas: cuando un partido se escinde (UP de SUMAR en 2027), las encuestas anteriores a la escisión solo listan al partido origen; hoy su serie no se reparte. Es el simétrico de este cambio y se puede resolver con la misma proporción que `build_umat` usa para los resultados.
- Comparación completa antes/después en `by_horizon.csv`: ver la tabla al final de este documento.

## Backtest completo antes y después (medias por horizonte, partidos principales)

| d | MAE cuotas antes → después | `bias_sum` antes → después | Cob. 95 % cuotas | MAE escaños antes → después | CRPS escaños | Brier bloque |
|---|---|---|---|---|---|---|
| 6 | 1,61 → 1,61 | 0,59 → 0,57 | 0,96 → 0,96 | 6,3 → 6,4 | 4,4 → 4,5 | 0,05 → 0,05 |
| 14 | 2,04 → 2,00 | 0,19 → 0,03 | 0,88 → 0,88 | 8,2 → 7,9 | 5,8 → 5,7 | 0,05 → 0,05 |
| 30 | 2,14 → 2,26 | −0,07 → 0,45 | 0,84 → 0,84 | 8,4 → 8,7 | 5,9 → 6,3 | 0,08 → 0,08 |
| 60 | 2,81 → 2,83 | −1,52 → 0,65 | 0,81 → 0,84 | 10,0 → 11,6 | 7,2 → 8,3 | 0,13 → 0,10 |
| 90 | 2,98 → 2,72 | −1,81 → 0,38 | 0,86 → 0,84 | 13,0 → 12,6 | 9,0 → 8,9 | 0,16 → 0,12 |
| 180 | 3,35 → 3,00 | −2,11 → −1,25 | 0,78 → 0,87 | 15,9 → 12,5 | 12,6 → 9,4 | 0,22 → 0,11 |

Lectura: el sesgo sistemático de la suma de los principales desaparece (de −1,5 a −2,1 puntos a 60-180 días a ±0,7), y a 90 y 180 días mejoran el error de cuotas, el de escaños, el CRPS y la probabilidad de bloque. A 30-60 días el error empeora ligeramente (+0,1 en cuotas, +1,6 escaños a 60 días) por dos casos: 2023 (SUMAR = suma de SUMAR, UP y MP, que sobreestima a la coalición: 14,7 frente a 12,3) y 2016 a 60 días (UP con IU). Es el coste de un tratamiento honesto de las coaliciones: la suma de las partes en las encuestas suele superar el resultado conjunto, y ese es un sesgo que un futuro ajuste de *house effects* o de coaliciones puede corregir con datos. Los resultados completos de esta ejecución están en `backtest/results/` (`run.json` identifica el commit); la tabla de `metodo-3-backtest.md` es la referencia anterior a este cambio.
