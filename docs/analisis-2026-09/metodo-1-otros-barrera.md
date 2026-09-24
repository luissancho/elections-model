# Método 1: bloque "otros" explícito y barrera del 3 % (24-09-2026)

Cambio en `mtpy/lib/simulator.py`. Tests: `tests/test_simulator_unit.py` (barrera), `tests/integration/test_threshold_official.py` (260 repartos oficiales) y `tests/integration/test_model.py` (coherencia provincial en 2027).

## Problema

El Forecaster deja en la columna `'-'` el resto del promedio de encuestas (`100 − Σ partidos`), que por diseño se interpreta como voto a candidaturas minoritarias que no compiten por escaños. El Simulator, sin embargo, (a) descartaba de la elección anterior a todos los partidos fuera de `parties.event` y al voto en blanco, (b) usaba como base de votos la suma de los partidos incluidos en lugar de los votos válidos, y (c) renormalizaba a 100 las cuotas provinciales de los incluidos antes de D'Hondt. Como D'Hondt es invariante a la escala, el reparto entre los incluidos no cambiaba, pero las cuotas provinciales publicadas eran incoherentes con el pronóstico nacional (media ponderada ≠ nacional) y no era posible aplicar la barrera legal, que se define sobre votos válidos. Además no se aplicaba la barrera del art. 163.1.a de la LOREG.

## Datos

- `events_results.pct` está calculado sobre votos válidos (candidaturas + blanco): `pct = 100 · votos / events_data.votes` (`loader.py:172`). Σ `pct` por provincia ≈ 99,2; el resto es el voto en blanco (`events_data.blank`, 0,81 % nacional en 2023).
- El partido 0 (`'-'`) de `events_results` tiene 0 votos desde 1979: cada candidatura pequeña tiene id propio. "Otros" es, por tanto, todo partido con resultado previo que no está en `parties.event` ni es movido por el `smap`. Para 2027 sobre 2023: 2,46 % nacional (PACMA 0,69, CUP 0,40, FO 0,19, NC 0,18, EV 0,15…), pero Soria 19,6 %, Teruel 16,6 %, León 9,2 %, Las Palmas 9,2 %, Barcelona 5,0 %, Madrid 1,3 %.
- `events_data.votes`/`blank` llegan como int32: se convierten a float64 antes de operar (`100 · votos` desbordaba a nivel nacional).

## Fórmulas

Por región `r` (0 = nacional), con `prev` = elección anterior y `names` = partidos simulados:

- Cuota previa de otros: `prev_otros[r] = max(0, 100 − Σ_names prev_pct[r] − blank_pct[r])`, calculada después de aplicar las reglas del `smap`.
- Residuo nacional simulado: `vpred['-'] = max(0, 100 − Σ_names vpred)`; en modo determinista coincide con `'-'` del Forecaster sobre los partidos seleccionados; en modo aleatorio absorbe la desviación de los sorteos independientes.
- Otros netos de blanco: `otros_fc = max(0, vpred['-'] − blank_pct[0])`; swing proporcional como cualquier partido: `vpred_otros[r] = prev_otros[r] · otros_fc / prev_otros[0]` (si `prev_otros[0] = 0`, cuota plana).
- El voto en blanco se supone constante en su cuota previa por provincia (no es un partido y no sigue el residuo de las encuestas).
- Categoría `'-'` en `unit()` y `frame()` = otros + blanco. Las cuotas `vpred_pct` se renormalizan a 100 por región sobre todas las categorías: ahora solo corrigen las pequeñas incoherencias del swing, no el voto a otras candidaturas.
- Barrera: `alloc_seats(votos, escaños, valid_votes, threshold=3.0)` excluye del D'Hondt a las candidaturas con menos del 3 % de los votos válidos de la provincia (`Simulator(threshold=...)`; `None` la desactiva). `'-'` nunca recibe escaños.

## Validación

- Barrera + D'Hondt sobre los votos oficiales reproduce escaño a escaño los 260 repartos provinciales de 2015-12, 2016-06, 2019-04, 2019-11 y 2023-07. Sin barrera falla exactamente en uno: Barcelona 2019-04, donde Front Republicà (2,72 %) habría obtenido el escaño que fue a Unidas Podemos.
- 2027 (`drange=6`, `seed=42`, `max_fc=3`): Σ `prev_pct` = Σ `vpred_pct` = 100 en las 52 provincias; la media provincial ponderada por votos válidos reproduce la cuota nacional (PP 31,81 frente a 31,91; PSOE 27,13 frente a 27,10; VOX 17,25 frente a 17,41; SUMAR 6,10 frente a 6,07).

## Impacto en 2027

| Magnitud | Antes | Después |
|---|---|---|
| DH (PP / PSOE / VOX / SUMAR / UP) | 138 / 113 / 61 / 8 / 2 | 138 / 113 / 61 / 8 / 2 (invariancia de escala; la barrera no actúa en el escenario central) |
| MT n=200, medianas (PP / PSOE / VOX / SUMAR / ERC) | 137 / 112 / 62 / 9 / 8 | 137 / 112 / 62 / 8 / 9 |
| MT n=200, cuantiles 2,5-97,5 % (PP / UP / SALF) | 114-155 / 0-6 / 0-4 | 115-155 / 0-6 / 0-4 |
| Madrid, cuota PP (`vpred_pct`) | 39,14 (Σ sin definir: regionales NaN, otros ausentes) | 39,30 (Σ = 99,99 con `'-'` = 3,26) |
| Soria, cuota PP | 35,92 | 30,99 (`'-'` = 31,81; previo 20,12) |
| Repartos provinciales cambiados por la barrera (200 simulaciones × 52 provincias) | — | 64 de 10.400: SALF-Madrid 34, UP-Barcelona 18, UP-Madrid 13, SALF-Barcelona 4, SUMAR-Madrid 1 |

Lectura: en el escenario central nada cambia (como predice la invariancia de escala), pero en la cola de simulaciones en que UP o SALF rozan el 3 % en Madrid o Barcelona la barrera les quita el escaño que D'Hondt puro les daba, y eso es lo que mueve los cuantiles y las probabilidades de bloque.

## Limitaciones documentadas

- Una candidatura local fuera de `parties.event` que en la realidad obtendría escaño (Teruel Existe en 2019-11, por ejemplo) no se modela: forma parte de `'-'`. La solución es incluirla en `parties.event` del evento.
- Con subconjuntos de `names` (`run(names=[…])`), si en una provincia ningún partido del subconjunto alcanza el 3 % no se reparte ningún escaño allí y la suma nacional no llega a 350; `alloc_seats` no hace ningún reparto de reserva a propósito.
- El swing de "otros" escala su cuota previa por el cociente nacional, igual que un partido; en provincias con mucho voto local (Soria, Teruel) eso infla `'-'` y reduce las cuotas de los incluidos, sin efecto sobre su reparto salvo a través de la barrera.
