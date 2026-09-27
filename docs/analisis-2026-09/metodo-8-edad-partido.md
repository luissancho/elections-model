# Método 8: deriva por edad del partido (26-09-2026)

Cambios: `DriftEstimator` con `k_young`, `age_max`, `multiplier` y el argumento `age` de `var`/`sigma`; `Computer.party_first_polls`, `party_ages` y la columna `age` en `get_drift_estimator`; `Forecaster.party_first_polls` (por partido, antes de agrupar en bloques); `Simulator.party_ages` (`sim.ages`) y su uso en `build_frame` y `fan` (`mtpy/lib/computer.py`, `mtpy/lib/forecaster.py`, `mtpy/lib/simulator.py`); `meta` del backtest con `drift_multiplier` y `young_parties`. Tests: `tests/test_simulator_unit.py` (M8), `tests/integration/test_model.py`, `tests/integration/test_backtest.py`. Notebook `PollsSimulations` (sección "Pronóstico a fecha"). Es la extensión que el método 5 dejó declarada.

## Problema

La deriva de la opinión (método 5) es un paseo aleatorio relativo, `σ_der = p·√(k·h)`, con una `k` común estimada como media geométrica entre partidos: la deriva del partido **establecido típico**. Al medirla aparecieron dos poblaciones: los consolidados derivan un 4-5 % de su cuota a 90 días y los que acaban de nacer varias veces más (UP y Cs en 2015, VOX en 2019, SUMAR y MP en 2023). Para esos partidos los intervalos a horizonte eran demasiado estrechos, y en el backtest son partidos principales, así que la subcobertura a 90-180 días tenía ahí una de sus dos causas. En 2027 afecta a SALF y a SUMAR.

## Evidencia

Deriva relativa (`rms / nivel`, tabla `drift`) por edad del partido en la elección de cada ciclo, contando la edad desde la primera encuesta que lo lista (23 partidos, 253 filas):

| Edad | n (90 d) | geométrica 90 d | geométrica 180 d | frente a ≥ 4 años (90 d / 180 d) |
|---|---|---|---|---|
| < 1 año | 5 | 0,121 | — | ×3,0 / — |
| 1-2 años | 2 | 0,179 | 0,492 | ×4,5 / ×7,9 |
| 2-4 años | 10 | 0,129 | 0,225 | ×3,2 / ×3,6 |
| 4-8 años | 13 | 0,044 | 0,068 | ×1,1 / ×1,1 |
| > 8 años | 58 | 0,036 | 0,057 | ×0,9 / ×0,9 |

Los casos jóvenes: UP en 2015 (1,7 años, 0,32), SUMAR en 2023 (0,3 años, 0,25), AMAIUR en 2011 (0,24), UPyD en 2008 (0,11), CDC en 2016 (0,10). El efecto es un **escalón**, no una pendiente: unas tres veces más hasta los cuatro años y nada después. Una curva exponencial ajustada a estos datos extrapola a ×9 en el nacimiento, fuera de lo observado; con 17 filas jóvenes a 90 días, lo honesto es un escalón con un solo parámetro.

## Método

- **Edad**: años desde la **primera encuesta que lista al partido** (`Computer.party_first_polls`, sobre las encuestas que entran en el modelo); para un partido nacido en el ciclo actual, su primera encuesta en la serie bruta del `Forecaster`, **por partido y antes de agrupar en bloques** (`Forecaster.party_first_polls`): SUMAR nace en 2023 aunque su bloque herede las encuestas de UP y MP desde 2019 (la revisión detectó que con los bloques salía con 3,6 años en 2023 en vez de 0,2). Se descartó usar el primer resultado oficial: adelantaría el nacimiento de partidos que existieron años de forma marginal (Cs tiene resultado desde 2008 con el 0,2 %, VOX desde 2015) cuando lo que deriva es la etiqueta desde que las encuestas empiezan a listarla. Con esta definición UPN tiene 3,7 años (las encuestas lo listan aparte desde 2023; antes iba dentro del PP y de Navarra Suma): etiqueta nueva a efectos del modelo, con una deriva de centésimas.
- **Dos constantes**: `DriftEstimator.fit` ajusta `k` sobre las filas de partidos con `age ≥ age_max = 4` años (o sin edad) y `k_young` sobre las de menos de 4, con la misma media geométrica ponderada; `multiplier = √(k_young / k)`, nunca por debajo de 1; con menos de `min_young = 6` filas jóvenes, 1 con aviso. Todo con los ciclos anteriores al evento: fuera de muestra en el backtest.
- **Aplicación**: `σ_der = multiplier · p · √(k·h)` para los partidos jóvenes, en el sorteo (`build_frame`) y en el abanico (`fan`). El nowcast (`h = 0`) no cambia.

Para 2027 (ciclos hasta 2023): `k` = 3,96·10⁻⁵, `k_young` = 18,3·10⁻⁵ (90 filas jóvenes a 60 días o más, contando todos los horizontes y todas las etiquetas, regionales incluidas; la tabla de arriba cuenta partido-elecciones a un solo horizonte), **multiplicador 2,15**. Un efecto colateral honesto: al separar las poblaciones, la `k` de los establecidos baja respecto a la del método 5 (5,0·10⁻⁵, que mezclaba jóvenes): la deriva relativa de PP, PSOE o VOX a un año pasa del 13 % al 11,6 % de su cuota.

## Efecto en 2027 (`as_of` 2026-09-14; edades: SALF 2,3, SUMAR 3,5, VOX 10,1, UP 12,4, PP 37,6)

Desviación típica del abanico (puntos) sin y con multiplicador:

| Partido | 0 | 90 | 180 | 342 (sin) | 342 (con) |
|---|---|---|---|---|---|
| PP | 2,65 | 3,26 | 3,77 | 4,54 | 4,54 |
| PSOE | 2,46 | 2,95 | 3,36 | 4,01 | 4,01 |
| VOX | 2,03 | 2,28 | 2,51 | 2,87 | 2,87 |
| SUMAR | 1,54 | 1,58 → 1,73 | 1,62 → 1,90 | 1,70 | 2,17 |
| SALF | 1,36 | 1,37 → 1,39 | 1,37 → 1,41 | 1,38 | 1,45 |

En modo `deadline` (n = 500) los escaños de SUMAR pasan de 8 (2-22) a 8 (0-27); SALF sigue en 0 (0-4) porque en un partido al 2 % el error de encuesta (1,36 puntos) domina sobre la deriva incluso triplicada; PP y la probabilidad de mayoría de la derecha no cambian (0,91).

## Backtest

Cinco elecciones, seis horizontes, 500 simulaciones por caso y modo, con la configuración por defecto (efectos de casa, rating 0,7/0,3, sorteo independiente). "Sin" es la ejecución de referencia anterior; "con" aplica el multiplicador, estimado con los ciclos anteriores a cada elección: 1,0 en 2015 (sin partidos jóvenes con datos suficientes antes de 2015), 2,9 en 2016, 2,8 en 2019-04, 2,4 en 2019-11 y 2,4 en 2023. Partidos jóvenes entre los simulados: Cs y UP en 2016, VOX en 2019, VOX y MP en 2019-11, SUMAR en 2023. El nowcast es idéntico por construcción; solo cambian las métricas a horizonte (`_h`):

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| Cobertura voto 50 % a horizonte: sin | 0,48 | 0,50 | 0,45 | 0,33 | 0,25 | 0,37 |
| con | 0,48 | 0,50 | 0,45 | 0,37 | 0,30 | 0,37 |
| Cobertura voto 95 % a horizonte (ambas) | 0,96 | 0,96 | 0,84 | 0,84 | 0,89 | 0,87 |
| Cobertura escaños 50 % a horizonte: sin | 0,62 | 0,59 | 0,42 | 0,31 | 0,32 | 0,35 |
| con | 0,62 | 0,59 | 0,42 | 0,36 | 0,33 | 0,35 |
| CRPS escaños a horizonte: sin | 4,56 | 5,28 | 5,94 | 8,32 | 9,03 | 9,06 |
| con | 4,56 | 5,26 | 5,90 | 8,24 | 9,02 | 9,07 |
| Brier mayoría `vs` a horizonte: sin | 0,041 | 0,043 | 0,076 | 0,099 | 0,111 | 0,107 |
| con | 0,041 | 0,044 | 0,078 | 0,104 | 0,114 | 0,105 |

Lectura: mejora modesta y coherente donde actúa (cobertura del 50 % de voto a 60-90 días 0,33 → 0,37 y 0,25 → 0,30; escaños a 60 días 0,31 → 0,36; CRPS de escaños algo menor a 14-90 días), sin coste apreciable (Brier ±0,005, dentro del ruido con cuatro elecciones). La cobertura del 95 % a 90-180 días no se mueve: los fallos que quedan fuera del intervalo (2019-11 a 90 días, 2019-04 a 180) son del PSOE, Cs o el PP, no de los partidos jóvenes, y en un partido pequeño el error de encuesta domina sobre la deriva aunque se multiplique. En 2015 el multiplicador es 1 porque no hay partidos jóvenes medidos antes de esa elección: el caso en que más falta hacía (UP y Cs) queda fuera de muestra por construcción.

**Decisión**: se mantiene activado (es lo que hace el estimador por defecto cuando la tabla tiene edades) y `backtest/results/` pasa a esta ejecución.

## Salvedades

- **Muestra joven pequeña** (17 filas a 90 días, 9 a 180) → escalón con un parámetro; `min_young` protege el ajuste.
- **`age_max = 4` fijo**, tomado del salto entre los tramos 2-4 y 4-8 años.
- **Coaliciones renombradas**: SUMAR hereda los votos de UP y MP (`smap`) pero su edad es la de la etiqueta SUMAR (2023). Es lo que quiere el modelo: la etiqueta nueva es la que deriva.
- **Disolución**: no es identificable ex ante y no se modela; el multiplicador cubre a los que nacen, no a los que desaparecen (Cs en 2019-11 sigue con la deriva de un establecido).
- **Partidos sin encuestas ni resultados previos** (SALF): la edad sale de la serie del ciclo actual.
