# Método 18: swing logístico en la proyección a las circunscripciones (10-11-2026)

Cambios:
- `utils.logistic_swing` (`mtpy/lib/utils.py`): la regla, función pura.
- `Simulator(swing='logistic')`, `Simulator.SWINGS` y `Simulator.swing_shares`; `build_umat` proyecta con ella a los partidos y a «otros» (`mtpy/lib/simulator.py`). `swing='proportional'` conserva la regla anterior.
- `SwingNoise.pair_residuals`, `Computer.get_swing_residuals(swing=...)` y `get_swing_noise(swing=...)`: las oscilaciones del método 9 se miden contra la regla que usa el simulador (`mtpy/lib/computer.py`).
- Backtest: `swing` en `run_case`, `run_backtest` y `backtest_scope` (`mtpy/lib/backtest.py`), en la pipeline y el job, y `--swing logistic|proportional` en `backtest/run_backtest.py`; se guarda en `run.json` y en la columna `swing` de `meta.csv`.
- Publicación: `DEFAULT_CORRECTORS['swing'] = 'logistic'`, que llega al `Simulator` y queda en `meta.correctors` de cada run (`mtpy/lib/publish/forecast.py`); `BACKTEST_RUN_DEFAULTS['swing'] = 'proportional'` para los `run.json` anteriores (`mtpy/lib/publish/backtest.py`).
- Textos: sección «El swing logístico», tabla de desviaciones y notas `swing` y `oscilaciones` de `/metodo/escanos` (`web/templates/metodo_escanos.html`); `docs/web/contrato.md` (`correctors.swing`).

Tests: `tests/test_utils.py` (sección «Swing logístico»: factor común sobre las odds, pesos, provincias sin base, una razón por partido, nunca pasa de 100, filas que no son distritos), `tests/test_simulator_unit.py` (`build_umat` con las dos reglas, valor de `swing` desconocido), `tests/test_computer_unit.py` (residuos contra cada regla), `tests/test_backtest_unit.py`, `tests/test_pipelines_backtest.py`, `tests/test_jobs_backtest.py` (la opción llega al `Simulator` y a `run.json`; valor inválido), `tests/test_publish_unit.py` (el run se construye con la logística, `correctors` puede pedir otra, un `run.json` sin `swing` es proporcional), `tests/test_pages.py` (frases y tabla nuevas de `/metodo/escanos`).

## Problema

`build_umat` llevaba el voto del conjunto a cada circunscripción con un swing proporcional: la cuota de cada partido en cada provincia era la de la elección anterior multiplicada por la razón entre su cuota sorteada en el conjunto y la anterior, y después cada provincia se reajustaba para sumar 100. Un partido que crece un 50 % en el conjunto crecía así un 50 % en todas partes, también donde ya tenía el 22 %. Con Vox del 12,4 % de 2023 al 19 % del promedio de noviembre de 2026 (razón 1,68 sobre su base, una vez apartada la parte de SALF), la regla lo ponía en el 33 % en Ceuta y en el 31,5 % en Murcia.

La pregunta venía de una revisión más amplia del módulo (10-11-2026): cuánto se pierde al traducir el voto del conjunto a provincias y de dónde sale esa pérdida.

## Evidencia

### Con el código del modelo

`Computer.get_swing_residuals` compara, para cada par de elecciones generales consecutivas, el voto real de cada partido en cada provincia con el que da la regla cuando se le entrega su resultado real en el conjunto. Con las elecciones anteriores a 2026 (trece pares, de 1982-1986 a 2019-2023; 2.415 celdas de partidos con más del 1 % del conjunto en las dos elecciones y al menos el 1 % previo en la provincia):

| | Proporcional | Logística |
|---|---|---|
| Error absoluto medio (puntos) | 2,05 | 1,88 |
| Raíz del error cuadrático medio | 3,18 | 2,84 |
| Desviación típica del residuo relativo | 0,184 | 0,179 |

La logística tiene menos error en los trece pares:

| Par (segunda elección) | 1986 | 1989 | 1993 | 1996 | 2000 | 2004 | 2008 | 2011 | 2015 | 2016 | 2019-04 | 2019-11 | 2023 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Proporcional | 3,01 | 1,85 | 2,35 | 1,87 | 2,27 | 2,67 | 2,22 | 2,23 | 2,21 | 0,90 | 1,84 | 1,04 | 3,18 |
| Logística | 2,85 | 1,84 | 1,97 | 1,79 | 2,06 | 2,30 | 2,15 | 2,12 | 2,12 | 0,85 | 1,66 | 0,99 | 2,75 |

La diferencia está en los niveles altos:

| Voto previsto en la provincia | 1-3 % | 3-10 % | 10-20 % | > 20 % |
|---|---|---|---|---|
| Celdas | 170 | 489 | 466 | 1.290 |
| Proporcional | 0,56 | 1,04 | 1,46 | 2,84 |
| Logística | 0,55 | 1,02 | 1,41 | 2,55 |

Estas cifras son de la regla sin el reajuste de cada provincia a 100, que es como se miden las oscilaciones del método 9.

### Con el reajuste a 100 y el reparto

El modelo no aplica la regla cruda: reajusta cada provincia a 100, y eso ya comprime algo los feudos (Vox en Murcia, 33,6 % con la proporcional cruda y 31,5 % tras el reajuste). La proporcional con reajuste es, de hecho, un paso del ajuste proporcional iterativo (IPF), la versión multinomial de un swing logístico. Para comparar como compara el modelo se hizo una réplica aparte (script desechable, no versionado): las quince parejas de elecciones generales de 1977 a 2023, los partidos con al menos el 1 % del conjunto en las dos elecciones proyectados con cada regla a partir de su resultado real en el conjunto, el resto de candidaturas fijadas a su resultado real, «otros» como una categoría más, reajuste a 100 y D'Hondt con la barrera del 3 %.

| Regla | Error por celda (puntos) | Escaños en partido y provincia equivocados | Escaños netos por partido |
|---|---|---|---|
| Proporcional con reajuste (la anterior) | 2,18 | 15,8 | 5,9 |
| IPF hasta converger | 2,16 | 15,7 | 5,5 |
| Logística por partido con reajuste (la nueva) | 2,01 | 14,9 | 5,1 |

La logística por partido mejora a la anterior en 14 de las 15 elecciones. Llevar la multinomial hasta el final (IPF) no aporta: lo que ayuda es la saturación de cada partido en sus feudos, no la coherencia de la tabla.

Otras alternativas medidas en la misma réplica, sin reajuste (proporcional 2,15 puntos): swing uniforme (sumar los mismos puntos en todas partes) 2,12; mitad proporcional y mitad uniforme 2,02; logística 1,99. Mezclar la geografía de la última elección con la de la anterior no mejora (2,10 con sólo la última, 2,09 con un 20 % de la anterior).

### Lo que la regla no arregla

En la misma réplica, si además del resultado del conjunto se conociera el de cada comunidad autónoma, el error por celda bajaría a 0,90 puntos, los escaños en provincia equivocada a 5,0 y los netos a 1,7. Dos tercios del error de la traducción a provincias es que las comunidades no se mueven como el conjunto (coherente con el 75 % de la varianza del método 9). La regla logística mejora la forma del swing, no ese problema.

## Método

**La regla** (`utils.logistic_swing`). Para cada partido, la razón `p / (100 − p)` de su cuota previa en cada circunscripción se multiplica por un mismo factor `k`, y se deshace la transformación. `k` es el que hace que el partido reúna en las circunscripciones, ponderadas por los votos válidos de la elección anterior, los mismos votos que le daba el swing proporcional (cuota previa por la razón entre su voto sorteado y su voto previo en el conjunto): el total del partido no cambia, sólo su reparto. Se resuelve por bisección sobre `ln k` (60 pasos entre −40 y 40, todos los partidos a la vez). Con cuotas pequeñas coincide con el proporcional; una cuota no pasa de 100. Donde el partido no tenía base sigue sin ella; un partido sin base en el conjunto queda sin proyección, como antes; el total del ámbito y la lista que vota todo el ámbito (la autonómica canaria) toman la razón del conjunto.

**En el simulador** (`Simulator.swing_shares`, llamado desde `build_umat`). Las reglas `smap` construyen la base como antes y la regla `geo` se aplica después del swing, también como antes. «Otros» (el residuo neto del voto en blanco) se mueve con la misma regla que los partidos. El reajuste de cada provincia a 100 y las oscilaciones del método 9 no cambian. La parte que hereda una regla `sub` con `keep` (Vox en Santa Cruz de Tenerife) se calcula con las razones del conjunto, exacta para el proporcional y una aproximación cercana para el logístico.

**Las oscilaciones** (`SwingNoise.pair_residuals`). El residuo de cada celda es `ln(real / previsto)` con el previsto de la regla en uso. Con la logística las curvas de 2026 pasan de comunidad `0,0095 + 0,158/L` y provincia `0,0047 + 0,035/L` a comunidad `0,0084 + 0,153/L` (desviación relativa 0,12 al 30 %, 0,15 al 10 %, 0,24 al 3 %) y provincia `0,0042 + 0,037/L` (0,07, 0,09 y 0,13): algo menos de ruido en los niveles altos, la misma parte de comunidad (tres cuartas partes). La tabla de `/metodo/escanos` (diez pares de 1993 a 2023, 1.877 celdas) queda en 31 %, 22 %, 12 % y 10 % de desviación relativa por tramo (antes 32, 22, 12 y 11) y 0,8, 1,3, 1,9 y 3,4 puntos (antes 3,7 en el tramo de más del 20 %).

## Efecto en 2026 (`es`, promedio del 10-11-2026, semilla 42)

El reparto central cambia en dos provincias:

| Provincia | Cambio | Voto central (proporcional → logística) |
|---|---|---|
| La Coruña | Vox 0 → 1, PSOE 3 → 2 | Vox 7,82 → 8,30; PSOE 24,59 → 24,26 |
| Álava | PSOE 1 → 2, PNV 1 → 0 | PNV 13,44 → 13,13; PSOE 26,48 → 26,37 |

En el total, Vox 68 → 69 y PNV 4 → 3; los demás, igual (PP 126, PSOE 112, FA 11, ERC 11, EHB 7, JxCat 5, AA 3, BNG 2, CC 1). Derecha 194 → 195.

| Vox, voto central | Base tras las reglas | Proporcional | Logística |
|---|---|---|---|
| Ceuta | 21,3 | 33,0 | 31,6 |
| Murcia | 20,0 | 31,5 | 30,4 |
| Almería | 19,5 | 29,4 | 28,4 |
| Toledo | 18,0 | 28,9 | 28,1 |
| Madrid | 12,9 | 21,7 | 21,7 |
| Barcelona | 6,9 | 11,8 | 12,3 |
| La Coruña | 4,7 | 7,8 | 8,3 |
| Guipúzcoa | 1,9 | 3,6 | 3,9 |

Tres celdas de partido y provincia cambian un punto o más (Vox en Ceuta, Murcia y Almería) y catorce, medio punto o más. El PNV, que cae en el conjunto (del 1,12 % al 0,82 %), pierde más donde es más débil: Álava 13,4 → 13,1, Vizcaya 22,4 → 22,6. Con 10.000 simulaciones emparejadas (mismas semillas; prueba previa con «otros» todavía proporcional), los rangos del 95 % son los mismos salvo un escaño en algún extremo, la mediana del PNV pasa de 4 a 3 y la probabilidad de mayoría absoluta de la Derecha del 96,5 % al 96,4 %.

## Backtest

Dos corridas de `es` con el modelo completo (el promedio de sondeos de cada momento, con su error), una por regla: cinco elecciones (2015 a 2023), horizontes de 6 y 30 días, 1.000 simulaciones, semilla 42 (`backtest/results_swing_proportional` y `backtest/results_swing_logistic`, carpetas locales sin versionar; `python backtest/run_backtest.py --scope es --horizons 6 30 [--swing proportional] --out ...`). El voto del conjunto es idéntico por construcción. Las métricas de escaños y de provincias, en los ocho casos con escaños válidos (2015 queda fuera, como en el método 9):

| Métrica | Proporcional | Logística | Diferencia | Casos mejor / peor |
|---|---|---|---|---|
| MAE de la cuota provincial (puntos) | 2,380 | 2,367 | −0,5 % | 5 / 3 |
| CRPS de los escaños provinciales | 0,1511 | 0,1495 | −1,1 % | 5 / 3 |
| MAE de los escaños provinciales | 0,2113 | 0,2100 | −0,6 % | 2 / 3 |
| Cobertura de la cuota provincial al 50 / 80 / 95 % | 0,535 / 0,839 / 0,932 | 0,535 / 0,831 / 0,934 | | |
| CRPS de los escaños del conjunto | 5,536 | 5,576 | +0,7 % | 5 / 3 |
| MAE de los escaños del conjunto (partidos principales) | 8,07 | 8,17 | +1,2 % | 2 / 2 |
| Brier de las mayorías absolutas de los bloques | 0,0575 | 0,0585 | +1,7 % | 3 / 5 |
| Cobertura de los escaños al 50 / 95 % | 0,535 / 0,948 | 0,510 / 0,948 | | |

Todas las diferencias quedan dentro del ±2 % que el proyecto trata como ruido (criterio 17). Las provincias mejoran unas décimas de punto porcentual y los escaños del conjunto empeoran otras tantas, con el caso de abril de 2019 (Vox desde una base del 0,2 %) como el que más pesa en contra (CRPS 5,73 → 5,91 a 6 días y 4,57 → 4,78 a 30). Entre las 57 celdas de provincia y partido donde las dos reglas difieren en un punto o más, la logística queda más cerca del resultado en 27.

La lectura: con el voto real del conjunto la logística es mejor de forma consistente (trece pares de trece), pero su ventaja, de 0,17 puntos por celda, es pequeña frente al error del promedio de sondeos (1,9 puntos de MAE del voto del conjunto en estos casos), y cuatro elecciones no bastan para verla en los escaños. El backtest no distingue las dos reglas.

## Decisión

Luis pidió el cambio el 10-11-2026 tras ver la medición con el voto real del conjunto y la prueba sobre el run del día; el backtest, hecho después, es neutro. La logística es la regla por defecto del `Simulator`, del backtest y de la publicación; la proporcional queda como opción (`swing='proportional'`, `--swing proportional`, `correctors={"swing": "proportional"}`) para comparar las dos, también con el resultado del 29-N.

## Salvedades

- **Sigue siendo un swing único para todo el ámbito.** La mejora es de un 8 % del error por celda y de menos de un escaño de los quince que se pierden en la traducción; el swing de cada comunidad, que es la parte grande, no se toca.
- **Partidos que caen.** La regla quita relativamente más donde el partido es más débil (el PNV en Álava). Es lo que ocurre de media en los datos (la celda `< 0,6` de la réplica: 1,19 puntos frente a 1,30), pero en un partido de pocas provincias decide escaños con diferencias de décimas.
- **La historia publicada.** Los runs anteriores al cambio se hicieron con la proporcional y no llevan `swing` en `meta.correctors`. Para una serie coherente hay que rellenar con `replace`.
- **Las reglas `sub` con `keep`** usan la razón del conjunto, que con la logística es una aproximación en cada provincia.
- **Coste.** La bisección añade unos milisegundos por simulación (en el run de `es` con 2.000 simulaciones, 146 s frente a 136 s en total).
