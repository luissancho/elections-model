# Método 9: oscilaciones autonómicas y provinciales en la proyección (27-09-2026)

Cambios: `SwingNoise` (`decompose`, `fit`, `wls_nonneg`), `Computer.get_swing_residuals` y `get_swing_noise` (`mtpy/lib/computer.py`); `Simulator(regional_noise=True)`, `rng_swing`, `region_groups`, `lognormal_shocks`, `apply_swing_noise`, `_add_swing_noise` en `build_umat`, `unit_summary` (`mtpy/lib/simulator.py`); evaluación provincial del backtest (`official_provinces`, `_provinces`, métricas `cov_prov_*`, `mae_prov_*`, `crps_prov_seats`, `n_prov_cells`, `n_prov_missing`, salida `provinces.csv`, flag `--no-regional-noise`). Tests: `tests/test_computer_unit.py` (M9), `tests/test_simulator_unit.py` (M9), `tests/integration/test_model.py`, `tests/integration/test_backtest.py`. Notebook `PollsSimulations` (sección "Provincias").

## Problema

`build_umat` proyectaba el sorteo nacional a las provincias con un swing proporcional: la cuota de cada partido en cada provincia era la de la elección anterior multiplicada por la razón entre su cuota nacional simulada y la anterior. Dado el sorteo nacional, todas las provincias se movían al unísono: la única incertidumbre provincial era la que se propagaba desde la nacional. Con eso, el modelo cubría poco en **cuotas por provincia** (tres elecciones a 6 días, 824 celdas provincia × partido con cuota ≥ 1 %: cobertura 0,35-0,50 al 50 % y 0,82-0,94 al 95 %; error absoluto medio 1,5-2,5 puntos; desviación simulada 2,0-2,9 frente a un RMSE real de 2,3-3,5). Los **escaños** por provincia, en cambio, ya cubrían de sobra (0,79-0,91 al 50 %): con 0-3 escaños por celda los intervalos discretos casi siempre contienen el resultado, así que la métrica que discrimina es la cuota.

## Evidencia: cómo se desvían las provincias del swing proporcional

Residuos `log(real / (anterior · razón nacional))` por provincia y partido entre elecciones consecutivas (1993-2023, 10 pares, 1877 celdas con cuota previa ≥ 1 %):

| Nivel previsto | n | sd relativa | sd en puntos |
|---|---|---|---|
| 1-3 % | 112 | 0,32 | 0,8 |
| 3-10 % | 354 | 0,22 | 1,3 |
| 10-20 % | 390 | 0,12 | 1,9 |
| > 20 % | 1021 | 0,11 | 3,7 |

Descomponiendo cada residuo en la media de su comunidad autónoma (para ese par y partido) y la desviación dentro de ella, el **86 % de la varianza es autonómica** en la descomposición directa. Esa cifra exagera la parte autonómica: en una comunidad de `k` provincias la desviación intraautonómica tiene varianza `σ_prov² · (1 − 1/k)` (cero en las nueve uniprovinciales) y la media autonómica arrastra `σ_prov² / k`; con las dos correcciones (ver Método) la parte autonómica queda en torno al **75 %**. Ajustando `var = a + b/nivel` por tramos, sin corregir: componente autonómica `0,0063 + 0,165/nivel` (sd relativa 0,11 al 30 %, 0,15 al 10 %, 0,25 al 3 %) y provincial `0,0020 + 0,013/nivel` (0,05-0,08). La componente autonómica es estable entre pares (sd 0,06-0,14 en los diez). Dentro de una provincia los residuos de los partidos nacionales están positivamente correlados (PSOE-VOX 0,44, Cs-VOX 0,55): donde los de ámbito estatal caen frente al swing es donde suben los autonómicos; el residuo `'-'` y las reglas `smap` ya recogen esa parte.

Lo que falta no es tanto "ruido provincial" como **oscilaciones autonómicas** frente al swing nacional: Andalucía, Cataluña o Galicia se mueven de forma distinta al conjunto, y las provincias de cada comunidad, juntas.

## Método

- **Estimación** (`SwingNoise.fit`): con las elecciones anteriores al evento (fuera de muestra en el backtest; para 2027, 13 pares y 2415 celdas), por tramos de nivel se calculan las varianzas de la desviación intraautonómica y de la media autonómica y se ajusta `a + b/nivel` a cada una por mínimos cuadrados ponderados (pesos √n) con `a, b ≥ 0` (`wls_nonneg`: si el ajuste libre da un coeficiente negativo se reajusta el otro bajo la restricción, no se recorta a posteriori). Dos correcciones que salieron en la revisión: la desviación dentro de una comunidad de `k` provincias se escala por `√(k / (k − 1))` y solo entran las comunidades con `k ≥ 2`; y a la varianza de la media autonómica se le resta `σ_prov² / k`, con la curva provincial ya ajustada. Además los partidos presentes en una sola comunidad (CDC, CiU, ERC, JxCat, PNV) quedan fuera del ajuste autonómico: su media autonómica es nula por construcción (la razón nacional es la suya) y deprimía la curva en los tramos del 10-40 %. Para 2027: autonómica `0,0095 + 0,158/nivel` (sd 0,12 al 30 %, 0,16 al 10 %, 0,25 al 3 %), provincial `0,0047 + 0,035/nivel` (0,08 al 30 %, 0,09 al 10 %, 0,13 al 3 %). La varianza total por celda es la misma que sin corregir (sd 0,14 al 30 %, 0,18 al 10 %, 0,28 al 3 %); lo que cambia es el reparto, 75 % autonómico en vez de 85 %, es decir, algo menos de correlación entre las provincias de una misma comunidad.
- **Aplicación** (`_add_swing_noise`, en cada simulación): sobre las cuotas provinciales del swing, ya renormalizadas a 100 con el residuo, cada partido recibe un choque multiplicativo por comunidad, `exp(ε)`, con `ε ~ N(−σ²/2, σ_reg(nivel en la comunidad)²)`, y otro por provincia con `σ_prov(nivel en la provincia)`; los choques se centran para que su multiplicador tenga media 1. Después (`apply_swing_noise`, dos pasadas) cada partido se reescala para que su media nacional ponderada por votos válidos sea la de la proyección, y cada provincia se renormaliza a 100 con el residuo. Los choques salen de un generador propio (`seed + 1`): **el sorteo nacional es idéntico con o sin ellos**, y las cuotas nacionales simuladas no cambian.
- **Salida**: `Simulator.unit_summary(provincia)` da, por partido, mediana e intervalo de cuota y de escaños y la probabilidad de escaño en esa provincia.

Dos errores que salieron en la verificación y quedaron corregidos: los choques sin centrar comprimían sistemáticamente a los partidos que absorben (media de `exp(ε)` mayor que 1), y aplicar los choques sobre las filas **sin** renormalizar forzaba a los partidos autonómicos a la media nacional del swing bruto, que en las provincias vascas suma menos de 100 (el PNV bajaba de 25,0 a 23,5 en Vizcaya). Con las filas normalizadas antes de los choques, las medianas provinciales se conservan. La revisión posterior añadió tres correcciones: el reparto sesgado entre varianza autonómica y provincial (arriba); la evaluación provincial del backtest, que contaba las celdas sin proyección como fallos de cobertura pero no en el error, e incluía 2015, donde UP y Cs no tienen geografía previa y la renormalización infla al PP y al PSOE 9-14 puntos en todas las provincias (la cobertura del 50 % a 6 días era 0,47 con 2015 y 0,57 sin él); y un `FutureWarning` de pandas en `pivot_table` sobre la columna categórica `party`, que hacía fallar los tests de integración cuando los avisos cuentan como errores.

## Efecto en 2027 (MT, n = 1000, semilla 42)

| | sin oscilaciones | con oscilaciones |
|---|---|---|
| Escaños PP / PSOE / VOX | 138 (116-156) / 112 (94-132) / 61 (43-80) | 136 (113-157) / 112 (92-132) / 62 (42-79) |
| Escaños ERC / PNV / EHB | 9 (8-12) / 5 (3-6) / 7 (6-9) | 9 (7-12) / 5 (3-6) / 7 (5-9) |
| P(mayoría absoluta Derecha) / P(PP primero) | 0,991 / 0,919 | 0,990 / 0,914 |
| Madrid, PP (IC 95 %) | 39,1 (34,2-44,2) | 39,0 (31,3-47,1) |
| Madrid, PSOE | 23,8 (19,8-27,8) | 23,8 (17,5-30,9) |
| Barcelona, PSOE / PP / ERC | 31,0 (24,5-37,2) / 13,4 (10,4-16,3) / 14,8 (12,3-17,2) | 30,6 (23,0-39,3) / 13,4 (8,8-18,7) / 14,8 (12,2-17,6) |
| Vizcaya, PNV / EHB / PSOE / PP | 25,0 (19,1-30,0) / 21,9 (18,1-25,9) / 24,5 (20,6-28,9) / 12,4 (10,2-14,6) | 24,9 (19,1-30,3) / 21,8 (17,2-26,7) / 24,2 (18,2-31,4) / 12,4 (8,4-17,6) |

Las medianas se conservan y los intervalos provinciales de los partidos nacionales se ensanchan de forma sustancial (el PP en Madrid pasa de ±5 a ±8 puntos; en Barcelona de ±3 a ±5), mientras que los de los partidos autonómicos, cuya oscilación de comunidad la deshace el reescalado a su media nacional, apenas cambian (ERC en Barcelona 12,3-17,2 → 12,2-17,6; PNV en Vizcaya 19,1-30,0 → 19,1-30,3). Los escaños totales se mueven poco: las oscilaciones autonómicas se compensan en buena parte entre comunidades. Con el estimador corregido en la revisión la tabla es la misma a una décima (la varianza total por celda no cambia; solo el reparto entre comunidad y provincia).

## Backtest

Cinco elecciones, seis horizontes, 500 simulaciones por caso, configuración por defecto (efectos de casa, deriva por edad, sorteo independiente); "sin" es el swing determinista, "con" añade las oscilaciones, estimadas con las elecciones anteriores a cada una (autonómica `a` 0,008-0,012, `b` 0,16-0,20). Métricas provinciales sobre las celdas provincia × partido con cuota oficial o simulada ≥ 1 % **que el modelo proyecta** (mediana finita; 260-278 por caso, 7831 filas en `provinces.csv`), en las cuatro elecciones con escaños válidos: 2015 queda fuera porque UP y Cs no tienen geografía previa y la proyección provincial entera carece de sentido (la renormalización infla al PP y al PSOE 9-14 puntos en todas las provincias). Las celdas sin proyección (`n_prov_missing`, 5-11 por caso: VOX en 18 provincias en 2019-04 y alguna de ERC, COMPROMÍS, NA+ y CUP) cuentan como cero escaños en las métricas de escaños y no entran en las de cuota. Las métricas nacionales de voto son idénticas por construcción.

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| Cobertura cuota provincial 50 %: sin | 0,41 | 0,39 | 0,37 | 0,34 | 0,34 | 0,34 |
| con | 0,58 | 0,53 | 0,52 | 0,47 | 0,46 | 0,44 |
| Cobertura cuota provincial 80 %: sin | 0,67 | 0,65 | 0,63 | 0,61 | 0,59 | 0,56 |
| con | 0,87 | 0,84 | 0,81 | 0,79 | 0,77 | 0,72 |
| Cobertura cuota provincial 95 %: sin | 0,86 | 0,85 | 0,83 | 0,80 | 0,77 | 0,72 |
| con | 0,94 | 0,93 | 0,91 | 0,89 | 0,87 | 0,86 |
| MAE cuota provincial (puntos): sin / con | 2,09 / 2,14 | 2,35 / 2,38 | 2,51 / 2,53 | 2,99 / 3,00 | 3,08 / 3,10 | 3,43 / 3,47 |
| CRPS escaños provinciales: sin / con | 0,136 / 0,140 | 0,148 / 0,150 | 0,159 / 0,159 | 0,206 / 0,202 | 0,216 / 0,209 | 0,229 / 0,223 |
| Cobertura escaños nacionales 50 %: sin | 0,62 | 0,59 | 0,42 | 0,31 | 0,22 | 0,35 |
| con | 0,55 | 0,53 | 0,46 | 0,36 | 0,27 | 0,25 |
| CRPS escaños nacionales: sin / con | 4,57 / 4,64 | 5,28 / 5,22 | 5,95 / 5,90 | 8,37 / 8,43 | 9,13 / 9,08 | 9,22 / 9,09 |
| Brier mayoría `vs`: sin / con | 0,041 / 0,033 | 0,043 / 0,033 | 0,077 / 0,052 | 0,097 / 0,084 | 0,110 / 0,097 | 0,103 / 0,090 |
| log-score `vs`: sin / con | 0,172 / 0,152 | 0,179 / 0,158 | 0,261 / 0,202 | 0,310 / 0,284 | 0,331 / 0,307 | 0,305 / 0,281 |

Lecturas:

1. **Las cuotas provinciales pasan de cubrir de menos a cubrir bien en las colas y algo de más en el centro**: cobertura del 50 % 0,41 → 0,58, del 80 % 0,67 → 0,87 y del 95 % 0,86 → 0,94 a 6 días, y lo mismo en todos los horizontes, sin cambiar el error (MAE 2,09 → 2,14: las medianas se conservan). Por elección, la del 95 % a 6 días es 1,00 (2016), 0,93 (2019-04), 0,88 (2019-11) y 0,96 (2023). Por nivel oficial, la del 50 % es 0,80 en las celdas al 1-3 % y 0,51-0,60 en el resto, y la del 95 % 0,94-0,99: la curva común es demasiado ancha para los partidos pequeños (ahí mandan el suelo de nivel y la pendiente `b/nivel`) y algo ancha en el centro para los demás, con las colas bien. Afinar la curva en los niveles bajos, o usar colas más pesadas con menos varianza, queda como trabajo futuro.
2. **Las mayorías mejoran claramente**: Brier de la mayoría absoluta de los bloques 0,041 → 0,033 a 6 días, 0,077 → 0,052 a 30 y 0,103 → 0,090 a 180; log-score en la misma línea. Repartir el ruido entre comunidades en lugar de moverlas al unísono cambia qué combinaciones de escaños son probables, y lo hace en la dirección correcta.
3. Los **escaños nacionales** quedan igual o algo mejor (CRPS 5,28 → 5,22 a 14 días, 5,95 → 5,90 a 30 y 9,22 → 9,09 a 180, 4,57 → 4,64 a 6; MAE 6,60 → 6,44 a 6 días) y su cobertura del 50 % baja hacia el nominal donde sobraba (0,62 → 0,55 a 6 días) y sube donde faltaba (0,42 → 0,46 a 30, 0,22 → 0,27 a 90); a 180 días baja (0,35 → 0,25), con cuatro elecciones, dentro del ruido. Los escaños provinciales (CRPS 0,136 → 0,140 a 6 días, 0,229 → 0,223 a 180) no discriminan, como se anticipó.

Antes de la revisión, con 2015 y las celdas sin proyección dentro, las coberturas eran 0,35 → 0,47 (50 %) y 0,73 → 0,81 (95 %) a 6 días y el MAE 3,2 puntos: la falta de cobertura al 95 % que se leía entonces era en buena parte el caso huérfano.

**Decisión**: activado por defecto; `backtest/results/` pasa a esta ejecución (con `provinces.csv`).

## Salvedades

- **Una curva de nivel para todos los partidos**: los de implantación desigual (VOX en Murcia, PSOE en Andalucía) tienen residuos mayores que la curva; se acepta y se declara.
- **Correlación entre partidos dentro de la provincia** (los nacionales caen juntos donde suben los autonómicos): no se modela explícitamente; la renormalización provincial y el residuo la recogen en parte.
- **Comunidades uniprovinciales** (Madrid, Murcia, Asturias, Cantabria, La Rioja, Navarra, Baleares): la oscilación autonómica y la provincial coinciden y se suman, como en los datos con los que se estimaron.
- **Partidos autonómicos**: su oscilación de comunidad la deshace en gran parte el reescalado a su media nacional (concurren en una sola comunidad), así que su incertidumbre provincial viene sobre todo del sorteo nacional y del choque provincial; es coherente con que su cuota nacional simulada no cambie.
- **Renormalización iterativa**: dos pasadas no son exactas; el error sobre la media nacional de cada partido es del orden de 0,05 puntos (0,06 como máximo en las simulaciones del test, con tolerancia 0,1); una tercera pasada lo bajaría a 0,04 a cambio de un tercio más de coste en ese paso.
- **Mediana frente a media**: los choques se centran para que su multiplicador tenga media 1, así que su mediana es `exp(−σ²/2)`: en un partido al 1-2 % (σ ≈ 0,4-0,5 en el suelo de nivel) la mediana provincial simulada queda un 5-7 % por debajo del swing determinista (una décima de punto); en los grandes, un 0,2-0,5 %. Se conserva exactamente la media nacional de cada partido; no se puede conservar a la vez la mediana.
- **Reparto autonómico/provincial**: corregido el sesgo de la descomposición (arriba), pero sigue habiendo una sola curva para todos los partidos y comunidades; en las uniprovinciales el reparto es irrelevante (se suman las dos).
- **Coste**: unas decenas de milisegundos por simulación.
