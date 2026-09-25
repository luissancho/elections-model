# Método 7: ruido conjunto entre partidos (25-09-2026)

Cambios: `Simulator(composition=None | 'auto' | r)` (independiente por defecto), `Simulator.equicorrelation`, `Simulator.draw_correlated_t`, sorteo conjunto en `build_frame`, `clip_rate()` (`mtpy/lib/simulator.py`); `Computer.composition_ratio` y `get_composition_ratio` (`mtpy/lib/computer.py`); `run_case(composition=...)`, columnas `composition_ratio`, `composition_rho` y `clip_rate` en `meta`, flag `--composition` (`mtpy/lib/backtest.py`, `backtest/run_backtest.py`). Tests: `tests/test_simulator_unit.py` (M7), `tests/test_computer_unit.py`, `tests/integration/test_model.py`, `tests/integration/test_backtest.py`. Notebook `PollsSimulations` (celda "Ruido conjunto").

## Problema

Cada simulación sorteaba la cuota nacional de cada partido **por separado**, `rand = pct + std_err · t(nobs − 2)`, y el residuo `'-'` (otras candidaturas y blanco) absorbía la diferencia hasta 100, recortado a 0 cuando la suma se pasaba; en ese caso `build_umat` renormalizaba las cuotas. Los intervalos por partido estaban bien calibrados (cobertura 0,96 al 95 % y 0,57 al 50 % a 6 días en el backtest), pero las **sumas** heredaban la varianza de independientes, que la historia no respalda, y el recorte del residuo era frecuente: en 2027, en uno de cada ocho sorteos.

## Evidencia

Tres fuentes, medidas en solo lectura:

| Fuente | Qué se mide | Resultado |
|---|---|---|
| Errores de las encuestas frente al resultado, últimos 90 días, cinco elecciones | correlación entre los errores de los partidos principales | casi todas negativas: media −0,16 a −0,46 dentro del mismo bloque, −0,08 a −0,23 entre bloques |
| Sesgo del sector por partido (`pollsters_parties.industry`), once elecciones | `r = Var(Σ error) / Σ Var(error)` | **0,20** (0,29 desde 2008): correlación equivalente entre pares ≈ −0,26 con cuatro partidos |
| Errores terminales del promedio (backtest, 6 días), cinco elecciones | desviación de la suma de cada bloque frente a la de independientes | Derecha 2,0 frente a 3,65; Izquierda 2,2 frente a 3,1 |
| Incrementos a 30 y 90 días de las series ajustadas | `r` de la deriva por ciclo; correlación dentro del bloque | 0,02-0,7 según el ciclo; PP-VOX +0,66 en 2023 y −0,69 en 2027 |

Una encuesta que sobreestima a un partido subestima a otros porque suma 100, y el promedio hereda esa estructura. Lo que **no** es estable es la correlación dentro de cada bloque: en 2023 PP y VOX subían juntos, en 2027 se transfieren voto. Por eso el modelo no impone estructura por bloques.

## Método

**Un parámetro.** `r`, la razón entre la varianza de la suma de los errores de los principales y la suma de sus varianzas (`Computer.composition_ratio`), estimada con el sesgo del sector por partido de las elecciones anteriores al evento (`skip=1`: fuera de muestra en el backtest), con `year_decay` y acotada a [0,05, 1]. Se estima sobre los partidos **nacionales** (`regional = 0`), la misma población sobre la que se impone la correlación. Para 2027, `r` = 0,225.

**Una correlación común.** Entre los partidos nacionales (`regional = 0`), `ρ = (r − 1) · Σσ² / ((Σσ)² − Σσ²)` (`Simulator.equicorrelation`), exacta para la varianza de la suma sean cuales sean las varianzas, con suelo `−1/(k − 1)` para que la matriz sea definida positiva. Con seis nacionales en 2027, ρ = −0,168. Los regionales se sortean independientes (sus errores son de décimas).

**Cópula gaussiana.** `Simulator.draw_correlated_t`: normales correladas por Cholesky, transformadas a la `t(nobs − 2)` de cada partido; las marginales son exactamente las de antes y la correlación se impone en las normales subyacentes. Con `composition=1` (o `None`) el sorteo es el anterior, bit a bit; la regresión numérica de referencia se conserva así.

**Una sola `r`** para todo el ruido (error de encuesta y deriva a horizonte): la de la deriva es inestable por ciclo y la del error terminal gobierna el nowcast. Se declara.

## Efecto en 2027 (MT, n = 10 000, semilla 42, efectos de casa)

| | independiente (`r` = 1) | conjunto (`r` = 0,225) |
|---|---|---|
| ρ | 0 | −0,168 |
| residuo recortado (`clip_rate`) | 0,126 | 0,011 |
| sd suma nacionales (independiente sería 4,9) | 4,80 | 2,36 |
| sd PP / PSOE / VOX | 2,64 / 2,45 / 2,04 | 2,68 / 2,49 / 2,06 |
| sd PP − PSOE | 3,62 | 3,95 |
| sd Derecha − Izquierda | 4,45 | 4,86 |
| escaños PP | 137 (114-156) | 138 (113-158) |
| P(mayoría absoluta Derecha) | 0,984 | 0,974 |
| P(PP primero) | 0,900 | 0,882 |

Las **marginales no cambian** (diferencias del 1-2 %, ruido de muestreo), como debe ser con la cópula. Lo que cambia es la estructura: la suma de los nacionales se contrae a `√r` (4,8 → 2,4) y el residuo casi nunca se recorta, y las **diferencias** entre partidos se ensanchan (PP − PSOE 3,6 → 4,0; Derecha − Izquierda 4,5 → 4,9), porque con correlación negativa cuando uno sube los demás bajan. Los escaños dependen de las diferencias, no de las sumas: por eso los intervalos de escaños se abren un poco y la probabilidad de mayoría absoluta de la derecha baja del 98 % al 97 %. El sorteo independiente era, en este sentido, demasiado seguro sobre quién queda por delante y demasiado inseguro sobre cuánto suman.

Dos precisiones técnicas: con seis partidos nacionales la correlación común es pequeña (−0,17) y la suma de un **par** (PP + VOX) apenas se contrae (un 7 %); lo que baja hasta `√r` es la suma de todos. Y bajo una cópula gaussiana la correlación de Pearson de las `t` con pocos grados de libertad queda atenuada (con `dof` = 3, la varianza de la suma es 0,32 en vez de 0,18 en términos de `r`); afecta solo a los partidos con muy pocas encuestas.

## Backtest

Cinco elecciones, seis horizontes, 500 simulaciones por caso y modo; efectos de casa y rating del método 6 en los tres casos. "Independiente" es la ejecución de referencia (`backtest/results/`); `auto` estima `r` con las elecciones anteriores a cada una (0,08-0,21 en los casos del backtest, más baja que el 0,225 de 2027); 0,5 es la sensibilidad.

| Horizonte (días) | 6 | 14 | 30 | 60 | 90 | 180 |
|---|---|---|---|---|---|---|
| MAE voto (las tres) | 1,63 | 1,91 | 2,15 | 2,81 | 2,79 | 2,99 |
| Cobertura voto 95 % (las tres) | 0,96 | 0,96 | 0,84 | 0,84 | 0,84 | 0,87 |
| Cobertura escaños 50 %: independiente | 0,62 | 0,59 | 0,42 | 0,31 | 0,22 | 0,35 |
| `auto` | 0,62 | 0,59 | 0,46 | 0,31 | 0,38 | 0,35 |
| Cobertura escaños 95 % (las tres) | 1,00 | 1,00 | 0,96 | 0,92 | 0,92 | 0,80 |
| CRPS escaños: independiente | 4,57 | 5,28 | 5,95 | 8,37 | 9,13 | 9,22 |
| `auto` | 4,67 | 5,39 | 6,02 | 8,40 | 9,02 | 8,99 |
| 0,5 | 4,66 | 5,39 | 6,02 | 8,40 | 9,03 | 8,99 |
| Brier mayoría `vs`: independiente | 0,041 | 0,043 | 0,077 | 0,097 | 0,110 | 0,103 |
| `auto` | 0,048 | 0,047 | 0,072 | 0,097 | 0,106 | 0,100 |
| 0,5 | 0,047 | 0,046 | 0,072 | 0,097 | 0,104 | 0,100 |
| log-score `vs`: independiente | 0,172 | 0,179 | 0,261 | 0,310 | 0,331 | 0,305 |
| `auto` | 0,193 | 0,195 | 0,250 | 0,314 | 0,328 | 0,303 |

Lectura: las métricas de voto son idénticas, como exige la construcción. En escaños y mayorías el resultado es **mixto**: a 6-14 días el sorteo conjunto es algo peor (CRPS +2 %, Brier +15 %, log-score +10 %: las probabilidades de mayoría se vuelven menos tajantes y en estas cuatro elecciones con escaños válidos la tajante acertó), a 30 días igual, y a 90-180 días algo mejor (CRPS −1 a −3 %, cobertura del 50 % de escaños 0,22 → 0,38 a 90 días). La sobrecobertura del 95 % a 6-14 días no se corrige: no venía de las sumas. Con cuatro elecciones, diferencias de esta magnitud no son distinguibles del ruido.

**Decisión**: el criterio fijado de antemano (no empeorar escaños ni mayorías) no se cumple a corto plazo, así que el sorteo conjunto queda **disponible pero desactivado por defecto** (`Simulator(composition='auto')`, `--composition auto`); `backtest/results/` conserva la ejecución independiente. Es la opción más fundada para las **sumas** (y para el residuo, que deja de recortarse) y para los horizontes largos; el nowcast, que es el titular, se queda con el sorteo calibrado. Si se quiere activar por defecto basta cambiar el valor de `composition` en el `Simulator` y en la CLI.

## Salvedades

- **`r` con once elecciones y cuatro partidos**: un solo parámetro, sensibilidad con 0,5 en el backtest.
- **Correlación común**: el modelo no distingue pares (PP-VOX frente a PP-PSOE) porque la evidencia por pares cambia de ciclo; una estructura por bloques con la correlación interna estimada sería ajustar ruido.
- **Regionales independientes**: sus errores son de décimas y no afectan a la suma.
- **Horizonte aleatorio**: `std_err` cambia por simulación con la deriva, así que `ρ` se recalcula en cada sorteo (barato).
- **El residuo `'-'` sigue siendo el resto**: con `r` ≈ 0,25 el recorte es raro (1 %), pero existe; un modelo composicional exacto (logístico-normal) lo eliminaría a costa de recalibrar las marginales.
