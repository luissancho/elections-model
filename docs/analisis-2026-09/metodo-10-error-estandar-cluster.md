# Método 10: error estándar del promedio con cluster por casa (28-09-2026)

Cambios: `Estimator.build_input(groups=...)`, `LeastSquaresEstimator(groups=..., cov_type='hac' | 'hc1' | 'cluster')` y `get_robust_cov`, `LocalKernelEstimator(groups=...)` (`mtpy/core/utils/stat.py`); `Forecaster.set_reg_params` (por defecto `cov_type='cluster'`) y `fit` (pasa `pollster_id` como grupo) (`mtpy/lib/forecaster.py`). Tests: `tests/test_stat.py` (M10, con statsmodels como oráculo), `tests/test_forecaster_unit.py`, `tests/integration/test_model.py`. Notebook `PollsSimulations` (banda del promedio).

## Problema

El error estándar del promedio local de encuestas (`err` en `fc_stat`, y con él la banda `cmin`/`cmax` que se dibuja y los `fc_lo`/`fc_hi` del backtest) era un Newey-West con un retardo **en orden de filas**. Las filas son encuestas, varias por día y con huecos irregulares, no pasos de tiempo, así que ese "retardo" no captura ninguna dependencia real. La que existe es **por casa**: los residuos de una misma encuestadora frente al promedio se parecen entre sí, porque cada casa tiene su método y su sesgo. El análisis de septiembre (M6 del informe, `forecaster.md` H4) lo midió sobre la serie bruta: error 0,67 con el Newey-West frente a 0,89-0,91 con cluster por encuestadora.

## Evidencia

Desde el método 6 el promedio se ajusta sobre las encuestas **corregidas de efectos de casa**, y eso cambia el cuadro. Medido en la ventana del `as_of` de 2027 (2026-09-21; 102 encuestas y 19 casas con peso en la ventana) y de 2023 (30 días antes; 118 encuestas, 18 casas):

| 2027 | Newey-West (antes) | HC1 | cluster por casa | cluster / antes | sd de la media por casa / sd residual |
|---|---|---|---|---|---|
| PP, corregida | 0,201 | 0,244 | 0,253 | 1,26 | 0,89 / 1,04 |
| PP, bruta | 0,571 | 0,666 | 0,746 | 1,31 | 2,39 / 2,88 |
| PSOE, corregida | 0,178 | 0,179 | 0,232 | 1,30 | 0,72 / 1,12 |
| VOX, corregida | 0,184 | 0,200 | 0,273 | 1,48 | 0,75 / 0,83 |
| SUMAR, corregida | 0,108 | 0,128 | 0,145 | 1,34 | 0,50 / 0,60 |
| 2023: PP / PSOE / VOX / SUMAR, corregidas | 0,250 / 0,280 / 0,130 / 0,408 | 0,217 / 0,261 / 0,153 / 0,347 | 0,236 / 0,320 / 0,183 / 0,320 | 0,94 / 1,14 / 1,41 / 0,78 | |

Tres lecturas:

1. **La corrección de casa hace la mayor parte del trabajo**: el residuo del PP baja de 2,9 a 1,0 puntos y el error de 0,57 a 0,20. Lo que queda de correlación intracasa es modesto pero positivo: el cluster supera al HC1 en 7 de 8 casos y al Newey-West en 6 de 8 (+26 % a +48 % en 2027).
2. **Bootstrap por casas con los efectos fijos** (remuestreo con reemplazo de las casas de la serie corregida, 100 réplicas, 2027): sd de la media 0,38 (PP), 0,22 (PSOE), 0,32 (VOX), 0,14 (SUMAR). El cluster (0,25 / 0,23 / 0,27 / 0,14) se acerca más que el Newey-West (0,20 / 0,18 / 0,18 / 0,11); en el PP se queda corto porque la ventana tiene solo **8,8 casas efectivas** (`1 / Σ cuota²` de peso local: Sigma Dos y 40dB 0,18 cada una, CIS y SocioMétrica 0,13) y el estimador cluster con pocos grupos sesga a la baja.
3. **Bootstrap con reestimación de los efectos de casa** (20 réplicas): sd 1,18 (PP), 0,98 (PSOE), 0,37 (VOX), 0,23 (SUMAR). No es lo que `err` debe medir. Como los efectos se centran (`center_effects`), el nivel del consenso es el de las casas presentes; remuestrear casas responde a "qué consenso habría con otras casas", que es el error del sector frente al resultado, y ese ya lo cubre el error histórico terminal `pct_err` (2,5 puntos para el PP a 6 días, método 3). Meterlo también en `err` lo contaría dos veces en la simulación.

Un cuarto término, la incertidumbre de los propios efectos de casa (`effect_err` ≈ 0,61 puntos para todas las casas del PP, dominado por el suelo `err_floor · prior_err` del método 6), daría `√(Σ cuota² · effect_err²)` = 0,24, pero el centrado lo anula en primer orden: el nivel no depende de la corrección, solo cuenta `Σ (cuota local − cuota del ciclo) · error`, del orden de 0,05-0,10. Se declara y no se modela.

## Método

`LeastSquaresEstimator` es una regresión ponderada con tamaño efectivo de Kish (`neff`, pesos reescalados a `neff`, `dof = neff − k`) y covarianza tipo sándwich `covb · Σ · covb` sobre las puntuaciones `s_i = (w · x · e)_i` del modelo blanqueado:

- `'hac'` (lo de antes): `Σ = Σ_i s_i s_iᵀ + Σ_l k_l (Γ_l + Γ_lᵀ)` con retardos en orden de filas y núcleo de Bartlett.
- `'hc1'`: `Σ = Σ_i s_i s_iᵀ`.
- `'cluster'`: `u_g = Σ_{i ∈ g} s_i` por grupo, `Σ = Σ_g u_g u_gᵀ`, con el factor CR1 `G/(G − 1) · (neff − 1)/neff`. Sumar las puntuaciones dentro de cada casa hace el error robusto a cualquier correlación entre las encuestas de esa casa, sin estimarla. La banda usa una t con `G − 1` grados de libertad (la convención habitual del cluster: con 19 casas, 2,10 en vez de 2,01), y con pocas casas se ensancha como debe. Sin grupos, o con uno solo en la ventana, cae al `'hc1'` con un aviso (una vez por ajuste).

Los tres llevan el factor de muestra pequeña `neff / dof` (`n/(n − k)` con pesos iguales), así que con pesos iguales `'hc1'` es el HC1 de statsmodels y `'cluster'` su `cov_type='cluster'`; los tests lo comprueban con esa referencia, y un Monte Carlo con efectos de grupo y pesos desiguales comprueba que el cluster recupera la dispersión real de la media (±15 %) y el HC1 no.

El promedio local (`LocalKernelEstimator`) recibe los grupos de todas las encuestas y cada ajuste de ventana los de las suyas (`get_local_estimator`). `Forecaster.set_reg_params` usa `cov_type='cluster'` por defecto y `fit` pasa el `pollster_id` de cada encuesta; `'hac'` y `'hc1'` siguen disponibles con `reg_params={'cov_type': ...}`. El estimador local del `Computer` (sesgo del sector por evento, cuyo `err` entra en el rating) y los estimadores de error y escaños (regresiones sin orden temporal, M14 del análisis) **no cambian**: siguen en Newey-West, y queda anotado.

## Efecto en 2027 (`as_of` 2026-09-21, con efectos de casa)

| Partido | promedio | encuestas / `neff` | error antes | error cluster | banda 95 % antes | banda 95 % ahora |
|---|---|---|---|---|---|---|
| PP | 31,7 | 102 / 49 | 0,20 | 0,25 | ±0,41 | ±0,53 |
| PSOE | 27,2 | 102 / 49 | 0,18 | 0,23 | ±0,36 | ±0,49 |
| VOX | 17,4 | 102 / 49 | 0,18 | 0,27 | ±0,37 | ±0,57 |
| SUMAR | 6,1 | 101 / 49 | 0,11 | 0,15 | ±0,22 | ±0,30 |
| UP | 2,9 | 106 / 53 | 0,11 | 0,13 | ±0,22 | ±0,28 |
| SALF | 2,0 | 96 / 46 | 0,07 | 0,08 | ±0,14 | ±0,17 |

La banda "ahora" combina el error cluster con la t de `G − 1` = 18 grados de libertad (factor 2,10 en vez de 2,01): se ensancha entre un 20 % (SALF) y un 55 % (VOX).

En la simulación `std_err = √(err² + pct_err² + deriva²)` con `pct_err` ≈ 2,6 para el PP: pasa de 2,65 a 2,65-2,66. **Los escaños y las probabilidades no cambian**; cambia la banda publicada del promedio, que era demasiado estrecha.

## Backtest

BACKTEST_SECTION

## Salvedades

- **Pocas casas efectivas** (≈ 9 en 2027): el cluster sesga a la baja (bootstrap por casas 0,38 frente a 0,25 para el PP). Una corrección de Bell-McCaffrey (CR2) o el bootstrap propio serían el paso siguiente si se quiere afinar; con menos de tres casas activas, el cluster es poco fiable y el `hc1` de respaldo también. `G` cuenta todas las casas con peso en la ventana (19), no las efectivas (9): el factor `G/(G − 1)` y los grados de libertad son los de la convención estándar, y la salvedad de las pocas casas efectivas queda declarada.
- **Término de dos etapas** (incertidumbre de los efectos de casa) no modelado: ≈ 0,05-0,10 puntos por el centrado.
- **La banda del promedio no es la banda del resultado**: mide dónde está el consenso de las casas que hay, no dónde estará el voto; el error del sector es `pct_err` y entra solo en la simulación. `cov_fc95` del backtest (cobertura de la banda frente al resultado) es por tanto una métrica baja por construcción, no una prueba de calibración.
- **`Computer` sigue en Newey-West** (rating y estimadores de error y escaños): coherencia pendiente, declarada.
