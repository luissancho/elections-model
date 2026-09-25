# Fase 0: higiene y arreglos verificados (24-09-2026)

Rama `fase-0-fixes`. Todos los cambios reproducen un fallo detectado en el análisis (`README.md` de esta carpeta) y llevan test. Suite: `python -m pytest` (21 tests; los de `tests/integration/` usan la base de datos local y se saltan si no está disponible). Los `FutureWarning` de pandas se tratan como errores en `pytest.ini` para detectar cuanto antes las APIs que desaparecen en pandas 3.

## Librería estadística (`mtpy/core/utils/stat.py`)

| Id | Arreglo |
|---|---|
| B1 | El ancho de banda calculado por método (`'isj'`, `'scott'`) se truncaba a 3-5 caracteres por el dtype `<U3` de `np.repeat`. Ahora se usa una lista. Con los sondeos del PP el ISJ pasa de 71,0 a 71,62 días |
| B11 | `Stat.winsorize('group')` mandaba los outliers bajos al máximo interior; ahora recorta a `[mín, máx]` interiores |
| B12 | `conf(n)`/`conf_mean(n)` con `n >= 1` devolvían 0,48σ y 1,69σ para 1 y 2 sigmas; ahora `n·σ` |
| B19 | `Stat.quantile` indexaba mal con NaN y `dropna=False` |
| B14 | El punto fijo del ISJ no se resolvía nunca con `n <= 50` (bracket `[0, 0]`); el suelo `2π·espaciado medio` se mantiene y queda documentado |
| B15 | `dof = neff − nº de coeficientes` también con `poly_deg > 1` y varias variables |
| B16 | R² ponderado de `LeastSquaresEstimator` (coincide con `statsmodels`) |
| B17 | Una ventana local sin observaciones suficientes deja NaN en ese punto en vez de abortar toda la predicción |
| B18 | `predict()` sin puntos no reconvierte fechas ya convertidas; `r2_score` no sobrescribe `result` |
| B13 | `KernelDensityEstimator.process_outliers`: `'remove'` lanzaba `IndexError` y `'group'` no hacía nada |

## Simulator

| Id | Arreglo |
|---|---|
| B4 | `reg_totals` y `prev_results` se indexan por `region_id` (código de provincia; 0 = total nacional) y las reglas `regions` del `smap` se comparan por ese código. `result()` y `unit()` siguen mostrando nombres. Tras aplicar una regla `regions`, la cuota nacional del partido se recompone como suma ponderada de las provincias conservadas, de modo que el swing proporcional reproduce el pronóstico nacional dentro de ellas |
| B3 | `run(names=[...])` con un subconjunto de partidos ya no lanza `KeyError` en eventos con `smap` (las reglas `sub` que necesitan partidos ausentes se omiten) |
| B5 | Aviso (con `verbose > 0`) para partidos con pronóstico pero sin resultado previo ni regla aplicable, que reciben 0 escaños |
| B20 | `standard_t` con `nobs − 2` acotado en 3 grados de libertad y sin NaN |
| — | La regla `sub` ya no divide por cero cuando el pronóstico del partido nuevo es 0 (misma álgebra: `prev_v · fc_k / (fc_k + fc_v)`) |
| B26 | `totals()` usa `iloc` (indexación posicional explícita) |

Datos corregidos en `files/params.json` (no versionado todavía; evento 2019-04-28): COMPROMIS `regions` `["3","13","46"]` → `["3","12","46"]` (Castellón es 12; 13 es Ciudad Real), NA+ `["34"]` → `["31"]` (Navarra es 31; 34 es Palencia), y JxCat hereda de `CDC` (nombre con el que figura el partido en los resultados de 2016) en lugar de `PDeCAT`. Efecto en 2019-04-28, modo DH: COMPROMIS 0 → 2 escaños (Alicante, Valencia), NA+ 0 → 2 (Navarra), JxCat 0 → 5 (reales: 1, 2, 7).

## Computer, Forecaster, datos

| Id | Arreglo |
|---|---|
| B9 | `merge_bmaps` ya no itera cadenas carácter a carácter, no falla si un evento no define el bmap y no muta `event_params` |
| B10 | `weight_rating` ausente se rellena con `quality` en la misma escala que el peso (`log1p(q/100)/log1p(media)`), no con 0-100 |
| B24 | `fit_forecast` omite con aviso los bloques sin sondeos; `build_blocks` no falla si el primer partido del bloque no aparece |
| B6 | El cargador de Wikipedia guarda el contexto (`exit`/`wban`) en `ctype`, que es la columna del modelo; antes se perdía y `drop_ctypes` no podía actuar sobre datos nuevos |
| B21 | `drop_contexts` filtra por `t.ctype` (la columna `context` no existe) |
| B7 | `save_model_data` ya no pone a NULL las columnas de todas las encuestas del evento antes del `upsert` |
| — | `path` por defecto `'.'` (relativo a la raíz del sistema de ficheros de la app, `files/`), en las tres clases y los cargadores |
| — | `num_events`, `num_polls`, `num_polls_w` a 0 de forma explícita para casas sin sondeos (antes dependía de `observed=False`) |

## Compatibilidad y entorno

- APIs eliminadas en NumPy 2 / pandas 3: `np.NaN` (notebook `PollstersEvent`), `np.in1d`, `Series.append`, `fillna(method=)`, `applymap`, `groupby(axis=1)`, `fillna` con downcasting en `format_data_bin`.
- Se retira el `warnings.filterwarnings('ignore')` global de `mtpy/mtpy.py`.
- `dataviz` importa `learning` de forma perezosa: importar el modelo ya no carga `xgboost` ni `mlxtend`.
- `Log` crea el directorio `log/` si no existe (`mtpy.run()` fallaba en un clon limpio).
- `requirements.txt` con las versiones con las que se ha probado (numpy 2.4.4, scipy 1.17.0, matplotlib 3.10.8, scikit-learn 1.8.0, statsmodels 0.14.6) y con `lxml`, `openpyxl` y `pytest`; `.env.example` con las variables necesarias.

## Impacto numérico (2027-08-22, `max_fc=3`, `seed=42`)

| Magnitud | Antes | Después |
|---|---|---|
| Ancho de banda ISJ (PP) | 71,0 días | 71,62 días |
| PP, media a 2026-09-14 | 31,915 (err 0,676) | 31,911 (err 0,674) |
| DH (PP / PSOE / VOX) | 139 / 112 / 61 | 138 / 113 / 61 |
| MT n=200, mediana (PP / PSOE / VOX / SUMAR) | 137 / 112 / 62 / 8 | sin cambios |
| MT n=200, cuantiles 2,5-97,5 % del PP | 114-155 | sin cambios |

## Pendiente (decisiones de método, fases siguientes)

composición y correlación entre partidos (M2), ruido provincial (M4), house effects (M5), `r2_score` de `LocalKernelEstimator` con índice temporal, y la clasificación de casas online (`mtype` sólo informado en 10 de 95).

## Reubicación de las entradas en `data/` (24-09-2026)

`files/` queda solo para lo generado en ejecución (`fc/`, `img/`) y sigue ignorado. Todo lo que el código lee pasa a `data/`, versionado: `params.json`, `es-*.csv`, `wikipedia/*.json` e `infoelectoral/es/*.{json,xlsx}` (1,3 MB; ver `data/README.md` para origen y licencias). `mtpy.run()` expone una segunda raíz de solo lectura, `app.data` (`app.datapath`, configurable con `DATA_PATH`), y las ocho lecturas de entradas (`lib/data.py`, `forecaster.py`, `simulator.py`, `loader.py`) la usan; las escrituras siguen en `app.fs`. El parámetro `path` de las clases del modelo afecta ahora solo a las salidas, así que los notebooks no cambian (salvo la lectura de `es-provinces.csv` en `PollsSimulations`). `.dockerignore` exceptúa `data/`. Nuevo `tests/test_data_files.py` comprueba que los códigos de `regions` del `smap` existen como provincias y que cada XLSX de Infoelectoral tiene su JSON de mapeo.


Hecho después de la fase 0: barrera del 3 % y bloque "otros" explícito (M1), ver `metodo-1-otros-barrera.md`; resúmenes de la simulación (B2), ver `metodo-2-resumenes.md`; backtest del modelo, ver `metodo-3-backtest.md`; sucesión de partidos en las encuestas, ver `metodo-4-sucesion-encuestas.md`; nowcast, deriva de la opinión y fecha incierta (M5), ver `metodo-5-nowcast-horizonte.md`; efectos de casa y rating con el error descompuesto (M6), ver `metodo-6-house-effects.md`.
