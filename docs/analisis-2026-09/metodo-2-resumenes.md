# Método 2: resúmenes de la simulación (24-09-2026)

Cambios en `mtpy/lib/simulator.py` (`totals`, `scenario`, `summary`, `probabilities`, `shares` y las funciones puras `round_proportional`, `central`, `describe_dist`, `summarize_seats`, `majority_probs`, `closest_simulation`) y en `mtpy/core/utils/dataviz.py` (`plot_kde_1d(ci_method=)`). Tests: `tests/test_simulator_unit.py` (funciones puras, sin base de datos) y `tests/integration/test_model.py`. Notebook `PollsSimulations.ipynb` actualizado (escenario mediano y sección "Resumen MT").

## Problema

La salida del modelo es una distribución (`results`, `dist()`), pero lo que llegaba al notebook eran tres resúmenes con problemas:

1. `totals()` truncaba las medianas y repartía la diferencia hasta 350 "por ciclos": `diff // n` escaños a **todos** los partidos y el resto por orden de desempate. En MT (n=200) la suma de medianas es 347 y los 3 escaños iban a PP, PSOE y VOX por el orden de `argsort`; en LS cada simulación suma 326 (el estimador lineal de escaños no está restringido) y `24 // 13 = 1` daba un escaño a los 13 partidos, incluidos UP, SALF y UPN con máximo 0 en todas las simulaciones. El autor entendía que expandía proporcionalmente; era uniforme.
2. `result()` tras un `run` aleatorio devolvía la simulación 0, que el notebook mostraba como "la tabla provincia × partido" del modo MT: un sorteo cualquiera.
3. No había intervalos ni probabilidades numéricas; el KDE dibujaba un intervalo paramétrico (`Stat.ci`: media ± z·desviación típica recortado al rango), que en distribuciones discretas y asimétricas (SALF: mediana 0, media 0,47, máximo 8) no coincide con los cuantiles empíricos.

## Qué se publica ahora

- **`totals(method='median'|'mean')`**: estadístico central por partido expandido proporcionalmente a los 350 escaños y redondeado por restos mayores (Hamilton): `escalado = v · 350 / Σv`, parte entera, y los escaños que faltan uno a uno por resto descendente (desempates: valor escalado descendente y posición). Un partido con 0 en todas las simulaciones se queda en 0 por construcción. Con `split=True` la media ya suma 350 y `method='mean'` es un redondeo exacto; en LS/MC la base (277-326) obliga a escalar mucho y el titular es solo indicativo.
- **`scenario()`**: índice de la simulación real más cercana a las medianas marginales (distancia L1; desempates por L2 y menor índice). `result(sim.scenario())` es la tabla provincia × partido publicable: cada fila es un reparto D'Hondt coherente y suma 350.
- **`summary(names=None, alpha=None, method='median')`**: por partido o por bloque (`names` = nombre de `bmap`, dict o lista, como en `plot_forecast_output`; debe ser una partición):

| Columna | Significado |
|---|---|
| `pct` | Estimación puntual del Forecaster (`forecast['mean']`) |
| `pct_mean`, `pct_lo`, `pct_hi` | Media y cuantiles empíricos (`alpha/2`, `1 − alpha/2`) de la cuota nacional simulada (`shares()`) |
| `seats` | Titular (`totals`); en bloques, suma del titular de sus partidos |
| `seats_mean`, `seats_median`, `seats_lo`, `seats_hi`, `seats_min`, `seats_max` | Distribución de escaños simulados |
| `p_seats` | Probabilidad de obtener al menos un escaño |
| `p_majority` | Probabilidad de mayoría absoluta (≥ 176) |
| `p_first` | Probabilidad de ser el primero (empates al primer partido del orden) |

- **`probabilities(groups, majority=None)`**: probabilidad de que cada bloque o coalición alcance `majority` (176 por defecto); `groups` es un nombre de `bmap` o un dict `{nombre: [partidos]}`, y las coaliciones pueden solaparse (cada una se suma por separado).
- `plot_dist_kde` dibuja ahora el intervalo empírico (`ci_method='quantile'`), el mismo de las tablas, y ordena las crestas por titular y media. `plot_kde_1d` conserva `ci_method='normal'` por defecto para otros usos.
- `alpha` del constructor (0,05) es el nivel de los intervalos de `summary`. Con `n_sim < 40` el intervalo del 95 % coincide con mínimo y máximo.

## Impacto en 2027 (`drange=6`, `seed=42`, `max_fc=3`)

| Modo | `totals()` antes | `totals()` ahora |
|---|---|---|
| LS (base 326) | PP 128, PSOE 107, VOX 63, SUMAR 12, UP 2, SALF 1, JxCat 6, ERC 11, EHB 7, PNV 5, BNG 4, CC 3, UPN 1 | PP 135, PSOE 113, VOX 66, SUMAR 11, **UP 0, SALF 0**, JxCat 4, ERC 10, EHB 5, PNV 3, BNG 2, CC 1, **UPN 0** |
| MC n=200 (base 277-377) | PP 128, PSOE 107, VOX 63, SUMAR 13, UP 1, SALF 1, … | PP 135, PSOE 112, VOX 66, SUMAR 12, UP 0, SALF 0, … |
| DH | 138 / 113 / 61 / 8 / 2 / 0 … | igual |
| MT n=200 (medianas, base 347) | 137 / 112 / 62 / 8 / 2 / 0 / 4 / 9 / 7 / 5 / 2 / 1 / 1 | igual (la expansión de 347 a 350 coincide con el reparto anterior) |
| MT n=200, `method='mean'` | — | PP 136, PSOE 112, VOX 61, SUMAR 9, UP 2, SALF 1, JxCat 4, ERC 9, EHB 7, PNV 5, BNG 2, CC 1, UPN 1 |

- `scenario()` en MT n=200: simulación 2, distancia L1 9 (4 simulaciones empatadas), fila nacional PP 139, PSOE 110, VOX 61, SUMAR 10, UP 2, SALF 0, JxCat 3, ERC 8, EHB 7, PNV 6, BNG 2, CC 1, UPN 1.
- `summary()` MT n=200 (cuantiles 2,5-97,5 %): PP 137 escaños (115-157, media 136,2), PSOE 112 (94-134), VOX 62 (42-81), SUMAR 8 (2-20,5), UP 2 (0-6,5; P(escaño) 0,68), SALF 0 (0-4; P(escaño) 0,32). P(PP primero) 0,92.
- Bloques (`vs`): Derecha 199 escaños (175,5-214), P(mayoría absoluta) 0,975; Izquierda 122 (104,5-146,5), P(mayoría) 0. Coaliciones: P(PP+VOX ≥ 176) 0,975; P(PP+VOX+UPN+CC ≥ 176) 0,985; P(PSOE+SUMAR+UP ≥ 176) 0.

## Limitaciones

- LS y MC son didácticos: sus simulaciones no suman 350 (el estimador lineal de escaños, M3), así que la expansión proporcional reparte entre 24 y 73 escaños que el modelo no asignó, hacia los partidos grandes.
- La expansión proporcional de las medianas mueve los escaños que faltan hacia los partidos con más escaños; es la regla deseada ("expansión proporcional") y está documentada.
- Con `run(names=[subconjunto])`, los bloques sin partidos presentes tienen fila NaN en `summary` y `probabilities`.
- `p_first` resuelve los empates a favor del primer partido en el orden de `names`.
- El intervalo del KDE cambia respecto a las figuras anteriores (empírico en lugar de gaussiano) y el orden de las crestas es determinista.
