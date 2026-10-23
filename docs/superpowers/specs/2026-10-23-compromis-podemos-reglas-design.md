# Compromís aparte y Podemos sin Cataluña: reglas del 29-N

Fecha: 2026-10-23. Estado: aprobado por Luis el 2026-10-23; construido el mismo día (plan
`docs/superpowers/plans/2026-10-23-compromis-podemos-reglas.md`).

Comprobación del 2026-10-23 con una simulación determinista contra la base, antes de la recarga de Luis:
Podemos sin base ni escaños en las cuatro provincias catalanas; FA con la base entera de Sumar en Barcelona
(15,22) y 4 escaños allí; FA y Podemos sin base en Alicante, Castellón y Valencia; Compromís con base 16,9
en Valencia pero sin ajuste (un solo sondeo útil en la base), como preveía el diseño. Previsualización del
estado tras la recarga, inyectando en memoria las dos cifras del CSV: Compromís ajustado con tres sondeos
(promedio 0,88), 1 escaño en Alicante y 1 en Valencia, 0 en Castellón; FA 0 allí; Podemos 0 en Cataluña.

## Intención

Para las generales del 2026-11-29, Compromís concurre fuera del Frente Amplio (FA) en Alicante, Castellón
y Valencia (Compromís-Front Valencià, registrada el 16-10) y Podemos no lleva lista en Cataluña (va dentro
de ERC). El modelo debe:

- simular a Compromís como partido propio en sus tres provincias, con la base de Sumar de 2023 allí, y
  mantener a FA fuera de ellas;
- dejar de fabricarle a Podemos una base en las cuatro provincias catalanas sin vaciar la de FA;
- guardar de forma reproducible las correcciones manuales de los sondeos que dan a Compromís aparte en
  una nota y lo incluyen en la cifra de Sumar.

## Por qué ahora

Compromís ya tiene tres sondeos útiles con cifra propia (Celeste-Tel 16-21 sep 1,3; SocioMétrica 10-11
oct 0,9; Sigma Dos 1-14 oct 0,9), el mínimo que exige `Forecaster.fit`. Hoy solo Sigma Dos tiene la cifra
en la base: los otros dos la llevan en Wikipedia como nota al pie de la celda de Sumar («Includes separate
figures for Compromís, with 1.3% and 3 seats»), que el loader ignora, y sus valores de Sumar (6,4 y 5,8)
incluyen a Compromís. En Barcelona, la pasada del 8-10 daba a Podemos 3,7 % y un escaño con probabilidad
0,65, salido de la base de los Comuns, que es de FA.

## Objetivos

1. Compromís vuelve a `parties.event`, al bloque Regionalista y recupera la regla `agg SUMAR` en 3/12/46.
2. Podemos deja de heredar base de FA en 8/17/25/43.
3. Una regla `sub` con `regions` hereda solo en las provincias que quedan, y las filas nacionales del
   partido y de sus fuentes se recalculan desde las provincias.
4. Las correcciones manuales de sondeos viven en un CSV curado que el loader aplica en cada carga.

## No objetivos

- Leer las notas al pie de Wikipedia (Luis prefiere corregir a mano).
- Trasladar a ERC el voto catalán de Podemos: es del orden de 0,1-0,2 puntos nacionales y las reglas del
  `smap` mueven bases de 2023, no promedios.
- Corregir los sondeos que funden a Compromís en FA sin dar cifra (IMOP 13-16 oct, «Within FA»).
- Canarias (Podemos con NC, IU y Sumar en una sola candidatura): esperar a la proclamación del 2-11.
- Herencia «vs 2023» de las reglas regionales (Compromís mostrará «–»): queda para después.
- Un ajuste de respaldo para partidos con 1-2 sondeos.

## Lo que sería un fracaso

- FA gana escaños en Alicante, Castellón o Valencia, o Podemos en Cataluña.
- FA pierde en Cataluña la parte de la base de los Comuns que hoy se lleva Podemos (iría a «otros», que
  nunca gana escaños, y el escaño fantasma acabaría en el PSOE).
- Sumar y Compromís cuentan dos veces el mismo voto en Celeste-Tel o SocioMétrica.
- Las correcciones se pierden en la siguiente recarga con `overwrite`.

## Restricciones

- `WikipediaLoader.select_series(overwrite=False)` descarta los sondeos que ya están en la base: las
  correcciones solo tocan sondeos que se van a escribir. Con `overwrite` se borran y reescriben todos los
  del evento, y ahí deben reaplicarse solas.
- Precedente de ficheros curados: `data/results/{scope}/{event_date}-seats.csv`, leído con
  `self.app.data.read_csv` y validado con `ValueError` cuando no casa con la tabla.
- `Forecaster.fit` exige tres sondeos con valor y peso positivo; ElectoPanel (`online`) está en
  `DROP_MTYPES` y no cuenta.
- Tests sin base de datos (`bare_loader` en `tests/test_loader_unit.py`); `pytest.ini` convierte
  `FutureWarning` en error. Identificadores en inglés; textos públicos en español.
- Luis hace commits en paralelo en el mismo árbol: comprobar `git status` antes de cada commit.

## Alternativas consideradas

**Persistencia de las correcciones.**
- A. SQL a mano tras cada recarga. Nada que construir; se pierde con cada `overwrite` y no deja rastro.
- B. CSV curado aplicado por el loader. Una cifra se escribe una vez, sobrevive a las recargas y queda en
  git. **Elegida.**
- C. Leer la nota de Wikipedia. Descartada por Luis: prefiere el control manual.

**Podemos en Cataluña.**
- Solo quitarle la herencia en las cuatro provincias. **Elegida.**
- Además trasladar su voto catalán a ERC. Descartada: magnitud dentro del ruido y sin mecanismo.

**Cómo quitar la herencia.**
- Solo `regions` en params: el código hereda en todas las provincias y después pone a cero a Podemos, con
  lo que FA pierde igualmente esa parte. Descartada.
- Ajuste en `build_umat` para que la herencia regional se calcule solo donde el partido concurre.
  **Elegida.**

## Decisiones

### 1. `data/params.json`, `es/2026-11-29`

- `parties.event`: añadir `COMPROMIS` al final (tras `CC`). `parties.polls` ya lo tiene.
- `bmaps.blocks.Regionalista`: `["PNV", "CC", "BNG", "COMPROMIS"]`.
- `smap`:
  - `COMPROMIS`: `{"type": "agg", "names": ["SUMAR"], "regions": ["3", "12", "46"]}` (bloque de d27b1fc).
  - `UP`: `{"type": "sub", "names": ["FA"], "regions": {"exclude": ["8", "17", "25", "43"]}}`.
  - `FA`, `PP` y `SALF` sin cambios. FA conserva su segunda regla, la exclusión de 3/12/46.

### 2. CSV de correcciones de sondeos

- Ruta: `data/polls/{scope}/{event_date}.csv`, leído con `self.app.data`, como el precedente de resultados.
- Columnas: `pollster, date, sponsor, party, pct, seats, note`. `pollster`, `sponsor` y `party` son los
  nombres canónicos de las tablas (`pollster` como queda en `polls.pollster`); `date` es la fecha de fin del
  trabajo de campo (`polls.date`); `sponsor` vacío equivale a sin patrocinador; `seats` opcional (fija
  `seats`, `seats_min` y `seats_max`); `note` libre.
- Semántica: valores absolutos. `pct` vacío elimina la fila del partido en ese sondeo. Un partido que no
  tenía fila se crea. `polls.parties` se recalcula tras aplicar.
- Dónde: al final de `WikipediaLoader.build_series`, sobre `self.polls` y `self.results`, antes de
  `select_series`, así vale para los notebooks nacionales y para `load/run_load.py`.
- Errores: `party` o `pollster` que no existen en las tablas lanzan `ValueError` (errata del fichero). Una
  fila cuyo sondeo no está entre los leídos no es error: se acumula en `fixes_missing` y se imprime como
  los `*_missing` (informe de `run_load` incluido).
- Contenido inicial (cifras de la nota de Wikipedia; Luis las confirma):

  | pollster | date | sponsor | party | pct | seats | note |
  |---|---|---|---|---|---|---|
  | Celeste Tel | 2026-09-21 | Onda Cero | SUMAR | 5.1 | | Compromís aparte: Sumar 6,4 incluía 1,3 |
  | Celeste Tel | 2026-09-21 | Onda Cero | COMPROMIS | 1.3 | 3 | |
  | SocioMétrica | 2026-10-11 | El Español | SUMAR | 4.9 | | Compromís aparte: Sumar 5,8 incluía 0,9 |
  | SocioMétrica | 2026-10-11 | El Español | COMPROMIS | 0.9 | 3 | |

  Las filas de EM-Analytics (misma nota, 1,0-1,1) quedan a criterio de Luis: están fuera del promedio.

### 3. `Simulator.build_umat`: reglas regionales

- La aplicación de las reglas del `smap` sobre `prev_pcts` se extrae a una función pura (método estático,
  nombre en inglés) que recibe `prev_pcts`, `smap`, `frame`, `default_region` y los votos por región, y
  devuelve `prev_pcts`; `build_umat` la llama. Así se prueba sin base de datos.
- Con `regions`, la máscara de provincias (la fila nacional siempre dentro) se calcula antes de aplicar
  el tipo de regla.
- `sub` con `regions`: la parte heredada se calcula solo en las provincias de la máscara (cero fuera) y
  solo allí se resta a las fuentes. Después se recalculan desde las provincias, ponderando por votos, las
  filas nacionales del partido y de cada fuente.
- `agg` y `split` no cambian de comportamiento. La exclusión posterior de `regions` sigue aplicándose.
- Tests unitarios: `agg` nacional; `agg` regional (Compromís 3/12/46); `sub` nacional (como hoy); `sub`
  regional: FA conserva la base completa en las provincias excluidas, Podemos 0 allí, reparto normal fuera,
  filas nacionales iguales a la suma ponderada de las provinciales; `split` sin cambios.

### 4. Comprobación y secuencia de puesta en marcha

Código: params, CSV, `build_umat`, tests completos. Comprobación con una pasada corta de `Simulator.run`
sobre `es` 2026-11-29 tras cargar las correcciones: Compromís ajustado con tres sondeos y con escaños solo
en 3/12/46 (del orden de 2-3; Sumar tuvo 1, 0 y 3 en 2023), FA sin escaños allí, Podemos sin escaños en
8/17/25/43 y FA con la base entera de Sumar en Barcelona (15,22).

Secuencia de Luis, en este orden: recarga de los sondeos del ciclo con `overwrite` (las correcciones se
aplican solas), `PollsCompute`, pasada de comprobación, publicación.

## Queda al constructor

- Nombres de la función extraída y de los helpers del CSV; forma de los fixtures de los tests.
- Texto exacto de los avisos de `fixes_missing`.
- Dónde imprime el notebook el aviso (basta un `print` con `verbose` y el atributo en el loader).
