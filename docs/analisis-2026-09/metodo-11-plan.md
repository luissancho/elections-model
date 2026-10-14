# Método 11: ámbitos autonómicos — plan de implementación

> **Para quien lo ejecute con agentes:** SUB-SKILL OBLIGATORIA: `superpowers:subagent-driven-development`
> (recomendada) o `superpowers:executing-plans`, tarea a tarea. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que el modelo cargue, puntúe y simule cualquier elección autonómica (`scope` `es-*`) igual que
la general, y que los errores autonómicos alimenten el ranking de casas con menor peso.

**Arquitectura:** enfoque B de la spec. Datos, pesos, errores y desviaciones siguen siendo por ámbito y
evento; sólo `compute_ratings` se hace transversal. Dos catálogos nuevos (`scopes`, `districts`) versionados
en `data/` dan a cada ámbito su umbral, su peso en el rating y sus circunscripciones. Un cargador nuevo lee
los resultados del mismo artículo de Wikipedia del que salen los sondeos.

**Stack:** Python 3.11, pandas 2.2, numpy 2.4, lxml, requests, PostgreSQL local (esquema `elections`), pytest.

**Spec:** `docs/analisis-2026-09/metodo-11-ambitos-autonomicos.md` (decisiones D1-D6). Léela antes de empezar.

## Restricciones globales

- Códigos de ámbito ISO 3166-2 en minúsculas (`es-md`, `es-cl`…); `es` sigue siendo el nacional (D1).
- Federaciones regionales de PP, PSOE, VOX, UP, SUMAR y Cs usan la sigla nacional; las marcas propias son
  partidos aparte, con `parent_id` cuando lo tienen (D2).
- `rating_weight`: 1 para `es`, 0,5 para todo `es-*` (D3).
- `region_id`: 0 = total del ámbito; provincias = código INE (1-52) en cualquier ámbito; circunscripciones
  no provinciales ≥ 100 (Asturias 101-103, Baleares 101-104, Canarias 101-107 y lista autonómica 100,
  Murcia hasta 2015 101-105).
- Alcance temporal: elecciones autonómicas desde 2009 (D5).
- **Esquema de la base de datos:** los `ALTER TABLE` y `DROP TABLE` los hace Luis a mano; el código sólo crea
  tablas nuevas cuando faltan (`Model.create(replace=False)`, como `save_drift_data`). Las curaciones de
  `parties`, `pollsters.quality` y `sponsors` las aprueba Luis. Los pasos marcados **[Luis]** son puertas:
  no se sigue hasta que estén hechos.
- PEP8 y docstring (estilo numpy, en inglés) en toda función nueva; nombres de test en inglés con el
  comentario en español, como el resto de `tests/`.
- Tests unitarios sin base de datos (`tests/*.py`); los que la necesitan van en `tests/integration/` con
  `pytestmark = pytest.mark.integration` y la fixture `app`.
- Regresión de `es`: la suite actual (86 unitarios más integración) sigue en verde al cerrar cada fase; los
  números del ámbito `es` no cambian.
- Cada fase cierra con sus tests en verde y un commit en `dev`.

## Precisiones respecto a la spec

Decisiones que la spec deja abiertas o que la exploración del 01-10-2026 obliga a ajustar. Luis puede vetar
cualquiera antes de ejecutar.

1. La columna `group` de `districts` se llama **`reg_code`** (`group` es palabra reservada en SQL y
   `es-provinces.csv` ya usa ese nombre).
2. Los catálogos se **leen del CSV** versionado (`get_scopes`, `get_districts`); las tablas `scopes` y
   `districts` son una réplica que sincroniza `save_catalogues()`. Así los tests no necesitan la base.
3. `self.keys` del `Computer` **no cambia**; `event_scope` entra en el índice sólo del conjunto transversal
   que usa `compute_ratings`.
4. Umbral sólo autonómico (Comunidad Valenciana; Murcia hasta 2015): `threshold` nulo y `threshold_scope` 5.
   Regla única: una candidatura entra si supera el umbral de circunscripción **o** el del ámbito; un umbral
   nulo no se puede superar; con los dos nulos no hay barrera.
5. `Simulator(threshold=None)` pasa a significar "el del ámbito"; para desactivar la barrera, `threshold=0`.
6. La lista autonómica canaria (100) tiene su propio porcentaje en Wikipedia (fila *Regional* de la tabla
   por circunscripción): se usa ése, no el del total.
7. En `wp-urls.json` la clave `polls` es opcional en los `es-*`: un evento sin tabla de sondeos (Asturias
   2019) se carga sólo con resultados, porque hace falta como elección previa del siguiente.
8. Ruido M9 en ámbitos autonómicos: siempre la curva provincial estimada en `es`. El ajuste propio por
   comunidad queda fuera de alcance.
9. Los enlaces de Wikipedia llegan unas veces codificados (`M%C3%A1s_Madrid`) y otras no: se normalizan con
   `urllib.parse.unquote` tanto el enlace como las claves de `wp-maps.json`.
10. `WikipediaLoader` deja de depender de la fila de `events` (usa `scope` y `event_date` del constructor),
    para poder cargar sondeos en R1 antes de que R2 cree los eventos.

## Focos de revisión

Entradas que la spec implica y que más fácilmente rompen el programa. Cada una tiene su test en la tarea
indicada.

1. **Evento con resultado y sin sondeos** (Asturias 2019 como previa de 2023): `get_event_params`,
   `Computer` y `Simulator` no fallan y la elección sirve de base del swing. Tareas 2 y 9.
2. **Candidatura sin alias en el ámbito**: aparece en `parties_missing`; nunca se asigna a un homónimo
   nacional ni se pierde en silencio. Tarea 3.
3. **Celdas no numéricas en la tabla de sondeos** (`[f]`, `?`, `Tie`, `–`): la celda se ignora o la fila se
   descarta, sin excepción. Tarea 3.
4. **Circunscripción que no está en el catálogo** (falta el alias `Biscay`): el cargador de resultados
   falla con un error que nombra la circunscripción, en vez de perder sus escaños. Tarea 6.
5. **Casa que sólo publica en autonómicas** (Sondaxe, Ikertalde): recibe rating en las filas de `es` y
   `pollsters.rating` no se pone a cero al calcular los ratings de otro ámbito. Tarea 7.

## Mapa de ficheros

| Fichero | Cambio |
|---|---|
| `data/es-scopes.csv`, `data/es-districts.csv` | Nuevos catálogos |
| `data/es-regions.csv`, `data/es-regions-ages.csv` | Códigos MIR → INE |
| `data/params.json`, `data/wikipedia/wp-urls.json`, `wp-maps.json` | Entradas por ámbito |
| `mtpy/models/elections.py` | `Scopes`, `Districts`, `EventsData.estimated`; se elimina `Provinces` |
| `mtpy/lib/data.py` | `get_scopes`, `get_districts`, `save_catalogues`, `resolve_thresholds`, `get_thresholds`, `update_featured`; `get_event_params` y `get_event_dmat` tolerantes; `save_ratings_data(update_pollsters)` |
| `mtpy/lib/loader.py` | `WikipediaLoader` generalizado; `WikipediaResultsLoader` nuevo |
| `mtpy/lib/computer.py` | Ratings transversales; cuota de escaños; `regional = 0` en `es-*` |
| `mtpy/lib/simulator.py` | Umbral por ámbito y alternativo; distritos; ruido sólo provincial; caída al padre |
| `mtpy/lib/backtest.py`, `backtest/run_backtest.py` | Eventos por defecto y salida por ámbito |
| `load/run_load.py` | Nuevo: carga por lotes |
| `tests/test_loader_unit.py`, `tests/test_data_unit.py` | Nuevos |
| `tests/fixtures/wikipedia/*.html`, `build_fixtures.py` | **Ya creados** (instantánea del 01-10-2026, 1,8 MB, sin commit) |
| `tests/integration/test_regional_load.py`, `test_ratings_scopes.py`, `test_regional_model.py` | Nuevos |

---

## Fase R0 — Catálogos

### Task 1: catálogos en `data/` y tests de coherencia

**Ficheros:**
- Crear: `data/es-scopes.csv`, `data/es-districts.csv`
- Modificar: `data/es-regions.csv`, `data/es-regions-ages.csv`, `data/README.md`
- Test: `tests/test_data_files.py`

**Interfaces:**
- Produce `es-scopes.csv` con columnas `scode,name,ine_code,parent,demonym,threshold,threshold_scope,rating_weight,seats`.
- Produce `es-districts.csv` con columnas `scope,region_id,name,slug,ine_code,reg_code,population,seats,aliases`.
  `ine_code` = código INE de la provincia o vacío; `reg_code` = comunidad INE; `seats` = escaños actuales
  (vacío en circunscripciones extinguidas); `aliases` = nombres en inglés que usa Wikipedia, separados por `|`.

- [ ] **Paso 1: escribir los tests que fallan** en `tests/test_data_files.py`

```python
def test_scopes_catalogue_is_valid():
    scopes = pd.read_csv(os.path.join(DATA, 'es-scopes.csv')).set_index('scode')
    assert scopes.index.is_unique and scopes.shape[0] == 18
    assert scopes.index.str.fullmatch(r'es(-[a-z]{2})?').all()
    assert pd.isnull(scopes.loc['es', 'parent']) and scopes.loc['es', 'rating_weight'] == 1
    assert scopes.loc['es', 'threshold'] == 3
    regional = scopes.drop(index='es')
    assert (regional['parent'] == 'es').all() and (regional['rating_weight'] == 0.5).all()
    assert sorted(regional['ine_code']) == list(range(1, 18))
    thresholds = scopes[['threshold', 'threshold_scope']]
    assert thresholds.notnull().any(axis=1).all()
    assert thresholds.stack().between(0, 100, inclusive='right').all()
    assert scopes.loc['es-cn', ['threshold', 'threshold_scope']].tolist() == [15, 4]
    assert pd.isnull(scopes.loc['es-vc', 'threshold']) and scopes.loc['es-vc', 'threshold_scope'] == 5


def test_districts_catalogue_is_valid():
    scopes = pd.read_csv(os.path.join(DATA, 'es-scopes.csv')).set_index('scode')
    provinces = pd.read_csv(os.path.join(DATA, 'es-provinces.csv')).set_index('code')
    d = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))
    assert not d.duplicated(['scope', 'region_id']).any()
    assert set(d['scope']) == set(scopes.index) and (d['region_id'] > 0).all()
    assert d.groupby('scope').size().to_dict() == {
        'es': 52, 'es-an': 8, 'es-ar': 3, 'es-as': 3, 'es-ib': 4, 'es-cn': 8, 'es-cb': 1, 'es-cl': 9,
        'es-cm': 5, 'es-ct': 4, 'es-vc': 3, 'es-ex': 2, 'es-ga': 4, 'es-md': 1, 'es-mc': 6, 'es-nc': 1,
        'es-pv': 3, 'es-ri': 1
    }
    prov = d.loc[d['ine_code'].notnull()]
    assert (prov['region_id'] == prov['ine_code']).all()
    assert (prov['reg_code'] == prov['ine_code'].map(provinces['reg_code'])).all()
    other = d.loc[d['ine_code'].isnull()]
    assert (other['region_id'] >= 100).all() and set(other['scope']) == {'es-as', 'es-ib', 'es-cn', 'es-mc'}
    regional = d.loc[d['scope'] != 'es']
    assert (regional['reg_code'] == regional['scope'].map(scopes['ine_code'])).all()
    current = d.loc[d['seats'].notnull()].groupby('scope')['seats'].sum()
    assert (current == scopes['seats'].reindex(current.index)).all()
    assert (d.loc[d['region_id'] != 100, 'population'] > 0).all()


def test_regions_use_ine_codes():
    regions = pd.read_csv(os.path.join(DATA, 'es-regions.csv')).set_index('code')
    ages = pd.read_csv(os.path.join(DATA, 'es-regions-ages.csv'))
    provinces = pd.read_csv(os.path.join(DATA, 'es-provinces.csv'))
    assert regions.loc[[7, 8, 10, 13, 14], 'region'].tolist() == ['C. León', 'C. La Mancha', 'Valencia', 'Madrid', 'Murcia']
    assert set(provinces['reg_code']) <= set(regions.index)
    assert set(ages['code']) == set(regions.index)
    assert (ages.groupby('code')['region'].first() == regions['region']).all()
```

- [ ] **Paso 2: ejecutar y ver que fallan**

`python -m pytest tests/test_data_files.py -q` → 3 fallos (`FileNotFoundError` y la aserción de códigos).

- [ ] **Paso 3: crear `data/es-scopes.csv`** con el catálogo de la spec. Umbrales de partida (los `seats` se
  toman del artículo `Next_{gentilicio}_regional_election` de cada comunidad):

| scode | name | ine | demonym | threshold | threshold_scope |
|---|---|---|---|---|---|
| es | España | 0 | Spanish | 3 | |
| es-an | Andalucía | 1 | Andalusian | 3 | |
| es-ar | Aragón | 2 | Aragonese | 3 | |
| es-as | Asturias | 3 | Asturian | 3 | |
| es-ib | Baleares | 4 | Balearic | 5 | |
| es-cn | Canarias | 5 | Canarian | 15 | 4 |
| es-cb | Cantabria | 6 | Cantabrian | 5 | |
| es-cl | Castilla y León | 7 | Castilian-Leonese | 3 | |
| es-cm | Castilla-La Mancha | 8 | Castilian-Manchegan | 3 | |
| es-ct | Cataluña | 9 | Catalan | 3 | |
| es-vc | Comunidad Valenciana | 10 | Valencian | | 5 |
| es-ex | Extremadura | 11 | Extremaduran | 5 | 5 |
| es-ga | Galicia | 12 | Galician | 5 | |
| es-md | Madrid | 13 | Madrilenian | 5 | |
| es-mc | Murcia | 14 | Murcian | 3 | |
| es-nc | Navarra | 15 | Navarrese | 3 | |
| es-pv | País Vasco | 16 | Basque | 3 | |
| es-ri | La Rioja | 17 | Riojan | 5 | |

- [ ] **Paso 4: verificar cada umbral contra su ley electoral** (sección *Electoral system* del artículo
  `Next_…` y, si hay duda, la ley autonómica) y anotar en `data/README.md` una fila por comunidad con el
  artículo de la ley, el umbral y su base (votos válidos de la circunscripción o del conjunto). Corregir el
  CSV donde difiera, en particular Comunidad Valenciana (reforma de 2022) y Extremadura. Anotar también los
  umbrales históricos distintos del vigente desde 2009: son los overrides de la tarea 8.

- [ ] **Paso 5: crear `data/es-districts.csv`**. Filas de `es` y de los ámbitos provinciales: derivadas de
  `es-provinces.csv` (`region_id = ine_code = code`, `reg_code`, `population`); los `seats` de las filas de
  `es` salen de `events_data` del evento 2027-08-22 y los de los ámbitos provinciales del artículo `Next_…`.
  Filas no provinciales a mano:
  Asturias 101 Central, 102 Occidental, 103 Oriental (alias `Central`, `Western`, `Eastern`); Baleares 101
  Mallorca, 102 Menorca, 103 Ibiza, 104 Formentera; Canarias 100 Lista autonómica (alias `Regional`, sin
  población), 101 El Hierro, 102 Fuerteventura, 103 Gran Canaria, 104 La Gomera, 105 La Palma, 106
  Lanzarote, 107 Tenerife; Murcia 30 (actual, 45 escaños) y 101-105 (distritos hasta 2015, `seats` vacío,
  alias según el artículo de 2015). Alias de provincias con nombre inglés distinto: `Biscay`, `Gipuzkoa`,
  `Álava|Araba`, `Girona`, `Lleida`, `A Coruña`, `Ourense`, `Balearic Islands`, `Castellón`, `Alicante`,
  `Valencia`, `Seville`, `Navarre`, `Asturias`, `La Rioja`. Población de islas: padrón del INE; de las zonas
  asturianas y los distritos murcianos antiguos: censo electoral del resultado oficial más reciente. Fuente
  de cada cifra, en `data/README.md`.

- [ ] **Paso 6: recodificar `es-regions.csv` y `es-regions-ages.csv`** de la numeración del Ministerio del
  Interior a la del INE con el mapa antiguo → nuevo
  `{7: 8, 8: 7, 10: 11, 11: 12, 12: 13, 13: 15, 14: 16, 15: 14, 16: 17, 17: 10}` (1-6, 9, 18 y 19 no
  cambian) y ordenar por código.

- [ ] **Paso 7: documentar** en `data/README.md` los dos catálogos y la numeración única INE.

- [ ] **Paso 8: ejecutar** `python -m pytest tests/test_data_files.py -q` → todo en verde.

### Task 2: modelos, helpers de datos y parámetros por ámbito

**Ficheros:**
- Modificar: `mtpy/models/elections.py`, `mtpy/lib/data.py`, `data/wikipedia/wp-maps.json`
- Test: `tests/test_data_unit.py` (nuevo), `tests/test_data_files.py`, `tests/integration/test_scopes.py` (nuevo)

**Interfaces:**
- Consume: los CSV de la tarea 1.
- Produce, en `mtpy/models/elections.py`: `Scopes` (tabla `elections.scopes`, `key = ['scode']`, columnas del
  CSV) y `Districts` (tabla `elections.districts`, `key = ['scope', 'region_id']`, columnas del CSV). Se
  elimina la clase `Provinces`.
- Produce, en `mtpy/lib/data.py`:
  - `get_scopes() -> pd.DataFrame` — `es-scopes.csv` indexado por `scode`.
  - `get_districts(scope: Optional[str] = None) -> pd.DataFrame` — `es-districts.csv`, filtrado por ámbito.
  - `save_catalogues() -> dict[str, int]` — crea las dos tablas si faltan y hace upsert de los CSV; devuelve
    filas escritas por tabla.
  - `get_event_params` no falla si `params.json` no tiene el ámbito (`.get(scope, {})`).
  - `get_event_dmat` admite un evento sin sondeos: la fila `(0, 'result')` conserva los partidos del resultado.

- [ ] **Paso 1: tests que fallan**

`tests/test_data_unit.py`:

```python
"""Tests de `mtpy.lib.data` que no necesitan base de datos."""
import pandas as pd

from mtpy.lib.data import get_event_dmat


def test_event_dmat_without_polls_keeps_the_result():
    # Foco 1: una elección sin sondeos (Asturias 2019) sigue teniendo partidos
    parties = pd.DataFrame({'name': ['PSOE', 'PP', 'FAC']})
    index = pd.MultiIndex.from_arrays([[], [], []], names=['event_date', 'days', 'pollster'])
    polls = pd.DataFrame(columns=['PSOE', 'PP', 'FAC'], index=index, dtype=float)
    events = pd.DataFrame({'PSOE': [35.3], 'PP': [17.5], 'FAC': [6.5]}, index=pd.Index(['2019-05-26'], name='date'))
    df = get_event_dmat('es-as', '2019-05-26', polls, events, parties)
    assert df.loc[(0, 'result')].dropna().index.tolist() == ['PSOE', 'PP', 'FAC']
```

`tests/test_data_files.py`: sustituir `test_every_event_with_polls_url_has_wikipedia_maps` y ampliar
`test_params_regions_are_valid_province_codes` (los códigos válidos de cada ámbito salen de
`es-districts.csv`):

```python
def test_scoped_json_files_only_use_known_scopes():
    scopes = set(pd.read_csv(os.path.join(DATA, 'es-scopes.csv'))['scode'])
    params = json.load(open(os.path.join(DATA, 'params.json')))
    urls = json.load(open(os.path.join(DATA, 'wikipedia', 'wp-urls.json')))
    maps = json.load(open(os.path.join(DATA, 'wikipedia', 'wp-maps.json')))
    assert {'parties', 'pollsters', 'sponsors', 'scopes'} <= set(maps)
    assert set(params) <= scopes and set(urls) <= scopes and set(maps['scopes']) <= scopes
    assert all('results' in conf for events in urls.values() for conf in events.values())
    assert all('polls' in conf for conf in urls['es'].values())
    for events in urls.values():
        for event_date in events:
            pd.Timestamp(event_date)
```

`tests/integration/test_scopes.py`:

```python
def test_catalogues_are_synced_and_es_params_unchanged(app):
    from mtpy.lib.data import get_districts, get_event_params, get_scopes, save_catalogues
    from mtpy.models.elections import Districts, Scopes
    assert save_catalogues() == {'scopes': 18, 'districts': 118}
    assert Scopes().get_results(formatted=True).shape[0] == get_scopes().shape[0] == 18
    assert Districts().get_results(formatted=True).shape[0] == get_districts().shape[0] == 118
    assert get_event_params('es', '2027-08-22', path='.')['bmaps']['main'][:2] == ['PP', 'PSOE']
```

- [ ] **Paso 2: ejecutar y ver que fallan** — `python -m pytest tests/test_data_unit.py tests/test_data_files.py -q`
- [ ] **Paso 3: añadir `"scopes": {}`** a `wp-maps.json`.
- [ ] **Paso 4: implementar** los modelos y los helpers de la sección Interfaces. En `get_event_dmat`, sin
  sondeos del evento, `all_parties` son los del resultado y las filas `count`/`mean`/`std` quedan en NaN.
- [ ] **Paso 5: [Luis]** dar el visto bueno a que `save_catalogues()` cree `elections.scopes` y
  `elections.districts`, y borrar a mano `elections.provinces` (52 filas, ningún módulo la lee) cuando quiera.
- [ ] **Paso 6: ejecutar** `python -m pytest -q` y `python -m pytest tests/integration -q` → en verde.
- [ ] **Paso 7: commit**

```bash
git add data mtpy/models/elections.py mtpy/lib/data.py tests
git commit -m "Scopes and districts catalogues for regional elections"
```

---

## Fase R1 — Sondeos autonómicos

### Task 3: `WikipediaLoader` generalizado

**Ficheros:**
- Modificar: `mtpy/lib/loader.py` (clase `WikipediaLoader`)
- Test: `tests/test_loader_unit.py` (nuevo); fixtures en `tests/fixtures/wikipedia/` (ya creados)

**Interfaces:**
- Produce (métodos estáticos, sin base de datos):
  - `WikipediaLoader.find_poll_table(doc: html.HtmlElement) -> Optional[html.HtmlElement]` — primera
    `wikitable` cuya primera fila empieza por "Polling firm" y cuyo encabezado anterior más cercano
    (`h2`/`h3`/`h4`) contiene "Voting intention"; si no hay, la primera con esa fila cuyo encabezado más
    cercano es "Opinion polls".
  - `WikipediaLoader.read_header(table: html.HtmlElement) -> tuple[dict[int, str], Optional[int]]` —
    columnas de partido `{índice de columna: clave wiki sin codificar}` (los `th` de la primera fila con
    enlace que no sea `File:`; el índice cuenta los `colspan`) e índice del `th` cuyo texto es "Lead"
    (`None` si no está en esa fila; entonces vale la última celda de cada fila).
- Cambia:
  - `read_cols(self, cols, parties: dict[int, str], lead_col: Optional[int], year)` usa esos índices en vez
    de `cols[4:-1]` y `cols[-1]`.
  - `read_table(self, table, year)` llama a `read_header`.
  - `read_data`: con `years` vacío usa `find_poll_table`; con `years` conserva el recorrido actual por
    tablas y encabezado de año (páginas del ámbito `es`).
  - `get_colmap(key)`: alias globales más los de `maps['scopes'][self.scope][key]`, que prevalecen; claves
    pasadas por `unquote`. `parse_party` aplica `unquote`.
  - El constructor no exige fila en `events`; `build_series` usa `self.event_date` y `self.scope`. `polls`
    de `self.params` es opcional: sin ella `read_data` deja `self.data = []`.

- [ ] **Paso 1: tests que fallan** en `tests/test_loader_unit.py`. Un cargador sin base de datos se
  construye con `WikipediaLoader.__new__` (patrón de `tests/test_forecaster_unit.py:119`) fijando `scope`,
  `charsep`, `maps` y los mapas `*_colmap`, `*_idmap`, `*_missing`; los `idmap` de casas y patrocinadores
  aceptan cualquier nombre (subclase de `dict` con `__contains__` siempre cierto). Helpers del módulo:
  `load(page)` parsea `tests/fixtures/wikipedia/{page}.html` con `lxml.html.fromstring`;
  `bare_loader(scope, maps=None, party_ids=None)` devuelve ese cargador (sin `party_ids`, ningún partido
  está mapeado); `DATA` es la ruta de `data/`, como en `tests/test_data_files.py`.

```python
# (filas tr, columnas de partido, columna Lead, sondeos leídos): instantánea del 01-10-2026.
# Sondeos = filas que no están en negrita y cuya última celda es numérica.
CASES = {
    '2023_Madrilenian_regional_election': (59, 6, 10, 50),
    'Next_Madrilenian_regional_election': (26, 8, 12, 14),
    '2022_Castilian-Leonese_regional_election': (77, 11, 15, 65),
    '2019_Asturian_regional_election': (36, 9, 13, 25),      # tabla bajo el h2, sin h3
}


@pytest.mark.parametrize('page', CASES)
def test_poll_table_header_and_rows(page):
    n_tr, n_parties, lead_col, n_polls = CASES[page]
    table = WikipediaLoader.find_poll_table(load(page))
    assert len(table.xpath('.//tr')) == n_tr
    parties, lead = WikipediaLoader.read_header(table)
    assert (len(parties), lead) == (n_parties, lead_col)
    rows = bare_loader('es-md').read_table(table, page[:4] if page[0].isdigit() else '2026')
    assert len(rows) == n_polls
    assert not any('regional election' in row['pollster'] for row in rows)   # sin filas de resultado


def test_madrid_2023_header_keys_and_first_poll():
    table = WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election'))
    parties, _ = WikipediaLoader.read_header(table)
    assert parties == {
        4: "People's_Party_of_the_Community_of_Madrid", 5: 'Más_Madrid',
        6: "Spanish_Socialist_Workers'_Party_of_the_Community_of_Madrid", 7: 'Vox_(political_party)',
        8: 'Unidas_Podemos', 9: 'Citizens_(Spanish_political_party)'
    }
    first = bare_loader('es-md').read_table(table, '2023')[0]
    assert (first['pollster'], first['sponsor']) == ('GAD3', 'RTVE-FORTA')
    assert (first['start_date'], first['end_date']) == (pd.Timestamp('2023-05-12'), pd.Timestamp('2023-05-27'))
    pp = first['results'][0]
    assert (pp['pct'], pp['seats_min'], pp['seats_max']) == (49.5, 70, 72)


def test_next_madrid_reads_the_last_party_column():
    parties, lead = WikipediaLoader.read_header(WikipediaLoader.find_poll_table(load('Next_Madrilenian_regional_election')))
    assert parties[11] == 'Se_Acabó_La_Fiesta' and lead == 12


def test_non_numeric_cells_are_skipped():
    # Foco 3: '[f]', '?', '–' y 'Tie' no rompen la lectura ni dejan resultados sin porcentaje
    table = WikipediaLoader.find_poll_table(load('2022_Castilian-Leonese_regional_election'))
    rows = bare_loader('es-cl').read_table(table, '2022')
    assert all(pd.notnull(r['pct']) and r['pct'] > 0 for row in rows for r in row['results'])


def test_scope_alias_wins_and_unknown_party_is_reported():
    # Foco 2: el alias del ámbito prevalece; lo no mapeado se lista, no se asigna
    maps = {
        'parties': {'PP': ['People%27s_Party_(Spain)']}, 'pollsters': {}, 'sponsors': {},
        'scopes': {'es-md': {'parties': {
            'PP': ['People%27s_Party_of_the_Community_of_Madrid'], 'MM': ['M%C3%A1s_Madrid']
        }}}
    }
    loader = bare_loader('es-md', maps=maps, party_ids={'PP': 2, 'MM': 600})
    colmap = loader.get_colmap('parties')
    assert colmap["People's_Party_of_the_Community_of_Madrid"] == 'PP' and colmap['Más_Madrid'] == 'MM'
    assert colmap["People's_Party_(Spain)"] == 'PP'
    rows = loader.read_table(WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election')), '2023')
    assert {r['party'] for r in rows[0]['results'] if r['party_id']} == {'PP', 'MM'}
    assert 'Vox_(political_party)' in loader.parties_missing
    assert all(r['party_id'] is None for r in rows[0]['results'] if r['party'] not in ('PP', 'MM'))
```

- [ ] **Paso 2: ejecutar y ver que fallan** — `python -m pytest tests/test_loader_unit.py -q`
- [ ] **Paso 3: implementar** los cambios de la sección Interfaces.
- [ ] **Paso 4: ejecutar** `python -m pytest tests/test_loader_unit.py -q` → en verde. Si un recuento de
  `CASES` no coincide, comparar fila a fila con el criterio del comentario antes de tocar el número.
- [ ] **Paso 5: regresión de `es`** (integración, red): `WikipediaLoader('es', '2027-08-22').read_data()`
  devuelve el mismo número de sondeos que antes del cambio (anotar el número con `git stash` o en la rama
  previa) y los mismos conjuntos `*_missing`.

### Task 4: inventario de URLs, carga por lotes y sondeos de los 17 ámbitos

**Ficheros:**
- Crear: `load/run_load.py`
- Modificar: `data/wikipedia/wp-urls.json`, `data/wikipedia/wp-maps.json`
- Test: `tests/integration/test_regional_load.py` (nuevo)

**Interfaces:**
- Consume: `WikipediaLoader` (tarea 3), `get_scopes` (tarea 2).
- Produce, en `load/run_load.py` (script con `main()` y `argparse`, como `backtest/run_backtest.py`):
  - `build_urls(scope: str, since: int = 2009) -> dict[str, dict[str, str]]` — parte de
    `Next_{demonym}_regional_election` y retrocede por el enlace "previous election" de la ficha hasta la
    última elección anterior a `since`; por evento, `{fecha ISO: {'results': url, 'polls': url}}`, con
    `polls` sólo si `find_poll_table` encuentra tabla. La fecha del próximo evento es el límite legal de la
    legislatura que da la ficha.
  - `run_load(scopes: list[str], what: list[str], since: int = 2009, save: bool = False, overwrite: bool = False, verbose: int = 1) -> pd.DataFrame`
    — un informe con una fila por `(scope, event_date)`: `polls_read`, `polls_saved`, `parties_missing`,
    `pollsters_missing`, `sponsors_missing` (listas ordenadas por frecuencia). En esta tarea `what` admite
    `urls` y `polls`; la tarea 6 añade `results` y `compute`, la 7 `ratings`.
  - CLI: `python load/run_load.py --scopes es-md es-cl | all --what urls polls --since 2009 [--save] [--overwrite]`.
    Sin `--save` no escribe en la base. El informe se guarda en `files/stage/load_report.csv`.

- [ ] **Paso 1: test de integración que falla** en `tests/integration/test_regional_load.py`

```python
INVENTORY = {'es-ct': 587, 'es-an': 326, 'es-md': 339, 'es-ga': 343, 'es-pv': 289, 'es-vc': 294}


def test_regional_polls_cover_the_inventory(app):
    from mtpy.models.elections import Polls
    polls = Polls().get_results(formatted=True)
    counts = polls.loc[polls['event_scope'] != 'es'].groupby('event_scope', observed=True).size()
    assert counts.shape[0] == 17
    assert counts.sum() >= 0.95 * 3479
    for scope, n in INVENTORY.items():
        assert counts[scope] >= 0.95 * n, scope
```

- [ ] **Paso 2: implementar `build_urls` y `run_load`**, y ejecutar
  `python load/run_load.py --scopes all --what urls` para rellenar `wp-urls.json` (17 ámbitos, eventos desde
  2009 más el próximo). Revisar a mano las fechas del próximo evento de cada comunidad.
- [ ] **Paso 3: primera pasada en seco** — `python load/run_load.py --scopes all --what polls` e imprimir del
  informe, por ámbito, lo que falta por mapear.
- [ ] **Paso 4: preparar la curación** en `files/stage/`: `parties_new.csv` (`name`, `fullname`,
  `parent_id`, `block`, `regional`, ámbitos donde aparece) siguiendo D2, `pollsters_new.csv` (`name`, número
  de sondeos, ámbitos) y `sponsors_new.csv`; y los alias propuestos para `wp-maps.json` (globales para casas
  y patrocinadores, por ámbito para partidos).
- [ ] **Paso 5: [Luis]** revisar las tres listas, fijar `quality` de las casas nuevas con su rúbrica y
  aprobar la inserción en `parties`, `pollsters` y `sponsors`.
- [ ] **Paso 6: insertar lo aprobado, completar `wp-maps.json` y repetir** el paso 3 hasta que el informe no
  liste ninguna casa con 3 o más sondeos sin mapear ni ningún partido con resultado ≥ 1 % sin mapear.
- [ ] **Paso 7: cargar** — `python load/run_load.py --scopes all --what polls --save`.
- [ ] **Paso 8: ejecutar** `python -m pytest -q` y `python -m pytest tests/integration/test_regional_load.py -q`
  → en verde. Anotar el recuento por ámbito para la documentación (tarea 11).
- [ ] **Paso 9: commit**

```bash
git add load data/wikipedia mtpy/lib/loader.py tests
git commit -m "Regional polls loader and batch load script"
```

---

## Fase R2 — Resultados autonómicos

### Task 5: parsers de resultados

**Ficheros:**
- Modificar: `mtpy/lib/loader.py` (clase nueva `WikipediaResultsLoader`, métodos estáticos)
- Test: `tests/test_loader_unit.py`

**Interfaces:**
- Produce (estáticos, sin base de datos):
  - `WikipediaResultsLoader.find_results_tables(doc: html.HtmlElement) -> tuple[html.HtmlElement, Optional[html.HtmlElement]]`
    — la tabla *Overall* (`wikitable` con `caption` que empieza por "← Summary of") y la tabla por
    circunscripción (encabezado más cercano "Distribution by constituency" y primer `th` "Constituency"),
    `None` en distrito único.
  - `WikipediaResultsLoader.parse_overall(table: html.HtmlElement) -> tuple[pd.DataFrame, dict[str, int]]`
    — una fila por candidatura con `key` (clave wiki sin codificar del enlace, o `None`), `label`, `abbr`
    (texto del último paréntesis), `votes`, `pct`, `seats`; y los totales `votes` (válidos), `blank`,
    `invalid`, `counted`, `abstentions`, `registered`, `seats`. Con dos bloques de votos (Canarias) se lee
    el primero. Las columnas se localizan por los `th` "Votes", "%" y "Total", no por posición.
  - `WikipediaResultsLoader.parse_constituencies(table: html.HtmlElement) -> pd.DataFrame` — formato largo
    `district`, `key`, `abbr`, `pct`, `seats` (0 donde la celda es "−"); sin la fila "Total"; las celdas
    vacías con `rowspan`/`colspan` no generan fila. Reutiliza la expansión de `WikipediaLoader.read_rows`.
  - `WikipediaResultsLoader.split_votes(total: int, weights: pd.Series) -> pd.Series` — reparto entero
    proporcional por mayores restos; suma exactamente `total`.

- [ ] **Paso 1: tests que fallan**

```python
def results(page):
    overall, const = WikipediaResultsLoader.find_results_tables(load(page))
    rows, totals = WikipediaResultsLoader.parse_overall(overall)
    return rows.set_index('abbr'), totals, None if const is None else WikipediaResultsLoader.parse_constituencies(const)


def test_overall_castile_and_leon_2022():
    rows, totals, const = results('2022_Castilian-Leonese_regional_election')
    assert rows.loc['PP', ['key', 'votes', 'pct', 'seats']].tolist() == ["People's_Party_of_Castile_and_León", 382157, 31.40, 31]
    assert {k: totals[k] for k in ['votes', 'invalid', 'counted', 'abstentions', 'registered', 'seats']} == {
        'votes': 1217164, 'invalid': 13435, 'counted': 1230599, 'abstentions': 864024, 'registered': 2094623, 'seats': 81
    }
    assert totals['blank'] > 0 and rows['seats'].sum() == 81
    assert rows['votes'].sum() + totals['blank'] == totals['votes']
    cell = const.set_index(['district', 'key'])
    assert const['district'].nunique() == 9 and const['seats'].sum() == 81
    assert cell.loc[('Ávila', "People's_Party_of_Castile_and_León")].tolist()[-2:] == [34.0, 3]
    assert cell.loc[('León', "Leonese_People's_Union"), 'pct'] == 21.3
    assert cell.loc[('Soria', 'Empty_Spain'), 'pct'] == 42.7
    assert const.groupby('district')['seats'].sum().to_dict() == {
        'Ávila': 7, 'Burgos': 11, 'León': 13, 'Palencia': 7, 'Salamanca': 10, 'Segovia': 6, 'Soria': 5,
        'Valladolid': 15, 'Zamora': 7
    }


def test_canaries_2023_has_a_regional_list_and_two_vote_blocks():
    rows, totals, const = results('2023_Canarian_regional_election')
    assert (totals['votes'], totals['blank'], totals['seats']) == (912219, 15947, 70)
    assert rows.loc['PSOE', ['votes', 'pct', 'seats']].tolist() == [247811, 27.17, 23]
    cell = const.set_index(['district', 'key'])
    assert const['district'].nunique() == 8 and const['seats'].sum() == 70
    assert cell.loc[('Regional', 'Socialist_Party_of_the_Canaries')].tolist()[-2:] == [32.4, 4]
    assert cell.loc[('El Hierro', 'Independent_Herrenian_Group')].tolist()[-2:] == [26.3, 1]


def test_asturias_2023_constituencies_are_not_provinces():
    _, totals, const = results('2023_Asturian_regional_election')
    assert const.groupby('district')['seats'].sum().to_dict() == {'Central': 34, 'Eastern': 5, 'Western': 6}
    assert (totals['votes'], totals['registered'], totals['seats']) == (537023, 958658, 45)


def test_madrid_2023_is_a_single_district():
    rows, totals, const = results('2023_Madrilenian_regional_election')
    assert const is None
    assert rows.loc['PP', ['votes', 'pct', 'seats']].tolist() == [1599186, 47.32, 70]
    assert (totals['votes'], totals['registered'], totals['seats']) == (3379477, 5211710, 135)


def test_split_votes_is_exact():
    out = WikipediaResultsLoader.split_votes(100, pd.Series({'a': 1, 'b': 1, 'c': 1}))
    assert out.tolist() == [34, 33, 33]
    assert WikipediaResultsLoader.split_votes(1217164, pd.Series({'x': 3., 'y': 7.})).sum() == 1217164
```

- [ ] **Paso 2: ejecutar y ver que fallan** — `python -m pytest tests/test_loader_unit.py -q`
- [ ] **Paso 3: implementar** los cuatro métodos estáticos.
- [ ] **Paso 4: ejecutar** → en verde.

### Task 6: `WikipediaResultsLoader`, `estimated`, eventos próximos y carga

**Ficheros:**
- Modificar: `mtpy/lib/loader.py`, `mtpy/models/elections.py` (`EventsData.meta['estimated'] = 'bin'`),
  `mtpy/lib/data.py`, `load/run_load.py`, `data/README.md`
- Test: `tests/test_loader_unit.py`, `tests/integration/test_regional_load.py`

**Interfaces:**
- Consume: parsers de la tarea 5, `get_districts` y `get_scopes` de la tarea 2.
- Produce:
  - `WikipediaResultsLoader.build_frames(scope: str, event_date: str, overall: pd.DataFrame, totals: dict[str, int], constituencies: Optional[pd.DataFrame], districts: pd.DataFrame, party_names: dict[str, str], official: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]`
    — estático y sin base de datos. `party_names` mapea clave wiki o abreviatura al nombre del partido.
    Devuelve las filas de `events_data` y de `events_results` (con `party`, sin `party_id`). Reglas:
    - `region_id = 0`: exacta, de *Overall*; `estimated = False`. Lo no mapeado suma a `'-'`.
    - Circunscripciones: se resuelven contra `districts` por `name` o por `aliases`; un nombre desconocido
      lanza `ValueError` con la lista de nombres sin resolver.
    - Votos válidos por circunscripción: `split_votes(totals['votes'], population)` con `estimated = True`;
      si `official` (columnas `region_id`, `votes` y opcional `blank`) trae la circunscripción, esa cifra y
      `estimated = False`. Votos del partido = `round(pct · votes_i / 100)`; `blank` repartido en proporción
      a `votes_i`; `seats` de la circunscripción = suma de los de sus candidaturas.
    - Lista autonómica canaria (`region_id` 100): fuera del reparto por censo; `votes` = válidos del total.
    - Distrito único (`constituencies is None`): una fila de circunscripción idéntica a la del total con el
      `region_id` del único distrito del catálogo (Madrid 28, Murcia 30, Navarra 31, La Rioja 26, Cantabria 39).
  - `WikipediaResultsLoader(scope: str, event_date: str, verbose: int = 0, path: Optional[str] = None)` con
    `read_data() -> Self`, `build_series() -> Self`, `save_event() -> Self`, `save_totals() -> Self`,
    `save_results() -> Self`, `show_summary() -> None` y los atributos `totals`, `results`,
    `parties_missing`. `official` se lee de `data/results/{scope}/{fecha}.csv` si existe. Para el evento
    próximo (sin resultado), `build_series` deja en `totals` una fila por circunscripción vigente con los
    `seats` de `es-districts.csv` más el total, sin votos, y `results` vacío.
  - `save_event()` hace upsert de la fila de `events`: `name = 'Elecciones {scopes.name} {Mes} {Año}'`
    (`'Elecciones {scopes.name} Próximas'` para el próximo), `featured = False`.
  - `update_featured(scope: str, min_polls: int = 10) -> int` en `mtpy/lib/data.py` — marca `featured` los
    eventos del ámbito con resultado oficial y al menos `min_polls` sondeos con `computed` y
    `weight_over > 0`; devuelve cuántos.
  - `run_load` admite `results` (cargador completo por evento) y `compute` (por ámbito, la secuencia de
    `notebooks/data-load/PollsCompute.ipynb` sin ratings: `compute_weights(save, overwrite=True)`,
    `compute_errors`, `compute_deviations`, `update_featured`, `compute_drift`, `compute_house_effects`).
    El informe gana `votes_diff` (suma de votos de partidos menos válidos sin blancos) y `seats`.

- [ ] **Paso 1: [Luis]** añadir la columna: `ALTER TABLE elections.events_data ADD COLUMN estimated boolean;`
- [ ] **Paso 2: tests unitarios que fallan** en `tests/test_loader_unit.py`

```python
def frames(page, scope, event_date, **kwargs):
    rows, totals, const = results(page)
    districts = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))
    names = {key: key for key in rows['key'].dropna()}       # identidad: todo mapeado
    return WikipediaResultsLoader.build_frames(
        scope, event_date, rows.reset_index(), totals, const, districts.loc[districts['scope'] == scope], names, **kwargs
    )


def test_build_frames_castile_and_leon_2022():
    data, res = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13')
    total = data.set_index('region_id').loc[0]
    assert total['votes'] == 1217164 and total['seats'] == 81 and not total['estimated']
    parts = data.loc[data['region_id'] > 0]
    assert sorted(parts['region_id']) == [5, 9, 24, 34, 37, 40, 42, 47, 49]
    assert parts['estimated'].all() and parts['votes'].sum() == 1217164 and parts['seats'].sum() == 81
    votes0 = res.loc[res['region_id'] == 0, 'votes'].sum()
    assert abs(votes0 - (total['votes'] - total['blank'])) <= 0.001 * total['votes']
    assert res.loc[res['region_id'] > 0, 'seats'].sum() == 81


def test_build_frames_official_votes_replace_the_estimate():
    official = pd.DataFrame({'region_id': [5], 'votes': [90000]})
    data, _ = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13', official=official)
    avila = data.set_index('region_id').loc[5]
    assert avila['votes'] == 90000 and not avila['estimated']


def test_build_frames_single_district_and_regional_list():
    data, res = frames('2023_Madrilenian_regional_election', 'es-md', '2023-05-28')
    assert sorted(data['region_id']) == [0, 28]
    assert data.set_index('region_id').loc[28, 'seats'] == 135
    data, _ = frames('2023_Canarian_regional_election', 'es-cn', '2023-05-28')
    by_id = data.set_index('region_id')
    assert sorted(by_id.index) == [0, 100, 101, 102, 103, 104, 105, 106, 107]
    assert by_id.loc[100, 'seats'] == 9 and by_id.loc[range(101, 108), 'votes'].sum() == 912219


def test_unknown_constituency_raises_with_its_name():
    # Foco 4: sin el alias, el error nombra la circunscripción
    rows, totals, const = results('2023_Asturian_regional_election')
    districts = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))
    districts = districts.loc[districts['scope'] == 'es-as'].assign(aliases='')
    with pytest.raises(ValueError, match='Eastern'):
        WikipediaResultsLoader.build_frames('es-as', '2023-05-28', rows.reset_index(), totals, const, districts, {})
```

- [ ] **Paso 3: test de integración que falla** en `tests/integration/test_regional_load.py`

```python
def test_regional_results_are_consistent(app):
    from mtpy.models.elections import EventsData, EventsResults
    data = EventsData().get_results(query=dict(filters=["scope <> 'es'"]), formatted=True)
    res = EventsResults().get_results(query=dict(filters=["scope <> 'es'"]), formatted=True)
    past = data.loc[(data['region_id'] == 0) & data['votes'].notnull()]
    assert past['scope'].nunique() == 17
    for _, row in past.iterrows():
        mine = res.loc[(res['scope'] == row['scope']) & (res['date'] == row['date'])]
        votes = mine.loc[mine['region_id'] == 0, 'votes'].sum()
        assert abs(votes - (row['votes'] - row['blank'])) <= 0.001 * row['votes'], (row['scope'], row['date'])
        assert mine.loc[mine['region_id'] > 0, 'seats'].sum() == row['seats'], (row['scope'], row['date'])
    upcoming = data.loc[(data['region_id'] == 0) & data['votes'].isnull()]
    assert upcoming['scope'].nunique() == 17 and (upcoming['seats'] > 0).all()


def test_event_params_work_for_an_event_without_polls(app):
    # Foco 1: Asturias sólo tiene sondeos desde 2023; 2019 es su elección previa
    from mtpy.lib.data import get_event_params
    params = get_event_params('es-as', '2023-05-28', path='.')
    assert 'PSOE' in params['parties']['event'] and 'PP' in params['bmaps']['main']
```

- [ ] **Paso 4: implementar** `build_frames`, la clase completa, `update_featured` y los modos `results` y
  `compute` de `run_load`.
- [ ] **Paso 5: cargar** — `python load/run_load.py --scopes all --what results compute --save` y completar
  alias de candidaturas y de circunscripciones hasta que `votes_diff` de cada evento esté dentro del 0,1 %.
- [ ] **Paso 6: ejecutar** `python -m pytest -q` y `python -m pytest tests/integration -q` → en verde.
- [ ] **Paso 7: documentar** `data/results/` y la columna `estimated` en `data/README.md`.
- [ ] **Paso 8: commit**

```bash
git add mtpy load data tests
git commit -m "Regional results loader from Wikipedia with estimated district votes"
```

---

## Fase R3 — Ratings transversales

### Task 7: `compute_ratings` con `weight_scope`

**Ficheros:**
- Modificar: `mtpy/lib/computer.py:1198-1577`, `mtpy/lib/data.py` (`save_ratings_data`), `load/run_load.py`
- Test: `tests/test_computer_unit.py`, `tests/integration/test_ratings_scopes.py` (nuevo)

**Interfaces:**
- Consume: `get_scopes()['rating_weight']`.
- Produce:
  - `Computer.compute_ratings(self, save: bool = False, scopes: Optional[list[str] | str] = None, rating_weights: Optional[dict[str, float]] = None) -> pd.DataFrame`
    — `scopes=None`: sólo el ámbito propio (comportamiento actual); `'all'`: los ámbitos de `get_scopes()`
    con peso > 0. `rating_weights` sustituye a los del catálogo (tests y calibración). Las filas se guardan
    con `event_scope = self.scope`.
  - `Computer.rating_pool(self, scopes: list[str]) -> pd.DataFrame` — sondeos filtrados de cada ámbito
    (el propio con `self.filter_polls()`; los demás con un `Computer` de ese ámbito construido con los
    mismos parámetros de filtro), sin columnas de partido, índice `['event_scope'] + self.keys`.
  - `Computer.rate_events(self, pool: pd.DataFrame, targets: list[pd.Timestamp], rating_weights: dict[str, float], pollsters: pd.DataFrame) -> pd.DataFrame`
    — sin base de datos. Para cada fecha de `targets`, `pollster_ratings` sobre las filas de `pool` con
    `event_date` **estrictamente anterior**; se salta el objetivo sin filas. Las filas de ámbitos con peso
    0 se descartan **antes** de calcular los pesos (si no, cambiarían el año máximo de `weight_year`).
    Devuelve `event_date`, `pollster_id` y las columnas de rating, en la escala interna.
  - `poll_rating_weights(self, polls, rating_weights: Optional[dict[str, float]] = None)` — agrupa por
    `(event_date, event_scope, pollster_id)` y `(event_date, event_scope)`, y añade `weight_scope`; si el
    índice no trae `event_scope`, todo es `self.scope` con peso 1.
  - `pollster_ratings(self, polls, rating_weights=None, pollsters=None)` — `weight` incluye `weight_scope`;
    `num_events` cuenta pares `(event_scope, event_date)`; las casas son las de `pollsters` (por defecto
    `self.pollsters`).
  - `compute_ratings` pasa a `rate_events` como `targets` las fechas de evento del ámbito propio con sondeos
    más la del próximo, y como `pollsters` las del ámbito propio más las que tienen sondeos con peso
    positivo en `pool`.
  - `self.ratings` pasa a contener todas las filas calculadas (unión de su índice y el del resultado), no
    sólo las que ya estaban en la base.
  - `save_ratings_data(data: pd.DataFrame, update_pollsters: bool = True) -> int`; `compute_ratings` lo
    llama con `update_pollsters=(self.scope == 'es')`.
  - `run_load` admite `ratings`: por ámbito, `compute_ratings(save, scopes='all')`.

- [ ] **Paso 1: tests unitarios que fallan** en `tests/test_computer_unit.py`. `bare_computer()` construye
  un `Computer` con `__new__` y fija `scope='es'`, `keys`, `pos_decay=.5`, `week_decay=.7`, `year_decay=.9`,
  `bias_dev_tau=.01`, `min_polls=3` y `pollsters` (A, B, C, D con `id` 1-4 y `quality` 60). `pool(rows)`
  construye el conjunto con las columnas que usa `pollster_ratings`: un sondeo por tupla, 10 días antes del
  evento, `weight_over = weight_sample = 1`, `bias_dev_err = 0.01` y `bias_dev_adj` de +0,01 para A y C y
  −0,01 para B y D (escala LOR).

```python
ES19 = [('es', '2019-11-10', 'A'), ('es', '2019-11-10', 'B')]
MD21 = [('es-md', '2021-05-04', 'A'), ('es-md', '2021-05-04', 'C')]
REG23 = [('es-md', '2023-05-28', 'C'), ('es-cl', '2023-05-28', 'D')]
JULY23, MAY23 = pd.Timestamp('2023-07-23'), pd.Timestamp('2023-05-28')


def test_scope_weight_zero_reproduces_single_scope_ratings():
    c = bare_computer()
    mixed = c.rate_events(pool(ES19 + MD21), [JULY23], {'es': 1., 'es-md': 0.}, c.pollsters)
    alone = c.rate_events(pool(ES19), [JULY23], {'es': 1.}, c.pollsters)
    pd.testing.assert_frame_equal(mixed, alone)


def test_scope_weight_one_equals_merging_the_events():
    c = bare_computer()
    mixed = c.rate_events(pool(ES19 + MD21), [JULY23], {'es': 1., 'es-md': 1.}, c.pollsters)
    merged = c.rate_events(pool(ES19 + [('es', d, p) for _, d, p in MD21]), [JULY23], {'es': 1.}, c.pollsters)
    pd.testing.assert_frame_equal(mixed, merged)


def test_same_day_events_do_not_see_each_other():
    c = bare_computer()
    weights = {'es': 1., 'es-md': .5, 'es-cl': .5}
    assert c.rate_events(pool(REG23), [MAY23], weights, c.pollsters).shape[0] == 0
    july = c.rate_events(pool(REG23), [JULY23], weights, c.pollsters).set_index('pollster_id')
    assert (july.loc[[3, 4], 'num_polls'] > 0).all()


def test_regional_only_pollster_is_rated_and_events_are_scope_date_pairs():
    # Foco 5: C sólo publica en Madrid y aun así tiene rating en un evento de `es`
    c = bare_computer()
    out = c.rate_events(pool(ES19 + MD21 + REG23), [JULY23], {'es': 1., 'es-md': .5, 'es-cl': .5}, c.pollsters)
    out = out.set_index('pollster_id')
    assert out.loc[3, 'num_events'] == 2 and out.loc[3, 'num_polls'] == 2      # C: Madrid 2021 y 2023
    assert out.loc[1, 'num_events'] == 2                                       # A: es 2019 y Madrid 2021
    assert out.loc[3, 'rating'] != out.loc[3, 'quality']
```

- [ ] **Paso 2: test de integración que falla** en `tests/integration/test_ratings_scopes.py`

```python
def test_es_ratings_are_invariant_when_regional_weights_are_zero(app):
    from mtpy.lib.computer import Computer
    from mtpy.lib.data import get_scopes
    zero = {s: (1. if s == 'es' else 0.) for s in get_scopes().index}
    base = Computer(scope='es', path='.').build_series()
    base.compute_ratings()
    alone = base.ratings.copy()
    cross = Computer(scope='es', path='.').build_series()
    cross.compute_ratings(scopes=list(zero), rating_weights=zero)
    pd.testing.assert_frame_equal(alone, cross.ratings)


def test_regional_weights_move_ratings_within_bounds(app):
    from mtpy.lib.computer import Computer
    def last_ratings(**kwargs):
        comp = Computer(scope='es', path='.').build_series()
        comp.compute_ratings(**kwargs)
        return comp.ratings.xs(comp.ratings.index.get_level_values('event_date').max(), level='event_date')

    alone, cross = last_ratings(), last_ratings(scopes='all')
    assert cross['rating'].between(0, 100).all()
    # Entran casas que sólo publican en autonómicas
    assert (cross['num_polls'] > 0).sum() > (alone['num_polls'] > 0).sum()
```

- [ ] **Paso 3: ejecutar y ver que fallan.**
- [ ] **Paso 4: implementar** la sección Interfaces.
- [ ] **Paso 5: ejecutar** `python -m pytest -q` y `python -m pytest tests/integration -q` → en verde.
- [ ] **Paso 6: informe de impacto** (para la tarea 11): con `scopes='all'` y los pesos del catálogo, las 15
  casas cuyo rating de `es` más cambia respecto a `scopes=None`, y el rating de Sondaxe, GESOP e Ikertalde.
- [ ] **Paso 7: [Luis]** revisar el informe y dar el visto bueno a guardar los ratings transversales
  (`python load/run_load.py --scopes all --what ratings --save`, con `es` el último).
- [ ] **Paso 8: commit**

```bash
git add mtpy load tests
git commit -m "Cross-scope pollster ratings with scope weights"
```

---

## Fase R4 — Forecaster y Simulator por ámbito

### Task 8: umbral por ámbito y umbral alternativo

**Ficheros:**
- Modificar: `mtpy/lib/simulator.py:33-146,940-977,1715-1727`, `mtpy/lib/data.py`, `data/params.json`
- Test: `tests/test_simulator_unit.py`, `tests/test_data_unit.py`, `tests/integration/test_model.py:86-104`

**Interfaces:**
- Produce:
  - `Simulator.alloc_seats(d_votes, n_seats, valid_votes=None, threshold=None, scope_shares: Optional[dict[str, float]] = None, threshold_scope: Optional[float] = None) -> dict[str, int]`
    — regla de la precisión 4; con `threshold_scope=None` el comportamiento es el actual.
  - `resolve_thresholds(scope_row: Mapping[str, Any], event_conf: Optional[Mapping[str, Any]] = None) -> tuple[Optional[float], Optional[float]]`
    en `mtpy/lib/data.py` — puro. Las claves `threshold` y `threshold_scope` presentes en `event_conf`
    prevalecen aunque valgan `null`; NaN se devuelve como `None`.
  - `get_thresholds(scope: str, event_date: str) -> tuple[Optional[float], Optional[float]]` — lee
    `get_scopes()` y la entrada del evento en `params.json`.
  - `Simulator(threshold: Optional[float] = None)`: `None` resuelve con `get_thresholds`; un número fija el
    umbral de circunscripción y anula el alternativo. Atributos `self.threshold` y `self.threshold_scope`.
    `simulate` pasa a `alloc_seats` las cuotas de la fila `default_region` de `umat['vpred_pct']`.

- [ ] **Paso 1: tests que fallan**

`tests/test_simulator_unit.py`:

```python
def test_alloc_seats_alternative_scope_threshold():
    # Canarias: 15 % en la isla o 4 % en el conjunto
    votes, shares = {'A': 50, 'B': 36, 'C': 14}, {'A': 40., 'B': 30., 'C': 5.}
    assert Simulator.alloc_seats(votes, 10, valid_votes=100, threshold=15.0)['C'] == 0
    assert Simulator.alloc_seats(votes, 10, valid_votes=100, threshold=15.0, scope_shares=shares, threshold_scope=4.0) == {'A': 5, 'B': 4, 'C': 1}
    assert Simulator.alloc_seats(votes, 10, valid_votes=100, threshold=15.0, scope_shares={**shares, 'C': 3.9}, threshold_scope=4.0)['C'] == 0


def test_alloc_seats_scope_threshold_only():
    # Comunidad Valenciana: sólo 5 % autonómico
    votes = {'A': 60, 'B': 34, 'C': 6}
    below, above = {'A': 60., 'B': 35.1, 'C': 4.9}, {'A': 60., 'B': 34., 'C': 6.}
    assert Simulator.alloc_seats(votes, 20, valid_votes=100, scope_shares=below, threshold_scope=5.0)['C'] == 0
    assert Simulator.alloc_seats(votes, 20, valid_votes=100, scope_shares=above, threshold_scope=5.0) == {'A': 12, 'B': 7, 'C': 1}
```

`tests/test_data_unit.py`:

```python
def test_resolve_thresholds_event_override_wins_even_when_null():
    nan = float('nan')
    assert resolve_thresholds({'threshold': 3.0, 'threshold_scope': nan}) == (3.0, None)
    assert resolve_thresholds({'threshold': 5.0, 'threshold_scope': nan}, {'smap': {}}) == (5.0, None)
    assert resolve_thresholds({'threshold': 3.0, 'threshold_scope': nan}, {'threshold': None, 'threshold_scope': 5.0}) == (None, 5.0)
    assert resolve_thresholds({'threshold': 15.0, 'threshold_scope': 4.0}, {'threshold': 30.0, 'threshold_scope': 6.0}) == (30.0, 6.0)
```

- [ ] **Paso 2: ejecutar y ver que fallan.**
- [ ] **Paso 3: implementar**, y añadir a `data/params.json` los overrides históricos anotados en la tarea 1
  (como mínimo `es-mc` 2011 y 2015: `"threshold": null, "threshold_scope": 5`; `es-cn` 2011 y 2015:
  `"threshold": 30, "threshold_scope": 6`). `test_data_files.py` comprueba que todo override está en (0, 100]
  o es nulo.
- [ ] **Paso 4: ejecutar** `python -m pytest -q` → en verde; `tests/integration/test_model.py` y
  `test_threshold_official.py` sin cambios en verde (en `es` el umbral resuelto es 3).

### Task 9: circunscripciones, ruido M9 y estimadores con caída al padre

**Ficheros:**
- Modificar: `mtpy/lib/simulator.py:190-256,360-375,508-528,565-585,1623-1661,1728-1731`,
  `mtpy/lib/computer.py:1905-1908,2555-2562,2621-2692`, `mtpy/lib/forecaster.py:598-605`, `data/params.json`
- Test: `tests/test_simulator_unit.py`, `tests/test_forecaster_unit.py`,
  `tests/integration/test_regional_model.py` (nuevo)

**Interfaces:**
- Consume: `get_scopes`, `get_districts`, `get_thresholds`, los datos cargados en R1-R2 y los ratings de R3.
- Produce:
  - `Simulator.estimator_scope(scope: str, parent: Optional[str], n_featured: int, min_events: int = 3) -> str`
    — estático: el ámbito propio con `n_featured >= min_events` o sin padre; si no, el padre.
  - `Simulator.computer` se construye sobre ese ámbito con sus eventos de fecha **estrictamente anterior** a
    `event_date` (hoy `skip=1`, que sólo vale dentro del propio ámbito). De él salen `v2seats`, `v2err`,
    `v2drift` y `composition_ratio`.
  - `Simulator.v2swing`: del `Computer` de `es` siempre (precisión 8); en `es` es el mismo objeto.
  - `Simulator.party_ages` toma los primeros sondeos del ámbito **propio**: un `Computer` del ámbito sólo
    cuando éste tiene eventos anteriores con sondeos (`get_event_dates(min_polls=1)`); si no, sólo el ciclo.
  - `region_groups` = `get_districts(scope)['reg_code']`; `region_names` = nombres del catálogo, con el de
    `events_data.region` como respaldo.
  - `_add_swing_noise`: con un solo grupo entre las circunscripciones, los choques por comunidad son 0 y no
    se sortean; con menos de dos circunscripciones no hace nada. En `es` la secuencia aleatoria no cambia.
  - Variable `regional`: 0 para todos los partidos cuando el ámbito tiene padre, en
    `Simulator.build_forecast`, `Computer.get_error_estimator_data`, `Computer.get_seats_estimator_data` y
    `Computer.get_composition_ratio`.
  - Estimador de escaños sobre la cuota: `get_seats_estimator_data` añade `seats_total` y
    `share = seats / seats_total`; `get_seats_estimator` regresa `share`; `Simulator.simulate` multiplica la
    predicción por `self.n_seats`.
  - `Forecaster.usable_history(history: pd.DataFrame, min_events: int = 3) -> pd.DataFrame` — estático:
    devuelve `history` vacío (mismas columnas) si tiene menos de `min_events` fechas de evento distintas;
    `load_house_history` lo aplica. Con menos de 3 eventos el prior de los efectos de casa se anula.

- [ ] **Paso 1: tests que fallan**

`tests/test_forecaster_unit.py`:

```python
def test_usable_history_needs_three_events():
    dates = pd.to_datetime(['2015-05-24', '2019-05-26', '2021-05-04'])
    history = pd.DataFrame({'event_date': dates.repeat(2), 'pollster_id': [1, 2] * 3, 'dev_result_c': 1.})
    assert Forecaster.usable_history(history).shape[0] == 6
    short = Forecaster.usable_history(history.iloc[:4])
    assert short.shape[0] == 0 and list(short.columns) == list(history.columns)
```

`tests/test_simulator_unit.py` (el segundo test es una guarda: ya pasa con el código actual):

```python
def test_estimator_scope_falls_back_to_the_parent():
    assert Simulator.estimator_scope('es-md', 'es', 2) == 'es'
    assert Simulator.estimator_scope('es-md', 'es', 3) == 'es-md'
    assert Simulator.estimator_scope('es', None, 0) == 'es'


def test_apply_swing_noise_without_regional_shocks_keeps_means():
    # Ámbito autonómico: sólo el choque por circunscripción; la cuota del ámbito se conserva
    shares = pd.DataFrame({'A': [40., 30., 20.], 'B': [30., 40., 50.], '-': [30., 30., 30.]}, index=[5, 9, 24])
    weights, groups = pd.Series([1., 2., 3.], index=shares.index), pd.Series(7, index=shares.index)
    target = shares[['A', 'B']].mul(weights, axis=0).sum() / weights.sum()
    eta = pd.DataFrame({'A': [.1, -.1, 0.], 'B': [0., .05, -.05]}, index=shares.index)
    out = Simulator.apply_swing_noise(shares, weights, target, pd.DataFrame(), eta, groups, ['A', 'B'], '-')
    assert np.allclose(out.sum(axis=1), 100)
    assert np.allclose(out[['A', 'B']].mul(weights, axis=0).sum() / weights.sum(), target, atol=0.1)
    assert not np.allclose(out['A'], shares['A'])
```

`tests/integration/test_regional_model.py`:

```python
def regional_sim(scope, event_date, **kwargs):
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope=scope, event_date=event_date, drange=6, seed=42, verbose=0, path='.', **kwargs)
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


@pytest.mark.parametrize('scope, event_date', [('es-md', '2023-05-28'), ('es-cl', '2022-02-13')])
def test_deterministic_seats_are_close_to_the_official_ones(app, scope, event_date):
    from mtpy.lib.backtest import official_results
    sim = regional_sim(scope, event_date)
    sim.run(split=True, random=False)
    model = sim.result().loc[scope]
    official = official_results(scope, event_date)['seats']
    main = sim.event_params['bmaps']['main']
    assert int(model.sum()) == sim.n_seats
    assert (model[main] - official.reindex(main).fillna(0)).abs().mean() <= 1.5


def test_scope_settings_are_resolved(app):
    md = regional_sim('es-md', '2023-05-28')
    assert (md.threshold, md.threshold_scope) == (5.0, None)
    assert md.regions == [0, 28] and md.region_names[28] == 'Madrid'
    md.run(split=True, random=False)
    assert (md.frame()['regional'] == 0).all()
    cn = regional_sim('es-cn', '2023-05-28')
    assert (cn.threshold, cn.threshold_scope) == (15.0, 4.0) and 100 in cn.regions


def test_random_run_in_a_single_district_scope(app):
    md = regional_sim('es-md', '2023-05-28')
    md.run(split=True, random=True, n_sim=20)
    assert (md.dist().sum(axis=1) == 135).all()
    md.run(split=False, random=False)
    assert int(md.totals().sum()) == 135


def test_previous_event_without_polls_is_a_valid_base(app):
    # Foco 1: Asturias 2023 se proyecta sobre 2019, que no tiene sondeos
    sim = regional_sim('es-as', '2023-05-28')
    assert sim.prev_date == '2019-05-26'
    sim.run(split=True, random=True, n_sim=10)
    assert (sim.dist().sum(axis=1) == 45).all()
```

- [ ] **Paso 2: ejecutar y ver que fallan.**
- [ ] **Paso 3: implementar** la sección Interfaces.
- [ ] **Paso 4: curar el `smap`** en `data/params.json` de los eventos autonómicos donde una candidatura
  hereda de otra (Cs → PP, UP → SUMAR o Podemos, coaliciones que cambian de nombre), empezando por los de los
  tests y los 17 eventos próximos. **[Luis]** revisa las reglas propuestas.
- [ ] **Paso 5: ejecutar** `python -m pytest -q` y `python -m pytest tests/integration -q` → en verde, incluida
  `test_nowcast_regression_seed_42` y `test_ls_totals_respect_zero_parties` sin tocar (la cuota de escaños
  con 350 fijos da la misma predicción).
- [ ] **Paso 6: anotar** para la tarea 11 el error absoluto medio de escaños de Madrid 2023 y Castilla y
  León 2022 y, si alguno de los 17 ámbitos no construye el `Simulator` de su próximo evento, el motivo.
- [ ] **Paso 7: commit**

```bash
git add mtpy data/params.json tests
git commit -m "Scope thresholds, districts and parent estimators in the simulator"
```

---

## Fase R5 — Backtest, notebooks y documentación

### Task 10: backtest por ámbito

**Ficheros:**
- Modificar: `mtpy/lib/backtest.py:25,393-418`, `backtest/run_backtest.py`, `backtest/README.md`
- Test: `tests/test_backtest_unit.py`, `tests/integration/test_backtest.py`

**Interfaces:**
- Produce:
  - `default_events(scope: str, since: str = '2019-01-01') -> list[str]` — `DEFAULT_EVENTS` para `es`; para
    el resto, los eventos `featured` del ámbito desde `since`. `run_backtest(events=None)` lo usa.
  - `results_dir(root: str, scope: str) -> str` en `backtest/run_backtest.py` — `root` para `es`,
    `root/{scope}` para los demás.
  - CLI: `--scopes es-md es-cl | all` (excluye a `--scope`); ejecuta el backtest de cada ámbito con eventos
    evaluables y escribe en su carpeta. Los ámbitos sin eventos se listan y se saltan.

- [ ] **Paso 1: tests que fallan**

```python
def test_results_dir_keeps_es_at_the_root():
    assert results_dir('backtest/results', 'es') == 'backtest/results'
    assert results_dir('backtest/results', 'es-md') == os.path.join('backtest/results', 'es-md')
```

```python
def test_run_case_regional_2023_6_days(app):
    from mtpy.lib.backtest import default_events, run_case
    assert '2023-05-28' in default_events('es-md') and '2027-08-22' not in default_events('es')
    case = run_case('es-md', '2023-05-28', 6, n_sim=50, nowcast_only=True)
    assert case['shares'].shape[0] >= 4 and int(case['seats']['official'].sum()) == 135
```

- [ ] **Paso 2: ejecutar y ver que fallan; implementar; ejecutar en verde.**
- [ ] **Paso 3: correr** `python backtest/run_backtest.py --scopes all --horizons 6 30 --n-sim 500` y comprobar
  que `backtest/results/es/…` **no** se crea y que `backtest/results/*.csv` de `es` no cambia.
- [ ] **Paso 4: tabla de cierre** — por ámbito con 3 o más eventos, cobertura del 50/80/95 % de las cuotas a
  6 días; criterio de la spec: dentro de ±0,15 del nominal. Los que no lo cumplan se documentan como
  salvedad, no bloquean.

### Task 11: notebooks y documentación

**Ficheros:**
- Modificar: `notebooks/data-load/PollsLoadWikipedia.ipynb`, `PollsCompute.ipynb`,
  `notebooks/PollsForecast.ipynb`, `notebooks/PollsSimulations.ipynb`, `data/README.md`,
  `backtest/README.md`, `docs/analisis-2026-09/metodo-11-ambitos-autonomicos.md`
- Crear: `notebooks/data-load/ResultsLoadWikipedia.ipynb`, `notebooks/events/es-md-202305/`
  (`PollsAnalysis.ipynb`, `PollsSimulations.ipynb`, `PollstersEvent.ipynb`, a partir de `es-202307/`)

- [ ] **Paso 1: notebooks.** Cada uno, una sola celda de parámetros con `scope`; ningún `'es'` literal fuera
  de ella. `ResultsLoadWikipedia.ipynb` sigue la estructura de `ResultsLoadInfoElectoral.ipynb` con
  `WikipediaResultsLoader`. `PollsCompute.ipynb` llama a `compute_ratings(save=save, scopes='all')`.
  Ejecutar `es-md-202305` de principio a fin.
- [ ] **Paso 2: `data/README.md`** — `es-scopes.csv`, `es-districts.csv`, `results/`, la estructura por ámbito
  de `wp-urls.json` y `wp-maps.json`. **`backtest/README.md`** — `--scopes` y las carpetas por ámbito.
- [ ] **Paso 3: `metodo-11-ambitos-autonomicos.md`** pasa al formato de los métodos anteriores: cambios
  hechos, tests e impacto numérico (sondeos y eventos por ámbito; las 15 casas que más cambian y las casas
  regionales; MAE de escaños de Madrid 2023 y Castilla y León 2022; coberturas del backtest), las
  precisiones de este plan y las salvedades encontradas. Este plan se cita desde allí.
- [ ] **Paso 4: suite completa** — `python -m pytest -q` y `python -m pytest tests/integration -q` → en verde.
- [ ] **Paso 5: commit**

```bash
git add backtest notebooks data/README.md docs
git commit -m "Regional backtest, notebooks and method 11 documentation"
```
