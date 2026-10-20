# Análisis «El voto CERA y la Ley de Nietos» (29-N) · plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar en la web la sección «Análisis» con la página viva `/analisis/voto-cera`: el estudio del voto CERA anclado a las generales del 29-N, calculado desde datos oficiales versionados, publicado como un «estudio» del paquete y comprobable tras el escrutinio.

**Architecture:** Un paquete de cálculo puro (`mtpy/lib/studies/cera/`) lee `data/cera/` y los Infoelectoral de `data/infoelectoral/es/`, reproduce el artículo y añade el 29-N; `publish/studies.py` lo exporta como partes JSON + CSV bajo `site/v1/studies/voto-cera/` (una clase de publicación nueva, mutable como `analysis`); la API las sirve en `/api/v1/studies/...`; las páginas Jinja2 (`/analisis`, `/analisis/voto-cera`) las pintan con textos de cifras vivas y una isla para los gráficos ECharts.

**Tech Stack:** Python 3.11, pandas, statsmodels (HAC), openpyxl + xlrd (libros INE), Jinja2, ECharts 5.6 (módulos ES en `web/dist/js`), pytest, node para las opciones de los gráficos.

**Spec:** `docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-design.md` (maqueta aprobada: `docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-mockup.html`).

## Global Constraints

- PEP8, docstrings numpydoc en toda función nueva (AGENTS.md); mensajes de commit en inglés como los del repo, cada uno terminado en `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Nada de base de datos en el cálculo del estudio: todo sale de `data/` (`App.get_().data`, un `FileSystem`).
- Textos públicos en español, con el tono de la infografía; ninguna cifra escrita a mano en las plantillas: todas salen de las partes publicadas. El artículo de Medium solo se menciona dentro de `#datos`.
- Veredictos exactos: «SÍ», «SIN EVIDENCIA», «NO». Titular exacto: «El voto CERA en España y el efecto de la «Ley de Nietos»»; subtítulo exacto: «¿Es posible orquestar con éxito un fraude electoral significativo a través del censo exterior?».
- Umbral de acierto `check.max_gain = 1` (2 o más escaños netos a favor del PSOE o de la izquierda es fallo).
- Colores de las tarjetas-pregunta: verde `#2e9e5b`, naranja `#e8912b`, rojo `#ce3b3b`, solo en ellas.
- Reparto D'Hondt con umbral del 3 % de los votos válidos (candidaturas + blancos) y **todas** las candidaturas del XLSX; 350/350 escaños de 2023 es test bloqueante.
- Las partes se escriben con `meta` la última; el estudio no toca runs, análisis, eventos ni backtest; queda fuera de la guardia LOREG.
- Sin `register=True` no hay registro; con árbol sucio (`provenance()['dirty']`) `register` falla.
- Hoy el repo tiene 817 tests en `dev` (`669f909`); todos deben seguir pasando.

## Review Focus

1. **Un libro anual del INE que revisa meses ya leídos** (el fichero del año en curso vuelve a bajarse con columnas nuevas y, a veces, cifras corregidas de meses anteriores): `read_monthly` conserva la lectura del fichero más reciente, sin duplicar filas (test en la tarea 2).
2. **Un fichero de escrutinio del 29-N con una candidatura que el mapa no conoce**: `scrutiny_votes` falla nombrando las columnas desconocidas en lugar de descartarlas en silencio (tarea 2).
3. **Circunscripciones de magnitud 1 y 2** (Ceuta, Melilla, Soria): el último escaño y la brecha están definidos (Ceuta 1.586 en 2023, tabla 10) (tarea 4).
4. **Estudio publicado sin registro y sin escrutinio del 29-N**: la página sale completa, dice «borrador» y no muestra píldora ni fecha de registro; con `check` del 29-N vacío la tarjeta de comprobación muestra solo 2019 y 2023 (tarea 13).
5. **`litigation.count = 0`** y **`register=True` con árbol sucio**: el escenario B coincide con el A, y la publicación se rechaza con un mensaje claro sin escribir nada (tareas 5 y 10).

---

## Estructura de ficheros

**Datos (versionados):**
- `data/cera/ine/cerapropais_06.xls` … `cerapropais_26.xlsx` (copias de la carpeta del artículo; el 26 descargado de nuevo, llega a 1-9-2026), `data/cera/ine/elecgen2026_tab6.xlsx`, `_tab7.xlsx`, `_tab8.xlsx` (del scratchpad de esta sesión o del INE).
- `data/cera/escrutinio/es-2019-11-10.csv`, `es-2023-07-23.csv` (del artículo, con `region_id`), `data/cera/escrutinio/candidatures.json` (columna del escrutinio → candidatura del XLSX, por elección).
- `data/cera/provinces.json` (etiqueta normalizada → `region_id`, para el INE y para Interior), `data/cera/countries.csv`, `data/cera/params.json`, `data/cera/autonomicas.csv`, `data/cera/consulados/{cyl-2026,extremadura-2023,extremadura-2025,aragon-2026}.csv`.

**Cálculo:** `mtpy/lib/studies/__init__.py`, `mtpy/lib/studies/cera/__init__.py`, `data.py` (lecturas), `census.py` (serie y crecimiento), `behaviour.py` (participación, voto, colocación), `seats.py` (D'Hondt, último escaño, incremento, sensibilidad, comprobación), `study.py` (orquestación: `compute_cera`).

**Publicación y API:** `mtpy/lib/bundle.py` (estudios), `mtpy/lib/publish/studies.py` (exportadores y `publish_study`), `mtpy/lib/publish/pointers.py` (`studies` en el manifiesto), `mtpy/lib/publish/__init__.py`, `mtpy/jobs/Publish.py` (`what='study'`), `mtpy/lib/webapi/checks.py`, `published.py`, `__init__.py` (rutas), `mtpy/controllers/Studies.py`.

**Carga:** `load/cera.py`, `load/run_load.py` (paso `cera`).

**Web:** `mtpy/lib/pages/analisis.py`, `mtpy/lib/pages/common.py` (menú, rutas, `api_url`), `mtpy/lib/pages/inicio.py` (banda), `mtpy/controllers/Analisis.py`, `web/templates/analisis.html`, `web/templates/analisis_voto_cera.html`, `web/templates/index.html`, `web/dist/css/site.css`, `web/dist/js/charts/{census,scatter,slope,radar}.js`, `web/dist/js/pages/analisis-cera.js`.

**Docs:** `data/README.md`, `docs/web/contrato.md`, la spec (estado y decisiones).

**Tests:** `tests/test_cera_data.py`, `tests/test_cera_census.py`, `tests/test_cera_behaviour.py`, `tests/test_cera_seats.py`, `tests/test_publish_studies.py`, `tests/test_load_cera.py`, ampliaciones de `tests/test_bundle_unit.py`, `tests/test_jobs_publish.py`, `tests/test_web_api.py`, `tests/test_webapi_unit.py` (helper `write_study`), `tests/test_pages.py`, `tests/test_js_charts.py`; fixtures `tests/fixtures/bundle/study-*.json` y `tests/fixtures/cera/`.

---

### Task 1: Los datos de entrada en `data/cera/`

**Files:**
- Create: `data/cera/ine/*` (24 ficheros), `data/cera/escrutinio/es-2019-11-10.csv`, `data/cera/escrutinio/es-2023-07-23.csv`, `data/cera/escrutinio/candidatures.json`, `data/cera/provinces.json`, `data/cera/countries.csv`, `data/cera/params.json`, `data/cera/autonomicas.csv`, `data/cera/consulados/*.csv`
- Modify: `data/README.md` (tabla de ficheros)
- Test: `tests/test_cera_data.py`

**Interfaces:**
- Produces: los ficheros con los formatos de abajo; `params.json` con las claves de la spec (D8) y `check.max_gain = 1`.

Formatos (decisiones de este plan):
- `escrutinio/es-{fecha}.csv`: columnas `region_id` (INE, 1–52; solo provincias: fuera comunidades y total), `label` (la etiqueta original de Interior), `electors`, `voters`, `invalid`, `blank` y después una columna por candidatura con la etiqueta de Interior tal cual (`PP`, `PSOE`, `SUMAR`, `VOX`, `PACMA`, …, `PSC`, `ERC-SOBIRANISTES`…). Se construyen una vez desde `data/cera_generales_20xx_partido.csv` de la carpeta del artículo con un script desechable (no se versiona).
- `escrutinio/candidatures.json`: `{"2019-11-10": {"PSOE": "PARTIDO SOCIALISTA OBRERO ESPAÑOL", "PSC": "PARTIT DELS SOCIALISTES DE CATALUNYA (PSC-PSOE)", ...}, "2023-07-23": {...}}`: cada columna del escrutinio → el nombre exacto de la candidatura en la fila 3 del `PROV_02_YYYYMM_1.xlsx`. Toda columna del escrutinio debe estar en el mapa (las que no son candidatura, como `ELECTORESCERA`, no van). El constructor construye el mapa comparando los dos ficheros y lo completa a mano.
- `provinces.json`: `{"ALBACETE": 2, "ACORUNA": 15, "CORUNAA": 15, "CDEMADRID": 28, "MADRID": 28, "ARABAALAVA": 1, "ALAVA": 1, "ILLESBALEARS": 7, "BALEARS": 7, "CADECEUTA": 51, ...}` con las etiquetas **normalizadas** (`normalise_label`, tarea 2) de las tres fuentes: libros mensuales (código de dos cifras en la primera columna, no hace falta etiqueta), `tab8` (nombre en mayúsculas), ficheros de Interior (`C.DEMADRID`, `R.DEMURCIA`, `C.F.DENAVARRA`, `LARIOJA`, `ILLESBALEARS`, `ARABA/ALAVA`, `ALICANTE/ALACANT`, `C.A.DECEUTA`, …).
- `countries.csv`: `code,name,region,exposure` con `code` el del INE (`126` Alemania), `region` ∈ `latam | europe | usa_canada | other`, `exposure` ∈ `alta | media | baja`; regla de la spec (decisión 7 de la revisión). Todos los países de `tab6` presentes.
- `autonomicas.csv`: `community,date,previous,electors,voters,pp,psoe,vox,other_left,left,right,other` (votos; `previous` = fecha del proceso anterior o vacío; `left`/`right`/`other` = votos por bloque con los bloques del artículo, de los PDF y de `cyl_cera_bloques.csv`).
- `consulados/{proceso}.csv`: `country,electors,voters` agregados por país (los ficheros del artículo traen varias filas por país; se suman). Aragón 2026: de la tabla 8 del artículo (PDF del BOA).
- `params.json`: el de la spec D8 con `"check": {"max_gain": 1}`, `"ecological_min_census": 2000` (ajústese para que solo queden fuera Ceuta y Melilla), `"blocks": {"Izquierda": ["PSOE", "SUMAR", "UP", "MP"], "Derecha": ["PP", "VOX", "CS"], "Separatista": ["ERC", "JXCAT", "JUNTS", "EHB", "CUP"], "Regionalista": ["PNV", "CC", "BNG", "UPN", "NA+"]}` sobre las **siglas** del `PROV_02_*.json` (el constructor ajusta las siglas a las que usan esos mapas), `"baseline_scrutiny": "2023-07-23"`, `"split_source": "2023-07-23"`.

- [ ] **Step 1: Copiar los ficheros del INE y crear la carpeta**

Copiar `cera/cera-mensual/cerapropais_*.xls*` de la carpeta del artículo a `data/cera/ine/`; sustituir `cerapropais_26.xlsx` por el descargado (1-9-2026) y añadir las tres tablas `elecgen2026_tab{6,7,8}.xlsx` (`https://www.ine.es/oficina_censo/elecgen2026/Elecgral_2026_cc_tab{n}.xlsx`).

- [ ] **Step 2: Escribir el test de la carpeta**

```python
# tests/test_cera_data.py
def test_cera_data_folder_is_complete():
    root = os.path.join(ROOT, 'data', 'cera')
    assert len(glob.glob(os.path.join(root, 'ine', 'cerapropais_*.xls*'))) == 21
    assert all(os.path.exists(os.path.join(root, 'ine', f'elecgen2026_tab{n}.xlsx')) for n in (6, 7, 8))
    params = json.load(open(os.path.join(root, 'params.json'), encoding='utf-8'))
    assert params['election']['date'] == '2026-11-29' and params['check']['max_gain'] == 1
    assert params['participation']['latam'] == 0.05 and params['litigation']['count'] == 163020
    countries = pd.read_csv(os.path.join(root, 'countries.csv'))
    assert set(countries['region']) <= {'latam', 'europe', 'usa_canada', 'other'} and set(countries['exposure']) == {'alta', 'media', 'baja'}
    provinces = json.load(open(os.path.join(root, 'provinces.json'), encoding='utf-8'))
    assert set(provinces.values()) == set(range(1, 53))
    for date in ('2019-11-10', '2023-07-23'):
        frame = pd.read_csv(os.path.join(root, 'escrutinio', f'es-{date}.csv'))
        assert sorted(frame['region_id']) == list(range(1, 53)) and {'electors', 'voters', 'invalid', 'blank', 'PP', 'PSOE'} <= set(frame.columns)
    assert frame.set_index('region_id').loc[28, 'voters'] == 44807  # Madrid 2023
```

- [ ] **Step 3: Ejecutar el test y ver que falla** (`pytest tests/test_cera_data.py -v` → FAIL, ficheros ausentes)

- [ ] **Step 4: Construir los CSV y JSON** con un script desechable (no se versiona) y a mano donde la spec lo dice; añadir las filas de la tabla de `data/README.md` (una por fichero o grupo, con origen y licencia: INE, Ministerio del Interior, juntas electorales, curación propia).

- [ ] **Step 5: Ejecutar el test y ver que pasa**

- [ ] **Step 6: Commit** — `git add data/cera data/README.md tests/test_cera_data.py` · `Add the CERA study inputs: INE census workbooks, election tables, CERA scrutinies and parameters`

---

### Task 2: Las lecturas: `studies/cera/data.py`

**Files:**
- Create: `mtpy/lib/studies/__init__.py`, `mtpy/lib/studies/cera/__init__.py`, `mtpy/lib/studies/cera/data.py`
- Test: `tests/test_cera_data.py`

**Interfaces:**
- Produces (todas con docstring numpydoc):
  - `normalise_label(text: str) -> str`: mayúsculas, sin acentos (`unidecode`), solo `[A-Z0-9]`.
  - `read_provinces(data: FileSystem) -> dict[str, int]` (lee `cera/provinces.json`); `province_id(label: str, provinces: dict) -> int` (`KeyError` con la etiqueta normalizada).
  - `read_monthly_workbook(path: str, provinces: dict) -> pd.DataFrame` con columnas `MONTHLY_COLUMNS = ['date', 'region_id', 'country_code', 'country', 'electors']`: `date` es `pd.Timestamp` del día 1; una fila por provincia, país y mes; guiones y vacíos → 0; sin filas de totales. Lee los dos formatos (2006–2014 `.xls` con dos filas de cabecera «Censo cerrado» / «a 1-12-2006»; 2015+ `.xlsx` con «Censo cerrado a 1-may-2026»); los meses en español abreviados (`ene`…`dic`) y numéricos (`1-12-2006`, `1-1-2009`).
  - `read_monthly(data: FileSystem, provinces: dict, folder: str = 'cera/ine') -> pd.DataFrame`: concatena los libros en orden de nombre, ordena por fecha y **elimina duplicados de `(date, region_id, country_code)` conservando la última lectura** (el libro más reciente manda).
  - `national_series(long: pd.DataFrame) -> pd.Series` (índice `date`, suma de electores); `province_series(long) -> pd.DataFrame` (`date` × `region_id`).
  - `read_election_census(path: str, provinces: dict) -> pd.DataFrame` (`['region_id', 'country_code', 'country', 'electors']` de `tab8`; las filas de provincia son etiquetas en mayúsculas seguidas de sus países con código).
  - `read_scrutiny(path: str) -> pd.DataFrame`: índice `region_id`; columnas `electors`, `voters`, `invalid`, `blank` y las candidaturas; `NaN` → 0.
  - `read_candidatures_map(data: FileSystem) -> dict[str, dict[str, str]]`.
  - `scrutiny_votes(scrutiny: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame` (`region_id` × candidatura del XLSX; **`ValueError` que lista las columnas de candidatura sin entrada en el mapa**).
  - `Results` (dataclass): `date: str`, `votes: pd.DataFrame` (`region_id` × candidatura, nombres de la fila 3 del XLSX), `seats: pd.DataFrame` (misma forma), `totals: pd.DataFrame` (`region_id` × `registered`, `registered_cera`, `counted`, `counted_cera`, `valid`, `candidates`, `blank`, `invalid`, `n_seats`), `names: pd.Series` (`region_id` → nombre de provincia), `siglas: dict[str, str | None]` (del `PROV_02_*.json`).
  - `read_results(data: FileSystem, date: str) -> Results` desde `infoelectoral/es/PROV_02_{YYYYMM}_1.xlsx` (fila 5 cabeceras, fila 3 nombres de candidatura, pares de columnas «Votos» / «Diputados» desde la columna 15; filas de provincia = código 1–52; `n_seats` = suma de diputados de la fila).
  - `read_consulates(path: str) -> pd.DataFrame` (`country`, `electors`, `voters`), `read_autonomicas(path) -> pd.DataFrame`, `read_countries(path) -> pd.DataFrame`, `load_params(data: FileSystem) -> dict`.

- [ ] **Step 1: Tests de lectura**

```python
def test_normalise_label_strips_accents_and_punctuation():
    assert normalise_label('C.F. de Navarra') == 'CFDENAVARRA' and normalise_label('Araba/Álava') == 'ARABAALAVA'

def test_read_monthly_workbook_reads_both_formats(data_fs, provinces):
    old = read_monthly_workbook(path('cera/ine/cerapropais_06.xls'), provinces)
    new = read_monthly_workbook(path('cera/ine/cerapropais_26.xlsx'), provinces)
    assert list(old.columns) == MONTHLY_COLUMNS and old['date'].nunique() == 12 and new['date'].nunique() == 9
    assert national_series(old)['2006-12-01'] == 1160209 and national_series(new)['2026-05-01'] == 2708083
    assert national_series(new)['2026-09-01'] == 2762829 and set(new['region_id']) == set(range(1, 53))

def test_read_monthly_keeps_the_newest_reading_of_a_month(tmp_path, provinces):
    # dos libros con el mismo mes y cifras distintas: manda el último por nombre de fichero
    folder = tmp_path / 'cera' / 'ine'
    folder.mkdir(parents=True)
    shutil.copy(path('cera/ine/cerapropais_26.xlsx'), folder / 'cerapropais_26.xlsx')
    book = openpyxl.load_workbook(folder / 'cerapropais_26.xlsx')
    sheet = book[book.sheetnames[0]]
    sheet.cell(row=8, column=8).value = 999  # Albacete · Alemania · censo cerrado a 1-ene-2026 (fila 7 y columna 7 en base 0)
    book.save(folder / 'cerapropais_27.xlsx')
    long = read_monthly(FileSystem(str(tmp_path)), provinces)
    assert long[(long['date'] == '2026-01-01') & (long['region_id'] == 2) & (long['country'] == 'ALEMANIA')]['electors'].item() == 999
    assert len(long) == len(read_monthly_workbook(str(folder / 'cerapropais_26.xlsx'), provinces))

def test_read_monthly_series_has_245_months_to_may_2026(data_fs, provinces):
    series = national_series(read_monthly(data_fs, provinces))
    assert len(series[:'2026-05-01']) == 245 and series.index.min() == pd.Timestamp('2006-01-01')

def test_read_election_census_totals(data_fs, provinces):
    census = read_election_census(path('cera/ine/elecgen2026_tab8.xlsx'), provinces)
    assert census['electors'].sum() == 2769645 and set(census['region_id']) == set(range(1, 53))
    assert census[census['region_id'] == 28]['electors'].sum() == 494799

def test_read_results_2023_matches_the_official_totals(data_fs):
    results = read_results(data_fs, '2023-07-23')
    assert results.seats.values.sum() == 350 and results.totals['registered_cera'].sum() == 2328261
    assert results.votes.loc[28, 'PARTIDO POPULAR'] > 1_000_000 and results.siglas['PARTIDO POPULAR'] == 'PP'

def test_scrutiny_votes_maps_columns_and_rejects_unknown_ones(data_fs):
    scrutiny = read_scrutiny(path('cera/escrutinio/es-2023-07-23.csv'))
    votes = scrutiny_votes(scrutiny, read_candidatures_map(data_fs)['2023-07-23'])
    assert votes.loc[28, 'PARTIDO POPULAR'] == scrutiny.loc[28, 'PP'] and votes.values.sum() == 203115
    with pytest.raises(ValueError, match='NUEVA'):
        scrutiny_votes(scrutiny.assign(NUEVA=1), read_candidatures_map(data_fs)['2023-07-23'])
```

(`data_fs` = `FileSystem(os.path.join(ROOT, 'data'))`, `provinces` = `read_provinces(data_fs)`, `path()` = `data_fs.get_path`.) El 203.115 es la cifra de votos a candidaturas del artículo: si las columnas del escrutinio incluyen alguna que no sea candidatura, ajústese el mapa, no el test.

- [ ] **Step 2: Ejecutar y ver que fallan** (`ImportError`)

- [ ] **Step 3: Implementar `data.py`** con las firmas de arriba. El `.xls` se lee con `pd.read_excel(engine='xlrd')` (`xlrd` ya está instalado; añadir `xlrd==2.0.2` a `requirements.txt`). Las columnas de mes se detectan por regex `a 1-(\w+)-(\d{4})` sobre las dos filas de cabecera concatenadas.

- [ ] **Step 4: Ejecutar los tests y ver que pasan**

- [ ] **Step 5: Commit** — `Read the CERA study inputs: INE workbooks, election tables, scrutinies and Infoelectoral results with every candidature`

---

### Task 3: Serie y crecimiento: `census.py`

**Files:**
- Create: `mtpy/lib/studies/cera/census.py`
- Test: `tests/test_cera_census.py`

**Interfaces:**
- Consumes: `read_monthly`, `national_series`, `read_election_census`, `read_countries` (tarea 2).
- Produces:
  - `MARKS = [('2022-10-21', 'Ley 20/2022'), ('2023-07-23', 'Generales 2023'), ('2025-10-22', 'Fin del plazo'), ('2026-08-01', 'Censo del 29-N')]`.
  - `series_points(series: pd.Series) -> list[list]` (`[['2006-01-01', 1142334], ...]`).
  - `two_periods(long: pd.DataFrame, periods: list[tuple[str, str]], key: str, countries: pd.DataFrame | None = None) -> pd.DataFrame` con `key` ∈ `region_id | country_code | region | exposure` y columnas `[key, 'name', 'c1_start', 'c1_end', 'c2_start', 'c2_end', 'g1', 'g2', 'ratio']` (`ratio = g2 / g1`, `inf`/`NaN` → `None` al exportar); `name` = nombre de provincia (`es-provinces.csv`) o de país.
  - `update_periods(periods, last_date: pd.Timestamp) -> list[tuple]`: el segundo periodo termina en `last_date` y el primero se acorta por el principio para durar lo mismo.
  - `election_growth(long, census: pd.DataFrame, baseline: str, key: str, countries=None) -> pd.DataFrame` (`[key, 'name', 'baseline', 'election', 'growth', 'pct']`).
  - `baseline_date(series: pd.Series, target: int) -> pd.Timestamp`: la fecha mensual cuyo total nacional está más cerca de `target` (2.328.261).
  - `trend_break(series: pd.Series, break_date: str = '2022-10-01') -> dict` con `slope_before`, `slope_after`, `net_before`, `net_after` (altas medias mensuales en los tramos nov-2019→jul-2023 y jul-2023→último), `p_slope`, `p_level` (OLS de las altas netas mensuales sobre tendencia, dummy de ruptura e interacción, `cov_type='HAC', cov_kwds={'maxlags': 12}`).

- [ ] **Step 1: Tests**

```python
def test_two_periods_reproduces_the_article_growth(long):
    by_prov = two_periods(long, PERIODS_ARTICLE, 'region_id')
    assert by_prov['g2'].sum() == 422253 and by_prov['g1'].sum() == 189290 and round(by_prov['g2'].sum() / by_prov['g1'].sum(), 1) == 2.2
    assert by_prov.set_index('region_id').loc[28, 'g2'] == 124195
    by_country = two_periods(long, PERIODS_ARTICLE, 'country_code', countries)
    assert by_country.set_index('name').loc['ARGENTINA', 'g2'] == 73855 and by_country.set_index('name').loc['MÉXICO', 'g2'] == 54954

def test_two_periods_by_origin_and_exposure(long, countries):
    origins = two_periods(long, PERIODS_ARTICLE, 'region', countries).set_index('region')
    assert origins.loc['latam', 'g2'] > origins.loc['latam', 'g1'] * 3
    exposure = two_periods(long, PERIODS_ARTICLE, 'exposure', countries)
    assert set(exposure['exposure']) == {'alta', 'media', 'baja'}

def test_update_periods_keeps_equal_lengths():
    assert update_periods([('2019-03-01', '2022-10-01'), ('2022-10-01', '2026-05-01')], pd.Timestamp('2026-09-01')) == [(pd.Timestamp('2019-07-01'), pd.Timestamp('2022-10-01')), (pd.Timestamp('2022-10-01'), pd.Timestamp('2026-09-01'))]

def test_election_growth_against_the_convocation_census(long, census, countries):
    base = baseline_date(national_series(long), 2328261)
    assert abs(national_series(long)[base] - 2328261) / 2328261 < 0.005
    growth = election_growth(long, census, base, 'region_id')
    assert growth['election'].sum() == 2769645 and growth.set_index('region_id').loc[28, 'election'] == 494799

def test_trend_break_is_significant(long):
    out = trend_break(national_series(long))
    assert out['p_slope'] < 0.05 and out['p_level'] < 0.05 and out['net_after'] > 2 * out['net_before']
```

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar `census.py`**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Compute the CERA census series, the two-period growth and the trend break`

---

### Task 4: D'Hondt validado y el último escaño: `seats.py` (1/4)

**Files:**
- Create: `mtpy/lib/studies/cera/seats.py`
- Test: `tests/test_cera_seats.py`

**Interfaces:**
- Consumes: `Results` (tarea 2), `Simulator.alloc_seats` (`mtpy/lib/simulator.py:1232`).
- Produces:
  - `THRESHOLD = 3.0`.
  - `allocate_results(results: Results, votes: pd.DataFrame | None = None) -> pd.DataFrame` (`region_id` × candidatura, escaños; `valid_votes` = `candidates + blank` de `totals`, recalculado cuando se pasan `votes` distintos: `votes.sum(axis=1) + blank`).
  - `check_official(results: Results) -> list[str]`: desajustes `'{provincia}: {candidatura} {calculado} != {oficial}'`; vacía si cuadra.
  - `last_seat(results: Results, votes: pd.DataFrame | None = None) -> pd.DataFrame` (`region_id`, `name`, `magnitude`, `winner`, `runner_up`, `gap`): `winner` = candidatura del menor cociente asignado, `runner_up` = la de mayor cociente no asignado, `gap` = votos que le faltan al `runner_up` para superar ese cociente (`ceil(q_min · (s_r + 1)) − v_r + 1`, mínimo 1).
  - `block_seats(seats: pd.DataFrame, siglas: dict, blocks: dict) -> dict[str, int]` (suma por bloque; candidaturas sin sigla o sin bloque → `'Otros'`).
  - `participation_history(results: dict[str, Results]) -> pd.DataFrame` (`region_id`, `mean`, `max`: participación CERA provincial, `counted_cera / registered_cera` en %, media y máximo sobre las elecciones de `results` que traen las dos columnas).
  - El fixture de módulo `study_inputs` de `tests/test_cera_seats.py` (`scope='module'`, un `SimpleNamespace`): `data_fs`, `params`, `countries`, `provinces`, `long`, `series`, `baseline_date`, `baseline` (censo provincia × país en esa fecha, columnas `region_id, country_code, electors`), `census` (tab8), `census_by_province`, `results_2023`, `results_2019`, `cera_2023`, `cera_2019` (`scrutiny_votes`), `siglas` (de 2023), `siglas_2019`, `blocks`, `split` (cuotas por provincia de `cera_2023` sobre su suma), `magnitudes` (`results_2023.totals['n_seats']`), `hist` (`participation_history` de 2019 y 2023). Las tareas 5–8 lo reutilizan.

- [ ] **Step 1: Tests**

```python
def test_dhondt_reproduces_the_official_350_seats(results_2023, results_2019):
    assert check_official(results_2023) == [] and check_official(results_2019) == []

def test_last_seat_gaps_match_table_10(results_2023):
    last = last_seat(results_2023).set_index('name')
    assert last.loc['Girona', 'gap'] == 284 and last.loc['Cantabria', 'gap'] == 403 and last.loc['Tarragona', 'gap'] == 1247
    assert last.loc['Madrid', 'gap'] == 1339 and last.loc['Ceuta', 'gap'] == 1586 and last.loc['Málaga', 'gap'] == 2428
    assert last.loc['Cantabria', 'winner'].startswith('VOX') and last.loc['Cantabria', 'runner_up'].startswith('PARTIDO POPULAR')
    assert len(last) == 52 and (last['magnitude'] >= 1).all()

def test_block_seats_2023(results_2023, params):
    blocks = block_seats(results_2023.seats, results_2023.siglas, params['blocks'])
    assert blocks['Izquierda'] == 152 and blocks['Derecha'] == 170
```

(Los nombres de provincia del test son los de `results.names`; si el XLSX escribe `Gerona`, el test usa esa forma.)

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Allocate the Congress seats of the CERA study with every candidature and measure the last-seat gap`

---

### Task 5: El modelo de incremento y los dos censos: `seats.py` (2/4)

**Files:**
- Modify: `mtpy/lib/studies/cera/seats.py`
- Test: `tests/test_cera_seats.py`

**Interfaces:**
- Consumes: tarea 4; `read_election_census`, `scrutiny_votes`, `province_series` (tarea 2); `baseline_date` (tarea 3).
- Produces:
  - `new_electors(census: pd.DataFrame, baseline: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame` (`region_id`, `origin` ∈ `latam | rest`, `electors`; `census` y `baseline` con columnas `region_id, country_code, electors`; diferencias negativas se mantienen).
  - `projected_census(baseline: pd.DataFrame, countries, factor_latam: float, factor_rest: float) -> pd.DataFrame` (el censo proyectado del artículo: ×1,264 y ×1,194 por origen).
  - `litigation_census(census: pd.DataFrame, long: pd.DataFrame, countries, count: int, since: str = '2022-10-01') -> pd.DataFrame`: resta `count` electores de las filas latinoamericanas de cada provincia en proporción a su crecimiento latinoamericano desde `since`; con `count == 0` devuelve el censo igual.
  - `increment_votes(new: pd.DataFrame, participation: dict, split: pd.DataFrame, valid: float = 0.99) -> pd.DataFrame` (`region_id` × candidatura): `electors · rate[origin] · valid` repartidos por las cuotas de `split` (votos del escrutinio CERA de 2023 de esa provincia sobre su total de candidaturas).
  - `increment_model(results: Results, extra: pd.DataFrame, siglas: dict, blocks: dict) -> dict` con `seats_before`, `seats_after` (DataFrames), `by_block` (`{block: {'before', 'after', 'change'}}`), `by_sigla` (igual), `changed` (lista de `region_id`), `votes_extra` (suma de `extra`).
  - `net_effect(last: pd.DataFrame, extra: pd.DataFrame, after: pd.DataFrame, before: pd.DataFrame) -> pd.DataFrame`: `last` + `d_runner`, `d_winner`, `net = d_runner − d_winner`, `flipped` (bool), `participation_needed` (factor por el que habría que multiplicar la participación para que `net` alcance `gap`; `None` si `net ≤ 0`).

- [ ] **Step 1: Tests**

```python
def test_article_projection_adds_540k_electors_and_37k_votes_and_moves_nothing(study_inputs):
    projected = projected_census(baseline, countries, 1.264, 1.194)
    new = new_electors(projected, baseline, countries)
    assert 530_000 <= new['electors'].sum() <= 550_000
    extra = increment_votes(new, {'latam': 0.05, 'rest': 0.10}, split)
    assert 35_000 <= extra.values.sum() <= 40_000
    out = increment_model(results_2023, extra, results_2023.siglas, params['blocks'])
    assert out['changed'] == [] and out['by_block']['Izquierda']['change'] == 0

def test_real_census_moves_nothing_in_both_scenarios(study_inputs):
    new = new_electors(census, baseline, countries)
    assert 430_000 <= new['electors'].sum() <= 450_000
    extra = increment_votes(new, params['participation'], split)
    assert increment_model(results_2023, extra, siglas, blocks)['changed'] == []
    reduced = litigation_census(census, long, countries, 163020)
    assert census['electors'].sum() - reduced['electors'].sum() == 163020
    extra_b = increment_votes(new_electors(reduced, baseline, countries), params['participation'], split)
    assert increment_model(results_2023, extra_b, siglas, blocks)['changed'] == [] and extra_b.values.sum() < extra.values.sum()
    assert litigation_census(census, long, countries, 0).equals(census)

def test_net_effect_signs(study_inputs):
    net = net_effect(last_seat(results_2023), extra, out['seats_after'], out['seats_before']).set_index('name')
    assert net.loc['Cantabria', 'net'] > 0 and net.loc['Madrid', 'net'] < 0 and not net['flipped'].any()
```

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Project the CERA vote of the 29-N over the 2023 results with two census scenarios`

---

### Task 6: Sensibilidad y colocación óptima: `seats.py` (3/4)

**Files:**
- Modify: `mtpy/lib/studies/cera/seats.py`
- Test: `tests/test_cera_seats.py`

**Interfaces:**
- Produces:
  - `peninsular_results(results: Results, cera: pd.DataFrame) -> Results` (votos menos el voto CERA por candidatura; `totals` con `candidates`/`valid` ajustados).
  - `exterior_grid(results: Results, cera_2023: pd.DataFrame, census_by_province: pd.Series, participations: list[float], left_fracs: list[float], siglas, blocks) -> pd.DataFrame` (índice participación, columnas fracción; valor = cambio neto de escaños de `Izquierda`): voto exterior completo = censo · participación; la fracción va al bloque `Izquierda` y el resto a `Derecha`, repartidos dentro de cada bloque como el CERA de 2023 de la provincia; separatistas, regionalistas y otros conservan su cuota del CERA de 2023 (definición 4 de la revisión).
  - `scenario_table(results, baseline, census, countries, participation, split, siglas, blocks, scenarios: list[dict]) -> pd.DataFrame` con filas `{'census_x', 'part_x', 'bias'}` (`bias` ∈ `'empirico' | 0.7 | 0.9`: fracción del voto nuevo al bloque de izquierda, repartida dentro del bloque como el split) y columnas `d_left`, `d_right`.
  - `placement_pool(long: pd.DataFrame, census: pd.DataFrame, countries, codes: list[str], since: str) -> pd.Series` (`region_id` → altas de los cuatro países desde `since` hasta el censo de la convocatoria).
  - `optimal_placement(results: Results, pool: pd.Series, participations: list[float], fracs: list[float], candidature: str, step: int = 500) -> dict` con `grid` (DataFrame participación × fracción, escaños que gana `candidature`) y `contrast` (el pool en sus provincias reales, participación y reparto observados → cambio de escaños, que debe ser 0). Algoritmo: retirar el pool de sus provincias; votos útiles = `pool.sum() · participation · frac`; asignar en bloques de `step` votos a la provincia donde `candidature` necesita menos votos para el siguiente escaño (coste marginal D'Hondt recalculado tras cada bloque) hasta agotar; contar escaños ganados frente al reparto base.
  - `report_replication(results: Results, census_by_province: pd.Series, hist: pd.DataFrame) -> dict` (`hist`: `region_id`, `mean`, `max` participación CERA histórica): `votes_mean`, `votes_max`, `n_mean`, `n_max` (provincias donde los votos superan la brecha).

- [ ] **Step 1: Tests**

```python
def test_exterior_grid_is_monotone_and_zero_at_the_observed_cell_with_real_split(study_inputs):
    grid = exterior_grid(results_2023, cera_2023, census_by_province, [0.09, 0.2, 0.3, 0.5, 0.7], [0.5, 0.6, 0.7, 0.8], siglas, blocks)
    assert grid.shape == (5, 4) and (grid.diff().fillna(0) >= 0).all().all() and (grid.T.diff().fillna(0) >= 0).all().all()
    assert grid.loc[0.09, 0.5] <= 1 and grid.loc[0.7, 0.8] >= 10

def test_scenario_table_rows(study_inputs):
    s = study_inputs
    table = scenario_table(s.results_2023, s.baseline, s.census, s.countries, s.params['participation'], s.split, s.siglas, s.blocks,
                           scenarios=[{'census_x': None, 'part_x': 1, 'bias': 'empirico'}, {'census_x': 2.0, 'part_x': 3, 'bias': 0.9}])
    assert table.iloc[0]['d_left'] == 0 and table.iloc[0]['d_right'] == 0 and table.iloc[1]['d_left'] >= 3

def test_optimal_placement_grows_with_the_levers_and_is_zero_without_placement(study_inputs):
    pool = placement_pool(long, census, countries, params['placement_countries'], '2022-10-01')
    assert 150_000 <= pool.sum() <= 200_000
    out = optimal_placement(results_2023, pool, [0.09, 0.2, 0.5, 1.0], [0.5, 0.6, 0.7, 0.8, 1.0], PSOE)
    grid = out['grid']
    assert grid.loc[0.09, 0.5] <= 2 and grid.loc[1.0, 1.0] >= grid.loc[0.5, 0.8] >= grid.loc[0.09, 0.5] and out['contrast'] == 0

def test_report_replication_counts_provinces(study_inputs):
    out = report_replication(study_inputs.results_2023, study_inputs.census_by_province, study_inputs.hist)
    assert out['n_max'] >= out['n_mean'] >= 9 and out['votes_max'] > out['votes_mean']
```

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar** (`optimal_placement` con `step` configurable; el constructor comprueba que con `step=500` tarda menos de 30 s).
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Stress the CERA projection: participation and bias grid, three-dimension scenarios and optimal placement`

---

### Task 7: La comprobación con y sin CERA: `seats.py` (4/4)

**Files:**
- Modify: `mtpy/lib/studies/cera/seats.py`
- Test: `tests/test_cera_seats.py`

**Interfaces:**
- Produces:
  - `with_without(results: Results, cera: pd.DataFrame, siglas: dict, blocks: dict) -> dict`: `rows` (una por provincia: `region_id`, `name`, `magnitude`, `winner_with`, `winner_without`, `gap` (brecha del reparto oficial), `voters_cera`, `over_gap` (bool), `decided` (bool: cambia algún escaño), `crosses_block` (bool)), `net_by_sigla` (`{sigla: escaños con − sin}`), `net_by_block`, `n_over_gap`, `n_decided`, `decided` (lista de `region_id`).
  - `verdict(net_psoe: int, net_left: int, max_gain: int) -> str` → `'acierto'` si `max(net_psoe, net_left) <= max_gain`, `'fallo'` si no.

- [ ] **Step 1: Tests**

```python
def test_with_without_cera_2023_madrid_and_nine_provinces(results_2023, cera_2023, siglas, blocks):
    out = with_without(results_2023, cera_2023, siglas, blocks)
    rows = out['rows'].set_index('name')
    assert out['n_over_gap'] == 9 and out['decided'] == [28]
    assert rows.loc['Madrid', 'winner_with'].startswith('PARTIDO POPULAR') and rows.loc['Madrid', 'winner_without'].startswith('PARTIDO SOCIALISTA')
    assert out['net_by_sigla']['PSOE'] == -1 and out['net_by_sigla']['PP'] == 1 and out['net_by_block']['Izquierda'] == -1
    assert rows.loc['Madrid', 'crosses_block'] and rows.loc['Girona', 'over_gap'] and not rows.loc['Girona', 'decided']

def test_with_without_cera_2019_bizkaia(results_2019, cera_2019, siglas_2019, blocks):
    out = with_without(results_2019, cera_2019, siglas_2019, blocks)
    assert out['decided'] == [48] and out['net_by_sigla']['PP'] == 1 and out['net_by_sigla']['PNV'] == -1

def test_verdict_threshold():
    assert verdict(-1, -1, 1) == verdict(0, 0, 1) == verdict(1, 1, 1) == 'acierto' and verdict(2, 0, 1) == verdict(0, 2, 1) == 'fallo'
```

(Si la sigla del PNV en `PROV_02_201911_1.json` es otra, el test usa esa.)

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Check whether the CERA vote decided any seat, with and without it, and the verdict rule`

---

### Task 8: Participación, voto y colocación: `behaviour.py`

**Files:**
- Create: `mtpy/lib/studies/cera/behaviour.py`
- Test: `tests/test_cera_behaviour.py`

**Interfaces:**
- Consumes: tareas 2–4.
- Produces:
  - `ORIGINS = ('europe', 'usa_canada', 'latam', 'other')`; `participation_by_origin(consulates: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame` (`origin`, `electors`, `voters`, `rate` en %); `participation_processes(data: FileSystem, countries) -> dict[str, pd.DataFrame]` (uno por fichero de `cera/consulados/`).
  - `participation_generales(results: dict[str, Results]) -> pd.DataFrame` (`date`, `electors`, `voters`, `rate`, `period` ∈ `antes | rogado | sin_rogado`: `rogado` entre `2011-03-28` y `2022-10-01`); las elecciones cuyo XLSX no trae `registered_cera`/`counted_cera` se omiten; `read_results` se intenta para todas las fechas con fichero `PROV_02_*` en `data/infoelectoral/es/` (función `available_results(data) -> list[str]`).
  - `shares_by_block(votes: pd.DataFrame, siglas: dict, blocks: dict) -> pd.DataFrame` (`region_id` × bloque, con `'Otros'`); `left_share(block_votes: pd.DataFrame) -> float` (`izq / (izq + der)` en %, sobre la suma nacional).
  - `generales_processes(cera: dict[str, pd.DataFrame], results: dict[str, Results], blocks) -> list[dict]` (por elección: `date`, `left_share`, `margin` (izq − der en puntos sobre izq+der), `psoe_pp_cera`, `psoe_pp_total`, `premium`).
  - `representation_ratios(cera_votes: pd.DataFrame, results: Results, siglas: dict, main: list[str]) -> pd.DataFrame` (`sigla`, `share_cera`, `share_total`, `ratio` en %; sobre votos a candidaturas).
  - `autonomic_processes(autonomicas: pd.DataFrame) -> list[dict]` (por comunidad: anterior y último con `date`, `electors`, `voters`, `rate`, `pp_pct`, `psoe_pct`, `vox_pct`, `other_left_pct`, `left_share`, `margin`, y los cambios `d_psoe`, `d_other_left`, `d_block`).
  - `ecological(cera_votes: pd.DataFrame, siglas, blocks, long: pd.DataFrame, at: pd.Timestamp, countries, names: pd.Series, min_census: int, only: list[int] | None = None) -> dict` (`rows` = `[{region_id, name, latam_share, left_share, census}]`, `r`, `slope`, `quartiles` = cuotas de izquierda medias por cuartil de `latam_share`, `n`).
  - `sensitive_share(long, countries, codes: list[str], magnitudes: pd.Series, start: str, end: pd.Timestamp, max_magnitude: int = 5) -> dict` (`stock_share`, `growth_share` en %).
  - `acceleration(two_periods_by_province: pd.DataFrame, magnitudes: pd.Series, gaps: pd.Series) -> dict` (`rows` con `ratio`, `magnitude`, `g2`, `gap`; `r_size_growth`, `r_size_ratio`, `r_gap_growth`, `share_sensitive_growth`, `share_sensitive_stock`).
  - `madrid_magnet(long, countries, codes: list[str], periods: list[tuple]) -> list[dict]` (`{'label', 'stock_share', 'share_p1', 'share_p2'}` para el conjunto y para México y Venezuela).

- [ ] **Step 1: Tests**

```python
def test_participation_by_origin_cyl_2026(data_fs, countries):
    rates = participation_by_origin(read_consulates(path('cera/consulados/cyl-2026.csv')), countries).set_index('origin')['rate']
    assert round(rates['europe'], 1) == 11.5 and round(rates['usa_canada'], 1) == 7.6 and round(rates['latam'], 1) == 5.0

def test_participation_generales_periods(results_all):
    table = participation_generales(results_all).set_index('date')
    assert round(table.loc['2023-07-23', 'rate'], 1) == 10.0 and table.loc['2019-11-10', 'period'] == 'rogado' and table.loc['2023-07-23', 'period'] == 'sin_rogado'

def test_representation_ratios_2023(study_inputs):
    ratios = representation_ratios(cera_2023, results_2023, siglas, ['PP', 'PSOE', 'SUMAR', 'VOX']).set_index('sigla')['ratio'].round()
    assert ratios['SUMAR'] == 120 and ratios['PP'] == 99 and ratios['PSOE'] == 92 and ratios['VOX'] == 90

def test_generales_left_share(study_inputs):
    rows = {row['date']: row for row in generales_processes(cera_all, results_all, blocks)}
    assert round(rows['2019-11-10']['left_share'], 1) == 55.2 and round(rows['2023-07-23']['left_share'], 1) == 50.0

def test_autonomic_processes_cyl(data_fs):
    cyl = [row for row in autonomic_processes(read_autonomicas(path('cera/autonomicas.csv'))) if row['community'] == 'Castilla y León'][0]
    assert cyl['last']['rate'] == 6.7 and round(cyl['last']['left_share'], 1) == 48.2 and round(cyl['previous']['left_share'], 1) == 51.7

def test_ecological_is_negative_over_all_provinces_but_ceuta_and_melilla(study_inputs):
    out = ecological(cera_2023, siglas, blocks, long, baseline, countries, results_2023.names, params['ecological_min_census'])
    assert out['n'] == 50 and out['r'] < -0.5 and out['slope'] < 0
    assert ecological(..., min_census=params['ecological_min_census'], only=ARTICLE_43)['r'] == pytest.approx(-0.77, abs=0.01)

def test_sensitive_share_article_periods(study_inputs):
    out = sensitive_share(long, countries, ['ARGENTINA', 'CUBA', 'VENEZUELA'], magnitudes, '2022-10-01', pd.Timestamp('2026-05-01'))
    assert round(out['stock_share'], 1) == 24.0 and round(out['growth_share'], 1) == 17.3
```

(`only` es un parámetro opcional de `ecological`: lista de `region_id` a usar, solo para la regresión del artículo; `ARTICLE_43` son las 52 menos Ceuta, Melilla, La Rioja, Navarra, Baleares, Murcia, Las Palmas, A Coruña y Madrid. Si el 24,0/17,3 no sale exacto por la definición del stock, el constructor fija el valor obtenido y lo anota en la spec.)

- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Measure who votes from abroad, whom they vote for and where the new electors register`

---

### Task 9: La orquestación: `study.py`

**Files:**
- Create: `mtpy/lib/studies/cera/study.py`
- Modify: `mtpy/lib/studies/cera/__init__.py` (re-exporta `compute_cera`, `CeraStudy`)
- Test: `tests/test_cera_seats.py` (un test de integración)

**Interfaces:**
- Produces: `@dataclass CeraStudy` con los resultados de las tareas 3–8 como atributos nombrados (`params`, `countries`, `long`, `series`, `series_last`, `marks`, `periods_article`, `periods_now`, `growth` (dict de DataFrames por clave y periodo), `election_growth` (dict), `regression`, `participation` (dict), `generales`, `autonomics`, `ratios`, `ecological`, `sensitive`, `acceleration`, `madrid`, `results` (dict fecha → Results), `cera` (dict fecha → votos por candidatura), `last`, `scenarios` (dict `convocatoria | sin_litigio` → `{census, new, extra, model, net}`), `grid`, `scenario_rows`, `placement`, `report`, `checks` (dict fecha → `with_without`), `verdict_input`), y `compute_cera(data: FileSystem, params: dict | None = None) -> CeraStudy` que lee todo (`params` por defecto de `load_params`), calcula en el orden de la spec y falla con `ValueError('cera: D'Hondt does not reproduce ...')` si `check_official` de 2023 no está vacío. Las elecciones de `checks` son las que tienen fichero en `cera/escrutinio/` y XLSX de Infoelectoral.

- [ ] **Step 1: Test** `test_compute_cera_runs_end_to_end(data_fs)`: `study = compute_cera(data_fs)`; `study.series_last == pd.Timestamp('2026-09-01')`; `set(study.scenarios) == {'convocatoria', 'sin_litigio'}`; `study.scenarios['convocatoria']['model']['changed'] == []`; `sorted(study.checks) == ['2019-11-10', '2023-07-23']`; `study.checks['2023-07-23']['n_over_gap'] == 9`. Marcar con `@pytest.mark.slow` si supera 60 s y documentarlo en `pytest.ini`.
- [ ] **Step 2: Ejecutar y ver que falla**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasa**
- [ ] **Step 5: Commit** — `Assemble the CERA study from its inputs in one call`

---

### Task 10: Los estudios en el paquete: `bundle.py` y el contrato

**Files:**
- Modify: `mtpy/lib/bundle.py` (constantes, `SCHEMAS`, `path_study`, `path_study_csv`, `BundleReader.study_exists`), `mtpy/lib/publish/pointers.py` (`update_manifest(..., studies=None)`), `docs/web/contrato.md` (sección «Estudios» tras «Eventos»; `manifest.json` con `studies`)
- Test: `tests/test_bundle_unit.py`, `tests/test_publish_unit.py` (manifest)

**Interfaces:**
- Produces: `STUDIES = ('voto-cera',)`, `STUDY_PARTS = ('meta', 'census', 'participation', 'vote', 'placement', 'prediction', 'sensitivity', 'check')`, `STUDY_SCHEMAS = {part: 'study-' + part ...}`, `STUDY_CSV = ('census-series', 'census-provinces', 'census-countries', 'participation', 'vote', 'ecological', 'last-seat', 'sensitivity-grid', 'sensitivity-scenarios', 'placement-grid', 'check')`, `STUDY_CSV_OF = {'census': ('census-series', 'census-provinces', 'census-countries'), 'participation': ('participation',), 'vote': ('vote', 'ecological'), 'prediction': ('last-seat',), 'sensitivity': ('sensitivity-grid', 'sensitivity-scenarios', 'placement-grid'), 'check': ('check',)}` (el primero de cada tupla es el CSV por defecto de la parte), `path_study(study, part) -> 'studies/{study}/{part}.json'`, `path_study_csv(study, name) -> 'studies/{study}/csv/{name}.csv'` (`ValueError` con estudio o nombre desconocidos, vía `_check_file` con `STUDIES`), `BundleReader.study_exists(study) -> bool`.
- `SCHEMAS` nuevos (claves obligatorias y tipos):
  - `study-meta`: `study: str, title: str, subtitle: str, election: dict, registered_at: NULLABLE_STR, registered_commit: NULLABLE_STR, registered: (dict, NoneType), registered_changed_at: NULLABLE_STR, checked_at: NULLABLE_STR, params: dict, sources: list, series_last: str, regression: dict, commit: NULLABLE_STR, dirty: bool, versions: dict, seconds: dict`
  - `study-census`: `series: dict, marks: list, periods: dict, provinces: list, countries: list, origins: list, exposure: list, election_provinces: list, election_countries: list`
  - `study-participation`: `origin: list, generales: list, autonomicas: list`
  - `study-vote`: `processes: list, ratios: list, ecological: dict`
  - `study-placement`: `sensitive: dict, acceleration: list, correlations: dict, madrid: list`
  - `study-prediction`: `scenarios: dict, last_seat: list, report: dict`
  - `study-sensitivity`: `grid: dict, scenarios: list, placement: dict`
  - `study-check`: `elections: list, rows: list, verdict: NULLABLE_STR`
- `update_manifest(writer, scopes=None, freeze=None, today=None, studies=None)`: `data['studies'] = {**data.get('studies', {}), **(studies or {})}`; el manifiesto siempre lleva la clave `studies` (`{}` por defecto); el schema `manifest` gana `'studies': dict`.

- [ ] **Step 1: Tests** (`tests/test_bundle_unit.py`): `test_study_paths_follow_the_published_layout_and_reject_bad_input` (`path_study('voto-cera', 'meta') == 'studies/voto-cera/meta.json'`, `path_study_csv('voto-cera', 'last-seat') == 'studies/voto-cera/csv/last-seat.csv'`, `ValueError` con `'nope'` y con una parte desconocida), `test_study_exists_needs_the_meta`, `test_study_schemas_validate_the_fixtures` (cada `tests/fixtures/bundle/study-*.json` pasa `validate`; los fixtures se crean en la tarea 12, así que este test se añade allí), y en `tests/test_publish_unit.py`: `test_update_manifest_merges_studies` (`update_manifest(writer, studies={'voto-cera': {'published_at': ..., 'registered_at': None, 'checked': False}})` conserva `scopes` y guarda `studies`).
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar** y escribir la sección del contrato: árbol `studies/{study}/{parte}.json` + `csv/`, la tabla de partes con sus claves, los CSV y el `manifest.studies`.
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Add the studies to the bundle: parts, schemas, paths and the manifest entry`

---

### Task 11: La publicación: `publish/studies.py` y el job

**Files:**
- Create: `mtpy/lib/publish/studies.py`
- Modify: `mtpy/lib/publish/__init__.py`, `mtpy/jobs/Publish.py` (`WHAT`, parámetros `study`, `register`; `_study_scope`, `_study_line`)
- Test: `tests/test_publish_studies.py`, `tests/test_jobs_publish.py`

**Interfaces:**
- Consumes: `CeraStudy`, `compute_cera` (tarea 9); `provenance`, `REPO_ROOT`, `NothingToPublish`, `round_cols`, `records` (`publish/common.py`); tarea 10.
- Produces:
  - `STUDY_TITLES = {'voto-cera': ('El voto CERA en España y el efecto de la «Ley de Nietos»', '¿Es posible orquestar con éxito un fraude electoral significativo a través del censo exterior?')}`.
  - Un exportador por parte, `export_study_{part}(study: CeraStudy) -> tuple[dict, dict[str, pd.DataFrame]]` (`data` según el schema; los frames por nombre de CSV de `STUDY_CSV_OF`). Contenido mínimo que las páginas usan (tarea 14), redondeos a `PCT_DECIMALS` para porcentajes y enteros para electores, votos y escaños:
    - `census`: `series = {'dates': [...], 'values': [...]}` (nacional), `marks = [{'date', 'label'}]`, `periods = {'article': [[start, end], [start, end]], 'now': [...]}`, `provinces`/`countries`/`origins`/`exposure` = filas de `two_periods` con las dos versiones (`g1`, `g2`, `ratio` del artículo y `g1_now`, `g2_now`, `ratio_now`), `election_provinces`/`election_countries` = filas de `election_growth`.
    - `participation`: `origin = [{'process', 'origin', 'electors', 'voters', 'rate'}]`, `generales = [{'date', 'electors', 'voters', 'rate', 'period'}]`, `autonomicas` = `autonomic_processes`.
    - `vote`: `processes` = generales + autonómicas con `left_share`, `margin`, `d_psoe`, `d_other_left`, `d_block`, `premium`; `ratios`; `ecological = {'rows', 'r', 'slope', 'quartiles', 'n', 'min_census'}`.
    - `placement`: `sensitive = {'stock_share', 'growth_share', 'period'}`, `acceleration` (filas), `correlations`, `madrid`.
    - `prediction`: `scenarios = {name: {'census', 'new_electors', 'voters_2023', 'voters_projected', 'votes_2023', 'votes_projected', 'by_block': [...], 'by_sigla': [...], 'changed': [...]}}` para `convocatoria` y `sin_litigio`; `last_seat` = filas de `net_effect` del escenario `convocatoria` (y `net_b` del otro); `report = {'votes_mean', 'votes_max', 'n_mean', 'n_max', 'participation_mean', 'participation_max'}`.
    - `sensitivity`: `grid = {'participations', 'left_fracs', 'values'}`, `scenarios` (filas de `scenario_table`), `placement = {'participations', 'fracs', 'values', 'contrast', 'pool'}`.
    - `check`: `elections = [{'date', 'voters', 'rate', 'n_over_gap', 'n_decided', 'decided': [{'region_id', 'name', 'winner_with', 'winner_without', 'crosses_block'}], 'net_psoe', 'net_left'}]`, `rows` = filas de `with_without` de la última elección con escrutinio, `verdict` = `verdict(...)` de la elección del estudio si tiene escrutinio, `None` si no.
    - `meta`: lo de la spec; `sources = [{'label', 'url'}]`; `election = {'scope', 'date', 'census_closed', 'census_total', 'census_source'}`; `registered = {'claim': 'El voto CERA del 29-N no dará al PSOE ni a la izquierda un saldo neto de más de {max_gain} escaños', 'max_gain', 'seats': {escenario: by_block}}`.
  - `publish_study(study_id: str, writer: BundleWriter, register: bool = False, force: bool = False, data: FileSystem | None = None, compute: Callable | None = None, prov: Callable | None = None, today=None) -> dict` (`{'study', 'registered_at', 'checked', 'seconds'}`): `study_id` fuera de `bundle.STUDIES` → `ValueError`; lee el `meta` previo si existe (`writer.read_json`) para heredar `registered_*`; con `register=True`: si `prov()['dirty']` → `PublishRefused('study: register needs a clean tree')`; si ya había `registered_at` y no `force` → `PublishRefused('study: already registered on ...')`; con `force` escribe `registered_changed_at = hoy`; escribe las partes en el orden de `STUDY_PARTS[1:]` con sus CSV y `meta` la última; `checked_at` = hoy cuando `check.verdict` no es `None` (y se hereda si ya estaba); actualiza el manifiesto (`update_manifest(writer, studies={study_id: {'published_at', 'registered_at', 'checked': bool}})`).
  - `Publish.run(..., what='study', study='voto-cera', register=False, ...)`: `WHAT` gana `'study'`; corre tras `backtest` y antes de `manifest`; `_study_scope(study, writer, register, force)` devuelve `published | refused | failed` con `reason`; `_study_line`. Sin guardia LOREG.

- [ ] **Step 1: Tests** (`tests/test_publish_studies.py`, con un `FakeStudy` o un `CeraStudy` mínimo construido a mano en `tests/fakes.py` (`synthetic_cera()`), sin leer `data/`):
  - `test_publish_study_writes_the_parts_in_order_and_meta_last` (ocho JSON, once CSV, `meta` último, `study_exists`).
  - `test_publish_study_does_not_register_by_default` (`registered_at is None`, sin píldora de registro en `manifest.studies`).
  - `test_publish_study_registers_only_with_a_clean_tree` (`register=True` + `fake_prov` con `dirty=True` → `PublishRefused`, nada escrito; con `dirty=False` → `registered_at` del reloj del writer y `registered['max_gain'] == 1`).
  - `test_publish_study_keeps_the_registration_and_refuses_to_register_twice` (segunda publicación sin `register` hereda; con `register` → `PublishRefused`; con `force` → `registered_changed_at`).
  - `test_publish_study_sets_checked_at_when_the_election_has_a_scrutiny`.
  - `test_export_study_parts_validate` (cada exportador pasa `validate` con su schema).
  - `tests/test_jobs_publish.py`: `test_publish_study_runs_the_study_action` (`Publish().run(what='study', study='voto-cera', fs=fs)` con `compute` parcheado → `results['voto-cera']['status'] == 'published'`; `what='study'` sin `study` → `ValueError`).
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Publish the CERA study as a bundle study, with an explicit registration of its prediction`

---

### Task 12: La API de los estudios y los fixtures

**Files:**
- Create: `mtpy/controllers/Studies.py`, `tests/fixtures/bundle/study-{meta,census,participation,vote,placement,prediction,sensitivity,check}.json`
- Modify: `mtpy/lib/webapi/checks.py`, `mtpy/lib/webapi/published.py`, `mtpy/lib/webapi/__init__.py` (rutas y exportaciones), `tests/test_webapi_unit.py` (`write_study`), `docs/web/contrato.md` (rutas)
- Test: `tests/test_web_api.py`, `tests/test_bundle_unit.py` (fixtures validan)

**Interfaces:**
- Produces: `checks.STUDIES = bundle.STUDIES`, `checks.STUDY_PARTS`, `check_study(study: str) -> str` (404 `'unknown study'`); `Site.study(study, part) -> bytes` (404 `'study not published'`), `Site.study_csv(study, name) -> bytes`, `Site.studies() -> list[str]` (los de `STUDIES` con `meta`); rutas `('/api/v1/studies', 'studies', 'index', ['GET'])` y `('/api/v1/studies/{study}/{part}', 'studies', 'part', ['GET'])` antes de las de `forecast`; `Studies.index_action` → `{'contract', 'studies': [{'study', 'title', 'subtitle', 'registered_at', 'checked_at'}]}`; `Studies.part_action(study, part)`: JSON, o con `format=csv` el CSV `table` (nombre de `STUDY_CSV_OF[part]`, por defecto el primero; `meta` o un `table` ajeno a la parte → 404 `'no csv for this part'`), fichero `{study}-{name}.csv`; `notice_freeze` como `Analysis`.
- `write_study(fs, study='voto-cera', meta=None)` en `tests/test_webapi_unit.py`: escribe los once CSV mínimos y las ocho partes desde los fixtures (`meta` el último, con `meta` sustituyendo campos).
- Fixtures `study-*.json`: sintéticos pero completos (tres provincias, dos países, dos escenarios, dos elecciones en `check` sin la del 29-N; `registered_at` no nulo en el fixture de `meta`), con los sobres de `envelope` como los demás fixtures.

- [ ] **Step 1: Tests** (`tests/test_web_api.py`): `test_studies_index_and_parts_and_csv` (404 antes de publicar; tras `write_study`, `/api/v1/studies` lista `voto-cera` con su título; `/api/v1/studies/voto-cera/prediction` devuelve el JSON; `?format=csv` da `voto-cera-last-seat.csv`; `?format=csv&table=census-countries` en `census`; cabeceras `public, max-age=60` y CORS), parametrizado de errores (`meta` + csv → 404 `no csv for this part`; `vote` + `table=last-seat` → 404; parte desconocida → 404 `unknown part`; estudio desconocido → 404 `unknown study`). `tests/test_bundle_unit.py::test_study_schemas_validate_the_fixtures`.
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar** y añadir la tabla de rutas al contrato.
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Serve the studies in the API with their CSV twins`

---

### Task 13: La carga del INE: `load/cera.py`

**Files:**
- Create: `load/cera.py`
- Modify: `load/run_load.py` (`STEPS += ['cera']`; el paso ignora `--scopes`)
- Test: `tests/test_load_cera.py`

**Interfaces:**
- Produces: `MONTHLY_URL = 'https://www.ine.es/oficina_censo/censo_cerrado/cerapropais_{yy:02d}.xlsx'`, `ELECTION_URLS = {'elecgen2026_tab6.xlsx': 'https://www.ine.es/oficina_censo/elecgen2026/Elecgral_2026_cc_tab6.xlsx', ...tab7, tab8}`, `fetch_cera(data: FileSystem, year: int, fetch: Callable[[str], bytes] = _download) -> dict` (`{'files': [nombres escritos], 'months_new': [fechas que el libro nuevo trae y el anterior no]}`): baja el libro del año (`yy = year % 100`) y las tablas de la elección, los escribe en `cera/ine/` y compara los meses con el libro anterior (si existe) usando `read_monthly_workbook`; `_download` usa `requests.get` con un `User-Agent` de navegador y `raise_for_status`.
- `run_load.py --what cera` llama a `fetch_cera(App.get_().data, date.today().year)` e imprime los meses nuevos.

- [ ] **Step 1: Tests**: `test_fetch_cera_writes_the_files_and_reports_new_months(tmp_path)` con un `fetch` falso que devuelve los bytes de los ficheros reales de `data/cera/ine/` (el libro de 2026 y las tablas) sobre un `FileSystem` temporal que ya tiene una copia del libro con solo 8 meses (recortar con openpyxl) → `months_new == ['2026-09-01']`; `test_fetch_cera_propagates_http_errors` (`fetch` que lanza → la excepción sube y no se escribe nada).
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Download the INE census workbook and the election tables of the CERA study`

---

### Task 14: Las páginas: contextos, controlador, menú e Inicio

**Files:**
- Create: `mtpy/lib/pages/analisis.py`, `mtpy/controllers/Analisis.py`, `web/templates/analisis.html`
- Modify: `mtpy/lib/pages/common.py` (`PAGES`, `ROUTES`, `api_url`), `mtpy/lib/pages/__init__.py` (exportaciones), `mtpy/lib/pages/inicio.py` (`studies`), `web/templates/index.html` (banda), `web/templates/_header.html` (nada: lee `PAGES`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces:
  - `pages.PAGES` con `{'href': '/analisis', 'label': 'Análisis'}` entre Ranking y Método; `ROUTES` con `('/analisis', 'analisis', 'index', ['GET'])` y `('/analisis/{study:str}', 'analisis', 'study', ['GET'])` antes de `/{scope}`; `api_url(scope, part, kind='study', fmt=None, table=None)` → `/api/v1/studies/{scope}/{part}[?format=&table=]` (el slug viaja en `scope`).
  - `analisis.STUDIES = {'voto-cera': {'template': 'analisis_voto_cera.html'}}`; `study_part(study: str, part: str) -> Optional[dict]` (`bundle_part(site().study, study, part)`); `study_cards() -> list[dict]` (`{'study', 'href', 'title', 'subtitle', 'verdicts': [{'label': 'SÍ', 'tone': 'q1'}, ...], 'registered_at', 'checked_at'}` por estudio publicado); `analisis_context(state) -> dict` (`common_context(state, '/analisis')` + `studies`); `study_context(state, study) -> dict` (404 `'study not found'` si no está en `STUDIES` o no está publicado; para `voto-cera`, `cera_context`).
  - `cera_context(state) -> dict`: el común (`active='/analisis'`) + `meta`, `kpis` (tres filas para `m.kpis`), `verdicts`, `census`, `participation`, `vote`, `placement`, `prediction`, `sensitivity`, `check`, `figures` (las cifras que los textos citan: `growth_pct`, `growth_abs`, `ratio_article`, `net_before`, `net_after`, `latam_rate`, `rest_rate`, `r`, `n_over_gap_2023`, `decided_2023`, `votes_projected`, `voters_projected`, `max_gain`, …), `registered` (`{'at', 'claim'}` o `None`), `checked` (bool: `check.verdict` no nulo), `sources`, `csv_links` (`[{'label', 'href'}]` de los once CSV vía `api_url`), `article = {'url': 'https://medium.com/@luis_sancho/el-voto-cera-en-espa%C3%B1a-y-el-efecto-de-la-ley-de-nietos-ae2cd1b063a5', 'date': '2026-06-30'}` y la isla `initial = {'census': {'series', 'marks'}, 'origins', 'participation', 'ecological', 'slope', 'cannibal', 'sensitive', 'gap' (por escenario), 'radar'}`.
  - `Analisis(Page)`: `active = '/analisis'`; `index_action()`; `study_action(study: str)` → `self.render(STUDIES[study]['template'], **study_context(self.state(), study))`.
  - `inicio_context` añade `studies=study_cards()`; `index.html` pinta, tras «Elecciones pasadas» y solo con estudios, `<div class="sec-h" id="analisis"><h2>Análisis</h2><a href="/analisis">Ver los análisis ›</a></div>` y una tarjeta-enlace por estudio (título, subtítulo, píldoras de veredicto).
  - `analisis.html`: titular «Análisis», línea «Preguntas del debate electoral contestadas con datos oficiales, comprobables después de cada elección» y las tarjetas; sin estudios, «Todavía no hay ningún análisis publicado.».

- [ ] **Step 1: Tests** (`tests/test_pages.py`, con `write_study(fresh_app.fs)` donde haga falta):
  - `test_menu_has_analisis_between_ranking_and_metodo` (orden de `href` en el `<nav>` de cualquier página).
  - `test_analisis_index_lists_the_study_or_says_none` (sin estudio: 200 y el texto; con estudio: título, `href="/analisis/voto-cera"`, tres veredictos).
  - `test_analisis_unknown_study_is_404` (`/analisis/nope` y `/analisis/voto-cera` sin publicar → 404).
  - `test_inicio_shows_the_studies_band_only_with_a_study`.
  - `test_api_url_study` (`api_url('voto-cera', 'prediction', kind='study', fmt='csv', table='last-seat') == '/api/v1/studies/voto-cera/prediction?format=csv&table=last-seat'`).
  - `test_cera_context_figures_come_from_the_parts` (con el fixture: `figures['n_over_gap_2023'] == 9` si el fixture lo dice; `registered['at']` del fixture; `checked is False`).
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar** (la plantilla del estudio llega en las tareas 15–17; en esta, `analisis_voto_cera.html` puede ser un esqueleto con el título para que la ruta responda).
- [ ] **Step 4: Ejecutar y ver que pasan**
- [ ] **Step 5: Commit** — `Add the Análisis section: routes, menu entry, index page and the home band`

---

### Task 15: La página del estudio (1/3): cabecera, cifras, resumen y pregunta 1

**Files:**
- Modify: `web/templates/analisis_voto_cera.html`, `web/dist/css/site.css`
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `cera_context` (tarea 14); macros `m.card_start`, `m.card_end`, `m.how`, `m.kpis`, `m.badge`.
- Produces: la plantilla hasta `#p1`, con: `<title>` = titular; píldoras («Análisis», «Generales del 29 de noviembre de 2026», estado según `registered`/`checked`); `h1.study-title` y `p.study-subtitle` (textos exactos); `nav.section-nav.toc` con las entradas y clases `q1/q2/q3`; `m.kpis` con tres filas; `#resumen` con `div.three` (tres `a.q1/.q2/.q3` con `div.vv` = veredicto) y la nota de cierre; `#p1` como `section.card.qcard.q1` con `div.qh` (número, `h2.qt`, `span.verdict`), dos contenedores de gráfico (`#census-chart`, `#periods-chart`), las dos tablas (`#countries-table`, `#provinces-table` con 8 filas y botón «Ver las 52 provincias» con `wireMore`), `div.sum` y `m.how`. Estilos nuevos en `site.css`: `.study-title`, `.study-subtitle`, `.qcard`, `.qh`, `.qn`, `.qt`, `.verdict`, `.q1/.q2/.q3` (`--qc`), `.band`, `.sum`, `.three`, `.claim` (tarea 17), `.toc a.q1/.q2/.q3`; móvil: título 26 px, rejillas a una columna.

- [ ] **Step 1: Tests**: `test_cera_page_head_and_question_1(api, fresh_app)`: `<title>Demoscopio · El voto CERA en España y el efecto de la «Ley de Nietos»</title>`; el subtítulo exacto; las píldoras; `id="resumen"`, `id="p1"`, `class="verdict"` tres veces con `SÍ`, `SIN EVIDENCIA`, `NO` en ese orden en `#resumen`; el `election.census_total` del fixture de `study-meta` formateado con el filtro `int` (puntos de millar) dentro de `class="kpis"`; `id="provinces-table"` con 8 `<tr>` visibles y el botón; `medium.com` aparece **solo** después de `id="datos"` (esta aserción vive en la tarea 17, aquí se comprueba que no aparece antes de `#p1`).
- [ ] **Step 2: Ejecutar y ver que falla**
- [ ] **Step 3: Implementar** (textos de la maqueta v4, cifras interpoladas desde `figures`).
- [ ] **Step 4: Ejecutar y ver que pasa**
- [ ] **Step 5: Commit** — `Render the CERA study page: head, figures, summary and question 1`

---

### Task 16: La página del estudio (2/3): preguntas 2 y 3

**Files:**
- Modify: `web/templates/analisis_voto_cera.html`, `web/dist/css/site.css` (sombreado de rejillas)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `#p2` (`qcard q2`): `#origin-chart`, `#ecological-chart`, `div.band`, `#slope-chart`, `#cannibal-chart`, `#sensitive-chart`, tabla `#madrid-table`, `div.sum`, `m.how`; `#p3` (`qcard q3`): `fieldset.seg#scenario` con dos radios (`convocatoria`, `sin_litigio`; visible sin JS, el primero marcado), tabla `#prediction-table` (una por escenario, la segunda `hidden`), nota del escenario B, `#gap-chart`, `h3` «¿Qué haría falta para que importara?», `#radar-chart`, tabla `#grid-table` (sombreado por valor con un filtro nuevo `shade_scale(x, lo, hi)` registrado en `environment()`, verde `--color-ok` creciente), tabla `#scenarios-table`, tabla `#placement-table`, `div.sum`, `m.how`.

- [ ] **Step 1: Tests**: `test_cera_page_questions_2_and_3`: ids presentes; el `fieldset#scenario` con los dos valores; `#prediction-table` muestra `152`/`170` del fixture y la segunda tabla lleva `hidden`; `#grid-table` tiene 5 filas × 4 celdas con `style="background` en las no nulas; la nota de la celda observada («reparto uniforme») presente; `#madrid-table` con `Madrid`.
- [ ] **Step 2: Ejecutar y ver que falla**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasa**
- [ ] **Step 5: Commit** — `Render questions 2 and 3 of the CERA study with the scenario selector and the sensitivity tables`

---

### Task 17: La página del estudio (3/3): el debate, la comprobación, datos y glosario

**Files:**
- Modify: `web/templates/analisis_voto_cera.html`, `web/dist/css/site.css` (`.claim`, `.verd`)
- Test: `tests/test_pages.py`

**Interfaces:**
- Produces: `#debate` con la ficha de los 23 escaños (`article.claim` con `dl`; cifras vivas: `report.n_mean`, `report.n_max`, `report.votes_mean`, `report.votes_max`), la tabla `#passed-table` («2023 ya pasó la prueba»: filas `over_gap` de `check.elections[2023]` con columna «¿Cambió?» = «Sí, al {sigla}» / «No») y su veredicto exacto «Con el criterio del informe, en 2023 había {n} escaños en juego; cambió {m}, en {provincias}, y en contra de la hipótesis» (la última cláusula solo si `net_left < 0`), las tres fichas (16 provincias, 162 pueblos, censo fabricado); `#comprobacion` con la tabla por elección (`#check-table`, filas 2019, 2023 y la del 29-N con «predicción» / «borrador» según `registered`), la nota de registro, y, solo con `checked`, `#check-rows` (20 filas + «Ver más») y la nota de acierto/fallo (`check.verdict`); `#datos` con `sources`, `csv_links` y, al final, el enlace al artículo; `details.how#glosario` con las nueve entradas (CERA, voto rogado, Ley de Nietos, censo de la convocatoria, brecha del último escaño, efecto neto, inferencia ecológica, provincias sensibles, colocación óptima).

- [ ] **Step 1: Tests**: `test_cera_page_debate_check_and_sources`: `#passed-table` con 9 filas si el fixture da 9 y «Sí, al PP» en Madrid; el veredicto de la ficha; `#check-table` con tres filas y «borrador» cuando `registered_at` es nulo (fixture con `meta={'registered_at': None, 'registered': None}`) y «Predicción registrada el» cuando no; sin `checked`, no hay `id="check-rows"`; con un fixture de `check` con `verdict: 'acierto'` y la elección del 29-N, aparece `id="check-rows"` y el texto «acierto»; `medium.com` solo tras `id="datos"` (`page.index('medium.com') > page.index('id="datos"')` y una sola aparición); el glosario con `<dt>Voto rogado</dt>`.
- [ ] **Step 2: Ejecutar y ver que falla**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasa**
- [ ] **Step 5: Commit** — `Render the debate, the check and the sources of the CERA study`

---

### Task 18: Los gráficos y el módulo de página

**Files:**
- Create: `web/dist/js/charts/census.js`, `web/dist/js/charts/scatter.js`, `web/dist/js/charts/slope.js`, `web/dist/js/charts/radar.js`, `web/dist/js/pages/analisis-cera.js`
- Modify: `web/templates/analisis_voto_cera.html` (`{% block scripts %}` con el módulo), `tests/test_js_charts.py` (arnés: cargar los cuatro módulos como `c`, `sc`, `sl`, `ra`)
- Test: `tests/test_js_charts.py`

**Interfaces:**
- Produces (todas sobre `charts/base.js`, con `THEME`, `tooltipStyle`, `axisLabelStyle`, `mountChart`):
  - `censusOption(points: [[iso, n]], marks: [{date, label}], {color}) -> option` (eje temporal, área suave, `markLine` vertical por marca con etiqueta rotada; eje Y en millones «2,75 M») y `renderCensus(el, points, marks, opts)`.
  - `scatterOption(rows, {x, y, name, labelled: string[], fit: [[x0, y0], [x1, y1]] | null, xFormatter, yFormatter}) -> option` (puntos azules, etiquetas solo de `labelled`, recta discontinua roja si `fit`) y `renderScatter`.
  - `slopeOption(rows: [{name, from, to, color}], {labels: [string, string], yFormatter}) -> option` (una línea por fila entre dos categorías, etiqueta del valor final, `markLine` en 0 y `markArea` sombreada bajo 0) y `renderSlope`.
  - `radarOption(indicators: [{name, max}], series: [{name, value: number[], color, filled}]) -> option` y `renderRadar`.
  - `pages/analisis-cera.js`: `readInitial()`; dibuja `#census-chart` (`renderCensus`), `#periods-chart` (`renderStacked` de `stacked.js` con dos filas y etiqueta del ritmo, o barras agrupadas de `bars.js`: decisión del constructor), `#origin-chart` y `#cannibal-chart` (`bars.js` agrupadas: si `renderBars` no admite varias series, añadir `renderGroupedBars(el, categories, series, opts)` a `bars.js`), `#ecological-chart`, `#slope-chart`, `#sensitive-chart` (`stacked.js`), `#gap-chart` (`renderDiverging` con `left = brecha`, `right = neto`, o barras agrupadas; leyenda abajo), `#radar-chart`; `wireMore('provinces-more', 'provinces-table')`, `wireMore('check-more', 'check-rows')`; el `fieldset#scenario` alterna las tablas `hidden` y redibuja `#gap-chart` con `initial.gap[scenario]`; `markSegments`.

- [ ] **Step 1: Tests** (`tests/test_js_charts.py`): `test_census_option_marks_every_date` (4 `markLine.data`, eje X `time`, formatter del eje Y no serializable pero `yAxis.axisLabel` presente), `test_scatter_option_labels_only_the_named_provinces_and_draws_the_fit` (serie `scatter` con `label.formatter` ausente tras serializar pero `labelled` aplicado vía `data[i].label.show`), `test_slope_option_one_line_per_row_with_zero_reference` (3 series `line`, `markLine` en `yAxis: 0`), `test_radar_option_two_series` (`radar.indicator` de 3, 2 series).
- [ ] **Step 2: Ejecutar y ver que fallan**
- [ ] **Step 3: Implementar**
- [ ] **Step 4: Ejecutar y ver que pasan**; abrir la página con un estudio publicado en local (`publish study` sobre `files/`) y comprobar a ojo los ocho gráficos y el selector de escenario, en escritorio y a 390 px.
- [ ] **Step 5: Commit** — `Draw the charts of the CERA study and wire its scenario selector`

---

### Task 19: Cierre: documentación, suite completa y spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-design.md` (estado «cerrada», commits, tests; decisiones del constructor en la sección final: celdas que difieran del artículo, lista de exposición media, umbral ecológico, mapa de candidaturas), `docs/web/contrato.md` (revisión final), `data/README.md` (revisión final)

- [ ] **Step 1: Ejecutar la suite completa** — `pytest -q` → todos en verde (817 + los nuevos); anotar el total.
- [ ] **Step 2: Publicar en local y pasear** — `python load/run_load.py --scopes es --what cera`, `python job.py publish '{"what": "study", "study": "voto-cera"}'` (sin `register`), levantar la web y recorrer `/`, `/analisis`, `/analisis/voto-cera`, `/api/v1/studies`, un CSV; comprobar las cifras de la pregunta 1 contra el INE (2.769.645; Madrid 494.799) y las de la pregunta 3 contra el artículo (cero cambios; Cantabria 403 / Madrid 1.339).
- [ ] **Step 3: Cerrar la spec** con el estado, el rango de commits y el número de tests, y la lista de decisiones del constructor.
- [ ] **Step 4: Commit** — `Close the CERA study spec with the builder's decisions`

---

## Lo que Luis hace después (fuera del plan)

- `register=True` cuando dé por buena la página con datos reales (`python job.py publish '{"what": "study", "study": "voto-cera", "register": true}'`, con el árbol limpio).
- Tras el escrutinio general del 29-N: añadir `data/cera/escrutinio/es-2026-11-29.csv` (del fichero «Resultados voto CERA» de Interior, mismo formato), su entrada en `candidatures.json`, cargar `PROV_02_202611_1.xlsx` + `.json` como con cada elección, y publicar de nuevo: la comprobación y `checked_at` se rellenan solos.
- Cada mes: `--what cera` y `publish study` para refrescar la serie.
