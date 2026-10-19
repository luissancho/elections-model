"""
Readers of the inputs of the CERA study (`data/cera/` and `data/infoelectoral/`).

Every function here only reads and reshapes files: the monthly INE workbooks of CERA electors by province and
country, the tables of the census closed for the election, the CERA scrutinies by province, the Infoelectoral
results with every candidature and the small CSV and JSON inputs. The computations of the study live elsewhere.
"""
import json
import re

from dataclasses import dataclass

import pandas as pd

from unidecode import unidecode

from ....core.io import FileSystem

MONTHLY_COLUMNS = ['date', 'region_id', 'country_code', 'country', 'electors']
CENSUS_COLUMNS = ['region_id', 'country_code', 'country', 'electors']
SCRUTINY_TOTALS = ['electors', 'voters', 'invalid', 'blank']
RESULTS_TOTALS = {
    'Total censo electoral': 'registered',
    'Censo CERA': 'registered_cera',
    'Total votantes': 'counted',
    'Total votantes CERA': 'counted_cera',
    'Votos válidos': 'valid',
    'Votos a candidaturas': 'candidates',
    'Votos en blanco': 'blank',
    'Votos nulos': 'invalid'
}
SPANISH_MONTHS = {
    'ENE': 1, 'FEB': 2, 'MAR': 3, 'ABR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12
}
MONTH_HEADER = re.compile(r'a 1-(\w+)-(\d{4})', re.IGNORECASE)
COUNTRY_CELL = re.compile(r'^(\d+)\s+(.+)$')
NATIONAL_TOTALS = ('', 'NACIONAL', 'GENERAL')
HEADER_ROWS = 10


def normalise_label(text: str) -> str:
    """
    Normalise a territorial label so that the spellings of every source compare equal.

    Parameters
    ----------
    text : str
        The label as written in the source (`Araba/Álava`, `C.F. de Navarra`, ...).

    Returns
    -------
    str
        The label in uppercase, without accents and with only the characters `[A-Z0-9]`.
    """
    return re.sub(r'[^A-Z0-9]', '', unidecode(str(text)).upper())


def read_provinces(data: FileSystem) -> dict[str, int]:
    """
    Read the map from normalised province labels to `region_id` (`cera/provinces.json`).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.

    Returns
    -------
    dict[str, int]
        Normalised label to `region_id` (the INE code of the province).
    """
    return json.loads(data.read('cera/provinces.json'))


def province_id(label: str, provinces: dict) -> int:
    """
    Find the `region_id` of a province label in any of its spellings.

    Parameters
    ----------
    label : str
        The label as written in the source.
    provinces : dict
        The map returned by `read_provinces`.

    Returns
    -------
    int
        The `region_id` of the province.

    Raises
    ------
    KeyError
        With the normalised label, if it is not in the map.
    """
    key = normalise_label(label)
    if key not in provinces:
        raise KeyError(key)

    return provinces[key]


def _cell_text(value) -> str:
    """
    Render a spreadsheet cell as stripped text, writing integral numbers without decimals.

    Parameters
    ----------
    value : Any
        The value of the cell.

    Returns
    -------
    str
        The text of the cell, or an empty string for a missing value.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def _counts(block: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a block of count cells into integers, reading dashes and blanks as zero.

    Parameters
    ----------
    block : pd.DataFrame
        The cells as read from the workbook.

    Returns
    -------
    pd.DataFrame
        The same block as `int64`.
    """
    block = block.map(lambda v: 0 if _cell_text(v) in ('', '-') else v)

    return block.apply(pd.to_numeric).round().astype('int64')


def _month(token: str, year: str) -> pd.Timestamp:
    """
    Build the first day of the month of a census header (`12`, `9`, `dic`, `sep`, ...).

    Parameters
    ----------
    token : str
        The month, numeric or abbreviated in Spanish.
    year : str
        The year with four digits.

    Returns
    -------
    pd.Timestamp
        The first day of that month.
    """
    month = int(token) if token.isdigit() else SPANISH_MONTHS[normalise_label(token)]

    return pd.Timestamp(int(year), month, 1)


def _read_sheet(path: str) -> pd.DataFrame:
    """
    Read the first sheet of a workbook as raw cells (`.xls` with `xlrd`, `.xlsx` with `openpyxl`).

    Parameters
    ----------
    path : str
        The full path to the workbook.

    Returns
    -------
    pd.DataFrame
        Every cell of the first sheet as `object`, without a header.
    """
    engine = 'xlrd' if path.lower().endswith('.xls') else 'openpyxl'

    return pd.read_excel(path, sheet_name=0, header=None, dtype=object, engine=engine)


def _month_columns(raw: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """
    Find the month of every count column of a monthly workbook.

    The header is read on the rows above the national total, joining the cells of each column, so that both
    layouts parse: «Censo cerrado» / «a 1-12-2006» on two rows and «Censo cerrado a 1-may-2026» on one. When a month
    appears twice (February 2011 has a «(nuevo)» reading beside the old one), the leftmost column is kept: the
    columns run from the newest reading to the oldest.

    Parameters
    ----------
    raw : pd.DataFrame
        The raw cells of the sheet.

    Returns
    -------
    dict[int, pd.Timestamp]
        Column position to the first day of its month, in the order of the sheet.
    """
    labels = raw[0].map(_cell_text).str.upper()
    first_total = labels[labels.str.startswith('TOTAL')].index[0]
    columns = {}
    for position in range(1, raw.shape[1]):
        header = ' '.join(_cell_text(v) for v in raw.iloc[:first_total, position])
        match = MONTH_HEADER.search(header)
        if match:
            date = _month(match.group(1), match.group(2))
            if date not in columns.values():
                columns[position] = date

    return columns


def read_monthly_workbook(path: str, provinces: dict) -> pd.DataFrame:
    """
    Read one INE workbook of CERA electors by province and country of residence, month by month.

    The province rows are the totals («Total Álava», «TOTAL ALBACETE», ...) followed by the country rows, whose
    first cell is the province code; the country code and name share one cell in 2006-2007 and sit in two cells
    since 2008. The national total, the province totals and the footnotes are left out. Each country row goes to
    the province of its own code, even when it sits misplaced under the total of another one (a single elector
    in 2010, 2011, 2013, 2014 and 2016). A country renamed within the year (MACEDONIA, SWAZILANDIA, ...) has two
    rows with the same code: they are added up under the name with most electors in the newest month.

    Parameters
    ----------
    path : str
        The full path to the `.xls` or `.xlsx` workbook.
    provinces : dict
        The map returned by `read_provinces`.

    Returns
    -------
    pd.DataFrame
        Columns `MONTHLY_COLUMNS`: one row per month, province and country, `date` being the first day of the
        month and dashes or blanks read as zero.

    Raises
    ------
    KeyError
        If a province total has a label that is not in `provinces`.
    ValueError
        If a country row comes before any province total, has an unknown province code or no country code.
    """
    raw = _read_sheet(path)
    months = _month_columns(raw)
    first_month = min(months)
    regions = set(provinces.values())

    rows = []
    region = None
    for index, values in raw.iterrows():
        label = _cell_text(values[0])
        if label.upper().startswith('TOTAL'):
            name = label[len('TOTAL'):]
            region = None if normalise_label(name) in NATIONAL_TOTALS else province_id(name, provinces)
        elif re.fullmatch(r'\d{1,2}', label):
            if region is None or int(label) not in regions:
                raise ValueError(f'{path}: row {index} has province {label} outside any known province')
            country = ' '.join(filter(None, (_cell_text(v) for v in values[1:first_month])))
            match = COUNTRY_CELL.match(country)
            if not match:
                raise ValueError(f'{path}: row {index} has no country code: {country!r}')
            rows.append([int(label), int(match.group(1)), match.group(2).strip(), index])

    places = pd.DataFrame(rows, columns=['region_id', 'country_code', 'country', 'row'])
    counts = _counts(raw.loc[places['row'], list(months)])
    counts.columns = list(months.values())
    counts.index = places.index
    wide = pd.concat([places.drop(columns='row'), counts], axis=1)
    long = wide.melt(id_vars=['region_id', 'country_code', 'country'], var_name='date', value_name='electors')
    long['date'] = pd.to_datetime(long['date'])

    newest = long[long['date'] == long['date'].max()]
    names = newest.groupby(['country_code', 'country'])['electors'].sum().reset_index()
    names = names.sort_values('electors', ascending=False, kind='stable').drop_duplicates('country_code')
    long = long.groupby(['date', 'region_id', 'country_code'], as_index=False)['electors'].sum()
    long['country'] = long['country_code'].map(names.set_index('country_code')['country'])

    return long[MONTHLY_COLUMNS].sort_values(['date', 'region_id', 'country_code'], ignore_index=True)


def read_monthly(data: FileSystem, provinces: dict, folder: str = 'cera/ine') -> pd.DataFrame:
    """
    Read every monthly INE workbook of a folder into one long table.

    The workbooks are read in the order of their file names; when a month is in more than one, the reading of the
    last workbook wins for each province and country.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    provinces : dict
        The map returned by `read_provinces`.
    folder : str, default 'cera/ine'
        The folder of the `cerapropais_*.xls[x]` workbooks, relative to `data`.

    Returns
    -------
    pd.DataFrame
        Columns `MONTHLY_COLUMNS`, sorted by date, province and country.
    """
    names = data.glob(f'{folder}/cerapropais_*.xls*')
    long = pd.concat([read_monthly_workbook(data.get_path(name), provinces) for name in names], ignore_index=True)
    long = long.drop_duplicates(subset=['date', 'region_id', 'country_code'], keep='last')

    return long.sort_values(['date', 'region_id', 'country_code'], ignore_index=True)


def national_series(long: pd.DataFrame) -> pd.Series:
    """
    Add up the CERA electors of every province and country by month.

    Parameters
    ----------
    long : pd.DataFrame
        A table with the columns `MONTHLY_COLUMNS`.

    Returns
    -------
    pd.Series
        Electors indexed by `date`.
    """
    return long.groupby('date')['electors'].sum()


def province_series(long: pd.DataFrame) -> pd.DataFrame:
    """
    Add up the CERA electors of every country by month and province.

    Parameters
    ----------
    long : pd.DataFrame
        A table with the columns `MONTHLY_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        Electors with `date` as index and one column per `region_id`.
    """
    return long.pivot_table(index='date', columns='region_id', values='electors', aggfunc='sum', fill_value=0)


def read_election_census(path: str, provinces: dict) -> pd.DataFrame:
    """
    Read the CERA census closed for an election by province and country (`elecgen*_tab8.xlsx`).

    Each province is a label in uppercase with its total, followed by continent headings (without figures) and the
    country rows `code | name | electors`. The national total and the footnotes are left out.

    Parameters
    ----------
    path : str
        The full path to the workbook.
    provinces : dict
        The map returned by `read_provinces`.

    Returns
    -------
    pd.DataFrame
        Columns `CENSUS_COLUMNS`, one row per province and country.

    Raises
    ------
    ValueError
        If a country row comes before any province label.
    KeyError
        If a label with a figure is neither a province nor the national total.
    """
    raw = _read_sheet(path)
    rows = []
    region = None
    for index, values in raw.iterrows():
        label, electors = _cell_text(values[0]), pd.to_numeric(values[2], errors='coerce')
        if re.fullmatch(r'\d+', label):
            if region is None:
                raise ValueError(f'{path}: row {index} has a country before any province')
            rows.append([region, int(label), _cell_text(values[1]), 0 if pd.isna(electors) else int(electors)])
        elif label and pd.notna(electors) and not normalise_label(label).startswith('TOTAL'):
            region = province_id(label, provinces)

    return pd.DataFrame(rows, columns=CENSUS_COLUMNS)


def read_scrutiny(path: str) -> pd.DataFrame:
    """
    Read a CERA scrutiny by province (`cera/escrutinio/{scope}-{date}.csv`).

    Parameters
    ----------
    path : str
        The full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Indexed by `region_id`, with the columns `SCRUTINY_TOTALS` followed by one column per candidature (the
        labels of the Ministry of the Interior), missing values read as zero.
    """
    scrutiny = pd.read_csv(path).set_index('region_id').drop(columns='label')

    return scrutiny.fillna(0).astype('int64')


def read_candidatures_map(data: FileSystem) -> dict[str, dict[str, str]]:
    """
    Read the map from scrutiny columns to Infoelectoral candidature names (`cera/escrutinio/candidatures.json`).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.

    Returns
    -------
    dict[str, dict[str, str]]
        `{date: {scrutiny column: candidature full name}}`.
    """
    return json.loads(data.read('cera/escrutinio/candidatures.json'))


def scrutiny_votes(scrutiny: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Express the CERA votes of a scrutiny with the candidature names of the Infoelectoral results.

    Parameters
    ----------
    scrutiny : pd.DataFrame
        A table returned by `read_scrutiny`.
    mapping : dict[str, str]
        Scrutiny column to candidature full name, for one election.

    Returns
    -------
    pd.DataFrame
        Votes with `region_id` as index and one column per candidature full name; the columns that map to the
        same candidature (PSOE and PSC, ...) are added up.

    Raises
    ------
    ValueError
        Listing the candidature columns of the scrutiny without an entry in the map.
    """
    columns = [c for c in scrutiny.columns if c not in SCRUTINY_TOTALS]
    unknown = [c for c in columns if c not in mapping]
    if unknown:
        raise ValueError(f'Scrutiny columns without a candidature in the map: {", ".join(unknown)}')

    return scrutiny[columns].T.groupby(lambda c: mapping[c], sort=False).sum().T


@dataclass
class Results:
    """
    Results of a general election by province, with every candidature (Infoelectoral `PROV_02_*`).

    Attributes
    ----------
    date : str
        The date of the election (`YYYY-MM-DD`).
    votes : pd.DataFrame
        Votes with `region_id` as index and one column per candidature full name.
    seats : pd.DataFrame
        Seats with the same shape as `votes`.
    totals : pd.DataFrame
        Indexed by `region_id`: `registered`, `registered_cera`, `counted`, `counted_cera`, `valid`,
        `candidates`, `blank`, `invalid` and `n_seats`.
    names : pd.Series
        `region_id` to the name of the province.
    siglas : dict[str, str | None]
        Candidature full name to its sigla in the model (`None` for the others).
    """
    date: str
    votes: pd.DataFrame
    seats: pd.DataFrame
    totals: pd.DataFrame
    names: pd.Series
    siglas: dict[str, str | None]


def read_results(data: FileSystem, date: str) -> Results:
    """
    Read the results of a general election by province from `infoelectoral/es/PROV_02_{YYYYMM}_1.xlsx`.

    The header row starts with «Nombre de Comunidad», the full candidature names sit two rows above it on the
    «Votos» column of each «Votos» / «Diputados» pair, and the province rows are those with a code from 1 to 52
    (the blank rows and the national total are left out).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    date : str
        The date of the election (`YYYY-MM-DD`).

    Returns
    -------
    Results
        Votes, seats, totals, province names and siglas of the election.

    Raises
    ------
    ValueError
        If the candidature columns do not come in «Votos» / «Diputados» pairs.
    """
    name = f'infoelectoral/es/PROV_02_{date[:4]}{date[5:7]}_1'
    raw = pd.read_excel(data.get_path(f'{name}.xlsx'), sheet_name=0, header=None, dtype=object)
    header_row = raw.index[raw[0].map(_cell_text) == 'Nombre de Comunidad'][0]
    headers = [_cell_text(v) for v in raw.iloc[header_row]]

    first_votes = headers.index('Votos')
    pairs = list(range(first_votes, len(headers), 2))
    if (len(headers) - first_votes) % 2 or any(headers[j:j + 2] != ['Votos', 'Diputados'] for j in pairs):
        raise ValueError(f'{name}.xlsx: the candidature columns are not «Votos» / «Diputados» pairs')
    candidatures = [_cell_text(raw.iat[header_row - 2, j]) for j in pairs]

    rows = raw.iloc[header_row + 1:]
    codes = pd.to_numeric(rows[headers.index('Código de Provincia')], errors='coerce')
    rows = rows[codes.between(1, 52)]
    index = pd.Index(codes[codes.between(1, 52)].astype(int), name='region_id')

    def block(positions: list[int]) -> pd.DataFrame:
        """Integer block of the given columns, one column per candidature, duplicates added up."""
        frame = _counts(rows[positions])
        frame.index, frame.columns = index, candidatures
        return frame.T.groupby(level=0, sort=False).sum().T

    votes = block(pairs)
    seats = block([j + 1 for j in pairs])
    totals = _counts(rows[[headers.index(h) for h in RESULTS_TOTALS]])
    totals.index, totals.columns = index, list(RESULTS_TOTALS.values())
    totals['n_seats'] = seats.sum(axis=1)
    names = rows[headers.index('Nombre de Provincia')].map(_cell_text)
    names.index = index

    return Results(
        date=date, votes=votes, seats=seats, totals=totals, names=names.rename('name'),
        siglas=json.loads(data.read(f'{name}.json'))
    )


def read_consulates(path: str) -> pd.DataFrame:
    """
    Read the CERA electors and voters of a regional election by country of residence (`cera/consulados/*.csv`).

    Parameters
    ----------
    path : str
        The full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns `country`, `electors` and `voters`.
    """
    return pd.read_csv(path)[['country', 'electors', 'voters']]


def read_autonomicas(path: str) -> pd.DataFrame:
    """
    Read the CERA results of the regional elections (`cera/autonomicas.csv`).

    Parameters
    ----------
    path : str
        The full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        One row per community and election, with its CERA electors, voters and votes by party and block.
    """
    return pd.read_csv(path)


def read_countries(path: str) -> pd.DataFrame:
    """
    Read the countries of residence with their region and exposure (`cera/countries.csv`).

    Parameters
    ----------
    path : str
        The full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns `code`, `name`, `region` and `exposure`.
    """
    return pd.read_csv(path)


def load_params(data: FileSystem) -> dict:
    """
    Read the parameters of the study (`cera/params.json`).

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.

    Returns
    -------
    dict
        The parameters as written in the file.
    """
    return json.loads(data.read('cera/params.json'))
