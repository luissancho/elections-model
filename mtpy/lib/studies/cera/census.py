"""
The CERA census series, the net additions in two periods and the trend break of the CERA study.

Every function takes the long monthly table of `data.read_monthly` (and, where needed, the convocation census of
`data.read_election_census` and the countries of `data.read_countries`) and returns plain pandas objects. Countries
are always joined by `country_code` against `countries.code`, never by name: the INE renamed some countries in the
middle of the series.
"""
import numpy as np
import pandas as pd

MARKS = [
    ('2022-10-21', 'Ley 20/2022'),
    ('2023-07-23', 'Generales 2023'),
    ('2025-10-22', 'Fin del plazo'),
    ('2026-08-01', 'Censo del 29-N')
]
GROUP_KEYS = ('region_id', 'country_code', 'region', 'exposure')
PERIOD_COLUMNS = ['c1_start', 'c1_end', 'c2_start', 'c2_end', 'g1', 'g2', 'ratio']
GROWTH_COLUMNS = ['baseline', 'election', 'growth', 'pct']
NET_SPANS = (('2019-11-01', '2023-07-01'), ('2023-07-01', None))
HAC_LAGS = 12


def series_points(series: pd.Series) -> list[list]:
    """
    Turn a monthly series into `[iso date, value]` pairs for the charts.

    Parameters
    ----------
    series : pd.Series
        Electors indexed by month (`pd.Timestamp`).

    Returns
    -------
    list[list]
        One `['YYYY-MM-DD', int]` pair per month, in the order of the index.
    """
    return [[date.strftime('%Y-%m-%d'), int(value)] for date, value in series.items()]


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Count the months from one first-of-month date to another.

    Parameters
    ----------
    start : pd.Timestamp
        The first date.
    end : pd.Timestamp
        The last date.

    Returns
    -------
    int
        `end - start` in months (negative if `end` comes first).
    """
    return (end.year - start.year) * 12 + end.month - start.month


def _keyed(frame: pd.DataFrame, key: str, countries: pd.DataFrame | None) -> pd.DataFrame:
    """
    Give a table of electors by province and country the column `key` to group by.

    Parameters
    ----------
    frame : pd.DataFrame
        A table with the columns `region_id`, `country_code` and `electors`.
    key : str
        One of `GROUP_KEYS`.
    countries : pd.DataFrame or None
        The table of `data.read_countries`; required unless `key` is `region_id`.

    Returns
    -------
    pd.DataFrame
        `frame` with the column `key` (added from `countries` for `region` and `exposure`).

    Raises
    ------
    ValueError
        If `key` is unknown, if `countries` is missing when needed, or if a country code is not in `countries`.
    """
    if key not in GROUP_KEYS:
        raise ValueError(f'cera: unknown key {key!r}, expected one of {GROUP_KEYS}')
    if key == 'region_id':
        return frame
    if countries is None:
        raise ValueError(f'cera: the key {key!r} needs the countries table')

    missing = sorted(set(frame['country_code']) - set(countries['code']))
    if missing:
        raise ValueError(f'cera: country codes not in countries.csv: {missing}')
    if key == 'country_code':
        return frame

    return frame.assign(**{key: frame['country_code'].map(countries.set_index('code')[key])})


def _names(keys: pd.Series, key: str, countries: pd.DataFrame | None,
           province_names: pd.Series | None) -> pd.Series:
    """
    Name the groups of a key: province, country or the group label itself.

    Parameters
    ----------
    keys : pd.Series
        The values of `key`.
    key : str
        One of `GROUP_KEYS`.
    countries : pd.DataFrame or None
        The table of `data.read_countries` (for `country_code`).
    province_names : pd.Series or None
        Province names indexed by `region_id` (`es-provinces.csv`, column `province` by `code`).

    Returns
    -------
    pd.Series
        One name per value of `keys`.

    Raises
    ------
    ValueError
        If `key` is `region_id` and `province_names` is missing, or a province has no name.
    """
    if key == 'region_id':
        if province_names is None:
            raise ValueError("cera: the key 'region_id' needs the province names")
        names = keys.map(province_names)
        if names.isna().any():
            raise ValueError(f'cera: provinces without a name: {sorted(keys[names.isna()])}')
        return names
    if key == 'country_code':
        return keys.map(countries.set_index('code')['name'])

    return keys


def _electors_at(long: pd.DataFrame, dates: list[pd.Timestamp], key: str,
                 countries: pd.DataFrame | None) -> pd.DataFrame:
    """
    Add up the monthly electors of each group of a key on some dates.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    dates : list[pd.Timestamp]
        First-of-month dates present in `long`.
    key : str
        One of `GROUP_KEYS`.
    countries : pd.DataFrame or None
        The table of `data.read_countries`.

    Returns
    -------
    pd.DataFrame
        Electors with the groups of `key` as index and one column per date (0 where a group has no row).

    Raises
    ------
    KeyError
        If a date is not in `long`.
    """
    available = set(long['date'])
    missing = [date for date in dates if date not in available]
    if missing:
        raise KeyError(f'cera: no monthly census on {[date.strftime("%Y-%m-%d") for date in missing]}')

    rows = _keyed(long[long['date'].isin(dates)], key, countries)
    table = rows.pivot_table(index=key, columns='date', values='electors', aggfunc='sum', fill_value=0)

    return table.reindex(columns=list(dict.fromkeys(dates)), fill_value=0)


def two_periods(long: pd.DataFrame, periods: list[tuple[str, str]], key: str,
                countries: pd.DataFrame | None = None, province_names: pd.Series | None = None) -> pd.DataFrame:
    """
    Count the net additions of CERA electors in two periods by province, country, origin or exposure.

    The growth of a period is the electors at its end minus the electors at its start, both first-of-month closes
    of the monthly series.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    periods : list[tuple[str, str]]
        Two `(start, end)` pairs of first-of-month dates.
    key : str
        `region_id`, `country_code`, `region` (origin) or `exposure` (to the Ley de Nietos).
    countries : pd.DataFrame, optional
        The table of `data.read_countries`; required unless `key` is `region_id`.
    province_names : pd.Series, optional
        Province names indexed by `region_id`; required when `key` is `region_id`.

    Returns
    -------
    pd.DataFrame
        Columns `[key, 'name', 'c1_start', 'c1_end', 'c2_start', 'c2_end', 'g1', 'g2', 'ratio']`, one row per group,
        sorted by `key`. `ratio = g2 / g1` is `inf` or `NaN` when `g1` is 0.

    Raises
    ------
    ValueError
        If `periods` is not two pairs, or as `_keyed` and `_names` do.
    KeyError
        If a date of `periods` is not in `long`.
    """
    if len(periods) != 2:
        raise ValueError(f'cera: two periods expected, got {len(periods)}')

    dates = [pd.Timestamp(date) for period in periods for date in period]
    table = _electors_at(long, dates, key, countries)
    out = pd.DataFrame({column: table[date] for column, date in zip(PERIOD_COLUMNS[:4], dates)}).reset_index()
    out['g1'] = out['c1_end'] - out['c1_start']
    out['g2'] = out['c2_end'] - out['c2_start']
    out['ratio'] = out['g2'] / out['g1']
    out.insert(1, 'name', _names(out[key], key, countries, province_names))

    return out[[key, 'name', *PERIOD_COLUMNS]].sort_values(key, ignore_index=True)


def update_periods(periods: list[tuple], last_date: pd.Timestamp) -> list[tuple]:
    """
    Move the end of the second period to the last close and give the first one the same length.

    The boundary between both periods stays; the first period gains or loses months at its start.

    Parameters
    ----------
    periods : list[tuple]
        Two `(start, end)` pairs of first-of-month dates, the second starting where the first ends.
    last_date : pd.Timestamp
        The last monthly close available.

    Returns
    -------
    list[tuple]
        Two `(start, end)` pairs of `pd.Timestamp` of equal length in months.
    """
    boundary, last_date = pd.Timestamp(periods[1][0]), pd.Timestamp(last_date)
    length = _months_between(boundary, last_date)
    end_first = pd.Timestamp(periods[0][1])

    return [(end_first - pd.DateOffset(months=length), end_first), (boundary, last_date)]


def election_growth(long: pd.DataFrame, census: pd.DataFrame, baseline: str, key: str,
                    countries: pd.DataFrame | None = None, province_names: pd.Series | None = None) -> pd.DataFrame:
    """
    Compare the monthly census on a baseline date with the census closed for the election.

    Parameters
    ----------
    long : pd.DataFrame
        The table of `data.read_monthly`.
    census : pd.DataFrame
        The convocation census of `data.read_election_census`.
    baseline : str
        A first-of-month date of `long` (see `baseline_date`).
    key : str
        `region_id`, `country_code`, `region` or `exposure`.
    countries : pd.DataFrame, optional
        The table of `data.read_countries`; required unless `key` is `region_id`.
    province_names : pd.Series, optional
        Province names indexed by `region_id`; required when `key` is `region_id`.

    Returns
    -------
    pd.DataFrame
        Columns `[key, 'name', 'baseline', 'election', 'growth', 'pct']`, one row per group in either census, sorted
        by `key`; `growth = election - baseline` and `pct` is that growth in % of `baseline`.

    Raises
    ------
    KeyError
        If `baseline` is not in `long`.
    ValueError
        As `_keyed` and `_names` do.
    """
    baseline = pd.Timestamp(baseline)
    before = _electors_at(long, [baseline], key, countries)[baseline].rename('baseline')
    after = _keyed(census, key, countries).groupby(key)['electors'].sum().rename('election')
    out = pd.concat([before, after], axis=1).fillna(0).astype(int).rename_axis(key).reset_index()
    out['growth'] = out['election'] - out['baseline']
    out['pct'] = 100 * out['growth'] / out['baseline']
    out.insert(1, 'name', _names(out[key], key, countries, province_names))

    return out[[key, 'name', *GROWTH_COLUMNS]].sort_values(key, ignore_index=True)


def baseline_date(series: pd.Series, target: int) -> pd.Timestamp:
    """
    Find the month whose national total is closest to a census figure.

    Parameters
    ----------
    series : pd.Series
        National electors indexed by month (`data.national_series`).
    target : int
        The census to match, e.g. the CERA census of an election in Infoelectoral.

    Returns
    -------
    pd.Timestamp
        The first month with the smallest absolute difference.
    """
    return (series - target).abs().idxmin()


def _mean_net(series: pd.Series, start: str, end: str | None) -> float:
    """
    Average the monthly net additions between two closes.

    Parameters
    ----------
    series : pd.Series
        National electors indexed by month.
    start : str
        The first close.
    end : str or None
        The last close; `None` for the last month of `series`.

    Returns
    -------
    float
        `(series[end] - series[start])` divided by the months between both.
    """
    start = pd.Timestamp(start)
    end = series.index[-1] if end is None else pd.Timestamp(end)

    return float(series[end] - series[start]) / _months_between(start, end)


def trend_break(series: pd.Series, break_date: str = '2022-10-01') -> dict:
    """
    Test the break in the trend of the monthly net additions at the Ley de Nietos.

    The monthly net additions (`series.diff()`) are regressed by OLS on a linear trend, a break dummy (1 from
    `break_date` on) and its interaction with the months since the break, with Newey-West errors (12 lags). The
    dummy measures the jump at the break and the interaction the change of slope. The regression uses the whole
    `series`: slice it to change the window.

    Parameters
    ----------
    series : pd.Series
        National electors indexed by month (`data.national_series`).
    break_date : str, default '2022-10-01'
        The first month of the new regime.

    Returns
    -------
    dict
        `slope_before` and `slope_after` (trend of the net additions, electors per month, before and after the
        break), `p_slope` (change of slope) and `p_level` (jump at the break), and `net_before` and `net_after`,
        the mean monthly net additions from November 2019 to July 2023 and from July 2023 to the last close.
    """
    import statsmodels.api as sm

    net = series.diff().dropna().astype(float)
    trend = np.arange(len(net), dtype=float)
    after = (net.index >= pd.Timestamp(break_date)).astype(float)
    since = np.where(after > 0, trend - trend[after.argmax()], 0.0)
    exog = pd.DataFrame({'const': 1.0, 'trend': trend, 'level': after, 'slope': since}, index=net.index)
    fit = sm.OLS(net, exog).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    return {
        'slope_before': float(fit.params['trend']),
        'slope_after': float(fit.params['trend'] + fit.params['slope']),
        'net_before': _mean_net(series, *NET_SPANS[0]),
        'net_after': _mean_net(series, *NET_SPANS[1]),
        'p_slope': float(fit.pvalues['slope']),
        'p_level': float(fit.pvalues['level'])
    }
