"""La serie del censo CERA, las altas en dos periodos y la ruptura de tendencia."""
import os

import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera.census import (
    baseline_date,
    election_growth,
    series_points,
    trend_break,
    two_periods,
    update_periods
)
from mtpy.lib.studies.cera.data import (
    national_series,
    read_countries,
    read_election_census,
    read_monthly,
    read_provinces
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PERIODS_ARTICLE = [('2019-03-01', '2022-10-01'), ('2022-10-01', '2026-05-01')]


@pytest.fixture(scope='module')
def data_fs():
    return FileSystem(os.path.join(ROOT, 'data'))


@pytest.fixture(scope='module')
def provinces(data_fs):
    return read_provinces(data_fs)


@pytest.fixture(scope='module')
def province_names(data_fs):
    return pd.read_csv(data_fs.get_path('es-provinces.csv')).set_index('code')['province']


@pytest.fixture(scope='module')
def long(data_fs, provinces):
    return read_monthly(data_fs, provinces)


@pytest.fixture(scope='module')
def countries(data_fs):
    return read_countries(data_fs.get_path('cera/countries.csv'))


@pytest.fixture(scope='module')
def census(data_fs, provinces):
    return read_election_census(data_fs.get_path('cera/ine/elecgen2026_tab8.xlsx'), provinces)


def test_series_points_are_iso_dates_and_integers(long):
    points = series_points(national_series(long))
    assert points[0] == ['2006-01-01', 1142334] and isinstance(points[-1][1], int) and len(points) == 249


def test_two_periods_reproduces_the_article_growth(long, countries, province_names):
    by_prov = two_periods(long, PERIODS_ARTICLE, 'region_id', province_names=province_names)
    assert list(by_prov.columns) == [
        'region_id', 'name', 'c1_start', 'c1_end', 'c2_start', 'c2_end', 'g1', 'g2', 'ratio'
    ]
    assert by_prov['g2'].sum() == 422253 and by_prov['g1'].sum() == 189290
    assert round(by_prov['g2'].sum() / by_prov['g1'].sum(), 1) == 2.2
    assert by_prov.set_index('region_id').loc[28, 'g2'] == 124195
    assert by_prov.set_index('region_id').loc[28, 'name'] == 'Madrid'
    by_country = two_periods(long, PERIODS_ARTICLE, 'country_code', countries)
    assert by_country.set_index('name').loc['ARGENTINA', 'g2'] == 73855
    assert by_country.set_index('name').loc['MÉXICO', 'g2'] == 54954


def test_two_periods_by_origin_and_exposure(long, countries):
    origins = two_periods(long, PERIODS_ARTICLE, 'region', countries).set_index('region')
    assert origins.loc['latam', 'g2'] > origins.loc['latam', 'g1'] * 3 and origins.loc['latam', 'name'] == 'latam'
    exposure = two_periods(long, PERIODS_ARTICLE, 'exposure', countries)
    assert set(exposure['exposure']) == {'alta', 'media', 'baja'}
    assert exposure['g2'].sum() == 422253


def test_update_periods_keeps_equal_lengths():
    assert update_periods(PERIODS_ARTICLE, pd.Timestamp('2026-09-01')) == [
        (pd.Timestamp('2018-11-01'), pd.Timestamp('2022-10-01')),
        (pd.Timestamp('2022-10-01'), pd.Timestamp('2026-09-01'))
    ]


def test_election_growth_against_the_convocation_census(long, census, countries, province_names):
    base = baseline_date(national_series(long), 2328260)
    assert abs(national_series(long)[base] - 2328260) / 2328260 < 0.005
    growth = election_growth(long, census, base, 'region_id', province_names=province_names)
    assert list(growth.columns) == ['region_id', 'name', 'baseline', 'election', 'growth', 'pct']
    assert growth['election'].sum() == 2769645 and growth.set_index('region_id').loc[28, 'election'] == 498874
    assert growth['growth'].sum() == 2769645 - national_series(long)[base]
    by_country = election_growth(long, census, base, 'country_code', countries)
    assert by_country['election'].sum() == 2769645


def test_trend_break_is_significant(long):
    out = trend_break(national_series(long))
    assert out['p_slope'] < 0.05 and out['net_after'] > 2 * out['net_before']
    assert out['slope_after'] > out['slope_before']
    # The monthly net additions climb gradually after the law: no significant jump at the break itself.
    assert out['p_level'] > 0.05
