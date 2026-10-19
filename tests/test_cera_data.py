"""Los datos de entrada del estudio del voto CERA (`data/cera/`)."""
import glob
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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
