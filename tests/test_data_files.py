"""Coherencia de los ficheros de entrada versionados en `data/`."""
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')


def test_params_regions_are_valid_province_codes():
    params = json.load(open(os.path.join(DATA, 'params.json')))
    districts = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))

    for scope, events in params.items():
        # Las reglas `regions` usan los `region_id` de las circunscripciones del ámbito
        codes = set(districts.loc[districts['scope'] == scope, 'region_id'].astype(int))
        for event_date, conf in events.items():
            for key, rules in conf.get('smap', {}).items():
                rules = rules if isinstance(rules, list) else [rules]
                for rule in rules:
                    regions = rule.get('regions')
                    if regions is None:
                        continue
                    listed = regions['exclude'] if isinstance(regions, dict) else regions
                    bad = [r for r in listed if int(r) not in codes]
                    assert not bad, (scope, event_date, key, bad)


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


def test_infoelectoral_files_come_in_pairs():
    folder = os.path.join(DATA, 'infoelectoral', 'es')
    xlsx = {f[:-5] for f in os.listdir(folder) if f.endswith('.xlsx')}
    jsons = {f[:-5] for f in os.listdir(folder) if f.endswith('.json')}
    assert xlsx and xlsx == jsons, xlsx ^ jsons


def test_scopes_catalogue_is_valid():
    """M11: catálogo de ámbitos con códigos ISO 3166-2, umbrales y pesos del rating."""
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
    """M11: circunscripciones por ámbito; las provincias conservan su código INE, el resto desde 100."""
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
    """M11: una sola numeración de comunidades en `data/` (la del INE)."""
    regions = pd.read_csv(os.path.join(DATA, 'es-regions.csv')).set_index('code')
    ages = pd.read_csv(os.path.join(DATA, 'es-regions-ages.csv'))
    provinces = pd.read_csv(os.path.join(DATA, 'es-provinces.csv'))
    assert regions.loc[[7, 8, 10, 13, 14], 'region'].tolist() == ['C. León', 'C. La Mancha', 'Valencia', 'Madrid', 'Murcia']
    assert set(provinces['reg_code']) <= set(regions.index)
    assert set(ages['code']) == set(regions.index)
    assert (ages.groupby('code')['region'].first() == regions['region']).all()


def test_threshold_overrides_are_valid():
    """M11: los umbrales históricos por evento de `params.json` son nulos o están en (0, 100]."""
    params = json.load(open(os.path.join(DATA, 'params.json')))
    found = 0
    for scope, events in params.items():
        for event_date, conf in events.items():
            for key in ('threshold', 'threshold_scope'):
                if key in conf:
                    found += 1
                    assert conf[key] is None or 0 < conf[key] <= 100, (scope, event_date, key)
    assert params['es-mc']['2015-05-24'] == {'threshold': None, 'threshold_scope': 5}
    assert params['es-cn']['2011-05-22'] == {'threshold': 30, 'threshold_scope': 6}
    assert found >= 8
