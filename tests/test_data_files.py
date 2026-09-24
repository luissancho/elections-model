"""Coherencia de los ficheros de entrada versionados en `data/`."""
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')


def test_params_regions_are_valid_province_codes():
    params = json.load(open(os.path.join(DATA, 'params.json')))
    codes = set(pd.read_csv(os.path.join(DATA, 'es-provinces.csv'))['code'].astype(int))

    for scope, events in params.items():
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


def test_every_event_with_polls_url_has_wikipedia_maps():
    urls = json.load(open(os.path.join(DATA, 'wikipedia', 'wp-urls.json')))
    maps = json.load(open(os.path.join(DATA, 'wikipedia', 'wp-maps.json')))
    assert {'parties', 'pollsters', 'sponsors'} <= set(maps)
    assert all('polls' in conf for events in urls.values() for conf in events.values())


def test_infoelectoral_files_come_in_pairs():
    folder = os.path.join(DATA, 'infoelectoral', 'es')
    xlsx = {f[:-5] for f in os.listdir(folder) if f.endswith('.xlsx')}
    jsons = {f[:-5] for f in os.listdir(folder) if f.endswith('.json')}
    assert xlsx and xlsx == jsons, xlsx ^ jsons
