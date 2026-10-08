"""Tests de `mtpy/lib/bundle.py`: disposición del paquete, sobre JSON y validación de esquemas."""
import copy
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mtpy.lib import bundle

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'bundle')


def load_fixture(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def test_run_id_is_utc_and_fits_the_route_alias():
    rid = bundle.run_id(datetime(2026, 10, 8, 14, 15, 3, tzinfo=timezone.utc))
    assert rid == '20261008-141503'
    assert bundle.RUN_ID_RE.match(rid) and bundle.ROUTE_ALIAS_RE.match(rid)
    assert bundle.run_id(datetime(2026, 10, 8, 16, 0, 0, tzinfo=timezone(timedelta(hours=2)))) == '20261008-140000'
    assert bundle.RUN_ID_RE.match(bundle.run_id())
    assert bundle.iso_utc(datetime(2026, 10, 8, 14, 15, 3, tzinfo=timezone.utc)) == '2026-10-08T14:15:03Z'


def test_jsonable_cleans_nan_and_numpy_at_any_depth():
    value = {'a': np.int64(3), 'b': [float('nan'), np.float64(1.5), float('inf')], 'c': {np.int64(28): pd.Timestamp('2026-10-08')},
             'd': np.array([1.0, np.nan]), 'e': pd.NaT, 'f': (1, 2), 'g': True}
    assert bundle.jsonable(value) == {'a': 3, 'b': [None, 1.5, None], 'c': {'28': '2026-10-08'}, 'd': [1.0, None], 'e': None,
                                      'f': [1, 2], 'g': True}
    assert bundle.loads(bundle.dumps({'x': float('nan')})) == {'x': None}
    assert bundle.dumps({'a': 1}) == b'{"a":1}'
    with pytest.raises(TypeError, match='records'):
        bundle.jsonable({'df': pd.DataFrame({'a': [1]})})


def test_paths_follow_the_published_layout():
    assert bundle.path_manifest() == 'manifest.json'
    assert bundle.path_history('es') == 'runs/es/history.json'
    assert bundle.path_run('es-md', '20261008-141503') == 'runs/es-md/20261008-141503'
    assert bundle.path_part('es', '20261008-141503', 'meta') == 'runs/es/20261008-141503/meta.json'
    assert bundle.path_part('es', '20261008-141503', 'vote', mode='nowcast') == 'runs/es/20261008-141503/nowcast/vote.json'
    assert bundle.path_csv('es', '20261008-141503', 'series') == 'runs/es/20261008-141503/csv/series.csv'
    assert bundle.path_csv('es', '20261008-141503', 'vote', mode='forecast') == 'runs/es/20261008-141503/csv/forecast-vote.csv'


def test_envelope_has_the_common_keys():
    env = bundle.envelope('vote', {'x': 1}, 'es', run_id='20261008-141503', mode='nowcast', generated_at='2026-10-08T14:15:03Z')
    assert env == {'schema': 'vote@1', 'contract': 1, 'scope': 'es', 'run_id': '20261008-141503', 'mode': 'nowcast',
                   'generated_at': '2026-10-08T14:15:03Z', 'data': {'x': 1}}
    assert bundle.envelope('manifest', {}, None)['generated_at'].endswith('Z')


@pytest.mark.parametrize('name', sorted(bundle.SCHEMAS))
def test_validate_accepts_every_fixture_and_rejects_a_missing_key(name):
    obj = load_fixture(name)
    assert bundle.validate(name, obj) is obj
    broken = copy.deepcopy(obj)
    key = next(iter(bundle.SCHEMAS[name]))
    del broken['data'][key]
    with pytest.raises(ValueError, match='missing keys'):
        bundle.validate(name, broken)


def test_validate_checks_the_envelope():
    obj = load_fixture('vote')
    with pytest.raises(ValueError, match='schema'):
        bundle.validate('summary', obj)
    with pytest.raises(ValueError, match='run_id'):
        bundle.validate('vote', {**obj, 'run_id': '2026-10-08'})
    with pytest.raises(ValueError, match='mode'):
        bundle.validate('vote', {**obj, 'mode': None})
    with pytest.raises(ValueError, match='wrong type for horizon'):
        bundle.validate('vote', {**obj, 'data': {**obj['data'], 'horizon': '0'}})
    with pytest.raises(ValueError, match='unknown schema'):
        bundle.validate('nope', obj)
