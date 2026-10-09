"""Tests de `mtpy/lib/bundle.py`: disposición del paquete, sobre JSON y validación de esquemas."""
import copy
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'bundle')


def load_fixture(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def fixed_clock():
    return datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


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


def test_headline_requires_the_backfill_flag():
    obj = load_fixture('headline')
    del obj['data']['backfill']
    with pytest.raises(ValueError, match='missing keys'):
        bundle.validate('headline', obj)


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


def test_writer_round_trip_and_csv(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)), clock=fixed_clock)
    data = load_fixture('vote')['data']
    name = writer.write_json(bundle.path_part('es', '20261008-120000', 'vote', 'nowcast'), 'vote', data, 'es',
                             run_id='20261008-120000', mode='nowcast')
    assert name == 'runs/es/20261008-120000/nowcast/vote.json'
    assert (tmp_path / 'site' / 'v1' / name).exists()
    env = bundle.BundleReader(FileSystem(str(tmp_path))).read_json(name)
    assert env['generated_at'] == '2026-10-08T12:00:00Z' and env['data'] == data
    writer.write_csv(bundle.path_csv('es', '20261008-120000', 'vote', 'nowcast'), pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}))
    assert (tmp_path / 'site' / 'v1' / 'runs' / 'es' / '20261008-120000' / 'csv' / 'nowcast-vote.csv').read_bytes() == b'a,b\n1,x\n2,y\n'


def test_writer_validates_before_writing(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    with pytest.raises(ValueError, match='missing keys'):
        writer.write_json('runs/es/20261008-120000/nowcast/vote.json', 'vote', {'horizon': 0}, 'es', run_id='20261008-120000', mode='nowcast')
    assert not (tmp_path / 'site').exists()


def test_list_runs_keeps_only_complete_runs(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)), prefix=bundle.DRY_PREFIX)
    assert writer.list_runs('es') == []
    headline = load_fixture('headline')['data']
    for rid in ('20261008-120001', '20261007-090000'):
        writer.write_json(bundle.path_part('es', rid, 'headline'), 'headline', {**headline, 'run_id': rid}, 'es', run_id=rid)
    writer.write_json(bundle.path_part('es', '20261008-130000', 'meta'), 'meta', load_fixture('meta')['data'], 'es', run_id='20261008-130000')
    (tmp_path / 'site-dry' / 'v1' / 'runs' / 'es' / 'history.json').write_text('{}')
    assert writer.list_runs('es') == ['20261007-090000', '20261008-120001']
    assert writer.exists('runs/es/20261008-130000/meta.json')


def test_begin_run_refuses_an_existing_run(tmp_path):
    writer = bundle.BundleWriter(FileSystem(str(tmp_path)))
    writer.begin_run('es', '20261008-120000')
    writer.write_json(bundle.path_part('es', '20261008-120000', 'meta'), 'meta', load_fixture('meta')['data'], 'es', run_id='20261008-120000')
    with pytest.raises(FileExistsError):
        writer.begin_run('es', '20261008-120000')
    writer.remove(bundle.path_run('es', '20261008-120000'))
    writer.begin_run('es', '20261008-120000')
