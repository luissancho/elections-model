"""Tests de `mtpy/lib/studies/herd/data.py`: parámetros y lectura de las entradas del estudio del rebaño desde el
paquete sintético de `write_herd_bundle`."""
import os

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.herd import data
from tests.fakes import HERD_RUN, write_herd_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
NAMES = {'es': 'España', 'es-md': 'Madrid'}


def test_load_params_reads_the_repository_file():
    assert PARAMS['scopes'] == 'all' and PARAMS['since'] is None and PARAMS['live_scope'] == 'es'
    assert (PARAMS['max_days'], PARAMS['pair_days'], PARAMS['min_rows'], PARAMS['n_series']) == (365, 7, 3, 4)
    assert PARAMS['periods'] == {'final': [0, 21], 'mid': [21, 56], 'far': [56, 365]}
    assert (PARAMS['exclude'], PARAMS['permutations'], PARAMS['seed']) == (['CIS'], 2000, 0)


def test_prepare_polls_filters_and_main_series(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    cycle = bundle.BundleReader(fs).read_json(bundle.path_event('es', '2019-11-10', 'cycle'))['data']
    polls, names = data.prepare_polls(cycle['polls'], cycle['parties'], '2019-11-10', PARAMS)
    assert names == ['PP', 'PSOE'] and len(polls) == 12
    assert list(polls.columns) == ['poll', 'pollster', 'days', 'n', 'PP', 'PSOE']
    assert set(polls['pollster']) == {'A', 'B'} and polls['days'].dtype.kind == 'i'
    assert polls.iloc[0].to_dict() == {'poll': '0', 'pollster': 'A', 'days': 200, 'n': 1000.0, 'PP': 30.0,
                                       'PSOE': 25.0}


def test_prepare_polls_applies_the_window_and_the_sample_filters():
    def record(end, **extra):
        return {'date': end, 'end_date': end, 'pollster': 'A', 'sample_size': 1000, 'weight': 1, 'mtype': None,
                'PP': 30.0, 'PSOE': 25.0} | extra

    records = [record('2026-11-20'), record('2025-11-20'), record('2026-11-30'), record(None, date='2026-11-10'),
               record('2026-11-10', sample_size=None), record('2026-11-10', weight=0),
               record('2026-11-10', mtype='online'), record('2026-11-10', pollster='CIS', PP=60.0)]
    polls, names = data.prepare_polls(records, ['PP', 'PSOE'], '2026-11-29', PARAMS)
    # 2025-11-20 is 374 days away and 2026-11-30 is after the election: out; online out; no sample, no weight and the
    # excluded house are kept out after the series are chosen
    assert list(polls['poll']) == ['0', '3'] and list(polls['days']) == [9, 19]
    assert names == ['PP', 'PSOE']


def test_read_elections_every_scope_with_results(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    elections, inputs = data.read_elections(bundle.BundleReader(fs), PARAMS, NAMES)
    assert [e.label for e in elections] == ['Generales 2019', 'Madrid 2021', 'Generales 2023']
    assert [(e.scope, e.kind, e.current, e.as_of) for e in elections] == [
        ('es', 'generales', False, None), ('es-md', 'autonómicas', False, None), ('es', 'generales', False, None)]
    assert elections[0].names == ['PP', 'PSOE'] and len(elections[0].polls) == 12
    assert inputs['es']['read'] == ['2019-11-10', '2023-07-23'] and inputs['es']['skipped'] == {}
    assert inputs['es-md']['read'] == ['2021-05-04']
    assert inputs['es-md']['skipped'] == {'2023-05-28': 'no results'}


def test_read_elections_since_and_scopes(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    reader = bundle.BundleReader(fs)
    elections, _ = data.read_elections(reader, PARAMS | {'since': '2022-01-01'}, NAMES)
    assert [e.label for e in elections] == ['Generales 2023']
    elections, inputs = data.read_elections(reader, PARAMS | {'scopes': ['es-md']}, NAMES)
    assert [e.label for e in elections] == ['Madrid 2021'] and list(inputs) == ['es-md']


def test_read_elections_labels_the_month_when_the_year_repeats(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    writer = bundle.BundleWriter(fs)
    for part, schema in (('meta', 'event-meta'), ('cycle', 'event-cycle'), ('results', 'event-results')):
        content = bundle.BundleReader(fs).read_json(bundle.path_event('es-md', '2021-05-04', part))['data']
        if part == 'meta':
            content = content | {'event_date': '2021-11-04'}
        writer.write_json(bundle.path_event('es-md', '2021-11-04', part), schema, content, 'es-md')
    elections, _ = data.read_elections(bundle.BundleReader(fs), PARAMS, NAMES)
    assert [e.label for e in elections if e.scope == 'es-md'] == ['Madrid may. 2021', 'Madrid nov. 2021']


def test_read_current_reads_the_latest_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    current, inputs = data.read_current(bundle.BundleReader(fs), PARAMS, NAMES)
    assert inputs == {'run_id': HERD_RUN, 'as_of': '2026-11-05', 'event_date': '2026-11-29'}
    assert (current.scope, current.event_date, current.label, current.kind) == (
        'es', '2026-11-29', 'Generales 2026', 'generales')
    assert current.current and current.as_of == '2026-11-05' and len(current.polls) == 8
    assert current.names == ['PP', 'PSOE']


def test_read_current_none_without_manifest_or_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    assert data.read_current(bundle.BundleReader(fs), PARAMS, NAMES) == (None, {'skipped': 'no manifest'})
    write_herd_bundle(fs)
    from mtpy.lib import publish
    publish.update_manifest(bundle.BundleWriter(fs), {'es': {'latest': None}})
    assert data.read_current(bundle.BundleReader(fs), PARAMS, NAMES) == (None, {'skipped': 'no run'})


def test_read_current_none_once_the_election_is_published(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    writer = bundle.BundleWriter(fs)
    reader = bundle.BundleReader(fs)
    cycle = reader.read_json(bundle.path_part('es', HERD_RUN, 'polls'))['data']
    writer.write_json(bundle.path_event('es', '2026-11-29', 'cycle'), 'event-cycle', cycle, 'es')
    writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results',
                      {'n_seats': 350, 'rows': []}, 'es')
    assert data.read_current(reader, PARAMS, NAMES) == (None, {'skipped': 'election published'})
