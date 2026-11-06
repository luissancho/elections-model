"""Tests de `mtpy/lib/studies/ban/data.py` y de `mtpy/lib/studies/labels.py`: parámetros, etiquetas y lectura de las
entradas del estudio de los sondeos «clandestinos» desde el paquete, con la guarda legal."""
import json
import os
from dataclasses import asdict

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.ban import data
from mtpy.lib.studies.ban.study import compute_ban
from mtpy.lib.studies.labels import election_code, election_label, short_year_label
from tests.fakes import BAN_CURRENT, BAN_CURRENT_SHARE, write_ban_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = FileSystem(os.path.join(ROOT, 'data'))
PARAMS = data.load_params(DATA)


def test_load_params_reads_the_repository_file():
    assert PARAMS == {'national': 'es', 'since': '1996-01-01', 'chart_since': '2015-01-01', 'max_days': 42,
                      'frozen_horizon': 6, 'min_pairs': 3, 'big_sample': 5000, 'exclude_mtypes_pairs': ['aggr'],
                      'exclude_mtypes_consensus': ['aggr', 'online'], 'half': 0.5}


def test_labels_moved_from_lean_keep_their_output():
    assert election_label('es', None, '2019-11-10', month=True) == 'Generales nov. 2019'
    assert election_label('es-an', 'Andalucía', '2018-12-02') == 'Andalucía 2018'
    assert election_code('es', '2023-07-23') == 'GEN 23' and election_code('es', '2019-11-10', month=True) == 'GEN 19N'
    assert short_year_label('2019-04-28', True) == '2019 abr.' and short_year_label('2023-07-23', False) == '2023'


def test_read_events_reads_every_scope_and_skips_the_election_without_results(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    events, inputs = data.read_events(bundle.BundleReader(fs), PARAMS)
    assert [(e.scope, e.event_date) for e in events] == [('es', '2019-11-10'), ('es-md', '2021-05-04'),
                                                         ('es', '2023-07-23')]
    assert [e.label for e in events] == ['2019', 'Madrid 2021', '2023'] and [e.national for e in events] == [
        True, False, True]
    assert inputs['es']['skipped'] == {BAN_CURRENT: 'no results'} and inputs['es-md']['skipped'] == {}
    assert [row['event_date'] for row in inputs['es']['read']] == ['2019-11-10', '2023-07-23']
    assert inputs['es']['read'][0]['parts'] == ['meta', 'ban', 'results', 'pollsters', 'model']
    assert inputs['es-md']['read'][0]['parts'] == ['meta', 'ban', 'results', 'pollsters']
    first = events[0]
    assert first.main == ['PSOE', 'PP', 'VOX'] and first.name == 'Generales nov. 2019' and len(first.polls) == 3
    assert list(first.frozen.index) == ['PSOE', 'PP', 'VOX'] and first.frozen.loc['VOX', 'model'] == 13.9
    assert first.with_ban.loc['PP', 'model'] == 19.9 and first.official.to_dict() == {'PSOE': 28.0, 'PP': 20.8,
                                                                                     'VOX': 15.1}
    assert events[2].with_ban is None and events[1].frozen is None and events[1].pollsters is not None


def test_read_events_never_reads_the_parts_of_an_election_without_results(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    reader = bundle.BundleReader(fs)
    read = []
    original = reader.read_json
    reader.read_json = lambda name: read.append(name) or original(name)
    data.read_events(reader, PARAMS)
    assert read and not [name for name in read if BAN_CURRENT in name or name.startswith('runs/')]


def test_read_event_without_the_frozen_horizon_has_no_model(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    event = data.read_event(bundle.BundleReader(fs), 'es', '2019-11-10', PARAMS | {'frozen_horizon': 14}, False)
    assert event.frozen is None and event.with_ban is None and len(event.polls) == 3
    event = data.read_event(bundle.BundleReader(fs), 'es', '2019-11-10', PARAMS | {'frozen_horizon': 30}, True)
    assert event.frozen.loc['PSOE', 'model'] == 32.4 and event.label == '2019 nov.'


def test_read_event_without_results_or_ban_is_none(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    bundle.BundleWriter(fs).remove(bundle.path_event('es-md', '2021-05-04', 'ban'))
    reader = bundle.BundleReader(fs)
    assert data.read_event(reader, 'es', BAN_CURRENT, PARAMS, False) is None
    assert data.read_event(reader, 'es-md', '2021-05-04', PARAMS, False) is None
    events, inputs = data.read_events(reader, PARAMS)
    assert inputs['es-md'] == {'read': [], 'skipped': {'2021-05-04': 'no ban'}} and len(events) == 2


def test_list_scopes_puts_the_national_scope_first(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    assert data.list_scopes(bundle.BundleReader(fs)) == ['es', 'es-md']
    assert data.list_scopes(bundle.BundleReader(FileSystem(str(tmp_path / 'empty')))) == []


def test_read_next_reports_the_run_election_and_whether_it_has_results(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    assert data.read_next(bundle.BundleReader(fs), PARAMS) == {'event_date': BAN_CURRENT, 'open': False}
    writer = bundle.BundleWriter(fs)
    writer.write_json(bundle.path_event('es', BAN_CURRENT, 'results'), 'event-results', {'n_seats': 350, 'rows': []},
                      'es')
    assert data.read_next(bundle.BundleReader(fs), PARAMS) == {'event_date': BAN_CURRENT, 'open': True}
    assert data.read_next(bundle.BundleReader(FileSystem(str(tmp_path / 'empty'))), PARAMS) is None


def test_compute_ban_ignores_an_election_without_results(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    study = compute_ban(bundle.BundleReader(fs), DATA)
    parts = asdict(study)
    published = json.dumps({key: parts[key] for key in ('summary', 'pairs', 'elections', 'period', 'cells', 'houses',
                                                        'big', 'latest')})
    assert BAN_CURRENT not in published and str(BAN_CURRENT_SHARE) not in published
    assert study.inputs['es']['skipped'][BAN_CURRENT] == 'no results' and study.next == {'event_date': BAN_CURRENT,
                                                                                         'open': False}
