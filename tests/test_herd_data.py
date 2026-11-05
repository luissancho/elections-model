"""Tests de `mtpy/lib/studies/herd/data.py`: parámetros y lectura de las entradas del estudio del rebaño desde el
paquete."""
import os

import pandas as pd

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.herd import data
from tests.fakes import write_herd_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS = data.load_params(FileSystem(os.path.join(ROOT, 'data')))


def test_load_params_reads_the_repository_file():
    params = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
    assert params['scope'] == 'es' and params['since'] == '2008-01-01' and params['max_days'] == 365
    assert (params['window_days'], params['final_days'], params['local_window']) == (56, 21, 10)
    assert (params['n_series'], params['min_polls'], params['exclude']) == (4, 3, ['CIS'])
    assert params['buckets'] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 11], [12, 17], [18, 26], [27, 52]]


def test_read_cycles_reads_the_published_generals(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    assert [c.event_date for c in cycles] == ['2019-11-10', '2023-07-23']
    assert [c.label for c in cycles] == ['2019', '2023']
    assert cycles[0].curve is None and cycles[1].curve.index.min() == pd.Timestamp('2022-07-23')
    assert cycles[1].polls.shape[0] == 14
    assert cycles[1].errors['GAD3'] == {'last_error_avg': 1.5, 'last_error_blocks': -2.0}
    assert cycles[0].errors['NC Report']['last_error_blocks'] == -1.0
    assert inputs['2023-07-23']['series'] and inputs['2019-11-10']['series'] is None


def test_read_cycles_casts_the_polls(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    cycle = cycles[1]
    assert list(cycle.polls.columns) == ['date', 'pollster_id', 'pollster', 'start_date', 'end_date', 'sample_size',
                                         'weight', 'PP', 'PSOE', 'VOX', 'SUMAR', 'Cs']
    assert cycle.parties == ['PP', 'PSOE', 'VOX', 'SUMAR', 'Cs'] and not cycle.current
    assert cycle.polls['sample_size'].dtype == float and cycle.polls['PP'].dtype == float
    panel = cycle.polls[cycle.polls['pollster'] == 'Panel X'].iloc[0]
    assert pd.isna(panel['sample_size']) and panel['weight'] == 0.5 and panel['end_date'] == '2023-07-10'
    assert cycle.curve.loc[pd.Timestamp('2023-07-17'), 'SUMAR'] == 12.0
    assert cycle.inputs == inputs['2023-07-23'] and inputs['2023-07-23']['pollsters']


def test_read_cycles_skips_an_event_without_the_cycle_part(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    bundle.BundleWriter(fs).remove(bundle.path_event('es', '2019-11-10', 'cycle'))
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS)
    assert [c.event_date for c in cycles] == ['2023-07-23'] and inputs['2019-11-10'] == {'skipped': 'no cycle part'}


def test_read_cycles_keeps_the_dates_since_the_parameter(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    cycles, inputs = data.read_cycles(bundle.BundleReader(fs), PARAMS | {'since': '2020-01-01'})
    assert [c.event_date for c in cycles] == ['2023-07-23'] and list(inputs) == ['2023-07-23']


def test_read_current_reads_the_latest_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    current, inputs = data.read_current(bundle.BundleReader(fs), PARAMS)
    assert current.current and current.event_date == '2026-11-29' and current.label == '2026'
    assert current.polls.shape[0] == 7 and current.curve.index.max() == pd.Timestamp('2026-11-05')
    assert current.errors == {}
    assert inputs == {'run_id': '20261105-120000', 'as_of': '2026-11-05', 'event_date': '2026-11-29'}
    assert current.parties == ['PP', 'PSOE', 'VOX', 'FA', 'Cs'] and current.inputs == inputs


def test_read_current_is_none_without_a_run_or_once_the_election_is_published(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    writer = bundle.BundleWriter(fs)   # the election of the run published with results
    writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results', {'n_seats': 350, 'rows': []},
                      'es')
    assert data.is_published(bundle.BundleReader(fs), 'es', '2026-11-29')
    assert data.read_current(bundle.BundleReader(fs), PARAMS) == (None, {'skipped': 'election published'})
    writer.remove(bundle.path_manifest())
    assert data.read_current(bundle.BundleReader(fs), PARAMS) == (None, {'skipped': 'no manifest'})


def test_read_current_is_none_without_the_scope_in_the_manifest(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    assert not data.is_published(bundle.BundleReader(fs), 'es', '2026-11-29')
    assert data.read_current(bundle.BundleReader(fs), PARAMS | {'scope': 'es-an'}) == (None, {'skipped': 'no run'})


def test_read_herding_without_the_part(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    rows, inputs = data.read_herding(bundle.BundleReader(fs), 'es')
    assert rows[('2023-07-23', 'GAD3')] == {'n': 8, 'ratio': 0.61, 'p_value': 0.05} and inputs['generated_at']
    bundle.BundleWriter(fs).remove(bundle.path_analysis('es', 'herding'))
    assert data.read_herding(bundle.BundleReader(fs), 'es') == ({}, {'skipped': 'no herding part'})


def test_cycle_label_adds_the_month_only_when_the_year_repeats():
    dates = ['2019-04-28', '2019-11-10', '2023-07-23']
    assert [data.cycle_label(d, dates) for d in dates] == ['2019 abr.', '2019 nov.', '2023']


def test_missing_input_names_the_part():
    error = data.MissingInput('events/es')
    assert error.part == 'events/es' and str(error) == 'missing input: events/es'
