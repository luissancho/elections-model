"""Tests de `mtpy/lib/studies/lean/data.py`: lectura de las entradas del estudio del sesgo del CIS desde el paquete."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, webapi
from mtpy.lib.studies.lean import data
from tests.fakes import bundle_fixture, write_lean_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CURRENT = {'CIS', 'GAD3', 'Sigma Dos', 'NC Report'}


@pytest.fixture
def catalogue(fresh_app):
    """Catálogo de ámbitos del repositorio, leído como lo lee la web (`App.data` apuntando a `data/`)."""
    fresh_app.data = FileSystem(os.path.join(ROOT, 'data'))

    return webapi.checks.catalogue()


def test_read_error_rows_keeps_only_published_events(catalogue, tmp_path):
    fs = FileSystem(str(tmp_path))
    expected = write_lean_bundle(fs)
    rows, inputs = data.read_error_rows(bundle.BundleReader(fs), catalogue)
    assert [r['event_date'] for r in rows if r['pollster'] == 'CIS' and r['scope'] == 'es'] == expected['cis_es_dates']
    assert '2012-11-25' not in {r['event_date'] for r in rows}
    cis23 = next(r for r in rows if r['pollster'] == 'CIS' and r['event_date'] == '2023-07-23')
    assert (cis23['label'], cis23['short'], cis23['year'], cis23['lean']) == ('Generales 2023', 'GEN 23', 2023, 5.9)
    an18 = next(r for r in rows if r['scope'] == 'es-an' and r['event_date'] == '2018-12-02' and r['pollster'] == 'CIS')
    assert (an18['label'], an18['short'], an18['scope_name']) == ('Andalucía 2018', 'AN 18', 'Andalucía')
    for code in ('es', 'es-an'):
        assert inputs[code]['generated_at'] and inputs[code]['run_id'] is None
    assert inputs['es-ct'] == {'skipped': 'no analysis'}
    assert rows[0]['scope'] == 'es' and rows[-1]['scope'] == 'es-an'


def test_read_error_rows_keeps_a_missing_block_as_none(catalogue, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    writer = bundle.BundleWriter(fs)
    row = {'event_date': '2019-04-28', 'event': '2019-28A', 'pollster': 'CIS', 'days': 5, 'sample_size': None,
           'weight': 1, 'Izquierda': 1.5}
    writer.write_json(bundle.path_analysis('es-nc', 'poll-errors'), 'poll-errors', {'blocks': [], 'rows': [row]},
                      'es-nc')
    writer.write_json(bundle.path_analysis('es-nc', 'meta'), 'analysis-meta',
                      bundle_fixture('analysis-meta') | {'scope': 'es-nc'}, 'es-nc')
    writer.write_json(bundle.path_event('es-nc', '2019-04-28', 'meta'), 'event-meta',
                      bundle_fixture('event-meta') | {'scope': 'es-nc'}, 'es-nc')
    rows, _ = data.read_error_rows(bundle.BundleReader(fs), catalogue)
    nc = next(r for r in rows if r['scope'] == 'es-nc')
    assert nc['Derecha'] is None and nc['lean'] is None and nc['short'] == 'NC 19'


def test_read_election_needs_its_three_parts(tmp_path):
    fs = FileSystem(str(tmp_path))
    expected = write_lean_bundle(fs)
    election = data.read_election(bundle.BundleReader(fs), 'es', '2023-07-23')
    assert set(election) == {'meta', 'polls', 'results'} and election['meta']['blocks']['Izquierda'] == ['PSOE', 'SUMAR']
    assert election == expected['election']
    with pytest.raises(data.MissingInput) as info:
        data.read_election(bundle.BundleReader(fs), 'es', '2019-11-10')
    assert info.value.part == 'events/es/2019-11-10/polls'


def test_active_pollsters_without_a_national_run(catalogue, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    rows, _ = data.read_error_rows(bundle.BundleReader(fs), catalogue)
    assert data.active_pollsters(bundle.BundleReader(fs), rows, '2026-01-01')[:2] == (CURRENT, 'run+events')
    assert data.active_pollsters(bundle.BundleReader(fs), rows, '2026-01-01')[2]
    fs.remove(bundle.BundleReader(fs).path(bundle.path_manifest()))
    assert data.active_pollsters(bundle.BundleReader(fs), rows, '2026-01-01') == (CURRENT, 'events', None)


def test_load_params_reads_the_repository_file():
    params = data.load_params(FileSystem(os.path.join(ROOT, 'data')))
    assert params['pollster'] == 'CIS' and params['since'] == '2018-09-01'
    assert params['control'] == ['2008-01-01', '2017-01-01']
    assert params['min_elections'] == 10 and params['max_days'] == 60 and params['active_from'] == '2026-01-01'
    assert params['election'] == {'scope': 'es', 'date': '2023-07-23'} and params['blocks'] == ['Izquierda', 'Derecha']
    assert params['labels'] == {'with': '2018 – 2026', 'without': '2008 – 2016'}
