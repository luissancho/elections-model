"""Tests de `mtpy/lib/studies/lean/study.py`: el estudio completo sobre el paquete sintético."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.pages import metodo
from mtpy.lib.studies.lean.data import MissingInput
from mtpy.lib.studies.lean.study import compute_lean
from tests.fakes import write_lean_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_compute_lean_on_the_synthetic_bundle(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    expected = write_lean_bundle(fs)
    study = compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})
    assert study.params['min_elections'] == 3 and study.params['pollster'] == 'CIS'
    assert study.periods['with'] == {'from': '2018-09-01', 'to': None, 'label': '2018 – 2026',
                                     'first_event': '2018-12-02', 'last_event': '2026-05-17', 'n': 6}
    assert study.periods['without']['n'] == 4 and study.periods['without']['label'] == '2008 – 2017'
    assert study.periods['without']['to'] == '2018-01-01'
    assert study.summary['with']['k_left'] == 5 and study.summary['without']['k_left'] == 2
    assert study.summary['election'] == expected['summary_election']
    assert [e['period'] for e in study.elections].count('with') == 6
    assert study.elections[0]['event_date'] == '2026-05-17'
    assert {'consensus', 'n_others', 'others_left'} <= set(study.elections[0])
    dates = [e['event_date'] for e in study.elections]
    assert dates == sorted(dates, reverse=True) and len(study.elections) == 10
    assert [h['pollster'] for h in study.houses] == expected['houses'] and study.matrix['houses'][0] == 'CIS'
    assert study.election['date'] == '2023-07-23' and study.election['scope'] == 'es'
    assert study.election['name'] == 'Generales 2023'
    assert study.inputs['active_source'] == 'run+events' and study.inputs['run_id']
    assert study.inputs['election']['generated_at'] and study.inputs['scopes']['es']['run_id'] is None
    assert len(study.rows) > len(study.elections)


def test_compute_lean_without_a_national_run_has_no_run_id(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    reader = bundle.BundleReader(fs)
    fs.remove(reader.path(bundle.path_manifest()))
    study = compute_lean(reader, FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})
    assert study.inputs['active_source'] == 'events' and study.inputs['run_id'] is None


def test_compute_lean_fails_without_the_national_analysis(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    reader = bundle.BundleReader(fs)
    fs.remove(reader.path(bundle.path_analysis('es', 'meta')))
    with pytest.raises(MissingInput) as info:
        compute_lean(reader, FileSystem(os.path.join(ROOT, 'data')))
    assert info.value.part == 'analysis/es'


def test_compute_lean_matches_the_method_card(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_lean_bundle(fs)
    fresh_app.fs = fs
    fresh_app.data = FileSystem(os.path.join(ROOT, 'data'))
    study = compute_lean(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params={'min_elections': 3})
    card = metodo.house_effects_example()
    assert (card['n'], card['over']['Izquierda']) == (study.summary['with']['n'], study.summary['with']['k_left'])
