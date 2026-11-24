"""Los votos de cada escenario del estudio CERA y la foto que los mide sobre los sorteos del modelo."""
import copy
import json
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import publish
from mtpy.lib.studies.cera.data import MAJORITY, check_majority, read_majority
from mtpy.lib.studies.cera.scenarios import (
    build_snapshot,
    exterior_change,
    expected_table,
    expected_votes,
    new_elector_votes,
    placement_change,
)
from mtpy.lib.studies.cera.study import GRID_PARTICIPATIONS, PLACEMENT_CANDIDATURE
from tests.cera_doubles import META, double_draws
from tests.conftest import ROOT


def test_the_votes_of_each_scenario_add_up_to_what_the_spec_says(cera_study):
    """Los votos de cada escenario suman lo que dice la spec y la tabla coincide con la que publica el estudio."""
    study = cera_study
    assert round(expected_votes(study, 'convocatoria').values.sum()) == 30482
    assert round(expected_votes(study, 'sin_litigio').values.sum()) == 22412
    table = expected_table(study, 'convocatoria')
    assert table == {'census': [2327388, 2769645], 'new_electors': 442257, 'voters': [208431, 239221],
                     'votes': [203152, 233634]}
    published = publish.export_study_prediction(study)[0]['scenarios']['sin_litigio']
    assert expected_table(study, 'sin_litigio')['voters'] == [published['voters_2023'], published['voters_projected']]
    assert [round(new_elector_votes(study, p, 0.8).values.sum()) for p in (0.09, 0.5, 0.7)] == [39405, 218917, 306484]
    cera = study.cera[study.params['baseline_scrutiny']]
    assert exterior_change(study, 0.3, 0.6).values.sum() == pytest.approx(2769645 * 0.3 * 0.99 - cera.values.sum())
    plan = placement_change(study, 1.0, 1.0)
    assert list(plan.columns) == [PLACEMENT_CANDIDATURE] and plan.values.sum() == pytest.approx(181032 * 0.99)
    assert int(study.placement['electors'].sum()) == study.placement['pool'] == 181032


def test_build_snapshot_has_every_scenario_on_its_axes(cera_study):
    """La foto trae cada escenario sobre sus ejes, con las claves de la spec y sólo tipos de JSON."""
    snapshot = build_snapshot(double_draws(cera_study), cera_study, META)
    assert set(snapshot) == {'run', 'majority', 'blocks', 'lineage', 'base', 'expected', 'new_electors', 'grid',
                             'placement'}
    assert snapshot['run'] == {key: META[key] for key in ('scope', 'event_date', 'run_id', 'as_of', 'n_sim', 'seed')}
    assert snapshot['blocks'] == META['bmaps']['blocks'] and snapshot['majority'] == 176
    assert snapshot['new_electors']['participations'] == snapshot['grid']['participations'] == GRID_PARTICIPATIONS
    assert [len(snapshot[key]['cells']) for key in ('new_electors', 'grid', 'placement')] == [5, 5, 4]
    assert len(snapshot['placement']['cells'][0]) == 5 and snapshot['placement']['pool'] == 181032
    assert snapshot['grid']['electors'] == 2769645 and snapshot['new_electors']['electors'] == 442257
    assert set(snapshot['expected']) == {'convocatoria', 'sin_litigio'}
    provinces = snapshot['expected']['convocatoria']['provinces']
    assert len(provinces) == 52 and set(provinces[0]) == {'region_id', 'name', 'moved', 'right_lost', 'right_gained'}
    assert snapshot['lineage']['SUMAR'] == 'FA' and snapshot['lineage']['UPN'] == 'PP'
    cell = snapshot['grid']['cells'][0][0]
    assert set(cell) == {'votes', 'lost', 'gained', 'moved', 'right', 'left', 'central'}
    assert sum(cell['right'].values()) == 20 and json.loads(json.dumps(snapshot)) == snapshot


@pytest.fixture(scope='module')
def good(cera_study):
    """Una foto válida: la de los sorteos sintéticos con el commit y el día que pone quien la toma."""
    snapshot = build_snapshot(double_draws(cera_study), cera_study, META)
    snapshot['run'].update(commit='abc1234', taken_on='2026-11-25')
    return snapshot


def test_read_majority_is_none_without_the_file_and_gives_back_a_valid_one(tmp_path, cera_study, good):
    """Sin fichero no hay foto; con una válida se lee tal cual."""
    fs = FileSystem(str(tmp_path))
    assert read_majority(fs, cera_study.params) is None and check_majority(good, cera_study.params) == []
    fs.write_bytes(json.dumps(good).encode(), MAJORITY)
    assert read_majority(fs, cera_study.params) == good


def test_read_majority_refuses_a_snapshot_that_is_not_this_study_s(tmp_path, cera_study, good):
    """Una foto de otra elección, con tablas mal medidas o sin una clave se rechaza nombrando lo que falla."""
    fs = FileSystem(str(tmp_path))
    for change, word in ((lambda s: s['run'].update(event_date='2030-01-01'), 'event_date'),
                         (lambda s: s['grid']['cells'].pop(), 'grid'),
                         (lambda s: s['expected']['convocatoria']['counts'].update(lost=21), 'lost'),
                         (lambda s: s['expected']['sin_litigio']['provinces'].pop(), 'provinces'),
                         (lambda s: s.pop('base'), 'base')):
        bad = copy.deepcopy(good)
        change(bad)
        fs.write_bytes(json.dumps(bad).encode(), MAJORITY)
        with pytest.raises(ValueError, match=word):
            read_majority(fs, cera_study.params)


def test_compute_cera_carries_the_snapshot_of_the_repository(cera_study):
    """El estudio trae la foto del repositorio (ninguna, hasta que se tome)."""
    assert cera_study.majority == read_majority(FileSystem(os.path.join(ROOT, 'data')), cera_study.params)
