"""La foto del estudio CERA sobre el run publicado: se reconstruye el run, se comprueba que es el publicado simulación
a simulación y se miden los escenarios de voto exterior sobre sus sorteos (`cera/majority.json`)."""
import dataclasses
import json

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.studies.cera import snapshot
from mtpy.lib.studies.cera.data import MAJORITY, check_majority
from mtpy.lib.studies.cera.snapshot import SnapshotRefused, take_snapshot
from tests import fakes
from tests.test_publish_unit import make_writer

RUN = '20261124-102045'


def consistent():
    """El simulador sintético tras un run de verdad: sus `results` son el reparto de sus `units`."""
    sim = fakes.synthetic_simulator()
    sim.run(reset=False, n_sim=200)
    return sim


def published(folder, sim, **meta):
    """Publica en `folder`, como `publish_mode`, el `meta` del run y el `dist` y el `scenario` del modo `forecast` de
    `sim`; devuelve un `BundleReader` sobre la misma carpeta."""
    writer = make_writer(folder)
    data = publish.export_meta(sim, RUN, '2026-11-24T10:20:45Z', 200, 10, publish.DEFAULT_CORRECTORS, {}, {}, None,
                               None, {'commit': 'abc1234', 'dirty': False, 'versions': {}}, False)
    data.update(scope='es', run_id=RUN, event_date='2026-11-29', as_of='2026-11-23', n_sim=200, seed=42, majority=6,
                bmaps={**data['bmaps'], 'blocks': fakes.BMAPS['blocks']})
    data.update(meta)
    writer.write_json(bundle.path_part('es', RUN, 'meta'), 'meta', data, 'es', RUN)
    for part, exporter in (('dist', publish.export_dist), ('scenario', publish.export_scenario)):
        writer.write_json(bundle.path_part('es', RUN, part, 'forecast'), part, exporter(sim)[0], 'es', RUN, 'forecast')
    return bundle.BundleReader(FileSystem(str(folder)))


def bare(study):
    """El estudio sin foto, sin tocar la fixture de la sesión."""
    return dataclasses.replace(study, majority=None)


def snap(tmp_path, sim, study, scope='es', **kwargs):
    kwargs.setdefault('prov', lambda: {'commit': 'abc1234', 'dirty': False})
    kwargs.setdefault('reader', published(tmp_path, consistent()))
    kwargs.setdefault('run_id', RUN)
    return take_snapshot(scope, data=FileSystem(str(tmp_path / 'data')), rebuild=lambda meta, **_: sim,
                         compute=lambda: bare(study), today='2026-11-25', **kwargs)


def test_take_snapshot_writes_the_file_from_the_published_run(tmp_path, cera_study):
    """La foto sale del run publicado, con el commit y el día en que se toma, y es válida para el estudio."""
    out = snap(tmp_path, consistent(), cera_study)
    written = json.loads((tmp_path / 'data' / MAJORITY).read_text(encoding='utf-8'))
    assert out['status'] == 'saved' and written['run'] == {
        'scope': 'es', 'event_date': '2026-11-29', 'run_id': RUN, 'as_of': '2026-11-23', 'n_sim': 200, 'seed': 42,
        'commit': 'abc1234', 'taken_on': '2026-11-25'}
    assert check_majority(written, cera_study.params) == []
    assert out['base']['with_majority'] == written['base']['with_majority'] and out['n_sim'] == 200
    assert [row['region_id'] for row in written['expected']['convocatoria']['provinces']] == [28, 8]


def test_take_snapshot_reports_its_phases_and_the_expected_vote(tmp_path, cera_study):
    """Cada fase pasa por `progress` y `report` da, a nivel 1, la base y el voto esperado de cada censo."""
    phases, lines = [], []
    out = snap(tmp_path, consistent(), cera_study, progress=lambda step, seconds: phases.append(step),
               report=lambda text, level: lines.append((text, level)))
    assert phases == ['read', 'rebuild', 'check', 'scenarios', 'save']
    assert set(out['seconds']) == {'read', 'rebuild', 'check', 'scenarios', 'save', 'total'}
    assert out['action'] == 'majority' and out['run_id'] == RUN and out['commit'] == 'abc1234'
    assert out['dirty'] is False and out['path'].endswith(MAJORITY)
    written = json.loads((tmp_path / 'data' / MAJORITY).read_text(encoding='utf-8'))
    central = written['base']['central']
    assert out['base'] == {'with_majority': written['base']['with_majority'], 'right': central['Derecha'],
                           'left': central['Izquierda']}
    counts = written['expected']['convocatoria']['counts']
    assert out['expected']['convocatoria'] == {key: counts[key] for key in ('votes', 'lost', 'gained', 'moved')}
    assert set(out['expected']) == {'convocatoria', 'sin_litigio'}
    assert len(lines) == 3 and all(level == 1 for _, level in lines)
    assert str(counts['votes']) in lines[1][0] + lines[2][0]


def test_take_snapshot_refuses_a_run_that_differs_in_one_simulation(tmp_path, cera_study):
    """Un run reconstruido que difiere del publicado en una sola simulación se rechaza y no se escribe nada."""
    sim = consistent()
    sim.results[7, 1] = [6, 0, 0]  # the six seats of Madrid to one party, in one simulation
    sim.results[7, 0] = sim.results[7, 1:].sum(axis=0)
    with pytest.raises(SnapshotRefused, match='1 of 200'):
        snap(tmp_path, sim, cera_study)
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_refuses_when_the_allocation_does_not_reproduce_the_run(tmp_path, cera_study):
    """Si los votos que salen del simulador no reparten los escaños guardados, la foto se rechaza."""
    sim = consistent()
    vpred = sim.cols_unit.index('vpred_pct')
    # PP and PSOE swapped in Madrid in one simulation: the stored seats (and so the published run) do not change
    sim.units[7, 1, [0, 1], vpred] = sim.units[7, 1, [1, 0], vpred]
    with pytest.raises(SnapshotRefused, match='does not reproduce'):
        snap(tmp_path, sim, cera_study)
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_refuses_when_the_central_allocation_differs(tmp_path, cera_study):
    """Un reparto central publicado distinto del reconstruido se rechaza."""
    other = consistent()
    other.central_result()
    other.central_table.iloc[[0, 1], 0] -= 1  # one seat of Madrid from PP to PSOE
    other.central_table.iloc[[0, 1], 1] += 1
    with pytest.raises(SnapshotRefused, match='scenario'):
        snap(tmp_path, consistent(), cera_study, reader=published(tmp_path / 'bundle', other))
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_refuses_a_scope_threshold(tmp_path, cera_study):
    """Con un umbral sobre todo el ámbito el reparto de la foto no sería el del simulador: se rechaza."""
    sim = consistent()
    sim.threshold_scope = 5.0
    with pytest.raises(SnapshotRefused, match='threshold'):
        snap(tmp_path, sim, cera_study)
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_refuses_another_scope_or_election(tmp_path, cera_study):
    """La foto es del ámbito y de la elección del estudio: otro ámbito u otra fecha se rechazan."""
    with pytest.raises(SnapshotRefused, match='scope'):
        snap(tmp_path, consistent(), cera_study, scope='es-md')
    with pytest.raises(SnapshotRefused, match='event_date'):
        snap(tmp_path, consistent(), cera_study,
             reader=published(tmp_path / 'bundle', consistent(), event_date='2027-05-23'))
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_does_not_overwrite_and_a_dry_run_writes_nothing(tmp_path, cera_study):
    """La foto no se pisa sin `overwrite`; sin `save` se comprueba todo y no se escribe nada."""
    assert snap(tmp_path, consistent(), cera_study)['status'] == 'saved'
    with pytest.raises(SnapshotRefused, match='exists'):
        snap(tmp_path, consistent(), cera_study)
    assert snap(tmp_path, consistent(), cera_study, overwrite=True)['status'] == 'saved'
    fresh = tmp_path / 'fresh'
    assert snap(fresh, consistent(), cera_study, save=False)['status'] == 'checked'
    assert not (fresh / 'data' / MAJORITY).exists()


def test_take_snapshot_needs_a_clean_tree_to_write_but_not_to_check(tmp_path, cera_study):
    """Para escribir la foto el árbol ha de estar limpio; para comprobarla no."""
    dirty = lambda: {'commit': 'abc1234', 'dirty': True}  # noqa: E731
    with pytest.raises(SnapshotRefused, match='clean tree'):
        snap(tmp_path, consistent(), cera_study, prov=dirty)
    assert not (tmp_path / 'data' / MAJORITY).exists()
    out = snap(tmp_path, consistent(), cera_study, prov=dirty, save=False)
    assert out['status'] == 'checked' and out['dirty'] is True
    assert not (tmp_path / 'data' / MAJORITY).exists()


def test_take_snapshot_takes_the_latest_run_and_refuses_without_one(tmp_path, cera_study):
    """Sin `run_id` se toma el `latest` del manifest; sin run publicado se rechaza."""
    reader = published(tmp_path, consistent())
    publish.update_manifest(make_writer(tmp_path), scopes={'es': {'latest': RUN}})
    assert snap(tmp_path, consistent(), cera_study, reader=reader, run_id=None)['run_id'] == RUN
    empty = bundle.BundleReader(FileSystem(str(tmp_path / 'empty')))
    with pytest.raises(SnapshotRefused, match='no published run'):
        snap(tmp_path / 'other', consistent(), cera_study, reader=empty, run_id=None)
    assert not (tmp_path / 'other' / 'data' / MAJORITY).exists()


def test_rebuild_run_follows_the_recipe_of_the_publication_with_the_meta_of_the_run(tmp_path):
    """El run se reconstruye con los parámetros de su `meta`, el ancla en su `as_of`, y se corre sólo el `forecast`."""
    meta = published(tmp_path, consistent()).read_json(bundle.path_part('es', RUN, 'meta'))['data']
    factory = fakes.FakeSimulator()
    sim = snapshot.rebuild_run(meta, simulator=factory)
    init = factory.calls[0][3]
    assert factory.calls[0][1:3] == ('es', '2026-11-29') and init['today'] == '2026-11-23'
    assert init['drange'] == (6, None) and init['seed'] == 42 and init['mode'] == 'nowcast'
    assert 'limit_date' not in init and init['swing'] == meta['correctors']['swing']
    assert [call[0] for call in factory.calls] == ['init', 'fit_forecast', 'run']
    assert factory.calls[-1][1:] == ('forecast', {'split': True, 'random': True, 'n_sim': 200, 'horizon': 'deadline'})
    assert sim.mode == 'forecast'


def test_take_snapshot_overwrites_a_file_the_study_cannot_read(tmp_path, cera_study, monkeypatch):
    """Por defecto el estudio se calcula sin abrir la foto, así que una foto ilegible se puede pisar."""
    data = FileSystem(str(tmp_path / 'data'))
    data.write_bytes(b'{}', MAJORITY)
    calls = []

    def recorder(fs, params=None, majority=True):
        calls.append((fs, majority))
        return bare(cera_study)

    monkeypatch.setattr(snapshot, 'compute_cera', recorder)
    sim = consistent()
    out = take_snapshot('es', run_id=RUN, overwrite=True, reader=published(tmp_path, consistent()), data=data,
                        rebuild=lambda meta, **_: sim, prov=lambda: {'commit': 'abc1234', 'dirty': False},
                        today='2026-11-25')
    assert out['status'] == 'saved' and calls == [(data, False)]
    assert check_majority(json.loads((tmp_path / 'data' / MAJORITY).read_text(encoding='utf-8')),
                          cera_study.params) == []
