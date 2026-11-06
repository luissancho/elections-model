"""Tests de `mtpy/lib/studies/ban/study.py`: el estudio de los sondeos «clandestinos» entero sobre el paquete
sintético de `write_ban_bundle`, con las cifras calculadas a mano."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.ban.data import MissingInput
from mtpy.lib.studies.ban.study import compute_ban
from tests.fakes import BAN_EVENTS, write_ban_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = FileSystem(os.path.join(ROOT, 'data'))


def _rounded(block):
    """A summary block with its floats rounded to 2 decimals (3 significant figures for ``p``)."""
    return {key: float(f'{value:.3g}') if key == 'p' else round(value, 2) if isinstance(value, float) else value
            for key, value in block.items()}


def test_compute_ban_summary_matches_the_hand_figures(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    figures = write_ban_bundle(fs)
    study = compute_ban(bundle.BundleReader(fs), DATA)
    summary, expected = study.summary, figures['summary']
    pairs = summary['pairs']
    assert _rounded({key: value for key, value in pairs.items() if key not in ('generals', 'regional', 'by_scope')}) \
        == {key: value for key, value in expected['pairs'].items() if key not in ('generals', 'regional')}
    assert pairs['p'] == pytest.approx(0.8125)
    assert _rounded(pairs['generals']) == expected['pairs']['generals']
    assert _rounded(pairs['regional']) == expected['pairs']['regional']
    assert [_rounded(row) for row in pairs['by_scope']] == [{'scope': 'es', 'n': 4, 'k': 1, 'mean_d': -0.07},
                                                            {'scope': 'es-md', 'n': 1, 'k': 1, 'mean_d': -0.3}]
    assert _rounded(summary['elections']) == expected['elections']
    assert summary['elections']['mean_legal'] == pytest.approx((0.94 + 6.08 / 3) / 2)
    assert summary['elections']['mean_frozen'] == pytest.approx((2.3 / 3 + 1.5) / 2)
    cells = summary['cells']
    assert {key: value for key, value in cells.items() if key != 'missed'} == expected['cells']
    assert cells['missed'] == [{'label': '2019', 'party': 'PSOE'}, {'label': '2019', 'party': 'PP'},
                               {'label': '2023', 'party': 'VOX'}]
    assert _rounded(summary['big']) == expected['big'] and summary['exception'] is None
    assert study.period == {'since': '1996-01-01', 'k': 0, 'n': 0, 'mean_d': None}
    assert study.next == figures['next'] and study.houses == []
    assert [(row['event_date'], row['pollster']) for row in study.pairs] == [
        ('2023-07-23', 'GAD3'), ('2023-07-23', '40dB'), ('2021-05-04', 'Sigma Dos'), ('2019-11-10', 'GAD3'),
        ('2019-11-10', 'ElectoPanel')]
    assert [(row['scope'], row['event_date']) for row in study.elections] == [
        ('es', '2023-07-23'), ('es-md', '2021-05-04'), ('es', '2019-11-10')]
    assert len(study.cells) == 7 and [row['pollster'] for row in study.big] == ['GAD3', 'GAD3']
    assert study.params['frozen_horizon'] == 6


def test_compute_ban_houses_with_a_lower_minimum(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    study = compute_ban(bundle.BundleReader(fs), DATA, {'min_pairs': 2})
    assert [(row['pollster'], row['n'], row['k'], row['share']) for row in study.houses] == [('GAD3', 2, 1, 0.5)]
    assert study.houses[0]['mean_d'] == pytest.approx(-0.3) and study.params['min_pairs'] == 2


def test_compute_ban_without_any_pair_raises_missing_input(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    with pytest.raises(MissingInput) as error:
        compute_ban(bundle.BundleReader(fs), DATA, {'max_days': 5})
    assert error.value.part == 'pairs'


def test_compute_ban_without_any_model(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    writer = bundle.BundleWriter(fs)
    for scope, date in BAN_EVENTS:
        if writer.exists(bundle.path_event(scope, date, 'model')):
            writer.remove(bundle.path_event(scope, date, 'model'))
    study = compute_ban(bundle.BundleReader(fs), DATA)
    assert study.summary['elections'] is None and study.cells == []
    assert study.summary['cells'] == {'n': 0, 'k_closer': 0, 'k_direction': 0, 'k_crossed': 0, 'k_half': 0,
                                      'missed': []}
    assert all(row['model'] is None for row in study.elections) and study.summary['pairs']['n'] == 5
    assert study.latest['frozen'] is None and study.latest['error']['frozen'] is None
    assert study.latest['error']['consensus'] == pytest.approx(1.15)


def test_latest_is_the_last_national_election_with_clandestine_polls(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path))
    write_ban_bundle(fs)
    latest = compute_ban(bundle.BundleReader(fs), DATA).latest
    assert (latest['scope'], latest['event_date'], latest['label']) == ('es', '2023-07-23', '2023')
    assert latest['parties'] == ['PP', 'PSOE', 'VOX', 'SUMAR'] and len(latest['polls']) == 3
    gad3 = next(poll for poll in latest['polls'] if poll['pollster'] == 'GAD3')
    assert gad3 == {'pollster_id': 2, 'pollster': 'GAD3', 'sponsor': 'Medio de GAD3', 'date': '2023-07-22', 'days': 1,
                    'sample_size': 10004, 'mtype': 'cati', 'error_avg': 2.1,
                    'pct': {'PP': 35.5, 'PSOE': 28.6, 'VOX': 12.7, 'SUMAR': 11.9}}
    assert latest['consensus'] == pytest.approx({'PP': 33.6, 'PSOE': 29.0, 'VOX': 13.25, 'SUMAR': 12.85})
    assert latest['frozen'] == {'PP': 34.2, 'PSOE': 28.3, 'VOX': 13.1, 'SUMAR': 13.1}
    assert latest['official'] == {'PP': 33.1, 'PSOE': 31.7, 'VOX': 12.4, 'SUMAR': 12.3}
    error = latest['error']
    assert error['legal'] == pytest.approx(2.03, abs=0.005) and error['frozen'] == pytest.approx(1.5)
    assert error['consensus'] == pytest.approx(1.15) and error['with'] is None
