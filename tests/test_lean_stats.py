"""Tests de `mtpy/lib/studies/lean/stats.py`: pruebas de azar, t de Student, ranking, matriz y tabla de una elección."""
import os

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, webapi
from mtpy.lib.pages import metodo
from mtpy.lib.studies.lean import data, stats
from tests.fakes import write_lean_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def lean_rows(tmp_path):
    """Filas de `read_error_rows` sobre el paquete sintético y las cifras que calcula `write_lean_bundle`."""
    fs = FileSystem(str(tmp_path))
    expected = write_lean_bundle(fs)
    rows, _ = data.read_error_rows(bundle.BundleReader(fs), webapi.checks.catalogue())

    return rows, expected


@pytest.fixture(autouse=True)
def repo_data(fresh_app):
    """Catálogo de ámbitos del repositorio, leído como lo lee la web."""
    fresh_app.data = FileSystem(os.path.join(ROOT, 'data'))


def test_sign_chance_matches_the_method_card():
    assert stats.sign_chance(41, 38) == metodo.sign_chance(41, 38)
    assert stats.sign_chance(41, 38)['ways'] == 11522 and stats.sign_chance(41, 38)['odds'] == 190000000
    assert stats.sign_chance(0, 0) is None


def test_t_test_known_values():
    out = stats.t_test([2.0, 4.0, 6.0, 8.0])
    assert (out['n'], out['mean'], out['sd']) == (4, 5.0, pytest.approx(2.582, abs=1e-2))
    assert out['t'] == pytest.approx(3.873, abs=1e-2)
    assert 0.02 < out['p'] < 0.04 and stats.t_test([1.0]) is None and stats.t_test([2.0, 2.0]) is None


def test_summarize_period_counts_the_synthetic_cis(tmp_path):
    rows, expected = lean_rows(tmp_path)
    with_rows = [r for r in rows if r['event_date'] >= '2018-09-01']
    out = stats.summarize_period(with_rows, 'CIS')
    assert (out['n'], out['k_left'], out['n_right'], out['k_right']) == (6, 5, 6, 6)
    assert out['mean_left'] == pytest.approx(4.13, abs=0.01) and out['mean_lean'] == pytest.approx(8.28, abs=0.01)
    assert out['exceptions'] == ['Andalucía 2026'] and out['dates'] == {'n': 6, 'k': 5}
    assert out['consensus']['n'] == 6 and out['consensus']['k'] == 5 and out['chance']['coin'] == stats.sign_chance(6, 5)
    assert out['industry'] == expected['industry']
    assert set(out['t']) == {'all', 'dates', 'consensus'} and out['t']['all']['n'] == 6
    assert len(out['rows']) == 6 and set(out['rows'][0]) == {'scope', 'event_date', 'consensus', 'n_others',
                                                             'others_left'}


def test_summarize_period_skips_rows_without_the_right_block(tmp_path):
    rows, _ = lean_rows(tmp_path)
    rows = [dict(r, Derecha=None, lean=None) if r['event_date'] == '2019-04-28' else r for r in rows]
    out = stats.summarize_period([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS')
    assert (out['n'], out['n_right'], out['n_lean']) == (6, 5, 5)


def test_consensus_and_industry_need_other_houses(tmp_path):
    rows, _ = lean_rows(tmp_path)
    alone = [r for r in rows if r['event_date'] >= '2018-09-01' and not (r['scope'] == 'es-an' and r['pollster'] != 'CIS')]
    out = stats.summarize_period(alone, 'CIS')
    assert out['n'] == 6 and out['consensus']['n'] == 3 and out['industry']['n_events'] == 1


def test_house_table_filters_and_orders(tmp_path):
    rows, _ = lean_rows(tmp_path)
    houses = stats.house_table([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS', 3,
                               {'CIS', 'GAD3', 'Sigma Dos', 'NC Report'})
    assert [h['pollster'] for h in houses] == ['GAD3', 'Sigma Dos', 'CIS'] and houses[-1]['cis'] is True
    assert 'slug' not in houses[0] and houses[0]['share_left'] == 0 and houses[1]['share_left'] == 1.0


def test_matrix_and_election_tables(tmp_path):
    rows, expected = lean_rows(tmp_path)
    houses = stats.house_table([r for r in rows if r['event_date'] >= '2018-09-01'], 'CIS', 3,
                               {'CIS', 'GAD3', 'Sigma Dos'})
    matrix = stats.matrix_table([r for r in rows if r['event_date'] >= '2018-09-01'], houses)
    assert matrix['houses'] == ['CIS', 'Sigma Dos', 'GAD3'] and matrix['events'][0]['short'] == 'AN 18'
    assert [c for c in matrix['cells'] if c[1] == 0 and c[0] == 2] == [[2, 0, 6.4]]
    election = stats.election_table(expected['election'], 'CIS', 60, ['Izquierda', 'Derecha'])
    assert [h['pollster'] for h in election['houses']][0] != 'Ipsos' and all(h['days'] <= 60 for h in election['houses'])
    assert all('slug' not in h for h in election['houses'])
    assert election['result'] == {'Izquierda': 44.01, 'Derecha': 45.44, 'gap': -1.43}
    assert election['cis']['e_left'] == 3.1 and election['consensus']['n'] == 4
    assert election['cis']['dist'] == pytest.approx(4.18, abs=0.01)
