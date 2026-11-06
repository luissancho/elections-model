"""Tests de `mtpy/lib/studies/ban/stats.py`: pares, prueba del azar, consenso clandestino, celdas, elecciones, casas
y grandes, con las cifras calculadas a mano de `write_ban_bundle`."""
import os

import numpy as np
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.ban import data, stats
from tests.fakes import write_ban_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARAMS = data.load_params(FileSystem(os.path.join(ROOT, 'data')))


@pytest.fixture
def ban(fresh_app, tmp_path):
    """The events of `write_ban_bundle` by ``(scope, date)`` and the hand figures."""
    fs = FileSystem(str(tmp_path))
    figures = write_ban_bundle(fs)
    names = FileSystem(os.path.join(ROOT, 'data')).read_csv('es-scopes.csv').set_index('scode')['name'].to_dict()
    events, _ = data.read_events(bundle.BundleReader(fs), PARAMS, names)
    return {(event.scope, event.event_date): event for event in events}, figures


def _pairs(events, params=PARAMS):
    return [pair for event in events.values() for pair in stats.pair_rows(event, params)]


def test_pair_rows_pair_each_house_with_its_last_legal_within_the_window(ban):
    events, figures = ban
    pairs = _pairs(events)
    assert {(p['scope'], p['event_date'], p['pollster']): round(p['d'], 2) for p in pairs} == figures['pairs']
    gad3 = next(p for p in pairs if p['event_date'] == '2019-11-10' and p['pollster'] == 'GAD3')
    assert gad3['date_ban'] == '2019-11-10' and gad3['days_ban'] == 0 and gad3['error_ban'] == 0.80
    assert gad3['date_legal'] == '2019-10-31' and gad3['days_legal'] == 10 and gad3['error_legal'] == 0.77
    assert gad3['pollster_id'] == 2 and gad3['sponsor'] == 'Medio de GAD3' and gad3['sample_size'] == 13000
    assert gad3['big'] and not gad3['closer'] and gad3['national'] and gad3['label'] == '2019' and 'slug' not in gad3
    closer = {(p['event_date'], p['pollster']) for p in pairs if p['closer']}
    assert closer == {('2023-07-23', 'GAD3'), ('2021-05-04', 'Sigma Dos')}
    assert {(p['event_date'], p['pollster']) for p in pairs if p['big']} == {('2019-11-10', 'GAD3'),
                                                                             ('2023-07-23', 'GAD3')}


def test_pairs_skip_a_legal_poll_older_than_the_window(ban):
    events, _ = ban
    madrid = stats.pair_rows(events['es-md', '2021-05-04'], PARAMS)
    assert [p['pollster'] for p in madrid] == ['Sigma Dos']  # el último legal de GAD3, a 60 días
    wide = stats.pair_rows(events['es-md', '2021-05-04'], PARAMS | {'max_days': 60})
    assert [p['pollster'] for p in wide] == ['GAD3', 'Sigma Dos'] and round(wide[0]['d'], 2) == -0.75


def test_pairs_leave_aggregators_out_but_keep_panels(ban):
    events, _ = ban
    assert [p['pollster'] for p in stats.pair_rows(events['es', '2023-07-23'], PARAMS)] == ['40dB', 'GAD3']
    panel = [p for p in stats.pair_rows(events['es', '2019-11-10'], PARAMS) if p['pollster'] == 'ElectoPanel']
    assert panel[0]['mtype'] == 'online' and not panel[0]['big']


def test_pair_rows_take_the_last_row_when_a_house_publishes_twice_the_same_day(ban):
    events, _ = ban
    event = events['es', '2019-11-10']
    twice = event.polls.iloc[[1]].assign(error_avg=0.5)
    event.polls = pd.concat([event.polls, twice], ignore_index=True)
    gad3 = next(p for p in stats.pair_rows(event, PARAMS) if p['pollster'] == 'GAD3')
    assert gad3['error_ban'] == 0.5 and gad3['closer']


def test_sign_test_is_the_binomial_of_the_wins():
    assert stats.sign_test(2, 5) == pytest.approx(0.8125)
    assert stats.sign_test(5, 5) == pytest.approx(1 / 32) and stats.sign_test(0, 0) is None


def test_summary_counts_a_tie_in_n_and_ties_but_not_in_k_and_tests_without_it():
    pairs = [{'scope': 'es', 'event_date': '2023-07-23', 'national': True, 'pollster_id': index, 'd': d,
              'closer': d < 0, 'sample_size': 1000, 'label': '2023'}
             for index, d in enumerate([-0.1, 0.0, 0.2])]
    block = stats.summarize([], pairs, [], [], PARAMS)['pairs']
    assert (block['n'], block['k'], block['ties']) == (3, 1, 1) and block['share'] == pytest.approx(1 / 3)
    assert block['p'] == pytest.approx(0.75) and block['p'] == pytest.approx(stats.sign_test(1, 2))
    assert block['mean_d'] == pytest.approx(0.1 / 3)



def test_the_exception_is_never_the_national_scope():
    """With the generals the worst scope (and enough pairs), the exception is the worst community, not «España»."""
    def scope_pairs(scope, national, wins, n):
        return [{'scope': scope, 'event_date': '2023-07-23', 'national': national, 'pollster_id': index,
                 'd': -0.1 if index < wins else 0.1, 'closer': index < wins, 'sample_size': 1000, 'label': '2023'}
                for index in range(n)]

    import types

    events = [types.SimpleNamespace(scope=scope, national=scope == 'es') for scope in ('es', 'es-md', 'es-an')]
    pairs = scope_pairs('es', True, 2, 12) + scope_pairs('es-md', False, 6, 12) + scope_pairs('es-an', False, 8, 12)
    summary = stats.summarize(events, pairs, [], [], PARAMS)
    assert [row['scope'] for row in summary['pairs']['by_scope']] == ['es', 'es-md', 'es-an']
    assert summary['exception'] == {'scope': 'es-md', 'n': 12, 'k': 6}
    only_national = stats.summarize(events[:1], scope_pairs('es', True, 2, 12), [], [], PARAMS)
    assert only_national['exception'] is None

def test_consensus_leaves_out_the_panel_and_the_aggregator(ban):
    events, figures = ban
    for date in ('2019-11-10', '2023-07-23'):
        consensus = stats.consensus(events['es', date], PARAMS)
        assert consensus == pytest.approx(figures['consensus'][date])
    assert stats.consensus(events['es', '2019-11-10'], PARAMS | {'exclude_mtypes_consensus': ['aggr']}) == \
        pytest.approx({'PSOE': 27.2, 'PP': 20.25, 'VOX': 15.75})


def test_consensus_averages_only_the_polls_with_the_party(ban):
    events, _ = ban
    event = events['es', '2023-07-23']
    event.polls.loc[event.polls['pollster'] == 'GAD3', 'VOX'] = np.nan
    consensus = stats.consensus(event, PARAMS)
    assert consensus['VOX'] == pytest.approx(13.8) and consensus['PP'] == pytest.approx(33.6)
    event.polls = event.polls[event.polls['mtype'] == 'aggr']
    assert stats.consensus(event, PARAMS) is None


def test_cell_rows_flag_closer_direction_crossed_and_half(ban):
    events, figures = ban
    cells = []
    for date in ('2019-11-10', '2023-07-23'):
        event = events['es', date]
        cells += stats.cell_rows(event, stats.consensus(event, PARAMS), PARAMS)
    flags = {(c['event_date'], c['party']): (c['closer'], c['direction'], c['crossed'], c['half']) for c in cells}
    assert flags == figures['cells']
    assert [c['party'] for c in cells] == ['PSOE', 'PP', 'VOX', 'PP', 'PSOE', 'VOX', 'SUMAR']
    vox = cells[2]
    assert (vox['frozen'], vox['consensus'], vox['official'], vox['label']) == (13.9, 16.0, 15.1, '2019')
    assert vox['move'] == pytest.approx(2.1) and vox['gap'] == pytest.approx(1.2)
    madrid = events['es-md', '2021-05-04']
    assert stats.cell_rows(madrid, stats.consensus(madrid, PARAMS), PARAMS) == []
    late = PARAMS | {'chart_since': '2020-01-01'}
    assert stats.cell_rows(events['es', '2019-11-10'], figures['consensus']['2019-11-10'], late) == []


def test_cell_rows_do_not_cross_on_an_exact_hit(ban):
    events, _ = ban
    event = events['es', '2019-11-10']
    cells = stats.cell_rows(event, {'PSOE': 28.0, 'PP': 20.8, 'VOX': 15.1}, PARAMS)
    assert not any(c['crossed'] for c in cells) and all(c['closer'] and c['half'] for c in cells)


def test_election_row_with_and_without_model(ban):
    events, figures = ban
    for date in ('2019-11-10', '2023-07-23'):
        event = events['es', date]
        pairs = stats.pair_rows(event, PARAMS)
        row = stats.election_row(event, pairs, stats.consensus(event, PARAMS), PARAMS)
        expected = figures['errors'][date]
        model = row['model']
        assert model['error_legal'] == pytest.approx(expected['legal'], abs=0.005)
        assert model['error_frozen'] == pytest.approx(expected['frozen'], abs=0.005)
        assert model['error_consensus'] == pytest.approx(expected['consensus'], abs=0.005)
        if expected['with'] is None:
            assert model['error_with'] is None and model['beaten_with'] is None
        else:
            assert model['error_with'] == pytest.approx(expected['with'], abs=0.005)
    first = stats.election_row(events['es', '2019-11-10'], stats.pair_rows(events['es', '2019-11-10'], PARAMS),
                               figures['consensus']['2019-11-10'], PARAMS)
    assert first['model']['error_legal'] == pytest.approx(0.94) and first['model']['beaten'] is False
    assert first['model']['beaten_with'] is True
    assert (first['n_ban'], first['n_pairs'], first['k_closer']) == (3, 2, 0)
    assert first['mean_d'] == pytest.approx(0.045) and first['label'] == '2019' and first['national']
    madrid = events['es-md', '2021-05-04']
    row = stats.election_row(madrid, stats.pair_rows(madrid, PARAMS), stats.consensus(madrid, PARAMS), PARAMS)
    assert row['model'] is None and (row['n_ban'], row['n_pairs'], row['k_closer']) == (2, 1, 1)
    assert row['mean_d'] == pytest.approx(-0.3)
    assert stats.election_row(madrid, [], None, PARAMS)['mean_d'] is None


def test_election_row_error_legal_needs_the_pollsters_part(ban):
    events, _ = ban
    event = events['es', '2023-07-23']
    event.pollsters = None
    row = stats.election_row(event, [], stats.consensus(event, PARAMS), PARAMS)
    assert row['model']['error_legal'] is None and row['model']['beaten'] is True


def test_house_rows_need_min_pairs(ban):
    events, _ = ban
    pairs = _pairs(events)
    assert stats.house_rows(pairs, PARAMS) == []
    rows = stats.house_rows(pairs, PARAMS | {'min_pairs': 2})
    assert len(rows) == 1 and rows[0]['pollster'] == 'GAD3' and rows[0]['pollster_id'] == 2
    assert (rows[0]['n'], rows[0]['k'], rows[0]['share']) == (2, 1, 0.5)
    assert rows[0]['mean_d'] == pytest.approx(-0.3) and 'slug' not in rows[0]
    everyone = stats.house_rows(pairs, PARAMS | {'min_pairs': 1})
    assert [row['pollster'] for row in everyone] == ['Sigma Dos', 'GAD3', 'ElectoPanel', '40dB']


def test_big_rows_take_the_pairs_over_the_threshold(ban):
    events, _ = ban
    rows = stats.big_rows(_pairs(events), PARAMS)
    assert [(row['label'], row['pollster'], row['sample_size']) for row in rows] == [('2019', 'GAD3', 13000),
                                                                                    ('2023', 'GAD3', 10004)]
    assert rows[1]['d'] == pytest.approx(-0.63) and rows[1]['error_legal'] == 2.73 and rows[1]['error_ban'] == 2.10
    assert rows[1]['sponsor'] == 'Medio de GAD3' and rows[1]['pollster_id'] == 2
    assert stats.big_rows(_pairs(events), PARAMS | {'big_sample': 20000}) == []
