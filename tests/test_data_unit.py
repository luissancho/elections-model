"""Tests de `mtpy.lib.data` que no necesitan base de datos."""
import pandas as pd

from mtpy.lib.data import get_event_dmat


def test_event_dmat_without_polls_keeps_the_result():
    # Foco 1: una elección sin sondeos (Asturias 2019) sigue teniendo partidos
    parties = pd.DataFrame({'name': ['PSOE', 'PP', 'FAC']})
    index = pd.MultiIndex.from_arrays([[], [], []], names=['event_date', 'days', 'pollster'])
    polls = pd.DataFrame(columns=['PSOE', 'PP', 'FAC'], index=index, dtype=float)
    events = pd.DataFrame({'PSOE': [35.3], 'PP': [17.5], 'FAC': [6.5]}, index=pd.Index(['2019-05-26'], name='date'))
    df = get_event_dmat('es-as', '2019-05-26', polls, events, parties)
    assert df.loc[(0, 'result')].dropna().index.tolist() == ['PSOE', 'PP', 'FAC']


def test_resolve_thresholds_event_override_wins_even_when_null():
    from mtpy.lib.data import resolve_thresholds
    nan = float('nan')
    assert resolve_thresholds({'threshold': 3.0, 'threshold_scope': nan}) == (3.0, None)
    assert resolve_thresholds({'threshold': 5.0, 'threshold_scope': nan}, {'smap': {}}) == (5.0, None)
    assert resolve_thresholds({'threshold': 3.0, 'threshold_scope': nan}, {'threshold': None, 'threshold_scope': 5.0}) == (None, 5.0)
    assert resolve_thresholds({'threshold': 15.0, 'threshold_scope': 4.0}, {'threshold': 30.0, 'threshold_scope': 6.0}) == (30.0, 6.0)
