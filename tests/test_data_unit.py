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
