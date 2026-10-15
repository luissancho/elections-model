"""Tests de `mtpy.lib.data` que no necesitan base de datos."""
import pandas as pd

from mtpy.lib.data import get_event_dmat
from mtpy.lib import data


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


def test_event_dmat_of_an_upcoming_event_without_polls_is_empty():
    # Una comunidad que acaba de votar: su próximo evento no tiene sondeos ni resultado
    parties = pd.DataFrame({'name': ['PSOE', 'PP']})
    index = pd.MultiIndex.from_arrays([[], [], []], names=['event_date', 'days', 'pollster'])
    polls = pd.DataFrame(columns=['PSOE', 'PP'], index=index, dtype=float)
    events = pd.DataFrame({'PSOE': [float('nan')], 'PP': [float('nan')]}, index=pd.Index(['2030-06-16'], name='date'))
    df = get_event_dmat('es-an', '2030-06-16', polls, events, parties)
    assert df.shape[1] == 0 and (0, 'result') in df.index


class FakeTable:
    """Modelo con `get_results(query, formatted)` que devuelve `frame` y anota las consultas."""

    frame = None
    calls = []

    def get_results(self, query=None, formatted=False):
        type(self).calls.append(dict(query))
        return type(self).frame.copy()


def test_get_pollster_events_counts_the_rated_polls_of_each_election(monkeypatch):
    class Polls(FakeTable):
        calls = []
        frame = pd.DataFrame({
            # `event_scope` es categórica en la tabla formateada, con ámbitos sin sondeos: no deben aparecer
            'event_scope': pd.Categorical(['es', 'es', 'es-md'], categories=['es', 'es-an', 'es-md']),
            'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2021-05-04']),
            'pollster_id': [2, 2, 2], 'date': pd.to_datetime(['2023-07-10', '2023-07-16', '2021-04-20']),
            'error_avg': [1.0, 1.6, 2.1], 'error_blocks': [1.2, 1.8, -0.8], 'bias_dev_adj': [-0.5, -1.5, 2.0],
        })
    monkeypatch.setattr(data, 'Polls', Polls)
    events = data.get_pollster_events(2)
    assert len(events) == 2
    assert list(events.columns) == ['event_scope', 'event_date', 'n_polls', 'error_avg', 'error_blocks', 'bias_dev_adj']
    assert events['event_scope'].tolist() == ['es-md', 'es'] and events['n_polls'].tolist() == [1, 2]
    # medias simples: el margen con signo, en puntos; la desviación frente a los comparables, con el signo del modelo
    assert events['error_avg'].tolist() == [2.1, 1.3] and events['error_blocks'].tolist() == [-0.8, 1.5]
    assert events['bias_dev_adj'].tolist() == [2.0, -1.0]
    assert events['event_date'].dt.strftime('%Y-%m-%d').tolist() == ['2021-05-04', '2023-07-23']
    assert Polls.calls[0]['filters'] == ['pollster_id = 2', 'featured', 'computed']  # los sondeos que entran en el rating


def test_get_pollster_ratings_reads_the_rows_of_the_pollster_in_every_scope(monkeypatch):
    class Ratings(FakeTable):
        calls = []
        frame = pd.DataFrame({
            'event_scope': ['es-md', 'es'], 'event_date': pd.to_datetime(['2021-05-04', '2023-07-23']),
            'pollster_id': [2, 2], 'rating': [71.0, 80.0],
        })
    monkeypatch.setattr(data, 'PollstersRatings', Ratings)
    ratings = data.get_pollster_ratings(2)
    assert ratings['rating'].tolist() == [71.0, 80.0] and Ratings.calls[0]['filters'] == ['pollster_id = 2']
