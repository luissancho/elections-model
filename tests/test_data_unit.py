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


def test_get_national_events_filters_scope_bias_and_ctype(monkeypatch):
    class Polls(FakeTable):
        calls = []
        frame = pd.DataFrame({
            'event_scope': pd.Categorical(['es', 'es', 'es', 'es', 'es', 'es', 'es-md', 'es'],
                                          categories=['es', 'es-md']),
            'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2023-07-23', '2023-07-23', '2023-07-23',
                                          '2019-11-10', '2021-05-04', '2019-04-28']),
            'pollster_id': [1, 1, 2, 3, 4, 5, 6, 7],
            'bias': [0.1, 0.2, 0.3, 0.4, 0.5, None, 0.6, 0.7],
            'ctype': [None, 'poll', 'wban', 'exit', None, None, None, 'poll'],
            'mtype': ['phone', 'phone', 'phone', 'phone', 'online', 'phone', 'phone', 'phone'],
        })
    monkeypatch.setattr(data, 'Polls', Polls)
    events = data.get_national_events()
    assert list(events.columns) == ['pollster_id', 'event_date']
    assert events['pollster_id'].tolist() == [7, 1, 4]  # not 2 (wban), 3 (exit), 5 (no bias), 6 (es-md); 1 once
    assert events['event_date'].dt.strftime('%Y-%m-%d').tolist() == ['2019-04-28', '2023-07-23', '2023-07-23']
    filters = ' '.join(Polls.calls[0]['filters'])
    assert "event_scope = 'es'" in filters and 'bias IS NOT NULL' in filters and 'wban' in filters


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


def test_get_pollsters_quality_without_table_is_empty(monkeypatch):
    class NoTable(FakeTable):
        meta = {'pollster_id': 1, 'pollster': 1, 'points': 1, 'quality': 1}

        def table_exists(self):
            return False
    monkeypatch.setattr(data, 'PollstersQuality', NoTable)
    frame = data.get_pollsters_quality()
    assert frame.empty and list(frame.columns) == ['pollster_id', 'pollster', 'points', 'quality']


def test_save_catalogues_replicates_the_csv_and_writes_every_pollster(monkeypatch):
    """La réplica borra las casas retiradas del CSV, hace upsert de las evaluadas y escribe `pollsters.quality`
    de todas (el mínimo para las no evaluadas)."""
    calls = []

    class Table:
        def __init__(self, name):
            self.name = name
            self.table = 'elections.pollsters_quality' if name == 'quality' else 'elections.' + name

        def table_exists(self):
            return False

        def create(self, replace=True):
            calls.append((self.name, 'create', replace))

        def format_data(self, df, **kwargs):
            return df

        def execute(self, sql):
            calls.append((self.name, 'execute', sql))

        def upsert(self, df):
            calls.append((self.name, 'upsert', df.copy()))
            return len(df)

    evaluated = pd.DataFrame({'pollster_id': [1], 'pollster': ['CIS'], 'points': [7], 'quality': [56.0]})
    monkeypatch.setattr(data, 'Scopes', lambda: Table('scopes'))
    monkeypatch.setattr(data, 'Districts', lambda: Table('districts'))
    monkeypatch.setattr(data, 'PollstersQuality', lambda: Table('quality'))
    monkeypatch.setattr(data, 'Pollsters', lambda: Table('pollsters'))
    monkeypatch.setattr(data, 'get_scopes', lambda: pd.DataFrame({'name': ['España']}, index=pd.Index(['es'], name='scode')))
    monkeypatch.setattr(data, 'get_districts', lambda: pd.DataFrame({'scope': ['es'], 'region_id': [0]}))
    evaluated = evaluated.assign(fieldwork=3, transparency=3, independence=0, identity=1, standards=0)
    monkeypatch.setattr(data, 'read_quality', lambda criteria=None: evaluated)
    monkeypatch.setattr(data, 'get_national_events', lambda: pd.DataFrame({
        'pollster_id': [1], 'event_date': pd.to_datetime(['2023-07-23'])}))
    monkeypatch.setattr(data, 'load_criteria', lambda: {'scale': {'per_point': 8, 'floor': 5}, 'criteria': {}})
    monkeypatch.setattr(data, 'get_pollsters', lambda: pd.DataFrame({'id': [1, 2], 'name': ['CIS', 'GAD3'], 'quality': [10.0, 76.0], 'rating': [16.0, 73.0]}))
    out = data.save_catalogues()
    assert out == {'scopes': 1, 'districts': 1, 'quality': 1, 'pollsters': 2}
    sql = [c[2] for c in calls if c[0] == 'quality' and c[1] == 'execute']
    assert len(sql) == 1 and 'DELETE FROM elections.pollsters_quality' in sql[0] and 'NOT IN (1)' in sql[0]
    written = [c[2] for c in calls if c[0] == 'pollsters' and c[1] == 'upsert'][0]
    # 6 manual + 1 implantation (a general before today)
    assert written['quality'].tolist() == [56.0, 5.0] and written['rating'].tolist() == [16.0, 73.0]


def test_format_data_with_extra_columns_raises_no_setting_with_copy_warning():
    """`format_data` recorta el frame a las columnas del modelo; con columnas de más (la evaluación de la calidad
    trae `manual` o `implantation`, que la réplica no guarda) el recorte no debe dar `SettingWithCopyWarning` al
    formatear, y el frame del llamador no cambia."""
    import warnings

    from mtpy.core.utils.helpers import format_data

    meta = {'pollster_id': ['int', 3], 'pollster': ['str', None, False], 'points': ['int', 2]}
    frame = pd.DataFrame({'pollster_id': [1], 'pollster': ['A'], 'points': [3], 'manual': [2]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = format_data(frame, meta=meta)
    assert [w for w in caught if type(w.message).__name__ == 'SettingWithCopyWarning'] == []
    assert list(out.columns) == ['pollster_id', 'pollster', 'points'] and list(frame.columns)[-1] == 'manual'
