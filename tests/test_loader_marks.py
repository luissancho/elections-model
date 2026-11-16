"""Tests de las marcas de carga de los cargadores (`mtpy.lib.loader`): instancias desnudas, sin base ni red."""
import pandas as pd

from mtpy.lib import loader as module
from mtpy.lib.loader import InfoElectoralLoader, WikipediaLoader, WikipediaResultsLoader


class FakeTable:
    """Modelo de tabla sin base: acepta la secuencia de guardado y anota los upserts."""
    table = 'elections.t'

    def __init__(self, current=None):
        self.current, self.upserted, self.executed = current, [], []

    def execute(self, query):
        self.executed.append(query)

    def stage_write(self, df, compression=None):
        pass

    def upsert(self, df=None):
        self.upserted.append(df)
        return 3

    def stage_clean(self):
        pass

    def vacuum(self):
        pass

    def get_results(self, query=None, formatted=False):
        return self.current if self.current is not None else pd.DataFrame()

    def format_data(self, df, **kwargs):
        return df


def bare(cls, scope='es', event_date='2026-11-29'):
    obj = cls.__new__(cls)
    obj.scope, obj.event_date, obj.verbose = scope, event_date, 0
    obj.filters = ["scope = '{}'".format(scope), "date = '{}'".format(event_date)]
    return obj


def marks(monkeypatch):
    seen = []
    monkeypatch.setattr(module, 'mark_loaded', lambda scope, event_date: seen.append((scope, event_date)))
    return seen


def test_wikipedia_loader_marks_the_event_after_saving_polls(monkeypatch):
    seen = marks(monkeypatch)
    wiki = bare(WikipediaLoader)
    wiki.m_polls = FakeTable()
    wiki.polls = pd.DataFrame({'x': [1, 2]})
    wiki.save_polls(overwrite=False)
    assert seen == [('es', '2026-11-29')] and len(wiki.m_polls.upserted) == 1


def test_wikipedia_loader_without_polls_saves_nothing_and_marks_nothing(monkeypatch):
    seen = marks(monkeypatch)
    wiki = bare(WikipediaLoader)
    wiki.m_polls = FakeTable()
    wiki.polls = pd.DataFrame()
    wiki.save_polls(overwrite=True)
    assert seen == [] and wiki.m_polls.upserted == []


def test_results_loaders_mark_the_event_only_when_they_write_results(monkeypatch):
    seen = marks(monkeypatch)
    wp = bare(WikipediaResultsLoader, 'es-md', '2023-05-28')
    wp.m_results = FakeTable()
    wp.results = pd.DataFrame({'party_id': [1]})
    wp.save_results()
    wp.results = pd.DataFrame()
    wp.save_results()
    ie = bare(InfoElectoralLoader, 'es', '2023-07-23')
    ie.m_results = FakeTable()
    ie.results = pd.DataFrame({'party_id': [1]})
    ie.save_results()
    ie.results = pd.DataFrame()
    ie.save_results()
    assert seen == [('es-md', '2023-05-28'), ('es', '2023-07-23')]


def test_save_event_carries_over_featured_and_the_marks_of_the_current_row():
    wp = bare(WikipediaResultsLoader, 'es', '2023-07-23')
    wp.event = pd.DataFrame([{'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'Elecciones España Julio 2023',
                              'featured': False}])
    wp.m_events = FakeTable(current=pd.DataFrame([{
        'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'viejo', 'featured': True,
        'loaded_at': pd.Timestamp('2026-11-01 09:00:00'), 'computed_at': pd.Timestamp('2026-11-02 09:00:00')}]))
    wp.save_event()
    saved = wp.m_events.upserted[0].iloc[0]
    assert saved['name'] == 'Elecciones España Julio 2023' and bool(saved['featured']) is True
    assert saved['loaded_at'] == pd.Timestamp('2026-11-01 09:00:00') and saved['computed_at'] == pd.Timestamp('2026-11-02 09:00:00')


def test_save_event_of_a_new_row_carries_no_marks():
    wp = bare(WikipediaResultsLoader, 'es', '2030-01-01')
    wp.event = pd.DataFrame([{'date': pd.Timestamp('2030-01-01'), 'scope': 'es', 'name': 'Próximas', 'featured': False}])
    wp.m_events = FakeTable(current=pd.DataFrame())
    wp.save_event()
    saved = wp.m_events.upserted[0]
    assert 'loaded_at' not in saved.columns and 'computed_at' not in saved.columns


def test_select_series_keeps_the_read_and_current_frames():
    from mtpy.models.elections import Polls, PollsResults

    def polls(days):
        return pd.DataFrame({'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es',
                             'date': pd.to_datetime(days), 'pollster_id': 15, 'sponsor_id': 1})

    def results(days):
        return polls(days).assign(party_id=1, pct=30.0)

    wiki = bare(WikipediaLoader)
    wiki.m_polls, wiki.m_results = FakeTable(current=polls(['2026-11-07'])), FakeTable(current=results(['2026-11-07']))
    wiki.m_polls.key, wiki.m_results.key = Polls.key, PollsResults.key
    wiki.polls, wiki.results = polls(['2026-11-07', '2026-11-14']), results(['2026-11-07', '2026-11-14'])
    asked = []
    current = wiki.m_results.get_results
    wiki.m_results.get_results = lambda **kwargs: asked.append(kwargs) or current(**kwargs)
    wiki.cur_results = None
    wiki.select_series(overwrite=False)
    assert list(wiki.polls['date']) == [pd.Timestamp('2026-11-14')] and len(wiki.results) == 1
    assert len(wiki.read_polls) == 2 and len(wiki.read_results) == 2
    assert list(wiki.cur_polls['date']) == [pd.Timestamp('2026-11-07')] and 'pollster_id' in wiki.cur_polls.columns
    # Los resultados de la base sólo sirven a la comparación: la selección no los pide
    assert asked == [] and wiki.cur_results is None
    assert list(wiki.fetch_cur_results()['pct']) == [30.0] and len(asked) == 1
    assert list(wiki.cur_results['pct']) == [30.0]


def keyed_polls(rows):
    """Sondeos `(día, pollster_id, sponsor_id)` con las columnas de la clave."""
    return pd.DataFrame([{'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es', 'date': pd.Timestamp(day),
                          'pollster_id': pollster_id, 'sponsor_id': sponsor_id}
                         for day, pollster_id, sponsor_id in rows])


def test_select_changed_adds_the_read_rows_once():
    from mtpy.models.elections import Polls, PollsResults
    wiki = bare(WikipediaLoader)
    wiki.m_polls, wiki.m_results = FakeTable(), FakeTable()
    wiki.m_polls.key, wiki.m_results.key = Polls.key, PollsResults.key
    rows = [('2026-11-07', 15, 1), ('2026-11-13', 3, 0), ('2026-11-14', 15, 1)]
    wiki.read_polls = keyed_polls(rows)
    wiki.read_results = pd.concat([keyed_polls(rows).assign(party_id=1, pct=30.0),
                                   keyed_polls(rows).assign(party_id=2, pct=20.0)], ignore_index=True)
    # El del 14 es nuevo y ya está en lo que se guarda; el del 13 (sin medio, id 0) ha cambiado
    wiki.polls = wiki.read_polls.iloc[[2]]
    wiki.results = wiki.read_results[wiki.read_results.date == pd.Timestamp('2026-11-14')]
    assert wiki.select_changed([('2026-11-13', 3, 0), ('2026-11-14', 15, 1)]) is wiki
    assert sorted(wiki.polls['date'].dt.strftime('%Y-%m-%d')) == ['2026-11-13', '2026-11-14']
    assert len(wiki.results) == 4 and sorted(wiki.results['date'].dt.day.unique()) == [13, 14]
    wiki.select_changed([])
    assert len(wiki.polls) == 2 and len(wiki.results) == 4


def test_select_changed_into_an_empty_selection():
    from mtpy.models.elections import Polls, PollsResults
    wiki = bare(WikipediaLoader)
    wiki.m_polls, wiki.m_results = FakeTable(), FakeTable()
    wiki.m_polls.key, wiki.m_results.key = Polls.key, PollsResults.key
    wiki.read_polls = keyed_polls([('2026-11-07', 15, 1), ('2026-11-13', 3, 0)])
    wiki.read_results = wiki.read_polls.assign(party_id=1, pct=30.0)
    wiki.polls, wiki.results = wiki.read_polls.iloc[:0], wiki.read_results.iloc[:0]
    wiki.select_changed([('2026-11-07', 15, 1)])
    assert list(wiki.polls['pollster_id']) == [15] and list(wiki.results['pct']) == [30.0]


def test_delete_polls_builds_one_filtered_delete_per_table():
    wiki = bare(WikipediaLoader)
    wiki.filters = ["event_scope = 'es'", "event_date = '2026-11-29'"]
    wiki.m_polls, wiki.m_results = FakeTable(), FakeTable()
    wiki.m_polls.table, wiki.m_results.table = 'elections.polls', 'elections.polls_results'
    assert wiki.delete_polls([]) is wiki and wiki.m_polls.executed == [] and wiki.m_results.executed == []
    wiki.delete_polls([('2026-11-14', 15, 1), ('2026-11-13', 3, None)])
    where = ("event_scope = 'es' AND event_date = '2026-11-29' AND ("
             "(date = '2026-11-14' AND pollster_id = 15 AND sponsor_id = 1) OR "
             "(date = '2026-11-13' AND pollster_id = 3 AND sponsor_id IS NULL))")
    assert wiki.m_results.executed == ['DELETE FROM elections.polls_results WHERE ' + where]
    assert wiki.m_polls.executed == ['DELETE FROM elections.polls WHERE ' + where]
