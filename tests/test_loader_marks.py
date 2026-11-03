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
