"""Tests de las marcas de carga y compute de `events` (`mtpy.lib.data`), con el modelo sustituido."""
from datetime import datetime

import pandas as pd
import pytest

from mtpy.lib import data


class FakeEvents:
    """`Events` sin base: anota las consultas ejecutadas y devuelve las filas y el reloj dados."""
    table = 'elections.events'

    def __init__(self, rows=None, now=datetime(2026, 11, 3, 10, 0, 0, 123456)):
        self.executed, self.queries, self.upserted = [], [], []
        self.rows, self.now = rows if rows is not None else [], now

    def execute(self, query):
        self.executed.append(query)

    def get_var(self, query=None):
        self.queries.append(query)
        return self.now

    def get_results(self, query=None, formatted=False):
        self.queries.append(query)
        return pd.DataFrame(self.rows, columns=['date'] if not self.rows else None)

    def format_data(self, df, **kwargs):
        return df

    def upsert(self, df):
        self.upserted.append(df.copy())
        return len(df)


@pytest.fixture
def events(monkeypatch):
    fake = FakeEvents()
    monkeypatch.setattr(data, 'Events', lambda: fake)
    return fake


def test_mark_loaded_stamps_the_event_with_the_database_clock(events):
    data.mark_loaded('es', '2026-11-29')
    assert events.executed == ["UPDATE elections.events SET loaded_at = now() WHERE scope = 'es' AND date = '2026-11-29'"]


def test_db_now_reads_the_clock_of_the_database(events):
    assert data.db_now() == datetime(2026, 11, 3, 10, 0, 0, 123456) and events.queries == ['SELECT now()']


def test_mark_computed_stamps_every_event_of_the_scope_with_the_start_instant(events):
    data.mark_computed('es-md', datetime(2026, 11, 3, 10, 0, 0, 123456))
    assert events.executed == ["UPDATE elections.events SET computed_at = '2026-11-03 10:00:00.123456' WHERE scope = 'es-md'"]


def test_pending_events_filters_in_the_database_and_returns_iso_dates(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2026-11-29')}, {'date': pd.Timestamp('2023-07-23')}])
    monkeypatch.setattr(data, 'Events', lambda: fake)
    assert data.pending_events('es') == ['2023-07-23', '2026-11-29']
    assert fake.queries == [{'columns': ['date'],
                             'filters': ["scope = 'es'", '(computed_at IS NULL OR loaded_at > computed_at)']}]


def test_pending_events_without_rows_is_empty(events):
    assert data.pending_events('es-xx') == []


def test_pending_events_treats_a_null_computed_at_as_pending():
    # la regla va en el filtro SQL: `computed_at IS NULL` entra aunque `loaded_at` sea nulo (tras el ALTER todo
    # está pendiente); la comprobación es sobre el texto del filtro, que es lo que ve la base
    assert '(computed_at IS NULL OR loaded_at > computed_at)' in data.PENDING_FILTER


def test_update_featured_keeps_the_marks_of_each_event(monkeypatch):
    fake = FakeEvents()
    monkeypatch.setattr(data, 'Events', lambda: fake)
    monkeypatch.setattr(data, 'get_events', lambda scope: pd.DataFrame({
        'date': pd.to_datetime(['2023-07-23', '2026-11-29']), 'scope': 'es', 'name': ['Julio 2023', 'Próximas'],
        'featured': [False, False], 'has_results': [True, False],
        'loaded_at': pd.to_datetime(['2026-11-01 09:00:00', '2026-11-03 09:00:00']),
        'computed_at': pd.to_datetime(['2026-11-02 09:00:00', None])}))

    class FakePolls:
        def get_results(self, query=None, formatted=False):
            return pd.DataFrame({'event_date': pd.to_datetime(['2023-07-23'] * 10), 'computed': True,
                                 'weight_over': 1.0, 'mtype': None, 'ctype': None})

    monkeypatch.setattr(data, 'Polls', FakePolls)
    assert data.update_featured('es') == 1
    saved = fake.upserted[0]
    assert list(saved['featured']) == [True, False] and 'has_results' not in saved.columns
    assert list(saved['loaded_at']) == list(pd.to_datetime(['2026-11-01 09:00:00', '2026-11-03 09:00:00']))
    assert saved['computed_at'].iloc[0] == pd.Timestamp('2026-11-02 09:00:00') and pd.isna(saved['computed_at'].iloc[1])


def test_result_dates_reads_the_whole_scope_rows_with_votes(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2023-07-23')}, {'date': pd.Timestamp('2019-11-10')},
                            {'date': pd.Timestamp('2023-07-23')}])
    monkeypatch.setattr(data, 'EventsData', lambda: fake)
    assert data.result_dates('es') == ['2019-11-10', '2023-07-23']
    assert fake.queries == [{'filters': ["scope = 'es'", 'region_id = 0', 'votes IS NOT NULL']}]


def test_result_dates_without_rows_is_empty(monkeypatch):
    monkeypatch.setattr(data, 'EventsData', lambda: FakeEvents())
    assert data.result_dates('es-xx') == []


def test_get_events_flags_the_results_through_result_dates(monkeypatch):
    fake = FakeEvents(rows=[{'date': pd.Timestamp('2023-07-23'), 'scope': 'es', 'name': 'Julio 2023', 'featured': True,
                             'loaded_at': None, 'computed_at': None},
                            {'date': pd.Timestamp('2026-11-29'), 'scope': 'es', 'name': 'Próximas', 'featured': False,
                             'loaded_at': None, 'computed_at': None}])
    monkeypatch.setattr(data, 'Events', lambda: fake)
    asked = []
    monkeypatch.setattr(data, 'result_dates', lambda scope: asked.append(scope) or ['2023-07-23'])
    events = data.get_events('es')
    assert asked == ['es'] and list(events['has_results']) == [True, False]  # fechas contra fechas, no cadenas
