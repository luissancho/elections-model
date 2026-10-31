"""Dobles compartidos por los tests del grupo `publish` (pipelines y job): la fixture `patched`, que sustituye la
publicación y la base, `entry`, una entrada del manifest, `fixed_prov`, la procedencia sin `git`, y
`warned_forecast`, un `publish_forecast` real que avisa al ajustar el promedio."""
import pytest

from mtpy.lib import publish
from mtpy.lib.forecaster import NoPollsError
from tests.fakes import FakeComputer, FakeSimulator, synthetic_event
from tests.test_webapi_unit import PARTIES


def entry(scope, run_id):
    return {'latest': run_id, 'run_at': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'as_of': '2026-10-05',
            'date_last': '2026-10-01', 'n_polls': 6}


def fixed_prov(monkeypatch):
    """Fija `publish.provenance` (sin `git` real) para los jobs de `backtest`."""
    monkeypatch.setattr(publish.backtest, 'provenance', lambda cwd=None: {'commit': 'abc1234', 'dirty': False, 'versions': {}})


@pytest.fixture
def patched(monkeypatch):
    """`resolve_scopes`, `next_event_date`, `publish_forecast`, `publish_analysis` y `publish_parties` sin base: `es`
    publica de verdad con `FakeSimulator` (o `FakeComputer`), `db_stats` y la procedencia sustituidos, `es-md` no
    tiene evento, `es-cb` no tiene sondeos, `es-ar` rompe y `es-ri` falla con un `ValueError` del paquete; el
    catálogo de partidos sale de `PARTIES`."""
    calls = []
    real_forecast = publish.publish_forecast
    real_analysis = publish.publish_analysis
    real_event = publish.publish_event

    def fake_forecast(scope, writer, run_id, event_date, **kwargs):
        calls.append((scope, run_id, event_date, kwargs))
        if scope == 'es-cb':
            raise publish.NothingToPublish('No polls for es-cb 2027-05-23: nothing to forecast')
        if scope == 'es-ri':
            raise ValueError("vote: missing keys ['rows']")
        if scope == 'es-ar':
            raise KeyError('x')
        return real_forecast(scope, writer, run_id, event_date, simulator=FakeSimulator(),
                             stats=lambda scope, event_date: (7, '2026-10-01'),
                             prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    def fake_analysis(scope, writer, **kwargs):
        calls.append((scope, 'analysis', kwargs))
        if scope == 'es-ar':
            raise KeyError('x')
        no_polls = NoPollsError('No polls for es-cb 2027-05-23: nothing to forecast') if scope == 'es-cb' else None
        return real_analysis(scope, writer, computer=FakeComputer(fail_effects=no_polls),
                             stats=lambda scope, event_date: (7, '2026-10-01'),
                             prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    def fake_event(scope, event_date, writer, **kwargs):
        ev = synthetic_event()
        return real_event(scope, event_date, writer, computer=FakeComputer(), events=lambda s: ev.events,
                          results=lambda s, d, r: ev.results.loc[ev.results.date == d], pollsters=lambda: ev.pollsters,
                          seats=lambda s, d, names: {},
                          prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    real_parties = publish.publish_parties

    def fake_parties(writer, **kwargs):
        # not recorded in `calls`: the tests of the scopes read it by position
        return real_parties(writer, parties=lambda: PARTIES, **kwargs)

    monkeypatch.setattr(publish, 'publish_parties', fake_parties)
    monkeypatch.setattr(publish, 'publish_event', fake_event)
    monkeypatch.setattr(publish, 'publish_analysis', fake_analysis)
    monkeypatch.setattr(publish, 'resolve_scopes', lambda scopes, catalogue=None: ['es', 'es-md', 'es-cb', 'es-ar'] if scopes == 'all' else list(scopes))
    next_dates = {'es': '2026-11-29', 'es-cb': '2027-05-23', 'es-ar': '2026-02-08', 'es-ri': '2027-05-23'}
    monkeypatch.setattr(publish, 'next_event_date', lambda scope, date_from=None: next_dates.get(scope))
    monkeypatch.setattr(publish, 'publish_forecast', fake_forecast)
    return calls


NO_POLLS = 'No polls available for `X`: skipped'


def warned_forecast(monkeypatch, messages):
    """`publish_forecast` real con un `FakeSimulator` que avisa `messages` al ajustar el promedio."""
    real = publish.forecast.publish_forecast

    def fake(scope, writer, run_id, event_date, **kwargs):
        return real(scope, writer, run_id, event_date, simulator=FakeSimulator(warn=messages),
                    stats=lambda scope, event_date: (7, '2026-10-01'),
                    prov=lambda: {'commit': 'abc1234', 'dirty': False, 'versions': {}}, **kwargs)

    monkeypatch.setattr(publish, 'publish_forecast', fake)
