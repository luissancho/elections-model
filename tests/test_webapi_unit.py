"""Tests del paquete `mtpy/lib/webapi/` sin servidor ni base de datos: validadores, cachés y lector del paquete."""
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from mtpy.core.api import HttpError
from mtpy.core.app import App
from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish, webapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'bundle')
RUN = '20261008-120000'


def fixture_data(name):
    with open(os.path.join(FIXTURES, name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)['data']


def write_bundle(fs, scope='es', run_id=RUN, manifest=True):
    """Paquete mínimo a partir de los fixtures: un run completo, su history y el manifest."""
    writer = bundle.BundleWriter(fs)
    for part in bundle.RUN_PARTS:
        writer.write_json(bundle.path_part(scope, run_id, part), part, fixture_data(part), scope, run_id=run_id)
    for mode in bundle.MODES:
        for part in bundle.MODE_PARTS:
            writer.write_json(bundle.path_part(scope, run_id, part, mode), part, fixture_data(part), scope,
                              run_id=run_id, mode=mode)
    writer.write_csv(bundle.path_csv(scope, run_id, 'series'), pd.DataFrame({'date': ['2026-10-05'], 'name': ['PP']}))
    writer.write_csv(bundle.path_csv(scope, run_id, 'vote', 'forecast'), pd.DataFrame({'name': ['PP'], 'pct': [40.0]}))
    publish.rebuild_history(writer, scope)
    if manifest:
        publish.update_manifest(writer, {scope: publish.latest_entry(fixture_data('headline') | {'run_id': run_id})})
    return writer


def write_analysis(fs, scope='es'):
    """Análisis publicado de un ámbito a partir de los fixtures: las cinco partes y sus gemelos CSV."""
    writer = bundle.BundleWriter(fs)
    tables = {
        'house-effects': pd.DataFrame({'pollster': ['CIS'], 'name': ['PP'], 'cycle': [-3.02]}),
        'herding': pd.DataFrame({'pollster': ['GAD3'], 'herding': [0.742]}),
        'herding-events': pd.DataFrame({'event_date': ['2023-07-23'], 'pollster': ['GAD3'], 'ratio': [0.781]}),
        'poll-errors': pd.DataFrame({'event_date': ['2023-07-23'], 'pollster': ['GAD3'], 'Derecha': [2.81]}),
        'error-data': pd.DataFrame({'date': ['2023-07-16'], 'pollster': ['GAD3'], 'error': [1.67]}),
    }
    for name, frame in tables.items():
        writer.write_csv(bundle.path_analysis_csv(scope, name), frame)
    for part in bundle.ANALYSIS_PARTS[1:] + bundle.ANALYSIS_PARTS[:1]:  # `meta` el último, como `publish`
        schema = bundle.ANALYSIS_SCHEMAS[part]
        writer.write_json(bundle.path_analysis(scope, part), schema, fixture_data(schema), scope)
    return writer


def write_event(fs, scope='es', date='2023-07-23', model=True):
    """Elección pasada publicada a partir de los fixtures: las cinco partes (con `scope` y `event_date`
    sustituidos) y sus CSV mínimos; sin `model`, el modelo va vacío, `has_model` falso y sin gemelos `model-*`."""
    writer = bundle.BundleWriter(fs)
    tables = {
        'pollsters': pd.DataFrame({'pollster': ['GAD3'], 'error_avg': [0.99]}),
        'polls': pd.DataFrame({'event_date': [date], 'pollster': ['GAD3']}),
        'results': pd.DataFrame({'party': ['PP'], 'pct': [33.06]}),
    }
    if model:
        tables |= {f'model-{name}': pd.DataFrame({'event_date': [date], 'horizon': [6]})
                   for name in ('shares', 'seats', 'blocks')}
    for name, frame in tables.items():
        writer.write_csv(bundle.path_event_csv(scope, date, name), frame)
    for part in bundle.EVENT_PARTS[1:] + bundle.EVENT_PARTS[:1]:  # `meta` el último, como `publish`
        schema = bundle.EVENT_SCHEMAS[part]
        data = fixture_data(schema)
        if part == 'meta':
            data = data | {'scope': scope, 'event_date': date, 'has_model': model}
        elif part == 'model' and not model:
            data = {key: [] for key in data}
        writer.write_json(bundle.path_event(scope, date, part), schema, data, scope)
    return writer


def write_backtest(fs, scope='es', meta=None):
    """Backtest publicado de un ámbito a partir de los fixtures: las seis partes y sus cinco CSV; `meta` sustituye
    campos de `meta.data`."""
    writer = bundle.BundleWriter(fs)
    for name in bundle.BACKTEST_CSV:
        writer.write_csv(bundle.path_backtest_csv(scope, name), pd.DataFrame({'event_date': ['2023-07-23'], 'horizon': [6]}))
    for part in bundle.BACKTEST_PARTS[1:] + bundle.BACKTEST_PARTS[:1]:  # `meta` el último, como `publish`
        schema = bundle.BACKTEST_SCHEMAS[part]
        data = fixture_data(schema)
        if part == 'meta':
            data = data | {'scope': scope} | (meta or {})
        writer.write_json(bundle.path_backtest(scope, part), schema, data, scope)
    return writer


@pytest.fixture(autouse=True)
def clear_queries():
    """La caché de consultas a base es del módulo: se vacía antes de cada test."""
    webapi.queries._QUERIES.clear()
    yield
    webapi.queries._QUERIES.clear()


def use_database(app, host='db.test'):
    """Base simulada en `app`: un `db` cualquiera y `config.db.params.host`; devuelve el `db`."""
    from mtpy.core.app import Config
    app.db = db = object()
    if app.config is None:
        app.set('config', Config({}))
    app.config.offset_set('db', {'params': {'host': host}})
    return db


@pytest.fixture
def site_app(fresh_app, tmp_path):
    """`App` con `fs` local, el catálogo de `data/` y un paquete mínimo publicado para `es`."""
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    write_bundle(fresh_app.fs)
    return fresh_app


def test_validators_accept_the_catalogue_and_reject_the_rest(site_app):
    assert webapi.check_scope('es') == 'es' and webapi.check_scope('es-md') == 'es-md'
    for bad in ("es'", 'es;', 'es-xx', 'ES', 'es/..', ''):
        with pytest.raises(HttpError) as err:
            webapi.check_scope(bad)
        assert err.value.status == 400
    assert webapi.check_run(None) is None and webapi.check_run('') is None
    assert webapi.check_run(RUN) == RUN
    for bad in ('2026-10-08', RUN + '\n', RUN + '/..', 'latest'):
        with pytest.raises(HttpError) as err:
            webapi.check_run(bad)
        assert err.value.status == 400
    assert webapi.check_mode('nowcast') == 'nowcast'
    with pytest.raises(HttpError) as err:
        webapi.check_mode('tomorrow')
    assert err.value.status == 400
    assert webapi.check_part('series', webapi.RUN_PARTS) == 'series'
    with pytest.raises(HttpError) as err:
        webapi.check_part('nope', webapi.RUN_PARTS)
    assert err.value.status == 404
    assert webapi.check_format(None) == 'json' and webapi.check_format('csv') == 'csv'
    with pytest.raises(HttpError) as err:
        webapi.check_format('xml')
    assert err.value.status == 400


def test_ttl_cache_with_an_injected_clock():
    now = [0.0]
    calls = []
    cache = webapi.TTLCache(ttl=60, clock=lambda: now[0])

    def loader():
        calls.append(1)
        return len(calls)
    assert cache.get('k', loader) == 1 and cache.get('k', loader) == 1
    now[0] = 59.9
    assert cache.get('k', loader) == 1
    now[0] = 60.0
    assert cache.get('k', loader) == 2

    def boom():
        raise FileNotFoundError('x')
    with pytest.raises(FileNotFoundError):
        cache.get('other', boom)
    assert cache.get('other', lambda: 'ok') == 'ok'


def test_lru_cache_evicts_the_oldest():
    cache = webapi.LRUCache(maxsize=2)
    assert cache.get('a', lambda: 1) == 1 and cache.get('b', lambda: 2) == 2
    assert cache.get('a', lambda: 99) == 1          # acierto: `a` pasa a ser el más reciente
    assert cache.get('c', lambda: 3) == 3           # expulsa `b`
    assert cache.get('b', lambda: 20) == 20 and len(cache) == 2


def test_settings_defaults_without_config(fresh_app):
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}
    from mtpy.core.app import Config
    fresh_app.set('config', Config({'web': {'prefix': 'site-dry/v1', 'cache_ttl': '5'}}))
    assert webapi.settings() == {'prefix': 'site-dry/v1', 'cache_ttl': 5.0}
    fresh_app.set('config', Config({'web': {'prefix': '', 'cache_ttl': ''}}))
    assert webapi.settings() == {'prefix': 'site/v1', 'cache_ttl': 60.0}


def test_site_reads_the_bundle_and_resolves_latest(site_app):
    site = webapi.site()
    assert site is webapi.site()
    manifest = bundle.loads(site.manifest())
    assert manifest['schema'] == 'manifest@1' and site.latest_run('es') == RUN
    assert bundle.loads(site.history('es'))['data']['runs'][0]['run_id'] == RUN
    assert bundle.loads(site.run_file('es', RUN, 'meta'))['schema'] == 'meta@1'
    assert bundle.loads(site.run_file('es', RUN, 'vote', 'forecast'))['mode'] == 'forecast'
    assert site.run_csv('es', RUN, 'series') == b'date,name\n2026-10-05,PP\n'
    assert site.run_csv('es', RUN, 'vote', 'forecast').startswith(b'name,pct\n')
    assert site.freeze() == {'active': False, 'message': None}
    for call, status in (
        (lambda: site.latest_run('es-md'), 404), (lambda: site.history('es-md'), 404),
        (lambda: site.run_file('es', '20200101-000000', 'meta'), 404), (lambda: site.run_csv('es', RUN, 'meta'), 404),
    ):
        with pytest.raises(HttpError) as err:
            call()
        assert err.value.status == status


def test_site_scopes_crosses_the_catalogue_with_the_manifest(site_app):
    data = webapi.site().scopes()
    codes = [row['code'] for row in data['scopes']]
    assert codes[:2] == ['es', 'es-an'] and len(codes) == 18 and data['contract'] == 1
    es = data['scopes'][0]
    assert es['simulable'] is True and es['latest'] == RUN and es['seats'] == 350 and es['parent'] is None
    md = [row for row in data['scopes'] if row['code'] == 'es-md'][0]
    assert md['simulable'] is False and md['latest'] is None and md['parent'] == 'es' and md['name'] == 'Madrid'


def test_site_without_manifest_is_a_503_and_caches_nothing(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    with pytest.raises(HttpError) as err:
        webapi.site().manifest()
    assert err.value.status == 503
    write_bundle(fresh_app.fs)
    assert bundle.loads(webapi.site().manifest())['data']['scopes']['es']['latest'] == RUN


def test_site_pointers_expire_with_the_ttl(fresh_app, tmp_path):
    fresh_app.set('fs', FileSystem(str(tmp_path)))
    fresh_app.set('data', FileSystem(os.path.join(ROOT, 'data')))
    writer = write_bundle(fresh_app.fs)
    now = [0.0]
    site = webapi.Site(bundle.BundleReader(fresh_app.fs), ttl=60, clock=lambda: now[0])
    assert site.latest_run('es') == RUN
    publish.update_manifest(writer, {'es': {'latest': '20261009-120000'}})
    assert site.latest_run('es') == RUN
    now[0] = 61.0
    assert site.latest_run('es') == '20261009-120000'


def test_site_requires_a_file_system(fresh_app):
    with pytest.raises(HttpError) as err:
        webapi.site()
    assert err.value.status == 503


def test_webapi_import_stays_light():
    """Los workers de la web no deben cargar el modelo: `webapi` solo importa `bundle`.

    scipy y statsmodels quedan fuera de la comprobación: `mtpy.core.io` (y con él `bundle`) ya los
    arrastra a través de `mtpy.core.utils.helpers`, ajeno a `webapi`.
    """
    import subprocess
    import sys
    code = ("import sys, mtpy.lib.webapi; "
            "print(sorted(m for m in ('mtpy.lib.data', 'mtpy.lib.publish', 'mtpy.lib.simulator') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == '[]'


def test_settings_fall_back_to_the_default_ttl_on_garbage(monkeypatch):
    fake = SimpleNamespace(config=SimpleNamespace(web=SimpleNamespace(prefix='', cache_ttl='abc')))
    monkeypatch.setattr(App, 'get_', staticmethod(lambda: fake))
    with pytest.warns(UserWarning):
        assert webapi.settings() == {'prefix': bundle.PREFIX, 'cache_ttl': 60.0}


def test_check_region_accepts_integers_only():
    assert webapi.check_region(None) is None and webapi.check_region('') is None
    assert webapi.check_region('28') == 28
    for bad in ('abc', '-1', '28.0', '1' * 7, '28;'):
        with pytest.raises(HttpError) as err:
            webapi.check_region(bad)
        assert err.value.status == 400 and err.value.message == 'invalid region'


def test_check_date_accepts_real_iso_dates_only():
    assert webapi.check_date('2026-11-29') == '2026-11-29'
    for bad in ('2026-13-01', '29/11/2026', '', None, '20261129', '2026-02-30', "2026-11-29'"):
        with pytest.raises(HttpError) as err:
            webapi.check_date(bad)
        assert err.value.status == 400 and err.value.message == 'invalid date'


def test_check_pollster_accepts_positive_integers_only():
    assert webapi.check_pollster('12') == 12
    for bad in ('abc', '-1', '', None, '1' * 7, '3;'):
        with pytest.raises(HttpError) as err:
            webapi.check_pollster(bad)
        assert err.value.status == 400 and err.value.message == 'invalid pollster'


def test_check_limit_defaults_and_bounds():
    assert webapi.check_limit(None) == 1000 and webapi.check_limit('') == 1000
    assert webapi.check_limit('50') == 50 and webapi.check_limit('5000') == 5000
    assert webapi.check_limit(None, default=10) == 10
    for bad in ('0', '9999', 'x', '-5', '5.0'):
        with pytest.raises(HttpError) as err:
            webapi.check_limit(bad)
        assert err.value.status == 400 and err.value.message == 'invalid limit'


def test_database_without_db_is_a_503(fresh_app):
    with pytest.raises(HttpError) as err:
        webapi.database()
    assert err.value.status == 503 and err.value.message == 'no database configured'
    sentinel = use_database(fresh_app)
    assert webapi.database() is sentinel


@pytest.mark.parametrize('host', ['', None])
def test_database_without_host_is_a_503_before_reading(fresh_app, monkeypatch, host):
    """`mtpy.run()` siempre crea el adaptador: con `DB_HOST` vacío no hay base, aunque `app.db` exista."""
    calls = []
    monkeypatch.setattr(webapi.queries, 'get_drift', lambda scope: calls.append(scope) or pd.DataFrame())
    use_database(fresh_app, host=host)
    with pytest.raises(HttpError) as err:
        webapi.query_drift('es')
    assert err.value.status == 503 and err.value.message == 'no database configured' and calls == []
    fresh_app.config.offset_unset('db')  # sin sección `db` tampoco hay base
    with pytest.raises(HttpError) as err:
        webapi.database()
    assert err.value.message == 'no database configured'
    use_database(fresh_app, host='db.test')
    assert webapi.query_drift('es').empty and calls == ['es']


def ratings_frame():
    """Ratings de dos eventos (el último es el próximo) y dos casas, como los devuelve `get_ratings`."""
    return pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23', '2023-07-23', '2026-11-29', '2026-11-29']),
        'pollster_id': [2, 3, 2, 3],
        'pollster': ['GAD3', 'CIS', 'GAD3', 'CIS'],
        'event_scope': ['es'] * 4,
        'rating': [1.4, 0.6, 1.5, float('nan')],
    })


def test_query_ratings_returns_the_next_event_and_caches(fresh_app, monkeypatch):
    calls = []

    def fake(scope):
        calls.append(scope)
        return ratings_frame()
    monkeypatch.setattr(webapi.queries, 'get_ratings', fake)
    with pytest.raises(HttpError) as err:
        webapi.query_ratings('es')
    assert err.value.status == 503 and calls == []  # sin base no se llega al lector
    use_database(fresh_app)
    frame = webapi.query_ratings('es')
    assert frame['event_date'].dt.strftime('%Y-%m-%d').unique().tolist() == ['2026-11-29']
    assert frame['pollster'].tolist() == ['GAD3', 'CIS'] and calls == ['es']
    webapi.query_ratings('es')
    assert calls == ['es']  # segunda llamada desde la caché
    past = webapi.query_ratings('es', '2023-07-23')
    assert past['rating'].tolist() == [1.4, 0.6] and calls == ['es']  # el evento se filtra fuera de la caché


def test_query_errors_become_503_and_are_logged(fresh_app, monkeypatch):
    errors = []
    use_database(fresh_app)
    fresh_app.set('logger', SimpleNamespace(error=errors.append))

    def boom(scope):
        raise RuntimeError('connection refused')
    monkeypatch.setattr(webapi.queries, 'get_ratings', boom)
    with pytest.raises(HttpError) as err:
        webapi.query_ratings('es')
    assert err.value.status == 503 and err.value.message == 'database unavailable'
    assert len(errors) == 1 and 'connection refused' in errors[0] and 'Traceback' in errors[0]
    calls = []
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: calls.append(scope) or ratings_frame())
    # el fallo no se cacheó: la llamada siguiente vuelve al lector
    assert webapi.query_ratings('es')['pollster'].tolist() == ['GAD3', 'CIS'] and calls == ['es']


def test_query_pollster_profile_and_unknown_pollster(fresh_app, monkeypatch):
    use_database(fresh_app)
    monkeypatch.setattr(webapi.queries, 'get_pollsters', lambda: pd.DataFrame({
        'id': [2, 3], 'name': ['GAD3', 'CIS'], 'mtype': ['private', 'public'], 'quality': [1.0, 0.8],
    }))
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame())
    monkeypatch.setattr(webapi.queries, 'get_house_effects', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [3], 'party': ['PP'], 'dev_result_c': [-1.2],
    }))
    monkeypatch.setattr(webapi.queries, 'get_herding', lambda scope: pd.DataFrame({
        'event_date': pd.to_datetime(['2023-07-23']), 'pollster_id': [2], 'ratio': [0.78],
    }))
    profile = webapi.query_pollster(3, 'es')
    assert list(profile) == ['pollster', 'ratings', 'house_effects', 'herding']
    assert profile['pollster'] == {'id': 3, 'name': 'CIS', 'mtype': 'public', 'quality': 0.8}
    assert [row['event_date'] for row in profile['ratings']] == ['2023-07-23', '2026-11-29']
    assert profile['ratings'][1]['rating'] is None
    assert profile['house_effects'][0]['dev_result_c'] == -1.2 and profile['herding'] == []
    with pytest.raises(HttpError) as err:
        webapi.query_pollster(999, 'es')
    assert err.value.status == 404 and err.value.message == 'pollster not found'


def test_query_pollster_reuses_the_cached_scope_tables(fresh_app, monkeypatch):
    use_database(fresh_app)
    calls = []

    def counted(name, frame):
        return lambda *args: calls.append(name) or frame
    pollsters = pd.DataFrame({'id': [2, 3], 'name': ['GAD3', 'CIS']})
    monkeypatch.setattr(webapi.queries, 'get_pollsters', counted('pollsters', pollsters))
    monkeypatch.setattr(webapi.queries, 'get_ratings', counted('ratings', ratings_frame()))
    monkeypatch.setattr(webapi.queries, 'get_house_effects', counted('house_effects', pd.DataFrame({'pollster_id': [3]})))
    monkeypatch.setattr(webapi.queries, 'get_herding', counted('herding', pd.DataFrame({'pollster_id': [2]})))
    webapi.query_ratings('es')
    webapi.query_house_effects('es')
    webapi.query_herding('es')
    webapi.query_pollsters()
    assert sorted(calls) == ['herding', 'house_effects', 'pollsters', 'ratings']
    assert webapi.query_pollster(3, 'es')['house_effects'] == [{'pollster_id': 3}]
    assert webapi.query_pollster(2, 'es')['herding'] == [{'pollster_id': 2}]
    assert len(calls) == 4  # ninguna lectura más


def test_query_polls_defaults_to_the_next_event_and_truncates(fresh_app, monkeypatch):
    use_database(fresh_app)
    seen = []

    def fake_series(scope, event_dates):
        seen.append((scope, event_dates))
        return pd.DataFrame({
            'event_date': pd.to_datetime(['2026-11-29'] * 3),
            'date': pd.to_datetime(['2026-09-01', '2026-10-01', '2026-09-15']),
            'pollster_id': [2, 3, 2], 'PP': [33.0, 34.0, 35.0],
        })
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: '2026-11-29')
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame())
    monkeypatch.setattr(webapi.queries, 'get_poll_series', fake_series)
    frame = webapi.query_polls('es', limit=2)
    assert seen == [('es', '2026-11-29')]
    assert frame['date'].dt.strftime('%Y-%m-%d').tolist() == ['2026-10-01', '2026-09-15']
    frame = webapi.query_polls('es', event='2026-11-29', pollster=2, limit=50)
    assert seen == [('es', '2026-11-29')] and frame['pollster_id'].tolist() == [2, 2]  # misma entrada de caché
    webapi.query_polls('es', event='2023-07-23')
    assert seen[-1] == ('es', '2023-07-23')
    with pytest.raises(HttpError) as err:
        webapi.query_polls('es', event='2020-01-01')
    assert err.value.status == 404 and err.value.message == 'event not found' and len(seen) == 2
    with pytest.raises(HttpError) as err:
        webapi.query_ratings('es', '2020-01-01')
    assert err.value.message == 'event not found'


def test_query_results_reads_each_election_once_for_every_region(fresh_app, monkeypatch):
    use_database(fresh_app)
    seen = []

    def fake_results(scope, date, region):
        seen.append(('results', scope, date, region))
        return pd.DataFrame({'date': pd.to_datetime([date] * 3), 'region_id': [0, 13, 28],
                             'party': ['PP'] * 3, 'pct': [33.1, 35.0, 40.2]})
    monkeypatch.setattr(webapi.queries, 'get_event_results', fake_results)
    monkeypatch.setattr(webapi.queries, 'get_event_dates', lambda scope: seen.append(('events', scope)) or ['2023-07-23'])
    assert webapi.query_results('es', '2023-07-23')['pct'].tolist() == [33.1]
    frame = webapi.query_results('es', '2023-07-23', 28)
    assert frame['pct'].tolist() == [40.2] and frame.index.tolist() == [0]
    assert webapi.query_results('es', '2023-07-23', 99).empty
    assert seen == [('events', 'es'), ('results', 'es', '2023-07-23', None)]
    with pytest.raises(HttpError) as err:
        webapi.query_results('es', '2020-01-01')
    assert err.value.status == 404 and err.value.message == 'event not found'
    assert len(seen) == 2  # una fecha desconocida no lee resultados
    assert sorted(webapi.queries._QUERIES._items) == [('events', 'es'), ('results', 'es', '2023-07-23')]


def test_resolved_events_of_ratings_and_polls(fresh_app, monkeypatch):
    use_database(fresh_app)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame())
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: '2026-11-29')
    assert webapi.ratings_event('es') == '2026-11-29' and webapi.ratings_event('es', '2023-07-23') == '2023-07-23'
    assert webapi.polls_event('es') == '2026-11-29' and webapi.polls_event('es', '2023-07-23') == '2023-07-23'
    with pytest.raises(HttpError) as err:
        webapi.ratings_event('es', '2020-01-01')
    assert err.value.message == 'event not found'
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame().iloc[0:0])
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: None)
    webapi.queries._QUERIES.clear()
    assert webapi.ratings_event('es') is None and webapi.polls_event('es') is None


def test_query_cache_keys_are_bounded_per_scope(fresh_app, monkeypatch):
    use_database(fresh_app)
    monkeypatch.setattr(webapi.queries, 'get_ratings', lambda scope: ratings_frame())
    monkeypatch.setattr(webapi.queries, 'get_next_event_date', lambda scope: '2026-11-29')
    monkeypatch.setattr(webapi.queries, 'get_poll_series', lambda scope, event: pd.DataFrame({'date': [], 'pollster_id': []}))
    webapi.query_ratings('es', '2023-07-23')
    webapi.query_polls('es', pollster=2, limit=5)
    webapi.query_polls('es', limit=50)
    with pytest.raises(HttpError):
        webapi.query_polls('es', event='1999-01-01')
    assert sorted(webapi.queries._QUERIES._items) == [('next_event', 'es'), ('polls', 'es', '2026-11-29'), ('ratings', 'es')]


def test_query_cache_evicts_beyond_its_cap():
    cache = webapi.queries._QUERIES
    assert cache.maxsize == webapi.QUERY_CACHE_SIZE == 128
    for i in range(webapi.QUERY_CACHE_SIZE + 1):
        cache.get(('k', i), lambda i=i: i)
    assert len(cache) == webapi.QUERY_CACHE_SIZE
    assert cache.get(('k', 0), lambda: 'reloaded') == 'reloaded'  # la primera entrada fue expulsada
    assert cache.get(('k', webapi.QUERY_CACHE_SIZE), lambda: 'miss') == webapi.QUERY_CACHE_SIZE


def test_ttl_cache_cap_keeps_the_recently_used():
    cache = webapi.TTLCache(ttl=60, maxsize=2)
    cache.get('a', lambda: 1)
    cache.get('b', lambda: 2)
    assert cache.get('a', lambda: 99) == 1  # acierto: `a` pasa a ser el más reciente
    cache.get('c', lambda: 3)               # expulsa `b`
    assert cache.get('b', lambda: 20) == 20 and cache.get('a', lambda: 99) == 99 and len(cache) == 2


def test_frame_to_records_and_csv():
    frame = pd.DataFrame({
        'event_date': pd.to_datetime(['2026-11-29', '2023-07-23']),
        'name': ['PP', 'PSOE'],
        'pct': [33.5, float('nan')],
    }).set_index('name')
    records = webapi.frame_to_records(frame)
    assert records == [
        {'name': 'PP', 'event_date': '2026-11-29', 'pct': 33.5},
        {'name': 'PSOE', 'event_date': '2023-07-23', 'pct': None},
    ]
    assert webapi.frame_to_records(frame.reset_index())[0] == records[0]  # índice por defecto: se descarta
    assert webapi.frame_to_csv(frame.reset_index()) == b'name,event_date,pct\nPP,2026-11-29,33.5\nPSOE,2023-07-23,\n'


def test_site_analysis_reads_the_published_files(site_app):
    site = webapi.site()
    with pytest.raises(HttpError) as err:
        site.analysis('es', 'herding')
    assert err.value.status == 404 and err.value.message == 'analysis not published'
    with pytest.raises(HttpError) as err:
        site.analysis_csv('es', 'herding')
    assert err.value.status == 404 and err.value.message == 'analysis not published'
    write_analysis(site_app.fs)
    assert bundle.loads(site.analysis('es', 'herding'))['schema'] == 'analysis-herding@1'
    assert site.analysis_csv('es', 'herding-events').startswith(b'event_date,pollster,ratio\n')


def test_webapi_package_layout():
    """Un submódulo por papel y ninguno tapado por un nombre exportado por el paquete (D1)."""
    import inspect
    import pkgutil
    names = sorted(m.name for m in pkgutil.iter_modules(webapi.__path__))
    assert names == ['caches', 'checks', 'published', 'queries']
    for name in names:
        assert inspect.ismodule(getattr(webapi, name)), name
    assert webapi.TTLCache.__module__ == 'mtpy.lib.webapi.caches'
    assert webapi.check_scope.__module__ == 'mtpy.lib.webapi.checks'
    assert webapi.Site.__module__ == 'mtpy.lib.webapi.published'
    assert webapi.query_ratings.__module__ == 'mtpy.lib.webapi.queries'
    assert webapi.ROUTES[0][0] == '/api/v1/health'


def test_site_events_and_backtest_read_the_published_files(site_app):
    fs = site_app.fs
    write_event(fs)
    write_event(fs, date='2019-11-10', model=False)
    write_backtest(fs)
    s = webapi.site()
    assert s.events('es') == ['2019-11-10', '2023-07-23'] and s.events('es-cb') == []
    assert json.loads(s.event('es', '2023-07-23', 'pollsters'))['schema'] == 'event-pollsters@1'
    assert s.event_csv('es', '2023-07-23', 'model-seats').startswith(b'event_date')
    assert json.loads(s.backtest('es', 'by-horizon'))['data']['rows'][0]['horizon'] == 6
    for call, message in ((lambda: s.event('es', '2020-01-01', 'meta'), 'event not published'),
                          (lambda: s.backtest('es-cb', 'metrics'), 'backtest not published')):
        with pytest.raises(HttpError) as err:
            call()
        assert err.value.status == 404 and err.value.message == message
