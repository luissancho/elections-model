"""Tests de los adaptadores de base de datos (`mtpy/core/data.py`, `mtpy/core/dal/`) sin servidor."""
import pytest

from mtpy.core import data as core_data
from mtpy.core.dal.MySQL import MySQL
from mtpy.core.dal.PostgreSQL import PostgreSQL

PARAMS = {'host': 'db.test', 'port': '5432', 'username': 'u', 'password': 'p', 'stage_dir': None}


@pytest.fixture
def engines(fresh_app, monkeypatch):
    """`create_engine` sustituido: guarda la URL y los argumentos de cada motor creado."""
    seen = []

    def fake_create_engine(url, **kwargs):
        seen.append((url, kwargs))
        return object()
    monkeypatch.setattr(core_data, 'create_engine', fake_create_engine)
    return seen


@pytest.mark.parametrize('timeout', [None, ''])
def test_postgresql_engine_has_a_connect_timeout_and_unbounded_statements(engines, timeout):
    PostgreSQL({**PARAMS, 'statement_timeout': timeout}, 'elections')._build_client()
    url, kwargs = engines[0]
    assert url == 'postgresql+psycopg2://u:p@db.test:5432/elections'
    assert kwargs == {'pool_pre_ping': True, 'connect_args': {'connect_timeout': 5}}


def test_postgresql_engine_bounds_statements_when_configured(engines):
    PostgreSQL({**PARAMS, 'statement_timeout': '10000'}, 'elections')._build_client()
    assert engines[0][1]['connect_args'] == {'connect_timeout': 5, 'options': '-c statement_timeout=10000'}
    with pytest.raises(ValueError, match='statement_timeout'):
        PostgreSQL({**PARAMS, 'statement_timeout': '10s'}, 'elections')._build_client()


def test_other_sql_adapters_pass_no_driver_arguments(engines):
    MySQL({**PARAMS, 'statement_timeout': '10000'}, 'elections')._build_client()
    assert engines[0][1]['connect_args'] == {}


def test_insert_values_escape_single_quotes_per_dialect(fresh_app):
    """PostgreSQL (`standard_conforming_strings`) dobla la comilla; MySQL la escapa con barra. Los dos puntos se
    escapan para `sqlalchemy.text` en ambos."""
    import pandas as pd
    frame = pd.DataFrame({'id': [1], 'name': ["Gabinet d'Estudis"], 'url': ['https://x.y/z']})
    pg = PostgreSQL({**PARAMS, 'statement_timeout': None}, 'elections').build_insert_query('t', frame)
    assert "'Gabinet d''Estudis'" in pg and "'https\\://x.y/z'" in pg
    my = MySQL({**PARAMS, 'statement_timeout': None}, 'elections').build_insert_query('t', frame)
    assert "'Gabinet d\\'Estudis'" in my and "'https\\://x.y/z'" in my
