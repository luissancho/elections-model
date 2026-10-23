"""Tests de `S3.connection()`: credenciales vacías caen a la cadena de botocore; sin caché de listados."""
from mtpy.core.services.s3 import S3


def test_empty_credentials_fall_back_to_the_botocore_chain(monkeypatch):
    """Con clave y secreto vacíos se pasa None (rol de tarea); con valores, se pasan tal cual."""
    seen = {}
    monkeypatch.setattr('mtpy.core.services.s3.S3FileSystem', lambda **kw: seen.update(kw) or object())
    S3({'key': '', 'secret': '', 'region': 'eu-west-1'}, 'bucket').connect()
    assert seen['key'] is None and seen['secret'] is None
    S3({'key': 'AK', 'secret': 'SK', 'region': 'eu-west-1'}, 'bucket').connect()
    assert seen['key'] == 'AK' and seen['secret'] == 'SK'


def test_listings_are_not_cached(monkeypatch):
    """Un proceso largo (la web) ve las carpetas que otro proceso publica después de su primer listado."""
    seen = {}
    monkeypatch.setattr('mtpy.core.services.s3.S3FileSystem', lambda **kw: seen.update(kw) or object())
    S3({'key': 'AK', 'secret': 'SK', 'region': 'eu-west-1'}, 'bucket').connect()
    assert seen['use_listings_cache'] is False
