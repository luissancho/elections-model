"""Tests de los clientes de boto3 de `mtpy/core/services/aws.py`: credenciales vacías y fallos del logger."""
import logging

import pytest

from mtpy.core.services import aws

AWS = {'key': '', 'secret': '', 'region': 'eu-west-1'}


class FailingLogs:
    """Cliente de CloudWatch Logs que falla al crear el flujo (sin permisos, sin grupo, sin red)."""

    def create_log_stream(self, **kwargs):
        raise RuntimeError('AccessDenied: logs:CreateLogStream')

    def put_log_events(self, **kwargs):
        raise RuntimeError('AccessDenied: logs:PutLogEvents')


@pytest.mark.parametrize('build', [
    lambda: aws.CloudWatchLogHandler(AWS, {'group': 'grupo'}).connect(),
    lambda: aws.Firehose(AWS, {'stream': 'flujo'}).connect(),
    lambda: aws.SQS(AWS, {'interval': 1}).connect(),
])
def test_empty_credentials_fall_back_to_the_botocore_chain(monkeypatch, build):
    """Con clave y secreto vacíos el cliente recibe None (cadena de botocore: rol de la tarea de ECS)."""
    seen = {}
    monkeypatch.setattr(aws.boto3, 'client', lambda service, **kw: seen.update(kw) or object())
    build()
    assert seen['aws_access_key_id'] is None and seen['aws_secret_access_key'] is None
    assert seen['region_name'] == 'eu-west-1'


def test_given_credentials_are_passed_through(monkeypatch):
    """Con clave y secreto, el cliente los recibe tal cual."""
    seen = {}
    monkeypatch.setattr(aws.boto3, 'client', lambda service, **kw: seen.update(kw) or object())
    aws.CloudWatchLogHandler({'key': 'AK', 'secret': 'SK', 'region': 'eu-west-1'}, {'group': 'grupo'}).connect()
    assert seen['aws_access_key_id'] == 'AK' and seen['aws_secret_access_key'] == 'SK'


def test_a_failing_cloudwatch_call_never_reaches_the_caller_of_the_logger(monkeypatch):
    """Si CloudWatch falla, `emit` llama a `handleError` y el `logger.error` del job o de la web no lanza."""
    monkeypatch.setattr(aws.boto3, 'client', lambda service, **kw: FailingLogs())
    handler = aws.CloudWatchLogHandler(AWS, {'group': 'grupo'})
    handled = []
    monkeypatch.setattr(handler, 'handleError', handled.append)
    logger = logging.getLogger('tests.aws.cwl')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.error('publish: failed scopes: es')
        logger.info('es: published')
    finally:
        logger.removeHandler(handler)
    assert [r.getMessage() for r in handled] == ['publish: failed scopes: es', 'es: published']


def test_default_handle_error_writes_to_stderr_and_does_not_raise(monkeypatch, capsys):
    """Con el `handleError` de `logging` por defecto, el fallo va a stderr y no se propaga."""
    monkeypatch.setattr(aws.boto3, 'client', lambda service, **kw: FailingLogs())
    handler = aws.CloudWatchLogHandler(AWS, {'group': 'grupo'})
    handler.handle(logging.makeLogRecord({'name': 'job', 'msg': 'alerta', 'levelno': logging.ERROR}))
    assert 'AccessDenied' in capsys.readouterr().err
