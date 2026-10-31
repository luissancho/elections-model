"""Tests del lanzador `deploy/ecs/job.py` con clientes de boto3 falsos (sin red, sin mtpy)."""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('ecs_job', ROOT / 'deploy' / 'ecs' / 'job.py')
ecs_job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ecs_job)

LIVE_TASKDEF = {
    'taskDefinitionArn': 'arn:aws:ecs:eu-west-1:1:task-definition/elections-model-cluster-elections-model-service:33',
    'family': 'elections-model-cluster-elections-model-service', 'revision': 33, 'status': 'ACTIVE',
    'networkMode': 'awsvpc', 'requiresCompatibilities': ['FARGATE'], 'cpu': '1024', 'memory': '2048',
    'runtimePlatform': {'cpuArchitecture': 'X86_64', 'operatingSystemFamily': 'LINUX'},
    'taskRoleArn': 'arn:aws:iam::1:role/ecsTaskExecutionRole', 'executionRoleArn': 'arn:aws:iam::1:role/ecsTaskExecutionRole',
    'requiresAttributes': [{'name': 'x'}], 'compatibilities': ['EC2', 'FARGATE'], 'registeredAt': 1.0, 'registeredBy': 'x',
    'containerDefinitions': [{'name': 'Main', 'image': '1.dkr.ecr.eu-west-1.amazonaws.com/elections-model:latest',
                              'essential': True, 'portMappings': [{'containerPort': 8042, 'protocol': 'tcp'}],
                              'environment': [], 'environmentFiles': [{'value': 'arn:aws:s3:::elections-s3-bucket/elections.env', 'type': 's3'}],
                              'logConfiguration': {'logDriver': 'awslogs', 'options': {'awslogs-group': '/ecs/elections-model', 'awslogs-region': 'eu-west-1', 'awslogs-stream-prefix': 'ecs'}}}]}
NETWORK = {'subnets': ['subnet-1', 'subnet-2'], 'securityGroups': ['sg-1'], 'assignPublicIp': 'ENABLED'}


class ResourceNotFound(Exception):
    pass


class FakeEcs:
    """Cliente ECS falso: anota las llamadas; `fail` simula un run-task sin capacidad."""
    def __init__(self, fail=False):
        self.calls, self.fail, self.statuses = [], fail, ['RUNNING', 'STOPPED']

    def describe_services(self, **kw):
        self.calls.append(('describe_services', kw))
        return {'services': [{'taskDefinition': LIVE_TASKDEF['taskDefinitionArn'],
                              'deployments': [{'networkConfiguration': {'awsvpcConfiguration': dict(NETWORK)}}]}]}

    def run_task(self, **kw):
        self.calls.append(('run_task', kw))
        if self.fail:
            return {'tasks': [], 'failures': [{'arn': 'x', 'reason': 'RESOURCE:CPU', 'detail': None}]}
        return {'tasks': [{'taskArn': 'arn:aws:ecs:eu-west-1:1:task/c/abc123'}], 'failures': []}

    def describe_tasks(self, **kw):
        self.calls.append(('describe_tasks', kw))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {'tasks': [{'lastStatus': status, 'stoppedReason': 'Essential container in task exited',
                           'containers': [{'exitCode': 3}]}]}

    def describe_task_definition(self, **kw):
        self.calls.append(('describe_task_definition', kw))
        return {'taskDefinition': json.loads(json.dumps(LIVE_TASKDEF))}

    def register_task_definition(self, **kw):
        self.calls.append(('register_task_definition', kw))
        return {'taskDefinition': {'family': kw['family'], 'revision': 2}}


class FakeLogs:
    """Cliente de logs falso: el flujo no existe en la primera llamada, luego devuelve dos eventos."""
    exceptions = types.SimpleNamespace(ResourceNotFoundException=ResourceNotFound)

    def __init__(self):
        self.calls = []

    def filter_log_events(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            raise ResourceNotFound('no stream yet')
        if len(self.calls) == 2:
            return {'events': [{'timestamp': 10, 'message': 'Run job [backtest]...'}, {'timestamp': 11, 'message': 'es: backtest completed'}]}
        return {'events': []}


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / 'ecs.env'
    path.write_text('# cluster\nECS_CLUSTER=elections-model-cluster\nECS_SERVICE=elections-model-service\nAWS_REGION=eu-west-1\nECS_LOG_GROUP=/ecs/elections-model\nECS_SUBNETS=\n')
    monkeypatch.setenv('ECS_ENV_FILE', str(path))
    return path


def run(argv, ecs=None, logs=None):
    out, err = io.StringIO(), io.StringIO()
    code = ecs_job.main(argv, ecs=ecs if ecs is not None else FakeEcs(), logs=logs if logs is not None else FakeLogs(), out=out.write, err=err.write)
    return code, out.getvalue(), err.getvalue()


def test_module_has_no_mtpy_import_and_imports_boto3_lazily():
    source = (ROOT / 'deploy' / 'ecs' / 'job.py').read_text()
    assert 'import mtpy' not in source and 'from mtpy' not in source
    assert source.count('import boto3') == 1 and source.index('import boto3') > source.index('def clients(')


def test_run_job_value_refuses_empty_bad_json_and_load_params():
    with pytest.raises(ecs_job.UsageError, match='job name is required'):
        ecs_job.run_job_value('', None)
    with pytest.raises(ecs_job.UsageError, match='invalid JSON'):
        ecs_job.run_job_value('publish', '{not json')
    with pytest.raises(ecs_job.UsageError, match='load params runs on the laptop'):
        ecs_job.run_job_value('load', '{"what":["polls","params"],"scopes":"all"}')
    with pytest.raises(ecs_job.UsageError, match='load params runs on the laptop'):
        ecs_job.run_job_value('load', '{"what":"params"}')
    with pytest.raises(ecs_job.UsageError, match='one line'):
        ecs_job.run_job_value('publish', '{\n"scopes": "all"\n}')
    assert ecs_job.run_job_value('backtest', None) == 'backtest'
    assert ecs_job.run_job_value('load', '{"scopes":["es"]}') == 'load {"scopes":["es"]}'


def test_run_job_value_refuses_laptop_only_backtest_actions():
    """`push`/`pull`/`out` del backtest y `load params` (también por `pipeline`) usan la copia del repositorio."""
    for job, params in (('backtest', '{"action":"pull","scopes":"all"}'), ('backtest', '{"action":"push"}'),
                        ('backtest', '{"out":"backtest/results"}')):
        with pytest.raises(ecs_job.UsageError, match='backtest push/pull/out run on the laptop'):
            ecs_job.run_job_value(job, params)
    for params in ('{"select":"load.params"}', '{"select":"backtest.backtest","action":"push"}',
                   '{"select":"backtest.backtest","out":"x"}'):
        with pytest.raises(ecs_job.UsageError, match='this pipeline runs on the laptop'):
            ecs_job.run_job_value('pipeline', params)
    assert ecs_job.run_job_value('backtest', '{"scopes":"all"}') == 'backtest {"scopes":"all"}'
    assert ecs_job.run_job_value('pipeline', '{"select":"publish.backtest","chunks":"es"}') == (
        'pipeline {"select":"publish.backtest","chunks":"es"}')
    assert ecs_job.run_job_value('pipeline', '{"select":"backtest.backtest","chunks":"es"}').startswith('pipeline ')


def test_usage_errors_exit_2_without_launching(env_file):
    ecs = FakeEcs()
    assert run([], ecs)[0] == 2 and 'usage' in run([], ecs)[2].lower()
    code, _, err = run(['load', '{"what":["params"]}'], ecs)
    assert code == 2 and 'load params runs on the laptop' in err
    assert run(['publish', '{bad'], ecs)[0] == 2
    assert not [c for c in ecs.calls if c[0] == 'run_task']


def test_missing_env_file_exits_2(tmp_path, monkeypatch):
    monkeypatch.setenv('ECS_ENV_FILE', str(tmp_path / 'none.env'))
    code, _, err = run(['backtest'])
    assert code == 2 and '.env.example' in err


def test_launch_builds_the_run_task_from_the_service_network(env_file):
    ecs = FakeEcs()
    params = '{"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}'
    code, out, _ = run(['publish', params], ecs)
    assert code == 0, out
    request = next(kw for name, kw in ecs.calls if name == 'run_task')
    assert request['cluster'] == 'elections-model-cluster' and request['taskDefinition'] == 'elections-model-job'
    assert request['launchType'] == 'FARGATE' and request['count'] == 1
    assert request['networkConfiguration'] == {'awsvpcConfiguration': NETWORK}
    override = request['overrides']['containerOverrides'][0]
    assert override['name'] == 'Main' and override['environment'] == [{'name': 'RUN_JOB', 'value': 'publish ' + params}]
    assert 'task abc123' in out and 'aws logs tail /ecs/elections-model --log-stream-names job/Main/abc123' in out
    assert 'console.aws.amazon.com/ecs/v2/clusters/elections-model-cluster/tasks/abc123' in out


def test_env_file_overrides_the_network(env_file, monkeypatch):
    env_file.write_text(env_file.read_text() + 'ECS_SUBNETS=subnet-9\nECS_PUBLIC_IP=DISABLED\n')
    ecs = FakeEcs()
    assert run(['backtest', '{"scopes":"all"}'], ecs)[0] == 0
    request = next(kw for name, kw in ecs.calls if name == 'run_task')
    assert request['networkConfiguration']['awsvpcConfiguration'] == {'subnets': ['subnet-9'], 'securityGroups': ['sg-1'], 'assignPublicIp': 'DISABLED'}
    assert request['overrides']['containerOverrides'][0]['environment'][0]['value'] == 'backtest {"scopes":"all"}'


def test_placement_failure_exits_1_with_the_reason(env_file):
    code, out, err = run(['backtest'], FakeEcs(fail=True))
    assert code == 1 and 'RESOURCE:CPU' in err


def test_follow_prints_the_log_and_exits_with_the_container_exit_code(env_file, monkeypatch):
    ecs, logs = FakeEcs(), FakeLogs()
    monkeypatch.setattr(ecs_job.time, 'sleep', lambda s: None)
    code, out, _ = run(['backtest', '{"scopes":["es"]}', '--follow'], ecs, logs)
    assert code == 3
    assert 'Run job [backtest]...' in out and 'es: backtest completed' in out and 'Essential container in task exited' in out
    assert logs.calls[0]['logStreamNames'] == ['job/Main/abc123'] and logs.calls[0]['logGroupName'] == '/ecs/elections-model'
    assert [name for name, _ in ecs.calls if name == 'describe_tasks'] == ['describe_tasks', 'describe_tasks']


def test_register_derives_the_job_definition_from_the_live_web_one(env_file):
    ecs = FakeEcs()
    code, out, _ = run(['--register'], ecs)
    assert code == 0 and 'registered elections-model-job:2' in out
    definition = next(kw for name, kw in ecs.calls if name == 'register_task_definition')
    assert definition['family'] == 'elections-model-job' and definition['cpu'] == '2048' and definition['memory'] == '4096'
    assert definition['taskRoleArn'].endswith('ecsTaskExecutionRole') and definition['requiresCompatibilities'] == ['FARGATE']
    assert not {'taskDefinitionArn', 'revision', 'status', 'requiresAttributes', 'compatibilities', 'registeredAt', 'registeredBy'} & set(definition)
    container = definition['containerDefinitions'][0]
    assert container['name'] == 'Main' and 'portMappings' not in container
    assert container['environment'] == [{'name': 'RUN_JOB', 'value': ''}, {'name': 'DB_STATEMENT_TIMEOUT', 'value': ''}]
    assert container['environmentFiles'] == LIVE_TASKDEF['containerDefinitions'][0]['environmentFiles']
    assert container['logConfiguration']['options']['awslogs-stream-prefix'] == 'job'
    assert not [c for c in ecs.calls if c[0] == 'run_task']


def test_missing_required_config_keys_exit_2(tmp_path, monkeypatch):
    path = tmp_path / 'ecs.env'
    path.write_text('AWS_REGION=eu-west-1\n')
    monkeypatch.setenv('ECS_ENV_FILE', str(path))
    code, _, err = run(['backtest'])
    assert code == 2 and 'needs ECS_CLUSTER and ECS_SERVICE' in err


def test_unknown_service_exits_1_with_a_message(env_file):
    class NoService(FakeEcs):
        def describe_services(self, **kw):
            return {'services': [], 'failures': [{'arn': 'x', 'reason': 'MISSING'}]}

    code, _, err = run(['backtest'], NoService())
    assert code == 1 and 'service elections-model-service not found in cluster elections-model-cluster' in err
    ecs = NoService()
    code, _, err = run(['--register'], ecs)
    assert code == 1 and 'service elections-model-service not found in cluster elections-model-cluster' in err
    assert not [name for name, _ in ecs.calls if name in ('describe_task_definition', 'register_task_definition')]


def test_aws_client_errors_exit_1_with_a_message(env_file):
    class ClientError(Exception):
        response = {'Error': {'Code': 'ClientException'}}

    class Failing(FakeEcs):
        def run_task(self, **kw):
            raise ClientError('TaskDefinition not found')

        def register_task_definition(self, **kw):
            raise ClientError('denied')

    code, _, err = run(['backtest'], Failing())
    assert code == 1 and 'run_task failed: TaskDefinition not found' in err
    code, _, err = run(['--register'], Failing())
    assert code == 1 and 'register_task_definition failed: denied' in err

    class FailingDescribe(FakeEcs):
        def describe_services(self, **kw):
            raise ClientError('AccessDenied')

        def describe_task_definition(self, **kw):
            raise ClientError('no such revision')

        def describe_tasks(self, **kw):
            raise ClientError('throttled')

    for argv in (['backtest'], ['--register']):
        code, _, err = run(argv, FailingDescribe())
        assert code == 1 and 'describe_services failed: AccessDenied' in err

    class FailingDefinition(FakeEcs):
        describe_task_definition = FailingDescribe.describe_task_definition

    code, _, err = run(['--register'], FailingDefinition())
    assert code == 1 and 'describe_task_definition failed: no such revision' in err

    class FailingTasks(FakeEcs):
        describe_tasks = FailingDescribe.describe_tasks

    code, _, err = run(['backtest', '--follow'], FailingTasks())
    assert code == 1 and 'describe_tasks failed: throttled' in err


def test_empty_region_omits_the_console_link_and_passes_none_to_boto3(env_file):
    env_file.write_text(env_file.read_text().replace('AWS_REGION=eu-west-1', 'AWS_REGION='))
    code, out, _ = run(['backtest'])
    assert code == 0 and 'task abc123' in out and 'aws logs tail' in out and 'console.aws.amazon.com' not in out
    assert ecs_job.region_of({'AWS_REGION': ''}) is None and ecs_job.region_of({}) is None
    assert ecs_job.region_of({'AWS_REGION': 'eu-west-1'}) == 'eu-west-1'
