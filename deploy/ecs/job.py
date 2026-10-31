"""Launch an elections-model batch job as a one-off ECS Fargate task from the laptop.

It runs with the AWS credentials of the owner (boto3 default chain) and does not
import ``mtpy``. Configuration comes from ``deploy/ecs.env`` (see
``deploy/ecs.env.example``) or from the file named by ``ECS_ENV_FILE``.

Usage::

    python deploy/ecs/job.py publish '{"what":["forecast","analysis"],"scopes":["es"],"n_sim":10000}' --follow
    python deploy/ecs/job.py backtest '{"scopes":"all"}'
    python deploy/ecs/job.py --register
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

CONTAINER = 'Main'
STREAM_PREFIX = 'job'


class UsageError(Exception):
    """Wrong command line arguments."""


class ConfigError(Exception):
    """Missing or unusable configuration file."""


class LaunchError(Exception):
    """AWS refused to start the task."""


def say(out, text: str) -> None:
    """Write one whole line through the ``out`` callable."""
    out(text + '\n')


def load_config(path: str) -> dict:
    """Read a ``KEY=VALUE`` file, ignoring blank lines and ``#`` comments.

    Parameters
    ----------
    path : str
        Path of the env file.

    Returns
    -------
    dict
        Keys and stripped values.

    Raises
    ------
    ConfigError
        If the file does not exist.
    """
    if not os.path.isfile(path):
        raise ConfigError(f'no ecs.env at {path} (copy deploy/ecs.env.example)')
    config = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    return config


def run_job_value(job: Optional[str], params: Optional[str]) -> str:
    """Build the ``RUN_JOB`` value ``'<job> <json>'`` keeping the JSON as received.

    Parameters
    ----------
    job : str or None
        Job name.
    params : str or None
        JSON text with the job parameters.

    Returns
    -------
    str
        The value for the ``RUN_JOB`` environment variable.

    Raises
    ------
    UsageError
        For an empty job, invalid JSON or ``load`` with the ``params`` target.
    """
    if not job:
        raise UsageError('job name is required')
    if params is None:
        return job
    try:
        parsed = json.loads(params)
    except ValueError as e:
        raise UsageError(f'invalid JSON: {e}')
    if '\n' in params:
        raise UsageError('params must be on one line')
    if job == 'load' and isinstance(parsed, dict):
        what = parsed.get('what')
        targets = what if isinstance(what, list) else [what]
        if 'params' in targets:
            raise UsageError('load params runs on the laptop (it rewrites data/params.json)')
    return f'{job} {params}'


def service_network(ecs, cluster: str, service: str) -> dict:
    """Return the ``awsvpcConfiguration`` of the service's current deployment."""
    response = ecs.describe_services(cluster=cluster, services=[service])
    services = response.get('services') or []
    if not services or not services[0].get('deployments'):
        raise LaunchError(f'service {service} not found in cluster {cluster}')
    deployment = services[0]['deployments'][0]
    return deployment['networkConfiguration']['awsvpcConfiguration']


def network_from(config: dict, live: dict) -> dict:
    """Copy the live network overriding what the config sets (non-empty keys only).

    Parameters
    ----------
    config : dict
        Launcher configuration (``ECS_SUBNETS``, ``ECS_SECURITY_GROUPS``, ``ECS_PUBLIC_IP``).
    live : dict
        ``awsvpcConfiguration`` of the running service.

    Returns
    -------
    dict
        The ``awsvpcConfiguration`` for ``run_task``.
    """
    network = dict(live)
    if config.get('ECS_SUBNETS'):
        network['subnets'] = [s.strip() for s in config['ECS_SUBNETS'].split(',') if s.strip()]
    if config.get('ECS_SECURITY_GROUPS'):
        network['securityGroups'] = [s.strip() for s in config['ECS_SECURITY_GROUPS'].split(',') if s.strip()]
    if config.get('ECS_PUBLIC_IP'):
        network['assignPublicIp'] = config['ECS_PUBLIC_IP']
    return network


def run_task_request(config: dict, network: dict, run_job: str) -> dict:
    """Build the keyword arguments of ``ecs.run_task`` for one job."""
    return {
        'cluster': config['ECS_CLUSTER'],
        'taskDefinition': config.get('ECS_TASK_FAMILY') or 'elections-model-job',
        'launchType': 'FARGATE',
        'count': 1,
        'networkConfiguration': {'awsvpcConfiguration': network},
        'overrides': {'containerOverrides': [
            {'name': CONTAINER, 'environment': [{'name': 'RUN_JOB', 'value': run_job}]}
        ]},
    }


def launch(ecs, request: dict) -> str:
    """Start the task and return its ARN.

    Raises
    ------
    LaunchError
        If AWS reports failures or returns no task.
    """
    response = aws_call('run_task', ecs.run_task, **request)
    failures = response.get('failures') or []
    if failures or not response.get('tasks'):
        first = failures[0] if failures else {}
        raise LaunchError(f"{first.get('reason', 'no task started')}: {first.get('detail')}")
    return response['tasks'][0]['taskArn']


def is_client_error(error: Exception) -> bool:
    """Tell whether an exception is a botocore ``ClientError`` (without importing botocore)."""
    return type(error).__name__ == 'ClientError' or hasattr(error, 'response')


def aws_call(name: str, function, **kwargs):
    """Call an AWS client method turning client errors into ``LaunchError``.

    Anything that is not an AWS client error is raised unchanged.
    """
    try:
        return function(**kwargs)
    except Exception as e:
        if not is_client_error(e):
            raise
        raise LaunchError(f'{name} failed: {e}')


def region_of(config: dict) -> Optional[str]:
    """Return ``AWS_REGION`` or None when empty or absent."""
    return config.get('AWS_REGION') or None


def task_id(arn: str) -> str:
    """Return the last segment of a task ARN."""
    return arn.rsplit('/', 1)[-1]


def log_stream(task_arn: str) -> str:
    """Return the CloudWatch stream name of the job container."""
    return f'{STREAM_PREFIX}/{CONTAINER}/{task_id(task_arn)}'


def dump_log(logs, log_group: str, stream: str, last: Optional[int], out) -> Optional[int]:
    """Print the log events newer than ``last`` and return the newest timestamp seen.

    A missing stream (task not started yet) is ignored.
    """
    kwargs = {'logGroupName': log_group, 'logStreamNames': [stream]}
    if last is not None:
        kwargs['startTime'] = last + 1
    while True:
        try:
            page = logs.filter_log_events(**kwargs)
        except logs.exceptions.ResourceNotFoundException:
            return last
        for event in page.get('events', []):
            say(out, event['message'].rstrip('\n'))
            if last is None or event['timestamp'] > last:
                last = event['timestamp']
        token = page.get('nextToken')
        if not token:
            return last
        kwargs['nextToken'] = token


def follow(ecs, logs, cluster: str, task_arn: str, log_group: str, poll: float = 15, sleep=None, out=None) -> int:
    """Stream the task log until it stops and return the container exit code.

    Parameters
    ----------
    ecs, logs : boto3 clients
        ECS and CloudWatch Logs clients.
    cluster : str
        Cluster name.
    task_arn : str
        ARN of the running task.
    log_group : str
        CloudWatch log group.
    poll : float
        Seconds between polls.
    sleep : callable, optional
        Defaults to ``time.sleep``.
    out : callable, optional
        Receives the text to write; defaults to ``sys.stdout.write``.

    Returns
    -------
    int
        Exit code of the first container, or 1 when AWS gives none.
    """
    sleep = sleep or time.sleep
    out = out or sys.stdout.write
    stream = log_stream(task_arn)
    last = None
    while True:
        last = dump_log(logs, log_group, stream, last, out)
        task = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])['tasks'][0]
        if task.get('lastStatus') == 'STOPPED':
            dump_log(logs, log_group, stream, last, out)
            say(out, f"stopped: {task.get('stoppedReason')}")
            code = (task.get('containers') or [{}])[0].get('exitCode')
            return 1 if code is None else code
        sleep(poll)


def derive_job_definition(web: dict, family: str) -> dict:
    """Derive the job task definition from the live web one.

    Keeps roles, platform and network mode, drops the port mappings, empties
    ``RUN_JOB``, uses 2 vCPU / 4 GB and the ``job`` log stream prefix.

    Parameters
    ----------
    web : dict
        ``describe_task_definition(...)['taskDefinition']`` of the web service.
    family : str
        Family name of the job definition.

    Returns
    -------
    dict
        Keyword arguments for ``register_task_definition``.
    """
    definition = {'family': family, 'cpu': '2048', 'memory': '4096'}
    for key in ('networkMode', 'requiresCompatibilities', 'runtimePlatform', 'taskRoleArn', 'executionRoleArn'):
        if key in web:
            definition[key] = web[key]
    containers = []
    for container in web['containerDefinitions']:
        derived = {key: container[key] for key in ('name', 'image', 'essential', 'environmentFiles') if key in container}
        derived['environment'] = [{'name': 'RUN_JOB', 'value': ''}]
        if 'logConfiguration' in container:
            log = json.loads(json.dumps(container['logConfiguration']))
            log.setdefault('options', {})['awslogs-stream-prefix'] = STREAM_PREFIX
            derived['logConfiguration'] = log
        containers.append(derived)
    definition['containerDefinitions'] = containers
    return definition


def register(ecs, cluster: str, service: str, family: str) -> str:
    """Register the job definition derived from the service's live one.

    Returns
    -------
    str
        ``'<family>:<revision>'``.
    """
    live = ecs.describe_services(cluster=cluster, services=[service])['services'][0]['taskDefinition']
    web = ecs.describe_task_definition(taskDefinition=live)['taskDefinition']
    definition = derive_job_definition(web, family)
    registered = aws_call('register_task_definition', ecs.register_task_definition, **definition)['taskDefinition']
    return f"{registered['family']}:{registered['revision']}"


def clients(region: Optional[str]) -> tuple:
    """Create the ECS and CloudWatch Logs clients (the only place importing boto3)."""
    import boto3
    return boto3.client('ecs', region_name=region), boto3.client('logs', region_name=region)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(prog='job.py', description='Run an elections-model job as an ECS Fargate task.')
    parser.add_argument('job', nargs='?', help='job name (load, compute, backtest, publish...)')
    parser.add_argument('params', nargs='?', help='job parameters as JSON text')
    parser.add_argument('--follow', action='store_true', help='stream the log and exit with the container exit code')
    parser.add_argument('--register', action='store_true', help='register the job task definition from the live web one')
    return parser


def default_env_file() -> str:
    """Return ``ECS_ENV_FILE`` or ``deploy/ecs.env``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get('ECS_ENV_FILE') or os.path.join(os.path.dirname(here), 'ecs.env')


def main(argv: list, ecs=None, logs=None, out=None, err=None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list
        Arguments without the program name.
    ecs, logs : boto3 clients, optional
        Injected clients (tests); created with ``clients`` when missing.
    out, err : callable, optional
        Receive the text to write; default to stdout and stderr.

    Returns
    -------
    int
        0 on success, 2 on usage or configuration errors, 1 when the task cannot
        start, or the container exit code with ``--follow``.
    """
    out = out or sys.stdout.write
    err = err or sys.stderr.write
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    try:
        if not args.job and not args.register:
            raise UsageError('job name is required')
        if args.register and args.job:
            raise UsageError('--register takes no job')
        config = load_config(default_env_file())
        if not config.get('ECS_CLUSTER') or not config.get('ECS_SERVICE'):
            raise ConfigError('ecs.env needs ECS_CLUSTER and ECS_SERVICE')
        if args.job:
            run_job = run_job_value(args.job, args.params)
    except UsageError as e:
        err(parser.format_usage())
        say(err, str(e))
        return 2
    except ConfigError as e:
        say(err, str(e))
        return 2
    if ecs is None or logs is None:
        made_ecs, made_logs = clients(region_of(config))
        ecs, logs = ecs or made_ecs, logs or made_logs
    cluster = config['ECS_CLUSTER']
    if args.register:
        family = config.get('ECS_TASK_FAMILY') or 'elections-model-job'
        try:
            say(out, f'registered {register(ecs, cluster, config["ECS_SERVICE"], family)}')
        except LaunchError as e:
            say(err, str(e))
            return 1
        return 0
    try:
        network = network_from(config, service_network(ecs, cluster, config['ECS_SERVICE']))
        arn = launch(ecs, run_task_request(config, network, run_job))
    except LaunchError as e:
        say(err, str(e))
        return 1
    log_group = config.get('ECS_LOG_GROUP') or '/ecs/elections-model'
    region = region_of(config)
    say(out, f'task {task_id(arn)}')
    say(out, f'aws logs tail {log_group} --log-stream-names {log_stream(arn)} --follow --since 1h')
    if region:
        say(out, f'https://{region}.console.aws.amazon.com/ecs/v2/clusters/{cluster}/tasks/{task_id(arn)}')
    if not args.follow:
        return 0
    return follow(ecs, logs, cluster, arn, log_group, out=out)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
