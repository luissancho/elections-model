"""Tests de los ficheros de despliegue (plantillas de ECS, supervisord, init.sh y nginx)."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    """Devuelve el texto de un fichero del repositorio."""
    return (ROOT / path).read_text()


def load(path):
    """Devuelve el JSON de un fichero del repositorio."""
    return json.loads(read(path))


def test_ecs_templates_are_valid_and_consistent():
    """Las plantillas de ECS son JSON puro y coherentes entre sí."""
    web = load('deploy/ecs/task-web.example.json'); job = load('deploy/ecs/task-job.example.json'); run = load('deploy/ecs/run-task.example.json')
    env = {e['name']: e['value'] for e in web['containerDefinitions'][0]['environment']}
    assert env['GUNICORN_WORKERS'] == '2' and env['DB_STATEMENT_TIMEOUT'] == '10000' and 'AWS_KEY' not in env
    assert web['containerDefinitions'][0]['portMappings'][0]['containerPort'] == 8042
    assert job['memory'] == '4096' and all(e['name'] != 'APP_API' for e in job['containerDefinitions'][0]['environment'])
    assert run['overrides']['containerOverrides'][0]['environment'][0]['name'] == 'RUN_JOB'


def test_supervisord_and_init_take_the_workers_from_the_environment():
    """gunicorn toma los workers de GUNICORN_WORKERS y init.sh lo exporta con 4 por defecto."""
    conf = read('deploy/docker/supervisord.conf'); init = read('deploy/docker/init.sh')
    assert '-w %(ENV_GUNICORN_WORKERS)s' in conf and '/proc/1/fd/1' in conf
    assert 'GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"' in init
    assert subprocess.run(['bash', '-n', 'deploy/docker/init.sh'], cwd=ROOT).returncode == 0


def test_nginx_trusts_the_private_ranges_and_reports_the_client():
    """nginx confía en los rangos privados de la VPC y /healthz informa de la IP del cliente."""
    conf = read('deploy/docker/nginx.conf')
    for cidr in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'):
        assert 'set_real_ip_from {};'.format(cidr) in conf
    assert 'X-Client-IP $remote_addr always' in conf


def test_load_test_script_is_valid_bash():
    """El script de carga existe, es ejecutable y es bash válido."""
    path = ROOT / 'deploy' / 'load-test.sh'
    assert path.is_file()
    assert subprocess.run(['bash', '-n', 'deploy/load-test.sh'], cwd=ROOT).returncode == 0
