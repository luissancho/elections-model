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
    # El clúster y la definición de la web se llaman `elections-model` (Luis, 2026-10-14); la del job sigue siendo
    # `elections-job` y la imagen de ECR `elections-web`
    assert web['family'] == 'elections-model' and run['cluster'] == 'elections-model'
    assert job['family'] == 'elections-job' and run['taskDefinition'] == 'elections-job'
    for template in (web, job):
        adapter = {e['name']: e['value'] for e in template['containerDefinitions'][0]['environment']}['DB_ADAPTER']
        # `mtpy.run()` resuelve el adaptador con `getattr(dal, DB_ADAPTER)`: debe nombrar un módulo de `mtpy/core/dal/`
        assert (ROOT / 'mtpy' / 'core' / 'dal' / '{}.py'.format(adapter)).is_file(), adapter
    job_env = {e['name']: e['value'] for e in job['containerDefinitions'][0]['environment']}
    assert job_env['GIT_COMMIT'] == '<git-sha>' and 'GIT_COMMIT' not in env


def test_supervisord_and_init_take_the_workers_from_the_environment():
    """gunicorn toma los workers de GUNICORN_WORKERS y init.sh lo exporta con 4 por defecto."""
    conf = read('deploy/docker/supervisord.conf'); init = read('deploy/docker/init.sh')
    assert '-w %(ENV_GUNICORN_WORKERS)s' in conf and '/proc/1/fd/1' in conf
    assert 'GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"' in init
    assert subprocess.run(['bash', '-n', 'deploy/docker/init.sh'], cwd=ROOT).returncode == 0


def test_init_passes_the_run_job_parameters_as_one_argument():
    """`init.sh` separa el nombre del job (primera palabra) y pasa el resto como un solo argumento (JSON con
    espacios)."""
    init = read('deploy/docker/init.sh')
    assert 'read -r job_name job_params <<< "$RUN_JOB"' in init
    assert 'python /app/job.py "$job_name" ${job_params:+"$job_params"}' in init
    assert 'python /app/job.py $RUN_JOB' not in init
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
