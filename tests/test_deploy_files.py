"""Tests de los ficheros de despliegue (plantillas de ECS, supervisord, init.sh y nginx) y de los activos de marca."""
import json
import re
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    """Devuelve el texto de un fichero del repositorio."""
    return (ROOT / path).read_text()


def load(path):
    """Devuelve el JSON de un fichero del repositorio."""
    return json.loads(read(path))


def test_ecs_templates_describe_the_live_infrastructure():
    """Las plantillas son JSON puro y reflejan la infraestructura real: configuración por fichero de entorno en S3,
    contenedor `Main`, definición de jobs derivada de la web (2 vCPU, 4 GB, sin puertos, prefijo `job`)."""
    web = load('deploy/ecs/task-web.example.json'); job = load('deploy/ecs/task-job.example.json'); run = load('deploy/ecs/run-task.example.json')
    for template in (web, job):
        container = template['containerDefinitions'][0]
        assert container['name'] == 'Main' and 'secrets' not in container
        assert container['environmentFiles'] == [{'value': 'arn:aws:s3:::<bucket>/elections.env', 'type': 's3'}]
        assert container['image'].endswith('.amazonaws.com/elections-model:latest')
        assert container['logConfiguration']['options']['awslogs-group'] == '/ecs/elections-model'
    assert 'environment' not in web['containerDefinitions'][0] and web['containerDefinitions'][0]['portMappings'][0]['containerPort'] == 8042
    assert web['family'] == 'elections-model-cluster-elections-model-service' and web['cpu'] == '1024'
    assert job['family'] == 'elections-model-job' and job['cpu'] == '2048' and job['memory'] == '4096'
    assert 'portMappings' not in job['containerDefinitions'][0]
    assert job['containerDefinitions'][0]['environment'] == [{'name': 'RUN_JOB', 'value': ''},
                                                             {'name': 'DB_STATEMENT_TIMEOUT', 'value': ''}]
    assert job['containerDefinitions'][0]['logConfiguration']['options']['awslogs-stream-prefix'] == 'job'
    assert run['cluster'] == 'elections-model-cluster' and run['taskDefinition'] == 'elections-model-job'
    assert run['networkConfiguration']['awsvpcConfiguration']['assignPublicIp'] == 'ENABLED'
    override = run['overrides']['containerOverrides'][0]
    assert override['name'] == 'Main' and override['environment'][0]['name'] == 'RUN_JOB'
    assert override['environment'][0]['value'].startswith('publish {"what":["forecast","analysis"]')


def test_dockerfile_bakes_the_commit():
    """La imagen lleva el commit como variable de entorno (no hay `.git` dentro)."""
    dockerfile = read('Dockerfile')
    assert 'ARG GIT_COMMIT=' in dockerfile and 'ENV GIT_COMMIT=${GIT_COMMIT}' in dockerfile


def test_supervisord_and_init_take_the_workers_from_the_environment():
    """gunicorn toma los workers de GUNICORN_WORKERS y init.sh lo exporta con 4 por defecto."""
    conf = read('deploy/docker/supervisord.conf'); init = read('deploy/docker/init.sh')
    assert '-w %(ENV_GUNICORN_WORKERS)s' in conf and '/proc/1/fd/1' in conf
    assert 'GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"' in init
    assert subprocess.run(['bash', '-n', 'deploy/docker/init.sh'], cwd=ROOT).returncode == 0


def fake_python(tmp_path, code=0):
    """Un `python` falso en el PATH que anota sus argumentos y termina con `code`."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'python'
    script.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "{}"\nexit {}\n'.format(tmp_path / 'args', code))
    script.chmod(0o755)
    return bin_dir


def run_init(tmp_path, env):
    """Ejecuta `init.sh` con el `python` falso delante en el PATH y solo las variables dadas."""
    import os
    bin_dir = fake_python(tmp_path, code=3)
    full = {'PATH': '{}:{}'.format(bin_dir, os.environ['PATH']), **env}
    return subprocess.run(['bash', 'deploy/docker/init.sh'], cwd=ROOT, env=full, capture_output=True, text=True)


def test_init_execs_the_job_with_the_json_as_one_argument_and_returns_its_exit_code(tmp_path):
    """El nombre del job es la primera palabra, el resto un solo argumento (JSON con espacios) y el contenedor
    termina con el código de salida del job."""
    init = read('deploy/docker/init.sh')
    assert 'exec python /app/job.py "$job_name" ${job_params:+"$job_params"}' in init
    done = run_init(tmp_path, {'RUN_JOB': 'publish {"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}'})
    assert done.returncode == 3
    assert (tmp_path / 'args').read_text().splitlines() == ['/app/job.py', 'publish', '{"what":["manifest"],"freeze":{"active":true,"message":"Veda electoral"}}']


def test_init_refuses_an_idle_container(tmp_path):
    """Sin `RUN_JOB` ni servicios el contenedor termina con 1 en vez de quedarse ocioso."""
    done = run_init(tmp_path, {})
    assert done.returncode == 1 and 'Nothing to run' in done.stderr
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


def test_brand_assets_match_the_templates_and_nginx_serves_the_favicon():
    """Los iconos y el logotipo que enlazan las plantillas existen en `web/dist/img`, el ICO trae 16, 32 y 48 px, el
    logotipo SVG declara el nombre y el tamaño que repite `_brand.html`, y nginx sirve `/favicon.ico`."""
    img = ROOT / 'web' / 'dist' / 'img'
    linked = set(re.findall(r'/dist/img/([\w.-]+)', read('web/templates/_icons.html') + read('web/templates/_brand.html')))
    assert linked == {'logo.svg', 'favicon.svg', 'favicon.ico', 'apple-touch-icon.png'}
    for name in linked:
        assert (img / name).is_file(), name
    logo = read('web/dist/img/logo.svg')
    assert logo.startswith('<svg xmlns="http://www.w3.org/2000/svg"') and '<title id="title">Demoscopio</title>' in logo
    width, height = re.search(r' width="([\d.]+)" height="([\d.]+)"', logo).groups()
    assert 'width="{}" height="{}"'.format(round(float(width)), round(float(height))) in read('web/templates/_brand.html')
    assert read('web/dist/img/favicon.svg').startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">')
    with Image.open(img / 'favicon.ico') as ico:
        assert ico.info['sizes'] == {(16, 16), (32, 32), (48, 48)}
    with Image.open(img / 'apple-touch-icon.png') as png:
        assert png.size == (180, 180)
    conf = read('deploy/docker/nginx.conf')
    assert 'location = /favicon.ico {' in conf and 'alias /app/web/dist/img/favicon.ico;' in conf
    assert conf.index('location = /favicon.ico') < conf.index('location / {')


def test_site_fonts_are_vendored_and_licensed():
    """Las dos fuentes variables que `site.css` declara existen en `web/dist/vendor/fonts`, pesan lo que un
    subconjunto latino, están en `LICENSE` y las antiguas ya no se referencian."""
    css = read('web/dist/css/site.css')
    faces = re.findall(r"@font-face\s*{[^}]*font-family:\s*'([^']+)'[^}]*font-weight:\s*([\d ]+);[^}]*url\('/dist/vendor/fonts/([\w.-]+)'\)", css, re.S)
    assert {(family, weight) for family, weight, _ in faces} == {('DM Sans', '400 700'), ('Outfit', '700 800')}
    licence = read('web/dist/vendor/fonts/LICENSE')
    for _, _, name in faces:
        path = ROOT / 'web' / 'dist' / 'vendor' / 'fonts' / name
        assert path.is_file() and path.stat().st_size < 70_000, name
        assert name in licence, name
    assert 'Source Serif' not in css and 'Source Sans' not in css
    assert 'Indian Type Foundry' in licence and 'Outfit Project Authors' in licence
    assert "--font-body: 'DM Sans'" in css and "--font-head: 'Outfit'" in css
    assert 'font-variant-numeric: tabular-nums' in css


def test_wordmark_is_set_in_outfit_and_the_old_fonts_are_gone():
    """El wordmark se genera con Outfit (la fuente variable del sitio, instanciada al peso 800), las fuentes Source
    ya no están y el favicon no cambia."""
    source = read('web/brand/build.py')
    assert "'Outfit.woff2'" in source and 'WEIGHT = 800' in source and 'SourceSerif4' not in source
    fonts = ROOT / 'web' / 'dist' / 'vendor' / 'fonts'
    assert not list(fonts.glob('Source*')) and 'Source' not in read('web/dist/vendor/fonts/LICENSE')
    logo = read('web/dist/img/logo.svg')
    assert logo.count('<path d=') == 9  # D e m o s c o p i: nueve glifos, la lente aparte
    assert read('web/dist/img/favicon.svg').startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">')
