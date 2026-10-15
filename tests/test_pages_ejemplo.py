"""El ejemplo vivo del promedio en `/metodo/promedio` (fase 9): contexto, plantilla e isla de datos."""
import os
import re

import pytest

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages
from mtpy.lib.pages import metodo
from tests.test_api_asgi import call
from tests.test_pages import api, html, initial  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data

STATE = {'scope': 'es', 'mode': None, 'run': None}


def without_weights(tmp_path):
    """Quita la parte `weights` del run de pruebas: un run publicado antes de la fase 9."""
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'weights')))


def test_weights_example_picks_the_leader_and_its_slice(api):
    example = metodo.weights_example()
    assert example['name'] == 'PP' and example['fullname'] == fixture_data('meta')['parties'][0]['fullname']
    assert example['color'] == '#1d84ce' and example['date'] == '2026-10-08' and example['run'] == RUN
    assert example['published'] == fixture_data('headline')['run_at'][:10] == '2026-10-08'
    assert example['n_rows'] == 3 == example['stats']['nobs']
    assert 'weight_max' not in example and 'rating_max' not in example and 'sample_max' not in example
    assert [row['date'] for row in example['rows']] == ['2026-10-05', '2026-10-01', '2026-09-24']   # los más nuevos arriba
    assert example['line']['dates'][-1] == '2026-10-08' and len(example['kernel']['values']) == 15


def test_weights_example_is_none_without_the_part_or_the_leader(api, tmp_path, monkeypatch):
    weights = fixture_data('weights')
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None:
                        {**weights, 'parties': ['PSOE']} if name == 'weights' else fixture_data(name))
    assert metodo.weights_example() is None
    monkeypatch.undo()
    without_weights(tmp_path)
    assert metodo.weights_example() is None


def test_weights_example_lets_other_errors_through(api, monkeypatch):
    def boom(scope, run, name, mode=None):
        raise HttpError(503, 'bundle unavailable')
    monkeypatch.setattr(metodo, 'part', boom)
    with pytest.raises(HttpError) as error:
        metodo.weights_example()
    assert error.value.status == 503


def test_promedio_context_carries_the_example_and_its_island(api):
    context = pages.metodo_section_context(STATE, 'promedio')
    assert context['example']['name'] == 'PP' and 'example_shown' not in context and 'example_hidden' not in context
    assert context['example_csv'] == '/api/v1/forecast/es/weights?format=csv'
    assert set(context['initial']) == {'state', 'example'}
    island = context['initial']['example']
    assert set(island) == {'name', 'color', 'date', 'stats', 'rows', 'line', 'kernel'}
    assert set(island['rows'][0]) == {'date', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight',
                                      'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'}
    other = pages.metodo_section_context(STATE, 'escanos')
    assert 'example' not in other and 'example_csv' not in other and 'initial' not in other


def test_promedio_context_without_the_part_has_no_island(api, tmp_path):
    without_weights(tmp_path)
    context = pages.metodo_section_context(STATE, 'promedio')
    assert context['example'] is None and 'initial' not in context


def test_promedio_page_shows_the_example_block(api):
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' in page and 'Un ejemplo: el promedio del PP hoy' in page
    assert 'href="#ejemplo"' in page and page.index('id="banda"') < page.index('id="ejemplo"') < page.index('id="voto-estimado"')
    assert 'id="weights-table"' in page and 'id="weights-chart"' in page and 'id="weights-more"' not in page
    assert page.index('id="weights-chart"') < page.index('id="weights-list"') < page.index('id="weights-table"')
    assert 'class="chart chart-tall"' in page and '<details class="weights-list" id="weights-list">' in page   # plegado
    assert re.search(r'<summary[^>]*>\s*Ver los 3 sondeos', page)
    assert len(re.findall(r'<tr>\s*<td class="f l">', page)) == 3 and '<tr hidden>' not in page
    headers = re.findall(r'<th scope="col"[^>]*>([^<]+)</th>', page[page.index('id="weights-table"'):])
    assert headers == ['Fecha', 'Casa', 'Muestra', 'Predicción', 'Peso', 'F(kernel)', 'F(solape)', 'F(muestra)', 'F(rating)']
    rows = re.findall(r'<tr>\s*<td class="f l">([^<]+)</td>', page)
    assert rows == ['5 oct', '1 oct', '24 sept']                                   # los más nuevos arriba
    text = re.sub(r'\s+', ' ', page)
    assert ('Los sondeos que aportan peso en el promedio del PP publicado el 8 oct 2026 y la recta que se ajusta a ellos: '
            'lo que las secciones anteriores explican, con los datos a día de hoy.') in text
    assert ('El 8 oct 2026 el promedio del PP se establece en el 33,1%, con una banda del 32,4% al 33,8%. Se calcula a partir '
            'de 3 sondeos, que por sus pesos equivalen a unos 2,6 sondeos efectivos en los últimos 42 días, penalizando los '
            'más antiguos a través del kernel gaussiano aplicado.') in text
    assert '<b>GAD3</b> / ABC' in text and '>–<' in page                             # muestra ausente → guion
    assert 'class="fill w"' in page and 'width: 17.1%' in page                      # peso 0,512 sobre 3
    table = page[page.index('id="weights-table"'):page.index('</table>')]
    assert 'cell-shade' not in table and table.count('class="cell-bar"') == 3       # solo el peso lleva formato
    assert '<td>0,64</td>' in table and '<td>0,80</td>' in table                      # los factores, a secas
    assert 'Descargar los pesos de todos los partidos (CSV)' in page and '/api/v1/forecast/es/weights?format=csv' in page
    data = initial(body)
    assert data['example']['name'] == 'PP' and len(data['example']['rows']) == 3 and 'fullname' not in data['example']


def test_promedio_page_dates_the_example_by_its_publication(api, monkeypatch):
    headline = {**fixture_data('headline'), 'run_at': '2026-10-09T08:00:00Z'}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: headline if name == 'headline' else fixture_data(name))
    status, _, body = call(api, '/metodo/promedio')
    text = re.sub(r'\s+', ' ', html(body))
    card = text[text.index('id="ejemplo"'):text.index('id="voto-estimado"')]
    assert status == 200 and 'El 9 oct 2026 el promedio del PP se establece en el 33,1%' in card and 'publicado el 9 oct 2026' in card
    assert '8 oct 2026' not in card                       # la fecha de lectura (as_of) no se imprime en la tarjeta
    assert initial(body)['example']['date'] == '2026-10-08'    # pero el gráfico sigue centrado en ella


def test_promedio_page_lists_every_poll_without_a_load_more(api, monkeypatch):
    weights = fixture_data('weights')
    row = weights['rows'][0]
    weights['rows'] = [{**row, 'date': '2026-{:02d}-{:02d}'.format(month, day)} for month in (7, 8, 9) for day in range(1, 10)]
    weights['stats']['PP']['nobs'] = 27
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: weights if name == 'weights' else fixture_data(name))
    page = html(call(api, '/metodo/promedio')[2])
    assert len(re.findall(r'<tr>\s*<td class="f l">', page)) == 27 and '<tr hidden>' not in page
    assert 'id="weights-more"' not in page and 'Cargar más' not in page and 'Ver los 27 sondeos' in page


def test_promedio_page_without_the_part_renders_as_before(api, tmp_path):
    without_weights(tmp_path)
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' not in page and 'href="#ejemplo"' not in page
    assert 'initial-data' not in page and 'id="voto-estimado"' in page and 'id="banda"' in page
