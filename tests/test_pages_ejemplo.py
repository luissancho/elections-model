"""Los ejemplos vivos de `/metodo/promedio`: el del promedio (fase 9: contexto, plantilla e isla de datos) y el de los
efectos de encuestadora (el CIS desde 2018: el último sondeo del CIS antes de cada elección analizada, por bloques)."""
import os
import re

import pytest

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages
from mtpy.lib.pages import metodo
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html, initial  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data, write_analysis, write_event

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


def test_weights_example_names_the_heaviest_poll(api):
    top = metodo.weights_example()['top']
    assert top == {'pollster': 'Sigma Dos', 'sponsor': top['sponsor'], 'date': '2026-10-01', 'weight': 1.138}


def test_cycle_counts_counts_the_polls_and_pollsters_of_the_cycle(api, monkeypatch):
    assert metodo.cycle_counts() == {'n_polls': 1, 'n_pollsters': 1}
    polls = {'parties': ['PP'], 'columns': ['date', 'pollster_id', 'pollster', 'weight'], 'results': [], 'polls': [
        {'date': '2026-10-01', 'pollster_id': 7, 'pollster': 'Sigma Dos', 'weight': 1.0},
        {'date': '2026-10-02', 'pollster_id': 7, 'pollster': 'Sigma Dos', 'weight': 0.0},   # descartado por solape
        {'date': '2026-10-03', 'pollster_id': 2, 'pollster': 'GAD3', 'weight': 0.5},
        {'date': '2023-07-23', 'pollster_id': None, 'pollster': None, 'weight': None}]}
    read = metodo.part
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: polls if name == 'polls' else read(scope, run, name, mode))
    assert metodo.cycle_counts() == {'n_polls': 3, 'n_pollsters': 2}
    assert pages.metodo_section_context(STATE, 'promedio')['cycle'] == {'n_polls': 3, 'n_pollsters': 2}


def test_cycle_counts_is_none_without_the_part_and_lets_other_errors_through(api, monkeypatch):
    def missing(scope, run, name, mode=None):
        raise HttpError(404, 'not found')
    monkeypatch.setattr(metodo, 'part', missing)
    assert metodo.cycle_counts() is None

    def boom(scope, run, name, mode=None):
        raise HttpError(503, 'bundle unavailable')
    monkeypatch.setattr(metodo, 'part', boom)
    with pytest.raises(HttpError):
        metodo.cycle_counts()


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
    table = page[page.index('id="weights-table"'):]
    table = table[:table.index('</table>')]
    assert len(re.findall(r'<tr>\s*<td class="f l">', table)) == 3 and '<tr hidden>' not in page
    headers = re.findall(r'<th scope="col"[^>]*>([^<]+)</th>', table)
    assert headers == ['Fecha', 'Casa', 'Entrevistas', 'Publicada', 'Corregida', 'Peso', 'Cercanía', 'Solape', 'Muestra',
                       'Rating']
    rows = re.findall(r'<tr>\s*<td class="f l">([^<]+)</td>', table)
    assert rows == ['5 oct', '1 oct', '24 sept']                                   # los más nuevos arriba
    assert '<td>32,4%</td>' in table and '<td>32,9%</td>' in table                  # publicada y corregida (GAD3)
    text = re.sub(r'\s+', ' ', page)
    assert ('En el último cálculo publicado, el del 8 oct 2026, el promedio del PP leído el 8 oct 2026 es del 33,1%, con '
            'una banda del 32,4% al 33,8%.') in text
    assert 'El día de lectura es posterior al del cálculo' not in text             # mismo día en la fixture
    assert 'Entran 3 sondeos, que por sus pesos equivalen a 2,6 sondeos independientes de igual peso' in text
    assert 'La campana de ese día tiene 42 días de anchura.' in text
    assert 'El sondeo que más pesa es el de Sigma Dos del 1 oct, con 1,14.' in text
    assert '<b>GAD3</b> / ABC' in text and '>–<' in page                             # muestra ausente → guion
    assert 'class="fill w"' in page and 'width: 17.1%' in page                      # peso 0,512 sobre 3
    assert 'cell-shade' not in table and table.count('class="cell-bar"') == 3       # solo el peso lleva formato
    assert '<td>0,64</td>' in table and '<td>0,80</td>' in table                      # los factores, a secas
    assert 'Descargar los pesos de todos los partidos (CSV)' in page and '/api/v1/forecast/es/weights?format=csv' in page
    data = initial(body)
    assert data['example']['name'] == 'PP' and len(data['example']['rows']) == 3 and 'fullname' not in data['example']
    # el texto va a la vista: la lista plegada justo bajo el gráfico, luego la lectura del gráfico y la de la tabla
    card = page[page.index('id="ejemplo"'):page.index('id="voto-estimado"')]
    assert '<details class="how"' not in card and 'Cómo leerlo' not in card
    assert re.search(r'id="weights-chart"[^>]*></div>\s*<details class="weights-list" id="weights-list">', card)
    assert card.index('</details>') < card.index('Cada punto es un sondeo') < card.index('La tabla lista los sondeos')
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_example_says_why_the_reading_day_is_later(api, monkeypatch):
    example = metodo.weights_example
    monkeypatch.setattr(metodo, 'weights_example', lambda: dict(example(), date='2026-10-18'))
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/promedio')[2]))
    assert 'leído el 18 oct 2026' in text and 'El día de lectura es posterior al del cálculo' in text


def test_promedio_regression_card_explains_step_by_step(api):
    page = html(call(api, '/metodo/promedio')[2])
    card = re.sub(r'\s+', ' ', page[page.index('id="regresion"'):page.index('id="banda"')])
    # each word of the name of the method, in order, then the titled blocks
    leads = ['Regresión.', 'Polinómica.', 'Local.', 'Ponderada.', 'Por kernel.']
    assert [card.index('<strong>' + lead + '</strong>') for lead in leads] == sorted(card.index('<strong>' + lead + '</strong>') for lead in leads)
    assert card.index('Por kernel.') < card.index('<math') < card.index('Por qué una recta y no una media')
    assert card.index('Por qué una recta y no una media') < card.index('id="anchura"') < card.index('id="consecuencias"')
    assert 'Sheather y Jones' in card and 'href="#peso"' in card
    assert '<a href="#ejemplo">El ejemplo de más abajo</a> dibuja la campana y la recta' in card


def test_promedio_regression_card_without_the_example_has_no_link_to_it(api, tmp_path):
    without_weights(tmp_path)
    page = html(call(api, '/metodo/promedio')[2])
    assert 'id="regresion"' in page and 'El ejemplo de más abajo' not in page   # sin el ejemplo del PP, sin enlace a él


def test_promedio_page_dates_the_example_by_its_publication(api, monkeypatch):
    headline = {**fixture_data('headline'), 'run_at': '2026-10-09T08:00:00Z'}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: headline if name == 'headline' else fixture_data(name))
    status, _, body = call(api, '/metodo/promedio')
    text = re.sub(r'\s+', ' ', html(body))
    card = text[text.index('id="ejemplo"'):text.index('id="voto-estimado"')]
    # the run and the reading day are two dates: the figure is the one of the reading day
    assert status == 200 and 'el del 9 oct 2026, el promedio del PP leído el 8 oct 2026 es del 33,1%' in card
    assert 'El día de lectura es posterior' not in card     # leído antes que el cálculo: nada que explicar
    assert initial(body)['example']['date'] == '2026-10-08'    # el gráfico sigue centrado en el día leído


def test_promedio_page_lists_every_poll_without_a_load_more(api, monkeypatch):
    weights = fixture_data('weights')
    row = weights['rows'][0]
    weights['rows'] = [{**row, 'date': '2026-{:02d}-{:02d}'.format(month, day)} for month in (7, 8, 9) for day in range(1, 10)]
    weights['stats']['PP']['nobs'] = 27
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: weights if name == 'weights' else fixture_data(name))
    page = html(call(api, '/metodo/promedio')[2])
    table = page[page.index('id="weights-table"'):]
    table = table[:table.index('</table>')]
    assert len(re.findall(r'<tr>\s*<td class="f l">', table)) == 27 and '<tr hidden>' not in page
    assert 'id="weights-more"' not in page and 'Cargar más' not in page and 'Ver los 27 sondeos' in page


def test_promedio_page_without_the_part_renders_as_before(api, tmp_path):
    without_weights(tmp_path)
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' not in page and 'href="#ejemplo"' not in page
    assert 'initial-data' not in page and 'id="voto-estimado"' in page and 'id="banda"' in page


# --- El ejemplo de los efectos de encuestadora: el CIS desde 2018 ---

# El último sondeo del CIS (y uno de GAD3, que no cuenta) en las dos elecciones publicadas de los tests.
HOUSE_ROWS = [
    {'event_date': '2023-07-23', 'event': '2023-23J', 'pollster': 'CIS', 'days': 11, 'sample_size': 29000, 'weight': 1.0,
     'error_avg': 1.5, 'bias_avg': 0.3, 'error_blocks': 2.9, 'bias_blocks': 1.5, 'bias': 0.7, 'Derecha': -2.84, 'Izquierda': 3.09},
    {'event_date': '2019-11-10', 'event': '2019-10N', 'pollster': 'CIS', 'days': 28, 'sample_size': 17000, 'weight': 1.0,
     'error_avg': 2.9, 'bias_avg': 0.1, 'error_blocks': 6.3, 'bias_blocks': 3.1, 'bias': 1.4, 'Derecha': -6.09, 'Izquierda': 6.44},
    {'event_date': '2019-11-10', 'event': '2019-10N', 'pollster': 'GAD3', 'days': 6, 'sample_size': 2000, 'weight': 1.0,
     'error_avg': 0.9, 'bias_avg': 0.2, 'error_blocks': 1.5, 'bias_blocks': 0.8, 'bias': 0.4, 'Derecha': 1.0, 'Izquierda': -0.5},
]
HOUSE_EFFECTS = [
    {'pollster': 'CIS', 'name': 'PP', 'n_events': 2, 'hist': -2.99, 'trend': -0.09, 'cycle': -5.9, 'n_cycle': 36},
    {'pollster': 'CIS', 'name': 'PSOE', 'n_events': 2, 'hist': 4.48, 'trend': 0.09, 'cycle': 4.29, 'n_cycle': 36},
    {'pollster': 'CIS', 'name': 'Derecha', 'n_events': 2, 'hist': -5.7, 'trend': -0.09, 'cycle': -6.7, 'n_cycle': 36},
    {'pollster': 'CIS', 'name': 'Izquierda', 'n_events': 2, 'hist': 4.48, 'trend': 0.09, 'cycle': 4.24, 'n_cycle': 36},
    {'pollster': 'GAD3', 'name': 'PP', 'n_events': 2, 'hist': 0.4, 'trend': 0.0, 'cycle': 0.5, 'n_cycle': 10},
]


def write_house_example(fs):
    """El análisis de `es` y sus dos elecciones publicadas, con el último sondeo del CIS en cada una (`HOUSE_ROWS`) y
    sus efectos del ciclo actual (`HOUSE_EFFECTS`)."""
    write_analysis(fs)
    write_event(fs)
    write_event(fs, date='2019-11-10', model=False)
    writer = bundle.BundleWriter(fs)
    errors = fixture_data(bundle.ANALYSIS_SCHEMAS['poll-errors'])
    errors['rows'] = errors['rows'] + HOUSE_ROWS
    writer.write_json(bundle.path_analysis('es', 'poll-errors'), bundle.ANALYSIS_SCHEMAS['poll-errors'], errors, 'es')
    effects = fixture_data(bundle.ANALYSIS_SCHEMAS['house-effects'])
    effects['rows'] = HOUSE_EFFECTS
    writer.write_json(bundle.path_analysis('es', 'house-effects'), bundle.ANALYSIS_SCHEMAS['house-effects'], effects, 'es')


@pytest.fixture
def house_site(api, fresh_app):
    write_house_example(fresh_app.fs)
    return api


def test_house_example_is_none_without_an_analysis(api):
    assert metodo.house_effects_example() is None
    context = pages.metodo_section_context(STATE, 'promedio')
    assert context['house_example'] is None and 'house' not in context['initial']
    page = html(call(api, '/metodo/promedio')[2])
    assert 'id="ejemplo-cis"' not in page and 'href="#ejemplo-cis"' not in page and 'id="house-chart"' not in page


def test_house_example_counts_the_errors_of_the_pollster_since_the_date(house_site, monkeypatch):
    example = metodo.house_effects_example()
    assert example['pollster'] == 'CIS' and example['since'] == '2018-09-01' and example['href'] == '/casas/cis'
    assert example['blocks'] == ['Derecha', 'Izquierda'] and example['left'] == 'Izquierda' and example['right'] == 'Derecha'
    assert [row['event_date'] for row in example['rows']] == ['2023-07-23', '2019-11-10']   # solo el CIS, las más nuevas arriba
    assert example['rows'][0]['name'] == 'Generales 2023' and example['rows'][0]['scope'] == 'es' and example['rows'][0]['days'] == 11
    assert example['n'] == 2 and example['first'] == '2019-11-10' and example['last'] == '2023-07-23'
    assert example['over'] == {'Izquierda': 2, 'Derecha': 0} and example['under'] == {'Izquierda': 0, 'Derecha': 2}
    assert example['cycle'] == {'PP': -5.9, 'PSOE': 4.29, 'Derecha': -6.7, 'Izquierda': 4.24} and example['n_cycle'] == 36
    # two elections out of two on the same side: a fair coin does it once in four
    assert example['chance'] == {'Izquierda': {'n': 2, 'k': 2, 'p': .25, 'odds': 4},
                                 'Derecha': {'n': 2, 'k': 2, 'p': .25, 'odds': 4}}
    monkeypatch.setitem(metodo.HOUSE_EXAMPLE, 'since', '2020-01-01')
    assert [row['event_date'] for row in metodo.house_effects_example()['rows']] == ['2023-07-23']
    monkeypatch.setitem(metodo.HOUSE_EXAMPLE, 'since', '2024-01-01')
    assert metodo.house_effects_example() is None


def test_sign_chance_is_the_binomial_tail_of_a_fair_coin():
    # 21 or more of 23 on the same side: (C(23,21) + C(23,22) + C(23,23)) / 2^23 = 277 / 8388608, one in 30283
    assert metodo.sign_chance(23, 21) == {'n': 23, 'k': 21, 'p': pytest.approx(277 / 2 ** 23), 'odds': 30000}
    assert metodo.sign_chance(10, 5)['odds'] == 1 and metodo.sign_chance(20, 15)['odds'] == 48
    assert metodo.sign_chance(0, 0) is None


def test_house_example_without_the_cycle_or_the_blocks(house_site, monkeypatch):
    read = metodo.analysis_part
    monkeypatch.setattr(metodo, 'analysis_part', lambda scope, name: None if name == 'house-effects' else read(scope, name))
    example = metodo.house_effects_example()
    assert example['n'] == 2 and example['cycle'] is None and example['n_cycle'] is None
    page = html(call(house_site, '/metodo/promedio')[2])
    assert 'En el ciclo actual de las generales' not in page and 'href="/casas/cis"' in page
    monkeypatch.undo()
    monkeypatch.setattr(metodo, 'error_blocks', lambda scopes, parts: ['Derecha', 'Centro'])
    assert metodo.house_effects_example() is None


def test_promedio_page_shows_the_house_example_card(house_site):
    status, _, body = call(house_site, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo-cis"' in page and 'Un ejemplo: el CIS desde 2018' in page
    assert 'href="#ejemplo-cis"' in page and page.index('href="#efectos-de-casa"') < page.index('href="#ejemplo-cis"') < page.index('href="#regresion"')
    assert page.index('id="efectos-de-casa"') < page.index('id="ejemplo-cis"') < page.index('id="regresion"')
    assert 'class="chart big" id="house-chart"' in page and '<details class="weights-list" id="house-list">' in page
    assert page.index('id="house-chart"') < page.index('id="house-list"') < page.index('id="house-table"')
    card = page[page.index('id="ejemplo-cis"'):page.index('id="regresion"')]
    text = re.sub(r'\s+', ' ', card)
    assert '<details class="how"' not in card and 'Cómo leerlo' not in card   # el texto va a la vista, bajo el gráfico
    # la lista plegada va pegada al gráfico; el texto que lo explica, debajo de las dos
    assert card.index('id="house-chart"') < card.index('id="house-list"') < card.index('Cada punto de la diana es el último sondeo del CIS')
    assert card.index('Cada punto de la diana') < card.index('El modelo no espera a la elección')
    assert 'José Félix Tezanos' in text and 'la «cocina»' in text and 'escaños' not in text
    assert ('Desde entonces el CIS ha publicado sondeos antes de 2 elecciones que hemos analizado, generales y autonómicas. '
            'En 2 de ellas su último sondeo dio a la Izquierda más voto del que obtuvo, y en 2, a la Derecha menos.') in text
    # with two elections chance still explains it: no claim of a house effect
    assert 'una entre 4' in text and 'no puede ser casualidad' not in text
    assert ('En el ciclo actual de las generales, con 36 sondeos del CIS, su efecto es de +4,2 puntos en la Izquierda y −6,7 en la '
            'Derecha (+4,3 en el PSOE y −5,9 en el PP): eso es lo que se descuenta hoy a cada uno de sus sondeos.') in text
    assert '<a href="/casas/cis">su perfil</a>' in text
    assert re.search(r'<summary[^>]*>\s*Ver las 2 elecciones', card)
    headers = re.findall(r'<th scope="col"[^>]*>([^<]+)</th>', card)
    assert headers == ['Elección', 'Fecha', 'Días antes', 'Izquierda', 'Derecha']
    rows = re.findall(r'<tr>\s*<td class="f l">([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>', card)
    assert [row[1:] for row in rows] == [('23 jul 2023', '11'), ('10 nov 2019', '28')] and rows[0][0] == 'Generales 2023'
    assert card.count('<td class="cell-shade"') == 4 and '>3,1</td>' in card and '>-2,8</td>' in card
    assert_public_wording(page, '/metodo/promedio')
    data = initial(body)
    assert set(data) == {'state', 'example', 'house'} and data['house']['blocks'] == ['Derecha', 'Izquierda']
    assert [row['event'] for row in data['house']['rows']] == ['2023-23J', '2019-10N']
    assert set(data['house']['rows'][0]) == {'event', 'event_date', 'days', 'scope', 'name', 'Derecha', 'Izquierda'}


def test_house_example_claims_the_effect_when_chance_cannot_explain_it(house_site, monkeypatch):
    house = metodo.house_effects_example
    monkeypatch.setattr(metodo, 'house_effects_example', lambda: dict(
        house(), n=23, over={'Izquierda': 21, 'Derecha': 1}, under={'Izquierda': 2, 'Derecha': 20},
        chance={'Izquierda': metodo.sign_chance(23, 21), 'Derecha': metodo.sign_chance(23, 20)}))
    page = html(call(house_site, '/metodo/promedio')[2])
    text = re.sub(r'\s+', ' ', page)
    assert ('La probabilidad de que una encuestadora sin inclinación se pasara con la Izquierda en 21 o más de 23 elecciones '
            'es de menos de una entre 30.000.') in text
    assert 'es un efecto de encuestadora sostenido' in text
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_page_shows_the_house_example_without_the_weights(house_site, tmp_path):
    without_weights(tmp_path)
    status, _, body = call(house_site, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo-cis"' in page and 'id="ejemplo"' not in page
    assert set(initial(body)) == {'state', 'house'}


def test_promedio_text_first_half(house_site):
    page = html(call(house_site, '/metodo/promedio')[2])
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', page[page.index('id="resumen"'):page.index('id="regresion"')]))
    # the headline is the title of the page as its h1; the «En resumen» card introduces the process (2026-10-22)
    assert '<h1 class="headline">¿Cómo promediamos las encuestas?</h1>' in page
    assert page.index('class="headline"') < page.index('class="section-nav toc"') < page.index('id="resumen"')
    assert ('El promedio de sondeos resume en una línea por partido lo que dicen todas las encuestas publicadas desde la '
            'elección anterior. Para trazarla seguimos cuatro pasos') in text
    assert 'con el ejemplo del CIS desde 2018' in text and 'de ahí parte la simulación de escaños' in text
    toc = page[page.index('<nav class="section-nav toc"'):]
    toc = toc[:toc.index('</nav>')]
    assert re.findall(r'href="#([a-z-]+)"', toc) == ['que-entra', 'peso', 'efectos-de-casa', 'ejemplo-cis', 'regresion',
                                                     'banda', 'ejemplo', 'voto-estimado', 'notas']
    peso = page[page.index('id="peso"'):page.index('id="efectos-de-casa"')]
    ids = [peso.index('id="{}"'.format(i)) for i in ('muestra', 'dispersion', 'sobrerrepresentacion', 'rating', 'cuentas')]
    assert ids == sorted(ids)
    for phrase in ('Por qué una raíz y no otra cosa', 'Cuando falta la muestra', 'de los 2.935 sondeos de generales',
                   'Por qué el rebaño se mide de otra manera', 'forman un grupo que pesa como un único sondeo',
                   'Por qué un logaritmo', 'pesa 3,7 veces más que el barómetro del CIS', 'Por qué fiarnos del consenso',
                   'Por qué los efectos suman cero', 'En las generales de este ciclo entra 1 sondeo de 1 encuestadora.'):
        assert phrase in text, phrase
    for phrase in ('El problema.', 'La idea.', 'Los campos', 'Un caso real', 'backtest'):
        assert phrase not in text, phrase
    calls = re.findall(r'id="llamada-([a-z]+)"', page)
    assert calls[:7] == ['tope', 'diseno', 'umbrales', 'ventana', 'suelo', 'comun', 'convenciones']
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page)[:7] == ['1', '2', '3', '4', '5', '6', '7']
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_text_without_a_run_has_no_cycle_figures(api, monkeypatch):
    monkeypatch.setattr(metodo, 'cycle_counts', lambda: None)
    status, _, body = call(api, '/metodo/promedio')
    assert status == 200 and 'En las generales de este ciclo' not in html(body)


def test_promedio_text_second_half(house_site):
    page = html(call(house_site, '/metodo/promedio')[2])
    regresion = page[page.index('id="regresion"'):page.index('id="banda"')]
    banda = page[page.index('id="banda"'):page.index('id="voto-estimado"')]
    assert regresion.index('id="anchura"') < regresion.index('id="consecuencias"')
    assert regresion.count('<math display="block"') >= 4 and banda.count('<math display="block"') >= 2
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', page[page.index('id="regresion"'):]))
    for phrase in ('regresión polinómica local ponderada por kernel', 'Por qué una recta y no una media',
                   '1,70 puntos por partido', 'Por qué agrupamos por encuestadora', 'Por qué una t de Student',
                   '93 %', '54 %'):
        assert phrase in text, phrase
    for phrase in ('El problema.', 'La idea.', 'Los campos', 'Un caso real', 'los de hace tres, casi nada'):
        assert phrase not in text, phrase
    keys = ['tope', 'diseno', 'umbrales', 'ventana', 'suelo', 'comun', 'convenciones', 'referencias', 'corte', 'isj',
            'abramson', 'cobertura', 'limites', 'escala']
    assert page.index('id="voto-estimado"') < page.index('id="notas"') < page.index('id="closing"')
    assert re.findall(r'<li id="nota-([a-z]+)">', page) == keys
    assert re.findall(r'id="llamada-([a-z]+)"', page) == keys
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page) == [str(n) for n in range(1, 15)]
    for target in re.findall(r'href="#(nota-[a-z]+)"', page):
        assert 'id="{}"'.format(target) in page, target
    # no byline after the notes: the footer signs the page (2026-10-22)
    assert 'Por Luis Sancho' not in text and 'class="byline"' not in page
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_text_dates_the_september_figures_and_gets_kish_right(api):
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', html(call(api, '/metodo/promedio')[2])))
    assert 'si uno pesa por los otros nueve juntos, informan como menos de cuatro' in text   # 18² / (81 + 9) = 3,6
    assert 'En el PP, en septiembre, agrupar por encuestadora' in text
    assert 'en septiembre, remuestreando por encuestadoras' in text
    assert 'en septiembre, las inclinaciones de las encuestadoras en el voto del PP' in text
    assert 'En el cálculo del 20 de octubre, solo 6 de los 328 sondeos' in text
    assert 'del mismo orden que lo que medimos directamente' in text
    assert 'Vamos de la última a la primera' not in text and 'Las vemos una a una' in text


def test_promedio_page_without_a_run_keeps_the_notes_and_drops_the_examples(api, monkeypatch):
    def missing(*args, **kwargs):
        raise HttpError(404, 'not found')
    monkeypatch.setattr(metodo, 'part', missing)
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' not in page and 'En las generales de este ciclo' not in page
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page) == [str(n) for n in range(1, 15)]
    for target in re.findall(r'href="#((?:nota|llamada)-[a-z]+)"', page):
        assert 'id="{}"'.format(target) in page, target

