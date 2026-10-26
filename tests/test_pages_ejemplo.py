"""Los ejemplos vivos de `/metodo/promedio`: el del promedio (fase 9: contexto, plantilla e isla de datos) y el de los
efectos de casa (el CIS desde 2018: el último sondeo del CIS antes de cada elección analizada, por bloques)."""
import os
import re

import pytest

from mtpy.core.api import HttpError
from mtpy.lib import bundle, pages, webapi
from mtpy.lib.pages import metodo
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html, initial  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data, write_analysis, write_event, write_study

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
    assert status == 200 and 'id="ejemplo"' in page and 'Un cálculo completo, con datos publicados' in page
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
    assert ('En el cálculo publicado el 8 oct 2026, el promedio del PP leído en el 8 oct 2026 era del 33,1%, con '
            'una banda del 32,4% al 33,8%.') in text
    assert 'Las fechas describen dos cosas distintas' not in text              # mismo día en la fixture
    assert 'En ese ajuste participaban 3 sondeos y la campana tenía una anchura de 42 días.' in text
    assert 'El tamaño efectivo de los pesos era 2,6.' in text and 'No significa que tuviéramos exactamente 2,6 sondeos independientes' in text
    assert 'El sondeo que más pesa es el de Sigma Dos del 1 oct, con 1,14.' in text
    assert '<b>GAD3</b> / ABC' in text and '>–<' in page                             # muestra ausente → guion
    assert 'class="fill w"' in page and 'width: 17.1%' in page                      # peso 0,512 sobre 3
    assert 'cell-shade' not in table and table.count('class="cell-bar"') == 3       # sólo el peso lleva formato
    assert '<td>0,64</td>' in table and '<td>0,80</td>' in table                      # los factores, a secas
    assert 'Descargar los pesos de todos los partidos (CSV)' in page and '/api/v1/forecast/es/weights?format=csv' in page
    data = initial(body)
    assert data['example']['name'] == 'PP' and len(data['example']['rows']) == 3 and 'fullname' not in data['example']
    # el texto va a la vista: el conmutador Publicada/Corregida (oculto sin JS) y la lista plegada junto al gráfico, luego
    # la lectura del gráfico y la de la tabla
    card = page[page.index('id="ejemplo"'):page.index('id="voto-estimado"')]
    assert '<details class="how"' not in card and 'Cómo leerlo' not in card
    assert '<fieldset id="weights-field" class="seg" hidden>' in card and 'name="weights-field" value="pct_adj"' in card
    assert card.index('id="weights-field"') < card.index('id="weights-chart"')
    assert re.search(r'id="weights-chart"[^>]*></div>\s*<details class="weights-list" id="weights-list">', card)
    assert card.index('</details>') < card.index('Cada punto es un sondeo') < card.index('La tabla recoge las fechas')
    # the two figures of a poll are explained before the chart, with the CIS as the example, in plain words
    assert card.index('Dos cifras para cada sondeo') < card.index('id="weights-field"') < card.index('id="weights-chart"')
    assert 'su cifra corregida es la que habría publicado si midiera como la media de las encuestadoras' in text
    assert 'La corrección no acerca los sondeos al resultado electoral' in text and '<legend>Cifra de los puntos</legend>' in card
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_example_says_why_the_reading_day_is_later(api, monkeypatch):
    example = metodo.weights_example
    monkeypatch.setattr(metodo, 'weights_example', lambda: dict(example(), date='2026-10-18'))
    text = re.sub(r'\s+', ' ', html(call(api, '/metodo/promedio')[2]))
    assert 'leído en el 18 oct 2026' in text
    assert 'Las fechas describen dos cosas distintas: el modelo se ejecutó el 8 oct 2026 y prolongó la estimación hasta el 18 oct 2026' in text


def test_promedio_regression_card_explains_step_by_step(api):
    page = html(call(api, '/metodo/promedio')[2])
    card = re.sub(r'\s+', ' ', page[page.index('id="regresion"'):page.index('id="banda"')])
    text = re.sub(r'<[^>]+>', '', card)
    # the blocks of the expert's text, in order: the line, the bell, the fit, why not a mean, the bandwidth, the past
    heads = ['Una recta para cada día', 'La campana que da más peso a lo cercano', 'Cómo elegimos la recta que mejor encaja',
             'Por qué una recta ayuda más que una media', 'Cuánto debe abarcar la campana', 'Por qué puede cambiar el pasado']
    assert [card.index('<h3') for _ in heads] and [card.index(head) for head in heads] == sorted(card.index(head) for head in heads)
    assert card.count('<math display="block"') == 6   # ŷ, K(u), u, el factor temporal, ω y el error del ajuste
    assert 'colocamos el día que queremos estimar en el cero del eje horizontal' in text and 'b = 0,02 representa una subida local' in text
    assert 'La constante 1/√(2π) se cancela al dividir.' in text
    # the worked example of three polls and the honest reading of the line
    assert 'Su media es el 29%, pero las tres describen una subida de un punto cada diez días' in text
    assert 'No supone que el último sondeo tenga razón ni que la subida vaya a continuar' in text
    assert card.index('id="anchura"') < card.index('id="consecuencias"') and '81 días' in text and '54%' in text
    assert 'El ejemplo de más abajo' not in card   # the regression card no longer links to the example


def test_promedio_regression_card_without_the_example_has_no_link_to_it(api, tmp_path):
    without_weights(tmp_path)
    page = html(call(api, '/metodo/promedio')[2])
    assert 'id="regresion"' in page and 'href="#ejemplo"' not in page   # sin el ejemplo del PP, sin enlaces a él


def test_promedio_page_dates_the_example_by_its_publication(api, monkeypatch):
    headline = {**fixture_data('headline'), 'run_at': '2026-10-09T08:00:00Z'}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: headline if name == 'headline' else fixture_data(name))
    status, _, body = call(api, '/metodo/promedio')
    text = re.sub(r'\s+', ' ', html(body))
    card = text[text.index('id="ejemplo"'):text.index('id="voto-estimado"')]
    # the run and the reading day are two dates: the figure is the one of the reading day
    assert status == 200 and 'En el cálculo publicado el 9 oct 2026, el promedio del PP leído en el 8 oct 2026 era del 33,1%' in card
    assert 'Las fechas describen dos cosas distintas' not in card     # leído antes que el cálculo: nada que explicar
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


# --- El ejemplo de los efectos de casa: el CIS desde 2018 ---

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


def test_method_card_links_the_study_only_when_published(house_site, fresh_app):
    page = html(call(house_site, '/metodo/promedio')[2])
    assert '/analisis/sesgo-cis' not in page
    write_study(fresh_app.fs, study='sesgo-cis'); webapi.site().manifest_cache_clear()
    page = html(call(house_site, '/metodo/promedio')[2])
    assert 'El análisis completo, con el CIS de antes de 2018, las demás casas y 2023, está en <a href="/analisis/sesgo-cis">' in page


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
    assert [row['event_date'] for row in example['rows']] == ['2023-07-23', '2019-11-10']   # sólo el CIS, las más nuevas arriba
    assert example['rows'][0]['name'] == 'Generales 2023' and example['rows'][0]['scope'] == 'es' and example['rows'][0]['days'] == 11
    assert example['n'] == 2 and example['first'] == '2019-11-10' and example['last'] == '2023-07-23'
    assert example['over'] == {'Izquierda': 2, 'Derecha': 0} and example['under'] == {'Izquierda': 0, 'Derecha': 2}
    assert example['cycle'] == {'PP': -5.9, 'PSOE': 4.29, 'Derecha': -6.7, 'Izquierda': 4.24} and example['n_cycle'] == 36
    # two elections out of two on the same side: a fair coin does it once in four
    coin = {'n': 2, 'k': 2, 'ways': 1, 'total': 4, 'p': .25, 'odds_exact': 4, 'odds': 4}
    assert example['chance'] == {'Izquierda': coin, 'Derecha': coin}
    monkeypatch.setitem(metodo.HOUSE_EXAMPLE, 'since', '2020-01-01')
    assert [row['event_date'] for row in metodo.house_effects_example()['rows']] == ['2023-07-23']
    monkeypatch.setitem(metodo.HOUSE_EXAMPLE, 'since', '2024-01-01')
    assert metodo.house_effects_example() is None


def test_sign_chance_is_the_binomial_tail_of_a_fair_coin():
    # 21 or more of 23 on the same side: (C(23,21) + C(23,22) + C(23,23)) / 2^23 = 277 / 8388608, one in 30284
    assert metodo.sign_chance(23, 21) == {'n': 23, 'k': 21, 'ways': 277, 'total': 8388608, 'p': pytest.approx(277 / 2 ** 23),
                                          'odds_exact': 30284, 'odds': 30000}
    assert metodo.sign_chance(10, 5)['odds'] == 1 and metodo.sign_chance(20, 15)['odds'] == 48
    # the figures of the CIS example of the expert review (38 or more of 41): 11.522 of 2⁴¹, one in 190.854.301
    assert metodo.sign_chance(41, 38)['ways'] == 11522 and metodo.sign_chance(41, 38)['odds_exact'] == 190854301
    assert metodo.sign_chance(41, 38)['odds'] == 190000000
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
    assert status == 200 and 'id="ejemplo-cis"' in page and 'El CIS: 2 de 2 elecciones en la misma dirección' in page
    assert 'href="#ejemplo-cis"' in page and page.index('href="#efectos-de-casa"') < page.index('href="#ejemplo-cis"') < page.index('href="#regresion"')
    assert page.index('id="efectos-de-casa"') < page.index('id="ejemplo-cis"') < page.index('id="regresion"')
    assert 'class="chart big" id="house-chart"' in page and '<details class="weights-list" id="house-list">' in page
    assert page.index('id="house-chart"') < page.index('id="house-list"') < page.index('id="house-table"')
    card = page[page.index('id="ejemplo-cis"'):page.index('id="regresion"')]
    text = re.sub(r'\s+', ' ', card)
    assert '<details class="how"' not in card and 'Cómo leerlo' not in card   # el texto va a la vista, bajo el gráfico
    # la lista plegada va pegada al gráfico; la lectura de la diana y la cuenta de las monedas, debajo de las dos
    assert card.index('id="house-chart"') < card.index('id="house-list"') < card.index('La diana muestra los dos errores')
    assert card.index('La diana muestra los dos errores') < card.index('Sería como lanzar una moneda equilibrada 2 veces')
    assert card.index('Sería como lanzar una moneda') < card.index('Esta comprobación y el ajuste del promedio tienen funciones distintas')
    assert 'cuando cambió su estimación de voto' in text and 'escaños' not in text
    assert ('hemos analizado 2 elecciones generales y autonómicas con sondeos suyos. En 2 de ellas su último sondeo dio al '
            'bloque de izquierdas más voto del que obtuvo. En 2, dio al bloque de derechas menos del que obtuvo.') in text
    # with two elections chance still explains it: no bold claim
    assert 'es de una entre 4: con tan pocas elecciones, el azar todavía podría explicarlo' in text
    assert '<strong>Bajo esos supuestos' not in text
    # the caveats of the coin count: dependence, no intentionality, the two blocks are one test, the CIS was picked after looking
    assert 'ni mide la probabilidad de imparcialidad o de intencionalidad del CIS' in text and 'son una prueba y no dos' in text
    assert 'elegimos el CIS después de mirar los datos, porque es el caso más llamativo' in text
    assert ('En el ciclo actual de las generales, con 36 sondeos del CIS, su efecto frente al consenso es de +4,2 puntos en la '
            'Izquierda y −6,7 en la Derecha (+4,3 en el PSOE y −5,9 en el PP). Eso es lo que se descuenta hoy a cada uno de sus '
            'sondeos.') in text
    assert 'El recuento de 2 sobreestimaciones no se utiliza como una penalización adicional' in text
    assert '<a href="/casas/cis">su perfil</a>' in text
    # the note of the binomial count exists only with the example, numbered after the one of the effects
    assert re.findall(r'id="llamada-([a-z]+)"', page) == ['muestra', 'pesos', 'efectos', 'binomial', 'regresion', 'intervalos']
    notes = re.sub(r'\s+', ' ', page[page.index('id="notas"'):])
    assert 'Binomial</mtext><mo>(</mo><mn>2</mn><mo>,</mo><mn>0,5</mn>' in notes and '<mfrac><mn>1</mn><mn>4</mn></mfrac>' in notes
    assert 'Equivale a una entre 4. La probabilidad bilateral' in notes
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
    assert ('<strong>Bajo esos supuestos, la probabilidad de sobreestimar a la Izquierda en 21 o más de las 23 elecciones es '
            'inferior a una entre 30.000.</strong>') in text
    # the note shows the count: 277 favourable outcomes of 2^23, one in 30.284, floored to two significant figures
    notes = text[text.index('id="notas"'):]
    assert '<mfrac><mn>277</mn><mn>8.388.608</mn></mfrac>' in notes
    assert 'Equivale a una entre 30.284, de donde procede «inferior a una entre 30.000»' in notes
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
    # the opening card is the expert's: the two-polls question, the four steps, the band is not the vote, the curve
    assert 'Una encuesta sitúa a un partido en el 31%; otra, pocos días después, en el 33%.' in text
    assert 'El proceso tiene cuatro pasos' in text and 'varias encuestas pueden coincidir y equivocarse juntas' in text
    assert 'De los sondeos a una curva' in text
    # the polls are placed on the last day of their fieldwork; the cut looks at the fieldwork, not at the publication;
    # the open online panels are defined, the professional online polls enter
    assert 'Situamos cada encuesta en el último día de su trabajo de campo' in text
    assert 'La regla mira al campo, no a la publicación' in text and 'como las de 40dB, entra como las demás' in text
    assert 'Dejamos fuera los paneles online abiertos' in text and 'Esa etiqueta debe precisarse' not in text
    toc = page[page.index('<nav class="section-nav toc"'):]
    toc = toc[:toc.index('</nav>')]
    assert re.findall(r'href="#([a-z-]+)"', toc) == ['que-entra', 'peso', 'efectos-de-casa', 'ejemplo-cis', 'regresion',
                                                     'banda', 'ejemplo', 'voto-estimado', 'comprobacion', 'limites', 'notas']
    peso = page[page.index('id="peso"'):page.index('id="efectos-de-casa"')]
    ids = [peso.index('id="{}"'.format(i)) for i in ('muestra', 'dispersion', 'sobrerrepresentacion', 'rating', 'cuentas')]
    assert ids == sorted(ids)
    for phrase in ('más entrevistas ayudan, pero no lo resuelven todo',
                   'Una encuesta con cuatro veces más entrevistas recibe el doble de peso por este factor',
                   '558 de los 2.935 sondeos desde 1977, el 19%',
                   'La comparación detecta un comportamiento; no demuestra por sí sola cuál es su causa',
                   'Es una equivalencia de ponderación', 'evitar contar varias veces lo mismo', 'sólo 6 de los 328 sondeos del ciclo',
                   'qué nos enseña su historial', 'Es una decisión de diseño, no un valor medido', 'Las cuentas de dos sondeos reales',
                   'unas 3,7 veces el peso base del CIS', 'Miramos el ciclo actual y el historial', 'entre un 40% y un 70%',
                   'El consenso sigue siendo una referencia, no una medición directa del voto verdadero',
                   'en catorce ciclos y 43 encuestadoras', 'el efecto del CIS para el PP era de −5,94 puntos',
                   'En el ciclo actual de las elecciones generales entra 1 sondeo de 1 encuestadora.'):
        assert phrase in text, phrase
    # the long stories of the former page, the Bayesian label, the "bias" wording and the reviewer's own caveats are out
    for phrase in ('El problema.', 'La idea.', 'bayesiana', 'sesgo sistemático', 'GAD3 publicó para ABC', 'Antena 3', 'auditad',
                   'deben precisarse'):
        assert phrase not in text, phrase
    assert re.findall(r'id="llamada-([a-z]+)"', page) == ['muestra', 'pesos', 'efectos', 'binomial', 'regresion', 'intervalos']
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page) == ['1', '2', '3', '4', '5', '6']
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_text_without_a_run_has_no_cycle_figures(api, monkeypatch):
    monkeypatch.setattr(metodo, 'cycle_counts', lambda: None)
    status, _, body = call(api, '/metodo/promedio')
    assert status == 200 and 'En el ciclo actual de las elecciones generales' not in html(body)


def test_promedio_text_second_half(house_site):
    page = html(call(house_site, '/metodo/promedio')[2])
    regresion = page[page.index('id="regresion"'):page.index('id="banda"')]
    banda = page[page.index('id="banda"'):page.index('id="voto-estimado"')]
    assert regresion.index('id="anchura"') < regresion.index('id="consecuencias"')
    assert regresion.count('<math display="block"') == 6 and banda.count('<math display="block"') == 1
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', page[page.index('id="regresion"'):]))
    for phrase in ('regresión local polinómica ponderada por kernel', 'Por qué una recta ayuda más que una media',
                   'Diez encuestas no siempre son diez confirmaciones distintas', 'error estándar robusto por grupos',
                   'Se calcula para cada fecha por separado', '93%', '54%', 'No significa que tuviéramos exactamente',
                   'Es la regla de combinación del modelo', 'Un método debe compararse con alternativas sencillas',
                   'La evaluación histórica compara, elección a elección',   # sin backtest publicado, la versión sin cifras
                   'Qué límites tiene el promedio', 'Un partido con menos de tres sondeos utilizables no se ajusta'):
        assert phrase in text, phrase
    # Luis's rule: the figures of the backtest are given without the reviewer's valuation; no reviewer caveats either
    for phrase in ('El problema.', 'La idea.', 'no basta para asegurar', 'es pequeña', 'auditad', 'deben precisarse',
                   'deben especificarse', 'debe especificarse'):
        assert phrase not in text, phrase
    keys = ['muestra', 'pesos', 'efectos', 'binomial', 'regresion', 'intervalos']
    assert page.index('id="voto-estimado"') < page.index('id="comprobacion"') < page.index('id="limites"') < page.index('id="notas"')
    assert page.index('id="notas"') < page.index('id="closing"')
    assert re.findall(r'<li id="nota-([a-z]+)">', page) == keys
    assert re.findall(r'id="llamada-([a-z]+)"', page) == keys
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page) == [str(n) for n in range(1, 7)]
    for target in re.findall(r'href="#(nota-[a-z]+)"', page):
        assert 'id="{}"'.format(target) in page, target
    # no byline after the notes: the footer signs the page (2026-10-22)
    assert 'Por Luis Sancho' not in text and 'class="byline"' not in page
    assert_public_wording(page, '/metodo/promedio')


def test_promedio_text_dates_the_snapshot_figures(api):
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', html(call(api, '/metodo/promedio')[2])))
    # the figures of the sections are a dated snapshot (September measurements, the run of 20 October); the examples are live
    assert 'En el PP, en septiembre, agrupar por encuestadora subía el error estándar de 0,20 a 0,25 puntos' in text
    assert 'Una comprobación publicada para el PP en septiembre de 2026 lo ilustra' in text
    assert 'con los efectos de casa ya fijados (100 réplicas)' in text and 'Reestimar también los efectos en cada réplica da 1,18' in text
    assert 'en septiembre, las inclinaciones de las encuestadoras en el voto del PP' in text
    assert 'En el cálculo del 20 de octubre, sólo 6 de los 328 sondeos' in text
    assert 'su barómetro de 4.020 entrevistas pesaba como uno de 610' in text
    assert 'En septiembre, la anchura del PP iba de 61 a 130 días' in text and text.count('Datos del cálculo del 20 de octubre de 2026') >= 4
    assert 'en septiembre de 2026 era de 4.715 entrevistas' in text


def test_promedio_comprobacion_quotes_the_published_backtest(api, fresh_app):
    """La tarjeta «Cómo comprobamos si funciona» da las cifras vivas del backtest publicado, sin valorarlas."""
    from tests.test_pages import write_backtest
    write_backtest(fresh_app.fs)
    page = html(call(api, '/metodo/promedio')[2])
    card = re.sub(r'\s+', ' ', page[page.index('id="comprobacion"'):page.index('id="limites"')])
    text = re.sub(r'<[^>]+>', '', card)
    assert 'sobre las 2 elecciones generales de 2019 a 2023 leídas 6 días antes de votar' in text
    rows = re.findall(r'<tr><td class="f l">([^<]+)</td><td>([^<]+)</td></tr>', card)
    assert rows == [('Promedio de Demoscopio', '1,69'), ('Último sondeo', '1,74'), ('Media de las cuatro semanas anteriores', '1,86')]
    assert 'El promedio fue el método con menor error en ese ejercicio.' in text
    assert 'incluyó el resultado en el 96% de los casos, frente al 95% al que aspira el cálculo' in text
    assert 'es pequeña' not in text and 'no basta' not in text and 'La evaluación histórica compara, elección a elección' not in text
    assert 'href="/metodo/backtest"' in card


def test_promedio_page_without_a_run_keeps_the_notes_and_drops_the_examples(api, monkeypatch):
    def missing(*args, **kwargs):
        raise HttpError(404, 'not found')
    monkeypatch.setattr(metodo, 'part', missing)
    status, _, body = call(api, '/metodo/promedio')
    page = html(body)
    assert status == 200 and 'id="ejemplo"' not in page and 'En el ciclo actual de las elecciones generales' not in page
    # without the CIS example its note is not there either: five notes, numbered without a gap
    assert re.findall(r'id="llamada-[a-z]+"[^>]*>(\d+)<', page) == [str(n) for n in range(1, 6)]
    assert re.findall(r'<li id="nota-([a-z]+)">', page) == ['muestra', 'pesos', 'efectos', 'regresion', 'intervalos']
    for target in re.findall(r'href="#((?:nota|llamada)-[a-z]+)"', page):
        assert 'id="{}"'.format(target) in page, target



def test_error_example_reads_the_three_errors_of_the_main_parties(api):
    """El ejemplo vivo de los tres errores (2026-10-25): por partido principal del último run nacional, `s`, `e`, `d`
    al horizonte de la elección y los totales de cada modo, con las elecciones que ajustan el estimador de error."""
    example = metodo.error_example()
    pp = pages.catalog(fixture_data('meta'))['PP']
    assert {k: example[k] for k in ('published', 'event_date', 'horizon', 'n_events', 'first_year', 'last_year', 'scope')} == {
        'published': '2026-10-08T12:00:00Z', 'event_date': '2026-11-29', 'horizon': 52, 'n_events': 5,
        'first_year': '2015', 'last_year': '2023', 'scope': 'es'}
    assert example['rows'] == [{'name': 'PP', 'fullname': pp['fullname'], 'color': pp['color'], 'pct': 33.1, 'err': 0.5,
                                'pct_err': 0.87, 'drift': 0.8, 'sd_now': 1.0, 'sd_election': 1.28, 'lo': 30.5, 'hi': 35.7,
                                'dof': 38}]


def test_error_example_is_none_without_the_components_and_skips_parties_without_a_fit(api, monkeypatch):
    fan, meta = fixture_data('fan'), fixture_data('meta')
    old = {**fan, 'rows': [{k: v for k, v in row.items() if k not in ('err', 'pct_err', 'drift', 'dof')} for row in fan['rows']]}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: old if name == 'fan' else fixture_data(name))
    assert metodo.error_example() is None  # a run published before the columns existed
    without = {**meta, 'diagnostics': {k: v for k, v in meta['diagnostics'].items() if k != 'error_events'}}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: without if name == 'meta' else fixture_data(name))
    assert metodo.error_example() is None
    wider = {**meta, 'bmaps': {**meta['bmaps'], 'main': ['PP', 'XX']}}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: wider if name == 'meta' else fixture_data(name))
    assert [row['name'] for row in metodo.error_example()['rows']] == ['PP']  # XX has no fit: no row
    nowcast_only = {**fan, 'horizons': [0], 'rows': [row for row in fan['rows'] if row['horizon'] == 0]}
    monkeypatch.setattr(metodo, 'part', lambda scope, run, name, mode=None: nowcast_only if name == 'fan' else fixture_data(name))
    row = metodo.error_example()['rows'][0]
    assert metodo.error_example()['horizon'] == 0 and row['drift'] == 0.0 and row['sd_election'] == row['sd_now']


def test_error_example_lets_other_errors_through_and_is_none_without_a_run(api, monkeypatch):
    def boom(scope, run, name, mode=None):
        raise HttpError(503, 'bundle unavailable')
    monkeypatch.setattr(metodo, 'part', boom)
    with pytest.raises(HttpError):
        metodo.error_example()

    def not_found(scope, run, name, mode=None):
        raise HttpError(404, 'run not found')
    monkeypatch.setattr(metodo, 'part', not_found)
    assert metodo.error_example() is None


def test_metodo_escanos_shows_the_three_errors_of_the_latest_run(api):
    """La tarjeta de los tres errores en /metodo/escanos (2026-10-25): tabla por partido principal con `s`, `e`, `d`,
    los totales de cada modo y el rango, y la prosa con las elecciones del estimador, visible y con la redacción
    pública."""
    page = html(call(api, '/metodo/escanos')[2])
    assert 'href="#errores">Los tres errores del último cálculo</a>' in page and 'id="errores"' in page
    assert 'Un ejemplo con datos publicados: los tres errores del 8 oct 2026' in page
    table = re.search(r'<table[^>]*id="errors-table">(.*?)</table>', page, re.S).group(1)
    assert re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', table) == [
        'Partido', 'Voto', 'Promedio (s)', 'Histórico (e)', 'Deriva (d), 52 días', 'Hoy (σ)', 'Elección (σ)',
        'Rango 95% en Elección']
    row = re.sub(r'\s+', ' ', re.findall(r'<tr>(.*?)</tr>', table, re.S)[1])
    for cell in ('PP', '33,1%', '0,50', '0,87', '0,80', '1,00', '1,28', '30,5–35,7'):
        assert cell in row, cell
    text = re.sub(r'\s+', ' ', page)
    assert 'las 5 elecciones generales con sondeos celebradas de 2015 a 2023' in text
    assert 'No tiene dirección' in text and 'los 52 días que faltan hasta el 29 nov 2026' in text
    assert_public_wording(page, '/metodo/escanos')


def test_metodo_escanos_hides_the_errors_card_on_an_old_run(api, monkeypatch):
    monkeypatch.setattr(pages.metodo, 'error_example', lambda: None)
    page = html(call(api, '/metodo/escanos')[2])
    assert 'id="errores"' not in page and '#errores' not in page and 'los tres errores' not in page


def test_method_texts_say_the_average_is_read_on_the_day_of_the_run(api):
    """Luis, 2026-10-26: el promedio se lee el día del cálculo, nunca después; los textos del método lo dicen."""
    seats = re.sub(r'\s+', ' ', html(call(api, '/metodo/escanos')[2]))
    assert 'la elección simulada se celebra en la fecha de la estimación, el día del cálculo' in seats
    assert 'que se prolonga unos días tras el último sondeo' not in seats
    average = re.sub(r'\s+', ' ', html(call(api, '/metodo/promedio')[2]))
    assert 'el promedio publicado se lee el día del cálculo, nunca después' in average
    assert 'prolonga la estimación hasta diez días después del último sondeo y después mantiene el valor' not in average
    assert_public_wording(average, '/metodo/promedio')
