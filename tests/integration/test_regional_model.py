"""Tests de integración del Simulator en ámbitos autonómicos (M11)."""
import pytest

pytestmark = pytest.mark.integration


def regional_sim(scope, event_date, **kwargs):
    from mtpy.lib.simulator import Simulator
    sim = Simulator(scope=scope, event_date=event_date, drange=6, seed=42, verbose=0, path='.', **kwargs)
    sim.fit_forecast(names=sim.params['names'], max_fc=3, fillna=True)
    return sim


def seats_error(sim, scope, event_date):
    """Error absoluto medio de escaños de los partidos principales en modo determinista."""
    from mtpy.lib.backtest import official_results
    model = sim.result().loc[scope]
    official = official_results(scope, event_date)['seats']
    main = [n for n in sim.event_params['bmaps']['main'] if n in model.index]
    assert int(model.sum()) == sim.n_seats
    return (model[main] - official.reindex(main).fillna(0)).abs().mean()


@pytest.mark.parametrize('scope, event_date', [('es-cl', '2022-02-13'), ('es-as', '2023-05-28'), ('es-cn', '2023-05-28')])
def test_deterministic_seats_are_close_to_the_official_ones(app, scope, event_date):
    # Nueve provincias, tres zonas no provinciales, e islas con lista autonómica y doble umbral
    sim = regional_sim(scope, event_date)
    sim.run(split=True, random=False)
    assert seats_error(sim, scope, event_date) <= 1.5


def test_madrid_2023_miss_is_the_threshold_cliff(app):
    """Madrid 2023: los sondeos daban a Podemos-IU un 5,1 % y sacó el 4,76 %, por debajo de la barrera del 5 %.
    Con la barrera mal resuelta el error es de 3,2 escaños por partido; sin ese partido, el reparto es bueno."""
    sim = regional_sim('es-md', '2023-05-28')
    sim.run(split=True, random=False)
    assert sim.frame().loc['UP', 'vpred'] >= 5 and sim.result().loc['es-md', 'UP'] > 0
    assert seats_error(sim, 'es-md', '2023-05-28') > 1.5
    sim.run(split=True, random=False, names=[n for n in sim.names if n != 'UP'])
    assert seats_error(sim, 'es-md', '2023-05-28') <= 1.5


def test_dhondt_with_scope_thresholds_reproduces_official_district_seats(app):
    """Los umbrales del catálogo (y sus excepciones por evento) con D'Hondt sobre los resultados cargados
    reproducen los escaños oficiales de cada circunscripción. Los votos por circunscripción se estiman de
    porcentajes con un decimal, así que se admite algún baile de un escaño en el último cociente."""
    from mtpy.lib.data import get_thresholds
    from mtpy.lib.simulator import Simulator
    from mtpy.models.elections import EventsData, EventsResults
    data = EventsData().get_results(query=dict(filters=["scope <> 'es'", 'votes IS NOT NULL']), formatted=True)
    res = EventsResults().get_results(query=dict(filters=["scope <> 'es'", 'party_id > 0']), formatted=True)
    total, wrong = 0, []
    for (scope, date), d in data.groupby(['scope', 'date'], observed=True):
        threshold, threshold_scope = get_thresholds(scope, date.strftime('%Y-%m-%d'))
        r = res.loc[(res['scope'] == scope) & (res['date'] == date)]
        shares = r.loc[r['region_id'] == 0].groupby('party', observed=True)['pct'].sum().to_dict()
        # Sólo las candidaturas con escaño en alguna circunscripción: las cuotas por circunscripción del resto
        # son las del conjunto (estimación plana), no un dato
        seated = r.loc[(r['region_id'] > 0) & (r['seats'] > 0), 'party'].unique()
        r = r.loc[r['party'].isin(seated)]
        for _, row in d.loc[d['region_id'] > 0].iterrows():
            rr = r.loc[r['region_id'] == row['region_id']]
            votes = rr.groupby('party', observed=True)['votes'].sum().astype(float).to_dict()
            official = rr.groupby('party', observed=True)['seats'].sum().astype(int).to_dict()
            model = Simulator.alloc_seats(
                votes, sum(official.values()), valid_votes=float(row['votes']), threshold=threshold,
                scope_shares=shares, threshold_scope=threshold_scope
            )
            total += 1
            if model != official:
                wrong.append((scope, date, int(row['region_id'])))
                assert max(abs(model[k] - official[k]) for k in model) == 1, wrong[-1]
    assert total >= 280 and len(wrong) <= 0.02 * total, wrong


def test_scope_settings_are_resolved(app):
    md = regional_sim('es-md', '2023-05-28')
    assert (md.threshold, md.threshold_scope) == (5.0, None)
    assert md.regions == [0, 28] and md.region_names[28] == 'Madrid'
    md.run(split=True, random=False)
    assert (md.frame()['regional'] == 0).all()
    cn = regional_sim('es-cn', '2023-05-28')
    assert (cn.threshold, cn.threshold_scope) == (15.0, 4.0) and 100 in cn.regions


def test_parties_of_part_of_the_community_are_regional(app):
    """Por Ávila y UPL sólo concurren en parte de Castilla y León: son "regionales" para los estimadores,
    con un error de sondeo acorde a su tamaño, y Por Ávila conserva su escaño en la mayoría de simulaciones."""
    cl = regional_sim('es-cl', '2022-02-13')
    cl.run(split=True, random=True, n_sim=200)
    frame = cl.frame()
    assert frame.loc[['XAV', 'UPL'], 'regional'].tolist() == [1, 1]
    assert frame.loc[['PP', 'PSOE', 'VOX'], 'regional'].tolist() == [0, 0, 0]
    assert frame.loc['XAV', 'std_err'] < 0.5
    assert (cl.dist()['XAV'] > 0).mean() > 0.8


def test_random_run_in_a_single_district_scope(app):
    md = regional_sim('es-md', '2023-05-28')
    md.run(split=True, random=True, n_sim=20)
    assert (md.dist().sum(axis=1) == 135).all()
    # Sin partidos "regionales" (distrito único) el gráfico de sus distribuciones no tiene nada que pintar
    assert md.plot_dist_kde(regional=True) is None
    md.run(split=False, random=False)
    assert int(md.totals().sum()) == 135


def test_previous_event_without_polls_is_a_valid_base(app):
    # Foco 1: una elección previa sin sondeos cargados sirve de base del swing
    sim = regional_sim('es-as', '2023-05-28')
    assert sim.prev_date == '2019-05-26'
    sim.run(split=True, random=True, n_sim=10)
    assert (sim.dist().sum(axis=1) == 45).all()


def test_new_party_without_inheritance_rule_does_not_break_the_projection(app):
    """Andalucía 2022: Jaén Merece Más no tiene resultado previo ni regla en `smap` (las de `params.json`
    sustituyen a las derivadas): se queda sin escaños, pero la simulación se hace y reparte los 109."""
    sim = regional_sim('es-an', '2022-06-19')
    assert 'JM+' in sim.names and 'JM+' not in sim.smap
    sim.run(split=True, random=True, n_sim=10)
    assert (sim.dist().sum(axis=1) == 109).all() and (sim.dist()['JM+'] == 0).all()
    assert (sim.dist()['PorA'] > 0).all()        # hereda la geografía de Adelante Andalucía 2018 (regla `agg`)


def test_districts_that_changed_since_the_previous_election(app):
    """Murcia 2019 se votó en distrito único y 2015 en cinco distritos: la base del distrito nuevo es el
    total de la elección anterior."""
    sim = regional_sim('es-mc', '2019-05-26')
    assert sim.regions == [0, 30] and sim.prev_date == '2015-05-24'
    sim.run(split=True, random=True, n_sim=10)
    assert (sim.dist().sum(axis=1) == 45).all()
    assert sim.prev_results['pct'].loc[30].equals(sim.prev_results['pct'].loc[0])


def test_house_effects_prior_is_global(app):
    """M11b: el prior del efecto de una casa se nutre de todas las elecciones, de cualquier ámbito, con el peso de
    cada ámbito, y el partido se resuelve por su raíz: el PSOE de Madrid hereda lo medido en las generales y
    en otras comunidades."""
    from mtpy.lib.forecaster import Forecaster
    fc = Forecaster(scope='es-md', event_date='2027-05-23', drange=6, verbose=0, path='.').build_series()
    history = fc.load_house_history()
    assert history['event_scope'].nunique() > 1 and (history['event_date'] < '2027-05-23').all()
    assert set(history['w_scope'].unique()) == {1.0, 0.5}
    assert (history.loc[history['event_scope'] == 'es', 'w_scope'] == 1.0).all()
    effects = fc.fit_house_effects()
    gad3 = effects.loc[effects['pollster'] == 'GAD3'].reset_index().set_index('name')
    assert gad3.loc['PSOE', 'prior'] != 0 and gad3.loc['PP', 'prior'] != 0
    assert (effects['prior'] != 0).mean() > 0.8      # casi todas las casas traen historia de otros ámbitos
    # Lo mismo visto desde `es`: la historia incluye las autonómicas al 0,5
    nat = Forecaster(scope='es', event_date='2027-08-22', drange=6, verbose=0, path='.').build_series().load_house_history()
    assert (nat['event_scope'] != 'es').any() and nat.loc[nat['event_scope'] != 'es', 'w_scope'].eq(0.5).all()
