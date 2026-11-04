"""Medidas puras de la validación del rating de encuestadoras.

Juzgan, fuera de muestra, si el rating que tenía una casa ANTES de una elección predice su desviación EN esa
elección, y si un promedio simple de sondeos mejora al ponderarlo por el rating. El módulo recibe dos frames
(``pairs``: una fila por casa y elección; ``polls``: una fila por sondeo y partido principal) y devuelve
diccionarios o frames; no toca la base, ficheros ni el modelo vivo.

Diseño: ``docs/superpowers/specs/2026-11-04-rating-validacion-design.md``.
Método: ``docs/analisis-2026-09/metodo-16-rating.md``.

Columnas de ``pairs``: ``event_scope``, ``event_date``, ``population``, ``pollster_id``, ``pollster``, ``d_obs``
(LOR), ``error_avg`` (puntos), ``n_polls``, ``n_event_polls``, ``rating``, ``rating_adj``, ``d_prev`` (LOR, NaN si
``num_events_w == 0``), ``num_events_w``, ``quality``, ``weight_rating``.
Columnas de ``polls``: ``event_scope``, ``event_date``, ``population``, ``pollster_id``, ``pollster``, ``party``,
``error`` (puntos), ``weight_over``, ``weight_sample``, ``weight_rating``.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

WINDOW_DAYS = 30
MIN_POLLSTERS = 3
MIN_EVENT_POLLS = 10
EVIDENCE_SPLIT = 2.0
REPLICAS = 2000
SEED = 0
QUARTERS = 4
POPULATIONS = ('all', 'generales', 'autonomicas')

_CLIP = 1e-3
_ELECTION_KEYS = ['event_scope', 'event_date']


def population_of(scope: str) -> str:
    """Devuelve la población de una elección: ``'generales'`` si el ámbito es ``es``, si no ``'autonomicas'``.

    Parameters
    ----------
    scope : str
        Ámbito de la elección.

    Returns
    -------
    str
        ``'generales'`` o ``'autonomicas'``.
    """
    return 'generales' if scope == 'es' else 'autonomicas'


def _subset(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    """Filtra un frame por población (``all`` lo devuelve entero).

    Parameters
    ----------
    frame : pd.DataFrame
        Frame con columna ``population``.
    population : str
        Una de ``POPULATIONS``.

    Returns
    -------
    pd.DataFrame
        Filas de esa población.
    """
    return frame if population == 'all' else frame[frame['population'] == population]


def _nan_ci() -> tuple:
    """Devuelve el IC vacío ``(nan, nan)``.

    Returns
    -------
    tuple
        Dos NaN.
    """
    return float('nan'), float('nan')


def _finite_prev(pairs: pd.DataFrame) -> pd.DataFrame:
    """Filtra los pares con ``d_prev`` y ``d_obs`` finitos.

    Parameters
    ----------
    pairs : pd.DataFrame
        Frame de pares.

    Returns
    -------
    pd.DataFrame
        Columnas ``d_prev`` y ``d_obs`` sólo con filas finitas.
    """
    sub = pairs[['d_prev', 'd_obs']].astype(float)
    return sub[np.isfinite(sub['d_prev']) & np.isfinite(sub['d_obs'])]


def _fit(sub: pd.DataFrame):
    """Ajusta ``d_obs`` sobre ``d_prev`` con constante.

    Parameters
    ----------
    sub : pd.DataFrame
        Pares finitos (salida de ``_finite_prev``).

    Returns
    -------
    tuple | None
        ``(pendiente, constante)``, o ``None`` con menos de 3 pares o sin variación en ``d_prev``.
    """
    if len(sub) < 3 or sub['d_prev'].nunique() < 2:
        return None
    a, b = np.polyfit(sub['d_prev'], sub['d_obs'], 1)
    return float(a), float(b)


def _bootstrap_by(frame: pd.DataFrame, key: str, statistic: Callable[[pd.DataFrame], float],
                  replicas: int, seed: int) -> tuple:
    """IC 95 % (percentiles 2,5 y 97,5) remuestreando los grupos de ``key`` con reemplazo.

    Parameters
    ----------
    frame : pd.DataFrame
        Datos con la columna ``key``.
    key : str
        Columna que define los grupos que se remuestrean.
    statistic : Callable[[pd.DataFrame], float]
        Estadístico sobre el frame remuestreado.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla de ``np.random.default_rng``.

    Returns
    -------
    tuple
        ``(lo, hi)``; ``(nan, nan)`` con menos de dos grupos.
    """
    groups = [g for _, g in frame.groupby(key)]
    if len(groups) < 2 or replicas < 1:
        return _nan_ci()
    rng = np.random.default_rng(seed)
    values = np.empty(replicas)
    for i in range(replicas):
        picks = rng.integers(0, len(groups), len(groups))
        values[i] = statistic(pd.concat([groups[j] for j in picks], ignore_index=True))
    values = values[np.isfinite(values)]
    if not len(values):
        return _nan_ci()
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(values_by_house: 'dict[int, np.ndarray] | pd.DataFrame',
                 statistic: Callable[[pd.DataFrame], float],
                 replicas: int = REPLICAS, seed: int = SEED) -> tuple:
    """IC 95 % de un estadístico remuestreando casas con reemplazo y conservando todos sus pares.

    Parameters
    ----------
    values_by_house : dict[int, np.ndarray] | pd.DataFrame
        Frame con columna ``pollster_id``, o diccionario casa -> valores (se convierte en un frame con las
        columnas ``pollster_id`` y ``value``).
    statistic : Callable[[pd.DataFrame], float]
        Estadístico sobre el frame remuestreado.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla de ``np.random.default_rng``.

    Returns
    -------
    tuple
        ``(lo, hi)``; ``(nan, nan)`` si hay menos de dos casas.
    """
    if isinstance(values_by_house, dict):
        values_by_house = pd.concat(
            [pd.DataFrame({'pollster_id': pid, 'value': np.atleast_1d(v)}) for pid, v in values_by_house.items()],
            ignore_index=True) if values_by_house else pd.DataFrame({'pollster_id': [], 'value': []})
    return _bootstrap_by(values_by_house, 'pollster_id', statistic, replicas, seed)


def _spearman_value(frame: pd.DataFrame) -> float:
    """Correlación de rango entre ``rating`` y ``d_obs`` (NaN si no hay variación).

    Parameters
    ----------
    frame : pd.DataFrame
        Pares con ``rating`` y ``d_obs``.

    Returns
    -------
    float
        Coeficiente de Spearman, o NaN.
    """
    sub = frame[['rating', 'd_obs']].dropna()
    if len(sub) < 3 or sub['rating'].nunique() < 2 or sub['d_obs'].nunique() < 2:
        return float('nan')
    return float(spearmanr(sub['rating'], sub['d_obs'])[0])


def _slope_value(frame: pd.DataFrame) -> float:
    """Pendiente de ``d_obs`` sobre ``d_prev`` con constante (NaN con menos de 3 pares o sin variación en x).

    Parameters
    ----------
    frame : pd.DataFrame
        Pares con ``d_prev`` y ``d_obs``.

    Returns
    -------
    float
        Pendiente, o NaN.
    """
    fit = _fit(_finite_prev(frame))
    return float('nan') if fit is None else fit[0]


def _measure(pairs: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], n: int,
             replicas: int, seed: int) -> dict:
    """Valor, IC por bootstrap de casas y ``n`` de una medida; sin IC si el valor no se puede calcular.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pares con ``pollster_id``.
    statistic : Callable[[pd.DataFrame], float]
        Estadístico del frame.
    n : int
        Número de pares que cuentan.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla del bootstrap.

    Returns
    -------
    dict
        ``{'value', 'lo', 'hi', 'n'}``.
    """
    value = statistic(pairs)
    lo, hi = _bootstrap_by(pairs, 'pollster_id', statistic, replicas, seed) if np.isfinite(value) else _nan_ci()
    return {'value': value, 'lo': lo, 'hi': hi, 'n': int(n)}


def spearman(pairs: pd.DataFrame, tau: float = None, replicas: int = REPLICAS, seed: int = SEED) -> dict:
    """Correlación de rango entre ``rating`` y ``d_obs`` con IC por bootstrap de casas.

    ``tau`` es sólo documental (no se usa aquí).

    Parameters
    ----------
    pairs : pd.DataFrame
        Pares con ``rating``, ``d_obs`` y ``pollster_id``.
    tau : float
        Desviación del prior; no se usa.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla del bootstrap.

    Returns
    -------
    dict
        ``{'value', 'lo', 'hi', 'n'}``.
    """
    n = len(pairs[['rating', 'd_obs']].dropna())
    return _measure(pairs, _spearman_value, n, replicas, seed)


def slope(pairs: pd.DataFrame, replicas: int = REPLICAS, seed: int = SEED) -> dict:
    """Pendiente de calibración de ``d_obs`` sobre ``d_prev`` (regresión con constante).

    Sólo cuentan los pares con ``d_prev`` finito; con menos de 3, ``value`` es NaN y no hay IC.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pares con ``d_prev``, ``d_obs`` y ``pollster_id``.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla del bootstrap.

    Returns
    -------
    dict
        ``{'value', 'lo', 'hi', 'n'}``.
    """
    n = len(_finite_prev(pairs))
    return _measure(pairs, _slope_value, n, replicas, seed)


def skill(pairs: pd.DataFrame, tau: float) -> dict:
    """Error cuadrático medio de predecir ``d_obs`` con nada, con la calidad previa y con el rating.

    ``pred_none = 0``; ``pred_prior = -tau * Phi^-1(quality)``; ``pred_rating = -tau * Phi^-1(rating)`` (las
    probabilidades se recortan a [1e-3, 1 - 1e-3]). ``skill_x = 1 - mse_x / mse_none``.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pares con ``d_obs``, ``quality`` y ``rating``.
    tau : float
        Desviación del prior.

    Returns
    -------
    dict
        ``{'mse_none', 'mse_prior', 'mse_rating', 'skill_prior', 'skill_rating', 'n'}``.
    """
    sub = pairs[['d_obs', 'quality', 'rating']].dropna()
    nan = float('nan')
    if sub.empty:
        return {'mse_none': nan, 'mse_prior': nan, 'mse_rating': nan, 'skill_prior': nan, 'skill_rating': nan, 'n': 0}
    d = sub['d_obs'].to_numpy()

    def mse(column: str) -> float:
        """Error cuadrático medio de la predicción ``-tau * Phi^-1(column)``.

        Parameters
        ----------
        column : str
            ``'quality'`` o ``'rating'``.

        Returns
        -------
        float
            MSE frente a ``d_obs``.
        """
        pred = -tau * norm.ppf(np.clip(sub[column].to_numpy(), _CLIP, 1 - _CLIP))
        return float(np.mean((d - pred) ** 2))

    mse_none = float(np.mean(d ** 2))
    mse_prior, mse_rating = mse('quality'), mse('rating')

    def ratio(m: float) -> float:
        """Skill ``1 - m / mse_none`` (NaN si ``mse_none`` es 0).

        Parameters
        ----------
        m : float
            MSE del predictor.

        Returns
        -------
        float
            Skill.
        """
        return 1 - m / mse_none if mse_none > 0 else nan

    return {'mse_none': mse_none, 'mse_prior': mse_prior, 'mse_rating': mse_rating,
            'skill_prior': ratio(mse_prior), 'skill_rating': ratio(mse_rating), 'n': int(len(sub))}


def quarters(pairs: pd.DataFrame) -> list:
    """Error medio (puntos) por cuarto del rating, ordenando las casas dentro de cada elección.

    Sólo entran elecciones con al menos ``QUARTERS`` casas; el cuarto 1 es el de mejor rating. Devuelve
    ``[{'population', 'quarter', 'error_avg', 'n'}]`` para ``all``, ``generales`` y ``autonomicas``.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pares con ``rating``, ``error_avg``, ``event_scope`` y ``event_date``.

    Returns
    -------
    list
        Doce diccionarios (tres poblaciones por cuatro cuartos).
    """
    parts = []
    for _, group in pairs.dropna(subset=['rating']).groupby(_ELECTION_KEYS):
        if len(group) < QUARTERS:
            continue
        rank = group['rating'].rank(ascending=False, method='first')
        parts.append(group.assign(quarter=pd.qcut(rank, QUARTERS, labels=False) + 1))
    ranked = pd.concat(parts, ignore_index=True) if parts else pairs.iloc[:0].assign(quarter=pd.Series(dtype=int))
    rows = []
    for population in POPULATIONS:
        sub = _subset(ranked, population)
        for quarter in range(1, QUARTERS + 1):
            q = sub[sub['quarter'] == quarter]
            rows.append({'population': population, 'quarter': quarter,
                         'error_avg': float(q['error_avg'].mean()) if len(q) else float('nan'), 'n': int(len(q))})
    return rows


def discrimination(pairs: pd.DataFrame, tau: float, replicas: int = REPLICAS, seed: int = SEED) -> dict:
    """Medidas de discriminación y calibración del rating para las tres poblaciones.

    Por población: recuentos, ``spearman`` (y por nivel de evidencia, ``num_events_w`` ``>= EVIDENCE_SPLIT`` o
    menos), ``slope`` (y ``slope_polled``, sólo elecciones con ``n_event_polls >= MIN_EVENT_POLLS``) y ``skill``.
    La clave ``'dates'`` lleva la fecha mínima y máxima de las elecciones.

    Parameters
    ----------
    pairs : pd.DataFrame
        Frame ``pairs``.
    tau : float
        Desviación del prior.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla base; cada medida usa ``seed + i``.

    Returns
    -------
    dict
        ``{population: {...}, 'dates': {'first', 'last'}}``.
    """
    out = {}
    for population in POPULATIONS:
        sub = _subset(pairs, population)
        high = sub[sub['num_events_w'] >= EVIDENCE_SPLIT]
        low = sub[sub['num_events_w'] < EVIDENCE_SPLIT]
        polled = sub[sub['n_event_polls'] >= MIN_EVENT_POLLS]
        out[population] = {
            'n_pairs': int(len(sub)),
            'n_pollsters': int(sub['pollster_id'].nunique()),
            'n_events': int(len(sub.drop_duplicates(_ELECTION_KEYS))),
            'spearman': spearman(sub, tau, replicas, seed),
            'spearman_evidence': {'high': spearman(high, tau, replicas, seed + 1),
                                  'low': spearman(low, tau, replicas, seed + 2)},
            'slope': slope(sub, replicas, seed + 3),
            'slope_polled': slope(polled, replicas, seed + 4),
            'skill': skill(sub, tau),
        }
    dates = pairs['event_date']
    out['dates'] = {'first': str(dates.min()) if len(dates) else None, 'last': str(dates.max()) if len(dates) else None}
    return out


def _party_errors(group: pd.DataFrame, rated: bool) -> dict:
    """Error absoluto del promedio ponderado de cada partido de una elección.

    Parameters
    ----------
    group : pd.DataFrame
        Sondeos de una elección con ``party``, ``error`` y los tres pesos (sin NaN).
    rated : bool
        Si es ``True`` el peso lleva además ``weight_rating``.

    Returns
    -------
    dict
        ``{party: abs(sum(w * error) / sum(w))}`` para los partidos con suma de pesos positiva.
    """
    weight = group['weight_over'] * group['weight_sample']
    if rated:
        weight = weight * group['weight_rating']
    out = {}
    for party, idx in group.groupby('party').groups.items():
        w = weight.loc[idx].to_numpy()
        if w.sum() > 0:
            out[party] = abs(float((w * group.loc[idx, 'error'].to_numpy()).sum() / w.sum()))
    return out


def _paired_mae(group: pd.DataFrame) -> tuple:
    """MAE sin y con rating sobre los partidos con suma de pesos positiva en ambos casos.

    Parameters
    ----------
    group : pd.DataFrame
        Sondeos de una elección (sin NaN, índice único).

    Returns
    -------
    tuple
        ``(mae_flat, mae_rated)``; NaN si ningún partido es válido en los dos.
    """
    flat, rated = _party_errors(group, False), _party_errors(group, True)
    common = [p for p in flat if p in rated]
    if not common:
        return float('nan'), float('nan')
    return float(np.mean([flat[p] for p in common])), float(np.mean([rated[p] for p in common]))


def simple_average_errors(polls: pd.DataFrame, min_pollsters: int = MIN_POLLSTERS) -> pd.DataFrame:
    """Error (MAE por partido) del promedio simple con y sin rating, una fila por elección.

    ``w_flat = weight_over * weight_sample`` y ``w_rated = w_flat * weight_rating``. Se descartan las elecciones
    con menos de ``min_pollsters`` casas distintas. ``diff = mae_rated - mae_flat``. Se ignoran las filas con
    ``error`` o pesos NaN, y los partidos con suma de pesos nula en cualquiera de los dos promedios se
    excluyen de ambos MAE. ``n_polls`` es el mayor número de filas por partido (``polls`` no trae id de sondeo).

    Parameters
    ----------
    polls : pd.DataFrame
        Un sondeo por fila y partido (el índice puede repetirse).
    min_pollsters : int
        Mínimo de casas distintas por elección.

    Returns
    -------
    pd.DataFrame
        Una fila por elección con ``mae_flat``, ``mae_rated`` y ``diff``.
    """
    columns = _ELECTION_KEYS + ['population', 'n_polls', 'n_pollsters', 'mae_flat', 'mae_rated', 'diff']
    rows = []
    polls = polls.dropna(subset=['error', 'weight_over', 'weight_sample', 'weight_rating']).reset_index(drop=True)
    for keys, group in polls.groupby(_ELECTION_KEYS):
        n_pollsters = int(group['pollster_id'].nunique())
        if n_pollsters < min_pollsters:
            continue
        mae_flat, mae_rated = _paired_mae(group)
        rows.append({'event_scope': keys[0], 'event_date': keys[1], 'population': population_of(keys[0]),
                     'n_polls': int(group.groupby('party').size().max()), 'n_pollsters': n_pollsters,
                     'mae_flat': mae_flat, 'mae_rated': mae_rated, 'diff': mae_rated - mae_flat})
    return pd.DataFrame(rows, columns=columns)


def summarize_simple_average(elections: pd.DataFrame, replicas: int = REPLICAS, seed: int = SEED) -> dict:
    """Resume ``simple_average_errors`` por población con IC por bootstrap de elecciones.

    Devuelve ``{population: {'n', 'mae_flat', 'mae_rated', 'diff', 'lo', 'hi', 'better', 'worse'}}`` donde
    ``better`` cuenta las elecciones con ``diff < 0`` (el rating ayudó) y ``worse`` las de ``diff > 0``.

    Parameters
    ----------
    elections : pd.DataFrame
        Salida de ``simple_average_errors``.
    replicas : int
        Número de remuestreos.
    seed : int
        Semilla base; cada población usa ``seed + 5 + i``.

    Returns
    -------
    dict
        ``{population: {'n', 'mae_flat', 'mae_rated', 'diff', 'lo', 'hi', 'better', 'worse'}}``.
    """
    out = {}
    for i, population in enumerate(POPULATIONS):
        sub = _subset(elections, population).dropna(subset=['diff']).reset_index(drop=True)
        if sub.empty:
            nan = float('nan')
            out[population] = {'n': 0, 'mae_flat': nan, 'mae_rated': nan, 'diff': nan, 'lo': nan, 'hi': nan,
                               'better': 0, 'worse': 0}
            continue
        lo, hi = _bootstrap_by(sub.assign(pollster_id=np.arange(len(sub))), 'pollster_id',
                               lambda f: float(f['diff'].mean()), replicas, seed + 5 + i)
        out[population] = {'n': int(len(sub)), 'mae_flat': float(sub['mae_flat'].mean()),
                           'mae_rated': float(sub['mae_rated'].mean()), 'diff': float(sub['diff'].mean()),
                           'lo': lo, 'hi': hi, 'better': int((sub['diff'] < 0).sum()),
                           'worse': int((sub['diff'] > 0).sum())}
    return out


def scatter_rows(pairs: pd.DataFrame) -> list:
    """Nube de pares con ``d_prev`` finito: diferenciales ×100 redondeados a 2 decimales.

    Parameters
    ----------
    pairs : pd.DataFrame
        Frame ``pairs``.

    Returns
    -------
    list
        ``[{'scope', 'event_date', 'pollster', 'd_prev', 'd_obs'}]``.
    """
    sub = pairs.loc[_finite_prev(pairs).index]
    return [{'scope': r.event_scope, 'event_date': r.event_date, 'pollster': r.pollster,
             'd_prev': round(float(r.d_prev) * 100, 2), 'd_obs': round(float(r.d_obs) * 100, 2)}
            for r in sub.itertuples()]


def fit_line(pairs: pd.DataFrame):
    """Recta de ``slope`` en los extremos de ``d_prev`` (×100): ``[[x0, y0], [x1, y1]]`` o ``None``.

    Parameters
    ----------
    pairs : pd.DataFrame
        Frame ``pairs``.

    Returns
    -------
    list | None
        Extremos de la recta, o ``None`` si no hay ajuste.
    """
    sub = _finite_prev(pairs)
    fit = _fit(sub)
    if fit is None:
        return None
    a, b = fit
    x0, x1 = float(sub['d_prev'].min()), float(sub['d_prev'].max())
    return [[x0 * 100, (a * x0 + b) * 100], [x1 * 100, (a * x1 + b) * 100]]


def _rounded(value, digits: int = 4):
    """Redondea recursivamente todos los floats de una estructura de dicts y listas.

    Parameters
    ----------
    value : object
        Estructura de dicts, listas y escalares.
    digits : int
        Decimales.

    Returns
    -------
    object
        La estructura con los floats redondeados.
    """
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def rating_check_summary(pairs: pd.DataFrame, polls: pd.DataFrame, tau: float, sigma: float, params: dict,
                         elections: Optional[pd.DataFrame] = None) -> dict:
    """Construye el ``data`` de la parte ``rating-check``.

    Devuelve ``{'summary', 'quarters', 'scatter', 'fit', 'dates', 'params'}`` con los floats a 4 decimales; cada
    población de ``summary`` lleva la clave ``average`` con el resumen del promedio simple. De ``skill`` se
    publican sólo ``skill_prior``, ``skill_rating`` y ``n``: los MSE (del orden de 1e-4 en LOR²) se quedarían en
    0,0002-0,0003 a 4 decimales y no dicen nada (``skill`` los sigue devolviendo).

    Parameters
    ----------
    pairs : pd.DataFrame
        Frame ``pairs``.
    polls : pd.DataFrame
        Frame ``polls``.
    tau : float
        Desviación del prior.
    sigma : float
        Desviación de la medida.
    params : dict
        Parámetros del llamador que se publican junto a los del módulo.
    elections : pd.DataFrame, optional
        Salida de ``simple_average_errors(polls)`` si el llamador ya la tiene (para no calcularla dos veces); si
        falta, se calcula aquí.

    Returns
    -------
    dict
        El ``data`` de la parte.
    """
    summary = discrimination(pairs, tau)
    dates = summary.pop('dates')
    if elections is None:
        elections = simple_average_errors(polls)
    average = summarize_simple_average(elections)
    for population in POPULATIONS:
        summary[population]['skill'] = {key: summary[population]['skill'][key]
                                        for key in ('skill_prior', 'skill_rating', 'n')}
        summary[population]['average'] = average[population]
    full_params = {**params, 'tau': tau, 'sigma': sigma, 'k': (sigma / tau) ** 2, 'window_days': WINDOW_DAYS,
                   'min_pollsters': MIN_POLLSTERS, 'min_event_polls': MIN_EVENT_POLLS,
                   'evidence_split': EVIDENCE_SPLIT, 'replicas': REPLICAS, 'seed': SEED, 'quarters': QUARTERS}
    return _rounded({'summary': summary, 'quarters': quarters(pairs), 'scatter': scatter_rows(pairs),
                     'fit': fit_line(pairs), 'dates': dates, 'params': full_params})
