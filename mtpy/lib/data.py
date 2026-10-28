import json
import pandas as pd
import warnings

from datetime import date
from typing import Any, Literal, Mapping, Optional

from ..core.app import App
from ..core.worker import Model
from ..models.elections import (
    Districts, Drift, Events, EventsData, EventsResults, Polls, PollsResults,
    Pollsters, PollstersHerding, PollstersParties, PollstersQuality, PollstersRatings, Parties, Scopes
)
from .mtypes import DROP_MTYPES
from .quality import count_national_events, load_criteria, pollsters_quality_values, prior_quality, read_quality


def get_scopes() -> pd.DataFrame:
    """
    Catalogue of election scopes (`data/es-scopes.csv`): `es` and the autonomous communities (`es-*`).

    Returns
    -------
    pd.DataFrame
        Indexed by `scode`: `name`, `ine_code`, `parent`, `demonym`, `threshold`, `threshold_scope`,
        `rating_weight` and `seats`.
    """
    return App.get_().data.read_csv('es-scopes.csv').set_index('scode')


def get_districts(scope: Optional[str] = None) -> pd.DataFrame:
    """
    Catalogue of electoral districts (`data/es-districts.csv`).

    Parameters
    ----------
    scope : str, optional
        Only the districts of this scope.

    Returns
    -------
    pd.DataFrame
        One row per district: `scope`, `region_id`, `name`, `slug`, `ine_code`, `reg_code`, `population`,
        `seats` and `aliases`.
    """
    districts = App.get_().data.read_csv('es-districts.csv')
    if scope is not None:
        districts = districts.loc[districts['scope'] == scope].reset_index(drop=True)

    return districts


def get_district_communities(scope: str) -> dict:
    """
    Autonomous community of each district of a scope.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    dict
        ``{region_id: {'code': int, 'name': str} | None}`` per district of the scope. ``None`` for the
        districts of an autonomous scope (their ``reg_code`` is the ``ine_code`` of the scope itself).
    """
    districts = get_districts(scope)
    own_code = get_scopes().loc[scope, 'ine_code']
    names = App.get_().data.read_csv('es-regions.csv').set_index('code')['region']
    communities = {}
    for region_id, reg_code in zip(districts['region_id'], districts['reg_code']):
        if reg_code == own_code or reg_code not in names.index:
            communities[int(region_id)] = None
        else:
            communities[int(region_id)] = {'code': int(reg_code), 'name': str(names[reg_code])}

    return communities


def get_scope_parent(scope: str) -> Optional[str]:
    """
    Parent of a scope in the catalogue (`es` for the autonomous communities); `None` for the national scope
    and for a scope that is not in the catalogue.
    """
    scopes = get_scopes()
    if scope not in scopes.index or pd.isnull(scopes.loc[scope, 'parent']):
        return None

    return str(scopes.loc[scope, 'parent'])


def resolve_thresholds(
    scope_row: Mapping[str, Any],
    event_conf: Optional[Mapping[str, Any]] = None
) -> tuple[Optional[float], Optional[float]]:
    """
    Legal thresholds of an event: those of its scope unless the event overrides them.

    Parameters
    ----------
    scope_row : Mapping
        Row of the scope in the catalogue: `threshold` (district) and `threshold_scope` (whole scope).
    event_conf : Mapping, optional
        Entry of the event in `data/params.json`. The keys `threshold` and `threshold_scope` present in it
        prevail even when they are null (e.g. Murcia until 2015: no district threshold, 5 % regional).

    Returns
    -------
    tuple
        District and scope thresholds, as percentages; `None` where there is none.
    """
    out = []
    for key in ('threshold', 'threshold_scope'):
        value = event_conf[key] if event_conf is not None and key in event_conf else scope_row[key]
        out.append(None if value is None or pd.isnull(value) else float(value))

    return out[0], out[1]


def get_thresholds(
    scope: str,
    event_date: str
) -> tuple[Optional[float], Optional[float]]:
    """
    Legal thresholds of an event (see `resolve_thresholds`), from the catalogue of scopes and the entry of
    the event in `data/params.json`.
    """
    params = json.loads(App.get_().data.read('params.json')).get(scope, {})

    return resolve_thresholds(get_scopes().loc[scope], params.get(event_date))


def save_catalogues() -> dict[str, int]:
    """
    Replicate the versioned catalogues into the database: scopes, districts and the pollster quality
    evaluations. The tables are created when missing (never replaced) and their rows upserted by key; the
    evaluations of the pollsters no longer in the CSV are deleted. Then `pollsters.quality` is derived for
    every pollster (the scale floor for the ones not evaluated).

    The replica `elections.pollsters_quality` holds nothing that is not in the CSV, so it can be dropped and
    recreated at will (`DROP TABLE elections.pollsters_quality`, then `save_catalogues()`). That is the way to
    change its column types: `create(replace=False)` only creates a missing table and never alters an
    existing one.

    Returns
    -------
    dict
        Rows written to `scopes`, `districts` and `quality`, and pollsters updated in `pollsters`.
    """
    out = {}
    for name, model, data in [
        ('scopes', Scopes(), get_scopes().reset_index()),
        ('districts', Districts(), get_districts())
    ]:
        if not model.table_exists():
            model.create(replace=False)

        df = model.format_data(data, int_type='nullable', bin_type='nullable', sort=True)
        model.upsert(df)
        out[name] = int(df.shape[0])

    criteria = load_criteria()
    quality = read_quality(criteria)
    n_national_events = count_national_events(get_national_events(), pd.Timestamp(date.today()))
    # the replica keeps the points and the quality of today: the implantation counts the generals held until now
    today = prior_quality(quality['pollster_id'], quality, criteria['scale'], n_national_events)
    quality = quality.assign(points=today['points'].to_numpy(), quality=today['quality'].to_numpy())
    qmodel = PollstersQuality()
    if not qmodel.table_exists():
        qmodel.create(replace=False)

    df = qmodel.format_data(quality, int_type='nullable', bin_type='nullable', sort=True)
    if df.shape[0]:
        ids = ', '.join(str(int(i)) for i in df['pollster_id'])
        qmodel.execute('DELETE FROM {} WHERE pollster_id NOT IN ({})'.format(qmodel.table, ids))
    else:
        qmodel.execute('DELETE FROM {}'.format(qmodel.table))
    qmodel.upsert(df)
    out['quality'] = int(df.shape[0])

    pollsters = get_pollsters()
    pollsters['quality'] = pollsters_quality_values(pollsters, quality, criteria['scale'], n_national_events)
    pmodel = Pollsters()
    pmodel.upsert(pmodel.format_data(pollsters, int_type='nullable', bin_type='nullable', sort=True))
    out['pollsters'] = int(pollsters.shape[0])

    return out


def get_event_dmat(
    scope: str,
    event_date: str,
    polls: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
    parties: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if polls is None:
        polls = get_poll_series(scope, event_date).set_index(['event_date', 'days', 'pollster']).sort_index()
    if events is None:
        events = get_event_series(scope, event_date).set_index('date').sort_index()
    if parties is None:
        parties = get_parties()

    poll_cols = [p for p in parties['name'] if p in polls.columns]
    if event_date in polls.index.get_level_values(0):
        dpolls = polls.loc[event_date][poll_cols].dropna(axis=1, how='all')
    else:
        # An election without polls (e.g. the one before the first polled one of a regional scope)
        dpolls = polls.iloc[:0].droplevel(0)[poll_cols].dropna(axis=1, how='all')
    poll_parties = dpolls.mean().sort_values(ascending=False).index.tolist()

    devent = events.loc[event_date].loc[[p for p in parties['name'] if p in events.columns]].dropna()
    event_parties = devent.sort_values(ascending=False).index.tolist()

    if dpolls.shape[0] > 0:
        final_parties = [p for p in event_parties if p in poll_parties]
    else:
        final_parties = event_parties
    all_parties = final_parties + [p for p in poll_parties if p not in final_parties]

    dpolls = dpolls.reindex(all_parties, axis=1)[all_parties]
    devent = devent.reindex(all_parties, axis=0).loc[all_parties]

    if dpolls.loc[1:42].shape[0] > 0:
        dpolls = dpolls.loc[1:42]

    if dpolls.shape[0] > 0:
        dagg = dpolls.agg(['count', 'mean', 'std'])
        dagg.loc['count'] /= dpolls.shape[0] / 100
    else:
        # No polls (and, for an upcoming event, no parties either): empty statistics
        dagg = pd.DataFrame(float('nan'), index=['count', 'mean', 'std', 'result'], columns=all_parties)
    if len(all_parties) > 0:
        dagg.loc['result'] = devent
    dagg['days'] = 0
    dagg = dagg.rename_axis('pollster').reset_index().set_index(['days', 'pollster']).round(2)[all_parties]

    df = pd.concat([dagg, dpolls])

    return df

PARAMS_FILE = 'params.json'
# Width of a line of a wrapped list of `params.json` (indentation included)
PARAMS_WIDTH = 100
# Sections of an election whose direct lists are written as wrapped blocks (`event`, `polls`, `max`, `main`...)
PARAMS_LIST_SECTIONS = ('parties', 'bmaps')


def read_params() -> dict:
    """Read ``data/params.json`` (the whole file: ``{scope: {date: entry}}``)."""
    return json.loads(App.get_().data.read(PARAMS_FILE))


def _format_inline(value) -> str:
    """JSON of a value on one line, as the file writes rules and short lists."""
    return json.dumps(value, ensure_ascii=False)


def _format_block_list(items: list, indent: str) -> str:
    """A list as a block: items joined by ``, `` on lines of at most `PARAMS_WIDTH` columns, one more level in."""
    if not items:
        return '[]'
    inner = indent + ' ' * 4
    lines, current = [], ''
    for item in items:
        text = _format_inline(item)
        candidate = text if not current else current + ', ' + text
        # the trailing comma of a continued line counts too
        if current and len(inner + candidate + ',') > PARAMS_WIDTH:
            lines.append(current)
            current = text
        else:
            current = candidate
    lines.append(current)
    return '[\n' + ',\n'.join(inner + line for line in lines) + '\n' + indent + ']'


def _format_value(value, keys: tuple, indent: str) -> str:
    """
    Format a value of the file at the path ``keys`` (scope, date, section, ...), with ``indent`` as the indentation
    of the line that holds its key.
    """
    depth = len(keys)
    if isinstance(value, dict):
        if not value or (depth >= 4 and keys[2] == 'smap'):
            return _format_inline(value)
        inner = indent + ' ' * 4
        lines = ['{}{}: {}'.format(inner, _format_inline(key), _format_value(item, keys + (key,), inner))
                 for key, item in value.items()]
        return '{\n' + ',\n'.join(lines) + '\n' + indent + '}'
    if isinstance(value, list):
        if depth == 4 and keys[2] in PARAMS_LIST_SECTIONS:
            return _format_block_list(value, indent)
        return _format_inline(value)
    return _format_inline(value)


def format_params(data: dict) -> str:
    """
    Text of ``params.json`` in the style of the file.

    Four-space indentation; dicts one key per line except the rules of ``smap`` (and empty dicts), written inline;
    lists inline except the direct lists of ``parties`` and ``bmaps`` (``event``, ``polls``, ``max``, ``main``),
    written as blocks wrapped at `PARAMS_WIDTH` columns. The text ends with a single newline.

    Parameters
    ----------
    data : dict
        ``{scope: {date: entry}}``.

    Returns
    -------
    str
        The file text.
    """
    return _format_value(data, (), '') + '\n'


def write_params(data: dict) -> str:
    """Write ``data`` as ``data/params.json`` (`format_params`); returns the path written."""
    return App.get_().data.write(format_params(data), PARAMS_FILE)


BLOCKS = ('Derecha', 'Izquierda', 'Regionalista', 'Separatista')
VS_BLOCKS = ('Derecha', 'Izquierda')


def agg_names(smap: Optional[dict]) -> list[str]:
    """
    Names absorbed by the ``agg`` rules of a seat map.

    Parameters
    ----------
    smap : dict, optional
        ``smap`` of an election: ``{party: rule | [rules]}`` where a rule is ``{'type', 'names', ...}``; a new
        party has ``[]``. Anything that is not a dict with ``type == 'agg'`` and ``names`` is ignored.

    Returns
    -------
    list of str
        The ``names`` of every ``agg`` rule, without duplicates, in order of appearance.
    """
    names = []
    for rules in (smap or {}).values():
        for rule in (rules if isinstance(rules, list) else [rules]):
            if isinstance(rule, dict) and rule.get('type') == 'agg':
                names.extend(n for n in (rule.get('names') or []) if n not in names)
    return names


def derive_event_params(df: pd.DataFrame, parties: pd.DataFrame, smap: Optional[dict] = None) -> dict:
    """
    Derive the ``parties`` and ``bmaps`` of an election from its cycle matrix and the block of each party.

    Parameters
    ----------
    df : pd.DataFrame
        Matrix of the cycle (`get_event_dmat`): one column per party, in its order (official result descending,
        then poll-only parties by their mean), with the rows ``(0, 'count')`` (share of polls that include the
        party, in percent), ``(0, 'mean')``, ``(0, 'std')`` and ``(0, 'result')``.
    parties : pd.DataFrame
        Party catalogue with ``name`` and ``block``.
    smap : dict, optional
        ``smap`` of the election, for an election without results: the names its ``agg`` rules absorb are not
        candidacies (`agg_names`).

    Returns
    -------
    dict
        ``{'parties': {'event', 'polls'}, 'bmaps': {'main', 'blocks', 'vs'}}``: ``polls`` every column; ``event``
        the parties with a result, or ``polls`` minus the absorbed names when there is none; ``main`` the
        parties of ``event`` with ``count > 95``; ``blocks`` the parties of ``polls`` by `BLOCKS` (empty blocks
        left out, parties without one of those blocks in none); ``vs`` the `VS_BLOCKS` entries of ``blocks``.
    """
    polls = df.columns.tolist()
    if polls:
        # `.loc[[key]]` keeps a frame whatever the index looks like; its first row is the series by party
        result = df.loc[[(0, 'result')]].iloc[0]
        count = df.loc[[(0, 'count')]].iloc[0]
        event = result.dropna().index.tolist()
    else:
        result = count = pd.Series(dtype=float)
        event = []
    if not event:
        absorbed = agg_names(smap)
        event = [p for p in polls if p not in absorbed]
    main = [p for p in event if count.get(p, float('nan')) > 95]

    block_of = dict(zip(parties['name'], parties['block']))
    blocks = {b: [p for p in polls if block_of.get(p) == b] for b in BLOCKS}
    blocks = {b: names for b, names in blocks.items() if names}
    vs = {b: blocks[b] for b in VS_BLOCKS if b in blocks}

    return {'parties': {'event': event, 'polls': polls}, 'bmaps': {'main': main, 'blocks': blocks, 'vs': vs}}


def derive_scope_params(
    scope: str,
    event_dates: Optional[list[str]] = None,
    overrides: Optional[dict] = None
) -> dict[str, dict]:
    """
    Derive the parameters of the elections of a scope from the database (`derive_event_params`).

    Parameters
    ----------
    scope : str
        Scope code.
    event_dates : list of str, optional
        Elections to derive (``YYYY-MM-DD``); all the elections of the scope by default.
    overrides : dict, optional
        Entries of ``params.json`` of the scope (``{date: entry}``): the ``smap`` of each election tells which
        polled names are not candidacies in an election without results.

    Returns
    -------
    dict
        ``{date: {'parties', 'bmaps', 'smap'}}``: `derive_event_params` plus ``smap`` with an empty rule for every
        party of ``event`` that had no result in the previous election of the scope.
    """
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    overrides = overrides or {}

    all_event_dates = get_event_dates(scope=scope)
    check_event_dates = []
    for dt in event_dates:
        check_event_dates.append(dt)
        ix = all_event_dates.index(dt)
        if ix > 0 and all_event_dates[ix - 1] not in check_event_dates:
            check_event_dates.append(all_event_dates[ix - 1])
    check_event_dates = sorted(check_event_dates)

    polls = get_poll_series(scope, check_event_dates).set_index(['event_date', 'days', 'pollster']).sort_index()
    events = get_event_series(scope, check_event_dates).set_index('date').sort_index()
    parties = get_parties()

    derived = {}
    for dt in event_dates:
        # Sorted index: `.loc[(0, ...)]` on the unsorted, non-unique (days, pollster) index
        # takes pandas' linear path and raises a PerformanceWarning per lookup.
        df = get_event_dmat(scope, dt, polls, events, parties).sort_index()
        params = derive_event_params(df, parties, smap=(overrides.get(dt) or {}).get('smap'))

        new_parties = []
        if all_event_dates.index(dt) > 0:
            dprev = events.loc[
                all_event_dates[all_event_dates.index(dt) - 1]
            ].loc[[p for p in parties['name'] if p in events.columns]].dropna()
            new_parties = [p for p in params['parties']['event'] if p not in dprev.index.tolist()]

        derived[dt] = params | {'smap': {p: [] for p in new_parties}}

    return derived


def get_event_params(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    path: Optional[str] = None
) -> dict:
    """
    Parameters of the elections of a scope: derived from the data (`derive_scope_params`) and overridden by the
    entries of ``data/params.json`` when ``path`` is given and the election is listed there.

    Parameters
    ----------
    scope : str
        Scope code.
    event_dates : list of str or str, optional
        Elections (``YYYY-MM-DD``); all by default. A single date returns its entry alone.
    path : str, optional
        Any value reads ``params.json`` from the data directory; ``None`` derives everything.

    Returns
    -------
    dict
        ``{date: params}``, or the params of the single date given.
    """
    is_single = False
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    elif isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    if path is not None:
        params = read_params().get(scope, {})
    else:
        params = {}

    derived = derive_scope_params(scope, event_dates, overrides=params)
    event_params = {dt: derived[dt] | params.get(dt, {}) for dt in event_dates}

    if is_single:
        event_params = event_params[event_dates[0]]

    return event_params


def get_event_data(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    region_id: Optional[int] = None
) -> pd.DataFrame:
    is_single = False
    if isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    filters = [
        "scope = '{}'".format(scope),
        "date IN ('{}')".format("', '".join(event_dates))
    ]
    if region_id is not None:
        filters.append("region_id = {}".format(region_id))

    data = EventsData().get_results(query={
        'filters': filters
    }, formatted=True)

    if is_single:
        data = data.loc[data['date'] == event_dates[0]]

    return data


def get_event_results(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    region_id: Optional[int] = None
) -> pd.DataFrame:
    is_single = False
    if isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    filters = [
        "scope = '{}'".format(scope),
        "date IN ('{}')".format("', '".join(event_dates))
    ]
    if region_id is not None:
        filters.append("region_id = {}".format(region_id))
    
    results = EventsResults().get_results(query={
        'filters': filters
    }, formatted=True)

    if is_single:
        results = results.loc[results['date'] == event_dates[0]]

    return results

def get_event_series(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    metric: Literal['pct', 'votes', 'seats'] = 'pct'
) -> pd.DataFrame:
    """
    Get the events final results.

    Parameters
    ----------
    scope : str
        Election scope.
    event_dates : list[str] | str, optional
        Election dates.
    metric : Literal['pct', 'votes', 'seats'], optional
        Which metric should be used to get the each party's results.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the events final results.
    """
    is_single = False
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    elif isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    filters = [
        "scope = '{}'".format(scope),
        "date IN ('{}')".format("', '".join(event_dates))
    ]

    events = Events().get_results(query=dict(
        filters=filters
    ), formatted=True)

    data = EventsData().get_results(query=dict(
        filters=filters + ['region_id = 0']
    ), formatted=True).drop(columns=['scope']).sort_values('date')

    events = events.merge(data, on='date', how='inner')

    results = EventsResults().get_results(query=dict(
        filters=filters + ['region_id = 0']
    ), formatted=True).drop(columns=['scope']).set_index([
        'date', 'party'
    ])[metric].unstack(level=-1).rename_axis(None, axis=1).reset_index()

    series = events.merge(results, on='date', how='left')

    if is_single:
        series = series.loc[series['date'] == event_dates[0]]

    return series


def get_poll_results(
    scope: str,
    event_dates: Optional[list[str] | str] = None
) -> pd.DataFrame:
    is_single = False
    if isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    filters = [
        "event_scope = '{}'".format(scope),
        "event_date IN ('{}')".format("', '".join(event_dates))
    ]
    
    results = PollsResults().get_results(query={
        'filters': filters
    }, formatted=True)

    if is_single:
        results = results.loc[results['event_date'] == event_dates[0]]

    return results


def get_poll_series(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    metric: str = 'pct',
    drange: Optional[tuple[int, int]] = None,
    drop_mtypes: Optional[list[str]] = None,
    drop_contexts: Optional[list[str]] = None
) -> pd.DataFrame:
    is_single = False
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    elif isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    filters = [
        "t.event_scope = '{}'".format(scope),
        "t.event_date IN ('{}')".format("', '".join(event_dates))
    ]
    if drange is not None:
        if drange[0] and drange[1]:
            filters.append('t.days BETWEEN {} AND {}'.format(*drange))
        elif drange[0]:
            filters.append('t.days >= {}'.format(drange[0]))
        elif drange[1]:
            filters.append('t.days <= {}'.format(drange[1]))
    if isinstance(drop_mtypes, (list, tuple)) and len(drop_mtypes) > 0:
        filters.append("(t.mtype IS NULL OR t.mtype NOT IN ('{}'))".format("', '".join(drop_mtypes)))
    if isinstance(drop_contexts, (list, tuple)) and len(drop_contexts) > 0:
        filters.append("(t.ctype IS NULL OR t.ctype NOT IN ('{}'))".format("', '".join(drop_contexts)))

    polls = Polls().get_results(query=dict(
        filters=filters
    ), formatted=True)

    results = PollsResults().get_results(query=dict(
        relations=[dict(
            model=Polls(),
            name='p'
        )],
        filters=[f.replace('t.', 'p.') for f in filters]
    ), formatted=True).drop(columns=['event_scope']).set_index([
        'date', 'pollster_id', 'sponsor_id', 'party'
    ])[metric].unstack(level=-1).rename_axis(None, axis=1).reset_index()

    series = polls.merge(results, on=['date', 'pollster_id', 'sponsor_id'], how='left')

    if is_single:
        series = series.loc[series['event_date'] == event_dates[0]]

    return series


def get_pollster_ratings(pollster_id: int) -> pd.DataFrame:
    """
    Ratings of one pollster before every election of every scope, the next ones included.

    Parameters
    ----------
    pollster_id : int
        Id of the pollster.

    Returns
    -------
    pd.DataFrame
        Its rows of the ratings table (``event_scope``, ``event_date``, ``rating``, ``rating_adj``...), by event
        date and scope.
    """
    ratings = PollstersRatings().get_results(query=dict(
        filters=['pollster_id = {}'.format(int(pollster_id))]
    ), formatted=True)

    return ratings.sort_values(['event_date', 'event_scope'], ignore_index=True)


def get_pollster_events(pollster_id: int) -> pd.DataFrame:
    """
    Elections of every scope a pollster published polls for, with how many of its polls enter the ratings.

    Parameters
    ----------
    pollster_id : int
        Id of the pollster.

    Returns
    -------
    pd.DataFrame
        ``event_scope``, ``event_date``, ``n_polls`` (its polls of featured elections with computed errors, the
        ones the rating algorithm reads), ``error_avg``, ``error_blocks`` and ``bias_dev_adj`` (the plain means
        of those polls: the error over the main parties and the signed error of the margin between the two
        blocks, both in points, and the deviation of their bias from the comparable polls, in the bias scale of
        the rating; null without a value), by event date and scope.
    """
    polls = Polls().get_results(query=dict(
        filters=['pollster_id = {}'.format(int(pollster_id)), 'featured', 'computed']
    ), formatted=True)
    # `event_scope` is categorical in the formatted table: only the combinations with polls, as plain strings
    polls = polls.assign(event_scope=polls['event_scope'].astype(str))
    measures = ['error_avg', 'error_blocks', 'bias_dev_adj']
    for column in measures:
        polls[column] = pd.to_numeric(polls[column], errors='coerce') if column in polls.columns else float('nan')
    events = polls.groupby(['event_scope', 'event_date'], sort=False, observed=True).agg(
        n_polls=('event_date', 'size'), **{column: (column, 'mean') for column in measures}
    ).reset_index()
    events = events.loc[events['n_polls'] > 0]

    return events.sort_values(['event_date', 'event_scope'], ignore_index=True)[
        ['event_scope', 'event_date', 'n_polls'] + measures]


def get_national_events() -> pd.DataFrame:
    """
    General elections each pollster polled, for the "implantation" criterion of the prior quality.

    A pollster counts for a general election (``event_scope = 'es'``) when it has at least one poll with a
    non-null ``bias`` that is not an exit poll or a ``wban`` (``ctype`` null counts); ``mtype`` is not filtered, so
    the public figure also holds for online panels and aggregators.

    Returns
    -------
    pd.DataFrame
        ``pollster_id`` and ``event_date``, one row per pollster and general election.
    """
    polls = Polls().get_results(query=dict(filters=[
        "event_scope = 'es'", 'bias IS NOT NULL', "(ctype IS NULL OR ctype NOT IN ('wban', 'exit'))"
    ]), formatted=True)
    polls = polls.assign(event_scope=polls['event_scope'].astype(str))
    polls = polls.loc[(polls['event_scope'] == 'es') & polls['bias'].notna() & ~polls['ctype'].isin(['wban', 'exit'])]

    return polls[['pollster_id', 'event_date']].drop_duplicates().sort_values(
        ['event_date', 'pollster_id'], ignore_index=True)


def get_ratings(
    scope: str,
    event_dates: Optional[list[str] | str] = None
) -> pd.DataFrame:
    is_single = False
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    elif isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    next_event_date = get_next_event_date(scope=scope, date_from=event_dates[-1])
    if next_event_date is not None:
        event_dates = event_dates + [next_event_date]

    filters = [
        "event_scope = '{}'".format(scope),
        "event_date IN ('{}')".format("', '".join(event_dates))
    ]

    pollsters = Pollsters().get_results(formatted=True)
    ratings = PollstersRatings().get_results(query=dict(
        filters=filters
    ), formatted=True)

    keys = ['event_date', 'pollster_id']
    ratings = ratings.merge(
        pollsters[['id', 'name', 'mtype']], left_on='pollster_id', right_on='id', how='left'
    ).rename(columns={'name': 'pollster'})[
        keys + ['pollster', 'mtype'] + [col for col in ratings.columns if col not in keys]
    ].sort_values(keys, ignore_index=True)

    if is_single:
        ratings = ratings.loc[ratings['event_date'] == event_dates[0]]

    return ratings


def get_parties() -> pd.DataFrame:
    parties = Parties().get_results(formatted=True)
    parties = parties.rename(index={0: parties.shape[0]}).sort_index(ignore_index=True)

    return parties


def get_events(scope: str) -> pd.DataFrame:
    """
    Elections of a scope.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    pd.DataFrame
        One row per election, with `date`, `scope`, `name`, `featured` and `has_results` (whether it has an
        official result: a row of the whole scope with votes in `events_data`).
    """
    events = Events().get_results(query={
        'filters': ["scope = '{}'".format(scope)]
    }, formatted=True)
    data = EventsData().get_results(query=dict(
        filters=["scope = '{}'".format(scope), 'region_id = 0', 'votes IS NOT NULL']
    ), formatted=True)
    events['has_results'] = events['date'].isin(data['date'])

    return events


def get_pollsters() -> pd.DataFrame:
    pollsters = Pollsters().get_results(formatted=True)

    return pollsters


def get_pollsters_quality() -> pd.DataFrame:
    """
    Read the replica of the pollster quality evaluations.

    Returns
    -------
    pd.DataFrame
        The formatted table, or an empty frame with the model columns if the table does not exist yet.
    """
    model = PollstersQuality()
    if not model.table_exists():
        return pd.DataFrame(columns=list(model.meta))

    return model.get_results(formatted=True)


def save_model_data(
    model: Model | str,
    data: pd.DataFrame
) -> int:
    """
    Save data to model's table into the database.

    Parameters
    ----------
    model : Model | str
        Model instance or name.
    data : pd.DataFrame
        Data to save.

    Returns
    -------
    int
        Number of rows updated.
    """
    df = data.copy().reset_index()
    
    if isinstance(model, str):
        model = globals()[model]()

    # Separate model keys from columns to be updated
    keys = model.key if isinstance(model.key, (tuple, list)) else [model.key]
    columns = [col for col in df.columns.tolist() if col not in keys]

    # Use the model formatter to get the data in the correct types to be saved
    df = model.format_data(df, int_type='nullable', bin_type='nullable', sort=True)[keys + columns]

    # The upsert overwrites the computed columns of the rows present in `df`; rows of the same event
    # that are not in `df` (incremental runs) keep their previous values
    return model.upsert(df)


def save_ratings_data(
    data: pd.DataFrame,
    update_pollsters: bool = True
) -> int:
    """
    Save data to the ratings table into the database.
    Also save each pollster's last rating to the pollsters table.

    Parameters
    ----------
    data : pd.DataFrame
        Ratings indexed by `event_scope`, `event_date` and `pollster_id`.
    update_pollsters : bool, optional
        Whether to write the last rating of each pollster to `pollsters.rating`. That column holds the
        rating of the national scope, so the ratings of a regional scope must not touch it.

    Returns
    -------
    int
        Number of rows updated.
    """
    rmodel = PollstersRatings()
    dr = data.copy().reset_index()

    # Use the model formatter to get the data in the correct types to be saved
    dr = rmodel.format_data(dr, int_type='nullable', bin_type='nullable', sort=True)

    # Remove previous data for the same election event
    rmodel.execute("DELETE FROM {} WHERE event_scope IN ('{}') AND event_date IN ('{}')".format(
        rmodel.table,
        "', '".join(dr.event_scope.unique()),
        "', '".join(dr.event_date.dt.strftime('%Y-%m-%d').unique())
    ))

    nrows = rmodel.upsert(dr)
    if not update_pollsters:
        return nrows

    rlast = dr.groupby('pollster_id')['rating'].last()

    pmodel = Pollsters()
    dp = get_pollsters()
    # aggregators and online panels have no rating (`DROP_MTYPES`): null, not the 0 of a house without evidence
    dp['rating'] = dp['id'].map(rlast).fillna(0).astype(int).astype('Int64')
    dp.loc[dp['mtype'].isin(DROP_MTYPES), 'rating'] = pd.NA

    # Use the model formatter to get the data in the correct types to be saved
    dp = pmodel.format_data(dp, int_type='nullable', bin_type='nullable', sort=True)

    pmodel.upsert(dp)

    return nrows


def get_herding(
    scope: Optional[str] = 'es',
    event_dates: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Load the herding table (see `Computer.compute_herding`): one row per election cycle and pollster. An
    empty frame with the model columns is returned when the table has not been created yet.

    Parameters
    ----------
    scope : str, optional
        Election scope; `None` loads every scope.
    event_dates : list of str, optional
        Restrict to these elections.
    """
    model = PollstersHerding()
    if not model.table_exists():
        return pd.DataFrame(columns=model.columns)

    filters = []
    if scope is not None:
        filters.append("event_scope = '{}'".format(scope))
    if event_dates is not None:
        if len(event_dates) == 0:
            return pd.DataFrame(columns=model.columns)
        filters.append("event_date IN ('{}')".format("', '".join(event_dates)))

    return model.get_results(query=dict(filters=filters), formatted=True)


def save_herding_data(
    data: pd.DataFrame
) -> int:
    """
    Save the herding data into the database, replacing the rows of the same elections. The table is created
    when missing (never replaced).

    Returns
    -------
    int
        Number of rows updated.
    """
    model = PollstersHerding()
    if not model.table_exists():
        model.create(replace=False)

    df = data.copy().reset_index(drop=True)
    df = df.loc[df['pollster_id'].notnull()]
    if df.shape[0] == 0:
        return 0

    df = model.format_data(df, int_type='nullable', bin_type='nullable', sort=True)

    # Remove previous data for the same election events
    model.execute("DELETE FROM {} WHERE event_scope IN ('{}') AND event_date IN ('{}')".format(
        model.table,
        "', '".join(df.event_scope.unique()),
        "', '".join(df.event_date.dt.strftime('%Y-%m-%d').unique())
    ))

    return model.upsert(df)


def get_drift(
    scope: str = 'es',
    event_dates: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Load the drift table (see `Computer.get_drift_data`): one row per past election, party and horizon.
    An empty frame with the model columns is returned when the table has not been created yet.

    Parameters
    ----------
    scope : str, default 'es'
        Election scope.
    event_dates : list of str, optional
        Restrict to these elections (e.g. the ones before the event being forecast, for a backtest).

    Returns
    -------
    pd.DataFrame
        Drift data.
    """
    model = Drift()
    if not model.table_exists():
        return pd.DataFrame(columns=model.columns)

    filters = ["event_scope = '{}'".format(scope)]
    if event_dates is not None:
        if len(event_dates) == 0:
            return pd.DataFrame(columns=model.columns)
        filters.append("event_date IN ('{}')".format("', '".join(event_dates)))

    return model.get_results(query=dict(filters=filters), formatted=True)


def save_drift_data(
    data: pd.DataFrame
) -> int:
    """
    Save the drift data into the database, replacing the rows of the same elections. The table is
    created when missing (never replaced: `create(replace=True)` would drop it).

    Returns
    -------
    int
        Number of rows updated.
    """
    model = Drift()
    if not model.table_exists():
        model.create(replace=False)

    df = data.copy().reset_index(drop=True)
    if df.shape[0] == 0:
        return 0

    # Use the model formatter to get the data in the correct types to be saved
    df = model.format_data(df, int_type='nullable', bin_type='nullable', sort=True)

    # Remove previous data for the same election events
    model.execute("DELETE FROM {} WHERE event_scope IN ('{}') AND event_date IN ('{}')".format(
        model.table,
        "', '".join(df.event_scope.unique()),
        "', '".join(df.event_date.dt.strftime('%Y-%m-%d').unique())
    ))

    return model.upsert(df)


def get_house_effects(
    scope: Optional[str] = 'es',
    event_dates: Optional[list[str]] = None,
    date_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the house effects table (see `Computer.compute_house_effects`): one row per election, pollster and
    party. An empty frame with the model columns is returned when the table is missing or has no columns yet.

    Parameters
    ----------
    scope : str, optional
        Election scope; `None` loads every scope (the global prior of the house effects, M11).
    event_dates : list of str, optional
        Restrict to these elections (e.g. the ones before the event being forecast).
    date_to : str, optional
        Only the elections held strictly before this date.
    """
    model = PollstersParties()
    if not model.table_exists() or set(model.columns) - set(model._dal.get_columns(model.table)):
        return pd.DataFrame(columns=model.columns)

    filters = []
    if scope is not None:
        filters.append("event_scope = '{}'".format(scope))
    if event_dates is not None:
        if len(event_dates) == 0:
            return pd.DataFrame(columns=model.columns)
        filters.append("event_date IN ('{}')".format("', '".join(event_dates)))
    if date_to is not None:
        filters.append("event_date < '{}'".format(pd.Timestamp(date_to).strftime('%Y-%m-%d')))

    return model.get_results(query=dict(filters=filters), formatted=True)


def save_house_effects_data(
    data: pd.DataFrame
) -> int:
    """
    Save the house effects data into the database, replacing the rows of the same elections. The table
    is created with its columns when it has none of them yet (it was created key-only and empty); if it
    already holds rows, the schema is extended with `Model.modify` (which rebuilds the table).

    Returns
    -------
    int
        Number of rows updated.
    """
    model = PollstersParties()
    if not model.table_exists():
        model.create(replace=False)
    elif set(model.columns) - set(model._dal.get_columns(model.table)):
        if model.get_results(query=dict(columns=list(model.key))).shape[0] == 0:
            model.create(replace=True)  # Empty key-only table: recreate it with the full schema
        else:
            warnings.warn('Extending the schema of {} (the table is rebuilt)'.format(model.table))
            model.modify()

    df = data.copy().reset_index(drop=True)
    df = df.loc[df['party_id'].notnull() & df['pollster_id'].notnull()]
    if df.shape[0] == 0:
        return 0

    # Use the model formatter to get the data in the correct types to be saved
    df = model.format_data(df, int_type='nullable', bin_type='nullable', sort=True)

    # Remove previous data for the same election events
    model.execute("DELETE FROM {} WHERE event_scope IN ('{}') AND event_date IN ('{}')".format(
        model.table,
        "', '".join(df.event_scope.unique()),
        "', '".join(df.event_date.dt.strftime('%Y-%m-%d').unique())
    ))

    return model.upsert(df)


def update_featured(
    scope: str,
    min_polls: int = 10,
    drop_mtypes: Optional[list[str]] = list(DROP_MTYPES),
    drop_ctypes: Optional[list[str]] = ['wban', 'exit']
) -> int:
    """
    Mark as `featured` the events of a scope with an official result and at least `min_polls` useful polls:
    computed (see `Computer.compute_weights`), with a positive overlap weight and not of the dropped
    methodology or context types. The other events of the scope are unmarked.

    Parameters
    ----------
    scope : str
        Election scope.
    min_polls : int, optional
        Useful polls needed.
    drop_mtypes : list of str, optional
        Methodology types that do not count (aggregators and online panels).
    drop_ctypes : list of str, optional
        Context types that do not count (polls within the ban and exit polls).

    Returns
    -------
    int
        Number of featured events of the scope.
    """
    model = Events()
    events = get_events(scope)

    polls = Polls().get_results(query=dict(filters=["event_scope = '{}'".format(scope)]), formatted=True)
    useful = polls.loc[
        polls['computed'].fillna(False).astype(bool) & (polls['weight_over'] > 0)
        & ~polls['mtype'].isin(drop_mtypes or []) & ~polls['ctype'].isin(drop_ctypes or [])
    ].groupby('event_date', observed=True).size()

    events['featured'] = events['has_results'] & (events['date'].map(useful).fillna(0) >= min_polls)
    model.upsert(model.format_data(events.drop(columns='has_results'), int_type='nullable', bin_type='nullable',
                                   sort=True))

    return int(events['featured'].sum())


def get_event_dates(
    scope: str = 'es',
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    skip: int = 0,
    min_polls: int = 0,
    featured: bool = False
) -> list[str]:
    """
    Get election dates from events or polls.

    Parameters
    ----------
    scope: str, default 'es'
        Election scope.
    date_from: str, default None
        Minimum election date.
    date_to: str, default None
        Maximum election date.
    skip: int, default 0
        Number of last elections to skip.
    min_polls: int, default 1
        Minimum number of polls for each event.
    featured: bool, default False
        Whether to include only featured events.

    Returns
    -------
    list
        Election dates.
    """
    filters = ["scope = '{}'".format(scope)]
    if featured:
        filters.append('featured IS TRUE')

    if min_polls > 0:
        dates = Polls().get_agg(query=dict(
            filters=[f.replace('scope', 'event_scope') for f in filters],
            groupby=['event_date'],
        ))
        dates = dates[dates['count'] >= min_polls].event_date
    else:
        dates = Events().get_agg(query=dict(
            filters=filters,
            groupby=['date'],
        )).date

    dates = dates.apply(pd.to_datetime).sort_values()

    if date_from is not None:
        dates = dates[dates >= date_from]
    if date_to is not None:
        dates = dates[dates <= date_to]
    if skip > 0:
        dates = dates.iloc[:-skip]

    dates = dates.dt.strftime('%Y-%m-%d').tolist()

    return dates


def get_next_event_date(
    scope: str = 'es',
    date_from: Optional[str] = None
) -> str:
    if date_from is None:
        date_from = 'now'
    date_from = pd.to_datetime(date_from).strftime('%Y-%m-%d')

    date = Events().get_var(query=dict(
        filters=[
            "scope = '{}'".format(scope),
            "date > '{}'".format(date_from)
        ]
    ))

    if date is not None:
        date = date.strftime('%Y-%m-%d')

    return date
