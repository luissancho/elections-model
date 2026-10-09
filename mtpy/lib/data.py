import json
import pandas as pd
import warnings

from typing import Any, Literal, Mapping, Optional

from ..core.app import App
from ..core.worker import Model
from ..models.elections import (
    Districts, Drift, Events, EventsData, EventsResults, Polls, PollsResults,
    Pollsters, PollstersHerding, PollstersParties, PollstersRatings, Parties, Scopes
)


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
    Replicate the versioned catalogues of scopes and districts into the database. The tables are created
    when missing (never replaced) and their rows upserted by key.

    Returns
    -------
    dict
        Rows written to `scopes` and `districts`.
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

def get_event_params(
    scope: str,
    event_dates: Optional[list[str] | str] = None,
    path: Optional[str] = None
) -> dict:
    is_single = False
    if event_dates is None:
        event_dates = get_event_dates(scope=scope)
    elif isinstance(event_dates, str):
        is_single = True
        event_dates = [event_dates]

    if path is not None:
        params = json.loads(
            App.get_().data.read('params.json')
        ).get(scope, {})

        event_params = {dt: params[dt] if dt in params else {} for dt in event_dates}
    else:
        event_params = {dt: {} for dt in event_dates}

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

    for dt in event_dates:
        params = event_params[dt]

        # Sorted index: `.loc[(0, ...)]` on the unsorted, non-unique (days, pollster) index
        # takes pandas' linear path and raises a PerformanceWarning per lookup.
        df = get_event_dmat(scope, dt, polls, events, parties).sort_index()

        all_parties = df.columns.tolist()
        final_parties = df.loc[(0, 'result')].squeeze().dropna().index.tolist()
        if len(final_parties) == 0:
            final_parties = all_parties

        df_count = df.loc[(0, 'count')].squeeze()
        main_parties = [p for p in final_parties if p in df_count[df_count > 95].index.tolist()]

        new_parties = []
        if all_event_dates.index(dt) > 0:
            dprev = events.loc[
                all_event_dates[all_event_dates.index(dt) - 1]
            ].loc[[p for p in parties['name'] if p in events.columns]].dropna()
            new_parties = [p for p in final_parties if p not in dprev.index.tolist()]

        bmaps = {
            'blocks': {'Derecha': [], 'Izquierda': [], 'Regionalista': [], 'Separatista': []},
            'vs': {'Derecha': [], 'Izquierda': []}
        }

        for block, names in parties.groupby('block', observed=True)['name'].apply(list).to_dict().items():
            block_parties = [p for p in names if p in all_parties]
            if len(block_parties) > 0 and block in bmaps['blocks']:
                bmaps['blocks'][block].extend(block_parties)
            
            vs_parties = [p for p in names if p in main_parties]
            if len(vs_parties) > 0 and block in bmaps['vs']:
                bmaps['vs'][block].extend(vs_parties)

        smap = {
            'parties': {
                'event': final_parties,
                'polls': all_parties
            },
            'bmaps': {
                'main': main_parties,
                'blocks': bmaps['blocks'],
                'vs': bmaps['vs']
            },
            'smap': {i: [] for i in new_parties}
        }

        event_params[dt] = smap | params

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
        pollsters[['id', 'name']], left_on='pollster_id', right_on='id', how='left'
    ).rename(columns={'name': 'pollster'})[
        keys + ['pollster'] + [col for col in ratings.columns if col not in keys]
    ].sort_values(keys, ignore_index=True)

    if is_single:
        ratings = ratings.loc[ratings['event_date'] == event_dates[0]]

    return ratings


def get_parties() -> pd.DataFrame:
    parties = Parties().get_results(formatted=True)
    parties = parties.rename(index={0: parties.shape[0]}).sort_index(ignore_index=True)

    return parties


def get_pollsters() -> pd.DataFrame:
    pollsters = Pollsters().get_results(formatted=True)

    return pollsters


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
    dp['rating'] = dp['id'].map(rlast).fillna(0).astype(int)

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
    drop_mtypes: Optional[list[str]] = ['aggr', 'online'],
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
    events = model.get_results(query=dict(filters=["scope = '{}'".format(scope)]), formatted=True)

    data = EventsData().get_results(query=dict(
        filters=["scope = '{}'".format(scope), 'region_id = 0', 'votes IS NOT NULL']
    ), formatted=True)

    polls = Polls().get_results(query=dict(filters=["event_scope = '{}'".format(scope)]), formatted=True)
    useful = polls.loc[
        polls['computed'].fillna(False).astype(bool) & (polls['weight_over'] > 0)
        & ~polls['mtype'].isin(drop_mtypes or []) & ~polls['ctype'].isin(drop_ctypes or [])
    ].groupby('event_date', observed=True).size()

    events['featured'] = events['date'].isin(data['date']) & (events['date'].map(useful).fillna(0) >= min_polls)
    model.upsert(model.format_data(events, int_type='nullable', bin_type='nullable', sort=True))

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
