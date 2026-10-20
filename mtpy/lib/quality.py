"""
Public rubric of the prior quality of the pollsters.

Reads and validates the two versioned files in `data/`: the catalogue of criteria with their levels and the scale
from points to quality (`pollsters-quality-criteria.json`), and the evaluation of each pollster against it
(`pollsters-quality.csv`). It depends only on the file-system object of the app, never on the database or the models:
the general elections of each pollster (criterion "implantation") arrive as an argument.
"""
import io
import json
import warnings

import pandas as pd

from datetime import date
from typing import Optional

from ..core.app import App

CRITERIA_FILE = 'pollsters-quality-criteria.json'
QUALITY_FILE = 'pollsters-quality.csv'
STATUSES = ('borrador', 'revisada')
COMPUTED = ('general',)
# CSV columns that are not criteria but are validated (0-1, with source if 1): the evidence of the condition of the
# transparency (identifiable entity) and of the level 2 of the implantation (standards)
EVIDENCE_KEYS = ('identity', 'standards')
COLUMNS = (
    'pollster_id', 'pollster',
    'fieldwork', 'fieldwork_source',
    'transparency', 'transparency_source',
    'independence', 'independence_source',
    'identity', 'identity_source',
    'standards', 'standards_source',
    'notes', 'status', 'checked_at'
)


def load_criteria() -> dict:
    """
    Read and validate the catalogue of criteria (`data/pollsters-quality-criteria.json`).

    Returns
    -------
    dict
        `{'scale': {'per_point': int, 'floor': int}, 'criteria': {key: {'name', 'max', 'levels'}}}`, with the
        criteria in the order of the file. A criterion may carry a `note` (text) and `computed` (`'general'`: not
        scored in the CSV but derived from the general elections polled).

    Raises
    ------
    ValueError
        If the scale is not made of positive integers, or a criterion has no name, a maximum below 1, levels
        different from "0" to "max", a `note` that is not text or an unknown `computed`.
    """
    criteria = json.loads(App.get_().data.read(CRITERIA_FILE))
    scale = criteria.get('scale', {})
    for name in ('per_point', 'floor'):
        value = scale.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{CRITERIA_FILE}: scale.{name} must be a positive integer')
    if not criteria.get('criteria'):
        raise ValueError(f'{CRITERIA_FILE}: no criteria')
    for key, entry in criteria['criteria'].items():
        if not entry.get('name'):
            raise ValueError(f'{CRITERIA_FILE}: criterion {key} has no name')
        maximum = entry.get('max')
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError(f'{CRITERIA_FILE}: criterion {key} needs an integer max of at least 1')
        if sorted(entry.get('levels', {})) != [str(level) for level in range(maximum + 1)]:
            raise ValueError(f'{CRITERIA_FILE}: criterion {key} must have the levels 0 to {maximum}')
        if 'computed' in entry and entry['computed'] not in COMPUTED:
            raise ValueError(f'{CRITERIA_FILE}: criterion {key} has an unknown computed value, use one of {COMPUTED}')
        if 'note' in entry and not isinstance(entry['note'], str):
            raise ValueError(f'{CRITERIA_FILE}: criterion {key} needs a text note')
    return criteria


def criteria_keys(criteria: dict) -> list:
    """Keys of the criteria, in the order of the catalogue."""
    return list(criteria['criteria'])


def evaluated_keys(criteria: dict) -> list:
    """Keys of the criteria the CSV scores (the ones without `computed`), in the order of the catalogue."""
    return [key for key, entry in criteria['criteria'].items() if not entry.get('computed')]


def max_points(criteria: dict) -> int:
    """Sum of the maximum points of every criterion."""
    return sum(entry['max'] for entry in criteria['criteria'].values())


def quality_from_points(points, scale: dict) -> float:
    """Quality of a pollster with `points`: `max(floor, per_point * points)`."""
    return max(scale['floor'], scale['per_point'] * points)


def manual_points(fieldwork, transparency, independence, identity):
    """
    Points scored by hand: `fieldwork + transparency (only with an identifiable entity) + independence`.

    Parameters
    ----------
    fieldwork, transparency, independence, identity : int or pd.Series
        The columns of the evaluation; `identity` is 1 with an identifiable entity. Scalars or Series.

    Returns
    -------
    int or pd.Series
        From 0 to 8.
    """
    return fieldwork + transparency * (identity == 1) + independence


def implantation_points(n_generals, standards):
    """
    Points of the criterion "implantation": 0 without general elections polled, else `1 + standards`.

    Parameters
    ----------
    n_generals : int or pd.Series
        General elections polled before the date of the quality.
    standards : int or pd.Series
        1 when the pollster is a member of Insights+Analytics España or ESOMAR, or ISO 20252 certified.

    Returns
    -------
    int or pd.Series
        From 0 to 2.
    """
    return (n_generals > 0) * (1 + standards)


def count_generals(general_events: pd.DataFrame, before: Optional[pd.Timestamp] = None) -> pd.Series:
    """
    General elections each pollster polled before a date.

    Parameters
    ----------
    general_events : pd.DataFrame
        `pollster_id` and `event_date`, one row per pollster and general election (`data.get_general_events()`).
    before : pd.Timestamp, optional
        Only the elections strictly before this date count; all of them with `None` or `NaT`.

    Returns
    -------
    pd.Series
        Distinct election dates by `pollster_id`; the pollsters without any are absent (fill with 0).
    """
    events = general_events[['pollster_id', 'event_date']].drop_duplicates()
    events['event_date'] = pd.to_datetime(events['event_date'])
    if before is not None and not pd.isna(before):
        events = events.loc[events['event_date'] < pd.Timestamp(before)]
    counts = events.groupby('pollster_id')['event_date'].nunique()
    counts.name = 'num_generals'
    return counts


def prior_quality(pollster_ids, evaluations: pd.DataFrame, scale: dict, n_generals: pd.Series) -> pd.DataFrame:
    """
    Prior quality of each pollster at the date `n_generals` was counted at.

    Parameters
    ----------
    pollster_ids : iterable of int
        The pollsters to value; the ones not in `evaluations` have no manual points and no standards.
    evaluations : pd.DataFrame
        `read_quality()` (`pollster_id`, `fieldwork`, `transparency`, `independence`, `identity`, `standards`).
    scale : dict
        The scale of the catalogue.
    n_generals : pd.Series
        `count_generals()`; the pollsters it lacks have none.

    Returns
    -------
    pd.DataFrame
        By `pollster_id`: `num_generals`, `manual`, `implantation`, `points` (their sum) and `quality`
        (`max(floor, per_point * points)`).
    """
    index = pd.Index(list(pollster_ids), name='pollster_id')
    evaluated = evaluations.set_index('pollster_id').reindex(index)
    columns = ['fieldwork', 'transparency', 'independence', 'identity', 'standards']
    evaluated = evaluated[columns].fillna(0).astype(int)
    out = pd.DataFrame(index=index)
    out['num_generals'] = n_generals.reindex(index).fillna(0).astype(int)
    out['manual'] = manual_points(evaluated['fieldwork'], evaluated['transparency'], evaluated['independence'],
                                  evaluated['identity']).astype(int)
    out['implantation'] = implantation_points(out['num_generals'], evaluated['standards']).astype(int)
    out['points'] = out['manual'] + out['implantation']
    out['quality'] = out['points'].map(lambda points: float(quality_from_points(points, scale)))
    return out


def split_points(record: dict) -> tuple:
    """
    Split the total points of an evaluation of the replica into the manual and the computed part.

    Parameters
    ----------
    record : dict
        A row of `pollsters_quality` (its `points` is the total, implantation included).

    Returns
    -------
    tuple
        `(manual, implantation, general)`: the points scored by hand (a void transparency counts 0), the rest, and
        whether the rest is above 0, that is, a general election was polled.
    """
    manual = int(manual_points(int(record['fieldwork']), int(record['transparency']), int(record['independence']),
                               int(record['identity'])))
    implantation = int(record['points']) - manual
    return manual, implantation, implantation > 0


def quality_breakdown(record: Optional[dict], criteria: dict) -> dict:
    """
    Split an evaluation of the replica into its manual and computed parts.

    Parameters
    ----------
    record : dict or None
        A row of `pollsters_quality` (its `points` is the total of the day it was replicated, implantation
        included); `None` when the pollster is not evaluated.
    criteria : dict
        The catalogue from `load_criteria()`.

    Returns
    -------
    dict
        `rows` (one `{'key', 'name', 'points', 'max', 'level', 'source'}` per scored criterion, `points` being the
        ones that count: 0 for a void transparency), `manual`, `implantation`, `general` (True when the
        implantation is above 0, that is, a general election was polled) and `transparency_void` (transparency
        scored without an identifiable entity, so it counts 0). Empty and 0 without a record.
    """
    out = {'rows': [], 'manual': 0, 'implantation': 0, 'general': False, 'transparency_void': False}
    if not record:
        return out
    void = int(record['identity']) != 1 and int(record['transparency']) > 0
    for key in evaluated_keys(criteria):
        points = int(record[key])
        if key == 'transparency' and void:
            points = 0
        out['rows'].append({'key': key, 'name': criteria['criteria'][key]['name'], 'points': points,
                            'max': criteria['criteria'][key]['max'], 'level': level_text(criteria, key, int(record[key])),
                            'source': record.get(f'{key}_source') or ''})
    out['manual'], out['implantation'], out['general'] = split_points(record)
    out['transparency_void'] = void
    return out


def level_text(criteria: dict, key: str, points: int) -> str:
    """Description of the level `points` of the criterion `key`."""
    return criteria['criteria'][key]['levels'][str(points)]


def read_quality(criteria: Optional[dict] = None, today: Optional[date] = None) -> pd.DataFrame:
    """
    Read and validate the evaluations (`data/pollsters-quality.csv`).

    Parameters
    ----------
    criteria : dict, optional
        The catalogue from `load_criteria()`; read if missing.
    today : date, optional
        Latest admitted `checked_at`; today by default.

    Returns
    -------
    pd.DataFrame
        One row per evaluated pollster with the `COLUMNS`, `checked_at` as datetime, plus `manual` (int, the points
        scored by hand: see `manual_points`). The points and the quality depend on the date: see `prior_quality`.

    Raises
    ------
    ValueError
        With the offending column in the message: wrong columns, repeated `pollster_id`, points out of range or
        without source, unknown `status`, or a `checked_at` that is not an ISO date or lies in the future.
    """
    criteria = criteria or load_criteria()
    today = today or date.today()
    frame = pd.read_csv(io.StringIO(App.get_().data.read(QUALITY_FILE)), dtype=str, keep_default_na=False)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f'{QUALITY_FILE}: columns must be {", ".join(COLUMNS)}')
    keys = evaluated_keys(criteria) + list(EVIDENCE_KEYS)
    for key in keys:
        try:
            frame[key] = frame[key].astype(int)
        except ValueError:
            raise ValueError(f'{QUALITY_FILE}: {key} must be an integer') from None
    frame['pollster_id'] = frame['pollster_id'].astype(int)
    repeated = frame.loc[frame['pollster_id'].duplicated(), 'pollster_id'].tolist()
    if repeated:
        raise ValueError(f'{QUALITY_FILE}: pollster_id repeated: {repeated}')
    for key in keys:
        maximum = criteria['criteria'][key]['max'] if key in criteria['criteria'] else 1
        wrong = frame.loc[~frame[key].between(0, maximum), 'pollster'].tolist()
        if wrong:
            raise ValueError(f'{QUALITY_FILE}: {key} must be between 0 and {maximum}: {wrong}')
        unsourced = frame.loc[(frame[key] > 0) & (frame[key + '_source'].str.strip() == ''), 'pollster'].tolist()
        if unsourced:
            raise ValueError(f'{QUALITY_FILE}: {key}_source is empty with points: {unsourced}')
    unknown = frame.loc[~frame['status'].isin(STATUSES), 'status'].tolist()
    if unknown:
        raise ValueError(f'{QUALITY_FILE}: status must be one of {STATUSES}: {unknown}')
    try:
        frame['checked_at'] = pd.to_datetime(frame['checked_at'], format='%Y-%m-%d')
    except ValueError:
        raise ValueError(f'{QUALITY_FILE}: checked_at must be a date as YYYY-MM-DD') from None
    future = frame.loc[frame['checked_at'] > pd.Timestamp(today), 'pollster'].tolist()
    if future:
        raise ValueError(f'{QUALITY_FILE}: checked_at is in the future: {future}')
    frame['manual'] = manual_points(frame['fieldwork'], frame['transparency'], frame['independence'],
                                    frame['identity']).astype(int)
    return frame.reset_index(drop=True)


def pollsters_quality_values(
    pollsters: pd.DataFrame,
    quality: pd.DataFrame,
    scale: dict,
    n_generals: pd.Series
) -> pd.Series:
    """
    Quality of every pollster of the catalogue, at the date `n_generals` was counted at.

    Parameters
    ----------
    pollsters : pd.DataFrame
        The catalogue, with `id` and `name`.
    quality : pd.DataFrame
        The evaluations from `read_quality()`.
    scale : dict
        The scale of the catalogue; its `floor` is the quality of the pollsters not evaluated or without points.
    n_generals : pd.Series
        `count_generals()`: general elections polled by `pollster_id`.

    Returns
    -------
    pd.Series
        Aligned with the index of `pollsters`.

    Raises
    ------
    ValueError
        If the evaluations include ids that are not in `pollsters`.

    Warns
    -----
    UserWarning
        For each evaluation whose `pollster` differs from the `name` in `pollsters`.
    """
    extra = sorted(set(quality['pollster_id']) - set(pollsters['id']))
    if extra:
        raise ValueError(f'{QUALITY_FILE}: pollster_id not in the pollsters: {extra}')
    names = pollsters.set_index('id')['name']
    for row in quality.itertuples():
        if names[row.pollster_id] != row.pollster:
            warnings.warn(f'{QUALITY_FILE}: pollster {row.pollster_id} is "{row.pollster}" in the file '
                          f'but "{names[row.pollster_id]}" in the database', UserWarning, stacklevel=2)
    values = prior_quality(pollsters['id'], quality, scale, n_generals)['quality']
    return pollsters['id'].map(values).astype(float)
