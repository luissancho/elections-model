"""
Public rubric of the prior quality of the pollsters.

Reads and validates the two versioned files in `data/`: the catalogue of criteria with their levels and the scale
from points to quality (`pollsters-quality-criteria.json`), and the evaluation of each pollster against it
(`pollsters-quality.csv`). It depends only on the file-system object of the app, never on the database or the models.
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
        criteria in the order of the file.

    Raises
    ------
    ValueError
        If the scale is not made of positive integers, or a criterion has no name, a maximum below 1 or levels
        different from "0" to "max".
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
    return criteria


def criteria_keys(criteria: dict) -> list:
    """Keys of the criteria, in the order of the catalogue."""
    return list(criteria['criteria'])


def max_points(criteria: dict) -> int:
    """Sum of the maximum points of every criterion."""
    return sum(entry['max'] for entry in criteria['criteria'].values())


def quality_from_points(points, scale: dict) -> float:
    """Quality of a pollster with `points`: `max(floor, per_point * points)`."""
    return max(scale['floor'], scale['per_point'] * points)


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
        One row per evaluated pollster with the `COLUMNS`, `checked_at` as datetime, plus `points` (int, sum of the
        criteria) and `quality` (float).

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
    keys = criteria_keys(criteria)
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
        maximum = criteria['criteria'][key]['max']
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
    frame['points'] = frame[keys].sum(axis=1).astype(int)
    frame['quality'] = frame['points'].map(lambda points: float(quality_from_points(points, criteria['scale'])))
    return frame.reset_index(drop=True)


def pollsters_quality_values(pollsters: pd.DataFrame, quality: pd.DataFrame, scale: dict) -> pd.Series:
    """
    Quality of every pollster of the catalogue.

    Parameters
    ----------
    pollsters : pd.DataFrame
        The catalogue, with `id` and `name`.
    quality : pd.DataFrame
        The evaluations from `read_quality()`.
    scale : dict
        The scale of the catalogue; its `floor` is the quality of the pollsters not evaluated.

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
    values = quality.set_index('pollster_id')['quality']
    return pollsters['id'].map(values).fillna(float(scale['floor'])).astype(float)
