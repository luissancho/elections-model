import datetime
import math

import numpy as np
import pandas as pd


def nan_to_none(value):
    """
    Replace the ``NaN`` floats of a value by ``None``, descending into nested lists.

    Parameters
    ----------
    value : Any
        A scalar or a (possibly nested) list, such as the output of ``numpy.ndarray.tolist``.

    Returns
    -------
    Any
        The same value with every ``float`` NaN replaced by ``None``.
    """
    if isinstance(value, list):
        return [nan_to_none(item) for item in value]

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def json_default(obj):
    """
    Convert the objects that ``json`` cannot serialise natively (``default`` of ``json.dumps``).

    Missing values (``pd.NaT``, ``pd.NA`` and the NaN inside numpy arrays and scalars) become
    ``None``, i.e. JSON ``null``. The order of the checks matters: ``pd.NaT`` is a ``datetime``
    subclass, so it is tested before the dates. A NaN in a native ``float`` (``np.float64``
    included) never reaches this function; with ``allow_nan=False`` ``json.dumps`` raises
    ``ValueError`` for it.

    Parameters
    ----------
    obj : Any
        The object that ``json`` does not know how to serialise.

    Returns
    -------
    Any
        A JSON-compatible value:

        - ``None`` for ``pd.NaT`` and ``pd.NA``.
        - A list, with the NaN replaced by ``None`` at any depth, for a ``numpy.ndarray``.
          ``datetime64`` arrays are cast to microseconds first, so their elements are dates
          (``NaT`` is ``None``) instead of nanosecond integers.
        - The equivalent Python scalar, ``None`` if it is a NaN, for a ``numpy.generic``
          (``datetime64`` is cast to microseconds first, as in arrays).
        - A ``YYYY-MM-DD`` string for a ``datetime`` (``pd.Timestamp`` included) at midnight
          without timezone, an ISO 8601 string for any other.
        - An ISO 8601 string for a ``date``.
        - A list for a ``set``.

    Raises
    ------
    TypeError
        If the type of the object is not supported, ``timedelta64`` (scalars and arrays)
        included.
    """
    if obj is pd.NaT or obj is pd.NA:
        return None

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == 'm':
            raise TypeError('timedelta64 values are not JSON serializable')

        if obj.dtype.kind == 'M':
            obj = obj.astype('datetime64[us]')

        return nan_to_none(obj.tolist())

    if isinstance(obj, np.generic):
        if isinstance(obj, np.timedelta64):
            raise TypeError('timedelta64 values are not JSON serializable')

        if isinstance(obj, np.datetime64):
            obj = obj.astype('datetime64[us]')

        if isinstance(obj, np.floating):
            return nan_to_none(float(obj))

        return nan_to_none(obj.item())

    if isinstance(obj, datetime.datetime):
        if obj.time() == datetime.time() and obj.tzinfo is None:
            return obj.strftime('%Y-%m-%d')

        return obj.isoformat()

    if isinstance(obj, datetime.date):
        return obj.isoformat()

    if isinstance(obj, set):
        return list(obj)

    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
