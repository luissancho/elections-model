"""Tests de `mtpy/core/utils/serialize.py`: conversiones a JSON de valores numpy y pandas."""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from mtpy.core.utils.serialize import json_default, nan_to_none


def test_longdouble_becomes_a_native_float():
    value = json_default(np.longdouble('1.5'))
    assert type(value) is float and value == 1.5
    assert json_default(np.longdouble('nan')) is None
    assert json.dumps({'x': np.longdouble('2.25')}, default=json_default) == '{"x": 2.25}'


def test_numpy_scalars_arrays_and_dates():
    assert json_default(np.int64(3)) == 3 and type(json_default(np.int64(3))) is int
    assert json_default(np.array([1.0, np.nan])) == [1.0, None]
    assert json_default(pd.Timestamp('2026-10-08')) == '2026-10-08'
    assert json_default(pd.Timestamp('2026-10-08 12:30')) == '2026-10-08T12:30:00'
    assert json_default(pd.NaT) is None and json_default(pd.NA) is None
    assert json_default(datetime.date(2026, 10, 8)) == '2026-10-08'
    assert sorted(json_default({1, 2})) == [1, 2]


def test_nan_to_none_is_recursive_and_timedelta_is_rejected():
    assert nan_to_none([1.0, [float('nan'), 2.0]]) == [1.0, [None, 2.0]]
    with pytest.raises(TypeError):
        json_default(np.timedelta64(1, 'D'))


def test_core_api_still_exports_json_default():
    from mtpy.core import api

    assert api.json_default is json_default
