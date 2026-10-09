import pickle

import numpy as np
import pytest

from rgakit.background import background_correct
from rgakit.stack import SpectrumStack

PERIOD = 10.0
OPEN, CLOSE = 60.0, 210.0
BASE = np.array([1.0, 5.0, 2.0])
RISE = np.array([3.0, 1.0, 4.0])
TAU = 15.0


def _signal(t):
    on = ((t >= OPEN) & (t < CLOSE)).astype(float)
    tail = np.where(t >= CLOSE, np.exp(-(t - CLOSE) / TAU), 0.0)
    return on + tail


def _shutter():
    st = np.arange(0.0, 290.0, 0.5)
    return ((st >= OPEN) & (st < CLOSE)).astype(int), st


def _data(offset=None):
    time = np.arange(5.0, 285.0, PERIOD)
    off = np.zeros(3) if offset is None else offset
    tc = time[:, None] + off[None, :]
    pressure = BASE + RISE * _signal(tc)
    pressure[0] += 0.7
    return time, pressure


def _stack(offset=None):
    time, pressure = _data(offset)
    shutter, st = _shutter()
    return SpectrumStack(time, pressure, shutter=shutter, shutter_time=st,
                         channel_offset=offset)


def test_linear_is_default_and_unchanged():
    time, pressure = _data()
    shutter, st = _shutter()
    default, o, c = background_correct(time, pressure, shutter, st)
    explicit, _, _ = background_correct(time, pressure, shutter, st, method="linear")
    assert np.array_equal(default, explicit)

    bg = ((time >= o - 35) & (time <= o - 5)) | ((time >= c + 10) & (time <= c + 40))
    for j in range(3):
        coeffs = np.polyfit(time[bg], pressure[bg, j], 1)
        assert np.allclose(default[:, j], pressure[:, j] - np.polyval(coeffs, time))


def test_pre_method_recovers_rise_despite_tail_and_first_scan():
    time, pressure = _data()
    shutter, st = _shutter()
    corrected, o, c = background_correct(time, pressure, shutter, st, window=60.0,
                                         method="pre", skip_first=1)
    on = (time >= o) & (time < c)
    assert np.allclose(corrected[on].mean(axis=0), RISE)

    linear, _, _ = background_correct(time, pressure, shutter, st)
    assert np.all(linear[on].mean(axis=0) < RISE * 0.99)


def test_pre_method_order_one_fits_slope():
    time, pressure = _data()
    pressure = pressure + 0.01 * time[:, None]
    shutter, st = _shutter()
    corrected, o, c = background_correct(time, pressure, shutter, st, window=60.0,
                                         method="pre", order=1, skip_first=1)
    on = (time >= o) & (time < c)
    assert np.allclose(corrected[on].mean(axis=0), RISE)


def test_pre_method_needs_points():
    time, pressure = _data()
    shutter, st = _shutter()
    with pytest.raises(ValueError):
        background_correct(time, pressure, shutter, st, window=3.0, gap_before=50.0,
                           method="pre")


def test_unknown_method_raises():
    time, pressure = _data()
    shutter, st = _shutter()
    with pytest.raises(ValueError):
        background_correct(time, pressure, shutter, st, method="spline")


def test_channel_offset_used_for_windows_and_average():
    offset = np.array([0.0, 4.0, 8.0])
    stack = _stack(offset).background_correct(window=60.0, method="pre", skip_first=1)
    ms = stack.averaged()
    assert np.allclose(ms.intensity, RISE)

    naive = _stack().background_correct(window=60.0, method="pre", skip_first=1)
    assert np.allclose(naive.averaged().intensity, RISE)


def test_channel_offset_preserved_and_old_pickles_load():
    offset = np.array([0.0, 4.0, 8.0])
    stack = _stack(offset)
    corrected = stack.background_correct(window=60.0, method="pre")
    assert np.array_equal(corrected.channel_offset, offset)
    assert np.array_equal(corrected.slice(0, 100).channel_offset, offset)
    assert corrected._bg_off2 is None

    state = stack.__dict__.copy()
    del state["channel_offset"]
    old = SpectrumStack.__new__(SpectrumStack)
    old.__setstate__(state)
    assert old.channel_offset is None
    assert pickle.loads(pickle.dumps(old)).channel_offset is None


def test_channel_offset_length_checked():
    time, pressure = _data()
    with pytest.raises(ValueError):
        SpectrumStack(time, pressure, channel_offset=np.zeros(2))
