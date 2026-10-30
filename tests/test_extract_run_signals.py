import numpy as np

from rgakit.background import extract_run_signals
from rgakit.stack import SpectrumStack

OPEN, CLOSE = 150.0, 350.0
STEP = 5.0
SIGMA = 0.5
WINDOW, GAP_BEFORE, SETTLE = 60.0, 10.0, 25.0


def _synthetic_pressure(time, rng):
    on = (time >= OPEN) & (time < CLOSE)
    col_a = 10.0 + STEP * on + rng.normal(scale=SIGMA, size=time.shape)
    col_b = 10.0 + rng.normal(scale=SIGMA, size=time.shape)
    return np.column_stack([col_a, col_b])


def test_extract_run_signals_recovers_known_step_and_se():
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 400.0, 1.0)
    pressure = _synthetic_pressure(time, rng)

    out = extract_run_signals(
        time, pressure,
        open_time=OPEN, close_time=CLOSE,
        window=WINDOW, gap_before=GAP_BEFORE, settle=SETTLE,
    )

    n_pre_true = int(np.sum((time >= OPEN - GAP_BEFORE - WINDOW) & (time <= OPEN - GAP_BEFORE)))
    n_on_true = int(np.sum((time >= OPEN + SETTLE) & (time <= CLOSE)))
    se_analytic = SIGMA * np.sqrt(1.0 / n_on_true + 1.0 / n_pre_true)

    # Channel A: known step.
    assert abs(out["rise"][0] - STEP) < 5 * se_analytic
    assert abs(out["se"][0] - se_analytic) / se_analytic < 0.20
    assert out["z"][0] > 10

    # Channel B: no step, should not trigger detection.
    assert abs(out["rise"][1]) < 5 * se_analytic
    assert abs(out["z"][1]) < 4


def test_extract_run_signals_wrapper_matches_function():
    rng = np.random.default_rng(1)
    time = np.arange(0.0, 400.0, 0.5)
    pressure = _synthetic_pressure(time, rng)

    shutter_time = np.arange(0.0, 400.0, 0.5)
    shutter = ((shutter_time >= OPEN) & (shutter_time < CLOSE)).astype(int)

    stack = SpectrumStack(time, pressure, mz=np.array([1, 2]),
                          shutter=shutter, shutter_time=shutter_time)

    out_wrapper = stack.extract_run_signals(window=WINDOW, gap_before=GAP_BEFORE, settle=SETTLE)
    ot, ct = stack.open_window
    out_direct = extract_run_signals(
        time, pressure,
        open_time=ot, close_time=ct,
        window=WINDOW, gap_before=GAP_BEFORE, settle=SETTLE,
        channel_offset=stack.channel_offset,
    )

    for key in ("rise", "se", "z", "level", "noise"):
        assert np.allclose(out_wrapper[key], out_direct[key], equal_nan=True)

    assert out_wrapper["z"][0] > 10
    assert abs(out_wrapper["z"][1]) < 4


def test_extract_run_signals_default_window_uses_every_pre_scan():
    time = np.arange(0.0, 400.0, 1.0)
    rng = np.random.default_rng(2)
    pressure = _synthetic_pressure(time, rng)
    skip_first = 1

    out = extract_run_signals(
        time, pressure,
        open_time=OPEN, close_time=CLOSE,
        gap_before=0.5, skip_first=skip_first,
    )

    n_pre_expected = int(np.sum((time >= time[skip_first]) & (time <= OPEN - 0.5)))
    assert out["n_pre"][0] == n_pre_expected
    assert out["n_pre"][0] > 100


def test_extract_run_signals_empty_pre_window_is_nan_not_hidden():
    time = np.arange(0.0, 50.0, 1.0)
    pressure = np.ones((len(time), 1)) * 2.0
    out = extract_run_signals(
        time, pressure,
        open_time=5.0, close_time=40.0,
        window=30.0, gap_before=10.0, settle=5.0,
    )
    assert out["n_pre"][0] == 0
    assert np.isnan(out["rise"][0])
    assert np.isnan(out["se"][0])
    assert np.isnan(out["z"][0])


def test_slice_keeps_raw_pressure_for_run_signals():
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 400.0, 0.5)
    shutter = ((time >= OPEN) & (time < CLOSE)).astype(int)
    stack = SpectrumStack(time, _synthetic_pressure(time, rng), mz=np.array([1, 2]),
                          shutter=shutter, shutter_time=time)
    corrected = stack.background_correct()
    full = corrected.extract_run_signals()
    sliced = corrected.slice(0.0, 400.0).extract_run_signals()
    np.testing.assert_allclose(sliced["level"], full["level"])
    assert sliced["level"][0] > 5.0
