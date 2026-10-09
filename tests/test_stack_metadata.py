from datetime import datetime
from types import SimpleNamespace

import numpy as np

from rgakit.stack import SpectrumStack


def _rga():
    tey_time = np.arange(0.0, 200.0, 1.0)
    shutter = ((tey_time >= 60) & (tey_time < 140)).astype(int)
    time = np.arange(0.0, 200.0, 10.0)
    return SimpleNamespace(
        time=time,
        mz=np.arange(1, 4),
        pressure=np.ones((len(time), 3)),
        tey_time=tey_time,
        tey_signal=np.linspace(1e-9, 2e-9, len(tey_time)),
        shutter=shutter,
        sample_name="TF000001",
        start_time=datetime(2026, 5, 14, 10, 30, 0),
        pd=90.2,
        dark_pd=-0.55,
        chamber_pressure=4.04e-7,
        x=39.5,
        y=-38.2,
        scan_settings={"scanspeed": 4.0},
    )


def test_from_rga_carries_tey_and_run_metadata():
    stack = SpectrumStack.from_rga(_rga())

    assert np.allclose(stack.tey, _rga().tey_signal)
    assert stack.metadata["start_time"] == "2026-05-14T10:30:00"
    assert stack.metadata["pd_ua"] == 90.2
    assert stack.metadata["dark_pd_ua"] == -0.55
    assert stack.metadata["chamber_torr"] == 4.04e-7
    assert stack.metadata["x"] == 39.5
    assert stack.metadata["scan_settings"] == {"scanspeed": 4.0}


def test_transforms_preserve_tey_and_metadata():
    stack = SpectrumStack.from_rga(_rga())

    corrected = stack.background_correct(window=20, gap_before=5, gap_after=5)
    sliced = corrected.slice(50, 150)

    for s in (corrected, sliced):
        assert s.metadata == stack.metadata
        assert np.allclose(s.tey, stack.tey)


def test_stack_without_metadata_defaults_to_empty():
    stack = SpectrumStack(np.arange(3.0), np.ones((3, 2)))

    assert stack.metadata == {}
    assert stack.tey is None
