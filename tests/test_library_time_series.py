import numpy as np

from rgakit.library import SpectraLibrary
from rgakit.spectrum import MassSpectrum
from rgakit.stack import SpectrumStack


def _single_compound_stack(n_times=6):
    mz = np.arange(10, 20)
    lib = SpectraLibrary([
        MassSpectrum(mz, np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.3]), name="X"),
    ])
    rng = np.random.default_rng(0)
    pressure = rng.normal(scale=0.01, size=(n_times, len(mz)))
    # Over-subtracted scan: all channels negative after background correction.
    pressure[2] = -np.abs(pressure[2]) - 0.05
    stack = SpectrumStack(time=np.arange(n_times, dtype=float), pressure=pressure, mz=mz)
    return lib, stack


def test_fit_time_series_clips_negative_scan_to_zero_weight():
    lib, stack = _single_compound_stack()
    result = lib.fit_time_series(stack)
    assert result.weights["X"][2] == 0.0


def test_fit_time_series_handles_all_zero_norm_scan():
    lib, stack = _single_compound_stack()
    stack.pressure[3] = 0.0
    result = lib.fit_time_series(stack)
    assert np.isfinite(result.weights["X"][3])


def test_spectra_library_contains_checks_name_only():
    spec = MassSpectrum(np.arange(10, 15), np.ones(5), name="X")
    lib = SpectraLibrary([spec])

    assert "X" in lib
    assert "Y" not in lib
    # Semantics changed from the previous __iter__-based fallback: membership
    # is by name, not by MassSpectrum identity or equality.
    assert spec not in lib
