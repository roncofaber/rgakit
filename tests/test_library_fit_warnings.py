import logging

import numpy as np

from rgakit.library import SpectraLibrary
from rgakit.spectrum import MassSpectrum


def test_fit_logs_compounds_with_no_peaks_on_grid(caplog):
    lib = SpectraLibrary([
        MassSpectrum(np.array([2]), np.array([1.0]), name="Hydrogen"),
        MassSpectrum(np.array([12, 13, 14]), np.array([1.0, 0.5, 0.2]), name="Methane"),
    ])
    # Fitting grid starts at m/z 12, so Hydrogen (only m/z 2) has no peaks on it.
    with caplog.at_level(logging.INFO, logger="rgakit.library"):
        lib.fit(np.array([12, 13, 14]), np.array([10.0, 5.0, 2.0]))

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("Hydrogen" in m and "no peaks on the fitting grid" in m for m in info_messages)
    assert not any("Methane" in m and "no peaks on the fitting grid" in m for m in info_messages)


def test_fit_does_not_log_when_every_compound_has_grid_coverage(caplog):
    lib = SpectraLibrary([
        MassSpectrum(np.array([12, 13, 14]), np.array([1.0, 0.5, 0.2]), name="Methane"),
        MassSpectrum(np.array([12, 16]), np.array([1.0, 0.3]), name="Other"),
    ])
    with caplog.at_level(logging.INFO, logger="rgakit.library"):
        lib.fit(np.array([12, 13, 14]), np.array([10.0, 5.0, 2.0]))

    assert not any("no peaks on the fitting grid" in r.message for r in caplog.records)
