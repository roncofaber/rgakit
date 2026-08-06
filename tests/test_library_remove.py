import numpy as np
import pytest

from rgakit.spectrum import MassSpectrum
from rgakit.library import SpectraLibrary


def _spec(name, inchikey):
    return MassSpectrum(
        mz=np.array([28, 44]),
        intensity=np.array([1.0, 0.5]),
        name=name,
        metadata={"inchikey": inchikey},
    )


def test_remove_by_inchikey_removes_matching_spectrum():
    lib = SpectraLibrary([
        _spec("Nitrogen", "IJGRMHOSHXDMSA-UHFFFAOYSA-N"),
        _spec("Carbon dioxide", "CURLTUGMZLYLDI-UHFFFAOYSA-N"),
    ])

    removed = lib.remove_by_inchikey("CURLTUGMZLYLDI-UHFFFAOYSA-N")

    assert removed is True
    assert "Carbon dioxide" not in lib.names()
    assert "Nitrogen" in lib.names()
    assert len(lib) == 1


def test_remove_by_inchikey_returns_false_when_not_found():
    lib = SpectraLibrary([_spec("Nitrogen", "IJGRMHOSHXDMSA-UHFFFAOYSA-N")])

    removed = lib.remove_by_inchikey("DOES-NOT-EXIST")

    assert removed is False
    assert len(lib) == 1


def test_remove_by_inchikey_ignores_spectra_without_metadata():
    spec = MassSpectrum(mz=np.array([28]), intensity=np.array([1.0]), name="Unknown")
    lib = SpectraLibrary([spec])

    removed = lib.remove_by_inchikey("IJGRMHOSHXDMSA-UHFFFAOYSA-N")

    assert removed is False
    assert len(lib) == 1
