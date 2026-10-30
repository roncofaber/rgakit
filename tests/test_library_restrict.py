import numpy as np

from rgakit.library import SpectraLibrary
from rgakit.spectrum import MassSpectrum


def _spec(name, **meta):
    return MassSpectrum(np.array([28, 44]), np.array([100.0, 50.0]), name=name, metadata=meta)


def test_restrict_elements_drops_disallowed_formula():
    lib = SpectraLibrary([
        _spec("Ethane", formula="C2H6"),
        _spec("Chloromethane", formula="CH3Cl"),
    ])
    restricted = lib.restrict_elements({"C", "H"})
    assert restricted.names() == ["Ethane"]


def test_restrict_elements_computes_formula_from_smiles_when_absent():
    lib = SpectraLibrary([
        _spec("Ethane", smiles="CC"),
        _spec("Chloroethane", smiles="CCCl"),
    ])
    restricted = lib.restrict_elements({"C", "H"})
    assert restricted.names() == ["Ethane"]


def test_restrict_elements_keeps_unresolvable_spectrum_by_default():
    lib = SpectraLibrary([
        _spec("Ethane", formula="C2H6"),
        _spec("Unknown"),  # no formula, no smiles
    ])
    restricted = lib.restrict_elements({"C", "H"})
    assert set(restricted.names()) == {"Ethane", "Unknown"}


def test_restrict_elements_strict_drops_unresolvable_spectrum():
    lib = SpectraLibrary([
        _spec("Ethane", formula="C2H6"),
        _spec("Unknown"),
    ])
    restricted = lib.restrict_elements({"C", "H"}, strict=True)
    assert restricted.names() == ["Ethane"]


def test_restrict_elements_exempts_environment_spectra_by_default():
    lib = SpectraLibrary([
        _spec("Ethane", formula="C2H6"),
        _spec("Argon", formula="Ar", environment=True, environment_set="chamber_gases"),
    ])
    restricted = lib.restrict_elements({"C", "H"})
    assert set(restricted.names()) == {"Ethane", "Argon"}


def test_restrict_elements_exempt_environment_false_still_restricts():
    lib = SpectraLibrary([
        _spec("Ethane", formula="C2H6"),
        _spec("Argon", formula="Ar", environment=True, environment_set="chamber_gases"),
    ])
    restricted = lib.restrict_elements({"C", "H"}, exempt_environment=False)
    assert restricted.names() == ["Ethane"]
