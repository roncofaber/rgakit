import numpy as np
import pytest

from rgakit.spectrum import MassSpectrum


def test_search_by_spectrum_elements_filters_before_truncation(fixture_db):
    query = MassSpectrum(np.array([16, 30, 50]), np.array([100.0, 100.0, 100.0]))

    unfiltered = fixture_db.search_by_spectrum(query, k=3, threshold=1e-9)
    assert {spec.name for spec, _ in unfiltered} == {"Methane", "Ethane", "Chloromethane"}

    filtered = fixture_db.search_by_spectrum(query, k=3, threshold=1e-9, elements={"C", "H"})
    names = {spec.name for spec, _ in filtered}
    assert names == {"Methane", "Ethane"}
    assert "Chloromethane" not in names


def test_search_by_spectrum_elements_still_returns_k_results(fixture_db):
    # With a pool of 3 and one excluded by elements=, k=2 must still come
    # back in full from the remaining 2, not be short because the
    # exclusion happened after a k-sized slice was already taken.
    query = MassSpectrum(np.array([16, 30, 50]), np.array([100.0, 100.0, 100.0]))
    filtered = fixture_db.search_by_spectrum(query, k=2, threshold=1e-9, elements={"C", "H"})
    assert len(filtered) == 2


def test_search_by_spectrum_elements_excludes_rows_with_no_formula_or_smiles(fixture_db):
    fixture_db.conn.execute(
        "UPDATE compounds SET molecular_formula = NULL, smiles = NULL "
        "WHERE inchikey = 'VNWKTOKETHGBQD-UHFFFAOYSA-N'"
    )
    fixture_db.conn.commit()

    query = MassSpectrum(np.array([16, 30, 50]), np.array([100.0, 100.0, 100.0]))
    filtered = fixture_db.search_by_spectrum(query, k=3, threshold=1e-9, elements={"C", "H"})
    names = {spec.name for spec, _ in filtered}
    assert names == {"Ethane"}


def test_search_by_spectrum_elements_uses_smiles_when_formula_missing(fixture_db):
    fixture_db.conn.execute(
        "UPDATE compounds SET molecular_formula = NULL "
        "WHERE inchikey = 'VNWKTOKETHGBQD-UHFFFAOYSA-N'"
    )
    fixture_db.conn.commit()
    query = MassSpectrum(np.array([16, 30, 50]), np.array([100.0, 100.0, 100.0]))
    filtered = fixture_db.search_by_spectrum(query, k=3, threshold=1e-9, elements={"C", "H"})
    assert {spec.name for spec, _ in filtered} == {"Methane", "Ethane"}


def test_get_returns_formula_in_metadata(fixture_db):
    spec = fixture_db.get(smiles="CC")
    assert spec.metadata["formula"] == "C2H6"


def _library_with_residual():
    from rgakit.library import SpectraLibrary
    from rgakit.result import FitResult

    grid = np.array([16, 30, 50])
    lib = SpectraLibrary([MassSpectrum(np.array([44]), np.array([100.0]), name="Other")])
    observed = np.array([100.0, 100.0, 100.0])
    result = FitResult(
        weights={"Other": 0.0},
        residual=1.0,
        fitted=np.zeros_like(observed),
        observed=observed,
        grid=grid,
        residual_spectrum=observed.copy(),
    )
    return lib, result


def test_suggest_compounds_elements_filters_before_truncation(fixture_db):
    lib, result = _library_with_residual()
    hits = lib.suggest_compounds(result, fixture_db, k=3, threshold=1e-9,
                                  min_overlap=0.0, elements={"C", "H"})
    names = {spec.name for spec, _ in hits}
    assert names == {"Methane", "Ethane"}


def test_suggest_compounds_raises_if_database_cannot_filter():
    class NoElementsDB:
        def search_by_spectrum(self, spectrum, k=10):
            return []

    lib, result = _library_with_residual()
    with pytest.raises(TypeError):
        lib.suggest_compounds(result, NoElementsDB(), elements={"C", "H"})
