import pytest

from rgakit.environment import (
    ENVIRONMENT_SETS,
    allowed_elements_from_ion_pair,
    build_environment_library,
)
from rgakit.review import DEFAULT_BACKGROUND



def test_environment_sets_are_named_consistently():
    for set_name, entry in ENVIRONMENT_SETS.items():
        assert len(entry["smiles"]) == len(entry["names"]), set_name
        assert entry["label"]


def test_chamber_gases_includes_ch4_and_nh3_for_default_background_compat():
    # review.DEFAULT_BACKGROUND predates environment sets and includes
    # Methane and Ammonia; chamber_gases must keep covering both.
    names = set(ENVIRONMENT_SETS["chamber_gases"]["names"])
    assert {"Methane", "Ammonia"} <= names


def test_default_background_pinned_to_original_names_and_order():
    assert DEFAULT_BACKGROUND == (
        "Hydrogen", "Methane", "Water", "Carbon dioxide",
        "Carbon monoxide", "Ammonia", "Oxygen", "Nitrogen",
    )


def test_build_environment_library_requires_database():
    with pytest.raises(ValueError):
        build_environment_library(["chamber_gases"], database=None)


def test_build_environment_library_rejects_unknown_set(fixture_db):
    with pytest.raises(ValueError):
        build_environment_library(["not_a_real_set"], database=fixture_db)


def test_build_environment_library_succeeds_against_a_fixture_database(fixture_db):
    # tests/conftest.py's fixture_db carries every chamber_gases compound,
    # so this exercises the success path without the real NIST database.
    lib = build_environment_library(["chamber_gases"], database=fixture_db)
    assert set(lib.names()) == set(ENVIRONMENT_SETS["chamber_gases"]["names"])
    for spec in lib:
        assert spec.metadata["environment"] is True
        assert spec.metadata["environment_set"] == "chamber_gases"
        assert spec.metadata["environment_source"] == "fixture"


def test_build_environment_library_raises_on_missing_compound(fixture_db):
    # fixture_db has no pump_oil compounds at all, so the first lookup misses.
    with pytest.raises(ValueError):
        build_environment_library(["pump_oil"], database=fixture_db)


def test_allowed_elements_from_ion_pair_basic():
    elements = allowed_elements_from_ion_pair("[NH3+]CCC(=O)O.[I-]")
    assert elements == {"C", "H", "N", "O", "I"}


def test_allowed_elements_from_ion_pair_extra_elements():
    elements = allowed_elements_from_ion_pair(
        "[NH3+]CCC(=O)O.[I-]", extra_elements={"Bi"},
    )
    assert elements == {"C", "H", "N", "O", "I", "Bi"}


def test_allowed_elements_from_ion_pair_include_environment():
    elements = allowed_elements_from_ion_pair(
        "[NH3+]CCC(=O)O.[I-]", include_environment=["chamber_gases"],
    )
    chamber_elements = {"H", "O", "C", "N", "Ar"}  # H2,H2O,CO,CO2,N2,O2,Ar,NH3,CH4
    assert chamber_elements <= elements
    assert "I" in elements


def test_allowed_elements_from_ion_pair_unknown_environment_set():
    with pytest.raises(ValueError):
        allowed_elements_from_ion_pair("C", include_environment=["not_a_real_set"])


def test_build_environment_library_nist_tags_source(nist_db_path):
    from rgakit.databases import NistDatabase

    db = NistDatabase(nist_db_path)
    lib = build_environment_library(["chamber_gases"], database=db)
    assert set(lib.names()) == set(ENVIRONMENT_SETS["chamber_gases"]["names"])
    for spec in lib:
        assert spec.metadata["environment"] is True
        assert spec.metadata["environment_source"] == "nist"


def test_pump_oil_mz_coverage_against_nist(nist_db_path):
    # Measured (NIST) relative-intensity coverage of the pump-oil cracking
    # markers m/z 39/41/43/55/57 by this set's six compounds; each channel
    # must have at least one compound at >= 10% of its own base peak.
    from rgakit.databases import NistDatabase

    db = NistDatabase(nist_db_path)
    markers = (39, 41, 43, 55, 57)
    coverage = {mz: 0.0 for mz in markers}
    for smi in ENVIRONMENT_SETS["pump_oil"]["smiles"]:
        spec = db.get(smiles=smi)
        assert spec is not None, smi
        base = float(spec.intensity.max())
        rel = {int(m): float(i) / base for m, i in zip(spec.mz, spec.intensity)}
        for mz in markers:
            coverage[mz] = max(coverage[mz], rel.get(mz, 0.0))

    for mz, rel in coverage.items():
        assert rel >= 0.10, (mz, coverage)


def test_environment_source_falls_back_to_database_source(fixture_db):
    lib = build_environment_library(["chamber_gases"], database=fixture_db)
    expected = {s.metadata.get("source") or getattr(fixture_db, "_SOURCE", None) or "unknown" for s in lib}
    assert {s.metadata["environment_source"] for s in lib} == expected
    assert "nist" not in expected or getattr(fixture_db, "_SOURCE", None) == "nist"


def test_review_tiers_environment_spectra_as_background(fixture_db):
    from rgakit.review import build_candidates
    lib = build_environment_library(["chamber_gases"], database=fixture_db)
    cands = build_candidates(lib, background_names=())
    assert all(c["environment"] for c in cands)
