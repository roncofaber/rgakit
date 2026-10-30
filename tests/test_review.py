import numpy as np

from rgakit.spectrum import MassSpectrum
from rgakit.library import SpectraLibrary
from rgakit.review import build_candidates, apply_decisions, load_decisions, save_decisions


def _spec(name, inchikey, smiles="C"):
    return MassSpectrum(
        mz=np.array([28, 44]),
        intensity=np.array([1.0, 0.5]),
        name=name,
        metadata={"inchikey": inchikey, "smiles": smiles, "formula": "C2H6"},
    )


def test_build_candidates_dedups_and_sorts():
    lib_a = SpectraLibrary([_spec("Nitrogen", "IK1")])
    lib_b = SpectraLibrary([_spec("Nitrogen", "IK1"), _spec("Ethane", "IK2")])

    comps = build_candidates([lib_a, lib_b])

    assert len(comps) == 2
    names = [c["name"] for c in comps]
    assert names.count("Nitrogen") == 1
    assert set(names) == {"Nitrogen", "Ethane"}


def test_build_candidates_background_first():
    lib = SpectraLibrary([_spec("Ethane", "IK1"), _spec("Methane", "IK2")])

    comps = build_candidates(lib, background_names=["Methane"])

    assert comps[0]["name"] == "Methane"


def test_build_candidates_weights_stats():
    lib = SpectraLibrary([_spec("Ethane", "IK1"), _spec("Methane", "IK2")])
    weights = {"Ethane": [0.0, 0.1, 0.3]}

    comps = build_candidates(lib, weights=weights)
    by_name = {c["name"]: c for c in comps}

    assert by_name["Ethane"]["n_samples"] == 2
    assert by_name["Ethane"]["mean_w"] == 0.2
    assert by_name["Ethane"]["max_w"] == 0.3
    assert by_name["Methane"]["n_samples"] == 0


def test_decisions_roundtrip_and_apply(tmp_path):
    path = tmp_path / "decisions.json"
    dec = {"IK1": {"name": "Nitrogen", "verdict": "remove",
                   "reason": "test", "decided_at": "2026-01-01"},
           "IK2": {"name": "Ethane", "verdict": "keep",
                   "reason": "", "decided_at": "2026-01-01"}}
    save_decisions(path, dec)
    assert load_decisions(path) == dec

    lib = SpectraLibrary([_spec("Nitrogen", "IK1"), _spec("Ethane", "IK2")])

    curated = apply_decisions(lib, path)

    assert "Nitrogen" not in curated.names()
    assert "Ethane" in curated.names()
    assert len(curated) == 1

    kept = apply_decisions(lib, path, verdict="keep")
    assert len(kept) == 1 and "Nitrogen" in kept.names()

    allowed = apply_decisions(lib, path, mode="whitelist")
    assert "Nitrogen" not in allowed.names()
    assert "Ethane" in allowed.names()
    assert len(allowed) == 1
