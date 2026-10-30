import numpy as np

from rgakit.molecule.draw import _build_display, _distinct_count, _fmt_formula, _frag_label
from rgakit.result import FitResult
from rgakit.spectrum import MassSpectrum


class _FakeFrag:
    def __init__(self, smiles, formula, n_heavy, mono_mass):
        self.smiles = smiles
        self.formula = formula
        self.n_heavy = n_heavy
        self.monoisotopic_mass = mono_mass


def test_frag_label_does_not_repeat_formula_when_unmatched():
    frag = _FakeFrag("CC", "C2H6", 2, 30.0)
    label = _frag_label(frag, None, {})
    assert label.count(_fmt_formula("C2H6")) == 1


def test_frag_label_shows_formula_once_when_matched():
    frag = _FakeFrag("CC", "C2H6", 2, 30.0)
    spec = MassSpectrum(np.array([30]), np.array([1.0]), name="Ethane")
    label = _frag_label(frag, spec, {})
    assert label.count(_fmt_formula("C2H6")) == 1
    assert "Ethane" in label


def test_frag_label_percentage_is_signal_share_not_raw_weight():
    frag = _FakeFrag("CC", "C2H6", 2, 30.0)
    spec = MassSpectrum(np.array([30]), np.array([1.0]), name="Ethane")
    label = _frag_label(frag, spec, {"Ethane": 0.25})
    assert "25% signal" in label


def test_distinct_count_dedupes_repeated_fragment_matches():
    spec = MassSpectrum(np.array([30]), np.array([1.0]), name="Ethane")
    frag_a = _FakeFrag("CC", "C2H6", 2, 30.0)
    frag_b = _FakeFrag("CC", "C2H6", 2, 30.0)  # same compound via a different cleavage
    matched = [(frag_a, spec), (frag_b, spec)]
    assert _distinct_count(matched) == 1


def test_distinct_count_counts_each_unique_unmatched_fragment():
    frag_a = _FakeFrag("CC", "C2H6", 2, 30.0)
    frag_b = _FakeFrag("CCC", "C3H8", 3, 44.0)
    assert _distinct_count([frag_a, frag_b]) == 2


def _fit_result(names_weights, grid):
    observed = np.full(len(grid), 10.0)
    return FitResult(
        weights=dict(names_weights),
        residual=0.0,
        fitted=np.zeros(len(grid)),
        observed=observed,
        grid=grid,
    )


def test_build_display_signal_share_matches_fraction_of_observed_signal():
    grid = np.array([10, 20, 30])
    # Pattern on-grid: max-normalized, so on_grid sum here is 1.0 + 0.5 = 1.5.
    spec = MassSpectrum(np.array([10, 20]), np.array([1.0, 0.5]), name="A",
                        metadata={"inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N"})
    frag = _FakeFrag("C", "CH4", 1, 16.0)
    result = _fit_result({"A": 2.0}, grid)
    # observed = [10, 10, 10] on this 3-channel grid -> sum = 30, max = 10.
    # weight * pattern_sum * obs_max / obs_sum = 2.0 * 1.5 * 10 / 30 = 1.0 (100%).

    class _Lib:
        def __iter__(self):
            return iter([spec])

    display, matched, unmatched, signal_shares = _build_display(
        [frag], _Lib(), result, show_unmatched=True, min_contribution=1e-4, max_frags=20,
    )
    assert signal_shares["A"] == 1.0
