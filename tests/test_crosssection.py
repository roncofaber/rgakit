import logging

import numpy as np
import pytest

from rgakit.crosssection import (
    REFERENCE_SIGMA_70EV,
    get_cross_section,
    molar_ratio,
)


def test_get_cross_section_resolves_by_smiles_name_and_inchikey():
    assert get_cross_section("C=C") == REFERENCE_SIGMA_70EV["C2H4"]["sigma_A2"]
    assert get_cross_section("Hydrogen sulfide") == REFERENCE_SIGMA_70EV["H2S"]["sigma_A2"]
    assert get_cross_section("S") == REFERENCE_SIGMA_70EV["H2S"]["sigma_A2"]
    assert get_cross_section("CC=C") == REFERENCE_SIGMA_70EV["C3H6"]["sigma_A2"]
    assert get_cross_section("VGGSQFUCUMXWEO-UHFFFAOYSA-N") == REFERENCE_SIGMA_70EV["C2H4"]["sigma_A2"]


def test_get_cross_section_does_not_collide_co_with_methanol():
    pytest.importorskip("pyscf")

    co_gas = REFERENCE_SIGMA_70EV["CO"]["sigma_A2"]
    assert get_cross_section("[C-]#[O+]") == co_gas
    assert get_cross_section("Carbon monoxide") == co_gas

    # "CO" is the table's own informal key for carbon monoxide, but it is
    # also valid SMILES for methanol: a bare-string match must never be
    # tried once the identifier has already parsed as a molecule, so this
    # must fall through to a real BEB calculation for methanol, not return
    # carbon monoxide's cross section.
    methanol = get_cross_section("CO")
    assert methanol != co_gas
    assert 2.0 < methanol < 10.0


def test_reference_table_values_cite_their_source_in_data():
    for key, info in REFERENCE_SIGMA_70EV.items():
        assert info["source"], key


@pytest.mark.parametrize("smiles, nist", [
    ("O", 2.28), ("O=C=O", 3.52), ("O=S=O", 4.99), ("CCC", 8.62), ("C#C", 4.37), ("[GeH4]", 5.27),
])
def test_beb_sigma_against_nist_srd107(smiles, nist):
    pytest.importorskip("pyscf")
    from rgakit.crosssection import beb_sigma
    assert abs(beb_sigma(smiles) - nist) / nist < 0.2


def test_beb_sigma_kim_rudd_n_scaling_lowers_u_for_heavy_atoms():
    pytest.importorskip("pyscf")
    from rgakit import crosssection
    with_scaling = crosssection.beb_sigma("[GeH4]")
    period = crosssection._period
    crosssection._period = lambda z: min(period(z), 2)
    try:
        without_scaling = crosssection.beb_sigma("[GeH4]")
    finally:
        crosssection._period = period
    assert with_scaling > without_scaling


def test_molar_ratio_hand_computed_two_compound_example():
    # Two synthetic species, each with two marker channels whose branching
    # ratios sum to 1 (already pattern-sum-normalized): num has true molar
    # amount 3 at sigma 2.0 (A = n*sigma = 6.0), den has true molar amount 1
    # at sigma 4.0 (A = 4.0). signal(m) = n * sigma * branching(m), so the
    # recovered molar_ratio must equal 3.0 / 1.0 = 3.0 under every method.
    pattern_num = {10: 0.6, 20: 0.4}
    pattern_den = {15: 0.3, 25: 0.7}
    sigma_num, sigma_den = 2.0, 4.0
    n_num, n_den = 3.0, 1.0

    spectrum = {
        10: n_num * sigma_num * pattern_num[10],
        20: n_num * sigma_num * pattern_num[20],
        15: n_den * sigma_den * pattern_den[15],
        25: n_den * sigma_den * pattern_den[25],
    }
    for method in ("geo", "sum", "single"):
        r = molar_ratio(
            spectrum, None,
            pattern_num=pattern_num, pattern_den=pattern_den,
            sigma_num=sigma_num, sigma_den=sigma_den,
            method=method,
        )
        assert np.isclose(r, n_num / n_den), (method, r)


def test_molar_ratio_raises_on_missing_channel_by_default():
    # mz 20 (part of pattern_num) is absent from the observed spectrum.
    pattern_num = {10: 0.6, 20: 0.4}
    pattern_den = {15: 1.0}
    spectrum = {10: 6.0, 15: 4.0}

    with pytest.raises(ValueError, match=r"m/z 20"):
        molar_ratio(
            spectrum, None,
            pattern_num=pattern_num, pattern_den=pattern_den,
            sigma_num=2.0, sigma_den=4.0,
            method="sum",
        )


def test_molar_ratio_on_missing_nan_logs_and_returns_nan(caplog):
    pattern_num = {10: 0.6, 20: 0.4}
    pattern_den = {15: 1.0}
    spectrum = {10: 6.0, 15: 4.0}

    with caplog.at_level(logging.WARNING, logger="rgakit"):
        r = molar_ratio(
            spectrum, None,
            pattern_num=pattern_num, pattern_den=pattern_den,
            sigma_num=2.0, sigma_den=4.0,
            method="sum",
            on_missing="nan",
        )

    assert np.isnan(r)
    assert any("m/z 20" in rec.message for rec in caplog.records)


def test_molar_ratio_excludes_contaminated_channel():
    # Same model as above, but a third marker channel (mz 30) is shared by
    # both species above contamination_rel; it must be dropped from both
    # patterns rather than bias the recovered amplitude.
    pattern_num = {10: 0.5, 20: 0.3, 30: 0.2}
    pattern_den = {15: 0.5, 25: 0.3, 30: 0.2}
    sigma_num, sigma_den = 2.0, 4.0
    n_num, n_den = 3.0, 1.0

    spectrum = {
        10: n_num * sigma_num * pattern_num[10],
        20: n_num * sigma_num * pattern_num[20],
        15: n_den * sigma_den * pattern_den[15],
        25: n_den * sigma_den * pattern_den[25],
        30: 999.0,  # heavily contaminated shared channel
    }
    for method in ("geo", "sum", "single"):
        r = molar_ratio(
            spectrum, None,
            pattern_num=pattern_num, pattern_den=pattern_den,
            sigma_num=sigma_num, sigma_den=sigma_den,
            method=method,
        )
        assert np.isclose(r, n_num / n_den), (method, r)


def test_molar_ratio_raises_when_contamination_leaves_no_channel():
    with pytest.raises(ValueError, match="no usable marker channel"):
        molar_ratio({10: 1.0, 15: 1.0}, None, pattern_num={10: 1.0}, pattern_den={10: 0.5, 15: 0.5},
                    sigma_num=1.0, sigma_den=1.0, method="sum")


def test_molar_ratio_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown amplitude method"):
        molar_ratio({10: 1.0, 15: 1.0}, None, pattern_num={10: 1.0}, pattern_den={15: 1.0},
                    sigma_num=1.0, sigma_den=1.0, method="bogus")
