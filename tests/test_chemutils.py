from rgakit.chemutils import elements_from_formula, formula_from_smiles
from rgakit.databases.utils import parse_elements


def test_elements_from_formula_is_parse_elements():
    assert elements_from_formula is not parse_elements  # it's a wrapper, not a re-export
    assert elements_from_formula("CH3I") == parse_elements("CH3I")


def test_cl_is_not_split_into_c_and_l():
    assert elements_from_formula("CCl4") == {"C", "Cl"}


def test_br_two_letter_symbol():
    assert elements_from_formula("C2H5Br") == {"C", "H", "Br"}


def test_si_two_letter_symbol():
    assert elements_from_formula("SiH4") == {"Si", "H"}


def test_sb_not_confused_with_s():
    assert elements_from_formula("SbCl3") == {"Sb", "Cl"}


def test_isotope_brackets_ignored():
    assert elements_from_formula("[13C]H4") == {"C", "H"}
    assert elements_from_formula("[2H]2O") == {"H", "O"}


def test_dot_split_ion_pair_formula():
    # beta-alaninium iodide written as separate cation/anion fragments
    assert elements_from_formula("C3H10NO2.I") == {"C", "H", "N", "O", "I"}


def test_charge_suffix_ignored():
    assert elements_from_formula("C2H5N+") == {"C", "H", "N"}
    assert elements_from_formula("C6H5-") == {"C", "H"}


def test_formula_from_smiles_merges_ion_pair_fragments():
    # RDKit's CalcMolFormula sums atoms across a .-joined ion pair into one
    # Hill formula rather than keeping per-fragment formulas; confirms
    # elements_from_formula doesn't need fragment-splitting for SMILES that
    # already went through formula_from_smiles.
    formula = formula_from_smiles("[NH3+]CCC(=O)O.[I-]")
    assert elements_from_formula(formula) == {"C", "H", "N", "O", "I"}
