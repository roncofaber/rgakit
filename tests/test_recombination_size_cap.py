from rgakit.molecule.compound import Compound, _carbon_count

# A C3 cation with a halide counter-ion: enough radical/anion chemistry to
# exercise both R1 and R2 of add_radical_pairs.
CATION_SMILES = "C[NH+](C)C.[Cl-]"

# Ethyldimethylammonium chloride, also checked against the inventory: at
# max_heavy=5/max_cuts=2/depth=2, do_fragmentation's own Rule 3 pairing
# (not do_solid_state_recombination) already produces products with more
# carbons than the parent, via cross-pairing of depth-2 fragments.
PAIRING_CATION_SMILES = "CC[NH+](C)C.[Cl-]"

# Its oxygens and nitrogen let C4-C5 products stay under a heavy-atom cap
# equal to the parent's, which max_carbons does catch.
BETA_ALANINIUM_IODIDE = "[NH3+]CCC(=O)O.[I-]"


def _fragment_smiles(frags):
    return {f.smiles for f in frags if f.smiles is not None}


def test_default_behaviour_unchanged_with_no_cap():
    c1 = Compound(CATION_SMILES)
    c1.do_fragmentation(max_heavy=4, depth=2)
    frags1 = c1.do_solid_state_recombination()

    c2 = Compound(CATION_SMILES)
    c2.do_fragmentation(max_heavy=4, depth=2)
    frags2 = c2.do_solid_state_recombination(max_carbons=None)

    assert _fragment_smiles(frags1) == _fragment_smiles(frags2)
    assert len(frags1) == len(frags2)


def test_cap_equal_to_parent_carbons_excludes_larger_products():
    uncapped = Compound(CATION_SMILES)
    uncapped.do_fragmentation(max_heavy=4, depth=2)
    uncapped_frags = uncapped.do_solid_state_recombination()

    parent_carbons = _carbon_count(uncapped.smiles)
    # The uncapped run must actually produce at least one oversized product
    # for this test to mean anything (self-pairing recombination can grow
    # beyond the parent, e.g. two CH3 radicals -> C2H6).
    assert any(_carbon_count(f.smiles) > parent_carbons for f in uncapped_frags if f.smiles), (
        "test fixture does not exercise the carbon-growing case"
    )

    capped = Compound(CATION_SMILES)
    capped.do_fragmentation(max_heavy=4, depth=2)
    capped_frags = capped.do_solid_state_recombination(max_carbons=parent_carbons)

    assert all(_carbon_count(f.smiles) <= parent_carbons for f in capped_frags if f.smiles)
    assert len(capped_frags) < len(uncapped_frags)


def test_recombine_smiles_cap_also_applies():
    uncapped = Compound(CATION_SMILES)
    uncapped.do_fragmentation(max_heavy=4, depth=2)
    uncapped_frags = uncapped.recombine_smiles()

    parent_carbons = _carbon_count(uncapped.smiles)
    capped = Compound(CATION_SMILES)
    capped.do_fragmentation(max_heavy=4, depth=2)
    capped_frags = capped.recombine_smiles(max_carbons=parent_carbons)

    assert all(_carbon_count(f.smiles) <= parent_carbons for f in capped_frags if f.smiles)
    assert len(capped_frags) <= len(uncapped_frags)


def test_do_fragmentation_default_unchanged_with_no_pairing_cap():
    c1 = Compound(PAIRING_CATION_SMILES)
    frags1 = c1.do_fragmentation(max_heavy=5, max_cuts=2, depth=2)

    c2 = Compound(PAIRING_CATION_SMILES)
    frags2 = c2.do_fragmentation(max_heavy=5, max_cuts=2, depth=2, max_carbons=None)

    assert _fragment_smiles(frags1) == _fragment_smiles(frags2)
    assert len(frags1) == len(frags2)


def test_do_fragmentation_pairing_cap_excludes_larger_products():
    uncapped = Compound(PAIRING_CATION_SMILES)
    uncapped_frags = uncapped.do_fragmentation(max_heavy=5, max_cuts=2, depth=2)
    parent_carbons = _carbon_count(uncapped.smiles)

    # Rule 3 pairing during do_fragmentation itself (not recombination)
    # must actually produce an oversized product here, or this test
    # proves nothing.
    assert any(_carbon_count(f.smiles) > parent_carbons for f in uncapped_frags if f.smiles)

    capped = Compound(PAIRING_CATION_SMILES)
    capped_frags = capped.do_fragmentation(
        max_heavy=5, max_cuts=2, depth=2, max_carbons=parent_carbons,
    )

    assert all(_carbon_count(f.smiles) <= parent_carbons for f in capped_frags if f.smiles)
    assert len(capped_frags) < len(uncapped_frags)


def test_max_carbons_catches_what_a_heavy_atom_cap_would_miss():
    # beta-alaninium's cation has 3 carbons but 6 heavy atoms total (2 O,
    # 1 N): a heavy-atom cap equal to the parent's own size would still
    # admit a pure-hydrocarbon recombination product with up to 6
    # carbons, since it carries no heteroatoms to spend that budget on.
    # max_carbons does not have this gap.
    uncapped = Compound(BETA_ALANINIUM_IODIDE)
    uncapped.do_fragmentation(max_heavy=8, max_cuts=2, depth=2)
    parent_carbons = _carbon_count(uncapped.smiles)
    assert parent_carbons == 3

    uncapped_frags = uncapped.do_solid_state_recombination()
    oversized = [f for f in uncapped_frags if f.smiles and _carbon_count(f.smiles) > parent_carbons]
    assert oversized, "test fixture does not exercise the heteroatom-dilution case"
    # At least one oversized-by-carbon product has few enough heavy atoms
    # that an equal-to-parent heavy-atom cap (6) would not have excluded
    # it -- the exact gap max_carbons closes.
    assert any(f.n_heavy <= uncapped.n_heavy for f in oversized)

    capped = Compound(BETA_ALANINIUM_IODIDE)
    capped.do_fragmentation(max_heavy=8, max_cuts=2, depth=2, max_carbons=parent_carbons)
    capped_frags = capped.do_solid_state_recombination(max_carbons=parent_carbons)
    assert all(_carbon_count(f.smiles) <= parent_carbons for f in capped_frags if f.smiles)
