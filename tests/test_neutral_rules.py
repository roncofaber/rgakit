"""
test_neutral_rules.py
Rule-based neutral-product generator for ion pairs.

Unit tests for each reaction rule on small benchmark molecules.
"""

from rgakit.neutrals import generate_neutral_products, neutral_products


def _pairs(prods):
    """Set of frozensets of component SMILES, from the product records."""
    return {frozenset(rec.components) for rec in prods}


# ---------------------------------------------------------------------------
# Dealkylation at a saturated quaternary N+/S+/P+ carbon
# ---------------------------------------------------------------------------

class TestDealkylation:
    def test_tripropylsulfonium_iodide(self):
        prods = neutral_products("dealkylation", "CCC[S+](CCC)CCC.[I-]")
        assert _pairs(prods) == {frozenset({"CCCI", "CCCSCCC"})}

    def test_tributylsulfonium_iodide(self):
        prods = neutral_products("dealkylation", "CCCC[S+](CCCC)CCCC.[I-]")
        assert _pairs(prods) == {frozenset({"CCCCI", "CCCCSCCCC"})}

    def test_benzyltrimethylammonium_iodide(self):
        prods = neutral_products("dealkylation", "C[N+](C)(C)Cc1ccc(C)cc1.[I-]")
        assert _pairs(prods) == {frozenset({"CI", "Cc1ccc(CN(C)C)cc1"})}

    def test_no_rx_from_pyridinium(self):
        prods = neutral_products("dealkylation", "C[n+]1ccccc1.[I-]")
        assert not _pairs(prods)

    def test_bromide_leaving_group(self):
        prods = neutral_products("dealkylation", "CCC[S+](CCC)CCC.[Br-]")
        assert _pairs(prods) == {frozenset({"CCCBr", "CCCSCCC"})}

    def test_no_rx_from_protonated_sulfonium(self):
        prods = neutral_products("dealkylation", "CC[SH+]CC.[Br-]")
        assert not _pairs(prods)

    def test_no_rx_from_protonated_phosphine(self):
        prods = neutral_products("dealkylation", "CC[PH+](CC)CC.[Br-]")
        assert not _pairs(prods)


class TestBenzyl:
    def test_homolysis_gives_toluene(self):
        prods = neutral_products("benzyl_homolysis", "C[N+](C)(C)Cc1ccc(C)cc1.[I-]")
        assert _pairs(prods) == {frozenset({"Cc1ccc(C)cc1", "CN(C)C"})}

    def test_homolysis_triethylbenzyl(self):
        prods = neutral_products("benzyl_homolysis", "CC[N+](CC)(CC)Cc1ccc(C)cc1.[I-]")
        assert _pairs(prods) == {frozenset({"Cc1ccc(C)cc1", "CCN(CC)CC"})}

    def test_benzyl_sn2(self):
        prods = neutral_products("benzyl_SN2", "C[N+](C)(C)Cc1ccc(C)cc1.[I-]")
        assert _pairs(prods) == {frozenset({"Cc1ccc(CI)cc1", "CN(C)C"})}

    def test_homolysis_from_protic_benzylammonium(self):
        prods = neutral_products("benzyl_homolysis", "[NH3+]Cc1ccc(Cl)cc1.[I-]")
        assert _pairs(prods) == {frozenset({"Cc1ccc(Cl)cc1", "N"})}


# ---------------------------------------------------------------------------
# Deprotonation of protic ammonium
# ---------------------------------------------------------------------------

class TestDeprotonation:
    def test_sec_butylammonium_bromide(self):
        prods = neutral_products("deprotonation", "CC(CC)[NH3+].[Br-]")
        assert _pairs(prods) == {frozenset({"CCC(C)N", "Br"})}

    def test_fluoroethylammonium_iodide(self):
        prods = neutral_products("deprotonation", "[NH3+]CCF.[I-]")
        assert _pairs(prods) == {frozenset({"NCCF", "I"})}


# ---------------------------------------------------------------------------
# Hofmann/E2-type elimination where beta-H exist
# ---------------------------------------------------------------------------

class TestElimination:
    def test_no_e2_from_benzyl(self):
        # benzyl has no sp3 beta-carbon (the ring ipso C carries no H),
        # so no Hofmann product is possible
        prods = neutral_products("hofmann", "C[N+](C)(C)Cc1ccc(C)cc1.[I-]")
        assert not _pairs(prods)

    def test_tributylsulfonium_iodide(self):
        prods = neutral_products("hofmann", "CCCC[S+](CCCC)CCCC.[I-]")
        assert _pairs(prods) == {frozenset({"C=CCC", "CCCCSCCCC"})}


class TestTertAlkylElimination:
    def test_cyclopropyl_dimethyl_ammonium(self):
        prods = neutral_products("tert_alkyl_elimination", "[I-].CC(C)([NH3+])C1CC1")
        assert _pairs(prods) == {frozenset({"C=C(C)C1CC1", "N"})}

    def test_cumylamine(self):
        prods = neutral_products("tert_alkyl_elimination", "[Br-].CC(C)([NH3+])c1ccccc1")
        assert _pairs(prods) == {frozenset({"C=C(C)c1ccccc1", "N"})}

    def test_no_elimination_when_n_carbon_is_not_tertiary(self):
        prods = neutral_products("tert_alkyl_elimination", "[I-].CC([NH3+])C(C)(C)C")
        assert not _pairs(prods)


# ---------------------------------------------------------------------------
# Decarboxylation of carboxylic acids
# ---------------------------------------------------------------------------

class TestDecarboxylation:
    def test_beta_alaninium_iodide(self):
        prods = neutral_products("decarboxylation", "NCCC(=O)O.[I-]")
        # decarboxylated cation-derived amine + CO2
        assert _pairs(prods) == {frozenset({"CCN", "O=C=O"})}

    def test_no_carboxylic_acid_no_reaction(self):
        prods = neutral_products("decarboxylation", "C[N+](C)(C)C.[I-]")
        assert not _pairs(prods)


# ---------------------------------------------------------------------------
# tert-butyl ester cleavage and tBuO beta-scission
# ---------------------------------------------------------------------------

class TestTertButylEster:
    def test_tbu_ester_branched_amine_iodide(self):
        prods = neutral_products("tBu_ester_cleavage", "CC(C)(C)OC(=O)CC(CC)[NH3+].[I-]")
        assert _pairs(prods) == {frozenset({"C=C(C)C", "CCC(N)CC(=O)O"})}

    def test_tbuo_beta_scission(self):
        prods = neutral_products("tBuO_scission", "CC(C)(C)OC(=O)CC(CC)[NH3+].[I-]")
        pairs = {frozenset(rec.components) for rec in prods if rec.rule == "tBuO_scission"}
        assert frozenset({"CC(C)=O", "CI"}) in pairs


# ---------------------------------------------------------------------------
# Library integration: candidates -> SpectraLibrary through NIST matching
# ---------------------------------------------------------------------------



def test_rule_list_has_no_evidence_tags():
    from rgakit.neutrals import ALL_RULES, NeutralRule
    assert {f for f in NeutralRule.__dataclass_fields__} == {"name", "smart", "n_reactants"}
    assert "generic_small" not in ALL_RULES


def test_rules_subset_and_unknown_rule():
    import pytest
    prods = generate_neutral_products("CCC[S+](CCC)CCC.[I-]", rules=["dealkylation"])
    assert {r.rule for r in prods} == {"dealkylation"}
    with pytest.raises(ValueError):
        generate_neutral_products("CCC[S+](CCC)CCC.[I-]", rules=["not_a_rule"])


def test_flag_shared_markers_keeps_both_and_names_partner():
    import numpy as np
    from rgakit.neutrals import _flag_shared_markers
    from rgakit.spectrum import MassSpectrum
    a = MassSpectrum(np.array([30, 45]), np.array([100.0, 60.0]), name="a")
    b = MassSpectrum(np.array([30, 45]), np.array([100.0, 60.0]), name="b")
    c = MassSpectrum(np.array([30, 57]), np.array([100.0, 80.0]), name="c")
    _flag_shared_markers([a, b, c], 0.25, 25)
    assert a.metadata["distinct_marker"] is False and b.metadata["distinct_marker"] is False
    assert c.metadata["distinct_marker"] is True
    assert a.metadata["shares_markers_with"] == ["b", "c"]


class TestLibraryIntegration:
    def test_candidates_library_nist(self, nist_db_path):
        from rgakit.neutrals import candidates_library
        lib = candidates_library("CCC[S+](CCC)CCC.[I-]", database=nist_db_path)
        names = set(lib.names())
        assert len(names) >= 3
        assert any("iodo" in n.lower() for n in names)
        assert any("sulfide" in n.lower() for n in names)

    def test_flag_shared_markers_drops_nothing(self, nist_db_path):
        from rgakit.neutrals import candidates_library
        plain = candidates_library("CC(C)(C)OC(=O)CC(CC)[NH3+].[I-]", database=nist_db_path)
        flagged = candidates_library("CC(C)(C)OC(=O)CC(CC)[NH3+].[I-]", database=nist_db_path,
                                     flag_shared_markers=True)
        assert flagged.names() == plain.names()
        co2 = flagged["Carbon dioxide"].metadata
        assert co2["distinct_marker"] is False
        assert "sec-Butylamine" in co2["shares_markers_with"]

    def test_rule_labels_kept_in_metadata(self, nist_db_path):
        from rgakit.neutrals import candidates_library
        lib = candidates_library("CCC[S+](CCC)CCC.[I-]", database=nist_db_path)
        rules = [s.metadata.get("rule") for s in lib]
        assert any(r is not None for r in rules)

    def test_missing_database_raises(self):
        from rgakit.neutrals import candidates_library
        import pytest
        with pytest.raises(ValueError):
            candidates_library("CCC[S+](CCC)CCC.[I-]", database="/nonexistent.db")


# ---------------------------------------------------------------------------
# Fragment order and multiply charged salts
# ---------------------------------------------------------------------------

class TestFragmentOrder:
    def test_anion_first_dealkylation(self):
        prods = generate_neutral_products("[I-].CCC[S+](CCC)CCC")
        assert frozenset({"CCCI", "CCCSCCC"}) in _pairs(prods)

    def test_anion_first_decarboxylation(self):
        prods = generate_neutral_products("[I-].[NH3+]CCC(=O)O")
        assert frozenset({"CCN", "O=C=O"}) in _pairs(prods)

    def test_anion_first_hx_listed(self):
        prods = generate_neutral_products("[Br-].CC(CC)[NH3+]")
        smis = {s for rec in prods for s in rec.components}
        assert "Br" in smis and "CCC(C)N" in smis

    def test_order_does_not_matter(self):
        a = _pairs(generate_neutral_products("CCC[S+](CCC)CCC.[I-]"))
        b = _pairs(generate_neutral_products("[I-].CCC[S+](CCC)CCC"))
        assert a == b

    def test_dication_with_two_anions(self):
        prods = generate_neutral_products("[Br-].[Br-].CC([NH3+])CCCC([NH3+])C")
        smis = {s for rec in prods for s in rec.components}
        assert "Br" in smis
        assert any("N" in s and s != "N" for s in smis)


class TestProticAndPartialProducts:
    def test_protic_dealkylation(self):
        prods = generate_neutral_products("[I-].CC(CC)[NH3+]")
        assert "dealkylation" not in {r.rule for r in prods}
        assert frozenset({"CCC(C)I", "N"}) in _pairs(
            [r for r in prods if r.rule == "protic_dealkylation"])

    def test_decarboxylation_survives_unneutralizable_cation(self):
        smi = "C[n+]1ccn(CCCN(C=O)CC(=O)O)c1.[I-]"
        prods = neutral_products("decarboxylation", smi)
        assert any("O=C=O" in rec.components for rec in prods)


class TestArylHetDealkAndDisulfide:
    def test_aryl_homolysis_triethylphenylammonium(self):
        prods = neutral_products("aryl_homolysis", "CC[N+](CC)(CC)c1ccccc1.[Cl-]")
        assert _pairs(prods) == {frozenset({"c1ccccc1", "CCN(CC)CC"})}

    def test_aryl_homolysis_phosphonium_and_anilinium(self):
        p_prods = neutral_products("aryl_homolysis", "CC[P+](C)(c1ccccc1)c1ccccc1.[Br-]")
        assert _pairs(p_prods) == {frozenset({"c1ccccc1", "CCP(C)c1ccccc1"})}
        n_prods = neutral_products("aryl_homolysis", "C[NH2+]c1ccccc1.[Cl-]")
        assert _pairs(n_prods) == {frozenset({"c1ccccc1", "CN"})}

    def test_heteroaromatic_dealkylation(self):
        for smi, expected in [
            ("CC[n+]1ccn(C)c1.[Br-]", {frozenset({"CCBr", "Cn1ccnc1"}), frozenset({"CBr", "CCn1ccnc1"})}),
            ("CCCCCC[n+]1cc(C)n(C)c1.[Br-]",
             {frozenset({"CBr", "CCCCCCn1cnc(C)c1"}), frozenset({"CCCCCCBr", "Cc1cncn1C"})}),
            ("C[n+]1ccccc1.[Br-]", {frozenset({"CBr", "c1ccncc1"})}),
        ]:
            prods = neutral_products("heteroaromatic_dealkylation", smi)
            assert _pairs(prods) == expected

    def test_disulfide_cleavage_preserves_stoichiometry(self):
        prods = neutral_products("disulfide_cleavage", "CSSC")
        assert len(prods) == 1
        assert sorted(prods[0].components) == ["CS", "CS"]

    def test_exocyclic_hofmann_cycloheptylammonium(self):
        prods = neutral_products("hofmann", "[Cl-].[NH3+]C1CCCCCC1")
        assert _pairs(prods) == {frozenset({"C1=CCCCCC1", "N"})}


class TestCascade:
    def test_tbu_ester_cascade_decarboxylation(self):
        smi = "CC(C)(C)OC(=O)CC(CC)[NH3+].[Cl-]"
        one_step = {c for r in generate_neutral_products(smi, cascade=False) for c in r.components}
        two_step = {c for r in generate_neutral_products(smi) for c in r.components}
        assert one_step < two_step
        assert "O=C=O" in two_step
