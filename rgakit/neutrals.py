"""
neutrals.py
-----------
Rule-based neutral-product generator for ion-pair precursors.

Given an organic cation SMILES (optionally with its counter-anion), enumerate
candidate neutral products from a library of reaction classes written as
RDKit reaction SMARTS. This is an alternative to the exhaustive bond-breaking
fragmentation in :mod:`rgakit.molecule.compound`: instead of every subgraph
cut, only the products of named reaction classes are generated, each
labelled with the rule that produced it.

The rules are candidate generators, not claims about which reactions occur
in a given sample; deciding that is the job of the fit and of the analyst.

Usage
-----
    from rgakit.neutrals import generate_neutral_products

    prods = generate_neutral_products("CCC[S+](CCC)CCC.[I-]")
    for rec in prods:
        print(rec.rule, rec.smiles)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NeutralProduct:
    """One generated neutral product set and the rule that produced it."""

    smiles: str
    rule: str
    components: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NeutralRule:
    """A named reaction class written as reaction SMARTS."""

    name: str
    smart: str
    n_reactants: int = 1


# ---------------------------------------------------------------------------
# Charge neutralization of reaction products
# ---------------------------------------------------------------------------

_HALOGENS = {9, 17, 35, 53}
_ONIUM_HETEROATOMS = {7, 8, 16, 15, 33}


def _neutralize_product(mol):
    """
    Neutralize a reaction product in place.

    RDKit keeps the reactant atom's formal charge on mapped product atoms, so
    salt-derived products come out charged.  Halide anions become the neutral
    hydride HX when free (implicit H fills valence 1) and stay H0 when bonded;
    N/S/P cations lose the charge and one H (amine/sulfide/phosphine
    counterpart, what EI-MS databases index); oxide anions (carboxylate
    leaving groups) lose the charge.
    """
    from rdkit import Chem

    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        q = atom.GetFormalCharge()
        if q < 0:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        elif q > 0 and atom.GetAtomicNum() in _ONIUM_HETEROATOMS:
            n_h = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            if n_h > 0:
                atom.SetNumExplicitHs(n_h - 1)
            atom.SetNoImplicit(True if n_h > 0 else False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    return mol


@contextmanager
def _quiet_rdkit():
    """Silence RDKit sanitization messages for products that cannot be neutralized."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


def _split_ions(mol):
    """
    Split a salt SMILES into cation, anion and neutral fragments by net
    charge, independent of the order the fragments are written in. Anions
    are deduplicated (a dihalide salt lists each halide once).
    """
    from rdkit import Chem

    cations, anions, neutrals = [], [], []
    seen_anions: set[str] = set()
    for frag in Chem.GetMolFrags(mol, asMols=True):
        charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        if charge > 0:
            cations.append(frag)
        elif charge < 0:
            key = Chem.MolToSmiles(frag)
            if key not in seen_anions:
                seen_anions.add(key)
                anions.append(frag)
        else:
            neutrals.append(frag)
    return cations, anions, neutrals


def _run_reaction(rule: "NeutralRule", mol):
    """
    Run *rule* on the (already split) reactant molecules, returning
    (component SMILES tuple) for every unique product set.
    """
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions as rdr

    rxn = rdr.ReactionFromSmarts(rule.smart)
    out: set[tuple[str, ...]] = set()
    with _quiet_rdkit():
        for pset in rxn.RunReactants(tuple(mol)):
            comps: list[str] = []
            for m in pset:
                try:
                    comps.append(Chem.MolToSmiles(_neutralize_product(Chem.Mol(m))))
                except Exception:
                    logger.debug("%s: one product component could not be neutralized; "
                                 "keeping the others", rule.name)
            if comps:
                out.add(tuple(sorted(comps)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Rule library
# ---------------------------------------------------------------------------

#: SN2 dealkylation of a quaternary N+/S+/P+ centre by the halide: RX + the
#: neutral amine/sulfide/phosphine, one per distinct alkyl group. Benzyl
#: groups are handled by :data:`_BENZYL_SN2`.
_DEALK = NeutralRule(
    name="dealkylation",
    smart="[C;X4;!R;!$(C-a):1][NX4+&H0,SX3+&H0,SX4+&H0,PX3+&H0,PX4+&H0:2].[I-,Br-,Cl-,F-:3]"
          ">>[C:1][I,Br,Cl,F:3].[N,S,P:2]",
    n_reactants=2,
)

#: Dealkylation of a protic ammonium by the halide: RX + NH3/amine.
_PROTIC_DEALK = NeutralRule(
    name="protic_dealkylation",
    smart="[C;X4;!R;!$(C-a):1][NX4+;!H0:2].[I-,Br-,Cl-,F-:3]"
          ">>[C:1][I,Br,Cl,F:3].[N:2]",
    n_reactants=2,
)

#: Benzyl substitution by the halide: benzyl halide + amine/sulfide/phosphine.
_BENZYL_SN2 = NeutralRule(
    name="benzyl_SN2",
    smart="[CH2;X4;!R:1](-[a:4])[NX4+,SX3+,SX4+,PX3+,PX4+:2].[I-,Br-,Cl-,F-:3]"
          ">>[a:4][C:1][I,Br,Cl,F:3].[N,S,P:2]",
    n_reactants=2,
)

#: Benzyl C-N homolysis followed by hydrogen abstraction: toluene + amine.
_BENZYL_HOMOLYSIS = NeutralRule(
    name="benzyl_homolysis",
    smart="[CH2;X4;!R:1](-[c:3])[NX4+:2]>>[CH3:1][c:3].[N:2]",
)

#: Aryl C-N/C-P/C-S homolysis followed by hydrogen abstraction: arene + amine/
#: phosphine/sulfide.
_ARYL_HOMOLYSIS = NeutralRule(
    name="aryl_homolysis",
    smart="[c:1]-!@[N+,n+0,PX4+,SX3+;+1:2]>>[cH:1].[N,P,S:2]",
)

#: N-dealkylation of an N-alkyl heteroaromatic cation (imidazolium/
#: pyridinium type): RX + heterocycle.
_HETEROAROMATIC_DEALK = NeutralRule(
    name="heteroaromatic_dealkylation",
    smart="[C;X4;!R:1][n&H0;r5,r6;$([n+]),$(n:c:[n+]):2].[I-,Br-,Cl-,F-:3]"
          ">>[C:1][I,Br,Cl,F:3].[n:2]",
    n_reactants=2,
)

#: Deprotonation of a protic ammonium: free amine + HX.
_DEPROT = NeutralRule(
    name="deprotonation",
    smart="[N+;H2,H3,H4:1].[I-,Br-,Cl-,F-:2]>>[N:1].[I,Br,Cl,F:2]",
    n_reactants=2,
)

#: Elimination from a tertiary alkyl on nitrogen: alkene + amine.
_TERT_ELIM = NeutralRule(
    name="tert_alkyl_elimination",
    smart="[CX4;!R:1]([CH3:3])([CH3:4])([#6:5])[NX4+:2]"
          ">>[C:1](=[C:3])([C:4])[#6:5].[N:2]",
)

#: Hofmann/E2-type elimination where beta-H exist: alkene + amine. The bond
#: to the onium centre must be exocyclic (``-!@``) so a ring is not opened.
_E2 = NeutralRule(
    name="hofmann",
    smart="[C;X4;!c:1]([C;X4;!c;H1,H2,H3:3])-!@[NX4+,SX3+,SX4+,PX3+,PX4+:2]"
          ">>[C:1]=[C:3].[N,S,P:2]",
)

#: Decarboxylation of a carboxylic acid or carboxylate: CO2 + residue.
_DECARB = NeutralRule(
    name="decarboxylation",
    smart="[C;!R:1][CX3:2](=[OX1:3])[OX2H1,OX1-:4]>>[C:1].[C:2](=[O:3])=[O:4]",
)

#: tert-Butyl ester cleavage: isobutene + carboxylic acid.
_TBUE = NeutralRule(
    name="tBu_ester_cleavage",
    smart="[CH3;X4:2][C;X4:1]([OX2:5][CX3:6]=[OX1:7])([CH3;X4:3])[CH3;X4:4]"
          ">>[C:1](=[CH2:2])([CH3:3])[CH3:4].[C:6](=[O:7])[OX2H1:5]",
)

#: tert-Butoxy radical beta-scission: acetone + methyl, trapped by the halide
#: as CH3X.
_TBUO = NeutralRule(
    name="tBuO_scission",
    smart="[CH3;X4:2][C;X4:1]([OX2:5]C=[OX1])([CH3;X4:3])[CH3;X4:4]"
          ".[I-,Br-,Cl-,F-:8]>>[C:1](=[O:5])([C:2])[C:3].[C:4][I,Br,Cl,F:8]",
    n_reactants=2,
)

#: Disulfide S-S cleavage with hydrogen uptake: two thiols.
_DISULFIDE_CLEAVAGE = NeutralRule(
    name="disulfide_cleavage",
    smart="[C:1][SX2:2][SX2:3][C:4]>>[C:1][SH:2].[C:4][SH:3]",
)


#: Rules in the order they are applied.
ALL_RULES: dict[str, NeutralRule] = {
    r.name: r for r in (_DEALK, _BENZYL_HOMOLYSIS, _BENZYL_SN2, _PROTIC_DEALK,
                        _ARYL_HOMOLYSIS, _HETEROAROMATIC_DEALK, _DEPROT,
                        _TERT_ELIM, _E2, _DECARB, _TBUE, _TBUO,
                        _DISULFIDE_CLEAVAGE)
}


def _select_rules(rules: Iterable[str] | None) -> dict[str, NeutralRule]:
    if rules is None:
        return ALL_RULES
    names = list(rules)
    unknown = sorted(set(names) - set(ALL_RULES))
    if unknown:
        raise ValueError(f"Unknown rule(s) {unknown}. Known: {sorted(ALL_RULES)}")
    return {n: ALL_RULES[n] for n in ALL_RULES if n in names}


def neutral_products(rule: str, smiles: str) -> list[NeutralProduct]:
    """
    Run one named rule on an ion-pair SMILES.

    Parameters
    ----------
    rule   : rule name (see :data:`ALL_RULES`)
    smiles : SMILES of the ion pair (cation.anion) or cation alone

    Returns
    -------
    list of :class:`NeutralProduct` (empty when the rule does not apply)
    """
    from rdkit import Chem

    if rule not in ALL_RULES:
        raise ValueError(f"Unknown rule {rule!r}. Known: {sorted(ALL_RULES)}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rule_obj = ALL_RULES[rule]
    cations, anions, neutrals = _split_ions(mol)
    if rule_obj.n_reactants == 1:
        reactant_sets = [(f,) for f in cations + neutrals]
    else:
        reactant_sets = [(c, a) for c in cations for a in anions]

    found: set[tuple[str, ...]] = set()
    for reactants in reactant_sets:
        found.update(_run_reaction(rule_obj, reactants))
    return [NeutralProduct(smiles=".".join(comps), rule=rule, components=list(comps))
            for comps in sorted(found)]


def generate_neutral_products(
    smiles: str,
    rules: Iterable[str] | None = None,
    cascade: bool = True,
) -> list[NeutralProduct]:
    """
    Enumerate the neutral products of an ion pair from the rule library.

    Parameters
    ----------
    smiles  : SMILES of the ion pair (organic cation . anion), or cation alone
    rules   : rule names to apply (default: all of :data:`ALL_RULES`)
    cascade : also apply the unimolecular rules to first-generation products
              (e.g. tert-butyl ester cleavage to an acid followed by its
              decarboxylation)
    """
    from rdkit import Chem

    if Chem.MolFromSmiles(smiles) is None:
        return []

    selected = _select_rules(rules)
    out: list[NeutralProduct] = []
    seen: set[str] = set()
    for name in selected:
        for rec in neutral_products(name, smiles):
            if rec.smiles not in seen:
                seen.add(rec.smiles)
                out.append(rec)

    if cascade:
        first_gen = [comp for rec in list(out) for comp in rec.components]
        for comp in first_gen:
            for name, rule_obj in selected.items():
                if rule_obj.n_reactants != 1:
                    continue
                for rec in neutral_products(name, comp):
                    if rec.smiles not in seen:
                        seen.add(rec.smiles)
                        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Library integration
# ---------------------------------------------------------------------------

def candidates_library(
    smiles: str,
    database,
    rules: Iterable[str] | None = None,
    cascade: bool = True,
    flag_shared_markers: bool = False,
    marker_rel: float = 0.25,
    min_mz: int = 25,
) -> "SpectraLibrary":
    """
    Build a SpectraLibrary from the neutral products of an ion pair.

    Each product SMILES is matched against *database* (e.g. an
    :class:`~rgakit.databases.NistDatabase`); products with no entry are
    dropped and duplicates are removed by InChIKey, then name. The rule that
    produced each candidate is recorded in ``metadata["rule"]``. Background
    species (chamber gases, solvents) are not generated here; see
    :mod:`rgakit.environment`.

    Parameters
    ----------
    smiles              : SMILES of the ion pair (organic cation . anion)
    database            : spectral database with a ``.get(smiles=...)``
                          interface, or a path to a NIST database file
    rules, cascade      : see :func:`generate_neutral_products`
    flag_shared_markers : annotate each candidate with
                          ``metadata["distinct_marker"]`` (whether it has a
                          channel at >= *marker_rel* of its base peak that no
                          other candidate has at >= *marker_rel*) and
                          ``metadata["shares_markers_with"]``. Nothing is
                          dropped: candidates that cannot be told apart are
                          left for the fit and the reviewer.
    marker_rel          : minimum relative intensity of a marker channel
    min_mz              : minimum m/z of a marker channel, ignored for a
                          candidate with no channel above it
    """
    from rgakit.library import SpectraLibrary
    from rgakit.spectrum import MassSpectrum

    if database is None:
        raise ValueError("candidates_library requires a spectral database.")
    if isinstance(database, str):
        from rgakit.databases import NistDatabase
        try:
            database = NistDatabase(database)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc

    records = []
    seen: set[str] = set()
    for rec in generate_neutral_products(smiles, rules=rules, cascade=cascade):
        for comp in rec.components:
            if comp not in seen:
                seen.add(comp)
                records.append((comp, rec.rule))

    spectra = []
    seen_names: set[str] = set()
    seen_iks: set[str] = set()
    for comp, rule in records:
        spec = database.get(smiles=comp)
        if spec is None:
            logger.debug("No database entry for product %r (%s)", comp, rule)
            continue
        meta = dict(spec.metadata)
        meta.setdefault("smiles", comp)
        meta["rule"] = rule
        spec = MassSpectrum(spec.mz, spec.intensity, name=spec.name, metadata=meta)
        ik = spec.inchikey
        if (ik and ik in seen_iks) or spec.name in seen_names:
            continue
        if ik:
            seen_iks.add(ik)
        seen_names.add(spec.name)
        spectra.append(spec)

    if not spectra:
        raise ValueError(f"No database entries found for any neutral product of {smiles!r}.")
    if flag_shared_markers:
        _flag_shared_markers(spectra, marker_rel=marker_rel, min_mz=min_mz)

    logger.info("candidates_library %r: %d candidate(s).", smiles, len(spectra))
    return SpectraLibrary(spectra)


def _flag_shared_markers(spectra, marker_rel: float, min_mz: int) -> None:
    rels = []
    for spec in spectra:
        peak = float(spec.intensity.max()) or 1.0
        rels.append({int(m): float(v) / peak for m, v in zip(spec.mz, spec.intensity)})

    for j, (spec, rel) in enumerate(zip(spectra, rels)):
        floor = min_mz if any(m >= min_mz and v >= marker_rel for m, v in rel.items()) else 0
        markers = {m for m, v in rel.items() if v >= marker_rel and m >= floor}
        sharing = {k for k, o in enumerate(rels) if k != j
                   and any(o.get(m, 0.0) >= marker_rel for m in markers)}
        distinct = any(all(o.get(m, 0.0) < marker_rel for k, o in enumerate(rels) if k != j)
                       for m in markers)
        spec.metadata["distinct_marker"] = distinct
        spec.metadata["shares_markers_with"] = sorted(spectra[k].name for k in sharing)
