#!/usr/bin/env python3
"""
compound.py
Compound class for molecular mass-spectrometry analysis.

A Compound wraps any valid molecular representation (SMILES, InChI, .mol/.sdf,
ASE Atoms, or RDKit Mol) and lazily enumerates all chemically connected
fragment subgraphs, exposing MS-relevant quantities (monoisotopic mass, m/z)
on both the parent molecule and every fragment.

Fragmentation returns a list of Compound objects in *fragment mode*: lightweight
instances that carry a pre-computed SMILES and/or ASE Atoms but no RDKit Mol.
All Compound properties (formula, mass, charge, spin, inchikey, …) work in
both parent and fragment mode.
"""

from __future__ import annotations

import logging
from functools import cached_property
from itertools import combinations
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

logger = logging.getLogger(__name__)

# Proton mass in Da (used for m/z calculation)
_PROTON_MASS = 1.007276466621


# ---------------------------------------------------------------------------
# ASE viewer helper
# ---------------------------------------------------------------------------

def view(atoms_or_list, traj_path: str | None = None):
    """
    Open atoms or a trajectory in the ASE GUI, working correctly inside
    Spyder / IPython kernels.

    ASE's default pipe viewer spawns a subprocess that inherits the inline
    matplotlib backend, which fails with a closed-file error.  This helper
    bypasses the pipe by writing to a temporary file (or reusing an existing
    .traj file) and launching ``ase gui <file>`` directly with
    ``MPLBACKEND=TkAgg`` set in the subprocess environment.

    Parameters
    ----------
    atoms_or_list : Atoms or list[Atoms]
    traj_path     : if the data is already on disk, pass the path to skip
                    writing a temp file
    """
    import os
    import subprocess
    import tempfile
    from ase.io import write

    env = os.environ.copy()
    env["MPLBACKEND"] = "TkAgg"

    if traj_path is not None:
        path = traj_path
    else:
        tmp  = tempfile.NamedTemporaryFile(suffix=".traj", delete=False)
        path = tmp.name
        tmp.close()
        atoms_list = (atoms_or_list
                      if isinstance(atoms_or_list, list)
                      else [atoms_or_list])
        write(path, atoms_list)

    return subprocess.Popen(["ase", "gui", path], env=env)


# ---------------------------------------------------------------------------
# Molecule loading helpers
# ---------------------------------------------------------------------------

from .load import (
    load_rdkit_mol, embed_fragment,
)
from .utils import (
    rdkit_to_ase,
    smiles_to_inchikey,
    monoisotopic_from_smiles, formal_charge_from_smiles,
    adjacency, is_connected,
    canonical_smiles_radical,
    atoms_to_graph, extract_components,
)


# Graph helpers, SMILES utilities, and format conversions all live in utils.


from .smiles import (
    hcap_smiles, deprotonate_smiles, anion_to_radical, combine_radicals,
    beta_scission_products, h_migration_products, alpha_dehydrogenation_products,
)


def is_labile(bond) -> bool:
    """
    Return True if *bond* (an RDKit Bond) is a valid primary cut site under EI.

    Aromatic bonds and double/triple bonds are never broken directly by EI
    ionisation — they require special rearrangement mechanisms (β-scission,
    retro-Diels-Alder) that are handled separately.  Only σ single bonds
    between heavy atoms are counted toward max_cuts.
    """
    if bond.GetIsAromatic():
        return False
    if bond.GetBondTypeAsDouble() >= 2.0:
        return False
    return True


def add_fragment(
    smi:        str | None,
    parent_idx: tuple[int, ...],
    results:    list,
    seen:       set[str],
) -> None:
    """
    Canonicalise *smi*, dedup against *seen*, embed geometry, and append a
    new fragment Compound to *results*.  No-ops silently if smi is None or
    already seen.

    For multi-component SMILES (ion pairs, A.B), falls back to embedding each
    component individually and concatenating the Atoms objects if joint embedding
    fails.
    """
    if smi is None:
        return
    from rdkit import Chem

    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return
    canon = Chem.MolToSmiles(parsed)
    if canon in seen:
        return
    seen.add(canon)

    frag_atoms = embed_fragment(canon)

    # Fallback for multi-component SMILES: embed each component separately.
    if frag_atoms is None and "." in canon:
        parts = [Chem.MolToSmiles(p)
                 for p in Chem.GetMolFrags(parsed, asMols=True)]
        embedded = [embed_fragment(p) for p in parts]
        if all(e is not None for e in embedded):
            frag_atoms = embedded[0]
            for e in embedded[1:]:
                frag_atoms = frag_atoms + e   # ASE Atoms concatenation

    if frag_atoms is None:
        return

    results.append(Compound.from_fragment(
        smiles     = canon,
        atoms      = frag_atoms,
        parent_idx = parent_idx,
    ))


def add_radical_pairs(
    raws:    list,
    results: list,
    seen:    set[str],
) -> None:
    """
    SMILES-level pairwise radical recombination over *raws*.

    Rule R1 — radical + radical:
        Both fragments carry unpaired electrons; combine_radicals enumerates
        all valid bond-formation products.  Residual radicals on the product
        are H-capped to guarantee a closed-shell species.

    Rule R2 — anion + radical:
        One fragment is a closed-shell anion (charge < 0, spin = 0); the
        other is a radical.  The anion is converted to its neutral radical via
        anion_to_radical (models EI electron stripping, e.g. [I-] → [I•]),
        then combine_radicals is applied as in R1.

    New fragments are appended to *results*; already-seen SMILES are skipped.
    """
    n_before = len(results)
    n_pairs  = 0

    for i, j in combinations(range(len(raws)), 2):
        fa, fb = raws[i], raws[j]

        if fa.spin > 0 and fb.spin > 0:
            # R1: radical + radical
            n_pairs += 1
            for smi in combine_radicals(fa.smiles, fb.smiles):
                smi = hcap_smiles(smi) or smi
                add_fragment(smi, fa.parent_idx + fb.parent_idx, results, seen)
        else:
            # R2: anion + radical (try both orderings)
            for anion, radical in ((fa, fb), (fb, fa)):
                if anion.charge >= 0 or anion.spin != 0:
                    continue
                if radical.spin == 0:
                    continue
                n_pairs += 1
                rad_smi = anion_to_radical(anion.smiles)
                for smi in combine_radicals(rad_smi, radical.smiles):
                    smi = hcap_smiles(smi) or smi
                    add_fragment(
                        smi,
                        anion.parent_idx + radical.parent_idx,
                        results,
                        seen,
                    )

    logger.debug(
        "  add_radical_pairs: %d reactive pair(s) → %d new product(s).",
        n_pairs, len(results) - n_before,
    )


def _collect_radical_pool(
    compounds,
    include_h_radical: bool = True,
) -> list[str]:
    """
    Build a deduplicated list of radical SMILES from one or more Compounds.

    For each compound's raw_fragments:
      - Radical fragments (spin > 0) are added directly.
      - Closed-shell anions (charge < 0, spin = 0) are converted to their
        neutral radical form via :func:`anion_to_radical` (e.g. I⁻ → I•).

    If *include_h_radical* is True, H• is appended to the pool to model
    C–H (or N–H, O–H) bond homolysis under photon irradiation.
    """
    from rdkit import Chem

    seen: set[str] = set()
    pool: list[str] = []

    for compound in compounds:
        for frag in compound.raw_fragments:
            if frag.spin > 0:
                smi = frag.smiles
            elif frag.charge < 0 and frag.spin == 0:
                smi = anion_to_radical(frag.smiles)
            else:
                continue
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if not any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                pool.append(canon)

    if include_h_radical and "[H]" not in seen:
        pool.append("[H]")

    return pool


# ---------------------------------------------------------------------------
# Compound class  (handles both parent and fragment modes)
# ---------------------------------------------------------------------------

class Compound:
    """
    A molecule and all its chemically connected fragment subgraphs.

    **Parent mode** — created via ``Compound(source, ...)``
    Wraps any valid molecular representation (SMILES, InChI, .mol/.sdf, ASE
    Atoms, or RDKit Mol).  Fragments are enumerated lazily.

    **Fragment mode** — created internally via ``Compound.from_fragment(...)``
    Lightweight instance carrying a pre-computed SMILES and/or ASE Atoms but
    no RDKit Mol.  All properties (formula, mass, charge, spin, inchikey, …)
    work in both modes.  Fragmentation methods are not available.

    Parameters
    ----------
    source      : SMILES string, InChI, path to .mol/.sdf, ASE Atoms, or RDKit Mol
    name        : optional human-readable label
    include_h   : attach H neighbours to each heavy-atom fragment (default True)
    deduplicate : skip fragments with the same canonical SMILES (default True)

    Attributes
    ----------
    graph : NetworkX Graph
        Bond graph of the parent molecule (nodes = atoms, edges = bonds).
    raw_fragments : list[Compound]
        Bare subgraph cuts with radical electrons marking broken bonds (no H added).
    fragments : list[Compound]
        Chemically stabilised species derived from raw_fragments:
        H-capped closed-shell species, SMILES-level radical recombination products,
        and charge-neutral ion pairs.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        source,
        name:        str | None = None,
        *,
        include_h:    bool       = True,
        deduplicate:  bool       = True,
    ):
        self.name        = name
        self.include_h   = include_h
        self.deduplicate = deduplicate
        self.max_heavy   = None
        self.max_cuts    = 2

        # Parent-mode fields
        self._mol           = load_rdkit_mol(source)
        self.graph          = atoms_to_graph(rdkit_to_ase(self._mol))
        self._raw_fragments = None
        self._fragments     = None

        # Fragment-mode fields (unused in parent mode)
        self.parent_idx   = ()
        self.boundary_idx = ()
        self.traj_path    = None

        # MS spectrum (populated by load_ms_spectra or to_spectrum)
        self.spectrum     = None

        logger.info(
            "Compound loaded: %s | formula=%s | mw=%.4f Da | charge=%+d | %d heavy atoms",
            self.name or "(unnamed)",
            self.formula,
            self.monoisotopic_mass,
            self.charge,
            self.n_heavy,
        )

    @classmethod
    def from_fragment(
        cls,
        smiles:       str | None,
        atoms:        Atoms,
        parent_idx:   tuple[int, ...] = (),
        boundary_idx: tuple[int, ...] = (),
    ) -> "Compound":
        """
        Create a Compound in *fragment mode* — no RDKit Mol, no fragmentation.

        The SMILES and Atoms are pre-seeded into the cached_property slots so
        that all property accesses resolve immediately without recomputation.

        Parameters
        ----------
        smiles       : canonical SMILES (None for MLIP-only recombination products)
        atoms        : ASE Atoms with 3-D positions
        parent_idx   : atom indices of this fragment inside the parent Mol
        boundary_idx : local atom indices of cut-bond atoms (radical centres)
        """
        obj = object.__new__(cls)

        # Fragment-mode state
        obj._mol           = None          # signals fragment mode
        obj._raw_fragments = None
        obj._fragments     = None
        obj.name           = None
        obj.include_h      = True
        obj.deduplicate    = True
        obj.max_heavy      = None
        obj.max_cuts       = 2
        obj.graph          = None
        obj.parent_idx     = parent_idx
        obj.boundary_idx   = boundary_idx
        obj.traj_path      = None
        obj.spectrum       = None   # populated by load_ms_spectra()

        # Pre-seed cached_property slots (avoids recomputation)
        obj.__dict__["atoms"]  = atoms
        obj.__dict__["smiles"] = smiles   # None is a valid value; descriptor skipped

        return obj

    # ------------------------------------------------------------------
    # Core structural properties  (work in both parent and fragment mode)
    # ------------------------------------------------------------------

    @cached_property
    def smiles(self) -> str | None:
        """Canonical SMILES (no explicit H). None for MLIP-only fragments."""
        if self._mol is None:
            return None  # fragment mode with smiles=None; pre-seeded otherwise
        from rdkit.Chem import MolToSmiles, RemoveHs
        return MolToSmiles(RemoveHs(self._mol))

    @cached_property
    def atoms(self) -> Atoms:
        """ASE Atoms with full 3-D geometry (explicit H)."""
        return rdkit_to_ase(self._mol)

    @cached_property
    def smiles_mol(self):
        """
        RDKit Mol built from self.smiles with all H made explicit.

        In fragment mode this is the authoritative source for formula and
        masses: it correctly accounts for H atoms that cap bonds cut during
        fragment extraction.  Returns None if SMILES is unavailable.
        """
        smi = self.smiles
        if smi is None:
            return None
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.AddHs(mol)

    @cached_property
    def formula(self) -> str:
        """Hill-ordered chemical formula string."""
        if self._mol is not None:
            # Parent mode: derive from ASE atoms (includes all H in the conformer)
            return self.atoms.get_chemical_formula()
        # Fragment mode: prefer SMILES mol (includes capping H not in conformer)
        if self.smiles_mol is not None:
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcMolFormula(self.smiles_mol)
        return self.atoms.get_chemical_formula()

    @cached_property
    def n_heavy(self) -> int:
        """Number of non-hydrogen atoms."""
        if self._mol is not None:
            return self._mol.GetNumHeavyAtoms()
        return sum(1 for s in self.atoms.get_chemical_symbols() if s != "H")

    @cached_property
    def monoisotopic_mass(self) -> float:
        """
        Exact (monoisotopic) mass (Da).

        This is the mass of the most abundant isotopologue and corresponds
        to the dominant peak in high-resolution MS for small molecules.
        Falls back to summing ASE atomic masses if SMILES is unavailable.
        """
        if self._mol is not None:
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcExactMolWt(self._mol)
        # Fragment mode
        if self.smiles is not None:
            m = monoisotopic_from_smiles(self.smiles)
            if m is not None:
                return m
        return sum(
            atomic_masses[atomic_numbers[s]]
            for s in self.atoms.get_chemical_symbols()
        )

    @cached_property
    def nominal_mass(self) -> int:
        """Nominal (integer) mass, rounded from the monoisotopic mass."""
        return round(self.monoisotopic_mass)

    @cached_property
    def charge(self) -> int:
        """Net formal charge."""
        if self._mol is not None:
            from rdkit import Chem
            return Chem.GetFormalCharge(self._mol)
        if self.smiles is not None:
            return formal_charge_from_smiles(self.smiles)
        return 0

    @cached_property
    def spin(self) -> int:
        """
        Number of unpaired electrons.

        0 for closed-shell species.  For fragments built from H-capped SMILES
        this is always 0.  Becomes non-zero only if the SMILES itself encodes
        a radical (e.g. '[CH3]').
        """
        if self.smiles_mol is not None:
            return sum(a.GetNumRadicalElectrons()
                       for a in self.smiles_mol.GetAtoms())
        return 0

    # ------------------------------------------------------------------
    # Identifier properties  (lazy, network-free where possible)
    # ------------------------------------------------------------------

    @cached_property
    def inchikey(self) -> str | None:
        """Standard InChIKey computed locally from SMILES via RDKit (no network)."""
        if self.smiles is None:
            return None
        return smiles_to_inchikey(self.smiles)

    @cached_property
    def pubchem_info(self) -> dict:
        """
        CAS registry number and IUPAC name from PubChem (lazy, one network call).

        Queried by InChIKey.  Returns an empty dict on any failure.
        """
        ik = self.inchikey
        if ik is None:
            return {}
        try:
            import pubchempy as pcp
            results = pcp.get_compounds(ik, "inchikey")
            if not results:
                return {}
            c = results[0]
            # CAS is typically the first numeric synonym (digits and hyphens only)
            cas = next(
                (s for s in (c.synonyms or [])
                 if s and s.replace("-", "").isdigit()),
                None,
            )
            return {"cas": cas, "common_name": c.iupac_name}
        except Exception as exc:
            logger.debug("PubChem lookup failed for %s: %s", ik, exc)
            return {}

    @cached_property
    def cas(self) -> str | None:
        """CAS registry number (fetched lazily from PubChem)."""
        return self.pubchem_info.get("cas")

    @cached_property
    def common_name(self) -> str | None:
        """IUPAC name (fetched lazily from PubChem)."""
        return self.pubchem_info.get("common_name")

    # ------------------------------------------------------------------
    # MS helpers
    # ------------------------------------------------------------------

    def mz(self, charge: int = 1) -> float:
        """
        Mass-to-charge ratio for the given charge state.

        Uses the protonation/deprotonation convention:
            m/z = (monoisotopic_mass + charge * m_proton) / |charge|

        Parameters
        ----------
        charge : signed integer charge state (e.g. +1 for [M+H]+, -1 for [M-H]-)
        """
        if charge == 0:
            raise ValueError("charge cannot be zero")
        return (self.monoisotopic_mass + charge * _PROTON_MASS) / abs(charge)

    # ------------------------------------------------------------------
    # Parent-mode geometry management
    # ------------------------------------------------------------------

    def update_geometry(self, atoms: Atoms) -> None:
        """
        Sync relaxed ASE Atoms positions back into the internal RDKit conformer.

        Call this after relaxing mol.atoms with an MLIP calculator so that
        raw_fragments use the physically relaxed parent geometry rather than
        the initial MMFF conformer.

        Clears the fragment caches and updates self.graph automatically.

        Parameters
        ----------
        atoms : relaxed Atoms object (must have the same atom count as the parent)
        """
        from rdkit.Geometry.rdGeometry import Point3D

        conf = self._mol.GetConformer()
        if len(atoms) != conf.GetNumAtoms():
            raise ValueError(
                f"Atoms length ({len(atoms)}) does not match "
                f"mol conformer size ({conf.GetNumAtoms()})."
            )

        for i, (x, y, z) in enumerate(atoms.get_positions()):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

        # Rebuild graph from updated geometry
        self.graph = atoms_to_graph(atoms)

        # Clear fragment caches so next access re-enumerates from new geometry
        self._raw_fragments = None
        self._fragments     = None

        # Clear the cached `atoms` property so it's rebuilt from the updated conformer
        self.__dict__.pop("atoms", None)

    # ------------------------------------------------------------------
    # Relaxation  (unified: parent mode copies + syncs; fragment mode in-place)
    # ------------------------------------------------------------------

    def relax(
        self,
        calc,
        fmax:       float      = 1e-3,
        steps:      int        = 500,
        spin:       int | None = None,
        trajectory: str | None = None,
        logfile:    str | None = None,
        log_dir                = None,
    ) -> Atoms:
        """
        Relax the geometry using an ASE calculator.

        **Parent mode**: relaxes a copy of the molecule, then calls
        ``update_geometry()`` to sync positions back into the RDKit conformer
        and rebuild the bond graph.

        **Fragment mode**: relaxes ``self.atoms`` in-place and stores the
        trajectory path in ``self.traj_path``.

        Parameters
        ----------
        calc       : ASE-compatible calculator (e.g. FAIRChemCalculator)
        fmax       : FIRE2 force convergence threshold (eV/Ang)
        steps      : maximum number of FIRE2 steps
        spin       : number of unpaired electrons; auto-detected from SMILES
                     in fragment mode (default None means auto-detect)
        trajectory : trajectory output path
        logfile    : log output path; None to suppress; '-' for stdout
        log_dir    : directory for auto-named trajectory/log files (fragment mode)
        """
        from ase.optimize import FIRE2

        if self._mol is not None:
            # ---- Parent mode ----
            _spin = spin if spin is not None else 0
            if trajectory is None:
                trajectory = "relaxation.traj"
            logger.info(
                "Relaxing parent %s (charge=%+d, spin=%d, fmax=%.1e, steps=%d) ...",
                self.formula, self.charge, _spin, fmax, steps,
            )
            system = self.atoms.copy()
            system.info["charge"] = self.charge
            system.info["spin"]   = _spin
            system.calc = calc
            opt = FIRE2(system, trajectory=trajectory, logfile=logfile)
            converged = opt.run(fmax=fmax, steps=steps)
            self.update_geometry(system)
            logger.info("Parent relaxation %s after %d steps.",
                        "converged" if converged else "did NOT converge", opt.get_number_of_steps())
            return system

        # ---- Fragment mode ----
        if len(self.atoms) < 2:
            logger.warning(
                "Skipping relaxation of %s: single-atom fragment has no "
                "internal degrees of freedom.", self.formula
            )
            return self.atoms

        _spin = spin if spin is not None else self.spin

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            if trajectory is None:
                trajectory = str(log_dir / f"{self.formula}.traj")
            if logfile is None:
                logfile = str(log_dir / f"{self.formula}.log")

        if trajectory is not None:
            self.traj_path = str(trajectory)

        logger.debug(
            "Relaxing fragment %s (charge=%+d, spin=%d, fmax=%.1e, steps=%d) ...",
            self.formula, self.charge, _spin, fmax, steps,
        )

        self.atoms.info["charge"] = self.charge
        self.atoms.info["spin"]   = _spin
        self.atoms.calc = calc

        opt = FIRE2(self.atoms, trajectory=trajectory, logfile=logfile)
        converged = opt.run(fmax=fmax, steps=steps)
        self.atoms.info["converged"] = converged
        logger.debug("  -> %s (%d steps).",
                     "converged" if converged else "did not converge",
                     opt.get_number_of_steps())
        return self.atoms

    # ------------------------------------------------------------------
    # Trajectory (fragment mode)
    # ------------------------------------------------------------------

    @property
    def trajectory(self):
        """
        Load and return the MLIP relaxation trajectory as a list of Atoms.

        Returns None if relax() has not been called or no trajectory was written.
        """
        if self.traj_path is None:
            return None
        from ase.io.trajectory import Trajectory
        return list(Trajectory(self.traj_path))

    def view_relaxation(self):
        """Open the MLIP relaxation trajectory in the ASE GUI viewer."""
        if self.traj_path is None:
            raise RuntimeError(
                f"No relaxation trajectory for {self.formula!r} — "
                "call relax() with a log_dir or trajectory path first."
            )
        return view(None, traj_path=self.traj_path)

    # ------------------------------------------------------------------
    # Fragmentation
    # ------------------------------------------------------------------

    def do_fragmentation(
        self,
        max_heavy: int | None = None,
        max_cuts:  int        = 1,
        depth:     int        = 1,
    ) -> list:
        """
        Enumerate chemically plausible EI-MS fragments from the parent molecule.

        **Physical model**

        Electron-ionisation (EI) removes one electron from the neutral molecule M,
        producing the radical cation M+•.  This species then undergoes sequential
        bond cleavage: one bond breaks at a time, each step producing an ion and a
        neutral radical.  The ion may fragment further in the same way.  A single
        EI event therefore looks like a *cascade*:

            M  --EI-->  M+•  -->  [A]+ + B•  -->  [C]+ + D  --> …

        This method models that cascade in two stages:

        **Stage 1 — bond enumeration** (:meth:`enumerate_raw`)

            Every connected subgraph of heavy atoms that can be isolated by cutting
            up to *max_cuts* labile σ bonds is enumerated.  Non-labile bonds
            (aromatic C–C, C=O, C=C) are *never* cut — EI cannot break them as a
            primary event.  Each surviving subgraph is stored as a *raw fragment*:
            a radical SMILES where open valences (cut bonds) are represented by
            unpaired electrons.

            If *depth* > 1, the same bond enumeration is applied recursively to each
            level-1 fragment, modelling secondary fragmentation events.

        **Stage 2 — stabilisation** (:meth:`build_stable`)

            Raw (radical) fragments are converted to observable, closed-shell
            species by four SMILES-level rules applied in order:

            1. **H-capping** — open valences are saturated with implicit H.
               e.g. ``[CH2]CC[NH2+]`` (radical after C–N cut) becomes ``CCC[NH2+]``
               then is deprotonated to ``CCCN`` (Rule 4).

            2. **β-scission** — for each radical centre α and each single bond
               α–β–γ, the β–γ bond is broken while α–β is upgraded to a double bond.
               This models radical elimination, the dominant secondary pathway for
               alkyl radicals.
               e.g. tert-butyl radical → isobutylene (C=C(C)C) + H•

            3. **Radical pairing** — all pairs of raw fragments are tried for
               covalent bond formation:
               * *radical + radical*: :func:`combine_radicals` enumerates every
                 possible single/double/triple bond between radical centres.
               * *anion + radical*: the anion is first converted to its neutral
                 radical (models EI electron-stripping, e.g. I⁻ → I•), then the
                 same bond-formation is attempted.  This is the route by which
                 ion-pair precursors (e.g. ammonium iodide salts) produce covalent
                 iodo-products (CH₃I, HI, …).

            4. **Deprotonation** — for each cationic fragment (charge > 0) the first
               protonated heteroatom (N, O, S) loses a proton.  EI-MS databases
               index *neutral* species, so this step is required to find amines,
               alcohols, etc. that originate from protonated precursors.

        Parameters
        ----------
        max_heavy : int or None
            Maximum number of heavy atoms in any single fragment.  Set this to
            limit the search space for large molecules (e.g. ``max_heavy=6``).
            Default: no limit (enumerate all sizes up to the parent).
        max_cuts : int
            Maximum number of labile σ bonds cut *per enumeration step*.
            ``max_cuts=1`` (default) cuts one bond at a time — appropriate for
            primary EI fragments.  ``max_cuts=2`` allows two simultaneous cuts
            (McLafferty rearrangement products, retro-Diels-Alder, …) but
            expands the search space significantly.
        depth : int
            Cascade depth for sequential fragmentation.
            ``depth=1`` (default): enumerate cuts on the parent only.
            ``depth=2``: also enumerate cuts on each level-1 fragment, producing
            secondary fragments (M → A → C).  Higher values are rarely needed.

        Returns
        -------
        list of Compound
            Stable (closed-shell) fragment objects, also stored in
            ``self.fragments``.  Raw radical intermediates are in
            ``self.raw_fragments`` and are used by :meth:`relax_fragments` and
            :meth:`relax_recombination`.

        Examples
        --------
        >>> mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
        >>> frags = mol.do_fragmentation(max_heavy=6, depth=2)
        >>> mol.summary()
        """
        # Store parameters and reset caches so re-calling gives fresh results.
        self.max_heavy      = max_heavy
        self.max_cuts       = max_cuts
        self._raw_fragments = None
        self._fragments     = None

        logger.info(
            "do_fragmentation: %s | max_heavy=%s | max_cuts=%d | depth=%d",
            self.name or self.formula,
            max_heavy if max_heavy is not None else "unlimited",
            max_cuts,
            depth,
        )

        # Level 1: enumerate labile σ-bond cuts on the parent molecule.
        self._raw_fragments = self.enumerate_raw()

        # Depth > 1: sequential cascade.
        if depth > 1:
            seen_smiles   = {f.smiles for f in self._raw_fragments if f.smiles}
            current_level = list(self._raw_fragments)

            for _level in range(depth - 1):
                next_level    = []
                level_hcapped = set()

                for raw in current_level:
                    stable_smi = hcap_smiles(raw.smiles)
                    if stable_smi is None or stable_smi in level_hcapped:
                        continue
                    level_hcapped.add(stable_smi)

                    try:
                        sub_mol = load_rdkit_mol(stable_smi)
                    except Exception:
                        continue

                    sub_frags = self.enumerate_raw(mol=sub_mol)
                    for sf in sub_frags:
                        if sf.smiles and sf.smiles not in seen_smiles:
                            seen_smiles.add(sf.smiles)
                            self._raw_fragments.append(sf)
                            next_level.append(sf)

                current_level = next_level
                if not current_level:
                    break

        # Build stable fragment list from raw cuts (SMILES-level).
        self._fragments = self.build_stable()
        return self._fragments

    def do_recombination(self) -> list:
        """
        Try all pairwise recombinations of raw_fragments and add stable
        products to the fragment library (SMILES-level, no calculator required).

        For each pair of raw fragments, :func:`combine_radicals` enumerates
        all chemically valid bond-formation products (single, double, and triple
        bonds between every pair of radical centres).  Products already in the
        fragment library (by canonical SMILES) are silently skipped.

        For geometry optimisation and MLIP-verified bond formation, call
        :meth:`relax_recombination` afterwards.

        Returns
        -------
        list of newly added Compound objects (in fragment mode)
        """
        # Raises RuntimeError if do_fragmentation() has not been called yet.
        _ = self.fragments
        return self.recombine_smiles()

    def recombine_smiles(self) -> list:
        """
        SMILES-level pairwise recombination (no MLIP).

        Two rules are applied over all pairs of raw fragments:

        **Rule R1 — radical + radical**
          Both fragments carry unpaired electrons; :func:`combine_radicals`
          enumerates all valid bond-formation products.

        **Rule R2 — anion + radical**
          One fragment is a closed-shell anion (charge < 0, spin = 0) and
          the other is a radical.  The anion is first converted to its neutral
          radical form via :func:`anion_to_radical` (models EI electron
          stripping, e.g. [I-] becomes [I•]), then :func:`combine_radicals` is
          applied as in Rule R1.  This allows ion-pair compounds such as
          ammonium iodides to produce covalent iodo-products (CH3I, HI, …).

        Products already present in the fragment library (by canonical SMILES)
        are silently skipped.  Returns the list of newly added Compound objects.
        """
        raws      = self.raw_fragments
        seen      = {f.smiles for f in self._fragments if f.smiles is not None}
        new_frags : list[Compound] = []

        add_radical_pairs(raws, new_frags, seen)

        self._fragments.extend(new_frags)
        self._fragments.sort(key=lambda f: (f.n_heavy, f.formula))
        logger.info(
            "do_recombination (SMILES): added %d product(s) to fragment library.",
            len(new_frags),
        )
        return new_frags

    @classmethod
    def solid_state_recombination(
        cls,
        compounds,
        include_h_radical: bool = True,
    ) -> list:
        """
        Model solid-state radical recombination across one or more compounds.

        In solid-state photolysis (UV, X-ray, synchrotron irradiation),
        neighboring molecules in the lattice share a common radical pool.
        This classmethod enables two reaction types that single-molecule EI
        models cannot produce:

        1. **Self-pairing** — the same radical reacts with another copy of
           itself (e.g. I• + I• → I₂, CH₃• + CH₃• → C₂H₆).
        2. **Cross-compound pairing** — radicals from different components of
           the film recombine (e.g. an organic radical + a halide radical from
           the inorganic framework).
        3. **H radical** — H• from bond homolysis combines with any radical
           in the pool (H• + I• → HI, H• + H• → H₂).

        **Algorithm**

        1. Collect unique radical SMILES from all compounds via
           :func:`_collect_radical_pool` (radicals directly; anions converted
           via :func:`anion_to_radical`; optionally H•).
        2. Enumerate all pairs *with replacement* using
           :func:`combine_radicals`.  H-cap residual radical centres.
        3. Products already present in any compound's fragment list are
           silently skipped.

        All compounds must have :meth:`do_fragmentation` called first.

        Parameters
        ----------
        compounds : list of Compound
            One or more compounds whose radical pools are merged.
        include_h_radical : bool
            Include H• in the pool (default True).

        Returns
        -------
        list of Compound
            Newly produced fragments (not yet added to any compound's list —
            the caller decides where they go).

        Examples
        --------
        >>> eai  = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
        >>> pbi2 = Compound("I[Pb]I",                        name="PbI2")
        >>> eai.do_fragmentation(max_heavy=6, depth=2)
        >>> pbi2.do_fragmentation(max_heavy=4, depth=1)
        >>> new_frags = Compound.solid_state_recombination([eai, pbi2])
        """
        from itertools import combinations_with_replacement
        from rdkit import RDLogger as _RDLogger

        pool  = _collect_radical_pool(compounds, include_h_radical)
        names = [c.name or c.formula for c in compounds]

        logger.info(
            "solid_state_recombination [%s]: %d radical species in pool: %s",
            ", ".join(names), len(pool), pool,
        )

        # Deduplicate against all existing fragments across all compounds.
        seen = {
            f.smiles
            for c in compounds
            for f in c.fragments
            if f.smiles is not None
        }

        new_frags: list[Compound] = []
        n_pairs = 0

        # Suppress RDKit warning triggered by H• intermediates.
        _rdlog = _RDLogger.logger()
        _rdlog.setLevel(_RDLogger.ERROR)
        try:
            for i, j in combinations_with_replacement(range(len(pool)), 2):
                products = combine_radicals(pool[i], pool[j])
                if products:
                    n_pairs += 1
                for smi in products:
                    add_fragment(hcap_smiles(smi) or smi, (), new_frags, seen)
        finally:
            _rdlog.setLevel(_RDLogger.WARNING)

        logger.info(
            "solid_state_recombination: %d reactive pair(s) → %d new product(s).",
            n_pairs, len(new_frags),
        )
        return new_frags

    def do_solid_state_recombination(
        self,
        include_h_radical: bool = True,
    ) -> list:
        """
        Solid-state radical recombination for this compound.

        Convenience wrapper around the classmethod
        :meth:`solid_state_recombination` for the single-compound case.
        Results are appended to ``self.fragments``.

        Requires :meth:`do_fragmentation` to have been called first.

        Parameters
        ----------
        include_h_radical : bool
            Include H• in the radical pool (default True).

        Returns
        -------
        list of newly added Compound objects (in fragment mode)
        """
        _ = self.fragments  # guard: raises if do_fragmentation() not called
        new_frags = Compound.solid_state_recombination([self], include_h_radical)
        self._fragments.extend(new_frags)
        self._fragments.sort(key=lambda f: (f.n_heavy, f.formula))
        return new_frags

    # ------------------------------------------------------------------
    # Fragment access  (lazy enumeration, parent mode only)
    # ------------------------------------------------------------------

    @property
    def raw_fragments(self) -> list:
        """
        Bare subgraph cuts with radical electrons marking broken bonds.

        These are the direct products of bond cleavage with no H added:
        boundary atoms carry SetNumRadicalElectrons() equal to the total
        bond order of cut bonds.  Use this for MLIP recombination studies
        or to inspect the primary fragmentation pattern.
        """
        if self._raw_fragments is None:
            raise RuntimeError(
                "raw_fragments is not available — call do_fragmentation() first."
            )
        return self._raw_fragments

    @property
    def fragments(self) -> list:
        """
        Chemically stabilised species derived from raw_fragments.

        Built by three rules applied in order:
          1. H-cap each radical raw fragment (closed-shell neutral/ion)
          2. SMILES-level radical bonding for every pair of radical fragments
             (radical+radical and anion+radical via combine_radicals)
          3. Deprotonated counterparts of cationic fragments (for DB lookup)

        Use raw_fragments for MLIP recombination; use this for MS peak assignment.
        """
        if self._fragments is None:
            raise RuntimeError(
                "fragments is not available — call do_fragmentation() first."
            )
        return self._fragments

    def fragments_by_size(self, n_heavy: int) -> list:
        """Return only fragments with exactly *n_heavy* heavy atoms."""
        return [f for f in self.fragments if f.n_heavy == n_heavy]

    def fragments_with(self, pattern: str) -> list:
        """
        Return fragments that contain *pattern* as a substructure.

        *pattern* is interpreted as SMARTS (falls back to SMILES).
        """
        from rdkit import Chem

        query = Chem.MolFromSmarts(pattern)
        if query is None:
            raise ValueError(f"Invalid SMARTS/SMILES pattern: {pattern!r}")
        results = []
        for frag in self.fragments:
            if frag.smiles is None:
                continue
            mol = Chem.MolFromSmiles(frag.smiles)
            if mol is not None and mol.HasSubstructMatch(query):
                results.append(frag)
        return results

    # ------------------------------------------------------------------
    # Internal enumeration
    # ------------------------------------------------------------------

    def enumerate_raw(self, mol=None) -> list:
        """
        Enumerate all connected subsets of heavy atoms as *raw* (radical) fragments.

        A raw fragment represents the molecule after one or more bonds have been
        cut but *before* any stabilisation (H-capping, β-scission, radical
        pairing).  The cut sites are encoded as unpaired radical electrons in the
        SMILES string — the same convention RDKit uses for radical species.

        **Algorithm**

        1. Every connected subset of heavy atoms of size 1 … *max_heavy* is
           considered (power-set enumeration).
        2. The boundary bonds — bonds connecting the subset to the rest of the
           molecule — are inspected for *lability*:
           * Any **non-labile** crossing bond (aromatic C–C, C=O, C=C conjugated,
             triple bonds) disqualifies the entire subset: EI cannot break these
             bonds as a primary event.
           * The number of labile crossing bonds must not exceed *max_cuts*.
        3. Surviving subsets are converted to SMILES with
           :func:`canonical_smiles_radical`, where each atom at a cut site carries
           one extra radical electron per broken bond.
        4. Duplicate SMILES are dropped (controlled by ``self.deduplicate``).
        5. 3-D positions are taken directly from the parent conformer — no
           re-embedding at this stage.

        Parameters
        ----------
        mol : RDKit Mol, optional
            Molecule to enumerate cuts on.  Defaults to ``self._mol`` (the parent).
            :meth:`do_fragmentation` passes H-capped level-1 fragments here when
            ``depth > 1`` to model secondary fragmentation cascades.

        Returns
        -------
        list of Compound
            Raw fragments sorted by (n_heavy, formula).  Each fragment has
            ``smiles``, ``atoms`` (ASE Atoms), and ``parent_idx`` set.
        """
        from rdkit import Chem

        mol  = mol if mol is not None else self._mol
        conf = mol.GetConformer()
        adj  = adjacency(mol)
        n    = mol.GetNumAtoms()

        # Kekulize so aromatic bonds become explicit SINGLE/DOUBLE.
        kmol = Chem.RWMol(mol)
        Chem.Kekulize(kmol, clearAromaticFlags=True)

        heavy_idx = [i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        h_set     = {i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() == 1}

        n_heavy   = len(heavy_idx)
        max_heavy = self.max_heavy if self.max_heavy is not None else n_heavy

        if n_heavy > 15:
            logger.warning(
                "Parent molecule has %d heavy atoms - enumeration may be slow. "
                "Consider setting max_heavy to limit the search.",
                n_heavy,
            )

        logger.info(
            "Enumerating raw fragments: %d heavy atoms + %d H  "
            "(max_heavy=%d, max_cuts=%d)",
            n_heavy, len(h_set), max_heavy, self.max_cuts,
        )

        seen:    set[str]      = set()
        results: list[Compound] = []

        for size in range(1, max_heavy + 1):
            for heavy_sub in combinations(heavy_idx, size):
                fs = frozenset(heavy_sub)
                if not is_connected(fs, adj):
                    continue

                subset = set(heavy_sub)
                if self.include_h:
                    for hi in h_set:
                        if adj[hi] & subset:
                            subset.add(hi)

                # Identify bonds crossing the fragment boundary.
                # If ANY crossing bond is non-labile (aromatic or double/triple),
                # skip this fragment — EI cannot break those bonds as a primary
                # event.  Among the labile crossings, enforce max_cuts.
                # Use the original mol (not kmol) so aromatic flags are intact.
                crossing = [
                    mol.GetBondBetweenAtoms(i, j)
                    for i in subset for j in adj[i]
                    if j not in subset
                ]
                if any(not is_labile(b) for b in crossing):
                    continue
                n_cuts = len(crossing)
                if n_cuts > self.max_cuts:
                    continue

                smi = canonical_smiles_radical(kmol, subset)

                if self.deduplicate and smi is not None:
                    if smi in seen:
                        continue
                    seen.add(smi)

                ordered    = sorted(subset)
                syms       = [mol.GetAtomWithIdx(i).GetSymbol() for i in ordered]
                pos        = np.array([list(conf.GetAtomPosition(i)) for i in ordered])
                frag_atoms = Atoms(symbols=syms, positions=pos)

                # Local indices of atoms that have at least one cut bond
                boundary = tuple(
                    k for k, pi in enumerate(ordered)
                    if any(j not in subset for j in adj[pi])
                )

                results.append(Compound.from_fragment(
                    smiles       = smi,
                    atoms        = frag_atoms,
                    parent_idx   = tuple(ordered),
                    boundary_idx = boundary,
                ))

        results.sort(key=lambda f: (f.n_heavy, f.formula))
        logger.info("Found %d unique raw fragments.", len(results))
        return results

    def build_stable(self) -> list:
        """
        Convert raw (radical) fragments to stable, database-queryable species.

        Applies four chemical rules in sequence to ``self.raw_fragments``.
        All results are deduplicated by canonical SMILES before being added.

        **Rule 1 — H-capping**
            Every radical open-valence is saturated with a hydrogen atom.
            This is the simplest stabilisation: the fragment retains its heavy-atom
            skeleton and all unpaired electrons are quenched by implicit H.
            Example: ``[CH2•]CC`` → ``CCC`` (propane)

        **Rule 2 — β-scission**
            For each radical centre α and each *single* bond α–β–γ:
            * Break the β–γ bond (γ gains a radical electron).
            * Upgrade α–β from single to double bond (α uses its radical electron
              to form the π bond).
            Both resulting fragments (the alkene/imine/etc. and the γ-radical, H-capped)
            are added to the stable list.  β-scission is the primary route by which
            alkyl radicals give alkenes; it also explains why EI spectra of
            carbamates and esters show strong olefin peaks.
            Example: tert-butyl radical ``[C•](C)(C)C`` → isobutylene ``C=C(C)C``

        **Rule 3 — Radical pairing**
            All pairs (A, B) of raw fragments are tried for covalent bond formation:

            * *Radical + radical*: both carry unpaired electrons;
              :func:`combine_radicals` enumerates single/double/triple bonds between
              every pair of radical centres.
            * *Anion + radical*: the anion (charge < 0, spin = 0) is converted to
              its neutral radical via :func:`anion_to_radical` (models EI
              electron-stripping, e.g. I⁻ → I•), then :func:`combine_radicals` is
              applied.  This is the mechanism by which ammonium iodide salts produce
              covalent iodo-products (CH₃I, HI, …) in the gas phase.

        **Rule 4 — Deprotonation**
            For each cationic fragment already in the stable list (charge > 0),
            :func:`deprotonate_smiles` removes one proton from the first protonated
            heteroatom (N, O, or S).  EI-MS reference databases (NIST, MassBank)
            index *neutral* closed-shell species, so cationic fragments such as
            ``CC[NH3+]`` must be deprotonated to ``CCN`` before a database lookup.

        .. note::
            Ion pairs are excluded.  See ``project_deferred.md``.

        Returns
        -------
        list of Compound
            Stable fragments sorted by (n_heavy, formula).
        """
        seen:    set[str]      = set()
        results: list[Compound] = []
        raws = self.raw_fragments

        # Rule 1: H-cap every raw fragment.
        for raw in raws:
            smi = hcap_smiles(raw.smiles)
            add_fragment(smi, raw.parent_idx, results, seen)
        n1 = len(results)
        logger.debug("  Rule 1 (H-cap):       %d fragment(s)", n1)

        # Rule 2: β-scission products of radical fragments.
        for raw in raws:
            if raw.spin == 0:
                continue
            for smi in beta_scission_products(raw.smiles):
                add_fragment(smi, raw.parent_idx, results, seen)
        n2 = len(results) - n1
        logger.debug("  Rule 2a (β-scission): %d new fragment(s)", n2)

        # Rule 2b: H-migration (McLafferty rearrangement) on radical fragments.
        _before_2b = len(results)
        for raw in raws:
            if raw.spin == 0:
                continue
            for smi in h_migration_products(raw.smiles):
                add_fragment(smi, raw.parent_idx, results, seen)
        n2b = len(results) - _before_2b
        logger.debug("  Rule 2b (H-migr):     %d new fragment(s)", n2b)

        # Rule 3: radical pairing (radical+radical and anion+radical).
        add_radical_pairs(raws, results, seen)
        n3 = len(results) - n1 - n2 - n2b
        logger.debug("  Rule 3 (rad pairing): %d new fragment(s)", n3)

        # Rule 4: deprotonated counterparts of cationic fragments.
        for frag in list(results):
            if frag.charge <= 0:
                continue
            smi = deprotonate_smiles(frag.smiles)
            if smi is not None:
                add_fragment(smi, frag.parent_idx, results, seen)
        n4 = len(results) - n1 - n2 - n2b - n3
        logger.debug("  Rule 4 (deprotonate): %d new fragment(s)", n4)

        # Rule 5: α-dehydrogenation — loss of H from carbons α to
        # heteroatoms followed by β-scission.  Produces vinyl halides,
        # enol ethers, etc. from closed-shell recombination products.
        _before_5 = len(results)
        for frag in list(results):
            if frag.spin > 0 or frag.charge != 0:
                continue
            for smi in alpha_dehydrogenation_products(frag.smiles):
                add_fragment(smi, frag.parent_idx, results, seen)
        n5 = len(results) - _before_5
        logger.debug("  Rule 5 (α-dehydro):   %d new fragment(s)", n5)

        results.sort(key=lambda f: (f.n_heavy, f.formula))
        logger.info(
            "Built %d stable fragments from %d raw fragments "
            "(H-cap: %d | β-scission: %d | H-migr: %d | rad-pairs: %d "
            "| deprotonate: %d | α-dehydro: %d).",
            len(results), len(raws), n1, n2, n2b, n3, n4, n5,
        )
        return results

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a compact MS-oriented table of the parent and all fragments."""
        frags = self.fragments
        label = f"  ({self.name})" if self.name else ""
        print(f"Compound : {self.smiles}{label}")
        print(f"Formula  : {self.formula}")
        print(f"Mono     : {self.monoisotopic_mass:.4f} Da  (nominal {self.nominal_mass})")
        print(f"Charge   : {self.charge:+d}")
        print(f"Fragments: {len(frags)}\n")

        header = (
            f"{'#':>4}  {'SMILES':<28}  {'Formula':<14}  "
            f"{'Mono (Da)':>10}  {'Nom':>5}  {'Q':>3}  {'Spin':>5}"
        )
        print(header)
        print("-" * len(header))
        for i, frag in enumerate(frags):
            print(
                f"{i+1:>4}  "
                f"{(frag.smiles or '?'):<28}  "
                f"{frag.formula:<14}  "
                f"{frag.monoisotopic_mass:>10.4f}  "
                f"{frag.nominal_mass:>5}  "
                f"{frag.charge:>+3}  "
                f"{frag.spin:>5}"
            )

    def summary_raw(self) -> None:
        """Print a compact table of raw (radical) fragments."""
        frags = self.raw_fragments
        label = f"  ({self.name})" if self.name else ""
        print(f"Compound : {self.smiles}{label}  [raw radical fragments]")
        print(f"Raw fragments: {len(frags)}\n")
        header = (
            f"{'#':>4}  {'SMILES':<28}  {'Formula':<14}  "
            f"{'Mono (Da)':>10}  {'Q':>3}  {'Spin':>5}"
        )
        print(header)
        print("-" * len(header))
        for i, frag in enumerate(frags):
            print(
                f"{i+1:>4}  "
                f"{(frag.smiles or '?'):<28}  "
                f"{frag.formula:<14}  "
                f"{frag.monoisotopic_mass:>10.4f}  "
                f"{frag.charge:>+3}  "
                f"{frag.spin:>5}"
            )

    def view(self):
        """Open this compound in the ASE GUI viewer."""
        return view(self.atoms)

    def view_fragments(self, indices=None, raw: bool = False):
        """
        Open fragments in the ASE GUI viewer.

        Parameters
        ----------
        indices : int or list of ints — which fragments to show (default: all)
        raw     : if True show raw radical fragments; otherwise stable (default False)
        """
        frags = self.raw_fragments if raw else self.fragments
        if indices is None:
            atoms_list = [f.atoms for f in frags]
        elif isinstance(indices, int):
            atoms_list = [frags[indices].atoms]
        else:
            atoms_list = [frags[i].atoms for i in indices]
        return view(atoms_list)

    def relax_fragments(self, calc, fmax: float = 5e-2, steps: int = 100,
                        log_dir=None):
        """
        Relax all stable fragment geometries in-place using an ASE calculator.

        Requires :meth:`do_fragmentation` to have been called first.

        Parameters
        ----------
        calc    : ASE-compatible calculator (e.g. FAIRChemCalculator)
        fmax    : FIRE2 force convergence threshold (eV/Ang)
        steps   : maximum number of FIRE2 steps per fragment
        log_dir : directory to write per-fragment trajectory and log files
                  named <index>_<formula>.traj / .log; created if absent
        """
        n_frags = len(self.fragments)
        logger.info("relax_fragments: relaxing %d fragment(s) (fmax=%.1e, steps=%d) ...",
                    n_frags, fmax, steps)

        for i, frag in enumerate(self.fragments):
            logger.debug("  [%d/%d] %s | charge=%+d | spin=%d",
                         i + 1, n_frags, frag.formula, frag.charge, frag.spin)
            traj = log = None
            if log_dir is not None:
                d    = Path(log_dir)
                d.mkdir(parents=True, exist_ok=True)
                stem = f"{i:03d}_{frag.formula}"
                traj = str(d / f"{stem}.traj")
                log  = str(d / f"{stem}.log")
            frag.relax(calc, fmax=fmax, steps=steps,
                       trajectory=traj, logfile=log)

        logger.info("relax_fragments: done.")

    def relax_recombination(
        self,
        calc,
        gap:       float = 1.5,
        fmax:      float = 5e-2,
        steps:     int   = 100,
        log_dir          = None,
        bond_mult: float = 1.2,
    ) -> list:
        """
        MLIP-verified pairwise recombination of raw_fragments.

        Each pair of radical raw fragments is placed with their reactive centres
        *gap* Å apart and relaxed with FIRE2.  Only candidates that converge to
        a single connected molecule are kept and added to the fragment library.
        Products receive ``smiles=None`` (3-D geometry only).

        Call :meth:`do_recombination` first for the fast SMILES-level step, then
        call this for the more expensive geometry-verified step.

        Requires :meth:`do_fragmentation` to have been called first.

        Parameters
        ----------
        calc      : ASE-compatible calculator (e.g. FAIRChemCalculator)
        gap       : reactive-centre separation for initial placement (Å)
        fmax      : FIRE2 force convergence threshold (eV/Å)
        steps     : maximum FIRE2 steps per candidate
        log_dir   : directory for trajectory/log files; created if absent
        bond_mult : covalent-radius multiplier for bond detection (default 1.2)

        Returns
        -------
        list of newly added Compound objects (in fragment mode, smiles=None)
        """
        # Raises RuntimeError if do_fragmentation() has not been called yet.
        _ = self.fragments

        from .recombination import relax_all_recombinations

        products = relax_all_recombinations(
            self.raw_fragments, calc,
            gap=gap, fmax=fmax, steps=steps,
            log_dir=log_dir, bond_mult=bond_mult,
        )

        added = []
        for atoms in products:
            frag = Compound.from_fragment(smiles=None, atoms=atoms, parent_idx=())
            self._fragments.append(frag)
            added.append(frag)

        logger.info(
            "relax_recombination: added %d MLIP-verified product(s) to fragment library.",
            len(added),
        )
        return added

    # ------------------------------------------------------------------
    # Bridge to rgakit
    # ------------------------------------------------------------------

    def load_ms_spectra(
        self,
        source,
        *,
        overwrite:   bool = False,
        max_workers: int  = 8,
    ) -> int:
        """
        Populate ``frag.spectrum`` for every fragment using *source*.

        Can be called multiple times with different sources to fill gaps
        progressively (lower-priority sources are tried first, higher-priority
        ones called later with ``overwrite=True`` to replace).

        Parameters
        ----------
        source      : one of:

                      * ``"nist"`` — NIST WebBook (requires network; uses InChIKey)
                      * :class:`~rgakit.databases.InSilicoDatabase` instance
                      * :class:`~rgakit.databases.MassBankDatabase` instance

        overwrite   : if False (default), skip fragments that already have a
                      spectrum.  Set True to replace existing entries.
        max_workers : thread-pool size for concurrent lookups (default 8).

        Returns
        -------
        int — number of fragments whose spectrum was newly set in this call.
        """
        from concurrent.futures import ThreadPoolExecutor
        from rgakit.spectrum import MassSpectrum

        # Build a source-specific fetch function
        if isinstance(source, str):
            if source.lower() != "nist":
                raise ValueError(
                    f"Unknown source string {source!r}. "
                    "Use 'nist' or pass an InSilicoDatabase / MassBankDatabase."
                )
            def _fetch(frag) -> "MassSpectrum | None":
                ik = frag.inchikey
                if ik is None:
                    return None
                try:
                    return MassSpectrum.from_nist(inchikey=ik)
                except Exception:
                    return None
        else:
            from rgakit.databases import InSilicoDatabase, MassBankDatabase
            if isinstance(source, InSilicoDatabase):
                def _fetch(frag) -> "MassSpectrum | None":
                    if frag.smiles is None:
                        return None
                    return source.get(smiles=frag.smiles)
            elif isinstance(source, MassBankDatabase):
                def _fetch(frag) -> "MassSpectrum | None":
                    ik = frag.inchikey
                    if ik is None:
                        return None
                    return source.get(inchikey=ik)
            else:
                raise TypeError(
                    f"Unsupported source type {type(source).__name__!r}. "
                    "Pass 'nist', an InSilicoDatabase, or a MassBankDatabase."
                )

        frags   = list(self.fragments)
        targets = [f for f in frags
                   if f.charge == 0 and (overwrite or f.spectrum is None)]

        if not targets:
            logger.debug("load_ms_spectra: no fragments to update.")
            return 0

        newly_set = 0

        def _fetch_and_set(frag):
            nonlocal newly_set
            spec = _fetch(frag)
            if spec is not None:
                frag.spectrum = spec
                newly_set += 1

        n_workers = min(max_workers, len(targets))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_fetch_and_set, targets))

        n_total = sum(1 for f in frags if f.spectrum is not None)
        logger.info(
            "load_ms_spectra: %d new hit(s) — %d/%d fragments have spectra.",
            newly_set, n_total, len(frags),
        )
        return newly_set

    def to_spectrum(self, nist_lookup: bool = False, **kwargs):
        """
        Convert this compound to a MassSpectrum for use in a SpectraLibrary.

        Delegates to :func:`~rgakit.molecule.bridge.fragment_to_spectrum`.
        All keyword arguments are forwarded (e.g. ``databases=``,
        ``nist_lookup=``, ``formula_fallback=``).

        Parameters
        ----------
        nist_lookup : attempt NIST lookup (default True); set False for offline use
        """
        from rgakit.molecule.bridge import fragment_to_spectrum
        return fragment_to_spectrum(self, nist_lookup=nist_lookup, **kwargs)

    def to_library(
        self,
        nist_lookup:      bool = False,
        include_raw:      bool = False,
        formula_fallback: bool = False,
        formula_all:      bool = False,
        max_workers:      int  = 8,
        databases               = None,
    ):
        """
        Build a SpectraLibrary from the stable fragments of this compound.

        Fragments with no hit in any configured source are silently skipped.
        Lookups are performed concurrently using a thread pool.

        Parameters
        ----------
        nist_lookup      : query NIST as a fallback when all provided databases
                           miss (default False)
        include_raw      : also include raw (radical) fragments (default False)
        formula_fallback : use formula-based NIST search when InChIKey lookup
                           fails (default False, ambiguous — first isomer only)
        formula_all      : for NIST misses, add ALL NIST isomers that share
                           the same molecular formula (default False)
        max_workers      : number of threads for concurrent lookups (default 8)
        databases        : one database or a list of databases, queried in order
                           (InSilicoDatabase, NistDatabase, MassBankDatabase,
                           MonaLocalDatabase, …); all share the same interface
        """
        from rgakit.molecule.bridge import compound_to_library
        return compound_to_library(
            self,
            nist_lookup      = nist_lookup,
            include_raw      = include_raw,
            formula_fallback = formula_fallback,
            formula_all      = formula_all,
            max_workers      = max_workers,
            databases        = databases,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        smi = self.smiles or "?"
        if self._mol is None:
            # Fragment mode
            return (
                f"Compound({smi!r}, "
                f"mono={self.monoisotopic_mass:.4f} Da, "
                f"charge={self.charge:+d}, "
                f"{self.n_heavy} heavy)"
            )
        # Parent mode
        n_raw    = len(self._raw_fragments) if self._raw_fragments is not None else "?"
        n_stable = len(self._fragments)     if self._fragments     is not None else "?"
        label    = f"{self.name!r}, " if self.name else ""
        return (
            f"Compound({label}smiles={smi!r}, "
            f"raw={n_raw}, stable={n_stable})"
        )

    def __len__(self) -> int:
        return len(self.fragments)
