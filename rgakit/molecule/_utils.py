"""
_utils.py
Stateless conversion and graph-utility functions for the molecule layer.

All functions here are pure conversions between representations with no 3-D
embedding, no radical chemistry, and no network calls.  They are imported
freely by other molecule submodules and by rgakit.databases without circular
issues.

Sections
Structure conversions
    ase_to_rdkit        convert ASE Atoms to RDKit Mol
    rdkit_to_ase        convert RDKit Mol to ASE Atoms

Identifier conversions
    smiles_to_inchikey  compute InChIKey from SMILES (RDKit, local)
    smiles_from_inchi   compute canonical SMILES from InChI (RDKit, local)

SMILES property extraction
    _monoisotopic_from_smiles  exact mass (Da) from SMILES
    _formal_charge_from_smiles net formal charge from SMILES

RDKit Mol graph helpers
    _adjacency               adjacency dict from RDKit Mol
    _is_connected            check connectivity of an atom subset
    _canonical_smiles_radical  canonical SMILES for an atom subset, with radical electrons on boundary atoms

ASE Atoms graph helpers
    atoms_to_graph           build a NetworkX Graph from ASE Atoms (public)
    _extract_components      split ASE Atoms into connected components
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structure conversions
# ---------------------------------------------------------------------------

def ase_to_rdkit(atoms: Atoms):
    """Convert an ASE Atoms object to an RDKit Mol via an XYZ block."""
    from rdkit import Chem

    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    xyz  = f"{len(syms)}\n\n" + "\n".join(
        f"{s}  {x:.6f}  {y:.6f}  {z:.6f}"
        for s, (x, y, z) in zip(syms, pos)
    )
    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        raise ValueError("Could not convert ASE Atoms to RDKit Mol.")
    return mol


def rdkit_to_ase(mol) -> Atoms:
    """Convert an RDKit Mol (with conformer) to an ASE Atoms object."""
    conf = mol.GetConformer()
    n    = mol.GetNumAtoms()
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    pos  = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
    return Atoms(symbols=syms, positions=pos)


# ---------------------------------------------------------------------------
# Identifier conversions
# ---------------------------------------------------------------------------

def smiles_to_inchikey(smiles: str) -> str | None:
    """Compute the InChIKey for a SMILES string via RDKit (no network)."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        RDLogger.DisableLog("rdApp.warning")
        try:
            inchi = MolToInchi(mol)
        finally:
            RDLogger.EnableLog("rdApp.warning")
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception as exc:
        logger.debug("InChIKey computation failed for %r: %s", smiles, exc)
        return None


def smiles_from_inchi(inchi: str) -> str | None:
    """Convert an InChI string to canonical SMILES via RDKit."""
    try:
        from rdkit.Chem.inchi import MolFromInchi
        from rdkit import Chem
        mol = MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as exc:
        logger.debug("InChI to SMILES failed for %r: %s", inchi, exc)
        return None


# ---------------------------------------------------------------------------
# SMILES property extraction
# ---------------------------------------------------------------------------

def _monoisotopic_from_smiles(smiles: str) -> float | None:
    """Exact (monoisotopic) mass from a SMILES string via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return rdMolDescriptors.CalcExactMolWt(mol)


def _formal_charge_from_smiles(smiles: str) -> int:
    """Net formal charge from a SMILES string via RDKit."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return Chem.GetFormalCharge(mol)


# ---------------------------------------------------------------------------
# RDKit Mol graph helpers
# ---------------------------------------------------------------------------

def _adjacency(mol) -> dict[int, set[int]]:
    """Build an adjacency dict {atom_idx: set(neighbour_idxs)} from an RDKit Mol."""
    adj: dict[int, set[int]] = {i: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _is_connected(subset: frozenset[int], adj: dict[int, set[int]]) -> bool:
    """Return True if all atoms in *subset* form a single connected component."""
    if len(subset) <= 1:
        return True
    start   = next(iter(subset))
    visited = {start}
    stack   = [start]
    while stack:
        for nb in adj[stack.pop()]:
            if nb in subset and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return visited == subset



def _canonical_smiles_radical(mol, subset: set[int]) -> str | None:
    """
    Like _canonical_smiles but marks unsatisfied valences as radical electrons.

    Each bond cut during fragment extraction contributes its bond order to the
    radical electron count of the boundary atom, giving the correct spin state
    for uncapped (non-H-saturated) raw fragments.
    """
    from rdkit import Chem

    _BO = {
        Chem.BondType.SINGLE:   1,
        Chem.BondType.DOUBLE:   2,
        Chem.BondType.TRIPLE:   3,
        Chem.BondType.AROMATIC: 1,
    }

    rw      = Chem.RWMol()
    idx_map = {}
    for new_i, old_i in enumerate(sorted(subset)):
        atom = mol.GetAtomWithIdx(old_i)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNoImplicit(True)
        rw.AddAtom(new_atom)
        idx_map[old_i] = new_i

    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bi in idx_map and bj in idx_map:
            rw.AddBond(idx_map[bi], idx_map[bj], bond.GetBondType())

    for old_i, new_i in idx_map.items():
        radicals = 0
        for bond in mol.GetAtomWithIdx(old_i).GetBonds():
            if bond.GetOtherAtomIdx(old_i) not in idx_map:
                radicals += _BO.get(bond.GetBondType(), 1)
        if radicals > 0:
            rw.GetAtomWithIdx(new_i).SetNumRadicalElectrons(radicals)

    try:
        return Chem.MolToSmiles(rw.GetMol())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ASE Atoms graph helpers
# ---------------------------------------------------------------------------

def atoms_to_graph(atoms: Atoms, mult: float = 1.2) -> nx.Graph:
    """
    Build a NetworkX bond graph from an ASE Atoms object.

    Bonds are detected via ASE natural covalent cutoffs scaled by *mult*.
    Nodes carry an 'element' attribute (chemical symbol).

    Parameters
    ----------
    atoms : ASE Atoms object
    mult  : covalent-radius multiplier for bond detection (default 1.2)
    """
    from ase.neighborlist import natural_cutoffs, NeighborList

    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=mult)
    nl      = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    G = nx.Graph()
    G.add_nodes_from((i, {"element": s}) for i, s in enumerate(symbols))
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            G.add_edge(i, j)
    return G


def _extract_components(atoms: Atoms, bond_mult: float = 1.2) -> list[Atoms]:
    """
    Return one Atoms object per connected component of the bond graph.

    Returns a list of length 1 (the original object) when the structure is
    fully connected.

    Parameters
    ----------
    atoms     : ASE Atoms object
    bond_mult : covalent-radius multiplier (default 1.2)
    """
    G          = atoms_to_graph(atoms, mult=bond_mult)
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return [atoms]
    return [atoms[sorted(c)] for c in components]
