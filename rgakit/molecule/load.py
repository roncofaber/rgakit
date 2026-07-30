"""
load.py
Molecule I/O and 3-D embedding helpers.

Parses any molecular representation into a sanitised RDKit Mol with explicit H
and a 3-D MMFF-optimised conformer.  Format conversions between rdkit and ase
live in utils; this module is purely about geometry generation.
"""

from __future__ import annotations

from ase import Atoms

from .utils import ase_to_rdkit, rdkit_to_ase

import numpy as np


def embed_single(mol):
    """Embed a single-component RDKit Mol (with explicit H) and MMFF-optimise."""
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise RuntimeError(
                "RDKit could not generate a 3-D conformer. "
                "Check that the SMILES is valid and the molecule is not too large."
            )
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def embed_multicomponent(mol):
    """
    Embed a multi-component RDKit Mol with chemically sensible ion placement.

    Each fragment is embedded and MMFF-optimised independently.  Counter-ions
    are then translated so that their charged atom sits ~ION_DIST Å from the
    oppositely-charged atom of their partner fragment, rather than being
    placed at the origin and overlapping the main molecule.

    Fragments with no formal charges are offset along X beyond the bounding
    box of already-placed fragments.

    Returns a single combined RDKit Mol with one conformer.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ION_DIST = 3.5  # Angstroms — typical ion-pair contact distance

    rng = np.random.default_rng(42)

    frag_mols = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    embedded  = []
    for frag in frag_mols:
        frag = Chem.AddHs(frag)
        embed_single(frag)
        embedded.append(frag)

    def _charged_atoms(frag):
        conf = frag.GetConformer()
        out  = []
        for atom in frag.GetAtoms():
            q = atom.GetFormalCharge()
            if q != 0:
                p = conf.GetAtomPosition(atom.GetIdx())
                out.append((atom.GetIdx(), np.array([p.x, p.y, p.z]), q))
        return out

    def _all_positions(frag):
        conf = frag.GetConformer()
        return np.array([list(conf.GetAtomPosition(i))
                         for i in range(frag.GetNumAtoms())])

    def _translate(frag, delta):
        conf = frag.GetConformer()
        for k in range(frag.GetNumAtoms()):
            p = conf.GetAtomPosition(k)
            conf.SetAtomPosition(k, (p.x + delta[0],
                                     p.y + delta[1],
                                     p.z + delta[2]))

    for i in range(1, len(embedded)):
        my_charges = _charged_atoms(embedded[i])

        if not my_charges:
            all_pos  = np.vstack([_all_positions(embedded[j]) for j in range(i)])
            offset_x = all_pos[:, 0].max() + 5.0 - _all_positions(embedded[i])[:, 0].min()
            _translate(embedded[i], np.array([offset_x, 0.0, 0.0]))
            continue

        partner_pos = None
        my_atom_idx = None
        for j in range(i):
            for ai, my_p, my_q in my_charges:
                for aj, their_p, their_q in _charged_atoms(embedded[j]):
                    if my_q * their_q < 0:
                        partner_pos = their_p
                        my_atom_idx = ai
                        break
                if partner_pos is not None:
                    break
            if partner_pos is not None:
                break

        if partner_pos is None:
            all_pos  = np.vstack([_all_positions(embedded[j]) for j in range(i)])
            offset_x = all_pos[:, 0].max() + 5.0 - _all_positions(embedded[i])[:, 0].min()
            _translate(embedded[i], np.array([offset_x, 0.0, 0.0]))
            continue

        conf    = embedded[i].GetConformer()
        p       = embedded[i].GetConformer().GetAtomPosition(my_atom_idx)
        cur_pos = np.array([p.x, p.y, p.z])

        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        target = partner_pos + ION_DIST * direction
        _translate(embedded[i], target - cur_pos)

    combined = embedded[0]
    for frag in embedded[1:]:
        combined = AllChem.CombineMols(combined, frag)
    return combined


def load_rdkit_mol(source):
    """
    Parse *source* into a sanitised RDKit Mol with explicit H and a 3-D
    conformer.

    *source* may be:
      - SMILES string (single or multi-component, e.g. ``'cation.anion'``)
      - InChI string  (starts with ``"InChI="``)
      - path to a ``.mol`` or ``.sdf`` file
      - an ASE Atoms object  (converted via XYZ block)
      - an already-constructed RDKit Mol

    For multi-component SMILES (ion pairs, salts), each component is embedded
    independently and counter-ions are positioned near their charge partners.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if isinstance(source, Atoms):
        mol = ase_to_rdkit(source)
    elif isinstance(source, (str,)):
        if source.startswith("InChI="):
            from rdkit.Chem.inchi import MolFromInchi
            mol = MolFromInchi(source)
        elif source.endswith((".mol", ".sdf")):
            mol = Chem.MolFromMolFile(source, removeHs=False)
        else:
            mol = Chem.MolFromSmiles(source)
    else:
        from pathlib import Path
        if isinstance(source, Path):
            mol = Chem.MolFromMolFile(str(source), removeHs=False)
        else:
            mol = source  # assume already an RDKit Mol

    if mol is None:
        raise ValueError(f"Could not parse molecule from: {source!r}")

    if len(Chem.GetMolFrags(mol)) > 1:
        return embed_multicomponent(mol)

    mol = Chem.AddHs(mol)
    return embed_single(mol)


def embed_fragment(smi: str) -> Atoms | None:
    """
    Build a properly H-capped 3-D structure for a fragment SMILES.

    Unlike taking atom positions directly from the parent conformer, this
    embeds the SMILES-derived mol (with all capping H explicit) so that
    the returned Atoms object is fully valence-saturated and ready for
    visualisation or geometry analysis.

    Returns None if embedding fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)
    return rdkit_to_ase(mol)
