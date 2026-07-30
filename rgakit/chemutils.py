"""
chemutils.py
------------
Chemistry conversion utilities with RDKit-first, Open Babel fallback.

All functions try RDKit first.  If RDKit fails (exotic valence, unsupported
bond types), Open Babel is attempted as a fallback.  Both are fully local —
no network access needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _suppress_ob_warnings():
    """Suppress Open Babel's stderr warnings."""
    try:
        from openbabel import openbabel as ob
        ob.obErrorLog.SetOutputLevel(0)
    except (ImportError, AttributeError):
        pass


_suppress_ob_warnings()


def smiles_from_inchi(inchi: str) -> str | None:
    """Convert an InChI string to canonical SMILES."""
    # RDKit first
    try:
        from rdkit.Chem.inchi import MolFromInchi
        from rdkit.Chem import MolToSmiles
        mol = MolFromInchi(inchi)
        if mol is not None:
            return MolToSmiles(mol)
    except Exception:
        pass

    # Open Babel fallback
    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("inchi", "smi")
        mol = ob.OBMol()
        if conv.ReadString(mol, inchi):
            smi = conv.WriteString(mol).strip()
            if smi:
                logger.debug("smiles_from_inchi: used Open Babel fallback for %s…", inchi[:30])
                return smi
    except ImportError:
        pass

    return None


def inchi_from_smiles(smiles: str) -> str | None:
    """Convert a SMILES string to InChI."""
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.inchi import MolToInchi
        mol = MolFromSmiles(smiles)
        if mol is not None:
            return MolToInchi(mol)
    except Exception:
        pass

    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("smi", "inchi")
        mol = ob.OBMol()
        if conv.ReadString(mol, smiles):
            inchi = conv.WriteString(mol).strip()
            if inchi:
                logger.debug("inchi_from_smiles: used Open Babel fallback")
                return inchi
    except ImportError:
        pass

    return None


def inchikey_from_smiles(smiles: str) -> str | None:
    """Convert a SMILES string to InChIKey."""
    inchi = inchi_from_smiles(smiles)
    if inchi is None:
        return None
    return inchikey_from_inchi(inchi)


def inchikey_from_inchi(inchi: str) -> str | None:
    """Convert an InChI string to InChIKey."""
    try:
        from rdkit.Chem.inchi import InchiToInchiKey
        return InchiToInchiKey(inchi)
    except Exception:
        pass

    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("inchi", "inchikey")
        mol = ob.OBMol()
        if conv.ReadString(mol, inchi):
            key = conv.WriteString(mol).strip()
            if key:
                return key
    except ImportError:
        pass

    return None


def formula_from_smiles(smiles: str) -> str | None:
    """Compute molecular formula from SMILES."""
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        mol = MolFromSmiles(smiles)
        if mol is not None:
            return CalcMolFormula(mol)
    except Exception:
        pass

    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("smi", "smi")
        mol = ob.OBMol()
        if conv.ReadString(mol, smiles):
            return mol.GetFormula()
    except ImportError:
        pass

    return None


def mol_weight_from_smiles(smiles: str) -> float | None:
    """Compute average molecular weight from SMILES."""
    try:
        from rdkit.Chem import MolFromSmiles, Descriptors
        mol = MolFromSmiles(smiles)
        if mol is not None:
            return round(Descriptors.MolWt(mol), 4)
    except Exception:
        pass

    try:
        from openbabel import openbabel as ob
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("smi", "smi")
        mol = ob.OBMol()
        if conv.ReadString(mol, smiles):
            return round(mol.GetMolWt(), 4)
    except ImportError:
        pass

    return None
