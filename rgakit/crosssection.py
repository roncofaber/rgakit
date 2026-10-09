"""
crosssection.py
---------------
70 eV electron-impact ionization cross sections for library scaling.

The BEB model (Kim, Santos & Barbarinho, At. Data Nucl. Data Tables 2000)
computes the total ionization cross section of a neutral molecule as a sum
over its occupied orbitals, using only ground-state quantities: orbital
binding energies B, orbital kinetic energies U, and occupancies N. These
come from a single-point Hartree-Fock calculation - no excited states or
response properties.

Scaling each library reference intensity by its compound's cross section
makes NNLS fitted weights proportional to relative partial pressures
instead of base-peak fractions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .library import SpectraLibrary
from .spectrum import MassSpectrum

logger = logging.getLogger("rgakit")

RYDBERG_EV = 13.605693
HARTREE_EV = 27.211386
BOHR_ANGSTROM = 0.529177


def _geometry(smiles: str) -> tuple[str, int, int]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0) == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
    conf = mol.GetConformer()
    atom_str = "; ".join(
        f"{atom.GetSymbol()} {conf.GetAtomPosition(i).x:.6f} "
        f"{conf.GetAtomPosition(i).y:.6f} {conf.GetAtomPosition(i).z:.6f}"
        for i, atom in enumerate(mol.GetAtoms())
    )
    charge = Chem.GetFormalCharge(mol)
    spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    return atom_str, charge, spin


def beb_sigma(smiles: str, basis: str = "sto-3g", energy: float = 70.0) -> float:
    """
    Total electron-impact ionization cross section (BEB model), in Å².

    Parameters
    ----------
    smiles : SMILES of the neutral molecule.
    basis  : Basis set for the single-point Hartree-Fock calculation
             (default STO-3G; sufficient for relative scaling).
    energy : Incident electron energy in eV (default 70).

    Returns
    -------
    Cross section in Å². Raises if the species cannot be computed (e.g.
    elements not covered by the basis set).
    """
    from pyscf import gto, scf

    atom_str, charge, spin = _geometry(smiles)
    pmol = gto.M(atom=atom_str, basis=basis, unit="Angstrom",
                 charge=charge, spin=spin, verbose=0)
    kin = pmol.intor("int1e_kin")

    sigma = 0.0
    if spin == 0:
        mf = scf.RHF(pmol)
        mf.kernel()
        channels = [(mf.mo_occ, mf.mo_energy, mf.mo_coeff)]
    else:
        mf = scf.UHF(pmol)
        mf.kernel()
        channels = [(mf.mo_occ[0], mf.mo_energy[0], mf.mo_coeff[0]),
                    (mf.mo_occ[1], mf.mo_energy[1], mf.mo_coeff[1])]

    for occ, en, coef in channels:
        U_mo = np.einsum("im,jm,ij->m", coef, coef, kin)
        for N, eps, U in zip(occ, en, U_mo):
            if N <= 0:
                continue
            B = -eps * HARTREE_EV
            if B <= 0.5 or B >= energy:
                continue
            t = energy / B
            u = U * HARTREE_EV / B
            S = 4 * np.pi * BOHR_ANGSTROM**2 * N * (RYDBERG_EV / B) ** 2
            term = (np.log(t) / 2) * (1 - 1 / t**2) + 1 - 1 / t - np.log(t) / (t + 1)
            sigma += S / (t + u + 1) * term

    if sigma <= 0:
        raise ValueError(f"BEB cross section is non-positive for {smiles!r}")
    return float(sigma)


def scale_library_by_cross_section(
    lib: SpectraLibrary,
    sigma_map: dict[str, float] | None = None,
    basis: str = "sto-3g",
    energy: float = 70.0,
    cache_path: str | Path | None = None,
) -> SpectraLibrary:
    """
    Return a new SpectraLibrary with each reference intensity scaled by its
    compound's total ionization cross section (BEB).

    Parameters
    ----------
    lib : SpectraLibrary to scale. Compounds without parseable SMILES or
        with failing calculations are kept unscaled.
    sigma_map : optional {compound name: cross section in Å²} overrides.
    basis : basis set for the single-point calculation.
    energy : incident electron energy in eV.
    cache_path : optional JSON cache of {inchikey: cross section}; results
        are stored and reused, making the scaling resumable.

    Returns
    -------
    New SpectraLibrary. After scaling, fitted weights are proportional to
    relative partial pressures (up to a common constant).
    """
    cache: dict[str, float] = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())

    new_specs = []
    for spec in lib:
        smi = spec.metadata.get("smiles")
        ik = spec.metadata.get("inchikey")
        s = (sigma_map or {}).get(spec.name)
        if s is None and ik and ik in cache:
            s = cache[ik]
        if s is None and smi:
            try:
                s = beb_sigma(smi, basis=basis, energy=energy)
                if ik:
                    cache[ik] = s
            except Exception as e:
                logger.warning(
                    "Cross section failed for %r (%s) - keeping unscaled.",
                    spec.name, e,
                )
        if s is None:
            new_specs.append(spec)
            continue
        new_specs.append(MassSpectrum(
            mz=spec.mz,
            intensity=spec.intensity * s,
            name=spec.name,
            metadata={**spec.metadata, "cross_section_A2": round(s, 4)},
        ))

    if cache_path is not None:
        Path(cache_path).write_text(json.dumps(cache, indent=1))

    return SpectraLibrary(new_specs)
