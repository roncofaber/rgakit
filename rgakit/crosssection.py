"""
crosssection.py
---------------
70 eV electron-impact ionization cross sections, and cross-section-weighted
molar ratios between co-products in a measured RGA spectrum.

The BEB model (Kim, Santos & Barbarinho, At. Data Nucl. Data Tables 2000)
computes the total ionization cross section of a neutral molecule as a sum
over its occupied orbitals, using only ground-state quantities: orbital
binding energies B, orbital kinetic energies U, and occupancies N. These
come from a single-point Hartree-Fock calculation - no excited states or
response properties.

:func:`molar_ratio` converts two co-product marker-channel intensities in a
raw spectrum into a cross-section-weighted molar ratio directly, without
going through :class:`~rgakit.library.SpectraLibrary`. Library reference
patterns cannot carry a cross-section factor, because
:meth:`MassSpectrum.on_grid` renormalizes each pattern by its own peak.
"""

from __future__ import annotations

import logging
import re

import numpy as np

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


def _period(z: int) -> int:
    return next(i for i, zmax in enumerate((2, 10, 18, 36, 54, 86, 118), 1) if z <= zmax)


def beb_sigma(smiles: str, basis: str = "def2-svp", energy: float = 70.0) -> float:
    """
    Total electron-impact ionization cross section (BEB model), in Å².

    Binary-encounter-Bethe model of Kim and Rudd (Phys. Rev. A 50, 3954
    (1994)) on Hartree-Fock orbitals, including their prescription of
    dividing the orbital kinetic energy by n - 2 for orbitals dominated by
    atomic orbitals of principal quantum number n > 2. Against 21 NIST
    SRD 107 values at 70 eV, the default basis gives a median ratio of 0.94
    with an rms deviation of 17 % (chlorofluorocarbons about 35 % low);
    STO-3G overestimates molecules with multiple bonds or lone pairs by up
    to a factor of 3.

    Parameters
    ----------
    smiles : SMILES of the neutral molecule.
    basis  : Basis set for the single-point Hartree-Fock calculation
             (default def2-SVP, which covers elements up to Rn).
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
    ovlp = pmol.intor("int1e_ovlp")
    ao_n = np.array([
        min(int(re.match(r"\d+", label[2]).group()),
            _period(pmol.atom_charge(label[0]) + pmol.atom_nelec_core(label[0])))
        for label in pmol.ao_labels(fmt=False)
    ])

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
        population = coef * (ovlp @ coef)
        for k, (N, eps, U) in enumerate(zip(occ, en, U_mo)):
            if N <= 0:
                continue
            B = -eps * HARTREE_EV
            if B <= 0.5 or B >= energy:
                continue
            n = ao_n[np.argmax(population[:, k])]
            if n > 2:
                U = U / (n - 2)
            t = energy / B
            u = U * HARTREE_EV / B
            S = 4 * np.pi * BOHR_ANGSTROM**2 * N * (RYDBERG_EV / B) ** 2
            term = (np.log(t) / 2) * (1 - 1 / t**2) + 1 - 1 / t - np.log(t) / (t + 1)
            sigma += S / (t + u + 1) * term

    if sigma <= 0:
        raise ValueError(f"BEB cross section is non-positive for {smiles!r}")
    return float(sigma)


#: Curated 70 eV total electron-impact ionization cross sections (in A^2),
#: limited to species with a verifiable NIST SRD 107 BEB value. Provenance
#: is per-entry in the "source" field, not here. Any other species falls
#: back to an on-demand :func:`beb_sigma` calculation rather than an
#: unverifiable literature or basis-set-corrected table entry.
REFERENCE_SIGMA_70EV: dict[str, dict[str, object]] = {
    "H2":   {"sigma_A2": 1.02, "smiles": "[H][H]", "inchikey": "UFHFLCQGNIYNRP-UHFFFAOYSA-N", "name": "Hydrogen",         "source": "NIST SRD 107 BEB"},
    "N2":   {"sigma_A2": 2.51, "smiles": "N#N",    "inchikey": "IJGRMHOSHXDMSA-UHFFFAOYSA-N", "name": "Nitrogen",        "source": "NIST SRD 107 BEB"},
    "CO":   {"sigma_A2": 2.52, "smiles": "[C-]#[O+]", "inchikey": "UGFAIRIUMAVXCW-UHFFFAOYSA-N", "name": "Carbon monoxide", "source": "NIST SRD 107 BEB"},
    "CO2":  {"sigma_A2": 3.52, "smiles": "O=C=O",  "inchikey": "CURLTUGMZLYLDI-UHFFFAOYSA-N", "name": "Carbon dioxide",  "source": "NIST SRD 107 BEB"},
    "H2O":  {"sigma_A2": 2.28, "smiles": "O",      "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N", "name": "Water",           "source": "NIST SRD 107 BEB"},
    "NH3":  {"sigma_A2": 3.04, "smiles": "N",      "inchikey": "QGZKDVFQNNGYKY-UHFFFAOYSA-N", "name": "Ammonia",         "source": "NIST SRD 107 BEB"},
    "CH4":  {"sigma_A2": 3.52, "smiles": "C",      "inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N", "name": "Methane",         "source": "NIST SRD 107 BEB"},
    "C2H4": {"sigma_A2": 5.12, "smiles": "C=C",    "inchikey": "VGGSQFUCUMXWEO-UHFFFAOYSA-N", "name": "Ethylene",        "source": "NIST SRD 107 BEB"},
    "C2H6": {"sigma_A2": 6.42, "smiles": "CC",     "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N", "name": "Ethane",          "source": "NIST SRD 107 BEB"},
    "C3H6": {"sigma_A2": 8.74, "smiles": "CC=C",   "inchikey": "QQONPFPTGQHPMA-UHFFFAOYSA-N", "name": "Propylene",       "source": "NIST SRD 107 BEB"},
    "H2S":  {"sigma_A2": 4.15, "smiles": "S",      "inchikey": "RWSOTUBLDIXVET-UHFFFAOYSA-N", "name": "Hydrogen sulfide", "source": "NIST SRD 107 BEB"},
}


def _build_lookup_tables() -> tuple[dict[str, float], dict[str, float]]:
    """
    Split the reference table into a structure-keyed table (RDKit canonical
    SMILES and InChIKey, checked after parsing the query as a molecule) and
    a name-keyed table (compound name and the table's own informal key,
    lowercased, checked only when the query does not parse as a molecule).

    A bare informal key like "CO" is also valid SMILES (methanol), so names
    are never consulted once the query parses as a molecule.
    """
    from rdkit import Chem

    canon: dict[str, float] = {}
    name: dict[str, float] = {}
    for key, info in REFERENCE_SIGMA_70EV.items():
        val = float(info["sigma_A2"])
        mol = Chem.MolFromSmiles(str(info["smiles"]))
        if mol is not None:
            canon[Chem.MolToSmiles(mol)] = val
        canon[str(info["inchikey"])] = val
        name[str(info["name"]).lower()] = val
        name[key.lower()] = val
    return canon, name


_CANON_LOOKUP, _NAME_LOOKUP = _build_lookup_tables()


def get_cross_section(
    identifier: str,
    basis: str = "def2-svp",
    energy: float = 70.0,
    use_reference: bool = True,
) -> float:
    """
    Retrieve the total electron-impact ionization cross section (in A^2) for
    a molecule specified by SMILES, InChIKey, or common name.

    When ``use_reference=True`` and ``energy == 70.0``, checks
    :data:`REFERENCE_SIGMA_70EV` first and falls back to :func:`beb_sigma`.
    A query that parses as a valid SMILES is resolved via its RDKit
    canonical form and InChIKey only, never by a bare-name/informal-key
    match: "CO" is valid SMILES for methanol, so it must not be matched
    against this table's own informal key for carbon monoxide. Only a
    query that fails to parse as SMILES is checked against the table's
    compound names.
    """
    if use_reference and abs(energy - 70.0) < 1e-3:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(identifier)
        finally:
            RDLogger.EnableLog("rdApp.*")
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if canon in _CANON_LOOKUP:
                return _CANON_LOOKUP[canon]
            ik = Chem.MolToInchiKey(mol)
            if ik in _CANON_LOOKUP:
                return _CANON_LOOKUP[ik]
        elif identifier in _CANON_LOOKUP:
            return _CANON_LOOKUP[identifier]  # a bare InChIKey does not parse as SMILES
        elif identifier.lower() in _NAME_LOOKUP:
            return _NAME_LOOKUP[identifier.lower()]
    return beb_sigma(identifier, basis=basis, energy=energy)


def _normalize_pattern(pattern: dict[int, float] | MassSpectrum) -> dict[int, float]:
    """
    Normalize a fragmentation pattern to branching ratios that sum to 1.

    This is the physically correct basis for :func:`molar_ratio`: for a
    species with total ionization cross section sigma and molar amount n,
    channel m carries signal ``n * sigma * branching(m)`` where
    ``sum_m branching(m) == 1``. Normalizing by the base peak instead (the
    pattern's max, as a "relative intensity" table conventionally reads)
    leaves a spurious ``sum(pattern) / max(pattern)`` factor in any amplitude
    recovered from more than one channel, which differs species to species.
    """
    if isinstance(pattern, MassSpectrum):
        raw = {int(m): float(v) for m, v in zip(pattern.mz, pattern.intensity) if v > 0}
    else:
        raw = {int(m): float(v) for m, v in pattern.items() if float(v) > 0}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {m: v / total for m, v in raw.items()}


def _species_amplitude(
    mz_to_signal: dict[int, float],
    pattern: dict[int, float],
    method: str,
    on_missing: str = "raise",
    species_label: str = "species",
) -> float:
    if not pattern:
        msg = (f"molar_ratio: no usable marker channel left for the {species_label} "
               f"after contamination exclusion.")
        if on_missing == "raise":
            raise ValueError(msg)
        logger.warning(msg)
        return np.nan
    if method == "single":
        m0 = max(pattern, key=lambda m: pattern[m])
        val = mz_to_signal.get(m0, np.nan)
        return val / pattern[m0] if (np.isfinite(val) and val > 0) else np.nan
    vals = []
    weights = []
    for m, f_m in pattern.items():
        val = mz_to_signal.get(m, np.nan)
        if not np.isfinite(val) or val <= 0 or f_m <= 0:
            msg = (
                f"molar_ratio: {species_label} marker channel m/z {m} is "
                f"missing or non-positive in the observed spectrum."
            )
            if on_missing == "raise":
                raise ValueError(msg)
            logger.warning(msg)
            return np.nan
        vals.append(val)
        weights.append(f_m)
    if method == "sum":
        return float(sum(vals) / sum(weights))
    if method == "geo":
        ratios = [v / f for v, f in zip(vals, weights)]
        return float(np.exp(np.mean(np.log(ratios))))
    raise ValueError(f"Unknown amplitude method {method!r}; use 'geo', 'sum', or 'single'.")


def molar_ratio(
    spectrum: np.ndarray | MassSpectrum | dict[int, float],
    mz: np.ndarray | None,
    pattern_num: dict[int, float] | MassSpectrum,
    pattern_den: dict[int, float] | MassSpectrum,
    sigma_num: float | str,
    sigma_den: float | str,
    method: str = "geo",
    other_patterns: list[dict[int, float] | MassSpectrum] | None = None,
    contamination_rel: float = 0.05,
    on_missing: str = "raise",
) -> float:
    """
    Compute the cross-section-weighted molar ratio ``(A_num / sigma_num) / (A_den / sigma_den)``
    between two co-products from an observed RGA spectrum.

    Any channel in ``pattern_num`` (or ``pattern_den``) where the other
    co-product or any species in ``other_patterns`` has a branching ratio
    ``>= contamination_rel`` is automatically excluded before computing the
    pattern-normalized species amplitude (see :func:`_normalize_pattern` for
    why the normalization is by pattern sum, not base peak).

    With ``method="sum"`` or ``"geo"``, every remaining marker channel of a
    species must be present and positive in the observed spectrum (``method=
    "single"`` only needs its own strongest channel, so this does not apply
    to it). When one is missing or non-positive, ``on_missing="raise"``
    (default) raises a ``ValueError`` naming the channel and species;
    ``on_missing="nan"`` instead logs a warning naming the same thing and
    returns NaN. The same applies when contamination exclusion leaves a
    species with no usable channel.

    Parameters
    ----------
    spectrum          : 1-D signal array aligned with *mz*, a :class:`MassSpectrum`,
                        or a ``{mz: intensity}`` mapping
    mz                : 1-D integer m/z array (optional when *spectrum* is a
                        :class:`MassSpectrum` or dict)
    pattern_num       : marker channel pattern ``{mz: intensity}`` or
                        :class:`MassSpectrum` for the numerator species
    pattern_den       : marker channel pattern ``{mz: intensity}`` or
                        :class:`MassSpectrum` for the denominator species
    sigma_num         : total 70 eV ionization cross section (A^2) or species
                        identifier for :func:`get_cross_section`
    sigma_den         : total 70 eV ionization cross section (A^2) or species
                        identifier for :func:`get_cross_section`
    method            : ``"geo"`` (geometric mean of pattern-normalized channels),
                        ``"sum"`` (sum of channels divided by sum of pattern weights),
                        or ``"single"`` (strongest marker channel only)
    other_patterns    : optional additional interfering species patterns
    contamination_rel : branching-ratio threshold above which a channel is
                        considered contaminated by another species (default 0.05)
    on_missing        : ``"raise"`` (default) or ``"nan"``; see above
    """
    if method not in ("geo", "sum", "single"):
        raise ValueError(f"Unknown amplitude method {method!r}; use 'geo', 'sum', or 'single'.")
    if on_missing not in ("raise", "nan"):
        raise ValueError(f"on_missing must be 'raise' or 'nan', got {on_missing!r}.")
    if isinstance(spectrum, MassSpectrum):
        mz_to_sig = {int(m): float(v) for m, v in zip(spectrum.mz, spectrum.intensity)}
    elif isinstance(spectrum, dict):
        mz_to_sig = {int(m): float(v) for m, v in spectrum.items()}
    else:
        if mz is None:
            raise ValueError("mz array is required when spectrum is a 1-D numpy array.")
        mz_to_sig = {int(m): float(v) for m, v in zip(mz, spectrum)}

    p_num = _normalize_pattern(pattern_num)
    p_den = _normalize_pattern(pattern_den)
    extra = [_normalize_pattern(p) for p in (other_patterns or [])]

    clean_num = {
        m: f for m, f in p_num.items()
        if all(other.get(m, 0.0) < contamination_rel for other in [p_den, *extra])
    }
    clean_den = {
        m: f for m, f in p_den.items()
        if all(other.get(m, 0.0) < contamination_rel for other in [p_num, *extra])
    }

    s_num = get_cross_section(sigma_num) if isinstance(sigma_num, str) else float(sigma_num)
    s_den = get_cross_section(sigma_den) if isinstance(sigma_den, str) else float(sigma_den)
    if s_num <= 0 or s_den <= 0:
        raise ValueError("Cross sections must be positive.")

    a_num = _species_amplitude(mz_to_sig, clean_num, method=method,
                               on_missing=on_missing, species_label="numerator")
    a_den = _species_amplitude(mz_to_sig, clean_den, method=method,
                               on_missing=on_missing, species_label="denominator")
    if not (np.isfinite(a_num) and np.isfinite(a_den) and a_den > 0):
        return np.nan
    return float((a_num / s_num) / (a_den / s_den))
