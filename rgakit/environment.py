"""
environment.py
---------------
Named sets of environment/background compounds (chamber gases, pump-oil
hydrocarbons, process solvents) as data, resolved through a spectral
database by SMILES rather than hard-coded in several places.

Each set lists SMILES and a human-readable name per compound (the name is
informational/for tests; the spectrum's actual name comes from whatever
database resolves it). A spectrum built from one of these sets always
carries ``metadata["environment"] = True`` and
``metadata["environment_set"]`` so callers (e.g.
:meth:`~rgakit.library.SpectraLibrary.restrict_elements`) can recognize it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ENVIRONMENT_SETS: dict[str, dict[str, object]] = {
    "chamber_gases": {
        "label": "Chamber background gases",
        "smiles": ["[HH]", "O", "[C-]#[O+]", "O=C=O", "N#N", "O=O", "[Ar]", "N", "C"],
        "names":  ["Hydrogen", "Water", "Carbon monoxide", "Carbon dioxide",
                   "Nitrogen", "Oxygen", "Argon", "Ammonia", "Methane"],
    },
    # m/z 39/41/43/55/57 from propene/propane/isobutane/1-butene/n-butane
    # cracking; n-hexane extends 43/57 coverage. See
    # tests/test_environment.py for the measured (NIST) per-channel
    # coverage this set gives.
    "pump_oil": {
        "label": "Pump-oil hydrocarbon cracking products",
        "smiles": ["C=CC", "CCC", "CC(C)C", "C=CCC", "CCCC", "CCCCCC"],
        "names":  ["Propene", "Propane", "Isobutane", "1-Butene", "Butane", "n-Hexane"],
    },
    "spin_solvent": {
        "label": "Spin-coating solvent",
        "smiles": ["CN(C)C=O"],
        "names":  ["N,N-Dimethylformamide"],
    },
    "cleaning_solvents": {
        "label": "Substrate-cleaning solvents",
        "smiles": ["CC(=O)C", "CC(C)O"],
        "names":  ["Acetone", "Isopropyl alcohol"],
    },
}


def build_environment_library(sets, database) -> "SpectraLibrary":
    """
    Build a :class:`~rgakit.library.SpectraLibrary` from named environment
    sets (see :data:`ENVIRONMENT_SETS`).

    *database* (e.g. a :class:`~rgakit.databases.NistDatabase`) is
    required: every compound is matched against it, same as the rest of a
    fit, and there is no offline fallback. A fabricated reference
    spectrum, even single-peak and clearly labeled, can end up in a fit;
    environment spectra come from the same real source as everything
    else. A compound with no entry in *database* raises a clear
    ``ValueError`` naming the compound and set.

    Every returned spectrum's metadata records ``environment=True``,
    ``environment_set`` (which named set it came from), and
    ``environment_source`` (the database's own ``source`` tag, e.g.
    ``"nist"``).
    """
    from .library import SpectraLibrary
    from .spectrum import MassSpectrum

    if database is None:
        raise ValueError("build_environment_library requires a database; there is no offline fallback.")

    specs = []
    for set_name in sets:
        if set_name not in ENVIRONMENT_SETS:
            raise ValueError(
                f"Unknown environment set {set_name!r}; known: {sorted(ENVIRONMENT_SETS)}"
            )
        for smi in ENVIRONMENT_SETS[set_name]["smiles"]:
            spec = database.get(smiles=smi)
            if spec is None:
                raise ValueError(
                    f"No database entry for {smi!r} in environment set {set_name!r}."
                )
            meta = dict(spec.metadata)
            meta["environment"] = True
            meta["environment_set"] = set_name
            meta["environment_source"] = (meta.get("source") or getattr(database, "_SOURCE", None)
                                          or "unknown")
            meta.setdefault("smiles", smi)
            specs.append(MassSpectrum(spec.mz, spec.intensity, name=spec.name, metadata=meta))

    return SpectraLibrary(specs)


def allowed_elements_from_ion_pair(
    smiles: str,
    *,
    extra_elements=None,
    include_environment=None,
) -> set[str]:
    """
    Derive the allowed element set for :meth:`~rgakit.library.SpectraLibrary.restrict_elements`
    from an ion-pair SMILES.

    Parameters
    ----------
    smiles              : SMILES of the ion pair (e.g. beta-alaninium
                          iodide), or any molecule/fragment whose elements
                          should be the restriction's basis
    extra_elements      : additional element symbols to union in (e.g. the
                          metal of a metal-halide matrix, which typically
                          arrives as a separate SMILES and is not part of
                          *smiles* itself)
    include_environment : names of :data:`ENVIRONMENT_SETS` entries whose
                          elements should also be allowed (e.g.
                          ``["chamber_gases", "pump_oil"]``)
    """
    from .chemutils import elements_from_formula, formula_from_smiles

    formula = formula_from_smiles(smiles)
    if formula is None:
        raise ValueError(f"Cannot parse ion-pair SMILES {smiles!r}.")
    elements = set(elements_from_formula(formula))

    if extra_elements:
        elements |= set(extra_elements)

    for set_name in include_environment or ():
        if set_name not in ENVIRONMENT_SETS:
            raise ValueError(
                f"Unknown environment set {set_name!r}; known: {sorted(ENVIRONMENT_SETS)}"
            )
        for smi in ENVIRONMENT_SETS[set_name]["smiles"]:
            f = formula_from_smiles(smi)
            if f:
                elements |= elements_from_formula(f)

    return elements
