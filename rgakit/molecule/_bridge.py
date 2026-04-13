"""
_bridge.py
Connects the molecule/fragmentation layer to the rgakit spectrum/library layer.

Provides:
  fragment_to_spectrum(frag)         returns a MassSpectrum
  compound_to_library(compound, ...) returns a SpectraLibrary
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _enrich_spectrum(spec, frag):
    """
    Return a copy of *spec* with the fragment's own metadata filled in where
    the database left fields blank, and with a human-readable name when the
    source used the raw SMILES string as the name (in-silico DB convention).

    Fields filled in from the fragment if missing in spec.metadata:
      smiles, formula, inchikey, mw (monoisotopic mass)

    Name rule: if the database set the name to the fragment's SMILES string,
    replace it with the molecular formula (always available, no network call).
    """
    from rgakit.spectrum import MassSpectrum

    meta = dict(spec.metadata)
    for key, val in [
        ("smiles",   frag.smiles),
        ("formula",  frag.formula),
        ("inchikey", frag.inchikey),
        ("mw",       frag.monoisotopic_mass),
    ]:
        if not meta.get(key) and val is not None:
            meta[key] = val

    # If the DB returned the SMILES as the name, use the formula instead.
    name = spec.name
    if frag.smiles and name == frag.smiles:
        name = frag.formula or frag.smiles

    if name == spec.name and meta == spec.metadata:
        return spec  # nothing changed — avoid unnecessary allocation
    return MassSpectrum(spec.mz, spec.intensity, name=name, metadata=meta)


def fragment_to_spectrum(
    frag,
    nist_lookup:      bool  = False,
    formula_fallback: bool  = False,
    db                      = None,
    mb_db                   = None,
):
    """
    Convert a Fragment to a MassSpectrum.

    Lookup order (each step only runs if the previous found nothing):
      1. Pre-loaded spectrum on the fragment (set by load_ms_spectra) — instant.
      2. In-silico database (if *db* is provided) — fast, offline, by InChIKey.
      3. MassBank REST API (if *mb_db* is provided) — online, by InChIKey.
      4. NIST WebBook by InChIKey (only if *nist_lookup=True*; requires network).
      5. NIST WebBook by formula (only if *formula_fallback=True*; ambiguous).

    Returns None when all configured sources miss.  No theoretical fallback is
    generated — fragments without a real spectrum are simply skipped.

    Parameters
    ----------
    frag             : Compound object (in fragment mode)
    nist_lookup      : query NIST WebBook as a fallback (default False)
    formula_fallback : also try formula-based NIST search when InChIKey lookup
                       fails (default False, ambiguous — first isomer only)
    db               : optional InSilicoDatabase; queried first (fast, offline)
    mb_db            : optional MassBankDatabase; queried after in-silico
    """
    from rgakit.molecule._utils import smiles_to_inchikey

    # Step 1: return a pre-loaded spectrum immediately.
    if getattr(frag, "spectrum", None) is not None:
        return frag.spectrum

    # Charged fragments are not in EI-MS databases (neutral species only).
    if frag.charge != 0:
        logger.debug("Skipping DB lookup for charged fragment: %s (charge=%+d)",
                     frag.formula, frag.charge)
        return None

    # Compute InChIKey once — used by all database sources.
    inchikey = smiles_to_inchikey(frag.smiles) if frag.smiles else None
    if inchikey is None and frag.smiles:
        logger.debug("Could not compute InChIKey for %s (%s)",
                     frag.formula, frag.smiles)

    # Step 2: in-silico database (local, no network).
    if db is not None and inchikey is not None:
        spec = db.get(inchikey=inchikey)
        if spec is not None:
            logger.info("In-silico hit: %s (%s)", frag.formula, frag.smiles)
            return _enrich_spectrum(spec, frag)

    # Step 3: MassBank REST API.
    if mb_db is not None and inchikey is not None:
        spec = mb_db.get(inchikey=inchikey)
        if spec is not None:
            logger.info("MassBank hit: %s (%s)", frag.formula, frag.smiles)
            return _enrich_spectrum(spec, frag)

    # Steps 4–5: NIST (explicit opt-in only).
    if nist_lookup:
        if inchikey is not None:
            try:
                from rgakit.spectrum import MassSpectrum
                spec = MassSpectrum.from_nist(inchikey=inchikey)
                logger.info("NIST hit: %s (%s)", frag.formula, frag.smiles)
                return _enrich_spectrum(spec, frag)
            except Exception as exc:
                logger.debug("NIST miss for %s: %s", frag.smiles, exc)

        if formula_fallback:
            try:
                from rgakit.spectrum import MassSpectrum
                spec = MassSpectrum.from_nist(formula=frag.formula)
                logger.warning(
                    "NIST hit (formula, ambiguous): %s gives %r; "
                    "verify this is the correct isomer.",
                    frag.formula, spec.name,
                )
                return _enrich_spectrum(spec, frag)
            except Exception as exc:
                logger.debug("NIST formula miss for %s: %s", frag.formula, exc)

        logger.info("NIST miss: %s", frag.formula)

    return None


def _nist_all_for_formula(formula: str) -> list:
    """
    Fetch every NIST EI-MS spectrum whose molecular formula matches *formula*.

    Unlike MassSpectrum.from_nist(formula=...), which returns only the first
    isomer that has MS data, this function iterates over all NIST hits and
    returns one MassSpectrum per compound (skipping those without MS data).

    Parameters
    ----------
    formula : Hill-notation molecular formula, e.g. "C3H6"

    Returns
    -------
    list of MassSpectrum objects (empty if no NIST hits or network failure)
    """
    try:
        import nistchempy as nist
    except ImportError:
        logger.warning("nistchempy not installed; formula_all requires it.")
        return []

    from rgakit.spectrum import MassSpectrum

    try:
        results = nist.run_search(formula, "formula")
    except Exception as exc:
        logger.debug("NIST formula search failed for %s: %s", formula, exc)
        return []

    if not results.compounds:
        return []

    spectra = []
    for compound in results.compounds:
        try:
            compound.get_ms_spectra()
            if not compound.ms_specs:
                continue
            spec = MassSpectrum.from_jdx(compound.ms_specs[0].jdx_text,
                                         name=compound.name)
            spectra.append(spec)
        except Exception as exc:
            logger.debug("Skipping %s (%s): %s", compound.name, formula, exc)

    logger.info(
        "NIST formula search %r: %d spectrum/spectra across %d hit(s).",
        formula, len(spectra), len(results.compounds),
    )
    return spectra


def compound_to_library(
    compound,
    nist_lookup:      bool = False,
    include_raw:      bool = False,
    formula_fallback: bool = False,
    formula_all:      bool = False,
    max_workers:      int  = 8,
    db                     = None,
    mb_db                  = None,
):
    """
    Build a SpectraLibrary from the stable fragments of a Compound.

    Each stable fragment is looked up via fragment_to_spectrum.  Fragments
    that return None (no hit in any configured source) are silently skipped.
    Duplicates (same InChIKey or name) are deduplicated.

    Parameters
    ----------
    compound         : Compound object (must have called do_fragmentation())
    nist_lookup      : query NIST as a fallback (default False)
    include_raw      : also include raw (radical) fragments (default False)
    formula_fallback : use formula-based NIST search on InChIKey miss
                       (default False, ambiguous)
    formula_all      : for NIST misses, add ALL NIST isomers sharing the same
                       molecular formula (default False)
    max_workers      : thread-pool size for concurrent lookups (default 8)
    db               : optional InSilicoDatabase; queried first (fast, offline)
    mb_db            : optional MassBankDatabase; queried after in-silico

    Returns
    -------
    SpectraLibrary
    """
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from rgakit.library import SpectraLibrary

    frags = list(compound.fragments)
    if include_raw:
        frags += compound.raw_fragments

    # Pass 1: per-fragment lookup (threaded).
    lookup = partial(
        fragment_to_spectrum,
        nist_lookup      = nist_lookup,
        formula_fallback = formula_fallback,
        db               = db,
        mb_db            = mb_db,
    )

    n_workers = min(max_workers, len(frags)) if frags else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        all_specs = list(pool.map(lookup, frags))

    spectra        = []
    seen_names     = set()
    seen_inchikeys = set()
    skipped        = 0
    missed_frags   = []

    for frag, spec in zip(frags, all_specs):
        if spec is None:
            skipped += 1
            if formula_all and frag.charge == 0:
                missed_frags.append(frag)
            continue
        # Deduplicate by InChIKey first (source-agnostic), then by name.
        ik = spec.inchikey
        if ik and ik in seen_inchikeys:
            continue
        if spec.name in seen_names:
            continue
        if ik:
            seen_inchikeys.add(ik)
        seen_names.add(spec.name)
        spectra.append(spec)

    # Pass 2: formula_all — search every unique formula concurrently.
    if formula_all and missed_frags:
        unique_formulas = list({f.formula for f in missed_frags if f.formula})
        logger.info(
            "formula_all: searching %d unique formula(s) across %d missed fragment(s).",
            len(unique_formulas), len(missed_frags),
        )
        n_workers2 = min(max_workers, len(unique_formulas))
        with ThreadPoolExecutor(max_workers=n_workers2) as pool:
            formula_results = list(pool.map(_nist_all_for_formula, unique_formulas))

        for formula_spectra in formula_results:
            for spec in formula_spectra:
                ik = spec.inchikey
                if ik and ik in seen_inchikeys:
                    continue
                if spec.name in seen_names:
                    continue
                if ik:
                    seen_inchikeys.add(ik)
                seen_names.add(spec.name)
                spectra.append(spec)
                skipped -= 1

    label = compound.name or compound.smiles
    logger.info(
        "Library for '%s': %d spectra, %d fragment(s) with no match.",
        label, len(spectra), max(skipped, 0),
    )

    if not spectra:
        raise ValueError(
            f"No spectra found for any fragment of {label!r}. "
            f"Pass nist_lookup=True, formula_fallback=True, or formula_all=True."
        )

    return SpectraLibrary(spectra)
