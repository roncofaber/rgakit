"""
nomenclature.py
---------------
Chemical name and metadata resolution via PubChem (pubchempy).

Public API
----------
normalize_nist_name(name)
    Convert a NIST-style inverted name to a readable form.

get_compound_info(identifier, namespace="cas")
    Fetch name, SMILES, formula, MW, CAS, InChIKey from PubChem.
    Results are cached on disk keyed by CAS.

resolve_name(cas, nist_name)
    Convenience wrapper for display names:
    hardcoded common name → PubChem IUPAC → normalised NIST name.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded common names
# (override PubChem IUPAC for universally recognised trivial names)
# ---------------------------------------------------------------------------

_COMMON_NAMES: dict[str, str] = {
    "7732-18-5":  "Water",
    "7664-41-7":  "Ammonia",
    "74-82-8":    "Methane",
    "7727-37-9":  "Nitrogen",
    "7782-44-7":  "Oxygen",
    "1333-74-0":  "Hydrogen",
    "630-08-0":   "Carbon monoxide",
    "124-38-9":   "Carbon dioxide",
    "10102-43-9": "Nitric oxide",
    "10544-72-6": "Nitrogen dioxide",
    "7783-06-4":  "Hydrogen sulfide",
    "67-56-1":    "Methanol",
    "64-17-5":    "Ethanol",
    "67-64-1":    "Acetone",
    "75-07-0":    "Acetaldehyde",
    "50-00-0":    "Formaldehyde",
}


# ---------------------------------------------------------------------------
# NIST inverted-name normalisation
# ---------------------------------------------------------------------------

_NIST_INVERTED_RE = re.compile(r"^(.+?),\s*(.+?)-\s*$")


def normalize_nist_name(name: str) -> str:
    """
    Convert a NIST-style inverted name to a more readable form.

    Examples
    --------
    >>> normalize_nist_name("Propane, 2-iodo-")
    '2-Iodopropane'
    >>> normalize_nist_name("Carbon dioxide")
    'Carbon dioxide'
    """
    m = _NIST_INVERTED_RE.match(name)
    if not m:
        return name
    parent, substituent = m.group(1).strip(), m.group(2).strip()
    sep      = "-" if parent[0].isdigit() else ""
    combined = f"{substituent}{sep}{parent}".lower()
    for i, c in enumerate(combined):
        if c.isalpha():
            return combined[:i] + c.upper() + combined[i + 1:]
    return combined


# ---------------------------------------------------------------------------
# PubChem compound info (with disk + in-process cache)
# ---------------------------------------------------------------------------

_CAS_RE     = re.compile(r'^\d+-\d+-\d+$')
_INFO_CACHE: dict | None = None   # populated lazily


def _cache_path() -> Path:
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("rgakit")) / "pubchem_compounds.json"
    except ImportError:
        return Path.home() / ".cache" / "rgakit" / "pubchem_compounds.json"


def _get_cache() -> dict:
    global _INFO_CACHE
    if _INFO_CACHE is None:
        p = _cache_path()
        try:
            _INFO_CACHE = json.loads(p.read_text()) if p.exists() else {}
            if _INFO_CACHE:
                logger.debug("Loaded PubChem cache: %d entries from %s", len(_INFO_CACHE), p)
        except Exception as e:
            logger.warning("Could not read PubChem cache at %s (%s) - starting empty.", p, e)
            _INFO_CACHE = {}
    return _INFO_CACHE


def _persist(cas: str, data: dict) -> None:
    cache = _get_cache()
    if cache.get(cas) == data:
        return
    cache[cas] = data
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2))
        logger.debug("PubChem cache updated: %s written to %s", cas, p)
    except Exception as e:
        logger.warning("Could not write PubChem cache to %s: %s", p, e)


def _cas_from_synonyms(synonyms: list[str]) -> str | None:
    return next((s for s in synonyms if _CAS_RE.match(s)), None)


def _cap(name: str) -> str:
    """Capitalize the first alphabetic character of a compound name."""
    for i, c in enumerate(name):
        if c.isalpha():
            return name[:i] + c.upper() + name[i + 1:]
    return name


# pubchempy namespace aliases (CAS numbers are looked up as names in PubChem)
_PCP_NAMESPACE = {
    "cas":      "name",
    "name":     "name",
    "smiles":   "smiles",
    "inchikey": "inchikey",
    "inchi":    "inchi",
}


def get_compound_info(
    identifier: str,
    namespace:  str = "cas",
) -> dict | None:
    """
    Fetch compound metadata from PubChem and cache the result.

    Parameters
    ----------
    identifier : CAS number, compound name, SMILES, or InChIKey
    namespace  : ``"cas"`` | ``"name"`` | ``"smiles"`` | ``"inchikey"``

    Returns
    -------
    dict with keys: ``name``, ``iupac``, ``smiles``, ``formula``,
    ``mw``, ``cas``, ``inchikey``  — or ``None`` if not found.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        raise ImportError("pubchempy is required: pip install pubchempy")

    # CAS cache hit (keyed by CAS string)
    if namespace == "cas":
        cache = _get_cache()
        if identifier in cache:
            logger.debug("PubChem cache hit: %s", identifier)
            return cache[identifier]

    pcp_ns    = _PCP_NAMESPACE.get(namespace, "name")
    logger.debug("PubChem lookup: %s=%r", namespace, identifier)
    compounds = pcp.get_compounds(identifier, pcp_ns)
    if not compounds:
        logger.debug("PubChem: no results for %s=%r", namespace, identifier)
        return None

    c = compounds[0]
    # Suppress pubchempy's own 404 log — no synonyms is expected for
    # unusual fragments and does not indicate a real error.
    _pcp_log = logging.getLogger("pubchempy")
    _old_lvl = _pcp_log.level
    _pcp_log.setLevel(logging.CRITICAL)
    try:
        cas = _cas_from_synonyms(c.synonyms)
    except Exception:
        cas = None
    finally:
        _pcp_log.setLevel(_old_lvl)

    # Cache hit via resolved CAS (e.g. name/smiles lookup whose CAS was cached)
    if cas and namespace != "cas":
        cache = _get_cache()
        if cas in cache:
            logger.debug("PubChem cache hit (via resolved CAS %s)", cas)
            return cache[cas]

    data = {
        "name":     _cap(_COMMON_NAMES.get(cas or "", "") or c.iupac_name or ""),
        "iupac":    c.iupac_name or "",
        "smiles":   c.smiles or "",
        "formula":  c.molecular_formula or "",
        "mw":       str(c.molecular_weight or ""),
        "cas":      cas or "",
        "inchikey": c.inchikey or "",
    }

    if cas:
        logger.debug("PubChem resolved %r: CAS=%s, name=%r", identifier, cas, data["name"])
        _persist(cas, data)
    else:
        logger.debug("PubChem resolved %r: name=%r (no CAS)", identifier, data["name"])
    return data


# ---------------------------------------------------------------------------
# Display-name resolver (used when loading spectra)
# ---------------------------------------------------------------------------

def resolve_name(cas: str = "", nist_name: str = "", **_) -> str:
    """
    Return the best display name for a compound.

    Priority: hardcoded common name → PubChem IUPAC → normalised NIST name.
    """
    if cas and cas in _COMMON_NAMES:
        return _COMMON_NAMES[cas]
    if cas:
        info = get_compound_info(cas, "cas")
        if info and info.get("name"):
            return _cap(info["name"])
    return normalize_nist_name(nist_name)
