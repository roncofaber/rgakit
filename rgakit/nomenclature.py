"""
nomenclature.py
---------------
Helpers for resolving and normalising chemical compound names.

Public API
----------
normalize_nist_name(name)
    Convert a NIST-style inverted name ("Propane, 2-iodo-") to a readable
    form ("2-Iodopropane").

cactus_preferred_name(cas, timeout)
    Query the NCI Cactus IUPAC-name endpoint for a given CAS number.
    Results are cached on disk so the network is queried at most once per CAS.

resolve_name(cas, nist_name, timeout)
    Convenience wrapper: try Cactus first, fall back to normalize_nist_name.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardcoded common names
# (override Cactus/IUPAC for universally recognised trivial names)
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

    NIST (and CAS) store names as ``"Parent, substituent-"`` so that related
    compounds sort together alphabetically.  This function reverses the
    inversion and applies title-casing to the first alphabetic character.

    Examples
    --------
    >>> normalize_nist_name("Propane, 2-iodo-")
    '2-Iodopropane'
    >>> normalize_nist_name("1-Pentene, 2,4,4-trimethyl-")
    '2,4,4-Trimethyl-1-pentene'
    >>> normalize_nist_name("Benzene, methyl-")
    'Methylbenzene'
    >>> normalize_nist_name("Carbon dioxide")   # unchanged — no comma pattern
    'Carbon dioxide'
    """
    m = _NIST_INVERTED_RE.match(name)
    if not m:
        return name
    parent, substituent = m.group(1).strip(), m.group(2).strip()
    # Use a hyphen only when the parent starts with a locant digit
    # (e.g. "1-Pentene") so substituent and parent don't run together.
    sep      = "-" if parent[0].isdigit() else ""
    combined = f"{substituent}{sep}{parent}".lower()
    for i, c in enumerate(combined):
        if c.isalpha():
            return combined[:i] + c.upper() + combined[i + 1:]
    return combined


# ---------------------------------------------------------------------------
# NCI Cactus IUPAC-name resolution (with disk + in-process cache)
# ---------------------------------------------------------------------------

_CACTUS_BASE  = "https://cactus.nci.nih.gov/chemical/structure/{}/iupac_name"
_NAME_CACHE: dict | None = None   # populated lazily from disk on first access


def _name_cache_path() -> Path:
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("rgakit")) / "cactus_names.json"
    except ImportError:
        return Path.home() / ".cache" / "rgakit" / "cactus_names.json"


def _get_name_cache() -> dict:
    global _NAME_CACHE
    if _NAME_CACHE is None:
        p = _name_cache_path()
        try:
            _NAME_CACHE = json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            _NAME_CACHE = {}
    return _NAME_CACHE


def _save_name_cache(cas: str, name: str) -> None:
    """Persist a single CAS → name entry to the cache file."""
    cache = _get_name_cache()
    if cache.get(cas) == name:
        return                          # already up-to-date, skip disk write
    cache[cas] = name
    p = _name_cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass                            # cache is best-effort


def _title_case_iupac(text: str) -> str:
    """Capitalise the first alphabetic character; lowercase the rest."""
    for i, c in enumerate(text):
        if c.isalpha():
            return text[:i] + c.upper() + text[i + 1:].lower()
    return text


def cactus_preferred_name(cas: str, timeout: int = 6) -> str | None:
    """
    Return a clean preferred name for *cas* from NCI Cactus.

    Uses the ``/iupac_name`` endpoint of the NCI Chemical Identifier Resolver
    (https://cactus.nci.nih.gov/).  Results are cached in
    ``~/.cache/rgakit/cactus_names.json`` so each CAS number is queried at
    most once per machine.

    Parameters
    ----------
    cas     : CAS registry number (e.g. ``"75-30-9"``)
    timeout : HTTP timeout in seconds

    Returns
    -------
    Cleaned IUPAC name string, or ``None`` on network failure, HTTP error, or
    when the returned name is garbled (e.g. LaTeX-notation like
    ``"$l^{1}-azane"`` for ammonia).
    """
    if not cas or cas == "N/A":
        return None

    # In-process + disk cache check
    cache = _get_name_cache()
    if cas in cache:
        return cache[cas] or None   # empty string marks a known failure

    url = _CACTUS_BASE.format(urllib.parse.quote(cas, safe=""))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode().strip().splitlines()[0].strip()
    except Exception:
        _save_name_cache(cas, "")
        return None

    # Reject garbled IUPAC notation (e.g. "$l^{1}-azane", "azane\nazane")
    if not text or any(c in text for c in ("$", "^", "{", "}")):
        _save_name_cache(cas, "")
        return None

    result = _title_case_iupac(text)
    _save_name_cache(cas, result)
    return result


def resolve_name(cas: str = "", nist_name: str = "", timeout: int = 6) -> str:
    """
    Return the best available name for a compound.

    Priority: hardcoded common name → Cactus IUPAC → normalised NIST name.

    Parameters
    ----------
    cas       : CAS registry number (optional)
    nist_name : raw NIST title string (used as fallback)
    timeout   : Cactus HTTP timeout in seconds
    """
    if cas and cas in _COMMON_NAMES:
        return _COMMON_NAMES[cas]
    return (cactus_preferred_name(cas, timeout=timeout)
            or normalize_nist_name(nist_name))
