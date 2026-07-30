"""
utils.py
--------
Shared SQL helpers for bulk spectrum access across database backends.
"""

from __future__ import annotations

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Full column list matching local.py's _SELECT
_FULL_SELECT = """
    SELECT s.id, s.inchikey, s.source,
           c.smiles, c.exact_mass, c.cas,
           s.mzs, s.intensities,
           s.precursor_mz, s.precursor_type,
           s.ionization_mode, s.ms_level,
           s.instrument_type, s.instrument, s.collision_energy,
           s.splash, s.score
    FROM   spectra s
    LEFT JOIN compounds c ON s.inchikey = c.inchikey
"""

_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


def parse_elements(formula: str) -> set[str]:
    """Extract element symbols from a molecular formula string."""
    return set(_ELEMENT_RE.findall(formula))


def iter_raw(db, source: str | None = None) -> list[tuple]:
    """Load ``(rowid, mzs_blob, intensities_blob, exact_mass, formula)``
    for every spectrum matching *source*."""
    sql = (
        "SELECT s.rowid, s.mzs, s.intensities, c.exact_mass, c.molecular_formula "
        "FROM spectra s LEFT JOIN compounds c ON s.inchikey = c.inchikey"
    )
    params: list = []
    if source:
        sql += " WHERE s.source = ?"
        params.append(source)
    return db.conn.execute(sql, params).fetchall()


def fetch_by_rowids(db, rowids: list[int]) -> list:
    """Return full MassSpectrum objects for the given SQLite rowids."""
    if not rowids:
        return []
    placeholders = ",".join("?" * len(rowids))

    full_rows = db.conn.execute(
        _FULL_SELECT + f" WHERE s.rowid IN ({placeholders})", rowids
    ).fetchall()

    id_rows = db.conn.execute(
        f"SELECT rowid, id FROM spectra WHERE rowid IN ({placeholders})",
        rowids,
    ).fetchall()
    rowid_to_id = {r: sid for r, sid in id_rows}

    id_to_spec = {}
    for row in full_rows:
        spec = db._to_spectrum(row)
        if spec is not None:
            id_to_spec[row[0]] = spec

    spectra = []
    for rid in rowids:
        sid  = rowid_to_id.get(rid)
        spec = id_to_spec.get(sid)
        if spec is not None:
            spectra.append(spec)
    return spectra
