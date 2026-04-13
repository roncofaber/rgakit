"""
nist.py
Interface to the local NIST EI-MS SQLite spectral library.

Uses the shared spectral library schema (schema.sql).  All query logic
is inherited from :class:`~rgakit.databases.local.LocalSpectralDatabase`.

Usage
-----
    from rgakit import NistDatabase

    db = NistDatabase("/path/to/spectral_libraries/nist_library.db")

    spec    = db.get(smiles="CCO")
    spec    = db.get(cas="64-17-5")
    spec    = db.get(name="ethanol")
    spectra = db.search(name="hexane")
    lib     = db.to_library(name="glucose")
"""

from __future__ import annotations

from .local import LocalSpectralDatabase


class NistDatabase(LocalSpectralDatabase):
    """
    Read-only interface to the local NIST EI-MS SQLite spectral library.

    Parameters
    ----------
    db_path         : path to ``nist_library.db``
    ionization_mode : default ionization mode filter (``None`` = no filter)
    ms_level        : default MS level filter (``None`` = no filter)
    instrument_type : default instrument type substring filter (``None`` = no filter)
    """

    _SOURCE = "nist"
