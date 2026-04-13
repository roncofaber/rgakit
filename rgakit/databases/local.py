"""
local.py
Base class for all local SQLite spectral libraries (in_silico, MoNA, NIST).

All three share the same schema (schema.sql) so the common logic lives here:
connection management, spectrum decoding, get/search/to_library.

Subclasses set ``_SOURCE`` to filter rows automatically and can extend
``search`` with extra capabilities (e.g. HNSW vector search for in_silico).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Columns returned by all queries (17 total)
_SELECT = """
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


class LocalSpectralDatabase:
    """
    Read-only interface to a local SQLite spectral library.

    Parameters
    ----------
    db_path         : path to the ``.db`` file
    source          : restrict queries to this source tag (e.g. ``'in_silico'``,
                      ``'mona'``, ``'nist'``).  ``None`` = no restriction.
    ionization_mode : default filter for :meth:`search`
    ms_level        : default filter for :meth:`search`
    instrument_type : default substring filter for :meth:`search`
    """

    _SOURCE: str | None = None   # set in subclasses

    def __init__(
        self,
        db_path:         str | Path,
        source:          str | None = None,
        ionization_mode: str | None = None,
        ms_level:        str | None = None,
        instrument_type: str | None = None,
    ):
        self._db_path         = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")

        self._source          = source or self._SOURCE
        self._ionization_mode = ionization_mode
        self._ms_level        = ms_level
        self._instrument_type = instrument_type
        self._local           = threading.local()

    # ------------------------------------------------------------------
    # Connection management (thread-local, safe for ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    def close(self) -> None:
        """Close the SQLite connection for the current thread."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        _ = self.conn
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Row → MassSpectrum
    # ------------------------------------------------------------------

    def _name_for(self, inchikey: str | None, fallback: str) -> str:
        """Look up the first non-computed name from compound_names."""
        if not inchikey:
            return fallback
        row = self.conn.execute(
            "SELECT name FROM compound_names "
            "WHERE inchikey = ? AND computed = 0 LIMIT 1",
            (inchikey,),
        ).fetchone()
        return row[0] if row else fallback

    def _to_spectrum(self, row: tuple, label: str | None = None):
        """Convert a DB row to a MassSpectrum."""
        from rgakit.spectrum import MassSpectrum

        (spec_id, inchikey, source,
         smiles, exact_mass, cas,
         mzs_blob, int_blob,
         precursor_mz, precursor_type, ionization_mode, ms_level,
         instrument_type, instrument, collision_energy,
         splash, score) = row

        mz        = np.frombuffer(mzs_blob, dtype=np.float64)
        intensity = np.frombuffer(int_blob,  dtype=np.float64)

        if intensity.size == 0 or intensity.max() == 0:
            return None

        name = label or self._name_for(inchikey, fallback=smiles or spec_id)

        return MassSpectrum(
            mz        = mz,
            intensity = intensity,
            name      = name,
            metadata  = {
                "source":          source,
                "smiles":          smiles,
                "mw":              exact_mass,
                "cas":             cas,
                "inchikey":        inchikey,
                "precursor_mz":    precursor_mz,
                "precursor_type":  precursor_type,
                "ionization_mode": ionization_mode,
                "ms_level":        ms_level,
                "instrument_type": instrument_type,
                "instrument":      instrument,
                "collision_energy":collision_energy,
                "splash":          splash,
                "score":           score,
                "db_id":           spec_id,
            },
        )

    # ------------------------------------------------------------------
    # SQL query builder
    # ------------------------------------------------------------------

    def _build_query(
        self,
        extra_where:     list[str],
        params:          list,
        ionization_mode: str | None = None,
        ms_level:        str | None = None,
        instrument_type: str | None = None,
    ) -> tuple[str, list]:
        """Combine base SELECT with WHERE clauses and instance-level defaults."""
        clauses = []
        if self._source:
            clauses.append("s.source = ?")
            params = [self._source] + list(params)

        clauses.extend(extra_where)

        eff_ion = ionization_mode if ionization_mode is not None else self._ionization_mode
        eff_lvl = ms_level        if ms_level        is not None else self._ms_level
        eff_ins = instrument_type if instrument_type is not None else self._instrument_type

        if eff_ion:
            clauses.append("s.ionization_mode = ?")
            params.append(eff_ion)
        if eff_lvl:
            clauses.append("s.ms_level = ?")
            params.append(eff_lvl)
        if eff_ins:
            clauses.append("s.instrument_type LIKE ?")
            params.append(f"%{eff_ins}%")

        sql = _SELECT
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return sql, params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        smiles:   str | None = None,
        inchikey: str | None = None,
        name:     str | None = None,
        cas:      str | None = None,
    ):
        """
        Retrieve the first matching MassSpectrum.

        Provide exactly one of *smiles*, *inchikey*, *name*, or *cas*.
        Returns a :class:`~rgakit.spectrum.MassSpectrum` or ``None``.
        """
        results = self.search(smiles=smiles, inchikey=inchikey, name=name, cas=cas)
        return results[0] if results else None

    def search(
        self,
        smiles:          str | None = None,
        inchikey:        str | None = None,
        name:            str | None = None,
        cas:             str | None = None,
        ionization_mode: str | None = None,
        ms_level:        str | None = None,
        instrument_type: str | None = None,
    ) -> list:
        """
        Query the database and return all matching spectra.

        Structural filters (*smiles*, *inchikey*, *name*, *cas*) and metadata
        filters (*ionization_mode*, *ms_level*, *instrument_type*) can be
        freely combined.

        Returns a list of :class:`~rgakit.spectrum.MassSpectrum` objects.
        """
        given = {k: v for k, v in
                 [("smiles", smiles), ("inchikey", inchikey),
                  ("name", name), ("cas", cas)]
                 if v is not None}
        if len(given) > 1:
            raise ValueError(
                f"Provide at most one structural filter. Got: {list(given)}"
            )

        extra_where: list[str] = []
        params:      list      = []
        label = None

        if "smiles" in given:
            from rgakit.molecule._utils import smiles_to_inchikey
            ik = smiles_to_inchikey(given["smiles"])
            if ik is None:
                logger.debug("%s: could not compute InChIKey for %r",
                             type(self).__name__, given["smiles"])
                return []
            extra_where.append("s.inchikey = ?")
            params.append(ik)

        elif "inchikey" in given:
            ik = given["inchikey"]
            extra_where.append("s.inchikey = ?")
            params.append(ik)
            # Also try connectivity prefix if exact match returns nothing
            rows = self.conn.execute(
                *self._build_query(extra_where, params,
                                   ionization_mode, ms_level, instrument_type)
            ).fetchall()
            if not rows:
                prefix = ik[:14] + "-"
                extra_where = ["s.inchikey LIKE ?"]
                params      = [prefix + "%"]

        elif "name" in given:
            extra_where.append(
                "s.inchikey IN "
                "(SELECT inchikey FROM compound_names WHERE name LIKE ?)"
            )
            params.append(f"%{given['name']}%")
            label = given["name"]

        elif "cas" in given:
            extra_where.append("c.cas = ?")
            params.append(given["cas"])

        sql, params = self._build_query(
            extra_where, params, ionization_mode, ms_level, instrument_type
        )
        rows = self.conn.execute(sql, params).fetchall()

        spectra = []
        for row in rows:
            spec = self._to_spectrum(row, label=label)
            if spec is not None:
                spectra.append(spec)

        logger.info("%s.search: %d spectra.", type(self).__name__, len(spectra))
        return spectra

    def to_library(self, **search_kwargs):
        """
        Build a :class:`~rgakit.library.SpectraLibrary` from a query.

        Keyword arguments are forwarded to :meth:`search`.
        Raises ``ValueError`` if no spectra are found.
        """
        from rgakit.library import SpectraLibrary

        spectra = self.search(**search_kwargs)
        if not spectra:
            raise ValueError(
                f"No spectra found for query: {search_kwargs}"
            )
        return SpectraLibrary(spectra)

    def __len__(self) -> int:
        if self._source:
            return self.conn.execute(
                "SELECT COUNT(*) FROM spectra WHERE source = ?", (self._source,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._db_path)!r})"
