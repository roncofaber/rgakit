"""
mona.py
REST API client for MassBank of North America (https://mona.fiehnlab.ucdavis.edu).

Provides MonaDatabase for querying mass spectra by SMILES, InChIKey, or
compound name, and for building SpectraLibrary objects from MoNA queries.

Usage:

    from rgakit import MonaDatabase

    mona = MonaDatabase()                             # public MoNA API

    spec = mona.get(smiles="CCO")                     # ethanol by SMILES
    spec = mona.get(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    spec = mona.get(name="ethanol")

    lib = mona.to_library(name="glucose")             # all matches as SpectraLibrary

By default only EI records are returned (ms_level="MS1" and instrument_type
contains "EI").  Change the defaults via constructor parameters.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request

import numpy as np

from .local import LocalSpectralDatabase

logger = logging.getLogger(__name__)

_BASE_URL = "https://mona.fiehnlab.ucdavis.edu/rest"


def _meta_value(metadata: list, name: str) -> str | None:
    """Return the value of the first metadata entry whose name matches (case-insensitive)."""
    key = name.lower()
    for entry in metadata or []:
        if (entry.get("name") or "").lower() == key:
            return entry.get("value")
    return None


class MonaDatabase:
    """
    Read-only REST API client for MassBank of North America (MoNA).

    Parameters
    ----------
    base_url         : API base URL.  Defaults to
                       ``https://mona.fiehnlab.ucdavis.edu/rest``.
    instrument_type  : Instrument type substring filter applied client-side
                       (default ``"EI"``).  Matches any record whose
                       ``instrument type`` metadata entry contains this string
                       (case-insensitive).  Use ``None`` to skip filtering.
    ms_level         : MS level filter applied client-side (default ``None`` —
                       no filter).  Set to ``"MS1"`` to restrict to MS1 records.
                       EI spectra often omit this field, so filtering by it can
                       silently drop valid hits.
    ion_mode         : Ion mode filter, e.g. ``"P"`` (positive) or ``"N"``
                       (negative).  Use ``None`` (default) to skip filtering.
    timeout          : HTTP request timeout in seconds (default 15).
    """

    def __init__(
        self,
        base_url:        str | None = None,
        instrument_type: str | None = "EI",
        ms_level:        str | None = None,
        ion_mode:        str | None = None,
        timeout:         int        = 15,
    ):
        self._base_url        = (base_url or _BASE_URL).rstrip("/")
        self._instrument_type = instrument_type
        self._ms_level        = ms_level
        self._ion_mode        = ion_mode
        self._timeout         = timeout

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(self, path: str, params: dict) -> list | dict | None:
        """GET {base_url}/{path}?{params}, return parsed JSON or None."""
        import json

        clean = {k: v for k, v in params.items() if v is not None}
        url   = f"{self._base_url}/{path}"
        if clean:
            url += "?" + urllib.parse.urlencode(clean)

        logger.debug("MoNA GET %s", url)
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("MoNA request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_spectrum_string(self, spectrum_str: str):
        """
        Parse MoNA spectrum string format into mz and intensity arrays.

        MoNA encodes peaks as a space-separated list of ``mz:intensity`` pairs,
        e.g. ``"50:100 51:200 52:150"``.
        """
        mz_list  = []
        int_list = []
        for token in spectrum_str.split():
            try:
                mz_str, int_str = token.split(":")
                mz_list.append(float(mz_str))
                int_list.append(float(int_str))
            except (ValueError, AttributeError):
                continue
        if not mz_list:
            return None, None
        return np.array(mz_list, dtype=float), np.array(int_list, dtype=float)

    def _to_spectrum(self, record: dict, label: str | None = None):
        """Convert a MoNA API record dict to a MassSpectrum."""
        from rgakit.spectrum import MassSpectrum

        try:
            spectrum_str = record.get("spectrum")
            if not spectrum_str:
                return None

            mz, intensity = self._parse_spectrum_string(spectrum_str)
            if mz is None or intensity.max() == 0:
                return None

            # Compound info — MoNA returns a list of compound objects
            compounds = record.get("compound") or []
            compound  = compounds[0] if compounds else {}

            comp_names = [n.get("name") for n in (compound.get("names") or []) if n.get("name")]
            name       = label or (comp_names[0] if comp_names else record.get("id", "unknown"))

            # InChIKey is a direct field on the compound object;
            # SMILES and formula live in compound metaData.
            inchikey  = compound.get("inchiKey")
            comp_meta = compound.get("metaData") or []
            smiles    = _meta_value(comp_meta, "smiles")
            formula   = _meta_value(comp_meta, "molecular formula")

            # Spectrum-level metadata
            spec_meta     = record.get("metaData") or []
            instrument    = _meta_value(spec_meta, "instrument type") or _meta_value(spec_meta, "instrument")
            ms_level      = _meta_value(spec_meta, "ms level")
            ion_mode_val  = _meta_value(spec_meta, "ion mode")
            precursor_mz  = _meta_value(spec_meta, "precursor m/z")

            return MassSpectrum(
                mz        = mz,
                intensity = intensity,
                name      = name,
                metadata  = {
                    "smiles":          smiles,
                    "formula":         formula,
                    "inchikey":        inchikey,
                    "instrument_type": instrument,
                    "ms_level":        ms_level,
                    "ion_mode":        ion_mode_val,
                    "precursor_mz":    precursor_mz,
                    "mona_id":         record.get("id"),
                    "splash":          (record.get("splash") or {}).get("splash"),
                    "source":          "mona",
                },
            )
        except Exception as exc:
            logger.debug("Could not parse MoNA record %r: %s", record.get("id"), exc)
            return None

    def _filter_records(self, records: list) -> list:
        """Client-side filter: instrument type, MS level, ion mode."""
        filtered = records
        if self._instrument_type:
            key = self._instrument_type.upper()
            filtered = [
                r for r in filtered
                if key in (_meta_value(r.get("metaData"), "instrument type") or "").upper()
            ]
        if self._ms_level:
            level = self._ms_level.upper()
            filtered = [
                r for r in filtered
                if (_meta_value(r.get("metaData"), "ms level") or "").upper() == level
            ]
        if self._ion_mode:
            mode = self._ion_mode.upper()
            filtered = [
                r for r in filtered
                if (_meta_value(r.get("metaData"), "ion mode") or "").upper() == mode
            ]
        return filtered

    # ------------------------------------------------------------------
    # InChIKey helper
    # ------------------------------------------------------------------

    @staticmethod
    def _inchikey(smiles: str) -> str | None:
        from rgakit.molecule._utils import smiles_to_inchikey
        return smiles_to_inchikey(smiles)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        smiles:   str | None = None,
        inchikey: str | None = None,
        name:     str | None = None,
    ):
        """
        Retrieve a MassSpectrum from MoNA by structure or name.

        Provide exactly one of *smiles*, *inchikey*, or *name*.
        Returns the first matching :class:`~rgakit.spectrum.MassSpectrum` or
        ``None`` when no record is found.

        SMILES are converted to InChIKey internally so the lookup is
        format-agnostic.
        """
        given = {k: v for k, v in
                 [("smiles", smiles), ("inchikey", inchikey), ("name", name)]
                 if v is not None}
        if len(given) != 1:
            raise ValueError(
                f"Provide exactly one of: smiles, inchikey, name.  Got: {list(given)}"
            )

        key, value = next(iter(given.items()))
        label      = None

        if key == "smiles":
            ik = self._inchikey(value)
            if ik is None:
                logger.debug("MoNA: could not compute InChIKey for %r", value)
                return None
            records = self._search_by_inchikey(ik)
        elif key == "inchikey":
            records = self._search_by_inchikey(value)
        else:
            records = self._search_by_name(value)
            label   = value

        filtered = self._filter_records(records)
        if not filtered:
            logger.debug("MoNA: no records for %s=%r", key, value)
            return None

        if len(filtered) > 1:
            logger.debug("MoNA: %d records for %r, using first (%s).",
                         len(filtered), value, filtered[0].get("id"))

        return self._to_spectrum(filtered[0], label=label)

    def _search_by_inchikey(self, inchikey: str) -> list:
        """Query MoNA using RSQL for an exact InChIKey match."""
        query   = f'compound.inchiKey=="{inchikey}"'
        records = self._request("spectra/search", {"query": query})
        return records or []

    def _search_by_name(self, name: str) -> list:
        """Query MoNA for spectra matching a compound name."""
        query   = f'compound.names=q=\'name=="{name}"\''
        records = self._request("spectra/search", {"query": query})
        return records or []

    def search(
        self,
        name:            str | None = None,
        inchikey:        str | None = None,
        smiles:          str | None = None,
        instrument_type: str | None = None,
        ms_level:        str | None = None,
        ion_mode:        str | None = None,
    ) -> list:
        """
        Query MoNA with flexible filters and return all matching spectra.

        Parameters
        ----------
        name, inchikey, smiles : structural/name filters (at most one)
        instrument_type : override the instance instrument_type filter
        ms_level        : override the instance ms_level filter
        ion_mode        : override the instance ion_mode filter

        Returns a list of :class:`~rgakit.spectrum.MassSpectrum` objects.
        """
        given = {k: v for k, v in
                 [("smiles", smiles), ("inchikey", inchikey), ("name", name)]
                 if v is not None}
        if len(given) > 1:
            raise ValueError(
                f"Provide at most one of: smiles, inchikey, name.  Got: {list(given)}"
            )

        label   = None
        records = []

        if "smiles" in given:
            ik = self._inchikey(given["smiles"])
            if ik is None:
                logger.debug("MoNA: could not compute InChIKey for %r", given["smiles"])
                return []
            records = self._search_by_inchikey(ik)
        elif "inchikey" in given:
            records = self._search_by_inchikey(given["inchikey"])
        elif "name" in given:
            records = self._search_by_name(given["name"])
            label   = given["name"]

        # Apply client-side filters (use overrides if provided)
        eff_it    = instrument_type if instrument_type is not None else self._instrument_type
        eff_level = ms_level        if ms_level        is not None else self._ms_level
        eff_mode  = ion_mode        if ion_mode        is not None else self._ion_mode

        if eff_it:
            key = eff_it.upper()
            records = [
                r for r in records
                if key in (_meta_value(r.get("metaData"), "instrument type") or "").upper()
            ]
        if eff_level:
            level = eff_level.upper()
            records = [
                r for r in records
                if (_meta_value(r.get("metaData"), "ms level") or "").upper() == level
            ]
        if eff_mode:
            mode = eff_mode.upper()
            records = [
                r for r in records
                if (_meta_value(r.get("metaData"), "ion mode") or "").upper() == mode
            ]

        spectra = []
        for rec in records:
            spec = self._to_spectrum(rec, label=label)
            if spec is not None:
                spectra.append(spec)

        logger.info("MoNA search: %d spectra.", len(spectra))
        return spectra

    def to_library(self, **search_kwargs):
        """
        Build a :class:`~rgakit.library.SpectraLibrary` from a MoNA query.

        Keyword arguments are forwarded to :meth:`search`.
        Raises ``ValueError`` if no spectra are found.

        Example
        -------
        ::

            mona = MonaDatabase()
            lib  = mona.to_library(name="glucose")
        """
        from rgakit.library import SpectraLibrary

        spectra = self.search(**search_kwargs)
        if not spectra:
            raise ValueError(
                f"No MoNA spectra found for query: {search_kwargs}"
            )
        return SpectraLibrary(spectra)

    def __repr__(self) -> str:
        return (
            f"MonaDatabase({self._base_url!r}, "
            f"instrument_type={self._instrument_type!r}, "
            f"ms_level={self._ms_level!r})"
        )


# ---------------------------------------------------------------------------
# Local SQLite backend
# ---------------------------------------------------------------------------


class MonaLocalDatabase(LocalSpectralDatabase):
    """
    Read-only interface to a local MoNA SQLite spectral library.

    Uses the shared spectral library schema (schema.sql).
    Filtering is done in SQL, making it much faster than the REST API for
    bulk queries.

    Parameters
    ----------
    db_path         : path to ``mona_library.db``
    ionization_mode : filter by ionization mode, e.g. ``"positive"`` or
                      ``"negative"``.  ``None`` = no filter (default).
    ms_level        : filter by MS level, e.g. ``"MS2"``.  ``None`` = no filter.
    instrument_type : substring filter on instrument type, e.g. ``"EI"``.
                      ``None`` = no filter.

    Usage
    -----
        db = MonaLocalDatabase("/path/to/spectral_libraries/mona_library.db")

        spec = db.get(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        spec = db.get(smiles="CCO")

        spectra = db.search(inchikey="...", ms_level="MS2")
        lib     = db.to_library(inchikey="...")
    """

    _SOURCE = "mona"

    def __repr__(self) -> str:
        return (
            f"MonaLocalDatabase({str(self._db_path)!r}, "
            f"ionization_mode={self._ionization_mode!r}, "
            f"ms_level={self._ms_level!r}, "
            f"instrument_type={self._instrument_type!r})"
        )
