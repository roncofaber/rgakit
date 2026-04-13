"""
massbank.py
REST API client for MassBank (https://massbank.eu).

Provides MassBankDatabase for querying mass spectra by SMILES, InChIKey, or
compound name, and for building SpectraLibrary objects from MassBank queries.

Usage:

    from rgakit import MassBankDatabase

    mb = MassBankDatabase()                         # public massbank.eu API
    mb = MassBankDatabase("http://localhost:8081")  # local instance

    spec = mb.get(smiles="CCO")          # ethanol by SMILES (resolved via InChIKey)
    spec = mb.get(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    spec = mb.get(name="ethanol")

    lib = mb.to_library(name="glucose")  # all matching spectra as SpectraLibrary

By default only EI-MS records are returned (instrument type contains "EI",
ms_type = "MS").  Change the defaults via constructor parameters.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

_BASE_URL = "https://massbank.eu/MassBank-api"


class MassBankDatabase:
    """
    Read-only REST API client for the MassBank spectral reference library.

    Parameters
    ----------
    base_url         : API base URL.  Defaults to ``https://massbank.eu/MassBank-api``.
                       Pass a local address for a self-hosted instance.
    ms_type          : MS type filter passed to the API (default ``"MS"``).
                       Use ``None`` to disable the filter.
    instrument_type  : Instrument type substring filter applied client-side
                       (default ``"EI"``).  Matches any record whose
                       ``instrument_type`` field contains this string
                       (case-insensitive).  Use ``None`` to skip filtering.
    timeout          : HTTP request timeout in seconds (default 10).
    """

    def __init__(
        self,
        base_url:        str | None = None,
        ms_type:         str | None = "MS",
        instrument_type: str | None = "EI",
        timeout:         int        = 10,
    ):
        self._base_url       = (base_url or _BASE_URL).rstrip("/")
        self._ms_type        = ms_type
        self._instrument_type = instrument_type
        self._timeout        = timeout

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

        logger.debug("MassBank GET %s", url)
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("MassBank request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _to_spectrum(self, record: dict, label: str | None = None):
        """Convert a MassBank API record dict to a MassSpectrum."""
        from rgakit.spectrum import MassSpectrum

        try:
            values = record.get("peak", {}).get("peak", {}).get("values", [])
            if not values:
                return None

            mz  = np.array([v["mz"]  for v in values], dtype=float)
            rel = np.array([v.get("rel", 0) for v in values], dtype=float)
            # Fall back to absolute intensity if rel is all-zero
            if rel.max() == 0:
                rel = np.array([v.get("intensity", 0) for v in values], dtype=float)
            if rel.max() == 0:
                return None

            compound  = record.get("compound", {})
            names     = compound.get("names") or []
            name      = label or (names[0] if names else record.get("accession", "unknown"))

            inchikey  = None
            for lnk in compound.get("link") or []:
                if lnk.get("database") == "INCHIKEY":
                    inchikey = lnk.get("identifier")
                    break

            ms_info   = (record.get("acquisition") or {}).get("mass_spectrometry") or {}

            return MassSpectrum(
                mz        = mz,
                intensity = rel,
                name      = name,
                metadata  = {
                    "smiles":           compound.get("smiles"),
                    "formula":          compound.get("formula"),
                    "mw":               compound.get("mass"),
                    "inchikey":         inchikey,
                    "accession":        record.get("accession"),
                    "instrument_type":  (record.get("acquisition") or {}).get("instrument_type"),
                    "ms_type":          ms_info.get("ms_type"),
                    "ion_mode":         ms_info.get("ion_mode"),
                    "source":           "massbank",
                },
            )
        except Exception as exc:
            logger.debug("Could not parse MassBank record %r: %s",
                         record.get("accession"), exc)
            return None

    def _filter_instrument(self, records: list) -> list:
        """Client-side filter: keep records whose instrument_type contains the substring."""
        if not self._instrument_type:
            return records
        key = self._instrument_type.upper()
        return [
            r for r in records
            if key in ((r.get("acquisition") or {}).get("instrument_type") or "").upper()
        ]

    # ------------------------------------------------------------------
    # InChIKey helper (re-uses rgakit.insilico to avoid duplication)
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
        Retrieve a MassSpectrum from MassBank by structure or name.

        Provide exactly one of *smiles*, *inchikey*, or *name*.
        Returns the first matching :class:`~rgakit.spectrum.MassSpectrum` or
        ``None`` when no record is found.

        SMILES are converted to InChIKey internally so the lookup is
        format-agnostic.  When multiple records match, the first is returned
        and a debug message is logged.
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
        params     = {}

        if key == "smiles":
            ik = self._inchikey(value)
            if ik is None:
                logger.debug("MassBank: could not compute InChIKey for %r", value)
                return None
            params["inchi_key"] = ik
        elif key == "inchikey":
            params["inchi_key"] = value
        else:
            params["compound_name"] = value
            label = value

        if self._ms_type:
            params["ms_type"] = self._ms_type

        records = self._request("records", params)
        if not records:
            logger.debug("MassBank: no records for %s=%r", key, value)
            return None

        filtered = self._filter_instrument(records)
        if not filtered:
            logger.debug("MassBank: records found but none match instrument_type=%r",
                         self._instrument_type)
            return None

        if len(filtered) > 1:
            logger.debug("MassBank: %d records for %r, using first (%s).",
                         len(filtered), value, filtered[0].get("accession"))

        return self._to_spectrum(filtered[0], label=label)

    def get_by_accession(self, accession: str):
        """
        Retrieve a single MassBank record by its accession ID.

        Returns a :class:`~rgakit.spectrum.MassSpectrum` or ``None``.
        """
        record = self._request(f"records/{accession}", {})
        if record is None:
            return None
        return self._to_spectrum(record)

    def search(
        self,
        name:            str | None = None,
        formula:         str | None = None,
        inchikey:        str | None = None,
        ms_type:         str | None = None,
        instrument_type: str | None = None,
    ) -> list:
        """
        Query MassBank with flexible filters and return all matching spectra.

        Parameters
        ----------
        name, formula, inchikey : structural/name filters
        ms_type         : override the instance ms_type for this query
        instrument_type : override the instance instrument_type filter

        Returns a list of :class:`~rgakit.spectrum.MassSpectrum` objects.
        """
        params = {}
        if name:
            params["compound_name"] = name
        if formula:
            params["formula"] = formula
        if inchikey:
            params["inchi_key"] = inchikey
        effective_ms_type = ms_type if ms_type is not None else self._ms_type
        if effective_ms_type:
            params["ms_type"] = effective_ms_type

        records = self._request("records", params)
        if not records:
            return []

        # Client-side instrument filter (use override if given)
        eff_it = instrument_type if instrument_type is not None else self._instrument_type
        if eff_it:
            key = eff_it.upper()
            records = [
                r for r in records
                if key in ((r.get("acquisition") or {}).get("instrument_type") or "").upper()
            ]

        spectra = []
        for rec in records:
            spec = self._to_spectrum(rec)
            if spec is not None:
                spectra.append(spec)

        logger.info("MassBank search → %d spectra.", len(spectra))
        return spectra

    def to_library(self, **search_kwargs):
        """
        Build a :class:`~rgakit.library.SpectraLibrary` from a MassBank query.

        Keyword arguments are forwarded to :meth:`search`.
        Raises ``ValueError`` if no spectra are found.

        Example
        -------
        ::

            mb  = MassBankDatabase()
            lib = mb.to_library(name="glucose")
        """
        from rgakit.library import SpectraLibrary

        spectra = self.search(**search_kwargs)
        if not spectra:
            raise ValueError(
                f"No MassBank spectra found for query: {search_kwargs}"
            )
        return SpectraLibrary(spectra)

    def __repr__(self) -> str:
        return (
            f"MassBankDatabase({self._base_url!r}, "
            f"ms_type={self._ms_type!r}, "
            f"instrument_type={self._instrument_type!r})"
        )
