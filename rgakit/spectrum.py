"""
spectrum.py
MassSpectrum: a single compound's electron-ionisation mass spectrum.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .nomenclature import normalize_nist_name, resolve_name, get_compound_info
from .io import parse_jdx, generate_jdx, RGA_META_KEYS, parse_msp_blocks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MassSpectrum
# ---------------------------------------------------------------------------

class MassSpectrum:
    """
    Electron-ionisation mass spectrum of a single compound.

    Constructors
    ------------
    MassSpectrum(mz, intensity)          raw arrays
    MassSpectrum.from_file(path)         tab-separated txt
    MassSpectrum.from_jdx(text)          JCAMP-DX string
    MassSpectrum.from_jdx_file(path)     .jdx file
    MassSpectrum.from_nist(name=...)     fetch from NIST WebBook
    MassSpectrum.from_rga(rga)           extract from RGAMeasurement
    """

    def __init__(
        self,
        mz:        np.ndarray,
        intensity: np.ndarray,
        name:      str = "",
        metadata:  dict | None = None,
        jdx_text:  str | None = None,
    ):
        self.mz        = np.asarray(mz, dtype=int)
        self.intensity = np.asarray(intensity, dtype=float)
        self.name      = name
        self.metadata  = metadata or {}
        self._jdx_text = jdx_text

        if len(self.mz) != len(self.intensity):
            raise ValueError("mz and intensity must have the same length.")

    # ------------------------------------------------------------------
    # Typed metadata properties
    # ------------------------------------------------------------------

    @property
    def inchikey(self) -> str | None:
        """Standard InChIKey (27 chars). Reads from metadata, normalising key names."""
        return self.metadata.get("inchikey") or self.metadata.get("inchi_key")

    @property
    def cas(self) -> str | None:
        """CAS registry number string, or None."""
        return self.metadata.get("cas")

    @property
    def formula(self) -> str | None:
        """Molecular formula string (Hill notation), or None."""
        return self.metadata.get("formula")

    @property
    def source(self) -> str | None:
        """Origin tag: ``'nist'``, ``'massbank'``, ``'in_silico'``, ``'theoretical'``, or None."""
        return self.metadata.get("source")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str | Path) -> "MassSpectrum":
        """Load from a tab-separated txt file (m/z \\t intensity)."""
        filepath = Path(filepath)
        try:
            data = np.loadtxt(filepath, delimiter="\t", skiprows=1)
        except OSError as e:
            raise OSError(f"Cannot read spectrum file '{filepath}': {e}") from e
        except ValueError as e:
            raise ValueError(f"Malformed spectrum file '{filepath}': {e}") from e
        if data.ndim == 1:
            data = data[np.newaxis, :]
        name = filepath.stem.replace("_ms_peaks", "").replace("_", " ")
        return cls(data[:, 0].astype(int), data[:, 1].astype(float), name=name)

    @classmethod
    def from_jdx(cls, jdx_text: str, name: str = "") -> "MassSpectrum":
        """Parse a JCAMP-DX string (e.g. nistchempy's ms_specs[0].jdx_text)."""
        mz, intensity, metadata = parse_jdx(jdx_text)
        cas           = metadata.get("cas", "")
        resolved_name = resolve_name(cas=cas, nist_name=name or metadata.get("name", ""))
        return cls(mz, intensity, name=resolved_name, metadata=metadata, jdx_text=jdx_text)

    @classmethod
    def from_jdx_file(cls, filepath: str | Path) -> "MassSpectrum":
        """Load from a saved .jdx file."""
        return cls.from_jdx(Path(filepath).read_text())

    @classmethod
    def from_nist(
        cls,
        name:      str | None        = None,
        cas:       str | None        = None,
        smiles:    str | None        = None,
        inchikey:  str | None        = None,
        formula:   str | None        = None,
        ms_index:  int               = 0,
        local_dir: str | Path | None = None,
    ) -> "MassSpectrum":
        """
        Load a mass spectrum from NIST, using a local cache to avoid repeat fetches.

        Lookup order:
          1. User cache (``platformdirs.user_cache_dir('rgakit') / 'MS'``)
          2. *local_dir* if provided — file is copied into the cache on first use.
          3. NIST WebBook (fetched and saved to cache for future calls).

        Files are stored as ``{compound.ID}_MS_{ms_index}.jdx`` (NIST naming).

        Parameters
        ----------
        name, cas, inchikey, smiles, formula : identifier (provide one).
            *inchikey* is the preferred route — it queries NIST directly with no
            intermediate PubChem call.  *smiles* resolves via PubChem first.
            Formula search is ambiguous; the first NIST hit with MS data is used.
        ms_index  : which spectrum to use when multiple are available
        local_dir : optional extra source directory (e.g. ~/Downloads/MS)
        """
        try:
            import nistchempy as nist
        except ImportError:
            raise ImportError("nistchempy is required: pip install nistchempy")
        from platformdirs import user_cache_dir
        import shutil

        # Compute InChIKey from SMILES via RDKit (avoids PubChem round-trip)
        if smiles is not None and inchikey is None:
            from rgakit.molecule.utils import smiles_to_inchikey
            inchikey = smiles_to_inchikey(smiles)
            if inchikey is None:
                # Fall back to PubChem CAS resolution
                try:
                    info = get_compound_info(smiles, "smiles")
                except Exception as exc:
                    raise ValueError(
                        f"PubChem lookup failed for SMILES {smiles!r}: {exc}"
                    ) from exc
                if not info:
                    raise ValueError(f"SMILES not found in PubChem: {smiles!r}")
                if info.get("cas"):
                    cas = info["cas"]
                elif info.get("iupac"):
                    name = info["iupac"]
                else:
                    raise ValueError(
                        f"PubChem found {smiles!r} but could not resolve CAS or IUPAC name"
                    )

        try:
            if inchikey is not None:
                results = nist.run_search(inchikey, "inchi")
            elif name is not None:
                results = nist.run_search(name, "name")
            elif cas is not None:
                results = nist.run_search(cas, "cas")
            elif formula is not None:
                results = nist.run_search(formula, "formula")
            else:
                raise ValueError("Provide inchikey, name, cas, smiles, or formula.")
        except Exception as exc:
            raise ValueError(
                f"NIST search failed for {inchikey or name or cas or smiles or formula!r}: {exc}"
            ) from exc

        if not results.compounds:
            raise ValueError(
                f"No NIST entry for: {inchikey or name or cas or smiles or formula!r}"
            )

        # For formula searches, skip to the first compound that actually has MS data
        if formula is not None and inchikey is None and name is None and cas is None:
            compound = None
            for candidate in results.compounds:
                try:
                    candidate.get_ms_spectra()
                    if candidate.ms_specs:
                        compound = candidate
                        break
                except Exception:
                    continue
            if compound is None:
                raise ValueError(
                    f"No MS spectra found for any isomer of formula {formula!r}"
                )
            logger.info(
                "Formula search %r matched %r (first isomer with MS data) — "
                "verify this is the intended compound.",
                formula, compound.name,
            )
        else:
            compound = results.compounds[0]

        filename = f"{compound.ID}_MS_{ms_index}.jdx"

        cache_dir = Path(user_cache_dir("rgakit")) / "MS"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / filename

        # Enrich metadata — use known inchikey/smiles first, fall back to PubChem
        pubchem: dict = {}
        if not inchikey and compound.cas_rn:
            try:
                pubchem = get_compound_info(compound.cas_rn, "cas") or {}
            except Exception:
                pubchem = {}

        metadata = {k: v for k, v in {
            "nist_id":  compound.ID,
            "cas":      compound.cas_rn,
            "formula":  pubchem.get("formula") or compound.formula,
            "mw":       pubchem.get("mw") or str(compound.mol_weight),
            "inchikey": inchikey or pubchem.get("inchikey") or getattr(compound, "inchi_key", None),
            "smiles":   smiles or pubchem.get("smiles"),
            "source":   "nist",
        }.items() if v}

        def _finalise(obj):
            obj.name = resolve_name(cas=compound.cas_rn, nist_name=compound.name)
            obj.metadata.update(metadata)
            return obj

        # 1 — user cache hit
        if cache_path.exists():
            logger.debug("NIST cache hit: %s", cache_path.name)
            return _finalise(cls.from_jdx_file(cache_path))

        # 2 — local_dir seed
        if local_dir is not None:
            seed = Path(local_dir) / filename
            if seed.exists():
                logger.debug("NIST local seed: copying %s to cache", seed.name)
                shutil.copy2(seed, cache_path)
                return _finalise(cls.from_jdx_file(cache_path))

        # 3 — fetch from NIST and cache
        logger.info("Fetching spectrum from NIST: %r (CAS %s)", compound.name, compound.cas_rn)
        compound.get_ms_spectra()
        if not compound.ms_specs:
            raise ValueError(f"No mass spectra available for {compound.name!r} on NIST.")

        jdx_text = compound.ms_specs[ms_index].jdx_text
        cache_path.write_text(jdx_text)
        logger.debug("NIST spectrum cached to %s", cache_path)
        return _finalise(cls.from_jdx(jdx_text, name=compound.name))

    @classmethod
    def from_rga(
        cls,
        rga,
        name: str | None = None,
    ) -> "MassSpectrum":
        """
        Extract the shutter-open spectrum from a background-corrected
        RGAMeasurement object.

        Call ``rga.background_correct()`` before this method.  If that has
        been done ``rga.open_time`` / ``rga.close_time`` are used directly;
        otherwise the shutter-open window is derived from ``rga.shutter``.

        Parameters
        ----------
        rga  : RGAMeasurement (clabs)
        name : label; defaults to rga.sample_name
        """
        mz_full = np.arange(1, rga.pressure.shape[1] + 1)

        if hasattr(rga, "open_time") and hasattr(rga, "close_time"):
            open_time  = rga.open_time
            close_time = rga.close_time
        else:
            open_mask = rga.shutter > 0
            if not open_mask.any():
                raise ValueError("Shutter is never open in this measurement.")
            open_time  = rga.tey_time[open_mask].min()
            close_time = rga.tey_time[open_mask].max()

        rga_mask = (rga.time >= open_time) & (rga.time <= close_time)
        if not rga_mask.any():
            raise ValueError(
                f"No RGA scans within shutter-open window "
                f"({open_time:.1f} – {close_time:.1f} s)."
            )

        mean_pressure = rga.pressure[rga_mask, :].mean(axis=0)
        nonzero       = mean_pressure > 0

        metadata = dict(
            sample_name        = rga.sample_name,
            x                  = rga.x,
            y                  = rga.y,
            pd_ua              = rga.pd,
            n_open_scans       = int(rga_mask.sum()),
            scan_settings      = rga.scan_settings,
            background_correct = hasattr(rga, "open_time"),
        )
        return cls(mz_full[nonzero], mean_pressure[nonzero],
                   name=name or rga.sample_name or "", metadata=metadata)

    @classmethod
    def from_msp(cls, text: str, index: int = 0) -> "MassSpectrum":
        """
        Parse a single spectrum from NIST MSP text.

        Parameters
        ----------
        text  : full content of an MSP file (may contain multiple entries)
        index : which entry to return (0-based)
        """
        blocks = parse_msp_blocks(text)
        if not blocks:
            raise ValueError("No spectra found in MSP text.")
        if index >= len(blocks):
            raise IndexError(
                f"index={index} out of range; file contains {len(blocks)} spectra."
            )
        block = blocks[index]
        mz        = block.pop("mz")
        intensity = block.pop("intensity")
        raw_name  = block.get("name", "")
        cas       = str(block.get("cas", ""))
        resolved  = resolve_name(cas=cas, nist_name=raw_name)
        return cls(mz, intensity, name=resolved, metadata=block)

    @classmethod
    def from_msp_file(cls, path: str | Path, index: int = 0) -> "MassSpectrum":
        """Load one spectrum from an MSP file (0-based *index*)."""
        return cls.from_msp(Path(path).read_text(), index=index)

    @classmethod
    def all_from_msp_file(cls, path: str | Path) -> list["MassSpectrum"]:
        """Load every spectrum in an MSP file and return them as a list."""
        text   = Path(path).read_text()
        blocks = parse_msp_blocks(text)
        spectra = []
        for block in blocks:
            mz        = block.pop("mz")
            intensity = block.pop("intensity")
            raw_name  = block.get("name", "")
            cas       = str(block.get("cas", ""))
            resolved  = resolve_name(cas=cas, nist_name=raw_name)
            spectra.append(cls(mz, intensity, name=resolved, metadata=block))
        return spectra

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save as a tab-separated txt file."""
        dest = Path(path)
        with open(dest, "w") as f:
            f.write("m/z\tintensity\n")
            for mz, inten in zip(self.mz, self.intensity):
                f.write(f"{mz}\t{inten}\n")
        return dest

    # RGA-specific metadata keys preserved in JDX round-trips (defined in io.jdx)
    _RGA_META_KEYS = RGA_META_KEYS

    def _generate_jdx(self) -> str:
        """Generate a JCAMP-DX string from the stored mz/intensity arrays."""
        max_i  = self.intensity.max() or 1.0
        scaled = (self.intensity / max_i * 9999).round().astype(int)
        pairs  = [f"{mz},{inten}" for mz, inten in zip(self.mz, scaled)]
        rows   = [" ".join(pairs[i:i+4]) for i in range(0, len(pairs), 4)]
        lines  = [
            f"##TITLE={self.name}",
            "##JCAMP-DX=4.24",
            "##DATA TYPE=MASS SPECTRUM",
            "##ORIGIN=rgakit",
            f"##CAS REGISTRY NO={self.metadata.get('cas', 'N/A')}",
        ]
        if self.metadata.get("formula"):
            lines.append(f"##MOLFORM={self.metadata['formula']}")
        if self.metadata.get("mw"):
            lines.append(f"##MW={self.metadata['mw']}")
        # Embed RGA-specific fields so they survive a save/load round-trip
        for key in self._RGA_META_KEYS:
            if key in self.metadata:
                lines.append(f"##${key}={self.metadata[key]}")
        lines += [
            "##XUNITS=M/Z",
            "##YUNITS=RELATIVE INTENSITY",
            "##XFACTOR=1",
            f"##YFACTOR={max_i / 9999:.6e}",
            f"##FIRSTX={self.mz.min()}",
            f"##LASTX={self.mz.max()}",
            "##MAXY=9999",
            f"##NPOINTS={len(self.mz)}",
            "##PEAK TABLE=(XY..XY)",
            *rows,
            "##END=",
        ]
        return "\n".join(lines)

    def save_jdx(self, path: str | Path | None = None) -> Path:
        """
        Write a JCAMP-DX file.

        Uses the original JDX text when available (from_nist / from_jdx),
        otherwise generates one from the stored arrays.
        If *path* is a directory the filename is derived from self.name.
        """
        jdx = self._jdx_text or self._generate_jdx()
        dest = Path(path) if path is not None else Path(".")
        if dest.is_dir():
            safe = re.sub(r'[^\w\s,().\-]', '', self.name).replace(" ", "_")
            dest = dest / f"{safe}.jdx"
        dest.write_text(jdx)
        return dest

    def save_msp(self, path: str | Path | None = None) -> Path:
        """
        Write a NIST MSP file.

        Intensities are scaled to integer [0, 999] (NIST convention).
        If *path* is a directory the filename is derived from ``self.name``.

        MSP is a plain-text format supported by NIST MS Search, mzCloud,
        and many other mass spec tools.  One file may contain multiple
        entries (see :meth:`all_from_msp_file`).
        """
        max_i  = self.intensity.max() or 1.0
        scaled = (self.intensity / max_i * 999).round().astype(int)

        lines = [f"Name: {self.name}"]
        if self.metadata.get("cas"):
            lines.append(f"CAS#: {self.metadata['cas']}")
        if self.metadata.get("formula"):
            lines.append(f"Formula: {self.metadata['formula']}")
        if self.metadata.get("mw"):
            lines.append(f"MW: {self.metadata['mw']}")

        lines.append(f"Num Peaks: {len(self.mz)}")
        # Up to 5 m/z-intensity pairs per line (NIST convention)
        pairs = [f"{mz} {inten}" for mz, inten in zip(self.mz, scaled)]
        for i in range(0, len(pairs), 5):
            lines.append(" ".join(pairs[i:i + 5]))
        lines.append("")   # blank line between entries

        msp_text = "\n".join(lines)
        dest = Path(path) if path is not None else Path(".")
        if dest.is_dir():
            safe = re.sub(r'[^\w\s,().\-]', '', self.name).replace(" ", "_")
            dest = dest / f"{safe}.msp"
        dest.write_text(msp_text)
        return dest

    # ------------------------------------------------------------------
    # Properties / derived views
    # ------------------------------------------------------------------

    @property
    def normalized(self) -> np.ndarray:
        """Intensities rescaled to [0, 1]."""
        peak = self.intensity.max()
        return self.intensity / peak if peak > 0 else np.zeros_like(self.intensity)

    def on_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Project onto an m/z grid, returning a dense array normalized to [0, 1].

        Intensities are divided by the spectrum's own maximum so that the
        largest peak maps to 1.0. m/z values absent from *grid* are silently
        set to 0. This normalization is intentional: it is what makes the NNLS
        weight matrix ``A`` scale-independent.
        """
        grid    = np.asarray(grid, dtype=int)
        out     = np.zeros(len(grid), dtype=float)
        idx_map = {mz: i for i, mz in enumerate(grid)}
        norm    = self.intensity.max() or 1.0
        for mz, inten in zip(self.mz, self.intensity):
            if mz in idx_map:
                out[idx_map[mz]] = inten / norm
        return out

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_mz(
        self,
        mz_min: int | None = None,
        mz_max: int | None = None,
        exclude: list[int] | None = None,
        min_intensity: float | None = None,
    ) -> "MassSpectrum":
        """
        Return a new MassSpectrum with peaks restricted by the given rules.

        Parameters
        ----------
        mz_min        : drop all peaks with m/z < mz_min
        mz_max        : drop all peaks with m/z > mz_max
        exclude       : list of specific m/z values to drop (e.g. [2] to remove H2)
        min_intensity : drop peaks whose intensity is below this fraction of the
                        base peak (e.g. 0.01 removes anything below 1% of max).

        Examples
        --------
        >>> ms.filter_mz(mz_min=3)                    # ignore H and H2
        >>> ms.filter_mz(mz_min=3, mz_max=80)         # keep 3–80
        >>> ms.filter_mz(exclude=[18, 28])            # drop water and CO/N2
        >>> ms.filter_mz(min_intensity=0.01)          # drop noise below 1% of base peak
        """
        mask = np.ones(len(self.mz), dtype=bool)
        if mz_min is not None:
            mask &= self.mz >= mz_min
        if mz_max is not None:
            mask &= self.mz <= mz_max
        if exclude is not None:
            mask &= ~np.isin(self.mz, exclude)
        if min_intensity is not None:
            peak = self.intensity.max() or 1.0
            mask &= self.intensity >= min_intensity * peak
        return MassSpectrum(
            self.mz[mask], self.intensity[mask],
            name=self.name, metadata=self.metadata.copy(),
            jdx_text=self._jdx_text,
        )

    # ------------------------------------------------------------------
    # Spectrum comparison
    # ------------------------------------------------------------------

    def cosine_similarity(self, other: "MassSpectrum") -> float:
        """Dot-product cosine similarity on the shared m/z grid, returns [0, 1]."""
        from .similarity import cosine
        return cosine(self, other)

    def similarity_score(
        self, other: "MassSpectrum", method: str = "cosine", **kwargs
    ) -> float:
        """
        Similarity to *other* using *method*.

        Parameters
        ----------
        method : ``"cosine"`` | ``"jaccard"`` | ``"pearson"`` | ``"entropy"``
        **kwargs : forwarded to the underlying metric (e.g. ``threshold`` for jaccard)
        """
        from .similarity import score
        return score(self, other, method=method, **kwargs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def plot(self, ax=None, label: str | None = None, **kwargs):
        """Bar plot of the spectrum."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3))
        ax.bar(self.mz, self.normalized, width=0.8,
               label=label if label is not None else self.name, **kwargs)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity")
        ax.set_title(self.name)
        return ax

    def __repr__(self) -> str:
        cas  = self.metadata.get("cas", "?")
        form = self.metadata.get("formula", "?")
        return f"MassSpectrum({self.name!r}, peaks={len(self.mz)}, CAS={cas}, formula={form})"
