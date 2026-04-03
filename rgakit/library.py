"""
library.py
----------
SpectraLibrary: collection of reference spectra + NNLS fitting.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .spectrum import MassSpectrum
from .result   import FitResult


class SpectraLibrary:
    """
    A collection of reference MassSpectrum objects used to decompose an
    unknown signal by non-negative least-squares fitting.

    Usage
    -----
        lib    = SpectraLibrary.from_dir("data/")
        result = lib.fit(mz, intensity)
        result.summary()
    """

    def __init__(self, spectra: list[MassSpectrum]):
        if not spectra:
            raise ValueError("Library must contain at least one spectrum.")
        self._spectra: dict[str, MassSpectrum] = {s.name: s for s in spectra}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "SpectraLibrary":
        """
        Load all spectra from a directory.

        Search order (first non-empty wins):
          1. ``*.jdx`` files (JCAMP-DX)
          2. ``*.msp`` files (NIST MSP — one or many entries per file)
          3. ``*_ms_peaks.txt`` files (legacy tab-separated)
        """
        data_dir  = Path(data_dir)
        jdx_files = sorted(data_dir.glob("*.jdx"))
        if jdx_files:
            return cls([MassSpectrum.from_jdx_file(f) for f in jdx_files])
        msp_files = sorted(data_dir.glob("*.msp"))
        if msp_files:
            spectra = []
            for f in msp_files:
                spectra.extend(MassSpectrum.all_from_msp_file(f))
            return cls(spectra)
        txt_files = sorted(data_dir.glob("*_ms_peaks.txt"))
        if txt_files:
            return cls([MassSpectrum.from_file(f) for f in txt_files])
        raise FileNotFoundError(
            f"No .jdx, .msp, or *_ms_peaks.txt files in {data_dir}"
        )

    @classmethod
    def from_msp(cls, path: str | Path) -> "SpectraLibrary":
        """Load all spectra from a single MSP file."""
        return cls(MassSpectrum.all_from_msp_file(path))

    @classmethod
    def from_nist(
        cls,
        names:    list[str] | None = None,
        cas_list: list[str] | None = None,
    ) -> "SpectraLibrary":
        """Build a library by fetching compounds from NIST."""
        spectra = [MassSpectrum.from_nist(name=n) for n in (names    or [])]
        spectra += [MassSpectrum.from_nist(cas=c)  for c in (cas_list or [])]
        return cls(spectra)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self)                         -> int:           return len(self._spectra)
    def __getitem__(self, name: str)          -> MassSpectrum:  return self._spectra[name]
    def __iter__(self):                                          return iter(self._spectra.values())
    def __repr__(self)                        -> str:           return f"SpectraLibrary({len(self)} compounds)"

    def names(self) -> list[str]:
        return list(self._spectra.keys())

    def add(self, spectrum: MassSpectrum, overwrite: bool = False) -> None:
        """
        Add a spectrum to the library.

        Parameters
        ----------
        spectrum  : MassSpectrum to add
        overwrite : if False (default) warn when replacing an existing entry
        """
        if spectrum.name in self._spectra and not overwrite:
            warnings.warn(
                f"Spectrum '{spectrum.name}' already in library and will be overwritten. "
                "Pass overwrite=True to suppress this warning.",
                stacklevel=2,
            )
        self._spectra[spectrum.name] = spectrum

    @property
    def grid(self) -> np.ndarray:
        """Union of all m/z values across every library spectrum."""
        return np.unique(np.concatenate([s.mz for s in self._spectra.values()]))

    # ------------------------------------------------------------------
    # Cosine search
    # ------------------------------------------------------------------

    def search(
        self,
        spectrum: "MassSpectrum",
        top_n:  int = 5,
        method: str = "cosine",
        **kwargs,
    ) -> list[tuple[str, float]]:
        """
        Rank library spectra by similarity against *spectrum*.

        Returns a list of ``(name, score)`` tuples, sorted best-first,
        limited to *top_n* results.  Scores are in [0, 1].

        Parameters
        ----------
        method : ``"cosine"`` | ``"jaccard"`` | ``"pearson"`` | ``"entropy"``
        **kwargs : forwarded to the similarity metric
        """
        from .similarity import score as _score
        results = [
            (name, _score(ref, spectrum, method=method, **kwargs))
            for name, ref in self._spectra.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        spectrum_or_mz,
        intensity: np.ndarray | None = None,
        grid:      np.ndarray | None = None,
    ) -> FitResult:
        """
        Decompose an unknown spectrum by non-negative least squares (NNLS).

        Each reference spectrum in the library is normalized to [0, 1] and
        projected onto a common m/z grid to form the design matrix ``A``.
        NNLS solves ``min ||A·w - y||`` subject to ``w >= 0``, where ``y``
        is the normalized observed spectrum. Weights are therefore in units
        of the observed spectrum's peak intensity, not absolute pressure.

        Parameters
        ----------
        spectrum_or_mz : MassSpectrum  *or*  array-like of integer m/z values
        intensity      : array-like of intensities (required when the first
                         argument is an m/z array, ignored otherwise)
        grid           : m/z values to use for fitting. Defaults to the
                         intersection of the observed m/z and the library's
                         union grid. A small intersection reduces fit quality;
                         a warning is issued when fewer than 10 channels overlap.

        Returns
        -------
        FitResult
        """
        if isinstance(spectrum_or_mz, MassSpectrum):
            mz_obs = spectrum_or_mz.mz
            y_obs  = spectrum_or_mz.intensity.astype(float)
        else:
            if intensity is None:
                raise ValueError(
                    "intensity must be provided when the first argument is an m/z array. "
                    "Alternatively, pass a MassSpectrum object directly."
                )
            mz_obs = np.asarray(spectrum_or_mz, dtype=int)
            y_obs  = np.asarray(intensity, dtype=float)

        if grid is None:
            grid = np.intersect1d(mz_obs, self.grid)

        if len(grid) == 0:
            raise ValueError("No m/z overlap between observed spectrum and library.")

        if len(grid) < 10:
            warnings.warn(
                f"Only {len(grid)} m/z channel(s) overlap between the observed spectrum "
                "and the library. Fit quality may be poor.",
                stacklevel=2,
            )

        y     = MassSpectrum(mz_obs, y_obs).on_grid(grid)
        names = self.names()
        A     = np.column_stack([self._spectra[n].on_grid(grid) for n in names])

        weights_arr, residual = nnls(A, y)

        return FitResult(
            weights  = dict(zip(names, weights_arr)),
            residual = residual,
            fitted   = A @ weights_arr,
            observed = y,
            grid     = grid,
        )
