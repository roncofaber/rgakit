"""
library.py
----------
SpectraLibrary: collection of reference spectra with spectrum fitting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .spectrum import MassSpectrum
from .result   import FitResult, TimeFitResult
from .stack    import SpectrumStack
from .solvers  import make_solver

logger = logging.getLogger(__name__)


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
            logger.info("Loading library: %d JDX files from %s", len(jdx_files), data_dir)
            lib = cls([MassSpectrum.from_jdx_file(f) for f in jdx_files])
            logger.info("Library loaded: %d compounds", len(lib))
            return lib
        msp_files = sorted(data_dir.glob("*.msp"))
        if msp_files:
            logger.info("Loading library: %d MSP file(s) from %s", len(msp_files), data_dir)
            spectra = []
            for f in msp_files:
                spectra.extend(MassSpectrum.all_from_msp_file(f))
            lib = cls(spectra)
            logger.info("Library loaded: %d compounds", len(lib))
            return lib
        txt_files = sorted(data_dir.glob("*_ms_peaks.txt"))
        if txt_files:
            logger.info("Loading library: %d TXT files from %s", len(txt_files), data_dir)
            lib = cls([MassSpectrum.from_file(f) for f in txt_files])
            logger.info("Library loaded: %d compounds", len(lib))
            return lib
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
        names:       list[str] | None = None,
        cas_list:    list[str] | None = None,
        smiles_list: list[str] | None = None,
    ) -> "SpectraLibrary":
        """Build a library by fetching compounds from NIST."""
        spectra  = [MassSpectrum.from_nist(name=n)   for n in (names       or [])]
        spectra += [MassSpectrum.from_nist(cas=c)     for c in (cas_list    or [])]
        spectra += [MassSpectrum.from_nist(smiles=s)  for s in (smiles_list or [])]
        return cls(spectra)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self)                         -> int:           return len(self._spectra)
    def __getitem__(self, name: str)          -> MassSpectrum:  return self._spectra[name]
    def __iter__(self):                                          return iter(self._spectra.values())
    def __repr__(self)                        -> str:           return f"SpectraLibrary({len(self)} compounds)"

    def save(self, path: str) -> None:
        """Serialize this SpectraLibrary to disk with dill."""
        import dill
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SpectraLibrary":
        """Load a SpectraLibrary previously saved with :meth:`save`."""
        import dill
        with open(path, "rb") as f:
            return dill.load(f)

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
            logger.warning(
                "Spectrum %r already in library — overwriting. "
                "Pass overwrite=True to suppress this warning.",
                spectrum.name,
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
        intensity:     np.ndarray | None = None,
        grid:          np.ndarray | None = None,
        mz_min:        int | None        = None,
        mz_max:        int | None        = None,
        exclude_mz:    list[int] | None  = None,
        min_intensity: float | None      = None,
        method:        str               = "nnls",
        alpha:         float             = 0.1,
    ) -> FitResult:
        """
        Decompose an unknown spectrum by non-negative least squares or sparse fitting.

        Each reference spectrum in the library is normalized to [0, 1] and
        projected onto a common m/z grid to form the design matrix ``A``.
        The solver minimises the reconstruction error subject to non-negative
        weights. Weights are in units of the observed spectrum's peak intensity,
        not absolute pressure.

        Parameters
        ----------
        spectrum_or_mz : MassSpectrum  *or*  array-like of integer m/z values
        intensity      : array-like of intensities (required when the first
                         argument is an m/z array, ignored otherwise)
        grid           : m/z values to use for fitting. Defaults to the
                         intersection of the observed m/z and the library's
                         union grid. A small intersection reduces fit quality;
                         a warning is issued when fewer than 10 channels overlap.
        mz_min         : exclude all m/z channels below this value from the fit
        mz_max         : exclude all m/z channels above this value from the fit
        exclude_mz     : list of specific m/z values to drop (e.g. ``[2]`` to
                         ignore H2, or ``[18, 28]`` to drop water and CO/N2)
        min_intensity  : drop m/z channels whose observed intensity is below
                         this fraction of the spectrum's base peak before
                         fitting (e.g. ``0.01`` removes noise below 1% of max).
        method         : ``"nnls"`` (default) — non-negative least squares via
                         scipy; ``"lasso"`` — non-negative LASSO (requires
                         scikit-learn) which promotes a sparse solution by adding
                         an L1 penalty, useful when only a few compounds are
                         expected to be present.
        alpha          : Relative L1 regularisation strength for ``method="lasso"``,
                         in (0, 1). Scaled internally by the per-call ``alpha_max``
                         so its meaning is independent of data scale or channel
                         count. Larger values yield fewer non-zero compounds;
                         0.1 is a good starting point. Ignored for ``"nnls"``.

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

        if min_intensity is not None:
            peak = y_obs.max() or 1.0
            keep = y_obs >= min_intensity * peak
            mz_obs = mz_obs[keep]
            y_obs  = y_obs[keep]

        if grid is None:
            grid = np.intersect1d(mz_obs, self.grid)

        if mz_min is not None:
            grid = grid[grid >= mz_min]
        if mz_max is not None:
            grid = grid[grid <= mz_max]
        if exclude_mz is not None:
            grid = grid[~np.isin(grid, exclude_mz)]

        if len(grid) == 0:
            raise ValueError("No m/z overlap between observed spectrum and library.")

        if len(grid) < 10:
            logger.warning(
                "Only %d m/z channel(s) overlap between the observed spectrum "
                "and the library — fit quality may be poor.",
                len(grid),
            )

        logger.debug(
            "Fitting: %d library compounds × %d m/z channels (method=%s)",
            len(self._spectra), len(grid), method,
        )
        y     = MassSpectrum(mz_obs, y_obs).on_grid(grid)
        names = self.names()
        A     = np.column_stack([self._spectra[n].on_grid(grid) for n in names])

        solve = make_solver(method, alpha)
        weights_arr, residual = solve(A, y)
        n_det = sum(1 for w in weights_arr if w > 0)
        logger.info(
            "Fit complete (%s): %d/%d compounds detected, residual=%.4f",
            method, n_det, len(names), residual,
        )

        fit_params = {k: v for k, v in {
            "mz_min":        mz_min,
            "mz_max":        mz_max,
            "exclude_mz":    exclude_mz,
            "min_intensity": min_intensity,
            "method":        method if method != "nnls" else None,
            "alpha":         alpha  if method == "lasso" else None,
        }.items() if v is not None}

        spectral_contributions = {
            names[j]: A[:, j] * weights_arr[j]
            for j in range(len(names))
        }

        return FitResult(
            weights                = dict(zip(names, weights_arr)),
            residual               = residual,
            fitted                 = A @ weights_arr,
            observed               = y,
            grid                   = grid,
            fit_params             = fit_params or None,
            spectral_contributions = spectral_contributions,
        )

    def fit_time_series(
        self,
        stack,
        time_range: tuple[float, float] | None = None,
        mz_min:     int | None                 = None,
        mz_max:     int | None                 = None,
        exclude_mz: list[int] | None           = None,
        method:     str                        = "nnls",
        alpha:      float                      = 0.1,
    ) -> TimeFitResult:
        """
        Fit every scan in a time-series and return per-scan contributions.

        The design matrix is built once from the library and reused for each
        scan. When ``method="lasso"``, a single model instance is reused with
        ``warm_start`` so each scan benefits from the previous solution as a
        starting point.

        Parameters
        ----------
        stack      : :class:`~rgakit.SpectrumStack` (preferred) *or* a clabs
                     ``RGAMeasurement`` (auto-wrapped for convenience).
                     Call :meth:`~rgakit.SpectrumStack.background_correct`
                     before fitting when background correction is needed.
        time_range : ``(t_start, t_end)`` in seconds.  Defaults to
                     :attr:`~rgakit.SpectrumStack.open_window`.
        mz_min     : exclude m/z channels below this value
        mz_max     : exclude m/z channels above this value
        exclude_mz : list of specific m/z values to drop
        method     : ``"nnls"`` (default) or ``"lasso"`` — see
                     :meth:`fit` for details.
        alpha      : Relative L1 regularisation strength for ``method="lasso"``,
                     in (0, 1). See :meth:`fit` for details. Ignored for ``"nnls"``.

        Returns
        -------
        TimeFitResult
        """
        if not isinstance(stack, SpectrumStack):
            stack = SpectrumStack.from_rga(stack)

        t_start, t_end = time_range if time_range is not None else stack.open_window
        mask = (stack.time >= t_start) & (stack.time <= t_end)
        if not mask.any():
            raise ValueError(
                f"No scans in time range [{t_start:.1f}, {t_end:.1f}] s."
            )

        times    = stack.time[mask]
        pressure = stack.pressure[mask, :].astype(float)

        # Map stack.mz → grid index (stack.mz may not start at 1)
        grid = np.intersect1d(stack.mz, self.grid)
        if mz_min     is not None: grid = grid[grid >= mz_min]
        if mz_max     is not None: grid = grid[grid <= mz_max]
        if exclude_mz is not None: grid = grid[~np.isin(grid, exclude_mz)]

        if len(grid) == 0:
            raise ValueError("No m/z overlap between stack and library after filters.")
        if len(grid) < 10:
            logger.warning(
                "Only %d m/z channels in the fitting grid — fit quality may be poor.",
                len(grid),
            )

        names = self.names()
        A     = np.column_stack([self._spectra[n].on_grid(grid) for n in names])

        # Project all scans onto grid at once
        mz_to_idx = {mz: i for i, mz in enumerate(stack.mz)}
        grid_idx  = np.array([mz_to_idx[mz] for mz in grid])
        Y         = pressure[:, grid_idx]          # (n_times, n_grid)
        norms     = Y.max(axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Y        /= norms

        n_times   = len(times)
        logger.info(
            "Time-series fit: %d scans x %d compounds x %d m/z channels "
            "(t=%.1f-%.1f s, method=%s)",
            n_times, len(names), len(grid), times[0], times[-1], method,
        )
        solve     = make_solver(method, alpha)
        W         = np.zeros((n_times, len(names)))
        residuals = np.zeros(n_times)
        log_every = max(1, n_times // 10)
        for i in range(n_times):
            W[i], residuals[i] = solve(A, Y[i])
            if (i + 1) % log_every == 0 or i == n_times - 1:
                logger.debug("Time-series fit: %d/%d scans done", i + 1, n_times)

        logger.info(
            "Time-series fit complete: mean residual=%.4f", residuals.mean()
        )
        weights = {name: W[:, j] for j, name in enumerate(names)}
        return TimeFitResult(times, weights, residuals, grid, names)
