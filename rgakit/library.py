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


def _channel_weights(y: np.ndarray, grid: np.ndarray, noise_floor: float = 0.01) -> np.ndarray:
    """Per-channel scale factors for weighted least squares.

    ``sw_i = y_i^(-0.25) × m/z_i^(0.1)``

    The mild inverse-intensity term (Poisson correction) gives small diagnostic
    peaks more weight; the mass term gently favours higher-m/z channels.
    """
    y_floor = np.maximum(y, noise_floor * max(float(y.max()), 1e-30))
    return np.power(y_floor, -0.25) * np.power(grid.astype(float), 0.1)


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

    @classmethod
    def from_fit(
        cls,
        spectrum:        "MassSpectrum",
        database,
        min_intensity:   float | None = 1e-4,
        min_improvement: float        = 0.005,
        n_trials:        int          = 1,
        temperature:     float        = 0.3,
        max_compounds:   int               = 30,
        max_mw:          float | None      = None,
        allowed_elements: set[str] | None  = None,
        weighted:        bool              = True,
        noise_floor:     float        = 0.01,
        min_overlap:     float        = 0.5,
        suggest_from                  = None,
        suggest_k:       int          = 10,
    ) -> tuple["SpectraLibrary", FitResult]:
        """
        Build a library from scratch by iteratively selecting compounds from
        a database that best explain the observed spectrum.

        Runs OMP directly on the full database: at each step the compound
        most correlated with the current residual is selected from **all**
        database spectra, not a pre-filtered subset.

        Parameters
        ----------
        spectrum       : the observed :class:`MassSpectrum` to decompose.
        database       : a database with spectra to search (e.g.
                         :class:`~rgakit.databases.NistDatabase`).
        min_intensity  : drop observed m/z channels below this fraction of
                         base peak before fitting (default 1e-4).
        min_improvement: OMP stopping threshold (default 0.5%).
        n_trials       : stochastic OMP trials (default 1 = greedy).
        temperature    : softmax temperature for stochastic trials.
        max_compounds  : maximum compounds to select (default 30).
        weighted       : apply channel weighting (default True).
        noise_floor    : noise floor for weighting (default 0.01).
        min_overlap    : minimum spectral overlap with observed spectrum
                         to include a database entry (default 0.5).

        Returns
        -------
        (SpectraLibrary, FitResult)
        """
        from scipy.optimize import nnls as _nnls

        # --- observed spectrum grid ----------------------------------------
        mz_obs = spectrum.mz
        y_obs  = spectrum.intensity.astype(float)
        _norm  = float(y_obs.max()) or 1.0
        obs_total_full = float(np.sum(y_obs)) / _norm

        if min_intensity is not None:
            peak = y_obs.max() or 1.0
            keep = y_obs >= min_intensity * peak
            mz_obs = mz_obs[keep]
            y_obs  = y_obs[keep]

        grid = mz_obs
        y    = MassSpectrum(mz_obs, y_obs).on_grid(grid)
        obs_mz_set = set(int(m) for m in grid)

        # --- load all database spectra onto the grid -----------------------
        from .databases.utils import iter_raw, parse_elements

        if max_mw is None:
            max_mw = float(grid.max()) + 1.0
        if allowed_elements is not None:
            allowed_elements = {e.capitalize() for e in allowed_elements}
            logger.info("from_fit: allowed elements = %s", allowed_elements)
        logger.info(
            "from_fit: loading database spectra onto %d-channel grid "
            "(max MW = %.0f) …", len(grid), max_mw,
        )

        rows = iter_raw(database, source=getattr(database, "_source", None))

        n_db      = len(rows)
        A_full    = np.zeros((len(grid), n_db), dtype=float)
        rowid_map = {}   # col_index -> rowid
        grid_map  = {int(m): i for i, m in enumerate(grid)}
        obs_signal_mz = set(int(m) for m, v in zip(grid, y) if v > 0.01)

        kept = 0
        for rowid, mzs_blob, int_blob, exact_mass, formula in rows:
            if exact_mass is not None and exact_mass > max_mw:
                continue
            if allowed_elements is not None and formula:
                if not parse_elements(formula).issubset(allowed_elements):
                    continue

            mz    = np.frombuffer(mzs_blob, dtype=np.float64)
            inten = np.frombuffer(int_blob,  dtype=np.float64)
            if inten.size == 0:
                continue
            imax = float(inten.max()) or 1.0

            total_inten = float(inten.sum())
            inside = sum(v for m, v in zip(mz, inten) if int(m) in obs_mz_set)
            if inside / total_inten < min_overlap:
                continue

            top_idx = np.argsort(-inten)[:2]
            top_mz  = [int(mz[j]) for j in top_idx]
            if not any(m in obs_signal_mz for m in top_mz):
                continue

            for m, v in zip(mz, inten):
                idx = grid_map.get(int(m))
                if idx is not None:
                    A_full[idx, kept] = v / imax
            rowid_map[kept] = rowid
            kept += 1

        A_full = A_full[:, :kept]
        logger.info(
            "from_fit: %d/%d database spectra pass filters (%.0f%%).",
            kept, n_db, 100 * kept / n_db if n_db else 0,
        )

        # --- channel weighting ---------------------------------------------
        if weighted:
            sw = _channel_weights(y, grid, noise_floor)
        else:
            sw = np.ones(len(grid))

        # --- OMP on full database matrix -----------------------------------
        from .solvers.omp import make_omp
        solve = make_omp(
            n_compounds=max_compounds,
            min_improvement=min_improvement,
            n_trials=n_trials,
            temperature=temperature,
        )
        w_full, _ = solve(A_full * sw[:, None], y * sw)

        active_cols = [j for j in range(kept) if w_full[j] > 0]
        if not active_cols:
            logger.warning("from_fit: no compounds selected.")
            lib = cls([])
            result = FitResult(
                weights={}, residual=float(np.linalg.norm(y)),
                fitted=np.zeros_like(y), observed=y, grid=grid,
                obs_total_full=obs_total_full,
                residual_spectrum=y,
            )
            return lib, result

        # --- fetch full MassSpectrum objects for selected compounds ---------
        from .databases.utils import fetch_by_rowids

        sel_rowids   = [rowid_map[j] for j in active_cols]
        spectra_list = fetch_by_rowids(database, sel_rowids)

        lib = cls(spectra_list)

        # --- build FitResult on the library's own grid ---------------------
        names      = lib.names()
        A_lib      = np.column_stack([lib[n].on_grid(grid) for n in names])
        w_lib, _   = _nnls(A_lib * sw[:, None], y * sw)
        residual   = float(np.linalg.norm(A_lib @ w_lib - y))
        fitted     = A_lib @ w_lib

        spectral_contributions = {
            names[j]: A_lib[:, j] * w_lib[j] for j in range(len(names))
        }

        active_idx = [j for j, w in enumerate(w_lib) if w > 0]
        cond = float(np.linalg.cond(A_lib[:, active_idx])) if len(active_idx) >= 2 else 1.0

        result = FitResult(
            weights                = dict(zip(names, w_lib)),
            residual               = residual,
            fitted                 = fitted,
            observed               = y,
            grid                   = grid,
            spectral_contributions = spectral_contributions,
            obs_total_full         = obs_total_full,
            residual_spectrum      = y - fitted,
            condition_number       = cond,
        )

        logger.info(
            "from_fit: selected %d compounds, residual=%.4f.",
            len(active_idx), residual,
        )

        if suggest_from is not None:
            result.suggestions = lib.suggest_compounds(
                result, suggest_from, k=suggest_k,
            )

        return lib, result

    def names(self) -> list[str]:
        return list(self._spectra.keys())

    def add(self, spectrum: MassSpectrum, overwrite: bool = False) -> None:
        """
        Add a spectrum to the library.

        Duplicates are detected by InChIKey.

        Parameters
        ----------
        spectrum  : MassSpectrum to add
        overwrite : if False (default) warn and skip when a duplicate exists
        """
        new_ik = (getattr(spectrum, "metadata", {}) or {}).get("inchikey")
        if not new_ik:
            logger.warning("Spectrum %r has no InChIKey — cannot check for duplicates.", spectrum.name)

        if new_ik:
            for existing_name, existing_spec in self._spectra.items():
                existing_ik = (getattr(existing_spec, "metadata", {}) or {}).get("inchikey")
                if existing_ik == new_ik:
                    if not overwrite:
                        logger.warning(
                            "Spectrum %r matches %r already in library (same InChIKey) "
                            "— skipping. Pass overwrite=True to force.",
                            spectrum.name, existing_name,
                        )
                        return
                    self._spectra.pop(existing_name)
                    break

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
    # Compound suggestion from residual
    # ------------------------------------------------------------------

    def suggest_compounds(
        self,
        result:      "FitResult",
        database,
        k:           int   = 20,
        threshold:   float = 0.01,
        min_overlap: float = 0.5,
    ) -> list[tuple]:
        """
        Suggest compounds that could explain the unexplained peaks in a fit.

        Extracts positive residual peaks (observed − fitted) above *threshold*
        (fraction of base peak), wraps them as a :class:`MassSpectrum`, and
        searches *database* for similar spectra.

        Parameters
        ----------
        result    : :class:`FitResult` from :meth:`fit`.
        database  : Database object.  If it has a ``search_by_spectrum`` method
                    (e.g. :class:`~rgakit.databases.InSilicoDatabase` with an
                    HNSW index), that is used for fast vector search.  Otherwise,
                    falls back to brute-force cosine similarity via ``get``/
                    ``search`` (only practical for small databases).
        k           : number of candidates to return (default 20).
        threshold   : minimum residual intensity (fraction of observed base peak)
                      to include a peak in the query spectrum (default 0.01).
        min_overlap : minimum spectral overlap with the observed spectrum
                      (default 0.5).  Defined as the fraction of a candidate's
                      total intensity that falls on m/z channels present in the
                      observed spectrum.  Compounds whose main peaks are at
                      m/z values never measured are rejected.

        Returns
        -------
        list of ``(MassSpectrum, float)`` tuples sorted by ascending distance
        (or descending similarity).  Compounds already in this library are
        excluded.
        """
        if result.residual_spectrum is None:
            raise ValueError(
                "FitResult has no residual_spectrum — refit with a current version."
            )

        residual = result.residual_spectrum
        grid     = result.grid
        thr      = threshold * float(result.observed.max())

        # Keep only positive residual peaks above the threshold
        mask     = residual > thr
        if not mask.any():
            logger.info("suggest_compounds: no residual peaks above threshold.")
            return []

        res_mz    = grid[mask]
        res_inten = residual[mask]

        query = MassSpectrum(res_mz, res_inten)
        logger.info(
            "suggest_compounds: querying with %d residual peaks (max m/z %d).",
            len(res_mz), int(res_mz.max()),
        )

        # Build a set of InChIKeys (+ names as fallback) for compounds
        # already in the library so we can skip duplicates.
        existing_names = set(self.names())
        existing_keys: set[str] = set()
        for n in existing_names:
            meta = getattr(self._spectra[n], "metadata", {}) or {}
            ik   = meta.get("inchikey")
            if ik:
                existing_keys.add(ik)
            else:
                existing_keys.add(n)

        # Set of observed m/z channels — for overlap filtering.
        obs_mz = set(int(m) for m in result.grid)

        if hasattr(database, "search_by_spectrum"):
            raw = database.search_by_spectrum(query, k=k + len(existing_keys))
        else:
            raise TypeError(
                f"Database {type(database).__name__} does not support "
                f"search_by_spectrum().  Use InSilicoDatabase with an HNSW index."
            )

        hits = []
        seen: set[str] = set()
        for spec, dist in raw:
            meta = getattr(spec, "metadata", {}) or {}
            ik   = meta.get("inchikey") or ""
            if ik in existing_keys:
                continue
            if ik and ik in seen:
                continue

            # Overlap filter: reject compounds whose main peaks fall on
            # m/z channels the instrument never measured.
            s_mz    = spec.mz.astype(int)
            s_inten = spec.intensity.astype(float)
            total   = float(s_inten.sum()) or 1.0
            inside  = float(sum(v for m, v in zip(s_mz, s_inten) if int(m) in obs_mz))
            if inside / total < min_overlap:
                continue

            if ik:
                seen.add(ik)
            hits.append((spec, dist))
            if len(hits) >= k:
                break

        logger.info("suggest_compounds: %d candidates returned.", len(hits))
        return hits

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    def check_collinearity(
        self,
        grid:      np.ndarray | None = None,
        threshold: float             = 0.95,
    ) -> list[tuple[str, str, float]]:
        """
        Check pairwise cosine similarity between all reference spectra.

        Pairs with similarity above *threshold* are hard for the solver to
        disentangle — the weight may be split arbitrarily between them.

        Parameters
        ----------
        grid      : m/z grid to project onto.  Defaults to the library union grid.
        threshold : cosine similarity above which a warning is issued (default 0.95).

        Returns
        -------
        List of ``(name_a, name_b, cosine)`` tuples for pairs above *threshold*,
        sorted by descending similarity.
        """
        if grid is None:
            grid = self.grid
        names = self.names()
        A     = np.column_stack([self._spectra[n].on_grid(grid) for n in names])
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        A_n   = A / norms           # column-normalised

        flagged = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                cos = float(A_n[:, i] @ A_n[:, j])
                if cos >= threshold:
                    flagged.append((names[i], names[j], cos))
        flagged.sort(key=lambda x: -x[2])
        return flagged

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
        method:          str               = "nnls",
        alpha:           float             = 0.1,
        l1_ratio:        float             = 0.7,
        n_compounds:     int | None        = None,
        min_improvement: float             = 0.005,
        prune_tolerance: float             = 0.01,
        n_trials:        int               = 1,
        temperature:     float             = 0.3,
        weighted:        bool              = True,
        noise_floor:     float             = 0.01,
        suggest_from                       = None,
        suggest_k:       int               = 10,
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
        grid           : m/z values to use for fitting. Defaults to all observed
                         m/z values. Library compounds produce zero contribution
                         at m/z channels not in their spectrum, so uncovered
                         channels appear directly in the residual. A warning is
                         issued when fewer than 10 channels overlap with the
                         library.
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
        n_trials       : Number of independent OMP trajectories (default 1).
                         The first is always greedy; additional trials use
                         stochastic selection and the best residual wins.
                         Ignored for non-OMP methods.
        temperature    : Exploration strength for stochastic OMP (default 0.3).
                         0 = deterministic, higher = more random.  Ignored
                         when ``n_trials=1``.
        weighted       : If True (default), apply channel weighting:
                         ``sw_i = y_i^(−0.25) × m/z_i^(0.1)``.  The inverse
                         intensity term (mild Poisson correction) gives small
                         diagnostic peaks more weight without de-prioritising
                         the base peak.  The mass term gently favours
                         higher-m/z channels, which tend to be more specific
                         to individual compounds.
        noise_floor    : Minimum observed intensity as a fraction of the base
                         peak (default 0.01).  Prevents the inverse-intensity
                         term from blowing up at zero-intensity channels.
                         Only used when ``weighted=True``.
        suggest_from   : Optional database object with a ``search_by_spectrum``
                         method (e.g. :class:`~rgakit.databases.NistDatabase`
                         or :class:`~rgakit.databases.InSilicoDatabase`).
                         After fitting, the residual spectrum is searched
                         against this database and the top candidates are
                         stored in :attr:`FitResult.suggestions`.
        suggest_k      : Number of suggestions to return (default 10).

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

        # Full-spectrum total (normalized by its own max) computed before any
        # grid or intensity filtering — so percentages can be expressed as a
        # fraction of everything the instrument measured, not just the library grid.
        _norm = float(y_obs.max()) or 1.0
        obs_total_full = float(np.sum(y_obs)) / _norm

        if min_intensity is not None:
            peak = y_obs.max() or 1.0
            keep = y_obs >= min_intensity * peak
            mz_obs = mz_obs[keep]
            y_obs  = y_obs[keep]

        if grid is None:
            grid = mz_obs

        if mz_min is not None:
            grid = grid[grid >= mz_min]
        if mz_max is not None:
            grid = grid[grid <= mz_max]
        if exclude_mz is not None:
            grid = grid[~np.isin(grid, exclude_mz)]

        if len(grid) == 0:
            raise ValueError("Observed spectrum has no m/z channels after filtering.")

        y     = MassSpectrum(mz_obs, y_obs).on_grid(grid)
        names = self.names()
        A     = np.column_stack([self._spectra[n].on_grid(grid) for n in names])

        # Channels where no library compound has any signal — purely residual.
        n_covered = int(A.any(axis=1).sum())
        if n_covered == 0:
            raise ValueError(
                "No library compound has peaks at any of the observed m/z channels. "
                "Check that the library is appropriate for this spectrum."
            )
        if n_covered < 10:
            logger.warning(
                "%d/%d observed m/z channel(s) are covered by the library — "
                "fit quality may be poor.",
                n_covered, len(grid),
            )

        logger.debug(
            "Fitting: %d compounds × %d observed channels (%d library-covered, method=%s)",
            len(self._spectra), len(grid), n_covered, method,
        )

        solve = make_solver(method, alpha=alpha, l1_ratio=l1_ratio,
                            n_compounds=n_compounds,
                            min_improvement=min_improvement,
                            prune_tolerance=prune_tolerance,
                            n_trials=n_trials, temperature=temperature)

        if weighted:
            sw = _channel_weights(y, grid, noise_floor)
            weights_arr, _ = solve(A * sw[:, None], y * sw)
            residual = float(np.linalg.norm(A @ weights_arr - y))
        else:
            weights_arr, residual = solve(A, y)

        n_det = sum(1 for w in weights_arr if w > 0)
        logger.info(
            "Fit complete (%s): %d/%d compounds detected, residual=%.4f",
            method, n_det, len(names), residual,
        )

        fitted = A @ weights_arr

        fit_params = {k: v for k, v in {
            "mz_min":          mz_min,
            "mz_max":          mz_max,
            "exclude_mz":      exclude_mz,
            "min_intensity":   min_intensity,
            "method":          method if method != "nnls" else None,
            "alpha":           alpha  if method in ("lasso", "elastic_net") else None,
            "l1_ratio":        l1_ratio if method == "elastic_net" else None,
            "n_trials":        n_trials if n_trials > 1 else None,
            "prune_tolerance": prune_tolerance if method == "romp" else None,
            "weighted":        weighted or None,
        }.items() if v is not None}

        spectral_contributions = {
            names[j]: A[:, j] * weights_arr[j]
            for j in range(len(names))
        }

        # Collinearity check on active (nonzero-weight) compounds only, so that
        # compounds clearly absent from the sample don't generate noise warnings.
        active = [j for j, w in enumerate(weights_arr) if w > 0]
        if len(active) >= 2:
            A_act   = A[:, active]
            norms   = np.linalg.norm(A_act, axis=0)
            norms[norms == 0] = 1.0
            A_n     = A_act / norms
            cos_mat = A_n.T @ A_n
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    cos = float(cos_mat[ii, jj])
                    if cos >= 0.97:
                        logger.debug(
                            "Near-collinear active compounds: %r and %r  "
                            "cosine = %.3f — weights may be unreliable.",
                            names[active[ii]], names[active[jj]], cos,
                        )

        cond = float(np.linalg.cond(A[:, active])) if len(active) >= 2 else 1.0
        if cond > 1000:
            logger.debug(
                "Design matrix condition number is %.0f — reference spectra are "
                "nearly collinear; solver may not reliably distinguish them.",
                cond,
            )

        result = FitResult(
            weights                = dict(zip(names, weights_arr)),
            residual               = residual,
            fitted                 = fitted,
            observed               = y,
            grid                   = grid,
            fit_params             = fit_params or None,
            spectral_contributions = spectral_contributions,
            obs_total_full         = obs_total_full,
            residual_spectrum      = y - fitted,
            condition_number       = cond,
        )

        if suggest_from is not None:
            result.suggestions = self.suggest_compounds(
                result, suggest_from, k=suggest_k,
            )

        return result

    def fit_time_series(
        self,
        stack,
        result:          "FitResult | None"            = None,
        mz_min:          int | None                    = None,
        mz_max:          int | None                    = None,
        exclude_mz:      list[int] | None              = None,
        method:          str                           = "nnls",
        alpha:           float                         = 0.1,
        l1_ratio:        float                         = 0.7,
        n_compounds:     int | None                    = None,
        min_improvement: float                         = 0.005,
        prune_tolerance: float                         = 0.01,
        n_trials:        int                           = 1,
        temperature:     float                         = 0.3,
        weighted:        bool                          = True,
        noise_floor:     float                         = 0.01,
        lambda_temporal: float                         = 0.0,
    ) -> TimeFitResult:
        """
        Fit every scan in the full time-series and return per-scan contributions.

        The design matrix is built once and reused for each scan.  When
        *result* is provided the library is reduced to the compounds detected
        in the averaged-spectrum fit, which improves conditioning and prevents
        spurious detections in individual scans.

        Temporal regularisation (``lambda_temporal > 0``) penalises large
        changes between adjacent scans by augmenting each scan's NNLS system
        with an identity block scaled by ``sqrt(lambda_temporal)``.  This
        encodes the prior that gas-phase composition changes smoothly in time.
        The penalty is in the weight domain and scaled by the number of m/z
        channels so that ``lambda_temporal = 1`` gives approximately equal
        weight to data fidelity and temporal continuity.

        Parameters
        ----------
        stack            : :class:`~rgakit.SpectrumStack`.  The full stack is
                           always fit; no time windowing is applied here.
        result           : Optional :class:`FitResult` from :meth:`fit` on the
                           averaged spectrum.  When provided, only the compounds
                           with non-zero weight in *result* are used, reducing
                           the library to the identified species.
        mz_min           : exclude m/z channels below this value
        mz_max           : exclude m/z channels above this value
        exclude_mz       : list of specific m/z values to drop
        method           : ``"nnls"`` (default), ``"lasso"``, or ``"omp"``.
        alpha            : L1 regularisation strength for ``"lasso"``.
        n_compounds      : maximum compounds for ``"omp"``.
        min_improvement  : stopping threshold for ``"omp"``.
        weighted         : intensity-weighted least squares (default True).
        noise_floor      : minimum channel weight fraction for weighted fit.
        lambda_temporal  : Tikhonov temporal regularisation strength (default
                           0 = independent scans, i.e. current behaviour).
                           Values in 0.1–10 enforce increasing smoothness.

        Returns
        -------
        TimeFitResult
        """
        import math

        if not isinstance(stack, SpectrumStack):
            stack = SpectrumStack.from_rga(stack)

        times    = stack.time
        pressure = stack.pressure.astype(float)

        # --- m/z grid ---------------------------------------------------
        # When a FitResult is provided, reuse its grid for consistency
        # (the averaged fit may have applied min_intensity filtering that
        # produced a different grid than intersect(stack, library)).
        if result is not None and hasattr(result, "grid"):
            grid = np.intersect1d(stack.mz, result.grid)
        else:
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

        # --- compound set -----------------------------------------------
        if result is not None:
            names = [n for n, w in result.weights.items() if w > 0]
            if not names:
                raise ValueError("The provided FitResult has no non-zero weights.")
            logger.info(
                "Library reduced to %d compounds from averaged-spectrum fit "
                "(%d in full library).", len(names), len(self._spectra),
            )
        else:
            names = self.names()

        A = np.column_stack([self._spectra[n].on_grid(grid) for n in names])
        n_names = len(names)

        # --- project all scans onto grid --------------------------------
        mz_to_idx = {mz: i for i, mz in enumerate(stack.mz)}
        grid_idx  = np.array([mz_to_idx[mz] for mz in grid])
        Y         = pressure[:, grid_idx]          # (n_times, n_grid)
        norms     = Y.max(axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Y        /= norms

        n_times   = len(times)
        logger.info(
            "Time-series fit: %d scans × %d compounds × %d m/z channels "
            "(t=%.1f–%.1f s, method=%s, lambda_temporal=%.3g)",
            n_times, n_names, len(grid),
            times[0], times[-1], method, lambda_temporal,
        )

        solve = make_solver(method, alpha=alpha, l1_ratio=l1_ratio,
                            n_compounds=n_compounds,
                            min_improvement=min_improvement,
                            prune_tolerance=prune_tolerance,
                            n_trials=n_trials, temperature=temperature)

        # Temporal regularisation: augment each scan's system with a penalty
        # that pulls weights toward the previous scan's values.
        # sqrt(λ) is used directly — no extra scaling by channel count or
        # channel weights.  λ=1 gives roughly equal weight to data fidelity
        # and temporal smoothness; typical values are 0.01–1.0.
        lam_sq = math.sqrt(lambda_temporal) if lambda_temporal > 0 else 0.0

        W         = np.zeros((n_times, n_names))
        residuals = np.zeros(n_times)
        log_every = max(1, n_times // 10)

        for i in range(n_times):
            y_i    = Y[i]
            w_prev = W[i - 1] if i > 0 else np.zeros(n_names)

            if weighted:
                sw = _channel_weights(y_i, grid, noise_floor)
                A_fit   = A * sw[:, None]
                y_fit   = y_i * sw
            else:
                A_fit  = A
                y_fit  = y_i

            if lam_sq > 0:
                A_fit = np.vstack([A_fit, lam_sq * np.eye(n_names)])
                y_fit = np.concatenate([y_fit, lam_sq * w_prev])

            W[i], _      = solve(A_fit, y_fit)
            residuals[i] = float(np.linalg.norm(A @ W[i] - y_i))

            if (i + 1) % log_every == 0 or i == n_times - 1:
                logger.debug("Time-series fit: %d/%d scans done", i + 1, n_times)

        logger.info(
            "Time-series fit complete: mean residual=%.4f", residuals.mean()
        )

        # Build weight dict with all library names (zeros for excluded compounds)
        all_names    = self.names()
        name_to_col  = {n: j for j, n in enumerate(names)}
        weights_dict = {}
        for n in all_names:
            j = name_to_col.get(n)
            weights_dict[n] = W[:, j] if j is not None else np.zeros(n_times)

        return TimeFitResult(times, weights_dict, residuals, grid, all_names)

    def fit_bootstrap(
        self,
        spectrum,
        n_bootstrap:     int   = 200,
        noise_level:     float = 0.05,
        rng:             "np.random.Generator | int | None" = None,
        **fit_kwargs,
    ) -> FitResult:
        """
        Fit with bootstrap uncertainty estimation.

        Runs :meth:`fit` once for the point estimate, then *n_bootstrap* times
        on copies of the spectrum with multiplicative Gaussian noise added.
        The standard deviation of each compound's weight across all bootstrap
        runs is stored in :attr:`~FitResult.uncertainties`.

        Compounds whose ``uncertainty > weight`` should be treated as
        unreliably detected.

        Parameters
        ----------
        spectrum     : MassSpectrum to fit
        n_bootstrap  : number of bootstrap iterations (default 200)
        noise_level  : relative noise amplitude added to each channel,
                       i.e. ``y_noisy = y * (1 + noise_level × N(0,1))``,
                       clipped at 0 (default 0.05 = 5 %).
        rng          : numpy Generator, integer seed, or None for a fresh
                       default Generator.
        **fit_kwargs : forwarded verbatim to :meth:`fit` for all runs.

        Returns
        -------
        FitResult  (same as :meth:`fit`, with ``uncertainties`` populated)
        """
        rng_obj = np.random.default_rng(rng)

        # Point estimate
        result = self.fit(spectrum, **fit_kwargs)
        names  = list(result.weights.keys())

        mz_obs   = spectrum.mz
        y_base   = spectrum.intensity.astype(float)
        # Freeze normalization factor to the point estimate's max so that
        # bootstrap noise doesn't shift the entire spectrum via on_grid()
        # re-normalization.  Each noisy replicate is expressed in the same
        # scale as the original spectrum.
        base_max = float(y_base.max()) or 1.0

        boot_weights = {name: np.empty(n_bootstrap) for name in names}

        for k in range(n_bootstrap):
            noise  = rng_obj.standard_normal(len(y_base))
            y_boot = np.maximum(y_base * (1.0 + noise_level * noise), 0.0)
            # Re-scale so the max matches the original spectrum's max.
            # This prevents on_grid() normalization from amplifying or
            # dampening the entire spectrum when the base peak fluctuates.
            boot_max = float(y_boot.max()) or 1.0
            y_boot   = y_boot * (base_max / boot_max)
            spec_k = MassSpectrum(mz_obs, y_boot)
            r_k    = self.fit(spec_k, **fit_kwargs)
            for name in names:
                boot_weights[name][k] = r_k.weights.get(name, 0.0)

        result.uncertainties = {
            name: float(np.std(boot_weights[name])) for name in names
        }
        logger.info("Bootstrap complete (%d runs, noise_level=%.3f)", n_bootstrap, noise_level)
        return result
