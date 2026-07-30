"""
result.py
---------
FitResult and TimeFitResult: output containers for SpectraLibrary.fit().
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FitResult:
    """Container for the output of SpectraLibrary.fit()."""

    def __init__(
        self,
        weights:                dict[str, float],
        residual:               float,
        fitted:                 np.ndarray,
        observed:               np.ndarray,
        grid:                   np.ndarray,
        metadata:               dict | None                    = None,
        fit_params:             dict | None                    = None,
        spectral_contributions: dict[str, np.ndarray] | None  = None,
        obs_total_full:         float | None                   = None,
        residual_spectrum:      np.ndarray | None              = None,
        condition_number:       float | None                   = None,
        uncertainties:          dict[str, float] | None        = None,
    ):
        self.weights                = weights
        self.residual               = residual
        self.fitted                 = fitted
        self.observed               = observed
        self.grid                   = grid
        self.metadata               = metadata   or {}
        self.fit_params             = fit_params or {}
        self.spectral_contributions = spectral_contributions  # {name: A[:,j]*w_j}
        # Total normalized intensity of the *full* observed spectrum (all m/z,
        # before grid intersection).  Used so that percentages are expressed as
        # a fraction of everything the instrument measured, not just what the
        # library covers.  None for objects loaded from old pickle files.
        self.obs_total_full         = obs_total_full
        # Per-channel residual vector (observed - fitted) on the fitting grid.
        # Spikes here indicate peaks unexplained by the library (missing compounds).
        self.residual_spectrum      = residual_spectrum
        # Condition number of the design matrix A.  Values above ~1000 mean
        # near-collinear reference spectra that the solver cannot distinguish.
        self.condition_number       = condition_number
        # Per-compound weight uncertainty (std) from bootstrap runs.
        # None unless lib.fit_bootstrap() was used.
        self.uncertainties          = uncertainties
        # Suggested compounds from external database search on the residual.
        # List of (MassSpectrum, distance) tuples, or None.
        self.suggestions: list[tuple] | None = None

    @property
    def contributions(self) -> dict[str, float]:
        """Non-zero weights only, sorted by descending contribution."""
        return dict(
            sorted(
                ((k, v) for k, v in self.weights.items() if v > 0),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    def plot(self, ax=None):
        """Plot observed vs fitted spectrum."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.bar(self.grid, self.observed, width=0.8, label="Observed", alpha=0.5, color="gray")
        ax.bar(self.grid, self.fitted,   width=0.8, label="Fitted",   alpha=0.6, color="steelblue", linewidth=0)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity")
        ax.legend()
        ax.set_title(f"Fit  |  residual = {self.residual:.4f}")
        return ax

    def plot_residual(self, ax=None, threshold: float = 0.01):
        """Bar chart of the per-m/z residual (observed − fitted).

        Channels above *threshold* (fraction of base peak) are highlighted in
        orange — these are peaks unexplained by the library.
        """
        import matplotlib.pyplot as plt

        if self.residual_spectrum is None:
            raise ValueError("No residual_spectrum stored — refit with a current version.")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        r   = self.residual_spectrum
        thr = threshold * self.observed.max()
        colors = ["#E65100" if abs(v) > thr else "#78909C" for v in r]
        ax.bar(self.grid, r, width=0.8, color=colors)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Residual (observed − fitted)")
        ax.set_title("Fit residual  —  orange = unexplained peaks")
        return ax

    def summary(self, threshold: float = 1e-4, print_output: bool = True) -> str:
        """
        Return (and optionally print) a formatted table of contributions.

        Parameters
        ----------
        threshold    : minimum weight to include a compound
        print_output : if True (default), also print to stdout
        """
        contribs  = {k: v for k, v in self.contributions.items() if v >= threshold}
        # Denominator: full-spectrum normalized total if available (so percentages
        # are wrt everything the instrument measured), else fall back to in-grid sum.
        obs_total = float(self.obs_total_full) if self.obs_total_full else (float(np.sum(self.observed)) or 1.0)

        # Spectral coverage: fraction of observed total explained by each compound
        sc = self.spectral_contributions
        if sc is not None:
            pct_map = {
                name: 100.0 * float(np.sum(sc[name])) / obs_total
                for name in contribs
            }
        else:
            # Fallback for old FitResult objects loaded from disk
            w_sum = sum(contribs.values()) or 1.0
            pct_map = {name: 100.0 * w / w_sum for name, w in contribs.items()}

        # Re-sort by spectral coverage (descending)
        contribs  = dict(sorted(contribs.items(), key=lambda kv: pct_map[kv[0]], reverse=True))

        total_pct = sum(pct_map.values())

        col_name   = max((len(n) for n in contribs), default=8)
        col_name   = max(col_name, 8)
        has_unc    = bool(self.uncertainties)
        col_weight = 17 if has_unc else 8   # "w.4f±u.4f" = 13 chars, padded to 17
        col_flag   = 2 if has_unc else 0    # " !" flag slot

        # Row layout: "  " + name(col_name) + "  " + weight(col_weight) + "   " + pct(6) + flag(col_flag)
        w_data  = 2 + col_name + 2 + col_weight + 3 + 6 + col_flag
        cond_str = f"   cond = {self.condition_number:.0f}" if self.condition_number is not None else ""
        info = f"  Fit result   residual = {self.residual:.4f}   grid = {len(self.grid)} m/z points{cond_str}"
        w_total = max(w_data, len(info))

        div  = "─" * w_total
        hdiv = "═" * w_total

        w_hdr = "Weight ± σ" if has_unc else "Weight"

        lines = []
        lines.append(f"╒{hdiv}╕")
        lines.append(f"│{info:<{w_total}}│")
        lines.append(f"╞{hdiv}╡")
        hdr = f"  {'Compound':<{col_name}}  {w_hdr:>{col_weight}}   {'% obs':>6}{'':>{col_flag}}"
        lines.append(f"│{hdr:<{w_total}}│")
        lines.append(f"├{div}┤")

        for name, w in contribs.items():
            pct = pct_map[name]
            if has_unc and name in self.uncertainties:
                unc     = self.uncertainties[name]
                w_str   = f"{w:.4f}±{unc:.4f}"
                flagged = " !" if unc > w else ""
            else:
                w_str   = f"{w:.4f}"
                flagged = ""
            row = f"  {name:<{col_name}}  {w_str:>{col_weight}}   {pct:>5.1f}%{flagged:<{col_flag}}"
            lines.append(f"│{row:<{w_total}}│")

        lines.append(f"├{div}┤")
        footer = f"  {'Total':<{col_name}}  {'':>{col_weight}}   {total_pct:>5.1f}%"
        lines.append(f"│{footer:<{w_total}}│")
        lines.append(f"╘{hdiv}╛")

        output = "\n".join(lines)
        if print_output:
            print(output)
            return None   # suppress repr display in IPython/Spyder
        return output

    def save(self, path: str) -> None:
        """Serialize this FitResult to disk with dill."""
        import dill
        with open(path, "wb") as f:
            dill.dump(self, f)
        logger.debug("FitResult saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FitResult":
        """Load a FitResult previously saved with :meth:`save`."""
        import dill
        with open(path, "rb") as f:
            obj = dill.load(f)
        logger.debug("FitResult loaded from %s", path)
        return obj

    def __repr__(self) -> str:
        top     = list(self.contributions.items())[:3]
        top_str = ", ".join(f"{n}: {w:.3f}" for n, w in top)
        return f"FitResult(residual={self.residual:.4f}, top=[{top_str}])"


class TimeFitResult:
    """Container for the output of SpectraLibrary.fit_time_series()."""

    def __init__(
        self,
        time:           np.ndarray,
        weights:        dict[str, np.ndarray],
        residuals:      np.ndarray,
        grid:           np.ndarray,
        compound_names: list[str],
    ):
        self.time           = np.asarray(time)
        self.weights        = weights        # {name: (n_times,)}
        self.residuals      = np.asarray(residuals)
        self.grid           = grid
        self.compound_names = compound_names

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def contributions(self, threshold: float = 1e-4) -> dict[str, np.ndarray]:
        """Compounds whose mean weight exceeds *threshold*, sorted descending."""
        return dict(sorted(
            ((k, v) for k, v in self.weights.items() if v.mean() > threshold),
            key=lambda kv: kv[1].mean(),
            reverse=True,
        ))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """
        Export to a pandas DataFrame (time as index, one column per compound
        plus a 'residual' column).  Requires pandas.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")
        df = pd.DataFrame(self.weights, index=self.time)
        df.index.name = "time"
        df["residual"] = self.residuals
        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax=None,
        top_n:      int   = 8,
        normalized: bool  = True,
        threshold:  float = 1e-4,
    ):
        """
        Stacked area plot of compound contributions vs time (matplotlib).

        Parameters
        ----------
        top_n      : maximum number of compounds to show
        normalized : if True, stack normalised to 100%; otherwise raw weights
        threshold  : minimum mean weight to include a compound
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        contribs = self.contributions(threshold)
        top      = list(contribs.items())[:top_n]
        if not top:
            raise ValueError("No contributions above threshold.")

        names = [n for n, _ in top]
        W     = np.array([w for _, w in top])   # (n_top, n_times)

        if normalized:
            total = W.sum(axis=0)
            total = np.where(total > 0, total, 1.0)
            W     = 100.0 * W / total
            ylabel = "Contribution (%)"
        else:
            ylabel = "Weight"

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        colors = cm.tab20.colors
        ax.stackplot(
            self.time, W,
            labels=[n for n in names],
            colors=[colors[i % len(colors)] for i in range(len(names))],
            alpha=0.85,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Time-resolved fit (top {top_n})")
        return ax

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize this TimeFitResult to disk with dill."""
        import dill
        with open(path, "wb") as f:
            dill.dump(self, f)
        logger.debug("TimeFitResult saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TimeFitResult":
        """Load a TimeFitResult previously saved with :meth:`save`."""
        import dill
        with open(path, "rb") as f:
            obj = dill.load(f)
        logger.debug("TimeFitResult loaded from %s", path)
        return obj

    def __repr__(self) -> str:
        top = list(self.contributions().items())[:3]
        top_str = ", ".join(n for n, _ in top)
        return (
            f"TimeFitResult({len(self.time)} scans, "
            f"t=[{self.time[0]:.1f}, {self.time[-1]:.1f}] s, "
            f"top=[{top_str}])"
        )
