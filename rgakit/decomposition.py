"""
decomposition.py
----------------
Non-negative Matrix Factorisation (NMF) for blind source separation of
time-resolved RGA mass spectra.

Given a SpectrumStack with shape (n_times, n_mz), NMF factors it into:

    P ≈ W × H

where W is (n_times, k) — concentration/time profiles, and H is (k, n_mz) —
pure component spectra.  Both W and H are non-negative.

This discovers latent chemical components without any reference library.
The resolved pure spectra can then be matched against a database (NIST, etc.)
for chemical identification.
"""

from __future__ import annotations

import logging

import numpy as np

from .spectrum import MassSpectrum

logger = logging.getLogger(__name__)


class DecompositionResult:
    """Container for NMF decomposition output."""

    def __init__(
        self,
        components:  list[MassSpectrum],
        profiles:    np.ndarray,
        time:        np.ndarray,
        mz:          np.ndarray,
        explained:   np.ndarray,
        reconstruction_error: float,
        matches:     list[list[tuple]] | None = None,
    ):
        self.components = components
        self.profiles   = profiles            # (n_times, k)
        self.time       = time                # (n_times,)
        self.mz         = mz                  # (n_mz,)
        self.explained  = explained           # (k,) fraction of total signal
        self.reconstruction_error = reconstruction_error
        self.matches    = matches             # per-component DB matches

    @property
    def n_components(self) -> int:
        return len(self.components)

    def plot(self, ax=None, top_n: int | None = None):
        """Plot time profiles of all components."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        k = top_n or self.n_components
        for i in range(min(k, self.n_components)):
            label = self.components[i].name
            ax.plot(self.time, self.profiles[:, i], label=label, linewidth=1.2)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"NMF components ({self.n_components})")
        return ax

    def plot_spectra(self, top_n: int | None = None):
        """Plot the resolved pure component spectra as a grid."""
        import matplotlib.pyplot as plt

        k = min(top_n or self.n_components, self.n_components)
        fig, axes = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        if k == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            spec = self.components[i]
            ax.bar(spec.mz, spec.normalized, width=0.8, color="steelblue", alpha=0.8)
            ax.set_ylabel("Rel. int.")
            title = spec.name
            if self.matches and self.matches[i]:
                best = self.matches[i][0]
                title += f"  →  {best[0]} (cos={best[1]:.2f})"
            ax.set_title(title, fontsize=9, loc="left")

        axes[-1].set_xlabel("m/z")
        fig.tight_layout()
        return fig

    def to_library(self) -> "SpectraLibrary":
        """Convert resolved components into a SpectraLibrary for fitting."""
        from .library import SpectraLibrary
        return SpectraLibrary(self.components)

    def summary(self, print_output: bool = True) -> str:
        """Print a summary table of the decomposition."""
        lines = []
        lines.append(f"NMF decomposition: {self.n_components} components, "
                      f"reconstruction error = {self.reconstruction_error:.4f}")
        lines.append("")
        lines.append(f"  {'#':>3s}  {'Signal %':>8s}  {'Component':30s}  {'Best match':30s}")
        lines.append("  " + "─" * 76)

        for i in range(self.n_components):
            pct  = 100 * self.explained[i]
            name = self.components[i].name
            if self.matches and self.matches[i]:
                best_name, best_cos = self.matches[i][0][:2]
                match_str = f"{best_name} (cos={best_cos:.2f})"
            else:
                match_str = ""
            lines.append(f"  {i+1:3d}  {pct:7.1f}%  {name:30s}  {match_str:30s}")

        output = "\n".join(lines)
        if print_output:
            print(output)
            return None
        return output

    def save(self, path: str) -> None:
        """Serialize to disk with dill."""
        import dill
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DecompositionResult":
        """Load a previously saved DecompositionResult."""
        import dill
        with open(path, "rb") as f:
            return dill.load(f)

    def __repr__(self) -> str:
        return (
            f"DecompositionResult({self.n_components} components, "
            f"error={self.reconstruction_error:.4f})"
        )


def decompose(
    stack,
    n_components:  int          = 8,
    time_range:    tuple | None = None,
    mz_min:        int | None   = None,
    mz_max:        int | None   = None,
    exclude_mz:    list[int] | None = None,
    database                    = None,
    match_top_n:   int          = 3,
    random_state:  int | None   = 0,
) -> DecompositionResult:
    """
    Decompose a SpectrumStack into pure components via NMF.

    Parameters
    ----------
    stack          : :class:`~rgakit.SpectrumStack` to decompose.
    n_components   : number of components to extract (default 8).
    time_range     : optional ``(t_start, t_end)`` to restrict the time axis.
    mz_min         : exclude m/z channels below this value.
    mz_max         : exclude m/z channels above this value.
    exclude_mz     : list of specific m/z values to drop.
    database       : optional database with ``search_by_spectrum`` to match
                     each resolved component against (e.g. NistDatabase).
    match_top_n    : number of database matches per component (default 3).
    random_state   : seed for reproducibility (default 0).

    Returns
    -------
    DecompositionResult
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError(
            "scikit-learn is required for NMF decomposition: "
            "pip install scikit-learn"
        )

    from .stack import SpectrumStack

    if not isinstance(stack, SpectrumStack):
        stack = SpectrumStack.from_rga(stack)

    times    = stack.time
    pressure = stack.pressure.astype(float)
    mz       = stack.mz

    # Time filtering
    if time_range is not None:
        t_mask   = (times >= time_range[0]) & (times <= time_range[1])
        times    = times[t_mask]
        pressure = pressure[t_mask]

    # m/z filtering
    mz_mask = np.ones(len(mz), dtype=bool)
    if mz_min is not None:
        mz_mask &= mz >= mz_min
    if mz_max is not None:
        mz_mask &= mz <= mz_max
    if exclude_mz is not None:
        mz_mask &= ~np.isin(mz, exclude_mz)

    mz       = mz[mz_mask]
    pressure = pressure[:, mz_mask]

    # Drop zero-signal channels
    col_sums = pressure.sum(axis=0)
    nonzero  = col_sums > 0
    mz       = mz[nonzero]
    pressure = pressure[:, nonzero]

    # NMF requires non-negative input; background-corrected data can go negative
    pressure = np.maximum(pressure, 0.0)

    if pressure.size == 0 or pressure.sum() == 0:
        raise ValueError("No signal in the selected time/m/z range.")

    logger.info(
        "NMF decomposition: %d scans × %d m/z channels → %d components",
        pressure.shape[0], pressure.shape[1], n_components,
    )

    # Run NMF
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=2000,
        random_state=random_state,
    )
    W = model.fit_transform(pressure)   # (n_times, k)
    H = model.components_               # (k, n_mz)

    # Relative reconstruction error: ||P - WH||_F / ||P||_F
    p_norm = float(np.linalg.norm(pressure, "fro")) or 1.0
    recon_error = float(model.reconstruction_err_) / p_norm

    # Compute each component's fraction of total signal
    total_signal = float(pressure.sum())
    component_signals = np.array([
        float((W[:, i:i+1] @ H[i:i+1, :]).sum())
        for i in range(n_components)
    ])
    explained = component_signals / total_signal if total_signal > 0 else component_signals

    # Sort components by signal fraction (descending)
    order = np.argsort(-explained)
    W         = W[:, order]
    H         = H[order, :]
    explained = explained[order]

    # Wrap each component as a MassSpectrum
    components = []
    for i in range(n_components):
        spec_h = H[i]
        components.append(MassSpectrum(
            mz=mz,
            intensity=spec_h,
            name=f"NMF-{i+1}",
        ))

    # Optional: match against database
    matches = None
    if database is not None:
        logger.info("Matching %d NMF components against database …", n_components)
        matches = []
        for i, spec in enumerate(components):
            if hasattr(database, "search_by_spectrum"):
                hits = database.search_by_spectrum(spec, k=match_top_n)
                component_matches = [
                    (getattr(s, "name", "?"), 1.0 - d, getattr(s, "metadata", {}))
                    for s, d in hits
                ]
            else:
                component_matches = []
            matches.append(component_matches)

            if component_matches:
                best = component_matches[0]
                logger.info(
                    "  NMF-%d (%.1f%% signal) → %s (cos=%.2f)",
                    i + 1, 100 * explained[i], best[0], best[1],
                )

    return DecompositionResult(
        components=components,
        profiles=W,
        time=times,
        mz=mz,
        explained=explained,
        reconstruction_error=recon_error,
        matches=matches,
    )
